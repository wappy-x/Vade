//! Lane nv-wide: the candidate `nv_rows_tile` (256-row form) against the base image's `nv_rows_tile` (128-row form, launched
//! with its own geometry) and the base `expert_group` NvRows path, every output byte, plus the rc.8 chain on the CPU for
//! sampled words, then timing on real local shapes. Run: NV_WIDE=1 kernel-bench --nv-rows <base ptx> <candidate ptx>, where
//! the base image predates lane nv-wide. Safe code only.
#![forbid(unsafe_code)]
use super::*;

fn launch_new(g:&Gpu,f:H,fx:&Fixture,out:u64)->Result<()> {launch_geometry(g,f,fx,out,256,128)}
fn launch_old(g:&Gpu,f:H,fx:&Fixture,out:u64)->Result<()> {launch_geometry(g,f,fx,out,128,64)}
fn launch_geometry(g:&Gpu,f:H,fx:&Fixture,out:u64,rows:usize,threads:u32)->Result<()> {
    let mut a=args(fx.table,fx.input,out,fx.lutp,fx.parts.len(),fx.units,fx.cols/16*17);
    let most=fx.parts.iter().map(|p|p.m).max().unwrap_or(1);
    g.launch(f,[most.div_ceil(8) as u32,fx.rows.div_ceil(rows) as u32,fx.parts.len() as u32],threads,0,&mut [a.as_mut_ptr().cast()])
}
fn floats(bytes:&[u8])->Vec<f32> {bytes.chunks_exact(4).map(|w|f32::from_le_bytes(w.try_into().unwrap())).collect()}
fn round(x:f32)->f32 {let b=x.to_bits();f32::from_bits(b.wrapping_add(0x7fff+(b>>16&1))&0xffff0000)}
/// The rc.8 chain on the CPU: one ascending FP32 sum of ((value * activation) * fold), no fused multiply-add.
fn chain(weights:&[u8],scales:&[u8],input:&[f32],params:(f32,f32))->f32 {
    let global=params.0*params.1;let mut sum=0.0f32;
    for (block,&scale) in scales.iter().enumerate() {
        let fold=(e4m3(scale)*input[block*17+16])*global;
        for e in 0..16 {let byte=weights[block*8+e/2];sum+=(e2m1(if e%2==0 {byte&15} else {byte>>4})*input[block*17+e])*fold;}
    }
    sum
}
struct K {reference:H,old:H,new:H}
fn check(g:&mut Gpu,k:&K,fx:&Fixture,label:&str,cpu_words:&mut usize)->Result<()> {
    let bytes=fx.units*4;
    let outs=[Buf::new(g,&vec![0;bytes])?,Buf::new(g,&vec![0;bytes])?,Buf::new(g,&vec![0;bytes])?];
    for o in &outs {o.poison(g)?;}
    launch_reference(g,k.reference,fx,outs[0].address)?;launch_old(g,k.old,fx,outs[1].address)?;launch_new(g,k.new,fx,outs[2].address)?;g.sync()?;
    let read:Vec<Vec<u8>>=outs.iter().map(|o|o.read(g)).collect::<Result<_>>()?;
    for (index,name) in [(1,"base nv_rows_tile"),(2,"candidate nv_rows_tile")] {
        if read[index]!=read[0] {let at=read[0].iter().zip(&read[index]).position(|(a,b)|a!=b).unwrap();
            return Err(format!("NVWIDE BYTE FAIL {label} {name} rows={} cols={} parts={:?} byte={at}",fx.rows,fx.cols,fx.parts.iter().map(|p|p.m).collect::<Vec<_>>()));}
    }
    if read[0][..16]!=GUARD || read[0][bytes+16..]!=GUARD {return Err("NVWIDE guard changed".into());}
    // CPU: first and last part, rows 0, 1 and the last, every token; finite words only (NaN payloads are not specified).
    let words=floats(&read[2][16..bytes+16]);   // candidate
    let total:usize=fx.parts.iter().map(|p|p.m).sum();
    let input=floats(&g.read(fx.input,total*fx.cols/16*17*4)?);
    let (mut first,mut first_token)=(0usize,0usize);
    for (index,part) in fx.parts.iter().enumerate() {
        if index==0 || index+1==fx.parts.len() {
            for row in [0usize,1,fx.rows-1] {
                if row>=fx.rows {continue;}
                let w=g.read(part.weight+(row*fx.cols/2) as u64,fx.cols/2)?;let s=g.read(part.scales+(row*fx.cols/16) as u64,fx.cols/16)?;
                for token in 0..part.m {
                    let x=&input[(first_token+token)*fx.cols/16*17..(first_token+token+1)*fx.cols/16*17];
                    let value=chain(&w,&s,x,fx.params);if value.is_nan() {continue;}
                    let got=words[first+row*part.m+token];
                    if round(value).to_bits()!=got.to_bits() {return Err(format!("NVWIDE CPU FAIL {label} part={index} row={row} token={token} cpu={:08x} gpu={:08x}",round(value).to_bits(),got.to_bits()));}
                    *cpu_words+=1;
                }
            }
        }
        first+=fx.rows*part.m;first_token+=part.m;
    }
    Ok(())
}
fn route(group:usize,skew:f64,seed:u64)->Vec<usize> {
    let mut random=Rng(seed);let mut counts=vec![0usize;288];
    let weights:Vec<f64>=(0..288).map(|i|((i+1) as f64).powf(-skew)).collect();let total:f64=weights.iter().sum();
    for _ in 0..group {
        let mut chosen=Vec::new();
        while chosen.len()<8 {
            let expert=if skew==0.0 {random.next() as usize%288} else {
                let mut u=(random.next()>>11) as f64/(1u64<<53) as f64*total;let mut pick=287;
                for (i,w) in weights.iter().enumerate() {if u<*w {pick=i;break;} u-=w;} pick};
            if !chosen.contains(&expert) {chosen.push(expert);counts[expert]+=1;}
        }
    }
    counts.retain(|&c|c>0);counts
}
pub(super) fn run(base_path:&str,candidate_path:&str)->Result<()> {
    let base=std::fs::read(base_path).map_err(|e|e.to_string())?;let candidate=std::fs::read(candidate_path).map_err(|e|e.to_string())?;
    println!("NVWIDE base_sha256={} candidate_sha256={}",sha256::digest(&base),sha256::digest(&candidate));
    let mut g=Gpu::new()?;let (mb,mc)=(g.module(&base)?,g.module(&candidate)?);
    let k=K {reference:g.function(mb,"expert_group")?,old:g.function(mb,"nv_rows_tile")?,new:g.function(mc,"nv_rows_tile")?};
    let started=Instant::now();let (mut fixtures,mut words,mut cpu_words)=(0usize,0usize,0usize);
    // All 256 weight bytes x all 256 scale bytes, every activation kind (incl. NaN, infinity, subnormal, signed zero).
    for (kind,params,counts) in [(0u32,(0.75f32,0.125f32),vec![3usize]),(1,(1.0,1.0),vec![17]),(2,(0.0009765625,0.5),vec![5,16]),(3,(3.0,0.0078125),vec![1]),(4,(0.75,0.125),vec![2,33]),(5,(0.7391,0.1173),vec![8,9])] {
        let fx=build(&mut g,65536,128,&counts,1,kind,params,11+kind as u64,None)?;check(&mut g,&k,&fx,"all-pairs",&mut cpu_words)?;fixtures+=1;words+=fx.units;g.clear()?;
    }
    println!("NVWIDE all 256x256 code pairs ok fixtures={fixtures} elapsed_s={:.1}",started.elapsed().as_secs_f64());
    // Family sweep (the nv_rows_tile L0 sweep): row and column tails, token counts across tile edges.
    let counts_list:Vec<Vec<usize>>=vec![vec![1],vec![7],vec![8],vec![9],vec![16],vec![17],vec![31,32,33],vec![1,2,3,50],vec![100],vec![64,1,16],vec![15,15,15,15,15,15,15,15],vec![9,4,22,1,35,16,8],vec![1,1,1,2,3,7,5,8,9,4,6,1]];
    let mut seed=500u64;
    for &rows in &[1usize,2,31,32,33,127,128,129,200,255,256,257,384,1024] {
        for &cols in &[128usize,256,384,512,1024,2048,4096,8192] {
            for counts in &counts_list {
                if rows*cols*counts.len()>24_000_000 || rows*cols>4_000_000 && counts.len()>3 {continue;}
                seed+=1;let fx=build(&mut g,rows,cols,counts,0,(seed%6) as u32,if seed.is_multiple_of(2) {(0.75,0.125)} else {(1.0/3.0,7.0)},seed,None)?;
                check(&mut g,&k,&fx,"sweep",&mut cpu_words)?;fixtures+=1;words+=fx.units;g.clear()?;
            }
        }
    }
    println!("NVWIDE BYTES PASS fixtures={fixtures} output_words={words} cpu_words={cpu_words} elapsed_s={:.1}",started.elapsed().as_secs_f64());
    // Timing: real local shapes, 288 experts with their own weights (cold), top-8 routing.
    let groups:Vec<usize>=std::env::var("NV_WIDE_GROUPS").ok().map(|v|v.split(',').filter_map(|x|x.parse().ok()).collect()).unwrap_or(vec![64,256,1024]);
    for (rows,cols,label) in [(1024usize,4096usize,"gate_up"),(2048,2048,"down")] {
        let mut shared=Vec::new();let mut weight_random=Rng(7);
        for _ in 0..288 {
            let w:Vec<u8>=(0..rows*cols/2).map(|_|weight_random.next() as u8).collect();let s:Vec<u8>=(0..rows*cols/16).map(|_|0x30+(weight_random.next()%20) as u8).collect();
            shared.push((Buf::new(&mut g,&w)?.address,Buf::new(&mut g,&s)?.address));
        }
        for &group in &groups {
            for (skew,routing) in [(0.0,"uniform"),(0.5,"skewed")] {
                let counts=route(group,skew,99+group as u64);
                let fx=build(&mut g,rows,cols,&counts,0,5,(0.7391,0.1173),31,Some(&shared))?;check(&mut g,&k,&fx,"timing-fixture",&mut cpu_words)?;
                let out=Buf::new(&mut g,&vec![0;fx.units*4])?;let gp=&g;
                let arms:[&dyn Fn()->Result<()>;2]=[&||launch_old(gp,k.old,&fx,out.address),&||launch_new(gp,k.new,&fx,out.address)];
                let mut samples=vec![Vec::new();2];
                for repeat in 0..9 {for i in 0..2 {let arm=if repeat%2==0 {i} else {1-i};arms[arm]()?;gp.sync()?;
                    let start=Instant::now();for _ in 0..3 {arms[arm]()?;} gp.sync()?;samples[arm].push(start.elapsed().as_secs_f64()*1e6/3.0);}}
                let t:Vec<f64>=samples.into_iter().map(median).collect();
                println!("NVWIDE TIME {label} group={group} routing={routing} parts={} most={} old_us={:.1} new_us={:.1} change_pct={:.1}",
                    counts.len(),counts.iter().max().unwrap(),t[0],t[1],100.0*(t[1]/t[0]-1.0));
            }
        }
        g.clear()?;
    }
    if std::env::var_os("NV_WIDE_COLS").is_some() {columns(&mut g,mb,mc,&groups)?;}
    println!("NVWIDE PASS");Ok(())
}

/// Column tiles (zero-skip layout): the base image's nv_rows_cols (128-row form) against the candidate's (256-row form) and the
/// candidate's row-major nv_rows_tile on the same logical matrices, every output byte, then timing. Part table rows carry
/// block_rows = 128 (`expert_ops::nv_tiled`).
fn columns(g:&mut Gpu,mb:H,mc:H,groups:&[usize])->Result<()> {
    let (old,new,tile)=(g.function(mb,"nv_rows_cols")?,g.function(mc,"nv_rows_cols")?,g.function(mc,"nv_rows_tile")?);
    for (rows,cols,label) in [(1024usize,4096usize,"gate_up"),(2048,2048,"down")] {
        let (mut row_major,mut tiled)=(Vec::new(),Vec::new());let mut weight_random=Rng(7);
        for _ in 0..288 {
            let w:Vec<u8>=(0..rows*cols/2).map(|_|weight_random.next() as u8).collect();let s:Vec<u8>=(0..rows*cols/16).map(|_|0x30+(weight_random.next()%20) as u8).collect();
            let nibble=|r:usize,j:usize|->u8 {(w[r*cols/2+j/2]>>((j%2)*4))&15};
            let (mut tw,mut ts)=(vec![0u8;rows*cols/2],vec![0u8;rows*cols/16]);
            for t in 0..rows/128 {for j in 0..cols {for i in 0..64 {tw[(t*cols+j)*64+i]=nibble(128*t+2*i,j)|(nibble(128*t+2*i+1,j)<<4);}}
                for b in 0..cols/16 {for r in 0..128 {ts[(t*cols/16+b)*128+r]=s[(128*t+r)*cols/16+b];}}}
            row_major.push((Buf::new(g,&w)?.address,Buf::new(g,&s)?.address));tiled.push((Buf::new(g,&tw)?.address,Buf::new(g,&ts)?.address));
        }
        for &group in groups {
            for (skew,routing) in [(0.0,"uniform"),(0.5,"skewed")] {
                let counts=route(group,skew,99+group as u64);
                let fx=build(g,rows,cols,&counts,0,5,(0.7391,0.1173),31,Some(&row_major))?;
                let fc=build(g,rows,cols,&counts,0,5,(0.7391,0.1173),31,Some(&tiled))?;
                // Mark the column-tile table rows: block_rows (third word) = 128.
                let mut table=Vec::new();let (mut first,mut first_row)=(0usize,0usize);
                for part in &fc.parts {let mut e=entry(rows,cols,part,first,first_row,fc.params);e[12..16].copy_from_slice(&128u32.to_le_bytes());table.extend(e);first+=rows*part.m;first_row+=part.m;}
                let tiled_table=Buf::new(g,&table)?;let fc=Fixture {table:tiled_table.address,..fc};
                let bytes=fx.units*4;let outs=[Buf::new(g,&vec![0;bytes])?,Buf::new(g,&vec![0;bytes])?,Buf::new(g,&vec![0;bytes])?];
                for o in &outs {o.poison(g)?;}
                launch_new(g,tile,&fx,outs[0].address)?;launch_old(g,old,&fc,outs[1].address)?;launch_new(g,new,&fc,outs[2].address)?;g.sync()?;
                let read:Vec<Vec<u8>>=outs.iter().map(|o|o.read(g)).collect::<Result<_>>()?;
                if read[1]!=read[0] || read[2]!=read[0] {return Err(format!("NVWIDE COLS BYTE FAIL {label} group={group} {routing} old_equal={} new_equal={}",read[1]==read[0],read[2]==read[0]));}
                let out=Buf::new(g,&vec![0;bytes])?;let gp=&*g;
                let arms:[&dyn Fn()->Result<()>;2]=[&||launch_old(gp,old,&fc,out.address),&||launch_new(gp,new,&fc,out.address)];
                let mut samples=vec![Vec::new();2];
                for repeat in 0..9 {for i in 0..2 {let arm=if repeat%2==0 {i} else {1-i};arms[arm]()?;gp.sync()?;
                    let start=Instant::now();for _ in 0..3 {arms[arm]()?;} gp.sync()?;samples[arm].push(start.elapsed().as_secs_f64()*1e6/3.0);}}
                let t:Vec<f64>=samples.into_iter().map(median).collect();
                println!("NVWIDE COLS {label} group={group} routing={routing} words={} equal_to_row_major=YES old_us={:.1} new_us={:.1} change_pct={:.1}",fx.units,t[0],t[1],100.0*(t[1]/t[0]-1.0));
            }
        }
        g.clear()?;
    }
    Ok(())
}
