//! Shared block FP8 forms. One-row output layout is normalized on the host for comparison.
#![forbid(unsafe_code)]
use super::*;
use super::super::super::{capture_graph,sample,ptr,TIMER};
struct Set { multi:Multi, split:Multi, singles:Vec<Vec<u8>>, out:Vec<Buf> }
fn single_args(input:u64,lut:u64,w:u64,sc:u64,out:u64,rows:usize,cols:usize)->Vec<u8> {
    let mut v=Vec::new();for p in [input,0,0,lut,w,sc,out] {v.extend(p.to_le_bytes());}
    for p in [8u32,rows as u32,cols as u32,128,128,1] {v.extend(p.to_le_bytes());}
    for p in [1.0f32,1.0,0.0] {v.extend(p.to_le_bytes());}v.extend(0u32.to_le_bytes());v.extend(0u64.to_le_bytes());v.extend(1u32.to_le_bytes());v.extend(0u32.to_le_bytes());v
}
fn launch(g:&Gpu,functions:&[H],a:usize,s:&Set)->Result<()> {
    let fx=&s.multi;let out=s.out[a].address;
    match a {
        0=>mlaunch_small(g,functions[a],&s.split,out),
        1=>mlaunch_multi(g,functions[a],fx,out),
        2=>mlaunch_few(g,functions[a],&s.split,out),
        3=>mlaunch_reference(g,functions[a],fx,out),
        4=>mlaunch_tile(g,functions[a],fx,out),
        _=>{for bytes in &s.singles {let mut bytes=bytes.clone();g.launch(functions[a],[fx.rows.div_ceil(16) as u32*fx.counts.len() as u32,1,1],160,4*16*128+4*128*4+4*128*4+3*128*4+2*16*132*4,&mut [bytes.as_mut_ptr().cast()])?;}Ok(())}
    }
}
pub(in super::super::super) fn run(path:&str)->Result<()> {
    let image=std::fs::read(path).map_err(|e|e.to_string())?;let mut g=Gpu::new()?;let module=g.module(&image)?;
    let names=["fp8_rows_small","fp8_rows_multi","fp8_rows_few","expert_group","fp8_rows_tile","expert_operation"];
    let fs=names.iter().map(|n|g.function(module,n)).collect::<Result<Vec<_>>>()?;let tm=g.module(TIMER.as_bytes())?;let timer=g.function(tm,"kb_stamp")?;
    for (label,rows,cols,parts) in [("fp8_gate_up",1024usize,4096usize,2usize),("fp8_down",2048,2048,1),("fp8_output",2048,16384,1),("fp8_qb",8192,1536,1),("fp8_qa",768,4096,1),("fp8_kva",256,4096,1)] {for m in 2..=4 {
        let current=if parts>1 {1} else if m==2||rows<=768 {0} else if rows*cols<=4_194_304 {2} else {1};
        let copies=(256usize<<20).div_ceil(rows*cols*parts).clamp(2,96);let mut sets=Vec::new();let mut equal=[true;6];
        for c in 0..copies {
            let mut rng=Rng(701+c as u64);let mut pool=Vec::new();
            for _ in 0..parts {let w=(0..rows*cols).map(|_|{let b=(rng.next()>>8) as u8;if b&127==127 {b^1} else {b}}).collect::<Vec<_>>();
                let sc=(0..rows.div_ceil(128)*cols.div_ceil(128)).map(|_|0.001*(1.0+rng.unit().abs())).collect::<Vec<_>>();pool.push((g.upload(&w)?,g.upload(&le(&sc))?));}
            let multi=build_table(&mut g,rows,cols,128,128,&vec![m;parts],0,0,601+c as u64,Some(&pool),true)?;
            // small/few select parts by first_row. Duplicate the exact shared input for those cumulative tables.
            let input=g.read(multi.input,m*cols*4)?;let copied=g.upload(&input.repeat(parts))?;
            let mut table=Vec::new();for (i,&(w,sc)) in pool.iter().enumerate() {let mut en=entry(Geometry {rows,cols,br:128,bc:128,tokens:m},w,sc);
                en[56..60].copy_from_slice(&((i*rows*m) as u32).to_le_bytes());en[60..64].copy_from_slice(&((i*m) as u32).to_le_bytes());table.extend(en);}
            let split=Multi {rows,cols,counts:vec![m;parts],table:g.upload(&table)?,input:copied,lutp:multi.lutp,units:multi.units};
            let mut out=Vec::new();for _ in 0..6 {out.push(Buf::new(&mut g,&vec![0xff;multi.units*4])?);}
            let mut singles=Vec::new();for t in 0..m {let mut entries=Vec::new();for (i,&(w,sc)) in pool.iter().enumerate() {
                let arg=single_args(multi.input+(t*cols*4) as u64,multi.lutp,w,sc,out[5].address+((i*m+t)*rows*4) as u64,rows,cols);entries.extend_from_slice(&arg[..96]);
            }let table=g.upload(&entries)?;let mut launch=entries[..96].to_vec();launch.extend(table.to_le_bytes());launch.extend((parts as u32).to_le_bytes());launch.extend(0u32.to_le_bytes());singles.push(launch);}
            let set=Set {multi,split,singles,out};for a in 0..6 {launch(&g,&fs,a,&set)?;}g.sync()?;
            let want=set.out[current].read(&g)?;
            for a in 0..6 {let mut bytes=set.out[a].read(&g)?;if bytes[..16]!=GUARD||bytes[bytes.len()-16..]!=GUARD {return Err("FP8 forms guard".into());}
                if a==5 {let raw=bytes.clone();for i in 0..parts {for r in 0..rows {for t in 0..m {let from=16+((i*m+t)*rows+r)*4;let to=16+((i*rows+r)*m+t)*4;bytes[to..to+4].copy_from_slice(&raw[from..from+4]);}}}}
                equal[a]&=bytes==want;
            }sets.push(set);
        }
        let times=g.upload(&[0;16])?;let mut samples=vec![Vec::new();6];
        {let mut graphs=Vec::new();for a in 0..6 {graphs.push(capture_graph(&g,|| {let mut stamp=times;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
            for set in &sets {launch(&g,&fs,a,set)?;}stamp=times+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;Ok(())})?);}
            for _ in 0..3 {for graph in &graphs {sample(graph,times)?;}}
            for repeat in 0..11 {for k in 0..6 {let a=(repeat+k)%6;samples[a].push(sample(&graphs[a],times)?.1/copies as f64);}}
        }
        for a in 0..6 {println!("FORMS family={label} m={m} current={} arm={} us={:.3} equal={} copies={copies} clock=graph_stamps layout_cost_included={} samples={:?}",names[current],names[a],median(samples[a].clone()),equal[a],a!=5,samples[a]);}
        g.clear()?;
    }}Ok(())
}
