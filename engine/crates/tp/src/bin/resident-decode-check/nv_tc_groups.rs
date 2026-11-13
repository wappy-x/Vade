//! Kernel kit: NVFP4 prompt groups on the tensor-core contract through the REAL host path (`Backend::expert_group`).
//! The word function comes from the matrix shape (`expert_ops::nv_chains`: first matrix = one chain, second matrix =
//! quarter chains). Part 1: every output word equals the CPU twin (`tc_twin`) on the serving choice (tensor-core
//! image for dense groups, decode jobs for groups of few tokens) AND with the tensor-core image switched off (decode
//! jobs or the generic body): three device paths, one word. Part 2: prompt groups of 256, 512 and 1,024 rows at the
//! real local shapes with 288 experts, timed against the bus floor of the launch.
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Op,Shape},tc_twin};
fn mixed(index:usize,job:usize)->u32 {
    let mut value=(index as u32).wrapping_add((job as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
fn upload<'a>(backend:&Backend<'a>,bytes:&[u8])->Result<DeviceVector<'a>> {
    backend.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())
}
/// One expert: its codes (one a byte, row-major) and scale codes for the twin, and its column-tile owners on the device.
struct Expert<'a> {codes:Vec<u8>,scales:Vec<u8>,weight:DeviceVector<'a>,scale:DeviceVector<'a>,shape:Shape}
fn expert<'a>(backend:&Backend<'a>,rows:u32,cols:u32,job:usize,keep:bool,chains:u32)->Result<Expert<'a>> {
    let (r,k)=(rows as usize,cols as usize);
    let bytes=(0..r*k/2).map(|index|mixed(index,job) as u8).collect::<Vec<_>>();
    let scales=(0..r*k/16).map(|index|if index%17==0 {0} else {0x28+(mixed(index,job+31)%64) as u8}).collect::<Vec<_>>();
    let weight=upload(backend,&expert_ops::nv_tile_codes(&bytes,r,k))?;let scale=upload(backend,&expert_ops::nv_tile_scales(&scales,r,k))?;
    let shape=Shape {op:Op::NvRows,rows,cols,block_rows:expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:0.75+(job%16) as f32/16.0,activation:0.125+(job%16) as f32/32.0,bound:0.0,role:if chains==4 {expert_ops::NV_SECOND} else {expert_ops::NV_FIRST}};
    // The twin reads one code a byte. Big timing fixtures keep only the rows the samples read.
    let codes=if keep {bytes.iter().flat_map(|b|[b&15,b>>4]).collect()} else {bytes[..k/2].iter().chain(&bytes[k/2..k]).chain(&bytes[(r-1)*k/2..]).flat_map(|b|[b&15,b>>4]).collect()};
    let scales=if keep {scales} else {scales[..k/16].iter().chain(&scales[k/16..k/8]).chain(&scales[(r-1)*k/16..]).copied().collect()};
    Ok(Expert {codes,scales,weight,scale,shape})
}
/// One quantized activation row as the quantizer writes it (17 words a block) and its codes for the twin.
fn token_row(cols:usize,token:usize,sparse:bool)->(Vec<f32>,Vec<u8>,Vec<u8>) {
    let (mut words,mut codes,mut scales)=(Vec::with_capacity(cols/16*17),Vec::with_capacity(cols),Vec::with_capacity(cols/16));
    for block in 0..cols/16 {
        for e in 0..16 {
            let h=mixed(block*16+e,token+1000);
            let code=if sparse && h%5<2 {(h>>8&8) as u8} else {(h>>4&15) as u8};
            codes.push(code);words.push(if code&8!=0 {-expert_ops::e2m1(code&7)} else {expert_ops::e2m1(code)});
        }
        let scale=if mixed(block,token+7).is_multiple_of(29) {(mixed(block,token+9)%8) as u8} else {0x30+(mixed(block,token+11)%20) as u8};
        scales.push(scale);words.push(expert_ops::e4m3(scale));
    }
    (words,codes,scales)
}
fn count(name:&str)->u64 {spark_gpu_probe::prompt_launches::line().split_whitespace().find_map(|word|word.strip_prefix(&format!("{name}="))).and_then(|value|value.parse().ok()).unwrap_or(0)}
/// Top-8 routing of `group` tokens over `experts` experts with a skew, as the kernel bench: the token list of each expert.
fn route(group:usize,experts:usize,seed:usize)->Vec<Vec<usize>> {
    let weights:Vec<f64>=(0..experts).map(|i|((i+1) as f64).powf(-0.5)).collect();let total:f64=weights.iter().sum();
    let mut lists=vec![Vec::new();experts];let mut draw=0usize;
    for token in 0..group {
        let mut chosen:Vec<usize>=Vec::new();
        while chosen.len()<8.min(experts) {
            draw+=1;let mut u=f64::from(mixed(draw,seed))/4294967296.0*total;let mut pick=experts-1;
            for (i,w) in weights.iter().enumerate() {if u<*w {pick=i;break;}u-=w;}
            if !chosen.contains(&pick) {chosen.push(pick);lists[pick].push(token);}
        }
    }
    lists
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("NV_TC_GROUPS host_path=Backend::expert_group word_function=from_the_matrix_shape timing=complete_public_call_including_allocation_table_and_pack\n");
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        if !backend.tc_image_present() {return Err("this build has no tensor-core image (cargo kernels --tc)".into());}
        let lut=backend.upload_vector(&expert_ops::decode_lut())?;
        // Part 1: every word against the CPU twin, both word functions, through the host path.
        for chains in [1u32,4] { for tensor_core in [true,false] {
            backend.set_nv_tc_groups(tensor_core);
            for (rows,cols) in [(128u32,256u32),(256,1024),(1024,4096),(2048,2048)] {
                // The generic body is one thread a word: keep its cases small.
                if !tensor_core && rows>256 {continue;}
                let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
                let experts=(0..5).map(|job|expert(backend,rows,cols,job,true,chains)).collect::<Result<Vec<_>>>()?;
                let mut checked=0usize;
                for (case,tokens) in [vec![8usize,3,9],vec![5,16],vec![1,1,1,1],vec![130,1,64],vec![200],vec![2,2,1,129,7]].into_iter().enumerate() {
                    let total:usize=tokens.iter().sum();
                    let rows_in:Vec<(Vec<f32>,Vec<u8>,Vec<u8>)>=(0..total).map(|t|token_row(k,t+37*case,(t+case)%3==0)).collect();
                    let input=backend.upload_vector(&rows_in.iter().flat_map(|row|row.0.iter().copied()).collect::<Vec<_>>())?;
                    let mut first=0;let mut parts=Vec::new();
                    for (part,&m) in tokens.iter().enumerate() {parts.push((Shape {tokens:m as u32,..experts[part].shape},Some((&experts[part].weight,&experts[part].scale)),first));first+=m;}
                    let before=count("fp4_cohort");
                    let output=backend.expert_group(&parts,&input,stride,&lut)?;backend.synchronize()?;
                    // Dense lists take the tensor-core image; lists of one token a part take decode jobs (the serving choice).
                    let dense=total>=2*tokens.len();
                    if tensor_core && dense && count("fp4_cohort")!=before+1 {return Err(format!("the tensor-core group kernel did not run rows={rows} cols={cols} case={case}"));}
                    if !tensor_core && count("fp4_cohort")!=before {return Err(format!("the tensor-core group kernel ran with the image switched off rows={rows} cols={cols} case={case}"));}
                    let got=backend.read_vector_values(&output)?;
                    let (mut at,mut first)=(0usize,0usize);
                    for (part,&m) in tokens.iter().enumerate() {
                        let e=&experts[part];
                        for row in 0..r {for token in 0..m {
                            let (_,codes,scales)=&rows_in[first+token];
                            let acc=tc_twin::accumulator(chains,&e.codes[row*k..][..k],codes,&e.scales[row*k/16..][..k/16],scales);
                            let want=tc_twin::word(acc,e.shape.global,e.shape.activation);
                            let have=got[at+row*m+token].to_bits();
                            if have!=want {return Err(format!("NV TC BITS DIFFER chains={chains} rows={rows} cols={cols} case={case} part={part} row={row} token={token} device={have:08x} twin={want:08x}"));}
                            checked+=1;
                        }}
                        at+=r*m;first+=m;
                    }
                }
                let line=format!("NV_TC_WORDS_PASS chains={chains} rows={rows} cols={cols} words={checked} differ=0 path={}\n",if !tensor_core {"decode jobs or generic body (tensor-core image off)"} else if chains==4 {"fp4_cohort_t64 + tc_nv_pack (dense), nv_cols jobs (few tokens)"} else {"fp4_cohort + tc_nv_pack (dense), nv_cols jobs (few tokens)"});
                print!("{line}");receipt.push_str(&line);
            }
        } }
        backend.set_nv_tc_groups(true);
        // Part 2: prompt groups of 256, 512 and 1,024 rows, 288 experts, both local shapes: today's kernel against
        // the tensor-core path, the same public call. Words: rows 0, 1 and the last of the first and last part.
        for (rows,cols,label,chains) in [(1024u32,4096u32,"gate_up",1u32),(2048,2048,"down",4)] {
            let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
            let experts=(0..288).map(|job|expert(backend,rows,cols,job,false,chains)).collect::<Result<Vec<_>>>()?;
            for group in [256usize,512,1024] {
                let lists:Vec<(usize,Vec<usize>)>=route(group,288,group).into_iter().enumerate().filter(|(_,l)|!l.is_empty()).collect();
                let routed:usize=lists.iter().map(|(_,l)|l.len()).sum();
                let mut words=Vec::with_capacity(routed*stride);let mut twin_rows=Vec::new();
                for (pi,(_,list)) in lists.iter().enumerate() {for &t in list {let (w,c,s)=token_row(k,t,t%3==0);words.extend(w);if pi==0 || pi+1==lists.len() {twin_rows.push((c,s));}}}
                let input=backend.upload_vector(&words)?;drop(words);
                let mut first=0;let mut parts=Vec::new();
                for (e,list) in &lists {parts.push((Shape {tokens:list.len() as u32,..experts[*e].shape},Some((&experts[*e].weight,&experts[*e].scale)),first));first+=list.len();}
                let time=||->Result<(f64,Vec<f32>)> {
                    let mut best=f64::MAX;let mut last=Vec::new();
                    for repeat in 0..6 {
                        let start=Instant::now();let output=backend.expert_group(&parts,&input,stride,&lut)?;backend.synchronize()?;
                        let seconds=start.elapsed().as_secs_f64();if repeat>0 {best=best.min(seconds);}
                        if repeat==5 {last=backend.read_vector_values(&output)?;}
                    }
                    Ok((best,last))
                };
                let (new,got)=time()?;
                // Sampled words of the tensor-core output against the twin.
                let (mut checked,mut at)=(0usize,0usize);let last_start=routed-lists[lists.len()-1].1.len();let _=last_start;
                let mut twin_at=0usize;
                for (pi,(e,list)) in lists.iter().enumerate() {
                    let m=list.len();
                    if pi==0 || pi+1==lists.len() {
                        let x=&experts[*e];
                        for (sample,row) in [0usize,1,r-1].into_iter().enumerate() {for token in 0..m {
                            let (codes,scales)=&twin_rows[twin_at+token];
                            let acc=tc_twin::accumulator(chains,&x.codes[sample*k..][..k],codes,&x.scales[sample*k/16..][..k/16],scales);
                            let want=tc_twin::word(acc,x.shape.global,x.shape.activation);let have=got[at+row*m+token].to_bits();
                            if have!=want {return Err(format!("NV TC BITS DIFFER {label} group={group} part={pi} row={row} token={token} device={have:08x} twin={want:08x}"));}
                            checked+=1;
                        }}
                        twin_at+=m;
                    }
                    at+=r*m;
                }
                // Bytes the product launch must move: each live expert's codes and scales once, the packed operands, the output.
                let bytes=lists.len()*(r*k/2+r*k/16)+routed*(k/2+k/16)+routed*r*4;let floor=bytes as f64/235e9;
                let line=format!("NV_TC_GROUP {label} chains={chains} rows={rows} cols={cols} group={group} parts={} routed_rows={routed} tensor_core_ms={:.3} bus_floor_ms={:.3} tensor_core_over_floor={:.2} sampled_words={checked} differ=0\n",
                    lists.len(),new*1e3,floor*1e3,new/floor);
                print!("{line}");receipt.push_str(&line);
            }
        }
        Ok(())
    })?;
    let line=format!("NV_TC_GROUPS_PASS launches: {}\n",spark_gpu_probe::prompt_launches::line());
    print!("{line}");receipt.push_str(&line);
    fs::write(out.join("RECEIPT"),receipt).map_err(|error|error.to_string())?;Ok(())
}
