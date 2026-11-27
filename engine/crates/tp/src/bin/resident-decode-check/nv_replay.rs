//! Kernel kit, part 4, replay side: run captured NVFP4 expert launches again on ONE box.
//! A capture folder (option `--nv-capture=`, written by the engine during a real step) holds everything a launch
//! read and what it wrote. For each call this tool (1) runs today's kernel through the same public host call and
//! compares every output word with the captured output (proof that the replay is the served launch), with its time;
//! (2) for a prompt group, runs the tensor-core path for each admitted word function, with its time and sampled
//! words against the CPU twin; (3) reports how many sampled words of the replayed output equal the twin of each word
//! function (on a build that has moved to the new contract, one of them is every word).
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Op,Shape},tc_contract,tc_twin};
fn words(path:&Path)->Result<Vec<f32>> {
    let bytes=fs::read(path).map_err(|e|format!("{}: {e}",path.display()))?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|w|f32::from_le_bytes(*w)).collect())
}
fn raw(values:&[f32])->Vec<u8> {values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect()}
fn field<'t>(line:&'t str,key:&str)->Result<&'t str> {
    line.split_whitespace().find_map(|w|w.strip_prefix(&format!("{key}="))).ok_or_else(||format!("capture META: no {key} in '{line}'"))
}
struct Part {shape:Shape,first_row:usize,weight:Vec<u8>,scales:Vec<u8>}
/// The twin's word for (weight row, one activation row of 17 words a block).
fn twin_word(chains:u32,p:&Part,row:usize,input:&[f32])->u32 {
    let k=p.shape.cols as usize;
    let w:Vec<u8>=(0..k).map(|col|{let (at,shift)=expert_ops::nv_code_at(&p.shape,row,col);(p.weight[at]>>shift)&15}).collect();
    let sw:Vec<u8>=(0..k/16).map(|b|p.scales[expert_ops::nv_scale_at(&p.shape,row,b)]).collect();
    let a:Vec<u8>=(0..k).map(|col|tc_contract::e2m1_code_f32(input[col/16*17+col%16].to_bits()) as u8).collect();
    let sa:Vec<u8>=(0..k/16).map(|b|tc_contract::ue4m3_code_f32(input[b*17+16].to_bits()) as u8).collect();
    tc_twin::word(tc_twin::accumulator(chains,&w,&a,&sw,&sa),p.shape.global,p.shape.activation)
}
fn best_of<'a>(backend:&Backend<'a>,mut call:impl FnMut()->Result<DeviceVector<'a>>)->Result<(f64,Vec<f32>)> {
    let mut best=f64::MAX;let mut last=Vec::new();
    for repeat in 0..5 {
        let start=Instant::now();let output=call()?;backend.synchronize()?;
        let seconds=start.elapsed().as_secs_f64();if repeat>0 {best=best.min(seconds);}
        if repeat==4 {last=backend.read_vector_values(&output)?;}
    }
    Ok((best,last))
}
/// `second` = treat every captured matrix whose columns are a multiple of 256 and whose rows are not below its
/// columns... no shape rule is safe, so the caller names the role: "second" makes every part a SECOND matrix (quarter
/// chains); anything else keeps the captured shape. A capture from a build before the rebase has no role in it.
pub fn run(captures:&Path,out:&Path,role:Option<&str>)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let mut calls:Vec<_>=fs::read_dir(captures).map_err(|e|e.to_string())?.filter_map(|e|e.ok()).map(|e|e.path()).filter(|p|p.join("META").is_file()).collect();
    calls.sort();
    if calls.is_empty() {return Err(format!("no captured call under {}",captures.display()));}
    let mut receipt=format!("NV_REPLAY captures={} calls={} timing=complete_public_call best_of_4_after_warmup\n",captures.display(),calls.len());
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        let lut=backend.upload_vector(&expert_ops::decode_lut())?;
        for dir in &calls {
            let meta=fs::read_to_string(dir.join("META")).map_err(|e|e.to_string())?;
            let head=|key:&str|->Result<String> {meta.lines().find_map(|l|l.strip_prefix(&format!("{key}="))).map(str::to_owned).ok_or_else(||format!("capture META: no {key}"))};
            if head("format")?!="vade-nv-capture-v1" {return Err("unknown capture format".into());}
            if head("tile_rows")?.parse::<u32>().ok()!=Some(expert_ops::NV_TILE_ROWS) {return Err("the capture's column-tile height differs from this build's".into());}
            let (kind,stride)=(head("kind")?,head("stride")?.parse::<usize>().map_err(|e|e.to_string())?);
            let mut parts=Vec::new();
            for (i,line) in meta.lines().filter(|l|l.starts_with("part ")).enumerate() {
                let n=|key:&str|->Result<u32> {field(line,key)?.parse().map_err(|_|format!("capture META: bad {key}"))};
                let f=|key:&str|->Result<f32> {Ok(f32::from_bits(u32::from_str_radix(field(line,key)?,16).map_err(|_|format!("capture META: bad {key}"))?))};
                let op=match n("op")? {1=>Op::NvProject,6=>Op::NvRows,other=>return Err(format!("capture part {i}: op {other} is not an NVFP4 product"))};
                let shape=Shape {op,rows:n("rows")?,cols:n("cols")?,block_rows:n("block_rows")?,block_cols:n("block_cols")?,tokens:n("tokens")?,global:f("global")?,activation:f("activation")?,
                    // A capture written before the named role has no `role` word: it is a first matrix unless the caller says "second".
                    bound:f("bound")?,role:if role==Some("second") {expert_ops::NV_SECOND} else {field(line,"role").ok().map_or(Ok(expert_ops::NV_FIRST),|v|v.parse::<u32>().map_err(|_|"capture META: bad role".to_string()))?}};
                parts.push(Part {shape,first_row:n("first_row")? as usize,weight:raw(&words(&dir.join(format!("w-{i}.bin")))?),scales:raw(&words(&dir.join(format!("s-{i}.bin")))?)});
            }
            let captured=words(&dir.join("output.bin"))?;
            let inputs:Vec<Vec<f32>>=(0..head("inputs")?.parse::<usize>().map_err(|e|e.to_string())?).map(|i|words(&dir.join(format!("input-{i}.bin")))).collect::<Result<_>>()?;
            let device:Vec<(DeviceVector<'_>,DeviceVector<'_>)>=parts.iter().map(|p|Ok((backend.upload_vector(&words_of(&p.weight))?,backend.upload_vector(&words_of(&p.scales))?))).collect::<Result<_>>()?;
            let device_inputs:Vec<DeviceVector<'_>>=inputs.iter().map(|v|backend.upload_vector(v)).collect::<std::result::Result<_,String>>()?;
            let (r,k)=(parts[0].shape.rows as usize,parts[0].shape.cols as usize);
            let routed:usize=parts.iter().map(|p|p.shape.tokens as usize).sum();
            let name=dir.file_name().and_then(|n|n.to_str()).unwrap_or("call");
            // The activation row of (part, token) and the output word index of (part, row, token).
            let row_of=|pi:usize,token:usize|->&[f32] {if kind=="group" {&inputs[0][(parts[pi].first_row+token)*stride..][..stride]} else {&inputs[pi][..stride]}};
            let offsets:Vec<usize>=parts.iter().scan(0usize,|at,p|{let here=*at;*at+=p.shape.rows as usize*p.shape.tokens as usize;Some(here)}).collect();
            // Sampled words (rows 0, 1, last of the first and last part, every token) of an output against a twin.
            let sample=|got:&[f32],chains:u32|->(usize,usize) {
                if tc_contract::chain_steps(k,chains).is_none() {return (0,0);}
                let (mut n,mut bad)=(0usize,0usize);
                for pi in [0,parts.len()-1] {let m=parts[pi].shape.tokens as usize;
                    for row in [0,1.min(r-1),r-1] {for token in 0..m {
                        n+=1;if got[offsets[pi]+row*m+token].to_bits()!=twin_word(chains,&parts[pi],row,row_of(pi,token)) {bad+=1;}
                    }}
                }
                (n,bad)
            };
            let (today,replayed)=if kind=="group" {
                let table:Vec<_>=parts.iter().zip(&device).map(|(p,(w,s))|(p.shape,Some((w,s)),p.first_row)).collect();
                best_of(backend,||backend.expert_group(&table,&device_inputs[0],stride,&lut))?
            } else {
                let jobs:Vec<_>=parts.iter().zip(&device).zip(&device_inputs).map(|((p,(w,s)),input)|(p.shape,input,w,s)).collect();
                best_of(backend,||backend.expert_projection_batch(&jobs))?
            };
            let differ=raw(&replayed).as_chunks::<4>().0.iter().zip(raw(&captured).as_chunks::<4>().0.iter()).filter(|(a,b)|a!=b).count()+replayed.len().abs_diff(captured.len());
            let (n1,b1)=sample(&replayed,1);let (n4,b4)=sample(&replayed,4);
            let bytes=parts.len()*(r*k/2+r*k/16)+routed*stride*4+routed*r*4;
            let mut line=format!("NV_REPLAY call={name} kind={kind} parts={} rows={r} cols={k} routed_rows={routed} this_build_ms={:.3} bus_floor_ms={:.3} words={} replay_vs_capture_differ={differ} sampled_vs_twin1_differ={b1}/{n1} sampled_vs_twin4_differ={b4}/{n4}",
                parts.len(),today*1e3,bytes as f64/235e9*1e3,captured.len());
            line+=&format!(" word_function_chains={}",expert_ops::nv_chains(&parts[0].shape));
            println!("{line}");receipt.push_str(&line);receipt.push('\n');
        }
        Ok(())
    })?;
    receipt.push_str("NV_REPLAY_DONE\n");
    fs::write(out.join("RECEIPT"),receipt).map_err(|e|e.to_string())?;Ok(())
}
fn words_of(bytes:&[u8])->Vec<f32> {bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect()}
