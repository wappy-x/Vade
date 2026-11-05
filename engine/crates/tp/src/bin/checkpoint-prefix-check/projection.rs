//! Real stored prefix followed by complete selected MXFP8 projection rows.
use super::{checkpoint,child,equal,floats,prompt,read,sha256,topology,write,Prefix};
#[cfg(target_os="linux")]
use super::oracle;
use engine_model::plan::{admission::CheckpointAdmission,layer::Topology,state::StatePlan};
use spark_tp::{Result,layer_driver::LayerBoundary,mxfp8_projection::Mxfp8Projection};
use std::{fs,path::Path};

const PLANES:[&str;7]=["embedding.f32","mhc-normalized.f32","coefficients.f32","collapsed.f32","input-norm.f32","projection-accumulator.f32","projection-bf16.f32"];
const META:[&str;6]=["COMMON","IDENTITY","SCOPE","INPUT_IDS","PLANE_WORDS","GEOMETRY"];
const SCOPE:&[u8]=b"stored_identity_candidate=true\nactivation=completed_norm_dtype_no_quantizer\nepilogue=rn_even_bf16\nfull_layer=false\ntransport=false\ngather_completed=false\noutput_token_ids=false\nruntime_load_policy=unqualified\ncheckpoint_linear_replacement=unqualified\n";
const COMPLETE:&[u8]=b"CHECKPOINT MXFP8 PROJECTION v1 PASS\n";
const SUITE:&[u8]=b"CHECKPOINT MXFP8 PROJECTION SUITE v1 PASS\n";

struct Projection<'a> {prefix:Prefix<'a>,bound:Mxfp8Projection<'a>,weight:Vec<u8>,scale:Vec<u8>}
impl<'a> Projection<'a> {
    fn new(c:&'a CheckpointAdmission,t:Topology)->Result<Self> {
        let prefix=Prefix::new(c,t)?;
        let state=StatePlan::new(c.description(),t,1,isize::MAX as usize).map_err(|e|e.to_string())?;
        let b=LayerBoundary::new(c.description(),c.headers(),&state,0,1,t,usize::MAX)?;
        let bound=b.prepare_mxfp8_projection_checkpoint(3,c)?;
        if bound.input_name()!=b.program().steps[2].outputs[0].name || bound.shape().cols()!=prefix.width {
            return Err("projection input is not the complete stored prefix norm".into());
        }
        let (weight,scale)=bound.read()?;Ok(Self {prefix,bound,weight,scale})
    }
    fn setup(&self,root:&Path,out:&Path,ids:&[u32])->Result<()> {
        let p=&self.bound;let s=p.shape();let t=p.topology();
        let common=self.prefix.identity(root,ids,t.parts)?;
        write(&out.join("COMMON"),&common)?;
        let identity=format!("checkpoint-mxfp8-projection-v1\ncommon={}\ncomponents={:?}\ncomponent_offsets={:?}\ncomponent_file_offsets={:?}\nweight={}\nscale={}\ninput={}\ninput_dtype={:?}\noutput={}\nepilogue=rn_even_bf16\nrequires_gather={}\n",
            sha256::digest(&common),p.component_names(),p.component_offsets(),p.component_file_offsets(),
            sha256::digest(&self.weight),sha256::digest(&self.scale),p.input_name(),s.input(),p.output_name(),p.requires_gather());
        write(&out.join("IDENTITY"),identity.as_bytes())?;
        write(&out.join("SCOPE"),SCOPE)?;
        write(&out.join("INPUT_IDS"),ids.iter().map(|id|format!("{id}\n")).collect::<String>().as_bytes())?;
        write(&out.join("GEOMETRY"),format!("{} {} {} {} {} {}\n",t.rank,t.parts,p.row_origin(),s.rows(),s.cols(),p.total_rows()).as_bytes())?;
        let m=self.prefix.block.mhc().shape();
        write(&out.join("PLANE_WORDS"),format!("{} {} {} {} {} {} {}\n",m.wide(),m.wide(),m.coefficients(),s.cols(),s.cols(),s.rows(),s.rows()).as_bytes())
    }
}
fn names(ids:&[u32])->Vec<String> {
    META.into_iter().map(str::to_owned).chain((0..ids.len()).flat_map(|i|PLANES.map(|n|format!("{i}-{n}")))).collect()
}
fn manifest(out:&Path,ids:&[u32])->Result<Vec<u8>> {
    let mut text=String::new();for name in names(ids) {
        text.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(&name))?)));
    }Ok(text.into_bytes())
}
fn seal(out:&Path,ids:&[u32])->Result<()> {
    write(&out.join("OUTPUT.sha256"),&manifest(out,ids)?)?;write(&out.join("COMPLETE"),COMPLETE)
}
fn numbers(path:&Path,count:usize)->Result<Vec<usize>> {
    let data=engine_model::safetensors::read_bounded(path,1024).map_err(|e|e.to_string())?;
    let values=std::str::from_utf8(&data).map_err(|e|e.to_string())?.split_whitespace()
        .map(|s|s.parse().map_err(|_|"invalid projection receipt integer".to_owned())).collect::<Result<Vec<usize>>>()?;
    if values.len()!=count {return Err("projection receipt extent differs".into());}Ok(values)
}
fn complete(out:&Path)->Result<Vec<u32>> {
    equal("projection completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;
    equal("projection scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let ids=prompt(&out.join("INPUT_IDS"))?;let g=numbers(&out.join("GEOMETRY"),6)?;
    topology(&g[0].to_string(),&g[1].to_string())?;
    if g[3]==0 || g[4]==0 || g[4]%32!=0 || g[2].checked_add(g[3]).is_none_or(|n|n>g[5]) {
        return Err("projection receipt has incomplete row geometry".into());
    }
    let words=numbers(&out.join("PLANE_WORDS"),PLANES.len())?;
    if words.iter().any(|&n|n==0 || n>isize::MAX as usize/4) || words[3..]!=[g[4],g[4],g[3],g[3]] {
        return Err("projection plane extents differ from their row geometry".into());
    }
    for i in 0..ids.len() {for (j,name) in PLANES.into_iter().enumerate() {
        let bytes=read(&out.join(format!("{i}-{name}")))?;
        if bytes.len()!=words[j]*4 || bytes.chunks_exact(4).any(|b| {
            let v=f32::from_le_bytes(b.try_into().unwrap());
            !v.is_finite() || j==6 && spark_gpu_probe::bf16::rounded(v).to_bits()!=v.to_bits()
        }) {return Err(format!("invalid projection plane {i}-{name}"));}
    }}
    equal("projection manifest",&read(&out.join("OUTPUT.sha256"))?,&manifest(out,&ids)?)?;Ok(ids)
}
fn compare(a:&Path,b:&Path)->Result<()> {
    let ids=complete(a)?;if complete(b)?!=ids {return Err("projection input IDs differ".into());}
    for name in names(&ids) {equal(&name,&read(&a.join(&name))?,&read(&b.join(&name))?)?;}
    println!("CHECKPOINT MXFP8 COMPARE PASS input_tokens={} output_token_ids=false",ids.len());Ok(())
}
fn bounded_prompt(input:&Path)->Result<Vec<u32>> {
    let ids=prompt(input)?;
    if ids.len()>64 {return Err("bounded projection probe permits at most 64 input IDs".into());}Ok(ids)
}
fn host(root:&Path,input:&Path,out:&Path,t:Topology)->Result<()> {
    let ids=bounded_prompt(input)?;let c=checkpoint(root)?;let p=Projection::new(&c,t)?;p.setup(root,out,&ids)?;
    for (i,&id) in ids.iter().enumerate() {
        let prefix=p.prefix.expected(id)?;
        let (acc,output)=p.bound.shape().reference(&p.weight,&p.scale,&prefix[4])?;
        for (name,values) in PLANES.into_iter().zip(prefix.into_iter().chain([acc,output])) {
            write(&out.join(format!("{i}-{name}")),&floats(&values)?)?;
        }
    }
    seal(out,&ids)?;println!("CHECKPOINT MXFP8 HOST PASS rank={} rows={} cols={} full_layer=false",t.rank,p.bound.shape().rows(),p.bound.shape().cols());Ok(())
}
#[cfg(target_os="linux")]
fn device(root:&Path,input:&Path,expected:&Path,out:&Path,t:Topology)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    let ids=bounded_prompt(input)?;if complete(expected)?!=ids {return Err("expected projection input IDs differ".into());}
    let c=checkpoint(root)?;let p=Projection::new(&c,t)?;p.setup(root,out,&ids)?;
    for name in META {equal(name,&read(&out.join(name))?,&read(&expected.join(name))?)?;}
    let s=p.bound.shape();let budget=oracle::budget(&p.prefix.block)?;
    let extra=s.rows().checked_mul(8).and_then(|n|n.checked_add(s.arena_bytes().ok()?)).ok_or("projection budget overflow")?;
    let device=budget.0.checked_add(extra).ok_or("projection budget overflow")?;
    let required=budget.1.checked_add(extra).ok_or("projection budget overflow")?;
    with_backend(0,layer_ptx()?,device,|b| {
        let available=b.available_memory_bytes()?;
        if required>available {return Err(format!("projection needs {required} bytes; available {available}"));}
        println!("PROJECTION BUDGET device={device} required={required} available={available} estimated=true");
        let block=p.prefix.block.upload(b)?;let projection=p.bound.load(b)?;
        for (i,&id) in ids.iter().enumerate() {
            let embedded=oracle::device_embedding(b,&p.prefix,id)?;
            let embedding=b.read_vector_values(&embedded)?;
            let pass=block.begin(b.round_bf16_lanes(&embedded)?)?;
            let projected=projection.apply(pass.normalized())?;
            let values=[embedding,b.read_vector_values(pass.mhc().normalized())?,b.read_vector_values(pass.mhc().coefficients())?,
                b.read_vector_values(pass.mhc().collapsed().vector())?,b.read_vector_values(pass.normalized().vector())?,
                b.read_vector_values(projected.accumulator())?,b.read_vector_values(projected.output().vector())?];
            for (name,values) in PLANES.into_iter().zip(values) {
                let name=format!("{i}-{name}");let bytes=floats(&values)?;write(&out.join(&name),&bytes)?;
                equal(&name,&bytes,&read(&expected.join(&name))?)?;
            }
            println!("PROJECTION rank={} position={i} input_token={id} output_token_ids=false",t.rank);
        }
        if let Some(e)=b.fatal_error() {return Err(e);}Ok(())
    })?;
    seal(out,&ids)?;println!("CHECKPOINT MXFP8 DEVICE PASS rank={} full_layer=false transport=false",t.rank);Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path,_:Topology)->Result<()> {Err("projection device cell requires Linux and CUDA".into())}
fn host_child(root:&Path,input:&Path,out:&Path,log:&Path,t:Topology)->Result<()> {
    let rank=t.rank.to_string();let parts=t.parts.to_string();
    child(&[Path::new("projection"),Path::new("host"),root,input,out,Path::new(&rank),Path::new(&parts)],log)
}
/// Check selected rows against the unsplit host oracle in global row order.
/// This is an offline comparison. It is not a completed device gather.
fn joined(full:&Path,shards:&[std::path::PathBuf])->Result<()> {
    let ids=complete(full)?;let f=numbers(&full.join("GEOMETRY"),6)?;let mut origin=0;
    let mut outputs=vec![[Vec::new(),Vec::new()];ids.len()];
    for (rank,path) in shards.iter().enumerate() {
        if complete(path)?!=ids {return Err("projection shard input IDs differ".into());}
        let g=numbers(&path.join("GEOMETRY"),6)?;
        let expected=[rank,shards.len(),origin,g[3],f[4],f[5]];
        if g!=expected {
            return Err("projection shards omit or overlap complete output rows".into());
        }
        origin=origin.checked_add(g[3]).ok_or("projection joined row overflow")?;
        for (i,output) in outputs.iter_mut().enumerate() {
            for name in &PLANES[..5] {equal("replicated prefix",&read(&full.join(format!("{i}-{name}")))?,&read(&path.join(format!("{i}-{name}")))?)?;}
            for (plane,name) in PLANES[5..].iter().enumerate() {output[plane].extend(read(&path.join(format!("{i}-{name}")))?);}
        }
    }
    if origin!=f[5] {return Err("projection shards do not cover the complete output".into());}
    for (i,output) in outputs.iter().enumerate() {for (plane,name) in PLANES[5..].iter().enumerate() {
        equal("joined output rows",&output[plane],&read(&full.join(format!("{i}-{name}")))?)?;
    }}
    println!("CHECKPOINT MXFP8 JOIN PASS rows={origin} transport=false gather_completed=false");Ok(())
}
fn suite(root:&Path,input:&Path,out:&Path,parts:u32)->Result<()> {
    bounded_prompt(input)?;topology("0",&parts.to_string())?;
    let full=out.join("host-full");host_child(root,input,&full,&out.join("host-full.log"),Topology {rank:0,parts:1})?;
    write(&out.join("PARTS"),format!("{parts}\n").as_bytes())?;
    for rank in 0..parts {
        let t=Topology {rank,parts};let host=out.join(format!("rank{rank}-host"));
        host_child(root,input,&host,&out.join(format!("rank{rank}-host.log")),t)?;
        for run in 1..=3 {
            let result=out.join(format!("rank{rank}-run{run}"));let rank_text=rank.to_string();let parts_text=parts.to_string();
            child(&[Path::new("projection"),Path::new("device"),root,input,&host,&result,Path::new(&rank_text),Path::new(&parts_text)],
                &out.join(format!("rank{rank}-run{run}.log")))?;
            compare(&host,&result)?;compare(&out.join(format!("rank{rank}-run1")),&result)?;
        }
    }
    for run in 1..=3 {joined(&full,&(0..parts).map(|r|out.join(format!("rank{r}-run{run}"))).collect::<Vec<_>>())?;}
    write(&out.join("COMPLETE"),SUITE)?;
    println!("CHECKPOINT MXFP8 SUITE PASS parts={parts} repeats=3 output_token_ids=false full_layer=false");Ok(())
}
fn compare_suite(a:&Path,b:&Path)->Result<()> {
    for p in [a,b] {equal("projection suite completion",&read(&p.join("COMPLETE"))?,SUITE)?;}
    let parts=numbers(&a.join("PARTS"),1)?;
    if parts!=numbers(&b.join("PARTS"),1)? || parts[0]==0 {return Err("projection suite topology differs".into());}
    compare(&a.join("host-full"),&b.join("host-full"))?;
    for run in 1..=3 {
        for rank in 0..parts[0] {
            let name=format!("rank{rank}-run{run}");compare(&a.join(&name),&b.join(&name))?;
            for p in [a,b] {compare(&p.join(format!("rank{rank}-host")),&p.join(&name))?;compare(&p.join(format!("rank{rank}-run1")),&p.join(&name))?;}
        }
        for p in [a,b] {joined(&p.join("host-full"),&(0..parts[0]).map(|r|p.join(format!("rank{r}-run{run}"))).collect::<Vec<_>>())?;}
    }
    println!("CHECKPOINT MXFP8 CROSS SPARK PASS parts={} repeats=3 output_token_ids=false transport=false",parts[0]);Ok(())
}
pub fn run(args:&[&str])->Result<()> {
    match args {
        ["host",root,input,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;host(Path::new(root),Path::new(input),Path::new(out),t)},
        ["device",root,input,expected,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;device(Path::new(root),Path::new(input),Path::new(expected),Path::new(out),t)},
        ["suite",root,input,out,parts]=>{let t=topology("0",parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;suite(Path::new(root),Path::new(input),Path::new(out),t.parts)},
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),
        ["compare-suite",a,b]=>compare_suite(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check projection host MODEL INPUT_IDS NEW_OUT RANK PARTS | device MODEL INPUT_IDS EXPECTED NEW_OUT RANK PARTS | suite MODEL INPUT_IDS NEW_OUT PARTS | compare LEFT RIGHT | compare-suite LEFT RIGHT".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64,Ordering};
    struct Temp(std::path::PathBuf);
    impl Temp {fn new()->Self {
        static NEXT:AtomicU64=AtomicU64::new(0);
        let p=std::env::temp_dir().join(format!("vade-projection-receipt-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&p).unwrap();Self(p)
    }}
    impl Drop for Temp {fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();}}
    fn receipt(root:&Path,rank:usize,parts:usize,origin:usize,rows:usize)->std::path::PathBuf {
        let out=root.join(format!("p{parts}-r{rank}"));fs::create_dir(&out).unwrap();
        for (name,data) in [("COMMON",b"fixture\n".as_slice()),("IDENTITY",b"synthetic receipt, no device proof\n"),
            ("SCOPE",SCOPE),("INPUT_IDS",b"1\n")] {write(&out.join(name),data).unwrap();}
        write(&out.join("GEOMETRY"),format!("{rank} {parts} {origin} {rows} 32 4\n").as_bytes()).unwrap();
        write(&out.join("PLANE_WORDS"),format!("32 32 8 32 32 {rows} {rows}\n").as_bytes()).unwrap();
        for (i,name) in PLANES.iter().enumerate() {
            let v=if i>=5 {(origin..origin+rows).map(|j|j as f32).collect::<Vec<_>>()}
                else {vec![1.0;if i==2 {8} else {32}]};
            write(&out.join(format!("0-{name}")),&floats(&v).unwrap()).unwrap();
        }seal(&out,&[1]).unwrap();out
    }
    #[test]
    fn byte_receipts_compare_complete_rows_and_reject_corruption() {
        let temp=Temp::new();let full=receipt(&temp.0,0,1,0,4);
        let a=receipt(&temp.0,0,2,0,2);let b=receipt(&temp.0,1,2,2,2);
        compare(&a,&a).unwrap();joined(&full,&[a.clone(),b.clone()]).unwrap();
        assert!(compare(&a,&b).is_err());assert!(joined(&full,&[b.clone(),a.clone()]).is_err());
        fs::write(b.join("0-projection-bf16.f32"),[0u8;8]).unwrap();assert!(complete(&b).is_err());
        fs::write(b.join("0-projection-bf16.f32"),floats(&[2.0,3.0]).unwrap()).unwrap();
        fs::write(b.join("COMPLETE"),b"FAILED\n").unwrap();assert!(complete(&b).is_err());
        fs::write(b.join("COMPLETE"),COMPLETE).unwrap();
        fs::write(b.join("GEOMETRY"),b"1 2 1 2 32 4\n").unwrap();
        fs::write(b.join("OUTPUT.sha256"),manifest(&b,&[1]).unwrap()).unwrap();
        assert!(joined(&full,&[a,b]).is_err());
    }
    #[test]
    fn receipt_refuses_non_bf16_epilogues_even_with_a_new_manifest() {
        let temp=Temp::new();let p=receipt(&temp.0,0,1,0,4);
        fs::write(p.join("0-projection-bf16.f32"),floats(&[1.0001,1.0,2.0,3.0]).unwrap()).unwrap();
        fs::write(p.join("OUTPUT.sha256"),manifest(&p,&[1]).unwrap()).unwrap();assert!(complete(&p).is_err());
        fs::write(p.join("0-projection-bf16.f32"),floats(&[0.0,1.0,2.0,3.0]).unwrap()).unwrap();
        fs::write(p.join("PLANE_WORDS"),b"32 32 8 32 16 4 4\n").unwrap();
        fs::write(p.join("OUTPUT.sha256"),manifest(&p,&[1]).unwrap()).unwrap();assert!(complete(&p).is_err());
    }
}
