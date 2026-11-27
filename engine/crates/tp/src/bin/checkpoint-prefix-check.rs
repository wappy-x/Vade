//! Real stored embedding to attention mHC and input norm. No decoder admission.
#![forbid(unsafe_code)]
use engine_model::{plan::{ConfigPlan,admission::CheckpointAdmission,layer::{Kind,Topology},state::StatePlan,storage::WeightStorage},
    safetensors::{read_bounded,Dtype},shapes::{CONFIG_BYTES,HEADER_BYTES}};
use spark_tp::{Result,layer_driver::LayerBoundary,layer_operands::ResidualBlockOperands};
use std::{fs,io::Write,path::Path};
#[path="../../../../tools/export/src/sha256.rs"] mod sha256;
#[allow(dead_code)]
#[path="../../../../crates/gpu/kernels/src/qsa_common.rs"] mod qsa_common;
#[allow(dead_code)]
#[path="../../../../crates/gpu/src/mhc_params.rs"] mod mhc_params;
#[allow(dead_code)]
#[allow(clippy::chunks_exact_to_as_chunks)] // Shared with the NVPTX crate; fix it with the kernels.
#[path="../../../../crates/gpu/kernels/src/mhc_math.rs"] mod mhc_math;
#[path="checkpoint-prefix-check/oracle.rs"] mod oracle;
#[path="checkpoint-prefix-check/projection.rs"] mod projection;
#[path="checkpoint-prefix-check/attention.rs"] mod attention;
#[path="checkpoint-prefix-check/layer.rs"] mod layer;
#[path="checkpoint-prefix-check/tokens.rs"] mod tokens;
#[path="checkpoint-prefix-check/head.rs"] mod head;
#[path="checkpoint-prefix-check/experts.rs"] mod experts;


const COMPLETE:&[u8]=b"CHECKPOINT PREFIX STORED IDENTITY v1 PASS\n";
const PLANES:[&str;5]=["embedding.f32","mhc-normalized.f32","coefficients.f32","collapsed.f32","input-norm.f32"];
const SCOPE:&[u8]=b"stored_identity_candidate=true\nfull_layer=false\ntransport=false\noutput_token_ids=false\nruntime_load_policy=unqualified\n";
const META:[&str;4]=["IDENTITY","SCOPE","INPUT_IDS","PLANE_WORDS"];
fn read(path:&Path)->Result<Vec<u8>> {fs::read(path).map_err(|e|e.to_string())}
fn write(path:&Path,bytes:&[u8])->Result<()> {
    fs::OpenOptions::new().write(true).create_new(true).open(path)
        .and_then(|mut f|f.write_all(bytes)).map_err(|e|e.to_string())
}
fn equal(name:&str,a:&[u8],b:&[u8])->Result<()> {
    if a!=b {return Err(format!("{name}: bytes differ actual={} expected={} first={:?}",a.len(),b.len(),a.iter().zip(b).position(|(a,b)|a!=b)));}Ok(())
}
fn floats(values:&[f32])->Result<Vec<u8>> {
    if values.iter().any(|v|!v.is_finite()) {return Err("prefix contains a nonfinite value".into());}
    Ok(values.iter().flat_map(|v|v.to_le_bytes()).collect())
}
fn prompt(path:&Path)->Result<Vec<u32>> {
    let bytes=read_bounded(path,64<<10).map_err(|e|e.to_string())?;
    let text=std::str::from_utf8(&bytes).map_err(|e|e.to_string())?;
    let ids=text.split_whitespace().map(|s|s.parse().map_err(|_|"prompt contains an invalid token ID".into())).collect::<Result<Vec<u32>>>()?;
    if ids.is_empty() || ids.len()>4096 {return Err("prefix prompt must contain between 1 and 4096 IDs".into());}Ok(ids)
}
fn checkpoint(root:&Path)->Result<CheckpointAdmission> {
    let config=read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|e|e.to_string())?;
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&config).map_err(|e|e.to_string())?
        else {return Err("prefix probe requires a deferred text description".into());};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|e|e.to_string())?;
    d.open_headers(root,&index).map_err(|e|e.to_string())
}
struct Prefix<'a> {checkpoint:&'a CheckpointAdmission,block:ResidualBlockOperands,embedding:String,width:usize,next:String}
impl<'a> Prefix<'a> {
    fn new(checkpoint:&'a CheckpointAdmission,t:Topology)->Result<Self> {
        t.validate().map_err(|e|e.to_string())?;
        let d=checkpoint.description();
        let state=StatePlan::new(d,t,1,isize::MAX as usize).map_err(|e|e.to_string())?;
        let boundary=LayerBoundary::new(d,checkpoint.headers(),&state,0,1,t,usize::MAX)?;
        if boundary.require_executable().is_ok() {return Err("prefix probe must not admit a complete layer".into());}
        let step=boundary.program().steps.first().ok_or("empty prefix program")?;
        if step.kind!=Kind::MhcPrepare || step.id!="attn.prepare" {return Err("first layer has no attention mHC prefix".into());}
        let block=boundary.prepare_residual_block_checkpoint(0,checkpoint)?;
        let next=boundary.program().steps.get(3).ok_or("prefix has no following producer")?;
        let next=format!("{} {:?} {:?}",next.id,next.kind,next.weights.iter().map(|w|(&w.name,w.storage)).collect::<Vec<_>>());
        let embedding="model.language_model.embed_tokens.weight".to_owned();
        let view=checkpoint.headers().weights().get(&embedding).ok_or("missing embedding")?;
        if view.logical().storage!=WeightStorage::Plain || view.logical().shape!=[d.vocab,d.hidden]
            || view.components().len()!=1 || view.components()[0].dtype!=Dtype::Bf16 {
            return Err("prefix requires an admitted plain BF16 embedding".into());
        }
        Ok(Self {checkpoint,block,embedding,width:usize::try_from(d.hidden).map_err(|_|"prefix width exceeds address space")?,next})
    }
    fn row(&self,token:u32)->Result<Vec<u8>> {
        if u64::from(token)>=self.checkpoint.description().vocab {return Err("input token exceeds the vocabulary".into());}
        let count=self.width.checked_mul(2).ok_or("embedding row extent overflow")?;
        let offset=u64::from(token).checked_mul(count as u64).ok_or("embedding row offset overflow")?;
        let mut bytes=vec![0;count];
        self.checkpoint.component_file(&self.embedding).map_err(|e|e.to_string())?
            .read_range(&self.embedding,offset,&mut bytes).map_err(|e|e.to_string())?;
        floats(&oracle::embedding(&bytes))?;Ok(bytes)
    }
    fn expected(&self,token:u32)->Result<[Vec<f32>;5]> {oracle::prefix(&self.row(token)?,&self.block)}
    fn identity(&self,root:&Path,ids:&[u32],parts:u32)->Result<Vec<u8>> {
        let mut text=format!("checkpoint-prefix-v1\nsource={}\nparts={parts}\ninput_ids={ids:?}\nnext={}\n",
            include_str!("../../../../REVISION").trim(),self.next);
        for name in ["config.json","model.safetensors.index.json"] {
            text.push_str(&format!("{name}={}\n",sha256::digest_reader(fs::File::open(root.join(name)).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?));
        }
        // Pin the consumed rows and parameter bits. Do not hash unused checkpoint payloads.
        for (i,&id) in ids.iter().enumerate() {text.push_str(&format!("embedding[{i}] token={id} sha256={}\n",sha256::digest(&self.row(id)?)));}
        let m=self.block.mhc();let n=self.block.norm();
        text.push_str(&format!("mhc_shape={:?}\nmhc_stored={:?}\nnorm_shape={:?}\nnorm_stored={:?}\n",m.shape(),m.stored_dtypes(),n.shape(),n.stored_dtype()));
        for (name,values) in [("fn",m.weights()),("base",m.base()),("scale",m.scale()),("norm",n.weight())] {
            text.push_str(&format!("{name}={}\n",sha256::digest(&floats(values)?)));
        }Ok(text.into_bytes())
    }
}
fn plane_files(ids:&[u32])->Vec<String> {(0..ids.len()).flat_map(|i|PLANES.map(|name|format!("{i}-{name}"))).collect()}
fn seal(out:&Path,ids:&[u32])->Result<()> {
    let mut manifest=String::new();
    for name in META.map(str::to_owned).into_iter().chain(plane_files(ids)) {
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(&name))?)));
    }
    write(&out.join("OUTPUT.sha256"),manifest.as_bytes())?;write(&out.join("COMPLETE"),COMPLETE)
}
fn complete(out:&Path)->Result<Vec<u32>> {
    equal("completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;
    equal("scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let ids=prompt(&out.join("INPUT_IDS"))?;
    let shape=read_bounded(&out.join("PLANE_WORDS"),1024).map_err(|e|e.to_string())?;
    let shape=std::str::from_utf8(&shape).map_err(|e|e.to_string())?.split_whitespace()
        .map(|s|s.parse::<usize>().map_err(|_|"invalid plane extent".to_owned())).collect::<Result<Vec<_>>>()?;
    if shape.len()!=PLANES.len() || shape.iter().any(|&n|n==0 || n>isize::MAX as usize/4) {return Err("invalid plane extents".into());}
    for i in 0..ids.len() {for (name,&words) in PLANES.into_iter().zip(&shape) {
        let bytes=read(&out.join(format!("{i}-{name}")))?;
        if bytes.len()!=words*4 || bytes.as_chunks::<4>().0.iter().any(|b|!f32::from_le_bytes(*b).is_finite()) {
            return Err("prefix plane has an invalid extent or value".into());
        }
    }}
    let mut manifest=String::new();
    for name in META.map(str::to_owned).into_iter().chain(plane_files(&ids)) {
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(&name))?)));
    }
    equal("manifest",&read(&out.join("OUTPUT.sha256"))?,manifest.as_bytes())?;Ok(ids)
}
fn compare(a:&Path,b:&Path)->Result<()> {
    let ids=complete(a)?;let other=complete(b)?;
    if ids!=other {return Err("prefix input IDs differ".into());}
    for name in META.map(str::to_owned).into_iter().chain(plane_files(&ids)) {
        equal(&name,&read(&a.join(&name))?,&read(&b.join(&name))?)?;
    }
    println!("CHECKPOINT PREFIX COMPARE PASS tokens={} output_token_ids=false transport=false",ids.len());Ok(())
}
fn setup(out:&Path,p:&Prefix<'_>,root:&Path,ids:&[u32],t:Topology)->Result<()> {
    write(&out.join("IDENTITY"),&p.identity(root,ids,t.parts)?)?;
    write(&out.join("INPUT_IDS"),ids.iter().map(|id|format!("{id}\n")).collect::<String>().as_bytes())?;
    write(&out.join("SCOPE"),SCOPE)?;
    let s=p.block.mhc().shape();
    write(&out.join("PLANE_WORDS"),format!("{} {} {} {} {}\n",s.wide(),s.wide(),s.coefficients(),p.width,p.width).as_bytes())?;
    write(&out.join("RANK"),format!("rank={} parts={} replicated_prefix=true\n",t.rank,t.parts).as_bytes())
}
fn host(root:&Path,input:&Path,out:&Path,t:Topology)->Result<()> {
    let ids=prompt(input)?;let checkpoint=checkpoint(root)?;let p=Prefix::new(&checkpoint,t)?;
    setup(out,&p,root,&ids,t)?;
    for (i,&id) in ids.iter().enumerate() {for (name,values) in PLANES.into_iter().zip(p.expected(id)?) {
        write(&out.join(format!("{i}-{name}")),&floats(&values)?)?;
    }}
    seal(out,&ids)?;println!("CHECKPOINT PREFIX HOST PASS rank={} input_ids={ids:?} full_layer=false",t.rank);Ok(())
}
#[cfg(target_os="linux")]
fn device(root:&Path,input:&Path,expected:&Path,out:&Path,t:Topology)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    let ids=prompt(input)?;
    if complete(expected)?!=ids {return Err("expected prefix input IDs differ".into());}
    let checkpoint=checkpoint(root)?;let p=Prefix::new(&checkpoint,t)?;
    equal("operand identity",&p.identity(root,&ids,t.parts)?,&read(&expected.join("IDENTITY"))?)?;
    setup(out,&p,root,&ids,t)?;
    let budget=oracle::budget(&p.block)?;
    with_backend(0,layer_ptx()?,budget.0,|b| {
        let available=b.available_memory_bytes()?;
        if budget.1>available {return Err(format!("prefix needs {} bytes; available {available} bytes",budget.1));}
        println!("PREFIX BUDGET device={} required={} available={available} estimated=true",budget.0,budget.1);
        let resident=p.block.upload(b)?;
        for (i,&id) in ids.iter().enumerate() {
            let embedded=oracle::device_embedding(b,&p,id)?;
            let embedding=b.read_vector_values(&embedded)?;
            let pass=resident.begin(b.round_bf16_lanes(&embedded)?)?;
            let actual=[embedding,b.read_vector_values(pass.mhc().normalized())?,b.read_vector_values(pass.mhc().coefficients())?,
                b.read_vector_values(pass.mhc().collapsed().vector())?,b.read_vector_values(pass.normalized().vector())?];
            for (name,values) in PLANES.into_iter().zip(actual) {
                let name=format!("{i}-{name}");let bytes=floats(&values)?;
                write(&out.join(&name),&bytes)?;
                equal(&name,&bytes,&read(&expected.join(&name))?)?;
            }
            println!("PREFIX rank={} input_token={id} position={i} output_token_ids=false",t.rank);
        }
        if let Some(error)=b.fatal_error() {return Err(error);}Ok(())
    })?;
    seal(out,&ids)?;
    println!("CHECKPOINT PREFIX DEVICE PASS rank={} input_ids={ids:?} bytes_sha256={} full_layer=false transport=false",
        t.rank,sha256::digest(&read(&out.join("OUTPUT.sha256"))?));Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path,_:Topology)->Result<()> {Err("prefix device cell requires Linux and CUDA".into())}
fn child(args:&[&Path],log:&Path)->Result<()> {
    let file=fs::OpenOptions::new().write(true).create_new(true).open(log).map_err(|e|e.to_string())?;
    let status=std::process::Command::new(std::env::current_exe().map_err(|e|e.to_string())?).args(args)
        .stdout(file.try_clone().map_err(|e|e.to_string())?).stderr(file).status().map_err(|e|e.to_string())?;
    if !status.success() {return Err(format!("prefix child failed; retain {}",log.display()));}Ok(())
}
fn suite(root:&Path,input:&Path,out:&Path,t:Topology)->Result<()> {
    let host_out=out.join("host");let rank=t.rank.to_string();let parts=t.parts.to_string();
    let rank=Path::new(&rank);let parts=Path::new(&parts);
    child(&[Path::new("host"),root,input,&host_out,rank,parts],&out.join("host.log"))?;
    for run in 1..=3 {
        let result=out.join(format!("run{run}"));
        child(&[Path::new("device"),root,input,&host_out,&result,rank,parts],&out.join(format!("run{run}.log")))?;
        compare(&host_out,&result)?;compare(&out.join("run1"),&result)?;
        println!("PREFIX REPEAT rank={} run={run} input_ids={:?} bytes_sha256={}",t.rank,prompt(input)?,sha256::digest(&read(&result.join("OUTPUT.sha256"))?));
    }
    write(&out.join("COMPLETE"),b"CHECKPOINT PREFIX SUITE PASS\n")?;
    println!("CHECKPOINT PREFIX SUITE PASS rank={} fresh_device_processes=3 output_token_ids=false",t.rank);Ok(())
}
fn topology(rank:&str,parts:&str)->Result<Topology> {
    let t=Topology {rank:rank.parse().map_err(|_|"invalid rank")?,parts:parts.parse().map_err(|_|"invalid parts")?};
    t.validate().map_err(|e|e.to_string())?;Ok(t)
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let result=(||match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["projection",rest @ ..]=>projection::run(rest),
        ["attention-projections",rest @ ..]=>attention::run(rest),
        ["layer-step",rest @ ..]=>layer::run(rest),
        ["tokens",rest @ ..]=>tokens::run(rest),
        ["head-path",rest @ ..]=>head::run(rest),
        ["expert-path",rest @ ..]=>experts::run(rest),

        ["host",root,input,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;host(Path::new(root),Path::new(input),Path::new(out),t)},
        ["device",root,input,expected,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;device(Path::new(root),Path::new(input),Path::new(expected),Path::new(out),t)},
        ["suite",root,input,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;suite(Path::new(root),Path::new(input),Path::new(out),t)},
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check host MODEL INPUT_IDS NEW_OUT RANK PARTS | device MODEL INPUT_IDS EXPECTED NEW_OUT RANK PARTS | suite MODEL INPUT_IDS NEW_OUT RANK PARTS | compare LEFT RIGHT | projection {host|device|suite|compare|compare-suite} | attention-projections {host|device|suite|compare|compare-suite}".into()),
    })();
    if let Err(e)=result {eprintln!("CHECKPOINT PREFIX FAIL {e}");std::process::exit(1);}
}

#[cfg(test)]
#[path="checkpoint-prefix-check/tests.rs"] mod tests;
