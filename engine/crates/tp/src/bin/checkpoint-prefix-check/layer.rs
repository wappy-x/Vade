//! Real checkpoint KDA/dense layer. Compare every produced plane as exact bytes.
use super::{checkpoint,prompt,read,write,equal,floats,sha256,child,Prefix};
use spark_tp::{Result,causal_layer::{Layer}};
use engine_model::plan::layer::Topology;
use std::{fs,path::{Path,PathBuf}};
const COMPLETE:&[u8]=b"CHECKPOINT CAUSAL LAYER v1 PASS\n";
const SCOPE:&[u8]=b"layer=0\nactivation=dynamic_mxfp8_block32_bf16_source\naccumulator=fp32_fixed_order\nprojection_epilogue=rn_even_bf16\nreference=causal_recipe_v1\nlogical_f32_stack_equal=unproved\nfull_layer=true\ntransport=false\noutput_token_ids=false\n";
fn bounded(path:&Path)->Result<Vec<u32>> {let ids=prompt(path)?;if ids.len()>64 {return Err("causal layer permits at most 64 input IDs".into());}Ok(ids)}
fn setup(root:&Path,input:&Path,out:&Path)->Result<(engine_model::plan::admission::CheckpointAdmission,Vec<u32>)> {
    let ids=bounded(input)?;let c=checkpoint(root)?;let p=Prefix::new(&c,Topology {rank:0,parts:1})?;
    let mut identity=p.identity(root,&ids,1)?;identity.extend(Layer::new(&c,0)?.parameter_record(sha256::digest)?.bytes());
    write(&out.join("IDENTITY"),&identity)?;write(&out.join("SCOPE"),SCOPE)?;
    write(&out.join("INPUT_IDS"),ids.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;Ok((c,ids))
}
fn seal(out:&Path,names:&[String])->Result<()> {
    write(&out.join("PLANES"),names.join("\n").as_bytes())?;
    let mut manifest=String::new();for name in ["IDENTITY","SCOPE","INPUT_IDS","PLANES"].into_iter().chain(names.iter().map(String::as_str)) {
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(name))?)));
    }write(&out.join("OUTPUT.sha256"),manifest.as_bytes())?;write(&out.join("COMPLETE"),COMPLETE)
}
fn host(root:&Path,input:&Path,out:&Path)->Result<()> {
    let (c,ids)=setup(root,input,out)?;let p=Prefix::new(&c,Topology {rank:0,parts:1})?;let l=Layer::new(&c,0)?;let mut state=l.zero_state()?;let mut names=Vec::new();
    for (i,&id) in ids.iter().enumerate() {let residual=super::oracle::embedding(&p.row(id)?).repeat(c.description().residual.streams as usize);
        let (values,next)=l.reference(&residual,&state)?;
        for (name,v) in values {let name=format!("{i}-{name}.f32");write(&out.join(&name),&floats(&v)?)?;names.push(name);}state=next;
    }
    names.sort();seal(out,&names)?;println!("CHECKPOINT CAUSAL LAYER HOST PASS input_tokens={} full_layer=true output_token_ids=false",ids.len());Ok(())
}
fn saved(out:&Path)->Result<(Vec<String>,Vec<u32>)> {
    equal("causal layer completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;equal("causal layer scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let text=String::from_utf8(read(&out.join("PLANES"))?).map_err(|e|e.to_string())?;
    let names=text.lines().map(str::to_owned).collect::<Vec<_>>();let ids=bounded(&out.join("INPUT_IDS"))?;
    if names.is_empty() || names.windows(2).any(|p|p[0]>=p[1]) || names.iter().any(|n|!n.ends_with(".f32") || n.contains('/') || n.contains('\\') || n.contains("..")) {return Err("invalid causal layer plane list".into());}
    let mut manifest=String::new();for name in ["IDENTITY","SCOPE","INPUT_IDS","PLANES"].into_iter().chain(names.iter().map(String::as_str)) {
        let data=read(&out.join(name))?;
        if name.ends_with(".f32") && (data.len()%4!=0 || data.as_chunks::<4>().0.iter().any(|b|!f32::from_le_bytes(*b).is_finite())) {return Err("invalid causal layer plane bytes".into());}
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&data)));
    }
    equal("causal layer manifest",&read(&out.join("OUTPUT.sha256"))?,manifest.as_bytes())?;Ok((names,ids))
}
fn compare(a:&Path,b:&Path)->Result<()> {
    let (names,ids)=saved(a)?;let other=saved(b)?;if names!=other.0 || ids!=other.1 {return Err("causal layer input or plane names differ".into());}
    for name in ["IDENTITY","SCOPE","INPUT_IDS"].into_iter().chain(names.iter().map(String::as_str)) {equal(name,&read(&a.join(name))?,&read(&b.join(name))?)?;}
    println!("CHECKPOINT CAUSAL LAYER COMPARE PASS input_tokens={} planes={} full_layer=true output_token_ids=false",ids.len(),names.len());Ok(())
}
#[cfg(target_os="linux")]
fn device(root:&Path,input:&Path,expected:&Path,out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    let (c,ids)=setup(root,input,out)?;let (expected_names,expected_ids)=saved(expected)?;
    if ids!=expected_ids {return Err("causal layer expected IDs differ".into());}
    for name in ["IDENTITY","SCOPE","INPUT_IDS"] {equal(name,&read(&out.join(name))?,&read(&expected.join(name))?)?;}
    let layer=Layer::new(&c,0)?;let prefix=Prefix::new(&c,Topology {rank:0,parts:1})?;
    let mut names=Vec::new();with_backend(0,layer_ptx()?,usize::MAX,|b| {
        let resident=layer.upload(b)?;let mut state=resident.zero_state()?;
        for (i,&id) in ids.iter().enumerate() {
            let embedded=super::oracle::device_embedding(b,&prefix,id)?;let residual=b.round_bf16_lanes(&embedded)?;
            for (name,v) in [("residual",residual.vector()),("kda.history",state.history()),("kda.state",state.recurrent())] {
                let name=format!("{i}-{name}.f32");let data=floats(&b.read_vector_values(v)?)?;write(&out.join(&name),&data)?;
                equal(&name,&data,&read(&expected.join(&name))?)?;names.push(name);
            }
            let (_,next)=resident.step(&residual,&state,|name,v| {
                let name=format!("{i}-{name}.f32");let data=floats(&b.read_vector_values(v)?)?;write(&out.join(&name),&data)?;
                equal(&name,&data,&read(&expected.join(&name))?)?;names.push(name);Ok(())
            })?;state=next;
        }Ok(())
    })?;
    names.sort();if names!=expected_names {return Err("resident causal layer omitted a reference plane".into());}
    seal(out,&names)?;compare(expected,out)?;println!("CHECKPOINT CAUSAL LAYER DEVICE PASS full_layer=true transport=false output_token_ids=false");Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path)->Result<()> {Err("causal layer device cell requires Linux and CUDA".into())}
fn suite(root:&Path,input:&Path,out:&Path)->Result<()> {
    bounded(input)?;let host=out.join("host");child(&[Path::new("layer-step"),Path::new("host"),root,input,&host],&out.join("host.log"))?;
    for run in 1..=3 {let output=out.join(format!("run{run}"));child(&[Path::new("layer-step"),Path::new("device"),root,input,&host,&output],&out.join(format!("run{run}.log")))?;
        compare(&host,&output)?;compare(&out.join("run1"),&output)?;
    }write(&out.join("COMPLETE"),COMPLETE)?;println!("CHECKPOINT CAUSAL LAYER SUITE PASS repeats=3 output_token_ids=false");Ok(())
}
fn compare_suite(a:&Path,b:&Path)->Result<()> {
    for p in [a,b] {equal("causal suite completion",&read(&p.join("COMPLETE"))?,COMPLETE)?;}
    compare(&a.join("host"),&b.join("host"))?;
    for run in 1..=3 {compare(&a.join(format!("run{run}")),&b.join(format!("run{run}")))?;}
    println!("CHECKPOINT CAUSAL LAYER CROSS SPARK PASS repeats=3 transport=false output_token_ids=false");Ok(())
}
pub fn run(args:&[&str])->Result<()> {
    let new=|s:&str|->Result<PathBuf> {fs::create_dir(s).map_err(|e|e.to_string())?;Ok(PathBuf::from(s))};
    match args {
        ["host",root,input,out]=>host(Path::new(root),Path::new(input),&new(out)?),
        ["device",root,input,expected,out]=>device(Path::new(root),Path::new(input),Path::new(expected),&new(out)?),
        ["suite",root,input,out]=>suite(Path::new(root),Path::new(input),&new(out)?),
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),["compare-suite",a,b]=>compare_suite(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check layer-step host MODEL INPUT_IDS NEW_OUT | device MODEL INPUT_IDS EXPECTED NEW_OUT | suite MODEL INPUT_IDS NEW_OUT | compare LEFT RIGHT | compare-suite LEFT RIGHT".into()),
    }
}
