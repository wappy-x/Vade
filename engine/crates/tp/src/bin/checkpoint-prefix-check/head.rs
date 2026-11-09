//! Saved final residual to logits and token feedback. Exact bytes on each plane.
use super::{checkpoint,read,write,equal,floats,sha256,child};
use spark_tp::{Result,head_path::{Head,HeadReference}};
use std::{fs,path::{Path,PathBuf}};
const COMPLETE:&[u8]=b"GLM HEAD-PATH v1 PASS\n";
const SCOPE:&[u8]=b"reference=head_recipe_v1\ninput=layer_zero_output_stand_in\nactivation=dynamic_mxfp8_block32_bf16_source_or_bf16\nprojection_epilogue=rn_even_bf16\nargmax_ties=lowest_token_id\nfull_decoder=false\ntransport=false\n";
const PLANES:[&str;6]=["mean.f32","normalized.f32","accumulator.f32","logits.f32","next-embedding.f32","token.u32"];
const META:[&str;3]=["IDENTITY","SCOPE","INPUT.f32"];
fn saved_input(dir:&Path)->Result<(String,Vec<u8>,Vec<u8>)> {
    let manifest=engine_model::safetensors::read_bounded(&dir.join("OUTPUT.sha256"),1024*1024).map_err(|e|e.to_string())?;
    let text=std::str::from_utf8(&manifest).map_err(|e|e.to_string())?;let mut last=None;
    for line in text.lines() {
        let fields=line.split_whitespace().collect::<Vec<_>>();if fields.len()!=2 {return Err("invalid saved layer manifest".into());}
        if let Some(index)=fields[1].strip_suffix("-output.f32") {
            let position=index.parse::<usize>().map_err(|_|"invalid saved final position")?;
            if fields[0].len()!=64 || !fields[0].bytes().all(|b|b.is_ascii_hexdigit()) {return Err("invalid saved input hash".into());}
            if last.as_ref().is_none_or(|(p,_,_)|position>*p) {last=Some((position,fields[1].to_owned(),fields[0].to_owned()));}
        }
    }
    let (_,name,hash)=last.ok_or("saved layer has no final output plane")?;
    let data=engine_model::safetensors::read_bounded(&dir.join(&name),16*1024*1024).map_err(|e|e.to_string())?;
    need_hash(&data,&hash)?;Ok((name,data,manifest))
}
fn need_hash(bytes:&[u8],expected:&str)->Result<()> {
    if sha256::digest(bytes)!=expected {return Err("saved head input hash differs".into());}Ok(())
}
fn values(bytes:&[u8])->Result<Vec<f32>> {
    if bytes.is_empty() || bytes.len()%4!=0 {return Err("head input byte extent differs".into());}
    let v=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    spark_gpu_probe::mhc_boundary::check_bf16(&v)?;Ok(v)
}
fn setup(root:&Path,input:&Path,out:&Path)->Result<(engine_model::plan::admission::CheckpointAdmission,Vec<f32>)> {
    let c=checkpoint(root)?;let h=Head::new(&c)?;let (name,data,manifest)=saved_input(input)?;let v=values(&data)?;
    if v.len()!=h.shape().residual() {return Err("saved head residual extent differs from checkpoint".into());}
    let mut identity=format!("GLM HEAD-PATH\nsource={}\ninput_plane={name}\ninput_sha256={}\nsource_manifest_sha256={}\n",
        include_str!("../../../../../REVISION").trim(),sha256::digest(&data),sha256::digest(&manifest));
    for file in ["config.json","model.safetensors.index.json","hf_quant_config.json"] {
        let path=root.join(file);if file=="hf_quant_config.json" && !path.try_exists().map_err(|e|e.to_string())? {continue;}
        let limit=if file=="model.safetensors.index.json" {engine_model::shapes::HEADER_BYTES} else {engine_model::shapes::CONFIG_BYTES};
        let bytes=engine_model::safetensors::read_bounded(&path,limit).map_err(|e|e.to_string())?;
        if file=="hf_quant_config.json" {
            let side=engine_model::plan::storage::StorageMap::from_json(&bytes).map_err(|e|e.to_string())?;
            if side!=c.description().storage {return Err("quantization sidecar differs from config storage map".into());}
        }
        identity.push_str(&format!("{file}={}\n",sha256::digest(&bytes)));
    }
    identity.push_str(&h.parameter_record(sha256::digest)?);
    write(&out.join("IDENTITY"),identity.as_bytes())?;write(&out.join("SCOPE"),SCOPE)?;write(&out.join("INPUT.f32"),&data)?;
    Ok((c,v))
}
fn outputs(out:&Path,r:&HeadReference)->Result<()> {
    for (name,v) in PLANES[..5].iter().zip([&r.mean,&r.normalized,&r.accumulator,&r.logits,&r.next_embedding]) {write(&out.join(name),&floats(v)?)?;}
    write(&out.join("token.u32"),&r.token.to_le_bytes())
}
fn seal(out:&Path)->Result<()> {
    let mut manifest=String::new();for name in META.into_iter().chain(PLANES) {manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(name))?)));}
    write(&out.join("OUTPUT.sha256"),manifest.as_bytes())?;write(&out.join("COMPLETE"),COMPLETE)
}
fn saved(out:&Path)->Result<()> {
    equal("head completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;equal("head scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let mut manifest=String::new();for name in META.into_iter().chain(PLANES) {
        let bytes=read(&out.join(name))?;
        if name.ends_with(".f32") && (bytes.is_empty() || bytes.len()%4!=0 || bytes.chunks_exact(4).any(|b|!f32::from_le_bytes(b.try_into().unwrap()).is_finite())) {return Err("invalid saved head plane".into());}
        if name=="token.u32" && bytes.len()!=4 {return Err("invalid saved token".into());}
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&bytes)));
    }
    equal("head manifest",&read(&out.join("OUTPUT.sha256"))?,manifest.as_bytes())?;
    let logits=read(&out.join("logits.f32"))?;let logits=logits.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    equal("saved greedy token",&read(&out.join("token.u32"))?,&spark_gpu_probe::text_edges::greedy(&logits)?.to_le_bytes())
}
fn compare(a:&Path,b:&Path)->Result<()> {
    saved(a)?;saved(b)?;for name in META.into_iter().chain(PLANES) {equal(name,&read(&a.join(name))?,&read(&b.join(name))?)?;}
    println!("GLM HEAD-PATH COMPARE PASS logits_bytes=true token=true next_embedding=true");Ok(())
}
fn host(root:&Path,input:&Path,out:&Path)->Result<()> {
    let (c,v)=setup(root,input,out)?;let head=Head::new(&c)?;let r=head.reference(&v)?;
    println!("GLM HEAD-PATH HOST token={} vocab={} storage={:?}",r.token,r.logits.len(),head.storage());
    outputs(out,&r)?;seal(out)?;saved(out)
}
#[cfg(target_os="linux")]
fn device(root:&Path,input:&Path,expected:&Path,out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    saved(expected)?;let (c,v)=setup(root,input,out)?;
    for name in META {equal(name,&read(&out.join(name))?,&read(&expected.join(name))?)?;}
    let head=Head::new(&c)?;
    with_backend(0,layer_ptx()?,usize::MAX,|b| {
        let resident=head.upload(b)?;let input=b.upload_vector(&v)?;b.synchronize()?;let pass=resident.step(&input)?;
        for (name,vector) in PLANES[..5].iter().zip([&pass.mean,&pass.normalized,&pass.accumulator,&pass.logits,&pass.next_embedding]) {
            let bytes=floats(&b.read_vector_values(vector)?)?;write(&out.join(name),&bytes)?;equal(name,&bytes,&read(&expected.join(name))?)?;
        }
        write(&out.join("token.u32"),&pass.token.to_le_bytes())?;
        println!("GLM HEAD-PATH DEVICE token={} vocab={} storage={:?}",pass.token,head.shape().vocab(),head.storage());Ok(())
    })?;seal(out)?;compare(expected,out)
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path)->Result<()> {Err("head device cell requires Linux and CUDA".into())}
fn suite(root:&Path,input:&Path,out:&Path)->Result<()> {
    let host=out.join("host");child(&[Path::new("head-path"),Path::new("host"),root,input,&host],&out.join("host.log"))?;
    for run in 1..=3 {let result=out.join(format!("run{run}"));child(&[Path::new("head-path"),Path::new("device"),root,input,&host,&result],&out.join(format!("run{run}.log")))?;
        compare(&host,&result)?;compare(&out.join("run1"),&result)?;
    }write(&out.join("COMPLETE"),COMPLETE)?;println!("GLM HEAD-PATH SUITE PASS repeats=3 full_decoder=false transport=false");Ok(())
}
fn compare_suite(a:&Path,b:&Path)->Result<()> {
    for dir in [a,b] {equal("head suite completion",&read(&dir.join("COMPLETE"))?,COMPLETE)?;}
    compare(&a.join("host"),&b.join("host"))?;for run in 1..=3 {compare(&a.join(format!("run{run}")),&b.join(format!("run{run}")))?;}
    println!("GLM HEAD-PATH CROSS SPARK PASS repeats=3 full_decoder=false transport=false");Ok(())
}
pub fn run(args:&[&str])->Result<()> {
    let new=|s:&str|->Result<PathBuf> {fs::create_dir(s).map_err(|e|e.to_string())?;Ok(PathBuf::from(s))};
    match args {
        ["host",root,input,out]=>host(Path::new(root),Path::new(input),&new(out)?),
        ["device",root,input,expected,out]=>device(Path::new(root),Path::new(input),Path::new(expected),&new(out)?),
        ["suite",root,input,out]=>suite(Path::new(root),Path::new(input),&new(out)?),
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),["compare-suite",a,b]=>compare_suite(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check head-path host MODEL SAVED_LAYER NEW_OUT | device MODEL SAVED_LAYER EXPECTED NEW_OUT | suite MODEL SAVED_LAYER NEW_OUT | compare LEFT RIGHT | compare-suite LEFT RIGHT".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {fn new()->Self {static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let p=std::env::temp_dir().join(format!("vade-head-probe-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
        fs::create_dir(&p).unwrap();Self(p)}}
    impl Drop for Temp {fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();}}
    #[test]
    fn selects_numeric_final_position_and_requires_original_bytes() {
        let dir=Temp::new();let mut manifest=String::new();for position in [9,10,2] {
            let bytes=floats(&[position as f32;32]).unwrap();let name=format!("{position}-output.f32");
            write(&dir.0.join(&name),&bytes).unwrap();manifest.push_str(&format!("{}  {name}\n",sha256::digest(&bytes)));
        }write(&dir.0.join("OUTPUT.sha256"),manifest.as_bytes()).unwrap();
        let input=saved_input(&dir.0).unwrap();assert_eq!(input.0,"10-output.f32");assert_eq!(values(&input.1).unwrap(),[10.0;32]);
        fs::write(dir.0.join("10-output.f32"),floats(&[1.0;32]).unwrap()).unwrap();assert!(saved_input(&dir.0).is_err());
        assert!(values(&[0]).is_err());assert!(values(&f32::NAN.to_le_bytes()).is_err());
    }
    #[test]
    fn saved_head_rejects_edited_logits_and_false_greedy_token() {
        let dir=Temp::new();write(&dir.0.join("IDENTITY"),b"fixture").unwrap();write(&dir.0.join("SCOPE"),SCOPE).unwrap();
        write(&dir.0.join("INPUT.f32"),&floats(&[1.0;32]).unwrap()).unwrap();
        let r=HeadReference {mean:vec![1.0;32],normalized:vec![1.0;32],accumulator:vec![3.0,3.0,2.0],logits:vec![3.0,3.0,2.0],next_embedding:vec![1.0;32],token:0};
        outputs(&dir.0,&r).unwrap();seal(&dir.0).unwrap();saved(&dir.0).unwrap();
        fs::write(dir.0.join("token.u32"),1u32.to_le_bytes()).unwrap();assert!(saved(&dir.0).is_err());
        let mut manifest=String::new();for name in META.into_iter().chain(PLANES) {manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&dir.0.join(name)).unwrap())));}
        fs::write(dir.0.join("OUTPUT.sha256"),manifest).unwrap();assert!(saved(&dir.0).unwrap_err().contains("greedy"));
    }
}
