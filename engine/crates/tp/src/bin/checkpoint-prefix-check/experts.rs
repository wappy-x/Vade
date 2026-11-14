//! Isolated expert-path proof from saved layer-zero hidden feature rows.
use super::{checkpoint,read,write,equal,floats,sha256,child};
use spark_tp::{Result,expert_path::{ExpertPath,Planes}};
use engine_model::{plan::admission::CheckpointAdmission,host_ops::router::Router,shapes::Ffn,safetensors::{Dtype,read_bounded}};
use std::{fs,path::{Path,PathBuf},collections::BTreeSet};
const COMPLETE:&[u8]=b"CHECKPOINT EXPERT PATH v1 PASS\n";
const SCOPE:&[u8]=b"reference=calibrated_expert_recipe_v1\nactivation=nvfp4_block16_e4m3_calibrated_global_bf16_source\nshared_activation=bf16\naccumulator=fp32_serial_feature_order\nepilogue=rn_even_bf16\nfold=ascending_expert_id_bf16_contribution_add_shared_last\nrouting=existing_host_router_replay\nfull_layer=false\ntransport=false\noutput_token_ids=false\n";
const TOKENS:usize=8;
fn values(bytes:&[u8])->Result<Vec<f32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {return Err("expert plane extent is invalid".into());}
    let v=bytes.as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect::<Vec<_>>();
    if v.iter().any(|v|!v.is_finite()) {return Err("expert plane has a nonfinite value".into());}Ok(v)
}
fn parameter(c:&CheckpointAdmission,name:&str)->Result<Vec<f32>> {
    let w=c.upload(name).map_err(|e|e.to_string())?;
    if w.storage!=engine_model::plan::storage::WeightStorage::Plain {return Err("probe router requires plain parameters".into());}
    match w.dtype {Dtype::F32=>values(&w.weight),Dtype::Bf16=>spark_gpu_probe::text_reference::embedding(&w.weight),_=>Err("invalid probe router parameter dtype".into())}
}
struct Input {rows:Vec<Vec<f32>>,ids:Vec<Vec<u32>>,weights:Vec<Vec<f32>>,identity:Vec<u8>}
fn setup(root:&Path,saved:&Path)->Result<(CheckpointAdmission,ExpertPath,Input)> {
    let c=checkpoint(root)?;let d=c.description();let layer=d.layers.iter().position(|l|matches!(l.ffn,Ffn::Experts(_))).ok_or("checkpoint has no expert layer")?;
    let Ffn::Experts(e)=d.layers[layer].ffn else {unreachable!()};
    equal("saved layer completion",&read(&saved.join("COMPLETE"))?,b"CHECKPOINT CAUSAL LAYER v1 PASS\n")?;
    let manifest=read(&saved.join("OUTPUT.sha256"))?;let manifest=std::str::from_utf8(&manifest).map_err(|e|e.to_string())?;
    let scope=read(&saved.join("SCOPE"))?;
    if !scope.starts_with(b"layer=0\n") {return Err("saved expert input must come from the layer-zero cell".into());}
    let router=Router::from_description(&d.routing,e).map_err(|e|e.to_string())?;
    let prefix=format!("model.language_model.layers.{layer}.mlp.gate");let matrix=parameter(&c,&format!("{prefix}.weight"))?;
    let bias=parameter(&c,&format!("{prefix}.e_score_correction_bias"))?;
    if matrix.len()!=e.count as usize*d.hidden as usize || bias.len()!=e.count as usize {return Err("probe router parameter extents differ".into());}
    let mut rows=Vec::new();let mut ids=Vec::new();let mut weights=Vec::new();
    let mut identity=format!("expert_layer={layer}\nsource_plane=layer0.ffn_output\nsource_manifest={}\nsource_identity={}\nconfig={}\nrouter_weight={}\nrouter_bias={}\n",
        sha256::digest(manifest.as_bytes()),sha256::digest(&read(&saved.join("IDENTITY"))?),sha256::digest(&read(&root.join("config.json"))?),
        sha256::digest(&floats(&matrix)?),sha256::digest(&floats(&bias)?));
    match fs::metadata(root.join("hf_quant_config.json")) {Ok(_)=>identity.push_str(&format!("hf_quant_config={}\n",sha256::digest(&read_bounded(&root.join("hf_quant_config.json"),1<<20).map_err(|e|e.to_string())?))),
        Err(e) if e.kind()==std::io::ErrorKind::NotFound=>identity.push_str("hf_quant_config=absent\n"),Err(e)=>return Err(e.to_string())}
    for i in 0..TOKENS {
        let name=format!("{i}-ffn_output.f32");let bytes=read(&saved.join(&name))?;
        let entry=format!("{}  {name}",sha256::digest(&bytes));if !manifest.lines().any(|l|l==entry) {return Err("saved expert input hash differs".into());}
        let row=values(&bytes)?;if row.len()!=d.hidden as usize {return Err("saved hidden input feature extent differs".into());}
        let logits=matrix.chunks_exact(row.len()).map(|w|w.iter().zip(&row).fold(0.0,|s,(&w,&x)|s+w*x)).collect::<Vec<_>>();
        let (route,weight)=router.route(&logits,&bias).map_err(|e|e.to_string())?;
        let route=route.into_iter().map(|v|v as u32).collect::<Vec<_>>();
        identity.push_str(&format!("token={i} input={} ids={route:?} weights={} logits={}\n",sha256::digest(&bytes),sha256::digest(&floats(&weight)?),sha256::digest(&floats(&logits)?)));
        rows.push(row);ids.push(route);weights.push(weight);
    }
    let union=ids.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();let path=ExpertPath::new(&c,layer,&union)?;
    identity.push_str(&path.parameter_record(sha256::digest));
    Ok((c,path,Input {rows,ids,weights,identity:identity.into_bytes()}))
}
fn metadata(out:&Path,input:&Input)->Result<()> {write(&out.join("IDENTITY"),&input.identity)?;write(&out.join("SCOPE"),SCOPE)}
fn plane(out:&Path,i:usize,name:&str,v:&[f32],names:&mut Vec<String>)->Result<()> {
    if name.ends_with(".quant") {
        use spark_gpu_probe::expert_ops::{e2m1,e4m3,nearest_e2m1,nearest_e4m3};
        if !v.len().is_multiple_of(17) || v.as_chunks::<17>().0.iter().any(|b|
            b[..16].iter().any(|&v|!v.is_finite() || e2m1(nearest_e2m1(v)).to_bits()!=v.to_bits())
            || !b[16].is_finite() || b[16].is_sign_negative() || e4m3(nearest_e4m3(b[16])).to_bits()!=b[16].to_bits()) {
            return Err("invalid expert quantizer decoded plane".into());
        }
        for (suffix,bytes) in [("e2m1",v.as_chunks::<17>().0.iter().flat_map(|b|b[..16].as_chunks::<2>().0.iter()
            .map(|p|nearest_e2m1(p[0])|nearest_e2m1(p[1])<<4)).collect::<Vec<_>>()),
            ("e4m3",v.as_chunks::<17>().0.iter().map(|b|nearest_e4m3(b[16])).collect::<Vec<_>>())] {
            let name=format!("{i}-{name}.{suffix}");write(&out.join(&name),&bytes)?;names.push(name);
        }
    }
    let name=format!("{i}-{name}.f32");write(&out.join(&name),&floats(v)?)?;names.push(name);Ok(())
}
fn seal(out:&Path,names:&mut [String])->Result<()> {
    names.sort();write(&out.join("PLANES"),names.join("\n").as_bytes())?;
    let mut manifest=String::new();for name in ["IDENTITY","SCOPE","PLANES"].into_iter().chain(names.iter().map(String::as_str)) {
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(name))?)));
    }write(&out.join("OUTPUT.sha256"),manifest.as_bytes())?;write(&out.join("COMPLETE"),COMPLETE)
}
fn saved(out:&Path)->Result<Vec<String>> {
    equal("expert completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;equal("expert scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let list=read(&out.join("PLANES"))?;let names=std::str::from_utf8(&list).map_err(|e|e.to_string())?.lines().map(str::to_owned).collect::<Vec<_>>();
    if names.is_empty() || names.windows(2).any(|p|p[0]>=p[1]) || names.iter().any(|n|n.contains('/') || n.contains('\\') || n.contains("..")
        || ![".f32",".e2m1",".e4m3"].iter().any(|suffix|n.ends_with(suffix))) {return Err("invalid expert plane manifest".into());}
    let mut manifest=String::new();for name in ["IDENTITY","SCOPE","PLANES"].into_iter().chain(names.iter().map(String::as_str)) {
        let data=read(&out.join(name))?;if name.ends_with(".f32") {values(&data)?;}
        manifest.push_str(&format!("{}  {name}\n",sha256::digest(&data)));
    }equal("expert manifest",&read(&out.join("OUTPUT.sha256"))?,manifest.as_bytes())?;Ok(names)
}
fn compare(a:&Path,b:&Path)->Result<()> {
    let names=saved(a)?;if names!=saved(b)? {return Err("expert plane names differ".into());}
    for name in ["IDENTITY","SCOPE"].into_iter().chain(names.iter().map(String::as_str)) {equal(name,&read(&a.join(name))?,&read(&b.join(name))?)?;}
    println!("CHECKPOINT EXPERT PATH COMPARE PASS input_tokens={TOKENS} planes={} full_layer=false",names.len());Ok(())
}
fn host(root:&Path,source:&Path,out:&Path)->Result<()> {
    let (_c,path,input)=setup(root,source)?;metadata(out,&input)?;let mut names=Vec::new();
    for i in 0..TOKENS {let planes:Planes=path.reference(&input.rows[i],&input.ids[i],&input.weights[i])?;
        for (name,v) in planes {plane(out,i,&name,&v,&mut names)?;}
    }seal(out,&mut names)?;println!("CHECKPOINT EXPERT PATH HOST PASS input_tokens={TOKENS}");Ok(())
}
#[cfg(target_os="linux")]
fn device(root:&Path,source:&Path,expected:&Path,out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    let (_c,path,input)=setup(root,source)?;metadata(out,&input)?;let expected_names=saved(expected)?;
    equal("expert checkpoint and route identity",&input.identity,&read(&expected.join("IDENTITY"))?)?;
    let mut names=Vec::new();with_backend(0,layer_ptx()?,usize::MAX,|b| {
        let resident=path.upload(b)?;
        for i in 0..TOKENS {let row=b.upload_vector(&input.rows[i])?;
            resident.apply(&row,&input.ids[i],&input.weights[i],|name,v| {
                let values=b.read_vector_values(v)?;plane(out,i,name,&values,&mut names)?;
                equal(name,&floats(&values)?,&read(&expected.join(format!("{i}-{name}.f32")))?)
            })?;
        }Ok(())
    })?;names.sort();if names!=expected_names {return Err("device expert path omitted a reference plane".into());}
    seal(out,&mut names)?;compare(expected,out)?;println!("CHECKPOINT EXPERT PATH DEVICE PASS transport=false output_token_ids=false");Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path)->Result<()> {Err("expert device cell requires Linux and CUDA".into())}
fn suite(root:&Path,source:&Path,out:&Path)->Result<()> {
    let host=out.join("host");child(&[Path::new("expert-path"),Path::new("host"),root,source,&host],&out.join("host.log"))?;
    for run in 1..=3 {let result=out.join(format!("run{run}"));child(&[Path::new("expert-path"),Path::new("device"),root,source,&host,&result],&out.join(format!("run{run}.log")))?;
        compare(&host,&result)?;compare(&out.join("run1"),&result)?;
    }write(&out.join("COMPLETE"),COMPLETE)?;println!("CHECKPOINT EXPERT PATH SUITE PASS input_tokens={TOKENS} repeats=3");Ok(())
}
fn compare_suite(a:&Path,b:&Path)->Result<()> {
    for p in [a,b] {equal("expert suite completion",&read(&p.join("COMPLETE"))?,COMPLETE)?;}
    compare(&a.join("host"),&b.join("host"))?;for run in 1..=3 {compare(&a.join(format!("run{run}")),&b.join(format!("run{run}")))?;}
    println!("CHECKPOINT EXPERT PATH CROSS SPARK PASS input_tokens={TOKENS} repeats=3 transport=false output_token_ids=false");Ok(())
}
pub fn run(args:&[&str])->Result<()> {
    let new=|s:&str|->Result<PathBuf> {fs::create_dir(s).map_err(|e|e.to_string())?;Ok(PathBuf::from(s))};
    match args {
        ["host",root,source,out]=>host(Path::new(root),Path::new(source),&new(out)?),
        ["device",root,source,expected,out]=>device(Path::new(root),Path::new(source),Path::new(expected),&new(out)?),
        ["suite",root,source,out]=>suite(Path::new(root),Path::new(source),&new(out)?),
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),["compare-suite",a,b]=>compare_suite(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check expert-path {host|device|suite|compare|compare-suite} MODEL SAVED_LAYER0 NEW_OUT".into()),
    }
}
