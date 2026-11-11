//! Default-off single-device kernel screening. No model execution or transport.
#![forbid(unsafe_code)]
use std::{collections::{BTreeMap,BTreeSet},fs,path::{Path,PathBuf},time::{Duration,Instant}};
use engine_model::plan::{admission::CheckpointAdmission,storage::WeightStorage,layer::Topology};
type Result<T>=std::result::Result<T,String>;
#[cfg(target_os="linux")]
#[path="kernel-bench/device.rs"] mod device;
#[path="kernel-bench/sweep.rs"] mod sweep;
#[path="../../../../tools/export/src/sha256.rs"] mod sha256;

#[derive(Clone,Debug)]
struct Selection {name:String,kind:&'static str,rows:usize,cols:usize,bytes:u64,occurrences:usize}
fn open(root:&Path)->Result<CheckpointAdmission> {
    use engine_model::{plan::ConfigPlan,safetensors::read_bounded,shapes::{CONFIG_BYTES,HEADER_BYTES}};
    let config=read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|error|error.to_string())?;
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&config).map_err(|error|error.to_string())? else {return Err("bench requires a deferred checkpoint recipe".into());};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|error|error.to_string())?;
    description.open_headers(root,&index).map_err(|error|error.to_string())
}
fn select(checkpoint:&CheckpointAdmission,parts:u32,per_kind:usize,limit:u64)->Result<Vec<Selection>> {
    let mut shapes=BTreeMap::new();
    for view in checkpoint.headers().weights().values() {
        let logical=view.logical();if logical.shape.len()!=2 {continue;}
        if !matches!(view.binding().role.place,engine_model::shapes::Place::Layer(_)) {continue;}
        let kind=match logical.storage {WeightStorage::Nvfp4 {..}=>"nvfp4",WeightStorage::Mxfp8=>"mxfp8",WeightStorage::Fp8Block {..}=>"block-fp8",
            WeightStorage::Plain if matches!(view.components()[0].dtype,engine_model::safetensors::Dtype::F32|engine_model::safetensors::Dtype::Bf16)
                && !matches!(view.binding().operation,engine_model::plan::glm_index::TensorRole::Mhc {..}|engine_model::plan::glm_index::TensorRole::Embedding|engine_model::plan::glm_index::TensorRole::Head)=>"plain-fp32",_=>continue};
        let rows=spark_tp::resident_weights::Rows::new(&logical.shape,logical.storage,Topology {rank:0,parts})?;
        let cols=usize::try_from(logical.shape[1]).map_err(|_|"columns exceed host address space")?;
        let bytes=view.components().iter().try_fold(0u64,|total,component|total.checked_add(component.bytes)).ok_or("component bytes overflow")?;
        shapes.entry((kind,rows.count,cols)).and_modify(|selection:&mut Selection|selection.occurrences+=1)
            .or_insert_with(||Selection {name:logical.name.clone(),kind,rows:rows.count,cols,bytes,occurrences:1});
    }
    let mut selections=shapes.into_values().collect::<Vec<_>>();
    selections.sort_by(|left,right|(right.rows as u128*right.cols as u128*right.occurrences as u128)
        .cmp(&(left.rows as u128*left.cols as u128*left.occurrences as u128)).then(left.name.cmp(&right.name)));
    let mut counts=BTreeMap::new();selections.retain(|selection| {let count=counts.entry(selection.kind).or_insert(0);*count+=1;*count<=per_kind});
    for kind in ["nvfp4","mxfp8","block-fp8","plain-fp32"] {if !selections.iter().any(|selection|selection.kind==kind) {return Err(format!("checkpoint has no admitted {kind} projection"));}}
    let bytes=selections.iter().try_fold(0u64,|total,selection|total.checked_add(selection.bytes)).ok_or("bench weight bytes overflow")?;
    if bytes>limit {return Err(format!("selected full source matrices need {bytes} bytes; limit={limit}; reduce --per-kind explicitly"));}Ok(selections)
}
fn token_layout(selection:&Selection,packed_tokens:u32)->Result<(u32,usize,usize)> {
    if packed_tokens==0 || packed_tokens>1024 {return Err("packed token rows must be within 1..1024".into());}
    let tokens=if matches!(selection.kind,"nvfp4"|"block-fp8") {packed_tokens} else {1};
    let input=selection.cols.checked_mul(tokens as usize).ok_or("token input length overflow")?;
    let output=selection.rows.checked_mul(tokens as usize).ok_or("token output length overflow")?;
    let bytes=input.checked_add(output).and_then(|values|values.checked_mul(4)).ok_or("token buffer bytes overflow")?;
    if bytes>128<<20 || input>u32::MAX as usize || output>u32::MAX as usize {return Err("token input/output exceed the 128 MiB per-case fixture budget".into());}
    Ok((tokens,input,output))
}
fn catalog(selections:&[Selection],packed_tokens:u32)->Result<()> {
    for (index,selection) in selections.iter().enumerate() {let (tokens,_,output)=token_layout(selection,packed_tokens)?;
        println!("CASE id=p{index} kind={} rows={} cols={} token_rows={tokens} matrices=1 output_values={output} macs={} inventory_matrices={} source_bytes={} tensor={}",selection.kind,selection.rows,selection.cols,selection.rows as u128*selection.cols as u128*u128::from(tokens),selection.occurrences,selection.bytes,selection.name);
    }Ok(())
}
#[cfg_attr(not(any(target_os="linux",test)),allow(dead_code))]
fn bytes(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_le_bytes()).collect()}
#[cfg_attr(not(any(target_os="linux",test)),allow(dead_code))]
fn first_difference(expected:&[u8],actual:&[u8])->Option<usize> {
    expected.iter().zip(actual).position(|(left,right)|left!=right).or_else(||(expected.len()!=actual.len()).then_some(expected.len().min(actual.len())))
}
#[cfg_attr(not(any(target_os="linux",test)),allow(dead_code))]
fn median(values:&[u128])->Result<u128> {
    if values.is_empty() {return Err("no timing samples".into());}let mut values=values.to_vec();values.sort_unstable();Ok(values[values.len()/2])
}
#[cfg_attr(not(any(target_os="linux",test)),allow(dead_code))]
fn chosen(text:&str,available:&[String])->Result<Vec<usize>> {
    if text=="all" {return Ok((0..available.len()).collect());}
    let mut seen=BTreeSet::new();text.split(',').map(|name| {
        if !seen.insert(name) {return Err("duplicate case in request".into());}
        available.iter().position(|value|value==name).ok_or_else(||format!("unknown case {name}"))
    }).collect()
}
fn options(arguments:&[String])->Result<(u32,usize,u64,u64,Option<PathBuf>,usize,u32)> {
    let (mut parts,mut per_kind,mut mib,mut seconds)=(2u32,3usize,512u64,55u64);
    let mut baseline=None;let mut cold_mib=0usize;let mut packed_tokens=1u32;
    if arguments.len()%2!=0 {return Err("options require a value".into());}
    for pair in arguments.chunks_exact(2) {match pair[0].as_str() {
        "--parts"=>parts=pair[1].parse().map_err(|_|"invalid parts")?,
        "--per-kind"=>per_kind=pair[1].parse().map_err(|_|"invalid per-kind")?,
        "--memory-mib"=>mib=pair[1].parse().map_err(|_|"invalid memory limit")?,
        "--seconds"=>seconds=pair[1].parse().map_err(|_|"invalid time budget")?,
        "--packed-tokens"=>packed_tokens=pair[1].parse().map_err(|_|"invalid packed token rows")?,
        "--baseline"=>baseline=Some(PathBuf::from(&pair[1])),"--cold-mib"=>cold_mib=pair[1].parse().map_err(|_|"invalid cache-pressure size")?,_=>return Err("unknown bench option".into()),
    }}
    if parts==0 || parts>8 || per_kind==0 || per_kind>32 || mib==0 || mib>4096 || seconds==0 || seconds>59 || cold_mib>1024 || packed_tokens==0 || packed_tokens>1024 {return Err("bench option exceeds fixture bounds".into());}
    Ok((parts,per_kind,mib<<20,seconds,baseline,cold_mib<<20,packed_tokens))
}
fn run()->Result<()> {
    let arguments=std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|value|value=="build-sweep") {return sweep::build(&arguments[1..]);}
    let mode=arguments.first().ok_or("usage: kernel-bench catalog|serve CHECKPOINT [--parts 2 --per-kind 3 --memory-mib 512 --seconds 55] | build-sweep SPEC BUILDER OUT ptx|cubin")?;
    if !matches!(mode.as_str(),"catalog"|"serve") {return Err("expected catalog, serve, or build-sweep".into());}
    let root=PathBuf::from(arguments.get(1).ok_or("checkpoint path is required")?);
    let option_start=if mode=="serve" {3} else {2};
    let out=if mode=="serve" {Some(PathBuf::from(arguments.get(2).ok_or("serve requires a new evidence directory")?))} else {None};
    let (parts,per_kind,limit,seconds,baseline,cold_bytes,packed_tokens)=options(&arguments[option_start..])?;let started=Instant::now();
    let checkpoint=open(&root)?;let selections=select(&checkpoint,parts,per_kind,limit)?;catalog(&selections,packed_tokens)?;
    if mode=="catalog" {return Ok(());}
    #[cfg(target_os="linux")]
    {device::serve(&checkpoint,&selections,parts,started+Duration::from_secs(seconds),out.as_deref().unwrap(),baseline.as_deref(),cold_bytes,packed_tokens)}
    #[cfg(not(target_os="linux"))]
    {let _=(checkpoint,started,out,baseline,cold_bytes,packed_tokens,Duration::from_secs(seconds));Err("serve requires Linux and CUDA; catalog and build-sweep are host-only".into())}
}
fn main() {if let Err(error)=run() {eprintln!("KERNEL BENCH ERROR {error}");std::process::exit(1);}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn byte_comparison_does_not_ignore_zero_sign_nan_or_trailing_bytes() {
        assert_eq!(first_difference(&bytes(&[0.0]),&bytes(&[-0.0])),Some(3));
        assert_eq!(first_difference(&bytes(&[f32::from_bits(0x7fc00001)]),&bytes(&[f32::from_bits(0x7fc00002)])),Some(0));
        assert_eq!(first_difference(&[1,2],&[1,2,3]),Some(2));assert_eq!(first_difference(&[1,2],&[1,2]),None);
    }
    #[test] fn case_names_and_budgets_fail_closed() {
        let available=vec!["p0".into(),"mhc".into()];assert_eq!(chosen("mhc,p0",&available).unwrap(),[1,0]);
        assert!(chosen("p0,p0",&available).is_err());assert!(chosen("missing",&available).is_err());
        assert!(options(&["--seconds".into(),"60".into()]).is_err());assert!(options(&["--parts".into(),"0".into()]).is_err());
        assert!(options(&["--cold-mib".into(),"1025".into()]).is_err());
        assert_eq!(options(&["--cold-mib".into(),"256".into()]).unwrap().5,256<<20);
        assert_eq!(options(&[]).unwrap().5,0);
        assert_eq!(median(&[9,1,5,3,7]).unwrap(),5);
    }
    #[test] fn packed_token_counts_are_bounded_and_default_to_one() {
        assert_eq!(options(&[]).unwrap().6,1);
        assert_eq!(options(&["--packed-tokens".into(),"128".into()]).unwrap().6,128);
        for value in ["0","1025","-1","many"] {assert!(options(&["--packed-tokens".into(),value.into()]).is_err());}
    }
    #[test] fn packed_rows_account_for_one_matrix_and_all_tokens() {
        let mut selection=Selection {name:"matrix".into(),kind:"nvfp4",rows:1024,cols:4096,bytes:1,occurrences:8};
        assert_eq!(token_layout(&selection,128).unwrap(),(128,524288,131072));
        selection.kind="block-fp8";assert_eq!(token_layout(&selection,8).unwrap(),(8,32768,8192));
        for kind in ["mxfp8","plain-fp32"] {selection.kind=kind;assert_eq!(token_layout(&selection,128).unwrap(),(1,4096,1024));}
        selection.kind="nvfp4";selection.cols=usize::MAX;assert!(token_layout(&selection,2).is_err());
        selection.cols=1<<20;assert!(token_layout(&selection,128).is_err());
    }
}
