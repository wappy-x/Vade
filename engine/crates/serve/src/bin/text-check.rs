//! Check pinned text vectors and real stored text edges. No decoder admission.
#![forbid(unsafe_code)]
use engine_model::{json::{self,Value},plan::{ConfigPlan,admission::CheckpointAdmission},safetensors::read_bounded,shapes::{CONFIG_BYTES,HEADER_BYTES}};
use spark_gpu_probe::{text_edges::greedy,text_reference};
use spark_serve::{Tokenizer,Decoder,tokenizer::ModelTokenizer,text_checkpoint::CheckpointTextEdges};
use std::{fs,io::Read,path::Path};
#[path="../../../../tools/export/src/sha256.rs"]
mod sha256;
#[cfg(any(target_os="linux",test))]
#[path="text-check/budget.rs"]
mod budget;
type Result<T>=std::result::Result<T,String>;
const VECTORS:&[u8]=include_bytes!("../../tests/fixtures/glm-text/source-vectors.json");
const COMPLETE:&[u8]=b"TEXT CHECKPOINT PRIMITIVES v1 PASS\n";
fn json(bytes:&[u8])->Result<Value>{json::parse(bytes,json::Limits {bytes:32<<20,nodes:2_000_000,string_bytes:1<<20,depth:32}).map_err(|e|e.to_string())}
fn fixture()->Result<Value>{json(VECTORS)}
fn ids(value:&Value)->Result<Vec<u32>>{value.array().map_err(|e|e.to_string())?.iter().map(|v|
    u32::try_from(v.u64().map_err(|e|e.to_string())?).map_err(|_|"token ID exceeds ABI".into())).collect()}
fn probe_ids()->Result<Vec<u32>>{ids(&fixture()?.object().map_err(|e|e.to_string())?["probe_ids"])}
fn write(path:&Path,bytes:&[u8])->Result<()>{fs::write(path,bytes).map_err(|e|e.to_string())}
fn read(path:&Path)->Result<Vec<u8>>{fs::read(path).map_err(|e|e.to_string())}
fn bytes(values:&[f32])->Vec<u8>{values.iter().flat_map(|v|v.to_le_bytes()).collect()}
fn equal(label:&str,a:&[u8],b:&[u8])->Result<()> {
    if a!=b {return Err(format!("{label}: bytes differ actual={} expected={} first={:?}",a.len(),b.len(),a.iter().zip(b).position(|(x,y)|x!=y)));} Ok(())
}
fn artifacts(root:&Path)->Result<ModelTokenizer>{
    let values=fixture()?;
    for (name,hash) in values.object().map_err(|e|e.to_string())?["artifacts"].object().map_err(|e|e.to_string())? {
        let file=fs::File::open(root.join(name)).map_err(|e|e.to_string())?;
        let actual=sha256::digest_reader(file).map_err(|e|e.to_string())?;
        equal(name,actual.as_bytes(),hash.string().map_err(|e|e.to_string())?.as_bytes())?;
    }
    ModelTokenizer::from_directory(root).map_err(|e|e.to_string())
}
fn tokens(root:&Path,out:&Path)->Result<()> {
    let model=artifacts(root)?;let vectors=fixture()?;let mut count=0;
    for vector in vectors.object().map_err(|e|e.to_string())?["plain"].array().map_err(|e|e.to_string())? {
        let fields=vector.object().map_err(|e|e.to_string())?;
        let text=fields["text"].string().map_err(|e|e.to_string())?;let expected=ids(&fields["ids"])?;
        let actual=Tokenizer::encode(&model,text)?;
        if actual!=expected {return Err(format!("token vector {count} differs: {actual:?} != {expected:?}"));}
        equal("decode",model.decode(&expected,false).map_err(|e|e.to_string())?.as_bytes(),text.as_bytes())?;
        let raw=expected.iter().map(|&id|model.token_bytes(id).map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?.concat();
        equal("raw token bytes",&raw,text.as_bytes())?;
        let mut decoder=model.decoder();let mut streamed=String::new();
        for &id in &expected {streamed.push_str(&Decoder::push(&mut decoder,id)?);}
        streamed.push_str(&Decoder::finish(&mut decoder)?);equal("stream decode",streamed.as_bytes(),text.as_bytes())?;
        println!("TEXT VECTOR {count} ids={actual:?} decode_sha256={}",sha256::digest(text.as_bytes()));count+=1;
    }
    // Compare selected fixed external prompt bytes. Their complete source
    // fixture and the broader renderer suite remain committed separately.
    let published=include_bytes!("../../tests/fixtures/glm-upstream/vllm-test-cases.json");
    equal("published prompt fixture",sha256::digest(published).as_bytes(),
        b"8660be3b335be039a57af9e180403be0baee8cd1455f151d7297a20285f39497")?;
    let published=json(published)?;let mut prompts=0;
    for case in published.object().map_err(|e|e.to_string())?["cases"].array().map_err(|e|e.to_string())? {
        let fields=case.object().map_err(|e|e.to_string())?;
        let name=fields["name"].string().map_err(|e|e.to_string())?;
        if name!="basic" && name!="effort_low" && name!="effort_high" {continue;}
        let effort=match name {"effort_low"=>",\"reasoning_effort\":\"low\"","effort_high"=>",\"reasoning_effort\":\"high\"",_=>""};
        let body=format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"你好，GLM！\"}}]{effort}}}");
        let request=spark_serve::protocol::parse(body.as_bytes(),true,&Default::default())?;
        let spark_serve::protocol::Prompt::Chat(chat)=request.prompt else {return Err("chat fixture parsed as plain text".into());};
        let expected=fields["expected"].string().map_err(|e|e.to_string())?;
        let actual=model.format_chat_input(&chat,8192)?;equal(name,actual.as_bytes(),expected.as_bytes())?;
        let encoded=model.encode_chat(&chat,8192)?;
        equal("chat decode",model.decode(&encoded,false).map_err(|e|e.to_string())?.as_bytes(),expected.as_bytes())?;
        write(&out.join(format!("prompt-{name}.txt")),actual.as_bytes())?;
        write(&out.join(format!("prompt-{name}.ids")),format!("{encoded:?}\n").as_bytes())?;
        prompts+=1;
    }
    if prompts!=3 {return Err("published prompt subset is incomplete".into());}
    write(&out.join("source-vectors.json"),VECTORS)?;
    println!("TEXT VECTORS PASS cases={count} external_prompts={prompts} proof=source-derived runtime_parity=open");Ok(())
}
fn checkpoint(root:&Path)->Result<CheckpointAdmission>{
    let config=read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|e|e.to_string())?;
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&config).map_err(|e|e.to_string())?
        else {return Err("probe needs a deferred text description".into());};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|e|e.to_string())?;
    description.open_headers(root,&index).map_err(|e|e.to_string())
}
struct Plane<'a,'c>{bound:&'a CheckpointTextEdges<'c>,offset:usize,left:usize}
impl Read for Plane<'_,'_>{fn read(&mut self,out:&mut [u8])->std::io::Result<usize>{
    let count=out.len().min(self.left);
    if count!=0 {self.bound.read_range(self.offset,&mut out[..count]).map_err(std::io::Error::other)?;}
    self.offset+=count;self.left-=count;Ok(count)
}}
fn identity(root:&Path,bound:&CheckpointTextEdges<'_>)->Result<Vec<u8>>{
    let layout=bound.layout();let s=layout.shape;
    let mut text=format!("text-check-v1\nsource={}\nfixture={}\nhidden={}\nvocab={}\nstreams={}\nids={:?}\n",
        env!("VADE_BUILD_GIT_SHA"),sha256::digest(VECTORS),s.hidden(),s.vocab(),s.streams(),probe_ids()?);
    for name in ["config.json","model.safetensors.index.json"] {
        let hash=sha256::digest_reader(fs::File::open(root.join(name)).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        text.push_str(&format!("{name}={hash}\n"));
    }
    let names=[&bound.edges().embedding.name,&bound.edges().head.name,&bound.edges().head_scales.name,&bound.edges().norm.name];
    for (name,range) in names.into_iter().zip(layout.ranges(bound.arena_bytes())?) {
        let hash=sha256::digest_reader(Plane{bound,offset:range.start,left:range.len()}).map_err(|e|e.to_string())?;
        text.push_str(&format!("{name}={hash}\n"));
    }
    Ok(text.into_bytes())
}
fn embeddings(bound:&CheckpointTextEdges<'_>)->Result<Vec<Vec<f32>>>{
    let layout=bound.layout();let s=layout.shape;
    probe_ids()?.into_iter().map(|id|{
        s.token(id)?;let mut row=vec![0;s.hidden()*2];
        bound.read_range(layout.embedding+id as usize*s.hidden()*2,&mut row)?;
        text_reference::embedding(&row)
    }).collect()
}
fn save_plane(out:&Path,name:&str,values:&[f32])->Result<()>{
    if values.iter().any(|x|!x.is_finite()) {return Err(format!("{name}: nonfinite output"));}
    write(&out.join(name),&bytes(values))
}
fn host(root:&Path,out:&Path)->Result<()> {
    let checkpoint=checkpoint(root)?;let bound=CheckpointTextEdges::new(&checkpoint)?;bound.validate_head_scales()?;
    let layout=bound.layout();let s=layout.shape;let inputs=embeddings(&bound)?;
    write(&out.join("IDENTITY"),&identity(root,&bound)?)?;
    for (i,row) in inputs.iter().enumerate(){
        save_plane(out,&format!("{i}-input.bin"),row)?;
        save_plane(out,&format!("{i}-embedding.bin"),&row.repeat(s.streams()))?;
    }
    let mut logits=vec![Vec::with_capacity(s.vocab());inputs.len()];
    // Bound host scratch by tensor dimensions. Keep each scale row intact.
    let batch=16usize;let h=s.hidden();
    for start in (0..s.vocab()).step_by(batch) {
        let rows=batch.min(s.vocab()-start);let mut weights=vec![0;rows*h];let mut scales=vec![0;rows*h/32];
        bound.read_range(layout.head+start*h,&mut weights)?;bound.read_range(layout.scales+start*h/32,&mut scales)?;
        for row in 0..rows {for (input,output) in inputs.iter().zip(&mut logits){
            output.push(text_reference::project_row(&weights[row*h..(row+1)*h],&scales[row*h/32..(row+1)*h/32],input)?);
        }}
    }
    for (i,row) in logits.iter().enumerate(){
        save_plane(out,&format!("{i}-logits.bin"),row)?;
        write(&out.join(format!("{i}-selected.bin")),&greedy(row)?.to_le_bytes())?;
    }
    seal(out)?;println!("TEXT CHECKPOINT HOST PASS ids={} hidden={} logits={} primitive_only=true",inputs.len(),h,s.vocab());Ok(())
}
fn expected_files()->Result<Vec<String>>{
    let mut names=vec!["IDENTITY".into()];
    for i in 0..probe_ids()?.len(){for plane in ["input","embedding","logits","selected"]{names.push(format!("{i}-{plane}.bin"));}}
    Ok(names)
}
fn seal(out:&Path)->Result<()> {
    let mut manifest=String::new();
    for name in expected_files()?{manifest.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(&name))?)));}
    write(&out.join("OUTPUT.sha256"),manifest.as_bytes())?;write(&out.join("COMPLETE"),COMPLETE)
}
fn complete(root:&Path)->Result<()> {
    equal("completion",&read(&root.join("COMPLETE"))?,COMPLETE)?;
    let mut expected=String::new();
    for name in expected_files()?{expected.push_str(&format!("{}  {name}\n",sha256::digest(&read(&root.join(&name))?)));}
    equal("output manifest",&read(&root.join("OUTPUT.sha256"))?,expected.as_bytes())
}
fn compare(left:&Path,right:&Path)->Result<()> {
    complete(left)?;complete(right)?;
    for name in expected_files()?{equal(&name,&read(&left.join(&name))?,&read(&right.join(&name))?)?;}
    println!("TEXT CHECKPOINT COMPARE PASS files={}",expected_files()?.len());Ok(())
}
fn child(args:&[&Path],log:&Path)->Result<()> {
    let output=fs::OpenOptions::new().write(true).create_new(true).open(log).map_err(|e|e.to_string())?;
    let status=std::process::Command::new(std::env::current_exe().map_err(|e|e.to_string())?)
        .args(args).stdout(output.try_clone().map_err(|e|e.to_string())?).stderr(output)
        .status().map_err(|e|e.to_string())?;
    if !status.success() {return Err(format!("child probe failed; read {}",log.file_name().unwrap().to_string_lossy()));} Ok(())
}
fn suite(root:&Path,out:&Path)->Result<()> {
    // Separate processes give each device run a fresh CUDA context. This
    // command does not contact another machine or compile any source.
    let hardware=std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version","--format=csv,noheader"]).output().map_err(|e|e.to_string())?;
    if !hardware.status.success() {return Err("hardware query failed".into());}
    write(&out.join("hardware.txt"),&hardware.stdout)?;
    let executable=std::env::current_exe().map_err(|e|e.to_string())?;
    write(&out.join("binary.sha256"),format!("{}\n",sha256::digest_reader(fs::File::open(executable).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?).as_bytes())?;
    let tokens=out.join("tokens");let host=out.join("host");
    child(&[Path::new("tokens"),root,&tokens],&out.join("tokens.log"))?;
    child(&[Path::new("host"),root,&host],&out.join("host.log"))?;
    let first=out.join("run1");
    for run in 1..=3 {
        let result=out.join(format!("run{run}"));
        child(&[Path::new("device"),root,&host,&result],&out.join(format!("run{run}.log")))?;
        compare(&host,&result)?;if run!=1 {compare(&first,&result)?;}
    }
    write(&out.join("COMPLETE"),b"TEXT CHECKPOINT SUITE PASS\n")?;
    println!("TEXT CHECKPOINT SUITE PASS fresh_device_processes=3 cross_device=not_run");Ok(())
}
#[cfg(target_os="linux")]
fn device(root:&Path,oracle:&Path,out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    complete(oracle)?;
    let checkpoint=checkpoint(root)?;let bound=CheckpointTextEdges::new(&checkpoint)?;let s=bound.layout().shape;
    let identity=identity(root,&bound)?;equal("checkpoint identity",&identity,&read(&oracle.join("IDENTITY"))?)?;
    write(&out.join("IDENTITY"),&identity)?;
    let budget=budget::Budget::new(bound.layout(),bound.arena_bytes())?;
    with_backend(0,layer_ptx()?,budget.device,|backend|{
        let available=backend.available_memory_bytes()?;
        budget.admit(available)?;
        println!("TEXT CHECKPOINT BUDGET arena={} device={} pinned={} required={} available={available} estimated=true",
            bound.arena_bytes(),budget.device,budget.pinned,budget.required);
        let edges=bound.upload(backend)?;
        for (i,id) in probe_ids()?.into_iter().enumerate(){
            let input=read(&oracle.join(format!("{i}-input.bin")))?;
            if input.len()!=s.hidden()*4 {return Err("oracle input extent differs".into());}
            let values=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            write(&out.join(format!("{i}-input.bin")),&input)?;
            let embedded=edges.embed(id)?;
            save_plane(out,&format!("{i}-embedding.bin"),&backend.read_vector_values(&embedded)?)?;
            let input=backend.upload_vector(&values)?;let logits=edges.project_fp32(&input)?;
            let logits=backend.read_vector_values(&logits)?;
            save_plane(out,&format!("{i}-logits.bin"),&logits)?;
            write(&out.join(format!("{i}-selected.bin")),&greedy(&logits)?.to_le_bytes())?;
            println!("TEXT CHECKPOINT DEVICE id={id} hidden={} logits={}",s.hidden(),s.vocab());
        }
        if let Some(error)=backend.fatal_error(){return Err(error);}Ok(())
    })?;
    // Keep raw failed outputs. Mark success only after every byte matches.
    for name in expected_files()?{equal(&name,&read(&out.join(&name))?,&read(&oracle.join(&name))?)?;}
    seal(out)?;println!("TEXT CHECKPOINT DEVICE PASS primitive_only=true decoder=absent norm_policy=unqualified");Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path)->Result<()>{Err("device probe requires Linux and CUDA".into())}
fn main(){
    println!("TEXT CHECK source={}",env!("VADE_BUILD_GIT_SHA"));
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let run=||->Result<()>{match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice(){
        ["tokens",root,out]=>{fs::create_dir(out).map_err(|e|e.to_string())?;tokens(Path::new(root),Path::new(out))},
        ["host",root,out]=>{fs::create_dir(out).map_err(|e|e.to_string())?;host(Path::new(root),Path::new(out))},
        ["device",root,oracle,out]=>{fs::create_dir(out).map_err(|e|e.to_string())?;device(Path::new(root),Path::new(oracle),Path::new(out))},
        ["compare",left,right]=>compare(Path::new(left),Path::new(right)),
        ["suite",root,out]=>{fs::create_dir(out).map_err(|e|e.to_string())?;suite(Path::new(root),Path::new(out))},
        _=>Err("usage: text-check tokens ARTIFACTS NEW_DIR | host CHECKPOINT NEW_DIR | device CHECKPOINT ORACLE NEW_DIR | compare LEFT RIGHT | suite CHECKPOINT NEW_DIR".into()),
    }};
    if let Err(error)=run(){eprintln!("TEXT CHECK FAIL {error}");std::process::exit(1);}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_probe_ids_are_distinct_and_source_vectors_are_explicit(){
        let ids=probe_ids().unwrap();let set:std::collections::BTreeSet<_>=ids.iter().collect();assert_eq!(set.len(),ids.len());
        let fixture=fixture().unwrap();let rows=fixture.object().unwrap()["plain"].array().unwrap();assert!(!rows.is_empty());
        for row in rows {let row=row.object().unwrap();row["text"].string().unwrap();super::ids(&row["ids"]).unwrap();}
    }
    #[test]
    fn incomplete_and_changed_outputs_fail_comparison(){
        let root=std::env::temp_dir().join(format!("vade-text-check-{}",std::process::id()));fs::create_dir(&root).unwrap();
        assert!(complete(&root).is_err());
        for name in expected_files().unwrap(){write(&root.join(name),b"fixture").unwrap();}
        seal(&root).unwrap();complete(&root).unwrap();
        write(&root.join("0-logits.bin"),b"changed").unwrap();assert!(complete(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
