//! Bounded first-token check through the executable checkpoint Factory.
use super::{prompt,read,write,equal,sha256,child};
#[cfg(target_os="linux")] use super::{checkpoint,floats};
use spark_tp::Result;
use std::{fs,path::Path};
const TOKENS:usize=64;
const COMPLETE:&[u8]=b"CHECKPOINT BOUNDED FIRST TOKENS v2 PASS\n";
const SCOPE:&[u8]=b"full_stack=true\nfactory=bounded_causal_recipe\nactivation_mxfp8=dynamic_bf16_source\nactivation_nvfp4=calibrated_block16_e4m3_e2m1_rne\nshared_fp8=bf16_source_fp32_fold_bf16_output\nweights=ordinary_mxfp8_resident_selected_experts_staged\nstate=device_resident\ntransport=false\ntopology=complete_rows_each_spark\ndrafting=false\nlong_context=false\nbackend_parity=unproved\n";
#[cfg(target_os="linux")]
fn metadata(root:&Path,input:&Path,request:&Path,out:&Path)->Result<Vec<u32>> {
    let ids=prompt(input)?;if ids.len()>64 {return Err("first-token prompt exceeds its bounded probe limit".into());}
    let mut identity=format!("source={}\ninput_ids={ids:?}\n",include_str!("../../../../../REVISION").trim());
    for name in ["config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json"] {
        identity.push_str(&format!("{name}={}\n",sha256::digest_reader(fs::File::open(root.join(name)).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?));
    }
    identity.push_str(&format!("request={}\n",sha256::digest(&read(request)?)));
    write(&out.join("IDENTITY"),identity.as_bytes())?;write(&out.join("SCOPE"),SCOPE)?;
    write(&out.join("INPUT_IDS"),ids.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;Ok(ids)
}
fn frontend(root:&Path,input:&Path,request:&Path)->Result<(Vec<u32>,spark_serve::protocol::Request,spark_serve::tokenizer::ModelTokenizer)> {
    use spark_serve::Tokenizer;
    let ids=prompt(input)?;if ids.len()>64 {return Err("first-token prompt exceeds its bounded probe limit".into());}
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(root)?;
    let context=ids.len().checked_add(TOKENS).ok_or("first-token context overflow")?;
    let limits=spark_serve::Limits {input_tokens:ids.len(),output_tokens:TOKENS,context_tokens:context,default_thinking:false,next_mtp_depth:Some(0),..Default::default()};
    let r=spark_serve::protocol::parse(&read(request)?,true,&limits)?;
    if ![8,32,64].contains(&r.max_tokens) || !r.ignore_eos || r.cache || r.mtp_depth!=Some(0) || r.draft!=Some(spark_serve::protocol::DraftMode::Plain)
        || r.sampling.top_k!=0 || r.sampling.top_p!=1.0 || r.sampling.min_p!=0.0 || r.sampling.temperature!=0.0 || r.sampling.repetition_penalty!=1.0 || r.sampling.presence_penalty!=0.0 || r.sampling.frequency_penalty!=0.0 {
        return Err(format!("first-token request contract differs: max={} ignore_eos={} cache={} depth={:?} draft={:?} sampling={:?}",r.max_tokens,r.ignore_eos,r.cache,r.mtp_depth,r.draft,r.sampling));
    }
    let spark_serve::protocol::Prompt::Chat(chat)=&r.prompt else {return Err("first-token request requires a chat prompt".into());};
    if tokenizer.encode_chat(chat,limits.body_bytes)?!=ids {return Err("checkpoint chat template IDs differ from the prepared prompt".into());}
    Ok((ids,r,tokenizer))
}
#[cfg(target_os="linux")]
fn device(root:&Path,input:&Path,request:&Path,out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    use spark_serve::{Engine,Session,Tokenizer};
    let started=std::time::Instant::now();let mut decode_seconds=0.0;
    let (ids,r,tokenizer)=frontend(root,input,request)?;let recorded=metadata(root,input,request,out)?;
    if ids!=recorded {return Err("first-token input changed during preparation".into());}
    write(&out.join("OUTPUT_LIMIT"),format!("{}\n",r.max_tokens).as_bytes())?;
    let c=checkpoint(root)?;let context=ids.len().checked_add(r.max_tokens).ok_or("first-token context overflow")?;
    let mut output=Vec::new();let mut hashes=String::new();
    with_backend(0,layer_ptx()?,usize::MAX,|backend| {
        let engine=spark_serve::text_program::TextEngine::from_checkpoint(spark_tp::bounded_decoder::Factory {checkpoint:&c,backend},&c,&tokenizer,context)?;
        let mut session=engine.session()?;session.configure_request(&r)?;let mut last=None;
        for (i,&id) in ids.iter().enumerate() {eprintln!("FIRST TOKENS prompt_step={i} token={id}");let logits=session.step(id)?;
            hashes.push_str(&format!("prompt-{i}={}\n",sha256::digest(&floats(&logits)?)));last=Some(logits);}
        let mut logits=last.ok_or("first-token prompt is empty")?;
        let decode_started=std::time::Instant::now();
        for at in 0..r.max_tokens {if logits.len()!=c.description().vocab as usize || logits.iter().any(|v|!v.is_finite()) {return Err("first-token logits are invalid".into());}
            let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}let id=best as u32;output.push(id);
            write(&out.join(format!("{at}-logits.f32")),&floats(&logits)?)?;eprintln!("FIRST TOKENS generated={at} token={id}");
            if at+1<r.max_tokens {logits=session.step(id)?;}
        }decode_seconds=decode_started.elapsed().as_secs_f64();backend.synchronize()?;if let Some(e)=backend.fatal_error() {return Err(e);}Ok(())
    })?;
    let mut decoder=tokenizer.decoder();let mut text=String::new();for &id in &output {text.push_str(&decoder.push(id).map_err(|e|e.to_string())?);}text.push_str(&decoder.finish());
    write(&out.join("OUTPUT_IDS"),output.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;
    write(&out.join("TEXT"),text.as_bytes())?;write(&out.join("PROMPT_LOGITS"),hashes.as_bytes())?;seal(out)?;
    println!("CHECKPOINT TIMING output_tokens={} total_seconds={:.6} decode_steps={} decode_seconds={decode_seconds:.6} decode_tokens_per_second={:.6}",output.len(),started.elapsed().as_secs_f64(),output.len()-1,(output.len()-1) as f64/decode_seconds);
    println!("CHECKPOINT FIRST TOKENS DEVICE PASS ids={output:?} decoded={text:?} transport=false factory=true");Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path)->Result<()> {Err("bounded first-token device check requires Linux and CUDA".into())}
fn names(out:&Path)->Result<Vec<String>> {let mut names=["IDENTITY","SCOPE","INPUT_IDS","OUTPUT_IDS","OUTPUT_LIMIT","TEXT","PROMPT_LOGITS"].map(str::to_owned).to_vec();
    names.extend((0..output_limit(out)?).map(|i|format!("{i}-logits.f32")));Ok(names)}
fn output_limit(out:&Path)->Result<usize> {let bytes=read(&out.join("OUTPUT_LIMIT"))?;let n=std::str::from_utf8(&bytes).map_err(|e|e.to_string())?.trim().parse::<usize>().map_err(|e|e.to_string())?;if ![8,32,64].contains(&n) {return Err("unsupported output receipt limit".into());}Ok(n)}
fn manifest(out:&Path)->Result<Vec<u8>> {let mut m=String::new();for name in names(out)? {m.push_str(&format!("{}  {name}\n",sha256::digest(&read(&out.join(&name))?)));}Ok(m.into_bytes())}
#[cfg(any(target_os="linux",test))]
fn seal(out:&Path)->Result<()> {write(&out.join("OUTPUT.sha256"),&manifest(out)?)?;write(&out.join("COMPLETE"),COMPLETE)}
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn complete(out:&Path)->Result<()> {
    equal("first-token completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;equal("first-token scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let ids=prompt(&out.join("OUTPUT_IDS"))?;if ids.len()!=output_limit(out)? {return Err("first-token output count differs".into());}
    let mut expected=None;
    for at in 0..ids.len() {let bytes=read(&out.join(format!("{at}-logits.f32")))?;
        if bytes.is_empty() || bytes.len()%4!=0 || expected.is_some_and(|n|n!=bytes.len()) {return Err("first-token saved logits extent differs".into());}expected=Some(bytes.len());
        let values=bytes.as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect::<Vec<_>>();
        if values.iter().any(|v|!v.is_finite()) {return Err("first-token saved logits are not finite".into());}
        let mut best=0;for i in 1..values.len() {if values[i]>values[best] {best=i;}}
        if ids[at] as usize!=best {return Err("first-token saved ID differs from deterministic argmax".into());}
    }
    equal("first-token manifest",&read(&out.join("OUTPUT.sha256"))?,&manifest(out)?)?;Ok(())
}
fn compare(a:&Path,b:&Path)->Result<()> {complete(a)?;complete(b)?;for name in names(a)? {equal(&name,&read(&a.join(&name))?,&read(&b.join(&name))?)?;}
    println!("CHECKPOINT FIRST TOKENS COMPARE PASS ids={:?} decoded={:?} full_logits=true transport=false",prompt(&a.join("OUTPUT_IDS"))?,String::from_utf8(read(&a.join("TEXT"))?).map_err(|e|e.to_string())?);Ok(())}
fn suite(root:&Path,input:&Path,request:&Path,out:&Path)->Result<()> {
    for run in 1..=3 {let p=out.join(format!("run{run}"));child(&[Path::new("tokens"),Path::new("device"),root,input,request,&p],&out.join(format!("run{run}.log")))?;
        compare(&out.join("run1"),&p)?;}
    write(&out.join("COMPLETE"),COMPLETE)?;println!("CHECKPOINT FIRST TOKENS SUITE PASS repeats=3 output_ids=recorded transport=false");Ok(())
}
fn compare_suite(a:&Path,b:&Path)->Result<()> {for p in [a,b] {equal("first-token suite completion",&read(&p.join("COMPLETE"))?,COMPLETE)?;}
    for run in 1..=3 {compare(&a.join(format!("run{run}")),&b.join(format!("run{run}")))?;}
    println!("CHECKPOINT FIRST TOKENS CROSS SPARK PASS repeats=3 output_ids=recorded transport=false");Ok(())}
pub fn run(args:&[&str])->Result<()> {
    let new=|p:&str|->Result<()> {fs::create_dir(p).map_err(|e|e.to_string())};
    match args {
        ["prepare",root,input,request]=>{let (ids,r,_)=frontend(Path::new(root),Path::new(input),Path::new(request))?;println!("CHECKPOINT FIRST TOKENS FRONTEND PASS ids={ids:?} output_limit={} draft_depth=0",r.max_tokens);Ok(())},
        ["device",root,input,request,out]=>{new(out)?;device(Path::new(root),Path::new(input),Path::new(request),Path::new(out))},
        ["suite",root,input,request,out]=>{new(out)?;suite(Path::new(root),Path::new(input),Path::new(request),Path::new(out))},
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),["compare-suite",a,b]=>compare_suite(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check tokens device MODEL INPUT_IDS REQUEST_JSON NEW_OUT | suite MODEL INPUT_IDS REQUEST_JSON NEW_OUT | compare LEFT RIGHT | compare-suite LEFT RIGHT".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_request_parser_preserves_explicit_zero_draft_depth() {
        let body=br#"{"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"cache":false,"ignore_eos":true,"enable_thinking":false,"draft":"plain","mtp_depth":0}"#;
        let limits=spark_serve::Limits {input_tokens:64,output_tokens:TOKENS,context_tokens:72,default_thinking:false,next_mtp_depth:Some(0),..Default::default()};
        let r=spark_serve::protocol::parse(body,true,&limits).unwrap();assert_eq!(r.mtp_depth,Some(0));assert_eq!(r.draft,Some(spark_serve::protocol::DraftMode::Plain));assert_eq!(r.max_tokens,8);assert!(r.ignore_eos);assert!(!r.cache);
        let ordinary=spark_serve::Limits {next_mtp_depth:None,..limits};assert_eq!(spark_serve::protocol::parse(body,true,&ordinary).unwrap().mtp_depth,None);
    }
    #[test]
    fn saved_first_token_ids_must_equal_the_full_logits_argmax() {
        for count in [8,32,64] {
            let root=std::env::temp_dir().join(format!("vade-token-receipt-{}-{count}",std::process::id()));fs::create_dir(&root).unwrap();
            for (name,bytes) in [("IDENTITY",b"synthetic".as_slice()),("SCOPE",SCOPE),("INPUT_IDS",b"0\n"),("TEXT",b"synthetic"),("PROMPT_LOGITS",b"synthetic")] {write(&root.join(name),bytes).unwrap();}
            write(&root.join("OUTPUT_IDS"),"1\n".repeat(count).as_bytes()).unwrap();write(&root.join("OUTPUT_LIMIT"),format!("{count}\n").as_bytes()).unwrap();
            for at in 0..count {write(&root.join(format!("{at}-logits.f32")),&[0.0f32,1.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();}
            seal(&root).unwrap();complete(&root).unwrap();
            fs::write(root.join("OUTPUT_IDS"),format!("0\n{}","1\n".repeat(count-1))).unwrap();
            fs::write(root.join("OUTPUT.sha256"),manifest(&root).unwrap()).unwrap();assert!(complete(&root).is_err());
            fs::write(root.join("OUTPUT_IDS"),"1\n".repeat(count-1)).unwrap();assert!(complete(&root).is_err());
            fs::write(root.join("OUTPUT_LIMIT"),b"65\n").unwrap();assert!(complete(&root).is_err());fs::remove_dir_all(root).unwrap();
        }
    }
}
