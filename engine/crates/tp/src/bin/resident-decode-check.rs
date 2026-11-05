//! Bounded resident row-split decoder evidence. Public service admission stays separate.
#![forbid(unsafe_code)]
use std::{fs,path::{Path,PathBuf}};
use spark_tp::Result;
#[cfg(target_os="linux")]
#[path="resident-decode-check/recurrence.rs"] mod recurrence;
#[cfg(target_os="linux")]
#[path="resident-decode-check/convolution.rs"] mod convolution;
#[cfg(target_os="linux")]
#[path="resident-decode-check/nvfp4.rs"] mod nvfp4;
#[cfg(target_os="linux")]
#[path="resident-decode-check/prompt_rows.rs"] mod prompt_rows;
#[cfg(target_os="linux")]
#[path="../../../../tools/export/src/sha256.rs"] mod sha256;
fn read(path:&Path)->Result<Vec<u8>> {fs::read(path).map_err(|e|e.to_string())}
#[cfg(target_os="linux")]
fn write(path:&Path,bytes:&[u8])->Result<()> {use std::io::Write;
    fs::OpenOptions::new().create_new(true).write(true).open(path).and_then(|mut f|f.write_all(bytes)).map_err(|e|e.to_string())}
fn ids(path:&Path)->Result<Vec<u32>> {let text=fs::read_to_string(path).map_err(|e|e.to_string())?;
    let values=text.split_whitespace().map(|v|v.parse().map_err(|_|"invalid token ID".into())).collect::<Result<Vec<_>>>()?;
    if values.is_empty() {return Err("empty token IDs".into());}Ok(values)}
fn prepare(root:&Path,request:&Path,out:&Path)->Result<()> {
    use spark_serve::Tokenizer;
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(root)?;
    let limits=spark_serve::Limits {input_tokens:512,output_tokens:256,context_tokens:768,next_mtp_depth:Some(0),..Default::default()};
    let request=spark_serve::protocol::parse(&read(request)?,true,&limits)?;
    let spark_serve::protocol::Prompt::Chat(chat)=request.prompt else {return Err("prompt preparation requires chat".into());};
    let ids=tokenizer.encode_chat(&chat,limits.body_bytes)?;
    if ids.is_empty() || ids.len()>512 {return Err("prepared prompt exceeds diagnostic bound".into());}
    use std::io::Write;
    let mut file=fs::OpenOptions::new().create_new(true).write(true).open(out).map_err(|e|e.to_string())?;
    file.write_all(ids.iter().map(|id|format!("{id}\n")).collect::<String>().as_bytes()).map_err(|e|e.to_string())?;
    println!("RESIDENT PREPARE PASS prompt_tokens={} output_limit={}",ids.len(),request.max_tokens);Ok(())
}
#[cfg(target_os="linux")]
fn floats(v:&[f32])->Result<Vec<u8>> {if v.iter().any(|v|!v.is_finite()) {return Err("nonfinite saved logits".into());}Ok(v.iter().flat_map(|v|v.to_le_bytes()).collect())}
fn same(a:&Path,b:&Path,name:&str)->Result<()> {if read(&a.join(name))?!=read(&b.join(name))? {return Err(format!("saved {name} differs"));}Ok(())}
fn verify(path:&Path)->Result<Vec<u32>> {
    let output=ids(&path.join("OUTPUT_IDS"))?;
    for (at,&id) in output.iter().enumerate() {let bytes=read(&path.join(format!("{at}-logits.f32")))?;
        if bytes.is_empty() || bytes.len()%4!=0 {return Err("invalid saved logits extent".into());}
        let v=bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        if v.iter().any(|v|!v.is_finite()) {return Err("nonfinite saved logits".into());}
        let mut best=0;for i in 1..v.len() {if v[i]>v[best] {best=i;}}
        if id as usize!=best {return Err("saved ID differs from complete-row argmax".into());}
    }Ok(output)
}
fn compare(a:&Path,b:&Path)->Result<()> {
    let left=verify(a)?;let right=verify(b)?;let count=left.len().min(right.len());
    if left[..count]!=right[..count] {return Err("generated token IDs differ".into());}
    same(a,b,"INPUT_IDS")?;same(a,b,"PROMPT_LOGITS")?;
    for at in 0..count {same(a,b,&format!("{at}-logits.f32"))?;}
    println!("RESIDENT COMPARE PASS tokens={count} complete_logits=true prompt_logits=true ids={:?}",&left[..count]);Ok(())
}
fn open_checkpoint(root:&Path)->Result<engine_model::plan::admission::CheckpointAdmission> {
    use engine_model::{plan::ConfigPlan,safetensors::read_bounded,shapes::{CONFIG_BYTES,HEADER_BYTES}};
    let config=read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|e|e.to_string())?;
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&config).map_err(|e|e.to_string())? else {return Err("resident probe requires a deferred recipe".into());};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|e|e.to_string())?;
    description.open_headers(root,&index).map_err(|e|e.to_string())
}
fn plan_time(root:&Path,repeats:usize)->Result<()> {
    use engine_model::plan::{layer::{LayerPlan,Topology},state::StatePlan};
    use spark_tp::layer_driver::LayerBoundary;
    if repeats==0 || repeats>10 {return Err("plan timing repeat bound exceeded".into());}
    let c=open_checkpoint(root)?;let d=c.description();let topology=Topology {rank:0,parts:1};
    for repeat in 1..=repeats {
        let started=std::time::Instant::now();
        for layer in 0..d.layers.len() {let state=StatePlan::new(d,topology,64,isize::MAX as usize).map_err(|e|e.to_string())?;
            std::hint::black_box(LayerBoundary::new(d,c.headers(),&state,layer,1,topology,usize::MAX)?);}
        let before=started.elapsed().as_secs_f64();let started=std::time::Instant::now();
        for layer in 0..d.layers.len() {std::hint::black_box(LayerPlan::new(d,layer,1,topology).map_err(|e|e.to_string())?);}
        let after=started.elapsed().as_secs_f64();
        println!("RESIDENT PLAN TIME repeat={repeat} layers={} prior_seconds={before:.9} direct_seconds={after:.9} saved_seconds={:.9} gpu=false",d.layers.len(),before-after);
    }Ok(())
}
#[cfg(any(target_os="linux",test))]
fn profile_options(args:&[String])->Result<(bool,bool,Vec<usize>)> {
    let (mut cost,mut gpu,mut rows)=(false,false,None);
    for arg in args {match arg.as_str() {
        "--cost" if !cost=>cost=true,
        "--gpu-cost" if !gpu=>gpu=true,
        value if value.starts_with("--prompt-rows=") && rows.is_none()=>{
            let parsed=value.trim_start_matches("--prompt-rows=").split(',').map(|value|value.parse::<usize>().ok().filter(|&n|n>0 && n<=spark_tp::bounded_decoder::PROMPT_ROWS).ok_or_else(||"diagnostic prompt row bound exceeded".to_string())).collect::<Result<Vec<_>>>()?;
            if parsed.len()>spark_tp::bounded_decoder::PROMPT_ROWS {return Err("diagnostic prompt schedule bound exceeded".into());}rows=Some(parsed);
        },
        _=>return Err("unknown or duplicate resident profiling option".into()),
    }}Ok((cost,gpu,rows.unwrap_or_else(||vec![spark_tp::bounded_decoder::PROMPT_ROWS])))
}
#[cfg(test)]
mod profile_tests {
    use super::profile_options;
    #[test]
    fn diagnostics_are_explicit_default_off_and_reject_duplicates() {
        assert_eq!(profile_options(&[]).unwrap(),(false,false,vec![spark_tp::bounded_decoder::PROMPT_ROWS]));
        for flags in [vec!["--cost"],vec!["--gpu-cost"],vec!["--gpu-cost","--cost"]] {
            let args=flags.iter().map(|v|(*v).to_owned()).collect::<Vec<_>>();
            assert_eq!(profile_options(&args).unwrap(),(flags.contains(&"--cost"),flags.contains(&"--gpu-cost"),vec![spark_tp::bounded_decoder::PROMPT_ROWS]));
        }
        for rows in 1..=spark_tp::bounded_decoder::PROMPT_ROWS {assert_eq!(profile_options(&[format!("--prompt-rows={rows}")]).unwrap(),(false,false,vec![rows]));}
        assert_eq!(profile_options(&["--prompt-rows=4,1".to_string()]).unwrap(),(false,false,vec![4,1]));
        for args in [vec!["--prompt-rows="],vec!["--prompt-rows=1,2,3,4,1"],vec!["--prompt-rows=0"],vec!["--prompt-rows=5"],vec!["--prompt-rows=1","--prompt-rows=4"]] {assert!(profile_options(&args.iter().map(|s|s.to_string()).collect::<Vec<_>>()).is_err());}
        for flags in [vec!["--cost","--cost"],vec!["--gpu-cost","--gpu-cost"],vec!["unknown"]] {
            assert!(profile_options(&flags.iter().map(|v|(*v).to_owned()).collect::<Vec<_>>()).is_err());
        }
    }
}
#[cfg(target_os="linux")]
fn run_pair(args:&[String])->Result<()> {
    use spark_serve::{Tokenizer,protocol::{self,DraftMode,Prompt}};
    use spark_gpu_probe::driver::with_backend;
    use spark_tp::{bounded_decoder::{Device,Decoder,Compute},transport::Ring};
    if args.len()<10 {return Err("usage: resident-decode-check RANK 2 LISTEN NEXT pair MODEL INPUT_IDS REQUEST_JSON NEW_OUT REPEATS".into());}
    let (cost,gpu_cost,prompt_groups)=profile_options(&args[10..])?;
    let root=Path::new(&args[5]);let input=ids(Path::new(&args[6]))?;let request=read(Path::new(&args[7]))?;let out=Path::new(&args[8]);
    let rank=args[0].parse::<usize>().map_err(|e|e.to_string())?;let listen=args[2].parse().map_err(|_|"invalid listen")?;let next=args[3].parse().map_err(|_|"invalid next")?;
    if args[1].parse::<usize>().map_err(|e|e.to_string())?!=spark_tp::layer_transport::PAIR_RANKS {return Err("resident pair probe requires its declared pair size".into());}
    let repeats=args[9].parse::<usize>().map_err(|e|e.to_string())?;
    if input.len()>512 || repeats==0 || repeats>10 {return Err("resident probe bounds exceeded".into());}
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(root)?;
    let limits=spark_serve::Limits {input_tokens:512,output_tokens:256,context_tokens:768,default_thinking:false,next_mtp_depth:Some(0),..Default::default()};
    let r=protocol::parse(&request,true,&limits)?;
    if !r.ignore_eos || r.cache || r.draft!=Some(DraftMode::Plain) || r.mtp_depth!=Some(0)
        || r.sampling.temperature!=0.0 || r.sampling.top_k!=0 || r.sampling.top_p!=1.0 || r.sampling.min_p!=0.0
        || r.sampling.repetition_penalty!=1.0 || r.sampling.presence_penalty!=0.0 || r.sampling.frequency_penalty!=0.0 {return Err("resident probe requires the baseline plain greedy request".into());}
    let Prompt::Chat(chat)=&r.prompt else {return Err("resident probe requires chat".into());};
    if tokenizer.encode_chat(chat,limits.body_bytes)?!=input {return Err("prepared input differs from the checkpoint chat template".into());}
    let checkpoint=open_checkpoint(root)?;
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let mut identity=format!("prompt_groups={prompt_groups:?}\nrequest={}\ninput_ids={input:?}\nkernels={}\n",sha256::digest(&request),sha256::digest(spark_gpu_probe::layer_ptx()?.to_bytes()));
    for name in ["config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json"] {identity.push_str(&format!("{name}={}\n",sha256::digest_reader(fs::File::open(root.join(name)).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?));}
    write(&out.join("IDENTITY"),identity.as_bytes())?;
    let mut ring=Ring::connect(rank,2,listen,next)?;let digest=sha256::digest(identity.as_bytes());let mut agreement=vec![0;digest.len()*ring.world()];
    let start=rank*digest.len();agreement[start..start+digest.len()].copy_from_slice(digest.as_bytes());ring.all_gather(&mut agreement,digest.len())?;
    if !agreement.chunks_exact(digest.len()).all(|v|v==digest.as_bytes()) {return Err("pair request, checkpoint metadata, or GPU program differs".into());}
    with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
        let compute=Device::with_ring_profiled(backend,Some(ring),cost);
        for repeat in 1..=repeats {let prompt_rows=prompt_groups[(repeat-1)%prompt_groups.len()];backend.set_profiling(gpu_cost && repeat>1);let saved=out.join(format!("run{repeat}"));fs::create_dir(&saved).map_err(|e|e.to_string())?;
            write(&saved.join("INPUT_IDS"),input.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;
            let decoder=Decoder::new(&checkpoint,&compute,(input.len()+r.max_tokens) as u64)?;let mut state=decoder.zero_state()?;let mut last=None;let mut hashes=String::new();compute.cost_report("setup");let prompt_started=std::time::Instant::now();
            let mut at=0;for tokens in input.chunks(prompt_rows) {let started=std::time::Instant::now();let (rows,next)=decoder.prefill(tokens,&state)?;state=next;
                for logits in rows {hashes.push_str(&format!("prompt-{at}={}\n",sha256::digest(&floats(&logits)?)));last=Some(logits);at+=1;}
                eprintln!("RESIDENT PROMPT GROUP rank={rank} repeat={repeat} rows={} end={at} seconds={:.9}",tokens.len(),started.elapsed().as_secs_f64());}
            let prompt_seconds=prompt_started.elapsed().as_secs_f64();write(&saved.join("PREFILL_TIMING"),format!("prompt_rows={prompt_rows} prompt_tokens={} prompt_seconds={prompt_seconds:.9} seconds_per_prompt_token={:.9}\n",input.len(),prompt_seconds/input.len() as f64).as_bytes())?;
            compute.cost_report("prefill");
            let mut logits=last.ok_or("empty prompt")?;let mut output=Vec::new();let started=std::time::Instant::now();
            for at in 0..r.max_tokens {let bytes=floats(&logits)?;let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
                output.push(best as u32);write(&saved.join(format!("{at}-logits.f32")),&bytes)?;
                eprintln!("RESIDENT TOKEN rank={rank} repeat={repeat} at={at} id={best} elapsed_seconds={:.6}",started.elapsed().as_secs_f64());
                if at+1<r.max_tokens {let (next,_,updated)=decoder.step(best as u32,&state,|_,_|Ok(()))?;logits=next;state=updated;}
            }
            backend.synchronize()?;let seconds=started.elapsed().as_secs_f64();let steps=r.max_tokens-1;
            compute.cost_report("decode");
            let timing=format!("decode_steps={steps} decode_seconds={seconds:.9} decode_tokens_per_second={:.9}\n",steps as f64/seconds);
            write(&saved.join("TIMING"),timing.as_bytes())?;write(&saved.join("PROMPT_LOGITS"),hashes.as_bytes())?;
            write(&saved.join("OUTPUT_IDS"),output.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;
            let mut decoder=tokenizer.decoder();let mut text=String::new();for &id in &output {text.push_str(&decoder.push(id).map_err(|e|e.to_string())?);}text.push_str(&decoder.finish());write(&saved.join("TEXT"),text.as_bytes())?;
            write(&saved.join("COMPLETE"),b"RESIDENT DECODE RUN PASS\n")?;
            println!("RESIDENT DECODE PASS rank={rank} repeat={repeat} tokens={} {timing}ids={output:?} text={text:?}",output.len());
            if repeat>1 {compare(&out.join("run1"),&saved)?;}
        }Ok(())
    })?;write(&out.join("COMPLETE"),b"RESIDENT DECODE SUITE PASS\n")
}
#[cfg(target_os="linux")]
fn layout(out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,causal_ops::{Shape,Op}};
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let input=[1.0,-0.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0];
    let expected=[1.0f32,-0.0,6.0,7.0,11.0,12.0,16.0,17.0,0.0,0.0,0.0,0.0,
        3.0,4.0,5.0,8.0,9.0,10.0,13.0,14.0,15.0,18.0,19.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0];
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let source=backend.upload_vector(&input)?;let s=Shape {op:Op::UnpackKv,heads:2,dim:2,taps:2,rows:3,cols:3,epsilon:0.0,bound:0.0};
        for repeat in 1..=3 {
            let result=backend.causal_operation(s,[Some(&source),None,None,None],None)?;let got=backend.read_vector_values(&result)?;
            if got.iter().map(|v|v.to_bits()).collect::<Vec<_>>()!=expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>() {return Err("device KV layout differs from the explicit row fixture".into());}
            write(&out.join(format!("run{repeat}.f32")),&floats(&got)?)?;
            let zero=backend.causal_operation(Shape {taps:0,..s},[None,None,None,None],None)?;
            if backend.read_vector_values(&zero)?.iter().any(|v|v.to_bits()!=0) {return Err("device KV empty selection is not positive-zero padding".into());}
        }Ok(())
    })?;write(&out.join("COMPLETE"),b"KV LAYOUT DEVICE PASS repeats=3 unequal_key_value_widths=true signed_zero=true padding=true\n")?;
    println!("KV LAYOUT DEVICE PASS repeats=3 unequal_key_value_widths=true signed_zero=true padding=true");Ok(())
}
#[cfg(target_os="linux")]
fn quantizer(out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,causal_ops::{Shape,Op},dynamic_mxfp8::Activation};
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut input=Vec::new();let mut seed=0x1d3a5b79u32;
    for exponent in 0..255u32 {let mut row=[0.0;32];for v in &mut row {seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);*v=f32::from_bits((seed&0x807fffff)|(exponent<<23));}
        if Activation::from_f32(&row).is_ok() {input.extend(row);}}
    input.extend([0.0;32]);input.extend([-0.0;32]);let mut edge=[0.0;32];edge[..8].copy_from_slice(&[448.0,1.0625,1.1875,-1.0625,-0.0,1.0/1024.0,3.0/1024.0,450.0]);input.extend(edge);
    let expected=Activation::from_f32(&input)?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {let source=backend.upload_vector(&input)?;
        for repeat in 1..=3 {for plane in 0..3 {let s=Shape {op:Op::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:plane,cols:input.len() as u32,epsilon:0.0,bound:0.0};
            let output=backend.causal_operation(s,[Some(&source),None,None,None],None)?;let got=backend.read_vector_values(&output)?;
            for (i,&v) in got.iter().enumerate() {let wanted=match plane {1=>expected.values()[i] as f32,2=>expected.scales()[i/32] as f32,_=>expected.decoded()[i]};
                if v.to_bits()!=wanted.to_bits() {return Err(format!("MXFP8 quantizer differs repeat={repeat} plane={plane} lane={i} device={:08x} host={:08x}",v.to_bits(),wanted.to_bits()));}}
            let hash=sha256::digest(&floats(&got)?);receipt.push_str(&format!("repeat={repeat} plane={plane} lanes={} sha256={hash} exact=true\n",input.len()));
        }}Ok(())
    })?;write(&out.join("RECEIPT"),receipt.as_bytes())?;println!("MXFP8 QUANTIZER DEVICE PASS repeats=3 lanes={} decoded_bits=true codes=true scales=true",input.len());Ok(())
}
#[cfg(target_os="linux")]
fn stream(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|b| {
        let words=64*1024*1024;let passes=32;
        let host=(0..words).map(|i|f32::from_bits(0x3f000000|(i as u32&0x7fffff))).collect::<Vec<_>>();
        let input=b.upload_vector(&host)?;
        for _ in 0..3 {let value=b.round_bf16_lanes(&input)?;b.synchronize()?;std::hint::black_box(value);}
        b.set_profiling(true);b.synchronize()?;let before=b.stats();let started=std::time::Instant::now();let mut last=None;
        for _ in 0..passes {let value=b.round_bf16_lanes(&input)?;b.synchronize()?;last=Some(value);}
        let wall=started.elapsed().as_nanos();let after=b.stats().since(before);b.set_profiling(false);
        let gpu=after.device_kernel_ns[..14].iter().sum::<u128>();if gpu==0 {return Err("stream probe has no GPU duration".into());}
        let values=b.read_vector_values(last.as_ref().ok_or("stream probe has no output")?.vector())?;
        for (&source,&value) in host.iter().zip(&values) {let bits=source.to_bits();let expected=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
            if value.to_bits()!=expected {return Err("stream probe round differs".into());}}
        let bytes=words as u128*8*passes;let text=format!("STREAM PASS words={words} passes={passes} bytes={bytes} gpu_ns={gpu} wall_ns={wall} gpu_GB_per_second={:.9} wall_GB_per_second={:.9}\n",bytes as f64/gpu as f64,bytes as f64/wall as f64);
        write(&out.join("TIMING"),text.as_bytes())?;print!("{text}");Ok(())
    })
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let result=match args.as_slice() {
        [mode,root,request,out] if mode=="prepare"=>prepare(Path::new(root),Path::new(request),Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="nvfp4"=>nvfp4::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="prompt-rows"=>prompt_rows::run(Path::new(out)),
        [mode,a,b] if mode=="compare"=>compare(&PathBuf::from(a),&PathBuf::from(b)),
        [mode,root,repeats] if mode=="plan-time"=>repeats.parse::<usize>().map_err(|e|e.to_string()).and_then(|n|plan_time(Path::new(root),n)),
        #[cfg(target_os="linux")]
        [_,_,_,_,mode,..] if mode=="pair"=>run_pair(&args),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="quantizer"=>quantizer(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="layout"=>layout(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="stream"=>stream(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="recurrence"=>recurrence::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="convolution"=>convolution::run(Path::new(out)),
        _=>Err("usage: resident-decode-check RANK 2 LISTEN NEXT pair MODEL INPUT_IDS REQUEST_JSON NEW_OUT REPEATS | quantizer NEW_OUT | layout NEW_OUT | compare LEFT RIGHT | plan-time MODEL REPEATS".into()),
    };if let Err(error)=result {eprintln!("RESIDENT CHECK FAIL {error}");std::process::exit(1);}
}
