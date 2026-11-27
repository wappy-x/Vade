//! Host tests of the start: the options, the places, the refusals, what the start reads from a model folder, and
//! one whole start on the host backend that serves a chat with text.
use super::*;
use crate::{bounded_decoder::Host,loop_decoder::service::{admit,tests::{CHAT_CONTEXT,Folder,chat_model_with,direct_at,health,ids,ok,post}}};
use engine_model::plan::loop_load::LoadBytes;
use std::{io::{Read,Write},net::TcpStream,sync::atomic::Ordering,time::Duration};

fn words(text:&str)->Vec<String> {text.split_whitespace().map(str::to_owned).collect()}
/// The places of a start on one Spark, then `options`.
fn start(context:&str,bind:&str,options:&str)->Vec<String> {words(&format!("0 1 127.0.0.1:32112 127.0.0.1:32112 flash.ptx model {context} {bind} 3 {options}"))}

#[test]
fn the_options_are_read_one_time_each_and_others_are_refused_with_the_list() {
    assert_eq!(options(&[]).unwrap(),Options::default());
    let all=options(&words("--mtp 0 --diagnostic-unqualified-text --no-prompt-keep --api-key-file key --diagnostic-host-arithmetic")).unwrap();
    assert_eq!(all,Options {diagnostic:true,host:true,pair:false,key:Some("key".into()),no_prompt_keep:true,depth:0,disk_cache_dir:None,disk_cache_bytes:None});
    let disk=options(&words("--diagnostic-unqualified-text --disk-cache-dir /tmp/vade-cache --disk-cache-bytes 1048576")).unwrap();
    assert_eq!(disk,Options {diagnostic:true,host:false,pair:false,key:None,no_prompt_keep:false,depth:0,disk_cache_dir:Some("/tmp/vade-cache".into()),disk_cache_bytes:Some(1_048_576)});
    assert_eq!(options(&words("--disk-cache-bytes 1024")).unwrap_err(),"--disk-cache-bytes requires --disk-cache-dir");
    assert_eq!(options(&words("--no-prompt-keep --disk-cache-dir /tmp/c")).unwrap_err(),"--disk-cache-dir cannot be combined with --no-prompt-keep");
    assert!(options(&words("--disk-cache-dir /tmp/c --disk-cache-bytes 0")).unwrap_err().contains("--disk-cache-bytes needs a positive byte count"));
    assert_eq!(options(&words("--disk-cache-dir")).unwrap_err(),"--disk-cache-dir needs a path");
    assert_eq!(options(&words("--disk-cache-bytes")).unwrap_err(),"--disk-cache-bytes needs a byte count");
    for (given,named) in [("--unknown","--unknown"),("--diagnostic-unqualified-text --diagnostic-unqualified-text","--diagnostic-unqualified-text"),
        ("--diagnostic-host-arithmetic --diagnostic-host-arithmetic","--diagnostic-host-arithmetic"),("--no-prompt-keep --no-prompt-keep","--no-prompt-keep"),
        ("--disk-cache-dir a --disk-cache-dir b","--disk-cache-dir"),("--disk-cache-bytes 1 --disk-cache-bytes 2","--disk-cache-bytes"),
        ("--mtp 0 --mtp 0","--mtp"),("--api-key-file a --api-key-file b","--api-key-file"),("--diagnostic-serving-trace","--diagnostic-serving-trace")] {
        let error=options(&words(given)).unwrap_err();
        assert!(error.starts_with(&format!("unsupported text service option: {named}; ")) && error.ends_with(OPTIONS),"{given}: {error}");
    }
    for depth in ["x","-1"] {let error=options(&words(&format!("--mtp {depth}"))).unwrap_err();assert!(error.starts_with("--mtp needs a nonnegative depth") && error.ends_with(OPTIONS),"{error}");}
    let keep=options(&words("--prompt-keep")).unwrap_err();assert!(keep.starts_with("unsupported text service option: --prompt-keep") && keep.ends_with(OPTIONS),"{keep}");
    assert_eq!(options(&words("--mtp 7")).unwrap().depth,7);
    assert_eq!(options(&words("--mtp")).unwrap_err(),"--mtp needs a depth");assert_eq!(options(&words("--api-key-file")).unwrap_err(),"--api-key-file needs a path");
    for name in [DIAGNOSTIC,HOST_ARITHMETIC,"--mtp DEPTH","--no-prompt-keep","--disk-cache-dir PATH","--disk-cache-bytes BYTES","--api-key-file PATH"] {assert!(OPTIONS.contains(name),"{name}");}
}
#[test]
fn the_places_are_read_and_a_start_that_cannot_be_served_is_refused() {
    // Starts cleanly without --diagnostic-unqualified-text.
    let plain=settings(&start("4096","127.0.0.1:18191","--mtp 0"),None).unwrap();
    assert_eq!((plain.rank,plain.world,plain.context,plain.bind,plain.requests,plain.output),(0,1,4096,"127.0.0.1:18191".parse().unwrap(),3,4096));
    assert_eq!(plain.options,Options::default());
    // Legacy --diagnostic-unqualified-text stays accepted as a backward-compatible no-op.
    let s=settings(&start("4096","127.0.0.1:18191","--diagnostic-unqualified-text --mtp 0"),None).unwrap();
    assert_eq!((s.rank,s.world,s.context,s.bind,s.requests,s.output),(0,1,4096,"127.0.0.1:18191".parse().unwrap(),3,4096));
    assert_eq!(s.options,Options {diagnostic:true,..Options::default()});
    // The output cap: the setting of the caller, else the context up to 8,192 tokens.
    let wide=|output:Option<&str>|settings(&start("32768","[::1]:18191",""),output.map(str::to_owned));
    assert_eq!((wide(None).unwrap().output,wide(Some("512")).unwrap().output),(8192,512));
    for output in ["0","many"] {assert!(wide(Some(output)).unwrap_err().contains("FLASH_SERVE_OUTPUT_TOKENS must be a positive token count"));}
    // Non-loopback HTTP binds are respected cleanly, with or without the legacy flag.
    for bind in ["0.0.0.0:18191","[::]:18191","192.0.2.1:18191"] {
        assert_eq!(settings(&start("4096",bind,""),None).unwrap().bind,bind.parse::<SocketAddr>().unwrap());
        assert_eq!(settings(&start("4096",bind,"--diagnostic-unqualified-text"),None).unwrap().bind,bind.parse::<SocketAddr>().unwrap());
    }
    assert!(settings(&start("0","127.0.0.1:18191",""),None).unwrap_err().contains("CONTEXT must be a positive token count"));
    assert!(settings(&start("many","127.0.0.1:18191",""),None).unwrap_err().contains("CONTEXT must be a whole number, got many"));
    let short=settings(&words("0 1 127.0.0.1:1 127.0.0.1:1 flash.ptx model 4096 127.0.0.1:18191"),None).unwrap_err();
    assert!(short.starts_with(PLACES) && short.ends_with(OPTIONS),"{short}");
}
/// The service has no transport: a place with a neighbor link is refused, at any size of the layout.
#[test]
fn the_spare_bound_holds_one_step_and_the_chats_of_a_full_cohort() {
    const GB:usize=1<<30;
    // 37 chats of 2 GB in a 74 GB chat pool, 7 GB a step: one step and 32 chats.
    assert_eq!(spares(7*GB,2*GB,37,74*GB),71*GB);
    // Fewer chats than a full cohort: one step and all of them.
    assert_eq!(spares(7*GB,2*GB,8,16*GB),23*GB);
    // The chats never count for more than the chat pool, and no chat leaves the step.
    assert_eq!(spares(7*GB,2*GB,32,40*GB),47*GB);
    assert_eq!(spares(7*GB,2*GB,0,74*GB),7*GB);
    assert_eq!(spares(usize::MAX,2*GB,32,74*GB),usize::MAX);
}
#[test]
fn a_place_with_a_neighbor_is_refused() {
    assert!(alone(0,1).is_ok());
    for (place,size,links) in [(0,2,1),(1,2,1),(0,4,1),(1,4,2),(3,4,1),(2,8,2)] {
        let error=alone(place,size).unwrap_err();
        assert!(error.starts_with("this text service has no transport between Sparks") && error.contains(&format!("has {links} neighbor links")) && error.ends_with("start it as RANK 0 of WORLD 1"),"{place} of {size}: {error}");
    }
    // A place outside its layout is refused too.
    assert!(alone(1,1).is_err() && alone(0,0).is_err());
    let mut pair=start("4096","127.0.0.1:18191","--diagnostic-unqualified-text");pair[1]="2".into();
    assert!(settings(&pair,None).unwrap_err().contains("no transport between Sparks"));
}
/// What the start reads from a model folder: the stop set is the EOS list of the checkpoint, and the config, the
/// generation metadata and the tokenizer must agree on it.
#[test]
fn the_stop_set_is_the_eos_list_of_the_checkpoint() {
    let folder=Folder::new("start-text");
    let read=text(&folder.root,&chat_model_with(folder.stop).description).unwrap();
    assert_eq!(read.eos,[folder.stop]);assert!(read.tokenizer.token_channels());
    // The config of the model names another EOS token than the generation metadata of the folder.
    assert_eq!(text(&folder.root,&chat_model_with(1).description).err().unwrap(),"service and tokenizer EOS sets differ");
    // A folder whose template is not the checked one of this family has no reply reader for it.
    std::fs::write(folder.root.join("chat_template.jinja"),"{{ messages }}").unwrap();
    assert!(text(&folder.root,&chat_model_with(folder.stop).description).is_err());
}
/// ONE WHOLE START on the host backend: the settings from a command line, the model folder, the memory rule, and
/// the serving loop. A chat with text is answered with the token IDs of the direct decode, and the health record
/// names the path and the backend that run.
#[test]
fn a_whole_start_serves_a_chat_with_text_on_the_host_backend() {
    let folder=Folder::new("start-serve");let fixture=chat_model_with(folder.stop);let d=&fixture.description;
    let settings=settings(&start(&CHAT_CONTEXT.to_string(),"127.0.0.1:0","--mtp 0 --no-prompt-keep --diagnostic-host-arithmetic"),Some("24".into())).unwrap();
    assert!(settings.options.host);
    let admitted=admit(d,&LoadBytes {device:1<<20,..Default::default()},settings.context,64<<20,4,super::super::tests::LIMITS).unwrap();
    let listener=TcpListener::bind(settings.bind).unwrap();let address=listener.local_addr().unwrap();
    let body=r#"{"messages":[{"role":"user","content":"Hello"}],"max_tokens":6,"temperature":0,"stream":true,"ignore_eos":true,"stream_options":{"include_usage":true}}"#;
    let read=text(&folder.root,d).unwrap();
    let request=spark_serve::protocol::parse(body.as_bytes(),true,&limits(settings.context,settings.output).unwrap()).unwrap();
    let spark_serve::protocol::Prompt::Chat(input)=&request.prompt else {panic!("the request is a chat")};
    let prompt=spark_serve::Tokenizer::encode_chat(&read.tokenizer,input,1<<20).unwrap();
    let expected=direct_at(&fixture,settings.context,&prompt,6);
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (stop,settings,admitted,root,fixture)=(&stop,&settings,&admitted,folder.root.as_path(),&fixture);let _stop=Stop(stop);
        scope.spawn(move ||serve(&Host,fixture,&fixture.description,read,admitted,settings,listener,root,"host",stop).unwrap());
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        let record=health(&send,address);
        assert!(record.contains("\"status\":\"ready\"") && record.contains("\"path\":\"loop_driver\"") && record.contains("\"backend\":\"host\"") && record.contains("\"qualification\":\"qualified\""),"{record}");
        for run in 0..2 {
            let reply=post(&send,address,"/v1/chat/completions",body);assert!(ok(&reply),"{reply}");
            let served=ids(&reply);assert!(!served.is_empty(),"the answer holds no token ID: {reply}");
            assert_eq!(served,expected,"run {run}: the served token IDs differ from the direct decode of the rendered prompt: {reply}");
        }
    });
}
/// At long contexts (`>= 131,072`) `serve_with_draft` raises the tokenizer's input byte limit to match
/// `limits.tokenizer_limits()`, so prompts above the default 1 MiB tokenizer cap are tokenized instead of refused.
#[test]
fn serve_raises_tokenizer_input_byte_limit_past_one_mib() {
    let folder=Folder::new("start-long-tok");
    let config=crate::loop_decoder::fixture::SMALL
        .replace(r#""vocab_size":24,"unpadded_vocab_size":20"#,r#""vocab_size":320,"unpadded_vocab_size":312"#)
        .replace(r#""model_max_length":64"#,r#""model_max_length":600000"#)
        .replace(r#""eos_token_id":1,"#,&format!(r#""eos_token_id":{},"#,folder.stop));
    let fixture=crate::loop_decoder::fixture::Fixture::new(&config).unwrap();
    let d=&fixture.description;
    let settings=settings(&words("0 1 127.0.0.1:32112 127.0.0.1:32112 flash.ptx model 600000 127.0.0.1:0 1 --diagnostic-unqualified-text --mtp 0 --no-prompt-keep --diagnostic-host-arithmetic"),Some("16384".into())).unwrap();
    let admitted=admit(d,&LoadBytes {device:1<<20,..Default::default()},settings.context,256<<20,4,super::super::tests::LIMITS).unwrap();
    let listener=TcpListener::bind(settings.bind).unwrap();let address=listener.local_addr().unwrap();
    let read=text(&folder.root,d).unwrap();
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (stop,settings,admitted,root,fixture)=(&stop,&settings,&admitted,folder.root.as_path(),&fixture);let _stop=Stop(stop);
        scope.spawn(move ||serve(&Host,fixture,&fixture.description,read,admitted,settings,listener,root,"host",stop).unwrap());
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        let word=format!("{} ","ab".repeat(32));
        let text_payload=word.repeat(17_000);
        assert!(text_payload.len()>1_100_000);
        let body=format!(r#"{{"prompt":"{text_payload}"}}"#);
        let reply=post(&send,address,"/tokenize",&body);
        assert!(ok(&reply) && !reply.contains("input byte limit exceeded") && reply.contains("\"count\":561000"),"{}",&reply[..reply.len().min(300)]);
    });
}
/// A DEVICE START WITHOUT THE KERNEL IMAGE STOPS AT THE START, NOT AT THE FIRST CHAT: one clear line that names the
/// build command. With the image, the start goes on.
#[test]
fn a_build_without_the_kernel_image_is_refused_at_the_start() {
    let refusal=need_kernel_image(false).unwrap_err();
    assert_eq!(refusal,NO_KERNEL_IMAGE);
    assert!(!refusal.contains('\n') && refusal.contains("cargo kernels-inkling"),"{refusal}");
    need_kernel_image(true).unwrap();
}
/// Weights whose every product fails, as a device step fails when the build has no kernel image the plan needs.
struct Failing<'f>(&'f crate::loop_decoder::fixture::Fixture);
impl crate::loop_decoder::mix::Weights<Host> for Failing<'_> {
    type RoutePlan = ();
    fn words(&self,weight:crate::loop_decoder::mix::Weight)->Result<&[f32]> {crate::loop_decoder::mix::Weights::<Host>::words(self.0,weight)}
    fn product(&self,_:&Host,_:crate::loop_decoder::mix::Weight,_:&<Host as crate::bounded_decoder::Compute>::Value)->Result<<Host as crate::bounded_decoder::Compute>::Value> {
        Err("test: this step cannot run".into())
    }
}
/// A FAILED GENERATION REACHES THE CLIENT AS AN ERROR. A chat whose first step fails gets, without streaming, an error
/// status with an error object (type, code, message); with streaming, an error event before the stream ends and no
/// normal finish (no finish reason). Both the chat route and the completions route. The cause stays in the log.
#[test]
fn a_failed_generation_reaches_the_client_as_an_error() {
    let folder=Folder::new("start-fail");let fixture=chat_model_with(folder.stop);let d=&fixture.description;
    // Four requests: the start serves 8, then stops.
    let settings=settings(&words(&format!("0 1 127.0.0.1:32112 127.0.0.1:32112 flash.ptx model {CHAT_CONTEXT} 127.0.0.1:0 8 --diagnostic-unqualified-text --mtp 0 --no-prompt-keep --diagnostic-host-arithmetic")),Some("24".into())).unwrap();
    let admitted=admit(d,&LoadBytes {device:1<<20,..Default::default()},settings.context,64<<20,4,super::super::tests::LIMITS).unwrap();
    let listener=TcpListener::bind(settings.bind).unwrap();let address=listener.local_addr().unwrap();
    let read=text(&folder.root,d).unwrap();let failing=Failing(&fixture);
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (stop,settings,admitted,root,failing)=(&stop,&settings,&admitted,folder.root.as_path(),&failing);let _stop=Stop(stop);
        scope.spawn(move ||serve(&Host,failing,d,read,admitted,settings,listener,root,"host",stop).unwrap());
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=Vec::new();
            // A reset after the reply keeps the bytes that came: the test reads what the client would read.
            let end=stream.read_to_end(&mut reply);let mut reply=String::from_utf8_lossy(&reply).into_owned();
            if let Err(error)=end {reply.push_str(&format!("\n[READ ERROR {error}]"));}reply};
        let chat=r#""messages":[{"role":"user","content":"Hello"}],"max_tokens":6,"temperature":0"#;
        let ids=r#""prompt":[1,2,3],"max_tokens":6,"temperature":0"#;
        for (path,fields) in [("/v1/chat/completions",chat),("/v1/completions",ids)] {
            let whole=post(&send,address,path,&format!("{{{fields}}}"));
            assert!(!ok(&whole),"{path} without streaming: a failed generation answered with a success: {whole}");
            let body=&whole[whole.find("\r\n\r\n").expect("a reply has a head")+4..];
            for field in ["\"error\"","\"type\"","\"code\"","\"message\""] {assert!(body.contains(field),"{path} without streaming: the error object has no {field}: {whole}");}
            assert!(!body.contains("finish_reason"),"{path} without streaming: a failed generation has a finish reason: {whole}");
            let streamed=post(&send,address,path,&format!(r#"{{{fields},"stream":true,"stream_options":{{"include_usage":true}}}}"#));
            eprintln!("FAILED GENERATION {path} without streaming: {whole}\nFAILED GENERATION {path} with streaming: {streamed}");
            let error=streamed.find("\"error\"").unwrap_or_else(||panic!("{path} with streaming: no error event: {streamed}"));
            if let Some(done)=streamed.find("[DONE]") {assert!(error<done,"{path} with streaming: the error event comes after the end: {streamed}");}
            for finish in ["\"finish_reason\":\"stop\"","\"finish_reason\":\"length\""] {assert!(!streamed.contains(finish),"{path} with streaming: a failed generation has a normal finish: {streamed}");}
            assert!(!streamed.contains("this step cannot run"),"{path}: the cause of the failure reached the client: {streamed}");
        }
    });
}
/// THE REAL MODEL, SERVED ON THE HOST BACKEND. Ignored by default: the model folder is private and stays outside the
/// tracked tree, as the link `.local/loop-host/model` at the repository root (the folder of the host tests of the
/// model crate). The chat is the first-token prompt of the reference set: one user message of one token. The
/// request names no sampling field. The served token IDs must equal the 9 tokens that the whole-model host form
/// gives after that prompt (its saved trace: the top rows of positions 15 to 23), and the rendered prompt must be
/// the 16 IDs of the trace. The test reads no environment variable. Its lines (`REAL`) are the receipt.
#[test]
#[ignore = "requires the real model folder outside the repository"]
fn the_real_model_serves_the_first_token_prompt_on_the_host_backend() {
    const PROMPT:[u32;16]=[200002,200004,34182,279,154554,150628,25,220,15,13,24,200010,200000,200004,32269,200010];
    const ANSWER:[u32;9]=[200001,200008,371,150409,150239,150330,196980,32269,1];
    let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host/model");
    assert!(root.exists(),"put the model folder in .local/loop-host/ for the ignored host tests");
    let settings=settings(&start("256","127.0.0.1:0","--diagnostic-unqualified-text --diagnostic-host-arithmetic"),None).unwrap();
    let body=r#"{"messages":[{"role":"user","content":"Hello"}],"max_tokens":9,"stream":true,"stream_options":{"include_usage":true}}"#;
    let listener=TcpListener::bind(settings.bind).unwrap();let address=listener.local_addr().unwrap();
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (stop,settings,root)=(&stop,&settings,root.as_path());let _stop=Stop(stop);
        scope.spawn(move ||serve_host(root,settings,48<<30,spark_gpu_probe::causal_ops::BUCKET_ROWS as u64,listener,stop).unwrap());
        let send=|text:String| {
            // The port answers when the weights are read.
            let mut stream=(0..600).find_map(|_|TcpStream::connect(address).ok().or_else(|| {std::thread::sleep(Duration::from_millis(500));None})).expect("the service opened no port");
            stream.set_read_timeout(Some(Duration::from_secs(3600))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        let record=health(&send,address);
        assert!(record.contains("\"path\":\"loop_driver\"") && record.contains("\"backend\":\"host\""),"{record}");
        let began=Instant::now();
        let reply=post(&send,address,"/v1/chat/completions",body);assert!(ok(&reply),"{reply}");
        let (served,seconds)=(ids(&reply),began.elapsed().as_secs_f64());
        eprintln!("REAL served_ids={served:?} seconds={seconds:.3}");
        let usage=reply.split("\"prompt_tokens\":").nth(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>());
        eprintln!("REAL prompt_tokens={usage:?} expected={} reply_bytes={}",PROMPT.len(),reply.len());
        assert_eq!(usage.as_deref(),Some("16"),"the rendered prompt is not the 16 tokens of the reference prompt: {reply}");
        assert_eq!(served,ANSWER,"the served token IDs differ from the host form's: {reply}");
        eprintln!("REAL EQUAL served token IDs equal the whole-model host form: {} of {}",served.len(),ANSWER.len());
        // The start answers 3 requests and stops: this is the third.
        assert!(health(&send,address).contains("\"status\":\"ready\""));
    });
}
/// THE REFERENCE CHATS RENDER TO THE REFERENCE PROMPTS. Ignored by default: the files are private and stay outside
/// the tracked tree, under `.local/loop-host/` at the repository root: `model` is the model folder, `ref` the
/// reference folder, and `bodies` a folder of chat request files named `<number>-<prompt>__<effort>.json`. Each
/// request is read by the protocol reader of the service and rendered by the chat template of the model folder.
/// The token IDs must equal `sets/contract-A/<prompt>__<effort>/INPUT_IDS` of the reference folder, ID for ID.
/// A request file with no reference prompt of its name is not compared. The lines `RENDER` are the receipt.
#[test]
#[ignore = "requires the real model folder, the reference set and the request files outside the repository"]
fn the_reference_chat_requests_render_to_the_reference_prompt_ids() {
    let local=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host");
    let (model,set,bodies)=(local.join("model"),local.join("ref/sets/contract-A"),local.join("bodies"));
    assert!(model.exists() && set.exists() && bodies.exists(),"put model, ref and bodies in .local/loop-host/ for this ignored test");
    let tokenizer=ModelTokenizer::from_directory(&model).unwrap();
    let limits=limits(32768,8192).unwrap();
    let mut names=std::fs::read_dir(&bodies).unwrap().map(|entry|entry.unwrap().file_name().to_string_lossy().into_owned()).filter(|name|name.ends_with(".json")).collect::<Vec<_>>();
    names.sort();
    let (mut equal,mut compared)=(0,0);
    for name in &names {
        let prompt=name.trim_end_matches(".json").split_once('-').expect("a request file starts with its number").1;
        let reference=set.join(prompt).join("INPUT_IDS");
        if !reference.exists() {eprintln!("RENDER {name}: no reference prompt of this name, not compared");continue;}
        let want=std::fs::read_to_string(&reference).unwrap().split_whitespace().map(|id|id.parse::<u32>().unwrap()).collect::<Vec<_>>();
        let request=spark_serve::protocol::parse(&std::fs::read(bodies.join(name)).unwrap(),true,&limits).unwrap();
        let spark_serve::protocol::Prompt::Chat(input)=&request.prompt else {panic!("{name} is no chat request")};
        let got=spark_serve::Tokenizer::encode_chat(&tokenizer,input,1<<20).unwrap();
        compared+=1;equal+=usize::from(got==want);
        eprintln!("RENDER {name} tokens={} reference={} equal={}",got.len(),want.len(),got==want);
    }
    eprintln!("RENDER SUMMARY equal={equal} of={compared} request_files={}",names.len());
    assert!(compared>0 && equal==compared,"{equal} of {compared} rendered prompts equal the reference prompts");
}
/// THE REAL MODEL ANSWERS REQUEST FILES ON THE HOST BACKEND. Ignored by default: the files are private and stay
/// under `.local/loop-host/` at the repository root: `model` is the model folder and `cpu-bodies` a folder of chat
/// request files. Each file is sent to one start of the host run, in name order, one chat at a time. The test
/// compares nothing: each line `REAL-CHAT` gives the name of the file, the finish reason and the served token IDs,
/// for a compare with the answers of the same requests on a device. The test reads no environment variable.
#[test]
#[ignore = "requires the real model folder and the request files outside the repository"]
fn the_real_model_answers_request_files_on_the_host_backend() {
    let local=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host");
    let (root,bodies)=(local.join("model"),local.join("cpu-bodies"));
    assert!(root.exists() && bodies.exists(),"put model and cpu-bodies in .local/loop-host/ for this ignored test");
    let mut names=std::fs::read_dir(&bodies).unwrap().map(|entry|entry.unwrap().file_name().to_string_lossy().into_owned()).filter(|name|name.ends_with(".json")).collect::<Vec<_>>();
    names.sort();assert!(!names.is_empty());
    // The start answers the health request, each chat, and one last health request.
    let mut places=start("1024","127.0.0.1:0","--diagnostic-unqualified-text --diagnostic-host-arithmetic");places[8]=(names.len()+2).to_string();
    let settings=settings(&places,None).unwrap();
    let listener=TcpListener::bind(settings.bind).unwrap();let address=listener.local_addr().unwrap();
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (stop,settings,root)=(&stop,&settings,root.as_path());let _stop=Stop(stop);
        scope.spawn(move ||serve_host(root,settings,48<<30,spark_gpu_probe::causal_ops::BUCKET_ROWS as u64,listener,stop).unwrap());
        let send=|text:String| {
            let mut stream=(0..600).find_map(|_|TcpStream::connect(address).ok().or_else(|| {std::thread::sleep(Duration::from_millis(500));None})).expect("the service opened no port");
            stream.set_read_timeout(Some(Duration::from_secs(7200))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        assert!(health(&send,address).contains("\"backend\":\"host\""));
        for name in &names {
            let body=std::fs::read_to_string(bodies.join(name)).unwrap();
            let began=Instant::now();
            let reply=post(&send,address,"/v1/chat/completions",&body);assert!(ok(&reply),"{name}: {reply}");
            let finish=reply.rsplit("\"finish_reason\":\"").next().map(|part|part.chars().take_while(char::is_ascii_lowercase).collect::<String>()).unwrap_or_default();
            let served=ids(&reply).iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
            eprintln!("REAL-CHAT {name} finish={finish} seconds={:.1} ids={served}",began.elapsed().as_secs_f64());
        }
        assert!(health(&send,address).contains("\"status\":\"ready\""));
    });
}

