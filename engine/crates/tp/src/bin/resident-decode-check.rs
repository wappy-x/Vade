//! Bounded resident row-split decoder evidence. Public service admission stays separate.
#![forbid(unsafe_code)]
#[cfg(target_os="linux")]
#[path="resident-decode-check/fp8_warp.rs"] mod fp8_warp;
#[cfg(target_os="linux")]
#[path="resident-decode-check/fp8_all_codes.rs"] mod fp8_all_codes;
#[cfg(target_os="linux")]
#[path="resident-decode-check/routing.rs"] mod routing;
use std::{fs,path::{Path,PathBuf}};
use spark_tp::Result;
#[cfg(target_os="linux")]
#[path="resident-decode-check/recurrence.rs"] mod recurrence;
#[cfg(target_os="linux")]
#[path="resident-decode-check/program.rs"] mod program;
#[cfg(target_os="linux")]
#[path="resident-decode-check/recorder.rs"] mod recorder;
#[cfg(target_os="linux")]
#[path="resident-decode-check/workspace.rs"] mod workspace;
#[cfg(target_os="linux")]
#[path="resident-decode-check/cache_append.rs"] mod cache_append;
#[cfg(target_os="linux")]
#[path="resident-decode-check/index_append.rs"] mod index_append;
#[cfg(target_os="linux")]
#[path="resident-decode-check/memo_slab.rs"] mod memo_slab;
#[cfg(target_os="linux")]
#[path="resident-decode-check/attention.rs"] mod attention;
#[cfg(target_os="linux")]
#[path="resident-decode-check/convolution.rs"] mod convolution;
#[cfg(target_os="linux")]
#[path="resident-decode-check/nvfp4.rs"] mod nvfp4;
#[cfg(target_os="linux")]
#[path="resident-decode-check/prompt_rows.rs"] mod prompt_rows;
#[cfg(target_os="linux")]
#[path="resident-decode-check/softmax_rows.rs"] mod softmax_rows;
#[cfg(target_os="linux")]
#[path="resident-decode-check/forced_union.rs"] mod forced_union;
#[cfg(target_os="linux")]
#[path="resident-decode-check/projection_batch.rs"] mod projection_batch;
#[cfg(target_os="linux")]
#[path="resident-decode-check/device_route.rs"] mod device_route;
#[cfg(target_os="linux")]
#[path="resident-decode-check/exchange_stage.rs"] mod exchange_stage;
#[cfg(target_os="linux")]
#[path="resident-decode-check/bus_shadow.rs"] mod bus_shadow;
#[cfg(target_os="linux")]
#[path="resident-decode-check/argmax.rs"] mod argmax;
#[cfg(target_os="linux")]
#[path="resident-decode-check/fill_free.rs"] mod fill_free;
#[cfg(target_os="linux")]
#[path="resident-decode-check/put_wide.rs"] mod put_wide;
#[cfg(target_os="linux")]
#[path="resident-decode-check/select_known.rs"] mod select_known;
#[cfg(target_os="linux")]
#[path="resident-decode-check/nv_group_rows.rs"] mod nv_group_rows;
#[cfg(target_os="linux")]
#[path="resident-decode-check/nv_b.rs"] mod nv_b;
#[cfg(target_os="linux")]
#[path="resident-decode-check/load_weights.rs"] mod load_weights;
#[cfg(target_os="linux")]
#[path="../../../../tools/export/src/sha256.rs"] mod sha256;
/// Diagnostic prompt bound. Long-prompt evidence needs more than the old 512 rows.
#[cfg(target_os="linux")]
const PROMPT_BOUND:usize=33000;
fn read(path:&Path)->Result<Vec<u8>> {fs::read(path).map_err(|e|e.to_string())}
#[cfg(target_os="linux")]
fn write(path:&Path,bytes:&[u8])->Result<()> {use std::io::Write;
    fs::OpenOptions::new().create_new(true).write(true).open(path).and_then(|mut f|f.write_all(bytes)).map_err(|e|e.to_string())}
fn ids(path:&Path)->Result<Vec<u32>> {let text=fs::read_to_string(path).map_err(|e|e.to_string())?;
    let values=text.split_whitespace().map(|v|v.parse().map_err(|_|"invalid token ID".into())).collect::<Result<Vec<_>>>()?;
    if values.is_empty() {return Err("empty token IDs".into());}Ok(values)}
#[cfg(target_os="linux")]
fn floats(v:&[f32])->Result<Vec<u8>> {if v.iter().any(|v|!v.is_finite()) {return Err("nonfinite saved logits".into());}Ok(v.iter().flat_map(|v|v.to_le_bytes()).collect())}
fn same(a:&Path,b:&Path,name:&str)->Result<()> {if read(&a.join(name))?!=read(&b.join(name))? {return Err(format!("saved {name} differs"));}Ok(())}
fn verify(path:&Path)->Result<Vec<u32>> {
    let output=ids(&path.join("OUTPUT_IDS"))?;
    for (at,&id) in output.iter().enumerate() {let bytes=read(&path.join(format!("{at}-logits.f32")))?;
        if bytes.is_empty() || bytes.len()%4!=0 {return Err("invalid saved logits extent".into());}
        let v=bytes.as_chunks::<4>().0.iter().map(|v|f32::from_le_bytes(*v)).collect::<Vec<_>>();
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
    // The distinct row counts of the model's two-dimensional weights and their halves (rank-split matrices): the totals and local rows a prompt transpose can see.
    {let mut rows=std::collections::BTreeSet::new();for (_,w) in d.required_weights().map_err(|e|e.to_string())? {if w.shape.len()==2 {rows.insert(w.shape[0] as usize);if w.shape[0]%2==0 {rows.insert(w.shape[0] as usize/2);}}}
        println!("MODEL MATRIX ROWS count={} values={:?} gpu=false",rows.len(),rows);}
    for context in [4096usize,8192,32768] {println!("RESIDENT PROMPT LEDGER context={context} group_rows={} group_workspace_bytes={} expanded_bytes={} gpu=false",spark_tp::bounded_decoder::PROMPT_GROUP,spark_tp::bounded_decoder::group_workspace_bytes(&c,context)?,spark_tp::bounded_decoder::expanded_bytes(&c,context));}
    for context in [32768usize,65536,65544] {
        let copies=spark_tp::service_memory::state_copy_bytes(&c)?;
        let (legacy_session,legacy_workspace)=spark_tp::row_split::service_charges(&c,context,true)?;
        println!("GROUP-B LEGACY context={context} session={legacy_session} workspace={legacy_workspace} copies_if_large={copies} gpu=false");
        let prompt=spark_tp::bounded_decoder::RESERVED_ROWS+1;let tokens=prompt+64;
        let state=StatePlan::new(d,topology,tokens as u64,isize::MAX as usize).map_err(|e|e.to_string())?;
        let before=2*state.state_layout().bytes()+2*spark_tp::bounded_decoder::expanded_bytes(&c,prompt)+d.vocab as usize*16;
        let after=before.max(spark_tp::service_memory::session(&c,prompt,tokens)?);
        println!("GROUP-B REFUSAL context={context} prompt={prompt} output_allowance=64 old_request={before} new_request={after} example_session_pool={} old_two_fit=true new_two_fit={} gpu=false",2*before,2*after<=2*before);

        for group in [256,512,1024] {let (session,workspace)=spark_tp::row_split::prompt_group_service_charges(&c,context,true,group)?;println!("GROUP-B LEDGER context={context} group={group} session={session} workspace={workspace} copies={} gpu=false",if group>256 {copies} else {0});}
    }
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
#[cfg(target_os="linux")]
fn prepare(root:&Path,request:&Path,out:&Path)->Result<()> {
    use spark_serve::{Tokenizer,protocol::{self,Prompt}};
    let limits=spark_serve::Limits {input_tokens:PROMPT_BOUND,output_tokens:256,context_tokens:PROMPT_BOUND+256,default_thinking:false,next_mtp_depth:Some(0),..Default::default()};
    let request=protocol::parse(&read(request)?,true,&limits)?;let Prompt::Chat(chat)=&request.prompt else {return Err("resident probe requires chat".into());};
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(root)?;let input=tokenizer.encode_chat(chat,limits.body_bytes)?;
    if input.is_empty() || input.len()>limits.input_tokens || input.len()+request.max_tokens>limits.context_tokens {return Err("prepared resident prompt exceeds bounds".into());}
    write(out,input.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;
    println!("RESIDENT PREPARE PASS input_tokens={} output_tokens={} gpu=false",input.len(),request.max_tokens);Ok(())
}
#[cfg(any(target_os="linux",test))]
type Options=(bool,bool,Vec<usize>,Option<(Vec<usize>,usize)>,Vec<usize>,bool,Option<usize>,usize,Option<(usize,usize,usize)>,Vec<(usize,u8)>,usize,usize);
#[cfg(any(target_os="linux",test))]
fn forced_case(value:&str)->Result<(usize,usize,usize)> {
    let parts=value.split(':').map(|v|v.parse::<usize>().map_err(|_|"invalid forced union extent".to_string())).collect::<Result<Vec<_>>>()?;
    let case=match parts.as_slice() {[budget]=>(*budget,256,1),[budget,group,chats]=>(*budget,*group,*chats),_=>return Err("forced union needs CEILING[:GROUP:CHATS]".into())};
    if case.0==0 || ![256,512,1024].contains(&case.1) || ![1,8].contains(&case.2) {return Err("forced union group or session bound exceeded".into());}Ok(case)
}
#[cfg(any(target_os="linux",test))]
fn profile_options(args:&[String])->Result<Options> {
    let (mut cost,mut gpu,mut rows)=(false,false,None);let mut groups=None;let mut last=0usize;let mut windows=Vec::new();let mut wire=false;let mut limit=None;let mut decode_rows=0usize;let mut forced=None;let mut verify=Vec::new();let mut propose=0usize;let mut capacity=0usize;
    for arg in args {match arg.as_str() {
        "--cost" if !cost=>cost=true,
        "--gpu-cost" if !gpu=>gpu=true,
        value if value.starts_with("--prompt-rows=") && rows.is_none()=>{
            let parsed=value.trim_start_matches("--prompt-rows=").split(',').map(|value|value.parse::<usize>().ok().filter(|&n|n>0 && n<=spark_tp::bounded_decoder::PROMPT_ROWS).ok_or_else(||"diagnostic prompt row bound exceeded".to_string())).collect::<Result<Vec<_>>>()?;
            if parsed.len()>spark_tp::bounded_decoder::PROMPT_ROWS {return Err("diagnostic prompt schedule bound exceeded".into());}rows=Some(parsed);
        },
        value if value.starts_with("--row-groups=") && groups.is_none()=>{
            groups=Some(value.trim_start_matches("--row-groups=").split(',').map(|value|value.parse::<usize>().ok().filter(|&n|n>0 && n<=spark_tp::bounded_decoder::PROMPT_GROUP_MAX).ok_or_else(||"diagnostic row group bound exceeded".to_string())).collect::<Result<Vec<_>>>()?);
        },
        "--device-wire" if !wire=>wire=true,
        // Decode through the row path, one row per group. Same bits, for timing only.
        // Decode through the row path, for comparison. The server decodes by step.
        "--decode-rows" if decode_rows==0=>decode_rows=1,
        value if value.starts_with("--decode-rows-from=") && decode_rows==0=>{decode_rows=value.trim_start_matches("--decode-rows-from=").parse::<usize>().ok().filter(|&k|k>0).ok_or("invalid decode rows session")?;},
        // Bits must not depend on where expanded rows stop persisting.
        value if value.starts_with("--expanded-limit=") && limit.is_none()=>{limit=Some(value.trim_start_matches("--expanded-limit=").parse::<usize>().map_err(|_|"invalid expanded limit")?);},
        // Worst-case union of selected rows at this device ceiling. Memory only: no bytes compared.
        value if value.starts_with("--forced-union=") && forced.is_none()=>{forced=Some(forced_case(value.trim_start_matches("--forced-union="))?);},
        // Sessions after the first decode in verify passes of this many rows. Proposals are the first
        // session's own tokens: all right, or wrong from a row that moves with each pass.
        // --verify-mix: one row and this many rows by turns, every proposal right.
        value if value.starts_with("--verify-rows=") || value.starts_with("--verify-miss=") || value.starts_with("--verify-mix=")=>{let miss=if value.starts_with("--verify-miss=") {1u8} else if value.starts_with("--verify-mix=") {2} else {0};
            for rows in value[value.find('=').unwrap()+1..].split(',') {verify.push((rows.parse::<usize>().ok().filter(|&n|n>0 && n<=spark_tp::bounded_decoder::VERIFY_ROWS).ok_or("invalid verify row count")?,miss));}
            if verify.len()>9 {return Err("verify schedule bound exceeded".into());}},
        // Count how many of the predictor layer's proposals equal the plain continuation. The
        // decode itself stays plain: nothing is verified or skipped.
        value if value.starts_with("--propose=") && propose==0=>{propose=value[10..].parse::<usize>().ok().filter(|n|(1..=3).contains(n)).ok_or("invalid proposal depth")?;},
        // The decoder's context capacity, as a server sets it (default: prompt plus outputs). Bits must not depend on it.
        value if value.starts_with("--capacity=") && capacity==0=>{capacity=value[11..].parse::<usize>().ok().filter(|n|*n>0).ok_or("invalid capacity")?;},
        "--head-last" if last==0=>last=1,
        // Sessions from K on run the head only on each group's last row, as a server does.
        value if value.starts_with("--head-last-from=") && last==0=>{last=value.trim_start_matches("--head-last-from=").parse::<usize>().ok().filter(|&k|k>0).ok_or("invalid head session")?;},
        // Scalar-path cost rows at chosen positions. Row groups carry the prompt between them.
        value if value.starts_with("--cost-windows=") && windows.is_empty()=>{
            windows=value.trim_start_matches("--cost-windows=").split(',').map(|value|value.parse::<usize>().map_err(|_|"invalid cost window".to_string())).collect::<Result<Vec<_>>>()?;
            if windows.is_empty() || windows.windows(2).any(|w|w[1]<w[0]+8) {return Err("cost windows must ascend by at least 8 rows".into());}
        },
        _=>return Err("unknown or duplicate resident profiling option".into()),
    }}if last>0 && groups.is_none() {return Err("--head-last requires --row-groups".into());}
    if !windows.is_empty() && (groups.is_none() || last==1) {return Err("--cost-windows requires --row-groups with all head rows".into());}
    Ok((cost,gpu,rows.unwrap_or_default(),groups.map(|g|(g,last)),windows,wire,limit,decode_rows,forced,verify,propose,capacity))
}
#[cfg(test)]
mod profile_tests {
    use super::profile_options;
    #[test]
    fn job_list_is_bounded_and_keeps_each_jobs_options() {
        assert!(super::job_rows("# comment\n  # indented comment\n\n").unwrap().is_empty());
        let jobs=super::job_rows("a.ids a.json a-out 3\n  b.ids b.json b-out 1 --row-groups=64 --fork-check\n").unwrap();
        assert_eq!(jobs.len(),2);assert_eq!(jobs[0],("a.ids".to_owned(),"a.json".to_owned(),"a-out".to_owned(),"3".to_owned(),vec![],false));
        assert_eq!((jobs[1].4.clone(),jobs[1].5),(vec!["--row-groups=64".to_owned()],true));
        assert!(super::job_rows("a b c").is_err());
        assert!(super::job_rows("a b c 1 --cost").is_err());assert!(super::job_rows("a b c 1 --device-wire").is_err());
        assert!(super::job_rows(&"a b c 1\n".repeat(15)).is_ok());assert!(super::job_rows(&"a b c 1\n".repeat(16)).is_err());
    }
    #[test]
    fn diagnostics_are_explicit_default_off_and_reject_duplicates() {
        assert_eq!(profile_options(&[]).unwrap(),(false,false,vec![],None,vec![],false,None,0,None,vec![],0,0));
        assert_eq!(profile_options(&["--verify-rows=1,16".to_owned(),"--verify-miss=4".to_owned()]).unwrap().9,vec![(1,0),(16,0),(4,1)]);assert_eq!(profile_options(&["--propose=3".to_owned()]).unwrap().10,3);assert!(profile_options(&["--propose=4".to_owned()]).is_err());assert_eq!(profile_options(&["--verify-mix=4".to_owned()]).unwrap().9,vec![(4,2)]);
        for bad in ["--verify-rows=0","--verify-rows=17","--verify-miss=","--verify-rows=1,1,1,1,1,1,1,1,1,1"] {assert!(profile_options(&[bad.to_owned()]).is_err(),"{bad}");}
        for flags in [vec!["--cost"],vec!["--gpu-cost"],vec!["--gpu-cost","--cost"]] {
            let args=flags.iter().map(|v|(*v).to_owned()).collect::<Vec<_>>();
            assert_eq!(profile_options(&args).unwrap(),(flags.contains(&"--cost"),flags.contains(&"--gpu-cost"),vec![],None,vec![],false,None,0,None,vec![],0,0));
        }
        for rows in 1..=spark_tp::bounded_decoder::PROMPT_ROWS {assert_eq!(profile_options(&[format!("--prompt-rows={rows}")]).unwrap(),(false,false,vec![rows],None,vec![],false,None,0,None,vec![],0,0));}
        assert_eq!(profile_options(&["--prompt-rows=4,1".to_string()]).unwrap(),(false,false,vec![4,1],None,vec![],false,None,0,None,vec![],0,0));
        assert_eq!(profile_options(&["--row-groups=64,256".to_string(),"--head-last".to_string()]).unwrap().3,Some((vec![64,256],1)));
        assert!(profile_options(&["--head-last".to_string()]).is_err());assert!(profile_options(&["--row-groups=1025".to_string()]).is_err());
        for args in [vec!["--prompt-rows="],vec!["--prompt-rows=1,2,3,4,1"],vec!["--prompt-rows=0"],vec!["--prompt-rows=5"],vec!["--prompt-rows=1","--prompt-rows=4"]] {assert!(profile_options(&args.iter().map(|s|s.to_string()).collect::<Vec<_>>()).is_err());}
        for flags in [vec!["--cost","--cost"],vec!["--gpu-cost","--gpu-cost"],vec!["unknown"]] {
            assert!(profile_options(&flags.iter().map(|v|(*v).to_owned()).collect::<Vec<_>>()).is_err());
        }
    }
}
/// One job list line: INPUT_IDS REQUEST_JSON NEW_OUT REPEATS, then options for that job.
/// Cost and wire options are set once, on the command line.
#[cfg(any(target_os="linux",test))]
fn job_rows(text:&str)->Result<Vec<(String,String,String,String,Vec<String>,bool)>> {
    let mut rows=Vec::new();
    for line in text.lines().map(str::trim).filter(|line|!line.is_empty() && !line.starts_with('#')) {
        let words=line.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        if words.len()<4 {return Err("job line needs INPUT_IDS REQUEST_JSON NEW_OUT REPEATS".into());}
        let flags=words[4..].iter().filter(|a|*a!="--fork-check").cloned().collect::<Vec<_>>();
        if flags.iter().any(|a|a=="--cost" || a=="--gpu-cost" || a=="--device-wire") {return Err("cost and wire options are set once, on the command line".into());}
        rows.push((words[0].clone(),words[1].clone(),words[2].clone(),words[3].clone(),flags,words[4..].iter().any(|a|a=="--fork-check")));
        if rows.len()>15 {return Err("resident job list exceeds bound".into());}
    }Ok(rows)
}
#[cfg(target_os="linux")]
fn run_pair(args:&[String])->Result<()> {
    use spark_serve::{Tokenizer,protocol::{self,DraftMode,Prompt}};
    use spark_gpu_probe::driver::with_backend;
    use spark_tp::{bounded_decoder::{Device,Decoder,Compute,HeadRows},transport::Ring};
    if args.len()<10 {return Err("usage: resident-decode-check RANK 2 LISTEN NEXT pair MODEL INPUT_IDS REQUEST_JSON NEW_OUT REPEATS".into());}
    // One load serves every job: the positional prompt first, then each line of --jobs=FILE.
    // --queue-per-step: sessions after the first open and close the resident queue around each
    // plain decode step, as the served session does. Same bits; for timing.
    let queue_per_step=args[10..].iter().any(|a|a=="--queue-per-step");
    let check_greedy=args[10..].iter().any(|a|a=="--check-greedy-head");
    // Proof run: each skipped fill stores all ones, so a read of a word no writer stored changes the logits.
    if args[10..].iter().any(|a|a=="--poison-skipped-fills") {spark_gpu_probe::fill_free::set_mode(spark_gpu_probe::fill_free::Mode::Poison);}
    let is_harness=|a:&&String|*a!="--fork-check" && *a!="--queue-per-step" && *a!="--check-greedy-head" && *a!="--poison-skipped-fills" && !a.starts_with("--jobs=");
    let options=profile_options(&args[10..].iter().filter(is_harness).cloned().collect::<Vec<_>>())?;let (cost,gpu_cost,device_wire)=(options.0,options.1,options.5);
    let root=Path::new(&args[5]);
    let rank=args[0].parse::<usize>().map_err(|e|e.to_string())?;let listen=args[2].parse::<std::net::SocketAddr>().map_err(|_|"invalid listen")?;let next=args[3].parse::<std::net::SocketAddr>().map_err(|_|"invalid next")?;
    if args[1].parse::<usize>().map_err(|e|e.to_string())?!=spark_tp::layer_transport::PAIR_RANKS {return Err("resident pair probe requires its declared pair size".into());}
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(root)?;
    let limits=spark_serve::Limits {input_tokens:PROMPT_BOUND,output_tokens:256,context_tokens:PROMPT_BOUND+256,default_thinking:false,next_mtp_depth:Some(0),..Default::default()};
    let checkpoint=open_checkpoint(root)?;
    let mut specs=vec![(args[6].clone(),args[7].clone(),args[8].clone(),args[9].clone(),args[10..].iter().filter(is_harness).cloned().collect::<Vec<_>>(),args[10..].iter().any(|a|a=="--fork-check"))];
    let lists=args[10..].iter().filter_map(|a|a.strip_prefix("--jobs=")).collect::<Vec<_>>();
    if lists.len()>1 {return Err("job list may be named once".into());}
    if let Some(list)=lists.first() {specs.extend(job_rows(&String::from_utf8(read(Path::new(list))?).map_err(|e|e.to_string())?)?);}
    let mut identity=format!("check_greedy_head={check_greedy}\n");let mut jobs=Vec::new();
    for (input,request,out,repeats,flags,fork_check) in specs {
        let input=ids(Path::new(&input))?;let request=read(Path::new(&request))?;let repeats=repeats.parse::<usize>().map_err(|e|e.to_string())?;
        if input.len()>PROMPT_BOUND || repeats==0 || repeats>10 {return Err("resident probe bounds exceeded".into());}
        let r=protocol::parse(&request,true,&limits)?;
        if input.len()+r.max_tokens>limits.context_tokens {return Err("resident prompt and output exceed context bound".into());}
        if !r.ignore_eos || r.cache || r.draft!=Some(DraftMode::Plain) || r.mtp_depth!=Some(0)
            || r.sampling.temperature!=0.0 || r.sampling.top_k!=0 || r.sampling.top_p!=1.0 || r.sampling.min_p!=0.0
            || r.sampling.repetition_penalty!=1.0 || r.sampling.presence_penalty!=0.0 || r.sampling.frequency_penalty!=0.0 {return Err("resident probe requires the baseline plain greedy request".into());}
        let Prompt::Chat(chat)=&r.prompt else {return Err("resident probe requires chat".into());};
        if tokenizer.encode_chat(chat,limits.body_bytes)?!=input {return Err("prepared input differs from the checkpoint chat template".into());}
        let options=profile_options(&flags)?;let (prompt_groups,row_groups,windows)=(&options.2,&options.3,&options.4);
        identity.push_str(&format!("prompt_groups={prompt_groups:?}\nrow_groups={row_groups:?}\nwindows={windows:?}\nforced={:?}\nrequest={}\ninput_ids={input:?}\n",options.8,sha256::digest(&request)));
        jobs.push((input,r.max_tokens,PathBuf::from(out),repeats,options,fork_check));
    }
    if jobs.iter().any(|j|j.4.8.is_some()) && jobs.iter().any(|j|j.4.8.is_none()) {return Err("forced memory jobs must not share a load with decode timing jobs".into());}
    identity.push_str(&format!("kernels={}\n",sha256::digest(spark_gpu_probe::layer_ptx()?.to_bytes())));
    for name in ["config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json"] {identity.push_str(&format!("{name}={}\n",sha256::digest_reader(fs::File::open(root.join(name)).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?));}
    for job in &jobs {fs::create_dir(&job.2).map_err(|e|e.to_string())?;write(&job.2.join("IDENTITY"),identity.as_bytes())?;}
    let _cpu_pin=spark_tp::runtime::defer_cpu_pin();
    let mut ring=Ring::connect(rank,2,listen,next)?;let digest=sha256::digest(identity.as_bytes());let mut agreement=vec![0;digest.len()*ring.world()];
    let start=rank*digest.len();agreement[start..start+digest.len()].copy_from_slice(digest.as_bytes());ring.all_gather(&mut agreement,digest.len())?;
    if !agreement.chunks_exact(digest.len()).all(|v|v==digest.as_bytes()) {return Err("pair request, checkpoint metadata, or GPU program differs".into());}
    let initial_budget=jobs.iter().filter_map(|job|job.4.8.map(|case|case.0)).min().unwrap_or(usize::MAX);
    with_backend(0,spark_gpu_probe::layer_ptx()?,initial_budget,|backend| {
        let mailbox=if device_wire {Some(backend.tp_mailbox(4<<20)?)} else {None};
        let conductor=spark_tp::host_conductor::Conductor::new(backend,&checkpoint,3)?;
        let mut placement=spark_tp::thread_affinity::Placement::new(&mut ring,conductor.as_ref().map_or(0,|conductor|conductor.cores()))?;
        let wire=if let Some(m)=&mailbox {
            let port=|mut a:std::net::SocketAddr|->Result<std::net::SocketAddr> {
                a.set_port(a.port().checked_add(1).ok_or("resident wire port overflow")?);Ok(a)
            };
            use spark_gpu_probe::tp_route::{Links,Side};
            let links=Links::new(ring.world(),false)?;
            let left=links.neighbor(rank,Side::Left).map(|_|port(listen)).transpose()?;
            let right=links.neighbor(rank,Side::Right).map(|_|port(next)).transpose()?;
            let lead=left.is_none();
            let (incoming,outgoing)=spark_seqpar::control::connect_peers(left,right,
                std::time::Instant::now()+std::time::Duration::from_secs(300))?;
            let link=incoming.or(outgoing).ok_or("resident device wire has no peer")?;
            let wire=spark_tp::tensor_wire::Wire::Write(spark_tp::tensor_wire::WriteLink::new(&link)?);
            Some(wire.bind(m.take_host().ok_or("resident mailbox endpoint is taken")?,lead,m.layout())?)
        } else {None};
        use std::sync::atomic::{AtomicBool,Ordering};
        let (stop,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
        std::thread::scope(|scope| {
            let workers=placement.workers();
            let host_worker=conductor.as_ref().map(|conductor|conductor.spawn(scope)).transpose()?;
            let handle=wire.map(|worker|spark_tp::tensor_wire::spawn_on(scope,worker,&stop,&attached,&failed,placement.wire_cpus())).transpose()?;
            drop(workers);
            let result=(|| {
                placement.pin_decode();
                if let Some(handle)=&handle {spark_tp::tensor_wire::wait_attached(&attached,handle)?;}
                let mut queued=if device_wire {Some(backend.resident_queue()?)} else {None};
                let compute=if let Some(m)=&mailbox {Device::with_mailbox(backend,ring,cost,m)?}
                    else {Device::with_ring_profiled(backend,Some(ring),cost)};
                let compute=match &conductor {Some(conductor)=>compute.with_mhc_host(conductor.link())?,None=>compute};

        for (input,max_tokens,out,repeats,options,fork_check) in &jobs {
        let (input,max_tokens,out,repeats,fork_check)=(input.as_slice(),*max_tokens,out.as_path(),*repeats,*fork_check);
        let (_,_,prompt_groups,row_groups,windows,_,expanded_limit,decode_rows,forced,verify,propose,capacity)=options.clone();let mut reference:Option<Vec<u32>>=None;
        if check_greedy && (decode_rows>0 || forced.is_some() || !verify.is_empty() || propose>0) {return Err("compact head check requires plain scalar decode".into());}
        if let Some((budget,group,chats))=forced {forced_union::run(rank,&checkpoint,&compute,backend,input,budget,group,chats)?;write(&out.join("COMPLETE"),b"FORCED UNION PASS\n")?;continue;}
        for repeat in 1..=repeats {
            // No path option: the row path, as the server runs it.
            let auto=row_groups.is_none() && prompt_groups.is_empty();let prompt_rows=if auto {spark_tp::bounded_decoder::PROMPT_GROUP} else if prompt_groups.is_empty() {spark_tp::bounded_decoder::PROMPT_ROWS} else {prompt_groups[(repeat-1)%prompt_groups.len()]};backend.set_profiling(gpu_cost && repeat>1);let saved=out.join(format!("run{repeat}"));fs::create_dir(&saved).map_err(|e|e.to_string())?;
            write(&saved.join("INPUT_IDS"),input.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;
            let mut decoder=Decoder::new(&checkpoint,&compute,capacity.max(input.len()+max_tokens) as u64)?;if propose>0 {decoder.enable_predictor()?;}let decoder=decoder;let mut proposals:Vec<(usize,usize,Vec<u32>,f64)>=Vec::new();let mut scored:Vec<(usize,Vec<u32>,Vec<[f32;2]>)>=Vec::new();let mut seen=None;let mut seen_prompt=None;let mut prompt_tail=None;if let Some(rows)=expanded_limit {decoder.set_expanded_limit(rows);}let mut state=decoder.zero_state()?;let mut last=None;let mut hashes=String::new();compute.cost_report("setup");let prompt_started=std::time::Instant::now();
            let head_last=row_groups.as_ref().is_some_and(|(_,from)|*from>0 && repeat>=*from);
            let row_group=row_groups.as_ref().map(|(g,_)|(g[(repeat-1)%g.len()],if head_last {HeadRows::Last} else {HeadRows::All}));
            // Each group: its end, whether it runs the row path, and its report label.
            let mut plan=Vec::new();let mut start=0;let size=row_group.map_or(prompt_rows,|g|g.0);
            for &window in windows.iter().filter(|&&w|w+8<=input.len()) {
                while start<window {let end=(start+size).min(window);plan.push((end,true,None));start=end;}
                // The first scalar group warms the projection memo. The second one is the sample.
                plan.push((window+4,false,Some(format!("warm{window}"))));plan.push((window+8,false,Some(format!("window{window}"))));start=window+8;
            }
            while start<input.len() {let end=(start+size).min(input.len());plan.push((end,row_group.is_some(),None));start=end;}
            let mut at=0;let mut begin=0;for (end,rows_path,label) in plan {let tokens=&input[begin..end];begin=end;let started=std::time::Instant::now();
                let pool_counts=||{let s=backend.vector_pool_stats();(s.allocated_count,s.allocated_bytes,s.reused_count)};let pool=pool_counts();
                let whole=compute.rows_peak_begin();let (rows,next)=if auto && propose>0 {let (rows,next,open)=decoder.prefill_rows_hidden(tokens,&state,HeadRows::All)?;prompt_tail=Some((tokens.to_vec(),open));(rows,next)} else if auto {decoder.prefill_rows(tokens,&state,HeadRows::All)?} else if let (true,Some((_,head)))=(rows_path,row_group) {decoder.prefill_rows(tokens,&state,head)?} else {decoder.prefill(tokens,&state)?};
                // A second continuation of a published state must not touch the first one's rows.
                if fork_check && state.position>0 {let reversed=tokens.iter().rev().copied().collect::<Vec<_>>();let _=decoder.prefill_rows(&reversed,&state,HeadRows::Last)?;eprintln!("RESIDENT FORK rank={rank} from={} rows={}",state.position,reversed.len());}
                state=next;
                if rows.len()<tokens.len() {at+=tokens.len()-rows.len();}
                for logits in rows {hashes.push_str(&format!("prompt-{at}={}\n",sha256::digest(&floats(&logits)?)));last=Some(logits);at+=1;}
                eprintln!("RESIDENT PROMPT GROUP rank={rank} repeat={repeat} rows={} end={at} seconds={:.9} allocated_bytes={} peak_bytes={} {}",tokens.len(),started.elapsed().as_secs_f64(),backend.allocated_bytes(),compute.rows_peak_end(whole),spark_tp::bounded_decoder::group_allocation_fields(Some(pool),Some(pool_counts())));
                if let Some(label)=label {compute.cost_report(&label);}
                else if !windows.is_empty() {compute.cost_report("carry");}}
            let prompt_seconds=prompt_started.elapsed().as_secs_f64();write(&saved.join("PREFILL_TIMING"),format!("prompt_rows={} prompt_tokens={} prompt_seconds={prompt_seconds:.9} seconds_per_prompt_token={:.9}\n",row_group.map_or(prompt_rows,|g|g.0),input.len(),prompt_seconds/input.len() as f64).as_bytes())?;
            placement.refresh_cuda();
            compute.cost_report("prefill");println!("PROMPT KERNEL COUNTS rank={rank} repeat={repeat} tokens={} {}",input.len(),spark_gpu_probe::prompt_launches::line());
            #[cfg(feature="kernel-bench")] let indirect_before=compute.fixture_attention_calls();
            backend.reset_vector_pool_bound();
            let bell_counted=mailbox.as_ref().map(|m|m.doorbell_stats());let mut logits=last.ok_or("empty prompt")?;let mut output=Vec::new();let parameter_counted=compute.parameter_counts();let counted=backend.stats();let copy_counted=backend.resident_copy_counts();let selection_counted=compute.selection_counts();let ahead_counted=compute.ahead_counts();let route_counted=spark_gpu_probe::device_route::counts();let stage_counted=spark_tp::projection_stage::counts();let prompt_peak=backend.peak_accounted_bytes();let prompt_live=backend.accounted_bytes();let started=std::time::Instant::now();
            if let Some(&(width,miss))=repeat.checked_sub(2).and_then(|i|verify.get(i)) {
                let wanted=reference.as_ref().ok_or("verify passes need the plain session first")?;let vocab=logits.len() as u32;
                let keep=|logits:&[f32],output:&mut Vec<u32>|->Result<u32> {let at=output.len();let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
                    if wanted.get(at)!=Some(&(best as u32)) {return Err(format!("verify pass token {at} differs from plain decoding"));}
                    write(&saved.join(format!("{at}-logits.f32")),&floats(logits)?)?;output.push(best as u32);
                    eprintln!("RESIDENT TOKEN rank={rank} repeat={repeat} at={at} id={best} elapsed_seconds={:.6}",started.elapsed().as_secs_f64());Ok(best as u32)};
                let mut next=keep(&logits,&mut output)?;let mut open=spark_tp::bounded_decoder::Verify::new(state);let mut pass=0usize;
                while output.len()<max_tokens {let at=output.len();let rows=if miss==2 && pass.is_multiple_of(2) {1} else {width.min(max_tokens-at)};let kept=if miss==1 {1+pass%rows} else {rows};
                    let tokens=(0..rows).map(|i|if i==0 {next} else if i<kept {wanted[at+i-1]} else {(wanted[at+i-1]+1)%vocab}).collect::<Vec<_>>();
                    let (entry,carried,pass_started)=(open.position(),open.carried(),std::time::Instant::now());let pool=backend.vector_pool_stats();
                    let rows_logits=open.rows(&decoder,&tokens)?;open.truncate(&decoder,entry+kept as u64)?;let seconds=pass_started.elapsed().as_secs_f64();
                    for row in &rows_logits[..kept] {next=keep(row,&mut output)?;}
                    let now=backend.vector_pool_stats();
                    eprintln!("RESIDENT VERIFY rank={rank} repeat={repeat} at={at} rows={rows} carried={carried} kept={kept} seconds={seconds:.9} new_buffers={} new_bytes={} reused_buffers={} released_bytes={}",now.allocated_count-pool.allocated_count,now.allocated_bytes-pool.allocated_bytes,now.reused_count-pool.reused_count,now.evicted_bytes+now.direct_freed_bytes-pool.evicted_bytes-pool.direct_freed_bytes);pass+=1;}
            } else {
            let stepwise=queue_per_step && repeat>1 && device_wire;if stepwise {drop(queued.take());}
            // Device vectors allocated since the previous token line: at>0 counts the step that made these logits.
            let pool_count=||{let s=backend.vector_pool_stats();(s.allocated_count,s.allocated_bytes)};let mut allocations=pool_count();let mut captures=backend.stats().graph_captures;
            for at in 0..max_tokens {let bytes=floats(&logits)?;let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
                output.push(best as u32);write(&saved.join(format!("{at}-logits.f32")),&bytes)?;
                let now=pool_count();
                let captured=backend.stats().graph_captures;eprintln!("RESIDENT TOKEN rank={rank} repeat={repeat} at={at} id={best} elapsed_seconds={:.6} {} captures={}",started.elapsed().as_secs_f64(),spark_tp::bounded_decoder::allocation_fields(Some(allocations),Some(now)),captured-captures);allocations=now;captures=captured;
                if at+1<max_tokens {
                    if decode_rows>0 && repeat>=decode_rows {let (mut rows,updated)=decoder.prefill_rows(&[best as u32],&state,HeadRows::Last)?;logits=rows.pop().ok_or("decode row has no logits")?;state=updated;}
                    else if propose>0 {
                        // Fourth count: the layer's cache starts with the last prompt rows (at most 64), then the decoded positions.
                        if let Some((tokens,open))=prompt_tail.take() {let timer=std::time::Instant::now();let rows=tokens.len();let first=rows.saturating_sub(64);
                            for row in first..rows {let after=if row+1<rows {tokens[row+1]} else {best as u32};
                                seen_prompt=Some(decoder.propose_after(&decoder.open_hidden(&open,row)?[1],after,0,seen_prompt.as_ref())?.1);}
                            eprintln!("RESIDENT PROPOSER PROMPT ROWS rank={rank} repeat={repeat} rows={} seconds={:.6}",rows-first,timer.elapsed().as_secs_f64());}
                        let (next,chosen,updated,hidden)=decoder.step_hidden(best as u32,&state,|_,_|Ok(()))?;logits=next;state=updated;
                        // `chosen` is the token at index at+1, so its proposals are for index at+2 onward.
                        for (variant,h) in hidden.iter().enumerate() {let timer=std::time::Instant::now();let tokens=decoder.propose(h,chosen,propose)?;proposals.push((at+2,variant,tokens,timer.elapsed().as_secs_f64()));}
                        // Third count: after the norm, with the layer's own cache of the decoded positions.
                        let timer=std::time::Instant::now();let (tokens,next)=decoder.propose_after(&hidden[1],chosen,propose,seen.as_ref())?;seen=Some(next);proposals.push((at+2,2,tokens,timer.elapsed().as_secs_f64()));
                        {let timer=std::time::Instant::now();let (tokens,_,_)=decoder.propose_chain(&hidden[1],chosen,propose,seen_prompt.as_ref(),false)?;proposals.push((at+2,4,tokens,timer.elapsed().as_secs_f64()));}
                        let timer=std::time::Instant::now();let (tokens,scores,next)=decoder.propose_scored(&hidden[1],chosen,propose,seen_prompt.as_ref())?;seen_prompt=Some(next);scored.push((at+2,tokens.clone(),scores));proposals.push((at+2,3,tokens,timer.elapsed().as_secs_f64()));}
                    else {
                        let step=|| {let compact=if check_greedy {Some(decoder.step_greedy(best as u32,&state,|_,_|Ok(()))?)} else {None};
                            let (logits,selected,next)=decoder.step(best as u32,&state,|_,_|Ok(()))?;
                            if let Some((chosen,compact))=compact {if chosen!=selected {return Err("compact head ID differs from full logits".into());}Ok((logits,compact))} else {Ok((logits,next))}};
                        let (next,updated)=if queue_per_step && repeat>1 && device_wire {compute.work(step)?} else {step()?};logits=next;state=updated;
                    }
                }
            }
            if stepwise {queued=Some(backend.resident_queue()?);backend.enable_resident_launches()?;}
            if propose>0 {let mut report=String::new();
                for (variant,label) in ["before_final_norm","after_final_norm","after_final_norm_with_own_cache","after_final_norm_with_cache_from_prompt_tail","the_same_with_a_chain_that_passes_the_raw_output"].into_iter().enumerate() {
                    let rows=proposals.iter().filter(|p|p.1==variant).collect::<Vec<_>>();let ms=rows.iter().map(|p|p.3).sum::<f64>()*1000.0/rows.len().max(1) as f64;
                    for depth in 0..propose {let (mut trials,mut alone,mut chain)=(0,0,0);
                        for (at,_,tokens,_) in rows.iter().map(|p|&**p) {if at+depth>=output.len() {continue;}trials+=1;
                            alone+=usize::from(tokens[depth]==output[at+depth]);chain+=usize::from((0..=depth).all(|i|tokens[i]==output[at+i]));}
                        report.push_str(&format!("hidden={label} depth={} trials={trials} right_alone={alone} right_with_all_before={chain} ms_per_call_of_{propose}={ms:.3}\n",depth+1));}}
                // Each proposal of the served form with its two scores, for a threshold study.
                let mut lines=String::new();for (at,tokens,scores) in &scored {for depth in 0..tokens.len() {if at+depth>=output.len() {continue;}
                    lines.push_str(&format!("at={at} depth={} right={} chain={} margin={:.4} probability={:.5}\n",depth+1,u8::from(tokens[depth]==output[at+depth]),u8::from((0..=depth).all(|i|tokens[i]==output[at+i])),scores[depth][0],scores[depth][1]));}}
                write(&saved.join("PROPOSALS"),lines.as_bytes())?;
                eprint!("RESIDENT PROPOSER rank={rank} repeat={repeat}\n{report}");write(&saved.join("PROPOSER"),report.as_bytes())?;}
            if repeat==1 {reference=Some(output.clone());}}
            backend.synchronize()?;let seconds=started.elapsed().as_secs_f64();let steps=max_tokens-1;compute.cost_report("decode");
            println!("DECODE KERNEL COUNTS rank={rank} repeat={repeat} tokens={} {}",input.len(),spark_gpu_probe::prompt_launches::line());
            #[cfg(feature="kernel-bench")] write(&saved.join("INDIRECT"),format!("decode_steps={steps} accepted_calls={}\n",compute.fixture_attention_calls()-indirect_before).as_bytes())?;
            let timing=format!("decode_steps={steps} decode_seconds={seconds:.9} decode_tokens_per_second={:.9}\n",steps as f64/seconds);
            // The shared workspace funds POOL_ROOM for spares at token start.
            // Live outputs use scratch/session charges; the memo has its own cap.
            let (samples,spares,cap)=backend.vector_pool_bound();let charge=spark_tp::bounded_decoder::POOL_ROOM;
            let slab=compute.projection_slab_bytes();let memo_limit=compute.memo_counts().1;
            if steps>0 && samples==0 || spares>charge || slab>memo_limit {return Err(format!("decode pool bound failed: trim_samples={samples} spare_bytes={spares} charged_spare_bytes={charge} slab_bytes={slab} charged_memo_bytes={memo_limit}"));}
            let bound=format!("rank={rank} repeat={repeat} prompt_tokens={} decode_steps={steps} trim_samples={samples} pool_cap_bytes={cap} charged_spare_bytes={charge} peak_spare_bytes_after_trim={spares} slab_bytes={slab} charged_memo_bytes={memo_limit} verified=1 scope=decode_start_free_spares_and_memo_slabs\n",input.len());
            eprint!("RESIDENT POOL BOUND {bound}");write(&saved.join("POOL_BOUND"),bound.as_bytes())?;
            // Host submissions and forced waits of the decode steps, for the structure of one token.
            let parameters=compute.parameter_counts();write(&saved.join("PARAMETER_KEYS"),format!("decode_steps={steps} key_builds={} static_binds={} static_hits={} scope=decode_nonempty_parameter_content_keys_load_keys_excluded\n",parameters.key_builds-parameter_counted.key_builds,parameters.static_binds-parameter_counted.static_binds,parameters.static_hits-parameter_counted.static_hits).as_bytes())?;
            let counts=backend.stats().since(counted);write(&saved.join("COUNTS"),format!("decode_steps={steps} graph_replays={} graph_captures={} stream_waits={} host_launches={} peak_bytes_after_prompt={prompt_peak} live_bytes_after_prompt={prompt_live} peak_bytes_after_decode={} live_bytes_after_decode={} memo_bytes={} memo_limit={} memo_rows_found={} memo_rows_added={} {} dtoh_calls={} budget_drains={} selection_waits={} selection_known={} memo_answers_ahead={} memo_answers_ahead_used={}\n",counts.graph_launches,counts.graph_captures,counts.stream_wait_calls,counts.queued_calls,backend.peak_accounted_bytes(),backend.accounted_bytes(),compute.memo_counts().0,compute.memo_counts().1,compute.memo_counts().2,compute.memo_counts().3,backend.resident_copy_counts().since(copy_counted).line(),counts.dtoh_calls,counts.budget_drains,compute.selection_counts()[0]-selection_counted[0],compute.selection_counts()[1]-selection_counted[1],compute.ahead_counts()[0]-ahead_counted[0],compute.ahead_counts()[1]-ahead_counted[1]).as_bytes())?;
            #[cfg(feature="kernel-bench")] write(&saved.join("ATTENTION_DISPATCH"),format!("decode_steps={steps} scores_table={} values_table={} scope=accepted_commands_completed_by_final_sync\n",counts.attention_dispatches[0],counts.attention_dispatches[1]).as_bytes())?;
            let route=spark_gpu_probe::device_route::counts().into_iter().zip(route_counted).map(|(after,before)|after.saturating_sub(before)).collect::<Vec<_>>();
            let stage=spark_tp::projection_stage::counts();write(&saved.join("EXCHANGE_STAGE"),format!("decode_steps={steps} accepted={} original_messages={} local_words={} scope=accepted_commands_completed_by_final_sync\n",stage.accepted-stage_counted.accepted,stage.original_messages-stage_counted.original_messages,stage.words-stage_counted.words).as_bytes())?;
            write(&saved.join("PROJECTION_DISPATCH"),format!("decode_steps={steps} nv_rows_stream={} fp8_tile={} project_f32_ordered={} mixed_projection={} causal_fast={} prompt_select_rows={} nv_tile={} nv_quantize={} nv_quantize_batch={} nv_quantize_legacy={} route_device={} route_device_gate_up={} route_device_down={} route_host_fallback={} project_bf16_ordered={} scope=accepted_commands_completed_by_final_sync\n",counts.projection_dispatches[0],counts.projection_dispatches[1],counts.projection_dispatches[2],counts.projection_dispatches[3],counts.projection_dispatches[4],counts.projection_dispatches[5],counts.projection_dispatches[6],counts.projection_dispatches[7],counts.projection_dispatches[8],counts.projection_dispatches[9],route[0]+route[1],route[0],route[1],route[2],counts.projection_dispatches[10]).as_bytes())?;
            write(&saved.join("RECORDED_KINDS"),format!("decode_steps={steps} zero_fill_launches={} zero_fill_rows={} zero_fill_words={} {} scope=decode_steps_recorded_calls_merged_clears_counted_only_in_zero_fill_launches\n",counts.zero_fill_launches,counts.zero_fill_rows,counts.zero_fill_words,spark_gpu_probe::resident_launch::Kind::ALL.iter().map(|kind|format!("{}={}",kind.kernel(),counts.recorded_kinds[*kind as usize])).collect::<Vec<_>>().join(" ")).as_bytes())?;
            if check_greedy {write(&saved.join("COMPACT_HEAD"),format!("checked_steps={steps} state=compact ids=full_logits_equal timing=diagnostic_only\n").as_bytes())?;}
            if let (Some(m),Some(before))=(&mailbox,bell_counted) {let now=m.doorbell_stats();write(&saved.join("DOORBELL"),format!("decode_steps={steps} parks={} parked_ms={:.3} lingered={} linger_ms={:.3} scope=this_rank_decode_steps\n",now[0]-before[0],(now[1]-before[1]) as f64/1e6,now[2]-before[2],(now[3]-before[3]) as f64/1e6).as_bytes())?;}
            write(&saved.join("TIMING"),timing.as_bytes())?;write(&saved.join("PROMPT_LOGITS"),hashes.as_bytes())?;
            write(&saved.join("OUTPUT_IDS"),output.iter().map(|v|format!("{v}\n")).collect::<String>().as_bytes())?;
            let mut decoder=tokenizer.decoder();let mut text=String::new();for &id in &output {text.push_str(&decoder.push(id).map_err(|e|e.to_string())?);}text.push_str(&decoder.finish());write(&saved.join("TEXT"),text.as_bytes())?;
            write(&saved.join("COMPLETE"),b"RESIDENT DECODE RUN PASS\n")?;
            println!("RESIDENT DECODE PASS rank={rank} repeat={repeat} tokens={} {timing}ids={output:?} text={text:?}",output.len());
            if repeat>1 && !head_last {compare(&out.join("run1"),&saved)?;}
        }
        write(&out.join("COMPLETE"),b"RESIDENT DECODE SUITE PASS\n")?;println!("RESIDENT JOB PASS rank={rank} out={}",out.display());
        }Ok(())
            })();
            let drained=backend.synchronize();let result=result.and(drained);
            if result.is_err() {if let Some(m)=&mailbox {m.fail();}if let Some(conductor)=&conductor {conductor.fail();}}
            stop.store(true,Ordering::Release);
            let joined=handle.map(|h|h.join().map_err(|_|"resident device wire panicked".to_owned()).and_then(|v|v)).unwrap_or(Ok(()));
            let host_joined=host_worker.map_or(Ok(()),|worker|worker.finish());
            result.and(joined).and(host_joined)
        })
    })
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
    input.extend([0.0;32]);input.extend([-0.0;32]);input.extend([-f32::from_bits(1);32]);let mut edge=[0.0;32];edge[..12].copy_from_slice(&[448.0,1.0625,1.1875,-1.0625,-0.0,1.0/1024.0,3.0/1024.0,450.0,1.0/2048.0,-1.0/2048.0,f32::from_bits(1),-f32::from_bits(1)]);input.extend(edge);
    let policies=[("legacy",Op::Mxfp8Quantize,Activation::from_f32(&input)?),("positive_zero",Op::Mxfp8QuantizePositiveZero,Activation::from_f32_positive_zero(&input)?)];let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {let source=backend.upload_vector(&input)?;
        for (policy,op,expected) in &policies {for repeat in 1..=3 {for plane in 0..3 {let s=Shape {op:*op,heads:0,dim:0,taps:0,rows:plane,cols:input.len() as u32,epsilon:0.0,bound:0.0};
            let output=backend.causal_operation(s,[Some(&source),None,None,None],None)?;let got=backend.read_vector_values(&output)?;
            for (i,&v) in got.iter().enumerate() {let wanted=match plane {1=>expected.values()[i] as f32,2=>expected.scales()[i/32] as f32,_=>expected.decoded()[i]};
                if v.to_bits()!=wanted.to_bits() {return Err(format!("MXFP8 quantizer differs repeat={repeat} plane={plane} lane={i} device={:08x} host={:08x}",v.to_bits(),wanted.to_bits()));}}
            let hash=sha256::digest(&floats(&got)?);receipt.push_str(&format!("policy={policy} repeat={repeat} plane={plane} lanes={} sha256={hash} exact=true\n",input.len()));
        }}}Ok(())
    })?;write(&out.join("RECEIPT"),receipt.as_bytes())?;println!("MXFP8 QUANTIZER DEVICE PASS policies=2 repeats=3 lanes={} decoded_bits=true codes=true scales=true",input.len());Ok(())
}
#[cfg(target_os="linux")]
fn views(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let values=(0..64).map(|i|if i%7==0 {-0.0} else {f32::from_bits(0x3f000000+i*1237)}).collect::<Vec<_>>();
        for _ in 0..3 {
            let source=b.upload_vector(&values)?.into_resident_shared()?;
            if source.resident_range(0..0).is_ok() || source.resident_range(0..65).is_ok() {return Err("resident view accepted an invalid extent".into());}
            let middle=source.resident_range(16..48)?;let tail=middle.resident_range(3..19)?;
            drop(source);drop(middle);
            let read=b.read_vector_values(&tail)?;
            if read.iter().zip(&values[19..35]).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err("resident nested view changed bits or lost its owner".into());}
            let rounded=b.round_bf16_lanes(&tail)?;drop(tail);
            let owned=rounded.into_vector();let read=b.read_vector_values(&owned)?;
            if read.iter().zip(&values[19..35]).any(|(a,b)|a.to_bits()!=((b.to_bits()+0x7fff+((b.to_bits()>>16)&1))&0xffff0000)) {return Err("resident view consumer differs".into());}
        }
        let patterns=[0u32,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x7fa00001,0x00000001,0xffffffff];
        let host=(0..1024).map(|i|f32::from_bits(patterns[i%patterns.len()])).collect::<Vec<_>>();
        let source=b.upload_vector(&host)?.into_resident_shared()?;let tail=source.resident_range(7..520)?;
        let mut table=None;
        let empty=b.resident_join(&[],&mut table,0)?;drop(empty);
        if table.is_some() {return Err("empty resident join allocated descriptors".into());}
        for lengths in [[1usize,513,0,65],[0,0,0,0],[7,3,8,1],[128,129,130,131]] {
            let parts=[(&source,lengths[0]),(&tail,lengths[1]),(&source,lengths[2]),(&source,lengths[3])];
            let out=b.resident_join(&parts,&mut table,0)?;
            let expected=host[..lengths[0]].iter().chain(&host[7..7+lengths[1]]).chain(&host[..lengths[2]]).chain(&host[..lengths[3]]).map(|v|v.to_bits()).collect::<Vec<_>>();
            if expected.is_empty() {continue;}
            let got=b.read_vector_values(&out)?;
            if got.iter().map(|v|v.to_bits()).collect::<Vec<_>>()!=expected {return Err("resident joined words differ".into());}
        }
        let address=table.as_ref().unwrap().device_address();
        let out=b.resident_join(&[(&tail,1)],&mut table,0)?;drop(out);
        if table.as_ref().unwrap().device_address()!=address {return Err("resident descriptor allocation was not reused".into());}
        if b.resident_join(&[(&source,1025)],&mut table,0).is_ok()
            || b.resident_join(&vec![(&source,1);65536],&mut table,0).is_ok() {return Err("resident join accepted an invalid extent or grid".into());}
        let shared=table.take().unwrap().into_resident_shared()?;let alias=shared.resident_range(0..1)?;table=Some(shared);
        if b.resident_join(&[(&alias,1)],&mut table,0).is_ok() {return Err("resident descriptors accepted an aliased source".into());}
        Ok(())
    })?;
    let text="RESIDENT VIEWS PASS repeats=3 nested_offsets=true retained_owners=true invalid_extents=true signed_zero=true queued_consumer=true joined_bits=true native_descriptor_reuse=true invalid_join=true\n";
    write(&out.join("RECEIPT"),text.as_bytes())?;print!("{text}");Ok(())
}
#[cfg(target_os="linux")]
fn queued(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        b.configure_vector_pool(16<<20);
        let expected=[1.0f32,-0.0,3.0,-4.0].repeat(32);
        for _ in 0..3 {
            let queue=b.resident_queue()?;b.enable_resident_launches()?;
            if b.resident_queue().is_ok() {return Err("resident queue accepted nested ownership".into());}
            let before=b.stats();let mut current=b.upload_vector(&expected)?;
            for _ in 0..20 {let next=b.round_bf16_lanes(&current)?.into_vector();drop(current);current=next;}
            if b.stats().since(before).stream_wait_calls!=0 {return Err("queued temporary owners drained the stream".into());}
            let got=b.read_vector_values(&current)?;if b.stats().since(before).graph_launches==0 {return Err("queued chain did not replay a captured piece".into());}
            if got.iter().zip(&expected).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err("queued owner lifetime changed bits".into());}
            drop(current);drop(queue);
            let before=b.stats();let input=b.upload_vector(&expected)?;let _out=b.round_bf16_lanes(&input)?;
            if b.stats().since(before).stream_wait_calls==0 {return Err("resident queue changed ordinary completion".into());}
        }
        let expected_error=(||->Result<()> {let _queue=b.resident_queue()?;b.enable_resident_launches()?;
            let input=b.upload_vector(&expected)?;let _out=b.round_bf16_lanes(&input)?;
            Err("deliberate queued consumer error".into())})();
        if expected_error.is_ok() {return Err("queued error fixture did not fail".into());}
        {
            use spark_gpu_probe::tp_mailbox_host::{SEND_SEQ,SEND_LEN};
            use std::sync::atomic::Ordering;
            let mailbox=b.tp_mailbox(4096)?;let host=mailbox.take_host().ok_or("queue fixture has no mailbox endpoint")?;
            let layout=mailbox.layout();
            let input=[1.0f32,-0.0,f32::from_bits(1),-7.5];let source=b.upload_vector(&input)?;
            b.tp_mailbox_put_finite(&mailbox,&source,1,input.len(),input.len(),0,1)?;b.synchronize()?;
            let slot=1%layout.slots;let words=host.words();
            if mailbox.failed() || words[SEND_SEQ+slot].load(Ordering::Acquire)!=1 || words[SEND_LEN+slot].load(Ordering::Acquire)!=16 {
                return Err("finite resident mailbox did not publish".into());
            }
            for (i,v) in input.iter().enumerate() {if words[layout.send_slot(slot)+i].load(Ordering::Relaxed)!=v.to_bits() {
                return Err("finite resident mailbox changed payload bits".into());
            }}
            let invalid=b.upload_vector(&[f32::NAN])?;
            b.tp_mailbox_put_finite(&mailbox,&invalid,1,1,1,0,2)?;b.synchronize()?;
            if !mailbox.failed() || words[SEND_SEQ+2%layout.slots].load(Ordering::Acquire)!=0 {return Err("resident mailbox published a nonfinite row".into());}
            if b.tp_mailbox_put_finite(&mailbox,&source,1,4,5,0,3).is_ok() {return Err("resident mailbox accepted an invalid span".into());}
        }
        {
            let _queue=b.resident_queue()?;b.enable_resident_launches()?;let mut table=None;let mut outputs=Vec::new();
            let before=b.stats();
            for parts in [2usize,5,1,7,3] {
                let data=(0..parts).map(|n|(0..n+3).map(|i|if i%3==0 {-0.0} else {(i+n) as f32}).collect::<Vec<_>>()).collect::<Vec<_>>();
                let inputs=data.iter().map(|v|b.upload_vector(v)).collect::<Result<Vec<_>>>()?;
                let sources=inputs.iter().map(|v|(v,v.len())).collect::<Vec<_>>();
                let output=b.resident_join(&sources,&mut table,0)?;outputs.push((output,data.concat()));
                drop(inputs);
            }
            let changed=b.stats().since(before);if changed.stream_wait_calls!=changed.graph_captures {return Err("queued descriptor reuse waited outside cold capture".into());}
            drop(table);
            for (output,expected) in outputs {if b.read_vector_values(&output)?.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err("queued descriptor reuse or growth changed output bits".into());}}
        }
        for already_failed in [false,true] {
            let mailbox=b.tp_mailbox(4096)?;if already_failed {mailbox.fail();}
            let _queue=b.resident_queue()?;b.enable_resident_launches()?;let local=[1.0f32,-0.0,-7.5,3.0];
            let input=b.upload_vector(&local)?;let mut gathered=b.vector(8)?;
            b.zero_vector(&mut gathered)?;b.copy_range(&input,&mut gathered,0,0,4)?;
            // No peer ever publishes sequence 1. Queue an actual consumer before checking failure.
            b.tp_mailbox_take(&mailbox,&mut gathered,1,8,4,4,1,1_000)?;
            let consumer=b.round_bf16_lanes(&gathered)?.into_vector();drop(gathered);drop(input);
            let got=b.read_vector_values(&consumer)?;let mut expected=local.to_vec();expected.extend([0.0;4]);
            if got.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err("failed take exposed uninitialized peer rows".into());}
            if mailbox.require_success().is_ok() {return Err("failed take was accepted as a successful request".into());}
        }
        let _reopened=b.resident_queue()?;Ok(())
    })?;
    let text="RESIDENT QUEUE PASS repeats=3 early_owner_drop=true nested_refusal=true ordinary_completion=true error_cleanup=true queued_join_reuse=true queued_join_growth=true missing_peer_consumer=true failed_mailbox_consumer=true failed_request=true indirect_graphs=true dynamic_arguments=true exact_bits=true\n";
    write(&out.join("RECEIPT"),text.as_bytes())?;print!("{text}");Ok(())
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
        [mode,a,b] if mode=="compare"=>compare(&PathBuf::from(a),&PathBuf::from(b)),
        [mode,root,repeats] if mode=="plan-time"=>repeats.parse::<usize>().map_err(|e|e.to_string()).and_then(|n|plan_time(Path::new(root),n)),
        #[cfg(target_os="linux")]
        [mode,root,request,out] if mode=="prepare"=>prepare(Path::new(root),Path::new(request),Path::new(out)),
        #[cfg(target_os="linux")]
        [_,_,_,_,mode,..] if mode=="pair"=>run_pair(&args),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="quantizer"=>quantizer(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,root,rank,parts,out] if mode=="load-hash"=>load_weights::check(Path::new(root),rank,parts,Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="layout"=>layout(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="stream"=>stream(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="views"=>views(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="program"=>program::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,root,out] if mode=="attention-time"=>attention::measure(Path::new(root),Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="queued"=>queued(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="recorder"=>recorder::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="argmax"=>argmax::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="workspace"=>workspace::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="cache-append"=>cache_append::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="index-append"=>index_append::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="memo-slab"=>memo_slab::check(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="recurrence"=>recurrence::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="convolution"=>convolution::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="nvfp4"=>nvfp4::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="prompt-rows"=>prompt_rows::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="softmax-rows"=>softmax_rows::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="softmax-prompt"=>softmax_rows::prompt(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,kind] if mode=="prompt-rows-bad"=>prompt_rows::bad_selection(kind),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="projection-batch"=>projection_batch::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="device-route"=>device_route::run(Path::new(out),true),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="exchange-stage"=>exchange_stage::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="bus-shadow"=>bus_shadow::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="fill-free"=>fill_free::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="put-wide"=>put_wide::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="select-known"=>select_known::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="device-route-smoke"=>device_route::run(Path::new(out),false),
        #[cfg(target_os="linux")]
        [mode,kind] if mode=="device-route-refuse"=>device_route::refuse(kind),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="nv-group-rows"=>nv_group_rows::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="nv-b-sweep"=>nv_b::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="fp8-serial"=>fp8_warp::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="fp8-all-codes"=>fp8_all_codes::run(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="route-edges"=>routing::edges(Path::new(out)),
        #[cfg(target_os="linux")]
        [mode,out] if mode=="route-bench"=>routing::bench(Path::new(out)),
        _=>Err("usage: resident-decode-check RANK 2 LISTEN NEXT pair MODEL INPUT_IDS REQUEST_JSON NEW_OUT REPEATS | prepare MODEL REQUEST_JSON NEW_IDS | quantizer NEW_OUT | layout NEW_OUT | compare LEFT RIGHT | plan-time MODEL REPEATS".into()),
    };if let Err(error)=result {eprintln!("RESIDENT CHECK FAIL {error}");std::process::exit(1);}
}
#[cfg(test)] #[test]
fn forced_memory_cases_bound_group_and_live_session_count() {
    assert_eq!(forced_case("109828583424").unwrap(),(109828583424,256,1));
    for group in [256,512,1024] {for chats in [1,8] {assert_eq!(forced_case(&format!("109828583424:{group}:{chats}")).unwrap(),(109828583424,group,chats));}}
    for bad in ["0","1:255:1","1:1025:8","1:512:9","1:512:2","1:512","1:256:1:65536"] {assert!(forced_case(bad).is_err());}
}
