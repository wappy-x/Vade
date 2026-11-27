#![forbid(unsafe_code)]
use std::{collections::BTreeMap,fs::{self,File,OpenOptions},io::{Read,Write},path::{Path,PathBuf},process::{Command,Stdio},sync::LazyLock,thread,time::{Duration,Instant}};
#[path="flywheel4_receipts.rs"] mod receipts;
type Result<T> = std::result::Result<T,String>;
// Private paths, box pairs and the build box come from the file named by VADE_KIT_HOSTS, one "KEY value..." line each
// (see tools/kit/hosts.example). A missing file or key stops the command before it does any work.
fn hosts()->Result<Vec<Vec<String>>> {
    // Tests use the placeholder file, so they need no private settings.
    let content=match std::env::var("VADE_KIT_HOSTS") {Ok(path)=>text(path)?,Err(_) if cfg!(test)=>include_str!("../hosts.example").into(),
        Err(_)=>return Err("set VADE_KIT_HOSTS to the private hosts file (see tools/kit/hosts.example)".into())};
    Ok(content.lines().map(|line|line.split('#').next().unwrap_or("").split_whitespace().map(str::to_owned).collect::<Vec<_>>()).filter(|fields|!fields.is_empty()).collect())
}
fn host(key:&str)->String {
    match hosts().and_then(|lines|lines.into_iter().find(|fields|fields.len()==2 && fields[0]==key).map(|fields|fields[1].clone()).ok_or(format!("the hosts file has no {key} line"))) {
        Ok(value)=>value,Err(error)=>{eprintln!("FLYWHEEL ERROR {error}");std::process::exit(1)}}
}
/// The ring network prefix of an authorized "PAIR <rank 0 box> <rank 1 box> <prefix>" line.
fn pair_network(first:&str,second:&str)->Result<String> {
    hosts()?.into_iter().find(|fields|fields.len()==4 && fields[0]=="PAIR" && fields[1]==first && fields[2]==second).map(|fields|fields[3].clone()).ok_or("screen pair is not authorized".into())
}
static VAULT:LazyLock<String>=LazyLock::new(||host("VAULT"));
const BASE:&str="a578ac0b408e6184e35f5269d6475a21d48a64eb";
const REFERENCE_BASE:&str=BASE;
static WORK:LazyLock<String>=LazyLock::new(||host("WORK"));
static KIT:LazyLock<String>=LazyLock::new(||host("KIT"));
static DRIVER:LazyLock<String>=LazyLock::new(||host("DRIVER"));
static SSH:LazyLock<String>=LazyLock::new(||host("SSH"));
static LOG:LazyLock<String>=LazyLock::new(||host("LOG"));
static RUN2:LazyLock<String>=LazyLock::new(||host("RUN2"));
static BUILD_BOX:LazyLock<String>=LazyLock::new(||host("BUILD_BOX"));
const ORDER_SEED:u64=0x2026_1107_0401_f17e;
#[derive(Clone,Copy)] struct Cut {id:&'static str,cuts:&'static str,options:&'static str}
const CUTS:[Cut;3]=[
    Cut {id:"head",cuts:"head",options:""},
    Cut {id:"defer",cuts:"head",options:"--defer-gather"},
    Cut {id:"conv",cuts:"head",options:""},
];
#[derive(Clone)] struct Variant {name:String,cuts:&'static str,options:&'static str,pass:usize,cut:Cut}
fn variants()->Vec<Variant> {
    let head=|index|Variant {name:format!("head-{index:02}"),cuts:"head",options:"",pass:0,cut:CUTS[0]};
    let mut result=vec![head(0)];let mut seed=ORDER_SEED;let mut previous=Vec::new();
    for pass in 1..=2 {let mut order=(1..CUTS.len()).collect::<Vec<_>>();
        for position in (1..order.len()).rev() {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;order.swap(position,seed as usize%(position+1));}
        if order==previous {order.reverse();}previous=order.clone();
        for index in order {let cut=CUTS[index];result.push(Variant {name:format!("p{pass}-{}",cut.id),cuts:cut.cuts,options:cut.options,pass,cut});result.push(head(result.len()/2));}
    }result
}
#[derive(Clone,Copy)] struct Job {name:&'static str,ids:&'static str,request:&'static str,input:usize,output:usize}
const JOBS:[Job;3]=[
    Job {name:"glm-tokens64",ids:"prompt.ids",request:"request.json",input:15,output:64},
    Job {name:"out127",ids:"p130.ids",request:"req127-64.json",input:127,output:64},
    Job {name:"out1040",ids:"p1000.ids",request:"req-1000.json",input:1040,output:32},
];
fn text(path:impl AsRef<Path>)->Result<String> {fs::read_to_string(path.as_ref()).map_err(|error|format!("{}: {error}",path.as_ref().display()))}
fn save(path:impl AsRef<Path>,value:impl AsRef<[u8]>)->Result<()> {fs::write(path.as_ref(),value).map_err(|error|format!("{}: {error}",path.as_ref().display()))}
fn output(command:&mut Command)->Result<String> {
    let mut child=command.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|error|error.to_string())?;
    let mut stdout=child.stdout.take().ok_or("missing command output")?;let mut stderr=child.stderr.take().ok_or("missing command diagnostics")?;
    thread::scope(|scope| {
        let out=scope.spawn(move|| {let mut bytes=Vec::new();stdout.read_to_end(&mut bytes).map(|_|bytes)});
        let err=scope.spawn(move|| {let mut bytes=Vec::new();stderr.read_to_end(&mut bytes).map(|_|bytes)});
        let status=wait_child(&mut child,Duration::from_secs(60));
        let stdout=out.join().map_err(|_|"output reader panicked")?.map_err(|error|error.to_string())?;
        let stderr=err.join().map_err(|_|"diagnostic reader panicked")?.map_err(|error|error.to_string())?;
        let status=status?;if !status.success() {return Err(format!("command exited {status}: {}",String::from_utf8_lossy(&stderr)));}
        String::from_utf8(stdout).map_err(|error|error.to_string())
    })
}
fn wait_child(child:&mut std::process::Child,budget:Duration)->Result<std::process::ExitStatus> {
    let started=Instant::now();loop {if let Some(status)=child.try_wait().map_err(|error|error.to_string())? {return Ok(status);}
        if started.elapsed()>=budget {let _=child.kill();let _=child.wait();return Err("owned command exceeded its time limit".into());}
        thread::sleep(Duration::from_millis(20));}
}
fn now()->Result<String> {Ok(output(Command::new("date").arg("+%H:%M:%S"))?.trim().into())}
fn log(message:&str)->Result<()> {
    if message.contains('\n') {return Err("log message must be one line".into());}
    let line=format!("- {} CST, lane FLYWHEEL {message}\n",now()?);
    OpenOptions::new().append(true).open(&*LOG).and_then(|mut file|file.write_all(line.as_bytes())).map_err(|error|error.to_string())
}
fn remote(box_name:&str,command:&str)->Result<String> {output(Command::new(&*SSH).args([box_name,command]))}
fn remote_input(box_name:&str,command:&str,bytes:&[u8])->Result<()> {
    let mut child=Command::new(&*SSH).args([box_name,command]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().map_err(|error|error.to_string())?;
    child.stdin.take().ok_or("missing upload pipe")?.write_all(bytes).map_err(|error|error.to_string())?;
    let result=child.wait_with_output().map_err(|error|error.to_string())?;
    if !result.status.success() {return Err(format!("private upload failed: {}",String::from_utf8_lossy(&result.stderr)));}Ok(())
}
fn valid_name(value:&str)->bool {!value.is_empty() && value.len()<=64 && value.bytes().all(|byte|byte.is_ascii_alphanumeric() || byte==b'-')}
fn variant_setup()->String {format!("mkdir -- {}\n",variants().iter().map(|variant|variant.name.as_str()).collect::<Vec<_>>().join(" "))}
fn copy_tree(source:&Path,destination:&Path)->Result<()> {
    fs::create_dir_all(destination).map_err(|error|error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error|error.to_string())? {let entry=entry.map_err(|error|error.to_string())?;let kind=entry.file_type().map_err(|error|error.to_string())?;
        let target=destination.join(entry.file_name());if kind.is_dir() {copy_tree(&entry.path(),&target)?;} else if kind.is_file() {fs::copy(entry.path(),target).map_err(|error|error.to_string())?;} else {return Err("unexpected non-file in receipt tree".into());}}
    Ok(())
}
#[derive(Clone)] struct Config {root:PathBuf,tag:String,revision:String,boxes:[String;2],network:String,long:bool}
impl Config {
    fn read(root:&Path)->Result<Self> {
        let content=text(root.join("CONFIG"))?;let fields=content.split_whitespace().collect::<Vec<_>>();
        if fields.len()!=6 || fields[0]!=BASE || !valid_name(fields[1]) || fields[2].len()!=40 || !fields[2].bytes().all(|byte|byte.is_ascii_hexdigit()) || !["short","long"].contains(&fields[5]) {return Err("invalid screen CONFIG".into());}
        let network=pair_network(fields[3],fields[4])?;
        Ok(Self {root:root.to_owned(),tag:fields[1].into(),revision:fields[2].into(),boxes:[fields[3].into(),fields[4].into()],network,long:fields[5]=="long"})
    }
    fn jobs(&self)->&[Job] {&JOBS[..if self.long {3} else {2}]}
    fn remote_dir(&self)->String {format!("$HOME/vade-flywheel/{}",self.tag)}
    fn pair(&self)->String {self.boxes.join(",")}
}
fn prepare(revision:&str,tag:&str,pair:&str,long:bool)->Result<PathBuf> {
    if !valid_name(tag) {return Err("invalid run tag".into());}
    if long {return Err("run 4 excludes the long job to stay below forty minutes".into());}
    let boxes=match pair.split_once(',') {Some((first,second)) if pair_network(first,second).is_ok()=>[first,second],_=>return Err("screen pair is not authorized".into())};
    let revision=output(Command::new("git").args(["-C",WORK.as_str(),"rev-parse","--verify",&format!("{revision}^{{commit}}")]))?.trim().to_owned();
    let payload=Path::new(&*KIT).join(format!("payload-{}-native",&revision[..9]));
    if text(payload.join("BUILD-REVISION"))?.trim()!=revision {return Err("payload revision differs".into());}
    let root=Path::new(&*VAULT).join("benchmarks/2026-11-07/flywheel").join(tag);
    fs::create_dir_all(root.parent().unwrap()).map_err(|error|error.to_string())?;fs::create_dir(&root).map_err(|error|error.to_string())?;
    save(root.join("CONFIG"),format!("{BASE} {tag} {revision} {} {} {}\n",boxes[0],boxes[1],if long {"long"} else {"short"}))?;
    let config=Config::read(&root)?;let upload=root.join("upload");fs::create_dir(&upload).map_err(|error|error.to_string())?;
    for name in ["resident-decode-check","BUILD-REVISION"] {fs::copy(payload.join(name),upload.join(name)).map_err(|error|error.to_string())?;}
    for name in ["SHA256SUMS","BUILD.log"] {fs::copy(payload.join(name),root.join(name)).map_err(|error|error.to_string())?;}
    for job in config.jobs() {for name in [job.ids,job.request] {fs::copy(Path::new(&*KIT).join("box").join(name),upload.join(name)).map_err(|error|error.to_string())?;}}
    let mut jobs=String::new();let mut variants=String::new();
    for (index,variant) in crate::variants().iter().enumerate() {
        variants.push_str(&format!("{}\t{}\t{}\t{}\t{}\n",variant.name,variant.cuts,variant.options,variant.pass,variant.cut.id));
        if index>0 {jobs.push_str(&format!("prompt.ids request.json {}/warmup 1 --flywheel={}:{} {}\n",variant.name,variant.name,variant.cuts,variant.options));}
        for job in config.jobs() {jobs.push_str(&format!("{} {} {}/{} 3 --flywheel={}:{} {}\n",job.ids,job.request,variant.name,job.name,variant.name,variant.cuts,variant.options));}
    }
    save(upload.join("jobs.txt"),jobs)?;save(root.join("VARIANTS.tsv"),variants)?;save(root.join("ORDER-SEED"),format!("{ORDER_SEED:016x}\n"))?;
    save(upload.join("engine.sh"),format!("#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$$\" > engine.pid\n{}exec ./resident-decode-check \"$1\" 2 \"$2\" \"$3\" pair \"$HOME/models/nvidia/GLM-5.3-Flash-FP4-Local\" prompt.ids request.json head-00/warmup 1 --cost --device-wire --flywheel=head-00:head --jobs=jobs.txt\n",variant_setup()))?;
    save(upload.join("run.sh"),"#!/bin/bash\nset -euo pipefail\ntrap 'status=$?; date -Is; printf \"%s\\n\" \"$status\" > run.exit' EXIT\nrank=$1; network=$2; budget=$3\nexport TP_TOPOLOGY=path\nIFS= read -r SEQPAR_CONTROL_SECRET < transport.token\nexport SEQPAR_CONTROL_SECRET\nexport TP_RUN_ID=\"flywheel-${PWD##*/}\"\ndate -Is\nsha256sum -c UPLOAD.sha256\ncat BUILD-REVISION\nif [ \"$rank\" = 0 ]; then own=$network.1; peer=$network.2; else own=$network.2; peer=$network.1; fi\ntimeout --signal=TERM --kill-after=15s \"${budget}s\" bash engine.sh \"$rank\" \"$own:19690\" \"$peer:19690\" &\nprintf '%s\\n' \"$!\" > worker.pid\nwait \"$!\"\n")?;
    let mut hash=Command::new("shasum");hash.args(["-a","256"]).current_dir(&upload);
    for entry in fs::read_dir(&upload).map_err(|error|error.to_string())? {hash.arg(entry.map_err(|error|error.to_string())?.file_name());}
    save(upload.join("UPLOAD.sha256"),output(&mut hash)?)?;
    let registry=text(Path::new(&*KIT).join(format!("headref-{}-{}.txt",boxes[0],boxes[1])))?;
    let reference_line=registry.lines().rev().find(|line| {let fields=line.split_whitespace().collect::<Vec<_>>();fields.len()>=6 && fields[2]==REFERENCE_BASE && fields[4]==boxes[0] && fields[5]==boxes[1]}).ok_or("no standard native head receipt for this pair")?;
    save(root.join("REFERENCE-REGISTRY.txt"),format!("{reference_line}\n"))?;fresh_reference(&config)?;
    let reference=reference_line.split_whitespace().nth(3).ok_or("reference folder missing")?;
    let reference=Path::new(&*KIT).join("evidence").join(reference).join(reference_line.split_whitespace().nth(1).ok_or("reference payload missing")?);save(root.join("REFERENCE-SOURCE.txt"),format!("{}\n",reference.display()))?;
    save(root.join("REFERENCE-CAVEAT.txt"),format!("Source baseline {BASE}; standard head {REFERENCE_BASE}; registry {reference_line}. Use the pair's newest a578 registry line under the run-three ruling. Report its age and same-hour status; thermal equality is not assumed. Only interleaved same-process heads decide screen eligibility. No bundle merges from this screen.\n"))?;
    for box_name in boxes {let source=reference.join(box_name);let target=root.join("reference").join(box_name);fs::create_dir_all(&target).map_err(|error|error.to_string())?;
        for name in ["BUILD-REVISION","glm.exit","glm-run.log"] {fs::copy(source.join(name),target.join(name)).map_err(|error|error.to_string())?;}
        for job in config.jobs() {copy_tree(&source.join(job.name),&target.join(job.name))?;}}
    for job in config.jobs() {copy_tree(&Path::new(&*KIT).join("evidence/golden").join(job.name),&root.join("golden").join(job.name))?;}
    let mut ptx=Command::new("bash");ptx.arg(Path::new(&*KIT).join("tools/ptx-check.sh")).arg(&payload).arg(Path::new(&*KIT).join("payload-a578ac0b4-native"));
    let ptx=output(&mut ptx)?;save(root.join("PTX-CHECK.txt"),&ptx)?;
    let expected=format!("PTX CHECK {} clean",&revision[..9]);if ptx.lines().last()!=Some(expected.as_str()) {return Err("PTX check did not give the required clean line".into());}
    for name in ["flywheel4.rs","flywheel4_receipts.rs","flywheel4_counters.rs","run-flywheel4.sh"] {fs::copy(Path::new(&*DRIVER).join(name),root.join(name)).map_err(|error|error.to_string())?;}
    save(root.join("SOURCE.diff"),output(Command::new("git").args(["-C",WORK.as_str(),"diff",BASE,&revision]))?)?;
    receipts::reference(&config)?;receipts::declare(&config)?;log(&format!("PTX CLEAN {expected}; receipt {}/PTX-CHECK.txt",root.display()))?;Ok(root)
}
fn upload(config:&Config,box_name:&str)->Result<()> {
    let archive=config.root.join("upload.tar");if !archive.exists() {output(Command::new("tar").args(["-cf"]).arg(&archive).arg("-C").arg(config.root.join("upload")).arg("."))?;}
    let source=File::open(&archive).map_err(|error|error.to_string())?;
    let command=format!("set -eu; test ! -e {dir}; mkdir -p {dir}; cd {dir}; tar -xf -",dir=config.remote_dir());
    let result=Command::new(&*SSH).args([box_name,&command]).stdin(Stdio::from(source)).output().map_err(|error|error.to_string())?;
    if !result.status.success() {return Err(format!("payload upload failed: {}",String::from_utf8_lossy(&result.stderr)));}Ok(())
}
fn collect(config:&Config,box_name:&str,budget:u64)->Result<()> {
    if budget==0 {return Err("no receipt collection time remains".into());}
    let target=config.root.join(box_name);fs::create_dir_all(&target).map_err(|error|error.to_string())?;
    let stderr=File::create(config.root.join(format!("collect-{box_name}.stderr"))).map_err(|error|error.to_string())?;
    let command=format!("cd {}; timeout {budget}s tar --exclude=./transport.token --exclude=./resident-decode-check -cf - .",config.remote_dir());
    let mut sender=Command::new(&*SSH).args([box_name,&command]).stdout(Stdio::piped()).stderr(stderr).spawn().map_err(|error|error.to_string())?;
    let mut unpack=Command::new("tar").args(["-xf","-","-C"]).arg(&target).stdin(Stdio::from(sender.stdout.take().ok_or("missing receipt pipe")?)).spawn().map_err(|error|error.to_string())?;
    let sent=wait_child(&mut sender,Duration::from_secs(budget+5));
    if sent.is_err() {let _=unpack.kill();let _=unpack.wait();return Err(format!("receipt collection timed out for {box_name}"));}
    let unpack=wait_child(&mut unpack,Duration::from_secs(10))?;
    if !unpack.success() || !sent?.success() {return Err(format!("receipt collection failed for {box_name}"));}Ok(())
}
fn owned(config:&Config,box_name:&str)->Result<bool> {let line=remote(box_name,"tail -1 ~/gpu.lock")?;Ok(line.trim().starts_with("SPARKS CLAIMED ") && line.contains(&format!(" lane-flywheel {box_name} {}",config.tag)))}
fn stop_owned(config:&Config,box_name:&str)->Result<()> {
    if !owned(config,box_name)? {return Err(format!("claim ownership changed on {box_name}"));}
    let command=format!("if [ -d {dir} ]; then cd {dir}; for file in engine.pid worker.pid launcher.pid; do if [ -f \"$file\" ]; then read -r process < \"$file\"; case \"$process\" in ''|*[!0-9]*) exit 3;; esac; if [ \"$(readlink /proc/$process/cwd 2>/dev/null)\" = \"$PWD\" ]; then kill -TERM \"$process\" 2>/dev/null || true; fi; fi; done; fi",dir=config.remote_dir());remote(box_name,&command)?;Ok(())
}
fn release(config:&Config,box_name:&str)->Result<()> {
    if !owned(config,box_name)? {return Ok(());}
    let apps=remote(box_name,"nvidia-smi --query-compute-apps=pid --format=csv,noheader")?;
    if !apps.trim().is_empty() {return Err(format!("RELEASE BLOCKED {box_name}: compute apps remain; claim retained"));}
    let clock=now()?;let line=format!("SPARKS RELEASED {clock} lane-flywheel {box_name} {}",config.tag);
    remote(box_name,&format!("rm -f {}/transport.token; printf '%s\\n' '{line}' >> ~/gpu.lock",config.remote_dir()))?;
    log(&format!("{line}; zero compute apps; receipt {}",config.root.display()))?;Ok(())
}
fn cleanup(config:&Config)->Result<()> {
    let mut failures=Vec::new();
    for box_name in &config.boxes {if owned(config,box_name)? {
        if let Err(error)=stop_owned(config,box_name) {failures.push(error);continue;}
        let started=Instant::now();loop {match release(config,box_name) {Ok(())=>break,Err(error)=>{if started.elapsed()>Duration::from_secs(60) {failures.push(error);break;}thread::sleep(Duration::from_secs(2));}}}
    }}
    if failures.is_empty() {Ok(())} else {let message=failures.join("; ");log(&message)?;Err(message)}
}
struct Claim {config:Config,active:bool}
impl Drop for Claim {fn drop(&mut self) {if self.active {let _=cleanup(&self.config);}}}
fn claim(config:&Config)->Result<Claim> {
    let mut guard=Claim {config:config.clone(),active:true};
    if config.boxes.iter().any(|box_name|*box_name==*BUILD_BOX) {
        let status=Command::new("bash").arg(Path::new(&*KIT).join("build-window.sh")).args(["busy",&config.pair()]).status().map_err(|error|error.to_string())?;
        if status.success() {guard.active=false;return Err("build window is busy; no claim made".into());}
        if status.code()!=Some(1) {guard.active=false;return Err("build window status is unreadable".into());}
    }
    let mut lines=Vec::new();for box_name in &config.boxes {let line=remote(box_name,"tail -1 ~/gpu.lock")?;let line=line.trim();
        if !line.starts_with("SPARKS RELEASED ") {guard.active=false;return Err(format!("{box_name} is held: {line}"));}
        if !line.bytes().all(|byte|byte.is_ascii_alphanumeric() || b" :-_,.".contains(&byte)) {return Err("unexpected lock text".into());}
        log(&format!("LOCK OBSERVED {box_name} {line}; source {box_name}:~/gpu.lock; receipt {}",config.root.display()))?;lines.push(line.to_owned());}
    for (box_name,prior) in config.boxes.iter().zip(&lines) {
        let clock=now()?;let line=format!("SPARKS CLAIMED {clock} lane-flywheel {box_name} {}",config.tag);
        let command=format!("set -eu; test \"$(tail -1 ~/gpu.lock)\" = '{prior}'; apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); test -z \"$apps\"; printf '%s\\n' '{line}' >> ~/gpu.lock");
        remote(box_name,&command)?;log(&format!("{line}; cap 40 minutes; receipt {}",config.root.display()))?;
    }
    for box_name in &config.boxes {if !owned(config,box_name)? {return Err("claim changed before staging".into());}}
    Ok(guard)
}
fn run(config:&Config)->Result<()> {
    if config.root.join("RUN-STARTED").exists() {return Err("screen was already started; use a fresh tag".into());}
    receipts::reference(config)?;
    fresh_reference(config)?;
    let mut claim=claim(config)?;let started=Instant::now();save(config.root.join("RUN-STARTED"),format!("{}\n",now()?))?;
    let result=(|| {
        let mut entropy=[0u8;32];File::open("/dev/urandom").and_then(|mut file|file.read_exact(&mut entropy)).map_err(|error|error.to_string())?;
        let mut token=entropy.iter().map(|byte|format!("{byte:02x}")).collect::<String>();token.push('\n');
        let mut temperatures=String::new();
        for box_name in &config.boxes {
            upload(config,box_name)?;remote_input(box_name,&format!("umask 077; cat > {}/transport.token",config.remote_dir()),token.as_bytes())?;
            save(config.root.join(format!("HARDWARE-{box_name}.txt")),remote(box_name,"nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader")?)?;
            temperatures.push_str(&format!("{box_name} before {}\n",remote(box_name,"nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader")?.trim()));
        }
        drop(token);save(config.root.join("TEMPS.txt"),&temperatures)?;
        let budget=2100u64.saturating_sub(started.elapsed().as_secs());if budget<600 {return Err("staging exhausted the load budget".into());}
        for rank in [1usize,0] {let box_name=&config.boxes[rank];if !owned(config,box_name)? {return Err("claim changed before load".into());}
            remote(box_name,&format!("cd {}; nohup bash run.sh {rank} {} {budget} > run.log 2>&1 < /dev/null & printf '%s\\n' \"$!\" > {}/launcher.pid",config.remote_dir(),config.network,config.remote_dir()))?;}
        let mut finished=[false;2];
        loop {
            for (rank,box_name) in config.boxes.iter().enumerate() {if !finished[rank] {
                let state=remote(box_name,&format!("cd {}; if [ -f run.exit ]; then printf 'EXIT '; cat run.exit; elif [ -f launcher.pid ] && kill -0 \"$(cat launcher.pid)\" 2>/dev/null; then echo RUNNING; else echo LOST; fi",config.remote_dir()))?;
                match state.trim() {"EXIT 0"=>finished[rank]=true,"RUNNING"=>{},other=>return Err(format!("{box_name} runner state {other}")),}
            }}
            if finished.iter().all(|done|*done) {break;}
            if started.elapsed()>Duration::from_secs(2150) {return Err("screen load deadline reached".into());}
            thread::sleep(Duration::from_secs(10));
        }
        for box_name in &config.boxes {
            let apps=remote(box_name,"nvidia-smi --query-compute-apps=pid --format=csv,noheader")?;if !apps.trim().is_empty() {return Err(format!("{box_name} still has compute apps after process exit"));}
            temperatures.push_str(&format!("{box_name} after {}\n",remote(box_name,"nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader")?.trim()));
            collect(config,box_name,120.min(2250u64.saturating_sub(started.elapsed().as_secs())))?;
        }
        save(config.root.join("TEMPS.txt"),temperatures)?;save(config.root.join("RUN-FINISHED"),format!("{} elapsed_seconds={}\n",now()?,started.elapsed().as_secs()))?;Ok(())
    })();
    if let Err(error)=&result {log(&format!("RUN BLOCKER {error}; receipt {}",config.root.display()))?;
        for box_name in &config.boxes {if owned(config,box_name).unwrap_or(false) {let _=stop_owned(config,box_name);let budget=60.min(2250u64.saturating_sub(started.elapsed().as_secs()));let _=collect(config,box_name,budget);}}}
    let released=cleanup(config);claim.active=released.is_err();result?;released?;
    let report=receipts::check(config)?;log(&format!("RUN DONE SCREEN ONLY, independent review required; receipt {}/TABLE.tsv",config.root.display()))?;
    for row in report.lines() {log(&format!("RUN DONE {row}; receipt {}/TABLE.tsv",config.root.display()))?;}
    println!("{report}");
    retain(config)?;Ok(())
}
fn fresh_reference(config:&Config)->Result<()> {
    let line=text(config.root.join("REFERENCE-REGISTRY.txt"))?;let fields=line.split_whitespace().collect::<Vec<_>>();
    if fields.len()<6 || fields[2]!=REFERENCE_BASE || fields[4]!=config.boxes[0] || fields[5]!=config.boxes[1] {return Err("reference registry identity differs".into());}
    let recorded=fields[0].parse::<u64>().map_err(|_|"invalid reference epoch")?;
    let current=output(Command::new("date").arg("+%s"))?.trim().parse::<u64>().map_err(|_|"invalid clock epoch")?;
    let age=current.checked_sub(recorded).ok_or("standard head reference is from the future")?;
    let ordered=false;
    if age>3600 && !ordered {return Err("standard head reference is not from the same hour; no claim made".into());}
    save(config.root.join("REFERENCE-AGE.txt"),format!("reference={} revision={REFERENCE_BASE} recorded_epoch={recorded} checked_epoch={current} age_seconds={age} same_hour={} explicit_older_reference_order={ordered}\n",fields[3],age<=3600))
}
fn retain(config:&Config)->Result<()> {
    let tool=Path::new(&*DRIVER).join("flywheel-retain");
    for action in ["compact","verify"] {let status=Command::new(&tool).arg(action).arg(&config.root).status().map_err(|error|error.to_string())?;
        if !status.success() {return Err(format!("receipt retention {action} failed"));}}
    let retention=text(config.root.join("RETENTION.txt"))?;
    let size=retention.lines().find_map(|line|line.strip_prefix("receipt_bytes_after=")).ok_or("receipt byte count missing")?;
    log(&format!("RUN DONE retained receipt_bytes={size}; SHA-256 for every logits file, canonical head and failed cuts retained; receipt {}/RETENTION.txt",config.root.display()))
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let result=match args.first().map(String::as_str) {
        Some("prepare") if args.len()==5=>prepare(&args[1],&args[2],&args[3],match args[4].as_str() {"long"=>true,"short"=>false,_=>{eprintln!("expected short or long");std::process::exit(2)}}).map(|root|println!("{}",root.display())),
        Some("run") if args.len()==2=>Config::read(Path::new(&args[1])).and_then(|config|run(&config)),
        Some("cleanup") if args.len()==2=>Config::read(Path::new(&args[1])).and_then(|config|cleanup(&config)),
        Some("check") if args.len()==2=>Config::read(Path::new(&args[1])).and_then(|config|receipts::check(&config)).map(|report|println!("{report}")),
        _=>Err("usage: flywheel prepare REV TAG PAIR short|long | run|check|cleanup RECEIPTS".into()),
    };
    if let Err(error)=result {eprintln!("FLYWHEEL ERROR {error}");std::process::exit(1);}
}
#[cfg(test)] #[test] fn variant_parents_exist_before_runner_creates_job_directories() {
    let root=std::env::temp_dir().join(format!("flywheel-parents-{}",std::process::id()));
    fs::create_dir(&root).unwrap();
    assert!(Command::new("bash").args(["-eu","-c",&variant_setup()]).current_dir(&root).status().unwrap().success());
    for variant in variants() {fs::create_dir(root.join(variant.name).join("warmup")).unwrap();}
    fs::remove_dir_all(root).unwrap();
}
