//! The token check of the block family with a loop, level 2 of the check program of the loop family: the decode
//! driver (`spark_tp::loop_decoder::Driver`) on one backend against the words of a host run.
//!
//! Mode `loop-token MODEL TRACE OUT` (needs a GPU): the checkpoint of `MODEL` is loaded on one device and the
//! driver runs each position of a saved trace of the whole-model host form. After each visit the stream is read
//! from the device and compared with the trace, then the logits and the top row. Each word must be equal.
//!
//! The trace folder (the host test `first_token_trace_is_written` of the model crate writes it):
//! - `prompt.ids`: the token IDs of the prompt, one decimal ID a line.
//! - `tokens.tsv`: one line a position: the position, a tab, the top row of the position.
//! - `pos<NNNN>.streams.f32`: the stream after each visit of the position in run order, little-endian 32-bit words.
//! - `pos<NNNN>.logits.f32`: the capped logits of the rows that the head can write.
//!
//! A position after the prompt is fed the top row that the host gave the position before it, so the device run
//! stays on the text of the host run.
//!
//! Mode `loop-token-driver MODEL TRACE OUT` (needs a GPU): the same run, but the host words are made in step by
//! the driver on the host backend with the weights of the model folder as host words (`loop_decoder::host_weights`).
//! The trace gives only the token IDs. The receipt also says where the host run of the driver stands apart from
//! the trace.
//!
//! Mode `loop-token-rate MODEL TRACE OUT STEPS` (needs a GPU): the clock time of `STEPS` token steps after the
//! prompt of the trace. Each step is fed the top row of the step before it. A step gives only its top row: no
//! stream and no logit word is read. The run does not stop at an end token. Receipt: `RATE-RESULT.txt`.
//!
//! Mode `loop-token-fixture OUT` (needs a GPU) and `loop-token-fixture-host OUT` (any host): the two made-up
//! checkpoints of the driver (`loop_decoder::fixture`), no model file. The driver runs one chat on the backend and
//! is compared in the same way with the driver on the host backend. Each run also shows that the check can see:
//! one changed stream word and one changed logit word of the host run must be found at their place.
//!
//! On the first difference the check stops with one line `LOOP-TOKEN DEVICE MISS`: the position, the visit, the
//! state name (`stream`, `logits` or `top`), the word, both bit patterns and the count of words of that state that
//! differ. Receipt: `RESULT.txt` with the counts, and `COMPLETE` when no word differs.
//! Every size comes from the description of the checkpoint and from the trace. No line here names a model.
use std::{fs,path::Path,time::Instant};
use engine_model::plan::description::LoopTextDescription;
use spark_tp::{Result,bounded_decoder::{Compute,Host},loop_decoder::{Driver,fixture::{Fixture,SMALL,WIDE},mix::Weights,state::State}};

/// The host words of one position: the token that was fed, the stream after each visit, the logits, the top row.
#[derive(Clone)]
pub struct Want {pub token:u32,pub streams:Vec<f32>,pub logits:Vec<f32>,pub top:u32}
/// What a run compared and found equal. `tops` holds the top row of each whole position.
#[derive(Debug,Default,Clone,PartialEq,Eq)]
pub struct Counts {pub visits:usize,pub stream_words:u64,pub logit_words:u64,pub tops:Vec<u32>}
/// The first difference of a run. `visit` is None for the logits and for the top row.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Miss {pub position:usize,pub token:u32,pub visit:Option<usize>,pub state:&'static str,pub word:usize,pub host:u32,pub device:u32,pub different:usize,pub of:usize}
impl Miss {
    /// One line. `prompt` is the number of the prompt in a list of prompts. A stream and a logit are bit patterns,
    /// a top row is a row number.
    pub fn line(&self,d:&LoopTextDescription,prompt:usize)->String {
        let place=match self.visit.and_then(|visit|Some((visit,d.visits.get(visit)?))) {
            Some((visit,v))=>format!("visit={visit} layer={} pass={}",v.layer,v.pass),
            None=>"visit=end".to_owned(),
        };
        let words=if self.state=="top" {format!("host={} device={}",self.host,self.device)} else {format!("host={:08x} device={:08x}",self.host,self.device)};
        format!("LOOP-TOKEN DEVICE MISS prompt={prompt} token={} id={} {place} state={} word={} {words} different={} of={}",self.position,self.token,self.state,self.word,self.different,self.of)
    }
}
/// The first word of `device` whose bits differ from the word of `host`, and the count of such words.
fn difference(host:&[f32],device:&[f32])->Result<Option<(usize,usize)>> {
    if host.len()!=device.len() {return Err(format!("loop token: the backend gave {} words where the host run has {}",device.len(),host.len()));}
    let mut other=host.iter().zip(device).enumerate().filter(|(_,(h,v))|h.to_bits()!=v.to_bits()).map(|(index,_)|index);
    Ok(other.next().map(|first|(first,1+other.count())))
}
const STOP:&str="loop token: the check stopped at a difference";
/// Run `positions` tokens of one chat on `c` from a clean state and compare each with `want`. The run stops at
/// the first difference. `line` gets one line of text for each position that is equal.
pub fn compare<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,capacity:u64,positions:usize,want:&mut dyn FnMut(usize)->Result<Want>,line:&mut dyn FnMut(String))->Result<(Counts,Option<Miss>)> {
    let driver=Driver::new(c,w,d,capacity)?;let mut state=driver.clean()?;
    let size=|n:u64|usize::try_from(n).map_err(|_|"loop token: a size of the description exceeds this host".to_owned());
    let (width,rows,visits)=(size(d.hidden)?,size(d.head_rows)?,d.visits.len());
    let mut counts=Counts::default();
    for position in 0..positions {
        let host=want(position)?;
        if host.streams.len()!=visits*width || host.logits.len()!=rows {
            return Err(format!("loop token: position {position} of the host run has {} stream words and {} logit words; the description gives {} and {rows}",host.streams.len(),host.logits.len(),visits*width));
        }
        let (started,mut miss)=(Instant::now(),None);
        let found=|visit,name,(word,different):(usize,usize),host_word:u32,device_word:u32,of|Miss {position,token:host.token,visit,state:name,word,host:host_word,device:device_word,different,of};
        let stepped=c.work(|| {
            let step=driver.step_seen(&mut state,host.token,|visit,stream| {
                let (expected,device)=(&host.streams[visit*width..(visit+1)*width],c.read(stream)?);
                match difference(expected,&device)? {
                    None=>{counts.visits+=1;counts.stream_words+=width as u64;Ok(())},
                    Some(first)=>{miss=Some(found(Some(visit),"stream",first,expected[first.0].to_bits(),device[first.0].to_bits(),width));Err(STOP.into())},
                }
            })?;
            Ok((c.read(&step.logits)?,step.top))
        });
        let (logits,top)=match (stepped,miss) {
            (_,Some(miss))=>return Ok((counts,Some(miss))),
            (Err(error),None)=>return Err(format!("loop token: position {position}, token {}: {error}",host.token)),
            (Ok(step),None)=>step,
        };
        if let Some(first)=difference(&host.logits,&logits)? {return Ok((counts,Some(found(None,"logits",first,host.logits[first.0].to_bits(),logits[first.0].to_bits(),rows))));}
        counts.logit_words+=rows as u64;
        if top!=host.top {return Ok((counts,Some(found(None,"top",(0,1),host.top,top,1))));}
        counts.tops.push(top);
        line(format!("LOOP-TOKEN POSITION token={position} id={} top={top} visits={visits} stream_words={} logit_words={rows} equal seconds_with_reads={:.3}",host.token,visits*width,started.elapsed().as_secs_f64()));
    }
    Ok((counts,None))
}

/// Tokens of the chat on a made-up checkpoint, and the capacity of its state. The chat crosses the window edge of
/// both checkpoints.
const FIXTURE_CHAT:usize=12;
const FIXTURE_CAPACITY:u64=16;
/// One token of the driver on the host backend: the words of the token.
fn host_step<W:Weights<Host>>(driver:&Driver<'_,Host,W>,state:&mut State<Vec<f32>>,token:u32)->Result<Want> {
    Host.work(|| {
        let mut streams=Vec::new();
        let step=driver.step_seen(state,token,|_,stream| {streams.extend_from_slice(stream);Ok(())})?;
        Ok(Want {token,streams,logits:step.logits,top:step.top})
    })
}
/// The host run of the driver on a made-up checkpoint: the words of each token of `tokens`.
fn host_run(fixture:&Fixture,tokens:&[u32])->Result<Vec<Want>> {
    let driver=Driver::new(&Host,fixture,&fixture.description,FIXTURE_CAPACITY)?;let mut state=driver.clean()?;
    tokens.iter().map(|&token|host_step(&driver,&mut state,token)).collect()
}
/// The driver on `c` against the driver on the host, on each made-up checkpoint. Receipt: `RESULT.txt`, and
/// `COMPLETE` when no word differs and each changed word was found.
pub fn fixtures<C:Compute>(c:&C,out:&Path,backend:&str)->Result<()> {
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let write=|name:&str,text:&str|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
    let (mut text,mut failed)=(String::new(),0);
    for (name,config) in [("SMALL",SMALL),("WIDE",WIDE)] {
        let fixture=Fixture::new(config)?;let (d,tokens)=(&fixture.description,fixture.chat(FIXTURE_CHAT));
        let host=host_run(&fixture,&tokens)?;
        let (last,visit,width,rows)=(tokens.len()-1,d.visits.len()-1,d.hidden as usize,d.head_rows as usize);
        // One run on the words of the host, then one with the last stream word of the chat changed in the host
        // words, then one with its last logit word changed. A changed word is bit 0 of the word turned over.
        let run=|changed:Option<bool>| {
            compare(c,&fixture,d,FIXTURE_CAPACITY,tokens.len(),&mut |position| {
                let mut want=host.get(position).cloned().ok_or("loop token: the host run is shorter than the chat")?;
                if let Some(stream)=changed && position==last {
                    let word=if stream {&mut want.streams[(visit+1)*width-1]} else {&mut want.logits[rows-1]};
                    *word=f32::from_bits(word.to_bits()^1);
                }
                Ok(want)
            },&mut |_| {})
        };
        let (counts,miss)=run(None)?;
        let line=match &miss {
            None=>format!("LOOP-TOKEN FIXTURE PASS checkpoint={name} backend={backend} tokens={} visits_equal={} stream_words_equal={} logit_words_equal={} tops_equal={}",tokens.len(),counts.visits,counts.stream_words,counts.logit_words,counts.tops.len()),
            Some(miss)=>{failed+=1;format!("LOOP-TOKEN FIXTURE MISS checkpoint={name} backend={backend} tokens_equal={} {}",counts.tops.len(),miss.line(d,0))},
        };
        eprintln!("{line}");text.push_str(&line);text.push('\n');
        if miss.is_some() {continue;}
        let stream=run(Some(true))?.1.is_some_and(|m|(m.position,m.visit,m.word,m.different,m.state)==(last,Some(visit),width-1,1,"stream"));
        let logit=run(Some(false))?.1.is_some_and(|m|(m.position,m.visit,m.word,m.different,m.state)==(last,None,rows-1,1,"logits"));
        let line=format!("LOOP-TOKEN FIXTURE SEES checkpoint={name} backend={backend} changed_stream_word_found={stream} changed_logit_word_found={logit}");
        eprintln!("{line}");text.push_str(&line);text.push('\n');
        if !(stream && logit) {failed+=1;}
    }
    write("RESULT.txt",&text)?;
    if failed>0 {return Err(format!("loop token: {failed} of the fixture checks did not pass; see RESULT.txt"));}
    write("COMPLETE","")
}
/// Mode `loop-token-fixture`: the made-up checkpoints on the device.
#[cfg(target_os="linux")]
pub fn fixtures_device(out:&Path)->Result<()> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend|fixtures(&spark_tp::bounded_decoder::Device::new(backend),out,"device"))
}

/// A saved trace of a host run of one chat.
#[cfg(target_os="linux")]
pub struct Trace {folder:std::path::PathBuf,prompt:usize,fed:Vec<u32>,tops:Vec<u32>}
#[cfg(target_os="linux")]
impl Trace {
    pub fn open(folder:&Path)->Result<Self> {
        let text=|name:&str|fs::read_to_string(folder.join(name)).map_err(|e|format!("{}: {e}",folder.join(name).display()));
        let id=|file:&str,line:usize,word:&str|word.trim().parse::<u32>().map_err(|_|format!("{file} line {}: {word} is no token ID",line+1));
        let prompt=text("prompt.ids")?.lines().enumerate().map(|(index,line)|id("prompt.ids",index,line)).collect::<Result<Vec<_>>>()?;
        let mut tops=Vec::new();
        for (index,line) in text("tokens.tsv")?.lines().enumerate() {
            let (position,top)=line.split_once('\t').ok_or_else(||format!("tokens.tsv line {}: no tab",index+1))?;
            if position.parse::<usize>().ok()!=Some(index) {return Err(format!("tokens.tsv line {}: the position is not {index}",index+1));}
            tops.push(id("tokens.tsv",index,top)?);
        }
        if prompt.is_empty() || tops.len()<prompt.len() {return Err("loop token: the trace has no prompt, or fewer positions than prompt tokens".into());}
        let fed=(0..tops.len()).map(|position|if position<prompt.len() {prompt[position]} else {tops[position-1]}).collect();
        Ok(Self {folder:folder.into(),prompt:prompt.len(),fed,tops})
    }
    /// The count of positions, and how many of them are the prompt.
    pub fn positions(&self)->usize {self.tops.len()}
    pub fn prompt(&self)->usize {self.prompt}
    fn words(&self,position:usize,kind:&str,count:usize)->Result<Vec<f32>> {
        let path=self.folder.join(format!("pos{position:04}.{kind}.f32"));
        let bytes=fs::read(&path).map_err(|e|format!("{}: {e}",path.display()))?;
        if bytes.len()!=count*4 {return Err(format!("{}: {} bytes; the description gives {} words",path.display(),bytes.len(),count));}
        Ok(bytes.as_chunks::<4>().0.iter().map(|word|f32::from_le_bytes(*word)).collect())
    }
    /// The host words of one position, with the sizes that the description gives.
    pub fn want(&self,position:usize,d:&LoopTextDescription)->Result<Want> {
        let (&token,&top)=self.fed.get(position).zip(self.tops.get(position)).ok_or("loop token: the trace has no such position")?;
        let size=|n:u64|usize::try_from(n).map_err(|_|"loop token: a size of the description exceeds this host".to_owned());
        Ok(Want {token,streams:self.words(position,"streams",d.visits.len()*size(d.hidden)?)?,logits:self.words(position,"logits",size(d.head_rows)?)?,top})
    }
}
/// The receipt of a run against a trace: `RESULT.txt`, and `COMPLETE` when no word differs. A difference is an
/// error of the mode, after the receipt is written.
#[cfg(target_os="linux")]
pub fn receipt(out:&Path,d:&LoopTextDescription,trace:&Trace,source:&str,outcome:Result<(Counts,Option<Miss>)>,lines:&[String],notes:&[String])->Result<()> {
    let write=|name:&str,text:String|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
    let mut text=notes.iter().chain(lines).map(|line|format!("{line}\n")).collect::<String>();
    let (counts,miss)=match outcome {
        Ok(outcome)=>outcome,
        Err(error)=>{text.push_str(&format!("LOOP-TOKEN DEVICE ERROR {error}\n"));write("RESULT.txt",text)?;return Err(error);},
    };
    let ids=|tops:&[u32]|tops.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
    // The first new token is the top row of the last position of the prompt.
    let new=counts.tops.get(trace.prompt()-1..).unwrap_or(&[]);
    let summary=format!("source={source} positions_equal={} of={} prompt_positions={} visits_equal={} stream_words_equal={} logit_words_equal={} tops_equal={} new_tokens_equal={} new_token_ids={}",
        counts.tops.len(),trace.positions(),trace.prompt(),counts.visits,counts.stream_words,counts.logit_words,counts.tops.len(),new.len(),ids(new));
    match miss {
        None=>{
            text.push_str(&format!("LOOP-TOKEN DEVICE PASS {summary}\n"));eprintln!("LOOP-TOKEN DEVICE PASS {summary}");
            write("RESULT.txt",text)?;write("COMPLETE",String::new())
        },
        Some(miss)=>{
            let line=miss.line(d,0);text.push_str(&format!("{line}\nLOOP-TOKEN DEVICE STOPPED {summary}\n"));eprintln!("{line}\nLOOP-TOKEN DEVICE STOPPED {summary}");
            write("RESULT.txt",text)?;Err(line)
        },
    }
}
/// Bytes of decoded expert matrices that the host run of the driver keeps. A host limit: it changes no word.
#[cfg(target_os="linux")]
const HOST_EXPERT_CACHE:usize=8<<30;
/// What a mode does with the loaded checkpoint.
#[cfg(target_os="linux")]
#[derive(Clone,Copy,PartialEq,Eq)]
pub enum Job {
    /// Mode `loop-token`: compare with the saved trace of the host form.
    Trace,
    /// Mode `loop-token-driver`: compare with the driver on the host backend, in step.
    Driver,
    /// Mode `loop-token-rate`: the clock time of this many token steps after the prompt.
    Rate(usize),
}
/// The clock time of `steps` token steps after the prompt of the trace, on `c`. Each step is fed the top row of
/// the step before it, and gives only its top row. The top rows are compared with the trace where it has them.
#[cfg(target_os="linux")]
fn rate<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,trace:&Trace,steps:usize,out:&Path,notes:&[String])->Result<()> {
    let capacity=(trace.prompt()+steps) as u64;
    let driver=Driver::new(c,w,d,capacity)?;let mut state=driver.clean()?;
    let mut step=|token:u32|->Result<(u32,f64)> {
        let started=Instant::now();let top=c.work(||Ok(driver.step(&mut state,token)?.top))?;
        Ok((top,started.elapsed().as_secs_f64()))
    };
    let (mut tops,mut prompt_seconds)=(Vec::new(),0.0);
    for &token in &trace.fed[..trace.prompt()] {let (top,seconds)=step(token)?;tops.push(top);prompt_seconds+=seconds;}
    let (first,mut times)=(tops[trace.prompt()-1],Vec::with_capacity(steps));
    let whole=Instant::now();
    for _ in 0..steps {let (top,seconds)=step(tops[tops.len()-1])?;tops.push(top);times.push(seconds);}
    let seconds=whole.elapsed().as_secs_f64();
    let compared=tops.len().min(trace.positions());
    let equal=tops.iter().zip(&trace.tops).filter(|(device,host)|device==host).count();
    let mut sorted=times.clone();sorted.sort_by(f64::total_cmp);
    let ms=|seconds:f64|seconds*1000.0;
    let ids=tops[trace.prompt()..].iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
    let mut text=notes.iter().map(|line|format!("{line}\n")).collect::<String>();
    let lines=[
        format!("LOOP-TOKEN RATE token_steps={steps} seconds={seconds:.4} tokens_per_second={:.3} step_ms_mean={:.3} step_ms_smallest={:.3} step_ms_middle={:.3} step_ms_largest={:.3} prompt_positions={} prompt_seconds={prompt_seconds:.4} capacity={capacity} read_a_step=the top row only",
            steps as f64/seconds,ms(times.iter().sum::<f64>()/steps as f64),ms(sorted[0]),ms(sorted[steps/2]),ms(sorted[steps-1]),trace.prompt()),
        format!("LOOP-TOKEN RATE TOKENS first_new={first} then={ids}"),
        format!("LOOP-TOKEN RATE TRACE top_rows_compared={compared} equal={equal}"),
    ];
    for line in &lines {eprintln!("{line}");text.push_str(line);text.push('\n');}
    let write=|name:&str,text:&str|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
    write("RATE-RESULT.txt",&text)?;
    if equal!=compared {return Err(format!("loop token: {} of {compared} top rows differ from the trace in the rate run",compared-equal));}
    write("COMPLETE","")
}
/// Mode `loop-token-rate`. `steps` is the count of token steps after the prompt, as text.
#[cfg(target_os="linux")]
pub fn run_rate(model:&Path,trace:&Path,out:&Path,steps:&str,pool:bool)->Result<()> {
    run_with_pool(model,trace,out,Job::Rate(steps.parse().map_err(|_|format!("loop token: STEPS must be a whole number, got {steps}"))?),pool)
}
/// Modes `loop-token`, `loop-token-driver` and `loop-token-rate`: the real checkpoint on one device.
#[cfg(target_os="linux")]
pub fn run(model:&Path,trace:&Path,out:&Path,job:Job)->Result<()> {run_with_pool(model,trace,out,job,false)}
/// `run` with the vector pool of the backend on (`Backend::configure_vector_pool`, as a server sets it) when `pool`:
/// a freed device vector waits for the next vector of its size, so a token step takes no new allocation after the
/// first. Modes `loop-token-pooled` and `loop-token-rate-pooled`.
#[cfg(target_os="linux")]
pub fn run_with_pool(model:&Path,trace:&Path,out:&Path,job:Job,pool:bool)->Result<()> {
    use engine_model::plan::{loop_load,loop_state::{LoopLedger,LoopStatePlan,context_limit}};
    use spark_tp::{bounded_decoder::Device,loop_decoder::{host_weights::HostWeights,operands::{device::DeviceStore,load,weights::LoadedWeights}}};
    let started=Instant::now();
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let trace=Trace::open(trace)?;
    let checkpoint=loop_load::read_folder(model).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    let driver=job==Job::Driver;
    let (d,capacity)=(&checkpoint.description,if let Job::Rate(steps)=job {if steps==0 {return Err("loop token: STEPS must be above zero".into());}(trace.prompt()+steps) as u64} else {trace.positions() as u64});
    // The memory rule of the load check: the free bytes less the larger of one tenth and 2 GiB, and the ledger of
    // the weights and of one chat of the trace's length must fit.
    let bucket=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
    let state=LoopStatePlan::new(d,capacity,bucket).map_err(|e|e.to_string())?;
    let ledger=LoopLedger::new(&checkpoint.load.bytes,&state).map_err(|e|e.to_string())?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let budget=free.saturating_sub((free/10).max(2<<30)) as u64;
    let limit=context_limit(d,&checkpoint.load.bytes,bucket,budget).map_err(|e|e.to_string())?;
    ledger.admit(budget,limit).map_err(|refusal|format!("text service memory refusal: {refusal}"))?;
    let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
    let source=match job {Job::Trace=>"trace",Job::Driver=>"driver",Job::Rate(_)=>"rate"};
    let host=if driver {Some(HostWeights::open(model,HOST_EXPERT_CACHE)?)} else {None};
    let mut notes=vec![format!("LOOP-TOKEN PLAN source={source} positions={} prompt_positions={} visits={} width={} head_rows={} capacity={capacity} ledger_bytes={} budget_bytes={budget}",trace.positions(),trace.prompt(),d.visits.len(),d.hidden,d.head_rows,ledger.total)];
    eprintln!("{}",notes[0]);
    let budget=usize::try_from(budget).map_err(|_|"budget exceeds the address space")?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,budget,|backend| {
        let image=started.elapsed().as_secs_f64();
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,model,&mut store,workers)?;backend.synchronize()?;
        let files=load::Files::open(&checkpoint,model)?;
        let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
        notes.push(format!("LOOP-TOKEN LOAD tensors={} units={} fingerprint={} host_weight_bytes={} device_bytes={} seconds_to_backend={image:.3} seconds_to_weights={:.3}",
            loaded.report.tensors,loaded.report.units,loaded.fingerprint(),weights.host_bytes(),backend.allocated_bytes(),started.elapsed().as_secs_f64()));
        eprintln!("{}",notes[1]);
        if pool {backend.configure_vector_pool(1<<30);notes.push("LOOP-TOKEN POOL vector_pool=on margin_bytes=1073741824".into());}
        let device=Device::new(backend);
        if let Job::Rate(steps)=job {return rate(&device,&weights,d,&trace,steps,out,&notes);}
        let mut host_run=match &host {
            Some((description,words))=>{let driver=Driver::new(&Host,words,description,capacity)?;let state=driver.clean()?;Some((driver,state))},
            None=>None,
        };
        let (mut lines,mut apart)=(Vec::new(),Vec::new());
        // The words that the device must give at a position. With the driver as the source, the host run of the
        // driver makes them now, and a line says where they stand apart from the trace.
        let mut want=|position:usize|->Result<Want> {
            let saved=trace.want(position,d)?;
            let Some((driver,state))=&mut host_run else {return Ok(saved);};
            let made=host_step(driver,state,saved.token)?;
            let differ=|a:&[f32],b:&[f32]|a.len()!=b.len() || a.iter().zip(b).any(|(a,b)|a.to_bits()!=b.to_bits());
            let width=d.hidden as usize;
            let visit=made.streams.chunks(width).zip(saved.streams.chunks(width)).position(|(a,b)|differ(a,b));
            let logits=made.logits.iter().zip(&saved.logits).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
            if visit.is_some() || logits>0 || made.top!=saved.top {
                apart.push(format!("LOOP-TOKEN DRIVER APART FROM TRACE token={position} id={} first_visit={} logit_words={logits} top_driver={} top_trace={}",saved.token,visit.map_or("none".to_owned(),|v|v.to_string()),made.top,saved.top));
            }
            Ok(made)
        };
        let outcome=compare(&device,&weights,d,capacity,trace.positions(),&mut want,&mut |text| {eprintln!("{text}");lines.push(text);});
        if driver {
            for line in &apart {eprintln!("{line}");}
            lines.push(format!("LOOP-TOKEN DRIVER AND TRACE positions_apart={}",apart.len()));lines.append(&mut apart);
        }
        receipt(out,d,&trace,source,outcome,&lines,&notes)
    })
}
