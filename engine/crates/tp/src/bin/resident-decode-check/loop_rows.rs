//! The rows check of the block family with a loop, level 3 of the check program of the loop family: the decode
//! driver (`spark_tp::loop_decoder::Driver`) on one backend in steps of many rows (`Driver::step_rows`) against
//! the words of a host run one row a step.
//!
//! Mode `loop-rows MODEL TRACE OUT ROWS PROMPT` (needs a GPU): the checkpoint of `MODEL` is loaded on one device.
//! `ROWS` is a list of step row limits, as `1,4,16`. For each limit `R` the mode runs two parts:
//! - The check. A driver with step row limit `R` reads the prompt of the trace in steps of `R` rows (the last step
//!   takes what is left), then each later position of the trace as a step of one row. After each visit the stream
//!   of all rows of the step is read from the device, and each row is compared with the words of its position in
//!   the trace, then its logits and its top row. Each word must be equal. This is the comparison of mode
//!   `loop-token` (`loop_token.rs`, which also describes the trace folder), one row at a time.
//! - The rate. A prompt of `PROMPT` rows is read from a clean state in steps of `R` rows. No stream and no logit
//!   word is read. Only the last row runs the end, and the clock stops when its top row is on the host. The line
//!   gives the metric of the rows work: prompt rows a second. The prompt is the prompt of the trace; when `PROMPT`
//!   is larger, its token IDs are taken again from the start. `PROMPT` 0 runs no rate part.
//!
//! Before the limits, the same prompt runs one row a step on the path of a token step (`Driver::read`, and
//! `Driver::step` for the last row). Its top row is the reference: the last row of each rate run must give it.
//!
//! After the check of a limit, mode `loop-rows` runs it once more with the positions after the prompt on the path
//! of a token step (`Driver::step_seen`) on the same state: a state that steps of rows made must serve a token
//! step too. A token step writes its rings in place, and a step of rows leaves the rings as gathered values.
//!
//! Mode `loop-rows-long MODEL REFERENCE OUT ROWS POSITIONS` (needs a GPU): the same two parts against a long
//! reference of the whole-model host form. Its folder holds a forced read: `hashes.tsv` (a line a position: the
//! position, the fed token ID, the top row, then three SHA-256 of little-endian 32-bit words, of which the last
//! is that of the logits) and `visits.tsv` (the position, then one SHA-256 for the stream after each visit).
//! Every position is a prompt row. The check reads the first `POSITIONS` of them (0: all) in steps of `R` rows
//! and compares the hash of each row's stream after each visit, the hash of its logits and its top row. The rate
//! part reads the same rows with no read-back; the top row of the last row must be the one of the reference. No
//! run one row a step is made here.
//!
//! Mode `loop-rows-fixture OUT` (needs a GPU) and `loop-rows-fixture-host OUT` (any host): the two made-up
//! checkpoints of the driver (`loop_decoder::fixture`), no model file. One chat runs on the backend in steps of
//! 1, 2, 3, 5 and 12 rows, with the byte limit of the keys of a token step and with a limit of 1 byte (one row a
//! group), and each row is compared with the driver on the host backend one row a step. Each run also shows that
//! the check can see: one changed stream word and one changed logit word of the host run must be found at their
//! place. Then the chat is read as a prompt on both paths of the rate part, and compared once against hashes of
//! the host words (the form of a long reference).
//!
//! On the first difference a check stops with one line `LOOP-ROWS DEVICE MISS`: the position, the step that holds
//! it, the visit, the state name (`stream`, `logits` or `top`), the word, both bit patterns and the count of words
//! of that state of that row that differ. Against hashes the word is 0, the patterns are the first 32 bits of
//! both hashes and the count is 0 (not known). Receipt: `RESULT.txt`, and `COMPLETE` when no word differs and
//! each rate run gave the reference top row.
//! Every size comes from the description of the checkpoint, from the trace and from the command line. No line
//! here names a model.
use std::{fs,path::Path,time::Instant};
use engine_model::plan::description::LoopTextDescription;
use spark_tp::{Result,bounded_decoder::{Compute,Host},loop_decoder::{ChatStep,Driver,Wanted,fixture::{Fixture,SMALL,WIDE},mix::Weights,rows::Limits}};
use crate::loop_token::Want;

/// What a check compared and found equal.
#[derive(Debug,Default,Clone,PartialEq,Eq)]
pub struct Counts {pub steps:usize,pub positions:usize,pub stream_words:u64,pub logit_words:u64}
/// The first difference of a check. `first` and `rows` name the step that holds the position. `visit` is None for
/// the logits and for the top row.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Miss {pub position:usize,pub token:u32,pub first:usize,pub rows:usize,pub visit:Option<usize>,pub state:&'static str,pub word:usize,pub host:u32,pub device:u32,pub different:usize,pub of:usize}
impl Miss {
    /// One line. A stream and a logit are bit patterns, a top row is a row number.
    pub fn line(&self)->String {
        let place=self.visit.map_or("visit=end".to_owned(),|visit|format!("visit={visit}"));
        let words=if self.state=="top" {format!("host={} device={}",self.host,self.device)} else {format!("host={:08x} device={:08x}",self.host,self.device)};
        format!("LOOP-ROWS DEVICE MISS token={} id={} step_first={} step_rows={} {place} state={} word={} {words} different={} of={}",self.position,self.token,self.first,self.rows,self.state,self.word,self.different,self.of)
    }
}
/// The first word of `device` whose bits differ from the word of `host`, and the count of such words.
fn difference(host:&[f32],device:&[f32])->Option<(usize,usize)> {
    let mut other=host.iter().zip(device).enumerate().filter(|(_,(h,v))|h.to_bits()!=v.to_bits()).map(|(index,_)|index);
    other.next().map(|first|(first,1+other.count()))
}
/// SHA-256 of words as little-endian bytes, in hex: the hash of a long reference.
#[cfg(target_os="linux")]
fn hash(words:&[f32])->String {crate::sha256::digest(&words.iter().flat_map(|word|word.to_le_bytes()).collect::<Vec<_>>())}
/// On a host with no GPU the check program holds no SHA-256, and no long reference can run there. The fixture
/// mode only has to tell the words of two rows apart: a plain 64-bit hash of the bit patterns.
#[cfg(not(target_os="linux"))]
fn hash(words:&[f32])->String {format!("{:016x}",words.iter().fold(0xcbf2_9ce4_8422_2325u64,|sum,word|(sum^u64::from(word.to_bits())).wrapping_mul(0x0100_0000_01b3)))}
/// The first 32 bits of a hash in hex, for the line of a difference.
fn head32(hex:&str)->u32 {hex.get(..8).and_then(|head|u32::from_str_radix(head,16).ok()).unwrap_or(0)}
/// What the host run gives for one position: its words, or the hash of the stream after each visit and the hash
/// of the logits.
pub enum Expect {Words(Want),Hashes {token:u32,visits:Vec<String>,logits:String,top:u32}}
impl Expect {
    fn token(&self)->u32 {match self {Self::Words(want)=>want.token,Self::Hashes {token,..}=>*token}}
    fn top(&self)->u32 {match self {Self::Words(want)=>want.top,Self::Hashes {top,..}=>*top}}
    /// True when the position has what a chat of `visits` visits, `width` stream words and `head` logits needs.
    fn fits(&self,visits:usize,width:usize,head:usize)->bool {
        match self {Self::Words(want)=>want.streams.len()==visits*width && want.logits.len()==head,Self::Hashes {visits:hashes,..}=>hashes.len()==visits}
    }
    /// Where `given` differs from the stream after `visit` (None: from the logits): the first word and the count
    /// of words that differ, then both bit patterns. Against a hash: word 0, count 0, the heads of both hashes.
    fn differs(&self,visit:Option<usize>,width:usize,given:&[f32])->Option<((usize,usize),u32,u32)> {
        match self {
            Self::Words(want)=>{
                let expected=match visit {Some(visit)=>&want.streams[visit*width..(visit+1)*width],None=>&want.logits[..]};
                difference(expected,given).map(|at|(at,expected[at.0].to_bits(),given[at.0].to_bits()))
            },
            Self::Hashes {visits,logits,..}=>{
                let (expected,made)=(visit.map_or(logits,|visit|&visits[visit]),hash(given));
                (*expected!=made).then(||((0,0),head32(expected),head32(&made)))
            },
        }
    }
}
/// How a chat of `positions` tokens is cut into steps: the first `prompt` of them in steps of `rows` rows (the
/// last of these takes what is left), each later one as a step of one row.
pub fn cuts(prompt:usize,positions:usize,rows:usize)->Vec<usize> {
    let (mut steps,mut left)=(Vec::new(),prompt.min(positions));
    while left>0 {let step=left.min(rows.max(1));steps.push(step);left-=step;}
    steps.extend(std::iter::repeat_n(1,positions.saturating_sub(prompt)));steps
}
const STOP:&str="loop rows: the check stopped at a difference";
/// Run one chat on `c` from a clean state in the steps of `cuts` and compare each row with `want` (the host words
/// of a position). The run stops at the first difference. `line` gets one line of text for each equal step.
/// With `token_path`, a step of one row at that position or after it runs on the path of a token step
/// (`Driver::step_seen`) on the same state.
pub fn compare<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,capacity:u64,limits:Limits,cuts:&[usize],token_path:Option<usize>,want:&mut dyn FnMut(usize)->Result<Expect>,line:&mut dyn FnMut(String))->Result<(Counts,Option<Miss>)> {
    let driver=Driver::with_limits(c,w,d,capacity,limits)?;let mut state=driver.clean()?;
    let size=|n:u64|usize::try_from(n).map_err(|_|"loop rows: a size of the description exceeds this host".to_owned());
    let (width,head,visits)=(size(d.hidden)?,size(d.head_rows)?,d.visits.len());
    let (mut counts,mut first)=(Counts::default(),0usize);
    for &rows in cuts {
        let hosts=(first..first+rows).map(&mut *want).collect::<Result<Vec<_>>>()?;
        if let Some(bad)=hosts.iter().position(|host|!host.fits(visits,width,head)) {
            return Err(format!("loop rows: position {} of the host run has not the words or hashes that the description gives: {visits} visits of {width} stream words, {head} logits",first+bad));
        }
        let tokens=hosts.iter().map(Expect::token).collect::<Vec<_>>();
        let (started,mut miss)=(Instant::now(),None);
        let found=|row:usize,visit,name,(word,different):(usize,usize),host_word:u32,device_word:u32,of|Miss {position:first+row,token:tokens[row],first,rows,visit,state:name,word,host:host_word,device:device_word,different,of};
        let stepped=c.work(|| {
            let mut seen=|visit:usize,stream:&C::Value|->Result<()> {
                let device=c.read(stream)?;
                if device.len()!=rows*width {return Err(format!("loop rows: the stream after visit {visit} has {} words; {rows} rows have {}",device.len(),rows*width));}
                for (row,host) in hosts.iter().enumerate() {
                    if let Some((at,host_word,device_word))=host.differs(Some(visit),width,&device[row*width..(row+1)*width]) {
                        miss=Some(found(row,Some(visit),"stream",at,host_word,device_word,width));return Err(STOP.into());
                    }
                }
                counts.stream_words+=(rows*width) as u64;Ok(())
            };
            if rows==1 && token_path.is_some_and(|from|first>=from) {
                let step=driver.step_seen(&mut state,tokens[0],&mut seen)?;
                return Ok((c.read(&step.logits)?,vec![step.top]));
            }
            let mut chats=[ChatStep {state:&mut state,tokens:&tokens,wanted:Wanted::All,logits:true}];
            let step=driver.step_rows_seen(&mut chats,&mut seen)?;
            Ok((c.read(step.logits.as_ref().ok_or("loop rows: the step gave no logits")?)?,step.tops))
        });
        let (logits,tops)=match (stepped,miss) {
            (_,Some(miss))=>return Ok((counts,Some(miss))),
            (Err(error),None)=>return Err(format!("loop rows: the step of {rows} rows at position {first}: {error}")),
            (Ok(step),None)=>step,
        };
        if logits.len()!=rows*head || tops.len()!=rows {return Err(format!("loop rows: the step of {rows} rows at position {first} gave {} logit words and {} top rows",logits.len(),tops.len()));}
        for (row,host) in hosts.iter().enumerate() {
            if let Some((at,host_word,device_word))=host.differs(None,width,&logits[row*head..(row+1)*head]) {return Ok((counts,Some(found(row,None,"logits",at,host_word,device_word,head))));}
            counts.logit_words+=head as u64;
            if tops[row]!=host.top() {return Ok((counts,Some(found(row,None,"top",(0,1),host.top(),tops[row],1))));}
            counts.positions+=1;
        }
        counts.steps+=1;
        line(format!("LOOP-ROWS STEP first={first} rows={rows} equal visits={visits} stream_words={} logit_words={} seconds_with_reads={:.3}",visits*rows*width,rows*head,started.elapsed().as_secs_f64()));
        first+=rows;
    }
    Ok((counts,None))
}
/// Read `prompt` from a clean state and give the top row of its last row, the clock seconds until that row is on
/// the host, and the count of steps. No stream and no logit word is read.
/// - `limits` None: one row a step on the path of a token step (`Driver::read`, then `Driver::step`).
/// - Else: steps of `limits.step_rows` rows (`Driver::step_rows`); only the last row of the prompt runs the end.
///
/// `settle` waits for the backend before the clock starts, so the clean state is not in the time.
pub fn read_prompt<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,prompt:&[u32],limits:Option<Limits>,settle:&dyn Fn()->Result<()>)->Result<(u32,f64,usize)> {
    let (last,capacity)=(prompt.len().checked_sub(1).ok_or("loop rows: the prompt has no token")?,prompt.len() as u64);
    let Some(limits)=limits else {
        let driver=Driver::new(c,w,d,capacity)?;let mut state=driver.clean()?;
        settle()?;let started=Instant::now();
        for &token in &prompt[..last] {c.work(||driver.read(&mut state,token))?;}
        let top=c.work(||Ok(driver.step(&mut state,prompt[last])?.top))?;
        return Ok((top,started.elapsed().as_secs_f64(),prompt.len()));
    };
    let driver=Driver::with_limits(c,w,d,capacity,limits)?;let mut state=driver.clean()?;
    settle()?;let (started,mut top,mut steps)=(Instant::now(),None,0);
    for tokens in prompt.chunks(limits.step_rows.max(1)) {
        steps+=1;
        let end=steps*limits.step_rows.max(1)>=prompt.len();
        let step=c.work(||driver.step_rows(&mut [ChatStep {state:&mut state,tokens,wanted:if end {Wanted::Last} else {Wanted::None},logits:false}]))?;
        if end {top=step.tops.first().copied();}
    }
    Ok((top.ok_or("loop rows: the last step of the prompt gave no top row")?,started.elapsed().as_secs_f64(),steps))
}

/// Tokens of the chat on a made-up checkpoint, and the capacity of its state. The chat crosses the window edge of
/// both checkpoints.
const FIXTURE_CHAT:usize=12;
const FIXTURE_CAPACITY:u64=16;
/// The step row limits of the fixture runs. The last is the whole chat in one step.
const FIXTURE_ROWS:[usize;5]=[1,2,3,5,12];
/// The host run of the driver on a made-up checkpoint, one row a step: the words of each token of `tokens`.
fn host_run(fixture:&Fixture,tokens:&[u32])->Result<Vec<Want>> {
    let driver=Driver::new(&Host,fixture,&fixture.description,FIXTURE_CAPACITY)?;let mut state=driver.clean()?;
    tokens.iter().map(|&token|Host.work(|| {
        let mut streams=Vec::new();
        let step=driver.step_seen(&mut state,token,|_,stream| {streams.extend_from_slice(stream);Ok(())})?;
        Ok(Want {token,streams,logits:step.logits,top:step.top})
    })).collect()
}
/// The driver on `c` in steps of rows against the driver on the host one row a step, on each made-up checkpoint.
/// Receipt: `RESULT.txt`, and `COMPLETE` when no word differs, each changed word was found and each prompt read
/// gave the top row of the host run.
pub fn fixtures<C:Compute>(c:&C,out:&Path,backend:&str)->Result<()> {
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let write=|name:&str,text:&str|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
    let (mut text,mut failed)=(String::new(),0);
    let mut say=|line:String| {eprintln!("{line}");text.push_str(&line);text.push('\n');};
    for (name,config) in [("SMALL",SMALL),("WIDE",WIDE)] {
        let fixture=Fixture::new(config)?;let (d,tokens)=(&fixture.description,fixture.chat(FIXTURE_CHAT));
        let host=host_run(&fixture,&tokens)?;
        let (last,visit,width,head)=(tokens.len()-1,d.visits.len()-1,d.hidden as usize,d.head_rows as usize);
        for rows in FIXTURE_ROWS {
            for key_bytes in [Limits::ONE.key_bytes,1] {
                let (limits,steps)=(Limits {step_rows:rows,key_bytes},cuts(tokens.len(),tokens.len(),rows));
                // One run on the words of the host, then one with the last stream word of the chat changed in the
                // host words, then one with its last logit word changed. A changed word is bit 0 turned over.
                let run=|changed:Option<bool>| {
                    compare(c,&fixture,d,FIXTURE_CAPACITY,limits,&steps,None,&mut |position| {
                        let mut want=host.get(position).cloned().ok_or("loop rows: the host run is shorter than the chat")?;
                        if let Some(stream)=changed && position==last {
                            let word=if stream {&mut want.streams[(visit+1)*width-1]} else {&mut want.logits[head-1]};
                            *word=f32::from_bits(word.to_bits()^1);
                        }
                        Ok(Expect::Words(want))
                    },&mut |_| {})
                };
                let (counts,miss)=run(None)?;
                match &miss {
                    None=>say(format!("LOOP-ROWS FIXTURE PASS checkpoint={name} backend={backend} rows={rows} key_bytes={key_bytes} steps={} positions_equal={} stream_words_equal={} logit_words_equal={}",counts.steps,counts.positions,counts.stream_words,counts.logit_words)),
                    Some(miss)=>{failed+=1;say(format!("LOOP-ROWS FIXTURE MISS checkpoint={name} backend={backend} rows={rows} key_bytes={key_bytes} positions_equal={} {}",counts.positions,miss.line()));continue;},
                }
                let stream=run(Some(true))?.1.is_some_and(|m|(m.position,m.visit,m.word,m.different,m.state)==(last,Some(visit),width-1,1,"stream"));
                let logit=run(Some(false))?.1.is_some_and(|m|(m.position,m.visit,m.word,m.different,m.state)==(last,None,head-1,1,"logits"));
                say(format!("LOOP-ROWS FIXTURE SEES checkpoint={name} backend={backend} rows={rows} key_bytes={key_bytes} changed_stream_word_found={stream} changed_logit_word_found={logit}"));
                if !(stream && logit) {failed+=1;}
            }
        }
        // The same compare against hashes of the host words, in steps of 5 rows; then with one changed hash.
        let hashed=|position:usize,changed:bool|->Result<Expect> {
            let want=host.get(position).ok_or("loop rows: the host run is shorter than the chat")?;
            let mut visits=want.streams.chunks(width).map(hash).collect::<Vec<_>>();
            if changed && position==last {visits[visit]=hash(&[0.0]);}
            Ok(Expect::Hashes {token:want.token,visits,logits:hash(&want.logits),top:want.top})
        };
        let (limits,steps)=(Limits {step_rows:5,..Limits::ONE},cuts(tokens.len(),tokens.len(),5));
        let (counts,miss)=compare(c,&fixture,d,FIXTURE_CAPACITY,limits,&steps,None,&mut |position|hashed(position,false),&mut |_| {})?;
        let sees=compare(c,&fixture,d,FIXTURE_CAPACITY,limits,&steps,None,&mut |position|hashed(position,true),&mut |_| {})?.1.is_some_and(|m|(m.position,m.visit,m.state)==(last,Some(visit),"stream"));
        say(format!("LOOP-ROWS FIXTURE HASHES checkpoint={name} backend={backend} rows=5 positions={} positions_equal={} changed_hash_found={sees}",tokens.len(),counts.positions));
        if miss.is_some() || counts.positions!=tokens.len() || !sees {failed+=1;}
        // Steps of rows, then token steps on the same state: 7 positions in steps of 5 and of 2 rows, then each
        // later position on the path of a token step.
        let steps=cuts(7,tokens.len(),5);
        let mixed=compare(c,&fixture,d,FIXTURE_CAPACITY,limits,&steps,Some(7),&mut |position|host.get(position).cloned().map(Expect::Words).ok_or_else(||"loop rows: the host run is shorter than the chat".to_owned()),&mut |_| {});
        match mixed {
            Ok((counts,None))=>say(format!("LOOP-ROWS FIXTURE TOKEN PATH AFTER ROWS PASS checkpoint={name} backend={backend} rows=5 positions={} positions_equal={}",tokens.len(),counts.positions)),
            Ok((counts,Some(miss)))=>{failed+=1;say(format!("LOOP-ROWS FIXTURE TOKEN PATH AFTER ROWS MISS checkpoint={name} backend={backend} rows=5 positions_equal={} {}",counts.positions,miss.line()));},
            Err(error)=>{failed+=1;say(format!("LOOP-ROWS FIXTURE TOKEN PATH AFTER ROWS ERROR checkpoint={name} backend={backend} rows=5 {error}"));},
        }
        let reads=std::iter::once(None).chain(FIXTURE_ROWS.into_iter().map(|rows|Some(Limits {step_rows:rows,..Limits::ONE})));
        for limits in reads {
            let (top,_,steps)=read_prompt(c,&fixture,d,&tokens,limits,&||Ok(()))?;
            let (rows,equal)=(limits.map_or("token_steps".to_owned(),|l|l.step_rows.to_string()),top==host[last].top);
            say(format!("LOOP-ROWS FIXTURE READ checkpoint={name} backend={backend} rows={rows} prompt_rows={} steps={steps} top_of_last_row={top} equal_to_host_run={equal}",tokens.len()));
            if !equal {failed+=1;}
        }
    }
    write("RESULT.txt",&text)?;
    if failed>0 {return Err(format!("loop rows: {failed} of the fixture checks did not pass; see RESULT.txt"));}
    write("COMPLETE","")
}
/// Mode `loop-rows-fixture`: the made-up checkpoints on the device.
#[cfg(target_os="linux")]
pub fn fixtures_device(out:&Path)->Result<()> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend|fixtures(&spark_tp::bounded_decoder::Device::new(backend),out,"device"))
}
/// A long reference of the whole-model host form: the fed token and the top row of each position, the hash of the
/// stream after each visit, and the hash of the logits.
#[cfg(target_os="linux")]
struct Long {fed:Vec<u32>,tops:Vec<u32>,visits:Vec<Vec<String>>,logits:Vec<String>}
#[cfg(target_os="linux")]
impl Long {
    fn open(folder:&Path,visits:usize)->Result<Self> {
        let text=|name:&str|fs::read_to_string(folder.join(name)).map_err(|e|format!("{}: {e}",folder.join(name).display()));
        let (mut fed,mut tops,mut logits,mut streams)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());
        for (index,line) in text("hashes.tsv")?.lines().enumerate() {
            let cells=line.split('\t').collect::<Vec<_>>();
            let id=|cell:&str|cell.parse::<u32>().map_err(|_|format!("hashes.tsv line {}: {cell} is no token ID",index+1));
            if cells.len()!=6 || cells[0].parse::<usize>().ok()!=Some(index) {return Err(format!("hashes.tsv line {}: not position {index} with 5 more cells",index+1));}
            fed.push(id(cells[1])?);tops.push(id(cells[2])?);logits.push(cells[5].to_owned());
        }
        for (index,line) in text("visits.tsv")?.lines().enumerate() {
            let mut cells=line.split('\t');
            if cells.next().and_then(|cell|cell.parse::<usize>().ok())!=Some(index) {return Err(format!("visits.tsv line {}: the position is not {index}",index+1));}
            let hashes=cells.map(str::to_owned).collect::<Vec<_>>();
            if hashes.len()!=visits {return Err(format!("visits.tsv line {}: {} hashes; the description has {visits} visits",index+1,hashes.len()));}
            streams.push(hashes);
        }
        if fed.is_empty() || streams.len()!=fed.len() {return Err("loop rows: the long reference has no position, or its two tables differ in length".into());}
        Ok(Self {fed,tops,visits:streams,logits})
    }
    fn expect(&self,position:usize)->Result<Expect> {
        match (self.fed.get(position),self.tops.get(position),self.visits.get(position),self.logits.get(position)) {
            (Some(&token),Some(&top),Some(visits),Some(logits))=>Ok(Expect::Hashes {token,visits:visits.clone(),logits:logits.clone(),top}),
            _=>Err("loop rows: the long reference has no such position".into()),
        }
    }
}
/// Modes `loop-rows` and `loop-rows-long` (`long`): the real checkpoint on one device. `rows` is the list of step
/// row limits. `prompt` is the count of prompt rows of the rate part, or for a long reference the count of its
/// positions that the run reads (0: all). Both are text.
#[cfg(target_os="linux")]
pub fn run(model:&Path,trace:&Path,out:&Path,rows:&str,prompt:&str,long:bool)->Result<()> {
    use engine_model::plan::{loop_load,loop_state::{LoopLedger,LoopStatePlan,context_limit}};
    use spark_tp::{bounded_decoder::Device,loop_decoder::operands::{device::DeviceStore,load,weights::LoadedWeights}};
    let started=Instant::now();
    let limits=rows.split(',').map(|word|word.trim().parse::<usize>().ok().filter(|&n|n>0).map(|step_rows|Limits {step_rows,..Limits::ONE})
        .ok_or_else(||format!("loop rows: ROWS is a list of counts above zero, as 1,4,16; got {rows}"))).collect::<Result<Vec<_>>>()?;
    let prompt_rows=prompt.parse::<usize>().map_err(|_|format!("loop rows: PROMPT is a count of prompt rows (0: no rate part), got {prompt}"))?;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let checkpoint=loop_load::read_folder(model).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    let d=&checkpoint.description;
    // The host run of the check, its positions and how many of them are read in steps of rows; the rows of the
    // rate part; the top row that its last row must give when the host run names it.
    let (source,expects,positions,prompt_positions,ids,known):(&str,Vec<Expect>,usize,usize,Vec<u32>,Option<u32>)=if long {
        let reference=Long::open(trace,d.visits.len())?;
        let positions=if prompt_rows==0 {reference.fed.len()} else {prompt_rows.min(reference.fed.len())};
        ("long",(0..positions).map(|position|reference.expect(position)).collect::<Result<Vec<_>>>()?,positions,positions,reference.fed[..positions].to_vec(),Some(reference.tops[positions-1]))
    } else {
        let trace=crate::loop_token::Trace::open(trace)?;
        let hosts=(0..trace.positions()).map(|position|trace.want(position,d)).collect::<Result<Vec<_>>>()?;
        let ids=(0..prompt_rows).map(|row|hosts[row%trace.prompt()].token).collect::<Vec<_>>();
        ("trace",hosts.into_iter().map(Expect::Words).collect(),trace.positions(),trace.prompt(),ids,None)
    };
    let prompt_rows=ids.len();
    let expect=|position:usize|->Result<Expect> {
        Ok(match expects.get(position).ok_or("loop rows: the host run has no such position")? {
            Expect::Words(want)=>Expect::Words(want.clone()),
            Expect::Hashes {token,visits,logits,top}=>Expect::Hashes {token:*token,visits:visits.clone(),logits:logits.clone(),top:*top},
        })
    };
    let capacity=positions.max(prompt_rows) as u64;
    // The memory rule of the load check: the free bytes less the larger of one tenth and 2 GiB, and the ledger of
    // the weights and of one chat of the longest length of this run must fit. The ledger does not count the rows
    // that a ring gets for a step row limit above 1: line LIMIT gives the state bytes of each limit.
    let bucket=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
    let state=LoopStatePlan::new(d,capacity,bucket).map_err(|e|e.to_string())?;
    let ledger=LoopLedger::new(&checkpoint.load.bytes,&state).map_err(|e|e.to_string())?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let budget=free.saturating_sub((free/10).max(2<<30)) as u64;
    let limit=context_limit(d,&checkpoint.load.bytes,bucket,budget).map_err(|e|e.to_string())?;
    ledger.admit(budget,limit).map_err(|refusal|format!("text service memory refusal: {refusal}"))?;
    let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
    let mut lines=vec![format!("LOOP-ROWS PLAN backend=device source={source} positions={positions} prompt_positions={prompt_positions} rate_prompt_rows={prompt_rows} step_row_limits={rows} visits={} width={} head_rows={} capacity={capacity} ledger_bytes={} budget_bytes={budget}",d.visits.len(),d.hidden,d.head_rows,ledger.total)];
    eprintln!("{}",lines[0]);
    let budget=usize::try_from(budget).map_err(|_|"budget exceeds the address space")?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,budget,|backend| {
        let image=started.elapsed().as_secs_f64();
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,model,&mut store,workers)?;backend.synchronize()?;
        let files=load::Files::open(&checkpoint,model)?;
        let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
        let mut say=|line:String| {eprintln!("{line}");lines.push(line);};
        say(format!("LOOP-ROWS LOAD tensors={} units={} fingerprint={} host_weight_bytes={} device_bytes={} seconds_to_backend={image:.3} seconds_to_weights={:.3}",
            loaded.report.tensors,loaded.report.units,loaded.fingerprint(),weights.host_bytes(),backend.allocated_bytes(),started.elapsed().as_secs_f64()));
        let (device,mut failed)=(Device::new(backend),0);
        let settle=||backend.synchronize();
        let one=Driver::new(&device,&weights,d,capacity)?.state_words();
        let reference=if prompt_rows==0 || known.is_some() {known} else {
            let (top,seconds,steps)=read_prompt(&device,&weights,d,&ids,None,&settle)?;
            say(format!("LOOP-ROWS RATE rows=token_steps prompt_rows={prompt_rows} steps={steps} seconds={seconds:.4} prompt_rows_a_second={:.3} top_of_last_row={top}",prompt_rows as f64/seconds));
            Some(top)
        };
        let against=if known.is_some() {"equal_to_reference"} else {"equal_to_token_steps"};
        for limits in &limits {
            let rows=limits.step_rows;
            let words=Driver::with_limits(&device,&weights,d,capacity,*limits)?.state_words();
            say(format!("LOOP-ROWS LIMIT rows={rows} key_bytes={} state_bytes_of_one_chat={} with_one_row_a_step={}",limits.key_bytes,words*4,one*4));
            let (steps,mut equal)=(cuts(prompt_positions,positions,rows),Vec::new());
            let outcome=compare(&device,&weights,d,positions as u64,*limits,&steps,None,&mut |position|expect(position),&mut |text|equal.push(text));
            for line in equal {say(line);}
            match outcome {
                Ok((counts,None))=>say(format!("LOOP-ROWS CHECK PASS rows={rows} steps={} positions={positions} positions_equal={} stream_words_equal={} logit_words_equal={}",counts.steps,counts.positions,counts.stream_words,counts.logit_words)),
                Ok((counts,Some(miss)))=>{failed+=1;say(format!("LOOP-ROWS CHECK MISS rows={rows} positions={positions} positions_equal={} {}",counts.positions,miss.line()));},
                Err(error)=>{failed+=1;say(format!("LOOP-ROWS CHECK ERROR rows={rows} positions={positions} {error}"));},
            }
            // The same chat with the positions after the prompt on the path of a token step, on the same state.
            if prompt_positions<positions {
                match compare(&device,&weights,d,positions as u64,*limits,&steps,Some(prompt_positions),&mut |position|expect(position),&mut |_| {}) {
                    Ok((counts,None))=>say(format!("LOOP-ROWS CHECK TOKEN PATH AFTER ROWS PASS rows={rows} positions={positions} positions_equal={} token_steps={}",counts.positions,positions-prompt_positions)),
                    Ok((counts,Some(miss)))=>{failed+=1;say(format!("LOOP-ROWS CHECK TOKEN PATH AFTER ROWS MISS rows={rows} positions={positions} positions_equal={} {}",counts.positions,miss.line()));},
                    Err(error)=>{failed+=1;say(format!("LOOP-ROWS CHECK TOKEN PATH AFTER ROWS ERROR rows={rows} positions={positions} {error}"));},
                }
            }
            let Some(reference)=reference else {continue;};
            match read_prompt(&device,&weights,d,&ids,Some(*limits),&settle) {
                Ok((top,seconds,steps))=>{
                    let same=top==reference;if !same {failed+=1;}
                    say(format!("LOOP-ROWS RATE rows={rows} prompt_rows={prompt_rows} steps={steps} seconds={seconds:.4} prompt_rows_a_second={:.3} top_of_last_row={top} {against}={same}",prompt_rows as f64/seconds));
                },
                Err(error)=>{failed+=1;say(format!("LOOP-ROWS RATE ERROR rows={rows} prompt_rows={prompt_rows} {error}"));},
            }
        }
        let write=|name:&str,text:String|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
        write("RESULT.txt",lines.iter().map(|line|format!("{line}\n")).collect())?;
        if failed>0 {return Err(format!("loop rows: {failed} of the checks and rate runs did not pass; see RESULT.txt"));}
        write("COMPLETE",String::new())
    })
}
