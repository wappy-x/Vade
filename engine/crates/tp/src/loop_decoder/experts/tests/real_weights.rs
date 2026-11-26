//! The whole driver on the real stored weights, on the host backend, against the whole-model host form of the
//! model crate (`engine_model::loop_host`). Ignored by default: the weight file and the prompt are private. They
//! stay outside the tracked tree, in the ignored folder `.local/loop-host/` at the repository root. It is the folder
//! of the host tests of the model crate: `model/` is the model folder and `ref/` is the reference folder with
//! `sets/contract-A/`. A link is enough for each.
//!
//! The weights of the driver come from `host_weights::HostWeights`. It reads each stored tensor through the file
//! reader of the host form and puts it into the layout of its `Weight` name. This is a second reading of the stored
//! layouts: the host form has its own, and so has the fixture.
//!
//! The test reads no environment variable. It prints lines that start with a capital word (`HOST`, `LOAD`,
//! `TOKEN`, `FIRST`, `MISS`, `SUMMARY`). A saved copy of the output is the receipt.
use super::super::super::{ChatStep,Driver,Wanted,host_weights::HostWeights,meter::{Meter,ROW_ARMS},rows};
use crate::bounded_decoder::Host;
use engine_model::loop_host::{Head,Limits,Model,Want};
use std::{path::{Path,PathBuf},time::Instant};

/// The limit of each expert cache in this test, in GiB: first that of the host form, then that of the driver's
/// weights. A host resource limit: it changes no word.
const CACHE_GIB:usize=8;
/// The prompt of the first-token check of the host form: 16 token IDs.
const PROMPT:&str="sets/contract-A/own-1-one-token__default/INPUT_IDS";

fn local(name:&str)->PathBuf {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host").join(name);
    assert!(path.exists(),"put {name} in .local/loop-host/ for the ignored host tests");path
}
/// Equal words of two rows and the first word that differs.
fn equal(mine:&[f32],theirs:&[f32])->(usize,Option<usize>) {
    assert_eq!(mine.len(),theirs.len(),"compared rows differ in length");
    (mine.iter().zip(theirs).filter(|(a,b)|a.to_bits()==b.to_bits()).count(),mine.iter().zip(theirs).position(|(a,b)|a.to_bits()!=b.to_bits()))
}

/// The first-token prompt of the host form through the driver, token by token from a clean state. Compared with
/// the whole-model host form, word for word: the normed embedding, the stream after each visit, the capped logits
/// and the top token of every position. The gate: no word differs.
#[test]
#[ignore = "requires the real weight file and the reference prompt outside the repository"]
fn the_driver_equals_the_whole_model_host_form_on_the_real_weights() {
    let root=local("model");
    let prompt=std::fs::read_to_string(local("ref").join(PROMPT)).unwrap().split_whitespace().map(|id|id.parse::<u32>().unwrap()).collect::<Vec<_>>();
    let threads=std::thread::available_parallelism().map_or(1,|n|n.get());
    // The host form first. It is dropped before the weights of the driver are read, so one of the two is in memory.
    let start=Instant::now();
    let outs={
        let model=Model::open(&root,Limits {expert_cache_bytes:CACHE_GIB<<30}).unwrap();
        let runner=model.runner().unwrap();let mut chat=runner.chat().unwrap();
        runner.read(&mut chat,&prompt,Want {head:Head::Every,records:false},threads).unwrap()
    };
    println!("HOST form: {} token IDs, {} visits a token, logits of every position, {:.1} s on {threads} threads, expert cache limit {CACHE_GIB} GiB",
        prompt.len(),outs[0].streams.len(),start.elapsed().as_secs_f64());
    let start=Instant::now();
    let (description,weights)=HostWeights::open(&root,CACHE_GIB<<30).unwrap();let d=&description;
    println!("LOAD driver weights: dense weights {:.2} GiB as 32-bit words, {:.1} s; width {}, {} layers, {} visits a token, {} head rows; expert cache limit {CACHE_GIB} GiB",
        weights.counts().unwrap().dense_words as f64*4.0/(1u64<<30) as f64,start.elapsed().as_secs_f64(),d.hidden,d.layers.len(),d.visits.len(),d.head_rows);
    assert_eq!(outs[0].streams.len(),d.visits.len());
    let driver=Driver::new(&Host,&weights,d,prompt.len() as u64).unwrap();let mut state=driver.clean().unwrap();
    let (mut same,mut all,mut first,mut tops)=(0usize,0usize,None::<String>,0usize);
    let run=Instant::now();
    for (position,(&token,theirs)) in prompt.iter().zip(&outs).enumerate() {
        let begin=Instant::now();let mut note=|what:String|{if first.is_none() {println!("FIRST difference: {what}");first=Some(what);}};
        let embedding=driver.embedding(token).unwrap();let (e_same,e_at)=equal(&embedding,&theirs.embedding);
        if let Some(word)=e_at {note(format!("position {position} embedding word {word}: driver {:e}, host form {:e}",embedding[word],theirs.embedding[word]));}
        let mut streams=Vec::new();
        let step=driver.step_seen(&mut state,token,|place,stream|{assert_eq!(place,streams.len());streams.push(stream.clone());Ok(())}).unwrap();
        let (mut s_same,mut s_all)=(0,0);
        for (visit,(mine,host)) in streams.iter().zip(&theirs.streams).enumerate() {
            let (count,at)=equal(mine,host);s_same+=count;s_all+=mine.len();
            if let Some(word)=at {note(format!("position {position} visit {visit} layer {} word {word}: driver {:e}, host form {:e}",d.visits[visit].layer,mine[word],host[word]));}
        }
        let host_logits=theirs.logits.as_ref().unwrap();let (l_same,l_at)=equal(&step.logits,host_logits);
        if let Some(word)=l_at {note(format!("position {position} logit of row {word}: driver {:e}, host form {:e}",step.logits[word],host_logits[word]));}
        let host_top=theirs.top.unwrap();tops+=usize::from(step.top==host_top);
        if step.top!=host_top {note(format!("position {position} top token: driver {}, host form {host_top}",step.top));}
        same+=e_same+s_same+l_same;all+=embedding.len()+s_all+step.logits.len();
        println!("TOKEN {position}: id {token}; embedding {e_same}/{} equal; streams after the visits {s_same}/{s_all} equal; logits {l_same}/{} equal; top {} (host form {host_top}); {:.1} s",
            embedding.len(),step.logits.len(),step.top,begin.elapsed().as_secs_f64());
    }
    let counts=weights.counts().unwrap();
    println!("SUMMARY real-weight host token: {} positions, {same} of {all} words equal, {} differing; top tokens equal {tops} of {}; expert matrices decoded {}, found in the cache {}, held {} ({:.1} GiB); driver {:.1} s on one thread, {:.3} tokens a second",
        prompt.len(),all-same,prompt.len(),counts.decoded,counts.hits,counts.held,counts.held_bytes as f64/(1u64<<30) as f64,run.elapsed().as_secs_f64(),prompt.len() as f64/run.elapsed().as_secs_f64());
    assert!(first.is_none() && same==all && tops==prompt.len(),"the driver differs from the whole-model host form: {first:?}");
}

/// The folder of the trace of the whole-model host form on the first-token prompt, under the reference folder:
/// `prompt.ids`, `tokens.tsv` (position, tab, the top token of that position) and for each position the stream
/// after each visit and the capped logits as little-endian 32-bit words.
/// The folder name ends with the commit of the host form that wrote the trace (after ruling R8, second part).
const TRACE:&str="vade-host-trace/first-7b147ce721";
fn trace_words(path:&Path)->Vec<f32> {
    let bytes=std::fs::read(path).unwrap_or_else(|e|panic!("{}: {e}",path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect()
}
/// The prompt of the trace, the top token of each position, and the token that each position is fed: the prompt,
/// then the top token of the position before.
fn trace_tokens(trace:&Path)->(Vec<u32>,Vec<u32>,Vec<u32>) {
    let numbers=|name:&str|std::fs::read_to_string(trace.join(name)).unwrap().split_whitespace().map(|v|v.parse::<u32>().unwrap()).collect::<Vec<_>>();
    let prompt=numbers("prompt.ids");
    // `tokens.tsv` holds a position and a top token a line.
    let tops=numbers("tokens.tsv").as_chunks::<2>().0.iter().enumerate().map(|(position,pair)|{assert_eq!(pair[0] as usize,position);pair[1]}).collect::<Vec<_>>();
    assert!(!prompt.is_empty() && tops.len()>=prompt.len());
    let fed=(0..tops.len()).map(|position|if position<prompt.len() {prompt[position]} else {tops[position-1]}).collect::<Vec<_>>();
    (prompt,tops,fed)
}
/// Every position of the trace of the host form through the driver, token by token from a clean state: the prompt,
/// then the new tokens (position `p` is fed the top token of position `p - 1`). Compared word for word with the
/// trace files: the stream after each visit, the capped logits and the top token. The gate: no word differs.
/// A visit that differs prints its position, its place, its layer, its pass and its first differing word.
#[test]
#[ignore = "requires the real weight file and the host trace outside the repository"]
fn the_driver_equals_the_host_trace_on_every_position() {
    let (root,trace)=(local("model"),local("ref").join(TRACE));
    let (prompt,tops,fed)=trace_tokens(&trace);
    let start=Instant::now();
    let (description,weights)=HostWeights::open(&root,CACHE_GIB<<30).unwrap();let d=&description;let width=d.hidden as usize;
    println!("LOAD driver weights: dense weights {:.2} GiB as 32-bit words, {:.1} s; trace {TRACE}, {} positions ({} of the prompt), {} visits a token; expert cache limit {CACHE_GIB} GiB",
        weights.counts().unwrap().dense_words as f64*4.0/(1u64<<30) as f64,start.elapsed().as_secs_f64(),fed.len(),prompt.len(),d.visits.len());
    let driver=Driver::new(&Host,&weights,d,fed.len() as u64).unwrap();let mut state=driver.clean().unwrap();
    let (mut same,mut all,mut first,mut tops_equal,mut misses)=(0usize,0usize,None::<String>,0usize,0usize);
    let run=Instant::now();
    for (position,&token) in fed.iter().enumerate() {
        let begin=Instant::now();
        let host_streams=trace_words(&trace.join(format!("pos{position:04}.streams.f32")));
        let host_logits=trace_words(&trace.join(format!("pos{position:04}.logits.f32")));
        assert_eq!(host_streams.len(),d.visits.len()*width);
        let mut streams=Vec::new();
        let step=driver.step_seen(&mut state,token,|place,stream|{assert_eq!(place,streams.len());streams.push(stream.clone());Ok(())}).unwrap();
        let (mut s_same,mut s_all)=(0,0);
        for (place,mine) in streams.iter().enumerate() {
            let host=&host_streams[place*width..(place+1)*width];let (count,at)=equal(mine,host);s_same+=count;s_all+=width;
            if let Some(word)=at {
                let visit=&d.visits[place];misses+=1;
                let what=format!("position {position} id {token} visit {place} layer {} pass {}: {} of {width} words differ, first word {word}: driver {:08x}, host trace {:08x}",
                    visit.layer,visit.pass,width-count,mine[word].to_bits(),host[word].to_bits());
                if misses<=12 {println!("MISS {what}");}
                if first.is_none() {first=Some(what);}
            }
        }
        let (l_same,l_at)=equal(&step.logits,&host_logits);
        if let Some(word)=l_at {
            let what=format!("position {position} id {token} logit of row {word}: driver {:08x}, host trace {:08x}; {} of {} logits differ",step.logits[word].to_bits(),host_logits[word].to_bits(),host_logits.len()-l_same,host_logits.len());
            if first.is_none() {println!("MISS {what}");first=Some(what);}
        }
        tops_equal+=usize::from(step.top==tops[position]);
        same+=s_same+l_same;all+=s_all+host_logits.len();
        println!("TOKEN {position}: id {token}; streams after the visits {s_same}/{s_all} equal; logits {l_same}/{} equal; top {} (host trace {}); {:.1} s",
            host_logits.len(),step.top,tops[position],begin.elapsed().as_secs_f64());
    }
    let counts=weights.counts().unwrap();
    println!("SUMMARY host {} positions: {same} of {all} words equal, {} differing; top tokens equal {tops_equal} of {}; visits that differ {misses}; first difference: {}; expert matrices decoded {}, found in the cache {}; driver {:.1} s on one thread, {:.3} tokens a second",
        fed.len(),all-same,fed.len(),first.as_deref().unwrap_or("none"),counts.decoded,counts.hits,run.elapsed().as_secs_f64(),fed.len() as f64/run.elapsed().as_secs_f64());
    assert!(first.is_none() && same==all && tops_equal==fed.len(),"the driver differs from the host trace: {first:?}");
}

/// Gate G2 of the rows work: the trace through `Driver::step_rows`, for three cuts of the prompt into steps: the
/// whole prompt in one step, steps of 8 rows and steps of 4 rows. Each new token is a step of one row. Every row is
/// compared word for word with the trace files, as in the test above. The gate: no word differs for any cut.
#[test]
#[ignore = "requires the real weight file and the host trace outside the repository"]
fn the_rows_of_a_step_equal_the_host_trace_for_each_cut_of_the_prompt() {
    let (root,trace)=(local("model"),local("ref").join(TRACE));
    let (prompt,tops,fed)=trace_tokens(&trace);
    let (description,weights)=HostWeights::open(&root,CACHE_GIB<<30).unwrap();let d=&description;let (width,head)=(d.hidden as usize,d.head_rows as usize);
    let host=(0..fed.len()).map(|position|(trace_words(&trace.join(format!("pos{position:04}.streams.f32"))),trace_words(&trace.join(format!("pos{position:04}.logits.f32"))))).collect::<Vec<_>>();
    let mut failed=Vec::new();
    for cut in [prompt.len(),8,4] {
        let driver=Driver::with_limits(&Host,&weights,d,fed.len() as u64,rows::Limits {step_rows:prompt.len(),key_bytes:1<<30}).unwrap();
        let (mut state,mut at,mut same,mut all,mut tops_equal,mut first,mut steps)=(driver.clean().unwrap(),0usize,0usize,0usize,0usize,None::<String>,0usize);
        let run=Instant::now();
        while at<fed.len() {
            let count=if at<prompt.len() {cut.min(prompt.len()-at)} else {1};
            let mut streams=Vec::new();
            let step=driver.step_rows_seen(&mut [ChatStep {state:&mut state,tokens:&fed[at..at+count],wanted:Wanted::All,logits:true}],
                |place,stream|{assert_eq!(place,streams.len());streams.push(stream.clone());Ok(())}).unwrap();
            let logits=step.logits.unwrap();assert_eq!((streams.len(),step.tops.len(),logits.len()),(d.visits.len(),count,count*head));
            for row in 0..count {
                let position=at+row;let (host_streams,host_logits)=&host[position];
                for (place,stream) in streams.iter().enumerate() {
                    let (count,miss)=equal(&stream[row*width..(row+1)*width],&host_streams[place*width..(place+1)*width]);same+=count;all+=width;
                    if let (Some(word),None)=(miss,&first) {first=Some(format!("position {position} visit {place} layer {} pass {} word {word}: {} of {width} words differ",d.visits[place].layer,d.visits[place].pass,width-count));}
                }
                let (count,miss)=equal(&logits[row*head..(row+1)*head],host_logits);same+=count;all+=head;
                if let (Some(word),None)=(miss,&first) {first=Some(format!("position {position} logit of row {word}: {} of {head} logits differ",head-count));}
                tops_equal+=usize::from(step.tops[row]==tops[position]);
            }
            at+=count;steps+=1;
        }
        println!("ROWS prompt in steps of {cut} rows: {steps} steps for {} positions; {same} of {all} words equal, {} differing; top tokens equal {tops_equal} of {}; first difference: {}; {:.1} s on one thread on the host",
            fed.len(),all-same,fed.len(),first.as_deref().unwrap_or("none"),run.elapsed().as_secs_f64());
        if first.is_some() || same!=all || tops_equal!=fed.len() {failed.push((cut,first));}
    }
    let counts=weights.counts().unwrap();
    println!("SUMMARY rows on the host trace {TRACE}: cuts of the prompt {:?}; cuts that differ {}; expert matrices decoded {}, found in the cache {}",[prompt.len(),8,4],failed.len(),counts.decoded,counts.hits);
    assert!(failed.is_empty(),"a step of rows differs from the host trace: {failed:?}");
}

/// The memory of a state and of a step of rows on the real description, as `Driver::state_words` and
/// `Driver::transient_words` give them: arithmetic on the sizes of the model, no launch and no state.
#[test]
#[ignore = "requires the real weight file outside the repository"]
fn the_bytes_of_a_state_and_of_a_step_on_the_real_description() {
    let (description,weights)=HostWeights::open(&local("model"),CACHE_GIB<<30).unwrap();let d=&description;
    for capacity in [8192u64,32768] {
        let driver=match Driver::with_limits(&Host,&weights,d,capacity,rows::Limits {step_rows:1024,key_bytes:1<<30}) {Ok(driver)=>driver,Err(error)=>{println!("BYTES capacity {capacity}: no driver: {error}");continue;}};
        println!("BYTES capacity {capacity}, step row limit 1024, 1 GiB over keys: a clean state of one chat is {} bytes",driver.state_words().saturating_mul(4));
        for (rows,chats,ends,logits) in [(1,1,1,1),(8,8,8,8),(256,1,1,1),(1024,1,0,0),(1024,1,1,1),(1024,8,8,8),(1024,1,1024,0),(1024,1,1024,1024)] {
            println!("BYTES capacity {capacity}: a step of {rows} rows of {chats} chats, {ends} ends, {logits} with logits: at most {} bytes of values besides the states, {} bytes of device buffers ({} buffers)",
                driver.transient_words(rows,chats,ends,logits).saturating_mul(4),driver.device_words(rows,chats,ends,logits).saturating_mul(4),driver.transient_buffers(rows,chats));
        }
    }
    let one=Driver::new(&Host,&weights,d,8192).unwrap();
    println!("BYTES capacity 8192, one row a step: a clean state of one chat is {} bytes; a step of 1 row with its logits: at most {} bytes besides the state",one.state_words().saturating_mul(4),one.transient_words(1,1,1,1).saturating_mul(4));
}

/// One step of the 16 prompt rows of the trace on the counting backend (`meter::Meter`), at the shapes of the real
/// model: every block arm that the driver sends through the launch for rows runs in one launch of the rows entry and has no unit launch
/// (but the end mix of the one wanted row),
/// and the words that the step holds at one time are under `Driver::transient_words`; the device buffers that live
/// at one time (the meter keeps them as the device allocator does) are under `Driver::device_words`.
#[test]
#[ignore = "requires the real weight file and the host trace outside the repository"]
fn a_step_of_rows_on_the_real_weights_is_wrapped_and_under_the_bound() {
    let (root,trace)=(local("model"),local("ref").join(TRACE));let (prompt,tops,fed)=trace_tokens(&trace);
    let (description,weights)=HostWeights::open(&root,CACHE_GIB<<30).unwrap();let d=&description;
    let meter=Meter::device();
    let driver=Driver::with_limits(&meter,&weights,d,fed.len() as u64,rows::Limits {step_rows:prompt.len(),key_bytes:1<<30}).unwrap();
    let mut state=driver.clean().unwrap();let base=meter.live();assert_eq!(base,driver.state_words());
    meter.restart();meter.start_device_step();let pages=meter.pages();let start=Instant::now();
    let step=driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&prompt,wanted:Wanted::Last,logits:true}]).unwrap();
    let (rows,most,bound)=(prompt.len(),meter.most()-base,driver.transient_words(prompt.len(),1,1,1));
    let counts=ROW_ARMS.map(|op|[false,true].map(|rows|meter.launched(rows,op)));
    println!("ROWS ENTRY real weights: one step of {rows} rows of one chat, {:.1} s on the host: unit launches of the 9 arms {:?}, rows launches {:?} (order {ROW_ARMS:?})",
        start.elapsed().as_secs_f64(),counts.map(|count|count[0]),counts.map(|count|count[1]));
    println!("BYTES real weights: that step held at most {} bytes at one time besides the state; the bound of the driver is {} bytes, {:.2} times; top row of the last prompt row {} (trace {})",
        most*4,bound.saturating_mul(4),bound as f64/most as f64,step.tops[0],tops[rows-1]);
    assert_eq!(step.tops,[tops[rows-1]]);
    // The end mix runs for the one wanted row only (`Token::end_mix_rows`): one unit launch of MixBlend, no other.
    let unit=ROW_ARMS.iter().zip(&counts).map(|(op,count)|count[0]-usize::from(*op==spark_gpu_probe::causal_ops::Op::MixBlend)).collect::<Vec<_>>();
    assert!(unit.iter().all(|&n|n==0) && counts.iter().all(|count|count[1]>0),"an arm ran a launch for each row: {counts:?}");
    assert!(most<=bound,"the step held {most} words at one time, above the bound {bound}");
    let ((most_pages,live),(device,buffers))=(meter.most_pages(),(driver.device_words(rows,1,1,1),driver.transient_buffers(rows,1)));
    let (most_pages,live)=(most_pages-pages.0,live-pages.1);
    println!("DEVICE real weights: the live device buffers of that step held at most {} bytes in {live} buffers; the device bound is {} bytes in {buffers} buffers, {:.2} times",
        most_pages*4,device.saturating_mul(4),device as f64/most_pages as f64);
    assert!(most_pages<=device && live<=buffers,"the live device buffers {live} of {most_pages} words are above the bound {buffers} of {device}");
}
