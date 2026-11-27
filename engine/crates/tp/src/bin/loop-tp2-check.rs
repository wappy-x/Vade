//! Decode check of the looped text family on a cabled pair of Sparks: one chat, greedy, with the large decode
//! products cut between the two Sparks (`spark_tp::loop_decoder::tp2`), against the same chat on one Spark.
//!
//! `loop-tp2-check HALF LISTEN NEXT FORM MODEL PROMPT NEW`
//! - `FORM` `pair`: one Spark of the pair, trading on the mapped mailbox of the device (`tp2::MailboxPair`; its wire
//!   link takes the ports of LISTEN and NEXT plus 2). `ring`: the same pair trading on the host ring (`tp2::RingPair`).
//!   Start both Sparks with the same FORM, MODEL, PROMPT and NEW: HALF 0 on one, HALF 1 on the other. LISTEN is this
//!   Spark's address on the cable, NEXT the other's. The four places before MODEL are those that differ between the
//!   two starts (the ring's run hash skips them, as for `flash-serve`).
//! - `FORM` `alone`: one Spark, the weights of one Spark: the reference. HALF, LISTEN and NEXT are not read.
//! - `PROMPT`: a text file. Its words are the prompt, read by the tokenizer of the model folder with no template.
//! - `NEW`: the tokens to decode after the prompt, the first one included.
//!
//! Both forms run the same steps with the driver of the service: the prompt in pieces of rows (whole on each Spark),
//! then one token step a token (`Driver::step`, the token pass the service runs for one chat). Lines on standard
//! error: `LOOP TP2 PROMPT`, `LOOP TP2 DECODE`, `LOOP TP2 TRADED` (the pair only) and `LOOP TP2 IDS`, then
//! `LOOP TP2 PASS`. A refusal is one `FAIL` line and exit code 1.
#![forbid(unsafe_code)]
use spark_tp::{Result,startup};
use std::{net::SocketAddr,path::PathBuf,time::Instant};

const USAGE:&str="loop-tp2-check HALF LISTEN NEXT (pair | ring | alone) MODEL PROMPT NEW";

#[cfg_attr(not(target_os="linux"),allow(dead_code))] // Only the device half, which is Linux code, reads every field.
struct Settings {model:PathBuf,prompt:PathBuf,new:usize,pair:Option<(usize,SocketAddr,SocketAddr)>,mailbox:bool}
fn settings(args:&[String])->Result<Settings> {
    let [half,listen,next,form,model,prompt,new]=args else {return Err(format!("usage: {USAGE}"))};
    let new=new.parse::<usize>().ok().filter(|&n|n>0).ok_or_else(||format!("NEW must be a positive whole number, got {new}"))?;
    let pair=match form.as_str() {
        "alone"=>None,
        "pair"|"ring"=>{
            let half=half.parse::<usize>().ok().filter(|&h|h<=1).ok_or_else(||format!("HALF must be 0 or 1, got {half}"))?;
            let address=|name:&str,text:&str|text.parse::<SocketAddr>().map_err(|e|format!("{name} must be an address with a port, got {text}: {e}"));
            Some((half,address("LISTEN",listen)?,address("NEXT",next)?))
        },
        _=>return Err(format!("usage: {USAGE}")),
    };
    Ok(Settings {model:model.into(),prompt:prompt.into(),new,pair,mailbox:form=="pair"})
}
#[cfg(target_os="linux")]
mod device {
    use super::Settings;
    use engine_model::plan::loop_load::{self,ExpertCodes};
    use spark_gpu_probe::{packed_product,staged_product};
    use spark_serve::Tokenizer;
    use spark_tp::{Result,bounded_decoder::{Compute,Device},startup,
        loop_decoder::{ChatStep,Driver,Wanted,mix::Weights,operands::{device::{DeviceStore,PanelPlan},load,weights::LoadedWeights},
            service::{STEP_LIMITS,admit,start},tp2::{MailboxPair,Pair,RingPair,Split,part}},tensor_wire};
    use std::{sync::atomic::{AtomicBool,Ordering},time::{Duration,Instant}};
    /// How long a Spark waits for the other to open the wire link of the mailbox (both start at the same moment).
    const LINK_WAIT:Duration=Duration::from_secs(60);
    /// The rule of the grid of a launch whose extent follows the context: a straight plane holds whole buckets.
    const BUCKET:u64=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
    /// The memory margin of the vector pool, as the service keeps it.
    const VECTOR_POOL_MARGIN:usize=1<<30;
    /// What one run gives: the token ids, the prompt seconds, and the seconds of each decode step after the first.
    struct Ran {ids:Vec<u32>,prompt_seconds:f64,steps:Vec<f64>}
    fn chat<'b,'d,W:Weights<Device<'b,'d>>>(compute:&Device<'b,'d>,weights:&W,d:&engine_model::plan::description::LoopTextDescription,context:u64,limits:spark_tp::loop_decoder::rows::Limits,prompt:&[u32],new:usize)->Result<Ran> {
        let driver=Driver::with_limits(compute,weights,d,context,limits)?;
        compute.step_spares(driver.device_words(limits.step_rows,1,1,0).saturating_mul(4).saturating_add(driver.state_words()*4));
        let mut state=driver.clean()?;
        let began=Instant::now();
        let (mut at,mut top)=(0,None);
        while at<prompt.len() {
            let end=(at+limits.step_rows).min(prompt.len());
            driver.ensure_capacity(&mut state,end)?;
            let wanted=if end==prompt.len() {Wanted::Last} else {Wanted::None};
            let step=compute.work(||driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&prompt[at..end],wanted,logits:false}]))?;
            top=step.tops.last().copied().or(top);
            at=end;
        }
        let first=top.ok_or("the prompt gave no token")?;
        driver.detach(&mut state)?;
        let prompt_seconds=began.elapsed().as_secs_f64();
        let (mut ids,mut steps)=(vec![first],Vec::with_capacity(new));
        while ids.len()<new {
            let began=Instant::now();
            let token=ids[ids.len()-1];
            let next=state.position as usize+1;driver.ensure_capacity(&mut state,next)?;
            let step=compute.work(||driver.step(&mut state,token))?;
            steps.push(began.elapsed().as_secs_f64());
            ids.push(step.top);
        }
        Ok(Ran {ids,prompt_seconds,steps})
    }
    /// The chat on the pair: the split weights over `pair`, and the cost of its trades.
    #[allow(clippy::too_many_arguments)]
    fn split_chat<'b,'d,P:Pair<Device<'b,'d>>>(compute:&Device<'b,'d>,weights:&LoadedWeights<'_,spark_gpu_probe::driver::DeviceVector<'d>>,pair:&P,d:&engine_model::plan::description::LoopTextDescription,context:u64,limits:spark_tp::loop_decoder::rows::Limits,prompt:&[u32],new:usize)->Result<Ran> {
        pair.ready()?;
        let split=Split::new(weights,pair);
        let ran=chat(compute,&split,d,context,limits,prompt,new)?;
        let traded=split.traded();
        let steps=ran.steps.len() as f64+1.0;
        eprintln!("LOOP TP2 TRADED swaps={} words={} host_seconds={:.4} swaps_per_token={:.1} host_us_per_swap={:.2}",
            traded.swaps,traded.words,traded.ns as f64/1e9,traded.swaps as f64/steps,traded.ns as f64/1e3/traded.swaps.max(1) as f64);
        Ok(ran)
    }
    fn report(ran:&Ran,prompt:usize) {
        eprintln!("LOOP TP2 PROMPT tokens={prompt} seconds={:.4} tokens_per_second={:.1}",ran.prompt_seconds,prompt as f64/ran.prompt_seconds);
        let mut sorted=ran.steps.clone();sorted.sort_by(f64::total_cmp);
        let (sum,median)=(sorted.iter().sum::<f64>(),sorted.get(sorted.len()/2).copied().unwrap_or(0.0));
        let mean=sum/sorted.len().max(1) as f64;
        eprintln!("LOOP TP2 DECODE steps={} mean_ms={:.3} median_ms={:.3} tokens_per_second={:.2}",sorted.len(),mean*1e3,median*1e3,if mean>0.0 {1.0/mean} else {0.0});
        eprintln!("LOOP TP2 IDS {}",ran.ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
    }
    pub fn run(s:&Settings,started:Instant)->Result<()> {
        let root=&s.model;
        let mut checkpoint=loop_load::read_folder(root).map_err(|e|format!("This engine cannot load this model: {e}."))?;
        // The load of the service: the packed experts where both packed projections take the shape.
        checkpoint.load.packed_experts_only(|packed| {
            let (Some(unit),Some([experts,..]))=(packed.unit(),packed.stack()) else {return false;};
            (1..=packed_product::PICKED_EXPERTS_MAX as u64).contains(&experts)
                && packed_product::shape_admitted(unit.rows,unit.columns,unit.codes==ExpertCodes::Mxfp8,unit.order.is_some())
                && staged_product::rows_shape_admitted(unit.rows,unit.columns,1)
        }).map_err(|e|format!("This engine cannot load this model: {e}."))?;
        let d=&checkpoint.description;
        startup::step("tokenizer");
        let text=start::text(root,d)?;
        let words=std::fs::read_to_string(&s.prompt).map_err(|e|format!("cannot read the prompt {}: {e}",s.prompt.display()))?;
        let prompt=text.tokenizer.encode(&words)?;
        if prompt.is_empty() {return Err("the prompt has no token".into());}
        let context=((prompt.len()+s.new) as u64).next_multiple_of(BUCKET);
        start::need_kernel_image(spark_gpu_probe::inkling_ptx().is_some())?;
        let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
        let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
        let budget=free.saturating_sub((free/10).max(2<<30));
        let panels=PanelPlan::new(&checkpoint.load)?;
        let ptx=spark_gpu_probe::layer_ptx()?;
        let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
        // The pair connects before the load and waits for the other Spark after it (`Pair::ready`). The mailbox form
        // also opens the wire link of the mailbox now: half 1 listens and half 0 dials, on the ports plus 2.
        let pair=s.pair.map(|(half,listen,next)|->Result<_> {
            startup::step("pair");
            let ring=RingPair::new(spark_tp::transport::Ring::connect(half,2,listen,next)?)?;
            let link=if s.mailbox {
                let shift=|a:std::net::SocketAddr|std::net::SocketAddr::new(a.ip(),a.port()+2);
                let (out,inc)=spark_seqpar::control::connect_peers((half>0).then_some(shift(listen)),(half==0).then_some(shift(next)),Instant::now()+LINK_WAIT)?;
                Some(inc.or(out).ok_or("the mailbox wire has no link to the other Spark")?)
            } else {None};
            Ok((ring,link,half))
        }).transpose()?;
        startup::step("gpu_context_and_kernel_image");
        spark_gpu_probe::driver::with_backend(0,ptx,budget,|backend| {
            backend.inkling_preload()?;
            let panel_load=panels.charge(checkpoint.load.bytes)?;
            let admitted=admit(d,&panel_load,context as usize,budget as u64,BUCKET,STEP_LIMITS)?;
            startup::step("weights");
            let mut store=DeviceStore::new(backend);
            let loaded=load::load(&checkpoint,root,&mut store,workers)?;
            let files=load::Files::open(&checkpoint,root)?;
            panels.upload(backend,&files,&loaded.operands)?;
            backend.synchronize()?;
            let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
            let compute=Device::new(backend);
            backend.configure_vector_pool(VECTOR_POOL_MARGIN);
            backend.configure_kv_reserve(VECTOR_POOL_MARGIN);
            compute.fund_kv_pages(admitted.room()?);
            eprintln!("LOOP TP2 LOAD fingerprint={} seconds={:.1} context={context} prompt_tokens={}",loaded.fingerprint(),started.elapsed().as_secs_f64(),prompt.len());
            let ran=match pair {
                None=>chat(&compute,&weights,d,context,admitted.limits,&prompt,s.new)?,
                Some((ring,None,_))=>split_chat(&compute,&weights,&ring,d,context,admitted.limits,&prompt,s.new)?,
                Some((ring,Some(link),half))=>{
                    // The mailbox holds the longest cut part of one token row: the head's (`tp2::part`).
                    let head=d.head_rows as usize;
                    // It holds the summed experts of each Spark's rows of a prompt step too (`Split::row_cut`).
                    let slot=part(head,true,0).len().max(part(head,true,1).len()).max(d.hidden as usize*admitted.limits.step_rows.div_ceil(2));
                    let mailbox=backend.tp_mailbox(slot*4)?;
                    let wire=tensor_wire::Wire::Write(tensor_wire::WriteLink::new(&link)?);
                    let worker=wire.bind(mailbox.take_host().ok_or("the mailbox host is taken")?,half==0,mailbox.layout())?;
                    let (stop,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
                    std::thread::scope(|scope|->Result<_> {
                        let thread=tensor_wire::spawn(scope,worker,&stop,&attached,&failed)?;
                        let ran=tensor_wire::wait_attached(&attached,&thread).and_then(|()|{
                            let pair=MailboxPair::new(&mailbox,ring);
                            split_chat(&compute,&weights,&pair,d,context,admitted.limits,&prompt,s.new)
                        });
                        stop.store(true,Ordering::Relaxed);
                        let wire=thread.join().map_err(|_|"the mailbox wire thread stopped".to_owned()).and_then(|ended|ended);
                        let ran=ran?;wire?;
                        if mailbox.failed() {return Err("the mailbox failed".into());}
                        Ok(ran)
                    })?
                },
            };
            report(&ran,prompt.len());
            backend.synchronize()?;
            Ok(())
        })
    }
}
fn run(args:&[String],started:Instant)->Result<()> {
    let s=settings(args)?;
    startup::begin(s.pair.map_or(0,|(half,..)|half),started,"settings");
    #[cfg(target_os="linux")]
    {device::run(&s,started)?;eprintln!("LOOP TP2 PASS");Ok(())}
    #[cfg(not(target_os="linux"))]
    {let _=s;Err("loop-tp2-check runs on a Spark (Linux and its GPU)".into())}
}
fn main() {
    let started=Instant::now();
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error)=run(&args,started) {
        startup::failed(&error);
        eprintln!("FAIL {error}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn args(text:&str)->Vec<String> {text.split_whitespace().map(str::to_owned).collect()}
    #[test]
    fn the_two_forms_read_their_places() {
        let alone=settings(&args("- - - alone m p 64")).unwrap();
        assert!(alone.pair.is_none() && alone.new==64);
        let pair=settings(&args("1 10.0.0.2:4000 10.0.0.1:4000 pair m p 8")).unwrap();
        assert_eq!(pair.pair.map(|(h,..)|h),Some(1));
        for bad in ["- - - alone m p 0","2 10.0.0.2:4000 10.0.0.1:4000 pair m p 8","0 x y pair m p 8","- - - both m p 8","m p 8"] {assert!(settings(&args(bad)).is_err(),"{bad}");}
    }
}
