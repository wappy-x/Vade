//! The draft model of the block family with a loop (`mtp.safetensors`) against the decode driver on one device.
//!
//! Mode `loop-mtp MODEL OUT NEW DEPTHS FOLDER...` (needs a GPU): the checkpoint of `MODEL` is loaded on one device,
//! with the vector pool on as the text service sets it. `DEPTHS` is a list of drafted tokens a check, as `1,3,7`.
//! Device-only depths use the served draft format. The format prefix `bf16:`, `mx:` or `nv:` selects a comparison.
//! The `-layers` and `-head` suffixes restrict packing to one part. Host comparison uses the BF16 reference by default.
//! Each `FOLDER` holds `INPUT_IDS`: the prompt, decimal token IDs with spaces between. For each prompt:
//! - `plain`: the prompt in steps of rows, then up to `NEW` greedy tokens, one token step each (`Driver::step`, the
//!   path of the text service for one chat), to the first end token of the checkpoint. The clock of each step.
//! - `taps`: the same chat in steps of rows that give the states the draft model reads (`Driver::step_rows_taps`).
//!   Its IDs must equal those of `plain`.
//! - `accept`: the host form of the draft model (`engine_model::mtp_draft`) at every answer position, fed the taps
//!   of the chat up to it and the next token of the chat: how many of its proposed tokens in a row equal the
//!   tokens of the chat, for 1 to all proposed tokens, and how many tokens a check the chat takes at each depth.
//! - `drafted`, one run for each depth: the chat, where each check step (`Driver::verify`) runs the newest token and
//!   the tokens that the host draft model proposes, and keeps the greedy ones (`Driver::accept`). Its IDs must equal
//!   those of `plain`. The clock of the draft and of the check, by check.
//!
//! Lines `LOOP-MTP`. Receipt: `RESULT.txt`, and `COMPLETE` when every run gave the IDs of `plain`. The draft model
//! runs on the host here: its time is not the time of a draft on the device, and the line says so.
//! Every size comes from the description of the checkpoint and from the config of the draft model.
use std::{fs,path::Path,time::{Duration,Instant}};
use engine_model::mtp_draft::{MtpChat,MtpModel};
use spark_tp::{Result,bounded_decoder::Compute,loop_decoder::{ChatStep,Driver,Wanted,mix::Weights,service::STEP_LIMITS,state::State}};

/// What a drafted run gave: the IDs, the checks, the kept tokens, the clock of each draft and of each check, the wall clock.
struct Drafted {out:Vec<u32>,checks:usize,kept:usize,draft:Vec<Duration>,check:Vec<Duration>,wall:Duration}
/// One drafted run of a chat whose prompt is read: each check runs the newest token and the tokens of `propose`,
/// keeps the greedy ones, and gives the taps of the kept rows to `push`. It stops at the length of `plain` or at an
/// end token.
#[allow(clippy::too_many_arguments)]
fn drafted<C:Compute,W:Weights<C>>(c:&C,driver:&Driver<'_,C,W>,state:&mut State<C::Value>,first:u32,plain:&[u32],taps:&[u64],width:usize,depth:usize,eos:&[u32],
    mut push:impl FnMut(&[Vec<Vec<f32>>])->Result<()>,mut propose:impl FnMut(u32,&[f32],usize)->Result<Vec<u32>>)->Result<Drafted> {
    let mut run=Drafted {out:Vec::new(),checks:0,kept:0,draft:Vec::new(),check:Vec::new(),wall:Duration::ZERO};let mut anchor=first;
    let began=Instant::now();
    while run.out.len()+1<plain.len() && !eos.contains(&anchor) {
        // The kept rows of a check end one token before the length of `plain`: the last top row is the next anchor.
        let room=plain.len()-run.out.len()-2;
        let begun=Instant::now();
        let embed=c.work(||c.read(&driver.embedding(anchor)?))?;
        let mut proposed=if room>0 {propose(anchor,&embed,depth.min(room))?} else {Vec::new()};proposed.truncate(depth.min(room));
        run.draft.push(begun.elapsed());
        let tokens=[&[anchor][..],&proposed].concat();
        let begun=Instant::now();
        let (tops,rows)=c.work(||{let (tops,t)=driver.verify(state,&tokens,taps)?;Ok((tops,rows_of(c,&t,width)?))})?;
        let good=proposed.iter().zip(&tops).take_while(|(a,b)|a==b).count();
        c.work(||driver.accept(state,tokens.len(),good+1))?;
        run.check.push(begun.elapsed());
        push(&rows[..good+1])?;
        run.out.push(anchor);run.out.extend_from_slice(&proposed[..good]);anchor=tops[good];run.checks+=1;run.kept+=good+1;
    }
    run.out.push(anchor);run.wall=began.elapsed();
    Ok(run)
}

fn ms(time:Duration)->f64 {time.as_secs_f64()*1.0e3}
fn mean(times:&[Duration])->f64 {if times.is_empty() {0.0} else {ms(times.iter().sum())/times.len() as f64}}
fn ids(path:&Path)->Result<Vec<u32>> {
    fs::read_to_string(path).map_err(|e|format!("{}: {e}",path.display()))?.split_whitespace().map(|word|word.parse::<u32>().map_err(|_|format!("{}: {word} is no token ID",path.display()))).collect()
}
/// The taps of every row of a step, read to the host: `[rows][taps][width]`.
fn rows_of<C:Compute>(c:&C,taps:&[C::Value],width:usize)->Result<Vec<Vec<Vec<f32>>>> {
    let words=taps.iter().map(|tap|c.read(tap)).collect::<Result<Vec<_>>>()?;
    let rows=words.first().map_or(0,|w|w.len()/width);
    Ok((0..rows).map(|row|words.iter().map(|w|w[row*width..(row+1)*width].to_vec()).collect()).collect())
}
/// One chat that reads its prompt in steps of rows with the taps of every row. Gives the state, the taps of the
/// prompt rows and the top row of the last prompt row.
fn prompt<C:Compute,W:Weights<C>>(c:&C,driver:&Driver<'_,C,W>,tokens:&[u32],taps:&[u64],width:usize)->Result<(State<C::Value>,Vec<Vec<Vec<f32>>>,u32)> {
    let mut state=driver.clean()?;let (mut rows,mut top)=(Vec::new(),None);
    for (number,piece) in tokens.chunks(driver.limits().step_rows).enumerate() {
        let last=(number+1)*driver.limits().step_rows>=tokens.len();
        let (step,tapped)=c.work(||{let (step,tapped)=driver.step_rows_taps(&mut [ChatStep {state:&mut state,tokens:piece,wanted:if last {Wanted::Last} else {Wanted::None},logits:false}],taps)?;Ok((step,rows_of(c,&tapped,width)?))})?;
        rows.extend(tapped);if last {top=step.tops.first().copied();}
    }
    Ok((state,rows,top.ok_or("loop mtp: the prompt gave no top row")?))
}
/// The draft context of a chat: one push of the taps of each committed token.
fn push(model:&MtpModel,chat:&mut MtpChat,rows:&[Vec<Vec<f32>>])->Result<()> {
    for row in rows {model.push_context(chat,&row.iter().map(Vec::as_slice).collect::<Vec<_>>()).map_err(|e|e.to_string())?;}
    Ok(())
}
#[cfg(target_os="linux")]
pub fn run(model_dir:&Path,out:&Path,new:&str,depths:&str,folders:&[String])->Result<()> {
    use engine_model::plan::loop_load::ExpertCodes::{Mxfp8,Nvfp4};
    let depths_input_has_prefix=depths.contains(':');
    let (mut layers,mut head,depths)=match depths.split_once(':') {
        Some(("bf16",depths))=>(None,None,depths),
        Some(("mx",depths))=>(Some(Mxfp8),Some(Mxfp8),depths),
        Some(("nv",depths))=>(Some(Nvfp4),Some(Nvfp4),depths),
        Some(("mx-head",depths))=>(None,Some(Mxfp8),depths),
        Some(("nv-head",depths))=>(None,Some(Nvfp4),depths),
        Some(("mx-layers",depths))=>(Some(Mxfp8),None,depths),
        Some(("nv-layers",depths))=>(Some(Nvfp4),None,depths),
        None=>(None,None,depths),_=>return Err("loop mtp: unknown packed draft format".into()),
    };
    // `rN/DEPTHS`: the repeat mode (N proposals at depths 3, 4 and 7, then a plain step and 4- and 8-row checks).
    let (repeat,depths)=match depths.split_once('/').and_then(|(r,rest)|r.strip_prefix('r').map(|n|(n,rest))) {
        Some((n,rest))=>(Some(n.parse::<usize>().ok().filter(|&n|n>0).ok_or("loop mtp: rN/ needs a count above zero")?),rest),
        None=>(None,depths),
    };
    // A depth list that starts with `d` runs the draft model on the device only (no host form runs).
    let (host_runs,depths)=match depths.strip_prefix('d') {Some(rest)=>(false,rest),None=>(true,depths)};
    if host_runs && (layers.is_some() || head.is_some()) {return Err("loop mtp: packed draft comparison needs device-only depths (d)".into());}
    if !host_runs && !depths_input_has_prefix {layers=Some(Nvfp4);head=Some(Nvfp4);}
    use engine_model::plan::{loop_load,loop_state::{LoopLedger,LoopStatePlan,context_limit}};
    use spark_tp::{bounded_decoder::Device,loop_decoder::operands::{device::{DeviceStore,PanelPlan},load,weights::LoadedWeights}};
    let new=new.parse::<usize>().ok().filter(|&n|n>0).ok_or("loop mtp: NEW must be a count above zero")?;
    let depths=depths.split(',').map(|word|word.parse::<usize>().ok().filter(|&n|n>0).ok_or_else(||format!("loop mtp: DEPTHS must be counts above zero, got {depths}"))).collect::<Result<Vec<_>>>()?;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let prompts=folders.iter().map(|folder|Ok((Path::new(folder).file_name().and_then(|n|n.to_str()).unwrap_or("prompt").to_owned(),ids(&Path::new(folder).join("INPUT_IDS"))?))).collect::<Result<Vec<_>>>()?;
    let checkpoint=loop_load::read_folder(model_dir).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    let d=&checkpoint.description;let width=d.hidden as usize;
    let taps=d.declared_draft.as_ref().ok_or("loop mtp: the checkpoint declares no draft model")?.taps.clone();
    let threads=std::thread::available_parallelism().map_or(4,|n|n.get());
    let started=Instant::now();
    let draft=MtpModel::open(model_dir,threads).map_err(|e|format!("loop mtp: the draft model: {e}"))?;
    let gamma=draft.config().gamma();
    if draft.config().target_layer_ids.iter().map(|&t|t as u64).collect::<Vec<_>>()!=taps {return Err("loop mtp: the taps of the draft config differ from the description".into());}
    if let Some(&most)=depths.iter().max() && most>gamma {return Err(format!("loop mtp: a depth above the {gamma} tokens that the draft model proposes"));}
    let mut lines=vec![format!("LOOP-MTP PLAN new={new} depths={depths:?} prompts={} taps={taps:?} gamma={gamma} draft_layers={} host_threads={threads} draft_load_seconds={:.1} draft_on=host",
        prompts.len(),draft.config().num_layers,started.elapsed().as_secs_f64())];
    eprintln!("{}",lines[0]);
    let capacity=(prompts.iter().map(|(_,p)|p.len()).max().unwrap_or(0)+new+gamma+2) as u64;
    let bucket=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
    let state=LoopStatePlan::for_steps(d,capacity,bucket,STEP_LIMITS.step_rows as u64).map_err(|e|e.to_string())?;
    let panels=PanelPlan::new(&checkpoint.load)?;
    let load_bytes=panels.charge(checkpoint.load.bytes)?;
    let ledger=LoopLedger::new(&load_bytes,&state).map_err(|e|e.to_string())?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let budget=free.saturating_sub((free/10).max(2<<30)) as u64;
    let limit=context_limit(d,&load_bytes,bucket,budget).map_err(|e|e.to_string())?;
    ledger.admit(budget,limit).map_err(|refusal|format!("text service memory refusal: {refusal}"))?;
    let workers=threads.min(16);
    let budget=usize::try_from(budget).map_err(|_|"budget exceeds the address space")?;
    let eos=d.eos_tokens.iter().filter_map(|&id|u32::try_from(id).ok()).collect::<Vec<_>>();
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,budget,|backend| {
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,model_dir,&mut store,workers)?;backend.synchronize()?;
        let files=load::Files::open(&checkpoint,model_dir)?;
        panels.upload(backend,&files,&loaded.operands)?;
        let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
        backend.configure_vector_pool(1<<30);
        lines.push(format!("LOOP-MTP LOAD tensors={} fingerprint={} device_bytes={} capacity={capacity} step_rows={} vector_pool=on",loaded.report.tensors,loaded.fingerprint(),backend.allocated_bytes(),STEP_LIMITS.step_rows));
        eprintln!("{}",lines[1]);
        let c=Device::new(backend);
        let driver=Driver::with_limits(&c,&weights,d,capacity,STEP_LIMITS)?;
        let began=Instant::now();
        let before_draft=backend.allocated_bytes();
        let raw=spark_tp::loop_decoder::draft::packed_weight_bytes(&draft,layers,head)?;
        store.set_panel_room(budget.saturating_sub(ledger.total as usize).saturating_sub(raw));
        let drafter=spark_tp::loop_decoder::draft::Drafter::with_packed(&draft,&mut store,&c,layers,head)?;backend.synchronize()?;
        eprintln!("LOOP-MTP DRAFT-ALLOC resident_bytes={}",backend.allocated_bytes().saturating_sub(before_draft));
        eprintln!("LOOP-MTP BF16 PANELS main_bytes={} draft_bytes={}", panels.bytes(),
            backend.allocated_bytes().saturating_sub(before_draft).saturating_sub(raw));
        lines.push(format!("LOOP-MTP DEVICE-DRAFT bytes={} upload_seconds={:.3}",drafter.bytes(),began.elapsed().as_secs_f64()));eprintln!("{}",lines.last().expect("a line"));
        let mut equal=true;
        let (mut all_runs,mut all_positions)=(vec![0usize;gamma+1],0usize);
        for (name,input) in &prompts {
            // Plain: the token path of the service for one chat.
            let (mut state,_,first)=prompt(&c,&driver,input,&[],width)?;
            let (mut plain,mut times)=(vec![first],Vec::new());
            while plain.len()<new && !eos.contains(plain.last().expect("one token")) {
                let token=*plain.last().expect("one token");let begun=Instant::now();
                let step=c.work(||driver.step(&mut state,token))?;times.push(begun.elapsed());plain.push(step.top);
            }
            drop(state);
            lines.push(format!("LOOP-MTP PLAIN prompt={name} prompt_tokens={} new_tokens={} ms_a_token={:.3} tokens_a_second={:.3}",input.len(),plain.len(),mean(&times),1.0e3/mean(&times).max(1.0e-9)));
            eprintln!("{}",lines.last().expect("a line"));
            // Price the actual verify and commit with the service's pool and draft taps.
            // Each timing starts after the same prompt and checks all output rows against plain.
            for n in [2usize,4,8].into_iter().filter(|&n|n<plain.len() && n<=gamma+1) {
                let mut trials=Vec::new();
                for _ in 0..3 {
                    let (mut state,_,_)=prompt(&c,&driver,input,&[],width)?;backend.synchronize()?;
                    let start=Instant::now();
                    let result=c.work(||{let (tops,_)=driver.verify(&mut state,&plain[..n],&taps)?;driver.accept(&mut state,n,n)?;Ok(tops)})?;
                    backend.synchronize()?;trials.push(start.elapsed());equal&=result==plain[1..=n];
                }
                trials.sort();let elapsed=ms(trials[1]);
                lines.push(format!("LOOP-MTP VERIFY-PRICE prompt={name} rows={n} median_ms={elapsed:.3} ratio={:.3}",elapsed/mean(&times)));
                eprintln!("{}",lines.last().expect("a line"));
            }
            // Taps: the same one-chat token path used by the service, with the states the draft model reads.
            // `step_rows_taps` is the multi-row/check path; using it here compared a different device arithmetic
            // path with `plain` and could report a false token-ID failure even though live drafted output was exact.
            let (mut state,mut rows,first)=prompt(&c,&driver,input,&taps,width)?;
            let mut tapped=vec![first];
            while tapped.len()<plain.len() {
                let token=*tapped.last().expect("one token");
                let (top,row)=c.work(||{let (step,t)=driver.step_taps(&mut state,token,&taps)?;Ok((step.top,rows_of(&c,&t,width)?))})?;
                rows.extend(row);tapped.push(top);
            }
            drop(state);
            let same=tapped==plain;equal&=same;
            let first_miss=tapped.iter().zip(&plain).position(|(a,b)|a!=b).map_or("none".to_owned(),|at|at.to_string());
            lines.push(format!("LOOP-MTP TAPS prompt={name} ids_equal_to_plain={same} first_difference={first_miss} rows={}",rows.len()));
            // Accept: the host draft model at each answer position, teacher-forced on the plain chat.
            let sequence=[&input[..],&plain[..]].concat();
            let (mut runs,mut chat,mut draft_times)=(vec![0usize;gamma+1],draft.chat(),Vec::new());
            let (mut device_runs,mut device_chat,mut device_times,mut agree,mut proposals)=(vec![0usize;gamma+1],drafter.chat(),Vec::new(),0usize,0usize);
            push(&draft,&mut chat,&rows[..input.len()-1])?;c.work(||drafter.push(&c,&mut device_chat,&rows[..input.len()-1]))?;
            for at in input.len()-1..sequence.len()-1 {
                push(&draft,&mut chat,&rows[at..at+1])?;c.work(||drafter.push(&c,&mut device_chat,&rows[at..at+1]))?;
                let anchor=sequence[at+1];let embed=c.work(||c.read(&driver.embedding(anchor)?))?;
                let begun=Instant::now();let device=c.work(||drafter.propose(&c,&weights,&device_chat,anchor,&embed,gamma,threads))?;device_times.push(begun.elapsed());
                let run=device.iter().zip(&sequence[at+2..]).take_while(|(a,b)|a==b).count();device_runs[run]+=1;
                if !host_runs {continue;}
                let begun=Instant::now();let proposed=draft.propose_block(&chat,anchor,&embed,threads).map_err(|e|e.to_string())?;draft_times.push(begun.elapsed());
                let run=proposed.iter().zip(&sequence[at+2..]).take_while(|(a,b)|a==b).count();runs[run]+=1;
                agree+=proposed.iter().zip(&device).filter(|(a,b)|a==b).count();proposals+=gamma;
            }
            lines.push(format!("LOOP-MTP DEVICE-ACCEPT prompt={name} positions={} runs_by_length={device_runs:?} device_draft_ms={:.3} proposals_equal_to_host_form={agree} of={proposals}",device_runs.iter().sum::<usize>(),mean(&device_times)));
            if host_runs {equal&=agree==proposals;}
            eprintln!("{}",lines.last().expect("a line"));
            // DEPTHS rN/...: N device proposals at each requested depth on the last position, each group
            // between idle gaps of 1.5 s, so a trace can cut out the kernels of the proposals alone. Then the run ends.
            if let Some(repeat)=repeat {
                let anchor=*sequence.last().expect("one token");let embed=c.work(||c.read(&driver.embedding(anchor)?))?;
                for &depth in &depths {
                    backend.synchronize()?;std::thread::sleep(std::time::Duration::from_millis(1500));
                    let mut each=Vec::with_capacity(repeat);let mut first:Option<Vec<u32>>=None;
                    for _ in 0..repeat {
                        let begun=Instant::now();let ids=c.work(||drafter.propose(&c,&weights,&device_chat,anchor,&embed,depth,threads))?;each.push(begun.elapsed());
                        equal&=first.get_or_insert_with(||ids.clone())==&ids;
                    }
                    backend.synchronize()?;each.sort();
                    lines.push(format!("LOOP-MTP DRAFT-REPEAT prompt={name} depth={depth} repeat={repeat} median_ms={:.3} mean_ms={:.3} same_ids={equal}",ms(each[repeat/2]),mean(&each)));
                    eprintln!("{}",lines.last().expect("a line"));
                }
                // Then the checks the drafted rounds pay: a plain step, and verify passes of 4 and 8 rows that keep one
                // row each (`accept(rows, 1)`), each group between the same idle gaps. Only the times are read.
                let (mut state,_,_)=prompt(&c,&driver,input,&taps,width)?;
                for n in [1usize,4,8] {
                    backend.synchronize()?;std::thread::sleep(std::time::Duration::from_millis(1500));
                    let mut each=Vec::with_capacity(repeat);let before=backend.vector_pool_stats();
                    for i in 0..repeat {
                        let at=i%(plain.len().saturating_sub(n).max(1));let begun=Instant::now();
                        if n==1 {c.work(||driver.step(&mut state,plain[at]).map(|_|()))?;}
                        else {c.work(||{driver.verify(&mut state,&plain[at..at+n],&taps)?;driver.accept(&mut state,n,1)})?;}
                        backend.synchronize()?;each.push(begun.elapsed());
                    }
                    each.sort();let after=backend.vector_pool_stats();
                    let per=|a:usize,b:usize|(a-b) as f64/repeat as f64;
                    lines.push(format!("LOOP-MTP CHECK-REPEAT prompt={name} rows={n} repeat={repeat} median_ms={:.3} mean_ms={:.3} pool_allocated_a_pass={:.1} pool_reused_a_pass={:.1} pool_returned_a_pass={:.1} pool_freed_a_pass={:.1} pool_evicted_a_pass={:.1}",ms(each[repeat/2]),mean(&each),
                        per(after.allocated_count,before.allocated_count),per(after.reused_count,before.reused_count),per(after.returned_count,before.returned_count),per(after.direct_freed_count,before.direct_freed_count),per(after.evicted_count,before.evicted_count)));
                    eprintln!("{}",lines.last().expect("a line"));
                }
                std::thread::sleep(std::time::Duration::from_millis(1500));
                fs::write(out.join("RESULT.txt"),lines.join("\n")+"\n").map_err(|e|e.to_string())?;
                return if equal {Ok(())} else {Err("loop mtp: repeated proposals differ".into())};
            }
            if !host_runs {runs=device_runs;}
            let positions=runs.iter().sum::<usize>();all_positions+=positions;
            for (all,one) in all_runs.iter_mut().zip(&runs) {*all+=one;}
            let mean_run=|k:usize|runs.iter().enumerate().map(|(run,n)|run.min(k)*n).sum::<usize>() as f64/positions.max(1) as f64;
            lines.push(format!("LOOP-MTP ACCEPT prompt={name} positions={positions} runs_by_length={runs:?} mean_run_k1..k{gamma}={} host_draft_ms={:.1}",
                (1..=gamma).map(|k|format!("{:.3}",mean_run(k))).collect::<Vec<_>>().join("/"),mean(&draft_times)));
            eprintln!("{}",lines.last().expect("a line"));
            // Drafted: each check step runs the newest token and `depth` drafted tokens, from the host form of the draft
            // model, then from the draft model on the device.
            for &depth in &depths {
                for on in ["host","device"] {
                    if on=="host" && !host_runs {continue;}
                    let (mut state,rows,first)=prompt(&c,&driver,input,&taps,width)?;
                    let run=if on=="host" {
                        let mut chat=draft.chat();push(&draft,&mut chat,&rows)?;
                        let chat=std::cell::RefCell::new(chat);
                        drafted(&c,&driver,&mut state,first,&plain,&taps,width,depth,&eos,|rows|push(&draft,&mut chat.borrow_mut(),rows),
                            |anchor,embed,_|draft.propose_block(&chat.borrow(),anchor,embed,threads).map_err(|e|e.to_string()))?
                    } else {
                        let mut chat=drafter.chat();c.work(||drafter.push(&c,&mut chat,&rows))?;
                        let chat=std::cell::RefCell::new(chat);
                        drafted(&c,&driver,&mut state,first,&plain,&taps,width,depth,&eos,|rows|c.work(||drafter.push(&c,&mut chat.borrow_mut(),rows)),
                            |anchor,embed,depth|c.work(||drafter.propose(&c,&weights,&chat.borrow(),anchor,embed,depth,threads)))?
                    };
                    drop(state);
                    let same=run.out==plain;equal&=same;
                    let first_miss=run.out.iter().zip(&plain).position(|(a,b)|a!=b).map_or("none".to_owned(),|at|at.to_string());
                    let wall=run.out.len() as f64/run.wall.as_secs_f64().max(1.0e-9);
                    lines.push(format!("LOOP-MTP DRAFTED prompt={name} draft_on={on} depth={depth} tokens={} checks={} tokens_a_check={:.3} check_ms={:.3} draft_ms={:.3} check_over_plain_step={:.3} round_ms={:.3} ids_equal_to_plain={same} first_difference={first_miss} wall_tokens_a_second={wall:.3} plain_tokens_a_second={:.3} speed_over_plain={:.3}",
                        run.out.len(),run.checks,run.kept as f64/run.checks.max(1) as f64,mean(&run.check),mean(&run.draft),mean(&run.check)/mean(&times).max(1.0e-9),mean(&run.check)+mean(&run.draft),
                        1.0e3/mean(&times).max(1.0e-9),wall*mean(&times)/1.0e3));
                    eprintln!("{}",lines.last().expect("a line"));
                }
            }
        }
        let mean_run=|k:usize|all_runs.iter().enumerate().map(|(run,n)|run.min(k)*n).sum::<usize>() as f64/all_positions.max(1) as f64;
        lines.push(format!("LOOP-MTP RESULT positions={all_positions} runs_by_length={all_runs:?} mean_run_k1..k{gamma}={} all_ids_equal={equal}",(1..=gamma).map(|k|format!("{:.3}",mean_run(k))).collect::<Vec<_>>().join("/")));
        eprintln!("{}",lines.last().expect("a line"));
        fs::write(out.join("RESULT.txt"),lines.join("\n")+"\n").map_err(|e|e.to_string())?;
        if equal {fs::write(out.join("COMPLETE"),"").map_err(|e|e.to_string())?;Ok(())} else {Err("loop mtp: a run gave other IDs than the plain chat".into())}
    })
}
