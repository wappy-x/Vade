//! Batch step check: several chats decode in one step, and each must get the IDs and the raw
//! logits of its own plain run. One load: each job's prompt is read once; each chat decodes alone
//! (the reference); then the same chats decode again from the same prompt states in batch steps,
//! with every partner, in both orders, alone in a batch of one, and as copies of one chat.
use spark_tp::{Result,bounded_decoder::{Compute,Decoder,HeadRows,State,PROMPT_GROUP,BATCH_ROWS,expert_union_counts}};
fn median(ms:&[f64])->f64 {let mut sorted=ms.to_vec();sorted.sort_by(f64::total_cmp);if sorted.is_empty() {0.0} else {sorted[sorted.len()/2]}}
fn mean(ms:&[f64])->f64 {if ms.is_empty() {0.0} else {ms.iter().sum::<f64>()/ms.len() as f64}}
/// The first maximum, as the decode loop of this harness picks it.
fn first_best(logits:&[f32])->u32 {let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}best as u32}
/// The batches one check runs, as lists of chat numbers: each chat alone, every pair in both
/// orders, every set that leaves one chat out, all chats in both orders, and 2, 4 and 8 copies of the first chat and of the last
/// chat (every row then picks the same experts: the least a batch can read).
fn batches(chats:usize)->Vec<Vec<usize>> {
    let mut runs=(0..chats).map(|chat|vec![chat]).collect::<Vec<_>>();
    for a in 0..chats {for b in a+1..chats {runs.push(vec![a,b]);runs.push(vec![b,a]);}}
    if chats>3 {for out in 0..chats {runs.push((0..chats).filter(|&chat|chat!=out).collect());}}
    if chats>2 {runs.push((0..chats).collect());runs.push((0..chats).rev().collect());}
    for chat in [0,chats-1] {for copies in [2,4,8] {if copies<=BATCH_ROWS && (chat==0 || chats>1) {runs.push(vec![chat;copies]);}}}
    runs
}
/// The short list: the first chat, the first two, ... all the chats (so 1 to 8 chats that differ, one run each).
fn prefix_sets(chats:usize)->Vec<Vec<usize>> {(1..=chats).map(|count|(0..count).collect()).collect()}
/// Host work of a run for one step, from the backend's counters before and after it: forced waits
/// for the device, queued launches, recorded-launch replays, device reads, and the ms in waits
/// and in launch calls. Structure, not speed: it tells a step that keeps the device fed from one
/// that stops for the host.
fn host_work(before:[u64;6],after:[u64;6],steps:usize)->String {
    let each=|i:usize|(after[i]-before[i]) as f64/steps as f64;
    format!("waits_a_step={:.1} launches_a_step={:.1} replays_a_step={:.1} reads_a_step={:.1} wait_ms_a_step={:.3} launch_ms_a_step={:.3}",each(0),each(1),each(2),each(3),each(4)/1e3,each(5)/1e3)
}
/// What a run launched for one step, by counter name (recorded kernel kinds, row kernels, fills,
/// copies): every counter that moved, as name=count a step.
fn census(before:&[(String,u64)],after:&[(String,u64)],steps:usize)->String {
    after.iter().filter_map(|(name,now)| {
        let was=before.iter().find(|(other,_)|other==name).map_or(0,|(_,value)|*value);
        (*now>was).then(||format!("{name}={:.1}",(*now-was) as f64/steps as f64))
    }).collect::<Vec<_>>().join(" ")
}
/// `stepwise`: each step opens and closes the resident queue, as a served step does. `prompt_queue`
/// is then the load's queue: it serves the prompts and closes before the first step.
/// `counts`: the backend's counters now (see `host_work`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // the step index reads every chat's plain run.
pub fn run<C:Compute,D:Compute,Q>(rank:usize,decoder:&Decoder<'_,C>,compute:&D,prompts:&[&[u32]],steps:usize,stepwise:bool,prompt_queue:Option<Q>,
    counts:&dyn Fn()->[u64;6],launched:&dyn Fn()->Vec<(String,u64)>,prefix:bool)->Result<String> {
    if prompts.is_empty() || prompts.len()>BATCH_ROWS || !(2..=256).contains(&steps) {return Err("batch check needs 1 to 8 prompts and 2 to 256 steps".into());}
    let timed=|run:&mut dyn FnMut()->Result<()>|->Result<f64> {
        let started=std::time::Instant::now();if stepwise {compute.work(&mut *run)?;} else {run()?;}Ok(started.elapsed().as_secs_f64()*1e3)};
    // Each chat's state after its prompt, and its first decode token.
    let mut starts:Vec<(State<C::Value>,u32)>=Vec::new();
    for prompt in prompts {
        let mut state=decoder.zero_state()?;let mut last=None;
        for group in prompt.chunks(PROMPT_GROUP) {let (mut rows,next)=decoder.prefill_rows(group,&state,HeadRows::Last)?;state=next;last=rows.pop();}
        starts.push((state,first_best(&last.ok_or("batch check prompt is empty")?)));
    }
    drop(prompt_queue);let mut report=String::new();compute.cost_report("batch-check-prompts");
    // Reference: each chat alone, with the plain step.
    let mut alone:Vec<Vec<(Vec<u32>,u32)>>=Vec::new();let mut alone_ms=Vec::new();
    for (chat,(start,first)) in starts.iter().enumerate() {
        let mut rows=Vec::with_capacity(steps);let mut ms=Vec::with_capacity(steps);let mut token=*first;let mut state:Option<State<C::Value>>=None;let before=counts();let kinds=launched();
        for _ in 0..steps {
            let mut made=None;
            ms.push(timed(&mut ||{made=Some(decoder.step(token,state.as_ref().unwrap_or(start),|_,_|Ok(()))?);Ok(())})?);
            let (logits,_,next)=made.ok_or("plain step made no result")?;
            token=first_best(&logits);rows.push((logits.iter().map(|value|value.to_bits()).collect(),token));state=Some(next);
        }
        let ids=rows.iter().map(|row|row.1).collect::<Vec<_>>();
        let line=format!("RESIDENT BATCH rank={rank} run=alone chat={chat} prompt_tokens={} steps={steps} median_ms={:.3} mean_ms={:.3} {} ids={ids:?}\n",
            prompts[chat].len(),median(&ms[2..]),mean(&ms[2..]),host_work(before,counts(),steps));
        eprint!("{line}");report.push_str(&line);alone.push(rows);alone_ms.push(median(&ms[2..]));
        let line=format!("RESIDENT BATCH CENSUS rank={rank} run=alone chat={chat} {}\n",census(&kinds,&launched(),steps));eprint!("{line}");report.push_str(&line);
        compute.cost_report(&format!("alone-chat-{chat}"));
    }
    for order in if prefix {prefix_sets(prompts.len())} else {batches(prompts.len())} {
        let before=expert_union_counts();let mut ms=Vec::with_capacity(steps);let work=counts();let kinds=launched();
        let mut tokens=order.iter().map(|&chat|starts[chat].1).collect::<Vec<_>>();let mut states:Option<Vec<State<C::Value>>>=None;
        for step in 0..steps {
            let mut made=None;
            ms.push(timed(&mut ||{
                let rows=order.iter().enumerate().map(|(row,&chat)|(tokens[row],states.as_ref().map_or(&starts[chat].0,|held|&held[row]))).collect::<Vec<_>>();
                made=Some(decoder.step_batch(&rows)?);Ok(())})?);
            let results=made.ok_or("batch step made no result")?;
            if results.len()!=order.len() {return Err("batch step row count differs".into());}
            let mut next=Vec::with_capacity(order.len());
            for (row,(&chat,(logits,best,state))) in order.iter().zip(results).enumerate() {
                let (wanted,id)=&alone[chat][step];
                if logits.len()!=wanted.len() || logits.iter().zip(wanted).any(|(got,want)|got.to_bits()!=*want) {
                    let words=logits.iter().zip(wanted).filter(|(got,want)|got.to_bits()!=**want).count();
                    return Err(format!("batch step differs from the plain run: batch={order:?} row={row} chat={chat} step={step} differing_logit_words={words} id={best} plain_id={id}"));
                }
                if first_best(&logits)!=*id || best!=*id {return Err(format!("batch step ID differs: batch={order:?} row={row} chat={chat} step={step}"));}
                tokens[row] = *id;next.push(state);
            }
            states=Some(next);
        }
        let batch=median(&ms[2..]);let one=order.iter().map(|&chat|alone_ms[chat]).sum::<f64>()/order.len() as f64;
        let (against,speed,host)=(batch/one,order.len() as f64*one/batch,host_work(work,counts(),steps));
        let line=format!("RESIDENT BATCH rank={rank} run=batch chats={order:?} rows={} steps={steps} median_ms={batch:.3} mean_ms={:.3} one_row_ms={one:.3} \
            cost_against_one_row={against:.3} total_speed={speed:.3} {host} logits_equal=1 ids_equal=1\n",order.len(),mean(&ms[2..]));
        eprint!("{line}");report.push_str(&line);
        let line=format!("RESIDENT BATCH CENSUS rank={rank} run=batch chats={order:?} {}\n",census(&kinds,&launched(),steps));eprint!("{line}");report.push_str(&line);
        compute.cost_report(&format!("batch-{}",order.iter().map(|chat|chat.to_string()).collect::<Vec<_>>().join("-")));
        // Experts read against experts picked, by layer, over this run.
        let after=expert_union_counts();let (mut picks,mut distinct,mut layers)=(0u64,0u64,String::new());
        for (layer,(now,was)) in after.iter().zip(&before).enumerate() {
            let (p,d)=(now[0]-was[0],now[1]-was[1]);if p==0 {continue;}
            picks+=p;distinct+=d;layers.push_str(&format!(" {layer}:{:.2}",d as f64*order.len() as f64/p as f64));
        }
        if picks>0 {
            let union=distinct as f64*order.len() as f64/picks as f64;
            let line=format!("RESIDENT BATCH UNION rank={rank} chats={order:?} rows={} picks={picks} distinct={distinct} union_in_rows={union:.3} by_layer={}\n",order.len(),layers.trim_start());
            eprint!("{line}");report.push_str(&line);
        }
    }
    Ok(report)
}
