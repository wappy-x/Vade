//! One box, small checkpoint, the real served stack with the predictor layer loaded (Factory, Split, Device,
//! resident queue): sessions that draft give the tokens of the plain session, and each part of a drafted cycle is timed.
//!
//! The checkpoint has the model's layer count and kinds (34 recurrent layers, 11 latent ones), the model's vocabulary
//! size and one predictor layer. Its matrices are small, so kernel times mean nothing here. What the host does around
//! the kernels in a cycle (commands, reads of the vocabulary, picks, kept views, verdicts) has the model's counts and
//! the model's vocabulary. Every time printed is host wall time on this box. Not a served grade.
//!
//! Cases, each a new session on the same prompt. PLAIN: the reference, by its logits. GREEDY: plain decode as the
//! scheduler runs it. CHOOSER d: strict, depth d, the server's speculator decides when to propose. HAND d: strict,
//! depth d, a proposal in every cycle; every second proposal is replaced by the plain continuation, so passes that
//! keep every row and passes that drop rows both run. Every kept token must be the plain token.
use super::*;
use super::first_step::OneRank;
use super::select_known::attention;
use spark_engine::draft::{Config,Speculator};
use spark_gpu_probe::driver::Backend;
use spark_tp::{bounded_decoder::Device,row_split::{Factory,Split}};
use spark_serve::{BatchOutput,BatchRows,Engine,Predictions,Session};
use engine_model::{plan::description::{TokenMixer,TextDescription},safetensors::Dtype};

/// The model's vocabulary size: a pass reads this many logits a row, and a pick scans them.
const VOCAB:u64=154_880;
const LAYERS:usize=45;
fn checkpoint(capacity:u64)->attention::Checkpoint {
    attention::Checkpoint::build_with(Dtype::Bf16,true,capacity,true,|d:&mut TextDescription| {
        let (recurrent,mut latent)=(d.layers[0],d.layers[1]);
        if let TokenMixer::LatentAttention(a)=&mut latent.mixer {a.indexer.token_budget=2048;a.indexer.pool=4;}
        d.layers=(0..LAYERS).map(|layer|if layer%4==3 {latent} else {recurrent}).collect();d.vocab=VOCAB;
    })
}
fn best(row:&[f32])->u32 {let mut best=0;for id in 1..row.len() {if row[id]>row[best] {best=id;}}best as u32}
fn ms(started:std::time::Instant)->f64 {started.elapsed().as_secs_f64()*1000.0}
/// Mean and median of a part's samples, or dashes when it never ran.
fn stat(samples:&[f64])->String {
    if samples.is_empty() {return "n=0 mean=- median=-".into();}
    let mut sorted=samples.to_vec();sorted.sort_by(f64::total_cmp);
    format!("n={} mean={:.3} median={:.3}",samples.len(),samples.iter().sum::<f64>()/samples.len() as f64,sorted[sorted.len()/2])
}
/// One pass as the scheduler sends it: a greedy request's rows.
fn pass<'s,E:Engine>(engine:&'s E,session:&mut E::Session<'s>,tokens:&[u32])->Result<Predictions> {
    let mut rows=[BatchRows {session,tokens,output:BatchOutput::All,greedy:true,top_k:0}];
    engine.step_batch(&mut rows).map_err(|error|error.message)?.pop().ok_or_else(||"pass has no result".into())
}
/// What one case measured, by part, in ms. `rows[n]`: passes of n rows.
#[derive(Default)]
struct Parts {proposal:Vec<f64>,waiting:Vec<f64>,rows:[Vec<f64>;5],full:Vec<f64>,partial:Vec<f64>,cycle:[Vec<f64>;5],kept:[usize;5],picked:usize,proposed:usize,accepted:usize}
impl Parts {
    fn lines(&self,case:&str,out:&mut String) {
        let mut put=|line:String| {println!("{line}");out.push_str(&line);out.push('\n');};
        put(format!("DRAFT_CYCLE case={case} part=one_row_pass {} picked_on_device={}",stat(&self.rows[1]),self.picked));
        put(format!("DRAFT_CYCLE case={case} part=proposal_call {} proposed_tokens={} accepted_tokens={}",stat(&self.proposal),self.proposed,self.accepted));
        put(format!("DRAFT_CYCLE case={case} part=waiting_rows {}",stat(&self.waiting)));
        for rows in 2..5 {if !self.rows[rows].is_empty() {
            put(format!("DRAFT_CYCLE case={case} part=pass rows={rows} {} kept_rows={}",stat(&self.rows[rows]),self.kept[rows]));
            put(format!("DRAFT_CYCLE case={case} part=cycle rows={rows} {}",stat(&self.cycle[rows])));
        }}
        put(format!("DRAFT_CYCLE case={case} part=verdict_all_kept {}",stat(&self.full)));
        put(format!("DRAFT_CYCLE case={case} part=verdict_rows_dropped {}",stat(&self.partial)));
    }
}
fn hand<E:Engine>(engine:&E,request:&spark_serve::protocol::Request,prompt:&[u32],plain:&[u32],depth:usize,parts:&mut Parts)->Result<()> {
    let mut session=engine.session()?;session.configure_request(request)?;
    if best(&session.prefill(prompt)?)!=plain[0] {return Err("draft-cycle: the strict session's first token differs from plain".into());}
    let (mut at,mut cycle)=(1usize,0usize);
    while at<plain.len() {
        let started=std::time::Instant::now();
        let target=session.draft_target().ok_or("draft-cycle: the strict session has no target")?;target.limit_model_depth(depth);
        let proposal=target.propose_model(&[],plain[at-1],depth)?.ok_or("draft-cycle: the session has no proposer")?;
        let mut rows=vec![plain[at-1]];rows.extend(&proposal.tokens);
        if !proposal.tokens.is_empty() {parts.proposal.push(proposal.proposal_ms);parts.waiting.push(proposal.catchup_ms);parts.proposed+=proposal.tokens.len();}
        // Every second proposal is the plain continuation: the pass then keeps every row.
        if cycle%2==1 {for (row,token) in rows.iter_mut().enumerate().skip(1) {if let Some(&id)=plain.get(at+row-1) {*token=id;}}}
        rows.truncate(plain.len()-at);let entry=target.checkpoint();
        let timer=std::time::Instant::now();let output=pass(engine,&mut session,&rows)?;parts.rows[rows.len()].push(ms(timer));
        let ids=match output {
            Predictions::Greedy(ids)=>{if rows.len()!=1 || ids.len()!=1 {return Err("draft-cycle: a picked pass has more than one row".into());}parts.picked+=1;ids},
            Predictions::Logits(logits)=>{if logits.len()!=rows.len() {return Err("draft-cycle: pass row count differs".into());}
                let kept=1+rows[1..].iter().zip(&logits).take_while(|(token,row)|best(row)==**token).count();logits[..kept].iter().map(|row|best(row)).collect()},
            _=>return Err("draft-cycle: a strict pass answers with tokens or logits".into()),
        };
        let timer=std::time::Instant::now();session.draft_target().ok_or("draft-cycle: the target is gone")?.truncate(entry+ids.len())?;
        if rows.len()>1 {if ids.len()==rows.len() {parts.full.push(ms(timer));} else {parts.partial.push(ms(timer));}
            parts.cycle[rows.len()].push(ms(started));parts.kept[rows.len()]+=ids.len();parts.accepted+=ids.len()-1;}
        for id in ids {if id!=plain[at] {return Err(format!("draft-cycle: drafted token {at} is {id}, plain decoding chose {}",plain[at]));}at+=1;}
        cycle+=1;
    }
    Ok(())
}
/// The server's speculator on the strict target, with passes as the scheduler sends them. Returns (cycles, proposed, accepted).
fn chooser<E:Engine>(engine:&E,request:&spark_serve::protocol::Request,prompt:&[u32],plain:&[u32],parts:&mut Parts)->Result<(usize,usize,usize)> {
    let mut session=engine.session()?;session.configure_request(request)?;let first=best(&session.prefill(prompt)?);
    let mut spec=Speculator::from_selected(prompt,prompt.len(),first,engine.vocab_size(),Config::strict(plain.len()-1),request.sampling)?;
    while !spec.is_done() {
        let target=session.draft_target().ok_or("draft-cycle: the strict session has no target")?;
        let plan=spec.prepare_with_target(target,16,|_|false,|_|true)?;let rows=plan.input.clone();
        let timer=std::time::Instant::now();let output=pass(engine,&mut session,&rows)?;parts.rows[rows.len().min(4)].push(ms(timer));
        let target=session.draft_target().ok_or("draft-cycle: the target is gone")?;
        match output {
            Predictions::Logits(logits)=>{spec.finish_exact(target,plan,&logits,|_|false,|_|true)?;},
            Predictions::Greedy(ids)=>{parts.picked+=1;spec.finish_greedy(target,plan,&ids,|_|false)?;},
            _=>return Err("draft-cycle: a strict pass answers with tokens or logits".into()),
        }
    }
    if first!=plain[0] || spec.tokens()!=&plain[1..] {return Err("draft-cycle: the speculator's tokens differ from plain decoding".into());}
    Ok((spec.stats.cycles,spec.stats.proposed,spec.stats.accepted))
}
fn cell(b:&Backend<'_>,c:&attention::Checkpoint,context:usize,count:usize,lines:&mut String)->Result<()> {
    let d=Device::new(b);let split=Split::new(&d,OneRank)?;let capacity=context+count+8;
    let service=Factory::new(&c.admission,split,capacity,vec![0])?.with_predictor(3)?;service.negotiate_available_bytes(1usize<<40)?;
    let mut limits=spark_tp::row_split::limits(capacity,count)?;limits.next_mtp_depth=Some(3);
    let request=|depth:usize|spark_serve::protocol::parse(format!(r#"{{"messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":{count},"mtp_depth":{depth}}}"#).as_bytes(),true,&limits);
    let prompt=super::first_step::prompt_tokens(context);let label=format!("c{context}");
    let put=|line:String,lines:&mut String| {println!("{line}");lines.push_str(&line);lines.push('\n');};
    // PLAIN: each step by its logits. The tokens and the logits are the reference of every other case.
    let mut plain=Vec::new();let mut reference=Vec::new();let mut steps=Vec::new();
    {let mut session=service.session()?;let mut logits=session.prefill(&prompt)?;
        for at in 0..count {let id=best(&logits);plain.push(id);reference.push(sha256::digest(&floats(&logits)?));
            if at+1<count {let timer=std::time::Instant::now();logits=session.step(id)?;steps.push(ms(timer));}}}
    put(format!("DRAFT_CYCLE case={label}:plain part=step_with_logits {}",stat(&steps)),lines);
    // GREEDY: twice, the first one warms the launch pieces of the greedy head.
    for round in 0..2 {let mut session=service.session()?;let mut steps=Vec::new();
        if best(&session.prefill(&prompt)?)!=plain[0] {return Err("draft-cycle: a second plain session differs".into());}
        for at in 1..count {let timer=std::time::Instant::now();let output=pass(&service,&mut session,&plain[at-1..at])?;steps.push(ms(timer));
            let Predictions::Greedy(ids)=output else {return Err("draft-cycle: plain greedy decode answered with logits".into());};
            if ids!=[plain[at]] {return Err(format!("draft-cycle: plain greedy token {at} differs from the logits' token"));}}
        put(format!("DRAFT_CYCLE case={label}:greedy round={round} part=step_picked {}",stat(&steps)),lines);}
    for depth in 1..=3 {let request=request(depth)?;
        let mut parts=Parts::default();let (cycles,proposed,accepted)=chooser(&service,&request,&prompt,&plain,&mut parts)?;
        put(format!("DRAFT_CYCLE case={label}:chooser{depth} ids_equal_plain=1 cycles={cycles} proposed={proposed} accepted={accepted}"),lines);
        parts.lines(&format!("{label}:chooser{depth}"),lines);
        for round in 0..2 {let mut parts=Parts::default();hand(&service,&request,&prompt,&plain,depth,&mut parts)?;
            put(format!("DRAFT_CYCLE case={label}:hand{depth} round={round} ids_equal_plain=1 tokens={count}"),lines);
            if round==1 {parts.lines(&format!("{label}:hand{depth}"),lines);}}
    }
    // A plain session after the drafting ones gives the reference logits again, bit for bit.
    {let mut session=service.session()?;let mut logits=session.prefill(&prompt)?;
        for at in 0..count {if sha256::digest(&floats(&logits)?)!=reference[at] {return Err(format!("draft-cycle: plain logits {at} differ after the drafting sessions"));}
            if at+1<count {logits=session.step(plain[at])?;}}}
    put(format!("DRAFT_CYCLE case={label} plain_logits_equal_after_drafting=1 ids_sha256={}",sha256::digest(&plain.iter().flat_map(|id|id.to_le_bytes()).collect::<Vec<_>>())),lines);
    if let Some(error)=service.fatal_error() {return Err(error);}
    service.shutdown()
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut lines=String::new();
    let c=checkpoint(1300);
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        b.configure_vector_pool(1<<30);
        for (context,count) in [(127usize,96usize),(1040,96)] {cell(b,&c,context,count,&mut lines)?;}Ok(())
    })?;
    let line=format!("DRAFT_CYCLE COMPLETE ids_equal_plain=1 layers={LAYERS} vocab={VOCAB} cells=127,1040 scope=one_box_fixture_host_wall_not_served_grade\n");
    print!("{line}");lines.push_str(&line);
    write(&out.join("LINES"),lines.as_bytes())?;write(&out.join("COMPLETE"),b"DRAFT-CYCLE PASS\n")?;Ok(())
}
