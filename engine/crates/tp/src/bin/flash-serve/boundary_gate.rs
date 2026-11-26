//! Diagnostic served API gate. No HTTP listener is opened. Both ranks use the normal Factory.
use spark_serve::{Engine,Session,BatchRows,BatchOutput,Predictions};
use spark_tp::Result;
use std::{fs,path::Path,io::Write};
const CHATS:usize=8;
const STEPS:usize=32;
const WIDTHS:[usize;9]=[1,2,4,8,5,4,3,4,5];
fn best(row:&[f32])->u32 {row.iter().enumerate().fold(0,|at,(i,&v)|if v>row[at] {i} else {at}) as u32}
fn bits(row:&[f32])->Vec<u32> {row.iter().map(|v|v.to_bits()).collect()}
fn fail(label:&str,got:&[f32],want:&[u32])->Result<()> {
    if got.len()!=want.len() || got.iter().zip(want).any(|(a,b)|a.to_bits()!=*b) {return Err(format!("boundary gate logit bits differ: {label}"));}Ok(())
}
fn put(out:&mut fs::File,line:String)->Result<()> {println!("{line}");writeln!(out,"{line}").map_err(|e|e.to_string())}
/// `sample`: accounted device bytes, peak device bytes, fused boundary node count.
/// `resident + workspace + sessions` is the startup admission bound supplied by the real service.
pub fn run<E:Engine>(engine:&E,dir:&Path,context:usize,charged:usize,sample:impl Fn()->(usize,usize,u64))->Result<()> {
    fs::create_dir_all(dir).map_err(|e|e.to_string())?;
    let mut log=fs::OpenOptions::new().write(true).create_new(true).open(dir.join("BOUNDARY-GATE")).map_err(|e|e.to_string())?;
    let mut exact=fs::OpenOptions::new().write(true).create_new(true).open(dir.join("ALONE-BITS")).map_err(|e|e.to_string())?;
    let mut exact_ids=fs::OpenOptions::new().write(true).create_new(true).open(dir.join("ALONE-IDS")).map_err(|e|e.to_string())?;
    let prompts=(0..CHATS).map(|chat| {let s=fs::read_to_string(dir.join(format!("{chat}.ids"))).map_err(|e|e.to_string())?;
        let ids=s.split_whitespace().map(|v|v.parse::<u32>().map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
        if ids.is_empty() || ids.len()+STEPS+4>context || ids.iter().any(|&id|id as usize>=engine.vocab_size()) {return Err("boundary gate prompt extent or ID differs".into());}Ok(ids)
    }).collect::<Result<Vec<_>>>()?;
    for (i,p) in prompts.iter().enumerate() {if prompts[..i].contains(p) {return Err("boundary gate needs different chats".into());}}
    let (start,peak,_)=sample();if start>charged {return Err("boundary gate startup exceeds admission".into());}
    put(&mut log,format!("BOUNDARY STARTUP BYTES PASS live={start} peak={peak} admission={charged}"))?;
    let mut limits=spark_tp::row_split::limits(context,STEPS)?;limits.next_mtp_depth=Some(3);
    let request=|depth|spark_serve::protocol::parse(format!(r#"{{"messages":[{{"role":"user","content":"Check."}}],"max_tokens":{STEPS},"temperature":0,"draft":"{}","mtp_depth":{depth}}}"#,if depth==0 {"plain"} else {"strict"}).as_bytes(),true,&limits);
    let plain=request(0)?;let strict=request(3)?;
    let mut reference=Vec::new();let mut first=Vec::new();
    for (chat,prompt) in prompts.iter().enumerate() {
        let mut session=engine.session()?;session.configure_request(&plain)?;let initial=session.prefill(prompt)?;let mut token=best(&initial);first.push((token,bits(&initial)));
        for word in &first[chat].1 {exact.write_all(&word.to_le_bytes()).map_err(|e|e.to_string())?;}
        exact_ids.write_all(&token.to_le_bytes()).map_err(|e|e.to_string())?;
        let mut rows=Vec::new();
        for step in 0..STEPS {let before=sample();let row=session.step(token)?;token=best(&row);let words=bits(&row);
            for word in &words {exact.write_all(&word.to_le_bytes()).map_err(|e|e.to_string())?;}
            exact_ids.write_all(&token.to_le_bytes()).map_err(|e|e.to_string())?;
            rows.push((token,words));
            if step==0 {let (live,peak,_)=sample();if live>charged || peak>charged {return Err("boundary gate first step exceeds admission".into());}
                put(&mut log,format!("BOUNDARY FIRST STEP BYTES PASS chat={chat} before={} live={live} peak={peak} admission={charged}",before.0))?;}
        }reference.push(rows);
    }
    exact.sync_all().map_err(|e|e.to_string())?;exact_ids.sync_all().map_err(|e|e.to_string())?;
    for (depth,picked) in [(0,false),(3,false),(0,true)] {
        let mut sessions=Vec::new();for (chat,prompt) in prompts.iter().enumerate() {let mut s=engine.session()?;s.configure_request(if depth==0 {&plain} else {&strict})?;
            let row=s.prefill(prompt)?;fail("prefill",&row,&first[chat].1)?;sessions.push((chat,s));}
        let mut at=[0;CHATS];let mut tick=0;let mut proposed=0;let mut checked=0;let mut dropped=0;let mut full=0;
        while at.iter().any(|&n|n<STEPS) {
            let width=WIDTHS[tick%WIDTHS.len()];sessions.rotate_left(1);
            let active=sessions.iter_mut().filter(|(chat,_)|at[*chat]<STEPS).take(width).collect::<Vec<_>>();
            let mut active=active;let ids=active.iter().map(|(chat,_)|*chat).collect::<Vec<_>>();
            let next=ids.iter().map(|&chat|if at[chat]==0 {first[chat].0} else {reference[chat][at[chat]-1].0}).collect::<Vec<_>>();
            let mut tokens=next.iter().map(|&id|vec![id]).collect::<Vec<_>>();
            if depth>0 {
                for (row,(_,s)) in active.iter_mut().enumerate() {
                    let target=s.draft_target().ok_or("boundary gate has no proposer")?;target.limit_model_depth(depth);
                    let guess=target.propose_model(&[],next[row],depth)?.ok_or("boundary gate has no model proposal")?;
                    proposed+=guess.tokens.len();tokens[row].extend(guess.tokens);
                    let chat=ids[row];tokens[row].truncate(STEPS-at[chat]);
                    // Exercise both kept and dropped rows even when this fixture's proposer always agrees.
                    if tokens[row].len()>1 {if tick%2==0 {for (i,t) in tokens[row].iter_mut().enumerate().skip(1) {*t=reference[chat][at[chat]+i-1].0;}}
                        else {tokens[row][1]=(reference[chat][at[chat]].0+1)%engine.vocab_size() as u32;}}
                }
            }
            let entries=active.iter_mut().map(|(_,s)|s.draft_target().map(|t|t.checkpoint())).collect::<Vec<_>>();
            let before=sample();let timer=std::time::Instant::now();
            let mut batch=active.iter_mut().zip(&tokens).map(|((_,s),tokens)|BatchRows {session:s,tokens,output:BatchOutput::All,greedy:picked,top_k:0}).collect::<Vec<_>>();
            let results=engine.step_batch(&mut batch).map_err(|e|e.message)?;let ms=timer.elapsed().as_secs_f64()*1000.0;
            if results.len()!=ids.len() {return Err("boundary gate result count differs".into());}
            for (row,result) in results.into_iter().enumerate() {
                if let Predictions::Greedy(got)=&result {
                    let chat=ids[row];
                    if !picked || got.as_slice()!=[reference[chat][at[chat]].0] {return Err("boundary gate picked ID differs".into());}
                    at[chat]+=1;checked+=1;continue;
                }
                let Predictions::Logits(logits)=result else {return Err("boundary gate requires every logit word".into());};
                if logits.len()!=tokens[row].len() {return Err("boundary gate logit row count differs".into());}
                let kept=1+tokens[row].iter().skip(1).zip(&logits).take_while(|(id,l)|**id==best(l)).count();let chat=ids[row];
                for (i,l) in logits.iter().take(kept).enumerate() {let (want,words)=&reference[chat][at[chat]+i];fail(&format!("depth={depth} chat={chat} step={}",at[chat]+i),l,words)?;
                    if best(l)!=*want {return Err("boundary gate ID differs".into());}checked+=1;}
                if depth>0 {active[row].1.draft_target().ok_or("boundary gate lost draft target")?.truncate(entries[row].ok_or("boundary gate has no draft target")?+kept)?;
                    if tokens[row].len()>1 {if kept==tokens[row].len() {full+=1;} else {dropped+=1;}}}
                at[chat]+=kept;
            }
            let (live,peak,nodes)=sample();if live>charged || peak>charged {return Err("boundary gate cohort exceeds admission".into());}
            put(&mut log,format!("BOUNDARY STEP depth={depth} picked={} tick={tick} chats={ids:?} rows={} ms={ms:.6} boundary_launches={} live={live} peak={peak} logits_checked={} ids_equal=1",u32::from(picked),ids.len(),nodes-before.2,u32::from(!picked)))?;
            tick+=1;
        }
        if depth>0 && (proposed==0 || full==0 || dropped==0) {return Err("boundary gate did not exercise proposals and both verdicts".into());}
        put(&mut log,format!("BOUNDARY COHORT PASS depth={depth} picked={} chats={CHATS} checked={checked} proposed={proposed} full={full} dropped={dropped} ids_equal=1 logit_bits_checked={}",u32::from(picked),u32::from(!picked)))?;
    }
    put(&mut log,"BOUNDARY PAIR PASS changing=1 drafting=off,on".into())
}
#[cfg(test)] mod tests {
    use super::*;
    struct Tiny {bad:bool}
    struct Chat {position:usize,strict:bool}
    impl Session for Chat {
        fn configure_request(&mut self,r:&spark_serve::protocol::Request)->Result<()> {self.strict=r.mtp_depth==Some(3);Ok(())}
        fn step(&mut self,token:u32)->Result<Vec<f32>> {self.position+=1;let mut row=vec![-(self.position as f32);16];row[(token as usize+1)%16]=self.position as f32;Ok(row)}
        fn draft_target(&mut self)->Option<&mut dyn spark_engine::draft::Target> {self.strict.then_some(self)}
    }
    impl spark_engine::draft::Target for Chat {
        fn checkpoint(&self)->usize {self.position}
        fn truncate(&mut self,p:usize)->Result<()> {self.position=p;Ok(())}
        fn step(&mut self,t:u32)->Result<Vec<f32>> {Session::step(self,t)}
        fn propose_model(&mut self,_:&[u32],next:u32,limit:usize)->Result<Option<spark_engine::draft::ModelProposal>> {
            Ok(Some(spark_engine::draft::ModelProposal {tokens:(1..=limit).map(|i|(next+i as u32)%16).collect(),proposal_ms:0.0,catchup_ms:0.0}))
        }
    }
    impl Engine for Tiny {
        type Session<'a>=Chat;
        fn session(&self)->Result<Chat> {Ok(Chat {position:0,strict:false})}
        fn context_limit(&self)->usize {128}
        fn vocab_size(&self)->usize {16}
        fn eos_token_ids(&self)->&[u32] {&[0]}
        fn step_batch(&self,rows:&mut [BatchRows<'_,Chat>])->std::result::Result<Vec<Predictions>,spark_serve::BatchError> {
            rows.iter_mut().map(|r| {let mut values=r.tokens.iter().map(|&t|Session::step(r.session,t)).collect::<Result<Vec<_>>>().map_err(|message|spark_serve::BatchError {request:None,message})?;
                if self.bad {values[0][0]=f32::from_bits(values[0][0].to_bits()^1);}Ok(if r.greedy {Predictions::Greedy(values.iter().map(|row|best(row)).collect())} else {Predictions::Logits(values)})}).collect()
        }
    }
    #[test]
    fn served_gate_checks_changing_chats_and_rejects_one_logit_bit() {
        for bad in [false,true] {
            let dir=std::env::temp_dir().join(format!("vade-boundary-gate-{}-{bad}",std::process::id()));fs::create_dir(&dir).unwrap();
            for chat in 0..CHATS {fs::write(dir.join(format!("{chat}.ids")),format!("{}",chat+1)).unwrap();}
            let result=run(&Tiny {bad},&dir,128,1,||(0,0,0));
            if bad {assert!(result.unwrap_err().contains("logit bits differ"));} else {result.unwrap();let log=fs::read_to_string(dir.join("BOUNDARY-GATE")).unwrap();
                assert!(log.contains("BOUNDARY PAIR PASS") && log.contains("BOUNDARY COHORT PASS depth=3"));
                for rows in [1,2,4,8] {assert!(log.contains(&format!("rows={rows} ms=")));}}
            fs::remove_dir_all(dir).unwrap();
        }
    }
}
