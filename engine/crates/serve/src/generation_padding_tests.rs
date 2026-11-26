use super::*;
use crate::{BatchOutput,BatchRows,Predictions,protocol::DraftMode};
use spark_engine::{draft::Target,sample::{Sampler,Sampling}};

// Shape from the tokenizer evidence fixture. Only the mock head uses these IDs.
const VOCAB:usize=154_880;
const VALID:u32=154_856;
#[derive(Clone,Copy,Debug)]
enum Path { Plain, PlainLogits, Trial, Over, Batch, DraftRow, DraftRows }
#[derive(Clone)]
struct Padded { path:Path, position:usize, best:u32 }
impl Padded {
    fn logits(&self)->Vec<f32> {let mut row=vec![-1.0;VOCAB];row[1]=8.0;row[self.best as usize]=9.0;row}
}
impl Session for Padded {
    fn fork(&self)->Result<Self,String> {Ok(self.clone())}
    fn reject_unmapped_greedy(&self)->bool {true}
    fn step(&mut self,_:u32)->Result<Vec<f32>,String> {self.position+=1;Ok(self.logits())}
    fn truncate(&mut self,p:usize)->Result<(),String> {
        if !matches!(self.path,Path::DraftRow|Path::DraftRows) {return Err("backend does not support prefix truncation".into());}
        self.position=p;Ok(())
    }
    fn draft_target(&mut self)->Option<&mut dyn Target> {
        if matches!(self.path,Path::DraftRow|Path::DraftRows) {Some(self)} else {None}
    }
}
impl Target for Padded {
    fn reject_unmapped_greedy(&self)->bool {true}
    fn checkpoint(&self)->usize {self.position}
    fn truncate(&mut self,p:usize)->Result<(),String> {self.position=p;Ok(())}
    fn step(&mut self,t:u32)->Result<Vec<f32>,String> {Session::step(self,t)}
}
struct Model;
impl Engine for Model {
    type Session<'a>=Padded;
    fn session(&self)->Result<Padded,String> {Ok(Padded{path:Path::Plain,position:1,best:VALID})}
    fn context_limit(&self)->usize {256}
    fn vocab_size(&self)->usize {VOCAB}
    fn eos_token_ids(&self)->&[u32] {&[]}
    fn step_batch(&self,rows:&mut [BatchRows<'_,Padded>])->Result<Vec<Predictions>,crate::BatchError> {
        Ok(rows.iter_mut().map(|r| {
            r.session.position+=r.tokens.len();
            if matches!(r.session.path,Path::DraftRows|Path::PlainLogits) {Predictions::Logits(r.tokens.iter().map(|_|r.session.logits()).collect())}
            else {Predictions::Greedy(vec![r.session.best])}
        }).collect())
    }
}
fn run(path:Path,best:u32)->Result<Vec<Event>,String> {
    let engine=Model;
    let mut generation=Generation::new(crate::cache::CachedSession {
        checkpoint_lease:None,session:Padded{path,position:1,best},tokens:vec![0],
        next_logits:Vec::new(),reserved_session_bytes:0,
    },1,Sampler::new(Sampling::greedy()).unwrap());
    generation.pending=Some(1);generation.usage.completion_tokens=1;
    generation.enable_draft(DraftMode::Strict,Sampling::greedy());
    generation.seed_draft(&[1,1,1,1]);
    let cap=if matches!(path,Path::DraftRows) {4} else {1};
    let (work,events)=generation.prepare(&engine,&[0],128,cap,1,|id|id<VALID)?;
    assert!(events.is_empty());let work=work.unwrap();
    if matches!(path,Path::DraftRows) {assert!((2..=cap).contains(&work.tokens().len()));}
    else {assert_eq!(work.tokens().len(),1);}
    let mut peer=Generation::new(crate::cache::CachedSession {
        checkpoint_lease:None,session:Padded{path:Path::Batch,position:1,best},tokens:vec![0],
        next_logits:Vec::new(),reserved_session_bytes:0,
    },1,Sampler::new(Sampling::greedy()).unwrap());
    peer.pending=Some(1);peer.usage.completion_tokens=1;
    let mut rows=vec![BatchRows {session:&mut generation.state.session,tokens:work.tokens(),output:BatchOutput::All,greedy:true,top_k:0}];
    if matches!(path,Path::Batch) {rows.push(BatchRows{session:&mut peer.state.session,tokens:&[1],output:BatchOutput::All,greedy:true,top_k:0});}
    let mut predictions=engine.step_batch(&mut rows).map_err(|e|e.message).unwrap();
    drop(rows);
    if matches!(path,Path::Batch) {
        assert_eq!(peer.complete(&engine,&[0],128,Work::Decode([1]),predictions.pop().unwrap(),|id|id<VALID),Err(spark_engine::draft::UNMAPPED_GREEDY.into()));
    }
    let prediction=predictions.remove(0);
    let events=generation.complete(&engine,&[0],128,work,prediction,|id|id<VALID)?;
    generation.prepare(&engine,&[0],128,cap,1,|id|id<VALID)?;
    Ok(events)
}
#[test]
fn padding_winner_has_the_same_result_in_plain_trial_stopped_batch_and_draft_steps() {
    for best in VALID..VOCAB as u32 {
        let plain=run(Path::Plain,best);
        assert_eq!(plain,Err(spark_engine::draft::UNMAPPED_GREEDY.into()));
        for path in [Path::PlainLogits,Path::Trial,Path::Over,Path::Batch,Path::DraftRow,Path::DraftRows] {
            assert_eq!(run(path,best),plain,"path {path:?}, padding ID {best}");
        }
    }
}
#[test]
fn padding_winner_in_a_multirow_verify_is_rejected() {
    assert!(run(Path::DraftRows,VALID).is_err());
}

fn verify(winners:&[u32],max:usize,eos:bool,compact:bool)->(Result<Vec<u32>,String>,Option<u32>,usize) {
    use spark_engine::draft::{Config,Speculator};
    let mut config=Config::strict(max);config.adaptive=false;
    let mut spec=Speculator::from_selected(&[0],1,1,VOCAB,config,Sampling::greedy()).unwrap();
    spec.seed(vec![1,1,1,1]);
    let plan=spec.prepare_exact_filtered(winners.len(),|t|eos && t==1,|id|id<VALID).unwrap();
    assert_eq!(plan.input.len(),winners.len());
    let mut target=Padded{path:Path::DraftRows,position:1+plan.input.len(),best:1};
    let result=if compact {spec.finish_greedy_filtered(&mut target,plan,winners,|t|eos && t==1,|id|id<VALID)}
        else {let rows=winners.iter().map(|&best|Padded{path:Path::DraftRows,position:0,best}.logits()).collect::<Vec<_>>();
            spec.finish_exact(&mut target,plan,&rows,|t|eos && t==1,|id|id<VALID)}.map(<[u32]>::to_vec);
    let result=result.and_then(|tokens|spec.check_greedy_error().map(|()|tokens));
    (result,spec.canonical_next(),target.position)
}
#[test]
fn every_consumed_draft_choice_including_the_bonus_refuses_padding() {
    for compact in [false,true] {for at in 0..4 {
        let mut winners=[1;4];winners[at]=VALID;
        let (result,_,position)=verify(&winners,16,false,compact);
        assert_eq!(result,Err(spark_engine::draft::UNMAPPED_GREEDY.into()),"choice {at}, compact {compact}");
        assert_eq!(position,at+2,"only the valid prefix commits before the failed choice");
    }}
}
#[test]
fn rejected_suffix_and_choices_after_eos_or_limit_do_not_fail() {
    for compact in [false,true] {
        let (result,next,position)=verify(&[2,VALID,VALID,VALID],16,false,compact);
        assert_eq!(result,Ok(vec![1]));assert_eq!(next,Some(2));assert_eq!(position,2);
        for (max,eos) in [(1,false),(16,true)] {
            let (result,next,position)=verify(&[VALID],max,eos,compact);
            assert_eq!(result,Ok(vec![1]));assert_eq!(next,None);assert_eq!(position,2);
        }
        let (result,next,position)=verify(&[1,1,1,1],16,false,compact);
        assert_eq!(result,Ok(vec![1,1,1,1]));assert_eq!(next,Some(1));assert_eq!(position,5);
    }
}
#[test]
fn the_first_prompt_choice_uses_the_same_unmapped_policy() {
    let engine=Model;
    let mut generation=Generation::new(crate::cache::CachedSession {
        checkpoint_lease:None,session:Padded{path:Path::Plain,position:1,best:VALID},tokens:vec![0],
        next_logits:Padded{path:Path::Plain,position:1,best:VALID}.logits(),reserved_session_bytes:0,
    },1,Sampler::new(Sampling::greedy()).unwrap());
    assert_eq!(generation.next(&engine,&[0],128,1,|id|id<VALID),Err(spark_engine::draft::UNMAPPED_GREEDY.into()));
}
#[test]
fn sampled_requests_keep_the_existing_tokenizer_mask() {
    let engine=Model;
    let mut generation=Generation::new(crate::cache::CachedSession {
        checkpoint_lease:None,session:Padded{path:Path::Plain,position:1,best:VALID},tokens:vec![0],
        next_logits:Padded{path:Path::Plain,position:1,best:VALID}.logits(),reserved_session_bytes:0,
    },1,Sampler::new(Sampling{temperature:0.6,top_k:1,..Sampling::greedy()}).unwrap());
    assert_eq!(generation.next(&engine,&[0],128,1,|id|id<VALID),Ok(Event::Token(1)));
}
#[test]
fn serial_filtered_drafting_uses_the_same_raw_winner_policy() {
    use spark_engine::draft::{Config,Speculator};
    for rows in [1,4] {
        let mut config=Config::strict(16);config.adaptive=false;
        let mut spec=Speculator::from_selected(&[0],1,1,VOCAB,config,Sampling::greedy()).unwrap();spec.seed(vec![1;4]);
        let mut target=Padded{path:Path::DraftRows,position:1,best:VALID};
        assert_eq!(spec.step_filtered(&mut target,rows,|_|false,|id|id<VALID),Ok(&[1][..]));
        assert_eq!(spec.step_filtered(&mut target,rows,|_|false,|id|id<VALID),Err(spark_engine::draft::UNMAPPED_GREEDY.into()));
        assert_eq!(target.position,2);
    }
}
#[test]
fn a_text_stop_can_consume_the_valid_prefix_before_a_deferred_padding_error() {
    let engine=Model;
    let mut generation=Generation::new(crate::cache::CachedSession {
        checkpoint_lease:None,session:Padded{path:Path::DraftRows,position:1,best:VALID},tokens:vec![0],
        next_logits:Vec::new(),reserved_session_bytes:0,
    },1,Sampler::new(Sampling::greedy()).unwrap());
    generation.pending=Some(1);generation.usage.completion_tokens=1;
    generation.enable_draft(DraftMode::Strict,Sampling::greedy());generation.seed_draft(&[1;4]);
    let (work,_)=generation.prepare(&engine,&[0],128,4,1,|id|id<VALID).unwrap();let work=work.unwrap();
    let rows=(0..work.tokens().len()).map(|i|Padded{path:Path::DraftRows,position:0,best:if i==0 {1} else {VALID}}.logits()).collect();
    generation.state.session.position+=work.tokens().len();
    let events=generation.complete(&engine,&[0],128,work,Predictions::Logits(rows),|id|id<VALID).unwrap();
    // The scheduler can match its byte stop or parser stop on this token and retire.
    assert_eq!(events,vec![Event::Token(1)]);
    // If no text stop matched, the next step fails without emitting the padding ID.
    assert_eq!(generation.prepare(&engine,&[0],128,4,1,|id|id<VALID).err(),Some(spark_engine::draft::UNMAPPED_GREEDY.into()));
}
#[test]
fn an_exact_kept_prompt_fork_refuses_the_saved_padding_winner_without_replay() {
    let engine=Model;
    let kept=crate::cache::CachedSession {
        checkpoint_lease:None,session:Padded{path:Path::Plain,position:1,best:VALID},tokens:vec![0],
        next_logits:Padded{path:Path::Plain,position:1,best:VALID}.logits(),reserved_session_bytes:0,
    };
    let restored=kept.fork_prefix_limit(&[0],1).unwrap().unwrap();
    assert_eq!(restored.tokens,kept.tokens);assert_eq!(restored.next_logits,kept.next_logits);
    let mut generation=Generation::new(restored,1,Sampler::new(Sampling::greedy()).unwrap());
    assert_eq!(generation.usage.cached_tokens,1);
    assert_eq!(generation.prepare(&engine,&[0],128,1,1,|id|id<VALID).err(),Some(spark_engine::draft::UNMAPPED_GREEDY.into()));
    assert_eq!(generation.state.session.position,1);assert_eq!(kept.session.position,1);
}
#[test]
fn every_chat_completes_and_a_padding_winner_does_not_fail_its_valid_batch_peer() {
    let engine=Model;
    for winners in [[VALID,1],[1,VALID]] {
        let mut chats=winners.map(|best|Generation::new(crate::cache::CachedSession {
            checkpoint_lease:None,session:Padded{path:Path::Batch,position:1,best},tokens:vec![0],
            next_logits:Vec::new(),reserved_session_bytes:0,
        },1,Sampler::new(Sampling::greedy()).unwrap()));
        for chat in &mut chats {chat.pending=Some(1);chat.usage.completion_tokens=1;}
        let mut rows=chats.iter_mut().map(|chat|BatchRows{session:&mut chat.state.session,tokens:&[1],output:BatchOutput::All,greedy:true,top_k:0}).collect::<Vec<_>>();
        let predictions=engine.step_batch(&mut rows).map_err(|e|e.message).unwrap();drop(rows);
        let results=chats.iter_mut().zip(predictions).map(|(chat,prediction)|chat.complete(&engine,&[0],128,Work::Decode([1]),prediction,|id|id<VALID)).collect::<Vec<_>>();
        assert_eq!(results.len(),2);
        for (result,winner) in results.into_iter().zip(winners) {
            assert_eq!(result,if winner==VALID {Err(spark_engine::draft::UNMAPPED_GREEDY.into())} else {Ok(vec![Event::Token(1)])});
        }
    }
}
