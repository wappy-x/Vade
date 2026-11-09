//! Host-side dense/hybrid command validation and gathered output verdicts.
use crate::Result;
use std::collections::{BTreeMap, BTreeSet};

pub const OPEN:u64=1;
pub const STEP:u64=2;
pub const PREFILL:u64=3;
pub const VERIFY:u64=4;
pub const TRUNCATE:u64=5;
pub const CLOSE:u64=6;
pub const STOP:u64=7;
pub const FORK:u64=8;
pub const GREEDY:u64=9;
pub const FAST:u64=10;
pub const KV_FORMAT:u64=11;
pub const PREFILL_ADVANCE:u64=12;
pub const PREFILL_END:u64=13;
pub const SEQUENCE_ADVANCE:u64=14;
pub const SEQUENCE_END:u64=15;
pub const SEQUENCE_START:u64=16;
pub const SEQUENCE_SLICE:u64=17;
pub const DECODE_BATCH:u64=18;
pub const PREFILL_BATCH:u64=19;
pub const DRAFT_DISABLE:u64=20; pub const DRAFT_AUTO:u64=21; pub const DRAFT_BUDGET:u64=22;
pub const DRAFT_PLAN:u64=23; pub const DRAFT_PROPOSE:u64=24; pub const DRAFT_PREPARE:u64=25;
pub const PREFIX:u64=26; pub const EXPAND:u64=27;
pub const OUTPUT_ERROR:&str="TP output rejected nonfinite logits";

pub fn validate_request(prompt:&[u32],request:&spark_serve::protocol::Request,context:usize,vocab:usize)->Result<()> {
    request.sampling.validate()?;
    if prompt.is_empty()||prompt.iter().any(|&t|t as usize>=vocab)
        ||prompt.len().checked_add(request.max_tokens).is_none_or(|end|end>context) {
        return Err("TP request outside vocabulary or context".into());
    }
    Ok(())
}

/// Validation never mutates the ledger or issues a collective. Workers compare
/// it with live positions before execution; a mismatch aborts the transport.
#[derive(Clone,Default)]
pub struct CommandState { pub sessions:BTreeMap<u64,usize> }
impl CommandState {
    pub fn payload_limit(&self,op:u64,context:usize)->usize {
        match op {
            DECODE_BATCH=>self.sessions.len().min(2048)*3,
            PREFILL_BATCH=>8192,
            PREFILL|VERIFY|FAST|PREFILL_ADVANCE|PREFILL_END|SEQUENCE_ADVANCE|SEQUENCE_END|SEQUENCE_START|DRAFT_PLAN=>context,
            DRAFT_PROPOSE=>context.saturating_add(1), DRAFT_AUTO=>2,
            _=>0,
        }
    }
    fn input(&self,id:u64,tokens:&[u32],context:usize,vocab:usize)->bool {
        !tokens.is_empty()&&tokens.iter().all(|&t|(t as usize)<vocab)
            &&self.sessions.get(&id).is_some_and(|&position|position.checked_add(tokens.len()).is_some_and(|end|end<=context))
    }
    pub fn validate(&self,op:u64,id:u64,arg:u64,tokens:&[u32],context:usize,vocab:usize)->Result<()> {
        let empty=tokens.is_empty();
        let position=self.sessions.get(&id).copied();
        let valid=match op {
            OPEN=>empty&&arg==0&&id!=0&&position.is_none(),
            STOP=>empty&&id==0&&arg==0&&self.sessions.is_empty(),
            CLOSE=>empty&&arg==0&&position.is_some(),
            FORK=>empty&&position.is_some()&&arg!=0&&!self.sessions.contains_key(&arg),
            STEP|GREEDY=>empty&&u32::try_from(arg).is_ok_and(|t|self.input(id,&[t],context,vocab)),
            TRUNCATE|PREFIX=>empty&&position.is_some_and(|p|arg<=p as u64),
            DRAFT_DISABLE=>empty&&arg==0&&position.is_some(),
            DRAFT_AUTO=>position.is_some()&&arg<=context as u64&&tokens.len()==2,
            DRAFT_BUDGET=>empty&&position.is_some()&&arg<=context as u64,
            DRAFT_PLAN=>position==Some(tokens.len())&&arg<=4&&tokens.iter().all(|&t|(t as usize)<vocab),
            DRAFT_PROPOSE=>position.is_some_and(|p|tokens.len()==p+1)&&arg<=4&&tokens.iter().all(|&t|(t as usize)<vocab),
            DRAFT_PREPARE=>empty&&position.is_some()&&arg<vocab as u64,
            EXPAND=>empty&&position.is_some()&&(1..=5).contains(&arg),
            KV_FORMAT=>empty&&position==Some(0)&&arg<=1,
            SEQUENCE_SLICE=>empty&&position.is_some(),
            PREFILL|VERIFY|FAST|SEQUENCE_START=>self.input(id,tokens,context,vocab)
                &&arg==tokens.len() as u64&&(!matches!(op,VERIFY|FAST)||tokens.len()<=512),
            PREFILL_ADVANCE|PREFILL_END|SEQUENCE_ADVANCE|SEQUENCE_END=>self.input(id,tokens,context,vocab)
                &&arg<=context as u64&&arg>=position.unwrap() as u64+tokens.len() as u64
                &&(!matches!(op,PREFILL_END|SEQUENCE_END)||arg==position.unwrap() as u64+tokens.len() as u64),
            DECODE_BATCH=>{
                let mut seen=BTreeSet::new();
                id==0&&arg<=1&&!empty&&tokens.len()%3==0&&tokens.len()/3<=2048
                    &&tokens.chunks_exact(3).all(|r|{
                        let id=u64::from(r[0])|(u64::from(r[1])<<32);
                        seen.insert(id)&&self.input(id,&r[2..],context,vocab)
                    })
            }
            PREFILL_BATCH=>{
                let (mut at,mut rows)=(0,0);
                let mut seen=BTreeSet::new();
                if id!=0||arg>1||empty||tokens.len()>8192{return Err("invalid_request_error: invalid TP prefill batch".into());}
                while at<tokens.len() {
                    if tokens.len()-at<3{return Err("invalid_request_error: truncated TP prefill batch".into());}
                    let id=u64::from(tokens[at])|(u64::from(tokens[at+1])<<32);
                    let count=tokens[at+2] as usize;at+=3;
                    if count>tokens.len()-at||!seen.insert(id)||!self.input(id,&tokens[at..at+count],context,vocab){return Err("invalid_request_error: invalid TP prefill row".into());}
                    rows+=count;at+=count;
                }
                rows<=2048
            }
            _=>false,
        };
        if valid {Ok(())}else{Err("invalid_request_error: invalid TP command or session state".into())}
    }
    pub fn apply(&mut self,op:u64,id:u64,arg:u64,tokens:&[u32]) {
        match op {
            OPEN=>{self.sessions.insert(id,0);}
            CLOSE=>{self.sessions.remove(&id);}
            FORK=>{let position=self.sessions[&id];self.sessions.insert(arg,position);}
            STEP|GREEDY=>*self.sessions.get_mut(&id).unwrap()+=1,
            PREFILL|VERIFY|FAST|PREFILL_ADVANCE|PREFILL_END|SEQUENCE_ADVANCE|SEQUENCE_END|SEQUENCE_START=>*self.sessions.get_mut(&id).unwrap()+=tokens.len(),
            TRUNCATE|PREFIX=>*self.sessions.get_mut(&id).unwrap()=arg as usize,
            DECODE_BATCH=>for r in tokens.chunks_exact(3){*self.sessions.get_mut(&(u64::from(r[0])|(u64::from(r[1])<<32))).unwrap()+=1;},
            PREFILL_BATCH=>{
                let mut at=0;
                while at<tokens.len(){let id=u64::from(tokens[at])|(u64::from(tokens[at+1])<<32);let count=tokens[at+2] as usize;*self.sessions.get_mut(&id).unwrap()+=count;at+=3+count;}
            }
            _=>(),
        }
    }
}

/// Invalid local argmax rows ride the existing candidate gather. Empty shards
/// carry -infinity with a real ID and never beat a finite vocabulary winner.
pub fn candidate(local:Result<(u32,f32)>,first:usize)->(u32,f32) {
    local.map(|(id,score)|(if score==f32::NEG_INFINITY{0}else{first as u32+id},score)).unwrap_or((u32::MAX,f32::NAN))
}
pub fn greedy(candidates:impl IntoIterator<Item=(u32,f32)>,vocab:usize)->Result<u32> {
    let mut best=(u32::MAX,f32::NEG_INFINITY);
    for(id,score)in candidates {
        if id as usize>=vocab||score.is_nan()||score==f32::INFINITY{return Err(OUTPUT_ERROR.into());}
        if score>best.1||score==best.1&&id<best.0 {best=(id,score);}
    }
    if best.1.is_finite(){Ok(best.0)}else{Err(OUTPUT_ERROR.into())}
}
pub fn finite(logits:&[f32])->Result<()> {
    if logits.iter().all(|v|v.is_finite()){Ok(())}else{Err(OUTPUT_ERROR.into())}
}
pub fn finite_bytes(logits:&[u8])->Result<()> {
    if logits.chunks_exact(4).all(|v|f32::from_le_bytes(v.try_into().unwrap()).is_finite()){Ok(())}else{Err(OUTPUT_ERROR.into())}
}

#[cfg(test)]
#[test]
fn invalid_commands_and_rank_verdicts() {
    let request=spark_serve::protocol::parse(br#"{"prompt":"probe","max_tokens":1}"#,false,&spark_serve::Limits::default()).unwrap();
    for (prompt,context) in [(&[0][..],1),(&[512][..],64),(&[][..],64)] {assert!(validate_request(prompt,&request,context,512).is_err());}
    let mut state=CommandState::default();state.apply(OPEN,1,0,&[]);
    for(op,id,arg,tokens)in [(STEP,1,512,vec![]),(VERIFY,1,2,vec![0,512]),(DECODE_BATCH,0,0,vec![1,0,0,1,0,0]),(TRUNCATE,1,1,vec![])] {assert!(state.validate(op,id,arg,&tokens,64,512).is_err());}
    for (op,arg,tokens) in [(DRAFT_DISABLE,0,vec![]),(DRAFT_AUTO,1,vec![1,0]),(DRAFT_BUDGET,1,vec![]),(DRAFT_PLAN,4,vec![]),(DRAFT_PROPOSE,4,vec![0]),(EXPAND,1,vec![]),(PREFIX,0,vec![])] {assert!(state.validate(op,1,arg,&tokens,64,512).is_ok());}
    for (op,arg,tokens) in [(DRAFT_PLAN,5,vec![]),(DRAFT_PROPOSE,4,vec![512]),(DRAFT_AUTO,1,vec![1]),(PREFIX,1,vec![])] {assert!(state.validate(op,1,arg,&tokens,64,512).is_err());}
    assert_eq!(state.sessions[&1],0);
    for rows in [[(0,1.0),candidate(Err("nonfinite logits".into()),0)], [candidate(Err("nonfinite logits".into()),0),(0,1.0)]] {assert_eq!(greedy(rows,512).unwrap_err(),OUTPUT_ERROR);}
    assert_eq!(greedy([(0,f32::NEG_INFINITY),(1,1.0),(2,1.0)],512).unwrap(),1);
    assert!(finite(&[1.0,f32::NAN]).is_err());assert!(finite_bytes(&f32::NAN.to_le_bytes()).is_err());assert_eq!(state.payload_limit(STEP,64),0);
}
