use super::{Bits,Host,Row,TestCheckpoint,bits,greedy_logits,on_ranks};
use crate::{Result,bounded_decoder::State,row_split::{Factory,Split}};
use spark_engine::draft::{Config,Speculator,Target};
use spark_serve::{Engine,Session as _};
use std::collections::BTreeMap;

const CONTEXT:usize=80;

#[derive(Debug,PartialEq,Eq)]
struct StateWords {
    position:u64,
    layers:Vec<BTreeMap<String,Bits>>,
    expanded:BTreeMap<usize,(usize,Bits)>,
}
impl StateWords {
    fn new(state:&State<Vec<f32>>)->Self {
        Self {position:state.position,layers:super::layer_bits(state),
            expanded:state.expanded.iter().map(|(layer,cache)|(*layer,(cache.1,bits(&cache.0)))).collect()}
    }
    fn same_serial_state(&self,serial:&Self) {
        assert_eq!(self.position,serial.position);
        assert_eq!(self.layers,serial.layers);
        // A grouped pass can retain more expanded rows than the serial path.
        // The appended caches above are authoritative and the continuation reads both forms.
    }
}

#[derive(Debug,PartialEq,Eq)]
struct Round {
    proposals:Vec<u32>,
    probability:Vec<Option<u32>>,
    checked:Vec<Bits>,
    accepted:Vec<u32>,
    predictor:BTreeMap<String,Bits>,
    predictor_rows:u64,
    trail:Vec<(u32,Bits)>,
    state:StateWords,
    continuation:Vec<Row>,
}

#[derive(Clone,Copy)]
enum Offers { Model, Accept, Reject }

fn rounds<C:FnMut(usize,bool,Offers)->Result<Round>>(mut run:C)->Result<Vec<Round>> {
    let mut out=Vec::new();
    for position in [3,7,15] {for default in [false,true] {
        for offers in [Offers::Model,Offers::Accept,Offers::Reject] {out.push(run(position,default,offers)?);}
    }}
    Ok(out)
}

fn drafted(checkpoint:&TestCheckpoint,parts:u32)->Vec<(Vec<Bits>,Vec<Round>)> {
    on_ranks(parts,|exchange| {
        let controller=exchange.controller.is_none();
        let factory=Factory::new(&checkpoint.admission,Split::new(Host,exchange)?,CONTEXT,vec![0])?.with_predictor(3)?;
        let results=if controller {
            let results=rounds(|position,default,offers|round(&factory,position,default,offers));
            factory.shutdown()?;
            results?
        } else {factory.worker()?;Vec::new()};
        let gathers=std::mem::take(&mut factory.split.exchange.borrow_mut().gathers);
        Ok((gathers,results))
    })
}

fn round(factory:&Factory<'_,Host,super::HostExchange>,position:usize,default:bool,offers:Offers)->Result<Round> {
    // Also exercise the whole strict depth when the fixture's confidence would shorten it.
    let whole_depth=!default && matches!(offers,Offers::Accept);
    factory.set_confidence(!whole_depth);
    let vocab=factory.checkpoint.description().vocab as usize;
    let prompt=(0..position).map(|row|((row*5+3)%vocab) as u32).collect::<Vec<_>>();
    let mut limits=crate::row_split::limits(CONTEXT,72)?;
    limits.next_mtp_depth=Some(3);
    let body=if default {
        br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":72,"temperature":0}"#.as_slice()
    } else {
        br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":72,"temperature":0,"draft":"strict","mtp_depth":2}"#.as_slice()
    };
    let mut request=spark_serve::protocol::parse(body,true,&limits).unwrap();
    if default {request.mtp_depth=Some(crate::row_split::TRIAL_DEPTH);}

    let mut plain=factory.session()?;
    let mut logits=plain.prefill(&prompt)?;
    let mut serial=Vec::new();
    for _ in 0..6 {
        let token=greedy_logits(&logits,vocab)?;
        logits=spark_serve::Session::step(&mut plain,token)?;
        serial.push((token,logits.clone(),StateWords::new(plain.state.as_ref().unwrap())));
    }
    drop(plain);

    let mut draft=factory.session()?;
    draft.configure_request(&request)?;
    assert_eq!(draft.alone,default);
    let logits=draft.prefill(&prompt)?;
    let first=greedy_logits(&logits,vocab)?;
    assert_eq!(first,serial[0].0);
    let logits=Target::step(&mut draft,first)?;
    assert_eq!(bits(&logits),bits(&serial[0].1));
    let next=serial[1].0;
    let depth=if default {3} else {2};
    let (proposals,mass,_)=draft.propose(next,depth)?;
    assert!(!proposals.is_empty());
    if whole_depth {assert_eq!(proposals.len(),depth);}
    let probability=mass.iter().map(|p|p.map(f32::to_bits)).collect();

    // Exercise the actual model's words and send known matching and mismatching
    // continuations through the same verifier so both verdict branches must run.
    let width=if default {1} else {2};
    let mut input=vec![next];
    match offers {
        Offers::Model=>input.extend(proposals.iter().take(width).copied()),
        Offers::Accept|Offers::Reject=>input.extend(serial[2..2+width].iter().map(|row|row.0)),
    }
    if matches!(offers,Offers::Reject) {input[1]=(input[1]+1)%vocab as u32;}
    let mut history=prompt;
    history.push(first);
    let config=Config {adaptive:false,..Config::strict(6)};
    let mut spec=Speculator::from_selected(&history,history.len(),next,vocab,config,request.sampling)?;
    spec.seed(input.clone());
    let pass=spec.prepare_exact(input.len(),|_|false)?;
    assert_eq!(pass.input,input);
    let rows=draft.verify_batch(&pass.input)?;
    let accepted=spec.finish_exact(&mut draft,pass,&rows,|_|false,|_|true)?.to_vec();
    let kept=accepted.len();
    assert!(kept>0 && kept<=input.len());
    assert_eq!(spec.stats.accepted,kept-1);
    if matches!(offers,Offers::Accept) {assert_eq!(kept,input.len());}
    if matches!(offers,Offers::Reject) {assert_eq!(kept,1);assert_eq!(spec.stats.rollbacks,1);}
    assert_eq!(accepted,serial[1..1+kept].iter().map(|row|row.0).collect::<Vec<_>>());
    let checked=rows.iter().map(|row|bits(row)).collect::<Vec<_>>();
    for (row,want) in checked[..kept].iter().zip(&serial[1..]) {assert_eq!(*row,bits(&want.1));}

    let (predictor,trail)=draft.open.as_ref().unwrap().predictor_words();
    let predictor=predictor.unwrap();
    let predictor_rows=predictor.rows();
    let predictor=predictor.layers().iter().map(|(name,value)|(name.clone(),bits(value))).collect();
    let trail=trail.iter().map(|(token,hidden)|(*token,bits(hidden))).collect::<Vec<_>>();
    assert_eq!(trail.iter().map(|row|row.0).collect::<Vec<_>>(),accepted);
    assert_eq!(draft.checkpoint(),position+1+kept);
    draft.settle()?;
    let state=StateWords::new(draft.state.as_ref().unwrap());
    state.same_serial_state(&serial[kept].2);

    let mut continuation=Vec::new();
    for (token,logits,expected) in &serial[1+kept..] {
        let output=draft.token(*token,false)?;
        assert_eq!(bits(&output),bits(logits));
        let after=draft.state.as_ref().unwrap();
        StateWords::new(after).same_serial_state(expected);
        continuation.push(Row::new(&output,greedy_logits(&output,vocab)?,after));
    }
    Ok(Round {proposals,probability,checked,accepted,predictor,predictor_rows,trail,state,continuation})
}

#[test]
fn drafted_round_keeps_proposals_verifier_rows_and_both_verdict_states_across_rank_counts() {
    let checkpoint=TestCheckpoint::new_with_predictor(CONTEXT as u64);
    let two=drafted(&checkpoint,2);
    let four=drafted(&checkpoint,4);
    for run in [&two,&four] {
        let (controller,workers)=run.split_first().unwrap();
        for peer in workers {assert!(peer.0==controller.0,"peer projection words differ");}
    }
    assert!(four[0].0==two[0].0,"two- and four-rank projection words differ");
    assert_eq!(four[0].1,two[0].1);
    eprintln!("draft rounds={} gathered_words={} checked_words={}",four[0].1.len(),
        four[0].0.iter().map(Vec::len).sum::<usize>(),four[0].1.iter().flat_map(|r|&r.checked).map(Vec::len).sum::<usize>());
}
