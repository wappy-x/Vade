//! Host decode and prompt rows with independently computed local projections on every rank.
use super::{Exchange,Split,Topology};
use crate::{Result,bounded_decoder::{Decoder,HeadRows,Host,State,TestCheckpoint,greedy_logits},resident_weights::{Rows,merge_argmax}};
use engine_model::{plan::storage::WeightStorage,safetensors::Dtype};
use std::{collections::BTreeMap,sync::mpsc::{Receiver,Sender,channel},time::Duration};

mod drafting;

type Bits=Vec<u32>;
fn bits(words:&[f32])->Bits {words.iter().map(|word|word.to_bits()).collect()}

struct HostExchange {
    topology:Topology,
    peers:Vec<Sender<Vec<f32>>>,
    sources:Vec<Receiver<Vec<f32>>>,
    gathers:Vec<Bits>,
    commands:Vec<Sender<Vec<u8>>>,
    controller:Option<Receiver<Vec<u8>>>,
}
impl Exchange for HostExchange {
    fn topology(&self)->Topology {self.topology}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {
        for peer in &self.peers {peer.send(local.to_vec()).map_err(|e|e.to_string())?;}
        let mut full=Vec::with_capacity(rows);
        for (origin,source) in self.sources.iter().enumerate() {
            let held=crate::rows(rows,origin,self.sources.len());
            let words=source.recv_timeout(Duration::from_secs(10)).map_err(|e|e.to_string())?;
            assert_eq!(words.len(),held.len(),"rank {origin}, gather {}",self.gathers.len());
            full.extend(words);
        }
        self.gathers.push(bits(&full));
        Ok(full)
    }
    fn command(&mut self,words:&mut [u8])->Result<()> {
        if let Some(controller)=&self.controller {
            let message=controller.recv_timeout(Duration::from_secs(10)).map_err(|e|e.to_string())?;
            if message.len()!=words.len() {return Err("host command length differs".into());}
            words.copy_from_slice(&message);
        } else {
            for rank in &self.commands {rank.send(words.to_vec()).map_err(|e|e.to_string())?;}
        }
        Ok(())
    }
    fn complete(&mut self)->Result<()> {Ok(())}
}

#[derive(Debug,PartialEq,Eq)]
struct Step {
    gathers:Vec<Bits>,
    residuals:Vec<Bits>,
    position:u64,
    state:Vec<BTreeMap<String,Bits>>,
    hidden:[Bits;2],
    logits:Bits,
    token:u32,
}

fn compact_head(logits:&[f32],parts:u32)->u32 {
    let shape=[logits.len() as u64,32];
    let owners=(0..parts).map(|rank|Rows::new(&shape,WeightStorage::Mxfp8,Topology {rank,parts}).unwrap()).collect::<Vec<_>>();
    let messages=owners.iter().flat_map(|owner| {
        let local=&logits[owner.start..][..owner.count];
        let pick=greedy_logits(local,local.len()).unwrap();
        [local[pick as usize],pick as f32,1.0]
    }).collect::<Vec<_>>();
    merge_argmax(&messages,&owners).unwrap()
}

fn on_ranks<T:Send>(parts:u32,run:impl Fn(HostExchange)->Result<T>+Sync)->Vec<T> {
    let mut ranks=(0..parts).map(|rank|HostExchange {
        topology:Topology {rank,parts},peers:Vec::new(),sources:Vec::new(),gathers:Vec::new(),
        commands:Vec::new(),controller:None,
    }).collect::<Vec<_>>();
    // Every origin has a FIFO to each rank, including itself. The gather reads all origins the same way.
    for origin in 0..parts as usize {for target in 0..parts as usize {
        let (send,receive)=channel();
        ranks[origin].peers.push(send);
        ranks[target].sources.push(receive);
    }}
    let (controller,workers)=ranks.split_first_mut().expect("the fixture needs a controller");
    for worker in workers {
        let (send,receive)=channel();
        controller.commands.push(send);
        worker.controller=Some(receive);
    }
    std::thread::scope(|scope| {
        let run=&run;
        let workers=ranks.into_iter().map(|exchange|scope.spawn(move || {
            let rank=exchange.topology.rank;
            run(exchange).unwrap_or_else(|error|panic!("rank {rank}/{parts}: {error}"))
        })).collect::<Vec<_>>();
        workers.into_iter().map(|worker|worker.join().unwrap()).collect()
    })
}

fn layer_bits(state:&State<Vec<f32>>)->Vec<BTreeMap<String,Bits>> {
    state.layers.iter().map(|layer|layer.iter().map(|(name,row)|(name.clone(),bits(row))).collect()).collect()
}

fn decode(checkpoint:&TestCheckpoint,parts:u32)->Vec<Step> {
    on_ranks(parts,|exchange| {
        let split=Split::new(Host,exchange)?;
        let decoder=Decoder::new(&checkpoint.admission,&split,8)?;
        let (_,_,state)=decoder.step(1,&decoder.zero_state()?,|_,_|Ok(()))?;
        split.exchange.borrow_mut().gathers.clear();
        // Start from a populated recurrent state and sparse cache.
        let mut residuals=Vec::new();
        let (logits,token,next,hidden)=decoder.step_hidden(2,&state,|_,row| {residuals.push(bits(row));Ok(())})?;
        assert_eq!(compact_head(&logits,parts),token);
        let gathers=std::mem::take(&mut split.exchange.borrow_mut().gathers);
        Ok(Step {gathers,residuals,position:next.position,state:layer_bits(&next),
            hidden:hidden.each_ref().map(|row|bits(row)),logits:bits(&logits),token})
    })
}

#[derive(Debug,PartialEq,Eq)]
struct Row {
    logits:Bits,
    token:u32,
    position:u64,
    layers:Vec<BTreeMap<String,Bits>>,
}
impl Row {
    fn new(logits:&[f32],token:u32,state:&State<Vec<f32>>)->Self {
        Self {logits:bits(logits),token,position:state.position,layers:layer_bits(state)}
    }
}

#[derive(Debug,PartialEq,Eq)]
struct Batch {
    gathers:Vec<Bits>,
    hidden:Vec<Bits>,
    rows:Vec<Row>,
}

fn batch(checkpoint:&TestCheckpoint,parts:u32,count:usize)->Vec<Batch> {
    on_ranks(parts,|exchange| {
        let split=Split::new(Host,exchange)?;
        let decoder=Decoder::new(&checkpoint.admission,&split,32)?;
        let zero=decoder.zero_state()?;
        let prompts=[[1,2,5,4],[2,4,3,1],[3,1,2,5],[4,5,1,3]];
        let states=(0..count).map(|chat| {
            let len=1+chat%4;
            decoder.prefill_rows(&prompts[chat/2][..len],&zero,HeadRows::Last).map(|(_,state)|state)
        }).collect::<Result<Vec<_>>>()?;
        let input=states.iter().enumerate().map(|(chat,state)|(1+(chat%5) as u32,state)).collect::<Vec<_>>();
        split.exchange.borrow_mut().gathers.clear();
        let (answers,hidden)=decoder.step_batch_tracked(&input,false)?;
        assert_eq!(answers.len(),input.len());
        let gathers=std::mem::take(&mut split.exchange.borrow_mut().gathers);
        let mut rows=Vec::new();
        for ((logits,token,next),&(input,before)) in answers.iter().zip(&input) {
            let result=Row::new(logits,*token,next);
            let (alone,greedy,state)=decoder.step(input,before,|_,_|Ok(()))?;
            assert_eq!(result,Row::new(&alone,greedy,&state));
            assert_eq!(*token,compact_head(logits,parts));
            rows.push(result);
        }
        Ok(Batch {gathers,hidden:hidden.iter().map(|row|bits(row)).collect(),rows})
    })
}

#[test]
fn two_four_and_eight_chat_rows_keep_the_same_words_across_rank_counts() {
    let checkpoint=TestCheckpoint::new_with_capacity(Dtype::Bf16,true,32);
    for count in [2,4,8] {
        let pair=batch(&checkpoint,2,count);
        let four=batch(&checkpoint,4,count);
        let expected=&pair[0];
        assert_eq!(expected.rows.len(),count);
        assert_eq!(expected.hidden.len(),count);
        assert!(expected.gathers.len()>20*count);
        for (rank,result) in pair.iter().chain(&four).enumerate() {
            assert_eq!(result,expected,"{count} chats, rank result {rank}");
        }
    }
}

#[derive(Debug,PartialEq,Eq)]
struct Prompt {
    gathers:Vec<Bits>,
    logits:Vec<Bits>,
    tokens:Vec<u32>,
    full:(u64,Vec<BTreeMap<String,Bits>>),
    prefix:(u64,Vec<BTreeMap<String,Bits>>),
    continuation:Row,
}

#[test]
fn grouped_prompt_and_retained_prefix_keep_the_same_words_across_rank_counts() {
    let checkpoint=TestCheckpoint::new_with_capacity(Dtype::Bf16,true,32);
    let tokens=[1,2,4,5,3,2,4,1,3,4,5,2];
    let prompt=|parts|on_ranks(parts,|exchange| {
        let split=Split::new(Host,exchange)?;
        let decoder=Decoder::new(&checkpoint.admission,&split,32)?;
        let zero=decoder.zero_state()?;
        let (logits,full,open)=decoder.prefill_rows_open(&tokens,&zero,HeadRows::All)?;
        let gathers=std::mem::take(&mut split.exchange.borrow_mut().gathers);
        let prefix=decoder.prefix_state(&open,&zero,&full,7)?;
        let (_,reference)=decoder.prefill_rows(&tokens[..7],&zero,HeadRows::Last)?;
        assert_eq!(layer_bits(&prefix),layer_bits(&reference));
        let (continued,greedy,next)=decoder.step(5,&prefix,|_,_|Ok(()))?;
        let tokens=logits.iter().map(|row|compact_head(row,parts)).collect();
        Ok(Prompt {gathers,logits:logits.iter().map(|row|bits(row)).collect(),tokens,
            full:(full.position,layer_bits(&full)),prefix:(prefix.position,layer_bits(&prefix)),
            continuation:Row::new(&continued,greedy,&next)})
    });
    let pair=prompt(2);
    let four=prompt(4);
    assert_eq!(pair[0].logits.len(),tokens.len());
    assert_eq!((pair[0].full.0,pair[0].prefix.0),(12,7));
    assert!(pair[0].gathers.len()>20*tokens.len());
    for (rank,result) in pair.iter().chain(&four).enumerate() {
        assert_eq!(result,&pair[0],"grouped prompt, rank result {rank}");
    }
}

#[test]
fn decode_step_keeps_every_projected_row_state_word_and_logit() {
    let checkpoint=TestCheckpoint::new_with_head(Dtype::Bf16,true);
    let pair=decode(&checkpoint,2);
    let four=decode(&checkpoint,4);
    let expected=&pair[0];
    assert_eq!(expected.residuals.len(),2);
    assert_eq!(expected.position,2);
    assert!(!expected.state[0].is_empty() && !expected.state[1].is_empty());
    assert!(expected.gathers.len()>20,"the fixture must run recurrent, sparse, dense and routed projections");
    for step in pair.iter().chain(&four) {
        assert_eq!(step.gathers.len(),expected.gathers.len());
        for (index,(actual,want)) in step.gathers.iter().zip(&expected.gathers).enumerate() {
            assert_eq!(actual,want,"projection gather {index}");
        }
        assert_eq!(step,expected);
    }
}

#[test]
fn compact_argmax_keeps_the_global_id_across_shard_boundaries() {
    let mut logits=[-8.0;16];
    for (first,second) in [(3,4),(7,8),(11,12)] {
    logits[first] = -0.0;
        logits[second]=0.0;
        assert_eq!(compact_head(&logits,2),first as u32);
        assert_eq!(compact_head(&logits,4),first as u32);
    logits[first] = -8.0;
        assert_eq!(compact_head(&logits,2),second as u32);
        assert_eq!(compact_head(&logits,4),second as u32);
    logits[second] = -8.0;
    }
}
