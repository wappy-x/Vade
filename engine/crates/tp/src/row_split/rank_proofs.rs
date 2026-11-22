//! One host decode step with independently computed local rows on every rank.
use super::{Exchange,Split,Topology};
use crate::{Result,bounded_decoder::{Decoder,Host,TestCheckpoint,greedy_logits},resident_weights::{Rows,merge_argmax}};
use engine_model::{plan::storage::WeightStorage,safetensors::Dtype};
use std::{collections::BTreeMap,sync::mpsc::{Receiver,Sender,channel},time::Duration};

type Bits=Vec<u32>;
fn bits(words:&[f32])->Bits {words.iter().map(|word|word.to_bits()).collect()}

struct HostExchange {
    topology:Topology,
    peers:Vec<Sender<Vec<f32>>>,
    sources:Vec<Receiver<Vec<f32>>>,
    gathers:Vec<Bits>,
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
    fn command(&mut self,_:&mut [u8])->Result<()> {Err("host decode step does not send commands".into())}
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

fn decode(checkpoint:&TestCheckpoint,parts:u32)->Vec<Step> {
    let mut ranks=(0..parts).map(|rank|HostExchange {
        topology:Topology {rank,parts},peers:Vec::new(),sources:Vec::new(),gathers:Vec::new(),
    }).collect::<Vec<_>>();
    // Every origin has a FIFO to each rank, including itself. The gather reads all origins the same way.
    for origin in 0..parts as usize {for target in 0..parts as usize {
        let (send,receive)=channel();
        ranks[origin].peers.push(send);
        ranks[target].sources.push(receive);
    }}
    std::thread::scope(|scope| {
        let workers=ranks.into_iter().map(|exchange|scope.spawn(move || {
            let rank=exchange.topology.rank;
            let run=|| -> Result<Step> {
                let split=Split::new(Host,exchange)?;
                let decoder=Decoder::new(&checkpoint.admission,&split,8)?;
                let (_,_,state)=decoder.step(1,&decoder.zero_state()?,|_,_|Ok(()))?;
                split.exchange.borrow_mut().gathers.clear();
                // Start from a populated recurrent state and sparse cache.
                let mut residuals=Vec::new();
                let (logits,token,next,hidden)=decoder.step_hidden(2,&state,|_,row| {residuals.push(bits(row));Ok(())})?;
                assert_eq!(compact_head(&logits,parts),token);
                let gathers=std::mem::take(&mut split.exchange.borrow_mut().gathers);
                Ok(Step {gathers,residuals,position:next.position,state:next.layers.into_iter()
                    .map(|layer|layer.into_iter().map(|(name,row)|(name,bits(&row))).collect()).collect(),
                    hidden:hidden.each_ref().map(|row|bits(row)),logits:bits(&logits),token})
            };
            run().unwrap_or_else(|error|panic!("rank {rank}/{parts}: {error}"))
        })).collect::<Vec<_>>();
        workers.into_iter().map(|worker|worker.join().unwrap()).collect()
    })
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
