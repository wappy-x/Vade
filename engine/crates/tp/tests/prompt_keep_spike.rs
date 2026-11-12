//! SPIKE (design proof, host twins, one chat): the state a prompt read leaves can be KEPT and
//! read again later. A decode step and a prompt group take the old state by reference and
//! return a new one, so the kept state needs no copy. A second turn that starts from the kept
//! state gives the same logit bits and the same IDs as a fresh read of the whole prompt.
use engine_model::safetensors::Dtype;
use spark_tp::bounded_decoder::{Decoder,HeadRows,Host,State};
#[path="support/attention_checkpoint.rs"] mod fixture;

fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
fn best(v:&[f32])->u32 {(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32}
type Kept=State<Vec<f32>>;

/// Read `tokens` one by one from `from`. Returns each row's logits and the state after the last.
fn scalar(decoder:&Decoder<'_,Host>,tokens:&[u32],from:Option<&Kept>)->(Vec<Vec<f32>>,Kept) {
    let zero;let mut owned:Option<Kept>=None;let mut rows=Vec::new();
    let start=match from {Some(state)=>state,None=>{zero=decoder.zero_state().unwrap();&zero}};
    for &token in tokens {let (logits,_,next)=decoder.step(token,owned.as_ref().unwrap_or(start),|_,_|Ok(())).unwrap();rows.push(logits);owned=Some(next);}
    (rows,owned.unwrap())
}
/// Decode `count` tokens greedily after `logits`, from `from`. The state `from` is only read.
fn decode(decoder:&Decoder<'_,Host>,logits:&[f32],from:&Kept,count:usize)->Vec<(u32,Vec<u32>)> {
    let mut out=Vec::new();let mut owned:Option<Kept>=None;let mut logits=logits.to_vec();
    for _ in 0..count {let id=best(&logits);let (next_logits,_,next)=decoder.step(id,owned.as_ref().unwrap_or(from),|_,_|Ok(())).unwrap();out.push((id,bits(&next_logits)));logits=next_logits;owned=Some(next);}
    out
}

#[test]
fn a_kept_prompt_state_gives_the_bits_of_a_fresh_read_on_the_scalar_path() {
    let _constructor:fn(Dtype)->fixture::Checkpoint=fixture::Checkpoint::new;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);let _=&c.operands;
    let decoder=Decoder::new(&c.admission,Host,80).unwrap();
    // Turn 1: a prompt of 13 tokens (past three pool boundaries of 4), then its answer.
    let prompt=[1u32,3,2,7,5,11,4,9,6,13,8,2,10];
    let (rows,kept)=scalar(&decoder,&prompt,None);
    let answer=decode(&decoder,rows.last().unwrap(),&kept,9);
    assert_eq!(kept.position,prompt.len() as u64,"the kept state stays at the prompt's end while the answer is decoded");
    // Turn 2 and turn 3: the same prompt, then two different continuations. Neither is the answer the model gave.
    for extra in [vec![12u32,1,14,3,5,7,2],vec![4u32,4,9,1]] {
        assert_ne!(extra[0],answer[0].0,"the continuation must leave the decoded answer at its first token");
        let (reused_rows,reused)=scalar(&decoder,&extra,Some(&kept));
        let whole=[prompt.as_slice(),extra.as_slice()].concat();
        let (fresh_rows,fresh)=scalar(&decoder,&whole,None);
        assert_eq!(reused.position,fresh.position);
        for (i,row) in reused_rows.iter().enumerate() {assert_eq!(bits(row),bits(&fresh_rows[prompt.len()+i]),"row {i} after the kept prompt");}
        // The answers that follow are the same IDs and the same logit bits.
        assert_eq!(decode(&decoder,reused_rows.last().unwrap(),&reused,6),decode(&decoder,fresh_rows.last().unwrap(),&fresh,6));
    }
}

#[test]
fn a_kept_prompt_state_gives_the_bits_of_a_fresh_read_on_the_grouped_prompt_path() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);let _=&c.operands;
    let decoder=Decoder::new(&c.admission,Host,80).unwrap();
    let prompt=[1u32,3,2,7,5,11,4,9,6,13,8,2,10];
    // Turn 1 as the server reads it: prompt groups (here 8 rows, then 5), the head on the last row only.
    let zero=decoder.zero_state().unwrap();
    let (_,middle)=decoder.prefill_rows(&prompt[..8],&zero,HeadRows::Last).unwrap();
    let (last,kept)=decoder.prefill_rows(&prompt[8..],&middle,HeadRows::Last).unwrap();
    let answer=decode(&decoder,last.last().unwrap(),&kept,9);
    for extra in [vec![12u32,1,14,3,5,7,2],vec![4u32,4,9,1]] {
        assert_ne!(extra[0],answer[0].0);
        // Turn 2 reads only the new rows, in one group, from the kept state.
        let (reused_rows,reused)=decoder.prefill_rows(&extra,&kept,HeadRows::All).unwrap();
        // A fresh read of the whole prompt, token by token: the reference every group must equal.
        let whole=[prompt.as_slice(),extra.as_slice()].concat();
        let (fresh_rows,fresh)=scalar(&decoder,&whole,None);
        assert_eq!(reused.position,fresh.position);
        for (i,row) in reused_rows.iter().enumerate() {assert_eq!(bits(row),bits(&fresh_rows[prompt.len()+i]),"row {i} after the kept prompt");}
        // A fresh read in groups cut at other places gives the same last row too.
        let (_,a)=decoder.prefill_rows(&whole[..5],&zero,HeadRows::Last).unwrap();
        let (grouped,b)=decoder.prefill_rows(&whole[5..],&a,HeadRows::Last).unwrap();
        assert_eq!(bits(grouped.last().unwrap()),bits(reused_rows.last().unwrap()));
        assert_eq!(decode(&decoder,reused_rows.last().unwrap(),&reused,6),decode(&decoder,grouped.last().unwrap(),&b,6));
        assert_eq!(decode(&decoder,reused_rows.last().unwrap(),&reused,6),decode(&decoder,fresh_rows.last().unwrap(),&fresh,6));
    }
    assert_eq!(kept.position,prompt.len() as u64);
}
