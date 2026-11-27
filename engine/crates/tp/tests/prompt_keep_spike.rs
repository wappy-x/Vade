//! SPIKE (design proof, host twins, one chat): the state a prompt read leaves can be KEPT and
//! read again later. A decode step and a prompt group take the old state by reference and
//! return a new one, so the kept state needs no copy. A second turn that starts from the kept
//! state gives the same logit bits and the same IDs as a fresh read of the whole prompt.
use engine_model::safetensors::Dtype;
use spark_tp::bounded_decoder::{Decoder,HeadRows,Host,State};
#[path="support/attention_checkpoint.rs"] mod fixture;
#[path="support/watch_compute.rs"] mod watch;

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

/// Step 3 (memory): a parked prompt holds no expanded rows, and a resume builds the same words again, as many rows a
/// layer as the prompt held. Checked word for word, with every prompt row held and with a prompt that passed the
/// held limit (only its first group's rows are held). A turn from the resumed state has the bits of a turn from the
/// state that was never parked, and of a fresh read.
#[test]
fn a_parked_prompt_drops_its_expanded_rows_and_a_resume_builds_the_same_words() {
    use spark_tp::bounded_decoder::{park_state,Expanded};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);let _=&c.operands;
    let prompt=[1u32,3,2,7,5,11,4,9,6,13,8,2,10];let extra=[12u32,1,14,3,5,7,2];
    for (limit,held) in [(spark_tp::bounded_decoder::RESERVED_ROWS,13),(8,8)] {
        let decoder=Decoder::new(&c.admission,Host,80).unwrap();decoder.set_expanded_limit(limit);
        let zero=decoder.zero_state().unwrap();
        let (_,middle)=decoder.prefill_rows(&prompt[..8],&zero,HeadRows::Last).unwrap();
        let (_,whole)=decoder.prefill_rows(&prompt[8..],&middle,HeadRows::Last).unwrap();
        assert!(!whole.expanded.is_empty() && whole.expanded.values().all(|rows|rows.1==held),"limit {limit}: the read holds {held} rows a layer");
        let parked=park_state(&Host,&whole).unwrap();
        assert!(parked.expanded.is_empty(),"a parked prompt holds no expanded rows");assert_eq!((parked.position,parked.held_rows()),(13,held));
        let resumed=decoder.resume_state(&parked).unwrap();assert_eq!(resumed.position,13);
        assert_eq!(resumed.expanded.keys().collect::<Vec<_>>(),whole.expanded.keys().collect::<Vec<_>>());
        for (layer,Expanded(rows,count,layout)) in &whole.expanded {
            let Expanded(again,again_count,again_layout)=&resumed.expanded[layer];
            assert!(again_count==count && again_layout==layout,"layer {layer}");
            assert!(!rows.is_empty());assert_eq!(bits(again),bits(rows),"layer {layer}: the rebuilt words are the dropped words");
        }
        // The same resume twice: a parked prompt is only read.
        let twice=decoder.resume_state(&parked).unwrap();
        for (layer,Expanded(rows,_,_)) in &resumed.expanded {assert_eq!(bits(&twice.expanded[layer].0),bits(rows));}
        let (resumed_rows,after)=decoder.prefill_rows(&extra,&resumed,HeadRows::All).unwrap();
        let (kept_rows,_)=decoder.prefill_rows(&extra,&whole,HeadRows::All).unwrap();
        let (fresh_rows,fresh)=scalar(&decoder,&[prompt.as_slice(),extra.as_slice()].concat(),None);
        for (i,row) in resumed_rows.iter().enumerate() {assert_eq!(bits(row),bits(&kept_rows[i]));assert_eq!(bits(row),bits(&fresh_rows[prompt.len()+i]),"limit {limit} row {i}");}
        assert_eq!(decode(&decoder,resumed_rows.last().unwrap(),&after,6),decode(&decoder,fresh_rows.last().unwrap(),&fresh,6));
    }
    // A state read by plain steps holds no expanded rows: parking and resuming it changes nothing.
    let decoder=Decoder::new(&c.admission,Host,80).unwrap();let (rows,stepped)=scalar(&decoder,&prompt,None);
    let parked=park_state(&Host,&stepped).unwrap();assert_eq!(parked.held_rows(),0);let resumed=decoder.resume_state(&parked).unwrap();assert!(resumed.expanded.is_empty());
    assert_eq!(decode(&decoder,rows.last().unwrap(),&resumed,6),decode(&decoder,rows.last().unwrap(),&stepped,6));
}

/// Step 3 (memory), a rebuild of MORE than one prompt group: a prompt that held 520 expanded rows a layer is parked
/// and resumed, so each layer is built again in three groups (256, 256, 8 rows). The words equal the dropped ones and
/// a turn from the resumed state has the bits of a turn from the state that was never parked.
/// The watched compute also shows what a device needs of the rebuild (review fault: it held every group's dropped
/// buffers to its end, captured graph pieces, and its last group allocated the layer a second time):
/// - launch recording is paused before the first row is expanded and resumed at the end, also when the rebuild fails;
/// - after EACH group comes a drain (`rows_keep`), so one group's scratch is held at a time;
/// - a layer's first group asks room for all the rows that follow, and every later group hands the rows built so far
///   back in, so a device appends where they are (`held::extend`; its own test is in held.rs).
#[test]
fn a_rebuild_of_three_groups_is_exact_and_holds_one_group_of_scratch_with_recording_paused() {
    use spark_tp::bounded_decoder::{park_state,Expanded,PROMPT_GROUP};
    use watch::{Event,Watch};
    let rows=2*PROMPT_GROUP+8;let capacity=rows+16;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,capacity as u64);let _=&c.operands;
    let prompt=(0..rows).map(|i|((i*7+3)%13+1) as u32).collect::<Vec<_>>();let extra=[12u32,1,14,3,5];
    let compute=Watch::default();let decoder=Decoder::new(&c.admission,&compute,capacity as u64).unwrap();
    let mut whole=decoder.zero_state().unwrap();
    for group in prompt.chunks(PROMPT_GROUP) {whole=decoder.prefill_rows(group,&whole,HeadRows::Last).unwrap().1;}
    assert!(!whole.expanded.is_empty() && whole.expanded.values().all(|held|held.1==rows),"the read holds {rows} rows a layer");
    let layers=whole.expanded.len();let parked=park_state(&&compute,&whole).unwrap();assert_eq!(parked.held_rows(),rows);
    compute.take();
    let resumed=decoder.resume_state(&parked).unwrap();let events=compute.take();
    for (layer,Expanded(kept,count,layout)) in &whole.expanded {
        let Expanded(again,again_count,again_layout)=&resumed.expanded[layer];
        assert!(again_count==count && again_layout==layout,"layer {layer}");assert_eq!(bits(again),bits(kept),"layer {layer}: the rebuilt words are the dropped words");
    }
    // Recording: paused first, resumed last, nothing built outside.
    assert_eq!(events.first(),Some(&Event::Pause(true)),"{events:?}");assert_eq!(events.last(),Some(&Event::Pause(false)),"{events:?}");
    assert_eq!(events.iter().filter(|event|matches!(event,Event::Pause(_))).count(),2,"{events:?}");
    // Each layer: three groups, a drain after each, the room asked once.
    let work=&events[1..events.len()-1];assert_eq!(work.len(),layers*6,"{events:?}");
    for layer in work.chunks(6) {
        let Event::Extend {held:false,rows:0,words,room}=layer[0] else {panic!("a layer's first group starts its storage: {layer:?}")};
        let width=words/PROMPT_GROUP;assert_eq!(words,PROMPT_GROUP*width);
        assert_eq!(room,(rows-PROMPT_GROUP)*width,"the first group asks room for every row that follows");
        assert_eq!(layer[2],Event::Extend {held:true,rows:PROMPT_GROUP,words:PROMPT_GROUP*width,room:8*width});
        assert_eq!(layer[4],Event::Extend {held:true,rows:2*PROMPT_GROUP,words:8*width,room:0});
        for drain in [1,3,5] {assert_eq!(layer[drain],Event::Keep,"a drain follows each group: {layer:?}");}
    }
    // A turn from the resumed state against the same turn from the state that was never parked.
    let (resumed_rows,after)=decoder.prefill_rows(&extra,&resumed,HeadRows::All).unwrap();
    let (kept_rows,kept_after)=decoder.prefill_rows(&extra,&whole,HeadRows::All).unwrap();
    for (i,row) in resumed_rows.iter().enumerate() {assert_eq!(bits(row),bits(&kept_rows[i]),"row {i} after the resumed prompt");}
    let step=|state:&State<Vec<f32>>,logits:&[f32]|bits(&decoder.step(best(logits),state,|_,_|Ok(())).unwrap().0);
    assert_eq!(step(&after,resumed_rows.last().unwrap()),step(&kept_after,kept_rows.last().unwrap()));
    // A rebuild that fails at a layer's second group, with one group built: the error comes back, recording is
    // resumed, the parked prompt is only read, and the next resume builds every row again.
    compute.take();compute.fail_extend.set(Some(compute.extends()+1));
    let error=decoder.resume_state(&parked).err().expect("the watched append fails");assert_eq!(error,"watched device refused the expanded rows");
    let events=compute.take();assert_eq!((events.first(),events.last()),(Some(&Event::Pause(true)),Some(&Event::Pause(false))),"{events:?}");
    assert_eq!(events.iter().filter(|event|matches!(event,Event::Extend {..})).count(),2,"one group was built before the failure: {events:?}");
    assert_eq!(events.iter().filter(|event|matches!(event,Event::Keep)).count(),2,"each attempted group drains, including the failed append: {events:?}");
    compute.fail_extend.set(None);let again=decoder.resume_state(&parked).unwrap();
    for (layer,Expanded(kept,_,_)) in &whole.expanded {assert_eq!(bits(&again.expanded[layer].0),bits(kept),"layer {layer} after a failed rebuild");}
}
