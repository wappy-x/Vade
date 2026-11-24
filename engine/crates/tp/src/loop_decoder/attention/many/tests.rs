//! Tests of the attention half for many rows: a chat read in steps of many rows against the same chat read one
//! row a step, on the two made-up checkpoints.
use super::*;
use crate::{bounded_decoder::Host,loop_decoder::{attention::half,fixture::{Fixture,made},mix::{Inputs,Run},rows::{ChatRows,Limits,Rows},state::{self,Layout,Ring,State,commit,discard,row_rings}}};
use engine_model::plan::description::{LoopTextDescription,Visit};

fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
type Span=ChatRows;
/// The byte limit of the tests that do not look at it: one group a chat.
const WHOLE:usize=usize::MAX;
/// The visit of stored layer `layer` and its layout for steps of up to `limit` rows, with groups under `key_bytes`.
fn place_with(d:&LoopTextDescription,layer:usize,limit:usize,key_bytes:usize)->(Visit,Layout) {
    let visit=*d.visits.iter().find(|visit|visit.layer==layer as u64).unwrap();(visit,Layout::for_steps(d,layer,16,Limits {step_rows:limit,key_bytes}).unwrap())
}
fn place(d:&LoopTextDescription,layer:usize,limit:usize)->(Visit,Layout) {place_with(d,layer,limit,WHOLE)}
/// The inputs of one position of one chat: the stream that the visit read, the three inputs of the norms, and the
/// input of the convolution after the experts (the other half makes that one; here it is made up).
struct Row {keep:Vec<f32>,inputs:[Vec<f32>;3],after:Vec<f32>}
fn row(chat:usize,position:u64,width:usize)->Row {
    let name=|what:&str|format!("{what} {chat} {position}");
    Row {keep:made(&name("keep"),width,8,-2,1),inputs:[made(&name("q"),width,24,-2,1),made(&name("k"),width,24,-2,1),made(&name("v"),width,24,-2,1)],after:made(&name("after"),width,24,-2,1)}
}
fn inputs_of(rows:&[Row],each:bool)->Inputs<Vec<f32>> {
    let all=|n:usize|rows.iter().flat_map(|row|row.inputs[n].clone()).collect::<Vec<_>>();
    if each {Inputs::Each([all(0),all(1),all(2)])} else {Inputs::Same(all(0))}
}
/// The words of one position: the residual of the attention half, and the finished key and value of the position.
#[derive(Debug,PartialEq,Eq)]
struct Words {residual:Vec<u32>,key:Vec<u32>,value:Vec<u32>}
fn kept(layout:&Layout,state:&State<Vec<f32>>,position:u64)->(Vec<u32>,Vec<u32>) {
    let (at,words)=(layout.slot(position).unwrap()*layout.row_words,layout.row_words);
    (bits(&state.visits[0].keys[at..at+words]),bits(&state.visits[0].values[at..at+words]))
}
/// One chat of `count` positions, one row a step, on a state for steps of one row. Gives the words of each position
/// and the rings after the last one.
fn one_by_one(fixture:&Fixture,layer:usize,chat:usize,count:u64,each:bool)->(Vec<Words>,[Vec<u32>;4]) {
    let d=&fixture.description;let run=Run::new(&Host,fixture,d).unwrap();let (visit,layout)=place(d,layer,1);let width=d.hidden as usize;
    let mut state=state::clean(&Host,&[layout]).unwrap();let mut out=Vec::new();
    for position in 0..count {
        let r=row(chat,position,width);
        let residual=half(&run,&visit,&layout,&mut state.visits[0],position,&r.keep,&inputs_of(std::slice::from_ref(&r),each)).unwrap();
        state::write_ring(&Host,&layout,Ring::AfterExperts,&mut state.visits[0].rings,position,&r.after).unwrap();
        let (key,value)=kept(&layout,&state,position);out.push(Words {residual:bits(&residual),key,value});
    }
    (out,std::array::from_fn(|ring|bits(&state.visits[0].rings[ring])))
}
/// One step of the rows `spans` of the chats `chats` (their numbers), on their states. Gives the residual of each row.
#[allow(clippy::too_many_arguments)]
fn step(fixture:&Fixture,layer:usize,limit:usize,chats:&[usize],states:&mut [State<Vec<f32>>],spans:&[Span],each:bool,key_bytes:usize)->crate::Result<Vec<Vec<u32>>> {
    let d=&fixture.description;let run=Run::new(&Host,fixture,d)?;let (visit,layout)=place_with(d,layer,limit,key_bytes);let width=d.hidden as usize;
    let list=Rows::new(spans.to_vec())?;
    let rows=chats.iter().zip(spans).flat_map(|(&chat,span)|(span.first..).take(span.count).map(move |position|row(chat,position,width))).collect::<Vec<_>>();
    let keep=rows.iter().flat_map(|row|row.keep.clone()).collect::<Vec<_>>();
    let after=rows.iter().flat_map(|row|row.after.clone()).collect::<Vec<_>>();
    let residual={
        let mut visits=states.iter_mut().map(|state|&mut state.visits[0]).collect::<Vec<_>>();
        let residual=half_rows(&run,&visit,&layout,&list,&mut visits,&keep,&inputs_of(&rows,each))?;
        row_rings(&Host,&layout,Ring::AfterExperts,&list,&mut visits,&after)?;residual
    };
    commit(&mut states.iter_mut().collect::<Vec<_>>(),&list)?;
    Ok(residual.chunks(width).map(bits).collect())
}
#[test]
fn a_chat_in_steps_of_many_rows_equals_the_chat_one_row_a_step() {
    let mut compared=0usize;
    for fixture in [Fixture::small(),Fixture::wide()] {
        // Layer 0 sees a window and has one norm. Layer 1 sees every token, has three norms and makes taps from the token.
        for layer in [0,1] {
            for each in [false,true] {
                let (want,rings)=one_by_one(&fixture,layer,0,12,each);
                // Step row limits. The windows are 4 and 3 rows, so a ring goes round inside a step and between steps.
                // The byte limit 1 gives groups of one row; the large one gives one group a chat.
                for (limit,key_bytes) in [(1,usize::MAX),(2,usize::MAX),(3,1),(5,usize::MAX),(12,1),(12,usize::MAX)] {
                    let (_,layout)=place(&fixture.description,layer,limit);
                    let mut states=vec![state::clean(&Host,&[layout]).unwrap()];let mut position=0u64;
                    while position<12 {
                        let count=limit.min((12-position) as usize);
                        let residual=step(&fixture,layer,limit,&[0],&mut states,&[Span {first:position,count}],each,key_bytes).unwrap();
                        for (row,residual) in residual.into_iter().enumerate() {
                            // The state keeps the window ring: a row more than a window before the step's end is gone
                            // from it (the later rows of the step read it, so the residuals still hold it).
                            let at=position+row as u64;
                            let (key,value)=if at+layout.window as u64>=position+count as u64 {kept(&layout,&states[0],at)} else {(want[at as usize].key.clone(),want[at as usize].value.clone())};
                            assert_eq!(Words {residual,key,value},want[at as usize],"layer {layer}, limit {limit}, position {at}");compared+=1;
                        }
                        position+=count as u64;assert_eq!(states[0].position,position);
                    }
                    assert_eq!(std::array::from_fn::<_,4,_>(|ring|bits(&states[0].visits[0].rings[ring])),rings,"the rings after the chat, limit {limit}");
                }
            }
        }
    }
    println!("positions of a chat in steps of many rows, each equal to its step of one row in every word: {compared}");
}
#[test]
fn chats_in_one_step_equal_each_chat_alone() {
    for fixture in [Fixture::small(),Fixture::wide()] {
        for layer in [0,1] {
            // Three chats of 11, 4 and 7 positions. Each step takes what is left of each chat, up to 3, 1 and 2 rows.
            let lengths=[11u64,4,7];let take=[3usize,1,2];let limit=3;
            let want=[0,1,2].map(|chat|one_by_one(&fixture,layer,chat,lengths[chat],true));
            let (_,layout)=place(&fixture.description,layer,limit);
            let mut states=(0..3).map(|_|state::clean(&Host,&[layout]).unwrap()).collect::<Vec<_>>();
            loop {
                // The chats that still have rows, in the order 2, 0, 1: the step order is not the chat number.
                let turn=[2usize,0,1].into_iter().filter(|&chat|states[chat].position<lengths[chat]).collect::<Vec<_>>();
                if turn.is_empty() {break;}
                let spans=turn.iter().map(|&chat|Span {first:states[chat].position,count:take[chat].min((lengths[chat]-states[chat].position) as usize)}).collect::<Vec<_>>();
                // The states of the step, in step order, taken out of the list and put back after the step.
                let mut held=turn.iter().map(|&chat|std::mem::replace(&mut states[chat],state::clean(&Host,&[layout]).unwrap())).collect::<Vec<_>>();
                let residual=step(&fixture,layer,limit,&turn,&mut held,&spans,true,usize::MAX).unwrap();
                let mut next=residual.into_iter();
                for ((&chat,span),state) in turn.iter().zip(&spans).zip(held) {
                    for at in (span.first..).take(span.count) {
                        let (key,value)=kept(&layout,&state,at);
                        assert_eq!(Words {residual:next.next().unwrap(),key,value},want[chat].0[at as usize],"layer {layer}, chat {chat}, position {at}");
                    }
                    states[chat]=state;
                }
            }
            for chat in 0..3 {
                assert_eq!(states[chat].position,lengths[chat]);
                assert_eq!(std::array::from_fn::<_,4,_>(|ring|bits(&states[chat].visits[0].rings[ring])),want[chat].1,"the rings of chat {chat}");
            }
        }
    }
}
#[test]
fn a_step_of_rows_that_fails_runs_again_with_the_same_words() {
    let fixture=Fixture::small();let d=&fixture.description;let width=d.hidden as usize;
    for layer in [0,1] {
        let (want,rings)=one_by_one(&fixture,layer,0,9,false);let (visit,layout)=place(d,layer,4);
        let mut states=vec![state::clean(&Host,&[layout]).unwrap()];
        step(&fixture,layer,4,&[0],&mut states,&[Span {first:0,count:4}],false,usize::MAX).unwrap();
        // The step of positions 4 to 7 runs its attention half and is then dropped: no commit. The planes hold its
        // rows, the rings and the position are those from before it.
        let run=Run::new(&Host,&fixture,d).unwrap();let spans=[Span {first:4,count:4}];
        let rows=(4..8).map(|position|row(0,position,width)).collect::<Vec<_>>();
        let keep=rows.iter().flat_map(|row|row.keep.clone()).collect::<Vec<_>>();
        let list=Rows::new(spans.to_vec()).unwrap();
        {let mut visits=vec![&mut states[0].visits[0]];half_rows(&run,&visit,&layout,&list,&mut visits,&keep,&inputs_of(&rows,false)).unwrap();}
        assert!(commit(&mut states.iter_mut().collect::<Vec<_>>(),&list).is_err(),"the ring after the experts has no rows yet");
        discard(&mut states.iter_mut().collect::<Vec<_>>());assert_eq!(states[0].position,4);
        // The same rows again, in another split: 3 rows, then 2.
        let mut position=4u64;
        for count in [3usize,2] {
            let residual=step(&fixture,layer,4,&[0],&mut states,&[Span {first:position,count}],false,usize::MAX).unwrap();
            for (row,residual) in residual.into_iter().enumerate() {
                let at=position+row as u64;let (key,value)=kept(&layout,&states[0],at);
                assert_eq!(Words {residual,key,value},want[at as usize],"layer {layer}, position {at} after a dropped step");
            }
            position+=count as u64;
        }
        assert_eq!(std::array::from_fn::<_,4,_>(|ring|bits(&states[0].visits[0].rings[ring])),rings);
    }
}
#[test]
fn a_step_of_rows_refuses_what_it_cannot_run() {
    let fixture=Fixture::small();let d=&fixture.description;
    let (_,layout)=place(d,0,2);let mut states=vec![state::clean(&Host,&[layout]).unwrap()];
    // A ring for steps of 2 rows takes 5 rows while it is empty, and 2 rows a step once the chat is past it.
    step(&fixture,0,2,&[0],&mut states,&[Span {first:0,count:5}],false,usize::MAX).unwrap();
    assert!(step(&fixture,0,2,&[0],&mut states,&[Span {first:5,count:3}],false,usize::MAX).unwrap_err().contains("more rows of one chat than its state takes"));
    assert_eq!(states[0].position,5);
    // No chat, a chat with no row, and chats that differ from the states.
    assert!(step(&fixture,0,2,&[],&mut states,&[],false,usize::MAX).is_err());
    assert!(step(&fixture,0,2,&[0],&mut states,&[Span {first:5,count:0}],false,usize::MAX).is_err());
    assert!(step(&fixture,0,2,&[0,1],&mut states,&[Span {first:5,count:1},Span {first:0,count:1}],false,usize::MAX).is_err());
    // A plane of 16 rows takes no row past its end.
    let (_,layout)=place(d,1,4);let mut states=vec![state::clean(&Host,&[layout]).unwrap()];states[0].position=14;
    assert!(step(&fixture,1,4,&[0],&mut states,&[Span {first:14,count:3}],true,usize::MAX).is_err());
}
