//! Tests of the state in a step of many rows: each is a step of many rows against the same rows one a step.
use super::*;
use crate::{bounded_decoder::Host,loop_decoder::{fixture::{Fixture,made},rows::Limits,state::{self,clean,pack,write_ring,write_row}}};

type Span=ChatRows;
fn total(spans:&[Span])->usize {spans.iter().map(|span|span.count).sum()}
fn list(spans:&[Span])->Rows {Rows::new(spans.to_vec()).unwrap()}
/// The layouts that a chat with many rows works in (`Layout::wide`) for steps of up to `limit` rows.
fn layouts_for(d:&engine_model::plan::description::LoopTextDescription,capacity:u64,limit:usize)->crate::Result<Vec<Layout>> {
    Ok(state::layouts_for(d,capacity,Limits {step_rows:limit,key_bytes:1<<20})?.iter().map(Layout::wide).collect())
}

fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
/// What a ring holds after each row of a chat, by the rule of a step of one row: the position that each slot
/// holds, or none where no position was written.
fn walk(ring:&mut [Option<u64>],span:Span)->Vec<Vec<Option<u64>>> {
    (span.first..).take(span.count).map(|position|{let taps=ring.len() as u64;ring[(position%taps) as usize]=Some(position);ring.to_vec()}).collect()
}
#[test]
fn the_gather_gives_each_row_the_ring_of_a_step_of_one_row() {
    for taps in [1,2,3,4,7] {
        for spans in [vec![Span {first:0,count:1}],vec![Span {first:0,count:9}],vec![Span {first:5,count:1},Span {first:0,count:6},Span {first:7,count:3}],
            vec![Span {first:100,count:2},Span {first:3,count:11},Span {first:1,count:1},Span {first:16_000_001,count:5}]] {
            let table=gather(&spans,taps).unwrap();let (chats,rows)=(spans.len(),total(&spans));
            assert_eq!((table.rows.len(),table.slots.len(),table.after.len()),(rows*taps,rows,chats*taps));
            // The ring of each chat before the step: the newest positions before its first row.
            let before=spans.iter().map(|span|{let mut ring=vec![None;taps];walk(&mut ring,Span {first:span.first.saturating_sub(taps as u64),count:(span.first.min(taps as u64)) as usize});ring}).collect::<Vec<_>>();
            // What a source row holds: a slot of a ring from before the step, or the position of a row of the step.
            let mut source=before.iter().flatten().copied().collect::<Vec<_>>();
            for span in &spans {source.extend((span.first..).take(span.count).map(Some));}
            let mut offset=0;
            for (chat,span) in spans.iter().enumerate() {
                let mut ring=before[chat].clone();
                for (row,want) in walk(&mut ring,*span).into_iter().enumerate() {
                    let got=table.rows[(offset+row)*taps..(offset+row+1)*taps].iter().map(|&at|source[at]).collect::<Vec<_>>();
                    assert_eq!(got,want,"taps {taps}, chat {chat}, row {row}");
                    assert_eq!(table.slots[offset+row] as u64,(span.first+row as u64)%taps as u64);
                }
                assert_eq!(table.after[chat*taps..(chat+1)*taps].iter().map(|&at|source[at]).collect::<Vec<_>>(),ring,"the ring after the step");
                offset+=span.count;
            }
        }
    }
    assert!(gather(&[],4).is_err() && gather(&[Span {first:0,count:0}],4).is_err() && gather(&[Span {first:0,count:1}],0).is_err());
}
#[test]
fn the_rings_of_the_rows_and_the_ring_after_the_step_equal_steps_of_one_row() {
    let fixture=Fixture::wide();let d=&fixture.description;let l=layouts_for(d,24,6).unwrap();let layout=&l[1];
    for which in Ring::ALL {
        let (channels,taps)=layout.ring_size(which);
        // Three chats. Each has a past of its own, then its rows of the step.
        let spans=[Span {first:5,count:4},Span {first:0,count:1},Span {first:2,count:6}];
        let input=|chat:usize,position:u64|made(&format!("ring {which:?} {chat} {position}"),channels,24,-3,1);
        let (mut one,mut many)=(Vec::new(),Vec::new());
        for (chat,span) in spans.iter().enumerate() {
            let mut state=clean(&Host,std::slice::from_ref(layout)).unwrap();
            for position in 0..span.first {write_ring(&Host,layout,which,&mut state.visits[0].rings,position,&input(chat,position)).unwrap();}
            state.position=span.first;
            let mut copy=clean(&Host,std::slice::from_ref(layout)).unwrap();copy.position=span.first;copy.visits[0].rings[which as usize]=state.visits[0].rings[which as usize].clone();
            one.push(state);many.push(copy);
        }
        // One row a step: the ring that each row reads, and its step word.
        let (mut want_rings,mut want_steps)=(Vec::new(),Vec::new());
        for (chat,span) in spans.iter().enumerate() {for position in (span.first..).take(span.count) {
            want_steps.extend(write_ring(&Host,layout,which,&mut one[chat].visits[0].rings,position,&input(chat,position)).unwrap());
            want_rings.extend_from_slice(&one[chat].visits[0].rings[which as usize]);
        }}
        let inputs=spans.iter().enumerate().flat_map(|(chat,span)|(span.first..).take(span.count).flat_map(move |position|input(chat,position))).collect::<Vec<_>>();
        let before=many.iter().map(|state|state.visits[0].rings[which as usize].clone()).collect::<Vec<_>>();
        let (rings,steps)={let mut visits=many.iter_mut().map(|state|&mut state.visits[0]).collect::<Vec<_>>();row_rings(&Host,layout,which,&list(&spans),&mut visits,&inputs).unwrap()};
        assert_eq!(rings.len(),total(&spans)*taps*channels);
        assert_eq!((bits(&rings),bits(&steps)),(bits(&want_rings),bits(&want_steps)),"{which:?}");
        // Before the commit the state holds the rings and the positions from before the step.
        for (state,old) in many.iter().zip(&before) {assert_eq!(bits(&state.visits[0].rings[which as usize]),bits(old));assert!(state.visits[0].next[which as usize].is_some());}
        // A commit needs every ring of every visit: one ring alone is refused and nothing changes.
        let mut chats=many.iter_mut().collect::<Vec<_>>();
        assert!(commit(&mut chats,&list(&spans)).unwrap_err().contains("left a ring without its rows"));
        assert_eq!(chats.iter().map(|state|state.position).collect::<Vec<_>>(),[5,0,2]);
        for state in chats.iter_mut() {for other in Ring::ALL {if other!=which {let ring=state.visits[0].rings[other as usize].clone();state.visits[0].next[other as usize]=Some(ring);}}}
        assert!(commit(&mut chats,&list(&[spans[0],spans[1]])).is_err() && commit(&mut chats,&list(&[spans[1],spans[0],spans[2]])).is_err());
        commit(&mut chats,&list(&spans)).unwrap();
        for (state,want) in many.iter().zip(&one) {
            assert_eq!(bits(&state.visits[0].rings[which as usize]),bits(&want.visits[0].rings[which as usize]),"the ring after the step");
            assert!(state.visits[0].next.iter().all(Option::is_none));
        }
        assert_eq!(many.iter().map(|state|state.position).collect::<Vec<_>>(),[9,1,8]);
    }
}
#[test]
fn a_step_that_is_dropped_leaves_the_rings_and_the_positions() {
    let fixture=Fixture::small();let l=layouts_for(&fixture.description,16,4).unwrap();let layout=&l[0];
    let mut state=clean(&Host,std::slice::from_ref(layout)).unwrap();
    let spans=[Span {first:0,count:3}];let inputs=made("dropped",3*layout.channels,24,-3,1);
    {let mut visits=vec![&mut state.visits[0]];row_rings(&Host,layout,Ring::AfterAttention,&list(&spans),&mut visits,&inputs).unwrap();}
    assert!(state.visits[0].next[Ring::AfterAttention as usize].is_some());
    discard(&mut [&mut state]);
    assert!(state.visits[0].next.iter().all(Option::is_none) && state.position==0);
    assert!(state.visits[0].rings.iter().all(|ring|ring.iter().all(|v|v.to_bits()==0)));
    // Inputs or rings of another size are refused.
    let mut visits=vec![&mut state.visits[0]];
    assert!(row_rings(&Host,layout,Ring::AfterAttention,&list(&spans),&mut visits,&inputs[1..].to_vec()).is_err());
    assert!(row_rings(&Host,layout,Ring::AfterAttention,&list(&[spans[0],spans[0]]),&mut visits,&inputs).is_err());
}
#[test]
fn the_rows_of_a_chat_go_into_a_plane_as_in_steps_of_one_row() {
    let fixture=Fixture::small();let d=&fixture.description;
    // Step row limits 1, 3 and 6: a ring of the window, of 2 more rows and of 5 more rows.
    for limit in [1usize,3,6] {
        let l=layouts_for(d,16,limit).unwrap();
        for layout in [&l[0],&l[1]] {
            let key=|position:u64|(0..layout.kv_channels as u32).map(|i|f32::from_bits((0x3f00+position as u32*64+i)<<16)).collect::<Vec<_>>();
            let (mut one,mut many)=(clean(&Host,std::slice::from_ref(layout)).unwrap(),clean(&Host,std::slice::from_ref(layout)).unwrap());
            let mut position=0u64;
            // Steps of the limit's rows, and a last step of what is left of 14 positions.
            while position<14 {
                let count=limit.min((14-position) as usize);let span=Span {first:position,count};
                assert!(count<=room(layout,position));
                let rows=(position..).take(count).flat_map(key).collect::<Vec<_>>();
                write_rows(&Host,layout,&mut many.visits[0].keys,span,&pack(&Host,&rows).unwrap()).unwrap();
                for at in (position..).take(count) {write_row(&Host,layout,&mut one.visits[0].keys,at,&key(at)).unwrap();}
                position+=count as u64;
                // Every key that a token at the newest position sees stands at the slot of its position.
                assert_eq!(bits(&many.visits[0].keys),bits(&one.visits[0].keys),"limit {limit}");
                let live=layout.live(position-1).unwrap();
                for seen in (position-live as u64)..position {
                    let slot=layout.slot(seen).unwrap();
                    assert_eq!(bits(&many.visits[0].keys[slot*layout.row_words..(slot+1)*layout.row_words]),bits(&pack(&Host,&key(seen)).unwrap()));
                }
            }
            let rows=layout.rows;
            // More rows than the state takes, and rows of another size, are refused.
            let too_many=room(layout,14)+1;
            let packed=vec![0.0;too_many*layout.row_words];
            if layout.ring {assert!(write_rows(&Host,layout,&mut many.visits[0].keys,Span {first:14,count:too_many},&packed).unwrap_err().contains("more rows of one chat"));}
            else {assert!(write_rows(&Host,layout,&mut many.visits[0].keys,Span {first:14,count:rows-14+1},&vec![0.0;(rows-14+1)*layout.row_words]).is_err());}
            assert!(write_rows(&Host,layout,&mut many.visits[0].keys,Span {first:0,count:1},&packed[1..].to_vec()).is_err());
            assert!(write_rows(&Host,layout,&mut many.visits[0].keys,Span {first:0,count:0},&Vec::new()).is_err());
        }
    }
    // The room: a ring of window 4 for steps of 3 takes 3 rows a step anywhere, and more while it is not yet full.
    let l=layouts_for(d,16,3).unwrap();
    assert_eq!((l[0].rows,room(&l[0],0),room(&l[0],2),room(&l[0],4),room(&l[0],1000)),(6,6,4,3,3));
    assert_eq!((room(&l[1],0),room(&l[1],10),room(&l[1],16),room(&l[1],99)),(16,6,0,0));
}
#[test]
fn the_step_words_of_the_rows_are_those_of_steps_of_one_row() {
    let fixture=Fixture::small();let l=layouts_for(&fixture.description,16,3).unwrap();
    for layout in [&l[0],&l[1]] {
        let (both,live)=key_steps(&Host,layout,3,5).unwrap();
        let want=(3..8).map(|position|(layout.live(position).unwrap() as f32,layout.slot(position).unwrap() as f32)).collect::<Vec<_>>();
        assert_eq!(both,want.iter().flat_map(|&(n,slot)|[n,slot]).collect::<Vec<_>>());
        assert_eq!(live,want.iter().map(|&(n,_)|n).collect::<Vec<_>>());
    }
    assert!(key_steps(&Host,&l[1],14,3).is_err(),"a row past the capacity");
    let _=state::STEP_WORD_LIMIT;
}
#[test]
fn a_chat_with_many_rows_works_in_a_wide_ring_and_the_state_keeps_the_window_ring_of_token_steps() {
    let fixture=Fixture::small();let d=&fixture.description;
    for limit in [2usize,3,6] {
        let l=state::layouts_for(d,16,Limits {step_rows:limit,key_bytes:1<<20}).unwrap();let (layout,wide)=(&l[0],l[0].wide());
        assert!(layout.ring && layout.rows==layout.window && wide.rows==layout.window+limit-1);
        let key=|position:u64|(0..layout.kv_channels as u32).map(|i|f32::from_bits((0x3f00+position as u32*64+i)<<16)).collect::<Vec<_>>();
        let (mut one,mut many)=(clean(&Host,std::slice::from_ref(layout)).unwrap(),clean(&Host,std::slice::from_ref(layout)).unwrap());
        let mut position=0u64;
        while position<14 {
            let count=limit.min((14-position) as usize);let span=Span {first:position,count};
            let mut plane=super::widen(&Host,layout,&many.visits[0].keys,position).unwrap();
            let rows=(position..).take(count).flat_map(key).collect::<Vec<_>>();
            write_rows(&Host,&wide,&mut plane,span,&pack(&Host,&rows).unwrap()).unwrap();
            // Every row of the step sees its keys at their wide slots.
            for at in (position..).take(count) {
                for seen in (at+1-wide.live(at).unwrap() as u64)..=at {
                    let slot=wide.slot(seen).unwrap();
                    assert_eq!(bits(&plane[slot*wide.row_words..(slot+1)*wide.row_words]),bits(&pack(&Host,&key(seen)).unwrap()),"limit {limit} row {at} key {seen}");
                }
            }
            many.visits[0].keys=super::narrow(&Host,layout,&plane,position+count as u64).unwrap();
            for at in (position..).take(count) {write_row(&Host,layout,&mut one.visits[0].keys,at,&key(at)).unwrap();}
            position+=count as u64;
            assert_eq!(bits(&many.visits[0].keys),bits(&one.visits[0].keys),"limit {limit} position {position}");
        }
        // A plane of another size is refused.
        assert!(super::widen(&Host,layout,&vec![0.0;1],4).is_err() && super::narrow(&Host,layout,&vec![0.0;1],4).is_err());
    }
    assert_eq!(super::moves(5,12,4,6),[(1,5,1),(2,0,2),(0,2,4)]);
}
#[test]
fn a_cohort_is_cut_into_the_fewest_direct_parts_of_even_size() {
    assert_eq!((direct_parts(1),direct_parts(2),direct_parts(16)),(vec![(0,1)],vec![(0,2)],vec![(0,16)]));
    assert_eq!((direct_parts(17),direct_parts(20),direct_parts(32),direct_parts(33)),(vec![(0,9),(9,8)],vec![(0,10),(10,10)],vec![(0,16),(16,16)],vec![(0,11),(11,11),(22,11)]));
    let most=spark_gpu_probe::fast_ops::DIRECT_CHATS_MAX;
    for rows in 2..=300 {
        let parts=direct_parts(rows);let mut next=0;
        for &(start,n) in &parts {assert!(start==next && (2..=most).contains(&n),"{rows} chats");next+=n;}
        let sizes=parts.iter().map(|part|part.1);
        assert!(next==rows && parts.len()==rows.div_ceil(most) && sizes.clone().max().unwrap()-sizes.min().unwrap()<=1,"{rows} chats");
    }
}
/// A decode cohort above the limit of one direct launch (`DIRECT_CHATS_MAX`) runs the direct convolution in parts.
/// The words of the output and the ring of each chat after the step are those of one launch a row.
#[test]
fn a_cohort_above_the_direct_limit_runs_the_direct_convolution_in_parts() {
    use crate::loop_decoder::attention::hand::{Hand,Mode};
    use spark_gpu_probe::block_ops::{Op,Shape};
    for fixture in [Fixture::small(),Fixture::wide()] {
        let l=layouts_for(&fixture.description,24,1).unwrap();let layout=&l[0];
        // The forms of the convolution in a decode step: the taps that the token makes (the key), the fixed taps
        // alone (a boundary), the addend (after attention) and the addend with a rounded sum (after the experts).
        for (which,cols) in [(Ring::Key,1u32),(Ring::Value,0),(Ring::AfterAttention,2),(Ring::AfterExperts,6)] {
            let (channels,taps)=layout.ring_size(which);let heads=if cols&1!=0 && channels.is_multiple_of(2) {2} else {1};
            let shape=Shape {op:Op::ShortConv,heads:heads as u32,dim:(channels/heads) as u32,taps:taps as u32,rows:0,cols,epsilon:0.0,bound:0.5};
            let fixed=made(&format!("direct taps {which:?}"),channels*taps,24,-3,0);
            for rows in [2usize,3,16,17,20,33] {
                // Each chat has a position of its own, so the rows of a part have different ring slots.
                let step=list(&(0..rows).map(|chat|Span {first:(chat as u64*5+3)%11,count:1}).collect::<Vec<_>>());
                let inputs=made(&format!("direct inputs {which:?} {rows}"),rows*channels,24,-3,1);
                let token=(cols&1!=0).then(||made(&format!("direct token {which:?} {rows}"),rows*heads*taps,24,-3,0));
                let addend=(cols&2!=0).then(||made(&format!("direct addend {which:?} {rows}"),rows*channels,24,-3,1));
                // The output, the pending ring of each chat, and the ring of each chat (a step changes no ring).
                let run=|hand:&Hand|{
                    let mut states=(0..rows).map(|chat|{
                        let mut state=clean(hand,std::slice::from_ref(layout)).unwrap();
                        state.visits[0].rings[which as usize]=made(&format!("direct ring {which:?} {chat}"),taps*channels,24,-3,1);state
                    }).collect::<Vec<_>>();
                    let before=states.iter().map(|state|bits(&state.visits[0].rings[which as usize])).collect::<Vec<_>>();
                    let output={
                        let mut visits=states.iter_mut().map(|state|&mut state.visits[0]).collect::<Vec<_>>();
                        convolve(hand,layout,which,&step,&mut visits,shape,&inputs,token.as_ref(),addend.as_ref(),&fixed).unwrap()
                    };
                    assert_eq!(states.iter().map(|state|bits(&state.visits[0].rings[which as usize])).collect::<Vec<_>>(),before);
                    (bits(&output),states.iter().map(|state|bits(state.visits[0].next[which as usize].as_ref().unwrap())).collect::<Vec<_>>())
                };
                // One launch a row: the hand arm, and the arm of the host.
                let rowwise=Hand::new(Mode::All);let want=run(&rowwise);
                assert_eq!((rowwise.count(Op::ShortConv),rowwise.direct.borrow().clone()),(rows,None));
                assert_eq!(run(&Hand::new(Mode::Refused)),want,"{which:?} {rows} chats: the host arm");
                // The direct form: parts of at most 16 chats, in row order, and no launch a row.
                let direct=Hand::new(Mode::All);*direct.direct.borrow_mut()=Some(Vec::new());
                assert_eq!(run(&direct),want,"{which:?} {rows} chats: the direct parts");
                let sizes=direct_parts(rows).iter().map(|part|part.1).collect::<Vec<_>>();
                assert_eq!((direct.launches(),direct.direct.borrow().clone()),(0,Some(sizes.clone())));
                // A refused part: the whole step runs one launch a row, with the same words and the same rings.
                for refused in 1..=sizes.len() {
                    let part=Hand::new(Mode::All);*part.direct.borrow_mut()=Some(Vec::new());part.direct_refused.set(refused);
                    assert_eq!(run(&part),want,"{which:?} {rows} chats: part {refused} refused");
                    assert_eq!((part.count(Op::ShortConv),part.direct.borrow().clone()),(rows,Some(sizes[..refused].to_vec())));
                }
            }
        }
    }
}
