//! Tests of the token step and of the whole driver on the two made-up checkpoints of `fixture.rs`.
//!
//! Three runs of one chat are compared word for word:
//! - the driver on the host backend: every launch runs on the shared arm, the code that the GPU also runs.
//!   This is the host run of the driver, the byte reference of its device run.
//! - the whole-model host form of the model crate (`engine_model::loop_host`), which shares no code with the driver.
//! - the driver on `Hand` in mode `All`: every block arm of the attention half and of the token step runs on this
//!   seat's hand form of its contract. These are the hand values.
use super::{*,attention::hand::{Hand,Mode,bits,softcap_is_ruled,unpacked},fixture::Fixture};
use crate::bounded_decoder::Host;
use engine_model::{loop_host::{Chat,Head,Limits,Model,TokenOut,Want},short_conv::Slot};
use spark_gpu_probe::{causal_ops::Shape,fixed_math::rn};

/// The words that one visit keeps: the key plane, the value plane, and the four convolution rings.
type Kept=[Vec<f32>;6];
/// What one token of a driver run gave: the normed embedding, the stream after each visit in run order, the capped
/// logits, the top row, and a copy of the whole state after the token, by state slot.
#[derive(Debug)]
struct Seen {embedding:Vec<f32>,streams:Vec<Vec<f32>>,logits:Vec<f32>,top:u32,state:Vec<Kept>}
fn copy(state:&State<Vec<f32>>)->Vec<Kept> {
    state.visits.iter().map(|v|[v.keys.clone(),v.values.clone(),v.rings[0].clone(),v.rings[1].clone(),v.rings[2].clone(),v.rings[3].clone()]).collect()
}
fn words(seen:&[Seen])->Vec<u32> {
    seen.iter().flat_map(|token|token.streams.iter().chain([&token.embedding,&token.logits]).chain(token.state.iter().flatten()).flat_map(|v|bits(v)).chain([token.top]).collect::<Vec<_>>()).collect()
}
/// One token of a chat, with everything a test looks at.
fn token<C:Compute<Value=Vec<f32>>,W:Weights<C>>(driver:&Driver<'_,C,W>,state:&mut State<Vec<f32>>,id:u32)->Result<Seen> {
    let (embedding,mut streams)=(driver.embedding(id)?,Vec::new());
    let step=driver.step_seen(state,id,|place,stream|{assert_eq!(place,streams.len(),"the visits are seen in run order");streams.push(stream.clone());Ok(())})?;
    Ok(Seen {embedding,streams,logits:step.logits,top:step.top,state:copy(state)})
}
/// Run `tokens` as one chat from a clean state.
fn run<C:Compute<Value=Vec<f32>>>(compute:&C,fixture:&Fixture,tokens:&[u32],capacity:u64)->Result<Vec<Seen>> {
    let driver=Driver::new(compute,fixture,&fixture.description,capacity)?;let mut state=driver.clean()?;
    tokens.iter().map(|&id|token(&driver,&mut state,id)).collect()
}
/// The same chat on the whole-model host form of the model crate, with the values inside each visit.
fn host_form(fixture:&Fixture,tokens:&[u32])->(Vec<TokenOut>,Chat) {
    let model=Model::new(&fixture.description,&fixture.index,Box::new(fixture.memory()),Limits {expert_cache_bytes:usize::MAX}).unwrap();
    let runner=model.runner().unwrap();let mut chat=runner.chat().unwrap();
    let outs=runner.read(&mut chat,tokens,Want {head:Head::Every,records:true},1).unwrap();(outs,chat)
}
/// The block arms of the driver that the host backend cannot run in their ruled form, by a launch of the smallest
/// shape of each. `CappedGate` would be in the list if the host arm took no weight of two reciprocals, and `Softcap`
/// if the host arm divided by the cap.
fn missing()->Vec<Op> {
    let shape=|op,heads,dim,taps,rows,cols,epsilon,bound|Shape {op,heads,dim,taps,rows,cols,epsilon,bound};
    let (one,two,plane,step,slot)=(vec![1.0f32],vec![1.0f32;2],vec![f32::from_bits(0x3f80_3f80)],vec![1.0f32,0.0],vec![0.0f32]);
    let probes:[(Shape,[Option<&Vec<f32>>;4],&[f32]);13]=[
        (shape(Op::HeadClip,1,2,0,0,0,0.0,1.0),[Some(&two),None,None,None],&[]),
        (shape(Op::RelBias,1,2,0,0,1,0.0,0.0),[Some(&two),None,None,None],&[1.0,1.0]),
        (shape(Op::ShortConv,1,2,1,0,0,0.0,0.5),[Some(&two),None,None,Some(&slot)],&[1.0,1.0]),
        (shape(Op::WindowScores,1,2,1,1,1,0.0,0.5),[Some(&two),Some(&plane),Some(&one),Some(&step)],&[]),
        (shape(Op::SinkSoftmax,1,0,0,1,0,0.0,0.0),[Some(&one),Some(&one),Some(&one),None],&[]),
        (shape(Op::WindowValues,1,2,1,1,0,0.0,0.0),[Some(&one),Some(&plane),Some(&two),Some(&step)],&[]),
        (shape(Op::MixBlend,1,2,0,0,1,0.0,0.5),[Some(&one),Some(&two),None,None],&[]),
        (shape(Op::GateBlend,0,1,0,0,0,0.0,0.0),[Some(&two),Some(&one),Some(&one),None],&[]),
        (shape(Op::CappedGate,0,1,0,1,0,4.0,25.0),[Some(&two),None,None,None],&[0.25,0.04]),
        (shape(Op::RouteFold,0,1,0,1,0,0.0,0.0),[Some(&one),Some(&one),None,None],&[]),
        (shape(Op::Softcap,0,0,0,1,0,0.5,2.0),[Some(&one),None,None,None],&[]),
        (shape(Op::PlainNorm,1,2,0,0,0,1.0e-6,0.0),[Some(&two),None,None,None],&[]),
        (shape(Op::RoundedNorm,1,2,0,0,0,1.0e-6,0.0),[Some(&two),None,None,None],&[]),
    ];
    probes.into_iter().filter(|(s,x,w)|Host.operation(*s,*x,w).is_err() || s.op==Op::Softcap && !softcap_is_ruled()).map(|(s,_,_)|s.op).collect()
}
/// Words of two rows that are equal, and the count of words.
fn equal(a:&[f32],b:&[f32])->(usize,usize) {assert_eq!(a.len(),b.len());(a.iter().zip(b).filter(|(x,y)|x.to_bits()==y.to_bits()).count(),a.len())}
/// The values of plane row `row`, not packed.
fn row_values(layout:&Layout,plane:&[f32],row:usize)->Vec<f32> {
    let words=&plane[row*layout.row_words..(row+1)*layout.row_words];(0..layout.kv_channels).map(|j|unpacked(words,j)).collect()
}
#[test]
fn the_visits_are_those_of_the_description_each_on_its_own_state() {
    for fixture in [Fixture::small(),Fixture::wide()] {
        let d=&fixture.description;let hand=Hand::new(Mode::All);let driver=Driver::new(&hand,&fixture,d,16).unwrap();
        assert_eq!(driver.layouts().len(),d.visits.len());
        let mut state=driver.clean().unwrap();let first=token(&driver,&mut state,fixture.chat(1)[0]).unwrap();
        // One stream for each visit, in run order, each of the width and each BF16: a visit rounds its output once.
        assert_eq!(first.streams.len(),d.visits.len());
        assert!(first.streams.iter().all(|stream|stream.len() as u64==d.hidden && stream.iter().all(|&v|rn(v).to_bits()==v.to_bits())));
        // Each visit ran the attention arms once and its four convolutions, and nothing ran them outside a visit.
        let visits=d.visits.len();
        assert_eq!([Op::HeadClip,Op::RelBias,Op::WindowScores,Op::SinkSoftmax,Op::WindowValues,Op::ShortConv,Op::PackBf16].map(|op|hand.count(op)),[3*visits,visits,visits,visits,visits,4*visits,2*visits]);
        assert_eq!(hand.count(Op::Softcap),1);
        // Each visit wrote the rows of token 0 into the state of its own slot, and no other row of that state.
        for (slot,(kept,layout)) in first.state.iter().zip(driver.layouts()).enumerate() {
            for plane in &kept[..2] {assert!(plane.iter().enumerate().all(|(i,v)|(v.to_bits()!=0)==(i<layout.row_words) || i<layout.row_words),"slot {slot}");assert!(plane[..layout.row_words].iter().any(|v|v.to_bits()!=0));}
            for (ring,which) in kept[2..].iter().zip(Ring::ALL) {let channels=layout.ring_size(which).0;assert!(ring[..channels].iter().any(|v|*v!=0.0) && ring[channels..].iter().all(|v|v.to_bits()==0),"slot {slot} {which:?}");}
        }
        // The list is data. A description with fewer visits runs fewer: here the token stops after the first pass.
        let mut shorter=d.clone();shorter.visits.retain(|visit|visit.pass<2);
        for (place,visit) in shorter.visits.iter_mut().enumerate() {visit.slot=place as u64;}
        let short=Driver::new(&hand,&fixture,&shorter,16).unwrap();let mut state=short.clean().unwrap();
        let cut=token(&short,&mut state,fixture.chat(1)[0]).unwrap();
        assert_eq!((cut.streams.len(),state.visits.len()),(shorter.visits.len(),shorter.visits.len()));assert!(cut.streams.len()<first.streams.len());
        // Up to the end of the first pass the two runs are the same token on the same weights.
        let pass=d.visits.iter().take_while(|visit|visit.pass<2).count();
        for place in 0..pass {assert_eq!(bits(&cut.streams[place]),bits(&first.streams[place]),"visit {place}");}
        // A list that the driver cannot run is refused when the driver is made.
        let mut twice=d.clone();twice.visits[1].slot=0;assert!(Driver::new(&hand,&fixture,&twice,16).is_err());
        let mut outside=d.clone();outside.visits[1].layer=d.layers.len() as u64;assert!(Driver::new(&hand,&fixture,&outside,16).is_err());
        let mut none=d.clone();none.visits.clear();assert!(Driver::new(&hand,&fixture,&none,16).is_err());
    }
}
#[test]
fn a_chat_of_12_tokens_crosses_the_window_edge_and_equals_the_whole_model_host_form() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(12);
        assert!(d.layers.iter().filter_map(|layer|layer.attention.window).all(|window|window<12) && d.conv.kernel<12);
        // The host run of the driver, and the same chat on the whole-model host form.
        let seen=run(&Host,&fixture,&tokens,16).unwrap();let (outs,chat)=host_form(&fixture,&tokens);
        let layouts=state::layouts(d,16).unwrap();
        let mut counts=std::collections::BTreeMap::<&str,(usize,usize)>::new();
        let mut add=|what:&'static str,(same,all):(usize,usize)|{let count=counts.entry(what).or_default();count.0+=same;count.1+=all;};
        for (position,(mine,theirs)) in seen.iter().zip(&outs).enumerate() {
            add("normed embedding",equal(&mine.embedding,&theirs.embedding));
            for (place,visit) in d.visits.iter().enumerate() {
                let (slot,record)=(visit.slot as usize,&theirs.records[place]);let (layout,kept)=(&layouts[slot],&mine.state[slot]);
                // A ring holds the 32-bit product; the host form records the BF16 value that every reader makes of it.
                let ring=|which:Ring|{let channels=layout.ring_size(which).0;let at=layout.ring_slot(which,position as u64)*channels;kept[2+which as usize][at..at+channels].iter().map(|&v|rn(v)).collect::<Vec<_>>()};
                add("attention output",equal(&ring(Ring::AfterAttention),&record.attention));
                add("expert output",equal(&ring(Ring::AfterExperts),&record.experts));
                add("stream after a visit",equal(&mine.streams[place],&theirs.streams[place]));
            }
            add("capped logits",equal(&mine.logits,theirs.logits.as_ref().unwrap()));
            add("top token",(usize::from(Some(mine.top)==theirs.top),1));
            assert!(u64::from(mine.top)<d.head_rows && mine.logits.len() as u64==d.head_rows && mine.logits.iter().all(|v|v.abs()<=d.caps.logits as f32));
        }
        // The state after the last token, visit by visit: the finished keys and values of every token that the host
        // form still keeps, and the inputs that each convolution still reads.
        let last=&seen[seen.len()-1].state;
        for (place,visit) in d.visits.iter().enumerate() {
            let slot=visit.slot as usize;let (layout,kept,theirs)=(&layouts[slot],&last[slot],&chat.visits()[place]);
            let first=tokens.len()-theirs.attention.len();
            let rows=|plane:&[f32]|(first..tokens.len()).flat_map(|position|row_values(layout,plane,layout.slot(position as u64).unwrap())).collect::<Vec<_>>();
            add("kept keys",equal(&rows(&kept[0]),theirs.attention.keys()));add("kept values",equal(&rows(&kept[1]),theirs.attention.values()));
            for (which,theirs) in Ring::ALL.into_iter().zip(Slot::ALL.map(|slot|theirs.conv.get(slot))) {
                let (channels,taps)=layout.ring_size(which);
                let inputs=(tokens.len()+1-taps..tokens.len()).flat_map(|position|{let at=layout.ring_slot(which,position as u64)*channels;kept[2+which as usize][at..at+channels].iter().map(|&v|rn(v))}).collect::<Vec<_>>();
                add("kept convolution inputs",equal(&inputs,theirs.inputs()));
            }
        }
        println!("{name}: the host run of the driver against the whole-model host form, 12 tokens, equal words of all: {counts:?}");
        for (what,(same,all)) in &counts {assert_eq!(same,all,"{name}: {what}");}
        assert_eq!(counts.len(),9);
        // A window layer has gone round: its ring is full and holds the last tokens, each in the slot of its position.
        let window=&layouts[0];assert!(window.ring && window.rows<12);
        assert!((0..window.rows).all(|row|last[0][0][row*window.row_words..(row+1)*window.row_words].iter().any(|v|v.to_bits()!=0)));
        // A layer that sees every token holds one row a token, in order, and nothing after them.
        let plane=&layouts[1];assert!(!plane.ring && plane.rows==16);
        assert!(last[1][0].chunks_exact(plane.row_words).enumerate().all(|(row,words)|words.iter().any(|v|v.to_bits()!=0)==(row<12)));
        // Token 5 of the chat is a stored row that the head cannot write: it is read, and no step picks a row like it.
        assert!(u64::from(tokens[5])>=d.head_rows);
    }
}
#[test]
fn the_state_bytes_of_each_visit_follow_the_slot_rule() {
    // After each token: the row that the token wrote in each plane and each ring is at the slot of its position,
    // every other word is the word of the state before, and a run that copies the state out and back in half way
    // goes on with the same words.
    for fixture in [Fixture::small(),Fixture::wide()] {
        let d=&fixture.description;let tokens=fixture.chat(12);let hand=Hand::new(Mode::All);
        let driver=Driver::new(&hand,&fixture,d,16).unwrap();let layouts=driver.layouts().to_vec();
        let (mut state,mut before,mut seen)=(driver.clean().unwrap(),copy(&driver.clean().unwrap()),Vec::new());
        for (position,&id) in tokens.iter().enumerate() {
            if position==6 {
                // Copy the state out and back in: new values with the same words. The ring has gone round by now.
                state=State {position:state.position,visits:copy(&state).into_iter().map(|[keys,values,a,b,c,e]|state::VisitState {keys,values,rings:[a,b,c,e],next:std::array::from_fn(|_|None),planes:None,every:std::array::from_fn(|_|None),written:None}).collect()};
            }
            let step=token(&driver,&mut state,id).unwrap();assert_eq!(state.position,position as u64+1);
            for (slot,(layout,(now,was))) in layouts.iter().zip(step.state.iter().zip(&before)).enumerate() {
                let row=layout.slot(position as u64).unwrap()*layout.row_words;
                for plane in 0..2 {
                    let changed=(0..now[plane].len()).filter(|&i|now[plane][i].to_bits()!=was[plane][i].to_bits()).collect::<Vec<_>>();
                    assert!(!changed.is_empty() && changed.iter().all(|i|(row..row+layout.row_words).contains(i)),"token {position} slot {slot} plane {plane}");
                    // A finished key or value is a BF16 value in each half of a word.
                    assert!(row_values(layout,&now[plane],row/layout.row_words).iter().all(|&v|v.is_finite() && rn(v).to_bits()==v.to_bits()));
                }
                for which in Ring::ALL {
                    let (channels,at)=(layout.ring_size(which).0,layout.ring_slot(which,position as u64)*layout.ring_size(which).0);let (now,was)=(&now[2+which as usize],&was[2+which as usize]);
                    assert!((0..now.len()).all(|i|(at..at+channels).contains(&i) || now[i].to_bits()==was[i].to_bits()),"token {position} slot {slot} {which:?}");
                    assert!(now[at..at+channels].iter().any(|v|*v!=0.0));
                }
            }
            before=step.state.clone();seen.push(step);
        }
        assert_eq!(words(&seen),words(&run(&hand,&fixture,&tokens,16).unwrap()),"the copied state went on with other words");
        // The capacity is a size of the state and of nothing else: a larger state gives the same words.
        let larger=run(&hand,&fixture,&tokens,24).unwrap();
        for (a,b) in seen.iter().zip(&larger) {assert_eq!((bits(&a.logits),a.top,a.streams.iter().map(|s|bits(s)).collect::<Vec<_>>()),(bits(&b.logits),b.top,b.streams.iter().map(|s|bits(s)).collect::<Vec<_>>()));}
    }
}
#[test]
fn the_host_run_equals_the_hand_values_and_no_launch_leaves_the_shared_arms() {
    assert_eq!(missing(),[],"a block arm of the driver has no body, or an older form");
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let tokens=fixture.chat(12);let host=run(&Host,&fixture,&tokens,16).unwrap();
        // Every arm that this seat launches, on the hand form of its contract.
        let hand=Hand::new(Mode::All);let by_hand=run(&hand,&fixture,&tokens,16).unwrap();
        assert!(hand.by_hand.borrow().len()>=8,"the hand forms did not run");
        assert_eq!(words(&host),words(&by_hand),"{name}: the host run differs from the hand values");
        // The same run with every launch counted: none needs a hand form, and no part of the driver calls the norm of
        // the backend (its host form rounds twice; the rounded norm of this family is arm `RoundedNorm`).
        let counted=Hand::new(Mode::Refused);assert_eq!(words(&run(&counted,&fixture,&tokens,16).unwrap()),words(&host));
        assert!(counted.by_hand.borrow().is_empty() && counted.norms.get()==0 && hand.norms.get()==0);
        println!("{name}: {} words of the host run equal the hand values; {} launches a token over {} visits, {} of them on a hand form in the hand run",
            words(&host).len(),counted.launches()/12,fixture.description.visits.len(),hand.by_hand.borrow().values().sum::<usize>()/12);
    }
}
#[test]
fn a_step_that_fails_changes_no_position_and_runs_again_with_the_same_words() {
    let fixture=Fixture::small();let tokens=fixture.chat(8);
    let whole=run(&Hand::new(Mode::All),&fixture,&tokens,16).unwrap();
    // Stop token 6 at six launches of its step: the first, one in the first visit, some in later visits and passes, the
    // last. (Launch 1 of `token` is the embedding that the test reads for itself, before the step.)
    let hand=Hand::new(Mode::All);let driver=Driver::new(&hand,&fixture,&fixture.description,16).unwrap();
    let count={let before=hand.launches();let mut state=driver.clean().unwrap();token(&driver,&mut state,tokens[0]).unwrap();hand.launches()-before};
    for stop in [2,9,count/3,count/2,count-1,count] {
        let mut state=driver.clean().unwrap();
        for &id in &tokens[..6] {token(&driver,&mut state,id).unwrap();}
        hand.fail_at.set(hand.launches()+stop);
        assert!(token(&driver,&mut state,tokens[6]).unwrap_err().contains("told to fail"));assert_eq!(state.position,6);
        hand.fail_at.set(0);
        for (place,&id) in tokens.iter().enumerate().skip(6) {
            let again=token(&driver,&mut state,id).unwrap();
            assert_eq!(words(std::slice::from_ref(&again)),words(std::slice::from_ref(&whole[place])),"stop {stop} token {place}");
        }
    }
}
#[test]
fn the_token_step_refuses_what_it_cannot_run() {
    let fixture=Fixture::small();let d=&fixture.description;let hand=Hand::new(Mode::All);
    let driver=Driver::new(&hand,&fixture,d,3).unwrap();let mut state=driver.clean().unwrap();
    // A token outside the stored rows. The last stored row is read, though the head cannot write it.
    assert!(token(&driver,&mut state,d.vocab as u32).unwrap_err().contains("outside the stored embedding rows"));assert_eq!(state.position,0);
    for id in [d.vocab as u32-1,0,3] {token(&driver,&mut state,id).unwrap();}
    // The state holds three tokens: the fourth has no row in a layer that sees every token.
    let full=copy(&state);
    assert!(token(&driver,&mut state,1).unwrap_err().contains("longer than the state's capacity"));assert_eq!(state.position,3);
    // The refused token wrote rows of position 3 only where a ring has them; every plane of a layer that sees every token is as it was.
    for (slot,layout) in driver.layouts().iter().enumerate() {if !layout.ring {assert_eq!(bits(&state.visits[slot].keys),bits(&full[slot][0]));}}
    // A state of another driver, and a capacity that the model cannot have.
    let other=Driver::new(&hand,&Fixture::wide(),&Fixture::wide().description,3).map(|_|()).is_ok();assert!(other);
    let wide=Fixture::wide();let wider=Driver::new(&hand,&wide,&wide.description,3).unwrap();
    assert!(wider.step(&mut driver.clean().unwrap(),0).unwrap_err().contains("the state differs from the visits"));
    assert!(Driver::new(&hand,&fixture,d,0).is_err() && Driver::new(&hand,&fixture,d,d.max_positions+1).is_err());
    // The longest chat of the model.
    let long=Driver::new(&hand,&fixture,d,d.max_positions).unwrap();let mut at_end=long.clean().unwrap();at_end.position=d.max_positions;
    assert!(long.step(&mut at_end,0).unwrap_err().contains("longest length"));
    // A description that the token step cannot run.
    let mut no_cap=d.clone();no_cap.caps.logits=0.0;assert!(Driver::new(&hand,&fixture,&no_cap,3).is_err());
    let mut no_rows=d.clone();no_rows.head_rows=d.vocab+1;assert!(Driver::new(&hand,&fixture,&no_rows,3).is_err());
}
#[test]
fn the_end_caps_the_head_rows_and_picks_the_lower_row_on_equal_logits() {
    let fixture=Fixture::small();let d=&fixture.description;let hand=Hand::new(Mode::All);
    let driver=Driver::new(&hand,&fixture,d,4).unwrap();let mut state=driver.clean().unwrap();
    let step=driver.step(&mut state,2).unwrap();let rows=d.head_rows as usize;
    // The head reads only the rows that the model can write.
    assert_eq!(step.logits.len(),rows);assert!(rows<d.vocab as usize);
    // The cap keeps the order of the logits: the top row is the top row of the head product.
    let best=step.logits.iter().enumerate().fold(0,|best,(row,&v)|if v>step.logits[best] {row} else {best});
    assert_eq!(step.top as usize,best);
    assert!(step.logits.iter().all(|v|v.abs()<d.caps.logits as f32));
    assert_eq!(crate::bounded_decoder::greedy_logits(&[1.0,3.0,3.0,2.0],4).unwrap(),1,"equal logits give the lower row");
}
#[test]
fn a_prompt_token_with_no_head_leaves_the_state_of_a_step_and_runs_no_end() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let tokens=fixture.chat(12);let whole=run(&Host,&fixture,&tokens,16).unwrap();
        let counted=Hand::new(Mode::Refused);let driver=Driver::new(&counted,&fixture,&fixture.description,16).unwrap();
        // Every cut of the chat: the tokens before the cut are read with no head, the tokens from the cut on are steps.
        for cut in [1,5,11] {
            let mut state=driver.clean().unwrap();
            let (caps,before)=(counted.count(Op::Softcap),counted.launches());
            for (place,&id) in tokens[..cut].iter().enumerate() {
                driver.read(&mut state,id).unwrap();
                assert_eq!(state.position,place as u64+1);
                assert_eq!(copy(&state).iter().flatten().flat_map(|v|bits(v)).collect::<Vec<_>>(),whole[place].state.iter().flatten().flat_map(|v|bits(v)).collect::<Vec<_>>(),"{name} cut {cut} token {place}: the state after a read differs from the state after a step");
            }
            assert_eq!(counted.count(Op::Softcap),caps,"{name}: a read ran the cap of the head");
            let read_launches=counted.launches()-before;
            for (place,&id) in tokens.iter().enumerate().skip(cut) {
                let step=token(&driver,&mut state,id).unwrap();
                assert_eq!(words(std::slice::from_ref(&step)),words(std::slice::from_ref(&whole[place])),"{name} cut {cut} token {place}");
            }
            // A step of the same tokens runs more launches: the end mix, the last norm, the head product and the cap.
            let (mut again,before)=(driver.clean().unwrap(),counted.launches());
            for &id in &tokens[..cut] {driver.step(&mut again,id).unwrap();}
            let step_launches=counted.launches()-before;
            assert!(step_launches>=read_launches+3*cut,"{name}: a read ran {read_launches} launches and a step {step_launches}");
            if cut==11 {println!("{name}: {} launches a token with no head, {} with the head",read_launches/cut,step_launches/cut);}
        }
        // A read refuses what a step refuses, and the position stays.
        let mut state=driver.clean().unwrap();
        assert!(driver.read(&mut state,fixture.description.vocab as u32).is_err());assert_eq!(state.position,0);
    }
}
/// One step of rows, with the stream after each visit, `[R][width]`.
fn rows_step<C:Compute<Value=Vec<f32>>,W:Weights<C>>(driver:&Driver<'_,C,W>,chats:&mut [ChatStep<'_,Vec<f32>>])->Result<(Vec<Vec<f32>>,StepRows<Vec<f32>>)> {
    let mut streams=Vec::new();
    let step=driver.step_rows_seen(chats,|place,stream|{assert_eq!(place,streams.len(),"the visits are seen in run order");streams.push(stream.clone());Ok(())})?;
    Ok((streams,step))
}
fn state_bits(state:&[Kept])->Vec<u32> {state.iter().flatten().flat_map(|v|bits(v)).collect()}
/// The gate of the rows work on the two made-up checkpoints (their windows are 4 and 3 tokens, so the rings wrap):
/// a row of a step of rows has the words of that row in steps of one row, for any cut of a chat into steps and
/// among other chats.
#[test]
fn a_row_of_a_step_of_rows_has_the_words_of_that_row_in_steps_of_one_row() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let (width,head)=(d.hidden as usize,d.head_rows as usize);
        let tokens=fixture.chat(12);let whole=run(&Host,&fixture,&tokens,16).unwrap();
        let driver=Driver::with_limits(&Host,&fixture,d,16,rows::Limits {step_rows:12,key_bytes:1<<20}).unwrap();
        assert_eq!((driver.limits().step_rows,driver.capacity()),(12,16));
        // The state after each token, from steps of one row on this driver: its state sizes are its own.
        let mut state=driver.clean().unwrap();assert_eq!(state_bits(&copy(&state)).len(),driver.state_words());
        let states=tokens.iter().map(|&id|{
            let step=driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&[id],wanted:Wanted::None,logits:false}]).unwrap();
            assert!(step.tops.is_empty() && step.logits.is_none());copy(&state)
        }).collect::<Vec<_>>();
        // One chat, cut into steps of 1 row, of 2 to 4 rows, and into one step.
        let mut compared=0usize;
        for cut in [vec![1;12],vec![2,3,4,3],vec![12]] {
            let (mut state,mut at)=(driver.clean().unwrap(),0);
            for count in cut.iter().copied() {
                let (streams,step)=rows_step(&driver,&mut [ChatStep {state:&mut state,tokens:&tokens[at..at+count],wanted:Wanted::All,logits:true}]).unwrap();
                let logits=step.logits.unwrap();assert_eq!((streams.len(),step.tops.len(),logits.len()),(d.visits.len(),count,count*head));
                for row in 0..count {
                    let one=&whole[at+row];
                    for (place,stream) in streams.iter().enumerate() {assert_eq!(bits(&stream[row*width..(row+1)*width]),bits(&one.streams[place]),"{name} cut {cut:?} token {} visit {place}",at+row);}
                    assert_eq!(bits(&logits[row*head..(row+1)*head]),bits(&one.logits),"{name} cut {cut:?} token {}",at+row);assert_eq!(step.tops[row],one.top);
                    compared+=streams.len()*width+head;
                }
                at+=count;assert_eq!(state.position,at as u64);
                assert_eq!(state_bits(&copy(&state)),state_bits(&states[at-1]),"{name} cut {cut:?}: the state after token {}",at-1);
            }
        }
        // Three chats of other tokens and lengths in the same steps. Each has the words of the chat alone.
        let lists=[tokens.clone(),tokens.iter().rev().copied().collect::<Vec<_>>(),tokens[3..10].to_vec()];
        let alone=[run(&Host,&fixture,&lists[0],16).unwrap(),run(&Host,&fixture,&lists[1],16).unwrap(),run(&Host,&fixture,&lists[2],16).unwrap()];
        let wants=[(Wanted::Last,true),(Wanted::All,false),(Wanted::None,true)];
        let mut held=[driver.clean().unwrap(),driver.clean().unwrap(),driver.clean().unwrap()];let mut at=[0usize;3];
        for plan in [[3,1,2],[1,4,1],[4,0,3],[4,7,1]] {
            let (mut chats,mut rows)=(Vec::new(),Vec::new());
            for (chat,state) in held.iter_mut().enumerate() {
                let count=plan[chat];if count==0 {continue;}
                for row in 0..count {
                    let wanted=match wants[chat].0 {Wanted::None=>false,Wanted::Last=>row+1==count,Wanted::All=>true};
                    rows.push((&alone[chat][at[chat]+row],wanted,wants[chat].1));
                }
                chats.push(ChatStep {state,tokens:&lists[chat][at[chat]..at[chat]+count],wanted:wants[chat].0,logits:wants[chat].1});
                at[chat]+=count;
            }
            let (streams,step)=rows_step(&driver,&mut chats).unwrap();
            for (row,(one,_,_)) in rows.iter().enumerate() {
                for (place,stream) in streams.iter().enumerate() {assert_eq!(bits(&stream[row*width..(row+1)*width]),bits(&one.streams[place]),"{name} plan {plan:?} row {row} visit {place}");}
                compared+=streams.len()*width;
            }
            assert_eq!(step.tops,rows.iter().filter(|row|row.1).map(|row|row.0.top).collect::<Vec<_>>(),"{name} plan {plan:?}");
            let logits=rows.iter().filter(|row|row.1 && row.2).flat_map(|row|bits(&row.0.logits)).collect::<Vec<_>>();
            assert_eq!(step.logits.as_deref().map(bits).unwrap_or_default(),logits,"{name} plan {plan:?}");compared+=logits.len();
        }
        assert_eq!(at,[12,12,7]);for (chat,state) in held.iter().enumerate() {assert_eq!(state.position,at[chat] as u64);}
        assert_eq!(state_bits(&copy(&held[0])),state_bits(&states[11]),"{name}: the state of a chat among other chats");
        // Eight chats in one step, one row each, for 6 steps. Each chat has the words of the chat alone. Chat 7 is
        // chat 0 again: two rows of one step with the same chat are equal.
        let eight=(0..8usize).map(|chat|{let mut list=tokens.clone();list.rotate_left(chat%7);list.truncate(6);list}).collect::<Vec<_>>();
        let apart=eight.iter().map(|list|run(&Host,&fixture,list,16).unwrap()).collect::<Vec<_>>();
        let mut group=(0..8).map(|_|driver.clean().unwrap()).collect::<Vec<_>>();
        for at in 0..6 {
            let mut chats=group.iter_mut().zip(&eight).enumerate().map(|(chat,(state,list))|ChatStep {state,tokens:&list[at..at+1],wanted:Wanted::All,logits:chat%2==0}).collect::<Vec<_>>();
            let (streams,step)=rows_step(&driver,&mut chats).unwrap();
            for (chat,one) in apart.iter().enumerate() {
                for (place,stream) in streams.iter().enumerate() {assert_eq!(bits(&stream[chat*width..(chat+1)*width]),bits(&one[at].streams[place]),"{name} eight chats: chat {chat} token {at} visit {place}");}
                assert_eq!(step.tops[chat],one[at].top,"{name} eight chats: chat {chat} token {at}");compared+=streams.len()*width;
            }
            let logits=step.logits.unwrap();assert_eq!(logits.len(),4*head);
            for (row,chat) in [0usize,2,4,6].into_iter().enumerate() {assert_eq!(bits(&logits[row*head..(row+1)*head]),bits(&apart[chat][at].logits),"{name} eight chats: chat {chat} token {at}");}
            compared+=4*head;
            for stream in &streams {assert_eq!(bits(&stream[..width]),bits(&stream[7*width..]));}
        }
        assert_eq!(state_bits(&copy(&group[0])),state_bits(&copy(&group[7])),"{name}: two copies of one chat in the same steps");
        assert_eq!(state_bits(&copy(&group[0])),state_bits(&states[5]),"{name}: the state of a chat among seven other chats");
        // Refused before any launch: the text names the chat, no position moves and no state word changes.
        let before=state_bits(&copy(&held[2]));let [first,_,last]=&mut held;
        let error=driver.step_rows(&mut [ChatStep {state:&mut *first,tokens:&[0],wanted:Wanted::Last,logits:false},ChatStep {state:&mut *last,tokens:&[d.vocab as u32],wanted:Wanted::Last,logits:false}]).unwrap_err();
        assert!(error.contains("chat 1 of the step") && error.contains("outside the stored embedding rows"),"{error}");
        assert!(driver.step_rows(&mut [ChatStep {state:&mut *last,tokens:&[],wanted:Wanted::None,logits:false}]).unwrap_err().contains("chat 0 of the step: it has no token"));
        assert!(driver.step_rows(&mut [ChatStep {state:&mut *last,tokens:&[0;13],wanted:Wanted::None,logits:false}]).unwrap_err().contains("longer than the state"));
        assert!(driver.step_rows(&mut [ChatStep {state:&mut *first,tokens:&[0;5],wanted:Wanted::None,logits:false}]).unwrap_err().contains("longer than the state"));
        assert_eq!((first.position,last.position,state_bits(&copy(last))),(12,7,before));
        let (one,mut fresh)=(Driver::new(&Host,&fixture,d,16).unwrap(),driver.clean().unwrap());
        assert!(one.step_rows(&mut [ChatStep {state:&mut fresh,tokens:&[0,1],wanted:Wanted::None,logits:false}]).unwrap_err().contains("step row limit"));
        println!("{name}: {compared} words of steps of rows equal the steps of one row");
    }
}
/// The gate of drafting on the two made-up checkpoints (windows of 4 and 3 tokens, so the rings and the window planes
/// wrap): a greedy chat whose next tokens are checked in steps of drafted rows (`Driver::verify`, `Driver::accept`)
/// gives the token IDs of the plain greedy chat, and after each check its state has the words of the plain chat at
/// the same position. The drafts are right for a while and then wrong, at every place of the step and with every
/// count of rows from 2 to 5, so a check keeps 1 to all of its rows.
#[test]
fn a_greedy_chat_checked_in_drafted_steps_keeps_the_ids_and_the_state_of_the_plain_chat() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let (width,visits)=(d.hidden as usize,d.visits.len() as u64);
        let (prompt,new)=(fixture.chat(3),24usize);let capacity=(prompt.len()+new+8) as u64;
        let driver=Driver::with_limits(&Host,&fixture,d,capacity,rows::Limits {step_rows:8,key_bytes:1<<20}).unwrap();
        // The plain greedy chat, one row a step: the IDs and the state after each position.
        let mut state=driver.clean().unwrap();let (mut ids,mut states)=(prompt.clone(),Vec::new());
        for at in 0..prompt.len()+new {
            let step=driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&[ids[at]],wanted:Wanted::Last,logits:false}]).unwrap();
            states.push(copy(&state));if at+1>=prompt.len() {ids.push(step.tops[0]);}
        }
        // The taps of a step of rows: the input of visit 0 is the normed embedding, and the state after the last norm
        // has one row of the width for each row.
        let mut tapped=driver.clean().unwrap();
        let (step,taps)=driver.step_rows_taps(&mut [ChatStep {state:&mut tapped,tokens:&prompt,wanted:Wanted::Last,logits:false}],&[0,visits]).unwrap();
        assert_eq!((step.tops[0],taps.len(),taps[1].len()),(ids[prompt.len()],2,prompt.len()*width));
        for (row,&id) in prompt.iter().enumerate() {assert_eq!(bits(&taps[0][row*width..(row+1)*width]),bits(&driver.embedding(id).unwrap()),"{name}: tap 0 row {row}");}
        assert_eq!(state_bits(&copy(&tapped)),state_bits(&states[prompt.len()-1]),"{name}: the state after a step with taps");
        // The drafted chat: the prompt, then checks of the anchor and up to 4 drafted tokens.
        let mut state=driver.clean().unwrap();
        driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&prompt[..prompt.len()-1],wanted:Wanted::None,logits:false}]).unwrap();
        let (mut out,mut round,mut kept)=(prompt[..prompt.len()-1].to_vec(),0usize,[0usize;6]);
        let mut anchor=prompt[prompt.len()-1];
        while out.len()+2<ids.len() {
            let at=out.len();let drafted=(round%4+1).min(ids.len()-at-2);
            // The draft is the plain chat with one wrong token at place `3 * round % 5` (none when that is past the end).
            let wrong=3*round%5;
            let draft=(0..drafted).map(|j|if j==wrong {(ids[at+1+j]+1)%d.head_rows as u32} else {ids[at+1+j]}).collect::<Vec<_>>();
            let tokens=[&[anchor][..],&draft].concat();
            let (tops,taps)=driver.verify(&mut state,&tokens,&[visits]).unwrap();
            assert_eq!((tops.len(),taps[0].len()),(tokens.len(),tokens.len()*width));
            assert!(driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&[0],wanted:Wanted::None,logits:false}]).unwrap_err().contains("still waits"),"{name}: a check that waits refuses a step");
            // Greedy accept: the drafted tokens that equal the top row before them, then the top row of the next row.
            let good=draft.iter().zip(&tops).take_while(|(drafted,top)|drafted==top).count();
            driver.accept(&mut state,tokens.len(),good+1).unwrap();kept[good+1]+=1;
            out.push(anchor);out.extend_from_slice(&draft[..good]);anchor=tops[good];
            assert_eq!(state.position,out.len() as u64);
            assert_eq!(&out[..],&ids[..out.len()],"{name} round {round}: the kept tokens");assert_eq!(anchor,ids[out.len()],"{name} round {round}: the next token");
            // A straight plane is compared up to the chat's end: a row of a token that the check did not keep stays
            // there, and the next step writes it again before any row reads it.
            let (got,want)=(copy(&state),&states[out.len()-1]);
            for (slot,((g,w),layout)) in got.iter().zip(want).zip(driver.layouts()).enumerate() {
                let live=if layout.ring {g[0].len()} else {out.len()*layout.row_words};
                for part in 0..6 {let n=if part<2 {live} else {g[part].len()};assert_eq!(bits(&g[part][..n]),bits(&w[part][..n]),"{name} round {round}: the state after {} tokens, slot {slot} part {part} (keys, values, then the four rings)",out.len());}
            }
            round+=1;
        }
        assert!(kept[1]>0 && kept[2..].iter().all(|&n|n>0),"{name}: every count of kept rows ran: {kept:?}");
        println!("{name}: {round} checks, kept rows {kept:?}, {} tokens equal the plain greedy chat",out.len()+1);
    }
}

/// The token pass with taps (`step_taps`) gives, step by step, the top row, the logits, every tap and the state of a
/// step of one row with taps (`step_rows_taps`): a drafting chat may take either for a plain step.
#[test]
fn the_token_pass_with_taps_has_the_words_of_a_step_of_one_row_with_taps() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let visits=d.visits.len() as u64;
        let (prompt,new)=(fixture.chat(3),12usize);let capacity=(prompt.len()+new+8) as u64;
        let driver=Driver::with_limits(&Host,&fixture,d,capacity,rows::Limits {step_rows:8,key_bytes:1<<20}).unwrap();
        let taps=[0,visits/2,visits];
        let (mut rows,mut token,mut plain)=(driver.clean().unwrap(),driver.clean().unwrap(),driver.clean().unwrap());let mut ids=prompt.clone();
        for at in 0..prompt.len()+new {
            let (want,want_taps)=driver.step_rows_taps(&mut [ChatStep {state:&mut rows,tokens:&[ids[at]],wanted:Wanted::Last,logits:true}],&taps).unwrap();
            let (got,got_taps)=driver.step_taps(&mut token,ids[at],&taps).unwrap();
            let untapped=driver.step(&mut plain,ids[at]).unwrap();
            assert_eq!(got.top,want.tops[0],"{name} at {at}: the top row");
            assert_eq!(bits(&got.logits),bits(want.logits.as_ref().unwrap()),"{name} at {at}: the logits");
            for (tap,(g,w)) in got_taps.iter().zip(&want_taps).enumerate() {assert_eq!(bits(g),bits(w),"{name} at {at}: tap {tap}");}
            assert_eq!(state_bits(&copy(&token)),state_bits(&copy(&rows)),"{name} at {at}: the state");
            assert_eq!(got.top,untapped.top,"{name} at {at}: tapped and plain top rows");
            assert_eq!(bits(&got.logits),bits(&untapped.logits),"{name} at {at}: tapped and plain logits");
            assert_eq!(state_bits(&copy(&token)),state_bits(&copy(&plain)),"{name} at {at}: tapped and plain state");
            if at+1>=prompt.len() {ids.push(got.top);}
        }
    }
}
