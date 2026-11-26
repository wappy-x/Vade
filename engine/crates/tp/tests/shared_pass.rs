//! SHARED PASS: one row pass for the rows of several chats gives each chat the bits of its own pass.
use spark_tp::bounded_decoder::{Decoder,Host,Verify,shared_pass};
#[allow(dead_code,reason="shared checkpoint fixture")]
#[path="support/attention_checkpoint.rs"] mod fixture;
fn bits(v:&[f32])->Vec<u32> {v.iter().map(|x|x.to_bits()).collect()}
/// Chats that draft: each has a twin that runs every pass alone. A round is one shared pass of all chats (the twins
/// take their own pass), then each chat's own verdict. The chats stand at different positions, bring different row
/// counts, keep different prefixes, and sometimes carry rows into the next pass.
#[test]
fn a_shared_pass_gives_each_chat_the_bits_of_its_own_pass_and_its_own_rollback() {
    let c=fixture::Checkpoint::new_with_predictor(48);
    let mut d=Decoder::new(&c.admission,Host,48).unwrap();d.enable_predictor().unwrap();
    for chats in [2usize,3,4] {
        let fresh=||{let mut v=Verify::new(d.zero_state().unwrap());v.track_proposer();v};
        let mut shared=(0..chats).map(|_|fresh()).collect::<Vec<_>>();let mut twins=(0..chats).map(|_|fresh()).collect::<Vec<_>>();
        // Different starts: each chat has its own position and history.
        for (at,rows) in [5usize,2,0,3].into_iter().take(chats).enumerate() {
            for row in 0..rows {let token=((row*5+at*3)%11+1) as u32;
                assert_eq!(shared[at].rows(&d,&[token]).unwrap().iter().map(|r|bits(r)).collect::<Vec<_>>(),twins[at].rows(&d,&[token]).unwrap().iter().map(|r|bits(r)).collect::<Vec<_>>());}
        }
        // Each round: the new rows of each chat, and how many of them each chat keeps. `twice`: the chat then gives up
        // one more row with a second verdict, which leaves it rows to carry into the next pass.
        let rounds:[([usize;4],[usize;4],[bool;4]);7]=[([4,4,4,4],[4,1,2,3],[false;4]),([4,1,3,2],[2,1,3,1],[false;4]),([1,1,1,1],[1,1,1,1],[false;4]),([4,4,2,3],[3,4,2,2],[true,false,false,false]),
            ([3,4,2,2],[3,2,1,1],[false,true,false,false]),([2,5,1,4],[1,5,1,2],[false;4]),([4,4,4,4],[4,4,4,4],[false;4])];
        for (round,(new,keep,twice)) in rounds.into_iter().enumerate() {
            let tokens=(0..chats).map(|at|(0..new[at]).map(|row|((round*7+at*5+row*3)%13+1) as u32).collect::<Vec<_>>()).collect::<Vec<_>>();
            let entry=shared.iter().map(Verify::position).collect::<Vec<_>>();let carried=shared.iter().map(Verify::carried).collect::<Vec<_>>();
            if round==4 {assert_eq!(carried[0],2,"chat 0 carries the rows its second verdict left");}
            if round==5 && chats>1 {assert_eq!(carried[1],1);}
            let logits={let mut rows=shared.iter_mut().zip(&tokens).map(|(v,t)|(v,t.as_slice())).collect::<Vec<_>>();shared_pass(&d,&mut rows).unwrap()};
            for at in 0..chats {
                assert_eq!(twins[at].carried(),carried[at]);
                let own=twins[at].rows(&d,&tokens[at]).unwrap();assert_eq!(logits[at].len(),new[at]);
                for (row,(a,b)) in logits[at].iter().zip(&own).enumerate() {assert_eq!(bits(a),bits(b),"chats {chats} round {round} chat {at} row {row}");}
                assert_eq!(shared[at].position(),entry[at]+new[at] as u64);assert_eq!(shared[at].waiting(),twins[at].waiting());
                // Each chat's own verdict: another chat's rows do not move with it.
                for v in [&mut shared[at],&mut twins[at]] {
                    v.truncate(&d,entry[at]+keep[at] as u64).unwrap();
                    if twice[at] {v.truncate(&d,entry[at]+keep[at] as u64-1).unwrap();}
                }
                assert_eq!((shared[at].position(),shared[at].carried(),shared[at].waiting()),(twins[at].position(),twins[at].carried(),twins[at].waiting()));
            }
            // The predictor layer reads each chat's kept hidden rows: equal proposals mean equal trails and caches.
            if round%2==1 {for at in 0..chats {assert_eq!(shared[at].propose(&d,5,3).unwrap().0,twins[at].propose(&d,5,3).unwrap().0,"round {round} chat {at}");}}
        }
        for (at,(a,b)) in shared.into_iter().zip(twins).enumerate() {
            let (a,b)=(a.settle(&d).unwrap(),b.settle(&d).unwrap());assert_eq!(a.position,b.position);
            for (layer,(x,y)) in a.layers.iter().zip(&b.layers).enumerate() {assert_eq!(x.len(),y.len());for (name,value) in x {assert_eq!(bits(value),bits(&y[name]),"chat {at} layer {layer} {name}");}}
            assert_eq!(a.expanded.len(),b.expanded.len());
            for (layer,owner) in &a.expanded {let other=&b.expanded[layer];assert_eq!(owner.1,other.1,"chat {at} held rows of layer {layer}");assert_eq!(bits(&owner.0),bits(&other.0));}
        }
    }
}
/// The bound and the refusals: a shared pass takes two chats or more and at most VERIFY_ROWS rows in all.
#[test]
fn a_shared_pass_refuses_one_chat_and_more_rows_than_a_check_pass() {
    let c=fixture::Checkpoint::new_with_predictor(48);
    let mut d=Decoder::new(&c.admission,Host,48).unwrap();d.enable_predictor().unwrap();
    let mut a=Verify::new(d.zero_state().unwrap());let mut b=Verify::new(d.zero_state().unwrap());
    assert!(shared_pass(&d,&mut [(&mut a,&[1,2][..])]).is_err());
    let nine=[1u32,2,3,4,5,6,7,8,9];
    assert!(shared_pass(&d,&mut [(&mut a,&nine[..]),(&mut b,&nine[..8])]).is_err(),"17 rows");
    assert!(shared_pass(&d,&mut [(&mut a,&[][..]),(&mut b,&nine[..2])]).is_err(),"a chat with no row");
    // These fresh states remain at position zero and can take a pass. This does not test refusal with open rows.
    assert_eq!((a.position(),b.position()),(0,0));
    assert_eq!(shared_pass(&d,&mut [(&mut a,&nine[..8]),(&mut b,&nine[..8])]).unwrap().iter().map(Vec::len).collect::<Vec<_>>(),[8,8]);
}
