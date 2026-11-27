//! ONE PASS: the proposal chains of a group, made word by word together (`propose_group`), are each chat's own chain
//! (`Verify::propose_mass`): the same tokens, the same score bits, and the same cache after it, so every later
//! proposal is the same too.
use spark_tp::bounded_decoder::{ChainStart,Decoder,Host,PROPOSER_HIDDEN,Verify,propose_group,shared_pass};
#[allow(dead_code,reason="shared checkpoint fixture")]
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn group_chains_are_each_chats_own_chain_tokens_scores_and_cache() {
    let c=fixture::Checkpoint::new_with_predictor(64);
    let mut d=Decoder::new(&c.admission,Host,64).unwrap();d.enable_predictor().unwrap();
    let fresh=|d:&Decoder<'_,Host>|{let mut v=Verify::new(d.zero_state().unwrap());v.track_proposer();v};
    let mut chats=(0..4).map(|_|fresh(&d)).collect::<Vec<_>>();let mut twins=(0..4).map(|_|fresh(&d)).collect::<Vec<_>>();
    // Each chat its own prompt, of its own length.
    for at in 0..4 {let prompt=(0..5+2*at).map(|i|((i*7+at*3)%13+1) as u32).collect::<Vec<_>>();chats[at].rows(&d,&prompt).unwrap();twins[at].rows(&d,&prompt).unwrap();}
    let (mut chains,mut stopped)=(0,0);
    for round in 0..8 {
        let next=[(round%9+1) as u32,((round*5+2)%11+1) as u32,((round*3+7)%13+1) as u32,((round*7+5)%13+1) as u32];
        // After a full-width check, vary the depths and let chat 3 continue plain after round 3.
        // A probability stop above one cuts chat 1 after its first proposal.
        let depths=if round==0 {[3;4]} else {[3,1+round%3,2,if round>=4 {0} else {3}]};
        let stops=[None,if round==0 {None} else {Some(if round%2==0 {0.5} else {1.5})},None,None];
        let made={let mut group=chats.iter_mut().zip(next).zip(depths).zip(stops).map(|(((v,n),depth),stop)|(v,n,depth,stop)).collect::<Vec<_>>();
            propose_group(&d,&mut group).unwrap()};
        assert_eq!(made.len(),4);
        let inputs=next.iter().zip(&made).map(|(&next,(tokens,_,_))|
            std::iter::once(next).chain(tokens.iter().copied()).collect::<Vec<_>>()).collect::<Vec<_>>();
        if round==0 {assert_eq!(inputs.iter().map(Vec::len).sum::<usize>(),16);}
        let entries=chats.iter().map(Verify::position).collect::<Vec<_>>();
        let logits={let mut group=chats.iter_mut().zip(&inputs).map(|(v,rows)|(v,rows.as_slice())).collect::<Vec<_>>();
            shared_pass(&d,&mut group).unwrap()};
        for at in 0..4 {
            let (tokens,mass,_)=twins[at].propose_mass(&d,next[at],depths[at],stops[at]).unwrap();
            assert_eq!(made[at].0,tokens,"round {round} chat {at}: the tokens");
            assert_eq!(made[at].1.iter().map(|p|p.map(f32::to_bits)).collect::<Vec<_>>(),mass.iter().map(|p|p.map(f32::to_bits)).collect::<Vec<_>>(),"round {round} chat {at}: the scores");
            if depths[at]>0 {chains+=1;assert!(!tokens.is_empty());}
            else {assert!(tokens.is_empty() && mass.is_empty());}
            if stops[at]==Some(1.5) && depths[at]>1 {stopped+=1;assert_eq!(tokens.len(),1,"a stop over every probability ends the chain after one word");}
            let solo=twins[at].rows(&d,&inputs[at]).unwrap();
            assert_eq!(logits[at].iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),
                solo.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),"round {round} chat {at}: the verified rows");
            // Keep the whole chain, only the canonical token, or part of the chain.
            let kept=match (round+at)%3 {0=>inputs[at].len(),1=>1,_=>inputs[at].len().min(2)};
            let end=entries[at]+kept as u64;
            chats[at].truncate(&d,end).unwrap();
            twins[at].truncate(&d,end).unwrap();
            assert_eq!(chats[at].waiting(),twins[at].waiting(),"round {round} chat {at}: the retained trail");
        }
    }
    assert_eq!(chains,28);assert!(stopped>=1,"the probability stop ends at least one chain");
}

#[test]
fn joined_chain_cache_matches_each_scalar_cache_word_for_word() {
    let c=fixture::Checkpoint::new_with_predictor(64);
    let mut d=Decoder::new(&c.admission,Host,64).unwrap();
    d.enable_predictor().unwrap();
    let starts=[(1,2),(4,5),(7,8),(10,11)];
    let rows=starts.iter().map(|&(prior,next)|{
        let (_,_,state,hidden)=d.step_hidden(prior,&d.zero_state().unwrap(),|_,_|Ok(())).unwrap();
        let (_,cache)=d.propose_after(&hidden[PROPOSER_HIDDEN],next,0,None).unwrap();
        let (_,_,_,after)=d.step_hidden(next,&state,|_,_|Ok(())).unwrap();
        (cache,after)
    }).collect::<Vec<_>>();
    for width in [2,3,4] {
        for depth in 1..=3 {
            let chains=rows.iter().zip(starts).take(width).map(|((cache,hidden),(_,next))|
                ChainStart {hidden:&hidden[PROPOSER_HIDDEN],token:next+1,depth,seen:Some(cache),stop:None}).collect::<Vec<_>>();
            let joined=d.propose_chains(&chains).unwrap();
            for (chain,(ids,mass,cache)) in chains.iter().zip(joined) {
                let (expected_ids,expected_mass,expected_cache)=
                    d.propose_after_mass(chain.hidden,chain.token,chain.depth,chain.seen,chain.stop).unwrap();
                assert_eq!(ids.len(),depth);
                assert_eq!(ids,expected_ids);
                assert_eq!(mass.iter().map(|p|p.map(f32::to_bits)).collect::<Vec<_>>(),
                    expected_mass.iter().map(|p|p.map(f32::to_bits)).collect::<Vec<_>>());
                assert_eq!(cache.rows(),expected_cache.rows());
                assert!(cache.layers().keys().eq(expected_cache.layers().keys()));
                for (name,layer) in cache.layers() {
                    assert!(layer.iter().map(|x|x.to_bits()).eq(expected_cache.layers()[name].iter().map(|x|x.to_bits())),
                        "{width} chats, depth {depth}: cache bits differ in {name}");
                }
            }
        }
    }
}
