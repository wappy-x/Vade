// Compile the production host ledger directly without exporting its private API.
#![allow(dead_code)]
use spark_tp::Result;
#[path="../src/service_commands.rs"] mod commands;
use commands::*;
#[test]
fn truncated_logit_word_is_rejected() {
    assert_eq!(finite_bytes(&[1.0f32.to_le_bytes(),(-2.0f32).to_le_bytes()].concat()),Ok(()));
    for tail in 1..=3 {
        let mut bytes=1.0f32.to_le_bytes().to_vec();bytes.extend(vec![255;tail]);
        assert_eq!(finite_bytes(&bytes),Err(OUTPUT_ERROR.into()),"tail {tail}");
    }
}
#[test]
fn prefill_batch_and_fork_preserve_independent_positions() {
    let mut state=CommandState::default();
    state.apply(OPEN,1,0,&[]);state.apply(OPEN,2,0,&[]);
    let payload=[1,0,2,3,4,2,0,1,5];
    state.validate(PREFILL_BATCH,0,0,&payload,4096,16).unwrap();state.apply(PREFILL_BATCH,0,0,&payload);
    assert_eq!(state.sessions[&1],2);assert_eq!(state.sessions[&2],1);
    state.validate(FORK,1,3,&[],4096,16).unwrap();state.apply(FORK,1,3,&[]);
    state.apply(STEP,3,7,&[]);assert_eq!(state.sessions[&3],3);assert_eq!(state.sessions[&1],2);
    let before=state.sessions.clone();
    for payload in [vec![1,0],vec![1,0,2,7],vec![1,0,1,7,1,0,1,8],vec![1,0,1,16]] {
        assert!(state.validate(PREFILL_BATCH,0,0,&payload,4096,16).is_err());assert_eq!(state.sessions,before);
    }
    assert!(state.validate(STOP,0,0,&[],4096,16).is_err());
    for id in [1,2,3] {state.validate(CLOSE,id,0,&[],4096,16).unwrap();state.apply(CLOSE,id,0,&[]);}
    state.validate(STOP,0,0,&[],4096,16).unwrap();assert!(state.sessions.is_empty());
}
