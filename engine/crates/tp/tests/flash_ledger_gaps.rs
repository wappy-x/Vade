#![allow(dead_code)]
#[path="../src/flash/settings.rs"] mod settings;
#[test]
fn snapshot_verify_commit_restore_and_batch_positions() {
    let mut s=settings::CommandState::default();
    let apply=|s:&mut settings::CommandState,op,id,arg,tokens:&[u32]| {
        s.validate(op,id,arg,false,tokens,128,16,8,3).unwrap();s.apply(op,id,arg,false,tokens);
    };
    apply(&mut s,1,1,64,&[]);apply(&mut s,1,2,64,&[]);
    apply(&mut s,2,1,0,&[1,2]);apply(&mut s,2,2,0,&[3]);
    apply(&mut s,6,0,0,&[1,0,4,0,2,0,5,0]);
    assert_eq!(s.sessions[&1].0,3);assert_eq!(s.sessions[&2].0,2);
    assert!(s.validate(6,0,0,false,&[1,0,4,0,1,0,5,0],128,16,8,3).is_err());
    apply(&mut s,7,9,1,&[0,0]);assert_eq!(s.saved[&9],(3,3));
    apply(&mut s,12,1,0,&[4,5,6]);assert_eq!(s.sessions[&1],(6,64,Some(3)));
    assert!(s.validate(13,1,3,false,&[],128,16,8,3).is_err());
    apply(&mut s,13,1,4,&[]);assert_eq!(s.sessions[&1],(4,64,None));
    apply(&mut s,8,2,9,&[]);assert_eq!(s.sessions[&2],(3,64,None));
    apply(&mut s,9,9,0,&[]);assert!(s.saved.is_empty());
    apply(&mut s,4,1,0,&[]);apply(&mut s,4,2,0,&[]);apply(&mut s,5,0,0,&[]);
    assert!(s.sessions.is_empty());
}
