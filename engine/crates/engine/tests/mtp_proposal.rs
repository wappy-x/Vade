use std::marker::PhantomData; type Result<T> = std::result::Result<T,String>; struct Config { hidden: u64 } struct Group(usize); impl Group { fn distributed(&self) -> bool { self.0 > 1 } } struct Bound { group: Group } struct Model { config: Config, capacity: usize, bound: Option<Bound> } #[derive(Clone)] struct Hidden(u32); impl Hidden { fn slice(&self, _: std::ops::Range<usize>) -> Result<Self> { Ok(self.clone()) } }
struct Draft { proposal_root: Option<(usize,u32)>, past: usize, last: Hidden, normalized: Hidden, cache: Vec<u32>, fail: Option<usize> } impl Draft { fn forward(&mut self, _: &Model, token: u32, hidden: &Hidden, _: bool) -> Result<Option<u32>> { self.cache[self.past]=token+hidden.0; self.past+=1; self.normalized=Hidden(token+1); if self.fail==Some(self.past) { Err("injected proposal failure".into()) } else { Ok(Some(token+1)) } } }
struct Policy { options: spark_engine::hybrid_draft::Options, rate: f64 } impl Policy { fn more(&self, c: f64, _: usize) -> bool { spark_engine::hybrid_draft::more(c, self.rate, 1.0, 1.0) } }
impl Draft { fn confidence(&mut self, _: &Model) -> Result<f64> { Ok(0.01) } }
struct Session<'m,'a,'b> { draft_depth: usize, external: Option<()>, draft_policy: Policy, model: &'m Model, mtp: Option<Draft>, position: usize, life: PhantomData<(&'a (), &'b ())> } impl Session<'_, '_, '_> { fn finish_mtp_verification(&mut self) -> Result<()> { Ok(()) } } include!("../src/hybrid/mtp_proposal.rs");
#[test] fn proposals_preserve_committed_prefix_on_success_retry_fork_and_failure() { let model=Model{config:Config{hidden:1},capacity:32,bound:None}; for depth in 1..=4 { let mut s=Session{draft_depth:3,external:None,draft_policy:Policy { options:Default::default(), rate:0.1 },model:&model,mtp:Some(Draft{proposal_root:None,past:7,last:Hidden(3),normalized:Hidden(0),cache:vec![99;32],fail:None}),position:8,life:PhantomData}; let a=s.propose_mtp_tokens(5,depth).unwrap(); assert_eq!(s.mtp.as_ref().unwrap().proposal_root,Some((7,5))); assert_eq!(s.mtp.as_ref().unwrap().past,7); assert_eq!(&s.mtp.as_ref().unwrap().cache[..7],&[99;7]); assert_eq!(s.propose_mtp_tokens(5,depth).unwrap(),a); let d=s.mtp.as_mut().unwrap(); assert_eq!(d.past+1,s.position); d.forward(&model,9,&d.last.clone(),false).unwrap(); assert_eq!(d.cache[7],12); d.past=7; s.draft_policy.options.gated_depth=true; assert_eq!(s.propose_mtp_tokens(5,depth).unwrap(),a[..1]); assert_eq!(s.mtp.as_ref().unwrap().past,7); s.draft_policy.options.gated_depth=false; for fail in 8..=7+depth { s.mtp.as_mut().unwrap().fail=Some(fail); assert!(s.propose_mtp_tokens(5,depth).is_err()); assert_eq!(s.mtp.as_ref().unwrap().proposal_root,None); assert_eq!(s.mtp.as_ref().unwrap().past,7); assert_eq!(&s.mtp.as_ref().unwrap().cache[..7],&[99;7]); assert_eq!(s.position,8); } } }

#[test]
fn default_depth_and_explicit_override_are_rank_consistent() {
    let propose = |world, rate, explicit: Option<&str>| {
        let model = Model { config: Config { hidden: 1 }, capacity: 32,
            bound: Some(Bound { group: Group(world) }) };
        let mut session = Session {
            draft_policy: Policy { options: spark_engine::hybrid_draft::Options {
                copy_first: true, gated_depth: true }, rate }, model: &model,
            mtp: Some(Draft { proposal_root: None, past: 7, last: Hidden(3),
                normalized: Hidden(0), cache: vec![99; 32], fail: None }),
            position: 8, life: PhantomData, external: None, draft_depth: spark_engine::hybrid_draft::default_depth(world > 1),
        };
        if let Some(value) = explicit { session.set_draft_depth(value.parse().unwrap()).unwrap(); }
        session.propose_mtp_tokens(5, session.draft_depth).unwrap()
    };
    if std::env::var_os("VADE_TEST_DEPTH_CHILD").is_some() { for world in 1..=4 { assert_eq!(propose(world, 0.0, Some(&std::env::var("MTP").unwrap())).len(), 4); } return; }
    assert!(std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact", "default_depth_and_explicit_override_are_rank_consistent"]).env("VADE_TEST_DEPTH_CHILD", "1").env("MTP", "4").status().unwrap().success());
    // These local measurements make singleton ranks take opposite gate branches.
    assert_eq!(propose(1, 0.0, None).len(), 3);
    assert_eq!(propose(1, 0.1, None).len(), 1);
    for world in 2..=4 {
        let expected = propose(world, 0.0, None);
        assert_eq!(expected.len(), 2);
        assert_eq!(propose(world, 0.1, Some("3")).len(), 3);
        for rate in [0.1, 1.0, f64::NAN] { assert_eq!(propose(world, rate, None), expected); }
    }
}
