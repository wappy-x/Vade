use spark_engine::{draft::{Config, ModelProposal, Speculator, Target}, hybrid_draft::{CopyIndex, Options, confidence, more}, sample::Sampling};
#[test] fn truncate_regrow_past_seen_has_no_stale_spans_and_identical_answer_bytes() {
    let ids = |text: &[u8]| text.iter().copied().map(u32::from).collect::<Vec<_>>();
    for (tail, next, expected) in [(b"12345678Yabcdefg".as_slice(), b'h', b"".as_slice()), (b"12345678Y1234567", b'8', b"Y123")] {
        let mut history=ids(b"abcdefghXabcdefg"); let mut copy=CopyIndex::default();
        assert_eq!(copy.propose(&history,u32::from(b'h'),4),ids(b"Xabc"));
        history.truncate(2); copy.truncate(2); history.extend(ids(tail));
        assert!(history.len()>16); assert_eq!(copy.propose(&history,u32::from(next),4),ids(expected));
        let at=history.len(); let mut canonical=history.clone(); canonical.push(u32::from(next)); canonical.extend(ids(b"fn run() { call(); }\nfn run() { fail(); }\n"));
        let mut replay=Replay{ids:canonical.clone(),at,copy,corrupt:true};
        let mut spec=Speculator::from_selected(&history,at,u32::from(next),256,Config::strict(canonical.len()-at),Sampling::greedy()).unwrap();
        while !spec.is_done() { spec.step(&mut replay,5,|_|false).unwrap(); }
        assert!(spec.stats.rollbacks>0);
        // Proposals never choose answers: target verification preserves every byte.
        assert_eq!(spec.tokens().iter().map(|&t|t as u8).collect::<Vec<_>>(),canonical[at..].iter().map(|&t|t as u8).collect::<Vec<_>>());
    }
}
struct Replay { ids: Vec<u32>, at: usize, copy: CopyIndex, corrupt: bool }impl Target for Replay {fn checkpoint(&self) -> usize { self.at }fn truncate(&mut self, at: usize) -> Result<(), String> { self.copy.truncate(at); self.at=at; Ok(()) }fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { self.at+=1; let mut row=vec![-1.0;256]; row[self.ids.get(self.at).copied().unwrap_or(0) as usize]=1.0; Ok(row) }fn model_draft_width(&self) -> usize { 5 } fn native_draft_policy(&self) -> bool { true }fn propose_model(&mut self, h: &[u32], n: u32, limit: usize) -> Result<Option<ModelProposal>,String> { let mut tokens=self.copy.propose(h,n,limit); if self.corrupt { if let Some(t)=tokens.first_mut() { *t^=1; } } Ok(Some(ModelProposal { tokens,proposal_ms:0.0,catchup_ms:0.0 })) }}
#[test] fn recorded_code_json_repeats_and_rejection_equal_plain() {
    for text in ["fn run() { call(); }\nfn run() { fail(); }\nfn run() { call(); }\n", "{\"tool\":\"edit\",\"args\":{\"x\":1}}\n{\"tool\":\"edit\",\"args\":{\"x\":2}}\n"] { let ids:Vec<_>=text.bytes().map(u32::from).collect(); for (depth,corrupt) in (1..=4).flat_map(|d|[(d,false),(d,true)]) { let mut r=Replay{ids:ids.clone(),at:8,copy:CopyIndex::default(),corrupt}; let mut c=Config::strict(ids.len()-8); c.max_rows=depth+1; let mut s=Speculator::from_selected(&ids[..8],8,ids[8],256,c,Sampling::greedy()).unwrap(); while !s.is_done() { s.step(&mut r,depth+1,|_|false).unwrap(); } assert_eq!(s.tokens(),&ids[8..]); assert_eq!(r.at,ids.len()); assert!(if corrupt {s.stats.rollbacks>0} else {s.stats.accepted>0}); } }
    let mut i=CopyIndex::default(); assert_eq!(i.propose(&[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7],8,4),[9,1,2,3]); assert!(i.propose(&[1,2],3,4).is_empty()); assert!(!Options::default().copy_first && !Options::default().gated_depth); assert_eq!(confidence(&[0.0,0.0]),0.5); assert!(more(0.9,0.1,1.0,1.0)); assert!(!more(0.1,0.1,1.0,1.0)); assert!(!more(f64::NAN,0.1,1.0,1.0));}

#[test] fn documented_depth_two_recipe_disables_copy_and_gate() {
    assert!(include_str!("../../../docs/speculative-decoding.md").contains(
        "`HYBRID_MTP_COPY=0 HYBRID_MTP_GATE=0` with\n`--mtp --draft strict --draft-depth 2 --draft-k 2`"));
}
