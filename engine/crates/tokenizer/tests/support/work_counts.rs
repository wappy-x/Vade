//! Counts exclude allocator, hash-table, standard-library search, and heap-comparison internals.
use super::*;
use std::cell::Cell;
thread_local! { static WORK: Cell<u64> = const { Cell::new(0) }; }
thread_local! { static BUDGET: Cell<u64> = const { Cell::new(u64::MAX) }; }
pub(super) fn add() {
    let work = WORK.get() + 1; WORK.set(work);
    assert!(work <= BUDGET.get(), "encoder operation budget exceeded: count={work} budget={}", BUDGET.get());
}
fn begin(budget: u64) { WORK.set(0); BUDGET.set(budget); }
fn take() -> u64 { BUDGET.set(u64::MAX); WORK.replace(0) }
#[test]
fn encode_scaling_three_sizes() {
    let glm = Tokenizer::from_json(include_bytes!("../glm-tokenizer-slice.json"), Limits::default()).unwrap();
    let json = include_str!("../fixtures/host_edges/tokenizer.json")
        .replace(r"\\p{L}+", r"[\\p{L}\\p{M}]+")
        .replace(r"[^\\s\\p{L}\\p{N}]", r"[^\\s\\p{L}\\p{M}\\p{N}]")
        .replace("151643", "248044").replace("151644", "248045").replace("151645", "248046");
    let hybrid = Tokenizer::from_json(json.as_bytes(), Limits::default()).unwrap();
    for (profile,t) in [("glm",glm),("hybrid",hybrid)] {
        for (case,unit) in [("letter","a"),("merges","hello"),("whitespace"," \t\r\n"),("combining","\u{301}\u{327}"),("special","<|im_end|>")] {
            let mut previous = None;
            for n in [4096,16384,65536] {
                let text = unit.repeat(n); begin(100 * text.len() as u64);
                let ids = t.encode(&text).unwrap(); let work = take();
                assert!(!ids.is_empty(), "empty output: {profile}/{case}/{n}");
                assert!(work > 0 && work <= 100 * text.len() as u64, "work bound: {profile}/{case}/{n} count={work}");
                if let Some(old) = previous { assert!(work <= 6 * old + 64, "scaling: {profile}/{case}/{n} previous={old} count={work}"); }
                eprintln!("WORK profile={profile} case={case} repeats={n} bytes={} count={work}",text.len());
                previous = Some(work);
            }
        }
    }
}
