//! The `decode_fuse` engine option (off | on), set once at startup from the server arguments.
//! It defaults to on for one Spark and off for more Sparks.
//! On: the launches of a decode step of up to four rows run as a programmatic chain. Each
//! kernel's bytes stay the same; only the start of each kernel moves.
use std::sync::atomic::{AtomicBool, Ordering};
static ON: AtomicBool = AtomicBool::new(false);
/// Freeze the option before the model loads.
pub fn set(on: bool) { ON.store(on, Ordering::Relaxed); }
pub fn on() -> bool { ON.load(Ordering::Relaxed) }
/// Use the topology default unless the user sets the option.
pub fn resolve(option: Option<bool>, single: bool) -> bool { option.unwrap_or(single) }
/// The server argument value.
pub fn parse(value: &str) -> Result<bool, String> {
    match value { "on" => Ok(true), "off" => Ok(false), _ => Err("--decode-fuse must be off or on".into()) }
}
/// Whether a launch of `tokens` rows runs its chain twin: the option is on and the step has
/// up to four rows.
pub fn chain_rows(on: bool, tokens: usize) -> bool { on && tokens <= 4 }
/// The grouped decode entry: all MXFP8 (`mx`), all BF16 (`bf16`) or mixed storage, `total`
/// rows of K `cols`. Returns the entry and whether it is a chain twin. The MXFP8 twin runs
/// wide8 at one row, so it needs latency_mx8_wide1 and an unsplit group.
pub fn pick_group(on: bool, tokens: usize, mx: bool, bf16: bool, total: usize, cols: usize) -> (&'static str, bool) {
    let split = crate::kernel_params::latency_split(total, true);
    if mx {
        let chain = chain_rows(on, tokens) && tokens == 1 && !split && crate::kernel_params::latency_mx8_wide1(tokens, cols);
        if chain { ("flash_mx8_mma_latency_group_chain", true) } else { ("flash_mx8_mma_latency_group", false) }
    } else if bf16 && split && chain_rows(on, tokens) {
        ("flash_bf16_mma_latency_group_chain", true)
    } else {
        ("flash_bf16_mma_latency_group", false)
    }
}
/// The solo MXFP8 decode entry: the chain twin for one row in 128 threads at a wide1 shape.
pub fn pick_solo(on: bool, tokens: usize, cols: usize, threads: u32) -> (&'static str, bool) {
    if chain_rows(on, tokens) && tokens == 1 && threads == 128 && crate::kernel_params::latency_mx8_wide1(tokens, cols) {
        ("flash_mx8_mma_latency_chain", true)
    } else { ("flash_mx8_mma_latency", false) }
}
#[cfg(test)]
#[test]
fn the_picks_hold_the_twins_to_their_shapes() {
    assert!(chain_rows(true, 4) && !chain_rows(true, 5) && !chain_rows(false, 1));
    // MXFP8 groups: one row, unsplit, wide1 K only.
    assert_eq!(pick_group(true, 1, true, false, 16480, 2560), ("flash_mx8_mma_latency_group_chain", true));
    assert_eq!(pick_group(true, 2, true, false, 16480, 2560).1, false);
    assert_eq!(pick_group(true, 1, true, false, 16480, 2560 + 64).1, false);
    assert_eq!(pick_group(true, 1, true, false, 4096, 2560).1, false);
    assert_eq!(pick_group(false, 1, true, false, 16480, 2560).1, false);
    // BF16 groups: split groups of up to four rows; mixed storage stays plain.
    assert_eq!(pick_group(true, 4, false, true, 324, 10240), ("flash_bf16_mma_latency_group_chain", true));
    assert_eq!(pick_group(true, 1, false, false, 1793, 2560).1, false);
    assert_eq!(pick_group(true, 1, false, true, 8192, 2560).1, false);
    assert_eq!(pick_solo(true, 1, 6144, 128), ("flash_mx8_mma_latency_chain", true));
    assert_eq!(pick_solo(true, 1, 6144, 32).1, false);
    assert_eq!(pick_solo(true, 2, 6144, 128).1, false);
    assert_eq!(pick_solo(true, 1, 6144 + 64, 128).1, false);
}
#[cfg(test)]
#[test]
fn the_option_takes_two_words() {
    assert_eq!(parse("on"), Ok(true));
    assert_eq!(parse("off"), Ok(false));
    for bad in ["", "ON", "1", "true"] { assert!(parse(bad).is_err()); }
}
#[cfg(test)]
#[test]
fn the_default_follows_the_topology_and_off_overrides_it() {
    assert!(resolve(None, true));
    assert!(!resolve(None, false));
    for single in [false, true] {
        assert!(!resolve(Some(false), single));
        assert!(resolve(Some(true), single));
    }
}
