//! One-chat drafting defaults. Each switch changes only what the drafter proposes, or which exact
//! verify entry computes the same words: verification still picks every token, so output
//! bytes equal the undrafted run's under every setting below.
//!
//! | switch | unset (default) | old default |
//! |---|---|---|
//! | VADE_FLASHNEXT_DRAFT_FP4 | 2: NVFP4 head screen + MXFP8 rescore | 0: MXFP8 drafter head (1: NVFP4 head) |
//! | VADE_FLASHNEXT_DRAFT_MX8_LAYER | 1: MXFP8 twins of the MTP layer's BF16 matrices | 0 |
//! | VADE_MTP_CONF | 0.4: a proposal stops at drafter p < 0.4 | off (p in [0, 1) sets another threshold) |
//! | VADE_FLASHNEXT_DRAFT_VOCAB | 65536: drafter argmax over ids below 65536 (the full head when the vocabulary is smaller) | 0: full head |
//! | VADE_VERIFY_MX4, VADE_VERIFY_MX4_PART, VADE_VERIFY_MX4_MIXED, VADE_VERIFY_MX_SOLO97 | 1: exact direct-input / solo entries | 0 |
//! | VADE_FLASHNEXT_MTP_DEPTH | 7 on one Spark and on the pipeline service | 3 |
//!
//! Old default in one line: VADE_FLASHNEXT_DRAFT_FP4=0 VADE_FLASHNEXT_DRAFT_MX8_LAYER=0
//! VADE_MTP_CONF=off VADE_FLASHNEXT_DRAFT_VOCAB=0 VADE_VERIFY_MX4=0 VADE_VERIFY_MX4_PART=0
//! VADE_VERIFY_MX4_MIXED=0 VADE_VERIFY_MX_SOLO97=0 VADE_FLASHNEXT_MTP_DEPTH=3.
//!
//! VADE_FLASHNEXT_MTP_DEPTH is the resident depth, the ceiling for VADE_MTP_ADAPT
//! (default on). A greedy chat alone whose request names no mtp_depth (or 3) drafts
//! up to the resident depth. The step rule (draft_steps.rs) can stop the chain before each step.
//! With VADE_MTP_CONF=off, the chat drafts the chooser's depth. Sampled chats use the chooser too.
//! Request depths 1-2, cohort chats and VADE_MTP_ADAPT=0 keep the request depth
//! (flash_next_lookup::DEFAULT_MTP_DEPTH = 3).
use std::sync::OnceLock;

pub const DRAFT_FP4: u8 = 2;
pub const DRAFT_VOCAB: u32 = 65536;
pub const MTP_CONF: f32 = 0.4;
/// Resident MTP depth on one Spark: one chat writes code +3 % and structured +12 % at 7 than
/// at 5 (prose ties). A pipeline verify pays every stage's fixed cost once per cycle (~37 ms
/// of a ~53 ms 6-row verify on two Sparks), so its chooser gets the same deep ceiling.
pub const RESIDENT_MTP_DEPTH: usize = 7;
pub const PIPELINE_MTP_DEPTH: usize = 7;
/// On/off switches whose unset value is on.
pub const ON_SWITCHES: [&str; 5] = [
    "VADE_FLASHNEXT_DRAFT_MX8_LAYER",
    "VADE_VERIFY_MX4",
    "VADE_VERIFY_MX4_PART",
    "VADE_VERIFY_MX4_MIXED",
    "VADE_VERIFY_MX_SOLO97",
];

/// Unset or "1" = on, "0" = off; anything else is an error.
pub fn parse_on(name: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(format!("{name} must be unset, 0 or 1 (got {v:?})")),
    }
}
/// VADE_FLASHNEXT_DRAFT_FP4: unset = 2, explicit 0, 1 or 2.
pub fn parse_draft_fp4(value: Option<&str>) -> Result<u8, String> {
    match value {
        None => Ok(DRAFT_FP4),
        Some("0") => Ok(0),
        Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some(v) => Err(format!("VADE_FLASHNEXT_DRAFT_FP4 must be unset, 0, 1 or 2 (got {v:?})")),
    }
}
/// VADE_FLASHNEXT_DRAFT_VOCAB: unset = 65536 rows, 0 = the full head, otherwise a row count
/// (Head::draft_rows keeps the full head unless it is a multiple of 8 below the vocabulary).
pub fn parse_draft_vocab(value: Option<&str>) -> Result<Option<u32>, String> {
    match value {
        None => Ok(Some(DRAFT_VOCAB)),
        Some(v) => v.parse::<u32>().map(|n| (n > 0).then_some(n))
            .map_err(|_| format!("VADE_FLASHNEXT_DRAFT_VOCAB must be a row count, 0 = full head (got {v:?})")),
    }
}
/// VADE_MTP_CONF: unset = 0.4, "off" = no threshold, otherwise p in [0, 1). "off" also ends the
/// one-chat step stop (draft_steps.rs), which reads p, and a chat alone keeps the width chooser.
pub fn parse_mtp_conf(value: Option<&str>) -> Result<Option<f32>, String> {
    match value {
        None => Ok(Some(MTP_CONF)),
        Some("off") => Ok(None),
        Some(v) => v.parse::<f32>().ok().filter(|t| (0.0..1.0).contains(t)).map(Some)
            .ok_or(format!("VADE_MTP_CONF must be unset, off or a number in [0, 1) (got {v:?})")),
    }
}
/// VADE_FLASHNEXT_MTP_DEPTH: unset = 7 on one Spark and on the pipeline; explicit 1..=15.
pub fn parse_resident_depth(value: Option<&str>, pipeline: bool) -> Result<usize, String> {
    match value {
        None => Ok(if pipeline { PIPELINE_MTP_DEPTH } else { RESIDENT_MTP_DEPTH }),
        Some(v) => v.parse::<usize>().ok().filter(|d| (1..=15).contains(d))
            .ok_or(format!("VADE_FLASHNEXT_MTP_DEPTH must be 1..=15 (got {v:?})")),
    }
}

fn var(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name}: {e}")),
    }
}
/// One ON_SWITCHES entry, read now (callers cache it).
pub fn switch_on(name: &str) -> Result<bool, String> {
    var(name).and_then(|v| parse_on(name, v.as_deref()))
}
pub fn draft_fp4() -> Result<u8, String> {
    static M: OnceLock<Result<u8, String>> = OnceLock::new();
    M.get_or_init(|| var("VADE_FLASHNEXT_DRAFT_FP4").and_then(|v| parse_draft_fp4(v.as_deref()))).clone()
}
pub fn draft_vocab() -> Result<Option<u32>, String> {
    static M: OnceLock<Result<Option<u32>, String>> = OnceLock::new();
    M.get_or_init(|| var("VADE_FLASHNEXT_DRAFT_VOCAB").and_then(|v| parse_draft_vocab(v.as_deref()))).clone()
}
pub fn mtp_conf() -> Result<Option<f32>, String> {
    static M: OnceLock<Result<Option<f32>, String>> = OnceLock::new();
    M.get_or_init(|| var("VADE_MTP_CONF").and_then(|v| parse_mtp_conf(v.as_deref()))).clone()
}
pub fn resident_mtp_depth(pipeline: bool) -> Result<usize, String> {
    var("VADE_FLASHNEXT_MTP_DEPTH").and_then(|v| parse_resident_depth(v.as_deref(), pipeline))
}
/// Validate every stack switch before allocation; the effective settings for the startup log.
pub fn describe() -> Result<String, String> {
    let mut s = format!("draft_stack fp4={} conf={} vocab={}", draft_fp4()?,
        mtp_conf()?.map_or("off".to_string(), |t| t.to_string()), draft_vocab()?.map_or("full".to_string(), |n| n.to_string()));
    for name in ON_SWITCHES {
        s += &format!(" {}={}", name.trim_start_matches("VADE_").to_lowercase(), u8::from(switch_on(name)?));
    }
    Ok(s)
}

#[test]
fn unset_switches_select_the_stack_and_explicit_values_restore_the_old_default() {
    for name in ON_SWITCHES {
        assert_eq!(parse_on(name, None), Ok(true));
        assert_eq!(parse_on(name, Some("1")), Ok(true));
        assert_eq!(parse_on(name, Some("0")), Ok(false));
        assert!(parse_on(name, Some("yes")).is_err());
    }
    assert_eq!(parse_draft_fp4(None), Ok(2));
    assert_eq!(parse_draft_fp4(Some("0")), Ok(0));
    assert_eq!(parse_draft_fp4(Some("1")), Ok(1));
    assert!(parse_draft_fp4(Some("3")).is_err());
    assert_eq!(parse_draft_vocab(None), Ok(Some(65536)));
    assert_eq!(parse_draft_vocab(Some("0")), Ok(None));
    assert_eq!(parse_draft_vocab(Some("32768")), Ok(Some(32768)));
    assert!(parse_draft_vocab(Some("all")).is_err());
    assert_eq!(parse_mtp_conf(None), Ok(Some(0.4)));
    assert_eq!(parse_mtp_conf(Some("off")), Ok(None));
    assert_eq!(parse_mtp_conf(Some("0")), Ok(Some(0.0)));
    assert!(parse_mtp_conf(Some("1")).is_err());
    assert_eq!(parse_resident_depth(None, false), Ok(7));
    assert_eq!(parse_resident_depth(None, true), Ok(7));
    assert_eq!(parse_resident_depth(Some("3"), false), Ok(3));
    assert!(parse_resident_depth(Some("0"), false).is_err());
    assert!(parse_resident_depth(Some("16"), true).is_err());
    // Requests default to depth 3; at or above it the adaptive chooser may draft up to resident 7.
    assert!(crate::flash_next_lookup::DEFAULT_MTP_DEPTH <= RESIDENT_MTP_DEPTH);
    assert!(crate::flash_next_lookup::DEFAULT_MTP_DEPTH <= PIPELINE_MTP_DEPTH);
}
