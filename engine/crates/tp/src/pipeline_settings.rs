//! Pipeline numerical recipe before either rank loads weights. Prompt options
//! use the owning parser's configured booleans: unset equals explicit default.
//! Cohort settings come from the executable's already-parsed route recipe.
//! The prefix cache also compares its effective value: unset is on for the
//! tensor split and off for the layer chain.
//! Other settings retain raw presence/value identities. Rank-local frontend
//! suppression is applied after verification and is not part of this recipe.
//! Paths, authentication, endpoints, logging and memory ceilings are excluded.
use std::ffi::OsString;
use spark_serve::mtp_batch::{Experiments, Mode};

const COHORT_NAMES: [&str; 6] = ["pipeline cohort guard (route)", "VADE_MTP_BATCH/VADE_MTP_C8 (effective)",
    "VADE_POOL_GRAPH_FALLBACK", "VADE_MTP_COHORT_ADAPT", "VADE_C8_GUARD_ROBUST", "VADE_MTP_COHORT"];
const PREFIX_CACHE: &str = "VADE_FLASHNEXT_PREFIX_CACHE";

pub(super) fn names() -> Vec<&'static str> {
    // flash/settings.rs pipeline companions; model/flash and engine/flash load,
    // QSA/GDN/head/verify dispatch, and serve/flashnext + mtp_batch policies.
    let mut names = vec![
        "FLASH_NVFP4", "FLASH_BF16_EXPERTS", "FLASH_VERIFY_FULL", "FLASH_MTP_SAMPLED",
        "SPARK_KV_BF16", "SPARK_ADAPTIVE_PREFILL",
        "VADE_PROMPT_TILE", PREFIX_CACHE,
        "VADE_FLASHNEXT_MTP_DEPTH", "VADE_FLASHNEXT_MX8", "VADE_FLASHNEXT_DRAFT_MX8",
        "VADE_FLASHNEXT_DRAFT_FP4", "VADE_FLASHNEXT_DRAFT_VOCAB", "VADE_MTP_CONF",
        "VADE_QSA_TC", "VADE_GDN_CHUNK", "VADE_GDN_PREP_SPLIT", "VADE_GDN_VERIFY_COPY",
        "VADE_GDN_COHORT", "VADE_GDN_COHORT_COPIES", "VADE_GDN_COHORT_KERNELS",
        "VADE_SESSION_ARENA", "VADE_VERIFY_ROWS", "VADE_VERIFY_POOL_NODE", "VADE_HEAD_LBF12",
        "VADE_MTP_HEAD32", "VADE_MTP_GRAPH", "VADE_BW_PDL", "VADE_MTP_TILE_ALL",
        "VADE_MTP_OBSERVE_ROWS", "VADE_MTP_DEFER_FEED", "VADE_MTP_LOOKUP",
        "VADE_ONE_ROW_POOLED", "VADE_MTP_ADAPT",
        "VADE_MTP_SAMPLED", "VADE_MTP_SAMPLED_TOPK", "VADE_COHORT_EVEN_ROWS",
        "VADE_BATCH_PREFILL", "VADE_BATCH_PREFILL_MAX_TOKENS", "VADE_TEST_ROUTING_REJECT",
    ];
    // Reuse the owning modules' lists so new load/verify flags join the identity.
    names.extend(COHORT_NAMES);
    names.extend(spark_engine::flash_next_options::Options::NAMES);
    names.extend(spark_engine::flash_prompt_options::Options::NAMES);
    names.extend(spark_engine::flash_draft_stack::ON_SWITCHES);
    names
}

/// `prefix_restore`: this topology has a prefix restore (the tensor split), so an unset cache
/// switch means on; the layer chain resolves it as off (prefix_switch::prefix_cache).
pub(super) fn fingerprints(experiments: Experiments, prefix_restore: bool, mut read: impl FnMut(&str) -> Option<OsString>) -> Result<Vec<u8>, String> {
    let bit = |value| if value { "1" } else { "0" };
    let mode = match experiments.batch_mode.ok_or("pipeline identity requires parsed startup cohort settings")? {
        Mode::Auto => "auto", Mode::Auto8 => "auto8", Mode::Off => "off", Mode::On => "on",
    };
    let cohort = [bit(experiments.pipeline_c8), mode, bit(experiments.graph_fallback),
        bit(experiments.cohort_cost), bit(experiments.guard_robust), if experiments.wide_tick { "wide" } else { "off" }];
    use spark_engine::flash_prompt_options::Options;
    let prompt = Options::parse(|name| read(name).map(|v| v.into_string()
        .map_err(|_| format!("{name} must be 0 or 1"))).transpose())?.configured_values();
    let mut bytes = Vec::new();
    for name in names() {
        if let Some(i) = COHORT_NAMES.iter().position(|v| *v == name) {
            bytes.extend_from_slice(super::sha256::digest(cohort[i].as_bytes()).as_bytes());
        } else if let Some(i) = Options::NAMES.iter().position(|v| *v == name) {
            let value = if prompt[i] { b"1" } else { b"0" };
            bytes.extend_from_slice(super::sha256::digest(value).as_bytes());
        } else if name == PREFIX_CACHE {
            let value = read(name);
            let value = match value.as_deref() {
                None if prefix_restore => b"1",
                None => b"0",
                Some(v) if v == "0" => b"0",
                Some(v) if v == "1" => b"1",
                _ => return Err(format!("{name} must be 0 or 1")),
            };
            bytes.extend_from_slice(super::sha256::digest(value).as_bytes());
        } else {
            match read(name) {
                None => bytes.extend_from_slice(&[0; 64]),
                Some(value) => bytes.extend_from_slice(super::sha256::digest(value.as_encoded_bytes()).as_bytes()),
            }
        }
    }
    Ok(bytes)
}

/// Parsed identities can describe both values without transmitting environment strings.
pub(super) fn describe(name: &str, value: &[u8]) -> String {
    if COHORT_NAMES.contains(&name) {
        for word in ["0", "1", "auto", "auto8", "off", "on"] {
            if value == super::sha256::digest(word.as_bytes()).as_bytes() { return word.into(); }
        }
    }
    if name == PREFIX_CACHE || spark_engine::flash_prompt_options::Options::NAMES.contains(&name) {
        for (word, description) in [(b"0", "0 (off)"), (b"1", "1 (on)")] {
            if value == super::sha256::digest(word).as_bytes() { return description.into(); }
        }
    }
    if value.iter().all(|v| *v == 0) { return "unset".into(); }
    if value.len() == 64 && value.iter().all(u8::is_ascii_hexdigit) {
        return format!("sha256:{}", String::from_utf8_lossy(value));
    }
    "invalid identity row".into()
}

#[cfg(test)]
#[test]
fn cohort_identity_uses_the_frozen_recipe_without_rereading_its_switches() {
    let parsed = Experiments::startup_with(2, true, |_| Ok(None)).unwrap();
    let identity = fingerprints(parsed, true, |name| {
        assert!(!["VADE_MTP_BATCH", "VADE_MTP_C8", "VADE_POOL_GRAPH_FALLBACK",
            "VADE_MTP_COHORT_ADAPT", "VADE_C8_GUARD_ROBUST", "VADE_MTP_COHORT"].contains(&name), "reread {name}");
        None
    }).unwrap();
    assert_eq!(identity.len(), names().len() * 64);
    assert!(fingerprints(Experiments::default(), true, |_| None).unwrap_err().contains("parsed startup"));
    let cache_identity = |split: bool, value: Option<&str>| {
        let mut bytes = vec![b'a'; super::identity_names(2).len() * 64];
        bytes.extend(fingerprints(parsed, split, |name| if name == PREFIX_CACHE { value.map(Into::into) } else { None }).unwrap());
        bytes
    };
    // Unset is on for the tensor split and off for the layer chain, as each rank resolves it.
    for (split, left, right, accepted) in [(true, None, Some("1"), true), (true, None, Some("0"), false),
        (false, None, Some("0"), true), (false, None, Some("1"), false),
        (true, Some("0"), Some("1"), false), (false, Some("0"), Some("0"), true)] {
        let (left, right) = (cache_identity(split, left), cache_identity(split, right));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut front = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut back, _) = listener.accept().unwrap();
        std::thread::scope(|scope| {
            let limit = std::time::Duration::from_secs(2);
            let peer = scope.spawn(move || super::exchange_pipeline_identity(&mut back, 2, &right, limit));
            for result in [super::exchange_pipeline_identity(&mut front, 2, &left, limit), peer.join().unwrap()] {
                if accepted { assert!(result.is_ok(), "{result:?}"); }
                else { let error = result.unwrap_err(); assert!(error.contains(PREFIX_CACHE) && error.contains("refusing startup"), "{error}"); }
            }
        });
    }
    assert!(fingerprints(parsed, true, |name| (name == PREFIX_CACHE).then(|| "yes".into())).unwrap_err().contains(PREFIX_CACHE));
}
