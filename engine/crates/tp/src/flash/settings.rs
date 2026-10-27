//! Flash startup and checkpoint generation settings; explicit fields take precedence.
use engine_model::json::{self, Value};
use spark_engine::sample::Sampling;
use std::path::Path;

/// Installer-only mode: compile modules and optionally verify an existing model.
pub fn warm_ptx_arg(args: &[String]) -> Result<Option<(&str, Option<&str>)>, String> {
    if args.first().map(String::as_str) != Some("--warm-ptx") { return Ok(None); }
    if !(2..=3).contains(&args.len()) || args[1..].iter().any(String::is_empty) {
        return Err("usage: flash-serve --warm-ptx PTX [MODEL]".into());
    }
    Ok(Some((&args[1], args.get(2).map(String::as_str))))
}

#[test]
fn cache_warmup_requires_only_a_ptx_path() {
    assert_eq!(warm_ptx_arg(&["--warm-ptx".into(), "/tmp/flash.ptx".into()]), Ok(Some(("/tmp/flash.ptx", None))));
    assert_eq!(warm_ptx_arg(&["--warm-ptx".into(), "ptx".into(), "/models/existing".into()]), Ok(Some(("ptx", Some("/models/existing")))));
    assert!(warm_ptx_arg(&["--warm-ptx".into()]).is_err());
    assert!(warm_ptx_arg(&["--warm-ptx".into(), "".into()]).is_err());
    assert!(warm_ptx_arg(&["--warm-ptx".into(), "a".into(), "".into()]).is_err());
    assert!(warm_ptx_arg(&["--warm-ptx".into(), "a".into(), "b".into(), "c".into()]).is_err());
    assert_eq!(warm_ptx_arg(&["0".into(), "1".into()]), Ok(None));
}

/// Preserve the explicit positional context while accepting the product default
/// in RANK WORLD LISTEN NEXT PTX MODEL BIND REQUESTS [options]. A socket address
/// cannot be confused with the numeric context in the original launch form.
pub fn startup_args(mut args: Vec<String>) -> Vec<String> {
    if args.get(6).is_some_and(|value| value.parse::<std::net::SocketAddr>().is_ok()) {
        args.insert(6, spark_tp::flash_pool::DEFAULT_CONTEXT_TOKENS.to_string());
    }
    args
}

/// Strict two-token drafting is the single-chat default; zero is explicit plain.
pub fn mtp_depth(value: Option<&str>) -> Result<usize, String> {
    let depth = value.map(|value| value.parse::<usize>().map_err(|_| "invalid MTP depth"))
        .transpose()?.unwrap_or(2);
    if depth > 3 { return Err("MTP depth must be 0..=3".into()); }
    Ok(depth)
}

/// The lookup tuning default applies only to world-one Next. Explicit choices
/// win; FP8 retains k2 and rank Next retains plain-only execution.
pub fn serving_mtp(next: bool, world: usize, explicit: Option<usize>) -> usize {
    if next && matches!(world, 2 | 4) {
        0
    } else {
        explicit.unwrap_or(if next {
            spark_engine::flash_next_lookup::DEFAULT_MTP_DEPTH
        } else {
            2
        })
    }
}

/// FP8 rank path, no explicit depth, VADE_MTP_ADAPT unset/1: the resident MTP depth, the
/// ceiling of the per-cycle chooser one chat drafts with.
pub const RANK_MTP_DEPTH: usize = 7;

/// The FP8 rank path's resident depth and whether one chat chooses its draft depth each
/// cycle from its measured acceptance and row costs (`adapt`: VADE_MTP_ADAPT unset or 1).
/// An explicit depth, VADE_MTP_ADAPT=0/full or plain serving keep the serving depth at full
/// width. VADE_FLASH_MTP_DEPTH (1..=7) sets the chooser's ceiling; it is always validated.
pub fn rank_mtp(serving: usize, explicit: bool, adapt: bool, ceiling: Option<&str>) -> Result<(usize, bool), String> {
    let depth = match ceiling {
        None => RANK_MTP_DEPTH,
        Some(v) => v.parse::<usize>().ok().filter(|n| (1..=7).contains(n))
            .ok_or(format!("VADE_FLASH_MTP_DEPTH must be 1..=7 (got {v:?})"))?,
    };
    Ok(if explicit || !adapt || serving == 0 { (serving, false) } else { (depth, true) })
}

/// The layer pipeline with MTP (world 2 by default: `pipeline`, `pipeline_mtp`) serves world 2
/// as the layer pipeline of the world-1 engine with MTP, so an explicit depth there is checked
/// as world 1 checks it (0..=3, already enforced by mtp_depth) instead of refused as rank Next.
pub fn startup_mtp(pipeline_mtp: bool, explicit: Option<usize>) -> Option<usize> {
    explicit.filter(|_| !pipeline_mtp)
}

/// World size that selects the pipeline by default. Larger chains are opt-in.
pub const PIPELINE_RANKS: usize = 2;

/// VADE_W2_PP (`switch`: None when unset): world 2 FP4-Local serves as the two-rank layer
/// pipeline of the world-1 engine unless VADE_W2_PP=0 (the rank path). Every other world keeps
/// unset = off: world 4 keeps the rank path; explicit 1 selects an N-stage cable chain.
/// World 1 never builds a pipeline because its peer set is empty.
pub fn pipeline(world: usize, switch: Option<bool>) -> bool {
    switch.unwrap_or(world == PIPELINE_RANKS)
}

/// Keep the existing explicit-depth refusal when MTP is disabled, for every chain size.
pub fn pipeline_depth(pipeline: bool, mtp: bool, explicit: Option<usize>) -> Result<(), String> {
    if pipeline && !mtp && explicit.is_some_and(|n| n > 0) {
        return Err("VADE_W2_PP_MTP=0 requires --mtp 0 or no explicit MTP depth".into());
    }
    Ok(())
}

/// VADE_W2_PP_MTP on the pipeline: unset = MTP drafts verified across it, 0 = plain decoding.
pub fn pipeline_mtp(switch: Option<bool>) -> bool {
    switch.unwrap_or(true)
}

/// Validate both switches before topology selection, including a local process.
/// Return pipeline enabled, pipeline drafting, and whether this rank has peers.
pub fn pipeline_recipe(
    mixed: bool,
    rank: usize,
    world: usize,
    mut read: impl FnMut(&str) -> Result<Option<bool>, String>,
) -> Result<(bool, bool, bool), String> {
    let requested = pipeline(world, read("VADE_W2_PP")?);
    let drafting = pipeline_mtp(read("VADE_W2_PP_MTP")?);
    let distributed = (0..world).any(|peer| peer != rank);
    let enabled = mixed && requested && distributed;
    Ok((enabled, enabled && drafting, distributed))
}

/// The pipeline's companions and their defaults, the measured 2-Spark recipe: MTP verify rows
/// cross the pipeline, non-final prompt tiles are queued, decode steps ride the RDMA wire,
/// 2048-row prompt tiles, and the front projects the bounded head (one Spark funds it from
/// 8192-row tile scratch; the front's 2048-row tiles already leave it more pool than one
/// Spark keeps; ranks without the head never load it). A pipeline rank pins every unset one
/// at startup (before anything reads them); a set one ("0", another tile) stays as given.
pub const PIPELINE_DEFAULTS: [(&str, &str); 5] = [
    ("VADE_W2_PP_MTP", "1"),
    ("VADE_W2_PP_ASYNC", "1"),
    ("VADE_W2_PP_RDMA", "1"),
    ("VADE_PROMPT_TILE", "2048"),
    ("VADE_HEAD_BOUNDS", "1"),
];

/// The companions a pipeline rank pins: those `set` reports unset, at their defaults.
pub fn pipeline_pins(set: impl Fn(&str) -> bool) -> Vec<(&'static str, &'static str)> {
    PIPELINE_DEFAULTS.into_iter().filter(|&(name, _)| !set(name)).collect()
}

/// VADE_W2_PATH (a debug switch; it goes when the layer pipeline's code is deleted): unset = the
/// tensor split, the one 2-Spark path; `pipeline` = the layer pipeline of the world-1 engine.
pub fn debug_pipeline(value: Result<String, std::env::VarError>) -> Result<bool, String> {
    match value.as_deref() {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok("pipeline") => Ok(true),
        _ => Err("VADE_W2_PATH must be unset or pipeline".into()),
    }
}

/// World 2 FP4-Local serves the two-rank tensor split (spark_engine flash::next_batched tp): the
/// pipeline's links, recipe and session protocol with rank 1 starting at layer 2, and both ranks
/// splitting each routed expert by columns. It is the default there; the layer pipeline stays
/// only for VADE_W2_PATH=pipeline (`pipeline_path`), VADE_W2_TP=0, an explicit VADE_W2_PP, and
/// VADE_W2_PP_ASYNC=0 or VADE_W2_PP_RDMA=0 (the split needs both). An explicit VADE_W2_TP=1 is
/// refused at any other world, with VADE_W2_PATH=pipeline and with any of those pipeline switches.
pub fn tensor_split(
    mixed: bool,
    world: usize,
    pipeline_path: bool,
    mut read: impl FnMut(&str) -> Result<Option<bool>, String>,
) -> Result<bool, String> {
    let explicit = match read("VADE_W2_TP")? {
        Some(false) => return Ok(false),
        Some(true) if pipeline_path => return Err("VADE_W2_TP=1 is refused together with VADE_W2_PATH=pipeline".into()),
        Some(true) => true,
        None if pipeline_path || !mixed || world != PIPELINE_RANKS => return Ok(false),
        None => false,
    };
    if !mixed || world != PIPELINE_RANKS {
        return Err("VADE_W2_TP=1 needs FP4-Local at world 2".into());
    }
    if !matches!(read("VADE_W2_PP"), Ok(None)) {
        return if explicit { Err("VADE_W2_TP=1 is refused together with an explicit VADE_W2_PP".into()) } else { Ok(false) };
    }
    // The split posts every pass through the pipeline's asynchronous queue and its exchanges
    // ride the RDMA wire: an explicit 0 for either keeps the pipeline, or is refused with
    // VADE_W2_TP=1, before any rank loads.
    for name in ["VADE_W2_PP_ASYNC", "VADE_W2_PP_RDMA"] {
        if read(name)? == Some(false) {
            return if explicit { Err(format!("VADE_W2_TP=1 needs {name} (leave it unset)")) } else { Ok(false) };
        }
    }
    Ok(true)
}

/// The tensor split's extra pin (with PIPELINE_DEFAULTS): rank 1 starts at layer 2, the
/// first layer the pipeline lets a back own (layer 1 consumes the frontend's PLE).
pub const TENSOR_PINS: [(&str, &str); 1] = [("VADE_W2_PP_SPLIT", "2")];

/// The tensor split's levels, every proven one on. A split rank pins each unset one at startup;
/// a set one stays as given. These names are temporary: each goes when its level is the only code.
pub const SPLIT_LEVELS: [(&str, &str); 4] = [
    ("VADE_W2_TP", "1"),
    ("VADE_W2_TP_WIRE", "write"),
    ("VADE_W2_TP_PROJ", "3"),
    ("VADE_W2_TP_HEAD", "1"),
];

/// The pins a tensor-split rank adds: the layer cut and the unset levels; an explicit
/// VADE_W2_PP_SPLIT is refused.
pub fn tensor_pins(tp: bool, set: impl Fn(&str) -> bool) -> Result<Vec<(&'static str, &'static str)>, String> {
    if !tp {
        return Ok(Vec::new());
    }
    if TENSOR_PINS.iter().any(|&(name, _)| set(name)) {
        return Err("the tensor split sets VADE_W2_PP_SPLIT itself; leave it unset (VADE_W2_PATH=pipeline: the layer pipeline)".into());
    }
    Ok(TENSOR_PINS.into_iter().chain(SPLIT_LEVELS.into_iter().filter(|&(name, _)| !set(name))).collect())
}

/// The checkpoint's mixed storage contract selects NextModel, never the FP8 loader.
pub fn next_model(
    mixed: bool,
    forced: bool,
    rank: usize,
    world: usize,
    mtp: Option<usize>,
) -> Result<bool, String> {
    if forced && !mixed {
        return Err("--fp4-local requires a mixed FP4-Local checkpoint".into());
    }
    if !mixed {
        if mtp.is_some_and(|n| n > 3) {
            return Err("FP8 MTP depth must be 0..=3".into());
        }
        return Ok(false);
    }
    if !matches!(world, 1 | 2 | 4) || rank >= world {
        return Err("FP4-Local NextModel requires rank < world in 1/2/4".into());
    }
    if mtp.is_some_and(|n| n > 3) {
        return Err("MTP depth must be 0..=3".into());
    }
    if matches!(world, 2 | 4) && mtp.is_some_and(|n| n > 0) {
        return Err(
            "FP4-Local world 2/4 does not support MTP yet; use --mtp 0 and mtp_depth: 0".into(),
        );
    }
    Ok(true)
}
/// Pinned native rank policy, applied before common request admission.
/// Rank payload loading does not implement the full-verification contract yet.
pub fn rank_limits(
    limits: &mut spark_serve::Limits,
    verify_full: Option<&str>,
) -> Result<(), String> {
    match verify_full {
        None | Some("0") => (),
        Some("1") => return Err("FP4-Local world 2/4 does not support --verify-full / FLASH_VERIFY_FULL=1; full payload verification requires world 1".into()),
        _ => return Err("FLASH_VERIFY_FULL must be 0 or 1".into()),
    }
    // Preserve checkpoint defaults for every named sampling request. Only a
    // completely absent sampling policy selects greedy in the shared parser.
    limits.greedy_if_unsampled = true;
    limits.next_mtp_depth = Some(0);
    Ok(())
}
/// The world-2/4 rank path runs the sequential GDN scan: its M1 recurrence
/// (Gdn::run_projected) refuses the chunk-form state used by the
/// default. The caller pins VADE_GDN_CHUNK=0 before anything sizes or loads a
/// GDN state, so the memory plan and Gdn::load_next agree; an explicit
/// chunk-form request is refused instead of failing the first token.
pub fn rank_gdn_chunk(value: Option<&str>) -> Result<(), String> {
    match value {
        None | Some("0") => Ok(()),
        _ => Err("FP4-Local world 2/4 runs the sequential GDN scan; unset VADE_GDN_CHUNK or set it to 0".into()),
    }
}

pub fn sparse_indexer(value: &str) -> Result<bool, String> {
    match value { "sparse" => Ok(true), "dense" => Ok(false), _ => Err("--indexer must be sparse or dense".into()) }
}

pub fn tokenizer(root: &Path, limits: &spark_serve::Limits) -> Result<engine_tokenizer::Tokenizer, String> {
    let read = |name: &str| engine_model::safetensors::read_bounded(&root.join(name), 2 * 1024 * 1024)
        .map_err(|e| e.to_string());
    let value = json::parse(&read("tokenizer_config.json")?, json::Limits { bytes: 2 * 1024 * 1024, ..Default::default() })
        .map_err(|e| e.to_string())?;
    let embedded = json::field(value.object().map_err(|e| e.to_string())?, "chat_template")
        .and_then(Value::string).map_err(|e| e.to_string())?;
    if root.join("chat_template.jinja").try_exists().map_err(|e| e.to_string())? {
        let template = read("chat_template.jinja")?;
        let template = std::str::from_utf8(&template).map_err(|e| e.to_string())?;
        if spark_serve::contract::normalize_template(embedded) != spark_serve::contract::normalize_template(template) {
            return Err("checkpoint chat_template.jinja differs from tokenizer configuration".into());
        }
    }
    let mut tokenizer = engine_tokenizer::Tokenizer::from_directory(root).map_err(|e| e.to_string())?;
    tokenizer.set_limits(limits.tokenizer_limits()).map_err(|e| e.to_string())?;
    Ok(tokenizer)
}

pub struct Generation {
    pub sampling: Sampling,
    pub eos: Vec<u32>,
}
impl Generation {
    pub fn open(root: &Path, eos: u32, vocab: usize) -> Result<Self, String> {
        let bytes = engine_model::safetensors::read_bounded(&root.join("generation_config.json"), 1024 * 1024)
            .map_err(|e| e.to_string())?;
        let value = json::parse(&bytes, json::Limits::default()).map_err(|e| e.to_string())?;
        let object = value.object().map_err(|e| e.to_string())?;
        let mut sampling = Sampling { temperature: 1.0, top_k: 50, ..Sampling::greedy() };
        for (name, target) in [("temperature", &mut sampling.temperature), ("top_p", &mut sampling.top_p),
            ("repetition_penalty", &mut sampling.repetition_penalty)] {
            if let Some(value) = object.get(name) { *target = value.f64().map_err(|e| e.to_string())?; }
        }
        if let Some(value) = object.get("top_k") {
            sampling.top_k = usize::try_from(value.u64().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        }
        if !object.get("do_sample").map(Value::boolean).transpose().map_err(|e| e.to_string())?.unwrap_or(false) {
            sampling.temperature = 0.0;
        }
        sampling.validate()?;
        let eos = match object.get("eos_token_id") {
            None => vec![eos],
            Some(value) => {
                let values = match value { Value::Array(values) => values.clone(), _ => vec![value.clone()] };
                values.iter().map(|value| value.u64().map_err(|e| e.to_string())
                    .and_then(|n| u32::try_from(n).map_err(|e| e.to_string()))).collect::<Result<Vec<_>, _>>()?
            }
        };
        if eos.is_empty() || eos.iter().any(|&id| id as usize >= vocab) { return Err("invalid checkpoint EOS IDs".into()); }
        Ok(Self { sampling, eos })
    }
}

/// Rank-zero command ledger. Validation is read-only and precedes transport.
#[derive(Clone, Default)]
pub struct CommandState {
    pub sessions: std::collections::BTreeMap<u64, (usize, usize, Option<usize>)>,
    pub saved: std::collections::BTreeMap<u64, (usize, usize)>,
}
impl CommandState {
    pub fn validate(&self, op: u64, id: u64, arg: u64, logits: bool, tokens: &[u32],
        context: usize, vocab: usize, slots: usize, depth: usize) -> Result<(), String> {
        let count = tokens.len();
        let session = self.sessions.get(&id);
        let input = session.is_some_and(|&(position, capacity, _)| count > 0
            && position.checked_add(count).is_some_and(|end| end <= capacity && end <= context)
            && tokens.iter().all(|&t| (t as usize) < vocab));
        let empty = count == 0;
        let valid = match op {
            1 => empty && !logits && id != 0 && !self.sessions.contains_key(&id)
                && !self.saved.contains_key(&id) && self.sessions.len() < slots
                && arg >= 64 && arg <= context as u64,
            2 => input,
            3 => input && count == 1,
            4 | 10 | 14 | 16 => empty && session.is_some(),
            5 => empty && self.sessions.is_empty(),
            6 => {
                let mut seen = std::collections::BTreeSet::new();
                id == 0 && !logits && count >= 8 && count <= slots * 4 && count % 4 == 0
                    && tokens.chunks_exact(4).all(|row| {
                        let id = row[0] as u64 | ((row[1] as u64) << 32);
                        seen.insert(id) && (row[2] as usize) < vocab && row[3] <= 1
                            && self.sessions.get(&id).is_some_and(|&(position, capacity, _)| position > 0
                                && position < capacity && position < context)
                    })
            }
            7 => count == 2 && id != 0 && self.sessions.contains_key(&arg)
                && !self.saved.contains_key(&id) && !self.sessions.contains_key(&id),
            8 => empty && session.is_some() && self.saved.get(&arg).is_some_and(|&(end, _)| end <= session.unwrap().1),
            9 => empty && self.saved.contains_key(&id),
            11 => empty && session.is_some_and(|&(position, capacity, _)| arg >= capacity as u64
                && arg >= position as u64 && arg <= context as u64),
            12 => input && count > 1 && count <= depth + 1,
            13 => empty && session.is_some_and(|&(position, _, entry)| arg == position as u64
                || entry.is_some_and(|entry| arg > entry as u64 && arg <= position as u64)),
            15 | 17 => session.is_some_and(|&(position, capacity, _)| position > 0
                && arg > 0 && arg <= depth as u64 && position.checked_add(arg as usize + 1).is_some_and(|end| end <= capacity)
                && (tokens.first().is_some_and(|&t| (t as usize) < vocab))
                && if op == 15 { count == 1 } else { count == 2 && tokens[1] <= 64 }),
            18 => input && count <= depth + 1 && arg <= 64,
            19 => empty && session.is_some() && arg >= 1 && arg <= (depth + 1) as u64,
            21 => empty && !logits && id == 0 && arg <= 1 && self.sessions.is_empty() && self.saved.is_empty(),
            20 => empty && !logits && arg == 0 && depth != 0 && self.sessions.len() == 1
                && session.is_some_and(|&(position, _, entry)| position == 0 && entry.is_none()),
            _ => false,
        };
        if valid { Ok(()) } else { Err("invalid_request_error: invalid Flash command or session state".into()) }
    }
    pub fn apply(&mut self, op: u64, id: u64, arg: u64, logits: bool, tokens: &[u32]) {
        match op {
            1 => { self.sessions.insert(id, (0, arg as usize, None)); }
            2 | 3 | 12 | 18 => {
                let state = self.sessions.get_mut(&id).unwrap();
                state.2 = matches!(op, 12 | 18).then_some(state.0);
                state.0 += tokens.len();
            }
            4 => { self.sessions.remove(&id); }
            6 => for row in tokens.chunks_exact(4) {
                self.sessions.get_mut(&(row[0] as u64 | ((row[1] as u64) << 32))).unwrap().0 += 1;
            },
            7 => { let end = self.sessions[&arg].0; self.saved.insert(id, (end, end)); }
            8 => { let (end, prefix) = self.saved[&arg]; let state = self.sessions.get_mut(&id).unwrap(); state.0 = if logits { prefix } else { end }; state.2 = None; }
            9 => { self.saved.remove(&id); }
            10 | 13 => { let state = self.sessions.get_mut(&id).unwrap(); if op == 13 { state.0 = arg as usize; } state.2 = None; }
            11 => { self.sessions.get_mut(&id).unwrap().1 = arg as usize; }
            _ => (),
        }
    }
}
#[cfg(test)]
#[test]
fn tensor_split_is_the_world_two_default_and_the_pipeline_needs_a_switch() {
    let env = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| -> Result<Option<bool>, String> {
        Ok(pairs.iter().find(|(n, _)| *n == name).map(|(_, v)| *v == "1"))
    };
    // No switch set: the split at world 2 FP4-Local, nothing else changes.
    assert_eq!(tensor_split(true, 2, false, env(&[])), Ok(true));
    assert_eq!(tensor_split(true, 1, false, env(&[])), Ok(false));
    assert_eq!(tensor_split(true, 4, false, env(&[])), Ok(false));
    assert_eq!(tensor_split(false, 2, false, env(&[])), Ok(false));
    // VADE_W2_PATH=pipeline, VADE_W2_TP=0, an explicit VADE_W2_PP or a pipeline switch the split
    // cannot honor keeps the layer pipeline (or the rank path).
    assert_eq!(tensor_split(true, 2, true, env(&[])), Ok(false));
    assert_eq!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "0")])), Ok(false));
    assert_eq!(tensor_split(true, 2, true, env(&[("VADE_W2_TP", "0")])), Ok(false));
    for off in [&[("VADE_W2_PP", "0")][..], &[("VADE_W2_PP_ASYNC", "0")], &[("VADE_W2_PP_RDMA", "0")]] {
        assert_eq!(tensor_split(true, 2, false, env(off)), Ok(false), "{off:?}");
    }
    assert_eq!(tensor_split(true, 2, false, env(&[("VADE_W2_PP", "1")])), Ok(false));
    assert_eq!(tensor_split(true, 2, false, env(&[("VADE_W2_PP_ASYNC", "1"), ("VADE_W2_PP_RDMA", "1")])), Ok(true));
    // The old name keeps working for the gates; its refusals stay.
    assert_eq!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "1")])), Ok(true));
    assert!(tensor_split(true, 2, true, env(&[("VADE_W2_TP", "1")])).is_err());
    assert!(tensor_split(true, 4, false, env(&[("VADE_W2_TP", "1")])).is_err());
    assert!(tensor_split(false, 2, false, env(&[("VADE_W2_TP", "1")])).is_err());
    assert!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "1"), ("VADE_W2_PP", "1")])).is_err());
    assert!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "1"), ("VADE_W2_PP", "0")])).is_err());
    assert!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "1"), ("VADE_W2_PP_ASYNC", "0")])).is_err());
    assert!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "1"), ("VADE_W2_PP_RDMA", "0")])).is_err());
    assert_eq!(tensor_split(true, 2, false, env(&[("VADE_W2_TP", "1"), ("VADE_W2_PP_ASYNC", "1")])), Ok(true));
    // Switch errors come first, whatever the topology.
    let bad = |name: &str| -> Result<Option<bool>, String> { if name == "VADE_W2_TP" { Err("VADE_W2_TP must be 0 or 1".into()) } else { Ok(None) } };
    assert!(tensor_split(true, 1, false, bad).is_err());
    use std::env::VarError::NotPresent;
    assert_eq!(debug_pipeline(Err(NotPresent)), Ok(false));
    assert_eq!(debug_pipeline(Ok("pipeline".into())), Ok(true));
    for value in ["", "split", "Pipeline", "0", "1"] { assert!(debug_pipeline(Ok(value.into())).is_err(), "{value:?}"); }
    assert!(debug_pipeline(Err(std::env::VarError::NotUnicode("x".into()))).is_err());
}

#[cfg(test)]
#[test]
fn a_split_rank_pins_the_layer_cut_and_every_unset_level() {
    let all = [("VADE_W2_PP_SPLIT", "2"), ("VADE_W2_TP", "1"), ("VADE_W2_TP_WIRE", "write"), ("VADE_W2_TP_PROJ", "3"), ("VADE_W2_TP_HEAD", "1")];
    assert_eq!(tensor_pins(false, |_| true), Ok(Vec::new()));
    assert_eq!(tensor_pins(false, |_| false), Ok(Vec::new()));
    assert_eq!(tensor_pins(true, |_| false), Ok(all.to_vec()));
    // A set level stays as given (a gate's VADE_W2_TP_PROJ=2, VADE_W2_TP_HEAD=0).
    assert_eq!(tensor_pins(true, |name| name == "VADE_W2_TP_PROJ" || name == "VADE_W2_TP_HEAD"), Ok(all[..3].to_vec()));
    assert_eq!(tensor_pins(true, |name| name.starts_with("VADE_W2_TP")), Ok(all[..1].to_vec()));
    assert!(tensor_pins(true, |name| name == "VADE_W2_PP_SPLIT").is_err());
    // The pins name only the split's switches, never a pipeline companion.
    assert!(SPLIT_LEVELS.iter().all(|(name, _)| PIPELINE_DEFAULTS.iter().all(|(other, _)| other != name)));
}

#[cfg(test)]
#[test]
fn mixed_checkpoint_selects_rows_and_rank_backends() {
    assert_eq!(mtp_depth(None).unwrap(), 2);
    for k in 0..=3 {
        assert_eq!(mtp_depth(Some(&k.to_string())).unwrap(), k);
    }
    assert!(mtp_depth(Some("4")).is_err());
    assert!(sparse_indexer("sparse").unwrap());
    assert!(!sparse_indexer("dense").unwrap());
    assert!(sparse_indexer("auto").is_err());
    for (mixed, forced, rank, world, mtp, expected) in [
        (true, false, 0, 1, None, Some(true)),
        (true, true, 0, 1, Some(0), Some(true)),
        (false, false, 0, 4, Some(2), Some(false)),
        (false, true, 0, 1, None, None),
        (true, false, 0, 2, None, Some(true)),
        (true, false, 1, 2, None, Some(true)),
        (true, true, 0, 1, Some(2), Some(true)),
    ] {
        assert_eq!(next_model(mixed, forced, rank, world, mtp).ok(), expected);
    }
    for world in [2, 4] {
        for rank in 0..world {
            assert_eq!(next_model(true, false, rank, world, Some(0)), Ok(true));
            for depth in 1..=3 {
                assert!(
                    next_model(true, false, rank, world, Some(depth))
                        .unwrap_err()
                        .contains("does not support MTP")
                );
            }
        }
    }
    for (rank, world) in [(1, 1), (2, 2), (4, 4), (0, 0), (0, 3), (0, 8)] {
        assert!(next_model(true, false, rank, world, None).is_err());
    }
}
#[test]
fn invalid_command_preserves_ring_state() {
    let mut state = CommandState::default(); state.apply(1, 1, 64, false, &[]); assert!(state.validate(7, 2, 1, false, &[0, 1], 64, 512, 8, 3).is_ok()); assert!(state.validate(7, 2, 1, false, &[], 64, 512, 8, 3).is_err());
    assert!(state.validate(3, 1, 0, false, &[512], 64, 512, 8, 3).is_err());
    assert!(state.validate(13, 1, 1, false, &[], 64, 512, 8, 3).is_err());
    assert_eq!(state.sessions[&1], (0, 64, None));
    assert!(state.validate(20, 1, 0, false, &[], 64, 512, 8, 2).is_ok());
    assert!(state.validate(20, 1, 0, false, &[], 64, 512, 8, 0).is_err());
    state.apply(1, 2, 64, false, &[]);
    assert!(state.validate(20, 1, 0, false, &[], 64, 512, 8, 2).is_err());
    state.apply(4, 2, 0, false, &[]);
    state.apply(3, 1, 0, false, &[1]);
    assert!(state.validate(20, 1, 0, false, &[], 64, 512, 8, 2).is_err());
}
#[test]
fn automatic_pool_context_and_chats() {
    let reserve = |n, caching| Ok(if caching { 787_776_888 + 69_892 * n } else { 670_697_200 + 68_096 * n });
    let request = spark_tp::flash_pool::Request::default();
    assert_eq!(request.plan(131072, 32_000_000_000, reserve, |_| Ok(8_000_000)).unwrap().pool_bytes, 32_000_000_000);
    // World two, MTP2, three graph segments; conservative 9 MB publication bound.
    for (tokens, chats) in [(131072, 1), (32768, 6)] { let plan = request.plan(tokens, 19_358_746_844, reserve, |_| Ok(9_000_000)).unwrap(); assert_eq!(plan.pool_bytes / (plan.reservation + plan.publication_bytes), chats); }
    let hint = spark_tp::flash_pool::file_cache_hint;
    assert!(hint(0, 1).is_empty()); assert!(hint((4 << 30) + 1, 1).is_empty());
    assert!(!hint((4 << 30) + 2, 1).is_empty());
    assert_eq!(hint(5_000_000_001, 1), " Reclaimable cache estimate: 4.66 GiB (5000000000 bytes); this is not a guarantee that CUDA can allocate it.");
    let error = request.plan(131072, 7_425_823_856, |n, _| Ok(if n == 131072 {19_847_557_088} else {1}), |_| Ok(0)).unwrap_err();
    for text in ["Requested size does not fit.", "18.48 GiB (19847557088 bytes)", "6.92 GiB (7425823856 bytes)", "11.57 GiB (12421733232 bytes)", "CONTEXT=131072", "OUTPUT", "FLASH_SERVE_MAX_ACTIVE", "SEQPAR_MEMORY_BYTES", "lowest rank"] { assert!(error.contains(text), "{error}"); }
    for available in [19_847_557_087, 19_847_557_088, 19_847_557_089] { assert_eq!(request.plan(131072, available, |n, _| Ok(if n == 131072 {19_847_557_088} else {1}), |_| Ok(0)).is_ok(), available >= 19_847_557_088); }
}

#[test]
fn unsampled_direct_pipeline_requests_match_world_one_token_ids() {
    use spark_engine::sample::Sampler;
    let root = std::env::temp_dir().join(format!("flash-direct-sampling-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    let _cleanup = Cleanup(root.clone());
    std::fs::write(root.join("generation_config.json"),
        br#"{"do_sample":true,"temperature":1.7,"top_k":3,"top_p":0.8,"repetition_penalty":1.2}"#).unwrap();
    let checkpoint = Generation::open(&root, 3, 4).unwrap().sampling;
    assert_eq!(checkpoint.temperature, 1.7);
    let logits = [[0., 1., 2., 1.], [0., 2., 2., 1.], [3., 1., 2., 0.], [0., 1., 2., 3.]];
    let tokens = |sampling| {
        let mut sampler = Sampler::new(sampling).unwrap();
        let mut ids = vec![1];
        for row in &logits {
            let id = sampler.sample(row, &ids, 1, ids.len(), |_| true).unwrap();
            ids.push(id);
        }
        ids[1..].to_vec()
    };
    let world_one = tokens(Sampling::greedy());
    assert_eq!(world_one, [2, 1, 0, 3]); // Includes the lowest-id argmax tie rule.
    for (world, rank_path) in [(1, false), (2, false), (2, true), (4, true)] {
        let next = next_model(true, false, 0, world, None).unwrap();
        let mut limits = spark_serve::Limits {
            default_sampling: Some(checkpoint), greedy_if_unsampled: next,
            ..Default::default()
        };
        if rank_path { rank_limits(&mut limits, None).unwrap(); }
        assert_eq!(limits.default_sampling, Some(checkpoint));
        for (chat, input) in [(false, r#""prompt_token_ids":[1]"#),
            (true, r#""messages":[{"role":"user","content":"hello"}]"#)] {
            for _ in 0..16 {
                let body = format!("{{{input},\"max_tokens\":4}}");
                let request = spark_serve::protocol::parse(body.as_bytes(), chat, &limits).unwrap();
                assert_eq!(tokens(request.sampling), world_one, "world={world} rank={rank_path}");
            }
            let body = format!("{{{input},\"max_tokens\":4,\"seed\":42}}");
            let named = spark_serve::protocol::parse(body.as_bytes(), chat, &limits).unwrap();
            let expected = Sampling { seed: 42, ..checkpoint };
            assert_eq!(named.sampling, expected, "world={world} rank={rank_path}");
            assert_eq!(tokens(named.sampling), tokens(expected));
            assert_ne!(tokens(named.sampling), world_one, "seed-only must retain checkpoint sampling");
            let body = format!("{{{input},\"max_tokens\":4,\"temperature\":0.7}}");
            let explicit = spark_serve::protocol::parse(body.as_bytes(), chat, &limits).unwrap();
            assert_eq!(explicit.sampling.temperature, 0.7);
            assert_eq!(explicit.sampling.top_k, checkpoint.top_k);
            assert_eq!(explicit.sampling.top_p, checkpoint.top_p);
            assert_eq!(explicit.sampling.repetition_penalty, checkpoint.repetition_penalty);
        }
        assert_eq!(limits.next_mtp_depth, rank_path.then_some(0));
    }
}

#[test]
fn rank_startup_keeps_checkpoint_base_and_pins_only_unsupported_features() {
    for world in [2, 4] {
        assert!(next_model(true, false, 0, world, None).unwrap());
        let sampled = Sampling { temperature: 1., top_k: 20, top_p: 0.95,
            repetition_penalty: 1.2, ..Sampling::greedy() };
        let mut limits = spark_serve::Limits { default_sampling: Some(sampled), ..Default::default() };
        rank_limits(&mut limits, None).unwrap();
        assert_eq!(limits.default_sampling, Some(sampled));
        assert!(limits.greedy_if_unsampled);
        assert_eq!(limits.next_mtp_depth, Some(0));
        let bare = spark_serve::protocol::parse(
            br#"{"prompt_token_ids":[1],"max_tokens":1}"#, false, &limits).unwrap();
        assert_eq!(bare.sampling.temperature, 0.);
        assert_eq!(bare.sampling.repetition_penalty, 1.);
        let named = spark_serve::protocol::parse(
            br#"{"prompt_token_ids":[1],"max_tokens":1,"seed":42}"#, false, &limits).unwrap();
        assert_eq!(named.sampling, Sampling { seed: 42, ..sampled });
        assert!(rank_limits(&mut limits, Some("1")).is_err());
    }
}

#[test]
fn rank_startup_refuses_unimplemented_full_payload_verification() {
    for value in [None, Some("0")] {
        rank_limits(&mut spark_serve::Limits::default(), value).unwrap();
    }
    let error = rank_limits(&mut spark_serve::Limits::default(), Some("1")).unwrap_err();
    assert!(error.contains("world 2/4"));
    assert!(error.contains("--verify-full / FLASH_VERIFY_FULL=1"));
    assert!(rank_limits(&mut spark_serve::Limits::default(), Some("yes")).is_err());
}

#[cfg(test)]
#[test]
fn rank_startup_pins_the_sequential_gdn_scan() {
    for value in [None, Some("0")] {
        rank_gdn_chunk(value).unwrap();
    }
    for value in [Some("1"), Some("yes"), Some("")] {
        assert!(rank_gdn_chunk(value).unwrap_err().contains("sequential GDN scan"));
    }
}

#[cfg(test)]
#[test]
fn next_lookup_default_preserves_fp8_and_explicit_mtp() {
    for (world, expected) in [(1, 3), (2, 0), (4, 0)] {
        assert_eq!(serving_mtp(true, world, None), expected);
        assert_eq!(serving_mtp(false, world, None), 2);
    }
    for depth in 0..=3 {
        for (world, expected) in [(1, depth), (2, 0), (4, 0)] {
            assert_eq!(serving_mtp(false, world, Some(depth)), depth);
            assert_eq!(serving_mtp(true, world, Some(depth)), expected);
        }
    }
}

#[cfg(test)]
#[test]
fn pipeline_mtp_accepts_explicit_depths() {
    for depth in 0..=3 {
        for rank in 0..2 {
            assert_eq!(next_model(true, false, rank, 2, startup_mtp(true, Some(depth))), Ok(true));
            assert_eq!(next_model(true, false, rank, 2, startup_mtp(false, Some(depth))).is_ok(), depth == 0);
            assert_eq!(next_model(true, false, rank, 4, startup_mtp(false, Some(depth))).is_ok(), depth == 0);
        }
        assert_eq!(serving_mtp(true, 1, Some(depth)), depth);
    }
    assert_eq!(startup_mtp(true, None), None);
    assert_eq!(startup_mtp(false, Some(2)), Some(2));
}

#[cfg(test)]
#[test]
fn world_two_defaults_to_the_pipeline_with_mtp() {
    // VADE_W2_PP unset: the pipeline at world 2 only; "0" is the rank path everywhere, "1" asks
    // for the pipeline (run_pp checks N-stage geometry; world 1 never reaches it).
    for (world, unset) in [(1, false), (2, true), (4, false)] {
        assert_eq!(pipeline(world, None), unset);
        assert!(!pipeline(world, Some(false)));
        assert!(pipeline(world, Some(true)));
    }
    // VADE_W2_PP_MTP unset: MTP across the pipeline; "0" plain decoding.
    assert!(pipeline_mtp(None));
    assert!(pipeline_mtp(Some(true)));
    assert!(!pipeline_mtp(Some(false)));
    // Every unset companion is pinned at the measured recipe; a set one is kept as given.
    let all = [("VADE_W2_PP_MTP", "1"), ("VADE_W2_PP_ASYNC", "1"), ("VADE_W2_PP_RDMA", "1"), ("VADE_PROMPT_TILE", "2048"),
        ("VADE_HEAD_BOUNDS", "1")];
    assert_eq!(pipeline_pins(|_| false), all);
    assert_eq!(pipeline_pins(|name| name == "VADE_W2_PP_MTP"), all[1..]);
    assert_eq!(pipeline_pins(|name| name == "VADE_PROMPT_TILE"), [&all[..3], &all[4..]].concat());
    assert_eq!(pipeline_pins(|name| name == "VADE_HEAD_BOUNDS"), all[..4]);
    assert!(pipeline_pins(|_| true).is_empty());
    // The pinned MTP companion agrees with pipeline_mtp's default.
    assert_eq!(pipeline_mtp(None), all.contains(&("VADE_W2_PP_MTP", "1")));
    for depth in 0..=3 {
        for rank in 0..2 {
            // Default world 2: an explicit depth is checked as world 1 checks it.
            let mtp = pipeline(2, None) && pipeline_mtp(None);
            assert_eq!(next_model(true, false, rank, 2, startup_mtp(mtp, Some(depth))), Ok(true));
            // "0" escapes: VADE_W2_PP=0 (rank path) or VADE_W2_PP_MTP=0 refuse depths 1..=3 as before.
            for mtp in [pipeline(2, Some(false)) && pipeline_mtp(None), pipeline(2, None) && pipeline_mtp(Some(false))] {
                assert_eq!(next_model(true, false, rank, 2, startup_mtp(mtp, Some(depth))).is_ok(), depth == 0);
            }
            // World 4 keeps the rank path and its refusal.
            let mtp = pipeline(4, None) && pipeline_mtp(None);
            assert_eq!(next_model(true, false, rank, 4, startup_mtp(mtp, Some(depth))).is_ok(), depth == 0);
        }
    }
    // Serving depth: the pipeline with MTP serves as world 1 (default 3, explicit kept); the
    // escapes serve plain like the rank path.
    assert_eq!(serving_mtp(true, 1, None), spark_engine::flash_next_lookup::DEFAULT_MTP_DEPTH);
    assert_eq!(serving_mtp(true, 2, None), 0);
}

#[test]
fn larger_chains_are_opt_in_and_plain_depth_refusal_is_preserved() {
    for world in [3, 4, 6, 24] {
        assert!(!pipeline(world, None));
        assert!(!pipeline(world, Some(false)));
        assert!(pipeline(world, Some(true)));
    }
    for depth in 1..=3 {
        assert!(pipeline_depth(true, false, Some(depth)).is_err());
        assert!(pipeline_depth(true, true, Some(depth)).is_ok());
        assert!(pipeline_depth(false, false, Some(depth)).is_ok());
    }
    assert!(pipeline_depth(true, false, None).is_ok());
    assert!(pipeline_depth(true, false, Some(0)).is_ok());
}

#[test]
fn pipeline_switch_errors_are_checked_before_topology_selection() {
    for world in [1, 2, 3, 4, 24] {
        for rank in 0..world {
            for mixed in [false, true] {
                for invalid in ["VADE_W2_PP", "VADE_W2_PP_MTP"] {
                    for selected in [Some(false), Some(true), None] {
                        let error = pipeline_recipe(mixed, rank, world, |name| {
                            if name == invalid { Err(format!("{name} must be 0 or 1")) }
                            else { Ok(selected) }
                        }).unwrap_err();
                        assert_eq!(error, format!("{invalid} must be 0 or 1"));
                    }
                }
            }
        }
    }
    assert_eq!(pipeline_recipe(true, 0, 1, |_| Ok(Some(true))).unwrap(), (false, false, false));
    for world in [2, 3, 4, 24] {
        for rank in 0..world {
            assert_eq!(pipeline_recipe(true, rank, world, |_| Ok(Some(true))).unwrap(), (true, true, true));
            assert_eq!(pipeline_recipe(true, rank, world, |_| Ok(Some(false))).unwrap(), (false, false, true));
        }
    }
}

#[cfg(test)]
#[test]
fn rank_mtp_adapts_only_the_default_depth() {
    // Unset: the resident ceiling, chosen per cycle; VADE_FLASH_MTP_DEPTH moves the ceiling.
    assert_eq!(rank_mtp(2, false, true, None), Ok((RANK_MTP_DEPTH, true)));
    for depth in 1..=7 { assert_eq!(rank_mtp(2, false, true, Some(&depth.to_string())), Ok((depth, true))); }
    // Explicit depths, VADE_MTP_ADAPT=0/full and plain serving keep the serving depth at full width.
    for serving in 0..=3 {
        assert_eq!(rank_mtp(serving, true, true, None), Ok((serving, false)));
        assert_eq!(rank_mtp(serving, false, false, Some("5")), Ok((serving, false)));
    }
    assert_eq!(rank_mtp(0, false, true, None), Ok((0, false)));
    for bad in ["0", "8", "", "x", "-1"] {
        assert!(rank_mtp(2, false, true, Some(bad)).is_err());
        assert!(rank_mtp(2, true, false, Some(bad)).is_err());
    }
}
