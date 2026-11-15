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

/// The pipeline service serves with the world-1 engine's MTP (the tensor split, the layer chain),
/// so an explicit depth there is checked as world 1 checks
/// it (0..=3, already enforced by mtp_depth) instead of refused as rank Next.
pub fn startup_mtp(pipeline: bool, explicit: Option<usize>) -> Option<usize> {
    explicit.filter(|_| !pipeline)
}

/// World size of the pair, where the tensor split runs every level (it serves on the pipeline service).
pub const PIPELINE_RANKS: usize = 2;

/// The engine chooses the path; no switch selects it. FP4-Local with peers serves on the
/// pipeline service: the tensor split at world 2 and over more ranks that tensor_split accepts,
/// else the layer chain at three or more ranks.
/// `--tensor-map` (`rank_path`) selects the older rank path instead, which serves world 4 only
/// (tensor_split refuses it at world 2). World 1 never builds a pipeline because its peer set
/// is empty. Return pipeline enabled and whether this rank has peers.
pub fn pipeline_recipe(mixed: bool, rank: usize, world: usize, rank_path: bool) -> (bool, bool) {
    let distributed = (0..world).any(|peer| peer != rank);
    (mixed && !rank_path && distributed, distributed)
}

/// The pipeline's companions and their defaults: 2048-row prompt tiles, and the front projects
/// the bounded head (one Spark funds it from 8192-row tile scratch; the front's 2048-row tiles
/// already leave it more pool than one Spark keeps; ranks without the head never load it). A
/// pipeline rank pins every unset one at startup (before anything reads them); a set one
/// ("0", another tile) stays as given. MTP verify rows always cross the pipeline and non-final
/// prompt tiles are always queued.
pub const PIPELINE_DEFAULTS: [(&str, &str); 2] = [
    ("VADE_PROMPT_TILE", "2048"),
    ("VADE_HEAD_BOUNDS", "1"),
];

/// The tensor split's prompt workspace (rows) instead of the pipeline's 2048: a cold prompt of up
/// to twice this many rows runs as the token split's two parts (spark_engine
/// flash_tp_switch::seq_first), each at most one workspace. It funds one chat fewer at context
/// 32768 (8 instead of 9) for a cold 8K first token of about 2.20 s instead of 2.79 s.
pub const SPLIT_PROMPT_TILE: &str = "4096";

/// The companions a pipeline rank pins: those `set` reports unset, at their defaults (the tensor
/// split's prompt tile is SPLIT_PROMPT_TILE).
pub fn pipeline_pins(tp: bool, set: impl Fn(&str) -> bool) -> Vec<(&'static str, &'static str)> {
    PIPELINE_DEFAULTS.into_iter().map(|(name, value)| (name, if tp && name == "VADE_PROMPT_TILE" { SPLIT_PROMPT_TILE } else { value }))
        .filter(|&(name, _)| !set(name)).collect()
}

/// The transport between Sparks, SEQPAR_RDMA: unset, 1 or auto is RDMA, 0 is TCP. The
/// pipeline's polled wire follows it, and the tensor split needs RDMA.
pub fn rdma(value: Result<String, std::env::VarError>) -> Result<bool, String> {
    match value.as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("1" | "auto") => Ok(true),
        Ok("0") => Ok(false),
        _ => Err("SEQPAR_RDMA must be unset, auto, 0 or 1".into()),
    }
}

/// World 2 FP4-Local serves the two-rank tensor split (spark_engine flash::next_batched tp), the
/// only 2-Spark path: the pipeline's links, recipe and session protocol with rank 1 starting at
/// layer 2, both ranks splitting each routed expert by columns, the projections and the
/// linear-attention heads, one-sided RDMA writes for the exchanges and the verify head split by
/// vocabulary halves. Its exchanges ride RDMA, so TCP (`rdma` false) and `--tensor-map`
/// (`rank_path`) are refused at world 2 before any rank loads. FP8 keeps its ring path.
/// A larger world serves the split when `splits` (the checkpoint's widths cut into whole parts
/// for it and its mailbox has room: split_accepts) and its links are RDMA; there it splits the
/// experts, the projections and the greedy head into parts (flash_tp_switch::PAIR has the levels
/// that stay on the pair). Otherwise, over TCP and with `--tensor-map`, it keeps its other path.
/// `splits` runs only for FP4-Local on two or more ranks: one Spark and FP8 ask nothing new. At
/// world 2 a checkpoint whose widths have no halves stops startup too, so the split stays the
/// only two-Spark path.
pub fn tensor_split(mixed: bool, world: usize, rank_path: bool, rdma: bool, splits: impl FnOnce() -> bool) -> Result<bool, String> {
    if !mixed || world < PIPELINE_RANKS {
        return Ok(false);
    }
    let pair = world == PIPELINE_RANKS;
    if rank_path {
        if !pair { return Ok(false); }
        return Err("two Sparks run the tensor split; --tensor-map selects the rank path, which serves four Sparks".into());
    }
    if !rdma {
        if !pair { return Ok(false); }
        return Err("two Sparks run the tensor split, which needs RDMA: unset SEQPAR_RDMA or set it to 1".into());
    }
    if !splits() {
        if !pair { return Ok(false); }
        return Err("two Sparks run the tensor split, and this checkpoint's widths have no whole halves".into());
    }
    Ok(true)
}
/// The split's layout accepts `world` for this checkpoint (TpLayout::for_config: every matrix the
/// split cuts has whole parts) and its relayed mailbox has room for a path of `world` ranks.
pub fn split_accepts(config: &engine_model::flash::Config, world: usize) -> bool {
    let links = spark_gpu_probe::tp_route::Links::new(world, false);
    spark_engine::flash_tp_switch::TpLayout::for_config(config, world).is_ok()
        && links.is_ok_and(|links| spark_gpu_probe::tp_route::Channels::new(&links, 1).is_ok())
}
/// The path the engine chose and why, for the receipt (None on one Spark): the tensor split (on
/// the pair or over `world` ranks), the FP4-Local layer chain, the FP4-Local rank path
/// (`--tensor-map`) or FP8's ring.
pub fn chosen_path(distributed: bool, next_model: bool, tp: bool, pp: bool, world: usize) -> Option<spark_engine::flash_tp_switch::Path> {
    use spark_engine::flash_tp_switch::Path;
    distributed.then_some(match (tp, pp, next_model) {
        (true, _, _) if world == PIPELINE_RANKS => Path::TENSOR_SPLIT,
        (true, _, _) => Path::TENSOR_PARTS,
        (false, true, _) => Path::LAYER_PIPELINE,
        (false, false, true) => Path::RANK_PATH,
        (false, false, false) => Path::RING,
    })
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
            ("min_p", &mut sampling.min_p), ("repetition_penalty", &mut sampling.repetition_penalty)] {
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
                id == 0 && !logits && count >= 8 && count <= slots * 4 && count.is_multiple_of(4)
                    && tokens.as_chunks::<4>().0.iter().all(|row| {
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
            6 => for row in tokens.as_chunks::<4>().0 {
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
fn tensor_split_is_the_only_world_two_path() {
    // World 2 FP4-Local runs the split (or stops startup); world 1, FP8 and larger worlds the layout
    // refuses never do.
    assert_eq!(tensor_split(true, 2, false, true, || true), Ok(true));
    for (mixed, world, splits) in [(true, 1, true), (true, 3, false), (true, 4, false), (false, 2, true), (false, 4, true)] {
        assert_eq!(tensor_split(mixed, world, false, true, || splits), Ok(false), "{mixed} {world}");
        assert_eq!(tensor_split(mixed, world, true, false, || splits), Ok(false), "{mixed} {world}");
    }
    // One Spark and FP8 never ask the layout: world 1 starts as it did before the split took a world.
    let asked = std::cell::Cell::new(0);
    for (mixed, world) in [(true, 1), (false, 1), (false, 2), (false, 4)] {
        assert_eq!(tensor_split(mixed, world, false, true, || { asked.set(asked.get() + 1); true }), Ok(false), "{mixed} {world}");
    }
    assert_eq!(asked.get(), 0);
    // At world 2 a refused layout stops startup (the split is the only two-Spark path).
    assert!(tensor_split(true, 2, false, true, || false).unwrap_err().contains("no whole halves"));
    // A world the layout accepts runs the split over RDMA; over TCP or with --tensor-map it keeps its path.
    assert_eq!(tensor_split(true, 4, false, true, || true), Ok(true));
    assert_eq!(tensor_split(true, 4, false, false, || true), Ok(false));
    assert_eq!(tensor_split(true, 4, true, true, || true), Ok(false));
    // TCP and --tensor-map are refused at world 2.
    assert!(tensor_split(true, 2, false, false, || true).unwrap_err().contains("needs RDMA"));
    assert!(tensor_split(true, 2, true, true, || true).unwrap_err().contains("--tensor-map"));
    // SEQPAR_RDMA: unset, 1 or auto is RDMA; 0 is TCP; anything else stops startup.
    use std::env::VarError::{NotPresent, NotUnicode};
    assert_eq!(rdma(Err(NotPresent)), Ok(true));
    for (value, on) in [("1", true), ("auto", true), ("0", false)] { assert_eq!(rdma(Ok(value.into())), Ok(on)); }
    for value in ["", "yes", "2", " 1"] { assert!(rdma(Ok(value.into())).is_err(), "{value:?}"); }
    assert!(rdma(Err(NotUnicode("x".into()))).is_err());
    // The receipt names each path and why; one Spark names none.
    use spark_engine::flash_tp_switch::Path;
    assert_eq!(chosen_path(true, true, true, true, 2), Some(Path::TENSOR_SPLIT));
    assert_eq!(chosen_path(true, true, true, true, 4), Some(Path::TENSOR_PARTS));
    assert_eq!(chosen_path(true, true, false, true, 4), Some(Path::LAYER_PIPELINE));
    assert_eq!(chosen_path(true, true, false, false, 4), Some(Path::RANK_PATH));
    assert_eq!(chosen_path(true, false, false, false, 2), Some(Path::RING));
    assert_eq!(Path::RING.reason, "FP8: the tensor split does not serve FP8");
    assert_eq!(chosen_path(false, true, false, false, 1), None);
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
fn the_pipeline_service_drafts_and_pins_its_companions() {
    // Every unset companion is pinned at the measured recipe; a set one is kept as given.
    let all = [("VADE_PROMPT_TILE", "2048"), ("VADE_HEAD_BOUNDS", "1")];
    assert_eq!(pipeline_pins(false, |_| false), all);
    assert_eq!(pipeline_pins(false, |name| name == "VADE_PROMPT_TILE"), all[1..]);
    assert_eq!(pipeline_pins(false, |name| name == "VADE_HEAD_BOUNDS"), all[..1]);
    assert!(pipeline_pins(false, |_| true).is_empty());
    // The tensor split pins its larger prompt tile (two pipeline tiles); a set one stays as given.
    assert_eq!(pipeline_pins(true, |_| false), [("VADE_PROMPT_TILE", SPLIT_PROMPT_TILE), all[1]]);
    assert_eq!(pipeline_pins(true, |name| name == "VADE_PROMPT_TILE"), all[1..]);
    assert_eq!(SPLIT_PROMPT_TILE.parse::<usize>(), Ok(2 * spark_engine::flash_tp_switch::PIPELINE_TILE));
    for depth in 0..=3 {
        for rank in 0..2 {
            // The pipeline service: an explicit depth is checked as world 1 checks it.
            assert_eq!(next_model(true, false, rank, 2, startup_mtp(true, Some(depth))), Ok(true));
            // World 4's rank path keeps its refusal.
            assert_eq!(next_model(true, false, rank, 4, startup_mtp(false, Some(depth))).is_ok(), depth == 0);
        }
    }
    // Serving depth: the pipeline with MTP serves as world 1 (default 3, explicit kept).
    assert_eq!(serving_mtp(true, 1, None), spark_engine::flash_next_lookup::DEFAULT_MTP_DEPTH);
    assert_eq!(serving_mtp(true, 2, None), 0);
}

#[test]
fn the_engine_chooses_the_pipeline_service_from_peers_and_tensor_map() {
    // World 1 never builds a pipeline; FP8 never does; --tensor-map selects the rank path.
    assert_eq!(pipeline_recipe(true, 0, 1, false), (false, false));
    for world in [2, 3, 4, 24] {
        for rank in 0..world {
            assert_eq!(pipeline_recipe(true, rank, world, false), (true, true));
            assert_eq!(pipeline_recipe(true, rank, world, true), (false, true));
            assert_eq!(pipeline_recipe(false, rank, world, false), (false, true));
        }
    }
}

#[cfg(test)]
#[test]
fn each_start_takes_its_path_from_its_own_checkpoint_in_any_order() {
    // The path is a function of this start's checkpoint and arguments. The arguments have no
    // room for an earlier run, so FP4-Local and FP8 on one pair keep their paths in both orders.
    use spark_engine::flash_tp_switch::Path;
    let choose = |mixed: bool| {
        let tp = tensor_split(mixed, PIPELINE_RANKS, false, true, || true).unwrap();
        let (pipeline, distributed) = pipeline_recipe(mixed, 0, PIPELINE_RANKS, false);
        chosen_path(distributed, mixed, tp, pipeline, PIPELINE_RANKS)
    };
    for mixed in [true, false, true, false, false, true, true] {
        assert_eq!(choose(mixed), Some(if mixed { Path::TENSOR_SPLIT } else { Path::RING }), "{mixed}");
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
/// The shipped FP4-Local geometry (the config.json text fields the flash loader reads).
#[cfg(test)]
fn shipped_config() -> engine_model::flash::Config {
    let layers = (0..48).map(|i| if i % 4 == 3 { "\"full_attention\"" } else { "\"linear_attention\"" }).collect::<Vec<_>>().join(",");
    let text = format!(r#"{{"model_type":"qwen4_exp","text_config":{{
        "model_type":"qwen4_exp_text","hidden_act":"silu","output_gate_type":"sigmoid",
        "mamba_ssm_dtype":"float32","attention_bias":false,"tie_word_embeddings":false,
        "mtp_use_dedicated_embeddings":false,"hidden_size":2560,"num_hidden_layers":48,
        "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_experts":512,
        "num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
        "hc_count":4,"hc_lowrank":320,"vocab_size":248320,"max_position_embeddings":262144,
        "eos_token_id":248044,"rms_norm_eps":0.000001,"partial_rotary_factor":0.25,
        "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,
        "indexer_compress_ratio":4,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,
        "ple_embed_dim":2560,"ple_conv_kernel_size":4,"split_ngram_parts":128,
        "mtp_num_hidden_layers":1,"layer_types":[{layers}],
        "rope_parameters":{{"rope_type":"default","mrope_interleaved":true,"rope_theta":10000000,
            "partial_rotary_factor":0.25,"mrope_section":[11,11,10]}},
        "mtp":{{"num_hidden_layers":1,"layer_types":["full_attention"],"hybrid":true,
            "mtp_use_hidden_state_from_layer":null,"rope_theta":10000000}}
        }},"quantization_config":{{"quant_method":"fp8","activation_scheme":"dynamic",
            "weight_per_tensor":false,"act_per_tensor":false,"weight_block_size":[128,128]}}}}"#);
    engine_model::flash::Config::parse(text.as_bytes()).unwrap()
}
#[cfg(test)]
#[test]
fn the_split_accepts_only_worlds_that_cut_every_matrix() {
    // The shipped geometry over 1..=8 ranks: two and four ranks cut every split matrix into
    // whole 64-row parts with room in the relayed mailbox. Five ranks cut hidden (512 rows) and
    // the expert width, but not the GDN z rows (6144 / 5) or the QSA q/gate rows (12288 / 5), so
    // five Sparks keep the layer chain. Three, six and seven do not cut hidden; eight has no
    // mailbox room for its receive sequence words.
    let config = shipped_config();
    let accepted: Vec<usize> = (1..=8).filter(|&world| split_accepts(&config, world)).collect();
    assert_eq!(accepted, [1, 2, 4]);
    let five = spark_engine::flash_tp_switch::TpLayout::for_config(&config, 5).unwrap_err();
    assert!(five.contains("GDN z rows (6144)"), "{five}");
    assert!(spark_engine::flash_tp_switch::TpLayout::new(config.hidden, config.branches, config.expert_width, config.top_k, config.experts, 5).is_ok());
    // World 1 accepts the layout but never runs the split (tensor_split stops below two ranks).
    assert_eq!(tensor_split(true, 1, false, true, || split_accepts(&config, 1)), Ok(false));
    assert_eq!(tensor_split(true, 5, false, true, || split_accepts(&config, 5)), Ok(false));
    assert_eq!(tensor_split(true, 4, false, true, || split_accepts(&config, 4)), Ok(true));
}

/// Admission only. This option never selects arithmetic or a precision policy.
pub fn diagnostic_text(options:&[String])->Result<bool,String> {
    let mut seen=false;let mut options=options.iter();
    while let Some(option)=options.next() {match option.as_str() {
        "--diagnostic-unqualified-text"=>{if seen {return Err("duplicate --diagnostic-unqualified-text".into());}seen=true;},
        "--mtp"|"--api-key-file"=>{options.next();},
        _=>{},
    }}Ok(seen)
}
#[test]
fn diagnostic_text_is_explicit_and_cannot_be_an_option_value() {
    let values=|v:&[&str]|v.iter().map(|s|s.to_string()).collect::<Vec<_>>();
    assert!(!diagnostic_text(&[]).unwrap());
    assert!(diagnostic_text(&values(&["--mtp","0","--diagnostic-unqualified-text"])).unwrap());
    assert!(!diagnostic_text(&values(&["--api-key-file","--diagnostic-unqualified-text"])).unwrap());
    assert!(diagnostic_text(&values(&["--diagnostic-unqualified-text","--diagnostic-unqualified-text"])).is_err());
}

pub fn diagnostic_text_bind(bind:std::net::SocketAddr)->Result<(),String> {
    if bind.ip().is_loopback() {Ok(())} else {Err("--diagnostic-unqualified-text requires a loopback HTTP bind".into())}
}
#[test]
fn unqualified_text_never_binds_a_remote_http_listener() {
    for value in ["127.0.0.1:18080","[::1]:18080"] {assert!(diagnostic_text_bind(value.parse().unwrap()).is_ok());}
    for value in ["0.0.0.0:18080","[::]:18080","192.0.2.1:18080"] {assert!(diagnostic_text_bind(value.parse().unwrap()).is_err());}
}
/// What the text service takes after REQUESTS. Its refusals print this list.
pub const TEXT_OPTIONS:&str="the text service takes --diagnostic-unqualified-text, --mtp 0..3, --api-key-file PATH, --diagnostic-serving-trace and one of --prompt-keep (the default) or --no-prompt-keep, each one time";
/// Extract the diagnostic served API gate before normal option validation. Option values stay opaque.
pub fn boundary_gate_options(args:&[String])->Result<(Option<std::path::PathBuf>,Vec<String>),String> {
    let mut gate=None;let mut kept=Vec::new();let mut args=args.iter();
    while let Some(arg)=args.next() {
        if let Some(path)=arg.strip_prefix("--diagnostic-boundary-gate=") {
            if gate.is_some() || path.is_empty() {return Err("boundary gate needs one nonempty output directory".into());}gate=Some(path.into());
        } else {kept.push(arg.clone());if matches!(arg.as_str(),"--mtp"|"--api-key-file") && let Some(value)=args.next() {kept.push(value.clone());}}
    }Ok((gate,kept))
}
#[test]
fn boundary_gate_option_preserves_values_and_rejects_duplicate_or_empty_paths() {
    let args=|s:&[&str]|s.iter().map(|s|s.to_string()).collect::<Vec<_>>();
    let (path,kept)=boundary_gate_options(&args(&["--diagnostic-boundary-gate=receipts","--mtp","3"])).unwrap();
    assert_eq!(path.unwrap(),std::path::PathBuf::from("receipts"));assert_eq!(kept,args(&["--mtp","3"]));
    assert!(boundary_gate_options(&args(&["--diagnostic-boundary-gate="])).is_err());
    assert!(boundary_gate_options(&args(&["--diagnostic-boundary-gate=a","--diagnostic-boundary-gate=b"])).is_err());
    assert!(boundary_gate_options(&args(&["--api-key-file","--diagnostic-boundary-gate=a"])).unwrap().0.is_none());
}
/// The text service options after REQUESTS: the API key file, the predictor depth, the serving trace and prompt retention.
/// An unknown option, or a second copy of one, stops the start with the list of what the service takes.
pub fn text_options(args:&[String])->Result<(Option<std::path::PathBuf>,Option<usize>,bool,bool),String> {
    let mut key=None;let mut mtp=None;let mut trace=false;let mut keep=None;let mut options=args.iter();
    while let Some(option)=options.next() {match option.as_str() {
        "--diagnostic-unqualified-text"=>{},
        "--diagnostic-serving-trace" if !trace=>trace=true,
        "--api-key-file" if key.is_none()=>key=Some(std::path::PathBuf::from(options.next().ok_or("--api-key-file needs a path")?)),
        // Keep each finished prompt on both ranks and start later turns of the chat from it (the server's prompt cache).
        // ON unless --no-prompt-keep is given: with that no prompt is kept and the served path is as it was before the
        // cache existed. --prompt-keep is still read, so a start script that names it goes on working. The two names
        // are one choice: a second one of either is refused. Every rank gets the same choice.
        "--prompt-keep" if keep.is_none()=>keep=Some(true),
        "--no-prompt-keep" if keep.is_none()=>keep=Some(false),
        "--mtp" if mtp.is_none()=>{let value=options.next().ok_or("--mtp needs a depth")?;mtp=Some(value.parse::<usize>().ok().filter(|depth|*depth<=3).ok_or("text service --mtp takes a depth of 0 to 3")?);},
        _=>return Err(format!("unsupported text service option: {option}; {TEXT_OPTIONS}")),
    }}
    Ok((key,mtp,trace,keep.unwrap_or(true)))
}
/// THE ROOM OF THE PROMPT CACHE, for the start line. The cache is on by default, and at a large context its byte cap
/// (the session pool less one full-size chat) can be small or zero: "cache=true" alone would tell a user the cache
/// is on while it can keep almost nothing. `charge` is the PLANNED charge of a kept prompt of that many tokens
/// (`Factory::kept_prompt_bytes` and the cache's own notes); it grows with the tokens. No prompt exists at start, so
/// this is not the charge the cache books for a real prompt: that one is the bytes the prompt's state really holds on
/// the device. This is the longest prompt, up to `context` tokens, whose planned charge fits under `cap`: 0 when not
/// even one token fits.
pub fn text_largest_kept_prompt(context:usize,cap:usize,charge:impl Fn(usize)->Option<usize>)->usize {
    let fits=|tokens:usize|charge(tokens).is_some_and(|bytes|bytes<=cap);
    if context==0 || !fits(1) {return 0;}
    let (mut kept,mut over)=(1,context+1);
    while over-kept>1 {let middle=kept+(over-kept)/2;if fits(middle) {kept=middle;} else {over=middle;}}
    kept
}
/// The words the start line ends with: the cap in bytes and the longest prompt whose planned charge fits, in tokens.
pub fn text_keep_room_words(cap:usize,largest:usize)->String {format!("kept_cap_bytes={cap} largest_kept_prompt_tokens={largest}")}
/// One plain line at start when the cache is on and its longest prompt is under 1,024 tokens: what it means and
/// what to do. `--context` is the option as `vade up` takes it.
pub fn text_keep_room_notice(keep:bool,context:usize,cap:usize,largest:usize)->Option<String> {
    (keep && largest<1024).then(||format!("the prompt cache is on, but at context {context} its room is {cap} bytes: the longest prompt whose planned charge fits in it has {largest} tokens, so it keeps little or nothing. A smaller context gives the cache room: start with a smaller --context (for example vade up --context 16384)"))
}
#[test]
fn the_start_line_gives_the_cache_room_and_a_small_room_gets_one_notice() {
    // The charge of the device gate: 162,065,920 bytes and 38,400 a token, and the cache's own notes.
    let charge=|tokens:usize|Some(162_065_920+38_400*tokens+4*tokens+619_520+(1+tokens/64)*160);
    // Context 32,768 on the gate's pair: the cap was 261,410,138 bytes. The longest prompt that fits, to the token.
    let largest=text_largest_kept_prompt(32_768,261_410_138,charge);
    assert!(charge(largest).unwrap()<=261_410_138 && charge(largest+1).unwrap()>261_410_138,"{largest}");assert_eq!(largest,2_570);
    assert_eq!(text_keep_room_words(261_410_138,largest),"kept_cap_bytes=261410138 largest_kept_prompt_tokens=2570");
    // Room for more than the context: the whole context. No room, or not for one token: 0. A charge that cannot be told: 0.
    assert_eq!(text_largest_kept_prompt(16_384,usize::MAX,charge),16_384);
    for cap in [0,162_065_919,162_685_000] {assert_eq!(text_largest_kept_prompt(32_768,cap,charge),0,"{cap}");}
    assert_eq!((text_largest_kept_prompt(32_768,usize::MAX,|_|None),text_largest_kept_prompt(0,usize::MAX,charge)),(0,0));
    // THE NOTICE: only with the cache on and a longest prompt under 1,024 tokens. It names the flag as vade up takes it.
    let notice=text_keep_room_notice(true,32_768,180_000_000,text_largest_kept_prompt(32_768,180_000_000,charge)).unwrap();
    assert!(notice.contains("at context 32768") && notice.contains("180000000 bytes") && notice.contains("has 450 tokens") && notice.contains("vade up --context"),"{notice}");
    assert!(text_keep_room_notice(true,32_768,0,0).unwrap().contains("has 0 tokens"));
    for (keep,largest) in [(true,1_024),(true,2_570),(true,16_384),(false,0),(false,500)] {assert_eq!(text_keep_room_notice(keep,32_768,261_410_138,largest),None,"{keep} {largest}");}
    // The service prints both, from the charge the cache itself uses, and nothing else follows from them.
    let source=include_str!("../bin/flash-serve/text.rs");let code=&source[..source.find("#[cfg(test)]").unwrap()];
    assert!(code.contains("let largest=if keep {super::settings::text_largest_kept_prompt(context,kept_cap,|tokens|spark_serve::cache::publication_metadata_bytes(tokens,engine.vocab_size(),engine.kept_prompt_bytes(tokens),0,0))} else {0};"));
    assert!(code.contains("if let Some(notice)=super::settings::text_keep_room_notice(keep,context,kept_cap,largest) {eprintln!(\"TEXT PROMPT KEEP rank={rank} {notice}\");}"));
    assert!(code.contains("cache={keep} {}\"") && code.contains("super::settings::text_keep_room_words(kept_cap,largest));"));
    assert_eq!((code.matches("let largest=").count(),code.matches(",largest)").count()),(1,2),"the room is worked out once and only printed");
}
/// The line a rank prints at start when it keeps prompts although FLASH_SERVE_KV_BYTES=0 is in its environment. That
/// variable is `vade up`'s "cache off" for the other services and this service does not read it (`vade up` gives this
/// service --no-prompt-keep instead), so a rank started by hand would keep prompts with no word about it. The line
/// changes nothing: the choice is the option's alone.
pub fn text_cache_size_notice(keep:bool,cache_bytes:Option<usize>)->Option<&'static str> {
    (keep && cache_bytes==Some(0)).then_some("FLASH_SERVE_KV_BYTES=0 is set, but the text service does not read it: PROMPTS ARE KEPT. To turn the prompt cache off, start every rank with --no-prompt-keep (vade up --cache-bytes 0 does that)")
}
#[test]
fn a_cache_size_of_zero_in_the_environment_gets_one_line_and_changes_nothing() {
    // Only the case that would surprise: the cache is on and the variable says 0.
    let line=text_cache_size_notice(true,Some(0)).unwrap();assert!(line.contains("FLASH_SERVE_KV_BYTES=0") && line.contains("--no-prompt-keep") && line.contains("PROMPTS ARE KEPT"),"{line}");
    for (keep,bytes) in [(true,None),(true,Some(1)),(true,Some(1<<30)),(false,Some(0)),(false,None)] {assert_eq!(text_cache_size_notice(keep,bytes),None,"{keep} {bytes:?}");}
    // The service prints it and takes nothing else from the environment for this: `keep` is the parser's answer.
    let source=include_str!("../bin/flash-serve/text.rs");let code=&source[..source.find("#[cfg(test)]").unwrap()];
    assert!(code.contains("if let Some(notice)=super::settings::text_cache_size_notice(keep,spark_tp::flash_pool::Request::from_env().ok().and_then(|request|request.cache_bytes)) {eprintln!(\"TEXT PROMPT KEEP rank={rank} {notice}\");}"));
    assert_eq!(code.matches("std::env::var").count(),0,"the text service itself reads no environment variable");
    assert_eq!(code.matches("text_cache_size_notice").count(),1);assert_eq!(code.matches("keep=").count(),0);
}
/// The prompt cache is ON by default and `--no-prompt-keep` is its one off switch. With the off switch the service
/// forks no session, the cache holds zero bytes and the scheduler looks nothing up, as before the cache existed:
/// every use of the choice in the service is named below, and each one's off arm is the value main had without
/// `--prompt-keep`.
#[test]
fn the_prompt_cache_is_on_unless_no_prompt_keep_is_given() {
    let values=|v:&[&str]|v.iter().map(|s|(*s).to_owned()).collect::<Vec<_>>();
    // On: no option at all, the other options alone, and the old switch.
    assert!(text_options(&[]).unwrap().3);
    assert!(text_options(&values(&["--diagnostic-unqualified-text","--mtp","3","--api-key-file","key.txt","--diagnostic-serving-trace"])).unwrap().3);
    assert!(text_options(&values(&["--diagnostic-unqualified-text","--prompt-keep"])).unwrap().3);
    // The gate extractor must preserve prompt retention and the parser's refusals.
    for switches in [&[][..],&["--prompt-keep"],&["--no-prompt-keep"],&["--prompt-keep","--prompt-keep"],&["--prompt-keep","--no-prompt-keep"],&["--prompt-keep=1"]] {
        let mut args=values(&["--diagnostic-unqualified-text","--mtp","3","--api-key-file","key.txt","--diagnostic-serving-trace"]);
        args.extend(values(switches));let expected=text_options(&args);
        for at in [0,args.len()] {
            let mut with_gate=args.clone();with_gate.insert(at,"--diagnostic-boundary-gate=receipts".into());
            let (gate,options)=boundary_gate_options(&with_gate).unwrap();
            assert_eq!(gate,Some(std::path::PathBuf::from("receipts")));assert_eq!(options,args);
            assert_eq!(text_options(&options),expected);
        }
    }
    // Off: only the off switch, at any place among the options.
    assert!(!text_options(&values(&["--no-prompt-keep"])).unwrap().3);
    assert!(!text_options(&values(&["--diagnostic-unqualified-text","--mtp","3","--no-prompt-keep","--diagnostic-serving-trace"])).unwrap().3);
    let parser=include_str!("settings.rs");let parser_code=&parser[..parser.find("fn the_prompt_cache_is_on_unless_no_prompt_keep_is_given").unwrap()];
    assert!(parser_code.contains("let mut keep=None;") && parser_code.contains("\"--prompt-keep\" if keep.is_none()=>keep=Some(true),") && parser_code.contains("\"--no-prompt-keep\" if keep.is_none()=>keep=Some(false),"));
    // One place sets the choice on, one sets it off, one gives the default.
    assert_eq!(parser_code.matches("keep=Some(true)").count(),1);assert_eq!(parser_code.matches("keep=Some(false)").count(),1);
    assert_eq!(parser_code.matches("Ok((key,mtp,trace,keep.unwrap_or(true)))").count(),1);assert_eq!(parser_code.matches("keep.unwrap_or(").count(),1);
    let source=include_str!("../bin/flash-serve/text.rs");let code=&source[..source.find("#[cfg(test)]").unwrap()];
    let extract=code.find("let (gate,options)=super::settings::boundary_gate_options(&args[9..])?;").unwrap();
    let parse=code.find("let (key,mtp,trace,keep)=super::settings::text_options(&options)?;").unwrap();assert!(extract<parse);
    // The service reads no cache size from its environment (the engine takes no new environment read): the off
    // switch is the option, and `vade up` gives it to every rank for a cache size of 0 (cli state.rs engine_args).
    assert!(!code.contains("FLASH_SERVE_KV_BYTES"));
    // The service never sets the choice itself: it only reads what the parser gave.
    assert_eq!(code.matches("keep=").count(),0);assert_eq!(code.matches("let keep").count(),0);assert_eq!(code.matches("mut keep").count(),0);
    // NO FORK with the off switch: the engine is the plain one.
    assert!(code.contains("if keep {engine.with_prompt_keep()} else {engine}"));assert_eq!(code.matches(".with_prompt_keep()").count(),1);
    // ZERO CACHE BYTES with the off switch. The cache's byte bound IS the engine's byte cap: one number, so the cache
    // never has room for a prompt the engine then refuses. The whole pool as the bound was review fault 1 of step 3.
    // NO LOOKUP with the off switch: `prefix_cache` is the choice, and the cache has main's one empty entry.
    assert!(code.contains("retained_bytes:kept_cap,") && !code.contains("retained_bytes:if keep {pool}") && code.contains("prefix_cache:keep,") && code.contains("cache={keep}"));
    assert!(code.contains("entries:if keep {spark_tp::row_split::KEPT_PROMPTS} else {1},"));
    // With the cache on its counters are private (`Sharing::private_counts`): /health, which asks no
    // credential, shows no live count of another caller's kept prompts. With the off switch nothing changes.
    assert!(code.contains("sharing:spark_serve::scheduler::Sharing {private_counts:keep,..Default::default()}"));
    // The engine's byte cap is set with the choice, before either rank serves.
    let cap=code.find("let kept_cap=if keep {engine.keep_within(pool)} else {0};").unwrap();assert!(cap<code.find("if !engine.controller() {return engine.worker();}").unwrap());assert_eq!(code.matches("keep_within(").count(),1);
    // The ranks compare the choice at start, so two ranks with different choices (or one build of before the
    // default and one of after it, both started with no option) refuse each other.
    assert!(code.contains("(context as u64|u64::from(keep)<<63|u64::from(keep)<<62)") && code.contains("if record[..72]!=records[..72] {return Err("));
    // These are all the places that read the choice.
    assert_eq!(code.matches("if keep ").count(),4);assert_eq!(code.matches("from(keep)").count(),2);assert_eq!(code.matches(":keep,").count(),2);assert_eq!(code.matches("{keep}").count(),1);
}
#[test]
fn text_options_take_the_four_names_and_refuse_the_rest_with_the_list() {
    let values=|v:&[&str]|v.iter().map(|s|(*s).to_owned()).collect::<Vec<_>>();
    let (key,mtp,trace,keep)=text_options(&values(&["--diagnostic-unqualified-text","--mtp","3","--api-key-file","key.txt","--diagnostic-serving-trace"])).unwrap();
    assert_eq!((key.as_deref(),mtp,trace,keep),(Some(std::path::Path::new("key.txt")),Some(3),true,true));
    assert_eq!(text_options(&values(&["--diagnostic-unqualified-text"])).unwrap(),(None,None,false,true));
    assert_eq!(text_options(&values(&["--mtp","0"])).unwrap().1,Some(0));
    // Options of the Flash-Next service, and a second copy of an option, are refused with the list.
    for refused in [&["--allow-remote"][..],&["--fp4-local"],&["--decode-fuse","off"],&["--mtp","1","--mtp","1"],&["--diagnostic-serving-trace","--diagnostic-serving-trace"]] {
        let error=text_options(&values(refused)).unwrap_err();
        assert!(error.starts_with(&format!("unsupported text service option: {}; ",refused[0])) && error.ends_with(TEXT_OPTIONS),"{error}");
    }
    for name in ["--diagnostic-unqualified-text","--mtp 0..3","--api-key-file PATH","--diagnostic-serving-trace"] {assert!(TEXT_OPTIONS.contains(name),"{name}");}
    // A depth outside 0 to 3, or an option without its value, keeps its own refusal.
    for depth in ["4","-1","x"] {assert_eq!(text_options(&values(&["--mtp",depth])).unwrap_err(),"text service --mtp takes a depth of 0 to 3");}
    assert_eq!(text_options(&values(&["--mtp"])).unwrap_err(),"--mtp needs a depth");
    assert_eq!(text_options(&values(&["--api-key-file"])).unwrap_err(),"--api-key-file needs a path");
}
#[test]
fn text_options_list_both_prompt_keep_names_and_refuse_two_of_them_with_the_list() {
    let values=|v:&[&str]|v.iter().map(|s|(*s).to_owned()).collect::<Vec<_>>();
    for name in ["--prompt-keep","--no-prompt-keep"] {assert!(TEXT_OPTIONS.contains(&format!(" {name} ")) || TEXT_OPTIONS.contains(&format!(" {name},")),"{name}: {TEXT_OPTIONS}");}
    let (key,mtp,trace,keep)=text_options(&values(&["--diagnostic-unqualified-text","--prompt-keep","--mtp","2"])).unwrap();
    assert_eq!((key,mtp,trace,keep),(None,Some(2),false,true));
    let (key,mtp,trace,keep)=text_options(&values(&["--diagnostic-unqualified-text","--no-prompt-keep","--mtp","2"])).unwrap();
    assert_eq!((key,mtp,trace,keep),(None,Some(2),false,false));
    // (the options, the one the refusal names). Both names at once are refused in either order, and so is a second
    // copy of one name: the refusal names the second and ends with the list, which holds both names.
    for (refused,named) in [(&["--unknown-option"][..],"--unknown-option"),(&["--prompt-keep=1"],"--prompt-keep=1"),(&["--no-prompt-keep=1"],"--no-prompt-keep=1"),
        (&["--prompt-keep","--prompt-keep"],"--prompt-keep"),(&["--no-prompt-keep","--no-prompt-keep"],"--no-prompt-keep"),
        (&["--prompt-keep","--no-prompt-keep"],"--no-prompt-keep"),(&["--no-prompt-keep","--prompt-keep"],"--prompt-keep"),
        (&["--no-prompt-keep","--mtp","3","--diagnostic-unqualified-text","--prompt-keep"],"--prompt-keep")] {
        let error=text_options(&values(refused)).unwrap_err();
        assert!(error.starts_with(&format!("unsupported text service option: {named}; ")) && error.ends_with(TEXT_OPTIONS) && error.contains("--prompt-keep (the default) or --no-prompt-keep"),"{refused:?}: {error}");
    }
}
/// The drafting words of the text service start line. `default_depth` is the service's default for a request that names
/// no draft mode: TRIAL_DEPTH when a predictor of at least that depth is loaded, else 0. With it, a greedy request longer
/// than TRIAL_START tokens runs a trial and drafts at that depth only if the trial wins (default_draft=trial); without it,
/// such a request decodes plain. `predictor` is the loaded predictor depth; 0 means that no predictor layer is loaded.
pub fn text_drafting_words(default_depth:usize,predictor:usize)->String {
    if default_depth>0 {format!("default_draft=trial default_depth={default_depth} predictor_depth={predictor}")} else {format!("default_draft=plain predictor_depth={predictor}")}
}
#[test]
fn the_text_start_line_names_the_default_draft_and_the_loaded_predictor_depth() {
    use spark_tp::row_split::TRIAL_DEPTH;
    for predictor in 0..=3 {
        // The default depth as the text service sets it from the loaded predictor depth.
        let default_depth=if predictor>=TRIAL_DEPTH {TRIAL_DEPTH} else {0};
        let words=if predictor>=TRIAL_DEPTH {format!("default_draft=trial default_depth={TRIAL_DEPTH} predictor_depth={predictor}")} else {format!("default_draft=plain predictor_depth={predictor}")};
        assert_eq!(text_drafting_words(default_depth,predictor),words);
    }
    // The start line takes these words from the scheduler's default depth and the loaded engine, keeps cache={keep},
    // and no longer says drafting=false.
    let source=include_str!("../bin/flash-serve/text.rs");let code=&source[..source.find("#[cfg(test)]").unwrap()];
    assert!(code.contains("let default_depth=if engine.predictor_depth()>=spark_tp::row_split::TRIAL_DEPTH {spark_tp::row_split::TRIAL_DEPTH} else {0};"));
    assert!(code.contains("mtp_depth:default_depth,"));
    assert!(code.contains("text_drafting_words(default_depth,engine.predictor_depth())") && !code.contains("drafting=false") && code.contains("cache={keep}") && !code.contains("cache=false"));
}
