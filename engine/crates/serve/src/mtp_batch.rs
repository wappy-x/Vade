//! VADE_MTP_BATCH: embedded AUTO drafts at 1..=5 chats; world-one executable
//! startup resolves its default to guarded Auto8 and passes that mode explicitly.
//! VADE_W2_PP_C8 extends that guard to the two-stage pipeline (unset/1 on, 0 old policy).
//! Off keeps main's table: one drafting chat, the scheduler's row budget and
//! the trait's automatic limit. On drafts for cohorts of up to eight chats and
//! caps a tick's verify rows at the decode width (16, blocks::DECODE_ROWS):
//! every launch then stays on the row kernels whose per-row sums never depend
//! on how many rows share them, so a chat's bytes equal its bytes alone.
//! VADE_VERIFY_ROWS=32 (spark_engine::flash_next_rows::verify_rows, default off) raises the
//! tick and the verify pass to 32 rows; above 16 rows a chat's bytes rest on each row
//! kernel's M 17-32 census, so that switch needs its own law proof per build.
use std::ops::Range;

/// Widest verify tile that keeps the decode row kernels (blocks::DECODE_ROWS).
pub const VERIFY_ROWS: usize = 16;
const COHORT_WORLDS: &[usize] = &[1];
const PIPELINE_COHORT_WORLDS: &[usize] = &[2];

/// Execution experiment controls, parsed once by the executable and passed
/// through service construction; constructors never reread their environment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Experiments {
    /// Parsed route gate, retained even when an explicit batch override wins.
    pub pipeline_c8: bool,
    pub graph_fallback: bool,
    pub cohort_cost: bool,
    /// VADE_C8_GUARD_ROBUST, parsed once. Unset/1: stay on draft unless plain is 8%
    /// cheaper per token. 0: the live 2% latch, decision-identical to that path.
    pub guard_robust: bool,
    /// Executable-owned mode; library constructors retain their legacy environment policy.
    pub batch_mode: Option<Mode>,
}
impl Experiments {
    pub fn parse(mode: Mode, world: usize, mut read: impl FnMut(&str) -> Result<Option<String>, String>) -> Result<Self, String> {
        let graph = read("VADE_POOL_GRAPH_FALLBACK")?;
        let graph_fallback = spark_engine::flash_next_rows::pool_graph_fallback_value(graph.as_deref())?;
        let cohort = read("VADE_MTP_COHORT_ADAPT")?;
        let cohort_cost = cohort_cost_world_value(mode, world, cohort.as_deref())?;
        let guard = read("VADE_C8_GUARD_ROBUST")?;
        let guard_robust = guard_robust_value(guard.as_deref())?;
        Ok(Self { pipeline_c8: false, graph_fallback, cohort_cost, guard_robust, batch_mode: None })
    }
    pub fn startup(world: usize) -> Result<Self, String> {
        Self::startup_pipeline(world, false)
    }
    /// The executable resolves its route before reading the cohort switches.
    /// Native rank serving and larger pipelines retain their previous policy.
    pub fn startup_pipeline(world: usize, pipeline: bool) -> Result<Self, String> {
        Self::startup_with(world, pipeline, |name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)), Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(format!("{name}: {error}")),
        })
    }
    /// Resolve a startup recipe without rereading environment during link agreement.
    pub fn startup_with(world: usize, pipeline: bool, mut read: impl FnMut(&str) -> Result<Option<String>, String>) -> Result<Self, String> {
        let batch=read("VADE_MTP_BATCH")?;
        let c8=read("VADE_MTP_C8")?;
        let graph=read("VADE_POOL_GRAPH_FALLBACK")?;
        let cohort=read("VADE_MTP_COHORT_ADAPT")?;
        let guard=read("VADE_C8_GUARD_ROBUST")?;
        let pp=read("VADE_W2_PP_C8")?;
        let pp = match pp.as_deref() {
            None | Some("1") => pipeline && PIPELINE_COHORT_WORLDS.contains(&world),
            Some("0") => false,
            _ => return Err("VADE_W2_PP_C8 must be unset, 0 or 1".into()),
        };
        let local=COHORT_WORLDS.contains(&world);
        let guarded=local || pp;
        let mode=cohort_mode(parse(batch.as_deref())?, c8.as_deref().or(guarded.then_some("1")))?;
        // Healthy capture fallback is a local-graph feature, not a pipeline feature.
        let graph=graph.as_deref().or(local.then_some("1"));
        // Explicit legacy tables and C8=0 remain usable without having to set a
        // second switch. Explicit cohort=1 still checks every support requirement.
        let cohort=cohort.as_deref().or((guarded && mode==Mode::Auto8).then_some("1"));
        let graph_fallback=spark_engine::flash_next_rows::pool_graph_fallback_value(graph)?;
        let cohort_cost=if pp { cohort_cost_value(mode,cohort)? }
            else { cohort_cost_world_value(mode,world,cohort)? };
        let guard_robust=guard_robust_value(guard.as_deref())?;
        Ok(Self { pipeline_c8:pp,graph_fallback,cohort_cost,guard_robust,batch_mode:Some(mode) })
    }
}
#[cfg(test)]
#[test]
fn experiment_inputs_are_read_once_then_reusable_without_environment_access() {
    for value in [None,Some("0"),Some("1")] {
        let mut reads=Vec::new();
        let parsed=Experiments::parse(Mode::Auto8,1,|name| { reads.push(name.to_owned()); Ok(value.map(str::to_owned)) }).unwrap();
        assert_eq!(reads,["VADE_POOL_GRAPH_FALLBACK","VADE_MTP_COHORT_ADAPT","VADE_C8_GUARD_ROBUST"]);
        assert_eq!(parsed.graph_fallback,value==Some("1")); assert_eq!(parsed.cohort_cost,value==Some("1"));
        assert_eq!(parsed.guard_robust, value!=Some("0"));
        let copied=parsed; assert_eq!(copied,parsed);
    }
}
#[cfg(test)]
#[test]
fn executable_defaults_match_opt_in_only_on_world_one_and_honor_rollbacks() {
    let resolve=|world, values: &[(&str,&str)]| {
        let mut reads=Vec::new();
        let result=Experiments::startup_with(world,false,|key| {
            reads.push(key.to_owned());Ok(values.iter().find(|(k,_)|*k==key).map(|(_,v)|v.to_string()))
        });
        assert_eq!(reads,["VADE_MTP_BATCH","VADE_MTP_C8","VADE_POOL_GRAPH_FALLBACK","VADE_MTP_COHORT_ADAPT","VADE_C8_GUARD_ROBUST","VADE_W2_PP_C8"]);
        result
    };
    let on=[("VADE_MTP_C8","1"),("VADE_POOL_GRAPH_FALLBACK","1"),("VADE_MTP_COHORT_ADAPT","1")];
    let off=[("VADE_MTP_C8","0"),("VADE_POOL_GRAPH_FALLBACK","0"),("VADE_MTP_COHORT_ADAPT","0")];
    assert_eq!(resolve(1,&[]),resolve(1,&on));
    assert_eq!(resolve(1,&off).unwrap(),Experiments { batch_mode:Some(Mode::Auto), guard_robust:true, ..Experiments::default() });
    for world in [2,4] {
        assert_eq!(resolve(world,&[]).unwrap(),Experiments { batch_mode:Some(Mode::Auto), guard_robust:true, ..Experiments::default() });
        assert!(resolve(world,&on).is_err());
    }
    for value in ["0","1"] {
        let parsed=resolve(1,&[("VADE_MTP_BATCH",value)]).unwrap();
        assert!(!parsed.cohort_cost); assert_eq!(parsed.batch_mode,Some(parse(Some(value)).unwrap()));
    }
    assert!(!resolve(1,&[("VADE_MTP_C8","0")]).unwrap().cohort_cost);
    assert!(!resolve(1,&[("VADE_MTP_COHORT_ADAPT","0")]).unwrap().cohort_cost);
    assert!(resolve(1,&[]).unwrap().guard_robust);
    assert!(!resolve(1,&[("VADE_C8_GUARD_ROBUST","0")]).unwrap().guard_robust);
    for key in ["VADE_MTP_C8","VADE_POOL_GRAPH_FALLBACK","VADE_MTP_COHORT_ADAPT","VADE_MTP_BATCH","VADE_C8_GUARD_ROBUST"] {
        assert!(resolve(1,&[(key,"invalid")]).is_err());
    }
}
#[cfg(test)]
#[test]
fn pipeline_c8_defaults_and_rollback_follow_the_actual_route() {
    let resolve=|world,pipeline,values: &[(&str,&str)]| {
        let mut reads=Vec::new();
        let result=Experiments::startup_with(world,pipeline,|key| {
            assert!(!reads.contains(&key.to_owned()), "switch read twice");
            reads.push(key.to_owned());
            Ok(values.iter().find(|(k,_)|*k==key).map(|(_,v)|v.to_string()))
        });
        assert_eq!(reads.len(),6);
        result
    };
    let old=Experiments { batch_mode:Some(Mode::Auto),guard_robust:true,..Experiments::default() };
    let enabled=resolve(2,true,&[]).unwrap();
    assert_eq!(enabled,Experiments { pipeline_c8:true,cohort_cost:true,batch_mode:Some(Mode::Auto8),graph_fallback:false,guard_robust:true });
    assert_eq!(resolve(2,true,&[("VADE_W2_PP_C8","1")]),Ok(enabled));
    assert_eq!(resolve(2,true,&[("VADE_W2_PP_C8","0")]),Ok(old));
    for (world,pipeline) in [(2,false),(4,false),(4,true)] {
        for value in ["0","1"] {
            assert_eq!(resolve(world,pipeline,&[("VADE_W2_PP_C8",value)]),Ok(old));
        }
        assert!(resolve(world,pipeline,&[("VADE_MTP_C8","1"),("VADE_MTP_COHORT_ADAPT","1")]).is_err());
    }
    for value in ["0","1"] {
        assert_eq!(resolve(1,false,&[("VADE_W2_PP_C8",value)]),resolve(1,false,&[]));
        let explicit=resolve(2,true,&[("VADE_MTP_BATCH",value)]).unwrap();
        assert!(!explicit.cohort_cost);
        assert_eq!(explicit.batch_mode,Some(parse(Some(value)).unwrap()));
    }
    assert_eq!(resolve(2,true,&[("VADE_MTP_C8","0")]),Ok(Experiments { pipeline_c8:true,..old }));
    assert!(!resolve(2,true,&[("VADE_MTP_COHORT_ADAPT","0")]).unwrap().cohort_cost);
    for value in ["", "true", "2", " 1"] {
        assert!(resolve(2,true,&[("VADE_W2_PP_C8",value)]).unwrap_err().contains("VADE_W2_PP_C8"));
    }
    // OFF keeps the pre-C8 world-two formulas, including explicit legacy overrides
    // and invalid combinations, rather than just matching the default tuple.
    for batch in [None, Some("0"), Some("1")] {
        for c8 in [None, Some("0"), Some("1")] {
            for graph in [None, Some("0"), Some("1")] {
                for cohort in [None, Some("0"), Some("1")] {
                    for guard in [None, Some("0"), Some("1")] {
                        let mut values = vec![("VADE_W2_PP_C8", "0")];
                        for (name, value) in [("VADE_MTP_BATCH", batch), ("VADE_MTP_C8", c8),
                            ("VADE_POOL_GRAPH_FALLBACK", graph), ("VADE_MTP_COHORT_ADAPT", cohort),
                            ("VADE_C8_GUARD_ROBUST", guard)] {
                            if let Some(value) = value { values.push((name, value)); }
                        }
                        assert_eq!(resolve(2,true,&values),resolve(2,false,&values), "{values:?}");
                    }
                }
            }
        }
    }
}
fn cohort_cost_world_value(mode: Mode, world: usize, value: Option<&str>) -> Result<bool, String> {
    let enabled = cohort_cost_value(mode, value)?;
    if enabled && !COHORT_WORLDS.contains(&world) {
        return Err("VADE_MTP_COHORT_ADAPT is world-one only; unset it for pipeline/rank serving".into());
    }
    Ok(enabled)
}
#[cfg(test)]
#[test]
fn cohort_switch_contract_covers_local_pipeline_and_native_rank_startup() {
    for (world, supported) in [(1,true),(2,false),(4,false)] {
        for value in [None,Some("0")] { assert_eq!(cohort_cost_world_value(Mode::Auto,world,value),Ok(false)); }
        assert_eq!(cohort_cost_world_value(Mode::Auto8,world,Some("1")).is_ok(),supported);
        for value in ["", "2", "true", " 1"] {
            assert!(cohort_cost_world_value(Mode::Auto8,world,Some(value)).unwrap_err().contains("VADE_MTP_COHORT_ADAPT"));
        }
    }
}
fn guard_robust_value(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        _ => Err("VADE_C8_GUARD_ROBUST must be unset, 0 or 1".into()),
    }
}
fn cohort_cost_value(mode: Mode, value: Option<&str>) -> Result<bool, String> {
    let enabled = match value {
        None | Some("0") => false,
        Some("1") => true,
        _ => return Err("VADE_MTP_COHORT_ADAPT must be unset, 0 or 1".into()),
    };
    if enabled && mode != Mode::Auto8 {
        return Err("VADE_MTP_COHORT_ADAPT=1 requires VADE_MTP_C8=1 and unset VADE_MTP_BATCH".into());
    }
    Ok(enabled)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// Largest active cohort whose chats may draft.
    pub draft_chats: usize,
    /// Cohort size above which automatic requests resolve to plain.
    pub automatic_chats: usize,
    /// Decode row budget per tick; None keeps the scheduler's own budget.
    pub tick_rows: Option<usize>,
}
impl Policy {
    pub const fn new(on: bool) -> Self {
        if on {
            Self { draft_chats: 8, automatic_chats: 8, tick_rows: Some(VERIFY_ROWS) }
        } else {
            Self { draft_chats: 1, automatic_chats: 4, tick_rows: None }
        }
    }
}

/// Explicit overrides retain their old tables; AUTO selects the complete table per tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Auto, Auto8, Off, On }
impl Mode {
    /// Temporary plain probes use the same deferred history path as AUTO C8.
    pub fn feed(self, chats: usize, plain: bool) -> (bool, bool) {
        (self.enabled(chats) && !plain, chats > self.policy(chats).draft_chats || plain)
    }
    pub const fn enabled(self, chats: usize) -> bool {
        match self { Self::Auto => chats >= 1 && chats <= 5,
            Self::Auto8 => chats >= 1 && chats <= 8, Self::Off => false, Self::On => true }
    }
    pub const fn policy(self, chats: usize) -> Policy { Policy::new(self.enabled(chats)) }
    /// AUTO pools an all-one-row cohort at every size: above the drafting boundary no chat
    /// drafts, and the pooled step is the one plain requests already take.
    pub fn pooled(self, chats: usize, all_one_row: bool) -> bool {
        (matches!(self, Self::Auto | Self::Auto8) || self.enabled(chats)) && chats > 1 && all_one_row
    }
}

/// Unset = AUTO, "0" = off, "1" = on; other values are errors.
pub fn parse(value: Option<&str>) -> Result<Mode, String> {
    match value {
        None => Ok(Mode::Auto),
        Some("0") => Ok(Mode::Off),
        Some("1") => Ok(Mode::On),
        Some(v) => Err(format!("VADE_MTP_BATCH must be unset, 0 or 1 (got {v:?})")),
    }
}

/// VADE_MTP_ADAPT defaults to 1; explicit 0 disables it. One-chat Flash-Next MTP picks its draft depth per cycle from the
/// request's measured depth survival and costs (explicit 0: always the request depth).
/// VADE_MTP_ADAPT=full (measurement): a request at the serving default depth drafts the
/// whole resident depth (VADE_FLASHNEXT_MTP_DEPTH) every cycle, with no chooser.
pub fn adapt_depth() -> Result<bool, String> {
    adapt_mode().map(|m| m != 0)
}
/// VADE_MTP_SAMPLED=1 (default off): one-chat Flash-Next keeps MTP drafting for sampled
/// requests (temperature > 0). Strict verification samples every row from the target's own
/// logits with the request seed at that row's absolute position (sample::uniform), and only
/// the prefix matching the drafts commits, so the output equals the undrafted run for the
/// same seed; drafts change only how many rows are verified per cycle.
pub fn sampled_draft() -> Result<bool, String> {
    sampled_mode().map(|m| m != 0)
}
/// VADE_MTP_SAMPLED=2: sampled chats also draft in cohorts; their rows join the cohort's
/// batched verify pass and generation samples each row from its retained logits.
pub fn sampled_cohort() -> Result<bool, String> {
    sampled_mode().map(|m| m == 2)
}
/// VADE_MTP_SAMPLED_TOPK=1 (default off): retain canonical logits but return the
/// existing exact top-k shortlist. Generation falls back to full retained rows
/// when the canonical nucleus boundary cannot be certified.
pub fn sampled_top_k() -> Result<bool, String> {
    match std::env::var("VADE_MTP_SAMPLED_TOPK") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "0" || v == "1" => Ok(v == "1"),
        _ => Err("VADE_MTP_SAMPLED_TOPK must be unset, 0 or 1".into()),
    }
}
fn sampled_mode() -> Result<u8, String> {
    static MODE: std::sync::OnceLock<Result<u8, String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| match std::env::var("VADE_MTP_SAMPLED") {
        Err(std::env::VarError::NotPresent) => Ok(0),
        Ok(v) => match v.as_str() { "0" => Ok(0), "1" => Ok(1), "2" => Ok(2), _ => Err(format!("VADE_MTP_SAMPLED must be unset, 0, 1 or 2 (got {v:?})")) },
        Err(e) => Err(format!("VADE_MTP_SAMPLED: {e}")),
    })
    .clone()
}
/// VADE_MTP_ADAPT=full: see adapt_depth.
pub fn adapt_full() -> Result<bool, String> {
    adapt_mode().map(|m| m == 2)
}
fn adapt_mode() -> Result<u8, String> {
    static MODE: std::sync::OnceLock<Result<u8, String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| match std::env::var("VADE_MTP_ADAPT") {
        Err(std::env::VarError::NotPresent) => Ok(1),
        Ok(v) => match v.as_str() { "0" => Ok(0), "1" => Ok(1), "full" => Ok(2), _ => Err(format!("VADE_MTP_ADAPT must be unset, 0, 1 or full (got {v:?})")) },
        Err(e) => Err(format!("VADE_MTP_ADAPT: {e}")),
    })
    .clone()
}

/// Read once per process; resolve AUTO against the active cohort each tick.
pub fn switch() -> Result<Mode, String> {
    static MODE: std::sync::OnceLock<Result<Mode, String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| {
        let value = match std::env::var("VADE_MTP_C8") {
            Ok(value) => Some(value), Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("VADE_MTP_C8 must be unset, 0 or 1".into()),
        };
        cohort_mode(read("VADE_MTP_BATCH")?, value.as_deref())
    }).clone()
}

/// VADE_MTP_C8=1: AUTO also drafts at C6..8 within the existing
/// 16-row tick budget. Explicit VADE_MTP_BATCH overrides retain their tables.
fn cohort_mode(mode: Mode, value: Option<&str>) -> Result<Mode, String> {
    match value {
        None | Some("0") => Ok(mode),
        Some("1") => Ok(if mode == Mode::Auto { Mode::Auto8 } else { mode }),
        _ => Err("VADE_MTP_C8 must be unset, 0 or 1".into()),
    }
}

/// VADE_ONE_ROW_POOLED: an all-one-row cohort skips eager verification for
/// the pooled greedy step. AUTO pools at every cohort size; 0 forces eager rows.
pub fn one_row_pooled() -> Result<Mode, String> {
    static MODE: std::sync::OnceLock<Result<Mode, String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| read("VADE_ONE_ROW_POOLED")).clone()
}
fn read(name: &str) -> Result<Mode, String> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => parse(None),
        Ok(v) => parse(Some(&v)).map_err(|_| format!("{name} must be unset, 0 or 1 (got {v:?})")),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

/// Consecutive chat groups of at most `limit` rows, in cohort order. A chat
/// wider than `limit` stays alone in its group.
pub fn verify_groups(rows: &[usize], limit: usize) -> Vec<Range<usize>> {
    let (mut groups, mut start, mut total) = (Vec::new(), 0, 0);
    for (i, &n) in rows.iter().enumerate() {
        if i > start && total + n > limit {
            groups.push(start..i);
            (start, total) = (i, 0);
        }
        total += n;
    }
    if start < rows.len() {
        groups.push(start..rows.len());
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohort_controller_requires_explicit_auto8_and_defaults_off() {
        for mode in [Mode::Auto, Mode::Auto8, Mode::Off, Mode::On] {
            assert_eq!(cohort_cost_value(mode, None), Ok(false));
            assert_eq!(cohort_cost_value(mode, Some("0")), Ok(false));
            assert_eq!(cohort_cost_value(mode, Some("1")).is_ok(), mode == Mode::Auto8);
            for value in ["", "2", "true", " 1"] {
                assert!(cohort_cost_value(mode, Some(value)).is_err());
            }
        }
    }

    #[test]
    fn cohort_plain_probe_defers_history_and_resume_restores_batching() {
        assert_eq!(Mode::Auto.feed(8, false), (false, true));
        assert_eq!(Mode::Auto8.feed(8, true), Mode::Auto.feed(8, false));
        for chats in [6, 7, 8] {
            assert_eq!(Mode::Auto8.feed(chats, false), (true, false));
            assert_eq!(Mode::Auto8.feed(chats, true), (false, true));
        }
        for mode in [Mode::Auto, Mode::Auto8, Mode::Off, Mode::On] {
            for chats in 0..=16 {
                assert_eq!(mode.feed(chats, false), (mode.enabled(chats), chats > mode.policy(chats).draft_chats));
            }
        }
    }

    #[test]
    fn switch_values() {
        assert_eq!(parse(None), Ok(Mode::Auto));
        assert_eq!(parse(Some("0")), Ok(Mode::Off));
        assert_eq!(parse(Some("1")), Ok(Mode::On));
        for v in ["", "2", "on", "true", " 1"] {
            assert!(parse(Some(v)).is_err(), "{v:?}");
        }
    }

    #[test]
    fn off_is_mains_table() {
        // flashnext.rs at a9339fe: single_chat_draft, the trait default
        // automatic_draft_limit 4, no row budget beyond the scheduler config.
        assert_eq!(Policy::new(false), Policy { draft_chats: 1, automatic_chats: 4, tick_rows: None });
        let on = Policy::new(true);
        assert_eq!((on.draft_chats, on.automatic_chats, on.tick_rows), (8, 8, Some(16)));
    }

    #[test]
    fn auto_and_overrides_select_complete_tables_and_pooled_routing() {
        for chats in 1..=16 {
            let on = chats <= 5;
            assert_eq!(parse(None).unwrap().policy(chats), Policy::new(on), "C{chats}");
            assert_eq!(Mode::Auto.enabled(chats), on);
            for (mode, enabled) in [(Mode::Auto, on), (parse(Some("0")).unwrap(), false), (parse(Some("1")).unwrap(), true)] {
                assert_eq!(mode.policy(chats), Policy::new(enabled), "{mode:?} C{chats}");
                assert_eq!(mode.pooled(chats, true), (mode == Mode::Auto || enabled) && chats > 1);
                assert!(!mode.pooled(chats, false));
            }
        }
        assert!(!Mode::Auto.enabled(0));
    }

    #[test]
    fn groups_keep_order_and_width() {
        assert_eq!(verify_groups(&[], 16), Vec::<Range<usize>>::new());
        assert_eq!(verify_groups(&[4, 4, 4, 4], 16), [0..4]);
        assert_eq!(verify_groups(&[4, 4, 4, 4, 4], 16), [0..4, 4..5]);
        assert_eq!(verify_groups(&[3, 3, 3, 3, 3], 16), [0..5]);
        assert_eq!(verify_groups(&[1; 17], 16), [0..16, 16..17]);
        assert_eq!(verify_groups(&[16, 1, 16], 16), [0..1, 1..2, 2..3]);
        for rows in [[4usize, 3, 2, 1, 4, 4, 3, 1], [1, 1, 1, 1, 4, 4, 4, 4]] {
            let groups = verify_groups(&rows, 16);
            assert_eq!(groups.first().unwrap().start, 0);
            assert_eq!(groups.last().unwrap().end, rows.len());
            assert!(groups.windows(2).all(|w| w[0].end == w[1].start));
            assert!(groups.iter().all(|g| rows[g.clone()].iter().sum::<usize>() <= 16));
        }
    }

    #[test]
    fn c8_opt_in_preserves_overrides_and_caps_the_cohort() {
        for mode in [Mode::Auto, Mode::Off, Mode::On] {
            for value in [None, Some("0")] { assert_eq!(cohort_mode(mode, value), Ok(mode)); }
            let opt = cohort_mode(mode, Some("1")).unwrap();
            for chats in 0..=16 {
                let on = if mode == Mode::Auto { (1..=8).contains(&chats) } else { mode.enabled(chats) };
                assert_eq!(opt.policy(chats), Policy::new(on));
                assert_eq!(opt.pooled(chats, true), (mode == Mode::Auto || on) && chats > 1);
            }
        }
        assert!(cohort_mode(Mode::Auto, Some("2")).is_err());
    }
}
