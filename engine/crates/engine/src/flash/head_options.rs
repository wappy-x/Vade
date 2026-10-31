//! Bounded target head: automatic only when removed scratch funds all its state.
use std::sync::OnceLock;
static ENABLED: OnceLock<Result<Option<bool>, String>> = OnceLock::new();
fn parse(value: Result<String, std::env::VarError>) -> Result<Option<bool>, String> {
    match value.as_deref() {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok("0") => Ok(Some(false)),
        Ok("1") => Ok(Some(true)),
        _ => Err("VADE_HEAD_BOUNDS must be unset, 0 or 1".into()),
    }
}
/// Validate once and report explicit opt-in before a model's funding is known.
pub fn load() -> Result<bool, String> {
    configured().map(|enabled| enabled.unwrap_or(false))
}
/// Refuse a malformed value at startup without fixing the switch: a pipeline rank pins it
/// (tp settings::PIPELINE_DEFAULTS) after argument parsing and before the model reads it.
pub fn validate() -> Result<(), String> {
    parse(std::env::var("VADE_HEAD_BOUNDS")).map(|_| ())
}
/// Rows of rank 0's vocabulary half under the tensor split's head split on the pair: the lower
/// half in whole 64-row tiles (the bounded screen's unit); rank 1 takes the rest. The split head
/// covers the greedy verify head of one-chat passes of 2..=RETAINED_ROWS rows.
pub fn split_rows(vocabulary: usize) -> usize { part_start(vocabulary, crate::flash_tp_switch::PAIR, 1) }
/// The first row of rank `rank`'s vocabulary part under the head split over `world` ranks: equal
/// parts of whole 64-row tiles, the last rank also takes the rest (`rank` = `world`: the end).
pub fn part_start(vocabulary: usize, world: usize, rank: usize) -> usize {
    if rank >= world { vocabulary } else { vocabulary / (64 * world) * 64 * rank }
}
/// The split head over `parts` ranks: each row's argmax from the parts' argmax rows (`rows` holds
/// `parts` blocks of the same rows, part p's ids from 0 at part_start), merge_split folded in rank
/// order. The rule is a total order, so the result is the whole head's argmax.
pub fn merge_parts(rows: &[(u32, f32)], vocabulary: usize, parts: usize) -> Result<Vec<(u32, f32)>, String> {
    if parts == 0 || !rows.len().is_multiple_of(parts) { return Err("split head rows differ".into()); }
    let n = rows.len() / parts;
    (1..parts).try_fold(rows[..n].to_vec(), |merged, p| merge_split(&merged, &rows[p * n..(p + 1) * n], part_start(vocabulary, parts, p) as u32))
}
/// The split head: each row's argmax over the whole vocabulary from the two halves' argmax rows
/// (`own` over rows [0, start), `peer` over rows [start, ..) with ids from 0): the larger logit,
/// ties to the lower id (always `own`'s), the order logits_argmax_finish uses within one head.
pub fn merge_split(own: &[(u32, f32)], peer: &[(u32, f32)], start: u32) -> Result<Vec<(u32, f32)>, String> {
    if own.len() != peer.len() { return Err("split head rows differ".into()); }
    own.iter().zip(peer).map(|(&(i, v), &(j, w))| {
        let j = j.checked_add(start).ok_or("split head id overflow")?;
        let (value, id) = spark_gpu_probe::next_math::better((v, i), (w, j));
        Ok((id, value))
    }).collect()
}
/// The head rows whose bounds this rank loads, as (start, rows): the whole head on a rank with the
/// head; under the tensor split each rank its part (on the pair the front its lower half and the
/// headless rank 1 its upper half). None: no bounds (a headless pipeline rank).
pub fn bounds_rows(vocabulary: usize, headless: bool) -> Option<(usize, usize)> {
    use crate::flash_tp_switch as tp;
    split_bounds_rows(vocabulary, headless, tp::on().then(|| (tp::rank().unwrap_or(usize::from(headless)), tp::world())))
}
/// `split`: (this rank, the split's ranks) under the tensor split.
fn split_bounds_rows(vocabulary: usize, headless: bool, split: Option<(usize, usize)>) -> Option<(usize, usize)> {
    match split {
        Some((rank, world)) => {
            let start = part_start(vocabulary, world, rank);
            Some((start, part_start(vocabulary, world, rank + 1) - start))
        }
        None => (!headless).then_some((0, vocabulary)),
    }
}
fn configured() -> Result<Option<bool>, String> {
    ENABLED.get_or_init(|| parse(std::env::var("VADE_HEAD_BOUNDS"))).clone()
}
/// Existing explicit switches keep their meaning; automatic mode must fund the
/// full twin, norms and retained inputs for every possible session, not just C1.
pub fn load_funded(saved: usize, twin: usize, vocabulary: usize, sessions: usize, mtp: bool) -> Result<bool, String> {
    configured().map(|enabled| enabled.unwrap_or_else(|| {
        let on = funded(saved, twin, vocabulary, sessions, mtp);
        if !on {
            engine_model::events::note("head_bounds_skipped", format_args!("reason={} saved_bytes={saved} twin_bytes={twin} vocabulary={vocabulary} sessions={sessions}",
                if !mtp { "no_draft" } else if twin == 0 { "no_twin" } else { "unfunded" }));
        }
        on
    }))
}
fn funded(saved: usize, twin: usize, vocabulary: usize, sessions: usize, mtp: bool) -> bool {
    if !mtp || twin == 0 { return false; }
    let needed = vocabulary.checked_mul(8).and_then(|n| n.checked_add(63)).map(|n| n / 64 * 64)
        .and_then(|norms| twin.checked_add(norms))
        .and_then(|n| sessions.checked_mul(RETAINED_INPUT_BYTES).and_then(|retained| n.checked_add(retained)));
    needed.is_some_and(|n| n <= saved)
}
/// Model-wide optional storage is forbidden on a pipeline rank that projects no head (the
/// pipeline back and middle ranks); the front projects it as one Spark does.
/// No callback (and thus no allocation or log) runs when disabled or unused.
pub fn load_local<T>(enabled: bool, headless: bool, load: impl FnOnce() -> Result<T, String>) -> Result<Option<T>, String> {
    if enabled && !headless { load().map(Some) } else { Ok(None) }
}
/// Verify passes of up to eight rows (one canonical head group) take the bounded head.
pub const RETAINED_ROWS: usize = 8;
pub const RETAINED_INPUT_BYTES: usize = RETAINED_ROWS * 2560 * 4;
pub fn retained_input_bytes(enabled: bool) -> usize { if enabled { RETAINED_INPUT_BYTES } else { 0 } }
#[cfg(test)]
mod tests {
    use super::{parse, load_local, retained_input_bytes, funded, split_bounds_rows, split_rows, merge_split, merge_parts, part_start};
    /// One argmax over a row: the larger logit, ties to the lower id (logits_argmax_finish).
    fn whole(row: &[f32]) -> (u32, f32) {
        row.iter().enumerate().fold((u32::MAX, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 || v == best.1 && (i as u32) < best.0 { (i as u32, v) } else { best } })
    }
    /// The merge equals one argmax over the concatenated row (larger logit, ties to the lower id),
    /// for every split point and with ties across the halves.
    #[test]
    fn split_merge_equals_the_whole_argmax() {
        let rows: Vec<Vec<f32>> = vec![vec![1.0, 3.0, 2.0, 3.0, 0.5, -1.0, 3.0, 2.5], vec![-2.0, -2.0, -1.0, -3.0, -1.0, -1.0, -5.0, -1.0],
            vec![0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0]];
        for start in 1..8usize {
            let own: Vec<_> = rows.iter().map(|r| whole(&r[..start])).collect();
            let peer: Vec<_> = rows.iter().map(|r| whole(&r[start..])).collect();
            let merged = merge_split(&own, &peer, start as u32).unwrap();
            for (row, m) in rows.iter().zip(merged) {
                let w = whole(row);
                assert_eq!((m.0, m.1.to_bits()), (w.0, w.1.to_bits()), "start {start} row {row:?}");
            }
        }
        assert!(merge_split(&[(0, 1.0)], &[], 4).is_err());
        assert!(merge_split(&[(0, 1.0)], &[(u32::MAX, 2.0)], 4).is_err());
    }
    /// The plain split head gives the whole head's ids: random rows of a few logit values (ties
    /// inside and across the halves, signed zeros) with the top value planted at one to three
    /// random ids, at each vocabulary's split row, merged for up to 16 chats at once. The halves'
    /// argmax also follows the device's partition: each of the 192 blocks takes one contiguous span
    /// of whole 8-row groups (lbf_project, flash_next_head), then one fold over the block winners.
    #[test]
    fn split_merge_equals_the_whole_argmax_on_random_rows_with_ties() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        let blocked = |row: &[f32]| {
            let (groups, blocks) = (row.len() / 8, 192);
            let span = groups.div_ceil(blocks);
            let winners: Vec<(u32, f32)> = (0..blocks).map(|block| (block * span * 8..((block + 1) * span).min(groups) * 8)
                .map(|i| (i, row[i])).fold((u32::MAX, f32::NEG_INFINITY), |best, (i, v)| {
                    if v > best.1 || v == best.1 && (i as u32) < best.0 { (i as u32, v) } else { best } })).collect();
            winners.into_iter().fold((u32::MAX, f32::NEG_INFINITY), |best, (i, v)| {
                if v > best.1 || v == best.1 && i < best.0 { (i, v) } else { best } })
        };
        let base = [-1.5f32, -0.0, 0.0, 0.25];
        let (mut own_wins, mut peer_wins) = (0, 0);
        for vocabulary in [248_320usize, 151_936, 64 * 1000 + 64] {
            let start = split_rows(vocabulary);
            for chats in [1usize, 3, 16] {
                let rows: Vec<Vec<f32>> = (0..chats).map(|_| {
                    let mut row: Vec<f32> = (0..vocabulary).map(|_| base[next() as usize % base.len()]).collect();
                    let top = if next() % 4 == 0 { 0.25 } else { 2.0 };
                    for _ in 0..1 + next() % 3 { row[next() as usize % vocabulary] = top; }
                    row
                }).collect();
                let own: Vec<_> = rows.iter().map(|r| blocked(&r[..start])).collect();
                let peer: Vec<_> = rows.iter().map(|r| blocked(&r[start..])).collect();
                for (r, (o, p)) in rows.iter().zip(own.iter().zip(&peer)) {
                    assert_eq!((o.0, o.1.to_bits()), { let w = whole(&r[..start]); (w.0, w.1.to_bits()) });
                    assert_eq!((p.0, p.1.to_bits()), { let w = whole(&r[start..]); (w.0, w.1.to_bits()) });
                }
                let merged = merge_split(&own, &peer, start as u32).unwrap();
                for (row, m) in rows.iter().zip(merged) {
                    let w = whole(row);
                    assert_eq!((m.0, m.1.to_bits()), (w.0, w.1.to_bits()), "vocabulary {vocabulary} chats {chats}");
                    if (m.0 as usize) < start { own_wins += 1 } else { peer_wins += 1 }
                }
            }
        }
        // Both halves won rows, so the test covered the merge in both directions.
        assert!(own_wins > 0 && peer_wins > 0, "own {own_wins} peer {peer_wins}");
    }
    /// The halves keep the whole head's row groups and K order: the split row starts a whole 8-row
    /// group, both halves are whole groups, and every weight of a row in rank 1's half sits at the
    /// same place relative to the half's first row as in the whole head (next_math::head_weight),
    /// inside the same 8-row group. So each logit is one warp's walk over the same group in the
    /// same K order, on either rank.
    #[test]
    fn split_halves_keep_the_head_row_groups_and_k_order() {
        use spark_gpu_probe::next_math::head_weight;
        let width = 2560;
        for vocabulary in [248_320usize, 151_936, 64 * 1000 + 64] {
            let start = split_rows(vocabulary);
            assert!(start.is_multiple_of(8) && (vocabulary - start).is_multiple_of(8));
            assert_eq!(head_weight(start, 0, width), start * width);
            for r in [start, start + 1, start + 7, start + 8, start + 9, vocabulary - 8, vocabulary - 1] {
                for c in [0, 1, 7, 8, 15, 16, 31, 32, 33, width - 1] {
                    let at = head_weight(r, c, width);
                    assert_eq!(at, start * width + head_weight(r - start, c, width), "row {r} col {c}");
                    assert_eq!(at / (8 * width), r / 8, "row {r} col {c}");
                }
            }
        }
    }
    /// Each rank's bounds cover its own half in whole 64-row tiles, the halves tile the head, and
    /// without the split the front keeps the whole head and a headless rank none.
    #[test]
    fn split_bounds_cover_each_ranks_half() {
        for vocabulary in [248_320usize, 151_936, 64 * 1000 + 64] {
            let half = split_rows(vocabulary);
            assert_eq!(half % 64, 0);
            assert!(half <= vocabulary - half && vocabulary - 2 * half < 128);
            assert_eq!(split_bounds_rows(vocabulary, false, Some((0, 2))), Some((0, half)));
            assert_eq!(split_bounds_rows(vocabulary, true, Some((1, 2))), Some((half, vocabulary - half)));
            assert_eq!(split_bounds_rows(vocabulary, false, None), Some((0, vocabulary)));
            assert_eq!(split_bounds_rows(vocabulary, true, None), None);
            // Four ranks: equal parts of whole 64-row tiles that tile the head; the last takes the rest.
            let parts: Vec<_> = (0..4).map(|r| split_bounds_rows(vocabulary, r != 0, Some((r, 4))).unwrap()).collect();
            assert!(parts.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0 && w[0].0 % 64 == 0 && w[0].1 == parts[0].1));
            assert_eq!((parts[0].0, parts[3].0 + parts[3].1), (0, vocabulary));
            assert!(parts[3].1 >= parts[0].1 && parts[3].1 - parts[0].1 < 256);
        }
        assert_eq!(split_rows(248_320), 124_160);
        assert_eq!((1..=4).map(|r| part_start(248_320, 4, r)).collect::<Vec<_>>(), [62_080, 124_160, 186_240, 248_320]);
    }
    /// The parts' merge equals one argmax over the whole row at 2, 3 and 4 parts, with ties
    /// inside and across the parts, and on the pair equals merge_split.
    #[test]
    fn parts_merge_equals_the_whole_argmax() {
        let vocabulary = 64 * 13 + 40;
        for seed in 0..200usize {
            let row: Vec<f32> = (0..vocabulary).map(|i| ((i * 7919 + seed * 104_729) % 11) as f32 - 5.0).collect();
            for parts in 2..=4usize {
                let rows: Vec<_> = (0..parts).map(|p| whole(&row[part_start(vocabulary, parts, p)..part_start(vocabulary, parts, p + 1)])).collect();
                let merged = merge_parts(&rows, vocabulary, parts).unwrap()[0];
                let w = whole(&row);
                assert_eq!((merged.0, merged.1.to_bits()), (w.0, w.1.to_bits()), "seed {seed} parts {parts}");
                if parts == 2 { assert_eq!(merge_split(&rows[..1], &rows[1..], split_rows(vocabulary) as u32).unwrap()[0], merged); }
            }
        }
        assert!(merge_parts(&[(0, 1.0); 3], vocabulary, 2).is_err());
    }
    /// Ties across the four parts of the real head (248320 rows, 62080 per part): equal top logits
    /// in every part, a tie of parts 1 and 3 above parts 0 and 2, and a signed-zero tie (-0.0 in
    /// part 0, +0.0 in part 1) each give the whole head's argmax: the lowest global id, its logit
    /// bits kept (-0.0 stays -0.0).
    #[test]
    fn four_part_ties_give_the_whole_argmax() {
        let vocabulary = 248_320;
        let start = |p| part_start(vocabulary, 4, p);
        assert_eq!((start(1), start(4) - start(3)), (62_080, 62_080));
        let mut equal = vec![-1.0f32; vocabulary];
        (0..4).for_each(|p| equal[start(p) + 10] = 1.5);
        let mut upper = vec![-1.0f32; vocabulary];
        (upper[start(1) + 5], upper[start(3)], upper[start(0) + 7], upper[start(2) + 1]) = (2.0, 2.0, 1.0, 1.0);
        let mut zeros = vec![-1.0f32; vocabulary];
        (zeros[start(0) + 7], zeros[start(1)]) = (-0.0, 0.0);
        for (row, want) in [(&equal, (10, 1.5f32)), (&upper, (62_085, 2.0)), (&zeros, (7, -0.0))] {
            let parts: Vec<_> = (0..4).map(|p| whole(&row[start(p)..start(p + 1)])).collect();
            let merged = merge_parts(&parts, vocabulary, 4).unwrap()[0];
            assert_eq!((merged.0, merged.1.to_bits()), (want.0, want.1.to_bits()));
            let w = whole(row);
            assert_eq!((merged.0, merged.1.to_bits()), (w.0, w.1.to_bits()));
        }
    }
    #[test]
    fn disabled_and_headless_models_never_allocate_optional_head_state() {
        for (enabled, headless) in [(false,false),(false,true),(true,true)] {
            let value: Option<()> = load_local(enabled,headless,|| panic!("unused head allocator called")).unwrap();
            assert!(value.is_none());
            assert_eq!(retained_input_bytes(value.is_some()),0);
        }
        let calls=std::cell::Cell::new(0);
        let value=load_local(true,false,|| { calls.set(calls.get()+1); Ok(17) }).unwrap();
        assert_eq!(value,Some(17)); assert_eq!(calls.get(),1);
        assert_eq!(retained_input_bytes(value.is_some()),81920);
    }
    #[test]
    fn bounded_head_switch_is_strict_and_default_depends_on_funding() {
        assert_eq!(parse(Err(std::env::VarError::NotPresent)), Ok(None));
        assert_eq!(parse(Ok("0".into())), Ok(Some(false)));
        assert_eq!(parse(Ok("1".into())), Ok(Some(true)));
        for value in ["", "check", "true", "yes", "2", " 1", "1 "] {
            assert_eq!(parse(Ok(value.into())).unwrap_err(), "VADE_HEAD_BOUNDS must be unset, 0 or 1");
        }
        assert!(parse(Err(std::env::VarError::NotUnicode("bad".into()))).is_err());
    }
    #[test]
    fn automatic_head_never_adds_to_total_admission_for_any_eligible_tile() {
        let (twin, vocab, slots) = (655_564_800, 248_320, 16);
        let extra = twin + vocab * 8 + slots * super::RETAINED_INPUT_BYTES;
        for rows in [1, 63, 64, 65, 128, 2048, 4096, 8192] {
            let saved = crate::flash_next_workspace::saved_bytes(2560, 48, 16, rows);
            let enabled = funded(saved, twin, vocab, slots, true);
            assert_eq!(enabled, rows == 8192);
            assert!(saved >= if enabled { extra } else { 0 });
            if enabled { assert_eq!(saved - extra, 96_112_512); }
        }
        assert!(!funded(extra - 1, twin, vocab, slots, true));
        assert!(funded(extra, twin, vocab, slots, true));
        assert!(!funded(usize::MAX, twin, vocab, slots, false));
        assert!(!funded(usize::MAX, 0, vocab, slots, true));
        assert!(!funded(usize::MAX, twin, usize::MAX, slots, true));
        assert!(!funded(usize::MAX, twin, vocab, usize::MAX, true));
    }
    #[test]
    fn unfunded_or_unused_automatic_head_does_not_allocate() {
        for (saved, mtp, headless) in [(0, true, false), (usize::MAX, false, false), (usize::MAX, true, true)] {
            let enabled = parse(Err(std::env::VarError::NotPresent)).unwrap()
                .unwrap_or(funded(saved, 1024, 64, 16, mtp));
            let state: Option<()> = load_local(enabled, headless, || panic!("unfunded or unused head allocated")).unwrap();
            assert!(state.is_none());
            assert_eq!(retained_input_bytes(state.is_some()), 0);
        }
    }
}
