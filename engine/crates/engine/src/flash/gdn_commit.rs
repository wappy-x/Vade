//! Pure commit math for the table GDN commit (gdn_replay::table_commit):
//! which form restores the state after a partial accept, and the copy rows
//! of the Swap form. Host-visible so its tests run without a GPU.

const CHUNK: usize = 64;
const DIM: usize = 128;

/// How a partial commit restores GDN state after a verify window of `tokens`
/// rows at absolute position `base` when `prefix` rows stay (commit_form).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommitForm {
    /// S and the accepted tail rows already stand in the live state: rebuild
    /// the accepted history from the tape in one launch (prefix 0 copies the
    /// entry taps).
    Slots,
    /// Swap the live states with the entry snapshots, rebuild the history
    /// from the tape, then copy the accepted tail slots back (tail_slot_rows).
    Swap,
    /// The tape replay (Gdn::replay_history / replay_next), as before.
    Replay,
}

/// Pure form choice; `slots` records at verify time that the table commit
/// can run. The scan form (tokens over CHUNK_ROWS_LIMIT) over row 63 keeps the
/// early rows' tail words in shared memory only, so an accept before row 63
/// replays. Chunk form is part of `slots` (gdn_replay::table_commit).
pub fn commit_form(slots: bool, base: usize, tokens: usize, prefix: usize) -> CommitForm {
    if !slots || tokens == 0 || prefix > tokens { return CommitForm::Replay; }
    let n0 = base % CHUNK;
    if n0 + tokens < CHUNK || prefix > CHUNK - 1 - n0 { return CommitForm::Slots; }
    if tokens <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT { CommitForm::Swap } else { CommitForm::Replay }
}

/// flash_hybrid_copy rows for the accepted tail slots [base % 64, base % 64 +
/// prefix) of one layer: key rows and v' rows into `wide`, gamma into
/// `narrow`. Offsets follow chunk_state_floats at this rank's heads and keys.
pub fn tail_slot_rows(
    source: u64, destination: u64, heads: usize, keys: usize, base: usize, prefix: usize,
    wide: &mut Vec<spark_gpu_probe::kernel_params::HybridCopyRow>,
    narrow: &mut Vec<spark_gpu_probe::kernel_params::HybridCopyRow>,
) {
    type Row = spark_gpu_probe::kernel_params::HybridCopyRow;
    let n0 = base % CHUNK;
    let (tk, tw) = (heads * DIM * DIM, heads * DIM * DIM + keys * CHUNK * DIM);
    let tg = tw + heads * CHUNK * DIM;
    let push = |rows: &mut Vec<Row>, at: usize, words: usize| {
        rows.push(Row { source: source + (at * 4) as u64, destination: destination + (at * 4) as u64,
            words: words as u32, padding: 0 });
    };
    for kh in 0..keys { push(wide, tk + kh * CHUNK * DIM + n0 * DIM, prefix * DIM); }
    for h in 0..heads {
        push(wide, tw + h * CHUNK * DIM + n0 * DIM, prefix * DIM);
        push(narrow, tg + h * CHUNK + n0, prefix);
    }
}

#[cfg(test)]
mod tests {
    use super::{commit_form, tail_slot_rows, CommitForm};
    #[test]
    fn commit_form_cases() {
        // No table commit (old PTX or chunk form off): always the old replay.
        assert_eq!(commit_form(false, 0, 4, 2), CommitForm::Replay);
        assert_eq!(commit_form(true, 0, 0, 0), CommitForm::Replay);
        assert_eq!(commit_form(true, 0, 4, 5), CommitForm::Replay);
        // Inside one chunk: only the history rebuild runs, at any width.
        assert_eq!(commit_form(true, 60, 3, 2), CommitForm::Slots);
        assert_eq!(commit_form(true, 40, 16, 9), CommitForm::Slots);
        assert_eq!(commit_form(true, 60, 3, 0), CommitForm::Slots);
        // Over row 63 (base 61: rows at 61, 62, 63, 0): an accept through
        // row 63 keeps the live S; a shorter accept swaps or replays.
        assert_eq!(commit_form(true, 61, 4, 3), CommitForm::Slots);
        assert_eq!(commit_form(true, 61, 4, 4), CommitForm::Slots);
        assert_eq!(commit_form(true, 61, 4, 2), CommitForm::Swap);
        assert_eq!(commit_form(true, 61, 4, 0), CommitForm::Swap);
        assert_eq!(commit_form(true, 55, 16, 4), CommitForm::Replay);
        assert_eq!(commit_form(true, 55, 16, 9), CommitForm::Slots);
        // Position 0 of a chunk never crosses at verify widths.
        assert_eq!(commit_form(true, 128, 8, 1), CommitForm::Slots);
    }
    #[test]
    fn tail_slot_rows_cover_the_accepted_slots_once() {
        let (heads, keys) = (6usize, 2usize);
        let (mut wide, mut narrow) = (Vec::new(), Vec::new());
        tail_slot_rows(0x1000, 0x2000, heads, keys, 61, 2, &mut wide, &mut narrow);
        assert_eq!((wide.len(), narrow.len()), (keys + heads, heads));
        let total = spark_gpu_probe::gdn_params::chunk_state_floats(heads, keys);
        let mut seen = vec![0u8; total];
        for row in wide.iter().chain(&narrow) {
            assert_eq!(row.destination - row.source, 0x1000);
            let at = ((row.source - 0x1000) / 4) as usize;
            seen[at..at + row.words as usize].iter_mut().for_each(|s| *s += 1);
        }
        // Exactly slots 61 and 62 of every tail run, nothing in S.
        let (tk, tw, tg) = (heads * 128 * 128, heads * 128 * 128 + keys * 64 * 128,
            heads * 128 * 128 + keys * 64 * 128 + heads * 64 * 128);
        for (at, &s) in seen.iter().enumerate() {
            let want = (tk..tw).contains(&at) && (61..63).contains(&(at % (64 * 128) / 128))
                || (tw..tg).contains(&at) && (61..63).contains(&(at % (64 * 128) / 128))
                || at >= tg && (61..63).contains(&(at % 64));
            assert_eq!(s, want as u8, "word {at}");
        }
    }
}
