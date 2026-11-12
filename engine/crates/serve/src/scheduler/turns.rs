//! The row budget of a tick on an engine that has no shared batch.

/// Rows of one tick. One chat: `lone`. Several chats: `shared` when at least `min` of them decode
/// (the engine joins their rows in one step), else one row: the chats take turns, and a token is
/// written when its own step ends.
pub(super) fn turn_rows(chats: usize, decoders: usize, lone: usize, shared: usize, min: usize) -> usize {
    if chats == 1 { lone } else if decoders >= min { shared.max(1) } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::turn_rows;

    #[test]
    fn a_tick_shares_rows_only_when_enough_chats_decode() {
        // One chat alone takes the lone-chat rows, whatever the shared numbers are.
        assert_eq!(turn_rows(1, 1, 16, 8, 3), 16);
        assert_eq!(turn_rows(1, 0, 16, 8, 3), 16);
        // Two chats in decode on an engine that joins three or more: one row a tick, the turns.
        assert_eq!(turn_rows(2, 2, 16, 8, 3), 1);
        // Four chats open, two still in their prompts: the two decoders keep the turns.
        assert_eq!(turn_rows(4, 2, 16, 8, 3), 1);
        // Three or more decoders: one row for each of up to eight chats.
        assert_eq!(turn_rows(3, 3, 16, 8, 3), 8);
        assert_eq!(turn_rows(8, 8, 16, 8, 3), 8);
        // An engine with no batch step (the defaults: one shared row, least count one) keeps one row.
        assert_eq!(turn_rows(2, 2, 1, 1, 1), 1);
        assert_eq!(turn_rows(8, 0, 1, 1, 1), 1);
        assert_eq!(turn_rows(8, 8, 1, 0, 1), 1);
    }
}
