//! Admission and launch geometry for the staged recurrent update `step_rows_tile` (`Op::StepRows`).
//! Shape only. Any other shape runs on the kernel it had (`prompt_rows_small`, `step_rows_block` or `prompt_rows`),
//! with the same bits.
//!
//! One file, two readers: `admitted` takes a shape ONLY when its head size and head count are listed below and its
//! token count is inside the interval, and the device sweep (`kernel-bench --step-tile`, which includes this file by
//! path) runs every listed value against `step_rows_block`, with the token count at every stage edge and at the
//! interval ends. So the admitted set cannot grow without the sweep growing.
pub const THREADS: u32 = 32;
/// The prompt group cap. Device checks reach this limit.
pub const MAX_TOKENS: usize = 1024;
/// Fewer tokens keep the kernel they have: a verify pass and a step of several chats have 1 to 16 token rows
/// (`prompt_writes::SMALL`), so with this floor no decode pass changes its launch; only prompt groups take the tile.
pub const MIN_TOKENS: usize = 17;
/// Head sizes swept: the model's 128 only. The kernel keeps a column in 128 registers; any other size keeps `step_rows_block`.
pub const DIMS: &[usize] = &[128];
/// Head counts swept: the model's local head counts, and small counts.
pub const HEADS: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 33, 64];
/// Token counts the sweep runs: both ends of the interval and both sides of the stage edges (4 tokens a stage).
pub const TOKEN_EDGES: &[usize] = &[17, 18, 19, 20, 21, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1021, 1022, 1023, 1024];
/// True when `step_rows_tile` may run this shape.
pub fn admitted(tokens: usize, heads: usize, dim: usize) -> bool { (MIN_TOKENS..=MAX_TOKENS).contains(&tokens) && HEADS.contains(&heads) && DIMS.contains(&dim) }
/// Blocks for the launch: one for each head and 32 values, as `step_rows_block`.
pub fn blocks(heads: usize, dim: usize) -> Option<u32> { u32::try_from(heads.checked_mul(dim / 32)?).ok() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_shape_and_edges_are_admitted() {
        // The model: local heads of 128; a prompt group of 17 to 1,024 tokens.
        assert!(admitted(1024, 32, 128) && admitted(1024, 64, 128) && admitted(17, 32, 128) && admitted(62, 33, 128) && !admitted(256, 1, 32) && !admitted(256, 32, 64) && !admitted(256, 32, 96));
        // A verify pass or a step of several chats (1 to 16 token rows) keeps its kernel; so does a group past the cap.
        for tokens in 0..=16 { assert!(!admitted(tokens, 32, 128) && !admitted(tokens, 64, 128), "tokens {tokens}"); }
        assert!(!admitted(1025, 32, 128) && !admitted(4096, 32, 128));
        assert_eq!(MIN_TOKENS, crate::prompt_writes::SMALL as usize + 1);
        assert_eq!(blocks(32, 128), Some(128));
        assert_eq!(blocks(64, 128), Some(256));
        assert_eq!(blocks(1, 128), Some(4));
    }
    #[test]
    fn admits_only_the_swept_lists_and_interval() {
        // Every head size and head count from 0 past the largest listed value: admitted only if listed.
        for dim in 0..=DIMS[DIMS.len() - 1] + 64 { assert_eq!(admitted(65, 2, dim), DIMS.contains(&dim), "dim {dim}"); }
        for heads in 0..=HEADS[HEADS.len() - 1] + 64 { assert_eq!(admitted(65, heads, 128), HEADS.contains(&heads), "heads {heads}"); }
        for tokens in 0..=MAX_TOKENS + 64 { assert_eq!(admitted(tokens, 32, 128), (MIN_TOKENS..=MAX_TOKENS).contains(&tokens), "tokens {tokens}"); }
        // The kernel's own limit: a head size of 128 (a column is 128 registers).
        assert!(DIMS == [128]);
        assert!(TOKEN_EDGES.contains(&MIN_TOKENS) && TOKEN_EDGES.contains(&MAX_TOKENS) && TOKEN_EDGES.iter().all(|&t| (MIN_TOKENS..=MAX_TOKENS).contains(&t)));
    }
}
