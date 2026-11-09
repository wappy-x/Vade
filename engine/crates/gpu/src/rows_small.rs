//! Admission and launch geometry for the small-count KDA prompt rows kernel `prompt_rows_small`: StepRows with dim 128 (the fused step in a loop over
//! the tokens). Shape only. The admitted shapes are exactly the lists of `small_sweep` (head counts, dim 128, 1 to 16 tokens), which the device sweep
//! (`kernel-bench --rows-small kda`) walks in full against `prompt_rows`; any other shape keeps lane R's `step_rows_block` or `prompt_rows`.
//! (Wrapped Prepare, GatedNorm and Route have lane P's `prompt_prepare`, `prompt_gated_norm` and `prompt_route`, which measured equal or faster.)
use crate::prompt_ops::{Op, Shape};
use crate::small_sweep::{step_swept, MAX_TOKENS};
pub const STEP_DIM: usize = 128;
/// True when `prompt_rows_small` may run this shape.
pub fn admitted(s: &Shape) -> bool { (1..=MAX_TOKENS).contains(&(s.tokens as usize)) && s.op == Op::StepRows && step_swept(s.heads as usize, s.dim as usize) }
/// Blocks and threads: `heads * 4` warps of 32 threads.
pub fn geometry(s: &Shape) -> (u32, u32) { (s.heads * (STEP_DIM as u32 / 32), 32) }
#[cfg(test)]
mod tests {
    use super::*;
    use crate::small_sweep::*;
    fn step_rows(h: u32, d: u32, tokens: usize) -> Shape { let mut s = Shape::new(Op::StepRows, tokens).unwrap(); s.heads = h; s.dim = d; s }
    #[test]
    fn the_model_kda_layers_are_admitted_with_their_geometry() {
        for tokens in 1..=16 { assert!(admitted(&step_rows(64, 128, tokens))); }
        assert_eq!(geometry(&step_rows(64, 128, 4)), (256, 32));
        assert_eq!(geometry(&step_rows(3, 128, 1)), (12, 32));
    }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        for h in 0..=70usize { for d in [0usize, 1, 64, 127, 128, 129, 256] { assert_eq!(admitted(&step_rows(h as u32, d as u32, 16)), step_swept(h, d), "heads {h} dim {d}"); } }
        for h in 0..=70usize { assert_eq!(admitted(&step_rows(h as u32, 128, 4)), STEP_HEADS.contains(&h), "heads {h}"); }
        // The reviewer's example (heads 4, tokens 16) is in the list; boundaries of heads, dim and tokens are not.
        assert!(admitted(&step_rows(4, 128, 16)) && !admitted(&step_rows(65, 128, 4)) && !admitted(&step_rows(64, 64, 4)) && !admitted(&step_rows(5, 128, 4)) && !admitted(&step_rows(0, 128, 4)));
        for tokens in 0..=20usize { assert_eq!(admitted(&step_rows(64, 128, tokens)), (1..=16).contains(&tokens), "tokens {tokens}"); }
        // Other operations are not this kernel's.
        assert!(!admitted(&Shape::new(Op::ProjectRows, 4).unwrap()) && !admitted(&Shape::new(Op::Transpose, 4).unwrap()));
    }
}
