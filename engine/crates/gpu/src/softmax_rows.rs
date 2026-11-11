//! Grouped softmax uses the plain block-per-head body. Admission bounds the
//! head count to 64 and width to 1..=2051. The block owns one head,
//! so additional token rows add independent blocks and do not change its fold.
//! Token counts: 1 to MAX_TOKENS, and the big group sizes of `prompt_admission::GROUP_BIG` (512, 768, 1,024),
//! each checked on a device against the generic kernel (`resident-decode-check softmax-prompt`).
use crate::prompt_ops::{Op,Shape};
pub const MAX_TOKENS:u32=256;
pub const HEADS:u32=64;
pub const SLOTS:u32=2051;
pub fn admitted(s:Shape)->bool {
    s.op==Op::SoftmaxRows && ((1..=MAX_TOKENS).contains(&s.tokens) || crate::prompt_admission::big_group(s.tokens as usize)) && s.heads==HEADS
        && s.slots==SLOTS && (1..=SLOTS).contains(&s.cols) && s.lengths().is_some()
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn grouped_softmax_admission_bounds_prompt_and_partial_groups() {
        for tokens in [0,1,16,17,31,32,62,63,64,127,128,255,256,257,511,512,513,767,768,769,1023,1024,1025,2048] {for heads in 63..=65 {for cols in 0..=2052 {for slots in [0,2050,2051,2052] {
            let s=Shape {heads,cols,slots,..Shape::new(Op::SoftmaxRows,tokens as usize).unwrap()};
            // 1 to 256 tokens, or a big group of exactly 512, 768 or 1,024 tokens.
            assert_eq!(admitted(s),((1..=MAX_TOKENS).contains(&tokens) || matches!(tokens,512|768|1024)) && heads==64 && (1..=2051).contains(&cols) && slots==2051);
        }}}}
        assert!(!admitted(Shape {heads:64,cols:1040,slots:2051,..Shape::new(Op::ValuesRows,4).unwrap()}));
    }
}
