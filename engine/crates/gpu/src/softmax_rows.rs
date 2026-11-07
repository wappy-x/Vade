//! Grouped softmax uses the plain block-per-head body. Admission matches the
//! device sweep: every row count 1..=16, head count 64, and width 1..=2051.
use crate::prompt_ops::{Op,Shape};
pub const MAX_TOKENS:u32=16;
pub const HEADS:u32=64;
pub const SLOTS:u32=2051;
pub fn admitted(s:Shape)->bool {
    s.op==Op::SoftmaxRows && (1..=MAX_TOKENS).contains(&s.tokens) && s.heads==HEADS
        && s.slots==SLOTS && (1..=SLOTS).contains(&s.cols) && s.lengths().is_some()
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn grouped_softmax_admission_is_the_device_sweep() {
        for tokens in 0..=17 {for heads in 63..=65 {for cols in 0..=2052 {for slots in [0,2050,2051,2052] {
            let s=Shape {heads,cols,slots,..Shape::new(Op::SoftmaxRows,tokens as usize).unwrap()};
            assert_eq!(admitted(s),(1..=16).contains(&tokens) && heads==64 && (1..=2051).contains(&cols) && slots==2051);
        }}}}
        assert!(!admitted(Shape {heads:64,cols:1040,slots:2051,..Shape::new(Op::ValuesRows,4).unwrap()}));
    }
}
