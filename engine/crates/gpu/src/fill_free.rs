//! Fresh outputs whose writers store every word, so the host skips their zero fill.
//! A skip applies only when each listed writer stores every word of the output with raw stores before any read
//! (`nv_quantize`, `fp8_tile`, `nv_tile`, the mixed batch), and no typed output reference exists before those stores. Anything else keeps the fill:
//! a body that forms output slices (MX quantize, the generic expert body), and an exchange output, because a failed take can leave some or all peer words unwritten and queued consumers still read them.
//! Production skips. A proof run can make a skip run the head's zero fill, for comparison, or a poison fill of all ones, so a word
//! that no writer stored shows in the output: `set_mode`, from `resident-decode-check fill-free` and its pair flag `--poison-skipped-fills`.
use std::sync::atomic::{AtomicU8,Ordering};
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Mode {Skip=1,Zero=2,Poison=3}
static MODE:AtomicU8=AtomicU8::new(Mode::Skip as u8);
/// What a skipped fill does in this process.
pub fn mode()->Mode {
    match MODE.load(Ordering::Relaxed) {2=>Mode::Zero,3=>Mode::Poison,_=>Mode::Skip}
}
/// A proof fixture runs the same work under each mode and compares the outputs.
pub fn set_mode(mode:Mode) {MODE.store(mode as u8,Ordering::Relaxed);}
/// The word a poison fill stores. A finite BF16 store cannot produce it.
pub const POISON:u32=u32::MAX;
/// `expert_operation`: the two bodies that write through raw stores. Its generic body forms output slices and keeps the fill.
pub fn expert_operation(op:crate::expert_ops::Op,threads:u32)->bool {
    use crate::expert_ops::{FP8_TILE_THREADS,Op,QUANTIZE_LANES_THREADS};
    matches!((op,threads),(Op::Quantize,QUANTIZE_LANES_THREADS)|(Op::Fp8Serial,FP8_TILE_THREADS))
}
/// `expert_projection_batch`: `nv_tile`, or the FP8 tile body. `nv_rows_stream` and the generic body keep the fill.
pub fn projection_batch(op:crate::expert_ops::Op,kind:crate::resident_launch::Kind,threads:u32)->bool {
    use crate::resident_launch::Kind;
    kind==Kind::NvTile || kind==Kind::Expert && expert_operation(op,threads)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_ops::{FP8_TILE_THREADS,Op,QUANTIZE_LANES_THREADS};
    use crate::resident_launch::Kind;
    #[test]
    fn only_the_raw_store_bodies_skip_the_fill() {
        assert!(expert_operation(Op::Quantize,QUANTIZE_LANES_THREADS) && expert_operation(Op::Fp8Serial,FP8_TILE_THREADS));
        for (op,threads) in [(Op::Quantize,32),(Op::Fp8Serial,32),(Op::NvProject,FP8_TILE_THREADS),(Op::NvProject,QUANTIZE_LANES_THREADS),(Op::Fp8Project,FP8_TILE_THREADS),(Op::PlainSerial,FP8_TILE_THREADS),(Op::SwiGlu,QUANTIZE_LANES_THREADS),(Op::Fold,QUANTIZE_LANES_THREADS)] {
            assert!(!expert_operation(op,threads),"{op:?} {threads}");
        }
        assert!(projection_batch(Op::NvProject,Kind::NvTile,160) && projection_batch(Op::Fp8Serial,Kind::Expert,FP8_TILE_THREADS));
        assert!(!projection_batch(Op::NvProject,Kind::NvStream,256) && !projection_batch(Op::NvProject,Kind::Expert,32) && !projection_batch(Op::Fp8Serial,Kind::Expert,32));
    }
    #[test]
    fn production_skips_until_a_proof_run_sets_a_mode() {
        assert_eq!(mode(),Mode::Skip);
    }
}
