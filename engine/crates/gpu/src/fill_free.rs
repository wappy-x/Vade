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
    use crate::expert_ops::{fp8_row_threads,Op,QUANTIZE_LANES_THREADS};
    matches!((op,threads),(Op::Quantize,QUANTIZE_LANES_THREADS)) || op==Op::Fp8Serial && fp8_row_threads(threads)
}
/// `expert_projection_batch`: `nv_tile`, or the FP8 tile body. `nv_rows_stream` and the generic body keep the fill.
pub fn projection_batch(op:crate::expert_ops::Op,kind:crate::resident_launch::Kind,threads:u32)->bool {
    use crate::resident_launch::Kind;
    kind==Kind::NvTile || matches!(kind,Kind::Expert|Kind::Fp8Tile) && expert_operation(op,threads)
}
/// An Inkling block arm (single-row `resident_launch_block_arms` and rows entries `block_rows`, `block_scores`,
/// `block_softmax`, `block_values` in `kernels-inkling/src/lib.rs`): every path stores each word of the checked
/// `units * words` output with a raw store before any read (generic `BLOCK_ARMS`/`BLOCK_ROWS` write `0.0` per word
/// before forming any slice; `BLOCK_SCORES`, `BLOCK_SOFTMAX`, `BLOCK_VALUES`, `row_dots`, `RelBias` and `RouterPick`
/// write every output word via raw pointers before any read), and no command reads its own output.
/// MEASURED: a proof run (`resident-decode-check loop-token-poison`) stores all ones in each skipped output and gives the
/// stream and logit words of the trace.
pub fn block_arm(op:crate::causal_ops::Op,_rows_launch:bool)->bool {crate::block_ops::is_block(op)}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_ops::{Op,QUANTIZE_LANES_THREADS};
    // A block of the block FP8 row body (4, 8, 16 or 32 warps).
    const FP8_TILE_THREADS:u32=256;
    use crate::resident_launch::Kind;
    #[test]
    fn only_the_one_row_block_arms_skip_their_fill() {
        use crate::causal_ops::Op as C;
        for op in [C::HeadClip,C::RelBias,C::ShortConv,C::WindowScores,C::SinkSoftmax,C::WindowValues,C::MixBlend,C::GateBlend,C::CappedGate,C::RouteFold,C::Softcap,C::PlainNorm,C::RoundedNorm,C::RowDots,C::RouterPick] {
            assert!(block_arm(op,false) && block_arm(op,true),"{op:?}");
        }
        for op in [C::Route,C::Pool,C::Select,C::Scores,C::Values,C::Project,C::Conv] {assert!(!block_arm(op,false) && !block_arm(op,true),"{op:?}");}
    }
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
