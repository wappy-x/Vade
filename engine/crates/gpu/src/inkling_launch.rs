//! The launch ABI of the Inkling image (crate `kernels-inkling`): its recorded-launch kind numbers and the layout of
//! one recorded command. One source for the host and the GPU. The general image reads `resident_launch.rs`, which
//! stays byte for byte private main's; the host test below ties these numbers and this layout to the host's `Kind`
//! and `Command`, so a command the host records is the command this image replays.
//!
//! How a seat adds an entry to the Inkling image: (1) one kind number here and the same number in the host's `Kind`
//! (`resident_launch_host.rs`, a line marked `// INKLING`, and its entry name in `Kind::kernel`); (2) in
//! `kernels-inkling/src/lib.rs` one arm in the `match` of `run` and one `entry` line. Its row in `REGISTERS.txt`.
/// `resident_launch_block_arms`: the 14 Inkling block arms, one launch.
pub const BLOCK_ARMS:u32=89;
/// `resident_launch_plain_packed`: the packed row product of one expert matrix (`packed_product`).
pub const PLAIN_PACKED:u32=88;
/// `resident_launch_plain_packed_group`: the packed row products of up to 32 matrices of one shape in one launch.
pub const PLAIN_PACKED_GROUP:u32=95;
/// `resident_launch_plain_packed_picked`: the grouped packed product whose experts the route picked on the device.
pub const PLAIN_PACKED_PICKED:u32=96;
/// `resident_launch_block_rows`: many rows of one block arm in one launch.
pub const BLOCK_ROWS:u32=91;
/// `resident_launch_plain_staged`: the staged BF16 row product (`staged_product`), one matrix with one input row.
pub const PLAIN_STAGED:u32=90;
/// `resident_launch_plain_staged_rows`: the rows tile of the staged BF16 product, one matrix with many input rows.
pub const PLAIN_STAGED_ROWS:u32=92;
/// `resident_launch_plain_staged_narrow`: the narrow staged BF16 row product (`staged_narrow`), one matrix of few rows
/// with one input row.
pub const PLAIN_STAGED_NARROW:u32=93;
/// `resident_launch_plain_packed_rows`: the expert runs of a visit in one launch of the rows tile, each run's matrix
/// read from its packed unit (`packed_product`, THE PACKED ROWS TILE).
pub const PLAIN_PACKED_ROWS:u32=100;
/// `resident_launch_embed_rows`: the stored BF16 embedding rows of the tokens of a step, widened to 32-bit words.
pub const EMBED_ROWS:u32=102;
/// `resident_launch_block_scores`: `WindowScores` at rows with the key rows staged a tile at a time (`block_scores`).
pub const BLOCK_SCORES:u32=97;
/// `resident_launch_block_softmax`: `SinkSoftmax` at rows, a block a (row, head) (`block_attention::softmax_rows_stage`).
pub const BLOCK_SOFTMAX:u32=98;
/// `resident_launch_plain_staged_few`: the small-batch BF16 product (`few_product`), one matrix with 2 to 16 input rows.
pub const PLAIN_STAGED_FEW:u32=101;
/// `resident_launch_plain_staged_one`: the one-row BF16 product (`few_product`, one input row), with the panel form.
pub const PLAIN_STAGED_ONE:u32=114;
/// `resident_launch_block_values`: `WindowValues` at rows with the value rows staged a block of distances at a time
/// (`block_values`).
pub const BLOCK_VALUES:u32=99;
/// `resident_launch_plain_tc_rows`: SM121a BF16 tensor-core MMA multi-row product (`tc_rows`).
pub const PLAIN_TC_ROWS:u32=103;
/// `resident_launch_moe_group_route`: deterministic GPU MoE router grouping (`tc_rows`).
pub const MOE_GROUP_ROUTE:u32=104;
/// `resident_launch_flash_window`: SM121a BF16 tensor-core fused causal/sliding-window attention (`flash_window`).
pub const FLASH_WINDOW_ATTN:u32=105;
/// `resident_launch_plain_packed_tc_rows`: SM121a BF16 tensor-core MMA packed MoE product (`tc_rows`).
pub const PLAIN_PACKED_TC_ROWS:u32=106;
/// `resident_launch_fast_mix_blend`: zero-stack cached-softmax `MixBlend` with optional transpose (`fast_ops`).
pub const FAST_MIX_BLEND:u32=107;
/// `resident_launch_fast_row_dots`: warp-parallel list-form `RowDots` (`fast_ops`).
pub const FAST_ROW_DOTS:u32=108;
/// `resident_launch_fast_norm`: zero-stack bit-exact `PlainNorm` and `RoundedNorm` (`fast_ops`).
pub const FAST_NORM:u32=109;
/// `resident_launch_fast_capped_gate`: zero-stack bit-exact `CappedGate` (`fast_ops`).
pub const FAST_CAPPED_GATE:u32=110;
/// `resident_launch_fast_router_pick`: warp-parallel bit-exact `RouterPick` (`fast_ops`).
pub const FAST_ROUTER_PICK:u32=111;
/// `resident_launch_fast_stream_ops`: zero-stack bit-exact window `ShortConv`, `RouteFold`, `GateBlend` (`fast_ops`).
pub const FAST_STREAM_OPS:u32=112;
pub const DRAFT_LANES:u32=113;
/// `resident_launch_plain_tc_big`: SM121a BF16 tensor-core MMA big-tile dense product for prefill (`tc_rows`).
pub const PLAIN_TC_BIG:u32=115;
/// `inkling_draft_panel`: the lane-interleaved NVFP4 draft panel products (`draft_ops::DraftOp::Panel`).
pub const DRAFT_PANEL:u32=116;
/// `inkling_draft_head_fold`: the combined vocabulary pass of the folded draft head (`draft_ops::DraftOp`).
pub const DRAFT_HEAD_FOLD:u32=117;
/// `resident_launch_flash_window_split16`: GQA-packed Split-KV tensor-core decode/verify attention (`<= 16` queries/KV-head).
pub const FLASH_WINDOW_SPLIT16:u32=118;
/// `resident_launch_flash_window_split32`: GQA-packed Split-KV tensor-core decode/verify attention (`<= 32` queries/KV-head).
pub const FLASH_WINDOW_SPLIT32:u32=119;
/// `resident_launch_flash_window_reduce`: Split-KV reduction with exact sink + output gate (`flash_window`).
pub const FLASH_WINDOW_REDUCE:u32=120;
/// `resident_launch_prefill_packed_tc`: bit-exact W4A16/W8A16 BF16 tensor-core prefill MoE GEMM, packer, and fold (`prefill_moe`).
pub const PREFILL_PACKED_TC:u32=121;
/// One recorded command: the layout of the host's `resident_launch::Command` (host test).
#[repr(C,align(16))]
#[derive(Clone,Copy)]
pub struct Command {pub kind:u32,pub reserved:[u32;3],pub payload:[u8;1024]}
/// The 32-bit words of one Command: the length of the one-command table of a direct launch.
pub const COMMAND_WORDS:usize=core::mem::size_of::<Command>()/4;
/// The per-chat KV addresses in a two-dimensional scores or values command.
pub const PLANES_AT:usize=128;
pub const PLANES_MAX:usize=16;
const _:()=assert!(PLANES_AT>=core::mem::size_of::<crate::block_ops::device::RowsArguments>() && PLANES_AT+8*PLANES_MAX<=1024);
/// Zero-copy ring + delta plane descriptor for `BLOCK_SCORES` / `BLOCK_VALUES` verify passes:
/// 4 u64 words: `[delta_ptr (0 = single plane), ring_rows, start_slot, prefix]`.
pub const RING_DELTA_AT:usize=256;
const _:()=assert!(RING_DELTA_AT>=PLANES_AT+8*PLANES_MAX && RING_DELTA_AT+32<=1024);
/// The mark in `reserved[2]` of a joined `ShortConv` arms command (`reserved[0]` the clip or norm) that also writes
/// the product of the token into its ring row and the finished row, packed, into its row of a key or value plane
/// (the host's `conv_finish_into`).
pub const CONV_FINISH:u32=1;
/// Payload words of that command: the address of the product and the address of the plane row.
pub const FINISH_PRODUCT_WORD:usize=17;
pub const FINISH_PLANE_WORD:usize=18;
/// The mark of two finish commands that the recorder joined into one launch of twice the grid (the key and then the
/// value of one token). The second command's operands are the payload words from `PAIR_AT`: its ring, made taps,
/// fixed taps, output, product and plane row addresses, then its clip or norm (low half) and parameter bits (high).
pub const CONV_FINISH_PAIR:u32=2;
pub const PAIR_AT:usize=19;
const _:()=assert!(8*(PAIR_AT+7)<=PLANES_AT+8*PLANES_MAX && PAIR_AT>FINISH_PLANE_WORD);
const _:()=assert!(8*FINISH_PRODUCT_WORD>=core::mem::size_of::<crate::block_ops::device::RowsArguments>() && FINISH_PLANE_WORD==FINISH_PRODUCT_WORD+1);
#[cfg(test)] mod tests {
    use super::*;
    use crate::resident_launch::{Command as Host,Kind};
    #[test] fn the_inkling_abi_is_the_hosts() {
        assert_eq!([Kind::BlockArms as u32,Kind::BlockRows as u32,Kind::PlainStaged as u32,Kind::PlainStagedRows as u32,Kind::PlainStagedFew as u32,Kind::PlainStagedOne as u32,Kind::PlainStagedNarrow as u32,Kind::PlainPacked as u32,Kind::PlainPackedGroup as u32,Kind::PlainPackedPicked as u32,Kind::BlockScores as u32,Kind::BlockValues as u32,Kind::BlockSoftmax as u32,Kind::PlainPackedRows as u32,Kind::EmbedRows as u32,Kind::PlainTcRows as u32,Kind::MoeGroupRoute as u32,Kind::FlashWindowAttn as u32,Kind::FlashWindowSplit16 as u32,Kind::FlashWindowSplit32 as u32,Kind::FlashWindowReduce as u32,Kind::PlainPackedTcRows as u32,Kind::PrefillPackedTc as u32,Kind::FastMixBlend as u32,Kind::FastRowDots as u32,Kind::FastNorm as u32,Kind::FastCappedGate as u32,Kind::FastRouterPick as u32,Kind::FastStreamOps as u32,Kind::DraftLanes as u32,Kind::PlainTcBig as u32,Kind::DraftPanel as u32,Kind::DraftHeadFold as u32],[BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_FEW,PLAIN_STAGED_ONE,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_VALUES,BLOCK_SOFTMAX,PLAIN_PACKED_ROWS,EMBED_ROWS,PLAIN_TC_ROWS,MOE_GROUP_ROUTE,FLASH_WINDOW_ATTN,FLASH_WINDOW_SPLIT16,FLASH_WINDOW_SPLIT32,FLASH_WINDOW_REDUCE,PLAIN_PACKED_TC_ROWS,PREFILL_PACKED_TC,FAST_MIX_BLEND,FAST_ROW_DOTS,FAST_NORM,FAST_CAPPED_GATE,FAST_ROUTER_PICK,FAST_STREAM_OPS,DRAFT_LANES,PLAIN_TC_BIG,DRAFT_PANEL,DRAFT_HEAD_FOLD]);
        assert_eq!([Kind::BlockArms.kernel(),Kind::BlockRows.kernel(),Kind::PlainStaged.kernel(),Kind::PlainStagedRows.kernel(),Kind::PlainStagedFew.kernel(),Kind::PlainStagedOne.kernel(),Kind::PlainStagedNarrow.kernel(),Kind::PlainPacked.kernel(),Kind::PlainPackedGroup.kernel(),Kind::PlainPackedPicked.kernel(),Kind::BlockScores.kernel(),Kind::BlockValues.kernel(),Kind::BlockSoftmax.kernel(),Kind::PlainPackedRows.kernel(),Kind::EmbedRows.kernel(),Kind::PlainTcRows.kernel(),Kind::MoeGroupRoute.kernel(),Kind::FlashWindowAttn.kernel(),Kind::FlashWindowSplit16.kernel(),Kind::FlashWindowSplit32.kernel(),Kind::FlashWindowReduce.kernel(),Kind::PlainPackedTcRows.kernel(),Kind::PrefillPackedTc.kernel(),Kind::FastMixBlend.kernel(),Kind::FastRowDots.kernel(),Kind::FastNorm.kernel(),Kind::FastCappedGate.kernel(),Kind::FastRouterPick.kernel(),Kind::FastStreamOps.kernel(),Kind::DraftLanes.kernel(),Kind::PlainTcBig.kernel(),Kind::DraftPanel.kernel(),Kind::DraftHeadFold.kernel()],["resident_launch_block_arms","resident_launch_block_rows","resident_launch_plain_staged","resident_launch_plain_staged_rows","resident_launch_plain_staged_few","resident_launch_plain_staged_one","resident_launch_plain_staged_narrow","resident_launch_plain_packed","resident_launch_plain_packed_group","resident_launch_plain_packed_picked","resident_launch_block_scores","resident_launch_block_values","resident_launch_block_softmax","resident_launch_plain_packed_rows","resident_launch_embed_rows","resident_launch_plain_tc_rows","resident_launch_moe_group_route","resident_launch_flash_window","resident_launch_flash_window_split16","resident_launch_flash_window_split32","resident_launch_flash_window_reduce","resident_launch_plain_packed_tc_rows","resident_launch_prefill_packed_tc","resident_launch_fast_mix_blend","resident_launch_fast_row_dots","resident_launch_fast_norm","resident_launch_fast_capped_gate","resident_launch_fast_router_pick","resident_launch_fast_stream_ops","inkling_draft_lanes","resident_launch_plain_tc_big","inkling_draft_panel","inkling_draft_head_fold"]);
        use core::mem::{align_of,offset_of,size_of};
        assert_eq!((size_of::<Command>(),align_of::<Command>()),(size_of::<Host>(),align_of::<Host>()));
        assert_eq!([offset_of!(Command,kind),offset_of!(Command,reserved),offset_of!(Command,payload)],[offset_of!(Host,kind),offset_of!(Host,reserved),offset_of!(Host,payload)]);
        // No general kind has an Inkling number.
        assert!(Kind::ALL.iter().filter(|k|[BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_FEW,PLAIN_STAGED_ONE,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_VALUES,BLOCK_SOFTMAX,PLAIN_PACKED_ROWS,EMBED_ROWS,PLAIN_TC_ROWS,MOE_GROUP_ROUTE,FLASH_WINDOW_ATTN,FLASH_WINDOW_SPLIT16,FLASH_WINDOW_SPLIT32,FLASH_WINDOW_REDUCE,PLAIN_PACKED_TC_ROWS,PREFILL_PACKED_TC,FAST_MIX_BLEND,FAST_ROW_DOTS,FAST_NORM,FAST_CAPPED_GATE,FAST_ROUTER_PICK,FAST_STREAM_OPS,DRAFT_LANES,PLAIN_TC_BIG,DRAFT_PANEL,DRAFT_HEAD_FOLD].contains(&(**k as u32))).count()==33);
    }
}
