//! The HOST view of `resident_launch.rs`: private main's file, plus the lines of the Inkling block arms (marked `// INKLING`; // INKLING
//! a main line they replace is kept above them as `//INKLING-MAIN: `). The general kernel image builds from // INKLING
//! `resident_launch.rs`, which stays byte for byte private main's; Inkling's kernels build in their own image // INKLING
//! (`kernels-inkling`). The test `host_views` in lib.rs proves the rule. // INKLING
//! Stable indirect launch ABI. Padding is initialized and never copied from Rust structs.
#[repr(u32)]
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub enum Kind {Causal=1,Expert,Plain,Mx,Round,Copy,Put,Take,Prepare,Collapse,Combine,RmsPartials,RmsApply,Elementwise,CopyWords,ZeroWords,Mixed,NvStream,CausalFast=19,NvTile=20,DeviceRoute=21,TakeScatter=22,PutWide=23,PlainBf16=26,MhcBoundaryRms=0,MhcBoundaryDots=24,MhcBoundaryFinish=25,
    /// The unchecked-payload put of the mHC host lane (tp_mailbox_put) and its prefetch-only launch (chain_shadow_prefetch).
    PutPlain=27,ShadowPrefetch=28,
    /// One exchange message's round, own-row placement and wide put in one launch (tp_mailbox_put_join).
    PutJoin=29,
    /// The same prefetch-only launch on a parallel branch of its graph piece: it forks after the command before it and
    /// joins at the piece's end, so the commands after it do not wait for it.
    ShadowSide=30,
    /// Rider bundle kinds (pointwise copy/round, ordinary norm tail, head-local causal chain), renumbered after main's 27..30.
    Pointwise=31,NormTail=32,CausalChain=33,
    /// Mixed-plain kinds (BF16 mixed batch and plain BF16 batch).
    MixedBf16=34,PlainBatch=35,
    /// Fixed-order latent absorption (SUM-ORDER v1): the dense query and value projections, then the latent scores and values.
    /// The exact train uses 30 to 35 for its own kinds. These two follow them, so the two lines of work never share a number.
    AbsorbDense=36,AbsorbAttention=37,
    /// D1: the whole one-row mHC boundary in one launch (mhc_boundary_fused).
    MhcPhaseRms=61,MhcPhaseDots=62,MhcPhaseFinish=63,MhcBoundaryFused=38,
    /// The pair split (D2). NvQuarters: the raw quarter accumulators of second matrices (nv_cols_q4 / q2 / q1; payload
    /// word NV_QUARTERS_WORD holds the quarters a row gets). QuarterFold: the fold inside each quarter and the join of an
    /// owner's quarters (nv_quarter_fold). ReduceTake: the take of the one sum exchange a sublayer (tp_reduce_take).
    NvQuarters=39,QuarterFold=40,ReduceTake=41,
    /// Original row kernels with indirect arguments. Their grids and arithmetic stay fixed.
    MxRows2,MxRows4,MxRows8,MxRows8w,MixedRows,Fp8RowsSmall,Fp8RowsMulti,Fp8RowsFew,PlainRowsSmall,PromptRows,RowPrepare,RowGatedNorm,RowRoute,RowFold,RowPlace,RowPermute,RowSelect,TransposeSmem,DeviceRouteRows,
    /// One launch of 2 to 8 same-shape causal chains, or same-shape recurrent causal-fast calls. Each block row runs one member's own command.
    CausalGroup=64,
    /// FOLD ROWS: the plain fold batch for 2 to 8 source rows, BF16 weight patterns and FP32 weights (row kernels with indirect arguments).
    FoldRows=65,FoldRowsF32=66,
    /// The indexer's pool scores of few tokens, a warp a (token, pool) (prompt_pool_scores; a row kernel with indirect arguments).
    RowPoolScores=67,
    /// LATENT ROWS (seat RB3, P7): the latent absorption of a check pass of 1 to 8 rows on the one-row step's bodies
    /// (latent_rows_dense: query and project; latent_rows_attention: scores and values), row kernels with indirect
    /// arguments. Their grids and arithmetic stay fixed. 73 to 81 are other seats' kinds tonight.
    LatentRowsDense=86,LatentRowsAttention=87,
    /// SHARE ROWS: block-FP8 rows with the same checked direct and recorded arguments.
    Fp8RowsShare=71,
    /// SHARE ROWS: the existing MXFP8 tile for 5 to 8 decode rows.
    MxRowsTile=68,
    /// Block FP8 decode tile on the tensor-core contract, distinct from main's row kinds.
    Fp8Tile=72,
    /// Grouped NVFP4 tensor-core device route builder and `fp4_cohort_t8` routed projection.
    DeviceRouteTcRows=74,Fp4CohortT8=75,
    /// A recurrent prompt layer of one chat's short row pass.
//INKLING-MAIN:     RowRecurrent=crate::recurrent_rows::RECORD_KIND}
    RowRecurrent=crate::recurrent_rows::RECORD_KIND, // INKLING
    /// The Inkling block arms (`block_ops`) in a kernel entry of their own (Ruling R6g): the payload is the causal // INKLING
    /// Arguments of one launch. // INKLING
    BlockArms=89, // INKLING
    /// The packed row product of one expert matrix in its stored 4-bit or 8-bit form (`packed_product`, Ruling R6h). // INKLING
    PlainPacked=88, // INKLING
    /// The packed row products of the picked experts of a visit in one launch: a block row of the grid a matrix. // INKLING
    PlainPackedGroup=95, // INKLING
    /// The grouped packed product whose experts the route picked on the device: block row `e` reads its expert // INKLING
    /// number from the route output (`packed_product::check_picked`). // INKLING
    PlainPackedPicked=96, // INKLING
    /// Many rows of one Inkling block arm in one launch (`block_rows`, Ruling R6g). // INKLING
    BlockRows=91, // INKLING
    /// `WindowScores` at rows with the key rows staged a tile at a time (`block_scores`): the payload is that of // INKLING
    /// `BlockRows`. // INKLING
    BlockScores=97, // INKLING
    /// `SinkSoftmax` at rows, a block a (row, head) (`block_attention::softmax_rows_stage`): the payload is that of // INKLING
    /// `BlockRows`. // INKLING
    BlockSoftmax=98, // INKLING
    /// `WindowValues` at rows with the value rows staged a block of distances at a time (`block_values`): the // INKLING
    /// payload is that of `BlockRows`. // INKLING
    BlockValues=99, // INKLING
    /// The staged BF16 row product of the Inkling image (`staged_product`, Ruling R6h). // INKLING
    PlainStaged=90, // INKLING
    /// The rows tile of the staged BF16 product of the Inkling image (`staged_product`, Ruling R6h). // INKLING
    PlainStagedRows=92, // INKLING
    /// The narrow staged BF16 row product of the Inkling image (`staged_narrow`, Ruling R6h). // INKLING
    PlainStagedNarrow=93, // INKLING
    /// The expert runs of a visit in one launch of the rows tile, each run read from its packed unit (`packed_product`, // INKLING
    /// THE PACKED ROWS TILE). // INKLING
    PlainPackedRows=100, // INKLING
    /// The stored BF16 embedding rows of the tokens of a step, widened to 32-bit words (`backend/embed.rs`). // INKLING
    EmbedRows=102, // INKLING
    /// The small-batch BF16 product of the Inkling image, 2 to 16 input rows (`few_product`, Ruling R6h). // INKLING
    PlainStagedFew=101, // INKLING
    /// The one-row BF16 product of the Inkling image, with the lossless panel form (`few_product`, Ruling R6h). // INKLING
    PlainStagedOne=114, // INKLING
    /// SM121a BF16 tensor-core MMA multi-row product (`tc_rows`, Ruling R6h). // INKLING
    PlainTcRows=103, // INKLING
    /// Deterministic GPU MoE router grouping (`tc_rows`, Ruling R6h). // INKLING
    MoeGroupRoute=104, // INKLING
    /// SM121a BF16 tensor-core fused causal/sliding-window attention (`flash_window`, Ruling R6h). // INKLING
    FlashWindowAttn=105, // INKLING
    /// SM121a BF16 tensor-core MMA packed MoE product (`tc_rows`, Ruling R6h). // INKLING
    PlainPackedTcRows=106, // INKLING
    /// Zero-stack fast Inkling block-op kernels (`fast_ops`, Ruling R6h). // INKLING
    FastMixBlend=107,FastRowDots=108,FastNorm=109,FastCappedGate=110,FastRouterPick=111,FastStreamOps=112,DraftLanes=113, // INKLING
    /// SM121a BF16 tensor-core MMA big-tile dense product for prefill (`tc_rows`, Ruling R6h). // INKLING
    PlainTcBig=115, // INKLING
    /// Lane-interleaved NVFP4 draft panel products (`draft_ops`, Ruling R6h). // INKLING
    DraftPanel=116, // INKLING
    /// The combined vocabulary pass of the folded draft head (`draft_ops`, Ruling R6h). // INKLING
    DraftHeadFold=117, // INKLING
    /// GQA-packed Split-KV tensor-core decode/verify attention (`flash_window`, Ruling R6h). // INKLING
    FlashWindowSplit16=118, // INKLING
    /// GQA-packed Split-KV tensor-core decode/verify attention (`flash_window`, Ruling R6h). // INKLING
    FlashWindowSplit32=119, // INKLING
    /// Split-KV reduction with sink and output gate (`flash_window`, Ruling R6h). // INKLING
    FlashWindowReduce=120, // INKLING
    /// Bit-exact W4A16/W8A16 BF16 tensor-core prefill MoE GEMM, packer, and fold (`prefill_moe`, Ruling R6h). // INKLING
    PrefillPackedTc=121, // INKLING
    /// High-occupancy 2-stage prefill causal/sliding-window attention (`flash_window`, Ruling R6h). // INKLING
    PrefillFlashWindow=122, // INKLING
    /// Fused prefill lazy message-meta and message-gate kernel (`prefill_gate`, Ruling R6h). // INKLING
    PrefillGateFuse=123, // INKLING
    /// SM121a BF16 tensor-core MMA product over `bf16_panels` for 33..=128 input rows (`tc_rows`, Ruling R6h). // INKLING
    PlainTcPanel64=124} // INKLING
/// The payload word of an NvQuarters command that holds the quarters a row gets (4, 2 or 1). It is the first word
/// past the command's LaunchArguments (112 bytes).
pub const NV_QUARTERS_WORD:usize=14;
/// Recorded command counts by kind. The length follows the largest kind (Kind::COUNT), so a new kind needs no size edit here.
/// std gives Default only to arrays of at most 32 elements.
#[cfg(not(target_arch="nvptx64"))] #[derive(Clone,Copy,Debug)] pub struct KindCounts(pub [u128;Kind::COUNT]);
#[cfg(not(target_arch="nvptx64"))] impl Default for KindCounts {fn default()->Self {Self([0;Kind::COUNT])}}
#[cfg(not(target_arch="nvptx64"))] impl core::ops::Deref for KindCounts {type Target=[u128;Kind::COUNT];fn deref(&self)->&[u128;Kind::COUNT] {&self.0}}
#[cfg(not(target_arch="nvptx64"))] impl core::ops::DerefMut for KindCounts {fn deref_mut(&mut self)->&mut [u128;Kind::COUNT] {&mut self.0}}
#[cfg(not(target_arch="nvptx64"))]
impl Kind {
    /// One more than the largest kind number: the length of a table indexed by kind.
    pub const COUNT:usize={let mut count=0;let mut at=0;while at<Self::ALL.len() {let next=Self::ALL[at] as usize+1;if next>count {count=next;}at+=1;}count};
//INKLING-MAIN:     pub const ALL:[Self;76]=[Self::DeviceRouteTcRows,Self::Fp4CohortT8,Self::LatentRowsDense,Self::LatentRowsAttention,Self::RowRecurrent,Self::Fp8Tile,Self::MxRowsTile,Self::Fp8RowsShare,Self::RowPoolScores,Self::FoldRows,Self::FoldRowsF32,Self::MhcPhaseRms,Self::MhcPhaseDots,Self::MhcPhaseFinish,Self::MxRows2,Self::MxRows4,Self::MxRows8,Self::MxRows8w,Self::MixedRows,Self::Fp8RowsSmall,Self::Fp8RowsMulti,Self::Fp8RowsFew,Self::PlainRowsSmall,Self::PromptRows,Self::RowPrepare,Self::RowGatedNorm,Self::RowRoute,Self::RowFold,Self::RowPlace,Self::RowPermute,Self::RowSelect,Self::TransposeSmem,Self::DeviceRouteRows,Self::NvQuarters,Self::QuarterFold,Self::ReduceTake,Self::MhcBoundaryFused,Self::AbsorbDense,Self::AbsorbAttention,Self::MixedBf16,Self::PlainBatch,Self::CausalChain,Self::NormTail,Self::Pointwise,Self::ShadowSide,Self::PutJoin,Self::PutPlain,Self::ShadowPrefetch,Self::PutWide,Self::Causal,Self::Expert,Self::Plain,Self::Mx,Self::Round,Self::Copy,Self::Put,Self::Take,Self::Prepare,Self::Collapse,Self::Combine,Self::RmsPartials,Self::RmsApply,Self::Elementwise,Self::CopyWords,Self::ZeroWords,Self::Mixed,Self::NvStream,Self::CausalFast,Self::NvTile,Self::DeviceRoute,Self::TakeScatter,Self::PlainBf16,Self::MhcBoundaryRms,Self::MhcBoundaryDots,Self::MhcBoundaryFinish,Self::CausalGroup];
    pub const ALL:[Self;112]=[Self::DeviceRouteTcRows,Self::Fp4CohortT8,Self::PlainTcPanel64,Self::PrefillGateFuse,Self::PrefillFlashWindow,Self::PrefillPackedTc,Self::FlashWindowReduce,Self::FlashWindowSplit32,Self::FlashWindowSplit16,Self::DraftHeadFold,Self::DraftPanel,Self::PlainTcBig,Self::PlainStagedOne,Self::DraftLanes,Self::FastStreamOps,Self::FastRouterPick,Self::FastCappedGate,Self::FastNorm,Self::FastRowDots,Self::FastMixBlend,Self::PlainPackedTcRows,Self::EmbedRows,Self::PlainPackedRows,Self::FlashWindowAttn,Self::MoeGroupRoute,Self::PlainTcRows,Self::BlockValues,Self::PlainStagedFew,Self::BlockSoftmax,Self::BlockScores,Self::PlainPackedPicked,Self::PlainPackedGroup,Self::PlainPacked,Self::PlainStaged,Self::PlainStagedRows,Self::PlainStagedNarrow,Self::BlockRows,Self::BlockArms,Self::LatentRowsDense,Self::LatentRowsAttention,Self::RowRecurrent,Self::Fp8Tile,Self::MxRowsTile,Self::Fp8RowsShare,Self::RowPoolScores,Self::FoldRows,Self::FoldRowsF32,Self::MhcPhaseRms,Self::MhcPhaseDots,Self::MhcPhaseFinish,Self::MxRows2,Self::MxRows4,Self::MxRows8,Self::MxRows8w,Self::MixedRows,Self::Fp8RowsSmall,Self::Fp8RowsMulti,Self::Fp8RowsFew,Self::PlainRowsSmall,Self::PromptRows,Self::RowPrepare,Self::RowGatedNorm,Self::RowRoute,Self::RowFold,Self::RowPlace,Self::RowPermute,Self::RowSelect,Self::TransposeSmem,Self::DeviceRouteRows,Self::NvQuarters,Self::QuarterFold,Self::ReduceTake,Self::MhcBoundaryFused,Self::AbsorbDense,Self::AbsorbAttention,Self::MixedBf16,Self::PlainBatch,Self::CausalChain,Self::NormTail,Self::Pointwise,Self::ShadowSide,Self::PutJoin,Self::PutPlain,Self::ShadowPrefetch,Self::PutWide,Self::Causal,Self::Expert,Self::Plain,Self::Mx,Self::Round,Self::Copy,Self::Put,Self::Take,Self::Prepare,Self::Collapse,Self::Combine,Self::RmsPartials,Self::RmsApply,Self::Elementwise,Self::CopyWords,Self::ZeroWords,Self::Mixed,Self::NvStream,Self::CausalFast,Self::NvTile,Self::DeviceRoute,Self::TakeScatter,Self::PlainBf16,Self::MhcBoundaryRms,Self::MhcBoundaryDots,Self::MhcBoundaryFinish,Self::CausalGroup]; // INKLING
    pub const ROWS:[Self;27]=[Self::LatentRowsDense,Self::LatentRowsAttention,Self::RowRecurrent,Self::MxRowsTile,Self::Fp8RowsShare,Self::RowPoolScores,Self::FoldRows,Self::FoldRowsF32,Self::MxRows2,Self::MxRows4,Self::MxRows8,Self::MxRows8w,Self::MixedRows,Self::Fp8RowsSmall,Self::Fp8RowsMulti,Self::Fp8RowsFew,Self::PlainRowsSmall,Self::PromptRows,Self::RowPrepare,Self::RowGatedNorm,Self::RowRoute,Self::RowFold,Self::RowPlace,Self::RowPermute,Self::RowSelect,Self::TransposeSmem,Self::DeviceRouteRows,];
    pub fn row(name:&str)->Option<Self> {match name {
        "latent_rows_dense"=>Some(Self::LatentRowsDense),"latent_rows_attention"=>Some(Self::LatentRowsAttention),
        "mx_rows_small2"=>Some(Self::MxRows2),
        "mx_rows_small4"=>Some(Self::MxRows4),
        "mx_rows_small8"=>Some(Self::MxRows8),
        "mx_rows_small8w"=>Some(Self::MxRows8w),
        "mx_rows_tile"=>Some(Self::MxRowsTile),
        "flash_resident_projection_bf16_rows"=>Some(Self::MixedRows),
        "flash_resident_projection_fold_rows"=>Some(Self::FoldRows),
        "flash_resident_projection_fold_rows_f32"=>Some(Self::FoldRowsF32),
        "fp8_rows_small"=>Some(Self::Fp8RowsSmall),
        "fp8_rows_multi"=>Some(Self::Fp8RowsMulti),
        "fp8_rows_few"=>Some(Self::Fp8RowsFew),
        "fp8_rows_share"=>Some(Self::Fp8RowsShare),
        "plain_rows_small"=>Some(Self::PlainRowsSmall),
        "prompt_rows"=>Some(Self::PromptRows),
        "prompt_prepare"=>Some(Self::RowPrepare),
        "prompt_gated_norm"=>Some(Self::RowGatedNorm),
        "recurrent_rows"=>Some(Self::RowRecurrent),
        "prompt_route"=>Some(Self::RowRoute),
        "prompt_fold"=>Some(Self::RowFold),
        "prompt_pool_scores"=>Some(Self::RowPoolScores),
        "prompt_place"=>Some(Self::RowPlace),
        "prompt_permute"=>Some(Self::RowPermute),
        "prompt_select_rows"=>Some(Self::RowSelect),
        "transpose_smem"=>Some(Self::TransposeSmem),
        "device_route_rows"=>Some(Self::DeviceRouteRows),
        _=>None}}
    pub fn kernel(self)->&'static str {match self {
        Self::LatentRowsDense=>"resident_launch_latent_rows_dense",Self::LatentRowsAttention=>"resident_launch_latent_rows_attention",
        Self::MxRows2=>"resident_launch_mx_rows_small2",
        Self::MxRows4=>"resident_launch_mx_rows_small4",
        Self::MxRows8=>"resident_launch_mx_rows_small8",
        Self::MxRowsTile=>"resident_launch_mx_rows_tile",
        Self::MxRows8w=>"resident_launch_mx_rows_small8w",
        Self::MixedRows=>"resident_launch_flash_resident_projection_bf16_rows",
        Self::FoldRows=>"resident_launch_flash_resident_projection_fold_rows",
        Self::FoldRowsF32=>"resident_launch_flash_resident_projection_fold_rows_f32",
        Self::Fp8RowsSmall=>"resident_launch_fp8_rows_small",
        Self::Fp8RowsMulti=>"resident_launch_fp8_rows_multi",
        Self::Fp8RowsFew=>"resident_launch_fp8_rows_few",
        Self::Fp8RowsShare=>"resident_launch_fp8_rows_share",
        Self::PlainRowsSmall=>"resident_launch_plain_rows_small",
        Self::PromptRows=>"resident_launch_prompt_rows",
        Self::RowPrepare=>"resident_launch_prompt_prepare",
        Self::RowGatedNorm=>"resident_launch_prompt_gated_norm",
        Self::RowRecurrent=>"resident_launch_recurrent_rows",
        Self::RowRoute=>"resident_launch_prompt_route",
        Self::RowFold=>"resident_launch_prompt_fold",
        Self::RowPoolScores=>"resident_launch_prompt_pool_scores",
        Self::RowPlace=>"resident_launch_prompt_place",
        Self::RowPermute=>"resident_launch_prompt_permute",
        Self::RowSelect=>"resident_launch_prompt_select_rows",
        Self::TransposeSmem=>"resident_launch_transpose_smem",
        Self::DeviceRouteRows=>"resident_launch_device_route_rows",
        Self::AbsorbDense=>"resident_launch_absorb_dense",Self::AbsorbAttention=>"resident_launch_absorb_attention",
        Self::Causal=>"resident_launch_causal",Self::Expert=>"resident_launch_expert",Self::Plain=>"resident_launch_plain",Self::Mx=>"resident_launch_mx",
        Self::Round=>"resident_launch_round",Self::Copy=>"resident_launch_copy",Self::Pointwise=>"resident_launch_pointwise",Self::Put=>"resident_launch_put",Self::Take=>"resident_launch_take",
        Self::Prepare=>"resident_launch_prepare",Self::Collapse=>"resident_launch_collapse",Self::Combine=>"resident_launch_combine",
        Self::RmsPartials=>"resident_launch_rms_partials",Self::RmsApply=>"resident_launch_rms_apply",Self::NormTail=>"resident_launch_norm_tail",Self::Elementwise=>"resident_launch_elementwise",
        Self::CopyWords=>"resident_launch_copy_words",Self::ZeroWords=>"resident_launch_zero_words",Self::Mixed=>"resident_launch_mixed",
        Self::MixedBf16=>"resident_launch_mixed_bf16",Self::PlainBatch=>"resident_launch_plain_batch",
        Self::NvStream=>"resident_launch_nv_stream",Self::CausalFast=>"resident_launch_causal_fast",Self::CausalChain=>"resident_launch_causal_chain",Self::CausalGroup=>"resident_launch_causal_group",Self::NvTile=>"resident_launch_nv_tile",Self::Fp8Tile=>"resident_launch_fp8_tile",
        Self::DeviceRoute=>"resident_launch_device_route",Self::TakeScatter=>"resident_launch_take_scatter",Self::PutWide=>"resident_launch_put_wide",Self::PlainBf16=>"resident_launch_plain_bf16",
        Self::MhcPhaseRms=>"resident_launch_mhc_phase_rms",Self::MhcPhaseDots=>"resident_launch_mhc_phase_dots",Self::MhcPhaseFinish=>"resident_launch_mhc_phase_finish",Self::MhcBoundaryRms=>"resident_launch_mhc_boundary_rms",Self::MhcBoundaryDots=>"resident_launch_mhc_boundary_dots",Self::MhcBoundaryFinish=>"resident_launch_mhc_boundary_finish",Self::MhcBoundaryFused=>"resident_launch_mhc_boundary_fused",
        Self::NvQuarters=>"resident_launch_nv_quarters",Self::QuarterFold=>"resident_launch_quarter_fold",Self::ReduceTake=>"resident_launch_reduce_take",
        Self::PutJoin=>"resident_launch_put_join",Self::PutPlain=>"resident_launch_put_plain",Self::ShadowPrefetch=>"resident_launch_shadow_prefetch",Self::ShadowSide=>"resident_launch_shadow_side",
        Self::DeviceRouteTcRows=>"resident_launch_device_route_tc_rows",Self::Fp4CohortT8=>"resident_launch_fp4_cohort_t8",
        Self::BlockArms=>"resident_launch_block_arms",Self::BlockRows=>"resident_launch_block_rows",Self::PlainPacked=>"resident_launch_plain_packed",Self::PlainPackedGroup=>"resident_launch_plain_packed_group",Self::PlainPackedPicked=>"resident_launch_plain_packed_picked", // INKLING
        Self::PlainPackedRows=>"resident_launch_plain_packed_rows",Self::EmbedRows=>"resident_launch_embed_rows", // INKLING
        Self::PlainStaged=>"resident_launch_plain_staged",Self::PlainStagedRows=>"resident_launch_plain_staged_rows",Self::PlainStagedNarrow=>"resident_launch_plain_staged_narrow",Self::PlainStagedFew=>"resident_launch_plain_staged_few",Self::PlainStagedOne=>"resident_launch_plain_staged_one",Self::BlockValues=>"resident_launch_block_values",Self::BlockScores=>"resident_launch_block_scores",Self::BlockSoftmax=>"resident_launch_block_softmax", // INKLING
        Self::PlainTcRows=>"resident_launch_plain_tc_rows",Self::MoeGroupRoute=>"resident_launch_moe_group_route",Self::FlashWindowAttn=>"resident_launch_flash_window",Self::FlashWindowSplit16=>"resident_launch_flash_window_split16",Self::FlashWindowSplit32=>"resident_launch_flash_window_split32",Self::FlashWindowReduce=>"resident_launch_flash_window_reduce",Self::PlainPackedTcRows=>"resident_launch_plain_packed_tc_rows",Self::PrefillPackedTc=>"resident_launch_prefill_packed_tc",Self::PrefillFlashWindow=>"resident_launch_prefill_flash_window",Self::PrefillGateFuse=>"resident_launch_prefill_gate_fuse", // INKLING
        Self::FastMixBlend=>"resident_launch_fast_mix_blend",Self::FastRowDots=>"resident_launch_fast_row_dots",Self::FastNorm=>"resident_launch_fast_norm",Self::FastCappedGate=>"resident_launch_fast_capped_gate",Self::FastRouterPick=>"resident_launch_fast_router_pick",Self::FastStreamOps=>"resident_launch_fast_stream_ops", // INKLING
        Self::DraftLanes=>"inkling_draft_lanes",Self::PlainTcBig=>"resident_launch_plain_tc_big",Self::PlainTcPanel64=>"resident_launch_plain_tc_panel64",Self::DraftPanel=>"inkling_draft_panel",Self::DraftHeadFold=>"inkling_draft_head_fold", // INKLING
    }}
}
#[repr(C,align(16))]
#[derive(Clone,Copy)]
pub struct Command {pub kind:u32,pub reserved:[u32;3],pub payload:[u8;1024]}
#[cfg(not(target_arch="nvptx64"))]
impl Command {
    pub fn new(kind:Kind)->Self {Self {kind:kind as u32,reserved:[0;3],payload:[0;1024]}}
    pub fn put(&mut self,offset:usize,bytes:&[u8]) {self.payload[offset..offset+bytes.len()].copy_from_slice(bytes);}
    pub fn word(&mut self,index:usize,value:u64) {self.put(index*8,&value.to_ne_bytes());}
}
const _:()={assert!(core::mem::size_of::<Command>()==1040);assert!(core::mem::offset_of!(Command,payload)==16);};
#[cfg(test)] mod tests {
    use super::*;
    /// A kind number is a table index and a kernel entry: two kinds must never share one, and the count table must hold all.
    #[test] fn kind_numbers_are_unique_and_the_count_table_holds_every_kind() {
        let mut numbers=Kind::ALL.map(|kind|kind as usize).to_vec();numbers.sort_unstable();
        assert!(numbers.windows(2).all(|pair|pair[0]!=pair[1]),"two kinds share a number");
        assert_eq!(Kind::COUNT,numbers[numbers.len()-1]+1);assert_eq!(KindCounts::default().len(),Kind::COUNT);
        let mut names=Kind::ALL.map(Kind::kernel).to_vec();names.sort_unstable();assert!(names.windows(2).all(|pair|pair[0]!=pair[1]),"two kinds share a kernel entry");
    }
    #[test] fn command_words_keep_raw_bits_and_initialized_padding() {
        let mut c=Command::new(Kind::CopyWords);c.word(0,0x80000000_00000001);c.word(1,u64::MAX);
        assert_eq!(&c.payload[..8],&0x80000000_00000001u64.to_ne_bytes());
        assert_eq!(&c.payload[8..16],&u64::MAX.to_ne_bytes());assert_eq!(c.reserved,[0;3]);
        assert!(c.payload[16..].iter().all(|&v|v==0));
    }
    #[test] fn every_original_struct_fits_the_aligned_payload() {
        use core::mem::{size_of,align_of};
        macro_rules! fits {($($t:ty),*)=>{$(assert!(size_of::<$t>()<=1024);assert!(align_of::<$t>()<=16);)*}}
        fits!(crate::causal_ops::Arguments,crate::expert_ops::Arguments,crate::expert_ops::LaunchArguments,crate::next_params::MxLayer,crate::next_params::ResidentProjectionBatch,
            crate::mhc_boundary::LaneCast,crate::kernel_params::HybridCopy,crate::mhc_params::Prepare,
            crate::mhc_params::Collapse,crate::mhc_params::Combine,crate::mhc_params::Boundary,crate::mhc_params::ShadowPrefetch,crate::kernel_params::ModelRms,crate::kernel_params::ModelElementwise);
    }
}

/// Successfully submitted resident replay nodes, not graph-capture construction.
/// Bytes are logical destination bytes; copy traffic also reads that many bytes.
#[cfg(not(target_arch="nvptx64"))]
#[derive(Clone,Copy,Default,Debug,PartialEq,Eq)]
pub struct CopyCounts {pub launches:[u64;3],pub bytes:[u64;3],pub merged_zero_launches:u64,pub merged_zero_bytes:u64,pub stage_full_flushes:u64,pub piece_evictions:u64,pub captures_new_key:u64,pub captures_seen_key:u64}
#[cfg(not(target_arch="nvptx64"))]
impl CopyCounts {
    pub fn since(self,before:Self)->Self {Self {launches:std::array::from_fn(|i|self.launches[i].saturating_sub(before.launches[i])),
        bytes:std::array::from_fn(|i|self.bytes[i].saturating_sub(before.bytes[i])),merged_zero_launches:self.merged_zero_launches.saturating_sub(before.merged_zero_launches),merged_zero_bytes:self.merged_zero_bytes.saturating_sub(before.merged_zero_bytes),stage_full_flushes:self.stage_full_flushes.saturating_sub(before.stage_full_flushes),piece_evictions:self.piece_evictions.saturating_sub(before.piece_evictions),captures_new_key:self.captures_new_key.saturating_sub(before.captures_new_key),captures_seen_key:self.captures_seen_key.saturating_sub(before.captures_seen_key)}}
    pub fn line(self)->String {format!("resident_launch_copy={} resident_launch_copy_bytes={} resident_launch_copy_words={} resident_launch_copy_words_bytes={} resident_launch_zero_words={} resident_launch_zero_words_bytes={} resident_merged_zero_launches={} resident_merged_zero_bytes={} stage_full_flushes={} piece_evictions={} captures_new_key={} captures_seen_key={}",
        self.launches[0],self.bytes[0],self.launches[1],self.bytes[1],self.launches[2],self.bytes[2],self.merged_zero_launches,self.merged_zero_bytes,self.stage_full_flushes,self.piece_evictions,self.captures_new_key,self.captures_seen_key)}
}

/// The recorder's table area. Zero rows are the first bytes, and every flush's tables follow them.
#[cfg(not(target_arch="nvptx64"))]
pub mod stage {
    pub const BYTES:usize=1<<20;pub const ZERO_ROWS:usize=256;pub const BASE:usize=ZERO_ROWS*24;
    /// A table that does not fit this flush's area forces a flush. This depends only on the tables
    /// of this flush, never on earlier flushes, so the same commands always cut at the same place.
    pub fn full(staged:usize,bytes:usize)->bool {BASE+staged+bytes>BYTES}
    #[cfg(test)] mod tests {
        use super::*;
        /// One recorded command: its grid id and its table bytes.
        type Call=(u32,usize);
        /// The recorder's cutting, over repeated steps. `next` is the area test. Returns each step's piece keys.
        fn keys(steps:usize,step:&[Call],waits:&[usize],mut next:impl FnMut(usize,usize)->bool,mut flushed:impl FnMut(usize))->Vec<Vec<Vec<u32>>> {
            let mut out=Vec::new();let (mut piece,mut staged)=(Vec::new(),0usize);
            for _ in 0..steps {
                let mut pieces=Vec::new();
                for (at,&(grid,bytes)) in step.iter().enumerate() {
                    if next(staged,bytes) && !piece.is_empty() {pieces.push(std::mem::take(&mut piece));flushed(staged);staged=0;}
                    piece.push(grid);staged+=bytes.div_ceil(16)*16;
                    if waits.contains(&at) {pieces.push(std::mem::take(&mut piece));flushed(staged);staged=0;}
                }
                out.push(pieces);
            }out
        }
        fn new_keys_after_warmup(stream:&[Vec<Vec<u32>>])->usize {
            let mut seen=std::collections::BTreeSet::new();let mut late=0;
            for (step,pieces) in stream.iter().enumerate() {for piece in pieces {if seen.insert(piece.clone()) && step>0 {late+=1;}}}late
        }
        /// A step shaped like a plain decode: 11 latent rounds, each a wait, with real gather families.
        /// Copies has more table bytes in each gather than the join it replaces.
        fn step(table:usize)->(Vec<Call>,Vec<usize>) {
            let mut calls=Vec::new();let mut waits=Vec::new();
            for round in 0..11u32 {for job in 0..20u32 {calls.push((round*100+job,table));}waits.push(calls.len()-1);}
            (calls,waits)
        }
        /// The rule the head and 15cf49b97 used: the offset carries over and wraps.
        fn carried(calls:&[Call],waits:&[usize],steps:usize)->Vec<Vec<Vec<u32>>> {
            let base=std::cell::Cell::new(BASE);
            keys(steps,calls,waits,|staged,bytes|{
                let full=base.get()+staged+bytes>BYTES;
                if full && base.get()+bytes>BYTES {base.set(BASE);}
                full
            },|staged|base.set(base.get()+staged))
        }
        #[test] fn the_key_stream_over_63_steps_has_no_new_key_after_the_first_step() {
            for table in [96,240,1024,4096] {
                let (calls,waits)=step(table);
                let stream=keys(63,&calls,&waits,full,|_|{});
                assert_eq!(new_keys_after_warmup(&stream),0,"table {table}");
                assert!(stream.windows(2).all(|p|p[0]==p[1]),"table {table}: a step cut differently");
            }
        }
        /// The control: the carried offset makes new keys, and more of them with bigger tables.
        /// This is the captures finding: same tokens, a new cut once the offset reaches the end.
        #[test] fn the_carried_offset_makes_history_dependent_keys_and_grows_with_table_bytes() {
            let new=|table:usize|{let (calls,waits)=step(table);new_keys_after_warmup(&carried(&calls,&waits,63))};
            assert!(new(1024)>0,"the control must fail");assert!(new(4096)>new(1024));
            // Same step, later in the stream: the cut moves.
            let (calls,waits)=step(4096);let stream=carried(&calls,&waits,63);assert!(stream.windows(2).any(|p|p[0]!=p[1]));
        }
        #[test] fn only_a_table_too_big_for_this_flush_forces_a_cut() {
            assert!(!full(0,BYTES-BASE));assert!(full(0,BYTES-BASE+1));assert!(!full(BYTES-BASE-16,16));assert!(full(BYTES-BASE-16,17));
        }
    }
}

/// Whether a captured piece's key was captured before. Print only: nothing reads this to decide.
#[cfg(not(target_arch="nvptx64"))]
pub mod capture_keys {
    use std::collections::BTreeSet;
    /// At most this many hashes, about 2 MiB. A long service must not grow without limit for a print.
    pub const MOST:usize=1<<16;
    #[derive(Default)]
    pub struct Seen {hashes:BTreeSet<u64>}
    /// A fixed-seed hash of the grid words (FNV-1a), the same in every process and on both ranks.
    pub fn hash(grids:impl Iterator<Item=[u32;6]>)->u64 {
        let mut h=0xcbf29ce484222325u64;
        for grid in grids {for word in grid {for byte in word.to_le_bytes() {h^=u64::from(byte);h=h.wrapping_mul(0x100000001b3);}}}h
    }
    impl Seen {
        /// True when this key is new. Past MOST, an unseen key reports new and is not kept, so a
        /// later repeat of it also reports new: captures_new_key can read high after MOST keys.
        pub fn note(&mut self,key:u64)->bool {
            if self.hashes.contains(&key) {return false;}
            if self.hashes.len()<MOST {self.hashes.insert(key);}true
        }
        pub fn len(&self)->usize {self.hashes.len()}
        pub fn is_empty(&self)->bool {self.hashes.is_empty()}
    }
    #[cfg(test)] mod tests {
        use super::*;
        use std::collections::VecDeque;
        /// The recorder's capture path in small: a limited table, oldest out, one count per capture.
        struct Table {live:VecDeque<u64>,limit:usize,seen:Seen,new:u64,again:u64,captures:u64,evictions:u64}
        impl Table {
            fn new(limit:usize)->Self {Self {live:VecDeque::new(),limit,seen:Seen::default(),new:0,again:0,captures:0,evictions:0}}
            fn replay(&mut self,key:u64) {
                if let Some(at)=self.live.iter().position(|k|*k==key) {let k=self.live.remove(at).unwrap();self.live.push_back(k);return;}
                if self.live.len()>=self.limit {self.live.pop_front();self.evictions+=1;}
                self.live.push_back(key);self.captures+=1;
                if self.seen.note(key) {self.new+=1;} else {self.again+=1;}
            }
        }
        #[test] fn the_counts_sum_to_the_captures_and_separate_new_keys_from_evicted_ones() {
            // A cycle larger than the table: the first pass is all new, later passes are all seen again.
            let mut t=Table::new(4);for _ in 0..3 {for key in 0..6 {t.replay(key);}}
            assert_eq!((t.captures,t.new,t.again),(18,6,12));assert_eq!(t.new+t.again,t.captures);
            // Keys that never repeat: all new, and the evictions alone could not tell these cases apart.
            let mut u=Table::new(4);for key in 0..18 {u.replay(key);}
            assert_eq!((u.captures,u.new,u.again),(18,18,0));assert_eq!(u.new+u.again,u.captures);assert_eq!(u.evictions,t.evictions);
            // Everything fits: only the first pass captures.
            let mut v=Table::new(8);for _ in 0..3 {for key in 0..6 {v.replay(key);}}assert_eq!((v.captures,v.new,v.again),(6,6,0));
        }
        #[test] fn the_set_stops_growing_at_its_bound_and_the_counts_still_sum() {
            let mut t=Table::new(4);for key in 0..(MOST as u64+10) {t.replay(key);}
            assert_eq!(t.seen.len(),MOST);assert_eq!(t.new+t.again,t.captures);assert_eq!(t.again,0);
        }
        #[test] fn the_hash_is_fixed_and_sees_every_field_and_the_order() {
            let a=[[1,2,3,4,5,6],[7,8,9,10,11,12]];
            assert_eq!(hash(a.into_iter()),hash(a.into_iter()));
            assert_ne!(hash(a.into_iter()),hash(a.into_iter().rev()));
            for field in 0..6 {let mut b=a;b[1][field]+=1;assert_ne!(hash(a.into_iter()),hash(b.into_iter()));}
            assert_ne!(hash(a.into_iter()),hash(a[..1].iter().copied()));
        }
    }
}
