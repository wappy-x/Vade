/// Register decode body and launch geometry must be selected together.
// Each CTA owns eight rows and eight output columns. Up to two disjoint
// row slabs use this entry; the admission bound is separate from CTA height.
pub const REGISTER_DECODE_TILE: [u32; 4] = [8, 8, 128, 1024];
pub const REGISTER_DECODE_ROWS: u32 = 16;
#[inline(always)]
pub const fn register_decode(input_kind: u32, rows: u32, outputs: u32) -> bool {
    input_kind == 0 && rows <= REGISTER_DECODE_ROWS && outputs <= 512
}

/// Canonical 128-thread groups in one dense projection CTA.
pub const DENSE_TILE_GROUPS: u32 = 2;

/// Launch geometry only; every tile retains the canonical K128 dot products.
#[derive(Clone, Copy)]
pub struct RoutedTile { pub rows: usize, pub cols: usize, pub planes: usize, pub threads: u32, pub shared: u32 }
impl RoutedTile {
    pub const fn new(tokens: u32) -> Self {
        // Half-full M128 tiles also cover the large partial prompt chunk.
        if tokens >= 2048 { Self { rows: 128, cols: 128, planes: 1, threads: 512, shared: 66560 } }
        else if tokens >= 256 { Self { rows: 16, cols: 128, planes: 8, threads: 128, shared: 36992 } }
        else if tokens > 16 && tokens <= 32 { Self { rows: 32, cols: 64, planes: 4, threads: 128, shared: 24832 } }
        else { Self { rows: 16, cols: 64, planes: if tokens < 128 { tokens.div_ceil(16) as usize } else { 8 }, threads: 128, shared: 20608 } }
    }
    pub fn blocks(self, width: usize, paired: bool) -> u32 {
        (width / if paired { self.cols / 2 } else { self.cols } * self.planes) as u32
    }
}

/// M32 prompt tiles need four planes per M128 task; M16 decode needs eight.
#[inline(always)]
pub fn grouped_planes(tokens: u32, prefill: bool) -> u32 { tokens.div_ceil(16).min(if prefill { 4 } else { 8 }) }

/// One borrowed row-major expert projection; scales are BF16 multipliers, widened in registers.
/// Three records (gate, up, down) per expert; coordinates are the selected original K128 blocks.
/// No implicit dimensions or duplicated resident weights. Loader validates all ranges before launch.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertProjection {
    pub weights: u64,
    pub scales: u64,
    pub rows: u32,
    pub cols: u32,
    pub scale_cols: u32,
    pub first_k_block: u32,
    pub first_n_block: u32,
    pub reserved: u32,
}
impl ExpertProjection {
    /// Local bytes/grids are contiguous; original block origins survive selection.
    pub fn checked(weights: u64, scales: u64, rows: u64, cols: u64, first_n: u64, first_k: u64) -> Option<Self> {
        if weights == 0 || scales == 0 || rows == 0 || cols == 0 || rows % 128 != 0 || cols % 128 != 0 { return None; }
        weights.checked_add(rows.checked_mul(cols)?)?;
        scales.checked_add((rows / 128).checked_mul(cols / 128)?.checked_mul(2)?)?;
        u32::try_from(first_n.checked_add(rows / 128)?).ok()?;
        u32::try_from(first_k.checked_add(cols / 128)?).ok()?;
        Some(Self { weights, scales, rows: u32::try_from(rows).ok()?, cols: u32::try_from(cols).ok()?,
            scale_cols: u32::try_from(cols / 128).ok()?, first_k_block: u32::try_from(first_k).ok()?,
            first_n_block: u32::try_from(first_n).ok()?, reserved: 0 })
    }
}

/// Integer layout only: no change to the projection/quantizer arithmetic.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RoutingLayout { pub selected: u32, pub output_stride: u32, pub intermediate: u32, pub packed_local: u32 }
impl RoutingLayout {
    pub const FLASH: Self = Self { selected: 10, output_stride: 11, intermediate: 640, packed_local: 1 };
    pub fn checked(selected: u32, intermediate: u32) -> Option<Self> {
        if selected == 0 || selected > 32 || intermediate == 0 || intermediate % 128 != 0 || intermediate > u32::MAX / 2 { return None; }
        Some(Self { selected, output_stride: selected + 1, intermediate, packed_local: 0 })
    }
    #[inline(always)]
    pub fn code_index(self, start: usize, row: usize, width: usize, column: usize) -> usize {
        start * self.intermediate as usize + row * if self.packed_local != 0 { width } else { self.intermediate as usize } + column
    }
    #[inline(always)]
    pub fn scale_index(self, start: usize, row: usize, width: usize, block: usize) -> usize {
        start * (self.intermediate as usize / 128) + row * if self.packed_local != 0 { width / 128 } else { self.intermediate as usize / 128 } + block
    }
    #[inline(always)]
    pub fn output_index(self, token: usize, slot: usize) -> usize { token * self.output_stride as usize + slot }
    /// Ascending expert rank, with input lane breaking ties; shared stays last.
    #[inline(always)]
    pub fn ordered_slot(self, slot: usize, mut expert_at: impl FnMut(usize) -> u32) -> usize {
        if slot == self.selected as usize { return slot; }
        let expert = expert_at(slot);
        let mut rank = 0;
        for lane in 0..self.selected as usize {
            let other = expert_at(lane);
            rank += usize::from(other < expert || (other == expert && lane < slot));
        }
        rank
    }
    /// # Safety
    /// Pair/status spans are valid, immutable and row-validated before histogramming.
    #[inline(always)]
    pub unsafe fn stable_row(self, route: GroupedRouteParams, pair: usize) -> u32 {
        let pairs = route.pairs as *const f32;
        let mut row = 0;
        for previous in 0..pair { row += (!route.failed(previous / self.selected as usize) && *pairs.add(previous * 3 + 1) == *pairs.add(pair * 3 + 1)) as u32; }
        row
    }
}
/// Same grouped scratch ABI, with three lane01 ExpertProjection records per expert.
/// Generic codes/scales have fixed intermediate row pitch, including zero padding.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RoutedParams { pub grouped: GroupedParams, pub layout: RoutingLayout }
impl RoutedParams {
    pub fn checked(grouped: GroupedParams, layout: RoutingLayout) -> Option<Self> {
        let r = grouped.route;
        let pairs = r.tokens.checked_mul(layout.selected)?;
        let rows = pairs.checked_add(15u32.checked_mul(pairs.min(r.experts))?)?;
        (r.row_capacity as u64).checked_mul(layout.intermediate as u64)?.checked_mul(8)?;
        (r.row_capacity as u64).checked_mul(grouped.cols as u64)?.checked_mul(4)?;
        (r.tokens as u64).checked_mul(layout.output_stride as u64)?.checked_mul(grouped.cols as u64)?.checked_mul(4)?;
        if r.tokens == 0 || r.tokens > 1 << 24 || r.experts > 1 << 24 || layout.selected > r.experts || pairs != r.pair_count || rows > r.row_capacity
            || pairs.div_ceil(128) + pairs.min(r.experts) > 65535
            || RoutingLayout::checked(layout.selected, layout.intermediate).is_none() || layout.output_stride != layout.selected + 1
            || layout.packed_local != 0 || grouped.cols == 0 || grouped.cols % 128 != 0 || grouped.activation_quantization != 1 { return None; }
        Some(Self { grouped, layout })
    }
}

#[cfg(test)]
mod descriptor_tests {
    use super::ExpertProjection as P;
    #[test] fn bounds() { assert_eq!(core::mem::size_of::<P>(),40); for (r,c) in [(512,2048),(1024,3072),(3072,512)] { let p=P::checked(128,256,r,c,2,3).unwrap(); assert_eq!((p.scale_cols,p.first_n_block,p.first_k_block),(c as u32/128,2,3)); } for (w,s,r,c,n,k) in [(0,1,128,128,0,0),(1,1,127,128,0,0),(u64::MAX,1,128,128,0,0),(1,u64::MAX,128,128,0,0),(1,1,u64::MAX,128,0,0),(1,1,128,128,u64::MAX,0),(1,1,128,128,0,u64::MAX)] { assert!(P::checked(w,s,r,c,n,k).is_none()); } }
    #[test] fn routing_layout_bytes() {
        for (tokens, planes, blocks) in [(1,1,8),(8,1,8),(16,1,8),(17,4,32),(31,4,32),(32,4,32),(33,3,24),(48,3,24),(49,4,32),(64,4,32),(65,5,40),(112,7,56),(113,8,64),(127,8,64),(128,8,64),(255,8,64),(256,8,64),(2047,8,64),(2048,1,8),(3840,1,8),(4095,1,8),(4096,1,8),(8192,1,8)] {
            let tile = super::RoutedTile::new(tokens);
            assert_eq!(tile.planes, planes);
            assert_eq!(tile.blocks(512, tile.cols == 128), blocks);
            assert_eq!(tile.shared as usize, 2 * (tile.rows * 128 + tile.cols * 128 + tile.rows * 4));
            assert_eq!(super::grouped_planes(tokens, true), tokens.div_ceil(16).min(4));
        }
        let f = super::RoutingLayout::FLASH;
        for width in [0,128,256,384,640] { let mut old=vec![0u8;640*32]; let mut new=old.clone();
            for row in 0..16 { for col in 0..width { let value=(row*31+col) as u8; old[16*640+row*width+col]=value; new[f.code_index(16,row,width,col)]=value; } for block in 0..width/128 { assert_eq!(f.scale_index(16,row,width,block),16*5+row*(width/128)+block); } }
            assert_eq!(old,new); for token in 0..257 { for slot in 0..11 { assert_eq!(f.output_index(token,slot),token*11+slot); } }
        }
        for width in [512,1024] { let l=super::RoutingLayout::checked(8,width).unwrap(); assert_eq!((l.selected,l.output_stride,l.packed_local),(8,9,0)); assert_eq!(l.code_index(16,3,256,128),19*width as usize+128); }
        for (top,width) in [(0,512),(33,512),(8,0),(8,513),(8,u32::MAX)] { assert!(super::RoutingLayout::checked(top,width).is_none()); }
    }
    #[test] fn removed_routed_planes_return_before_writing() {
        for tokens in 1..=16 {
            let tile = super::RoutedTile::new(tokens);
            assert_eq!((tile.rows, tile.planes), (16, 1));
            // Distinct top-k IDs give m <= tokens; the sole M128 task starts at 0.
            for m in 0..=tokens as usize {
                for plane in tile.planes..8 {
                    let mb = plane * tile.rows;
                    // routed_project returns on mb >= m before project_tile can write.
                    assert!(mb >= m, "tokens={tokens} m={m} removed plane={plane}");
                }
            }
        }
    }
    #[test] fn finish_bounds() {
        use super::{FinishLayout as L, MoeFinishParams as F, OrderedFinishParams as O};
        assert_eq!((core::mem::size_of::<F>(),core::mem::size_of::<O>()),(64,96));
        for h in [2048,3072] { let l=L::checked(super::RoutingLayout::checked(8,512).unwrap(),h).unwrap(); let p=F {error:4,ids:4,weights:4,partials:4,shared_gate:4,output:4,activation_quantization:1,column_start:0,columns:h}; assert!(O::checked(p,l,257,4).is_some()); for t in [0,65536] { assert!(O::checked(p,l,t,0).is_none()); } for a in [0,1,u64::MAX-3] { assert!(O::checked(F {output:a,..p},l,1,0).is_none()); } assert!(O::checked(F {columns:h+1,..p},l,1,0).is_none()); assert!(O::checked(F {activation_quantization:0,..p},l,1,0).is_none()); }
        assert!(L::checked(super::RoutingLayout::FLASH,2560).is_none());
    }
}

/// Resident FP8 projection, shared by the host and NVPTX entry points.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DecodeParams {
    pub weights: u64,
    pub scales: u64,
    pub input: u64,
    pub output: u64,
    pub tokens: u32,
    pub rows: u32,
    pub cols: u32,
    pub round_output: u32,
    pub weight_fragments: u32,
    pub packed_input: u32,
}

/// W8A8 block projection. Input kind: quantized row-major (0), FP32 (1), packed MMA fragments (2).
/// Every row uses the same 128-K order.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockParams {
    pub weights: u64,
    pub weight_scales: u64,
    pub input: u64,
    pub input_scales: u64,
    pub output: u64,
    pub rows: u32,
    pub outputs: u32,
    pub cols: u32,
    pub input_kind: u32,
    pub round_output: u32,
}

/// Reusable dynamic activation quantization, with an explicit BF16 boundary.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantizeParams {
    pub input: u64,
    pub output: u64,
    pub scales: u64,
    pub rows: u32,
    pub cols: u32,
    pub round_input: u32,
    pub packed_output: u32,
}

/// Flash routing metadata shared by histogram, prefix, ordering and gather.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GroupedRouteParams {
    /// Device status array, one sticky word per token; failed rows have no expert tasks.
    pub error: u64,
    pub pairs: u64,
    pub counts: u64,
    pub starts: u64,
    pub pair_to_row: u64,
    pub token_pairs: u64,
    pub token_count: u64,
    pub tasks: u64,
    pub sorted_row: u64,
    /// Expert-major row -> token * 11 + ascending expert slot.
    pub output_row: u64,
    pub experts: u32,
    pub tokens: u32,
    pub pair_count: u32,
    pub row_capacity: u32,
}

/// Borrowed expert descriptors and activations; no resident weight copy.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GroupedParams {
    pub route: GroupedRouteParams,
    pub weights: u64,
    pub input: u64,
    pub input_scales: u64,
    pub output: u64,
    pub cols: u32,
    pub activation_quantization: u32,
    /// 1: TP2 ownership; otherwise world in high 16 bits, owned columns in low 16.
    pub output_shard: u32,
}

/// Block size of fp8_grouped_single's multi-token branch: one thread per expert and
/// per routed pair, so it serves at most this many experts and pairs.
pub const SMALL_ROUTE_THREADS: u32 = 512;
/// fp8_grouped_gate_accurate mode that also writes the gather's E4M3 middle (decode).
pub const FUSED_GATE: u32 = 2;
impl GroupedParams {
    /// The gate reads neither pair_to_row, token_pairs nor output_row, so in FUSED_GATE
    /// mode they carry the middle codes, their scales and the arrival counters (zero
    /// between launches, `fused_gate_counters` words).
    pub fn fused_gate(self, codes: u64, scales: u64, counters: u64) -> Self {
        Self { route: GroupedRouteParams { pair_to_row: codes, token_pairs: scales, output_row: counters, ..self.route },
            activation_quantization: FUSED_GATE, ..self }
    }
}
/// Channels of one projection per fused CTA; a K128 group spans 128 / this many CTAs.
pub const FUSED_GATE_CHANNELS: usize = 16;
/// Sixteen-row planes per task (grouped_planes' decode bound).
pub const FUSED_GATE_PLANES: usize = 8;
/// Counter words of the fused gate for a task capacity and intermediate width.
pub const fn fused_gate_counters(tasks: usize, intermediate: usize) -> usize { tasks * FUSED_GATE_PLANES * (intermediate / 128) }

/// Ten local E4M3/scale records (at most three K128 blocks), then the peer's
/// 1280 shared-expert output columns. No activation is requantized on the wire.
pub const MIDDLE_PACKET_WORDS: usize = 10 * (384 / 4 + 3) + 1280;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MiddleParams {
    pub route: GroupedRouteParams,
    pub weights: u64,
    pub local_codes: u64,
    pub local_scales: u64,
    pub send: u64,
    pub receive: u64,
    pub codes: u64,
    pub scales: u64,
    pub partials: u64,
    pub rank: u32,
}

/// General worlds gather rank-major packets; TP2 retains its original ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GeneralMiddleParams { pub middle: MiddleParams, pub world: u32, pub output: u64 }

pub fn middle_packet_words(world: usize) -> usize {
    if world == 2 { MIDDLE_PACKET_WORDS } else { 10 * 33 * 5usize.div_ceil(world) + 2560 }
}
pub fn middle_columns(rank: usize, world: usize) -> core::ops::Range<usize> {
    let start = rank * (20 / world) + rank.min(20 % world);
    start * 128..(start + 20 / world + usize::from(rank < 20 % world)) * 128
}
/// Same rotating contiguous five-block ownership as RankPlan::expert_blocks.
pub fn middle_expert_end(expert: usize, rank: usize, world: usize) -> usize {
    let first = expert % world;
    let extra = 5 % world;
    let before = extra.min(world - first);
    let wrapped = extra - before;
    let next = rank + 1;
    next * (5 / world) + next.min(first + before) - next.min(first) + next.min(wrapped)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MoeFinishParams {
    pub error: u64,
    pub ids: u64,
    pub weights: u64,
    pub partials: u64,
    pub shared_gate: u64,
    pub output: u64,
    pub activation_quantization: u32,
    pub column_start: u32,
    pub columns: u32,
}

/// Rank-reduced partial rows are ascending expert ID, then the shared expert.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FinishLayout { pub selected: u32, pub hidden: u32, pub output_stride: u32 }
impl FinishLayout {
    pub const FLASH: Self = Self { selected: 10, hidden: 2560, output_stride: 11 };
    pub fn checked(route: RoutingLayout, hidden: u32) -> Option<Self> {
        if RoutingLayout::checked(route.selected, route.intermediate).is_none() || route.packed_local != 0
            || route.output_stride != route.selected + 1 || hidden == 0 || hidden % 128 != 0 { return None; }
        Some(Self { selected: route.selected, hidden, output_stride: route.output_stride })
    }
    /// Down and weight -> BF16 product -> FP32 ID fold -> BF16 routed sum.
    /// The legacy Flash descriptor alone also supports its historical unrounded path.
    #[inline(always)]
    pub fn routed_sum(self, quantized: bool, mut partial: impl FnMut(usize) -> f32,
        mut weight: impl FnMut(usize) -> f32, rounded: impl Fn(f32) -> f32) -> f32 {
        let mut sum = 0.0;
        for row in 0..self.selected as usize {
            let (value, weight) = (partial(row), weight(row));
            sum += if quantized { rounded(rounded(value) * rounded(weight)) } else { value * weight };
        }
        if quantized { rounded(sum) } else { sum }
    }
}
/// Fixed rounded finish; residual=0 delegates the single residual boundary to the caller.
/// Otherwise add the one FP32 residual stream after shared-last, then round to BF16.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OrderedFinishParams { pub finish: MoeFinishParams, pub layout: FinishLayout, pub tokens: u32, pub residual: u64, pub shared: u64 }
impl OrderedFinishParams {
    pub fn checked(finish: MoeFinishParams, layout: FinishLayout, tokens: u32, residual: u64) -> Option<Self> {
        if layout.selected == 0 || layout.selected > 32 || layout.output_stride != layout.selected + 1
            || layout.hidden == 0 || layout.hidden % 128 != 0 || tokens == 0 || tokens > 65535
            || finish.activation_quantization != 1 || finish.columns == 0
            || finish.column_start.checked_add(finish.columns)? > layout.hidden { return None; }
        let t = tokens as u64;
        let partials = t.checked_mul(layout.output_stride as u64)?.checked_mul(layout.hidden as u64)?;
        for (address, elements) in [(finish.error,t),(finish.ids,t*layout.selected as u64),
            (finish.weights,t*layout.selected as u64),(finish.partials,partials),(finish.shared_gate,t),
            (finish.output,t*finish.columns as u64)] {
            if address == 0 || address % 4 != 0 { return None; }
            address.checked_add(elements.checked_mul(4)?)?;
        }
        if residual != 0 { if residual % 4 != 0 { return None; } residual.checked_add(t.checked_mul(layout.hidden as u64)?.checked_mul(4)?)?; }
        Some(Self { finish, layout, tokens, residual, shared: 0 })
    }
}

/// Validate numeric route IDs before any float-to-index conversion.
#[inline(always)]
pub fn valid_route(token: f32, expert: f32, weight: f32, tokens: u32, experts: u32) -> bool {
    token >= 0.0 && token < tokens as f32 && token == token as u32 as f32
        && expert >= 0.0 && expert < experts as f32 && expert == expert as u32 as f32
        && weight.is_finite()
}
impl GroupedRouteParams {
    #[inline(always)]
    pub unsafe fn failed(&self, token: usize) -> bool { unsafe { *(self.error as *const u32).add(token) != 0 } }
}
