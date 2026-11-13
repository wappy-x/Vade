//! Logical FP32 mHC ABI. All counts come from operation data, not model names.
//! Buffers are row-major. Distinct fields must own disjoint device ranges.
//! Each caller must call `Shape::validate` before allocation or launch.
//! Kernel entries do not validate shapes or buffer extents.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub rows: u32, pub streams: u32, pub hidden: u32, pub iterations: u32,
    pub rms_eps: f32, pub eps: f32,
}
impl Shape {
    /// Bound work and address products before any allocation or launch.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0 || self.streams == 0 || self.hidden == 0 || self.iterations == 0
            || self.iterations > 1_048_576 { return Err("invalid mHC dimensions or iteration count"); }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 || !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("invalid mHC epsilon");
        }
        let n = u64::from(self.streams); let h = u64::from(self.hidden); let t = u64::from(self.rows);
        let w = n.checked_mul(h).ok_or("mHC width overflow")?;
        let p = n.checked_mul(n + 2).ok_or("mHC coefficient count overflow")?;
        for count in [w.checked_mul(p), t.checked_mul(w), t.checked_mul(p)] {
            if count.is_none_or(|v| v > u32::MAX as u64) { return Err("mHC buffer exceeds ABI limit"); }
        }
        Ok(())
    }
    pub fn wide(self) -> usize { self.streams as usize * self.hidden as usize }
    pub fn coefficients(self) -> usize { self.streams as usize * (self.streams as usize + 2) }
    pub fn prepare_threads(self) -> u32 {(self.coefficients().min(32)*32) as u32}
    /// Shapes that may use the split RMS and dots launches. Shape only. Byte-equal to the one-block project on the cases of
    /// `split_cases` (`kernel-bench --mhc-sweep` runs exactly those, and a host test enumerates the admitted set against them).
    /// Any other valid shape keeps the one-block project.
    pub fn split_project(self) -> bool { split_admitted(self.rows, self.streams, self.hidden) }
}
pub const SPLIT_MAX_ROWS: u32 = 8;
pub const SPLIT_MAX_STREAMS: u32 = 32;
/// Hidden widths swept at up to 8 streams: every fold size class (groups of 16 words, odd and even group counts from 1 to 64, tails).
pub const SPLIT_SMALL: [u32; 25] = [1, 2, 3, 5, 15, 16, 17, 19, 31, 32, 33, 47, 48, 49, 80, 100, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025];
/// Hidden widths swept at 9 to 32 streams.
pub const SPLIT_MID: [u32; 8] = [1, 3, 16, 17, 19, 127, 129, 1025];
/// Stream counts and widths swept around the model's 4,096 (all rows at four streams, rows 1, 6 and 8 elsewhere).
pub const SPLIT_WIDE_STREAMS: [u32; 11] = [1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32];
pub const SPLIT_WIDE: [u32; 3] = [4095, 4096, 4097];
/// True when (rows, streams, hidden) is one of the swept cases.
pub fn split_admitted(rows: u32, streams: u32, hidden: u32) -> bool {
    (1..=SPLIT_MAX_ROWS).contains(&rows) && (1..=SPLIT_MAX_STREAMS).contains(&streams) && (
        (streams <= 8 && SPLIT_SMALL.contains(&hidden)) || (streams > 8 && SPLIT_MID.contains(&hidden))
        || (SPLIT_WIDE_STREAMS.contains(&streams) && SPLIT_WIDE.contains(&hidden) && (streams == 4 || [1, 6, 8].contains(&rows))))
}
/// The sweep's cases: (streams, hidden, rows list). The device sweep runs every row of every list.
#[cfg(not(target_arch = "nvptx64"))]
pub fn split_cases() -> Vec<(u32, u32, Vec<u32>)> {
    let mut cases = Vec::new();
    for n in 1..=SPLIT_MAX_STREAMS { for &h in if n <= 8 { &SPLIT_SMALL[..] } else { &SPLIT_MID[..] } { cases.push((n, h, (1..=SPLIT_MAX_ROWS).collect())); } }
    for &n in &SPLIT_WIDE_STREAMS { for &h in &SPLIT_WIDE { cases.push((n, h, if n == 4 { (1..=SPLIT_MAX_ROWS).collect() } else { vec![1, 6, 8] })); } }
    cases
}
#[cfg(test)]
mod split_tests {
    use super::*;
    fn shape(rows: u32, streams: u32, hidden: u32) -> Shape { Shape { rows, streams, hidden, iterations: 20, rms_eps: 1e-6, eps: 1e-6 } }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        let mut swept = std::collections::HashSet::new();
        for (n, h, rows) in split_cases() { for r in rows { swept.insert((r, n, h)); } }
        let mut admitted = std::collections::HashSet::new();
        for rows in 0..=10u32 { for streams in 0..=34u32 { for hidden in 0..=4100u32 {
            if shape(rows, streams, hidden).split_project() { admitted.insert((rows, streams, hidden)); assert!(swept.contains(&(rows, streams, hidden)), "{rows}x{streams}x{hidden} is admitted but not swept"); }
        } } }
        assert_eq!(admitted, swept, "the sweep list holds a shape the predicate rejects");
        // The model's shape is admitted at every row count, and the odd fold group counts of 3 and more are in the list (streams 1, hidden 48).
        assert!((1..=8).all(|r| shape(r, 4, 4096).split_project()) && shape(1, 1, 48).split_project() && shape(1, 1, 80).split_project());
    }
    #[test]
    fn row_boundary_is_eight_and_nine() {
        assert!(shape(1, 4, 4096).split_project() && shape(8, 4, 4096).split_project());
        assert!(!shape(9, 4, 4096).split_project() && !shape(0, 4, 4096).split_project());
    }
    #[test]
    fn the_review_counterexample_is_inside_the_swept_family() {
        let s = shape(6, 7, 19); assert!(s.validate().is_ok() && s.split_project());
    }
    #[test]
    fn unproven_streams_and_widths_take_the_one_block_project() {
        assert!(shape(1, 32, 4097).split_project() && !shape(1, 33, 4096).split_project() && !shape(1, 0, 4096).split_project());
        assert!(!shape(1, 4, 4098).split_project() && !shape(1, 1, 1_048_576).split_project() && !shape(1, 4, 0).split_project());
        assert!(shape(1, 1, 1).split_project() && shape(8, 32, 1).split_project());
        // Between the swept widths, or rows the wide cases skip, the one-block project runs.
        assert!(!shape(1, 4, 4000).split_project() && !shape(1, 12, 100).split_project() && !shape(2, 5, 4096).split_project());
    }
}
/// Input `[T,N,D]`, weights `[P,N*D]`, base `[P]`, scale `[3]`.
/// Scratch normalized `[T,N*D]`; coefficients `[T,P]` = pre, post, `comb[src,dst]`.
/// Collapsed `[T,D]` contains BF16 values in FP32 lanes.
/// One block owns each token. Block barriers order normalization, complete
/// coefficient dots, and activation with Sinkhorn before output consumption.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Prepare {
    pub input: u64, pub weights: u64, pub base: u64, pub scale: u64,
    pub normalized: u64, pub coefficients: u64, pub collapsed: u64, pub shape: Shape,
}
/// Two ordered launches share these disjoint FP32 planes. With previous_input
/// zero, input is `[T,N*D]`. Otherwise input is the previous block `[T,D]` and
/// previous_input/previous_coefficients supply its deferred post operation.
/// Each post result and collapse keeps its original BF16 boundary. The norm
/// retains 128-feature partials and the original warp butterfly fold.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Boundary {
    pub input:u64,pub previous_input:u64,pub previous_coefficients:u64,
    pub residual:u64,pub weights:u64,pub base:u64,pub scale:u64,
    pub mixing_normalized:u64,pub coefficients:u64,pub collapsed:u64,
    pub norm_weight:u64,pub norm_partials:u64,pub output:u64,
    pub shape:Shape,pub norm_epsilon:f32,pub norm_bf16:u32,
    /// CHAIN-SHADOW: a static weight range that blocks past the arithmetic grid of
    /// `mhc_boundary_dots` and `mhc_boundary_finish` pull toward L2. Zero address or
    /// zero bytes means none. The arithmetic blocks never read these fields.
    pub prefetch:u64,pub prefetch_bytes:u64,
    /// 0: `prefetch.global.L2`; 1: `prefetch.global.L2::evict_last`. Other values trap.
    pub prefetch_policy:u32,pub prefetch_reserved:u32,
}
/// D1 (one-launch boundary, `mhc_boundary_fused`): the split chain's planes and arguments, plus the
/// strip-major weight copy and a scratch plane. Block k of the grid owns strip k of SUM-ORDER v1.
/// `strips`: weight of row r at feature 256 p + 8 k + e sits at `fused_strip_index(shape, k, p, r, e)`
/// (0 past the last feature); BF16 words when `bf16` is 1, FP32 words when 0. `scratch`:
/// `FUSED_SCRATCH_WORDS` words; words 0 and 1 are meeting counters that are 0 between launches.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct FusedBoundary {pub boundary:Boundary,pub strips:u64,pub scratch:u64,pub bf16:u32,pub reserved:u32}
pub const FUSED_BLOCKS:usize=32;
pub const FUSED_THREADS:usize=256;
/// Shared bytes for one strip's BF16 weights (periods x rows x 16 bytes); in the tail they hold the collapsed plane.
pub const FUSED_STAGE_BYTES:usize=25_600;
/// Words of one strip's x and y planes (8 a period).
pub const FUSED_STRIP_WORDS:usize=576;
/// Scratch: 2 counters, 32 RMS strip sums, then 32 strip sums for each of up to 32 coefficient rows.
pub const FUSED_SUMS:usize=2;
/// Then, 8-byte aligned, the phase stamps of the last launch (%globaltimer ns): FUSED_STAMP_SLOTS a block, then
/// one word for the end of the norm. Slots: 0 start, 1 residual, 2 RMS chain, 3 meeting 1 passed, 4 y and weights
/// ready, 5 dot chains done; the last block also 6 rows joined, 7 Sinkhorn done. A slot a block does not reach is 0.
pub const FUSED_STAMPS_AT:usize=(FUSED_SUMS+32+32*32).div_ceil(2)*2;
pub const FUSED_STAMP_SLOTS:usize=8;
pub const FUSED_SCRATCH_WORDS:usize=FUSED_STAMPS_AT+2*(32*FUSED_STAMP_SLOTS+1);
/// One-row shapes the fused launch takes: the 4 x 4 Sinkhorn of one warp (streams <= 4, so <= 24 rows),
/// a strip's x and y in FUSED_STRIP_WORDS, the collapsed plane in the stage area.
pub fn fused_admitted(s:Shape)->bool {
    s.validate().is_ok() && s.rows==1 && s.streams<=4 && s.wide().div_ceil(256)*8<=FUSED_STRIP_WORDS && s.hidden as usize*4<=FUSED_STAGE_BYTES
}
/// True when the BF16 strip copy of this shape fits the stage area.
pub fn fused_stage_fits(s:Shape)->bool {s.wide().div_ceil(256)*s.coefficients()*16<=FUSED_STAGE_BYTES}
/// Word index of the strip-major copy (see FusedBoundary).
pub const fn fused_strip_index(periods:usize,count:usize,strip:usize,period:usize,row:usize,element:usize)->usize {((strip*periods+period)*count+row)*8+element}
/// The most rows of one fused boundary (the chats of a batched decode step, the rows of a drafting check pass):
/// one `mhc_boundary_fused` launch a row, so a row has the bits of its own one-row launch.
#[cfg(not(target_arch="nvptx64"))]
pub const FUSED_ROWS_MAX:usize=8;
/// The arguments of row `row` of a fused boundary of several rows. `whole` names every plane at its row 0 with the
/// one-row shape. Every plane a row owns is row-major (input, previous residual and coefficients, residual, mixing,
/// coefficients, collapsed, norm partials, output), so the row's launch reads and writes only its own words of them.
/// The parameters (base, scale, norm weight), the strip copy and the scratch plane are the same for every row;
/// the rows' launches run one after another on the ordered stream, and the scratch counters rest at 0 between them.
/// Host only: the kernel crate is built without it, so the kernel's code does not change.
#[cfg(not(target_arch="nvptx64"))]
pub fn fused_row(whole:FusedBoundary,row:usize)->FusedBoundary {
    let p=whole.boundary;let shape=p.shape;let previous=p.previous_input!=0;
    let at=|words:usize|4*(row*words) as u64;
    let (wide,count,hidden)=(shape.wide(),shape.coefficients(),shape.hidden as usize);
    let boundary=Boundary {input:p.input+at(if previous {hidden} else {wide}),
        previous_input:if previous {p.previous_input+at(wide)} else {0},previous_coefficients:if previous {p.previous_coefficients+at(count)} else {0},
        residual:p.residual+at(wide),mixing_normalized:p.mixing_normalized+at(wide),coefficients:p.coefficients+at(count),
        collapsed:p.collapsed+at(hidden),norm_partials:p.norm_partials+at(hidden.div_ceil(128)),output:p.output+at(hidden),..p};
    FusedBoundary {boundary,..whole}
}
#[cfg(test)]
mod fused_row_tests {
    use super::*;
    /// Planes laid out one after another with a gap; every plane holds `rows` rows.
    fn whole(shape:Shape,rows:usize,previous:bool)->(FusedBoundary,Vec<(u64,usize)>) {
        let (wide,count,hidden)=(shape.wide(),shape.coefficients(),shape.hidden as usize);let mut next=4096u64;
        let mut plane=|words:usize|{let at=next;next+=4*(rows*words) as u64+256;at};
        let input=plane(if previous {hidden} else {wide});let (last,last_coefficients)=if previous {(plane(wide),plane(count))} else {(0,0)};
        let (residual,mixing,coefficients,collapsed,partials,output)=(plane(wide),plane(wide),plane(count),plane(hidden),plane(hidden.div_ceil(128)),plane(hidden));
        let boundary=Boundary {input,previous_input:last,previous_coefficients:last_coefficients,residual,weights:16,base:32,scale:48,mixing_normalized:mixing,coefficients,collapsed,
            norm_weight:64,norm_partials:partials,output,shape,norm_epsilon:1e-6,norm_bf16:1,prefetch:0,prefetch_bytes:0,prefetch_policy:0,prefetch_reserved:0};
        let widths=vec![(input,if previous {hidden} else {wide}),(last,wide),(last_coefficients,count),(residual,wide),(mixing,wide),(coefficients,count),(collapsed,hidden),(partials,hidden.div_ceil(128)),(output,hidden)];
        (FusedBoundary {boundary,strips:80,scratch:96,bf16:1,reserved:0},widths)
    }
    fn planes(f:FusedBoundary)->[u64;9] {let p=f.boundary;[p.input,p.previous_input,p.previous_coefficients,p.residual,p.mixing_normalized,p.coefficients,p.collapsed,p.norm_partials,p.output]}
    #[test]
    fn a_row_owns_its_own_words_of_every_row_major_plane() {
        for (streams,hidden) in [(4u32,4096u32),(4,4095),(4,4097),(1,1),(2,17),(3,129),(4,1025)] {for rows in 1..=FUSED_ROWS_MAX {for previous in [false,true] {
            let shape=Shape {rows:1,streams,hidden,iterations:20,rms_eps:1e-6,eps:1e-6};let (f,widths)=whole(shape,rows,previous);
            assert_eq!(planes(fused_row(f,0)),planes(f),"row 0 is the one-row launch");
            for row in 0..rows {
                let r=fused_row(f,row);let p=r.boundary;
                // The shared arguments and the one-row shape do not move.
                assert_eq!((p.weights,p.base,p.scale,p.norm_weight,r.strips,r.scratch,r.bf16,p.shape.rows),(16,32,48,64,80,96,1,1));
                assert_eq!((p.shape.streams,p.shape.hidden,p.norm_bf16,p.norm_epsilon.to_bits()),(streams,hidden,1,1e-6f32.to_bits()));
                for (got,&(base,words)) in planes(r).into_iter().zip(&widths) {
                    if base==0 {assert_eq!(got,0,"an absent previous plane stays absent");continue;}
                    // The row's words start right after the rows before it and end inside the plane of `rows` rows.
                    assert_eq!(got,base+4*(row*words) as u64);assert!(got+4*words as u64<=base+4*(rows*words) as u64);
                }
            }
        }}}
    }
    #[test]
    fn a_row_keeps_the_16_byte_alignment_the_tail_loads_need() {
        // The tail's fast path (hidden a multiple of 4) loads residual quads at 16-byte addresses: a row's stride is a multiple of 16 bytes there.
        for (streams,hidden) in [(4u32,4096u32),(1,4),(3,1024),(2,128)] {
            let shape=Shape {rows:1,streams,hidden,iterations:20,rms_eps:1e-6,eps:1e-6};let (f,_)=whole(shape,FUSED_ROWS_MAX,true);
            for row in 0..FUSED_ROWS_MAX {assert_eq!((fused_row(f,row).boundary.residual-f.boundary.residual)%16,0);}
        }
    }
}
/// The largest CHAIN-SHADOW grid tail. A tail block only issues prefetches.
pub const SHADOW_MAX_BLOCKS: u32 = 64;
#[repr(C)]
#[derive(Clone,Copy)]
pub struct ShadowPrefetch {pub prefetch:u64,pub prefetch_bytes:u64,pub prefetch_policy:u32,pub reserved:u32}
/// Decode CHAIN-SHADOW: bytes of the next stage's weights a boundary pulls toward L2, and the
/// dots-launch tail blocks that issue them. Screen de9aa38ef on one Spark (BENCH-1): 8 MiB with 2
/// tail blocks saved 18.7 us a boundary; 24 tail blocks slowed the chain and saved 3.0 us.
pub const SHADOW_DECODE_BYTES: usize = 16 << 20;
pub const SHADOW_DECODE_BLOCKS: u32 = 2;
/// Decode CHAIN-SHADOW per launch (rms, dots, finish): a tail block lives as long as the DRAM time of
/// what it prefetches (GB10: ~190 GB/s for 2 tail blocks), so each launch takes only what fits inside
/// its own chain (head nsys, c15: rms 42.7 us, dots 40.3 us, finish 22.5 us). The spans walk the next
/// stage's static projections in plan order. Zero: no tail for that launch. The rms chain slows by
/// ~4 us per prefetched MB (its producers' loads queue behind the prefetches), more than a MB saves.
pub const SHADOW_DECODE_STAGE_BYTES: [usize; 3] = [0, 7 << 20, 4 << 20];
/// The largest add count the probe binder accepts (about 0.12 s of adds at 1.8 ns).
pub const PROBE_MAX_ADDS: u32 = 1 << 26;
/// The probe's loop counts: complete groups of 16 adds, then the single adds left. The kernel
/// counts groups up to `adds / 16`, so no counter can pass `adds` or wrap for any `u32`.
pub const fn probe_chunks(adds: u32) -> (u32, u32) { (adds / 16, adds % 16) }
/// Host replay of the `chain_add_latency` probe: `adds` serial FP32 adds of `step` from `start`.
pub fn serial_add_bits(start: f32, step: f32, adds: u32) -> u32 { let mut value = start; for _ in 0..adds { value += step; } value.to_bits() }
/// One prefetch per 128-byte line, at `prefetch + SHADOW_LINE*line`.
pub const SHADOW_LINE: u64 = 128;
/// Lines a tail prefetches for `bytes`. Line `lines-1` starts below `prefetch+bytes`.
pub const fn shadow_lines(bytes: u64) -> u64 { bytes.div_ceil(SHADOW_LINE) }
/// Inputs and outputs of the `chain_add_latency` probe: one thread folds `adds`
/// dependent FP32 adds `acc += step` from `start`, the same instruction the
/// boundary chains use. `output` receives the result bits, SM cycles and
/// global-timer nanoseconds around the chain (three u64 words).
#[repr(C)]
#[derive(Clone,Copy)]
pub struct AddLatency {pub output:u64,pub start:f32,pub step:f32,pub adds:u32,pub reserved:u32}
/// Input `[T,N,D]`, coefficients `[T,P]`, output `[T,D]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Collapse {
    pub input: u64, pub coefficients: u64, pub output: u64,
    pub rows: u32, pub streams: u32, pub hidden: u32,
}
/// Raw input `[T,N,D]`, coefficients `[T,P]`, block `[T,D]`, output `[T,N,D]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Combine {
    pub input: u64, pub coefficients: u64, pub block: u64, pub output: u64,
    pub rows: u32, pub streams: u32, pub hidden: u32,
}
/// Raw input `[T,N,D]`, output `[T,D]`. Final RMS norm is a separate operation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Head { pub input: u64, pub output: u64, pub rows: u32, pub streams: u32, pub hidden: u32 }
