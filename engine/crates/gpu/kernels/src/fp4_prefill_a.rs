//! Shared A operand staging. Same compact row and RN BF16 boundary as the gather.
use super::*;

// A 144-byte row gives conflict-free scalar fragment loads across the warp.
// Gate has 256 threads (one row per eight threads); down has 128 (two rows).
#[inline(always)]
pub(super) unsafe fn prefetch<const DOWN: bool>(
    p: &GroupedParams,
    start: usize,
    mb: usize,
    m: usize,
    k: usize,
    kt: usize,
) -> [[u32; 4]; 2] {
    let tid = _thread_idx_x() as usize;
    let mut pairs = [[0; 4]; 2];
    two!(H, {
        if DOWN || H == 0 {
            let local = tid / 8 + H * 16;
            if let Some(row) =
                crate::compact::input_row(p.route.sorted_row, start, mb, local, m, DOWN)
            {
                four!(J, {
                    let col = kt * 16 + tid % 8 * 2 + J * 16;
                    pairs[H][J] = if DOWN {
                        *(p.input as *const u32).add(row * k / 2 + col / 2)
                    } else {
                        let x = (p.input as *const f32).add(row * k + col);
                        crate::gemm::bf16_pair(*x, *x.add(1))
                    };
                });
            }
        }
    });
    pairs
}

#[inline(always)]
pub(super) unsafe fn stage<const DOWN: bool>(shared: u32, pairs: [[u32; 4]; 2]) {
    let tid = _thread_idx_x() as usize;
    two!(H, {
        if DOWN || H == 0 {
            four!(J, {
                let at = (tid / 8 + H * 16) * 144 + (tid % 8 * 2 + J * 16) * 2;
                ptx::store(shared + at as u32, f32::from_bits(pairs[H][J]));
            });
        }
    });
}

#[inline(always)]
pub(super) unsafe fn fragment(shared: u32, i: usize, lane: usize, z: usize) -> [u32; 4] {
    let at = shared + ((i * 16 + lane / 4) * 144 + (z * 16 + lane % 4 * 2) * 2) as u32;
    [
        ptx::load(at).to_bits(),
        ptx::load(at + 8 * 144).to_bits(),
        ptx::load(at + 16).to_bits(),
        ptx::load(at + 8 * 144 + 16).to_bits(),
    ]
}

/// Gate A gather split from its rounding: raw F32 pairs for one K64 step, so the
/// global loads can be issued before the MMAs and rounded/stored after them.
#[inline(always)]
pub(super) unsafe fn gather_gate(p: &GroupedParams, row: Option<usize>, k: usize, kt: usize) -> [[f32; 2]; 4] {
    let tid = _thread_idx_x() as usize;
    let mut v = [[0.0f32; 2]; 4];
    if let Some(row) = row {
        four!(J, {
            let col = kt * 16 + tid % 8 * 2 + J * 16;
            let x = (p.input as *const f32).add(row * k + col);
            v[J] = [*x, *x.add(1)];
        });
    }
    v
}

/// This thread's token row for the gate gather (constant over the K loop).
#[inline(always)]
pub(super) unsafe fn gate_row(p: &GroupedParams, start: usize, mb: usize, m: usize) -> Option<usize> {
    let tid = _thread_idx_x() as usize;
    crate::compact::input_row(p.route.sorted_row, start, mb, tid / 8, m, false)
}

/// Round the gathered F32 pairs exactly as prefetch does (RN BF16 pair; rows past
/// the expert stay zero) and store them into the padded A tile.
#[inline(always)]
pub(super) unsafe fn stage_gate(shared: u32, v: [[f32; 2]; 4]) {
    let mut pairs = [[0u32; 4]; 2];
    four!(J, {
        pairs[0][J] = crate::gemm::bf16_pair(v[J][0], v[J][1]);
    });
    stage::<false>(shared, pairs);
}

/// Vectorized gate gather: this thread's token rows tid/16 and tid/16 + 16 of the
/// M32 plane. A half-warp owns one row's 64 K values of a step: 16-byte loads and
/// conflict-free 8-byte stores (the 144-byte row pitch is unchanged).
#[inline(always)]
pub(super) unsafe fn gate_rows(p: &GroupedParams, start: usize, mb: usize, m: usize) -> [Option<usize>; 2] {
    let tid = _thread_idx_x() as usize;
    [
        crate::compact::input_row(p.route.sorted_row, start, mb, tid / 16, m, false),
        crate::compact::input_row(p.route.sorted_row, start, mb, tid / 16 + 16, m, false),
    ]
}

/// K values step*64 + 4*(tid%16) .. +3 of both rows (rows past the expert stay
/// zero). The input rows are 10,240 bytes and the arena is 64-byte aligned.
#[inline(always)]
pub(super) unsafe fn gather_gate4(p: &GroupedParams, rows: [Option<usize>; 2], k: usize, step: usize) -> [[f32; 4]; 2] {
    let tid = _thread_idx_x() as usize;
    let mut v = [[0.0f32; 4]; 2];
    two!(H, {
        if let Some(row) = rows[H] {
            let x = (p.input as *const f32).add(row * k + step * 64 + tid % 16 * 4);
            let (a, b, c, d): (f32, f32, f32, f32);
            core::arch::asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) x);
            v[H] = [a, b, c, d];
        }
    });
    v
}

/// Same RN BF16 pairs as stage_gate, written as one 8-byte store per row.
#[inline(always)]
pub(super) unsafe fn stage_gate4(shared: u32, v: [[f32; 4]; 2]) {
    let tid = _thread_idx_x() as usize;
    two!(H, {
        let at = shared + ((tid / 16 + H * 16) * 144 + tid % 16 * 8) as u32;
        let lo = crate::gemm::bf16_pair(v[H][0], v[H][1]);
        let hi = crate::gemm::bf16_pair(v[H][2], v[H][3]);
        core::arch::asm!("st.shared.v2.b32 [{p}], {{{a}, {b}}};", p = in(reg32) at, a = in(reg32) lo, b = in(reg32) hi);
    });
}

/// M64 gate plane: token rows tid/16 + 16h (h = 0..3); the same half-warp-per-row
/// rule as gate_rows, so each row's 64 K values are one 256-byte load group.
#[inline(always)]
pub(super) unsafe fn gate_rows64(p: &GroupedParams, start: usize, mb: usize, m: usize) -> [Option<usize>; 4] {
    let tid = _thread_idx_x() as usize;
    let mut rows = [None; 4];
    four!(H, {
        rows[H] = crate::compact::input_row(p.route.sorted_row, start, mb, tid / 16 + H * 16, m, false);
    });
    rows
}

#[inline(always)]
unsafe fn load_f32x4(x: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    core::arch::asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) x);
    [a, b, c, d]
}

/// K values step*64 + 4*(tid%16) .. +3 of the four rows (absent rows stay zero).
#[inline(always)]
pub(super) unsafe fn gather_gate64(p: &GroupedParams, rows: [Option<usize>; 4], k: usize, step: usize) -> [[f32; 4]; 4] {
    let tid = _thread_idx_x() as usize;
    let mut v = [[0.0f32; 4]; 4];
    four!(H, {
        if let Some(row) = rows[H] {
            v[H] = load_f32x4((p.input as *const f32).add(row * k + step * 64 + tid % 16 * 4));
        }
    });
    v
}

/// L2 prefetch of the two 128-byte lines per row that gather_gate64 reads at
/// `step` (threads 0 and 8 of each row's 16; hint only, no value or order effect).
#[inline(always)]
pub(super) unsafe fn prefetch_gate64(p: &GroupedParams, rows: [Option<usize>; 4], k: usize, step: usize) {
    let tid = _thread_idx_x() as usize;
    if tid % 8 == 0 {
        four!(H, {
            if let Some(row) = rows[H] {
                super::prefetch_l2((p.input as *const f32).add(row * k + step * 64 + tid % 16 * 4) as *const u8);
            }
        });
    }
}

/// Same RN BF16 pairs as stage_gate4 for the four rows of the M64 tile.
#[inline(always)]
pub(super) unsafe fn stage_gate64(shared: u32, v: [[f32; 4]; 4]) {
    let tid = _thread_idx_x() as usize;
    four!(H, {
        let at = shared + ((tid / 16 + H * 16) * 144 + tid % 16 * 8) as u32;
        let lo = crate::gemm::bf16_pair(v[H][0], v[H][1]);
        let hi = crate::gemm::bf16_pair(v[H][2], v[H][3]);
        core::arch::asm!("st.shared.v2.b32 [{p}], {{{a}, {b}}};", p = in(reg32) at, a = in(reg32) lo, b = in(reg32) hi);
    });
}
