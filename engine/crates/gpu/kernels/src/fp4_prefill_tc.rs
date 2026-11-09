//! W4A16 prompt expert GEMMs on mma.m16n8k16 BF16 with NVFP4 weights dequantized in
//! registers. Gate: one CTA per M128 task and N128 tile (eight warps of M128 x N16 gate +
//! N16 up, IL live M16 tiles). Down: one CTA per M64 plane and ten N128 tiles (eight
//! warps of M64 x N16). Every output keeps its ascending K16 MMA chain.
use super::*;

macro_rules! four {
    ($j:ident,$body:block) => {{
        const $j: usize = 0;
        $body
    }
    {
        const $j: usize = 1;
        $body
    }
    {
        const $j: usize = 2;
        $body
    }
    {
        const $j: usize = 3;
        $body
    }};
}
macro_rules! eight {
    ($j:ident,$body:block) => {{
        four!($j, $body);
        {
            const $j: usize = 4;
            $body
        }
        {
            const $j: usize = 5;
            $body
        }
        {
            const $j: usize = 6;
            $body
        }
        {
            const $j: usize = 7;
            $body
        }
    }};
}
macro_rules! two {
    ($i:ident,$body:block) => {{
        const $i: usize = 0;
        $body
    }
    {
        const $i: usize = 1;
        $body
    }};
}

#[path = "fp4_prefill_a.rs"]
mod activation;

/// B fragment of lane t for one K16 block: RN_BF16((q*s)*g) at k = 2t,2t+1 (b[0])
/// and 2t+8,2t+9 (b[1]), from byte t of the shuffled low/high words.
/// Each E2M1 nibble placed as F16 bits s<<15 | (e1 e0 m)<<9 is q*2^-14 exactly
/// (0, 0.5 and -0 included). fma.f32.f16 then forms (q*2^-14)*s exactly (<= 6
/// significant bits), and multiplying by g*2^14 (an exact power-of-two scaling of
/// the FP32 global) is the same real product as (q*s)*g, so the single FP32
/// rounding and the BF16 rounding are the existing ones bit for bit.
#[inline(always)]
unsafe fn dequant<const HIGH: bool>(lo: u32, hi: u32, select: u32, s: u32, global14: f32) -> [u32; 2] {
    // HIGH: the scale is the upper F16 of s (odd K16 block of a scale pair).
    let (b0, b1) = dequant_raw::<HIGH>(lo, hi, select, s, global14);
    [b0, b1]
}

#[inline(always)]
unsafe fn dequant_raw<const HIGH: bool>(lo: u32, hi: u32, select: u32, s: u32, global14: f32) -> (u32, u32) {
    let (b0, b1): (u32, u32);
    macro_rules! body { ($sv:literal) => { core::arch::asm!(
        "{{",
        ".reg .b32 w, x, y, od, ev;",
        ".reg .b16 o0, o1, e0, e1, s0, s1;",
        ".reg .f32 f0, f1, f2, f3;",
        "prmt.b32 w, {lo}, {hi}, {sel};",
        "shr.b32 x, w, 3;",
        "and.b32 x, x, 0x0E000E00;",
        "and.b32 y, w, 0x80008000;",
        "or.b32 od, x, y;",
        "shl.b32 x, w, 1;",
        "and.b32 x, x, 0x0E000E00;",
        "shl.b32 y, w, 4;",
        "and.b32 y, y, 0x80008000;",
        "or.b32 ev, x, y;",
        "mov.b32 {{o0, o1}}, od;",
        "mov.b32 {{e0, e1}}, ev;",
        "mov.b32 {{s0, s1}}, {s};",
        concat!("fma.rn.f32.f16 f0, e0, ", $sv, ", 0f80000000;"),
        concat!("fma.rn.f32.f16 f1, o0, ", $sv, ", 0f80000000;"),
        concat!("fma.rn.f32.f16 f2, e1, ", $sv, ", 0f80000000;"),
        concat!("fma.rn.f32.f16 f3, o1, ", $sv, ", 0f80000000;"),
        "mul.rn.f32 f0, f0, {g};",
        "mul.rn.f32 f1, f1, {g};",
        "mul.rn.f32 f2, f2, {g};",
        "mul.rn.f32 f3, f3, {g};",
        "cvt.rn.bf16x2.f32 {b0}, f1, f0;",
        "cvt.rn.bf16x2.f32 {b1}, f3, f2;",
        "}}",
        lo = in(reg32) lo, hi = in(reg32) hi, sel = in(reg32) select, s = in(reg32) s,
        g = in(reg32) global14, b0 = out(reg32) b0, b1 = out(reg32) b1,
        options(pure, nomem, nostack),
    ) } }
    if HIGH {
        body!("s1");
    } else {
        body!("s0");
    }
    (b0, b1)
}

/// Four E4M3 scales (K16 blocks 0..3 of one row) -> F16 pairs (blocks 0,1) and (2,3).
#[inline(always)]
unsafe fn scales_f16(word: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    core::arch::asm!(
        "{{ .reg .b16 l, h; mov.b32 {{l, h}}, {w}; cvt.rn.f16x2.e4m3x2 {a}, l; cvt.rn.f16x2.e4m3x2 {b}, h; }}",
        w = in(reg32) word, a = out(reg32) a, b = out(reg32) b, options(pure, nomem, nostack));
    [a, b]
}

/// One m16k16 A fragment from the padded tile; same registers as four LDS.32.
#[inline(always)]
unsafe fn ldsm4(p: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    core::arch::asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a}, {b}, {c}, {d}}}, [{p}];",
        a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg32) p);
    [a, b, c, d]
}

#[inline(always)]
unsafe fn wait_group<const N: usize>() {
    core::arch::asm!("cp.async.wait_group {n};", n = const N);
}

#[inline(always)]
unsafe fn load_b64(p: u32) -> (u32, u32) {
    let (a, b): (u32, u32);
    core::arch::asm!("ld.shared.v2.b32 {{{a}, {b}}}, [{p}];", a = out(reg32) a, b = out(reg32) b, p = in(reg32) p);
    (a, b)
}

#[inline(always)]
unsafe fn load_b32(p: u32) -> u32 {
    let a: u32;
    core::arch::asm!("ld.shared.b32 {a}, [{p}];", a = out(reg32) a, p = in(reg32) p);
    a
}

/// Gate ring stage: packed B (gate, up) x 128 rows x 32 B, then the scale words
/// (gate, up) x 128 rows x 4 B; GATE_STAGES stages follow the A stage.
const GATE_STAGE_B: usize = 2 * 128 * 32;
const GATE_STAGE_S: usize = 2 * 128 * 4;
const GATE_STAGES: usize = 3;

/// Gate CTA = one M128 task x one N128 tile (x = tile, x >= 5 exits), 256 threads.
/// Warp w owns gate and up columns nb + 16w .. +16 for all of the task's rows: M16
/// tiles I < IL (live rows / 16, up to 8), N8 tiles J = 0, 1 (gate) and 2, 3 (up). One dequantized B fragment
/// feeds IL MMAs and the activation epilogue stays in registers. Per-output K16
/// chains stay ascending over the full K; operands are the RN BF16 A rows and the
/// exactly dequantized weights; outputs cross to BF16 once, in the epilogue.
/// A stage: up to 128 rows (IL 8, the whole M128 task) x 128 bytes, 16-byte chunk
/// c of row r at chunk c ^ (r & 7) (ldmatrix rows of one 8x8 matrix hit distinct banks).
const G64_A: usize = 128 * 128;
const G64_A_BYTES: usize = 2 * G64_A;
const G64_SHARED: usize = G64_A_BYTES + GATE_STAGES * (GATE_STAGE_B + GATE_STAGE_S);
const _: () = assert!(G64_SHARED == crate::kernel_params::FP4_PREFILL_GATE_SHARED_BYTES as usize);

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gate64<const IL: usize>(
    p: &GroupedParams,
    expert: Expert,
    shared: u32,
    start: usize,
    mb: usize,
    m: usize,
    nb: usize,
    width: usize,
    g14: [f32; 2],
    select: u32,
) {
    const K: usize = 2560;
    const STEPS: usize = K / 64;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let p = *p;
    let b_base = shared + G64_A_BYTES as u32;
    let s_base = b_base + (GATE_STAGES * GATE_STAGE_B) as u32;
    // Producer: chunk c of row r for both matrices (gate rows at 0, up rows at
    // +4096) and one scale word per thread.
    let (r, c) = ((tid >> 1) & 127, tid & 1);
    let src_g = (expert.gate.weights as *const u8).add((nb + r) * (K / 2) + c * 16);
    let src_u = (expert.up.weights as *const u8).add((nb + r) * (K / 2) + c * 16);
    let dst_w = (r * 32 + 16 * (c ^ ((r >> 2) & 1))) as u32;
    let (ms, rs) = (tid >> 7, tid & 127);
    let scales = if ms == 0 { expert.gate.scales } else { expert.up.scales };
    let src_s = (scales as *const u8).add((nb + rs) * (K / 16));
    let dst_s = ((ms * 128 + rs) * 4) as u32;
    // Whole 128-byte lines of this thread's weight row (gate for c = 0, up for
    // c = 1) are requested from L2 when the ring copies their first 32-byte
    // sector (GATE_PF lines ahead), so DRAM sees line-sized reads per row instead
    // of one sector per K64 step.
    let pf_row = (if c == 0 { expert.gate.weights } else { expert.up.weights } as *const u8).add((nb + r) * (K / 2));
    let mut line = 0;
    while line < GATE_PF {
        prefetch_l2(pf_row.add(line * 128));
        line += 1;
    }
    prefetch_l2(src_s);
    prefetch_l2(src_s.add(128));
    macro_rules! issue {
        ($arg:expr, $slot:expr) => {{
            let i: usize = $arg;
        let st: u32 = $slot;
        let wb = b_base + st * GATE_STAGE_B as u32 + dst_w;
        if i % 4 == 0 && i / 4 + GATE_PF < K / 256 {
            prefetch_l2(pf_row.add((i / 4 + GATE_PF) * 128));
        }
        ptx::copy16(wb, src_g.add(i * 32) as *const u32);
        ptx::copy16(wb + 4096, src_u.add(i * 32) as *const u32);
        ptx::copy4(s_base + st * GATE_STAGE_S as u32 + dst_s, src_s.add(i * 4) as *const u32);
        }};
    }
    // A: the chunk's BF16 rows (fp4_prefill_tc_a16, the same RN pairs stage_gate64
    // forms), 128 bytes per row and K64 step, copied by cp.async into the A double
    // buffer one step ahead; thread tid moves 16 bytes of rows tid/8 and tid/8 + 32
    // (absent rows are zero-filled, as the register gather left them zero).
    let a16 = p.input as *const u8;
    // Local rows tid/8 + 32h (h < 4) below 16*IL: Some(Some(row)) copies, Some(None)
    // zero-fills a row past the expert, None is outside the tile.
    let mut a_rows = [None; 4];
    four!(H, {
        let local = tid / 8 + H * 32;
        if local < IL * 16 {
            a_rows[H] = Some(crate::compact::input_row(p.route.sorted_row, start, mb, local, m, false));
        }
    });
    let a_dst = ((tid / 8) * 128 + (((tid % 8) ^ ((tid / 8) & 7)) * 16)) as u32;
    macro_rules! a_issue {
        ($arg:expr, $off:expr) => {{
            let s: usize = $arg;
            let off: u32 = $off;
        four!(H, {
            let at = shared + off + (H * 32 * 128) as u32 + a_dst;
            match a_rows[H] {
                Some(Some(row)) => copy16_zfill(at, a16.add((row * K + s * 64) * 2 + (tid % 8) * 16), true),
                Some(None) => copy16_zfill(at, a16, false),
                None => {}
            }
        });
        }};
    }
    macro_rules! a_prefetch {
        ($arg:expr) => {{
            let s: usize = $arg;
        if tid % 8 == 0 {
            four!(H, {
                if let Some(Some(row)) = a_rows[H] {
                    prefetch_l2(a16.add((row * K + s * 64) * 2));
                }
            });
        }
        }};
    }
    // Groups: B0, A0, B1, then per step A(s+1), B(s+2): waiting for all but the
    // newest group at the top of step s leaves only B(s+1) in flight.
    issue!(0, 0);
    ptx::commit();
    a_issue!(0, 0);
    ptx::commit();
    if 1 < STEPS {
        issue!(1, 1);
    }
    ptx::commit();
    let mut t = 1;
    while t < GATE_A_PF && t < STEPS {
        a_prefetch!(t);
        t += 1;
    }
    // Rows 16w + 8(J&1) + lane/4 of each matrix; swizzle bit (row>>2)&1 = (lane>>4)&1.
    let own = warp * 16 + lane / 4;
    let sw = (lane >> 4) & 1;
    let b_lane = b_base + (own * 32) as u32;
    let zoff = [(16 * sw) as u32, (16 - 16 * sw) as u32];
    let s_lane = s_base + (own * 4) as u32;
    // ldmatrix row lane&15, logical chunk 2Z + lane/16, stored at chunk ^ (row & 7).
    let q = (lane >> 4) ^ (lane & 7);
    let a_lane = ((lane & 15) * 128) as u32;
    let a_z = [(q * 16) as u32, ((2 ^ q) * 16) as u32, ((4 ^ q) * 16) as u32, ((6 ^ q) * 16) as u32];
    let mut sums = [[ZERO; 4]; 8];
    let mut step = 0;
    // Running ring positions: B slot of this step and of the previous step (the slot that
    // B(step + GATE_STAGES - 1) refills), A buffer offset of this step and of the next.
    let (mut b_cur, mut b_prev) = (0u32, (GATE_STAGES - 1) as u32);
    let (mut a_cur, mut a_next) = (0u32, G64_A as u32);
    while step < STEPS {
        wait_group::<1>();
        ptx::sync();
        // Producer block of this step: A(step + 1) and B(step + 2) cp.async groups and
        // the A row L2 prefetch. It writes only ring slots the barrier above released
        // (read in step - 1), so where it sits inside the step changes issue timing only.
        macro_rules! produce {
            () => {{
                if step + 1 < STEPS {
                    a_issue!(step + 1, a_next);
                }
                ptx::commit();
                if step + GATE_STAGES - 1 < STEPS {
                    issue!(step + GATE_STAGES - 1, b_prev);
                }
                ptx::commit();
                if step + GATE_A_PF < STEPS {
                    a_prefetch!(step + GATE_A_PF);
                }
            }};
        }
        if GATE_PRODUCE_LO == 0 && GATE_PRODUCE_HI == 0 {
            produce!();
        } else if (warp < 4 && GATE_PRODUCE_LO == 0) || (warp >= 4 && GATE_PRODUCE_HI == 0) {
            produce!();
        }
        let st = b_cur;
        let current = shared + a_cur + a_lane;
        let bq = [b_lane + st * GATE_STAGE_B as u32 + zoff[0], b_lane + st * GATE_STAGE_B as u32 + zoff[1]];
        let sq = s_lane + st * GATE_STAGE_S as u32;
        let mut scale = [[0u32; 2]; 4];
        four!(J, {
            scale[J] = scales_f16(load_b32(sq + ((J & 1) * 32 + (J >> 1) * 512) as u32));
        });
        // B fragments of K16 sub-step 0 here; fragment J of sub-step Z + 1 is loaded and
        // dequantized right after M16 tile 2J + 1's MMAs of sub-step Z, so the dequant
        // fills the HMMA spacing instead of running as a block before the MMAs. Same
        // loads, dequant inputs and per-accumulator MMA order: placement only.
        let mut b = [[0u32; 2]; 4];
        four!(J, {
            let (lo, hi) = load_b64(bq[0] + ((J & 1) * 256 + (J >> 1) * 4096) as u32);
            b[J] = dequant::<false>(lo, hi, select, scale[J][0], g14[J >> 1]);
        });
        four!(Z, {
            let mut bn = [[0u32; 2]; 4];
            eight!(I, {
                if I < IL {
                    let a = ldsm4(current + a_z[Z] + (I * 16 * 128) as u32);
                    four!(J, {
                        sums[I][J] = madd(a, b[J], sums[I][J]);
                    });
                }
                if Z < 3 && I % 2 == 1 {
                    const ZN: usize = (Z + 1) & 3;
                    const JN: usize = (I / 2) & 3;
                    let (lo, hi) = load_b64(bq[ZN >> 1] + ((JN & 1) * 256 + (JN >> 1) * 4096 + 8 * (ZN & 1)) as u32);
                    bn[JN] = dequant::<{ ZN & 1 == 1 }>(lo, hi, select, scale[JN][ZN >> 1], g14[(JN >> 1) & 1]);
                }
            });
            carry::<Z>(&mut b, bn);
            if (GATE_PRODUCE_LO == Z + 1 && warp < 4) || (GATE_PRODUCE_HI == Z + 1 && warp >= 4) {
                produce!();
            }
        });
        b_prev = b_cur;
        b_cur = if b_cur + 1 == GATE_STAGES as u32 { 0 } else { b_cur + 1 };
        (a_cur, a_next) = (a_next, a_cur);
        step += 1;
    }
    let y = (p.output as *mut u32).add(start * width / 2);
    eight!(I, {
        if I < IL {
            two!(J, {
                let (g, u) = (sums[I][J], sums[I][J + 2]);
                let row = mb + I * 16 + lane / 4;
                let col = nb + warp * 16 + J * 8 + lane % 4 * 2;
                if row < m {
                    *y.add(row * width / 2 + col / 2) = next_activation_pair(g.a, g.b, u.a, u.b);
                }
                if row + 8 < m {
                    *y.add((row + 8) * width / 2 + col / 2) = next_activation_pair(g.c, g.d, u.c, u.d);
                }
            });
        }
    });
}

#[inline(always)]
unsafe fn run_gate64(p: GroupedParams) {
    let tasks = p.route.tasks as *const u32;
    let task = _block_idx_y() as usize;
    let bx = _block_idx_x() as usize;
    if task >= *tasks as usize {
        return;
    }
    let id = *tasks.add(1 + task * 2) as usize;
    let mb = *tasks.add(2 + task * 2) as usize;
    let m = *(p.route.counts as *const u32).add(id) as usize;
    if mb >= m {
        return;
    }
    let expert = *(p.weights as *const Expert).add(id);
    let width = expert.width as usize;
    let nb = bx * 128;
    if nb >= width {
        return;
    }
    let start = *(p.route.starts as *const u32).add(id) as usize;
    let g14 = [
        *(expert.gate.global_scale as *const f32) * 16384.0,
        *(expert.up.global_scale as *const f32) * 16384.0,
    ];
    let lane = _thread_idx_x() as usize % 32;
    let select = (lane % 4) as u32 * 0x1111 + 0x4400;
    // Dynamic shared memory (FP4_PREFILL_GATE_SHARED_BYTES), no static region.
    let shared = 0u32;
    // One CTA per M128 task and N128 tile: IL = live M16 tiles (1..8).
    match (m - mb).min(128).div_ceil(16) {
        8 => gate64::<8>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        7 => gate64::<7>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        6 => gate64::<6>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        5 => gate64::<5>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        4 => gate64::<4>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        3 => gate64::<3>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        2 => gate64::<2>(&p, expert, shared, start, mb, m, nb, width, g14, select),
        _ => gate64::<1>(&p, expert, shared, start, mb, m, nb, width, g14, select),
    }
}

#[inline(always)]
unsafe fn st_shared_u32(at: u32, v: u32) {
    core::arch::asm!("st.shared.u32 [{a}], {v};", a = in(reg32) at, v = in(reg32) v);
}

/// copy16_zfill through L1 (.ca): for rows a CTA re-reads.
#[inline(always)]
unsafe fn copy16_zfill_ca(dst: u32, src: *const u8, live: bool) {
    core::arch::asm!("cp.async.ca.shared.global [{d}], [{s}], 16, {n};",
        d = in(reg32) dst, s = in(reg64) src, n = in(reg32) if live { 16u32 } else { 0 });
}

/// L2 prefetch of the line holding `p` (no data returned, no ordering effect).
#[inline(always)]
unsafe fn prefetch_l2(p: *const u8) {
    core::arch::asm!("prefetch.global.L2 [{p}];", p = in(reg64) p);
}

/// 16-byte cp.async that zero-fills (reads nothing) when `live` is false.
#[inline(always)]
unsafe fn copy16_zfill(dst: u32, src: *const u8, live: bool) {
    core::arch::asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
        d = in(reg32) dst, s = in(reg64) src, n = in(reg32) if live { 16u32 } else { 0 });
}

/// Gate weight-row lines prefetched ahead of the ring (one line = 4 K64 steps).
/// 0 (each line requested when its first 32-byte sector is copied) measured
/// best of 0..5: earlier requests only add DRAM queueing.
const GATE_PF: usize = 0;
/// K64 steps ahead that gate64 prefetches its gathered A rows into L2 (the
/// register gather itself runs one step ahead, so 2 gives it one step of lead).
const GATE_A_PF: usize = 2;
/// K16 sub-step after whose MMAs warps 0-3 (LO) and 4-7 (HI) issue the step's producer
/// block (0 = before the first sub-step). Issue placement only: same slots, groups, MMAs.
/// Warps w and w + 4 share an SM sub-partition; with LO at the top and HI after sub-step
/// 0, one of the pair issues MMAs while the other issues its cp.async block.
const GATE_PRODUCE_LO: usize = 0;
const GATE_PRODUCE_HI: usize = 1;
/// Down ring stage: packed B (128 x 32 B) and scale words (128 x 4 B) after the
/// CTA's padded A tile.
const DOWN_STAGE_B: usize = 128 * 32;
const DOWN_STAGE_S: usize = 128 * 4;
const DOWN_STAGES: usize = 3;
/// Staged output tile: 32 rows x 128 BF16 at a 272-byte row stride (ring stage 0).
const DOWN_OUT_STRIDE: usize = 272;

/// Down M64 CTA: 256 threads, eight warps of M64 x N16 (IL live M16
/// tiles, 1..4), so each staged B fragment feeds up to four M16 MMAs and the
/// B block streams once per 64 rows. Per stage: A 64 x 144 B, B 128 x 32 B,
/// scales 128 x 4 B. Same fragments, dequant and ascending K16 chains per output.
const D64_STAGE_A: usize = 64 * 144;
const D64_STAGE: usize = D64_STAGE_A + DOWN_STAGE_B + DOWN_STAGE_S;
const D64_SHARED: usize = DOWN_STAGES * D64_STAGE;
const D64_SLOTS: usize = D64_SHARED;
const D64_SHARED_ALL: usize = D64_SHARED + 64 * 4;
/// Live down64 CTAs per M128 task: x % 2 = M64 plane, x / 2 = first N128 tile.
const D64_X: usize = 4;
const D64_TILES: usize = 2560 / 128 / (D64_X / 2);
const D64_TILE_STRIDE: usize = D64_X / 2 * 128;
#[no_mangle]
pub static FP4_PREFILL_TC_DOWN_THREADS: u32 = 256;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn mainloop_down64<const IL: usize>(
    p: &GroupedParams,
    matrix: Matrix,
    shared: u32,
    start: usize,
    mb: usize,
    m: usize,
    k: usize,
    nb: usize,
    global14: f32,
    select: u32,
) -> [[Acc; 2]; 4] {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let steps = k / 64;
    // Producer: A rows tid/8 and 32+tid/8 (chunk tid%8) below 16*IL; B chunk c of
    // row r; scale word of row tid (tid < 128).
    let (ar, ach) = (tid / 8, tid % 8);
    let a_src = [(ar, mb + ar < m), (ar + 32, mb + ar + 32 < m)].map(|(row, live)| {
        let at = if live { start + mb + row } else { start };
        ((p.input as *const u8).add(at * k * 2 + ach * 16), live)
    });
    let a_dst = [(ar * 144 + ach * 16) as u32, ((ar + 32) * 144 + ach * 16) as u32];
    let a_need = [ar < IL * 16, ar + 32 < IL * 16];
    let (r, c) = (tid >> 1, tid & 1);
    let b_src = (matrix.weights as *const u8).add((nb + r) * (k / 2) + c * 16);
    let b_dst = (D64_STAGE_A + r * 32 + 16 * (c ^ ((r >> 2) & 1))) as u32;
    let s_src = (matrix.scales as *const u8).add((nb + (tid & 127)) * (k / 16));
    let s_dst = (D64_STAGE_A + DOWN_STAGE_B + (tid & 127) * 4) as u32;
    let b_blk = (matrix.weights as *const u8).add(nb * (k / 2));
    let mut line = tid;
    while line < k / 2 {
        prefetch_l2(b_blk.add(line * 128));
        line += 256;
    }
    if tid < k / 16 {
        prefetch_l2((matrix.scales as *const u8).add(nb * (k / 16) + tid * 128));
    }
    let issue = |i: usize, off: u32| unsafe {
        let base = shared + off;
        two!(H, {
            if a_need[H] {
                copy16_zfill_ca(base + a_dst[H], a_src[H].0.add(i * 128), a_src[H].1);
            }
        });
        ptx::copy16(base + b_dst, b_src.add(i * 32) as *const u32);
        if tid < 128 {
            ptx::copy4(base + s_dst, s_src.add(i * 4) as *const u32);
        }
    };
    let mut i = 0;
    while i < DOWN_STAGES - 1 {
        if i < steps {
            issue(i, (i * D64_STAGE) as u32);
        }
        ptx::commit();
        i += 1;
    }
    let own = warp * 16 + lane / 4;
    let sw = (lane >> 4) & 1;
    let b_lane = (D64_STAGE_A + own * 32) as u32;
    let zoff = [(16 * sw) as u32, (16 - 16 * sw) as u32];
    let s_lane = (D64_STAGE_A + DOWN_STAGE_B + own * 4) as u32;
    let a_lane = ((lane & 15) * 144 + (lane >> 4) * 16) as u32;
    let mut sums = [[ZERO; 2]; 4];
    let mut step = 0;
    // Ring offsets of this step's slot and of the slot read in the previous step (the one
    // step + DOWN_STAGES - 1 refills), carried instead of recomputed modulo DOWN_STAGES.
    let mut cur = 0u32;
    let mut prev = ((DOWN_STAGES - 1) * D64_STAGE) as u32;
    while step < steps {
        wait_group::<{ DOWN_STAGES - 2 }>();
        ptx::sync();
        if step + DOWN_STAGES - 1 < steps {
            issue(step + DOWN_STAGES - 1, prev);
        }
        ptx::commit();
        let base = shared + cur;
        let bq = [base + b_lane + zoff[0], base + b_lane + zoff[1]];
        let current = base + a_lane;
        let mut scale = [[0u32; 2]; 2];
        two!(J, {
            scale[J] = scales_f16(load_b32(base + s_lane + (J * 32) as u32));
        });
        four!(Z, {
            let mut a = [[0u32; 4]; 4];
            four!(I, {
                if I < IL {
                    a[I] = ldsm4(current + (I * 16 * 144 + Z * 32) as u32);
                }
            });
            two!(J, {
                let (lo, hi) = load_b64(bq[Z >> 1] + (J * 256 + 8 * (Z & 1)) as u32);
                let b = dequant::<{ Z & 1 == 1 }>(lo, hi, select, scale[J][Z >> 1], global14);
                four!(I, {
                    if I < IL {
                        sums[I][J] = madd(a[I], b, sums[I][J]);
                    }
                });
            });
        });
        prev = cur;
        cur = if cur as usize + D64_STAGE == D64_SHARED { 0 } else { cur + D64_STAGE as u32 };
        step += 1;
    }
    sums
}

/// down64 epilogue: BF16 pairs staged in the drained ring (row stride
/// DOWN_OUT_STRIDE), then 16-byte row stores into each row's slot.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn down64_store<const IL: usize>(
    p: &GroupedParams,
    sums: &[[Acc; 2]; 4],
    shared: u32,
    mb: usize,
    m: usize,
    nb: usize,
    warp: usize,
    lane: usize,
) {
    let tid = warp * 32 + lane;
    ptx::sync();
    four!(I, {
        if I < IL {
            two!(J, {
                let sum = sums[I][J];
                let col = (warp * 2 + J) * 8 + lane % 4 * 2;
                let row = I * 16 + lane / 4;
                st_shared_u32(shared + (row * DOWN_OUT_STRIDE + col * 2) as u32,
                    crate::next_math::bf16(sum.a) | (crate::next_math::bf16(sum.b) << 16));
                st_shared_u32(shared + ((row + 8) * DOWN_OUT_STRIDE + col * 2) as u32,
                    crate::next_math::bf16(sum.c) | (crate::next_math::bf16(sum.d) << 16));
            });
        }
    });
    ptx::sync();
    let out = p.output as *mut u8;
    four!(T, {
        let chunk = tid + 256 * T;
        let (row, cc) = (chunk / 16, chunk % 16);
        if row < IL * 16 && mb + row < m {
            let slot = load_b32(shared + (D64_SLOTS + row * 4) as u32) as usize;
            let (a, b, c, d): (u32, u32, u32, u32);
            core::arch::asm!("ld.shared.v4.b32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
                p = in(reg32) shared + (row * DOWN_OUT_STRIDE + cc * 16) as u32);
            let dst = out.add(slot * 10240 + (nb + cc * 8) * 2);
            core::arch::asm!("st.global.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                p = in(reg64) dst, a = in(reg32) a, b = in(reg32) b, c = in(reg32) c, d = in(reg32) d);
        }
    });
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn down64_tiles<const IL: usize>(
    p: &GroupedParams,
    matrix: Matrix,
    shared: u32,
    start: usize,
    mb: usize,
    m: usize,
    k: usize,
    nb: usize,
    global14: f32,
    select: u32,
) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let mut t = 0;
    while t < D64_TILES {
        let nbt = nb + t * D64_TILE_STRIDE;
        if t > 0 {
            ptx::sync();
        }
        let sums = mainloop_down64::<IL>(p, matrix, shared, start, mb, m, k, nbt, global14, select);
        down64_store::<IL>(p, &sums, shared, mb, m, nbt, warp, lane);
        t += 1;
    }
}

/// Grid (D64_X = 4,task_capacity,1), block 256 (FP4_PREFILL_TC_DOWN_THREADS); a wider x exits.
#[inline(always)]
unsafe fn run_down64(p: GroupedParams) {
    let tasks = p.route.tasks as *const u32;
    let task = _block_idx_y() as usize;
    if task >= *tasks as usize {
        return;
    }
    let bx = _block_idx_x() as usize;
    if bx >= D64_X {
        return;
    }
    let id = *tasks.add(1 + task * 2) as usize;
    let mb = *tasks.add(2 + task * 2) as usize + (bx % 2) * 64;
    let m = *(p.route.counts as *const u32).add(id) as usize;
    if mb >= m {
        return;
    }
    let expert = *(p.weights as *const Expert).add(id);
    let width = expert.width as usize;
    let start = *(p.route.starts as *const u32).add(id) as usize;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let nb = (bx / 2) * 128;
    if width == 0 {
        let mut t = 0;
        while t < D64_TILES {
            let nbt = nb + t * D64_TILE_STRIDE;
            four!(I, { two!(J, {
                write_ranked(p,start,ZERO,mb+I*16+lane/4,
                    nbt+(warp*2+J)*8+lane%4*2,m);
            }); });
            t += 1;
        }
        return;
    }
    let matrix = expert.down;
    let global14 = *(matrix.global_scale as *const f32) * 16384.0;
    let select = (lane % 4) as u32 * 0x1111 + 0x4400;
    let shared = ptx::shared::<D64_SHARED_ALL>();
    if tid < 64 {
        let r = mb + tid;
        let v = if r < m {
            let pair = *(p.route.sorted_row as *const u32).add(start + r) as usize;
            *(p.route.pair_to_row as *const u32).add(pair)
        } else {
            u32::MAX
        };
        st_shared_u32(shared + (D64_SLOTS + tid * 4) as u32, v);
    }
    match (m - mb).min(64).div_ceil(16) {
        3 | 4 => down64_tiles::<4>(&p, matrix, shared, start, mb, m, width, nb, global14, select),
        _ => down64_tiles::<2>(&p, matrix, shared, start, mb, m, width, nb, global14, select),
    }
}

/// Grid (5,task_capacity,1), block256, dynamic shared FP4_PREFILL_GATE_SHARED_BYTES
/// (no static region); x < width/128 are the N128 tiles of the M128 task y.
/// Dynamic shared bytes of fp4_prefill_tc_gate; hosts read it from the module so a
/// launch always matches the kernel it runs.
#[no_mangle]
pub static FP4_PREFILL_TC_GATE_SHARED: u32 = crate::kernel_params::FP4_PREFILL_GATE_SHARED_BYTES;
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_prefill_tc_gate(p: GroupedParams) {
    run_gate64(p);
}
/// Gate A operand for one chunk: FP32 rows -> RN BF16 pairs (crate::gemm::bf16_pair,
/// the rounding the gate applied when it gathered FP32), `groups` x 8 values.
/// Grid groups.div_ceil(256), block 256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_prefill_tc_a16(input: *const f32, output: *mut u32, groups: u32) {
    let i = (_block_idx_x() * 256 + _thread_idx_x()) as usize;
    if i >= groups as usize {
        return;
    }
    let v: [f32; 8];
    {
        let (a, b, c, d, e, f, g, h): (f32, f32, f32, f32, f32, f32, f32, f32);
        let src = input.add(i * 8);
        core::arch::asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) src);
        core::arch::asm!("ld.global.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) e, b = out(reg32) f, c = out(reg32) g, d = out(reg32) h, p = in(reg64) src.add(4));
        v = [a, b, c, d, e, f, g, h];
    }
    let w = [
        crate::gemm::bf16_pair(v[0], v[1]),
        crate::gemm::bf16_pair(v[2], v[3]),
        crate::gemm::bf16_pair(v[4], v[5]),
        crate::gemm::bf16_pair(v[6], v[7]),
    ];
    core::arch::asm!("st.global.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
        p = in(reg64) output.add(i * 4), a = in(reg32) w[0], b = in(reg32) w[1], c = in(reg32) w[2], d = in(reg32) w[3]);
}
/// Grid (4,task_capacity,1), block 256 (FP4_PREFILL_TC_DOWN_THREADS); static
/// shared D64_SHARED_ALL; no split-K. x % 2 = M64 plane, x < D64_X live.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_prefill_tc_down(p: GroupedParams) {
    run_down64(p);
}

/// Benchmark-only probe (never a default or serve path): the prompt gate
/// with A4 activations, NVFP4 per-16 E4M3 block scales from fp4_prefill_tc_a4probe_quant,
/// on mma.m16n8k64.kind::mxf4nvf4 straight from the stored weights and scales (no
/// dequant). Same tasks, tiles and outputs as fp4_prefill_tc_gate; values differ by
/// design (A rounded to E2M1). Needs the --fp4 (sm_121a) PTX. Per stage: A 128 rows x
/// 32 B (padded to 48) + 4 scale bytes, B 128 gate + 128 up rows likewise.
/// Rows are padded to A4P_ROW bytes so the eight fragment rows of a load hit distinct banks.
#[cfg(feature = "fp4")]
const A4P_ROW: usize = 48;
#[cfg(feature = "fp4")]
const A4P_A: usize = 128 * A4P_ROW;
#[cfg(feature = "fp4")]
const A4P_AS: usize = 128 * 4;
#[cfg(feature = "fp4")]
const A4P_B: usize = 256 * A4P_ROW;
#[cfg(feature = "fp4")]
const A4P_STAGE: usize = A4P_A + A4P_AS + A4P_B + 256 * 4;
#[cfg(feature = "fp4")]
const A4P_STAGES: usize = 3;
#[cfg(feature = "fp4")]
const _: () = assert!(A4P_STAGES * A4P_STAGE <= crate::kernel_params::FP4_PREFILL_GATE_SHARED_BYTES as usize);

#[cfg(feature = "fp4")]
#[inline(always)]
unsafe fn a4p_gate<const IL: usize>(p: &GroupedParams, expert: Expert, start: usize, mb: usize, m: usize, nb: usize, width: usize) {
    const K: usize = 2560;
    const STEPS: usize = K / 64;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let shared = 0u32;
    let rows_total = p.route.tokens as usize;
    let a4 = p.input as *const u8;
    let a4s = a4.add(rows_total * K / 2);
    // Producer: A row tid/2 chunk tid&1, A scale word of row tid (< 128), B row tid
    // (gate rows 0..128, up rows 128..256) both chunks plus its scale word.
    let arow = if tid / 2 < IL * 16 { Some(crate::compact::input_row(p.route.sorted_row, start, mb, tid / 2, m, false)) } else { None };
    let srow = if tid < IL * 16 { crate::compact::input_row(p.route.sorted_row, start, mb, tid, m, false) } else { None };
    let (bm, br) = if tid < 128 { (expert.gate, nb + tid) } else { (expert.up, nb + tid - 128) };
    let b_src = (bm.weights as *const u8).add(br * (K / 2));
    let bs_src = (bm.scales as *const u8).add(br * (K / 16));
    macro_rules! issue {
        ($arg:expr) => {{
            let s: usize = $arg;
            let base = shared + ((s % A4P_STAGES) * A4P_STAGE) as u32;
            match arow {
                Some(Some(r)) => copy16_zfill(base + ((tid / 2) * A4P_ROW + (tid & 1) * 16) as u32, a4.add(r * (K / 2) + s * 32 + (tid & 1) * 16), true),
                Some(None) => copy16_zfill(base + ((tid / 2) * A4P_ROW + (tid & 1) * 16) as u32, a4, false),
                None => {}
            }
            if let Some(r) = srow {
                ptx::copy4(base + (A4P_A + tid * 4) as u32, a4s.add(r * (K / 16) + s * 4) as *const u32);
            }
            let bb = base + (A4P_A + A4P_AS) as u32;
            ptx::copy16(bb + (tid * A4P_ROW) as u32, b_src.add(s * 32) as *const u32);
            ptx::copy16(bb + (tid * A4P_ROW + 16) as u32, b_src.add(s * 32 + 16) as *const u32);
            ptx::copy4(bb + (A4P_B + tid * 4) as u32, bs_src.add(s * 4) as *const u32);
        }};
    }
    let mut i = 0;
    while i < A4P_STAGES - 1 {
        if i < STEPS {
            issue!(i);
        }
        ptx::commit();
        i += 1;
    }
    let (g, q) = (lane / 4, lane % 4);
    let mut sums = [[ZERO; 4]; 8];
    let mut step = 0;
    while step < STEPS {
        wait_group::<{ A4P_STAGES - 2 }>();
        ptx::sync();
        if step + A4P_STAGES - 1 < STEPS {
            issue!(step + A4P_STAGES - 1);
        }
        ptx::commit();
        let base = shared + ((step % A4P_STAGES) * A4P_STAGE) as u32;
        let (ab, asb) = (base, base + A4P_A as u32);
        let (bb, bsb) = (base + (A4P_A + A4P_AS) as u32, base + (A4P_A + A4P_AS + A4P_B) as u32);
        let mut b = [[0u32; 2]; 4];
        let mut sb = [0u32; 4];
        four!(J, {
            // J 0/1: gate rows warp*16 + 8J + g; J 2/3: up rows 128 + warp*16 + 8(J-2) + g.
            let row = (J >> 1) * 128 + warp * 16 + (J & 1) * 8 + g;
            b[J] = [load_b32(bb + (row * A4P_ROW + q * 4) as u32), load_b32(bb + (row * A4P_ROW + (q + 4) * 4) as u32)];
            sb[J] = load_b32(bsb + (row * 4) as u32);
        });
        eight!(I, {
            if I < IL {
                let r = I * 16 + g;
                let a = [
                    load_b32(ab + (r * A4P_ROW + q * 4) as u32),
                    load_b32(ab + ((r + 8) * A4P_ROW + q * 4) as u32),
                    load_b32(ab + (r * A4P_ROW + (q + 4) * 4) as u32),
                    load_b32(ab + ((r + 8) * A4P_ROW + (q + 4) * 4) as u32),
                ];
                let sa = load_b32(asb + ((r + (q % 2) * 8) * 4) as u32);
                four!(J, {
                    sums[I][J] = crate::fp4::mma(a, b[J], sa, sb[J], sums[I][J]);
                });
            }
        });
        step += 1;
    }
    let (gs, us) = (*(expert.gate.global_scale as *const f32), *(expert.up.global_scale as *const f32));
    let y = (p.output as *mut u32).add(start * width / 2);
    eight!(I, {
        if I < IL {
            two!(J, {
                let (gg, uu) = (sums[I][J], sums[I][J + 2]);
                let row = mb + I * 16 + g;
                let col = nb + warp * 16 + J * 8 + q * 2;
                if row < m {
                    *y.add(row * width / 2 + col / 2) = next_activation_pair(gg.a * gs, gg.b * gs, uu.a * us, uu.b * us);
                }
                if row + 8 < m {
                    *y.add((row + 8) * width / 2 + col / 2) = next_activation_pair(gg.c * gs, gg.d * gs, uu.c * us, uu.d * us);
                }
            });
        }
    });
}

/// Grid (5,task_capacity,1), block 256, dynamic shared FP4_PREFILL_GATE_SHARED_BYTES
/// (kbench passes fp4_prefill_tc_gate's), input = fp4_prefill_tc_a4probe_quant's buffer.
#[cfg(feature = "fp4")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_prefill_tc_a4probe_gate(p: GroupedParams) {
    let tasks = p.route.tasks as *const u32;
    let task = _block_idx_y() as usize;
    let bx = _block_idx_x() as usize;
    if task >= *tasks as usize {
        return;
    }
    let id = *tasks.add(1 + task * 2) as usize;
    let mb = *tasks.add(2 + task * 2) as usize;
    let m = *(p.route.counts as *const u32).add(id) as usize;
    if mb >= m {
        return;
    }
    let expert = *(p.weights as *const Expert).add(id);
    let width = expert.width as usize;
    let nb = bx * 128;
    if nb >= width {
        return;
    }
    let start = *(p.route.starts as *const u32).add(id) as usize;
    match (m - mb).min(128).div_ceil(16) {
        8 => a4p_gate::<8>(&p, expert, start, mb, m, nb, width),
        7 => a4p_gate::<7>(&p, expert, start, mb, m, nb, width),
        6 => a4p_gate::<6>(&p, expert, start, mb, m, nb, width),
        5 => a4p_gate::<5>(&p, expert, start, mb, m, nb, width),
        4 => a4p_gate::<4>(&p, expert, start, mb, m, nb, width),
        3 => a4p_gate::<3>(&p, expert, start, mb, m, nb, width),
        2 => a4p_gate::<2>(&p, expert, start, mb, m, nb, width),
        _ => a4p_gate::<1>(&p, expert, start, mb, m, nb, width),
    }
}

/// Probe pre-pass: FP32 rows -> NVFP4 (E2M1 nibbles, element 2j in the low nibble) with
/// one E4M3 scale (RN of amax / 6) per 16 values; packed rows first (rows*K/2 bytes),
/// then the scales. `groups` = rows*K/8 (the a16 contract); one thread per 16 values.
#[cfg(feature = "fp4")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_prefill_tc_a4probe_quant(input: *const f32, output: *mut u32, groups: u32) {
    let n = groups as usize / 2;
    let i = (_block_idx_x() * 256 + _thread_idx_x()) as usize;
    if i >= n {
        return;
    }
    let x = core::array::from_fn::<f32, 16, _>(|j| *input.add(i * 16 + j));
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s8: u32;
    core::arch::asm!("{{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, {z}, {v}; cvt.u32.u16 {d}, t; }}",
        d = out(reg32) s8, z = in(reg32) 0.0f32, v = in(reg32) amax / 6.0);
    let s8 = s8 & 0xff;
    let sdec: f32;
    core::arch::asm!("{{ .reg .b16 t; .reg .b32 h; .reg .b16 lo; .reg .b16 hi; cvt.u16.u32 t, {s}; cvt.rn.f16x2.e4m3x2 h, t; mov.b32 {{lo, hi}}, h; cvt.f32.f16 {d}, lo; }}",
        d = out(reg32) sdec, s = in(reg32) s8);
    let inv = if sdec > 0.0 { 1.0 / sdec } else { 0.0 };
    let mut w = [0u32; 2];
    let mut j = 0;
    while j < 8 {
        let byte: u32;
        core::arch::asm!("{{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, {hi}, {lo}; cvt.u32.u8 {d}, t; }}",
            d = out(reg32) byte, hi = in(reg32) x[2 * j + 1] * inv, lo = in(reg32) x[2 * j] * inv);
        w[j / 4] |= (byte & 0xff) << ((j % 4) * 8);
        j += 1;
    }
    *output.add(i * 2) = w[0];
    *output.add(i * 2 + 1) = w[1];
    *(output as *mut u8).add(n * 8 + i) = s8 as u8;
}

/// Carry the next K16 sub-step's B fragments; sub-step 3 has none. Written through a
/// reference so the unrolled sub-step 3 does not read as a never-read assignment.
#[inline(always)]
fn carry<const Z: usize>(b: &mut [[u32; 2]; 4], bn: [[u32; 2]; 4]) {
    if Z < 3 {
        *b = bn;
    }
}
