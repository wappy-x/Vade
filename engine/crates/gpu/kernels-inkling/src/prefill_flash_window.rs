//! SM121a BF16 tensor-core high-occupancy 2-stage prefill causal/sliding-window attention
//! (`resident_launch_prefill_flash_window`, kind `PREFILL_FLASH_WINDOW = 122`).
//!
//! Produces bit-for-bit identical output to `FLASH_WINDOW_ATTN` (kind 105) with:
//! - `BQ_POS = 64` query positions x `gpack` GQA heads (`192` Q rows for 3:1 GQA, `384` threads = `12` warps,
//!   `1` blk/SM): 2x K/V/T reuse per query row vs `BQ_POS = 32` while keeping 12 warps/SM occupancy at 168 regs.
//! - True 2-stage double-buffered 64-key `K`/`V` and 32-key `T` (`97,088 B <= 101,184 B` static SMEM):
//!   **1 single `_syncthreads()` per 64 interior keys** (or per 32 boundary/bias keys), with `Q` dims `0..96`
//!   pinned in 24 registers (`q_reg0..5`) and dims `96..128` in permanent bank-free `Q_HI` (`12,288 B`).
//! - Global LPT scheduling (`q_tile_rev` outermost across all KV heads) for zero tail skew and 6x cross-CTA L2 reuse.
//! - Branchless bit-exact `exp_nonpos` (eliminating saturating `f32 as i32` `F2I` + `FSETP` + `BRA`), `cvt.rn.bf16x2`
//!   paired BF16 rounding, zero-branch `interior_rel` fast path for sliding-window layers, XOR-swizzled bank-free `M`,
//!   two-pass `RelBias` fed by four-term `ld.shared.v4` loads (48 `T` loads per tile, not 192), and coalesced
//!   dead-SMEM `st.global.v4.b32` BF16 epilogue.

use crate::fixed_math::rn;
use crate::flash_window_abi::{
    FW_BKV, FW_CLIP_IN, FW_CLIP_WORD, FW_HALF_OUT, FW_HEAD_DIM, FW_PREFILL_BQ_POS, FW_QG_HALF, FW_REL_DIM,
    flash_window_gpack, flash_window_prefill_admitted,
};
use crate::tc_ptx::{
    cp_async_wait_zero, cvt_bf16x2, l2_evict_first, ld_global_nc_u32, ld_global_nc_v4,
    ld_global_nc_v4_u32_hint, ldmatrix_x4, ldmatrix_x4_trans, load_u32, mma_m16n8k16_bf16,
    shfl_xor_f32, shfl_xor_u32, st_shared_v4, st_shared_zero16, store_u16, store_u32,
};

const MAX_Q_ROWS: usize = 3 * FW_PREFILL_BQ_POS; // 192
const K_BYTES: usize = FW_BKV * FW_HEAD_DIM * 2; // 8,192 (one 32-key sub-tile)
const V_BYTES: usize = FW_BKV * FW_HEAD_DIM * 2; // 8,192 (one 32-key sub-tile)
const K_STAGE_BYTES: usize = 2 * K_BYTES; // 16,384 (64-key double-buffered stage once Q moves into 32 regs)
const V_STAGE_BYTES: usize = 2 * V_BYTES; // 16,384 (64-key double-buffered stage once Q moves into 32 regs)
const Q_BYTES: usize = MAX_Q_ROWS * FW_HEAD_DIM * 2; // 49,152 (prologue-only temp buffer overlaid on K/V)
const Q_HI_BYTES: usize = MAX_Q_ROWS * 32 * 2; // 12,288 (permanent SMEM for Q dims 96..128; dims 0..96 stay in 24 regs)
const M_BYTES: usize = MAX_Q_ROWS * FW_REL_DIM * 2; // 6,144 (XOR-swizzled, bank-conflict-free)
const L_BYTES: usize = FW_PREFILL_BQ_POS * 4; // 256
/// Staged `T` is distance-major inside four term planes: plane `q` holds, for each tile distance slot, the four
/// FP32 terms `j = 4q..4q+3` in one 16 B chunk, so one `ld.shared.v4` feeds four terms of the bias sum.
/// A `64 x 32` tile reads slots `0..=94` (`95` distances); the plane pitch stays 100 slots.
const T_SLOTS: usize = FW_PREFILL_BQ_POS + FW_BKV - 1; // 95
const T_PITCH: usize = 100;
const T_PLANE_BYTES: usize = T_PITCH * 16; // 1,600
const T_STAGE_BYTES: usize = (FW_REL_DIM / 4) * T_PLANE_BYTES; // 6,400
const C_BYTES: usize = 64; // 16 x u32 CTA constants (keeps loop-live regs <= 168 with 0 spills)
/// `Q` is staged transiently in `[0 .. 49,152)` during the prologue: dims `0..96` load once into 24 regs (`q_reg0..5`),
/// dims `96..128` persist in `Q_HI` (`12,288 B`), and `[0 .. 65,536)` holds two 64-key `K`/`V` stages (1 barrier/64 keys).
const SMEM_BYTES: usize = 2 * K_STAGE_BYTES + 2 * V_STAGE_BYTES + Q_HI_BYTES + M_BYTES + L_BYTES + 2 * T_STAGE_BYTES + C_BYTES; // 97,088

const Q_OFF: u32 = 0; // prologue-only temp [0 .. 49,152), strictly below Q_HI_OFF (65,536)
const K_OFF: u32 = 0; // 2 x 16,384 = 32,768
const V_OFF: u32 = (2 * K_STAGE_BYTES) as u32; // 32,768 .. 65,536
const Q_HI_OFF: u32 = V_OFF + (2 * V_STAGE_BYTES) as u32; // 65,536 .. 77,824
const M_OFF: u32 = Q_HI_OFF + Q_HI_BYTES as u32; // 77,824
const L_OFF: u32 = M_OFF + M_BYTES as u32; // 83,968
const T_OFF: u32 = L_OFF + L_BYTES as u32; // 84,224
const C_OFF: u32 = T_OFF + (2 * T_STAGE_BYTES) as u32; // 97,024

const _: () = assert!(FW_PREFILL_BQ_POS == 64 && FW_BKV == 32 && FW_HEAD_DIM == 128 && FW_REL_DIM == 16);
const _: () = assert!(SMEM_BYTES == 97_088 && SMEM_BYTES <= 101_184 && Q_BYTES <= Q_HI_OFF as usize);
const _: () = assert!(T_SLOTS == 95 && T_SLOTS <= T_PITCH && FW_REL_DIM == 16 && T_STAGE_BYTES == 6_400);
const _: () = assert!(K_OFF % 16 == 0 && V_OFF % 16 == 0 && Q_HI_OFF % 16 == 0 && M_OFF % 16 == 0 && T_OFF % 16 == 0 && C_OFF % 16 == 0);
/// Dead SMEM reused by coalesced BF16 epilogue: 12 warps x 16 rows x 68 u32 (272 B/row, bank-free) = 52,224 B.
const EPI_ROW_BYTES: usize = 68 * 4; // 272 B
const EPI_WARP_BYTES: usize = 16 * EPI_ROW_BYTES; // 4,352 B
const _: () = assert!(12 * EPI_WARP_BYTES <= C_OFF as usize);
const _: () = assert!(core::f32::consts::LOG2_E.to_bits() == 0x3fb8_aa3b && 12_582_912.0f32.to_bits() == 0x4b40_0000);

/// Zero-bank-conflict XOR-swizzled byte offset within a 256-byte-stride BF16 tile (`row`, 16B `chunk` in 0..16).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 256 + ((chunk ^ (row & 7)) * 16)) as u32
}

/// Zero-bank-conflict XOR-swizzled byte offset within 64-byte-stride `Q_HI` (`row`, 16B `chunk` in 0..4):
/// `(row >> 1) & 3` maps every 8 consecutive rows to all 32 banks without conflict on `ldmatrix.x4`.
#[inline(always)]
const fn swiz64(row: usize, chunk: usize) -> u32 {
    (row * 64 + ((chunk ^ ((row >> 1) & 3)) * 16)) as u32
}

/// Zero-bank-conflict 8-byte XOR-swizzled byte offset within `M` (`row`, 4-BF16 quad `jq` in 0..4):
/// rows 0..3 occupy banks 0, 8, 16, 24; rows 4..7 flip bit 0 of `jq` so all 8 rows hit 16 distinct banks on `ld.shared.v2.b32`.
#[inline(always)]
const fn swiz_m8(row: usize, jq: usize) -> u32 {
    ((row * 4 + (jq ^ ((row >> 2) & 1))) * 8) as u32
}

/// Paired BF16 round-to-nearest-even via single `cvt.rn.bf16x2.f32` (bit-identical to `(rn(a), rn(b))` for finite inputs).
/// Uses `prmt.b32` (ALU pipe) for the low-half shift so BF16 unpacking does not contend with FMA/IMAD pipe.
/// SAFETY: pure register instructions, touches no memory.
#[inline(always)]
unsafe fn rn2(a: f32, b: f32) -> (f32, f32) {
    let p = cvt_bf16x2(a, b);
    let lo: u32;
    core::arch::asm!(
        "prmt.b32 {lo}, {p}, 0, 0x1044;",
        lo = out(reg32) lo, p = in(reg32) p,
        options(pure, nomem, nostack),
    );
    (f32::from_bits(lo), f32::from_bits(p & 0xffff_0000))
}

/// Bit-exact `fixed_math::exp(t)` for non-positive inputs `t <= 0.0` (including `-0.0` and `-87.0` clamping),
/// eliminating the dead `t > 87.0` branch and Rust's saturating `f32 as i32` `F2I`/`FSETP`/`BRA` sequence.
/// SAFETY: pure register instructions, touches no memory.
#[inline(always)]
unsafe fn exp_nonpos(t: f32) -> f32 {
    let t = if t < -87.0f32 { -87.0f32 } else { t };
    let k: f32;
    let n: f32;
    core::arch::asm!(
        "{{ .reg .f32 p; \
         mul.rn.f32 p, {t}, 0f3fb8aa3b; \
         add.rn.f32 {k}, p, 0f4b400000; \
         add.rn.f32 {n}, {k}, 0fcb400000; }}",
        k = out(reg32) k,
        n = out(reg32) n,
        t = in(reg32) t,
        options(pure, nomem, nostack),
    );
    let r = core::f32::math::mul_add(-n, 0.693145751953125f32, t);
    let r = core::f32::math::mul_add(-n, 0.0000014286068f32, r);
    let p = core::f32::math::mul_add(1.0 / 5040.0f32, r, 1.0 / 720.0);
    let p = core::f32::math::mul_add(p, r, 1.0 / 120.0);
    let p = core::f32::math::mul_add(p, r, 1.0 / 24.0);
    let p = core::f32::math::mul_add(p, r, 1.0 / 6.0);
    let p = core::f32::math::mul_add(p, r, 0.5);
    let p = core::f32::math::mul_add(p, r, 1.0);
    let p = core::f32::math::mul_add(p, r, 1.0);
    p * f32::from_bits(k.to_bits().wrapping_add(127) << 23)
}

/// Replay one recorded command of kind `PREFILL_FLASH_WINDOW` (kind 122).
/// SAFETY (caller contract, `resident_launch_prefill_flash_window`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `PREFILL_FLASH_WINDOW` recorded by `Backend::prefill_flash_window_into` after
/// `flash_window::check_flash_window_form` verified every extent, alignment, and non-overlap.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind traps before reading payload words.
    // - Host verified via `check_flash_window_form_in` and `flash_window_prefill_admitted` that `query`/`gate` hold
    //   `rows * heads * 128` FP32 (with `FW_QG_HALF`: as many BF16 values, 16 bits each, 16B aligned), `keys`/`values`
    //   hold `view * kv_heads * 64` u32 packed BF16, `moved` holds `rows * heads * 16` FP32, `rel_table` holds
    //   `16 * extent` FP32 (16B aligned), `sink` holds `rows * heads` FP32, `steps` holds `rows * 2` FP32, `output`
    //   holds `rows * heads * 128` FP32 (or `/ 2` u32 packed BF16, 16B aligned), disjoint from all inputs.
    // - Device re-checks `flash_window_prefill_admitted`, contiguous chat step invariants (`1 <= live <= view`,
    //   `slot < view`, `slot_r == (slot_0 + r) % view`, `arc_len <= view`), and traps on any malformed step.
    // - Shared memory stays strictly inside the block's static `SMEM_BYTES` (97,088 bytes <= 101,184 bytes).
    // - The bias loop reads staged `T` only at slots `0..=94` of the four term planes of the current stage. The
    //   stage issue writes every one of those `95 x 16` words (a `cp.async` copy or a zero store) before the
    //   `cp.async.wait_group` and `_syncthreads()` that publish the stage, so no load reads unwritten memory.
    // - All `threads` threads participate in every `_syncthreads()` and `cp.async.wait_group`; every warp participates
    //   uniformly in every `ldmatrix`/`mma`/`shfl`/`bar.warp.sync`.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PREFILL_FLASH_WINDOW {
            core::arch::nvptx::trap();
        }
        let w = c.payload.as_ptr().cast::<u64>();
        let query = *w as *const f32;
        let keys = *w.add(1) as *const u32;
        let values = *w.add(2) as *const u32;
        let moved = *w.add(3) as *const f32;
        let rel_table = *w.add(4) as *const f32;
        let sink = *w.add(5) as *const f32;
        let gate = *w.add(6) as *const f32;
        let steps = *w.add(7) as *const f32;
        let output = *w.add(8) as *mut f32;
        let rows = *w.add(9) as usize;
        let heads = *w.add(10) as usize;
        let kv_heads = *w.add(11) as usize;
        let view = *w.add(12) as usize;
        let extent = *w.add(13) as usize;
        let scale = f32::from_bits(*w.add(14) as u32);
        let first_key_phase = *w.add(15) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;

        if !flash_window_prefill_admitted(
            rows, heads, kv_heads, view, extent,
            query as usize, keys as usize, values as usize, moved as usize,
            rel_table as usize, gate as usize, output as usize, threads,
        ) {
            core::arch::nvptx::trap();
        }

        let gpack = flash_window_gpack(heads, kv_heads);
        let group = heads / kv_heads;
        let sub_groups = group / gpack;
        let q_tiles = rows.div_ceil(FW_PREFILL_BQ_POS);
        // Global LPT scheduling: `q_tile_rev` outermost so deepest Q tiles run first across ALL KV heads
        // simultaneously (zero wave tail skew + 6x cross-CTA L2 reuse of K/V).
        let sub_g = block % sub_groups;
        let tmp = block / sub_groups;
        let kv_head = tmp % kv_heads;
        let q_tile_rev = tmp / kv_heads;
        if q_tile_rev >= q_tiles {
            return;
        }
        let q_tile = q_tiles - 1 - q_tile_rev;
        let head_base = kv_head * group + sub_g * gpack;
        let q0 = q_tile * FW_PREFILL_BQ_POS;
        let m_cnt = (rows - q0).min(FW_PREFILL_BQ_POS);

        let parse_count = |v: f32, max_v: usize| -> usize {
            let n = v as usize;
            if n as f32 != v || n > max_v {
                core::arch::nvptx::trap();
            }
            n
        };
        let live0 = parse_count(*steps.add(2 * q0), view);
        let slot0 = parse_count(*steps.add(2 * q0 + 1), view - 1);
        if live0 == 0 {
            core::arch::nvptx::trap();
        }
        let arc_start = if live0 - 1 <= slot0 { slot0 - (live0 - 1) } else { slot0 + view - (live0 - 1) };
        let arc_len = (live0 - 1) + m_cnt;
        if arc_len > view {
            core::arch::nvptx::trap();
        }
        let pad = (first_key_phase % FW_BKV + q0 + FW_BKV - (live0 - 1) % FW_BKV) % FW_BKV;
        let arc_end = pad + arc_len;

        let smem = crate::ptx::shared::<SMEM_BYTES>();
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid >> 5, tid & 31);
        let g = lane >> 2;
        let q_rows = gpack * FW_PREFILL_BQ_POS;
        let h_local = warp >> 2;
        let w_pos = warp & 3;
        let head = head_base + h_local;
        let q_row_base = h_local * FW_PREFILL_BQ_POS;

        let q_base = smem + Q_OFF;
        let m_base = smem + M_OFF;
        let l_base = smem + L_OFF;
        let c_base = smem + C_OFF;

        // Warp 0 validates all m_cnt step rows (2 per lane), stores `live_r` into L, reduces `max_min_col`,
        // and writes all CTA-uniform constants into `c_base` before the single prologue `_syncthreads()`.
        if tid < 32 {
            let check_step = |r: usize| -> u32 {
                if r < m_cnt {
                    let lr = parse_count(*steps.add(2 * (q0 + r)), view);
                    let sr = parse_count(*steps.add(2 * (q0 + r) + 1), view - 1);
                    let expect_s = { let s = slot0 + r; if s >= view { s - view } else { s } };
                    if lr == 0 || lr > live0 + r || sr != expect_s { core::arch::nvptx::trap(); }
                    store_u32(l_base + (r * 4) as u32, lr as u32);
                    (pad + live0 + r - lr) as u32
                } else {
                    store_u32(l_base + (r * 4) as u32, 0);
                    0
                }
            };
            let (mc_a, mc_b) = (check_step(tid), check_step(tid + 32));
            let mut max_mc = if mc_b > mc_a { mc_b } else { mc_a };
            let mut sh = 16u32;
            while sh > 0 {
                let o = shfl_xor_u32(max_mc, sh);
                if o > max_mc { max_mc = o; }
                sh >>= 1;
            }
            if tid == 0 {
                store_u32(c_base, view as u32);
                store_u32(c_base + 4, extent as u32);
                store_u32(c_base + 8, pad as u32);
                store_u32(c_base + 12, arc_start as u32);
                store_u32(c_base + 16, arc_end as u32);
                store_u32(c_base + 20, live0 as u32);
                store_u32(c_base + 24, (kv_heads * 64) as u32);
                store_u32(c_base + 28, m_cnt as u32);
                store_u32(c_base + 32, q0 as u32);
                store_u32(c_base + 36, head_base as u32);
                store_u32(c_base + 40, heads as u32);
                store_u32(c_base + 44, kv_head as u32);
                store_u32(c_base + 48, max_mc);
                store_u32(c_base + 52, threads as u32);
                // Precompute contiguous non-wrapping, no-bias, fully-unmasked 64-key interior interval `[0, t_int_end)`
                // for the 384-thread GQA=3 fast loop (covers ~96-98% of global 128K-300K tiles with 1 barrier/64 keys).
                let pad32 = pad as u32;
                let lo = if pad32 > max_mc { pad32 } else { max_mc };
                let hi_rel = if pad32 + live0 as u32 >= extent as u32 + FW_BKV as u32 {
                    (pad32 + live0 as u32 - extent as u32) & !31
                } else {
                    0
                };
                let wrap_t = pad32 + (view as u32 - arc_start as u32);
                let hi_nw = if wrap_t >= FW_BKV as u32 { wrap_t & !31 } else { 0 };
                let hi = (if hi_rel < hi_nw { hi_rel } else { hi_nw }) & !63;
                let (t_is, t_ie) = if m_cnt == FW_PREFILL_BQ_POS && threads == 384 && lo == 0 && hi >= 64 {
                    (0u32, hi)
                } else {
                    (u32::MAX, 0u32)
                };
                store_u32(c_base + 56, t_is);
                store_u32(c_base + 60, t_ie);
            }
        }

        // Load Q and M (with optional inline `FW_CLIP_IN` HeadClip, bit-exact with `fast_norm` mode 2).
        if (*w.add(FW_CLIP_WORD) & FW_CLIP_IN) != 0 {
            let (query_scale, moved_scale, position) = (*w.add(17) as *const f32, *w.add(18) as *const f32, *w.add(19) as *const f32);
            let (query_bound, moved_bound) = (f32::from_bits(*w.add(20) as u32), f32::from_bits(*w.add(21) as u32));
            let has_qs = !query_scale.is_null();
            let has_ms = !moved_scale.is_null();
            let has_pos = !position.is_null();
            // Hoist per-head `query_scale` (4 floats/lane) and `moved_scale` (4 floats/lane for `lane & 3`), plus
            // this warp's 16 `position` factors once before the 16-row loop (4x fewer scale reads, coalesced v4 loads).
            let qs4 = if !has_qs {
                [1.0f32; 4]
            } else if (query_scale as usize) & 15 == 0 {
                ld_global_nc_v4(query_scale.add(head * FW_HEAD_DIM + lane * 4))
            } else {
                let p = query_scale.add(head * FW_HEAD_DIM + lane * 4);
                [*p, *p.add(1), *p.add(2), *p.add(3)]
            };
            let pos_lane = if has_pos && (w_pos * 16 + (lane & 15)) < m_cnt {
                *position.add(q0 + w_pos * 16 + (lane & 15))
            } else {
                0.0f32
            };
            let (ql, bl) = (lane >> 3, lane & 7);
            let q_st_off = swiz128(0, lane >> 1) + ((lane & 1) * 8) as u32;
            macro_rules! clip_q_row {
                ($r:expr, $x4:expr, $fac:expr) => {{
                    let r: usize = $r;
                    let qr = q_row_base + r;
                    let dst = q_base + (qr * 256) as u32 + (q_st_off ^ (((qr & 7) * 16) as u32));
                    if r < m_cnt {
                        let x4: [f32; 4] = $x4;
                        let (mut v0, mut v1) = rn2(x4[0], x4[1]);
                        let (mut v2, mut v3) = rn2(x4[2], x4[3]);
                        if has_qs {
                            v0 *= qs4[0]; v1 *= qs4[1]; v2 *= qs4[2]; v3 *= qs4[3];
                        }
                        let (a0, a1, a2, a3) = (v0 * v0, v1 * v1, v2 * v2, v3 * v3);
                        // 4-shuffle exact 4x4 transpose across `ql = lane >> 3` so lane `(u, b)` holds
                        // `[sq[4b+u], sq[32+4b+u], sq[64+4b+u], sq[96+4b+u]]` in exact rising index order.
                        let p0 = shfl_xor_f32(if (ql & 1) == 0 { a1 } else { a0 }, 8);
                        let (e0, e1) = if (ql & 1) == 0 { (a0, p0) } else { (p0, a1) };
                        let p1 = shfl_xor_f32(if (ql & 1) == 0 { a3 } else { a2 }, 8);
                        let (e2, e3) = if (ql & 1) == 0 { (a2, p1) } else { (p1, a3) };
                        let s0 = shfl_xor_f32(if (ql & 2) == 0 { e2 } else { e0 }, 16);
                        let s1 = shfl_xor_f32(if (ql & 2) == 0 { e3 } else { e1 }, 16);
                        let (t0, t1, t2, t3) = if (ql & 2) == 0 { (e0, e1, s0, s1) } else { (s0, s1, e2, e3) };
                        let mut sum = 0.0f32;
                        sum += t0; sum += t1; sum += t2; sum += t3;
                        // Exact `block_ops::join` order on `c = 4*b + u`: xor 16, 8, 4, 2, 1.
                        sum += shfl_xor_f32(sum, 4);
                        sum += shfl_xor_f32(sum, 2);
                        sum += shfl_xor_f32(sum, 1);
                        sum += shfl_xor_f32(sum, 16);
                        sum += shfl_xor_f32(sum, 8);
                        let qscale = crate::block_attention::clip_factor(crate::block_ops::root(sum), query_bound);
                        let (pw0, pw1) = if !has_pos {
                            (cvt_bf16x2(v0 * qscale, v1 * qscale), cvt_bf16x2(v2 * qscale, v3 * qscale))
                        } else {
                            let fac: f32 = $fac;
                            let (c0, c1) = rn2(v0 * qscale, v1 * qscale);
                            let (c2, c3) = rn2(v2 * qscale, v3 * qscale);
                            (cvt_bf16x2(c0 * fac, c1 * fac), cvt_bf16x2(c2 * fac, c3 * fac))
                        };
                        core::arch::asm!(
                            "st.shared.v2.b32 [{dst}], {{{a}, {b}}};",
                            dst = in(reg32) dst, a = in(reg32) pw0, b = in(reg32) pw1, options(nostack)
                        );
                    } else {
                        core::arch::asm!(
                            "st.shared.v2.b32 [{dst}], {{0, 0}};",
                            dst = in(reg32) dst, options(nostack)
                        );
                    }
                }};
            }
            // Process 16 rows in 4 batches of 4 rows: all 4 coalesced 128B `query` row loads in flight together.
            let mut batch = 0usize;
            while batch < 4 {
                let r0 = w_pos * 16 + batch * 4;
                // `FW_QG_HALF`: the query holds the pair rounding of each product word, 16 bits a value. Each value
                // is the word that `rn2` in `clip_q_row` makes of the product word, and `rn2` keeps such a word.
                let ld_q = |ri: usize| {
                    if ri >= m_cnt {
                        [0.0f32; 4]
                    } else if (*w.add(FW_CLIP_WORD) & FW_QG_HALF) != 0 {
                        let (a, b): (u32, u32);
                        core::arch::asm!("ld.global.nc.v2.u32 {{{a}, {b}}}, [{p}];", a = out(reg32) a, b = out(reg32) b,
                            p = in(reg64) query.cast::<u32>().add(((q0 + ri) * heads + head) * (FW_HEAD_DIM / 2) + lane * 2), options(readonly, nostack));
                        [f32::from_bits(a << 16), f32::from_bits(a & 0xffff_0000), f32::from_bits(b << 16), f32::from_bits(b & 0xffff_0000)]
                    } else {
                        ld_global_nc_v4(query.add(((q0 + ri) * heads + head) * FW_HEAD_DIM + lane * 4))
                    }
                };
                let (x0, x1, x2, x3) = (ld_q(r0), ld_q(r0 + 1), ld_q(r0 + 2), ld_q(r0 + 3));
                macro_rules! bcast_pos {
                    ($idx:expr) => {{
                        let bits: u32;
                        core::arch::asm!(
                            "shfl.sync.idx.b32 {got}, {v}, {src}, 31, 0xffffffff;",
                            got = out(reg32) bits, v = in(reg32) pos_lane.to_bits(), src = in(reg32) ($idx) as u32,
                            options(nomem, nostack)
                        );
                        f32::from_bits(bits)
                    }};
                }
                let f0 = if has_pos { bcast_pos!(batch * 4) } else { 0.0f32 };
                let f1 = if has_pos { bcast_pos!(batch * 4 + 1) } else { 0.0f32 };
                let f2 = if has_pos { bcast_pos!(batch * 4 + 2) } else { 0.0f32 };
                let f3 = if has_pos { bcast_pos!(batch * 4 + 3) } else { 0.0f32 };
                clip_q_row!(r0, x0, f0);
                clip_q_row!(r0 + 1, x1, f1);
                clip_q_row!(r0 + 2, x2, f2);
                clip_q_row!(r0 + 3, x3, f3);
                batch += 1;
            }
            // Clip `moved` (`16` rows x `16` floats) in 2 batches of 8 rows (`4` lanes/row, 1 `ld_global_nc_v4`/lane).
            let (m_grp, m_c) = (lane >> 2, lane & 3);
            let ms4 = if !has_ms {
                [1.0f32; 4]
            } else if (moved_scale as usize) & 15 == 0 {
                ld_global_nc_v4(moved_scale.add(head * FW_REL_DIM + m_c * 4))
            } else {
                let p = moved_scale.add(head * FW_REL_DIM + m_c * 4);
                [*p, *p.add(1), *p.add(2), *p.add(3)]
            };
            let r_m0 = w_pos * 16 + m_grp;
            let r_m1 = r_m0 + 8;
            let m_in0 = if r_m0 < m_cnt { ld_global_nc_v4(moved.add(((q0 + r_m0) * heads + head) * FW_REL_DIM + m_c * 4)) } else { [0.0f32; 4] };
            let m_in1 = if r_m1 < m_cnt { ld_global_nc_v4(moved.add(((q0 + r_m1) * heads + head) * FW_REL_DIM + m_c * 4)) } else { [0.0f32; 4] };
            macro_rules! clip_m_row {
                ($r:expr, $x4:expr, $pos_src:expr) => {{
                    let r: usize = $r;
                    let qr = q_row_base + r;
                    let fac = if has_pos {
                        let bits: u32;
                        core::arch::asm!(
                            "shfl.sync.idx.b32 {got}, {v}, {src}, 31, 0xffffffff;",
                            got = out(reg32) bits, v = in(reg32) pos_lane.to_bits(), src = in(reg32) ($pos_src) as u32,
                            options(nomem, nostack)
                        );
                        f32::from_bits(bits)
                    } else { 0.0f32 };
                    let active = r < m_cnt;
                    let x4: [f32; 4] = if active { $x4 } else { [0.0f32; 4] };
                    let (mut v0, mut v1) = rn2(x4[0], x4[1]);
                    let (mut v2, mut v3) = rn2(x4[2], x4[3]);
                    if has_ms { v0 *= ms4[0]; v1 *= ms4[1]; v2 *= ms4[2]; v3 *= ms4[3]; }
                    // Exact 16-element `block_ops::join` across 4 lanes (`m_c = 0..3`), warp-uniform (all 32 lanes
                    // execute every `shfl_xor_f32` even when `m_cnt` is not a multiple of 8):
                    // xor 8 (`m_c ^ 2`), xor 4 (`m_c ^ 1`), then xor 2 and xor 1 within each lane's 4 floats.
                    let mut s0 = v0 * v0; let mut s1 = v1 * v1; let mut s2 = v2 * v2; let mut s3 = v3 * v3;
                    s0 += shfl_xor_f32(s0, 2); s1 += shfl_xor_f32(s1, 2);
                    s2 += shfl_xor_f32(s2, 2); s3 += shfl_xor_f32(s3, 2);
                    s0 += shfl_xor_f32(s0, 1); s1 += shfl_xor_f32(s1, 1);
                    s2 += shfl_xor_f32(s2, 1); s3 += shfl_xor_f32(s3, 1);
                    let (pw0, pw1) = if active {
                        let sum = (s0 + s2) + (s1 + s3);
                        let mscale = crate::block_attention::clip_factor(crate::block_ops::root(sum), moved_bound);
                        if !has_pos {
                            (cvt_bf16x2(v0 * mscale, v1 * mscale), cvt_bf16x2(v2 * mscale, v3 * mscale))
                        } else {
                            let (c0, c1) = rn2(v0 * mscale, v1 * mscale);
                            let (c2, c3) = rn2(v2 * mscale, v3 * mscale);
                            (cvt_bf16x2(c0 * fac, c1 * fac), cvt_bf16x2(c2 * fac, c3 * fac))
                        }
                    } else { (0u32, 0u32) };
                    let m_dst = m_base + swiz_m8(qr, m_c);
                    core::arch::asm!(
                        "st.shared.v2.b32 [{dst}], {{{a}, {b}}};",
                        dst = in(reg32) m_dst, a = in(reg32) pw0, b = in(reg32) pw1, options(nostack)
                    );
                }};
            }
            clip_m_row!(r_m0, m_in0, m_grp);
            clip_m_row!(r_m1, m_in1, m_grp + 8);
            let _ = bl;
        } else {
            if (*w.add(FW_CLIP_WORD) & FW_QG_HALF) != 0 { core::arch::nvptx::trap(); }
            let mut idx = tid;
            let total_q = q_rows * 16;
            while idx < total_q {
                let qr = idx >> 4;
                let ch = idx & 15;
                let hl = qr >> 6;
                let r = qr & 63;
                let dst = q_base + swiz128(qr, ch);
                if r < m_cnt {
                    let p = query.add(((q0 + r) * heads + head_base + hl) * FW_HEAD_DIM + ch * 8);
                    let v0 = ld_global_nc_v4(p);
                    let v1 = ld_global_nc_v4(p.add(4));
                    st_shared_v4(dst, [
                        cvt_bf16x2(v0[0], v0[1]),
                        cvt_bf16x2(v0[2], v0[3]),
                        cvt_bf16x2(v1[0], v1[1]),
                        cvt_bf16x2(v1[2], v1[3]),
                    ]);
                } else {
                    st_shared_zero16(dst);
                }
                idx += threads;
            }
            let mut midx = tid;
            let total_m = q_rows * FW_REL_DIM;
            while midx < total_m {
                let qr = midx >> 4;
                let j = midx & 15;
                let hl = qr >> 6;
                let r = qr & 63;
                let bits = if r < m_cnt {
                    rn(*moved.add(((q0 + r) * heads + head_base + hl) * FW_REL_DIM + j)).to_bits() >> 16
                } else { 0 };
                store_u16(m_base + swiz_m8(qr, j >> 2) + ((j & 3) * 2) as u32, bits as u16);
                midx += threads;
            }
        }

        macro_rules! issue_kvt {
            ($st:expr, $tb:expr, $view:expr, $ext:expr, $pad:expr, $asrt:expr, $aend:expr, $l0:expr, $kvs:expr, $kvh:expr, $thr:expr) => {{
                let st: u32 = $st;
                let tb: u32 = $tb;
                let view_c: u32 = $view;
                let ext_c: u32 = $ext;
                let pad_c: u32 = $pad;
                let asrt_c: u32 = $asrt;
                let aend_c: u32 = $aend;
                let l0_c: u32 = $l0;
                let kvs_c: u32 = $kvs;
                let kvh_off: usize = ($kvh as usize) * 64;
                let thr_c: u32 = $thr;
                let tid_c = core::arch::nvptx::_thread_idx_x() as u32;
                let keys_h = (*w.add(1) as *const u32).add(kvh_off);
                let values_h = (*w.add(2) as *const u32).add(kvh_off);
                let k_dst = smem + K_OFF + st * K_STAGE_BYTES as u32;
                let v_dst = smem + V_OFF + st * V_STAGE_BYTES as u32;
                let mut idx = tid_c;
                if tb >= pad_c && tb + FW_BKV as u32 <= aend_c {
                    let p0 = { let x = asrt_c + tb - pad_c; if x >= view_c { x - view_c } else { x } };
                    if p0 + FW_BKV as u32 <= view_c {
                        while idx < (FW_BKV * 16) as u32 {
                            let r = idx >> 4;
                            let ch = idx & 15;
                            let sw = swiz128(r as usize, ch as usize);
                            let off = (p0 + r) as usize * kvs_c as usize + (ch * 4) as usize;
                            crate::ptx::copy16(k_dst + sw, keys_h.add(off));
                            crate::ptx::copy16(v_dst + sw, values_h.add(off));
                            idx += thr_c;
                        }
                    } else {
                        while idx < (FW_BKV * 16) as u32 {
                            let r = idx >> 4;
                            let ch = idx & 15;
                            let sw = swiz128(r as usize, ch as usize);
                            let mut phys = p0 + r;
                            if phys >= view_c { phys -= view_c; }
                            let off = phys as usize * kvs_c as usize + (ch * 4) as usize;
                            crate::ptx::copy16(k_dst + sw, keys_h.add(off));
                            crate::ptx::copy16(v_dst + sw, values_h.add(off));
                            idx += thr_c;
                        }
                    }
                } else {
                    while idx < (FW_BKV * 16) as u32 {
                        let r = idx >> 4;
                        let ch = idx & 15;
                        let sw = swiz128(r as usize, ch as usize);
                        let col = tb + r;
                        if col >= pad_c && col < aend_c {
                            let mut phys = asrt_c + col - pad_c;
                            if phys >= view_c { phys -= view_c; }
                            let off = phys as usize * kvs_c as usize + (ch * 4) as usize;
                            crate::ptx::copy16(k_dst + sw, keys_h.add(off));
                            crate::ptx::copy16(v_dst + sw, values_h.add(off));
                        } else {
                            st_shared_zero16(k_dst + sw);
                            st_shared_zero16(v_dst + sw);
                        }
                        idx += thr_c;
                    }
                }
                if tb + FW_BKV as u32 + ext_c > pad_c + l0_c {
                    let rel_p = *w.add(4) as *const u32;
                    let t_dst_stage = smem + T_OFF + st * T_STAGE_BYTES as u32;
                    let first = (pad_c + l0_c - 1) as i32 - (tb + FW_BKV as u32 - 1) as i32;
                    // Coalesced, bank-conflict-free staging of the `4` term planes x `95` slots (`12` warp-chunks of
                    // `32` consecutive distances each): every warp reads 4 coalesced 128B cache lines and executes
                    // one bank-free `st.shared.v4.b32` per lane (0 excessive global sectors, 0 shared bank conflicts).
                    let warps_c = thr_c >> 5;
                    let mut wc = tid_c >> 5;
                    let ln = tid_c & 31;
                    while wc < 12 {
                        let q = wc / 3;
                        let chunk = wc - q * 3;
                        let slot = chunk * 32 + ln;
                        if slot < T_SLOTS as u32 {
                            let d = first + slot as i32;
                            let dst = t_dst_stage + q * T_PLANE_BYTES as u32 + slot * 16;
                            if d >= 0 && (d as u32) < ext_c {
                                let base_idx = ((q * 4) * ext_c + d as u32) as usize;
                                let ext_u = ext_c as usize;
                                let v0 = ld_global_nc_u32(rel_p.add(base_idx));
                                let v1 = ld_global_nc_u32(rel_p.add(base_idx + ext_u));
                                let v2 = ld_global_nc_u32(rel_p.add(base_idx + 2 * ext_u));
                                let v3 = ld_global_nc_u32(rel_p.add(base_idx + 3 * ext_u));
                                st_shared_v4(dst, [v0, v1, v2, v3]);
                            } else {
                                st_shared_zero16(dst);
                            }
                        }
                        wc += warps_c;
                    }
                }
            }};
        }

        // Step 1 of prologue: sync so Q_TEMP [0..49,152), M, L, C are visible; load Q dims 0..96 once into 24 regs
        // (`q_reg0..5`) and copy dims 96..128 into permanent bank-free `Q_HI` (`12,288 B`); sync so [0..65,536) is free for K/V.
        core::arch::nvptx::_syncthreads();
        let mut qhi_idx = tid;
        let total_qhi = q_rows * 4;
        while qhi_idx < total_qhi {
            let qr = qhi_idx >> 2;
            let ch = qhi_idx & 3;
            let v = crate::ptx::load4(q_base + swiz128(qr, 12 + ch));
            st_shared_v4(smem + Q_HI_OFF + swiz64(qr, ch), v);
            qhi_idx += threads;
        }
        let q_ld_row_p = (h_local * 64 + w_pos * 16 + (lane & 15)) as u32;
        let q_cur_p = smem + Q_OFF + q_ld_row_p * 256;
        let qvs_p = (((lane & 7) ^ (lane >> 4)) * 16) as u32;
        let q_reg0 = ldmatrix_x4(q_cur_p + qvs_p);
        let q_reg1 = ldmatrix_x4(q_cur_p + (32 ^ qvs_p));
        let q_reg2 = ldmatrix_x4(q_cur_p + (64 ^ qvs_p));
        let q_reg3 = ldmatrix_x4(q_cur_p + (96 ^ qvs_p));
        let q_reg4 = ldmatrix_x4(q_cur_p + (128 ^ qvs_p));
        let q_reg5 = ldmatrix_x4(q_cur_p + (160 ^ qvs_p));
        core::arch::nvptx::_syncthreads();

        // Step 2 of prologue: stage initial K/V (64 keys for fast64 interior path, else 32 keys + T for general path).
        let fast64_init = load_u32(c_base + 56) == 0;
        if fast64_init {
            let tid_p = tid as u32;
            let r0_p = tid_p >> 4;
            let ch0_p = tid_p & 15;
            let sw0_p = swiz128(r0_p as usize, ch0_p as usize);
            let kvs_p = kv_heads * 64;
            let off0_p = (arc_start + r0_p as usize) * kvs_p + kv_head * 64 + (ch0_p as usize) * 4;
            let kp_p = (*w.add(1) as *const u32).add(off0_p);
            let vp_p = (*w.add(2) as *const u32).add(off0_p);
            let k0_dst = smem + K_OFF;
            let v0_dst = smem + V_OFF;
            crate::ptx::copy16_stream(k0_dst + sw0_p, kp_p);
            crate::ptx::copy16_stream(v0_dst + sw0_p, vp_p);
            crate::ptx::copy16_stream(k0_dst + (K_BYTES as u32 + sw0_p), kp_p.add(32 * kvs_p));
            crate::ptx::copy16_stream(v0_dst + (V_BYTES as u32 + sw0_p), vp_p.add(32 * kvs_p));
            if tid_p < 128 {
                crate::ptx::copy16_stream(k0_dst + (sw0_p + 6144), kp_p.add(24 * kvs_p));
                crate::ptx::copy16_stream(v0_dst + (sw0_p + 6144), vp_p.add(24 * kvs_p));
                crate::ptx::copy16_stream(k0_dst + (K_BYTES as u32 + sw0_p + 6144), kp_p.add(56 * kvs_p));
                crate::ptx::copy16_stream(v0_dst + (V_BYTES as u32 + sw0_p + 6144), vp_p.add(56 * kvs_p));
            }
        } else {
            issue_kvt!(
                0, 0, view as u32, extent as u32, pad as u32, arc_start as u32, arc_end as u32,
                live0 as u32, (kv_heads * 64) as u32, kv_head as u32, threads as u32
            );
        }
        crate::ptx::commit();
        cp_async_wait_zero();
        core::arch::nvptx::_syncthreads();

        // Fused `ldmatrix.x4` (prefetch next) + 2x `mma.m16n8k16` (consume current) in one asm block so ptxas
        // cannot sink `LDSM` next to its consumer `HMMA` (eliminates 25-cycle LDSM->HMMA short_scoreboard stalls!).
        macro_rules! ldm_and_mma2 {
            ($ptr:expr, $qa:expr, $kb:expr, $sa:ident, $sb:ident) => {{
                let (n0, n1, n2, n3): (u32, u32, u32, u32);
                core::arch::asm!(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{n0}, {n1}, {n2}, {n3}}}, [{p}];\n\
                     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{a0},{a1},{a2},{a3}}}, {{{q0},{q1},{q2},{q3}}}, {{{b0},{b1}}}, {{{a0},{a1},{a2},{a3}}};\n\
                     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{c0},{c1},{c2},{c3}}}, {{{q0},{q1},{q2},{q3}}}, {{{b2},{b3}}}, {{{c0},{c1},{c2},{c3}}};",
                    n0 = out(reg32) n0, n1 = out(reg32) n1, n2 = out(reg32) n2, n3 = out(reg32) n3,
                    p = in(reg32) ($ptr),
                    q0 = in(reg32) $qa[0], q1 = in(reg32) $qa[1], q2 = in(reg32) $qa[2], q3 = in(reg32) $qa[3],
                    b0 = in(reg32) $kb[0], b1 = in(reg32) $kb[1], b2 = in(reg32) $kb[2], b3 = in(reg32) $kb[3],
                    a0 = inout(reg32) $sa[0], a1 = inout(reg32) $sa[1], a2 = inout(reg32) $sa[2], a3 = inout(reg32) $sa[3],
                    c0 = inout(reg32) $sb[0], c1 = inout(reg32) $sb[1], c2 = inout(reg32) $sb[2], c3 = inout(reg32) $sb[3],
                    options(nostack),
                );
                [n0, n1, n2, n3]
            }};
        }
        macro_rules! ldm_trans_and_mma2 {
            ($ptr:expr, $pa:expr, $vb:expr, $oa:ident, $ob:ident) => {{
                let (n0, n1, n2, n3): (u32, u32, u32, u32);
                core::arch::asm!(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{n0}, {n1}, {n2}, {n3}}}, [{p}];\n\
                     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{a0},{a1},{a2},{a3}}}, {{{p0},{p1},{p2},{p3}}}, {{{b0},{b1}}}, {{{a0},{a1},{a2},{a3}}};\n\
                     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{c0},{c1},{c2},{c3}}}, {{{p0},{p1},{p2},{p3}}}, {{{b2},{b3}}}, {{{c0},{c1},{c2},{c3}}};",
                    n0 = out(reg32) n0, n1 = out(reg32) n1, n2 = out(reg32) n2, n3 = out(reg32) n3,
                    p = in(reg32) ($ptr),
                    p0 = in(reg32) $pa[0], p1 = in(reg32) $pa[1], p2 = in(reg32) $pa[2], p3 = in(reg32) $pa[3],
                    b0 = in(reg32) $vb[0], b1 = in(reg32) $vb[1], b2 = in(reg32) $vb[2], b3 = in(reg32) $vb[3],
                    a0 = inout(reg32) $oa[0], a1 = inout(reg32) $oa[1], a2 = inout(reg32) $oa[2], a3 = inout(reg32) $oa[3],
                    c0 = inout(reg32) $ob[0], c1 = inout(reg32) $ob[1], c2 = inout(reg32) $ob[2], c3 = inout(reg32) $ob[3],
                    options(nostack),
                );
                [n0, n1, n2, n3]
            }};
        }

        // Load per-row sink initializers (m_r0, m_r1) and zero output accumulators.
        let arc_end_loop = load_u32(c_base + 16);
        let m_cnt_init = load_u32(c_base + 28);
        let w_min_row_init = (w_pos * 16) as u32;
        let r0_init = w_min_row_init + (g as u32);
        let mut m_r0 = if r0_init < m_cnt_init { rn(*sink.add((q0 + r0_init as usize) * heads + head)) } else { 0.0f32 };
        let mut m_r1 = if r0_init + 8 < m_cnt_init { rn(*sink.add((q0 + r0_init as usize + 8) * heads + head)) } else { 0.0f32 };
        let mut l_r0 = 1.0f32;
        let mut l_r1 = 1.0f32;

        // 16 output MMA accumulators (`16 x 128` per warp) strictly in registers (0 stack, 0 spill).
        let (mut o0, mut o1, mut o2, mut o3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o4, mut o5, mut o6, mut o7) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o8, mut o9, mut o10, mut o11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o12, mut o13, mut o14, mut o15) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        let mut stage = 0u32;
        let mut t_base = 0u32;

        macro_rules! fmax_hw {
            ($a:expr, $b:expr) => {{
                let d: f32;
                core::arch::asm!(
                    "max.f32 {d}, {a}, {b};",
                    d = out(reg32) d, a = in(reg32) ($a as f32), b = in(reg32) ($b as f32),
                    options(pure, nomem, nostack)
                );
                d
            }};
        }
        macro_rules! sc_fast {
            ($s:ident) => {{
                let (a0, a1) = rn2($s[0], $s[1]); let (a2, a3) = rn2($s[2], $s[3]);
                let (b0, b1) = rn2(a0 * scale, a1 * scale); let (b2, b3) = rn2(a2 * scale, a3 * scale);
                $s = [b0, b1, b2, b3];
            }};
        }

        // Fast 2-Stage 64-Key Pure-Interior No-Bias Ping-Pong Loop (`Q` 0..96 in regs, 1 barrier/64 keys).
        if load_u32(smem + C_OFF + 56) == 0 {
            macro_rules! qk_sub {
                ($k_sub:expr) => {{
                    let mut tid_k = core::arch::nvptx::_thread_idx_x() as u32;
                    core::arch::asm!("/* {0} */", inout(reg32) tid_k, options(nomem, nostack));
                    let lane_k = tid_k & 31;
                    let k_ld_row0 = ((lane_k >> 4) * 8) + (lane_k & 7);
                    let mut k_cur = smem + K_OFF + stage * K_STAGE_BYTES as u32 + ($k_sub) + k_ld_row0 * 256;
                    let mut ks_m = ((k_ld_row0 & 7) ^ ((lane_k >> 3) & 1)) * 16;
                    core::arch::asm!("/* {0} {1} */", inout(reg32) k_cur, inout(reg32) ks_m, options(nomem, nostack));
                    let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                    let mut kb0 = ldmatrix_x4(k_cur + ks_m);
                    let mut kb1 = ldmatrix_x4(k_cur + 4096 + ks_m);
                    macro_rules! qkf {
                        ($ks:expr, $qr:ident) => {{
                            let no = ((($ks + 1) * 32) as u32) ^ ks_m;
                            kb0 = ldm_and_mma2!(k_cur + no, $qr, kb0, s0, s1);
                            kb1 = ldm_and_mma2!(k_cur + 4096 + no, $qr, kb1, s2, s3);
                        }};
                    }
                    qkf!(0, q_reg0); qkf!(1, q_reg1); qkf!(2, q_reg2);
                    qkf!(3, q_reg3); qkf!(4, q_reg4); qkf!(5, q_reg5);
                    let qhi_row = ((tid_k >> 7) * 64) + (((tid_k >> 5) & 3) * 16) + (tid_k & 15);
                    let qhi_base = smem + Q_HI_OFF + qhi_row * 64;
                    let qhi_sw = ((lane_k >> 4) ^ ((qhi_row >> 1) & 3)) * 16;
                    let mut qhi = ldmatrix_x4(qhi_base + qhi_sw);
                    qkf!(6, qhi);
                    qhi = ldmatrix_x4(qhi_base + (32 ^ qhi_sw));
                    s0 = mma_m16n8k16_bf16(qhi, [kb0[0], kb0[1]], s0);
                    s1 = mma_m16n8k16_bf16(qhi, [kb0[2], kb0[3]], s1);
                    s2 = mma_m16n8k16_bf16(qhi, [kb1[0], kb1[1]], s2);
                    s3 = mma_m16n8k16_bf16(qhi, [kb1[2], kb1[3]], s3);
                    (s0, s1, s2, s3)
                }};
            }
            macro_rules! pv0_exp_sub {
                ($v_cur:expr, $qvs:expr, $pa0:expr, $s2:expr, $s3:expr, $s0a:expr, $s1a:expr, $chg:expr, $a0:expr, $a1:expr) => {{
                    macro_rules! pe {
                        ($pair:expr, $oa:ident, $ob:ident, $sv:expr, $mr:expr) => {{
                            let vb0 = ldmatrix_x4_trans($v_cur + ((($pair * 32) as u32) ^ $qvs));
                            let pv = exp_nonpos($sv - $mr);
                            $oa = mma_m16n8k16_bf16($pa0, [vb0[0], vb0[1]], $oa);
                            $ob = mma_m16n8k16_bf16($pa0, [vb0[2], vb0[3]], $ob);
                            pv
                        }};
                    }
                    let p20 = pe!(0, o0, o1, $s2[0], m_r0); let p21 = pe!(1, o2, o3, $s2[1], m_r0);
                    let pa1_0 = cvt_bf16x2(p20, p21); let s0_b0 = p20 + p21;
                    let p22 = pe!(2, o4, o5, $s2[2], m_r1); let p23 = pe!(3, o6, o7, $s2[3], m_r1);
                    let pa1_1 = cvt_bf16x2(p22, p23); let s1_b0 = p22 + p23;
                    let p30 = pe!(4, o8, o9, $s3[0], m_r0); let p31 = pe!(5, o10, o11, $s3[1], m_r0);
                    let pa1_2 = cvt_bf16x2(p30, p31); let s0_b1 = p30 + p31;
                    let p32 = pe!(6, o12, o13, $s3[2], m_r1); let p33 = pe!(7, o14, o15, $s3[3], m_r1);
                    let pa1_3 = cvt_bf16x2(p32, p33); let s1_b1 = p32 + p33;
                    let mut sum0 = $s0a + (s0_b0 + s0_b1);
                    sum0 += shfl_xor_f32(sum0, 1); sum0 += shfl_xor_f32(sum0, 2);
                    let mut sum1 = $s1a + (s1_b0 + s1_b1);
                    sum1 += shfl_xor_f32(sum1, 1); sum1 += shfl_xor_f32(sum1, 2);
                    if $chg { l_r0 = l_r0 * $a0 + sum0; l_r1 = l_r1 * $a1 + sum1; } else { l_r0 += sum0; l_r1 += sum1; }
                    [pa1_0, pa1_1, pa1_2, pa1_3]
                }};
            }
            macro_rules! copy32_int {
                ($nst:expr, $sub_off:expr, $col_base:expr) => {{
                    let cb_c = smem + C_OFF;
                    let mut tc = core::arch::nvptx::_thread_idx_x() as u32;
                    core::arch::asm!("/* {0} */", inout(reg32) tc, options(nomem, nostack));
                    let kvs_c = load_u32(cb_c + 24) as usize;
                    let r0_c = tc >> 4;
                    let ch0_c = tc & 15;
                    let sw = (r0_c * 256) + ((ch0_c ^ (r0_c & 7)) * 16);
                    let p_row = (load_u32(cb_c + 12) + ($col_base) + r0_c) as usize;
                    let off = p_row * kvs_c + (load_u32(cb_c + 44) as usize) * 64 + (ch0_c as usize) * 4;
                    let kp = (*w.add(1) as *const u32).add(off);
                    let vp = (*w.add(2) as *const u32).add(off);
                    let k_dst = smem + K_OFF + ($nst) * K_STAGE_BYTES as u32 + ($sub_off);
                    let v_dst = smem + V_OFF + ($nst) * V_STAGE_BYTES as u32 + ($sub_off);
                    crate::ptx::copy16_stream(k_dst + sw, kp);
                    crate::ptx::copy16_stream(v_dst + sw, vp);
                    if tc < 128 {
                        let row24 = 24 * kvs_c;
                        crate::ptx::copy16_stream(k_dst + (sw + 6144), kp.add(row24));
                        crate::ptx::copy16_stream(v_dst + (sw + 6144), vp.add(row24));
                    }
                }};
            }

            // Ping-Pong 64-key loop (`t_base += 64`): defers Sub-tile 0's `pa1 * V[16..32]` (`16` HMMAs) and
            // interleaves it 1-for-1 with Sub-tile 1's `sc_fast + fmax + ballot + p00..p13` (zero TC bubble!).
            while t_base < load_u32(smem + C_OFF + 60) {
                let t_int_end = load_u32(smem + C_OFF + 60);
                let next_stage = stage ^ 1;
                let next64 = t_base + 64;
                if next64 < t_int_end {
                    copy32_int!(next_stage, 0, next64);
                }

                // === Sub-tile 0 (keys 0..32 of `stage`) ===
                let (mut s0, mut s1, mut s2, mut s3) = qk_sub!(0);
                sc_fast!(s0); sc_fast!(s1); sc_fast!(s2); sc_fast!(s3);
                let mut mt0 = fmax_hw!(fmax_hw!(fmax_hw!(s0[0], s0[1]), fmax_hw!(s1[0], s1[1])), fmax_hw!(fmax_hw!(s2[0], s2[1]), fmax_hw!(s3[0], s3[1])));
                mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 1)); mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 2));
                let mut mt1 = fmax_hw!(fmax_hw!(fmax_hw!(s0[2], s0[3]), fmax_hw!(s1[2], s1[3])), fmax_hw!(fmax_hw!(s2[2], s2[3]), fmax_hw!(s3[2], s3[3])));
                mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 1)); mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 2));
                let chg0 = crate::ptx::ballot(mt0 > m_r0 || mt1 > m_r1) != 0;
                let (a0_0, a1_0) = if chg0 {
                    let mn0 = fmax_hw!(m_r0, mt0); let mn1 = fmax_hw!(m_r1, mt1);
                    let a0 = if mn0 > m_r0 { exp_nonpos(m_r0 - mn0) } else { 1.0f32 };
                    let a1 = if mn1 > m_r1 { exp_nonpos(m_r1 - mn1) } else { 1.0f32 };
                    m_r0 = mn0; m_r1 = mn1;
                    if a0 != 1.0f32 { macro_rules! sc { ($($o:ident)*) => { $($o[0] *= a0; $o[1] *= a0;)* }; } sc!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                    if a1 != 1.0f32 { macro_rules! sc { ($($o:ident)*) => { $($o[2] *= a1; $o[3] *= a1;)* }; } sc!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                    (a0, a1)
                } else { (1.0f32, 1.0f32) };
                let p00 = exp_nonpos(s0[0] - m_r0); let p01 = exp_nonpos(s0[1] - m_r0);
                let p02 = exp_nonpos(s0[2] - m_r1); let p03 = exp_nonpos(s0[3] - m_r1);
                let p10 = exp_nonpos(s1[0] - m_r0); let p11 = exp_nonpos(s1[1] - m_r0);
                let p12 = exp_nonpos(s1[2] - m_r1); let p13 = exp_nonpos(s1[3] - m_r1);
                let pa0_0 = [cvt_bf16x2(p00, p01), cvt_bf16x2(p02, p03), cvt_bf16x2(p10, p11), cvt_bf16x2(p12, p13)];
                let mut tid_v0 = core::arch::nvptx::_thread_idx_x() as u32;
                core::arch::asm!("/* {0} */", inout(reg32) tid_v0, options(nomem, nostack));
                let lv0 = tid_v0 & 31;
                let mut qvs0 = ((lv0 & 7) ^ (lv0 >> 4)) * 16;
                let mut v_cur0 = smem + V_OFF + stage * V_STAGE_BYTES as u32 + (lv0 & 15) * 256;
                core::arch::asm!("/* {0} {1} */", inout(reg32) v_cur0, inout(reg32) qvs0, options(nomem, nostack));
                let pa1_0 = pv0_exp_sub!(v_cur0, qvs0, pa0_0, s2, s3, (p00 + p01) + (p10 + p11), (p02 + p03) + (p12 + p13), chg0, a0_0, a1_0);
                // Stash `pa1_0` (6,144 B <= 6,400 B) in dead `T[stage]` across `qk_sub!(K_BYTES)` (0 stack, 0 spill).
                st_shared_v4(smem + T_OFF + stage * T_STAGE_BYTES as u32 + tid_v0 * 16, pa1_0);

                // Issue second 32 keys of `next64` + commit before Sub-tile 1.
                if next64 < t_int_end {
                    copy32_int!(next_stage, K_BYTES as u32, next64 + 32);
                    crate::ptx::commit();
                }

                // === Sub-tile 1 (keys 32..64 of `stage`) ===
                let (mut s0, mut s1, mut s2, mut s3) = qk_sub!(K_BYTES as u32);

                // Ping-Pong: interleave Sub-tile 0's `pa1_0 * V[16..32]` with Sub-tile 1's `sc_fast + fmax + p00..p13`.
                let mut tid_v1 = core::arch::nvptx::_thread_idx_x() as u32;
                core::arch::asm!("/* {0} */", inout(reg32) tid_v1, options(nomem, nostack));
                let pa1_0 = crate::ptx::load4(smem + T_OFF + stage * T_STAGE_BYTES as u32 + tid_v1 * 16);
                let lv1 = tid_v1 & 31;
                let mut qvs1 = ((lv1 & 7) ^ (lv1 >> 4)) * 16;
                let mut v_base_st = smem + V_OFF + stage * V_STAGE_BYTES as u32 + (lv1 & 15) * 256;
                core::arch::asm!("/* {0} {1} */", inout(reg32) v_base_st, inout(reg32) qvs1, options(nomem, nostack));
                let v1_0 = v_base_st + 4096;
                macro_rules! pp_mma {
                    ($vb:expr, $oa:ident, $ob:ident) => {{
                        $oa = mma_m16n8k16_bf16(pa1_0, [$vb[0], $vb[1]], $oa);
                        $ob = mma_m16n8k16_bf16(pa1_0, [$vb[2], $vb[3]], $ob);
                    }};
                }
                let vb = ldmatrix_x4_trans(v1_0 + qvs1); sc_fast!(s0); pp_mma!(vb, o0, o1);
                let vb = ldmatrix_x4_trans(v1_0 + (32 ^ qvs1)); sc_fast!(s1); pp_mma!(vb, o2, o3);
                let vb = ldmatrix_x4_trans(v1_0 + (64 ^ qvs1)); sc_fast!(s2); pp_mma!(vb, o4, o5);
                let vb = ldmatrix_x4_trans(v1_0 + (96 ^ qvs1)); sc_fast!(s3); pp_mma!(vb, o6, o7);
                let vb = ldmatrix_x4_trans(v1_0 + (128 ^ qvs1));
                let mut mt0 = fmax_hw!(fmax_hw!(fmax_hw!(s0[0], s0[1]), fmax_hw!(s1[0], s1[1])), fmax_hw!(fmax_hw!(s2[0], s2[1]), fmax_hw!(s3[0], s3[1])));
                mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 1)); mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 2));
                let mut mt1 = fmax_hw!(fmax_hw!(fmax_hw!(s0[2], s0[3]), fmax_hw!(s1[2], s1[3])), fmax_hw!(fmax_hw!(s2[2], s2[3]), fmax_hw!(s3[2], s3[3])));
                mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 1)); mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 2));
                let chg1 = crate::ptx::ballot(mt0 > m_r0 || mt1 > m_r1) != 0;
                let (a0_1, a1_1) = if chg1 {
                    let mn0 = fmax_hw!(m_r0, mt0); let mn1 = fmax_hw!(m_r1, mt1);
                    let a0 = if mn0 > m_r0 { exp_nonpos(m_r0 - mn0) } else { 1.0f32 };
                    let a1 = if mn1 > m_r1 { exp_nonpos(m_r1 - mn1) } else { 1.0f32 };
                    m_r0 = mn0; m_r1 = mn1;
                    (a0, a1)
                } else { (1.0f32, 1.0f32) };
                pp_mma!(vb, o8, o9);
                let vb = ldmatrix_x4_trans(v1_0 + (160 ^ qvs1));
                let p00 = exp_nonpos(s0[0] - m_r0); let p01 = exp_nonpos(s0[1] - m_r0);
                let pa0_a = cvt_bf16x2(p00, p01); let s0_a0 = p00 + p01;
                let p02 = exp_nonpos(s0[2] - m_r1); let p03 = exp_nonpos(s0[3] - m_r1);
                let pa0_b = cvt_bf16x2(p02, p03); let s1_a0 = p02 + p03;
                pp_mma!(vb, o10, o11);
                let vb = ldmatrix_x4_trans(v1_0 + (192 ^ qvs1));
                let p10 = exp_nonpos(s1[0] - m_r0); let p11 = exp_nonpos(s1[1] - m_r0);
                let pa0_c = cvt_bf16x2(p10, p11); let s0_a1 = s0_a0 + (p10 + p11);
                pp_mma!(vb, o12, o13);
                let vb = ldmatrix_x4_trans(v1_0 + (224 ^ qvs1));
                let p12 = exp_nonpos(s1[2] - m_r1); let p13 = exp_nonpos(s1[3] - m_r1);
                let pa0_1 = [pa0_a, pa0_b, pa0_c, cvt_bf16x2(p12, p13)];
                let s1_a1 = s1_a0 + (p12 + p13);
                pp_mma!(vb, o14, o15);

                // Now Sub-tile 0's `o0..o15` is complete -> apply Sub-tile 1's rescale (if any).
                if chg1 {
                    if a0_1 != 1.0f32 { macro_rules! sc { ($($o:ident)*) => { $($o[0] *= a0_1; $o[1] *= a0_1;)* }; } sc!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                    if a1_1 != 1.0f32 { macro_rules! sc { ($($o:ident)*) => { $($o[2] *= a1_1; $o[3] *= a1_1;)* }; } sc!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                }
                let v0_1 = v_base_st + V_BYTES as u32;
                let pa1_1 = pv0_exp_sub!(v0_1, qvs1, pa0_1, s2, s3, s0_a1, s1_a1, chg1, a0_1, a1_1);

                // Fused 2-step pipelined `pa1_1 * V[48..64]` (16 HMMAs, 4-HMMA LDSM.T hide distance).
                let v1_1 = v0_1 + 4096;
                let mut vb_a = ldmatrix_x4_trans(v1_1 + qvs1);
                let mut vb_b = ldmatrix_x4_trans(v1_1 + (32 ^ qvs1));
                vb_a = ldm_trans_and_mma2!(v1_1 + (64 ^ qvs1), pa1_1, vb_a, o0, o1);
                vb_b = ldm_trans_and_mma2!(v1_1 + (96 ^ qvs1), pa1_1, vb_b, o2, o3);
                vb_a = ldm_trans_and_mma2!(v1_1 + (128 ^ qvs1), pa1_1, vb_a, o4, o5);
                vb_b = ldm_trans_and_mma2!(v1_1 + (160 ^ qvs1), pa1_1, vb_b, o6, o7);
                vb_a = ldm_trans_and_mma2!(v1_1 + (192 ^ qvs1), pa1_1, vb_a, o8, o9);
                vb_b = ldm_trans_and_mma2!(v1_1 + (224 ^ qvs1), pa1_1, vb_b, o10, o11);
                o12 = mma_m16n8k16_bf16(pa1_1, [vb_a[0], vb_a[1]], o12);
                o13 = mma_m16n8k16_bf16(pa1_1, [vb_a[2], vb_a[3]], o13);
                o14 = mma_m16n8k16_bf16(pa1_1, [vb_b[0], vb_b[1]], o14);
                o15 = mma_m16n8k16_bf16(pa1_1, [vb_b[2], vb_b[3]], o15);

                if next64 < t_int_end {
                    cp_async_wait_zero();
                    core::arch::nvptx::_syncthreads();
                }
                stage ^= 1;
                t_base = next64;
            }
            // Stage the first suffix tile ONCE outside the Ping-Pong loop into the completely idle `stage` buffer
            // (eliminates `issue_kvt!` live regs inside Ping-Pong so it compiles to 0 stack, 0 spill!).
            if t_base < arc_end_loop {
                let cb_s = smem + C_OFF;
                issue_kvt!(
                    stage, t_base,
                    load_u32(cb_s), load_u32(cb_s + 4), load_u32(cb_s + 8), load_u32(cb_s + 12),
                    arc_end_loop, load_u32(cb_s + 20), load_u32(cb_s + 24), load_u32(cb_s + 44), 384u32
                );
                crate::ptx::commit();
                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
            }
        }

        // Compute per-warp `(warp_min_col, warp_max_col)` only when general/suffix tiles remain.
        if t_base < arc_end_loop {
            let mut tid_g = core::arch::nvptx::_thread_idx_x() as u32;
            core::arch::asm!("/* {0} */", inout(reg32) tid_g, options(nomem, nostack));
            let lane_g = tid_g & 31;
            let w_min_row_g = ((tid_g >> 5) & 3) * 16;
            let cb_g = smem + C_OFF;
            let pad_g = load_u32(cb_g + 8);
            let live0_g = load_u32(cb_g + 20);
            let m_cnt_g = load_u32(cb_g + 28);
            let w_row_probe = w_min_row_g + (lane_g & 15);
            let mut warp_min_col = if w_row_probe < m_cnt_g {
                pad_g + live0_g + w_row_probe - load_u32(smem + L_OFF + w_row_probe * 4)
            } else {
                u32::MAX
            };
            let mut ws = 8u32;
            while ws > 0 {
                let o = shfl_xor_u32(warp_min_col, ws);
                if o < warp_min_col { warp_min_col = o; }
                ws >>= 1;
            }
            let w_last_row = if m_cnt_g > w_min_row_g { (w_min_row_g + 15).min(m_cnt_g - 1) } else { 0 };
            let warp_max_col = pad_g + (live0_g - 1) + w_last_row;

            while t_base < arc_end_loop {
                let next_t = t_base + FW_BKV as u32;
            // Issue stage `stage ^ 1` (K, V, T) for `next_t` overlapped with ALL 64 HMMAs + softmax of `t_base`!
            if next_t < arc_end_loop {
                let cb = smem + C_OFF;
                issue_kvt!(
                    stage ^ 1, next_t,
                    load_u32(cb), load_u32(cb + 4), load_u32(cb + 8), load_u32(cb + 12),
                    arc_end_loop, load_u32(cb + 20), load_u32(cb + 24), load_u32(cb + 44), load_u32(cb + 52)
                );
                crate::ptx::commit();
            }

            let mut tid_l = core::arch::nvptx::_thread_idx_x() as u32;
            core::arch::asm!("/* {0} */", inout(reg32) tid_l, options(nomem, nostack));
            let lane_l = tid_l & 31;
            let warp_l = tid_l >> 5;
            let h_loc = warp_l >> 2;
            let w_pos_l = warp_l & 3;
            let w_min_r = w_pos_l * 16;
            let cb = smem + C_OFF;
            let m_cnt32 = load_u32(cb + 28);

            if w_min_r < m_cnt32 && t_base <= warp_max_col && t_base + FW_BKV as u32 > warp_min_col {
                // 1. S = Q * K[stage]^T (`16 x 32` per warp = 32 HMMAs) from register `q_reg0..q_reg7` (0 `LDSM(Q)`!).
                let k_ld_row0 = ((lane_l >> 4) * 8) + (lane_l & 7);
                let mut k_cur = smem + K_OFF + stage * K_STAGE_BYTES as u32 + k_ld_row0 * 256;
                let mut ks_m = ((k_ld_row0 & 7) ^ ((lane_l >> 3) & 1)) * 16;
                core::arch::asm!("/* {0} {1} */", inout(reg32) k_cur, inout(reg32) ks_m, options(nomem, nostack));
                let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                let mut kb0 = ldmatrix_x4(k_cur + ks_m);
                let mut kb1 = ldmatrix_x4(k_cur + 4096 + ks_m);
                macro_rules! qk_gen {
                    ($ks:expr, $qreg:ident) => {{
                        let next_off = ((($ks + 1) * 32) as u32) ^ ks_m;
                        kb0 = ldm_and_mma2!(k_cur + next_off, $qreg, kb0, s0, s1);
                        kb1 = ldm_and_mma2!(k_cur + 4096 + next_off, $qreg, kb1, s2, s3);
                    }};
                }
                qk_gen!(0, q_reg0); qk_gen!(1, q_reg1); qk_gen!(2, q_reg2);
                qk_gen!(3, q_reg3); qk_gen!(4, q_reg4); qk_gen!(5, q_reg5);
                let qhi_row_g = h_loc * 64 + w_min_r + (lane_l & 15);
                let qhi_base_g = smem + Q_HI_OFF + qhi_row_g * 64;
                let qhi_sw_g = ((lane_l >> 4) ^ ((qhi_row_g >> 1) & 3)) * 16;
                let mut qhi_g = ldmatrix_x4(qhi_base_g + qhi_sw_g);
                qk_gen!(6, qhi_g);
                qhi_g = ldmatrix_x4(qhi_base_g + (32 ^ qhi_sw_g));
                s0 = mma_m16n8k16_bf16(qhi_g, [kb0[0], kb0[1]], s0);
                s1 = mma_m16n8k16_bf16(qhi_g, [kb0[2], kb0[3]], s1);
                s2 = mma_m16n8k16_bf16(qhi_g, [kb1[0], kb1[1]], s2);
                s3 = mma_m16n8k16_bf16(qhi_g, [kb1[2], kb1[3]], s3);

                // 2. Scale, optional relative bias, and masking (zero-branch Fast Interior for both full & sliding window!).
                let extent32 = load_u32(cb + 4);
                let pad32 = load_u32(cb + 8);
                let live0_32 = load_u32(cb + 20);
                let max_min_col = load_u32(cb + 48);
                let need_rel = t_base + FW_BKV as u32 + extent32 > pad32 + live0_32;
                let interior_geom = m_cnt32 == FW_PREFILL_BQ_POS as u32
                    && t_base >= max_min_col
                    && t_base >= pad32
                    && t_base + FW_BKV as u32 <= pad32 + live0_32;

                if interior_geom && !need_rel {
                    sc_fast!(s0); sc_fast!(s1); sc_fast!(s2); sc_fast!(s3);
                } else {
                    // Pack `rn(rn(s) * scale)` into 8 BF16x2 u32 registers BEFORE `RelBias` (frees 8 live regs so
                    // `q_reg0..q_reg7` + `o0..o15` + `RelBias` fit in <= 168 regs with 0 stack, 0 spill!).
                    macro_rules! pack_sc {
                        ($s:ident) => {{
                            let (a0, a1) = rn2($s[0], $s[1]);
                            let (a2, a3) = rn2($s[2], $s[3]);
                            [cvt_bf16x2(a0 * scale, a1 * scale), cvt_bf16x2(a2 * scale, a3 * scale)]
                        }};
                    }
                    macro_rules! unpack_bf16x2 {
                        ($p:expr) => {{
                            let p: u32 = $p;
                            let lo: u32;
                            core::arch::asm!(
                                "prmt.b32 {lo}, {p}, 0, 0x1044;",
                                lo = out(reg32) lo, p = in(reg32) p,
                                options(pure, nomem, nostack),
                            );
                            (f32::from_bits(lo), f32::from_bits(p & 0xffff_0000))
                        }};
                    }
                    let ps0 = pack_sc!(s0);
                    let ps1 = pack_sc!(s1);
                    let ps2 = pack_sc!(s2);
                    let ps3 = pack_sc!(s3);

                    let r0 = w_min_r + (lane_l >> 2);
                    let t32 = lane_l & 3;
                    let qpos_r0 = pad32 + (live0_32 - 1) + r0;
                    let fast_rel = interior_geom && need_rel && (pad32 + live0_32 + (FW_PREFILL_BQ_POS as u32 - 2) - t_base) < extent32;
                    let live_r0 = if fast_rel { 0 } else { load_u32(smem + L_OFF + r0 * 4) };
                    let live_r1 = if fast_rel { 0 } else { load_u32(smem + L_OFF + (r0 + 8) * 4) };
                    macro_rules! finalize_ps {
                        ($ps:expr, $col_off:expr, $b:expr) => {{
                            let [p01, p23]: [u32; 2] = $ps;
                            let (c0, c1) = unpack_bf16x2!(p01);
                            let (c2, c3) = unpack_bf16x2!(p23);
                            let [b00, b01, b10, b11]: [f32; 4] = $b;
                            if fast_rel {
                                let (rb0, rb1) = rn2(b00, b01);
                                let (rb2, rb3) = rn2(b10, b11);
                                let (d0, d1) = rn2(c0 + rb0, c1 + rb1);
                                let (d2, d3) = rn2(c2 + rb2, c3 + rb3);
                                [d0, d1, d2, d3]
                            } else {
                                let col0 = t_base + $col_off + t32 * 2;
                                let col1 = col0 + 1;
                                macro_rules! eval_sc {
                                    ($scaled:expr, $row_local:expr, $qpos:expr, $live_r:expr, $col:expr, $bsum:expr) => {{
                                        let col: u32 = $col;
                                        let qpos: u32 = $qpos;
                                        if $row_local >= m_cnt32 || col >= arc_end_loop || col > qpos {
                                            f32::NEG_INFINITY
                                        } else {
                                            let dist = qpos - col;
                                            if dist >= $live_r {
                                                f32::NEG_INFINITY
                                            } else if need_rel && dist < extent32 {
                                                rn($scaled + rn($bsum))
                                            } else {
                                                $scaled
                                            }
                                        }
                                    }};
                                }
                                [
                                    eval_sc!(c0, r0, qpos_r0, live_r0, col0, b00),
                                    eval_sc!(c1, r0, qpos_r0, live_r0, col1, b01),
                                    eval_sc!(c2, r0 + 8, qpos_r0 + 8, live_r1, col0, b10),
                                    eval_sc!(c3, r0 + 8, qpos_r0 + 8, live_r1, col1, b11),
                                ]
                            }
                        }};
                    }
                    if need_rel {
                        // Two-pass relative bias: bank-free `ld.shared.v2.b32` `M` loads (`swiz_m8`, ALU `prmt.b32`
                        // unpack) + `ld.shared.v4` `T` loads (reusing shared `at`/`at-16` and `at-256`/`at-272` terms).
                        let t_rel_off = 16 * (r0 + FW_BKV as u32 - 1 - t32 * 2);
                        let mut t_start = smem + T_OFF + stage * T_STAGE_BYTES as u32 + t_rel_off;
                        let qr0_u = h_loc * 64 + r0;
                        let mut m_row0 = smem + M_OFF + qr0_u * 32;
                        let mut m_swz = ((qr0_u >> 2) & 1) * 8;
                        core::arch::asm!("/* {0} {1} {2} */", inout(reg32) t_start, inout(reg32) m_row0, inout(reg32) m_swz, options(nomem, nostack));
                        macro_rules! m_quad {
                            ($row:expr, $jq:expr) => {{
                                let [w0, w1] = crate::ptx::load2($row + (($jq * 8) ^ m_swz));
                                let (m0, m1) = unpack_bf16x2!(w0);
                                let (m2, m3) = unpack_bf16x2!(w1);
                                [m0, m1, m2, m3]
                            }};
                        }
                        macro_rules! t_quad {
                            ($at:expr) => {{
                                let t = crate::ptx::load4($at);
                                [f32::from_bits(t[0]), f32::from_bits(t[1]), f32::from_bits(t[2]), f32::from_bits(t[3])]
                            }};
                        }
                        macro_rules! acc {
                            ($b:ident, $m:ident, $t:ident) => {
                                $b += $m[0] * $t[0]; $b += $m[1] * $t[1]; $b += $m[2] * $t[2]; $b += $m[3] * $t[3];
                            };
                        }
                        let (mut b00, mut b01, mut b02, mut b03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        let (mut b10, mut b11, mut b12, mut b13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        let mut jq = 0u32;
                        let mut at = t_start;
                        while jq < 4 {
                            let a = m_quad!(m_row0, jq);
                            let m = m_quad!(m_row0 + 256, jq);
                            let t = t_quad!(at + 128); acc!(b02, m, t);
                            let t = t_quad!(at + 112); acc!(b03, m, t);
                            let t = t_quad!(at);       acc!(b00, a, t); acc!(b12, m, t);
                            let t = t_quad!(at - 16);  acc!(b01, a, t); acc!(b13, m, t);
                            let t = t_quad!(at - 128); acc!(b10, a, t);
                            let t = t_quad!(at - 144); acc!(b11, a, t);
                            at += T_PLANE_BYTES as u32;
                            jq += 1;
                        }
                        macro_rules! pack_hi16 {
                            ($a:expr, $b:expr) => {{
                                let d: u32;
                                core::arch::asm!(
                                    "prmt.b32 {d}, {a}, {b}, 0x7632;",
                                    d = out(reg32) d, a = in(reg32) ($a as f32).to_bits(), b = in(reg32) ($b as f32).to_bits(),
                                    options(pure, nomem, nostack),
                                );
                                d
                            }};
                        }
                        let f0 = finalize_ps!(ps0, 0, [b00, b01, b02, b03]);
                        let f1 = finalize_ps!(ps1, 8, [b10, b11, b12, b13]);
                        let pf0 = [pack_hi16!(f0[0], f0[1]), pack_hi16!(f0[2], f0[3])];
                        let pf1 = [pack_hi16!(f1[0], f1[1]), pack_hi16!(f1[2], f1[3])];
                        let (mut b20, mut b21, mut b22, mut b23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        let (mut b30, mut b31, mut b32, mut b33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        jq = 0; at = t_start;
                        while jq < 4 {
                            let a = m_quad!(m_row0, jq);
                            let m = m_quad!(m_row0 + 256, jq);
                            let t = t_quad!(at - 128); acc!(b22, m, t);
                            let t = t_quad!(at - 144); acc!(b23, m, t);
                            let t = t_quad!(at - 256); acc!(b20, a, t); acc!(b32, m, t);
                            let t = t_quad!(at - 272); acc!(b21, a, t); acc!(b33, m, t);
                            let t = t_quad!(at - 384); acc!(b30, a, t);
                            let t = t_quad!(at - 400); acc!(b31, a, t);
                            at += T_PLANE_BYTES as u32;
                            jq += 1;
                        }
                        s2 = finalize_ps!(ps2, 16, [b20, b21, b22, b23]);
                        s3 = finalize_ps!(ps3, 24, [b30, b31, b32, b33]);
                        let (u00, u01) = unpack_bf16x2!(pf0[0]); let (u02, u03) = unpack_bf16x2!(pf0[1]);
                        let (u10, u11) = unpack_bf16x2!(pf1[0]); let (u12, u13) = unpack_bf16x2!(pf1[1]);
                        s0 = [u00, u01, u02, u03]; s1 = [u10, u11, u12, u13];
                    } else {
                        s0 = finalize_ps!(ps0, 0, [0.0; 4]);
                        s1 = finalize_ps!(ps1, 8, [0.0; 4]);
                        s2 = finalize_ps!(ps2, 16, [0.0; 4]);
                        s3 = finalize_ps!(ps3, 24, [0.0; 4]);
                    }
                }

                // 3. Online softmax: hardware FMNMX warp-butterfly max + warp-uniform `ballot` skip + `exp_nonpos`.
                let mut mt0 = fmax_hw!(fmax_hw!(fmax_hw!(s0[0], s0[1]), fmax_hw!(s1[0], s1[1])), fmax_hw!(fmax_hw!(s2[0], s2[1]), fmax_hw!(s3[0], s3[1])));
                mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 1)); mt0 = fmax_hw!(mt0, shfl_xor_f32(mt0, 2));
                let mut mt1 = fmax_hw!(fmax_hw!(fmax_hw!(s0[2], s0[3]), fmax_hw!(s1[2], s1[3])), fmax_hw!(fmax_hw!(s2[2], s2[3]), fmax_hw!(s3[2], s3[3])));
                mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 1)); mt1 = fmax_hw!(mt1, shfl_xor_f32(mt1, 2));
                let changed = crate::ptx::ballot(mt0 > m_r0 || mt1 > m_r1) != 0;
                let (alpha0, alpha1) = if changed {
                    let (m_new0, m_new1) = (fmax_hw!(m_r0, mt0), fmax_hw!(m_r1, mt1));
                    let a0 = if m_new0 > m_r0 { exp_nonpos(m_r0 - m_new0) } else { 1.0f32 };
                    let a1 = if m_new1 > m_r1 { exp_nonpos(m_r1 - m_new1) } else { 1.0f32 };
                    m_r0 = m_new0; m_r1 = m_new1;
                    if a0 != 1.0f32 { macro_rules! sc { ($($o:ident)*) => { $($o[0] *= a0; $o[1] *= a0;)* }; } sc!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                    if a1 != 1.0f32 { macro_rules! sc { ($($o:ident)*) => { $($o[2] *= a1; $o[3] *= a1;)* }; } sc!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15); }
                    (a0, a1)
                } else { (1.0f32, 1.0f32) };

                macro_rules! to_p {
                    ($v:expr, $mn:expr) => {{
                        let v: f32 = $v;
                        let e = exp_nonpos(v - $mn);
                        if !interior_geom && v == f32::NEG_INFINITY { 0.0f32 } else { e }
                    }};
                }
                let p00 = to_p!(s0[0], m_r0); let p01 = to_p!(s0[1], m_r0);
                let p02 = to_p!(s0[2], m_r1); let p03 = to_p!(s0[3], m_r1);
                let p10 = to_p!(s1[0], m_r0); let p11 = to_p!(s1[1], m_r0);
                let p12 = to_p!(s1[2], m_r1); let p13 = to_p!(s1[3], m_r1);
                let pa0 = [cvt_bf16x2(p00, p01), cvt_bf16x2(p02, p03), cvt_bf16x2(p10, p11), cvt_bf16x2(p12, p13)];
                let s0_a = (p00 + p01) + (p10 + p11);
                let s1_a = (p02 + p03) + (p12 + p13);

                // 4. O += P * V[stage]: interleave `pa0 * V[0..16]` (16 HMMAs) with `p20..p33` exps, then fused `pa1 * V[16..32]`.
                let v_ld_r0 = lane_l & 15;
                let mut v_cur = smem + V_OFF + stage * V_STAGE_BYTES as u32 + v_ld_r0 * 256;
                let mut vs = ((v_ld_r0 & 7) ^ (lane_l >> 4)) * 16;
                core::arch::asm!("/* {0} {1} */", inout(reg32) v_cur, inout(reg32) vs, options(nomem, nostack));
                macro_rules! pe_g {
                    ($pair:expr, $oa:ident, $ob:ident, $sv:expr, $mr:expr) => {{
                        let vb0 = ldmatrix_x4_trans(v_cur + ((($pair * 32) as u32) ^ vs));
                        let pv = to_p!($sv, $mr);
                        $oa = mma_m16n8k16_bf16(pa0, [vb0[0], vb0[1]], $oa);
                        $ob = mma_m16n8k16_bf16(pa0, [vb0[2], vb0[3]], $ob);
                        pv
                    }};
                }
                let p20 = pe_g!(0, o0, o1, s2[0], m_r0); let p21 = pe_g!(1, o2, o3, s2[1], m_r0);
                let pa1_0 = cvt_bf16x2(p20, p21); let s0_b0 = p20 + p21;
                let p22 = pe_g!(2, o4, o5, s2[2], m_r1); let p23 = pe_g!(3, o6, o7, s2[3], m_r1);
                let pa1_1 = cvt_bf16x2(p22, p23); let s1_b0 = p22 + p23;
                let p30 = pe_g!(4, o8, o9, s3[0], m_r0); let p31 = pe_g!(5, o10, o11, s3[1], m_r0);
                let pa1_2 = cvt_bf16x2(p30, p31); let s0_b1 = p30 + p31;
                let p32 = pe_g!(6, o12, o13, s3[2], m_r1); let p33 = pe_g!(7, o14, o15, s3[3], m_r1);
                let pa1 = [pa1_0, pa1_1, pa1_2, cvt_bf16x2(p32, p33)];
                let mut sum0 = s0_a + (s0_b0 + s0_b1);
                sum0 += shfl_xor_f32(sum0, 1); sum0 += shfl_xor_f32(sum0, 2);
                let mut sum1 = s1_a + (s1_b0 + (p32 + p33));
                sum1 += shfl_xor_f32(sum1, 1); sum1 += shfl_xor_f32(sum1, 2);
                if changed { l_r0 = l_r0 * alpha0 + sum0; l_r1 = l_r1 * alpha1 + sum1; } else { l_r0 += sum0; l_r1 += sum1; }

                let v1_g = v_cur + 4096;
                let mut vb_a = ldmatrix_x4_trans(v1_g + vs);
                let mut vb_b = ldmatrix_x4_trans(v1_g + (32 ^ vs));
                vb_a = ldm_trans_and_mma2!(v1_g + (64 ^ vs), pa1, vb_a, o0, o1);
                vb_b = ldm_trans_and_mma2!(v1_g + (96 ^ vs), pa1, vb_b, o2, o3);
                vb_a = ldm_trans_and_mma2!(v1_g + (128 ^ vs), pa1, vb_a, o4, o5);
                vb_b = ldm_trans_and_mma2!(v1_g + (160 ^ vs), pa1, vb_b, o6, o7);
                vb_a = ldm_trans_and_mma2!(v1_g + (192 ^ vs), pa1, vb_a, o8, o9);
                vb_b = ldm_trans_and_mma2!(v1_g + (224 ^ vs), pa1, vb_b, o10, o11);
                o12 = mma_m16n8k16_bf16(pa1, [vb_a[0], vb_a[1]], o12);
                o13 = mma_m16n8k16_bf16(pa1, [vb_a[2], vb_a[3]], o13);
                o14 = mma_m16n8k16_bf16(pa1, [vb_b[0], vb_b[1]], o14);
                o15 = mma_m16n8k16_bf16(pa1, [vb_b[2], vb_b[3]], o15);
            }

                if next_t < arc_end_loop {
                    cp_async_wait_zero();
                    core::arch::nvptx::_syncthreads();
                }
                stage ^= 1;
                t_base = next_t;
            }
        }

        // Epilogue: load CTA constants from `C_OFF` (zero divisions, zero panics, zero loop-live epilogue regs).
        // Fence `tid_e32` with an empty asm so LLVM does not keep prologue `warp`/`lane` live across the KV loop.
        let cb = smem + C_OFF;
        let m_cntu = load_u32(cb + 28) as usize;
        let q0u = load_u32(cb + 32) as usize;
        let head_base_u = load_u32(cb + 36) as usize;
        let heads_e = load_u32(cb + 40) as usize;
        let thr_e = load_u32(cb + 52) as usize;
        let gate_e = *w.add(6) as *const u32;
        let mut tid_e32 = core::arch::nvptx::_thread_idx_x() as u32;
        core::arch::asm!("/* {0} */", inout(reg32) tid_e32, options(nomem, nostack));
        let tid_e = tid_e32 as usize;

        // Reclaim dead SMEM [0 .. 52,224 B]: cooperatively load `gate` via coalesced 16B `L2::evict_first` reads,
        // round to BF16x2 into bank-free `EPI_ROW_BYTES` (`272 B/row`) slots, then overwrite each thread's own slots
        // in place for coalesced `st.global.v4` output (0 strided gate loads, 0 extra SMEM).
        core::arch::nvptx::_syncthreads();
        let gpack_e = thr_e >> 7;
        let total_g = gpack_e * FW_PREFILL_BQ_POS * 16;
        let evict = l2_evict_first();
        let mut gidx = tid_e;
        while gidx < total_g {
            let qr = gidx >> 4;
            let ch = gidx & 15;
            let hl = qr >> 6;
            let r = qr & 63;
            if r < m_cntu && (*w.add(FW_CLIP_WORD) & FW_QG_HALF) != 0 {
                // The gate as BF16 pairs (`FW_QG_HALF`): the words that `cvt_bf16x2` below makes of the product words.
                let p = gate_e.add(((q0u + r) * heads_e + head_base_u + hl) * (FW_HEAD_DIM / 2) + ch * 4);
                st_shared_v4(smem + (qr * EPI_ROW_BYTES + ch * 16) as u32, ld_global_nc_v4_u32_hint(p, evict));
            } else if r < m_cntu {
                let p = gate_e.add(((q0u + r) * heads_e + head_base_u + hl) * FW_HEAD_DIM + ch * 8);
                let v0 = ld_global_nc_v4_u32_hint(p, evict);
                let v1 = ld_global_nc_v4_u32_hint(p.add(4), evict);
                st_shared_v4(smem + (qr * EPI_ROW_BYTES + ch * 16) as u32, [
                    cvt_bf16x2(f32::from_bits(v0[0]), f32::from_bits(v0[1])),
                    cvt_bf16x2(f32::from_bits(v0[2]), f32::from_bits(v0[3])),
                    cvt_bf16x2(f32::from_bits(v1[0]), f32::from_bits(v1[1])),
                    cvt_bf16x2(f32::from_bits(v1[2]), f32::from_bits(v1[3])),
                ]);
            }
            gidx += thr_e;
        }
        core::arch::nvptx::_syncthreads();

        // Branchless IEEE-exact `1.0f32 / d` for normal `d` with exponent in `[2, 253]` (identical to SM121a's
        // `rcp.rn.f32` fast path: `MUFU.RCP` + 1 exact FMA residual step, zero slow-path `CALL`/`BSSY`/`BRA`).
        macro_rules! rcp_rn_normal {
            ($d:expr) => {{
                let d: f32 = $d;
                let y: f32;
                core::arch::asm!(
                    "{{ .reg .f32 r, e, ne; \
                     rcp.approx.ftz.f32 r, {d}; \
                     fma.rn.f32 e, r, {d}, 0fbf800000; \
                     neg.f32 ne, e; \
                     add.rn.ftz.f32 ne, ne, 0f80000000; \
                     fma.rn.f32 {y}, r, ne, r; }}",
                    y = out(reg32) y, d = in(reg32) d,
                    options(pure, nomem, nostack),
                );
                y
            }};
        }

        // Branchless bit-exact `fixed_math::sigmoid(x)` for BF16-rounded `x`: `t in [-87, +87]` guarantees
        // `d = 1.0 + exp(t) in [1.0, 6.08e37]` (exponent in `[127, 252]`), strictly inside `rcp_rn_normal!`.
        macro_rules! sigmoid_fast {
            ($x:expr) => {{
                let t0 = -$x;
                let t = if t0 > 87.0f32 { 87.0f32 } else if t0 < -87.0f32 { -87.0f32 } else { t0 };
                let k: f32;
                let n: f32;
                core::arch::asm!(
                    "{{ .reg .f32 p; \
                     mul.rn.f32 p, {t}, 0f3fb8aa3b; \
                     add.rn.f32 {k}, p, 0f4b400000; \
                     add.rn.f32 {n}, {k}, 0fcb400000; }}",
                    k = out(reg32) k, n = out(reg32) n, t = in(reg32) t,
                    options(pure, nomem, nostack),
                );
                let r = core::f32::math::mul_add(-n, 0.693145751953125f32, t);
                let r = core::f32::math::mul_add(-n, 0.0000014286068f32, r);
                let p = core::f32::math::mul_add(1.0 / 5040.0f32, r, 1.0 / 720.0);
                let p = core::f32::math::mul_add(p, r, 1.0 / 120.0);
                let p = core::f32::math::mul_add(p, r, 1.0 / 24.0);
                let p = core::f32::math::mul_add(p, r, 1.0 / 6.0);
                let p = core::f32::math::mul_add(p, r, 0.5);
                let p = core::f32::math::mul_add(p, r, 1.0);
                let p = core::f32::math::mul_add(p, r, 1.0);
                let e = p * f32::from_bits(k.to_bits().wrapping_add(127) << 23);
                rcp_rn_normal!(1.0f32 + e)
            }};
        }

        let (warp_e, lane_e) = (tid_e >> 5, tid_e & 31);
        let (g_e, tu) = (lane_e >> 2, lane_e & 3);
        let w_pos_e = warp_e & 3;
        let r0u = w_pos_e * 16 + g_e;
        let r1u = r0u + 8;
        let inv0 = rcp_rn_normal!(l_r0);
        let inv1 = rcp_rn_normal!(l_r1);
        let warp_smem = smem + (warp_e * EPI_WARP_BYTES) as u32;
        let row0_smem = warp_smem + (g_e * EPI_ROW_BYTES) as u32;
        let row1_smem = warp_smem + ((g_e + 8) * EPI_ROW_BYTES) as u32;

        // Single shared stage pass for both BF16 (`FW_HALF_OUT`) and FP32 outputs (cuts epilogue code size in half).
        macro_rules! stage_o {
            ($ni:expr, $o:ident) => {{
                let off = (($ni * 4 + tu) * 4) as u32;
                if r0u < m_cntu {
                    let gp = load_u32(row0_smem + off);
                    let (g0, g1) = (f32::from_bits(gp << 16), f32::from_bits(gp & 0xffff_0000));
                    let (a0, a1) = rn2($o[0] * inv0, $o[1] * inv0);
                    store_u32(row0_smem + off, cvt_bf16x2(a0 * sigmoid_fast!(g0), a1 * sigmoid_fast!(g1)));
                }
                if r1u < m_cntu {
                    let gp = load_u32(row1_smem + off);
                    let (g0, g1) = (f32::from_bits(gp << 16), f32::from_bits(gp & 0xffff_0000));
                    let (a0, a1) = rn2($o[2] * inv1, $o[3] * inv1);
                    store_u32(row1_smem + off, cvt_bf16x2(a0 * sigmoid_fast!(g0), a1 * sigmoid_fast!(g1)));
                }
            }};
        }
        stage_o!(0, o0); stage_o!(1, o1); stage_o!(2, o2); stage_o!(3, o3);
        stage_o!(4, o4); stage_o!(5, o5); stage_o!(6, o6); stage_o!(7, o7);
        stage_o!(8, o8); stage_o!(9, o9); stage_o!(10, o10); stage_o!(11, o11);
        stage_o!(12, o12); stage_o!(13, o13); stage_o!(14, o14); stage_o!(15, o15);
        crate::ptx::warp_sync();

        let output_e = *w.add(8) as *mut f32;
        let half_out = (*w.add(FW_CLIP_WORD) & FW_HALF_OUT) != 0;
        let headu = load_u32(cb + 36) as usize + (warp_e >> 2);
        let q0_out = load_u32(cb + 32) as usize;
        let heads_out = load_u32(cb + 40) as usize;
        let r_valid = m_cntu.saturating_sub(w_pos_e * 16).min(16);
        if half_out {
            let output_u32 = output_e.cast::<u32>();
            let mut p = 0usize;
            while p < 8 {
                let cidx = p * 32 + lane_e;
                let rl = cidx >> 4;
                let ch = cidx & 15;
                if rl < r_valid {
                    let vals = crate::ptx::load4(warp_smem + (rl * EPI_ROW_BYTES + ch * 16) as u32);
                    let dst = output_u32.add(((q0_out + w_pos_e * 16 + rl) * heads_out + headu) * 64 + ch * 4);
                    core::arch::asm!(
                        "st.global.v4.b32 [{dst}], {{{a0}, {a1}, {a2}, {a3}}};",
                        dst = in(reg64) dst,
                        a0 = in(reg32) vals[0], a1 = in(reg32) vals[1], a2 = in(reg32) vals[2], a3 = in(reg32) vals[3],
                        options(nostack),
                    );
                }
                p += 1;
            }
        } else {
            let mut p = 0usize;
            while p < 8 {
                let cidx = p * 32 + lane_e;
                let rl = cidx >> 4;
                let ch = cidx & 15;
                if rl < r_valid {
                    let vals = crate::ptx::load4(warp_smem + (rl * EPI_ROW_BYTES + ch * 16) as u32);
                    let dst = output_e.add(((q0_out + w_pos_e * 16 + rl) * heads_out + headu) * FW_HEAD_DIM + ch * 8);
                    core::arch::asm!(
                        "st.global.v4.b32 [{dst}], {{{a0}, {a1}, {a2}, {a3}}};",
                        dst = in(reg64) dst,
                        a0 = in(reg32) (vals[0] << 16), a1 = in(reg32) (vals[0] & 0xffff_0000),
                        a2 = in(reg32) (vals[1] << 16), a3 = in(reg32) (vals[1] & 0xffff_0000),
                        options(nostack),
                    );
                    core::arch::asm!(
                        "st.global.v4.b32 [{dst}], {{{a0}, {a1}, {a2}, {a3}}};",
                        dst = in(reg64) dst.add(4),
                        a0 = in(reg32) (vals[2] << 16), a1 = in(reg32) (vals[2] & 0xffff_0000),
                        a2 = in(reg32) (vals[3] << 16), a3 = in(reg32) (vals[3] & 0xffff_0000),
                        options(nostack),
                    );
                }
                p += 1;
            }
        }
    }
}
