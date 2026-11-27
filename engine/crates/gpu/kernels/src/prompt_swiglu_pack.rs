//! Dedicated fast SwiGLU (`causal_swiglu`) and fused routed `SwiGlu + ExpertPlaceTile + nv_quantize_pack`
//! (`prompt_swiglu_place_pack`). Both preserve bit-exact arithmetic with `causal_ops::unit`, `prompt_move::prompt_place`,
//! and `expert_ops::quantize_packed_with`.
use core::arch::asm;
use crate::causal_ops::{Arguments, Op, round, sigmoid};
use crate::tc_abi::{SwiGluPlacePack, SWIGLU_PLACE_PACK_THREADS};

/// Vectorized SwiGLU: `round(round(g * sigmoid(g)) * u)` with `g = gate.min(bound)` and `u = up.clamp(-bound, bound)`.
/// Separate tiny entry (~20 regs, 0 shared memory) instead of the 90-reg generic `causal_operation`.
#[no_mangle]
pub extern "ptx-kernel" fn causal_swiglu(p: Arguments) {
    // SAFETY: The host (`Backend::resident_causal_into`) checks `Shape::lengths` for `Op::SwiGlu`: `input[0]`, `input[1]`
    // and `output` each hold `shape.rows` contiguous FP32 words, inputs are immutable and disjoint from the private output,
    // and all owners stay live through stream completion. Vectorized 16-byte loads/stores run only when all three base
    // addresses are 16-byte aligned and only for full 4-word quads strictly below `rows / 4`; remaining elements run scalar.
    // Each output index `< rows` has exactly one writer. No shared memory, no barriers.
    unsafe {
        use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};
        let s = p.shape;
        if s.op != Op::SwiGlu || s.rows == 0 || !(s.bound > 0.0 && s.bound.is_finite()) {
            core::arch::nvptx::trap();
        }
        let r = s.rows as usize;
        let bound = s.bound;
        let tid = (_block_idx_x() as usize) * (_block_dim_x() as usize) + (_thread_idx_x() as usize);
        let stride = (_grid_dim_x() as usize) * (_block_dim_x() as usize);
        let g_ptr = p.input[0] as *const f32;
        let u_ptr = p.input[1] as *const f32;
        let out_ptr = p.output as *mut f32;
        let eval = |g: f32, u: f32| -> f32 {
            let g = g.min(bound);
            let u = u.clamp(-bound, bound);
            round(round(g * sigmoid(g)) * u)
        };
        if (p.input[0] | p.input[1] | p.output) & 15 == 0 {
            let vecs = r / 4;
            let mut v = tid;
            while v < vecs {
                let off = v * 4;
                let (g0, g1, g2, g3): (f32, f32, f32, f32);
                let (u0, u1, u2, u3): (f32, f32, f32, f32);
                asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                    a = out(reg32) g0, b = out(reg32) g1, c = out(reg32) g2, d = out(reg32) g3,
                    p = in(reg64) g_ptr.add(off), options(nostack, readonly));
                asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                    a = out(reg32) u0, b = out(reg32) u1, c = out(reg32) u2, d = out(reg32) u3,
                    p = in(reg64) u_ptr.add(off), options(nostack, readonly));
                let y0 = eval(g0, u0);
                let y1 = eval(g1, u1);
                let y2 = eval(g2, u2);
                let y3 = eval(g3, u3);
                asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                    p = in(reg64) out_ptr.add(off),
                    a = in(reg32) y0, b = in(reg32) y1, c = in(reg32) y2, d = in(reg32) y3, options(nostack));
                v += stride;
            }
            let tail = vecs * 4 + tid;
            if tail < r {
                *out_ptr.add(tail) = eval(*g_ptr.add(tail), *u_ptr.add(tail));
            }
        } else {
            let mut i = tid;
            while i < r {
                *out_ptr.add(i) = eval(*g_ptr.add(i), *u_ptr.add(i));
                i += stride;
            }
        }
    }
}

/// Fused routed `SwiGlu + ExpertPlaceTile(taps=1) + nv_quantize_pack`.
/// Reads channel-major expert gate/up (`[expert][local][expert_tokens]`), evaluates exact SwiGLU and finiteness check,
/// transposes 32 slots x 32 channels in shared memory, and quantizes directly into packed NVFP4 `(codes, scale_codes)`.
#[no_mangle]
pub extern "ptx-kernel" fn prompt_swiglu_place_pack(p: SwiGluPlacePack) {
    // SAFETY: The host (`Backend::nv_swiglu_place_pack_planes`) checks: `gate` and `up` hold `rows * cols` FP32 words,
    // `table` holds `4 * rows` FP32 words (built by `prompt_route_tables` as a permutation of `0..rows`), `activations`
    // holds `rows` FP32 words, `codes` holds `rows * cols / 2` bytes (8-byte aligned), `scales` holds `rows * cols / 16`
    // bytes, `cols > 0` and `cols % 64 == 0`, `bound > 0` finite, all owners disjoint and live through stream completion.
    // The block must have `SWIGLU_PLACE_PACK_THREADS` (256) threads (else trap). Grid `(ceil(rows/32), cols/32)` assigns
    // each block a disjoint 32-slot x 32-channel tile. Table entries are data and checked before use (`row < rows`,
    // `earlier + at == slot`, `source < rows * cols`, else trap). Shared memory is this kernel's static 4,480-byte array
    // (`32*33` float tile + `32` row indices + `32` activation scales). Every thread reaches both barriers and every warp
    // shuffle executes unconditionally across all 32 lanes (`mask = -1`). Each output code pair and scale byte has one writer.
    unsafe {
        use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _grid_dim_x, _grid_dim_y, _thread_idx_x};
        if _block_dim_x() != SWIGLU_PLACE_PACK_THREADS
            || p.rows == 0 || p.cols == 0 || p.cols % 64 != 0
            || !(p.bound > 0.0 && p.bound.is_finite())
            || (p.gate | p.up | p.table | p.activations | p.scales) & 3 != 0
            || p.codes & 7 != 0
        {
            core::arch::nvptx::trap();
        }
        let (rows, cols) = (p.rows as usize, p.cols as usize);
        let blocks_per_row = cols / 16;
        let check_finite = p.check_finite != 0;
        let bound = p.bound;
        let tid = _thread_idx_x() as usize;
        let (tx, ty) = (tid % 32, tid / 32);
        let (blk, sub) = (tid / 4, tid % 4);
        let (local_s, half) = (blk / 2, blk % 2);
        let gate = p.gate as *const f32;
        let up = p.up as *const f32;
        let table = p.table as *const f32;
        let acts = p.activations as *const f32;
        let codes = p.codes as *mut u32;
        let scales = p.scales as *mut u8;
        let smem = crate::ptx::shared::<4864>();
        const ROWS_AT: u32 = 4224;
        const ACTS_AT: u32 = 4352;
        const EARLIER_AT: u32 = 4480;
        const TOKENS_AT: u32 = 4608;
        const AT_AT: u32 = 4736;
        let slot_tiles = rows.div_ceil(32);
        let chan_tiles = cols / 32;
        let total = rows * cols;
        let mut st = _block_idx_x() as usize;
        while st < slot_tiles {
            let s0 = st * 32;
            let slot = s0 + tx;
            // Stage routing metadata and activation scales ONCE per 32-slot tile across all channel tiles (ty == 0).
            if ty == 0 && slot < rows {
                let row = *table.add(3 * rows + slot) as usize;
                if row >= rows { core::arch::nvptx::trap(); }
                let e = table.add(3 * row);
                let earlier = *e as usize;
                let tokens = *e.add(1) as usize;
                let at = *e.add(2) as usize;
                if earlier + at != slot { core::arch::nvptx::trap(); }
                let act = *acts.add(row);
                let off = (tx as u32) * 4;
                asm!("st.shared.u32 [{a}], {v};", a = in(reg32) (smem + ROWS_AT + off), v = in(reg32) row as u32, options(nostack));
                asm!("st.shared.f32 [{a}], {v};", a = in(reg32) (smem + ACTS_AT + off), v = in(reg32) act, options(nostack));
                asm!("st.shared.u32 [{a}], {v};", a = in(reg32) (smem + EARLIER_AT + off), v = in(reg32) earlier as u32, options(nostack));
                asm!("st.shared.u32 [{a}], {v};", a = in(reg32) (smem + TOKENS_AT + off), v = in(reg32) tokens as u32, options(nostack));
                asm!("st.shared.u32 [{a}], {v};", a = in(reg32) (smem + AT_AT + off), v = in(reg32) at as u32, options(nostack));
            }
            asm!("bar.sync 0;");
            let (earlier, tokens, at) = if slot < rows {
                let off = (tx as u32) * 4;
                let (e0, t0, a0): (u32, u32, u32);
                asm!("ld.shared.u32 {v}, [{a}];", v = out(reg32) e0, a = in(reg32) (smem + EARLIER_AT + off), options(nostack, readonly));
                asm!("ld.shared.u32 {v}, [{a}];", v = out(reg32) t0, a = in(reg32) (smem + TOKENS_AT + off), options(nostack, readonly));
                asm!("ld.shared.u32 {v}, [{a}];", v = out(reg32) a0, a = in(reg32) (smem + AT_AT + off), options(nostack, readonly));
                (e0 as usize, t0 as usize, a0 as usize)
            } else { (0, 0, 0) };
            let mut ct = _block_idx_y() as usize;
            while ct < chan_tiles {
                let c0 = ct * 32;
                if slot < rows {
                    let base_src = earlier * cols + at;
                    let mut j = 0;
                    while j < 4 {
                        let ch = c0 + ty + 8 * j;
                        let source = base_src + ch * tokens;
                        if source >= total { core::arch::nvptx::trap(); }
                        let (gr, ur): (f32, f32);
                        asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) gr, p = in(reg64) gate.add(source), options(nostack, readonly));
                        asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) ur, p = in(reg64) up.add(source), options(nostack, readonly));
                        let g = gr.min(bound);
                        let u = ur.clamp(-bound, bound);
                        let v = round(round(g * sigmoid(g)) * u);
                        if check_finite && (v.to_bits() & 0x7f80_0000 == 0x7f80_0000) {
                            core::arch::nvptx::trap();
                        }
                        asm!("st.shared.f32 [{a}], {v};",
                            a = in(reg32) (smem + (((ty + 8 * j) * 33 + tx) as u32) * 4),
                            v = in(reg32) v, options(nostack));
                        j += 1;
                    }
                }
                asm!("bar.sync 0;");
                let live = s0 + local_s < rows;
                let mut row_idx = 0u32;
                let mut activation = 1.0f32;
                let mut v = [0.0f32; 4];
                let mut maximum = 0.0f32;
                if live {
                    asm!("ld.shared.u32 {v}, [{a}];", v = out(reg32) row_idx, a = in(reg32) (smem + ROWS_AT + (local_s as u32) * 4), options(nostack));
                    asm!("ld.shared.f32 {v}, [{a}];", v = out(reg32) activation, a = in(reg32) (smem + ACTS_AT + (local_s as u32) * 4), options(nostack));
                    let c_base = half * 16 + sub * 4;
                    let mut k = 0;
                    while k < 4 {
                        let val: f32;
                        asm!("ld.shared.f32 {v}, [{a}];",
                            v = out(reg32) val,
                            a = in(reg32) (smem + (((c_base + k) * 33 + local_s) as u32) * 4), options(nostack));
                        v[k] = val;
                        maximum = maximum.max(crate::expert_ops::round(val).abs());
                        k += 1;
                    }
                }
                let other2: f32;
                asm!("shfl.sync.bfly.b32 {o}, {v}, 2, 31, -1;", o = out(reg32) other2, v = in(reg32) maximum, options(nomem, nostack));
                maximum = maximum.max(other2);
                let other1: f32;
                asm!("shfl.sync.bfly.b32 {o}, {v}, 1, 31, -1;", o = out(reg32) other1, v = in(reg32) maximum, options(nomem, nostack));
                maximum = maximum.max(other1);
                let mut nib = 0u32;
                let mut scale_code = 0u32;
                if live {
                    let scale = crate::expert_ops::nearest_e4m3((maximum / activation) / 6.0);
                    let block = crate::expert_ops::e4m3(scale);
                    scale_code = u32::from(scale & 0x7f);
                    let mut k = 0;
                    while k < 4 {
                        let val = v[k];
                        let code = if block == 0.0 { 0 } else { crate::expert_ops::nearest_e2m1((crate::expert_ops::round(val).abs() / activation) / block) };
                        let q = code | if val < 0.0 { 8 } else { 0 };
                        nib |= u32::from(q) << (4 * k);
                        k += 1;
                    }
                }
                let partner: u32;
                asm!("shfl.sync.bfly.b32 {o}, {v}, 1, 31, -1;", o = out(reg32) partner, v = in(reg32) nib, options(nomem, nostack));
                let word = nib | (partner << 16);
                let hi: u32;
                asm!("shfl.sync.bfly.b32 {o}, {v}, 2, 31, -1;", o = out(reg32) hi, v = in(reg32) word, options(nomem, nostack));
                if live && sub == 0 {
                    let block_col = ct * 2 + half;
                    let dst = codes.add(row_idx as usize * (blocks_per_row * 2) + block_col * 2);
                    asm!("st.global.v2.u32 [{p}], {{{a},{b}}};", p = in(reg64) dst, a = in(reg32) word, b = in(reg32) hi, options(nostack));
                    *scales.add(row_idx as usize * blocks_per_row + block_col) = scale_code as u8;
                }
                asm!("bar.sync 0;");
                ct += _grid_dim_y() as usize;
            }
            st += _grid_dim_x() as usize;
        }
    }
}
