//! Grouped prompt softmax: 1 warp (32 threads) per (token, head) with align-aware 128-bit loads/stores for 100% of
//! heads (even when `cols` is odd, e.g. 2051!), 8,240-byte shared memory staging (`count <= 2052`, 12 CTAs/SM, 0
//! `__syncthreads`, ~40 regs), and optional packed BF16 output (`Op::AbsorbSoftmaxPacked`, 16-byte aligned per head,
//! halving DRAM write & values-kernel read traffic). Preserves exact max, exact `exp(v - m)`, exact ascending `0..count`
//! FP32 denominator sum, and exact BF16 round.
use core::arch::asm;
use crate::prompt_ops::{Arguments, Op};
use crate::causal_ops::round;
use crate::next_math::exp;

const THREADS: usize = 32;
const MAX_CACHED: usize = 2052;

#[export_name="prompt_softmax_rows"]
pub extern "ptx-kernel" fn entry_prompt_softmax_rows(p:Arguments) {prompt_softmax_rows(p)}
#[inline(always)]
pub(crate) fn prompt_softmax_rows(p:Arguments) {
    // SAFETY: Shape::lengths and softmax_rows/latent_absorb_rows bound all extents on the host: input[0] holds
    // tokens*heads*cols score words, input[1] holds tokens*(slots+1) selection words with count at offset `slots`,
    // and output holds tokens*heads*cols words (unpacked) or tokens*heads*prob_words(cols) u32 words (`AbsorbSoftmaxPacked`,
    // cols <= MAX_CACHED, 16-byte aligned output base). A live count above cols traps. The block has 32 threads
    // (1 complete warp); all 32 lanes reach every warp shuffle and syncwarp uniformly because `count` and `head` are
    // uniform per block. For `count <= MAX_CACHED`, shared memory (2060 words = 8,240 bytes) stages raw scores shifted
    // by `mis = ((row >> 2) & 3)` so every 16-byte aligned vector load from `row` aligns with shared memory for 100% of
    // heads regardless of odd `cols`. Larger counts keep the global multi-pass unpacked path using `output` as scratch.
    unsafe {
        let s = p.shape;
        if core::arch::nvptx::_block_dim_x() as usize != THREADS { core::arch::nvptx::trap(); }
        let packed_out = s.op == Op::AbsorbSoftmaxPacked;
        let heads_per_token = s.heads as usize;
        let count_stride = s.slots as usize + 1;
        let heads = s.tokens as usize * heads_per_token;
        let r = s.cols as usize;
        if packed_out && (r > MAX_CACHED || ((p.output as usize) & 15) != 0) { core::arch::nvptx::trap(); }
        let pw = crate::absorb_tc::prob_words(r);
        let x0 = p.input[0] as *const f32;
        let x1 = (p.input[1] as *const f32).add(s.slots as usize);
        let lane = core::arch::nvptx::_thread_idx_x() as usize;
        let smem = crate::ptx::shared::<{(MAX_CACHED + 8) * 4}>();
        let mut head = core::arch::nvptx::_block_idx_x() as usize;
        let stride = core::arch::nvptx::_grid_dim_x() as usize;
        while head < heads {
            let count = *x1.add((head / heads_per_token) * count_stride) as usize;
            if count > r { core::arch::nvptx::trap(); }
            let row = x0.add(head * r);
            if !packed_out {
                let out = (p.output as *mut f32).add(head * r);
                let mut i = count + lane;
                while i < r { *out.add(i) = 0.0; i += THREADS; }
            }
            if count > 0 && count <= MAX_CACHED {
                let mis = ((row as usize) >> 2) & 3;
                let prefix = ((4 - mis) & 3).min(count);
                let rem = count - prefix;
                let quads = rem / 4;
                let mut m = f32::NEG_INFINITY;
                if lane < prefix {
                    let v: f32;
                    asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) row.add(lane), options(nostack, readonly));
                    m = m.max(v);
                    crate::ptx::store(smem + ((lane + mis) * 4) as u32, v);
                }
                let mut q = lane;
                while q < quads {
                    let off = prefix + q * 4;
                    let (a, b, c, d): (f32, f32, f32, f32);
                    asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                        a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
                        p = in(reg64) row.add(off), options(nostack, readonly));
                    m = m.max(a).max(b).max(c).max(d);
                    asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                        p = in(reg32) smem + ((off + mis) * 4) as u32,
                        a = in(reg32) a, b = in(reg32) b, c = in(reg32) c, d = in(reg32) d, options(nostack));
                    q += THREADS;
                }
                let tail = prefix + quads * 4 + lane;
                if tail < count {
                    let v: f32;
                    asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) row.add(tail), options(nostack, readonly));
                    m = m.max(v);
                    crate::ptx::store(smem + ((tail + mis) * 4) as u32, v);
                }
                let mut distance = 16u32;
                while distance != 0 {
                    let other: f32;
                    asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) other, v = in(reg32) m, d = in(reg32) distance, options(nomem, nostack));
                    m = m.max(other);
                    distance /= 2;
                }
                if lane < prefix {
                    let v = crate::ptx::load(smem + ((lane + mis) * 4) as u32);
                    crate::ptx::store(smem + ((lane + mis) * 4) as u32, exp(v - m));
                }
                let mut q = lane;
                while q < quads {
                    let soff = ((prefix + q * 4 + mis) * 4) as u32;
                    let v = crate::ptx::load4(smem + soff);
                    let e0 = exp(f32::from_bits(v[0]) - m);
                    let e1 = exp(f32::from_bits(v[1]) - m);
                    let e2 = exp(f32::from_bits(v[2]) - m);
                    let e3 = exp(f32::from_bits(v[3]) - m);
                    asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                        p = in(reg32) smem + soff, a = in(reg32) e0, b = in(reg32) e1, c = in(reg32) e2, d = in(reg32) e3, options(nostack));
                    q += THREADS;
                }
                if tail < count {
                    let v = crate::ptx::load(smem + ((tail + mis) * 4) as u32);
                    crate::ptx::store(smem + ((tail + mis) * 4) as u32, exp(v - m));
                }
                asm!("bar.warp.sync -1;");
                let mut denom = 0.0f32;
                if lane == 0 {
                    let mut at = 0usize;
                    while at < prefix {
                        denom += crate::ptx::load(smem + ((at + mis) * 4) as u32);
                        at += 1;
                    }
                    while at + 4 <= count {
                        let v = crate::ptx::load4(smem + ((at + mis) * 4) as u32);
                        denom += f32::from_bits(v[0]);
                        denom += f32::from_bits(v[1]);
                        denom += f32::from_bits(v[2]);
                        denom += f32::from_bits(v[3]);
                        at += 4;
                    }
                    while at < count {
                        denom += crate::ptx::load(smem + ((at + mis) * 4) as u32);
                        at += 1;
                    }
                }
                let denominator: f32;
                asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;", o = out(reg32) denominator, v = in(reg32) denom, options(nomem, nostack));
                if packed_out {
                    let out_u32 = (p.output as *mut u32).add(head * pw);
                    let pairs = count.div_ceil(2);
                    let full_octs = count / 8;
                    let mut oq = lane;
                    while oq < full_octs {
                        let base = oq * 8 + mis;
                        let y0 = round(crate::ptx::load(smem + (base * 4) as u32) / denominator);
                        let y1 = round(crate::ptx::load(smem + ((base + 1) * 4) as u32) / denominator);
                        let y2 = round(crate::ptx::load(smem + ((base + 2) * 4) as u32) / denominator);
                        let y3 = round(crate::ptx::load(smem + ((base + 3) * 4) as u32) / denominator);
                        let y4 = round(crate::ptx::load(smem + ((base + 4) * 4) as u32) / denominator);
                        let y5 = round(crate::ptx::load(smem + ((base + 5) * 4) as u32) / denominator);
                        let y6 = round(crate::ptx::load(smem + ((base + 6) * 4) as u32) / denominator);
                        let y7 = round(crate::ptx::load(smem + ((base + 7) * 4) as u32) / denominator);
                        let w0 = (y0.to_bits() >> 16) | (y1.to_bits() & 0xffff_0000);
                        let w1 = (y2.to_bits() >> 16) | (y3.to_bits() & 0xffff_0000);
                        let w2 = (y4.to_bits() >> 16) | (y5.to_bits() & 0xffff_0000);
                        let w3 = (y6.to_bits() >> 16) | (y7.to_bits() & 0xffff_0000);
                        asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",
                            p = in(reg64) out_u32.add(oq * 4),
                            a = in(reg32) w0, b = in(reg32) w1, c = in(reg32) w2, d = in(reg32) w3, options(nostack));
                        oq += THREADS;
                    }
                    let pair = full_octs * 4 + lane;
                    if pair < pairs {
                        let i0 = pair * 2;
                        let y0 = round(crate::ptx::load(smem + ((i0 + mis) * 4) as u32) / denominator);
                        let y1 = if i0 + 1 < count { round(crate::ptx::load(smem + ((i0 + 1 + mis) * 4) as u32) / denominator) } else { 0.0f32 };
                        *out_u32.add(pair) = (y0.to_bits() >> 16) | (y1.to_bits() & 0xffff_0000);
                    }
                } else {
                    let out = (p.output as *mut f32).add(head * r);
                    if ((out as usize) & 15) == ((row as usize) & 15) {
                        if lane < prefix {
                            *out.add(lane) = round(crate::ptx::load(smem + ((lane + mis) * 4) as u32) / denominator);
                        }
                        let mut q = lane;
                        while q < quads {
                            let off = prefix + q * 4;
                            let v = crate::ptx::load4(smem + ((off + mis) * 4) as u32);
                            let y0 = round(f32::from_bits(v[0]) / denominator);
                            let y1 = round(f32::from_bits(v[1]) / denominator);
                            let y2 = round(f32::from_bits(v[2]) / denominator);
                            let y3 = round(f32::from_bits(v[3]) / denominator);
                            asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                                p = in(reg64) out.add(off), a = in(reg32) y0, b = in(reg32) y1, c = in(reg32) y2, d = in(reg32) y3, options(nostack));
                            q += THREADS;
                        }
                        if tail < count {
                            *out.add(tail) = round(crate::ptx::load(smem + ((tail + mis) * 4) as u32) / denominator);
                        }
                    } else {
                        let mut idx = lane;
                        while idx < count {
                            *out.add(idx) = round(crate::ptx::load(smem + ((idx + mis) * 4) as u32) / denominator);
                            idx += THREADS;
                        }
                    }
                }
                asm!("bar.warp.sync -1;");
            } else if count > 0 {
                let out = (p.output as *mut f32).add(head * r);
                let mut m = f32::NEG_INFINITY;
                let mut i = lane;
                while i < count {
                    let v: f32;
                    asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) row.add(i), options(nostack, readonly));
                    m = m.max(v);
                    i += THREADS;
                }
                let mut distance = 16u32;
                while distance != 0 {
                    let other: f32;
                    asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) other, v = in(reg32) m, d = in(reg32) distance, options(nomem, nostack));
                    m = m.max(other);
                    distance /= 2;
                }
                i = lane;
                while i < count {
                    let v: f32;
                    asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) row.add(i), options(nostack, readonly));
                    *out.add(i) = exp(v - m);
                    i += THREADS;
                }
                asm!("bar.warp.sync -1;");
                let mut denom = 0.0f32;
                if lane == 0 {
                    let mut at = 0usize;
                    while at < count { denom += *out.add(at); at += 1; }
                }
                let denominator: f32;
                asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;", o = out(reg32) denominator, v = in(reg32) denom, options(nomem, nostack));
                i = lane;
                while i < count { *out.add(i) = round(*out.add(i) / denominator); i += THREADS; }
                asm!("bar.warp.sync -1;");
            }
            head += stride;
        }
    }
}
