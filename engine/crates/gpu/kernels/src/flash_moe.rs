//! Flash routing and ordered reduction of rank-sharded expert partials.
use crate::{index, layer::{maximum, sum}, shuffle_xor};

#[inline(always)]
unsafe fn bf16(value: f32) -> f32 {
    let bits: u16;
    core::arch::asm!("cvt.rn.bf16.f32 {out}, {input};", out = out(reg16) bits, input = in(reg32) value);
    f32::from_bits((bits as u32) << 16)
}

/// Shared-expert gate is row major; one CTA owns a complete independent token.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_shared(p: crate::kernel_params::HybridSharedExpert) {
    let token = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    if token >= p.tokens as usize { return; }
    let h = p.hidden as usize;
    let input = (p.input as *const f32).add(token * h);
    let gate = p.gate as *const u16;
    let mut dot = 0.0;
    for col in (tid % 32..h).step_by(32) {
        dot += *input.add(col) * f32::from_bits((*gate.add(col) as u32) << 16);
    }
    let sigmoid = bf16(1.0 / (1.0 + exp(-bf16(sum(dot)))));
    let shared = (p.shared as *const f32).add(token * h);
    let output = (p.output as *mut f32).add(token * h);
    for col in (tid..h).step_by(128) {
        *output.add(col) = bf16(*output.add(col) + bf16(bf16(*shared.add(col)) * sigmoid));
    }
}

/// FP64 range reduction avoids the ex2.approx rounding used by fast attention
/// paths. Fourteen Taylor terms on [-ln(2)/2,ln(2)/2] retain FP32 exp accuracy.
#[inline(always)]
pub(crate) fn exp(value: f32) -> f32 {
    if value > 88.72284 { return f32::INFINITY; }
    if value < -103.97208 { return 0.0; }
    let x = value as f64;
    let exponent = x * core::f64::consts::LOG2_E;
    let n = (exponent + if exponent >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x - n as f64 * core::f64::consts::LN_2;
    let mut polynomial = 1.0 / 87178291200.0;
    polynomial = 1.0 / 6227020800.0 + r * polynomial;
    polynomial = 1.0 / 479001600.0 + r * polynomial;
    polynomial = 1.0 / 39916800.0 + r * polynomial;
    polynomial = 1.0 / 3628800.0 + r * polynomial;
    polynomial = 1.0 / 362880.0 + r * polynomial;
    polynomial = 1.0 / 40320.0 + r * polynomial;
    polynomial = 1.0 / 5040.0 + r * polynomial;
    polynomial = 1.0 / 720.0 + r * polynomial;
    polynomial = 1.0 / 120.0 + r * polynomial;
    polynomial = 1.0 / 24.0 + r * polynomial;
    polynomial = 1.0 / 6.0 + r * polynomial;
    polynomial = 0.5 + r * polynomial;
    polynomial = 1.0 + r * polynomial;
    polynomial = 1.0 + r * polynomial;
    (polynomial * f64::from_bits(((n + 1023) as u64) << 52)) as f32
}

// Split exponential table and implementation.
/// 2^(j/64) as an f32 pair hi, lo (j = 0..63).
static EXP2_64: [u32; 128] = [
    0x3f800000, 0x00000000, 0x3f8164d2, 0xb1c43fd0, 0x3f82cd87, 0xb34ea7a9, 0x3f843a29, 0xb2f14c87,
    0x3f85aac3, 0x334f9891, 0x3f871f62, 0xb352c2e6, 0x3f88980f, 0xb37eda4b, 0x3f8a14d5, 0x336a92de,
    0x3f8b95c2, 0xb260aba1, 0x3f8d1adf, 0x3336fcb7, 0x3f8ea43a, 0xb3697465, 0x3f9031dc, 0x330628cd,
    0x3f91c3d3, 0x33675624, 0x3f935a2b, 0x32bc4f9c, 0x3f94f4f0, 0xb32e0212, 0x3f96942d, 0x32dc8061,
    0x3f9837f0, 0x33231b71, 0x3f99e046, 0xb359be90, 0x3f9b8d3a, 0xb30c5563, 0x3f9d3eda, 0xb331a601,
    0x3f9ef532, 0x33412342, 0x3fa0b051, 0x31fb9715, 0x3fa27043, 0x30c3125a, 0x3fa43516, 0xb323ec33,
    0x3fa5fed7, 0xb32c9d5e, 0x3fa7cd94, 0xb3162d36, 0x3fa9a15b, 0xb3162b08, 0x3fab7a3a, 0xb314ad82,
    0x3fad583f, 0xb22deaf6, 0x3faf3b79, 0xb3252deb, 0x3fb123f6, 0xb37c5aa8, 0x3fb311c4, 0x32154889,
    0x3fb504f3, 0x32cfe77a, 0x3fb6fd92, 0xb266b974, 0x3fb8fbaf, 0x330ec5f7, 0x3fbaff5b, 0xb31bd983,
    0x3fbd08a4, 0xb3414fe8, 0x3fbf179a, 0xb3130b1a, 0x3fc12c4d, 0xb2d6663e, 0x3fc346cd, 0xb2976da2,
    0x3fc5672a, 0x320aa837, 0x3fc78d75, 0xb2dd5119, 0x3fc9b9be, 0xb37323a2, 0x3fcbec15, 0xb006c6c2,
    0x3fce248c, 0x3228fc24, 0x3fd06334, 0xb2944353, 0x3fd2a81e, 0xb35c1daa, 0x3fd4f35b, 0xb3286024,
    0x3fd744fd, 0xb2d4a58a, 0x3fd99d16, 0xb2f61d41, 0x3fdbfbb8, 0xb3504a1c, 0x3fde60f5, 0xb37b43e3,
    0x3fe0ccdf, 0xb21eab59, 0x3fe33f89, 0x33657d15, 0x3fe5b907, 0xb2441be6, 0x3fe8396a, 0x33207898,
    0x3feac0c7, 0xb24116de, 0x3fed4f30, 0x3276cca1, 0x3fefe4ba, 0xb348464a, 0x3ff28177, 0x32f167ff,
    0x3ff5257d, 0x32292436, 0x3ff7d0df, 0x336615a2, 0x3ffa83b3, 0xb2923758, 0x3ffd3e0c, 0x31cf486c,
];
/// `exp` bit for bit, in FP32 (proved on all 2^32 inputs). value = (64 n + j) ln2/64 + r with
/// |r| <= ln2/128 (r as a float pair, ln2/64 in three parts); 2^(j/64) e^r as a float pair
/// (degree-5 Taylor, pair error < 2^-44 relative; `exp`'s own FP64 error is below that). When the
/// pair +- 2^-42 of itself rounds to one f32, that f32 is `exp`'s; otherwise (about 1 input in
/// 10^5) and outside [-87, 88] (results not normal, infinity, NaN) it returns `exp`.
#[inline(always)]
pub(crate) fn exp_split(value: f32) -> f32 {
    if value >= -87.0 && value <= 88.0 {
        let fma = core::f32::math::mul_add;
        let t = value * f32::from_bits(0x42b8aa3b);
        let shifted = t + 12582912.0;
        let k = shifted.to_bits() as i32 - 0x4b40_0000;
        let (n, j, kf) = (k >> 6, (k & 63) as usize, shifted - 12582912.0);
        // k = t rounded to an integer by the 1.5 * 2^23 shift (|t| < 2^13); k * C_HI is exact
        // (C_HI has 9 bits) and so is r1.
        let r1 = fma(-kf, f32::from_bits(0x3c318000), value);
        let p = kf * f32::from_bits(0xb65e8083);
        let pe = fma(kf, f32::from_bits(0xb65e8083), -p);
        let rh = r1 - p;
        let bb = rh - r1;
        let rl = fma(-kf, f32::from_bits(0x28e7bcd6), ((r1 - (rh - bb)) + (-p - bb)) - pe);
        let q = rh * rh;
        let t3 = rh * q * fma(rh, fma(rh, 1.0 / 120.0, 1.0 / 24.0), 1.0 / 6.0);
        let small = rl + rh * rl + 0.5 * fma(rh, rh, -q) + t3;
        let a1 = 1.0 + rh;
        let h = 0.5 * q;
        let a2 = a1 + h;
        let lo = (rh - (a1 - 1.0)) + (h - (a2 - a1)) + small;
        let (th, tl) = unsafe { (f32::from_bits(*EXP2_64.get_unchecked(2 * j)), f32::from_bits(*EXP2_64.get_unchecked(2 * j + 1))) };
        let zh = th * a2;
        let zl = fma(th, a2, -zh) + fma(th, lo, tl * a2);
        let d = zh * f32::from_bits(0x2a800000);
        let below = zh + (zl - d);
        if below == zh + (zl + d) {
            return below * f32::from_bits(((n + 127) as u32) << 23);
        }
    }
    exp(value)
}
// exp_split: end

/// Full 512-way FP32 softmax followed by top ten and optional renormalization.
/// One CTA per token computes exps; its first warp keeps the original sums.
/// Output [tokens,10] in descending probability order.
/// Equal probabilities choose the lower expert ID; the caller must reject a
/// tied oracle row whose torch.topk order differs from that deterministic order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_route(p: crate::kernel_params::FlashMoeRoute) {
    let token = core::arch::nvptx::_block_idx_x() as usize;
    let lane = core::arch::nvptx::_thread_idx_x() as usize;
    if token >= p.tokens as usize { return; }
    route_token::<false, false>(p, token, lane);
}
/// flash_moe_route with each exp as `exp_split` (the same f32 bits, FP32 instead of FP64), on
/// route_token's WIDE copy (inline max.f32 / shfl, the same values at 128 threads): the plain copy
/// with exp_split perturbed flash_next_rows_feed's and qsa_fp32_select's PTX.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_route_fast(p: crate::kernel_params::FlashMoeRoute) {
    let token = core::arch::nvptx::_block_idx_x() as usize;
    let lane = core::arch::nvptx::_thread_idx_x() as usize;
    if token >= p.tokens as usize { return; }
    route_token::<true, true>(p, token, lane);
}
/// The scan and butterfly above on one u64 key per candidate: value bits + 1 (0 when taken, the
/// scan's -inf) high, !candidate low. The values are finite and >= +0.0 or -inf (taken; exp and
/// a positive denominator never give -0.0, NaN or inf), so the keys' integer order is "greater
/// value, then lower candidate", a total order: a pairwise tree per lane (depth 4) and the warp
/// max of the keys return the scan's (best, expert) with integer compares only.
#[inline(always)]
unsafe fn warp_best(values: &[f32; 16], taken: u32, lane: usize) -> (f32, u32) {
    let c = |j: usize| (if taken & (1 << j) != 0 { 0 } else { (values[j].to_bits() as u64 + 1) << 32 }) | !((lane + j * 32) as u32) as u64;
    let m = |a: u64, b: u64| if b > a { b } else { a };
    let q = |j: usize| m(m(c(j), c(j + 1)), m(c(j + 2), c(j + 3)));
    let key = m(m(q(0), q(4)), m(q(8), q(12)));
    // Warp max of the keys: the high words' max, then the low words' max over the lanes holding
    // it (0 elsewhere; a low word !candidate is never 0), i.e. the u64 max.
    let high = wredux_max((key >> 32) as u32);
    let low = wredux_max(if (key >> 32) as u32 == high { key as u32 } else { 0 });
    (if high == 0 { f32::NEG_INFINITY } else { f32::from_bits(high - 1) }, !low)
}
/// redux.sync.max.u32 over the full warp (sm_80+): every lane gets the maximum.
#[inline(always)]
unsafe fn wredux_max(value: u32) -> u32 {
    let max: u32;
    core::arch::asm!("redux.sync.max.u32 {m}, {v}, -1;", m = out(reg32) max, v = in(reg32) value);
    max
}
/// route_token::<true>'s warp exchange: the same shfl.sync.bfly (all lanes, clamp 31) as the
/// llvm.nvvm.shfl.sync.bfly intrinsic behind layer::other and shuffle_xor, as inline asm.
#[inline(always)]
unsafe fn wshfl(value: u32, distance: u32) -> u32 {
    let other: u32;
    core::arch::asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) other, v = in(reg32) value, d = in(reg32) distance);
    other
}
/// f32::max as the max.f32 it lowers to (maxNum: a NaN operand yields the other).
#[inline(always)]
unsafe fn wmax(a: f32, b: f32) -> f32 {
    let m: f32;
    core::arch::asm!("max.f32 {m}, {a}, {b};", m = out(reg32) m, a = in(reg32) a, b = in(reg32) b);
    m
}
/// layer::maximum / layer::sum (same butterfly order) on wshfl and wmax.
#[inline(always)]
unsafe fn wmaximum(mut value: f32) -> f32 {
    let mut distance = 16;
    while distance != 0 { value = wmax(value, f32::from_bits(wshfl(value.to_bits(), distance))); distance /= 2; }
    value
}
#[inline(always)]
unsafe fn wsum(mut value: f32) -> f32 {
    let mut distance = 16;
    while distance != 0 { value += f32::from_bits(wshfl(value.to_bits(), distance)); distance /= 2; }
    value
}
/// flash_moe_route's body for one token; `WIDE`: a block of more than 128 threads
/// (flash_next_route_compact1), whose threads >= 128 skip the exps and the selection
/// but reach every block barrier. The WIDE copy takes f32::max as wmax (a new caller of
/// f32::max perturbed five other entries' PTX) and its warp exchanges as wshfl. `FAST`: each
/// exp as `exp_split`.
#[inline(always)]
pub(crate) unsafe fn route_token<const WIDE: bool, const FAST: bool>(p: crate::kernel_params::FlashMoeRoute, token: usize, lane: usize) {
    let logits = (p.logits as *const f32).add(token * 512);
    let ids = (p.ids as *mut u32).add(token * 10);
    let weights = (p.weights as *mut f32).add(token * 10);
    let mut values = [0.0f32; 16];
    let mut highest = f32::NEG_INFINITY;
    let mut invalid = 0.0f32;
    let mut j = 0;
    let sh = crate::ptx::shared::<2056>();
    if lane < 32 {
        while j < 16 {
            values[j] = *logits.add(lane + 32 * j);
            if !values[j].is_finite() { invalid = 1.0; }
            highest = if WIDE { wmax(highest, values[j]) } else { highest.max(values[j]) };
            j += 1;
        }
        invalid = if WIDE { wmaximum(invalid) } else { maximum(invalid) };
        highest = if WIDE { wmaximum(highest) } else { maximum(highest) };
        if lane == 0 {
            crate::ptx::store(sh + 2048, highest);
            crate::ptx::store(sh + 2052, invalid);
        }
    }
    crate::ptx::sync();
    if crate::ptx::load(sh + 2052) != 0.0 {
        if lane < 10 { *ids.add(lane) = u32::MAX; *weights.add(lane) = f32::NAN; }
        return;
    }
    highest = crate::ptx::load(sh + 2048);
    // Each exp has the identical f32 subtraction and FP64 polynomial. Shared
    // storage transfers its f32 bits; no reduction or approximation is added.
    if !WIDE || lane < 128 {
        for expert in (lane..512).step_by(128) {
            if FAST { crate::ptx::store(sh + (expert * 4) as u32, exp_split(*logits.add(expert) - highest)); continue; }
            crate::ptx::store(sh + (expert * 4) as u32, exp(*logits.add(expert) - highest));
        }
    }
    crate::ptx::sync();
    if lane >= 32 { return; }
    let mut denominator = 0.0;
    j = 0;
    while j < 16 {
        values[j] = crate::ptx::load(sh + ((lane + 32 * j) * 4) as u32);
        denominator += values[j];
        j += 1;
    }
    denominator = if WIDE { wsum(denominator) } else { sum(denominator) };
    if !denominator.is_finite() || denominator <= 0.0 {
        if lane < 10 { *ids.add(lane) = u32::MAX; *weights.add(lane) = f32::NAN; }
        return;
    }
    j = 0;
    while j < 16 {
        values[j] /= denominator;
        j += 1;
    }
    // Lane `slot` keeps that slot's probability, and bit j of `taken` marks
    // this lane's candidate j as selected (it then compares as -inf, as when
    // it was overwritten): no dynamically indexed array, values stay in registers.
    let mut selected = 0.0f32;
    let mut taken = 0u32;
    let mut selected_sum = 0.0;
    let mut slot = 0;
    while slot < 10 {
        let mut best = f32::NEG_INFINITY;
        let mut expert = u32::MAX;
        if FAST { (best, expert) = warp_best(&values, taken, lane); } else {
        j = 0;
        while j < 16 {
            let candidate = (lane + j * 32) as u32;
            let value = if taken & (1 << j) != 0 { f32::NEG_INFINITY } else { values[j] };
            if value > best || (value == best && candidate < expert) {
                best = value;
                expert = candidate;
            }
            j += 1;
        }
        let mut distance = 16;
        while distance != 0 {
            let other = f32::from_bits(if WIDE { wshfl(best.to_bits(), distance) } else { shuffle_xor(u32::MAX, best.to_bits(), distance, 31) });
            let other_id = if WIDE { wshfl(expert, distance) } else { shuffle_xor(u32::MAX, expert, distance, 31) };
            if other > best || (other == best && other_id < expert) {
                best = other;
                expert = other_id;
            }
            distance /= 2;
        }
        }
        if expert >= 512 {
            if lane == 0 {
                j = 0;
                while j < 10 { *ids.add(j) = u32::MAX; *weights.add(j) = f32::NAN; j += 1; }
            }
            return;
        }
        if expert as usize % 32 == lane { taken |= 1 << (expert / 32); }
        if lane == slot { selected = best; }
        selected_sum += best;
        if lane == 0 { *ids.add(slot) = expert; }
        slot += 1;
    }
    if lane < 10 {
        *weights.add(lane) = selected / if p.norm_topk != 0 { selected_sum } else { 1.0 };
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_silu(gate: *mut f32, up: *const f32, n: u32) {
    silu_at(gate, up, n, index());
}
/// flash_moe_silu with exp as `exp_split` (the same f32 bits; the prompt's shared expert).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_silu_fast(gate: *mut f32, up: *const f32, n: u32) {
    let i = index();
    if i < n as usize {
        let value = *gate.add(i);
        *gate.add(i) = (value / (1.0 + exp_split(-value))) * *up.add(i);
    }
}
/// flash_moe_silu's element `i` (also run by flash_next_compact_route_silu's CTAs >= 1).
#[inline(always)]
pub(crate) unsafe fn silu_at(gate: *mut f32, up: *const f32, n: u32, i: usize) {
    if i < n as usize {
        let value = *gate.add(i);
        *gate.add(i) = (value / (1.0 + exp(-value))) * *up.add(i);
    }
}

/// silu_at with exp as `exp_split` (the same f32 bits): the decode route launches' CTAs >= 1.
#[inline(always)]
pub(crate) unsafe fn silu_split_at(gate: *mut f32, up: *const f32, n: u32, i: usize) {
    if i < n as usize {
        let value = *gate.add(i);
        *gate.add(i) = (value / (1.0 + exp_split(-value))) * *up.add(i);
    }
}

/// Convert router IDs to the grouped FFN's numeric assignment ABI on device.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_assignments(ids: *const u32, weights: *const f32,
    assignments: *mut f32, tokens: u32) {
    let i = index();
    if i < tokens as usize * 10 {
        *assignments.add(i * 3) = (i / 10) as f32;
        *assignments.add(i * 3 + 1) = *ids.add(i) as f32;
        *assignments.add(i * 3 + 2) = *weights.add(i);
    }
}

/// Rank-reduced expert rows are ordered by ID; weight and add shared last.
/// Tile tokens use the second grid dimension (11 partial rows per token).
/// Each CTA sorts its token's weights once; columns keep independent sums.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_moe_finish(p: crate::fp8_abi::MoeFinishParams) {
    finish::<10, 40>(p, crate::fp8_abi::FinishLayout::FLASH, 0, 0);
}

/// Same ordered arithmetic for every cohort; caller folds ranks before launch.
/// Router validates unique IDs/finite weights and owns each row's sticky status.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn ordered_moe_finish(p: crate::fp8_abi::OrderedFinishParams) {
    if core::arch::nvptx::_block_idx_y() as u32 >= p.tokens { return; }
    finish::<32, 128>(p.finish, p.layout, p.residual, p.shared);
}

#[inline(always)]
unsafe fn finish<const TOP: usize, const BYTES: usize>(p: crate::fp8_abi::MoeFinishParams,
    layout: crate::fp8_abi::FinishLayout, residual: u64, shared: u64) {
    let token = core::arch::nvptx::_block_idx_y() as usize;
    let (selected, hidden) = (layout.selected as usize, layout.hidden as usize);
    let (ids, weights) = ((p.ids as *const u32).add(token * selected), (p.weights as *const f32).add(token * selected));
    let (partials, shared_gate) = ((p.partials as *const f32).add(token * layout.output_stride as usize * hidden), (p.shared_gate as *const f32).add(token));
    let output = (p.output as *mut f32).add(token * p.columns as usize);
    let rounded = crate::bf16::rounded;
    let local = index();
    let ordered_weights = crate::ptx::shared::<BYTES>();
    if core::arch::nvptx::_thread_idx_x() == 0 && *(p.error as *const u32).add(token) == 0 {
        let mut order = [0usize; TOP];
        for slot in 0..selected {
            let mut pos = slot;
            while pos > 0 && *ids.add(order[pos - 1]) > *ids.add(slot) {
                order[pos] = order[pos - 1]; pos -= 1;
            }
            order[pos] = slot;
        }
        for row in 0..selected {
            crate::ptx::store(ordered_weights + (row * 4) as u32, *weights.add(order[row]));
        }
    }
    // Tail and rejected columns participate before returning. Rejected routes
    // never dereference IDs/weights or consume the uninitialized shared tile.
    crate::ptx::sync();
    if local >= p.columns as usize { return; }
    if *(p.error as *const u32).add(token) != 0 { *output.add(local) = 0.0; return; }
    let col = local + p.column_start as usize;
    let sum = layout.routed_sum(p.activation_quantization != 0, |row| *partials.add(row * hidden + col),
        |row| crate::ptx::load(ordered_weights + (row * 4) as u32), rounded);
    let shared = if shared == 0 { partials.add(selected * hidden + col) } else { (shared as *const f32).add(token * hidden + col) };
    let value = sum + *shared * (1.0 / (1.0 + exp(-*shared_gate)));
    *output.add(local) = if residual == 0 { value }
        else { rounded(*(residual as *const f32).add(token * hidden + col) + value) };
}
