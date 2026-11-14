//! Flash Next embedding, four-stream hyper-connections and lexical residuals.

//! Official BF16/FP8 weights are converted in registers; activations remain FP32.
use crate::{index, layer::{exp, sqrt, sum}, math::bf16_to_f32 as bf};
use crate::flash_gdn::rounded;
#[cfg(feature = "tensor-cores")]
#[path = "flash_bf16.rs"]
mod bf16_mma;

#[no_mangle]
pub extern "ptx-kernel" fn flash_hybrid_copy(p: crate::kernel_params::HybridCopy) {resident_flash_hybrid_copy(p)}
#[inline(always)]
pub(crate) fn resident_flash_hybrid_copy(p:crate::kernel_params::HybridCopy) {
    // SAFETY: Preserve the original kernel contract and launch geometry. The
    // direct entry and checked indirect command adapter retain the same operands
    // through ordered completion. This is the original kernel body verbatim.
    unsafe {
    let p = if p.rows == 0 { p } else {
        let row = &*(p.rows as *const crate::kernel_params::HybridCopyRow).add(core::arch::nvptx::_block_idx_y() as usize);
        crate::kernel_params::HybridCopy { source: row.source as _, destination: row.destination as _,
            words: row.words, padding: 0, rows: 0 }
    };
    let i = core::arch::nvptx::_block_idx_x() as usize * 128 + core::arch::nvptx::_thread_idx_x() as usize;
    if i < p.words as usize { *p.destination.add(i) = *p.source.add(i); }

    }
}

#[inline(always)]
unsafe fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + exp(-x)) }

#[inline(always)]
fn fp8(v: u8) -> f32 {
    let a = (v & 127) as u32;
    let magnitude = if a < 8 {
        a as f32 * (1.0 / 512.0)
    } else if a == 127 {
        f32::NAN
    } else {
        f32::from_bits((a << 20) + (120 << 23))
    };
    f32::from_bits(magnitude.to_bits() | (((v as u32) & 128) << 24))
}

/// One gathered BF16 vocabulary row becomes four identical residual streams.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_embedding_expand(
    data: *const u16, out: *mut f32, h: u32,
) {
    let i = index();
    if i < 4 * h as usize { *out.add(i) = bf(*data.add(i % h as usize)); }
}

/// Decode only the 16 requested 160-wide lexical rows after rank exchange.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_ple_decode(
    data: *const u8, out: *mut f32, scale: f32,
) {
    let i = index();
    if i < 2560 { *out.add(i) = fp8(*data.add(i)) * scale; }
}

/// One warp per group. Each feature has its own official norm weight.
/// A grid with S warps per group lets every warp recompute the identical
/// square sum and write 1/S of the group; one warp per group is the original.
/// In place (x == y) one warp owns each group: another warp could otherwise
/// overwrite inputs that a slower warp has not yet summed.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_norm(
    x: *const f32, w: *const u16, y: *mut f32,
    group: u32, groups: u32, zero_centered: u32,
    epsilon: f32, repeat_weight: u32, precision: u32,
) {
    normalize(x, w, y, group, groups, zero_centered, epsilon, repeat_weight, precision, 0);
}

/// Retain FP32 normalized streams and optionally emit the next HC projection's
/// BF16 operands in the decode or full prompt-tile layout.
/// Grid: block128, grid.y=rows; grid.x=4 for rows>16, else 1 in-place or 8.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_norm(p: crate::kernel_params::FlashHcNorm) {
    if core::arch::nvptx::_grid_dim_y() > 16 {
        hc_norm_prompt(p);
        return;
    }
    normalize(p.input as *const f32, p.weight as *const u16, p.output as *mut f32,
        2560, 4, 1, 1e-6, 0, 0, p.packed);
}

/// Flash-Next retains this identical solo norm for every row count.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_norm(p: crate::kernel_params::FlashHcNorm) {
    normalize(p.input as *const f32, p.weight as *const u16, p.output as *mut f32,
        2560, 4, 1, 1e-6, 0, 0, 0);
}

/// One CTA caches each stream. Each warp computes eight original lane sums;
/// warp zero retains the XOR tree and shares all 32 lane scales bit for bit.
#[inline(always)]
unsafe fn hc_norm_prompt(p: crate::kernel_params::FlashHcNorm) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let stream = core::arch::nvptx::_block_idx_x() as usize;
    if stream >= 4 { return; }
    let token = core::arch::nvptx::_block_idx_y() as usize;
    let base = token * 10240 + stream * 2560;
    let x = (p.input as *const f32).add(base);
    let y = (p.output as *mut f32).add(base);
    let w = (p.weight as *const u16).add(stream * 2560);
    let sh = crate::ptx::shared::<10368>();
    let mut i = tid;
    while i < 2560 {
        let (a, b, c, d) = (*x.add(i), *x.add(i + 128), *x.add(i + 256), *x.add(i + 384));
        crate::ptx::store(sh + (i * 4) as u32, a);
        crate::ptx::store(sh + ((i + 128) * 4) as u32, b);
        crate::ptx::store(sh + ((i + 256) * 4) as u32, c);
        crate::ptx::store(sh + ((i + 384) * 4) as u32, d);
        i += 512;
    }
    crate::ptx::sync();
    if tid % 32 < 8 {
        let lane = tid / 32 * 8 + tid % 32;
        let mut squares = 0.0f32;
        i = lane;
        while i < 2560 {
            let a = crate::ptx::load(sh + (i * 4) as u32);
            let b = crate::ptx::load(sh + ((i + 32) * 4) as u32);
            let c = crate::ptx::load(sh + ((i + 64) * 4) as u32);
            let d = crate::ptx::load(sh + ((i + 96) * 4) as u32);
            squares = (((squares + a * a) + b * b) + c * c) + d * d;
            i += 128;
        }
        crate::ptx::store(sh + 10240 + (lane * 4) as u32, squares);
    }
    crate::ptx::sync();
    if tid < 32 {
        let squares = crate::ptx::load(sh + 10240 + (tid * 4) as u32);
        let scale = 1.0 / sqrt(sum(squares) / 2560.0 + 1e-6);
        crate::ptx::store(sh + 10240 + (tid * 4) as u32, scale);
    }
    crate::ptx::sync();
    let scale = crate::ptx::load(sh + 10240 + (tid % 32 * 4) as u32);
    i = tid;
    while i < 2560 {
        let value = crate::ptx::load(sh + (i * 4) as u32) * scale * (bf(*w.add(i)) + 1.0);
        *y.add(i) = value;
        #[cfg(feature = "tensor-cores")]
        if p.packed != 0 {
            let (hi, lo) = crate::gemm::split2(value);
            let plane = (core::arch::nvptx::_grid_dim_y() as usize).div_ceil(16) * 10240 * 16;
            let column = stream * 2560 + i;
            crate::gemm::store_bf16_input_bits(p.packed as *mut u16, token, column, 10240, hi);
            crate::gemm::store_bf16_input_bits((p.packed as *mut u16).add(plane), token, column, 10240, lo);
        }
        i += 128;
    }
}

// Keep the iterator loop's PTX operand order when its register batches unroll.
#[inline(always)]
pub(crate) unsafe fn norm_square_add(acc: f32, value: f32) -> f32 {
    let out: f32;
    core::arch::asm!("{{ .reg .f32 sq; mul.rn.f32 sq, {v}, {v}; add.rn.f32 {out}, {acc}, sq; }}",
        v = in(reg32) value, acc = in(reg32) acc, out = out(reg32) out);
    out
}

#[inline(always)]
unsafe fn normalize(x: *const f32, w: *const u16, y: *mut f32,
    group: u32, groups: u32, zero_centered: u32, epsilon: f32,
    repeat_weight: u32, precision: u32, packed: u64) {
    let t = index();
    let warps = core::arch::nvptx::_grid_dim_x() as usize * core::arch::nvptx::_block_dim_x() as usize / 32;
    let split = if x as usize == y as usize { 1 } else { (warps / (groups as usize).max(1)).max(1) };
    let g = t / 32 / split;
    let part = t / 32 % split;
    let lane = t % 32;
    if g >= groups as usize { return; }
    let group = group as usize;
    // Tile rows use the second grid dimension; every row reuses the same weights.
    let row = core::arch::nvptx::_block_idx_y() as usize * groups as usize * group;
    let (x, y) = (x.add(row), y.add(row));
    let base = g * group;
    let mut squares = [0.0f32; 4];
    let mut i = lane;
    if precision == 0 {
        // Same ascending per-lane order; sixteen, then four loads are issued per step.
        while i + 480 < group {
            let v: [f32; 16] = core::array::from_fn(|j| *x.add(base + i + j * 32));
            macro_rules! add { ($($j:literal),*) => { $(squares[0] = norm_square_add(squares[0], v[$j]);)* }; }
            add!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            i += 512;
        }
        while i + 96 < group {
            let v = [*x.add(base + i), *x.add(base + i + 32), *x.add(base + i + 64), *x.add(base + i + 96)];
            squares[0] = norm_square_add(norm_square_add(norm_square_add(norm_square_add(squares[0], v[0]), v[1]), v[2]), v[3]);
            i += 128;
        }
    }
    while i < group {
        for j in 0..4 {
            if (precision != 0 || j == 0) && i + j * 32 < group {
                let value = *x.add(base + i + j * 32);
                squares[j] += value * value;
            }
        }
        i += if precision == 0 { 32 } else { 128 };
    }
    let scale = 1.0 / sqrt(sum(((squares[0] + squares[1]) + squares[2]) + squares[3]) / group as f32 + epsilon);
    let chunk = group.div_ceil(split).div_ceil(32) * 32;
    let end = group.min((part + 1) * chunk);
    i = part * chunk + lane;
    while i < end {
        let weight = if repeat_weight == 0 { base + i } else { i };
        let value = *x.add(base + i) * scale * (bf(*w.add(weight)) + zero_centered as f32);
        *y.add(base + i) = if precision == 0 { value } else { rounded(value) };
        #[cfg(feature = "tensor-cores")]
        if packed != 0 {
            let (hi, lo) = crate::gemm::split2(value);
            let width = groups as usize * group;
            if core::arch::nvptx::_grid_dim_y() == 1 {
                *(packed as *mut u16).add(base + i) = hi;
                *(packed as *mut u16).add(width + base + i) = lo;
            } else {
                let token = core::arch::nvptx::_block_idx_y() as usize;
                let plane_elements = (core::arch::nvptx::_grid_dim_y() as usize).div_ceil(16) * (width / 16) * 256;
                crate::gemm::store_bf16_input_bits(packed as *mut u16, token, base + i, width, hi);
                crate::gemm::store_bf16_input_bits((packed as *mut u16).add(plane_elements), token, base + i, width, lo);
            }
        }
        i += 32;
    }
}

/// Native CUDA mean with four adjacent accumulators and the descending
/// warp tree. Keep one row's order independent of the number of pooled rows.
#[inline(always)]
unsafe fn native_square_sum(sh: u32, width: usize, tid: usize, order: u32) -> f32 {
    if tid >= 32 { return 0.0; }
    let groups = if order == 2 { 16 } else { 1 };
    let mut partials = [0.0f32; 16];
    for group in 0..groups {
        let mut values = [0.0f32; 4];
        let mut i = (tid + group * 32) * 4;
        while i < width {
            for j in 0..4 {
                if i + j < width { values[j] += crate::ptx::load(sh + ((i + j) * 4) as u32); }
            }
            i += groups * 128;
        }
        partials[group] = ((values[0] + values[1]) + values[2]) + values[3];
    }
    let mut half = groups / 2;
    while half != 0 {
        for i in 0..half { partials[i] += partials[i + half]; }
        half /= 2;
    }
    sum(partials[0])
}

/// BF16 rows use the native CUDA reduction or four prefill accumulators.
/// # Safety
/// Block128/grid=rows, group <=5120; X/Y rows and BF16 W must be in bounds.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_norm_decode(p: crate::kernel_params::HybridNorm) {
    let crate::kernel_params::HybridNorm { input: x, weight: w, output: y,
        width: group, zero_centered, epsilon, order: four, .. } = p;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let row = core::arch::nvptx::_block_idx_x() as usize;
    let x = x.add(row * group as usize);
    let y = y.add(row * group as usize);
    let sh = crate::ptx::shared::<20496>();
    let mut i = tid;
    while i < group as usize {
        let value = *x.add(i);
        crate::ptx::store(sh + (i * 4) as u32, value * value);
        i += 128;
    }
    crate::ptx::sync();
    let native = if four != 1 { native_square_sum(sh, group as usize, tid, four) } else { 0.0 };
    if tid < 32 {
        let mut squares = [0.0f32; 4];
        i = tid;
        while four == 1 && i < group as usize {
            for j in 0..4 {
                if i + j * 32 < group as usize {
                    squares[j] += crate::ptx::load(sh + ((i + j * 32) * 4) as u32);
                }
            }
            i += 128;
        }
        let mean = if four != 1 { native * (1.0 / group as f32) }
            else { sum(((squares[0] + squares[1]) + squares[2]) + squares[3]) / group as f32 };
        let scale = 1.0 / sqrt(mean + epsilon);
        if tid == 0 { crate::ptx::store(sh + 20480, scale); }
    }
    crate::ptx::sync();
    let scale = crate::ptx::load(sh + 20480);
    i = tid;
    while i < group as usize {
        *y.add(i) = rounded(*x.add(i) * scale * (bf(*w.add(i)) + zero_centered as f32));
        i += 128;
    }
}

/// Native BF16 residual boundary, without a separate cast launch.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_residual_bf16(x: *mut f32, residual: *const f32, width: u32) {
    let i = index();
    if i < width as usize { *x.add(i) = rounded(*x.add(i) + *residual.add(i)); }
}

/// BF16 residual and the next norm share a row CTA. The square sum keeps the
/// native reduction; shared squares keep multiply and accumulation separate.
/// The rounded residual also stays in shared (second array at +20480, scale
/// at +40960), so the output loop reuses the exact bits loop 1 wrote to
/// global x instead of reloading them.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_residual_norm_bf16(p: crate::kernel_params::HybridNorm) {
    let crate::kernel_params::HybridNorm { input: x, residual, weight: w,
        output: y, width, epsilon, order: four, .. } = p;
    let row = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let threads = core::arch::nvptx::_block_dim_x() as usize;
    let sh = crate::ptx::shared::<40976>();
    let x = x.add(row * width as usize);
    let residual = residual.add(row * width as usize);
    let y = y.add(row * width as usize);
    let mut i = tid;
    while i < width as usize {
        let value = rounded(*x.add(i) + *residual.add(i));
        *x.add(i) = value;
        crate::ptx::store(sh + (i * 4) as u32, value * value);
        crate::ptx::store(sh + 20480 + (i * 4) as u32, value);
        i += threads;
    }
    crate::ptx::sync();
    let native = if four != 1 { native_square_sum(sh, width as usize, tid, four) } else { 0.0 };
    if tid < 32 {
        let mut squares = [0.0f32; 4];
        i = tid;
        if four == 1 {
            while i < width as usize {
                for j in 0..4 {
                    if i + j * 32 < width as usize {
                        squares[j] += crate::ptx::load(sh + ((i + j * 32) * 4) as u32);
                    }
                }
                i += 128;
            }
        }
        let mean = if four != 1 { native * (1.0 / width as f32) }
            else { sum(((squares[0] + squares[1]) + squares[2]) + squares[3]) / width as f32 };
        let scale = 1.0 / sqrt(mean + epsilon);
        if tid == 0 { crate::ptx::store(sh + 40960, scale); }
    }
    crate::ptx::sync();
    let scale = crate::ptx::load(sh + 40960);
    i = tid;
    while i < width as usize {
        *y.add(i) = rounded(crate::ptx::load(sh + 20480 + (i * 4) as u32) * scale * (bf(*w.add(i)) + 1.0));
        i += threads;
    }
}

/// Normalize the GEMM residual directly into the next BF16 MMA A fragments.
/// Block128 per row; width is a multiple of128 and rows a multiple of256.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_norm_partials_bf16(
    x: *const f32, partials: *const f32, w: *const u16, y: *mut u32,
    width: u32, epsilon: f32,
) {
    let row = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let sh = crate::ptx::shared::<4>();
    if tid < 32 {
        let mut squares = 0.0;
        let mut i = tid;
        while i < width as usize / 128 {
            squares += *partials.add(row * (width as usize / 128) + i);
            i += 32;
        }
        let scale = 1.0 / sqrt(sum(squares) / width as f32 + epsilon);
        if tid == 0 { crate::ptx::store(sh, scale); }
    }
    crate::ptx::sync();
    let scale = crate::ptx::load(sh);
    let mut i = tid * 2;
    while i < width as usize {
        let a = *x.add(row * width as usize + i) * scale * (bf(*w.add(i)) + 1.0);
        let b = *x.add(row * width as usize + i + 1) * scale * (bf(*w.add(i + 1)) + 1.0);
        let pair: u32;
        core::arch::asm!("cvt.rn.bf16x2.f32 {pair}, {b}, {a};",
            pair=out(reg32)pair, a=in(reg32)a, b=in(reg32)b, options(nomem, nostack, pure));
        let tile = row / 16 * (width as usize / 16) + i / 16;
        let lane = row % 8 * 4 + i % 8 / 2;
        *y.add(tile * 128 + lane * 4 + row % 16 / 8 + i % 16 / 8 * 2) = pair;
        i += 256;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_silu(
    x: *mut f32, up: *const f32, n: u32, divisor: f32, precision: u32,
) {
    let i = index();
    if i < n as usize {
        let factor = if up.is_null() { 1.0 } else { *up.add(i) };
        *x.add(i) = silu(*x.add(i), factor, divisor, precision);
    }
}

#[inline(always)]
pub(crate) unsafe fn silu(value: f32, factor: f32, divisor: f32, precision: u32) -> f32 {
    let value = value / divisor;
    let value = if precision & 2 != 0 { rounded(value) } else { value };
    let activated = value * sigmoid(value);
    let activated = if precision == 0 { activated } else { rounded(activated) };
    let factor = if precision & 2 != 0 { rounded(factor) } else { factor };
    let value = activated * factor;
    if precision == 0 { value } else { rounded(value) }
}

/// Mean of the four normalized, independently gated residual streams.
/// Tile tokens use the second grid dimension.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_mix(
    normalized: *const f32, gates: *const f32, out: *mut f32, h: u32, packed: u64,
) {
    let i = index();
    let token = core::arch::nvptx::_block_idx_y() as usize;
    let (normalized, gates) = (normalized.add(token * 4 * h as usize), gates.add(token * 4 * h as usize));
    let out = out.add(token * h as usize);
    if i < h as usize {
        let mut value = 0.0;
        let mut stream = 0;
        while stream < 4 {
            let at = stream * h as usize + i;
            value += *normalized.add(at) * sigmoid(*gates.add(at));
            stream += 1;
        }
        *out.add(i) = value * 0.25;
        #[cfg(feature = "tensor-cores")]
        if packed != 0 {
            crate::gemm::store_split_input(packed as *mut u16, token, i, h as usize,
                core::arch::nvptx::_grid_dim_y() as usize * h as usize, value * 0.25);
        }
    }
}

/// Inject into the original residual, using four projected stream gates.
/// Tile tokens use the second grid dimension.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_add(p: crate::kernel_params::FlashHcAdd) {
    if core::arch::nvptx::_grid_dim_y() > 16 && p.hidden % 128 == 0 {
        hc_add_prompt(p);
        return;
    }
    let i = index();
    let token = core::arch::nvptx::_block_idx_y() as usize;
    let h = p.hidden as usize;
    let residual = (p.residual as *mut f32).add(token * 4 * h);
    let gates = (p.gates as *const f32).add(token * 4);
    if i < 4 * h {
        let col = i % h;
        let value = if p.peer == 0 { *((p.block as *const f32).add(token * h + col)) }
            else {
                let packet = if col / (h / 2) == p.rank as usize { p.block } else { p.peer };
                *((packet as *const f32).add(token * (h / 2) + col % (h / 2)))
            };
        *residual.add(i) += 2.0 * sigmoid(*gates.add(i / h) * 0.25) * value;
    }
}

/// Reuse one block value across its four independent residual streams. A CTA
/// evaluates the four identical gate expressions once, then shares their bits.
#[inline(always)]
unsafe fn hc_add_prompt(p: crate::kernel_params::FlashHcAdd) {
    let col = index();
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let token = core::arch::nvptx::_block_idx_y() as usize;
    let h = p.hidden as usize;
    let sh = crate::ptx::shared::<16>();
    if tid < 4 {
        let gate = *((p.gates as *const f32).add(token * 4 + tid));
        crate::ptx::store(sh + (tid * 4) as u32, 2.0 * sigmoid(gate * 0.25));
    }
    crate::ptx::sync();
    if col < h {
        let value = if p.peer == 0 { *((p.block as *const f32).add(token * h + col)) }
            else {
                let packet = if col / (h / 2) == p.rank as usize { p.block } else { p.peer };
                *((packet as *const f32).add(token * (h / 2) + col % (h / 2)))
            };
        for stream in 0..4 {
            let residual = (p.residual as *mut f32).add(token * 4 * h + stream * h + col);
            *residual += crate::ptx::load(sh + (stream * 4) as u32) * value;
        }
    }
}

/// FP32-weight twin of flash_hc_norm for checkpoints whose HC/PLE norm
/// weights are stored F32: same warp split, square order and scale, reading
/// `(1 + w)` as F32. With packed != 0 (prompt tiles and decode rows) it also writes each
/// stored value pair's BF16 hi/lo split (split2_pair) into the dense prompt operand
/// layout of flash_bf16_input (grid.y rows, K 10240), so the projections skip
/// that pack; on the 8-CTA split (decode rows and mix_chunk prompt tiles) row token + 8 of
/// a 16-row tile past grid.y gets its 0 word, other rows past grid.y in the last M16 tile
/// stay unwritten (MMA rows are independent and those outputs are masked).
/// Grid: block128, grid.y=rows, grid.x as flash_hc_norm (1 in-place or 8).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_norm_f32(p: crate::kernel_params::FlashHcNorm) {
    hc_norm_f32(p);
}
/// flash_hc_norm_f32 for a chained decode step: the next kernel may start, this one waits.
/// # Safety
/// As flash_hc_norm_f32, on a plain or programmatic launch.
// SAFETY: griddepcontrol touches no memory; the next kernel waits for this one before it
// reads this kernel's output, and this kernel waits before it reads its input.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_norm_f32_chain(p: crate::kernel_params::FlashHcNorm) {
    core::arch::asm!("griddepcontrol.launch_dependents;", "griddepcontrol.wait;", options(nostack));
    hc_norm_f32(p);
}
// SAFETY: the body of flash_hc_norm_f32 and of its chain twin, moved here unchanged; each entry's
// contract holds for it.
#[inline(always)]
unsafe fn hc_norm_f32(p: crate::kernel_params::FlashHcNorm) {
    let (group, groups) = (2560usize, 4usize);
    let t = index();
    let warps = core::arch::nvptx::_grid_dim_x() as usize * core::arch::nvptx::_block_dim_x() as usize / 32;
    let split = if p.input == p.output { 1 } else { (warps / groups).max(1) };
    let (g, part, lane) = (t / 32 / split, t / 32 % split, t % 32);
    if g >= groups { return; }
    let row = core::arch::nvptx::_block_idx_y() as usize * groups * group;
    let (x, y, w) = ((p.input as *const f32).add(row), (p.output as *mut f32).add(row), p.weight as *const f32);
    let (base, mut squares) = (g * group, 0.0f32);
    let chunk = group.div_ceil(split).div_ceil(64) * 64;
    let end = group.min((part + 1) * chunk);
    // 8-CTA split (decode rows and mix_chunk prompt tiles; chunk <= 320: at most 5 value pairs
    // per lane, the packed loop's pairs): the outputs' values and weights are read before the
    // squares too, so no output read waits behind the previous output's store. The same values,
    // products and packed words as the loops below (the decode HC down reads these planes in
    // place of its own pack).
    let early = chunk <= 320;
    let at = |j: usize| base + (part * chunk + lane * 2 + 64 * j).min(group - 2);
    let pre = |j: usize| [*x.add(at(j)), *x.add(at(j) + 1), *w.add(at(j)), *w.add(at(j) + 1)];
    let outs = if early { [pre(0), pre(1), pre(2), pre(3), pre(4)] } else { [[0.0f32; 4]; 5] };
    // The asm add is a memory clobber, so loads issued one per add would pay one L2
    // round trip each; loading first keeps the same values in the same order.
    // group = 2560 = 80 values per lane, all loaded before the first add.
    let lane_x = x.add(base + lane);
    macro_rules! squares { ($($j:literal)*) => {
        let v = [$(*lane_x.add($j * 32)),*];
        $(squares = norm_square_add(squares, v[$j]);)*
    } }
    squares!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
        40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79);
    let scale = 1.0 / sqrt(sum(squares) / group as f32 + 1e-6);
    // Store the same values. The early and packed paths store pairs per lane.
    // Each packed pair also lands in its A-fragment word. The unpacked loop stores one value per lane.
    let rows = core::arch::nvptx::_grid_dim_y() as usize;
    let token = core::arch::nvptx::_block_idx_y() as usize;
    let kt = groups * group / 16;
    let plane = rows.div_ceil(16) * kt * 128;
    let packed = p.packed as *mut u32;
    let word = |c: usize| ((token / 16 * kt + c / 16) * 32 + token % 8 * 4 + c % 8 / 2) * 4 + token % 16 / 8 + c % 16 / 8 * 2;
    if early {
        macro_rules! store { ($($j:literal)*) => { $(
            let i = part * chunk + lane * 2 + 64 * $j;
            if i < end {
                let o = outs[$j];
                let (a, b) = (o[0] * scale * (1.0 + o[2]), o[1] * scale * (1.0 + o[3]));
                *y.add(base + i) = a;
                *y.add(base + i + 1) = b;
                if p.packed != 0 {
                    let (hi, lo) = crate::gemm::split2_pair(a, b);
                    *packed.add(word(base + i)) = hi;
                    *packed.add(plane + word(base + i)) = lo;
                    // Row token + 8 of this 16-row tile is padding: its word is 0, as the pack's.
                    if token % 16 < 8 && token + 8 >= rows {
                        *packed.add(word(base + i) + 1) = 0;
                        *packed.add(plane + word(base + i) + 1) = 0;
                    }
                }
            }
        )* } }
        store!(0 1 2 3 4);
        return;
    }
    if p.packed == 0 {
        let mut i = part * chunk + lane;
        while i < end { *y.add(base + i) = *x.add(base + i) * scale * (1.0 + *w.add(base + i)); i += 32; }
        return;
    }
    let mut i = part * chunk + lane * 2;
    while i < end {
        let a = *x.add(base + i) * scale * (1.0 + *w.add(base + i));
        let b = *x.add(base + i + 1) * scale * (1.0 + *w.add(base + i + 1));
        *y.add(base + i) = a;
        *y.add(base + i + 1) = b;
        let (hi, lo) = crate::gemm::split2_pair(a, b);
        *packed.add(word(base + i)) = hi;
        *packed.add(plane + word(base + i)) = lo;
        i += 64;
    }
}

/// Prompt HC combine (flash_next_hc_add, peer 0) fused with the next mix's
/// flash_hc_norm_f32: one CTA per token, warp s owns stream s. Each lane forms
/// residual + 2 sigmoid(gate/4) block for its elements lane, lane+32, ... in index order,
/// stores it and adds its square in that order, as flash_hc_norm_f32 does; the same warp sum
/// gives the scale, and the outputs (+ packed hi/lo planes when packed != 0) follow.
/// Grid 1 x ceil(rows/16)*8, block256, dynamic shared 8 * 2560 * 4 + 32: CTA b owns tokens
/// t0 = 16(b / 8) + b % 8 and t0 + 8, the two rows whose words share each packed 16-byte chunk.
/// Warp w forms row (t0 + 8(w / 4), stream w % 4) exactly as add_norm does; then every lane
/// writes whole outputs and hi/lo chunks of both rows. scales != 0: each row's scale goes to
/// scales[token * 4 + stream] and the normalized output is not written (prompt HC up forms it
/// from the residual, flash_next_hc_up_tiled with residual != 0); with packed == 0 too, the
/// residual rows and the scales are all it writes (HC down forms the planes from them,
/// flash_bf16_mma_prefill_tiled_hc with norm != 0).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_add_norm(p: crate::kernel_params::FlashHcAddNorm, rows: u32) {
    let b = core::arch::nvptx::_block_idx_y() as usize;
    let t0 = b / 8 * 16 + b % 8;
    add_norm_pair(p, rows as usize, |r, i| *(p.block as *const f32).add((t0 + r * 8) * 2560 + i));
}

/// flash_next_hc_add_norm's token-pair body; `block_at(r, i)` is element i of row t0 + 8r's block.
#[inline(always)]
unsafe fn add_norm_pair(p: crate::kernel_params::FlashHcAddNorm, rows: usize, block_at: impl Fn(usize, usize) -> f32) {
    let (group, groups) = (2560usize, 4usize);
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let b = core::arch::nvptx::_block_idx_y() as usize;
    let t0 = b / 8 * 16 + b % 8;
    let scales = (8 * group * 4) as u32;
    {
        let (token, stream) = (t0 + warp / 4 * 8, warp % 4);
        if token < rows {
            let residual = (p.residual as *mut f32).add(token * groups * group + stream * group);
            let gate = *(p.gates as *const f32).add(token * groups + stream);
            let sh = (warp * group * 4) as u32;
            let mut squares = 0.0f32;
            // Sixteen values per batch are formed and stored before their (memory-clobbering)
            // square adds, which keep each lane's index order.
            let mut i = lane;
            while i < group {
                macro_rules! batch { ($($j:literal)*) => {
                    let v = [$(crate::next_math::hc_add(*residual.add(i + $j * 32), gate, block_at(warp / 4, i + $j * 32),
                        |x| crate::flash_next::hc_sigmoid(x))),*];
                    $(*residual.add(i + $j * 32) = v[$j];
                      crate::ptx::store(sh + ((i + $j * 32) * 4) as u32, v[$j]);)*
                    $(squares = norm_square_add(squares, v[$j]);)*
                } }
                batch!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
                i += 16 * 32;
            }
            let scale = 1.0 / sqrt(sum(squares) / group as f32 + 1e-6);
            if lane == 0 {
                crate::ptx::store(scales + (warp * 4) as u32, scale);
                if p.scales != 0 {
                    *(p.scales as *mut f32).add(token * groups + stream) = scale;
                }
            }
        }
    }
    // Scales and no planes (flash_bf16_mma_prefill_tiled_hc with norm != 0 forms them): nothing more to write.
    if p.scales != 0 && p.packed == 0 { return; }
    crate::ptx::sync();
    let kt = groups * group / 16;
    let plane = rows.div_ceil(16) * kt * 128;
    let (w, packed) = (p.weight as *const f32, p.packed as *mut u32);
    let pair = t0 + 8 < rows;
    // Chunk k: K16 block k / 4, column pair q = k % 4 (columns c, c + 1, c + 8, c + 9).
    // The next chunk's weights are issued before this chunk's (memory-clobbering) stores.
    let weights_of = |k: usize| {
        let c = k / 4 * 16 + k % 4 * 2;
        let (a, b, x, y): (f32, f32, f32, f32);
        core::arch::asm!("ld.global.nc.v2.f32 {{{a}, {b}}}, [{s}];", a = out(reg32) a, b = out(reg32) b, s = in(reg64) w.add(c));
        core::arch::asm!("ld.global.nc.v2.f32 {{{a}, {b}}}, [{s}];", a = out(reg32) x, b = out(reg32) y, s = in(reg64) w.add(c + 8));
        [a, b, x, y]
    };
    let mut k = tid;
    let mut next = if k < kt * 4 && t0 < rows { weights_of(k) } else { [0.0; 4] };
    while k < kt * 4 && t0 < rows {
        let (blk, q) = (k / 4, k % 4);
        let c = blk * 16 + q * 2;
        let (stream, e) = (c / group, c % group);
        let weight = next;
        if k + 256 < kt * 4 {
            next = weights_of(k + 256);
        }
        let mut a = [[0.0f32; 4]; 2];
        let mut r = 0;
        while r < 2 {
            if r == 0 || pair {
                let at = (((r * 4 + stream) * group + e) * 4) as u32;
                let scale = crate::ptx::load(scales + ((r * 4 + stream) * 4) as u32);
                let v = [crate::ptx::load(at), crate::ptx::load(at + 4), crate::ptx::load(at + 32), crate::ptx::load(at + 36)];
                let mut j = 0;
                while j < 4 {
                    a[r][j] = v[j] * scale * (1.0 + weight[j]);
                    j += 1;
                }
                if p.scales == 0 {
                    let y = (p.output as *mut f32).add((t0 + r * 8) * groups * group + c);
                    core::arch::asm!("st.global.v2.f32 [{y}], {{{a}, {b}}};", y = in(reg64) y, a = in(reg32) a[r][0], b = in(reg32) a[r][1]);
                    core::arch::asm!("st.global.v2.f32 [{y}], {{{a}, {b}}};", y = in(reg64) y.add(8), a = in(reg32) a[r][2], b = in(reg32) a[r][3]);
                }
            }
            r += 1;
        }
        if p.packed != 0 {
            // Words +0..+3: (t0; c, c+1), (t0 + 8; c, c+1), (t0; c+8, c+9), (t0 + 8; c+8, c+9).
            let word = ((t0 / 16 * kt + blk) * 32 + t0 % 8 * 4 + q) * 4;
            let (h0, l0) = crate::gemm::split2_pair(a[0][0], a[0][1]);
            let (h2, l2) = crate::gemm::split2_pair(a[0][2], a[0][3]);
            if pair {
                let (h1, l1) = crate::gemm::split2_pair(a[1][0], a[1][1]);
                let (h3, l3) = crate::gemm::split2_pair(a[1][2], a[1][3]);
                core::arch::asm!("st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg64) packed.add(word),
                    a = in(reg32) h0, b = in(reg32) h1, c = in(reg32) h2, d = in(reg32) h3);
                core::arch::asm!("st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg64) packed.add(plane + word),
                    a = in(reg32) l0, b = in(reg32) l1, c = in(reg32) l2, d = in(reg32) l3);
            } else {
                // Row t0 + 8 is past the last token: its words stay unwritten, as before.
                *packed.add(word) = h0;
                *packed.add(word + 2) = h2;
                *packed.add(plane + word) = l0;
                *packed.add(plane + word + 2) = l2;
            }
        }
        k += 256;
    }
}

/// flash_next_hc_add_norm whose block rows are the tokens' MoE outputs, formed here exactly as
/// flash_next_k16_finish forms them (one thread per token ranks the ten routed slots; per column
/// the ten BF16 expert rows in rank order by routed_add, then the gated shared expert,
/// BF16-rounded) instead of being written by k16_finish and read back. The rows are BF16 values,
/// kept as their bits. Grid 1 x ceil(rows/16)*8, block256, dynamic shared 8*2560*4 + 32 +
/// 2*2560*2 + 2*16*4.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_finish_add_norm(p: crate::kernel_params::FlashHcAddNorm,
    f: crate::fp8_abi::MoeFinishParams, rows: u32) {
    hc_finish_add_norm::<false>(p, f, rows);
}
/// flash_next_hc_finish_add_norm with the shared expert's gate exp as `exp_split` (the same
/// f32 bits as exp on all 2^32 inputs; every thread formed it with the FP64 exp).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_finish_add_norm_fast(p: crate::kernel_params::FlashHcAddNorm,
    f: crate::fp8_abi::MoeFinishParams, rows: u32) {
    hc_finish_add_norm::<true>(p, f, rows);
}
#[inline(always)]
unsafe fn hc_finish_add_norm<const FAST: bool>(p: crate::kernel_params::FlashHcAddNorm, f: crate::fp8_abi::MoeFinishParams, rows: u32) {
    let rows = rows as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let b = core::arch::nvptx::_block_idx_y() as usize;
    let t0 = b / 8 * 16 + b % 8;
    let (block_sh, weights_sh) = ((8 * 2560 * 4 + 32) as u32, (8 * 2560 * 4 + 32 + 2 * 2560 * 2) as u32);
    // Threads 128r.. form row t0 + 8r.
    let (r, local) = (tid / 128, tid % 128);
    let row = t0 + r * 8;
    let error = row >= rows || *(f.error as *const u32).add(row) != 0;
    let weights_r = weights_sh + (r * 16 * 4) as u32;
    if local == 0 && !error {
        let ids = (f.ids as *const u32).add(row * 10);
        let weights = (f.weights as *const f32).add(row * 10);
        for slot in 0..10 {
            let rank = crate::fp8_abi::RoutingLayout::FLASH.ordered_slot(slot, |j| *ids.add(j));
            crate::ptx::store(weights_r + (rank * 4) as u32, *weights.add(slot));
        }
    }
    crate::ptx::sync();
    let block_r = block_sh + (r * 2560 * 2) as u32;
    let mut col = local;
    if !error {
        let gate = if FAST { 1.0 / (1.0 + crate::flash_moe::exp_split(-*(f.shared_gate as *const f32).add(row))) }
            else { 1.0 / (1.0 + crate::flash_moe::exp(-*(f.shared_gate as *const f32).add(row))) };
        // fp4_prefill_tc: each thread forms 8 adjacent columns from one
        // 16-byte load per routed rank (8 BF16) and two of the FP32 shared row; every column
        // keeps routed_add's rank order and the same gate add, so the bits are unchanged.
        let mut g8 = local;
        while g8 < 2560 / 8 {
            let c0 = g8 * 8;
            let routed = (f.partials as *const u16).add(row * 22 * 2560 + c0);
            let mut value = [0.0f32; 8];
            for rank in 0..10 {
                let w = crate::ptx::load(weights_r + (rank * 4) as u32);
                let (q0, q1, q2, q3): (u32, u32, u32, u32);
                core::arch::asm!("ld.global.nc.v4.b32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                    a = out(reg32) q0, b = out(reg32) q1, c = out(reg32) q2, d = out(reg32) q3,
                    p = in(reg64) routed.add(rank * 2 * 2560));
                let q = [q0, q1, q2, q3];
                for j in 0..8 {
                    let bits = if j % 2 == 0 { q[j / 2] << 16 } else { q[j / 2] & 0xffff_0000 };
                    value[j] = crate::next_math::routed_add(value[j], f32::from_bits(bits), w);
                }
            }
            let shared_row = (f.partials as *const f32).add(row * 11 * 2560 + 10 * 2560 + c0);
            let mut packed = [0u32; 4];
            for j in 0..8 {
                let out = crate::next_math::bf16(crate::next_math::routed_add(value[j], *shared_row.add(j), gate)) as u32 & 0xffff;
                packed[j / 2] |= out << (16 * (j % 2));
            }
            core::arch::asm!("st.shared.v4.b32 [{a}], {{{b}, {c}, {d}, {e}}};", a = in(reg32) block_r + (c0 * 2) as u32,
                b = in(reg32) packed[0], c = in(reg32) packed[1], d = in(reg32) packed[2], e = in(reg32) packed[3]);
            g8 += 128;
        }
    } else {
        while col < 2560 {
            core::arch::asm!("st.shared.u16 [{a}], {v};", a = in(reg32) block_r + (col * 2) as u32, v = in(reg16) 0u16);
            col += 128;
        }
    }
    crate::ptx::sync();
    add_norm_pair(p, rows, |r, i| f32::from_bits(crate::ptx::load_u16(block_sh + ((r * 2560 + i) * 2) as u32) << 16));
}

/// y += scale * x in fixed index order (one thread per element, no atomics).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_axpy(y: *mut f32, x: *const f32, scale: f32, n: u32) {
    let i = index();
    if i < n as usize { *y.add(i) += scale * *x.add(i); }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_add(x: *mut f32, y: *const f32, n: u32, repeat: u32) {
    let i = index();
    let row = core::arch::nvptx::_block_idx_y() as usize;
    if i < n as usize {
        let input = if repeat == 1 { row * n as usize + i } else {
            let width = n as usize / repeat as usize;
            row * width + i % width
        };
        *x.add(row * n as usize + i) += *y.add(input);
    }
}

/// Four stream-wise signed-square-root gates share one projected value row.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_ple_gate(p: crate::kernel_params::FlashPleGate) {
    let h = p.width;
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let q = (p.query as *const f32).add(row * 4 * h as usize);
    let k = (p.key as *const f32).add(row * 4 * h as usize);
    let v = (p.value as *const f32).add(row * h as usize);
    let y = (p.output as *mut f32).add(row * 4 * h as usize);
    let t = index();
    let stream = t / 32;
    let lane = t % 32;
    if stream >= 4 { return; }
    let mut dot = 0.0;
    let mut i = lane;
    while i < h as usize {
        dot += *q.add(stream * h as usize + i) * *k.add(stream * h as usize + i);
        i += 32;
    }
    dot = sum(dot) / sqrt(h as f32);
    let signed_root = if dot == 0.0 { 0.0 } else {
        sqrt(dot.abs().max(1e-6)) * if dot < 0.0 { -1.0 } else { 1.0 }
    };
    let gate = sigmoid(signed_root);
    i = lane;
    while i < h as usize {
        *y.add(stream * h as usize + i) = *v.add(i) * gate;
        i += 32;
    }
}

/// PLE's dilation-three convolution, followed by the gated and residual adds
/// in their original order. One thread owns a channel through the whole tile;
/// earlier rows come from the normalized tile or its nine-token history.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_ple_conv(p: crate::kernel_params::FlashPleConv) { ple_conv(p); }
#[inline(always)]
pub(crate) unsafe fn ple_conv(p: crate::kernel_params::FlashPleConv) {
    let i = index();
    let width = p.width as usize;
    let tokens = p.tokens as usize;
    if i >= width { return; }
    let past = (p.history as *mut f32).add(i * 9);
    let x = p.input as *const f32;
    let weights = (p.weight as *const u16).add(i * 4);
    for token in 0..tokens {
        let mut value = 0.0;
        for j in 0..3 {
            let lag = 9 - j * 3;
            let input = if token < lag { *past.add(token + j * 3) }
                else { *x.add((token - lag) * width + i) };
            value += input * bf(*weights.add(j));
        }
        let offset = token * width + i;
        value += *x.add(offset) * bf(*weights.add(3));
        let added = value * sigmoid(value) + *(p.gated as *const f32).add(offset);
        *(p.residual as *mut f32).add(offset) += added;
    }
    // Ascending writes also preserve the old suffix for tiles shorter than nine.
    for j in 0..9 {
        *past.add(j) = if tokens + j < 9 { *past.add(tokens + j) }
            else { *x.add((tokens + j - 9) * width + i) };
    }
}
