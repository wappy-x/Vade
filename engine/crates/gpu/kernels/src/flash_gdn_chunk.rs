//! 64-token gated-delta chunks. Independent chunk systems are solved in parallel;
//! FP32 recurrent state advances only at chunk boundaries.
use crate::{index, layer::exp, ptx};
use crate::flash_gdn::PromptRow;
use crate::kernel_params::{GdnChunkPrompts, GdnScan, GdnScanPrompts, GdnSolve, GdnSystem, GDN_SCAN_COLUMNS, GDN_SCAN_THREADS, GDN_SYSTEM_THREADS};
use core::arch::nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x};
const C: usize = 64;
const D: usize = 128;

/// N elements, four global loads in flight before their shared stores. Each
/// asm store is a memory barrier, so interleaving would serialize every load.
#[inline(always)]
unsafe fn stage<const N: usize, T: Copy>(load: impl Fn(usize) -> T, store: impl Fn(usize, T)) {
    const B: usize = 4;
    let mut base = 0;
    while base < N {
        let values: [T; B] = core::array::from_fn(|n| load(base + n));
        for n in 0..B { store(base + n, values[n]); }
        base += B;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_cumulative(
    g: *const f32, cumulative: *mut f32, chunk_decay: *mut f32, heads: u32, tokens: u32,
) {
    cumulative_values(g, cumulative, chunk_decay, heads, tokens, index());
}
#[inline(always)]
unsafe fn cumulative_values(g: *const f32, cumulative: *mut f32, chunk_decay: *mut f32,
    heads: u32, tokens: u32, row: usize) {
    if row >= heads as usize * (tokens as usize).div_ceil(C) { return; }
    let h = row % heads as usize;
    let start = row / heads as usize * C;
    let mut value = 0.0;
    let mut i = 0;
    while i < C {
        let t = start + i;
        if t < tokens as usize { value += *g.add(h * tokens as usize + t); }
        *cumulative.add(row * C + i) = value;
        i += 1;
    }
    *chunk_decay.add(row) = exp(value);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_cumulative_prompts(rows: *const PromptRow, heads: u32) {
    let row = &*rows.add(_block_idx_y() as usize);
    cumulative_values(row.g, row.cumulative, row.chunk_decay, heads, row.tokens, index());
}

/// Strictly lower UT coefficients and inclusive causal attention products.
/// 32x32 tiles share the FP32 operands across four cells per thread. Upper
/// tiles only clear masked cells. Every dot keeps its ascending FP32 sum.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_system(params: GdnSystem) { system(params); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_system_prompts(params: GdnChunkPrompts) {
    let GdnChunkPrompts { rows, heads, key_repeat, .. } = params;
    let r = &*(rows as *const PromptRow).add(_block_idx_y() as usize);
    system(GdnSystem { q: r.q, k: r.k, beta: r.beta, cumulative: r.cumulative, system: r.system,
        attention: r.attention, heads, tokens: r.tokens, key_repeat });
}
#[inline(always)]
unsafe fn system(params: GdnSystem) {
    let GdnSystem { q, k, beta, cumulative, system, attention, heads, tokens, key_repeat } = params;
    const THREADS: usize = GDN_SYSTEM_THREADS as usize;
    const ROWS: usize = 32;
    const COLS: usize = 32;
    const TILES: usize = C / ROWS;
    const R: usize = ROWS * COLS / THREADS;
    let block = _block_idx_x() as usize;
    let row = block / (TILES * TILES);
    let first = block % (TILES * TILES) / TILES * ROWS;
    let first_column = block % TILES * COLS;
    if row >= heads as usize * (tokens as usize).div_ceil(C) { return; }
    let h = row % heads as usize;
    let start = row / heads as usize * C;
    let tid = _thread_idx_x() as usize;
    let keys = 0u32;
    let queries = keys + (D * (COLS + 1) * 4) as u32;
    let learned = queries + (ROWS * D * 4) as u32;
    if first < first_column {
        for n in 0..R {
            let at = tid + n * THREADS;
            let target = (row * C + first + at / COLS) * C + first_column + at % COLS;
            *system.add(target) = 0.0;
            *attention.add(target) = 0.0;
        }
        return;
    }
    // Transposed keys give each lane a separate bank; row operands broadcast.
    stage::<{ COLS * D / THREADS }, _>(|n| {
        let at = tid + n * THREADS;
        let t = start + first_column + at / D;
        if t < tokens as usize { *k.add((h / key_repeat as usize * tokens as usize + t) * D + at % D) }
        else { 0.0 }
    }, |n, key| {
        let at = tid + n * THREADS;
        ptx::store(keys + ((at % D * (COLS + 1) + at / D) * 4) as u32, key);
    });
    stage::<{ ROWS * D / THREADS }, _>(|n| {
        let at = tid + n * THREADS;
        let t = start + first + at / D;
        if t < tokens as usize {
            let source = (h / key_repeat as usize * tokens as usize + t) * D + at % D;
            (*q.add(source), *k.add(source) * *beta.add(h * tokens as usize + t))
        } else { (0.0, 0.0) }
    }, |n, (query, learning)| {
        let at = tid + n * THREADS;
        ptx::store(queries + (at * 4) as u32, query);
        ptx::store(learned + (at * 4) as u32, learning);
    });
    ptx::sync();
    // Constant row indices keep both dot products in registers.
    macro_rules! rows {
        ($r:ident, $body:block) => {
            { const $r: usize = 0; $body } { const $r: usize = 1; $body }
            { const $r: usize = 2; $body } { const $r: usize = 3; $body }
        };
    }
    let j = first_column + tid % COLS;
    let group = tid / COLS;
    let mut kk = [0.0f32; R];
    let mut qk = [0.0f32; R];
    let mut d = 0;
    while d < D {
        let mut key = [0.0f32; 4];
        for z in 0..4 { key[z] = ptx::load(keys + (((d + z) * (COLS + 1) + tid % COLS) * 4) as u32); }
        rows!(ROW, {
            let at = (((group + ROW * (THREADS / COLS)) * D + d) * 4) as u32;
            let learning = ptx::load4(learned + at).map(f32::from_bits);
            let query = ptx::load4(queries + at).map(f32::from_bits);
            for z in 0..4 { kk[ROW] += learning[z] * key[z]; qk[ROW] += query[z] * key[z]; }
        });
        d += 4;
    }
    rows!(ROW, {
        let i = first + group + ROW * (THREADS / COLS);
        let (mut a, mut b) = (0.0, 0.0);
        if j <= i && start + i < tokens as usize && start + j < tokens as usize {
            let decay = exp(*cumulative.add(row * C + i) - *cumulative.add(row * C + j));
            a = kk[ROW] * decay;
            b = qk[ROW] * decay;
        }
        *system.add((row * C + i) * C + j) = if j < i { a } else { 0.0 };
        *attention.add((row * C + i) * C + j) = b;
    });
}

/// Canonical FP32 blocked triangular inverse, followed by its original
/// ascending RHS products. Keep the inverse on chip and reuse its scratch:
/// the two coordinate CTAs avoid the inverse global write/read and launch.
#[inline(always)]
unsafe fn inverse_tile(system: *const f32, row: usize) -> u32 {
    let tid = _thread_idx_x() as usize;
    let sh = ptx::shared::<49152>();
    let source = sh;
    let inverse = sh + (C * C * 4) as u32;
    let product = inverse + (C * C * 4) as u32;
    let mut at = tid;
    while at < C * C {
        ptx::store(source + (at * 4) as u32, *system.add(row * C * C + at));
        ptx::store(inverse + (at * 4) as u32, if at / C == at % C { 1.0 } else { 0.0 });
        at += 512;
    }
    ptx::sync();
    let mut stage = 0;
    while stage < 6 {
        let half = 1usize << stage;
        let cells = C / 2 * half;
        at = tid;
        while at < cells {
            let base = (at >> (2 * stage)) << (stage + 1);
            let i = base + half + ((at >> stage) & (half - 1));
            let j = base + (at & (half - 1));
            let mut value = 0.0;
            let mut k = base;
            while k < base + half {
                value += ptx::load(source + ((i * C + k) * 4) as u32)
                    * ptx::load(inverse + ((k * C + j) * 4) as u32);
                k += 1;
            }
            ptx::store(product + ((i * C + j) * 4) as u32, value);
            at += 512;
        }
        ptx::sync();
        at = tid;
        while at < cells {
            let base = (at >> (2 * stage)) << (stage + 1);
            let i = base + half + ((at >> stage) & (half - 1));
            let j = base + (at & (half - 1));
            let mut value = 0.0;
            let mut k = base + half;
            while k <= i {
                value += ptx::load(inverse + ((i * C + k) * 4) as u32)
                    * ptx::load(product + ((k * C + j) * 4) as u32);
                k += 1;
            }
            ptx::store(inverse + ((i * C + j) * 4) as u32, -value);
            at += 512;
        }
        ptx::sync();
        stage += 1;
    }
    inverse
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_solve(params: GdnSolve) { solve(params); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_solve_prompts(params: GdnChunkPrompts) {
    let GdnChunkPrompts { rows, heads, key_repeat, qkv, conv, keys, precision } = params;
    let r = &*(rows as *const PromptRow).add(_block_idx_y() as usize);
    solve(GdnSolve { qkv: (qkv as *const f32).add(r.offset as usize * (2 * keys + heads) as usize * D),
        conv: conv as _, history: r.history, keys, precision,
        k: r.k, v: core::ptr::null(), beta: r.beta, cumulative: r.cumulative,
        system: r.system, new_values: r.values, decayed_keys: r.decayed_keys,
        heads, tokens: r.tokens, key_repeat });
}
#[inline(always)]
unsafe fn solve(params: GdnSolve) {
    let GdnSolve { qkv, conv, history, keys: key_heads, precision, k, v, beta, cumulative, system, new_values, decayed_keys,
        heads, tokens, key_repeat } = params;
    const V: usize = 64;
    const THREADS: usize = 512;
    let block = _block_idx_x() as usize;
    let row = block / (D / V);
    if row >= heads as usize * (tokens as usize).div_ceil(C) { return; }
    let column = block % (D / V) * V;
    let h = row % heads as usize;
    let start = row / heads as usize * C;
    let tid = _thread_idx_x() as usize;
    let inverse = inverse_tile(system, row);
    // Source/product storage is dead after inversion; reuse it for the RHS.
    let values = inverse - (C * C * 4) as u32;
    let keys = inverse + (C * C * 4) as u32;
    stage::<{ C * V / THREADS }, _>(|n| {
        let at = tid + n * THREADS;
        let t = start + at / V;
        if t < tokens as usize {
            let source = (h * tokens as usize + t) * D + column + at % V;
            let key = k.add((h / key_repeat as usize * tokens as usize + t) * D + column + at % V);
            // Right-hand side beta*K*exp(prefix), decayed on load.
            // Consume the canonical convolution/SiLU value here, avoiding its
            // full prompt write/read. History is committed only after solve.
            let learning = *beta.add(h * tokens as usize + t);
            let value = if v.is_null() {
                crate::flash_gdn::convolution(qkv, conv, history,
                    (2 * key_heads as usize + h) * D + column + at % V, t,
                    (2 * key_heads + heads) as usize * D, precision) * learning
            } else { *v.add(source) };
            (value, (*key * learning) * exp(*cumulative.add(row * C + at / V)))
        } else { (0.0, 0.0) }
    }, |n, (value, key)| {
        let at = tid + n * THREADS;
        ptx::store(values + (at * 4) as u32, value);
        ptx::store(keys + (at * 4) as u32, key);
    });
    ptx::sync();
    let mut at = tid;
    while at < C * V {
        let i = at / V;
        let d = at % V;
        let mut value = 0.0;
        let mut key = 0.0;
        let mut j = 0;
        while j <= i {
            let coefficient = ptx::load(inverse + ((i * C + j) * 4) as u32);
            value += coefficient * ptx::load(values + ((j * V + d) * 4) as u32);
            key += coefficient * ptx::load(keys + ((j * V + d) * 4) as u32);
            j += 1;
        }
        let output = (row * C + i) * D + column + d;
        *new_values.add(output) = value;
        *decayed_keys.add(output) = key;
        at += THREADS;
    }
}

/// A head's value columns are independent. Each CTA owns 64 columns and keeps
/// its prior state across every chunk. Each thread holds register tiles (two
/// rows x four columns for predicted/inter/intra, 4x4 for the state update) so
/// shared operands are reused; every output keeps one FP32 accumulator in
/// ascending reduction order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_chunk_scan(params: GdnScan) { scan(params); }
/// Whole prompts from chunk 0; each descriptor row owns its state.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_scan_prompts(params: GdnScanPrompts) {
    let GdnScanPrompts { rows, heads, key_repeat } = params;
    let r = &*(rows as *const PromptRow).add(_block_idx_y() as usize);
    scan(GdnScan { q: r.q, k: r.k, attention: r.attention, new_values: r.values, decayed_keys: r.decayed_keys,
        chunk_decay: r.chunk_decay, cumulative: r.cumulative, state: r.state, output: r.core,
        heads, tokens: r.tokens, start: 0, key_repeat });
}
#[inline(always)]
unsafe fn scan(params: GdnScan) {
    let GdnScan { q, k, attention, new_values, decayed_keys, chunk_decay, cumulative, state, output, heads, tokens, start, key_repeat } = params;
    const V: usize = GDN_SCAN_COLUMNS;
    const THREADS: usize = GDN_SCAN_THREADS as usize;
    const X: usize = V / 4; // column lanes: thread columns 4tx..4tx+3
    const Y: usize = THREADS / X; // row lanes: rows ty and Y+ty
    const _: () = assert!(C == 2 * Y && D == 4 * Y && C / 2 * D == 8 * THREADS && C * C == 8 * THREADS);
    macro_rules! halves {
        ($half:ident, $body:block) => {
            { const $half: usize = 0; $body }
            { const $half: usize = 1; $body }
        };
    }
    let block = _block_idx_x() as usize;
    let h = block / (D / V);
    if h >= heads as usize { return; }
    let column = block % (D / V) * V;
    let tid = _thread_idx_x() as usize;
    let (tx, ty) = (tid % X, tid / X);
    let qk_head = h / key_repeat as usize;
    let prior = 0u32;
    let tile = prior + (D * V * 4) as u32;
    let queries = tile + (C * D * 4) as u32;
    let final_keys = tile + (C * C * 4) as u32;
    let correction = final_keys + (C * D * 4) as u32;
    let load4 = |at: u32| ptx::load4(at).map(f32::from_bits);
    let mut at = tid;
    while at < D * V {
        ptx::store(prior + (at * 4) as u32, *state.add((h * D + at / V) * D + column + at % V));
        at += THREADS;
    }
    let mut start = start as usize;
    let mut staged = false;
    while start < tokens as usize {
        let chunk = start / C * heads as usize + h;
        let mut predicted = [[0.0f32; 4]; 2];
        let mut inter = [[0.0f32; 4]; 2];
        // Stage both row halves together and reuse each prior-state load.
        for n in 0..4 {
            let c = tid + n * THREADS;
            let row = c / (D / 4);
            let feature = c % (D / 4) * 4;
            if !staged || n >= 2 {
                ptx::copy16(tile + (c * 16) as u32, decayed_keys.add((chunk * C + row) * D + feature).cast());
            }
            if start + row < tokens as usize {
                ptx::copy16(queries + (c * 16) as u32, q.add((qk_head * tokens as usize + start + row) * D + feature).cast());
            } else {
                for z in 0..4 { ptx::store(queries + ((c * 4 + z) * 4) as u32, 0.0); }
            }
        }
        ptx::wait();
        for n in 0..4 {
            let c = tid + n * THREADS;
            let row = c / (D / 4);
            if start + row < tokens as usize {
                let initial = exp(*cumulative.add(chunk * C + row));
                for z in 0..4 {
                    let at = queries + ((c * 4 + z) * 4) as u32;
                    ptx::store(at, ptx::load(at) * initial);
                }
            }
        }
        ptx::sync();
        let mut key = 0;
        while key < D {
            let old: [[f32; 4]; 4] = core::array::from_fn(|z|
                load4(prior + (((key + z) * V + tx * 4) * 4) as u32));
            halves!(HALF, {
                let solved = load4(tile + (((HALF * Y + ty) * D + key) * 4) as u32);
                let query = load4(queries + (((HALF * Y + ty) * D + key) * 4) as u32);
                for z in 0..4 {
                    for n in 0..4 {
                        predicted[HALF][n] += solved[z] * old[z][n];
                        inter[HALF][n] += query[z] * old[z][n];
                    }
                }
            });
            key += 4;
        }
        ptx::sync();
        let sources: [[f32; 4]; 2] = core::array::from_fn(|half| core::array::from_fn(|n|
            *new_values.add((chunk * C + half * Y + ty) * D + column + tx * 4 + n)));
        // Reuse the tile for attention, then final-decay keys.
        for n in 0..2 {
            let c = tid + n * THREADS;
            ptx::copy16(tile + (c * 16) as u32, attention.add(chunk * C * C + c * 4).cast());
        }
        ptx::commit();
        for half in 0..2 {
            for n in 0..4 {
                ptx::store(correction + (((half * Y + ty) * V + tx * 4 + n) * 4) as u32, sources[half][n] - predicted[half][n]);
            }
        }
        // K rows [64][128]; rows past the prompt are zero.
        for n in 0..4 {
            let c = tid + n * THREADS;
            let row = c / (D / 4);
            if start + row < tokens as usize {
                ptx::copy16(final_keys + (c * 16) as u32, k.add((qk_head * tokens as usize + start + row) * D + c % (D / 4) * 4).cast());
            } else {
                for z in 0..4 { ptx::store(final_keys + ((c * 4 + z) * 4) as u32, 0.0); }
            }
        }
        ptx::commit();
        ptx::wait_one();
        ptx::sync();
        let mut intra = [[0.0f32; 4]; 2];
        let mut j = 0;
        while j < C {
            let weight: [[f32; 4]; 2] = core::array::from_fn(|half| load4(tile + (((half * Y + ty) * C + j) * 4) as u32));
            for z in 0..4 {
                let delta = load4(correction + (((j + z) * V + tx * 4) * 4) as u32);
                for half in 0..2 {
                    for n in 0..4 { intra[half][n] += weight[half][z] * delta[n]; }
                }
            }
            j += 4;
        }
        for half in 0..2 {
            let t = start + half * Y + ty;
            if t < tokens as usize {
                for n in 0..4 { *output.add((h * tokens as usize + t) * D + column + tx * 4 + n) = inter[half][n] + intra[half][n]; }
            }
        }
        ptx::sync();
        ptx::wait();
        for n in 0..4 {
            let c = tid + n * THREADS;
            let row = c / (D / 4);
            if start + row < tokens as usize {
                // K decays to the chunk end: k*exp(last - prefix).
                let scale = exp(*cumulative.add(chunk * C + C - 1) - *cumulative.add(chunk * C + row));
                for z in 0..4 {
                    let at = final_keys + ((c * 4 + z) * 4) as u32;
                    ptx::store(at, ptx::load(at) * scale);
                }
            }
        }
        ptx::sync();
        // Attention's first 16 KiB is dead. Read the next chunk's first
        // half while the current chunk updates its independent state rows.
        staged = start + C < tokens as usize;
        if staged {
            for n in 0..2 {
                let c = tid + n * THREADS;
                ptx::copy16(tile + (c * 16) as u32,
                    decayed_keys.add((chunk + heads as usize) * C * D + c * 4).cast());
            }
            ptx::commit();
        }
        // State rows ty + m*Y, columns 4tx..4tx+3.
        let mut updates = [[0.0f32; 4]; 4];
        let mut row = 0;
        while row < C {
            let delta = load4(correction + ((row * V + tx * 4) * 4) as u32);
            for m in 0..4 {
                let key = ptx::load(final_keys + ((row * D + ty + m * Y) * 4) as u32);
                for n in 0..4 { updates[m][n] += key * delta[n]; }
            }
            row += 1;
        }
        let decay = *chunk_decay.add(chunk);
        // Each thread owns its cells; every prior read finished before the last barrier.
        for m in 0..4 {
            for n in 0..4 {
                let at = prior + (((ty + m * Y) * V + tx * 4 + n) * 4) as u32;
                ptx::store(at, ptx::load(at) * decay + updates[m][n]);
            }
        }
        ptx::sync();
        start += C;
    }
    at = tid;
    while at < D * V {
        *state.add((h * D + at / V) * D + column + at % V) = ptx::load(prior + (at * 4) as u32);
        at += THREADS;
    }
}
