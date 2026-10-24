//! QSA softmax for decode, verify and pooled rows.
use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
use crate::ptx::{load4, shared};

macro_rules! four { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body }
    { const $i: usize = 2; $body } { const $i: usize = 3; $body }
}; }

/// main's stored softmax (one warp per row, same maximum, same exp, same
/// ascending denominator chain) with each 32-slot chunk's probabilities passed
/// through a 128-byte warp tile and read back as eight broadcast v4 loads
/// instead of 32 shuffles. Block 128 (4 rows), static shared 512 B.
#[inline(always)]
unsafe fn stored_softmax_v(saved: *mut f32, length: usize, lane: usize, tile: u32) -> f32 {
    use crate::layer::{exp, maximum};
    use crate::ptx::store;
    let mut highest = f32::NEG_INFINITY;
    let mut at = lane;
    while at < length { highest = highest.max(*saved.add(at)); at += 32; }
    highest = maximum(highest);
    let mut denominator = 0.0;
    let mut slot = 0;
    // the next chunk's score is loaded before this chunk's work (its slots are not written yet).
    let mut next = if lane < length { *saved.add(lane) } else { 0.0 };
    while slot < length {
        let mine = slot + lane < length;
        let score = next;
        if slot + 32 + lane < length { next = *saved.add(slot + 32 + lane); }
        let probability = if mine { exp(score - highest) } else { 0.0 };
        store(tile + (lane * 4) as u32, probability);
        core::arch::asm!("bar.warp.sync 0xffffffff;");
        let count = (length - slot).min(32);
        four!(Q, { four!(Q2, {
            let q = 4 * Q + Q2;
            if q < 8 {
                let v = load4(tile + (q * 16) as u32).map(f32::from_bits);
                four!(E, { if 4 * q + E < count { denominator += v[E]; } });
            }
        }); });
        core::arch::asm!("bar.warp.sync 0xffffffff;");
        if mine { *saved.add(slot + lane) = probability; }
        slot += 32;
    }
    denominator
}
#[inline(always)]
unsafe fn softmax_v(p: crate::kernel_params::QsaAttention, row: usize) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let tile = shared::<512>() + (warp * 128) as u32;
    if row >= p.query_heads as usize * p.tokens as usize { return; }
    let token = row / p.query_heads as usize;
    let scores = (p.scores as *mut f32).add(row * p.stride as usize);
    let denominator = stored_softmax_v(scores, crate::flash_qsa::selected_length(p.past as usize + token + 1), lane, tile);
    if lane == 0 { *(p.denominators as *mut f32).add(row) = denominator; }
}
/// Block 128, grid ceil(query rows / 4) (main's grid).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_softmax_v(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    softmax_v(p, _block_idx_x() as usize * 4 + _thread_idx_x() as usize / 32);
}
/// Sessions in grid-y, grid x = ceil(local query heads / 4) (main's grid).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_softmax_v_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = crate::flash_qsa::batch_row(p, row) else { return; };
    softmax_v(crate::flash_qsa::batch_attention(p, r, row), _block_idx_x() as usize * 4 + _thread_idx_x() as usize / 32);
}

/// the pooled decode prepare stages in one launch. Blocks [0, P) run
/// flash_qsa_prepare_batch's rows (P = (query + key heads) / 4 rounded up, its grid),
/// blocks [P, P + 2) run flash_qsa_index_prepare_batch's five index rows; same per-row
/// code and thread mapping as the two entries, so the same bytes. Block 128, grid (P + 2, rows).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_prepare2_batch(p: crate::kernel_params::QsaBatch) {
    use crate::flash_qsa::{batch_row, index_prepare, prepare};
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    let (first, block, threads) = ((p.query_heads + p.key_heads).div_ceil(4) as usize, _block_idx_x() as usize, core::arch::nvptx::_block_dim_x() as usize);
    let thread = block * threads + _thread_idx_x() as usize;
    if block < first {
        prepare((p.query_gate as *const f32).add(row * p.query_heads as usize * 512),
            (p.key as *const f32).add(row * p.key_heads as usize * 256),
            (p.value as *const f32).add(row * p.key_heads as usize * 256),
            r.norm as *const f32, (r.norm as *const f32).add(256), r.query as *mut f32,
            r.keys as *mut f32, r.values as *mut f32, p.query_heads, p.key_heads, 1, r.past, 0,
            core::ptr::null(), 0, 1e-6, core::ptr::null(), thread);
    } else {
        index_prepare((p.index as *const f32).add(row * 640), (r.norm as *const f32).add(512),
            r.index_query as *mut f32, r.raw_index as *mut f32, 1, r.past, 0, thread - first * threads);
    }
}
