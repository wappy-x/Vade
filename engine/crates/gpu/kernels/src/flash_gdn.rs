//! Head-sharded Flash GatedDeltaNet. Weights retain their checkpoint dtype;
//! projections, convolution history and recurrent matrices remain FP32.
use crate::kernel_params::GdnGate;
use crate::{index, layer::{exp, sqrt, sum}, math::bf16_to_f32 as bf};
#[path = "../../src/gdn_params.rs"]
mod gdn_params;
use gdn_params::{GdnDecodeRow as DecodeRow, GdnSessionRow, GdnConvBatch, GdnPrepareBatch, GdnStepBatch, GdnStepGateBatch};

#[inline(always)]
pub(super) fn rounded(x: f32) -> f32 {
    let bits = x.to_bits();
    // Rejected rows need no canonical NaN payload; keep the overnight path.
    if bits & 0x7f800000 == 0x7f800000 { return x; }
    #[cfg(target_arch = "nvptx64")]
    {
        let pair: u32;
        unsafe { core::arch::asm!("cvt.rn.bf16x2.f32 {pair}, {x}, {x};",
            pair=out(reg32)pair, x=in(reg32)x, options(nomem, nostack, pure)); }
        f32::from_bits(pair & 0xffff0000)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    { crate::bf16::rounded(x) }
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.lg2.approx.ftz.f"]
    fn log2(value: f32) -> f32;
}

#[inline(always)]
unsafe fn sigmoid(value: f32) -> f32 { 1.0 / (1.0 + exp(-value)) }

macro_rules! eight {
    ($i:ident, $body:block) => {
        { const $i: usize = 0; $body } { const $i: usize = 1; $body }
        { const $i: usize = 2; $body } { const $i: usize = 3; $body }
        { const $i: usize = 4; $body } { const $i: usize = 5; $body }
        { const $i: usize = 6; $body } { const $i: usize = 7; $body }
    };
}

#[inline(always)]
pub(super) unsafe fn gate_values(a: f32, b: f32, alog: *const u16, dt: *const u16,
    head: usize, precision: u32) -> (f32, f32) {
    let bias = if precision & 4 == 0 { bf(*dt.add(head)) } else { *dt.cast::<f32>().add(head) };
    let activation = a + bias;
    let softplus = if activation > 20.0 { activation } else {
        let exponential = exp(activation);
        let sum = 1.0 + exponential;
        if sum == 1.0 { exponential }
        else { (log2(sum) * core::f32::consts::LN_2) * (exponential / (sum - 1.0)) }
    };
    let learning = sigmoid(b);
    let log_decay = if precision & 2 == 0 { bf(*alog.add(head)) } else { *alog.cast::<f32>().add(head) };
    (if precision == 0 { learning } else { rounded(learning) }, -exp(log_decay) * softplus)
}

#[inline(always)]
pub(super) unsafe fn convolution(x: *const f32, w: *const u16, history: *const f32,
    channel: usize, token: usize, channels: usize, precision: u32) -> f32 {
    let mut value = 0.0;
    let mut tap = 0;
    while tap < 4 {
        let source = token as isize + tap as isize - 3;
        let sample = if source >= 0 {
            let sample = *x.add(source as usize * channels + channel);
            if precision == 0 { sample } else { rounded(sample) }
        } else {
            let at = channel * 4 + (source + 4) as usize;
            if precision == 0 { *history.add(at) } else { bf(*history.cast::<u16>().add(at)) }
        };
        value += sample * bf(*w.add(channel * 4 + tap));
        tap += 1;
    }
    if precision != 0 { value = rounded(value); }
    let value = value * sigmoid(value);
    if precision == 0 { value } else { rounded(value) }
}

#[inline(always)]
pub(crate) unsafe fn prepare_values(query: [f32; 4], key: [f32; 4], value: [f32; 4],
    a: *const f32, b: *const f32, alog: *const u16, dt: *const u16,
    q: *mut f32, k: *mut f32, v: *mut f32, beta: *mut f32, g: *mut f32,
    row: usize, qk_row: Option<usize>, lane: usize, heads: usize, tokens: usize, precision: u32, scale: bool) {
    let mut query_squares = 0.0;
    let mut key_squares = 0.0;
    for feature in 0..4 {
        query_squares += query[feature] * query[feature];
        key_squares += key[feature] * key[feature];
    }
    let query_scale = 1.0 / sqrt(sum(query_squares) + 1e-6);
    let key_scale = 1.0 / sqrt(sum(key_squares) + 1e-6);
    let head = row / tokens;
    let projection = row % tokens * heads + head;
    let learning = if a.is_null() { *beta.add(row) } else {
        let learning = sigmoid(*b.add(projection));
        if precision == 0 { learning } else { rounded(learning) }
    };
    for feature in 0..4 {
        let offset = row * 128 + lane + feature * 32;
        if let Some(qk_row) = qk_row {
            let at = qk_row * 128 + lane + feature * 32;
            *q.add(at) = (query[feature] * query_scale) * (1.0 / sqrt(128.0));
            *k.add(at) = key[feature] * key_scale;
        }
        // The chunk path consumes beta-scaled values; decode keeps them raw.
        if !v.is_null() { *v.add(offset) = if scale { value[feature] * learning } else { value[feature] }; }
    }
    // A null projection means the fused input GEMM already emitted the gates.
    if lane == 0 && !a.is_null() {
        let head = row / tokens;
        let projection = row % tokens * heads + head;
        let (learning, decay) = gate_values(*a.add(projection), *b.add(projection), alog, dt, head, precision);
        *beta.add(row) = learning;
        *g.add(row) = decay;
    }
}

/// Dense GDN compatibility projection with packed BF16 weights and rounded
/// BF16 input/output. Flash projections use the canonical split MMA kernels.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_mv(
    w: *const u16, x: *const f32, y: *mut f32,
    rows: u32, cols: u32, tokens: u32,
) {
    let thread = index();
    let output = thread / 32;
    let lane = thread % 32;
    if output >= rows as usize * tokens as usize { return; }
    let row = output % rows as usize;
    let token = output / rows as usize;
    let x = x.add(token * cols as usize);
    let mut accum = [0.0f32; 4];
    let mut col = lane * 4;
    while col < cols as usize {
        let mut j = 0;
        while j < 4 {
            let c = col + j;
            // The existing eight-row BF16 tile layout, including pair order.
            let pair = c % 16 / 2;
            let permuted = pair / 4 + (pair % 4) * 2;
            let offset = (row / 8 * (cols as usize / 16) + c / 16) * 128
                + row % 8 * 16 + permuted * 2 + c % 2;
            let x = *x.add(c);
            accum[j] += bf(*w.add(offset)) * rounded(x);
            j += 1;
        }
        col += 128;
    }
    let value = sum((accum[0] + accum[1]) + (accum[2] + accum[3]));
    if lane == 0 { *y.add(output) = rounded(value); }
}

/// Independent causal convolution rows; history is immutable until this launch
/// completes. State is [channels,4], oldest first, matching the cached layout.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv(
    x: *const f32, w: *const u16, state: *mut f32, y: *mut f32,
    channels: u32, tokens: u32, precision: u32,
) {
    let at = index();
    if at >= channels as usize * tokens as usize { return; }
    let channel = at % channels as usize;
    let token = at / channels as usize;
    *y.add(at) = convolution(x, w, state, channel, token, channels as usize, precision);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_history(p: crate::kernel_params::GdnHistory) { history_token(p); }
/// Grid (ceil(max words / 1024), 4) x 256: range blockIdx.y copies 16-byte group
/// g = x * 256 + tid (words 4g..4g+4; a misaligned range or its last partial group goes
/// word by word). CTA (0, 0) thread 0 also writes the descriptor. Ranges must not overlap
/// each other or the descriptor; every copied word keeps its stored bits. The range is
/// picked by branches on blockIdx.y (an indexed parameter array would spill to local).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_verify_copy(p: crate::kernel_params::GdnVerifyCopy) {
    use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x}};
    let (r, g) = (_block_idx_y(), _block_idx_x() as u64 * 256 + _thread_idx_x() as u64);
    let (source, destination, words) = match r {
        0 => (p.source[0], p.destination[0], p.words[0]),
        1 => (p.source[1], p.destination[1], p.words[1]),
        2 => (p.source[2], p.destination[2], p.words[2]),
        3 => (p.source[3], p.destination[3], p.words[3]),
        _ => (0, 0, 0),
    };
    let words = words as u64;
    if (source | destination) & 15 == 0 {
        if 4 * g + 4 <= words {
            let (a, b, c, d): (u32, u32, u32, u32);
            asm!("ld.global.v4.u32 {{{a}, {b}, {c}, {d}}}, [{s}];", a = out(reg32) a, b = out(reg32) b,
                c = out(reg32) c, d = out(reg32) d, s = in(reg64) source + 16 * g, options(readonly, nostack));
            asm!("st.global.v4.u32 [{t}], {{{a}, {b}, {c}, {d}}};", t = in(reg64) destination + 16 * g,
                a = in(reg32) a, b = in(reg32) b, c = in(reg32) c, d = in(reg32) d, options(nostack));
        } else {
            let mut i = 4 * g;
            while i < words && i < 4 * g + 4 {
                *(destination as *mut u32).add(i as usize) = *(source as *const u32).add(i as usize);
                i += 1;
            }
        }
    } else {
        let mut i = 4 * g;
        while i < words && i < 4 * g + 4 {
            *(destination as *mut u32).add(i as usize) = *(source as *const u32).add(i as usize);
            i += 1;
        }
    }
    if r == 0 && g == 0 && p.descriptor != 0 {
        let d = p.descriptor as *mut u64;
        *d = p.addresses[0];
        *d.add(1) = p.addresses[1];
        *d.add(2) = p.addresses[2];
    }
}
/// Cohort verify form (VADE_GDN_COHORT_KERNELS): grid-Y = chat, whose GdnHistory is entry y
/// of the table at `table`; flash_gdn_history's code per chat (each owns its history).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_history_cohort(table: u64) {
    history_token(core::ptr::read((table as *const crate::kernel_params::GdnHistory).add(core::arch::nvptx::_block_idx_y() as usize)));
}
#[inline(always)]
pub(crate) unsafe fn history_token(p: crate::kernel_params::GdnHistory) {
    let channel = index();
    history(p.x, p.history, p.channels, p.tokens, p.precision, channel);
    // Each thread owns all four history slots for one channel; publish the
    // retained prefix in the same launch, preserving the stored bits.
    if p.rows != 0 && channel < p.channels as usize {
        let words = if p.precision == 0 { 4 } else { 2 };
        for tap in 0..words {
            let at = channel * words + tap;
            *(p.rows as *mut u32).add(at) = *p.history.cast::<u32>().add(at);
        }
    }
}
#[inline(always)]
unsafe fn history(x: *const f32, state: *mut f32, channels: u32, tokens: u32, precision: u32, channel: usize) {
    if channel >= channels as usize { return; }
    let mut history = [0.0; 4];
    let mut tap = 0;
    while tap < 4 {
        let source = tokens as isize + tap as isize - 4;
        history[tap] = if source >= 0 {
            *x.add(source as usize * channels as usize + channel)
        } else {
            let at = channel * 4 + (source + 4) as usize;
            if precision == 0 { *state.add(at) } else { bf(*state.cast::<u16>().add(at)) }
        };
        tap += 1;
    }
    tap = 0;
    while tap < 4 {
        let at = channel * 4 + tap;
        if precision == 0 { *state.add(at) = history[tap]; }
        else { *state.cast::<u16>().add(at) = (rounded(history[tap]).to_bits() >> 16) as u16; }
        tap += 1;
    }
}

/// Expand shared Q/K heads using their global ownership, L2-normalize them,
/// and compute beta/g. Q includes 1/sqrt(128). Outputs are head-major:
/// q/k/v [value_heads,tokens,128], beta/g [value_heads,tokens].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_prepare(
    qkv: *const f32, a: *const f32, b: *const f32,
    alog: *const u16, dt: *const u16,
    q: *mut f32, k: *mut f32, v: *mut f32,
    beta: *mut f32, g: *mut f32,
    key_heads: u32, value_heads: u32, key_start: u32, value_start: u32,
    tokens: u32, precision: u32, scale: u32,
) {
    prepare(qkv, a, b, alog, dt, q, k, v, beta, g, key_heads, value_heads,
        key_start, value_start, tokens, precision, scale != 0, index());
}
#[inline(always)]
unsafe fn prepare(qkv: *const f32, a: *const f32, b: *const f32, alog: *const u16, dt: *const u16,
    q: *mut f32, k: *mut f32, v: *mut f32, beta: *mut f32, g: *mut f32,
    key_heads: u32, value_heads: u32, key_start: u32, value_start: u32,
    tokens: u32, precision: u32, scale: bool, thread: usize) {
    let row = thread / 32;
    let lane = thread % 32;
    if row >= value_heads as usize * tokens as usize { return; }
    let head = row / tokens as usize;
    let token = row % tokens as usize;
    let repeats = if precision != 0 { value_heads / key_heads } else { 3 };
    let key_head = (head + value_start as usize) / repeats as usize - key_start as usize;
    let channels = (key_heads as usize * 2 + value_heads as usize) * 128;
    let input = qkv.add(token * channels);
    let query = input.add(key_head * 128);
    let key = input.add((key_heads as usize + key_head) * 128);
    let value = input.add((key_heads as usize * 2 + head) * 128);
    prepare_values(core::array::from_fn(|i| *query.add(lane + i * 32)),
        core::array::from_fn(|i| *key.add(lane + i * 32)),
        core::array::from_fn(|i| *value.add(lane + i * 32)),
        a, b, alog, dt, q, k, v, beta, g, row, Some(row), lane, value_heads as usize, tokens as usize, precision, scale);
}

/// Convolve, activate and normalize directly into the head-major chunk input.
/// One warp per (key head, token) convolves Q/K once for every value head that
/// shares them (value_start must be a multiple of the repeat count). The chunk
/// path (scale) stores Q/K once per key head; decode keeps one per value head.
/// History remains read-only until the following history-update launch.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare(p: crate::kernel_params::GdnPrepare) {
    let crate::kernel_params::GdnPrepare { x, w, history, a, b, alog, dt, q, k, v, beta, g,
        key_heads, value_heads, key_start, value_start, tokens, precision, scale } = p;
    conv_prepare(x, w, history, a, b, alog, dt, q, k, v, beta, g,
        key_heads, value_heads, key_start, value_start, tokens, precision, scale != 0, index());
}
/// The single-row arithmetic above, with independent token-major operands.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_rows(p: crate::kernel_params::GdnPrepareRows) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let p0 = p.prepare;
    let hd = row * p0.value_heads as usize * 128;
    let heads = row * p0.value_heads as usize;
    let history = (*(p.rows as *const DecodeRow).add(row)).history as *const f32;
    conv_prepare(p0.x.add(row * (2 * p0.key_heads + p0.value_heads) as usize * 128), p0.w, history,
        p0.a.add(heads), p0.b.add(heads), p0.alog, p0.dt, p0.q.add(hd), p0.k.add(hd), p0.v.add(hd),
        p0.beta.add(heads), p0.g.add(heads), p0.key_heads, p0.value_heads, p0.key_start, p0.value_start,
        1, p0.precision, false, index());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_history_rows(p: crate::kernel_params::GdnHistory) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let state = &*(p.rows as *const DecodeRow).add(row);
    history(p.x.add(row * p.channels as usize), state.history as *mut f32,
        p.channels, 1, p.precision, index());
}
#[inline(always)]
unsafe fn conv_prepare(
    x: *const f32, w: *const u16, history: *const f32, a: *const f32, b: *const f32,
    alog: *const u16, dt: *const u16, q: *mut f32, k: *mut f32, v: *mut f32,
    beta: *mut f32, g: *mut f32, key_heads: u32, value_heads: u32,
    key_start: u32, value_start: u32, tokens: u32, precision: u32, scale: bool, thread: usize,
) {
    let row = thread / 32;
    let lane = thread % 32;
    let repeats = if precision == 0 { 3 } else { value_heads / key_heads };
    if row >= (value_heads / repeats) as usize * tokens as usize { return; }
    let first = (row as u32 / tokens) * repeats;
    let token = (row as u32 % tokens) as usize;
    let key_head = (((first + value_start) / repeats) - key_start) as usize;
    let channels = ((key_heads * 2 + value_heads) * 128) as usize;
    let conv = |channel| convolution(x, w, history, channel, token, channels, precision);
    let load = |base| [conv(base + lane), conv(base + lane + 32),
        conv(base + lane + 64), conv(base + lane + 96)];
    let query = load(key_head * 128);
    let key = load((key_heads as usize + key_head) * 128);
    for head in first as usize..(first + repeats) as usize {
        let value = if v.is_null() { [0.0; 4] } else { load((key_heads as usize * 2 + head) * 128) };
        let row = head * tokens as usize + token;
        let qk_row = if !scale { Some(row) } else if head == first as usize { Some(key_head * tokens as usize + token) } else { None };
        prepare_values(query, key, value, a, b, alog, dt, q, k, v, beta, g,
            row, qk_row, lane, value_heads as usize, tokens as usize, precision, scale);
    }
}

/// One token only: each channel is convolved by exactly one lane, so after
/// conv_prepare has read that channel's history taps the same lane may publish
/// its new history (and the optional snapshot words) for it. No other lane
/// reads or writes that channel's history, and the step never reads history.
/// History precision 1 matches the typed decode update launch verbatim.
#[inline(always)]
unsafe fn conv_prepare_hist(
    x: *const f32, w: *const u16, hist: *const f32, a: *const f32, b: *const f32,
    alog: *const u16, dt: *const u16, q: *mut f32, k: *mut f32, v: *mut f32,
    beta: *mut f32, g: *mut f32, key_heads: u32, value_heads: u32,
    key_start: u32, value_start: u32, precision: u32, snapshot: u64, thread: usize,
) {
    conv_prepare(x, w, hist, a, b, alog, dt, q, k, v, beta, g,
        key_heads, value_heads, key_start, value_start, 1, precision, false, thread);
    let row = thread / 32;
    let lane = thread % 32;
    let repeats = if precision == 0 { 3 } else { value_heads / key_heads };
    if row >= (value_heads / repeats) as usize { return; }
    let first = row as u32 * repeats;
    let key_head = (((first + value_start) / repeats) - key_start) as usize;
    let channels = ((key_heads * 2 + value_heads) * 128) as usize;
    for section in 0..2 + repeats as usize {
        let base = match section {
            0 => key_head * 128,
            1 => (key_heads as usize + key_head) * 128,
            s => (key_heads as usize * 2 + first as usize + s - 2) * 128,
        };
        for offset in [0, 32, 64, 96] {
            let channel = base + lane + offset;
            history(x, hist as *mut f32, channels as u32, 1, 1, channel);
            if snapshot != 0 {
                for tap in 0..2 {
                    let at = channel * 2 + tap;
                    *(snapshot as *mut u32).add(at) = *hist.cast::<u32>().add(at);
                }
            }
        }
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_hist(p: crate::kernel_params::GdnPrepareHist) {
    let p0 = p.prepare;
    if p.rows == 0 {
        conv_prepare_hist(p0.x, p0.w, p0.history, p0.a, p0.b, p0.alog, p0.dt, p0.q, p0.k, p0.v,
            p0.beta, p0.g, p0.key_heads, p0.value_heads, p0.key_start, p0.value_start,
            p0.precision, p.snapshot, index());
    } else {
        let row = core::arch::nvptx::_block_idx_y() as usize;
        let hd = row * p0.value_heads as usize * 128;
        let heads = row * p0.value_heads as usize;
        let history = (*(p.rows as *const DecodeRow).add(row)).history as *const f32;
        conv_prepare_hist(p0.x.add(row * (2 * p0.key_heads + p0.value_heads) as usize * 128), p0.w, history,
            p0.a.add(heads), p0.b.add(heads), p0.alog, p0.dt, p0.q.add(hd), p0.k.add(hd), p0.v.add(hd),
            p0.beta.add(heads), p0.g.add(heads), p0.key_heads, p0.value_heads, p0.key_start, p0.value_start,
            p0.precision, 0, index());
    }
}

/// Independent prompt work, token-major for the unchanged ordered step.
/// Read the tile-entry history; each causal tap, warp sum and BF16 boundary
/// is identical to conv_prepare(tokens=1) after earlier history publications.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_tokens(p: crate::kernel_params::GdnPrepare) { conv_prepare_tokens(p); }
/// Chunk-form prompts (VADE_GDN_CHUNK=1): the same words, but q/k are written
/// once per key group, on its first value head h0 = key * repeats, the only
/// q/k row the chunk prep, scan, rows and slot kernels read. Non-order: every
/// written word equals flash_gdn_conv_prepare_tokens'; 2/3 of the q/k bytes go.
/// With v null no value rows are convolved or written (the chunk scan does it).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_tokens_kh(p: crate::kernel_params::GdnPrepare) { conv_prepare_tokens_with::<true>(p); }
/// flash_gdn_conv_prepare_tokens_kh with one warp per (token, key group, value head)
/// instead of one per (token, key group) looping over the group's value heads: every warp
/// convolves its group's q/k sections (the same taps, sums and scales, so the same words)
/// and prepares its own head; the group's first head writes q/k, exactly the words the
/// looping entry writes. Verify windows (few tokens) get value-heads/key-heads times the
/// warps. Grid ceil(key_heads * tokens * repeats * 32 / 128) x 128.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_tokens_khs(p: crate::kernel_params::GdnPrepare) {
    let thread=index(); let lane=thread%32; let warp=thread/32;
    let keys=p.key_heads as usize; let repeats=(p.value_heads/p.key_heads) as usize;
    if repeats==0 || warp>=keys*p.tokens as usize*repeats { return; }
    let (row, rep)=(warp/repeats, warp%repeats);
    let token=row/keys; let key=row%keys;
    let channels=(2*p.key_heads+p.value_heads) as usize*128;
    let conv=|channel| convolution(p.x,p.w,p.history,channel,token,channels,p.precision);
    let load=|base| [conv(base+lane),conv(base+lane+32),conv(base+lane+64),conv(base+lane+96)];
    let query=load(key*128); let key_values=load((keys+key)*128);
    let head=key*repeats+rep;
    let value=if p.v.is_null() { [0.0; 4] } else { load((2*keys+head)*128) };
    let features=token*p.value_heads as usize*128; let heads=token*p.value_heads as usize;
    prepare_values(query,key_values,value,p.a.add(heads),p.b.add(heads),p.alog,p.dt,
        p.q.add(features),p.k.add(features),if p.v.is_null() { p.v } else { p.v.add(features) },p.beta.add(heads),p.g.add(heads),
        head,if rep==0 { Some(head) } else { None },lane,p.value_heads as usize,1,p.precision,false);
}
/// Cohort verify form (VADE_GDN_COHORT_KERNELS): grid-Y = chat, whose GdnPrepare is entry y
/// of the table at `table`; flash_gdn_conv_prepare_tokens_kh's code per chat (blocks past a
/// chat's rows return as there; each chat writes only its own rows).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_tokens_kh_cohort(table: u64) {
    conv_prepare_tokens_with::<true>(core::ptr::read((table as *const crate::kernel_params::GdnPrepare).add(core::arch::nvptx::_block_idx_y() as usize)));
}
#[inline(always)]
pub(crate) unsafe fn conv_prepare_tokens(p: crate::kernel_params::GdnPrepare) { conv_prepare_tokens_with::<false>(p); }
#[inline(always)]
unsafe fn conv_prepare_tokens_with<const FIRST_QK: bool>(p: crate::kernel_params::GdnPrepare) {
    let thread=index(); let lane=thread%32; let row=thread/32;
    if row>=p.key_heads as usize*p.tokens as usize { return; }
    let token=row/p.key_heads as usize; let key=row%p.key_heads as usize;
    let repeats=p.value_heads/p.key_heads; let channels=(2*p.key_heads+p.value_heads) as usize*128;
    let conv=|channel| convolution(p.x,p.w,p.history,channel,token,channels,p.precision);
    let load=|base| [conv(base+lane),conv(base+lane+32),conv(base+lane+64),conv(base+lane+96)];
    let query=load(key*128); let key_values=load((p.key_heads as usize+key)*128);
    let first=key*repeats as usize;
    for head in first..first+repeats as usize {
        // A null v: the chunk scan convolves the value rows itself (beta/g still here).
        let value=if p.v.is_null() { [0.0; 4] } else { load((2*p.key_heads as usize+head)*128) };
        let features=token*p.value_heads as usize*128; let heads=token*p.value_heads as usize;
        prepare_values(query,key_values,value,p.a.add(heads),p.b.add(heads),p.alog,p.dt,
            p.q.add(features),p.k.add(features),if p.v.is_null() { p.v } else { p.v.add(features) },p.beta.add(heads),p.g.add(heads),
            head,if !FIRST_QK || head==first { Some(head) } else { None },lane,p.value_heads as usize,1,p.precision,false);
    }
}

/// One decode token (precision 0, raw v), one warp per 128-channel qkv
/// section: query sections 0..keys, key sections keys..2*keys, then one value
/// section per value head. Every output word is computed exactly as
/// conv_prepare_tokens(tokens = 1) computes it (same taps in the same order,
/// same per-lane squares, the same warp sum, scales and gates); the q/k rows of
/// a key group are written by its query/key warp for each value head. Each lane
/// then publishes the new FP32 history of the four channels it convolved, the
/// words history_token writes ([tap 1, tap 2, tap 3, x]), so the separate
/// history launch is gone. No lane reads another lane's channel, so the
/// in-place publication cannot race, and nothing else reads history until the
/// next token.
#[inline(always)]
pub(crate) unsafe fn conv_prepare_section(p: crate::kernel_params::GdnPrepare, history: *mut f32, section: usize, lane: usize) {
    let (keys, values) = (p.key_heads as usize, p.value_heads as usize);
    if section >= 2 * keys + values { return; }
    let repeats = values / keys;
    let mut taps = [[0.0f32; 4]; 4];
    let mut out = [0.0f32; 4];
    for f in 0..4 {
        let c = section * 128 + lane + f * 32;
        let s = [*history.add(c * 4 + 1), *history.add(c * 4 + 2), *history.add(c * 4 + 3), *p.x.add(c)];
        let mut value = 0.0;
        for tap in 0..4 { value += s[tap] * bf(*p.w.add(c * 4 + tap)); }
        out[f] = value * sigmoid(value);
        taps[f] = s;
    }
    if section < 2 * keys {
        let mut squares = 0.0;
        for f in 0..4 { squares += out[f] * out[f]; }
        let scale = 1.0 / sqrt(sum(squares) + 1e-6);
        let query = section < keys;
        let group = if query { section } else { section - keys };
        let dst = if query { p.q } else { p.k };
        for head in group * repeats..(group + 1) * repeats {
            for f in 0..4 {
                *dst.add(head * 128 + lane + f * 32) =
                    if query { (out[f] * scale) * (1.0 / sqrt(128.0)) } else { out[f] * scale };
            }
        }
    } else {
        let head = section - 2 * keys;
        for f in 0..4 { *p.v.add(head * 128 + lane + f * 32) = out[f]; }
        if lane == 0 {
            let (learning, decay) = gate_values(*p.a.add(head), *p.b.add(head), p.alog, p.dt, head, p.precision);
            *p.beta.add(head) = learning;
            *p.g.add(head) = decay;
        }
    }
    for f in 0..4 {
        let c = section * 128 + lane + f * 32;
        for tap in 0..4 { *history.add(c * 4 + tap) = taps[f][tap]; }
    }
}

/// One cached token; state [heads,128 key,128 value] is updated in place.
/// Each value coordinate uses a fixed ascending key reduction.
/// Block128 places one complete head in each CTA.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_step(
    q: *const f32, k: *const f32, v: *const f32,
    beta: *const f32, g: *const f32, state: *mut f32, out: *mut f32,
    heads: u32,
) {
    step(q, k, v, beta, g, state, state, out, heads, index());
}
#[inline(always)]
unsafe fn state_load(p: *const f32) -> f32 {
    let value: f32;
    core::arch::asm!("ld.global.L1::evict_first.L2::256B.f32 {v}, [{p}];",
        v=out(reg32)value, p=in(reg64)p, options(readonly, nostack));
    value
}

#[inline(always)]
unsafe fn state_store(p: *mut f32, value: f32) {
    core::arch::asm!("st.global.cs.f32 [{p}], {v};",
        p=in(reg64)p, v=in(reg32)value, options(nostack));
}

#[inline(always)]
unsafe fn step(q: *const f32, k: *const f32, v: *const f32, beta: *const f32, g: *const f32,
    state: *const f32, next_state: *mut f32, out: *mut f32, heads: u32, output: usize) {
    if output >= heads as usize * 128 { return; }
    *out.add(output) = step_value(q, k, v, beta, g, state, next_state, heads, output);
}
#[inline(always)]
unsafe fn step_value(q: *const f32, k: *const f32, v: *const f32, beta: *const f32, g: *const f32,
    state: *const f32, next_state: *mut f32, heads: u32, output: usize) -> f32 {
    if output >= heads as usize * 128 { return 0.0; }
    let head = output / 128;
    let feature = output % 128;
    let query = q.add(head * 128);
    let key = k.add(head * 128);
    let state = state.add(head * 16384 + feature);
    let next_state = next_state.add(head * 16384 + feature);
    let decay = exp(*g.add(head));
    // Constant indices keep the decayed column in registers across both
    // ascending reductions: one global read and one write per state cell.
    let mut cells = [0.0f32; 128];
    let mut prediction = 0.0;
    eight!(GROUP, { eight!(PAIR, {
        let coordinate = GROUP * 16 + PAIR * 2;
        cells[coordinate] = state_load(state.add(coordinate * 128)) * decay;
        prediction += *key.add(coordinate) * cells[coordinate];
        cells[coordinate + 1] = state_load(state.add((coordinate + 1) * 128)) * decay;
        prediction += *key.add(coordinate + 1) * cells[coordinate + 1];
    }); });
    let delta = (*v.add(output) - prediction) * *beta.add(head);
    let mut value = 0.0;
    eight!(GROUP, { eight!(PAIR, {
        let coordinate = GROUP * 16 + PAIR * 2;
        let next = cells[coordinate] + *key.add(coordinate) * delta;
        state_store(next_state.add(coordinate * 128), next);
        value += *query.add(coordinate) * next;
        let next = cells[coordinate + 1] + *key.add(coordinate + 1) * delta;
        state_store(next_state.add((coordinate + 1) * 128), next);
        value += *query.add(coordinate + 1) * next;
    }); });
    value
}

// Immediate offsets avoid keeping 128 state addresses live across the loop.
// Reload immutable keys instead of retaining another 128 values beside cells.
#[inline(always)]
unsafe fn scan_load<const OFFSET: usize>(p: *const f32) -> f32 {
    let value: f32;
    core::arch::asm!("ld.global.f32 {v}, [{p}+{offset}];", v=out(reg32)value,
        p=in(reg64)p, offset=const OFFSET, options(readonly, nostack));
    value
}
#[inline(always)]
unsafe fn scan_store<const OFFSET: usize>(p: *mut f32, value: f32) {
    core::arch::asm!("st.global.cs.f32 [{p}+{offset}], {v};", p=in(reg64)p,
        offset=const OFFSET, v=in(reg32)value, options(nostack));
}

#[inline(always)]
unsafe fn scan_shared<const OFFSET: usize>(p: u32) -> f32 {
    let value: f32;
    core::arch::asm!("ld.shared.f32 {v}, [{p}+{offset}];", v=out(reg32)value,
        p=in(reg32)p, offset=const OFFSET, options(readonly, nostack));
    value
}

/// Consecutive live prompt tokens, token-major inputs/output; count is tokens.
/// The host admits only identical state/history descriptors without snapshots.
/// Each thread retains its 128 FP32 cells until the end of the tile.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_scan_exact(p: GdnStepBatch) {
    let output = index();
    if output >= p.heads as usize * 128 { return; }
    let head = output / 128;
    let feature = output % 128;
    let width = p.heads as usize * 128;
    let state = ((&*(p.rows as *const DecodeRow)).state as *mut f32).add(head * 16384 + feature);
    let mut cells = [0.0f32; 128];
    eight!(GROUP, { eight!(PAIR, {
        const COORDINATE: usize = GROUP * 16 + PAIR * 2;
        cells[COORDINATE] = scan_load::<{ COORDINATE * 512 }>(state);
        cells[COORDINATE + 1] = scan_load::<{ (COORDINATE + 1) * 512 }>(state);
    }); });
    // Cooperatively stage immutable operands; the recurrence still visits each
    // token and each key coordinate in the canonical ascending order.
    let shared = crate::ptx::shared::<4096>();
    let lane = core::arch::nvptx::_thread_idx_x() as usize;
    for tile in (0..p.count as usize).step_by(4) {
        let count = core::cmp::min(4, p.count as usize - tile);
        for t in 0..count {
            let at = (tile + t) * width + head * 128 + lane * 4;
            crate::ptx::copy16(shared + (t * 1024 + lane * 16) as u32, (p.q as *const u32).add(at));
            crate::ptx::copy16(shared + (t * 1024 + 512 + lane * 16) as u32, (p.k as *const u32).add(at));
        }
        crate::ptx::wait();
        crate::ptx::warp_sync();
        for t in 0..count {
        let token = tile + t;
        let query = shared + (t * 1024) as u32;
        let key = query + 512;
        let decay = exp(*(p.g as *const f32).add(token * p.heads as usize + head));
        let mut prediction = 0.0;
        eight!(GROUP, { eight!(PAIR, {
            const COORDINATE: usize = GROUP * 16 + PAIR * 2;
            cells[COORDINATE] *= decay;
            prediction += scan_shared::<{ COORDINATE * 4 }>(key) * cells[COORDINATE];
            cells[COORDINATE + 1] *= decay;
            prediction += scan_shared::<{ (COORDINATE + 1) * 4 }>(key) * cells[COORDINATE + 1];
        }); });
        let delta = (*(p.v as *const f32).add(token * width + output) - prediction)
            * *(p.beta as *const f32).add(token * p.heads as usize + head);
        let mut value = 0.0;
        eight!(GROUP, { eight!(PAIR, {
            const COORDINATE: usize = GROUP * 16 + PAIR * 2;
            let next = cells[COORDINATE] + scan_shared::<{ COORDINATE * 4 }>(key) * delta;
            cells[COORDINATE] = next;
            value += scan_shared::<{ COORDINATE * 4 }>(query) * next;
            let next = cells[COORDINATE + 1] + scan_shared::<{ (COORDINATE + 1) * 4 }>(key) * delta;
            cells[COORDINATE + 1] = next;
            value += scan_shared::<{ (COORDINATE + 1) * 4 }>(query) * next;
        }); });
        *(p.out as *mut f32).add(token * width + output) = value;
        }
        crate::ptx::warp_sync();
    }
    eight!(GROUP, { eight!(PAIR, {
        const COORDINATE: usize = GROUP * 16 + PAIR * 2;
        scan_store::<{ COORDINATE * 512 }>(state, cells[COORDINATE]);
        scan_store::<{ (COORDINATE + 1) * 512 }>(state, cells[COORDINATE + 1]);
    }); });
}


/// Session-owned workspaces and state; only projected inputs are pooled.
#[repr(C)]
pub(super) struct PromptRow {
    pub state: *mut f32, pub history: *mut f32,
    pub q: *mut f32, pub k: *mut f32, pub v: *mut f32, pub beta: *mut f32, pub g: *mut f32,
    pub core: *mut f32, pub cumulative: *mut f32, pub system: *mut f32, pub attention: *mut f32,
    pub values: *mut f32, pub decayed_keys: *mut f32, pub chunk_decay: *mut f32,
    pub offset: u32, pub tokens: u32,
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_prepare_prompts(
    x: *const f32, w: *const u16, a: *const f32, b: *const f32, alog: *const u16, dt: *const u16,
    rows: *const PromptRow, keys: u32, heads: u32, precision: u32,
) {
    let row = &*rows.add(core::arch::nvptx::_block_idx_y() as usize);
    conv_prepare(x.add(row.offset as usize * (2 * keys + heads) as usize * 128), w, row.history,
        a.add(row.offset as usize * heads as usize), b.add(row.offset as usize * heads as usize), alog, dt,
        row.q, row.k, core::ptr::null_mut(), row.beta, row.g, keys, heads, 0, 0, row.tokens, precision, true, index());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_history_prompts(p: crate::kernel_params::GdnHistory) {
    let row = &*(p.rows as *const PromptRow).add(core::arch::nvptx::_block_idx_y() as usize);
    history(p.x.add(row.offset as usize * p.channels as usize), row.history, p.channels, row.tokens, p.precision, index());
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_conv_batch(p: GdnConvBatch) {
    let (channels, count) = (p.channels, p.count);
    let thread = index();
    let slot = thread / channels as usize;
    if slot >= count as usize { return; }
    let (row, cache, y) = if p.sessions == 0 {
        (slot, (*(p.rows as *const DecodeRow).add(slot)).history as *mut f32,
            (p.y as *mut f32).add(slot * channels as usize))
    } else {
        let s = &*(p.sessions as *const GdnSessionRow).add(slot);
        if s.padding != 0 { return; }
        (s.row as usize, s.history as *mut f32, s.convolved as *mut f32)
    };
    let offset = row * channels as usize;
    let channel = thread % channels as usize;
    let x = (p.x as *const f32).add(offset);
    *y.add(channel) = convolution(x, p.w as *const u16, cache, channel, 0, channels as usize, p.precision);
    history(x, cache, channels, 1, p.precision, channel);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_prepare_batch(p: GdnPrepareBatch) {
    let (key_heads, value_heads, count) = (p.key_heads, p.value_heads, p.count);
    let threads = value_heads as usize * 32;
    let thread = index();
    let row = thread / threads;
    if row >= count as usize { return; }
    let hd = row * value_heads as usize * 128;
    let heads = row * value_heads as usize;
    let (input, qkv, q, k, v, beta, g) = if p.sessions == 0 {
        (row, (p.qkv as *const f32).add(row * (2 * key_heads + value_heads) as usize * 128),
            (p.q as *mut f32).add(hd), (p.k as *mut f32).add(hd), (p.v as *mut f32).add(hd),
            (p.beta as *mut f32).add(heads), (p.g as *mut f32).add(heads))
    } else {
        let s = &*(p.sessions as *const GdnSessionRow).add(row);
        if s.padding != 0 { return; }
        (s.row as usize, s.convolved as *const f32, s.q as *mut f32, s.k as *mut f32,
            s.v as *mut f32, s.beta as *mut f32, s.g as *mut f32)
    };
    prepare(qkv, (p.a as *const f32).add(input * value_heads as usize),
        (p.b as *const f32).add(input * value_heads as usize), p.alog as *const u16, p.dt as *const u16,
        q, k, v, beta, g, key_heads, value_heads, p.key_start, p.value_start,
        1, p.precision, false, thread % threads);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_step_batch(p: GdnStepBatch) {
    let (heads, count) = (p.heads, p.count);
    let width = heads as usize * 128;
    let thread = index();
    let row = thread / width;
    if row >= count as usize { return; }
    let offset = row * width;
    if p.sessions == 0 {
        let state = &*(p.rows as *const DecodeRow).add(row);
        let destination = if state.output_state == 0 { state.state } else { state.output_state };
        step((p.q as *const f32).add(offset), (p.k as *const f32).add(offset), (p.v as *const f32).add(offset),
            (p.beta as *const f32).add(row * heads as usize), (p.g as *const f32).add(row * heads as usize),
            state.state as *const f32, destination as *mut f32, (p.out as *mut f32).add(offset), heads, thread % width);
    } else {
        let s = &*(p.sessions as *const GdnSessionRow).add(row);
        if s.padding != 0 { return; }
        step(s.q as _, s.k as _, s.v as _, s.beta as _, s.g as _, s.state as _, s.state as _, s.core as _, heads, thread % width);
    }
}

/// Step-batch plus gate in one launch, block128 (one head per CTA).
/// Every thread computes the same core value step_batch would store, stages
/// it in shared, and warp 0 then replays gate()'s warp reduction verbatim:
/// lanes read features lane+i*32, accumulate squares in i order, take the
/// same butterfly sum, scale, and apply the same weight/sigmoid/rounding.
/// No thread returns before the barrier; the grid is exactly heads*count CTAs.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_step_gate_batch(p: GdnStepGateBatch) {
    let (heads, count, precision) = (p.heads, p.count, p.precision);
    let width = heads as usize * 128;
    let thread = index();
    let row = thread / width;
    let local = thread % width;
    let offset = row * width;
    let valid = row < count as usize;
    let value = if valid {
        let state = &*(p.rows as *const DecodeRow).add(row);
        let destination = if state.output_state == 0 { state.state } else { state.output_state };
        step_value((p.q as *const f32).add(offset), (p.k as *const f32).add(offset), (p.v as *const f32).add(offset),
            (p.beta as *const f32).add(row * heads as usize), (p.g as *const f32).add(row * heads as usize),
            state.state as *const f32, destination as *mut f32, heads, local)
    } else { 0.0 };
    let sh = crate::ptx::shared::<512>();
    crate::ptx::store(sh + ((local % 128) * 4) as u32, value);
    crate::ptx::sync();
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    if !valid || tid >= 32 { return; }
    let lane = tid;
    let base = offset + (local / 128) * 128;
    let (z, w, y) = (p.z as *const f32, p.norm as *const f32, p.out as *mut f32);
    let mut squares = 0.0;
    let mut values = [0.0; 4];
    for i in 0..4 {
        let value = crate::ptx::load(sh + ((lane + i * 32) * 4) as u32);
        values[i] = if precision == 0 { value } else { rounded(value) };
        squares += values[i] * values[i];
    }
    let scale = 1.0 / sqrt(sum(squares) / 128.0 + p.epsilon);
    for i in 0..4 {
        let feature = lane + i * 32;
        let weight = if precision == 0 || precision & 8 != 0 { *w.add(feature) } else { bf(*w.cast::<u16>().add(feature)) };
        let value = if precision == 0 { values[i] * scale * weight }
            else { rounded(rounded(values[i] * scale) * weight) };
        let gate = *z.add(base + feature);
        let value = value * sigmoid(gate) * if precision == 0 { 1.0 } else { gate };
        *y.add(base + feature) = if precision == 0 { value } else { rounded(value) };
    }
}

/// Head-major recurrence -> token-major features, direct RMS weight and
/// sigmoid gate. The complete shared norm vector has 128 FP32 coordinates.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_gate(params: GdnGate) { gate_token(params); }
#[inline(always)]
pub(crate) unsafe fn gate_token(params: GdnGate) {
    if params.sessions == 0 { gate(params, index()); }
    else {
        let s = &*(params.sessions as *const GdnSessionRow).add(core::arch::nvptx::_block_idx_y() as usize);
        let offset = s.row as usize * params.heads as usize * 128;
        if s.padding != 0 {
            let thread = index();
            // The launch uses one warp per head, four features per lane.
            if thread < params.heads as usize * 32 {
                for i in 0..4 {
                    *params.output.add(offset + thread / 32 * 128 + thread % 32 + i * 32) = 0.0;
                }
            }
            return;
        }
        gate(GdnGate { x: s.core as _, norm: s.norm as _, z: params.z.add(offset),
            output: params.output.add(offset), ..params }, index());
    }
}
#[inline(always)]
unsafe fn gate(params: GdnGate, thread: usize) {
    let GdnGate { x, z, norm: w, output: y, heads, tokens, precision, packed, epsilon, .. } = params;
    let row = thread / 32;
    let lane = thread % 32;
    if row >= heads as usize * tokens as usize { return; }
    let head = row / tokens as usize;
    let token = row % tokens as usize;
    let mut squares = 0.0;
    let mut values = [0.0; 4];
    for i in 0..4 {
        let value = *x.add(row * 128 + lane + i * 32);
        values[i] = if precision == 0 { value } else { rounded(value) };
        squares += values[i] * values[i];
    }
    let scale = 1.0 / sqrt(sum(squares) / 128.0 + epsilon);
    for i in 0..4 {
        let feature = lane + i * 32;
        let column = head * 128 + feature;
        let output = token * heads as usize * 128 + column;
        let weight = if precision == 0 || precision & 8 != 0 { *w.add(feature) } else { bf(*w.cast::<u16>().add(feature)) };
        let value = if precision == 0 { values[i] * scale * weight }
            else { rounded(rounded(values[i] * scale) * weight) };
        let gate = *z.add(output);
        let value = value * sigmoid(gate) * if precision == 0 { 1.0 } else { gate };
        if packed == 0 {
            *y.add(output) = if precision == 0 { value } else { rounded(value) };
        } else if packed > 1 {
            crate::gemm::store_split_input(y.cast(), token, column, heads as usize * 128, packed as usize, value);
        } else {
            crate::gemm::store_bf16_input_element(y.cast(), token, column, heads as usize * 128, rounded(value));
        }
    }
}

/// gate() for one token at precision 0, unpacked (the decode slot launch):
/// the same words in the same order (lane-wise squares, warp sum, scale,
/// weight, sigmoid gate, the same products), with every load issued before
/// the first store so the four features no longer wait on each other's
/// store-to-load ordering.
#[inline(always)]
pub(crate) unsafe fn gate_plain(params: GdnGate, thread: usize) {
    let GdnGate { x, z, norm: w, output: y, heads, epsilon, .. } = params;
    let row = thread / 32;
    let lane = thread % 32;
    if row >= heads as usize { return; }
    let (mut values, mut weights, mut gates) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    for i in 0..4 {
        let feature = lane + i * 32;
        values[i] = *x.add(row * 128 + feature);
        weights[i] = *w.add(feature);
        gates[i] = *z.add(row * 128 + feature);
    }
    let mut squares = 0.0;
    for i in 0..4 { squares += values[i] * values[i]; }
    let scale = 1.0 / sqrt(sum(squares) / 128.0 + epsilon);
    for i in 0..4 {
        let value = values[i] * scale * weights[i];
        let value = value * sigmoid(gates[i]) * 1.0;
        *y.add(row * 128 + lane + i * 32) = value;
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_gdn_gate_prompts(params: GdnGate) {
    let row = &*(params.x as *const PromptRow).add(core::arch::nvptx::_block_idx_y() as usize);
    let offset = row.offset as usize * params.heads as usize * 128;
    gate(GdnGate { x: row.core, z: params.z.add(offset), output: params.output.add(offset),
        tokens: row.tokens, ..params }, index());
}
