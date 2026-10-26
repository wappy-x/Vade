//! VADE_W2_TP=1 in-graph mailbox: rows cross between two Sparks inside the CUDA stream.
//! `tp_mailbox_put` copies one rank's rows (or row halves) into a mapped host slot and then
//! publishes the slot's sequence word; a host proxy thread forwards the slot over RDMA and
//! publishes it in the peer's receive slot, where `tp_mailbox_take` waits for the word and
//! copies the rows into place. Only raw 32-bit words move: no arithmetic, no reduction.
use core::arch::asm;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};

#[inline(always)]
unsafe fn load_acquire(word: *const u32) -> u32 {
    let v: u32;
    asm!("ld.acquire.sys.global.u32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) word);
    v
}
#[inline(always)]
unsafe fn store_release(word: *mut u32, v: u32) {
    asm!("st.release.sys.global.u32 [{p}], {v};", p = in(reg64) word, v = in(reg32) v);
}
#[inline(always)]
unsafe fn now_ns() -> u64 {
    let t: u64;
    asm!("mov.u64 {t}, %globaltimer;", t = out(reg64) t);
    t
}

/// Copy four words (one 16-byte load and store); both addresses 16-byte aligned. Used inside
/// the kernels' bodies only.
macro_rules! copy4 {
    ($dst:expr, $src:expr) => {{
        let (dst, src): (*mut u32, *const u32) = ($dst, $src);
        let (a, b, c, d): (u32, u32, u32, u32);
        asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) src);
        asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};", p = in(reg64) dst, a = in(reg32) a, b = in(reg32) b, c = in(reg32) c, d = in(reg32) d);
    }};
}
/// Copies may move four words at a time: asked for, and every row offset and both ends 16-byte aligned.
#[inline(always)]
fn quads(asked: bool, half: usize, width: usize, offset: usize, a: usize, b: usize) -> bool {
    asked && (half | width | offset) % 4 == 0 && (a | b) % 16 == 0
}

/// One CTA: words `own..own + half` of each of `rows` rows of `width` words at `src` go to
/// `dst` row-major (`half` words per row), four at a time when bit 32 of `seq` asks and the
/// shape allows; then, after the CTA barrier and a system fence, `len` (bytes) and the low
/// word of `seq` are published (sequence word last, release).
/// # Safety
/// `src` covers `rows * width` words, `dst` is mapped host memory covering `rows * half`
/// words, `len_word`/`seq_word` are mapped words; `own + half <= width`. Launch one CTA.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn tp_mailbox_put(src: *const u32, rows: u64, width: u64, half: u64, own: u64,
    dst: *mut u32, len_word: *mut u32, seq_word: *mut u32, seq: u64) {
    if _block_idx_x() != 0 { return; }
    let (rows, width, half, own) = (rows as usize, width as usize, half as usize, own as usize);
    let mut i = _thread_idx_x() as usize;
    if quads(seq >> 32 & 1 != 0, half, width, own, src as usize, dst as usize) {
        let q = half / 4;
        while i < rows * q {
            copy4!(dst.add(4 * i), src.add(i / q * width + own + 4 * (i % q)));
            i += _block_dim_x() as usize;
        }
    } else {
        while i < rows * half {
            *dst.add(i) = *src.add(i / half * width + own + i % half);
            i += _block_dim_x() as usize;
        }
    }
    crate::ptx::sync();
    if _thread_idx_x() == 0 {
        asm!("fence.sc.sys;");
        *len_word = (rows * half * 4) as u32;
        store_release(seq_word, seq as u32);
    }
}

/// Every CTA: thread 0 waits (acquire, with nanosleep backoff) until `seq_word` holds the low
/// word of `seq`, at most the low 47 bits of `limit_ns` (ns; its high 16 bits cap the backoff,
/// 0 = 2048 ns; bit 47 asks for four-word copies), or until `error` is set; on timeout it sets `error` and
/// the CTA copies nothing. Then words
/// of `src` (row-major, `half` per row) go to words `peer..peer + half` of each `width`-word
/// row of `dst`, split over the grid.
/// # Safety
/// `src` is mapped host memory covering `rows * half` words, `dst` covers `rows * width`
/// words, `seq_word`/`error` are mapped words; `peer + half <= width`. Any 1D grid.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn tp_mailbox_take(dst: *mut u32, rows: u64, width: u64, half: u64, peer: u64,
    src: *const u32, seq_word: *const u32, seq: u64, error: *mut u32, limit_ns: u64) {
    let sh = crate::ptx::shared::<4>();
    if _thread_idx_x() == 0 {
        let start = now_ns();
        let (limit, cap) = (limit_ns & ((1 << 47) - 1), (limit_ns >> 48) as u32);
        let cap = if cap == 0 { 2048 } else { cap };
        let (mut ok, mut pause) = (1u32, 32u32.min(cap));
        while load_acquire(seq_word) != seq as u32 {
            if load_acquire(error) != 0 || now_ns().wrapping_sub(start) > limit {
                store_release(error, 1);
                ok = 0;
                break;
            }
            // Back off between polls (doubling to the cap, 2 us unless the host set one): the
            // wait leaves the SM's issue slots and the memory system to the rest of the GPU.
            asm!("nanosleep.u32 {t};", t = in(reg32) pause);
            pause = (pause * 2).min(cap);
        }
        crate::ptx::store(sh, f32::from_bits(ok));
    }
    crate::ptx::sync();
    if crate::ptx::load(sh).to_bits() == 0 { return; }
    let (rows, width, half, peer) = (rows as usize, width as usize, half as usize, peer as usize);
    let step = (_grid_dim_x() * _block_dim_x()) as usize;
    let mut i = (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize;
    if quads(limit_ns >> 47 & 1 != 0, half, width, peer, src as usize, dst as usize) {
        let q = half / 4;
        while i < rows * q {
            copy4!(dst.add(i / q * width + peer + 4 * (i % q)), src.add(4 * i));
            i += step;
        }
    } else {
        while i < rows * half {
            *dst.add(i / half * width + peer + i % half) = *src.add(i);
            i += step;
        }
    }
}
