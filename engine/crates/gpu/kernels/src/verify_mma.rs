//! Canonical K ranges run in parallel warps; their balanced merge stays on chip.
use crate::{canonical::{Acc, ZERO, madd, add, range, weight_tile}, ptx};
use core::arch::nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x};

#[inline(always)]
unsafe fn stage<const COLS: usize, const CK: usize, const MROWS: usize>(sh: u32, w: *const u32, x: *const u32,
    mt: usize, nt: usize, kt: usize, c: usize, tid: usize, tokens: usize) {
    let mut i = tid;
    while i < 4 * MROWS * CK * 32 {
        let (part, row, z, lane) = (i / (MROWS*CK*32), i % (MROWS*CK*32) / (CK*32), i % (CK*32) / 32, i % 32);
        let dst = sh + (i * 16) as u32;
        if (mt + row) * 16 < tokens {
            ptx::copy16(dst, x.add(((mt + row) * kt + range(kt, part).0 + c + z) * 128 + lane * 4));
        } else {
            for j in 0..4 { ptx::store(dst + j * 4, 0.0); }
        }
        i += if COLS == 8 {512} else {256};
    }
    i = tid;
    while i < 4 * COLS * CK * 16 {
        let (part, col, z, lane) = (i / (COLS*CK*16), i % (COLS*CK*16) / (CK*16), i % (CK*16) / 16, i % 16);
        ptx::copy16_stream(sh + (4*MROWS*CK*512 + i*16) as u32,
            w.add(weight_tile(nt + col, range(kt, part).0 + c + z, kt) * 64 + lane * 4));
        i += if COLS == 8 {512} else {256};
    }
}

#[inline(always)]
unsafe fn load_acc(p: u32) -> Acc {
    let v = ptx::load4(p);
    Acc { a: f32::from_bits(v[0]), b: f32::from_bits(v[1]), c: f32::from_bits(v[2]), d: f32::from_bits(v[3]) }
}

/// Block256/grid(ceil(T/32),N/32), shared32768. K%128=0, N%32=0.
/// Each warp accumulates one fixed K range; no global partials or weight copy.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn verify_bf16_f32_mma(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32,
) {
    verify::<4, 2, 2>(w, x, y, n, k, t, ptx::shared::<32768>());
}

/// 32x64 tile with contiguous 64-column stages in each canonical K range.
/// Block512/grid(ceil(T/32),N/64), dynamic shared98304. K%256=0.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn verify_bf16_blocked_mma(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32,
) {
    verify::<8, 4, 2>(w, x, y, n, k, t, 0);
}

#[inline(always)]
unsafe fn verify<const COLS: usize, const CK: usize, const MROWS: usize>(
    w: *const u32, x: *const u32, y: *mut f32, n: u32, k: u32, t: u32, sh: u32,
) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let part = warp % 4;
    let row_tile = warp / 4 % MROWS;
    let column = warp / (4*MROWS) * 4;
    let (mt, nt) = (_block_idx_x() as usize * MROWS, _block_idx_y() as usize * COLS);
    let kt = k as usize / 16;
    let xbytes = 4*MROWS*CK*512;
    let wbytes = 4*COLS*CK*256;
    let stride = xbytes + wbytes;
    let mut sums = [ZERO; 4];
    let (mut c, mut buffer) = (0, 0);
    stage::<COLS, CK, MROWS>(sh, w, x, mt, nt, kt, 0, tid, t as usize);
    ptx::wait(); ptx::sync();
    while c < kt / 4 {
        let current = sh + (buffer * stride) as u32;
        let next = sh + ((buffer ^ 1) * stride) as u32;
        if c + CK < kt / 4 { stage::<COLS, CK, MROWS>(next, w, x, mt, nt, kt, c + CK, tid, t as usize); }
        for z in 0..CK {
            let a = ptx::load4(current + (part*MROWS*CK*512 + row_tile*CK*512 + z*512 + lane*16) as u32);
            macro_rules! column { ($j:expr) => {{
                let b = ptx::load2(current + (xbytes + part*COLS*CK*256 + (column+$j)*CK*256 + z*256 + lane*8) as u32);
                sums[$j] = madd(a, b, sums[$j]);
            }}; }
            column!(0); column!(1); column!(2); column!(3);
        }
        ptx::wait(); ptx::sync();
        c += CK;
        buffer ^= 1;
    }
    for j in 0..4 {
        let p = sh + ((warp * 4 + j) * 512 + lane * 16) as u32;
        let s = sums[j];
        ptx::store(p, s.a); ptx::store(p + 4, s.b); ptx::store(p + 8, s.c); ptx::store(p + 12, s.d);
    }
    ptx::sync();
    if part == 0 {
        for j in 0..4 {
            let p = sh + ((warp * 4 + j) * 512 + lane * 16) as u32;
            let s = add(add(load_acc(p), load_acc(p + 2048)), add(load_acc(p + 4096), load_acc(p + 6144)));
            let row = (mt + row_tile) * 16 + lane / 4;
            let col = (nt + column + j) * 8 + lane % 4 * 2;
            if row < t as usize { *y.add(row * n as usize + col) = s.a; *y.add(row * n as usize + col + 1) = s.b; }
            if row + 8 < t as usize { *y.add((row + 8) * n as usize + col) = s.c; *y.add((row + 8) * n as usize + col + 1) = s.d; }
        }
    }
}
