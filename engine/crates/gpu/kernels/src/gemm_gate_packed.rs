//! Larger gate/up tile retaining the native packed shared-memory layout.
use super::*;
use crate::ptx;
use core::arch::asm;

const STAGE_BYTES: usize = 49152;

macro_rules! tile_unroll {
    (2, $i:ident, $body:block) => {
        { const $i: usize = 0; $body } { const $i: usize = 1; $body }
    };
    (4, $i:ident, $body:block) => {
        tile_unroll!(2, $i, $body);
        { const $i: usize = 2; $body } { const $i: usize = 3; $body }
    };
}

#[inline(always)]
#[allow(named_asm_labels)]
unsafe fn stage_x(
    sh: u32, x: *const u32, mb: usize, kt: usize, c: usize, tid: usize, t: usize,
) {
    // Form copy addresses only while issuing them, not across the MMA loop.
    asm!(r#"{{
        .reg .u32 r<8>;
        .reg .u64 p<3>;
        .reg .pred valid;
        shr.u32 r0, {tid}, 6;
        add.u32 r0, r0, {mb};
        mul.wide.u32 p0, r0, {kt};
        shr.u32 r1, {tid}, 5;
        and.b32 r1, r1, 1;
        add.u32 r1, r1, {c};
        cvt.u64.u32 p1, r1;
        add.u64 p0, p0, p1;
        shl.b64 p0, p0, 9;
        and.b32 r1, {tid}, 31;
        mad.wide.u32 p0, r1, 16, p0;
        add.u64 p0, p0, {x};
        mad.lo.u32 r2, {tid}, 16, {sh};
        shl.b32 r0, r0, 4;
        setp.lt.u32 valid, r0, {t};
        selp.b32 r4, 16, 0, valid;
        cp.async.cg.shared.global [r2], [p0], 16, r4;
    }}"#, sh=in(reg32) sh, x=in(reg64) x, mb=in(reg32) mb as u32,
        kt=in(reg32) kt as u32, c=in(reg32) c as u32,
        tid=in(reg32) tid as u32, t=in(reg32) t as u32);
}

#[inline(always)]
unsafe fn stage_w(sh: u32, w: *const u32, u: *const u32, nb: usize, kt: usize, c: usize, tid: usize) {
    let col = nb + tid / 32;
    let z = tid / 16 % 2;
    let at = weight_tile(col, c + z, kt) * 64 + tid % 16 * 4;
    let dst = sh + 16384 + (tid * 16) as u32;
    ptx::copy16(dst, w.add(at));
    ptx::copy16(dst + 8192, u.add(at));
}

#[inline(always)]
unsafe fn load_b<const OFFSET: usize>(base: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    asm!("{{ .reg .u32 p; add.u32 p, {base}, {offset}; ld.shared.v2.b32 {{{a}, {b}}}, [p]; }}",
        base=in(reg32) base, offset=const OFFSET, a=out(reg32) a, b=out(reg32) b);
    [a, b]
}

#[inline(always)]
unsafe fn load_a<const OFFSET: usize>(base: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("{{ .reg .u32 p; add.u32 p, {base}, {offset}; ld.shared.v4.b32 {{{a}, {b}, {c}, {d}}}, [p]; }}",
        base=in(reg32) base, offset=const OFFSET,
        a=out(reg32) a, b=out(reg32) b, c=out(reg32) c, d=out(reg32) d);
    [a, b, c, d]
}

#[inline(always)]
unsafe fn pair(ga: f32, gb: f32, ua: f32, ub: f32, row: usize, t: usize) -> u32 {
    if row >= t { return 0; }
    let a = ga / (1.0 + ptx::exp(-ga)) * ua;
    let b = gb / (1.0 + ptx::exp(-gb)) * ub;
    bf16_pair(a, b)
}

/// Grouped 128x128 gate/up, two asynchronous stages, BF16 input and output.
/// Block512/grid(ceil(padded rows/128),N/128), dynamic shared98304, K%64 == 0.
#[no_mangle]
#[allow(non_upper_case_globals)]
pub unsafe extern "ptx-kernel" fn moe_prefill_gate_up_wide(
    tables: *const u64, counts: *const u32, offsets: *const u32,
    packed: *const u32, activated: *mut u32, n: u32, k: u32, experts: u32,
) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (mtile, ntile) = grouped_tiles::<128>(n);
    let (expert, row0) = expert_tile::<128>(mtile, offsets, experts as usize);
    if expert == experts as usize { return; }
    let t = *counts.add(expert);
    if row0 >= t as usize { return; }
    let offset = *offsets.add(expert) as usize;
    let w = *tables.add(expert * 3) as *const u32;
    let u = *tables.add(expert * 3 + 1) as *const u32;
    let x = packed.add(offset * k as usize / 2);
    let y = activated.add(offset * n as usize / 2);
    let (mb, nb) = (row0 / 16, ntile * 16);
    let column = warp % 8;
    let (mw, nw) = (warp / 8 * 4, column * 2);
    let kt = k as usize / 16;
    // This entry owns no static shared allocation; dynamic shared starts at zero.
    let sh = 0u32;
    let mut gates = [[ZERO; 2]; 4];
    let mut ups = [[ZERO; 2]; 4];
    stage_x(sh, x, mb, kt, 0, tid, t as usize);
    stage_x(sh + 8192, x, mb, kt, 2, tid, t as usize);
    stage_w(sh, w, u, nb, kt, 0, tid);
    stage_w(sh + 16384, w, u, nb, kt, 2, tid);
    ptx::commit();
    ptx::wait();
    ptx::sync();
    let mut c = 0;
    let mut current = 0usize;
    let mut next = STAGE_BYTES;
    while c < kt {
        let ahead = c + 4;
        if ahead < kt {
            stage_x(sh + next as u32, x, mb, kt, ahead, tid, t as usize);
            stage_x(sh + next as u32 + 8192, x, mb, kt, ahead + 2, tid, t as usize);
        }
        let mut z = 0;
        while z < 4 {
            {
                let base_b = (current + 16384 + z / 2 * 16384 + nw * 512 + z % 2 * 256 + lane * 8) as u32;
                let base_a = (current + z / 2 * 8192 + mw * 1024 + z % 2 * 512 + lane * 16) as u32;
                let a = [load_a::<0>(base_a), load_a::<1024>(base_a),
                    load_a::<2048>(base_a), load_a::<3072>(base_a)];
                tile_unroll!(2, j, {
                    let bg = load_b::<{ j * 512 }>(base_b);
                    let bu = load_b::<{ (j + 16) * 512 }>(base_b);
                    tile_unroll!(4, i, {
                        gates[i][j] = madd(a[i], bg, gates[i][j]);
                        ups[i][j] = madd(a[i], bu, ups[i][j]);
                    });
                });
                // Issue gate/up copies among the MMA groups, then commit the
                // complete future stage together with its earlier X copies.
                if z < 2 && ahead < kt {
                    stage_w(sh + next as u32 + z as u32 * 16384, w, u, nb, kt, ahead + z * 2, tid);
                    if z == 1 { ptx::commit(); }
                }
            }
            z += 1;
        }
        ptx::wait();
        ptx::sync();
        c += 4;
        current += STAGE_BYTES;
        if current == 2 * STAGE_BYTES { current = 0; }
        next += STAGE_BYTES;
        if next == 2 * STAGE_BYTES { next = 0; }
    }
    tile_unroll!(4, i, {
        let row_tile = mb + mw + i;
        if row_tile * 16 < t as usize {
            let row = row_tile * 16 + lane / 4;
            {
                let (g0, g1) = (gates[i][0], gates[i][1]);
                let (u0, u1) = (ups[i][0], ups[i][1]);
                let p0 = pair(g0.a, g0.b, u0.a, u0.b, row, t as usize);
                let p1 = pair(g0.c, g0.d, u0.c, u0.d, row + 8, t as usize);
                let p2 = pair(g1.a, g1.b, u1.a, u1.b, row, t as usize);
                let p3 = pair(g1.c, g1.d, u1.c, u1.d, row + 8, t as usize);
                let tile = row_tile * (n as usize / 16) + nb / 2 + column;
                *y.cast::<AFrag>().add(tile * 32 + lane) = AFrag([p0, p1, p2, p3]);
            }
        }
    });
}
