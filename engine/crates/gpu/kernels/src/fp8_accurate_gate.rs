//! Rank-sharded W8A8 gate/up; explicit legacy BF16 operands remain available.
use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x}};
use crate::ptx;

#[inline(always)]
pub(crate) unsafe fn promote_pair(raw: u16, scale: f32) -> u32 {
    let high: u32;
    asm!("{{ .reg .b32 h; .reg .b16 a, b; .reg .f32 x, y;
        cvt.rn.f16x2.e4m3x2 h, {raw}; mov.b32 {{a, b}}, h;
        cvt.f32.f16 x, a; cvt.f32.f16 y, b;
        mul.f32 x, x, {scale}; mul.f32 y, y, {scale};
        cvt.rn.bf16x2.f32 {high}, y, x;
    }}", raw=in(reg16)raw, scale=in(reg32)scale, high=out(reg32)high);
    high
}

#[inline(always)]
pub(crate) unsafe fn mma(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}},
        {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
        d0=out(reg32)d0,d1=out(reg32)d1,d2=out(reg32)d2,d3=out(reg32)d3,
        a0=in(reg32)a[0],a1=in(reg32)a[1],a2=in(reg32)a[2],a3=in(reg32)a[3],
        b0=in(reg32)b[0],b1=in(reg32)b[1],
        c0=in(reg32)c[0],c1=in(reg32)c[1],c2=in(reg32)c[2],c3=in(reg32)c[3]);
    [d0, d1, d2, d3]
}

macro_rules! twice {
    ($i:ident, $body:block) => {{ let $i=0; $body } { let $i=1; $body }};
}
macro_rules! four {
    ($i:ident, $body:block) => {
        { let $i=0; $body } { let $i=1; $body }
        { let $i=2; $body } { let $i=3; $body }
    };
}

#[inline(always)]
unsafe fn stage(sh: u32, w: *const u8, ws: *const u16, xh: *const u32, up: *const u8, ups: *const u16,
    mb: usize, nb: usize, kb: usize, m: usize, k: usize, tid: usize) {
        let row = tid / 32;
        let lane = tid % 32;
        let dst = sh + (tid * 16) as u32;
        if mb + row * 16 < m {
            let offset = (((mb / 16 + row) * (k / 16) + kb / 16) * 32 + lane) * 4;
            ptx::copy16(dst, xh.add(offset));
        } else {
            four!(i, { ptx::store(dst + i * 4, 0.0); });
        }
        let weight_row = nb + tid / 4 % 32;
        let (w, ws) = if tid < 128 { (w, ws) } else { (up, ups) };
        let raw = crate::fp8_bf16::load_weight4(w.add(weight_row * k + kb + tid % 4 * 4));
        let scale = f32::from_bits((*ws.add(weight_row / 128 * (k / 128) + kb / 128) as u32) << 16);
        let h0 = promote_pair(raw as u16, scale);
        let h1 = promote_pair((raw >> 16) as u16, scale);
        let p = sh + 8192 + (tid / 4 * 48 + tid % 4 * 8) as u32;
        asm!("st.shared.v2.b32 [{p}], {{{h0},{h1}}};",
            p=in(reg32)p,h0=in(reg32)h0,h1=in(reg32)h1);
}

#[inline(always)]
unsafe fn fragment(p: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    asm!("ld.shared.b32 {a}, [{p}]; ld.shared.b32 {b}, [{p}+16];",
        a=out(reg32)a,b=out(reg32)b,p=in(reg32)p);
    [a, b]
}

/// M128, 32 gate + 32 up columns, BF16 tensor cores and fused SiLU.
/// # Safety
/// W points to 512 seven-u64 resident expert descriptors. Packed input uses
/// width K; middle rows use the expert's local width in a 640-wide reservation.
/// Legacy BF16 uses block256. W8A8 uses block128, grid(20*planes,tasks,1),
/// and the canonical K128 helper with row-major BF16-rounded FP32 output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_gate_accurate(p: crate::fp8_abi::GroupedParams) {
    let shared = ptx::shared::<28672>();
    if p.activation_quantization != 0 {
        crate::fp8::grouped_project::<false>(p, false, shared);
        return;
    }
    let (w, x, y) = (p.weights as *const u8, p.input as *const u32, p.output as *mut f32);
    let (counts, starts) = (p.route.counts as *const u32, p.route.starts as *const u32);
    let task_count = p.route.tasks as *const u32;
    let tasks = task_count.add(1);
    let k = p.cols;

    let task = _block_idx_y() as usize;
    if task >= *task_count as usize { return; }
    let expert = *tasks.add(task * 2) as usize;
    let mb = *tasks.add(task * 2 + 1) as usize;
    let m = *counts.add(expert) as usize;
    if mb >= m { return; }
    let start = *starts.add(expert) as usize;
    let k = k as usize;
    let table = w.cast::<u64>().add(expert * 7);
    let width = *table.add(6) as usize;
    if _block_idx_x() as usize * 32 >= width { return; }
    let w = *table as *const u8;
    let ws = *table.add(1) as *const u16;
    let up = *table.add(2) as *const u8;
    let ups = *table.add(3) as *const u16;
    let xh = x.add(start * k / 2);
    let y = y.cast::<u32>().add(start * 640 / 2);
    let nb = _block_idx_x() as usize * 32;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (mr, nr) = (warp / 2 * 32, warp % 2 * 32);
    let mut sums = [[[0.0f32; 4]; 4]; 2];
    stage(shared, w, ws, xh, up, ups, mb, nb, 0, m, k, tid);
    ptx::wait(); ptx::sync();
    let mut kb = 0;
    while kb < k {
        let current = shared + ((kb / 16 % 2) * 14336) as u32;
        if kb + 16 < k {
            stage(shared + (((kb / 16 + 1) % 2) * 14336) as u32,
                w, ws, xh, up, ups, mb, nb, kb + 16, m, k, tid);
        }
            let col = lane % 4 * 2;
            let ap = current + ((mr / 16 * 32 + lane) * 16) as u32;
            let ah = [ptx::load4(ap), ptx::load4(ap + 512)];
            four!(j, {
                let row = nr + j * 8 + lane / 4;
                let bp = current + 8192 + (row * 48 + col * 2) as u32;
                let bh = fragment(bp);
                twice!(i, {
                    sums[i][j] = mma(ah[i], bh, sums[i][j]);
                });
            });
        ptx::wait(); ptx::sync();
        kb += 16;
    }
    if warp % 2 == 0 {
        twice!(i, {
            let row = mr + i * 16 + lane / 4;
            four!(j, {
            let col = j * 8 + lane % 4 * 2;
            let p = shared + ((row * 32 + col) * 4) as u32;
            let v = sums[i][j];
            ptx::store(p, v[0]); ptx::store(p + 4, v[1]);
            ptx::store(p + 1024, v[2]); ptx::store(p + 1028, v[3]);
            });
        });
    }
    ptx::sync();
    if warp % 2 == 1 {
        twice!(i, {
            let row = mb + mr + i * 16 + lane / 4;
            four!(j, {
            let col = nb + j * 8 + lane % 4 * 2;
            let p = shared + (((row - mb) * 32 + col - nb) * 4) as u32;
            let v = sums[i][j];
            if row < (m + 15) / 16 * 16 {
                crate::fp8_bf16::store_activation(y, width, row, col, ptx::load(p), ptx::load(p + 4), v[0], v[1]);
            }
            if row + 8 < (m + 15) / 16 * 16 {
                crate::fp8_bf16::store_activation(y, width, row + 8, col,
                    ptx::load(p + 1024), ptx::load(p + 1028), v[2], v[3]);
            }
            });
        });
    }
}
