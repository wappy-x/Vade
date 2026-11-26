//! Minimal pure PTX wrappers for SM121a BF16 tensor-core MMA and shared memory transfers,
//! private to `kernels-inkling` so `kernels/` stays untouched.
use core::arch::asm;

/// Convert two FP32 values to packed BF16x2 (round-to-nearest-even): `lo` in bits 0..16, `hi` in bits 16..32.
/// SAFETY: pure register instruction, touches no memory.
#[inline(always)]
pub(crate) unsafe fn cvt_bf16x2(lo: f32, hi: f32) -> u32 {
    let d: u32;
    asm!("cvt.rn.bf16x2.f32 {d}, {hi}, {lo};", d = out(reg32) d, hi = in(reg32) hi, lo = in(reg32) lo, options(pure, nomem, nostack));
    d
}

/// 16-byte global read-only non-coherent load (`ld.global.nc.v4.f32`).
/// SAFETY: `p` must be a valid 16-byte aligned global pointer to 4 readable `f32` words.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_v4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) p);
    [a, b, c, d]
}

/// 8-byte global read-only non-coherent load (`ld.global.nc.v2.u32`).
/// SAFETY: `p` must be a valid 8-byte aligned global pointer to 2 readable `u32` words.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_v2_u32(p: *const u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    asm!("ld.global.nc.v2.u32 {{{a}, {b}}}, [{p}];",
        a = out(reg32) a, b = out(reg32) b, p = in(reg64) p);
    [a, b]
}

/// 16-byte global read-only load with an L2 eviction priority (`ld.global.nc.L2::cache_hint.v4.u32`).
/// SAFETY: `p` must be a valid 16-byte aligned global pointer to 4 readable `u32` words that no thread writes.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_v4_u32_hint(p: *const u32, policy: u64) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // SAFETY: the caller's contract of `ld_global_nc_v4_u32`; the policy changes only the L2 eviction priority.
    asm!("ld.global.nc.L2::cache_hint.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}], {q};",
        a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) p, q = in(reg64) policy);
    [a, b, c, d]
}

/// 4-byte global read-only load with an L2 eviction priority (`ld.global.nc.L2::cache_hint.u32`).
/// SAFETY: `p` must be a valid 4-byte aligned global pointer to one readable `u32` that no thread writes.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_u32_hint(p: *const u32, policy: u64) -> u32 {
    let a: u32;
    asm!("ld.global.nc.L2::cache_hint.u32 {a}, [{p}], {q};", a = out(reg32) a, p = in(reg64) p, q = in(reg64) policy);
    a
}

/// 4-byte global store with an L2 eviction priority (`st.global.L2::cache_hint.f32`).
/// SAFETY: `p` must be a valid 4-byte aligned global pointer to one `f32` that this thread alone writes.
#[inline(always)]
pub(crate) unsafe fn st_global_f32_hint(p: *mut f32, v: f32, policy: u64) {
    asm!("st.global.L2::cache_hint.f32 [{p}], {v}, {q};", p = in(reg64) p, v = in(reg32) v, q = in(reg64) policy, options(nostack));
}

/// An L2 policy that marks every address it covers as the first to evict.
/// SAFETY: register instruction, touches no memory (`createpolicy.fractional`, sm_80 and later).
#[inline(always)]
pub(crate) unsafe fn l2_evict_first() -> u64 {
    let p: u64;
    asm!("createpolicy.fractional.L2::evict_first.b64 {p}, 1.0;", p = out(reg64) p, options(nomem, nostack));
    p
}

/// 16-byte global read-only non-coherent load (`ld.global.nc.v4.u32`).
/// SAFETY: `p` must be a valid 16-byte aligned global pointer to 4 readable `u32` words.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_v4_u32(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, p = in(reg64) p);
    [a, b, c, d]
}

/// 4-byte global read-only non-coherent load (`ld.global.nc.u32`).
/// SAFETY: `p` must be a valid 4-byte aligned global pointer to 1 readable `u32` word.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_u32(p: *const u32) -> u32 {
    let v: u32;
    asm!("ld.global.nc.u32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) p);
    v
}

/// 1-byte global read-only non-coherent load (`ld.global.nc.u8`).
/// SAFETY: `p` must be a valid global pointer to 1 readable `u8`.
#[inline(always)]
pub(crate) unsafe fn ld_global_nc_u8(p: *const u8) -> u32 {
    let v: u32;
    asm!("ld.global.nc.u8 {v}, [{p}];", v = out(reg32) v, p = in(reg64) p);
    v
}

/// 16-byte shared-memory store (`st.shared.v4.b32`).
/// SAFETY: `p` must be a 16-byte aligned shared address with 16 writable bytes inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn st_shared_v4(p: u32, v: [u32; 4]) {
    asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
        p = in(reg32) p, a = in(reg32) v[0], b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3]);
}

/// 16-byte shared-memory zero fill (`st.shared.v4.b32`).
/// SAFETY: `p` must be a 16-byte aligned shared address with 16 writable bytes inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn st_shared_zero16(p: u32) {
    asm!("st.shared.v4.b32 [{p}], {{0, 0, 0, 0}};", p = in(reg32) p);
}

/// Warp-synchronous 2-matrix 8x8 b16 shared load (`ldmatrix.sync.aligned.m8n8.x2.shared.b16`).
/// SAFETY: all 32 lanes of the warp must participate; lanes 0..16 must pass 16-byte aligned shared addresses
/// pointing to 16 readable bytes inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn ldmatrix_x2(p: u32) -> [u32; 2] {
    let (r0, r1): (u32, u32);
    asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{r0}, {r1}}}, [{p}];",
        r0 = out(reg32) r0, r1 = out(reg32) r1, p = in(reg32) p);
    [r0, r1]
}

/// Warp-synchronous 4-matrix 8x8 b16 shared load (`ldmatrix.sync.aligned.m8n8.x4.shared.b16`).
/// SAFETY: all 32 lanes of the warp must participate; each lane's `p` must be a 16-byte aligned shared address
/// pointing to 16 readable bytes inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn ldmatrix_x4(p: u32) -> [u32; 4] {
    let (r0, r1, r2, r3): (u32, u32, u32, u32);
    asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{r0}, {r1}, {r2}, {r3}}}, [{p}];",
        r0 = out(reg32) r0, r1 = out(reg32) r1, r2 = out(reg32) r2, r3 = out(reg32) r3, p = in(reg32) p);
    [r0, r1, r2, r3]
}

/// Warp-synchronous transposed 4-matrix 8x8 b16 shared load (`ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`).
/// SAFETY: all 32 lanes of the warp must participate; each lane's `p` must be a 16-byte aligned shared address
/// pointing to 16 readable bytes inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn ldmatrix_x4_trans(p: u32) -> [u32; 4] {
    let (r0, r1, r2, r3): (u32, u32, u32, u32);
    asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{r0}, {r1}, {r2}, {r3}}}, [{p}];",
        r0 = out(reg32) r0, r1 = out(reg32) r1, r2 = out(reg32) r2, r3 = out(reg32) r3, p = in(reg32) p);
    [r0, r1, r2, r3]
}

/// Warp-synchronous butterfly shuffle (`shfl.sync.bfly.b32`) on FP32 across all 32 lanes.
/// SAFETY: all 32 lanes of the warp must participate; touches no memory. Never `pure`: must not be sunk into divergent branches.
#[inline(always)]
pub(crate) unsafe fn shfl_xor_f32(v: f32, mask: u32) -> f32 {
    let d: f32;
    asm!("shfl.sync.bfly.b32 {d}, {v}, {m}, 0x1f, 0xffffffff;",
        d = out(reg32) d, v = in(reg32) v, m = in(reg32) mask, options(nomem, nostack));
    d
}

/// Warp-synchronous butterfly shuffle (`shfl.sync.bfly.b32`) on u32 across all 32 lanes.
/// SAFETY: all 32 lanes of the warp must participate; touches no memory. Never `pure`: must not be sunk into divergent branches.
#[inline(always)]
pub(crate) unsafe fn shfl_xor_u32(v: u32, mask: u32) -> u32 {
    let d: u32;
    asm!("shfl.sync.bfly.b32 {d}, {v}, {m}, 0x1f, 0xffffffff;",
        d = out(reg32) d, v = in(reg32) v, m = in(reg32) mask, options(nomem, nostack));
    d
}

/// Warp-synchronous match (`match.any.sync.b32`) across all 32 lanes: the mask of the lanes whose `v` equals this lane's.
/// SAFETY: all 32 lanes of the warp must participate; touches no memory. Never `pure`: must not be sunk into divergent branches.
#[inline(always)]
pub(crate) unsafe fn match_any_u32(v: u32) -> u32 {
    let d: u32;
    asm!("match.any.sync.b32 {d}, {v}, 0xffffffff;", d = out(reg32) d, v = in(reg32) v, options(nomem, nostack));
    d
}

/// Warp-synchronous SM121a BF16 tensor-core MMA (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`).
/// SAFETY: all 32 lanes of the warp must participate; touches no memory. Never `pure`: warp-collective instruction.
#[inline(always)]
pub(crate) unsafe fn mma_m16n8k16_bf16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
        d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
        a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
        b0 = in(reg32) b[0], b1 = in(reg32) b[1],
        c0 = in(reg32) c[0], c1 = in(reg32) c[1], c2 = in(reg32) c[2], c3 = in(reg32) c[3],
        options(nomem, nostack));
    [d0, d1, d2, d3]
}

/// Wait for all committed `cp.async` groups of this thread to complete (`cp.async.wait_group 0`).
/// SAFETY: synchronization instruction on this thread's async copy queue; touches no memory directly.
#[inline(always)]
pub(crate) unsafe fn cp_async_wait_zero() {
    asm!("cp.async.wait_group 0;");
}

/// Wait until at most the newest committed `cp.async` group of this thread is pending (`cp.async.wait_group 1`).
/// SAFETY: synchronization instruction on this thread's async copy queue; touches no memory directly.
#[inline(always)]
pub(crate) unsafe fn cp_async_wait_one() {
    asm!("cp.async.wait_group 1;");
}

/// Wait until at most the two newest committed `cp.async` groups of this thread are pending (`cp.async.wait_group 2`).
/// SAFETY: synchronization instruction on this thread's async copy queue; touches no memory directly.
#[inline(always)]
pub(crate) unsafe fn cp_async_wait_two() {
    asm!("cp.async.wait_group 2;");
}

/// Shared memory u16 store (`st.shared.u16`).
/// SAFETY: `p` must be a 2-byte aligned shared address inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn store_u16(p: u32, v: u16) {
    asm!("st.shared.u16 [{p}], {v};", p = in(reg32) p, v = in(reg16) v);
}

/// Shared memory u32 store (`st.shared.u32`).
/// SAFETY: `p` must be a 4-byte aligned shared address inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn store_u32(p: u32, v: u32) {
    asm!("st.shared.u32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v);
}

/// Shared memory u32 load (`ld.shared.u32`).
/// SAFETY: `p` must be a 4-byte aligned shared address inside the CTA's shared buffer.
#[inline(always)]
pub(crate) unsafe fn load_u32(p: u32) -> u32 {
    let v: u32;
    asm!("ld.shared.u32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p);
    v
}
