//! Small instructions missing from Rust's NVPTX address-space interface.
use core::arch::asm;
#[inline(always)]
pub unsafe fn shared<const BYTES: usize>() -> u32 {
    let p: u32;
    asm!("{{ .shared .align 16 .b8 scratch[{size}]; mov.u32 {p}, scratch; }}", size = const BYTES, p = out(reg32) p);
    p
}
#[inline(always)]
pub unsafe fn load(p: u32) -> f32 {
    let v: f32;
    asm!("ld.shared.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p);
    v
}
#[inline(always)]
pub unsafe fn load_u16(p: u32) -> u32 {
    let v: u32;
    asm!("ld.shared.u16 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p);
    v
}
#[inline(always)]
pub unsafe fn load_u8(p: u32) -> u32 {
    let v: u32;
    asm!("ld.shared.u8 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p);
    v
}
#[inline(always)]
pub unsafe fn store(p: u32, v: f32) {
    asm!("st.shared.f32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v);
}
#[inline(always)]
pub unsafe fn sync() {
    asm!("bar.sync 0;");
}
#[inline(always)]
pub unsafe fn exp(x: f32) -> f32 {
    let y: f32;
    asm!("ex2.approx.ftz.f32 {y}, {x};", y = out(reg32) y, x = in(reg32) (x * core::f32::consts::LOG2_E));
    y
}
#[inline(always)]
pub unsafe fn copy16(dst: u32, src: *const u32) {
    asm!("cp.async.cg.shared.global [{d}], [{s}], 16;", d = in(reg32) dst, s = in(reg64) src);
}
#[inline(always)]
pub unsafe fn copy8(dst: u32, src: *const u32) {
    asm!("cp.async.ca.shared.global [{d}], [{s}], 8;", d = in(reg32) dst, s = in(reg64) src);
}
#[inline(always)]
pub unsafe fn copy4(dst: u32, src: *const u32) {
    asm!("cp.async.ca.shared.global [{d}], [{s}], 4;", d = in(reg32) dst, s = in(reg64) src);
}
#[inline(always)]
pub unsafe fn copy16_stream(dst: u32, src: *const u32) {
    asm!("cp.async.cg.shared.global.L2::256B [{d}], [{s}], 16;", d = in(reg32) dst, s = in(reg64) src);
}
#[inline(always)]
pub unsafe fn copy16_cached(dst: u32, src: *const u32) {
    asm!("cp.async.ca.shared.global [{d}], [{s}], 16;", d = in(reg32) dst, s = in(reg64) src);
}
#[inline(always)]
pub unsafe fn wait() {
    asm!("cp.async.wait_all;");
}
#[inline(always)]
pub unsafe fn commit() {
    asm!("cp.async.commit_group;");
}
#[inline(always)]
pub unsafe fn wait_one() {
    asm!("cp.async.wait_group 1;");
}
#[inline(always)]
pub unsafe fn load4(p: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("ld.shared.v4.b32 {{{a}, {b}, {c}, {d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg32)p);
    [a, b, c, d]
}
#[inline(always)]
pub unsafe fn load2(p: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    asm!("ld.shared.v2.b32 {{{a}, {b}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,p=in(reg32)p);
    [a, b]
}
#[inline(always)]
pub unsafe fn warp_sync() {
    asm!("bar.warp.sync 0xffffffff;");
}
/// Bit n is set when lane n's predicate holds; all 32 lanes must participate.
#[inline(always)]
pub unsafe fn ballot(predicate: bool) -> u32 {
    let bits: u32;
    asm!("{{ .reg .pred p; setp.ne.u32 p, {x}, 0; vote.sync.ballot.b32 {b}, p, 0xffffffff; }}",
        b = out(reg32) bits, x = in(reg32) predicate as u32);
    bits
}
