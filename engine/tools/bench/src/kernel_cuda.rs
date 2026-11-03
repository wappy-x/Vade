//! CUDA Driver API, loaded dynamically so the benchmark has no SDK or crate dependency.
use super::{Case, Options, Result, grade, median, sha256};
use std::{
    ffi::{CString, c_char, c_void},
    ptr::null_mut,
    time::Instant,
};
#[allow(dead_code)]
#[path = "../../../crates/gpu/src/fp4_abi.rs"]
mod fp4;
#[allow(dead_code, unsafe_op_in_unsafe_fn)]
#[path = "../../../crates/gpu/src/fp8_abi.rs"]
mod fp8;
#[allow(dead_code)]
#[path = "../../../crates/gpu/kernel_params.rs"]
mod params;
#[path = "kernel_qsa.rs"]
mod qsa;
#[path = "kernel_head.rs"] mod head;
pub(super) fn gather_ptx() -> String { qsa::gather_ptx() }
#[path = "kernel_gdn.rs"]
mod gdn;
#[path = "kernel_routed.rs"]
mod bw_routed;
use bw_routed::{flush_init, flush_on, icflush, launch_finish, launch_layer, launch_moe_route, launch_route, routed_fixture};
#[path = "kernel_prompt.rs"]
mod prompt;
#[path = "kernel_frontend.rs"]
mod frontend;
#[path = "kernel_fp4_prompt.rs"]
mod fp4_prompt;
#[path = "kernel_fp8.rs"] pub(super) mod fp8group;
#[path = "kernel_rows.rs"] pub(super) mod rows;
#[path = "kernel_mxgroup.rs"]
mod mxgroup;
#[path = "kernel_fp8moe.rs"] mod fp8moe;
pub(super) use fp8moe::run as fp8_moe;
use mxgroup::{all_mx, group_mx, mx_range_warps, mxgroup_fixture, mxgroup_weight, mxrows_a, mxrows_entries, mxrows_shared, mxrows_x};
/// Latency-kernel shared bytes per warp, as the engine launches them.
const RING: u32 = params::LATENCY_RING as u32;
type H = *mut c_void;
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> H;
    fn dlsym(lib: H, name: *const c_char) -> H;
}
macro_rules! driver {
    ($($name:ident($($arg:ident:$ty:ty),*) => $symbol:literal;)*) => {
        struct Driver { $($name:unsafe extern "C" fn($($ty),*)->i32,)* }
        impl Driver { unsafe fn load()->Result<Self> {
            let lib=unsafe {dlopen(c"libcuda.so.1".as_ptr(),2)};
            if lib.is_null() { return Err("cannot load libcuda.so.1".into()); }
            Ok(Self { $($name: {let p=unsafe {dlsym(lib,concat!($symbol,"\0").as_ptr().cast())}; if p.is_null(){return Err(format!("missing {}",$symbol));} unsafe {std::mem::transmute::<H,unsafe extern "C" fn($($ty),*)->i32>(p)} },)* })
        } }
    }
}
driver! {
 init(flags:u32)=>"cuInit";
 context(ctx:*mut H,flags:u32,device:i32)=>"cuCtxCreate_v2";
 destroy(ctx:H)=>"cuCtxDestroy_v2";
 stream(s:*mut H,flags:u32)=>"cuStreamCreate";
 stream_destroy(s:H)=>"cuStreamDestroy_v2";
 sync(s:H)=>"cuStreamSynchronize";
 alloc(p:*mut u64,bytes:usize)=>"cuMemAlloc_v2";
 free(p:u64)=>"cuMemFree_v2";
 upload(p:u64,host:*const c_void,bytes:usize)=>"cuMemcpyHtoD_v2";
 download(host:*mut c_void,p:u64,bytes:usize)=>"cuMemcpyDtoH_v2";
 memset(p:u64,value:u8,bytes:usize)=>"cuMemsetD8_v2";
 module(m:*mut H,image:*const c_void)=>"cuModuleLoadData";
 unload(m:H)=>"cuModuleUnload";
 function(f:*mut H,m:H,name:*const c_char)=>"cuModuleGetFunction";
 func_attr(f:H,attr:i32,value:i32)=>"cuFuncSetAttribute";
 launch(f:H,gx:u32,gy:u32,gz:u32,bx:u32,by:u32,bz:u32,sh:u32,s:H,args:*mut H,extra:*mut H)=>"cuLaunchKernel";
 launch_ex(config:*const c_void,f:H,args:*mut H,extra:*mut H)=>"cuLaunchKernelEx";
 event(e:*mut H,flags:u32)=>"cuEventCreate";
 event_record(e:H,s:H)=>"cuEventRecord";
 event_sync(e:H)=>"cuEventSynchronize";
 event_elapsed(ms:*mut f32,a:H,b:H)=>"cuEventElapsedTime";
 event_destroy(e:H)=>"cuEventDestroy_v2";
 capture(s:H,mode:i32)=>"cuStreamBeginCapture";
 capture_end(s:H,g:*mut H)=>"cuStreamEndCapture";
 instantiate(exec:*mut H,g:H,flags:u64)=>"cuGraphInstantiateWithFlags";
 graph_launch(exec:H,s:H)=>"cuGraphLaunch";
 graph_destroy(g:H)=>"cuGraphDestroy";
 exec_destroy(exec:H)=>"cuGraphExecDestroy";
 attribute(f:H,attr:i32,value:i32)=>"cuFuncSetAttribute";
 copy(dst:u64,src:u64,bytes:usize)=>"cuMemcpyDtoD_v2";
 global(p:*mut u64,bytes:*mut usize,m:H,name:*const c_char)=>"cuModuleGetGlobal_v2";
}
fn check(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(format!("CUDA Driver error {code}")) }
}
fn ptr<T>(v: &mut T) -> H {
    (v as *mut T).cast()
}
/// Weight rotation span per case: 1 GiB, or KBENCH_ROTATE_MIB (a TLB-cold probe of the
/// production working set; timing tooling only, never part of a proof).
fn rotate_bytes() -> usize {
    std::env::var("KBENCH_ROTATE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).map_or(1 << 30, |m| m << 20)
}
struct Gpu {
    d: Driver,
    ctx: H,
    s: H,
    allocs: Vec<u64>,
    modules: Vec<H>,
}
impl Gpu {
    fn new() -> Result<Self> {
        unsafe {
            let d = Driver::load()?;
            check((d.init)(0))?;
            let mut ctx = null_mut();
            check((d.context)(&mut ctx, 0, 0))?;
            let mut s = null_mut();
            check((d.stream)(&mut s, 0))?;
            Ok(Self { d, ctx, s, allocs: vec![], modules: vec![] })
        }
    }
    fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let mut p = 0;
        unsafe {
            check((self.d.alloc)(&mut p, bytes.max(4)))?;
        }
        self.allocs.push(p);
        Ok(p)
    }
    fn upload_at(&self, p: u64, data: &[u8]) -> Result<()> {
        unsafe { check((self.d.upload)(p, data.as_ptr().cast(), data.len())) }
    }
    fn upload(&mut self, data: &[u8]) -> Result<u64> {
        let p = self.alloc(data.len())?;
        unsafe {
            check((self.d.upload)(p, data.as_ptr().cast(), data.len()))?;
        }
        Ok(p)
    }
    fn words(&mut self, words: &[u32]) -> Result<u64> {
        self.upload(&words.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }
    fn read(&self, p: u64, n: usize) -> Result<Vec<u8>> {
        let mut data = vec![0; n];
        unsafe {
            check((self.d.download)(data.as_mut_ptr().cast(), p, n))?;
        }
        Ok(data)
    }
    fn fill(&mut self, n: usize, kind: u8) -> Result<u64> {
        let p = self.alloc(n)?;
        self.fill_at(p, n, kind)?;
        Ok(p)
    }
    fn fill_at(&self, p: u64, n: usize, kind: u8) -> Result<()> {
        let mut buf = vec![0u8; n.min(4 * 1024 * 1024)];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = match kind {
                0 => {
                    if i % 2 == 0 {
                        ((i / 2 * 13) % 256) as u8
                    } else {
                        0x3b
                    }
                }
                1 => {
                    if i % 4 == 3 {
                        0x3c
                    } else {
                        0
                    }
                }
                2 => 0x21,
                3 => 0x30,
                _ => 0,
            };
        }
        for off in (0..n).step_by(buf.len()) {
            let len = buf.len().min(n - off);
            unsafe {
                check((self.d.upload)(p + off as u64, buf.as_ptr().cast(), len))?;
            }
        }
        Ok(())
    }
    fn module(&mut self, data: &[u8]) -> Result<H> {
        let text = CString::new(data).map_err(|_| "PTX contains NUL")?;
        let mut m = null_mut();
        unsafe {
            check((self.d.module)(&mut m, text.as_ptr().cast()))?;
        }
        self.modules.push(m);
        Ok(m)
    }
    fn function(&self, m: H, name: &str) -> Result<H> {
        let c = CString::new(name).map_err(|_| "invalid symbol")?;
        let mut f = null_mut();
        unsafe {
            check((self.d.function)(&mut f, m, c.as_ptr()))?;
        }
        Ok(f)
    }
    /// A u32 module global, if the module defines it.
    fn module_u32(&self, m: H, name: &str) -> Option<u32> {
        let c = CString::new(name).ok()?;
        let (mut p, mut bytes) = (0u64, 0usize);
        unsafe {
            if (self.d.global)(&mut p, &mut bytes, m, c.as_ptr()) != 0 || bytes != 4 {
                return None;
            }
        }
        let v = self.read(p, 4).ok()?;
        Some(u32::from_le_bytes(v[..4].try_into().ok()?))
    }
    fn launch(&self, f: H, grid: [u32; 3], threads: u32, shared: u32, args: &mut [H]) -> Result<()> {
        unsafe { check((self.d.launch)(f, grid[0], grid[1], grid[2], threads, 1, 1, shared, self.s, args.as_mut_ptr(), null_mut())) }
    }
    /// cuLaunchKernelEx with CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION (6) = 1:
    /// the kernel may start once the previous one triggers; it must griddepcontrol.wait.
    fn launch_pdl(&self, f: H, grid: [u32; 3], threads: u32, shared: u32, args: &mut [H]) -> Result<()> {
        #[repr(C)]
        struct Attr { id: u32, pad: u32, value: [u32; 16] }
        #[repr(C)]
        struct Config { grid: [u32; 3], block: [u32; 3], shared: u32, stream: H, attrs: *const Attr, count: u32 }
        let mut value = [0u32; 16];
        value[0] = 1;
        let attr = Attr { id: 6, pad: 0, value };
        let config = Config { grid, block: [threads, 1, 1], shared, stream: self.s, attrs: &attr, count: 1 };
        unsafe { check((self.d.launch_ex)((&config as *const Config).cast(), f, args.as_mut_ptr(), null_mut())) }
    }
    fn sync(&self) -> Result<()> {
        unsafe { check((self.d.sync)(self.s)) }
    }
    fn clear(&mut self) -> Result<()> {
        self.sync()?;
        for p in self.allocs.drain(..).rev() {
            unsafe {
                check((self.d.free)(p))?;
            }
        }
        Ok(())
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            (self.d.sync)(self.s);
            for p in self.allocs.drain(..).rev() {
                (self.d.free)(p);
            }
            for m in self.modules.drain(..).rev() {
                (self.d.unload)(m);
            }
            (self.d.stream_destroy)(self.s);
            (self.d.destroy)(self.ctx);
        }
    }
}
const TIMER: &str = ".version 8.8\n.target sm_121\n.address_size 64\n.visible .entry kb_stamp(.param .u64 out) { .reg .b64 p,t; ld.param.u64 p,[out]; mov.u64 t,%globaltimer; st.global.u64 [p],t; ret; }\n";
struct Fixture {
    rows: params::Bf16Rows,
    group: Option<fp8::GroupedParams>,
    ring: Vec<u64>,
    out: u64,
    out_bytes: usize,
    aux: u64,
    aux_bytes: usize,
    grid: [u32; 3],
    threads: u32,
    shared: u32,
    stored: usize,
    flops: usize,
    chunks: usize,
    /// Prefill down only: token*11 output slots; experts write BF16 into the
    /// first 5120 bytes of ranks 0..9, the rest must stay poisoned.
    down_slots: usize,
    /// kind 9/10: raw FP32 [M][K] activation (X reads it) and main's pack kernel (A arm).
    raw: u64,
    pack: H,
    /// kind 64: flash_hc_norm_f32 of each arm (A packs separately; X emits packed planes).
    norms: [H; 2],
    /// Rotated weight bytes per copy (all matrices one launch may read).
    weight_bytes: usize,
    /// Output byte ranges the kernel owns; empty means the whole buffer.
    written: Vec<(usize, usize)>,
    qsa: Option<qsa::Fix>,
    gather: Option<qsa::Gather>,
    /// kind 11 (MXFP8): X arm E4M3 weight copies and their UE8M0 scale copies.
    ring_x: Vec<u64>,
    scales_x: Vec<u64>,
}
fn prefill(kind: u8) -> bool { kind == 5 || kind == 6 }
fn routed(kind: u8) -> bool { (20..=29).contains(&kind) }
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
/// E4M3FN value (codes 0x7f/0xff are NaN and never generated).
fn e4m3(b: u8) -> f64 {
    let (e, m) = ((b >> 3) & 15, (b & 7) as f64);
    let v = if e == 0 { m / 8. * 2f64.powi(-6) } else { (1. + m / 8.) * 2f64.powi(e as i32 - 7) };
    if b & 0x80 != 0 { -v } else { v }
}
/// kind 11: one E4M3 matrix (every code but the NaNs, zeros and subnormals included) with
/// UE8M0 scales 112..=124; A copies hold main's expand_to_bf16 of it (exact here), X copies
/// the bytes in the packed element order plus [row / 8][K / 32][8] scales (padding 127).
/// One E4M3 matrix (salted per matrix): its exact BF16 expansion in the packed element
/// order and its twin image (E4M3 bytes, then at the next 256-byte boundary the scales).
fn mx_matrix(name: &str, n: usize, k: usize, salt: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let code = |r: usize, col: usize| {
        let r = r + salt * 7919;
        let h = (r.wrapping_mul(2654435761) ^ col.wrapping_mul(40503)).wrapping_mul(2246822519) >> 9;
        let h = h % 254;
        (if h >= 0x7f { h + 1 } else { h }) as u8
    };
    let scale = |r: usize, block: usize| (112 + (r * 7 + block * 3) % 13) as u8;
    let (mut f8, mut bf) = (vec![0u8; n * k], vec![0u8; n * k * 2]);
    for r in 0..n {
        for col in 0..k {
            let at = params::flash_bf16_element(r, col, k, n);
            let b = code(r, col);
            f8[at] = b;
            let value = (e4m3(b) * 2f64.powi(scale(r, col / 32) as i32 - 127)) as f32;
            let bits = value.to_bits();
            if bits & 0xffff != 0 {
                return Err(format!("{name}: expansion not exact in BF16"));
            }
            bf[at * 2..at * 2 + 2].copy_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
    }
    let mut sc = vec![127u8; n.div_ceil(8) * (k / 32) * 8];
    for r in 0..n {
        for block in 0..k / 32 {
            sc[(r / 8 * (k / 32) + block) * 8 + r % 8] = scale(r, block);
        }
    }
    let mut img = f8;
    img.resize(img.len().div_ceil(256) * 256, 0);
    img.extend_from_slice(&sc);
    img.resize(img.len().div_ceil(256) * 256, 127);
    Ok((bf, img))
}
fn mx_fixture(g: &mut Gpu, c: &Case, order: bool) -> Result<Fixture> {
    let mut v = fixture(g, &Case { kind: 0, ..c.clone() }, order)?;
    let (n, k) = (c.n, c.k);
    let (bf, img) = mx_matrix(&c.name, n, k, 0)?;
    for &p in &v.ring {
        g.upload_at(p, &bf)?;
    }
    let wslot = (n * k).div_ceil(256) * 256;
    let slab = g.alloc(img.len() * v.ring.len())?;
    for i in 0..v.ring.len() {
        let w = slab + (i * img.len()) as u64;
        g.upload_at(w, &img)?;
        v.ring_x.push(w);
        v.scales_x.push(w + wslot as u64);
    }
    Ok(v)
}
/// kind 12 (law check): raw FP32 [M][K] activation whose row r depends on r only, the
/// production pack (flash_bf16_input, latency lanes at M <= 16) and the production
/// kernel and launch shape for M; output [M][N] FP32.
fn law_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, n, k) = (c.m, c.n, c.k);
    let weight_bytes = n * k * 2;
    let slab = g.fill(weight_bytes * 2, 0)?;
    let ring = vec![slab, slab + weight_bytes as u64];
    let raw: Vec<u8> = (0..m * k)
        .flat_map(|i| ((((i / k) * 131 + (i % k) * 7) % 257) as f32 / 64. - 2.).to_le_bytes())
        .collect();
    let raw = g.upload(&raw)?;
    let plane = m.div_ceil(16) * k.div_ceil(16) * 128;
    let input = g.fill(plane * 2 * 4, 0)?;
    let out_bytes = m * n * 4;
    let out = g.alloc(out_bytes)?;
    let aux = g.fill(4 * m * n * 4, 1)?;
    let rows = params::Bf16Rows {
        weight: ring[0], output: out, rows: n as u32, stride: n as u32, weight2: 0, output2: 0, rows2: 0, stride2: 0,
        input, cols: k as u32, tokens: m as u32, input_bf16: 1, hc_silu: 0, input_plane_words: plane as u64,
    };
    let (grid, threads, shared) = if m <= 16 {
        ([192, 1, 1], if n <= 640 { 32 } else { 128 }, if n <= 640 { RING } else { 4 * RING })
    } else if m < 64 {
        ([(n.div_ceil(32) * m.div_ceil(16)) as u32, 1, 1], 128, 0)
    } else {
        ([m.div_ceil(64) as u32, n.div_ceil(128) as u32, 1], 256, 65536)
    };
    Ok(Fixture {
        rows, group: None, ring, out, out_bytes, aux, aux_bytes: 0, grid, threads, shared,
        stored: weight_bytes + m * k * 4 + out_bytes, flops: 2 * m * n * k, chunks: 1, down_slots: 0, raw,
        pack: null_mut(), norms: [null_mut(); 2], weight_bytes, written: Vec::new(), ring_x: vec![], scales_x: vec![], qsa: None, gather: None,
    })
}
fn fixture(g: &mut Gpu, c: &Case, order: bool) -> Result<Fixture> {
    if c.kind == 11 {
        return mx_fixture(g, c, order);
    }
    if c.kind == 77 { return head::norm_fixture(g, c); }
    if c.kind == 12 {
        return law_fixture(g, c);
    }
    if c.kind == 13 {
        return mxgroup_fixture(g, c, order);
    }
    if ((60..=69).contains(&c.kind) && c.kind != 63) || c.kind == 74 {
        return prompt::fixture(g, c);
    }
    if c.kind == 9 || c.kind == 10 {
        let mut base = c.clone();
        base.kind = if c.kind == 9 { 0 } else { 8 };
        let mut v = fixture(g, &base, order)?;
        v.raw = g.fill(c.m * c.k * 4, 0)?;
        v.stored += c.m * c.k * 4;
        return Ok(v);
    }
    let rows = params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0,
        input: 0, cols: 0, tokens: 0, input_bf16: 0, hc_silu: 0, input_plane_words: 0 };
    if c.kind == 63 {
        let out_bytes = c.m * 256 * 4;
        let out = g.alloc(out_bytes)?;
        return Ok(Fixture { rows, group: None, ring: vec![out, out], out, out_bytes, aux: 0, aux_bytes: 0,
            grid: [c.m as u32, 1, 1], threads: 256, shared: 0, stored: 0, flops: c.m * 8 * c.n * 8 * 4096, chunks: 1,
            down_slots: 0, raw: 0, pack: null_mut(), norms: [null_mut(); 2], weight_bytes: 0, written: Vec::new(),
            qsa: None, gather: None, ring_x: vec![], scales_x: vec![] });
    }
    if c.kind == 33 {
        let (gt, out_bytes, stored) = qsa::gather_fixture(g, c.m, c.n, c.k != 0)?;
        return Ok(Fixture { rows, group: None, ring: vec![gt.out, gt.out], out: gt.out, out_bytes, aux: 0, aux_bytes: 0,
            grid: [(c.m * qsa::KHEADS) as u32, 1, 1], threads: 128, shared: 0, stored, flops: 0, chunks: 1,
            down_slots: 0,
            raw: 0, pack: null_mut(), norms: [null_mut(); 2],
            weight_bytes: 0, written: Vec::new(),
            qsa: None, gather: Some(gt), ring_x: vec![], scales_x: vec![] });
    }
    if (30..=32).contains(&c.kind) || (34..=38).contains(&c.kind) {
        let mut q = if c.kind == 38 { qsa::batch_fixture(g, c.k, c.m, c.n)? } else { qsa::fixture(g, c.kind - 23, c.m, c.n, order)? };
        qsa::set_launch(c, order, &mut q)?; // qsa_prod_* / qsa_batch_*: production entries and grids
        return Ok(Fixture { rows, group: None, ring: vec![q.out, q.out], out: q.out, out_bytes: q.out_bytes, aux: 0, aux_bytes: 0,
            grid: q.launch[0].0, threads: q.launch[0].1, shared: q.launch[0].2, stored: q.stored, flops: 0, chunks: 1,
            down_slots: 0,
            raw: 0, pack: null_mut(), norms: [null_mut(); 2],
            weight_bytes: 0, written: Vec::new(),
            qsa: Some(q), gather: None, ring_x: vec![], scales_x: vec![] });
    }
    let pad = std::env::var("VADE_KBENCH_ARENA_PAD_MIB").unwrap_or_else(|_| "0".into()).parse::<usize>().map_err(|_| "invalid arena padding")?.min(64);
    if pad > 0 { g.fill(pad * 1024 * 1024, 0)?; }
    if prefill(c.kind) {
        return prefill_fixture(g, c);
    }
    if routed(c.kind) {
        return routed_fixture(g, c);
    }
    if matches!(c.kind,71..=73) { return fp8group::fixture(g,c); }
    if matches!(c.kind, 75 | 76) { return rows::fixture(g, c); }
    let (m, n, k) = (c.m, c.n, c.k);
    if c.kind == 8 {
        return group_fixture(g, c);
    }
    let grouped = c.kind >= 3;
    let down = c.kind == 4 || c.kind == 6;
    let prefill = c.kind >= 5;
    let experts = if prefill { 512 } else { 10 };
    let rows = if prefill { m * 10 } else { experts * m };
    let weight_bytes = if grouped { experts * super::floor::nvfp4(n, k) * if down { 1 } else { 2 } } else { n * k * 2 };
    let latency = c.kind == 0 && (m <= 16 || c.name.starts_with("mx-")); // mx- at M > 16: latency launches in both arms (73bc4d15)
    let full = rotate_bytes().div_ceil(weight_bytes).max(2);
    let copies = if latency { full.min((64 * 1024 * 1024usize).div_ceil(weight_bytes).max(16384)) } else { full };
    let out_bytes = if grouped { rows * n * if down { 4 } else { 2 } } else { m * if c.kind == 2 { 2560 } else { n } * 4 };
    let out = g.alloc(out_bytes)?;
    let aux_bytes = if c.kind == 1 {
        m * 192 * 3 * 4
    } else if c.kind == 2 {
        m * 10240 * 4
    } else {
        4 * m * n * 4
    };
    let aux = g.fill(aux_bytes, 1)?;
    let mut ring = Vec::new();
    let mut group = None;
    let input_bytes = if grouped {
        if down { rows * k * 2 } else { m * k * 4 }
    } else if c.kind == 1 {
        m.div_ceil(8) * (k / 16) * 32 * 16
    } else {
        m.div_ceil(16) * k.div_ceil(16) * 128 * 8
    };
    let input = g.fill(input_bytes, if grouped && !down { 1 } else { 0 })?;
    let mut p = params::Bf16Rows {
        weight: 0,
        output: out,
        rows: n as u32,
        stride: if c.kind == 2 { 2560 } else { n } as u32,
        weight2: 0,
        output2: 0,
        rows2: 0,
        stride2: 0,
        input,
        cols: k as u32,
        tokens: m as u32,
        input_bf16: 1,
        hc_silu: 0,
        input_plane_words: (m.div_ceil(16) * k.div_ceil(16) * 128) as u64,
    };
    if grouped {
        let count = if prefill { 40 } else { m };
        let counts = g.words(&(0..512).map(|id| if id < experts { count as u32 } else { 0 }).collect::<Vec<_>>())?;
        let starts = g.words(&(0..513).map(|id| (id.min(experts) * count) as u32).collect::<Vec<_>>())?;
        let sorted = g.words(&(0..rows).map(|i| if down { i } else { i % m } as u32).collect::<Vec<_>>())?;
        let pair_to_row = if down { g.words(&(0..rows as u32).collect::<Vec<_>>())? } else { 0 };
        let task_capacity = if prefill { m * 10 / 128 + 512 } else { m * 10 / 128 + 10 };
        let mut tasks = vec![experts as u32];
        for id in 0..experts {
            tasks.extend([id as u32, 0]);
        }
        let tasks = g.words(&tasks)?;
        let route = fp8::GroupedRouteParams {
            error: 0,
            pairs: 0,
            counts,
            starts,
            pair_to_row,
            token_pairs: 0,
            token_count: 0,
            tasks,
            sorted_row: sorted,
            output_row: 0,
            experts: 512,
            tokens: m as u32,
            pair_count: (m * 10) as u32,
            row_capacity: rows as u32,
        };
        let slot = super::floor::nvfp4(n, k).div_ceil(256) * 256;
        let slab = g.alloc(copies * experts * if down { 1 } else { 2 } * slot)?;
        eprintln!("weights case={} slab={}", c.name, slab);
        let mut offset = 0;
        for _ in 0..copies {
            let mut descriptors = Vec::new();
            for _ in 0..experts {
                let mut matrix = || -> Result<fp4::Matrix> {
                    let weights = slab + offset as u64;
                    offset += slot;
                    let scales = weights + (n * k / 2) as u64;
                    let global_scale = scales + (n * k / 16) as u64;
                    g.fill_at(weights, n * k / 2, 2)?;
                    g.fill_at(scales, n * k / 16, 3)?;
                    unsafe {
                        check((g.d.upload)(global_scale, 1f32.to_le_bytes().as_ptr().cast(), 4))?;
                    }
                    Ok(fp4::Matrix { weights, scales, global_scale })
                };
                let a = matrix()?;
                let b = if down { fp4::Matrix::default() } else { matrix()? };
                let ex = fp4::Expert {
                    gate: if down { fp4::Matrix::default() } else { a },
                    up: b,
                    down: if down { a } else { fp4::Matrix::default() },
                    width: 640,
                    reserved: 0,
                };
                for v in [
                    ex.gate.weights,
                    ex.gate.scales,
                    ex.gate.global_scale,
                    ex.up.weights,
                    ex.up.scales,
                    ex.up.global_scale,
                    ex.down.weights,
                    ex.down.scales,
                    ex.down.global_scale,
                ] {
                    descriptors.extend(v.to_le_bytes());
                }
                descriptors.extend(ex.width.to_le_bytes());
                descriptors.extend(0u32.to_le_bytes());
            }
            descriptors.resize(512 * 80, 0);
            ring.push(g.upload(&descriptors)?);
        }
        group = Some(fp8::GroupedParams {
            route,
            weights: ring[0],
            input,
            input_scales: 0,
            output: out,
            cols: 2560,
            activation_quantization: 0,
            output_shard: 0,
        });
        return Ok(Fixture {
            rows: p,
            group,
            ring,
            out,
            out_bytes,
            aux,
            aux_bytes: 0,
            grid: [
                if prefill {
                    if down { 20 } else { 5 }
                } else if down {
                    160
                } else {
                    40
                },
                task_capacity as u32,
                if prefill { 4 } else { 1 },
            ],
            threads: if prefill { if down { 128 } else { 256 } } else { 64 },
            shared: 0,
            stored: weight_bytes + input_bytes + out_bytes,
            flops: 2 * rows * n * k * if down { 1 } else { 2 },
            chunks: if prefill { 4 } else { 1 },
            down_slots: 0,
            raw: 0,
            pack: null_mut(), norms: [null_mut(); 2],
            weight_bytes,
            written: if c.kind == 4 { (0..rows).map(|r| (r * 2560 * 4, 1280 * 4)).collect() } else { Vec::new() },
            qsa: None,
            gather: None,
            ring_x: vec![],
            scales_x: vec![],
        });
    }
    let slab = g.fill(weight_bytes * copies, 0)?;
    ring.extend((0..copies).map(|i| slab + (i * weight_bytes) as u64));
    p.weight = ring[0];
    Ok(Fixture {
        rows: p,
        group,
        ring,
        out,
        out_bytes,
        aux,
        aux_bytes: if c.kind == 1 { aux_bytes } else { 0 },
        grid: if c.kind == 1 {
            [192, 1, 1]
        } else if c.kind == 2 {
            [320, m.div_ceil(16) as u32, 1]
        } else if latency {
            [192, 1, 1]
        } else {
            [n.div_ceil(32) as u32, 1, 1]
        },
        threads: if c.kind == 2 && (c.kernel.contains("latency4") || c.kernel.contains("latency16")) { 128 } else if c.kind == 2 || (latency && n <= 640) { 32 } else { 128 },
        shared: if latency { if n <= 640 { RING } else { 4 * RING } } else { 0 },
        stored: weight_bytes + input_bytes + out_bytes + if c.kind == 1 || c.kind == 2 { aux_bytes } else { 0 },
        flops: 2 * m * n * k,
        chunks: 1,
        down_slots: 0,
        raw: 0,
        pack: null_mut(), norms: [null_mut(); 2],
        weight_bytes,
        written: Vec::new(),
        qsa: None,
        gather: None,
        ring_x: vec![],
        scales_x: vec![],
    })
}
/// Deterministic pseudo-random bytes: weights any nibble, positive E4M3 scales in
/// [2^-4, 7.5], finite F32 tokens in +-[2^-4, 4), BF16 middles in +-[2^-7, 4).
fn noise(seed: u64, n: usize, kind: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 8);
    let mut i = 0u64;
    while out.len() < n {
        let h = mix(seed ^ i.wrapping_mul(0x1000_0000_01b3));
        i += 1;
        match kind {
            0 => out.extend(h.to_le_bytes()),
            1 => out.extend((0..8).map(|j| {
                let b = (h >> (j * 8)) as u8;
                (((3 + b % 7) << 3) | (b >> 5)) & 0x7f
            })),
            2 => {
                for j in 0..2 {
                    let w = (h >> (j * 32)) as u32;
                    let bits = (w & 0x8000_0000) | ((123 + (w >> 23) % 6) << 23) | (w & 0x7f_ffff);
                    out.extend(bits.to_le_bytes());
                }
            }
            _ => {
                for j in 0..4 {
                    let w = (h >> (j * 16)) as u32 & 0xffff;
                    let bits = (w & 0x8000) | ((120 + (w >> 7) % 9) << 7) | (w & 0x7f);
                    out.extend((bits as u16).to_le_bytes());
                }
            }
        }
    }
    out.truncate(n);
    out
}
/// Production-shaped prompt chunk: grid (20|80, tasks(2048), 1), block 256|128,
/// expert-major M128 tasks in ascending expert order, compact rows sorted by pair,
/// pair -> token*11 + ascending-ID rank, tokens scattered across the chunk.
fn prefill_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (t, n, k) = (c.m, c.n, c.k);
    let down = c.kind == 6;
    let counts = fp4_prompt::counts(c.real, &c.name, t)?;
    let pairs = t * 10;
    if counts.iter().sum::<u32>() as usize != pairs || counts.iter().any(|&v| v as usize >= t) {
        return Err("prompt counts must cover 20480 pairs with < 2048 rows per expert".into());
    }
    let mut starts = vec![0u32; 513];
    for e in 0..512 {
        starts[e + 1] = starts[e] + counts[e];
    }
    let token_of = |i: usize| (i % t) * 1237 % t;
    let mut expert_of = vec![0u32; pairs];
    let mut rows_of: Vec<Vec<u32>> = vec![Vec::new(); 512];
    for e in 0..512 {
        for i in starts[e] as usize..starts[e + 1] as usize {
            let pair = token_of(i) * 10 + i / t;
            expert_of[pair] = e as u32;
            rows_of[e].push(pair as u32);
        }
    }
    let mut sorted = Vec::with_capacity(pairs);
    for rows in rows_of.iter_mut() {
        rows.sort_unstable();
        sorted.extend_from_slice(rows);
    }
    let mut pair_to_row = vec![0u32; pairs];
    for tok in 0..t {
        let ids = &expert_of[tok * 10..tok * 10 + 10];
        for slot in 0..10 {
            let rank = ids.iter().filter(|&&id| id < ids[slot]).count();
            pair_to_row[tok * 10 + slot] = (tok * 11 + rank) as u32;
        }
    }
    let capacity = pairs.div_ceil(128) + pairs.min(512);
    let mut tasks = vec![0u32; 1 + 2 * capacity];
    let mut count = 0usize;
    for e in 0..512 {
        for first in (0..counts[e]).step_by(128) {
            tasks[1 + 2 * count] = e as u32;
            tasks[2 + 2 * count] = first;
            count += 1;
        }
    }
    tasks[0] = count as u32;
    let active = counts.iter().filter(|&&v| v > 0).count();
    let route = fp8::GroupedRouteParams {
        error: 0,
        pairs: 0,
        counts: g.words(&counts)?,
        starts: g.words(&starts)?,
        pair_to_row: g.words(&pair_to_row)?,
        token_pairs: 0,
        token_count: 0,
        tasks: g.words(&tasks)?,
        sorted_row: g.words(&sorted)?,
        output_row: 0,
        experts: 512,
        tokens: t as u32,
        pair_count: pairs as u32,
        row_capacity: pairs as u32,
    };
    let input = if down {
        g.upload(&noise(0x51de_0000_0000_0003, pairs * k * 2, 3))?
    } else {
        g.upload(&noise(0x51de_0000_0000_0002, t * k * 4, 2))?
    };
    let (out, out_bytes) = if down {
        let bytes = t * 11 * 2560 * 4;
        (g.alloc(bytes)?, bytes)
    } else {
        let bytes = pairs * n * 2;
        (g.alloc(bytes)?, bytes)
    };
    let matrices = if down { 1 } else { 2 };
    let matrix_bytes = super::floor::nvfp4(n, k);
    let slot = matrix_bytes.div_ceil(256) * 256;
    let weight_bytes = 512 * matrices * matrix_bytes;
    let copies = (1024 * 1024 * 1024usize).div_ceil(weight_bytes).max(2);
    let slab = g.alloc(copies * 512 * matrices * slot)?;
    eprintln!("weights case={} slab={} copies={} active_experts={} tasks={}", c.name, slab, copies, active, count);
    let mut images = Vec::with_capacity(512 * matrices);
    for e in 0..512u64 {
        for w in 0..matrices as u64 {
            let seed = mix(e * 4 + w + if down { 2 } else { 0 });
            let mut image = noise(seed, n * k / 2, 0);
            image.extend(noise(seed ^ 0x5ca1e, n * k / 16, 1));
            let global = f32::from_bits(0x3e00_0000 | (mix(seed ^ 0x91) as u32 & 0x7f_ffff));
            image.extend(global.to_le_bytes());
            images.push(image);
        }
    }
    let mut ring = Vec::new();
    let mut offset = 0u64;
    for _ in 0..copies {
        let mut descriptors = Vec::new();
        for e in 0..512 {
            let mut place = |w: usize| -> Result<fp4::Matrix> {
                let weights = slab + offset;
                offset += slot as u64;
                let image = &images[e * matrices + w];
                unsafe {
                    check((g.d.upload)(weights, image.as_ptr().cast(), image.len()))?;
                }
                Ok(fp4::Matrix { weights, scales: weights + (n * k / 2) as u64, global_scale: weights + (n * k / 2 + n * k / 16) as u64 })
            };
            let a = place(0)?;
            let b = if down { fp4::Matrix::default() } else { place(1)? };
            let ex = fp4::Expert {
                gate: if down { fp4::Matrix::default() } else { a },
                up: b,
                down: if down { a } else { fp4::Matrix::default() },
                width: 640,
                reserved: 0,
            };
            for v in [
                ex.gate.weights,
                ex.gate.scales,
                ex.gate.global_scale,
                ex.up.weights,
                ex.up.scales,
                ex.up.global_scale,
                ex.down.weights,
                ex.down.scales,
                ex.down.global_scale,
            ] {
                descriptors.extend(v.to_le_bytes());
            }
            descriptors.extend(ex.width.to_le_bytes());
            descriptors.extend(0u32.to_le_bytes());
        }
        ring.push(g.upload(&descriptors)?);
    }
    let rows = params::Bf16Rows {
        weight: 0,
        output: out,
        rows: n as u32,
        stride: n as u32,
        weight2: 0,
        output2: 0,
        rows2: 0,
        stride2: 0,
        input,
        cols: k as u32,
        tokens: t as u32,
        input_bf16: 1,
        hc_silu: 0,
        input_plane_words: 0,
    };
    let group = fp8::GroupedParams {
        route,
        weights: ring[0],
        input,
        input_scales: 0,
        output: out,
        cols: 2560,
        activation_quantization: 0,
        output_shard: 0,
    };
    let stored = active * matrices * matrix_bytes
        + if down { pairs * k * 2 + pairs * n * 2 } else { t * k * 4 + pairs * n * 2 };
    Ok(Fixture {
        rows,
        group: Some(group),
        ring,
        out,
        out_bytes,
        aux: if down { 0 } else { g.alloc(t * k * 2)? },
        aux_bytes: 0,
        grid: [if down { 80 } else { 20 }, capacity as u32, 1],
        threads: if down { 128 } else { 256 },
        shared: 0,
        stored,
        flops: 2 * pairs * n * k * matrices,
        chunks: (8192 / t).max(1),
        down_slots: if down { t * 11 } else { 0 },
        raw: 0,
        pack: null_mut(), norms: [null_mut(); 2],
        weight_bytes,
        written: Vec::new(),
        qsa: None,
        gather: None,
        ring_x: vec![],
        scales_x: vec![],
    })
}
/// kind 8: matrices j = 0..4 share one packed M<=16 input; per ring copy their weights are
/// consecutive (each N_j*K*2 bytes), outputs and 4*M*N_j partial floats likewise.
fn group_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, k) = (c.m, c.k);
    let total: usize = c.mats.iter().sum();
    let weight_bytes = total * k * 2;
    let full = rotate_bytes().div_ceil(weight_bytes).max(2);
    let copies = full.min((64 * 1024 * 1024usize).div_ceil(weight_bytes).max(16384));
    let out_bytes = m * total * 4;
    let out = g.alloc(out_bytes)?;
    let aux = g.fill(4 * m * total * 4, 1)?;
    let input_bytes = m.div_ceil(16) * k.div_ceil(16) * 128 * 8;
    let input = g.fill(input_bytes, 0)?;
    let rows = params::Bf16Rows {
        weight: 0,
        output: out,
        rows: total as u32,
        stride: total as u32,
        weight2: 0,
        output2: 0,
        rows2: 0,
        stride2: 0,
        input,
        cols: k as u32,
        tokens: m as u32,
        input_bf16: 1,
        hc_silu: 0,
        input_plane_words: (m.div_ceil(16) * k.div_ceil(16) * 128) as u64,
    };
    let slab = g.fill(weight_bytes * copies, 0)?;
    let ring = (0..copies).map(|i| slab + (i * weight_bytes) as u64).collect();
    let threads = if total <= 640 { 32 } else { 128 };
    Ok(Fixture {
        rows,
        group: None,
        ring,
        out,
        out_bytes,
        aux,
        aux_bytes: 0,
        grid: [192, 1, 1],
        threads,
        shared: threads / 32 * RING,
        stored: weight_bytes + input_bytes + out_bytes,
        flops: 2 * m * total * k,
        chunks: 1,
        down_slots: 0,
        raw: 0,
        pack: null_mut(), norms: [null_mut(); 2],
        weight_bytes,
        written: Vec::new(),
        qsa: None,
        gather: None,
        ring_x: vec![],
        scales_x: vec![],
    })
}
/// Main's pack of the raw activation into the reserved packed tile (A arm of kind 9/10).
fn launch_pack(g: &Gpu, c: &Case, v: &Fixture) -> Result<()> {
    let mut p = params::FlashBf16Input { input: v.raw, output: v.rows.input, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: 0 };
    let words = c.m.div_ceil(16) * c.k.div_ceil(16) * 128;
    g.launch(v.pack, [words.div_ceil(4).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut p)])
}
/// Matrix j of a kind-8 case at ring copy i: (weight, output, rows, partials).
fn group_matrix(c: &Case, v: &Fixture, i: usize, j: usize) -> (u64, u64, usize, u64) {
    let before: usize = c.mats[..j].iter().sum();
    (
        v.ring[i] + (before * c.k * 2) as u64,
        v.out + (c.m * before * 4) as u64,
        c.mats[j],
        v.aux + (4 * c.m * before * 4) as u64,
    )
}
fn launch_group(g: &Gpu, f: H, merge: H, c: &Case, v: &mut Fixture, i: usize, arm: usize) -> Result<()> {
    if arm == 0 && c.kind != 13 {
        // Main: one solo latency launch per matrix (+ its merge when N<=640), as production.
        for j in 0..4 {
            let (w, y, n, part) = group_matrix(c, v, i, j);
            if n == 0 {
                continue;
            }
            let mut p = v.rows;
            (p.weight, p.output, p.rows, p.stride) = (w, y, n as u32, n as u32);
            let mut partials = part;
            let threads = if n <= 640 { 32 } else { 128 };
            g.launch(f, [192, 1, 1], threads, threads / 32 * RING, &mut [ptr(&mut p), ptr(&mut partials)])?;
            if n <= 640 {
                g.launch(merge, [(n * c.m).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut p), ptr(&mut partials)])?;
            }
        }
        return Ok(());
    }
    let mut pairs = params::Bf16Pairs { pairs: [v.rows; 2] };
    let rowsx = c.kind == 13 && mxrows_x().is_some();
    if c.kind == 10 {
        for pair in pairs.pairs.iter_mut() {
            (pair.input, pair.input_bf16) = (v.raw, 2);
        }
    }
    for (slot, pair) in pairs.pairs.iter_mut().enumerate() {
        let (w0, y0, n0, _) = group_matrix(c, v, i, 2 * slot);
        let (w1, y1, n1, _) = group_matrix(c, v, i, 2 * slot + 1);
        (pair.weight, pair.output, pair.rows, pair.stride) = (w0, y0, n0 as u32, n0 as u32);
        (pair.weight2, pair.output2, pair.rows2, pair.stride2) = (w1, y1, n1 as u32, n1 as u32);
        // As the engine: M5-16 flags MXFP8 only in all-MXFP8 groups (latency_mx8_wide/16 K).
        // KBENCH_MXROWS_X: A = main's M5-8 launch (MoE group BF16 throughout), X = twins.
        if if rowsx { arm == 1 || all_mx(c) || mxrows_a().is_some() } else { c.kind == 13 && arm == 1 && (c.m <= 4 || all_mx(c) && (params::latency_mx8_wide(c.m, c.k) || params::latency_mx8_wide16(c.m, c.k) || params::latency_mx8_widen(c.m, c.k))) } {
            pair.weight = mxgroup_weight(c, v, i, 2 * slot, w0);
            pair.weight2 = mxgroup_weight(c, v, i, 2 * slot + 1, w1);
        }
    }
    // Group launch: 128 threads always; groups of <= 640 rows merge inside the CTA.
    let _ = merge;
    if c.name.starts_with("hc_down_act") { return head::launch_down_act(g, f, c, v, &mut pairs, arm); }
    let mut partials = v.aux;
    let shared = if rowsx { mxrows_shared(c, arm) } else if c.kind == 13 && arm == 1 && all_mx(c) { params::latency_mx8_shared(c.m, c.k, 4) as u32 } else { 4 * RING };
    g.launch(f, [192, 1, 1], 128, shared, &mut [ptr(&mut pairs), ptr(&mut partials)])
}
/// An arm's gate A pre-pass, <KERNEL>_SHARED bytes and <KERNEL>_THREADS, when its module has them.
type Pre = (Option<H>, Option<u32>, Option<u32>);
fn launch(g: &Gpu, f: H, merge: H, c: &Case, v: &mut Fixture, i: usize, arm: usize, (pre, shared, threads): Pre) -> Result<()> {
    if c.kind == 1 { return head::launch(g, f, c, v, i, arm); }
    if matches!(c.kind, 75 | 76) { return rows::launch(g, f, c, v, i, arm); }
    if c.kind == 2 && c.real || c.kind == 77 { return head::launch_hc(g, f, c, v, i, arm); }
    if let Some(q) = v.qsa.as_mut() {
        return qsa::launch(g, f, q, arm);
    }
    if let Some(t) = v.gather.as_mut() {
        return g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut t.ids), ptr(&mut t.base), ptr(&mut t.out),
            ptr(&mut t.tokens), ptr(&mut t.stride), ptr(&mut t.slow)]);
    }
    // An arm whose module exports <KERNEL>_SHARED launches with its own dynamic shared bytes.
    if (60..=69).contains(&c.kind) || c.kind == 74 {
        return prompt::launch(g, f, merge, c, v, i, arm, shared);
    }
    if c.kind == 12 {
        // KBENCH_LAW_LANES=32: main's pack has no lanes field and packs every lane.
        let main_abi = std::env::var("KBENCH_LAW_LANES").is_ok_and(|v| v == "32");
        let lanes = if c.m <= 16 && c.k % 64 == 0 && !main_abi { params::flash_latency_lanes(c.m) } else { 32 };
        let mut q = params::FlashBf16Input { input: v.raw, output: v.rows.input, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: lanes as u32 };
        let words = c.m.div_ceil(16) * c.k.div_ceil(16) * 128;
        g.launch(v.pack, [(words / 128 * lanes).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        v.rows.weight = v.ring[i];
        let mut args = vec![ptr(&mut v.rows)];
        if c.m <= 16 {
            args.push(ptr(&mut v.aux));
        }
        g.launch(f, v.grid, v.threads, shared.unwrap_or(v.shared), &mut args)?;
        if !merge.is_null() {
            g.launch(merge, [(c.n * c.m).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut v.rows), ptr(&mut v.aux)])?;
        }
        return Ok(());
    }
    if c.kind == 11 {
        if arm == 0 {
            return if c.m > 16 { let r0 = v.rows; for (h, t) in [(0usize, 16usize), (1, c.m - 16)] { v.rows.input = r0.input + (h * c.k / 16 * 512) as u64; v.rows.output = r0.output + (h * 16 * c.n * 4) as u64; v.rows.tokens = t as u32; launch(g, f, merge, &Case { kind: 0, m: t, ..c.clone() }, v, i, arm, (pre, shared, threads))?; } v.rows = r0; Ok(()) } else { launch(g, f, merge, &Case { kind: 0, ..c.clone() }, v, i, arm, (pre, shared, threads)) };
        }
        v.rows.weight = v.ring_x[i];
        if c.kernel == "flash_mx8_mma_latency_group" { return mx_range_warps(g, f, v, i); }
        let mut scales = v.scales_x[i];
        let shared = if c.kernel == "flash_verify_mx_solo97" { params::verify_solo97_shared(c.k) } else { params::latency_mx8_shared(c.m, c.k, v.threads as usize / 32) } as u32;
        g.launch(f, v.grid, v.threads, shared, &mut [ptr(&mut v.rows), ptr(&mut scales), ptr(&mut v.aux)])?;
        if !merge.is_null() {
            g.launch(merge, [(c.n * c.m).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut v.rows), ptr(&mut v.aux)])?;
        }
        v.rows.weight = v.ring[i];
        return Ok(());
    }
    if (c.kind == 9 || c.kind == 10) && arm == 0 {
        launch_pack(g, c, v)?;
    }
    if c.kind == 8 || c.kind == 10 || c.kind == 13 {
        return launch_group(g, f, merge, c, v, i, arm);
    }
    if c.kind == 9 && arm == 1 {
        let packed = v.rows.input;
        (v.rows.input, v.rows.input_bf16) = (v.raw, 2);
        let r = launch(g, f, merge, &Case { kind: 0, ..c.clone() }, v, i, arm, (pre, shared, threads));
        (v.rows.input, v.rows.input_bf16) = (packed, 1);
        return r;
    }
    if c.kind == 26 || (matches!(c.kind, 20..=22) && flush_on()) {
        icflush(g)?;
        if c.kind == 26 {
            return Ok(());
        }
    }
    if c.kind == 22 {
        return launch_layer(g, f, merge, v, i, arm);
    }
    if matches!(c.kind, 23 | 27 | 28 | 29) {
        return launch_route(g, f, merge, v, arm);
    }
    if c.kind == 24 {
        return launch_moe_route(g, f, v, arm);
    }
    if c.kind == 25 {
        return launch_finish(g, f, v);
    }
    if let Some(p) = v.group.as_mut() {
        p.weights = v.ring[i];
        p.input = v.rows.input;
        if let Some(pre) = pre {
            let (mut src, mut dst, mut groups) = (v.rows.input, v.aux, (c.m * c.k / 8) as u32);
            g.launch(pre, [groups.div_ceil(256), 1, 1], 256, 0, &mut [ptr(&mut src), ptr(&mut dst), ptr(&mut groups)])?;
            p.input = v.aux;
        }
        // KBENCH_XGRIDX: X's grid x (several column blocks per CTA; prompt grid trims).
        let mut grid = v.grid;
        if arm == 1 && (routed(c.kind) || matches!(c.kind, 5 | 6)) {
            if let Some(x) = std::env::var("KBENCH_XGRIDX").ok().and_then(|x| x.parse().ok()) { grid[0] = x; }
        }
        if fp8group::paired(c,arm) { grid[0]/=2; }
        g.launch(f, grid, threads.unwrap_or(v.threads), shared.unwrap_or(v.shared), &mut [ptr(p)])
    } else {
        v.rows.weight = v.ring[i];
        let mut args = vec![ptr(&mut v.rows)];
        if c.kind == 1 || c.kind == 2 || ((c.kind == 0 || c.kind == 9) && c.m <= 16) {
            args.push(ptr(&mut v.aux));
        }
        g.launch(f, v.grid, if arm == 1 && super::fold_x(c) { 128 } else { v.threads }, if arm == 1 && super::fold_x(c) { params::latency_fold_shared(c.m, c.k, c.n) as u32 } else { v.shared }, &mut args)?;
        if !merge.is_null() {
            // Stream-ordered second pass: (p0+p1)+(p2+p3) over the four range partials.
            let mut margs = vec![ptr(&mut v.rows), ptr(&mut v.aux)];
            g.launch(merge, [(c.n * c.m).div_ceil(128) as u32, 1, 1], 128, 0, &mut margs)?;
        }
        Ok(())
    }
}
struct Graph<'a> {
    g: &'a Gpu,
    raw: H,
    exec: H,
    start: H,
    end: H,
}
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.g.d.exec_destroy)(self.exec);
            (self.g.d.graph_destroy)(self.raw);
            (self.g.d.event_destroy)(self.start);
            (self.g.d.event_destroy)(self.end);
        }
    }
}
fn graph<'a>(g: &'a Gpu, f: H, merge: H, arm: usize, px: Pre, timer: H, times: u64, c: &Case, v: &mut Fixture) -> Result<Graph<'a>> {
    capture_graph(g, || {
        let mut t = times;
        g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut t)])?;
        for chunk in 0..v.chunks {
            for i in 0..v.ring.len() {
                launch(g, f, merge, c, v, (i + chunk) % v.ring.len(), arm, px)?;
            }
        }
        t = times + 8;
        g.launch(timer, [1, 1, 1], 1, 0, &mut [ptr(&mut t)])
    })
}
/// One CUDA graph of the launches `body` enqueues (it brackets them with its own stamps).
fn capture_graph<'a>(g: &'a Gpu, body: impl FnOnce() -> Result<()>) -> Result<Graph<'a>> {
    let (mut raw, mut exec, mut start, mut end) = (null_mut(), null_mut(), null_mut(), null_mut());
    unsafe {
        check((g.d.capture)(g.s, 0))?;
    }
    body()?;
    unsafe {
        check((g.d.capture_end)(g.s, &mut raw))?;
        check((g.d.instantiate)(&mut exec, raw, 0))?;
        check((g.d.event)(&mut start, 0))?;
        check((g.d.event)(&mut end, 0))?;
    }
    Ok(Graph { g, raw, exec, start, end })
}
fn sample(graph: &Graph<'_>, times: u64) -> Result<(f64, f64)> {
    let g = graph.g;
    let mut ms = 0.;
    unsafe {
        check((g.d.event_record)(graph.start, g.s))?;
        check((g.d.graph_launch)(graph.exec, g.s))?;
        check((g.d.event_record)(graph.end, g.s))?;
        check((g.d.event_sync)(graph.end))?;
        check((g.d.event_elapsed)(&mut ms, graph.start, graph.end))?;
    }
    let bytes = g.read(times, 16)?;
    let a = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let b = u64::from_le_bytes(bytes[8..].try_into().unwrap());
    if ms <= 0. || b <= a {
        return Err("invalid device timestamps".into());
    }
    Ok((ms as f64 * 1000., (b - a) as f64 / 1000.))
}
fn output_hash(g: &Gpu, v: &Fixture, c: &Case) -> Result<String> {
    if matches!(c.kind,71..=73) { return fp8group::output_hash(g,v); }
    if prompt::hcup_a16(c) {let raw=g.read(v.out,v.out_bytes)?;return Ok(sha256::digest(&prompt::hc_a16_logical_rows(&raw,c.m,c.n)?));}
    if v.down_slots > 0 {
        let all = g.read(v.out, v.out_bytes)?;
        let mut written = Vec::with_capacity(v.down_slots / 11 * 10 * 5120);
        for (slot, row) in all.chunks_exact(10240).enumerate() {
            let used = if slot % 11 < 10 { 5120 } else { 0 };
            if row[used..].iter().any(|&b| b != 0xff) {
                return Err(format!("{} wrote outside its BF16 slot halves (slot {slot})", c.name));
            }
            if row[..used].chunks_exact(2).any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80) {
                return Err(format!("{} nonfinite or unwritten output (slot {slot})", c.name));
            }
            written.extend_from_slice(&row[..used]);
        }
        return Ok(sha256::digest(&written));
    }
    let mut bytes = g.read(v.out, v.out_bytes)?;
    if !v.written.is_empty() {
        bytes = v.written.iter().flat_map(|&(at, len)| bytes[at..at + len].to_vec()).collect();
    }
    if v.aux_bytes > 0 {
        bytes.extend(g.read(v.aux, v.aux_bytes)?);
    }
    // Both packed BF16 and FP32 outputs must be finite; poison catches unwritten rows.
    let finite = if let Some(q) = v.qsa.as_ref() {
        qsa::finite(q, &bytes)
    } else if v.gather.is_some() || matches!(c.kind, 23 | 24 | 26 | 27 | 28 | 29) {
        true
    } else if matches!(c.kind, 3 | 4 | 5 | 20 | 21 | 22) {
        bytes.chunks_exact(2).all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80)
    } else {
        bytes.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    };
    if !finite {
        return Err(format!("{} nonfinite or unwritten output", c.name));
    }
    Ok(sha256::digest(&bytes))
}
pub(super) fn run(o: &Options, cases: &[Case]) -> Result<()> {
    let started = Instant::now();
    let a = std::fs::read(&o.a).map_err(|e| e.to_string())?;
    let x = std::fs::read(&o.x).map_err(|e| e.to_string())?;
    let hashes = [sha256::digest(&a), sha256::digest(&x)];
    let source = sha256::digest(&[include_bytes!("kernel_bench.rs").as_slice(), include_bytes!("kernel_prompt.rs").as_slice(), include_bytes!("kernel_frontend.rs").as_slice(), include_bytes!("kernel_qsa.rs").as_slice(), include_bytes!("kernel_cuda.rs").as_slice(), include_bytes!("kernel_gdn.rs").as_slice(), include_bytes!("kernel_floor.rs").as_slice(),include_bytes!("kernel_routed.rs").as_slice(),include_bytes!("kernel_fp4_prompt.rs").as_slice(),include_bytes!("kernel_fp8.rs").as_slice()].concat());
    let mut g = Gpu::new()?;
    let modules = [g.module(&a).map_err(|e| format!("A PTX: {e}"))?, g.module(&x).map_err(|e| format!("X PTX: {e}"))?];
    let tm = g.module(TIMER.as_bytes()).map_err(|e| format!("timer PTX: {e}"))?;
    let timer = g.function(tm, "kb_stamp")?;
    flush_init(&mut g, flush_on() || cases.iter().any(|c| c.kind == 26))?;
    // Exactness law: the rows that exist must have identical bytes at every M (inputs of
    // M<=16 share one padded tile, so row r's operands are the same bytes at every M).
    let mut rows_seen: std::collections::HashMap<(usize, usize, u8), Vec<u8>> = Default::default();
    for c in cases {
        eprintln!("prepare {} elapsed_s={:.1}", c.name, started.elapsed().as_secs_f64());
        if c.kind==70 {frontend::run(&mut g,modules,c,o,&hashes,&source)?;continue;}
        if (40..=55).contains(&c.kind) {
            // kernel_gdn.rs numbers its cases 7..22.
            gdn::run_case(&mut g, modules, timer, &Case { kind: c.kind - 33, ..c.clone() }, o, &hashes, &source, started)?;
            continue;
        }
        let mut v = fixture(&mut g, c, o.order)?;
        head::prepare(&mut g, modules[0], c, &mut v)?;
        if c.kind == 75 && c.mats[3] & 2 != 0 { rows::twins(&mut g, modules[1], c, &mut v)?; }
        if c.kind == 75 && c.mats[3] & 4 != 0 { v.pack = g.function(modules[1], "flash_bf16_input")?; }
        if c.kind == 9 || c.kind == 10 || c.kind == 12 || c.kind == 77 || c.name.starts_with("hc_up_planes") {
            v.pack = g.function(modules[0], "flash_bf16_input")?;
        }
        let prompt_functions = if (60..=69).contains(&c.kind) || c.kind == 74 { prompt::prepare(&mut g, modules, c, &mut v)? } else { None };
        if c.kind == 35 || c.kind == 31 { qsa::chains(&g, c, modules[1], v.qsa.as_mut().ok_or("QSA fixture")?)?; }
        if c.kind == 36 { qsa::ssm_chain(&g, modules[1], v.qsa.as_mut().ok_or("QSA fixture")?)?; }
        let times = g.alloc(16)?;
        eprintln!("arena case={} base={} copies={} setup_epoch={:?}", c.name, v.ring[0], v.ring.len(), std::time::SystemTime::now());
        // KBENCH_XKERNEL overrides the X entry of QSA cases (the _tc2 screen in
        // order mode; a same-bytes prompt entry such as _k8 / _wide otherwise) and of the
        // routed fp4_bw gate cases (a same-bytes entry such as fp4_bw_gate_c1); KBENCH_XDOWN
        // overrides X's down entry in the down and layer cases (fp4_bw_down_pdl / _x2 / _x4).
        let xover = if c.kind == 75 && c.mats[3] & 4 != 0 { Some("flash_bf16_mma_latency_group".into()) } else if c.name.starts_with("hc_add_norm") { Some("flash_next_hc_add_norm_rows".into()) } else if c.name.starts_with("hc_down_act") { Some("flash_hc_down_latency_group".into()) } else if prompt::hcup_a16(c) {Some("flash_next_hc_up_tiled_a16".into())} else if c.name.starts_with("proute-") {std::env::var("KBENCH_ROUTE_PREFILL_X").ok()} else if c.name.starts_with("hc_up_prod-") && std::env::var("KBENCH_HCUP_M32").is_ok_and(|v|v=="1") {Some("flash_next_hc_up_m32".to_string())} else if c.name.starts_with("pgprod-") { std::env::var("KBENCH_PGPROD_X").ok().filter(|x|matches!(x.as_str(),"w16"|"u32"|"plan")).map(|x|format!("flash_bf16_mma_prefill_tiled_{x}")) } else if super::fold_x(c) { Some("flash_bf16_mma_latency_fold".to_string()) } else if (c.kind == 31 || c.kind == 32) && o.order && c.m < 64 && qsa::small_x().is_some() { qsa::small_x() }
            else if c.kind == 21 || c.kind == 22 { std::env::var("KBENCH_XDOWN").ok() } else if c.kind == 27 { Some("flash_next_compact_route_ids".into()) } else if c.kind == 29 || c.kind == 28 { Some(format!("{}{}", if c.kind == 29 { "flash_next_compact_route" } else { "flash_next_route_compact1" }, if c.name.contains("pack") && c.name.contains("mask") { "_pack_mask" } else if c.name.contains("pack") { "_pack" } else if c.name.contains("mask") { "_silu_mask" } else { "_silu" })) } else if c.kind == 24 && c.name.contains("fast") { Some(if c.name.contains("silu") { "flash_moe_silu_fast" } else { "flash_moe_route_fast" }.into()) }
            else { std::env::var("KBENCH_XKERNEL").ok().filter(|_| c.kind == 1 || (30..=32).contains(&c.kind) || c.kind == 20 || matches!(c.kind,71..=73) || matches!(c.kind, 75 | 76) || c.kind == 34 || c.kind >= 37 && c.kind <= 38) };
        let xname: &str = match &xover { Some(n) => n.as_str(), None if o.order && !c.xkernel.is_empty() => c.xkernel, None => c.kernel };
        let mut rows_entries = [String::new(), String::new()]; // KBENCH_MXROWS_X: printed as "entry"
        let functions = if let Some(f) = prompt_functions {
            f
        } else if c.kind == 33 {
            let m = g.module(qsa::gather_ptx().as_bytes()).map_err(|e| format!("gather PTX: {e}"))?;
            let f = g.function(m, "kb_gather")?;
            [f, f]
        } else if c.kind == 8 || c.kind == 10 || c.kind == 11 {
            [g.function(modules[0], "flash_bf16_mma_latency")?, g.function(modules[1], c.kernel)?]
        } else if c.kind == 13 && mxrows_x().is_some() {
            rows_entries = mxrows_entries(c);
            [g.function(modules[0], &rows_entries[0])?, g.function(modules[1], &rows_entries[1])?]
        } else if c.kind == 13 && (0..4).all(|j| c.mats[j] == 0 || group_mx(c, j)) {
            // Every matrix MXFP8: X runs the single-storage group entry when its PTX has one
            // (the engine's choice for such groups).
            let x = g.function(modules[1], if all_mx(c) && params::latency_mx8_widen(c.m, c.k) { "flash_mx8_mma_latency_r_group" } else { "flash_mx8_mma_latency_group" }).or_else(|_| g.function(modules[1], c.kernel))?;
            [g.function(modules[0], c.kernel)?, x]
        } else {
            [g.function(modules[0], &qsa::a_name(c))?, g.function(modules[1], xname)?]
        };
        let merges = if c.kind == 22 {
            // fp4_bw layer: each arm's gate runs before its down (KBENCH_XGATE = X's gate entry).
            let xgate = std::env::var("KBENCH_XGATE").unwrap_or_else(|_| "fp4_bw_gate".into());
            [g.function(modules[0], "fp4_bw_gate")?, g.function(modules[1], &xgate)?]
        } else if matches!(c.kind, 67 | 27 | 28 | 29) { // A arm's MoE output: main's k16_finish (X forms it in its kernel); route_ids: main's assignments.
            if c.name.contains("mask") { v.norms[0] = g.function(modules[0], "flash_next_route_mask")?; } if c.name.contains("pack") { v.pack = g.function(modules[0], "flash_bf16_input")?; } // A's mask / pack
            [g.function(modules[0], match c.kind { 27 => "flash_moe_assignments", 28 => "flash_moe_route", 29 => "flash_moe_silu", _ => "flash_next_k16_finish" })?, null_mut()]
        } else if c.kind == 62 || c.kind == 64 || c.kind == 66 {
            // A arm's second launch: main's pairs kernel on the raw input.
            let pairs = g.function(modules[0], "flash_bf16_mma_prefill_pairs")?;
            unsafe { check((g.d.func_attr)(pairs, 8, params::FLASH_BF16_MMA_SHARED as i32))? };
            [pairs, null_mut()]
        } else if c.kind == 8 || c.kind == 10 {
            [g.function(modules[0], "flash_bf16_mma_latency_merge")?, null_mut()]
        } else if (c.kind == 0 || c.kind == 9 || c.kind == 11 || c.kind == 12) && c.m <= 16 && c.n <= 640 {
            [g.function(modules[0], "flash_bf16_mma_latency_merge")?, if super::fold_x(c) { null_mut() } else { g.function(modules[1], "flash_bf16_mma_latency_merge")? }]
        } else {
            [null_mut(), null_mut()]
        };
        if c.kind == 13 && mxrows_x().is_some() { // As the engine's function_with_shared: each arm's launch bytes.
            for (arm, f) in functions.into_iter().enumerate() { unsafe { check((g.d.func_attr)(f, 8, mxrows_shared(c, arm) as i32))? }; }
        } else if c.kind == 11 || c.kind == 13 || super::fold_x(c) {
            // The MXFP8 entries' M5-8 CTA input copy (latency_mx8_shared) exceeds 48 KiB.
            unsafe {
                check((g.d.func_attr)(functions[1], 8, 101_376))?;
            }
        }
        // The arms' entries, then any helper launch above 48 KiB (prompt::extra_shared).
        for (f, bytes) in functions.into_iter().map(|f| (f, v.shared)).chain(prompt::extra_shared(c, &v)) {
            if bytes > 48 * 1024 {
                // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, as the engine sets it.
                unsafe {
                    check((g.d.func_attr)(f, 8, bytes as i32))?;
                }
            }
        }
        // Activation probe cases quantize A with the probe's pre-pass in the arm that has it.
        let probe = |m: H| if c.xkernel.contains("a4probe") { g.function(m, "fp4_prefill_tc_a4probe_quant").ok() } else { None };
        let pre = |m: H| if c.kind == 5 { probe(m).or_else(|| g.function(m, "fp4_prefill_tc_a16").ok()) } else { None };
        let mut pxs: [Pre; 2] = [(None, None, None); 2];
        for (a, (&m, f)) in modules.iter().zip(functions).enumerate() {
            let shared = g.module_u32(m, &format!("{}_SHARED", c.kernel.to_uppercase()));
            let entry_threads=g.module_u32(m, &format!("{}_THREADS", c.kernel.to_uppercase()));
            if let Some(bytes) = shared {
                // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
                unsafe { check((g.d.func_attr)(f, 8, bytes as i32))? };
            }
            pxs[a] = (pre(m), shared, entry_threads);
        }
        let mut digest = Vec::new();
        let mut outputs = Vec::new();
        for (arm, (f, merge)) in functions.into_iter().zip(merges).enumerate() {
            let mut all = Vec::new();
            for i in 0..v.ring.len() {
                unsafe {
                    check((g.d.memset)(v.out, 0xff, v.out_bytes))?;
                }
                eprintln!("verify case={} arm={} copy={} shared={}",c.name,arm,i,pxs[arm].1.unwrap_or(v.shared));
                launch(&g, f, merge, c, &mut v, i, arm, pxs[arm])?;
                g.sync()?;
                all.push(if c.name.starts_with("proute-") {bw_routed::prefill_hash(&g,c,&v,arm)?}else{output_hash(&g, &v, c)?});
            }
            // Main's softmax rewrites its input in place: only its timing counts.
            if all.iter().any(|h| h != &all[0]) && !(c.kind == 34 || c.kind == 38 && c.k == 4) {
                return Err(format!("{} rotating copies differ (arm {arm})", c.name));
            }
            digest.push(all[0].clone());
            if (30..=32).contains(&c.kind) || (c.kind == 35 && std::env::var("KBENCH_QSA_U32").is_ok_and(|v| v == "1")) { outputs.push(g.read(v.out, v.out_bytes)?); }
        }
        let mut m_rows = "\"-\"".to_string();
        if c.kind == 1 || ((c.kind == 0 || c.kind == 9) && c.m <= 16 && c.name.starts_with("lat")) || (c.kind == 12 || (60..=62).contains(&c.kind) || c.kind == 65) {
            // Arm X output of the last launch (copy ring.len()-1) is still in `out`.
            let raw = g.read(v.out, v.out_bytes)?;
            // HC's planes follow M-dependent FP32/hi/lo boundaries. Compare all live
            // FP32 and packed words in row order, never an incompatible buffer prefix.
            let bytes = if prompt::hcup_a16(c) {prompt::hc_a16_logical_rows(&raw,c.m,c.n)?} else if c.kind == 65 { prompt::hc_logical_rows(&raw, c.m, c.n)? } else { raw };
            let entry = rows_seen.entry((c.n, c.k, c.kind)).or_insert_with(|| bytes.clone());
            let common = entry.len().min(bytes.len());
            let same = entry[..common] == bytes[..common];
            if !same {
                // First differing FP32 output: token row and output column.
                let at = (0..common).find(|&i| entry[i] != bytes[i]).unwrap_or(0) / 4;
                let differ = (0..common / 4).filter(|&i| entry[i * 4..i * 4 + 4] != bytes[i * 4..i * 4 + 4]).count();
                eprintln!("ROWS MISMATCH {} across M (first {} bytes): first row {} col {}, {} of {} outputs differ", c.name, common, at / c.n, at % c.n, differ, common / 4);
            }
            if bytes.len() > entry.len() {
                *entry = bytes.clone();
            }
            m_rows = format!("{same}");
        }
        // Each owner's order-change switch (--det 1, --allow-diff, --order) lifts the A==X gate.
        if digest[0] != digest[1]
            && !o.det
            && !o.allow_diff
            && !o.order
        {
            return Err(format!("{} bytes A==X: no A={} X={}", c.name, digest[0], digest[1]));
        }
        // QSA score cases: FP64 error on sampled rows and alone == chunked bytes per arm.
        let mut extra = if matches!(c.kind,71..=73) {fp8group::metadata(c,&v,[0,1].map(|a|pxs[a].2.unwrap_or(v.threads)))} else {String::new()};
        if c.kind == 30 && qsa::prod_launch(c).is_none() {
            let host = v.qsa.as_ref().and_then(|q| q.host.as_ref()).ok_or("QSA host data")?;
            extra = format!(",{}", qsa::score_errors(host, qsa::MAXPOS, &outputs[0], &outputs[1]));
            for (arm, f) in functions.into_iter().enumerate() {
                let q = v.qsa.as_mut().unwrap();
                let (rows, bad) = qsa::chunk_check(&mut g, f, q, arm, o.order, &outputs[arm])?;
                extra += &format!(",\"chunk_rows_{}\":{rows},\"chunk_mismatch_{}\":{bad}", ["a", "x"][arm], ["a", "x"][arm]);
                if bad != 0 { eprintln!("CHUNK MISMATCH case={} arm={arm} rows={bad}/{rows}", c.name); }
            }
        }
        if c.kind == 32 {
            let q = v.qsa.as_ref().ok_or("QSA fixture")?;
            let host = q.ihost.as_ref().ok_or("QSA index host data")?;
            extra = format!(",{}", qsa::index_errors(host, &outputs[0], &outputs[1]));
            for (arm, f) in functions.into_iter().enumerate() {
                let small = match qsa::small_x() { Some(n) if arm == 1 => Some(g.function(modules[1], &n)?), _ => None };
                let q = v.qsa.as_mut().unwrap();
                let (rows, bad) = qsa::index_chunk_check(&mut g, f, small, q, arm, o.order, &outputs[arm])?;
                extra += &format!(",\"chunk_rows_{}\":{rows},\"chunk_mismatch_{}\":{bad}", ["a", "x"][arm], ["a", "x"][arm]);
                if bad != 0 { eprintln!("CHUNK MISMATCH case={} arm={arm} rows={bad}/{rows}", c.name); }
            }
        }
        if c.kind == 31 && qsa::prod_launch(c).is_none() {
            let q = v.qsa.as_ref().ok_or("QSA fixture")?;
            let host = q.vhost.as_ref().ok_or("QSA values host data")?;
            extra = format!(",{}", qsa::value_errors(host, qsa::MAXPOS, &outputs[0], &outputs[1]));
            for (arm, f) in functions.into_iter().enumerate() {
                let small = match qsa::small_x() { Some(n) if arm == 1 => Some(g.function(modules[1], &n)?), _ => None };
                let q = v.qsa.as_mut().unwrap();
                let (rows, bad) = qsa::values_chunk_check(&mut g, f, small, q, arm, o.order, &outputs[arm])?;
                extra += &format!(",\"chunk_rows_{}\":{rows},\"chunk_mismatch_{}\":{bad}", ["a", "x"][arm], ["a", "x"][arm]);
                if bad != 0 { eprintln!("CHUNK MISMATCH case={} arm={arm} rows={bad}/{rows}", c.name); }
            }
        }
        if c.kind == 35 && qsa::prod_launch(c).is_some() && std::env::var("KBENCH_QSA_U32").is_ok_and(|v| v == "1") {
            for (arm, f) in functions.into_iter().enumerate() {
                let q = v.qsa.as_mut().ok_or("QSA fixture")?;
                let (rows, bad) = qsa::fused_chunk_check(&mut g, f, q, &outputs[arm])?;
                extra += &format!(",\"chunk_rows_{}\":{rows},\"chunk_mismatch_{}\":{bad}", ["a", "x"][arm], ["a", "x"][arm]);
                if bad != 0 { return Err(format!("{} fused chunk mismatch arm={arm}: {bad}/{rows}", c.name)); }
            }
        }
        let ga = graph(&g, functions[0], merges[0], 0, pxs[0], timer, times, c, &mut v)?;
        let gx = graph(&g, functions[1], merges[1], 1, pxs[1], timer, times, c, &mut v)?;
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 250 {
            sample(&ga, times)?;
            sample(&gx, times)?;
        }
        let mut events = [vec![], vec![]];
        let mut global = [vec![], vec![]];
        let count = (v.ring.len() * v.chunks) as f64;
        while events[0].len() < std::env::var("KBENCH_MIN_SAMPLES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(9) || events.iter().any(|v| v.iter().sum::<f64>() < o.ms as f64 * 1000.) {
            let order = if events[0].len() % 2 == 0 { [0, 1] } else { [1, 0] };
            for arm in order {
                let (event, gt) = sample(if arm == 0 { &ga } else { &gx }, times)?;
                events[arm].push(event);
                global[arm].push(gt);
            }
        }
        for arm in 0..2 {
            eprintln!("samples case={} arm={} count={} event_us={:?}", c.name, arm, count, events[arm]);
            let e = events[arm].iter().map(|v| v / count).collect::<Vec<_>>();
            let t = global[arm].iter().map(|v| v / count).collect::<Vec<_>>();
            let mid = median(&e);
            let avg = e.iter().sum::<f64>() / e.len() as f64;
            let cv = (e.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / e.len() as f64).sqrt() / avg * 100.;
            let (stored, basis) = if matches!(c.kind,71..=73) {(v.stored,fp8group::basis())} else {super::floor::account(c, arm, v.stored)};
            let bw = stored as f64 / mid / 1000.;
            let floor = (stored as f64 / (std::env::var("KBENCH_READ_GBPS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(230.) * 1000.)).max(if prefill(c.kind) || (60..=69).contains(&c.kind) || c.kind == 74 { v.flops as f64 / 90e6 } else { 0. });
            let pct = floor / mid * 100.;
            println!(
                concat!(
                    "{{\"case\":{:?},\"kernel\":{:?},\"arm\":{:?},\"m\":{},\"n\":{},\"k\":{},",
                    "\"calls_per_token\":{},",
                    "\"samples\":{},\"launches_per_sample\":{},\"ring_copies\":{},\"rotating_weight_bytes\":{},\"stored_bytes\":{},\"floor_bytes_basis\":{:?},\"flops\":{},\"chunks_2048\":{},\"median_us\":{:.6},\"min_us\":{:.6},\"max_us\":{:.6},\"cv_pct\":{:.6},\"globaltimer_us\":{:.6},\"gbps\":{:.6},\"floor_us\":{:.6},\"pct_floor\":{:.3},\"grade\":{:?},\"sha256\":{:?},\"bytes_equal\":{},",
                    "\"m_rows_equal\":{},",
                    "\"order\":{},\"entry\":{:?},",
                    "\"ptx_sha256\":{:?},\"bench_source_sha256\":{:?}",
                    "{}",
                    "}}"
                ),
                c.name,
                c.kernel,
                if arm == 0 { "A" } else { "X" },
                c.m,
                c.n,
                c.k,
                c.calls,
                e.len(),
                count as usize,
                v.ring.len(),
                (if c.kind == 11 && arm == 1 { super::floor::mxfp8(c.n, c.k) } else { v.weight_bytes }) * v.ring.len(),
                stored,
                basis,
                v.flops,
                if prefill(c.kind) { 4 } else { 0 },
                mid,
                e.iter().copied().fold(f64::INFINITY, f64::min),
                e.iter().copied().fold(0., f64::max),
                cv,
                median(&t),
                bw,
                floor,
                pct,
                grade(pct),
                digest[arm],
                digest[0] == digest[1],
                m_rows,
                o.order,
                if !rows_entries[arm].is_empty() { rows_entries[arm].as_str() } else if arm == 0 { c.kernel } else { xname },
                hashes[arm],
                source,
                extra
            );
        }
        drop(ga);
        drop(gx);
        g.clear()?;
    }
    Ok(())
}
