//! Device byte checks through the existing safe benchmark CUDA wrapper.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, ptr, sha256};
#[path = "kernel_nvbatch.rs"] pub(super) mod nvbatch;
#[path = "kernel_rowstile.rs"] pub(super) mod rowstile;
#[path = "kernel_scorestile.rs"] pub(super) mod scorestile;
#[path = "kernel_transp.rs"] pub(super) mod transp;
#[allow(dead_code)] #[path = "../../../crates/gpu/src/small_sweep.rs"] pub(super) mod envelope;
#[path = "kernel_rowsmall.rs"] pub(super) mod rowsmall;
#[path = "kernel_nofill.rs"] pub(super) mod nofill;
#[path = "kernel_nvsmall.rs"] pub(super) mod nvsmall;
#[path = "kernel_fp8small.rs"] pub(super) mod fp8small;
#[path = "kernel_mxsmall.rs"] pub(super) mod mxsmall;
#[path = "kernel_fp8tile.rs"] pub(super) mod fp8tile;
#[path = "kernel_mxrows.rs"] pub(super) mod mxrows;
#[path = "kernel_fp8rows.rs"] pub(super) mod fp8rows;
#[path = "kernel_attnsweep.rs"] pub(super) mod attnsweep;
use crate::{mhc_params::*, mhc_reference::{Fixture, Output, bytes, shapes}};
#[path = "kernel_mhcproj.rs"] pub(super) mod project;
#[allow(clippy::needless_range_loop)]
#[path = "kernel_mxwave.rs"] pub(super) mod mxwave;
#[path = "kernel_nvrows.rs"] pub(super) mod nvrows;
const GUARD: [u8; 16] = [0x5a; 16];
struct Buffer { base: u64, address: u64, words: usize }
impl Buffer {
    fn new(g: &mut Gpu, values: &[f32]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), &bytes(values), GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, words: values.len() })
    }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> {
        let data = g.read(self.base, self.words * 4 + 32)?;
        if data[..16] != GUARD || data[data.len() - 16..] != GUARD { return Err("mHC buffer guard changed".into()); }
        Ok(data[16..data.len() - 16].to_vec())
    }
    fn poison(&self, g: &Gpu) -> Result<()> { g.fill_at(self.address, self.words * 4, 2) }
    fn offset(&self, row: usize, stride: usize) -> u64 { self.address + (row * stride * 4) as u64 }
}
fn equal(label: &str, got: &[u8], expected: &[u8]) -> Result<()> {
    if got != expected {
        return Err(format!("mHC {label}: byte mismatch at {:?}; got={} expected={}",
            got.iter().zip(expected).position(|(a, b)| a != b), sha256::digest(got), sha256::digest(expected)));
    }
    Ok(())
}
struct Cell {
    input: Buffer, weights: Buffer, base: Buffer, scale: Buffer, block: Buffer,
    normalized: Buffer, coefficients: Buffer, collapsed: Buffer, combined: Buffer, head: Buffer,
}
impl Cell {
    fn new(g: &mut Gpu, fixture: &Fixture, out: &Output) -> Result<Self> {
        let blank = |n| vec![f32::NAN; n];
        Ok(Self { input: Buffer::new(g, &fixture.input)?, weights: Buffer::new(g, &fixture.weights)?,
            base: Buffer::new(g, &fixture.base)?, scale: Buffer::new(g, &fixture.scale)?, block: Buffer::new(g, &fixture.block)?,
            normalized: Buffer::new(g, &blank(out.normalized.len()))?, coefficients: Buffer::new(g, &blank(out.coefficients.len()))?,
            collapsed: Buffer::new(g, &blank(out.collapsed.len()))?, combined: Buffer::new(g, &blank(out.combined.len()))?,
            head: Buffer::new(g, &blank(out.head.len()))? })
    }
    fn outputs(&self) -> [&Buffer; 5] { [&self.normalized, &self.coefficients, &self.collapsed, &self.combined, &self.head] }
    fn span(&self, g: &Gpu, functions: &[H; 4], mut shape: Shape, first: usize, rows: usize, threads: u32) -> Result<()> {
        if rows == 0 { return Ok(()); }
        shape.rows = rows as u32; shape.validate().map_err(str::to_string)?;
        let (w, c, h) = (shape.wide(), shape.coefficients(), shape.hidden as usize);
        let mut prepare = Prepare { input: self.input.offset(first, w), weights: self.weights.address, base: self.base.address,
            scale: self.scale.address, normalized: self.normalized.offset(first, w), coefficients: self.coefficients.offset(first, c),
            collapsed: self.collapsed.offset(first, h), shape };
        let mut collapse = Collapse { input: prepare.input, coefficients: prepare.coefficients, output: self.collapsed.offset(first, h),
            rows: shape.rows, streams: shape.streams, hidden: shape.hidden };
        let mut combine = Combine { input: prepare.input, coefficients: prepare.coefficients, block: self.block.offset(first, h),
            output: self.combined.offset(first, w), rows: shape.rows, streams: shape.streams, hidden: shape.hidden };
        let mut head = Head { input: combine.output, output: self.head.offset(first, h), rows: shape.rows, streams: shape.streams, hidden: shape.hidden };
        // Overlaunch by one block. Each kernel must retain both guards.
        g.launch(functions[0], [shape.rows+1, 1, 1], shape.prepare_threads(), 0, &mut [ptr(&mut prepare)])?;
        for (function, words, argument) in [(functions[1], rows * h, ptr(&mut collapse)),
            (functions[2], rows * w, ptr(&mut combine)), (functions[3], rows * h, ptr(&mut head))] {
            g.launch(function, [words.div_ceil(threads as usize) as u32 + 1, 1, 1], threads, 0, &mut [argument])?;
        }
        Ok(())
    }
    fn check(&self, g: &Gpu, expected: &Output) -> Result<String> {
        let mut joined = Vec::new();
        for (buffer, values) in self.outputs().into_iter().zip(expected.buffers()) {
            let actual = buffer.read(g)?; equal("output", &actual, &bytes(values))?; joined.extend(actual);
        }
        Ok(sha256::digest(&joined))
    }
}
pub(crate) fn run(path: &str) -> Result<()> {
    let ptx = std::fs::read(path).map_err(|e| e.to_string())?;
    println!("MHC KIT ptx_sha256={} device=CUDA-ordinal-0", sha256::digest(&ptx));
    let mut g = Gpu::new()?; let module = g.module(&ptx)?;
    let functions = [g.function(module, "mhc_prepare")?, g.function(module, "mhc_collapse")?,
        g.function(module, "mhc_combine")?, g.function(module, "mhc_head")?];
    for shape in shapes() {
        let fixture = Fixture::new(shape); let out = fixture.reference(); let peak = fixture.source_check(&out)?;
        let cell = Cell::new(&mut g, &fixture, &out)?; let t = shape.rows as usize;
        let mut digest = String::new();
        for threads in [32, 128, 256] {
            for cut in 0..=t {
                for buffer in cell.outputs() { buffer.poison(&g)?; }
                // Reverse the span order to expose cross-row scratch reuse.
                cell.span(&g, &functions, shape, cut, t - cut, threads)?;
                cell.span(&g, &functions, shape, 0, cut, threads)?;
                g.sync()?; digest = cell.check(&g, &out)?;
            }
            for buffer in cell.outputs() { buffer.poison(&g)?; }
            for row in (0..t).rev() { cell.span(&g, &functions, shape, row, 1, threads)?; }
            g.sync()?; cell.check(&g, &out)?;
        }
        for (buffer, expected) in [(&cell.input, &fixture.input), (&cell.weights, &fixture.weights),
            (&cell.base, &fixture.base), (&cell.scale, &fixture.scale), (&cell.block, &fixture.block)] {
            equal("input mutation", &buffer.read(&g)?, &bytes(expected))?;
        }
        println!("MHC BYTES N={} D={} rows={} iterations={} cuts={} thread_counts=3 sha256={digest}",
            shape.streams, shape.hidden, shape.rows, shape.iterations, t + 1);
        println!("MHC NUMERIC independent-source max_abs={peak} tolerance=0.000003*max(1,abs(expected))");
    }
    println!("MHC KIT PASS; logical FP32 primitives only; no token-ID or BF16-boundary gate");
    Ok(())
}
