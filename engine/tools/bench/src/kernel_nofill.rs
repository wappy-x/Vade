//! Device sweep for the fill-free prompt transposes (`prompt_writes`): the generic `prompt_rows` kernel and `transpose_tiles` run
//! on an output poisoned with 0xff and must write every word, byte-equal to the host reference. Safe code only.
#![forbid(unsafe_code)]
use super::super::{Gpu, H, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)); op Transpose is 2.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
struct Buffer { base: u64, address: u64, bytes: usize }
impl Buffer {
    fn new(g: &mut Gpu, data: &[u8]) -> Result<Self> { let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?; Ok(Self { base, address: base + 16, bytes: data.len() }) }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }
    fn poison(&self, g: &Gpu) -> Result<()> { g.upload_at(self.address, &vec![0xff; self.bytes]) }
}
/// Empty input slots hold the aligned non-null address 4, as backend/prompt_ops.rs passes: the kernel builds zero-length slices from them.
fn args(input: &Buffer, output: &Buffer, rows: u32, cols: u32, units: u64) -> Arguments {
    let inner = Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    Arguments { input: [input.address, 4, 4, 4, 4], lengths: [(rows as u64) * cols as u64, 0, 0, 0, 0], weight: 4, weights: 0, output: output.address, units, words: 1,
        shape: PShape { op: 2, tokens: 1, heads: 0, dim: 0, taps: 0, rows, cols, first: 0, slots: 0, inner } }
}
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    let _ = base_path;
    let b = std::fs::read(candidate_path).map_err(|e| e.to_string())?;
    println!("NOFILL candidate_sha256={}", sha256::digest(&b));
    let mut g = Gpu::new()?; let mb = g.module(&b)?;
    let generic: H = g.function(mb, "prompt_rows")?; let tiles: H = g.function(mb, "transpose_tiles")?;
    let started = Instant::now(); let (mut checks, mut tile_checks, mut bytes) = (0u64, 0u64, 0u64);
    for (r, k) in crate::prompt_writes::transpose_sweep() {
        let n = (r as usize) * k as usize;
        let x: Vec<f32> = (0..n).map(|i| f32::from_bits(0x3f80_0000 ^ (i as u32).wrapping_mul(2654435761))).collect();
        let expected: Vec<f32> = (0..n).map(|index| x[(index % r as usize) * k as usize + index / r as usize]).collect();
        let input = Buffer::new(&mut g, &le(&x))?; let want = le(&expected);
        let out = Buffer::new(&mut g, &vec![0xffu8; n * 4])?;
        // The generic kernel: one thread a word.
        out.poison(&g)?; let mut a = args(&input, &out, r, k, n as u64);
        g.launch(generic, [(n.div_ceil(128)) as u32, 1, 1], 128, 0, &mut [ptr(&mut a)])?; g.sync()?;
        let y = out.read(&g)?;
        if y[16..16 + n * 4] != want[..] { let at = y[16..16 + n * 4].iter().zip(&want).position(|(p, q)| p != q); return Err(format!("NOFILL BYTE FAIL generic rows={r} cols={k} first_byte={at:?}")); }
        if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("NOFILL guard changed generic rows={r} cols={k}")); }
        checks += 1; bytes += (n * 4) as u64;
        // The tile kernel where the host takes it: whole 4 x 4 tiles.
        if r % 4 == 0 && k % 4 == 0 {
            out.poison(&g)?; let mut a = args(&input, &out, r, k, (n / 16) as u64);
            g.launch(tiles, [((n / 16).div_ceil(128)) as u32, 1, 1], 128, 0, &mut [ptr(&mut a)])?; g.sync()?;
            let y = out.read(&g)?;
            if y[16..16 + n * 4] != want[..] { let at = y[16..16 + n * 4].iter().zip(&want).position(|(p, q)| p != q); return Err(format!("NOFILL BYTE FAIL tiles rows={r} cols={k} first_byte={at:?}")); }
            if y[..16] != GUARD || y[y.len() - 16..] != GUARD { return Err(format!("NOFILL guard changed tiles rows={r} cols={k}")); }
            tile_checks += 1;
        }
        g.clear()?;
    }
    println!("NOFILL BYTES PASS generic_checks={checks} tile_checks={tile_checks} output_bytes={bytes} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}
