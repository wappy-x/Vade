//! Compare the existing recurrent row pipeline, the same pipeline without
//! clears, and one fused launch. Verify all the public and kept words first.
#![forbid(unsafe_code)]
use super::{Arguments, Buffer, GUARD, Gpu, H, Inner, PShape, Result, median, pattern, ptr, sha256};
use super::super::super::{capture_graph, sample};

#[path = "../../../crates/gpu/src/recurrent_rows.rs"]
mod abi;
use abi::{DIM, Shape, TAPS, THREADS};

const WRAPPED: u32 = 0;
const CONV: u32 = 4;
const HISTORY: u32 = 5;
const DECAY: u32 = 6;
const STEP: u32 = 7;
const PREPARE: u32 = 1;
const NORM: u32 = 3;

struct Launches {
    stage: [H; 6],
    zero: H,
}

impl Launches {
    fn load(g: &Gpu, module: H) -> Result<Self> {
        Ok(Self {
            stage: [g.function(module, "prompt_conv")?, g.function(module, "prompt_rows")?,
                g.function(module, "prompt_prepare")?, g.function(module, "prompt_prepare")?,
                g.function(module, "prompt_rows_small")?, g.function(module, "prompt_gated_norm")?],
            zero: g.function(module, "resident_launch_zero_words")?,
        })
    }
}

struct Fixture {
    args: [Arguments; 6],
    stage: [Buffer; 6],
    clear: u64,
    fused: abi::Arguments,
    fused_output: [Buffer; 5],
    fused_bare: abi::Arguments,
    bare_output: [Buffer; 3],
    replay_table: u64,
}

fn guarded(g: &mut Gpu, words: usize) -> Result<Buffer> {
    Buffer::new(g, &vec![f32::NAN; words])
}

fn fixture(g: &mut Gpu, shape: Shape, kind: u32, seed: u64) -> Result<Fixture> {
    let (inputs, weights, [history, state, rows, prepared, decay]) =
        shape.lengths(true).ok_or("recurrent bench shape is outside admission")?;
    let (m, h) = (shape.tokens as usize, shape.heads as usize);
    let width = h * DIM;
    let data = |g: &mut Gpu, len: usize, index: u64| {
        // Keep weights and normal state finite. Kind 2 includes signed zeros
        // and denormals. Each input has a different phase in the pattern.
        let src = pattern(kind, len, seed + 7919 * (index + 1));
        g.upload(&super::le(&src))
    };
    let mut input = [0; 7];
    let mut weight = [0; 3];
    for (i, (at, len)) in input.iter_mut().zip(inputs).enumerate() { *at = data(g, len, i as u64)?; }
    for (i, (at, len)) in weight.iter_mut().zip(weights).enumerate() { *at = data(g, len, 7 + i as u64)?; }
    let gate = data(g, rows, 10)?;
    let stage = [guarded(g, 3 * rows)?, guarded(g, history)?, guarded(g, prepared)?,
        guarded(g, decay)?, guarded(g, state + rows)?, guarded(g, rows)?];
    let out = stage.each_ref().map(|b| b.address);
    let outer = PShape { op: CONV, tokens: shape.tokens, heads: shape.heads, dim: DIM as u32, taps: TAPS as u32,
        rows: 0, cols: 0, first: 0, slots: 0,
        inner: Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 } };
    let inner = Inner { op: PREPARE, heads: shape.heads, dim: DIM as u32, taps: TAPS as u32, rows: 0,
        cols: 0, epsilon: shape.epsilon, bound: shape.bound };
    let with = |shape, input, lengths: [usize; 5], weight, weights, output, units, words| Arguments {
        shape, input, lengths: lengths.map(|n| n as u64), weight, weights: weights as u64,
        output, units: units as u64, words: words as u64,
    };
    let parts = [input[0], input[1], input[2], input[3], 4];
    let lengths = [rows, rows, rows, history, 0];
    let args = [
        with(outer, parts, lengths, weight[0], weights[0], out[0], 3 * rows, 1),
        with(PShape { op: HISTORY, ..outer }, parts, lengths, 4, 0, out[1], history, 1),
        with(PShape { op: WRAPPED, inner, ..outer }, [out[0], input[4], input[5], 4, 4],
            [3 * rows, rows, m * h, 0, 0], weight[1], weights[1], out[2], m * h, 4 * DIM + 1),
        with(PShape { op: DECAY, ..outer }, [out[2], 4, 4, 4, 4], [prepared, 0, 0, 0, 0], 4, 0, out[3], decay, 1),
        with(PShape { op: STEP, ..outer }, [out[2], input[6], out[3], 4, 4], [prepared, state, decay, 0, 0],
            4, 0, out[4], width, DIM + m),
        with(PShape { op: WRAPPED, inner: Inner { op: NORM, bound: 0.0, ..inner }, ..outer },
            [out[4] + (state * 4) as u64, gate, 4, 4, 4], [rows, rows, 0, 0, 0], weight[2], weights[2], out[5], m * h, DIM),
    ];

    // The same checked command that the resident launcher uses for the two clears.
    let mut clears = Vec::new();
    for (address, words) in [(out[1], history), (out[4], state + rows)] {
        let mut command = [0u8; 1040];
        command[..4].copy_from_slice(&16u32.to_le_bytes());
        command[24..32].copy_from_slice(&address.to_le_bytes());
        command[32..40].copy_from_slice(&(words as u64).to_le_bytes());
        clears.extend(command);
    }
    let clear = g.upload(&clears)?;
    let fused_output = [guarded(g, history)?, guarded(g, state)?, guarded(g, rows)?, guarded(g, prepared)?, guarded(g, decay)?];
    let fused = abi::Arguments { shape, inputs: input, gate, weights: weight, outputs: fused_output.each_ref().map(|v| v.address) };
    let bare_output = [guarded(g, history)?, guarded(g, state)?, guarded(g, rows)?];
    let fused_bare = abi::Arguments { outputs: [bare_output[0].address, bare_output[1].address, bare_output[2].address, 0, 0], ..fused };
    let mut table = Vec::new();
    for args in [fused, fused_bare] {
        let mut command = [0u8; 1040];
        command[..4].copy_from_slice(&abi::RECORD_KIND.to_ne_bytes());
        let bytes = args.payload();
        command[16..16 + bytes.len()].copy_from_slice(&bytes);
        table.extend(command);
    }
    let replay_table = g.upload(&table)?;
    Ok(Fixture { args, stage, clear, fused, fused_output, fused_bare, bare_output, replay_table })
}

fn pipeline(g: &Gpu, kernels: &Launches, f: &Fixture, fills: bool) -> Result<()> {
    let s = f.fused.shape;
    let width = s.heads * DIM as u32;
    let rows = s.tokens * width;
    let warps = (s.tokens * s.heads).div_ceil(4);
    let geometry = [(3 * width / 128, 128), ((f.args[1].units as u32).div_ceil(128), 128),
        (warps, 128), (rows.div_ceil(512), 128), (s.heads * 4, 32), (warps, 128)];
    for (i, (blocks, threads)) in geometry.into_iter().enumerate() {
        if fills && (i == 1 || i == 4) {
            let mut table = f.clear;
            let mut index = u64::from(i == 4);
            let words = f.args[i].units * f.args[i].words;
            g.launch(kernels.zero, [(words.div_ceil(128)) as u32, 1, 1], 128, 0, &mut [ptr(&mut table), ptr(&mut index)])?;
        }
        let mut args = f.args[i];
        g.launch(kernels.stage[i], [blocks, 1, 1], threads, 0, &mut [ptr(&mut args)])?;
    }
    Ok(())
}

fn fused(g: &Gpu, kernel: H, mut args: abi::Arguments) -> Result<()> {
    g.launch(kernel, [args.shape.heads, 1, 1], THREADS, 0, &mut [ptr(&mut args)])
}

fn replay(g: &Gpu, kernel: H, f: &Fixture, keep: bool) -> Result<()> {
    let mut table = f.replay_table;
    let mut index = u64::from(!keep);
    g.launch(kernel, [f.fused.shape.heads, 1, 1], THREADS, 0, &mut [ptr(&mut table), ptr(&mut index)])
}

fn body(g: &Gpu, buffer: &Buffer) -> Result<Vec<u8>> {
    let bytes = buffer.read(g)?;
    if bytes[..16] != GUARD || bytes[bytes.len() - 16..] != GUARD { return Err("RECURRENT buffer guard changed".into()); }
    Ok(bytes[16..bytes.len() - 16].to_vec())
}

fn stage_outputs(g: &Gpu, f: &Fixture) -> Result<[Vec<u8>; 5]> {
    let mut state = body(g, &f.stage[4])?;
    state.truncate(f.fused.shape.heads as usize * DIM * DIM * 4);
    Ok([body(g, &f.stage[1])?, state, body(g, &f.stage[5])?, body(g, &f.stage[2])?, body(g, &f.stage[3])?])
}

fn same(got: &[u8], want: &[u8], s: Shape, kind: u32, arm: &str, output: usize) -> Result<usize> {
    if got.len() != want.len() { return Err("RECURRENT extent mismatch".into()); }
    if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
        let start = at / 4 * 4;
        return Err(format!("RECURRENT BYTE FAIL arm={arm} heads={} tokens={} pattern={kind} output={output} word={} old={:02x?} new={:02x?}",
            s.heads, s.tokens, at / 4, &want[start..start + 4], &got[start..start + 4]));
    }
    Ok(want.len())
}

pub(super) fn run(base: &str, candidate: &str) -> Result<()> {
    let a = std::fs::read(base).map_err(|e| e.to_string())?;
    let b = std::fs::read(candidate).map_err(|e| e.to_string())?;
    println!("RECURRENT baseline_sha256={} candidate_sha256={}", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?;
    let ma = g.module(&a)?;
    let mb = g.module(&b)?;
    let baseline = Launches::load(&g, ma)?;
    let no_fill = Launches::load(&g, mb)?;
    let candidate = g.function(mb, "recurrent_rows")?;
    let recorded = g.function(mb, "resident_launch_recurrent_rows")?;

    let mut comparisons = 0;
    let mut bytes = 0;
    for heads in [32, 64] {
        for tokens in 2..=8 {
            for (kind, bound, epsilon) in [(0, -5.0, 1e-6), (2, -8.0, 1e-5), (5, -1.0, 1e-4)] {
                let s = Shape { heads, tokens, bound, epsilon };
                let f = fixture(&mut g, s, kind, 701)?;
                for o in f.stage.iter().chain(&f.fused_output).chain(&f.bare_output) { o.poison(&g)?; }
                pipeline(&g, &baseline, &f, true)?;
                g.sync()?;
                let want = stage_outputs(&g, &f)?;
                for o in &f.stage { o.poison(&g)?; }
                pipeline(&g, &no_fill, &f, false)?;
                fused(&g, candidate, f.fused)?;
                fused(&g, candidate, f.fused_bare)?;
                g.sync()?;
                let no_fill_words = stage_outputs(&g, &f)?;
                for (i, expected) in want.iter().enumerate() {
                    bytes += same(&no_fill_words[i], expected, s, kind, "no_fill", i)?;
                    bytes += same(&body(&g, &f.fused_output[i])?, expected, s, kind, "fused_keep", i)?;
                    comparisons += 2;
                    if let Some(output) = f.bare_output.get(i) {
                        bytes += same(&body(&g, output)?, expected, s, kind, "fused_bare", i)?;
                        comparisons += 1;
                    }
                }
                for o in f.fused_output.iter().chain(&f.bare_output) { o.poison(&g)?; }
                replay(&g, recorded, &f, true)?;
                replay(&g, recorded, &f, false)?;
                g.sync()?;
                for (i, expected) in want.iter().enumerate() {
                    bytes += same(&body(&g, &f.fused_output[i])?, expected, s, kind, "recorded_keep", i)?;
                    comparisons += 1;
                    if let Some(output) = f.bare_output.get(i) {
                        bytes += same(&body(&g, output)?, expected, s, kind, "recorded_bare", i)?;
                        comparisons += 1;
                    }
                }
                g.clear()?;
            }
        }
    }
    println!("RECURRENT BYTES PASS comparisons={comparisons} output_bytes={bytes}");

    // Rotate twelve layers. Each arm uses the same inputs in the same order.
    // The capture is outside the timed interval. Warm each graph, then rotate
    // the arm order before reporting every sample and its arm median.
    const LAYERS: usize = 12;
    const REPEATS: usize = 9;
    for heads in [32, 64] {
        for tokens in [2, 4, 8] {
            let s = Shape { heads, tokens, bound: -5.0, epsilon: 1e-6 };
            let ring = (0..LAYERS).map(|i| fixture(&mut g, s, 0, 17 + i as u64 * 101)).collect::<Result<Vec<_>>>()?;
            {
                let launch = |arm| {
                    capture_graph(&g, || {
                        for f in &ring {
                            match arm {
                                0 => pipeline(&g, &baseline, f, true)?,
                                1 => pipeline(&g, &no_fill, f, false)?,
                                2 => fused(&g, candidate, f.fused)?,
                                _ => replay(&g, recorded, f, true)?,
                            }
                        }
                        Ok(())
                    })
                };
                let graphs = [launch(0)?, launch(1)?, launch(2)?, launch(3)?];
                let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for graph in &graphs { sample(graph, 0)?; }
                for repeat in 0..REPEATS {
                    for arm in (0..graphs.len()).map(|i| (i + repeat) % graphs.len()) {
                        let us = sample(&graphs[arm], 0)?.0 / LAYERS as f64;
                        samples[arm].push(us);
                        println!("RECURRENT SAMPLE heads={heads} tokens={tokens} repeat={repeat} arm={arm} us={us:.4}");
                    }
                }
                let [old, clear_free, direct, replay] = samples.map(median);
                println!("RECURRENT TIME heads={heads} tokens={tokens} old_us={old:.4} no_fill_us={clear_free:.4} fused_us={direct:.4} recorded_us={replay:.4}");
            }
            g.clear()?;
        }
    }
    Ok(())
}
