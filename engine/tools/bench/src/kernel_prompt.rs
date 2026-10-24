//! prefill_gemm owner cases (kinds 60..68): production prompt GEMM fixtures and launches,
//! split from kernel_cuda.rs (mechanical; same code).
use super::*;

pub(super) fn hcup_a16(c: &Case) -> bool {
    c.name.starts_with("hc_up_prod-") && std::env::var("KBENCH_HCUP_A16").is_ok_and(|v|v=="1")
}
/// kind 63: BF16 mma.sync m16n8k16 issue rate, registers only. Each warp runs `iters` rounds
/// of eight independent accumulator chains; one f32 per thread is stored (finite, repeatable).
fn mma_peak_ptx() -> String {
    let mut s = String::from(".version 8.8\n.target sm_121\n.address_size 64\n");
    s += ".visible .entry kb_mma(.param .u64 out, .param .u32 iters)\n{\n.reg .pred p;\n.reg .b32 a<4>, b<2>, i, n, t;\n.reg .f32 c<32>, x;\n.reg .b64 o, w;\n";
    s += "ld.param.u64 o,[out]; ld.param.u32 n,[iters];\n";
    s += "mov.b32 a0,0x3c003c00; mov.b32 a1,0x3c003c00; mov.b32 a2,0x3c003c00; mov.b32 a3,0x3c003c00; mov.b32 b0,0x3c003c00; mov.b32 b1,0x3c003c00;\n";
    s += "MOVS\n";
    s += "mov.u32 i,0;\nLOOP:\n";
    s += "MMAS\n";
    s += "add.u32 i,i,1; setp.lt.u32 p,i,n; @p bra LOOP;\n";
    s += "mov.f32 x,0f00000000;\nADDS\n";
    s += "mov.u32 t,%ctaid.x; mov.u32 i,%ntid.x; mov.u32 n,%tid.x; mad.lo.u32 t,t,i,n; mul.wide.u32 w,t,4; add.u64 o,o,w; st.global.f32 [o],x; ret;\n}\n";
    s.replace("MOVS", &(0..32).map(|r| format!("mov.f32 c{r},0f00000000;")).collect::<String>())
        .replace("MMAS", &(0..8).map(|j| format!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{c{0},c{1},c{2},c{3}}},{{a0,a1,a2,a3}},{{b0,b1}},{{c{0},c{1},c{2},c{3}}};\n", 4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3)).collect::<String>())
        .replace("ADDS", &(0..32).map(|r| format!("add.f32 x,x,c{r};")).collect::<String>())
}
/// kind 60 (prefill_gemm owner): the production flash_bf16_mma_prefill_tiled launch on the
/// production pack (flash_bf16_input, 32 lanes) of a raw FP32 [M][K] activation whose row r
/// depends on r only; run() packs once, so the timed graph holds only the tiled launches.
/// kind 61: the production flash_bf16_mma_prefill_pairs launch on the raw activation; c.mats
/// gives pairs[0].{rows,rows2}, pairs[1].{rows,rows2}; outputs share one [M][sum] buffer.
pub(super) fn prompt_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, n, k) = (c.m, c.n, c.k);
    let weight_bytes = n * k * 2;
    let production = c.name.starts_with("pgprod-") || c.name.starts_with("hc_up_prod-");
    let ring = if production {
        let w:Vec<u8> = (0..n*k).flat_map(|i| {
            let x = (((i*7+i/64*11)%251) as f32-125.)/1024.;
            ((x.to_bits()>>16) as u16).to_le_bytes()
        }).collect();
        vec![g.upload(&w)?,g.upload(&w)?]
    } else {
        let slab = g.fill(weight_bytes * 2, 0)?;
        vec![slab,slab+weight_bytes as u64]
    };
    let raw: Vec<u8> = (0..m * k)
        .flat_map(|i| (if production { (((i/k*37+i%k*11)%523) as f32-261.)/127.1 } else { (((i/k)*131+(i%k)*7)%257) as f32/64.-2. }).to_le_bytes())
        .collect();
    let raw = g.upload(&raw)?;
    let plane = m.div_ceil(16) * k.div_ceil(16) * 128;
    let out_bytes = m * n * 4;
    let out = g.alloc(out_bytes)?;
    let tiled = c.kind != 61;
    let input = if tiled { g.fill(plane * 2 * 4, 0)? } else { raw };
    let mut rows = params::Bf16Rows {
        weight: ring[0], output: out, rows: n as u32, stride: n as u32, weight2: 0, output2: 0, rows2: 0, stride2: 0,
        input, cols: k as u32, tokens: m as u32, input_bf16: u32::from(tiled), hc_silu: 0, input_plane_words: plane as u64,
    };
    if c.name.starts_with("pgprod-") {
        rows.rows=c.mats[0] as u32;
        rows.rows2=c.mats[1] as u32;
        rows.stride2=n as u32;
        rows.output2=out+(c.mats[0]*4) as u64;
        rows.weight2=ring[0]+(c.mats[0]*k*2) as u64;
    }
    if c.kind == 62 {
        (rows.rows, rows.stride) = (c.mats[0] as u32, n as u32);
    }
    let (grid, shared) = if tiled {
        ([m.div_ceil(64) as u32, n.div_ceil(128) as u32, 1], params::FLASH_BF16_PREFILL_TILED_SHARED as u32)
    } else {
        let p = prompt_pairs(c, &rows, ring[0]);
        rows = p.pairs[0];
        let blocks: usize = p.pairs.iter().map(|&q| params::flash_bf16_prefill_blocks(q)).sum();
        ([blocks as u32, 1, 1], params::FLASH_BF16_MMA_SHARED as u32)
    };
    Ok(Fixture {
        rows, group: None, ring, out, out_bytes, aux: 0, aux_bytes: 0, grid, threads: 256, shared,
        // The hi and lo chains are two BF16 MMA streams: 4MNK, as the card's floor counts them.
        stored: weight_bytes + m * k * 4 + out_bytes, flops: 4 * m * n * k, chunks: 1, down_slots: 0, raw,
        pack: null_mut(), norms: [null_mut(); 2], weight_bytes, written: Vec::new(), ring_x: vec![], scales_x: vec![], qsa: None, gather: None,
    })
}
/// KBENCH_TILED_CTAS=G: the X arm launches the tiled kernel persistent, G CTAs in x
/// (at most one per tile), as the engine does with its multiprocessor count.
/// kind 64 (prefill_gemm owner): the HC prompt block on a raw FP32 residual [M][10240]:
/// flash_hc_norm_f32 (F32 weights) -> HC down (320) + injection (4) projections. A = main's
/// norm (packed 0) + flash_bf16_input + tiled down + pairs injection; X = the norm writing
/// the packed planes + one tiled launch with the injection folded. Hash: normalized
/// streams u [M][10240] then the projections [M][324] (one buffer).
fn hc_norm_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, k) = (c.m, 10240);
    let mut base = prompt_fixture(g, &Case { kind: 62, k, ..c.clone() })?;
    let residual: Vec<u8> = (0..m * k)
        .flat_map(|i| ((((i / k) * 37 + (i % k) * 11) % 263) as f32 / 32. - 4.).to_le_bytes())
        .collect();
    let norm: Vec<u8> = (0..k).flat_map(|i| (((i * 7) % 61) as f32 / 256. - 0.1).to_le_bytes()).collect();
    let (u_bytes, out_bytes) = (m * k * 4, m * c.n * 4);
    let all = g.alloc(u_bytes + out_bytes)?;
    // raw = residual input of the norm; aux = F32 norm weights; out = [u | projections].
    base.raw = g.upload(&residual)?;
    base.aux = g.upload(&norm)?;
    base.out = all;
    base.out_bytes = u_bytes + out_bytes;
    base.stored += m * k * 4 + k * 4;
    Ok(base)
}
/// KBENCH_TILED_THREADS=T: the X arm's tiled block size (default the fixture's 256).
fn tiled_threads(threads: u32, arm: usize) -> u32 {
    let t = std::env::var("KBENCH_TILED_THREADS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    if arm == 1 && t > 0 { t } else { threads }
}
fn persist(grid: [u32; 3], arm: usize) -> [u32; 3] {
    let ctas = std::env::var("KBENCH_TILED_CTAS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    if arm == 1 && ctas > 0 { [(grid[0] * grid[1]).min(ctas), 1, 1] } else { grid }
}
/// kind 61 launch parameters for one weight copy: matrices back to back in the copy, output
/// columns back to back in one [M][sum] row; absent matrices are (0, 0, 0, 0) as in production.
fn prompt_pairs(c: &Case, base: &params::Bf16Rows, weights: u64) -> params::Bf16Pairs {
    let mut pairs = [params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0, ..*base }; 2];
    let mut at = 0;
    for (j, &r) in c.mats.iter().enumerate() {
        if r == 0 {
            continue;
        }
        let (w, y, rows, stride) = (weights + (at * c.k * 2) as u64, base.output + (at * 4) as u64, r as u32, c.n as u32);
        let p = &mut pairs[j / 2];
        if j % 2 == 0 {
            (p.weight, p.output, p.rows, p.stride) = (w, y, rows, stride);
        } else {
            (p.weight2, p.output2, p.rows2, p.stride2) = (w, y, rows, stride);
        }
        at += r;
    }
    params::Bf16Pairs { pairs }
}
/// kind 65 (prefill_gemm owner): prompt HC up on the production pack of a raw FP32 [M][320]
/// low activation (packed once), 4H x 320 blocked weights and FP32 normalized streams
/// [M][4H]; output [M][H]. A = main's flash_next_hc_up_latency4 (grid H/8 x M/16, block128),
/// X = flash_next_hc_up_tiled (grid M/64 x H/32, block256, 96 KiB).
fn hc_up_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, h) = (c.m, 2560usize);
    let mut v = prompt_fixture(g, &Case { kind: 60, n: 4 * h, k: 320, ..c.clone() })?;
    let normalized: Vec<u8> = (0..m * 4 * h)
        .flat_map(|i| ((((i / (4 * h)) * 29 + (i % (4 * h)) * 13) % 241) as f32 / 64. - 1.9).to_le_bytes())
        .collect();
    v.aux = g.upload(&normalized)?;
    // KBENCH_HCUP_SCALES=1: X forms the streams from a residual, power-of-two scales [M][4] (in
    // aux) and weights with 1 + w in {0.5, 1, 2, 4}, residual = normalized / (scale (1 + w)), so
    // residual * scale * (1 + w) gives the same streams exactly and the hashes stay comparable.
    if c.name.starts_with("hc_up_prod-") || std::env::var("KBENCH_HCUP_SCALES").is_ok_and(|v| v == "1") {
        let scale = |t: usize, s: usize| (2f32).powi(((t * 4 + s) % 5) as i32 - 2);
        let weight = |j: usize| [-0.5f32, 0., 1., 3.][(j * 7 + j / 16) % 4];
        let at = |i: usize| f32::from_le_bytes(normalized[i * 4..i * 4 + 4].try_into().unwrap());
        let residual: Vec<u8> = (0..m * 4 * h)
            .flat_map(|i| (at(i) / (scale(i / (4 * h), i % (4 * h) / h) * (1. + weight(i % (4 * h))))).to_le_bytes()).collect();
        let scales: Vec<u8> = (0..m * 4).flat_map(|i| scale(i / 4, i % 4).to_le_bytes()).collect();
        let weights: Vec<u8> = (0..4 * h).flat_map(|j| weight(j).to_le_bytes()).collect();
        v.ring_x = vec![g.upload(&residual)?, g.upload(&scales)?, g.upload(&weights)?];
    }
    // out = [output [M][H] | the output's dense hi/lo planes]; planes hashed at M % 16 == 0
    // (main's pack zero-fills the padded rows the tiled epilogue leaves unwritten).
    let plane = m.div_ceil(16) * (h / 16) * 128;
    v.out_bytes = m * h * 4 + 2 * plane * 4;
    v.written = if m % 16 == 0 { Vec::new() } else { vec![(0, m * h * 4)] };
    if hcup_a16(c) {
        v.out_bytes += m * h * 2;
        v.out = g.alloc(v.out_bytes)?;
        v.rows.output = v.out;
        // output_hash uses all live FP32, packed hi/lo, and BF16 words. Padding is excluded.
        v.written.clear();
    }
    v.rows.stride = h as u32;
    v.grid = [m.div_ceil(64) as u32, (h / 32 / if hcup_a16(c) {params::FLASH_HC_UP_SLICES} else {1}) as u32, 1];
    v.shared = params::FLASH_BF16_PREFILL_TILED_SHARED as u32;
    v.stored = 4 * h * 320 * 2 + m * 320 * 4 + m * 4 * h * 4 + m * h * 4;
    Ok(v)
}
/// kind 69 (prefill_gemm owner): flash_bf16_input alone (32 lanes) on kind 60's raw FP32
/// [M][K] activation, each arm its own module's pack; the hash covers both BF16 planes.
fn pack_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let mut v = prompt_fixture(g, c)?;
    let plane = c.m.div_ceil(16) * c.k.div_ceil(16) * 128;
    (v.out, v.out_bytes) = (v.rows.input, plane * 2 * 4);
    (v.grid, v.threads, v.shared) = ([(plane / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0);
    (v.stored, v.flops) = (c.m * c.k * 4 + plane * 2 * 4, 0);
    Ok(v)
}
/// Fixtures of kinds 60..68 except 63 (the caller routes only those here).
pub(super) fn fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    match c.kind {
        64 => hc_norm_fixture(g, c),
        65 => hc_up_fixture(g, c),
        66 => hc_add_fixture(g, c),
        67 => hc_finish_fixture(g, c),
        68 => hc_low_fixture(g, c),
        69 => pack_fixture(g, c),
        _ => prompt_fixture(g, c),
    }
}
/// Per-case setup after the fixture: main's production pack once for kinds 60/62/65, the
/// extra kernels of kinds 64/66; Some(functions) where the arms' entries are not c.kernel.
pub(super) fn prepare(g: &mut Gpu, modules: [H; 2], c: &Case, v: &mut Fixture) -> Result<Option<[H; 2]>> {
    if matches!(c.kind, 64 | 65 | 66 | 67 | 68) {
        v.pack = g.function(modules[0], "flash_bf16_input")?;
    }
    if c.kind == 68 {
        v.norms[0] = g.function(modules[0], "flash_next_hc_activate")?;
    }
    if c.kind == 64 {
        v.norms = [g.function(modules[0], "flash_hc_norm_f32")?, g.function(modules[1], "flash_hc_norm_f32")?];
    }
    if matches!(c.kind, 60 | 62 | 65 | 68) {
        // Main's production pack, once: both arms read the same packed planes.
        let pack = g.function(modules[0], "flash_bf16_input")?;
        let mut q = params::FlashBf16Input { input: v.raw, output: v.rows.input, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: 32 };
        let words = c.m.div_ceil(16) * c.k.div_ceil(16) * 128;
        g.launch(pack, [(words / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        g.sync()?;
    }
    Ok(match c.kind {
        // A = main's prompt HC up (latency4), X = the tiled twin of the same graph.
        65 => {
            let prod=c.name.starts_with("hc_up_prod-");
            if hcup_a16(c) { v.norms[0]=g.function(modules[0],"fp4_prefill_tc_a16")?; }
            let x=if hcup_a16(c) {"flash_next_hc_up_tiled_a16"} else if prod && std::env::var("KBENCH_HCUP_M32").is_ok_and(|v|v=="1") {"flash_next_hc_up_m32"}else{c.kernel};
            Some([g.function(modules[0],if prod {c.kernel}else{"flash_next_hc_up_latency4"})?,g.function(modules[1],x)?])
        },
        69 => Some([g.function(modules[0], c.kernel)?, g.function(modules[1], c.kernel)?]),
        // A = main's tiled HC down (+ activate + pack below), X = the fused HC down.
        68 => Some([g.function(modules[0], c.kernel)?, g.function(modules[1], "flash_bf16_mma_prefill_tiled_hc")?]),
        63 => {
            let m = g.module(mma_peak_ptx().as_bytes()).map_err(|e| format!("mma PTX: {e}"))?;
            let f = g.function(m, "kb_mma")?;
            Some([f, f])
        }
        66 | 67 => {
            // A: main's hc_add (norms[0]) + flash_hc_norm_f32 (this entry) + pack; X: the fusion.
            // Both arms first restore the residual with kb_copy (norms[1]).
            let m = g.module(copy_ptx().as_bytes()).map_err(|e| format!("copy PTX: {e}"))?;
            v.norms = [g.function(modules[0], "flash_next_hc_add")?, g.function(m, "kb_copy")?];
            // "fast" names (fp4_bw): X = the _fast entry (gate exp as exp_split).
            let x = if c.name.contains("fast") { "flash_next_hc_finish_add_norm_fast" } else { c.kernel };
            Some([g.function(modules[0], if c.name.starts_with("hcnorm-prod-") { x } else { "flash_hc_norm_f32" })?, g.function(modules[1], x)?])
        }
        _ => None,
    })
}
/// kb_copy(dst, src, n): n 16-byte words, one per thread.
pub(super) fn copy_ptx() -> String {
    String::from(concat!(".version 8.8\n.target sm_121\n.address_size 64\n",
        ".visible .entry kb_copy(.param .u64 dst, .param .u64 src, .param .u32 n)\n{\n.reg .pred p;\n.reg .b32 r<8>;\n.reg .b64 d<6>;\n",
        "ld.param.u64 d1,[dst]; ld.param.u64 d2,[src]; ld.param.u32 r1,[n];\n",
        "mov.u32 r2,%ctaid.x; mov.u32 r3,%ntid.x; mov.u32 r4,%tid.x; mad.lo.u32 r2,r2,r3,r4; setp.ge.u32 p,r2,r1; @p bra DONE;\n",
        "mul.wide.u32 d3,r2,16; add.u64 d4,d2,d3; add.u64 d5,d1,d3;\n",
        "ld.global.v4.u32 {r4,r5,r6,r7},[d4]; st.global.v4.u32 [d5],{r4,r5,r6,r7};\nDONE:\nret;\n}\n"))
}
/// kind 68 (prefill_gemm owner): prompt HC down on M tokens, c.mats = [low rows, injection
/// rows] (the injection rides in the last N tile). A = main's tiled launch into FP32 low,
/// flash_next_hc_activate and flash_bf16_input (K = low); X = flash_bf16_mma_prefill_tiled_hc
/// writing the activated planes. Hash: [planes | injection] (planes only when M % 16 == 0:
/// A's pack zero-fills the padded rows the fused kernel leaves unwritten).
fn hc_low_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, low, inject) = (c.m, c.mats[0], c.mats[1]);
    let mut v = prompt_fixture(g, &Case { kind: 60, ..c.clone() })?;
    let plane = m.div_ceil(16) * (low / 16) * 128;
    v.out_bytes = 2 * plane * 4 + m * inject * 4;
    v.out = g.alloc(v.out_bytes)?;
    v.aux = g.alloc(m * low * 4)?;
    v.written = if m % 16 == 0 { Vec::new() } else { vec![(2 * plane * 4, m * inject * 4)] };
    v.stored = v.weight_bytes + m * c.k * 4 + v.out_bytes;
    Ok(v)
}
/// kind 66 (prefill_gemm owner): prompt HC combine + next mix norm on M tokens: residual
/// [M][4H] restored from a pristine copy each launch, block [M][H], gates [M][4], F32 norm
/// weights [4H]. A = main's flash_next_hc_add + flash_hc_norm_f32 + flash_bf16_input; X =
/// flash_next_hc_add_norm writing the planes. Hash: [residual | u | planes] (planes only when
/// M % 16 == 0: A's pack zero-fills the padded rows the fused kernel leaves unwritten).
fn hc_add_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, w, h) = (c.m, 10240usize, 2560usize);
    let mut v = prompt_fixture(g, &Case { kind: 60, n: 4, k: w, ..c.clone() })?;
    let f32s = |n: usize, f: &dyn Fn(usize) -> f32| -> Vec<u8> { (0..n).flat_map(|i| f(i).to_le_bytes()).collect() };
    v.raw = g.upload(&f32s(m * w, &|i| ((i / w * 37 + i % w * 11) % 263) as f32 / 32. - 4.))?;
    v.rows.weight = g.upload(&f32s(m * h, &|i| ((i / h * 7 + i % h * 5) % 127) as f32 / 64. - 1.))?;
    v.rows.weight2 = g.upload(&f32s(m * 4, &|i| (i % 13) as f32 / 4. - 1.5))?;
    v.rows.output2 = g.upload(&f32s(w, &|i| ((i * 7) % 61) as f32 / 256. - 0.1))?;
    let plane = m.div_ceil(16) * (w / 16) * 128;
    v.out_bytes = 2 * m * w * 4 + 2 * plane * 4;
    v.out = g.alloc(v.out_bytes)?;
    v.written = if m % 16 == 0 { Vec::new() } else { vec![(0, 2 * m * w * 4)] };
    // KBENCH_HCNORM_SCALES=1: X writes the norm scales into u instead of the normalized rows;
    // both arms hash [residual | planes] only (u: HC up's KBENCH_HCUP_SCALES check, and bytegate).
    if c.name.starts_with("hcnorm-prod-") || std::env::var("KBENCH_HCNORM_SCALES").is_ok_and(|v| v == "1") {
        v.written = if m % 16 == 0 { vec![(0, m * w * 4), (2 * m * w * 4, 2 * plane * 4)] } else { vec![(0, m * w * 4)] };
    }
    v.stored = m * w * 4 * 3 + m * h * 4 + 2 * m * w * 4;
    if c.name.starts_with("hcnorm-prod-") || std::env::var("KBENCH_HCNORM_SCALES").is_ok_and(|v| v == "1") {
        v.stored -= m * w * 4;
    }
    if c.name.starts_with("hcnorm-prod-") && (c.name.starts_with("hcnorm-prod-") || std::env::var("KBENCH_HCNORM_SCALES").is_ok_and(|v| v == "1")) {
        v.written.push((m * w * 4, m * 4 * 4));
    }
    v.flops = 0;
    if c.kind == 66 || c.kind == 67 {
        // Lets a token-pair add_norm / finish_add_norm (KBENCH_HCNORM=pair) take its stage.
        v.shared = (8 * 2560 * 4 + 32 + 2 * 2560 * 2 + 2 * 16 * 4) as u32;
    }
    Ok(v)
}
/// kind 67: kind 66 whose block row is the token's MoE output: A = main's k16_finish (writes
/// block) + the kind-66 chain; X = flash_next_hc_finish_add_norm forming it in shared memory.
/// Synthetic routing: ten distinct ids per token, finite weights, BF16 expert rows and a FP32
/// shared row per token in the [M][11][2560] partials, one shared-gate logit, no error rows.
fn hc_finish_fixture(g: &mut Gpu, c: &Case) -> Result<Fixture> {
    let (m, h) = (c.m, 2560usize);
    let mut v = hc_add_fixture(g, c)?;
    let bytes = |n: usize, f: &dyn Fn(usize) -> [u8; 4]| -> Vec<u8> { (0..n).flat_map(f).collect() };
    let error = g.upload(&bytes(m, &|_| 0u32.to_le_bytes()))?;
    let ids = g.upload(&bytes(m * 10, &|i| ((((i / 10) * 37 + (i % 10) * 51) % 512) as u32).to_le_bytes()))?;
    let weights = g.upload(&bytes(m * 10, &|i| ((i % 10 + 1) as f32 / 16.).to_le_bytes()))?;
    let partials = if c.name.starts_with("hcnorm-prod-") {
        // Nonzero BF16 routed rows (with their ABI padding) and an FP32 shared row.
        let mut data = vec![0u8; m * 11 * h * 4];
        for row in 0..m {
            for rank in 0..10 {
                for col in 0..h {
                    let value = (((row * 13 + rank * 17 + col * 7) % 127) as f32 - 63.) / 64.;
                    let bits = (value.to_bits() >> 16) as u16;
                    let at = (row * 22 * h + rank * 2 * h + col) * 2;
                    data[at..at + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
            for col in 0..h {
                let value = (((row * 11 + col * 3) % 61) as f32 - 30.) / 32.;
                let at = (row * 11 * h + 10 * h + col) * 4;
                data[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        g.upload(&data)?
    } else { g.fill(m * 11 * h * 4, 0)? };
    let gate = g.upload(&bytes(m, &|i| ((i % 9) as f32 / 4. - 1.).to_le_bytes()))?;
    v.ring_x = vec![error, ids, weights, partials, gate];
    v.stored += m * 11 * h * 4 + m * 10 * 8;
    Ok(v)
}
/// Launches of kinds 60..68 (the caller routes only those here).
#[allow(clippy::too_many_arguments)]
pub(super) fn launch(g: &Gpu, f: H, merge: H, c: &Case, v: &mut Fixture, i: usize, arm: usize, shared: Option<u32>) -> Result<()> {
    let arm = if c.name.starts_with("hcnorm-prod-") { 1 } else { arm };
    if c.kind == 63 {
        let (mut out, mut iters) = (v.out, c.n as u32);
        return g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut out), ptr(&mut iters)]);
    }
    if c.kind == 69 {
        let mut q = params::FlashBf16Input { input: v.raw, output: v.out, tokens: c.m as u32, cols: c.k as u32, hc_silu: 0, lanes: 32 };
        return g.launch(f, v.grid, v.threads, 0, &mut [ptr(&mut q)]);
    }
    if c.kind == 60 {
        v.rows.weight = v.ring[i];
        if c.name.starts_with("pgprod-") {
            if shared.unwrap_or(v.shared) != params::FLASH_BF16_PREFILL_TILED_SHARED as u32 {
                return Err("production prompt GEMM requires its full three-stage shared allocation".into());
            }
            v.rows.weight2=v.ring[i]+(c.mats[0]*c.k*2) as u64;
            let threads=if arm==1 && std::env::var("KBENCH_PGPROD_X").is_ok_and(|v|v=="w16") {512} else {256};
            return g.launch(f,v.grid,threads,shared.unwrap_or(v.shared),&mut [ptr(&mut v.rows)]);
        }
        return g.launch(f, persist(v.grid, arm), tiled_threads(v.threads, arm), shared.unwrap_or(v.shared), &mut [ptr(&mut v.rows)]);
    }
    if c.kind == 64 {
        let (m, k) = (c.m, 10240usize);
        let (down, inject, n) = (c.mats[0], c.mats[1], c.n as u32);
        let u = v.out;
        let y = v.out + (m * k * 4) as u64;
        let packed = v.rows.input;
        let mut np = params::FlashHcNorm { input: v.raw, weight: v.aux, output: u, packed: if arm == 0 { 0 } else { packed } };
        g.launch(v.norms[arm], [8, m as u32, 1], 128, 0, &mut [ptr(&mut np)])?;
        let (w, w2) = (v.ring[i], v.ring[i] + (down * k * 2) as u64);
        let mut tp = params::Bf16Rows { weight: w, output: y, rows: down as u32, stride: n, weight2: w2, output2: y + (down * 4) as u64,
            rows2: if arm == 0 { 0 } else { inject as u32 }, stride2: n, ..v.rows };
        if arm == 0 {
            let mut q = params::FlashBf16Input { input: u, output: packed, tokens: m as u32, cols: k as u32, hc_silu: 0, lanes: 32 };
            let words = m.div_ceil(16) * k.div_ceil(16) * 128;
            g.launch(v.pack, [(words / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        }
        let grid = [m.div_ceil(64) as u32, (down + inject).div_ceil(128) as u32, 1];
        g.launch(f, persist(grid, arm), tiled_threads(v.threads, arm), shared.unwrap_or(v.shared), &mut [ptr(&mut tp)])?;
        if arm == 0 {
            let solo = params::Bf16Rows { weight: w2, output: y + (down * 4) as u64, rows: inject as u32, stride: n, weight2: 0, output2: 0,
                rows2: 0, stride2: 0, input: u, input_bf16: 0, ..v.rows };
            let zero = params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, ..solo };
            let mut pp = params::Bf16Pairs { pairs: [solo, zero] };
            let blocks = params::flash_bf16_prefill_blocks(solo);
            g.launch(merge, [blocks as u32, 1, 1], 256, params::FLASH_BF16_MMA_SHARED as u32, &mut [ptr(&mut pp)])?;
        }
        return Ok(());
    }
    if c.kind == 68 {
        let (m, low, inject) = (c.m, c.mats[0], c.mats[1]);
        let plane = m.div_ceil(16) * (low / 16) * 128;
        let (planes, injection) = (v.out, v.out + (2 * plane * 4) as u64);
        let w = v.ring[i];
        let mut tp = params::Bf16Rows { weight: w, output: if arm == 0 { v.aux } else { planes }, rows: low as u32, stride: low as u32,
            weight2: w + (low * c.k * 2) as u64, output2: injection, rows2: inject as u32, stride2: inject as u32, ..v.rows };
        let grid = [m.div_ceil(64) as u32, (low + inject).div_ceil(128) as u32, 1];
        g.launch(f, grid, v.threads, shared.unwrap_or(v.shared), &mut [ptr(&mut tp)])?;
        if arm == 0 {
            let (mut x, mut n) = (v.aux, (m * low) as u32);
            g.launch(v.norms[0], [(m * low).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut x), ptr(&mut n)])?;
            let mut q = params::FlashBf16Input { input: v.aux, output: planes, tokens: m as u32, cols: low as u32, hc_silu: 0, lanes: 32 };
            g.launch(v.pack, [(plane / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)])?;
        }
        return Ok(());
    }
    if c.kind == 62 {
        // Columns and weight copies in mats order: big1 | small1 | big2 | small2.
        let [b1, s1, b2, s2] = c.mats;
        let at = |o: usize| (v.ring[i] + (o * c.k * 2) as u64, v.out + (o * 4) as u64);
        let (rows, n) = (v.rows, c.n as u32);
        let tiled = |big: usize, o: usize, small: usize| {
            let ((w, y), (w2, y2)) = (at(o), at(o + big));
            let small = if arm == 0 { 0 } else { small };
            let p = params::Bf16Rows { weight: w, output: y, rows: big as u32, stride: n, weight2: w2, output2: y2,
                rows2: small as u32, stride2: n, ..rows };
            ([c.m.div_ceil(64) as u32, (big + small).div_ceil(128) as u32, 1], p)
        };
        for (big, o, small) in [(b1, 0, s1), (b2, b1 + s1, s2)] {
            if big != 0 {
                let (grid, mut p) = tiled(big, o, small);
                g.launch(f, persist(grid, arm), tiled_threads(v.threads, arm), shared.unwrap_or(v.shared), &mut [ptr(&mut p)])?;
            }
        }
        if arm == 0 {
            let ((w, y), (w2, y2)) = (at(b1), at(b1 + s1 + b2));
            let solo = params::Bf16Rows { weight: w, output: y, rows: s1 as u32, stride: n, weight2: w2, output2: y2,
                rows2: s2 as u32, stride2: n, input: v.raw, input_bf16: 0, ..rows };
            let zero = params::Bf16Rows { weight: 0, output: 0, rows: 0, stride: 0, weight2: 0, output2: 0, rows2: 0, stride2: 0, ..solo };
            let mut p = params::Bf16Pairs { pairs: [solo, zero] };
            let blocks = params::flash_bf16_prefill_blocks(solo);
            g.launch(merge, [blocks as u32, 1, 1], 256, params::FLASH_BF16_MMA_SHARED as u32, &mut [ptr(&mut p)])?;
        }
        return Ok(());
    }
    if c.kind == 61 {
        // pairs[0] keeps the base input/output addresses; rebuild both pairs for this copy.
        let base = params::Bf16Rows { output: v.out, ..v.rows };
        let mut p = prompt_pairs(c, &base, v.ring[i]);
        return g.launch(f, v.grid, v.threads, shared.unwrap_or(v.shared), &mut [ptr(&mut p)]);
    }
    if c.kind == 65 {
        // A: latency4 + main's pack of its output; X: the tiled kernel writing both.
        v.rows.weight = v.ring[i];
        let (mut normalized, mut planes) = (v.aux, v.out + (c.m * 2560 * 4) as u64);
        if arm == 0 && !c.name.starts_with("hc_up_prod-") {
            g.launch(f, [2560 / 8, c.m.div_ceil(16) as u32, 1], 128, 0, &mut [ptr(&mut v.rows), ptr(&mut normalized)])?;
            let mut q = params::FlashBf16Input { input: v.out, output: planes, tokens: c.m as u32, cols: 2560, hc_silu: 0, lanes: 32 };
            let words = c.m.div_ceil(16) * 160 * 128;
            return g.launch(v.pack, [(words / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)]);
        }
        // residual 0: the X kernel reads the normalized streams (older X modules ignore the extra arguments).
        let (mut residual, mut weight) = (0u64, 0u64);
        if let [r, s, w] = v.ring_x[..] {
            (residual, normalized, weight) = (r, s, w);
        }
        let small=arm==1 && c.name.starts_with("hc_up_prod-") && std::env::var("KBENCH_HCUP_M32").is_ok_and(|v|v=="1");
        let (grid,threads,smem)=if small {([c.m.div_ceil(32) as u32,80,1],128,49152)}else{(v.grid,256,shared.unwrap_or(v.shared))};
        if hcup_a16(c) {
            v.rows.output2 = v.out + (v.out_bytes - c.m * 2560 * 2) as u64;
        }
        g.launch(f, grid, threads, smem,
            &mut [ptr(&mut v.rows), ptr(&mut normalized), ptr(&mut planes), ptr(&mut residual), ptr(&mut weight)])?;
        if hcup_a16(c) && arm == 0 {
            let (mut input, mut output, mut groups) = (v.out, v.rows.output2, (c.m * 2560 / 8) as u32);
            g.launch(v.norms[0], [groups.div_ceil(256),1,1],256,0,&mut [ptr(&mut input),ptr(&mut output),ptr(&mut groups)])?;
        }
        return Ok(());
    }
    if c.kind == 67 && arm == 0 {
        let x = &v.ring_x;
        let mut fp = fp8::MoeFinishParams { error: x[0], ids: x[1], weights: x[2], partials: x[3], shared_gate: x[4],
            output: v.rows.weight, activation_quantization: 1, column_start: 0, columns: 2560 };
        g.launch(merge, [10, c.m as u32, 1], 256, 0, &mut [ptr(&mut fp)])?;
    }
    if c.kind == 66 || c.kind == 67 {
        let (m, w) = (c.m, 10240usize);
        let (residual, u) = (v.out, v.out + (m * w * 4) as u64);
        let planes = v.out + (2 * m * w * 4) as u64;
        let (mut dst, mut src, mut words) = (residual, v.raw, (m * w / 4) as u32);
        g.launch(v.norms[1], [words.div_ceil(256), 1, 1], 256, 0, &mut [ptr(&mut dst), ptr(&mut src), ptr(&mut words)])?;
        let (block, gates, weight) = (v.rows.weight, v.rows.weight2, v.rows.output2);
        if arm == 0 {
            let mut a = params::FlashHcAdd { residual, block, peer: 0, gates, hidden: 2560, rank: 0 };
            g.launch(v.norms[0], [20, m as u32, 1], 128, 0, &mut [ptr(&mut a)])?;
            let mut n = params::FlashHcNorm { input: residual, weight, output: u, packed: 0 };
            g.launch(f, [8, m as u32, 1], 128, 0, &mut [ptr(&mut n)])?;
            let mut q = params::FlashBf16Input { input: u, output: planes, tokens: m as u32, cols: w as u32, hc_silu: 0, lanes: 32 };
            let words = m.div_ceil(16) * (w / 16) * 128;
            return g.launch(v.pack, [(words / 128 * 32).div_ceil(128) as u32, 1, 1], 128, 0, &mut [ptr(&mut q)]);
        }
        // KBENCH_HCNORM_NOPACK=1 (diagnostic): the X kernel skips the planes (their bytes differ).
        let nopack = std::env::var("KBENCH_HCNORM_NOPACK").is_ok_and(|v| v == "1");
        let scales = if c.name.starts_with("hcnorm-prod-") || std::env::var("KBENCH_HCNORM_SCALES").is_ok_and(|v| v == "1") { u } else { 0 };
        let mut p = params::FlashHcAddNorm { residual, block, gates, weight, output: u, packed: if nopack { 0 } else { planes }, scales };
        // KBENCH_HCNORM=pair: an X module whose add_norm / finish_add_norm run one CTA per token
        // pair (t, t + 8) (rows as the last argument).
        let pair = c.name.starts_with("hcnorm-prod-") || std::env::var("KBENCH_HCNORM").is_ok_and(|v| v == "pair");
        let mut rows = m as u32;
        if c.kind == 66 && pair {
            return g.launch(f, [1, (m.div_ceil(16) * 8) as u32, 1], 256, 8 * 2560 * 4 + 32, &mut [ptr(&mut p), ptr(&mut rows)]);
        }
        if c.kind == 67 {
            let x = &v.ring_x;
            let mut fp = fp8::MoeFinishParams { error: x[0], ids: x[1], weights: x[2], partials: x[3], shared_gate: x[4],
                output: 0, activation_quantization: 1, column_start: 0, columns: 2560 };
            if pair {
                return g.launch(f, [1, (m.div_ceil(16) * 8) as u32, 1], 256, 8 * 2560 * 4 + 32 + 2 * 2560 * 2 + 2 * 16 * 4,
                    &mut [ptr(&mut p), ptr(&mut fp), ptr(&mut rows)]);
            }
            return g.launch(f, [1, m as u32, 1], 128, 5 * 2560 * 4 + 64, &mut [ptr(&mut p), ptr(&mut fp)]);
        }
        return g.launch(f, [1, m as u32, 1], 128, 4 * 2560 * 4, &mut [ptr(&mut p)]);
    }
    Err(format!("{}: not a prefill_gemm kind", c.name))
}

/// Cross-M HC equality covers every live FP32 output and both packed BF16 planes.
/// Padding has no logical row; same-shape A/X hashes remain unchanged.
pub(super) fn hc_logical_rows(raw: &[u8], m: usize, h: usize) -> Result<Vec<u8>> {
    let plane = m.div_ceil(16) * (h / 16) * 128;
    if h % 16 != 0 || raw.len() != m * h * 4 + 2 * plane * 4 { return Err("HC cross-M output layout".into()); }
    let mut out = Vec::with_capacity(m * h * 8);
    for t in 0..m {
        out.extend_from_slice(&raw[t*h*4..(t+1)*h*4]);
        for half in 0..2 {
            for c in (0..h).step_by(2) {
                let word = ((t / 16 * (h / 16) + c / 16) * 32 + t % 8 * 4 + c % 8 / 2) * 4 + t % 16 / 8 + c % 16 / 8 * 2;
                let at = m * h * 4 + (half * plane + word) * 4;
                out.extend_from_slice(&raw[at..at+4]);
            }
        }
    }
    Ok(out)
}

/// Canonical rows include the new BF16 output as well as every previous live output.
pub(super) fn hc_a16_logical_rows(raw: &[u8], m: usize, h: usize) -> Result<Vec<u8>> {
    let extra=m*h*2;
    let base=raw.len().checked_sub(extra).ok_or("HC a16 length")?;
    let canonical=hc_logical_rows(&raw[..base],m,h)?;
    let mut out=Vec::with_capacity(m*h*10);
    for t in 0..m {
        let live=&canonical[t*h*8..(t+1)*h*8];
        if live[..h*4].chunks_exact(4).any(|b|!f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            || live[h*4..].chunks_exact(2).any(|b|u16::from_le_bytes([b[0],b[1]])&0x7f80==0x7f80) {
            return Err("HC a16 prior outputs nonfinite/unwritten".into());
        }
        out.extend_from_slice(live);
        let row=&raw[base+t*h*2..base+(t+1)*h*2];
        if row.chunks_exact(2).any(|b|u16::from_le_bytes([b[0],b[1]])&0x7f80==0x7f80) {return Err("HC a16 nonfinite/unwritten".into());}
        out.extend_from_slice(row);
    }
    Ok(out)
}
