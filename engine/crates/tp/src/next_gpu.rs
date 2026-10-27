//! Rank-local FP4-Local adapter. Kernel ABI is the reviewed card integration's
//! 35397f8; its kernel source is reused verbatim in the production PTX.
pub use crate::next::native_projection;
use crate::{
    Result,
    next::{RankKernels, all_finite},
    next_bw::{BW_TABLES, bw_route},
};
use engine_model::{
    flash::plan::Selection,
    flash::{
        PLE_PART_ROWS, PLE_PREFIX, nvfp4,
        refcpu::{self, PleHash},
        shard::ShardCheckpoint,
    },
};
use spark_engine::flash::{
    Weights,
    blocks::{self, H, Hc, HcScratch, Ple, W},
    gdn::{Gdn, GdnScratch},
    head::Head,
    next_native::{self, Matrix},
    qsa::{Qsa, QsaScratch},
};
use spark_gpu_probe::{
    driver::{Backend, DeviceVector},
    kernel_params::*,
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    ops::Range,
};
#[path = "next_chunk.rs"]
mod chunk;
fn raw(c: &ShardCheckpoint, n: &str) -> Result<Vec<u8>> {
    let t = c
        .plan
        .tensors
        .get(n)
        .ok_or_else(|| format!("missing tensor {n}"))?;
    let mut v = vec![0; t.bytes() as usize];
    c.read_local(n, 0, &mut v).map_err(|e| e.to_string())?;
    Ok(v)
}
fn launch<T>(b: &Backend<'_>, name: &str, grid: usize, threads: u32, p: &mut T) -> Result<()> {
    unsafe { b.launch_flash_shared(name, grid as u32, threads, 0, &mut [(p as *mut T).cast()]) }
}
pub struct Arena<'a> {
    pub weights: Weights<'a>,
    native: BTreeMap<String, Matrix>,
    tables: Vec<DeviceVector<'a>>,
    up_tables: Vec<DeviceVector<'a>>,
    /// Whole-expert EP (expert TP 1): per layer, main's fp4_abi::Expert table over all 512 ids
    /// (width 0 = owned by a peer) and the F32 globals it points at; see `Gpu::bw_experts`.
    bw_tables: Vec<DeviceVector<'a>>,
    _bw_globals: Vec<DeviceVector<'a>>,
    _scales: Vec<DeviceVector<'a>>,
    rows_work: RefCell<spark_engine::flash::next_batched::RankRows<'a>>,
    ple_global: f32,
    pub bytes: usize,
}
impl<'a> Arena<'a> {
    pub fn load(b: &Backend<'a>, c: &ShardCheckpoint) -> Result<Self> {
        let weights = Weights::load_shard(b, c)?;
        let mut native = BTreeMap::new();
        if c.plan.shape.hidden != H as u64 || c.plan.shape.layers.len() != 48 {
            return Err("GPU adapter requires qualified Flash-Next geometry".into());
        }
        let config = engine_model::flash::Config::open(&c.root().join("config.json"))
            .map_err(|e| e.to_string())?;
        let plan = config
            .mixed
            .as_ref()
            .ok_or("GPU adapter requires mixed storage")?;
        let engine_model::plan::Format::Mixed(q) = &plan.format else {
            return Err("GPU adapter requires mixed storage".into());
        };
        for (name, t) in &c.plan.tensors {
            let Some(module) = native_projection(name, t, q) else {
                continue;
            };
            if t.tensor.shape.len() != 2 {
                return Err(format!("{name}: MXFP8 target is not a matrix"));
            }
            let cols = match &t.selection {
                Selection::Columns(r) => (r.end - r.start) as usize,
                _ => t.tensor.shape[1] as usize,
            };
            let rows = t.bytes() as usize / cols;
            native.insert(
                module.into(),
                Matrix {
                    weights: weights.device_address(name)?,
                    scales: weights.device_address(&format!("{module}.weight_scale"))?,
                    rows: rows as u32,
                    cols: cols as u32,
                },
            );
        }
        if native.len() != 384 {
            return Err(format!(
                "expected 384 native rank projections, got {}",
                native.len()
            ));
        }
        let (mut scales, mut tables, mut up_tables) = (Vec::new(), Vec::new(), Vec::new());
        let (mut bw_tables, mut bw_globals) = (Vec::new(), Vec::new());
        let tp = c.plan.expert_tp;
        let scale_bytes = 102400 / tp;
        let owned = crate::rows(512, c.plan.rank / tp, c.plan.world / tp);
        for li in 0..48 {
            let mut scale = vec![0u8; owned.len() * 3 * scale_bytes];
            let mut table = vec![0u8; 512 * 72];
            for (local, id) in owned.clone().enumerate() {
                for (pi, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let m = format!("model.language_model.layers.{li}.mlp.experts.{id}.{proj}");
                    let fixed = raw(c, &format!("{m}.weight_scale.nvfp4_csf_fixed"))?;
                    let words = raw(c, &format!("{m}.weight_scale.nvfp4_csf_exceptions"))?;
                    if words.len() % 4 != 0 {
                        return Err("invalid CSF exceptions".into());
                    }
                    let ex = words
                        .chunks_exact(4)
                        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let (r, k) = if pi == 2 { (2560, 640) } else { (640, 2560) };
                    let expanded =
                        nvfp4::csf_expand(&fixed, &ex, r, k).map_err(|e| e.to_string())?;
                    if expanded.iter().any(|s| *s >= 127) {
                        return Err("nonfinite CSF scale".into());
                    }
                    let at = (c.plan.rank % tp) * scale_bytes;
                    scale[(local * 3 + pi) * scale_bytes..(local * 3 + pi + 1) * scale_bytes]
                        .copy_from_slice(&expanded[at..at + scale_bytes]);
                    let global = f32::from_le_bytes(
                        raw(c, &format!("{m}.weight_scale_2"))?
                            .try_into()
                            .map_err(|_| "bad expert global")?,
                    );
                    if !global.is_finite() || global <= 0. {
                        return Err("invalid expert global".into());
                    }
                    let at = (id * 3 + pi) * 24;
                    table[at..at + 8].copy_from_slice(
                        &weights
                            .device_address(&format!("{m}.weight"))?
                            .to_le_bytes(),
                    );
                    table[at + 16..at + 20].copy_from_slice(&global.to_le_bytes());
                }
            }
            let mut v = b.vector(scale.len() / 4)?;
            b.write_vector_bytes(&mut v, &scale)?;
            for (local, id) in owned.clone().enumerate() {
                for pi in 0..3 {
                    let at = (id * 3 + pi) * 24 + 8;
                    table[at..at + 8].copy_from_slice(
                        &(v.device_address() + ((local * 3 + pi) * scale_bytes) as u64)
                            .to_le_bytes(),
                    );
                }
            }
            let mut t = b.vector(table.len() / 4)?;
            b.write_vector_bytes(&mut t, &table)?;
            if tp == 1 {
                // The same weight, expanded-scale and global addresses as `table`, in main's
                // fp4_abi::Expert layout (fp4_bw_gate_c1 / fp4_bw_down read these).
                use spark_gpu_probe::driver::fp4_abi::{Expert, Matrix};
                let globals = (0..512 * 3)
                    .map(|i| f32::from_le_bytes(table[i * 24 + 16..i * 24 + 20].try_into().unwrap()))
                    .collect::<Vec<_>>();
                let mut g = b.vector(globals.len())?;
                b.write_vector(&mut g, &globals)?;
                let word = |at: usize| u64::from_le_bytes(table[at..at + 8].try_into().unwrap());
                let mut descriptors = vec![Expert::default(); 512];
                for id in owned.clone() {
                    let m = |pi: usize| Matrix {
                        weights: word((id * 3 + pi) * 24),
                        scales: word((id * 3 + pi) * 24 + 8),
                        global_scale: g.device_address() + ((id * 3 + pi) * 4) as u64,
                    };
                    descriptors[id] = Expert { gate: m(0), up: m(1), down: m(2), width: 640, reserved: 0 };
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        descriptors.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(descriptors.as_slice()),
                    )
                };
                let mut e = b.vector(bytes.len() / 4)?;
                b.write_vector_bytes(&mut e, bytes)?;
                bw_tables.push(e);
                bw_globals.push(g);
            }
            scales.push(v);
            tables.push(t);
            if tp == 2 {
                for id in owned.clone() {
                    let up = table[id * 72 + 24..id * 72 + 48].to_vec();
                    table[id * 72..id * 72 + 24].copy_from_slice(&up);
                }
                let mut t = b.vector(table.len() / 4)?;
                b.write_vector_bytes(&mut t, &table)?;
                up_tables.push(t);
            }
        }
        let ple_global = f32::from_le_bytes(
            raw(c, &format!("{PLE_PREFIX}.ngram_embedding.weight_scale_2"))?
                .try_into()
                .map_err(|_| "PLE global")?,
        );
        let before_rows = b.accounted_bytes();
        let rows_work = spark_engine::flash::next_batched::RankRows::new(b)?;
        let bytes = weights.resident_bytes()
            + (b.accounted_bytes() - before_rows)
            + scales
                .iter()
                .chain(&tables)
                .chain(&up_tables)
                .chain(&bw_tables)
                .chain(&bw_globals)
                .map(|v| v.len() * 4)
                .sum::<usize>();
        Ok(Self {
            weights,
            rows_work: RefCell::new(rows_work),
            ple_global,
            native,
            tables,
            up_tables,
            bw_tables,
            _bw_globals: bw_globals,
            _scales: scales,
            bytes,
        })
    }
    fn matrix(&self, n: &str) -> Result<Matrix> {
        self.native
            .get(n)
            .copied()
            .ok_or_else(|| format!("missing native projection {n}"))
    }
    /// First GPU cell: owned expert bytes, expanded scales and pointer table
    /// against the host oracle before allocating a full causal session.
    pub fn smoke(&self, b: &Backend<'a>, c: &ShardCheckpoint) -> Result<()> {
        let tp = c.plan.expert_tp;
        let id = crate::rows(512, c.plan.rank / tp, c.plan.world / tp).start;
        let m = format!("model.language_model.layers.0.mlp.experts.{id}.gate_proj");
        let rows = 640 / tp;
        let fixed = raw(c, &format!("{m}.weight_scale.nvfp4_csf_fixed"))?;
        let ex = raw(c, &format!("{m}.weight_scale.nvfp4_csf_exceptions"))?
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        let scales = nvfp4::csf_expand(&fixed, &ex, 640, H).map_err(|e| e.to_string())?;
        let at = c.plan.rank % tp * rows * H / 16;
        let global = f32::from_le_bytes(
            raw(c, &format!("{m}.weight_scale_2"))?
                .try_into()
                .map_err(|_| "smoke global")?,
        );
        let mut reference = vec![0.; rows];
        nvfp4::project_ref(
            &vec![0x3f80; H],
            &raw(c, &format!("{m}.weight"))?,
            &scales[at..at + rows * H / 16],
            global,
            rows,
            H,
            &mut reference,
        )
        .map_err(|e| e.to_string())?;
        let mut v = b.vector_arena(&[H / 2, rows / 2, 3, 1])?;
        b.write_vector(&mut v[0], &vec![f32::from_bits(0x3f803f80); H / 2])?;
        b.write_vector(&mut v[2], &[id as u32, 0, 1].map(f32::from_bits))?;
        b.write_vector(&mut v[3], &[f32::from_bits(1)])?;
        let mut p = NextRouted {
            experts: self.tables[0].device_address(),
            input: v[0].device_address(),
            output: v[1].device_address(),
            tiles: v[2].device_address(),
            tile_count: v[3].device_address(),
            rows: rows as u32,
            cols: H as u32,
            ..Default::default()
        };
        launch(b, "flash_next_routed", 192, 256, &mut p)?;
        let packed = b.read_vector(&v[1])?;
        let (mut error, mut norm) = (0f64, 0f64);
        for (i, r) in reference.iter().enumerate() {
            let g = nvfp4::bf16_to_f32((packed[i / 2].to_bits() >> ((i % 2) * 16)) as u16);
            error += (g as f64 - *r as f64).powi(2);
            norm += (*r as f64).powi(2);
        }
        if error.sqrt() > 1e-5 * (rows as f64).sqrt() + norm.sqrt() / 128. {
            return Err("rank shard projection normwise failed".into());
        }
        println!(
            "PASS rank_shard_project rank={} expert={id} rows={rows} relative_l2={}",
            c.plan.rank,
            error.sqrt() / norm.sqrt().max(1e-30)
        );
        Ok(())
    }
}
/// One token's MoE on main's world-one expert kernels (fp4_bw_gate_c1 + fp4_bw_down) for
/// whole-expert EP ranks. Per element they run the same arithmetic as flash_next_routed +
/// flash_next_routed_activate: RN BF16 A rows of the mixed F32 input, next_pair weights,
/// one ascending m16n8k16 K16 chain from zero, BF16 gate/up, the same SwiGLU and a BF16
/// down boundary (fp4_experts.rs), so each owned expert's BF16 rows keep their bits and the
/// host fold below is unchanged.
/// fp4_bw_down's output layout (fp4_experts.rs write_slots): slot `token * 11 + rank`, each slot
/// an FP32-sized stride of 2560 words whose first 1280 hold the row's BF16 pairs.
const BW_SLOTS: usize = 11;
const BW_SLOT_WORDS: usize = H;
struct BwScratch<'a> {
    route: DeviceVector<'a>,
    middle: DeviceVector<'a>,
    out: DeviceVector<'a>,
}
impl<'a> BwScratch<'a> {
    fn new(b: &Backend<'a>, arena: &Arena<'a>) -> Result<Option<Self>> {
        if arena.bw_tables.is_empty()
            || !b.has_flash("fp4_bw_gate_c1")
            || !b.has_flash("fp4_bw_down")
        {
            return Ok(None);
        }
        let mut v = b.vector_arena(&[
            BW_TABLES + spark_gpu_probe::compact::words(1),
            10 * 640 / 2,
            BW_SLOTS * BW_SLOT_WORDS,
        ])?;
        // Slot 10 (the shared row in main's layout) is never written here: keep it finite.
        b.zero_vector(&mut v[2])?;
        let out = v.pop().unwrap();
        let middle = v.pop().unwrap();
        let route = v.pop().unwrap();
        Ok(Some(Self { route, middle, out }))
    }
}
/// Session owns all causal state, scratch and trace files. Only weights are shared.
pub struct Gpu<'b, 'a, 'w> {
    b: &'b Backend<'a>,
    arena: &'w Arena<'a>,
    hash: PleHash,
    lexical: DeviceVector<'a>,
    residual: DeviceVector<'a>,
    hc: HcScratch<'a>,
    hc_state: DeviceVector<'a>,
    hc_previous: DeviceVector<'a>,
    ah: Vec<Hc<'w>>,
    mh: Vec<Hc<'w>>,
    mixer: Hc<'w>,
    ple: Ple<'a, 'w>,
    head: Head<'a, 'w>,
    gs: GdnScratch<'a>,
    qs: QsaScratch<'a>,
    g: Vec<Option<Gdn<'a, 'w>>>,
    q: Vec<Option<Qsa<'a, 'w>>>,
    norms: Vec<DeviceVector<'a>>,
    block: DeviceVector<'a>,
    buffers: Vec<DeviceVector<'a>>,
    bw: Option<BwScratch<'a>>,
    p: NextMoe,
    routes: Vec<(u32, f32)>,
    /// Prompt-chunk buffers (next_chunk.rs), allocated by the first chunk.
    chunk: Option<chunk::Chunk<'a>>,
    rank: usize,
    world: usize,
    capacity: usize,
    position: usize,
    ep2: bool,
    shared_output: Vec<f32>,
    tp: usize,
    exchange: Box<dyn FnMut(&mut [u8], usize) -> Result<()> + 'w>,
    pub histogram: Vec<Vec<u64>>,
    trace: Option<BufWriter<File>>,
    routes_file: Option<BufWriter<File>>,
}
impl<'b, 'a, 'w> Gpu<'b, 'a, 'w> {
    pub fn new(
        b: &'b Backend<'a>,
        arena: &'w Arena<'a>,
        c: &ShardCheckpoint,
        capacity: usize,
        exchange: Box<dyn FnMut(&mut [u8], usize) -> Result<()> + 'w>,
    ) -> Result<Self> {
        let w = &arena.weights;
        let ints = |s: &str| -> Result<Vec<i64>> {
            Ok(raw(c, &format!("{PLE_PREFIX}.{s}"))?
                .chunks_exact(8)
                .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
                .collect())
        };
        let hash = PleHash::new(
            ints("layer_multipliers")?,
            ints("ngram_heads_vocab_sizes")?,
            ints("ngram_heads_offsets")?,
            248044,
        );
        let hc = |role: &str| -> Result<Vec<Hc<'w>>> {
            (0..48)
                .map(|li| {
                    Hc::load_next(
                        w,
                        &format!("model.language_model.layers.{li}.{role}_hyper_connection"),
                    )
                })
                .collect()
        };
        let mut norms = Vec::new();
        for li in 0..48 {
            let p = format!("model.language_model.layers.{li}");
            let names = if li % 4 == 3 {
                vec![
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "self_attn.indexer.q_layernorm.weight",
                    "self_attn.indexer.k_layernorm.weight",
                ]
            } else {
                vec!["linear_attn.norm.weight"]
            };
            let values = names
                .iter()
                .map(|n| w.values_f32(&format!("{p}.{n}")))
                .collect::<Result<Vec<_>>>()?
                .concat();
            let mut v = b.vector_arena(&[values.len()])?.pop().unwrap();
            b.write_vector(&mut v, &values)?;
            norms.push(v);
        }
        let buffers = b.vector_arena(&[
            512,
            10,
            10,
            512 * 3,
            1,
            10,
            10,
            10 * H / 2,
            10 * 1280 / 2,
            10 * 640 / 2,
            10 * H / 2,
            H,
            1,
            H,
            1280,
            640,
        ])?;
        let a = buffers
            .iter()
            .map(|v| v.device_address())
            .collect::<Vec<_>>();
        let p = NextMoe {
            logits: a[0],
            ids: a[1],
            weights: a[2],
            tiles: a[3],
            tile_count: a[4],
            rows: a[5],
            positions: a[6],
            gathered: a[7],
            gate_up: a[8],
            middle: a[9],
            down: a[10],
            shared_down: a[11],
            shared_gate: a[12],
            combined: a[13],
            tokens: 1,
            ..Default::default()
        };
        Ok(Self {
            b,
            arena,
            hash,
            lexical: b.vector(H)?,
            residual: b.vector(W)?,
            hc: HcScratch::new(b)?,
            hc_state: b.vector(W)?,
            hc_previous: b.vector(4)?,
            ah: hc("attn")?,
            mh: hc("mlp")?,
            mixer: Hc::mixer_next(w, "model.language_model.hyper_connection_mixer")?,
            ple: Ple::load(b, w)?,
            head: Head::load(b, w)?,
            gs: GdnScratch::new(b, w, 1, &[0.; 128])?,
            qs: QsaScratch::new(b, &w.plan, 1, capacity)?,
            g: (0..48).map(|_| None).collect(),
            q: (0..48).map(|_| None).collect(),
            norms,
            block: b.vector(H)?,
            buffers,
            bw: BwScratch::new(b, arena)?,
            p,
            routes: Vec::new(),
            chunk: None,
            ep2: c.plan.ep2,
            shared_output: vec![0.; H],
            rank: c.plan.rank,
            world: c.plan.world,
            capacity,
            position: 0,
            tp: c.plan.expert_tp,
            exchange,
            histogram: vec![vec![0; c.plan.world / c.plan.expert_tp]; 48],
            trace: None,
            routes_file: None,
        })
    }
    pub fn capture(&mut self, trace: File, routes: File) {
        self.trace = Some(BufWriter::with_capacity(1 << 20, trace));
        self.routes_file = Some(BufWriter::new(routes));
    }
    // Full arrays, little-endian F32; no first16 shortcut. Name and length precede every row.
    fn record(&mut self, name: &str, v: &[f32]) -> Result<()> {
        if let Some(f) = &mut self.trace {
            writeln!(f, "{} {} {}", self.position, name, v.len()).map_err(|e| e.to_string())?;
            for x in v {
                f.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    fn projections(&self, li: usize, kind: &str, names: [&str; 5]) -> Result<[Matrix; 5]> {
        names
            .map(|n| {
                self.arena
                    .matrix(&format!("model.language_model.layers.{li}.{kind}.{n}"))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| "projection count".into())
    }
    fn shared(&mut self, li: usize) -> Result<Vec<f32>> {
        self.shared_launch(li)?;
        let gate = self.b.read_vector(&self.buffers[14])?;
        let up = self.b.read_vector(&self.buffers[15])?;
        self.shared_finish(li, &gate, &up)
    }
    /// The shared expert's gate/up projections (buffers 14/15); shared_finish reads them.
    fn shared_launch(&mut self, li: usize) -> Result<()> {
        let b = self.b;
        let x = self.hc.mixed().device_address();
        let p = format!("model.language_model.layers.{li}.mlp");
        let a = self.arena.matrix(&format!("{p}.shared_expert.gate_proj"))?;
        let u = self.arena.matrix(&format!("{p}.shared_expert.up_proj"))?;
        next_native::project(
            b,
            &[
                (a, x, self.buffers[14].device_address()),
                (u, x, self.buffers[15].device_address()),
            ],
            1,
            false,
            false,
        )
    }
    /// SwiGLU of the read gate/up rows, the down projection and the gated scale (one read).
    fn shared_finish(&mut self, li: usize, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        let b = self.b;
        let x = self.hc.mixed().device_address();
        let p = format!("model.language_model.layers.{li}.mlp");
        let width = if self.ep2 { 640 } else { 640 / self.world };
        let mid = gate[..width]
            .iter()
            .zip(&up[..width])
            .map(|(&g, &u)| refcpu::silu(g) * u)
            .collect::<Vec<_>>();
        b.write_vector(&mut self.buffers[14].slice(0..width)?, &mid)?;
        next_native::project(
            b,
            &[(
                self.arena.matrix(&format!("{p}.shared_expert.down_proj"))?,
                self.buffers[14].device_address(),
                self.p.shared_down,
            )],
            1,
            false,
            false,
        )?;
        blocks::bf16_rows(
            b,
            self.arena
                .weights
                .device_address(&format!("{p}.shared_expert_gate.weight"))?,
            x,
            self.p.shared_gate,
            1,
            H,
            1,
            1,
        )?;
        // Shared down row and gate scalar: one stream wait (buffers 11 and 12 are adjacent).
        let read = b.read_vectors_values(&self.buffers[11..13])?;
        if !read.iter().all(|v| all_finite(v)) {
            return Err("nonfinite device output".into());
        }
        let scale = refcpu::sigmoid(read[1][0]);
        Ok(read[0].iter().map(|&v| v * scale).collect())
    }
    fn prepare_attention(&mut self, li: usize) -> Result<()> {
        let b = self.b;
        if li == 1 {
            // Trace rows only: a read-back is a stream wait, so skip it untraced.
            if self.trace.is_some() {
                let e = b.read_vector(&self.lexical)?;
                self.record("layer1.ple_e", &e)?;
            }
            self.ple.project(b, &mut self.residual, &self.lexical)?;
            if self.trace.is_some() {
                let v = b.read_vector(&self.residual)?;
                self.record("layer1.ple", &v)?;
            }
        }
        self.ah[li].mix_native(
            b,
            &mut self.hc,
            &mut self.hc_state,
            &mut self.hc_previous,
            &mut self.residual,
        )?;
        Ok(())
    }
    fn route(&mut self, li: usize) -> Result<()> {
        let b = self.b;
        for i in [8, 9, 10, 14, 15] {
            b.zero_vector(&mut self.buffers[i])?;
        }
        self.mh[li].mix_native(
            b,
            &mut self.hc,
            &mut self.hc_state,
            &mut self.hc_previous,
            &mut self.residual,
        )?;
        self.p.input = self.hc.mixed().device_address();
        self.p.experts = self.arena.tables[li].device_address();
        let mut dense = NextDense {
            weight: self
                .arena
                .weights
                .device_address(&format!("model.language_model.layers.{li}.mlp.gate.weight"))?,
            input: self.p.input,
            output: self.p.logits,
            rows: 512,
            cols: H as u32,
            tokens: 1,
            packed: 1,
            ..Default::default()
        };
        launch(b, "flash_next_dense", 384, 32, &mut dense)?;
        launch(b, "flash_next_route", 1, 256, &mut self.p)?;
        // Expert IDs and weights: one stream wait (buffers 1 and 2 are adjacent).
        let read = b.read_vectors_values(&self.buffers[1..3])?;
        if !read.iter().all(|v| all_finite(v)) {
            return Err("nonfinite device output".into());
        }
        self.routes = read[0]
            .iter()
            .zip(&read[1])
            .map(|(i, &w)| (i.to_bits(), w))
            .collect();
        self.routes.sort_by_key(|p| p.0);
        if self.routes.iter().any(|p| p.0 >= 512 || !p.1.is_finite()) {
            return Err("invalid GPU route".into());
        }
        Ok(())
    }
    /// Owned experts of one token through fp4_bw (see BwScratch); adds each expert's BF16
    /// down row times its weight into `out` in `selected` order, exactly as the routed path.
    /// The shared expert's gate/up projections are launched behind the experts and read with
    /// their rows in one stream wait; returns those gate/up rows for shared_finish.
    /// Ok(None): not applicable here (expert TP 2, switch off or entries missing).
    fn moe_bw(&mut self, li: usize, selected: &[u32], out: &mut [f32]) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        use spark_gpu_probe::fp8_abi::GroupedParams;
        let b = self.b;
        if self.tp != 1 {
            return Ok(None);
        }
        let Some(bw) = self.bw.as_mut() else { return Ok(None) };
        let (words, route) = bw_route(&self.routes, bw.route.device_address())?;
        b.write_vector(
            &mut bw.route,
            &words.iter().copied().map(f32::from_bits).collect::<Vec<_>>(),
        )?;
        let tasks = spark_gpu_probe::compact::tasks(1) as u32;
        let mut p = GroupedParams {
            route,
            weights: self.arena.bw_tables[li].device_address(),
            input: self.hc.mixed().device_address(),
            input_scales: 0,
            output: bw.middle.device_address(),
            cols: H as u32,
            activation_quantization: 0,
            output_shard: 0,
        };
        unsafe {
            b.launch_flash_grid("fp4_bw_gate_c1", 40, tasks, 1, 64, 0,
                &mut [(&mut p as *mut GroupedParams).cast()])?;
            p.input = bw.middle.device_address();
            p.output = bw.out.device_address();
            b.launch_flash_grid("fp4_bw_down", 160, tasks, 1, 64, 0,
                &mut [(&mut p as *mut GroupedParams).cast()])?;
        }
        let rows = bw.out.slice(0..bw.out.len())?;
        self.shared_launch(li)?;
        let (gate, up) = (&self.buffers[14], &self.buffers[15]);
        let mut read = b.read_vectors_values(&[rows, gate.slice(0..gate.len())?, up.slice(0..up.len())?])?;
        if !read.iter().all(|v| all_finite(v)) {
            return Err("nonfinite device output".into());
        }
        let (up, gate, down) = (read.pop().unwrap(), read.pop().unwrap(), read.pop().unwrap());
        let pair_to_row = &words[BW_TABLES + 1025..BW_TABLES + 1035];
        for &id in selected {
            let pair = self.routes.iter().position(|p| p.0 == id).ok_or("fp4_bw route mismatch")?;
            let (slot, wt) = (pair_to_row[pair] as usize, self.routes[pair].1);
            if slot >= 10 || down.len() < BW_SLOTS * BW_SLOT_WORDS {
                return Err("fp4_bw output slot outside its buffer".into());
            }
            for i in 0..H {
                let word = down[slot * BW_SLOT_WORDS + i / 2].to_bits();
                out[i] += nvfp4::bf16_to_f32((word >> ((i % 2) * 16)) as u16) * wt;
            }
        }
        Ok(Some((gate, up)))
    }
}
impl RankKernels for Gpu<'_, '_, '_> {
    fn front(
        &mut self,
        c: &ShardCheckpoint,
        token: u32,
        position: usize,
        out: &mut [f32],
    ) -> Result<()> {
        self.position = position;
        let rows = self.hash.advance(token);
        let embed = "model.language_model.embed_tokens.weight";
        let t = &c.plan.tensors[embed];
        if let Some(at) = t
            .selection
            .local_byte_offset(&t.tensor, token as u64 * (H * 2) as u64)
        {
            let mut v = vec![0; H * 2];
            c.read_local(embed, at, &mut v).map_err(|e| e.to_string())?;
            for (o, v) in out[..H].iter_mut().zip(v.chunks_exact(2)) {
                *o = nvfp4::bf16_to_f32(u16::from_le_bytes(v.try_into().unwrap()));
            }
        }
        for (h, row) in rows.into_iter().enumerate() {
            let m = format!("{PLE_PREFIX}.ngram_embedding.shard_{}", row / PLE_PART_ROWS);
            let n = format!("{m}.weight");
            let t = &c.plan.tensors[&n];
            if let Some(at) = t
                .selection
                .local_byte_offset(&t.tensor, row % PLE_PART_ROWS * 80)
            {
                let mut w = [0; 80];
                let mut s = [0; 10];
                c.read_local(&n, at, &mut w).map_err(|e| e.to_string())?;
                let n = format!("{m}.weight_scale");
                let t = &c.plan.tensors[&n];
                let at = t
                    .selection
                    .local_byte_offset(&t.tensor, row % PLE_PART_ROWS * 10)
                    .ok_or("PLE scale ownership mismatch")?;
                c.read_local(&n, at, &mut s).map_err(|e| e.to_string())?;
                let global = self.arena.ple_global;
                refcpu::ple_row_nvfp4(&w, &s, global, &mut out[H + h * 160..H + (h + 1) * 160])
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    fn embed(&mut self, rows: &[f32]) -> Result<()> {
        let x = rows[..H].repeat(4);
        self.record("embed", &x)?;
        self.b.write_vector(&mut self.residual, &x)?;
        self.b.write_vector(&mut self.lexical, &rows[H..])
    }
    fn attention(&mut self, li: usize, _: usize, out: &mut [f32]) -> Result<()> {
        let b = self.b;
        self.prepare_attention(li)?;
        if li % 4 == 3 {
            self.qs.use_norm(&self.norms[li])?;
            let m = self.projections(
                li,
                "self_attn",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "indexer.index_qk_proj",
                    "o_proj",
                ],
            )?;
            if self.q[li].is_none() {
                let mut q = Qsa::load_next(
                    b,
                    &self.arena.weights,
                    li,
                    self.capacity,
                    m.map(|m| m.weights),
                    None,
                )?;
                q.use_native(m)?;
                self.q[li] = Some(q);
            }
            let q = self.q[li].as_mut().ok_or("missing QSA owner")?;
            q.project(b, &mut self.qs, self.hc.mixed(), 1)?;
            q.attend(b, &mut self.qs)?;
            out.copy_from_slice(&b.read_vector(self.qs.output())?);
        } else {
            self.gs.copy_norm(b, &self.norms[li])?;
            let m = self.projections(
                li,
                "linear_attn",
                [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_a",
                    "in_proj_b",
                    "out_proj",
                ],
            )?;
            if self.g[li].is_none() {
                self.g[li] = Some(Gdn::load_next(
                    b,
                    &self.arena.weights,
                    li,
                    m.map(|m| m.weights),
                    None,
                )?);
            }
            let g = self.g[li].as_mut().ok_or("missing GDN owner")?;
            g.run_next_native(b, &mut self.gs, self.hc.mixed(), m)?;
            out.copy_from_slice(&b.read_vector(self.gs.output())?);
        }
        Ok(())
    }
    fn attention_rows(rows: &mut [(&mut Self, usize, &mut [f32])], li: usize) -> Vec<Result<()>> {
        if rows.is_empty() {
            return Vec::new();
        }
        if li % 4 == 3 {
            return rows
                .iter_mut()
                .map(|(s, p, out)| s.attention(li, *p, out))
                .collect();
        }
        let result = (|| {
            let b = rows[0].0.b;
            let mut matrices = Vec::new();
            for (s, _, _) in rows.iter_mut() {
                s.prepare_attention(li)?;
                s.gs.copy_norm(b, &s.norms[li])?;
                let m = s.projections(
                    li,
                    "linear_attn",
                    [
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_a",
                        "in_proj_b",
                        "out_proj",
                    ],
                )?;
                if s.g[li].is_none() {
                    s.g[li] = Some(Gdn::load_next(
                        b,
                        &s.arena.weights,
                        li,
                        m.map(|m| m.weights),
                        None,
                    )?);
                }
                matrices.push(m);
            }
            spark_engine::flash::next_batched::native_gdn_rows(
                b,
                &mut rows
                    .iter_mut()
                    .zip(matrices)
                    .map(|((s, _, _), m)| {
                        Ok((
                            s.g[li].as_mut().ok_or("missing GDN owner")?,
                            &mut s.gs,
                            s.hc.mixed(),
                            m,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            )?;
            for (s, _, out) in rows.iter_mut() {
                out.copy_from_slice(&b.read_vector(s.gs.output())?);
            }
            Ok(())
        })();
        vec![result; rows.len()]
    }
    fn inject_attention(&mut self, li: usize, v: &[f32]) -> Result<()> {
        self.record(&format!("layer{li}.attn"), v)?;
        self.b.write_vector(&mut self.block, v)?;
        self.ah[li].combine(self.b, &mut self.hc, &mut self.residual, &self.block)
    }
    fn moe(&mut self, li: usize, out: &mut [f32]) -> Result<()> {
        self.route(li)?;
        let b = self.b;
        let mut tiles = Vec::new();
        let mut selected = Vec::new();
        for &(id, _) in &self.routes {
            let owner = (id as usize * (self.world / self.tp)) / 512;
            self.histogram[li][owner] += 1;
            if owner == self.rank / self.tp {
                let at = selected.len();
                selected.push(id);
                tiles.extend([id as u32, at as u32, 1]);
            }
        }
        if let Some(f) = &mut self.routes_file {
            writeln!(
                f,
                "{}",
                engine_model::flash::dump::moe_line(
                    li,
                    &self
                        .routes
                        .iter()
                        .map(|&(i, w)| (i as usize, w))
                        .collect::<Vec<_>>()
                )
            )
            .map_err(|e| e.to_string())?;
        }
        let bw_shared = self.moe_bw(li, &selected, out)?;
        if bw_shared.is_none() {
        if !tiles.is_empty() {
            b.write_vector(
                &mut self.buffers[3].slice(0..tiles.len())?,
                &tiles
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
            )?;
        }
        b.write_vector(
            &mut self.buffers[4],
            &[f32::from_bits(selected.len() as u32)],
        )?;
        b.zero_vector(&mut self.buffers[5])?;
        launch(
            b,
            "flash_next_gather",
            (10 * H).div_ceil(256),
            256,
            &mut self.p,
        )?;
        let mut p = NextRouted {
            experts: self.p.experts,
            input: self.p.gathered,
            output: self.p.gate_up,
            tiles: self.p.tiles,
            tile_count: self.p.tile_count,
            rows: 1280,
            cols: H as u32,
            ..Default::default()
        };
        if self.tp == 1 {
            launch(b, "flash_next_routed", 192, 256, &mut p)?;
            let mut a = NextActivation {
                input: self.p.gate_up,
                output: self.p.middle,
                rows: 10,
                reserved: 0,
            };
            launch(b, "flash_next_routed_activate", 25, 256, &mut a)?;
        } else {
            p.rows = 320;
            launch(b, "flash_next_routed", 192, 256, &mut p)?;
            p.experts = self.arena.up_tables[li].device_address();
            p.output = self.buffers[15].device_address();
            // Up needs ten compact rows; reuse the larger down buffer instead.
            p.output = self.p.down;
            launch(b, "flash_next_routed", 192, 256, &mut p)?;
            let gate = b.read_vector(&self.buffers[8])?;
            let up = b.read_vector(&self.buffers[10])?;
            let mut wire = vec![0u8; self.world * 10 * 640];
            for (row, _) in selected.iter().enumerate() {
                for i in 0..320 {
                    let read = |v: &[f32]| {
                        nvfp4::bf16_to_f32(
                            (v[(row * 320 + i) / 2].to_bits() >> ((i % 2) * 16)) as u16,
                        )
                    };
                    let mid = nvfp4::swiglu_ref(read(&gate), read(&up));
                    let at = self.rank * 6400 + row * 640 + i * 2;
                    wire[at..at + 2].copy_from_slice(&mid.to_le_bytes());
                }
            }
            (self.exchange)(&mut wire, 6400)?;
            let mut middle = vec![0u8; 10 * 640 * 2];
            let group = self.rank / 2 * 2;
            for row in 0..selected.len() {
                for peer in 0..2 {
                    middle[row * 1280 + peer * 640..row * 1280 + (peer + 1) * 640].copy_from_slice(
                        &wire[(group + peer) * 6400 + row * 640
                            ..(group + peer) * 6400 + (row + 1) * 640],
                    );
                }
            }
            b.write_vector_bytes(&mut self.buffers[9], &middle)?;
        }
        p.experts = self.p.experts;
        p.input = self.p.middle;
        p.output = self.p.down;
        p.rows = (H / self.tp) as u32;
        p.cols = 640;
        p.down = 1;
        launch(b, "flash_next_routed", 384, 256, &mut p)?;
        let down = b.read_vector(&self.buffers[10])?;
        let width = H / self.tp;
        let start = (self.rank % self.tp) * width;
        for (row, id) in selected.into_iter().enumerate() {
            let wt = self.routes.iter().find(|p| p.0 == id).unwrap().1;
            for i in 0..width {
                let word = down[(row * width + i) / 2].to_bits();
                out[start + i] += nvfp4::bf16_to_f32((word >> ((i % 2) * 16)) as u16) * wt;
            }
        }
        }
        // EP2 communicates only routed sums; the identical shared expert is
        // added once on each rank after the ordered reduction.
        let shared = match bw_shared {
            Some((gate, up)) => self.shared_finish(li, &gate, &up)?,
            None => self.shared(li)?,
        };
        if self.ep2 {
            self.shared_output = shared;
        } else {
            out[H..2 * H].copy_from_slice(&shared);
        }
        Ok(())
    }
    fn moe_rows(rows: &mut [(&mut Self, &mut [f32])], li: usize) -> Vec<Result<()>> {
        if rows.is_empty() {
            return Vec::new();
        }
        // With fp4_bw on, each row runs the one-token whole-expert MoE (main's
        // C1 expert kernels): the same BF16 expert rows and fold as the routed
        // rows tile (C8 == C1 bytes), at a fraction of its cost for M <= 8.
        if rows[0].0.tp != 1 || rows.iter().all(|(s, _)| s.bw.is_some()) {
            return rows.iter_mut().map(|(s, out)| s.moe(li, out)).collect();
        }
        let result = (|| {
            for (s, _) in rows.iter_mut() {
                s.route(li)?;
                for &(id, _) in &s.routes {
                    s.histogram[li][id as usize * s.world / 512] += 1;
                }
            }
            let b = rows[0].0.b;
            let arena = rows[0].0.arena;
            let mut work = arena
                .rows_work
                .try_borrow_mut()
                .map_err(|_| "rank rows scratch is busy")?;
            let partials = work.experts(
                b,
                rows[0].0.arena.tables[li].device_address(),
                crate::rows(512, rows[0].0.rank, rows[0].0.world),
                &rows.iter().map(|(s, _)| s.hc.mixed()).collect::<Vec<_>>(),
                &rows
                    .iter()
                    .map(|(s, _)| s.routes.clone())
                    .collect::<Vec<_>>(),
            );
            drop(work);
            for ((s, out), v) in rows.iter_mut().zip(partials?) {
                out[..H].copy_from_slice(&v);
                let shared = s.shared(li)?;
                if s.ep2 {
                    s.shared_output = shared;
                } else {
                    out[H..2 * H].copy_from_slice(&shared);
                }
                if let Some(f) = &mut s.routes_file {
                    writeln!(
                        f,
                        "{}",
                        engine_model::flash::dump::moe_line(
                            li,
                            &s.routes
                                .iter()
                                .map(|&(i, w)| (i as usize, w))
                                .collect::<Vec<_>>()
                        )
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        })();
        vec![result; rows.len()]
    }
    fn routing(&self) -> &[(u32, f32)] {
        &self.routes
    }
    fn chunk_rows(&self) -> usize {
        self.chunk_capacity()
    }
    fn embed_chunk(&mut self, rows: &[f32], n: usize) -> Result<()> {
        self.embed_rows(rows, n)
    }
    fn attention_chunk(&mut self, li: usize, out: &mut [f32], n: usize) -> Result<()> {
        self.attention_rows_chunk(li, out, n)
    }
    fn inject_attention_chunk(&mut self, li: usize, v: &[f32], n: usize) -> Result<()> {
        self.inject_rows(li, v, n, true)
    }
    fn moe_chunk(&mut self, li: usize, out: &mut [f32], n: usize) -> Result<()> {
        self.moe_rows_chunk(li, out, n)
    }
    fn chunk_routing(&self, row: usize) -> &[(u32, f32)] {
        self.chunk_route(row)
    }
    fn inject_moe_chunk(&mut self, li: usize, v: &[f32], n: usize) -> Result<()> {
        self.inject_rows(li, v, n, false)
    }
    fn finish_chunk(&mut self, n: usize) -> Result<()> {
        self.finish_rows(n)
    }
    fn cohort_rows(&self) -> usize {
        self.chunk_capacity()
    }
    fn cohort_attention(rows: &mut [&mut Self], li: usize, out: &mut [f32]) -> Result<()> {
        Self::cohort_attention_rows(rows, li, out)
    }
    fn cohort_finish(rows: &mut [&mut Self]) -> Result<()> {
        Self::cohort_hand_back(rows)
    }
    fn separate_shared(&self) -> bool {
        !self.ep2
    }
    fn add_replicated_shared(&mut self, v: &mut [f32]) -> Result<()> {
        for (x, s) in v.iter_mut().zip(&self.shared_output) {
            *x += s;
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err("EP2 shared addition overflow".into());
        }
        Ok(())
    }
    fn inject_moe(&mut self, li: usize, v: &[f32]) -> Result<()> {
        self.b.write_vector(&mut self.block, v)?;
        self.mh[li].combine(self.b, &mut self.hc, &mut self.residual, &self.block)?;
        if self.trace.is_none() {
            return Ok(());
        }
        let v = self.b.read_vector(&self.residual)?;
        self.record(&format!("layer{li}.x"), &v)
    }
    fn logits(&mut self, _: Range<u64>, out: &mut [f32]) -> Result<()> {
        self.mixer.mix_native(
            self.b,
            &mut self.hc,
            &mut self.hc_state,
            &mut self.hc_previous,
            &mut self.residual,
        )?;
        if self.trace.is_some() {
            let v = self.b.read_vector(self.hc.mixed())?;
            self.record("final", &v)?;
        }
        self.head.project(self.b, self.hc.mixed())?;
        out.copy_from_slice(&self.head.logits(self.b)?);
        self.record("logits", out)?;
        if let Some(f) = &mut self.trace {
            f.flush().map_err(|e| e.to_string())?;
        }
        if let Some(f) = &mut self.routes_file {
            f.flush().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.hash.reset();
        self.ple.reset(self.b)?;
        for g in self.g.iter_mut().flatten() {
            g.reset(self.b)?;
        }
        for q in self.q.iter_mut().flatten() {
            q.reset();
        }
        self.position = 0;
        Ok(())
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use engine_model::{
        flash::{
            mixed::{Mixed, Storage},
            shard::{Placement, TensorShard},
        },
        safetensors::{Dtype, Tensor},
        shapes::{Part, Place, Role, Store},
    };
    #[test]
    fn real_f8_inventory_excludes_ple_scales() {
        // Names/dtypes/shapes extracted from the qualified checkpoint headers;
        // storage and residency isolate each part of the selection contract.
        let q = Mixed {
            ignore: vec![],
            layers: BTreeMap::from([("model".into(), Storage::Mxfp8)]),
        };
        let (mut weights, mut scales) = (0, 0);
        for row in include_str!("../tests/fixtures/mxfp8-real.tsv").lines() {
            let fields = row.split('\t').collect::<Vec<_>>();
            assert_eq!(fields[1], "F8_E4M3");
            let name = fields[0];
            let scale = name.ends_with(".weight_scale");
            let shape = fields[2]
                .split(',')
                .map(|v| v.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            let t = TensorShard {
                file: "model.safetensors".into(),
                tensor: Tensor {
                    dtype: Dtype::F8E4M3,
                    end: shape.iter().product(),
                    start: 0,
                    shape,
                },
                selection: Selection::All,
                role: Role {
                    place: Place::Layer(1),
                    part: if scale {
                        Part::Lexical { part: 0 }
                    } else {
                        Part::Output
                    },
                    store: if scale {
                        Store::BlockScale
                    } else {
                        Store::Weight
                    },
                },
                placement: if scale {
                    Placement::Disk
                } else {
                    Placement::Resident
                },
            };
            assert_eq!(native_projection(name, &t, &q).is_some(), !scale);
            if scale {
                scales += 1;
                let mut resident = t.clone();
                resident.placement = Placement::Resident;
                assert!(native_projection(name, &resident, &q).is_none());
            } else {
                weights += 1;
                let mut disk = t.clone();
                disk.placement = Placement::Disk;
                assert!(native_projection(name, &disk, &q).is_none());
                let plain = Mixed {
                    ignore: vec!["*".into()],
                    layers: q.layers.clone(),
                };
                assert!(native_projection(name, &t, &plain).is_none());
            }
        }
        assert_eq!((weights, scales), (384, 128));
    }
}
