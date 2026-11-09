//! Rank chunk prefill (world 2/4): up to 128 consecutive prompt tokens of one session run
//! layer by layer. Every launch is a kernel the one-token path already runs, in its
//! row-batched form with the same per-row arithmetic: HC pre/up/add with rows in grid Y,
//! MXFP8 projections in groups of <= 16 rows (no M-dependent reduction), flash_next_dense /
//! flash_next_route over token rows, and fp4_bw over the chunk's compact routes (MMA output
//! row i depends only on A row i). The causal parts stay one token at a time in prompt
//! order: PLE (layer 1, rows in groups of 8: the M-invariant decode BF16 rows and ordered
//! conv), the GDN recurrence (a cold first row through Gdn::recur_row, cached
//! rows through Gdn::scan_rows: the ordered prompt scan, per row recur(1)'s words) and
//! QSA attention (Qsa::attend_rows over groups of 32 rows: the one-row cache/score/softmax/
//! value entries, rows by token; projections and o_proj run over all rows). Weights stream once per chunk, not per token.
use super::*;
use crate::next_bw::bw_route_rows;

/// Prompt rows per chunk.
pub(super) fn chunk_rows() -> usize { crate::next_http::CHUNK }
/// next_native::project over n row-major rows in groups of at most 16 (its M limit);
/// each row's output is the same as its own M-row launch.
fn project_rows(b: &Backend<'_>, matrices: &[(Matrix, u64, u64)], n: usize) -> Result<()> {
    for g in (0..n).step_by(16) {
        let at = |m: &Matrix, x: u64, y: u64| (*m, x + (g * m.cols as usize * 4) as u64, y + (g * m.rows as usize * 4) as u64);
        let group = matrices.iter().map(|(m, x, y)| at(m, *x, *y)).collect::<Vec<_>>();
        next_native::project(b, &group, (n - g).min(16), false, false)?;
    }
    Ok(())
}

pub(super) struct Chunk<'a> {
    capacity: usize,
    hc: HcScratch<'a>,
    gs: GdnScratch<'a>,
    qs: QsaScratch<'a>,
    v: Vec<DeviceVector<'a>>,
    routes: Vec<Vec<(u32, f32)>>,
}
// Row-major buffers in `v`.
const RESIDUAL: usize = 0;
const STATE: usize = 1;
const PREVIOUS: usize = 2;
const LEXICAL: usize = 3;
const MIXED: usize = 4;
const BLOCK: usize = 5;
const ATTENTION: usize = 6;
const LOGITS: usize = 7;
const IDS: usize = 8;
const WEIGHTS: usize = 9;
const ROUTE: usize = 10;
const MIDDLE: usize = 11;
const OUT: usize = 12;
const SHARED_GATE: usize = 13;
const SHARED_UP: usize = 14;
const SHARED_DOWN: usize = 15;
const SHARED_SCALE: usize = 16;
const QSA_QG: usize = 17;
const QSA_K: usize = 18;
const QSA_V: usize = 19;
const QSA_INDEX: usize = 20;
const QSA_GATED: usize = 21;
const GDN_ROWS: usize = 22;
/// Device MoE fold of a chunk: per row 21 plan words, then the n x H routed partials.
const FOLD_PLAN: usize = 23;
const FOLD: usize = 24;
/// RN BF16 normalized HC rows, owned only by the rank chunk.
const HC_PACKED: usize = 25;
/// Reused row-major BF16 HC-up weight scratch; never a per-layer cache.
const HC_UP_RAW: usize = 26;
/// Rows per QSA attention pass (< 64: the one-row entries) and per PLE pass (BF16 decode rows).
const QSA_ROWS: usize = 32;
const PLE_ROWS: usize = 8;
/// GDN workspace rows (GdnScratch holds <= 64): longer chunks run it in row pieces, in order.
const GDN_ROWS_MAX: usize = 64;
const _: () = assert!(RESIDUAL == 0 && STATE == 1 && PREVIOUS == 2, "Chunk::mix destructures v[..3]");
impl<'a> Chunk<'a> {
    /// `qsa`: this rank's QSA q/k/v/index projection rows and o_proj columns.
    pub(super) fn new(b: &Backend<'a>, w: &Weights<'_>, n: usize, qsa: [usize; 5], history: usize) -> Result<Self> {
        let mut v = b.vector_arena(&[
            n * W,
            n * W,
            4 * n,
            n * H,
            n * H,
            n * H,
            n * H,
            n * 512,
            n * 10,
            n * 10,
            n + 30 * n + spark_gpu_probe::compact::words(n),
            n * 10 * 640 / 2,
            n * 11 * H,
            n * 1280,
            n * 640,
            n * H,
            n,
            n * qsa[0],
            n * qsa[1],
            n * qsa[2],
            n * qsa[3],
            n * qsa[4],
            6,
            n * 21,
            n * H,
            n * W / 2,
            HcScratch::RANK_UP_WORDS,
        ])?;
        // fp4_bw_down writes only owned slots; the rest stay finite for the read-back.
        b.zero_vector(&mut v[OUT])?;
        Ok(Self {
            capacity: n,
            hc: HcScratch::with_tokens(b, n)?,
            gs: GdnScratch::new(b, w, GDN_ROWS_MAX.min(n), &[0.; 128])?,
            qs: QsaScratch::new(b, &w.plan, QSA_ROWS.min(n), history)?,
            v,
            routes: vec![Vec::new(); n],
        })
    }
    fn row(&self, i: usize, n: usize, r: usize) -> Result<DeviceVector<'a>> {
        self.v[i].slice(r * n..(r + 1) * n)
    }
    fn rows(&self, i: usize, n: usize) -> Result<DeviceVector<'a>> {
        self.v[i].slice(0..n)
    }
    fn at(&self, i: usize, words: usize) -> u64 {
        self.v[i].device_address() + (words * 4) as u64
    }
    /// mix_native for n rows into MIXED (residual/state and gate buffers swap as there).
    fn mix(&mut self, b: &Backend<'_>, hc: &Hc<'_>, n: usize) -> Result<()> {
        let output = self.at(MIXED, 0);
        let norm_bf16 = self.rows(HC_PACKED, n * W / 2)?.device_address();
        let up_rowmajor = self.rows(HC_UP_RAW, HcScratch::RANK_UP_WORDS)?.device_address();
        let [residual, state, previous, ..] = &mut self.v[..] else {
            return Err("prompt chunk buffers".into());
        };
        hc.mix_native_rank_rows(b, &mut self.hc, state, previous, residual, output, norm_bf16, up_rowmajor, n)
    }
}

impl<'a> Gpu<'_, 'a, '_> {
    pub(super) fn chunk_capacity(&self) -> usize {
        if self.bw.is_none() || self.tp != 1 || self.ep2 || self.trace.is_some() || !self.b.has_flash("fp4_bw_gate") {
            1
        } else {
            chunk_rows()
        }
    }
    fn take_chunk(&mut self, n: usize) -> Result<Chunk<'a>> {
        if n < 2 || n > self.chunk_capacity() {
            return Err("prompt chunk outside this rank's chunk form".into());
        }
        match self.chunk.take() {
            Some(c) if c.capacity >= n => Ok(c),
            _ => {
                // Layer owners first, in the single-token path's first-token order, so their
                // KV/recurrent storage keeps that path's placement; chunk buffers come after.
                for li in 0..48 {
                    self.owner(li)?;
                }
                let m = self.qsa_matrices(3)?;
                let widths = [m[0].rows, m[1].rows, m[2].rows, m[3].rows, m[4].cols].map(|v| v as usize);
                Chunk::new(self.b, &self.arena.weights, self.chunk_capacity(), widths, self.capacity)
            }
        }
    }
    fn qsa_matrices(&self, li: usize) -> Result<[Matrix; 5]> {
        self.projections(li, "self_attn", ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj"])
    }
    fn gdn_matrices(&self, li: usize) -> Result<[Matrix; 5]> {
        self.projections(li, "linear_attn", ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"])
    }
    /// Load layer li's QSA/GDN owner exactly as attention() does on its first token.
    fn owner(&mut self, li: usize) -> Result<()> {
        let b = self.b;
        if li % 4 == 3 {
            if self.q[li].is_none() {
                let m = self.qsa_matrices(li)?;
                let mut q = Qsa::load_next(b, &self.arena.weights, li, self.capacity, m.map(|m| m.weights), None)?;
                q.use_native(m)?;
                self.q[li] = Some(q);
            }
        } else if self.g[li].is_none() {
            let m = self.gdn_matrices(li)?;
            self.g[li] = Some(Gdn::load_next(b, &self.arena.weights, li, m.map(|m| m.weights), None)?);
        }
        Ok(())
    }
    fn with_chunk<T>(&mut self, n: usize, f: impl FnOnce(&mut Self, &mut Chunk<'a>) -> Result<T>) -> Result<T> {
        let mut c = self.take_chunk(n)?;
        let result = f(self, &mut c);
        self.chunk = Some(c);
        result
    }
    pub(super) fn embed_rows(&mut self, rows: &[f32], n: usize) -> Result<()> {
        self.with_chunk(n, |s, c| {
            let mut x = Vec::with_capacity(n * W);
            let mut lexical = Vec::with_capacity(n * H);
            for r in rows.as_chunks::<{ 2 * H }>().0.iter().take(n) {
                for _ in 0..4 {
                    x.extend_from_slice(&r[..H]);
                }
                lexical.extend_from_slice(&r[H..]);
            }
            s.b.write_vector(&mut c.rows(RESIDUAL, n * W)?, &x)?;
            s.b.write_vector(&mut c.rows(LEXICAL, n * H)?, &lexical)
        })
    }
    pub(super) fn attention_rows_chunk(&mut self, li: usize, out: &mut [f32], n: usize) -> Result<()> {
        self.with_chunk(n, |s, c| {
            let b = s.b;
            if li == 1 {
                for g in (0..n).step_by(PLE_ROWS) {
                    let rows = PLE_ROWS.min(n - g);
                    for r in 0..rows {
                        s.ple.set_lexical(b, &c.row(LEXICAL, H, g + r)?, r)?;
                    }
                    s.ple.project_rows(b, c.at(RESIDUAL, g * W), rows)?;
                }
            }
            let mixed = c.at(MIXED, 0);
            c.mix(b, &s.ah[li], n)?;
            if li % 4 == 3 {
                c.qs.use_norm(&s.norms[li])?;
                let m = s.qsa_matrices(li)?;
                s.owner(li)?;
                let q = s.q[li].as_mut().ok_or("missing QSA owner")?;
                let at = [QSA_QG, QSA_K, QSA_V, QSA_INDEX].map(|i| c.at(i, 0));
                project_rows(b, &(0..4).map(|i| (m[i], mixed, at[i])).collect::<Vec<_>>(), n)?;
                for g in (0..n).step_by(QSA_ROWS) {
                    let rows = [0, 1, 2, 3].map(|i| at[i] + (g * m[i].rows as usize * 4) as u64);
                    let gated = c.at(QSA_GATED, g * m[4].cols as usize);
                    q.attend_rows(b, &mut c.qs, rows, gated, QSA_ROWS.min(n - g))?;
                }
                project_rows(b, &[(m[4], c.at(QSA_GATED, 0), c.at(ATTENTION, 0))], n)?;
            } else {
                c.gs.use_norm(&s.norms[li])?;
                let m = s.gdn_matrices(li)?;
                s.owner(li)?;
                let g = s.g[li].as_mut().ok_or("missing GDN owner")?;
                let ys = c.gs.row_addresses();
                for piece in (0..n).step_by(GDN_ROWS_MAX) {
                    let rows = GDN_ROWS_MAX.min(n - piece);
                    let x = mixed + (piece * H * 4) as u64;
                    project_rows(b, &(0..4).map(|i| (m[i], x, ys[i])).collect::<Vec<_>>(), rows)?;
                    let first = usize::from(!g.is_cached());
                    if first == 1 {
                        g.recur_row(b, &c.gs, 0, rows)?;
                    }
                    if first < rows {
                        g.scan_rows(b, &c.gs, first, rows - first, &mut c.v[GDN_ROWS])?;
                    }
                    project_rows(b, &[(m[4], ys[4], c.at(ATTENTION, piece * H))], rows)?;
                }
            }
            out[..n * H].copy_from_slice(&b.read_vector(&c.rows(ATTENTION, n * H)?)?);
            Ok(())
        })
    }
    pub(super) fn inject_rows(&mut self, li: usize, v: &[f32], n: usize, attention: bool) -> Result<()> {
        self.with_chunk(n, |s, c| {
            let b = s.b;
            b.write_vector(&mut c.rows(BLOCK, n * H)?, &v[..n * H])?;
            let hc = if attention { &s.ah[li] } else { &s.mh[li] };
            let (low, high) = c.v.split_at_mut(BLOCK);
            hc.combine_rows(b, &mut c.hc, &mut low[RESIDUAL], &high[0], n)
        })
    }
    /// moe() for each row: the same routes, owned experts in ascending ID, the same fp4_bw
    /// expert rows and ordered fold, then the gated shared TP partial at [H..2H].
    pub(super) fn moe_rows_chunk(&mut self, li: usize, out: &mut [f32], n: usize) -> Result<()> {
        self.with_chunk(n, |s, c| {
            let b = s.b;
            b.zero_vector(&mut c.v[SHARED_GATE])?;
            b.zero_vector(&mut c.v[SHARED_UP])?;
            let mixed = c.at(MIXED, 0);
            c.mix(b, &s.mh[li], n)?;
            let mut dense = NextDense {
                weight: s.arena.weights.device_address(&format!("model.language_model.layers.{li}.mlp.gate.weight"))?,
                input: mixed,
                output: c.at(LOGITS, 0),
                rows: 512,
                cols: H as u32,
                tokens: n as u32,
                packed: 1,
                ..Default::default()
            };
            launch(b, "flash_next_dense", 384, 32, &mut dense)?;
            let mut route = NextMoe { logits: c.at(LOGITS, 0), ids: c.at(IDS, 0), weights: c.at(WEIGHTS, 0), tokens: n as u32, ..s.p };
            launch(b, "flash_next_route", n, 256, &mut route)?;
            let read = b.read_vectors_values(&[c.rows(IDS, n * 10)?, c.rows(WEIGHTS, n * 10)?])?;
            if !read.iter().all(|v| all_finite(v)) {
                return Err("nonfinite device output".into());
            }
            let owner = s.rank / s.tp;
            let mut selected = vec![Vec::new(); n];
            for r in 0..n {
                let mut routes = read[0][r * 10..(r + 1) * 10]
                    .iter()
                    .zip(&read[1][r * 10..(r + 1) * 10])
                    .map(|(i, &w)| (i.to_bits(), w))
                    .collect::<Vec<_>>();
                routes.sort_by_key(|p| p.0);
                if routes.iter().any(|p| p.0 >= 512 || !p.1.is_finite()) {
                    return Err("invalid GPU route".into());
                }
                for &(id, _) in &routes {
                    let o = (id as usize * (s.world / s.tp)) / 512;
                    s.histogram[li][o] += 1;
                    if o == owner {
                        selected[r].push(id);
                    }
                }
                c.routes[r] = routes;
            }
            // Owned experts of all rows through fp4_bw (grouped by expert, rows in token order).
            let (words, route) = bw_route_rows(&c.routes[..n], c.at(ROUTE, 0))?;
            b.write_vector(&mut c.rows(ROUTE, words.len())?, &words.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
            // An expert holds at most one row per token: n rows fit ceil(n / 16) M16 planes.
            let (tasks, planes) = (spark_gpu_probe::compact::tasks(n) as u32, n.div_ceil(16) as u32);
            let mut p = spark_gpu_probe::fp8_abi::GroupedParams {
                route,
                weights: s.arena.bw_tables[li].device_address(),
                input: mixed,
                input_scales: 0,
                output: c.at(MIDDLE, 0),
                cols: H as u32,
                activation_quantization: 0,
                output_shard: 0,
            };
            unsafe {
                b.launch_flash_grid("fp4_bw_gate", 40, tasks, planes, 64, 0, &mut [(&mut p as *mut spark_gpu_probe::fp8_abi::GroupedParams).cast()])?;
                p.input = c.at(MIDDLE, 0);
                p.output = c.at(OUT, 0);
                b.launch_flash_grid("fp4_bw_down", 160, tasks, planes, 64, 0, &mut [(&mut p as *mut spark_gpu_probe::fp8_abi::GroupedParams).cast()])?;
            }
            // Shared expert (TP partial) for all rows: native projections at M = n.
            let width = 640 / s.world;
            let pre = format!("model.language_model.layers.{li}.mlp");
            let gate = s.arena.matrix(&format!("{pre}.shared_expert.gate_proj"))?;
            let up = s.arena.matrix(&format!("{pre}.shared_expert.up_proj"))?;
            if gate.rows as usize != width || up.rows as usize != width {
                return Err("shared expert rows differ from the TP width".into());
            }
            project_rows(b, &[(gate, mixed, c.at(SHARED_GATE, 0)), (up, mixed, c.at(SHARED_UP, 0))], n)?;
            let shared_gate_w = s.arena.weights.device_address(&format!("{pre}.shared_expert_gate.weight"))?;
            // Independent scalar gates share the canonical decode K16 graph;
            // stay within its row width so this never selects a prompt GEMM.
            let group = spark_gpu_probe::kernel_params::BF16_ROWS_TOKENS;
            for r in (0..n).step_by(group) {
                blocks::bf16_rows(b, shared_gate_w, mixed + (r * H * 4) as u64,
                    c.at(SHARED_SCALE, r), 1, H, (n - r).min(group), 1)?;
            }
            // Each row's owned experts (ascending ID, as moe_bw): the device folds their BF16 rows
            // (flash_next_bw_fold_rows, the same f32 product then f32 add from 0 as the host fold),
            // so only the n x H partials are read back, not the n x 11 x H expert slots.
            let pair_base = n + 30 * n + 1025;
            let mut plan = vec![0u32; n * 21];
            for r in 0..n {
                plan[r * 21] = selected[r].len() as u32;
                for (q, &id) in selected[r].iter().enumerate() {
                    let pair = c.routes[r].iter().position(|p| p.0 == id).ok_or("fp4_bw route mismatch")?;
                    let (slot, wt) = (words[pair_base + r * 10 + pair] as usize, c.routes[r][pair].1);
                    if slot < r * 11 || slot >= r * 11 + 10 || q >= 10 {
                        return Err("fp4_bw output slot outside its row".into());
                    }
                    plan[r * 21 + 1 + 2 * q] = slot as u32;
                    plan[r * 21 + 2 + 2 * q] = wt.to_bits();
                }
            }
            b.write_vector(&mut c.rows(FOLD_PLAN, n * 21)?, &plan.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
            let mut fold = NextMoe { tiles: c.at(FOLD_PLAN, 0), down: c.at(OUT, 0), combined: c.at(FOLD, 0), tokens: n as u32, ..s.p };
            launch(b, "flash_next_bw_fold_rows", (n * H / 2).div_ceil(256), 256, &mut fold)?;
            let read = b.read_vectors_values(&[c.rows(SHARED_GATE, n * width)?, c.rows(SHARED_UP, n * width)?])?;
            if !read.iter().all(|v| all_finite(v)) {
                return Err("nonfinite device output".into());
            }
            let (g, u) = (&read[0], &read[1]);
            let mid = g.iter().zip(u).map(|(&g, &u)| refcpu::silu(g) * u).collect::<Vec<_>>();
            b.write_vector(&mut c.rows(SHARED_GATE, n * width)?, &mid)?;
            project_rows(b, &[(s.arena.matrix(&format!("{pre}.shared_expert.down_proj"))?, c.at(SHARED_GATE, 0), c.at(SHARED_DOWN, 0))], n)?;
            let read = b.read_vectors_values(&[c.rows(FOLD, n * H)?, c.rows(SHARED_DOWN, n * H)?, c.rows(SHARED_SCALE, n)?])?;
            if !read.iter().all(|v| all_finite(v)) {
                return Err("nonfinite device output".into());
            }
            for r in 0..n {
                out[r * 2 * H..r * 2 * H + H].copy_from_slice(&read[0][r * H..(r + 1) * H]);
            }
            let read = &read[1..];
            for r in 0..n {
                let scale = refcpu::sigmoid(read[1][r]);
                for (o, &v) in out[r * 2 * H + H..(r + 1) * 2 * H].iter_mut().zip(&read[0][r * H..(r + 1) * H]) {
                    *o = v * scale;
                }
            }
            Ok(())
        })
    }
    /// Cohort attention: row r is rows[r]'s decode row on rows[0]'s chunk buffers. HC mix and
    /// the projections run over all rows; PLE, the QSA attention and the GDN recurrence run on
    /// each row's own session with the one-token launches (Ple::project, attend_row, recur_row).
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    pub(super) fn cohort_attention_rows(rows: &mut [&mut Self], li: usize, out: &mut [f32]) -> Result<()> {
        let n = rows.len();
        let mut c = rows[0].take_chunk(n)?;
        let result = (|| {
            let b = rows[0].b;
            if li == 1 {
                for r in 0..n {
                    rows[r].ple.project(b, &mut c.row(RESIDUAL, W, r)?, &c.row(LEXICAL, H, r)?)?;
                }
            }
            let mixed = c.at(MIXED, 0);
            c.mix(b, &rows[0].ah[li], n)?;
            if li % 4 == 3 {
                c.qs.use_norm(&rows[0].norms[li])?;
                let m = rows[0].qsa_matrices(li)?;
                let at = [QSA_QG, QSA_K, QSA_V, QSA_INDEX].map(|i| c.at(i, 0));
                project_rows(b, &(0..4).map(|i| (m[i], mixed, at[i])).collect::<Vec<_>>(), n)?;
                for r in 0..n {
                    rows[r].owner(li)?;
                    let q = rows[r].q[li].as_mut().ok_or("missing QSA owner")?;
                    let row = [0, 1, 2, 3].map(|i| at[i] + (r * m[i].rows as usize * 4) as u64);
                    let gated = c.at(QSA_GATED, r * m[4].cols as usize);
                    q.attend_row(b, &mut c.qs, row, gated)?;
                }
                project_rows(b, &[(m[4], c.at(QSA_GATED, 0), c.at(ATTENTION, 0))], n)?;
            } else {
                c.gs.use_norm(&rows[0].norms[li])?;
                let m = rows[0].gdn_matrices(li)?;
                let ys = c.gs.row_addresses();
                project_rows(b, &(0..4).map(|i| (m[i], mixed, ys[i])).collect::<Vec<_>>(), n)?;
                for r in 0..n {
                    rows[r].owner(li)?;
                    rows[r].g[li].as_mut().ok_or("missing GDN owner")?.recur_row(b, &c.gs, r, n)?;
                }
                project_rows(b, &[(m[4], ys[4], c.at(ATTENTION, 0))], n)?;
            }
            out[..n * H].copy_from_slice(&b.read_vector(&c.rows(ATTENTION, n * H)?)?);
            Ok(())
        })();
        rows[0].chunk = Some(c);
        result
    }
    /// Row r's final residual -> rows[r].residual (its head mixes it as after a single step).
    pub(super) fn cohort_hand_back(rows: &mut [&mut Self]) -> Result<()> {
        let c = rows[0].chunk.take().ok_or("missing cohort rows")?;
        let result = (0..rows.len()).try_for_each(|r| {
            let b = rows[r].b;
            b.copy_range(&c.v[RESIDUAL], &mut rows[r].residual, r * W, 0, W)
        });
        rows[0].chunk = Some(c);
        result
    }
    pub(super) fn chunk_route(&self, r: usize) -> &[(u32, f32)] {
        self.chunk.as_ref().and_then(|c| c.routes.get(r)).map_or(&[], |r| &r[..])
    }
    /// The last row's residual becomes the single-token residual `logits` mixes.
    pub(super) fn finish_rows(&mut self, n: usize) -> Result<()> {
        let c = self.chunk.as_ref().ok_or("missing prompt chunk")?;
        self.b.copy_range(&c.v[RESIDUAL], &mut self.residual, (n - 1) * W, 0, W)
    }
}
