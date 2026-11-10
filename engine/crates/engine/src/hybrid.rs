//! One-Spark dense hybrid text model. BF16 or resident E4M3 weights are loaded once;
//! delta-net and gated full attention share the Flash execution kernels.
use engine_model::{flash::LayerKind, qwen3_5::{Config, PREFIX}, safetensors::{Limits, TensorFile}};
use spark_gpu_probe::{driver::{Backend, DecodeGraph, DeviceVector, Matrix, ExpertTable, MOE_ROUTING_ERROR}, load::{WeightRows, WeightShard}};
use crate::flash::{gdn::{Gdn, GdnScratch, GdnState, launch, Arg::{Ptr, U32, F32}}, qsa::{DenseScratch, DenseState}};
use std::{collections::BTreeMap, path::Path, time::{Duration, Instant}};
use std::os::unix::fs::FileExt;

type Result<T> = std::result::Result<T, String>;
const INITIAL_TILE: usize = 64;
const PREFIX_SNAPSHOTS: usize = memory::PREFIX_SNAPSHOTS;
const PREFIX_INTERVAL: usize = 2048;

fn launch_norm(b: &Backend<'_>, x: &DeviceVector<'_>, residual: Option<&DeviceVector<'_>>,
    weight: u64, y: &DeviceVector<'_>, model: &Model<'_, '_>, count: usize, order: Option<u32>, four: bool) -> Result<()> {
    let (c, (threads, order)) = (&model.config, linear::norm_launch(model.bound.is_some(), residual.is_some(), count, order, four));
    let norm = linear::NormBinding::new(c.hidden as usize, c.epsilon as f32)?;
    let elements = norm.elements(count)?;
    if x.len() < elements || y.len() < elements || residual.is_some_and(|r| r.len() < elements) {
        return Err("fixed norm operand extent".into());
    }
    let mut p = spark_gpu_probe::kernel_params::HybridNorm {
        input: x.device_address() as _, residual: residual.map_or(0, DeviceVector::device_address) as _,
        weight: weight as _, output: y.device_address() as _, width: c.hidden as u32,
        zero_centered: 1, epsilon: norm.epsilon, order,
    };
    let name = if residual.is_some() { "flash_residual_norm_bf16" } else { "flash_norm_decode" };
    unsafe { b.launch_flash_shared(name, count as u32, threads, 0,
        &mut [(&mut p as *mut spark_gpu_probe::kernel_params::HybridNorm).cast()]) }
}

fn prefill_tile(c: &Config) -> usize { if c.fp8 { 32768 } else { 4096 } }

fn launch_silu(b: &Backend<'_>, gate: &DeviceVector<'_>, up: &DeviceVector<'_>) -> Result<()> {
    launch_silu_precision(b, gate, up, 1)
}
fn launch_silu_precision(b: &Backend<'_>, gate: &DeviceVector<'_>, up: &DeviceVector<'_>, precision: u32) -> Result<()> {
    let mut p = spark_gpu_probe::kernel_params::HybridSilu {
        input: gate.device_address() as _, up: up.device_address() as _,
        count: gate.len() as u32, divisor: 1.0, precision,
    };
    unsafe { b.launch_flash_shared("flash_silu", gate.len().div_ceil(128) as u32, 128, 0,
        &mut [(&mut p.input as *mut *mut f32).cast(), (&mut p.up as *mut *const f32).cast(),
            (&mut p.count as *mut u32).cast(), (&mut p.divisor as *mut f32).cast(),
            (&mut p.precision as *mut u32).cast()]) }
}

fn session_lengths(c: &Config, tile: usize) -> [usize; 10] {
    let (h, intermediate) = (c.hidden as usize, c.intermediate as usize);
    let columns = h.max(c.linear_value_heads as usize * c.linear_value_dim as usize)
        .max(c.query_heads as usize * c.head_dim as usize);
    // Unconverted projections retain their BF16 packing scratch.
    let input_pack = (tile.div_ceil(16) * columns * 8).max(if c.fp8 { tile.max(16) * intermediate / 4 + tile * intermediate / 128 } else { 0 });
    let gate = if c.fp8 { tile * intermediate }
        else { (tile.div_ceil(16) * 16 * intermediate / 2).max(intermediate) };
    // Quantized weights are consumed resident; no promoted-weight allocation.
    let weights = 1;
    [tile * h, tile * h, tile * h, gate, if c.fp8 { tile * intermediate } else { intermediate },
        weights, input_pack, tile, h, c.vocab as usize]
}

#[path = "hybrid_decode.rs"]
pub mod decode;
#[path = "hybrid_prefill.rs"]
pub mod prefill;
#[path = "hybrid_mtp.rs"]
pub mod mtp;
#[path = "hybrid_moe.rs"]
mod moe;
pub mod plan;
pub use crate::arms;
pub mod weights;
pub mod linear;
mod graph;
pub mod tp;
mod forward;
pub mod schedule;
pub mod coverage;
pub mod memory;
pub mod cache;
pub mod mtp_contract;

pub struct Model<'a, 'b> {
    backend: &'b Backend<'a>,
    config: Config,
    weights: BTreeMap<String, Matrix<'a>>,
    experts: Vec<ExpertTable<'a>>,
    frequencies: DeviceVector<'a>,
    capacity: usize,
    weight_bytes: usize,
    mtp: bool,
    draft_options: crate::hybrid_draft::Options,
    arms: arms::Table,
    profile_plan: Option<engine_model::plan::ModelPlan>,
    bound: Option<forward::Bound<'a, 'b>>,
}
impl<'a, 'b> Model<'a, 'b> {
    pub fn load(b: &'b Backend<'a>, root: &Path, capacity: usize, world: usize) -> Result<Self> {
        Self::load_with_mtp(b, root, capacity, world, false)
    }
    pub fn load_with_mtp(b: &'b Backend<'a>, root: &Path, capacity: usize, world: usize, mtp: bool) -> Result<Self> {
        let draft_options = crate::hybrid_draft::Options::startup();
        let (runtime_plan, checkpoint) = plan::inspect(root, mtp)?;
        // Reuse the inspected headers when entering the same typed loader as load_generic.
        if plan::uses_typed_loader(&runtime_plan) {
            return Self::load_typed(b, root, capacity, world, mtp, runtime_plan, checkpoint.index, checkpoint.shards);
        }
        let arms = arms::Table::new(&runtime_plan,world,mtp);
        let profile_plan = Some(runtime_plan);
        let c = &checkpoint.config;
        Self::require_world(world)?;
        if capacity == 0 || capacity as u64 > c.max_positions {
            return Err(spark_gpu_probe::admission::token_refusal(capacity as u128, c.max_positions as usize, &format!("CONTEXT={capacity}; OUTPUT included in CONTEXT; context must be positive"), "the checkpoint position ceiling", spark_gpu_probe::admission::SERVE));
        }
        arms::Limits::shared(c)?;
        b.require_layer_kernels()?;
        b.require_batch_kernels()?;
        let mut requests = BTreeMap::<String, Vec<(String, WeightRows)>>::new();
        let mut parameters = Vec::new();
        let mut mtp_bytes = 0;
        use engine_model::shapes::{Place, Store};
        let resident = profile_plan.as_ref().unwrap().shape.tensors.iter().filter(|t|
            matches!(t.role.place, Place::Model | Place::Layer(_)) || mtp && matches!(t.role.place, Place::Draft { .. }));
        let shapes = resident.clone().map(|t| (t.name.clone(), (t.shape.clone(), t.dtype))).collect::<BTreeMap<_, _>>();
        let projections = resident.filter(|t| t.role.store == Store::Weight && t.dtype == engine_model::safetensors::Dtype::F8E4M3).map(|t| t.name.clone());
        for (name, (shape, dtype)) in shapes {
            let shard = checkpoint.index.weight_map.get(&name).ok_or_else(|| format!("missing hybrid weight {name}"))?.clone();
            let tensor = checkpoint.tensor(&name).map_err(|e| e.to_string())?;
            if name.starts_with("mtp.") {
                if tensor.shape != shape || tensor.dtype != dtype {
                    return Err(format!("{name}: MTP requires the declared {dtype:?} shape {shape:?}"));
                }
                mtp_bytes += tensor.byte_len() as usize;
            }
            if tensor.dtype == engine_model::safetensors::Dtype::F32 {
                parameters.push((name, shard.clone(), checkpoint.shards[&shard].data_start + tensor.start, tensor.byte_len() as usize));
                continue;
            }
            let fp8 = tensor.dtype == engine_model::safetensors::Dtype::F8E4M3;
            let expert = name.contains(".mlp.experts.");
            let packed = (expert || shape.len() == 2 && name.ends_with(".weight")
                && (c.tied_embeddings || !name.ends_with("embed_tokens.weight")))
                && !name.ends_with("shared_expert_gate.weight") && !fp8;
            let (rows, columns) = if expert { ((shape[0] * shape[1]) as usize, shape[2] as usize) }
                else if shape.len() == 2 { (shape[0] as usize, shape[1] as usize) }
                else { (1, shape.iter().product::<u64>() as usize) };
            if packed && (rows % 8 != 0 || columns % 16 != 0) {
                return Err(format!("{name}: unsupported packed BF16 projection dimensions"));
            }
            requests.entry(shard.clone()).or_default().push((name, WeightRows {
                file_offset: checkpoint.shards[&shard].data_start + tensor.start,
                rows: 0..rows, columns, packed, fp8,
            }));
        }
        // Mixed text/vision shards are opened, but only these text ranges are
        // uploaded. Keep immutable file descriptors through the entire load.
        let shard_names = requests.keys().cloned().collect::<Vec<_>>();
        let files = shard_names.iter().map(|name| TensorFile::open(&root.join(name), Limits::default())
            .map_err(|e| e.to_string())).collect::<Result<Vec<_>>>()?;
        for ((name, _), file) in requests.iter().zip(&files) {
            if file.header().file_bytes != checkpoint.shards[name].file_bytes
                || file.header().tensors != checkpoint.shards[name].tensors {
                return Err("checkpoint changed after text inspection".into());
            }
        }
        let mut weights = BTreeMap::new();
        // Hybrid projections use packed8. The shared MoE backend consumes
        // 64-row blocks for both the router and the selected expert matrices.
        // Partition tensor ranges before uploading, so each weight is read and
        // stored once, in the layout of the kernels that consume it.
        let blocked = |name: &str| name.contains(".mlp.experts.")
            || c.experts != 0 && name.ends_with(".mlp.gate.weight");
        for group in [false, true] {
            let rows = requests.values().map(|entries| entries.iter()
                .filter(|(name, _)| blocked(name) == group)
                .map(|(_, row)| WeightRows { file_offset: row.file_offset,
                    rows: row.rows.clone(), columns: row.columns, packed: row.packed, fp8: row.fp8 })
                .collect::<Vec<_>>()).collect::<Vec<_>>();
            let shards = files.iter().zip(&rows).filter(|(_, rows)| !rows.is_empty())
                .map(|(file, rows)| WeightShard { file: file.file(), matrices: rows }).collect::<Vec<_>>();
            if shards.is_empty() { continue; }
            let loaded = if group { b.load_bf16_shards_in_arena(&shards)? }
                else { b.load_text_shards_in_arena(&shards)? };
            for ((entries, _), matrices) in requests.values().zip(&rows)
                .filter(|(_, rows)| !rows.is_empty()).zip(loaded) {
                for ((name, _), matrix) in entries.iter()
                    .filter(|(name, _)| blocked(name) == group).zip(matrices) {
                    weights.insert(name.clone(), matrix);
                }
            }
        }
        for name in projections {
            let matrix = weights.remove(&name).unwrap();
            let scales = weights.remove(&format!("{name}_scale_inv")).unwrap();
            weights.insert(name, matrix.with_fp8_scales(scales)?);
        }
        for (name, shard, offset, bytes) in parameters {
            let mut data = vec![0; bytes];
            let file = &files[shard_names.iter().position(|s| s == &shard).ok_or("FP32 parameter has no text shard")?];
            file.file().read_exact_at(&mut data, offset).map_err(|e| e.to_string())?;
            let values = data.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect::<Vec<_>>();
            weights.insert(name, b.upload_f32_parameters(&values)?);
        }
        if c.tied_embeddings {
            let embedding = &weights[&format!("{PREFIX}.embed_tokens.weight")];
            weights.insert("lm_head.weight".into(), embedding.rows_view(0..embedding.rows())?);
        }
        let mut experts = Vec::new();
        for layer in 0..if c.experts == 0 { 0 } else { c.layers.len() } {
            let p = format!("{PREFIX}.layers.{layer}.mlp.experts");
            let pair = &weights[&format!("{p}.gate_up_proj")];
            let down = &weights[&format!("{p}.down_proj")];
            let (m, h) = (c.intermediate as usize, c.hidden as usize);
            let mut gates = Vec::new(); let mut ups = Vec::new(); let mut downs = Vec::new();
            for expert in 0..c.experts as usize {
                gates.push(pair.rows_view(expert * 2 * m..expert * 2 * m + m)?);
                ups.push(pair.rows_view(expert * 2 * m + m..(expert + 1) * 2 * m)?);
                downs.push(down.rows_view(expert * h..(expert + 1) * h)?);
            }
            experts.push(b.expert_table(gates, ups, downs)?);
        }
        let mut frequencies = b.vector((c.rotary_dim / 2) as usize)?;
        b.write_vector(&mut frequencies, &(0..c.rotary_dim / 2)
            .map(|i| 1.0 / (c.rope_theta as f32).powf(2.0 * i as f32 / c.rotary_dim as f32)).collect::<Vec<_>>())?;
        Ok(Self { backend: b, config: checkpoint.config, weights, experts, frequencies, capacity,
            weight_bytes: checkpoint.text_bytes as usize + mtp_bytes, mtp, draft_options, arms, profile_plan, bound: None })
    }
    pub fn config(&self) -> &Config { &self.config }
    /// Reconciled runtime authority; the optional field name is retained for existing profiling callers.
    pub fn model_plan(&self) -> &engine_model::plan::ModelPlan { self.profile_plan.as_ref().unwrap() }
    fn profile_reads(&self, layer: Option<usize>, positions: impl FnOnce() -> Vec<(usize, usize)>) {
        use engine_model::shapes::{Place, Mixer};
        self.backend.generic_plan(|| {
            let p = self.profile_plan.as_ref().unwrap();
            let place = layer.map_or(Place::Model, |i| Place::Layer(i as u32));
            self.backend.prefill_profile_layer(|| layer.map_or("output", |i| if matches!(p.shape.layers[i].mixer, Mixer::Linear(_)) { "gdn" } else { "full_attention" }));
            for class in 0..11 { self.backend.generic_read_bytes(|| (class, p.profile_bytes(place, class))); }
            if let Some(i) = layer.filter(|&i| matches!(p.shape.layers[i].mixer, Mixer::Linear(_))) { for (past, rows) in positions() { self.backend.generic_read_bytes(|| p.profile_state(i, past, rows)); } }
        });
    }
    pub fn max_tokens(&self) -> usize { self.capacity }
    pub fn weight_bytes(&self) -> usize { self.weight_bytes }
    pub fn mtp_enabled(&self) -> bool { self.mtp }
    pub fn backend(&self) -> &Backend<'a> { self.backend }
    /// The scheduler's cache is confined to this immutable loaded model.
    pub fn cache_identity(&self) -> Result<cache::Identity<'_>> {
        let p = self.model_plan();
        if let Some(bound)=&self.bound { let g=&bound.group; if !g.valid() { return Err("hybrid rank group invalid".into()); }
            return cache::Identity::new(self as *const _ as usize,p,linear::ARITHMETIC_ID,g.split(),g.rank() as u32)?.with_group(g.members(),g.incarnation());
        }
        let split = Self::reservation_split(p, self.mtp)?;
        cache::Identity::new(self as *const _ as usize, p, "hybrid-canonical-tile/v1", &split, 0)
    }
    pub fn rank_group(&self) -> Result<&tp::Group> { self.bound.as_ref().map(|b|&b.group).ok_or("typed rank group absent".into()) }
    pub fn fixed_rows(&self) -> bool { self.bound.is_some() }
    fn reservation_split(p: &engine_model::plan::ModelPlan, draft: bool) -> Result<engine_model::split::SplitPlan> {
        engine_model::split::SplitPlan::new(&p.shape, &engine_model::split::SplitRequest {
            ranks:1, memory:vec![u64::MAX], reserve:vec![0], context:0, sequences:0,
            cache_bytes:2, state_bytes:4, draft, expert_rows:false }).map_err(|e| e.to_string())
    }
    /// Header-only startup bound, including aligned arena, descriptor owners and packing peak.
    pub fn resident_reservation_bytes(p: &engine_model::plan::ModelPlan, index: &engine_model::safetensors::Index,
        headers: &BTreeMap<String, engine_model::safetensors::Header>, draft: bool) -> Result<usize> {
        let split = Self::reservation_split(p, draft)?;
        let load = weights::LoadPlan::new(p,index,headers,&split)?;
        let tables = p.shape.layers.iter().chain(p.shape.draft.iter().filter(|_| draft)).try_fold(0usize,|n,l| {
            let experts=if let engine_model::shapes::Ffn::Experts(e)=l.ffn { usize::try_from(e.count).ok()? } else { 0 };
            let records=experts.checked_mul(3)?;
            n.checked_add(records.checked_mul(40)?)?.checked_add(records.checked_next_power_of_two()?.checked_mul(40)?) });
        let packing = load.tensors().values().filter(|v| v.tensor.dtype == engine_model::safetensors::Dtype::Bf16
            && v.tensor.shape.len()==2).map(|v| v.cols.checked_mul(128)).collect::<Option<Vec<_>>>()
            .ok_or("canonical preparation overflow")?.into_iter().max().unwrap_or(0).max(64*1024*1024);
        load.bytes().checked_add(tables.ok_or("descriptor owner overflow")?).and_then(|n| n.checked_add(packing))
            .and_then(|n| n.checked_add(usize::try_from(p.rope.rotary_dim).ok()?.checked_mul(2)?)).ok_or("resident byte overflow".into())
    }
    /// Bound every unchecked legacy length helper before using it for fixed-row storage.
    fn fixed_geometry(c: &Config, capacity: usize) -> Option<()> {
        if capacity==0 || capacity as u64>c.max_positions || c.max_positions>u32::MAX as u64
            || arms::Limits::fixed(c).is_err() || c.kv_heads==0 || c.query_heads==0 || c.vocab==0
            || c.experts==0 || c.selected_experts==0 || c.selected_experts>c.experts || c.layers.is_empty() { return None; }
        let width = [c.hidden,c.intermediate,c.shared_intermediate,c.vocab,c.query_heads.checked_mul(c.head_dim)?,
            c.linear_value_heads.checked_mul(128)?,c.linear_key_heads.checked_mul(128)?].into_iter().max()?;
        let bound = (width as u128).checked_mul(c.experts as u128+1)?.checked_mul(c.selected_experts as u128+1)?
            .checked_mul(decode::MAX_VERIFY_ROWS as u128*32768)?;
        usize::try_from(bound).ok()?;
        usize::try_from((c.layers.len() as u128).checked_mul(c.linear_value_heads as u128+2*c.linear_key_heads as u128)?
            .checked_mul(128*128*16)?).ok()?;
        usize::try_from((c.max_positions as u128).checked_mul(c.query_heads as u128+c.kv_heads as u128)?
            .checked_mul(c.layers.len() as u128+1)?.checked_mul(4096)?).ok()?;
        Some(())
    }
    pub fn fixed_decode_bytes(c: &Config, rows: usize) -> Option<usize> {
        Self::fixed_geometry(c,1)?;
        if !matches!(rows, decode::MAX_ROWS | decode::MAX_CHAT_ROWS | decode::MAX_VERIFY_ROWS | schedule::PROMPT_ROWS | schedule::PROMPT_GROUP | schedule::ROUTED_GROUP) { return None; }
        // Typed rows never instantiate the legacy BF16 expert workspace.
        decode::Decode::reservation_bytes_for_rows(c,rows).checked_sub(moe::Workspace::reservation_bytes(c,rows))?
            .checked_add(rows.min(schedule::PROMPT_ROWS).checked_mul(c.query_heads as usize)?.checked_mul(c.max_positions as usize)?.checked_mul(4)?)
    }
    pub fn fixed_request_bytes(c: &Config, tokens: usize, ceiling: usize, draft: bool) -> Option<usize> {
        Self::fixed_request_rows(c,tokens,ceiling,draft,schedule::ROUTED_GROUP,true)
    }
    /// Scores can grow to the configured context, never the checkpoint ceiling.
    /// The second score span funds replacement while its predecessor is live.
    pub fn fixed_decode_bytes_at(c: &Config, rows: usize, context: usize) -> Option<usize> {
        Self::fixed_geometry(c,context)?;
        memory::workspace(c,rows,context,Self::fixed_decode_bytes(c,rows)?)
    }
    pub fn fixed_pooled_request_bytes(c: &Config, tokens: usize, ceiling: usize) -> Option<usize> {
        Self::fixed_request_rows(c,tokens,ceiling,false,memory::private_rows(true),true)
    }
    pub fn fixed_prompt_bytes(c: &Config, context: usize) -> Option<usize> {
        Self::fixed_decode_bytes_at(c,schedule::ROUTED_GROUP,context)?
            .checked_add(memory::host_logits(c,schedule::PROMPT_ROWS)?)
    }
    /// Live admission selects retention; startup wrappers conservatively fund snapshots.
    pub fn fixed_request_rows(c: &Config, tokens: usize, ceiling: usize, draft: bool, rows: usize, retain: bool) -> Option<usize> {
        Self::fixed_geometry(c,ceiling)?;
        if tokens==0 || tokens>ceiling { return None; }
        let capacity=tokens.checked_next_power_of_two()?.max(INITIAL_TILE).min(ceiling);
        let arms=arms::Table::for_config(c,arms::Limits::LOCAL_WORLD,draft);
        let scratch=arms.arm(arms::Kind::Gdn).reserve(|| Some(GdnScratch::reservation_bytes(c.hidden as usize,c.linear_value_heads as usize,c.linear_key_heads as usize,1)))?
            + arms.arm(arms::Kind::Attention).reserve(|| Some(DenseScratch::reservation_bytes(c,1)))? + session_lengths(c,1).iter().map(|n| (n*4).div_ceil(64)*64).sum::<usize>();
        // Private verifier is retained by Session; shared Decode is funded at startup.
        memory::request_with_retention(c,tokens,ceiling,rows,scratch,Self::fixed_decode_bytes_at(c,rows,ceiling)?,draft,
            arms.arm(arms::Kind::Mtp).reserve(|| mtp::fixed_reservation_bytes(c,capacity,crate::hybrid_draft::Options::startup()))?,retain)
    }
    pub fn prefill_chunk_size(&self) -> usize { if self.fixed_rows() { schedule::PROMPT_ROWS.min(self.capacity) } else { prefill_tile(&self.config).min(self.capacity) } }
    pub fn session(&self) -> Result<Session<'_, 'a, 'b>> { Session::new(self, self.mtp) }
    pub fn default_draft_depth(&self) -> usize {
        crate::hybrid_draft::default_depth(self.bound.as_ref().is_some_and(|bound| bound.group.distributed()))
    }
    pub fn plain_session(&self) -> Result<Session<'_, 'a, 'b>> { Session::new(self, false) }
    /// Full-capacity sessions include BF16 KV, declared FP32 recurrent state,
    /// one shared layer scratch, vocabulary logits and bounded host bookkeeping.
    pub fn session_reservation_bytes(&self) -> usize {
        self.request_reservation_bytes(self.capacity)
    }
    pub fn request_reservation_bytes(&self, tokens: usize) -> usize {
        if self.fixed_rows() { return Self::fixed_request_bytes(&self.config,tokens,self.capacity,self.mtp).unwrap_or(usize::MAX); }
        let Some(capacity) = tokens.max(1).checked_next_power_of_two() else { return usize::MAX; };
        let capacity = capacity.max(INITIAL_TILE).min(self.capacity);
        Self::session_bytes(&self.config, capacity) + self.mtp_reservation(capacity)
    }
    fn mtp_reservation(&self, capacity: usize) -> usize {
        self.arms.arm(arms::Kind::Mtp).reserve(|| Some(mtp::reservation_bytes(&self.config, capacity, self.draft_options))).unwrap_or(usize::MAX)
    }
    pub fn session_bytes(c: &Config, capacity: usize) -> usize {
        let tile = prefill_tile(c).min(capacity);
        c.state_bytes() as usize + capacity * c.kv_bytes_per_token() as usize
            + GdnScratch::reservation_bytes(c.hidden as usize, c.linear_value_heads as usize, c.linear_key_heads as usize, tile)
            + DenseScratch::reservation_bytes(c, tile)
            + capacity * c.query_heads as usize * 4
            + session_lengths(c, tile).iter().map(|n| (n * 4).div_ceil(64) * 64).sum::<usize>()
            + 1024 * 1024
            + c.state_bytes() as usize + c.hidden as usize * 4
            + moe::Workspace::reservation_bytes(c, tile)
            // Typed prefill/step retains its canonical executor even without MTP.
            + if c.fp8 && c.experts > 0 { decode::Decode::reservation_bytes_for_rows(c,schedule::ROUTED_GROUP) } else { 0 }
            // Private partial pages and compact-to-paged migration coexist
            // briefly with the source. Whole immutable pages are funded once.
            + Self::kv_page_slack(c)
    }
    fn kv_page_slack(c: &Config) -> usize {
        c.layers.iter().filter(|&&kind| kind == LayerKind::SparseAttention).count() * 4 * 2 * 1024 * 1024
    }
    fn matrix(&self, name: &str) -> Result<&Matrix<'a>> {
        if name == "lm_head.weight" && self.config.tied_embeddings {
            return self.weights.get(&format!("{PREFIX}.embed_tokens.weight")).ok_or("missing tied hybrid embedding".into());
        }
        self.weights.get(name).ok_or_else(|| format!("missing hybrid weight {name}"))
    }
    fn address(&self, name: &str) -> Result<u64> {
        if let Some(v)=self.bound.as_ref().and_then(|b|b.norms.get(name)) { return Ok(v.device_address()); }
        self.matrix(name).map(Matrix::device_address)
    }
    fn scalar_precision(&self, prefix: &str) -> u32 {
        1 | (u32::from(self.weights[&format!("{prefix}.A_log")].scalar_f32()) << 1)
            | (u32::from(self.weights[&format!("{prefix}.dt_bias")].scalar_f32()) << 2)
            | (u32::from(self.weights[&format!("{prefix}.norm.weight")].scalar_f32()) << 3)
    }
}

struct PendingPrefill {
    input: Vec<u32>,
    appended: usize,
    offset: usize,
    layer: usize,
    logits: bool,
}

pub struct Session<'m, 'a, 'b> {
    graphs: [Option<DecodeGraph<'a>>; 2],
    graph_position: Option<DeviceVector<'a>>,
    decode_lifetime: std::rc::Rc<()>,
    model: &'m Model<'a, 'b>,
    gdn_cache: Vec<Option<GdnState<'a>>>,
    full_cache: Vec<Option<DenseState<'a>>>,
    gdn: GdnScratch<'a>,
    attention: DenseScratch<'a>,
    moe: Option<moe::Workspace<'a>>,
    hidden: DeviceVector<'a>,
    normalized: DeviceVector<'a>,
    block: DeviceVector<'a>,
    gate: DeviceVector<'a>,
    up: DeviceVector<'a>,
    weight_pack: DeviceVector<'a>,
    input_pack: DeviceVector<'a>,
    ids: DeviceVector<'a>,
    final_row: DeviceVector<'a>,
    logits: DeviceVector<'a>,
    tile: usize,
    prompt_tile: usize,
    position: usize,
    valid: bool,
    cache_bytes: usize,
    mtp: Option<mtp::Draft<'a>>,
    external: Option<Box<mtp::External<'m, 'a, 'b>>>,
    draft_depth: usize,
    draft_policy: mtp::Policy,
    batch_drafts: bool,
    pending_draft: Option<usize>,
    verifier: Option<Box<decode::Decode<'m, 'a, 'b>>>,
    prompt_workspace: Option<decode::PromptWorkspace<'m, 'a, 'b>>,
    greedy_logits: Option<usize>,
    verify_hidden: Option<DeviceVector<'a>>,
    verify_entry: usize,
    verify_tokens: Vec<u32>,
    rollback_state: Vec<Vec<Option<GdnState<'a>>>>,
    prefill_tail: Vec<u32>,
    cached_prompt_tokens: Option<usize>,
    prefill_saved: Vec<Option<GdnState<'a>>>,
    // Immutable canonical tile boundaries survive forks and history rewrites.
    prefix_saved: Vec<(usize, std::rc::Rc<Vec<Option<GdnState<'a>>>>)>,
    retain_prefix: bool,
    prefill_mtp: Option<DeviceVector<'a>>,
    pending_prefill: Option<PendingPrefill>,
}
impl<'m, 'a, 'b> Session<'m, 'a, 'b> {
    fn new(model: &'m Model<'a, 'b>, mtp: bool) -> Result<Self> {
        let c = &model.config;
        let b = model.backend;
        let tile = 1;
        let h = c.hidden as usize;
        let gdn = GdnScratch::with_dimensions(b, h, c.linear_value_heads as usize, c.linear_key_heads as usize, tile, &[0.0; 128])?;
        let attention = DenseScratch::new(b, c, tile)?;
        let lengths = session_lengths(c, tile);
        let mut v = b.vector_arena(&lengths)?.into_iter();
        Ok(Self { graphs: [None, None], graph_position: Some(b.vector(4)?), decode_lifetime: std::rc::Rc::new(()), greedy_logits: None, model, gdn_cache: (0..c.layers.len()).map(|_| None).collect(), full_cache: (0..c.layers.len()).map(|_| None).collect(),
            gdn, attention, moe: moe::Workspace::new(model, tile)?, hidden: v.next().unwrap(), normalized: v.next().unwrap(), block: v.next().unwrap(),
            gate: v.next().unwrap(), up: v.next().unwrap(), weight_pack: v.next().unwrap(), input_pack: v.next().unwrap(),
            ids: v.next().unwrap(), final_row: v.next().unwrap(), logits: v.next().unwrap(), tile, prompt_tile: tile, position: 0, valid: true, cache_bytes: 0,
            mtp: if mtp { Some(mtp::Draft::new(model)?) } else { None }, external: None, draft_depth: model.default_draft_depth(), draft_policy: mtp::Policy::configured(model.draft_options), batch_drafts: false, pending_draft: None, verifier: None, verify_hidden: None,
            verify_entry: 0, verify_tokens: Vec::new(), rollback_state: Vec::new(),
            prefill_tail: Vec::new(), cached_prompt_tokens: None, prefill_saved: Vec::new(), prefill_mtp: None,
            prefix_saved: Vec::new(), retain_prefix: false, pending_prefill: None, prompt_workspace: None })
    }
    /// Typed sessions borrow one serialized prompt executor, including MTP.
    pub fn share_prompt_workspace(&mut self, workspace: decode::PromptWorkspace<'m, 'a, 'b>) -> Result<()> {
        if !self.model.fixed_rows() { return Err("shared prompt requires typed session".into()); }
        self.prompt_workspace=Some(workspace); Ok(())
    }
    fn decode_rows(&self) -> usize { memory::private_rows(self.prompt_workspace.is_some()) }
    pub fn len(&self) -> usize { self.position }
    /// Rows inherited from a prior request, excluding canonical tail replay.
    pub fn cached_prompt_tokens(&self) -> Option<usize> { self.cached_prompt_tokens }
    pub fn set_prefix_retention(&mut self, retain: bool) {
        self.retain_prefix = retain;
        if !retain { self.prefix_saved.clear(); }
    }
    fn record_prompt_replay(&mut self) {
        if let Some(cached) = &mut self.cached_prompt_tokens { *cached = (*cached).min(self.position); }
    }
    pub fn is_empty(&self) -> bool { self.position == 0 }
    /// Qualification receipt: exact visible target state, excluding draft and scratch.
    /// Keep these potentially large bytes on the native test machine.
    pub fn target_state_bytes(&mut self) -> Result<Vec<u8>> {
        self.finish_mtp_verification()?;
        if !self.valid || self.pending_prefill.is_some() || self.pending_draft.is_some() { return Err("target state is not committed".into()); }
        let (b,c)=(self.model.backend,&self.model.config);
        if self.position>0 && c.layers.iter().enumerate().any(|(i,k)| match k { LayerKind::DeltaNet=>self.gdn_cache[i].is_none(), LayerKind::SparseAttention=>self.full_cache[i].is_none() }) { return Err("target state layer absent".into()); }
        let mut spans=Vec::new();
        let address=|d:&[f32],i:usize| u64::from(d[i].to_bits()) | (u64::from(d[i+1].to_bits())<<32);
        for state in self.gdn_cache.iter().flatten() { let d=state.descriptor();
            spans.push((address(&d,0),(c.linear_value_heads*c.linear_key_dim*c.linear_value_dim) as usize));
            spans.push((address(&d,2),((2*c.linear_key_heads*c.linear_key_dim+c.linear_value_heads*c.linear_value_dim)*c.linear_conv/2) as usize));
        }
        for state in self.full_cache.iter().flatten() { let d=state.descriptor(self.position);
            for i in [0,2] { spans.push((address(&d,i),self.position*(c.kv_heads*c.head_dim) as usize/2)); }
        }
        let mut bytes=(self.position as u64).to_le_bytes().to_vec();
        for (source,words) in spans { if words==0 { continue; } let count=u32::try_from(words).map_err(|_| "target receipt span exceeds copy ABI")?; let copy=b.vector(words)?;
            let mut p=spark_gpu_probe::kernel_params::HybridCopy { source:source as _,destination:copy.device_address() as _,words:count,padding:0,rows:0 };
            unsafe { b.launch_flash_rows("flash_hybrid_copy",words.div_ceil(128) as u32,1,128,0,&mut [(&mut p as *mut spark_gpu_probe::kernel_params::HybridCopy).cast()])?; }
            bytes.extend(b.read_vector_values(&copy)?.into_iter().flat_map(|v|v.to_bits().to_le_bytes()));
        }
        Ok(bytes)
    }
    pub fn can_fork(&self) -> bool { self.valid && self.verify_tokens.is_empty() && self.pending_prefill.is_none()
        && self.pending_draft.is_none() && self.replay_entry().is_ok() && self.model.bound.as_ref().is_none_or(|b|b.group.valid()) }
    pub fn cache_identity(&self) -> Result<cache::Identity<'_>> { self.model.cache_identity() }
    pub fn require_cache_identity(&self, model: &Model<'a, 'b>) -> Result<()> {
        // Same immutable instance fixes the checkpoint, plan, arithmetic and owners.
        if std::ptr::eq(self.model, model) { return self.cache_identity().map(|_|()); }
        if self.cache_identity()? != model.cache_identity()? { return Err("hybrid cache identity mismatch".into()); }
        Ok(())
    }
    fn replay_entry(&self) -> Result<usize> {
        let snapshots = self.model.config.layers.iter().enumerate().all(|(i, kind)|
            *kind != LayerKind::DeltaNet || self.prefill_saved.get(i).is_some_and(Option::is_some));
        cache::replay_entry(self.position, self.prefill_tail.len(), snapshots)
    }
    fn restore_replay_entry(&mut self) -> Result<()> {
        self.valid = false;
        let entry = self.replay_entry()?;
        let b = self.model.backend;
        if entry == 0 {
            for live in self.gdn_cache.iter_mut().flatten() { live.reset(b)?; }
        } else {
            for (saved, live) in self.prefill_saved.iter().zip(&mut self.gdn_cache) {
                if let (Some(saved), Some(live)) = (saved, live) { saved.copy_into(b, live)?; }
            }
        }
        self.position = entry;
        self.record_prompt_replay();
        Ok(())
    }
    /// A fork owns its mutable recurrence and aliases immutable KV pages.
    /// Keep the canonical tile entry as well: appending to its partial tile
    /// must replay the same arithmetic as a cold prompt. An in-step MTP
    /// history (and its tile entry row) comes along, so cached turns draft.
    pub fn fork(&self) -> Result<Self> {
        if !self.can_fork() { return Err("hybrid fork requires a committed session".into()); }
        let (b, c) = (self.model.backend, &self.model.config);
        let mut next = self.model.plain_session()?;
        next.prompt_workspace = self.prompt_workspace.clone();
        next.draft_depth = self.draft_depth;
        next.draft_policy.options = self.draft_policy.options;
        if let Some(mtp) = self.mtp.as_ref() {
            next.mtp = mtp.fork(self.model, self.position)?;
            next.batch_drafts = self.batch_drafts && next.mtp.is_some();
        }
        if next.mtp.is_some()
            && let Some(last) = &self.prefill_mtp {
                let mut copy = b.vector(last.len())?;
                b.copy_vector_range(last, &mut copy, 0, 0, last.len())?;
                next.prefill_mtp = Some(copy);
            }
        if self.position == 0 { return Ok(next); }
        next.reserve_state()?;
        let capacity = self.position.next_power_of_two().max(INITIAL_TILE).min(self.model.capacity);
        next.full_cache = DenseState::fork_layers(b, c, &self.full_cache, self.position, capacity, self.model.capacity)?;
        next.cache_bytes += next.full_cache.iter().flatten().map(DenseState::device_bytes).sum::<usize>();
        for (live, copy) in self.gdn_cache.iter().zip(&mut next.gdn_cache) {
            if let (Some(live), Some(copy)) = (live, copy) { live.copy_into(b, copy)?; }
        }
        if !self.prefill_saved.is_empty() {
            next.prefill_saved = GdnState::dense_layers(b, c)?;
            for (live, copy) in self.prefill_saved.iter().zip(&mut next.prefill_saved) {
                if let (Some(live), Some(copy)) = (live, copy) { live.copy_into(b, copy)?; }
            }
        }
        next.position = self.position;
        next.cached_prompt_tokens = Some(self.position);
        next.prefill_tail = self.prefill_tail.clone();
        next.prefix_saved = self.prefix_saved.clone();
        next.retain_prefix = self.retain_prefix;
        b.synchronize()?;
        Ok(next)
    }
    /// Conservative bound for the compact fork, including its private scratch.
    pub fn fork_bytes(&self) -> usize {
        self.fork_private_bytes() + self.kv_allocations().iter().map(|(_, bytes)| bytes).sum::<usize>()
    }
    /// Snapshot recurrence/scratch remains private; page identities exclude it.
    pub fn fork_private_bytes(&self) -> usize {
        Model::session_bytes(&self.model.config, 1)
            + self.prefix_snapshot_bytes()
            - if self.model.fixed_rows() { moe::Workspace::reservation_bytes(&self.model.config,1) } else { 0 }
            // A compact typed fork has no canonical prompt executor; it is
            // created lazily (or borrowed from the shared prompt workspace).
            - if self.model.fixed_rows() { decode::Decode::reservation_bytes_for_rows(&self.model.config,schedule::PROMPT_GROUP) } else { 0 }
            - Model::kv_page_slack(&self.model.config) - self.model.config.kv_bytes_per_token() as usize
            + self.prefill_tail.capacity() * 4
            + self.verifier.as_ref().filter(|_| self.model.bound.is_none()).map_or(0, |_| decode::Decode::reservation_bytes(&self.model.config))
            + self.mtp.as_ref().map_or(0, |d| d.fork_bytes(self.model))
    }
    pub fn kv_allocations(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for state in self.full_cache.iter().flatten() { state.allocations(&mut out); }
        out.sort_unstable_by_key(|&(id, _)| id);
        out.dedup_by_key(|entry| entry.0);
        out
    }
    pub fn prefix_allocations(&self, tokens: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for state in self.full_cache.iter().flatten() {
            state.prefix_allocations(&self.model.config, tokens.min(self.position), &mut out);
        }
        out
    }
    pub fn checkpoint_reusable_tokens(&self) -> usize { self.position - self.prefill_tail.len() }
    /// A recurrent prefix is reusable only where an immutable snapshot exists.
    pub fn prefix_reusable_tokens(&self, position: usize) -> Option<usize> {
        if position > self.position { return None; }
        let entry = self.checkpoint_reusable_tokens();
        if position >= entry { return Some(entry); }
        self.prefix_saved.iter().rev().find(|(at, _)| *at <= position).map(|(at, _)| *at)
    }
    pub fn truncate_prefix(&mut self, position: usize) -> Result<()> {
        if !self.can_fork() || position > self.position {
            return Err("hybrid prefix requires a committed position".into());
        }
        if position == self.position { return Ok(()); }
        self.draft_policy.copy.truncate(position);
        let entry = self.position - self.prefill_tail.len();
        if position < entry {
            let at = self.prefix_reusable_tokens(position).ok_or("hybrid prefix precedes retained recurrent checkpoints")?;
            if position != at { return Err("hybrid prefix must use a retained recurrent boundary".into()); }
            self.valid = false;
            let saved = &self.prefix_saved.iter().find(|(p, _)| *p == at).unwrap().1;
            for (saved, live) in saved.iter().zip(&mut self.gdn_cache) {
                if let (Some(saved), Some(live)) = (saved, live) { saved.copy_into(self.model.backend, live)?; }
            }
            self.position = at;
            self.record_prompt_replay();
            self.graphs = [None, None];
            self.prefill_tail.clear();
            self.prefill_saved.clear();
            self.prefill_mtp = None;
            self.prefix_saved.retain(|(p, _)| *p <= at);
            // Older snapshots contain target state only, never a draft history.
            self.disable_model_draft();
            self.valid = true;
            // The caller replays the verified token suffix from this boundary.
            return Ok(());
        }
        // A prefix cache miss before this entry falls back to a cold session.
        // Never claim that stale recurrent state represents an earlier prefix.
        let tail = self.prefill_tail[..position - entry].to_vec();
        self.graphs = [None, None];
        self.restore_replay_entry()?;
        self.prefill_tail.clear();
        // The MTP history rewinds with the target to the same tile entry.
        if entry == 0 || self.prefill_mtp.is_some() { self.restore_prefill_mtp()?; } else { self.disable_model_draft(); }
        self.valid = true;
        if !tail.is_empty() { self.prefill_advance(&tail)?; }
        Ok(())
    }
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes + self.mtp_cache_bytes() + self.external.as_ref().map_or(0, |d| d.bytes())
            + self.prefix_snapshot_bytes()
            + self.rollback_state.iter().flatten().flatten().map(GdnState::device_bytes).sum::<usize>()
            + self.prefill_saved.iter().flatten().map(GdnState::device_bytes).sum::<usize>()
            + self.prefill_mtp.as_ref().map_or(0, |v| v.len() * 4)
    }
    fn prefix_snapshot_bytes(&self) -> usize {
        self.prefix_saved.iter().flat_map(|(_, states)| states.iter().flatten()).map(GdnState::device_bytes).sum()
    }
    fn save_prefix_boundary(&mut self) -> Result<()> {
        // Observe completed canonical tiles; never split or reshape arithmetic.
        if !self.retain_prefix || !self.model.fixed_rows() || !self.position.is_multiple_of(PREFIX_INTERVAL)
            || self.prefix_saved.last().is_some_and(|(at, _)| *at == self.position) { return Ok(()); }
        if self.prefix_saved.len() == PREFIX_SNAPSHOTS { self.prefix_saved.remove(0); }
        let b = self.model.backend;
        let mut saved = GdnState::dense_layers(b, &self.model.config)?;
        for (live, saved) in self.gdn_cache.iter().zip(&mut saved) {
            if let (Some(live), Some(saved)) = (live, saved) { live.copy_into(b, saved)?; }
        }
        self.prefix_saved.push((self.position, std::rc::Rc::new(saved)));
        Ok(())
    }
    fn reserve_state(&mut self) -> Result<()> {
        if self.gdn_cache.iter().all(Option::is_none) {
            self.gdn_cache = GdnState::dense_layers(self.model.backend, &self.model.config)?;
            self.cache_bytes += self.gdn_cache.iter().flatten().map(GdnState::device_bytes).sum::<usize>();
        }
        Ok(())
    }
    fn reserve_workspace(&mut self, count: usize) -> Result<()> {
        self.reserve_workspace_kind(count, false)
    }
    fn reserve_workspace_kind(&mut self, count: usize, shared: bool) -> Result<()> {
        if count <= self.tile { return Ok(()); }
        let tile = count.next_power_of_two().min(self.model.prefill_chunk_size());
        self.replace_workspace(tile, shared)
    }
    fn replace_workspace(&mut self, tile: usize, shared: bool) -> Result<()> {
        self.graphs = [None, None];
        self.decode_lifetime = std::rc::Rc::new(());
        let (b, c) = (self.model.backend, &self.model.config);
        let gdn = if shared {
            GdnScratch::shared_projections(b, c.hidden as usize, c.linear_value_heads as usize, c.linear_key_heads as usize, tile)?
        } else { GdnScratch::with_dimensions(b, c.hidden as usize, c.linear_value_heads as usize,
            c.linear_key_heads as usize, tile, &[0.0; 128])? };
        let attention = DenseScratch::new(b, c, tile)?;
        let mut v = b.vector_arena(&session_lengths(c, tile))?.into_iter();
        self.gdn = gdn;
        self.attention = attention;
        self.moe = moe::Workspace::new(self.model, tile)?;
        self.hidden = v.next().unwrap(); self.normalized = v.next().unwrap();
        self.block = v.next().unwrap(); self.gate = v.next().unwrap();
        self.up = v.next().unwrap(); self.weight_pack = v.next().unwrap();
        self.input_pack = v.next().unwrap(); self.ids = v.next().unwrap();
        self.final_row = v.next().unwrap(); self.logits = v.next().unwrap();
        self.tile = tile;
        self.prompt_tile = tile;
        Ok(())
    }
    fn reserve_prompt_workspace(&mut self, count: usize) -> Result<()> {
        if count <= self.prompt_tile { return Ok(()); }
        self.graphs = [None, None];
        let (b, c) = (self.model.backend, &self.model.config);
        let tile = count.next_power_of_two().min(self.model.prefill_chunk_size());
        self.gdn = GdnScratch::prompt_workspace(b, c.hidden as usize, c.linear_value_heads as usize,
            c.linear_key_heads as usize, tile, c.fp8)?;
        self.attention = DenseScratch::new(b, c, tile)?;
        self.prompt_tile = tile;
        Ok(())
    }
    fn reserve_kv(&mut self, end: usize) -> Result<()> {
        let (b, c) = (self.model.backend, &self.model.config);
        if !c.layers.iter().enumerate().any(|(i, kind)| *kind == LayerKind::SparseAttention
            && self.full_cache[i].as_ref().is_none_or(|s| s.capacity(c) < end || !s.writable_from(c, self.position))) { return Ok(()); }
        // Both solo and shared graphs must be rebuilt when KV addresses move.
        self.graphs = [None, None];
        self.decode_lifetime = std::rc::Rc::new(());
        let capacity = end.next_power_of_two().max(INITIAL_TILE).min(self.model.capacity);
        let before: usize = self.full_cache.iter().flatten().map(DenseState::device_bytes).sum();
        if before != 0 && self.full_cache.iter().flatten().all(|s| s.paged() && s.writable_from(c, self.position)) {
            for state in self.full_cache.iter_mut().flatten() { state.grow(b, c, capacity)?; }
        } else {
            let next = DenseState::fork_layers(b, c, &self.full_cache, self.position, capacity, self.model.capacity)?;
            if before != 0 { b.synchronize()?; }
            self.full_cache = next;
        }
        self.cache_bytes = self.cache_bytes - before + self.full_cache.iter().flatten().map(DenseState::device_bytes).sum::<usize>();
        Ok(())
    }
    pub fn device_peak_bytes(&self) -> usize { self.model.backend.peak_allocated_bytes() }
    pub fn prefill_chunk_size(&self) -> usize { self.model.prefill_chunk_size() }
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> { self.forward(&[token], false, true) }
    /// Typed plain singleton selection keeps the canonical row on-device for token filtering.
    pub fn step_greedy(&mut self, token: u32) -> Result<u32> {
        if self.model.bound.is_none() || self.mtp.is_some() || self.external.is_some() { return Err("greedy singleton requires typed hybrid without drafting".into()); }
        self.greedy_logits = None;
        let mut executor = self.verifier.take().map(Ok).unwrap_or_else(|| decode::Decode::with_rows(self.model, self.decode_rows()).map(Box::new))?;
        let result = executor.step_greedy(&mut [decode::Row { session: self, token, logits: true }])
            .and_then(|ids| if executor.rejected().is_empty() { Ok(ids[0]) } else { Err(MOE_ROUTING_ERROR.into()) });
        self.verifier = Some(executor);
        if result.is_ok() { self.greedy_logits = Some(self.position); }
        result
    }
    /// Download only when the selected singleton token needs the host filter.
    pub fn decode_logits(&mut self, rows: usize) -> Result<Option<Vec<Vec<f32>>>> {
        if rows != 1 || !self.valid || self.greedy_logits.take() != Some(self.position) { return Ok(None); }
        self.verifier.as_ref().ok_or("missing singleton logits")?.read_logits(rows).map(Some)
    }
    pub fn advance(&mut self, token: u32) -> Result<()> { self.forward(&[token], false, false).map(|_| ()) }
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> { self.prefill_inner(tokens, true) }
    pub fn prefill_advance(&mut self, tokens: &[u32]) -> Result<()> { self.prefill_inner(tokens, false).map(|_| ()) }
    fn prefill_inner(&mut self, tokens: &[u32], logits: bool) -> Result<Vec<f32>> {
        self.prefill_slice(tokens, logits, None)?.ok_or("unbudgeted hybrid prefill did not complete".into())
    }
    /// Resume a canonical tile at its next layer, keeping its shape and scratch.
    /// A pending tile cannot be forked, decoded or exposed as a committed prefix.
    pub fn prefill_slice(&mut self, tokens: &[u32], logits: bool, budget: Option<Duration>) -> Result<Option<Vec<f32>>> {
        let started = Instant::now();
        let mut pending = if let Some(pending) = self.pending_prefill.take() {
            if pending.logits != logits || pending.appended != tokens.len()
                || pending.input[pending.input.len() - pending.appended..] != *tokens {
                self.valid = false;
                return Err("hybrid prefill must resume the same chunk".into());
            }
            pending
        } else {
            if !self.valid || tokens.is_empty() || self.position.checked_add(tokens.len()).is_none_or(|n| n > self.model.capacity)
                || tokens.iter().any(|&t| t as u64 >= self.model.config.vocab) {
                return Err("invalid hybrid prompt or context capacity".into());
            }
            self.finish_mtp_verification()?;
            // Rebuild an inherited partial tile once, with the same row count
            // and recurrence order as cold prefill. Scheduling never shortens it.
            if !self.prefill_tail.is_empty() {
                self.restore_replay_entry()?;
                self.restore_prefill_mtp()?;
                self.valid = true;
            }
            let mut input = std::mem::take(&mut self.prefill_tail);
            self.prefill_saved.clear();
            self.prefill_mtp = None;
            input.extend_from_slice(tokens);
            PendingPrefill { input, appended: tokens.len(), offset: 0, layer: 0, logits }
        };
        let tile = self.model.prefill_chunk_size();
        // Distributed passes stay inside the sealed PROMPT_GROUP envelope.
        // Tile boundaries, tails and logits rows stay those of 256-row chunks.
        let group = if let Some(bound) = &self.model.bound {
            if bound.group.distributed() { schedule::PROMPT_GROUP / tile } else { schedule::ROUTED_GROUP / tile }
        } else { 1 };
        loop {
            let tiles = ((pending.input.len() - pending.offset - 1) / tile).clamp(1, group);
            // Grouped passes still end at every retained canonical snapshot boundary.
            let tiles = if self.retain_prefix { tiles.min((PREFIX_INTERVAL - self.position % PREFIX_INTERVAL).div_ceil(tile)) } else { tiles };
            let end = (pending.offset + tile * tiles).min(pending.input.len());
            let chunk = &pending.input[pending.offset..end];
            if pending.layer == 0 && chunk.len() < tile { self.save_prefill_entry()?; }
            let remaining = budget.map(|b| b.saturating_sub(started.elapsed()));
            let output = self.forward_slice(chunk, true, logits && end == pending.input.len(), remaining, &mut pending.layer)?;
            let Some(output) = output else { self.pending_prefill = Some(pending); return Ok(None); };
            if chunk.len() % tile == 0 { self.save_prefix_boundary()?; }
            if chunk.len() < tile { self.prefill_tail.extend_from_slice(chunk); }
            if end == pending.input.len() { return Ok(Some(output)); }
            pending.offset = end;
            pending.layer = 0;
            if budget.is_some_and(|b| started.elapsed() >= b) {
                self.pending_prefill = Some(pending);
                return Ok(None);
            }
        }
    }
    fn save_prefill_entry(&mut self) -> Result<()> {
        if self.position == 0 { return Ok(()); }
        self.valid = false;
        self.reserve_state()?;
        let b = self.model.backend;
        self.prefill_saved = GdnState::dense_layers(b, &self.model.config)?;
        for (live, saved) in self.gdn_cache.iter().zip(&mut self.prefill_saved) {
            if let (Some(live), Some(saved)) = (live, saved) { live.copy_into(b, saved)?; }
        }
        self.save_prefill_mtp()?;
        self.valid = true;
        Ok(())
    }
    fn begin_decode(&mut self) -> Result<()> {
        self.finish_mtp_verification()?;
        // Prefill's expert and GDN workspaces are large and contain no state.
        // Each active chat retains only one-row scratch while decoding.
        if self.tile >= 256 || self.prompt_tile >= 256 { self.replace_workspace(1, false)?; }
        // Decode arithmetic is independent of chunk prefill. Keep its token
        // history and the last canonical entry until an append can rebuild it.
        if self.prefill_tail.is_empty() && self.prefill_saved.is_empty() { self.save_prefill_entry()?; }
        Ok(())
    }
    fn project(b: &Backend<'_>, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize) -> Result<()> {
        Self::project_reusing(b, matrix, input, output, weight_pack, input_pack, tokens, false)
    }
    fn project_reusing(b: &Backend<'_>, matrix: &Matrix<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, tokens: usize, reuse: bool) -> Result<()> {
        if tokens == 1 && !matrix.is_fp8() { return b.matvec_bf16_device(matrix, input, output); }
        b.matmul_prefill_bf16_device(matrix, input, output, weight_pack, input_pack, tokens, reuse)
    }
    /// Output projection added to the residual stream; FP8 fragments skip promotion.
    fn residual(b: &Backend<'_>, matrix: &Matrix<'_>, weight_pack: &DeviceVector<'_>, input: &DeviceVector<'_>,
        hidden: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, count: usize) -> Result<()> {
        if b.prefill_fp8_direct(&[matrix]) { return b.matmul_packed_residual_fp8_device(matrix, input, hidden, partials, count); }
        let matrix = b.prefill_weight_bf16(matrix, weight_pack, 0)?;
        b.matmul_packed_residual_bf16_device(&matrix, input, hidden, partials, count)
    }
    fn normalize(b: &Backend<'_>, model: &Model<'_, '_>, x: &DeviceVector<'_>, w: u64, y: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        Self::normalize_order(b, model, x, w, y, tokens, Some(2), false)
    }
    fn normalize_order(b: &Backend<'_>, model: &Model<'_, '_>, x: &DeviceVector<'_>, w: u64, y: &DeviceVector<'_>, tokens: usize, order: Option<u32>, four: bool) -> Result<()> {
        let c = &model.config;
        b.profile_phase(1, || if arms::Limits::norm_supported(c.hidden) {
            launch_norm(b, x, None, w, y, model, tokens, order, four)
        } else {
            launch(b, "flash_norm", tokens * 32, &mut [Ptr(x.device_address()), Ptr(w), Ptr(y.device_address()),
                U32(c.hidden as u32), U32(tokens as u32), U32(1), F32(c.epsilon as f32), U32(1), U32(1)])
        })
    }
    fn residual_norm(&self, w: u64, count: usize, fused: bool, four: bool) -> Result<()> {
        let (b, c) = (self.model.backend, &self.model.config);
        if fused {
            return b.profile_phase(1, || launch(b, "flash_norm_partials_bf16", count * 128,
                &mut [Ptr(self.hidden.device_address()), Ptr(self.block.device_address()), Ptr(w),
                    Ptr(self.input_pack.device_address()), U32(c.hidden as u32), F32(c.epsilon as f32)]));
        }
        if arms::Limits::norm_supported(c.hidden) {
            b.profile_phase(1, || launch_norm(b, &self.hidden, Some(&self.block), w, &self.normalized, self.model, count, None, four))
        } else {
            b.profile_phase(7, || launch(b, "flash_residual_bf16", self.hidden.len(),
                &mut [Ptr(self.hidden.device_address()), Ptr(self.block.device_address()), U32(self.hidden.len() as u32)]))?;
            Self::normalize(b, self.model, &self.hidden, w, &self.normalized, count)
        }
    }
    fn forward(&mut self, tokens: &[u32], prefill: bool, save_logits: bool) -> Result<Vec<f32>> {
        if self.pending_prefill.is_some() { return Err("hybrid prefill is still in flight".into()); }
        self.forward_slice(tokens, prefill, save_logits, None, &mut 0)?
            .ok_or("unbudgeted hybrid forward did not complete".into())
    }
    fn forward_slice(&mut self, tokens: &[u32], prefill: bool, save_logits: bool,
        budget: Option<Duration>, next_layer: &mut usize) -> Result<Option<Vec<f32>>> {
        self.greedy_logits = None;
        let (m, at) = (self.model, self.position);
        m.backend.generic_profile(prefill, || (at, tokens.len(), m.profile_plan.as_ref().unwrap().profile_classes(), 0),
            || self.forward_slice_profiled(tokens, prefill, save_logits, budget, next_layer))
    }
    fn forward_slice_profiled(&mut self, tokens: &[u32], prefill: bool, save_logits: bool,
        budget: Option<Duration>, next_layer: &mut usize) -> Result<Option<Vec<f32>>> {
        if self.model.bound.is_some() {
            if !self.valid || tokens.is_empty() || self.position.checked_add(tokens.len()).is_none_or(|n| n > self.model.capacity)
                || tokens.iter().any(|&t| t as u64 >= self.model.config.vocab) { return Err("invalid typed hybrid rows or context".into()); }
            if prefill && let Some(workspace)=self.prompt_workspace.clone() {
                let mut workspace=workspace.try_borrow_mut().map_err(|_| "hybrid prompt workspace busy")?;
                let executor=workspace.as_mut().ok_or("hybrid prompt workspace not reserved")?;
                let result=executor.prompt(self,tokens,save_logits);
                *next_layer=self.model.config.layers.len(); return result.map(Some);
            }
            let mut executor = self.verifier.take().map(Ok).unwrap_or_else(|| decode::Decode::with_rows(self.model, self.decode_rows()).map(Box::new))?;
            if prefill {
                let result = executor.prompt(self,tokens,save_logits);
                self.verifier = Some(executor); *next_layer = self.model.config.layers.len();
                return result.map(Some);
            }
            let mut logits = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                let mut rows = [decode::Row { session:self, token, logits:save_logits && i+1==tokens.len() }];
                let result = if rows[0].session.mtp.is_some() {
                    executor.step_with_mtp(rows[0].session,token,save_logits && i+1==tokens.len())
                } else { executor.step(&mut rows) };
                if result.is_err() || !executor.rejected().is_empty() { self.verifier = Some(executor); return result.and(Err(MOE_ROUTING_ERROR.into())); }
                logits = result?.pop().unwrap();
            }
            self.verifier = Some(executor); *next_layer = self.model.config.layers.len();
            return Ok(Some(logits));
        }
        let start = Instant::now();
        let model = self.model;
        let (b, c) = (model.backend, &model.config);
        let count = tokens.len();
        if *next_layer == 0 {
            self.finish_mtp_verification()?;
            if !self.valid || count == 0 || count > model.prefill_chunk_size()
                || self.position.checked_add(count).is_none_or(|n| n > model.capacity)
                || tokens.iter().any(|&t| t as u64 >= c.vocab) {
                return Err("invalid hybrid session, token IDs or context capacity; drop after an error".into());
            }
            if !prefill {
                self.begin_decode()?;
            }
            self.reserve_workspace(count)?;
            self.reserve_kv(self.position + count)?;
            self.reserve_state()?;
            if count == 1 && self.attention.reserve_decode_scores(b, c.query_heads as usize, 1, self.position + 1, model.capacity)? {
                self.graphs = [None, None];
            }
            self.valid = false;
            let h = c.hidden as usize;
            let n = count * h;
            self.hidden.resize_within_capacity(n)?; self.normalized.resize_within_capacity(n)?;
            self.block.resize_within_capacity(n)?;
            self.gate.resize_within_capacity(if c.fp8 { count * c.intermediate as usize } else if count == 1 { c.intermediate as usize }
                else { count.div_ceil(16) * 16 * c.intermediate as usize / 2 })?;
            self.ids.resize_within_capacity(count)?;
            // The shared embedding kernel reads numeric FP32 token IDs.
            b.write_vector(&mut self.ids, &tokens.iter().map(|&t| t as f32).collect::<Vec<_>>())?;
        }
        if count == 1 && !prefill && self.position > 0 && !b.is_profiling() {
            b.write_vector(self.graph_position.as_mut().unwrap(), &[0.0, 0.0, 0.0, f32::from_bits(self.position as u32)])?;
            let index = usize::from(save_logits);
            if self.graphs[index].is_none() {
                // Prefill created every state buffer. Recording changes no GPU
                // state; the first replay performs this token exactly once.
                let position = self.graph_position.take().unwrap();
                let captured = unsafe { b.capture_decode(&position, self.position, c.layers.len(), count,
                    || self.enqueue_forward(count, false, save_logits)) };
                self.graph_position = Some(position);
                self.graphs[index] = Some(captured?);
            }
            // Session-owned scratch, KV and recurrence never move after capture.
            unsafe { b.replay_decode(self.graphs[index].as_ref().unwrap())?; }
        } else if !self.enqueue_forward_slice(count, prefill, save_logits, None, None, next_layer, budget)? {
            return Ok(None);
        }
        let logits = if save_logits { b.read_vector(&self.logits)? }
            else { b.synchronize()?; Vec::new() };
        // The session stays invalid: a rejected router row fails this request.
        if let Some(moe) = &self.moe { moe.check(b, count)?; }
        if self.mtp.is_some() { self.mtp_observe(tokens)?; }
        self.position += count;
        if !prefill { self.prefill_tail.extend_from_slice(tokens); }
        self.valid = true;
        if count == 1 && !prefill {
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            self.draft_policy.plain_ms = if self.draft_policy.plain_ms == 0.0 { ms } else { self.draft_policy.plain_ms * 0.75 + ms * 0.25 };
        }
        Ok(Some(logits))
    }
    fn enqueue_forward(&mut self, count: usize, prefill: bool, save_logits: bool) -> Result<()> {
        self.enqueue_forward_rows(count, prefill, save_logits, None, None)
    }
    fn enqueue_forward_rows(&mut self, count: usize, prefill: bool, save_logits: bool,
        rows: Option<&mut [prefill::Row<'_, '_, 'a, 'b>]>, gdn_rows: Option<&DeviceVector<'_>>) -> Result<()> {
        self.enqueue_forward_slice(count, prefill, save_logits, rows, gdn_rows, &mut 0, None).map(|_| ())
    }
    fn enqueue_forward_slice(&mut self, count: usize, prefill: bool, save_logits: bool,
        mut rows: Option<&mut [prefill::Row<'_, '_, 'a, 'b>]>, gdn_rows: Option<&DeviceVector<'_>>,
        next_layer: &mut usize, budget: Option<Duration>) -> Result<bool> {
        let started = Instant::now();
        let model = self.model;
        let (b, c) = (model.backend, &model.config);
        let h = c.hidden as usize;
        let fused = c.experts == 0 && !c.fp8 && rows.is_none() && count.is_multiple_of(256) && h.is_multiple_of(128);
        // Retain the CUDA chat reduction; the larger prefill envelope uses
        // four independent accumulators. Pooling chats must not change it.
        let four = rows.is_none() && count >= 256;
        let prompt_lengths: Vec<_> = rows.as_deref().map_or_else(Vec::new,
            |rows| rows.iter().map(|row| row.tokens.len()).collect());
        if *next_layer == 0 {
            b.batch_embedding_device(model.matrix(&format!("{PREFIX}.embed_tokens.weight"))?, &self.ids, &mut self.hidden)?;
        }
        for (layer, kind) in c.layers.iter().enumerate().skip(*next_layer) {
            model.profile_reads(Some(layer), || rows.as_deref().map_or_else(|| vec![(self.position, count)], |r| r.iter().map(|r| (r.session.position, r.tokens.len())).collect()));
            let p = format!("{PREFIX}.layers.{layer}");
            if layer == 0 {
                Self::normalize_order(b, model, &self.hidden, model.matrix(&format!("{p}.input_layernorm.weight"))?.device_address(), &self.normalized, count, None, four)?;
                if fused { b.pack_prefill_bf16_device(&self.normalized, &mut self.input_pack, h, count)?; }
            }
            match kind {
                LayerKind::DeltaNet => {
                    let a = format!("{p}.linear_attn");
                    let fused_gates = fused && !c.fp8 && arms::Limits::fused_gates(c.linear_value_heads) && c.linear_value_heads % 8 == 0;
                    if fused_gates {
                        let fresh = self.gdn_cache[layer].is_none();
                        let gdn = Gdn::load_dense(b, &model.weights, c, layer, self.gdn_cache[layer].take())?;
                        if fresh { self.cache_bytes += gdn.state_bytes(); }
                        let [qkv, z, ga, gb] = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
                            .map(|name| model.matrix(&format!("{a}.{name}.weight")));
                        b.profile_phase(27, || gdn.project_packed(b, &mut self.gdn,
                            [qkv?, z?, ga?, gb?], &self.input_pack, count))?;
                        self.gdn_cache[layer] = Some(gdn.into_state());
                    } else {
                    for (name, out, reuse) in [("in_proj_qkv", &mut self.gdn.qkv, false), ("in_proj_z", &mut self.gdn.z, true),
                        ("in_proj_a", &mut self.gdn.a, true), ("in_proj_b", &mut self.gdn.b, true)] {
                        let matrix = model.matrix(&format!("{a}.{name}.weight"))?;
                        out.resize_within_capacity(count * matrix.rows())?;
                        if matrix.rows() <= 48 && rows.is_some() {
                            if c.fp8 {
                                let mut offset = 0;
                                for row in rows.as_deref().unwrap() {
                                    let n = row.tokens.len();
                                    Self::project(b, matrix, &self.normalized.slice(offset * h..(offset + n) * h)?,
                                        &mut out.slice(offset * matrix.rows()..(offset + n) * matrix.rows())?,
                                        &mut self.weight_pack, &mut self.input_pack, n)?;
                                    offset += n;
                                }
                            } else {
                                // Pool rows without the long-prefill K partitions:
                                // each short prompt keeps the same MMA sum order.
                                b.profile_phase(0, || {
                                    b.matmul_prefill_native_short_device(matrix, &self.normalized, out,
                                        &mut self.input_pack, count, reuse)?;
                                    b.round_logits_bf16(out)
                                })?;
                            }
                        } else {
                            Self::project_reusing(b, matrix, &self.normalized, out, &mut self.weight_pack, &mut self.input_pack, count,
                                (fused || reuse) && !c.fp8)?;
                        }
                    }
                    }
                    self.gdn.gated.resize_within_capacity(count * c.linear_value_heads as usize * c.linear_value_dim as usize)?;
                    if let Some(rows) = rows.as_deref_mut() {
                        if c.fp8 {
                            // FP8 checkpoints retain their canonical chunk sums
                            // and official FP32 parameter reads for each prompt.
                            let mut offset = 0;
                            for row in rows {
                                let (s, n) = (&mut row.session, row.tokens.len());
                                let fresh = s.gdn_cache[layer].is_none();
                                let mut gdn = Gdn::load_dense(b, &model.weights, c, layer, s.gdn_cache[layer].take())?;
                                if fresh { s.cache_bytes += gdn.state_bytes(); }
                                for (source, out, width) in [(&self.gdn.qkv, &mut s.gdn.qkv, (2 * c.linear_key_heads + c.linear_value_heads) as usize * c.linear_value_dim as usize),
                                    (&self.gdn.z, &mut s.gdn.z, (c.linear_value_heads * c.linear_value_dim) as usize),
                                    (&self.gdn.a, &mut s.gdn.a, c.linear_value_heads as usize),
                                    (&self.gdn.b, &mut s.gdn.b, c.linear_value_heads as usize)] {
                                    out.resize_within_capacity(n * width)?;
                                    b.copy_range(source, out, offset * width, 0, n * width)?;
                                }
                                s.gdn.gated.resize_within_capacity(n * (c.linear_value_heads * c.linear_value_dim) as usize)?;
                                b.profile_phase(18, || gdn.run_projected(b, &mut s.gdn,
                                    &self.normalized.slice(offset * h..(offset + n) * h)?, n, true, false, 0))?;
                                b.copy_range(&s.gdn.gated, &mut self.gdn.gated, 0,
                                    offset * (c.linear_value_heads * c.linear_value_dim) as usize, s.gdn.gated.len())?;
                                s.gdn_cache[layer] = Some(gdn.into_state());
                                offset += n;
                            }
                        } else {
                            let weights = [model.matrix(&format!("{a}.conv1d.weight"))?.device_address(),
                                model.address(&format!("{a}.A_log"))?, model.address(&format!("{a}.dt_bias"))?,
                                model.address(&format!("{a}.norm.weight"))?];
                            let offset = layer * decode::MAX_ROWS * 30;
                            let descriptors = gdn_rows.ok_or("shared prompts require GDN descriptors")?
                                .slice(offset..offset + rows.len() * 30)?;
                            let longest = rows.iter().map(|r| r.tokens.len()).max().unwrap();
                            b.profile_phase(18, || self.gdn.prefill_batch(b, weights, &descriptors, rows.len(), longest, c.epsilon as f32, model.scalar_precision(&a)))?;
                        }
                    } else {
                        let fresh = self.gdn_cache[layer].is_none();
                        let mut gdn = Gdn::load_dense(b, &model.weights, c, layer, self.gdn_cache[layer].take())?;
                        if fresh { self.cache_bytes += gdn.state_bytes(); }
                        b.profile_phase(18, || gdn.run_projected(b, &mut self.gdn, &self.normalized, count, prefill || self.position == 0, fused_gates,
                            if fused && !c.fp8 { self.input_pack.device_address() } else { 0 }))?;
                        self.gdn_cache[layer] = Some(gdn.into_state());
                    }
                    let matrix = model.matrix(&format!("{a}.out_proj.weight"))?;
                    if fused {
                        if c.fp8 { b.pack_prefill_bf16_device(&self.gdn.gated, &mut self.input_pack, matrix.cols(), count)?; }
                        b.profile_phase(0, || Self::residual(b, matrix, &self.weight_pack, &self.input_pack, &mut self.hidden, &mut self.block, count))?;
                    } else {
                        Self::project(b, matrix, &self.gdn.gated, &mut self.block, &mut self.weight_pack, &mut self.input_pack, count)?;
                    }
                }
                LayerKind::SparseAttention => {
                    self.attention.resize(c, count)?;
                    let a = format!("{p}.self_attn");
                    if fused {
                        let q = model.matrix(&format!("{a}.q_proj.weight"))?;
                        let k = model.matrix(&format!("{a}.k_proj.weight"))?;
                        let v = model.matrix(&format!("{a}.v_proj.weight"))?;
                        if b.prefill_fp8_direct(&[q, k, v]) {
                            b.profile_phase(0, || b.prefill_qkv_packed_fp8_device(q, k, v,
                                &self.input_pack, &mut self.attention.qg, &mut self.attention.k, &mut self.attention.v, count))?;
                        } else {
                        let q = b.prefill_weight_bf16(q, &self.weight_pack, 0)?;
                        let k = b.prefill_weight_bf16(k, &self.weight_pack, q.rows() * q.cols() / 2)?;
                        let v = b.prefill_weight_bf16(v, &self.weight_pack, (q.rows() + k.rows()) * q.cols() / 2)?;
                        b.profile_phase(0, || b.prefill_qkv_packed_bf16_device(&q, &k, &v,
                            &self.input_pack, &mut self.attention.qg, &mut self.attention.k, &mut self.attention.v, count))?;
                        }
                    } else if count > 1 && !c.fp8 {
                        b.profile_phase(0, || b.prefill_qkv_bf16_device(model.matrix(&format!("{a}.q_proj.weight"))?,
                            model.matrix(&format!("{a}.k_proj.weight"))?, model.matrix(&format!("{a}.v_proj.weight"))?,
                            &self.normalized, &mut self.input_pack, &mut self.attention.qg, &mut self.attention.k, &mut self.attention.v, count))?;
                    } else {
                        for (name, out) in [("q_proj", &mut self.attention.qg), ("k_proj", &mut self.attention.k), ("v_proj", &mut self.attention.v)] {
                            Self::project(b, model.matrix(&format!("{a}.{name}.weight"))?, &self.normalized, out, &mut self.weight_pack, &mut self.input_pack, count)?;
                        }
                    }
                    let norms = [model.matrix(&format!("{a}.q_norm.weight"))?.device_address(), model.matrix(&format!("{a}.k_norm.weight"))?.device_address()];
                    if let Some(rows) = rows.as_deref_mut() {
                        let mut offset = 0;
                        for row in rows {
                            let (s, n) = (&mut row.session, row.tokens.len());
                            s.attention.resize(c, n)?;
                            for (source, out, width) in [(&self.attention.qg, &mut s.attention.qg, 2 * (c.query_heads * c.head_dim) as usize),
                                (&self.attention.k, &mut s.attention.k, (c.kv_heads * c.head_dim) as usize),
                                (&self.attention.v, &mut s.attention.v, (c.kv_heads * c.head_dim) as usize)] {
                                b.copy_range(source, out, offset * width, 0, n * width)?;
                            }
                            s.attention.attend(b, c, s.full_cache[layer].as_mut().unwrap(), norms,
                                model.frequencies.device_address(), n, s.position, 0)?;
                            b.copy_range(&s.attention.gated, &mut self.attention.gated, 0,
                                offset * (c.query_heads * c.head_dim) as usize, s.attention.gated.len())?;
                            offset += n;
                        }
                    } else {
                        self.attention.attend(b, c, self.full_cache[layer].as_mut().unwrap(), norms,
                            model.frequencies.device_address(), count, self.position,
                            if fused && !c.fp8 { self.input_pack.device_address() } else { 0 })?;
                    }
                    let matrix = model.matrix(&format!("{a}.o_proj.weight"))?;
                    if fused {
                        if c.fp8 { b.pack_prefill_bf16_device(&self.attention.gated, &mut self.input_pack, matrix.cols(), count)?; }
                        b.profile_phase(0, || Self::residual(b, matrix, &self.weight_pack, &self.input_pack, &mut self.hidden, &mut self.block, count))?;
                    } else {
                        Self::project(b, matrix, &self.attention.gated, &mut self.block, &mut self.weight_pack, &mut self.input_pack, count)?;
                    }
                }
            }
            self.residual_norm(model.matrix(&format!("{p}.post_attention_layernorm.weight"))?.device_address(), count, fused, four)?;
            if let Some(moe) = &mut self.moe {
                moe.run(model, layer, &self.normalized, &mut self.block, &mut self.weight_pack,
                    &mut self.input_pack, count, !prefill || count < 256 || rows.is_some(), &prompt_lengths)?;
            } else if c.fp8 {
                self.up.resize_within_capacity(count * c.intermediate as usize)?;
                for (name, out, reuse) in [("gate_proj", &mut self.gate, false), ("up_proj", &mut self.up, true)] {
                    Self::project_reusing(b, model.matrix(&format!("{p}.mlp.{name}.weight"))?, &self.normalized,
                        out, &mut self.weight_pack, &mut self.input_pack, count, reuse)?;
                }
                launch(b, "flash_silu", self.gate.len(), &mut [Ptr(self.gate.device_address()), Ptr(self.up.device_address()), U32(self.gate.len() as u32), F32(1.0), U32(1)])?;
                Self::project(b, model.matrix(&format!("{p}.mlp.down_proj.weight"))?, &self.gate,
                    &mut self.block, &mut self.weight_pack, &mut self.input_pack, count)?;
            } else if count == 1 {
                b.gate_up_swiglu_bf16_device(model.matrix(&format!("{p}.mlp.gate_proj.weight"))?,
                    model.matrix(&format!("{p}.mlp.up_proj.weight"))?, &self.normalized, &mut self.gate, &mut self.up)?;
                Self::project(b, model.matrix(&format!("{p}.mlp.down_proj.weight"))?, &self.gate, &mut self.block, &mut self.weight_pack, &mut self.input_pack, count)?;
            } else {
                let gate = model.matrix(&format!("{p}.mlp.gate_proj.weight"))?;
                let up = model.matrix(&format!("{p}.mlp.up_proj.weight"))?;
                let gate = b.prefill_weight_bf16(gate, &self.weight_pack, 0)?;
                let up = b.prefill_weight_bf16(up, &self.weight_pack, gate.rows() * gate.cols() / 2)?;
                if fused {
                    b.prefill_gate_up_packed_bf16_device(&gate, &up, &self.input_pack, &mut self.gate, count)?;
                } else {
                    b.prefill_gate_up_bf16_device(&gate, &up, &self.normalized, &mut self.gate, &mut self.input_pack, count)?;
                }
                let down = model.matrix(&format!("{p}.mlp.down_proj.weight"))?;
                if fused {
                    b.profile_phase(17, || Self::residual(b, down, &self.weight_pack, &self.gate, &mut self.hidden, &mut self.block, count))?;
                } else {
                    let down = b.prefill_weight_bf16(down, &self.weight_pack, 0)?;
                    b.profile_phase(17, || b.matmul_packed_bf16_device(&down, &self.gate, &mut self.block, count))?;
                }
            }
            if layer + 1 < c.layers.len() {
                self.residual_norm(model.matrix(&format!("{PREFIX}.layers.{}.input_layernorm.weight", layer + 1))?.device_address(), count, fused, four)?;
            } else if !fused {
                b.profile_phase(7, || launch(b, "flash_residual_bf16", self.hidden.len(), &mut [Ptr(self.hidden.device_address()), Ptr(self.block.device_address()), U32(self.hidden.len() as u32)]))?;
            }
            *next_layer = layer + 1;
            if let Some(budget) = budget.filter(|_| *next_layer < c.layers.len()) {
                b.synchronize()?;
                if started.elapsed() >= budget { return Ok(false); }
            }
        }
        if save_logits {
            model.profile_reads(None, Vec::new);
            b.copy_range(&self.hidden, &mut self.final_row, (count - 1) * h, 0, h)?;
            self.normalized.resize_within_capacity(h)?;
            Self::normalize_order(b, model, &self.final_row, model.matrix(&format!("{PREFIX}.norm.weight"))?.device_address(), &self.normalized, 1, Some(if prefill && count > 1 { 0 } else { 2 }), false)?;
            b.profile_output_head(|| Self::project(b, model.matrix("lm_head.weight")?, &self.normalized, &mut self.logits,
                &mut self.weight_pack, &mut self.input_pack, 1))?;
        }
        Ok(true)
    }
}
