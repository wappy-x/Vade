//! Typed rank bindings feeding the existing decode layer loop in every phase.
use super::*;
use engine_model::{plan::{ModelPlan, runtime::{Projection, ProjectionStorage}}, shapes::{Ffn, ExpertStorage, Mixer, Place, Part}, split::{SplitPlan, SplitRequest}};
use spark_gpu_probe::fp8_abi::*;

pub(super) struct Bound<'a, 'b> {
    pub(super) canonical: BTreeMap<String, linear::CanonicalBf16<'a, 'b>>,
    projections: BTreeMap<String, (usize, Option<usize>)>,
    tables: Vec<ExpertTable<'a>>, steps: Vec<Option<decode::gdn::StepBinding>>, pub(super) group:tp::Group, pub(super) norms:BTreeMap<String,DeviceVector<'a>>,
}
struct ExpertTable<'a> { device: DeviceVector<'a>, projections: Vec<ExpertProjection> }
impl<'a, 'b> Model<'a, 'b> {
    pub(super) fn require_world(world: usize) -> Result<()> { arms::Limits::world(world) }
    /// Explicit typed entry; Model::load selects the same loader for per-expert FP8.
    pub fn load_generic(b: &'b Backend<'a>, root: &Path, capacity: usize, world: usize) -> Result<Self> {
        Self::load_generic_with_mtp(b,root,capacity,world,false)
    }
    pub fn load_generic_with_mtp(b: &'b Backend<'a>, root: &Path, capacity: usize, world: usize, mtp: bool) -> Result<Self> {
        let (p,index,headers)=ModelPlan::inspect_parts(root).map_err(|e| e.to_string())?;
        Self::load_typed(b,root,capacity,world,mtp,p,index,headers)
    }
    pub(super) fn load_typed(b: &'b Backend<'a>, root: &Path, capacity: usize, world: usize, mtp: bool,
        p: ModelPlan, index: engine_model::safetensors::Index, headers: BTreeMap<String, engine_model::safetensors::Header>) -> Result<Self> {
        Self::require_world(world)?;
        let split = SplitPlan::new(&p.shape, &SplitRequest { ranks:world as u32, memory:vec![u64::MAX;world], reserve:vec![0;world], context:capacity as u64,
            sequences:1, cache_bytes:2, state_bytes:4, draft:mtp, expert_rows:false }).map_err(|e| e.to_string())?;
        Self::load_typed_group(b,root,capacity,mtp,p,&index,&headers,0,split,tp::Group::local)
    }
    /// Rank entry remains refused at world2 until the complete typed state/service proof lands.
    pub fn load_generic_rank<E:engine_fabric::exchange::Exchange+'static>(b:&'b Backend<'a>,root:&Path,capacity:usize,mtp:bool,
        p:ModelPlan,index:&engine_model::safetensors::Index,headers:&BTreeMap<String,engine_model::safetensors::Header>,split:SplitPlan,
        rank:usize,members:&[u64],incarnation:u64,exchange:E)->Result<Self> {
        Self::require_world(split.ranks as usize)?;
        Self::load_typed_group(b,root,capacity,mtp,p,index,headers,rank,split,|p,s|tp::Group::new(p,s,rank,members,incarnation,exchange))
    }
    fn load_typed_group(b:&'b Backend<'a>,root:&Path,capacity:usize,mtp:bool,p:ModelPlan,index:&engine_model::safetensors::Index,
        headers:&BTreeMap<String,engine_model::safetensors::Header>,rank:usize,split:SplitPlan,group:impl FnOnce(&ModelPlan,SplitPlan)->Result<tp::Group>)->Result<Self> {
        let draft_options = crate::hybrid_draft::Options::startup();
        let mut c = plan::config(&p)?;
        let arms = arms::Table::new(&p,split.ranks as usize,mtp); arms.validate(&p)?;
        arms::Limits::fixed(&c)?;
        if !matches!(p.format, engine_model::plan::Format::Fp8 { .. }) { return Err("typed routed forward requires block FP8 storage".into()); }
        for tensor in &p.shape.tensors { coverage::use_of(tensor)?; }
        // Public entries enforce world admission; internal tests qualify this same loader.
        if c.tied_embeddings || c.vocab>1<<24 || capacity == 0 || capacity as u64 > c.max_positions {
            return Err("typed hybrid requires world1, untied text head and valid context".into());
        }
        if mtp { mtp_contract::validate(&p)?; }
        linear::NormBinding::new(c.hidden as usize, c.epsilon as f32)?;
        let group=group(&p,split)?; let owner=&group.split().owners[rank];
        let mut steps=Vec::new();
        for (layer, shape) in p.shape.layers.iter().enumerate() {
            if !matches!(shape.ffn, Ffn::Experts(e) if e.storage == ExpertStorage::PerExpert && e.shared_width > 0) { return Err("typed forward requires separate routed and shared experts".into()); }
            match shape.mixer {
                Mixer::Linear(_) => { let step=decode::gdn::StepBinding::owned(&p,layer,owner)?; step.execution_ready()?; steps.push(Some(step)); },
                Mixer::Attention(a) => { crate::flash::qsa::attention::FullAttentionBinding::new(a, &p.rope, p.epsilon as f32, owner.query_heads.start as u32..owner.query_heads.end as u32, owner.kv_heads.start as u32..owner.kv_heads.end as u32)?; steps.push(None); },
            }
        }
        if mtp { if let Mixer::Attention(a)=p.shape.draft[0].mixer {
            crate::flash::qsa::attention::FullAttentionBinding::new(a,&p.rope,p.epsilon as f32,owner.query_heads.start as u32..owner.query_heads.end as u32,owner.kv_heads.start as u32..owner.kv_heads.end as u32)?;
        } }
        if group.distributed() { b.set_graph_storage_budget(0)?; }
        b.require_layer_kernels()?; b.require_batch_kernels()?;
        arms.setup(|arm| b.require_arm_kernels(arm.kernels()))?;
        for projection in p.hybrid_projections().map_err(|e| e.to_string())? {
            if (matches!(projection.weight.role.place,Place::Model|Place::Layer(_)) || mtp && matches!(projection.weight.role.place,Place::Draft {..})) && projection.weight.role.part!=Part::Embedding {
                if matches!(projection.storage,ProjectionStorage::Bf16) { linear::bf16_boundary(projection.weight.role)?; }
            }
        }
        let load = weights::LoadPlan::for_rank(&p, index, headers, group.split(),rank as u32)?;
        let bytes = load.bytes(); let resident = load.load(b, root)?;
        c.query_heads=owner.query_heads.end-owner.query_heads.start; c.kv_heads=owner.kv_heads.end-owner.kv_heads.start;
        c.linear_key_heads=owner.key_heads.end-owner.key_heads.start; c.linear_value_heads=owner.value_heads.end-owner.value_heads.start;
        c.intermediate=resident.fp8[&format!("{PREFIX}.layers.0.mlp.experts.0.gate_proj.weight")].rows as u64;
        c.shared_intermediate=resident.matrices[&format!("{PREFIX}.layers.0.mlp.shared_expert.gate_proj.weight")].rows() as u64;
        let mut norms=BTreeMap::new();
        if group.distributed() { for t in tp::Binding::new(&p,group.split(),rank)?.norms() {
            let view=&load.tensors()[&t.name]; let mut local=vec![0;view.bytes]; resident.arena.read(view.offset,&mut local)?;
            let full=group.norm(t,&local)?; let mut device=b.vector(full.len().div_ceil(4))?; b.write_vector_bytes(&mut device,&full)?; norms.insert(t.name.clone(),device);
        } }
        let mut weights = resident.matrices; let mut canonical = BTreeMap::new(); let mut projections = BTreeMap::new();
        let inventory: BTreeMap<_,_> = p.shape.tensors.iter().enumerate().map(|(i,t)| (t.name.as_str(),i)).collect();
        for projection in p.hybrid_projections().map_err(|e| e.to_string())? {
            if !(matches!(projection.weight.role.place, Place::Model | Place::Layer(_)) || mtp && matches!(projection.weight.role.place,Place::Draft {..})) || projection.weight.role.part == Part::Embedding { continue; }
            let name = &projection.weight.name;
            let scale = match projection.storage { ProjectionStorage::Bf16 => {
                linear::bf16_boundary(projection.weight.role)?;
                canonical.insert(name.clone(), unsafe { linear::CanonicalBf16::prepare(b, weights.remove(name).ok_or("BF16 projection absent")?)? }); None
            }, ProjectionStorage::Fp8 { scales, .. } => Some(inventory[scales.name.as_str()]) };
            projections.insert(name.clone(), (inventory[name.as_str()], scale));
        }
        let mut tables = Vec::new();
        for layer in 0..c.layers.len()+usize::from(mtp) {
            let prefix = mtp_contract::block_prefix(c.layers.len(),layer);
            let mut descriptors: Vec<_> = (0..c.experts).flat_map(|e| ["gate", "up", "down"].map(|part| format!("{prefix}.mlp.experts.{e}.{part}_proj.weight")))
                .map(|n| resident.fp8.get(&n).copied().ok_or("routed FP8 projection absent")).collect::<std::result::Result<_,_>>()?;
            if !group.distributed() && c.shared_intermediate == c.intermediate {
                let shared = ["gate","up","down"].map(|part| resident.fp8.get(&format!("{prefix}.mlp.shared_expert.{part}_proj.weight")).copied());
                if shared.iter().all(Option::is_some) { descriptors.extend(shared.into_iter().flatten()); }
            }
            let mut table = b.vector(descriptors.len()*10)?;
            let bytes: Vec<_> = descriptors.iter().flat_map(|d| d.weights.to_le_bytes().into_iter().chain(d.scales.to_le_bytes())
                .chain([d.rows,d.cols,d.scale_cols,d.first_k_block,d.first_n_block,d.reserved].into_iter().flat_map(u32::to_le_bytes))).collect();
            b.write_vector_bytes(&mut table, &bytes)?; tables.push(ExpertTable { device:table,projections:descriptors });
        }
        let mut frequencies = b.vector((c.rotary_dim/2) as usize)?;
        b.write_vector(&mut frequencies, &(0..c.rotary_dim/2).map(|i| 1.0/(c.rope_theta as f32).powf(2.0*i as f32/c.rotary_dim as f32)).collect::<Vec<_>>())?;
        Ok(Self { backend:b, config:c, weights, experts:Vec::new(), frequencies, capacity, weight_bytes:bytes, mtp, draft_options, arms,
            bound:Some(Bound { group, norms, canonical, projections, tables, steps }), profile_plan:Some(p) })
    }
    pub(super) fn step_binding(&self, layer:usize) -> Result<decode::gdn::StepBinding> {
        if let Some(bound)=&self.bound { bound.steps.get(layer).and_then(Clone::clone).ok_or("GDN layer binding absent".into()) }
        else { decode::gdn::StepBinding::whole(self.model_plan(),layer) }
    }
    pub(super) fn fixed_project(&self, name: &str, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, rows: usize) -> Result<()> {
        self.fixed_project_reusing(name, input, output, scratch, rows, false)
    }
    // Paired FP8 projections retain input and scratch; BF16 invalidates packed reuse.
    pub(super) fn fixed_project_reusing(&self, name: &str, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, rows: usize, reuse: bool) -> Result<()> {
        let bound = self.bound.as_ref().ok_or("typed forward binding absent")?;
        let &(index, scale) = bound.projections.get(name).ok_or("unconsumed projection role")?;
        let weight = &self.model_plan().shape.tensors[index];
        let [n,k] = if let Some(receipt)=bound.canonical.get(name) { receipt.dimensions() }
            else { let m=self.matrix(name)?; [m.rows() as u64,m.cols() as u64] };
        let projection = Projection { weight, rows:n, cols:k, storage:scale.map_or(ProjectionStorage::Bf16,
            |s| ProjectionStorage::Fp8 { scales:&self.model_plan().shape.tensors[s], block:[128,128] }) };
        for range in schedule::ranges(rows) {
            let input = input.slice(range.start*projection.cols as usize..range.end*projection.cols as usize)?;
            let mut output = output.slice(range.start*projection.rows as usize..range.end*projection.rows as usize)?;
            if let Some(receipt) = bound.canonical.get(name) { receipt.project(&projection, &input, &mut output, scratch, range.len())?; }
            else { unsafe { linear::project_reusing(self.backend, &projection, self.matrix(name)?, &input, &mut output, scratch, range.len(),
                if matches!(weight.role.part, Part::Shared(engine_model::shapes::Proj::Down) | Part::Head) || bound.group.distributed() && matches!(weight.role.part,Part::Output|Part::LinearOut) { linear::OutputBoundary::Fp32 } else { linear::OutputBoundary::Bf16 }, reuse && rows <= schedule::PROMPT_GROUP)?; } }
        }
        Ok(())
    }
    // Only Fp32-boundary canonical BF16 pairs may share this unrounded launch.
    pub(super) fn fixed_project_pair(&self, names: [&str;2], input: &DeviceVector<'_>, outputs: [&mut DeviceVector<'_>;2], scratch: &mut DeviceVector<'_>, rows: usize) -> Result<bool> {
        let bound = self.bound.as_ref().ok_or("typed forward binding absent")?;
        let eligible = names.iter().all(|n| bound.projections.get(*n).is_some_and(|&(i,s)| s.is_none()
            && linear::bf16_boundary(self.model_plan().shape.tensors[i].role) == Ok(linear::OutputBoundary::Fp32)));
        let (Some(first),Some(second)) = (bound.canonical.get(names[0]),bound.canonical.get(names[1])) else { return Ok(false); };
        if !eligible { return Ok(false); }
        let [a,b]=outputs; first.project_pair_rows(second, input, a, b, scratch, rows)
    }
}

pub(super) struct Moe<'a> { vectors: Vec<DeviceVector<'a>>, layout: RoutingLayout, padded: usize }
impl<'a> Moe<'a> {
    pub(super) fn allocations(&self, out:&mut Vec<(usize,usize)>) { for v in &self.vectors { v.allocations(out); } }
    pub(super) fn new(model: &Model<'a, '_>, tokens: usize) -> Result<Option<Self>> {
        if model.bound.is_none() || !model.arms.arm(arms::Kind::MoeFp8).on { return Ok(None); }
        let c = &model.config; let (k,i) = (c.selected_experts as usize,c.intermediate as usize);
        let sizes=Self::lengths(c,tokens);
        Ok(Some(Self { vectors:model.backend.vector_arena(&sizes)?, layout:RoutingLayout::checked(k as u32,i as u32).ok_or("routed layout")?, padded:sizes[11] }))
    }
    fn lengths(c: &Config, tokens:usize) -> [usize;24] {
        let (e,k,h,i,s)=(c.experts as usize,c.selected_experts as usize,c.hidden as usize,c.intermediate as usize,c.shared_intermediate as usize);
        let pairs=tokens*k; let padded=pairs+15*pairs.min(e); let tasks=pairs.div_ceil(128)+pairs.min(e);
        [tokens*e,pairs,pairs,tokens,pairs*3,e,e+1,pairs,pairs,tokens,1+2*tasks,padded,padded,tokens*(k+1)*h,
            (tokens*h).div_ceil(4),tokens*h/128,padded*2*i,(padded*i).div_ceil(4),padded*i/128,padded*h,tokens,tokens*s,tokens*s,tokens*h]
    }
    pub(super) fn reservation_bytes(c: &Config, tokens:usize) -> usize {
        if c.experts==0 || !c.fp8 { 0 } else { Self::lengths(c,tokens).iter().map(|n| (n*4).div_ceil(64)*64).sum::<usize>() + (4*128+2)*c.linear_value_heads as usize*4+384 }
    }
    pub(super) fn rejected(&self, b: &Backend<'_>, rows: usize) -> Result<Vec<usize>> {
        Ok(spark_gpu_probe::routing::rejected_rows(&b.read_vector_values(&self.vectors[3].slice(0..rows)?)?))
    }
    pub(super) fn selections(&self, b: &Backend<'_>, selected: &DeviceVector<'_>, rows: usize)
        -> Result<(Vec<Result<(u32,f32)>>,Vec<usize>)> {
        b.read_argmax_status(&selected.slice(0..rows*3)?,&self.vectors[3].slice(0..rows)?,rows)
    }
    pub(super) fn status(&self, rows: usize) -> Result<DeviceVector<'a>> { self.vectors[3].slice(0..rows) }
    // The fused boundary moves rows*(k+1)*h partials through pageable copies;
    // that beats one host sync only at decode/verify row counts. Prompt passes
    // keep the early consensus and the in-place pinned exchange.
    fn decode_shaped(rows: usize) -> bool { schedule::fused_router_boundary(true, rows, decode::MAX_VERIFY_ROWS) }
    pub(super) fn run(&mut self, model: &Model<'a, '_>, layer: usize, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>, scratch: &mut DeviceVector<'_>, rows: usize) -> Result<()> {
        let (b,c) = (model.backend,&model.config); let (e,k,h,s) = (c.experts as usize,c.selected_experts as usize,c.hidden as usize,c.shared_intermediate as usize);
        let p = format!("{}.mlp",mtp_contract::block_prefix(c.layers.len(),layer));
        let bound = model.bound.as_ref().unwrap();
        let small = rows <= 4 && !bound.group.distributed();
        let fused = bound.group.distributed() && Self::decode_shaped(rows);
        // Router and shared-expert gate read the same normalized input, which
        // the routed block never overwrites, so one paired BF16 launch at
        // decode rows replaces the gate projection at the end of this block.
        let paired = { let (scores,rest)=self.vectors.split_at_mut(20);
            model.fixed_project_pair([&format!("{p}.gate.weight"),&format!("{p}.shared_expert_gate.weight")],input,[&mut scores[0],&mut rest[0]],scratch,rows)? };
        if !paired { model.fixed_project(&format!("{p}.gate.weight"), input, &mut self.vectors[0], scratch, rows)?; }
        if layer==0 || layer==c.layers.len() { b.clear_moe_routing(&mut self.vectors[3], rows)?; }
        let mut route = spark_gpu_probe::kernel_params::MoeRouteBatch { scores:self.vectors[0].device_address(),
            indices:self.vectors[1].device_address(), weights:self.vectors[2].device_address(),
            status:self.vectors[3].device_address(), experts:e as u32, selected:k as u32, tokens:rows as u32,
            renormalize:u32::from(model.model_plan().renormalize_top_k) };
        if !small { unsafe { b.moe_route_rows(&mut route,self.vectors[4].device_address())?; } }
        if bound.group.distributed() && !fused {
            let route=b.read_vectors_values(&[self.vectors[3].slice(0..rows)?,self.vectors[1].slice(0..rows*k)?,self.vectors[2].slice(0..rows*k)?])?;
            let rejected=spark_gpu_probe::routing::rejected_rows(&route[0]);
            let ids=route[1].iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let weights=route[2].iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let votes=bound.group.routes(layer,rows,&ids,&weights,&(0..rows).map(|r|rejected.contains(&r)).collect::<Vec<_>>())?;
            b.write_vector(&mut self.vectors[3].slice(0..rows)?,&votes.into_iter().map(|r|f32::from_bits(u32::from(r))).collect::<Vec<_>>())?;
        }
        let fused_route = small.then_some(route);
        let a = |i:usize| self.vectors[i].device_address();
        let route = GroupedRouteParams { error:a(3),pairs:a(4),counts:a(5),starts:a(6),pair_to_row:a(7),token_pairs:a(8),token_count:a(9),tasks:a(10),sorted_row:a(11),output_row:a(12),
            experts:e as u32,tokens:rows as u32,pair_count:(rows*k) as u32,row_capacity:self.padded as u32 };
        let grouped = GroupedParams { route,weights:bound.tables[layer].device.device_address(),input:input.device_address(),input_scales:0,output:a(13),cols:h as u32,activation_quantization:1,output_shard:0 };
        let shape=&model.model_plan().shape;
        let global_intermediate=match shape.layers.iter().chain(&shape.draft).nth(layer).map(|l|l.ffn) { Some(Ffn::Experts(e))=>e.width, _=>return Err("routed expert shape absent".into()) };
        let shared_fused = small && bound.tables[layer].projections.len() == (e+1)*3;
        let descriptors = &bound.tables[layer].projections[..(e+usize::from(shared_fused))*3];
        unsafe { moe::run_fp8(b,RoutedParams::checked(grouped,self.layout).ok_or("routed forward spans")?, descriptors,global_intermediate,
            [a(14),a(15),a(16),a(17),a(18),a(19)],fused_route)?; }
        if !shared_fused {
            for (part,index) in [("gate",21),("up",22)] { model.fixed_project_reusing(&format!("{p}.shared_expert.{part}_proj.weight"),input,&mut self.vectors[index],scratch,rows,part=="up")?; }
            let gate = self.vectors[21].slice(0..rows*s)?; let up = self.vectors[22].slice(0..rows*s)?;
            launch_silu_precision(b,&gate,&up,3)?;
            model.fixed_project(&format!("{p}.shared_expert.down_proj.weight"),&gate,&mut self.vectors[23],scratch,rows)?;
        }
        if !paired { model.fixed_project(&format!("{p}.shared_expert_gate.weight"),input,&mut self.vectors[20],scratch,rows)?; }
        if bound.group.distributed() {
            // One strided copy places every row's shared output in its last partial slot.
            let (partials,shared) = self.vectors.split_at_mut(23);
            b.copy_strided(&shared[0],&mut partials[13],0,k*h,h,(k+1)*h,h,rows)?;
            if fused {
            // Compute from local routes, then agree before consuming any partials.
            // Routes and FF partials share one download/wait on the producer stream.
            let mut values=b.read_vectors_values(&[self.vectors[3].slice(0..rows)?,self.vectors[1].slice(0..rows*k)?,self.vectors[2].slice(0..rows*k)?,self.vectors[13].slice(0..rows*(k+1)*h)?])?;
            let rejected=spark_gpu_probe::routing::rejected_rows(&values[0]);
            let ids=values[1].iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let weights=values[2].iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let votes=bound.group.routes(layer,rows,&ids,&weights,&(0..rows).map(|r|rejected.contains(&r)).collect::<Vec<_>>())?;
            bound.group.reduce(tp::Step::FeedForward(model.layer_place(layer)),rows,&mut values[3])?;
            // Both uploads precede finish on this stream; OR-voted rows stay gated.
            b.write_vector(&mut self.vectors[3].slice(0..rows)?,&votes.into_iter().map(|r|f32::from_bits(u32::from(r))).collect::<Vec<_>>())?;
            b.write_vector(&mut self.vectors[13].slice(0..rows*(k+1)*h)?,&values[3])?;
            }
        }
        if !fused { model.reduce_partial(tp::Step::FeedForward(model.layer_place(layer)),rows,&mut self.vectors[13].slice(0..rows*(k+1)*h)?)?; }
        let finish = MoeFinishParams { error:self.vectors[3].device_address(),ids:self.vectors[1].device_address(),weights:self.vectors[2].device_address(),partials:self.vectors[13].device_address(),
            shared_gate:self.vectors[20].device_address(),output:output.device_address(),activation_quantization:1,column_start:0,columns:h as u32 };
        let mut finish=OrderedFinishParams::checked(finish,FinishLayout::checked(self.layout,h as u32).ok_or("finish layout")?,rows as u32,0).ok_or("finish rows")?;
        if !bound.group.distributed() && !shared_fused { finish.shared=self.vectors[23].device_address(); }
        unsafe { moe::finish_fp8(b,finish) }
    }
}

impl Model<'_, '_> {
    pub(super) fn layer_place(&self,layer:usize)->Place { if layer<self.config.layers.len() { Place::Layer(layer as u32) } else { Place::Draft{layer:Some(0)} } }
    pub(super) fn reduce_partial(&self,step:tp::Step,rows:usize,v:&mut DeviceVector<'_>)->Result<()> {
        if let Some(bound)=self.bound.as_ref().filter(|b|b.group.distributed()) {
            self.backend.exchange_vector(v,|v|bound.group.reduce(step,rows,v))?;
        } Ok(())
    }
    pub(super) fn embedding_rows(&self,ids:&DeviceVector<'_>,out:&mut DeviceVector<'_>)->Result<()> {
        let matrix=self.matrix(&format!("{PREFIX}.embed_tokens.weight"))?;
        if let Some(bound)=self.bound.as_ref().filter(|b|b.group.distributed()) {
            let ids=self.backend.read_vector_values(ids)?.into_iter().map(|id| { if !id.is_finite() || id<0.0 || id.fract()!=0.0 || id as u64>=self.config.vocab { Err("invalid global embedding ID".into()) } else { Ok(id as u32) } }).collect::<Result<Vec<_>>>()?;
            let full=bound.group.embedding(&ids,|row|self.backend.shared_bf16_row(matrix,row))?;
            self.backend.write_vector(out,&full)
        } else { self.backend.batch_embedding_device(matrix,ids,out) }
    }
    // Decode owns these same validated IDs before uploading them. A distributed
    // embedding needs their host values, so do not download that upload again.
    pub(super) fn embedding_tokens(&self,ids:&[u32],out:&mut DeviceVector<'_>)->Result<()> {
        let bound=self.bound.as_ref().ok_or("typed embedding binding absent")?;
        if ids.len().checked_mul(self.config.hidden as usize)!=Some(out.len()) || ids.iter().any(|&id|u64::from(id)>=self.config.vocab) {
            return Err("invalid global embedding IDs or output extent".into());
        }
        let matrix=self.matrix(&format!("{PREFIX}.embed_tokens.weight"))?;
        let full=bound.group.embedding(ids,|row|self.backend.shared_bf16_row(matrix,row))?;
        self.backend.write_vector(out,&full)
    }
    pub(super) fn head_rows(&self,input:&DeviceVector<'_>,out:&mut DeviceVector<'_>,scratch:&mut DeviceVector<'_>,rows:usize)->Result<()> {
        let bound=self.bound.as_ref().ok_or("typed head binding absent")?;
        let owner=&bound.group.split().owners[bound.group.rank()].vocab; let local_vocab=(owner.end-owner.start) as usize;
        self.fixed_project("lm_head.weight",input,&mut out.slice(0..rows*local_vocab)?,scratch,rows)?;
        if bound.group.distributed() { let local=self.backend.read_vector_values(&out.slice(0..rows*local_vocab)?)?;
            self.backend.write_vector(out,&bound.group.heads(rows,&local)?)?; } Ok(())
    }
}

#[cfg(test)] #[test] fn fused_router_boundary_is_decode_shaped_only() {
    // Plain decode, MTP draft/verify and full decode batches stay fused.
    assert!(Moe::decode_shaped(1) && Moe::decode_shaped(3) && Moe::decode_shaped(decode::MAX_ROWS));
    assert!(Moe::decode_shaped(decode::MAX_VERIFY_ROWS));
    // The first prompt-shaped count and every full prompt pass stay unfused.
    assert!(!Moe::decode_shaped(decode::MAX_VERIFY_ROWS+1));
    assert!(!Moe::decode_shaped(schedule::PROMPT_ROWS) && !Moe::decode_shaped(schedule::PROMPT_GROUP));
}
