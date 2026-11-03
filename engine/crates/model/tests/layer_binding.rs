use engine_model::{plan::{ConfigPlan, TextDescription, description::*, layer::{*, reference::*}, storage::WeightStorage},
    shapes::{Ffn,ExpertShape,ExpertStorage}};
use std::collections::BTreeMap;

fn description(latent: bool, experts: bool) -> TextDescription {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else { panic!() };
    d.hidden=8; d.max_positions=64; d.residual=SinkhornDescription { streams:2,iterations:4,epsilon:1e-5 };
    d.layers=vec![DecoderDescription { mixer:if latent { TokenMixer::LatentAttention(LatentDescription {
        heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
        indexer:PoolIndexerDescription { heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false } }) }
        else { TokenMixer::Kda(KdaDescription { heads:2,head_dim:4,conv:4,gate_lower_bound:-5.0 }) },
        ffn:if experts { Ffn::Experts(ExpertShape { count:4,selected:2,width:6,shared_width:4,storage:ExpertStorage::PerExpert }) }
            else { Ffn::Dense { width:6 } } }];
    d.routing.groups=2; d.routing.selected_groups=1;
    // The synthetic host tensors have decoded FP32 values. They have no packed recipe.
    d.storage.layers.clear(); d.storage.excluded.clear(); d
}
fn plan(d: &TextDescription,context: u64) -> LayerPlan { LayerPlan::new(d,0,context,Topology { parts:1,rank:0 }).unwrap() }
struct Random(u64);
impl Random {
    fn values(&mut self,count: usize) -> Vec<f32> { (0..count).map(|_| { self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0>>40) as i32-8388608) as f32*(0.25/8388608.0) }).collect() }
}
fn fixture(p: &LayerPlan,seed: u64) -> (Tensors,Tensors) {
    let mut random=Random(seed); let mut weights=BTreeMap::new();
    for w in p.steps.iter().flat_map(|s| &s.weights) { let count=Buffer::new(&w.name,&w.shape).elements().unwrap();
        weights.entry(w.name.clone()).or_insert_with(|| Tensor::f32(&w.shape,random.values(count)).unwrap()); }
    let inputs=p.inputs.iter().map(|b| { let tensor=if b.element==Element::U32 { Tensor { shape:b.shape.clone(),data:Data::U32((0..b.elements().unwrap() as u32).collect()) } }
        else { Tensor::f32(&b.shape,random.values(b.elements().unwrap())).unwrap() }; (b.name.clone(),tensor) }).collect();
    (weights,inputs)
}
#[test]
fn random_layer_forward_matches_plan_order_shapes_and_replay() {
    for latent in [false,true] { for experts in [false,true] { for context in [1,2,3,7,8] { for seed in [1,19,903] {
        let d=description(latent,experts); let p=plan(&d,context); let (w,x)=fixture(&p,seed);
        let f=forward(&d,0,context,&w,&x).unwrap(); f.check_plan(&p).unwrap();
        let replay=replay(&d,&p,&w,&x).unwrap(); assert_eq!(f.trace,replay.trace); assert_eq!(f.tensors,replay.tensors);
        let output=f.tensors["output"].floats().unwrap(); assert_eq!(f.tensors["output"].shape,[2,8]);
        assert!(output.iter().all(|v| v.is_finite())); assert_ne!(output,x["residual"].floats().unwrap());
        if latent { let Data::U32(ids)=&f.tensors["selected"].data else { panic!() };
            let Data::U32(count)=&f.tensors["selected_count"].data else { panic!() };
            assert!(ids[..count[0] as usize].iter().all(|&id| id<context as u32));
            assert!(ids[..count[0] as usize].windows(2).all(|a| a[0]<a[1]));
        }
        println!("LAYER_TRACE latent={latent} experts={experts} context={context} seed={seed} ops={} output={:?}",p.steps.len(),f.tensors["output"].shape);
    } } } }
}
#[test]
fn independent_trace_rejects_reordered_missing_and_reshaped_ops() {
    let d=description(true,true); let p=plan(&d,7); let (w,x)=fixture(&p,77); let f=forward(&d,0,7,&w,&x).unwrap();
    let mut bad=p.clone(); bad.steps.swap(0,1); assert!(f.check_plan(&bad).is_err());
    let mut bad=p.clone(); bad.steps.remove(8); assert!(f.check_plan(&bad).is_err());
    let mut bad=p.clone(); bad.steps[0].outputs[0].shape[0]+=1; assert!(f.check_plan(&bad).is_err());
    let mut bad=p.clone(); bad.steps[0].weights[0].shape[0]+=1; assert!(f.check_plan(&bad).is_err());
    let mut bad=p.clone(); bad.settings.swiglu_limit*=2.0; assert!(f.check_plan(&bad).is_err());
    let mut bad=p.clone(); bad.steps.iter_mut().find(|s| s.id=="expert_middle").unwrap().gather.clear();
    assert!(bad.validate().is_err());
    let mut bad=x.clone(); bad.get_mut("residual").unwrap().shape=vec![16]; assert!(forward(&d,0,7,&w,&bad).unwrap().check_plan(&p).is_err());
    let mut bad=w.clone(); bad.remove(p.steps[0].weights[0].name.as_str()); assert!(forward(&d,0,7,&bad,&x).is_err());
}
#[test]
fn every_operation_rejects_missing_and_extra_operands_before_replay() {
    let mut kinds=Vec::new();
    for latent in [false,true] { for experts in [false,true] { for shared in [false,true] {
        let mut d=description(latent,experts);
        if !shared { if let Ffn::Experts(e)=&mut d.layers[0].ffn { e.shared_width=0; } }
        let p=plan(&d,7); let (w,x)=fixture(&p,5);
        replay(&d,&p,&w,&x).unwrap();
        for (index,step) in p.steps.iter().enumerate() {
            if !kinds.contains(&step.kind) { kinds.push(step.kind); }
            for field in ["inputs","weights","outputs"] { for extra in [false,true] {
                if field=="weights" && !extra && step.weights.is_empty() { continue; }
                let mut bad=p.clone(); let s=&mut bad.steps[index];
                match field {
                    "inputs" => { if extra { s.inputs.push(s.inputs[0].clone()); } else { s.inputs.pop(); } },
                    "weights" => { if extra { s.weights.push(p.steps[0].weights[0].clone()); } else { s.weights.pop(); } },
                    _ => { if extra { s.outputs.push(s.outputs[0].clone()); } else { s.outputs.pop(); } },
                }
                assert_eq!(bad.validate().unwrap_err().to_string(),"layer operation arity differs","{} {field} extra={extra}",step.id);
                assert_eq!(replay(&d,&bad,&w,&x).unwrap_err().to_string(),"layer operation arity differs");
            } }
        }
    } } }
    for kind in [Kind::MhcPrepare,Kind::Collapse,Kind::Norm,Kind::Project,Kind::Conv,Kind::KdaPrepare,Kind::KdaStep,Kind::GatedNorm,
        Kind::CacheAppend,Kind::KeyNorm,Kind::Pool,Kind::ScorePools,Kind::Select,Kind::Gather,Kind::ExpandKv,Kind::Scores,Kind::Softmax,
        Kind::Values,Kind::Route,Kind::ExpertProject,Kind::SwiGlu,Kind::ExpertDown,Kind::ExpertSum,Kind::Combine] {
        assert!(kinds.contains(&kind),"untested operation {kind:?}");
    }
}
#[test]
fn tensor_splits_cover_global_axes_and_gathers_precede_global_dots() {
    for latent in [false,true] { for experts in [false,true] {
        let d=description(latent,experts); let mut plans=Vec::new();
        for rank in 0..2 { plans.push(LayerPlan::new(&d,0,7,Topology { parts:2,rank }).unwrap()); }
        assert_eq!(plans[0].steps,plans[1].steps);
        for step in &plans[0].steps { for (buffer,&split) in step.outputs.iter().zip(&step.output_splits) {
            let interval=plans[0].topology.interval(&buffer.shape,split).unwrap();
            if let Some((axis,start,width))=interval {
                let (other_axis,other_start,other_width)=plans[1].topology.interval(&buffer.shape,split).unwrap().unwrap();
                assert_eq!((axis,start,width),(other_axis,0,other_width)); assert_eq!(other_start,width);
                assert_eq!(width+other_width,buffer.shape[axis]);
            }
        } }
        let gathers=plans[0].steps.iter().flat_map(|s| &s.gather).map(String::as_str).collect::<Vec<_>>();
        assert!(gathers.contains(&if latent {"attended"} else {"gated"})); assert!(gathers.contains(&"attention"));
        if experts { assert!(gathers.contains(&"expert_middle")); assert!(gathers.contains(&"expert_down")); assert!(gathers.contains(&"shared.middle")); }
        else { assert!(gathers.contains(&"dense.middle")); }
        for p in &plans { let life=p.lifetimes(); for name in &p.returns { assert_eq!(life[name].1,p.steps.len()); }
            assert_eq!(p.placements()["input_norm"],Split::Replicated);
        }
    } }
}
#[test]
fn saved_config_schedule_targets_and_gaps_are_explicit() {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else { panic!() };
    let patterns=include_str!("fixtures/glm/index-patterns.txt").lines().map(|line| line.split_whitespace().nth(1).unwrap()).collect::<std::collections::BTreeSet<_>>();
    let mut bound=0; let mut ops=0; let mut missing=0; let mut kinds=std::collections::BTreeSet::new(); let mut kernels=std::collections::BTreeSet::new();
    for index in 0..d.layers.len() { let p=LayerPlan::new(&d,index,7,Topology { parts:2,rank:1 }).unwrap();
        assert!(p.missing().is_empty(),"{index}: {:?}",p.missing());
        assert_eq!(p.steps[0].target.revision,MHC); assert_eq!(p.steps.last().unwrap().kind,Kind::Combine);
        for step in &p.steps { assert!(!step.target.entries.is_empty()); assert_eq!(step.target.revision.len(),40);
            for weight in &step.weights {
                d.bind_tensor(&weight.name).unwrap_or_else(|e| panic!("{}: {e}",weight.name));
                let mut parts=weight.name.split('.').collect::<Vec<_>>();
                for label in ["layers","experts"] { if let Some(at)=parts.iter().position(|&s| s==label) { parts[at+1]=if label=="layers" {"N"} else {"E"}; } }
                assert!(patterns.contains(parts.join(".").as_str()),"absent saved index pattern: {}",weight.name);
                bound+=1;
            }
            kinds.insert(format!("{:?}",step.kind)); for &entry in &step.target.entries { kernels.insert((step.target.revision,entry)); } }
        ops+=p.steps.len(); missing+=p.missing().len();
        println!("MODEL_LAYER index={index} mixer={:?} ffn={:?} ops={} missing={} gathers={:?}",d.layers[index].mixer,d.layers[index].ffn,p.steps.len(),p.missing().len(),p.steps.iter().flat_map(|s| &s.gather).collect::<Vec<_>>());
    }
    println!("MODEL_BINDING layers={} ops={ops} kinds={} kernels={} missing={missing}",d.layers.len(),kinds.len(),kernels.len());
    println!("MODEL_WEIGHT_BINDING bound={bound}");
    for (revision,entry) in kernels { println!("KERNEL_TARGET {revision} {entry}"); }
    // A program is not permission to load or serve this checkpoint.
    assert!(ConfigPlan::Deferred(d).into_model_plan().is_err());
    let small=description(false,false); let p=plan(&small,1);
    for id in ["kda.prepare","kda.step","gated_norm"] { assert!(p.missing().iter().any(|(name,_)| *name==id)); }
}
#[test]
fn pool_gate_uses_the_bare_checkpoint_tensor() {
    let d=description(true,false); let p=plan(&d,1);
    let gate=&p.steps.iter().find(|s| s.id=="index_gate").unwrap().weights[0];
    assert_eq!(gate.name,"model.language_model.layers.0.self_attn.indexer.index_kpool_compress_gate");
    assert_eq!(gate.shape,[3,8]); assert_eq!(gate.storage,WeightStorage::Plain);
    assert!(d.bind_tensor(&gate.name).is_ok()); assert!(d.bind_tensor(&format!("{}.weight",gate.name)).is_err());
    let (w,x)=fixture(&p,6); let f=forward(&d,0,1,&w,&x).unwrap(); f.check_plan(&p).unwrap();
    assert_eq!(f.tensors,replay(&d,&p,&w,&x).unwrap().tensors);
}
#[test]
fn local_slots_require_the_counted_identity_prefix_only() {
    let d=description(true,false);
    for context in [1,7] {
        let p=plan(&d,context); let (w,mut x)=fixture(&p,7);
        let f=forward(&d,0,context,&w,&x).unwrap();
        let Data::U32(count)=&f.tensors["selected_count"].data else { panic!() }; let used=count[0] as usize;
        let Data::U32(slots)=&mut x.get_mut("local_slots").unwrap().data else { panic!() };
        slots[used..].fill(u32::MAX);
        assert_eq!(forward(&d,0,context,&w,&x).unwrap().tensors,replay(&d,&p,&w,&x).unwrap().tensors);
        let Data::U32(slots)=&mut x.get_mut("local_slots").unwrap().data else { panic!() };
        // An ascending offset table still violates the compact slot contract.
        for slot in &mut slots[..used] { *slot+=1; }
        assert_eq!(forward(&d,0,context,&w,&x).unwrap_err().to_string(),"invalid local attention slots");
        assert_eq!(replay(&d,&p,&w,&x).unwrap_err().to_string(),"invalid local attention slots");
    }
}
#[test]
fn lifetimes_record_birth_last_use_and_return_sentinel_without_scratch() {
    let p=plan(&description(true,true),7); let life=p.lifetimes();
    let combine=p.steps.iter().position(|s| s.id=="attn.combine").unwrap();
    assert_eq!(life["residual"],(None,combine));
    assert_eq!(life["attn.coefficients"],(Some(0),combine));
    assert_eq!(life["output"],(Some(p.steps.len()-1),p.steps.len()));
    for name in &p.returns { assert_eq!(life[name].1,p.steps.len()); }
    for scratch in p.steps.iter().flat_map(|s| &s.scratch) { assert!(!life.contains_key(&scratch.name)); }
    assert!(!p.steps.iter().all(|s| s.scratch.is_empty()));
}
#[test]
fn cached_reference_updates_are_reused_on_next_token() {
    for latent in [false,true] { let d=description(latent,true); let p=plan(&d,1); let (w,mut x)=fixture(&p,123);
        for context in 1..9 { let p=plan(&d,context); let f=forward(&d,0,context,&w,&x).unwrap(); f.check_plan(&p).unwrap();
            let same=forward(&d,0,context,&w,&x).unwrap(); assert_eq!(f.tensors,same.tensors);
            let pairs=if latent { vec![("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates")] }
                else { vec![("kda.state","next_state"),("kda.history","next_history")] };
            for (input,output) in pairs { x.insert(input.into(),f.tensors[output].clone()); }
        }
    }
}
#[test]
fn hand_layer_uses_raw_streams() {
    let d=description(false,true); let p=plan(&d,1); let (mut w,mut x)=fixture(&p,2);
    for tensor in w.values_mut() { let Data::F32(v)=&mut tensor.data else { panic!() }; v.fill(0.0); }
    // Zero blocks leave two uniform residual mixes. The denominator epsilon remains.
    x.insert("residual".into(),Tensor::f32(&[2,8],(0..16).map(|j| if j<8 {2.0} else {4.0}).collect()).unwrap());
    let f=forward(&d,0,1,&w,&x).unwrap(); f.check_plan(&p).unwrap();
    let coeff=f.tensors["attn.coefficients"].floats().unwrap();
    let expected=6.0*coeff[4]; let expected=2.0*expected*f.tensors["ffn.coefficients"].floats().unwrap()[4];
    for &v in f.tensors["output"].floats().unwrap() { assert!((v-expected).abs()<1e-6); }
    assert!(f.tensors["ffn_output"].floats().unwrap().iter().all(|&v| v==0.0));
}
#[test]
fn shared_output_is_capped_and_added_once_without_router_weight() {
    let mut d=description(false,true); d.swiglu_limit=1.0; let p=plan(&d,1); let (mut w,x)=fixture(&p,3);
    for tensor in w.values_mut() { let Data::F32(v)=&mut tensor.data else { panic!() }; v.fill(0.0); }
    let prefix="model.language_model.layers.0.";
    let Data::F32(norm)=&mut w.get_mut(&format!("{prefix}post_attention_layernorm.weight")).unwrap().data else { panic!() }; norm.fill(1.0);
    let first=forward(&d,0,1,&w,&x).unwrap(); let value=first.tensors["post_norm"].floats().unwrap()[0];
    assert!(value.abs()>0.01);
    for (name,scale) in [("gate",2.0),("up",3.0)] {
        let Data::F32(v)=&mut w.get_mut(&format!("{prefix}mlp.shared_experts.{name}_proj.weight")).unwrap().data else { panic!() };
        v[0]=scale/value;
    }
    let Data::F32(down)=&mut w.get_mut(&format!("{prefix}mlp.shared_experts.down_proj.weight")).unwrap().data else { panic!() };
    for row in down.chunks_exact_mut(4) { row[0]=1.0; }
    let f=forward(&d,0,1,&w,&x).unwrap(); f.check_plan(&p).unwrap();
    let expected=1.0/(1.0+(-1.0f32).exp());
    for &v in f.tensors["ffn_output"].floats().unwrap() { assert!((v-expected).abs()<1e-6); }
    let mut other=d; other.routing.scale=7.0;
    let g=forward(&other,0,1,&w,&x).unwrap(); assert_eq!(f.tensors["ffn_output"],g.tensors["ffn_output"]);
}
#[test]
fn invalid_geometry_storage_and_topology_fail_without_allocating() {
    let d=description(true,true);
    assert!(LayerPlan::new(&d,1,1,Topology {parts:1,rank:0}).is_err());
    for topology in [Topology {parts:0,rank:0},Topology {parts:2,rank:2},Topology {parts:3,rank:0}] { assert!(LayerPlan::new(&d,0,1,topology).is_err()); }
    for context in [0,65,u64::MAX] { assert!(LayerPlan::new(&d,0,context,Topology {parts:1,rank:0}).is_err()); }
    let mut bad=d.clone(); bad.hidden=u64::MAX; assert!(LayerPlan::new(&bad,0,1,Topology {parts:1,rank:0}).is_err());
    let mut bad=d.clone(); bad.residual.streams=u64::MAX; assert!(LayerPlan::new(&bad,0,1,Topology {parts:1,rank:0}).is_err());
    let mut bad=d.clone(); bad.swiglu_limit=f64::MIN_POSITIVE; assert!(LayerPlan::new(&bad,0,1,Topology {parts:1,rank:0}).is_err());
    let mut bad=d; bad.storage.layers.insert("model.language_model.layers.0.self_attn.q_a_proj".into(),WeightStorage::Fp8Block {block:[128,3]});
    assert!(plan(&bad,1).missing().iter().any(|(id,_)| *id=="query_down"));
}
#[test]
fn replay_rejects_an_invalid_description_with_matching_plan_settings() {
    let mut d=description(false,true); let mut p=plan(&d,1); let (w,x)=fixture(&p,8);
    d.routing.kind=engine_model::host_ops::router::RouterKind::SoftmaxTopK;
    assert!(LayerPlan::new(&d,0,1,p.topology).is_err());
    p.settings=Settings::from_description(&d,0).unwrap();
    p.validate().unwrap();
    assert_eq!(replay(&d,&p,&w,&x).unwrap_err().to_string(),"invalid corrected routing description");
}
