use engine_model::{plan::{ConfigPlan,TextDescription,description::*,admission::*,layer::{Topology,Kind,cast::Dtype},
    state::{StatePlan,SessionState},storage::WeightStorage},safetensors::{Tensor,Dtype as StoredDtype},shapes::Ffn};
use spark_tp::layer_driver::{LayerBoundary,PhysicalSpan,Packing,MhcOperands};
use engine_fabric::exchange::Exchange;
use std::{collections::BTreeMap,sync::Arc};

fn description() -> TextDescription {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else { panic!() };
    d.hidden=8; d.vocab=16; d.max_positions=8; d.declared_draft_layers=0;
    d.residual=SinkhornDescription {streams:2,iterations:4,epsilon:1e-5};
    d.layers=vec![DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:2,head_dim:4,conv:4,gate_lower_bound:-5.0}),ffn:Ffn::Dense {width:6}}];
    d.storage.layers.clear(); d.storage.excluded.clear(); d
}
fn headers(d: &TextDescription) -> BTreeMap<String,Tensor> {
    let mut offset=0; let mut out=BTreeMap::new();
    let mut insert=|name: String,dtype: StoredDtype,shape: Vec<u64>| {
        let start=offset; offset+=shape.iter().product::<u64>()*dtype.bytes();
        out.insert(name,Tensor {dtype,shape,start,end:offset});
    };
    for (name,w) in d.required_weights().unwrap() {
        if w.storage!=WeightStorage::Plain {
            let module=name.strip_suffix(".weight").unwrap();
            match w.storage {
                WeightStorage::Mxfp8=>{insert(format!("{module}.weight_scale"),StoredDtype::U8,vec![w.shape[0],w.shape[1]/32]);},
                WeightStorage::Fp8Block {block:[r,c]}=>{insert(format!("{module}.weight_scale"),StoredDtype::F32,vec![w.shape[0].div_ceil(r),1,w.shape[1].div_ceil(c),1]);},
                _=>panic!("fixture supports plain, MXFP8, and FP8 block headers"),
            }
            insert(name,StoredDtype::F8E4M3,w.shape);continue;
        }
        let dtype=if name.ends_with("_base") || name.ends_with("_scale") || name.ends_with("A_log") || name.ends_with("dt_bias")
            || name.ends_with("e_score_correction_bias") {StoredDtype::F32} else {StoredDtype::Bf16};
        let mut shape=w.shape;
        if name.contains("_conv1d.weight") {shape.insert(1,1);}
        insert(name,dtype,shape);
    } out
}
fn state(d: &TextDescription,topology: Topology) -> StatePlan {StatePlan::new(d,topology,4,isize::MAX as usize).unwrap()}
fn payload(c: &ComponentView) -> engine_model::Result<Vec<u8>> {
    let pattern=match c.dtype {StoredDtype::Bf16=>0x8000u16.to_le_bytes().to_vec(),StoredDtype::F32=>0x3f800001u32.to_le_bytes().to_vec(),_=>panic!()};
    Ok(pattern.into_iter().cycle().take(c.bytes as usize).collect())
}
fn spans(p: &MhcOperands,rank: u32) -> [PhysicalSpan;7] {
    let s=p.shape(); let mut address=4096;
    [s.wide(),p.weights().len(),p.base().len(),p.scale().len(),s.wide(),s.coefficients(),s.hidden as usize].map(|words| {
        let out=PhysicalSpan {address,words,context:7,rank,dtype:Dtype::F32,packing:Packing::RowMajor}; address+=words as u64*4+64; out
    })
}

#[test]
fn physical_headers_and_every_mhc_argument_are_bound() {
    let d=description(); let a=d.admit_headers(&headers(&d)).unwrap();
    for rank in 0..2 {
        let t=Topology {parts:2,rank}; let s=state(&d,t); let b=LayerBoundary::new(&d,&a,&s,0,1,t,1024).unwrap();
        let mut read=Vec::new(); let p=b.prepare_mhc(0,|c| {read.push(c.name.clone());payload(c)}).unwrap();
        assert_eq!(read,b.weights(0).unwrap().iter().flat_map(|w| w.view().components().iter().map(|c| c.name.clone())).collect::<Vec<_>>());
        assert_eq!(p.stored_dtypes(),[StoredDtype::Bf16,StoredDtype::F32,StoredDtype::F32]);
        assert!(p.weights().iter().all(|v| v.to_bits()==0x80000000));
        assert!(p.base().iter().chain(p.scale()).all(|v| v.to_bits()==0x3f800001));
        let regions=spans(&p,rank); let args=p.arguments(regions).unwrap();
        assert_eq!([args.input,args.weights,args.base,args.scale,args.normalized,args.coefficients,args.collapsed],regions.map(|r| r.address));
        assert_eq!((args.shape.rows,args.shape.streams,args.shape.hidden,args.shape.iterations),(1,2,8,4));
        assert_eq!(args.shape.rms_eps.to_bits(),(d.epsilon as f32).to_bits());
        assert_eq!(args.shape.eps.to_bits(),(d.residual.epsilon as f32).to_bits());
        let q=b.program().steps.iter().position(|s| s.id=="q").unwrap();
        let rows=b.weights(q).unwrap()[0].rows().unwrap();
        assert_eq!(rows.input_features(),d.hidden); assert_eq!(rows.rows(),u64::from(rank)*4..u64::from(rank+1)*4);
        let conv=b.program().steps.iter().position(|s| s.kind==Kind::Conv).unwrap();
        assert_eq!(b.weights(conv).unwrap()[0].view().components()[0].shape,[8,1,4]);
        assert!(b.require_executable().unwrap_err().contains("conditional"));
    }
}

#[test]
fn mhc_operands_bind_for_each_decoder_form_without_admitting_other_operators() {
    use engine_model::shapes::{ExpertShape,ExpertStorage};
    for latent in [false,true] {for experts in [false,true] {
        let mut d=description();d.routing.groups=2;d.routing.selected_groups=1;
        if latent {d.layers[0].mixer=TokenMixer::LatentAttention(LatentDescription {
            heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,
                select_tail:true,share_for_draft_iteration:false} });}
        if experts {d.layers[0].ffn=Ffn::Experts(ExpertShape {count:4,selected:2,width:6,shared_width:4,storage:ExpertStorage::PerExpert});}
        let a=d.admit_headers(&headers(&d)).unwrap();
        for rank in 0..2 {
            let t=Topology {parts:2,rank};let s=state(&d,t);let b=LayerBoundary::new(&d,&a,&s,0,3,t,1024).unwrap();
            assert!(b.require_executable().is_err());
            for (i,step) in b.program().steps.iter().enumerate() {
                if step.kind==Kind::MhcPrepare {
                    let p=b.prepare_mhc(i,payload).unwrap();p.arguments(spans(&p,rank)).unwrap();
                } else {
                    assert!(b.prepare_mhc(i,|_| panic!("unbound operation read payloads")).is_err());
                }
            }
        }
    }}
}

#[test]
fn each_span_rejects_wrong_extent_alignment_context_rank_and_overlap() {
    let d=description();let a=d.admit_headers(&headers(&d)).unwrap();let t=Topology {parts:2,rank:1};let s=state(&d,t);
    let b=LayerBoundary::new(&d,&a,&s,0,1,t,1024).unwrap();let p=b.prepare_mhc(0,payload).unwrap();let good=spans(&p,t.rank);
    for i in 0..good.len() {
        for case in 0..9 {
            let mut bad=good;
            match case {0=>bad[i].words-=1,1=>bad[i].words+=1,2=>bad[i].address+=1,
                3=>bad[i].context=0,4=>bad[i].rank=0,5=>bad[i].address=0,6=>bad[i].address=u64::MAX-3,
                7=>bad[i].dtype=Dtype::Bf16,_=>bad[i].packing=Packing::Bf16Fragments}
            assert!(p.arguments(bad).is_err(),"span {i}, case {case}");
        }
        for j in 0..i {let mut bad=good;bad[i].address=bad[j].address;assert!(p.arguments(bad).err().unwrap().contains("overlap"));}
    }
    let mut bad=good;bad[2].context=99;assert!(p.arguments(bad).is_err());
}

#[test]
fn saved_config_dense_layer_prepares_metadata_without_weight_payloads() {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    // Keep the first decoder's dimensions and recipes. Omit other layers from
    // this header fixture. These generated headers are not real shard evidence.
    d.layers.truncate(1);d.declared_draft_layers=0;
    let a=d.admit_headers(&headers(&d)).unwrap();
    for rank in 0..2 {
        let t=Topology {parts:2,rank};let s=state(&d,t);let b=LayerBoundary::new(&d,&a,&s,0,1,t,usize::MAX).unwrap();
        assert_eq!(b.mhc_shape(0).unwrap().hidden,u32::try_from(d.hidden).unwrap());
        for (i,step) in b.program().steps.iter().enumerate() {for (bound,w) in b.weights(i).unwrap().iter().zip(&step.weights) {
            assert_eq!(bound.view().logical().shape,w.shape);
            if let Some(rows)=bound.rows() {assert_eq!(rows.input_features(),w.shape[1]);}
        }}
        assert!(b.require_executable().is_err());
    }
}

#[test]
fn unsupported_steps_refuse_before_payload_reads() {
    let d=description();let a=d.admit_headers(&headers(&d)).unwrap();let t=Topology {parts:1,rank:0};let s=state(&d,t);
    let b=LayerBoundary::new(&d,&a,&s,0,1,t,1024).unwrap();
    for (i,step) in b.program().steps.iter().enumerate() {
        if step.kind==Kind::MhcPrepare {continue;}
        let error=b.prepare_mhc(i,|_| panic!("unsupported operation read a weight")).err().unwrap();
        assert!(error.contains(&step.id) && error.contains("missing checked device ABI"),"{error}");
    }
    assert!(b.mhc_shape(usize::MAX).is_err());
    assert!(b.prepare_mhc(0,|c| Ok(vec![0;c.bytes as usize-1])).is_err());
    // A physical F32 function tensor stays F32. Header dtype controls the widening.
    let mut h=headers(&d);let name="model.language_model.layers.0.hc_attn_fn";
    let tensor=h.get_mut(name).unwrap();tensor.dtype=StoredDtype::F32;tensor.end=tensor.start+tensor.shape.iter().product::<u64>()*4;
    let a=d.admit_headers(&h).unwrap();let b=LayerBoundary::new(&d,&a,&s,0,1,t,1024).unwrap();
    let p=b.prepare_mhc(0,payload).unwrap();assert!(p.weights().iter().all(|v| v.to_bits()==0x3f800001));
}

#[test]
fn changed_headers_state_and_scratch_layouts_refuse() {
    let d=description();let a=d.admit_headers(&headers(&d)).unwrap();let t=Topology {parts:2,rank:0};let s=state(&d,t);
    let b=LayerBoundary::new(&d,&a,&s,0,1,t,1024).unwrap();
    let offsets=b.scratch().entries().iter().map(|e| e.bytes.start).collect::<Vec<_>>();
    b.check_scratch_offsets(&offsets,b.scratch().bytes()).unwrap();
    let mut bad=offsets.clone();let out=b.scratch().entries().iter().position(|e| e.name=="attn.coefficients").unwrap();
    bad[out]=bad[0];assert!(b.check_scratch_offsets(&bad,b.scratch().bytes()).unwrap_err().contains("overlap"));
    bad=offsets.clone();bad[0]+=1;assert!(b.check_scratch_offsets(&bad,b.scratch().bytes()).is_err());
    assert!(b.check_scratch_offsets(&offsets,b.scratch().bytes()-1).is_err());
    assert!(LayerBoundary::new(&d,&a,&s,0,5,t,1024).is_err());
    assert!(LayerBoundary::new(&d,&a,&s,0,1,Topology {parts:2,rank:1},1024).is_err());
    assert!(LayerBoundary::new(&d,&a,&s,0,1,t,0).is_err());
    let mut different=d.clone();different.epsilon*=2.0;
    assert!(LayerBoundary::new(&different,&a,&s,0,1,t,1024).err().unwrap().contains("different operation program"));
    let mut different=d.clone();different.hidden=10;let other=state(&different,t);
    assert!(LayerBoundary::new(&different,&a,&other,0,1,t,1024).err().unwrap().contains("admitted headers"));
    let mut wrong=headers(&d);wrong.get_mut("model.language_model.layers.0.hc_attn_fn").unwrap().dtype=StoredDtype::F16;
    assert!(d.admit_headers(&wrong).is_err());
}

#[test]
fn refused_layer_preserves_saved_and_committed_state() {
    let d=description();let a=d.admit_headers(&headers(&d)).unwrap();let t=Topology {parts:1,rank:0};let s=Arc::new(state(&d,t));
    let mut session=SessionState::new(s.clone()).unwrap();let saved=session.save(s.state_layout().bytes()).unwrap();
    let before=s.planes().iter().map(|p| session.read(p.layer,p.kind).unwrap().to_vec()).collect::<Vec<_>>();
    {
        let b=LayerBoundary::new(&d,&a,session.plan(),0,1,t,1024).unwrap();
        assert!(b.require_executable().is_err());
        assert_eq!(b.casts().inputs.iter().find(|p| p.buffer.name=="kda.state").unwrap().dtype,Dtype::F32);
        assert_eq!(b.casts().inputs.iter().find(|p| p.buffer.name=="kda.history").unwrap().dtype,Dtype::Unresolved);
        // A live plan cannot be replaced through this immutable session borrow.
        assert_eq!(b.state(),s.as_ref());
    }
    {
        let mut append=session.begin_append().unwrap();let plane=&s.planes()[0];
        append.write(plane.layer,plane.kind,&vec![0x6b;plane.used_bytes(1).unwrap()]).unwrap();
        // Failure before the complete layer must drop the append, never commit it.
        let b=LayerBoundary::new(&d,&a,&s,0,append.context(),t,1024).unwrap();assert!(b.require_executable().is_err());
    }
    assert_eq!(session.context(),0);
    for (p,want) in s.planes().iter().zip(&before) {assert_eq!(session.read(p.layer,p.kind).unwrap(),want);}
    session.restore(&saved).unwrap();assert_eq!(session.context(),0);
}

struct Wire {packets: Vec<Vec<u8>>,calls: usize,fail: bool}
impl Exchange for Wire {
    fn gather_bytes(&mut self,b: &mut [u8],offsets: &[usize]) -> Result<(),String> {
        self.calls+=1;
        for (i,p) in self.packets.iter().enumerate().rev() {b[offsets[i]..offsets[i+1]].copy_from_slice(p);}
        if self.fail {Err("injected exchange failure".into())} else {Ok(())}
    }
    fn gather_f32(&mut self,_: &mut [f32],_: &[usize]) -> Result<(),String> {panic!("unexpected cast")}
    fn reduce(&mut self,_: &mut [f32],_: &mut [f32]) -> Result<(),String> {panic!("unexpected reduction")}
}
#[test]
fn every_declared_gather_uses_global_rank_order_and_exact_bf16_bytes() {
    let d=description();let a=d.admit_headers(&headers(&d)).unwrap();
    for parts in [1,2] {for rank in 0..parts {
        let t=Topology {parts,rank};let s=state(&d,t);let b=LayerBoundary::new(&d,&a,&s,0,1,t,1024).unwrap();
        let declared=b.program().steps.iter().enumerate().flat_map(|(i,s)| s.gather.iter().map(move |n| (i,n.as_str()))).collect::<Vec<_>>();
        assert_eq!(b.gathers().iter().map(|g| (g.step(),g.name())).collect::<Vec<_>>(),declared);
        for g in b.gathers() {
            assert_eq!(g.dtype(),Dtype::Bf16);let layout=g.layout();
            let bytes=(0..layout.bytes()).map(|i| (i*37) as u8).collect::<Vec<_>>();
            let packets=(0..parts).map(|r| bytes[layout.packet(r).unwrap()].to_vec()).collect::<Vec<_>>();
            let local=packets[rank as usize].clone();let mut wire=Wire {packets,calls:0,fail:false};let mut got=vec![0;bytes.len()];
            b.gather(g.step(),g.name(),&mut wire,&local,&mut got).unwrap();assert_eq!(got,bytes);
            wire.fail=true;got.fill(0xa5);assert!(b.gather(g.step(),g.name(),&mut wire,&local,&mut got).is_err());assert!(got.iter().all(|v| *v==0xa5));
            let calls=wire.calls;assert!(b.gather(usize::MAX,g.name(),&mut wire,&local,&mut got).is_err());assert_eq!(wire.calls,calls);
        }
    }}
}

#[test]
fn chain_shadow_targets_the_first_static_reader_of_each_boundary() {
    use engine_model::shapes::{ExpertShape,ExpertStorage};
    for latent in [false,true] {for experts in [false,true] {
        let mut d=description();d.routing.groups=2;d.routing.selected_groups=1;
        if latent {d.layers[0].mixer=TokenMixer::LatentAttention(LatentDescription {
            heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
            indexer:PoolIndexerDescription {heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,
                select_tail:true,share_for_draft_iteration:false} });}
        if experts {d.layers[0].ffn=Ffn::Experts(ExpertShape {count:4,selected:2,width:6,shared_width:4,storage:ExpertStorage::PerExpert});}
        let a=d.admit_headers(&headers(&d)).unwrap();let t=Topology {parts:2,rank:0};let s=state(&d,t);
        let b=LayerBoundary::new(&d,&a,&s,0,3,t,1024).unwrap();let steps=&b.program().steps;let mut found=0;
        for (i,step) in steps.iter().enumerate() {
            if step.kind!=Kind::MhcPrepare {continue;}
            let name=spark_tp::bounded_decoder::shadow_target(steps,i).unwrap_or_else(||panic!("no target latent={latent} experts={experts} step={i}"));
            let norm=&steps[i+2].outputs[0].name;
            let target=steps.iter().position(|s|s.kind==Kind::Project && s.weights.len()==1 && s.weights[0].name==name).unwrap();
            assert!(target>i+2 && steps[target].inputs[0]==*norm,"target reads the boundary output");
            assert!(steps[i+3..target].iter().all(|s|s.kind!=Kind::MhcPrepare),"target precedes the next boundary");
            assert!(!name.contains(".experts."),"routed expert rows are never a target: {name}");
            found+=1;
        }
        assert_eq!(found,2,"both boundaries of the layer have a target");
    }}
    // A boundary at the end of a program, or one without a norm, has no target.
    let d=description();let a=d.admit_headers(&headers(&d)).unwrap();let t=Topology {parts:2,rank:0};let s=state(&d,t);
    let b=LayerBoundary::new(&d,&a,&s,0,3,t,1024).unwrap();let steps=&b.program().steps;
    assert!(spark_tp::bounded_decoder::shadow_target(steps,steps.len()).is_none());
    assert!(spark_tp::bounded_decoder::shadow_target(&steps[..2],0).is_none());
}
