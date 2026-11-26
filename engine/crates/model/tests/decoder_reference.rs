//! Logical stack tests. These fixtures do not contain checkpoint payloads.
use engine_model::plan::{ConfigPlan,TextDescription,description::*,decoder_reference::{Decoder,Output},
    layer::{Buffer,Element,LayerPlan,Topology,reference::{self,Tensor,Tensors,Data}}};
use engine_model::shapes::{Ffn,ExpertShape,ExpertStorage};

fn description() -> TextDescription {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else {panic!()};
    d.hidden=8; d.vocab=5; d.max_positions=12; d.tied_embeddings=false;
    d.residual=SinkhornDescription {streams:2,iterations:4,epsilon:1e-5};
    let kda=TokenMixer::Kda(KdaDescription {heads:2,head_dim:4,conv:4,gate_lower_bound:-5.0});
    let mla=TokenMixer::LatentAttention(LatentDescription {heads:2,query_rank:4,kv_rank:4,key_dim:3,value_dim:2,rotary_dim:0,
        indexer:PoolIndexerDescription {heads:2,dim:3,token_budget:4,pool:2,learned_pool:true,select_tail:true,share_for_draft_iteration:false}});
    let dense=Ffn::Dense {width:6};
    let experts=Ffn::Experts(ExpertShape {count:4,selected:2,width:6,shared_width:4,storage:ExpertStorage::PerExpert});
    d.layers=vec![DecoderDescription {mixer:kda,ffn:dense},DecoderDescription {mixer:mla,ffn:experts},
        DecoderDescription {mixer:kda,ffn:experts},DecoderDescription {mixer:mla,ffn:dense}];
    d.routing.groups=2; d.routing.selected_groups=1; d.storage.layers.clear(); d.storage.excluded.clear(); d
}
fn plan(d: &TextDescription,layer: usize,context: u64) -> LayerPlan {
    LayerPlan::new(d,layer,context,Topology {parts:1,rank:0}).unwrap()
}
fn weights(d: &TextDescription) -> Tensors {
    let mut out=Tensors::new(); let mut seed=17u64;
    let mut add=|name: &str,shape: &[u64]| {
        let values=(0..Buffer::new(name,shape).elements().unwrap()).map(|_| {
            seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed>>40) as i32-8388608) as f32*(0.125/8388608.0)
        }).collect();
        out.insert(name.into(),Tensor::f32(shape,values).unwrap());
    };
    for layer in 0..d.layers.len() { for w in plan(d,layer,1).steps.iter().flat_map(|s| &s.weights) { add(&w.name,&w.shape); } }
    add("model.language_model.embed_tokens.weight",&[d.vocab,d.hidden]);
    add("model.language_model.norm.weight",&[d.hidden]); add("lm_head.weight",&[d.vocab,d.hidden]); out
}
fn bits(t: &Tensor) -> Vec<u32> {t.floats().unwrap().iter().map(|v| v.to_bits()).collect()}
fn same(a: &Output,b: &Output) {
    assert_eq!((a.context,a.input_token,a.next_token),(b.context,b.input_token,b.next_token));
    for (x,y) in [(&a.residual,&b.residual),(&a.hidden,&b.hidden),(&a.normalized,&b.normalized),(&a.logits,&b.logits)] {
        assert_eq!(x.shape,y.shape); assert_eq!(bits(x),bits(y));
    }
}
fn initial(d: &TextDescription) -> Vec<Tensors> {
    (0..d.layers.len()).map(|layer| plan(d,layer,1).inputs.iter().filter(|b| b.name!="residual").map(|b| {
        let size=b.elements().unwrap();
        let value=if b.element==Element::U32 {Tensor {shape:b.shape.clone(),data:Data::U32((0..size as u32).collect())}}
            else {Tensor::f32(&b.shape,vec![0.0;size]).unwrap()};
        (b.name.clone(),value)
    }).collect()).collect()
}
// This loop connects public layer calls directly. It does not call Decoder.
fn manual(d: &TextDescription,w: &Tensors,state: &mut [Tensors],context: u64,token: u32) -> Output {
    let h=d.hidden as usize; let n=d.residual.streams as usize;
    let table=w["model.language_model.embed_tokens.weight"].floats().unwrap();
    let values=table[token as usize*h..(token as usize+1)*h].repeat(n);
    let mut residual=Tensor::f32(&[n as u64,h as u64],values).unwrap();
    for (layer,x) in state.iter_mut().enumerate() {
        x.insert("residual".into(),residual);
        let f=reference::forward(d,layer,context,w,x).unwrap();
        f.check_plan(&plan(d,layer,context)).unwrap(); residual=f.tensors["output"].clone();
        for name in plan(d,layer,context+1).inputs.iter().map(|b| b.name.as_str()).collect::<Vec<_>>() {
            let output=match name {"kda.state"=>"next_state","kda.history"=>"next_history","latent_cache"=>"next_latent_cache",
                "index_keys"=>"next_index_keys","index_gates"=>"next_index_gates",_=>continue};
            x.insert(name.into(),f.tensors[output].clone());
        }
    }
    let raw=residual.floats().unwrap();
    let mean=(0..h).map(|j| (0..n).fold(0.0,|sum,s| sum+raw[s*h+j])/n as f32).collect::<Vec<_>>();
    let inv=1.0/(mean.iter().fold(0.0,|sum,&v| sum+v*v)/h as f32+d.epsilon as f32).sqrt();
    let gain=w["model.language_model.norm.weight"].floats().unwrap();
    let norm=mean.iter().zip(gain).map(|(&v,&w)| (v*inv)*w).collect::<Vec<_>>();
    let head=w[if d.tied_embeddings {"model.language_model.embed_tokens.weight"} else {"lm_head.weight"}].floats().unwrap();
    let logits=head.chunks_exact(h).map(|row| row.iter().zip(&norm).fold(0.0,|sum,(&a,&b)| sum+a*b)).collect::<Vec<_>>();
    let mut best=0; for id in 1..logits.len() {if logits[id]>logits[best] {best=id;}}
    Output {context,input_token:token,residual,hidden:Tensor::f32(&[h as u64],mean).unwrap(),
        normalized:Tensor::f32(&[h as u64],norm).unwrap(),logits:Tensor::f32(&[d.vocab],logits).unwrap(),next_token:best as u32}
}
#[test]
fn stack_matches_direct_layer_composition_and_all_text_edges() {
    let d=description(); let w=weights(&d); let mut state=initial(&d); let mut decoder=Decoder::new(&d,&w,10).unwrap();
    for (index,token) in [1,3,0,4,2,1,3,2].into_iter().enumerate() {
        let expected=manual(&d,&w,&mut state,index as u64+1,token); let actual=decoder.step(token).unwrap(); same(&expected,&actual);
        println!("DECODER_REFERENCE context={} input={} next={} logits={:?}",actual.context,token,actual.next_token,bits(&actual.logits));
    }
}
#[test]
fn chunk_boundaries_saved_state_and_lanes_keep_exact_fp32_bits() {
    let d=description(); let w=weights(&d); let tokens=[1,2,3,4,0,2,1,4];
    let mut alone=Decoder::new(&d,&w,10).unwrap(); let expected=alone.append(&tokens).unwrap();
    for chunk in 1..=tokens.len() {
        let mut lane=Decoder::new(&d,&w,10).unwrap(); let mut actual=Vec::new();
        for part in tokens.chunks(chunk) {actual.extend(lane.append(part).unwrap());}
        for (a,b) in expected.iter().zip(&actual) {same(a,b);}
    }
    let mut lane=Decoder::new(&d,&w,10).unwrap(); let empty=lane.save(); lane.append(&tokens[..3]).unwrap(); let saved=lane.save();
    let first=lane.append(&tokens[3..]).unwrap(); lane.restore(&saved).unwrap();
    assert_eq!(lane.context(),3); assert_eq!(saved.context(),3);
    let second=lane.append(&tokens[3..]).unwrap(); for (a,b) in first.iter().zip(&second) {same(a,b);}
    lane.restore(&empty).unwrap(); let fresh=lane.append(&tokens).unwrap();
    for (a,b) in expected.iter().zip(&fresh) {same(a,b);}
    assert!(alone.restore(&saved).is_err()); assert_eq!(alone.context(),8);
    // Independent lanes may run in interleaved host order.
    let mut a=Decoder::new(&d,&w,10).unwrap(); let mut b=Decoder::new(&d,&w,10).unwrap();
    for (index,&token) in tokens.iter().enumerate() {same(&expected[index],&a.step(token).unwrap()); same(&expected[index],&b.step(token).unwrap());}
}
#[test]
fn invalid_chunk_and_capacity_leave_the_committed_boundary_intact() {
    let d=description(); let w=weights(&d); let mut lane=Decoder::new(&d,&w,3).unwrap();
    lane.step(1).unwrap(); let saved=lane.save();
    assert!(lane.append(&[2,5]).is_err()); assert_eq!(lane.context(),1);
    assert!(lane.append(&[2,3,4]).is_err()); assert_eq!(lane.context(),1);
    let actual=lane.append(&[2,3]).unwrap(); assert_eq!(lane.context(),3);
    assert!(lane.step(0).is_err()); assert!(lane.append(&[]).unwrap().is_empty());
    lane.restore(&saved).unwrap(); let expected=lane.append(&[2,3]).unwrap();
    for (a,b) in actual.iter().zip(&expected) {same(a,b);}
    for capacity in [0,13,u64::MAX] {assert!(Decoder::new(&d,&w,capacity).is_err());}
}
#[test]
fn late_logit_failure_rolls_back_all_layers_and_earlier_tokens_in_chunk() {
    let d=description(); let mut w=weights(&d);
    w.insert("model.language_model.embed_tokens.weight".into(),Tensor::f32(&[5,8],
        (0..40).map(|j| if j<8 {0.0} else {1.0}).collect()).unwrap());
    w.insert("model.language_model.norm.weight".into(),Tensor::f32(&[8],vec![1.0;8]).unwrap());
    w.insert("lm_head.weight".into(),Tensor::f32(&[5,8],vec![f32::MAX;40]).unwrap());
    let mut lane=Decoder::new(&d,&w,10).unwrap(); let mut control=Decoder::new(&d,&w,10).unwrap();
    assert!(lane.append(&[0,1]).is_err()); assert_eq!(lane.context(),0);
    same(&control.step(0).unwrap(),&lane.step(0).unwrap());
    let saved=lane.save(); assert!(lane.step(1).is_err()); assert_eq!(lane.context(),1);
    let actual=lane.step(0).unwrap(); lane.restore(&saved).unwrap(); same(&actual,&lane.step(0).unwrap());
}
#[test]
fn tied_head_ties_and_admission_failures_are_explicit() {
    let mut d=description(); let mut w=weights(&d); d.tied_embeddings=true; w.remove("lm_head.weight");
    let mut lane=Decoder::new(&d,&w,10).unwrap(); let mut state=initial(&d);
    same(&manual(&d,&w,&mut state,1,2),&lane.step(2).unwrap());
    d.tied_embeddings=false; assert!(Decoder::new(&d,&w,10).is_err());
    w.insert("lm_head.weight".into(),Tensor::f32(&[5,8],vec![0.0;40]).unwrap());
    let out=Decoder::new(&d,&w,10).unwrap().step(2).unwrap(); assert_eq!(out.next_token,0); assert_eq!(bits(&out.logits),vec![0;5]);
    let mut bad=w.clone(); bad.get_mut("lm_head.weight").unwrap().shape=vec![40]; assert!(Decoder::new(&d,&bad,10).is_err());
    let mut bad=w.clone(); let Data::F32(v)=&mut bad.get_mut("lm_head.weight").unwrap().data else {panic!()}; v[0]=f32::NAN;
    assert!(Decoder::new(&d,&bad,10).unwrap().step(2).is_err());
    let mut bad=w; bad.remove("model.language_model.layers.3.self_attn.q_a_proj.weight");
    assert!(Decoder::new(&d,&bad,10).is_err());
    let mut bad=d; bad.layers.clear(); assert!(Decoder::new(&bad,&Tensors::new(),10).is_err());
    assert!(ConfigPlan::Deferred(description()).into_model_plan().is_err());
}
#[test]
fn observer_and_deferred_weight_errors_leave_the_lane_unchanged() {
    use engine_model::plan::layer::reference::WeightSource;
    use std::cell::{Cell,RefCell};
    struct Source {weights:Tensors,fail:Cell<bool>,loads:RefCell<Vec<String>>}
    impl WeightSource for Source {
        fn profile(&self) -> engine_model::plan::layer::profile::Profile { self.weights.profile() }
        fn shape(&self,name: &str) -> engine_model::Result<Vec<u64>> {self.weights.shape(name)}
        fn load(&self,name: &str) -> engine_model::Result<Tensor> {
            self.loads.borrow_mut().push(name.into());
            if self.fail.get() && name=="lm_head.weight" {return Err(std::io::Error::other("injected head read failure").into());}
            self.weights.load(name)
        }
    }
    let d=description(); let source=Source {weights:weights(&d),fail:Cell::new(false),loads:RefCell::new(Vec::new())};
    let mut lane=Decoder::new(&d,&source,10).unwrap(); assert!(source.loads.borrow().is_empty());
    let mut visits=Vec::new();
    assert!(lane.append_observed(&[1,2],|context,layer,_| {
        visits.push((context,layer));
        if context==2 && layer+1==d.layers.len() {return Err(std::io::Error::other("injected observer failure").into());}
        Ok(())
    }).is_err());
    assert_eq!(visits,(1..=2).flat_map(|context| (0..d.layers.len()).map(move |layer| (context,layer))).collect::<Vec<_>>());
    assert_eq!(lane.context(),0);
    let mut control=Decoder::new(&d,&source.weights,10).unwrap(); same(&control.step(1).unwrap(),&lane.step(1).unwrap());
    source.fail.set(true); assert!(lane.step(2).is_err()); assert_eq!(lane.context(),1);
    source.fail.set(false); same(&control.step(2).unwrap(),&lane.step(2).unwrap());
}

#[test]
fn runtime_profiles_fail_before_metadata_payloads_or_observers() {
    use engine_model::plan::layer::{profile::{Profile,Boundary,UnsupportedProfile},reference::WeightSource};
    struct Unbound(Profile);
    impl WeightSource for Unbound {
        fn profile(&self) -> Profile {self.0}
        fn shape(&self,_: &str) -> engine_model::Result<Vec<u64>> {panic!("unbound profile must not read headers")}
        fn load(&self,_: &str) -> engine_model::Result<Tensor> {panic!("unbound profile must not read payloads")}
    }
    assert_eq!(Profile::LogicalF32V1.name(),"logical-f32-v1");
    assert_eq!(Profile::LogicalF32V1.require_scalar(),Ok(()));
    let d=description();
    for profile in [Profile::ConditionalBf16,Profile::CheckpointRuntime] {
        assert_eq!(profile.require_scalar(),Err(UnsupportedProfile {requested:profile,boundary:Boundary::EffectiveParameterLoad}));
        let source=Unbound(profile);
        let error=Decoder::new(&d,&source,10).err().unwrap().to_string();
        assert!(error.contains(profile.name()) && error.contains("effective runtime parameter load is unbound"));
        let error=reference::forward_from(&d,0,1,&source,&Tensors::new()).unwrap_err().to_string();
        assert!(error.contains(profile.name()) && error.contains("logical-f32-v1"));
    }
}

#[test]
fn changed_source_profile_rejects_append_without_state_or_value_changes() {
    use engine_model::plan::layer::{profile::Profile,reference::WeightSource};
    use std::cell::Cell;
    struct Source {weights:Tensors,profile:Cell<Profile>,reads:Cell<usize>}
    impl WeightSource for Source {
        fn profile(&self) -> Profile {self.profile.get()}
        fn shape(&self,name: &str) -> engine_model::Result<Vec<u64>> {self.weights.shape(name)}
        fn load(&self,name: &str) -> engine_model::Result<Tensor> {
            self.reads.set(self.reads.get()+1);self.weights.load(name)
        }
    }
    let d=description();
    let source=Source {weights:weights(&d),profile:Cell::new(Profile::LogicalF32V1),reads:Cell::new(0)};
    let mut lane=Decoder::new(&d,&source,10).unwrap();
    let mut control=Decoder::new(&d,&source.weights,10).unwrap();
    same(&control.step(1).unwrap(),&lane.step(1).unwrap());
    let reads=source.reads.get();
    for profile in [Profile::ConditionalBf16,Profile::CheckpointRuntime] {
        source.profile.set(profile);
        assert!(lane.append_observed(&[2,3],|_,_,_| panic!("unbound profile must not observe a layer")).is_err());
        assert!(lane.append(&[]).is_err());assert_eq!(lane.context(),1);assert_eq!(source.reads.get(),reads);
    }
    source.profile.set(Profile::LogicalF32V1);
    same(&control.step(2).unwrap(),&lane.step(2).unwrap());
}
