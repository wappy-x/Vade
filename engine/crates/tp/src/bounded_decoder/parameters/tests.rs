use super::*;
use spark_gpu_probe::causal_ops::{Shape,Op};
fn prepared(words:&[u32])->Rc<StaticParameters> {
    let table=Rc::new(BTreeMap::from([("conv.q".into(),words.iter().copied().map(f32::from_bits).collect())]));
    Rc::new(StaticParameters::new(table,vec!["conv.q".into()]).unwrap())
}
#[derive(Debug,PartialEq,Eq)]
struct Launch {shape:[u32;8],inputs:[u64;4],weight:u64,output:u64,words:Vec<u32>}
#[derive(Clone)]struct Allocation {address:u64,words:Vec<u32>}
fn launch(s:Shape,inputs:[u64;4],output:u64,weight:&Allocation)->Launch {
    Launch {shape:[s.op as u32,s.heads,s.dim,s.taps,s.rows,s.cols,s.epsilon.to_bits(),s.bound.to_bits()],inputs,weight:weight.address,output,words:weight.words.clone()}
}
#[test]
fn prepared_and_content_key_paths_record_identical_launch_arguments_and_weight_bytes() {
    // Use complete, valid weight extents and preserve every exceptional bit pattern.
    let bits=[0,0x80000000,1,0x007fffff,0x00800000,0x7f800000,0xff800000,0x7fc12345,0x7fa12345,0xffc54321];
    let mut baseline=ParameterCache::<Allocation>::default();let mut memo=ParameterCache::<Allocation>::default();
    let mut expected=Vec::new();let mut actual=Vec::new();
    for (index,op) in [Op::ConvParts,Op::ConvChannels,Op::Prepare,Op::GatedNorm,Op::KeyNorm].into_iter().enumerate() {
        let shape=Shape {op,heads:64,dim:128,taps:4,rows:0,cols:0,epsilon:f32::from_bits(0x3727c5ac),bound:-5.0};
        let (lengths,n,_,_)=shape.lengths().unwrap();let words=(0..n).map(|i|bits[i%bits.len()]).collect::<Vec<_>>();let p=prepared(&words);
        let upload=|values:&[f32]|Ok(Allocation {address:0x800000+index as u64*0x100000,words:values.iter().map(|v|v.to_bits()).collect()});
        // A prompt already uploaded these exact contents. Decode must retain that address.
        let old=baseline.resolve(p.values(),None,upload).unwrap().unwrap();let held=memo.resolve(p.values(),None,upload).unwrap().unwrap();
        let before=memo.counts();
        for tick in 0..3u64 {
            let inputs=std::array::from_fn(|i|if lengths[i]==0 {0} else {0x10000+i as u64*0x100000+tick*256});let output=0x1000000+tick*1024;
            let reference=baseline.resolve(p.values(),None,|_|panic!("baseline uploaded again")).unwrap().unwrap();
            let candidate=memo.resolve(p.values(),Some(&p),|_|panic!("memo uploaded again")).unwrap().unwrap();
            assert!(Rc::ptr_eq(&reference,&old));assert!(Rc::ptr_eq(&candidate,&held));
            expected.push(launch(shape,inputs,output,&reference));actual.push(launch(shape,inputs,output,&candidate));
        }
        assert_eq!(memo.counts().key_builds-before.key_builds,0);
    }
    assert_eq!(actual,expected);assert_eq!(actual.len(),15);assert_eq!(&actual[0].words[..bits.len()],&bits);
    assert_eq!(memo.counts().key_builds,5);assert_eq!(baseline.counts().key_builds,20);
    assert_eq!(memo.counts().static_binds,5);assert_eq!(memo.counts().static_hits,10);
}
#[test]
fn different_table_identities_and_hash_collisions_cannot_reuse_wrong_contents() {
    let first=prepared(&[0x80000000,0x7fc00001]);let mut second=prepared(&[0,0x7fc00002]);
    // Deliberate same hash. Identity and the full content map still distinguish it.
    Rc::get_mut(&mut second).unwrap().hash=first.hash;
    let mut cache=ParameterCache::<Vec<u32>>::default();
    let a=cache.resolve(first.values(),Some(&first),|v|Ok(v.iter().map(|v|v.to_bits()).collect())).unwrap().unwrap();
    let b=cache.resolve(second.values(),Some(&second),|v|Ok(v.iter().map(|v|v.to_bits()).collect())).unwrap().unwrap();
    assert_ne!(*a,*b);assert_eq!(*a,first.bits());assert_eq!(*b,second.bits());
    let same=prepared(first.bits());let c=cache.resolve(same.values(),Some(&same),|_|panic!("equal bytes must share the old upload")).unwrap().unwrap();assert!(Rc::ptr_eq(&a,&c));
}
#[test]
fn ended_sessions_release_host_tables_and_cache_clear_rebinds() {
    let mut cache=ParameterCache::<Vec<u32>>::default();let p=prepared(&[1,2,3]);let weak=Rc::downgrade(&p);
    cache.resolve(p.values(),Some(&p),|v|Ok(v.iter().map(|v|v.to_bits()).collect())).unwrap();drop(p);assert!(weak.upgrade().is_none());
    let next=prepared(&[4]);cache.resolve(next.values(),Some(&next),|_|Ok(vec![4])).unwrap();assert_eq!(cache.bindings.len(),1);
    cache.clear();assert_eq!(cache.counts(),ParameterCounts::default());assert!(cache.contents.is_empty());assert!(cache.bindings.is_empty());
    let value=cache.resolve(next.values(),Some(&next),|_|Ok(vec![4])).unwrap().unwrap();assert_eq!(*value,vec![4]);
}
#[test]
fn failed_upload_is_not_bound_and_empty_weights_do_not_build_a_key() {
    let mut cache=ParameterCache::<Vec<u32>>::default();assert!(cache.resolve(&[],None,|_|panic!("empty upload")).unwrap().is_none());
    let p=prepared(&[0]);assert!(cache.resolve(p.values(),Some(&p),|_|Err("upload failed".into())).is_err());assert_eq!(cache.counts(),ParameterCounts::default());
    cache.resolve(p.values(),Some(&p),|_|Ok(vec![0])).unwrap();assert_eq!(cache.counts().static_binds,1);
}
#[test]
fn plan_keeps_declared_weight_order_and_reuses_the_identity() {
    use engine_model::{plan::layer::{LayerPlan,Topology},plan::ConfigPlan};
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(include_bytes!("../../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let plan=LayerPlan::new(&description,0,1,Topology {rank:0,parts:1}).unwrap();let step=plan.steps.iter().find(|s|s.kind==Kind::Conv).unwrap();
    let names=step.weights.iter().map(|w|w.name.clone()).collect::<Vec<_>>();let table=Rc::new(names.iter().enumerate().map(|(i,n)|(n.clone(),vec![f32::from_bits(i as u32+1),-0.0])).collect::<BTreeMap<_,_>>());
    let memo=StaticParameterPlan::new(table.clone(),vec![(0,step.id.clone(),names)]).unwrap();let a=memo.get(0,&table,step).unwrap();let b=memo.get(0,&table,step).unwrap();
    assert!(Rc::ptr_eq(a,b));assert_eq!(a.bits(),&[1,0x80000000,2,0x80000000,3,0x80000000]);
}
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected="static parameter table changed")]
fn debug_check_rejects_another_checkpoint_table() {
    use engine_model::{plan::layer::{LayerPlan,Topology},plan::ConfigPlan};
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(include_bytes!("../../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let plan=LayerPlan::new(&description,0,1,Topology {rank:0,parts:1}).unwrap();let step=plan.steps.iter().find(|s|s.kind==Kind::Conv).unwrap();
    let p=prepared(&[1]);p.check(&Rc::new(BTreeMap::new()),step);
}
use super::super::host::shadow_fixture as fixture;
// Record the real decoder-to-Compute boundary. The cache test above separately
// proves that resolving those weights preserves the original device address.
#[derive(Debug,PartialEq,Eq)]
struct Call {shape:[u32;8],inputs:[Option<Vec<u32>>;4],weights:Vec<u32>}
#[derive(Default)]
struct Recorder {calls:std::cell::RefCell<Vec<Call>>}
impl super::super::Compute for Recorder {
    type Value=Vec<f32>;type Pass=super::super::host::Pass;
    fn upload(&self,v:&[f32])->Result<Self::Value> {super::super::Host.upload(v)}
    fn read(&self,v:&Self::Value)->Result<Vec<f32>> {super::super::Host.read(v)}
    fn words(&self,v:&Self::Value)->usize {v.len()}
    fn range(&self,v:&Self::Value,start:usize,len:usize)->Result<Self::Value> {super::super::Host.range(v,start,len)}
    fn join(&self,v:&[&Self::Value])->Result<Self::Value> {super::super::Host.join(v)}
    fn zeros(&self,n:usize)->Result<Self::Value> {super::super::Host.zeros(n)}
    fn select_known(&self,ids:&Self::Value,count:&Self::Value,row:&[f32])->Result<()> {super::super::Host.select_known(ids,count,row)}
    fn prompt_end(&self)->Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,_:usize)->Result<()> {self.prepare(c)}
    fn operation(&self,s:Shape,x:[Option<&Self::Value>;4],w:&[f32])->Result<Self::Value> {
        self.calls.borrow_mut().push(Call {shape:[s.op as u32,s.heads,s.dim,s.taps,s.rows,s.cols,s.epsilon.to_bits(),s.bound.to_bits()],inputs:x.map(|v|v.map(|v|v.iter().map(|v|v.to_bits()).collect())),weights:w.iter().map(|v|v.to_bits()).collect()});
        super::super::Host.operation(s,x,w)
    }
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Self::Value,w:&[f32])->Result<Self::Value> {super::super::Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value)->Result<(Self::Pass,Self::Value)> {super::super::Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&Self::Value)->Result<Self::Value> {super::super::Host.finish(p,x)}
    fn mx(&self,p:&crate::mxfp8_projection::Mxfp8Projection<'_>,x:&Self::Value)->Result<Self::Value> {super::super::Host.mx(p,x)}
    fn round(&self,x:&Self::Value)->Result<Self::Value> {super::super::Host.round(x)}
    fn mean(&self,x:&Self::Value,streams:u32,hidden:u32)->Result<Self::Value> {super::super::Host.mean(x,streams,hidden)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Self::Value>;5],w:&[f32])->Result<Self::Value> {super::super::Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Self::Value,m:usize,w:&[f32])->Result<Self::Value> {super::super::Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value,m:usize)->Result<(Self::Pass,Self::Value)> {super::super::Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&Self::Value,m:usize)->Result<Self::Value> {super::super::Host.rows_finish(p,x,m)}
}
#[test]
fn decoder_memo_keeps_the_complete_operation_argument_stream() {
    use super::super::Decoder;
    let checkpoint=fixture::Checkpoint::new_with_head(engine_model::safetensors::Dtype::Bf16,true);
    let baseline=Recorder::default();let memo=Recorder::default();
    let mut old=Decoder::new(&checkpoint.admission,&baseline,8).unwrap();
    let new=Decoder::from_prepared(&old.prepared(),&memo,8,0).unwrap();
    // This is the original concatenation path, with the same loaded table.
    old.static_parameters=Rc::new(StaticParameterPlan::default());
    let (mut a,mut b)=(old.zero_state().unwrap(),new.zero_state().unwrap());let mut token=1;
    for _ in 0..3 {
        let (x,id,next_a)=old.step(token,&a,|_,_|Ok(())).unwrap();let (y,jd,next_b)=new.step(token,&b,|_,_|Ok(())).unwrap();
        assert_eq!(id,jd);assert_eq!(x.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),y.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        a=next_a;b=next_b;token=id;
    }
    let a=baseline.calls.borrow();let b=memo.calls.borrow();assert!(!a.is_empty());assert_eq!(*a,*b);
    assert!(a.iter().any(|call|matches!(call.shape[0],x if x==Op::Conv as u32 || x==Op::ConvParts as u32) && !call.weights.is_empty()));
}
