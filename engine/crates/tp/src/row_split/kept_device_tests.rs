//! Actual owners after StepRows, cache growth, parking, and scratch draining.
use super::*;
use super::kept_charge_tests::{fixture,Local};
use crate::bounded_decoder::{Device,Host,park_state};
use engine_model::plan::{ConfigPlan,description::TokenMixer,state::{StatePlan,PlaneKind}};
use spark_gpu_probe::{driver::{Backend,with_backend},prompt_ops::{Shape as Rows,Op as Row}};
const CHARGE_INCLUDES_TAIL:bool=true;
fn drain(backend:&Backend<'_>)->Result<()> {backend.release_group_spares()?;backend.synchronize()?;backend.release_group_spares()}
fn pool_parent(backend:&Backend<'_>,factory:&Factory<'_,Host,Local>)->Result<()> {
    drain(backend)?;let base=backend.allocated_bytes();let device=Device::new(backend);let queue=backend.resident_queue()?;
    let shape=Shape {op:Op::Pool,heads:1,dim:128,taps:4,rows:256,cols:0,epsilon:1e-5,bound:0.0};
    let (lengths,weights,_,_)=shape.lengths().ok_or("invalid pool shape")?;
    let x=lengths.map(|words|device.zeros(words)).into_iter().collect::<Result<Vec<_>>>()?;
    let parent=device.operation(shape,[Some(&x[0]),Some(&x[1]),Some(&x[2]),Some(&x[3])],&vec![0.0;weights])?;
    let mut state=State {position:256,layers:vec![BTreeMap::from([
        ("index_keys".into(),device.range(&parent,0,256*128)?),("index_gates".into(),device.range(&parent,256*128,256*128)?),
        ("pool_keys".into(),device.range(&parent,512*128,64*128)?)])],expanded:BTreeMap::new()};
    let first=park_state(&device,&state)?;factory.keep_count(&first);
    state.layers[0].insert("index_keys".into(),device.zeros(257*128)?);state.layers[0].insert("index_gates".into(),device.zeros(257*128)?);state.position=257;
    let second=park_state(&device,&state)?;factory.keep_count(&second);
    drop(state);drop(parent);drop(x);drop(queue);drop(device);drain(backend)?;
    let real=backend.allocated_bytes()-base;assert_eq!((real,factory.kept_prompts().1),(565_248,565_248));
    println!("KEPT POOL DEVICE MEASURED phase=replaced_keys_gates owners=3 real_bytes={real} rank_charge_bytes={} rounding_limit_bytes=12288",factory.kept_prompts().1);
    factory.release_count(&first);drop(first);drain(backend)?;assert_eq!(backend.allocated_bytes()-base,565_248);assert_eq!(factory.kept_prompts().1,565_248);
    factory.release_count(&second);drop(second);drain(backend)?;assert_eq!(backend.allocated_bytes(),base);assert_eq!(factory.kept_prompts(),(0,0));Ok(())
}
fn build_state<'b,'d>(device:&Device<'b,'d>,d:&engine_model::plan::TextDescription,plan:&StatePlan,topology:Topology,tokens:usize,group:usize,append:bool)->Result<(State<crate::bounded_decoder::Value<'b,'d>>,usize)> {
                let mut state=State {position:tokens as u64,layers:(0..d.layers.len()).map(|_|BTreeMap::new()).collect(),expanded:BTreeMap::new()};
                let mut copies=0usize;
                for plane in plan.planes() {
                    let mut bytes=plane.used_bytes(tokens as u64).map_err(|e|e.to_string())?;
                    if matches!(plane.kind,PlaneKind::KdaState|PlaneKind::KdaHistory) {
                        let layout=crate::kda_heads::layout(d.layers[plane.layer].mixer,topology)?;bytes=bytes/layout.total_heads()*layout.local_heads();
                    }
                    let value=if plane.kind==PlaneKind::KdaState {
                        let TokenMixer::Kda(k)=d.layers[plane.layer].mixer else {return Err("expected recurrent plane".into());};
                        let heads=crate::kda_heads::layout(d.layers[plane.layer].mixer,topology)?.local_heads();
                        let mut previous=device.zeros(bytes/4)?;let mut at=0;
                        while at<tokens {
                            let rows=group.min(tokens-at);let mut shape=Rows::new(Row::StepRows,rows).unwrap();shape.heads=heads as u32;shape.dim=k.head_dim as u32;
                            let (lengths,_,_,_)=shape.lengths().ok_or("invalid recurrent shape")?;
                            let input=device.zeros(lengths[0])?;let decay=device.zeros(lengths[2])?;
                            let columns=device.prompt(shape,[Some(&input),Some(&previous),Some(&decay),None,None],&[])?;
                            previous=device.range(&columns,0,bytes/4)?;copies+=usize::from(rows>256);at+=rows;
                        }previous
                    } else if matches!(plane.kind,PlaneKind::Latent|PlaneKind::IndexKeys|PlaneKind::IndexGates|PlaneKind::Pools) {
                        let mut previous=device.zeros(0)?;let mut at=0;
                        while at<tokens {
                            let next=(at+group).min(tokens);let have=plane.used_bytes(at as u64).map_err(|e|e.to_string())?/4;
                            let needed=plane.used_bytes(next as u64).map_err(|e|e.to_string())?/4;
                            let new=device.zeros(needed-have)?;previous=device.join(&[&previous,&new])?;at=next;
                        }
                        if append {let new=device.zeros(1)?;previous=device.rows_extend(Some(&previous),bytes/4,&new,1,1024)?;}previous
                    } else {device.zeros(bytes/4)?};
                    if plane.kind==PlaneKind::KdaState && plane.layer==0 {
                        let alias=device.range(&value,0,device.words(&value))?;state.layers[plane.layer].insert("owner_alias".into(),alias);
                    }
                    state.layers[plane.layer].insert(plane.allocation.name.clone(),value);
                }
    Ok((state,copies))
}
#[test]
#[ignore="requires an authorized CUDA device; allocation proof, not served token IDs"]
fn parked_device_allocations_cover_final_group_owners() {
    with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),8usize<<30,|backend| {
        let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
        let plan=StatePlan::new(&d,Topology {rank:0,parts:1},16384,isize::MAX as usize).map_err(|e|e.to_string())?;
        let mut layered=d.with_draft_layer().map_err(|e|e.to_string())?.unwrap();layered.residual=d.residual;
        struct Scratch(std::path::PathBuf);impl Drop for Scratch {fn drop(&mut self) {std::fs::remove_dir_all(&self.0).unwrap();}}
        let root=Scratch(std::env::temp_dir().join(format!("vade-kept-device-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())));
        let mut small=fixture::description();small.max_positions=16384;
        let index=fixture::checkpoint(&root.0,&small)?;let checkpoint=small.open_headers(&root.0,&index).map_err(|e|e.to_string())?;
        let mut factory=Factory::new(&checkpoint,Split::new(Host,Local)?,16384,vec![0])?;
        factory.with_layer=Some(crate::admission_plan::StateChargePlan::new(&layered,16384)?);
        backend.configure_vector_pool(0);
        for rank in 0..2 {
            let topology=Topology {rank,parts:2};
            factory.charge_plan=crate::admission_plan::SessionChargePlan::for_sparse_heads(&d,topology,16384)?;
            factory.kept_tail=super::kept_charge::Tail::new(&d,topology)?;
            for (kind,tokens,group,append) in [("final",1,1,false),("final",153,153,false),("final",256,256,false),("final",257,257,false),("final",512,512,false),("final",1024,1024,false),
                ("whole",1040,1024,false),("whole",4158,1024,false),("whole",8192,1024,false),("append",1040,1024,true)] {
                let device=Device::new(backend);drain(backend)?;let base=backend.allocated_bytes();
                let queue=backend.resident_queue()?;let (state,copies)=build_state(&device,&d,&plan,topology,tokens,group,append)?;
                let parked=park_state(&device,&state)?;drop(state);drop(queue);drain(backend)?;
                let observed_copies=device.prompt_state_copies();let with_compute=backend.allocated_bytes().checked_sub(base).ok_or("allocation baseline changed")?;
                // Whole-prompt joins keep a descriptor table in Device::copies.
                // It belongs to execution scratch, not to the parked state.
                drop(device);drain(backend)?;
                let real=backend.allocated_bytes().checked_sub(base).ok_or("allocation baseline changed")?;
                let charged=factory.parked_charge(&parked);
                let owners=plan.planes().len();let rounding_limit=owners*4096;let excess=charged as i128-real as i128;
                println!("KEPT DEVICE MEASURED exact={CHARGE_INCLUDES_TAIL} rank={rank} kind={kind} tokens={tokens} group={group} owners={owners} real_bytes={real} retained_charge_bytes={charged} excess_bytes={excess} rounding_limit_bytes={rounding_limit} compute_scratch_bytes={} prefix_copies={observed_copies} safety_bound={} full_tight_bound={}",with_compute-real,charged>=real,excess>=0 && excess<=rounding_limit as i128);
                assert_eq!(observed_copies,copies);
                if CHARGE_INCLUDES_TAIL {assert_eq!(charged,real,"tokens={tokens}: pooled owner charge must be exact");}
                let reader=Device::new(backend);
                let second=park_state(&reader,&parked)?;assert_eq!(factory.parked_charge(&second),charged);drop(second);
                for planes in &parked.layers {for value in planes.values() {assert!(reader.read(value)?.iter().all(|&v|v==0.0));}}
                drop(parked);drop(reader);drain(backend)?;assert_eq!(backend.allocated_bytes(),base);
            }
            // A second holder, partial shared prefix, unrelated roots, and every
            // cache removal use the same rank ledger as the real adapter.
            let mut make=|| {
                let builder=Device::new(backend);let queue=backend.resident_queue()?;
                let (state,_)=build_state(&builder,&d,&plan,topology,256,256,false)?;
                let parked=park_state(&builder,&state)?;drop(state);drop(queue);drop(builder);drain(backend)?;Ok(parked)
            };
            drain(backend)?;let base=backend.allocated_bytes();let first=make()?;let reader=Device::new(backend);
            super::kept_device_cache::exercise(backend,&factory,&reader,base,first,&mut make,|old| {
                let queue=backend.resident_queue()?;let mut state=crate::bounded_decoder::share_state(&reader,old)?;
                let plane=plan.planes().iter().find(|p|p.kind==PlaneKind::Latent).ok_or("missing latent plane")?;
                let TokenMixer::LatentAttention(a)=d.layers[plane.layer].mixer else {return Err("missing latent layout".into());};
                let width=a.kv_rank as usize;let value=&state.layers[plane.layer][&plane.allocation.name];let new=reader.zeros(width)?;
                let next=reader.cache_append(value,&new,256,width,128*width)?;
                state.layers[plane.layer].insert(plane.allocation.name.clone(),next);state.position+=1;
                let parked=park_state(&reader,&state)?;drop(state);drop(queue);drain(backend)?;Ok(parked)
            })?;drop(reader);drain(backend)?;assert_eq!(backend.allocated_bytes(),base);pool_parent(backend,&factory)?;
        }Ok(())
    }).unwrap();
}
