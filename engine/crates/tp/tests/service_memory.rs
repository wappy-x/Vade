use engine_model::{plan::{layer::Topology,state::StatePlan},safetensors::Dtype};
use spark_tp::{bounded_decoder::{expanded_bytes,RESERVED_ROWS},service_memory};
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn session_covers_growing_prefix_and_shared_retained_rows_at_every_phase() {
    let _constructor:fn(Dtype)->fixture::Checkpoint=fixture::Checkpoint::new;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let state=|n|StatePlan::new(c.admission.description(),Topology {rank:0,parts:1},n as u64,isize::MAX as usize).unwrap().state_layout().bytes();
    let vocab=c.admission.description().vocab as usize*16;
    for prompt in [1,64,1152,1173,32768,65544] {
        let tokens=(prompt+64).min(70000);let charge=service_memory::session(&c.admission,prompt,tokens).unwrap();
        for position in [1,64,1152,1173,32768,65544,tokens].into_iter().filter(|&p|p<=tokens) {
            let retained=expanded_bytes(&c.admission,prompt.min(position));
            let copies=if position<=prompt.min(RESERVED_ROWS) {2} else {1};
            assert!(2*state(position)+copies*retained+vocab<=charge,"prompt={prompt} position={position}");
        }
    }
    assert!(service_memory::session(&c.admission,2,1).is_err());
    assert!(service_memory::session(&c.admission,0,0).is_err());
}
#[test]
fn shared_group_admission_funds_two_four_and_eight_sessions_at_32768() {
    let _constructor:fn(Dtype)->fixture::Checkpoint=fixture::Checkpoint::new;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let context=32768;let prompt=context-64;
    let charge=service_memory::session(&c.admission,prompt,context).unwrap();
    let (_,workspace)=spark_tp::row_split::prompt_group_service_charges(&c.admission,context,true,1024).unwrap();
    let resident=123;
    for count in [2,4,8] {
        let budget=resident+workspace+count*charge;
        let mut ledger=spark_tp::group_admission::PromptGroupAdmission::new(&c.admission,context,budget,resident,true,1024).unwrap();
        for id in 1..=count {assert_eq!(spark_tp::row_split::admitted_prompt_group(&mut ledger,id as u64,prompt,context).unwrap(),1024);assert_eq!(ledger.live(),id);assert!(ledger.funded_bytes().unwrap()<=budget);}
        assert_eq!(ledger.funded_bytes().unwrap(),budget);
        assert!(spark_tp::row_split::admitted_prompt_group(&mut ledger,99,prompt,context).is_err());
        assert!(spark_tp::row_split::admitted_prompt_group(&mut ledger,1,prompt,context).is_err());
        assert_eq!(ledger.live(),count);assert_eq!(ledger.funded_bytes().unwrap(),budget);
        ledger.release(1).unwrap();assert!(ledger.release(1).is_err());
        assert_eq!(spark_tp::row_split::admitted_prompt_group(&mut ledger,99,prompt,context).unwrap(),1024);
        let mut single=spark_tp::group_admission::PromptGroupAdmission::new(&c.admission,context,resident+workspace+charge,resident,true,1024).unwrap();
        assert_eq!(spark_tp::row_split::admitted_prompt_group(&mut single,1,prompt,context).unwrap(),1024);
        for id in 2..=count {assert!(spark_tp::row_split::admitted_prompt_group(&mut single,id as u64,prompt,context).is_err());}
        assert_eq!(single.live(),1);
    }
    assert!(spark_tp::group_admission::PromptGroupAdmission::new(&c.admission,context,usize::MAX,usize::MAX,true,1024).unwrap().reserve(1,prompt,context).is_err());
}

#[test]
fn spares_kept_after_a_prompt_group_fit_the_shared_workspace_pool_allowance() {
    use spark_tp::bounded_decoder::{group_workspace_bytes_for,POOL_ROOM,PROJECTION_MEMO_BYTES};
    // Independent literals: the room is 1 GiB; the pool's least cap is 2,000,000,000 B;
    // so at most 926,258,176 B of spares stay held between a prompt group and what follows.
    assert_eq!(POOL_ROOM,1_073_741_824);
    assert_eq!(spark_gpu_probe::spares_after_trim(POOL_ROOM),926_258_176);
    assert!(spark_gpu_probe::spares_after_trim(POOL_ROOM)<=POOL_ROOM);
    // Both workspace charges carry exactly that 1 GiB on top of their scratch peaks, so the
    // kept spares fit beside either phase: prompt groups and decode are charged as
    // alternatives (the larger of the two), and the spares never exceed the allowance.
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    for context in [4096usize,8192,32768] {for group in [64usize,256] {
        let state=StatePlan::new(c.admission.description(),Topology {rank:0,parts:1},context as u64,isize::MAX as usize).unwrap();
        let decode=2*state.scratch_bytes()+PROJECTION_MEMO_BYTES;let prompt=group_workspace_bytes_for(&c.admission,context,group).unwrap();
        assert_eq!(service_memory::workspace(&c.admission,context,0,group).unwrap(),decode.max(prompt)+1_073_741_824,"context={context} group={group}");
        assert!(decode.max(prompt)+spark_gpu_probe::spares_after_trim(POOL_ROOM)<=service_memory::workspace(&c.admission,context,0,group).unwrap());
    }}
}
