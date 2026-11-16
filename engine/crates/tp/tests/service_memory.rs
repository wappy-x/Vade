use engine_model::{plan::{layer::Topology,state::StatePlan},safetensors::Dtype};
use spark_tp::{bounded_decoder::{expanded_bytes,RESERVED_ROWS},service_memory};
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn sparse_service_and_group_admission_use_the_same_local_charge() {
    let checkpoint=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,8192);
    let topology=Topology {rank:1,parts:2};let context=8192;let prompt=4158;
    let local=service_memory::session_for_sparse_heads(&checkpoint.admission,topology,prompt,context).unwrap();
    let full=service_memory::session(&checkpoint.admission,prompt,context).unwrap();
    assert!(local<full);
    let plan=spark_tp::admission_plan::SessionChargePlan::for_sparse_heads(checkpoint.admission.description(),topology,context).unwrap();
    assert_eq!(plan.session(prompt,context).unwrap(),local);
    let (_,workspace)=spark_tp::row_split::prompt_group_sparse_charges(&checkpoint.admission,topology,context,true,256).unwrap();
    let (_,full_workspace)=spark_tp::row_split::prompt_group_service_charges(&checkpoint.admission,context,true,256).unwrap();
    assert_eq!(workspace,full_workspace);
    let mut admission=spark_tp::group_admission::PromptGroupAdmission::for_sparse_heads(&checkpoint.admission,topology,context,local+workspace,0,true,256).unwrap();
    assert_eq!(admission.reserve(1,prompt,context).unwrap(),256);
    assert_eq!(admission.funded_bytes().unwrap(),local+workspace);
    assert!(admission.reserve(2,prompt,context).is_err());
}
/// The served charge is the sum: the device keeps its memo while a session reads its prompt (MEMO_KEEP, the seed of
/// the last prompt group, and the rows of other open sessions), so memo and prompt scratch are funded together.
/// The phase rule of `service_memory::workspace` funds the larger of the two only; it must stay the smaller charge.
#[test]
fn the_served_workspace_charge_funds_the_whole_memo_beside_a_prompt_group() {
    use spark_tp::bounded_decoder::{group_workspace_bytes,MEMO_KEEP,PROJECTION_MEMO_BYTES};
    for context in [4096usize,8192] {
        let checkpoint=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,8192);let topology=Topology {rank:1,parts:2};
        let (_,served)=spark_tp::row_split::sparse_service_charges(&checkpoint.admission,topology,context,true).unwrap();
        let prompt=group_workspace_bytes(&checkpoint.admission,context).unwrap();
        let scratch=StatePlan::new(checkpoint.admission.description(),Topology {rank:0,parts:1},context as u64,isize::MAX as usize).unwrap().scratch_bytes();
        assert_eq!(served,2*scratch+prompt+PROJECTION_MEMO_BYTES+(1<<30));
        const {assert!(MEMO_KEEP<=PROJECTION_MEMO_BYTES)};
        let phase=service_memory::workspace(&checkpoint.admission,context,0,256).unwrap();
        assert_eq!(phase,(2*scratch+PROJECTION_MEMO_BYTES).max(prompt)+(1<<30));
        assert!(served>=phase+prompt.min(2*scratch+PROJECTION_MEMO_BYTES));
    }
}
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

#[test]
fn prompt_workspace_charge_covers_the_five_planes_of_absorbed_attention() {
    // An absorbed prompt group holds five planes at once: the absorbed query and the mixed latent (latent width),
    // scores and probabilities (one word a selected row), and the projected output (value width).
    use engine_model::plan::description::TokenMixer;
    use spark_tp::bounded_decoder::{absorbed_group_words,absorbed_chat_words,group_workspace_bytes_for,BATCH_ROWS};
    // The model's shapes: a 256-token group, latent 512, value 256, 2,051 selected rows. 32 heads is one rank of two
    // Sparks, 64 heads is every head. The two latent planes that were charged before are a fifth of the second number.
    assert_eq!(absorbed_group_words(256,32,512,256,2051),Some(44_089_344));
    assert_eq!(absorbed_group_words(256,64,512,256,2051),Some(88_178_688));
    assert_eq!(absorbed_group_words(256,64,512,0,0),Some(16_777_216),"the two latent planes alone");
    assert_eq!(absorbed_group_words(usize::MAX,2,1,1,1),None);
    assert_eq!(absorbed_chat_words(usize::MAX,2,1,1,1,1),None);
    assert_eq!(absorbed_chat_words(8,1,1,1,1,usize::MAX),None);
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,8192);let mut seen=0;
    for layer in &c.admission.description().layers {
        let TokenMixer::LatentAttention(a)=layer.mixer else {continue;};seen+=1;
        let slots=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}) as usize;
        for group in [1usize,16,256,512,1024] {for context in [1usize,15,127,slots/2,slots,slots+1,2*slots,8192] {
            let live=slots.min(context);
            let planes=4*absorbed_group_words(group,a.heads as usize,a.kv_rank as usize,a.value_dim as usize,live).unwrap();
            let charge=group_workspace_bytes_for(&c.admission,context,group).unwrap();
            assert!(charge>=planes,"group {group} context {context}: charge {charge} is under the five planes {planes}");
            let chats=4*absorbed_chat_words(BATCH_ROWS,a.heads as usize,a.kv_rank as usize,a.key_dim as usize,a.value_dim as usize,live).unwrap();
            assert!(charge>=chats,"group {group} context {context}: charge {charge} is under the chat owners {chats}");
        }}
    }
    assert!(seen>0,"the fixture has no latent attention layer");
}

#[test]
fn the_prompt_group_is_the_largest_one_the_budget_funds_and_256_rows_change_no_charge() {
    use spark_tp::bounded_decoder::{group_workspace_bytes,group_workspace_bytes_for};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    for context in [4096usize,8192,32768] {
        let (session,workspace)=spark_tp::row_split::service_charges(&c.admission,context,true).unwrap();
        // 256 rows: the charge every service had before.
        assert_eq!(service_memory::workspace_for_prompt_group(&c.admission,context,workspace,256).unwrap(),workspace);
        let copies=service_memory::state_copy_bytes(&c.admission).unwrap();let small=group_workspace_bytes(&c.admission,context).unwrap();
        let mut charges=Vec::new();
        for group in [512usize,1024] {
            let charge=service_memory::workspace_for_prompt_group(&c.admission,context,workspace,group).unwrap();
            // Exactly the larger group's buffers in place of the 256-row ones, and the state copies.
            assert_eq!(charge,workspace-small+group_workspace_bytes_for(&c.admission,context,group).unwrap()+copies,"context={context} group={group}");
            assert!(charge>workspace);charges.push(charge);
        }
        assert!(charges[1]>charges[0]);
        let resident=1usize<<30;let pick=|budget:usize|service_memory::largest_prompt_group(&c.admission,context,workspace,resident,session,budget).unwrap();
        // One byte short of a group's charge funds the next smaller group. Nothing below 256.
        assert_eq!(pick(resident+charges[1]+session),1024);assert_eq!(pick(resident+charges[1]+session-1),512);
        assert_eq!(pick(resident+charges[0]+session),512);assert_eq!(pick(resident+charges[0]+session-1),256);
        assert_eq!(pick(0),256);assert_eq!(pick(usize::MAX),1024);
    }
}

#[test]
fn absorbed_latent_layers_are_not_charged_the_expanded_rows_of_the_whole_context() {
    use engine_model::plan::description::TokenMixer;
    use spark_tp::bounded_decoder::{absorbed_group_words,absorbed_chat_words,expanded_row_words,group_workspace_bytes_for,group_workspace_bytes_with,LatentRows,BATCH_ROWS};
    // The served model's shapes, as independent literals: 64 heads x (256 key + 256 value) = 32,768 words an expanded row,
    // latent 512, 2,051 selected rows. One decode row of an unabsorbed layer: 2,051 x (3 x 32,768 + 512) + 256 x 32,768 words
    // (844 MB). The rows of one layer for a 131,072 context that the expanded form charges: (131,072 + 256) x 16,896 + 256 x 32,768
    // words (8.9 GB).
    assert_eq!(expanded_row_words(32_768,512,2051),Some(211_060_224));
    assert_eq!((131_072usize+256)*(32_768/2+512)+256*32_768,2_227_306_496);
    assert_eq!(expanded_row_words(usize::MAX,1,1),None);assert_eq!(expanded_row_words(1,1,usize::MAX),None);
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let a=c.admission.description().layers.iter().find_map(|l|match l.mixer {TokenMixer::LatentAttention(a)=>Some(a),_=>None}).expect("the fixture has a latent layer");
    let slots=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}) as usize;
    let (row,rank)=((a.heads*(a.key_dim+a.value_dim)) as usize,a.kv_rank as usize);
    let (plain,draft)=(LatentRows::Absorbed {draft:false},LatentRows::Absorbed {draft:true});let mut lower=0;
    for group in [1usize,64,256,512,1024] {for context in [1usize,slots,4096,8192,32768,65536] {
        let expanded=group_workspace_bytes_for(&c.admission,context,group).unwrap();
        // The name every caller used before: the same charge.
        assert_eq!(group_workspace_bytes_with(&c.admission,context,group,LatentRows::Expanded).unwrap(),expanded,"group {group} context {context}");
        let (absorbed,with)=(group_workspace_bytes_with(&c.admission,context,group,plain).unwrap(),group_workspace_bytes_with(&c.admission,context,group,draft).unwrap());
        // Never above what the server was charged before, and an unabsorbed predictor layer never below a server without one.
        assert!(absorbed<=with && with<=expanded,"group {group} context {context}: {absorbed} {with} {expanded}");
        if with<expanded {lower+=1;}
        // An absorbed group still holds its five planes, and a chat step its owners.
        let live=slots.min(context);
        assert!(absorbed>=4*absorbed_group_words(group,a.heads as usize,rank,a.value_dim as usize,live).unwrap(),"group {group} context {context}");
        assert!(absorbed>=4*absorbed_chat_words(BATCH_ROWS,a.heads as usize,rank,a.key_dim as usize,a.value_dim as usize,live).unwrap(),"group {group} context {context}");
        // The predictor's row is charged in full wherever the old charge held that much, and the old charge otherwise.
        let union=(context.min(group*live)+group)*(row/2+rank)+group*row;
        assert!(with>=4*union.min(expanded_row_words(row,rank,live).unwrap()),"group {group} context {context}");
    }}
    assert!(lower>0,"no context of the fixture is charged less: the test proves nothing");
    // Past the rows a token can select, the absorbed charge of a 256-row group no longer follows the context (the Scores
    // map stops at 16,384 rows). The expanded one follows the largest of its three alternatives: where that is the union
    // of selected rows, one packed row and its latent for every context token.
    let largest=|context:usize| {let live=slots.min(context);((context.min(256*live)+256)*(row/2+rank)+256*row)
        .max(absorbed_group_words(256,a.heads as usize,rank,a.value_dim as usize,live).unwrap()).max(absorbed_chat_words(BATCH_ROWS,a.heads as usize,rank,a.key_dim as usize,a.value_dim as usize,live).unwrap())};
    for (near,far) in [(32768usize,65536usize),(16384,65536)] {
        assert_eq!(group_workspace_bytes_with(&c.admission,near,256,plain).unwrap(),group_workspace_bytes_with(&c.admission,far,256,plain).unwrap());
        assert_eq!(group_workspace_bytes_for(&c.admission,far,256).unwrap()-group_workspace_bytes_for(&c.admission,near,256).unwrap(),4*(largest(far)-largest(near)));
    }
    // The service's workspace: only the group's own buffers change, and the larger groups follow the same form.
    for context in [8192usize,32768,65536] {
        let (_,workspace)=spark_tp::row_split::service_charges(&c.admission,context,true).unwrap();
        assert_eq!(service_memory::workspace_for_latent_rows(&c.admission,context,workspace,LatentRows::Expanded).unwrap(),workspace);
        let small=service_memory::workspace_for_latent_rows(&c.admission,context,workspace,plain).unwrap();
        assert_eq!(workspace-small,group_workspace_bytes_for(&c.admission,context,256).unwrap()-group_workspace_bytes_with(&c.admission,context,256,plain).unwrap());
        assert_eq!(service_memory::workspace_for_prompt_group_with(&c.admission,context,small,256,plain).unwrap(),small);
        let copies=service_memory::state_copy_bytes(&c.admission).unwrap();
        for group in [512usize,1024] {
            let charge=service_memory::workspace_for_prompt_group_with(&c.admission,context,small,group,plain).unwrap();
            assert_eq!(charge,small-group_workspace_bytes_with(&c.admission,context,256,plain).unwrap()+group_workspace_bytes_with(&c.admission,context,group,plain).unwrap()+copies);
            assert!(charge<=service_memory::workspace_for_prompt_group(&c.admission,context,workspace,group).unwrap());
            let (resident,session)=(1usize<<30,1usize<<28);
            let pick=|budget:usize|service_memory::largest_prompt_group_with(&c.admission,context,small,resident,session,budget,plain).unwrap();
            assert!(pick(resident+charge+session)>=group);assert!(pick(resident+charge+session-1)<group);
        }
    }
}

