//! One real context-wide selected union beside every admitted session owner.
//! Logits are not quality evidence. All allocations use the funded ceiling.
#![forbid(unsafe_code)]
use super::Result;
use spark_gpu_probe::driver::Backend;
use spark_tp::bounded_decoder::{Compute,Decoder,Device,HeadRows,RESERVED_ROWS};
const CONTEXT:usize=32768;
pub fn run(rank:usize,checkpoint:&engine_model::plan::admission::CheckpointAdmission,compute:&Device<'_,'_>,backend:&Backend<'_>,input:&[u32],budget:usize,group:usize,chats:usize)->Result<()> {
    backend.set_allocation_budget(budget)?;compute.clear_projection_memo()?;compute.rows_release()?;
    let decoder=Decoder::new(checkpoint,compute,CONTEXT as u64)?;let resident=backend.accounted_bytes();
    let mut owner=spark_tp::group_admission::PromptGroupAdmission::for_sparse_heads(checkpoint,compute.sparse_topology(),CONTEXT,budget,resident,compute.resident_weights_ready(),group)?;
    let chosen=spark_tp::row_split::admitted_prompt_group(&mut owner,1,CONTEXT-group,CONTEXT)?;
    if chosen!=group {return Err("requested forced group was not admitted".into());}
    for id in 2..=chats {let chosen=spark_tp::row_split::admitted_prompt_group(&mut owner,id as u64,32,96)?;if chosen!=group {return Err("live sessions changed forced group".into());}}
    let charge=owner.funded_bytes()?;backend.set_allocation_budget(charge)?;
    // Inactive chats hold both banks, expanded prefix owners, and the tail
    // backing that a small StepRows group can retain. These are actual buffers.
    let mut others=Vec::new();let mut held=Vec::new();
    for _ in 1..chats {for _ in 0..2 {
        // Hold both inactive banks at the end of their complete output allowance.
        let mut state=decoder.zero_state_at(96)?;
        for (layer,description) in checkpoint.description().layers.iter().enumerate() {
            use engine_model::plan::description::TokenMixer;
            match description.mixer {
                TokenMixer::LatentAttention(a)=>{let layout=spark_tp::sparse_heads::Layout::new(a.heads,a.key_dim,a.value_dim,compute.sparse_topology())?;
                    let words=32*layout.row_words();state.expanded.insert(layer,spark_tp::bounded_decoder::Expanded(std::rc::Rc::new(compute.rows_reserve(words)?),32,layout));},
                TokenMixer::Kda(k)=>held.push(backend.vector(32*(k.heads*k.head_dim) as usize)?),
            }
        }others.push(state);
    }}
    let first=CONTEXT-group;let tokens=input.get(..group).ok_or("forced union needs a complete prompt group")?;
    let mut state=decoder.zero_state_at(first as u64)?;decoder.set_forced_union();let mut kept=0usize;
    for (layer,description) in checkpoint.description().layers.iter().enumerate() {
        use engine_model::plan::description::TokenMixer;
        match description.mixer {
            TokenMixer::LatentAttention(a)=>{let layout=spark_tp::sparse_heads::Layout::new(a.heads,a.key_dim,a.value_dim,compute.sparse_topology())?;
                let words=RESERVED_ROWS*layout.row_words();state.expanded.insert(layer,spark_tp::bounded_decoder::Expanded(std::rc::Rc::new(compute.rows_reserve(words)?),RESERVED_ROWS,layout));kept+=words*4;},
            TokenMixer::Kda(k)=>held.push(backend.vector(256*(k.heads*k.head_dim) as usize)?),
        }
    }
    let copy_before=compute.prompt_state_copies();let before=backend.accounted_bytes();let started=std::time::Instant::now();let whole=compute.rows_peak_begin();
    let (rows,_next)=decoder.prefill_rows(tokens,&state,HeadRows::Last)?;backend.synchronize()?;let peak=compute.rows_peak_end(whole);
    let union=decoder.forced_union().ok_or("forced selection did not run")?;
    if union<CONTEXT || rows.len()!=1 || peak>charge || charge>budget {return Err(format!("forced union extent or funding failed: union={union} peak={peak} charge={charge} budget={budget}"));}
    let copies=compute.prompt_state_copies()-copy_before;let expected=if group>256 {checkpoint.description().layers.iter().filter(|l|matches!(l.mixer,engine_model::plan::description::TokenMixer::Kda(_))).count()} else {0};if copies!=expected {return Err(format!("prefix copy count {copies} differs from {expected}"));}
    println!("GROUP API B FORCED PASS rank={rank} context={CONTEXT} group={group} chats={chats} state_prefix_copies={copies} inactive_banks={} inactive_position=96 union={union} budget={budget} resident={resident} charge={charge} before={before} peak={peak} charge_margin={} ceiling_margin={} kept_rows_bytes={kept} memo_limit={} seconds={:.3}",others.len(),charge-peak,budget-peak,compute.memo_counts().1,started.elapsed().as_secs_f64());
    Ok(())
}
