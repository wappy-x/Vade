//! WIDE PHASE: the memory charge of a prompt group above the whole-charged lease sizes, by the two phases of its wide
//! pass (`Decoder::layer_wide`): one run of `PROMPT_GROUP_MAX` rows in a mixer half, then the routed half of every
//! run. Rule 2 of 2026-11-17. Every other group keeps the whole-group charge (`whole_group_workspace`).
use super::*;
/// WIDE PHASE: the groups that take the phase charge (`wide_phase_workspace`): every wide lease group above
/// `PROMPT_GROUP_MAX` that is a whole number of runs of `PROMPT_GROUP_MAX` rows, whenever every base latent layer is
/// absorbed (the draft predictor never expands during a prompt group, and its single decode row remains charged inside
/// `whole_group_workspace`). Every other group keeps the whole-group charge.
pub(super) fn wide_phase_group(group:usize,rows:LatentRows)->bool {
    group>PROMPT_GROUP_MAX && group.is_multiple_of(PROMPT_GROUP_MAX) && matches!(rows,LatentRows::Absorbed {..})
}
/// Every routed layer's native route metadata epoch for a group of `group` rows, cut in sets of
/// `prompt_route::MAX_ROWS`, as `whole_group_workspace` charges it (E).
fn route_epoch_bytes(c:&CheckpointAdmission,group:usize)->Result<usize> {
    let chunk=spark_gpu_probe::prompt_route::MAX_ROWS as usize;
    let sizes=(0..group).step_by(chunk).map(|first|(group-first).min(chunk)).collect::<Vec<_>>();
    route_epochs_for(c,&sizes)
}
fn route_epochs_for(c:&CheckpointAdmission,sizes:&[usize])->Result<usize> {
    let d=c.description();let mut total=0usize;
    for layer in &d.layers {
        if let engine_model::shapes::Ffn::Experts(e)=layer.ffn {
            let sets=sizes.iter().map(|&rows|spark_gpu_probe::prompt_route::Shape {
                rows:rows as u32,picks:e.selected as u32,experts:e.count as u32,
            }).collect::<Vec<_>>();
            if let Some(bytes)=spark_gpu_probe::prompt_route::epoch_workspace_bytes(&sets) {total=total.checked_add(bytes).ok_or("prompt route epoch charge overflow")?;}
        }
    }
    Ok(total)
}
/// The cohort drains between mixers and after each routed half. Each request's
/// reservation funds its old and unpublished state, including append room.
pub(crate) fn prompt_cohort_fits(c:&CheckpointAdmission,context:usize,funded:usize,latent:LatentRows,sizes:&[usize])->Result<bool> {
    let total=sizes.iter().sum::<usize>();
    if sizes.is_empty() || total>funded {return Ok(false);}
    let (mixer,routed)=prompt_cohort_bounds(c,context,latent,sizes)?;
    Ok(mixer.max(routed)<=whole_group_workspace(c,context,funded,latent)?)
}
/// The cohort's mixer and routed bounds, in bytes. Mixer: the largest over the chats of that chat's mixer half (`member_bound`).
/// Where a whole group's charge grows by at least 4 T a row between the chats' sizes, it is the largest chat's.
fn prompt_cohort_bounds(c:&CheckpointAdmission,context:usize,latent:LatentRows,sizes:&[usize])->Result<(usize,usize)> {
    let total=sizes.iter().sum::<usize>();
    let epochs=route_epochs_for(c,sizes)?;
    let tail=wide_tail_words(c)?.checked_mul(4).ok_or("cohort workspace overflow")?;
    let mut mixer=0usize;
    for &rows in sizes {mixer=mixer.max(member_bound(c,context,latent,rows,total,tail,epochs)?);}
    let routed=total.checked_mul(tail).and_then(|n|n.checked_add(epochs)).ok_or("cohort routed workspace overflow")?;
    Ok((mixer,routed))
}
/// One chat's mixer half charged as a whole group of its `rows`, beside the tail of every other row of the cohort
/// (`total`), and the route epochs of the cohort (`epochs`) that this group's charge does not hold.
fn member_bound(c:&CheckpointAdmission,context:usize,latent:LatentRows,rows:usize,total:usize,tail:usize,epochs:usize)->Result<usize> {
    let single_epoch=route_epoch_bytes(c,rows)?;
    Ok(whole_group_workspace(c,context,rows,latent)?
        .checked_add((total-rows).checked_mul(tail).ok_or("cohort workspace overflow")?)
        .and_then(|n|n.checked_add(epochs.saturating_sub(single_epoch))).ok_or("cohort mixer workspace overflow")?)
}
/// At service start: the whole-group charge at PROMPT_GROUP - 1 and PROMPT_GROUP rows, and 4 T, in bytes. No charge reads it.
pub(crate) fn prompt_cohort_growth(c:&CheckpointAdmission,context:usize,latent:LatentRows)->Result<(usize,usize,usize)> {
    Ok((whole_group_workspace(c,context,PROMPT_GROUP-1,latent)?,whole_group_workspace(c,context,PROMPT_GROUP,latent)?,
        wide_tail_words(c)?.checked_mul(4).ok_or("cohort workspace overflow")?))
}
/// WIDE PHASE: T, the words one row of a wide group can hold while its run is not in a mixer half (EXPC, PR 384).
/// Between two layers and beside another run's mixer half a row holds its residual or its carry: a pending value
/// owns its output and block, and its mHC pass owns source, residual, mixing, coefficients, collapsed, partials and
/// normalized output; at most one earlier pass survives (`begin_after_shadowed` clears older passes before submit),
/// and both full passes are counted (`carry`). In the routed half (`wide_cut_of`: the steps from the cut on) each
/// run's values keep every output of the half to the layer's end, with the operand packs, the route scratch and the
/// exchange margin of the whole-layer charge. The largest over the layers that have a cut, and never less than the
/// carry. Nothing here assumes that a dead value's buffer is used again. No step of that half reads history, so T
/// does not read the context.
fn wide_carry_words(c:&CheckpointAdmission)->Result<usize> {
    let d=c.description();
    let (streams,hidden)=(d.residual.streams as usize,d.hidden as usize);
    let wide=streams.checked_mul(hidden).ok_or("wide carry overflow")?;
    let coeff=streams.checked_mul(streams.checked_add(2).ok_or("wide carry overflow")?).ok_or("wide carry overflow")?;
    wide.checked_mul(7).and_then(|n|n.checked_add(hidden.checked_mul(5)?))
        .and_then(|n|n.checked_add(coeff.checked_mul(2)?)).and_then(|n|n.checked_add(hidden.div_ceil(128).checked_mul(2)?))
        .ok_or_else(||"wide carry overflow".into())
}
fn wide_tail_words(c:&CheckpointAdmission)->Result<usize> {
    let d=c.description();
    let carry=wide_carry_words(c)?;
    let mut per_row=carry;
    for layer in 0..d.layers.len() {
        let plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
        // No cut: the layer runs whole, a run at a time, and every other row holds its residual (the carry).
        let Some((cut,_))=wide_cut_of(&plan) else {continue;};
        let (mut words,mut widest)=(carry,0usize);
        for spec in plan.steps[cut..].iter().flat_map(|step|&step.outputs) {
            let elements=spec.elements().map_err(|e|e.to_string())?;
            words=words.checked_add(elements).ok_or("wide tail overflow")?;widest=widest.max(elements);
        }
        if let engine_model::shapes::Ffn::Experts(e)=d.layers[layer].ffn {
            let packed=(e.selected as usize).checked_mul(((d.hidden as usize/16)*17).max(e.width as usize/16*17)).and_then(|n|n.checked_mul(2)).ok_or("wide tail overflow")?;
            let route=(e.selected as usize*2+e.count as usize*2+d.routing.groups as usize*2).checked_add(e.selected as usize*d.hidden as usize).ok_or("wide tail overflow")?;
            words=words.checked_add(packed).and_then(|n|n.checked_add(route)).ok_or("wide tail overflow")?;
        }
        let exchange=widest.checked_mul(3).map(|n|n.div_ceil(2)).ok_or("wide exchange overflow")?;
        per_row=per_row.max(words.checked_add(exchange).ok_or("wide tail overflow")?);
    }
    Ok(per_row)
}
/// Single-layer state swap bound for `layer_wide`: one layer executes at a time, draining between runs, so at most one
/// layer's next state, one layer's expanded rows, and the append reservation coexist with the session's input state.
fn wide_layer_state_bytes(c:&CheckpointAdmission,context:usize)->Result<usize> {
    let d=c.description();
    let plan=engine_model::plan::state::StatePlan::new(d,Topology {rank:0,parts:1},context as u64,isize::MAX as usize).map_err(|e|e.to_string())?;
    let mut by_layer=vec![0usize;d.layers.len()];
    for plane in plan.planes() {
        by_layer[plane.layer]=by_layer[plane.layer].checked_add(plane.allocation.bytes.len()).ok_or("wide layer state overflow")?;
    }
    let state=by_layer.into_iter().max().unwrap_or(0);
    let expanded=d.layers.iter().map(|l|match l.mixer {
        TokenMixer::LatentAttention(a)=>(a.heads*(a.key_dim+a.value_dim)) as usize*2*context.min(RESERVED_ROWS),
        _=>0,
    }).max().unwrap_or(0);
    let append=crate::admission_plan::append_reservation_bytes(d)?;
    state.checked_add(expanded).and_then(|n|n.checked_add(append)).ok_or_else(||"wide layer state overflow".into())
}
/// Routed-half peak bytes for a wide group of `group` rows: mHC passes are already counted twice in `carry`, routed
/// gate/up (phase 1) and down/sum (phase 2) never coexist, and exchange buffers hold one `PROMPT_GROUP_MAX` run at a time.
fn wide_routed_bytes(c:&CheckpointAdmission,group:usize)->Result<usize> {
    let over=||"wide routed charge overflow".to_string();
    let d=c.description();
    let carry=wide_carry_words(c)?;
    let mut per_row=carry;
    let mut exchange_words=0usize;
    for layer in 0..d.layers.len() {
        let plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
        let Some((cut,_))=wide_cut_of(&plan) else {continue;};
        let (mut base,mut phase1,mut phase2,mut widest)=(carry,0usize,0usize,0usize);
        for step in &plan.steps[cut..] {
            let elements=step.outputs.iter().try_fold(0usize,|acc,spec|{
                let n=spec.elements().map_err(|e|e.to_string())?;widest=widest.max(n);acc.checked_add(n).ok_or_else(over)
            })?;
            match step.kind {
                Kind::MhcPrepare|Kind::Collapse|Kind::Norm|Kind::Combine=>{}
                Kind::ExpertProject=>phase1=phase1.checked_add(elements).ok_or_else(over)?,
                Kind::SwiGlu if step.id.contains("expert")=>{
                    phase1=phase1.checked_add(elements).ok_or_else(over)?;
                    phase2=phase2.checked_add(elements).ok_or_else(over)?;
                }
                Kind::ExpertDown|Kind::ExpertSum=>phase2=phase2.checked_add(elements).ok_or_else(over)?,
                _=>base=base.checked_add(elements).ok_or_else(over)?,
            }
        }
        if let engine_model::shapes::Ffn::Experts(e)=d.layers[layer].ffn {
            let packed_in=(e.selected as usize).checked_mul((d.hidden as usize/16)*17).ok_or_else(over)?;
            let packed_mid=(e.selected as usize).checked_mul((e.width as usize/16)*17).ok_or_else(over)?;
            let route=(e.selected as usize*2+e.count as usize*2+d.routing.groups as usize*2).checked_add(e.selected as usize*d.hidden as usize).ok_or_else(over)?;
            phase1=phase1.checked_add(packed_in).and_then(|n|n.checked_add(route)).ok_or_else(over)?;
            phase2=phase2.checked_add(packed_mid).ok_or_else(over)?;
        }
        per_row=per_row.max(base.checked_add(phase1.max(phase2)).ok_or_else(over)?);
        let ex=widest.checked_mul(3).map(|n|n.div_ceil(2)).ok_or_else(over)?;
        exchange_words=exchange_words.max(ex);
    }
    let rows_bytes=group.checked_mul(per_row).and_then(|n|n.checked_mul(4)).ok_or_else(over)?;
    let ex_bytes=PROMPT_GROUP_MAX.checked_mul(exchange_words).and_then(|n|n.checked_mul(4)).ok_or_else(over)?;
    rows_bytes.checked_add(ex_bytes).ok_or_else(over)
}
/// WIDE PHASE (rule 2 of 2026-11-17, `truth/WIDE4-CHARGE.md`): the charge of a prompt group of `group` rows that
/// passes each layer as runs of R = `PROMPT_GROUP_MAX` rows (`Decoder::layer_wide`):
/// ```text
///     A = whole(R) + 4 (group - R) carry + E(group) - E(R)  the mixer phase
///     B = wide_routed_bytes(group) + E(group)               the routed half
///     X = one layer's state + expanded + append room        single-layer swap bound
///     W = max(A, B) + X
/// ```
/// Strictly monotone in `group` across all wide lease sizes (`2048, 3072, 4096, 8192`).
pub(super) fn wide_phase_workspace(c:&CheckpointAdmission,context:usize,group:usize,rows:LatentRows)->Result<usize> {
    let over=||"wide phase charge overflow".to_string();
    let carry=wide_carry_words(c)?;
    // The rows first: a row count that cannot be charged is refused before its route sets are listed.
    let others=(group-PROMPT_GROUP_MAX).checked_mul(carry).and_then(|n|n.checked_mul(4)).ok_or_else(over)?;
    let (all,one)=(route_epoch_bytes(c,group)?,route_epoch_bytes(c,PROMPT_GROUP_MAX)?);
    let whole=whole_group_workspace(c,context,PROMPT_GROUP_MAX,rows)?;
    // `whole` holds E(R); when E(group) is not above it, nothing is added and the bound still holds.
    let mixer=whole.checked_add(others).and_then(|n|n.checked_add(all.saturating_sub(one))).ok_or_else(over)?;
    let routed=wide_routed_bytes(c,group)?.checked_add(all).ok_or_else(over)?;
    let state=wide_layer_state_bytes(c,context)?;
    mixer.max(routed).checked_add(state).ok_or_else(over)
}
/// Test support: the layers whose program has a wide cut (`wide_cut_of`); a wide pass ends each of them with one more drain.
#[cfg(test)] pub(crate) fn wide_cut_layers(c:&CheckpointAdmission)->usize {
    let d=c.description();
    (0..d.layers.len()).filter(|&layer|LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).is_ok_and(|plan|wide_cut_of(&plan).is_some())).count()
}
#[cfg(test)]
mod tests {
    use super::*;
    const ABSORBED:LatentRows=LatentRows::Absorbed {draft:false};
    fn fixture()->crate::bounded_decoder::TestCheckpoint {crate::bounded_decoder::TestCheckpoint::build_with(engine_model::safetensors::Dtype::Bf16,true,8193,false,|_|{})}
    /// WIDE PHASE: every whole number of runs above PROMPT_GROUP_MAX, when base latent layers are absorbed (with or
    /// without a draft predictor), takes the phase charge. Every other group is charged as a whole group.
    #[test] fn only_absorbed_whole_runs_above_1024_rows_take_the_phase_charge() {
        let c=fixture();let c=&c.admission;
        for group in [2048usize,3072,4096,8192] {
            for rows in [ABSORBED,LatentRows::Absorbed {draft:true}] {assert!(wide_phase_group(group,rows),"{group} {rows:?}");}
        }
        for (group,rows) in [(4096,LatentRows::Expanded),(8192,LatentRows::Expanded),(1024,ABSORBED),(256,ABSORBED),(4095,ABSORBED),(3073,ABSORBED)] {
            assert!(!wide_phase_group(group,rows),"{group} {rows:?}");
            assert_eq!(group_workspace_bytes_with(c,8193,group,rows).unwrap(),whole_group_workspace(c,8193,group,rows).unwrap(),"{group} {rows:?}");
        }
    }
    /// WIDE PHASE: the charge is strictly monotone across all wide lease sizes and holds the larger of the two phase
    /// bounds plus one layer's state swap bound.
    #[test] fn the_phase_charge_is_the_larger_phase_bound_and_one_layer_state() {
        let c=fixture();let c=&c.admission;
        let carry=wide_carry_words(c).unwrap();
        let tail=wide_tail_words(c).unwrap();assert!(tail>=carry,"{tail} {carry}");
        for context in [1usize,256,1025,4096,8193] {
            let whole=whole_group_workspace(c,context,PROMPT_GROUP_MAX,ABSORBED).unwrap();
            let layer_state=wide_layer_state_bytes(c,context).unwrap();
            let one=route_epoch_bytes(c,PROMPT_GROUP_MAX).unwrap();
            let mut prev=whole;
            for group in [2048usize,3072,4096,8192] {
                let all=route_epoch_bytes(c,group).unwrap();assert!(one>0 && all>=one,"{one} {all}");
                let mixer=whole+(group-PROMPT_GROUP_MAX)*carry*4+all.saturating_sub(one);
                let routed=wide_routed_bytes(c,group).unwrap()+all;
                let charge=group_workspace_bytes_with(c,context,group,ABSORBED).unwrap();
                assert_eq!(charge,mixer.max(routed)+layer_state,"context {context} group {group}");
                assert!(charge>prev,"context {context} group {group}: must be strictly monotone");
                prev=charge;
            }
        }
    }
    /// The gated head's mixer bound: the largest chat's whole group beside every other row's tail.
    fn head_mixer(c:&CheckpointAdmission,context:usize,latent:LatentRows,sizes:&[usize])->usize {
        let (total,largest)=(sizes.iter().sum::<usize>(),*sizes.iter().max().unwrap());
        let (epochs,single)=(route_epochs_for(c,sizes).unwrap(),route_epoch_bytes(c,largest).unwrap());
        whole_group_workspace(c,context,largest,latent).unwrap()+(total-largest)*wide_tail_words(c).unwrap()*4+epochs.saturating_sub(single)
    }
    fn every_member(c:&CheckpointAdmission,context:usize,latent:LatentRows,sizes:&[usize])->Vec<usize> {
        let (total,tail,epochs)=(sizes.iter().sum::<usize>(),wide_tail_words(c).unwrap()*4,route_epochs_for(c,sizes).unwrap());
        sizes.iter().map(|&rows|member_bound(c,context,latent,rows,total,tail,epochs).unwrap()).collect()
    }
    /// PROMPT COHORT (a): with one very small and one large chat, the head charged less than the small chat's own bound
    /// on this checkpoint; the charge now covers every member.
    #[test] fn a_cohort_mixer_bound_covers_its_smallest_chat() {
        let c=fixture();let c=&c.admission;let (context,sizes)=(256,[200,1,1]);
        let members=every_member(c,context,LatentRows::Expanded,&sizes);
        assert!(head_mixer(c,context,LatentRows::Expanded,&sizes)<members[1],"{members:?}");
        let (mixer,_)=prompt_cohort_bounds(c,context,LatentRows::Expanded,&sizes).unwrap();
        assert!(members.iter().all(|&bound|mixer>=bound),"{mixer} {members:?}");
    }
    /// PROMPT COHORT (b): over sizes 1 to 256 and 2 to 8 chats the charge is never below the head's, and equals it
    /// whenever the largest chat gives the largest member bound.
    #[test] fn a_cohort_mixer_bound_equals_the_head_where_the_largest_chat_decides() {
        let c=fixture();let c=&c.admission;let mut seed=0x9e37_79b9_7f4a_7c15u64;
        let mut next=|bound:u64|{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;(seed%bound) as usize+1};
        for _ in 0..200 {
            let (context,latent)=([256,1025,8193][next(3)-1],[LatentRows::Expanded,ABSORBED,LatentRows::Absorbed {draft:true}][next(3)-1]);
            let sizes=(0..next(7)+1).map(|_|next(256)).collect::<Vec<_>>();
            let (head,(mixer,_))=(head_mixer(c,context,latent,&sizes),prompt_cohort_bounds(c,context,latent,&sizes).unwrap());
            let largest=sizes.iter().position(|rows|rows==sizes.iter().max().unwrap()).unwrap();
            let members=every_member(c,context,latent,&sizes);
            assert!(mixer>=head && members[largest]==head,"{sizes:?}");
            if members.iter().all(|&bound|bound<=members[largest]) {assert_eq!(mixer,head,"{sizes:?}");}
        }
    }
    /// PROMPT COHORT (c): the growth of a whole group's charge a row beside 4 T on this checkpoint, printed.
    #[test] fn print_the_whole_group_charge_growth_beside_one_tail() {
        let c=fixture();let c=&c.admission;
        for context in [256usize,8193] {for latent in [LatentRows::Expanded,ABSORBED] {
            let (before,at,tail)=prompt_cohort_growth(c,context,latent).unwrap();
            let first=whole_group_workspace(c,context,2,latent).unwrap() as i128-whole_group_workspace(c,context,1,latent).unwrap() as i128;
            println!("PROMPT COHORT GROWTH context={context} {latent:?} 1_to_2={first} 255_to_256={} tail_bytes={tail}",at as i128-before as i128);
        }}
    }
    /// WIDE PHASE: a row count whose charge does not fit a word is an error, before any route set is listed.
    #[test] fn a_phase_charge_that_overflows_is_refused() {
        let c=fixture();let huge=usize::MAX-usize::MAX%PROMPT_GROUP_MAX;
        assert!(wide_phase_group(huge,ABSORBED));
        assert!(group_workspace_bytes_with(&c.admission,8193,huge,ABSORBED).is_err());
    }
}
