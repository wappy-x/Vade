//! WIDE PHASE: the memory charge of a prompt group above the whole-charged lease sizes, by the two phases of its wide
//! pass (`Decoder::layer_wide`): one run of `PROMPT_GROUP_MAX` rows in a mixer half, then the routed half of every
//! run. Rule 2 of 2026-11-17. Every other group keeps the whole-group charge (`whole_group_workspace`).
use super::*;
/// WIDE PHASE: the groups that take the phase charge (`wide_phase_workspace`): above the largest lease size that is
/// charged as a whole group, a whole number of runs of `PROMPT_GROUP_MAX` rows, every latent layer absorbed and no
/// predictor that expands. Every other group keeps the whole-group charge.
pub(super) fn wide_phase_group(group:usize,rows:LatentRows)->bool {
    group>PROMPT_GROUP_LEASE[1] && group.is_multiple_of(PROMPT_GROUP_MAX) && matches!(rows,LatentRows::Absorbed {draft:false})
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
fn wide_tail_words(c:&CheckpointAdmission)->Result<usize> {
    let d=c.description();
    let (streams,hidden)=(d.residual.streams as usize,d.hidden as usize);
    let wide=streams.checked_mul(hidden).ok_or("wide carry overflow")?;
    let coeff=streams.checked_mul(streams.checked_add(2).ok_or("wide carry overflow")?).ok_or("wide carry overflow")?;
    let carry=wide.checked_mul(7).and_then(|n|n.checked_add(hidden.checked_mul(5)?))
        .and_then(|n|n.checked_add(coeff.checked_mul(2)?)).and_then(|n|n.checked_add(hidden.div_ceil(128).checked_mul(2)?)).ok_or("wide carry overflow")?;
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
/// WIDE PHASE (rule 2 of 2026-11-17, `truth/WIDE4-CHARGE.md`): the charge of a prompt group of `group` rows that
/// passes each layer as runs of R = `PROMPT_GROUP_MAX` rows (`Decoder::layer_wide`):
/// ```text
///     A = whole(R) + 4 (group - R) T + E(group) - E(R)      the mixer phase
///     B = 4 group T + E(group)                              the routed half
///     X = one full session at this context (`service_memory::session`)
///     W = max(A, B) + X
/// ```
/// The mixer phase: ONE run is in a mixer half, charged as a whole group of R rows (`whole`: `whole_group_workspace`,
/// the charge every group pays); every other row holds its residual or its carry (at most T, `wide_tail_words`);
/// the route metadata of all sets of all routed layers is E(group), of which `whole(R)` holds E(R). The routed
/// half: every row holds at most T, and the route metadata E(group). X covers what a wide pass holds and a whole pass
/// does not: the state the run before made, which lives while the next run makes its own beside the group's input
/// state, and held rows a run made. That is one state of ONE layer with its append room and its held rows; X is at
/// least two banks of EVERY layer's state, every layer's held rows and append room (`session_with_expanded`).
/// W is a maximum and a sum of these bounds, so it covers both phases for every checkpoint and context; no measured
/// peak and no number of a model is in it. It rests on two things the pass does (`layer_wide`): a run's mixer half
/// hands on only its carry, the next state and held rows, and the stream is drained (`Compute::rows_phase_end`) after
/// each mixer run, once the state before it is dropped, and at each layer's end, so no buffer dropped before a run
/// is still charged while it works. Spares stay in the pool under its caps and the group ceiling, as in every group.
pub(super) fn wide_phase_workspace(c:&CheckpointAdmission,context:usize,group:usize)->Result<usize> {
    let over=||"wide phase charge overflow".to_string();
    let tail=wide_tail_words(c)?;
    let held=|rows:usize|rows.checked_mul(tail).and_then(|n|n.checked_mul(4)).ok_or_else(over);
    // The rows first: a row count that cannot be charged is refused before its route sets are listed.
    let (others,every)=(held(group-PROMPT_GROUP_MAX)?,held(group)?);
    let (all,one)=(route_epoch_bytes(c,group)?,route_epoch_bytes(c,PROMPT_GROUP_MAX)?);
    let whole=whole_group_workspace(c,context,PROMPT_GROUP_MAX,LatentRows::Absorbed {draft:false})?;
    // `whole` holds E(R); when E(group) is not above it, nothing is added and the bound still holds.
    let mixer=whole.checked_add(others).and_then(|n|n.checked_add(all.saturating_sub(one))).ok_or_else(over)?;
    let routed=every.checked_add(all).ok_or_else(over)?;
    let state=crate::service_memory::session(c,context,context)?;
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
    /// WIDE PHASE: only a whole number of runs above 3,072 rows, with every latent layer absorbed and no predictor that
    /// expands, takes the phase charge. Every other group is charged as a whole group, as before.
    #[test] fn only_absorbed_whole_runs_above_3072_rows_take_the_phase_charge() {
        let c=fixture();let c=&c.admission;
        assert!(wide_phase_group(4096,ABSORBED));
        for (group,rows) in [(4096,LatentRows::Expanded),(4096,LatentRows::Absorbed {draft:true}),(3072,ABSORBED),(2048,ABSORBED),(1024,ABSORBED),(256,ABSORBED),(4095,ABSORBED),(3073,ABSORBED)] {
            assert!(!wide_phase_group(group,rows),"{group} {rows:?}");
            assert_eq!(group_workspace_bytes_with(c,8193,group,rows).unwrap(),whole_group_workspace(c,8193,group,rows).unwrap(),"{group} {rows:?}");
        }
    }
    /// WIDE PHASE (rule 2): the charge is the larger of the two phase bounds plus one full session, at every context, and
    /// that session holds one more bank of every layer's state with its held rows and append room. T is at least the
    /// carry, and the route metadata of all sets is in both bounds.
    #[test] fn the_phase_charge_is_the_larger_phase_bound_and_one_full_session() {
        let c=fixture();let c=&c.admission;let d=c.description();
        let tail=wide_tail_words(c).unwrap();
        let (streams,hidden)=(d.residual.streams as usize,d.hidden as usize);
        let carry=streams*hidden*7+hidden*5+streams*(streams+2)*2+hidden.div_ceil(128)*2;assert!(tail>=carry,"{tail} {carry}");
        let (all,one)=(route_epoch_bytes(c,4096).unwrap(),route_epoch_bytes(c,PROMPT_GROUP_MAX).unwrap());assert!(one>0 && all>one,"{one} {all}");
        for context in [1usize,256,1025,4096,8193] {
            let whole=whole_group_workspace(c,context,PROMPT_GROUP_MAX,ABSORBED).unwrap();
            let session=crate::service_memory::session(c,context,context).unwrap();
            let (mixer,routed)=(whole+(4096-PROMPT_GROUP_MAX)*tail*4+all-one,4096*tail*4+all);
            let charge=group_workspace_bytes_with(c,context,4096,ABSORBED).unwrap();
            assert_eq!(charge,mixer.max(routed)+session,"context {context}");
            let state=engine_model::plan::state::StatePlan::new(d,Topology {rank:0,parts:1},context as u64,isize::MAX as usize).unwrap().state_layout().bytes();
            let extra=state+expanded_bytes(c,context)+crate::admission_plan::append_reservation_bytes(d).unwrap();
            assert!(session>=state+extra,"context {context}: a session is two banks, the held rows and the append room");
            assert!(charge>=mixer+extra && charge>=routed+extra,"context {context}");
            // The mixer phase alone: one whole run, and the carry of every other row.
            assert!(mixer>=whole+(4096-PROMPT_GROUP_MAX)*carry*4);
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
