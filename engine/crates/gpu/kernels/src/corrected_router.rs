//! Corrected sigmoid selection with one fixed serial fold per token.
use crate::router_abi::{CorrectedSigmoidRoute, MAX_EXPERTS, MAX_GROUPS, ROUTER_SHARED_BYTES};

#[no_mangle]
pub extern "ptx-kernel" fn corrected_sigmoid_route(p: CorrectedSigmoidRoute) {
    // SAFETY: The caller checks p with CorrectedSigmoidRoute::checked and owns
    // its disjoint device spans through completion. The grid has one block per
    // token and ROUTER_THREADS threads per block. Each lane writes distinct
    // shared cells. The barrier precedes the single-thread selection and fold.
    // Config limits bound all shared offsets and all global row accesses.
    unsafe {
        let token = core::arch::nvptx::_block_idx_x() as usize;
        let lane = core::arch::nvptx::_thread_idx_x() as usize;
        if token >= p.tokens as usize { return; }
        let c = p.config;
        let (experts,selected,groups,keep) = (c.experts as usize,c.selected as usize,c.groups as usize,c.selected_groups as usize);
        let logits = (p.logits as *const f32).add(token * experts);
        let bias = p.bias as *const f32;
        let ids = (p.ids as *mut u32).add(token * selected);
        let weights = (p.weights as *mut f32).add(token * selected);
        let status = (p.status as *mut u32).add(token);
        let scores = crate::ptx::shared::<ROUTER_SHARED_BYTES>();
        let choice = scores + MAX_EXPERTS * 4;
        let group_scores = choice + MAX_EXPERTS * 4;
        let kept = group_scores + MAX_GROUPS * 4;
        for expert in (lane..experts).step_by(crate::router_abi::ROUTER_THREADS as usize) {
            let (logit,b) = (*logits.add(expert),*bias.add(expert));
            let score = 1.0 / (1.0 + crate::flash_moe::exp(-logit));
            let corrected = if logit.is_finite() && b.is_finite() { score + b } else { f32::NAN };
            crate::ptx::store(scores + (expert * 4) as u32,score);
            crate::ptx::store(choice + (expert * 4) as u32,corrected);
        }
        crate::ptx::sync();
        if lane != 0 { return; }
        let width = experts / groups;
        let mut invalid = *status != 0;
        for expert in 0..experts { invalid |= !crate::ptx::load(choice + (expert * 4) as u32).is_finite(); }
        if !invalid {
            for group in 0..groups {
                let (mut first,mut second) = (f32::NEG_INFINITY,f32::NEG_INFINITY);
                for expert in group*width..(group+1)*width {
                    let v = crate::ptx::load(choice + (expert * 4) as u32);
                    // Ascending IDs and strict comparisons retain lower-ID ties.
                    if v > first { second = first; first = v; } else if v > second { second = v; }
                }
                let sum = first + second;
                invalid |= !sum.is_finite();
                crate::ptx::store(group_scores + (group * 4) as u32,sum);
                crate::ptx::store(kept + (group * 4) as u32,0.0);
            }
        }
        if !invalid {
            for _ in 0..keep {
                let (mut best,mut id) = (f32::NEG_INFINITY,0usize);
                for group in 0..groups {
                    let v = crate::ptx::load(group_scores + (group * 4) as u32);
                    if v > best { best = v; id = group; }
                }
                crate::ptx::store(kept + (id * 4) as u32,1.0);
                crate::ptx::store(group_scores + (id * 4) as u32,f32::NEG_INFINITY);
            }
            let mut sum = 0.0f32;
            for slot in 0..selected {
                let (mut best,mut id) = (f32::NEG_INFINITY,0usize);
                for expert in 0..experts {
                    if crate::ptx::load(kept + (expert / width * 4) as u32) == 0.0 { continue; }
                    let v = crate::ptx::load(choice + (expert * 4) as u32);
                    if v > best { best = v; id = expert; }
                }
                let weight = crate::ptx::load(scores + (id * 4) as u32);
                *ids.add(slot) = id as u32;
                *weights.add(slot) = weight;
                sum += weight;
                crate::ptx::store(choice + (id * 4) as u32,f32::NEG_INFINITY);
            }
            let denominator = sum + 1e-20f32;
            for slot in 0..selected {
                let weight = if c.normalize != 0 { *weights.add(slot) / denominator } else { *weights.add(slot) };
                *weights.add(slot) = weight * c.scale;
            }
        } else {
            // A rejected row cannot become an expert index in a later stage.
            if *status == 0 { *status = 1; }
            for slot in 0..selected { *ids.add(slot) = u32::MAX; *weights.add(slot) = f32::NAN; }
        }
    }
}
