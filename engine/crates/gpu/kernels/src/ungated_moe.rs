//! BF16 expert-ID fold with FP32 route weights and shared added last.
use crate::router_abi::{UngatedMoeFinish, MAX_EXPERTS};

#[no_mangle]
pub extern "ptx-kernel" fn ungated_moe_finish(p: UngatedMoeFinish) {
    // SAFETY: The caller checks p with UngatedMoeFinish::checked. It retains
    // disjoint device spans through completion. Successful rows have unique
    // IDs and finite weights from the router. Partials and shared are complete
    // rank-reduced rows. The block has ROUTER_THREADS threads. Hidden width is
    // a multiple of that count, so a block contains columns of one token only.
    // Lane zero sorts bounded slots, then every lane reaches the barrier.
    unsafe {
        let at = crate::index();
        let (h,k) = (p.hidden as usize,p.selected as usize);
        let token = at / h;
        if token >= p.tokens as usize { return; }
        let out = (p.output as *mut f32).add(at);
        if *(p.status as *const u32).add(token) != 0 { *out = 0.0; return; }
        let col = at % h;
        let ids = (p.ids as *const u32).add(token * k);
        let weights = (p.weights as *const f32).add(token * k);
        let partials = (p.partials as *const f32).add(token * k * h);
        let order = crate::ptx::shared::<{ MAX_EXPERTS as usize * 4 }>();
        if core::arch::nvptx::_thread_idx_x() == 0 {
            for slot in 0..k {
                let mut pos = slot;
                while pos > 0 {
                    let previous = crate::ptx::load(order + ((pos-1) * 4) as u32).to_bits() as usize;
                    if *ids.add(previous) < *ids.add(slot) { break; }
                    crate::ptx::store(order + (pos * 4) as u32,f32::from_bits(previous as u32));
                    pos -= 1;
                }
                crate::ptx::store(order + (pos * 4) as u32,f32::from_bits(slot as u32));
            }
        }
        crate::ptx::sync();
        let rounded = crate::bf16::rounded;
        let mut sum = 0.0f32;
        for rank in 0..k {
            let slot = crate::ptx::load(order + (rank * 4) as u32).to_bits() as usize;
            let product = rounded(*partials.add(slot * h + col)) * *weights.add(slot);
            let contribution = rounded(product);
            sum = rounded(sum + contribution);
        }
        *out = rounded(sum + rounded(*(p.shared as *const f32).add(at)));
    }
}
