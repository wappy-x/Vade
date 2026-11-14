// The chain entries of a one-row decode step. Part of dense_latency.rs.

/// flash_mx8_mma_latency for a chained decode step (M1, not split, wide1 shape): the next
/// kernel may start at once, and this kernel reads its first ring before the wait.
/// # Safety
/// As flash_mx8_mma_latency, with latency_mx8_wide1(1, K), and a programmatic or plain
/// launch: the input is read only after griddepcontrol.wait.
// SAFETY: griddepcontrol touches no memory. Before its wait this kernel reads weights and
// scales only, which no kernel writes during a step; it reads the input after the wait.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_chain(mut p: Bf16Rows, _scales: u64, _partials: u64) {
    p.weight |= 1;
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    wide8::<1, true>(solo(p));
}

/// flash_mx8_mma_latency_group for a chained decode step (M1, more than 4096 rows).
/// # Safety
/// As flash_mx8_mma_latency_group, with latency_mx8_wide1(1, K) and not latency_split.
// SAFETY: as flash_mx8_mma_latency_chain: weights and scales before the wait, input after it.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_group_chain(
    g: crate::kernel_params::Bf16Pairs,
    _partials: u64,
) {
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    wide8::<1, true>(group(g));
}

/// flash_bf16_mma_latency_group for a chained decode step (M1-4, split group, BF16 matrices
/// only): each warp issues its first range's D weight stages before the wait. Same ranges,
/// chains, merge. One storage path and no 128-register cap: the D weight stages before the
/// wait spill under it, and the HC groups use 41 of the 192 CTAs, so a second wave costs nothing.
/// # Safety
/// As flash_bf16_mma_latency_group, with M <= 4, latency_split(total, true), no flagged matrix.
// SAFETY: as flash_mx8_mma_latency_chain: weights before the wait, input after it.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_latency_group_chain(
    g: crate::kernel_params::Bf16Pairs,
    _partials: u64,
) {
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    let m = group(g);
    match m.a.tokens {
        1 => intra::<4, 13, 12, 22, 21, 0, 0, true>(m),
        2 => intra::<8, 10, 9, 15, 14, 0, 0, true>(m),
        _ => intra::<16, 6, 5, 9, 8, 0, 0, true>(m),
    }
}

/// Commit order of range()'s prologue and walk: e weight groups, then D groups that hold
/// the input (and the weights past e), then one group per tile. At tile t, wait_group(D-1)
/// must leave both groups of tile t complete, for main's start and for the chain's.
#[cfg(test)]
#[test]
fn chain_ring_waits_cover_each_tile() {
    fn check(d: usize, e: usize, tiles: usize) {
        for t in 0..tiles {
            let (weight, input) = if t < e { (t, e + t) } else { (e + t, e + t) };
            // Before tile t's wait: e + d prologue groups and t walk groups are committed.
            let committed = e + d + t;
            let complete = committed - (d - 1);
            assert!(weight < complete && input < complete, "D {d} e {e} tile {t}");
        }
    }
    for (d, tiles) in [(13, 160), (10, 160), (6, 160), (4, 160), (22, 160)] {
        check(d, 2.min(d), tiles);
        check(d, 0, tiles);
        check(d, d, tiles);
    }
}
