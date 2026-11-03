//! Kind 13 (MXFP8 groups): fixture, twin addresses and production-MXFP8 membership (moved
//! verbatim from kernel_cuda.rs), plus KBENCH_MXROWS_X arms.
use super::{fixture, mx_matrix, params, ptr, Case, Fixture, Gpu, Result, H, RING};
/// Whether matrix j of a kind-13 group is MXFP8 in production: N % 4 == 0 and stored as
/// MXFP8 in the checkpoint (the MoE router and every HC matrix are BF16 there).
pub(super) fn all_mx(c: &Case) -> bool {
    (0..4).all(|j| c.mats[j] == 0 || group_mx(c, j))
}
pub(super) fn group_mx(c: &Case, j: usize) -> bool {
    let n = c.mats[j];
    n > 0 && n % 4 == 0 && !c.name.starts_with("mxg-hc") && !(c.name.starts_with("mxg-moe") && j == 0)
}
/// kind 13: a kind-8 group whose production-MXFP8 matrices (group_mx) are MXFP8; A = one group launch on
/// their exact BF16 expansion, X = the same launch with the twins flagged (weight bit 0).
pub(super) fn mxgroup_fixture(g: &mut Gpu, c: &Case, order: bool) -> Result<Fixture> {
    let mut v = fixture(g, &Case { kind: 8, ..c.clone() }, order)?;
    let k = c.k;
    let mut images = Vec::new();
    for (j, &n) in c.mats.iter().enumerate() {
        if group_mx(c, j) {
            let (bf, img) = mx_matrix(&c.name, n, k, j)?;
            let before: usize = c.mats[..j].iter().sum();
            for &p in &v.ring {
                g.upload_at(p + (before * k * 2) as u64, &bf)?;
            }
            images.push(img);
        }
    }
    let per: usize = images.iter().map(|i| i.len()).sum();
    let slab = g.alloc(per.max(256) * v.ring.len())?;
    for i in 0..v.ring.len() {
        let mut at = slab + (i * per) as u64;
        v.ring_x.push(at);
        for img in &images {
            g.upload_at(at, img)?;
            at += img.len() as u64;
        }
    }
    Ok(v)
}
/// mxs- X arm: one MXFP8 matrix through the group entry, its four ranges on a CTA's warps.
pub(super) fn mx_range_warps(g: &Gpu, f: H, v: &mut Fixture, i: usize) -> Result<()> {
    let mut none = v.rows;
    (none.weight, none.output, none.rows, none.stride) = (0, 0, 0, 0);
    let mut pairs = params::Bf16Pairs { pairs: [v.rows, none] };
    (pairs.pairs[0].weight, pairs.pairs[0].weight2) = (v.ring_x[i] | 1, 0);
    let mut partials = v.aux;
    let result = g.launch(f, v.grid, 128, 4 * RING, &mut [ptr(&mut pairs), ptr(&mut partials)]);
    v.rows.weight = v.ring[i];
    result
}
/// Matrix j's twin in X ring copy i (kind 13), flagged; BF16 matrices keep their address.
pub(super) fn mxgroup_weight(c: &Case, v: &Fixture, i: usize, j: usize, bf16: u64) -> u64 {
    let k = c.k;
    if !group_mx(c, j) {
        return bf16;
    }
    let img = |n: usize| (n * k).div_ceil(256) * 256 + (n.div_ceil(8) * (k / 32) * 8).div_ceil(256) * 256;
    let off: usize = (0..j).filter(|&q| group_mx(c, q)).map(|q| img(c.mats[q])).sum();
    (v.ring_x[i] + off as u64) | 1
}
/// KBENCH_MXROWS_X=<entry>: kind 13 arm X runs this group entry of the
/// X PTX with KBENCH_MXROWS_XSHARED dynamic shared bytes; arm A runs main's production M5-8
/// entry, or KBENCH_MXROWS_A of the A PTX with KBENCH_MXROWS_ASHARED bytes (twins flagged).
pub(super) fn mxrows_x() -> Option<String> {
    std::env::var("KBENCH_MXROWS_X").ok().filter(|v| !v.is_empty())
}
pub(super) fn mxrows_a() -> Option<String> {
    std::env::var("KBENCH_MXROWS_A").ok().filter(|v| !v.is_empty())
}
/// Dynamic shared bytes per arm under KBENCH_MXROWS_X (A: the engine's group launch bytes).
pub(super) fn mxrows_shared(c: &Case, arm: usize) -> u32 {
    let named = |v: &str| std::env::var(v).ok().and_then(|v| v.parse().ok()).unwrap_or(4 * RING);
    if arm == 1 {
        return named("KBENCH_MXROWS_XSHARED");
    }
    if mxrows_a().is_some() {
        return named("KBENCH_MXROWS_ASHARED");
    }
    let total: usize = c.mats.iter().sum();
    if all_mx(c) { params::latency_mx8_shared_rows(c.m, c.k, 4, total, true) as u32 } else { 4 * RING }
}
/// The group entries each arm launches under KBENCH_MXROWS_X: [A, X].
pub(super) fn mxrows_entries(c: &Case) -> [String; 2] {
    let total: usize = c.mats.iter().sum();
    let a = if !all_mx(c) { "flash_bf16_mma_latency_group" }
        else if !params::latency_split(total, true) && params::latency_mx8_widen(c.m, c.k) { "flash_mx8_mma_latency_r_group" }
        else { "flash_mx8_mma_latency_group" };
    [mxrows_a().unwrap_or(a.into()), mxrows_x().unwrap_or_default()]
}
