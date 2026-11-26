//! fp4_prefill_tc prompt-case expert row counts (moved out of kernel_cuda.rs to keep it
//! under the repository ratchet's 1,500 lines).
use crate::floor::COLD8K_COUNTS;

/// Expert row counts of a prompt case of `t` rows (2048 = one chunk, 4096 / 8192 = half or
/// a whole 8K prompt in one call, ): the captured cold-8K counts scaled by t / 2048,
/// the -edges table (row counts around every tile boundary, cycled over experts, last one
/// topped up to 20,480 pairs) or uniform 40 per expert per 2048 rows.
pub(super) fn counts(real: bool, name: &str, t: usize) -> Result<Vec<u32>, String> {
    if t != 2048 && t != 4096 && t != 8192 {
        return Err("prompt cases replay one 2048-token chunk, half or a whole 8K prompt".into());
    }
    let scale = (t / 2048) as u32;
    Ok(if real {
        COLD8K_COUNTS.iter().map(|&v| v * scale).collect()
    } else if name.ends_with("-edges") {
        const EDGES: [u32; 30] = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 127, 128, 129, 143, 144, 145, 255, 256, 257, 383, 385];
        let mut v = vec![0u32; 512];
        let (mut sum, mut e) = (0u32, 0usize);
        while sum + EDGES[e % 30] <= 20480 && e < 511 {
            v[e] = EDGES[e % 30];
            sum += v[e];
            e += 1;
        }
        v[e] = 20480 - sum;
        v
    } else {
        vec![40 * scale; 512]
    })
}
