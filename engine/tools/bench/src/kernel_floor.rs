//! Compulsory bytes, independent of CUDA allocation/rotation and of timing.
use super::Case;
#[allow(dead_code, unsafe_code)]
#[path = "../../../crates/gpu/kernel_params.rs"]
mod params;

pub(super) fn bf16(n: usize, k: usize) -> usize { 2 * n * k }
/// model/flash/mxfp8.rs: BLOCK=32. blocks::mx8_image stores E4M3 in
/// flash_bf16_element order, then UE8M0 [row/8][K/32][8]. The kernel
/// reads all eight scale lanes, including padded rows (not alignment gaps).
pub(super) fn mxfp8(n: usize, k: usize) -> usize {
    n * k + n.div_ceil(8) * (k / engine_model::flash::mxfp8::BLOCK) * 8
}
/// gpu/src/fp4_abi.rs: row-major E2M1 nibbles, E4M3 per K16,
/// and one FP32 global multiplier per matrix. No allocation alignment.
pub(super) fn nvfp4(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / engine_model::flash::nvfp4::BLOCK + 4
}
fn packed_input(c: &Case) -> usize { c.m.div_ceil(16) * c.k.div_ceil(16) * 128 * 8 }
fn live_input(c: &Case) -> usize {
    c.m.div_ceil(16) * c.k.div_ceil(16) * params::flash_latency_lanes(c.m) * 16 * 2
}
/// Counts/starts only for active experts; live task records plus their count;
/// sorted pair IDs and (down only) pair->output-slot IDs. Expert descriptors
/// contribute the consumed Matrix pointer triples and width, not unused fields.
fn route_bytes(active: usize, tasks: usize, pairs: usize, down: bool) -> usize {
    active * (8 + 24 * if down { 1 } else { 2 } + 4)
        + 4 + tasks * 8 + pairs * 4 * if down { 2 } else { 1 }
}
/// `legacy` is the fixture's original byte count. Families outside the matrix
/// cases retain their existing floor exactly. MX A reads expanded BF16, X MXFP8.
pub(super) fn account(c: &Case, arm: usize, legacy: usize) -> (usize, String) {
    let (m, n, k) = (c.m, c.n, c.k);
    let (mut bytes, mut basis) = (legacy, match c.kind {
        30 => "FP32 Q + distinct selected K + visible FP32 scores",
        31 => "distinct FP32 V + probabilities + gates + denominators + FP32 output",
        32 => "FP32 index query + pooled keys + scores",
        33 => "FP32 gather: selected 1024-byte rows per KV head and token",
        40..=45 => "FP32 q/k/v/output + beta/g + state read/write per piece + metadata",
        9 | 10 => "legacy BF16 floor: 2 B/elem x N x K + FP32 input + reserved packed input + FP32 output",
        _ => "legacy BF16 floor: 2 B/elem x N x K + reserved packed input + FP32 output + head/HC scratch",
    }.to_owned());
    match c.kind {
        1 => {
            let wide = arm == 1 && std::env::var("KBENCH_XKERNEL").is_ok_and(|n| n.starts_with("flash_next_head32"));
            let reads = m.div_ceil(if wide { 32 } else { 16 });
            // This deterministic head fixture's BF16 exponent span fits every LBF12 slot.
            let weights = if c.real { n * k * 3 / 2 + n * k / 32 + n * 2 } else { bf16(n, k) };
            bytes = weights * reads + m.div_ceil(8) * (k / 16) * 32 * 16 + m * n * 4 + m * 192 * 3 * 4;
            basis = format!("head {} as stored: {} B/pass x {} passes + packed input + FP32 logits/argmax partials", if c.real { "lossless LBF12" } else { "BF16" }, weights, reads);
        }
        11 => {
            let input = live_input(c);
            bytes = legacy - packed_input(c) + input;
            let weights = if arm == 1 { mxfp8(n, k) } else { bf16(n, k) };
            if arm == 1 { bytes = bytes - bf16(n, k) + weights; }
            basis = format!("{}; weights/scales {} B + packed BF16 hi/lo input {} B ({} lanes/K16, two planes) + FP32 output {} B; no row tables",
                if arm == 1 { "mxfp8 as stored: 1 B/elem + UE8M0/32, [ceil(N/8)][K/32][8] scales" }
                else { "bf16 as stored: 2 B/elem x N x K" }, weights, input, params::flash_latency_lanes(m), m * n * 4);
        }
        3..=6 | 20 | 21 => {
            let down = matches!(c.kind, 4 | 6 | 21);
            let pairs = m * 10;
            let (active, tasks) = if matches!(c.kind, 5 | 6) {
                if c.real { (COLD8K_COUNTS.iter().filter(|&&v| v != 0).count(), COLD8K_COUNTS.iter().map(|&v| (v as usize).div_ceil(128)).sum()) }
                else { (512, 512 * (40 * m / 2048).div_ceil(128)) }
            } else { let a = match (c.kind, m) { (20 | 21, 8) => 60, (20 | 21, 5) => 39, (20 | 21, 4) => 30, (20 | 21, 2) => 16, _ => 10 }; (a, a) };
            let tables = route_bytes(active, tasks, pairs, down);
            // kind 4 reserved FP32-stride slots but writes N BF16 values per pair.
            if c.kind == 4 { bytes -= pairs * n * 2; }
            bytes += tables;
            basis = format!("nvfp4 as stored: 0.5 B/elem + E4M3/16 + 4 B FP32 global/matrix; {} active experts x {} matrices x {} B; {} input {} B + BF16 output {} B + route/descriptor tables {} B",
                active, if down { 1 } else { 2 }, nvfp4(n, k), if down { "BF16" } else { "FP32" },
                if down { pairs * k * 2 } else { m * k * 4 }, pairs * n * 2, tables);
        }
        _ => {}
    }
    (bytes, basis)
}

/// Captured expert counts of one cold-8K prompt chunk (2048 tokens, top-10, 205 active
/// experts).
pub(super) const COLD8K_COUNTS: [u32; 512] = [
    70, 0, 0, 246, 423, 0, 0, 142, 0, 333, 75, 0, 141, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 47, 0, 70, 0,
    141, 0, 70, 0, 71, 1, 0, 0, 0, 0, 0, 81, 281, 0, 142, 0, 0, 140, 70, 0, 72, 1, 2, 0, 13, 0, 210, 0, 0, 2, 187, 140,
    71, 1, 71, 19, 2, 0, 3, 141, 107, 0, 0, 69, 0, 0, 0, 0, 1, 70, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 224, 0, 72,
    64, 352, 144, 0, 70, 0, 0, 0, 0, 72, 0, 0, 0, 14, 0, 0, 0, 0, 0, 66, 0, 71, 0, 3, 0, 0, 0, 0, 62, 0, 423, 0,
    0, 0, 141, 0, 0, 141, 7, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 0, 146, 70, 0, 0, 0, 70, 4, 0, 283, 0, 114, 0,
    212, 0, 299, 0, 70, 1, 0, 141, 141, 1, 2, 0, 0, 0, 0, 1, 212, 0, 140, 70, 423, 0, 0, 0, 0, 66, 0, 0, 70, 0, 0, 0,
    1, 0, 0, 70, 57, 0, 281, 0, 563, 71, 0, 279, 0, 141, 0, 20, 140, 0, 0, 1, 0, 0, 0, 1, 0, 258, 505, 2, 0, 3, 0, 0,
    140, 0, 0, 0, 131, 0, 0, 0, 0, 70, 70, 111, 0, 30, 0, 0, 0, 0, 140, 0, 0, 0, 70, 1, 0, 1, 0, 282, 0, 0, 3, 0,
    21, 0, 0, 29, 0, 0, 115, 0, 71, 0, 0, 0, 0, 0, 140, 0, 0, 0, 423, 282, 0, 0, 70, 68, 1, 0, 73, 0, 0, 70, 71, 0,
    0, 101, 70, 0, 0, 141, 29, 0, 1, 0, 0, 70, 54, 0, 77, 0, 70, 140, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 71, 0, 0, 141,
    70, 0, 0, 0, 0, 0, 0, 208, 1, 0, 141, 1, 70, 0, 140, 0, 0, 0, 75, 0, 1, 70, 149, 0, 0, 0, 0, 144, 422, 353, 0, 0,
    1, 0, 1, 0, 1, 0, 140, 0, 141, 0, 0, 1, 0, 61, 0, 0, 0, 141, 0, 141, 0, 0, 0, 38, 1, 1, 0, 140, 0, 70, 65, 0,
    1, 0, 0, 0, 0, 0, 0, 69, 70, 0, 3, 1, 0, 0, 71, 1, 0, 144, 0, 8, 0, 0, 0, 70, 0, 139, 0, 0, 140, 0, 1, 213,
    70, 0, 67, 0, 0, 0, 0, 0, 0, 141, 184, 0, 0, 0, 0, 141, 0, 0, 0, 1, 0, 138, 0, 132, 0, 0, 0, 10, 0, 0, 1, 0,
    0, 0, 0, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 70, 0, 0, 0, 0, 0, 69, 0, 128, 70, 0, 17, 70, 0, 0, 72,
    0, 281, 0, 0, 0, 210, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 1, 8, 0, 0, 0, 72, 0, 70, 71, 0, 0, 0
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn bf16_bytes() { assert_eq!(bf16(4, 64), 512); }
    #[test] fn mxfp8_bytes() {
        assert_eq!(mxfp8(8, 64), 528);
        assert_eq!(mxfp8(4, 64), 272); // padded scale rows, not 264
        assert_eq!(mxfp8(12288, 2560), 32_440_320);
    }
    #[test] fn nvfp4_bytes() { assert_eq!(nvfp4(640, 2560), 921_604); }
}
