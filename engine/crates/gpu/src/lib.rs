//! GB10 CUDA driver ownership, packed-weight loading and kernel dispatch.
#[path = "../../base/src/admission.rs"]
pub mod admission;
#[path = "../kernel_params.rs"]
pub mod kernel_params;
#[cfg(target_os = "linux")]
pub mod driver;
pub mod tp_mailbox_host;
pub mod tp_route;
#[cfg(target_os = "linux")]
pub mod tp_rdma;
#[cfg(all(test, not(target_os = "linux")))]
#[allow(dead_code)]
#[path = "decode_profile.rs"]
mod decode_profile;

#[cfg(target_os = "linux")]
pub mod load;
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // CUDA wait registrations are Linux-only.
pub mod teardown;
pub mod host_prep;
pub mod routing;
pub mod frontend_math;
pub mod qsa;
#[cfg(target_os = "linux")]
mod qsa_device;
#[path = "../kernels/src/bf16.rs"]
pub mod bf16;
#[cfg(test)]
#[path = "../kernels/src/dense_latency.rs"]
mod dense_latency;
pub mod decode_fuse;
pub mod gdn_params;
pub mod next_params;
pub mod next_math;
pub mod compact;
#[cfg(any(target_os = "linux", test))]
mod reserve_policy;
#[cfg(any(target_os = "linux", test))]
pub use reserve_policy::KvArenaStats;
#[path = "fp8_abi.rs"]
pub mod fp8_abi;

/// Bytes of the dense prompt input tile (flash_bf16_input words, both planes): sized for
/// 4096 rows at the widest input (FLASH_BF16_INPUT_BYTES), twice that with VADE_PROMPT_TILE=8192
/// or auto, so a whole 8192-row tile packs once and its fused producers fit (read once per process,
/// the same values the engine's prompt_tile() accepts).
pub fn flash_dense_reserve_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| match std::env::var("VADE_PROMPT_TILE").ok().as_deref() {
        None | Some("8192") | Some("auto") => 2 * kernel_params::FLASH_BF16_INPUT_BYTES,
        _ => kernel_params::FLASH_BF16_INPUT_BYTES,
    })
}

/// Bytes of the mixed dense planes (H-wide prompt rows, both planes): 4096 rows
/// (FLASH_BF16_MIXED_BYTES), twice that with VADE_PROMPT_TILE=8192, like the dense input tile.
pub fn flash_mixed_reserve_bytes() -> usize {
    flash_dense_reserve_bytes() / kernel_params::FLASH_BF16_INPUT_BYTES * kernel_params::FLASH_BF16_MIXED_BYTES
}

/// The startup routing hook accepts `N:ROW:ROWS[:RANK_MASK]`, as well as `N:ROW`.
pub const TEST_ROUTING_REJECT_WIDTH: bool = true;

/// Exact embedded layer module, without changing the bytes CUDA receives.
pub fn layer_ptx() -> Result<&'static std::ffi::CStr, String> {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/layer.ptx"));
    if bytes.len() <= 1 { return Err("layer PTX missing: build layer kernels before the host executable".into()); }
    std::ffi::CStr::from_bytes_with_nul(bytes).map_err(|e| e.to_string())
}

/// `cubin_sha256=<hex>` when this build embeds a compiled image; empty otherwise.
/// The release packager reads this text in the binary to prove the image is there.
pub const CUBIN_STAMP: &str = include!(concat!(env!("OUT_DIR"), "/cubin_stamp.rs"));

#[cfg(test)]
#[allow(dead_code)]
#[path = "../cubin.rs"]
mod cubin_build_tests;

#[cfg(any(target_os = "linux", test))]
mod module_loading;

#[cfg(any(target_os = "linux", test))]
fn cubin_matches(ptx: &[u8], original: &[u8], device: (i32, i32), driver: i32, minimum: i32) -> bool {
    minimum >= 13000 && driver >= minimum && device == (12, 1) && ptx == original
}

#[test]
fn compiled_image_selection_keeps_ptx_fallback_for_every_mismatch() {
    assert!(cubin_matches(b"PTX", b"PTX", (12, 1), 13000, 13000));
    assert!(cubin_matches(b"PTX", b"PTX", (12, 1), 13010, 13000));
    assert!(!cubin_matches(b"other", b"PTX", (12, 1), 13000, 13000));
    assert!(!cubin_matches(b"PTX", b"PTX", (12, 0), 13000, 13000));
    assert!(!cubin_matches(b"PTX", b"PTX", (12, 1), 12090, 13000));
    assert!(!cubin_matches(b"PTX", b"PTX", (12, 1), 13000, 0));
}

pub fn launch_blocks(elements: usize, threads: u32) -> Result<u32, String> {
    if threads == 0 || threads > 1024 {
        return Err("threads must be in 1..=1024".into());
    }
    let count = u32::try_from(elements).map_err(|_| "element count exceeds kernel ABI")?;
    // Avoid overflow in the usual (n + block - 1) expression.
    Ok(count / threads + u32::from(count % threads != 0))
}

/// A bounded FP32 subrange, independent of whether storage has an arena owner.
pub fn vector_byte_range(elements: usize, range: std::ops::Range<usize>) -> Result<std::ops::Range<usize>, String> {
    if range.start > range.end || range.end > elements { return Err("device clear exceeds vector".into()); }
    let start = range.start.checked_mul(4).ok_or("device clear byte offset overflow")?;
    let end = range.end.checked_mul(4).ok_or("device clear byte offset overflow")?;
    Ok(start..end)
}

/// Host-testable upload contracts; mapped padding never changes logical shape.
pub fn check_vector_shape(logical: usize, mapped_bytes: usize, same_context: bool, requested: usize) -> Result<(), String> {
    if !same_context || logical != requested {
        return Err(format!("device vector context or shape mismatch: requested={requested} logical={logical} mapped_bytes={mapped_bytes} same_context={same_context}"));
    }
    Ok(())
}
pub fn check_vector_upload(logical: usize, mapped_bytes: usize, same_context: bool, bytes: usize) -> Result<(), String> {
    check_vector_shape(logical, mapped_bytes, same_context, logical)?;
    if bytes > logical.checked_mul(4).ok_or("vector byte size overflow")? { return Err("row exceeds device scratch".into()); }
    if bytes > mapped_bytes { return Err("upload exceeds buffer".into()); }
    Ok(())
}

/// Conservative driver metadata funding; independent of CUDA allocation queries.
pub struct GraphAllowance { nodes: usize, bytes: usize }
impl GraphAllowance {
    pub const FLASH_MAX_NODES: usize = 512;
    pub const EXEC_STORAGE_BYTES: usize = 256 * 1024;
    pub const NODE_STORAGE_BYTES: usize = 16 * 1024;

    /// Whole-model captures include projections, recurrence, routed/shared MLPs
    /// and output work. Fund 64 nodes per layer and 32 outside the layers for
    /// every row, including sequential proposal rows. Batched work can use less;
    /// successful capture refunds the difference to its actual node count.
    pub fn whole_model(layers: usize, rows: usize) -> Result<Self, String> {
        if layers == 0 || rows == 0 { return Err("invalid whole-model graph shape".into()); }
        let nodes = layers.checked_mul(64).and_then(|n| n.checked_add(32))
            .and_then(|n| n.checked_mul(rows)).ok_or("CUDA graph node allowance overflow")?;
        Self::new(nodes)
    }
    /// Flash captures one segment at a time, using its serving-funded limit.
    pub fn flash(nodes: usize) -> Result<Self, String> {
        if nodes > Self::FLASH_MAX_NODES { return Err("invalid Flash graph node allowance".into()); }
        Self::new(nodes)
    }
    fn new(nodes: usize) -> Result<Self, String> {
        if nodes == 0 { return Err("invalid CUDA graph node allowance".into()); }
        let bytes = nodes.checked_mul(Self::NODE_STORAGE_BYTES)
            .and_then(|n| n.checked_add(Self::EXEC_STORAGE_BYTES)).ok_or("graph storage accounting overflow")?;
        Ok(Self { nodes, bytes })
    }
    pub fn maximum_bytes(&self) -> usize { self.bytes }
    /// Called after cuGraphGetNodes and before instantiation. The constructor
    /// checked the maximum, so every admitted actual charge fits usize too.
    pub fn storage_bytes(&self, nodes: usize) -> Result<usize, String> {
        if nodes > self.nodes {
            return Err(format!("CUDA graph has {nodes} nodes; funded maximum is {}", self.nodes));
        }
        Ok(Self::EXEC_STORAGE_BYTES + nodes * Self::NODE_STORAGE_BYTES)
    }
}

#[cfg(test)]
mod split_order_tests;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_down_group_check_holds_the_safety_list() {
        let row = |rows2, tokens, weight| kernel_params::Bf16Rows { weight, output: 0, rows: 4, stride: 4, weight2: 32, output2: 0, rows2, stride2: rows2,
            input: 4096, cols: 10240, tokens, input_bf16: 1, hc_silu: 0, input_plane_words: 0 };
        let pairs = |first| kernel_params::Bf16Pairs { pairs: [first, kernel_params::Bf16Rows { rows: 0, rows2: 0, ..row(0, 1, 0) }] };
        let (good, input, down) = (pairs(row(320, 16, 16)), 2 * 640 * 128 * 4, 2 * 20 * 128 * 4);
        let check = |p: &kernel_params::Bf16Pairs, planes| kernel_params::hc_down_group_check(p, planes, 4096, input + down);
        assert!(check(&good, 4096 + input as u64).is_ok());
        assert!(check(&good, 4096 + input as u64 - 4).is_err() && check(&good, 4096 + input as u64 + 4).is_err() && check(&good, u64::MAX).is_err());
        assert!(check(&pairs(row(320, 17, 16)), 4096 + input as u64).is_err() && check(&pairs(row(320, 16, 17)), 4096 + input as u64).is_err());
        assert!(check(&pairs(row(321, 16, 16)), 4096 + input as u64).is_err() && check(&kernel_params::Bf16Pairs { pairs: [good.pairs[0], good.pairs[0]] }, 4096 + input as u64).is_err());
    }
    #[test]
    fn register_decode_selection_boundaries() {
        for rows in [1,2,3,4,5,8,9,16,17,32] { for outputs in [128,256,512,640,1024] { for kind in [0,1,2] {
            assert_eq!(fp8_abi::register_decode(kind,rows,outputs), kind==0 && rows<=16 && outputs<=512);
        } } }
        assert_eq!(fp8_abi::REGISTER_DECODE_TILE,[8,8,128,1024]);
    }
    #[test]
    fn invalid_router_selection() {
        for expert in [f32::NAN, f32::INFINITY, -1.0, 512.0, u32::MAX as f32, 1.5] {
            assert!(!fp8_abi::valid_route(0.0, expert, 0.1, 1, 512));
        }
        assert!(fp8_abi::valid_route(0.0, 511.0, 0.1, 1, 512));
    }
    #[test]
    fn rejected_route_stays_in_range() {
        let (mut ids, mut weights, mut status) = ([u32::MAX; 8], [f32::NAN; 8], 7);
        let mut route = |valid, first| unsafe {
            kernel_params::finish_route(ids.as_mut_ptr(), weights.as_mut_ptr(), &mut status, 8, valid, first);
            (ids, weights, status)
        };
        assert_eq!(route(false, false), ([0, 1, 2, 3, 4, 5, 6, 7], [0.0; 8], 1));
        assert_eq!(route(true, false).2, 1, "a later finite layer keeps the row rejected");
        assert_eq!(route(true, true), ([0, 1, 2, 3, 4, 5, 6, 7], [0.0; 8], 0), "the next forward restarts");
    }
    #[test]
    fn hd256_rejects_trapping_operands() {
        assert!(kernel_params::hd256_operand(1.0));
        for value in [f32::NAN, f32::INFINITY, f32::MAX, -f32::MAX] { assert!(!kernel_params::hd256_operand(value)); }
    }
    #[test]
    fn launch_tail_and_overflow() {
        for (n, blocks) in [(0, 0), (1, 1), (255, 1), (256, 1), (257, 2)] {
            assert_eq!(launch_blocks(n, 256).unwrap(), blocks);
        }
        assert_eq!(launch_blocks(u32::MAX as usize, 256).unwrap(), 16_777_216);
        assert!(launch_blocks(1, 0).is_err());
        assert!(launch_blocks(1, 1025).is_err());
        if usize::BITS > 32 { assert!(launch_blocks(u32::MAX as usize + 1, 256).is_err()); }
    }
    #[test]
    fn whole_model_graph_exceeds_flash_cap() {
        for rows in [1, 4] { assert_eq!(GraphAllowance::whole_model(36, rows).unwrap().storage_bytes(600 * rows).unwrap(),
            GraphAllowance::EXEC_STORAGE_BYTES + 600 * rows * GraphAllowance::NODE_STORAGE_BYTES); }
        assert!(GraphAllowance::flash(512).unwrap().storage_bytes(513).is_err());
    }

    // CPU mirror of route_row's first-strictly-greater scan vs route_warp's
    // strided scan + butterfly (kernels/src/moe.rs), the pair the small-row
    // warp router swaps between. GPU differential runs on the Spark proof.
    #[test]
    fn warp_topk_matches_serial_including_ties() {
        fn pick(scores: &[f32], warp: bool) -> u32 {
            if !warp {
                let (mut best, mut value) = (0usize, f32::NEG_INFINITY);
                for (e, &c) in scores.iter().enumerate() { if c > value { value = c; best = e; } }
                return best as u32;
            }
            let mut lanes: Vec<(f32, u32)> = (0..32).map(|lane| {
                let (mut value, mut best) = (f32::NEG_INFINITY, u32::MAX);
                let mut e = lane;
                while e < scores.len() { if scores[e] > value { value = scores[e]; best = e as u32; } e += 32; }
                (value, best)
            }).collect();
            for d in [16usize, 8, 4, 2, 1] {
                let before = lanes.clone();
                for lane in 0..32 {
                    let (other, id) = before[lane ^ d];
                    if other > lanes[lane].0 || other == lanes[lane].0 && id < lanes[lane].1 { lanes[lane] = (other, id); }
                }
            }
            assert!(lanes.iter().all(|&l| l == lanes[0]));
            lanes[0].1
        }
        fn topk(mut scores: Vec<f32>, selected: usize, warp: bool) -> Vec<u32> {
            (0..selected).map(|_| { let b = pick(&scores, warp); scores[b as usize] = f32::NEG_INFINITY; b }).collect()
        }
        let mut cases: Vec<Vec<f32>> = vec![vec![1.0; 256],
            (0..256).map(|i| i as f32).collect(), (0..256).map(|i| (255 - i) as f32).collect(),
            (0..37).map(|i| ((i * 7) % 5) as f32).collect(), (0..8).map(|i| (i % 3) as f32).collect()];
        let mut state = 0x243f6a88u32;
        for experts in [8usize, 31, 32, 33, 64, 256] {
            cases.push((0..experts).map(|_| { state = state.wrapping_mul(1664525).wrapping_add(1013904223); ((state >> 16) % 9) as f32 / 4.0 }).collect());
        }
        for scores in cases { for selected in [1, 2, 8] { if selected <= scores.len() {
            assert_eq!(topk(scores.clone(), selected, true), topk(scores.clone(), selected, false), "experts={}", scores.len());
        } } }
    }

    // Index arithmetic for the N64 gate/up epilogue and fp8_quantize_activated
    // (kernels/src/fp8.rs): two N64 CTAs plus one warp per 128 group must
    // cover the same columns, codes and scale slots as the fused N128 tile.
    #[test]
    fn n64_gate_up_partitions_match_n128() {
        assert_eq!(2 * (16 * 128 + 64 * 128 + 16 * 4), 20608);
        assert_eq!(2 * 20608, 41216);
        for i in [128usize, 512, 640, 1024] {
            let mut cols = vec![0u32; i];
            for nb in (0..i).step_by(64) { for lane in 0..16 { for j in 0..4 { cols[nb + lane * 4 + j] += 1; } } }
            assert!(cols.iter().all(|&c| c == 1), "epilogue i={i}");
            for rows in [1usize, 9, 36] {
                let (mut input, mut output, mut scales) =
                    (vec![0u32; rows * 2 * i], vec![0u32; rows * i], vec![0u32; rows * (i / 128)]);
                for group in 0..rows * (i / 128) {
                    let (row, base) = (group / (i / 128), group % (i / 128) * 128);
                    for lane in 0..32 { for j in 0..4 {
                        input[row * 2 * i + base + lane * 4 + j] += 1;
                        output[row * i + base + lane * 4 + j] += 1;
                    } }
                    scales[row * (i / 128) + group % (i / 128)] += 1;
                }
                assert!(output.iter().all(|&c| c == 1) && scales.iter().all(|&c| c == 1), "quantize i={i} rows={rows}");
                for row in 0..rows {
                    assert!(input[row * 2 * i..row * 2 * i + i].iter().all(|&c| c == 1));
                    assert!(input[row * 2 * i + i..(row + 1) * 2 * i].iter().all(|&c| c == 0));
                }
            }
        }
    }

    // Shared layout of flash_residual_norm_bf16 (kernels/src/flash_ops.rs):
    // squares [0,20480), rounded x [20480,40960), scale at 40960, and the
    // tid-strided loops must cover every column exactly once.
    #[test]
    fn residual_norm_shared_layout() {
        assert!(40976 <= 48 * 1024, "static shared budget");
        for width in [1usize, 127, 128, 2048, 2560, 5120] {
            assert!(width * 4 <= 20480);
            assert!(20480 + width * 4 <= 40960);
            for threads in [32usize, 128] {
                let mut seen = vec![0u32; width];
                for tid in 0..threads { let mut i = tid; while i < width { seen[i] += 1; i += threads; } }
                assert!(seen.iter().all(|&c| c == 1), "width={width} threads={threads}");
            }
        }
    }
}

#[cfg(test)]
#[path = "../kernels/src/hc_up.rs"]
mod hc_up_oracle;

#[cfg(test)]
mod ab_tests;

/// Conservative envelope shared with the optional target-head kernel.
pub mod head_bound;
