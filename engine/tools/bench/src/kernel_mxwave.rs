//! MX-WAVE (lane CHAIN-2): the recorded MXFP8 decode entry `resident_launch_mx`, as the model
//! graph replays it, from a baseline image against a candidate image, plus the direct entry
//! `flash_next_mxfp8_m1` from the baseline. Real decode shapes (MX o 2048x8192, MX down
//! 2048x12288, a 4096x4096 control), synthetic finite E4M3 weights and E8M0 scales, cold
//! weights by rotating copies. Every output is compared byte for byte across all arms.
//! Safe code only; the existing driver wrapper holds the GPU boundary.
#![forbid(unsafe_code)]
use super::super::{Gpu, Result, ptr, sha256};
use std::time::Instant;
/// Mirrors gpu/src/next_params.rs (repr(C)); the size checks below pin the layout.
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MxProjection { weights: u64, scales: u64, input: u64, output: u64, rows: u32, cols: u32, tokens: u32, input_bf16: u32, round: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MxLayer { projections: [MxProjection; 9], ends: [u32; 10], count: u32 }
fn layer_bytes(l: &MxLayer) -> Vec<u8> {
    // Field by field, so padding bytes are zero, as the recorder's encoder writes them.
    let mut out = vec![0u8; std::mem::size_of::<MxLayer>()];
    for (i, p) in l.projections.iter().enumerate() {
        let at = i * std::mem::size_of::<MxProjection>();
        for (k, v) in [p.weights, p.scales, p.input, p.output].iter().enumerate() { out[at + k * 8..at + k * 8 + 8].copy_from_slice(&v.to_le_bytes()); }
        for (k, v) in [p.rows, p.cols, p.tokens, p.input_bf16, p.round].iter().enumerate() { out[at + 32 + k * 4..at + 36 + k * 4].copy_from_slice(&v.to_le_bytes()); }
    }
    let ends = 9 * std::mem::size_of::<MxProjection>();
    for (i, v) in l.ends.iter().enumerate() { out[ends + i * 4..ends + i * 4 + 4].copy_from_slice(&v.to_le_bytes()); }
    out[ends + 40..ends + 44].copy_from_slice(&l.count.to_le_bytes());
    out
}
struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
pub(crate) fn run(base_path: &str, candidate_path: &str) -> Result<()> {
    assert_eq!(std::mem::size_of::<MxProjection>(), 56); assert_eq!(std::mem::size_of::<MxLayer>(), 552);
    let (a, b) = (std::fs::read(base_path).map_err(|e| e.to_string())?, std::fs::read(candidate_path).map_err(|e| e.to_string())?);
    println!("MXWAVE baseline_sha256={} candidate_sha256={} scope=one-projection-kernel-not-model", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?; let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let (rec_a, rec_b, direct) = (g.function(ma, "resident_launch_mx")?, g.function(mb, "resident_launch_mx")?, g.function(ma, "flash_next_mxfp8_m1")?);
    // (label, rows, cols, copies): copies x weight bytes is at least 4x the 24 MB L2.
    for (label, rows, cols, copies) in [("mx_o", 2048u32, 8192u32, 8usize), ("mx_down", 2048, 12288, 6), ("mx_4096", 4096, 4096, 8)] {
        let (n, k) = (rows as usize, cols as usize);
        let mut r = Rng(0x9e37 + n as u64 * 31 + k as u64);
        let mut copies_w = Vec::new(); let mut copies_s = Vec::new();
        for _ in 0..copies {
            // Finite E4M3 codes (no 0x7f/0xff) and E8M0 scales near 2^0.
            let w: Vec<u8> = (0..n * k).map(|_| { let c = (r.next() >> 24) as u8; if c & 127 == 127 { c & 0xfe } else { c } }).collect();
            let s: Vec<u8> = (0..n * k / 32).map(|_| 120 + (r.next() % 14) as u8).collect();
            copies_w.push(g.upload(&w)?); copies_s.push(g.upload(&s)?);
        }
        for bf16 in [0u32, 1] {
            let input: Vec<u8> = if bf16 == 1 { (0..k).flat_map(|_| (((r.next() >> 33) as u16 & 0x3fff) | 0x3800).to_le_bytes()).collect() }
                else { (0..k).flat_map(|_| (((r.next() >> 40) as f32 / 16777216.0) * 2.0 - 1.0).to_le_bytes()).collect() };
            let x = g.upload(&input)?;
            let outs = [g.upload(&vec![0xffu8; n * 4])?, g.upload(&vec![0xffu8; n * 4])?, g.upload(&vec![0xffu8; n * 4])?];
            let layer = |copy: usize, out: u64| -> MxLayer {
                let mut l = MxLayer::default();
                l.projections[0] = MxProjection { weights: copies_w[copy], scales: copies_s[copy], input: x, output: out, rows, cols, tokens: 1, input_bf16: bf16, round: 0 };
                l.ends[1] = rows.div_ceil(4); l.count = 1; l
            };
            let blocks = rows.div_ceil(4);
            // One command table per (copy, arm output), uploaded once; index 0.
            let mut tables = Vec::new();
            for copy in 0..copies { for arm in 0..2 {
                let mut command = vec![0u8; 1040]; command[..4].copy_from_slice(&4u32.to_le_bytes());
                let payload = layer_bytes(&layer(copy, outs[arm])); command[16..16 + payload.len()].copy_from_slice(&payload);
                tables.push(g.upload(&command)?);
            } }
            let launch = |g: &Gpu, arm: usize, copy: usize| -> Result<()> {
                if arm == 2 { let mut l = layer(copy, outs[2]); return g.launch(direct, [blocks, 1, 1], 128, 0, &mut [ptr(&mut l)]); }
                let (mut t, mut index) = (tables[copy * 2 + arm], 0u64);
                g.launch(if arm == 0 { rec_a } else { rec_b }, [blocks, 1, 1], 128, 0, &mut [ptr(&mut t), ptr(&mut index)])
            };
            // Bytes: every copy, all three arms equal.
            for copy in 0..copies {
                for o in &outs { g.upload_at(*o, &vec![0xffu8; n * 4])?; }
                for arm in 0..3 { launch(&g, arm, copy)?; }
                g.sync()?;
                let planes = outs.iter().map(|o| g.read(*o, n * 4)).collect::<Result<Vec<_>>>()?;
                if planes[0] != planes[1] || planes[0] != planes[2] {
                    let at = planes[0].iter().zip(&planes[1]).position(|(p, q)| p != q);
                    return Err(format!("MXWAVE BYTE FAIL shape={label} bf16={bf16} copy={copy} first_byte_base_vs_candidate={at:?} direct_equal={}", planes[0] == planes[2]));
                }
            }
            println!("MXWAVE BYTES PASS shape={label} rows={rows} cols={cols} input_bf16={bf16} copies={copies} arms=3 digest={}", sha256::digest(&g.read(outs[0], n * 4)?));
            // Time: 9 repeats, arms alternate, 4 x copies launches each (rotating copies: cold weights).
            let mut t = vec![Vec::new(); 3]; let launches = copies * 4;
            for repeat in 0..9 { for k3 in 0..3 {
                let arm = if repeat % 2 == 0 { k3 } else { 2 - k3 };
                launch(&g, arm, copies - 1)?; g.sync()?;
                let started = Instant::now();
                for i in 0..launches { launch(&g, arm, i % copies)?; }
                g.sync()?; t[arm].push(started.elapsed().as_secs_f64() * 1e6 / launches as f64);
            } }
            let m = t.into_iter().map(median).collect::<Vec<_>>();
            let bytes = (n * k + n * k / 32) as f64;
            println!("MXWAVE TIME shape={label} input_bf16={bf16} blocks={blocks} recorded_base_us={:.2} recorded_candidate_us={:.2} direct_base_us={:.2} saved_us={:.2} base_gbs={:.1} candidate_gbs={:.1} direct_gbs={:.1} samples=9x{launches} scope=host-wall-per-launch-back-to-back",
                m[0], m[1], m[2], m[0] - m[1], bytes / m[0] / 1e3, bytes / m[1] / 1e3, bytes / m[2] / 1e3);
        }
        g.clear()?;
    }
    println!("MXWAVE PASS");
    Ok(())
}
