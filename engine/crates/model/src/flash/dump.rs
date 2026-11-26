//! VADEDBG1 per-layer debug dump lines written by the FP32 CPU reference
//! (`refcpu::forward_traced`) and the GPU decode path, so a box run diffs
//! layer by layer against CPU goldens (checksums + first 16 values per
//! stage, MoE routes, per-token top-5 logits). Lines: `VADEDBG1 ...` header;
//! `token <pos> <id>`; `rec <stage> <len> <fnv-hex> <bits,...>` (FNV-1a 64
//! over little-endian FP32 bit patterns; up to 16 values as 8-hex bits);
//! `moe <layer> <ids> <weight-bits>` (ascending expert id);
//! `top <argmax> <id:bits,...>` (five entries, lowest-id ties). Parsing and
//! comparison live with the GPU gate test that consumes both files.
pub fn checksum(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for v in values { for byte in v.to_bits().to_le_bytes() { hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3); } }
    hash
}
pub fn rec_line(stage: &str, values: &[f32]) -> String {
    let first: Vec<String> = values.iter().take(16).map(|v| format!("{:08x}", v.to_bits())).collect();
    format!("rec {stage} {} {:016x} {}", values.len(), checksum(values), first.join(","))
}
pub fn moe_line(layer: usize, pairs: &[(usize, f32)]) -> String {
    let ids: Vec<String> = pairs.iter().map(|p| p.0.to_string()).collect();
    let w: Vec<String> = pairs.iter().map(|p| format!("{:08x}", p.1.to_bits())).collect();
    format!("moe {layer} {} {}", ids.join(","), w.join(","))
}
/// Five largest logits, descending value, lowest token id winning ties.
pub fn top5(logits: &[f32]) -> (Vec<(u32, f32)>, u32) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let top: Vec<(u32, f32)> = order.iter().take(5).map(|&i| (i as u32, logits[i])).collect();
    (top.clone(), top.first().map_or(0, |t| t.0))
}
pub fn top_line(logits: &[f32]) -> String {
    let (top, argmax) = top5(logits);
    let entries: Vec<String> = top.iter().map(|(id, v)| format!("{id}:{:08x}", v.to_bits())).collect();
    format!("top {argmax} {}", entries.join(","))
}
