//! Checked admission for FP32 key/value storage. This is a per-session payload
//! budget, not an OS-memory reservation or a long-context quality guarantee.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]
use std::ops::Range;

// Allow 4096 FP32 KV tokens at 64 layers, eight KV heads and head width 128.
// This is an admission ceiling; sessions allocate only their live cache.
pub const DEFAULT_KV_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

// Single BF16 operands: projection input, gate and packed K/V use two bytes per element.
pub(crate) fn prefill_workspace(tokens: usize, capacity: usize, h: usize, q: usize, k: usize, m: usize, packed_k: usize, shared_query: bool) -> [(usize, usize); 10] {
    let projection_input = tokens.div_ceil(16) * 16 * q.max(h) / 2;
    let gate = tokens.div_ceil(16) * 16 * m / 2;
    let keys_len = capacity.div_ceil(64) * 64 * packed_k / 2;
    let values_len = keys_len;
    let row = (tokens * h.max(q)).div_ceil(16) * 16;
    let query = row + projection_input.max(values_len).div_ceil(16) * 16;
    let keys = query + if shared_query {0} else {tokens * q};
    // Projection operands and the packed KV prefix have disjoint lifetimes.
    let offsets = [0, if shared_query {0} else {query}, keys, keys + tokens*k, 0, 0, query, row, keys, row];
    let lengths = [tokens*h, tokens*q, tokens*k, tokens*k, tokens*q, tokens*h,
        gate, projection_input, keys_len, values_len];
    std::array::from_fn(|i| (offsets[i], lengths[i]))
}

/// Shared by sequence execution and its cost model: bound scratch relative to
/// ordinary prefill of the local span, including the retained hidden rows.
pub fn sequence_chunk_size(tokens: usize, end: usize, h: usize, q: usize, k: usize, m: usize, packed_k: usize) -> usize {
    if tokens < 6144 { return 512; }
    let extent = |rows, capacity, packed_k, shared_query| prefill_workspace(rows, capacity, h, q, k, m, packed_k, shared_query).into_iter().map(|(offset, len)| offset + len).max().unwrap();
    let mut rows = 4096;
    let budget = (rows * h + extent(rows, tokens, k, false)) * 4 + 99_000_000;
    while rows > 128 && (tokens * h + extent(rows, end, packed_k, true)) * 4 > budget {
        rows /= 2;
    }
    rows
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvCapacity {
    pub stride: usize,
    pub elements_per_layer: usize,
    pub bytes: usize,
}

pub(crate) fn admit_kv(
    layers: Range<usize>,
    total_layers: u64,
    kv_heads: u64,
    head_dim: u64,
    max_tokens: usize,
    max_positions: u64,
    budget_bytes: usize,
) -> Result<KvCapacity, String> {
    let total = usize::try_from(total_layers).map_err(|_| "layer count exceeds host address space")?;
    if layers.start >= layers.end || layers.end > total {
        return Err("invalid KV layer range".into());
    }
    if kv_heads == 0 || head_dim == 0 {
        return Err("KV head count and dimension must be nonzero".into());
    }
    if max_tokens == 0 {
        return Err("KV token capacity must be nonzero".into());
    }
    if budget_bytes == 0 {
        return Err("KV byte budget must be nonzero".into());
    }
    let tokens = u64::try_from(max_tokens).map_err(|_| "KV token capacity exceeds u64")?;
    if tokens > max_positions {
        return Err(spark_gpu_probe::admission::token_refusal(tokens as u128, max_positions as usize, &format!("CONTEXT={tokens}; OUTPUT included in CONTEXT"), "the configured position ceiling", spark_gpu_probe::admission::SERVE));
    }
    let stride = kv_heads.checked_mul(head_dim).ok_or("KV stride overflow")?;
    let stride = usize::try_from(stride).map_err(|_| "KV stride exceeds host address space")?;
    let elements_per_layer = stride.checked_mul(max_tokens).ok_or("KV element count overflow")?;
    let vector_bytes = elements_per_layer.checked_mul(std::mem::size_of::<f32>())
        .ok_or("KV vector byte count overflow")?;
    if vector_bytes > isize::MAX as usize {
        return Err(spark_gpu_probe::admission::refusal(vector_bytes as u128, isize::MAX as usize, &format!("KV vector/session x 1; CONTEXT={max_tokens}; OUTPUT included in CONTEXT; weights/rank not supplied"), "the Rust allocation size limit for one contiguous vector", spark_gpu_probe::admission::SERVE));
    }
    let bytes = vector_bytes.checked_mul(2).and_then(|n| n.checked_mul(layers.len()))
        .ok_or("KV total byte count overflow")?;
    if bytes > budget_bytes {
        return Err(spark_gpu_probe::admission::refusal(bytes as u128, budget_bytes, &format!("KV payload/session {} x 1 session; CONTEXT={max_tokens}; OUTPUT included in CONTEXT; weights/rank not supplied", spark_gpu_probe::admission::bytes(bytes as u128)), "the caller's supplied budget for logical KV payload", spark_gpu_probe::admission::SERVE));
    }
    Ok(KvCapacity { stride, elements_per_layer, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_context_admission_boundaries() {
        // Independently hand-calculated dense Qwen3-8B FP32 K+V:
        // 36 layers * 8 KV heads * 128 elements * 2 arrays * 4 bytes.
        let full = |tokens, budget| admit_kv(0..36, 36, 8, 128, tokens, 40960, budget);
        let legacy = full(3640, DEFAULT_KV_BUDGET_BYTES).unwrap();
        assert_eq!(legacy.bytes, 1_073_479_680);
        assert_eq!(legacy.stride, 1024);
        assert_eq!(legacy.elements_per_layer, 3_727_360);
        let required = 12_079_595_520usize;
        assert_eq!(full(40960, required).unwrap().bytes, required);
        assert!(full(40960, required - 1).unwrap_err().contains("supplied budget"));
        for text in ["Requested size does not fit.", "11.25 GiB (12079595520 bytes)", "12079595519 bytes", "Gap: 0.00 GiB (1 bytes)", "CONTEXT=40960"] { assert!(full(40960, required - 1).unwrap_err().contains(text)); }
        assert!(full(40961, usize::MAX).unwrap_err().contains("position ceiling"));
        assert!(full(0, usize::MAX).unwrap_err().contains("nonzero"));
        assert!(full(1, 0).unwrap_err().contains("budget must be nonzero"));
        assert_eq!(admit_kv(18..36, 36, 8, 128, 40960, 40960, required).unwrap().bytes, 6_039_797_760);
        // Unequal ranges charge only their actual local layers, regardless of start.
        assert_eq!(admit_kv(2..5, 7, 2, 3, 5, 9, 720).unwrap().bytes, 720);
        for range in [0..0, std::ops::Range { start: 4, end: 3 }, 0..8] {
            assert!(admit_kv(range, 7, 2, 3, 5, 9, 720).unwrap_err().contains("layer range"));
        }
        for (heads, dimension) in [(0, 3), (2, 0)] {
            assert!(admit_kv(0..1, 1, heads, dimension, 1, 9, usize::MAX).unwrap_err().contains("nonzero"));
        }
        assert!(admit_kv(0..1, 1, u64::MAX, 2, 1, 9, usize::MAX).unwrap_err().contains("stride overflow"));
        assert!(admit_kv(0..1, 1, 2, 1, usize::MAX, u64::MAX, usize::MAX).unwrap_err().contains("element count overflow"));
        assert!(admit_kv(0..1, 1, 1, 1, usize::MAX / 4, u64::MAX, usize::MAX).unwrap_err().contains("allocation size limit"));
        assert!(admit_kv(0..5, 5, 1, 1, usize::MAX / 16, u64::MAX, usize::MAX).unwrap_err().contains("total byte count overflow"));
    }
}
