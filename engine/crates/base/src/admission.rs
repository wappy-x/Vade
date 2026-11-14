//! One diagnostic for byte-budget refusals and the memory input it reports; this module makes no admission decisions.
pub const CUDA: &str = "Linux MemAvailable (free + reclaimable page cache), capped by CUDA total, less this process's CUDA allocations since its first query";
pub const SERVE: &str = "CONTEXT / OUTPUT or max sessions: spark-serve --context-tokens / --output-tokens / --max-active; Flash positional CONTEXT / FLASH_SERVE_OUTPUT_TOKENS / FLASH_SERVE_MAX_ACTIVE";
pub const FLASH: &str = "CONTEXT / OUTPUT (positional context / FLASH_SERVE_OUTPUT_TOKENS) or FLASH_SERVE_MAX_ACTIVE";
// Last MemAvailable read for admission in this process; 0 until one is read.
static MEM_AVAILABLE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub fn bytes(n: u128) -> String { format!("{:.2} GiB ({n} bytes)", n as f64 / 1073741824.0) }
pub fn refusal(required: u128, available: usize, asked: &str, source: &str, controls: &str) -> String {
    let seen = MEM_AVAILABLE.load(std::sync::atomic::Ordering::Relaxed);
    format!("Requested size does not fit.\nAsked: {asked}; total {}.\nAvailable: {} from {source}.\nGap: {}.\nNext: 1. Lower {controls}. 2. In containers, spark-serve supports --gpu-budget-mib N; size N by rounding up (weights + shared workspace + session reservations + retained cache + headroom) / 1048576, within verified host capacity. Flash uses SEQPAR_MEMORY_BYTES in bytes; its CUDA safety cap still applies. 3. Use more Sparks with supported tensor parallelism.{}",
        bytes(required), bytes(available as u128), bytes(required.saturating_sub(available as u128)),
        if seen == 0 { String::new() } else { format!("\n{}", memory_line(seen, required, available)) })
}
/// The numbers a memory refusal saw and the fix, on one line.
pub fn memory_line(mem_available: usize, required: u128, ceiling: usize) -> String {
    format!("Memory: MemAvailable {}; required {}; ceiling {}. Fix: MemAvailable already counts page cache as free, so dropping caches does not help; stop other processes holding memory (see free -g and nvidia-smi) or lower CONTEXT or chats.",
        bytes(mem_available as u128), bytes(required), bytes(ceiling as u128))
}
pub fn token_refusal(required: u128, available: usize, asked: &str, source: &str, controls: &str) -> String {
    format!("Requested size does not fit.\nAsked: {asked}; total {required} tokens; weights/rank and per-session bytes are not part of this token check.\nAvailable: {available} tokens from {source}.\nGap: {} tokens; GiB gap is not applicable to a token ceiling.\nNext: Lower {controls}. A GPU budget override or more Sparks cannot raise this token ceiling.", required.saturating_sub(available as u128))
}
pub fn allocation(required: u128, available: usize, asked: &str) -> String {
    refusal(required, available, asked, &format!("the configured allocation ceiling (derived from {CUDA} with the server reserve, or an explicit budget), including live allocations and graph storage"),
        SERVE)
}
/// One /proc/meminfo field in bytes; the kernel reports kB.
pub fn meminfo_bytes(meminfo: &str, key: &str) -> Result<usize, String> {
    meminfo.lines().find_map(|line| line.strip_prefix(key))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(|value| value.checked_mul(1024))
        .ok_or_else(|| format!("missing or invalid {key} in /proc/meminfo"))
}
/// Free memory on GB10's unified DRAM: MemAvailable counts reclaimable page cache as
/// free and CUDA allocations as used. CUDA free is not an input: in a container it is
/// the host's free pages, without the host cache that the container cannot see.
pub fn unified_available(meminfo: &str, cuda_total: usize) -> Result<usize, String> {
    Ok(meminfo_bytes(meminfo, "MemAvailable:")?.min(cuda_total))
}
/// Remember the MemAvailable an admission read, for the refusal text.
pub fn record_mem_available(bytes: usize) { MEM_AVAILABLE.store(bytes, std::sync::atomic::Ordering::Relaxed); }

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: usize = 1 << 30;
    fn meminfo(free_gib: usize, available_gib: usize, cached_gib: usize) -> String {
        format!("MemTotal:       {} kB\nMemFree:        {} kB\nMemAvailable:   {} kB\nBuffers:           2048 kB\nCached:         {} kB\nSwapFree:              0 kB\n",
            121 * GIB / 1024, free_gib * GIB / 1024, available_gib * GIB / 1024, cached_gib * GIB / 1024)
    }
    #[test]
    fn page_cache_counts_as_free_and_low_mem_available_refuses() {
        let total = 121 * GIB + GIB / 2;
        // A checkpoint just read: little MemFree, ~100 GiB of clean cache, high MemAvailable.
        let cached = unified_available(&meminfo(3, 110, 104), total).unwrap();
        assert_eq!(cached, 110 * GIB);
        // The unchanged ceiling: 10% safety reserve, at least 2 GiB.
        let ceiling = cached - (cached / 10).max(2 << 30);
        assert!(ceiling > 61 * GIB, "a 61.48 GiB load must fit: {ceiling}");
        // Low MemAvailable refuses whatever MemFree or the cache says.
        let low = unified_available(&meminfo(3, 4, 1), total).unwrap();
        assert!(low - (low / 10).max(2 << 30) < 61 * GIB);
        // Capped by CUDA total; container meminfo can exceed what CUDA reports.
        assert_eq!(unified_available(&meminfo(116, 116, 0), 100 * GIB).unwrap(), 100 * GIB);
        for bad in ["MemFree: 1 kB\n", "MemAvailable: x kB\n", "MemAvailable:\n", "MemAvailable: 18446744073709551615 kB\n"] {
            assert!(unified_available(bad, total).unwrap_err().contains("MemAvailable"), "{bad}");
        }
        assert_eq!(meminfo_bytes("MemFree:  7 kB\nMemAvailable:  9 kB\n", "MemFree:").unwrap(), 7 * 1024);
    }
    #[test]
    fn memory_line_names_the_numbers_and_one_fix() {
        let line = memory_line(3 * GIB + GIB / 4, 66016133892, 1338480640);
        for text in ["MemAvailable 3.25 GiB (3489660928 bytes)", "required 61.48 GiB (66016133892 bytes)", "ceiling 1.25 GiB (1338480640 bytes)", "Fix: "] {
            assert!(line.contains(text), "{line}");
        }
        assert!(!line.contains('\n'));
    }
}
