use engine_model::cartridge::Cartridge;
use spark_serve::{budget::minimum_free_bytes, protocol::quote};
use std::{fs, path::{Path, PathBuf}};

pub fn list(root: &Path) -> Result<(), String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    #[cfg(target_os = "linux")]
    let (cuda_free, total) = spark_gpu_probe::driver::device_memory_info(0)?;
    // Same unified-memory measure as serving admission (MemAvailable; page cache counts).
    #[cfg(target_os = "linux")]
    let available = spark_gpu_probe::driver::unified_available_bytes(total)?;
    #[cfg(not(target_os = "linux"))]
    let (cuda_free, total, available) = (0usize, 121usize * 1024 * 1024 * 1024, 121usize * 1024 * 1024 * 1024);
    let memory_source = if cfg!(target_os = "linux") { "device" } else { "GB10_capacity_estimate" };
    if available == 0 { return Err("no free unified memory available".into()); }
    eprintln!("memory_source={memory_source} free_memory_bytes={available} cuda_free_bytes={cuda_free} total_device_bytes={total}; fits and needs_sparks estimate memory only, including serving/OS reserve");
    let mut paths = Vec::new();
    collect_models(&root, &mut paths)?;
    paths.sort();
    if paths.is_empty() { return Err("no model folders containing config.json found".into()); }
    for path in paths {
        let name = path.strip_prefix(&root).unwrap_or(&path);
        let name = if name.as_os_str().is_empty() { path.file_name().unwrap_or_default() } else { name.as_os_str() };
        let size = directory_bytes(&path)?;
        match Cartridge::inspect(&path) {
            Ok(cartridge) => {
                let supported = cartridge.reason.is_none();
                let required = if let Some(config) = &cartridge.config {
                    minimum_free_bytes(config)? as u64
                } else if let Some(config) = &cartridge.gemma4 {
                    config.minimum_free_bytes().map_err(|e| e.to_string())?
                } else if cartridge.kind == engine_model::cartridge::Kind::Qwen3_5 {
                    cartridge.fit(28672.min(cartridge.max_positions), available as u64)
                        .map_err(|e| e.to_string())?.required_bytes
                        .checked_add(3 * 1024 * 1024 * 1024).ok_or("model memory size overflow")?
                } else {
                    // No runtime contract yet: retain native checkpoint bytes and
                    // budget 3 GiB for serving/OS. Do not promise architecture support.
                    cartridge.weight_bytes.checked_add(3 * 1024 * 1024 * 1024)
                        .ok_or("model memory size overflow")?
                };
                let sparks = required.div_ceil(available as u64).max(1);
                let geometry = if let Some(c) = &cartridge.gemma4 {
                    format!(" text_weight_bytes={} weights_complete={} memory_estimate=yes kv_bytes_at_max_context={} kv_context_tokens={} rope_bytes_at_max_context={} required_free_bytes_at_max_context={} sliding_layers={} full_layers={} sliding_window={} head_dim={} global_head_dim={} vocabulary={}",
                        c.weight_bytes().map_err(|e| e.to_string())?, yes(cartridge.weights_complete),
                        c.kv_bytes(c.max_positions).map_err(|e| e.to_string())?, c.max_positions,
                        c.rope_bytes(c.max_positions).map_err(|e| e.to_string())?,
                        c.required_free_bytes(c.max_positions, 1).map_err(|e| e.to_string())?,
                        c.layer_types.iter().filter(|&&k| k == engine_model::gemma4::Attention::Sliding).count(),
                        c.layer_types.iter().filter(|&&k| k == engine_model::gemma4::Attention::Full).count(),
                        c.sliding_window, c.head_dim, c.global_head_dim, c.vocab_size)
                } else { String::new() };
                println!("name={} architecture={} size_bytes={size} weight_bytes={} precision={} supported={} required_free_bytes={required}{geometry} fits={} needs_sparks={sparks}{}",
                    quote(&name.to_string_lossy()), quote(&cartridge.architecture), cartridge.weight_bytes,
                    quote(&cartridge.precision), yes(supported), yes(required <= available as u64),
                    cartridge.reason.as_ref().map(|reason| format!(" reason={}", quote(reason))).unwrap_or_default());
            }
            Err(error) => println!("name={} architecture=unknown size_bytes={size} precision=unknown supported=no fits=no needs_sparks=unknown reason={}",
                quote(&name.to_string_lossy()), quote(&error.to_string())),
        }
    }
    Ok(())
}

fn yes(value: bool) -> &'static str { if value { "yes" } else { "no" } }

fn collect_models(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if root.join("config.json").is_file() {
        out.push(root.to_owned());
        return Ok(());
    }
    for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_dir()
            && !entry.file_name().to_string_lossy().starts_with('.') {
            collect_models(&entry.path(), out)?;
        }
    }
    Ok(())
}

fn directory_bytes(root: &Path) -> Result<u64, String> {
    let mut bytes = 0u64;
    for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        let size = if kind.is_dir() { directory_bytes(&entry.path())? }
            else if entry.path().is_file() { fs::metadata(entry.path()).map_err(|e| e.to_string())?.len() }
            else { 0 };
        bytes = bytes.checked_add(size).ok_or("directory size overflow")?;
    }
    Ok(bytes)
}
