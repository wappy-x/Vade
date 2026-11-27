//! Tool-side reuse of the server's MiB parser and checkpoint/device sizing.
// Engine binaries cannot depend on serve (which depends on engine).
#[allow(dead_code)]
#[path = "../../../serve/src/budget.rs"]
mod shared;

pub(super) fn take(args: &mut Vec<String>) -> Result<Option<usize>, String> {
    let Some(i) = args.iter().position(|s| s == "--gpu-budget-mib") else { return Ok(None); };
    let value = args.get(i + 1).ok_or("--gpu-budget-mib requires a value")?;
    let bytes = shared::parse_mib(value, "--gpu-budget-mib")?;
    args.drain(i..=i + 1);
    if args.iter().any(|s| s == "--gpu-budget-mib") { return Err("duplicate --gpu-budget-mib".into()); }
    Ok(Some(bytes))
}

#[cfg(target_os = "linux")]
pub(super) fn configure(backend: &spark_gpu_probe::driver::Backend<'_>, root: &std::path::Path,
    context: usize, gpu_bytes: Option<usize>) -> Result<(), String> {
    let config = engine_model::qwen3::Config::open(&root.join("config.json")).map_err(|e| e.to_string())?;
    let sized = shared::Sizing { context_tokens: Some(context), gpu_bytes, ..Default::default() }
        .resolve(&config, backend.available_memory_bytes()?, None)?;
    backend.set_allocation_budget(sized.gpu_bytes)
}
