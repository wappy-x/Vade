//! Keep one model warm for idle-peer sequence prefill jobs.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use spark_engine::qwen::Model;
    use spark_gpu_probe::driver::{with_backend_mode, MatvecMode};
    use std::{ffi::CString, fs, net::TcpListener, path::Path};
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 { return Err("usage: seqpar-worker PTX MODEL CONTROL_LISTEN MAX_TOKENS".into()); }
    let max_tokens = args[3].parse::<usize>().map_err(|_| "invalid MAX_TOKENS")?;
    let secret = spark_seqpar::control::control_secret()?;
    let listener = TcpListener::bind(&args[2]).map_err(|e| e.to_string())?;
    if !listener.local_addr().map_err(|e| e.to_string())?.ip().is_loopback() && secret.is_none() {
        return Err("non-loopback worker control requires SEQPAR_CONTROL_SECRET".into());
    }
    let memory_budget = spark_seqpar::memory_budget()?;
    let ptx = CString::new(fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    with_backend_mode(0, &ptx, memory_budget, MatvecMode::Warp32, |backend| {
        backend.use_vec8(false)?;
        let mut model = Model::load_with_kv_budget(backend, Path::new(&args[1]), max_tokens, memory_budget)?;
        model.prepare_batch_weights()?;
        eprintln!("WORKER ready max_tokens={max_tokens} model_bytes={} memory_budget_bytes={memory_budget}", backend.allocated_bytes());
        spark_seqpar::coordinator::serve_peer(&model, listener)
    })
}

fn main() {
    #[cfg(target_os = "linux")]
    if let Err(error) = run() { eprintln!("FAIL {error}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("requires Linux CUDA driver"); std::process::exit(1); }
}
