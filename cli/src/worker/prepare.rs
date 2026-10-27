//! Install preparation under the server operation lock, without a resident owner.
use super::{Fields, Store, check_hash, expanded_path, fields, gpu_empty, persistent_cuda_cache, required, stale_line};
use std::{path::Path, time::{Duration, Instant}};

/// Check every checkpoint file into the local launcher's private receipt and
/// stamp cache, which the launcher names in this helper's environment. An
/// existing receipt is a strict expectation; without one, the hashes become
/// the receipt. Only NVFP4 checkpoints keep receipts outside the checkpoint,
/// so other formats are left to the engine's own startup checks.
fn verify_local(model: &Path, worker: &Path) -> Result<Fields, String> {
    let model = model.canonicalize()
        .map_err(|_| "Cannot read the checkpoint; run vade doctor before retrying.")?;
    let state = worker.parent().ok_or("Local verification needs a private state folder.")?;
    let manifest = state.join("model-manifest.sha256");
    let cache = state.join("cache");
    if manifest.starts_with(&model) || cache.starts_with(&model)
        || std::env::var_os("FLASH_NVFP4_MANIFEST").as_deref() != Some(manifest.as_os_str())
        || std::env::var_os("XDG_CACHE_HOME").as_deref() != Some(cache.as_os_str())
    {
        return Err("Checkpoint verification needs private receipt and cache paths outside the model; choose a separate VADE_HOME and retry.".into());
    }
    crate::text::private_dir(&cache)?;
    crate::text::reject_symlinks(&manifest)?;
    let plan = engine_model::flash::manifest::ReceiptPlan::open(&model).map_err(|e| e.to_string())?;
    if plan.format != engine_model::flash::WeightFormat::Nvfp4 { return Ok(Fields::new()); }
    let began = Instant::now();
    let receipt = match plan.cached().map_err(|e| e.to_string())? {
        Some(receipt) => receipt,
        None => plan.finish().map_err(|e| format!("Checkpoint check failed: {e}. Restore the file from the publisher, then retry."))?,
    };
    let listed = verify_listed(&model, &manifest)?;
    Ok(fields(&[("model_verify_ms", &format!("{:.3}", began.elapsed().as_secs_f64() * 1000.0)),
        ("model_hashed_bytes", &(receipt.hashed_bytes + listed).to_string()),
        ("model_reused_files", &receipt.reused_files.to_string())]))
}

/// The receipt covers the index and the shards it names. Hash every other
/// supplied entry too (config, tokenizer and template files), so a checked
/// checkpoint means every listed file matched. Returns the bytes hashed.
fn verify_listed(model: &Path, manifest: &Path) -> Result<u64, String> {
    const INDEX: &str = "model.safetensors.index.json";
    let failed = |name: &str, what: &str| format!("Checkpoint check failed: {what}: {name}. Restore the file from the publisher, then retry.");
    let rows = std::fs::read_to_string(manifest).map_err(|_| "Cannot read the private checkpoint receipt; retry.")?;
    let covered = match std::fs::read_to_string(model.join(INDEX)) {
        Ok(index) => super::model::indexed_shards(&index)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ["model.safetensors".to_owned()].into(),
        Err(_) => return Err(failed(INDEX, "cannot read")),
    };
    let mut bytes = 0u64;
    for row in rows.lines().filter(|row| !row.is_empty()) {
        let (Some(expected), Some(name)) = (row.get(..64), row.get(66..)) else { return Err(failed(row, "invalid checksum row")) };
        if name == INDEX || covered.contains(name) { continue; }
        let file = model.join(name).canonicalize().ok().filter(|path| path.starts_with(model))
            .and_then(|path| std::fs::File::open(path).ok()).ok_or_else(|| failed(name, "cannot read"))?;
        let size = file.metadata().map_err(|_| failed(name, "cannot read"))?.len();
        if !crate::sha256::digest_reader(file).map_err(|_| failed(name, "cannot read"))?.eq_ignore_ascii_case(expected) {
            return Err(failed(name, "SHA-256 mismatch"));
        }
        bytes = bytes.saturating_add(size);
    }
    Ok(bytes)
}

pub(super) fn run(request: &Fields) -> Result<Fields, String> {
    // Serialize with every server operation, without publishing a server owner.
    // This scoped compiler job uses the runner's timeout and Linux parent-death
    // signal, so loss of the SSH helper cannot leave a CUDA process behind.
    let mut ownership = request.clone();
    ownership.insert("token".into(), crate::text::random_hex(32)?);
    let store = Store::lock(&ownership)?;
    store.ensure_available()?;
    let engine = expanded_path(required(request, "engine")?)?;
    let ptx = expanded_path(required(request, "ptx_path")?)?;
    check_hash(&engine, required(request, "binary_sha256")?)?;
    let ptx_hash = required(request, "ptx_sha256")?;
    check_hash(&ptx, ptx_hash)?;
    let model = request.get("model_path").map(|path| expanded_path(path)).transpose()?;
    let verify = request.get("verify_model").map(|path| expanded_path(path)).transpose()?;
    let exec = ptx_warmup_command(&store.root, &engine, &ptx, ptx_hash, model.as_deref())?;
    let stale = gpu_empty()?;
    let began = Instant::now();
    use crate::runner::Runner;
    // Compilation is GPU and CPU work; hashing is disk work. Run them together.
    let (output, verified) = std::thread::scope(|scope| {
        let worker = verify.as_deref().map(|model| scope.spawn(|| verify_local(model, &store.root)));
        let output = crate::runner::SystemRunner { dry: false }.run(&exec);
        (output, worker.map(|w| w.join().unwrap_or_else(|_| Err("Checkpoint verification stopped unexpectedly.".into()))))
    });
    let output = output?;
    let verified = verified.transpose()?;
    if !output.success {
        return Err("Install preparation failed. Inspect the retained command logs, then retry install.".into());
    }
    let mut reply = fields(&[("ptx_cache", "warm"), ("elapsed_ms", &format!("{:.3}", began.elapsed().as_secs_f64() * 1000.0))]);
    if let Some(v) = stale { reply.insert("gpu_note".into(), stale_line(v)); }
    let mut timings = crate::text::parse(&output.stdout)?;
    timings.extend(verified.unwrap_or_default());
    for key in ["ptx_warmup_ms", "model_verify_ms", "model_hashed_bytes", "model_reused_files"] {
        if let Some(value) = timings.get(key).filter(|value| value.parse::<f64>().is_ok_and(|n| n.is_finite() && n >= 0.0)) {
            reply.insert(key.into(), value.clone());
        }
    }
    Ok(reply)
}

fn ptx_warmup_command(root: &Path, engine: &Path, ptx: &Path, ptx_hash: &str, model: Option<&Path>) -> Result<crate::runner::Exec, String> {
    let mut environment = Fields::new();
    persistent_cuda_cache(root, ptx_hash, &mut environment)?;
    let mut exec = crate::runner::Exec::new("PTX cache warm-up", &engine.to_string_lossy(),
        vec!["--warm-ptx".into(), ptx.to_string_lossy().into_owned()]);
    if let Some(model) = model { exec.args.push(model.to_string_lossy().into_owned()); }
    exec.timeout = Duration::from_secs(300);
    // A cache preparation has no model options or profiling outputs. Preserve
    // the system CUDA installation, but discard inherited engine experiments.
    exec.unset = std::env::vars_os().filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| ["FLASH_", "SPARK_", "TP_", "SEQPAR_", "VADE_"]
            .iter().any(|prefix| name.starts_with(prefix))).collect();
    for (key, value) in environment {
        if let Some(name) = key.strip_prefix("env.") { exec.env.insert(name.into(), value); }
        else if let Some(name) = key.strip_prefix("env_unset.") { exec.unset.push(name.into()); }
    }
    Ok(exec)
}

#[cfg(test)]
mod tests {
    use super::{Fields, fields, persistent_cuda_cache, ptx_warmup_command};
    use crate::worker::handle;
    use std::{fs, path::Path};

    #[test]
    fn checkpoint_check_refuses_receipts_outside_the_private_state() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-verify-local-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root.join("worker")).unwrap();
        fs::create_dir_all(root.join("model")).unwrap();
        // Without the launcher's private paths in its environment, the helper
        // reads and writes nothing, so no receipt can land in the checkpoint.
        let error = super::verify_local(&root.join("model"), &root.join("worker")).unwrap_err();
        assert!(error.contains("private receipt"), "{error}");
        assert!(!root.join("cache").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_check_hashes_every_listed_file_beyond_the_shards() {
        let model = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-verify-listed-{}", crate::text::random_hex(8).unwrap()));
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("model.safetensors.index.json"), r#"{"metadata":{},"weight_map":{"fixture":"model.safetensors"}}"#).unwrap();
        fs::write(model.join("model.safetensors"), [0; 32]).unwrap();
        fs::write(model.join("tokenizer.json"), "{\"model\":1}").unwrap();
        let manifest = model.join("manifest.sha256");
        let hash = |name: &str| crate::sha256::digest_reader(fs::File::open(model.join(name)).unwrap()).unwrap();
        // The shard row is the receipt plan's to check, so this helper skips it.
        fs::write(&manifest, format!("{}  model.safetensors.index.json\n{}  model.safetensors\n{}  tokenizer.json\n",
            hash("model.safetensors.index.json"), "0".repeat(64), hash("tokenizer.json"))).unwrap();
        assert_eq!(super::verify_listed(&model, &manifest).unwrap(), 11);
        // Same length, valid JSON, different bytes: install must refuse it.
        fs::write(model.join("tokenizer.json"), "{\"model\":2}").unwrap();
        let error = super::verify_listed(&model, &manifest).unwrap_err();
        assert!(error.contains("SHA-256 mismatch: tokenizer.json"), "{error}");
        fs::remove_file(model.join("tokenizer.json")).unwrap();
        assert!(super::verify_listed(&model, &manifest).unwrap_err().contains("cannot read: tokenizer.json"));
        fs::remove_dir_all(model).unwrap();
    }

    #[test]
    fn existing_paths_operation_advertises_preparation_without_accessing_files() {
        let response = handle(&fields(&[("op", "paths"), ("ptx_path", "/missing/flash.ptx"),
            ("model_path", "/missing/model")])).unwrap();
        assert_eq!(response["ptx_path"], "/missing/flash.ptx");
        assert_eq!(response["model_path"], "/missing/model");
        assert_eq!(response["install_prepare"], "warm-ptx-v1");
    }

    #[test]
    fn cache_warmup_reuses_startup_cache_and_refuses_an_owned_server() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-warm-ptx-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let hash = "a".repeat(64);
        let mut environment = Fields::new();
        persistent_cuda_cache(&root, &hash, &mut environment).unwrap();
        let exec = ptx_warmup_command(&root, Path::new("/staged/flash-serve"), Path::new("/staged/flash.ptx"), &hash, None).unwrap();
        assert_eq!(exec.program, "/staged/flash-serve");
        assert_eq!(exec.args, ["--warm-ptx", "/staged/flash.ptx"]);
        assert_eq!(exec.env["CUDA_CACHE_PATH"], environment["env.CUDA_CACHE_PATH"]);
        assert_eq!(exec.env["CUDA_CACHE_DISABLE"], "0");
        assert!(exec.unset.contains(&"CUDA_FORCE_PTX_JIT".into()));
        assert!(exec.unset.contains(&"CUDA_DISABLE_PTX_JIT".into()));
        assert!(ptx_warmup_command(&root, Path::new("engine"), Path::new("ptx"), "invalid", None).is_err());
        let model = Path::new("/models/existing checkpoint");
        let exec = ptx_warmup_command(&root, Path::new("engine"), Path::new("ptx"), &hash, Some(model)).unwrap();
        assert_eq!(exec.args, ["--warm-ptx", "ptx", "/models/existing checkpoint"]);
        let request = fields(&[("op", "warm-ptx"), ("state_dir", root.to_str().unwrap())]);
        crate::text::atomic_write(&root.join("active.conf"), "reserved").unwrap();
        assert!(handle(&request).unwrap_err().contains("already reserved"));
        assert_eq!(fs::read_to_string(root.join("active.conf")).unwrap(), "reserved");
        fs::remove_dir_all(root).unwrap();
    }

}
