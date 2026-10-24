//! Install preparation under the server operation lock, without a resident owner.
use super::{Fields, Store, check_hash, expanded_path, fields, gpu_empty, persistent_cuda_cache, required};
use std::{path::Path, time::{Duration, Instant}};

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
    let exec = ptx_warmup_command(&store.root, &engine, &ptx, ptx_hash, model.as_deref())?;
    gpu_empty()?;
    let began = Instant::now();
    use crate::runner::Runner;
    let output = crate::runner::SystemRunner { dry: false }.run(&exec)?;
    if !output.success {
        return Err("Install preparation failed. Inspect the retained command logs, then retry install.".into());
    }
    let mut reply = fields(&[("ptx_cache", "warm"), ("elapsed_ms", &format!("{:.3}", began.elapsed().as_secs_f64() * 1000.0))]);
    let timings = crate::text::parse(&output.stdout)?;
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
