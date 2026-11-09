//! Actual-checkpoint cache equivalence gate, without a tokenizer or mock backend.
#[cfg(target_os = "linux")]
pub(super) fn run(args: &[String], gpu_budget: Option<usize>) -> Result<(), String> {
    use std::{ffi::CString, path::Path};
    if args.len() != 2 { return Err("usage: cache-check PTX CHECKPOINT_ROOT (fixture token IDs1,42,1024,17 require corresponding vocabulary)".into()); }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    fn same(a: &[f32], b: &[f32], label: &str) -> Result<(), String> {
        if a.len() != b.len() { return Err(format!("{label}: logit length mismatch")); }
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            if a.to_bits() != b.to_bits() { return Err(format!("{label}: logit{i} differs {a} vs {b}")); }
        }
        Ok(())
    }
    let vocabulary = spark_gpu_probe::driver::with_backend_mode(0, &ptx, gpu_budget.unwrap_or(0), spark_engine::configured_matvec_mode()?, |backend| {
        super::budget::configure(backend, Path::new(&args[1]), 1024, gpu_budget)?;
        let mut model = spark_engine::qwen::Model::load(backend, Path::new(&args[1]), 1024)?;
        model.prepare_batch_weights()?;
        let mut source = model.session();
        if source.forward_batch(&[1;513]).is_ok() || source.verify_batch(&[1;513]).is_ok() || !source.is_empty() {
            return Err("oversized batch committed state".into());
        }
        let mut goldens = vec![source.step(1)?, source.step(42)?];
        let position = source.checkpoint();
        let mut fork = source.fork()?;
        goldens.push(source.step(1024)?);
        goldens.push(source.step(17)?);
        let branch = fork.step(17)?;
        if source.len() != 4 || fork.len() != 3 { return Err("fork position aliasing".into()); }
        source.truncate(position)?;
        same(&source.step(17)?, &branch, "truncate branch replay")?;
        let retained = source.cache_bytes();
        source.reset();
        if !source.is_empty() || source.cache_bytes() != retained { return Err("reset position/capacity mismatch".into()); }
        for (&token, golden) in [1,42,1024,17].iter().zip(&goldens) { same(&source.step(token)?, golden, "reset full replay")?; }
        source.truncate(2)?;
        if source.step_with_observer(1024, |boundary, _| {
            if boundary == "layer-3" { Err("injected observer failure".into()) } else { Ok(()) }
        }).is_ok() || source.len() != 2 { return Err("observer failure committed KV".into()); }
        same(&source.step(1024)?, &goldens[2], "failed observation replay")?;
        let before = source.len();
        if source.truncate(before + 1).is_ok() || source.len() != before { return Err("invalid truncate mutated state".into()); }
        if source.step(u32::MAX).is_ok() || source.len() != before { return Err("invalid token mutated state".into()); }
        fork.reset();
        if source.len() != before { return Err("fork reset affected original".into()); }
        if source.cache_bytes() > source.max_cache_bytes() || fork.cache_bytes() > fork.max_cache_bytes() { return Err("KV budget exceeded".into()); }
        let vocabulary = model.config().vocab_size;
        let layers = model.config().layers as usize;
        drop(source); drop(fork); drop(model);
        let mut left = spark_engine::qwen::PartitionModel::load(backend, Path::new(&args[1]), 16, 0, layers / 2, true, false)?;
        let mut right = spark_engine::qwen::PartitionModel::load(backend, Path::new(&args[1]), 16, layers / 2, layers, false, true)?;
        left.prepare_batch_weights()?; right.prepare_batch_weights()?;
        let (mut left, mut right) = (left.session(), right.session());
        for (position, (&token, golden)) in [1, 42, 1024, 17].iter().zip(&goldens).enumerate() {
            let hidden = left.step_hidden(position, &left.embed(token)?)?;
            let hidden = right.step_hidden(position, &hidden)?;
            same(&right.logits(&hidden)?, golden, "partition composition")?;
        }
        Ok(vocabulary)
    })?;
    println!("PASS actual_checkpoint_cache exact_logit_comparisons=10 logits_per_comparison={vocabulary} fork=isolated truncate=replay reset=replay observer_failure=rollback partition=exact invalid_inputs=unchanged cleanup=checked");
    Ok(())
}
