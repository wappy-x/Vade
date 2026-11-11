//! Actual-checkpoint bit equality of head skipping within decode and prompt phases.
#[cfg(target_os = "linux")]
pub(super) fn run(args: &[String], gpu_budget: Option<usize>) -> Result<(), String> {
    use std::{ffi::CString, path::Path};
    if args.len() != 2 { return Err("usage: prefill-check PTX CHECKPOINT_ROOT (fixture IDs 1,42,1024,17)".into()); }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    fn same(a: &[f32], b: &[f32], label: &str, row: usize) -> Result<(), String> {
        if a.len() != b.len() || a.is_empty() { return Err(format!("{label}: row={row} invalid logit widths a={} b={}", a.len(), b.len())); }
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            if a.to_bits() != b.to_bits() || !a.is_finite() || !b.is_finite() {
                return Err(format!("{label}: row={row} index={i} a={a:?} a_bits=0x{:08x} a_nan={} a_inf={} b={b:?} b_bits=0x{:08x} b_nan={} b_inf={}",
                    a.to_bits(), a.is_nan(), a.is_infinite(), b.to_bits(), b.is_nan(), b.is_infinite()));
            }
        }
        println!("parity={label} row={row} logits={} bit_identical=true finite=true", a.len());
        Ok(())
    }
    let (vocabulary, generated) = spark_gpu_probe::driver::with_backend_mode(0, &ptx,
        gpu_budget.unwrap_or(0), spark_engine::configured_matvec_mode()?, |backend| {
        super::budget::configure(backend, Path::new(&args[1]), 6, gpu_budget)?;
        let mut model = spark_engine::qwen::Model::load(backend, Path::new(&args[1]), 6)?;
        model.prepare_batch_weights()?;
        let mut reference = model.session();
        for token in [1, 42, 1024] { reference.step(token)?; }
        let mut goldens = vec![reference.step(17)?];
        let mut generated = Vec::new();
        for _ in 0..2 {
            let token = spark_engine::greedy_token(goldens.last().ok_or("missing golden")?)?;
            generated.push(token);
            goldens.push(reference.step(token)?);
        }
        let mut candidate = model.session();
        for token in [1, 42, 1024] { candidate.advance(token)?; }
        if candidate.len() != 3 { return Err("advance did not commit prompt KV".into()); }
        let checkpoint = candidate.checkpoint();
        if candidate.advance(u32::MAX).is_ok() || candidate.len() != checkpoint {
            return Err("invalid advance mutated logical state".into());
        }
        let mut logits = candidate.step(17)?;
        same(&logits, &goldens[0], "sequential final prompt", 3)?;
        for (i, &expected) in generated.iter().enumerate() {
            let token = spark_engine::greedy_token(&logits)?;
            if token != expected { return Err("generated sequence differs".into()); }
            logits = candidate.step(token)?;
            same(&logits, &goldens[i + 1], "sequential continuation", i + 4)?;
        }
        let retained = candidate.cache_bytes();
        if candidate.len() != 6 || candidate.advance(1).is_ok() || candidate.len() != 6
            || candidate.cache_bytes() != retained { return Err("capacity rejection mutated state".into()); }
        candidate.reset();
        if !candidate.is_empty() || candidate.cache_bytes() != retained { return Err("reset accounting differs".into()); }
        for tokens in [&[][..], &[1, u32::MAX][..], &[1; 7][..]] {
            if candidate.forward_batch(tokens).is_ok() || !candidate.is_empty() {
                return Err("invalid batch partially committed".into());
            }
        }
        // Prompt All/Last/None share prompt order; sequential goldens use decode order.
        let prompt = [1, 42, 1024, 17];
        let rows = candidate.forward_batch(&prompt)?;
        if rows.len() != 4 { return Err("batch row count differs".into()); }
        let next = spark_engine::greedy_token(&rows[3])?;
        let next_kept = candidate.step(next)?;
        candidate.reset();
        same(&candidate.prefill(&prompt)?, &rows[3], "prompt Last vs All", 3)?;
        let greedy = candidate.step_greedy(next)?;
        if greedy != spark_engine::greedy_token(&next_kept)? { return Err("device argmax differs".into()); }
        candidate.reset();
        candidate.prefill_advance(&prompt)?;
        same(&candidate.step(next)?, &next_kept, "next decode after prompt None vs All", 4)?;
        candidate.reset();
        candidate.prefill_advance(&prompt[..3])?;
        let skipped = candidate.prefill(&prompt[3..])?;
        same(&skipped, &rows[3], "skipped prefix vs prompt All", 3)?;
        candidate.reset();
        candidate.forward_batch(&prompt[..3])?;
        same(&candidate.prefill(&prompt[3..])?, &skipped, "kept vs skipped prefix", 3)?;
        Ok((model.config().vocab_size, generated))
    })?;
    println!("PASS actual_checkpoint_prefill logit_comparisons=7 logits_per_comparison={vocabulary} skipped_prompt_heads=3 generated={generated:?} sequential=bit_identical prompt_heads=bit_identical next_decode=bit_identical argmax=equal invalid_inputs=unchanged reset=retained cleanup=checked");
    Ok(())
}
