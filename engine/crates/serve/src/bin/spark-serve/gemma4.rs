//! Default admission for the dense text cartridge, using its actual cache layout.
use spark_serve::{Engine, Limits, budget::Sizing, cache::CacheLimits, scheduler};
use spark_gpu_probe::driver::Backend;
use spark_gpu_probe::admission::{refusal, CUDA, SERVE};
use std::{net::TcpListener, path::Path, sync::atomic::{AtomicBool, Ordering}};
pub fn serve(b: &Backend<'_>, checkpoint: &Path, listener: &TcpListener, tokenizer: &mut spark_serve::tokenizer::ModelTokenizer,
    limits: &mut Limits, sizing: &Sizing, output_bytes_explicit: bool, sharing: scheduler::Sharing, prefix_cache: bool,
    assistant: Option<&Path>, draft_mode: spark_serve::protocol::DraftMode, draft_k: Option<usize>, shutdown: &AtomicBool, admin: &mut scheduler::ModelAdmin) -> Result<(), String> {
    let c = engine_model::cartridge::Cartridge::preflight(checkpoint).map_err(|e| e.to_string())?.gemma4.ok_or("expected Gemma text")?;
    let assistant_bytes = assistant.map(spark_engine::gemma4::Assistant::open_config).transpose()?
        .map_or(0, |a| a.weight_bytes() as usize);
    let available = b.available_memory_bytes()?;
    let gpu = sizing.gpu_bytes.unwrap_or(available.saturating_sub((available/10).max(1024*1024*1024)));
    let weights = c.weight_bytes().map_err(|e| e.to_string())? as usize;
    let plan = |tokens: usize| -> Result<usize, String> {
        let fixed = weights + assistant_bytes + c.rope_bytes(tokens as u64).map_err(|e| e.to_string())? as usize + 64*1024*1024;
        let error = |required, capacity, reason| refusal(required, capacity, &format!("{reason}; weights/rank {} bytes; fixed weights/RoPE/loader {fixed} bytes; CONTEXT={tokens}; OUTPUT={}", weights + assistant_bytes, limits.output_tokens), &format!("{CUDA}; available input {available} bytes, capped by GPU ceiling {gpu} bytes (--gpu-budget-mib or 10% reserve, at least 1 GiB), session and KV ceilings"), SERVE);
        let budget = gpu.checked_sub(fixed).ok_or_else(|| error(fixed as u128, gpu, "weights and fixed allocations"))?;
        let budget = sizing.session_bytes.unwrap_or(budget).min(budget);
        let reservation = spark_engine::gemma4::Model::session_bytes(&c, tokens, tokens)? + if assistant.is_some() {32*1024*1024} else {0};
        if reservation > budget/3 { return Err(error(reservation as u128 * 3, budget, &format!("per-session {reservation} bytes x 2 sessions + 1 retained cache slot"))); }
        if sizing.kv_bytes.is_some_and(|n| c.kv_bytes(tokens as u64).map_or(true, |bytes| bytes as usize > n)) {
            return Err(error(c.kv_bytes(tokens as u64).map_err(|e| e.to_string())? as u128, sizing.kv_bytes.unwrap(), "KV payload/session x 1 session (--kv-budget-mib)"));
        }
        Ok(budget)
    };
    let tokens = if let Some(tokens) = sizing.context_tokens {
        if tokens == 0 || tokens as u64 > c.max_positions { return Err(spark_gpu_probe::admission::token_refusal(tokens as u128, c.max_positions as usize, &format!("CONTEXT={tokens}; OUTPUT={}; context must be positive", limits.output_tokens), "the checkpoint position ceiling", SERVE)); }
        plan(tokens)?; tokens
    } else {
        let (mut low, mut high) = (1usize, (c.max_positions as usize).min(32768));
        plan(low)?;
        while low < high { let mid = low+(high-low).div_ceil(2); if plan(mid).is_ok() {low=mid} else {high=mid-1} }
        low
    };
    let budget = plan(tokens)?;
    let active = sizing.max_active.unwrap_or(32);
    if !(2..=32).contains(&active) { return Err("--max-active must be in 2..32".into()); }
    b.set_allocation_budget(gpu)?;
    b.configure_kv_reserve(available.saturating_sub(gpu).max(1024*1024*1024));
    limits.context_tokens = tokens;
    limits.input_tokens = limits.input_tokens.min(tokens.saturating_sub(1).max(1));
    limits.output_tokens = limits.output_tokens.min(tokens.saturating_sub(1).max(1));
    limits.body_bytes = limits.body_bytes.max(limits.input_tokens.saturating_mul(16)).min(4 * 1024 * 1024);
    if !output_bytes_explicit {
        limits.output_bytes = limits.output_bytes.max(limits.output_tokens.saturating_mul(256)
            .saturating_add(4096)).min(16 * 1024 * 1024);
    }
    limits.validate()?;
    tokenizer.set_limits(limits.tokenizer_limits()).map_err(|e| e.to_string())?;
    let mut model = spark_engine::gemma4::Model::load(b, checkpoint, tokens)?;
    if let Some(path) = assistant {
        model.load_assistant(path)?; model.set_assistant_width(draft_k.unwrap_or(4))?;
        eprintln!("gemma_assistant draft_k={} verification=strict concurrent_chats=plain", draft_k.unwrap_or(4));
    }
    let engine = spark_serve::gemma4::Gemma(&model);
    if let Some(metadata) = &limits.metadata { metadata.start(); }
    let retained = budget/3;
    let cache = CacheLimits { entries: 128, retained_bytes: retained, tokens_per_entry: tokens,
        logits_per_entry: engine.vocab_size(), idle_ttl: std::time::Duration::from_secs(300) };
    eprintln!("ready model={} context_tokens={tokens} device_bytes={} peak_device_bytes={} session_budget_bytes={budget} max_active={active} kv_precision=BF16", admin.model, b.allocated_bytes(), b.peak_allocated_bytes());
    let swap = scheduler::serve_model(listener, &engine, Some(tokenizer), limits, &cache,
        &scheduler::Config { active, connections: 32.max(active), prefill_chunk: 512,
            foreground_rows: 1, background_rows: 32, active_bytes: budget-retained, session_budget_bytes: budget,
            draft: if assistant.is_some() {draft_mode} else {spark_serve::protocol::DraftMode::Plain}, mtp_depth: 0, draft_global_bank: false, prefix_cache, sharing }, shutdown, admin)?;
    if swap.is_some() && !shutdown.load(Ordering::Relaxed) { return Err("Gemma model swap requires a fresh server launch".into()); }
    Ok(())
}
