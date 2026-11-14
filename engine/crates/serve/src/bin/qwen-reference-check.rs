//! Compare real chat tokenization and teacher-forced logits with the one-time HF oracle.
#[cfg(any(target_os = "linux", test))]
fn logit_error(actual: f32, reference: f64) -> Result<f64, String> {
    let error = (actual as f64 - reference).abs();
    if !error.is_finite() || error > 1e-2 {
        return Err(format!("logit error {error} exceeds 0.01 (actual={actual}, reference={reference})"));
    }
    Ok(error)
}
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use engine_model::json::{self, Value, field};
    use spark_serve::{Tokenizer, protocol::ChatInput};
    use std::{ffi::CString, path::Path, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: qwen-reference-check PTX MODEL FIXTURE.json...".into());
    }
    let codec = engine_tokenizer::Tokenizer::from_directory(&args[1]).map_err(|e| e.to_string())?;
    let mut fixtures = Vec::new();
    for name in &args[2..] {
        let bytes = std::fs::read(name).map_err(|e| e.to_string())?;
        fixtures.push(json::parse(&bytes, json::Limits::default()).map_err(|e| e.to_string())?);
    }
    let get = |v: &Value, name: &str| -> Result<Value, String> {
        field(v.object().map_err(|e| e.to_string())?, name)
            .cloned()
            .map_err(|e| e.to_string())
    };
    let ids = |v: &Value| -> Result<Vec<u32>, String> {
        v.array()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|v| u32::try_from(v.u64().map_err(|e| e.to_string())?).map_err(|e| e.to_string()))
            .collect()
    };
    let mut capacity = 0;
    for fixture in &fixtures {
        capacity = capacity.max(
            ids(&get(fixture, "prompt_ids")?)?.len()
                + get(fixture, "rows")?
                    .array()
                    .map_err(|e| e.to_string())?
                    .len(),
        );
    }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend_mode(
        0,
        &ptx,
        20 * 1024 * 1024 * 1024,
        spark_engine::configured_matvec_mode()?,
        |backend| {
            let load_start = Instant::now();
            let before_load = backend.stats();
            let model = spark_engine::qwen::Model::load(backend, Path::new(&args[1]), capacity)?;
            let weight_bytes = backend.stats().since(before_load).weight_bytes;
            let model_bytes = backend.allocated_bytes();
            println!("load_seconds={:.3}", load_start.elapsed().as_secs_f64());
            let mut failures = 0;
            for (fixture, filename) in fixtures.iter().zip(&args[2..]) {
                let name = get(fixture, "name")?;
                let name = name.string().map_err(|e| e.to_string())?;
                let prompt = ids(&get(fixture, "prompt_ids")?)?;
                let messages = get(fixture, "messages")?;
                let input = ChatInput {
                    messages: messages
                        .array()
                        .map_err(|e| e.to_string())?
                        .iter()
                        .map(spark_serve::protocol::stringify)
                        .collect(),
                    tools: Vec::new(),
                    enable_thinking: false, thinking_off_requested: false,
                    reasoning_effort: Default::default(),
                    tool_choice: Default::default(), parallel_tool_calls: true,
                };
                let formatted = spark_serve::tokenizer::format_chat(&input, 1 << 20)?;
                if formatted
                    != get(fixture, "formatted")?
                        .string()
                        .map_err(|e| e.to_string())?
                {
                    return Err(format!("{name}: chat template differs from HF"));
                }
                let actual_ids = codec.encode_chat(&input, 1 << 20)?;
                if actual_ids != prompt {
                    let first = actual_ids.iter().zip(&prompt).position(|(a, b)| a != b);
                    return Err(format!(
                        "{name}: tokenizer differs first={first:?} actual={} expected={}",
                        actual_ids.len(),
                        prompt.len()
                    ));
                }
                let rows = get(fixture, "rows")?;
                let rows = rows.array().map_err(|e| e.to_string())?;
                if rows.is_empty() { return Err("fixture has no reference rows".into()); }
                let mut session = model.session();
                let prefill = Instant::now();
                let mut logits = session.prefill(&prompt)?;
                let ttft_ms = prefill.elapsed().as_secs_f64() * 1000.0;
                let decode = Instant::now();
                let mut agreement = 0;
                let mut full_max = 0f64;
                let mut top_max = 0f64;
                let mut full_rows = 0;
                let mut generated = Vec::new();
                for (index, row) in rows.iter().enumerate() {
                    let token = get(row, "token")?.u64().map_err(|e| e.to_string())? as u32;
                    let actual = spark_engine::greedy_token(&logits)?;
                    let top_ids = ids(&get(row, "top_ids")?)?;
                    let top_values = get(row, "top_logits")?;
                    let top_values = top_values.array().map_err(|e| e.to_string())?;
                    if top_ids.len() != top_values.len() { return Err("reference top-k width mismatch".into()); }
                    let mut row_max = 0f64;
                    for (&id, value) in top_ids.iter().zip(top_values) {
                        row_max = row_max.max(logit_error(
                            *logits.get(id as usize).ok_or("reference token outside vocabulary")?,
                            value.f64().map_err(|e| e.to_string())?,
                        )?);
                    }
                    top_max = top_max.max(row_max);
                    if let Some(path) = row.object().map_err(|e| e.to_string())?.get("full_logits")
                    {
                        let path = Path::new(filename)
                            .parent()
                            .unwrap()
                            .join(path.string().map_err(|e| e.to_string())?);
                        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
                        if bytes.len() != logits.len() * 4 {
                            return Err("reference logit width mismatch".into());
                        }
                        for (bytes, &logit) in bytes.as_chunks::<4>().0.iter().zip(&logits) {
                            let reference = f32::from_le_bytes(*bytes);
                            full_max = full_max.max(logit_error(logit, reference as f64)?);
                        }
                        full_rows += 1;
                    }
                    if token == actual {
                        agreement += 1;
                    } else {
                        failures += 1;
                    }
                    println!(
                        "fixture={name} position={} reference={token} actual={actual} agree={} top20_max_abs={row_max:.7}",
                        prompt.len() + index,
                        token == actual
                    );
                    generated.push(actual);
                    // Keep each comparison at the exact same reference prefix.
                    if index + 1 < rows.len() {
                        logits = session.step(token)?;
                    }
                }
                let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;
                let ms_per_token = if rows.len() > 1 { decode_ms / (rows.len() - 1) as f64 } else { 0.0 };
                println!(
                    "fixture={name} tokenizer=exact prompt_tokens={} top1={agreement}/{} full_rows={full_rows} full_vocab_max_abs={full_max:.7} top20_max_abs={top_max:.7} ttft_ms={ttft_ms:.3} decode_ms_per_token={:.3} decode_tokens_per_second={:.3}",
                    prompt.len(),
                    rows.len(),
                    ms_per_token,
                    if ms_per_token > 0.0 { 1000.0 / ms_per_token } else { 0.0 }
                );
                println!(
                    "fixture={name} continuation={:?}",
                    codec.decode(&generated, false).map_err(|e| e.to_string())?
                );
                let memory = std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?;
                let gpu_kv_bytes = session.cache_bytes();
                println!("fixture={name} gpu_weights_bytes={weight_bytes} model_aux_bytes={} kv_capacity_bytes={} session_scratch_bytes={} gpu_total_bytes={} host_peak={}",
                    model_bytes as u128 - weight_bytes, session.cache_bytes(),
                    backend.allocated_bytes() - model_bytes - gpu_kv_bytes, backend.allocated_bytes(),
                    memory.lines().find(|s| s.starts_with("VmHWM:")).unwrap_or("unavailable"));
            }
            if failures > 0 {
                return Err(format!("{failures} top-1 mismatches against HF"));
            }
            println!(
                "PASS all compared top-1 tokens agree with HF; every saved logit differs by at most 0.01"
            );
            Ok(())
        },
    )
}
#[test]
fn enforces_logit_tolerance() {
    assert!(logit_error(0., 0.01).is_ok());
    assert!(logit_error(0., 0.010001).is_err());
    assert!(logit_error(f32::NAN, 0.).is_err());
    assert!(logit_error(0., f64::INFINITY).is_err());
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(error) = run() {
        eprintln!("FAIL {error}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("reference comparison requires Linux CUDA driver");
        std::process::exit(1);
    }
}
