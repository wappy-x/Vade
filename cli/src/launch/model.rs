//! Find existing checkpoints without reading weight payloads or changing them.
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs,
    io::{self, IsTerminal, Read, Write},
    os::fd::AsRawFd,
    path::{Component, Path, PathBuf},
    time::Duration,
};

#[cfg(test)]
use std::fs::File;

const DEFAULT_CHECKPOINT: &str = "Qwen3.8-Flash-Next-FP4-Local";
const MODEL_HELP: &str = "Use nvidia/Qwen3.8-Flash-Next-FP4-Local. Download to a checkpoint folder with: hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local";

#[derive(Clone, Debug)]
pub(super) struct Model {
    pub path: PathBuf,
    pub bytes: u64,
    pub context_limit: u32,
    pub fp4_local: bool,
}

pub(super) fn discover(home: &Path, saved: Option<&Path>) -> Result<Model, String> {
    discover_with(
        home,
        saved,
        std::env::var_os("VADE_MODELS").as_deref(),
        |reason| {
            check_interrupted()?;
            if !io::stdin().is_terminal() {
                return Err(format!(
                    "{reason}\n{MODEL_HELP}\nSet VADE_MODELS to the checkpoint folder, then run vade launch."
                ));
            }
            crate::output::line(format_args!("{reason}"))?;
            crate::output::line(format_args!("{MODEL_HELP}"))?;
            crate::output::text(format_args!("FP4-Local model folder (existing weights): "))?;
            io::stdout()
                .flush()
                .map_err(|_| "Cannot display the model folder prompt.")?;
            let answer = read_prompt_line()?;
            let answer = answer.trim();
            if answer.is_empty() || answer.len() > 4096 {
                return Err("Give one existing model folder; nothing was installed.".into());
            }
            Ok(PathBuf::from(answer))
        },
    )
}

pub(super) fn discover_existing(home: &Path, saved: Option<&Path>) -> Result<Model, String> {
    discover_with(home, saved, std::env::var_os("VADE_MODELS").as_deref(), |reason| {
        Err(format!("{reason}\n{MODEL_HELP}\nSet VADE_MODELS to the checkpoint folder, then run vade launch."))
    })
}

fn check_interrupted() -> Result<(), String> {
    if crate::runner::interrupted() {
        Err("Model selection interrupted; nothing was installed.".into())
    } else {
        Ok(())
    }
}

unsafe extern "C" {
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn read(fd: i32, buffer: *mut std::ffi::c_void, length: usize) -> isize;
}

struct PromptInput {
    fd: i32,
    flags: i32,
}
impl Drop for PromptInput {
    fn drop(&mut self) {
        // stdin belongs to the caller and subsequent interactive chat needs
        // its original flags, including on cancellation and invalid input.
        unsafe { fcntl(self.fd, crate::unix_flags::F_SETFL, self.flags) };
    }
}

// Read one bounded line without buffering any following chat input. A signal
// handler can be restarted by the OS, so every read itself must be nonblocking.
fn read_prompt_line() -> Result<String, String> {
    use crate::unix_flags::{F_GETFL, F_SETFL, O_NONBLOCK};
    let input = io::stdin().lock();
    let fd = input.as_raw_fd();
    let flags = unsafe { fcntl(fd, F_GETFL) };
    if flags < 0 || unsafe { fcntl(fd, F_SETFL, flags | O_NONBLOCK) } < 0 {
        return Err("Cannot prepare the model folder prompt.".into());
    }
    let _restore = PromptInput { fd, flags };
    let mut bytes = Vec::new();
    while bytes.len() < 4097 {
        check_interrupted()?;
        let mut byte = 0u8;
        // Reading directly avoids StdinLock's buffer consuming a later line.
        let count = unsafe { read(fd, (&mut byte as *mut u8).cast(), 1) };
        if count < 0 {
            match io::Error::last_os_error().kind() {
                io::ErrorKind::Interrupted => continue,
                io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(25));
                    continue;
                }
                _ => return Err("Cannot read the model folder.".into()),
            }
        }
        if count == 0 || byte == b'\n' {
            break;
        }
        bytes.push(byte);
    }
    check_interrupted()?;
    String::from_utf8(bytes).map_err(|_| "Use a model folder path in valid text.".into())
}

#[cfg(test)]
#[path = "model_prompt_tests.rs"]
mod prompt_tests;

fn discover_with(
    home: &Path,
    saved: Option<&Path>,
    environment: Option<&OsStr>,
    mut prompt: impl FnMut(&str) -> Result<PathBuf, String>,
) -> Result<Model, String> {
    // Keep a valid saved choice, but let owners recover from an incompatible
    // first selection through VADE_MODELS or the same first-run prompt.
    let mut reason = match saved {
        Some(path) => match inspect(path) {
            Ok(model) => return Ok(model),
            Err(reason) => reason,
        },
        None => "No FP4-Local checkpoint has been selected.".into(),
    };
    if let Some(path) = environment.filter(|path| !path.is_empty()) {
        let path = expand(home, Path::new(path))?;
        match locate(&path) {
            Ok(model) => return Ok(model),
            Err(error) => reason = error,
        }
    } else if saved.is_none() {
        let models = home.join("models");
        let preferred = [models.join("nvidia").join(DEFAULT_CHECKPOINT), models.join(DEFAULT_CHECKPOINT)]
            .into_iter().find(|path| path.exists());
        match locate(preferred.as_deref().unwrap_or(&models)) {
            Ok(model) => return Ok(model),
            Err(error) => reason = error,
        }
    }
    let path = expand(home, &prompt(&reason)?)?;
    locate(&path)
}

fn expand(home: &Path, path: &Path) -> Result<PathBuf, String> {
    let value = path
        .to_str()
        .ok_or("Use a model folder path in valid text.")?;
    if value.chars().any(char::is_control) {
        return Err("Use a model folder path without control characters.".into());
    }
    Ok(if value == "~" {
        home.to_owned()
    } else if let Some(suffix) = value.strip_prefix("~/") {
        home.join(suffix)
    } else {
        path.to_owned()
    })
}

fn locate(root: &Path) -> Result<Model, String> {
    let root = fs::canonicalize(root)
        .map_err(|_| format!("Model folder {} is unavailable.", display(root)))?;
    if !root.is_dir() {
        return Err(format!(
            "Model folder {} is not a directory.",
            display(&root)
        ));
    }
    let mut pending = vec![(root.clone(), 0usize, false)];
    let mut candidates = BTreeMap::new();
    let mut first_error = None;
    let mut entries_seen = 0usize;
    while let Some((directory, depth, linked_checkpoint)) = pending.pop() {
        if candidates.contains_key(&directory) {
            continue;
        }
        if directory.join("config.json").is_file() {
            match inspect(&directory) {
                Ok(model) => {
                    candidates.insert(model.path.clone(), model);
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
            if candidates.len() > 1 {
                return Err(format!(
                    "More than one model is available in {}; choose a checkpoint folder.",
                    display(&root)
                ));
            }
            continue;
        }
        if depth == 3 || linked_checkpoint {
            continue;
        }
        for entry in fs::read_dir(&directory)
            .map_err(|_| format!("Cannot inspect model folder {}.", display(&directory)))?
        {
            entries_seen += 1;
            if entries_seen > 10_000 {
                return Err(
                    "Model search is too large; choose the checkpoint folder directly.".into(),
                );
            }
            let entry = entry.map_err(|_| "Cannot inspect model folder entries.")?;
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let kind = entry
                .file_type()
                .map_err(|_| "Cannot inspect a model folder entry.")?;
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1, false));
            } else if kind.is_symlink() {
                // Owners commonly link checkpoints on another disk into ~/models.
                // Inspect direct checkpoints only; never recurse through symlink
                // containers. Canonical candidate keys deduplicate aliases.
                if let Ok(target) = fs::canonicalize(entry.path()) {
                    if target.is_dir() && target.join("config.json").is_file() {
                        pending.push((target, depth + 1, true));
                    }
                }
            }
        }
    }
    candidates
        .pop_first()
        .map(|(_, model)| model)
        .ok_or_else(|| {
            first_error
                .unwrap_or_else(|| format!("No complete model was found in {}.", display(&root)))
        })
}

fn inspect(path: &Path) -> Result<Model, String> {
    let path = fs::canonicalize(path)
        .map_err(|_| format!("Saved model folder {} is unavailable.", display(path)))?;
    if path.to_str().is_none() || path.to_string_lossy().chars().any(char::is_control) {
        return Err("Use a model folder path in valid text without control characters.".into());
    }
    let config = metadata(&path.join("config.json"), 1024 * 1024)?;
    if !crate::json::valid(&config) {
        return Err("Model config.json is not valid JSON.".into());
    }
    let layout = compatible(&path, &config)?;
    let limits = engine_model::flash::checkpoint_limits(layout.format, layout.mixed.is_some());
    // ReceiptPlan includes the index itself in its confined checkpoint entries.
    // Internal links and aliases of the whole checkpoint remain valid.
    let index_path = path.join("model.safetensors.index.json");
    let index_path = fs::canonicalize(&index_path).map_err(|_| {
        format!("Model metadata {} is missing or unreadable.", display(&index_path))
    })?;
    if !index_path.starts_with(&path) {
        return Err(format!("Model shard index symlink escapes the checkpoint folder. Hugging Face cache snapshots need a separate --local-dir download. {MODEL_HELP}; then run vade launch."));
    }
    let index = metadata(&index_path, limits.header_bytes as u64)?;
    let index = if layout.mixed.is_some() {
        engine_model::safetensors::Index::parse_mixed(index.as_bytes(), limits)
    } else {
        engine_model::safetensors::Index::parse(index.as_bytes(), limits)
    }
    .map_err(|error| {
        format!(
            "This engine cannot load the model shard index: {}.",
            error.to_string().replace(['\r', '\n'], " ")
        )
    })?;
    let shards = index.weight_map.into_values().collect::<BTreeSet<_>>();
    let mut bytes = 0u64;
    for shard in shards {
        if !shard.ends_with(".safetensors")
            || shard.contains('\\')
            || shard.chars().any(char::is_control)
            || !Path::new(&shard)
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err("Model shard index contains an unsafe filename.".into());
        }
        let payload = fs::canonicalize(path.join(&shard)).map_err(|_| {
            format!(
                "Model shard {} is missing or unreadable.",
                display(Path::new(&shard))
            )
        })?;
        if !payload.starts_with(&path) {
            return Err(format!("Model shard symlink escapes the checkpoint folder. Hugging Face cache snapshots need a separate --local-dir download. {MODEL_HELP}; then run vade launch."));
        }
        let meta = fs::metadata(&payload).map_err(|_| {
            format!(
                "Model shard {} is missing or unreadable.",
                display(Path::new(&shard))
            )
        })?;
        if !meta.is_file() || meta.len() == 0 {
            return Err(format!(
                "Model shard {} is empty or invalid.",
                display(Path::new(&shard))
            ));
        }
        bytes = bytes
            .checked_add(meta.len())
            .ok_or("Model size is too large.")?;
    }
    serving_metadata(&path, &layout)?;
    let mut limit = 32768.0;
    for keys in [
        &["text_config", "max_position_embeddings"][..],
        &["max_position_embeddings"][..],
    ] {
        if crate::json::path_exists(&config, keys) {
            limit = crate::json::path_number(&config, keys)
                .ok_or("Model context limit must be a number in config.json.")?;
            break;
        }
    }
    if limit < 64.0 || limit.fract() != 0.0 || limit > u32::MAX as f64 {
        return Err("Model context limit is invalid in config.json.".into());
    }
    Ok(Model {
        path,
        bytes,
        context_limit: limit as u32,
        fp4_local: layout.mixed.is_some(),
    })
}

// Use the engine's metadata parser so architecture, dimensions and quantization
// cannot drift from flash-serve's accepted formats. No weights are loaded here.
fn compatible(path: &Path, config: &str) -> Result<engine_model::flash::Config, String> {
    let parsed = engine_model::flash::Config::parse(config.as_bytes()).map_err(|error| {
        format!(
            "This engine cannot load this model: {}.",
            error.to_string().replace(['\r', '\n'], " ")
        )
    })?;
    if parsed.format == engine_model::flash::WeightFormat::Nvfp4 && parsed.mixed.is_none() {
        return Err("Local launch requires the mixed nvidia/Qwen3.8-Flash-Next-FP4-Local checkpoint. Set VADE_MODELS to that checkpoint, then run vade launch; legacy NVFP4 is unsupported here.".into());
    }
    // ReceiptPlan::open runs this same function at startup. In particular,
    // mixed FP4-Local does not bypass a present sidecar's full validation.
    let detected = engine_model::flash::nvfp4::detect(path).map_err(|error| {
        format!(
            "This engine cannot load the model quantization sidecar: {}.",
            error.to_string().replace(['\r', '\n'], " ")
        )
    })?;
    if detected != parsed.format {
        return Err(
            "This engine cannot load a model whose config.json and quantization sidecar disagree on weight format."
                .into(),
        );
    }
    Ok(parsed)
}

// flash-serve unconditionally opens the generation and tokenizer metadata
// below at startup, so discovery must refuse a checkpoint the engine would
// refuse after installation. The engine tree is frozen and its settings
// module is private to a GPU binary the CLI cannot depend on, so these
// mirrors reuse the engine's own CPU-only bounded reader, JSON parser and
// tokenizer crate, with the same limits, fields and acceptance order.
fn serving_metadata(path: &Path, layout: &engine_model::flash::Config) -> Result<(), String> {
    generation_config(path, layout)?;
    tokenizer_metadata(path)
}

fn serving_error(file: &str, error: impl std::fmt::Display) -> String {
    format!(
        "This engine cannot load the model {file}: {}.",
        error.to_string().replace(['\r', '\n'], " ")
    )
}

fn generation_config(path: &Path, layout: &engine_model::flash::Config) -> Result<(), String> {
    use engine_model::json::{self, Value};
    const FILE: &str = "generation_config.json";
    let bytes = engine_model::safetensors::read_bounded(&path.join(FILE), 1024 * 1024)
        .map_err(|e| serving_error(FILE, e))?;
    let value =
        json::parse(&bytes, json::Limits::default()).map_err(|e| serving_error(FILE, e))?;
    let object = value.object().map_err(|e| serving_error(FILE, e))?;
    // Engine defaults before checkpoint overrides: temperature 1.0 and
    // top_k 50 over greedy top_p/repetition 1.0. Presence and frequency
    // penalties keep their valid greedy defaults at startup.
    let (mut temperature, mut top_p, mut repetition_penalty) = (1.0f64, 1.0f64, 1.0f64);
    for (name, target) in [
        ("temperature", &mut temperature),
        ("top_p", &mut top_p),
        ("repetition_penalty", &mut repetition_penalty),
    ] {
        if let Some(value) = object.get(name) {
            *target = value.f64().map_err(|e| serving_error(FILE, e))?;
        }
    }
    if let Some(value) = object.get("top_k") {
        usize::try_from(value.u64().map_err(|e| serving_error(FILE, e))?)
            .map_err(|e| serving_error(FILE, e))?;
    }
    let sampled = object
        .get("do_sample")
        .map(Value::boolean)
        .transpose()
        .map_err(|e| serving_error(FILE, e))?
        .unwrap_or(false);
    if !sampled {
        temperature = 0.0;
    }
    // Sampling::validate bounds for the checkpoint-controlled fields.
    if !temperature.is_finite()
        || temperature < 0.0
        || !top_p.is_finite()
        || top_p <= 0.0
        || top_p > 1.0
        || !repetition_penalty.is_finite()
        || repetition_penalty <= 0.0
    {
        return Err(serving_error(FILE, "invalid sampling parameters"));
    }
    let eos = match object.get("eos_token_id") {
        None => vec![layout.eos_token as u32],
        Some(value) => {
            let values = match value {
                Value::Array(values) => values.clone(),
                _ => vec![value.clone()],
            };
            values
                .iter()
                .map(|value| {
                    value
                        .u64()
                        .map_err(|e| serving_error(FILE, e))
                        .and_then(|n| u32::try_from(n).map_err(|e| serving_error(FILE, e)))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let vocab = layout.vocab as usize;
    if eos.is_empty() || eos.iter().any(|&id| id as usize >= vocab) {
        return Err(serving_error(FILE, "invalid checkpoint EOS IDs"));
    }
    Ok(())
}

fn tokenizer_metadata(path: &Path) -> Result<(), String> {
    use engine_model::json::{self, Value};
    let read = |name: &str| {
        engine_model::safetensors::read_bounded(&path.join(name), 2 * 1024 * 1024)
            .map_err(|e| serving_error(name, e))
    };
    let bytes = read("tokenizer_config.json")?;
    let value = json::parse(
        &bytes,
        json::Limits {
            bytes: 2 * 1024 * 1024,
            ..Default::default()
        },
    )
    .map_err(|e| serving_error("tokenizer_config.json", e))?;
    let object = value
        .object()
        .map_err(|e| serving_error("tokenizer_config.json", e))?;
    let embedded = json::field(object, "chat_template")
        .and_then(Value::string)
        .map_err(|e| serving_error("tokenizer_config.json", e))?;
    if path
        .join("chat_template.jinja")
        .try_exists()
        .map_err(|e| serving_error("chat_template.jinja", e))?
    {
        let template = read("chat_template.jinja")?;
        let template =
            std::str::from_utf8(&template).map_err(|e| serving_error("chat_template.jinja", e))?;
        // The engine compares templates with trailing CR/LF trimmed.
        if embedded.trim_end_matches(['\r', '\n']) != template.trim_end_matches(['\r', '\n']) {
            return Err(
                "This engine cannot load the model chat_template.jinja: it differs from tokenizer_config.json."
                    .into(),
            );
        }
    }
    // The engine opens tokenizer.json directly; keep discovery from
    // blocking on a nonregular file it could never serve from.
    let file = fs::metadata(path.join("tokenizer.json"))
        .map_err(|_| "Model tokenizer.json is missing or unreadable.")?;
    if !file.is_file() {
        return Err("Model tokenizer.json must be a regular file.".into());
    }
    let mut tokenizer = engine_tokenizer::Tokenizer::from_directory(path)
        .map_err(|e| serving_error("tokenizer metadata", e))?;
    // flash-serve derives this bound from its 4 MiB body limit and the
    // tokenizer default token bound; set_limits only rejects zero bounds.
    tokenizer
        .set_limits(engine_tokenizer::Limits {
            max_input_bytes: 4 * 1024 * 1024,
            max_tokens: engine_tokenizer::Limits::default().max_tokens,
        })
        .map_err(|e| serving_error("tokenizer metadata", e))?;
    Ok(())
}

fn metadata(path: &Path, limit: u64) -> Result<String, String> {
    let file = super::open_regular(path).map_err(|error| {
        if error.kind() == io::ErrorKind::InvalidInput {
            "Model metadata must be a regular file.".into()
        } else {
            format!("Model metadata {} is missing or unreadable.", display(path))
        }
    })?;
    let mut data = String::new();
    file.take(limit + 1)
        .read_to_string(&mut data)
        .map_err(|_| format!("Cannot read model metadata {}.", display(path)))?;
    if data.len() as u64 > limit {
        return Err(format!(
            "Model metadata {} exceeds its size limit.",
            display(path)
        ));
    }
    Ok(data)
}

fn display(path: &Path) -> String {
    crate::text::quoted(&path.to_string_lossy())
}

#[cfg(test)]
pub(super) fn test_index() -> &'static str {
    r#"{"metadata":{},"weight_map":{"fixture":"model.safetensors"}}"#
}

#[cfg(test)]
pub(super) fn test_config(context: u32) -> String {
    let layers = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"model_type":"qwen4_exp","text_config":{{
        "model_type":"qwen4_exp_text","hidden_act":"silu","output_gate_type":"sigmoid",
        "mamba_ssm_dtype":"float32","attention_bias":false,"tie_word_embeddings":false,
        "mtp_use_dedicated_embeddings":false,"hidden_size":2560,"num_hidden_layers":48,
        "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_experts":512,
        "num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
        "hc_count":4,"hc_lowrank":320,"vocab_size":248320,"max_position_embeddings":{context},
        "eos_token_id":248044,"rms_norm_eps":0.000001,"partial_rotary_factor":0.25,
        "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,
        "indexer_compress_ratio":4,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,
        "ple_embed_dim":2560,"ple_conv_kernel_size":4,"split_ngram_parts":128,
        "mtp_num_hidden_layers":1,"layer_types":[{layers}],
        "rope_parameters":{{"rope_type":"default","mrope_interleaved":true,"rope_theta":10000000,
            "partial_rotary_factor":0.25,"mrope_section":[11,11,10]}},
        "mtp":{{"num_hidden_layers":1,"layer_types":["full_attention"],"hybrid":true,
            "mtp_use_hidden_state_from_layer":null,"rope_theta":10000000}}
        }},"quantization_config":{{"quant_method":"fp8","activation_scheme":"dynamic",
            "weight_per_tensor":false,"act_per_tensor":false,"weight_block_size":[128,128]}}}}"#
    )
}

#[cfg(test)]
pub(super) fn test_mixed_config(context: u32) -> String {
    test_config(context).replace(
        r#"{"quant_method":"fp8","activation_scheme":"dynamic",
            "weight_per_tensor":false,"act_per_tensor":false,"weight_block_size":[128,128]}"#,
        r#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,"ignore":[],"quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf"},"model.language_model.layers.0.self_attn.q_proj":{"quant_algo":"MXFP8","group_size":32}}}"#,
    )
}

#[cfg(test)]
pub(super) fn test_generation_config() -> &'static str {
    r#"{"eos_token_id":248044,"do_sample":false}"#
}

#[cfg(test)]
pub(super) fn test_chat_template() -> &'static str {
    include_str!("../../../engine/crates/tokenizer/templates/qwen3.jinja")
}

#[cfg(test)]
fn test_json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if !(0x20..0x7f).contains(&(c as u32)) => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
pub(super) fn test_tokenizer_config() -> String {
    format!(
        r#"{{"chat_template":{},"add_bos_token":false,"add_prefix_space":false,"clean_up_tokenization_spaces":false,"split_special_tokens":false,"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}}"#,
        test_json_string(test_chat_template())
    )
}

// Smallest tokenizer.json the engine tokenizer accepts: the full byte
// alphabet as vocab, no merges, and the mandatory special control tokens.
#[cfg(test)]
pub(super) fn test_tokenizer_json() -> String {
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let mut extra = 256u32;
    let vocab = (0u32..256)
        .map(|b| {
            let n = if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
                b
            } else {
                let n = extra;
                extra += 1;
                n
            };
            format!("{}:{}", test_json_string(&char::from_u32(n).unwrap().to_string()), b)
        })
        .collect::<Vec<_>>()
        .join(",");
    let bytelevel = r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"#;
    let added = [("<|endoftext|>", 151643), ("<|im_start|>", 151644), ("<|im_end|>", 151645)]
        .iter()
        .map(|(text, id)| format!(r#"{{"id":{id},"content":{},"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}"#, test_json_string(text)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{added}],"normalizer":{{"type":"NFC"}},"pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":{}}},"behavior":"Isolated","invert":false}},{bytelevel}]}},"post_processor":{bytelevel},"decoder":{bytelevel},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{{{vocab}}},"merges":[]}}}}"#,
        test_json_string(pattern)
    )
}

#[cfg(test)]
pub(super) fn test_write_serving_metadata(path: &Path) {
    fs::write(path.join("generation_config.json"), test_generation_config()).unwrap();
    fs::write(path.join("tokenizer_config.json"), test_tokenizer_config()).unwrap();
    fs::write(path.join("tokenizer.json"), test_tokenizer_json()).unwrap();
}

#[cfg(test)]
#[path = "model_serving_tests.rs"]
mod serving_tests;

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) struct Fixture(pub(super) PathBuf);
    impl Fixture {
        pub(super) fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "vade-launch-model-{}",
                crate::text::random_hex(8).unwrap()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(fs::canonicalize(path).unwrap())
        }
        pub(super) fn model(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("config.json"), test_config(8192)).unwrap();
            test_write_serving_metadata(&path);
            fs::write(path.join("model.safetensors"), [0; 16]).unwrap();
            fs::write(path.join("model.safetensors.index.json"), test_index()).unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn environment_then_preferred_default_then_one_prompt_and_saved_path() {
        let f = Fixture::new();
        let env = f.model("elsewhere/checkpoint");
        let default = f.model("models/org/checkpoint");
        let no_prompt = |_: &str| panic!("unexpected prompt");
        let found = discover_with(&f.0, None, Some(env.as_os_str()), no_prompt).unwrap();
        assert_eq!(found.path, env);
        assert_eq!((found.bytes, found.context_limit), (16, 8192));
        let preferred = f.model("models/nvidia/Qwen3.8-Flash-Next-FP4-Local");
        assert_eq!(discover_with(&f.0, None, None, no_prompt).unwrap().path, preferred);
        fs::remove_dir_all(preferred).unwrap();
        assert_eq!(discover_with(&f.0, None, None, no_prompt).unwrap().path, default);
        assert_eq!(
            discover_with(&f.0, Some(&env), None, no_prompt)
                .unwrap()
                .path,
            env
        );
        fs::remove_dir_all(&default).unwrap();
        assert_eq!(
            discover_with(&f.0, None, Some(env.as_os_str()), no_prompt)
                .unwrap()
                .path,
            env
        );
        let mut prompts = 0;
        assert_eq!(
            discover_with(&f.0, None, None, |_| {
                prompts += 1;
                Ok(env.clone())
            })
            .unwrap()
            .path,
            env
        );
        assert_eq!(prompts, 1);
        let mut prompts = 0;
        assert!(
            discover_with(&f.0, None, None, |_| {
                prompts += 1;
                Ok(f.0.join("missing"))
            })
            .is_err()
        );
        assert_eq!(prompts, 1);
    }
    #[test]
    fn ambiguity_and_saved_missing_require_an_explicit_selection() {
        let f = Fixture::new();
        let a = f.model("models/a");
        f.model("models/b");
        assert!(
            locate(&f.0.join("models"))
                .unwrap_err()
                .contains("More than one")
        );
        assert_eq!(
            discover_with(&f.0, None, None, |_| Ok(a.clone()))
                .unwrap()
                .path,
            a
        );
        let mut prompts = 0;
        let recovered = discover_with(&f.0, Some(&f.0.join("missing")), None, |reason| {
            assert!(reason.contains("unavailable"));
            prompts += 1;
            Ok(a.clone())
        })
        .unwrap();
        assert_eq!(prompts, 1);
        assert_eq!(recovered.path, a);
    }
    #[test]
    fn shard_inventory_counts_once_and_rejects_missing_empty_or_escape() {
        let f = Fixture::new();
        let path = f.model("models/checkpoint");
        let index = path.join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"metadata":{},"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}"#,
        )
        .unwrap();
        assert_eq!(inspect(&path).unwrap().bytes, 16);
        for value in [
            r#"{"metadata":{},"weight_map":{"a":"missing.safetensors"}}"#,
            r#"{"metadata":{},"weight_map":{"a":"../model.safetensors"}}"#,
            r#"{"metadata":{},"weight_map":{}}"#,
            r#"{"metadata":{},"weight_map":{"a":42}}"#,
            r#"{"metadata":{},"weight_map":{"a":"model.safetensors"},}"#,
        ] {
            fs::write(&index, value).unwrap();
            assert!(inspect(&path).is_err(), "{value}");
        }
        fs::write(index, test_index()).unwrap();
        fs::write(path.join("model.safetensors"), "").unwrap();
        assert!(inspect(&path).unwrap_err().contains("empty"));
    }
    #[test]
    fn payload_links_stay_inside_checkpoint_and_directory_loops_are_skipped() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let path = f.model("models/checkpoint");
        let blob = f.0.join("blob");
        fs::write(&blob, [0; 24]).unwrap();
        fs::remove_file(path.join("model.safetensors")).unwrap();
        symlink(blob, path.join("model.safetensors")).unwrap();
        symlink(f.0.join("models"), f.0.join("models/loop")).unwrap();
        assert!(
            locate(&f.0.join("models"))
                .unwrap_err()
                .contains("escapes the checkpoint")
        );
        fs::remove_file(path.join("model.safetensors")).unwrap();
        fs::write(path.join("inside.safetensors"), [0; 24]).unwrap();
        symlink("inside.safetensors", path.join("model.safetensors")).unwrap();
        assert_eq!(locate(&f.0.join("models")).unwrap().bytes, 24);
        fs::remove_dir_all(path).unwrap();
        assert!(
            locate(&f.0.join("models"))
                .unwrap_err()
                .contains("No complete")
        );
    }
    #[test]
    fn checkpoint_symlink_aliases_resolve_once_and_distinct_targets_are_ambiguous() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let checkpoint = f.model("other-volume/checkpoint");
        let models = f.0.join("models");
        fs::create_dir_all(&models).unwrap();
        symlink(&checkpoint, models.join("primary")).unwrap();
        symlink(&checkpoint, models.join("alias")).unwrap();
        let found = discover_with(&f.0, None, None, |_| {
            panic!("linked checkpoint should be found")
        })
        .unwrap();
        assert_eq!(found.path, checkpoint);
        assert_eq!(found.bytes, 16);
        let second = f.model("other-volume/second");
        symlink(second, models.join("another")).unwrap();
        assert!(locate(&models).unwrap_err().contains("More than one model"));
    }
    #[test]
    fn dangling_loop_and_container_symlinks_do_not_expand_discovery() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let checkpoint = f.model("other-volume/checkpoint");
        let models = f.0.join("models");
        fs::create_dir_all(&models).unwrap();
        symlink(models.join("cycle-b"), models.join("cycle-a")).unwrap();
        symlink(models.join("cycle-a"), models.join("cycle-b")).unwrap();
        symlink(&models, models.join("self")).unwrap();
        symlink(f.0.join("missing"), models.join("dangling")).unwrap();
        symlink(f.0.join("other-volume"), models.join("container")).unwrap();
        assert!(locate(&models).unwrap_err().contains("No complete model"));
        symlink(&checkpoint, models.join("direct")).unwrap();
        assert_eq!(locate(&models).unwrap().path, checkpoint);
    }
    #[test]
    fn engine_metadata_filters_architecture_quantization_and_recovers_saved_choice() {
        let f = Fixture::new();
        let supported = f.model("elsewhere/fp4-local");
        fs::write(supported.join("config.json"), test_mixed_config(8192)).unwrap();
        let unsupported = f.model("models/qwen3-8b");
        fs::write(
            unsupported.join("config.json"),
            test_config(8192).replace("qwen4_exp", "qwen3"),
        )
        .unwrap();
        let error = inspect(&unsupported).unwrap_err();
        assert!(
            error.contains("cannot load") && error.contains("model_type"),
            "{error}"
        );
        assert!(!error.contains('\n'));
        assert_eq!(
            discover_with(&f.0, None, Some(supported.as_os_str()), |_| panic!())
                .unwrap()
                .path,
            supported
        );
        assert_eq!(
            discover_with(
                &f.0,
                Some(&unsupported),
                Some(supported.as_os_str()),
                |_| panic!()
            )
            .unwrap()
            .path,
            supported
        );
        assert!(
            discover_with(&f.0, Some(&unsupported), None, |reason| Err(reason.into()))
                .unwrap_err()
                .contains("model_type")
        );
        // A supported architecture is insufficient: reject incompatible quantization,
        // cache settings and mixed-layer group sizes using the real engine parser.
        let config = test_config(65536);
        for invalid in [
            config.replace("qwen4_exp_text", "qwen3_text"),
            config.replace("\"fp8\"", "\"awq\""),
            config.replace("[128,128]", "[64,128]"),
            config.replace("\"dynamic\"", "\"static\""),
            test_mixed_config(65536).replace("\"group_size\":16", "\"group_size\":32"),
            test_mixed_config(65536).replace(
                "\"kv_cache_quant_algo\":null",
                "\"kv_cache_quant_algo\":\"FP8\"",
            ),
        ] {
            fs::write(supported.join("config.json"), invalid).unwrap();
            assert!(inspect(&supported).unwrap_err().contains("cannot load"));
        }
        for supported_config in [config, test_mixed_config(65536)] {
            fs::write(supported.join("config.json"), supported_config).unwrap();
            assert_eq!(inspect(&supported).unwrap().context_limit, 65536);
        }
    }

    #[test]
    fn metadata_context_is_read_from_supported_engine_geometry() {
        let f = Fixture::new();
        let path = f.model("model");
        fs::write(path.join("config.json"), test_config(65536)).unwrap();
        assert_eq!(inspect(&path).unwrap().context_limit, 65536);
        for value in ["-1", "\"8192\"", "null", "[]", "false"] {
            fs::write(
                path.join("config.json"),
                test_config(8192).replace(
                    "\"max_position_embeddings\":8192",
                    &format!("\"max_position_embeddings\":{value}"),
                ),
            )
            .unwrap();
            assert!(inspect(&path).is_err(), "{value}");
        }
    }

    #[test]
    fn legacy_nvfp4_and_malformed_sidecars_refuse_before_admission() {
        let f = Fixture::new();
        let path = f.model("checkpoint");
        let config = legacy_config(8192);
        let parsed = engine_model::flash::Config::parse(config.as_bytes()).unwrap();
        assert_eq!(parsed.format, engine_model::flash::WeightFormat::Nvfp4);
        assert!(parsed.mixed.is_none());
        fs::write(path.join("config.json"), config).unwrap();
        assert!(
            inspect(&path)
                .unwrap_err()
                .contains("legacy NVFP4 is unsupported here")
        );
        fs::write(path.join("config.json"), test_config(8192)).unwrap();
        fs::write(path.join("hf_quant_config.json"), "{}").unwrap();
        assert!(inspect(&path).unwrap_err().contains("quantization sidecar"));
        fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
        assert!(inspect(&path).unwrap_err().contains("quantization sidecar"));
        fs::remove_file(path.join("hf_quant_config.json")).unwrap();
        assert_eq!(inspect(&path).unwrap().context_limit, 8192);
    }

    fn sidecar() -> String {
        let mut layers = (0..48)
            .map(|layer| {
                format!(r#""model.language_model.layers.{layer}.mlp.experts":{{"quant_algo":"NVFP4","group_size":16}}"#)
            })
            .collect::<Vec<_>>();
        layers.push(
            r#""mtp.layers.0.mlp.experts":{"quant_algo":"FP8_BLOCK_SCALES","group_size":128}"#
                .into(),
        );
        layers.push(format!(
            r#""{}.ngram_embedding":{{"quant_algo":"FP8"}}"#,
            engine_model::flash::PLE_PREFIX
        ));
        format!(
            r#"{{"quantization":{{"quant_algo":"MIXED_PRECISION","group_size":16,"kv_cache_quant_algo":null,"quantized_layers":{{{}}}}}}}"#,
            layers.join(",")
        )
    }

    #[test]
    fn mixed_sidecar_admission_matches_the_complete_startup_contract() {
        use engine_model::flash::{WeightFormat, manifest::ReceiptPlan};
        let f = Fixture::new();
        let path = f.model("mixed");
        let config = test_mixed_config(8192);
        let parsed = engine_model::flash::Config::parse(config.as_bytes()).unwrap();
        assert_eq!(parsed.format, WeightFormat::Nvfp4);
        assert!(parsed.mixed.is_some());
        fs::write(path.join("config.json"), config).unwrap();
        // Both an absent sidecar and a fully valid legacy sidecar are supported.
        assert_eq!(inspect(&path).unwrap().context_limit, 8192);
        assert_eq!(
            ReceiptPlan::open(&path).unwrap().format,
            WeightFormat::Nvfp4
        );
        let valid = sidecar();
        fs::write(path.join("hf_quant_config.json"), &valid).unwrap();
        assert_eq!(inspect(&path).unwrap().context_limit, 8192);
        assert_eq!(
            ReceiptPlan::open(&path).unwrap().format,
            WeightFormat::Nvfp4
        );
        let sidecar_path = path.join("hf_quant_config.json");
        let permissions = fs::metadata(&sidecar_path).unwrap().permissions();
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&sidecar_path, fs::Permissions::from_mode(0o000)).unwrap();
        }
        // DAC-override identities may read mode-000 files. Exercise denial
        // only when this runner actually experiences it, then continue every
        // schema assertion below for both privileged and ordinary runners.
        let unreadable = fs::read(&sidecar_path).is_err();
        let admission = inspect(&path);
        let startup = ReceiptPlan::open(&path);
        fs::set_permissions(&sidecar_path, permissions).unwrap();
        if unreadable {
            let admission = admission.unwrap_err();
            let startup = startup.unwrap_err().to_string();
            assert!(admission.contains(&startup), "{admission}; {startup}");
        } else {
            assert_eq!(admission.unwrap().context_limit, 8192);
            assert_eq!(startup.unwrap().format, WeightFormat::Nvfp4);
        }
        for invalid in [
            "{".into(),
            "{}".into(),
            valid.replace("MIXED_PRECISION", "NVFP4"),
            valid.replacen("\"group_size\":16", "\"group_size\":32", 1),
            valid.replace(
                "\"kv_cache_quant_algo\":null",
                "\"kv_cache_quant_algo\":\"FP8\"",
            ),
            valid.replace(
                "\"quantized_layers\":{",
                "\"quantized_layers\":{\"extra\":{},",
            ),
            valid.replace("layers.47.mlp.experts", "layers.48.mlp.experts"),
            valid.replace("\"NVFP4\"", "\"FP8\""),
            valid.replace("\"group_size\":16", "\"group_size\":32"),
            valid.replace("FP8_BLOCK_SCALES", "FP8_PB_WO"),
            valid.replace("\"group_size\":128", "\"group_size\":64"),
            valid.replace("\"FP8\"", "\"NVFP4\""),
        ] {
            fs::write(path.join("hf_quant_config.json"), &invalid).unwrap();
            let admission = inspect(&path).unwrap_err();
            let startup = ReceiptPlan::open(&path).unwrap_err().to_string();
            assert!(
                admission.contains(&startup),
                "admission: {admission}; startup: {startup}"
            );
            assert!(
                discover_with(&f.0, Some(&path), None, |reason| Err(reason.into())).is_err(),
                "invalid saved model must not be republished"
            );
        }
        let sidecar_path = path.join("hf_quant_config.json");
        let file = File::create(&sidecar_path).unwrap();
        file.set_len(1024 * 1024 + 1).unwrap();
        assert!(inspect(&path).unwrap_err().contains("byte limit"));
        assert!(
            ReceiptPlan::open(&path)
                .unwrap_err()
                .to_string()
                .contains("byte limit")
        );
        fs::remove_file(&sidecar_path).unwrap();
        fs::create_dir(&sidecar_path).unwrap();
        assert!(inspect(&path).unwrap_err().contains("regular file"));
        assert!(
            ReceiptPlan::open(&path)
                .unwrap_err()
                .to_string()
                .contains("regular file")
        );
        fs::remove_dir(&sidecar_path).unwrap();
        std::os::unix::fs::symlink(path.join("missing"), &sidecar_path).unwrap();
        assert!(inspect(&path).is_err());
        assert!(ReceiptPlan::open(&path).is_err());
    }

    #[test]
    fn mixed_fifo_sidecar_refuses_in_discovery_and_startup_without_a_writer() {
        use engine_model::flash::manifest::ReceiptPlan;
        let _guard = crate::runner::process_test_lock();
        let f = Fixture::new();
        let path = f.model("mixed");
        fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
        let sidecar_path = path.join("hf_quant_config.json");
        let fifo = path.join("fifo");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        for linked in [false, true] {
            if linked {
                std::os::unix::fs::symlink(&fifo, &sidecar_path).unwrap();
            } else {
                fs::rename(&fifo, &sidecar_path).unwrap();
            }
            let root = path.clone();
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                let admission = inspect(&root).unwrap_err();
                let startup = ReceiptPlan::open(&root).unwrap_err().to_string();
                sender.send((admission, startup)).unwrap();
            });
            let (admission, startup) = receiver
                .recv_timeout(Duration::from_secs(10))
                .expect("mixed sidecar reader blocked on FIFO");
            worker.join().unwrap();
            assert!(admission.contains("regular file"), "{admission}");
            assert!(startup.contains("regular file"), "{startup}");
            if linked {
                fs::remove_file(&sidecar_path).unwrap();
            } else {
                fs::rename(&sidecar_path, &fifo).unwrap();
            }
        }
    }

    fn legacy_config(context: u32) -> String {
        let experts = (0..48)
            .map(|layer| format!("model.language_model.layers.{layer}.mlp.experts"))
            .collect::<Vec<_>>();
        let mtp = "mtp.layers.0.mlp.experts";
        let ple = format!("{}.ngram_embedding", engine_model::flash::PLE_PREFIX);
        let mut layers = experts
            .iter()
            .map(|name| format!(r#"{name:?}:{{"quant_algo":"NVFP4","group_size":16}}"#))
            .collect::<Vec<_>>();
        layers.push(format!(
            r#"{mtp:?}:{{"quant_algo":"FP8_PB_WO","group_size":128}}"#
        ));
        layers.push(format!(r#"{ple:?}:{{"quant_algo":"FP8"}}"#));
        let mut groups = Vec::new();
        for (id, bits, group, dynamic, targets) in [
            (0, 4, Some(16), false, experts),
            (1, 8, Some(128), true, vec![mtp.into()]),
            (2, 8, None, false, vec![ple]),
        ] {
            let size = group
                .map(|size| format!(r#", "group_size":{size}"#))
                .unwrap_or_default();
            let operand = |dynamic| {
                format!(r#"{{"num_bits":{bits},"type":"float","dynamic":{dynamic}{size}}}"#)
            };
            let activation = if group.is_some() {
                format!(r#", "input_activations":{}"#, operand(dynamic))
            } else {
                String::new()
            };
            groups.push(format!(
                r#""group_{id}":{{"targets":{targets:?},"weights":{}{activation}}}"#,
                operand(false)
            ));
        }
        let quant = format!(
            r#"{{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{{{}}},"config_groups":{{{}}}}}"#,
            layers.join(","),
            groups.join(",")
        );
        let config = test_config(context);
        let (prefix, _) = config.rsplit_once("\"quantization_config\":").unwrap();
        format!(r#"{prefix}"quantization_config":{quant}}}"#)
    }

    #[test]
    fn shard_index_uses_engine_format_and_filename_limits() {
        let f = Fixture::new();
        let path = f.model("checkpoint");
        let index = path.join("model.safetensors.index.json");
        fs::remove_file(&index).unwrap();
        assert!(
            inspect(&path)
                .unwrap_err()
                .contains("model.safetensors.index.json")
        );
        fs::create_dir(path.join("nested")).unwrap();
        fs::write(path.join("nested/weight.safetensors"), [0; 16]).unwrap();
        let nested = r#"{"metadata":{},"weight_map":{"a":"nested/weight.safetensors"}}"#;
        fs::write(&index, nested).unwrap();
        assert!(inspect(&path).unwrap_err().contains("single relative"));
        fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
        assert_eq!(inspect(&path).unwrap().bytes, 16);
        for shard in [
            "nested//weight.safetensors",
            "nested/./weight.safetensors",
            "has space.safetensors",
        ] {
            fs::write(
                &index,
                format!(r#"{{"metadata":{{}},"weight_map":{{"a":{shard:?}}}}}"#),
            )
            .unwrap();
            assert!(
                inspect(&path).unwrap_err().contains("unsafe relative"),
                "{shard}"
            );
        }
        for malformed in [
            r#"{"weight_map":{"a":"model.safetensors"}}"#,
            r#"{"metadata":{},"weight_map":{"":"model.safetensors"}}"#,
        ] {
            fs::write(&index, malformed).unwrap();
            assert!(
                inspect(&path)
                    .unwrap_err()
                    .contains("cannot load the model shard index")
            );
        }
    }

    #[test]
    fn shard_index_links_match_startup_checkpoint_containment() {
        use engine_model::flash::manifest::ReceiptPlan;
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let path = f.model("mixed");
        fs::write(path.join("config.json"), test_mixed_config(8192)).unwrap();
        let index = path.join("model.safetensors.index.json");
        let external = f.0.join("external-index.json");
        fs::rename(&index, &external).unwrap();
        symlink(&external, &index).unwrap();
        let startup = ReceiptPlan::open(&path).unwrap_err().to_string();
        assert!(startup.contains("escapes checkpoint"), "{startup}");
        let admission = inspect(&path).unwrap_err();
        assert!(admission.contains("index") && admission.contains("escapes"), "{admission}");
        assert!(discover_with(&f.0, Some(&path), None, |reason| Err(reason.into())).is_err());

        // The same index bytes can be linked within the checkpoint, including
        // when the owner chooses an alias of the whole directory.
        fs::remove_file(&index).unwrap();
        let internal = path.join("actual-index.json");
        fs::rename(&external, &internal).unwrap();
        symlink("actual-index.json", &index).unwrap();
        assert_eq!(inspect(&path).unwrap().path, path);
        ReceiptPlan::open(&path).unwrap();
        let alias = f.0.join("alias");
        symlink(&path, &alias).unwrap();
        assert_eq!(inspect(&alias).unwrap().path, path);
        ReceiptPlan::open(&alias).unwrap();
    }

    #[test]
    fn fifo_metadata_is_rejected_without_waiting_for_a_writer() {
        let _guard = crate::runner::process_test_lock();
        let f = Fixture::new();
        let path = f.model("model");
        for name in [
            "model.safetensors.index.json",
            "hf_quant_config.json",
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
        ] {
            let target = path.join(name);
            let original = fs::read(&target).ok();
            if original.is_some() {
                fs::remove_file(&target).unwrap();
            }
            assert!(
                std::process::Command::new("mkfifo")
                    .arg(&target)
                    .status()
                    .unwrap()
                    .success()
            );
            let target = path.clone();
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                let _ = sender.send(inspect(&target));
            });
            let error = receiver
                .recv_timeout(Duration::from_secs(10))
                .expect("metadata reader blocked on FIFO")
                .unwrap_err();
            worker.join().unwrap();
            assert!(
                error.contains(if name == "hf_quant_config.json" {
                    "quantization sidecar"
                } else {
                    "regular file"
                }),
                "{name}: {error}"
            );
            fs::remove_file(path.join(name)).unwrap();
            if let Some(original) = original {
                fs::write(path.join(name), original).unwrap();
            }
        }
        fs::remove_file(path.join("model.safetensors.index.json")).unwrap();
        fs::create_dir(path.join("model.safetensors.index.json")).unwrap();
        assert!(inspect(&path).unwrap_err().contains("regular file"));
    }

    #[test]
    fn large_checkpoint_index_fits_the_bound_and_oversized_index_refuses() {
        let f = Fixture::new();
        let path = f.model("large-index");
        let index = path.join("model.safetensors.index.json");
        let mut file = File::create(&index).unwrap();
        file.write_all(br#"{"metadata":{},"weight_map":{"#).unwrap();
        // Whitespace exercises the byte bound without violating the engine's
        // independent 1 MiB limit for an individual JSON string.
        io::copy(&mut io::repeat(b' ').take(48 * 1024 * 1024), &mut file).unwrap();
        file.write_all(br#""a":"model.safetensors"}}"#).unwrap();
        drop(file);
        assert_eq!(inspect(&path).unwrap().bytes, 16);
        let file = File::create(&index).unwrap();
        file.set_len(64 * 1024 * 1024 + 1).unwrap();
        assert!(
            inspect(&path)
                .unwrap_err()
                .contains("exceeds its size limit")
        );
    }
}
