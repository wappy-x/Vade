//! Additive, versioned engine contract shared by single- and multi-rank serving.
use crate::{Limits, Tokenizer, generation, http, protocol};
use engine_model::json::{self, Value};
use std::{fs::File, io::Read, path::Path};

#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;

pub const VERSION: &str = "vade.engine.v1";

#[test]
fn nested_weight_identity() {
    let root = std::env::temp_dir().join(format!(
        "vade-identity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("tensors")).unwrap();
    for (name, text) in [
        (
            "config.json",
            r#"{"model_type":"qwen4_exp","quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,"ignore":[],"quantized_layers":{"a":{"quant_algo":"NVFP4","group_size":16}}}}"#,
        ),
        ("tokenizer.json", "{}"),
        ("tokenizer_config.json", r#"{"chat_template":"template"}"#),
        (
            "model.safetensors.index.json",
            r#"{"metadata":{},"weight_map":{"weight":"tensors/model.safetensors"}}"#,
        ),
    ] {
        std::fs::write(root.join(name), text).unwrap();
    }
    std::fs::write(
        root.join("tensors/model.safetensors"),
        [2u64.to_le_bytes().as_slice(), b"{}"].concat(),
    )
    .unwrap();
    assert_eq!(Identity::load(&root).unwrap().weight_bytes, 10);
    let embedded=Identity::load(&root).unwrap().fingerprint();
    std::fs::write(root.join("chat_template.jinja"),b"template").unwrap();
    std::fs::write(root.join("tokenizer_config.json"),b"{}").unwrap();
    assert_eq!(Identity::load(&root).unwrap().fingerprint(),embedded);
    std::fs::write(root.join("tokenizer_config.json"),br#"{"chat_template":"template"}"#).unwrap();
    let metadata=crate::metadata::Metadata::resident_text(&Identity::load(&root).unwrap(),2,"ring");
    let diagnostic=json::parse(metadata.json().as_bytes(),json::Limits::default()).unwrap();
    assert_eq!(diagnostic.object().unwrap().get("qualification").unwrap().string().unwrap(),"unqualified");
    assert_eq!(diagnostic.object().unwrap().get("diagnostic_option").unwrap().string().unwrap(),"--diagnostic-unqualified-text");
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"qwen4_exp","quantization_config":{"quant_method":"fp8"}}"#,
    )
    .unwrap();
    assert!(Identity::load(&root).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

/// Flash accepts an optional template copy with different trailing line endings.
pub fn normalize_template(text: &str) -> &str { text.trim_end_matches(['\r', '\n']) }

#[derive(Clone, Debug)]
pub struct Identity {
    weights: String,
    weight_bytes: u64,
    tokenizer: String,
    template: String,
    tools: bool,
    thinking: bool,
}
impl Identity {
    /// Fingerprint checkpoint metadata once at load, using 64 KiB of streaming
    /// hash scratch. Safetensors payloads are never read or content-digested.
    /// Like the loader, this requires files not to change during their lifetime.
    pub fn load(root: &Path) -> Result<Self, String> {
        let read = |name: &str| engine_model::safetensors::read_bounded(&root.join(name), 4 << 20)
            .map_err(|_| format!("cannot read identity artifact {name}"));
        let config = read("config.json")?;
        let value = json::parse(&config, json::Limits { bytes: 4 << 20, ..Default::default() })
            .map_err(|e| e.to_string())?;
        let kind = value.object().map_err(|e| e.to_string())?.get("model_type")
            .and_then(|v| v.string().ok()).unwrap_or("");
        let thinking = matches!(kind, "qwen3" | "qwen3_moe" | "qwen3_5" | "qwen3_5_moe" | "qwen4_exp" | "gemma4" | "glm5_next" | "glm5_next_text");
        let tools = thinking || kind == "llama";
        let mut names = std::fs::read_dir(root).map_err(|_| "cannot list checkpoint artifacts")?
            .map(|entry| entry.map(|e| e.path()).map_err(|_| "cannot list checkpoint artifact"))
            .collect::<Result<Vec<_>, _>>()?;
        names.retain(|p| p.extension().is_some_and(|s| s == "safetensors"));
        // Indexed checkpoints can keep shards in subdirectories. Use the
        // validated index names and still reject symlink escapes.
        if names.is_empty()
            && (matches!(engine_model::plan::ConfigPlan::from_config(&config),Ok(engine_model::plan::ConfigPlan::Deferred(_)))
                || value.object().map_err(|e|e.to_string())?.get("quantization_config")
                    .is_some_and(|q|q.object().is_ok_and(|q|engine_model::flash::mixed::Mixed::parse(q).is_ok())))
            && root
                .join("model.safetensors.index.json")
                .try_exists()
                .map_err(|e| e.to_string())?
        {
            let index = engine_model::safetensors::Index::parse_mixed(
                &engine_model::safetensors::read_bounded(
                    &root.join("model.safetensors.index.json"),
                    64 << 20,
                )
                .map_err(|e| e.to_string())?,
                engine_model::flash::checkpoint_limits(engine_model::flash::WeightFormat::Nvfp4, true),
            )
            .map_err(|e| e.to_string())?;
            let canonical = root.canonicalize().map_err(|e| e.to_string())?;
            for name in index
                .weight_map
                .values()
                .collect::<std::collections::BTreeSet<_>>()
            {
                let path = root.join(name);
                if !path
                    .canonicalize()
                    .map_err(|e| e.to_string())?
                    .starts_with(&canonical)
                {
                    return Err("weight identity artifact escapes checkpoint".into());
                }
                names.push(path);
            }
        }

        names.sort();
        if names.is_empty() { return Err("no checkpoint weight artifacts".into()); }
        let mut records = format!("config.json:{}\n", sha256::digest(&config));
        match File::open(root.join("model.safetensors.index.json")) {
            Ok(file) => records.push_str(&format!("index:{}\n",
                sha256::digest_reader(file).map_err(|_| "cannot hash weight index")?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(_) => return Err("cannot open weight index".into()),
        }
        let mut weight_bytes = 0u64;
        for path in names {
            let mut file = File::open(&path).map_err(|_| "cannot open weight identity artifact")?;
            let bytes = file.metadata().map_err(|_| "cannot stat weight identity artifact")?.len();
            weight_bytes = weight_bytes.checked_add(bytes).ok_or("weight identity byte overflow")?;
            let mut prefix = [0; 8];
            file.read_exact(&mut prefix).map_err(|_| "cannot read weight header length")?;
            let header_bytes = u64::from_le_bytes(prefix);
            if header_bytes == 0 || header_bytes > engine_model::safetensors::Limits::default().header_bytes as u64
                || header_bytes > bytes.saturating_sub(8) {
                return Err("invalid weight identity header length".into());
            }
            let mut header = file.take(header_bytes);
            let digest = sha256::digest_reader(prefix.as_slice().chain(&mut header))
                .map_err(|_| "cannot hash weight identity header")?;
            if header.limit() != 0 { return Err("truncated weight identity header".into()); }
            let name = path.strip_prefix(root).ok().and_then(|s| s.to_str()).ok_or("invalid weight artifact name")?;
            records.push_str(&format!("{}:{bytes}:{digest}\n", protocol::quote(name)));
        }
        let tokenizer = sha256::digest_reader(File::open(root.join("tokenizer.json"))
            .map_err(|_| "cannot open tokenizer identity artifact")?).map_err(|_| "cannot hash tokenizer artifact")?;
        let config = read("tokenizer_config.json")?;
        let mut config = json::parse(&config, json::Limits { bytes: 4 << 20, ..Default::default() })
            .map_err(|e| e.to_string())?;
        let Value::Object(ref mut fields) = config else { return Err("invalid tokenizer configuration".into()); };
        // Gemma renders the separate canonical artifact; other families use the
        // embedded text. Optional copies are validated by the Flash loader.
        let template = if kind == "gemma4" || !fields.contains_key("chat_template") {
            String::from_utf8(read("chat_template.jinja")?).map_err(|_| "invalid template UTF-8")?
        } else {
            json::field(fields, "chat_template").and_then(Value::string)
                .map_err(|e| e.to_string())?.to_owned()
        };
        fields.insert("chat_template".into(), Value::String(normalize_template(&template).into()));
        // Include the authored renderer/translator as well as checkpoint framing.
        let mut renderer = protocol::stringify(&config).into_bytes();
        for source in [include_bytes!("../../tokenizer/src/chat.rs").as_slice(),
            include_bytes!("../../tokenizer/src/chat/llama.rs").as_slice(),
            include_bytes!("../../tokenizer/src/chat/gemma4.rs").as_slice(),
            include_bytes!("../../tokenizer/src/lib.rs").as_slice(),
            include_bytes!("tokenizer/glm.rs").as_slice(),
            include_bytes!("tokenizer.rs").as_slice(), include_bytes!("protocol.rs").as_slice(),
            include_bytes!("chat_output.rs").as_slice(), include_bytes!("chat_output/glm.rs").as_slice(),
            include_bytes!("responses.rs").as_slice(), include_bytes!("anthropic.rs").as_slice()] {
            renderer.extend_from_slice(&(source.len() as u64).to_le_bytes());
            renderer.extend_from_slice(source);
        }
        Ok(Self { weights: sha256::digest(records.as_bytes()), weight_bytes, tokenizer,
            template: sha256::digest(&renderer), tools, thinking })
    }
    fn json(&self) -> String {
        format!("{{\"weights\":{{\"fingerprint\":\"sha256:{}\",\"fingerprint_kind\":\"safetensors_header_metadata\",\"artifact_bytes\":{}}},\"tokenizer_fingerprint\":\"sha256:{}\",\"template_fingerprint\":\"sha256:{}\"}}",
            self.weights, self.weight_bytes, self.tokenizer, self.template)
    }
    /// Metadata identity only. This does not hash weight payload bytes.
    pub fn fingerprint(&self)->String {sha256::digest(self.json().as_bytes())}
}

/// A fresh load epoch. Reporting identifiers confer no rights and never select KV.
pub(crate) struct Domain(crate::cache::SessionKey);
impl Domain {
    pub fn new() -> Result<Self, String> { crate::cache::SessionKey::generate().map(Self) }
    pub fn id(&self, scope: Option<&str>) -> String {
        let mut bytes = self.0.expose().into_bytes();
        bytes.push(u8::from(scope.is_some()));
        bytes.extend_from_slice(scope.unwrap_or("").as_bytes());
        sha256::digest(&bytes)
    }
    pub fn key(&self, scope: Option<&str>) -> crate::cache::SessionKey {
        crate::cache::SessionKey::parse(&self.id(scope)).expect("SHA256 is 64 lowercase hex digits")
    }
    pub fn capabilities(&self, model: &str, context: usize, limits: &Limits,
        scope: Option<&str>, scoped: bool, cache: bool, handles: bool, cache_limits: &crate::cache::CacheLimits) -> String {
        let identity = limits.identity.as_ref().map(Identity::json).unwrap_or_else(|| "null".into());
        let support = |f: fn(&Identity) -> bool| limits.identity.as_ref()
            .map(|i| f(i).to_string()).unwrap_or_else(|| "null".into());
        let checkpoints = if handles { format!("{{\"cache_domain\":{},\"scope\":\"authorization\",\"max_handles_per_scope\":{},\"max_bytes_per_scope\":{},\"ttl_ms\":{},\"release\":\"/v1/kv/release\"}}",
            protocol::quote(&self.id(scope)), cache_limits.checkpoint_entries(), cache_limits.checkpoint_bytes(),
            cache_limits.idle_ttl.as_millis()) } else { "null".into() };
        format!("{{\"object\":\"engine.capabilities\",\"contract_version\":{},\"model\":{},\"identity\":{identity},\"context_limit\":{},\"input_limit\":{},\"output_limit\":{},\"tools\":{},\"thinking\":{},\"default_thinking\":{},\"cache_domain\":{},\"cache_scope\":{},\"prefix_cache\":{cache},\"prompt_cache_key\":\"scheduling_hint\",\"cache_salt\":\"cache_namespace\",\"token_id_prompts\":true,\"ignore_eos\":true,\"admission\":\"/v1/admission\",\"admission_kind\":\"context_fit\",\"usage_timing\":\"admitted_wall\",\"kv_branch_handles\":{handles},\"kv_checkpoints\":{checkpoints},\"endpoints\":[\"/tokenize\",\"/v1/chat/completions\",\"/v1/responses\",\"/v1/completions\",\"/v1/messages\",\"/v1/messages/count_tokens\"],\"anthropic\":{{\"tool_choice\":[\"auto\",\"none\",\"any\",\"tool\"],\"disable_parallel_tool_use\":true,\"thinking_budget_enforced\":false,\"cache_control\":\"ignored_automatic_prefix_cache\",\"cache_creation_input_tokens\":0,\"content\":[\"text\",\"thinking\",\"tool_use\",\"tool_result\"]}}}}",
            protocol::quote(VERSION), protocol::quote(model), context.min(limits.context_tokens),
            limits.input_tokens, limits.output_limit(context), support(|i| i.tools), support(|i| i.thinking),
            limits.default_thinking, protocol::quote(&self.id(if scoped { scope } else { None })),
            protocol::quote(if scoped { "authorization" } else { "shared" }))
    }
}

/// Render the complete native request with the same code as inference. This is
/// a context-fit decision, not a promise of a free GPU slot or a KV reservation.
pub(crate) fn admission<T: Tokenizer>(body: &[u8], context: usize, vocab: usize, eos: &[u32],
    tokenizer: Option<&T>, limits: &Limits, model: &str) -> Result<String, String> {
    let value = json::parse(body, json::Limits { bytes: limits.body_bytes, depth: 28,
        nodes: limits.input_tokens.saturating_add(65536), string_bytes: limits.body_bytes })
        .map_err(|e| e.to_string())?;
    let object = value.object().map_err(|e| e.to_string())?;
    if let Some(version) = protocol::present(object, "contract_version") {
        if version.string().map_err(|e| e.to_string())? != VERSION { return Err("unsupported engine contract version".into()); }
    }
    let endpoint = json::field(object, "endpoint").and_then(Value::string).map_err(|e| e.to_string())?;
    if !matches!(endpoint, "/v1/chat/completions" | "/v1/responses" | "/v1/completions" | "/v1/messages" | "/v1/messages/count_tokens") {
        return Err("unsupported admission endpoint".into());
    }
    let reservation = usize::try_from(json::field(object, "output_tokens").and_then(Value::u64)
        .map_err(|e| e.to_string())?).map_err(|_| "output reservation overflow")?;
    if endpoint == "/v1/messages/count_tokens" && reservation != 0 {
        return Err("count_tokens admission requires zero output_tokens".into());
    }
    let raw = protocol::raw_member(std::str::from_utf8(body).map_err(|_| "invalid UTF-8")?, "request")?
        .ok_or("missing admission request")?;
    let http = http::HttpRequest { path: endpoint.into(), body: raw.as_bytes().to_vec(), authorization: None };
    let (request, prompt, _, _) = http::render_request(&http, context, tokenizer, limits, model, Some(reservation))?;
    http::validate_token_mapping(&prompt, tokenizer)?;
    http::validate_stop_mapping(eos, tokenizer)?;
    let error = if endpoint == "/v1/messages/count_tokens" {
        http::validate_count_prompt(&prompt, context, vocab, tokenizer, limits).err()
    } else { generation::validate_bounds(context, vocab, eos, &prompt, request.max_tokens, limits).err() };
    Ok(format!("{{\"object\":\"engine.admission\",\"contract_version\":{},\"model\":{},\"prompt_tokens\":{},\"output_tokens\":{reservation},\"context_limit\":{},\"input_limit\":{},\"output_limit\":{},\"admitted\":{},\"reason\":{},\"admission_kind\":\"context_fit\",\"reserves_resources\":false}}",
        protocol::quote(VERSION), protocol::quote(model), prompt.len(), context.min(limits.context_tokens),
        limits.input_tokens, limits.output_limit(context), error.is_none(),
        error.as_deref().map(protocol::quote).unwrap_or_else(|| "null".into())))
}
