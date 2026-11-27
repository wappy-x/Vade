//! Which text service a checkpoint declares (`Loop` for Inkling 2 Mini, `Resident`
//! for GLM-5.3-Flash) and the draft depths it accepts.
//!
//! The classification comes from the checkpoint plan ([`engine_model::plan::ConfigPlan`]),
//! never from a folder name alone.
use crate::config::Machine;
use crate::state::TEXT_DRAFT_DEPTH;
use crate::text::Fields;
use std::path::Path;

/// Two selected checkpoints declare different text services.
pub const DIFFERENT_SERVICES: &str = "The selected checkpoints do not declare the same text service. Point every machine at one checkpoint. Nothing started.";

/// The greatest draft depth the text services accept (`--mtp 0..7`).
pub const MAX_DIAGNOSTIC_DEPTH: u32 = 7;

/// Which resident text service family a checkpoint belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Looped block family (Inkling 2 Mini): runs on 1 Spark or as a 2-Spark RDMA `--pair`.
    Loop,
    /// Resident row-split family (GLM-5.3-Flash): runs on linked Sparks without arbitrary Spark caps.
    Resident,
}

/// The text service a checkpoint declares, and the draft depths it accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextService {
    /// Which text service handles this checkpoint.
    pub kind: Kind,
    /// Nominal Sparks that serve one model together (1 for Loop, 2 for Resident).
    pub sparks: usize,
    /// Depth used when `--mtp` is not set.
    pub default_depth: u32,
    /// The greatest depth this service accepts. A named `--mtp` above it is refused.
    pub max_depth: u32,
}

/// Result of inspecting local `config.json` paths on the controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalPlan {
    /// Checkpoints live on remote Sparks (or paths are not readable locally).
    Remote,
    /// Local `config.json` was read and is not a text service (for example Qwen FP4-Local / FP8).
    NonText,
    /// Local `config.json` declares this text service.
    Service(TextService),
}

impl TextService {
    /// Resident row-split text service (GLM-5.3-Flash): default depth 3, max depth 7.
    pub fn pair() -> Self {
        Self { kind: Kind::Resident, sparks: 2, default_depth: TEXT_DRAFT_DEPTH, max_depth: MAX_DIAGNOSTIC_DEPTH }
    }
    /// Alias for [`Self::pair`]: the qualified resident row-split text service.
    pub fn resident() -> Self {
        Self::pair()
    }
    /// Looped block checkpoint with no declared draft model (`max_depth == 0`).
    #[cfg(test)]
    pub fn alone() -> Self {
        Self::alone_with_max_depth(0)
    }
    /// Looped block checkpoint with the greatest draft depth its checkpoint plan declares.
    /// Unnamed depth defaults to `max_depth` (`--mtp <max_depth>`, e.g. 7 for Inkling 2 Mini,
    /// or 0 when the checkpoint declares no draft model); explicit `--mtp 0..=max_depth` overrides it.
    pub fn alone_with_max_depth(max_depth: u32) -> Self {
        Self { kind: Kind::Loop, sparks: 1, default_depth: max_depth, max_depth }
    }
    /// Whether this checkpoint belongs to the looped block family (Inkling).
    pub fn is_alone(self) -> bool {
        self.kind == Kind::Loop
    }
    /// Whether this checkpoint belongs to the looped block family (Inkling).
    pub fn is_loop(self) -> bool {
        self.kind == Kind::Loop
    }
    /// Whether this checkpoint belongs to the resident row-split family (GLM).
    pub fn is_resident(self) -> bool {
        self.kind == Kind::Resident
    }
    /// Wire token for `text_kind` in the worker `text-service` reply.
    pub fn kind_word(self) -> &'static str {
        match self.kind {
            Kind::Loop => "loop",
            Kind::Resident => "resident",
        }
    }
    /// Whether this text service can run on a group of `group` Sparks.
    /// Loop checkpoints run on 1 Spark or a 2-Spark RDMA `--pair`.
    /// Resident text checkpoints have no arbitrary Spark cap (Rule 17); hardware/engine admission decides.
    pub fn accepts_group(self, group: usize) -> bool {
        match self.kind {
            Kind::Loop => matches!(group, 1 | 2),
            Kind::Resident => group >= 1,
        }
    }
    /// Why `group` cannot start, and the command that can.
    pub fn group_refusal(self) -> String {
        match self.kind {
            Kind::Loop => "Inkling 2 Mini runs on 1 Spark or a 2-Spark RDMA pair. Start it as: vade up inkling --sparks 1, vade up inkling --sparks 2, vade up inkling --replicas N, or vade up inkling --replicas N --pairs.".into(),
            Kind::Resident => "Set --sparks to at least 1.".into(),
        }
    }
    /// Why a named draft depth cannot start.
    pub fn draft_refusal(self) -> String {
        if self.max_depth == 0 {
            "Drafting is not built for this checkpoint's text service. Start it with --mtp 0, or leave --mtp out.".into()
        } else {
            format!("Set MTP to a whole number from 0 to {}.", self.max_depth)
        }
    }
    /// The service a worker reported, or `Ok(None)` when the checkpoint is not a text service (e.g. Qwen).
    /// Backward-compatible with older workers that omit `text_kind`.
    pub fn from_report(fields: &Fields) -> Result<Option<Self>, String> {
        let unknown = || "The Spark reported a text service this controller does not know. Run vade install from current source. Nothing started.".to_string();
        match fields.get("text_kind").map(String::as_str) {
            Some("none") => Ok(None),
            Some("loop") => {
                let default_depth: u32 = whole(fields, "text_draft_default")?;
                let max_depth: u32 = whole(fields, "text_draft_max")?;
                if default_depth == max_depth && max_depth <= MAX_DIAGNOSTIC_DEPTH {
                    Ok(Some(Self::alone_with_max_depth(max_depth)))
                } else {
                    Err(unknown())
                }
            }
            Some("resident") => {
                let default_depth: u32 = whole(fields, "text_draft_default")?;
                let max_depth: u32 = whole(fields, "text_draft_max")?;
                if default_depth <= max_depth && max_depth <= MAX_DIAGNOSTIC_DEPTH {
                    Ok(Some(Self { kind: Kind::Resident, sparks: 2, default_depth, max_depth }))
                } else {
                    Err(unknown())
                }
            }
            Some(_) => Err(unknown()),
            None => {
                let sparks: usize = whole(fields, "text_sparks")?;
                let default_depth: u32 = whole(fields, "text_draft_default")?;
                let max_depth: u32 = whole(fields, "text_draft_max")?;
                if sparks == 1 && default_depth == max_depth && max_depth <= MAX_DIAGNOSTIC_DEPTH {
                    Ok(Some(Self::alone_with_max_depth(max_depth)))
                } else if (sparks, default_depth, max_depth) == (2, TEXT_DRAFT_DEPTH, MAX_DIAGNOSTIC_DEPTH) {
                    Ok(Some(Self::resident()))
                } else {
                    Err(unknown())
                }
            }
        }
    }
}

/// Classify `config.json` bytes strictly:
/// - `ConfigPlan::Loop` -> `Some(Kind::Loop)` with `max_depth` from `mtp_config` gamma (`block_size - 1`, capped at 7)
/// - `ConfigPlan::Deferred` (GLM) -> `Some(Kind::Resident)`
/// - Everything else (Qwen FP4-Local, FP8, unknown) -> `None`
pub fn classify_config_bytes(bytes: &[u8]) -> Option<TextService> {
    match engine_model::plan::ConfigPlan::from_config(bytes) {
        Ok(engine_model::plan::ConfigPlan::Loop(description)) => {
            let max_depth = description.declared_draft.as_ref()
                .and_then(|_| engine_model::mtp_draft::MtpConfig::from_config_json(bytes).ok())
                .and_then(|cfg| u32::try_from(cfg.gamma()).ok())
                .map(|gamma| gamma.min(MAX_DIAGNOSTIC_DEPTH))
                .unwrap_or(0);
            Some(TextService::alone_with_max_depth(max_depth))
        }
        Ok(engine_model::plan::ConfigPlan::Deferred(_)) => Some(TextService::resident()),
        _ => None,
    }
}

/// Classify `config.json` bytes, falling back to [`TextService::pair`] when the plan does not parse.
pub fn from_config_bytes(bytes: &[u8]) -> TextService {
    classify_config_bytes(bytes).unwrap_or_else(TextService::pair)
}

/// Classify `config.json` under `root`, or `None` when it is absent, unreadable, or not a text service.
pub fn from_dir(root: &Path) -> Option<TextService> {
    let bytes = engine_model::safetensors::read_bounded(&root.join("config.json"), engine_model::shapes::CONFIG_BYTES).ok()?;
    classify_config_bytes(&bytes)
}

/// Inspect absolute `config.json` paths on this computer. A `~/` path is the Spark's
/// path, not this computer's, and is left unread. Two local checkpoints that declare
/// different plans are an error.
pub fn local_plan(machines: &[Machine]) -> Result<LocalPlan, String> {
    let mut found: Option<Option<TextService>> = None;
    for machine in machines {
        let root = Path::new(&machine.model_path);
        if !root.is_absolute() {
            continue;
        }
        let config = root.join("config.json");
        if !config.exists() {
            continue;
        }
        let bytes = engine_model::safetensors::read_bounded(&config, engine_model::shapes::CONFIG_BYTES)
            .map_err(|error| format!("Cannot read the checkpoint config at {}. {error} Nothing started.", config.display()))?;
        let service = classify_config_bytes(&bytes);
        if let Some(previous) = found && previous != service {
            return Err(DIFFERENT_SERVICES.into());
        }
        found = Some(service);
    }
    Ok(match found {
        None => LocalPlan::Remote,
        Some(None) => LocalPlan::NonText,
        Some(Some(service)) => LocalPlan::Service(service),
    })
}

/// Whether config model/repo hints at Inkling 2 Mini.
pub fn is_inkling_hint(model: &str, repo: &str) -> bool {
    model.to_ascii_lowercase().contains("inkling") || repo.to_ascii_lowercase().contains("inkling")
}

/// Whether config model/repo hints at GLM-5.3-Flash.
pub fn is_glm_hint(model: &str, repo: &str) -> bool {
    crate::card::is_glm_model(model) || repo.to_ascii_lowercase().contains("glm")
}

/// Whether config model/repo hints at Qwen Flash-Next (FP4-Local / FP8).
pub fn is_qwen_hint(model: &str, repo: &str) -> bool {
    model.to_ascii_lowercase().contains("qwen") || repo.to_ascii_lowercase().contains("qwen")
}

fn whole<T: std::str::FromStr>(fields: &Fields, key: &str) -> Result<T, String> {
    fields.get(key).filter(|value| !value.is_empty()).and_then(|value| value.parse().ok())
        .ok_or_else(|| format!("The text-service reply has a bad {key}. Nothing started."))
}

/// The smallest config the plan accepts as a loop text service, plus the
/// architecture label the checkpoint file carries. The plan reads `model_type`.
#[cfg(test)]
pub(crate) const LOOP_CONFIG: &str = r#"{"model_type":"inkling2_loop","architectures":["Inkling2LoopForConditionalGeneration"],"eos_token_id":1,"text_config":{"model_type":"inkling2_loop_text",
"model_max_length":64,"torch_dtype":"bfloat16","hidden_size":32,"num_hidden_layers":4,"vocab_size":24,"unpadded_vocab_size":20,
"num_attention_heads":6,"num_key_value_heads":2,"head_dim":4,"d_rel":4,"rel_extent":6,"q_bias":false,"o_bias":false,
"log_scaling_n_floor":5,"log_scaling_alpha":0.25,"rms_norm_eps":1e-06,"use_value_norm_no_gamma":true,
"qk_l2_clip_threshold":1.5,"rel_l2_clip_threshold":0.75,"local_layer_ids":[0,2],"sconv_kernel_size":4,
"final_logit_softcapping":30.0,"swa_head_dim":8,"swa_num_attention_heads":4,"swa_num_key_value_heads":2,
"sliding_window_size":4,"n_layers_per_attn_res_block":2,"attn_res_normalize_values":true,"n_routed_experts":6,
"num_experts_per_tok":2,"n_shared_experts":0,"intermediate_size":64,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,
"route_scale":2.0,"use_gate_bias":true,
"loop_config":{"core_start":2,"core_end":4,"num_passes":2,"gate_rank":3,"is_varloop":false}}}"#;

/// A loop checkpoint config with a declared draft model (`block_size=8` -> `gamma=7`).
#[cfg(test)]
pub(crate) const LOOP_DRAFT_CONFIG: &str = include_str!("../../engine/crates/model/tests/fixtures/inkling-index/config.json");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_names_the_service_and_classifies_loop_resident_and_other() {
        assert_eq!(classify_config_bytes(LOOP_CONFIG.as_bytes()), Some(TextService::alone()));
        assert_eq!(classify_config_bytes(LOOP_DRAFT_CONFIG.as_bytes()), Some(TextService::alone_with_max_depth(7)));
        assert!(TextService::alone_with_max_depth(7).is_loop());
        assert!(TextService::alone_with_max_depth(7).accepts_group(1));
        assert!(TextService::alone_with_max_depth(7).accepts_group(2));
        assert!(!TextService::alone_with_max_depth(7).accepts_group(4));
        assert!(TextService::resident().is_resident());
        assert!(TextService::resident().accepts_group(1));
        assert!(TextService::resident().accepts_group(2));
        assert!(TextService::resident().accepts_group(4));
        assert_eq!(classify_config_bytes(br#"{"model_type":"other","architectures":["Inkling2LoopForConditionalGeneration"]}"#), None);
        assert_eq!(classify_config_bytes(br#"{"model_type":"inkling2_loop"}"#), None);
        assert_eq!(classify_config_bytes(b"{}"), None);
        assert_eq!(from_config_bytes(b"{}"), TextService::pair());
        assert_eq!(TextService::pair().default_depth, TEXT_DRAFT_DEPTH);
        assert!(TextService::alone().draft_refusal().contains("Drafting is not built"));
        assert_eq!(TextService::alone_with_max_depth(7).default_depth, 7);
        assert_eq!(TextService::alone().default_depth, 0);
        assert_eq!(TextService::alone_with_max_depth(7).draft_refusal(), "Set MTP to a whole number from 0 to 7.");
        let mut fields = Fields::new();
        fields.insert("text_sparks".into(), "1".into());
        fields.insert("text_draft_default".into(), "0".into());
        fields.insert("text_draft_max".into(), "0".into());
        assert_eq!(TextService::from_report(&fields).unwrap(), Some(TextService::alone()));
        fields.insert("text_draft_default".into(), "7".into());
        fields.insert("text_draft_max".into(), "7".into());
        assert_eq!(TextService::from_report(&fields).unwrap(), Some(TextService::alone_with_max_depth(7)));
        fields.insert("text_kind".into(), "none".into());
        assert_eq!(TextService::from_report(&fields).unwrap(), None);
        fields.insert("text_kind".into(), "resident".into());
        fields.insert("text_draft_default".into(), "3".into());
        fields.insert("text_draft_max".into(), "7".into());
        assert_eq!(TextService::from_report(&fields).unwrap(), Some(TextService::resident()));
        fields.remove("text_kind");
        fields.insert("text_sparks".into(), "1".into());
        fields.insert("text_draft_default".into(), "8".into());
        fields.insert("text_draft_max".into(), "8".into());
        assert!(TextService::from_report(&fields).unwrap_err().contains("does not know"));
        fields.insert("text_draft_max".into(), "7".into());
        fields.insert("text_draft_default".into(), "3".into());
        assert!(TextService::from_report(&fields).unwrap_err().contains("does not know"));
    }
}
