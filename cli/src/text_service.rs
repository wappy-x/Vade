//! How many Sparks a checkpoint's text service runs on.
//!
//! The number comes from the checkpoint plan ([`engine_model::plan::ConfigPlan`]),
//! the same way the build manifest answers `supports_fp4_chain`. A folder name
//! is not an answer. A plan this controller cannot read stays the two-Spark wire,
//! so a current-family start is still refused on any other group.
use crate::config::Machine;
use crate::state::TEXT_DRAFT_DEPTH;
use crate::text::Fields;
use std::path::Path;

/// Two selected checkpoints declare different text services.
pub const DIFFERENT_SERVICES: &str = "The selected checkpoints do not declare the same text service. Point every machine at one checkpoint. Nothing started.";

/// The greatest draft depth the diagnostic text CLI accepts (`--mtp 0..7`).
pub const MAX_DIAGNOSTIC_DEPTH: u32 = 7;

/// The text service a checkpoint declares, and the draft depths it accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextService {
    /// Sparks that serve one model together.
    pub sparks: usize,
    /// Depth used when `--diagnostic-unqualified-text` is set and `--mtp` is not.
    pub default_depth: u32,
    /// The greatest depth this service accepts. A named `--mtp` above it is refused.
    pub max_depth: u32,
}

impl TextService {
    /// The wire the current text families run on. Unnamed diagnostic depth is the
    /// GLM text service's depth. The greatest named depth is the diagnostic parse range.
    pub fn pair() -> Self {
        Self { sparks: 2, default_depth: TEXT_DRAFT_DEPTH, max_depth: MAX_DIAGNOSTIC_DEPTH }
    }
    /// One place and no neighbor, with no declared draft model (`max_depth == 0`).
    /// The loop text service refuses any other place: its own words are "start it as
    /// RANK 0 of WORLD 1". Unnamed depth stays 0 so the engine is told `--mtp 0`.
    pub fn alone() -> Self {
        Self::alone_with_max_depth(0)
    }
    /// One place and no neighbor, with the greatest draft depth its checkpoint plan declares.
    /// Unnamed depth stays 0 (`--mtp 0`); named `--mtp 0..=max_depth` is passed through.
    pub fn alone_with_max_depth(max_depth: u32) -> Self {
        Self { sparks: 1, default_depth: 0, max_depth }
    }
    /// Whether this service runs on one Spark with no transport.
    pub fn is_alone(self) -> bool {
        self.sparks == 1
    }
    /// Why `group` cannot start, and the command that can.
    pub fn group_refusal(self) -> String {
        if self.is_alone() {
            "This checkpoint's text service has no transport between Sparks. It runs on one Spark: RANK 0 of WORLD 1. Start it as: vade up --sparks 1 --diagnostic-unqualified-text.".into()
        } else {
            // Main's text, byte for byte: a pair checkpoint on a wrong Spark count is refused as before.
            "--diagnostic-unqualified-text runs on exactly two Sparks. Start it as: vade up --sparks 2 --diagnostic-unqualified-text.".into()
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
    /// The service a worker reported. Any other shape is refused: this controller
    /// only starts the services it knows.
    pub fn from_report(fields: &Fields) -> Result<Self, String> {
        let service = Self {
            sparks: whole(fields, "text_sparks")?,
            default_depth: whole(fields, "text_draft_default")?,
            max_depth: whole(fields, "text_draft_max")?,
        };
        if service == Self::pair()
            || (service.sparks == 1 && service.default_depth == 0 && service.max_depth <= MAX_DIAGNOSTIC_DEPTH)
        {
            Ok(service)
        } else {
            Err("The Spark reported a text service this controller does not know. Run vade install from current source. Nothing started.".into())
        }
    }
}

/// Classify `config.json` bytes. [`ConfigPlan::Loop`](engine_model::plan::ConfigPlan::Loop)
/// is the one-Spark service, with `max_depth` taken from its declared `mtp_config` gamma
/// (`block_size - 1`, capped at [`MAX_DIAGNOSTIC_DEPTH`]) when present, or `0` otherwise.
/// Every other plan, including a plan that does not parse, is the two-Spark wire.
pub fn from_config_bytes(bytes: &[u8]) -> TextService {
    match engine_model::plan::ConfigPlan::from_config(bytes) {
        Ok(engine_model::plan::ConfigPlan::Loop(description)) => {
            let max_depth = description.declared_draft.as_ref()
                .and_then(|_| engine_model::mtp_draft::MtpConfig::from_config_json(bytes).ok())
                .and_then(|cfg| u32::try_from(cfg.gamma()).ok())
                .map(|gamma| gamma.min(MAX_DIAGNOSTIC_DEPTH))
                .unwrap_or(0);
            TextService::alone_with_max_depth(max_depth)
        }
        _ => TextService::pair(),
    }
}

/// Classify `config.json` under `root`, or `None` when it is absent or unreadable.
pub fn from_dir(root: &Path) -> Option<TextService> {
    let bytes = engine_model::safetensors::read_bounded(&root.join("config.json"), engine_model::shapes::CONFIG_BYTES).ok()?;
    Some(from_config_bytes(&bytes))
}

/// Read an absolute `config.json` on this computer. A `~/` path is the Spark's
/// path, not this computer's, and is left unread. No config file is `Ok(None)`.
/// Two selected checkpoints that declare different services are an error.
pub fn from_local(machines: &[Machine]) -> Result<Option<TextService>, String> {
    let mut found = None;
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
        let service = from_config_bytes(&bytes);
        if let Some(previous) = found && previous != service {
            return Err(DIFFERENT_SERVICES.into());
        }
        found = Some(service);
    }
    Ok(found)
}

/// Whether a selected checkpoint whose config.json is on this computer declares the one-Spark service. A file
/// that is absent or cannot be read says nothing here: a start on two Sparks then takes main's steps.
pub fn any_local_alone(machines: &[Machine]) -> bool {
    machines.iter().any(|machine| {
        let root = Path::new(&machine.model_path);
        root.is_absolute() && from_dir(root).is_some_and(TextService::is_alone)
    })
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
    fn the_plan_names_the_service_and_a_failed_plan_stays_the_pair() {
        assert_eq!(from_config_bytes(LOOP_CONFIG.as_bytes()), TextService::alone());
        assert_eq!(from_config_bytes(LOOP_DRAFT_CONFIG.as_bytes()), TextService::alone_with_max_depth(7));
        assert!(TextService::alone_with_max_depth(7).is_alone());
        assert!(!TextService::pair().is_alone());
        assert_eq!(from_config_bytes(br#"{"model_type":"other","architectures":["Inkling2LoopForConditionalGeneration"]}"#), TextService::pair());
        assert_eq!(from_config_bytes(br#"{"model_type":"inkling2_loop"}"#), TextService::pair());
        assert_eq!(from_config_bytes(b"{}"), TextService::pair());
        assert_eq!(TextService::pair().default_depth, TEXT_DRAFT_DEPTH);
        assert!(TextService::alone().draft_refusal().contains("Drafting is not built"));
        assert_eq!(TextService::alone_with_max_depth(7).draft_refusal(), "Set MTP to a whole number from 0 to 7.");
        let mut fields = Fields::new();
        fields.insert("text_sparks".into(), "1".into());
        fields.insert("text_draft_default".into(), "0".into());
        fields.insert("text_draft_max".into(), "0".into());
        assert_eq!(TextService::from_report(&fields).unwrap(), TextService::alone());
        fields.insert("text_draft_max".into(), "7".into());
        assert_eq!(TextService::from_report(&fields).unwrap(), TextService::alone_with_max_depth(7));
        fields.insert("text_draft_max".into(), "8".into());
        assert!(TextService::from_report(&fields).unwrap_err().contains("does not know"));
        fields.insert("text_draft_max".into(), "7".into());
        fields.insert("text_draft_default".into(), "3".into());
        assert!(TextService::from_report(&fields).unwrap_err().contains("does not know"));
    }
}
