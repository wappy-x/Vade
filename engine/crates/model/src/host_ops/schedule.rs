//! Host FFN schedule from config fields. No model name selects a layer kind.
use crate::{Result, invalid, json::{self, Value}, shapes::{CONFIG_BYTES, MAX_EXPERTS, MAX_LAYERS}};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind { Dense, Experts }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfnSchedule {
    /// Layer kinds by absolute layer index. `from_config` gives decoder layers only.
    /// The crate-internal `from_object` appends draft layers after them at absolute indices.
    pub layers: Vec<FfnKind>,
}

impl FfnSchedule {
    /// Read a text config or a root with `text_config`. The result has decoder layers only.
    /// An explicit list takes priority over `first_k_dense_replace`.
    /// Without a list, the prefix is dense. Later layers use experts if present.
    /// An absent or null expert count means no experts in that fallback.
    /// An explicit list states layer kinds without requiring expert geometry.
    /// Source: transformers f90e93c4f83974f634860e246cdfd2c68666d605,
    /// modeling_glm5_next.py and modeling_deepseek_v3.py.
    pub fn from_config(config_json: &[u8]) -> Result<Self> {
        let doc = json::parse(config_json, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = match root.get("text_config") { Some(v) => v.object()?, None => root };
        Self::from_object(o, 0)
    }

    /// Use the same decoder rules, then append draft layers at absolute indices.
    pub(crate) fn from_object(o: &BTreeMap<String, Value>, draft: u64) -> Result<Self> {
        let count = number(o, "num_hidden_layers")?.ok_or_else(|| invalid("missing num_hidden_layers"))?;
        if count == 0 || count > MAX_LAYERS { return Err(invalid("num_hidden_layers is zero or above its limit")); }
        if draft > MAX_LAYERS { return Err(invalid("draft layer count is above its limit")); }
        let prefix = number(o, "first_k_dense_replace")?.unwrap_or(0);
        if prefix > count { return Err(invalid("first_k_dense_replace exceeds num_hidden_layers")); }
        let experts = expert_count(o)?;
        let inferred = |i, has_experts: bool| if i < prefix || !has_experts { FfnKind::Dense } else { FfnKind::Experts };
        let mut layers = if let Some(value) = o.get("mlp_layer_types").filter(|v| **v != Value::Null) {
            let types = value.array()?;
            if types.len() as u64 != count { return Err(invalid("mlp_layer_types length differs from num_hidden_layers")); }
            types.iter().map(|v| match v.string()? {
                "dense" => Ok(FfnKind::Dense), "sparse" => Ok(FfnKind::Experts),
                _ => Err(invalid("mlp_layer_types must contain dense or sparse")),
            }).collect::<Result<Vec<_>>>()?
        } else {
            (0..count).map(|i| inferred(i, experts.unwrap_or(0) != 0)).collect()
        };
        if experts == Some(0) && layers.contains(&FfnKind::Experts) {
            return Err(invalid("sparse layer requires a positive expert count"));
        }
        // A list can state expert layers before expert geometry is available.
        let has_experts = experts.map_or_else(|| layers.contains(&FfnKind::Experts), |n| n != 0);
        layers.extend((count..count + draft).map(|i| inferred(i, has_experts)));
        Ok(Self { layers })
    }
}

fn number(o: &BTreeMap<String, Value>, field: &str) -> Result<Option<u64>> {
    o.get(field).map(|v| v.u64().map_err(|_| invalid(format!("{field} must be an unsigned integer")))).transpose()
}

/// The one expert-count reader for the router and the schedule.
/// The public config source maps `num_local_experts` to `n_routed_experts`.
/// A null field counts as absent. Present counts must agree.
pub(super) fn expert_count(o: &BTreeMap<String, Value>) -> Result<Option<u64>> {
    let mut count = None;
    for field in ["num_experts", "n_routed_experts", "num_local_experts"] {
        let Some(value) = o.get(field).filter(|v| **v != Value::Null) else { continue };
        let value = value.u64().map_err(|_| invalid(format!("{field} must be an unsigned integer")))?;
        if value > MAX_EXPERTS { return Err(invalid(format!("{field} is above its limit"))); }
        if count.is_some_and(|n| n != value) { return Err(invalid("expert counts differ")); }
        count = Some(value);
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use FfnKind::{Dense, Experts};

    #[test]
    fn dense_prefix_golden() {
        let schedule = FfnSchedule::from_config(br#"{"num_hidden_layers":5,"n_routed_experts":4,"first_k_dense_replace":2}"#).unwrap();
        assert_eq!(schedule.layers, [Dense, Dense, Experts, Experts, Experts]);
    }

    #[test]
    fn explicit_schedule_golden_and_prefix_override() {
        let schedule = FfnSchedule::from_config(br#"{"text_config":{"num_hidden_layers":5,"n_routed_experts":4,"first_k_dense_replace":2,"mlp_layer_types":["sparse","dense","sparse","dense","dense"]}}"#).unwrap();
        assert_eq!(schedule.layers, [Experts, Dense, Experts, Dense, Dense]);
        let explicit = FfnSchedule::from_config(br#"{"num_hidden_layers":2,"mlp_layer_types":["dense","sparse"]}"#).unwrap();
        assert_eq!(explicit.layers, [Dense, Experts]);
    }

    #[test]
    fn absent_schedule_preserves_uniform_layers() {
        for text in [
            br#"{"num_hidden_layers":3}"#.as_slice(),
            br#"{"num_hidden_layers":3,"n_routed_experts":null}"#,
            br#"{"num_hidden_layers":3,"num_experts":0}"#,
        ] { assert_eq!(FfnSchedule::from_config(text).unwrap().layers, [Dense; 3]); }
        for field in ["num_experts", "n_routed_experts"] {
            let text = format!(r#"{{"num_hidden_layers":3,"{field}":4,"mlp_layer_types":null}}"#);
            assert_eq!(FfnSchedule::from_config(text.as_bytes()).unwrap().layers, [Experts; 3]);
        }
    }

    #[test]
    fn current_checkpoint_schedules_stay_uniform() {
        for bytes in [
            include_bytes!("../../tests/fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice(),
            include_bytes!("../../tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice(),
        ] {
            let schedule = FfnSchedule::from_config(bytes).unwrap();
            let doc = json::parse(bytes, json::Limits::default()).unwrap();
            let declared = doc.object().unwrap()["text_config"].object().unwrap()["num_hidden_layers"].u64().unwrap();
            assert_eq!(schedule.layers.len() as u64, declared);
            assert!(schedule.layers.iter().all(|&kind| kind == Experts));
        }
    }

    #[test]
    fn prefix_boundaries_are_exact() {
        for (prefix, expected) in [(0, vec![Experts; 3]), (3, vec![Dense; 3])] {
            let text = format!(r#"{{"num_hidden_layers":3,"n_routed_experts":4,"first_k_dense_replace":{prefix}}}"#);
            assert_eq!(FfnSchedule::from_config(text.as_bytes()).unwrap().layers, expected);
        }
        let maximum = format!(r#"{{"num_hidden_layers":{MAX_LAYERS}}}"#);
        assert_eq!(FfnSchedule::from_config(maximum.as_bytes()).unwrap().layers.len(), MAX_LAYERS as usize);
    }

    #[test]
    fn invalid_schedule_is_rejected() {
        for text in [
            r#"{}"#, r#"{"text_config":null}"#, r#"{"num_hidden_layers":0}"#,
            r#"{"num_hidden_layers":1025}"#, r#"{"num_hidden_layers":1.0}"#,
            r#"{"num_hidden_layers":2,"first_k_dense_replace":3}"#,
            r#"{"num_hidden_layers":2,"first_k_dense_replace":-1}"#,
            r#"{"num_hidden_layers":2,"first_k_dense_replace":null}"#,
            r#"{"num_hidden_layers":2,"mlp_layer_types":[]}"#,
            r#"{"num_hidden_layers":1,"mlp_layer_types":"dense"}"#,
            r#"{"num_hidden_layers":1,"mlp_layer_types":[0]}"#,
            r#"{"num_hidden_layers":1,"mlp_layer_types":["moe"]}"#,
            r#"{"num_hidden_layers":1,"num_experts":1025}"#,
            r#"{"num_hidden_layers":1,"num_experts":-1}"#,
            r#"{"num_hidden_layers":1,"num_experts":4,"n_routed_experts":8}"#,
            r#"{"num_hidden_layers":1,"num_experts":0,"mlp_layer_types":["sparse"]}"#,
        ] { assert!(FfnSchedule::from_config(text.as_bytes()).is_err(), "{text}"); }
    }
    #[test]
    fn explicit_list_is_authoritative_without_model_dispatch() {
        let schedule = FfnSchedule::from_config(br#"{"model_type":"unrelated","text_config":{"num_hidden_layers":3,"first_k_dense_replace":2,"mlp_layer_types":["sparse","dense","sparse"]}}"#).unwrap();
        assert_eq!(schedule.layers, [Experts, Dense, Experts]);
    }
    #[test]
    fn uniform_lists_and_layer_boundaries() {
        for (name, kind) in [("dense", Dense), ("sparse", Experts)] {
            for count in [1, MAX_LAYERS] {
                let list = std::iter::repeat_n(format!("\"{name}\""), count as usize).collect::<Vec<_>>().join(",");
                let text = format!(r#"{{"num_hidden_layers":{count},"mlp_layer_types":[{list}]}}"#);
                assert_eq!(FfnSchedule::from_config(text.as_bytes()).unwrap().layers, vec![kind; count as usize]);
            }
        }
    }
    #[test]
    fn schedule_and_router_share_the_expert_count() {
        let schedule = |t: &str| FfnSchedule::from_config(format!(r#"{{"num_hidden_layers":2,{t}}}"#).as_bytes());
        let router = |t: &str| crate::host_ops::Router::from_config(format!(r#"{{"num_experts_per_tok":2,{t}}}"#).as_bytes());
        let local = r#""num_local_experts":4"#;
        assert_eq!(schedule(local).unwrap().layers, [Experts; 2]);
        assert_eq!(router(local).unwrap(), router(r#""n_routed_experts":4"#).unwrap());
        for conflict in [r#""num_local_experts":8,"n_routed_experts":4"#, r#""num_local_experts":8,"num_experts":4"#] {
            assert!(schedule(conflict).is_err(), "{conflict}");
            assert!(router(conflict).is_err(), "{conflict}");
        }
        assert_eq!(schedule(r#""num_local_experts":4,"n_routed_experts":4"#).unwrap().layers, [Experts; 2]);
    }
}
