//! Corrected sigmoid routing on one row of projected f32 logits.
//! Source: `Glm5NextTextTopkRouter`, local public modeling source, lines 143-182.
//! ROUTING.md records the source hash, fixed order, and supported limits.
//! This is the deferred layer oracle. `host_ops::Router` keeps config defaults.
use crate::{Result, invalid, json::{self, Value}, shapes::{CONFIG_BYTES, MAX_EXPERTS}};
use std::collections::BTreeMap;

/// Operation choice comes from config fields, without a model-name test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterKind { SoftmaxTopK, CorrectedSigmoid }
impl RouterKind {
    pub fn from_selectors(score: &str, selection: &str) -> Option<Self> {
        match (score, selection) {
            ("sigmoid", "noaux_tc") => Some(Self::CorrectedSigmoid),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Router {
    kind: RouterKind,
    experts: usize,
    selected: usize,
    groups: usize,
    selected_groups: usize,
    normalize: bool,
    scale: f32,
}

impl Router {
    /// Read a text config or a root with `text_config`.
    /// Require explicit `sigmoid` and `noaux_tc` selectors and operation fields.
    /// The saved public config labels the source formula with these selectors.
    /// No model name or missing-field default selects an operation.
    pub fn from_config(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = match root.get("text_config") { Some(v) => v.object()?, None => root };
        if RouterKind::from_selectors(field(o, "scoring_func")?.string()?, field(o, "topk_method")?.string()?) != Some(RouterKind::CorrectedSigmoid) {
            return Err(invalid("host router requires scoring_func sigmoid and topk_method noaux_tc"));
        }
        let experts = expert_count(o)?;
        let selected = count(o, "num_experts_per_tok")?;
        let groups = count(o, "n_group")?;
        let selected_groups = count(o, "topk_group")?;
        if groups > experts || experts % groups != 0 {
            return Err(invalid("n_group must divide the expert count"));
        }
        let width = experts / groups;
        if width < 2 { return Err(invalid("corrected sigmoid routing requires at least two experts per group")); }
        if selected_groups > groups { return Err(invalid("topk_group exceeds n_group")); }
        if selected > selected_groups * width { return Err(invalid("num_experts_per_tok exceeds the selected groups")); }
        let normalize = field(o, "norm_topk_prob")?.boolean()?;
        let scale = match field(o, "routed_scaling_factor")? {
            Value::Number(v) => v.parse::<f32>().map_err(|_| invalid("routed_scaling_factor must fit f32"))?,
            _ => return Err(invalid("routed_scaling_factor must be a number")),
        };
        if !scale.is_finite() || scale <= 0.0 { return Err(invalid("routed_scaling_factor must be positive and finite in f32")); }
        Ok(Self { kind: RouterKind::CorrectedSigmoid, experts, selected, groups, selected_groups, normalize, scale })
    }

    /// Bind the deferred operation data. This does not admit device execution.
    pub fn from_description(d: &crate::plan::description::RoutingDescription, e: crate::shapes::ExpertShape) -> Result<Self> {
        let (experts, selected, groups, selected_groups) = (e.count, e.selected, d.groups, d.selected_groups);
        if d.kind != RouterKind::CorrectedSigmoid || experts == 0 || experts > MAX_EXPERTS || groups == 0
            || groups > experts || experts % groups != 0 || experts / groups < 2
            || selected_groups == 0 || selected_groups > groups || selected == 0 || selected > selected_groups * (experts / groups) {
            return Err(invalid("invalid corrected sigmoid routing description"));
        }
        let scale = d.scale as f32;
        if !scale.is_finite() || scale <= 0.0 { return Err(invalid("routed_scaling_factor must be positive and finite in f32")); }
        Ok(Self { kind: d.kind, experts: experts as usize, selected: selected as usize, groups: groups as usize,
            selected_groups: selected_groups as usize, normalize: d.normalize_selected, scale })
    }
    /// Use the existing normalized softmax host reference without new arithmetic.
    pub fn softmax(experts: usize, selected: usize) -> Result<Self> {
        if experts == 0 || experts > MAX_EXPERTS as usize || selected == 0 || selected > experts {
            return Err(invalid("invalid softmax routing dimensions"));
        }
        Ok(Self { kind: RouterKind::SoftmaxTopK, experts, selected, groups: 1, selected_groups: 1, normalize: true, scale: 1.0 })
    }
    pub fn kind(&self) -> RouterKind { self.kind }
    pub fn experts(&self) -> usize { self.experts }
    pub fn selected(&self) -> usize { self.selected }
    pub fn groups(&self) -> usize { self.groups }
    pub fn selected_groups(&self) -> usize { self.selected_groups }
    pub fn normalize(&self) -> bool { self.normalize }
    pub fn scale(&self) -> f32 { self.scale }

    /// Return expert IDs and weights for one row. Bias affects selection only.
    /// Group scores add the largest corrected score to the next largest score.
    /// Groups and experts sort by descending score, then lower ID for ties.
    /// The weight sum follows the returned expert order, from left to right.
    /// Add the source epsilon after that sum. Divide before routed scaling.
    /// The source does not fix top-k tie order. This reference fixes it here.
    /// These f32 operations do not claim bit parity with a device library.
    pub fn route(&self, logits: &[f32], correction_bias: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        if logits.len() != self.experts || logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("router logits must match the expert count and be finite"));
        }
        if self.kind == RouterKind::SoftmaxTopK {
            if !correction_bias.is_empty() { return Err(invalid("softmax routing does not use correction bias")); }
            return Ok(crate::flash::refcpu::route(logits, self.selected));
        }
        if correction_bias.len() != self.experts || correction_bias.iter().any(|v| !v.is_finite()) {
            return Err(invalid("e_score_correction_bias must match the expert count and be finite"));
        }
        let scores: Vec<f32> = logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        let choice: Vec<f32> = scores.iter().zip(correction_bias).map(|(&s, &b)| s + b).collect();
        if choice.iter().any(|v| !v.is_finite()) { return Err(invalid("corrected router score exceeds f32 range")); }
        let width = self.experts / self.groups;
        let mut group_scores = Vec::with_capacity(self.groups);
        for group in 0..self.groups {
            let mut ids: Vec<usize> = (group * width..(group + 1) * width).collect();
            descending(&mut ids, &choice);
            let sum = choice[ids[0]] + choice[ids[1]];
            if !sum.is_finite() { return Err(invalid("router group score exceeds f32 range")); }
            group_scores.push(sum);
        }
        let mut group_ids: Vec<usize> = (0..self.groups).collect();
        descending(&mut group_ids, &group_scores);
        let mut keep = vec![false; self.groups];
        for &group in &group_ids[..self.selected_groups] { keep[group] = true; }
        // Removing excluded IDs is the source's negative-infinity mask.
        // A zero mask would fail when kept corrected scores are negative.
        let mut ids: Vec<usize> = (0..self.experts).filter(|&id| keep[id / width]).collect();
        descending(&mut ids, &choice);
        ids.truncate(self.selected);
        let mut weights: Vec<f32> = ids.iter().map(|&id| scores[id]).collect();
        if self.normalize {
            let mut sum = 0.0f32;
            for &weight in &weights { sum += weight; }
            let denominator = sum + 1e-20f32;
            for weight in &mut weights { *weight /= denominator; }
        }
        for weight in &mut weights { *weight *= self.scale; }
        if weights.iter().any(|v| !v.is_finite()) { return Err(invalid("router weight exceeds f32 range")); }
        Ok((ids, weights))
    }
}

fn descending(ids: &mut [usize], scores: &[f32]) {
    // Treat signed zeros as equal scores. All scores were checked as finite.
    ids.sort_by(|&a, &b| {
        if scores[a] == scores[b] { a.cmp(&b) } else { scores[b].total_cmp(&scores[a]) }
    });
}
fn field<'a>(o: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a Value> {
    o.get(key).ok_or_else(|| invalid(format!("missing router field {key}; defaults are not inferred")))
}
fn count(o: &BTreeMap<String, Value>, key: &str) -> Result<usize> {
    let n = field(o, key)?.u64()?;
    if n == 0 || n > MAX_EXPERTS { return Err(invalid(format!("{key} is zero or above its limit"))); }
    Ok(n as usize)
}
fn expert_count(o: &BTreeMap<String, Value>) -> Result<usize> {
    let mut result = None;
    // The public config source maps num_local_experts to n_routed_experts.
    for key in ["n_routed_experts", "num_local_experts"] {
        if !o.contains_key(key) { continue; }
        let n = count(o, key)?;
        if result.is_some_and(|old| old != n) { return Err(invalid("router expert counts differ")); }
        result = Some(n);
    }
    result.ok_or_else(|| invalid("missing n_routed_experts or num_local_experts"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(experts: usize, selected: usize, groups: usize, keep: usize, normalize: bool) -> String {
        format!(r#"{{"n_routed_experts":{experts},"num_experts_per_tok":{selected},"n_group":{groups},"topk_group":{keep},"norm_topk_prob":{normalize},"routed_scaling_factor":2.5,"scoring_func":"sigmoid","topk_method":"noaux_tc"}}"#)
    }
    fn router(experts: usize, selected: usize, groups: usize, keep: usize, normalize: bool) -> Router {
        Router::from_config(config(experts, selected, groups, keep, normalize).as_bytes()).unwrap()
    }
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() < 1e-6, "{a} != {b}"); }
    }
    #[test]
    fn correction_selects_but_does_not_weight() {
        // Sigmoid scores: [3/4,1/2,1/4,1/2]. Corrected: [1/4,1/2,1,5/8].
        // Group sums: [3/4,13/8]. IDs [2,3] get weights [1/4,1/2], then / (3/4) * (5/2).
        let logits = [3.0f32.ln(), 0.0, -3.0f32.ln(), 0.0];
        let (ids, weights) = router(4, 2, 2, 1, true).route(&logits, &[-0.5, 0.0, 0.75, 0.125]).unwrap();
        assert_eq!(ids, [2, 3]); close(&weights, &[5.0 / 6.0, 5.0 / 3.0]);
    }
    #[test]
    fn group_score_adds_only_the_largest_pair() {
        // Corrected groups: [1,1/4,1/4], [3/4,3/4,-1]. Pair sums: [5/4,3/2].
        // A group max or a full-group sum would select the wrong group.
        let (ids, weights) = router(6, 2, 2, 1, true).route(&[0.0; 6], &[0.5, -0.25, -0.25, 0.25, 0.25, -1.5]).unwrap();
        assert_eq!(ids, [3, 4]); assert_eq!(weights, [1.25, 1.25]);
    }
    #[test]
    fn mask_excludes_groups_even_when_kept_scores_are_negative() {
        let (ids, weights) = router(4, 2, 2, 1, true).route(&[0.0; 4], &[-2.0, -2.0, -1.0, -1.0]).unwrap();
        assert_eq!(ids, [2, 3]); assert_eq!(weights, [1.25, 1.25]);
    }
    #[test]
    fn ties_use_lower_group_then_lower_expert_id() {
        let op = router(6, 3, 3, 2, true);
        let expected = op.route(&[0.0; 6], &[0.0; 6]).unwrap();
        assert_eq!(expected.0, [0, 1, 2]); close(&expected.1, &[5.0 / 6.0; 3]);
        for _ in 0..3 { assert_eq!(op.route(&[0.0; 6], &[0.0; 6]).unwrap(), expected); }
        let mut ids = [2, 1, 0];
        descending(&mut ids, &[-0.0, 0.0, -0.0]);
        assert_eq!(ids, [0, 1, 2]);
    }
    #[test]
    fn expert_selection_is_global_within_kept_groups() {
        // Corrected groups: [2,7/4], [3/2,1/2], [1,1]. Keep groups [0,1].
        // Selected IDs are [0,1,2], with no per-group expert quota.
        let (ids, weights) = router(6, 3, 3, 2, false).route(&[0.0; 6], &[1.5, 1.25, 1.0, 0.0, 0.5, 0.5]).unwrap();
        assert_eq!(ids, [0, 1, 2]); assert_eq!(weights, [1.25; 3]);
    }
    #[test]
    #[allow(clippy::excessive_precision)] // literal constants stay as written.
    fn normalization_scaling_and_epsilon_are_separate() {
        let logits = [3.0f32.ln(), 0.0];
        close(&router(2, 2, 1, 1, false).route(&logits, &[0.0; 2]).unwrap().1, &[1.875, 1.25]);
        close(&router(2, 2, 1, 1, true).route(&logits, &[0.0; 2]).unwrap().1, &[1.5, 1.0]);
        assert_eq!(router(2, 1, 1, 1, true).route(&[0.0; 2], &[0.0; 2]).unwrap().1, [2.5]);
        assert_eq!(router(2, 2, 1, 1, true).route(&[-1000.0; 2], &[0.0; 2]).unwrap().1, [0.0; 2]);
        // sigmoid(-50) is about 1.92874985e-22. The epsilon must affect this weight.
        close(&router(2, 1, 1, 1, true).route(&[-50.0, -60.0], &[0.0; 2]).unwrap().1, &[0.0473063255]);
        let scale = config(2, 1, 1, 1, false).replace("2.5", "0.5");
        assert_eq!(Router::from_config(scale.as_bytes()).unwrap().route(&[0.0; 2], &[0.0; 2]).unwrap().1, [0.25]);
    }
    #[test]
    fn config_is_model_generic_and_accepts_the_public_alias() {
        let bytes = config(4, 2, 1, 1, true);
        let plain = Router::from_config(bytes.as_bytes()).unwrap();
        let wrapped = format!(r#"{{"model_type":"unrelated","text_config":{bytes}}}"#);
        assert_eq!(plain, Router::from_config(wrapped.as_bytes()).unwrap());
        assert_eq!(plain, Router::from_config(bytes.replace("n_routed_experts", "num_local_experts").as_bytes()).unwrap());
    }
    #[test]
    fn config_rejects_invalid_counts_and_unknown_rules() {
        for (experts, top, groups, keep) in [(0, 1, 1, 1), (4, 0, 1, 1), (4, 5, 1, 1), (4, 1, 0, 1),
            (4, 1, 3, 1), (4, 1, 2, 3), (4, 3, 2, 1), (4, 1, 4, 1), (1025, 1, 1, 1)] {
            assert!(Router::from_config(config(experts, top, groups, keep, true).as_bytes()).is_err());
        }
        let base = config(4, 2, 1, 1, true);
        for (from, to) in [("sigmoid", "softmax"), ("noaux_tc", "greedy"), ("2.5", "0"), ("2.5", "-1"),
            ("2.5", "1e40"), ("2.5", "1e-60"), ("2.5", "null"), ("true", "null"), ("\"n_group\":1", "\"n_group\":1.0")] {
            assert!(Router::from_config(base.replace(from, to).as_bytes()).is_err(), "{from} -> {to}");
        }
        let conflict = base.replace("\"n_routed_experts\":4", "\"n_routed_experts\":4,\"num_local_experts\":8");
        assert!(Router::from_config(conflict.as_bytes()).is_err());
        for key in ["scoring_func", "topk_method", "n_group", "topk_group", "norm_topk_prob", "routed_scaling_factor"] {
            assert!(Router::from_config(base.replace(key, "unknown_field").as_bytes()).is_err());
        }
        for bytes in ["{}", "[]", r#"{"text_config":null}"#] { assert!(Router::from_config(bytes.as_bytes()).is_err()); }
    }
    #[test]
    fn route_rejects_invalid_shapes_nonfinite_values_and_overflow() {
        let op = router(4, 2, 1, 1, true);
        assert!(op.route(&[0.0; 3], &[0.0; 4]).is_err());
        assert!(op.route(&[0.0; 4], &[0.0; 3]).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(op.route(&[bad; 4], &[0.0; 4]).is_err());
            assert!(op.route(&[0.0; 4], &[bad; 4]).is_err());
        }
        assert!(op.route(&[0.0; 4], &[f32::MAX; 4]).is_err());
        assert!(op.route(&[0.0; 4], &[-f32::MAX; 4]).is_err());
    }
}
