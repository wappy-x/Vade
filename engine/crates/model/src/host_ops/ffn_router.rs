//! Host router from scoring, selection, grouping and scale fields.
//! Config-default formula reference. See `host_ops/README.md` for oracle selection.
//! The description-bound `host_ops::router::Router` has a different input contract.
//! The corrected sigmoid formula follows `Glm5NextTextTopkRouter` in
//! transformers f90e93c4f83974f634860e246cdfd2c68666d605.
//! See <https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/glm5_next/modeling_glm5_next.py>.
use crate::{Result, invalid, json::{self, Value}, shapes::{CONFIG_BYTES, MAX_EXPERTS}};
use crate::flash::refcpu::{sigmoid, softmax};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method { Softmax, CorrectedSigmoid }

#[derive(Debug, Clone, PartialEq)]
pub struct Router {
    method: Method,
    experts: usize,
    selected: usize,
    groups: usize,
    selected_groups: usize,
    normalize: bool,
    scale: f32,
}

impl Router {
    /// Read a text config or a root with `text_config`.
    /// Missing scoring and method fields select softmax and greedy top-k.
    /// `sigmoid` with `noaux_tc` requires a correction tensor at each call.
    pub fn from_config(config_json: &[u8]) -> Result<Self> {
        let doc = json::parse(config_json, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = match root.get("text_config") { Some(v) => v.object()?, None => root };
        let method = match (text(o, "scoring_func", "softmax")?, text(o, "topk_method", "greedy")?) {
            ("softmax", "greedy") => Method::Softmax,
            ("sigmoid", "noaux_tc") => Method::CorrectedSigmoid,
            _ => return Err(invalid("unsupported scoring_func and topk_method pair")),
        };
        let experts = expert_count(o)?;
        let selected = count(o, "num_experts_per_tok", None)?;
        let groups = count(o, "n_group", Some(1))?;
        let selected_groups = count(o, "topk_group", Some(1))?;
        if selected > experts { return Err(invalid("num_experts_per_tok exceeds the expert count")); }
        if groups > experts || experts % groups != 0 || selected_groups > groups {
            return Err(invalid("router groups do not divide the expert count or topk_group exceeds n_group"));
        }
        let group_width = experts / groups;
        if selected > selected_groups * group_width {
            return Err(invalid("num_experts_per_tok exceeds the selected groups"));
        }
        if method == Method::CorrectedSigmoid && group_width < 2 {
            return Err(invalid("noaux_tc requires at least two experts per group"));
        }
        if method == Method::Softmax && (groups != 1 || selected_groups != 1) {
            return Err(invalid("softmax greedy routing does not use groups"));
        }
        let normalize = o.get("norm_topk_prob").map(Value::boolean).transpose()?.unwrap_or(true);
        let scale = match o.get("routed_scaling_factor") {
            Some(Value::Number(v)) => v.parse::<f32>().map_err(|_| invalid("routed_scaling_factor must fit f32"))?,
            Some(_) => return Err(invalid("routed_scaling_factor must be a number")), None => 1.0,
        };
        if !scale.is_finite() || scale <= 0.0 { return Err(invalid("routed_scaling_factor must be positive and finite in f32")); }
        Ok(Self { method, experts, selected, groups, selected_groups, normalize, scale })
    }

    /// Return expert IDs and weights for one row of finite f32 logits.
    /// The bias affects selection only. It must be absent for softmax.
    /// Groups and experts sort by descending selection score, then lower ID.
    /// The normalization sum follows that expert order. Group scores add the
    /// first score to the second score. This order does not depend on batching.
    /// Torch top-k leaves tie order unspecified. This reference fixes that order.
    pub fn route(&self, logits: &[f32], correction_bias: Option<&[f32]>) -> Result<(Vec<usize>, Vec<f32>)> {
        if logits.len() != self.experts || logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("router logits must match the expert count and be finite"));
        }
        let mut scores = vec![0.0; self.experts];
        let mut order: Vec<usize> = (0..self.experts).collect();
        match self.method {
            Method::Softmax => {
                if correction_bias.is_some() { return Err(invalid("softmax routing does not use a correction bias")); }
                softmax(logits, &mut scores);
                descending(&mut order, &scores);
            }
            Method::CorrectedSigmoid => {
                let bias = correction_bias.ok_or_else(|| invalid("noaux_tc requires e_score_correction_bias"))?;
                if bias.len() != self.experts || bias.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("e_score_correction_bias must match the expert count and be finite"));
                }
                for (score, &logit) in scores.iter_mut().zip(logits) { *score = sigmoid(logit); }
                let choice = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect::<Vec<_>>();
                if choice.iter().any(|v| !v.is_finite()) { return Err(invalid("corrected router score exceeds f32 range")); }
                let width = self.experts / self.groups;
                let mut group_scores = Vec::with_capacity(self.groups);
                for group in 0..self.groups {
                    let mut ids = (group * width..(group + 1) * width).collect::<Vec<_>>();
                    descending(&mut ids, &choice);
                    let score = choice[ids[0]] + choice[ids[1]];
                    if !score.is_finite() { return Err(invalid("router group score exceeds f32 range")); }
                    group_scores.push(score);
                }
                let mut groups = (0..self.groups).collect::<Vec<_>>();
                descending(&mut groups, &group_scores);
                let mut keep = vec![false; self.groups];
                for &group in &groups[..self.selected_groups] { keep[group] = true; }
                // Exclude groups before sorting. A zero mask would admit excluded
                // experts when every corrected score in a kept group is negative.
                order.retain(|&id| keep[id / width]);
                descending(&mut order, &choice);
            }
        }
        order.truncate(self.selected);
        let mut weights = order.iter().map(|&id| scores[id]).collect::<Vec<_>>();
        if self.normalize {
            let sum = match self.method {
                Method::Softmax => weights.iter().copied().sum::<f32>(),
                Method::CorrectedSigmoid => {
                    let mut sum = 0.0f32;
                    for &weight in &weights { sum += weight; }
                    sum + 1e-20
                }
            };
            for weight in &mut weights { *weight /= sum; }
        }
        for weight in &mut weights { *weight *= self.scale; }
        if weights.iter().any(|v| !v.is_finite()) { return Err(invalid("router weight exceeds f32 range")); }
        Ok((order, weights))
    }
}

fn descending(ids: &mut [usize], scores: &[f32]) {
    ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
}
fn text<'a>(o: &'a BTreeMap<String, Value>, field: &str, default: &'a str) -> Result<&'a str> {
    o.get(field).map(Value::string).transpose().map(|v| v.unwrap_or(default))
}
fn count(o: &BTreeMap<String, Value>, field: &str, default: Option<usize>) -> Result<usize> {
    let value = match o.get(field) {
        Some(v) => v.u64()?, None => return default.ok_or_else(|| invalid(format!("missing {field}"))),
    };
    if value == 0 || value > MAX_EXPERTS { return Err(invalid(format!("{field} is zero or above its limit"))); }
    Ok(value as usize)
}
fn expert_count(o: &BTreeMap<String, Value>) -> Result<usize> {
    match super::schedule::expert_count(o)? {
        None => Err(invalid("missing num_experts, n_routed_experts or num_local_experts")),
        Some(0) => Err(invalid("expert count is zero")),
        Some(n) => Ok(n as usize),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(experts: usize, top: usize, groups: usize, keep: usize, normalize: bool) -> String {
        format!(r#"{{"n_routed_experts":{experts},"num_experts_per_tok":{top},"scoring_func":"sigmoid","topk_method":"noaux_tc","n_group":{groups},"topk_group":{keep},"norm_topk_prob":{normalize},"routed_scaling_factor":2.5}}"#)
    }
    fn router(experts: usize, top: usize, groups: usize, keep: usize, normalize: bool) -> Router {
        Router::from_config(config(experts, top, groups, keep, normalize).as_bytes()).unwrap()
    }
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() < 1e-6, "{a} != {b}"); }
    }
    #[test]
    fn corrected_sigmoid_weights_use_original_scores() {
        // Scores are [3/4, 1/2, 1/4, 1/2]. Corrected group sums are 3/4 and 13/8.
        // Select IDs [2,3]. Normalize [1/4,1/2], then scale by 5/2.
        let logits = [3.0f32.ln(), 0.0, -3.0f32.ln(), 0.0];
        let bias = [-0.5, 0.0, 0.75, 0.125];
        let (ids, weights) = router(4, 2, 2, 1, true).route(&logits, Some(&bias)).unwrap();
        assert_eq!(ids, [2, 3]);
        close(&weights, &[5.0 / 6.0, 5.0 / 3.0]);
    }
    #[test]
    fn group_score_uses_two_largest_corrected_scores() {
        // Corrected scores: [1,1/4,1/4], [3/4,3/4,-1].
        // Top-two sums: 5/4 and 3/2. A max or a full-group sum picks the wrong group.
        let (ids, weights) = router(6, 2, 2, 1, true).route(&[0.0; 6], Some(&[0.5, -0.25, -0.25, 0.25, 0.25, -1.5])).unwrap();
        assert_eq!(ids, [3, 4]); assert_eq!(weights, [1.25, 1.25]);
    }
    #[test]
    fn excluded_groups_cannot_win_against_negative_scores() {
        let (ids, weights) = router(4, 2, 2, 1, true).route(&[0.0; 4], Some(&[-2.0, -2.0, -1.0, -1.0])).unwrap();
        assert_eq!(ids, [2, 3]); assert_eq!(weights, [1.25, 1.25]);
    }
    #[test]
    fn ties_use_lower_group_and_expert_ids() {
        let config = router(6, 3, 3, 2, true);
        let expected = config.route(&[0.0; 6], Some(&[0.0; 6])).unwrap();
        assert_eq!(expected.0, [0, 1, 2]); close(&expected.1, &[5.0 / 6.0; 3]);
        for _ in 0..3 { assert_eq!(config.route(&[0.0; 6], Some(&[0.0; 6])).unwrap(), expected); }
    }
    #[test]
    fn normalization_and_scaling_are_separate() {
        let bias = [0.0; 2];
        assert_eq!(router(2, 1, 1, 1, false).route(&[0.0; 2], Some(&bias)).unwrap(), (vec![0], vec![1.25]));
        assert_eq!(router(2, 1, 1, 1, true).route(&[0.0; 2], Some(&bias)).unwrap(), (vec![0], vec![2.5]));
        // Underflowed sigmoid scores remain zero after the source's +1e-20 divisor.
        assert_eq!(router(2, 2, 1, 1, true).route(&[-1000.0; 2], Some(&bias)).unwrap().1, [0.0; 2]);
        let small = router(2, 1, 1, 1, true).route(&[-50.0, -60.0], Some(&bias)).unwrap().1;
        close(&small, &[0.0473063255]); // 2.5 * sigmoid(-50) / (sigmoid(-50) + 1e-20).
    }
    #[test]
    fn expert_selection_is_global_within_kept_groups() {
        // Corrected groups: [2,7/4], [3/2,1/2], [1,1]. Keep groups [0,1].
        // Select IDs [0,1,2]. There is no expert quota for each group.
        let (ids, weights) = router(6, 3, 3, 2, false).route(&[0.0; 6], Some(&[1.5, 1.25, 1.0, 0.0, 0.5, 0.5])).unwrap();
        assert_eq!(ids, [0, 1, 2]); assert_eq!(weights, [1.25; 3]);
    }
    #[test]
    fn public_expert_alias_must_agree() {
        let bytes = config(4, 2, 1, 1, true);
        let plain = Router::from_config(bytes.as_bytes()).unwrap();
        assert_eq!(plain, Router::from_config(bytes.replace("n_routed_experts", "num_local_experts").as_bytes()).unwrap());
        let same = bytes.replace("\"n_routed_experts\":4", "\"n_routed_experts\":4,\"num_local_experts\":4");
        assert_eq!(plain, Router::from_config(same.as_bytes()).unwrap());
        let conflict = same.replace("\"num_local_experts\":4", "\"num_local_experts\":8");
        assert!(Router::from_config(conflict.as_bytes()).is_err());
    }
    #[test]
    fn config_fields_select_behavior_without_model_names() {
        let config = config(4, 2, 1, 1, true);
        let plain = Router::from_config(config.as_bytes()).unwrap();
        let wrapped = format!(r#"{{"model_type":"arbitrary","text_config":{config}}}"#);
        assert_eq!(plain, Router::from_config(wrapped.as_bytes()).unwrap());
        let softmax = Router::from_config(br#"{"num_experts":4,"num_experts_per_tok":2}"#).unwrap();
        let (ids, weights) = softmax.route(&[0.0, 1.0, 2.0, 3.0], None).unwrap();
        assert_eq!(ids, [3, 2]); close(&weights, &[0.7310586, 0.26894143]);
    }
    #[test]
    fn invalid_config_is_rejected() {
        for (experts, top, groups, keep) in [(0, 1, 1, 1), (4, 0, 1, 1), (4, 5, 1, 1), (4, 1, 0, 1),
            (4, 1, 3, 1), (4, 1, 2, 3), (4, 3, 2, 1), (4, 1, 4, 1), (1025, 1, 1, 1)] {
            assert!(Router::from_config(config(experts, top, groups, keep, true).as_bytes()).is_err());
        }
        let base = config(4, 2, 1, 1, true);
        for (from, to) in [("sigmoid", "softmax"), ("noaux_tc", "unknown"), ("2.5", "0"),
            ("2.5", "-1"), ("2.5", "1e40"), ("2.5", "1e-60"), ("true", "null")] {
            assert!(Router::from_config(base.replace(from, to).as_bytes()).is_err());
        }
        assert!(Router::from_config(br#"{"num_experts":4,"n_routed_experts":8,"num_experts_per_tok":2}"#).is_err());
        assert!(Router::from_config(br#"{"num_experts":4,"num_experts_per_tok":2,"n_group":2}"#).is_err());
    }
    #[test]
    fn invalid_inputs_are_rejected() {
        let config = router(4, 2, 1, 1, true);
        assert!(config.route(&[0.0; 4], None).is_err());
        assert!(config.route(&[0.0; 3], Some(&[0.0; 4])).is_err());
        assert!(config.route(&[0.0; 4], Some(&[0.0; 3])).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(config.route(&[invalid; 4], Some(&[0.0; 4])).is_err());
            assert!(config.route(&[0.0; 4], Some(&[invalid; 4])).is_err());
        }
        assert!(config.route(&[0.0; 4], Some(&[f32::MAX; 4])).is_err());
        let plain = Router::from_config(br#"{"num_experts":4,"num_experts_per_tok":2}"#).unwrap();
        assert!(plain.route(&[0.0; 4], Some(&[0.0; 4])).is_err());
    }
}
