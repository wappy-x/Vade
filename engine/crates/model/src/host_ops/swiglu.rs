//! SwiGLU on projected gate and up vectors, with an optional config cap.
//! Logical f32 formula only. Use `crate::swiglu::SwiGlu` for epilogue rounding.
use crate::{Result, invalid, json::{self, Value}, shapes::CONFIG_BYTES};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Swiglu { limit: Option<f32> }

impl Swiglu {
    /// Read a text config or a config with a `text_config` object.
    /// An absent or null `swiglu_limit` keeps the uncapped operation.
    pub fn from_config(config_json: &[u8]) -> Result<Self> {
        let doc = json::parse(config_json, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let config = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        if let Some(act) = config.get("hidden_act") {
            if act.string()? != "silu" { return Err(invalid("SwiGLU requires hidden_act silu")); }
        }
        let limit = match config.get("swiglu_limit") {
            None | Some(Value::Null) => None,
            Some(Value::Number(value)) => {
                let value: f32 = value.parse().map_err(|_| invalid("swiglu_limit must fit f32"))?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(invalid("swiglu_limit must be a finite positive f32"));
                }
                Some(value)
            }
            Some(_) => return Err(invalid("swiglu_limit must be a number or null")),
        };
        Ok(Self { limit })
    }
    pub fn limit(&self) -> Option<f32> { self.limit }

    /// Compute `silu(gate) * up` in f32, one element at a time.
    /// With a cap, gate has an upper bound only. Up has both bounds.
    /// Both clamps run before SiLU. The negative gate has no lower bound.
    /// Source: transformers f90e93c4f83974f634860e246cdfd2c68666d605,
    /// `modeling_glm5_next.py`, `Glm5NextTextMLP` and `Glm5NextTextExperts`.
    pub fn apply(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() { return Err(invalid("SwiGLU gate and up lengths differ")); }
        if !gate.iter().chain(up).all(|v| v.is_finite()) {
            return Err(invalid("SwiGLU gate and up must be finite"));
        }
        Ok(gate.iter().zip(up).map(|(&gate, &up)| {
            let (gate, up) = match self.limit {
                Some(limit) => (gate.min(limit), up.clamp(-limit, limit)),
                None => (gate, up),
            };
            // Keep the existing SiLU and multiply order for uncapped configs.
            crate::flash::refcpu::silu(gate) * up
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.000001, "{actual} != {expected}");
        }
    }
    #[test]
    fn cap_before_silu_hand_computed() {
        let op = Swiglu::from_config(br#"{"hidden_act":"silu","swiglu_limit":2}"#).unwrap();
        let actual = op.apply(&[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            &[3.0, 3.0, -3.0, 3.0, -3.0, -3.0, -3.0]).unwrap();
        // For positive gate at the cap: 2/(1+exp(-2))*2.
        // Below zero, gate is unchanged: -3/(1+exp(3))*(-2).
        close(&actual, &[3.5231883, 3.5231883, -1.4621172, 0.0, 0.53788286, 0.47681168, 0.28455523]);
        assert_eq!(actual[0].to_bits(), actual[1].to_bits());
    }
    #[test]
    fn both_up_bounds_and_in_range_hand_computed() {
        let op = Swiglu::from_config(br#"{"text_config":{"swiglu_limit":1}}"#).unwrap();
        let actual = op.apply(&[2.0, 2.0, 2.0, -2.0], &[2.0, -2.0, 0.25, 0.25]).unwrap();
        // sigmoid(1)=0.73105858, sigmoid(-2)=0.11920292.
        close(&actual, &[0.7310586, -0.7310586, 0.18276465, -0.05960146]);
        assert_eq!(op.limit(), Some(1.0));
    }
    #[test]
    fn absent_and_null_caps_keep_uncapped_bits() {
        let gate = [-100.0, -3.0, -1.0, -0.0, 0.0, 1.0, 3.0, 100.0];
        let up = [3.0, -3.0, -0.0, 2.0, -2.0, 0.25, 3.0, -3.0];
        let expected = gate.iter().zip(up).map(|(&g, u)| (crate::flash::refcpu::silu(g) * u).to_bits()).collect::<Vec<_>>();
        for config in ["{}", r#"{"swiglu_limit":null}"#, r#"{"text_config":{"hidden_act":"silu"}}"#] {
            let op = Swiglu::from_config(config.as_bytes()).unwrap();
            assert_eq!(op.limit(), None);
            assert_eq!(op.apply(&gate, &up).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
        }
    }
    #[test]
    fn config_rejects_invalid_caps_and_other_activations() {
        for value in ["0", "-1", "1e40", "1e-60", "true", r#""2""#, "[]", "{}"] {
            assert!(Swiglu::from_config(format!(r#"{{"swiglu_limit":{value}}}"#).as_bytes()).is_err(), "{value}");
        }
        for config in [r#"{"hidden_act":"gelu"}"#, r#"{"hidden_act":null}"#, r#"{"text_config":null}"#, "[]"] {
            assert!(Swiglu::from_config(config.as_bytes()).is_err(), "{config}");
        }
    }
    #[test]
    fn vectors_reject_bad_shapes_and_nonfinite_values() {
        let op = Swiglu::default();
        assert!(op.apply(&[1.0], &[]).is_err());
        assert!(op.apply(&[f32::NAN], &[1.0]).is_err());
        assert!(op.apply(&[1.0], &[f32::INFINITY]).is_err());
        assert_eq!(op.apply(&[], &[]).unwrap(), []);
    }
    #[test]
    fn config_does_not_use_model_names() {
        let text = r#"{"hidden_act":"silu","swiglu_limit":1.5}"#;
        let plain = Swiglu::from_config(text.as_bytes()).unwrap();
        let nested = format!(r#"{{"model_type":"unrelated","text_config":{text}}}"#);
        assert_eq!(plain, Swiglu::from_config(nested.as_bytes()).unwrap());
    }
}
