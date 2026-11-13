//! Deferred FFN tests use the shared host schedule builder.
use super::*;
use super::adapter::ConfigPlan;

#[cfg(test)]
mod tests {
    use super::*;
    use FfnKind::{Dense, Experts};

    fn parse(fields: &str) -> Result<DeferredPlan> {
        let config = format!(r#"{{"model_type":"glm5_next_text","hidden_size":128,"num_hidden_layers":5,
            "num_nextn_predict_layers":1,"layer_types":["linear_attention","linear_attention",
            "linear_attention","linear_attention","linear_attention"],"moe_intermediate_size":64,
            "n_shared_experts":0,"quantization_config":{{"quant_method":"modelopt",
            "quant_algo":"MIXED_PRECISION","ignore":[],"quantized_layers":{{"model":{{
            "quant_algo":"MXFP8","group_size":32}}}}}}{fields}}}"#);
        let ConfigPlan::Deferred(plan) = ConfigPlan::from_config(config.as_bytes())? else { panic!("expected deferred plan"); };
        Ok(plan)
    }

    #[test]
    fn all_dense_includes_appended_mtp() {
        for fields in ["", r#", "n_routed_experts":0"#, r#", "num_experts":null"#,
            r#", "mlp_layer_types":["dense","dense","dense","dense","dense"]"#] {
            assert_eq!(parse(fields).unwrap().ffn_layers, [Dense; 6]);
        }
    }
    #[test]
    fn all_experts_includes_appended_mtp() {
        for field in ["n_routed_experts", "num_experts"] {
            assert_eq!(parse(&format!(r#", "{field}":4"#)).unwrap().ffn_layers, [Experts; 6]);
        }
    }
    #[test]
    fn dense_prefix_three_uses_absolute_layer_indices() {
        let p = parse(r#", "n_routed_experts":4,"first_k_dense_replace":3"#).unwrap();
        assert_eq!(p.ffn_layers, [Dense, Dense, Dense, Experts, Experts, Experts]);
        assert_eq!(p.ffn_layers[p.decoder_layers as usize], Experts);
    }
    #[test]
    fn explicit_list_sets_each_decoder_layer() {
        let fields = r#", "mlp_layer_types":["sparse","dense","dense","sparse","dense"]"#;
        for fields in [fields.to_owned(), format!(r#"{fields},"n_routed_experts":4"#)] {
            assert_eq!(parse(&fields).unwrap().ffn_layers, [Experts, Dense, Dense, Experts, Dense, Experts]);
        }
    }
    #[test]
    fn matching_keys_and_prefix_boundaries() {
        let labels = r#", "mlp_layer_types":["dense","dense","dense","sparse","sparse"]"#;
        assert_eq!(parse(&format!(r#"{labels},"first_k_dense_replace":3"#)).unwrap().ffn_layers,
            [Dense, Dense, Dense, Experts, Experts, Experts]);
        assert_eq!(parse(r#", "n_routed_experts":4,"first_k_dense_replace":0"#).unwrap().ffn_layers, [Experts; 6]);
        // A dense decoder prefix ends before the appended draft index.
        assert_eq!(parse(r#", "n_routed_experts":4,"first_k_dense_replace":5"#).unwrap().ffn_layers,
            [Dense, Dense, Dense, Dense, Dense, Experts]);
    }
    #[test]
    fn wrong_list_lengths_refuse() {
        for labels in ["[]", r#"["dense","dense","dense","dense"]"#,
            r#"["dense","dense","dense","dense","dense","sparse"]"#] {
            assert_eq!(parse(&format!(r#", "mlp_layer_types":{labels}"#)).unwrap_err(),
                PlanError::Invalid { field: "mlp_layer_types", reason: "length differs from num_hidden_layers" });
        }
    }
    #[test]
    fn explicit_lists_override_prefix_in_the_shared_builder() {
        for (labels, expected) in [(r#"["dense","dense","sparse","sparse","sparse"]"#, [Dense, Dense, Experts, Experts, Experts, Experts]),
            (r#"["dense","dense","dense","dense","sparse"]"#, [Dense, Dense, Dense, Dense, Experts, Experts])] {
            let fields = format!(r#", "n_routed_experts":4,"first_k_dense_replace":3,"mlp_layer_types":{labels}"#);
            assert_eq!(parse(&fields).unwrap().ffn_layers, expected);
        }
    }
    #[test]
    fn invalid_fields_and_expert_counts_refuse() {
        for fields in [
            r#", "first_k_dense_replace":6"#, r#", "first_k_dense_replace":-1"#,
            r#", "first_k_dense_replace":null"#, r#", "first_k_dense_replace":true"#,
            r#", "mlp_layer_types":null"#, r#", "mlp_layer_types":"dense"#,
            r#", "mlp_layer_types":["dense","dense","dense","dense",0]"#,
            r#", "mlp_layer_types":["dense","dense","dense","dense","unknown"]"#,
            r#", "n_routed_experts":0,"mlp_layer_types":["dense","dense","dense","dense","sparse"]"#,
            r#", "n_routed_experts":4,"num_experts":8"#, r#", "n_routed_experts":1025"#,
            r#", "n_routed_experts":-1"#, r#", "num_experts":true"#,
        ] { assert!(parse(fields).is_err(), "{fields}"); }
    }
}
