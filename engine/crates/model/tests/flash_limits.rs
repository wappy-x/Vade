use engine_model::{
    flash::{PLE_PREFIX, WeightFormat, checkpoint_limits},
    safetensors::Index,
};
use std::{collections::BTreeSet, fmt::Write};

// Enumerate the retained legacy loader's required names, including MTP and PLE.
fn legacy_inventory() -> BTreeSet<String> {
    let mut names = BTreeSet::from([
        "lm_head.weight".into(),
        "model.language_model.embed_tokens.weight".into(),
    ]);
    let hc = |names: &mut BTreeSet<String>, p: &str, inject: bool| {
        for n in ["hc_norm", "input_mix_weight_down", "input_mix_weight_up"] {
            names.insert(format!("{p}.{n}.weight"));
        }
        if inject {
            names.insert(format!("{p}.block_inject_weight.weight"));
        }
    };
    for p in [
        "model.language_model.hyper_connection_mixer",
        "mtp.hyper_connection_mixer",
    ] {
        hc(&mut names, p, false);
    }
    for (prefix, layers) in [("model.language_model.layers", 48), ("mtp.layers", 1)] {
        for li in 0..layers {
            let p = format!("{prefix}.{li}");
            for role in ["attn", "mlp"] {
                hc(&mut names, &format!("{p}.{role}_hyper_connection"), true);
            }
            for expert in 0..512 {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    let suffixes: &[&str] = if layers == 48 {
                        &["weight", "weight_scale", "weight_scale_2", "input_scale"]
                    } else {
                        &["weight", "weight_scale_inv"]
                    };
                    for suffix in suffixes {
                        names.insert(format!("{p}.mlp.experts.{expert}.{proj}.{suffix}"));
                    }
                }
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                names.insert(format!("{p}.mlp.shared_expert.{proj}.weight"));
            }
            for n in ["gate", "shared_expert_gate"] {
                names.insert(format!("{p}.mlp.{n}.weight"));
            }
            let qsa = layers == 1 || li % 4 == 3;
            let tensors = if qsa {
                [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                    "indexer.index_qk_proj.weight",
                    "indexer.q_layernorm.weight",
                    "indexer.k_layernorm.weight",
                ]
            } else {
                [
                    "A_log",
                    "dt_bias",
                    "conv1d.weight",
                    "in_proj_a.weight",
                    "in_proj_b.weight",
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "norm.weight",
                    "out_proj.weight",
                ]
            };
            for n in tensors {
                names.insert(format!(
                    "{p}.{}.{n}",
                    if qsa { "self_attn" } else { "linear_attn" }
                ));
            }
        }
    }
    for n in [
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "fc_embedding",
        "fc_hidden",
    ] {
        names.insert(format!("mtp.{n}.weight"));
    }
    for n in [
        "conv1d",
        "key_proj",
        "value_proj",
        "norm_conv",
        "norm_key",
        "norm_query",
    ] {
        names.insert(format!("model.language_model.layers.1.ple.{n}.weight"));
    }
    for n in [
        "layer_multipliers",
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
        "ngram_embedding.weight_scale",
    ] {
        names.insert(format!("{PLE_PREFIX}.{n}"));
    }
    for part in 0..128 {
        names.insert(format!("{PLE_PREFIX}.ngram_embedding.shard_{part}.weight"));
    }
    names
}
#[test]
fn flash_format_bounds_accept_legacy_inventory_without_relaxing_fp8() {
    let names = legacy_inventory();
    assert_eq!(names.len(), 299_212);
    let index = |names: &BTreeSet<String>| {
        let mut text = String::from("{\"metadata\":{},\"weight_map\":{");
        for (i, n) in names.iter().enumerate() {
            write!(
                text,
                "{}{n:?}:\"model.safetensors\"",
                if i == 0 { "" } else { "," }
            )
            .unwrap();
        }
        text.push_str("}}");
        text
    };
    let fp8 = checkpoint_limits(WeightFormat::Fp8, false);
    let legacy = checkpoint_limits(WeightFormat::Nvfp4, false);
    let mixed = checkpoint_limits(WeightFormat::Nvfp4, true);
    assert_eq!(
        (fp8.tensors, legacy.tensors, mixed.tensors),
        (200_000, 320_000, 500_000)
    );
    let text = index(&names);
    assert!(text.len() < fp8.header_bytes);
    assert!(Index::parse(text.as_bytes(), fp8).is_err());
    assert_eq!(
        Index::parse(text.as_bytes(), legacy)
            .unwrap()
            .weight_map
            .len(),
        299_212
    );
    for (limit, mixed_paths) in [(fp8, false), (legacy, false), (mixed, true)] {
        let mut names: BTreeSet<_> = (0..limit.tensors).map(|i| format!("audit_{i}")).collect();
        let parse = |text: &str| {
            if mixed_paths {
                Index::parse_mixed(text.as_bytes(), limit)
            } else {
                Index::parse(text.as_bytes(), limit)
            }
        };
        assert_eq!(
            parse(&index(&names)).unwrap().weight_map.len(),
            limit.tensors
        );
        names.insert("audit_overflow".into());
        assert!(parse(&index(&names)).is_err());
    }
}
