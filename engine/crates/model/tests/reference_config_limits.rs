use engine_model::{dsa, kda, mhc, mla, shapes::CONFIG_BYTES};

#[test]
fn host_references_share_the_config_byte_limit() {
    // This small operation config is valid for each reference. Padding checks
    // the byte boundary without changing JSON node count or operation data.
    let source = br#"{"hidden_size":2,"hidden_act":"silu","rms_norm_eps":0.00001,
        "q_lora_rank":2,"kv_lora_rank":2,"num_attention_heads":2,
        "qk_nope_head_dim":2,"qk_rope_head_dim":0,"v_head_dim":2,
        "linear_attn_config":{"num_heads":2,"head_dim":2,"short_conv_kernel_size":2,"gate_lower_bound":-5},
        "index_n_heads":2,"index_head_dim":2,"index_topk":4,"index_kpool":2,
        "index_kpool_always_select_tail":true,"hc_mult":2,"hc_sinkhorn_iters":2,"hc_eps":0.000001}"#;
    for wrapped in [false, true] {
        let mut bytes = if wrapped { [b"{\"text_config\":".as_slice(), source, b"}"].concat() } else { source.to_vec() };
        bytes.resize(CONFIG_BYTES, b' ');
        let (d, k, m, h) = (dsa::Config::from_json(&bytes).unwrap(), kda::Config::from_json(&bytes).unwrap(),
            mla::Config::from_json(&bytes).unwrap(), mhc::Config::from_json(&bytes).unwrap());
        assert_eq!((d.hidden_size, k.hidden_size, m.hidden_size, h.hidden_size), (2, 2, 2, 2));
        bytes.push(b' ');
        for result in [dsa::Config::from_json(&bytes).map(|_| ()), kda::Config::from_json(&bytes).map(|_| ()),
            mla::Config::from_json(&bytes).map(|_| ()), mhc::Config::from_json(&bytes).map(|_| ())] {
            assert!(result.unwrap_err().to_string().contains("byte limit"));
        }
    }
}
