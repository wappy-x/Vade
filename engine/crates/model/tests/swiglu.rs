use engine_model::{flash::nvfp4::{bf16_to_f32, f32_to_bf16}, plan::ConfigPlan, swiglu::SwiGlu};
fn round(x: f32) -> f32 { bf16_to_f32(f32_to_bf16(x)) }

#[test]
fn checkpoint_bound_is_operation_data() {
    let ConfigPlan::Deferred(p) = ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap()
        else { panic!("expected deferred operation data") };
    assert_eq!(p.swiglu().unwrap().limit(), Some(10.0));
    assert_eq!(SwiGlu::default().limit(), None);
    assert_eq!(SwiGlu::new(None).unwrap(), SwiGlu::default());
    for v in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e100, 1e-100,
        f32::MAX as f64] { assert!(SwiGlu::new(Some(v)).is_err(), "limit={v}"); }
    assert_eq!(SwiGlu::new(Some(3.5)).unwrap().limit(), Some(3.5));
    let fixture = std::str::from_utf8(include_bytes!("fixtures/glm/config.json")).unwrap();
    for value in ["0", "-1", "1e100", "1e-100", "3.4028235e38"] {
        let config = fixture.replace("\"swiglu_limit\": 10.0", &format!("\"swiglu_limit\": {value}"));
        assert!(ConfigPlan::from_config(config.as_bytes()).is_err(), "config bound={value}");
    }
    let config = fixture.replace("\"swiglu_limit\": 10.0", "\"swiglu_limit\": 3.5");
    let ConfigPlan::Deferred(p) = ConfigPlan::from_config(config.as_bytes()).unwrap() else { panic!() };
    assert_eq!(p.swiglu().unwrap().limit(), Some(3.5));
}

#[test]
fn cap_precedes_silu_and_product_and_has_no_gate_lower_bound() {
    let c = SwiGlu::new(Some(10.0)).unwrap();
    assert_eq!(c.reference(20.0, 20.0, true).to_bits(), 100.0f32.to_bits());
    assert_eq!(c.reference(20.0, -20.0, true).to_bits(), (-100.0f32).to_bits());
    assert_eq!(c.reference(20.0, 2.0, true).to_bits(), 20.0f32.to_bits());
    for bf16 in [false, true] {
        assert_eq!(c.reference(-20.0, 20.0, bf16).to_bits(), c.reference(-20.0, 10.0, bf16).to_bits());
        assert_ne!(c.reference(-20.0, 10.0, bf16).to_bits(), c.reference(-10.0, 10.0, bf16).to_bits());
        assert_eq!(c.reference(20.0, 20.0, bf16).to_bits(), c.reference(10.0, 10.0, bf16).to_bits());
        assert!(c.reference(10.0, 10.0, bf16) > 10.0);
    }
    // A different bound must change arithmetic. No checkpoint constant is used.
    assert_eq!(SwiGlu::new(Some(4.0)).unwrap().reference(20.0, 20.0, true), 15.6875);
}

#[test]
fn bf16_projection_clamp_silu_and_product_are_separate_boundaries() {
    let c = SwiGlu::new(Some(1.00390625)).unwrap();
    // The scalar bound is also converted to the tensor dtype, ties to even.
    let expected = round(round(1.0 / (1.0 + (-1.0f32).exp())) * 1.0);
    assert_eq!(c.reference(1.01, 1.02, true).to_bits(), expected.to_bits());
    let c = SwiGlu::new(Some(10.0)).unwrap();
    let mut differs_from_final_only = false;
    for i in 1..2048 {
        let (g, u) = (i as f32 / 1024.0, 1.0 + i as f32 / 2048.0);
        let got = c.reference_with_exp(g, u, true, f32::exp);
        let expected = round(round(round(g) / (1.0 + (-round(g)).exp())) * round(u));
        assert_eq!(got.to_bits(), expected.to_bits());
        differs_from_final_only |= got.to_bits() != round(g / (1.0 + (-g).exp()) * u).to_bits();
    }
    assert!(differs_from_final_only);
}

#[test]
fn uncapped_reference_preserves_legacy_rounding_bits() {
    let c = SwiGlu::default();
    let mut state = 0x12345678u32;
    for i in 0..65536u32 {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        let g = (i as f32 - 32768.0) / 512.0;
        let u = (state as i32 as f32) / 134217728.0;
        for bf16 in [false, true] {
            let old = if bf16 {
                let g = round(g);
                round(round(g / (1.0 + (-g).exp())) * round(u))
            } else { g / (1.0 + (-g).exp()) * u };
            assert_eq!(c.reference_with_exp(g, u, bf16, f32::exp).to_bits(), old.to_bits());
        }
    }
    for g in [0.0, -0.0] { for u in [0.0, -0.0, -2.0, 2.0] { for bf16 in [false, true] {
        assert_eq!(c.reference_with_exp(g, u, bf16, f32::exp).to_bits(), (g * u).to_bits());
    } } }
}
