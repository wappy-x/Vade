use super::*;

fn config(iters: usize) -> Config {
    Config { hidden_size: 2, hc_mult: 2, hc_sinkhorn_iters: iters, rms_norm_eps: 3.0, hc_eps: 0.25 }
}
fn close(actual: &[f32], expected: &[f32], bound: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= bound, "{a} != {b}, bound {bound}"); }
}
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
fn hand_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // x=[0,0,2,0], mean(x^2)=1, rms_norm_eps=3: normalized x=[0,0,1,0].
    // The selected column sets each projected coefficient directly.
    let l = 3.0f32.ln();
    let projected = [0.5 * l, 0.0, 0.0, 2.0 * l, 0.5 * l, 0.0, 0.0, 0.0];
    let mut projection = vec![0.0; 32];
    for (row, &value) in projected.iter().enumerate() { projection[row * 4 + 2] = value; }
    (projection, vec![0.0, -l, -l, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2.0, 0.5, 2.0])
}

#[test]
fn hand_projection_gates_collapse_and_residual_orientation() {
    let (projection, base, scale) = hand_weights();
    let reference = Reference::new(config(1), Weights { projection: &projection, base: &base, scale: &scale }).unwrap();
    let x = [0.0, 0.0, 2.0, 0.0];
    let mixed = reference.mix(&x).unwrap();
    // Gate logits: pre=[ln3,-ln3], post=[-ln3,ln3]. Add epsilon only to pre.
    close(&mixed.pre, &[1.0, 0.5], 2e-7);
    close(&mixed.post, &[0.5, 1.5], 2e-7);
    // Row softmax plus epsilon gives [1,1/2;3/4,3/4].
    // Column divisors, with epsilon, are [2,3/2].
    close(&mixed.comb, &[0.5, 1.0 / 3.0, 3.0 / 8.0, 0.5], 2e-7);
    close(&mixed.collapsed, &[1.0, 0.0], 2e-7);
    // Output stream j uses sum_i(comb[i,j] * x[i]), plus post[j] * block.
    close(&reference.combine(&x, &mixed, &[4.0, -2.0]).unwrap(), &[11.0 / 4.0, -1.0, 7.0, -3.0], 1e-6);
}

#[test]
fn configured_iteration_count_includes_initial_column_normalization() {
    let (projection, base, scale) = hand_weights();
    let weights = || Weights { projection: &projection, base: &base, scale: &scale };
    let first = Reference::new(config(1), weights()).unwrap().mix(&[0.0, 0.0, 2.0, 0.0]).unwrap();
    let second = Reference::new(config(2), weights()).unwrap().mix(&[0.0, 0.0, 2.0, 0.0]).unwrap();
    close(&first.comb, &[0.5, 1.0 / 3.0, 3.0 / 8.0, 0.5], 2e-7);
    // The next row pass gives [6/13,4/13;1/3,4/9].
    // Its column divisors are 163/156 and 469/468.
    close(&second.comb, &[72.0 / 163.0, 144.0 / 469.0, 52.0 / 163.0, 208.0 / 469.0], 2e-7);
    assert_ne!(bits(&first.comb), bits(&second.comb));
    assert_eq!(bits(&first.pre), bits(&second.pre));
    assert_eq!(bits(&first.post), bits(&second.post));
    assert_eq!(bits(&first.collapsed), bits(&second.collapsed));
}

#[test]
fn zero_weights_keep_epsilon_and_use_raw_streams() {
    let projection = [0.0; 32];
    let reference = Reference::new(config(1), Weights { projection: &projection, base: &[0.0; 8], scale: &[0.0; 3] }).unwrap();
    let x = [1.0, -2.0, 3.0, 4.0];
    let mixed = reference.mix(&x).unwrap();
    assert_eq!(mixed.pre, [0.75; 2]);
    assert_eq!(mixed.post, [1.0; 2]);
    close(&mixed.comb, &[3.0 / 7.0; 4], 1e-7);
    assert_eq!(mixed.collapsed, [3.0, 1.5]);
    close(&reference.combine(&x, &mixed, &[0.5, -1.0]).unwrap(),
        &[31.0 / 14.0, -1.0 / 7.0, 31.0 / 14.0, -1.0 / 7.0], 5e-7);
}

#[test]
fn one_stream_keeps_configured_epsilon_in_every_divisor() {
    let d = Config { hidden_size: 2, hc_mult: 1, hc_sinkhorn_iters: 3, rms_norm_eps: 1e-5, hc_eps: 0.25 };
    let reference = Reference::new(d, Weights { projection: &[0.0; 6], base: &[0.0; 3], scale: &[0.0; 3] }).unwrap();
    let x = [3.0, 4.0];
    let mixed = reference.mix(&x).unwrap();
    assert_eq!(mixed.pre, [0.75]);
    assert_eq!(mixed.post, [1.0]);
    assert_eq!(mixed.collapsed, [2.25, 3.0]);
    // Start at 5/4. Apply v/(v+1/4) once, then twice per later iteration:
    // 5/6, 10/13, 40/53, 160/213, 640/853.
    close(&mixed.comb, &[640.0 / 853.0], 2e-7);
    close(&reference.combine(&x, &mixed, &[-1.0, 2.0]).unwrap(), &[1067.0 / 853.0, 4266.0 / 853.0], 1e-6);
}

#[test]
fn nonuniform_sinkhorn_rows_and_columns_meet_fixture_bound() {
    // This bound applies to this fixture. A fixed iteration count does not
    // guarantee convergence for every matrix.
    let d = Config { hidden_size: 2, hc_mult: 3, hc_sinkhorn_iters: 20, rms_norm_eps: 1e-5, hc_eps: 1e-6 };
    let projection = vec![0.0; 90];
    let mut base = vec![0.0; 6];
    // Before epsilon: [1/2,1/3,1/6;1/4,1/2,1/4;1/6,1/6,2/3].
    base.extend([3.0f32.ln(), 2.0f32.ln(), 0.0, 0.0, 2.0f32.ln(), 0.0, 0.0, 0.0, 4.0f32.ln()]);
    let reference = Reference::new(d, Weights { projection: &projection, base: &base, scale: &[1.0; 3] }).unwrap();
    let mixed = reference.mix(&[-2.0, 1.0, 3.0, 2.0, -1.0, 4.0]).unwrap();
    let bound = 4e-6;
    let mut max_row_error = 0.0f32;
    let mut max_column_error = 0.0f32;
    for i in 0..3 {
        let mut row = 0.0f32;
        let mut column = 0.0f32;
        for j in 0..3 { row += mixed.comb[i * 3 + j]; column += mixed.comb[j * 3 + i]; }
        max_row_error = max_row_error.max((row - 1.0).abs());
        max_column_error = max_column_error.max((column - 1.0).abs());
        assert!((row - 1.0).abs() <= bound, "row {i}: {row}, bound {bound}");
        assert!((column - 1.0).abs() <= bound, "column {i}: {column}, bound {bound}");
    }
    println!("Sinkhorn fixture N=3, iterations=20, epsilon=1e-6: max_row_error={max_row_error}, max_column_error={max_column_error}, bound={bound}");
    assert!(mixed.comb.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn repeated_calls_keep_all_output_bits() {
    let (projection, base, scale) = hand_weights();
    let reference = Reference::new(config(7), Weights { projection: &projection, base: &base, scale: &scale }).unwrap();
    let x = [0.0, 0.0, 2.0, 0.0];
    let first = reference.mix(&x).unwrap();
    let output = reference.combine(&x, &first, &[4.0, -2.0]).unwrap();
    for _ in 0..5 {
        let mixed = reference.mix(&x).unwrap();
        assert_eq!(bits(&mixed.pre), bits(&first.pre));
        assert_eq!(bits(&mixed.post), bits(&first.post));
        assert_eq!(bits(&mixed.comb), bits(&first.comb));
        assert_eq!(bits(&mixed.collapsed), bits(&first.collapsed));
        assert_eq!(bits(&reference.combine(&x, &mixed, &[4.0, -2.0]).unwrap()), bits(&output));
    }
}

#[test]
fn config_fields_work_without_a_model_name() {
    let plain = r#"{"mhc":true,"hidden_size":2,"hc_mult":2,"hc_sinkhorn_iters":7,"rms_norm_eps":3.0,"hc_eps":0.25}"#;
    let wrapped = format!(r#"{{"model_type":"arbitrary","text_config":{plain}}}"#);
    for input in [plain.as_bytes(), wrapped.as_bytes()] {
        let d = Config::from_json(input).unwrap();
        assert_eq!((d.hidden_size, d.hc_mult, d.hc_sinkhorn_iters), (2, 2, 7));
        assert_eq!((d.rms_norm_eps.to_bits(), d.hc_eps.to_bits()), (3.0f32.to_bits(), 0.25f32.to_bits()));
    }
    for (from, to) in [("\"hidden_size\":2", "\"hidden_size\":0"), ("\"hc_mult\":2", "\"hc_mult\":0"),
        ("\"hc_sinkhorn_iters\":7", "\"hc_sinkhorn_iters\":0"), ("\"hc_sinkhorn_iters\":7", "\"hc_sinkhorn_iters\":1.5"),
        ("\"rms_norm_eps\":3.0", "\"rms_norm_eps\":-1.0"), ("\"hc_eps\":0.25", "\"hc_eps\":1e300"),
        ("\"mhc\":true", "\"mhc\":false")] {
        assert!(Config::from_json(plain.replace(from, to).as_bytes()).is_err());
    }
    assert!(Config::from_json(b"{}").is_err());
}

#[test]
fn saved_description_binds_operation_data_and_keeps_admission_closed() {
    use crate::plan::ConfigPlan;
    let bytes = include_bytes!("../../tests/fixtures/glm/config.json");
    let plan = ConfigPlan::from_config(bytes).unwrap();
    let ConfigPlan::Deferred(mut d) = plan.clone() else { panic!() };
    let config = Config::from_description(&d).unwrap();
    assert_eq!(config, Config::from_json(bytes).unwrap());
    assert_eq!((config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters), (4096, 4, 20));
    assert_eq!((config.rms_norm_eps, config.hc_eps), (1e-5, 1e-6));
    assert!(plan.into_model_plan().is_err());
    d.hidden = 7; d.residual.streams = 3; d.residual.iterations = 2;
    assert_eq!((Config::from_description(&d).unwrap().hidden_size, Config::from_description(&d).unwrap().hc_mult), (7, 3));
    for invalid in [f64::INFINITY, f64::NAN, 1e300, 1e-300] {
        d.residual.epsilon = invalid; assert!(Config::from_description(&d).is_err());
    }
}

#[test]
fn head_uses_raw_unweighted_stream_mean_in_fixed_order() {
    let c = Config { hc_mult: 3, hidden_size: 2, ..config(1) };
    let x = [16777216.0, 3.0, 1.0, -6.0, -16777216.0, 9.0];
    assert_eq!(bits(&c.mean_head(&x).unwrap()), bits(&[0.0, 2.0]));
    assert!(c.mean_head(&x[..5]).is_err());
    assert!(c.mean_head(&[f32::NAN; 6]).is_err());
    assert!(c.mean_head(&[f32::MAX; 6]).is_err());
    let one = Config { hc_mult: 1, ..config(1) };
    assert_eq!(one.mean_head(&[-3.0, 7.0]).unwrap(), [-3.0, 7.0]);
}

#[test]
fn rows_and_chunk_partitions_do_not_change_two_block_or_head_bits() {
    let (projection, base, scale) = hand_weights();
    let attention = Reference::new(config(2), Weights { projection: &projection, base: &base, scale: &scale }).unwrap();
    let ffn = Reference::new(config(7), Weights { projection: &projection, base: &base, scale: &[0.25, 0.75, 0.5] }).unwrap();
    let rows = [[0.0, 0.0, 2.0, 0.0], [1.0, -2.0, 3.0, 4.0], [-3.0, 1.0, 0.0, 2.0]];
    let run = |x: &[f32]| {
        let a = attention.mix(x).unwrap();
        let x = attention.combine(x, &a, &a.collapsed).unwrap();
        let b = ffn.mix(&x).unwrap();
        let y = ffn.combine(&x, &b, &b.collapsed).unwrap();
        bits(&config(2).mean_head(&y).unwrap())
    };
    let expected: Vec<_> = rows.iter().map(|x| run(x)).collect();
    for cut in 0..=rows.len() {
        let actual: Vec<_> = rows[..cut].iter().chain(&rows[cut..]).map(|x| run(x)).collect();
        assert_eq!(actual, expected);
    }
    for i in (0..rows.len()).rev() { assert_eq!(run(&rows[i]), expected[i]); }
}

#[test]
fn invalid_config_is_rejected_before_allocation() {
    let mut invalid = config(1); invalid.hidden_size = 0; assert!(invalid.validate().is_err());
    let mut invalid = config(1); invalid.hc_mult = 0; assert!(invalid.validate().is_err());
    let mut invalid = config(0); assert!(invalid.validate().is_err());
    let mut huge = config(1); huge.hc_sinkhorn_iters = crate::shapes::MAX_WIDTH as usize + 1;
    assert!(huge.validate().is_err());
    invalid.hidden_size = usize::MAX; assert!(invalid.validate().is_err());
    let mut invalid = config(1); invalid.hidden_size = usize::MAX; assert!(invalid.validate().is_err());
    let mut invalid = config(1); invalid.hc_mult = usize::MAX; assert!(invalid.validate().is_err());
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = config(1); invalid.rms_norm_eps = value; assert!(invalid.validate().is_err());
        let mut invalid = config(1); invalid.hc_eps = value; assert!(invalid.validate().is_err());
    }
    let invalid = Config { hidden_size: usize::MAX, ..config(1) };
    assert!(Reference::new(invalid, Weights { projection: &[], base: &[], scale: &[] }).is_err());
}

#[test]
fn invalid_weights_inputs_and_mix_shapes_are_rejected() {
    let (projection, base, scale) = hand_weights();
    for weights in [Weights { projection: &projection[..31], base: &base, scale: &scale },
        Weights { projection: &projection, base: &base[..7], scale: &scale },
        Weights { projection: &projection, base: &base, scale: &scale[..2] }] {
        assert!(Reference::new(config(1), weights).is_err());
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = projection.clone(); bad[0] = invalid;
        assert!(Reference::new(config(1), Weights { projection: &bad, base: &base, scale: &scale }).is_err());
        let mut bad = base.clone(); bad[0] = invalid;
        assert!(Reference::new(config(1), Weights { projection: &projection, base: &bad, scale: &scale }).is_err());
        let mut bad = scale.clone(); bad[0] = invalid;
        assert!(Reference::new(config(1), Weights { projection: &projection, base: &base, scale: &bad }).is_err());
    }
    let reference = Reference::new(config(1), Weights { projection: &projection, base: &base, scale: &scale }).unwrap();
    let x = [0.0, 0.0, 2.0, 0.0];
    assert!(reference.mix(&x[..3]).is_err());
    assert!(reference.mix(&[f32::MAX; 4]).is_err());
    let mixed = reference.mix(&x).unwrap();
    assert!(reference.combine(&x, &mixed, &[f32::MAX, 0.0]).is_err());
    assert!(reference.combine(&x[..3], &mixed, &[0.0; 2]).is_err());
    assert!(reference.combine(&x, &mixed, &[0.0; 1]).is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(reference.mix(&[invalid; 4]).is_err());
        assert!(reference.combine(&[invalid; 4], &mixed, &[0.0; 2]).is_err());
        assert!(reference.combine(&x, &mixed, &[invalid; 2]).is_err());
        let mut bad = reference.mix(&x).unwrap(); bad.post[0] = invalid;
        assert!(reference.combine(&x, &bad, &[0.0; 2]).is_err());
        let mut bad = reference.mix(&x).unwrap(); bad.comb[0] = invalid;
        assert!(reference.combine(&x, &bad, &[0.0; 2]).is_err());
    }
    let mut bad = reference.mix(&x).unwrap(); bad.post.pop();
    assert!(reference.combine(&x, &bad, &[0.0; 2]).is_err());
    let mut bad = reference.mix(&x).unwrap(); bad.comb.pop();
    assert!(reference.combine(&x, &bad, &[0.0; 2]).is_err());
    let overflow = vec![f32::MAX; 32];
    let reference = Reference::new(config(1), Weights { projection: &overflow, base: &base, scale: &scale }).unwrap();
    assert!(reference.mix(&[1.0; 4]).is_err());
}
