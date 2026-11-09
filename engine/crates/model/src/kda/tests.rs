use super::*;

fn config() -> Config {
    Config { hidden_size: 2, num_heads: 1, head_dim: 2, conv_kernel_size: 2,
        gate_lower_bound: Some(-4.0), rms_norm_eps: 0.25 }
}
fn bits(x: &[f32]) -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() }
fn same_state(a: &State, b: &State) {
    assert_eq!(a.config, b.config);
    assert_eq!(bits(&a.s), bits(&b.s));
    assert_eq!(bits(&a.q_history), bits(&b.q_history));
    assert_eq!(bits(&a.k_history), bits(&b.k_history));
    assert_eq!(bits(&a.v_history), bits(&b.v_history));
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 2e-6, "entry {i}: {x} differs from {y}");
    }
}

struct Owned {
    q_proj: Vec<f32>, k_proj: Vec<f32>, v_proj: Vec<f32>,
    q_conv1d: Vec<f32>, k_conv1d: Vec<f32>, v_conv1d: Vec<f32>,
    b_proj: Vec<f32>, f_a_proj: Vec<f32>, f_b_proj: Vec<f32>,
    g_a_proj: Vec<f32>, g_b_proj: Vec<f32>, dt_bias: Vec<f32>,
    a_log: Vec<f32>, o_norm: Vec<f32>, o_proj: Vec<f32>,
}
impl Owned {
    fn weights(&self) -> Weights<'_> {
        Weights { q_proj: &self.q_proj, k_proj: &self.k_proj, v_proj: &self.v_proj,
            q_conv1d: &self.q_conv1d, k_conv1d: &self.k_conv1d, v_conv1d: &self.v_conv1d,
            b_proj: &self.b_proj, f_a_proj: &self.f_a_proj, f_b_proj: &self.f_b_proj,
            g_a_proj: &self.g_a_proj, g_b_proj: &self.g_b_proj, dt_bias: &self.dt_bias,
            a_log: &self.a_log, o_norm: &self.o_norm, o_proj: &self.o_proj }
    }
    fn hand() -> Self {
        Self { q_proj: vec![1.0, 2.0, -1.0, 1.0], k_proj: vec![2.0, 0.0, 0.0, -2.0],
            v_proj: vec![1.0, -1.0, 2.0, 1.0], q_conv1d: vec![2.0, 1.0, -1.0, 2.0],
            k_conv1d: vec![1.0, 0.5, 2.0, -1.0], v_conv1d: vec![0.5, 2.0, -1.0, 1.0],
            b_proj: vec![0.5, 1.0], f_a_proj: vec![1.0, 0.0, 0.0, 2.0],
            f_b_proj: vec![0.5, 1.0, -1.0, -0.5], g_a_proj: vec![1.0, 1.0, 1.0, -1.0],
            g_b_proj: vec![2.0, 0.0, -1.0, 2.0], dt_bias: vec![1.0, 2.0],
            a_log: vec![2.0f32.ln()], o_norm: vec![2.0, 0.5], o_proj: vec![1.0, 2.0, -1.0, 1.0] }
    }
    fn varied(d: Config) -> Self {
        let width = d.num_heads * d.head_dim;
        let data = |n, shift| (0..n).map(|i| (((i * 7 + shift) % 19) as f32 - 9.0) / 16.0).collect();
        Self { q_proj: data(width * d.hidden_size, 1), k_proj: data(width * d.hidden_size, 2),
            v_proj: data(width * d.hidden_size, 3), q_conv1d: data(width * d.conv_kernel_size, 4),
            k_conv1d: data(width * d.conv_kernel_size, 5), v_conv1d: data(width * d.conv_kernel_size, 6),
            b_proj: data(d.num_heads * d.hidden_size, 7), f_a_proj: data(d.head_dim * d.hidden_size, 8),
            f_b_proj: data(width * d.head_dim, 9), g_a_proj: data(d.head_dim * d.hidden_size, 10),
            g_b_proj: data(width * d.head_dim, 11), dt_bias: data(width, 12),
            a_log: data(d.num_heads, 13), o_norm: data(d.head_dim, 14),
            o_proj: data(d.hidden_size * width, 15) }
    }
}
fn seeded_state(d: Config) -> State {
    let mut s = State::new(d).unwrap();
    for (j, data) in [&mut s.s, &mut s.q_history, &mut s.k_history, &mut s.v_history].into_iter().enumerate() {
        for (i, x) in data.iter_mut().enumerate() { *x = ((i * 5 + j * 3) % 13) as f32 / 32.0 - 0.125; }
    }
    s
}
fn hand_state() -> State {
    let mut state = State::new(config()).unwrap();
    state.s = vec![1.0, 2.0, -1.0, 0.5];
    state.q_history = vec![1.0, 2.0];
    state.k_history = vec![-1.0, 3.0];
    state.v_history = vec![2.0, -2.0];
    state
}

#[test]
fn separate_convs_and_low_rank_gates_match_hand_vectors() {
    // For x=[2,-1], raw Q=[0,-3], K=[4,2], V=[3,3].
    // Oldest-to-current convolution gives [2,-8], [1,4], [7,5] before SiLU.
    // f_a=[2,-2], f_b=[-1,-1], dt=[1,2], rate=2. Thus g=[-2,-4*sigmoid(2)].
    // g_a=[1,3], g_b=[2,5]. The b projection is zero, so beta=1/2.
    let owned = Owned::hand();
    let reference = Reference::new(config(), owned.weights()).unwrap();
    let mut state = hand_state();
    let p = reference.prepare(&mut state, &[2.0, -1.0]).unwrap();
    close(&p.query, &[0.7071058472, -0.0010768793]);
    close(&p.key, &[0.1829702301, 0.9831184223]);
    close(&p.value, &[6.9936226416, 4.9665357454]);
    close(&p.log_decay, &[-2.0, -3.5231883119]);
    assert_eq!(p.beta, [0.5]);
    assert_eq!(p.output_gate, [2.0, 5.0]);
    assert_eq!(state.q_history, [0.0, -3.0]);
    assert_eq!(state.k_history, [4.0, 2.0]);
    assert_eq!(state.v_history, [3.0, 3.0]);
}

#[test]
fn complete_layer_matches_hand_vector() {
    // Continue the hand vector above from S=[[1,2],[-1,1/2]].
    // Per-channel decay precedes memory prediction. Normalize each output head,
    // multiply norm weights [2,1/2], then sigmoid([2,5]), then [[1,2],[-1,1]].
    let owned = Owned::hand();
    let reference = Reference::new(config(), owned.weights()).unwrap();
    let mut state = hand_state();
    let y = reference.step(&mut state, &[2.0, -1.0]).unwrap();
    close(&state.s, &[0.7755359903, 0.7191770319, 3.4103609816, 2.4246253408]);
    close(&y, &[2.0153296794, -0.9762996335]);
}

#[test]
fn recurrence_decays_key_rows_before_prediction() {
    // The decay values are [1/2,1/4], one per key row. After decay,
    // S=[[1,2],[2,4]], memory=[0,0], and delta=[3/2,-1].
    let d = config();
    let p = Prepared { query: vec![2.0, -1.0], key: vec![0.5, -0.25], value: vec![3.0, -2.0],
        beta: vec![0.5], log_decay: vec![0.5f32.ln(), 0.25f32.ln()], output_gate: vec![0.0; 2] };
    let mut state = vec![2.0, 4.0, 8.0, 16.0];
    let y = recurrent(d, &mut state, &p);
    assert_eq!(bits(&state), bits(&[1.75, 1.5, 1.625, 4.25]));
    assert_eq!(bits(&y), bits(&[1.875, -1.25]));
    let y = recurrent(d, &mut state, &p);
    assert_eq!(bits(&state), bits(&[1.541015625, 0.22265625, 0.0732421875, 1.326171875]));
    assert_eq!(bits(&y), bits(&[3.0087890625, -0.880859375]));
}

#[test]
fn recurrence_keeps_heads_independent() {
    let d = Config { num_heads: 2, ..config() };
    let p = Prepared { query: vec![2.0, -1.0, 1.0, 0.0], key: vec![0.5, -0.25, 0.0, 1.0],
        value: vec![3.0, -2.0, -1.0, 2.0], beta: vec![0.5, 0.25],
        log_decay: vec![0.5f32.ln(), 0.25f32.ln(), 0.0, 0.5f32.ln()], output_gate: vec![0.0; 4] };
    let mut state = vec![2.0, 4.0, 8.0, 16.0, 1.0, 2.0, 3.0, 4.0];
    let y = recurrent(d, &mut state, &p);
    // Head one uses the prior hand vector. Head two has memory=[3/2,2],
    // delta=[-5/8,0], and a query that selects its first key row.
    assert_eq!(bits(&state), bits(&[1.75, 1.5, 1.625, 4.25, 1.0, 2.0, 0.875, 2.0]));
    assert_eq!(bits(&y), bits(&[1.875, -1.25, 1.0, 2.0]));
}

#[test]
fn unbounded_forget_gate_uses_per_channel_softplus() {
    let d = Config { gate_lower_bound: None, ..config() };
    let mut owned = Owned::hand();
    owned.f_a_proj.fill(0.0);
    owned.dt_bias = vec![0.0, 21.0];
    let reference = Reference::new(d, owned.weights()).unwrap();
    let mut state = State::new(d).unwrap();
    let p = reference.prepare(&mut state, &[2.0, -1.0]).unwrap();
    close(&p.log_decay, &[-1.3862943611, -42.0]);
    assert_eq!(p.output_gate, [2.0, 5.0]);
    assert_eq!(softplus(21.0), 21.0);
}

#[test]
fn every_chunk_start_and_end_matches_steps_in_bits() {
    // Every interval starts from its saved prefix state. Each comparison covers
    // output, recurrent state, and all separate convolution histories.
    let count = 7;
    for (heads, dim, taps, bound) in [(1, 1, 1, None), (2, 2, 2, Some(-3.0)),
        (2, 3, 4, None), (1, 3, 3, Some(-5.0))] {
        let d = Config { hidden_size: 3, num_heads: heads, head_dim: dim,
            conv_kernel_size: taps, gate_lower_bound: bound, rms_norm_eps: 1e-5 };
        let owned = Owned::varied(d);
        let reference = Reference::new(d, owned.weights()).unwrap();
        let input: Vec<f32> = (0..count * d.hidden_size).map(|i| ((i * 11 % 23) as f32 - 11.0) / 8.0).collect();
        for initial in [State::new(d).unwrap(), seeded_state(d)] {
            let mut state = initial;
            let mut saved = vec![state.clone()];
            let mut output = Vec::new();
            for x in input.chunks_exact(d.hidden_size) {
                output.extend(reference.step(&mut state, x).unwrap());
                saved.push(state.clone());
            }
            for start in 0..=count {
                for end in start..=count {
                    let mut got = saved[start].clone();
                    let y = reference.scan(&mut got, &input[start * d.hidden_size..end * d.hidden_size]).unwrap();
                    assert_eq!(bits(&y), bits(&output[start * d.hidden_size..end * d.hidden_size]),
                        "interval {start}..{end}, geometry {heads}/{dim}/{taps}");
                    same_state(&got, &saved[end]);
                }
            }
        }
    }
}

#[test]
fn every_partition_keeps_state_and_output_bits() {
    let d = Config { hidden_size: 3, num_heads: 2, head_dim: 3, conv_kernel_size: 4,
        gate_lower_bound: Some(-5.0), rms_norm_eps: 1e-5 };
    let owned = Owned::varied(d);
    let reference = Reference::new(d, owned.weights()).unwrap();
    let count = 7;
    let input: Vec<f32> = (0..count * d.hidden_size).map(|i| ((i * 3 % 17) as f32 - 8.0) / 4.0).collect();
    let initial = seeded_state(d);
    let mut expected_state = initial.clone();
    let mut expected = Vec::new();
    for x in input.chunks_exact(d.hidden_size) { expected.extend(reference.step(&mut expected_state, x).unwrap()); }
    for boundaries in 0..1usize << (count - 1) {
        let mut state = initial.clone();
        let mut output = Vec::new();
        let mut start = 0;
        for end in 1..=count {
            if end == count || boundaries & (1 << (end - 1)) != 0 {
                output.extend(reference.scan(&mut state, &input[start * d.hidden_size..end * d.hidden_size]).unwrap());
                start = end;
            }
        }
        assert_eq!(bits(&output), bits(&expected), "partition {boundaries}");
        same_state(&state, &expected_state);
    }
}

const NESTED: &str = r#"{"hidden_size":2,"hidden_act":"silu","rms_norm_eps":0.25,
    "linear_attn_config":{"num_heads":1,"head_dim":2,"short_conv_kernel_size":2,"gate_lower_bound":-4.0}}"#;
const FLAT: &str = r#"{"hidden_size":2,"hidden_act":"silu","rms_norm_eps":0.25,
    "linear_num_heads":1,"linear_head_dim":2,"linear_conv_kernel_dim":2,"linear_lower_bound":-4.0}"#;

#[test]
fn config_reads_explicit_geometry_without_model_names() {
    assert_eq!(Config::from_json(NESTED.as_bytes()).unwrap(), config());
    assert_eq!(Config::from_json(FLAT.as_bytes()).unwrap(), config());
    let wrapped = format!("{{\"text_config\":{NESTED}}}");
    assert_eq!(Config::from_json(wrapped.as_bytes()).unwrap(), config());
    let unbounded = NESTED.replace("-4.0", "null");
    assert_eq!(Config::from_json(unbounded.as_bytes()).unwrap().gate_lower_bound, None);
}

#[test]
fn config_rejects_missing_ambiguous_or_invalid_fields() {
    for bad in [NESTED.replace("\"hidden_size\":2,", ""),
        NESTED.replace("\"gate_lower_bound\":-4.0", "\"unrelated\":-4.0"),
        NESTED.replace("\"hidden_act\":\"silu\"", "\"hidden_act\":\"relu\""),
        NESTED.replace("\"num_heads\":1", "\"num_heads\":0"),
        NESTED.replace("\"head_dim\":2", "\"head_dim\":1.5"),
        NESTED.replace("\"short_conv_kernel_size\":2", "\"short_conv_kernel_size\":0"),
        NESTED.replace("\"rms_norm_eps\":0.25", "\"rms_norm_eps\":0"),
        NESTED.replace("-4.0", "1.0"),
        NESTED.replacen('{', "{\"linear_num_heads\":1,", 1)] {
        assert!(Config::from_json(bad.as_bytes()).is_err(), "accepted {bad}");
    }
}

#[test]
fn config_checks_size_overflow_and_nonfinite_scalars() {
    for bad in [Config { hidden_size: 0, ..config() }, Config { num_heads: usize::MAX, ..config() },
        Config { hidden_size: usize::MAX, ..config() }, Config { conv_kernel_size: usize::MAX, ..config() },
        Config { gate_lower_bound: Some(f32::NAN), ..config() },
        Config { gate_lower_bound: Some(f32::NEG_INFINITY), ..config() },
        Config { gate_lower_bound: Some(0.0), ..config() },
        Config { rms_norm_eps: f32::INFINITY, ..config() }, Config { rms_norm_eps: -1.0, ..config() }] {
        assert!(bad.validate().is_err());
        assert!(State::new(bad).is_err());
    }
}

#[test]
fn all_weight_shapes_and_finite_values_are_checked() {
    let owned = Owned::hand();
    let w = owned.weights();
    for bad in [Weights { q_proj: &[], ..w }, Weights { k_proj: &[], ..w }, Weights { v_proj: &[], ..w },
        Weights { q_conv1d: &[], ..w }, Weights { k_conv1d: &[], ..w }, Weights { v_conv1d: &[], ..w },
        Weights { b_proj: &[], ..w }, Weights { f_a_proj: &[], ..w }, Weights { f_b_proj: &[], ..w },
        Weights { g_a_proj: &[], ..w }, Weights { g_b_proj: &[], ..w }, Weights { dt_bias: &[], ..w },
        Weights { a_log: &[], ..w }, Weights { o_norm: &[], ..w }, Weights { o_proj: &[], ..w }] {
        assert!(Reference::new(config(), bad).is_err());
    }
    let mut owned = Owned::hand();
    owned.q_proj[0] = f32::NAN;
    assert!(Reference::new(config(), owned.weights()).is_err());
    owned.q_proj[0] = 1.0;
    owned.dt_bias[0] = f32::INFINITY;
    assert!(Reference::new(config(), owned.weights()).is_err());
}

#[test]
fn invalid_input_leaves_state_bits_unchanged() {
    let owned = Owned::hand();
    let reference = Reference::new(config(), owned.weights()).unwrap();
    let initial = hand_state();
    for input in [&[][..], &[1.0][..], &[1.0, 2.0, 3.0][..], &[f32::NAN, 1.0][..], &[1.0, f32::INFINITY][..]] {
        let mut state = initial.clone();
        assert!(reference.step(&mut state, input).is_err());
        same_state(&state, &initial);
    }
    for input in [&[1.0][..], &[1.0, 2.0, 3.0][..], &[1.0, 2.0, f32::NAN, 1.0][..]] {
        let mut state = initial.clone();
        assert!(reference.scan(&mut state, input).is_err());
        same_state(&state, &initial);
    }
    let mut wrong = State::new(Config { conv_kernel_size: 1, ..config() }).unwrap();
    let before = wrong.clone();
    assert!(reference.step(&mut wrong, &[1.0, 2.0]).is_err());
    same_state(&wrong, &before);
}

#[test]
fn arithmetic_error_leaves_step_and_chunk_state_unchanged() {
    let mut owned = Owned::hand();
    owned.o_proj.fill(f32::MAX);
    let reference = Reference::new(config(), owned.weights()).unwrap();
    let initial = hand_state();
    let mut state = initial.clone();
    assert!(reference.step(&mut state, &[2.0, -1.0]).is_err());
    same_state(&state, &initial);
    // The first zero token is valid. A later output projection overflows.
    let initial = State::new(config()).unwrap();
    let mut state = initial.clone();
    assert_eq!(reference.step(&mut state, &[0.0, 0.0]).unwrap(), [0.0, 0.0]);
    let mut state = initial.clone();
    assert!(reference.scan(&mut state, &[0.0, 0.0, 2.0, -1.0]).is_err());
    same_state(&state, &initial);
}
