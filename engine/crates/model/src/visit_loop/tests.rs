use super::*;
use super::Source::{Embedding as E, Visit as V};

fn close(actual: &[f32], expected: &[f32], bound: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= bound, "{a} != {b}, bound {bound}"); }
}
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
fn config(hidden_size: usize, layers: usize, block_layers: usize, loop_start: usize, loop_end: usize, passes: usize) -> Config {
    Config { hidden_size, layers, block_layers, loop_start, loop_end, passes, gate_width: 1, rms_norm_eps: 1e-6 }
}
fn entry(layer: usize, pass: usize, slot: usize, boundary: bool, stream: Source, blocks: &[Source], messages: Option<(usize, &[Source])>) -> Visit {
    Visit { layer, pass, slot, boundary, stream, blocks: blocks.to_vec(),
        messages: messages.map(|(loop_layer, from)| MessageRead { loop_layer, from: from.to_vec() }) }
}

#[test]
fn four_layers_with_a_loop_over_two_and_two_passes() {
    // Blocks of one layer: each layer is a boundary and puts its input on the
    // block list. Layers 1 and 2 run two times. Pass 2 restarts from the
    // stream after visit 0 and from the list [embedding]. Its messages read
    // the block outputs of pass 1: the streams after visits 1 and 2.
    let s = Schedule::new(config(1, 4, 1, 1, 3, 2)).unwrap();
    let expected = [
        entry(0, 1, 0, true, E, &[], None),
        entry(1, 1, 1, true, V(0), &[E], None),
        entry(2, 1, 2, true, V(1), &[E, V(0)], None),
        entry(1, 2, 4, true, V(0), &[E], Some((0, &[V(1), V(2)]))),
        entry(2, 2, 5, true, V(3), &[E, V(0)], Some((1, &[V(1), V(2)]))),
        entry(3, 1, 3, true, V(4), &[E, V(0), V(3)], None),
    ];
    assert_eq!(s.visits(), &expected);
    assert_eq!((s.end_blocks(), s.end_stream()), (&[E, V(0), V(3), V(4)][..], V(5)));
    assert_eq!(s.config().visits(), 6);
}

#[test]
fn four_layers_in_blocks_of_two_with_the_loop_on_the_last_block() {
    // Layers 0 and 2 are boundaries. The loop holds one block, so a message
    // reads one vector: the stream that the pass before ended with.
    let s = Schedule::new(config(1, 4, 2, 2, 4, 2)).unwrap();
    let expected = [
        entry(0, 1, 0, true, E, &[], None),
        entry(1, 1, 1, false, V(0), &[E], None),
        entry(2, 1, 2, true, V(1), &[E], None),
        entry(3, 1, 3, false, V(2), &[E, V(1)], None),
        entry(2, 2, 4, true, V(1), &[E], Some((0, &[V(3)]))),
        entry(3, 2, 5, false, V(4), &[E, V(1)], Some((1, &[V(3)]))),
    ];
    assert_eq!(s.visits(), &expected);
    assert_eq!((s.end_blocks(), s.end_stream()), (&[E, V(1)][..], V(5)));
}

#[test]
fn twenty_four_layers_with_three_passes_over_twelve_give_48_visits() {
    // 24 layers in blocks of 6, the loop on layers 6 to 17, 3 passes.
    let s = Schedule::new(config(1, 24, 6, 6, 18, 3)).unwrap();
    let v = s.visits();
    assert_eq!(v.len(), 48);
    // Run order: layers 0-5, then 6-17 three times, then 18-23.
    let layers: Vec<usize> = (0..6).chain(6..18).chain(6..18).chain(6..18).chain(18..24).collect();
    assert_eq!(v.iter().map(|e| e.layer).collect::<Vec<_>>(), layers);
    let passes: Vec<usize> = (0..48).map(|k| if (18..30).contains(&k) { 2 } else if (30..42).contains(&k) { 3 } else { 1 }).collect();
    assert_eq!(v.iter().map(|e| e.pass).collect::<Vec<_>>(), passes);
    // Slots: 0-5, 6-17, 24-35, 36-47, 18-23. Each visit has its own slot.
    let slots: Vec<usize> = (0..18).chain(24..48).chain(18..24).collect();
    assert_eq!(v.iter().map(|e| e.slot).collect::<Vec<_>>(), slots);
    for e in v { assert_eq!(e.boundary, e.layer % 6 == 0); }
    // Each pass after the first restarts from the stream after visit 5.
    for (k, e) in v.iter().enumerate() {
        let expected = match k { 0 => E, 18 | 30 => V(5), _ => V(k - 1) };
        assert_eq!(e.stream, expected, "visit {k}");
    }
    // Block list before each visit.
    for (k, e) in v.iter().enumerate() {
        let expected: Vec<Source> = match k {
            0 => vec![],
            1..=6 | 18 | 30 => vec![E],
            7..=12 | 19..=24 | 31..=36 => vec![E, V(5)],
            13..=17 => vec![E, V(5), V(11)],
            25..=29 => vec![E, V(5), V(23)],
            37..=42 => vec![E, V(5), V(35)],
            _ => vec![E, V(5), V(35), V(41)],
        };
        assert_eq!(e.blocks, expected, "visit {k}");
    }
    // Messages: none in pass 1 and outside the loop. Pass 2 reads the two
    // block outputs of pass 1. Pass 3 reads those of pass 2, not of pass 1.
    for (k, e) in v.iter().enumerate() {
        let expected = match k {
            18..=29 => Some(MessageRead { loop_layer: k - 18, from: vec![V(11), V(17)] }),
            30..=41 => Some(MessageRead { loop_layer: k - 30, from: vec![V(23), V(29)] }),
            _ => None,
        };
        assert_eq!(e.messages, expected, "visit {k}");
    }
    // The end mix reads the embedding, block 0, blocks 1 and 2 of pass 3 and
    // the last stream. It reads no block output of pass 1 or 2.
    assert_eq!((s.end_blocks(), s.end_stream()), (&[E, V(5), V(35), V(41)][..], V(47)));
    // At visit 29: the embedding and the streams after visits 5, 11, 17, 23 and 28.
    assert_eq!(s.peak_held(), 6);
}

#[test]
fn one_pass_has_no_restart_and_no_messages() {
    let s = Schedule::new(config(1, 4, 2, 2, 4, 1)).unwrap();
    assert_eq!(s.visits().len(), 4);
    assert!(s.visits().iter().all(|e| e.messages.is_none() && e.pass == 1));
    assert_eq!(s.visits().iter().map(|e| e.stream).collect::<Vec<_>>(), [E, V(0), V(1), V(2)]);
    assert_eq!(s.peak_held(), 3);
}

#[test]
fn config_reads_the_loop_fields_and_refuses_the_rest() {
    let text = |extra: &str, lp: &str| format!(r#"{{"text_config":{{"hidden_size":8,"num_hidden_layers":24,"n_layers_per_attn_res_block":6,"rms_norm_eps":1e-06{extra},"loop_config":{{{lp}}}}}}}"#);
    let full = r#""core_start":6,"core_end":18,"num_passes":3,"gate_rank":4,"is_varloop":false"#;
    let c = Config::from_json(text("", full).as_bytes()).unwrap();
    assert_eq!(c, Config { hidden_size: 8, layers: 24, block_layers: 6, loop_start: 6, loop_end: 18, passes: 3, gate_width: 4, rms_norm_eps: 1e-6 });
    assert_eq!((c.loop_layers(), c.visits()), (12, 48));
    // A direct text config reads the same.
    let direct = text("", full);
    let inner = &direct[r#"{"text_config":"#.len()..direct.len() - 1];
    assert_eq!(Config::from_json(inner.as_bytes()).unwrap(), c);
    // No default fills a missing field.
    for missing in ["core_start", "core_end", "num_passes", "gate_rank"] {
        let lp: Vec<&str> = full.split(',').filter(|part| !part.contains(missing)).collect();
        assert!(Config::from_json(text("", &lp.join(",")).as_bytes()).is_err(), "{missing}");
    }
    assert!(Config::from_json(br#"{"hidden_size":8,"num_hidden_layers":24,"n_layers_per_attn_res_block":6,"rms_norm_eps":1e-06}"#).is_err());
    // A variable pass count is refused.
    assert!(Config::from_json(text("", &full.replace("false", "true")).as_bytes()).is_err());
    // The loop must start after layer 0, on a block edge, and end on one.
    let base = config(8, 24, 6, 6, 18, 3);
    for bad in [Config { passes: 0, ..base }, Config { loop_start: 0, ..base }, Config { loop_start: 7, ..base },
        Config { loop_end: 17, ..base }, Config { loop_end: 30, ..base }, Config { loop_start: 18, ..base },
        Config { block_layers: 0, ..base }, Config { hidden_size: 0, ..base }, Config { layers: 0, ..base },
        Config { gate_width: 0, ..base }, Config { rms_norm_eps: 0.0, ..base }, Config { passes: MAX_VISITS, ..base }] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(Schedule::new(bad).is_err());
    }
    assert!(base.validate().is_ok());
}

#[test]
fn run_hands_each_visit_the_vectors_that_the_list_names() {
    // The visit returns its own number. So each input shows which earlier
    // vector the driver handed in.
    let s = Schedule::new(config(1, 24, 6, 6, 18, 3)).unwrap();
    let mut seen = Vec::new();
    let streams = s.run(&[-1.0], |visit, inputs| {
        seen.push((inputs.stream[0], inputs.blocks.iter().map(|b| b[0]).collect::<Vec<_>>(),
            inputs.messages_from.as_ref().map(|m| m.iter().map(|b| b[0]).collect::<Vec<_>>())));
        assert_eq!(visit.messages.is_some(), inputs.messages_from.is_some());
        Ok(vec![seen.len() as f32 - 1.0])
    }).unwrap();
    assert_eq!(streams.len(), 48);
    // Visit 18 starts pass 2: stream after visit 5, list [embedding], messages from visits 11 and 17.
    assert_eq!(seen[18], (5.0, vec![-1.0], Some(vec![11.0, 17.0])));
    // Visit 30 starts pass 3 from the same stream and list. Its messages come from pass 2.
    assert_eq!(seen[30], (5.0, vec![-1.0], Some(vec![23.0, 29.0])));
    assert_eq!(seen[41], (40.0, vec![-1.0, 5.0, 35.0], Some(vec![23.0, 29.0])));
    // Visit 42 goes on from the end of pass 3.
    assert_eq!(seen[42], (41.0, vec![-1.0, 5.0, 35.0], None));
    assert_eq!(seen[47], (46.0, vec![-1.0, 5.0, 35.0, 41.0], None));
    assert_eq!(seen[6], (5.0, vec![-1.0], None));
    // A wrong output shape or a value that is not finite stops the run.
    assert!(s.run(&[-1.0], |_, _| Ok(vec![0.0, 0.0])).is_err());
    assert!(s.run(&[-1.0], |_, _| Ok(vec![f32::NAN])).is_err());
    assert!(s.run(&[-1.0, 0.0], |_, _| Ok(vec![0.0])).is_err());
}

#[test]
fn messages_blend_the_normed_block_outputs_of_the_pass_before() {
    // Hidden size 4: the score scale is 0.5. The two sources norm to
    // [2,0,0,0] and [0,-2,0,0]. Loop layer 1 has these queries:
    // query [ln 3,0,0,0]: scores [ln 3, 0], weights [3/4, 1/4].
    // key [0,-ln 3,0,0]: scores [0, ln 3], weights [1/4, 3/4].
    // value zero: weights [1/2, 1/2].
    // expert [ln 9,0,0,0]: scores [ln 9, 0], weights [9/10, 1/10].
    let c = config(4, 4, 1, 1, 3, 2);
    let (l3, l9) = (3.0f32.ln(), 9.0f32.ln());
    let mut queries = vec![0.0f32; 2 * MESSAGES * 4];
    queries[16] = l3;
    queries[21] = -l3;
    queries[28] = l9;
    let from: [&[f32]; 2] = [&[3.0, 0.0, 0.0, 0.0], &[0.0, -5.0, 0.0, 0.0]];
    let m = messages(&c, &queries, 1, &from).unwrap();
    assert_eq!(m.len(), MESSAGES);
    close(&m[0], &[1.5, -0.5, 0.0, 0.0], 2e-6);
    close(&m[1], &[0.5, -1.5, 0.0, 0.0], 2e-6);
    close(&m[2], &[1.0, -1.0, 0.0, 0.0], 2e-6);
    close(&m[3], &[1.8, -0.2, 0.0, 0.0], 2e-6);
    // Loop layer 0 has zero queries: four equal blends.
    for message in messages(&c, &queries, 0, &from).unwrap() { close(&message, &[1.0, -1.0, 0.0, 0.0], 2e-6); }
    // One source: the message is that source after the plain norm.
    close(&messages(&c, &queries, 1, &from[1..]).unwrap()[3], &[0.0, -2.0, 0.0, 0.0], 2e-6);
    assert!(messages(&c, &queries, 2, &from).is_err());
    assert!(messages(&c, &queries[1..], 1, &from).is_err());
    assert!(messages(&c, &queries, 1, &[]).is_err());
}

#[test]
fn stored_message_queries_merge_to_query_key_value_expert() {
    // Stored rows: [query 0, expert 0, query 1, expert 1] and [key 0, value 0, key 1, value 1].
    let merged = merge_message_queries(2, &[1.0, 1.5, 4.0, 4.5, 5.0, 5.5, 8.0, 8.5], &[2.0, 2.5, 3.0, 3.5, 6.0, 6.5, 7.0, 7.5]).unwrap();
    assert_eq!(merged, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]);
    assert!(merge_message_queries(2, &[0.0; 4], &[0.0; 8]).is_err());
    assert!(merge_message_queries(2, &[0.0; 6], &[0.0; 6]).is_err());
    assert!(merge_message_queries(0, &[], &[]).is_err());
}

// Gate tests use a norm epsilon that makes the divisors exact, as the config
// supplies that epsilon. With hidden size 2 and epsilon 3, the vector [1,1]
// has mean(x^2) + 3 = 4: its plain norm is [1/2, 1/2].
fn gate_config(gate_width: usize) -> Config {
    Config { hidden_size: 2, layers: 4, block_layers: 1, loop_start: 1, loop_end: 3, passes: 2, gate_width, rms_norm_eps: 3.0 }
}

#[test]
fn a_zero_gate_returns_the_value() {
    // Zero forget and gain weights: tanh(0) = 0, so (1 - 0) * value + 0 * message.
    let c = gate_config(1);
    let gate = Gate::new(&c, GateWeights { down: &[1.0, -2.0, 0.5, 4.0], forget: &[0.0, 0.0], gain: &[0.0, 0.0] }).unwrap();
    let value = [0.3, -7.25];
    assert_eq!(bits(&gate.apply(&value, &[100.0, -3.0]).unwrap()), bits(&value));
}

#[test]
fn gate_hand_values() {
    // value [1,1] norms to [1/2,1/2]. message [1,-1] norms to [1/2,-1/2].
    // Inner: [1,1,2,0] . [1/2,1/2,1/2,-1/2] = 2.
    // tanh(ln(3)/2) = 1/2. Forget weight [ln(3)/4, 0]: forget = [1/2, 0].
    // Gain weight [0, -ln(3)/4]: gain = [0, -1/2].
    // Result: [(1 - 1/2) * 1 + 0 * 1, (1 - 0) * 1 + (-1/2) * (-1)] = [1/2, 3/2].
    let a = 0.25 * 3.0f32.ln();
    let c = gate_config(1);
    let (forget, gain) = ([a, 0.0], [0.0, -a]);
    let gate = Gate::new(&c, GateWeights { down: &[1.0, 1.0, 2.0, 0.0], forget: &forget, gain: &gain }).unwrap();
    close(&gate.apply(&[1.0, 1.0], &[1.0, -1.0]).unwrap(), &[0.5, 1.5], 1e-6);
}

#[test]
fn gate_weights_are_row_major_by_output() {
    // Gate width 2. Down row 0 picks the first normed value number: 1/2.
    // Down row 1 picks the second normed message number: -1/2.
    // Forget is [hidden][width] = [[ln 3, 0], [0, ln 3]]: forget = [tanh(ln(3)/2), tanh(-ln(3)/2)] = [1/2, -1/2].
    // Gain is [[0, -ln 3], [0, 0]]: gain = [tanh(ln(3)/2), 0] = [1/2, 0].
    // Result: [(1 - 1/2) * 1 + 1/2 * 1, (1 + 1/2) * 1 + 0 * (-1)] = [1, 3/2].
    let l = 3.0f32.ln();
    let c = gate_config(2);
    let (forget, gain) = ([l, 0.0, 0.0, l], [0.0, -l, 0.0, 0.0]);
    let gate = Gate::new(&c, GateWeights { down: &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], forget: &forget, gain: &gain }).unwrap();
    close(&gate.apply(&[1.0, 1.0], &[1.0, -1.0]).unwrap(), &[1.0, 1.5], 1e-6);
}

#[test]
fn gate_rounds_its_inner_input_and_its_inner_product_to_bf16() {
    // Both vectors have x^2 + y^2 = 2, so each divisor is sqrt(1 + 3) = 2.
    // value[0] = 1.001 norms to 0.5005. BF16 keeps 8 bits: it rounds to 0.5.
    // message[0] = 1.0079 norms to 0.50395. It rounds to 0.50390625.
    // Inner: 0.5 + 0.50390625 = 1.00390625. That is the middle between the BF16
    // numbers 1 and 1.0078125. It rounds to the even one: 1.
    // So forget[0] = tanh(1). With no rounding it would be tanh(1.00445).
    let (x, a) = (1.001f32, 1.0079f32);
    let (value, message) = ([x, (2.0 - x * x).sqrt()], [a, (2.0 - a * a).sqrt()]);
    let c = gate_config(1);
    let gate = Gate::new(&c, GateWeights { down: &[1.0, 0.0, 1.0, 0.0], forget: &[1.0, 0.0], gain: &[0.0, 0.0] }).unwrap();
    let out = gate.apply(&value, &message).unwrap();
    close(&out, &[(1.0 - 1.0f32.tanh()) * x, value[1]], 1e-6);
    assert!((out[0] - (1.0 - 1.00445f32.tanh()) * x).abs() > 1e-4);
}

#[test]
fn the_gate_norm_sums_squares_in_chunks_of_128() {
    // Elements 0, 128 and 129. Chunk 0 sums to 4096^2 = 2^24. Chunk 1 sums
    // to 1 + 1 = 2. The chunk sums join: 2^24 + 2. One 32-lane tree over all
    // elements puts elements 0 and 128 in lane 0, where 2^24 + 1 rounds back
    // to 2^24, and the last join does the same.
    let mut x = [0.0f32; 130];
    (x[0], x[128], x[129]) = (4096.0, 1.0, 1.0);
    assert_eq!(chunked_square_sum(&x), 16_777_218.0);
    assert_eq!(block_mix::square_sum(&x), 16_777_216.0);
    // The norm uses the chunk order, and each result is a BF16 value.
    let scale = 1.0 / (16_777_218.0f32 / 130.0 + 3.0).sqrt();
    let expected: Vec<f32> = x.iter().map(|&v| bf16(v * scale)).collect();
    assert_eq!(bits(&gate_norm(&x, 3.0).unwrap()), bits(&expected));
    assert!(gate_norm(&[f32::NAN], 3.0).is_err());
}

#[test]
fn gate_refuses_bad_weights_and_inputs() {
    let c = gate_config(1);
    let good = GateWeights { down: &[1.0, 1.0, 2.0, 0.0], forget: &[0.0, 0.0], gain: &[0.0, 0.0] };
    assert!(Gate::new(&c, good).is_ok());
    // 1.001 is not a BF16 value.
    assert!(Gate::new(&c, GateWeights { down: &[1.001, 1.0, 2.0, 0.0], ..good }).is_err());
    assert!(Gate::new(&c, GateWeights { down: &[1.0, 1.0, 2.0], ..good }).is_err());
    assert!(Gate::new(&c, GateWeights { forget: &[0.0], ..good }).is_err());
    assert!(Gate::new(&c, GateWeights { gain: &[0.0, f32::NAN], ..good }).is_err());
    let gate = Gate::new(&c, good).unwrap();
    assert!(gate.apply(&[1.0], &[1.0, 1.0]).is_err());
    assert!(gate.apply(&[1.0, 1.0], &[1.0, f32::INFINITY]).is_err());
}

// Layer tests use a hidden size of 1. A number norms to 1 or -1, to 5 parts
// in 10^7 for a size of 1 or more, and the score scale is 1. A query of
// ln(3)/2 gives a weight ratio of 3 between an entry at 1 and an entry at -1.
// The gate epsilon is 3/4, so 1/2 and -1/2 norm to themselves.
fn layer_config(passes: usize) -> Config {
    Config { hidden_size: 1, layers: 8, block_layers: 2, loop_start: 2, loop_end: 6, passes, gate_width: 1, rms_norm_eps: 0.75 }
}
const DOWN: [f32; 2] = [1.0, -1.0];
fn gate_set<'a>(forget: &'a [[f32; 1]; 4], gain: &'a [[f32; 1]; 4]) -> [GateWeights<'a>; 4] {
    [0, 1, 2, 3].map(|i| GateWeights { down: &DOWN, forget: &forget[i], gain: &gain[i] })
}

#[test]
fn the_first_pass_has_no_messages_and_skips_the_gates() {
    // Entries 2 and -3 norm to 1 and -1. Scores [q, -q] with q = ln(3)/2:
    // weights [3/4, 1/4]. Mix: 3/4 - 1/4 = 1/2, for query, key and value.
    let q = [0.5 * 3.0f32.ln()];
    let (forget, gain) = ([[5.0]; 4], [[7.0]; 4]);
    let c = layer_config(3);
    let layer = Layer::new(&c, 2, LayerWeights { attention_queries: Some(&q), expert_query: &q, gates: Some(gate_set(&forget, &gain)) }).unwrap();
    let inputs = layer.attention_inputs(&[&[2.0]], &[-3.0], None).unwrap();
    for input in &inputs { close(input, &[0.5], 1e-6); }
    assert_eq!(bits(&inputs[0]), bits(&inputs[1]));
    assert_eq!(bits(&inputs[0]), bits(&inputs[2]));
    // With messages the gates change the inputs, so the case above did skip them.
    let gated = layer.attention_inputs(&[&[2.0]], &[-3.0], Some(&[vec![-0.5], vec![-0.5], vec![-0.5], vec![-0.5]])).unwrap();
    assert!((gated[0][0] - 0.5).abs() > 0.1);
}

#[test]
fn later_passes_gate_each_input_with_its_own_message() {
    // The mix gives 1/2 for each input, as in the first pass. Down is [1,-1].
    // Query: message -1/2. Inner: 1/2 + 1/2 = 1. Forget tanh(ln(3)/2) = 1/2,
    // gain 1/2: (1 - 1/2) * 1/2 + 1/2 * (-1/2) = 0.
    // Key: message 1/2. Inner: 1/2 - 1/2 = 0. Forget 0, gain 0: 1/2.
    // Value: message -1/2. Inner 1. Forget 1/2, gain 0: 1/4.
    let h = 0.5 * 3.0f32.ln();
    let q = [h];
    let (forget, gain) = ([[h], [h], [h], [0.0]], [[h], [h], [0.0], [h]]);
    let c = layer_config(3);
    let layer = Layer::new(&c, 2, LayerWeights { attention_queries: Some(&q), expert_query: &q, gates: Some(gate_set(&forget, &gain)) }).unwrap();
    let messages = [vec![-0.5], vec![0.5], vec![-0.5], vec![-0.5]];
    let inputs = layer.attention_inputs(&[&[2.0]], &[-3.0], Some(&messages)).unwrap();
    close(&inputs[0], &[0.0], 1e-6);
    close(&inputs[1], &[0.5], 1e-6);
    close(&inputs[2], &[0.25], 1e-6);
    // Expert input, not a boundary: residual -3 + 1 = -2. Entries 2 and -2
    // norm to 1 and -1: the mix is 1/2. Expert gate: message -1/2, inner 1,
    // forget 0, gain 1/2: 1/2 + 1/2 * (-1/2) = 1/4.
    let (residual, input) = layer.expert_input(false, &[&[2.0]], &[-3.0], &[1.0], Some(&messages)).unwrap();
    assert_eq!(bits(&residual), bits(&[-2.0]));
    close(&input, &[0.25], 1e-6);
    // The same call with no messages gives the mix alone.
    close(&layer.expert_input(false, &[&[2.0]], &[-3.0], &[1.0], None).unwrap().1, &[0.5], 1e-6);
    // Three messages are not a full set.
    assert!(layer.attention_inputs(&[&[2.0]], &[-3.0], Some(&messages[..3])).is_err());
}

#[test]
fn three_attention_queries_give_query_key_and_value_in_that_order() {
    // Entries norm to 1 and -1. Query row ln(3)/2: weights [3/4, 1/4], mix 1/2.
    // Key row -ln(3)/2: weights [1/4, 3/4], mix -1/2. Value row 0: mix 0.
    let h = 0.5 * 3.0f32.ln();
    let c = layer_config(3);
    let queries = [h, -h, 0.0];
    let layer = Layer::new(&c, 7, LayerWeights { attention_queries: Some(&queries), expert_query: &[0.0], gates: None }).unwrap();
    let inputs = layer.attention_inputs(&[&[2.0]], &[-3.0], None).unwrap();
    close(&inputs[0], &[0.5], 1e-6);
    close(&inputs[1], &[-0.5], 1e-6);
    close(&inputs[2], &[0.0], 1e-6);
}

#[test]
fn the_first_layer_reads_the_stream_with_no_mix() {
    let c = layer_config(3);
    let layer = Layer::new(&c, 0, LayerWeights { attention_queries: None, expert_query: &[0.0], gates: None }).unwrap();
    let inputs = layer.attention_inputs(&[], &[-3.0], None).unwrap();
    for input in &inputs { assert_eq!(bits(input), bits(&[-3.0])); }
}

#[test]
fn the_block_edge_puts_the_input_on_the_list_and_not_on_the_stream() {
    // Boundary: the residual is the attention output alone, 1. The list
    // gets the input: entries 2, -3 and 1 norm to 1, -1 and 1. Scores
    // [q, -q, q] with q = ln(3)/2: weights [3/7, 1/7, 3/7]. Mix: 5/7.
    let q = [0.5 * 3.0f32.ln()];
    let c = layer_config(1);
    let layer = Layer::new(&c, 4, LayerWeights { attention_queries: Some(&q), expert_query: &q, gates: None }).unwrap();
    let (residual, input) = layer.expert_input(true, &[&[2.0]], &[-3.0], &[1.0], None).unwrap();
    assert_eq!(bits(&residual), bits(&[1.0]));
    close(&input, &[5.0 / 7.0], 1e-6);
    // Not a boundary: the residual is -3 + 1 = -2. The list stays [2].
    // Entries 2 and -2: weights [3/4, 1/4]. Mix: 1/2.
    let (residual, input) = layer.expert_input(false, &[&[2.0]], &[-3.0], &[1.0], None).unwrap();
    assert_eq!(bits(&residual), bits(&[-2.0]));
    close(&input, &[0.5], 1e-6);
}

#[test]
fn the_residual_stays_f32_and_the_visit_output_rounds_once() {
    // 256 + 1 = 257 needs 9 bits. BF16 keeps 8. The residual keeps 257.
    let c = layer_config(1);
    let layer = Layer::new(&c, 3, LayerWeights { attention_queries: Some(&[0.0]), expert_query: &[0.0], gates: None }).unwrap();
    let (residual, _) = layer.expert_input(false, &[&[2.0]], &[256.0], &[1.0], None).unwrap();
    assert_eq!(bits(&residual), bits(&[257.0]));
    // 257 + 1 = 258 is a BF16 number. Rounding 257 first would give 256 + 1 = 257, then 256.
    assert_eq!(bits(&visit_output(&residual, &[1.0]).unwrap()), bits(&[258.0]));
    // 257 is the middle between 256 and 258. It rounds to the even one: 256.
    assert_eq!(bits(&visit_output(&residual, &[0.0]).unwrap()), bits(&[256.0]));
    assert!(visit_output(&residual, &[0.0, 0.0]).is_err());
    assert!(visit_output(&[f32::MAX], &[f32::MAX]).is_err());
}

#[test]
fn layer_weights_must_fit_the_place_of_the_layer() {
    let c = layer_config(3);
    let (forget, gain) = ([[0.0]; 4], [[0.0]; 4]);
    let gates = Some(gate_set(&forget, &gain));
    let q = [0.0f32];
    let w = LayerWeights { attention_queries: Some(&q), expert_query: &q, gates: None };
    // Layer 0 has no attention mix. Other layers need one, with 1 or 3 rows.
    assert!(Layer::new(&c, 0, w).is_err());
    assert!(Layer::new(&c, 1, LayerWeights { attention_queries: None, ..w }).is_err());
    assert!(Layer::new(&c, 1, LayerWeights { attention_queries: Some(&[0.0, 0.0]), ..w }).is_err());
    assert!(Layer::new(&c, 1, w).is_ok());
    // Gates belong to loop layers, and a loop layer needs them for passes 2 and 3.
    assert!(Layer::new(&c, 1, LayerWeights { gates, ..w }).is_err());
    assert!(Layer::new(&c, 6, LayerWeights { gates, ..w }).is_err());
    assert!(Layer::new(&c, 2, w).is_err());
    assert!(Layer::new(&c, 2, LayerWeights { gates, ..w }).is_ok());
    assert!(Layer::new(&layer_config(1), 2, w).is_ok());
    assert!(Layer::new(&c, 8, w).is_err());
    // Messages need gates.
    let plain = Layer::new(&c, 1, w).unwrap();
    assert!(plain.attention_inputs(&[&[2.0]], &[-3.0], Some(&[vec![0.5], vec![0.5], vec![0.5], vec![0.5]])).is_err());
}

// A body with fixed arithmetic for the driver test. It is not a model.
struct Toy;
fn toy_attention(layer: usize, query: &[f32], key: &[f32], value: &[f32]) -> Vec<f32> {
    (0..query.len()).map(|d| bf16(0.5 * query[d] - 0.25 * key[d] + value[(d + 1) % value.len()] + 0.125 * layer as f32)).collect()
}
fn toy_experts(layer: usize, input: &[f32]) -> Vec<f32> {
    input.iter().enumerate().map(|(d, &v)| bf16(1.5 * v - 0.375 * (d + layer) as f32)).collect()
}
impl Body for Toy {
    fn attention(&mut self, visit: &Visit, query: &[f32], key: &[f32], value: &[f32]) -> Result<Vec<f32>> { Ok(toy_attention(visit.layer, query, key, value)) }
    fn experts(&mut self, visit: &Visit, input: &[f32]) -> Result<Vec<f32>> { Ok(toy_experts(visit.layer, input)) }
}
fn numbers(count: usize, seed: &mut u32) -> Vec<f32> {
    (0..count).map(|_| { *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); (*seed >> 8) as f32 / 8_388_608.0 - 1.0 }).collect()
}

#[test]
fn the_driver_agrees_with_the_visits_written_out_one_by_one() {
    // 8 layers in blocks of 2, the loop on layers 2 to 5, 3 passes: 16 visits.
    // The same shape as 24, 6, 6 to 18 and 3, with smaller numbers.
    let (hidden, width) = (3usize, 2usize);
    let c = Config { hidden_size: hidden, layers: 8, block_layers: 2, loop_start: 2, loop_end: 6, passes: 3, gate_width: width, rms_norm_eps: 1e-6 };
    let mut seed = 11u32;
    // The last layer of each block has three attention queries.
    let attention: Vec<Vec<f32>> = (0..8).map(|layer| numbers(if layer % 2 == 1 { 3 * hidden } else { hidden }, &mut seed)).collect();
    let expert: Vec<Vec<f32>> = (0..8).map(|_| numbers(hidden, &mut seed)).collect();
    let down: Vec<Vec<f32>> = (0..16).map(|_| numbers(width * 2 * hidden, &mut seed).iter().map(|&v| bf16(v)).collect()).collect();
    let forget: Vec<Vec<f32>> = (0..16).map(|_| numbers(hidden * width, &mut seed)).collect();
    let gain: Vec<Vec<f32>> = (0..16).map(|_| numbers(hidden * width, &mut seed)).collect();
    let message_queries = numbers(4 * MESSAGES * hidden, &mut seed);
    let output_query = numbers(hidden, &mut seed);
    let embedding: Vec<f32> = numbers(hidden, &mut seed).iter().map(|&v| bf16(3.0 * v)).collect();
    let layers: Vec<LayerWeights<'_>> = (0..8).map(|layer| LayerWeights {
        attention_queries: if layer == 0 { None } else { Some(&attention[layer]) }, expert_query: &expert[layer],
        gates: (2..6).contains(&layer).then(|| [0, 1, 2, 3].map(|i| { let g = (layer - 2) * 4 + i; GateWeights { down: &down[g], forget: &forget[g], gain: &gain[g] } })),
    }).collect();
    let driver = Driver::new(c, Weights { layers: layers.clone(), message_queries: &message_queries, output_query: &output_query }).unwrap();
    let token = driver.token(&embedding, &mut Toy).unwrap();
    assert_eq!(token.streams.len(), 16);

    // The same token, one visit a line, with no schedule.
    let made: Vec<Layer<'_>> = layers.iter().enumerate().map(|(layer, w)| Layer::new(&c, layer, *w).unwrap()).collect();
    let step = |layer: usize, boundary: bool, blocks: &[&[f32]], stream: &[f32], from: Option<&[&[f32]]>| -> Vec<f32> {
        let read = from.map(|from| messages(&c, &message_queries, layer - 2, from).unwrap());
        let [query, key, value] = made[layer].attention_inputs(blocks, stream, read.as_deref()).unwrap();
        let attended = toy_attention(layer, &query, &key, &value);
        let (residual, input) = made[layer].expert_input(boundary, blocks, stream, &attended, read.as_deref()).unwrap();
        visit_output(&residual, &toy_experts(layer, &input)).unwrap()
    };
    let x0 = &embedding[..];
    // Layers 0 and 1, once. The stream after layer 1 is the output of block 0.
    let v0 = step(0, true, &[], x0, None);
    let b0 = step(1, false, &[x0], &v0, None);
    // Pass 1: no messages.
    let v2 = step(2, true, &[x0], &b0, None);
    let b1a = step(3, false, &[x0, &b0], &v2, None);
    let v4 = step(4, true, &[x0, &b0], &b1a, None);
    let b2a = step(5, false, &[x0, &b0, &b1a], &v4, None);
    // Pass 2: restart from block 0. Messages from the block outputs of pass 1.
    let from: [&[f32]; 2] = [&b1a, &b2a];
    let v6 = step(2, true, &[x0], &b0, Some(&from));
    let b1b = step(3, false, &[x0, &b0], &v6, Some(&from));
    let v8 = step(4, true, &[x0, &b0], &b1b, Some(&from));
    let b2b = step(5, false, &[x0, &b0, &b1b], &v8, Some(&from));
    // Pass 3: restart again. Messages from the block outputs of pass 2.
    let from: [&[f32]; 2] = [&b1b, &b2b];
    let v10 = step(2, true, &[x0], &b0, Some(&from));
    let b1c = step(3, false, &[x0, &b0], &v10, Some(&from));
    let v12 = step(4, true, &[x0, &b0], &b1c, Some(&from));
    let b2c = step(5, false, &[x0, &b0, &b1c], &v12, Some(&from));
    // Layers 6 and 7 go on from pass 3.
    let v14 = step(6, true, &[x0, &b0, &b1c], &b2c, None);
    let b3 = step(7, false, &[x0, &b0, &b1c, &b2c], &v14, None);
    let end = block_mix::end_mix(hidden, &output_query, &[x0, &b0, &b1c, &b2c], &b3).unwrap();
    let expected = [&v0, &b0, &v2, &b1a, &v4, &b2a, &v6, &b1b, &v8, &b2b, &v10, &b1c, &v12, &b2c, &v14, &b3];
    for (k, (actual, expected)) in token.streams.iter().zip(expected).enumerate() { assert_eq!(bits(actual), bits(expected), "visit {k}"); }
    assert_eq!(bits(&token.end), bits(&end));
    // The passes do differ: the messages change the stream.
    assert_ne!(bits(&b2a), bits(&b2b));
    assert_ne!(bits(&b2b), bits(&b2c));
    // Each stream holds BF16 values. The end mix result is f32 and is not rounded.
    for stream in &token.streams { assert!(stream.iter().all(|&v| bf16(v).to_bits() == v.to_bits())); }
    // A second run gives the same bits.
    assert_eq!(driver.token(&embedding, &mut Toy).unwrap(), token);
}

#[test]
fn the_driver_checks_its_weights_against_the_config() {
    let c = Config { hidden_size: 1, layers: 4, block_layers: 2, loop_start: 2, loop_end: 4, passes: 2, gate_width: 1, rms_norm_eps: 1e-6 };
    let (q, zero, down) = ([0.0f32], [0.0f32], [1.0f32, 1.0]);
    let gate = GateWeights { down: &down, forget: &zero, gain: &zero };
    let plain = LayerWeights { attention_queries: Some(&q), expert_query: &q, gates: None };
    let layers = vec![LayerWeights { attention_queries: None, ..plain }, plain, LayerWeights { gates: Some([gate; 4]), ..plain }, LayerWeights { gates: Some([gate; 4]), ..plain }];
    let message_queries = [0.0f32; 8];
    let good = Weights { layers: layers.clone(), message_queries: &message_queries, output_query: &q };
    let driver = Driver::new(c, good.clone()).unwrap();
    assert_eq!(driver.schedule().visits().len(), 6);
    assert!(driver.layer(3).is_some() && driver.layer(4).is_none());
    assert!(Driver::new(c, Weights { layers: layers[..3].to_vec(), ..good.clone() }).is_err());
    assert!(Driver::new(c, Weights { message_queries: &message_queries[..4], ..good.clone() }).is_err());
    assert!(Driver::new(c, Weights { output_query: &[0.0, 0.0], ..good.clone() }).is_err());
    let mut no_gates = layers.clone();
    no_gates[3] = plain;
    assert!(Driver::new(c, Weights { layers: no_gates, ..good.clone() }).is_err());
    // One pass needs no message queries and no gates.
    let one = Config { passes: 1, ..c };
    assert!(Driver::new(one, Weights { layers: vec![layers[0], plain, plain, plain], message_queries: &[], output_query: &q }).is_ok());
}
