//! Host proof of the composite's public layout and ordered recurrence.
use spark_gpu_probe::{causal_ops as scalar, next_math::exp, prompt_ops as rows, recurrent_rows as fused};

#[test]
fn recorded_payload_keeps_every_argument_bit() {
    let shape = [4_u32, 32, (-5.0_f32).to_bits(), (-0.0_f32).to_bits()];
    let addresses: [u64; 16] = std::array::from_fn(|i| 0x8102_0304_0506_0700 | i as u64);
    let args = fused::Arguments {
        shape: fused::Shape { tokens: shape[0], heads: shape[1],
            bound: f32::from_bits(shape[2]), epsilon: f32::from_bits(shape[3]) },
        inputs: addresses[..7].try_into().unwrap(),
        gate: addresses[7],
        weights: addresses[8..11].try_into().unwrap(),
        outputs: addresses[11..].try_into().unwrap(),
    };
    let expected: Vec<_> = shape.into_iter().flat_map(u32::to_ne_bytes)
        .chain(addresses.into_iter().flat_map(u64::to_ne_bytes)).collect();
    assert_eq!(args.payload().as_slice(), expected.as_slice());
}

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| {
        if (i + salt).is_multiple_of(31) { -0.0 }
        else { (((i * 73 + salt * 23) % 137) as f32 - 68.0) / 128.0 }
    }).collect()
}

fn same(name: &str, a: &[f32], b: &[f32], heads: usize, tokens: usize) {
    assert_eq!(a.len(), b.len(), "{name} extent");
    let diff = a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits());
    assert_eq!(diff, None, "{name} heads={heads} tokens={tokens}");
}

fn check(heads: usize, tokens: usize) {
    let (d, taps) = (fused::DIM, fused::TAPS);
    let width = heads * d;
    let shape = fused::Shape { tokens: tokens as u32, heads: heads as u32, bound: -5.0, epsilon: 1e-6 };
    let (inputs, weights, outputs) = shape.lengths(true).unwrap();
    let [q, k, v, history, forget, beta, state] = std::array::from_fn(|i| pattern(inputs[i], i + 1));
    let [cw, pw, nw] = std::array::from_fn(|i| pattern(weights[i], i + 8));
    let gate = pattern(tokens * width, 11);
    let mut conv = rows::Shape::new(rows::Op::ConvRows, tokens).unwrap();
    conv.heads = shape.heads;
    conv.dim = d as u32;
    conv.taps = taps as u32;
    let one = scalar::Shape { op: scalar::Op::Conv, heads: shape.heads, dim: d as u32, taps: taps as u32,
        rows: 0, cols: 0, epsilon: shape.epsilon, bound: shape.bound };
    let prepare = scalar::Shape { op: scalar::Op::Prepare, ..one };
    let norm = scalar::Shape { op: scalar::Op::GatedNorm, ..one };

    let convolution = rows::reference(conv, [&q, &k, &v, &history, &[]], &cw).unwrap();
    let history_rows = rows::reference(rows::Shape { op: rows::Op::HistoryRows, ..conv },
        [&q, &k, &v, &history, &[]], &[]).unwrap();
    let prepared_rows = rows::reference(rows::Shape::wrapped(prepare, tokens).unwrap(),
        [&convolution, &forget, &beta, &[], &[]], &pw).unwrap();
    let decay_rows = rows::reference(rows::Shape { op: rows::Op::StepDecay, ..conv },
        [&prepared_rows, &[], &[], &[], &[]], &[]).unwrap();
    let columns = rows::reference(rows::Shape { op: rows::Op::StepRows, ..conv },
        [&prepared_rows, &state, &decay_rows, &[], &[]], &[]).unwrap();
    let (state_rows, recurrence) = columns.split_at(width * d);
    let gated_rows = rows::reference(rows::Shape::wrapped(norm, tokens).unwrap(),
        [recurrence, &gate, &[], &[], &[]], &nw).unwrap();

    let mut carried_history = history;
    let mut carried_state = state;
    let mut prepared = Vec::new();
    let mut decay = Vec::new();
    let mut gated = Vec::new();
    for t in 0..tokens {
        let features = t * width..(t + 1) * width;
        let packed = [&q[features.clone()], &k[features.clone()], &v[features.clone()]].concat();
        let convolution = scalar::reference(one, [&packed, &carried_history, &[], &[]], &cw).unwrap();
        let mut current = Vec::new();
        carried_history.clear();
        for plane in convolution.chunks_exact(taps * width) {
            let (row, old) = plane.split_at(width);
            current.extend_from_slice(row);
            carried_history.extend_from_slice(old);
        }

        let ready = scalar::reference(prepare, [&current, &forget[features.clone()],
            &beta[t * heads..(t + 1) * heads], &[]], &pw).unwrap();
        prepared.extend_from_slice(&ready);
        for record in ready.chunks_exact(4 * d + 1) {
            decay.extend(record[3 * d..4 * d].iter().copied().map(exp));
        }
        let block = scalar::reference(scalar::Shape { op: scalar::Op::Step, ..one },
            [&ready, &carried_state, &[], &[]], &[]).unwrap();
        let mut recurrent = Vec::new();
        carried_state.clear();
        for head in block.chunks_exact(d * d + d) {
            let (state, row) = head.split_at(d * d);
            carried_state.extend_from_slice(state);
            recurrent.extend_from_slice(row);
        }
        gated.extend(scalar::reference(norm, [&recurrent, &gate[features], &[], &[]], &nw).unwrap());
    }

    let group = [&history_rows[..], state_rows, &gated_rows, &prepared_rows, &decay_rows];
    let ordered = [&carried_history[..], &carried_state, &gated, &prepared, &decay];
    assert_eq!(group.map(<[f32]>::len), outputs);
    assert_eq!(shape.lengths(false).unwrap().2, [outputs[0], outputs[1], outputs[2], 0, 0]);
    for ((name, group), ordered) in ["history", "state", "gated", "prepare", "decay"].into_iter().zip(group).zip(ordered) {
        same(name, group, ordered, heads, tokens);
    }
}

#[test]
fn short_rows_match_serial_history_state_and_kept_words() {
    for heads in [16, 32, 64] {
        for tokens in 2..=8 { check(heads, tokens); }
    }
}

#[test]
fn composite_rejects_other_shapes_and_invalid_parameters() {
    let ok = fused::Shape { tokens: 4, heads: 32, bound: -5.0, epsilon: 1e-6 };
    assert!(ok.admitted());
    for bad in [
        fused::Shape { tokens: 1, ..ok }, fused::Shape { tokens: 9, ..ok },
        fused::Shape { heads: 15, ..ok }, fused::Shape { heads: 17, ..ok }, fused::Shape { heads: 128, ..ok },
        fused::Shape { bound: 0.0, ..ok }, fused::Shape { bound: f32::NEG_INFINITY, ..ok },
        fused::Shape { epsilon: 0.0, ..ok }, fused::Shape { epsilon: f32::NAN, ..ok },
    ] {
        assert!(bad.lengths(true).is_none());
    }
}

#[cfg(target_os = "linux")]
mod device {
    use super::{fused, pattern, same};
    use spark_gpu_probe::{driver::{Backend, DeviceVector, with_backend}, layer_ptx, resident_launch::Kind};

    struct Inputs<'d> {
        values: [DeviceVector<'d>; 7],
        gate: DeviceVector<'d>,
        weights: [DeviceVector<'d>; 3],
    }

    impl<'d> Inputs<'d> {
        fn upload(b: &Backend<'d>, s: fused::Shape, salt: usize) -> Result<Self, String> {
            let (inputs, weights, outputs) = s.lengths(false).ok_or("recurrent proof shape")?;
            let upload = |count, offset| b.upload_vector(&pattern(count, salt + offset));
            let values = inputs.into_iter().enumerate().map(|(i, n)| upload(n, i)).collect::<Result<Vec<_>, _>>()?;
            let weights = weights.into_iter().enumerate().map(|(i, n)| upload(n, 8 + i)).collect::<Result<Vec<_>, _>>()?;
            Ok(Self { values: values.try_into().map_err(|_| "recurrent inputs")?,
                gate: upload(outputs[2], 11)?, weights: weights.try_into().map_err(|_| "recurrent weights")? })
        }

        fn all(&self) -> impl Iterator<Item = &DeviceVector<'d>> {
            self.values.iter().chain([&self.gate]).chain(&self.weights)
        }

        fn launch(&self, b: &Backend<'d>, s: fused::Shape, keep: bool) -> Result<Vec<DeviceVector<'d>>, String> {
            let (next, kept) = b.recurrent_rows(s, self.values.each_ref(), &self.gate, self.weights.each_ref(), keep)?;
            Ok(next.into_iter().chain(kept.into_iter().flatten()).collect())
        }
    }

    #[test]
    #[ignore = "requires CUDA and the compiled layer image"]
    fn capture_and_replay_keep_dropped_recurrent_inputs() {
        with_backend(0, layer_ptx().unwrap(), 256 << 20, |b| {
            let mut cases = 0;
            let mut deferred = 0;
            let mut captures = 0;
            let mut replays = 0;
            let mut words = 0;
            let queue = b.resident_queue()?;
            b.enable_resident_launches()?;
            for heads in [16, 32, 64] {
                b.pause_resident_launches(true)?;
                let mut ready = Vec::new();
                // Keep all input sets live until the capture starts. Every command
                // then names a different set of addresses, even on a cached grid.
                for (i, (tokens, keep)) in [(2, true), (4, false), (8, true), (4, true)].into_iter().enumerate() {
                    let s = fused::Shape {heads, tokens, bound: -5.0, epsilon: 1e-6};
                    let inputs = Inputs::upload(b, s, 97 * (i + 1))?;
                    let direct = inputs.launch(b, s, keep)?;
                    let want = b.read_vectors_values(&direct)?;
                    ready.push((s, keep, inputs, want));
                }
                b.pause_resident_launches(false)?;
                for (i, (s, keep, inputs, want)) in ready.into_iter().enumerate() {
                    let (fence, mark) = b.fence_frees()?;
                    b.wait_fence_frees(fence, mark)?;
                    b.synchronize()?;
                    b.resident_row_pass(s.tokens as usize);
                    let before = b.stats();
                    let outputs = inputs.launch(b, s, keep)?;
                    let queued = b.stats().since(before);
                    assert_eq!(queued.recorded_kinds[Kind::RowRecurrent as usize], 1);
                    if i == 0 { assert_eq!(queued.graph_launches, 0, "first capture submitted before the owner drop"); }

                    let old: Vec<_> = inputs.all().map(DeviceVector::device_address).collect();
                    let pressure_words = inputs.values[6].len();
                    drop(inputs);
                    let pressure = b.vector(pressure_words)?;
                    assert!(!old.contains(&pressure.device_address()), "pending storage returned to the pool");
                    assert_eq!(b.stats().since(before).stream_wait_calls, queued.stream_wait_calls, "dropping owners drained the stream");
                    let (fence, new_mark) = b.fence_frees()?;
                    assert_eq!(new_mark - mark, old.len(), "a source did not enter the deferred list");
                    b.wait_fence_frees(fence, new_mark)?;
                    let ran = b.stats().since(before);
                    assert_eq!(ran.graph_launches, 1);
                    assert_eq!(ran.graph_captures, u128::from(i == 0), "the retained grid did not replay");
                    assert_eq!(ran.zero_fill_launches, 0);
                    let got = b.read_vectors_values(&outputs)?;
                    assert_eq!(got.len(), want.len());
                    for (got, want) in got.iter().zip(&want) {
                        same("recorded output", got, want, heads as usize, s.tokens as usize);
                        words += got.len();
                    }
                    drop(pressure);
                    cases += 1;
                    deferred += new_mark - mark;
                    captures += ran.graph_captures;
                    replays += ran.graph_launches;
                }
            }
            drop(queue);
            println!("RECURRENT OWNER PROOF PASS cases={cases} deferred_owners={deferred} graph_captures={captures} graph_launches={replays} output_words={words}");
            Ok(())
        }).unwrap();
    }
}
