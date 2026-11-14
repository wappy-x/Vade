//! Compare the config-selected references with the frozen main arithmetic.
//! Source: main e9216b9bec6343184f44915236eb44fbc9bc16a1,
//! `flash/refcpu.rs`: `softmax`, `route`, `sigmoid`, `silu`, and `dense_expert`.
//! These are host checks. They do not establish device or token exactness.
use engine_model::{flash::refcpu, json, host_ops::{Router, Swiglu}};

fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|x| x.to_bits()).collect() }

// Keep this arithmetic frozen. Do not use the new operation implementation here.
fn main_route(logits: &[f32], topk: usize) -> (Vec<usize>, Vec<f32>) {
    let mut p = vec![0.0; logits.len()];
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for (o, &v) in p.iter_mut().zip(logits) { *o = (v - m).exp(); s += *o; }
    for o in p.iter_mut() { *o /= s; }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    order.truncate(topk);
    let s: f32 = order.iter().map(|&i| p[i]).sum();
    let w: Vec<f32> = order.iter().map(|&i| p[i] / s).collect();
    (order, w)
}

fn main_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| {
        let sigmoid = 1.0 / (1.0 + (-g).exp());
        let silu = g * sigmoid;
        silu * u
    }).collect()
}

fn current_configs() -> Vec<Vec<u8>> {
    // Flash fields come from Dims::real and Config::parse at the main SHA above.
    // Both current Flash storage formats use these operation fields.
    vec![
        include_bytes!("fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").to_vec(),
        include_bytes!("fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").to_vec(),
        br#"{"text_config":{"num_experts":512,"num_experts_per_tok":10,"hidden_act":"silu"}}"#.to_vec(),
        br#"{"text_config":{"num_experts":512,"num_experts_per_tok":10,"hidden_act":"silu","norm_topk_prob":true}}"#.to_vec(),
    ]
}

fn routing_shape(config: &[u8]) -> (usize, usize) {
    let doc = json::parse(config, json::Limits::default()).unwrap();
    let root = doc.object().unwrap();
    let text = root.get("text_config").unwrap().object().unwrap();
    (text["num_experts"].u64().unwrap() as usize, text["num_experts_per_tok"].u64().unwrap() as usize)
}

#[test]
fn current_softmax_router_matches_main_bits() {
    for config in current_configs() {
        let router = Router::from_config(&config).unwrap();
        let (count, topk) = routing_shape(&config);
        let mut extreme = vec![-f32::MAX; count];
        extreme[0] = f32::MAX;
        extreme[count - 1] = f32::MAX;
        let rows = [
            vec![0.0; count],
            (0..count).map(|i| if i % 2 == 0 { -0.0 } else { 0.0 }).collect(),
            (0..count).map(|i| i as f32 * 0.125 - 32.0).collect(),
            (0..count).map(|i| ((i * 73 + 19) % 257) as f32 * 0.03125 - 4.0).collect(),
            (0..count).map(|i| f32::from_bits((i % 127) as u32)).collect(),
            extreme,
        ];
        for row in rows {
            let expected = main_route(&row, topk);
            let legacy = refcpu::route(&row, topk);
            let actual = router.route(&row, None).unwrap();
            assert_eq!(legacy.0, expected.0);
            assert_eq!(bits(&legacy.1), bits(&expected.1));
            assert_eq!(actual.0, expected.0);
            assert_eq!(bits(&actual.1), bits(&expected.1));
        }
    }
}

#[test]
fn current_uncapped_swiglu_matches_main_bits() {
    let mut gate = vec![-f32::MAX, -100.0, -20.0, -1.0, -0.0, 0.0,
        f32::from_bits(1), -f32::from_bits(1), 1.0, 10.0, 20.0, 100.0];
    gate.extend((0..512).map(|i| ((i * 31 + 7) % 257) as f32 * 0.0625 - 8.0));
    let up: Vec<f32> = (0..gate.len()).map(|i| match i % 6 {
        0 => -2.0, 1 => -0.0, 2 => 0.0, 3 => 0.25, 4 => 2.0, _ => -0.5,
    }).collect();
    let expected = main_swiglu(&gate, &up);
    let legacy: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| refcpu::silu(g) * u).collect();
    assert_eq!(bits(&legacy), bits(&expected));
    for config in current_configs() {
        let swiglu = Swiglu::from_config(&config).unwrap();
        assert_eq!(bits(&swiglu.apply(&gate, &up).unwrap()), bits(&expected));
    }
}
