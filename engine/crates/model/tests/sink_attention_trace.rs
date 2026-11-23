//! Sink attention and short convolutions against a saved reference trace and the real stored weights.
//! Ignored by default: the weight file and the trace are private. They stay
//! outside the tracked tree, in the ignored folder `.local/capped-reference/`
//! at the repository root:
//!
//! - `model/`: the model folder (`config.json` and one weight file). A link is enough.
//! - `trace/`: a trace folder with one `trace-index*.json` and the files it names.
//!
//! The test reads no environment variable. It takes the token-by-token run of
//! the index and checks two layer visits over every token: a layer with a
//! window and a layer with full reach. The visit input comes from the trace.
//! The trace holds no attention output, so the test compares two later
//! tensors: the input of the expert block and the state after the visit. For
//! each it prints one line. It then runs each opposite layout, which must
//! differ from the trace in nearly every value.
//!
//! The source's library does not fix the order of its sums, so this test
//! cannot ask for equal bits. Its floor is the trace against itself: one call
//! over the prompt against token by token. The test prints that floor.
//! The block mix and the weighted norm below are small local stand-ins for
//! operations of other modules. They call the platform exponential.
use engine_model::{flash::nvfp4::{bf16_to_f32, f32_to_bf16}, json::{self, Value}, safetensors::{Dtype, Limits, TensorFile},
    short_conv::{FixedTaps, Slot, Taps, TokenTaps}, sink_attention::{Config, Reference, State, Weights}};
use std::path::{Path, PathBuf};

/// The two checked visits of the first pass: a layer with a window at a block edge, and the first layer with
/// full reach inside a block. The mix code of `visit` holds for these two only.
const CHECKED_VISITS: [usize; 2] = [0, 5];

fn local(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/capped-reference").join(name);
    assert!(path.exists(), "put {name} in .local/capped-reference/ for the ignored host test");
    path
}
fn rb(x: f32) -> f32 { bf16_to_f32(f32_to_bf16(x)) }
fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}
fn load(file: &TensorFile, name: &str) -> Vec<f32> {
    let tensor = file.header().tensors.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
    let mut bytes = vec![0u8; tensor.byte_len() as usize];
    file.read_range(name, 0, &mut bytes).unwrap();
    match tensor.dtype {
        Dtype::Bf16 => bytes.as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect(),
        Dtype::F32 => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        other => panic!("{name} has type {other:?}"),
    }
}
fn inverse_rms(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for &v in x { sum += v * v; }
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}
/// Stand-in: weighted norm, rounded to BF16.
fn weighted_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let scale = inverse_rms(x, eps);
    x.iter().zip(weight).map(|(&v, &w)| rb((v * scale) * w)).collect()
}
/// Stand-in: softmax blend of the plain-normed vectors of a block, all in f32.
fn block_mix(list: &[&[f32]], query: &[f32]) -> Vec<f32> {
    let n = query.len();
    let normed: Vec<Vec<f32>> = list.iter().map(|v| { let s = inverse_rms(v, 1e-6); v.iter().map(|&x| x * s).collect() }).collect();
    let scale = (1.0 / (n as f64).sqrt()) as f32;
    let scores: Vec<f32> = normed.iter().map(|v| { let mut s = 0.0f32; for i in 0..n { s += query[i] * v[i]; } s * scale }).collect();
    let max = scores.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    let mut out = vec![0.0f32; n];
    for (e, v) in exps.iter().zip(&normed) { let w = e / total; for i in 0..n { out[i] += w * v[i]; } }
    out
}
#[derive(Default)]
struct Diff { count: usize, equal: usize, one_step: usize, max_abs: f32, sum_sq: f64, trace_sq: f64 }
impl Diff {
    fn add(&mut self, mine: &[f32], theirs: &[f32]) {
        for (&a, &b) in mine.iter().zip(theirs) {
            let d = (a - b).abs();
            // One BF16 step at the size of the larger of the two values.
            let top = a.abs().max(b.abs());
            let step = if top == 0.0 { 0.0 } else { f32::from_bits(top.to_bits() & 0x7f80_0000) / 128.0 };
            self.count += 1;
            self.equal += usize::from(d == 0.0);
            self.one_step += usize::from(d <= step);
            self.max_abs = self.max_abs.max(d);
            self.sum_sq += (d as f64) * (d as f64);
            self.trace_sq += (b as f64) * (b as f64);
        }
    }
    fn line(&self) -> String {
        format!("values {} equal {} ({:.2}%) within one BF16 step {} ({:.2}%) largest difference {:e} relative error {:.3e}", self.count, self.equal,
            100.0 * self.equal as f64 / self.count as f64, self.one_step, 100.0 * self.one_step as f64 / self.count as f64, self.max_abs, (self.sum_sq / self.trace_sq).sqrt())
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum Layout { AsRead, FixedTapsReversed, TokenLagsReversed, TokenIndexLagMajor, NoRelativeTerm, NoSink }

/// One run of the trace index: states `[states, tokens, hidden]` and expert block input and output `[visits, 2, tokens, hidden]`.
struct Trace { tokens: usize, hidden: usize, states: Vec<f32>, expert_io: Vec<f32>, layers: Vec<usize>, outputs: Vec<usize> }
impl Trace {
    fn open(folder: &Path, entry: &Value) -> Self {
        let entry = entry.object().unwrap();
        let shape = |name: &str| -> (PathBuf, Vec<usize>) {
            let file = json::field(entry["files"].object().unwrap(), name).unwrap().object().unwrap();
            (folder.join(file["file"].string().unwrap()), file["shape"].array().unwrap().iter().map(|v| v.u64().unwrap() as usize).collect())
        };
        let ((state_file, s), (io_file, io)) = (shape("states"), shape("expert-io"));
        let visit = |name: &str| -> Vec<usize> { entry["visits"].array().unwrap().iter().map(|v| v.object().unwrap()[name].u64().unwrap() as usize).collect() };
        let trace = Self { tokens: s[1], hidden: s[2], states: floats(&state_file), expert_io: floats(&io_file), layers: visit("layer_module"), outputs: visit("state_index_of_its_output") };
        assert_eq!((io[1], io[2], io[3], io[0]), (2, trace.tokens, trace.hidden, trace.layers.len()));
        assert_eq!((trace.states.len(), trace.expert_io.len()), (s[0] * s[1] * s[2], io[0] * 2 * io[2] * io[3]));
        trace
    }
    fn state(&self, index: usize, token: usize) -> &[f32] { let at = (index * self.tokens + token) * self.hidden; &self.states[at..at + self.hidden] }
    fn expert(&self, visit: usize, side: usize, token: usize) -> &[f32] { let at = ((visit * 2 + side) * self.tokens + token) * self.hidden; &self.expert_io[at..at + self.hidden] }
}
fn reversed_rows(data: &[f32], width: usize) -> Vec<f32> { data.chunks_exact(width).flat_map(|row| row.iter().rev().copied().collect::<Vec<f32>>()).collect() }
fn kv_taps<'a>(a: &'a [f32], b: &'a [f32], window: bool, kv: usize, taps: usize, heads: usize, h: usize) -> Taps<'a> {
    if window { Taps::Fixed(FixedTaps::new(kv, taps, a).unwrap()) } else { Taps::TokenMade(TokenTaps::new(kv, heads, taps, h, a, b).unwrap()) }
}

/// Run one visit over every token of the trace. Returns the differences of the expert block input and of the state after the visit.
fn visit(file: &TensorFile, config_bytes: &[u8], trace: &Trace, visit: usize, layout: Layout) -> (Diff, Diff) {
    let layer = trace.layers[visit];
    assert_eq!(layer, visit, "the mix code here holds for the first pass only");
    let p = format!("model.llm.layers.{layer}.");
    // The kind of the layer comes from its tensors: a layer with fixed key taps has a window.
    let window = file.header().tensors.contains_key(&format!("{p}attn.k_sconv.weight"));
    let config = Config::from_json(config_bytes, window).unwrap();
    let (h, taps, kv, heads) = (config.hidden_size, config.conv_taps, config.kv_width(), config.num_kv_heads);
    let get = |name: &str| load(file, &format!("{p}{name}"));
    let (q, q_scale, k, v) = (get("attn.wq_du.weight"), get("attn.q_l2_clip.scale"), get("attn.wk_dv.weight"), get("attn.wv_dv.weight"));
    let (rel, rel_scale, mut table) = (get("attn.wr_du.weight"), get("attn.rel_l2_clip.scale"), get("attn.rel_logits_proj.proj"));
    let (mut sink, gate, o) = (get("attn.attn_sink_proj.weight"), get("attn.gate_attn.weight"), get("attn.wo_ud.weight"));
    if layout == Layout::NoRelativeTerm { table.fill(0.0); }
    if layout == Layout::NoSink { sink.fill(0.0); }
    let (mut attn_taps, mut mlp_taps) = (get("attn_sconv.weight"), get("mlp_sconv.weight"));
    if layout == Layout::FixedTapsReversed { attn_taps = reversed_rows(&attn_taps, taps); mlp_taps = reversed_rows(&mlp_taps, taps); }
    let (mlp_norm, mlp_query, eps) = (get("mlp_norm.weight"), load(file, &format!("model.llm.attn_res.mlp_queries.{layer}")), config.rms_norm_eps);
    let (mut k_a, mut v_a) = if window { (get("attn.k_sconv.weight"), get("attn.v_sconv.weight")) } else { (get("attn.k_sconv.static_w"), get("attn.v_sconv.static_w")) };
    let (mut k_b, mut v_b) = if window { (Vec::new(), Vec::new()) } else { (get("attn.k_sconv.weight_proj"), get("attn.v_sconv.weight_proj")) };
    if window {
        if layout == Layout::FixedTapsReversed { k_a = reversed_rows(&k_a, taps); v_a = reversed_rows(&v_a, taps); }
    } else {
        for (a, b) in [(&mut k_a, &mut k_b), (&mut v_a, &mut v_b)] {
            if layout == Layout::TokenLagsReversed {
                // The other lag order: static row n and projected tap n of each head weight the token (taps - 1 - n) back.
                *a = a.chunks_exact(kv).rev().flat_map(|row| row.to_vec()).collect();
                *b = b.chunks_exact(h * taps).flat_map(|head| head.chunks_exact(h).rev().flat_map(|row| row.to_vec()).collect::<Vec<f32>>()).collect();
            }
            if layout == Layout::TokenIndexLagMajor {
                // The other index layout: projection row = lag * heads + head.
                let mut swapped = vec![0.0f32; b.len()];
                for head in 0..heads { for lag in 0..taps {
                    swapped[(head * taps + lag) * h..(head * taps + lag + 1) * h].copy_from_slice(&b[(lag * heads + head) * h..(lag * heads + head + 1) * h]);
                } }
                *b = swapped;
            }
        }
    }
    let (k_conv, v_conv) = (kv_taps(&k_a, &k_b, window, kv, taps, heads, h), kv_taps(&v_a, &v_b, window, kv, taps, heads, h));
    let weights = Weights { q_proj: &q, q_scale: &q_scale, k_proj: &k, v_proj: &v, k_conv, v_conv,
        rel_proj: &rel, rel_scale: &rel_scale, rel_table: &table, sink_proj: &sink, gate_proj: &gate, o_proj: &o };
    let reference = Reference::new(config, weights).unwrap();
    let (after_attention, after_experts) = (FixedTaps::new(h, taps, &attn_taps).unwrap(), FixedTaps::new(h, taps, &mlp_taps).unwrap());
    let (mut state, mut histories) = (State::new(config).unwrap(), config.histories().unwrap());
    // Layer inputs: one norm in a window layer, three norms and three mix queries in a full layer.
    let norms: Vec<Vec<f32>> = if window { vec![get("attn_norm.weight")] } else { vec![get("q_input_norm.weight"), get("k_input_norm.weight"), get("v_input_norm.weight")] };
    let edge = layer == 0;
    let attn_queries = if edge { Vec::new() } else { load(file, &format!("model.llm.attn_res.attn_queries.{}", layer - 1)) };
    let (mut expert_input, mut state_after) = (Diff::default(), Diff::default());
    for t in 0..trace.tokens {
        let (x0, keep) = (trace.state(0, t), trace.state(visit, t));
        let mixed: Vec<Vec<f32>> = if edge { vec![keep.to_vec()] } else { attn_queries.chunks_exact(h).map(|query| block_mix(&[x0, keep], query)).collect() };
        let input = |i: usize| weighted_norm(&mixed[i.min(mixed.len() - 1)], &norms[i.min(norms.len() - 1)], eps);
        let (xq, xk, xv) = (input(0), input(1), input(2));
        let attention = reference.step(&mut state, &mut histories, &xq, &xk, &xv).unwrap();
        let c = after_attention.step(histories.get_mut(Slot::AfterAttention), &attention).unwrap();
        let r: Vec<f32> = if edge { c.clone() } else { keep.iter().zip(&c).map(|(&a, &b)| a + b).collect() };
        let list: Vec<&[f32]> = if edge { vec![keep, &r] } else { vec![x0, &r] };
        expert_input.add(&weighted_norm(&block_mix(&list, &mlp_query), &mlp_norm, eps), trace.expert(visit, 0, t));
        let f = after_experts.step(histories.get_mut(Slot::AfterExperts), trace.expert(visit, 1, t)).unwrap();
        let out: Vec<f32> = r.iter().zip(&f).map(|(&a, &b)| rb(a + b)).collect();
        state_after.add(&out, trace.state(trace.outputs[visit], t));
    }
    (expert_input, state_after)
}

#[test]
#[ignore = "requires the real weight file and a reference trace outside the repository"]
fn attention_and_convolutions_match_the_reference_trace() {
    let (model, folder) = (local("model"), local("trace"));
    let config_bytes = std::fs::read(model.join("config.json")).unwrap();
    let weights = std::fs::read_dir(&model).unwrap().map(|e| e.unwrap().path())
        .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with("model-") && p.extension().is_some_and(|e| e == "safetensors")).expect("weight file");
    let file = TensorFile::open(&weights, Limits::default()).unwrap();
    let mut indexes: Vec<PathBuf> = std::fs::read_dir(&folder).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("trace-index") && p.extension().is_some_and(|e| e == "json")).collect();
    indexes.sort();
    let index = json::parse(&std::fs::read(indexes.first().expect("a trace-index file in the trace folder")).unwrap(), json::Limits::default()).unwrap();
    let entries = index.object().unwrap()["files"].array().unwrap();
    let word = |entry: &Value, name: &str| entry.object().unwrap()[name].string().unwrap().to_string();
    let stepped = entries.iter().find(|e| word(e, "way") == "token-by-token").expect("a token-by-token run in the index");
    let whole = entries.iter().find(|e| word(e, "way") == "one-pass" && word(e, "effort") == word(stepped, "effort")).expect("a one-pass run in the index");
    let (trace, floor) = (Trace::open(&folder, stepped), Trace::open(&folder, whole));
    println!("run effort {} tokens {} hidden {}", word(stepped, "effort"), trace.tokens, trace.hidden);
    for at in CHECKED_VISITS {
        let (mut a, mut b) = (Diff::default(), Diff::default());
        for t in 0..trace.tokens { a.add(floor.expert(at, 0, t), trace.expert(at, 0, t)); b.add(floor.state(floor.outputs[at], t), trace.state(trace.outputs[at], t)); }
        println!("FLOOR visit {at} one-pass against token-by-token: expert block input: {}", a.line());
        println!("FLOOR visit {at} one-pass against token-by-token: state after the visit: {}", b.line());
        let window = file.header().tensors.contains_key(&format!("model.llm.layers.{}.attn.k_sconv.weight", trace.layers[at]));
        let (expert_input, state) = visit(&file, &config_bytes, &trace, at, Layout::AsRead);
        println!("visit {at} as read: expert block input: {}", expert_input.line());
        println!("visit {at} as read: state after the visit: {}", state.line());
        assert!(state.equal * 10 >= state.count * 9, "visit {at}: under 9 of 10 state values equal the trace");
        assert!(expert_input.equal * 5 >= expert_input.count * 4, "visit {at}: under 4 of 5 expert block inputs equal the trace");
        let (_, against_whole) = visit(&file, &config_bytes, &floor, at, Layout::AsRead);
        println!("visit {at} as read, against the one-pass run: state after the visit: {}", against_whole.line());
        let opposite: &[Layout] = if window { &[Layout::FixedTapsReversed, Layout::NoRelativeTerm, Layout::NoSink] }
            else { &[Layout::FixedTapsReversed, Layout::TokenLagsReversed, Layout::TokenIndexLagMajor, Layout::NoRelativeTerm, Layout::NoSink] };
        for &layout in opposite {
            let (_, state) = visit(&file, &config_bytes, &trace, at, layout);
            println!("visit {at} {layout:?}: state after the visit: {}", state.line());
            assert!(state.equal * 10 < state.count, "visit {at} {layout:?}: the opposite layout must differ from the trace");
        }
    }
}
