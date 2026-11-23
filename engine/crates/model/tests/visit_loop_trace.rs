//! Ignored check of the block mix and the visit loop against a layer trace of the reference code.
//! It needs real files that are not in the repository. Put them (or links to them) in
//! `.local/visit-loop-trace/` at the repository root:
//!
//! - `model/`: the model folder (`config.json`, `model.safetensors.index.json` and the weight shards).
//! - `trace/`: the trace folder with one or more `trace-index-*.json` files and the files they name.
//!   One trace file set holds, for one prompt, the states `[visits + 3][tokens][hidden]` (the
//!   embedding after its norm, the stream after each visit, the input and the output of the last
//!   norm) and the input and output of the expert block of each visit `[visits][2][tokens][hidden]`,
//!   as little-endian f32.
//!
//! Run it from `engine/`:
//! `cargo +1.99.0 test --offline --locked -p engine-model --test visit_loop_trace -- --ignored --nocapture`
//!
//! Checks, for each trace file set and each token position:
//! A. The visit list made from `config.json` has the layers and passes of the trace, in run order.
//! B. Replay: each visit hands back the trace's own stream. The streams that the driver holds after
//!    the last layer of each block must be the trace's. This is zero by construction when the
//!    driver's indexing is right. It does not check arithmetic.
//! C. End mix: the mix over the embedding, the block outputs and the last stream of the trace, with
//!    the stored output query, against the trace's input of the last norm.
//! D. Expert input of every visit: the block mix, and in later passes the expert message and its
//!    gate, against the trace's input of the expert block. The residual of a visit is not in the
//!    trace. The check makes it from the trace: the stream after the visit minus the short
//!    convolution of the expert output. That stream is rounded to BF16, so the residual is known
//!    only to half a BF16 step of it. So D is an approximate check with a floor. Controls with one
//!    deliberate error each show how far a wrong reading moves the result.
use engine_model::{json::{self, Value, field}, visit_loop::{self, Config, GateWeights, Layer, LayerWeights, Schedule, Source}};
use std::{collections::BTreeMap, fs::File, io::{Read, Seek, SeekFrom}, path::{Path, PathBuf}};

type R<T> = Result<T, String>;
fn e<T>(r: engine_model::Result<T>) -> R<T> { r.map_err(|x| x.0) }
fn io<T>(r: std::io::Result<T>, what: &str) -> R<T> { r.map_err(|x| format!("{what}: {x}")) }
fn bf16(v: f32) -> f32 { let b = v.to_bits(); f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000) }
fn local(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/visit-loop-trace").join(name);
    assert!(path.exists(), "put {name} in .local/visit-loop-trace/ for the ignored trace test");
    path
}

/// One weight shard: its header index and reads of single tensors.
struct Shard { file: File, base: u64, index: BTreeMap<String, (String, Vec<usize>, u64, u64)> }
impl Shard {
    fn open(path: &Path) -> R<Self> {
        let mut file = io(File::open(path), "open weight shard")?;
        let mut n = [0u8; 8];
        io(file.read_exact(&mut n), "read header length")?;
        let n = u64::from_le_bytes(n);
        if n > 64 << 20 { return Err("weight shard header is too large".into()); }
        let mut text = vec![0u8; n as usize];
        io(file.read_exact(&mut text), "read header")?;
        let doc = e(json::parse(&text, json::Limits { bytes: 64 << 20, nodes: 4_000_000, ..Default::default() }))?;
        let mut index = BTreeMap::new();
        for (name, v) in e(doc.object())? {
            if name == "__metadata__" { continue; }
            let o = e(v.object())?;
            let dtype = e(e(field(o, "dtype"))?.string())?.to_string();
            let shape = e(e(field(o, "shape"))?.array())?.iter().map(|d| e(d.u64()).map(|d| d as usize)).collect::<R<Vec<_>>>()?;
            let offsets = e(e(field(o, "data_offsets"))?.array())?;
            if offsets.len() != 2 { return Err(format!("{name}: bad offsets")); }
            index.insert(name.clone(), (dtype, shape, e(offsets[0].u64())?, e(offsets[1].u64())?));
        }
        Ok(Self { file, base: 8 + n, index })
    }
}
/// The weight files of a model folder, found through its index file.
struct Store { folder: PathBuf, map: BTreeMap<String, String>, shards: BTreeMap<String, Shard> }
impl Store {
    fn open(folder: &Path) -> R<Self> {
        let text = io(std::fs::read(folder.join("model.safetensors.index.json")), "read the weight index")?;
        let doc = e(json::parse(&text, json::Limits { bytes: 64 << 20, nodes: 4_000_000, ..Default::default() }))?;
        let mut map = BTreeMap::new();
        for (name, shard) in e(e(field(e(doc.object())?, "weight_map"))?.object())? { map.insert(name.clone(), e(shard.string())?.to_string()); }
        Ok(Self { folder: folder.to_path_buf(), map, shards: BTreeMap::new() })
    }
    fn shard(&mut self, name: &str) -> R<&mut Shard> {
        let file = self.map.get(name).ok_or(format!("missing tensor {name}"))?.clone();
        if !self.shards.contains_key(&file) { self.shards.insert(file.clone(), Shard::open(&self.folder.join(&file))?); }
        Ok(self.shards.get_mut(&file).expect("shard was just opened"))
    }
    fn shape(&mut self, name: &str) -> R<Vec<usize>> {
        Ok(self.shard(name)?.index.get(name).ok_or(format!("missing tensor {name}"))?.1.clone())
    }
    /// Read one tensor as f32. BF16 widens with no change of value.
    fn get(&mut self, name: &str, shape: &[usize]) -> R<Vec<f32>> {
        let shard = self.shard(name)?;
        let (dtype, have, begin, end) = shard.index.get(name).ok_or(format!("missing tensor {name}"))?.clone();
        if have != shape { return Err(format!("{name}: shape {have:?}, expected {shape:?}")); }
        let mut bytes = vec![0u8; (end - begin) as usize];
        io(shard.file.seek(SeekFrom::Start(shard.base + begin)), "seek")?;
        io(shard.file.read_exact(&mut bytes), "read tensor")?;
        let count: usize = shape.iter().product();
        match dtype.as_str() {
            "F32" if bytes.len() == 4 * count => Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
            "BF16" if bytes.len() == 2 * count => Ok(bytes.as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect()),
            _ => Err(format!("{name}: type {dtype} with {} bytes", bytes.len())),
        }
    }
}
fn read_f32(path: &Path, count: usize) -> R<Vec<f32>> {
    let bytes = io(std::fs::read(path), "read trace file")?;
    if bytes.len() != 4 * count { return Err(format!("{}: {} bytes, expected {}", path.display(), bytes.len(), 4 * count)); }
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

/// Running statistics of |mine - trace| over BF16 values.
#[derive(Default, Clone, Copy)]
struct Stat { n: u64, equal: u64, near: u64, sum: f64, max: f32, reference: f64 }
impl Stat {
    fn add(&mut self, mine: f32, trace: f32) {
        let d = (mine - trace).abs();
        // The distance to the two BF16 neighbors of the trace value.
        let bits = trace.to_bits();
        let step = (f32::from_bits(bits.wrapping_add(0x1_0000)) - trace).abs().max((trace - f32::from_bits(bits.wrapping_sub(0x1_0000))).abs());
        self.n += 1;
        if d == 0.0 { self.equal += 1; }
        if d <= step { self.near += 1; }
        self.sum += f64::from(d);
        self.max = self.max.max(d);
        self.reference += f64::from(trace.abs());
    }
    fn merge(&mut self, o: &Stat) {
        self.n += o.n; self.equal += o.equal; self.near += o.near; self.sum += o.sum; self.max = self.max.max(o.max); self.reference += o.reference;
    }
    /// Mean difference as a share of the mean size of the trace values.
    fn share(&self) -> f64 { if self.reference > 0.0 { self.sum / self.reference } else { 0.0 } }
    fn line(&self) -> String {
        let percent = |a: u64| 100.0 * a as f64 / self.n.max(1) as f64;
        format!("values {:>8}  equal {:>6.2}%  within one BF16 step {:>6.2}%  mean diff / mean size {:.5}  largest diff {:.5}",
            self.n, percent(self.equal), percent(self.near), self.share(), self.max)
    }
}

struct Model {
    config: Config,
    schedule: Schedule,
    kernel: usize,
    output_query: Vec<f32>,
    message_queries: Vec<f32>,
    expert_norm: Vec<Vec<f32>>,
    expert_conv: Vec<Vec<f32>>,
    attention_queries: Vec<Option<Vec<f32>>>,
    expert_queries: Vec<Vec<f32>>,
    gates: Vec<Option<Vec<[Vec<f32>; 3]>>>,
}
impl Model {
    fn load(folder: &Path) -> R<Self> {
        let config_text = io(std::fs::read(folder.join("config.json")), "read config.json")?;
        let config = e(Config::from_json(&config_text))?;
        let schedule = e(Schedule::new(config))?;
        let root = e(json::parse(&config_text, json::Limits::default()))?;
        let text = e(e(field(e(root.object())?, "text_config"))?.object())?;
        let kernel = e(e(field(text, "sconv_kernel_size"))?.u64())? as usize;
        let (h, loops, width) = (config.hidden_size, config.loop_layers(), config.gate_width);
        let mut store = Store::open(folder)?;
        let p = "model.llm.";
        let stored_qe = store.get(&format!("{p}middle_cycle_cross_pass_queries"), &[2 * loops, h])?;
        let stored_kv = store.get(&format!("{p}middle_cycle_cross_pass_kv_queries"), &[2 * loops, h])?;
        let mut model = Self { config, schedule, kernel, output_query: store.get(&format!("{p}attn_res.output_query"), &[h])?,
            message_queries: e(visit_loop::merge_message_queries(h, &stored_qe, &stored_kv))?, expert_norm: Vec::new(),
            expert_conv: Vec::new(), attention_queries: Vec::new(), expert_queries: Vec::new(), gates: Vec::new() };
        for layer in 0..config.layers {
            model.expert_norm.push(store.get(&format!("{p}layers.{layer}.mlp_norm.weight"), &[h])?);
            model.expert_conv.push(store.get(&format!("{p}layers.{layer}.mlp_sconv.weight"), &[h, 1, kernel])?);
            model.expert_queries.push(store.get(&format!("{p}attn_res.mlp_queries.{layer}"), &[h])?);
            // The stored attention query number is the layer number minus one. Layer 0 has none.
            model.attention_queries.push(if layer == 0 { None } else {
                let name = format!("{p}attn_res.attn_queries.{}", layer - 1);
                let shape = store.shape(&name)?;
                Some(store.get(&name, &shape)?)
            });
            model.gates.push(if (config.loop_start..config.loop_end).contains(&layer) {
                let mut set = Vec::new();
                // Stored gate names in the order query, key, value, expert.
                for kind in ["attn", "k", "v", "mlp"] {
                    let c = format!("{p}layers.{layer}.controllers.{kind}.");
                    set.push([store.get(&format!("{c}down_weight"), &[width, 2 * h])?, store.get(&format!("{c}forget_weight"), &[h, width])?,
                        store.get(&format!("{c}gain_weight"), &[h, width])?]);
                }
                Some(set)
            } else { None });
        }
        Ok(model)
    }
    /// Bind the stored tensors through the module's own checks: shapes and BF16 down weights.
    fn layers(&self) -> R<Vec<Layer<'_>>> {
        (0..self.config.layers).map(|layer| {
            let gates = self.gates[layer].as_ref().map(|set| [0, 1, 2, 3].map(|i| GateWeights { down: &set[i][0], forget: &set[i][1], gain: &set[i][2] }));
            e(Layer::new(&self.config, layer, LayerWeights { attention_queries: self.attention_queries[layer].as_deref(), expert_query: &self.expert_queries[layer], gates }))
        }).collect()
    }
}

/// Which reading of the model the expert input check uses. `Right` is the reading of the module.
#[derive(Clone, Copy, PartialEq)]
enum Reading { Right, NoGate, MessagesOfAnEarlierPass, TapsTheOtherWay, NoBlockEdge, NoConvolution }
const CONTROLS: [(&str, Reading); 5] = [("no gate in the later passes", Reading::NoGate),
    ("pass 3 reads the block outputs of pass 1", Reading::MessagesOfAnEarlierPass), ("convolution taps the other way", Reading::TapsTheOtherWay),
    ("boundary input not put on the block list", Reading::NoBlockEdge), ("stream after the visit used as the residual", Reading::NoConvolution)];

/// Fixed-tap short convolution over the token axis, then `(sum + input) / sqrt(kernel + 1)`,
/// rounded to BF16. `weights` is `[channel][tap]`. With `current_last`, the last tap is the
/// current token and tap 0 is the oldest one.
fn short_conv(weights: &[f32], inputs: &[&[f32]], kernel: usize, current_last: bool) -> Vec<f32> {
    let t = inputs.len() - 1;
    let scale = ((kernel + 1) as f64).sqrt() as f32;
    (0..inputs[t].len()).map(|c| {
        let mut sum = 0.0f32;
        for tap in 0..kernel {
            let back = if current_last { kernel - 1 - tap } else { tap };
            if back <= t { sum += weights[c * kernel + tap] * inputs[t - back][c]; }
        }
        bf16((sum + inputs[t][c]) / scale)
    }).collect()
}
/// Weighted norm in f32, rounded to BF16.
fn weighted_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut squares = 0.0f32;
    for &v in x { squares += v * v; }
    let inverse = 1.0 / (squares / x.len() as f32 + eps).sqrt();
    x.iter().zip(weight).map(|(&v, &w)| bf16(w * (v * inverse))).collect()
}

#[derive(Default)]
struct Totals { files: usize, end: f32, right: Stat, later_passes: Stat, controls: BTreeMap<&'static str, Stat> }

fn check_file(model: &Model, layers: &[Layer<'_>], trace_dir: &Path, file: &Value, totals: &mut Totals) -> R<()> {
    let (config, schedule, kernel) = (&model.config, &model.schedule, model.kernel);
    let (h, n) = (config.hidden_size, schedule.visits().len());
    let o = e(file.object())?;
    let label = format!("{} / {}", e(e(field(o, "effort"))?.string())?, e(e(field(o, "way"))?.string())?);
    let names = e(e(field(o, "files"))?.object())?;
    let entry = |key: &str| -> R<(String, Vec<usize>)> {
        let f = e(e(field(names, key))?.object())?;
        Ok((e(e(field(f, "file"))?.string())?.to_string(),
            e(e(field(f, "shape"))?.array())?.iter().map(|d| e(d.u64()).map(|d| d as usize)).collect::<R<Vec<_>>>()?))
    };
    let (states_name, states_shape) = entry("states")?;
    let (io_name, io_shape) = entry("expert-io")?;
    let tokens = *states_shape.get(1).ok_or("the states shape has no token axis")?;
    if states_shape != [n + 3, tokens, h] || io_shape != [n, 2, tokens, h] { return Err(format!("{label}: trace shapes {states_shape:?} {io_shape:?} do not fit the visit list")); }
    let states = read_f32(&trace_dir.join(&states_name), (n + 3) * tokens * h)?;
    let expert_io = read_f32(&trace_dir.join(&io_name), n * 2 * tokens * h)?;
    let state = |s: usize, t: usize| &states[(s * tokens + t) * h..(s * tokens + t + 1) * h];
    let expert = |k: usize, side: usize, t: usize| &expert_io[((k * 2 + side) * tokens + t) * h..((k * 2 + side) * tokens + t + 1) * h];
    let source = |s: Source, t: usize| match s { Source::Embedding => state(0, t), Source::Visit(k) => state(k + 1, t) };
    println!("\n== {label}: {tokens} tokens, files {states_name} and {io_name} ==");

    // A. The visit list against the trace's own record of the layer calls.
    let traced = e(e(field(o, "visits"))?.array())?;
    let mut same = 0;
    for (visit, t) in schedule.visits().iter().zip(traced) {
        let t = e(t.object())?;
        if e(e(field(t, "layer_module"))?.u64())? as usize == visit.layer && e(e(field(t, "pass_of_that_module"))?.u64())? as usize == visit.pass { same += 1; }
    }
    println!("A visit list: {same} of {n} visits have the layer and the pass of the trace, in run order (the trace lists {})", traced.len());
    assert_eq!((same, traced.len()), (n, n), "{label}: the visit list differs from the trace");

    // B. Replay through the driver, at the last layer of each block.
    let marks: Vec<usize> = (0..n).filter(|&k| k + 1 == n || schedule.visits()[k + 1].boundary).collect();
    let (mut replay, mut end, mut end_size) = (0.0f32, 0.0f32, 0.0f32);
    for t in 0..tokens {
        let mut k = 0;
        let streams = e(schedule.run(state(0, t), |_, _| { k += 1; Ok(state(k, t).to_vec()) }))?;
        for &mark in &marks {
            for (a, b) in streams[mark].iter().zip(state(mark + 1, t)) { replay = replay.max((a - b).abs()); }
        }
        // C. The end mix against the input of the last norm.
        let mixed = e(schedule.end_mix(&model.output_query, state(0, t), &streams))?;
        for (a, b) in mixed.iter().zip(state(n + 1, t)) { end = end.max((a - b).abs()); end_size = end_size.max(b.abs()); }
    }
    println!("B replay: largest difference of the stream after visits {marks:?} to the trace: {replay:e} (zero by construction when the indexing is right)");
    println!("C end mix: largest difference to the trace's state {} over {tokens} tokens: {end:.3e} (largest value there {end_size:.3})", n + 1);
    assert_eq!(replay, 0.0, "{label}: the driver does not hand back the trace's streams");
    assert!(end <= 5e-5, "{label}: the end mix differs from the trace by {end}");
    totals.end = totals.end.max(end);

    // D. The expert input of every visit.
    let expert_input = |k: usize, t: usize, reading: Reading| -> R<Vec<f32>> {
        let visit = &schedule.visits()[k];
        let stream = source(visit.stream, t);
        let blocks: Vec<&[f32]> = visit.blocks.iter().map(|&s| source(s, t)).collect();
        let history: Vec<&[f32]> = (0..=t).map(|u| expert(k, 1, u)).collect();
        let f = short_conv(&model.expert_conv[visit.layer], &history, kernel, reading != Reading::TapsTheOtherWay);
        let residual: Vec<f32> = if reading == Reading::NoConvolution { state(k + 1, t).to_vec() }
            else { state(k + 1, t).iter().zip(&f).map(|(&out, &f)| out - f).collect() };
        let boundary = visit.boundary && reading != Reading::NoBlockEdge;
        // `expert_input` makes the residual again: from this value alone at a boundary, or with the stream added.
        let attended: Vec<f32> = if boundary { residual.clone() } else { residual.iter().zip(stream).map(|(&r, &s)| r - s).collect() };
        let read = match (&visit.messages, reading) {
            (None, _) | (_, Reading::NoGate) => None,
            // The messages that the same layer got one pass earlier, when it got any.
            (Some(m), Reading::MessagesOfAnEarlierPass) => Some(schedule.visits().iter().find(|v| v.layer == visit.layer && v.pass + 1 == visit.pass).and_then(|v| v.messages.as_ref()).unwrap_or(m)),
            (Some(m), _) => Some(m),
        };
        let messages = match read {
            Some(m) => Some(e(visit_loop::messages(config, &model.message_queries, m.loop_layer, &m.from.iter().map(|&s| source(s, t)).collect::<Vec<_>>()))?),
            None => None,
        };
        let (_, input) = e(layers[visit.layer].expert_input(boundary, &blocks, stream, &attended, messages.as_deref()))?;
        Ok(weighted_norm(&input, &model.expert_norm[visit.layer], config.rms_norm_eps))
    };
    let measure = |k: usize, reading: Reading| -> R<Stat> {
        let mut stat = Stat::default();
        for t in 0..tokens {
            let mine = expert_input(k, t, reading)?;
            for (&a, &b) in mine.iter().zip(expert(k, 0, t)) { stat.add(a, b); }
        }
        Ok(stat)
    };
    let mut right = Vec::with_capacity(n);
    let (mut total, mut first, mut later) = (Stat::default(), Stat::default(), Stat::default());
    for k in 0..n {
        let s = measure(k, Reading::Right)?;
        total.merge(&s);
        if schedule.visits()[k].messages.is_some() { later.merge(&s); } else { first.merge(&s); }
        right.push(s);
    }
    println!("D expert input, the module's reading, all {n} visits: {}", total.line());
    println!("D   visits with no messages: {}", first.line());
    println!("D   visits with messages and gates: {}", later.line());
    assert!(total.share() <= 0.005, "{label}: the expert inputs differ from the trace by {}", total.share());
    assert!(later.n == 0 || later.share() <= 0.005, "{label}: the gated expert inputs differ from the trace by {}", later.share());
    totals.right.merge(&total);
    totals.later_passes.merge(&later);
    // Controls: one deliberate error each, on the visits that the error touches.
    for (name, reading) in CONTROLS {
        let touched: Vec<usize> = (0..n).filter(|&k| { let v = &schedule.visits()[k]; match reading {
            Reading::NoGate => v.messages.is_some(),
            Reading::MessagesOfAnEarlierPass => v.pass >= 3,
            Reading::NoBlockEdge => v.boundary,
            _ => true,
        } }).collect();
        let (mut wrong, mut good) = (Stat::default(), Stat::default());
        for &k in &touched { wrong.merge(&measure(k, reading)?); good.merge(&right[k]); }
        println!("D control '{name}' on {} visits: {}", touched.len(), wrong.line());
        assert!(touched.is_empty() || wrong.share() >= 20.0 * good.share().max(1e-4), "{label}: the control '{name}' does not move the result");
        totals.controls.entry(name).or_default().merge(&wrong);
    }
    println!("D per visit (visit, layer, pass: equal %, within one step %, mean diff / mean size, largest diff):");
    for (k, s) in right.iter().enumerate() {
        let v = &schedule.visits()[k];
        println!("D   {k:>2} {:>2} {} : {:>6.2} {:>6.2} {:.5} {:.5}", v.layer, v.pass, 100.0 * s.equal as f64 / s.n as f64, 100.0 * s.near as f64 / s.n as f64, s.share(), s.max);
    }
    totals.files += 1;
    Ok(())
}

#[test]
#[ignore = "requires the model folder and a reference layer trace under .local/visit-loop-trace/"]
fn block_mix_and_visit_loop_agree_with_the_reference_trace() {
    let (model_dir, trace_dir) = (local("model"), local("trace"));
    let model = Model::load(&model_dir).unwrap();
    let c = &model.config;
    println!("config: hidden {} layers {} block {} loop {}..{} passes {} gate width {} -> {} visits, at most {} vectors held",
        c.hidden_size, c.layers, c.block_layers, c.loop_start, c.loop_end, c.passes, c.gate_width, model.schedule.visits().len(), model.schedule.peak_held());
    let layers = model.layers().unwrap();
    println!("module checks: {} layers bound with the stored tensors", layers.len());
    let mut indexes: Vec<PathBuf> = std::fs::read_dir(&trace_dir).unwrap().filter_map(|entry| entry.ok()).map(|entry| entry.path())
        .filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("trace-index-") && name.ends_with(".json"))).collect();
    indexes.sort();
    assert!(!indexes.is_empty(), "no trace-index-*.json in .local/visit-loop-trace/trace/");
    let mut totals = Totals::default();
    for path in &indexes {
        println!("\n==== index {} ====", path.file_name().unwrap().to_string_lossy());
        let index = json::parse(&std::fs::read(path).unwrap(), json::Limits::default()).unwrap();
        for file in field(index.object().unwrap(), "files").unwrap().array().unwrap() { check_file(&model, &layers, &trace_dir, file, &mut totals).unwrap(); }
    }
    println!("\n== ALL {} TRACE FILE SETS ==", totals.files);
    println!("C end mix: largest difference to the trace: {:.3e}", totals.end);
    println!("D expert input, the module's reading: {}", totals.right.line());
    println!("D   visits with messages and gates: {}", totals.later_passes.line());
    for (name, wrong) in &totals.controls { println!("D control '{name}': {}", wrong.line()); }
    assert!(totals.files > 0);
}
