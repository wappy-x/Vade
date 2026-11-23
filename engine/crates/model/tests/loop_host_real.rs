//! The whole-model host form on the real stored weights, against files that
//! the reference code wrote. Ignored by default: the weights and the reference
//! files are private. They stay outside the tracked tree, in the ignored
//! folder `.local/loop-host/` at the repository root:
//!
//! - `model/`: the model folder (config, storage sidecar, shard index, weight file). A link is enough.
//! - `ref/`: the reference folder with `contract-A/ops/`, `contract-A/trace/` and `sets/`.
//! - `out/`: made by the tests. The score files of this host form.
//! - `items.txt`: optional. One answer folder name a line for the answers test, run in that order.
//! - `forced-only.txt`: optional. With it the answers test stops each free run at the token where it parts.
//! - `decode-items.txt`: optional. One answer folder name a line for the decode rate test.
//! - `long-items.txt`: optional. One answer folder name a line for the long reference test.
//! - `cache-gib.txt`: optional. The limit of the expert cache as a whole number of GiB. Without it: 24.
//! - `threads.txt`: optional. The thread count of a read as a whole number. Without it: every core.
//!
//! The tests read no environment variable. Each prints its numbers as lines
//! that start with a capital word (`LAYOUT`, `FORCED`, `FREE`, `TOKEN`,
//! `ITEM`, `SUMMARY`). A saved copy of the output is the receipt.
//!
//! Bit equality with the reference is not the gate. The reference leaves its
//! sum orders to a library, and its own two ways to compute one prompt differ
//! (see `ref/README.md`). The gate of each test is stated at its assert.
use engine_model::{capped_moe::{CappedActivation, Expert, Routing, fold}, expert_widen::{Stored, StoredMatrix, widen},
    json::{self, Value}, loop_host::{Folder, Head, Limits, Model, Record, Runner, Source, VisitOut, Want},
    safetensors::Dtype, short_conv::{FixedTaps, History, TokenTaps}};
use std::{collections::BTreeMap, fmt, fs::File, io::{Read, Seek, SeekFrom, Write}, path::{Path, PathBuf}, time::Instant};

/// Bytes of widened experts that the cache may hold: 24 GiB, or the number of GiB in `cache-gib.txt`.
/// A host resource limit. It changes no word (unit test `the_expert_cache_changes_no_word`).
fn cache_bytes() -> usize {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host/cache-gib.txt");
    let gib = std::fs::read_to_string(file).map_or(24, |text| text.trim().parse::<usize>().expect("cache-gib.txt holds one whole number"));
    gib << 30
}

fn local(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host").join(name);
    assert!(path.exists(), "put {name} in .local/loop-host/ for the ignored host tests");
    path
}
/// Threads of a read: every core, or the number in `threads.txt`. A host resource limit. It changes no word
/// (unit test `a_text_gives_the_same_words_at_every_thread_count_and_every_cut`).
fn threads() -> usize {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/loop-host/threads.txt");
    std::fs::read_to_string(file).map_or_else(|_| std::thread::available_parallelism().map_or(1, |n| n.get()), |text| text.trim().parse::<usize>().expect("threads.txt holds one whole number"))
}
fn open() -> Model {
    let start = Instant::now();
    let model = Model::open(&local("model"), Limits { expert_cache_bytes: cache_bytes() }).unwrap();
    println!("LOAD dense weights read and checked in {:.1} s; {} visits a token, width {}, {} head rows; {} threads; expert cache limit {} GiB", start.elapsed().as_secs_f64(),
        model.schedule().visits().len(), model.hidden(), model.head_rows(), threads(), cache_bytes() >> 30);
    model
}
fn floats(path: &Path, first: usize, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    let mut file = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    file.seek(SeekFrom::Start(first as u64 * 4)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}
fn ints(path: &Path, first: usize, count: usize) -> Vec<i32> { floats(path, first, count).iter().map(|v| v.to_bits() as i32).collect() }
fn ids(path: &Path) -> Vec<u32> { std::fs::read_to_string(path).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect() }
fn parse(path: &Path) -> Value { json::parse(&std::fs::read(path).unwrap(), json::Limits { nodes: 4_000_000, ..Default::default() }).unwrap() }
fn numbers(value: &Value) -> Vec<usize> { value.array().unwrap().iter().map(|v| v.u64().unwrap() as usize).collect() }
/// A whole BF16 or FP32 tensor of the weight files, as FP32.
fn stored_floats(source: &dyn Source, name: &str) -> Vec<f32> {
    let (dtype, shape) = source.tensor(name).unwrap();
    let values = shape.iter().product::<u64>() as usize;
    let mut bytes = vec![0u8; values * if dtype == Dtype::Bf16 { 2 } else { 4 }];
    source.read(name, 0, &mut bytes).unwrap();
    match dtype {
        Dtype::Bf16 => bytes.as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect(),
        Dtype::F32 => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        other => panic!("{name} has type {other:?}"),
    }
}

/// How near one set of values is to the reference's.
#[derive(Debug, Clone, Copy, Default)]
struct Near { equal: usize, total: usize, worst: f32, difference: f64, size: f64 }
impl Near {
    fn of(mine: &[f32], theirs: &[f32]) -> Self { let mut near = Self::default(); near.add(mine, theirs); near }
    fn add(&mut self, mine: &[f32], theirs: &[f32]) {
        assert_eq!(mine.len(), theirs.len(), "compared vectors differ in length");
        for (a, b) in mine.iter().zip(theirs) {
            self.equal += usize::from(a.to_bits() == b.to_bits() || a == b);
            self.worst = self.worst.max((a - b).abs());
            self.difference += ((a - b) as f64).powi(2); self.size += (*b as f64).powi(2);
        }
        self.total += mine.len();
    }
    fn join(&mut self, other: Near) {
        self.equal += other.equal; self.total += other.total; self.worst = self.worst.max(other.worst);
        self.difference += other.difference; self.size += other.size;
    }
    /// Length of the difference over the length of the reference: the measure of the reference's own noise file.
    fn relative(&self) -> f64 { if self.size == 0.0 { self.difference.sqrt() } else { (self.difference / self.size).sqrt() } }
    fn share(&self) -> f64 { self.equal as f64 / self.total.max(1) as f64 }
}
impl fmt::Display for Near {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "equal {}/{} ({:.2} percent), relative {:.3e}, largest difference {:.3e}", self.equal, self.total, 100.0 * self.share(), self.relative(), self.worst)
    }
}

/// The module dumps of the reference: every call inside 7 layer visits on a prompt of 8 tokens.
struct Dumps { root: PathBuf, calls: Vec<(String, String, BTreeMap<String, (PathBuf, Vec<usize>)>)>, visits: Vec<(String, usize, usize)>, ids: Vec<u32> }
impl Dumps {
    fn open(root: PathBuf) -> Self {
        let index = parse(&root.join("ops-index.json"));
        let object = index.object().unwrap();
        let calls = object["calls"].array().unwrap().iter().map(|call| {
            let call = call.object().unwrap();
            let files = call["inputs"].array().unwrap().iter().chain(call["outputs"].array().unwrap()).map(|entry| {
                let entry = entry.object().unwrap();
                (entry["slot"].string().unwrap().to_owned(), (root.join(entry["file"].string().unwrap()), numbers(&entry["shape"])))
            }).collect();
            (call["folder"].string().unwrap().to_owned(), call["module"].string().unwrap().to_owned(), files)
        }).collect();
        let visits = object["visits"].array().unwrap().iter().map(|v| { let v = v.object().unwrap();
            (v["folder"].string().unwrap().to_owned(), v["visit_0_based"].u64().unwrap() as usize, v["layer_module"].u64().unwrap() as usize) }).collect();
        Self { root, calls, visits, ids: numbers(&object["input_ids"]).into_iter().map(|v| v as u32).collect() }
    }
    /// One saved tensor of call `nth` of `module` in a visit folder, with its shape. `None` when the call has no such slot.
    fn get(&self, folder: &str, module: &str, nth: usize, slot: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let (_, _, files) = self.calls.iter().filter(|(f, m, _)| f == folder && m == module).nth(nth)?;
        let (path, shape) = files.get(slot)?;
        Some((shape.clone(), floats(path, 0, shape.iter().product())))
    }
    fn need(&self, folder: &str, module: &str, slot: &str) -> Vec<f32> {
        self.get(folder, module, 0, slot).unwrap_or_else(|| panic!("no dump {folder} {module} {slot}")).1
    }
}

/// States of a reference run: `[51][tokens][hidden]`. State 0 is the normed embedding, state `v + 1` the stream after visit `v`.
struct States { values: Vec<f32>, tokens: usize, hidden: usize }
impl States {
    fn read(path: &Path, tokens: usize, hidden: usize, visits: usize) -> Self { Self { values: floats(path, 0, (visits + 3) * tokens * hidden), tokens, hidden } }
    fn at(&self, state: usize, token: usize) -> &[f32] { &self.values[(state * self.tokens + token) * self.hidden..][..self.hidden] }
    /// The streams after every visit of one token, as the reference had them.
    fn streams(&self, token: usize, visits: usize) -> Vec<Vec<f32>> { (0..visits).map(|v| self.at(v + 1, token).to_vec()).collect() }
}
/// Run every visit with the reference's own inputs: the stream, the block list and the message inputs come
/// from the reference states, and only the state of the same visit for the earlier tokens is this form's own.
/// So an error of one visit does not reach the next visit. Returns `[visit][token]`.
fn forced(runner: &Runner<'_>, states: &States) -> Vec<Vec<VisitOut>> {
    let visits = runner.model().schedule().visits().len();
    let fresh = runner.chat().unwrap();
    let streams: Vec<Vec<Vec<f32>>> = (0..states.tokens).map(|t| states.streams(t, visits)).collect();
    let mut outs: Vec<Vec<VisitOut>> = (0..visits).map(|_| Vec::new()).collect();
    let each = visits.div_ceil(threads().min(visits));
    std::thread::scope(|scope| {
        for (part, outs) in outs.chunks_mut(each).enumerate() {
            let (fresh, streams) = (&fresh, &streams);
            scope.spawn(move || for (i, out) in outs.iter_mut().enumerate() {
                let visit = part * each + i;
                let mut state = fresh.visits()[visit].clone();
                for (token, before) in streams.iter().enumerate().take(states.tokens) {
                    let inputs = runner.inputs(visit, states.at(0, token), before).unwrap();
                    out.push(runner.visit(visit, &mut state, &inputs, 1, true).unwrap());
                }
            });
        }
    });
    outs
}
fn same_set(mine: &[usize], theirs: &[i32]) -> bool {
    let (mut a, mut b): (Vec<usize>, Vec<usize>) = (mine.to_vec(), theirs.iter().map(|&v| v as usize).collect());
    a.sort_unstable(); b.sort_unstable();
    a == b
}

/// The stored bytes of one expert, for the controls of the layout facts.
struct StoredExpert { hidden: usize, width: usize, codes13: Vec<u8>, scales13: Vec<u8>, codes2: Vec<u8>, scales2: Vec<u8>, second13: Option<(f32, f32)>, second2: Option<(f32, f32)> }
impl StoredExpert {
    fn read(source: &dyn Source, layer: usize, expert: usize) -> Self {
        let part = |base: &str| {
            let (dtype, shape) = source.tensor(base).unwrap();
            let (_, scale) = source.tensor(&format!("{base}_scale")).unwrap();
            let (each, each_scale) = ((shape[1] * shape[2]) as usize, (scale[1] * scale[2]) as usize);
            let (mut codes, mut scales) = (vec![0u8; each], vec![0u8; each_scale]);
            source.read(base, (expert * each) as u64, &mut codes).unwrap();
            source.read(&format!("{base}_scale"), (expert * each_scale) as u64, &mut scales).unwrap();
            let second = (dtype == Dtype::U8).then(|| (stored_floats(source, &format!("{base}_amax"))[expert], stored_floats(source, &format!("{base}_outer_scale_denominator"))[0]));
            (codes, scales, second, shape[1] as usize, shape[2] as usize * if dtype == Dtype::U8 { 2 } else { 1 })
        };
        let (codes13, scales13, second13, _, hidden) = part(&format!("model.llm.layers.{layer}.mlp.experts.w13_weight"));
        let (codes2, scales2, second2, _, width) = part(&format!("model.llm.layers.{layer}.mlp.experts.w2_weight"));
        Self { hidden, width, codes13, scales13, codes2, scales2, second13, second2 }
    }
    fn matrix<'a>(rows: usize, columns: usize, codes: &'a [u8], scales: &'a [u8], second: Option<(f32, f32)>) -> StoredMatrix<'a> {
        let data = match second { Some((amax, denominator)) => Stored::Packed4 { codes, scales, amax, denominator }, None => Stored::Float8 { values: codes, scales } };
        StoredMatrix { rows, columns, data }
    }
    fn bits(&self) -> (Vec<u16>, Vec<u16>) {
        (widen(&Self::matrix(2 * self.width, self.hidden, &self.codes13, &self.scales13, self.second13)).unwrap(),
            widen(&Self::matrix(self.hidden, self.width, &self.codes2, &self.scales2, self.second2)).unwrap())
    }
    /// Control: the two codes of each byte the other way round.
    fn nibbles_swapped(&self) -> Expert {
        let swap = |codes: &[u8]| -> Vec<u8> { codes.iter().map(|b| b.rotate_left(4)).collect() };
        let swapped = Self { codes13: swap(&self.codes13), codes2: swap(&self.codes2), scales13: self.scales13.clone(), scales2: self.scales2.clone(), ..*self };
        let (gate_up, down) = swapped.bits();
        Expert::from_bf16(self.hidden, self.width, &gate_up, &down).unwrap()
    }
    /// Control: the first half of the stored rows read as gate rows and the second half as up rows.
    fn rows_as_halves(&self) -> Expert {
        let (stored, down) = self.bits();
        let mut gate_up = vec![0u16; stored.len()];
        for i in 0..self.width {
            gate_up[2 * i * self.hidden..][..self.hidden].copy_from_slice(&stored[i * self.hidden..][..self.hidden]);
            gate_up[(2 * i + 1) * self.hidden..][..self.hidden].copy_from_slice(&stored[(self.width + i) * self.hidden..][..self.hidden]);
        }
        Expert::from_bf16(self.hidden, self.width, &gate_up, &down).unwrap()
    }
}

/// Step 1. Single modules and single visits on the prompt of 8 tokens, against the module dumps.
/// It settles four layout facts that were read in the reference source and never run as one piece
/// with the real weights: the order of the fixed taps, the index of the taps made from the token,
/// the gate and up row order, and the order of the two 4-bit codes in a byte.
#[test]
#[ignore = "requires the real weight file and the reference module dumps outside the repository"]
fn single_visits_match_the_module_dumps() {
    let (model, dumps) = (open(), Dumps::open(local("ref").join("contract-A/ops")));
    let runner = model.runner().unwrap();
    let source = Folder::open(&local("model")).unwrap();
    let (tokens, hidden, visits) = (dumps.ids.len(), model.hidden(), model.schedule().visits().len());
    let text = parse(&local("model").join("config.json"));
    let text = text.object().unwrap()["text_config"].object().unwrap();
    let (kv_heads, head_dim, taps) = (text["num_key_value_heads"].u64().unwrap() as usize, text["head_dim"].u64().unwrap() as usize, text["sconv_kernel_size"].u64().unwrap() as usize);
    let kv = kv_heads * head_dim;

    // Fact 1: fixed taps. The last stored tap weights the current token.
    let convolve = |weights: &[f32], channels: usize, input: &[f32]| -> Vec<f32> {
        let (conv, mut history) = (FixedTaps::new(channels, taps, weights).unwrap(), History::new(channels, taps).unwrap());
        input.chunks_exact(channels).flat_map(|token| conv.step(&mut history, token).unwrap()).collect()
    };
    let reversed = |weights: &[f32]| -> Vec<f32> { weights.chunks_exact(taps).flat_map(|row| row.iter().rev().copied()).collect() };
    let mut fact1 = (Near::default(), Near::default());
    for (folder, layer) in [("L01-pass1", 1), ("L20-pass1", 20)] {
        for (module, tensor, channels) in [("post_attention_convolution", "attn_sconv.weight", hidden), ("post_mlp_convolution", "mlp_sconv.weight", hidden), ("self_attn.k_proj.sconv", "attn.k_sconv.weight", kv), ("self_attn.v_proj.sconv", "attn.v_sconv.weight", kv)] {
            let weights = stored_floats(&source, &format!("model.llm.layers.{layer}.{tensor}"));
            let (input, want) = (dumps.need(folder, module, "in0"), dumps.need(folder, module, "out0"));
            let (built, control) = (Near::of(&convolve(&weights, channels, &input), &want), Near::of(&convolve(&reversed(&weights), channels, &input), &want));
            println!("LAYOUT fixed taps {folder} {module}: as built {built}; control with the taps reversed {control}");
            fact1.0.join(built); fact1.1.join(control);
        }
    }
    println!("LAYOUT fact 1 (fixed taps, oldest first): as built {}; control {}", fact1.0, fact1.1);

    // Fact 2: taps made from the token. Stored row 0 weights the current token, and the projected index is head * taps + tap.
    let mut fact2 = (Near::default(), Near::default(), Near::default());
    for (folder, layer) in [("L05-pass1", 5), ("L11-pass1", 11)] {
        for stream in ["k", "v"] {
            let module = format!("self_attn.{stream}_proj.sconv");
            let static_taps = stored_floats(&source, &format!("model.llm.layers.{layer}.attn.{stream}_sconv.static_w"));
            let projection = stored_floats(&source, &format!("model.llm.layers.{layer}.attn.{stream}_sconv.weight_proj"));
            let (input, from, want) = (dumps.need(folder, &module, "in0"), dumps.need(folder, &module, "in1"), dumps.need(folder, &module, "out0"));
            let run = |static_taps: &[f32], projection: &[f32]| -> Vec<f32> {
                let (conv, mut history) = (TokenTaps::new(kv, kv_heads, taps, hidden, static_taps, projection).unwrap(), History::new(kv, taps).unwrap());
                input.chunks_exact(kv).zip(from.chunks_exact(hidden)).flat_map(|(token, source)| conv.step(&mut history, token, source).unwrap()).collect()
            };
            let rows = |order: &dyn Fn(usize, usize) -> usize| -> Vec<f32> {
                (0..kv_heads * taps).flat_map(|row| projection[order(row / taps, row % taps) * hidden..][..hidden].to_vec()).collect()
            };
            // Control A: the lags the other way round, in the static taps and in each head's projected taps.
            let static_back: Vec<f32> = (0..taps).flat_map(|tap| static_taps[(taps - 1 - tap) * kv..][..kv].to_vec()).collect();
            let lags_back = Near::of(&run(&static_back, &rows(&|head, tap| head * taps + (taps - 1 - tap))), &want);
            // Control B: the projected index read as tap * heads + head.
            let index_other = Near::of(&run(&static_taps, &rows(&|head, tap| tap * kv_heads + head)), &want);
            let built = Near::of(&run(&static_taps, &projection), &want);
            println!("LAYOUT token taps {folder} {module}: as built {built}; control lags reversed {lags_back}; control index tap * heads + head {index_other}");
            fact2.0.join(built); fact2.1.join(lags_back); fact2.2.join(index_other);
        }
    }
    println!("LAYOUT fact 2 (tap index head * taps + tap, row 0 is the current token): as built {}; control lags reversed {}; control other index {}", fact2.0, fact2.1, fact2.2);

    // Facts 3 and 4: the gate and up rows, and the two codes of a byte. The expert block of the reference on its own
    // input, with this form's pick order. Layer 1 holds packed 4-bit experts and layer 20 holds 8-bit float experts.
    let routing = Routing::new(text["n_routed_experts"].u64().unwrap() as usize, text["num_experts_per_tok"].u64().unwrap() as usize, text["route_scale"].f64().unwrap()).unwrap();
    let activation = CappedActivation::new(text["mlp_gate_softcap"].f64().unwrap(), text["mlp_up_softcap"].f64().unwrap()).unwrap();
    let (mut fact3, mut fact4) = ((Near::default(), Near::default()), (Near::default(), Near::default()));
    for (folder, layer) in [("L01-pass1", 1usize), ("L20-pass1", 20)] {
        let bias = stored_floats(&source, &format!("model.llm.layers.{layer}.mlp.gate.bias"));
        let (z, logits, want) = (dumps.need(folder, "mlp.experts", "in0"), dumps.need(folder, "mlp.gate.score_fn", "in0"), dumps.need(folder, "mlp.experts", "out0"));
        let picks: Vec<i32> = dumps.need(folder, "mlp.experts", "in1").iter().map(|v| v.to_bits() as i32).collect();
        let packed = source.tensor(&format!("model.llm.layers.{layer}.mlp.experts.w13_weight")).unwrap().0 == Dtype::U8;
        let (mut built, mut halves, mut swapped, mut sets) = (Near::default(), Near::default(), Near::default(), 0);
        for token in 0..tokens {
            let (z, want) = (&z[token * hidden..][..hidden], &want[token * hidden..][..hidden]);
            let route = routing.route(&logits[token * routing.experts()..][..routing.experts()], &bias).unwrap();
            sets += usize::from(same_set(&route.ids, &picks[token * routing.selected()..][..routing.selected()]));
            let stored: Vec<StoredExpert> = route.ids.iter().map(|&id| StoredExpert::read(&source, layer, id)).collect();
            let block = |experts: Vec<Vec<f32>>| fold(&route.weights, &experts).unwrap();
            built.add(&block(route.ids.iter().map(|&id| model.expert(layer, id).unwrap().run(z, activation).unwrap()).collect()), want);
            halves.add(&block(stored.iter().map(|e| e.rows_as_halves().run(z, activation).unwrap()).collect()), want);
            if packed { swapped.add(&block(stored.iter().map(|e| e.nibbles_swapped().run(z, activation).unwrap()).collect()), want); }
        }
        println!("LAYOUT experts {folder} ({}): picked sets equal {sets}/{tokens}; as built {built}; control rows as two halves {halves}{}",
            if packed { "packed 4-bit" } else { "8-bit float" }, if packed { format!("; control codes of a byte swapped {swapped}") } else { String::new() });
        assert_eq!(sets, tokens, "a picked set differs from the reference on its own router scores");
        fact3.0.join(built); fact3.1.join(halves);
        if packed { fact4.0.join(built); fact4.1.join(swapped); }
    }
    println!("LAYOUT fact 3 (stored row 2i is gate row i, row 2i + 1 is up row i): as built {}; control {}", fact3.0, fact3.1);
    println!("LAYOUT fact 4 (the low 4 bits of a byte are the even column): as built {}; control {}", fact4.0, fact4.1);

    // Whole visits with the reference's inputs, stage by stage where a dump exists.
    let states = States::read(&dumps.root.join("states.f32"), tokens, hidden, visits);
    let outs = forced(&runner, &states);
    for (folder, visit, layer) in &dumps.visits {
        let records: Vec<&Record> = outs[*visit].iter().map(|out| out.record.as_ref().unwrap()).collect();
        let stage = |module: &str, slot: &str, mine: &dyn Fn(&Record) -> Vec<f32>| -> String {
            match dumps.get(folder, module, 0, slot) {
                Some((_, want)) => format!("{}", Near::of(&records.iter().flat_map(|r| mine(r)).collect::<Vec<_>>(), &want)),
                None => "no dump".into(),
            }
        };
        println!("STAGE visit {visit} layer {layer} {folder}: query input: {}", stage("self_attn", "kw-hidden_states", &|r| r.attention_inputs[0].clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: key input: {}", stage("self_attn", "kw-key_hidden_states", &|r| r.attention_inputs[1].clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: value input: {}", stage("self_attn", "kw-value_hidden_states", &|r| r.attention_inputs[2].clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: attention output: {}", stage("self_attn", "out0", &|r| r.attention.clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: after its convolution: {}", stage("post_attention_convolution", "out0", &|r| r.attended.clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: expert input: {}", stage("mlp", "in0", &|r| r.expert_input.clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: router scores: {}", stage("mlp.gate.score_fn", "in0", &|r| r.router_logits.clone()));
        println!("STAGE visit {visit} layer {layer} {folder}: expert output: {}", stage("mlp", "out0", &|r| r.experts.clone()));
        let picks: Vec<i32> = dumps.need(folder, "mlp.experts", "in1").iter().map(|v| v.to_bits() as i32).collect();
        let selected = picks.len() / tokens;
        let sets = records.iter().enumerate().filter(|(t, r)| same_set(&r.route.ids, &picks[t * selected..][..selected])).count();
        let stream = Near::of(&outs[*visit].iter().flat_map(|out| out.stream.clone()).collect::<Vec<_>>(), &dumps.need(folder, "LAYER", "out0"));
        println!("STAGE visit {visit} layer {layer} {folder}: picked sets equal {sets}/{tokens}; stream after the visit: {stream}");
    }
    let mut all = Near::default();
    for (visit, outs) in outs.iter().enumerate() {
        let mut near = Near::default();
        for (token, out) in outs.iter().enumerate() { near.add(&out.stream, states.at(visit + 1, token)); }
        println!("FORCED dumps visit {visit} layer {}: stream {near}", model.schedule().visits()[visit].layer);
        all.join(near);
    }
    println!("SUMMARY dumps, {visits} visits with the reference's inputs, {tokens} tokens: stream {all}");

    // The gates. Each layout fact as built is far nearer to the reference than its control.
    for (name, built, control) in [("fixed tap order", fact1.0, fact1.1), ("token tap lag order", fact2.0, fact2.1), ("token tap index", fact2.0, fact2.2),
        ("gate and up rows", fact3.0, fact3.1), ("codes of a byte", fact4.0, fact4.1)] {
        assert!(built.share() > 0.9 && control.share() < 0.5 && built.relative() * 10.0 < control.relative(), "layout fact {name}: as built {built}; control {control}");
    }
    assert!(all.relative() < 0.01, "a visit with the reference's inputs is more than 1 percent from the reference");
}

/// What one reference run saved.
struct Run { name: String, ids: Vec<u32>, states: States, picks: Option<Vec<i32>>, expert_io: Option<PathBuf>, logits: PathBuf }
/// The top logit of each position, with the token that holds it. The lower token wins a tie.
fn top(logits: &[f32]) -> (u32, f32) {
    let mut best = 0usize;
    for (token, &value) in logits.iter().enumerate() { if value > logits[best] { best = token; } }
    (best as u32, logits[best])
}
/// Levels C of the plan on one reference run: every visit with the reference's inputs, then the whole
/// run from token IDs alone. Returns (forced relative difference of all states, free picked sets equal, pairs, top tokens equal).
fn compare(runner: &Runner<'_>, run: &Run) -> (f64, usize, usize, usize) {
    let model = runner.model();
    let (visits, tokens, hidden, rows, last) = (model.schedule().visits().len(), run.ids.len(), model.hidden(), model.head_rows(), run.ids.len() - 1);
    let selected = run.picks.as_ref().map_or(0, |p| p.len() / (visits * tokens));
    let picks = |visit: usize, token: usize| run.picks.as_ref().map(|p| &p[(visit * tokens + token) * selected..][..selected]);
    let sets_text = |sets: usize, pairs: usize| if run.picks.is_some() { format!("picked sets equal {sets}/{pairs}") } else { "picked sets: this run stores no picks".to_owned() };
    let report = |kind: &str, streams: &dyn Fn(usize, usize) -> Vec<f32>, record: &dyn Fn(usize, usize) -> Record| -> (Near, usize) {
        let (mut all, mut sets_all) = (Near::default(), 0);
        for visit in 0..visits {
            let (mut near, mut at_last, mut sets, mut input, mut output) = (Near::default(), Near::default(), 0, Near::default(), Near::default());
            for token in 0..tokens {
                let (mine, record) = (streams(visit, token), record(visit, token));
                near.add(&mine, run.states.at(visit + 1, token));
                if token == last { at_last.add(&mine, run.states.at(visit + 1, token)); }
                if let Some(theirs) = picks(visit, token) { sets += usize::from(same_set(&record.route.ids, theirs)); }
                if let Some(path) = &run.expert_io {
                    input.add(&record.expert_input, &floats(path, ((visit * 2) * tokens + token) * hidden, hidden));
                    output.add(&record.experts, &floats(path, ((visit * 2 + 1) * tokens + token) * hidden, hidden));
                }
            }
            let entry = &model.schedule().visits()[visit];
            println!("{kind} {} visit {visit} layer {} pass {}: last token: {at_last}; all {tokens} tokens: {near}; {}; expert input relative {:.3e}; expert output relative {:.3e}",
                run.name, entry.layer, entry.pass, sets_text(sets, tokens), input.relative(), output.relative());
            all.join(near); sets_all += sets;
        }
        (all, sets_all)
    };
    let outs = forced(runner, &run.states);
    let (forced_near, forced_sets) = report("FORCED", &|visit, token| outs[visit][token].stream.clone(), &|visit, token| outs[visit][token].record.clone().unwrap());
    println!("SUMMARY {} forced ({visits} visits with the reference's inputs, {tokens} tokens): stream {forced_near}; {}", run.name, sets_text(forced_sets, visits * tokens));

    let start = Instant::now();
    let free = runner.read(&mut runner.chat().unwrap(), &run.ids, Want { head: Head::Every, records: true }, threads()).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let (free_near, free_sets) = report("FREE", &|visit, token| free[token].streams[visit].clone(), &|visit, token| free[token].records[visit].clone());
    let (mut end, mut normed, mut embedding, mut tops, mut worst_top) = (Near::default(), Near::default(), Near::default(), 0, 0.0f32);
    for (token, out) in free.iter().enumerate() {
        embedding.add(&out.embedding, run.states.at(0, token));
        end.add(&out.end, run.states.at(visits + 1, token));
        normed.add(&out.normed, run.states.at(visits + 2, token));
        let theirs = floats(&run.logits, token * rows, rows);
        let (mine, (their_top, their_logit)) = (out.logits.as_ref().unwrap(), top(&theirs));
        let equal = out.top == Some(their_top);
        tops += usize::from(equal);
        worst_top = worst_top.max((mine[their_top as usize] - their_logit).abs());
        println!("FREE {} position {token} token {}: top token {} (logit {:.4}), the reference's {their_top} (logit {their_logit:.4}, this form gives it {:.4}), {}; logits {}",
            run.name, run.ids[token], out.top.unwrap(), mine[out.top.unwrap() as usize], mine[their_top as usize], if equal { "equal" } else { "DIFFERENT" }, Near::of(mine, &theirs));
    }
    println!("SUMMARY {} free (from token IDs alone, {seconds:.1} s for {tokens} tokens with every head): stream {free_near}; {}; embedding {embedding}; end mix {end}; last norm {normed}; top token equal {tops}/{tokens}; largest move of the reference's top logit {worst_top:.4}",
        run.name, sets_text(free_sets, visits * tokens));
    (forced_near.relative(), free_sets, visits * tokens, tops)
}

/// Step 2. The 48 visits against the layer trace of the reference (level C of the plan), on the prompt of
/// 16 tokens and on the prompt of 14, each against both ways the reference computed it, and on the dump prompt.
#[test]
#[ignore = "requires the real weight file and the reference trace outside the repository"]
fn every_visit_matches_the_layer_trace() {
    let model = open();
    let runner = model.runner().unwrap();
    let (hidden, visits) = (model.hidden(), model.schedule().visits().len());
    let folder = local("ref").join("contract-A/trace");
    let index = parse(&folder.join("trace-index-t1.json"));
    let mut runs = Vec::new();
    for entry in index.object().unwrap()["files"].array().unwrap() {
        let entry = entry.object().unwrap();
        let ids: Vec<u32> = numbers(&entry["input_ids"]).into_iter().map(|v| v as u32).collect();
        let file = |name: &str| folder.join(entry["files"].object().unwrap()[name].object().unwrap()["file"].string().unwrap());
        let picks = ints(&file("router-picks"), 0, visits * ids.len() * numbers(&entry["files"].object().unwrap()["router-picks"].object().unwrap()["shape"])[2]);
        runs.push(Run { name: format!("trace {} {}", entry["effort"].string().unwrap(), entry["way"].string().unwrap()), states: States::read(&file("states"), ids.len(), hidden, visits),
            picks: Some(picks), expert_io: Some(file("expert-io")), logits: file("logits"), ids });
    }
    let dumps = Dumps::open(local("ref").join("contract-A/ops"));
    runs.push(Run { name: "dumps 8 tokens one-pass".into(), states: States::read(&dumps.root.join("states.f32"), dumps.ids.len(), hidden, visits), picks: None, expert_io: None,
        logits: dumps.root.join("logits.f32"), ids: dumps.ids.clone() });
    for run in &runs {
        let (forced, sets, pairs, tops) = compare(&runner, run);
        let sets = if run.picks.is_some() { format!("free picked sets equal {sets}/{pairs}") } else { "no picks stored".to_owned() };
        println!("GATE {}: forced relative difference {forced:.3e} (gate: under 1e-2); {sets}; top token equal {tops}/{}", run.name, run.ids.len());
        assert!(forced < 0.01, "{}: a visit with the reference's inputs is more than 1 percent from the reference", run.name);
    }
    let counts = model.cache_counts().unwrap();
    println!("CACHE experts widened {}, cache hits {}, held {} ({:.1} GiB)", counts.widened, counts.hits, counts.held, counts.held_bytes as f64 / (1u64 << 30) as f64);
}

/// One row of a score file: position, top token, top logit, target token, target logit, log-sum-exp of all logits.
/// The format of `PROMPT_SCORES.tsv` of the reference sets. The log-sum-exp is made in 64-bit.
fn score_row(position: usize, logits: &[f32], target: Option<u32>) -> String {
    let (top_token, top_logit) = top(logits);
    let sum: f64 = logits.iter().map(|&v| ((v - top_logit) as f64).exp()).sum();
    let lse = top_logit as f64 + sum.ln();
    match target {
        Some(target) => format!("{position}\t{top_token}\t{top_logit:.9e}\t{target}\t{:.9e}\t{lse:.12e}\n", logits[target as usize]),
        None => format!("{position}\t{top_token}\t{top_logit:.9e}\t-1\tnan\t{lse:.12e}\n"),
    }
}
fn second(logits: &[f32], top_token: u32) -> f32 { logits.iter().enumerate().filter(|(t, _)| *t != top_token as usize).map(|(_, &v)| v).fold(f32::NEG_INFINITY, f32::max) }

/// Step 3. The first token from token IDs alone: the prompt of the layer trace, then 8 more tokens, one a read.
#[test]
#[ignore = "requires the real weight file and the reference answers outside the repository"]
fn first_token_from_token_ids() {
    let model = open();
    let runner = model.runner().unwrap();
    let item = local("ref").join("sets/contract-A/own-1-one-token__default");
    let (prompt, answer) = (ids(&item.join("INPUT_IDS")), ids(&item.join("OUTPUT_IDS")));
    let mut chat = runner.chat().unwrap();
    let start = Instant::now();
    let outs = runner.read(&mut chat, &prompt, Want { head: Head::Last, records: false }, threads()).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let first = outs.last().unwrap();
    let (token, logits) = (first.top.unwrap(), first.logits.as_ref().unwrap());
    println!("TOKEN first: prompt own-1-one-token (effort default, {} token IDs in), token {token}, the maker's token {}, {seconds:.2} s for the prompt, {:.3} tokens a second; top logit {:.4}, second logit {:.4}; {}",
        prompt.len(), answer[0], prompt.len() as f64 / seconds, logits[token as usize], second(logits, token), if token == answer[0] { "EQUAL" } else { "MISS" });
    let counts = model.cache_counts().unwrap();
    println!("CACHE after the prompt: experts widened {}, cache hits {}, held {} ({:.1} GiB)", counts.widened, counts.hits, counts.held, counts.held_bytes as f64 / (1u64 << 30) as f64);
    let (mut written, mut next) = (vec![token], token);
    let start = Instant::now();
    for (step, theirs) in answer.iter().enumerate().take(9).skip(1) {
        let begin = Instant::now();
        let out = runner.read(&mut chat, &[next], Want { head: Head::Last, records: false }, threads()).unwrap().pop().unwrap();
        next = out.top.unwrap();
        println!("TOKEN new token {step}: {next}, the maker's {theirs}, {:.2} s", begin.elapsed().as_secs_f64());
        written.push(next);
    }
    let seconds = start.elapsed().as_secs_f64();
    let shared = written.iter().zip(&answer).take_while(|(a, b)| a == b).count();
    println!("TOKEN writing: 8 tokens in {seconds:.2} s, {:.3} tokens a second, one token a read; the first {shared} of 9 tokens equal the maker's", 8.0 / seconds);
    assert_eq!(token, answer[0], "the first token differs from the maker's");
}

/// Writes the trace of this host form for the prompt of the first-token test and its first 8 new tokens, for a
/// device run to compare with: `.local/loop-host/out/vade-host-trace/first/`.
/// - `prompt.ids`: the prompt, one token ID a line. Position `p` after the prompt is fed the top token of `p - 1`.
/// - `tokens.tsv`: position, a tab, the top token of that position.
/// - `pos<NNNN>.streams.f32`: the stream after each visit of the visit list, in run order, little-endian FP32.
/// - `pos<NNNN>.logits.f32`: the logits of that position after the cap, one FP32 for each used head row.
#[test]
#[ignore = "requires the real weight file and the reference answers outside the repository"]
fn first_token_trace_is_written() {
    let model = open();
    let runner = model.runner().unwrap();
    let prompt = ids(&local("ref").join("sets/contract-A/own-1-one-token__default/INPUT_IDS"));
    let out = local("ref").parent().unwrap().join("out/vade-host-trace/first");
    std::fs::create_dir_all(&out).unwrap();
    let want = Want { head: Head::Every, records: false };
    let mut chat = runner.chat().unwrap();
    let mut outs = runner.read(&mut chat, &prompt, want, threads()).unwrap();
    for _ in 0..8 {
        let next = outs.last().unwrap().top.unwrap();
        outs.extend(runner.read(&mut chat, &[next], want, threads()).unwrap());
    }
    let write = |name: String, bytes: Vec<u8>| File::create(out.join(name)).unwrap().write_all(&bytes).unwrap();
    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let mut tokens = String::new();
    for (position, token) in outs.iter().enumerate() {
        assert_eq!((token.streams.len(), token.streams[0].len()), (model.schedule().visits().len(), model.hidden()));
        write(format!("pos{position:04}.streams.f32"), token.streams.iter().flat_map(|stream| bytes(stream)).collect());
        write(format!("pos{position:04}.logits.f32"), bytes(token.logits.as_ref().unwrap()));
        tokens += &format!("{position}\t{}\n", token.top.unwrap());
    }
    write("prompt.ids".into(), prompt.iter().map(|id| format!("{id}\n")).collect::<String>().into_bytes());
    write("tokens.tsv".into(), tokens.clone().into_bytes());
    println!("TRACE {} positions ({} of the prompt, 8 new), {} visits, width {}, {} logits a position; top tokens: {}", outs.len(), prompt.len(),
        model.schedule().visits().len(), model.hidden(), model.head_rows(), tokens.lines().map(|l| l.split('\t').nth(1).unwrap()).collect::<Vec<_>>().join(" "));
}

/// The trace of `first_token_trace_is_written` again in three other ways: all 24 tokens in one read, one token a
/// read from a clean chat, and one read on one thread. Each way must give the streams and the logits of the trace
/// files bit for bit. It reads the files that the other test wrote.
#[test]
#[ignore = "requires the real weight file and the trace that first_token_trace_is_written wrote"]
fn the_first_token_trace_is_the_same_at_every_cut_and_thread_count() {
    let model = open();
    let runner = model.runner().unwrap();
    let folder = local("ref").parent().unwrap().join("out/vade-host-trace/first");
    let tops: Vec<u32> = std::fs::read_to_string(folder.join("tokens.tsv")).unwrap().lines().map(|l| l.split('\t').nth(1).unwrap().parse().unwrap()).collect();
    let mut fed = ids(&folder.join("prompt.ids"));
    let prompt = fed.len();
    fed.extend_from_slice(&tops[prompt - 1..tops.len() - 1]);
    let (visits, hidden, rows) = (model.schedule().visits().len(), model.hidden(), model.head_rows());
    let want = Want { head: Head::Every, records: false };
    let one_read = |threads: usize| runner.read(&mut runner.chat().unwrap(), &fed, want, threads).unwrap();
    let singles = || { let mut chat = runner.chat().unwrap(); fed.iter().flat_map(|&t| runner.read(&mut chat, &[t], want, threads()).unwrap()).collect::<Vec<_>>() };
    let ways: [(&str, Vec<_>); 3] = [("all tokens in one read", one_read(threads())), ("one token a read", singles()), ("one read on one thread", one_read(1))];
    for (way, outs) in &ways {
        let (mut words, mut differing, mut first) = (0usize, 0usize, None);
        for (position, token) in outs.iter().enumerate() {
            let streams = floats(&folder.join(format!("pos{position:04}.streams.f32")), 0, visits * hidden);
            let logits = floats(&folder.join(format!("pos{position:04}.logits.f32")), 0, rows);
            let mine = token.streams.iter().flatten().chain(token.logits.as_ref().unwrap());
            for (word, (a, b)) in mine.zip(streams.iter().chain(&logits)).enumerate() {
                words += 1;
                if a.to_bits() != b.to_bits() { differing += 1; first.get_or_insert((position, word / hidden, word % hidden)); }
            }
            assert_eq!(token.top, Some(tops[position]), "{way}: another top token at position {position}");
        }
        println!("CUT {way}: {} positions, {words} words (streams and logits), {differing} differ from the trace files{}", outs.len(),
            first.map_or(String::new(), |(p, v, w)| format!("; first at position {p}, visit {v}, word {w}")));
        assert_eq!(differing, 0, "{way}: the words depend on the cut or on the thread count");
    }
}

/// Writes the long reference of this host form for a device run to compare with: a forced read of the whole token
/// sequence (input and answer of the reference set) of each prompt named in `.local/loop-host/long-items.txt`;
/// without the file, the prompt of 2,264 tokens. Folder `.local/loop-host/out/vade-host-long/<prompt>/`:
/// - `prompt.ids`: every fed token ID, one a line. Every position is a prompt row.
/// - `tokens.tsv`: position, a tab, the top token of that position.
/// - `hashes.tsv`: position, fed ID, top ID, then three SHA-256: of the stream after the last visit, of the streams
///   after all visits in run order, of the logits. Each hash is of the little-endian FP32 bytes.
/// - `visits.tsv`: position, then one SHA-256 for the stream after each visit, in run order.
/// - `scores.tsv`: one score row a position (see `score_row`); the target is the next fed token. So a forced read
///   of another run's answer shows, at each position, if the top token is that run's token and how far its logit is.
/// - `pos<NNNN>.streams.f32`, `pos<NNNN>.logits.f32`: the words themselves, as in the short trace, at the first two
///   positions, at the last one, and at each multiple of 1,024 with its two neighbours.
///
/// The text is read in pieces of 256 tokens, so the memory does not grow with the logits of every position. A read
/// gives the same words however a text is cut.
#[test]
#[ignore = "requires the real weight file and the reference answers outside the repository"]
fn long_reference_is_written() {
    let model = open();
    let runner = model.runner().unwrap();
    let set = local("ref").join("sets/contract-A");
    let listed = local("ref").parent().unwrap().join("long-items.txt");
    let items: Vec<String> = if listed.exists() { std::fs::read_to_string(&listed).unwrap().split_whitespace().map(str::to_owned).collect() } else { vec!["own-7-log-2k__none".to_owned()] };
    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    for item in &items {
        let fed = ids(&set.join(item).join("SEQ_IDS"));
        let out = local("ref").parent().unwrap().join("out/vade-host-long").join(item);
        std::fs::create_dir_all(&out).unwrap();
        let write = |name: String, data: &[u8]| File::create(out.join(name)).unwrap().write_all(data).unwrap();
        let (mut tokens, mut hashes, mut visits, mut scores, mut full) = (String::new(), String::new(), String::new(), String::new(), Vec::new());
        let mut chat = runner.chat().unwrap();
        let start = Instant::now();
        for (piece, chunk) in fed.chunks(256).enumerate() {
            let outs = runner.read(&mut chat, chunk, Want { head: Head::Every, records: false }, threads()).unwrap();
            for (k, token) in outs.iter().enumerate() {
                let position = piece * 256 + k;
                let streams: Vec<Vec<u8>> = token.streams.iter().map(|stream| bytes(stream)).collect();
                let (all, logits, top) = (streams.concat(), bytes(token.logits.as_ref().unwrap()), token.top.unwrap());
                tokens += &format!("{position}\t{top}\n");
                hashes += &format!("{position}\t{}\t{top}\t{}\t{}\t{}\n", fed[position], engine_hash::digest(streams.last().unwrap()), engine_hash::digest(&all), engine_hash::digest(&logits));
                visits += &format!("{position}\t{}\n", streams.iter().map(|stream| engine_hash::digest(stream)).collect::<Vec<_>>().join("\t"));
                scores += &score_row(position, token.logits.as_ref().unwrap(), fed.get(position + 1).copied());
                if position + 1 == fed.len() || position % 1024 <= 1 || position % 1024 == 1023 {
                    write(format!("pos{position:04}.streams.f32"), &all);
                    write(format!("pos{position:04}.logits.f32"), &logits);
                    full.push(position);
                }
            }
            println!("LONG {item}: {} of {} positions, {:.0} s", (piece * 256 + chunk.len()).min(fed.len()), fed.len(), start.elapsed().as_secs_f64());
        }
        write("prompt.ids".into(), fed.iter().map(|id| format!("{id}\n")).collect::<String>().as_bytes());
        write("tokens.tsv".into(), tokens.as_bytes());
        write("hashes.tsv".into(), hashes.as_bytes());
        write("visits.tsv".into(), visits.as_bytes());
        write("scores.tsv".into(), scores.as_bytes());
        let seconds = start.elapsed().as_secs_f64();
        println!("LONG {item}: {} positions in {seconds:.0} s, {:.2} positions a second; {} visits, width {}, {} logits a position; full words at positions {full:?}; SHA-256 of hashes.tsv {}, of visits.tsv {}",
            fed.len(), fed.len() as f64 / seconds, model.schedule().visits().len(), model.hidden(), model.head_rows(), engine_hash::digest(hashes.as_bytes()), engine_hash::digest(visits.as_bytes()));
    }
}

/// The decode rate of a greedy run from token IDs alone, one token a read, 64 new tokens for each prompt. The
/// prompts are the names in `.local/loop-host/decode-items.txt`, or the three decode prompts of the reference set.
/// The rate is (new tokens - 1) divided by the seconds from the first new token to the last, as the notes of the
/// reference give its own rate. The time to the first token starts when the read of the prompt starts: the dense
/// weights are open then, and the experts are widened as the tokens pick them.
#[test]
#[ignore = "requires the real weight file and the reference answers outside the repository"]
fn decode_rates_from_token_ids() {
    let model = open();
    let runner = model.runner().unwrap();
    let set = local("ref").join("sets/contract-A");
    let listed = local("ref").parent().unwrap().join("decode-items.txt");
    let items: Vec<String> = if listed.exists() { std::fs::read_to_string(&listed).unwrap().split_whitespace().map(str::to_owned).collect() } else {
        ["rig-decode-prose-1__default", "rig-decode-code-1__default", "rig-decode-structured-1__default"].map(str::to_owned).to_vec()
    };
    for item in &items {
        let (prompt, answer) = (ids(&set.join(item).join("INPUT_IDS")), ids(&set.join(item).join("OUTPUT_IDS")));
        let count = answer.len().min(64);
        let want = Want { head: Head::Last, records: false };
        let mut chat = runner.chat().unwrap();
        let start = Instant::now();
        let mut written = vec![runner.read(&mut chat, &prompt, want, threads()).unwrap().pop().unwrap().top.unwrap()];
        let first = start.elapsed().as_secs_f64();
        let after_first = Instant::now();
        while written.len() < count {
            let last = *written.last().unwrap();
            written.push(runner.read(&mut chat, &[last], want, threads()).unwrap().pop().unwrap().top.unwrap());
        }
        let seconds = after_first.elapsed().as_secs_f64();
        let equal = written.iter().zip(&answer).filter(|(a, b)| a == b).count();
        let shared = written.iter().zip(&answer).take_while(|(a, b)| a == b).count();
        let counts = model.cache_counts().unwrap();
        println!("RATE {item}: {:.3} tokens a second after the first token ({} tokens in {seconds:.2} s), {count} new tokens, first token after {first:.2} s ({} input tokens in one read), {equal} of {count} tokens equal to the maker's (shared start {shared}); greedy, one token a read, host arithmetic on a CPU, no GPU, {} threads; expert cache limit {} GiB, experts widened {}, cache hits {}, held {}",
            (count - 1) as f64 / seconds, count - 1, prompt.len(), threads(), cache_bytes() >> 30, counts.widened, counts.hits, counts.held);
        println!("TOKENS {item}: {}", written.iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
        println!("MAKER  {item}: {}", answer[..count].iter().map(u32::to_string).collect::<Vec<_>>().join(" "));
    }
}

/// Step 4. Levels B1 and B2 of the plan on the answer folders of the reference set.
/// B1: the maker's tokens are given (one read of the maker's input and answer). For each position: is the top
/// token of this form the maker's next token, and how far is the score of that token.
/// B2: a free greedy run from the maker's input. A read gives the same words however a text is cut, so the free
/// run equals the forced run up to the first position where the top token is not the maker's. The run goes on
/// from there token by token; with the file `forced-only.txt` it stops at the parting token, which is all that
/// B1 and B2 need. The score files go to `.local/loop-host/out/vade-host/` in the layout of the sets.
#[test]
#[ignore = "requires the real weight file and the reference answers outside the repository"]
fn greedy_answers_match_the_reference_set() {
    let model = open();
    let runner = model.runner().unwrap();
    let set = local("ref").join("sets/contract-A");
    let out_root = local("ref").parent().unwrap().join("out/vade-host");
    let listed = local("ref").parent().unwrap().join("items.txt");
    let forced_only = local("ref").parent().unwrap().join("forced-only.txt").exists();
    let items: Vec<String> = if listed.exists() { std::fs::read_to_string(&listed).unwrap().split_whitespace().map(str::to_owned).collect() } else {
        // With no list: every answer folder whose input and answer hold 400 tokens or fewer, in name order.
        let mut all: Vec<String> = std::fs::read_dir(&set).unwrap().map(|e| e.unwrap().path()).filter(|p| p.join("SEQ_IDS").exists() && ids(&p.join("SEQ_IDS")).len() <= 400)
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned()).collect();
        all.sort();
        all
    };
    let (mut positions, mut top_equal, mut answer_positions, mut answer_equal, mut gap_sum, mut whole, mut shared_sum, mut new_sum) = (0usize, 0usize, 0usize, 0usize, 0.0f64, 0usize, 0usize, 0usize);
    let begin = Instant::now();
    for item in &items {
        let folder = set.join(item);
        let (input, answer, sequence) = (ids(&folder.join("INPUT_IDS")), ids(&folder.join("OUTPUT_IDS")), ids(&folder.join("SEQ_IDS")));
        assert_eq!(sequence, [input.clone(), answer.clone()].concat(), "{item}: the sequence is not the input and the answer");
        let reference: Vec<Vec<String>> = std::fs::read_to_string(folder.join("SCORES.tsv")).unwrap().lines().map(|l| l.split('\t').map(str::to_owned).collect()).collect();
        let start = Instant::now();
        // Two reads: the input, then the answer. A read gives the same words however a text is cut. The chat
        // after the input is kept as the fork point of the free run.
        let mut chat = runner.chat().unwrap();
        let mut outs = runner.read(&mut chat, &input, Want { head: Head::Every, records: false }, threads()).unwrap();
        let mut free = chat.clone();
        outs.extend(runner.read(&mut chat, &answer, Want { head: Head::Every, records: false }, threads()).unwrap());
        let seconds = start.elapsed().as_secs_f64();
        let out = out_root.join(item);
        std::fs::create_dir_all(&out).unwrap();
        let write = |name: &str, text: String| File::create(out.join(name)).unwrap().write_all(text.as_bytes()).unwrap();
        let join = |tokens: &[u32]| tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" ") + "\n";
        let rows: Vec<String> = outs.iter().enumerate().map(|(p, o)| score_row(p, o.logits.as_ref().unwrap(), sequence.get(p + 1).copied())).collect();
        write("INPUT_IDS", join(&input)); write("SEQ_IDS", join(&sequence));
        write("SCORES.tsv", rows.concat()); write("PROMPT_SCORES.tsv", rows[..input.len() - 1].concat() + &score_row(input.len() - 1, outs[input.len() - 1].logits.as_ref().unwrap(), None));
        // B1 on this item. The gap is the difference of the target's score in nats: log-sum-exp minus the target logit.
        let (mut equal, mut equal_answer, mut gap) = (0usize, 0usize, 0.0f64);
        for (p, o) in outs.iter().enumerate().take(sequence.len() - 1) {
            let (logits, target, theirs) = (o.logits.as_ref().unwrap(), sequence[p + 1], &reference[p]);
            assert_eq!(theirs[3].parse::<u32>().unwrap(), target, "{item}: the reference scores another target at position {p}");
            let is_top = o.top == Some(target);
            equal += usize::from(is_top);
            if p + 1 >= input.len() { equal_answer += usize::from(is_top); }
            let (_, top_logit) = top(logits);
            let sum: f64 = logits.iter().map(|&v| ((v - top_logit) as f64).exp()).sum();
            let mine = top_logit as f64 + sum.ln() - logits[target as usize] as f64;
            gap += (mine - (theirs[5].parse::<f64>().unwrap() - theirs[4].parse::<f64>().unwrap())).abs();
        }
        // B2: the shared start of the free run, then the rest of the free run.
        let shared = (0..answer.len()).take_while(|&k| outs[input.len() - 1 + k].top == Some(answer[k])).count();
        let mut own: Vec<u32> = answer[..shared].to_vec();
        let mut own_rows: Vec<String> = (0..shared).map(|k| score_row(input.len() - 1 + k, outs[input.len() - 1 + k].logits.as_ref().unwrap(), Some(answer[k]))).collect();
        let mut part = String::new();
        if shared < answer.len() {
            let at = input.len() - 1 + shared;
            let (logits, mine) = (outs[at].logits.as_ref().unwrap(), outs[at].top.unwrap());
            part = format!("; parts at new token {shared} (position {at}): this form writes {mine} (logit {:.4}) and gives the maker's token {} the logit {:.4}; the maker's own top logit there is {}",
                logits[mine as usize], answer[shared], logits[answer[shared] as usize], reference[at][2]);
            // Go on from the fork after the input: read the shared start of the answer again, then write token by token.
            // With `forced-only.txt` the own answer ends with the parting token.
            let end = if forced_only { shared + 1 } else { answer.len() };
            if shared > 0 && !forced_only { runner.read(&mut free, &answer[..shared], Want { head: Head::None, records: false }, threads()).unwrap(); }
            if forced_only { part += "; the free run is not continued after the parting token"; }
            let (mut token, mut row) = (mine, score_row(at, logits, Some(mine)));
            for k in shared..end {
                own_rows.push(row.clone());
                own.push(token);
                if k + 1 < end {
                    let last = runner.read(&mut free, &[token], Want { head: Head::Last, records: false }, threads()).unwrap().pop().unwrap();
                    token = last.top.unwrap();
                    row = score_row(input.len() + k, last.logits.as_ref().unwrap(), Some(token));
                }
            }
        }
        write("OUTPUT_IDS", join(&own)); write("OUTPUT_SCORES.tsv", own_rows.concat());
        let scored = sequence.len() - 1;
        println!("ITEM {item}: {} input tokens, {} new tokens; B1 top token is the maker's token at {equal}/{scored} positions ({equal_answer}/{} in the answer), mean gap {:.4} nats; B2 shared start {shared}/{} new tokens{part}; {seconds:.1} s for the forced read, {:.2} tokens a second",
            input.len(), answer.len(), answer.len(), gap / scored as f64, answer.len(), sequence.len() as f64 / seconds);
        positions += scored; top_equal += equal; answer_positions += answer.len(); answer_equal += equal_answer; gap_sum += gap;
        whole += usize::from(shared == answer.len()); shared_sum += shared; new_sum += answer.len();
    }
    let counts = model.cache_counts().unwrap();
    println!("CACHE experts widened {}, cache hits {}, held {} ({:.1} GiB)", counts.widened, counts.hits, counts.held, counts.held_bytes as f64 / (1u64 << 30) as f64);
    println!("SUMMARY answers, {} items in {:.0} s: B1 top token is the maker's token at {top_equal}/{positions} positions ({:.2} percent), in the answers {answer_equal}/{answer_positions} ({:.2} percent), mean gap {:.4} nats a token; B2 whole answer equal in {whole}/{} items, shared start {shared_sum}/{new_sum} new tokens",
        items.len(), begin.elapsed().as_secs_f64(), 100.0 * top_equal as f64 / positions.max(1) as f64, 100.0 * answer_equal as f64 / answer_positions.max(1) as f64, gap_sum / positions.max(1) as f64, items.len());
    assert!(!items.is_empty() && positions > 0, "no answer folder was scored");
}
