//! Synthetic mixed checkpoint: real packed FP4/CSF, MXFP8 column/scale shards,
//! disk rows, recurrent head state, shared TP and vocabulary ownership over TCP.
use engine_model::{
    flash::{
        mxfp8, nvfp4 as fp4,
        shard::{Placement, ShardCheckpoint, ShardPlan},
    },
    safetensors::Dtype,
    shapes::*,
};
use spark_tp::{
    next::{NextModel, RankKernels, SharedTransport},
    transport::Ring,
};
use std::{
    collections::BTreeMap,
    fs,
    net::TcpListener,
    ops::Range,
    path::{Path, PathBuf},
};
type R<T> = Result<T, String>;
fn raw(c: &ShardCheckpoint, n: &str) -> Vec<u8> {
    let mut b = vec![0; c.plan.tensors[n].bytes() as usize];
    c.read_local(n, 0, &mut b).unwrap();
    b
}
fn fixture(root: &Path) -> (ModelShape, BTreeMap<String, String>) {
    let mut tensors = Vec::new();
    let mut payload = [Vec::<u8>::new(), Vec::<u8>::new()];
    let mut headers = [Vec::new(), Vec::new()];
    let mut files = BTreeMap::new();
    let mut add = |name: String, part, store, dtype: Dtype, shape: Vec<u64>, bytes: Vec<u8>| {
        let f = tensors.len() % 2;
        let start = payload[f].len();
        payload[f].extend(&bytes);
        headers[f].push(format!(
            "{name:?}:{{\"dtype\":{:?},\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
            match dtype {
                Dtype::U8 => "U8",
                Dtype::U32 => "U32",
                Dtype::F32 => "F32",
                Dtype::Bf16 => "BF16",
                _ => "F8_E4M3",
            },
            payload[f].len()
        ));
        files.insert(name.clone(), format!("model{f}.safetensors"));
        tensors.push(TensorEntry {
            name,
            dtype,
            shape,
            role: Role {
                place: if matches!(part, Part::Embedding | Part::Head) {
                    Place::Model
                } else {
                    Place::Layer(0)
                },
                part,
                store,
            },
        });
    };
    for (n, p) in [("embed", Part::Embedding), ("head", Part::Head)] {
        add(
            n.into(),
            p,
            Store::Weight,
            Dtype::Bf16,
            vec![8, 32],
            (0..256)
                .flat_map(|i| fp4::f32_to_bf16(((i * 7 % 23) as f32 - 11.) / 32.).to_le_bytes())
                .collect(),
        );
    }
    for part in 0..2 {
        add(
            format!("ple{part}"),
            Part::Lexical { part },
            Store::Weight,
            Dtype::U8,
            vec![16, 16],
            vec![0x12; 256],
        );
    }
    for (name, part, r, k) in [
        ("attn", Part::Output, 32, 128),
        ("sg", Part::Shared(Proj::Gate), 128, 32),
        ("su", Part::Shared(Proj::Up), 128, 32),
        ("sd", Part::Shared(Proj::Down), 32, 128),
    ] {
        add(
            name.into(),
            part,
            Store::Weight,
            Dtype::F8E4M3,
            vec![r, k],
            (0..r * k).map(|i| 0x18 + (i % 5) as u8).collect(),
        );
        add(
            format!("{name}s"),
            part,
            Store::BlockScale,
            Dtype::U8,
            vec![r, k / 32],
            vec![127; (r * k / 32) as usize],
        );
    }
    for id in 0..4 {
        for (p, proj) in [("g", Proj::Gate), ("u", Proj::Up), ("d", Proj::Down)] {
            let part = Part::Expert { id: Some(id), proj };
            add(
                format!("e{id}{p}"),
                part,
                Store::Weight,
                Dtype::U8,
                vec![32, 16],
                vec![0x12 + id as u8; 512],
            );
            let mut fixed = vec![0; 64];
            fixed[..16].fill(0x38);
            fixed[32..48].fill(0x38);
            add(
                format!("e{id}{p}f"),
                part,
                Store::BlockScale,
                Dtype::U8,
                vec![2, 32],
                fixed,
            );
            add(
                format!("e{id}{p}x"),
                part,
                Store::BlockScale,
                Dtype::U32,
                vec![0],
                vec![],
            );
            add(
                format!("e{id}{p}s"),
                part,
                Store::Scalar,
                Dtype::F32,
                vec![],
                0.03125f32.to_le_bytes().to_vec(),
            );
        }
    }
    for f in 0..2 {
        let header = format!("{{{}}}", headers[f].join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.bytes());
        bytes.extend(&payload[f]);
        fs::write(root.join(format!("model{f}.safetensors")), bytes).unwrap();
    }
    let map = files
        .iter()
        .map(|(n, f)| format!("{n:?}:{f:?}"))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        root.join("model.safetensors.index.json"),
        format!("{{\"metadata\":{{}},\"weight_map\":{{{map}}}}}"),
    )
    .unwrap();
    (
        ModelShape {
            hidden: 32,
            vocab: 8,
            max_positions: 32,
            layers: vec![LayerShape {
                mixer: Mixer::Attention(AttentionShape {
                    query_heads: 4,
                    kv_heads: 1,
                    head_dim: 32,
                    rotary_dim: 8,
                    gated: false,
                    index: None,
                }),
                ffn: Ffn::Experts(ExpertShape {
                    count: 4,
                    selected: 2,
                    width: 32,
                    shared_width: 128,
                    storage: ExpertStorage::PerExpert,
                }),
            }],
            draft: vec![],
            residual: Residual::Single,
            lexical: Some(LexicalShape {
                layer: 0,
                parts: 2,
                part_rows: 16,
                row_dim: 32,
            }),
            tensors,
        },
        files,
    )
}
struct Cpu {
    c: ShardCheckpoint,
    x: Vec<f32>,
    state: [f32; 4],
    token: u32,
    trace: Vec<f32>,
    logits: Vec<f32>,
    fail: bool,
    routes: Vec<(u32, f32)>,
}
impl Cpu {
    fn mx(&self, n: &str, input: &[f32]) -> Vec<f32> {
        let w = raw(&self.c, n);
        let mut f = vec![0.; w.len()];
        mxfp8::dequantize(&w, &raw(&self.c, &format!("{n}s")), &mut f).unwrap();
        f.chunks_exact(input.len())
            .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
            .collect()
    }
    fn fp(&self, n: &str, input: &[u16]) -> Vec<f32> {
        let scales = fp4::csf_expand(&raw(&self.c, &format!("{n}f")), &[], 32, 32).unwrap();
        let mut out = vec![0.; 32];
        fp4::project_ref(
            input,
            &raw(&self.c, n),
            &scales,
            f32::from_le_bytes(raw(&self.c, &format!("{n}s")).try_into().unwrap()),
            32,
            32,
            &mut out,
        )
        .unwrap();
        out
    }
}
impl RankKernels for Cpu {
    fn front(&mut self, c: &ShardCheckpoint, t: u32, _: usize, out: &mut [f32]) -> R<()> {
        self.token = t;
        for (n, row, stride) in [
            ("embed".to_string(), t as u64, 64),
            (format!("ple{}", t / 16), t as u64 % 16, 16),
        ] {
            let s = &c.plan.tensors[&n];
            if let Some(at) = s.selection.local_byte_offset(&s.tensor, row * stride) {
                let mut b = vec![0; stride as usize];
                c.read_local(&n, at, &mut b).unwrap();
                if n == "embed" {
                    for (i, b) in b.as_chunks::<2>().0.iter().enumerate() {
                        out[i] = fp4::bf16_to_f32(u16::from_le_bytes(*b));
                    }
                } else {
                    fp4::dequantize(&b, &[0x38; 2], 0.03125, &mut out[32..]).unwrap();
                }
            }
        }
        Ok(())
    }
    fn embed(&mut self, r: &[f32]) -> R<()> {
        self.x = r[..32].iter().zip(&r[32..]).map(|(a, b)| a + b).collect();
        Ok(())
    }
    fn attention(&mut self, _: usize, _: usize, out: &mut [f32]) -> R<()> {
        if self.fail || self.token == 7 && self.c.plan.rank == 0 {
            return Err("injected local kernel failure".into());
        }
        let rank = self.c.plan.rank;
        let world = self.c.plan.world;
        let heads = 4 / world;
        let mut input = Vec::new();
        for head in rank * heads..(rank + 1) * heads {
            self.state[head] = self.state[head] * 0.75 + self.x[head];
            input.extend((0..32).map(|i| self.x[i] + self.state[head] * 0.01));
        }
        out.copy_from_slice(&self.mx("attn", &input));
        Ok(())
    }
    fn inject_attention(&mut self, _: usize, r: &[f32]) -> R<()> {
        for (a, b) in self.x.iter_mut().zip(r) {
            *a += b * 0.01
        }
        Ok(())
    }
    fn moe(&mut self, _: usize, out: &mut [f32]) -> R<()> {
        let input = self
            .x
            .iter()
            .map(|&v| fp4::f32_to_bf16(v))
            .collect::<Vec<_>>();
        let mut ids = [self.token % 4, (self.token + 1) % 4];
        ids.sort();
        self.routes = ids
            .map(|id| {
                (
                    id,
                    if self.token == 0 && self.c.plan.rank == 1 {
                        0.51
                    } else {
                        0.5
                    },
                )
            })
            .to_vec();
        for id in ids {
            if self.c.plan.tensors[&format!("e{id}g")].placement == Placement::Absent {
                continue;
            }
            let g = self.fp(&format!("e{id}g"), &input);
            let u = self.fp(&format!("e{id}u"), &input);
            let mid = g
                .iter()
                .zip(u)
                .map(|(&g, u)| fp4::swiglu_ref(g, u))
                .collect::<Vec<_>>();
            for (a, b) in out.iter_mut().zip(self.fp(&format!("e{id}d"), &mid)) {
                *a += b * 0.5
            }
        }
        let g = self.mx("sg", &self.x);
        let u = self.mx("su", &self.x);
        let mid = g
            .iter()
            .zip(u)
            .map(|(&g, u)| fp4::bf16_to_f32(fp4::swiglu_ref(g, u)))
            .collect::<Vec<_>>();
        for (a, b) in out[32..].iter_mut().zip(self.mx("sd", &mid)) {
            *a += b * 0.25
        }
        Ok(())
    }
    fn separate_shared(&self) -> bool {
        true
    }
    fn routing(&self) -> &[(u32, f32)] {
        &self.routes
    }
    fn inject_moe(&mut self, _: usize, r: &[f32]) -> R<()> {
        for (a, b) in self.x.iter_mut().zip(r) {
            *a += b * 0.01
        }
        self.trace.extend(&self.x);
        Ok(())
    }
    fn logits(&mut self, _: Range<u64>, out: &mut [f32]) -> R<()> {
        let w = raw(&self.c, "head");
        for (o, row) in out.iter_mut().zip(w.as_chunks::<64>().0.iter()) {
            *o = row
                .as_chunks::<2>().0.iter()
                .zip(&self.x)
                .map(|(b, x)| fp4::bf16_to_f32(u16::from_le_bytes(*b)) * x)
                .sum()
        }
        self.logits = out.to_vec();
        Ok(())
    }
    fn reset(&mut self) -> R<()> {
        self.state = [0.; 4];
        self.trace.clear();
        Ok(())
    }
}
fn checkpoint(
    root: &Path,
    s: &ModelShape,
    f: &BTreeMap<String, String>,
    rank: usize,
    world: usize,
) -> ShardCheckpoint {
    ShardCheckpoint::bind(root, ShardPlan::new(s, f, rank, world).unwrap()).unwrap()
}
fn run(
    root: &Path,
    s: &ModelShape,
    f: &BTreeMap<String, String>,
    world: usize,
    path: bool,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let listeners = (0..world)
        .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect::<Vec<_>>();
    let addresses = listeners
        .iter()
        .map(|l| l.local_addr().unwrap())
        .collect::<Vec<_>>();
    drop(listeners);
    std::thread::scope(|scope| {
        let jobs = (0..world)
            .map(|rank| {
                let addresses = &addresses;
                scope.spawn(move || {
                    let model =
                        NextModel::from_checkpoint(checkpoint(root, s, f, rank, world)).unwrap();
                    let cpu = Cpu {
                        c: checkpoint(root, s, f, rank, world),
                        x: vec![0.; 32],
                        state: [0.; 4],
                        token: 0,
                        trace: vec![],
                        logits: vec![],
                        fail: false,
                        routes: vec![],
                    };
                    let ring = Ring::connect_options(
                        rank,
                        world,
                        addresses[rank],
                        addresses[(rank + 1) % world],
                        path,
                    )
                    .unwrap();
                    let shared = SharedTransport::new(ring);
                    let mut session = model.session(shared.clone(), cpu, 1, 16).unwrap();
                    let mut ids = session.prefill(&[1, 3, 2, 5]).unwrap();
                    ids.extend(session.prefill(&[6, 4]).unwrap());
                    let make_cpu = || Cpu {
                        c: checkpoint(root, s, f, rank, world),
                        x: vec![0.; 32],
                        state: [0.; 4],
                        token: 0,
                        trace: vec![],
                        logits: vec![],
                        fail: false,
                        routes: vec![],
                    };
                    let mut rows = (0..8)
                        .map(|i| {
                            model
                                .session(shared.clone(), make_cpu(), 100 + i, 16)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let mut solo = (0..8)
                        .map(|i| {
                            model
                                .session(shared.clone(), make_cpu(), 200 + i, 16)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    for (i, (a, b)) in rows.iter_mut().zip(&mut solo).enumerate() {
                        for _ in 0..i % 3 {
                            assert_eq!(a.step(3).unwrap(), b.step(3).unwrap());
                        }
                    }
                    for step in 0..6 {
                        let tokens = (0..8).map(|i| 1 + (i + step) % 6).collect::<Vec<_>>();
                        let expected = solo
                            .iter_mut()
                            .zip(&tokens)
                            .map(|(s, &t)| s.step(t).unwrap())
                            .collect::<Vec<_>>();
                        assert_eq!(
                            spark_tp::next::NextSession::step_rows(
                                &mut rows.iter_mut().collect::<Vec<_>>(),
                                &tokens
                            )
                            .unwrap(),
                            expected
                        );
                        for (a, b) in rows.iter().zip(&solo) {
                            for (a, b) in [
                                (&a.kernels().trace, &b.kernels().trace),
                                (&a.kernels().logits, &b.kernels().logits),
                            ] {
                                assert_eq!(
                                    a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                                );
                            }
                        }
                    }
                    let positions = rows.iter().map(|s| s.position()).collect::<Vec<_>>();
                    assert!(
                        spark_tp::next::NextSession::step_rows(
                            &mut rows.iter_mut().collect::<Vec<_>>(),
                            &[99; 8]
                        )
                        .is_err()
                    );
                    assert_eq!(
                        positions,
                        rows.iter().map(|s| s.position()).collect::<Vec<_>>()
                    );
                    assert!(
                        spark_tp::next::NextSession::step_rows(
                            &mut rows.iter_mut().collect::<Vec<_>>(),
                            &[7; 8]
                        )
                        .is_err()
                    );
                    assert!(rows.iter().all(|s| s.is_failed()));
                    let trace = session.kernels().trace.clone();
                    assert_eq!(session.position(), 6);
                    assert!(session.step(99).is_err());
                    assert_eq!(session.position(), 6);
                    session.reset().unwrap();
                    assert_eq!(ids, session.prefill(&[1, 3, 2, 5, 6, 4]).unwrap());
                    assert_eq!(trace, session.kernels().trace);
                    assert!(session.step(7).is_err());
                    assert!(session.is_failed());
                    session.reset().unwrap();
                    let mismatch = session.step(0);
                    assert_eq!(mismatch.is_err(), world.is_multiple_of(2));
                    if let Err(error) = mismatch {
                        assert!(error.contains("routing mismatch"));
                    }
                    session.reset().unwrap();
                    let cpu = Cpu {
                        c: checkpoint(root, s, f, rank, world),
                        x: vec![0.; 32],
                        state: [0.; 4],
                        token: 0,
                        trace: vec![],
                        logits: vec![],
                        fail: false,
                        routes: vec![],
                    };
                    let mut other = model.session(shared, cpu, 2, 16).unwrap();
                    session.reset().unwrap();
                    let mut interleaved = Vec::new();
                    for t in [1, 3, 2, 5, 6, 4] {
                        interleaved.push(session.step(t).unwrap());
                        other.step(2).unwrap();
                    }
                    assert_eq!(ids, interleaved);
                    assert_eq!(trace, session.kernels().trace);
                    (ids, trace)
                })
            })
            .collect::<Vec<_>>();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    })
}
#[test]
fn mixed_cross_world_placement_and_token_agreement() {
    let root: PathBuf = std::env::temp_dir().join(format!("next-tp-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let (shape, files) = fixture(&root);
    for (world, rank) in [2, 4].into_iter().flat_map(|w| (0..w).map(move |r| (w, r))) {
        let ep = ShardPlan::new(&shape, &files, rank, world)
            .unwrap()
            .with_ep2()
            .unwrap();
        assert!(ep.ep2 && ep.expert_tp == 1);
        for n in ["attn", "sg", "su", "sd"] {
            assert_eq!(
                ep.tensors[n].selection,
                engine_model::flash::plan::Selection::All
            );
        }
        for id in 0..4 {
            assert_eq!(
                ep.tensors[&format!("e{id}g")].placement == Placement::Resident,
                spark_tp::rows(4, rank, world).contains(&id)
            );
        }
        assert_eq!(ep.execution_plan().unwrap().query_heads, 0..24);
    }
    let reference = run(&root, &shape, &files, 1, false).pop().unwrap();
    for (world, path) in [(2, false), (4, false), (4, true)] {
        for (ids, trace) in run(&root, &shape, &files, world, path) {
            assert_eq!(ids, reference.0);
            let diff = trace
                .iter()
                .zip(&reference.1)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm = reference.1.iter().map(|a| a * a).sum::<f32>().sqrt();
            assert!(diff / norm < 1e-5, "world={world} normwise={}", diff / norm);
        }
    }
    for rank in 0..4 {
        let hybrid = ShardPlan::new(&shape, &files, rank, 4)
            .unwrap()
            .ep2tp2()
            .unwrap();
        let id = if rank < 2 { 0 } else { 2 };
        assert_eq!(hybrid.expert_tp, 2);
        assert_eq!(hybrid.tensors[&format!("e{id}g")].bytes(), 256);
        assert_eq!(hybrid.tensors[&format!("e{id}gf")].bytes(), 64);
        assert_eq!(
            hybrid.tensors[&format!("e{}g", (id + 2) % 4)].placement,
            Placement::Absent
        );
    }
    let mut draft = shape.clone();
    let mut df = files.clone();
    let mut entry = draft
        .tensors
        .iter()
        .find(|t| t.name == "e0g")
        .unwrap()
        .clone();
    entry.name = "draft".into();
    entry.role.place = Place::Draft { layer: Some(0) };
    draft.tensors.push(entry);
    df.insert("draft".into(), "model.safetensors".into());
    assert_eq!(
        ShardPlan::new(&draft, &df, 0, 4)
            .unwrap()
            .ep2tp2()
            .unwrap()
            .tensors["draft"]
            .placement,
        Placement::Absent
    );
    let c = checkpoint(&root, &shape, &files, 1, 2);
    let order = c.resident_order();
    assert!(order.windows(2).all(|w| w[0].1.file < w[1].1.file
        || w[0].1.file == w[1].1.file
            && c.source(w[0].0).unwrap().1 <= c.source(w[1].0).unwrap().1));
    let mut names = Vec::new();
    assert_eq!(
        c.stream_resident(37, |n, _, _| {
            names.push(n.to_owned());
            Ok(())
        })
        .unwrap(),
        c.plan.budget.resident
    );
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("e0") || n == "embed" || n.starts_with("ple"))
    );
    assert!(c.read_local("e0g", 0, &mut [0; 1]).is_err());
    let mut altered = shape.clone();
    altered.tensors[0].dtype = Dtype::F32;
    assert!(ShardCheckpoint::bind(&root, ShardPlan::new(&altered, &files, 0, 2).unwrap()).is_err());
    let mut wrong = files.clone();
    wrong.insert("embed".into(), "other.safetensors".into());
    assert!(ShardCheckpoint::bind(&root, ShardPlan::new(&shape, &wrong, 0, 2).unwrap()).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn serving_rejects_experimental_split_and_uses_checkpoint_name() {
    use spark_tp::next::{model_id, require_serving_split};
    for value in [None, Some("ep")] {
        assert!(require_serving_split(value).is_ok());
    }
    for value in ["ep2tp2", "", "EP", "unknown"] {
        assert!(require_serving_split(Some(value)).is_err());
    }
    for root in ["models/Qwen-Flash-FP4-Local", "archive/Qwen-Flash-FP4-Local/"] {
        assert_eq!(model_id(Path::new(root)).unwrap(), "qwen-flash-fp4-local");
    }
    assert!(model_id(Path::new("/")).is_err());
}
/// Chunk hooks over the CPU fake: each row keeps its own activation and runs the
/// single-token callbacks layer by layer in prompt order (shared recurrent state).
struct Chunked {
    cpu: Cpu,
    tokens: Vec<u32>,
    xs: Vec<Vec<f32>>,
    routes: Vec<Vec<(u32, f32)>>,
    /// Cohort rows' tokens (one per session), set by cohort_attention at layer 0.
    cohort: Vec<u32>,
}
impl Chunked {
    fn rows(&mut self, n: usize, mut f: impl FnMut(&mut Cpu, usize) -> R<()>) -> R<()> {
        let first = self.tokens.len().saturating_sub(n);
        for r in 0..n {
            self.cpu.x = self.xs[r].clone();
            self.cpu.token = if self.cohort.len() == n { self.cohort[r] } else { self.tokens[first + r] };
            f(&mut self.cpu, r)?;
            self.xs[r] = self.cpu.x.clone();
        }
        Ok(())
    }
}
impl RankKernels for Chunked {
    fn front(&mut self, c: &ShardCheckpoint, t: u32, p: usize, out: &mut [f32]) -> R<()> {
        self.tokens.push(t);
        self.cpu.front(c, t, p, out)
    }
    fn embed(&mut self, r: &[f32]) -> R<()> {
        self.cpu.embed(r)
    }
    fn attention(&mut self, l: usize, p: usize, out: &mut [f32]) -> R<()> {
        self.cpu.attention(l, p, out)
    }
    fn inject_attention(&mut self, l: usize, r: &[f32]) -> R<()> {
        self.cpu.inject_attention(l, r)
    }
    fn moe(&mut self, l: usize, out: &mut [f32]) -> R<()> {
        self.cpu.moe(l, out)
    }
    fn separate_shared(&self) -> bool {
        true
    }
    fn routing(&self) -> &[(u32, f32)] {
        self.cpu.routing()
    }
    fn inject_moe(&mut self, l: usize, r: &[f32]) -> R<()> {
        self.cpu.inject_moe(l, r)
    }
    fn logits(&mut self, v: Range<u64>, out: &mut [f32]) -> R<()> {
        self.cpu.logits(v, out)
    }
    fn reset(&mut self) -> R<()> {
        self.cpu.reset()
    }
    fn chunk_rows(&self) -> usize {
        4
    }
    fn embed_chunk(&mut self, rows: &[f32], n: usize) -> R<()> {
        self.xs.clear();
        for r in rows.as_chunks::<64>().0.iter().take(n) {
            self.cpu.embed(r)?;
            self.xs.push(self.cpu.x.clone());
        }
        self.routes = vec![Vec::new(); n];
        Ok(())
    }
    fn attention_chunk(&mut self, l: usize, out: &mut [f32], n: usize) -> R<()> {
        self.rows(n, |c, r| c.attention(l, 0, &mut out[r * 32..(r + 1) * 32]))
    }
    fn inject_attention_chunk(&mut self, l: usize, v: &[f32], n: usize) -> R<()> {
        self.rows(n, |c, r| c.inject_attention(l, &v[r * 32..(r + 1) * 32]))
    }
    fn moe_chunk(&mut self, l: usize, out: &mut [f32], n: usize) -> R<()> {
        let mut routes = vec![Vec::new(); n];
        self.rows(n, |c, r| {
            c.moe(l, &mut out[r * 64..(r + 1) * 64])?;
            routes[r] = c.routes.clone();
            Ok(())
        })?;
        self.routes = routes;
        Ok(())
    }
    fn chunk_routing(&self, row: usize) -> &[(u32, f32)] {
        &self.routes[row]
    }
    fn inject_moe_chunk(&mut self, l: usize, v: &[f32], n: usize) -> R<()> {
        self.rows(n, |c, r| c.inject_moe(l, &v[r * 32..(r + 1) * 32]))
    }
    fn finish_chunk(&mut self, n: usize) -> R<()> {
        self.cpu.x = self.xs[n - 1].clone();
        Ok(())
    }
    fn cohort_rows(&self) -> usize {
        4
    }
    fn cohort_attention(rows: &mut [&mut Self], l: usize, out: &mut [f32]) -> R<()> {
        if l == 0 {
            rows[0].cohort = rows.iter().map(|k| *k.tokens.last().unwrap()).collect();
        }
        for r in 0..rows.len() {
            let (x, t) = (rows[0].xs[r].clone(), rows[0].cohort[r]);
            rows[r].cpu.x = x;
            rows[r].cpu.token = t;
            rows[r].cpu.attention(l, 0, &mut out[r * 32..(r + 1) * 32])?;
            rows[0].xs[r] = rows[r].cpu.x.clone();
        }
        Ok(())
    }
    fn cohort_finish(rows: &mut [&mut Self]) -> R<()> {
        for r in 0..rows.len() {
            rows[r].cpu.x = rows[0].xs[r].clone();
        }
        rows[0].cohort.clear();
        Ok(())
    }
}
#[test]
fn prompt_chunks_match_token_steps() {
    let root: PathBuf = std::env::temp_dir().join(format!("next-tp-chunk-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let (s, f) = fixture(&root);
    for world in [1, 2] {
        let listeners = (0..world)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect::<Vec<_>>();
        let addresses = listeners.iter().map(|l| l.local_addr().unwrap()).collect::<Vec<_>>();
        drop(listeners);
        let (root, s, f) = (&root, &s, &f);
        std::thread::scope(|scope| {
            for rank in 0..world {
                let addresses = &addresses;
                scope.spawn(move || {
                    let model = NextModel::from_checkpoint(checkpoint(root, s, f, rank, world)).unwrap();
                    let cpu = || Cpu {
                        c: checkpoint(root, s, f, rank, world),
                        x: vec![0.; 32],
                        state: [0.; 4],
                        token: 0,
                        trace: vec![],
                        logits: vec![],
                        fail: false,
                        routes: vec![],
                    };
                    let ring = Ring::connect_options(rank, world, addresses[rank], addresses[(rank + 1) % world], false).unwrap();
                    let shared = SharedTransport::new(ring);
                    let mut plain = model.session(shared.clone(), cpu(), 1, 16).unwrap();
                    let ids = plain.prefill(&[1, 3, 2, 5, 6, 4]).unwrap();
                    let after = [3, 1, 2].map(|t| plain.step(t).unwrap());
                    let chunked = Chunked { cpu: cpu(), tokens: vec![], xs: vec![], routes: vec![], cohort: vec![] };
                    let mut c = model.session(shared.clone(), chunked, 2, 16).unwrap();
                    assert_eq!(c.chunk(&[1, 3, 2], false).unwrap(), 0);
                    assert_eq!(c.position(), 3);
                    assert_eq!(c.chunk(&[5, 6, 4], true).unwrap(), ids[5]);
                    assert_eq!(c.position(), 6);
                    assert_eq!([3, 1, 2].map(|t| c.step(t).unwrap()), after);
                    // Longer than the hooks' rows: token by token, same result.
                    let mut long = model.session(shared.clone(), Chunked { cpu: cpu(), tokens: vec![], xs: vec![], routes: vec![], cohort: vec![] }, 3, 16).unwrap();
                    assert_eq!(long.chunk(&[1, 3, 2, 5, 6, 4], true).unwrap(), ids[5]);
                    // Cohort rows: two sessions stepped together == each stepped alone.
                    let mut a = model.session(shared.clone(), cpu(), 10, 16).unwrap();
                    let mut b = model.session(shared.clone(), cpu(), 11, 16).unwrap();
                    let alone = [a.step(1).unwrap(), b.step(5).unwrap(), a.step(3).unwrap(), b.step(2).unwrap()];
                    let fresh = || Chunked { cpu: cpu(), tokens: vec![], xs: vec![], routes: vec![], cohort: vec![] };
                    let (mut ca, mut cb) = (model.session(shared.clone(), fresh(), 12, 16).unwrap(), model.session(shared.clone(), fresh(), 13, 16).unwrap());
                    let one = spark_tp::next::NextSession::step_rows(&mut [&mut ca, &mut cb], &[1, 5]).unwrap();
                    let two = spark_tp::next::NextSession::step_rows(&mut [&mut ca, &mut cb], &[3, 2]).unwrap();
                    assert_eq!([one[0], one[1], two[0], two[1]], alone);
                    assert_eq!(ca.step(4).unwrap(), a.step(4).unwrap());
                });
            }
        });
    }
    fs::remove_dir_all(&root).unwrap();
}
