//! Single-device, paired PTX benchmark.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
#[path = "kernel_cuda.rs"]
mod cuda;
#[cfg(target_os = "linux")] use cuda::fp8group;
#[cfg(target_os = "linux")] use cuda::rows;
#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
#[path = "kernel_rows.rs"] mod rows;
#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
#[path = "kernel_fp8.rs"] mod fp8group;
#[allow(dead_code)]
#[path = "../../export/src/sha256.rs"]
mod sha256;
#[path = "gdn_cpu_proof.rs"]
mod gdn_cpu_proof;
#[path = "gdn_rows_bounds.rs"]
mod gdn_rows_bounds;
#[allow(dead_code)]
#[path = "kernel_floor.rs"]
mod floor;
type Result<T> = std::result::Result<T, String>;
/// Case kinds (one registry for every kernel family): 0-6 general kernels (dense latency, head, hc_up,
/// fp4_bw gate/down, fp4_prefill_tc gate/down); 8-10 flash_bf16_mma_latency grouped/fused;
/// 20-21 fp4_bw routed C1/C8; 77 decode HC norm and activation (+ packed planes); 30-33 QSA attention scores/values/index scores, L2 gather;
/// 40-45 GDN seq/chunk/slot/fla0/split/cslot (kernel_gdn.rs and kernel_qsa.rs still see their
/// own 7.. numbering: the dispatch subtracts the family base); 75 FP8 ring BF16 decode rows.
#[derive(Clone, Debug, Default)]
struct Case {
    name: String,
    kernel: &'static str,
    m: usize,
    n: usize,
    k: usize,
    kind: u8,
    /// Prefill only: captured cold-8K routing (true) or uniform 40 rows/expert (false).
    real: bool,
    /// Launches per C1 decode token in production (nsys, 675 total); 0 = not a production row.
    calls: usize,
    /// kind 8 (grouped decode projections of one input): rows of up to four matrices.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    mats: [usize; 4],
    /// X-arm entry under --order (an order-changing twin behind a host switch).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    xkernel: &'static str,
}
/// Projections that share one decode input (rows; K), with C1 groups per token:
/// GDN in qkv/z/a/b (36), MoE router/shared-gate/gate/up (48), HC inject+down (96),
/// QSA pool qg/k/v/index (12).
const GROUPS: [(&str, [usize; 4], usize, usize); 4] = [
    ("gdnin", [10240, 6144, 48, 48], 2560, 36),
    ("moe", [512, 1, 640, 640], 2560, 48),
    ("hc", [4, 320, 0, 0], 10240, 96),
    ("qsa", [12288, 512, 512, 640], 2560, 12),
];
/// flash_bf16_mma_latency production decode shapes (N, K, launches per C1 token; sum 675),
/// from the pooled-rows callers (dense_latency.rs SHAPES, nsys 675 calls/token).
const LATENCY_SHAPES: [(usize, usize, usize); 12] = [
    (4, 10240, 96),
    (320, 10240, 97),
    (10240, 2560, 37),
    (6144, 2560, 36),
    (48, 2560, 72),
    (2560, 6144, 48),
    (12288, 2560, 12),
    (512, 2560, 72),
    (640, 2560, 108),
    (1, 2560, 48),
    (2560, 640, 48),
    (2560, 2560, 1),
];
/// flash_bf16_mma_prefill_tiled prompt shapes (M, N, K, launches per cold 8K prompt):
/// GDN qkv/z (36 layers), GDN + QSA out (48), QSA qg (12), router + QSA index/k (48 + 24),
/// shared gate/up + QSA v (96 + 12), shared down (48), HC down (96), PLE key/value (64-row tiles).
const PROMPT_TILED: [(usize, usize, usize, usize); 10] = [
    (2048, 10240, 2560, 144),
    (2048, 6144, 2560, 144),
    (2048, 2560, 6144, 192),
    (2048, 12288, 2560, 48),
    (2048, 512, 2560, 288),
    (2048, 640, 2560, 432),
    (2048, 2560, 640, 192),
    (2048, 320, 10240, 384),
    (64, 10240, 2560, 128),
    (64, 2560, 2560, 128),
];
/// flash_bf16_mma_prefill_pairs prompt launches (tag, pair rows, K, launches per cold 8K):
/// HC injection (2 per layer), GDN a/b after the tiled qkv/z, MoE shared gate after the
/// tiled router/gate/up.
const PROMPT_PAIRS: [(&str, [usize; 4], usize, usize); 3] = [
    ("hcinj", [4, 0, 0, 0], 10240, 384),
    ("gdnab", [0, 0, 48, 48], 2560, 144),
    ("sgate", [0, 1, 0, 0], 2560, 192),
];
struct Options {
    a: String,
    x: String,
    only: String,
    ms: u64,
    det: bool,
    /// Time an order-changing X whose bytes differ from A (bytes_equal false).
    allow_diff: bool,
    order: bool,
    /// --fp8-moe CONFIG: the FP8 W8A8 grouped MoE layer family at that checkpoint's shapes.
    fp8_moe: String,
}
fn cases() -> Vec<Case> {
    let mut out = fp8group::cases();
    out.extend(rows::cases());
    if std::env::var("KBENCH_FRONTEND").is_ok_and(|v|v=="1") {
        for m in [1,65,2048,8191,8192] {out.push(Case {name:format!("prompt-frontend-m{m}"),kernel:"flash_next_prompt_frontend",m,n:2560,k:0,kind:70,..Default::default()});}
    }
    for m in [1, 8, 16] {
        for (name, n, k) in [("q", 12288, 2560), ("o", 2560, 6144), ("gdn-in", 10240, 2560), ("gdn-out", 2560, 6144)] {
            out.push(Case {
                name: format!("dense_latency-{name}-m{m}"),
                kernel: if m <= 16 { "flash_bf16_mma_latency" } else { "flash_bf16_mma" },
                m,
                n,
                k,
                kind: 0,
                ..Default::default()
            });
        }
        out.push(Case { name: format!("hc_up-m{m}"), kernel: "flash_next_hc_up_latency", m, n: 10240, k: 320, kind: 2, ..Default::default() });
        // The production entry (one warp per stream, 128 threads); 97 per C1 token.
        out.push(Case { name: format!("hc_up4-m{m}"), kernel: "flash_next_hc_up_latency4", m, n: 10240, k: 320, kind: 2, calls: if m == 1 { 97 } else { 0 }, ..Default::default() });
    }
    // The M1 production entry when the up weight's LBF12 twin is built (97 per C1 token): real
    // = BF16 middles, every slot packed; -lbf12raw- leaves one slot in seven raw (BF16 loads).
    for raw in [false, true] {
        out.push(Case { name: format!("hc_up4-lbf12{}-m1", if raw { "raw" } else { "" }), kernel: "flash_next_hc_up_latency4_lbf12",
            m: 1, n: 10240, k: 320, kind: 2, real: true, calls: if raw { 0 } else { 97 }, ..Default::default() });
    }
    // Its M2-16 twin entry (drafted verify rows), same fixtures.
    for (m, raw) in [(2, false), (4, false), (8, false), (16, false), (4, true)] {
        out.push(Case { name: format!("hc_up16-lbf12{}-m{m}", if raw { "raw" } else { "" }), kernel: "flash_next_hc_up_latency16_lbf12",
            m, n: 10240, k: 320, kind: 2, real: true, ..Default::default() });
    }
    // The decode HC norm before each HC down (100 per C1 token at M1; drafted verify rows M4),
    // and with HC down's packed planes; the HC activation after it with HC up's planes. M3 and
    // M12 leave staged padding rows unwritten / have t + 8 padding for only some rows.
    for m in [1, 4] {
        out.push(Case { name: format!("hc_norm-m{m}"), kernel: "flash_hc_norm_f32", m, n: 0, k: 10240, kind: 77, calls: if m == 1 { 100 } else { 0 }, ..Default::default() });
    }
    for m in [1, 3, 4, 12] {
        out.push(Case { name: format!("hc_norm_pack-m{m}"), kernel: "flash_hc_norm_f32", m, n: 0, k: 10240, kind: 77, ..Default::default() });
        out.push(Case { name: format!("hc_act_pack-m{m}"), kernel: "flash_next_hc_activate", m, n: 0, k: 320, kind: 77, ..Default::default() });
    }
    // Verify rows: HC up at M 12 / 20 / 32 (production hc_up4 at every M <= 63).
    for m in [12, 20, 32] {
        out.push(Case { name: format!("hc_up4-m{m}"), kernel: "flash_next_hc_up_latency4", m, n: 10240, k: 320, kind: 2, ..Default::default() });
    }
    // Card headline row: the replay prompt's HC up at M 2048 (kind-2 sizes follow m).
    out.push(Case { name: "hc_up-m2048".into(), kernel: "flash_next_hc_up_latency", m: 2048, n: 10240, k: 320, kind: 2, ..Default::default() });
    out.push(Case { name: "flash_next_head".into(), kernel: "flash_next_head", m: 1, n: 248320, k: 2560, kind: 1, ..Default::default() });
    // Vocabulary traffic at verify widths, including the 16/17 boundary.
    for m in [5, 8, 12, 15, 16, 17, 20, 24, 32] {
        out.push(Case { name: format!("flash_next_head-m{m}"), kernel: "flash_next_head", m, n: 248320, k: 2560, kind: 1, ..Default::default() });
        out.push(Case { name: format!("flash_next_head-lbf12-m{m}"), kernel: "flash_next_head_lbf12", m, n: 248320, k: 2560, kind: 1, real: true, ..Default::default() });
    }
    for m in [1, 4, 8, 16] {
        for down in [false, true] {
            out.push(Case {
                name: format!("fp4_bw_{}-w{m}", if down { "down" } else { "gate" }),
                kernel: if down { "fp4_bw_down" } else { "fp4_bw_gate" },
                m,
                n: if down { 2560 } else { 640 },
                k: if down { 640 } else { 2560 },
                kind: if down { 4 } else { 3 },
                ..Default::default()
            });
        }
    }
    // Production decode shapes: C1 = one token's ten experts; C8 = a captured
    // eight-chat route (60 distinct experts, 80 assignments), task capacity and
    // grid exactly as next_execution.rs launches them; varied finite data.
    // C5 = five tokens of that route (five-chat workload); C2/C4 = the 2- and 4-row
    // decode buckets (MTP verify of one chat).
    for chats in [1, 2, 4, 5, 8] {
        for down in [false, true] {
            out.push(Case {
                name: format!("fp4_bw_{}-c{chats}", if down { "down" } else { "gate" }),
                kernel: if down { "fp4_bw_down" } else { "fp4_bw_gate" },
                m: chats,
                n: if down { 2560 } else { 640 },
                k: if down { 640 } else { 2560 },
                kind: if down { 21 } else { 20 },
                ..Default::default()
            });
        }
    }
    // Layer: gate then down per copy (down reads the gate's output); KBENCH_PDL=1
    // launches X's down with programmatic dependent launch.
    for chats in [1, 5, 8] {
        out.push(Case { name: format!("fp4_bw_layer-c{chats}"), kernel: "fp4_bw_down", m: chats, n: 2560, k: 640, kind: 22, ..Default::default() });
    }
    // Instruction-cache flush alone (KBENCH_ICFLUSH baseline; kernel_routed.rs).
    out.push(Case { name: "icflush".into(), kernel: "fp4_bw_gate", m: 1, n: 1, k: 1, kind: 26, ..Default::default() });
    // Decode route tables before the gate (one CTA); -bad = one malformed token.
    for tag in ["c1", "c5", "c8", "c8-bad"] {
        let m = if tag == "c1" { 1 } else if tag == "c5" { 5 } else { 8 };
        out.push(Case { name: format!("route-{tag}"), kernel: "flash_next_compact_route", m, n: 1, k: 1, kind: 23, ..Default::default() });
    }
    // The same route with flash_moe_assignments folded in (X = flash_next_compact_route_ids).
    for m in [1, 5, 8] {
        out.push(Case { name: format!("route_ids-c{m}"), kernel: "flash_next_compact_route", m, n: 1, k: 1, kind: 27, ..Default::default() });
    }
    // kind 29: the same route plus the shared expert's silu (A = flash_moe_silu + compact_route_ids, X = one launch).
    for m in [1, 5, 8] {
        out.push(Case { name: format!("route_silu-c{m}"), kernel: "flash_next_compact_route_ids", m, n: 1, k: 1, kind: 29, ..Default::default() });
    }
    // kind 28: one token's logits through A = flash_moe_route + flash_next_compact_route_silu vs X = one launch.
    out.push(Case { name: "route_full_silu-c1".into(), kernel: "flash_next_compact_route_silu", m: 1, n: 1, k: 1, kind: 28, ..Default::default() });
    // the pooled decode's route mask (A = flash_next_route_mask + the route launch, X = its _mask entry);
    // c5 = pooled width 8 with 5 live slots, c8 = 8 live.
    for (tag, m) in [("c1", 1), ("c5", 8), ("c8", 8)] {
        out.push(Case { name: format!("route_silu_mask-{tag}"), kernel: "flash_next_compact_route_silu", m, n: 1, k: 1, kind: 29, ..Default::default() });
    }
    out.push(Case { name: "route_full_silu_mask-c1".into(), kernel: "flash_next_compact_route_silu", m: 1, n: 1, k: 1, kind: 28, ..Default::default() });
    // the same plus the shared down's dense planes (A = the _mask path + flash_bf16_input, X = the _pack_mask entry).
    for (tag, m) in [("c1", 1), ("c5", 8), ("c8", 8)] {
        out.push(Case { name: format!("route_pack_mask-{tag}"), kernel: "flash_next_compact_route_silu_mask", m, n: 1, k: 1, kind: 29, ..Default::default() });
    }
    // verify / draft rows (no mask): A = flash_next_compact_route_silu + flash_bf16_input, X = flash_next_compact_route_pack.
    for (tag, m) in [("c1", 1), ("m4", 4), ("c8", 8)] {
        out.push(Case { name: format!("route_pack-{tag}"), kernel: "flash_next_compact_route_silu", m, n: 1, k: 1, kind: 29, ..Default::default() });
    }
    out.push(Case { name: "route_full_pack_mask-c1".into(), kernel: "flash_next_route_compact1_silu_mask", m: 1, n: 1, k: 1, kind: 28, ..Default::default() });
    // Plain C1 uses the same packed entry with an empty mask. A keeps the four-lane latency pack.
    out.push(Case { name: "route_full_pack_mask-c1-plain".into(), kernel: "flash_next_route_compact1_silu", m: 1, n: 1, k: 1, kind: 28, ..Default::default() });
    // A = flash_moe_route, X = flash_moe_route_fast (each exp as exp_split, same bits).
    // A = flash_moe_silu, X = flash_moe_silu_fast on m x 640 (a 2048-token prompt chunk; decode 8).
    for m in [8, 2048] {
        out.push(Case { name: format!("moe_silu_fast-m{m}"), kernel: "flash_moe_silu", m, n: 640, k: 1, kind: 24, ..Default::default() });
    }
    for m in [1, 8, 16] {
        out.push(Case { name: format!("moe_route_fast-c{m}"), kernel: "flash_moe_route", m, n: 512, k: 1, kind: 24, ..Default::default() });
    }
    for m in [1, 8] {
        out.push(Case { name: format!("moe_route-c{m}"), kernel: "flash_moe_route", m, n: 512, k: 1, kind: 24, ..Default::default() });
        out.push(Case { name: format!("finish-c{m}"), kernel: "flash_next_k16_finish", m, n: 2560, k: 1, kind: 25, ..Default::default() });
    }
    // Prompt cases: one 2048-token chunk on the production grid. "-8k" replays the
    // captured cold-8K expert counts (205 active experts); "-8k-uniform" spreads 40 rows
    // over all 512 experts (diverse-text worst case for weight streaming).
    for down in [false, true] {
        for real in [true, false] {
            out.push(Case {
                name: format!("fp4_prefill_tc_{}-8k{}", if down { "down" } else { "gate" }, if real { "" } else { "-uniform" }),
                kernel: if down { "fp4_prefill_tc_down" } else { "fp4_prefill_tc_gate" },
                m: 2048,
                n: if down { 2560 } else { 640 },
                k: if down { 640 } else { 2560 },
                kind: if down { 6 } else { 5 },
                real,
                ..Default::default()
            });
        }
    }
    // Production launch shape: N<=640 runs 32 threads plus the stream-ordered merge pass.
    // Verify-row shapes: M 5, 12, 15 as well; M 20/24/32 run
    // the production entry above 16 (flash_bf16_mma) through the law cases' kind-12 dispatch.
    for m in [1, 4, 5, 8, 12, 15, 16, 20, 24, 32] {
        for (n, k, calls) in LATENCY_SHAPES {
            out.push(Case {
                name: format!("lat-{n}x{k}-m{m}"),
                kernel: if m <= 16 { "flash_bf16_mma_latency" } else { "flash_bf16_mma" },
                m,
                n,
                k,
                kind: if m <= 16 { 0 } else { 12 },
                calls: if m == 1 { calls } else { 0 },
                mats: [0; 4],
                ..Default::default()
            });
        }
    }
    // The parked fused-FP32-input cases (latf-/grpf-, kind 9/10: production never launches
    // that mode, and their A and X arms run different kernels) are not in the case table.
    // A = main's solo launches per matrix (+merges); X = one flash_bf16_mma_latency_group launch.
    for m in [1, 4, 8, 16] {
        for (name, mats, k, calls) in GROUPS {
            out.push(Case {
                name: format!("grp-{name}-m{m}"),
                kernel: "flash_bf16_mma_latency_group",
                m,
                n: mats.iter().sum(),
                k,
                kind: 8,
                calls: if m == 1 { calls } else { 0 },
                mats,
                ..Default::default()
            });
        }
    }
    // QSA prompt attention at production shape (24 Q / 2 KV heads, head dimension 256,
    // 2048-row chunks, 512 groups + tail): M = tokens, N = past, K unused.
    // 32k = the last chunk of a 32K prompt; p0 = the first chunk (dense rows);
    // edge = rows crossing 2051/2052 with every tail length; m1/m16 = decode
    // and verify launch shapes (m4 = C1 MTP3 verify rows; m1s / m1k = one-token rows at 300 / 1500 tokens, the C1 plain gate's short contexts). Each case also re-scores rows as 1- and
    // 16-token launches and compares bytes (alone == chunked).
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("p0", 2048, 0), ("edge", 16, 2040), ("m16", 16, 30725), ("m4", 4, 31233), ("m1", 1, 31237), ("m1s", 1, 299), ("m1k", 1, 1499)] {
        out.push(Case { name: format!("qsa_attn_scores-{tag}"), kernel: "flash_qsa_attention_scores", m: tokens, n: past, k: 0, kind: 30, xkernel: "flash_qsa_attention_scores_tc", ..Default::default() });
    }
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("p0", 2048, 0), ("edge", 16, 2040), ("m16", 16, 30725), ("m4", 4, 31233), ("m1", 1, 31237), ("m1s", 1, 299), ("m1k", 1, 1499)] {
        out.push(Case { name: format!("qsa_attn_values-{tag}"), kernel: "flash_qsa_attention_values", m: tokens, n: past, k: 0, kind: 31, xkernel: "flash_qsa_attention_values_tc", ..Default::default() });
    }
    // Softmax (main in place vs the pairs-only softmax_md of the raw-score TC values).
    // The alternative X entry is an external benchmark input, not a shipped kernel.
    // Known issue: A runs in place on the scores buffer X also reads, so long runs drift X's
    // input ("rotating copies differ (arm 1)" at --ms 3000); give X its own scores copy first.
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("p0", 2048, 0), ("m1", 1, 31237), ("m4", 4, 31233), ("m16", 16, 30725), ("m1s", 1, 299), ("m1k", 1, 1499)] {
        out.push(Case { name: format!("qsa_attn_softmax-{tag}"), kernel: "flash_qsa_attention_softmax", m: tokens, n: past, k: 0, kind: 34, xkernel: "flash_qsa_attention_softmax_md", ..Default::default() });
    }
    if std::env::var("KBENCH_QSAPROD").is_ok_and(|v|v=="1") {
        for (tag,tokens,past) in [("whole8k",8192,0),("tail8191",8191,0),("smoke64",64,0),("tail65",65,0),("whole32k",32768,0)] {
            out.push(Case { name:format!("qsa_prod_on_fused-{tag}"),kernel:"flash_qsa_attention_fused_tc",
                m:tokens,n:past,k:0,kind:35,calls:if tokens==8192 {12} else {0},..Default::default() });
        }
    }
    // Default-path scores + softmax (kind 36 = kernel_qsa.rs 13, --order): A = main's scores then
    // softmax (X PTX), X = flash_qsa_attention_scores_ts; output = P rows +
    // denominators, byte for byte.
    // Production prompt entries in BOTH arms (A = base PTX, X = candidate PTX, same
    // entry and grid): A == X is a byte law, us is the production kernel's.
    // off = switch-off prompt path (scores_t; values_wide on main c37630c,
    // values_wide4t), on = VADE_QSA_TC prompt path (fused_tc).
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("p0", 2048, 0)] {
        out.push(Case { name: format!("qsa_prod_off_scores-{tag}"), kernel: "flash_qsa_attention_scores_t", m: tokens, n: past, k: 0, kind: 30, xkernel: "flash_qsa_attention_scores_t", ..Default::default() });
        out.push(Case { name: format!("qsa_prod_off_values-{tag}"), kernel: "flash_qsa_attention_values_wide", m: tokens, n: past, k: 0, kind: 31, xkernel: "flash_qsa_attention_values_wide", ..Default::default() });
        out.push(Case { name: format!("qsa_prod_off_values4t-{tag}"), kernel: "flash_qsa_attention_values_wide4t", m: tokens, n: past, k: 0, kind: 31, xkernel: "flash_qsa_attention_values_wide4t", ..Default::default() });
        out.push(Case { name: format!("qsa_prod_on_fused-{tag}"), kernel: "flash_qsa_attention_fused_tc", m: tokens, n: past, k: 0, kind: 35, xkernel: "flash_qsa_attention_fused_tc", ..Default::default() });
    }
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("p0", 2048, 0), ("t65", 65, 0), ("e64", 64, 2000)] {
        out.push(Case { name: format!("qsa_attn_ssm-{tag}"), kernel: "flash_qsa_attention_scores", m: tokens, n: past, k: 0, kind: 36, xkernel: "flash_qsa_attention_scores_ts", ..Default::default() });
    }
    // Fused prompt attention (kind 35 = kernel_qsa.rs 12, --order): A = the X-on chain from the X PTX
    // (scores _tc_w, main's softmax, values _tc), X = flash_qsa_attention_fused_tc;
    // the values outputs must match byte for byte. t65 = a 65-token prompt,
    // e64 = 64 rows crossing the 2051/2052 slot edge.
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("p0", 2048, 0), ("t65", 65, 0), ("e64", 64, 2000)] {
        out.push(Case { name: format!("qsa_attn_fused-{tag}"), kernel: "flash_qsa_attention_values", m: tokens, n: past, k: 0, kind: 35, xkernel: "flash_qsa_attention_fused_tc", ..Default::default() });
    }
    // Real-selection cells (KBENCH_QSA_IDS from a long8k-prose prefill, chunk past 12288).
    out.push(Case { name: "qsa_attn_fused-real".into(), kernel: "flash_qsa_attention_values", m: 2048, n: 12288, k: 0, kind: 35, xkernel: "flash_qsa_attention_fused_tc", ..Default::default() });
    out.push(Case { name: "qsa_attn_scores-real".into(), kernel: "flash_qsa_attention_scores", m: 2048, n: 12288, k: 0, kind: 30, xkernel: "flash_qsa_attention_scores_tc", ..Default::default() });
    out.push(Case { name: "qsa_attn_values-real".into(), kernel: "flash_qsa_attention_values", m: 2048, n: 12288, k: 0, kind: 31, xkernel: "flash_qsa_attention_values_tc", ..Default::default() });
    for (tag, tokens, past) in [("32k", 2048, 30720), ("8k", 2048, 6144), ("edge", 16, 2040), ("m16", 16, 30725), ("m4", 4, 31233), ("m1", 1, 31237)] {
        out.push(Case { name: format!("qsa_index_scores-{tag}"), kernel: "flash_qsa_scores", m: tokens, n: past, k: 0, kind: 32, xkernel: "flash_qsa_scores_tc", ..Default::default() });
    }
    // Index top-512 selection at decode shapes (kind 37, in place: the first launch
    // turns a row's first 512 scores into IDs, later replays select from those
    // bytes; A and X see the same sequence while their outputs agree).
    // chunk = a 2048-row prompt chunk at past 30720 (the prompt path selects per row too).
    for (tag, tokens, past) in [("32k-m1", 1, 32767), ("128k-m1", 1, 131071), ("256k-m1", 1, 262143), ("128k-m8", 8, 131064), ("32k-chunk", 2048, 30720)] {
        out.push(Case { name: format!("qsa_select-{tag}"), kernel: "flash_qsa_select", m: tokens, n: past, k: 0, kind: 37, xkernel: "flash_qsa_select_c", ..Default::default() });
    }
    // Pooled C8 decode stages (kind 38 = kernel_qsa.rs 15): 8 sessions (session s at past - s),
    // one launch per stage at attend_pool's grids; K = stage. A = the switch-off entry on
    // main c03c112d, X (--order) = the TC default entry; same entry in both arms otherwise.
    for (tag, past) in [("c8-8k", 8191), ("c8-32k", 32767)] {
        for (stage, k, a, x) in [("scores", 0, "flash_qsa_attention_scores_batch", "flash_qsa_attention_scores_tc_batch"),
            ("values", 1, "flash_qsa_attention_values_batch", "flash_qsa_attention_values_tc_m_batch"),
            ("index", 2, "flash_qsa_scores_batch", "flash_qsa_scores_tc_batch"), ("select", 3, "flash_qsa_select_batch", "flash_qsa_select_batch"),
            ("softmax", 4, "flash_qsa_attention_softmax_batch", "flash_qsa_attention_softmax_batch")] {
            out.push(Case { name: format!("qsa_batch_{stage}-{tag}"), kernel: a, m: 8, n: past, k, kind: 38, xkernel: x, ..Default::default() });
        }
    }
    // Pooled decode prepare (stage 5): Q norm + rope, K norm + rope + append, V append, C1 and C8.
    for (tag, m) in [("m1", 1), ("c8-32k", 8)] {
        out.push(Case { name: format!("qsa_batch_prepare-{tag}"), kernel: "flash_qsa_prepare_batch", m, n: 32767, k: 5, kind: 38, xkernel: "flash_qsa_prepare_batch", ..Default::default() });
    }
    // GB10 L2 random-gather ceiling: the QSA selection pattern, 1 KB FP32 K rows,
    // kv head interleaved (k=0, today's grids) or the slow grid index (k=1).
    for (tag, past) in [("32k", 30720), ("8k", 6144)] {
        for (order, slow) in [("inter", 0), ("slow", 1)] {
            out.push(Case { name: format!("l2_gather-{tag}-{order}"), kernel: "kb_gather", m: 2048, n: past, k: slow, kind: 33, xkernel: "kb_gather", ..Default::default() });
        }
    }
    if std::env::var("KBENCH_ROUTE_PREFILL").is_ok_and(|v|v=="1") {
        for(m,tag)in [(1,"m1"),(64,"m64"),(65,"m65"),(1023,"m1023"),(1024,"m1024"),(1025,"m1025"),(8191,"m8191"),(8192,"8k"),(8192,"8k-bad"),(8192,"8k-hot"),(8193,"m8193"),(16384,"m16384"),(32768,"m32768")] {
            out.push(Case{name:format!("proute-{tag}"),kernel:"flash_next_compact_route_prefill",m,n:512,k:10,kind:23,calls:48,..Default::default()});
        }
    }
    // GDN (owner gdn_scan): m = tokens per piece, n = pieces, k = rows (chats).
    for (pieces, tag) in [(1usize, "2k"), (4, "8k")] {
        out.push(Case { name: format!("gdn_seq-{tag}"), kernel: "flash_next_gdn_scan", m: 2048, n: pieces, k: 1, kind: 40, ..Default::default() });
        out.push(Case { name: format!("gdn_chunk-{tag}"), kernel: "flash_next_gdn_chunk", m: 2048, n: pieces, k: 1, kind: 41, ..Default::default() });
    }
    out.push(Case { name: "gdn_fla0-2k".into(), kernel: "flash_gdn_chunk_scan", m: 2048, n: 1, k: 1, kind: 43, ..Default::default() });
    out.push(Case { name: "gdn_slot-c8".into(), kernel: "flash_next_gdn_scan_slot", m: 1, n: 1, k: 8, kind: 42, ..Default::default() });
    out.push(Case { name: "gdn_split-263".into(), kernel: "flash_next_gdn_chunk", m: 263, n: 1, k: 1, kind: 44, ..Default::default() });
    out.push(Case { name: "gdn_cslot-c8".into(), kernel: "flash_next_gdn_chunk_slot", m: 1, n: 1, k: 8, kind: 45, ..Default::default() });
    // C1 decode step: A = main's slot kernel, X = the chunk slot
    // kernel with the open chunk holding 4 (default), 0, 32, 62 rows, or 63
    // rows (the step that closes the chunk); n = 2 + open rows when n > 1.
    out.push(Case { name: "gdn_cslot-c1".into(), kernel: "flash_next_gdn_chunk_slot", m: 1, n: 1, k: 1, kind: 45, ..Default::default() });
    for open in [0usize, 32, 62, 63] {
        out.push(Case { name: format!("gdn_cslot-c1-o{open}"), kernel: "flash_next_gdn_chunk_slot", m: 1, n: 2 + open, k: 1, kind: 45, ..Default::default() });
    }
    // The decode step with its output gate: slot + gate_slot (A) vs the slot applying
    // the gate itself (X); bytes A == X.
    // The whole decode step: prepare_hist_slot + the chunk slot with its gate; X's slot
    // a programmatic launch under VADE_KBENCH_GDN_PDL=1; bytes A == X.
    for chats in [1usize, 8] {
        out.push(Case { name: format!("gdn_cstep-c{chats}"), kernel: "flash_next_gdn_chunk_slot", m: 1, n: 1, k: chats, kind: 53, ..Default::default() });
    }
    for chats in [1usize, 8] {
        out.push(Case { name: format!("gdn_cslotg-c{chats}"), kernel: "flash_next_gdn_chunk_slot", m: 1, n: 1, k: chats, kind: 52, ..Default::default() });
    }
    out.push(Case { name: "gdn_cslot-c8-o63".into(), kernel: "flash_next_gdn_chunk_slot", m: 1, n: 65, k: 8, kind: 45, ..Default::default() });
    // One GDN layer step (kind 54): main's prepare + rows + history + gate (A) vs the
    // rows kernel with its fused prepare and gate + history (X); m = rows, n = open chunk
    // rows, k = key heads (8: the tensor split's rank, 16: one Spark); o0 starts a chunk,
    // o63 and r8-o56 close it, r4-o62 / r8-o60 cross it.
    for (rows, open, keys) in [(1usize, 4usize, 8usize), (1, 31, 8), (1, 63, 8), (4, 30, 8), (4, 62, 8), (8, 60, 8), (1, 31, 16), (4, 62, 16),
        (1, 0, 8), (2, 31, 8), (3, 31, 8), (8, 56, 8),
        // One Spark at the served verify window sizes (1 to 8 rows).
        (1, 4, 16), (2, 31, 16), (3, 31, 16), (3, 61, 16), (4, 31, 16), (6, 31, 16), (8, 31, 16), (8, 60, 16)] {
        out.push(Case { name: format!("gdn_fstep-k{keys}-r{rows}-o{open}"), kernel: "flash_next_gdn_chunk_rows", m: rows, n: open, k: keys, kind: 54, ..Default::default() });
    }
    // One cohort verify GDN layer step (kind 55): main's cohort prepare + rows + history (A) vs
    // the cohort rows kernel with its fused prepare + history (X), then one gate over every
    // row; m = rows per chat, n = open chunk rows, k = chats (one Spark, 16 key heads).
    for (rows, open, chats) in [(2usize, 31usize, 4usize), (4, 31, 4), (8, 31, 4), (4, 61, 4), (2, 31, 8), (4, 31, 8), (8, 31, 8), (4, 61, 8)] {
        out.push(Case { name: format!("gdn_cfstep-c{chats}-r{rows}-o{open}"), kernel: "flash_next_gdn_chunk_rows_cohort", m: rows, n: open, k: chats, kind: 55, ..Default::default() });
    }
    // Small execute-path calls: an MTP verify window (4 rows) and a one-row
    // replay, chunk prep + scan vs main's sequential scan over the same rows.
    for rows in [4usize, 1] {
        out.push(Case { name: format!("gdn_chunk-r{rows}"), kernel: "flash_next_gdn_chunk", m: rows, n: 1, k: 1, kind: 41, ..Default::default() });
    }
    // C1/C8 decode prepare: prepare_slot + history_slot
    // vs the fused prepare (non-order, bytes A == X).
    for rows in [1usize, 8] {
        out.push(Case { name: format!("gdn_prep-c{rows}"), kernel: "flash_next_gdn_prepare_hist_slot", m: 1, n: 1, k: rows, kind: 46, ..Default::default() });
        out.push(Case { name: format!("gdn_gate-c{rows}"), kernel: "flash_next_gdn_gate_slot", m: 1, n: 1, k: rows, kind: 47, ..Default::default() });
    }
    // Prompt prepare, one 2,048-token piece: main vs q/k once per key group.
    out.push(Case { name: "gdn_cprep-2k".into(), kernel: "flash_gdn_conv_prepare_tokens_kh", m: 2048, n: 1, k: 1, kind: 48, ..Default::default() });
    // Output gate fused into the chunk scan vs the scan + flash_gdn_gate (both
    // from the X module; non-order: gated + state bytes equal).
    for (pieces, tag) in [(1usize, "2k"), (4, "8k")] {
        out.push(Case { name: format!("gdn_chunkg-{tag}"), kernel: "flash_next_gdn_chunk_scan", m: 2048, n: pieces, k: 1, kind: 49, ..Default::default() });
    }
    // Chunk-form prompt pipeline on real conv inputs: prepare + chunk
    // prep + gated scan, v through memory (A) vs convolved in the scan (X).
    for (pieces, tag) in [(1usize, "2k"), (4, "8k")] {
        out.push(Case { name: format!("gdn_cpipe-{tag}"), kernel: "flash_next_gdn_chunk_scan", m: 2048, n: pieces, k: 1, kind: 50, ..Default::default() });
    }
    // The fused pipeline writing the out-projection's packed BF16 input itself (X)
    // vs the FP32 gated output + flash_bf16_input (A): same packed words (A1).
    for (pieces, tag) in [(1usize, "2k"), (4, "8k")] {
        out.push(Case { name: format!("gdn_cpipeq-{tag}"), kernel: "flash_next_gdn_chunk_scan_cv", m: 2048, n: pieces, k: 1, kind: 51, ..Default::default() });
    }
    // The pipeline on the same 8K rows in 1,024 / 512-row host pieces (prep's
    // q/k/P/kk of a piece then fit L2 for the scan; cuts give the same bytes).
    // The production pipeline on the same 8K rows in 4,096 / 8,192-row calls (a
    // 4,096-row prompt tile with GDN pieces of 4,096 rows; same bytes).
    for piece in [4096usize, 8192] {
        out.push(Case { name: format!("gdn_cpipeq-8k-p{piece}"), kernel: "flash_next_gdn_chunk_scan_cv", m: piece, n: 8192 / piece, k: 1, kind: 51, ..Default::default() });
    }
    for piece in [1024usize, 512] {
        out.push(Case { name: format!("gdn_cpipe-8k-p{piece}"), kernel: "flash_next_gdn_chunk_scan", m: piece, n: 8192 / piece, k: 1, kind: 50, ..Default::default() });
    }
    // The same 8K rows in host pieces of 1,024 / 512 rows (the A/P
    // workspace of a piece fits L2; cuts give the same bytes as one call).
    for piece in [1024usize, 512] {
        out.push(Case { name: format!("gdn_chunk-8k-p{piece}"), kernel: "flash_next_gdn_chunk", m: piece, n: 8192 / piece, k: 1, kind: 41, ..Default::default() });
        out.push(Case { name: format!("gdn_chunkg-8k-p{piece}"), kernel: "flash_next_gdn_chunk_scan", m: piece, n: 8192 / piece, k: 1, kind: 49, ..Default::default() });
    }
    // Law check (16:15 note): the MoE router projection (N 512, K 2560) runs the latency
    // kernel at M <= 16, flash_bf16_mma at 17..63 and flash_bf16_mma_prefill_tiled at
    // M >= 64 (production packs and launch shapes); rows shared by every M must match.
    for m in [1, 16, 17, 63, 64, 65, 128] {
        out.push(Case {
            name: format!("law-router-m{m}"),
            kernel: if m <= 16 {
                "flash_bf16_mma_latency"
            } else if m < 64 {
                "flash_bf16_mma"
            } else {
                "flash_bf16_mma_prefill_tiled"
            },
            m,
            n: 512,
            k: 2560,
            kind: 12,
            calls: 0,
            mats: [0; 4],
            ..Default::default()
        });
    }
    // Law across the 16/17 boundary for the small-group shapes (HC [4, 320] x 10240 and the
    // MoE group's 640 and 1 x 2560): latency kernel at M <= 16, flash_bf16_mma at 17..32 (the earlier variant
    // runs these groups at M17-32 as two latency launches); rows shared by every M must match.
    for (tag, n, k) in [("hc320", 320, 10240), ("hc4", 4, 10240), ("moe640", 640, 2560), ("gate1", 1, 2560)] {
        for m in [4, 16, 17, 20, 32] {
            out.push(Case {
                name: format!("law-{tag}-m{m}"),
                kernel: if m <= 16 { "flash_bf16_mma_latency" } else { "flash_bf16_mma" },
                m,
                n,
                k,
                kind: 12,
                calls: 0,
                mats: [0; 4],
                ..Default::default()
            });
        }
    }
    // Standalone credited sixteen-output-row: exact expansion oracle at every verify width.
    for m in [2, 3, 4] {
        for (n, k, _) in LATENCY_SHAPES {
            if n % 64 == 0 && n > 640 && k % 256 == 0 && k <= 6144 {
                out.push(Case { name: format!("mx-solo97-{n}x{k}-m{m}"), kernel: "flash_verify_mx_solo97", m, n, k, kind: 11, ..Default::default() });
            }
        }
    }
    // MXFP8 groups (kind 13): the production groups with their production-MXFP8 matrices
    // (group_mx) as twins; A = the group launch on the exact BF16 expansion. Bytes equal.
    // M5/M6 = verify rows at resident MTP depth 4/5 (KBENCH_MXROWS_X: A = main's entry).
    for m in [1, 2, 4, 5, 6, 8, 12, 16] {
        for (name, mats, k, calls) in GROUPS {
            out.push(Case {
                name: format!("mxg-{name}-m{m}"),
                kernel: "flash_bf16_mma_latency_group",
                m,
                n: mats.iter().sum(),
                k,
                kind: 13,
                calls: if m == 1 { calls } else { 0 },
                mats,
                ..Default::default()
            });
        }
    }
    // prefill_gemm owner: production prompt projections (engine flash/blocks.rs next_prefill).
    // kind 60 = flash_bf16_mma_prefill_tiled on the production pack of a raw FP32 activation
    // (packed once, outside the timed graph); kind 61 = flash_bf16_mma_prefill_pairs on the raw
    // FP32 activation, mats = rows of pairs[0].{rows,rows2}, pairs[1].{rows,rows2}. calls =
    // launches per cold 8K prompt (four 2,048-row chunks + 64-row PLE tiles; sums 2,080 tiled /
    // 720 pairs vs nsys main-profile-kernels 2,081 / 720). Edge cases (calls 0) check the
    // exactness law: rows shared by every M must match (63/64/65, 127/128/129, M16 tails).
    // Floor max(bytes/230 GB/s, 4MNK/90 TFLOPS): the hi and lo chains are two MMA streams.
    for (m, n, k, calls) in PROMPT_TILED {
        out.push(Case { name: format!("pg-{n}x{k}-m{m}"), kernel: "flash_bf16_mma_prefill_tiled", m, n, k, kind: 60, calls, ..Default::default() });
    }
    // Actual 8192-row production shapes, nonzero BF16 weights and FP32 hi/lo operands.
    if std::env::var("KBENCH_PGPROD").is_ok_and(|v| v == "1") {
        for m in [64, 65, 2048, 8191, 8192] {
            for (n, k, n2, calls) in [(10240,2560,48,36),(6144,2560,48,36),
                (2560,6144,0,48),(12288,2560,0,12),(640,2560,1,48),
                (512,2560,0,72),(640,2560,0,60),(2560,640,0,48)] {
                out.push(Case { name: format!("pgprod-{n}x{k}+{n2}-m{m}"),
                    kernel: if std::env::var("KBENCH_PGPROD_BASE_U32").is_ok_and(|v|v=="1") {"flash_bf16_mma_prefill_tiled_u32"} else {"flash_bf16_mma_prefill_tiled"}, m, n:n+n2, k, kind:60,
                    mats:[n,n2,0,0], calls:if m==8192 {calls} else {0}, ..Default::default() });
            }
        }
    }
    // 4096-row prompt tiles (VADE_PROMPT_TILE default): rows shared with M 2048 must match.
    for (m, n, k, _) in PROMPT_TILED {
        if m == 2048 {
            out.push(Case { name: format!("pg-{n}x{k}-m4096"), kernel: "flash_bf16_mma_prefill_tiled", m: 4096, n, k, kind: 60, ..Default::default() });
        }
    }
    for m in [64, 65, 127, 128, 129, 2047, 2049] {
        out.push(Case { name: format!("pg-512x2560-m{m}"), kernel: "flash_bf16_mma_prefill_tiled", m, n: 512, k: 2560, kind: 60, ..Default::default() });
    }
    // PLE key/value projections on 256-row slices (four 64-row tiles per weight pass).
    for (n, k) in [(10240, 2560), (2560, 2560)] {
        out.push(Case { name: format!("pg-{n}x{k}-m256"), kernel: "flash_bf16_mma_prefill_tiled", m: 256, n, k, kind: 60, ..Default::default() });
    }
    for (tag, mats, k, calls) in PROMPT_PAIRS {
        for m in [2048, 64, 65, 2047] {
            out.push(Case {
                name: if m == 2048 { format!("pp-{tag}") } else { format!("pp-{tag}-m{m}") },
                kernel: "flash_bf16_mma_prefill_pairs",
                m,
                n: mats.iter().sum(),
                k,
                kind: 61,
                calls: if m == 2048 { calls } else { 0 },
                mats,
                ..Default::default()
            });
        }
    }
    // Card replay-shape names: the fixtures of pg-320x10240-m2048 and pp-hcinj (replay pairs
    // [{M 2048, N 4, K 10240}, {M 2048, N 0, K 10240}]); calls 0 keeps the per-prompt sums.
    out.push(Case { name: "tiled-320x10240-m2048".into(), kernel: "flash_bf16_mma_prefill_tiled", m: 2048, n: 320, k: 10240, kind: 60, ..Default::default() });
    out.push(Case { name: "pairs-2048".into(), kernel: "flash_bf16_mma_prefill_pairs", m: 2048, n: 4, k: 10240, kind: 61, mats: [4, 0, 0, 0], ..Default::default() });
    // kind 62 (fold), mats = [big1, small1, big2, small2]: A = main's tiled launches of the big
    // matrices + main's pairs launch of the small ones (FP32 input); X = one tiled launch per
    // big matrix carrying its small matrix as the second (HC: injection in the down's N tail;
    // GDN: a after qkv, b after z as extra N tiles; MoE: the row-major shared gate after the
    // shared up). Outputs share one [M][sum] row; bytes A==X.
    for (tag, mats, k, calls) in [("hc", [320, 4, 0, 0], 10240, 384), ("gdn", [10240, 48, 6144, 48], 2560, 144), ("moe", [640, 1, 0, 0], 2560, 192)] {
        for m in [2048, 64, 65, 2047, 4096] {
            out.push(Case {
                name: if m == 2048 { format!("pgf-{tag}") } else { format!("pgf-{tag}-m{m}") },
                kernel: "flash_bf16_mma_prefill_tiled",
                m,
                n: mats.iter().sum(),
                k,
                kind: 62,
                calls: if m == 2048 { calls } else { 0 },
                mats,
                ..Default::default()
            });
        }
    }
    // kind 64: the HC prompt block, norm -> (pack) -> down + injection (main) vs the norm
    // writing the packed planes and one folded tiled launch; u and projection bytes A==X.
    for m in [2048, 64, 65, 2047, 4096] {
        out.push(Case {
            name: if m == 2048 { "pgn-hc".into() } else { format!("pgn-hc-m{m}") },
            kernel: "flash_bf16_mma_prefill_tiled",
            m,
            n: 324,
            k: 10240,
            kind: 64,
            calls: if m == 2048 { 384 } else { 0 },
            mats: [320, 4, 0, 0],
            ..Default::default()
        });
    }
    // kind 68: prompt HC down, main's tiled launch + flash_next_hc_activate + pack of low vs the
    // tiled launch writing the activated low's planes; planes and injection bytes A==X.
    for m in [2048, 64, 2047, 4096] {
        out.push(Case {
            name: if m == 2048 { "pgd-hc".into() } else { format!("pgd-hc-m{m}") },
            kernel: "flash_bf16_mma_prefill_tiled",
            m,
            n: 324,
            k: 10240,
            kind: 68,
            calls: if m == 2048 { 384 } else { 0 },
            mats: [320, 4, 0, 0],
            ..Default::default()
        });
    }
    // kind 65: prompt HC up, main's flash_next_hc_up_latency4 vs the tiled kernel of the same
    // K16 graph and stream-ordered mix (bytes A==X); 4 per layer per 2,048-row chunk.
    for m in [2048, 64, 65, 2047, 4096] {
        out.push(Case {
            name: if m == 2048 { "pgu-hc".into() } else { format!("pgu-hc-m{m}") },
            kernel: "flash_next_hc_up_tiled",
            m,
            n: 2560,
            k: 320,
            kind: 65,
            calls: if m == 2048 { 384 } else { 0 },
            ..Default::default()
        });
    }
    if std::env::var("KBENCH_HCUPPROD").is_ok_and(|v|v=="1") {
        for m in [64,65,2048,8191,8192,32768] {
            out.push(Case { name:format!("hc_up_prod-m{m}"),kernel:"flash_next_hc_up_tiled",
                m,n:2560,k:320,kind:65,calls:if m==8192 {97}else{0},..Default::default() });
        }
    }
    // kind 66: prompt HC combine + next mix norm (+ projection planes), main's three launches
    // vs flash_next_hc_add_norm; residual, u (and planes at M % 16 == 0) bytes A==X.
    for m in [2048, 64, 65, 2047, 4096] {
        out.push(Case {
            name: if m == 2048 { "pga-hc".into() } else { format!("pga-hc-m{m}") },
            kernel: "flash_next_hc_add_norm",
            m,
            n: 10240,
            k: 10240,
            kind: 66,
            calls: if m == 2048 { 376 } else { 0 },
            ..Default::default()
        });
    }
    // kind 74 (pgr-: prompt GEMM on the residual rows): prompt HC norm + HC down (low 320 +
    // injection 4), main's norm planes through memory vs the scales-only norm and
    // flash_bf16_mma_prefill_tiled_hc with norm != 0; all bytes A==X.
    for m in [2048, 64, 4096] {
        out.push(Case {
            name: if m == 2048 { "pgr-hc".into() } else { format!("pgr-hc-m{m}") },
            kernel: "flash_bf16_mma_prefill_tiled_hc",
            m,
            n: 324,
            k: 10240,
            kind: 74,
            mats: [320, 4, 0, 0],
            ..Default::default()
        });
    }
    // Production HC entry in both arms; includes identical residual reset overhead.
    // Opt-in diagnostic cases avoid changing the historical emulation cases.
    if std::env::var("KBENCH_HCNORM_PRODUCTION").is_ok_and(|v| v == "1") {
        for m in [64, 65, 2048, 8191, 8192] {
            for finish in [false, true] {
                out.push(Case {
                    name: format!("hcnorm-prod-{}-m{m}", if finish { "fast" } else { "add" }),
                    kernel: if finish { "flash_next_hc_finish_add_norm_fast" } else { "flash_next_hc_add_norm" },
                    m, n: 10240, k: 10240, kind: if finish { 67 } else { 66 },
                    calls: if m == 8192 { if finish { 46 } else { 48 } } else { 0 },
                    ..Default::default()
                });
            }
        }
    }
    // fp4_bw: pgk-hc-fast = pgk-hc with X = flash_next_hc_finish_add_norm_fast.
    out.push(Case { name: "pgk-hc-fast".into(), kernel: "flash_next_hc_finish_add_norm", m: 2048, n: 10240, k: 10240, kind: 67, ..Default::default() });
    // kind 67: kind 66 with the MoE output formed in the fused kernel (main: k16_finish first).
    for m in [2048, 64, 65, 2047, 4096] {
        out.push(Case {
            name: if m == 2048 { "pgk-hc".into() } else { format!("pgk-hc-m{m}") },
            kernel: "flash_next_hc_finish_add_norm",
            m,
            n: 10240,
            k: 10240,
            kind: 67,
            calls: if m == 2048 { 184 } else { 0 },
            ..Default::default()
        });
    }
    // kind 69: flash_bf16_input alone (32 lanes, the prompt pack) on kind 60's raw FP32 [M][K]
    // activation, each arm its own module's pack; both BF16 planes bytes A==X.
    for k in [2560, 6144, 10240] {
        for m in [2048, 64, 65, 2047, 4096] {
            out.push(Case { name: format!("pk-{k}-m{m}"), kernel: "flash_bf16_input", m, n: 64, k, kind: 69, ..Default::default() });
        }
    }
    // kind 63: BF16 m16n8k16 mma.sync issue rate (registers only, 8 independent chains per
    // warp); flops / median_us is the MMA ceiling the hi + lo chains share (floor 90 TFLOPS).
    for ctas in [48, 96, 192] {
        out.push(Case { name: format!("mma-peak-c{ctas}"), kernel: "kb_mma", m: ctas, n: 4096, k: 8, kind: 63, ..Default::default() });
    }
    // MXFP8 read-as-stored (kind 11): A = flash_bf16_mma_latency on the exact BF16
    // expansion (expand_to_bf16), X = flash_mx8_mma_latency on the E4M3 bytes + UE8M0
    // scales; packed shapes only (N % 4 == 0). Bytes must match.
    // M 20/24/32 (wide-eligible shapes): arm A = the BF16 latency launch on token tiles
    // 0-15 and 16..M.
    for m in [1, 2, 4, 5, 8, 12, 15, 16, 20, 24, 32] {
        for (n, k, calls) in LATENCY_SHAPES {
            if n % 4 == 0 && (m <= 16 || n > 640 && k <= 6144 && k % 256 == 0) {
                out.push(Case {
                    name: format!("mx-{n}x{k}-m{m}"),
                    // Matches kernel_params::latency_mx8_widen with N > 640 (the engine's 128-thread launches, _r)
                    kernel: if n > 640 && k <= 6144 && k % 256 == 0 && ((9..=32).contains(&m) || (5..=8).contains(&m) && k > 2560) { "flash_mx8_mma_latency_r" } else { "flash_mx8_mma_latency" },
                    m,
                    n,
                    k,
                    kind: 11,
                    calls: if m == 1 { calls } else { 0 },
                    mats: [0; 4],
                    ..Default::default()
                });
            }
        }
    }
    // mx-r-: the wide-eligible MXFP8 cases with N % 64 == 0 and X =
    // flash_mx8_mma_latency_r (16-row warp tasks); mx- runs _r too instead of _w;
    // the mx- prefix keeps the latency fixture at M > 16.
    for m in [5, 8, 12, 15, 16, 20, 24, 32] {
        for (n, k, _) in LATENCY_SHAPES {
            if n > 640 && n % 64 == 0 && k <= 6144 && k % 256 == 0 && (m >= 9 || k > 2560) {
                out.push(Case { name: format!("mx-r-{n}x{k}-m{m}"), kernel: "flash_mx8_mma_latency_r", m, n, k, kind: 11, calls: 0, mats: [0; 4], ..Default::default() });
            }
        }
    }
    // Full-prompt probe: one call over a whole 8K prompt (captured counts x4) vs four 2048-row
    // calls; "-8k-half": one call over 4096 rows (counts x2), the VADE_PROMPT_TILE=4096 shape.
    for (suffix, m) in [("whole", 8192), ("half", 4096)] { for down in [false, true] {
        out.push(Case {
            name: format!("fp4_prefill_tc_{}-8k-{suffix}", if down { "down" } else { "gate" }),
            kernel: if down { "fp4_prefill_tc_down" } else { "fp4_prefill_tc_gate" },
            m,
            n: if down { 2560 } else { 640 },
            k: if down { 640 } else { 2560 },
            kind: if down { 6 } else { 5 },
            real: true,
            ..Default::default()
        });
    } }
    // Uniform 160 rows x 512 experts over a whole 8K prompt: a diverse-text
    // W4A16 MoE tile shape.
    for down in [false, true] {
        out.push(Case {
            name: format!("fp4_prefill_tc_{}-8k-uniform-whole", if down { "down" } else { "gate" }),
            kernel: if down { "fp4_prefill_tc_down" } else { "fp4_prefill_tc_gate" },
            m: 8192,
            n: if down { 2560 } else { 640 },
            k: if down { 640 } else { 2560 },
            kind: if down { 6 } else { 5 },
            real: false,
            ..Default::default()
        });
    }
    // Benchmark-only activation probe: X = the A4 gate (NVFP4 activations on
    // mma.m16n8k64 mxf4nvf4, --fp4 PTX) under --order; values differ by design.
    for (suffix, m) in [("8k", 2048), ("8k-half", 4096)] {
        out.push(Case {
            name: format!("fp4_prefill_tc-a4probe-gate-{suffix}"),
            kernel: "fp4_prefill_tc_gate",
            xkernel: "fp4_prefill_tc_a4probe_gate",
            m,
            n: 640,
            k: 2560,
            kind: 5,
            real: true,
            ..Default::default()
        });
    }
    // Tile-boundary counts (1, 7-9, 15-17, ... 255-257 rows per expert): bytes only
    // matter here, the kernels' row tiling must not change any row's output.
    for down in [false, true] {
        out.push(Case {
            name: format!("fp4_prefill_tc_{}-8k-edges", if down { "down" } else { "gate" }),
            kernel: if down { "fp4_prefill_tc_down" } else { "fp4_prefill_tc_gate" },
            m: 2048,
            n: if down { 2560 } else { 640 },
            k: if down { 640 } else { 2560 },
            kind: if down { 6 } else { 5 },
            real: false,
            ..Default::default()
        });
    }
    out
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[(s.len() - 1) / 2] + s[s.len() / 2]) * 0.5
}
#[cfg(target_os = "linux")]
fn grade(p: f64) -> &'static str {
    if p >= 95. {
        "A+"
    } else if p >= 85. {
        "A"
    } else if p >= 65. {
        "B"
    } else if p >= 40. {
        "C"
    } else if p >= 15. {
        "D"
    } else {
        "F"
    }
}
fn run() -> Result<()> {
    let mut o = Options {
        a: String::new(),
        x: String::new(),
        only: "all".into(),
        ms: 5000,
        det: false,
        allow_diff: false,
        order: false,
        fp8_moe: String::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--list" {
            for c in cases() {
                println!("{} {} M={} N={} K={} calls/token={}", c.name, c.kernel, c.m, c.n, c.k, c.calls);
            }
            return Ok(());
        }
        if flag == "--allow-diff" {
            o.allow_diff = true;
            continue;
        }
        if flag == "--gdn-cpu-proof" {
            let (_, failures) = gdn_cpu_proof::run();
            return if failures == 0 { Ok(()) } else { Err(format!("gdn cpu proof: {failures} mismatches")) };
        }
        if flag == "--gdn-rows-bounds" {
            let (_, failures) = gdn_rows_bounds::run();
            return if failures == 0 { Ok(()) } else { Err(format!("gdn rows bounds: {failures} failures")) };
        }
        if flag == "--self-test" {
            if sha256::digest(b"abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
                return Err("SHA self-test".into());
            }
            assert_eq!(median(&[3., 1., 2.]), 2.);
            // 26 general cases + each kernel family's cases (fp4_prefill_tc 2 + 2 (85f85b5 -8k-whole) + 2 (5e93cd8 -8k-half) + 2 (c42ec41 -8k-edges), flash_bf16_mma_latency 64 (ed97b6f: latf-/grpf- 64
            // out of the table), e2a8617 law/mx 51 + 11 (a583a9f mx-*-m2), fp4_bw 4 + 6 (c2/c4/c5) + 3 (layer) + 4 (route) + 2 (moe_route) + 2 (finish) + 1 (icflush), flash_qsa 18 + 17 (a79a5ab softmax 4,
            // ssm 5, fused 5, index edge/m16/m1 3), gdn_scan 8 + 21 (d8f87da cslot-c1 5, cslot-c8-o63, chunk-r 2, prep/gate 4,
            // cprep, chunkg 2, cpipe 2, 8k pieces 4; its kinds 13..17 -> 46..50)).
            let expect = 26
                + 2 // a4probe gate 8k / 8k-half
                + 2 // -8k-uniform-whole gate / down
                + 2
                + 2
                + 2
                + 2
                + 64
                + 22
                + 35
                + 29
                + 62
                // MXFP8 groups and their twins.
                + 21
                // prefill_gemm: pg- 19, pp- 12, pgf- 12, mma-peak 3, pgn- 4 (kinds 60..64).
                + 31
                + 12
                + 3
                + 4
                // pgu- 4 (kind 65), pga- 4 (kind 66), pgk- 4 (kind 67)
                + 4
                + 4
                + 4
                // flash_qsa: 13 cases the count missed at 9aaacac (--self-test 336 vs 323), then
                // qsa_batch_* 18 (C8 pooled stages), m4 scores / values / index 3, select 32k-chunk 1, softmax m4 / m16 2, short-context m1s / m1k scores / values / softmax 6.
                + 13
                + 30
                // prefill_gemm on main df70005f: tiled-320x10240-m2048, pairs-2048, pgd-hc x3 (kind 68).
                + 5
                // main 44d69aff 7558c8f4: hc_up-m2048.
                + 1
                // flash_qsa: qsa_batch_prepare-m1 / -c8-32k.
                + 2
                // Existing main110 + BW24 listed cases; BF member absent, fixtures unchanged.
                + 110 + 24
                // pk- 12 (kind 69)
                + 12
                // 6 cases main added without the count (384 vs 378 at 44d69aff), then
                // gdn_cpipeq-8k-p4096 / -p8192.
                + 6
                + 2
                // gdn_cslotg-c1 / -c8.
                + 2
                // gdn_cstep-c1 / -c8.
                + 2
                // M 4096 prompt tiles: pg- 8, pgf- 3, pgn/pgd/pgu/pga/pgk 1 each, pk- 3
                + 19
                // BF16/LBF12 head widths: 18.
                + 18
                // law-hc320/hc4/moe640/gate1 at M4/16/17/20/32: 20
                + 20
                // mx- at M20/24/32 for the wide-eligible shapes: 15
                + 15
                // mx-r-: 5 shapes x M12/15/16/20/24/32 + 2560x6144 at M5/8: 32
                + 32
                // Sixteen-row verify entry: five qualifying shapes x M2/3/4.
                + 15
                // mxg- at M5/M6: 4 groups x 2.
                + 8
                + 39 // FP8 grouped decode and prompt gate fixtures.
                + 1 // Plain C1 route and shared-input pack.
                + 3 // pgr-hc M2048 / M64 / M4096 (kind 74).
                + 20 // gdn_fstep (kind 54).
                + 8 // gdn_cfstep (kind 55).
                + 74 // FP8 stack kernel_rows.rs: kind 75/76 decode rows + fp8a wave probe.
                + 7 // fp8k-hcgroup-m2..m8 (kind 75, kernel_rows.rs).
                + 2 // hc_up4-lbf12-m1 / -lbf12raw-m1.
                + 5 // hc_up16-lbf12-m2/4/8/16, -lbf12raw-m4.
                + 10 // hc_norm-m1 / -m4, hc_norm_pack- and hc_act_pack-m1 / -m3 / -m4 / -m12 (kind 77).
                ;
            assert_eq!(cases().len(), expect);
            // --case selects by name: every name names one case.
            assert_eq!(cases().iter().map(|c| c.name.as_str()).collect::<std::collections::BTreeSet<_>>().len(), expect);
            println!("PASS host SHA/statistics/shape self-test; GPU A==A requires --a PTX --x SAME_PTX");
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        if flag == "--dump-gather-ptx" {
            print!("{}", cuda::gather_ptx());
            return Ok(());
        }
        if flag == "--order" {
            // Order-changing X: run each case's xkernel; A==X bytes are not
            // required; X must repeat bit-exactly and match itself across
            // launch shapes; FP64 error stats are printed for score cases.
            o.order = true;
            continue;
        }
        if flag == "--help" {
            println!(
                concat!(
                    "kernel-bench --a A.ptx --x X.ptx [--case all|NAME,...] [--ms 5000]",
                    " [--det 1]",
                    " [--allow-diff]\n       kernel-bench --a A.ptx --x X.ptx --fp8-moe CONFIG.json [--ms 5000] [--allow-diff]",
                    "\n",
                    "--det 1: order-change mode, A/X bytes may differ (each arm's SHA is printed; compare X across processes).\n",
                    "{} real shapes",
                    " (grp-NAME-mM = grouped projections: A solo launches, X one group launch; lat-NxK-mM = flash_bf16_mma_latency production decode rows, merge included for N<=640)",
                    "; paired modules in one context; synthetic finite fixtures; JSONL stdout.\nWeights rotate across >=64 MiB per case. CUDA-event and globaltimer graph brackets exclude setup/hash.\n--ms is minimum measured GPU milliseconds per arm/case; default full try about five minutes.\nM16 dense uses product flash_bf16_mma dispatch; latency entry supports only M<=8.\n",
                    "Prefill: production grid, four 2048-token chunks; -8k = captured cold-8K expert counts, -8k-uniform = 40 rows x 512 experts.\n",
                    "GDN: gdn_seq-* = main's prompt scan both arms (bytes A==X); gdn_chunk-*/gdn_fla0-2k = order change on X (SHA per arm + error vs A, determinism = same X SHA across processes); gdn_slot-c8 = decode slot, 8 chats; gdn_split-263 = chunk-form boundary law (cuts at 63/64/65 etc., one-row calls and decode-slot rows must equal one call, bytes); gdn_cslot-c8 = chunk-form decode step, 8 chats (X) vs main slot (A).\n",
                    "--fp8-moe: the W8A8 grouped MoE layer at CONFIG's shapes instead of the cases (KBENCH_FP8MOE_M, _RING, _X, _XROUTE, _XENTRIES, _A; see the README). Plain-text lines, not JSONL; layer time = globaltimer interior per graph replay over the route sets, stage times between stamps; there --ms is wall-clock milliseconds of replays per arm.\n",
                    "Floors: stored compulsory bytes/230GB/s; prompt max(bytes/230GB/s,2MNK/90TFLOPS).\nGraph brackets include two timestamp nodes; globaltimer measures their interior. No profiler speed claims."
                ),
                cases().len()
            );
            return Ok(());
        }
        let v = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--a" => o.a = v,
            "--x" => o.x = v,
            "--case" => o.only = v,
            "--ms" => o.ms = v.parse().map_err(|_| "invalid --ms")?,
            "--det" => o.det = v == "1",
            "--fp8-moe" => o.fp8_moe = v,
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if o.a.is_empty() || o.x.is_empty() || !(10..=60000).contains(&o.ms) {
        return Err("require --a/--x PTX and --ms 10..60000".into());
    }
    let mut selected = cases();
    if o.only != "all" {
        for name in o.only.split(',') {
            if !selected.iter().any(|c| c.name == name) {
                return Err(format!("unknown case {name}"));
            }
        }
        selected.retain(|c| o.only.split(',').any(|n| n == c.name));
    }
    #[cfg(target_os = "linux")]
    {
        if !o.fp8_moe.is_empty() { return cuda::fp8_moe(&o, &o.fp8_moe); }
        cuda::run(&o, &selected)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = selected;
        Err("CUDA execution requires Linux; --list/--self-test are portable".into())
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("kernel-bench: {e}");
        std::process::exit(1);
    }
}
/// KBENCH_LAT_FOLD=1: arm X launches BF16 latency cases with latency_fold1 (M1, N <= 64) as
/// the engine does: flash_bf16_mma_latency_fold, one 128-thread launch.
#[cfg(target_os = "linux")]
fn fold_x(c: &Case) -> bool {
    std::env::var("KBENCH_LAT_FOLD").is_ok_and(|v| v == "1") && c.kind == 0 && c.kernel == "flash_bf16_mma_latency"
        && c.m == 1 && c.k % 256 == 0 && c.k <= 10240 && c.n % 4 == 0 && c.n <= 64 // = kernel_params::latency_fold1
}
