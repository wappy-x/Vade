//! Small local socket check; exact Python/Rust replay evidence stays outside Git.
use super::{Config, cohort_metrics, run_cohort, run_prefill, stream, value::*};
#[test]
#[allow(clippy::excessive_precision)] // literal constants stay as written.
fn fp8_host_contract() {
    use crate::fp8_reference::*;
    for code in 0..=255 { let v = decode_e4m3(code); if !v.is_nan() { assert_eq!(encode_e4m3(v), code); } }
    assert_eq!((decode_e4m3(126), decode_e4m3(1)), (448.0, 1.0 / 512.0));
    assert_eq!((encode_e4m3(1.0625), encode_e4m3(1.1875)), (56, 58));
    assert_eq!((encode_bf16(f32::from_bits(0x3f80_8000)), widen_bf16(0x3f80)), (0x3f80, 1.0));
    let q = quantize(&vec![448.0; 256], 256).unwrap();
    let w = vec![56; 128 * 256];
    let full = block_gemm(&q, &w, &[0x3f80; 2], 128, 0..2).unwrap();
    let batch = quantize(&[vec![448.0; 256], vec![0.0; 256]].concat(), 256).unwrap();
    assert_eq!(block_gemm(&batch, &w, &[0x3f80; 2], 128, 0..2).unwrap(), [full.clone(), vec![0.0; 128]].concat());
    let (a, b) = (block_gemm(&q, &w, &[0x3f80; 2], 128, 0..1).unwrap(), block_gemm(&q, &w, &[0x3f80; 2], 128, 1..2).unwrap());
    assert_eq!(full, fold_ranks(&[&a, &b]).unwrap());
    assert_eq!(full, block_gemm_scalar(&q, &w, &[0x3f80; 2], 128, 0..2).unwrap());
    assert_eq!(canonical_k16_ranges(80).unwrap(), [0..1, 1..2, 2..3, 3..5]);
    let mut x = vec![0.0; 80]; x[0] = 16777216.0; x[16] = 1.0; x[32] = -16777216.0; x[48] = 2.0;
    assert_eq!(bf16_gemm(&x, &[0x3f80; 80], 80, 1).unwrap(), [2.0]);
    assert_eq!(bf16_gemm(&[1.001953125; 16], &[0x3f80; 16], 16, 1).unwrap(), [16.03125]);
    assert!(bf16_gemm(&[f32::NAN; 16], &[0x3f80; 16], 16, 1).is_err());
    let ranks = [&[2.0f32][..]];
    assert_eq!(ordered_finish(&[Expert { id: 3, weight: 0.5, ranks: &ranks }], &[&[4.0]], 0.5).unwrap(), [3.0]);
}
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

fn serve(mut socket: TcpStream, model: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(socket.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let mut length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line.is_empty() {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    if request.starts_with("GET ") {
        let health = encode(&o([
            ("status", s("ready")),
            ("model", s(model)),
            ("build_hash", s("stub-build")),
            ("checkpoint_hash", s("stub-fp8")),
            ("mtp_depth", n(0)),
            ("device_bytes", n(4096)),
        ]));
        write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{health}",
            health.len()
        )?;
        return Ok(());
    }
    let mut raw = vec![0; length];
    reader.read_exact(&mut raw)?;
    let body = parse(&raw).map_err(std::io::Error::other)?;
    let text = body
        .get("messages")
        .arr()
        .first()
        .map(|v| v.get("content").str().unwrap_or(""))
        .unwrap_or("");
    let explicit_sampling = match stream::sampling() {
        Value::Object(items) => items.iter().all(|(key, value)| body.get(key) == value),
        _ => false,
    };
    let seedless = matches!(&body, Value::Object(fields) if fields.iter().all(|(key, _)| key != "seed"));
    let valid = explicit_sampling && seedless
        && body.get("cache") == &b(false)
        && body.get("draft").str() == Some("plain");
    if !valid || text == "http-error" {
        socket.write_all(
            b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?;
        return Ok(());
    }
    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
    let mut chunks = vec![
        o([(
            "choices",
            a(vec![o([("delta", o([("role", s("assistant"))]))])]),
        )]),
        o([(
            "choices",
            a(vec![o([("delta", o([("reasoning_content", s("é"))]))])]),
        )]),
        o([(
            "choices",
            a(vec![o([(
                "delta",
                o([("content", s("two chunks are not token counts 😀"))]),
            )])]),
        )]),
        o([(
            "choices",
            a(vec![o([("delta", o([])), ("finish_reason", s("length"))])]),
        )]),
    ];
    if text != "missing-usage" {
        let tokens = body.get("max_tokens").uint().unwrap_or(0) - u64::from(text == "short");
        chunks.push(o([
            ("choices", a(vec![])),
            (
                "usage",
                o([
                    ("prompt_tokens", n(if text == "8k" { 8192 } else { 450 })),
                    ("completion_tokens", n(tokens)),
                    (
                        "prompt_tokens_details",
                        o([("cached_tokens", n(u64::from(text == "cache")))]),
                    ),
                ]),
            ),
        ]));
    }
    if text != "fallback" {
        chunks.last_mut().unwrap().put(
            "vade",
            o([
                ("sampling", stream::sampling()),
                ("mtp_depth", n(u64::from(text == "bad-mtp"))),
                ("draft", s("plain")),
            ]),
        );
    }
    let mut events = ": heartbeat\r\n\r\n".to_owned();
    for chunk in chunks {
        events.push_str(&format!("data: {}\r\n\r\n", encode(&chunk)));
    }
    events.push_str("data: [DONE]\r\n\r\n");
    thread::sleep(Duration::from_millis(10));
    for fragment in events.as_bytes().chunks(7) {
        write!(socket, "{:x}\r\n", fragment.len())?;
        socket.write_all(fragment)?;
        socket.write_all(b"\r\n")?;
    }
    socket.write_all(b"0\r\n\r\n")?;
    socket.flush()
}

pub(super) fn self_test(args: &Config) -> Result<(), String> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let model = args.model.clone();
    let worker = thread::spawn(move || {
        for socket in listener.incoming() {
            if stopping.load(Ordering::Relaxed) {
                break;
            }
            let model = model.clone();
            if let Ok(socket) = socket {
                thread::spawn(move || {
                    let _ = serve(socket, &model);
                });
            }
        }
    });
    let mut args = args.clone();
    args.api_key = None;
    args.draft = Some("plain".into());
    args.checkpoint_hash = Some("stub-fp8".into());
    args.observe_cmd.clear();
    let url = format!("http://{address}/v1");
    let prompt = |text| {
        o([
            ("messages", a(vec![o([("role", s("user")), ("content", s(text))])])),
            ("seed", n(42)),
        ])
    };
    let check = |condition: bool, message: &str| {
        if condition {
            Ok(())
        } else {
            Err(format!("self-test: {message}"))
        }
    };
    let result: Result<(), String> = (|| {
        super::support::validate_health(&stream::get_health(&url, args.timeout)?, &args, true)?;
        for count in [1, 8, 16] {
            let cell = run_cohort(&url, &[prompt("echo")], count, 1, &args, None, "single")?;
            check(
                cell.get("eligible").yes() && cell.get("ok").uint() == Some(count as u64),
                "concurrent complete cohort",
            )?;
            check(
                cell.get("completion_tokens").uint() == Some((count * args.max_tokens) as u64),
                "server token counts",
            )?;
            for record in cell.get("requests").arr() {
                check(
                    record.get("sampling_source").str() == Some("server_echo")
                        && record.get("ttft_ms").num().is_some_and(|n| n >= 8.),
                    "fragmented UTF-8/chunked SSE",
                )?;
            }
        }
        for text in ["missing-usage", "bad-mtp", "short", "cache", "http-error"] {
            let cell = run_cohort(
                &url,
                &[prompt("echo"), prompt(text)],
                2,
                1,
                &args,
                None,
                "single",
            )?;
            check(
                !cell.get("eligible").yes() && cell.get("ok").uint() == Some(1),
                "failed cohort eligibility",
            )?;
            check(
                cell.get("aggregate_tokens_per_s") == &Value::Null
                    && cell
                        .get("requests")
                        .arr()
                        .iter()
                        .all(|r| r.get("tokens_per_s") == &Value::Null),
                "failed cohort publishes no rates",
            )?;
        }
        let record = stream::stream_chat(
            &url,
            stream::request_body(&prompt("fallback"), &args, 256),
            &args,
            None,
        );
        check(
            record.get("ok").yes() && record.get("sampling_source").str() == Some("request"),
            "missing echo fallback",
        )?;
        let probe = run_prefill(
            &url,
            &prompt("8k"),
            &o([("synthetic", b(true))]),
            &args,
            None,
            "single",
        );
        check(
            probe.get("eligible").yes() && probe.get("runs").arr().len() == 3,
            "three pinned cold 8k probes",
        )?;
        let fixed = vec![
            o([
                ("send_clock_s", n(10)),
                ("complete_clock_s", n(12)),
                ("usage", o([("completion_tokens", n(256))])),
                ("ttft_ms", n(100)),
                ("ok", b(true)),
            ]),
            o([
                ("send_clock_s", n("10.1")),
                ("complete_clock_s", n(14)),
                ("usage", o([("completion_tokens", n(256))])),
                ("ttft_ms", n(300)),
                ("ok", b(true)),
            ]),
        ];
        let metric = cohort_metrics(&fixed);
        check(
            metric.get("aggregate_tokens_per_s").num() == Some(128.)
                && metric.get("ttft_p50_ms").num() == Some(200.)
                && metric.get("ttft_p95_ms").num() == Some(290.),
            "exact cohort arithmetic",
        )?;
        let events = b"data: {\"choices\":[{\"delta\":{\"reasoning\":\"thinking\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":450,\"completion_tokens\":256}}\n\ndata: [DONE]\n\n";
        let replay = stream::replay_chat(
            stream::request_body(&prompt("echo"), &args, 256),
            &args,
            events,
            200,
            &[("Content-Type", "text/event-stream")],
            &[10., 10.1, 10.25, 11., 12., 14.],
        );
        let writing = 255. * 1000. / (4000. - (10.1 - 10.) * 1000.);
        check(
            replay.get("ok").yes()
                && replay
                    .get("ttft_ms")
                    .num()
                    .is_some_and(|v| (v - 100.).abs() < 1e-9)
                && replay.get("first_content_ms").num() == Some(250.)
                && replay.get("writing_tokens_per_s").num() == Some(writing),
            "fixed-clock legacy writing after reasoning TTFT; answer rate unknown",
        )?;
        Ok(())
    })();
    stop.store(true, Ordering::Relaxed);
    let _ = TcpStream::connect(address);
    let _ = worker.join();
    result?;
    crate::write_output(format_args!(
        "SELF-TEST PASS: local chunked/fragmented UTF-8 SSE; usage counts; c1/c8/c16; failed-cell null rates; sampling echo/fallback; three pinned 8k probes. Synthetic device_bytes=4096; no GPU."
    ), true)?;
    crate::write_output(format_args!(
        "SELF-TEST arithmetic: 512 tokens / 4.000 s = 128.000 tok/s; TTFT p50=200.000 ms p95=290.000 ms."
    ), true)?;
    Ok(())
}

#[cfg(test)]
mod release_defaults_tests {
    use super::*;
    use std::{net::SocketAddr, sync::Mutex, thread::JoinHandle};

    struct Server {
        address: SocketAddr,
        stop: Arc<AtomicBool>,
        requests: Arc<Mutex<Vec<Value>>>,
        worker: Option<JoinHandle<Result<(), String>>>,
    }

    impl Server {
        fn start(model: Option<&'static str>, drift: bool, forced_echo: Option<&'static str>) -> Self {
            Self::start_with_depth(model, 3, drift, forced_echo)
        }

        fn start_with_depth(model: Option<&'static str>, server_depth: usize,
            drift: bool, forced_echo: Option<&'static str>,
        ) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let address = listener.local_addr().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let stopping = stop.clone();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let captured = requests.clone();
            let worker = thread::spawn(move || {
                let mut health_count = 0;
                for socket in listener.incoming() {
                    if stopping.load(Ordering::Relaxed) { break; }
                    serve_release(socket.map_err(|e| e.to_string())?, model, server_depth, drift,
                        forced_echo, &mut health_count, &captured).map_err(|e| e.to_string())?;
                }
                Ok(())
            });
            Self { address, stop, requests, worker: Some(worker) }
        }

        fn args(&self, extra: &[&str]) -> Config {
            let mut flags = vec!["--url".into(), format!("http://{}/v1", self.address)];
            flags.extend(extra.iter().map(|flag| (*flag).to_owned()));
            Config::parse(&flags).unwrap()
        }

        fn finish(mut self) -> Vec<Value> {
            self.stop.store(true, Ordering::Relaxed);
            let _ = TcpStream::connect(self.address);
            self.worker.take().unwrap().join().unwrap().unwrap();
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            let _ = TcpStream::connect(self.address);
            if let Some(worker) = self.worker.take() { let _ = worker.join(); }
        }
    }

    fn serve_release(mut socket: TcpStream, model: Option<&str>, server_depth: usize, drift: bool,
        forced_echo: Option<&str>, health_count: &mut usize, requests: &Mutex<Vec<Value>>,
    ) -> std::io::Result<()> {
        socket.set_read_timeout(Some(Duration::from_secs(5)))?;
        socket.set_write_timeout(Some(Duration::from_secs(5)))?;
        let mut reader = BufReader::new(socket.try_clone()?);
        let mut request = String::new();
        reader.read_line(&mut request)?;
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line == "\r\n" || line.is_empty() { break; }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse().unwrap_or(0);
            }
        }
        if request.starts_with("GET /health ") {
            *health_count += 1;
            let depth = if drift && *health_count >= 3 { server_depth.saturating_sub(1) } else { server_depth };
            let mut health = o([
                ("status", s("ready")),
                ("vade", o([
                    ("build", o([("git_sha", s("stub-release")), ("dirty", b(false))])),
                    ("checkpoint", o([("sha256", s("stub-checkpoint"))])),
                    ("numerics", o([("mtp_depth", n(depth))])),
                ])),
            ]);
            if let Some(model) = model { health.put("model", s(model)); }
            let encoded = encode(&health);
            return write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{encoded}", encoded.len());
        }
        let mut raw = vec![0; length];
        reader.read_exact(&mut raw)?;
        let body = parse(&raw).map_err(std::io::Error::other)?;
        requests.lock().unwrap().push(body.clone());
        let mode = forced_echo.unwrap_or_else(|| body.get("draft").str().unwrap_or("strict"));
        let plain = mode == "plain";
        let chunk = o([
            ("model", body.get("model").clone()),
            ("choices", a(vec![o([
                ("delta", o([("content", s("Synthetic output; counts come from usage."))])),
                ("finish_reason", s("length")),
            ])])),
            ("usage", o([
                ("prompt_tokens", n(if body.get("prompt").str() == Some("8k probe") { 8192 } else { 450 })),
                ("completion_tokens", body.get("max_tokens").clone()),
            ])),
            ("vade", o([
                ("sampling", stream::sampling()),
                ("draft", s(mode)),
                ("mtp_depth", n(if plain { 0 } else { 3 })),
                ("draft_used", b(!plain)),
                ("model_draft_used", b(!plain)),
                // Release metadata retains the configured depth in plain replies.
                ("numerics", o([("mtp_depth", n(server_depth))])),
            ])),
        ]);
        let events = format!("data: {}\n\ndata: [DONE]\n\n", encode(&chunk));
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{events}", events.len())
    }

    fn assert_header(receipt: &Value, model: &str, mode: &str, drafting: &str, depth: usize) {
        assert_eq!(receipt.get("model"), &s(model));
        assert_eq!(receipt.path("decoding.request_mode"), &s(mode));
        assert_eq!(receipt.path("decoding.drafting"), &s(drafting));
        assert_eq!(receipt.path("decoding.mtp_depth"), &Value::Null);
        assert_eq!(receipt.path("decoding.server_mtp_depth"), &n(depth));
        assert_eq!(receipt.path("decoding.server_mtp_depth_label"), &s(super::super::drafting::SERVER_DEPTH_LABEL));
        assert_eq!(receipt.path("decoding.server_mtp_depth_field"), &s("vade.numerics.mtp_depth"));
        assert_eq!(receipt.get("server_build_sha"), &s("stub-release"));
        assert_eq!(receipt.path("decoding.server_draft_mode"), &Value::Null);
    }

    #[test]
    fn a_glm_request_asks_low_effort_and_never_thinking_off() {
        let template = o([("messages", a(vec![o([("role", s("user")), ("content", s("hi"))])])),
            ("enable_thinking", b(false)), ("chat_template_kwargs", o([("enable_thinking", b(false))]))]);
        let mut args = Config::parse(&["--url".into(), "http://127.0.0.1:9/v1".into()]).unwrap();
        args.model = "glm-5.3-flash-fp4-local".into();
        let glm = stream::request_body(&template, &args, 8);
        assert_eq!((glm.get("reasoning_effort"), glm.get("enable_thinking"), glm.get("chat_template_kwargs")), (&s("low"), &Value::Null, &Value::Null));
        args.model = "qwen3.8-flash-next-fp8".into();
        let qwen = stream::request_body(&template, &args, 8);
        assert_eq!((qwen.get("reasoning_effort"), qwen.get("enable_thinking")), (&Value::Null, &b(false)));
        assert_eq!(encode(qwen.get("chat_template_kwargs")), r#"{"enable_thinking":false}"#);
    }

    fn assert_requests(requests: &[Value], model: &str, mode: Option<&str>) {
        assert_eq!(requests.len(), 31);
        for body in requests {
            assert_eq!(body.get("model"), &s(model));
            assert_eq!(body.get("draft"), &mode.map(s).unwrap_or(Value::Null));
            assert_eq!(body.get("max_tokens"), &n(256));
            assert_eq!(body.get("ignore_eos"), &b(true));
            assert_eq!(body.get("cache"), &b(false));
            assert_eq!(body.get("enable_thinking"), &b(false));
            assert_eq!(body.get("messages").arr()[0].get("content"), &s(include_str!("prompt.txt")));
        }
    }

    #[test]
    fn drafting_server_zero_flag_defaults_and_request_overrides() {
        for mode in [None, Some("plain"), Some("strict")] {
            let server = Server::start(Some("release-model-from-health"), false, None);
            let extra = mode.map_or(vec![], |mode| vec!["--draft", mode]);
            let args = server.args(&extra);
            assert_eq!(args.output, "./vade-bench.json");
            assert!(args.prompts.is_none() && !args.model_explicit);
            assert_eq!(args.concurrency, [1, 2, 4, 8, 16]);
            let receipt = super::super::execute(&args);
            assert_eq!(receipt.get("status"), &s("READY"), "{}", encode(&receipt));
            let drafting = match mode { Some("plain") => "off", Some("strict") => "strict-requested", _ => "server-default" };
            let depth = if mode == Some("plain") { 0 } else { 3 };
            assert_header(&receipt, "release-model-from-health", mode.unwrap_or("server"), drafting, 3);
            let arm = &receipt.get("arms").arr()[0];
            assert_header(arm, "release-model-from-health", mode.unwrap_or("server"), drafting, 3);
            assert_eq!(arm.get("cells").arr().len(), 5);
            assert_eq!(receipt.path("prompt_source.builtin"), &s("writing-v1"));
            assert_eq!(receipt.path("prompt_source.sha256"), &s(digest(include_bytes!("prompt.txt"))));
            for cell in arm.get("cells").arr() {
                assert!(cell.get("eligible").yes());
                for request in cell.get("requests").arr() {
                    assert_eq!(request.get("requested_draft_mode"), &s(mode.unwrap_or("server")));
                    assert_eq!(request.path("sampling_echo.vade.numerics.mtp_depth"), &n(3));
                    assert_eq!(request.path("sampling_echo.vade.mtp_depth"), &n(depth));
                }
            }
            assert_requests(&server.finish(), "release-model-from-health", mode);
        }
    }

    #[test]
    fn missing_or_blank_health_model_uses_release_fallback() {
        for model in [None, Some(""), Some("   ")] {
            let server = Server::start(model, false, None);
            let receipt = super::super::execute(&server.args(&[]));
            assert_eq!(receipt.get("status"), &s("READY"), "{}", encode(&receipt));
            assert_header(&receipt, "qwen3.8-flash-next-fp4-local", "server", "server-default", 3);
            assert_requests(&server.finish(), "qwen3.8-flash-next-fp4-local", None);
        }
    }


    #[test]
    fn comparison_requires_the_same_resolved_model_on_every_arm() {
        struct Probe(std::path::PathBuf);
        impl Drop for Probe {
            fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
        }
        let probe = Probe(std::env::temp_dir().join(format!("vade-bench-model-comparison-{}.json", std::process::id())));
        std::fs::write(&probe.0, br#"{"prompt":"8k probe","max_tokens":2}"#).unwrap();
        for candidate_model in ["model-a", "model-b"] {
            let baseline = Server::start(Some("model-a"), false, None);
            let candidate = Server::start(Some(candidate_model), false, None);
            let a_url = format!("http://{}", baseline.address);
            let b_url = format!("http://{}", candidate.address);
            let args = baseline.args(&[
                "--ab-urls", &a_url, &b_url, "--concurrency", "1", "--max-tokens", "2",
                "--prefill-body", probe.0.to_str().unwrap(), "--checkpoint-hash", "stub-checkpoint",
            ]);
            let receipt = super::super::execute(&args);
            assert!(receipt.get("arms").arr().iter().all(|arm| arm.get("eligible").yes()), "{}", encode(&receipt));
            let same_model = candidate_model == "model-a";
            assert_eq!(receipt.get("status"), &s(if same_model { "READY" } else { "FAIL" }));
            assert_eq!(encode(receipt.get("errors")).contains("comparison server model IDs differ"), !same_model);
            assert!(receipt.get("receipts").arr().iter().all(|row|
                (row.get("verdict") == &s("INVALID")) != same_model));
            assert_eq!(baseline.finish().len(), 8);
            assert_eq!(candidate.finish().len(), 8);
        }
    }

    #[test]
    fn non_string_health_model_is_invalid_even_with_auto_discovery() {
        let args = Config::parse(&["--self-test".into()]).unwrap();
        for model in [n(1), b(false), a(vec![]), o([])] {
            let health = o([("model", model), ("mtp_depth", n(3))]);
            let resolved = super::super::support::resolve_model(&args, &health);
            assert!(super::super::support::validate_health(&health, &resolved, false).is_err());
        }
    }

    #[test]
    fn server_build_sha_is_verbatim_and_independent_of_build_hash_override() {
        let args = Config::parse(&["--self-test".into(), "--build-hash-field".into(), "alternate".into()]).unwrap();
        let health = o([
            ("vade", o([("build", o([("git_sha", s(" reported-sha "))]))])),
            ("alternate", s("comparison-identity")),
        ]);
        let identity = super::super::support::validate_health(&health, &args, false).unwrap();
        assert_eq!(identity.get("server_build_sha"), &s(" reported-sha "));
        assert_eq!(identity.get("build_hash"), &s("comparison-identity"));
    }

    #[test]
    fn explicit_model_mismatch_fails_before_sending_a_chat() {
        let server = Server::start(Some("actual-model"), false, None);
        let receipt = super::super::execute(&server.args(&["--model", "wrong-model"]));
        assert_eq!(receipt.get("status"), &s("FAIL"));
        assert!(encode(receipt.get("arms")).contains("/health model differs from --model"));
        assert!(server.finish().is_empty());
    }

    #[test]
    fn health_drafting_drift_invalidates_rates() {
        let server = Server::start(Some("release-model"), true, None);
        let receipt = super::super::execute(&server.args(&[]));
        assert_eq!(receipt.get("status"), &s("FAIL"));
        let arm = &receipt.get("arms").arr()[0];
        assert!(!arm.get("eligible").yes());
        assert!(encode(arm.get("errors")).contains("server identity changed"));
        let first = &arm.get("cells").arr()[0];
        assert!(!first.get("eligible").yes());
        assert_eq!(first.get("aggregate_tokens_per_s"), &Value::Null);
        server.finish();
    }

    #[test]
    fn plain_override_rejects_a_drafting_response() {
        let server = Server::start(Some("release-model"), false, Some("strict"));
        let receipt = super::super::execute(&server.args(&["--draft", "plain"]));
        assert_eq!(receipt.get("status"), &s("FAIL"));
        let arm = &receipt.get("arms").arr()[0];
        assert!(encode(arm.get("errors")).contains("drafting echo disagrees"));
        assert_eq!(arm.get("cells").arr()[0].get("aggregate_tokens_per_s"), &Value::Null);
        assert_eq!(server.finish().len(), 1);
    }

    #[test]
    fn native_health_zero_retains_requested_mode_and_reports_without_blocking_rates() {
        for mode in [None, Some("strict")] {
            // Native immutable health metadata reports zero while responses
            // report depth three. Record both without claiming default is OFF.
            let server = Server::start_with_depth(Some("release-model"), 0, false, None);
            let extra = mode.map_or(vec![], |mode| vec!["--draft", mode]);
            let receipt = super::super::execute(&server.args(&extra));
            assert_eq!(receipt.get("status"), &s("READY"), "{}", encode(&receipt));
            assert_eq!(receipt.get("schema_version"), &n(4));
            let label = if mode.is_some() { "strict-requested" } else { "server-default" };
            for container in [&receipt, &receipt.get("arms").arr()[0]] {
                assert_header(container, "release-model", mode.unwrap_or("server"), label, 0);
                assert_eq!(container.path("draft_observations.observed_draft_modes"), &a(vec![s("strict")]));
                assert_eq!(container.path("draft_observations.observed_mtp_depths"), &a(vec![n(3)]));
            }
            let arm = &receipt.get("arms").arr()[0];
            assert!(arm.get("eligible").yes());
            assert_eq!(arm.get("cells").arr().len(), 5);
            for cell in arm.get("cells").arr() {
                assert!(cell.get("eligible").yes());
                assert!(cell.get("requested_drafting_consistent").yes());
                assert_eq!(cell.path("health_before.vade.numerics.mtp_depth"), &n(0));
                assert_eq!(cell.path("identity.server_build_sha"), &s("stub-release"));
                assert_eq!(cell.path("identity.decoding.server_mtp_depth_label"), &s(super::super::drafting::SERVER_DEPTH_LABEL));
                assert!(cell.get("aggregate_tokens_per_s").num().is_some());
                for request in cell.get("requests").arr() {
                    assert_eq!(request.get("requested_draft_mode"), &s(mode.unwrap_or("server")));
                    assert_eq!(request.path("sampling_echo.vade.numerics.mtp_depth"), &n(0));
                    assert_eq!(request.path("sampling_echo.vade.mtp_depth"), &n(3));
                    assert!(request.get("tokens_per_s").num().is_some());
                }
            }
            assert_requests(&server.finish(), "release-model", mode);
        }
    }

    #[test]
    fn native_health_zero_allows_a_confirmed_plain_request_override() {
        let server = Server::start_with_depth(Some("release-model"), 0, false, None);
        let receipt = super::super::execute(&server.args(&["--draft", "plain"]));
        assert_eq!(receipt.get("status"), &s("READY"), "{}", encode(&receipt));
        assert_eq!(receipt.get("model"), &s("release-model"));
        assert_eq!(receipt.path("decoding.request_mode"), &s("plain"));
        assert_eq!(receipt.path("decoding.drafting"), &s("off"));
        assert_eq!(receipt.path("decoding.mtp_depth"), &Value::Null);
        assert_eq!(receipt.path("decoding.server_mtp_depth"), &n(0));
        assert_eq!(receipt.get("server_build_sha"), &s("stub-release"));
        for cell in receipt.get("arms").arr()[0].get("cells").arr() {
            assert!(cell.get("requests").arr().iter().all(|request| request.get("requested_draft_mode") == &s("plain")));
        }
        assert_eq!(receipt.path("draft_observations.observed_draft_modes"), &a(vec![s("plain")]));
        assert_eq!(receipt.path("draft_observations.observed_mtp_depths"), &a(vec![n(0)]));
        assert_requests(&server.finish(), "release-model", Some("plain"));
    }
}
