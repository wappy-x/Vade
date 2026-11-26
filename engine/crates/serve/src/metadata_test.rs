//! Public health and response receipts must describe serving policy, not loader defaults.
use super::model_response;
use crate::{Limits, metadata::Metadata, protocol::{self, DraftMode, SamplingEcho}};
use engine_model::json::{self, Value};
use std::{path::PathBuf, time::{Instant, SystemTime, UNIX_EPOCH}};

struct Fixture { root: PathBuf, metadata: Metadata }
impl Fixture {
    fn new(depth: usize) -> Self {
        let root = std::env::temp_dir().join(format!("vade-draft-metadata-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("model.safetensors"), b"fixture").unwrap();
        let metadata = Metadata::prepare(&root, c"ptx", depth, 1, "local", SystemTime::now()).unwrap();
        Self { root, metadata }
    }
}
impl Drop for Fixture { fn drop(&mut self) { std::fs::remove_dir_all(&self.root).unwrap(); } }
fn parse(body: &str) -> Value { json::parse(body.as_bytes(), Default::default()).unwrap() }
fn assert_drafting(vade: &Value, mode: DraftMode, depth: usize, request: bool) {
    let vade = vade.object().unwrap();
    let numerics = vade["numerics"].object().unwrap();
    assert_eq!(numerics["draft"].string().unwrap(), mode.name());
    assert_eq!(numerics["mtp_depth"].u64().unwrap(), depth as u64);
    if request {
        assert_eq!(vade["draft"], numerics["draft"]);
        assert_eq!(vade["mtp_depth"], numerics["mtp_depth"]);
    }
}
#[test]
fn health_uses_serving_default_depth_in_both_modes() {
    // Next prepares the loader receipt at zero and keeps more resident draft slots
    // than the serving default. FP8 prepares its configured depth directly.
    for (prepared, resident, default) in [(0, Some(5), 3), (2, None, 2)] {
        let fixture = Fixture::new(prepared);
        let limits = Limits { metadata: Some(fixture.metadata.clone()), next_mtp_depth: resident, ..Limits::default() };
        for (mode, depth, expected) in [(DraftMode::Strict, default, default),
            (DraftMode::Plain, 0, 0), (DraftMode::Plain, default, 0), (DraftMode::Strict, 0, 0)] {
            let health = model_response("/health", "ready", "test", 0, 0, 0, None, &limits, 2048, Some((mode, depth)), None, false).unwrap();
            let health = parse(&health);
            assert_drafting(&health.object().unwrap()["vade"], mode, expected, false);
        }
        let unchanged = parse(fixture.metadata.json());
        assert_eq!(unchanged.object().unwrap()["numerics"].object().unwrap()["mtp_depth"].u64().unwrap(), prepared as u64);
    }
}
#[test]
fn response_metadata_tracks_effective_drafting_without_changing_shared_receipt() {
    for (prepared, resident, default) in [(0, Some(5), 3), (2, None, 2)] {
        let fixture = Fixture::new(prepared);
        let limits = Limits { metadata: Some(fixture.metadata.clone()), next_mtp_depth: resident, ..Limits::default() };
        let request = protocol::parse(br#"{"prompt":"hi","temperature":0}"#, false, &limits).unwrap();
        for (mode, depth, expected) in [(DraftMode::Strict, default, default),
            (DraftMode::Plain, default, 0), (DraftMode::Plain, 0, 0), (DraftMode::Strict, 0, 0),
            (DraftMode::Strict, 1, 1)] {
            let mut echo = SamplingEcho::new(&request, mode, depth).with_metadata(&limits);
            let body = echo.attach("{\"choices\":[]}".into());
            assert_drafting(&parse(&body).object().unwrap()["vade"], mode, expected, true);
            let events = echo.attach_events("data: {\"choices\":[]}\n\ndata: [DONE]\n\n".into());
            let body = events.lines().next().unwrap().strip_prefix("data: ").unwrap();
            assert_drafting(&parse(body).object().unwrap()["vade"], mode, expected, true);
            // The scheduler can resolve auto/strict to plain after admission.
            echo.draft = DraftMode::Plain;
            assert_drafting(&parse(&echo.json()), DraftMode::Plain, 0, true);
        }
        // The immutable hash worker is still shared by health and request clones.
        fixture.metadata.start();
        let started = Instant::now();
        while !fixture.metadata.json().contains("\"hash_state\":\"ready\"") && started.elapsed().as_secs() < 30 {
            std::thread::yield_now();
        }
        let echo = SamplingEcho::new(&request, DraftMode::Strict, default).with_metadata(&limits);
        let response = parse(&echo.json());
        assert_eq!(response.object().unwrap()["checkpoint"].object().unwrap()["hash_state"].string().unwrap(), "ready");
        assert_drafting(&response, DraftMode::Strict, default, true);
        let unchanged = parse(fixture.metadata.json());
        assert_eq!(unchanged.object().unwrap()["numerics"].object().unwrap()["mtp_depth"].u64().unwrap(), prepared as u64);
    }
}
#[test]
fn health_without_the_key_holds_the_state_only() {
    let fixture = Fixture::new(0);
    let limits = Limits { metadata: Some(fixture.metadata.clone()), ..Limits::default() };
    let health = |path, state| model_response(path, state, "test", 3, 2, 1, Some(1 << 30), &limits, 2048, Some((DraftMode::Strict, 2)), Some("{\"entries\":1}".into()), false).unwrap();
    // The full body holds the build, the checkpoint and the live load.
    let full = health("/health", "ready");
    for fact in ["\"active\":3", "\"queued\":2", "\"preparing\":1", "\"device_bytes\":1073741824", "\"prefix_cache\":{\"entries\":1}", "\"vade\":{", "\"checkpoint\":{\"directory\":", "\"events\":", "\"caps\":{"] {
        assert!(full.contains(fact), "{fact}: {full}");
    }
    // A request that did not show the key gets only what a liveness probe needs.
    for state in ["ready", "loading", "draining", "failed"] {
        assert_eq!(health(crate::http::HEALTH_LIVE, state), format!("{{\"status\":\"{state}\"}}"));
    }
}

/// The device bytes grow by what a kept prompt holds, so with private counts `/health` (which asks no credential)
/// would tell a caller of another cache scope how long someone's prompt is. With private counts it shows null.
#[test]
fn health_shows_no_device_bytes_with_private_counts() {
    let limits = Limits::default();
    let health = |private| model_response("/health", "ready", "test", 0, 0, 0, Some(98_517_856_264), &limits, 2048, None, None, private).unwrap();
    assert!(health(false).contains("\"device_bytes\":98517856264"), "{}", health(false));
    assert!(health(true).contains("\"device_bytes\":null") && !health(true).contains("98517856264"), "{}", health(true));
    // WHERE THIS MEETS THE SHORT KEYLESS /health: a request that showed the key gets the full body, and with private
    // counts that body holds no device bytes and no cache event name. A request that did not show the key gets the
    // state only, with private counts or without: the short body wins, and private counts add nothing to it.
    assert!(health(true).contains("\"model\":\"test\"") && health(true).contains("\"caps\":{") && !health(true).contains("prefix_cache_"), "{}", health(true));
    for private in [false, true] {
        assert_eq!(model_response(crate::http::HEALTH_LIVE, "ready", "test", 3, 2, 1, Some(98_517_856_264), &limits, 2048, None, Some("{\"entries\":1}".into()), private).unwrap(), "{\"status\":\"ready\"}", "private {private}");
    }
}
