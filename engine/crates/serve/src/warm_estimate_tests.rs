use super::*;

const NONE: Duration = Duration::ZERO;
fn at(ms: f64) -> Duration { Duration::from_secs_f64(ms / 1000.0) }
fn within(estimate: f64, actual: f64, bound: f64) -> bool { ((estimate - actual) / actual).abs() <= bound }

#[test]
fn seed_reproduces_the_cold_8k_card() {
    let e = Estimator::SEED.estimate(8192, 0, NONE);
    assert!((e.warm_ms - 3761.0).abs() < 1e-6 && e.cold_ms == e.warm_ms, "{e:?}");
    assert_eq!((e.prompt_tokens, e.cached_tokens), (8192, 0));
}

#[test]
fn formula_parts() {
    let seed = Estimator::SEED;
    // An exact replay prefills nothing: only the time already spent remains.
    let replay = seed.estimate(8192, 8192, at(2.0));
    assert!((replay.warm_ms - 2.0).abs() < 1e-9 && (replay.cold_ms - 3761.0).abs() < 1e-6, "{replay:?}");
    // A tail of up to 256 rows costs the fixed time at any context length.
    for (prompt, cached) in [(300, 100), (32_305, 32_064), (129_000, 128_744)] {
        assert_eq!(seed.estimate(prompt, cached, NONE).warm_ms, 350.0);
    }
    // More cached context makes each uncached row dearer, never cheaper.
    let near = seed.estimate(4096 + 2048, 4096, NONE).warm_ms;
    let far = seed.estimate(65_536 + 2048, 65_536, NONE).warm_ms;
    assert!(far > near && near > 350.0, "{near} {far}");
    // Cached tokens above the prompt length count as the whole prompt.
    assert_eq!(seed.estimate(100, 500, NONE).cached_tokens, 100);
}

#[test]
fn rates_start_from_the_seed_and_converge() {
    let mut e = Estimator::SEED;
    let tail = e.estimate(1000, 900, NONE);
    // The seed counts as one time: a mean of the seed and the first three.
    e.observe(&tail, at(250.0));
    assert_eq!(e.fixed_ms, 300.0);
    e.observe(&tail, at(450.0));
    assert_eq!(e.fixed_ms, 350.0);
    e.observe(&tail, at(350.0));
    assert_eq!((e.fixed_ms, e.fixed_seen), (350.0, 3));
    // After that, each time moves the rate a quarter of the way.
    e.observe(&tail, at(750.0));
    assert_eq!(e.fixed_ms, 450.0);
    // Setup time is not prefill time; a replay teaches nothing.
    e.observe(&e.estimate(1000, 900, at(100.0)), at(550.0));
    assert_eq!(e.fixed_ms, 450.0);
    let before = e;
    e.observe(&e.estimate(1000, 1000, NONE), at(9000.0));
    assert_eq!(e, before);
    // A middle-sized prefill teaches neither rate.
    e.observe(&e.estimate(1000, 0, NONE), at(9000.0));
    assert_eq!(e, before);
    // A bulk prefill moves the row rate: the same prompt's estimate goes
    // half way to its measured time.
    let cold = e.estimate(32_099, 0, NONE);
    e.observe(&cold, at(15_000.0));
    assert!((e.estimate(32_099, 0, NONE).warm_ms - (cold.warm_ms + 15_000.0) / 2.0).abs() < 1e-6);
    // A server that runs twice as fast converges to its own rate.
    let mut fast = Estimator::SEED;
    for _ in 0..16 { let e = fast.estimate(8192, 0, NONE); fast.observe(&e, at(1880.0)); }
    assert!(within(fast.estimate(8192, 0, NONE).warm_ms, 1880.0, 0.01), "{fast:?}");
}

#[test]
fn headers_and_json() {
    let e = Estimator::SEED.estimate(8000, 7812, at(12.4));
    assert_eq!(e.headers(), "X-Vade-Prompt-Tokens: 8000\r\nX-Vade-Cached-Tokens: 7812\r\nX-Vade-Estimated-First-Token-Ms: 362\r\nX-Vade-Estimated-Cold-Ms: 3677\r\n");
    assert_eq!(e.json(), "\"prefill_estimated_ms\":362.4,\"prefill_cold_estimated_ms\":3677.4");
}

/// The one-Spark FP4-Local card at main 380d8e5 (docs/benchmarks.md, client
/// seconds to the first token), estimated from the seed with nothing learned.
#[test]
fn seed_is_within_20_percent_on_the_one_spark_card() {
    let seed = Estimator::SEED;
    for (prompt, cached, seconds) in [(8192, 0, 3.761), (32_099, 0, 15.66), (64_217, 0, (32.86 + 33.08) / 2.0),
        (129_220, 0, (71.31 + 71.74) / 2.0), (32_305, 32_064, 0.36)] {
        let estimate = seed.estimate(prompt, cached, NONE).warm_ms;
        assert!(within(estimate, seconds * 1000.0, 0.20), "{prompt}/{cached}: {estimate} ms vs {seconds} s");
    }
    // The uncached rows alone at the cold 8K rate miss the warm turn by far.
    let naive = (32_305 - 32_064) as f64 * 3761.0 / 8192.0;
    assert!(!within(naive, 360.0, 0.20), "{naive}");
}

/// Canned admission inputs from one-Spark FP4-Local receipts (2026-10-30, 8K to 128K):
/// each row is one server start with three turns of one chat: prompt tokens,
/// reused tokens, restore ms, then the engine's admission_to_first_logits_ms
/// and the client's seconds to the first token.
const CURVE: [[(usize, usize, f64, f64, f64); 3]; 8] = [
    [(8372, 0, 0.0, 4330.226, 4.346), (8462, 8320, 2.307, 371.498, 0.384), (8546, 8448, 2.243, 360.198, 0.372)],
    [(8372, 0, 0.0, 4334.825, 4.378), (8462, 8320, 25.979, 397.041, 0.434), (8546, 8448, 17.971, 377.657, 0.408)],
    [(32_099, 0, 0.0, 15_981.358, 15.997), (32_189, 32_064, 2.270, 417.911, 0.431), (32_273, 32_128, 2.571, 341.497, 0.355)],
    [(32_099, 0, 0.0, 16_017.015, 16.098), (32_189, 32_064, 81.976, 508.162, 0.594), (32_273, 32_128, 64.810, 415.233, 0.492)],
    [(63_854, 0, 0.0, 33_299.354, 33.319), (63_944, 63_808, 2.386, 336.588, 0.352), (64_026, 63_936, 2.240, 319.490, 0.335)],
    [(63_854, 0, 0.0, 33_264.981, 33.427), (63_944, 63_808, 126.194, 489.417, 0.654), (64_026, 63_936, 125.871, 463.782, 0.643)],
    [(128_332, 0, 0.0, 72_110.501, 72.135), (128_422, 128_320, 2.504, 373.882, 0.413), (128_504, 128_384, 2.488, 400.920, 0.434)],
    [(128_332, 0, 0.0, 71_937.223, 71.966), (128_422, 0, 0.0, 71_840.515, 71.863), (128_504, 0, 0.0, 71_909.808, 71.947)],
];

/// Every turn is estimated before its prefill and then learned, as dispatch
/// does, on a fresh server per start and on one server that sees every start.
#[test]
fn canned_dispatch_inputs_are_within_20_percent() {
    let mut shared = Estimator::SEED;
    let (mut engine, mut client_worst) = (0.0f64, 0.0f64);
    for start in CURVE {
        let mut fresh = Estimator::SEED;
        for (prompt, cached, restore, first, client) in start {
            for e in [&mut fresh, &mut shared] {
                let estimate = e.estimate(prompt, cached, at(restore));
                assert!(within(estimate.warm_ms, first, 0.20), "{prompt}/{cached}: {} ms vs {first} ms", estimate.warm_ms);
                engine = engine.max(((estimate.warm_ms - first) / first).abs());
                client_worst = client_worst.max(((estimate.warm_ms - client * 1000.0) / (client * 1000.0)).abs());
                println!("{prompt}/{cached}: estimate {:.0} ms, engine {first:.0} ms, client {:.0} ms", estimate.warm_ms, client * 1000.0);
                e.observe(&estimate, at(first));
            }
        }
    }
    println!("worst relative error: engine {:.1} %, client {:.1} %", engine * 100.0, client_worst * 100.0);
}

#[test]
fn prefill_carries_and_learns_the_estimate() {
    use std::time::Instant;
    let admitted = Instant::now();
    let mut prefill = crate::usage::Prefill::estimated(64, 0, admitted);
    let estimate = prefill.estimate.unwrap();
    assert_eq!((estimate.prompt_tokens, estimate.cached_tokens), (64, 0));
    let headers = prefill.headers(true);
    assert!(headers.starts_with("X-Prefix-Reused-Tokens: 0\r\nX-Vade-Prompt-Tokens: 64\r\nX-Vade-Cached-Tokens: 0\r\n"), "{headers}");
    assert!(!prefill.headers(false).contains("X-Prefix-Reused-Tokens"));
    prefill.begin(admitted);
    prefill.commit(64, admitted + Duration::from_millis(5), true);
    let json = prefill.json();
    assert!(json.contains(&format!(",\"admission_to_first_logits_ms\":5.000,{}", estimate.json())), "{json}");
    // A queued cancellation has no estimate and keeps its bytes.
    let plain = crate::usage::Prefill::new(64, 0, admitted);
    assert!(plain.estimate.is_none() && plain.headers(false).is_empty() && !plain.json().contains("estimated"));
}
