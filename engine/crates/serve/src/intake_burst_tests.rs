use super::{Burst, DEFAULT_WINDOW, MAX_WINDOW, View, hold, parse};
use std::time::{Duration, Instant};

#[test]
fn window_setting_is_strict() {
    assert_eq!(parse(Err(std::env::VarError::NotPresent)), Ok(None));
    assert_eq!(parse(Ok("0".into())), Ok(Some(Duration::ZERO)));
    assert_eq!(parse(Ok("10".into())), Ok(Some(DEFAULT_WINDOW)));
    assert_eq!(parse(Ok("100".into())), Ok(Some(MAX_WINDOW)));
    for value in ["", "101", "-1", "+5", "05", " 5", "5 ", "5ms", "1.5"] {
        assert_eq!(parse(Ok(value.into())).unwrap_err(), "VADE_INTAKE_BURST_MS must be unset or whole milliseconds 0..=100", "{value:?}");
    }
    assert!(parse(Err(std::env::VarError::NotUnicode("bad".into()))).is_err());
}

#[test]
fn hold_needs_evidence_room_and_time() {
    let ms = Duration::from_millis;
    let view = View { oldest: ms(1), cold: 1, arriving: true, room: true };
    assert!(hold(DEFAULT_WINDOW, view));
    // A lone request with nothing arriving never waits.
    assert!(!hold(DEFAULT_WINDOW, View { arriving: false, ..view }));
    // Two cold prompts are a burst even when nothing else is arriving.
    assert!(hold(DEFAULT_WINDOW, View { cold: 2, arriving: false, ..view }));
    // No free slot: nothing else can join the pass.
    assert!(!hold(DEFAULT_WINDOW, View { room: false, ..view }));
    // The window bounds the hold, and a zero window is main's behavior.
    assert!(!hold(DEFAULT_WINDOW, View { oldest: DEFAULT_WINDOW, ..view }));
    assert!(!hold(Duration::ZERO, View { oldest: Duration::ZERO, ..view }));
    assert!(!hold(DEFAULT_WINDOW, View { cold: 0, ..view }));
}

#[test]
fn burst_counts_only_time_the_hold_added() {
    let t0 = Instant::now();
    let at = |us| t0 + Duration::from_micros(us);
    let ms = Duration::from_millis;
    let view = |oldest| View { oldest, cold: 1, arriving: true, room: true };
    let mut burst = Burst::new(DEFAULT_WINDOW);
    // Before the prompt age the pass is not ready: the hold adds nothing yet.
    assert!(!burst.ready(at(0), true, false, view(ms(0))));
    assert_eq!(burst.park(), None);
    assert!(!burst.ready(at(2_000), true, true, view(ms(2))));
    assert!(!burst.ready(at(5_000), true, true, view(ms(5))));
    // The window ends the hold.
    assert!(burst.ready(at(10_000), true, true, view(ms(10))));
    let (held, line) = burst.started(at(10_000), 4, ms(10));
    assert_eq!(held, ms(8));
    assert_eq!(line, "intake_burst chats=4 waited_ms=10.000 held_ms=8.000 window_ms=10");
    // A batch that is not idle is never held and clears the measurement.
    assert!(!burst.ready(at(20_000), true, true, view(ms(1))));
    assert!(burst.ready(at(21_000), false, true, view(ms(2))));
    assert_eq!(burst.started(at(21_000), 1, ms(2)).0, Duration::ZERO);
}

#[test]
fn held_turns_park_until_the_window() {
    let t0 = Instant::now();
    let ms = Duration::from_millis;
    let view = |oldest| View { oldest, cold: 1, arriving: true, room: true };
    let mut burst = Burst::new(DEFAULT_WINDOW);
    // A turn that only the hold stops parks for 1 ms at most, and never past the window.
    assert!(!burst.ready(t0, true, true, view(ms(2))));
    assert_eq!(burst.park(), Some(ms(1)));
    assert_eq!(burst.park(), None);
    assert!(!burst.ready(t0, true, true, view(Duration::from_micros(9_600))));
    assert_eq!(burst.park(), Some(Duration::from_micros(400)));
    // A turn that does not hold, or that is not ready for main's own reasons, does not park.
    assert!(burst.ready(t0, true, true, view(ms(10))));
    assert_eq!(burst.park(), None);
    assert!(!burst.ready(t0, true, false, view(ms(2))));
    assert_eq!(burst.park(), None);
    assert!(burst.ready(t0, false, true, view(ms(1))));
    assert_eq!(burst.park(), None);
}

#[test]
fn cancelled_burst_leaves_no_held_time() {
    let t0 = Instant::now();
    let at = |us| t0 + Duration::from_micros(us);
    let ms = Duration::from_millis;
    let mut burst = Burst::new(DEFAULT_WINDOW);
    // The hold starts at 2 ms, then every held chat is cancelled.
    assert!(!burst.ready(at(2_000), true, true, View { oldest: ms(2), cold: 2, arriving: false, room: true }));
    // A lone request accepted after the hold began is not held and logs no
    // held time: in the same turn as the cancel (3 ms) or after an idle gap.
    for (now, accepted) in [(3_000, 2_500), (52_000, 50_000)] {
        let oldest = Duration::from_micros(now - accepted);
        let lone = View { oldest, cold: 1, arriving: false, room: true };
        assert!(burst.ready(at(now), true, true, lone));
        assert_eq!(burst.started(at(now), 1, oldest).0, Duration::ZERO);
        assert!(!burst.ready(at(2_000), true, true, View { oldest: ms(2), cold: 2, arriving: false, room: true }));
    }
    // A chat that was there when the hold began keeps its held time.
    assert!(!burst.ready(at(4_000), true, true, View { oldest: ms(4), cold: 2, arriving: false, room: true }));
    assert!(burst.ready(at(12_000), true, true, View { oldest: ms(12), cold: 1, arriving: false, room: true }));
    assert_eq!(burst.started(at(12_000), 1, ms(12)).0, ms(10));
}

/// One chat of a fake arrival sequence: milliseconds after the first connect.
#[derive(Clone, Copy)]
struct Arrival { accept: f64, bytes: f64 }
/// The first prompt pass of an idle batch in the fake loop.
#[derive(Debug, PartialEq)]
struct Pass { start: f64, chats: Vec<usize>, held: f64, max_hold_age: f64 }

// Fake costs in milliseconds. ADMIT is the serial admission spacing in the
// request_admitted lines of the split rounds (about 1.5 ms per chat).
const TURN: f64 = 0.05;
const PREPARE: f64 = 0.3;
const ADMIT: f64 = 1.5;
const PROMPT_AGE: f64 = 2.0;
const COLLECT: f64 = 100.0;
const SLOTS: usize = 16;

/// The intake steps of one serving loop turn, on a fake clock: accept,
/// read, CPU preparation, serial admission (it waits while a request is in
/// preparation, as admission does), a second accept, the one-turn deferral
/// for a connection with unread bytes, the 2 ms prompt age, the preparation
/// collect, the burst hold and its park. With `lone_serial`, a lone cold
/// prompt takes the serial prompt path at once, as the card prompts do
/// (above the shared prefill limit alone). Returns the first prompt pass.
fn first_pass(arrivals: &[Arrival], window: Duration, lone_serial: bool) -> Pass {
    #[derive(Clone, Copy, PartialEq)]
    enum State { Out, Reading, Preparing(f64), Queued, Admitted(f64) }
    let base = Instant::now();
    let clock = |ms: f64| base + Duration::from_secs_f64(ms / 1000.0);
    let mut state = vec![State::Out; arrivals.len()];
    let mut burst = Burst::new(window);
    let mut max_hold_age: f64 = 0.0;
    let (mut now, mut deferred) = (0.0, false);
    loop {
        for (i, a) in arrivals.iter().enumerate() {
            if state[i] == State::Out && a.accept <= now { state[i] = State::Reading; }
            if state[i] == State::Reading && a.bytes <= now { state[i] = State::Preparing(now + PREPARE); }
        }
        for s in state.iter_mut() { if let State::Preparing(done) = *s { if done <= now { *s = State::Queued; } } }
        let preparing = state.iter().any(|s| matches!(s, State::Preparing(_)));
        let oldest_queued = (0..arrivals.len()).filter(|&i| state[i] == State::Queued)
            .map(|i| arrivals[i].accept).fold(f64::INFINITY, f64::min);
        if oldest_queued.is_finite() && !(preparing && now - oldest_queued < COLLECT) {
            for s in state.iter_mut() { if *s == State::Queued { now += ADMIT; *s = State::Admitted(now); } }
        }
        // The loop accepts again after admission, before the prompt decision.
        for (i, a) in arrivals.iter().enumerate() { if state[i] == State::Out && a.accept <= now { state[i] = State::Reading; } }
        let cold: Vec<_> = (0..arrivals.len()).filter(|&i| matches!(state[i], State::Admitted(_))).collect();
        if !cold.is_empty() {
            let admitted = |i: usize| match state[i] { State::Admitted(at) => at, _ => unreachable!() };
            let collecting = preparing && cold.iter().map(|&i| now - admitted(i)).fold(0.0, f64::max) < COLLECT;
            let pending = !deferred && (0..arrivals.len()).any(|i| state[i] == State::Reading && arrivals[i].bytes <= now);
            deferred = pending;
            // The serial path has no prompt age and no preparation collect.
            let ready = !pending && (lone_serial && cold.len() == 1 || !collecting && cold.iter().any(|&i| now - admitted(i) >= PROMPT_AGE));
            let oldest = cold.iter().map(|&i| arrivals[i].accept).fold(f64::INFINITY, f64::min);
            // A connection without request bytes counts only for one window after its accept.
            let arriving = (0..arrivals.len()).any(|i| match state[i] {
                State::Reading => now - arrivals[i].accept < window.as_secs_f64() * 1000.0,
                s => matches!(s, State::Preparing(_) | State::Queued) });
            let view = View { oldest: Duration::from_secs_f64((now - oldest).max(0.0) / 1000.0), cold: cold.len(), arriving, room: cold.len() < SLOTS };
            let held = hold(window, view);
            if ready && held { max_hold_age = max_hold_age.max(now - oldest); }
            if burst.ready(clock(now), true, ready, view) {
                let held = burst.started(clock(now), cold.len(), view.oldest).0.as_secs_f64() * 1000.0;
                return Pass { start: now, chats: cold, held, max_hold_age };
            }
        }
        // A held turn parks; a finished preparation unparks the loop.
        let park = burst.park().map_or(0.0, |wait| wait.as_secs_f64() * 1000.0);
        let unpark = state.iter().filter_map(|s| match *s { State::Preparing(done) => Some(done), _ => None }).fold(now + park, f64::min);
        now = unpark.max(now) + TURN;
        assert!(now < 1000.0, "no prompt pass");
    }
}

fn arrivals(accept: &[f64], late: &[usize], bytes_delay: f64) -> Vec<Arrival> {
    accept.iter().enumerate().map(|(i, &accept)| Arrival { accept,
        bytes: accept + if late.contains(&i) { bytes_delay } else { 0.0 } }).collect()
}

#[test]
fn receipt_split_patterns_form_one_pass() {
    // Accept offsets (ms) of split rounds in rank-0 server.logs of main 285fa8a
    // two-Spark cards (admitted_at_ms - queue_ms). The logs do not give when
    // the late chats' request bytes came; the fake sequence sends them 4 ms
    // after their accept, which splits the round under main's rules.
    // With lone_serial, the first chat that is admitted alone takes the
    // serial path at once (own-intake-screen-2 s3 C8 round 3 on 2fa1c06).
    let rounds: [(&[f64], &[usize]); 5] = [
        (&[0.00, 0.00, 0.01, 0.02], &[1, 2, 3]),
        (&[0.00, 0.25, 1.83, 1.84], &[1, 2, 3]),
        (&[0.00, 1.06, 1.07, 2.96, 2.96, 2.96, 2.97, 2.97], &[1, 2, 3, 4, 5, 6, 7]),
        (&[0.00, 0.00, 0.01, 0.01, 0.37, 0.37, 2.31, 2.32], &[0, 2, 3, 4, 5, 6, 7]),
        // One Spark, own-intake-sat r1 C8 round 3: eight accepts within 0.006 ms.
        (&[0.0; 8], &[1, 2, 3, 4, 5, 6, 7]),
    ];
    for ((accept, late), lone_serial) in rounds.into_iter().flat_map(|r| [(r, false), (r, true)]) {
        let sequence = arrivals(accept, late, 4.0);
        let main = first_pass(&sequence, Duration::ZERO, lone_serial);
        assert!(main.chats.len() < accept.len(), "main must split this round: {main:?}");
        assert_eq!(main.held, 0.0);
        let burst = first_pass(&sequence, DEFAULT_WINDOW, lone_serial);
        assert_eq!(burst.chats.len(), accept.len(), "{burst:?}");
        // The hold stops before the window; the joiners then add their admission work.
        let joined = (burst.chats.len() - main.chats.len()) as f64;
        assert!(burst.max_hold_age < 10.0 && burst.held <= 10.0 + 3.0 * TURN + PREPARE + joined * ADMIT, "{burst:?}");
    }
}

#[test]
fn lone_request_and_complete_burst_keep_main_timing() {
    for lone_serial in [false, true] {
        for bytes in [0.0, 0.5, 3.0] {
            let lone = [Arrival { accept: 0.0, bytes }];
            assert_eq!(first_pass(&lone, DEFAULT_WINDOW, lone_serial), first_pass(&lone, Duration::ZERO, lone_serial));
        }
        // Eight chats read at once: serial admission (8 x 1.5 ms) outlasts the window.
        let eight = arrivals(&[0.0; 8], &[], 0.0);
        assert_eq!(first_pass(&eight, DEFAULT_WINDOW, lone_serial), first_pass(&eight, Duration::ZERO, lone_serial));
    }
}

#[test]
fn hold_bound_over_fake_arrival_sequences() {
    // Small linear congruential generator: fixed sequences, no dependency.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = |limit: f64| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 * limit
    };
    let mut joined = 0;
    for case in 0..500 {
        let chats = 1 + (next(8.0) as usize).min(7);
        let mut sequence: Vec<_> = (0..chats).map(|_| { let accept = next(15.0); Arrival { accept, bytes: accept + next(12.0) } }).collect();
        sequence[0].accept = 0.0;
        sequence[0].bytes = sequence[0].bytes.min(next(1.0));
        let first = sequence.iter().map(|a| a.accept).fold(f64::INFINITY, f64::min);
        for (window, lone_serial) in [DEFAULT_WINDOW, Duration::from_millis(3), MAX_WINDOW].into_iter().flat_map(|w| [(w, false), (w, true)]) {
            let w = window.as_secs_f64() * 1000.0;
            let main = first_pass(&sequence, Duration::ZERO, lone_serial);
            let burst = first_pass(&sequence, window, lone_serial);
            // The hold never acts at or after the oldest accept plus the window.
            assert!(burst.max_hold_age < w, "case {case}: {burst:?}");
            // The pass waits for no fewer chats than main's.
            assert!(main.chats.iter().all(|c| burst.chats.contains(c)), "case {case}: {main:?} {burst:?}");
            // After the window, main's rules (the preparation collect and the
            // deferral of unread bytes) can still take a chat whose bytes come
            // later. So the pass starts by main's start, the window end or the
            // bytes of the last chat that joined, whichever is last, plus three
            // loop turns, one preparation and the joiners' admission work.
            let joiners: Vec<_> = burst.chats.iter().filter(|c| !main.chats.contains(c)).collect();
            let last = joiners.iter().map(|&&c| sequence[c].bytes).fold(main.start.max(first + w), f64::max);
            let extra = 3.0 * TURN + PREPARE + joiners.len() as f64 * ADMIT + 1e-9;
            assert!(burst.start <= last + extra && burst.start - burst.held <= main.start + 1e-9, "case {case}: {main:?} {burst:?}");
            if chats == 1 { assert_eq!(burst, main, "case {case}"); }
            joined += burst.chats.len() - main.chats.len();
        }
    }
    assert!(joined > 0);
}
