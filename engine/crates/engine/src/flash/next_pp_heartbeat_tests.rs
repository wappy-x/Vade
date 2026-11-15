//! Heartbeat tests on a manual clock. The test back (test_clock::Back) steps the clock in
//! lockstep with the front, in steps smaller than the production heartbeat interval; its
//! heartbeats come from the production heartbeat wake at the production interval. No
//! assertion waits on how fast the machine is: under the manual clock the front's header
//! and payload reads have no real-time limit (the payload waits on the manual clock,
//! Link::test_payload_within), so a stalled test thread cannot fail them. The one
//! real-time limit is the 60 s hang guard.
use super::test_clock::{plus_one, within_a_minute, Back};
use super::*;

const LIMIT: Duration = Duration::from_millis(200);
/// One manual-clock step: smaller than the heartbeat interval.
const STEP: Duration = Duration::from_millis(25);
/// The long requests run more than 4x the limit of manual time.
const RUN: Duration = Duration::from_millis(900);

/// Where the front's manual clock stopped: the moment its wait ended.
fn stopped_at(link: &Link) -> Duration {
    Duration::from_millis(link.test_clock.as_ref().unwrap().1.load(Ordering::Relaxed))
}

#[test]
fn pp_heartbeats_keep_a_long_request_alive() {
    within_a_minute(|| {
        let (near, far) = pair();
        let mut link = front(near, LIMIT);
        let mut back = Back::manual(&mut link, far);
        assert!(STEP < back.period && RUN > 4 * LIMIT);
        let worker = std::thread::spawn(move || {
            let request = back.tcp_request(1);
            back.mark(Some(word(&request, 4)));
            while back.now() < RUN {
                assert!(back.step(STEP, true), "the front stopped waiting at {:?}", back.now());
            }
            back.mark(None);
            back.answer(&plus_one(&request));
            back
        });
        let mut rows = vec![1.5f32; W];
        let result = link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows);
        link.test_steps = None;
        let back = worker.join().unwrap();
        result.unwrap();
        assert!(back.now() > 4 * LIMIT);
        assert!(rows.iter().all(|&v| v == 2.5));
        // One heartbeat per production interval of the run, every one read as live.
        let implied = (RUN.as_nanos() / back.period.as_nanos()) as u64;
        assert_eq!((back.sent, link.beats, link.late), (implied, implied, 0));
    });
}

/// A TCP request whose back heartbeats while `beating(now)`; the front's error and the
/// manual time at which its wait ended.
fn tcp_until_failure(beating: fn(Duration, Duration) -> bool) -> (String, Duration, Link, Back) {
    let (near, far) = pair();
    let mut link = front(near, LIMIT);
    let mut back = Back::manual(&mut link, far);
    let worker = std::thread::spawn(move || {
        let request = back.tcp_request(1);
        back.mark(Some(word(&request, 4)));
        // Long enough for any wrong outcome to show; then answer, so a revived front succeeds.
        while back.now() < 8 * LIMIT {
            let on = beating(back.now() + STEP, back.period);
            if !back.step(STEP, on) { return back; }
        }
        back.mark(None);
        back.answer(&plus_one(&request));
        back
    });
    let mut rows = vec![1.5f32; W];
    let error = link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).expect_err("the request must fail");
    let at = stopped_at(&link);
    link.test_steps = None;
    let back = worker.join().unwrap();
    (error, at, link, back)
}

#[test]
fn pp_silent_back_fails_at_the_limit() {
    within_a_minute(|| {
        let (error, at, link, back) = tcp_until_failure(|_, _| false);
        assert!(error.contains("timed out after 200 ms without a heartbeat"), "{error}");
        assert_eq!((at, back.sent, link.beats), (LIMIT, 0, 0));
    });
}

#[test]
fn pp_heartbeat_after_the_deadline_does_not_revive() {
    within_a_minute(|| {
        // Two heartbeats, then silence until the deadline (the last heartbeat + the limit),
        // then heartbeats again: the one written at the deadline must not revive the request.
        let (error, at, link, back) = tcp_until_failure(|now, period| now <= 2 * period || now >= 2 * period + LIMIT);
        assert!(error.contains("without a heartbeat"), "{error}");
        assert_eq!(at, 2 * back.period + LIMIT);
        assert_eq!((back.sent, link.beats), (3, 2));
    });
}

#[test]
fn pp_wire_heartbeats_keep_a_long_request_alive_and_late_ones_are_ignored() {
    within_a_minute(|| {
        let (near, far) = pair();
        let mut link = front(near, LIMIT);
        let (mine, mut theirs) = pipes();
        link.fast = Some(Box::new(mine));
        let mut back = Back::manual(&mut link, far);
        let worker = std::thread::spawn(move || {
            // Request 1 on the wire: heartbeats on TCP, read by the wire wait's own periodic
            // link check (not forced: driven by the manual clock).
            let request = theirs.from.recv().unwrap();
            assert_eq!(word(&request, 4), 1);
            back.mark(Some(1));
            while back.now() < RUN {
                assert!(back.step(STEP, true), "the front stopped waiting at {:?}", back.now());
            }
            back.mark(None);
            back.step_with(Duration::ZERO, false, |_| { theirs.send(&[&plus_one(&request)]).unwrap(); 0 });
            // Request 2 (two rows: too big for the wire) on TCP gets no heartbeat of its own;
            // a late heartbeat of request 1 comes 3/4 of the limit in. It must not count as
            // live or move the deadline.
            let request = back.tcp_request(2);
            assert_eq!(word(&request, 4), 2);
            let start = back.now();
            while back.now() < start + 8 * LIMIT {
                let late = back.now() + STEP == start + 6 * STEP;
                if !back.step_with(STEP, false, |tcp| { if late { tcp.write_all(&working_head(1)).unwrap(); } late as usize }) {
                    return (back, start);
                }
            }
            back.answer(&plus_one(&request));
            (back, start)
        });
        let mut rows = vec![1.5f32; W];
        link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap();
        assert!(stopped_at(&link) > 4 * LIMIT);
        assert!(rows.iter().all(|&v| v == 2.5));
        let mut two = vec![0.25f32; 2 * W];
        let error = link.exchange(KIND_ROWS, &[[1, 8, 1, 2]], &mut two).expect_err("request 2 must fail");
        let at = stopped_at(&link);
        link.test_steps = None;
        let (back, start) = worker.join().unwrap();
        assert!(error.contains("without a heartbeat"), "{error}");
        assert_eq!(at, start + LIMIT, "the late heartbeat moved the deadline");
        // Request 1 sent one heartbeat per production interval; the planted one is late.
        let implied = (RUN.as_nanos() / back.period.as_nanos()) as u64;
        assert_eq!(back.sent, implied);
        assert!(link.late >= 1, "late {}", link.late);
        assert_eq!(link.beats + link.late, implied + 1, "beats {} late {}", link.beats, link.late);
        assert_eq!(link.carried, 1);
    });
}

/// A wire request whose back heartbeats while `beating(now)`; the front's error and the
/// manual time at which its wait ended.
fn wire_until_failure(beating: fn(Duration, Duration) -> bool) -> (String, Duration, Link, Back) {
    let (near, far) = pair();
    let mut link = front(near, LIMIT);
    let (mine, mut theirs) = pipes();
    link.fast = Some(Box::new(mine));
    let mut back = Back::manual(&mut link, far);
    let worker = std::thread::spawn(move || {
        let request = theirs.from.recv().unwrap();
        back.mark(Some(word(&request, 4)));
        while back.now() < 8 * LIMIT {
            let on = beating(back.now() + STEP, back.period);
            if !back.step(STEP, on) { return back; }
        }
        back.mark(None);
        back.step_with(Duration::ZERO, false, |_| { theirs.send(&[&plus_one(&request)]).unwrap(); 0 });
        back
    });
    let mut rows = vec![1.5f32; W];
    let error = link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).expect_err("the request must fail");
    let at = stopped_at(&link);
    link.test_steps = None;
    let back = worker.join().unwrap();
    (error, at, link, back)
}

#[test]
fn pp_wire_silent_back_fails_at_the_limit() {
    within_a_minute(|| {
        let (error, at, link, back) = wire_until_failure(|_, _| false);
        assert!(error.contains("timed out after 200 ms without a heartbeat"), "{error}");
        // The wire wait fails once more than the limit has passed: the first step after it.
        assert_eq!((at, back.sent, link.beats), (LIMIT + STEP, 0, 0));
    });
}

#[test]
fn pp_wire_heartbeat_after_the_deadline_does_not_revive() {
    within_a_minute(|| {
        // Two heartbeats, then silence past the deadline, then heartbeats again: one sits
        // unread when the wire wait's link check runs after the deadline.
        let (error, at, link, back) = wire_until_failure(|now, period| now <= 2 * period || now >= 2 * period + LIMIT + period);
        assert!(error.contains("without a heartbeat"), "{error}");
        assert!(at > 2 * back.period + LIMIT, "failed at {at:?}");
        assert!(back.sent > 2, "no heartbeat came after the deadline");
        assert_eq!(link.beats, 2, "a heartbeat after the deadline counted as live");
    });
}
