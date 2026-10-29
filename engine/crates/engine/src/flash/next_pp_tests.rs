use super::*;
use std::collections::VecDeque;
use std::net::{Shutdown, TcpListener};

fn prefetch_frame(id: u32, kind: u32, rows: &[f32]) -> Vec<u8> {
    let count = rows.len() / W;
    let mut frame = Vec::new();
    for value in [MAGIC_REQUEST, kind, count as u32, 1, id, 7, 64, id - 1, count as u32] {
        put(&mut frame, value);
    }
    frame.extend_from_slice(f32_bytes(rows));
    frame
}
fn prefetched(reader: &Prefetch) -> Result<Incoming> {
    let begun = Instant::now();
    loop {
        if let Some(request) = reader.poll(true)? { return Ok(request); }
        assert!(begun.elapsed() < Duration::from_secs(3), "prefetch did not produce a result");
    }
}

#[test]
fn pp_prefetch_receives_while_current_is_owned_and_reuses_three_row_buffers() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let (mut front, back) = pair();
    let reader = Prefetch::start(&back, 8, &STOP, Duration::from_secs(1)).unwrap();
    let rows: Vec<_> = (0..W).map(|i| f32::from_bits([0, 0x80000000, 0x7fc01234, i as u32][i % 4])).collect();
    let mut current = Vec::new();
    let mut addresses = std::collections::BTreeSet::new();
    for id in 1..=12 {
        front.write_all(&prefetch_frame(id, KIND_ROWS_KEEP, &rows)).unwrap();
        // Current remains owned by the serving thread until the next complete
        // request is received: its computation cannot be a prerequisite to I/O.
        let Incoming::Request(mut request) = prefetched(&reader).unwrap() else { panic!("early EOF") };
        assert_eq!((request.id, request.kind, request.count), (id, KIND_ROWS_KEEP, 1));
        assert_eq!(request.meta, [[7, 64, id - 1, 1]]);
        assert_eq!(f32_bytes(&request.rows), f32_bytes(&rows));
        addresses.insert(request.rows.as_ptr() as usize);
        std::mem::swap(&mut current, &mut request.rows);
        reader.recycle(request.rows).unwrap();
    }
    assert!(addresses.len() <= 3, "unexpected row allocations: {addresses:?}");
    front.shutdown(Shutdown::Write).unwrap();
    assert!(matches!(prefetched(&reader).unwrap(), Incoming::Closed));
}

#[test]
fn pp_prefetch_terminal_results_follow_completed_frames() {
    static STOP: AtomicBool = AtomicBool::new(false);
    for truncated in [false, true] {
        let (mut front, back) = pair();
        let reader = Prefetch::start(&back, 8, &STOP, Duration::from_secs(1)).unwrap();
        let rows = vec![f32::from_bits(0x80000000); W];
        front.write_all(&prefetch_frame(1, KIND_ROWS_KEEP, &rows)).unwrap();
        if truncated { front.write_all(&[1; 8]).unwrap(); }
        front.shutdown(Shutdown::Write).unwrap();
        let Incoming::Request(request) = prefetched(&reader).unwrap() else { panic!("lost complete frame") };
        assert_eq!(request.id, 1);
        assert_eq!(f32_bytes(&request.rows), f32_bytes(&rows));
        if truncated {
            assert!(prefetched(&reader).err().unwrap().contains("closed mid-message"));
        } else {
            assert!(matches!(prefetched(&reader).unwrap(), Incoming::Closed));
        }
    }
}

#[test]
fn pp_prefetch_rejects_bad_shape_before_body_and_bounds_partial_payload_wait() {
    static STOP: AtomicBool = AtomicBool::new(false);
    for invalid in [true, false] {
        let (mut front, back) = pair();
        let reader = Prefetch::start(&back, 8, &STOP, Duration::from_millis(100)).unwrap();
        let mut frame = prefetch_frame(1, KIND_ROWS, &vec![0.0; W]);
        if invalid {
            frame[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
            front.write_all(&frame[..REQUEST_HEAD]).unwrap();
        } else {
            front.write_all(&frame[..REQUEST_HEAD + 16 + 4]).unwrap();
        }
        let begun = Instant::now();
        let error = prefetched(&reader).err().unwrap();
        assert!(error.contains(if invalid { "header invalid" } else { "timed out" }), "{error}");
        assert!(begun.elapsed() < Duration::from_secs(1));
    }
}

#[test]
fn pp_prefetch_drop_joins_idle_full_queue_and_spare_wait_readers() {
    static STOP: AtomicBool = AtomicBool::new(false);
    for state in 0..3 {
        let (mut front, back) = pair();
        let reader = Prefetch::start(&back, 8, &STOP, Duration::from_secs(1)).unwrap();
        if state != 0 {
            let rows = vec![0.0; W];
            for id in 1..=3 { front.write_all(&prefetch_frame(id, KIND_ROWS_KEEP, &rows)).unwrap(); }
            // The reader has consumed two messages and cannot take a third:
            // one frame occupies ready, the other blocks its bounded send.
            let begun = Instant::now();
            let mut head = [0u8; REQUEST_HEAD];
            loop {
                if back.peek(&mut head).unwrap_or(0) == head.len() && word(&head, 4) == 3 { break; }
                assert!(begun.elapsed() < Duration::from_secs(2));
                std::thread::yield_now();
            }
            if state == 2 {
                // Consume both without recycling: the reader now waits for a
                // spare instead of a ready-channel slot. Drop must wake both.
                assert!(matches!(prefetched(&reader).unwrap(), Incoming::Request(_)));
                assert!(matches!(prefetched(&reader).unwrap(), Incoming::Request(_)));
            }
        }
        let begun = Instant::now();
        drop(reader);
        assert!(begun.elapsed() < Duration::from_secs(1), "reader failed to join in state {state}");
    }
}

#[test]
fn pp_prefetch_idle_reader_obeys_stop() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let (_front, back) = pair();
    let reader = Prefetch::start(&back, 8, &STOP, Duration::from_secs(1)).unwrap();
    STOP.store(true, Ordering::Relaxed);
    assert!(prefetched(&reader).err().unwrap().contains("stopped by signal"));
}

#[test]
fn pp_prefetch_stage_cancels_before_another_queued_step() {
    static STOP: AtomicBool = AtomicBool::new(false);
    static FRONT_STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(1), asynchronous: true, mtp: false, tp: false };
    let (front, back) = pair();
    let (posted, wait_posted) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut calls = 0;
        let result = serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
            64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| {
                calls += 1;
                wait_posted.recv_timeout(Duration::from_secs(2)).unwrap();
                STOP.store(true, Ordering::Relaxed);
                Ok(())
            });
        assert!(result.unwrap_err().contains("stopped by signal"));
        assert_eq!(calls, 1);
    });
    let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &FRONT_STOP, recipe).unwrap();
    for pos in 0..2 { link.post(&[[1, 64, pos, 1]], vec![0.0; W], None).unwrap(); }
    posted.send(()).unwrap();
    assert!(link.drain().is_err());
    link.poisoned = true;
    worker.join().unwrap();
}

#[test]
fn pp_prefetch_stage_rejects_wrong_tcp_exchange_id_before_step() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(1), asynchronous: false, mtp: false, tp: false };
    let (front, back) = pair();
    let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| panic!("wrong ID executed")));
    let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.stream.write_all(&prefetch_frame(2, KIND_ROWS, &vec![0.0; W])).unwrap();
    assert!(worker.join().unwrap().unwrap_err().contains("id 2, expected 1"));
    link.poisoned = true;
}

#[test]
fn tp_mail_tile_handoff_and_verify_kinds_reach_the_back_in_order() {
    use crate::flash_tp_switch::{HEAD, HEADS, KIND_HANDOFF, KIND_VERIFY as TP_VERIFY, MAIL, TILE};
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: true };
    let (front, back) = pair();
    let worker = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
            |kind, count, meta, _| { seen.push((kind, count, meta[0][2])); Ok(()) }).unwrap();
        seen
    });
    let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    // A pipeline tile (TILE) whose pass also posts mailbox meta (MAIL); then the handoff the
    // tile's chats owe; then a split verify pass by GDN heads (MAIL, HEADS), a plain step whose
    // greedy head splits (MAIL, HEAD) and a plain whole pass: each flag is one-shot and reaches
    // the back's step unchanged.
    (link.tp_tile, link.tp_mail) = (true, true);
    link.post(&[[1, 64, 0, 2]], vec![0.0; 2 * W], None).unwrap();
    assert!(!link.tp_mail);
    link.tp_tile = false;
    link.drain().unwrap();
    link.tp_handoff = true;
    link.post(&[[1, 64, 2, 1]], vec![0.0; W], None).unwrap();
    assert!(!link.tp_handoff);
    (link.tp_verify, link.tp_mail, link.tp_heads) = (true, true, true);
    link.post(&[[1, 64, 3, 1]], vec![0.0; W], None).unwrap();
    assert!(!link.tp_heads);
    (link.tp_mail, link.tp_head) = (true, true);
    link.post(&[[1, 64, 4, 1]], vec![0.0; W], None).unwrap();
    assert!(!link.tp_head);
    link.drain().unwrap();
    link.exchange(KIND_ROWS, &[[1, 64, 5, 1]], &mut vec![0.0; W]).unwrap();
    drop(link);
    assert_eq!(worker.join().unwrap(), [(KIND_ROWS_ASYNC | MAIL | TILE, 2, 0), (KIND_HANDOFF, 1, 2), (TP_VERIFY | MAIL | HEADS, 1, 3), (KIND_ROWS_ASYNC | MAIL | HEAD, 1, 4), (KIND_ROWS, 1, 5)]);
}

#[test]
fn tp_kinds_are_refused_by_a_pipeline_only_back_and_tp_hello_must_match() {
    use crate::flash_tp_switch::{KIND_HANDOFF, KIND_VERIFY as TP_VERIFY};
    static STOP: AtomicBool = AtomicBool::new(false);
    let plain = Recipe { limit: Duration::from_secs(1), asynchronous: true, mtp: true, tp: false };
    for kind in [KIND_HANDOFF, TP_VERIFY] {
        let (front, back) = pair();
        let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
            64, 8, 8, &STOP, plain, |_| Ok(()), || Ok(()), |_, _, _, _| panic!("split kind executed without the split")));
        let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, plain).unwrap();
        link.stream.write_all(&prefetch_frame(1, kind, &vec![0.0; W])).unwrap();
        assert!(worker.join().unwrap().unwrap_err().contains("outside the agreed recipe"));
        link.poisoned = true;
    }
    // A split front and a pipeline back (or the reverse) never reach a pass.
    for (front_tp, back_tp) in [(true, false), (false, true)] {
        let (front, back) = pair();
        let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
            64, 8, 8, &STOP, Recipe { tp: back_tp, ..plain }, |_| Ok(()), || Ok(()), |_, _, _, _| panic!("mismatched split ran")));
        let connected = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, Recipe { tp: front_tp, ..plain });
        assert!(connected.is_err());
        assert!(worker.join().unwrap().is_err());
    }
}

#[test]
fn pp_tensor_split_kinds_carry_mail_tile_and_handoff_in_order() {
    // The tensor split: a pipeline tile (TILE), a mailbox pass (MAIL), a state handoff and a split
    // verify reach the back in posting order, each one-shot flag consumed by its own post.
    use crate::flash_tp_switch::{KIND_HANDOFF, KIND_VERIFY as TP_VERIFY, MAIL, TILE};
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: true };
    let (front, back) = pair();
    let worker = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
            |kind, count, meta, _| { seen.push((kind, count, meta[0][2])); Ok(()) }).unwrap();
        seen
    });
    let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    assert!(link.tp);
    link.tp_tile = true;
    link.post(&[[1, 64, 0, 1]], vec![0.0; W], None).unwrap();
    link.tp_tile = false;
    link.tp_handoff = true;
    link.post(&[[1, 64, 1, 1]], vec![0.0; W], None).unwrap();
    assert!(!link.tp_handoff);
    link.tp_mail = true;
    link.post(&[[1, 64, 2, 1]], vec![0.0; W], None).unwrap();
    assert!(!link.tp_mail);
    link.tp_verify = true;
    link.post(&[[1, 64, 3, 1]], vec![0.0; W], None).unwrap();
    assert!(!link.tp_verify);
    link.drain().unwrap();
    link.exchange(KIND_ROWS, &[[1, 64, 4, 1]], &mut vec![0.0; W]).unwrap();
    drop(link);
    assert_eq!(worker.join().unwrap(), [(KIND_ROWS_ASYNC | TILE, 1, 0), (KIND_HANDOFF, 1, 1), (KIND_ROWS_ASYNC | MAIL, 1, 2),
        (TP_VERIFY, 1, 3), (KIND_ROWS, 1, 4)]);
}

#[test]
fn pp_tensor_split_kinds_are_refused_outside_the_agreed_recipe() {
    // A pipeline-only back never runs a handoff or a split verify, and a TP front cannot
    // connect to it at all: the hello carries the mode.
    static STOP: AtomicBool = AtomicBool::new(false);
    let plain = Recipe { limit: Duration::from_secs(1), asynchronous: true, mtp: true, tp: false };
    for kind in [crate::flash_tp_switch::KIND_HANDOFF, crate::flash_tp_switch::KIND_VERIFY] {
        let (front, back) = pair();
        let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
            64, 8, 8, &STOP, plain, |_| Ok(()), || Ok(()), |_, _, _, _| panic!("split kind executed")));
        let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, plain).unwrap();
        link.stream.write_all(&prefetch_frame(1, kind, &vec![0.0; W])).unwrap();
        assert!(worker.join().unwrap().unwrap_err().contains("outside the agreed recipe"));
        link.poisoned = true;
    }
    let (front, back) = pair();
    let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
        64, 8, 8, &STOP, plain, |_| Ok(()), || Ok(()), |_, _, _, _| panic!("mismatched mode executed")));
    let split = Recipe { tp: true, ..plain };
    assert!(Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, split).is_err());
    assert!(worker.join().unwrap().is_err());
}

#[test]
fn wire_setup_replaces_auth_deadline_on_two_three_and_four_ranks() {
    for ranks in 2..=4 {
        let mut edges: Vec<_> = (1..ranks).map(|_| {
            let (near, far) = pair();
            for stream in [&near, &far] {
                stream.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
                stream.set_write_timeout(Some(Duration::from_millis(20))).unwrap();
            }
            (Some(near), Some(far))
        }).collect();
        let mut front = edges[0].0.take().unwrap();
        let mut workers = Vec::new();
        for rank in 1..ranks {
            let mut incoming = edges[rank - 1].1.take().unwrap();
            let outgoing = edges.get_mut(rank).and_then(|edge| edge.0.take());
            workers.push(std::thread::spawn(move || -> Result<()> {
                negotiate_wire(&mut incoming, false, Duration::from_secs(2))?;
                if let Some(mut outgoing) = outgoing {
                    negotiate_wire(&mut outgoing, false, Duration::from_secs(2))?;
                }
                Ok(())
            }));
        }
        // The back is already authenticated while the front is still starting.
        std::thread::sleep(Duration::from_millis(100));
        let front_result = negotiate_wire(&mut front, false, Duration::from_secs(2));
        let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(front_result.is_ok() && results.iter().all(Result::is_ok),
            "{ranks} ranks: front={front_result:?}, workers={results:?}");
    }
}

/// Integer transforms model ordered layers without hiding NaNs or signed zero behind
/// floating-point equality. Returned rows must never undergo another stage transform.
fn layers(rows: &mut [f32], first: usize, end: usize) {
    for layer in first..end {
        for value in rows.iter_mut() {
            *value = f32::from_bits(value.to_bits().rotate_left(3).wrapping_add(layer as u32 + 1));
        }
    }
}
fn chain_case(ranks: usize, fast_wire: bool) {
    chain_capacity(ranks, fast_wire.then_some(REQUEST_HEAD + 16 + W * 4));
}

#[test]
fn pp_draft_transition_reclaims_warmed_worker_pools_before_rows_on_every_stage() {
    // Exercise the production receiver/forwarder with warmed per-rank storage.
    // The injected cleanup stands in for Backend::synchronize followed by the
    // model's decode.clear; it must run before the injected GPU step.
    for ranks in 2..=4 {
        for fast_wire in [false, true] {
            static STOP: AtomicBool = AtomicBool::new(false);
            let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
            let mut edges: Vec<_> = (1..ranks).map(|_| {
                let (near, far) = pair();
                let (nf, ff) = if fast_wire {
                    let (a, b) = pipes_with_capacity(REQUEST_HEAD + 16 + W * 4);
                    (Some(Box::new(a) as Box<dyn PpWire>), Some(Box::new(b) as Box<dyn PpWire>))
                } else { (None, None) };
                (Some((near, nf)), Some((far, ff)))
            }).collect();
            let (front, fast) = edges[0].0.take().unwrap();
            let mut workers = Vec::new();
            for rank in 1..ranks {
                let (stream, fast) = edges[rank - 1].1.take().unwrap();
                let next = edges.get_mut(rank).and_then(|e| e.0.take());
                workers.push(std::thread::spawn(move || {
                    let stage = Stage::new(rank, ranks, None).unwrap();
                    let pool = std::cell::RefCell::new(Vec::<Vec<u8>>::new());
                    let released = std::cell::Cell::new(0);
                    let mut calls = 0;
                    serve_stage(stream, fast, next, stage, 64, 8, 8, &STOP, recipe,
                        |_| Ok(()),
                        || {
                            assert!(!pool.borrow().is_empty(), "duplicate/spurious release");
                            pool.borrow_mut().clear();
                            released.set(released.get() + 1);
                            Ok(())
                        },
                        |kind, _, _, rows| {
                            if kind == KIND_POOLED { pool.borrow_mut().push(vec![7; 1024]); }
                            let expected_empty = [false, false, true, true, false, true, false, false, true, true][calls];
                            assert_eq!(pool.borrow().is_empty(), expected_empty, "stage {rank}, step {calls}");
                            layers(rows, stage.first, stage.end);
                            calls += 1;
                            Ok(())
                        }).unwrap();
                    assert_eq!((calls, released.get()), (10, 3));
                }));
            }
            let stage = Stage::new(0, ranks, None).unwrap();
            let mut link = Link::connect(front, fast, stage, 64, 8, 8, &STOP, recipe).unwrap();
            // A normal single row/fairness hold must retain the pool. Both a
            // verify transition and a filtered single-row transition release it.
            for (pos, (kind, release)) in [
                (KIND_POOLED, false), (KIND_ROWS, false), (KIND_VERIFY, true),
                (KIND_ROWS, false), (KIND_POOLED, false), (KIND_ROWS, true),
                (KIND_POOLED, false),
            ].into_iter().enumerate() {
                link.release_pool = release; // no front cache is required
                let mut rows = vec![f32::from_bits(0x8000_0000); W];
                let mut expected = rows.clone();
                layers(&mut rows, 0, stage.end);
                layers(&mut expected, 0, LAYERS);
                link.exchange(kind, &[[1, 64, pos as u32, 1]], &mut rows).unwrap();
                assert!(!link.release_pool);
                assert_eq!(f32_bytes(&rows), f32_bytes(&expected));
            }
            // Reclamation must follow a previously queued prompt, then be
            // consumed by the next queued request and forwarded through middles.
            link.post(&[[1, 64, 7, 1]], vec![0.0; W], None).unwrap();
            link.release_pool = true;
            link.post(&[[1, 64, 8, 1]], vec![0.0; W], None).unwrap();
            assert!(!link.release_pool);
            link.exchange(KIND_ROWS, &[[1, 64, 9, 1]], &mut vec![0.0; W]).unwrap();
            drop(link);
            for worker in workers { worker.join().unwrap(); }
        }
    }
}

#[test]
fn pp_failed_pool_reclamation_refuses_draft_execution_and_retains_pending_flag() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(1), asynchronous: false, mtp: true, tp: false };
    let (front, back) = pair();
    let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Err("pool synchronization failed".into()),
        |_, _, _, _| panic!("draft ran before pool synchronization")));
    let mut link = Link::connect(front, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.release_pool = true;
    let error = link.exchange(KIND_VERIFY, &[[1, 64, 0, 1]], &mut vec![0.0; W]).unwrap_err();
    assert!(error.contains("pool synchronization failed"), "{error}");
    assert!(link.release_pool);
    link.poisoned = true;
    assert!(worker.join().unwrap().unwrap_err().contains("pool synchronization failed"));
}

fn chain_capacity(ranks: usize, capacity: Option<usize>) {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
    let mut edges = Vec::new();
    for _ in 1..ranks {
        let (near, far) = pair();
        let (nf, ff) = if let Some(capacity) = capacity {
            let (a, b) = pipes_with_capacity(capacity);
            (Some(Box::new(a) as Box<dyn PpWire>), Some(Box::new(b) as Box<dyn PpWire>))
        } else { (None, None) };
        edges.push((Some((near, nf)), Some((far, ff))));
    }
    let (stream, fast) = edges[0].0.take().unwrap();
    let mut workers = Vec::new();
    for rank in 1..ranks {
        let (stream, fast) = edges[rank - 1].1.take().unwrap();
        let next = edges.get_mut(rank).and_then(|edge| edge.0.take());
        workers.push(std::thread::spawn(move || {
            let stage = Stage::new(rank, ranks, None).unwrap();
            let mut seen = Vec::new();
            let result = serve_stage(stream, fast, next, stage, 64, 8, 8, &STOP, recipe,
                |active| { assert_eq!(active, 8); Ok(()) }, || Ok(()),
                |kind, count, meta, rows| {
                    seen.push((kind, count, meta.to_vec()));
                    layers(rows, stage.first, stage.end);
                    Ok(())
                });
            result.unwrap();
            seen
        }));
    }
    let stage = Stage::new(0, ranks, None).unwrap();
    let mut link = Link::connect(stream, fast, stage, 64, 8, 8, &STOP, recipe).unwrap();
    let input = |n: usize| (0..n * W).map(|i| f32::from_bits([0, 0x80000000, 0x7fc01234, 0x7f800000, i as u32][i % 5])).collect::<Vec<_>>();
    let mut expected_requests = Vec::new();
    let mut expected_kept = Vec::new();
    let mut expected_second = Vec::new();
    for (kind, count, pos) in [(KIND_ROWS_ASYNC, 2, 0), (KIND_ROWS_KEEP, 3, 2), (KIND_ROWS_KEEP, 2, 4)] {
        let meta = if count == 3 { vec![[1, 64, pos, 2], [2, 64, 0, 1]] }
            else { vec![[1, 64, pos, count as u32]] };
        let mut rows = input(count);
        if kind == KIND_ROWS_KEEP {
            let mut all = rows.clone();
            layers(&mut all, 0, LAYERS);
            if count == 3 {
                expected_second = all.split_off(2 * W);
            }
            expected_kept.push(all);
        }
        layers(&mut rows, 0, stage.end);
        let keep = (kind == KIND_ROWS_KEEP).then(|| {
            let mut start = 0;
            meta.iter().map(|m| {
                let part = KeepPart { slot: m[0], pos: m[2] as usize, tokens: vec![7; m[3] as usize], start };
                start += m[3] as usize;
                part
            }).collect()
        });
        link.post(&meta, rows, keep).unwrap();
        expected_requests.push((kind, count, meta.to_vec()));
    }
    // Carrier depends on the complete reply, including five kept rows at the first step.
    for (kind, count, pos) in [(KIND_ROWS, 1, 6), (KIND_VERIFY, 3, 7), (KIND_ROWS, 8, 10), (KIND_POOLED, 1, 18)] {
        let meta = [[1, 64, pos, count as u32]];
        let mut rows = input(count);
        let mut expected = rows.clone();
        layers(&mut rows, 0, stage.end);
        layers(&mut expected, 0, LAYERS);
        link.exchange(kind, &meta, &mut rows).unwrap();
        assert_eq!(f32_bytes(&rows), f32_bytes(&expected), "ranks={ranks} kind={kind}");
        expected_requests.push((kind, count, meta.to_vec()));
    }
    let kept = link.take_kept(1);
    assert_eq!(kept.iter().map(|row| row.0).collect::<Vec<_>>(), [2, 4]);
    for (actual, expected) in kept.iter().zip(&expected_kept) {
        assert_eq!(f32_bytes(&actual.2), f32_bytes(expected));
    }
    let second = link.take_kept(2);
    assert_eq!(second.len(), 1);
    assert_eq!(f32_bytes(&second[0].2), f32_bytes(&expected_second));
    assert!(link.kept.is_empty());
    // Two ranks return kept rows in each KEEP ack, so the next exchange's reply is one tile.
    let kept_reply_rows = [6, 6, 1, 6, 6][ranks];
    let carried = capacity.map_or(0, |cap| [
        (REQUEST_HEAD + 16 + W * 4).max(REPLY_HEAD + kept_reply_rows * W * 4),
        REQUEST_HEAD + 16 + 3 * W * 4,
        REQUEST_HEAD + 16 + 8 * W * 4,
        REQUEST_HEAD + 16 + W * 4,
    ].into_iter().filter(|&need| need <= cap).count());
    assert_eq!(link.carried, carried as u64);
    drop(link); // clean half-close must cascade through every middle rank
    for worker in workers {
        assert_eq!(worker.join().unwrap(), expected_requests);
    }
}

fn read_pipe_request(back: &mut TcpStream, limit: Duration) -> Result<(u32, u32, Vec<f32>)> {
    let mut head = [0u8; REQUEST_HEAD];
    if !read_message(back, &mut head, None, limit, true)? {
        return Err("pipeline front closed".into());
    }
    let (kind, count, n, id) = (word(&head, 1), word(&head, 2) as usize, word(&head, 3) as usize, word(&head, 4));
    let mut meta = vec![0u8; n * 16];
    read_rest(back, &mut meta, None, limit)?;
    let mut rows = vec![0f32; count * W];
    read_rest(back, f32_bytes_mut(&mut rows), None, limit)?;
    Ok((kind, id, rows))
}

fn write_status(back: &mut TcpStream, id: u32, rows: Option<&[f32]>) {
    let mut reply = Vec::new();
    let len = rows.map(|r| r.len() * 4).unwrap_or(0) as u32;
    for v in [MAGIC_RESPONSE, 0, len, id] {
        put(&mut reply, v);
    }
    back.write_all(&reply).unwrap();
    if let Some(rows) = rows {
        back.write_all(f32_bytes(rows)).unwrap();
    }
}

#[test]
fn pp_final_tile_is_on_the_wire_before_the_queued_ack() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: false, tp: false };
    let (near, mut far) = pair();
    let back = std::thread::spawn(move || {
        far.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut hello = [0u8; 40];
        far.read_exact(&mut hello).unwrap();
        let mut ack = Vec::new();
        put(&mut ack, MAGIC_HELLO);
        put(&mut ack, 0);
        far.write_all(&ack).unwrap();
        let (kind, id, _) = read_pipe_request(&mut far, Duration::from_secs(2)).unwrap();
        assert_eq!(kind, KIND_ROWS_ASYNC);
        // No ack yet. The final tile has to arrive while this request is still open.
        let next = read_pipe_request(&mut far, Duration::from_millis(800));
        let (kind2, id2, rows2) = next.expect("final tile waited behind drain");
        assert_eq!((kind2, id2), (KIND_ROWS_FINAL, id.wrapping_add(1)));
        write_status(&mut far, id, None);
        let mut answered = rows2;
        for value in &mut answered {
            *value += 1.0;
        }
        write_status(&mut far, id2, Some(&answered));
    });
    let mut link = Link::connect(near, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.post(&[[1, 64, 0, 1]], vec![1.0; W], None).unwrap();
    let mut rows = vec![3.0; W];
    link.post_final(&[[1, 64, 1, 1]], rows.clone()).unwrap();
    link.finish_final(&mut rows).unwrap();
    assert!(rows.iter().all(|&value| value == 4.0));
    assert_eq!(link.pending, 0);
    back.join().unwrap();
}

#[test]
fn pp_final_tile_returns_kept_rows_on_tcp_and_on_the_wire() {
    for capacity in [None, Some(REPLY_HEAD + 3 * W * 4)] {
        static STOP: AtomicBool = AtomicBool::new(false);
        let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
        let (near, far) = pair();
        let (front_fast, back_fast) = match capacity {
            Some(capacity) => {
                let (a, b) = pipes_with_capacity(capacity);
                (Some(Box::new(a) as Box<dyn PpWire>), Some(Box::new(b) as Box<dyn PpWire>))
            }
            None => (None, None),
        };
        let worker = std::thread::spawn(move || {
            let mut seen = Vec::new();
            serve_stage(far, back_fast, None, Stage::new(1, 2, None).unwrap(), 64, 8, 8, &STOP, recipe,
                |_| Ok(()), || Ok(()),
                |kind, _, _, rows| {
                    seen.push(kind);
                    layers(rows, 24, LAYERS);
                    Ok(())
                }).unwrap();
            seen
        });
        let stage = Stage::new(0, 2, None).unwrap();
        let mut link = Link::connect(near, front_fast, stage, 64, 8, 8, &STOP, recipe).unwrap();
        let mut kept_rows = vec![1.0; 2 * W];
        let mut kept_expected = kept_rows.clone();
        layers(&mut kept_expected, 0, LAYERS);
        layers(&mut kept_rows, 0, stage.end);
        link.post(&[[1, 64, 0, 2]], kept_rows, Some(vec![KeepPart { slot: 1, pos: 0, tokens: vec![7, 7], start: 0 }])).unwrap();
        let mut final_rows = vec![2.0; W];
        let mut expected = final_rows.clone();
        layers(&mut final_rows, 0, stage.end);
        layers(&mut expected, 0, LAYERS);
        link.post_final(&[[1, 64, 2, 1]], final_rows).unwrap();
        let mut got = vec![0.0; W];
        link.finish_final(&mut got).unwrap();
        assert_eq!(f32_bytes(&got), f32_bytes(&expected), "capacity={capacity:?}");
        let kept = link.take_kept(1);
        assert_eq!(kept.len(), 1);
        assert_eq!(f32_bytes(&kept[0].2), f32_bytes(&kept_expected));
        assert_eq!(link.carried, u64::from(capacity.is_some()));
        drop(link);
        assert_eq!(worker.join().unwrap(), vec![KIND_ROWS_KEEP, KIND_ROWS_FINAL]);
    }
}

#[test]
fn pp_final_tile_crosses_a_middle_rank() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let tail = std::thread::spawn(move || {
        serve_stage(back_s, None, None, Stage::new(2, 3, None).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, rows| {
            layers(rows, 32, LAYERS);
            Ok(())
        }).unwrap();
    });
    let middle = std::thread::spawn(move || {
        serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, None).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, rows| {
            layers(rows, 16, 32);
            Ok(())
        }).unwrap();
    });
    let stage = Stage::new(0, 3, None).unwrap();
    let mut link = Link::connect(front_s, None, stage, 64, 8, 8, &STOP, recipe).unwrap();
    let mut kept_rows = vec![1.0; 2 * W];
    let mut kept_expected = kept_rows.clone();
    layers(&mut kept_expected, 0, LAYERS);
    layers(&mut kept_rows, 0, stage.end);
    link.post(&[[1, 64, 0, 2]], kept_rows, Some(vec![KeepPart { slot: 1, pos: 0, tokens: vec![7, 7], start: 0 }])).unwrap();
    let mut final_rows = vec![2.0; W];
    let mut expected = final_rows.clone();
    layers(&mut final_rows, 0, stage.end);
    layers(&mut expected, 0, LAYERS);
    link.post_final(&[[1, 64, 2, 1]], final_rows).unwrap();
    let mut got = vec![0.0; W];
    link.finish_final(&mut got).unwrap();
    assert_eq!(f32_bytes(&got), f32_bytes(&expected));
    let kept = link.take_kept(1);
    assert_eq!(f32_bytes(&kept[0].2), f32_bytes(&kept_expected));
    drop(link);
    middle.join().unwrap();
    tail.join().unwrap();
}

#[test]
fn pp_final_tile_across_a_middle_agrees_on_the_carrier_after_early_relay() {
    // Regression: a middle that consumed a kept ack early (rows moved into its
    // relay buffer) must still count those rows in the final reply's wire fit
    // test, or it picks the wire while the front waits on TCP. The front-facing
    // wire fits 80 rows: the final tile alone is 32, kept + final is 96, so the
    // two ends agree on TCP only if relayed rows are counted. The tail sleeps
    // on its first tile while large filler tiles fill the middle's sender queue,
    // which is what makes the middle peek and relay the kept ack early.
    static STOP: AtomicBool = AtomicBool::new(false);
    static SLEPT: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(10), asynchronous: true, mtp: true, tp: false };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let (front_fast, middle_fast) = pipes_with_capacity(REPLY_HEAD + 80 * W * 4);
    let tail = std::thread::spawn(move || {
        serve_stage(back_s, None, None, Stage::new(2, 3, None).unwrap(), 64, 8, 512, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, rows| {
            if !SLEPT.swap(true, std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(2000));
            }
            layers(rows, 32, LAYERS);
            Ok(())
        }).unwrap();
    });
    let middle = std::thread::spawn(move || {
        serve_stage(middle_in, Some(Box::new(middle_fast) as Box<dyn PpWire>), Some((middle_out, None)), Stage::new(1, 3, None).unwrap(), 64, 8, 512, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, rows| {
            layers(rows, 16, 32);
            Ok(())
        }).unwrap();
    });
    let stage = Stage::new(0, 3, None).unwrap();
    let mut link = Link::connect(front_s, Some(Box::new(front_fast) as Box<dyn PpWire>), stage, 64, 8, 512, &STOP, recipe).unwrap();
    let mut kept_rows = vec![1.0; 64 * W];
    let mut kept_expected = kept_rows.clone();
    layers(&mut kept_expected, 0, LAYERS);
    layers(&mut kept_rows, 0, stage.end);
    link.post(&[[1, 64, 0, 64]], kept_rows, Some(vec![KeepPart { slot: 1, pos: 0, tokens: vec![7; 64], start: 0 }])).unwrap();
    for i in 0..12u32 {
        link.post(&[[1, 64, 64 + 512 * i, 512]], vec![3.0; 512 * W], None).unwrap();
    }
    let mut final_rows = vec![2.0; 32 * W];
    let mut expected = final_rows.clone();
    layers(&mut final_rows, 0, stage.end);
    layers(&mut expected, 0, LAYERS);
    link.post_final(&[[1, 64, 64 + 12 * 512, 32]], final_rows).unwrap();
    let mut got = vec![0.0; 32 * W];
    link.finish_final(&mut got).unwrap();
    assert_eq!(f32_bytes(&got), f32_bytes(&expected));
    let kept = link.take_kept(1);
    assert_eq!(kept.len(), 1);
    assert_eq!(f32_bytes(&kept[0].2), f32_bytes(&kept_expected));
    drop(link);
    middle.join().unwrap();
    tail.join().unwrap();
}

#[test]
fn pp_chain_two_ranks_exact_forward_return_and_kept_rows() {
    chain_case(2, false);
    chain_case(2, true);
}
#[test]
fn pp_chain_three_ranks_exact_forward_return_and_kept_rows() {
    chain_case(3, false);
    chain_case(3, true);
}
#[test]
fn pp_chain_four_ranks_exact_forward_return_and_kept_rows() {
    chain_case(4, false);
    chain_case(4, true);
}
#[test]
fn pp_stage_geometry_covers_each_layer_once_and_keeps_ple_on_front() {
    for ranks in 2..=24 {
        let stages = (0..ranks).map(|r| Stage::new(r, ranks, None).unwrap()).collect::<Vec<_>>();
        assert!(stages[0].end >= 2);
        assert_eq!(stages.iter().flat_map(|s| s.first..s.end).collect::<Vec<_>>(), (0..48).collect::<Vec<_>>());
        assert!(!pool_advances_position(stages[0].end, true));
        for stage in &stages[1..] { assert!(pool_advances_position(stage.end, false)); }
    }
    assert_eq!(Stage::new(1, 4, None).unwrap(), Stage { rank: 1, ranks: 4, first: 12, end: 24 });
    assert_eq!(Stage::new(1, 2, Some(17)).unwrap().first, 17);
    assert!(Stage::new(1, 4, Some(24)).is_err());
    for ranks in [0, 1, 25, usize::MAX] { assert!(Stage::new(0, ranks, None).is_err()); }
    assert!(Stage::new(4, 4, None).is_err());
    assert!(!pool_advances_position(1, true)); // frontend's layer-zero graph
    assert!(pool_advances_position(48, true)); // world-one tail / front final head
}

#[test]
fn pp_chain_tail_failure_reaches_front_without_a_success_reply() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(1), asynchronous: false, mtp: false, tp: false };
    let (front, middle_in) = pair();
    let (middle_out, back) = pair();
    let tail = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(2, 3, None).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| Err("tail rejected a row".into())));
    let middle = std::thread::spawn(move || serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, None).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| Ok(())));
    let mut link = Link::connect(front, None, Stage::new(0, 3, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    let mut rows = vec![0.0; W];
    let error = link.exchange(KIND_POOLED, &[[1, 64, 0, 1]], &mut rows).unwrap_err();
    assert!(error.contains("tail rejected a row"), "{error}");
    link.poisoned = true;
    assert!(tail.join().unwrap().unwrap_err().contains("tail rejected a row"));
    assert!(middle.join().unwrap().unwrap_err().contains("tail rejected a row"));
}

#[test]
fn pp_chain_mismatched_world_fails_admission_before_any_work() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(1), asynchronous: false, mtp: false, tp: false };
    let (front, back) = pair();
    let worker = std::thread::spawn(move || serve_stage(back, None, None, Stage::new(1, 2, None).unwrap(),
        64, 8, 8, &STOP, recipe, |_| panic!("must not admit"), || panic!("must not release"), |_, _, _, _| panic!("must not execute")));
    assert!(Link::connect(front, None, Stage::new(0, 4, None).unwrap(), 64, 8, 8, &STOP, recipe).is_err());
    assert!(worker.join().unwrap().is_err());
}

#[test]
fn pp_middle_enforces_downstream_timeout_while_heartbeating_upstream() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_millis(200), asynchronous: false, mtp: false, tp: false };
    let (front, middle_in) = pair();
    let (middle_out, mut tail) = pair();
    let silent = std::thread::spawn(move || {
        let mut hello = [0u8; 40];
        tail.read_exact(&mut hello).unwrap();
        let mut ack = Vec::new();
        put(&mut ack, MAGIC_HELLO); put(&mut ack, 0);
        tail.write_all(&ack).unwrap();
        let mut request = vec![0u8; REQUEST_HEAD + 16 + W * 4];
        read_rest(&mut tail, &mut request, None, Duration::from_secs(1)).unwrap();
        std::thread::sleep(Duration::from_millis(600));
    });
    let middle = std::thread::spawn(move || serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, None).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| {
            // Local work can outlast a hop's timeout when it continues heartbeating.
            std::thread::sleep(Duration::from_millis(300));
            Ok(())
        }));
    let mut link = Link::connect(front, None, Stage::new(0, 3, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    let started = Instant::now();
    let mut rows = vec![1.5; W];
    let error = link.exchange(KIND_POOLED, &[[1, 64, 0, 1]], &mut rows).unwrap_err();
    assert!(error.contains("pipeline back failed") && error.contains("timed out"), "{error}");
    assert!(link.beats > 0);
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(rows.iter().all(|&x| x == 1.5));
    link.poisoned = true;
    assert!(middle.join().unwrap().unwrap_err().contains("timed out"));
    silent.join().unwrap();
}

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let near = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (far, _) = listener.accept().unwrap();
    for s in [&near, &far] {
        s.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
    }
    (near, far)
}
fn front(stream: TcpStream, limit: Duration) -> Link {
    static DOWN: AtomicBool = AtomicBool::new(false);
    Link {
        release_pool: false, released: 0,
        cache: cache::Tickets::default(),
        stream,
        split: LAYERS / 2,
        buffer: Vec::new(),
        exchanges: 0,
        bytes: 0,
        wait: Duration::ZERO,
        queue: None,
        free: None,
        pending: 0,
        posted: 0,
        fast: None,
        message: Vec::new(),
        carried: 0,
        down: &DOWN,
        poisoned: false, tp: false, tp_verify: false, tp_mail: false, tp_handoff: false, tp_tile: false, tp_heads: false, tp_head: false, tp_seq: false,
        sent: 0,
        replied: 0,
        limit,
        beats: 0,
        late: 0,
        floor: 1,
        mtp: false,
        kept: Vec::new(),
        owed: 0,
        relay_kept: None,
        acks: VecDeque::new(),
        inline_kept: false,
    }
}
/// A back that reads one pooled one-row request and answers it with `id` and rows + 1.
fn answer(back: &mut TcpStream, id: Option<u32>) -> u32 {
    let mut request = vec![0u8; REQUEST_HEAD + 16 + W * 4];
    assert!(read_message(back, &mut request, None, Duration::from_secs(5), true).unwrap());
    let sent = word(&request, 4);
    let mut reply = Vec::new();
    for v in [MAGIC_RESPONSE, 0, (W * 4) as u32, id.unwrap_or(sent)] {
        put(&mut reply, v);
    }
    let rows: Vec<f32> = request[REQUEST_HEAD + 16..].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) + 1.0).collect();
    reply.extend_from_slice(f32_bytes(&rows));
    back.write_all(&reply).unwrap();
    sent
}

#[test]
fn pp_clean_close_between_messages_is_the_end() {
    let (near, mut far) = pair();
    drop(near);
    let mut head = [0u8; REQUEST_HEAD];
    assert!(!read_message(&mut far, &mut head, None, Duration::from_millis(100), true).unwrap());
}

#[test]
fn pp_truncated_message_is_an_error() {
    // A whole header, then a write-half-close before the payload: never a partial message.
    let (mut near, mut far) = pair();
    near.write_all(&[7u8; REQUEST_HEAD]).unwrap();
    near.write_all(&[1u8; 10]).unwrap();
    near.shutdown(Shutdown::Write).unwrap();
    let mut head = [0u8; REQUEST_HEAD];
    assert!(read_message(&mut far, &mut head, None, Duration::from_secs(1), true).unwrap());
    let mut rows = [0u8; 64];
    let e = read_rest(&mut far, &mut rows, None, Duration::from_secs(1)).unwrap_err();
    assert!(e.contains("closed mid-message (10 of 64 bytes)"), "{e}");
    // Closed right at the payload's first byte: also an error (it used to read as Ok).
    let (mut near, mut far) = pair();
    near.write_all(&[7u8; REQUEST_HEAD]).unwrap();
    near.shutdown(Shutdown::Write).unwrap();
    assert!(read_message(&mut far, &mut head, None, Duration::from_secs(1), true).unwrap());
    let e = read_rest(&mut far, &mut rows, None, Duration::from_secs(1)).unwrap_err();
    assert!(e.contains("closed mid-message (0 of 64 bytes)"), "{e}");
}

#[test]
fn pp_stall_mid_message_times_out_within_the_limit() {
    let (mut near, mut far) = pair();
    near.write_all(&[1u8; 8]).unwrap();
    let (limit, started) = (Duration::from_millis(150), Instant::now());
    let mut head = [0u8; REQUEST_HEAD];
    let e = read_message(&mut far, &mut head, None, limit, true).unwrap_err();
    assert!(e.contains("timed out after 150 ms (8 of 20 bytes)"), "{e}");
    assert!(started.elapsed() < limit + Duration::from_secs(1));
}

#[test]
fn pp_idle_wait_has_no_limit_but_obeys_stop() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let (_near, mut far) = pair();
    let started = Instant::now();
    let signal = std::thread::spawn(|| {
        std::thread::sleep(Duration::from_millis(300));
        STOP.store(true, Ordering::Relaxed);
    });
    let mut head = [0u8; REQUEST_HEAD];
    let e = read_message(&mut far, &mut head, Some(&STOP), Duration::from_millis(50), true).unwrap_err();
    assert!(e.contains("stopped by signal"), "{e}");
    assert!(started.elapsed() >= Duration::from_millis(300));
    signal.join().unwrap();
}

#[test]
fn pp_reply_must_answer_its_own_request() {
    let (near, mut far) = pair();
    let mut link = front(near, Duration::from_millis(300));
    let back = std::thread::spawn(move || {
        assert_eq!(answer(&mut far, None), 1);
        // Request 2 gets no answer in time; its late reply then meets request 3.
        let mut late = vec![0u8; REQUEST_HEAD + 16 + W * 4];
        assert!(read_message(&mut far, &mut late, None, Duration::from_secs(5), true).unwrap());
        std::thread::sleep(Duration::from_millis(600));
        let mut reply = Vec::new();
        for v in [MAGIC_RESPONSE, 0, (W * 4) as u32, 2] {
            put(&mut reply, v);
        }
        reply.extend_from_slice(&late[REQUEST_HEAD + 16..]);
        far.write_all(&reply).unwrap();
        let mut next = vec![0u8; REQUEST_HEAD + 16 + W * 4];
        assert!(read_message(&mut far, &mut next, None, Duration::from_secs(5), true).unwrap());
        word(&next, 4)
    });
    let mut rows = vec![1.5f32; W];
    link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap();
    assert!(rows.iter().all(|&v| v == 2.5));
    let started = Instant::now();
    let e = link.exchange(KIND_POOLED, &[[1, 8, 1, 1]], &mut rows).unwrap_err();
    assert!(e.contains("timed out"), "{e}");
    assert!(started.elapsed() < Duration::from_millis(300) + Duration::from_secs(1));
    std::thread::sleep(Duration::from_millis(500));
    let e = link.exchange(KIND_POOLED, &[[1, 8, 2, 1]], &mut rows).unwrap_err();
    assert!(e.contains("out of order: id 2, expected 3"), "{e}");
    assert_eq!(back.join().unwrap(), 3);
}

/// A back that reads one pooled request, heartbeats every `beat` for `work`, then
/// answers it (rows + 1) with the request's id.
fn slow_answer(back: &mut TcpStream, beat: Option<Duration>, work: Duration, beat_id: Option<u32>) {
    let mut request = vec![0u8; REQUEST_HEAD + 16 + W * 4];
    assert!(read_message(back, &mut request, None, Duration::from_secs(5), true).unwrap());
    let id = word(&request, 4);
    let started = Instant::now();
    while started.elapsed() < work {
        match beat {
            Some(period) => {
                std::thread::sleep(period);
                if back.write_all(&working_head(beat_id.unwrap_or(id))).is_err() {
                    return;
                }
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    let mut reply = Vec::new();
    for v in [MAGIC_RESPONSE, 0, (W * 4) as u32, id] {
        put(&mut reply, v);
    }
    let rows: Vec<f32> = request[REQUEST_HEAD + 16..].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) + 1.0).collect();
    reply.extend_from_slice(f32_bytes(&rows));
    let _ = back.write_all(&reply);
}

#[test]
fn pp_heartbeats_keep_a_long_request_alive() {
    // The back works 4x the limit but heartbeats: the reply arrives and is right.
    let (near, mut far) = pair();
    let limit = Duration::from_millis(200);
    let mut link = front(near, limit);
    let back = std::thread::spawn(move || slow_answer(&mut far, Some(beat_period(limit)), 4 * limit, None));
    let mut rows = vec![1.5f32; W];
    let started = Instant::now();
    link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap();
    assert!(started.elapsed() >= 4 * limit);
    assert!(rows.iter().all(|&v| v == 2.5));
    assert!(link.beats >= 8, "beats {}", link.beats);
    back.join().unwrap();
}

#[test]
fn pp_silent_back_fails_within_the_limit() {
    // No heartbeat (a stopped back): the reply wait ends in an error within the limit.
    let (near, mut far) = pair();
    let limit = Duration::from_millis(200);
    let mut link = front(near, limit);
    let back = std::thread::spawn(move || slow_answer(&mut far, None, Duration::from_secs(2), None));
    let mut rows = vec![1.5f32; W];
    let started = Instant::now();
    let e = link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap_err();
    assert!(e.contains("timed out after 200 ms"), "{e}");
    assert!(started.elapsed() < limit + Duration::from_millis(500));
    assert!(rows.iter().all(|&v| v == 1.5));
    back.join().unwrap();
}

#[test]
fn pp_heartbeat_for_another_request_is_an_error() {
    let (near, mut far) = pair();
    let mut link = front(near, Duration::from_millis(300));
    let back = std::thread::spawn(move || slow_answer(&mut far, Some(Duration::from_millis(50)), Duration::from_millis(100), Some(7)));
    let mut rows = vec![1.5f32; W];
    let e = link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap_err();
    assert!(e.contains("heartbeat out of order: id 7, expected 1"), "{e}");
    back.join().unwrap();
}

/// An in-process message wire (the RDMA wire's contract: whole messages, polled).
struct Pipe {
    capacity: usize,
    to: mpsc::Sender<Vec<u8>>,
    from: mpsc::Receiver<Vec<u8>>,
}
impl PpWire for Pipe {
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn send(&mut self, parts: &[&[u8]]) -> Result<()> {
        if parts.iter().map(|p| p.len()).sum::<usize>() > self.capacity {
            return Err("test wire send exceeds capacity".into());
        }
        self.to.send(parts.concat()).map_err(|e| e.to_string())
    }
    fn try_recv(&mut self, out: &mut Vec<u8>) -> Result<bool> {
        match self.from.try_recv() {
            Ok(m) => {
                *out = m;
                Ok(true)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(false),
            Err(e) => Err(e.to_string()),
        }
    }
}
fn pipes() -> (Pipe, Pipe) { pipes_with_capacity(REQUEST_HEAD + 16 + W * 4) }
fn pipes_with_capacity(capacity: usize) -> (Pipe, Pipe) {
    let (a, b) = (mpsc::channel(), mpsc::channel());
    (Pipe { capacity, to: a.0, from: b.1 }, Pipe { capacity, to: b.0, from: a.1 })
}
/// A back that takes one wire request, heartbeats on `tcp` every `beat` for `work`, then
/// answers on the wire (rows + 1); returns the request id.
fn wire_answer(wire: &mut Pipe, tcp: &mut TcpStream, beat: Option<Duration>, work: Duration) -> u32 {
    let request = wire.from.recv_timeout(Duration::from_secs(5)).unwrap();
    let id = word(&request, 4);
    let started = Instant::now();
    while started.elapsed() < work {
        match beat {
            Some(period) => {
                std::thread::sleep(period);
                tcp.write_all(&working_head(id)).unwrap();
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    let mut reply = Vec::new();
    for v in [MAGIC_RESPONSE, 0, (W * 4) as u32, id] {
        put(&mut reply, v);
    }
    let rows: Vec<f32> = request[REQUEST_HEAD + 16..].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) + 1.0).collect();
    reply.extend_from_slice(f32_bytes(&rows));
    wire.send(&[&reply]).unwrap();
    id
}

#[test]
fn pp_wire_reply_clock_starts_after_the_queued_acks() {
    // A queued tile runs 3x the limit (heartbeating), then a one-row exchange goes on the
    // wire and is answered at once: the wire wait must not count the drain's time.
    let (near, mut far) = pair();
    let limit = Duration::from_millis(200);
    let mut link = front(near, limit);
    let (mine, mut theirs) = pipes();
    link.fast = Some(Box::new(mine));
    (link.sent, link.pending) = (1, 1);
    link.acks.push_back(None);
    let back = std::thread::spawn(move || {
        let started = Instant::now();
        while started.elapsed() < 3 * limit {
            std::thread::sleep(beat_period(limit));
            far.write_all(&working_head(1)).unwrap();
        }
        let mut ack = Vec::new();
        for v in [MAGIC_RESPONSE, 0, 0, 1] {
            put(&mut ack, v);
        }
        far.write_all(&ack).unwrap();
        wire_answer(&mut theirs, &mut far, None, Duration::from_millis(20))
    });
    let mut rows = vec![1.5f32; W];
    let started = Instant::now();
    link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap();
    assert!(started.elapsed() >= 3 * limit);
    assert!(rows.iter().all(|&v| v == 2.5));
    assert_eq!((link.carried, link.pending), (1, 0));
    assert_eq!(back.join().unwrap(), 2);
}

#[test]
fn pp_wire_heartbeats_keep_a_long_request_alive_and_late_ones_are_ignored() {
    let (near, mut far) = pair();
    let limit = Duration::from_millis(200);
    let mut link = front(near, limit);
    let (mine, mut theirs) = pipes();
    link.fast = Some(Box::new(mine));
    let back = std::thread::spawn(move || {
        // Request 1 on the wire: 4x the limit, heartbeating on TCP; then a heartbeat of
        // request 1 that the wire reply overtook; then request 2 (two rows: too big for
        // the wire) on TCP.
        assert_eq!(wire_answer(&mut theirs, &mut far, Some(beat_period(limit)), 4 * limit), 1);
        std::thread::sleep(Duration::from_millis(100));
        far.write_all(&working_head(1)).unwrap();
        let mut request = vec![0u8; REQUEST_HEAD + 16 + 2 * W * 4];
        assert!(read_message(&mut far, &mut request, None, Duration::from_secs(5), true).unwrap());
        let id = word(&request, 4);
        let mut reply = Vec::new();
        for v in [MAGIC_RESPONSE, 0, (2 * W * 4) as u32, id] {
            put(&mut reply, v);
        }
        reply.extend_from_slice(&request[REQUEST_HEAD + 16..]);
        far.write_all(&reply).unwrap();
        id
    });
    let mut rows = vec![1.5f32; W];
    let started = Instant::now();
    link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap();
    assert!(started.elapsed() >= 4 * limit);
    assert!(rows.iter().all(|&v| v == 2.5));
    assert!(link.beats >= 4, "beats {}", link.beats);
    let mut two = vec![0.25f32; 2 * W];
    link.exchange(KIND_ROWS, &[[1, 8, 1, 2]], &mut two).unwrap();
    assert!(two.iter().all(|&v| v == 0.25));
    // The planted late heartbeat, plus any the wire reply overtook while the front had
    // not read the TCP link yet (it looks every 50 ms).
    assert!(link.carried == 1 && link.late >= 1, "carried {} late {}", link.carried, link.late);
    assert_eq!(back.join().unwrap(), 2);
}

#[test]
fn pp_wire_silent_back_fails_within_the_limit() {
    let (near, mut far) = pair();
    let limit = Duration::from_millis(200);
    let mut link = front(near, limit);
    let (mine, mut theirs) = pipes();
    link.fast = Some(Box::new(mine));
    let back = std::thread::spawn(move || wire_answer(&mut theirs, &mut far, None, Duration::from_secs(1)));
    let mut rows = vec![1.5f32; W];
    let started = Instant::now();
    let e = link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap_err();
    assert!(e.contains("timed out after 200 ms without a heartbeat"), "{e}");
    assert!(started.elapsed() < limit + Duration::from_millis(300));
    assert!(rows.iter().all(|&v| v == 1.5));
    back.join().unwrap();
}

#[test]
fn pp_front_close_with_an_unread_heartbeat_is_a_clean_end() {
    // A late heartbeat the front never read must not turn its close into a reset.
    let (near, mut far) = pair();
    let link = front(near, Duration::from_millis(500));
    far.write_all(&working_head(1)).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let closer = std::thread::spawn(move || drop(link));
    let mut head = [0u8; REQUEST_HEAD];
    assert!(!read_message(&mut far, &mut head, None, Duration::from_secs(2), true).unwrap());
    drop(far);
    closer.join().unwrap();
}

#[test]
fn pp_kept_rows_return_ahead_of_the_next_reply() {
    // An abandoned tile still owes wire rows. Retained earlier rows must not be read
    // again, and the new shared/full tiles return ahead of the synchronous reply.
    let (near, mut far) = pair();
    let mut link = front(near, Duration::from_secs(2));
    let part = |slot, pos, n, start| KeepPart { slot, pos, tokens: vec![7; n], start };
    link.kept.push(KeptTile { parts: vec![part(3, 0, 1, 0)], rows: 1, data: Some(vec![8.0; W]) });
    link.kept.push(KeptTile { parts: vec![part(4, 0, 1, 0)], rows: 1, data: None });
    link.kept.push(KeptTile { parts: vec![part(1, 0, 2, 0), part(2, 0, 1, 2)], rows: 3, data: None });
    link.kept.push(KeptTile { parts: vec![part(1, 2, 2, 0)], rows: 2, data: None });
    link.owed = 6;
    link.drop_kept(4);
    assert_eq!(link.kept.len(), 4);
    let back = std::thread::spawn(move || {
        let mut request = vec![0u8; REQUEST_HEAD + 16 + W * 4];
        assert!(read_message(&mut far, &mut request, None, Duration::from_secs(5), true).unwrap());
        let mut rows: Vec<f32> = (0..6 * W).map(|i| (i / W) as f32).collect();
        rows.extend((0..W).map(|_| 9.0f32));
        let mut reply = Vec::new();
        for v in [MAGIC_RESPONSE, 0, (7 * W * 4) as u32, word(&request, 4)] {
            put(&mut reply, v);
        }
        reply.extend_from_slice(f32_bytes(&rows));
        far.write_all(&reply).unwrap();
    });
    let mut rows = vec![1.5f32; W];
    link.exchange(KIND_POOLED, &[[1, 8, 4, 1]], &mut rows).unwrap();
    back.join().unwrap();
    assert!(rows.iter().all(|&v| v == 9.0));
    assert_eq!(link.owed, 0);
    assert_eq!(link.kept.len(), 3);
    let whole = link.kept[2].data.as_ref().unwrap().as_ptr();
    let shared = link.kept[1].data.as_ref().unwrap().as_ptr();
    let one = link.take_kept(1);
    assert_eq!(one.iter().map(|k| k.0).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(one[0].2[..].iter().step_by(W).copied().collect::<Vec<_>>(), vec![1.0, 2.0]);
    assert_eq!(one[1].2[..].iter().step_by(W).copied().collect::<Vec<_>>(), vec![4.0, 5.0]);
    assert_eq!(one[1].2.as_ptr(), whole, "a whole tile transfers its allocation");
    assert_eq!(link.kept[1].data.as_ref().unwrap().as_ptr(), shared, "another slot still owns the shared tile");
    let two = link.take_kept(2);
    assert_eq!(two.len(), 1);
    assert_eq!(two[0].2[..].iter().step_by(W).copied().collect::<Vec<_>>(), vec![3.0]);
    assert_eq!(two[0].2.len(), W, "a sole remaining partial span is not the whole tile");
    assert_eq!(link.take_kept(3)[0].2, vec![8.0; W]);
    assert!(link.kept.is_empty());
}

#[test]
fn pp_kept_row_accounting_fails_before_payload_is_consumed() {
    for counts in [vec![1], vec![3], vec![usize::MAX, 1]] {
        let (near, mut far) = pair();
        let mut link = front(near, Duration::from_secs(2));
        link.owed = 2;
        link.kept = counts.into_iter().map(|rows| KeptTile { parts: Vec::new(), rows, data: None }).collect();
        let back = std::thread::spawn(move || {
            let mut request = vec![0u8; REQUEST_HEAD + 16 + W * 4];
            far.read_exact(&mut request).unwrap();
            let mut reply = Vec::new();
            for v in [MAGIC_RESPONSE, 0, (3 * W * 4) as u32, word(&request, 4), 0x7fc0_1234] {
                put(&mut reply, v);
            }
            far.write_all(&reply).unwrap();
        });
        let mut rows = vec![1.5; W];
        assert_eq!(link.exchange(KIND_POOLED, &[[1, 8, 0, 1]], &mut rows).unwrap_err(), "pipeline kept row accounting mismatch");
        let mut untouched = [0; 4];
        link.stream.read_exact(&mut untouched).unwrap();
        assert_eq!(u32::from_le_bytes(untouched), 0x7fc0_1234);
        assert_eq!(link.owed, 2);
        assert!(link.kept.iter().all(|t| t.data.is_none()));
        assert_eq!(rows, vec![1.5; W]);
        back.join().unwrap();
    }
}

#[test]
fn pp_kept_rows_are_bounded_while_retained_and_released_with_their_chat() {
    // keep_rows counts every retained tile (owed or back on the front, not yet observed);
    // a chat that ends drops its spans, and a tile leaves once no chat holds a span in it.
    let (near, _far) = pair();
    let mut link = front(near, Duration::from_secs(2));
    let part = |slot, pos, n, start| KeepPart { slot, pos, tokens: vec![7; n], start };
    let big = link.keep_rows() - 200;
    link.kept.push(KeptTile { parts: vec![part(1, 0, 64, 0), part(2, 0, 64, 64)], rows: big, data: Some(Vec::new()) });
    link.kept.push(KeptTile { parts: vec![part(3, 0, 64, 0)], rows: 64, data: None });
    assert!(link.has_kept(1) && link.has_kept(2) && link.has_kept(3) && !link.has_kept(4));
    assert!(link.keep_room(136) && !link.keep_room(137));
    assert!(link.post(&[[4, 8, 0, 1]], vec![0.0; 137 * W], Some(vec![part(4, 0, 137, 0)])).is_err());
    link.drop_kept(1);
    assert!(!link.has_kept(1) && link.has_kept(2) && !link.keep_room(137));
    link.drop_kept(2);
    // The first tile left with its last span; the owed tile stays until its rows come back.
    assert_eq!(link.kept.len(), 1);
    assert!(link.keep_room(link.keep_rows() - 64) && !link.keep_room(link.keep_rows() - 63));
    link.drop_kept(3);
    assert_eq!(link.kept.len(), 1);
    assert!(!link.has_kept(3));
}

#[test]
fn pp_taking_kept_rows_for_room_takes_the_oldest_returned_tiles_only() {
    // Past keep_rows the drafter takes the oldest rows that came back, only until the next tile
    // fits; later returned tiles wait for the final tile and the owed tile keeps counting.
    let (near, _far) = pair();
    let mut link = front(near, Duration::from_secs(2));
    let part = |pos, n| KeepPart { slot: 1, pos, tokens: vec![7; n], start: 0 };
    let half = link.keep_rows() / 2 - 64;
    link.kept.push(KeptTile { parts: vec![part(0, 64)], rows: half, data: Some(vec![1.0; 64 * W]) });
    link.kept.push(KeptTile { parts: vec![part(64, 64)], rows: half, data: Some(vec![2.0; 64 * W]) });
    link.kept.push(KeptTile { parts: vec![part(128, 64)], rows: 128, data: None });
    let room = link.keep_rows() / 2 - 64;
    assert!(!link.keep_room(room));
    let taken = link.take_kept_room(1, Some(room));
    assert_eq!(taken.iter().map(|t| (t.0, t.2[0])).collect::<Vec<_>>(), vec![(0, 1.0)]);
    assert!(link.keep_room(room) && link.kept.len() == 2);
    assert_eq!(link.take_kept_room(1, Some(room)).len(), 0);
    assert_eq!(link.take_kept(1).iter().map(|t| t.0).collect::<Vec<_>>(), vec![64]);
    assert!(link.keep_room(link.keep_rows() - 128) && !link.keep_room(link.keep_rows() - 127));
}

#[test]
fn pp_kept_row_limit_by_link_and_the_chain_reply_length_boundary() {
    // Direct front: KEEP_ROWS; chain front or middle: CHAIN_KEEP_ROWS + widest tile fit a u32 reply.
    let mut link = front(pair().0, Duration::from_secs(2));
    let limits = [(false, None), (true, None), (true, Some(Vec::new()))].map(|(i, r)| { (link.inline_kept, link.relay_kept) = (i, r); link.keep_rows() });
    assert!(limits == [CHAIN_KEEP_ROWS, KEEP_ROWS, CHAIN_KEEP_ROWS] && reply_len((CHAIN_KEEP_ROWS + 8192) * W).is_some() && reply_len(104_857 * W).is_some() && reply_len(104_858 * W).is_none() && reply_len(usize::MAX).is_none());
}
#[test]
fn pp_peer_error_text_is_sanitized() {
    assert_eq!(peer_text(b"bad\x1b[31m\nrow"), "bad?[31m?row");
    assert_eq!(peer_text(&[b'x'; 2000]).len(), 512);
}

#[test]
fn pp_long_wire_kept_reply_exact_boundary_and_tcp_fallback_all_chain_sizes() {
    let reply = REPLY_HEAD + 6 * W * 4;
    for ranks in [2, 3, 4] {
        chain_capacity(ranks, Some(reply)); // three synchronous exchanges really use wire
        chain_capacity(ranks, Some(reply - 1)); // first request fits, but kept reply uses TCP
    }
}
#[test]
fn pp_wire_capacity_geometry_and_checked_message_bounds() {
    assert_eq!(wire_capacity(1, 1).unwrap(), MIN_WIRE_BYTES);
    assert_eq!(wire_capacity(8192, 2048).unwrap(), 8192 * W * 4 + REQUEST_HEAD + POOL_SIZE * 16);
    assert_eq!(wire_capacity(32768, 2048).unwrap(), MAX_WIRE_BYTES);
    for (context, tile) in [(0, 2048), (8192, 0), (1, usize::MAX)] { assert!(wire_capacity(context, tile).is_err()); }
    for slot in [0, MIN_WIRE_BYTES - 1, MAX_WIRE_BYTES + 1, usize::MAX] { assert!(validate_wire_capacity(slot).is_err()); }
    assert!(!wire_fits(usize::MAX, usize::MAX, 1, 0));
    assert!(!wire_fits(usize::MAX, 0, 1, usize::MAX));
    assert!(wire_fits(128, 32, 96, 0));
    assert!(!wire_fits(127, 32, 96, 0));
}
#[test]
fn pp_wire_capacity_agreement_precedes_region_setup() {
    for (a, b) in [(MIN_WIRE_BYTES, MIN_WIRE_BYTES), (MIN_WIRE_BYTES, MAX_WIRE_BYTES)] {
        let (mut near, mut far) = pair();
        let peer = std::thread::spawn(move || negotiate_capacity(&mut far, b));
        assert_eq!(negotiate_capacity(&mut near, a).is_ok(), a == b);
        assert_eq!(peer.join().unwrap().is_ok(), a == b);
    }
}
#[test]
fn pp_wire_kept_invalid_length_and_accounting_fail_before_data_changes() {
    for accounting in [false, true] {
        let (near, _far) = pair();
        let mut link = front(near, Duration::from_millis(100));
        link.poisoned = true; // do not wait for test peer shutdown on drop
        link.sent = 1;
        link.owed = 2;
        link.kept.push(KeptTile { parts: vec![KeepPart { slot: 1, pos: 0, tokens: vec![1], start: 0 }],
            rows: if accounting { 1 } else { 2 }, data: None });
        let mut message = Vec::new();
        let bytes = (3 * W * 4) as u32;
        for word in [MAGIC_RESPONSE, 0, bytes + u32::from(!accounting), 1] { put(&mut message, word); }
        message.resize(REPLY_HEAD + bytes as usize, 0);
        let mut residual = vec![f32::from_bits(0x80000000); W];
        let error = link.wire_reply(&message, &mut residual).unwrap_err();
        assert!(error.contains(if accounting { "accounting" } else { "size mismatch" }), "{error}");
        assert_eq!(link.owed, 2);
        assert!(link.kept[0].data.is_none());
        assert!(residual.iter().all(|x| x.to_bits() == 0x80000000));
    }
}

#[test]
fn pp_kept_rows_arrive_on_the_ack_and_skip_the_later_wire_reply() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let (near, far) = pair();
    let (a, b) = pipes();
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
    let worker = std::thread::spawn(move || serve_stage(far, Some(Box::new(b)), None,
        Stage::new(1, 2, None).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| Ok(())));
    let mut link = Link::connect(near, Some(Box::new(a)), Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.post(&[[1, 64, 0, 2]], vec![0.0; 2 * W], Some(vec![KeepPart {
        slot: 1, pos: 0, tokens: vec![1, 2], start: 0,
    }])).unwrap();
    link.drain().unwrap();
    // The kept rows came back on the ack, so a later one-row wire reply does not carry them.
    assert_eq!(link.owed, 0);
    assert!(link.kept.iter().all(|t| t.data.is_some()));
    link.sent += 1;
    let head = request_head(KIND_ROWS, 1, &[[1, 64, 2, 1]], link.sent);
    let mut fast = link.fast.take().unwrap();
    let mut rows = vec![1.0; W];
    link.wire_exchange(&mut *fast, &head, &mut rows).unwrap();
    link.poisoned = true;
    assert!(rows.iter().all(|&x| x == 1.0));
    drop(link);
    worker.join().unwrap().unwrap();
}

#[test]
fn pp_wire_demux_consumes_dropped_chat_rows_before_live_rows() {
    let (near, _far) = pair();
    let mut link = front(near, Duration::from_millis(100));
    link.poisoned = true;
    link.sent = 1;
    link.owed = 2;
    link.kept.push(KeptTile { parts: Vec::new(), rows: 1, data: None });
    link.kept.push(KeptTile { parts: vec![KeepPart { slot: 2, pos: 0, tokens: vec![7], start: 0 }], rows: 1, data: None });
    let data: Vec<f32> = (0..3 * W).map(|i| f32::from_bits([0x80000000, 0x7fc01234, 0x7f800000][i / W])).collect();
    let mut message = Vec::new();
    for word in [MAGIC_RESPONSE, 0, (data.len() * 4) as u32, 1] { put(&mut message, word); }
    message.extend_from_slice(f32_bytes(&data));
    let mut current = vec![0.0; W];
    link.wire_reply(&message, &mut current).unwrap();
    assert_eq!(link.owed, 0);
    assert_eq!(link.kept.len(), 1);
    let live = link.take_kept(2);
    assert_eq!(f32_bytes(&live[0].2), f32_bytes(&data[W..2 * W]));
    assert_eq!(f32_bytes(&current), f32_bytes(&data[2 * W..]));
    assert!(link.kept.is_empty());
}

#[test]
fn every_rank_derives_the_same_stage_intervals() {
    // World 2 at the default and an explicit uneven split, world 3 and 4 even: every rank's
    // list is identical and contains its own interval (each funds the costliest stage).
    for (ranks, split) in [(2, None), (2, Some(20)), (2, Some(30)), (3, None), (4, None)] {
        let lists: Vec<_> = (0..ranks).map(|r| Stage::new(r, ranks, split).unwrap()).collect();
        for stage in &lists {
            assert_eq!(stage.intervals(), lists[0].intervals(), "{ranks} {split:?}");
            assert_eq!(stage.intervals()[stage.rank], stage.first..stage.end);
        }
        assert_eq!(lists[0].intervals().len(), ranks);
    }
    assert_eq!(Stage::new(1, 2, Some(20)).unwrap().intervals(), vec![0..20, 20..48]);
}

/// A KEEP ack carries its tile's rows; the front reads it while it waits for its own GPU
/// (reading_acks), not at the next drain, so the back's replier is never left blocked.
#[test]
fn pp_kept_ack_is_read_while_the_front_waits_for_its_gpu() {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
    let (near, mut far) = pair();
    let peer = near.try_clone().unwrap();
    let (written, ack) = mpsc::channel();
    let back = std::thread::spawn(move || {
        far.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut hello = [0u8; 40];
        far.read_exact(&mut hello).unwrap();
        let mut reply = Vec::new();
        put(&mut reply, MAGIC_HELLO);
        put(&mut reply, 0);
        far.write_all(&reply).unwrap();
        let (kind, id, rows) = read_pipe_request(&mut far, Duration::from_secs(2)).unwrap();
        assert_eq!(kind, KIND_ROWS_KEEP);
        let answered: Vec<f32> = rows.iter().map(|v| v + 1.0).collect();
        write_status(&mut far, id, Some(&answered));
        written.send(()).unwrap();
        // The front reads the ack without drain, then closes the link.
        assert!(read_pipe_request(&mut far, Duration::from_secs(2)).is_err());
    });
    let mut link = Link::connect(near, None, Stage::new(0, 2, None).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.post(&[[1, 64, 0, 2]], vec![2.0; 2 * W], Some(vec![KeepPart { slot: 1, pos: 0, tokens: vec![1, 2], start: 0 }])).unwrap();
    assert_eq!(link.pending, 1);
    // The GPU wait ends once the ack is on the socket and taken: nothing is left to peek.
    link.reading_acks(|| {
        ack.recv_timeout(Duration::from_secs(2)).map_err(err)?;
        let begun = Instant::now();
        peer.set_nonblocking(true).map_err(err)?;
        while peer.peek(&mut [0u8; 1]).is_ok_and(|n| n > 0) && begun.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(1));
        }
        peer.set_nonblocking(false).map_err(err)
    }).unwrap();
    assert_eq!((link.pending, link.owed), (0, 0));
    assert_eq!(link.kept.len(), 1);
    assert!(link.kept[0].data.as_ref().unwrap().iter().all(|&v| v == 3.0));
    link.poisoned = true;
    drop((link, peer));
    back.join().unwrap();
}
