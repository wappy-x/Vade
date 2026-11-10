//! Tiles on the wire: the first and the final tile of a prompt over a chain of three and four
//! ranks, and a back that takes requests in id order from either carrier.
use super::*;
use super::super::prefetch::Request;
use super::super::tests::{layers, pair, pipes_with_capacity};

#[test]
fn pp_chain_first_and_final_tiles_cross_on_the_wire_in_order() {
    for ranks in [3, 4] {
        static STOP: AtomicBool = AtomicBool::new(false);
        let recipe = Recipe { limit: Duration::from_secs(5), asynchronous: true, mtp: true, tp: false };
        // Room for any tile below and for a final reply with every kept row.
        let capacity = REQUEST_HEAD + 16 + 64 * W * 4;
        let mut edges = Vec::new();
        for _ in 1..ranks {
            let (near, far) = pair();
            let (a, b) = pipes_with_capacity(capacity);
            edges.push((Some((near, Some(Box::new(a) as Box<dyn PpWire>))), Some((far, Some(Box::new(b) as Box<dyn PpWire>)))));
        }
        let (stream, fast) = edges[0].0.take().unwrap();
        let mut workers = Vec::new();
        for rank in 1..ranks {
            let (stream, fast) = edges[rank - 1].1.take().unwrap();
            let next = edges.get_mut(rank).and_then(|edge| edge.0.take());
            workers.push(std::thread::spawn(move || {
                let stage = Stage::new(rank, ranks, None).unwrap();
                serve_stage(stream, fast, next, stage, 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
                    |_, _, _, rows| {
                        std::thread::sleep(Duration::from_millis(20));
                        layers(rows, stage.first, stage.end);
                        Ok(())
                    })
            }));
        }
        let stage = Stage::new(0, ranks, None).unwrap();
        let mut link = Link::connect(stream, fast, stage, 64, 8, 8, &STOP, recipe).unwrap();
        let (mut expected, mut pos) = (Vec::new(), 0);
        for t in 0..4 {
            let n = 2 + t % 2;
            let mut rows: Vec<f32> = (0..n * W).map(|i| (t * 7919 + i) as f32 * 0.25).collect();
            let mut all = rows.clone();
            layers(&mut all, 0, LAYERS);
            expected.push(all);
            layers(&mut rows, 0, stage.end);
            let keep = vec![KeepPart { slot: 1, pos, tokens: vec![7; n], start: 0 }];
            link.post(&[[1, 64, pos as u32, n as u32]], rows, Some(keep)).unwrap();
            pos += n;
        }
        // The first tile went on the wire (nothing was queued); the next ones queued behind it on TCP.
        assert_eq!(link.carried, 1, "ranks={ranks}");
        let mut last: Vec<f32> = (0..3 * W).map(|i| i as f32 - 0.5).collect();
        let mut want = last.clone();
        layers(&mut last, 0, stage.end);
        layers(&mut want, 0, LAYERS);
        link.post_final(&[[1, 64, pos as u32, 3]], last.clone()).unwrap();
        link.finish_final(&mut last).unwrap();
        assert_eq!(f32_bytes(&last), f32_bytes(&want));
        // The final tile and its reply crossed on the wire too.
        assert_eq!(link.carried, 3, "ranks={ranks}");
        let kept = link.take_kept(1);
        assert_eq!(kept.len(), expected.len());
        for (actual, expected) in kept.iter().zip(&expected) { assert_eq!(f32_bytes(&actual.2), f32_bytes(expected)); }
        drop(link);
        for worker in workers { worker.join().unwrap().unwrap(); }
    }
}

#[test]
fn early_requests_wait_for_their_id_on_either_carrier() {
    let request = |id: u32| Request { id, kind: KIND_ROWS_ASYNC, count: 1, meta: vec![[1, 64, 0, 1]], rows: vec![0.0; W] };
    let wire = |id: u32| request_head(KIND_ROWS_FINAL, 1, &[[1, 64, 0, 1]], id);
    let (mut early, mut message) = (Early::new(Duration::from_secs(60)), Vec::new());
    // The final tile (id 3) came on the wire while tile 2 is still on TCP.
    early.wire = Some(wire(3));
    assert_eq!(early.next(2, &mut message).unwrap(), None);
    early.tcp = Some(request(2));
    assert_eq!(early.next(2, &mut message).unwrap(), Some(false));
    assert_eq!(early.tcp.take().unwrap().id, 2);
    assert_eq!(early.next(3, &mut message).unwrap(), Some(true));
    assert_eq!(message, wire(3));
    assert!(early.wire.is_none() && early.tcp.is_none());
    // Neither carrier can hold the expected id: a broken link.
    (early.wire, early.tcp) = (Some(wire(5)), Some(request(6)));
    assert_eq!(early.next(4, &mut message).unwrap_err(), "pipeline request out of order: neither carrier holds id 4");
    // A TCP request behind a parked one comes next, in order.
    (early.wire, early.tcp) = (None, None);
    early.later.extend([request(7), request(8)]);
    assert_eq!(early.next(7, &mut message).unwrap(), Some(false));
    assert_eq!(early.tcp.take().unwrap().id, 7);
    assert_eq!(early.next(8, &mut message).unwrap(), Some(false));
    assert!(early.later.is_empty());
    assert_eq!(wire_id(&[0u8; 4], 9), 9);
}

/// However many requests wait behind a missing id, the back stops at the link limit: the prefetch
/// reader may not see a closed link then. An id that comes clears the count.
#[test]
fn parked_requests_stop_at_the_link_limit() {
    let request = |id: u32| Request { id, kind: KIND_ROWS_ASYNC, count: 1, meta: vec![[1, 64, 0, 1]], rows: vec![0.0; W] };
    let until_refused = |early: &mut Early, expect: u32, message: &mut Vec<u8>| {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match early.next(expect, message) {
                Ok(None) if Instant::now() < deadline => std::thread::yield_now(),
                Ok(other) => panic!("no refusal: {other:?}"),
                Err(why) => return why,
            }
        }
    };
    let mut message = Vec::new();
    // Two parked on TCP; id 4 never comes.
    let mut early = Early::new(Duration::from_millis(5));
    early.tcp = Some(request(6));
    early.later.push_back(request(7));
    assert_eq!(until_refused(&mut early, 4, &mut message), "pipeline request out of order: id 4 did not come within 5 ms");
    assert_eq!((early.tcp.as_ref().map(|r| r.id), early.later.len()), (Some(6), 1));
    // One parked on the wire.
    let mut early = Early::new(Duration::from_millis(5));
    early.wire = Some(request_head(KIND_ROWS_FINAL, 1, &[[1, 64, 0, 1]], 9));
    assert_eq!(until_refused(&mut early, 8, &mut message), "pipeline request out of order: id 8 did not come within 5 ms");
    // No request waits: an idle back counts nothing, also long after the limit of an earlier wait.
    let mut early = Early::new(Duration::ZERO);
    assert_eq!(early.next(1, &mut message).unwrap(), None);
    // The expected id clears the count, so the next wait starts again.
    let mut early = Early::new(Duration::from_secs(60));
    early.tcp = Some(request(3));
    assert_eq!(early.next(2, &mut message).unwrap(), None);
    assert!(early.parked.is_some());
    assert_eq!(early.next(3, &mut message).unwrap(), Some(false));
    assert!(early.parked.is_none());
}
