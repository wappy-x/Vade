//! Kept rows reach the front through the middles of a chain before the final reply.
use super::*;
use super::super::tests::{front, layers, pair};

#[test]
fn pp_chain_kept_rows_reach_the_front_before_the_final_reply() {
    // Three and four ranks: a middle passes up the kept tiles that came back to it in its next
    // KEEP ack, so the front holds early tiles' rows while later tiles still run; every row
    // comes back exact and in post order.
    for ranks in [3, 4] {
        static STOP: AtomicBool = AtomicBool::new(false);
        // Tiles the last rank finished: a middle runs a tile only after the tiles before it ran
        // there, so a stall of that rank cannot hide the early acks this test reads.
        static DONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        DONE.store(0, Ordering::SeqCst);
        let recipe = Recipe { limit: Duration::from_secs(5), asynchronous: true, mtp: true, tp: false };
        let mut edges = Vec::new();
        for _ in 1..ranks { let (near, far) = pair(); edges.push((Some((near, None)), Some((far, None)))); }
        let (stream, fast) = edges[0].0.take().unwrap();
        let mut workers = Vec::new();
        for rank in 1..ranks {
            let (stream, fast) = edges[rank - 1].1.take().unwrap();
            let next = edges.get_mut(rank).and_then(|edge| edge.0.take());
            workers.push(std::thread::spawn(move || {
                let stage = Stage::new(rank, ranks, None).unwrap();
                let mut seen = 0;
                serve_stage(stream, fast, next, stage, 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
                    move |_, _, _, rows| {
                        layers(rows, stage.first, stage.end);
                        if stage.end == LAYERS {
                            DONE.fetch_add(1, Ordering::SeqCst);
                        } else {
                            let deadline = Instant::now() + Duration::from_secs(120);
                            while DONE.load(Ordering::SeqCst) < seen {
                                if Instant::now() > deadline { return Err("the last rank stopped".into()); }
                                std::thread::yield_now();
                            }
                            // The last rank's ack travels back while this tile still runs.
                            std::thread::sleep(Duration::from_millis(40));
                        }
                        seen += 1;
                        Ok(())
                    })
            }));
        }
        let stage = Stage::new(0, ranks, None).unwrap();
        let mut link = Link::connect(stream, fast, stage, 64, 8, 8, &STOP, recipe).unwrap();
        assert!(link.chain_kept && !link.inline_kept);
        let tiles = 8;
        let mut expected = Vec::new();
        for t in 0..tiles {
            let n = 1 + t % 3;
            let pos = expected.iter().map(|e: &Vec<f32>| e.len() / W).sum::<usize>();
            let meta = [[1, 64, pos as u32, n as u32]];
            let mut rows: Vec<f32> = (0..n * W).map(|i| (t * 1000 + i) as f32 * 0.5).collect();
            let mut all = rows.clone();
            layers(&mut all, 0, LAYERS);
            expected.push(all);
            layers(&mut rows, 0, stage.end);
            link.post(&meta, rows, Some(vec![KeepPart { slot: 1, pos, tokens: vec![7; n], start: 0 }])).unwrap();
        }
        link.drain().unwrap();
        let early = link.kept.iter().filter(|t| t.data.is_some()).count();
        assert!(early >= 1 && early < tiles, "ranks={ranks}: {early} tiles back before the final reply");
        assert_eq!(link.owed, link.kept.iter().filter(|t| t.data.is_none()).map(|t| t.rows).sum::<usize>());
        let pos = expected.iter().map(|e| e.len() / W).sum::<usize>() as u32;
        let mut rows = vec![0.25; W];
        let mut last = rows.clone();
        layers(&mut rows, 0, stage.end);
        layers(&mut last, 0, LAYERS);
        link.exchange(KIND_ROWS, &[[1, 64, pos, 1]], &mut rows).unwrap();
        assert_eq!(f32_bytes(&rows), f32_bytes(&last));
        let kept = link.take_kept(1);
        assert_eq!(kept.len(), tiles);
        for (actual, expected) in kept.iter().zip(&expected) { assert_eq!(f32_bytes(&actual.2), f32_bytes(expected)); }
        assert!(link.kept.is_empty() && link.owed == 0);
        drop(link);
        for worker in workers { worker.join().unwrap().unwrap(); }
    }
}

#[test]
fn a_kept_ack_is_refused_before_its_rows_are_read() {
    let (near, _far) = pair();
    let mut link = front(near, Duration::from_secs(1));
    let tile = |rows| KeptTile { parts: Vec::new(), rows, data: None };
    let error = |link: &Link, rows, one| link.check_kept(rows, one).unwrap_err();
    link.owed = 2;
    assert_eq!(error(&link, 3, false), "pipeline kept ack exceeds owed rows");
    assert_eq!(error(&link, 2, false), "pipeline kept ack with no tile");
    link.kept = vec![tile(2), tile(3)];
    link.owed = 9;
    assert_eq!(error(&link, 1, false), "pipeline kept ack does not match the tile");
    assert_eq!(error(&link, 5, true), "pipeline kept ack does not match the tile");
    assert_eq!(error(&link, 4, false), "pipeline kept ack splits a tile");
    assert_eq!(error(&link, 6, false), "pipeline kept ack exceeds the posted tiles");
    assert!(link.check_kept(2, true).is_ok() && link.check_kept(5, false).is_ok());
}

/// A refused ack leaves its rows on the socket: `queued_frame` reads no byte of them, and the
/// link's counts and tiles stay.
#[test]
fn queued_frame_reads_no_rows_of_a_refused_ack() {
    use std::io::Write;
    for (ack, bytes, owed, why) in [
        (Ack::Rows(1), W * 4, 0, "pipeline kept ack exceeds owed rows"),
        (Ack::Rows(2), W * 4, 2, "pipeline kept ack size mismatch"),
        (Ack::Tiles, W * 4 + 4, 2, "pipeline kept ack size mismatch"),
        (Ack::Tiles, W * 4, 2, "pipeline kept ack does not match the tile"),
    ] {
        let (near, mut far) = pair();
        let mut link = front(near, Duration::from_secs(1));
        link.owed = owed;
        link.pending = 1;
        link.sent = 1;
        link.acks.push_back(ack);
        link.kept = vec![KeptTile { parts: Vec::new(), rows: 3, data: None }];
        let id = link.replied.wrapping_add(1);
        let writer = std::thread::spawn(move || {
            let mut message = Vec::new();
            for word in [MAGIC_RESPONSE, 0, bytes as u32, id] { put(&mut message, word); }
            message.resize(REPLY_HEAD + bytes, 0xA5);
            far.write_all(&message).unwrap();
            far
        });
        let mut heard = Instant::now();
        assert_eq!(link.queued_frame(&mut heard).unwrap_err(), why);
        let _far = writer.join().unwrap();
        let mut rows = vec![0u8; bytes];
        read_rest(&mut link.stream, &mut rows, None, Duration::from_secs(1)).unwrap();
        assert!(rows.iter().all(|&b| b == 0xA5));
        assert_eq!((link.owed, link.kept.len(), link.kept[0].data.is_none()), (owed, 1, true));
    }
}
