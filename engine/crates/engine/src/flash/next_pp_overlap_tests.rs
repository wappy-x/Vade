//! Host protocol tests: two verify groups in flight through the production middle ranks.
use super::*;
use std::net::TcpListener;

struct Pipe {
    capacity: usize,
    to: mpsc::Sender<Vec<u8>>,
    from: mpsc::Receiver<Vec<u8>>,
}
impl PpWire for Pipe {
    fn capacity(&self) -> usize { self.capacity }
    fn send(&mut self, parts: &[&[u8]]) -> Result<()> {
        if parts.iter().map(|p| p.len()).sum::<usize>() > self.capacity { return Err("test wire send exceeds capacity".into()); }
        self.to.send(parts.concat()).map_err(|e| e.to_string())
    }
    fn try_recv(&mut self, out: &mut Vec<u8>) -> Result<bool> {
        match self.from.try_recv() {
            Ok(m) => { *out = m; Ok(true) }
            Err(mpsc::TryRecvError::Empty) => Ok(false),
            Err(e) => Err(e.to_string()),
        }
    }
}
fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let near = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    (near, listener.accept().unwrap().0)
}
fn layers(rows: &mut [f32], first: usize, end: usize) {
    for layer in first..end {
        for value in rows.iter_mut() {
            *value = f32::from_bits(value.to_bits().rotate_left(3).wrapping_add(layer as u32 + 1));
        }
    }
}

/// The front posts group 1 (2 rows) and group 2 (3 rows), then takes both replies in order,
/// then runs one exchange. The last rank holds group 1 until rank 1 runs group 2: a middle
/// that waited for group 1's reply first would never run group 2 (the test fails on time).
/// `release`: group 1 carries a pool release, so every middle runs it one at a time.
fn two_groups(ranks: usize, release: bool) {
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: false };
    let capacity = REQUEST_HEAD + 2 * 16 + 4 * W * 4;
    let mut edges = Vec::new();
    for _ in 1..ranks {
        let (near, far) = pair();
        let ((a0, a1), (b0, b1)) = (mpsc::channel(), mpsc::channel());
        let (a, b) = (Pipe { capacity, to: a0, from: b1 }, Pipe { capacity, to: b0, from: a1 });
        edges.push((Some((near, Some(Box::new(a) as Box<dyn PpWire>))), Some((far, Some(Box::new(b) as Box<dyn PpWire>)))));
    }
    let second_ran = Arc::new(AtomicBool::new(false));
    let mut workers = Vec::new();
    for rank in 1..ranks {
        let (stream, fast) = edges[rank - 1].1.take().unwrap();
        let next = edges.get_mut(rank).and_then(|edge| edge.0.take());
        let second_ran = second_ran.clone();
        workers.push(std::thread::spawn(move || {
            let stage = Stage::new(rank, ranks, None).unwrap();
            let mut seen = Vec::new();
            serve_stage(stream, fast, next, stage, 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
                |kind, count, meta, rows| {
                    let pos = meta[0][2];
                    if rank == 1 && pos == 20 { second_ran.store(true, Ordering::SeqCst); }
                    if stage.successor().is_none() && pos == 10 && !release {
                        let started = Instant::now();
                        while !second_ran.load(Ordering::SeqCst) {
                            assert!(started.elapsed() < Duration::from_secs(1), "rank 1 did not run group 2 while group 1 was downstream");
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    seen.push((kind & 0xff, count, meta.to_vec()));
                    layers(rows, stage.first, stage.end);
                    Ok(())
                }).unwrap();
            seen
        }));
    }
    let stage = Stage::new(0, ranks, None).unwrap();
    let (stream, fast) = edges[0].0.take().unwrap();
    let mut link = Link::connect(stream, fast, stage, 64, 8, 8, &STOP, recipe).unwrap();
    // The tensor split never posts.
    link.tp = true;
    assert!(!link.can_post(1, W * 4));
    link.tp = false;
    let input = |n: usize, seed: u32| (0..n * W).map(|i| f32::from_bits(seed.wrapping_mul(2654435761) ^ i as u32)).collect::<Vec<_>>();
    let groups = [(vec![[1, 64, 10, 1], [2, 64, 11, 1]], 2, 1), (vec![[3, 64, 20, 3]], 3, 2)];
    let mut expected = Vec::new();
    link.release_pool = release;
    for (meta, count, seed) in &groups {
        let mut rows = input(*count, *seed);
        let mut all = rows.clone();
        layers(&mut all, 0, LAYERS);
        expected.push(all);
        layers(&mut rows, 0, stage.end);
        assert!(link.can_post(meta.len(), rows.len() * 4));
        link.post_wire(KIND_VERIFY, meta, &rows).unwrap();
    }
    // Two are on the wire: a third does not post.
    assert!(!link.can_post(1, W * 4));
    assert!(link.post_wire(KIND_VERIFY, &[[4, 64, 30, 1]], &input(1, 3)).is_err());
    for ((_, count, _), want) in groups.iter().zip(&expected) {
        let mut rows = vec![0.0; count * W];
        link.collect(&mut rows).unwrap();
        assert_eq!(f32_bytes(&rows), f32_bytes(want), "ranks={ranks} release={release}");
    }
    assert!(!link.posted());
    // The one-at-a-time exchange follows on the same ids.
    let mut rows = input(1, 4);
    let mut want = rows.clone();
    layers(&mut rows, 0, stage.end);
    layers(&mut want, 0, LAYERS);
    link.exchange(KIND_ROWS, &[[1, 64, 12, 1]], &mut rows).unwrap();
    assert_eq!(f32_bytes(&rows), f32_bytes(&want));
    drop(link);
    let requests: Vec<_> = groups.iter().map(|(meta, count, _)| (KIND_VERIFY, *count, meta.clone()))
        .chain([(KIND_ROWS, 1, vec![[1, 64, 12, 1]])]).collect();
    for worker in workers {
        assert_eq!(worker.join().unwrap(), requests);
    }
}

#[test]
fn pp_two_verify_groups_cross_three_and_four_ranks_in_flight_and_in_order() {
    for ranks in [3, 4] {
        two_groups(ranks, false);
        two_groups(ranks, true);
    }
}

#[test]
fn pp_wide_tick_splits_into_verify_tile_groups_in_chat_order() {
    // Eight chats at depth 3: one 32-row tick -> two 16-row groups.
    assert_eq!(split_groups(&[4; 8], GROUP_ROWS), vec![0..4, 4..8]);
    // Mixed widths keep chat order; a chat wider than the tile stays alone.
    assert_eq!(split_groups(&[4, 4, 4, 3, 2, 4], GROUP_ROWS), vec![0..4, 4..6]);
    assert_eq!(split_groups(&[20, 2], GROUP_ROWS), vec![0..1, 1..2]);
    // Four chats fill one tile: one group (no split, no overlap).
    assert_eq!(split_groups(&[4; 4], GROUP_ROWS), vec![0..4]);
}
