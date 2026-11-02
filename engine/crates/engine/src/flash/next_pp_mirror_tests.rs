//! The tensor split over more than two ranks on the pipeline links: a middle sends each pass
//! down before it runs it (its takes wait for the ranks after it), with the same kind, flags
//! and rows, and refuses a pass that waits for rows.
use super::*;
use std::net::TcpListener;

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let near = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (far, _) = listener.accept().unwrap();
    (near, far)
}

/// The tensor split's stages: rank 0 keeps the first layers, and every other rank mirrors the rest.
#[test]
fn split_stages_mirror_every_layer_after_the_cut() {
    assert_eq!(Stage::new(0, 4, Some(2)).unwrap(), Stage { rank: 0, ranks: 4, first: 0, end: 2 });
    for rank in 1..4 { assert_eq!(Stage::new(rank, 4, Some(2)).unwrap(), Stage { rank, ranks: 4, first: 2, end: 48 }); }
}

/// Each pass reaches the last rank before the middle's step runs it (the middle's step waits
/// for the last rank's), with the kind word unchanged; both ranks run every pass in order.
#[test]
fn a_split_middle_sends_each_pass_down_before_it_runs_it() {
    use crate::flash_tp_switch::{KIND_VERIFY as TP_VERIFY, MAIL, SPLIT};
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: true };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let (ran, wait) = mpsc::channel();
    let tail = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve_stage(back_s, None, None, Stage::new(2, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
            |kind, count, meta, rows| {
                seen.push((kind, count, meta[0][2], rows[0]));
                ran.send(meta[0][2]).unwrap();
                Ok(())
            }).unwrap();
        seen
    });
    let middle = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe,
            |_| Ok(()), || Ok(()), |kind, count, meta, rows| {
                // The last rank ran this pass already: the middle sent it down first.
                assert_eq!(wait.recv_timeout(Duration::from_secs(2)), Ok(meta[0][2]), "the pass did not go down before the step");
                seen.push((kind, count, meta[0][2], rows[0]));
                Ok(())
            }).unwrap();
        seen
    });
    let mut link = Link::connect(front_s, None, Stage::new(0, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.tp_mail = true;
    link.post(&[[1, 64, 0, 2]], vec![0.0; 2 * W], None).unwrap();
    link.post(&[[1, 64, 2, 1]], vec![3.0; W], None).unwrap();
    (link.tp_verify, link.tp_mail) = (true, true);
    link.post(&[[1, 64, 3, 1]], vec![0.0; W], None).unwrap();
    link.drain().unwrap();
    drop(link);
    let want = [(KIND_ROWS_ASYNC | MAIL, 2, 0, 0.0), (KIND_ROWS_ASYNC, 1, 2, 3.0), (TP_VERIFY | MAIL, 1, 3, 0.0)];
    assert_eq!(middle.join().unwrap(), want);
    assert_eq!(tail.join().unwrap(), want);
}

/// A middle's step waits for the rank after it (the gather's takes, a mirrored pass's takes), and
/// that rank may still have kept tiles to ack, with their rows. The middle reads those acks during
/// its step, or the rank after it stops on its full reply queue and never reaches the same step.
fn middle_reads_kept_acks_during(pass: u32, post: impl FnOnce(&mut Link)) {
    use crate::flash_tp_switch::SPLIT;
    use std::sync::Arc;
    static STOP: AtomicBool = AtomicBool::new(false);
    const TILES: u32 = 8;
    const ROWS: usize = 128;
    let recipe = Recipe { limit: Duration::from_secs(20), asynchronous: true, mtp: true, tp: true };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let (middle_there, last_there) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (middle_flag, last_flag) = (middle_there.clone(), last_there.clone());
    let tail = std::thread::spawn(move || {
        serve_stage(back_s, None, None, Stage::new(2, 3, Some(SPLIT)).unwrap(), 2048, 8, ROWS, &STOP, recipe, |_| Ok(()), || Ok(()),
            move |kind, _, meta, _| {
                if kind == pass { last_flag.store(true, Ordering::SeqCst); return Ok(()); }
                // The last tiles run only after the middle started its step.
                if meta[0][2] as usize >= 3 * ROWS {
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while !middle_flag.load(Ordering::SeqCst) {
                        if Instant::now() > deadline { return Err("the middle did not reach its step".into()); }
                        std::thread::yield_now();
                    }
                }
                Ok(())
            })
    });
    let (middle_flag, last_flag) = (middle_there.clone(), last_there.clone());
    let middle = std::thread::spawn(move || {
        serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, Some(SPLIT)).unwrap(), 2048, 8, ROWS, &STOP, recipe,
            |_| Ok(()), || Ok(()), move |kind, _, _, _| {
                if kind != pass { return Ok(()); }
                middle_flag.store(true, Ordering::SeqCst);
                let deadline = Instant::now() + Duration::from_secs(3);
                while !last_flag.load(Ordering::SeqCst) {
                    if Instant::now() > deadline { return Err("the last rank did not reach the step".into()); }
                    std::thread::yield_now();
                }
                Ok(())
            })
    });
    let mut link = Link::connect(front_s, None, Stage::new(0, 3, Some(SPLIT)).unwrap(), 2048, 8, ROWS, &STOP, recipe).unwrap();
    assert!(link.chain_kept);
    for tile in 0..TILES {
        let pos = tile as usize * ROWS;
        link.tp_tile = true;
        link.post(&[[1, 2048, pos as u32, ROWS as u32]], vec![0.0; ROWS * W], Some(vec![KeepPart { slot: 1, pos, tokens: vec![7; ROWS], start: 0 }])).unwrap();
    }
    link.drain().unwrap();
    link.tp_tile = false;
    post(&mut link);
    link.drain().unwrap();
    assert!(last_there.load(Ordering::SeqCst));
    drop(link);
    middle.join().unwrap().unwrap();
    tail.join().unwrap().unwrap();
}

#[test]
fn a_split_middle_reads_kept_acks_during_its_handoff() {
    middle_reads_kept_acks_during(crate::flash_tp_switch::KIND_HANDOFF, |link| {
        link.tp_handoff = true;
        link.post(&[[1, 2048, 1024, 1]], vec![0.0; W], None).unwrap();
    });
}

/// Another chat's mirrored pass runs while the first chat's kept tiles still come back.
#[test]
fn a_split_middle_reads_kept_acks_during_a_mirrored_pass() {
    middle_reads_kept_acks_during(KIND_ROWS_ASYNC | crate::flash_tp_switch::MAIL, |link| {
        link.tp_mail = true;
        link.post(&[[2, 2048, 0, 1]], vec![0.0; W], None).unwrap();
    });
}

/// A pass whose reply carries rows has no place on the split over more ranks: the middle stops.
#[test]
fn a_split_middle_refuses_a_pass_that_waits_for_rows() {
    use crate::flash_tp_switch::SPLIT;
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: true };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let tail = std::thread::spawn(move || serve_stage(back_s, None, None, Stage::new(2, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP,
        recipe, |_| Ok(()), || Ok(()), |_, _, _, _| Ok(())));
    let middle = std::thread::spawn(move || serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, Some(SPLIT)).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| panic!("a row-reply pass ran on the middle")));
    let mut link = Link::connect(front_s, None, Stage::new(0, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    assert!(link.exchange(KIND_ROWS, &[[1, 64, 0, 1]], &mut vec![0.0; W]).is_err());
    assert!(middle.join().unwrap().unwrap_err().contains("only asynchronous passes"));
    drop(link);
    let _ = tail.join().unwrap();
}

/// A chain tile (TILE) is not mirrored: the middle runs its tile layers first, then sends the
/// tile down with its rows and the TILE bit, as a chain does; the last rank sees the middle's rows.
#[test]
fn a_split_middle_sends_a_chain_tile_down_after_its_layers() {
    use crate::flash_tp_switch::{SPLIT, TILE};
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp: true };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let tail = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve_stage(back_s, None, None, Stage::new(2, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
            |kind, count, meta, rows| { seen.push((kind, count, meta[0][2], rows[0])); Ok(()) }).unwrap();
        seen
    });
    let middle = std::thread::spawn(move || serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, Some(SPLIT)).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, rows| { rows.iter_mut().for_each(|r| *r += 1.0); Ok(()) }).unwrap());
    let mut link = Link::connect(front_s, None, Stage::new(0, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.tp_tile = true;
    link.post(&[[1, 64, 0, 2]], vec![5.0; 2 * W], None).unwrap();
    link.tp_tile = false;
    link.tp_mail = true;
    link.post(&[[1, 64, 2, 1]], vec![0.0; W], None).unwrap();
    link.drain().unwrap();
    drop(link);
    middle.join().unwrap();
    assert_eq!(tail.join().unwrap(), [(KIND_ROWS_ASYNC | TILE, 2, 0, 6.0), (KIND_ROWS_ASYNC | crate::flash_tp_switch::MAIL, 1, 2, 0.0)]);
}

/// A middle pass longer than the link limit: the acks the last rank sent meanwhile wait in the
/// socket, and the middle reads them after its step instead of calling the last rank silent.
#[test]
fn a_split_middle_reads_the_acks_after_a_pass_longer_than_the_limit() {
    use crate::flash_tp_switch::SPLIT;
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_millis(300), asynchronous: true, mtp: true, tp: true };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    let tail = std::thread::spawn(move || serve_stage(back_s, None, None, Stage::new(2, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP,
        recipe, |_| Ok(()), || Ok(()), |_, _, _, _| Ok(())));
    let middle = std::thread::spawn(move || serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, Some(SPLIT)).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| { std::thread::sleep(Duration::from_millis(700)); Ok(()) }));
    let mut link = Link::connect(front_s, None, Stage::new(0, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.tp_mail = true;
    for row in 0..2 { link.post(&[[1, 64, row, 1]], vec![0.0; W], None).unwrap(); }
    link.drain().unwrap();
    drop(link);
    middle.join().unwrap().unwrap();
    tail.join().unwrap().unwrap();
}

/// A middle chain tile (TILE) longer than the link limit: the tile goes down after the middle's
/// step, and the limit for its ack counts from then, not from before the step.
#[test]
fn a_split_middle_chain_tile_longer_than_the_limit_keeps_the_link() {
    use crate::flash_tp_switch::{SPLIT, TILE};
    static STOP: AtomicBool = AtomicBool::new(false);
    let recipe = Recipe { limit: Duration::from_millis(300), asynchronous: true, mtp: true, tp: true };
    let (front_s, middle_in) = pair();
    let (middle_out, back_s) = pair();
    // The last rank records each pass it gets: both must arrive, in order, with their flags.
    let tail = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve_stage(back_s, None, None, Stage::new(2, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()),
            |kind, count, meta, _| { seen.push((kind, count, meta[0][2])); Ok(()) }).map(|()| seen)
    });
    let middle = std::thread::spawn(move || serve_stage(middle_in, None, Some((middle_out, None)), Stage::new(1, 3, Some(SPLIT)).unwrap(),
        64, 8, 8, &STOP, recipe, |_| Ok(()), || Ok(()), |_, _, _, _| { std::thread::sleep(Duration::from_millis(700)); Ok(()) }));
    let mut link = Link::connect(front_s, None, Stage::new(0, 3, Some(SPLIT)).unwrap(), 64, 8, 8, &STOP, recipe).unwrap();
    link.tp_tile = true;
    link.post(&[[1, 64, 0, 2]], vec![0.0; 2 * W], None).unwrap();
    link.tp_tile = false;
    link.tp_mail = true;
    link.post(&[[1, 64, 2, 1]], vec![0.0; W], None).unwrap();
    link.drain().unwrap();
    drop(link);
    middle.join().unwrap().unwrap();
    assert_eq!(tail.join().unwrap().unwrap(), [(KIND_ROWS_ASYNC | TILE, 2, 0), (KIND_ROWS_ASYNC | crate::flash_tp_switch::MAIL, 1, 2)]);
}
