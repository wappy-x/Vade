use super::*;

const PAGE: u64 = 2 * 1024 * 1024;
/// Per-chat fixed bytes fitted to the 10-27 two-Spark session plans (2.28 GB at 32768,
/// 16.02 GB at 262144): recurrent state, draft state and small arenas.
const FIXED: u64 = 317_000_000;

/// Test data in the shape of one tensor-split rank: twelve indexed attention layers and
/// one draft layer, every layer with two KV heads of 256 values and a 128-value index row.
fn geometry(format: Format, window: u64) -> Geometry {
    Geometry::new(PAGE, (0..13).flat_map(|_| Geometry::attention(2, 256, 128, format)).collect(), FIXED, window).unwrap()
}
fn pool(window: u64, bytes: u64, slots: usize) -> Pool {
    let g = geometry(Format::Fp32, window);
    let step = g.step();
    Pool::new(g.clone(), Pool::pages_for_bytes(&g, bytes), slots, step).unwrap()
}

#[test]
fn switch_is_off_unless_one() {
    assert_eq!(parse_switch(None), Ok(false));
    assert_eq!(parse_switch(Some("0")), Ok(false));
    assert_eq!(parse_switch(Some("1")), Ok(true));
    assert!(parse_switch(Some("on")).is_err() && parse_switch(Some("")).is_err());
}

#[test]
fn geometry_counts_the_bytes_the_engine_reserves_per_token() {
    // FP32 K, V and index rows of 13 layers: the measured 59.9 KB per token per rank.
    let fp32 = geometry(Format::Fp32, 32768);
    assert_eq!(fp32.token_bytes(), 59_904);
    assert_eq!(geometry(Format::Bf16, 32768).token_bytes(), 33_280);
    assert_eq!(geometry(Format::Fp8, 32768).token_bytes(), 20_176);
    // One page holds 1024 FP32 K rows; index rows fill a page every fourth step.
    assert_eq!(fp32.step(), 1024);
    assert_eq!(fp32.pages(1024), 152 + 13 * 3);
    assert_eq!(fp32.pages(4096), 152 + 13 * 9);
    // A full 32768 window is 1088 pages: 2.28 GB, the measured per-chat reservation.
    assert_eq!(fp32.pages(32768), 1088);
    assert_eq!(fp32.pages(0), 0);
    assert_eq!((fp32.extent(1), fp32.extent(1024), fp32.extent(1025), fp32.extent(40000)), (1024, 1024, 2048, 32768));
    assert!(Geometry::new(PAGE, vec![], FIXED, 32768).is_err() && Geometry::new(PAGE, vec![4, 0], FIXED, 32768).is_err());
    assert!(Geometry::with_step(PAGE, vec![4], FIXED, 32768, 0).is_err() && Geometry::with_step(PAGE, vec![4], FIXED, 32768, 32769).is_err());
    assert_ne!(fp32.digest(), geometry(Format::Fp32, 65536).digest());
    assert_ne!(fp32.digest(), geometry(Format::Bf16, 32768).digest());
}

#[test]
fn admission_by_use_keeps_the_window_count_for_full_chats_and_more_for_short_ones() {
    // The 10-27 pool at 32768: 21.48 GB per rank funds 9 full-window chats.
    let bytes = 21_480_000_000;
    let mut full = pool(32768, bytes, 16);
    let mut admitted = 0;
    while full.admit(32768, 32768).is_ok() { admitted += 1; full.check(); }
    assert_eq!(admitted, 9);
    assert!(matches!(full.admit(32768, 32768), Err(Refusal::Full { need: 1088, .. })));
    // Chats of 2048 prompt tokens and 1024 output tokens claim 3072 tokens each: all 16 slots fill.
    let mut short = pool(32768, bytes, 16);
    for _ in 0..16 { short.admit(2048, 3072).unwrap(); short.check(); }
    assert_eq!(short.admit(2048, 3072), Err(Refusal::NoSlot));
    assert_eq!(short.chat(0), Some((2048, 152 + 13 * 7, 152 + 13 * 5)));
    // The request limit caps the claim, and the window caps the limit.
    let mut capped = pool(32768, bytes, 16);
    capped.admit(100, 100).unwrap();
    assert_eq!(capped.chat(0).unwrap().1, capped.geometry().pages(1024));
    capped.admit(100, 1 << 40).unwrap();
    assert_eq!(capped.chat(1).unwrap().1, capped.geometry().pages(1124_u64.next_multiple_of(1024)));
    assert!(matches!(capped.admit(0, 10), Err(Refusal::Invalid(_))) && matches!(capped.admit(11, 10), Err(Refusal::Invalid(_))));
}

#[test]
fn growth_maps_pages_in_steps_and_extends_the_claim_ahead() {
    let mut p = pool(131072, 8_000_000_000, 4);
    let g = p.geometry().clone();
    let (slot, work) = p.admit(1500, 131072).unwrap();
    assert_eq!(work, vec![Map::Grow { slot, from: 0, to: 2048 }]);
    // Claim: the prompt plus one step ahead.
    assert_eq!(p.chat(slot), Some((1500, g.pages(3072), g.pages(2048))));
    // Inside the mapped extent: no device work.
    assert_eq!(p.grow(slot, 2048), Ok(vec![]));
    // Past the extent: one step more, still inside the claim.
    assert_eq!(p.grow(slot, 2049), Ok(vec![Map::Grow { slot, from: 2048, to: 3072 }]));
    assert_eq!(p.chat(slot).unwrap().1, g.pages(3072));
    // Past the claim: the claim moves one step ahead of the new length.
    assert_eq!(p.grow(slot, 3500), Ok(vec![Map::Grow { slot, from: 3072, to: 4096 }]));
    assert_eq!(p.chat(slot).unwrap().1, g.pages(5120));
    // A rewind keeps the pages; a length past the limit is refused.
    assert_eq!(p.grow(slot, 3000), Ok(vec![]));
    assert_eq!(p.chat(slot).unwrap().2, g.pages(4096));
    assert!(matches!(p.grow(slot, 131073), Err(Refusal::Invalid(_))) && matches!(p.grow(3, 10), Err(Refusal::Invalid(_))));
    p.check();
}

#[test]
fn a_released_slot_keeps_its_pages_for_the_next_chat() {
    let mut p = pool(32768, 21_480_000_000, 16);
    let (a, work) = p.admit(8192, 8200).unwrap();
    assert_eq!(work, vec![Map::Grow { slot: a, from: 0, to: 8192 }]);
    let free = p.free();
    p.release(a).unwrap();
    assert_eq!((p.live(), p.unclaimed(), p.free()), (0, p.total(), free));
    // The next 8K chat takes the same slot and maps nothing: no fresh mapping on the prefill path.
    assert_eq!(p.admit(8192, 8200), Ok((a, vec![])));
    p.release(a).unwrap();
    // A smaller chat prefers a fresh slot to a trim; a bigger one grows the cached slot.
    let (b, work) = p.admit(1000, 1000).unwrap();
    assert_ne!(b, a);
    assert_eq!(work, vec![Map::Grow { slot: b, from: 0, to: 1024 }]);
    assert_eq!(p.admit(9000, 9000), Ok((a, vec![Map::Grow { slot: a, from: 8192, to: 9216 }])));
    assert!(p.release(5).is_err());
    p.check();
}

#[test]
fn a_full_pool_trims_idle_slots_oldest_first_and_trim_returns_pages() {
    let g = geometry(Format::Fp32, 32768);
    // Room for two full chats.
    let mut p = Pool::new(g.clone(), 2 * g.pages(32768), 4, 1024).unwrap();
    let (a, _) = p.admit(32768, 32768).unwrap();
    let (b, _) = p.admit(32768, 32768).unwrap();
    p.release(a).unwrap();
    p.release(b).unwrap();
    assert_eq!(p.free(), 0);
    // A short chat takes a fresh slot (191 pages to map, against 897 to unmap from a
    // cached slot). The pages come from slot a, which ended first: it keeps 25600 tokens.
    let (c, work) = p.admit(1000, 1000).unwrap();
    assert_eq!(g.pages(25600), 893);
    assert_eq!((c, work), (2, vec![Map::Trim { slot: a, from: 32768, to: 25600 }, Map::Grow { slot: 2, from: 0, to: 1024 }]));
    // A full chat takes slot b as it is: no device work.
    assert_eq!(p.admit(32768, 32768), Ok((b, vec![])));
    p.release(c).unwrap();
    p.check();
    // Memory pressure: idle mappings go back, oldest first.
    assert_eq!(p.trim(p.total()), vec![Map::Trim { slot: a, from: 25600, to: 0 }, Map::Trim { slot: c, from: 1024, to: 0 }]);
    assert_eq!(p.free(), p.total() - g.pages(32768));
    p.check();
}

#[test]
fn growth_waits_when_the_pool_is_full_and_the_newest_chat_is_the_victim() {
    let g = geometry(Format::Fp32, 32768);
    let mut p = Pool::new(g.clone(), g.pages(8192) + g.pages(8192), 4, 1024).unwrap();
    let (a, _) = p.admit(4096, 32768).unwrap();
    let (b, _) = p.admit(4096, 32768).unwrap();
    let mut len = [4096u64; 2];
    // Both chats grow in turn until neither can pass its claim.
    loop {
        let grown = [a, b].iter().enumerate().filter(|&(i, &s)| p.grow(s, len[i] + 1).map(|_| len[i] += 1).is_ok()).count();
        p.check();
        if grown == 0 { break; }
    }
    assert_eq!(p.grow(a, len[0] + 1), Err(Refusal::Wait));
    assert_eq!(p.grow(b, len[1] + 1), Err(Refusal::Wait));
    assert_eq!(p.chat(a).unwrap().0, len[0]);
    assert!(p.unclaimed() < p.geometry().pages(1024));
    // The newest admission yields; then the other chat grows again.
    assert_eq!(p.victim(), Some(b));
    let parked = p.park(b).unwrap();
    assert_eq!(parked.len, len[1]);
    // An eager resume in the same round cannot take back the pages: it needs its full
    // look-ahead. So no park and resume loop; the other chat moves.
    let unclaimed = p.unclaimed();
    assert!(matches!(p.resume(parked), Err(Refusal::Full { left, .. }) if left == unclaimed));
    assert!(p.grow(a, len[0] + 1).is_ok());
    p.check();
    // The parked chat comes back when the other chat ends.
    while p.grow(a, len[0] + 1).is_ok() { len[0] += 1; }
    assert!(matches!(p.resume(parked), Err(Refusal::Full { .. })));
    p.release(a).unwrap();
    let (slot, _) = p.resume(parked).unwrap();
    assert_eq!(p.chat(slot).unwrap().0, len[1]);
    assert!(p.grow(slot, len[1] + 1).is_ok());
    p.check();
}

#[test]
fn a_scheduler_that_resumes_at_once_still_finishes_every_chat() {
    // The eager order: every round resumes parked chats and admits queued ones first, then
    // grows each live chat by 1..=4 rows (a drafted step). When no chat moves and some
    // wait, the victim parks. The pool holds three 8K chats; 24 chats of up to 18K tokens.
    let g = geometry(Format::Fp32, 32768);
    for seed in [1, 7, 20261028] {
        let (mut r, mut p) = (Lcg(seed), Pool::new(g.clone(), 3 * g.pages(8192), 8, 1024).unwrap());
        let mut queued: Vec<Parked> = (0..24).map(|_| { let len = 1 + r.next(6000); Parked { len, limit: len + 1 + r.next(12000) } }).collect();
        let (mut parked, mut live, mut done, mut rounds, mut parks) = (Vec::<Parked>::new(), Vec::<(usize, Parked)>::new(), 0, 0, 0);
        while done < 24 {
            rounds += 1;
            assert!(rounds < 100_000, "seed {seed}: no progress");
            parked.retain(|&c| p.resume(c).map(|(slot, _)| live.push((slot, c))).is_err());
            queued.retain(|&c| p.admit(c.len, c.limit).map(|(slot, _)| live.push((slot, c))).is_err());
            let (mut moved, mut waited) = (false, false);
            live.retain_mut(|(slot, c)| {
                let len = (c.len + 1 + r.next(4)).min(c.limit);
                match p.grow(*slot, len) {
                    Ok(_) if len == c.limit => { p.release(*slot).unwrap(); done += 1; false }
                    Ok(_) => { (c.len, moved) = (len, true); true }
                    Err(Refusal::Wait) => { waited = true; true }
                    Err(e) => panic!("seed {seed}: {e:?}"),
                }
            });
            if !moved && waited {
                let slot = p.victim().unwrap();
                let i = live.iter().position(|&(s, _)| s == slot).unwrap();
                parked.push(p.park(live.remove(i).0).unwrap());
                parks += 1;
            }
            p.check();
        }
        assert_eq!((p.live(), p.unclaimed()), (0, p.total()));
        assert!(parks > 0, "seed {seed}: the stream never parked");
    }
}

#[test]
fn a_lone_chat_at_the_pool_edge_grows_to_the_end_and_then_stops_for_good() {
    // The pool holds one 8192-token chat (386 pages); a 9216 extent needs 425.
    let g = geometry(Format::Fp32, 32768);
    let total = g.pages(8192);
    assert_eq!((total, g.pages(9216)), (386, 425));
    let mut p = Pool::new(g.clone(), total, 4, 1024).unwrap();
    let (a, _) = p.admit(4096, 32768).unwrap();
    // The look-ahead shrinks at the edge: the chat never waits, it grows to 8192.
    for len in 4097..=8192 { assert!(p.grow(a, len).is_ok(), "len {len}"); p.check(); }
    assert_eq!(p.chat(a), Some((8192, total, total)));
    // Past the whole pool: a terminal refusal, not a wait; nothing changes.
    assert_eq!(p.grow(a, 8193), Err(Refusal::TooLong { need: 425, total }));
    assert_eq!(p.chat(a).unwrap().0, 8192);
    // Park at the edge and resume into the empty pool: the chat takes back its length.
    let parked = p.park(a).unwrap();
    assert_eq!(p.resume(parked), Ok((a, vec![])));
    assert_eq!(p.chat(a), Some((8192, total, total)));
    p.release(a).unwrap();
    // Admission: a prompt that can never fit is TooLong; one that fits only without the
    // full look-ahead gets the pool; a busy pool is Full, which passes when chats end.
    assert_eq!(p.admit(8193, 32768), Err(Refusal::TooLong { need: 425, total }));
    let (b, _) = p.admit(7171, 32768).unwrap();
    assert_eq!(p.chat(b).unwrap().1, total);
    assert!(matches!(p.admit(1, 10), Err(Refusal::Full { need, .. }) if need <= total));
    p.check();
}

#[test]
fn extents_stop_at_a_window_that_is_not_a_step_multiple_and_large_values_do_not_overflow() {
    let g = geometry(Format::Fp32, 40000);
    assert_eq!((g.extent(39936), g.extent(39937), g.extent(40000), g.extent(u64::MAX)), (39936, 40000, 40000, 40000));
    assert_eq!((g.pages(u64::MAX), g.pages(50000)), (g.pages(40000), g.pages(40000)));
    let mut p = Pool::new(g.clone(), 2 * g.pages(40000), 2, 1024).unwrap();
    let (a, work) = p.admit(39000, 40000).unwrap();
    assert_eq!(work, vec![Map::Grow { slot: a, from: 0, to: 39936 }]);
    assert_eq!(p.grow(a, 40000), Ok(vec![Map::Grow { slot: a, from: 39936, to: 40000 }]));
    p.release(a).unwrap();
    // A trim keeps the longest step multiple below a short last step.
    assert_eq!(p.trim(p.free() + 1), vec![Map::Trim { slot: a, from: 40000, to: 39936 }]);
    p.check();
    // The look-ahead must be 1..=window; the largest one admits without overflow.
    assert!(Pool::new(g.clone(), 100, 1, 0).is_err() && Pool::new(g.clone(), 100, 1, 40001).is_err());
    // Absurd fixed bytes or widths are refused at the geometry, not wrapped later.
    assert!(Geometry::new(1, vec![1], u64::MAX, 10).is_err());
    assert!(Geometry::new(PAGE, Geometry::attention(u64::MAX, 256, 128, Format::Fp32), FIXED, 1).is_err());
    let mut wide = Pool::new(g.clone(), u64::MAX, 1, 40000).unwrap();
    assert!(wide.admit(1, 10).is_ok() && wide.grow(0, 10).is_ok());
    assert_eq!(plan_slots(&g, 0, u64::MAX, 4, |_| 0), (1, 0));
}

#[test]
fn a_parked_chat_frees_its_claim_and_resumes_at_its_length() {
    let mut p = pool(131072, 8_000_000_000, 2);
    let (a, _) = p.admit(60000, 131072).unwrap();
    p.grow(a, 61000).unwrap();
    let unclaimed = p.unclaimed();
    let parked = p.park(a).unwrap();
    assert_eq!(parked, Parked { len: 61000, limit: 131072 });
    assert!(p.unclaimed() > unclaimed && p.live() == 0);
    // The parked chat's slot serves another chat meanwhile.
    let (b, _) = p.admit(2000, 4000).unwrap();
    assert_eq!(b, 1);
    let (slot, work) = p.resume(parked).unwrap();
    assert_eq!((slot, work), (a, vec![]));
    assert_eq!(p.chat(a).unwrap().0, 61000);
    p.check();
}

/// Deterministic operation stream (no outside crate): admit, grow, release, park, trim.
struct Lcg(u64);
impl Lcg { fn next(&mut self, n: u64) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 33) % n } }
fn drive(p: &mut Pool, seed: u64, steps: usize) -> Vec<String> {
    let (mut r, mut out, mut parked) = (Lcg(seed), Vec::new(), Vec::new());
    for _ in 0..steps {
        let slot = r.next(16) as usize;
        let line = match r.next(10) {
            0..=2 => { let prompt = 1 + r.next(40000); format!("{:?}", p.admit(prompt, prompt + r.next(30000))) }
            3..=6 => { let len = p.chat(slot).map_or(1, |c| c.0 + r.next(3000)); format!("{:?}", p.grow(slot, len)) }
            7 => format!("{:?}", p.release(slot)),
            8 => match p.park(slot) { Ok(c) => { parked.push(c); "parked".into() } Err(e) => format!("{e:?}") },
            _ => match parked.pop() { Some(c) => format!("{:?}", p.resume(c)), None => format!("{:?}", p.trim(r.next(p.total() + 1))) },
        };
        p.check();
        out.push(format!("{line} {:#x}", p.digest()));
    }
    out
}

#[test]
fn two_ranks_with_different_free_memory_agree_and_make_the_same_decisions() {
    let g = geometry(Format::Fp32, 65536);
    // The ranks see different free bytes; they agree on the smaller page count.
    let (rank0, rank1) = (Pool::pages_for_bytes(&g, 22_910_000_000), Pool::pages_for_bytes(&g, 22_270_000_000));
    let total = Pool::agree((rank0, g.digest()), (rank1, g.digest())).unwrap();
    assert_eq!(total, rank1);
    assert_eq!(Pool::agree((rank1, g.digest()), (rank0, g.digest())), Ok(total));
    assert!(Pool::agree((rank0, g.digest()), (rank1, geometry(Format::Bf16, 65536).digest())).is_err());
    for seed in [1, 7, 20261027] {
        let (mut front, mut back) = (Pool::new(g.clone(), total, 16, 1024).unwrap(), Pool::new(g.clone(), total, 16, 1024).unwrap());
        let (a, b) = (drive(&mut front, seed, 4000), drive(&mut back, seed, 4000));
        assert_eq!(a, b, "seed {seed}");
        assert_eq!(front, back);
        // The stream reaches the hard cases: refusals, waits and trims.
        for case in ["Full", "NoSlot", "Wait", "Trim", "parked"] { assert!(a.iter().any(|l| l.contains(case)), "seed {seed}: no {case}"); }
    }
    // Without the agreement the ledgers differ from the first operation on.
    let (mut front, mut back) = (Pool::new(g.clone(), rank0, 16, 1024).unwrap(), Pool::new(g.clone(), rank1, 16, 1024).unwrap());
    assert_ne!(drive(&mut front, 7, 1)[0], drive(&mut back, 7, 1)[0]);
}

#[test]
fn the_claim_fit_is_the_longest_coverage_that_fits_the_room() {
    // A window that is not a step multiple, so the last step is short.
    let p = Pool::new(geometry(Format::Fp32, 40000), 4000, 1, 1024).unwrap();
    let (g, mut r) = (p.geometry().clone(), Lcg(5));
    for _ in 0..3000 {
        let len = 1 + r.next(40000);
        let (want, room) = (len + r.next(40001 - len), r.next(2000));
        // Candidates: the full look-ahead, then each step multiple from len's extent up.
        let brute = std::iter::once(want).chain((0..want.div_ceil(1024)).rev().map(|n| n * 1024).filter(|&e| e >= g.extent(len) && e < want))
            .find(|&c| g.pages(g.extent(c)) <= room);
        assert_eq!(p.fit(len, want, room), brute, "len {len} want {want} room {room}");
    }
}

#[test]
fn startup_funds_the_most_slots_whose_shared_reserve_leaves_one_small_chat_each() {
    let g = geometry(Format::Fp32, 32768);
    // Graph banks 1, 2, 4, 8, 16 up to the slot count, about 63 MB each per row (10-27 fit).
    let shared = |slots: usize| 63_300_000 * (2 * slots.next_power_of_two() as u64 - 1);
    let available = 23_510_000_000 - 2048 * 32768;
    let (slots, pages) = plan_slots(&g, available, 1024, 16, shared);
    assert_eq!(slots, 16);
    assert_eq!(pages, (available - shared(16)) / PAGE);
    // A pool too small for sixteen small chats funds fewer slots; one slot always plans.
    let small = g.pages(2048) * 3 * PAGE;
    assert_eq!(plan_slots(&g, small + shared(3), 1024, 16, shared).0, 3);
    assert_eq!(plan_slots(&g, 0, 1024, 16, shared), (1, 0));
}
