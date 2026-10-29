//! Public fabric behavior without a peer: link layout, the profiling wrapper
//! around an Exchange, and the fixed size of a collective profile.
use engine_fabric::{exchange::{Exchange, ExchangeStats, Profiled}, layout::PeerLayout,
    profile::{Bytes, Kind, Parts, Path, Profile}};

#[test]
fn peer_layout_separates_pair_duplex_and_path() {
    let ring_pair = PeerLayout::new(2, false);
    assert!(ring_pair.is_pair() && ring_pair.duplex() && !ring_pair.is_path());
    // A two-rank path is still a pair, but it has no shared duplex socket.
    let path_pair = PeerLayout::new(2, true);
    assert!(path_pair.is_pair() && !path_pair.duplex() && path_pair.is_path());
    for world in [1, 3, 4, 6] {
        for path in [false, true] {
            let layout = PeerLayout::new(world, path);
            assert!(!layout.is_pair() && !layout.duplex(), "world {world} path {path}");
            assert_eq!(layout.is_path(), path);
        }
    }
}

/// A local Exchange that records what the wrapper forwards to it.
#[derive(Default)]
struct Recorder { profiling: bool, fail: bool, flash: Vec<u64>, points: Vec<(usize, u64, bool)>, steps: Vec<String>, mailbox: Vec<u8> }
impl Exchange for Recorder {
    fn gather_bytes(&mut self, values: &mut [u8], offsets: &[usize]) -> Result<(), String> {
        if self.fail { return Err("gather_bytes failed".into()); }
        assert_eq!(offsets.last(), Some(&values.len()));
        values.iter_mut().for_each(|v| *v = v.wrapping_add(1));
        Ok(())
    }
    fn gather_f32(&mut self, values: &mut [f32], _offsets: &[usize]) -> Result<(), String> {
        if self.fail { return Err("gather_f32 failed".into()); }
        values.iter_mut().for_each(|v| *v *= 2.0);
        Ok(())
    }
    fn reduce(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<(), String> {
        if self.fail { return Err("reduce failed".into()); }
        assert_eq!(values.len(), scratch.len());
        values.iter_mut().for_each(|v| *v += 1.0);
        Ok(())
    }
    /// The wrapper hides its inner exchange, so the counts come back through the stats.
    fn profile_stats(&self) -> Option<ExchangeStats> {
        Some(ExchangeStats { elapsed_ns: 0, sent: self.steps.len() as u64, received: self.points.len() as u64, exchanges: self.flash.len() as u64 })
    }
    fn collective_profiling(&self) -> bool { self.profiling }
    fn profile_flash(&mut self, ns: u64) { self.flash.push(ns); }
    fn profile_point(&mut self, kind: usize, ns: u64, ok: bool) { self.points.push((kind, ns, ok)); }
    fn profile_begin_step(&mut self, phase: &'static str, rows: usize) { self.steps.push(format!("begin {phase} {rows}")); }
    fn profile_end_step(&mut self, ok: bool, device_bytes: usize) { self.steps.push(format!("end {ok} {device_bytes}")); }
    fn mailbox_memory(&mut self) -> Option<(*mut u8, usize)> {
        if self.mailbox.is_empty() { None } else { Some((self.mailbox.as_mut_ptr(), self.mailbox.len())) }
    }
}
/// Run every collective once and return the wrapper's view of the recorder.
fn exercise(profiling: bool) -> Profiled<Recorder> {
    let mut wrapped = Profiled::new(Recorder { profiling, ..Default::default() });
    let mut bytes = [1u8, 2, 3];
    wrapped.gather_bytes(&mut bytes, &[0, 1, 3]).unwrap();
    assert_eq!(bytes, [2, 3, 4], "gather results pass through unchanged");
    let mut floats = [1.0f32, -2.0];
    wrapped.gather_f32(&mut floats, &[0, 2]).unwrap();
    assert_eq!(floats, [2.0, -4.0]);
    let mut scratch = [0.0f32; 2];
    wrapped.reduce(&mut floats, &mut scratch).unwrap();
    assert_eq!(floats, [3.0, -3.0]);
    wrapped.profile_point(1, 5, false);
    wrapped.profile_begin_step("decode", 4);
    wrapped.profile_end_step(true, 64);
    wrapped
}

#[test]
fn profiled_times_collectives_and_gates_step_hooks_on_the_inner_switch() {
    let on = exercise(true).profile_stats().unwrap();
    assert_eq!(on.exchanges, 3, "one flash span per collective");
    assert_eq!(on.sent, 2, "begin and end step reach the inner exchange");
    let off = exercise(false).profile_stats().unwrap();
    assert_eq!(off.exchanges, 0, "disabled profiling takes no timestamp");
    assert_eq!(off.sent, 0, "disabled profiling skips the step hooks");
    // Points are measured by the caller and always forwarded.
    assert_eq!((on.received, off.received), (1, 1));
    assert!(exercise(true).collective_profiling() && !exercise(false).collective_profiling());
}

#[test]
fn profiled_returns_inner_errors_unchanged_and_still_times_them() {
    let mut failing = Profiled::new(Recorder { profiling: true, fail: true, ..Default::default() });
    let (mut bytes, mut floats, mut scratch) = ([0u8; 2], [0.0f32; 2], [0.0f32; 2]);
    assert_eq!(failing.gather_bytes(&mut bytes, &[0, 2]).unwrap_err(), "gather_bytes failed");
    assert_eq!(failing.gather_f32(&mut floats, &[0, 2]).unwrap_err(), "gather_f32 failed");
    assert_eq!(failing.reduce(&mut floats, &mut scratch).unwrap_err(), "reduce failed");
    assert_eq!(failing.profile_stats().unwrap().exchanges, 3, "a failed collective is still timed");
}

#[test]
fn default_pipelines_refuse_and_mailbox_memory_passes_through() {
    let mut plain = Profiled::new(Recorder::default());
    let error = plain.exchange_slots_pipeline::<()>(&[(1, 1)], |_, _| Ok(()), |_, _| Ok(()), |_| Ok(())).unwrap_err();
    assert_eq!(error, "transport has no mailbox pipeline");
    let error = plain.gather_slots_pipeline::<()>(&[vec![0, 1]], |_, _, _| Ok(()), |_, _, _, _| Ok(()), |_| Ok(())).unwrap_err();
    assert_eq!(error, "transport has no gather mailbox pipeline");
    assert!(plain.mailbox_memory().is_none());
    let mut mailbox = Profiled::new(Recorder { mailbox: vec![0; 4096], ..Default::default() });
    let (base, bytes) = mailbox.mailbox_memory().unwrap();
    assert!(!base.is_null());
    assert_eq!(bytes, 4096);
}

/// Profiles are allocated once: recording more samples never grows them.
#[test]
fn profile_storage_is_fixed_by_construction() {
    let mut profile = Profile::default();
    let size = profile.bytes();
    assert!(size > std::mem::size_of::<Profile>());
    let kinds = [Kind::Barrier, Kind::Kv, Kind::PrefixKv, Kind::GatherBytes, Kind::GatherF32, Kind::ReduceF32, Kind::Slots, Kind::Pipeline, Kind::Command];
    for round in 0..1000u64 {
        for (i, kind) in kinds.iter().enumerate() {
            for path in [Path::Tcp, Path::Rdma, Path::Mailbox] {
                let bytes = Bytes { sent: round, received: i as u64, control_sent: 1, control_received: 1, pieces: 2 };
                profile.record(*kind, path, round << (i % 40), Parts { producer_ns: round, consumer_ns: 1 }, bytes, round % 7 != 0);
            }
        }
        profile.record_flash(round);
        profile.record_point((round % 3) as usize, u64::MAX - round, true);
    }
    assert_eq!(profile.bytes(), size);
}
