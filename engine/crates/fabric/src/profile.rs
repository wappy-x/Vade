//! Bounded collective profiles. No allocation, sorting, locks or output per sample.
use std::{fmt::{self, Write as _}, io::{self, Write as _}, sync::OnceLock, time::Instant};

/// Format before taking the stderr lock so each enabled profile block is one write.
pub fn write_profile(output: &str) {
    io::stderr().lock().write_all(output.as_bytes()).expect("failed to write profile");
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SPARK_PROFILE_COLLECTIVES").as_deref() == Ok("1"))
}

/// Preserve SPARK_PROFILE CUDA layer events and existing parser output.
pub fn layers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SPARK_PROFILE").as_deref() == Ok("1"))
}

#[derive(Clone, Copy)]
pub enum Kind { Barrier, Kv, PrefixKv, GatherBytes, GatherF32, ReduceF32, Slots, Pipeline, Command }
const KINDS: [&str; 9] = ["barrier", "kv", "prefix_kv", "gather_bytes", "gather_f32", "reduce_f32", "slots", "slots_pipeline", "command"];
#[derive(Clone, Copy)]
pub enum Path { Tcp, Rdma, Mailbox }
const PATHS: [&str; 3] = ["tcp", "rdma", "mailbox"];

/// Four sub-buckets per power of two, in nanoseconds. Quantiles are bucket
/// upper bounds (at most 25% above a sample, exact below 8 ns), not exact ranks.
#[derive(Clone)]
struct Latency { bins: [u64; 256], count: u64, ns: u128 }
impl Default for Latency {
    fn default() -> Self { Self { bins: [0; 256], count: 0, ns: 0 } }
}
impl Latency {
    fn add(&mut self, ns: u64) {
        let value = ns.max(1);
        let power = 63 - value.leading_zeros() as usize;
        let base = 1u64 << power;
        let part = if power >= 2 { (value - base) >> (power - 2) } else { (value - base) << (2 - power) } as usize;
        self.bins[power * 4 + part] += 1;
        self.count += 1;
        self.ns += ns as u128;
    }
    fn percentile(&self, percent: u64) -> u64 {
        if self.count == 0 { return 0; }
        let target = (self.count as u128 * percent as u128).div_ceil(100);
        let mut count = 0u128;
        for (i, &n) in self.bins.iter().enumerate() {
            count += n as u128;
            if count >= target {
                let base = 1u128 << (i / 4);
                return (base + (base * (i % 4 + 1) as u128).div_ceil(4) - 1)
                    .min(u64::MAX as u128) as u64;
            }
        }
        0
    }
}

/// Sparse bin indices/counts are mergeable across an arbitrary tick window.
impl fmt::Display for Latency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        for (i, n) in self.bins.iter().enumerate().filter(|(_, n)| **n != 0) {
            write!(f, "{separator}{i}:{n}")?;
            separator = ",";
        }
        if separator.is_empty() { f.write_str("-")?; }
        Ok(())
    }
}
#[derive(Clone, Copy, Default)]
pub struct Parts { pub producer_ns: u64, pub consumer_ns: u64 }
const PHASES: [&str; 19] = ["prefill", "decode", "decode_batch", "verify", "verify_sampled",
    "draft", "draft_sampled", "claim_mtp", "decode_logits", "commit", "open", "reserve",
    "close", "stop", "save", "restore", "release", "canonicalize", "disable_mtp"];
struct Phases(u32);
impl fmt::Display for Phases {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        for (i, phase) in PHASES.iter().enumerate() {
            if self.0 & (1 << i) != 0 { write!(f, "{separator}{phase}")?; separator = ","; }
        }
        if separator.is_empty() { f.write_str("-")?; }
        Ok(())
    }
}

#[derive(Default, Clone)]
struct Stats {
    transport: Latency,
    flash: Latency,
    wait: Latency,
    producer: Latency,
    consumer: Latency,
    failed: u64,
    pieces: u64,
    sent: u64,
    received: u64,
    control_sent: u64,
    control_received: u64,
    bytes_min: u64,
    bytes_max: u64,
}
#[derive(Clone, Copy, Default)]
pub struct Bytes { pub sent: u64, pub received: u64, pub control_sent: u64, pub control_received: u64, pub pieces: u64 }
impl Stats {
    fn add(&mut self, ns: u64, parts: Parts, bytes: Bytes, ok: bool) {
        let payload = bytes.sent + bytes.received;
        if self.transport.count == 0 { self.bytes_min = payload; }
        self.bytes_min = self.bytes_min.min(payload);
        self.bytes_max = self.bytes_max.max(payload);
        self.sent += bytes.sent;
        self.received += bytes.received;
        self.control_sent += bytes.control_sent;
        self.control_received += bytes.control_received;
        self.failed += u64::from(!ok);
        self.pieces += bytes.pieces;
        self.transport.add(ns);
        self.producer.add(parts.producer_ns);
        self.consumer.add(parts.consumer_ns);
        self.wait.add(ns.saturating_sub(parts.producer_ns).saturating_sub(parts.consumer_ns));
    }
}

#[derive(Default, Clone)]
struct Point { wall: Latency, failed: u64 }
const POINTS: [&str; 3] = ["reduce_device", "middle", "command_broadcast"];

pub struct Profile {
    step: Box<[Stats]>,
    total: Box<[Stats]>,
    points: [Point; 3],
    point_total: [Point; 3],
    last: Option<usize>,
    active: Option<Instant>,
    tick: u64,
    ticks: u64,
    depth: usize,
    command_scope: bool,
    step_ok: bool,
    phase: &'static str,
    phases: u32,
    rows: usize,
    outside: u64,
}
impl Default for Profile {
    fn default() -> Self {
        Self { step: vec![Stats::default(); KINDS.len() * 3].into_boxed_slice(),
            total: vec![Stats::default(); KINDS.len() * 3].into_boxed_slice(),
            points: Default::default(), point_total: Default::default(), last: None,
            active: None, tick: 0, ticks: 0, depth: 0, command_scope: false,
            step_ok: true, phase: "setup", phases: 0, rows: 0, outside: 0 }
    }
}
impl Profile {
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + (self.step.len() + self.total.len()) * std::mem::size_of::<Stats>()
    }
    pub fn record(&mut self, kind: Kind, path: Path, ns: u64, parts: Parts, bytes: Bytes, ok: bool) {
        let i = kind as usize * 3 + path as usize;
        self.total[i].add(ns, parts, bytes, ok);
        if self.active.is_some() { self.step[i].add(ns, parts, bytes, ok); }
        else if !matches!(kind, Kind::Command) { self.outside += 1; }
        if !matches!(kind, Kind::Command) { self.last = Some(i); }
    }
    /// Flash's caller span includes its transport lock and pipeline callbacks.
    pub fn record_flash(&mut self, ns: u64) {
        if let Some(i) = self.last.take() {
            self.total[i].flash.add(ns);
            if self.active.is_some() { self.step[i].flash.add(ns); }
        }
    }
    /// Full exchange point, including staging/synchronization before transport
    /// and queued H2D/consumer work after it. Never adds a GPU synchronization.
    pub fn record_point(&mut self, kind: usize, ns: u64, ok: bool) {
        self.point_total[kind].wall.add(ns);
        self.point_total[kind].failed += u64::from(!ok);
        if self.active.is_some() {
            self.points[kind].wall.add(ns);
            self.points[kind].failed += u64::from(!ok);
        }
    }
    fn add_phase(&mut self, phase: &'static str) {
        if let Some(i) = PHASES.iter().position(|p| *p == phase) { self.phases |= 1 << i; }
    }
    pub fn begin_step(&mut self, phase: &'static str, rows: usize) {
        if self.active.is_none() {
            self.tick += 1;
            self.phase = phase;
            self.rows = 0;
            self.phases = 0;
            self.step_ok = true;
            self.active = Some(Instant::now());
        }
        if !self.command_scope { self.rows += rows; }
        if phase != "scheduler" { self.add_phase(phase); }
        self.depth += 1;
    }
    pub fn begin_command(&mut self) {
        self.begin_step("command", 0);
        self.command_scope = true;
    }
    pub fn command_phase(&mut self, phase: &'static str, rows: usize) {
        self.add_phase(phase);
        if self.command_scope { self.phase = phase; self.rows = rows; }
    }
    pub fn end_step(&mut self, rank: usize, world: usize, ok: bool, device_bytes: usize) {
        self.step_ok &= ok;
        self.depth = self.depth.saturating_sub(1);
        if self.depth != 0 { return; }
        if let Some(start) = self.active.take() {
            let ns = start.elapsed().as_nanos();
            let collectives = self.step[..Kind::Command as usize * 3].iter();
            let mut output = format!("collective_tick rank={rank} world={world} tick={} phase={} phases={} rows={} collectives={} command_parts={} transfer_pieces={} flash_exchanges={} exchange_points={} command_broadcasts={} elapsed_ns={ns} device_bytes={device_bytes} profile_bytes={} ok={}\n",
                self.tick, self.phase, Phases(self.phases), self.rows,
                collectives.clone().map(|s| s.transport.count).sum::<u64>(),
                self.step[Kind::Command as usize * 3..].iter().map(|s| s.transport.count).sum::<u64>(),
                collectives.clone().map(|s| s.pieces).sum::<u64>(), collectives.map(|s| s.flash.count).sum::<u64>(),
                self.points[..2].iter().map(|s| s.wall.count).sum::<u64>(), self.points[2].wall.count, self.bytes(),
                self.step_ok && self.step.iter().all(|s| s.failed == 0) && self.points.iter().all(|s| s.failed == 0));
            self.append(&mut output, "tick", rank, world, &self.step, &self.points);
            write_profile(&output);
            for s in &mut self.step { if s.transport.count != 0 { *s = Stats::default(); } }
            self.points = Default::default();
            self.command_scope = false;
            self.ticks += 1;
        }
    }
    pub fn begin_tick(&mut self, tick: u64) {
        self.begin_step("scheduler", 0);
        self.tick = tick;
    }
    pub fn end_tick(&mut self, rank: usize, world: usize, device_bytes: usize) {
        self.end_step(rank, world, true, device_bytes);
    }
    pub fn report(&self, rank: usize, world: usize) {
        let collectives = self.total[..Kind::Command as usize * 3].iter();
        let mut output = format!("collective_exit rank={rank} world={world} ticks={} collectives={} outside_tick_collectives={} command_parts={} transfer_pieces={} flash_exchanges={} profile_bytes={} unfinished_tick={}\n",
            self.ticks, collectives.clone().map(|s| s.transport.count).sum::<u64>(), self.outside,
            self.total[Kind::Command as usize * 3..].iter().map(|s| s.transport.count).sum::<u64>(),
            collectives.clone().map(|s| s.pieces).sum::<u64>(), collectives.map(|s| s.flash.count).sum::<u64>(), self.bytes(), self.active.is_some());
        self.append(&mut output, "exit", rank, world, &self.total, &self.point_total);
        write_profile(&output);
    }
    fn append(&self, output: &mut String, scope: &str, rank: usize, world: usize, stats: &[Stats], points: &[Point]) {
        for (i, s) in stats.iter().enumerate().filter(|(_, s)| s.transport.count != 0) {
            writeln!(output, "collective_profile scope={scope} rank={rank} world={world} tick={} kind={} path={} count={} transfer_pieces={} failed={} transport_ns={} transport_p50_upper_ns={} transport_p99_upper_ns={} transport_bins={} wait_ns={} wait_p50_upper_ns={} wait_p99_upper_ns={} wait_bins={} producer_ns={} producer_p50_upper_ns={} producer_p99_upper_ns={} producer_bins={} consumer_ns={} consumer_p50_upper_ns={} consumer_p99_upper_ns={} consumer_bins={} flash_count={} flash_ns={} flash_p50_upper_ns={} flash_p99_upper_ns={} flash_bins={} sent_bytes={} received_bytes={} control_sent_bytes={} control_received_bytes={} payload_bytes_per_collective={:.1} payload_bytes_min={} payload_bytes_max={}",
                self.tick, KINDS[i / 3], PATHS[i % 3], s.transport.count, s.pieces, s.failed,
                s.transport.ns, s.transport.percentile(50), s.transport.percentile(99), s.transport,
                s.wait.ns, s.wait.percentile(50), s.wait.percentile(99), s.wait,
                s.producer.ns, s.producer.percentile(50), s.producer.percentile(99), s.producer,
                s.consumer.ns, s.consumer.percentile(50), s.consumer.percentile(99), s.consumer,
                s.flash.count, s.flash.ns, s.flash.percentile(50), s.flash.percentile(99), s.flash,
                s.sent, s.received, s.control_sent, s.control_received,
                (s.sent + s.received) as f64 / s.transport.count as f64, s.bytes_min, s.bytes_max).unwrap();
        }
        for (i, point) in points.iter().enumerate().filter(|(_, p)| p.wall.count != 0) {
            writeln!(output, "collective_exchange scope={scope} rank={rank} world={world} tick={} kind={} count={} failed={} wall_ns={} wall_p50_upper_ns={} wall_p99_upper_ns={} wall_bins={}",
                self.tick, POINTS[i], point.wall.count, point.failed, point.wall.ns,
                point.wall.percentile(50), point.wall.percentile(99), point.wall).unwrap();
        }
    }
}
