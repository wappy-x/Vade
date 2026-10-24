//! Opt-in decode and per-chunk prefill diagnosis. GPU mode adds events and
//! metadata probes; its host gaps and wall times include instrumentation.
#[cfg(target_os = "linux")]
use std::cell::RefCell;
use std::{ffi::c_long, time::Instant};

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut c_long) -> i32;
    fn sched_setaffinity(pid: i32, bytes: usize, mask: *const u64) -> i32;
}
#[derive(Default)]
struct Window {
    seen: usize, steps: u128, wall: u128, cpu: [u128; 4], cpu_max: [u128; 4],
    waits: [u128; 2], event: [[u128; 6]; 2], tails: [Vec<u128>; 2],
    gpu: [u128; 5], bytes: [u128; 4],
    classes: Vec<(&'static str, usize, [u128; 4])>,
}
struct Prefill {
    position: usize, rows: usize, rank: usize, previous: bool, start: Instant,
    names: &'static [&'static str], layer: &'static str, scope: Option<usize>,
    values: Vec<(&'static str, usize, [u128; 4])>, last: Option<Instant>,
    generic: bool, decode: bool, ffn_class: usize,
    next_rows: Option<bool>,
    host_api: [u128; 3],
}
impl Prefill {
    fn slot(&mut self, class: usize) -> usize {
        if let Some(i) = self.values.iter().position(|v| (v.0, v.1) == (self.layer, class)) { return i; }
        self.values.push((self.layer, class, [0; 4])); self.values.len() - 1
    }
    fn class(&mut self, kind: usize, expert: bool) -> usize {
        let class = self.scope.filter(|_| self.next_rows.is_none()).unwrap_or(
            if self.next_rows.is_some() {
                match kind {
                    0 | 14..=17 | 20 | 27..=29 => 0,
                    18 | 19 | 23..=26 => 1,
                    2..=5 => 2,
                    21 | 22 => 3,
                    13 => 4,
                    _ => 5,
                }
            } else if self.generic {
                match kind {
                    13 => 8,
                    21 | 22 => 5,
                    18 => 3,
                    19 => 4,
                    1 | 7 => 7,
                    2..=5 | 11 => 2,
                    6 | 16 | 17 | 20 => self.ffn_class,
                    0 | 14 | 15 | 27..=29 => 0,
                    _ => 10,
                }
            } else if kind == 13 {
                8
            } else if expert {
                if kind == 0 { 5 } else { 4 }
            } else if kind == 18 || kind == 19 || (23..27).contains(&kind) {
                1
            } else if (2..6).contains(&kind) {
                2
            } else if kind == 1 {
                7
            } else if kind == 0
                || (14..18).contains(&kind)
                || (20..23).contains(&kind)
                || (27..30).contains(&kind)
            {
                0
            } else {
                10
            },
        );
        self.slot(class) + 5
    }
    fn bytes(&mut self, class: usize, bytes: impl FnOnce() -> usize) {
        if self.next_rows.is_some() {
            return;
        }
        let i = self.slot(class); self.values[i].2[1] += bytes() as u128;
    }
    fn enqueue(&mut self, class: usize, start: Instant, launch_ns: u128) {
        let v = &mut self.values[class - 5].2;
        if let Some(last) = self.last { v[2] += start.saturating_duration_since(last).as_nanos(); }
        v[3] += launch_ns; self.last = Some(start + std::time::Duration::from_nanos(launch_ns.min(u64::MAX as u128) as u64));
    }
}
pub(super) struct DecodeProfile {
    gpu: bool, enabled: bool, core: Option<usize>, pinned: bool,
    windows: [Window; 2], active: Option<usize>, rank: usize,
    start: Option<(Instant, [u128; 4])>, completion: Option<(usize, Instant)>,
    pub(super) previous: bool,
    prefill: Option<Prefill>,
    generic_depth: usize,
}
fn cpu_usage(read: impl FnOnce(i32, *mut c_long) -> i32) -> Result<[u128; 3], String> {
    // Linux LP64 rusage: two timevals followed by fourteen long counters.
    let mut r = [0 as c_long; 18];
    if read(1, r.as_mut_ptr()) != 0 { return Err(std::io::Error::last_os_error().to_string()); } // RUSAGE_THREAD
    Ok([((r[0] + r[2]) as u128 * 1_000_000 + (r[1] + r[3]) as u128) * 1000, r[16] as u128, r[17] as u128])
}
pub(super) fn fp8_read_bytes(rows: usize, cols: usize) -> usize { rows * cols + rows.div_ceil(128) * cols.div_ceil(128) * 4 }
pub(super) fn selected_positions(length: usize, ids: &[u32]) -> Result<usize, String> {
    if length / 4 <= 512 { return Ok(length); }
    if ids.len() != 512 || ids.iter().any(|&id| id as usize >= length / 4) || ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err("decode byte probe: invalid QSA selection".into());
    }
    Ok(ids.len() * 4 + length % 4)
}
#[cfg(target_os = "linux")]
fn usage() -> Result<[u128; 4], String> {
    let cpu = cpu_usage(|who, r| unsafe { getrusage(who, r) })?;
    let sched = std::fs::read_to_string("/proc/thread-self/sched").map_err(|e| e.to_string())?;
    let migrations = sched.lines().find_map(|l| {
        let (name, value) = l.split_once(':')?;
        (name.trim() == "se.nr_migrations").then(|| value.trim().parse::<u128>().ok()).flatten()
    }).ok_or("thread migration counter unavailable")?;
    Ok([cpu[0], cpu[1], cpu[2], migrations])
}
impl DecodeProfile {
    #[cfg(target_os = "linux")]
    pub(super) fn configured() -> Result<Option<Box<RefCell<Self>>>, String> {
        let mode = std::env::var("VADE_DECODE_PROFILE").unwrap_or_default();
        let core = std::env::var("VADE_DECODE_CPU")
            .ok()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| "VADE_DECODE_CPU must be a core index".to_string())
            })
            .transpose()?;
        if !matches!(mode.as_str(), "" | "0" | "host" | "gpu") {
            return Err("VADE_DECODE_PROFILE must be host, gpu or 0".into());
        }
        if core.is_some_and(|c| c >= 1024) {
            return Err("VADE_DECODE_CPU exceeds affinity mask".into());
        }
        if matches!(mode.as_str(), "" | "0") && core.is_none() {
            return Ok(None);
        }
        Ok(Some(Box::new(RefCell::new(Self {
            gpu: mode == "gpu",
            enabled: mode == "host" || mode == "gpu",
            core,
            pinned: false,
            windows: Default::default(),
            active: None,
            rank: 0,
            start: None,
            completion: None,
            previous: false,
            prefill: None,
            generic_depth: 0,
        }))))
    }
    pub(super) fn prefill_begin(
        &mut self,
        position: usize,
        rows: usize,
        rank: usize,
        previous: bool,
        names: &'static [&'static str],
    ) -> bool {
        if !self.enabled || !self.gpu {
            return false;
        }
        self.prefill = Some(Prefill {
            position,
            rows,
            rank,
            previous,
            start: Instant::now(),
            names,
            layer: "input",
            scope: None,
            values: Vec::new(),
            last: None,
            generic: false,
            decode: false,
            ffn_class: 1,
            next_rows: None,
            host_api: [0; 3],
        });
        true
    }
    pub(super) fn prefill_active(&self) -> bool {
        self.prefill.is_some()
    }
    pub(super) fn generic_active(&self) -> bool {
        self.prefill.as_ref().is_some_and(|p| p.generic)
    }
    pub(super) fn generic_enter(&mut self) -> bool {
        self.generic_depth += 1;
        self.generic_depth == 1
    }
    pub(super) fn generic_leave(&mut self) -> bool {
        self.generic_depth -= 1;
        self.generic_depth == 0
    }
    pub(super) fn gpu_enabled(&self) -> bool {
        self.enabled && self.gpu
    }
    pub(super) fn next_rows(&mut self, verify: bool) {
        if let Some(p) = &mut self.prefill {
            p.next_rows = Some(verify);
        }
    }
    pub(super) fn next_rows_host(&mut self, values: [u128; 3]) {
        if let Some(p) = &mut self.prefill {
            p.host_api = values;
        }
    }
    pub(super) fn generic_begin(&mut self, decode: bool, ffn_class: usize) { if let Some(p) = &mut self.prefill { p.generic = true; p.decode = decode; p.ffn_class = ffn_class; } }
    pub(super) fn generic_bytes(&mut self, read: impl FnOnce() -> (usize, usize)) {
        if self.generic_active() { let (class, bytes) = read(); self.prefill_bytes(class, || bytes); }
    }
    pub(super) fn prefill_event_total(&self) -> u128 { self.prefill.as_ref().map_or(0, |p| p.values.iter().map(|v| v.2[0]).sum()) }
    pub(super) fn prefill_layer(&mut self, layer: &'static str) { if let Some(p) = &mut self.prefill { p.layer = layer; } }
    pub(super) fn prefill_scope(&mut self, scope: Option<usize>) -> Option<usize> { self.prefill.as_mut().and_then(|p| std::mem::replace(&mut p.scope, scope)) }
    pub(super) fn prefill_bytes(&mut self, class: usize, bytes: impl FnOnce() -> usize) { if let Some(p) = &mut self.prefill { p.bytes(class, bytes); } }
    pub(super) fn prefill_enqueue(&mut self, class: Option<usize>, start: Instant, ns: u128) {
        if let (Some(p), Some(class)) = (&mut self.prefill, class.filter(|c| *c >= 5)) { p.enqueue(class, start, ns); }
    }
    pub(super) fn prefill_end(&mut self, ok: bool, device_bytes: usize) -> Option<bool> {
        let p = self.prefill.take()?;
        let wall = p.start.elapsed().as_nanos();
        if let Some(verify) = p.next_rows {
            let mut stages = [0u128; 6];
            let (mut gaps, mut launches) = (0, 0);
            for (_, class, v) in p.values {
                stages[class] += v[0];
                gaps += v[2];
                launches += v[3];
            }
            eprintln!(
                "next_rows_profile mode={} position={} rows={} dense_ns={} gdn_ns={} attention_ns={} experts_ns={} head_ns={} other_gpu_ns={} host_ns={} host_wait_ns={} host_launch_ns={} host_profiler_ns={} host_enqueue_ns={} host_gap_ns={} wall_ns={wall} device_bytes={device_bytes} ok={ok} gpu_instrumented=true host_scope=launch_and_wait_apis_overlaps_gpu",
                if verify { "verify" } else { "plain" },
                p.position,
                p.rows,
                stages[0],
                stages[1],
                stages[2],
                stages[3],
                stages[4],
                stages[5],
                p.host_api[0] + p.host_api[1],
                p.host_api[0],
                p.host_api[1],
                p.host_api[2],
                launches,
                gaps
            );
            return Some(p.previous);
        }
        if p.decode {
            let w = &mut self.windows[0];
            for (layer, class, values) in p.values {
                let at = w.classes.iter().position(|v| (v.0, v.1) == (layer, class)).unwrap_or_else(|| {
                    w.classes.push((layer, class, [0; 4])); w.classes.len() - 1 });
                for (total, value) in w.classes[at].2.iter_mut().zip(values) { *total += value; }
            }
            if w.steps == 63 { for (layer, class, v) in &w.classes {
                eprintln!("decode_profile rank={} mode=plain probe=gpu steps=64 warmup=8 layer_class={layer} class={} event_ns_per_step={} logical_read_bytes_per_step={} host_enqueue_gap_ns_per_step={} host_launch_ns_per_step={} device_bytes={device_bytes} ok={ok} byte_scope=logical_useful_inputs unknown_bytes=unprobed_activations_scratch gpu_instrumented=true",
                    p.rank, p.names[*class], v[0]/64, v[1]/64, v[2]/64, v[3]/64);
            } }
            return Some(p.previous);
        }
        for (layer, class, v) in p.values {
            eprintln!("prefill_profile rank={} mode=prefill probe=gpu position={} rows={} layer_class={layer} class={} event_ns_per_chunk={} logical_read_bytes_per_chunk={} host_enqueue_gap_ns_per_chunk={} host_launch_ns_per_chunk={} wall_ns_per_chunk={wall} device_bytes={device_bytes} ok={ok} byte_scope=logical_useful_inputs unknown_bytes=unprobed_activations_scratch gpu_instrumented=true",
                p.rank, p.position, p.rows, p.names[class], v[0], v[1], v[2], v[3]);
        }
        Some(p.previous)
    }
    #[cfg(target_os = "linux")]
    pub(super) fn begin(&mut self, mode: usize, rank: usize, previous: bool) -> Result<bool, String> {
        if !self.pinned {
            if let Some(core) = self.core {
                let mut mask = [0u64; 16]; mask[core / 64] = 1 << (core % 64);
                if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
                    return Err(format!("decode affinity: {}", std::io::Error::last_os_error()));
                }
            }
            self.pinned = true;
        }
        let w = &mut self.windows[mode]; w.seen += 1;
        if !self.enabled || w.seen <= 8 || w.steps == 64 { return Ok(false); }
        self.start = Some((Instant::now(), usage()?));
        self.active = Some(mode); self.rank = rank; self.previous = previous;
        Ok(self.gpu)
    }
    pub(super) fn timer(&self, class: usize) -> Option<Instant> {
        (self.active.is_some() && class < 2).then(Instant::now)
    }
    pub(super) fn event(&mut self, class: usize, metric: usize, start: Option<Instant>) {
        if let Some(start) = start { self.windows[self.active.unwrap()].event[class][metric] += start.elapsed().as_nanos(); }
    }
    pub(super) fn completed(&mut self, class: usize, at: Option<Instant>) {
        if let Some(at) = at { self.windows[self.active.unwrap()].waits[class] += 1; self.completion = Some((class, at)); }
    }
    pub(super) fn enqueue(&mut self) {
        if let Some((class, at)) = self.completion.take() {
            let ns = at.elapsed().as_nanos(); let w = &mut self.windows[self.active.unwrap()];
            w.event[class][5] += ns; w.tails[class].push(ns);
        }
    }
    pub(super) fn gpu_class(&mut self, kind: usize, expert: bool) -> Option<usize> {
        if let Some(p) = &mut self.prefill { return Some(p.class(kind, expert)); }
        (self.active.is_some() && self.gpu).then_some(if kind == 13 { 3 } else if expert { 1 }
            else if kind == 0 || (14..18).contains(&kind) { 0 }
            else if (2..6).contains(&kind) || kind == 18 || kind == 19 || (23..27).contains(&kind) { 2 } else { 4 })
    }
    pub(super) fn kernel(&mut self, class: Option<usize>, ns: u128) {
        if let (Some(p), Some(class)) = (&mut self.prefill, class.filter(|c| *c >= 5)) { p.values[class - 5].2[0] += ns; return; }
        if let (Some(class), Some(mode)) = (class, self.active) { self.windows[mode].gpu[class] += ns; }
    }
    pub(super) fn bytes(&mut self, class: usize, bytes: impl FnOnce() -> usize) {
        if self.generic_active() { return; } // Generic plans supply their own compulsory-read census.
        if let Some(p) = &mut self.prefill { let class = match class { 0 => p.scope.unwrap_or(0), 1 => 5, 2 => p.scope.unwrap_or(2), _ => 8 }; p.bytes(class, bytes); return; }
        if self.gpu { if let Some(mode) = self.active { self.windows[mode].bytes[class] += bytes() as u128; } }
    }
    pub(super) fn collecting_bytes(&self) -> bool { self.gpu && (self.active.is_some() || self.prefill.is_some()) }
    #[cfg(target_os = "linux")]
    pub(super) fn end(&mut self, ok: bool, device_bytes: usize) -> Result<Option<bool>, String> {
        let Some(mode) = self.active.take() else { return Ok(None); };
        self.completion = None;
        let (start, before) = self.start.take().unwrap(); let wall = start.elapsed().as_nanos();
        let after = usage()?; let w = &mut self.windows[mode];
        if !ok { self.enabled = false; eprintln!("decode_profile aborted rank={} mode={mode}", self.rank); return Ok(Some(self.previous)); }
        w.steps += 1; w.wall += wall;
        for i in 0..4 { let delta = after[i].saturating_sub(before[i]); w.cpu[i] += delta; w.cpu_max[i] = w.cpu_max[i].max(delta); }
        if w.steps == 64 && w.classes.is_empty() {
            let mean = |a: &[u128]| a.iter().map(|n| *n as f64 / 64.0).collect::<Vec<_>>();
            let tails: Vec<_> = w.tails.iter_mut().map(|v| { v.sort_unstable();
                [v.get(v.len().saturating_sub(1) * 95 / 100).copied().unwrap_or(0), v.last().copied().unwrap_or(0)] }).collect();
            eprintln!("decode_profile rank={} mode={} probe={} core={:?} steps=64 warmup=8 device_bytes={device_bytes} wall_ns_per_step={} cpu_ns_cs_vol_cs_invol_migrations_per_step={:?} cpu_counters_max_step={:?} waits_moe_attention_per_step={:?} event_create_record_destroy_query_sync_enqueue_ns_per_step={:?} enqueue_p95_max_ns={tails:?} gpu_dense_experts_attention_gdn_head_other_ns_per_step={:?} logical_read_bytes_dense_experts_attention_head_per_step={:?} byte_scope=weights_scales_state_kv byte_probe=gpu_only unknown_bytes=activations_other gpu_instrumented={}",
                self.rank, ["plain", "verify2_rows3"][mode], if self.gpu {"gpu"} else {"host"}, self.core, w.wall / 64,
                mean(&w.cpu), w.cpu_max, mean(&w.waits), w.event.map(|a| a.map(|n| n / 64)), mean(&w.gpu), mean(&w.bytes), self.gpu);
        }
        Ok(Some(self.previous))
    }
}
#[cfg(test)]
#[test]
fn thread_cpu_accounting() {
    assert_eq!(cpu_usage(|who, out| { assert_eq!(who, 1); let r = unsafe { std::slice::from_raw_parts_mut(out, 18) };
        (r[0], r[1], r[2], r[3], r[16], r[17]) = (2, 7, 3, 11, 13, 17); 0 }).unwrap(), [5_000_018_000, 13, 17]);
    assert!(cpu_usage(|who, _| { assert_eq!(who, 1); -1 }).is_err());
}
#[cfg(test)]
#[test]
fn prefill_tags_and_disabled_probes() {
    let mut p = DecodeProfile { gpu: true, enabled: false, core: None, pinned: false, windows: Default::default(),
        active: None, rank: 0, start: None, completion: None, previous: false, prefill: None, generic_depth: 0 };
    assert!(!p.prefill_begin(0, 8192, 0, false, &["a"; 11]));
    p.bytes(0, || panic!("disabled byte probe")); p.prefill_bytes(0, || panic!("disabled prefill probe"));
    p.enabled = true; assert!(p.prefill_begin(0, 4096, 1, true, &["a"; 11]));
    p.prefill_layer("gdn"); let dense = p.gpu_class(0, false).unwrap(); p.bytes(0, || 128);
    p.prefill_layer("qsa"); let attention = p.gpu_class(3, false).unwrap(); p.kernel(Some(dense), 99);
    p.prefill_scope(Some(6)); let shared = p.gpu_class(0, false).unwrap(); p.bytes(0, || 7);
    let at = Instant::now(); p.prefill_enqueue(Some(attention), at, 100); p.prefill_enqueue(Some(shared), at + std::time::Duration::from_nanos(500), 20);
    let v = &p.prefill.as_ref().unwrap().values;
    assert_eq!(v[dense - 5], ("gdn", 0, [99, 128, 0, 0])); assert_eq!(v[shared - 5], ("qsa", 6, [0, 7, 400, 20]));
    assert_eq!(p.prefill_end(false, 0), Some(true)); assert!(!p.collecting_bytes());
    p.generic_bytes(|| panic!("OFF generic shape probe")); p.enabled = false; assert!(!p.prefill_begin(0, 1, 0, false, &["a"; 11]));
    p.enabled = true; p.prefill_begin(0, 1, 0, false, &["a"; 11]); p.generic_begin(false, 1);
    for (kind, expected) in [(21, 5), (22, 5), (18, 3), (19, 4), (3, 2), (7, 7), (13, 8)] { let tag = p.gpu_class(kind, false).unwrap(); assert_eq!(p.prefill.as_ref().unwrap().values[tag - 5].1, expected);
    }
    p.generic_bytes(|| (5, 123)); assert_eq!(p.prefill.as_ref().unwrap().values[0].2[1], 123);
    assert!(p.generic_enter()); assert!(!p.generic_enter()); assert!(!p.generic_leave()); assert!(p.generic_leave());
    p.prefill.as_mut().unwrap().decode = true; assert_eq!(p.prefill_end(false, 0), Some(false)); assert_eq!(p.windows[0].classes[0], ("input", 5, [0, 123, 0, 0]));
    p.prefill_begin(
        19,
        3,
        0,
        false,
        &["dense", "gdn", "attention", "experts", "head", "other"],
    );
    p.next_rows(true);
    p.prefill_scope(Some(7));
    p.prefill_bytes(7, || {
        panic!("row timers do not collect legacy census classes")
    });
    for (kind, expected) in [(0, 0), (18, 1), (3, 2), (21, 3), (13, 4), (12, 5)] {
        let tag = p.gpu_class(kind, false).unwrap();
        p.kernel(Some(tag), 17);
        assert_eq!(p.prefill.as_ref().unwrap().values[tag - 5].1, expected);
    }
    assert_eq!(p.prefill_end(true, 4096), Some(false));
}
