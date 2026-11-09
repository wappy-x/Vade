//! Layer-stepped prefill with replicated weights and ring K/V exchange.
use crate::{Partition, Result, Ring, TransportStats};
use spark_engine::qwen::Session;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct PrefillStats {
    pub elapsed: Duration, pub begin: Duration, pub project: Duration,
    pub export: Duration, pub exchange: Duration, pub import: Duration,
    pub attention_ffn: Duration, pub head: Duration, pub overlapped: Duration,
    pub schedule_wait: Duration,
    pub allocation: Duration, pub gpu_wait: Duration,
    pub host_exchange_bytes: usize, pub transport: TransportStats,
}

impl PrefillStats {
    pub fn report(&self, mode: &str, rank: usize) {
        eprintln!("prefill_phases mode={mode} rank={rank} begin_ms={:.3} project_ms={:.3} export_ms={:.3} exchange_ms={:.3} import_ms={:.3} attention_ffn_ms={:.3} head_ms={:.3} rendezvous_ms={:.3} allocation_ms={:.3} gpu_wait_ms={:.3} overlap_ms={:.3} sent_bytes={} received_bytes={}",
            self.begin.as_secs_f64()*1000.0,self.project.as_secs_f64()*1000.0,self.export.as_secs_f64()*1000.0,
            self.exchange.as_secs_f64()*1000.0,self.import.as_secs_f64()*1000.0,self.attention_ffn.as_secs_f64()*1000.0,
            self.head.as_secs_f64()*1000.0,self.schedule_wait.as_secs_f64()*1000.0,self.allocation.as_secs_f64()*1000.0,
            self.gpu_wait.as_secs_f64()*1000.0,self.overlapped.as_secs_f64()*1000.0,self.transport.sent,self.transport.received);
    }
    pub fn add_backend(&mut self, stats: spark_gpu_probe::driver::BackendStats) {
        self.allocation += Duration::from_nanos(stats.allocation_ns.min(u64::MAX as u128) as u64);
        self.gpu_wait += Duration::from_nanos(stats.stream_wait_ns.min(u64::MAX as u128) as u64);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillOptions {
    pub overlap: bool,
    /// Profile replicated ranks sharing one GPU. Only one rank submits GPU
    /// work at a time; exchange remains collective. This is not a TTFT run.
    pub serialize_compute: bool,
}

impl Default for PrefillOptions {
    fn default() -> Self { Self { overlap: true, serialize_compute: false } }
}

// Every work closure completes its GPU work before handing off the turn.
fn in_turn(ring: &mut Option<&mut Ring>, serial: bool, work: impl FnOnce() -> Result<()>) -> Result<Duration> {
    if !serial || ring.is_none() { work()?; return Ok(Duration::ZERO); }
    let ring = ring.as_deref_mut().unwrap();
    let start = Instant::now();
    let mut work = Some(work);
    let mut compute = Duration::ZERO;
    for owner in 0..ring.world() {
        if ring.rank() == owner {
            let phase = Instant::now();
            work.take().unwrap()()?;
            compute = phase.elapsed();
        }
        ring.barrier()?;
    }
    Ok(start.elapsed().saturating_sub(compute))
}

/// Process one contiguous slice per rank. Each rank retains its causal prefix;
/// the last rank has the complete cache and the prompt's final-row logits.
/// The first rank overlaps attention/FFN with its send; the other ranks overlap
/// their receive with the preceding layer and local projection. CUDA work stays
/// on the calling thread. No weights or KV are quantized.
/// All ranks must supply the same prompt/partition and an empty FP32 session.
/// After a failure, reset sessions and reconnect every rank before retrying.
pub fn prefill(session: &mut Session<'_, '_, '_>, prompt: &[u32], ring: Option<&mut Ring>,
    partition: &Partition, overlap: bool) -> Result<(Vec<f32>, PrefillStats)>
{
    prefill_with_options(session, prompt, ring, partition, PrefillOptions { overlap, ..Default::default() })
}

pub fn prefill_with_options(session: &mut Session<'_, '_, '_>, prompt: &[u32], ring: Option<&mut Ring>,
    partition: &Partition, options: PrefillOptions) -> Result<(Vec<f32>, PrefillStats)>
{
    let mut tail = Duration::ZERO;
    let (logits, mut stats) = prefill_with_finish(session, prompt, ring, partition, options, |session, pending_layer| {
        if let Some(layer) = pending_layer {
            let start = Instant::now();
            session.sequence_finish_layer(layer)?;
            tail = start.elapsed();
        }
        session.sequence_finish()
    })?;
    stats.head = stats.head.saturating_sub(tail);
    stats.attention_ffn += tail;
    Ok((logits, stats))
}

/// Finish the final layer and projection while the caller overlaps cache hand-off.
/// The caller must finish the optional pending layer, then the sequence, on this
/// thread before returning logits. All cache rows are ready before this callback.
pub fn prefill_with_finish<'a>(session: &mut Session<'_, 'a, '_>, prompt: &[u32], mut ring: Option<&mut Ring>,
    partition: &Partition, options: PrefillOptions,
    finish: impl FnOnce(&mut Session<'_, 'a, '_>, Option<usize>) -> Result<Vec<f32>>,
) -> Result<(Vec<f32>, PrefillStats)>
{
    let rank = ring.as_ref().map_or(0, |r| r.rank());
    let world = ring.as_ref().map_or(1, |r| r.world());
    if world != partition.world() || prompt.len() != partition.tokens() {
        return Err("prefill prompt, partition, and ring shapes differ".into());
    }
    if partition.offsets()[1..world].iter().any(|offset| offset % Partition::ALIGNMENT != 0) {
        return Err("prefill partition boundaries must align to attention tiles".into());
    }
    let local = partition.range(rank);
    if local.is_empty() { return Err("each prefill rank must own at least one token".into()); }
    let layers = session.config().layers as usize;
    let width = (session.config().kv_heads * session.config().head_dim) as usize;
    let start = Instant::now();
    let device_before = session.backend_stats();
    let exchange_tokens = if world > 1 { ring.as_ref().unwrap().prefix_tokens(partition)? } else { 0 };
    let exchange_len = exchange_tokens.checked_mul(width).ok_or("K/V exchange size overflow")?;
    let mut staging = ring.as_deref_mut().map(|r| std::mem::take(&mut r.staging)).unwrap_or_default();
    staging.reserve(exchange_len);
    let mut stats = PrefillStats { host_exchange_bytes: staging.keys.len() * 8, ..Default::default() };
    let receiving = options.overlap && !options.serialize_compute && rank > 0;
    let tail = receiving && rank + 1 == world;
    if let Some(ring) = ring.as_deref_mut() { ring.retain_regions(true); }
    // Keep registrations inside the staging allocation's lifetime even when a
    // session operation fails or a debug build unwinds.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Vec<f32>> {
        let mut begin = || {
            let phase = Instant::now();
            session.sequence_begin(&prompt[local.clone()], local.start, local.end, prompt.len())?;
            stats.begin = phase.elapsed();
            Ok(())
        };
        if !options.serialize_compute {
            if let Some(ring) = ring.as_deref_mut() {
                // Host page registration can run while CUDA allocates the
                // existing cache and embeds the first local prompt slice.
                std::thread::scope(|scope| {
                    let prepare = scope.spawn(|| unsafe { ring.prepare_staging(&mut staging) });
                    let result = begin();
                    prepare.join().map_err(|_| "K/V preparation worker panicked".to_string())??;
                    result
                })?;
            } else { begin()?; }
        } else {
            if let Some(ring) = ring.as_deref_mut() {
                unsafe { ring.prepare_staging(&mut staging)?; }
            }
            stats.schedule_wait += in_turn(&mut ring, true, begin)?;
        }
        let (keys, values) = (&mut staging.keys[..exchange_len], &mut staging.values[..exchange_len]);
        for layer in 0..layers {
            if !receiving {
                stats.schedule_wait += in_turn(&mut ring, options.serialize_compute, || {
                    let phase = Instant::now();
                    session.sequence_project(layer)?;
                    stats.project += phase.elapsed();
                    // No rank consumes the final slice except its local GPU.
                    if rank + 1 < world {
                        let phase = Instant::now();
                        session.sequence_export_into(layer, local.start,
                            &mut keys[local.start * width..local.end * width], &mut values[local.start * width..local.end * width])?;
                        stats.export += phase.elapsed();
                    }
                    Ok(())
                })?;
            }
            let mut finished = false;
            if let Some(ring) = ring.as_deref_mut().filter(|_| world > 1) {
                let before = ring.stats;
                let phase = Instant::now();
                if options.overlap && !options.serialize_compute && rank == 0 {
                    let compute = ring.exchange_prefix_kv_while(keys, values, partition, width, || {
                        let phase = Instant::now();
                        session.sequence_finish_layer(layer)?;
                        Ok(phase.elapsed())
                    })?;
                    let exchange = Duration::from_nanos(ring.stats.since(before).elapsed_ns.min(u64::MAX as u128) as u64);
                    stats.overlapped += (exchange + compute).saturating_sub(phase.elapsed());
                    stats.exchange += exchange;
                    stats.attention_ffn += compute;
                    finished = true;
                } else if tail {
                    let (compute, project) = ring.exchange_prefix_kv_while(keys, values, partition, width, || {
                        let phase = Instant::now();
                        if layer > 0 { session.sequence_finish_layer(layer - 1)?; }
                        let compute = phase.elapsed();
                        let phase = Instant::now();
                        session.sequence_project(layer)?;
                        Ok((compute, phase.elapsed()))
                    })?;
                    let exchange = Duration::from_nanos(ring.stats.since(before).elapsed_ns.min(u64::MAX as u128) as u64);
                    stats.overlapped += (exchange + compute + project).saturating_sub(phase.elapsed());
                    stats.exchange += exchange;
                    stats.attention_ffn += compute;
                    stats.project += project;
                    finished = true;
                } else if receiving {
                    let (compute, project, export) = ring.exchange_prefix_kv_producing(keys, values, partition, width, |keys, values| {
                        let phase = Instant::now();
                        if layer > 0 { session.sequence_finish_layer(layer - 1)?; }
                        let compute = phase.elapsed();
                        let phase = Instant::now();
                        session.sequence_project(layer)?;
                        let project = phase.elapsed();
                        let phase = Instant::now();
                        session.sequence_export_into(layer, local.start, keys, values)?;
                        Ok((compute, project, phase.elapsed()))
                    })?;
                    let exchange = Duration::from_nanos(ring.stats.since(before).elapsed_ns.min(u64::MAX as u128) as u64);
                    stats.overlapped += (exchange + compute + project + export).saturating_sub(phase.elapsed());
                    stats.exchange += exchange;
                    stats.attention_ffn += compute;
                    stats.project += project;
                    stats.export += export;
                    finished = true;
                } else {
                    ring.exchange_prefix_kv(keys, values, partition, width)?;
                    stats.exchange += phase.elapsed();
                }
                let delta = ring.stats.since(before);
                stats.transport.sent += delta.sent;
                stats.transport.received += delta.received;
                stats.transport.exchanges += delta.exchanges;
                stats.transport.elapsed_ns += delta.elapsed_ns;
                stats.transport.send_ns += delta.send_ns;
                stats.transport.recv_ns += delta.recv_ns;
                stats.transport.wait_ns += delta.wait_ns;
            }
            stats.schedule_wait += in_turn(&mut ring, options.serialize_compute, || {
                let phase = Instant::now();
                // Each upload fits the backend's existing staging allocation;
                // whole peer slices would leave prompt-sized pinned storage.
                let rows = (2 * 1024 * 1024 / std::mem::size_of::<f32>() / width).max(1);
                for start in (0..local.start).step_by(rows) {
                    let end = (start + rows).min(local.start);
                    session.sequence_import(layer, start,
                        &keys[start * width..end * width], &values[start * width..end * width])?;
                }
                stats.import += phase.elapsed();
                if !finished {
                    let phase = Instant::now();
                    session.sequence_finish_layer(layer)?;
                    stats.attention_ffn += phase.elapsed();
                }
                Ok(())
            })?;
        }
        let pending_layer = (receiving && layers > 0).then(|| layers - 1);
        let mut logits = Vec::new();
        stats.schedule_wait += in_turn(&mut ring, options.serialize_compute, || {
            let phase = Instant::now();
            logits = finish(session, pending_layer)?;
            stats.head = phase.elapsed();
            Ok(())
        })?;
        Ok(logits)
    }));
    if let Some(ring) = ring.as_deref_mut() {
        ring.retain_regions(false);
        ring.staging = staging;
    }
    let logits = match result {
        Ok(result) => result?,
        Err(panic) => std::panic::resume_unwind(panic),
    };
    stats.elapsed = start.elapsed();
    stats.add_backend(session.backend_stats().since(device_before));
    Ok((logits, stats))
}
