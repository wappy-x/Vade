//! A model-owned graph per padded decode width. Slot tables outlive all graphs;
//! session state allocations remain funded only while their slot is admitted.
use super::*;
use spark_gpu_probe::{
    driver::{DecodeGraph, PromptBuffer},
    next_math::SlotStep,
};

pub(crate) struct Pool<'a> {
    pub decode: BTreeMap<usize, Pooled<'a>>,
    pub used: [bool; 17],
    pub session_limit: usize,
    pub table: DeviceVector<'a>,
    pub capacity: usize,
}
impl<'a> Pool<'a> {
    pub fn new(b: &Backend<'a>) -> Result<Self> {
        let mut table = crate::flash_next_storage::state_slots(b)?;
        b.zero_vector(&mut table)?;
        let mut used = [false; 17];
        used[0] = true;
        Ok(Self {
            used,
            session_limit: POOL_SIZE,
            table,
            capacity: 0,
            decode: BTreeMap::new(),
        })
    }
    pub fn reservation_bytes(capacity: usize, arms: usize, mtp_depth: usize, sessions: usize, frontend: bool) -> Result<usize> {
        if !(1..=POOL_SIZE).contains(&sessions) {
            return Err("Flash-Next session limit must be 1..=16".into());
        }
        let plan = RankPlan::new(0, 1).map_err(err)?;
        [1, 2, 4, 8, 16]
            .iter()
            .filter(|&&n| n <= sessions.next_power_of_two())
            .map(|&n| {
                Ok(Workspace::reservation_bytes(capacity, n, mtp_depth, frontend)
                    + n * QsaScratch::reservation_bytes(&plan, 1, capacity, false)
                    + n * (2 * W + H) * 4
                    + 4096
                    + (n * 38 + 70) * 4
                    + (n * 16 * 90).div_ceil(64) * 64
                    + 256
                    + arms * (spark_gpu_probe::GraphAllowance::whole_model(1, n)?.maximum_bytes()
                        + spark_gpu_probe::GraphAllowance::whole_model(47, n)?.maximum_bytes()))
            })
            .sum()
    }
    /// The decode-graph term of reservation_bytes: the graph storage the shared
    /// reservation funds. Other captures (verify segments) use unreserved bytes.
    pub fn graph_allowance_bytes(arms: usize, sessions: usize) -> Result<usize> {
        if !(1..=POOL_SIZE).contains(&sessions) {
            return Err("Flash-Next session limit must be 1..=16".into());
        }
        [1, 2, 4, 8, 16]
            .iter()
            .filter(|&&n| n <= sessions.next_power_of_two())
            .map(|&n| Ok(arms * (spark_gpu_probe::GraphAllowance::whole_model(1, n)?.maximum_bytes()
                + spark_gpu_probe::GraphAllowance::whole_model(47, n)?.maximum_bytes())))
            .sum()
    }
}
#[cfg(test)]
#[test]
fn graph_allowance_is_the_funded_decode_graph_term() {
    for sessions in [1, 2, 3, 8, 16] {
        let one = Pool::reservation_bytes(262144, 1, 3, sessions, false).unwrap();
        let two = Pool::reservation_bytes(262144, 2, 3, sessions, false).unwrap();
        assert_eq!(two - one, Pool::graph_allowance_bytes(1, sessions).unwrap());
    }
    // One session: whole_model(1, 1) + whole_model(47, 1) at 16 KiB per node + 256 KiB each.
    assert_eq!(Pool::graph_allowance_bytes(1, 1).unwrap(), 51_904_512);
}
/// What a pooled part does after its last layer.
#[derive(Clone, Copy, PartialEq)]
enum Finish {
    /// Nothing: a later part holds the head and the position update.
    Later,
    /// Mixer, the head (the bounded head when `bounded`), positions, for a part that ends at
    /// layer 48. A chain front's tail ends at `split` and does none of these: `fin` runs them.
    Head { bounded: bool },
    /// A pipeline worker: positions only.
    Position,
}
// Drop the graphs before their scratch and template views. One step replays
// (layer 0, layers 1-47): the host reads the step's PLE rows while layer 0 runs;
// they are first used by layer 1's PLE projection. On one Spark, mixer, head and
// positions end the second graph, or run behind it as eager launches when a step's
// reader picks its head (head_after_tail). A chain front's head runs in `fin`.
pub(crate) struct Pooled<'a> {
    graph: [Option<(DecodeGraph<'a>, DecodeGraph<'a>)>; 2],
    capture_retry: [crate::flash_next_rows::CaptureBackoff; 2],
    /// Pipeline front: mixer + head + positions after the back's rows return.
    fin: [Option<DecodeGraph<'a>>; 2],
    /// Pipeline back: layers [split, 48) + positions.
    back: [Option<DecodeGraph<'a>>; 2],
    work: Workspace<'a>,
    slots: DeviceVector<'a>,
    templates: DeviceVector<'a>,
    _qsa: Vec<QsaScratch<'a>>,
    ple: Vec<DeviceVector<'a>>,
    frontend: Vec<DeviceVector<'a>>,
    params: [i64; 35],
    staged: (PromptBuffer<'a>, DeviceVector<'a>),
}
impl<'a> Pooled<'a> {
    fn new(b: &Backend<'a>, model: &NextModel<'a>, capacity: usize, width: usize) -> Result<Self> {
        b.prepare_flash_dense()?;
        b.flash_mixed_planes()?;
        let qsa = (0..width)
            .map(|_| QsaScratch::new(b, &model.weights.plan, 1, capacity))
            .collect::<Result<Vec<_>>>()?;
        let records = qsa
            .iter()
            .map(QsaScratch::pool_template)
            .collect::<Vec<_>>();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                records.as_ptr().cast::<u8>(),
                std::mem::size_of_val(records.as_slice()),
            )
        };
        let mut templates = b.vector(bytes.len() / 4)?;
        b.write_vector_bytes(&mut templates, bytes)?;
        let (mult, sizes, offsets, _) = &model.hash;
        let params: [i64; 35] = [mult.as_slice(), sizes, offsets]
            .concat()
            .try_into()
            .map_err(|_| "unexpected PLE hash shape")?;
        let frontend = {
            let mut v = b.vector_arena(&[width * 6, 70, width * 32])?;
            b.write_vector_bytes(&mut v[1], as_bytes(&params))?;
            v
        };
        Ok(Self {
            graph: [None, None],
            capture_retry: [crate::flash_next_rows::CaptureBackoff::default(); 2],
            fin: [None, None],
            back: [None, None],
            work: Workspace::new(b, model, capacity, width)?,
            slots: b.vector(width)?,
            templates,
            _qsa: qsa,
            frontend,
            params,
            staged: (
                b.prompt_buffer(width * 16 * 90)?,
                b.vector(width * 16 * 90 / 4)?,
            ),
            ple: b.vector_arena(&[width * W, width * H, width * W])?,
        })
    }
    fn enqueue(
        &mut self,
        b: &Backend<'a>,
        model: &NextModel<'a>,
        source: &NextSession<'a, '_>,
        mut step: SlotStep,
        layers: std::ops::Range<usize>,
        finish: Finish,
    ) -> Result<()> {
        let width = step.rows as usize;
        let w = &mut self.work;
        if layers.start == 0 {
            let v = &self.frontend;
            let mut feed = step;
            let (mut input, mut params, mut rows, mut embed, mut out) = (
                v[0].device_address(),
                v[1].device_address(),
                v[2].device_address(),
                model.weights.device_address(EMBED)?,
                w.residual.device_address(),
            );
            unsafe {
                b.launch_flash(
                    "flash_next_rows_feed",
                    (width * W).div_ceil(128) as u32,
                    128,
                    &mut [
                        (&mut input as *mut u64).cast(),
                        (&mut params as *mut u64).cast(),
                        (&mut rows as *mut u64).cast(),
                        (&mut embed as *mut u64).cast(),
                        (&mut out as *mut u64).cast(),
                        (&mut feed as *mut SlotStep).cast(),
                    ],
                )?;
            }
        }
        for li in layers.clone() {
            step.layer = li as u32;
            if li == 1 {
                use gdn::Arg::{F32, Ptr};
                gdn::launch(
                    b,
                    "flash_next_rows_ple_staged",
                    width * H,
                    &mut [
                        Ptr(self.frontend[2].device_address()),
                        Ptr(self.staged.1.device_address()),
                        Ptr(w.lexical.device_address()),
                        F32(model.ple_global),
                    ],
                )?;
                source.ple.project_pool(
                    b,
                    w.residual.device_address(),
                    w.lexical.device_address(),
                    [
                        self.ple[0].device_address(),
                        self.ple[1].device_address(),
                        self.ple[2].device_address(),
                    ],
                    step,
                )?;
            }
            source.attn_hc[li].mix_tiled(b, &mut w.hc, &w.residual, &w.mixed, width)?;
            if li % 4 == 3 {
                let batch = w
                    .qsa_batch
                    .as_mut()
                    .ok_or("missing pooled attention workspace")?;
                batch.pool_project(b, model.qsa_proj[li], &w.mixed, width)?;
                batch.pool_attend(
                    b,
                    &model.weights.plan,
                    &self.templates,
                    model.qsa_norms[li]
                        .as_ref()
                        .ok_or("missing pooled QSA norms")?
                        .device_address(),
                    w.capacity,
                    step,
                )?;
                batch.pool_finish(b, model.qsa_proj[li][4], &w.block, width)?;
            } else {
                Gdn::pool_next(
                    b,
                    &model.weights,
                    model.gdn_proj[li],
                    &w.gdn,
                    &w.mixed,
                    &w.block,
                    model.gdn_norms[li]
                        .as_ref()
                        .ok_or("missing pooled GDN norms")?
                        .device_address(),
                    step,
                )?;
            }
            source.mlp_hc[li].combine_mix_tiled(&source.attn_hc[li], b, &mut w.hc, &mut w.residual, &w.block, &w.mixed, width, None)?;
            // The MLP combine forms this layer's MoE finish (combine_finish_tiled).
            w.defer_finish = true;
            let moe = w.moe(b, model, li, width, Some(step));
            w.defer_finish = false;
            moe?;
            let finish = w.deferred_finish.take();
            source.mlp_hc[li].combine_finish_tiled(b, &mut w.hc, &mut w.residual, &w.block, width, finish)?;
        }
        // Front fragments defer position publication until the final head part. Every
        // worker publishes once after its own last layer, including middle ranks whose
        // interval ends before layer 48; otherwise its next pooled step reuses old KV pos.
        if finish == Finish::Later || !super::super::pp::pool_advances_position(layers.end, finish != Finish::Position) {
            return Ok(());
        }
        // Finish::Position: a pipeline worker, which only advances its slot positions.
        if let Finish::Head { bounded } = finish {
            source
                .mixer
                .mix_tiled(b, &mut w.hc, &w.residual, &w.mixed, width)?;
            let weight = model.weights.device_address("lm_head.weight")?;
            if bounded {
                Head::project_bound_rows(b, weight, 0, model.vocabulary(), width, &w.mixed, &mut w.head_logits,
                    &mut w.head_partials, &mut w.head_result, model.head_bounds.as_ref().ok_or("head bounds missing")?)?;
            } else {
                Head::project_next_rows(b, weight, model.vocabulary(), width, &w.mixed, &mut w.head_logits,
                    &mut w.head_partials, &mut w.head_result)?;
            }
        }
        unsafe {
            b.launch_flash(
                "flash_next_slot_position",
                1,
                128,
                &mut [(&mut step as *mut SlotStep).cast()],
            )
        }
    }
}
impl<'a, 'w> NextSession<'a, 'w> {
    /// This session's last pass was a verify pass that left sparse logits: the bounded head or
    /// the verify split head (Draft::bound_at). A later pass, a reset, a rollback and decode_top_k
    /// clear the mark. For checks of the path a pass took.
    pub fn verify_left_sparse(&self) -> bool { self.draft.as_ref().is_some_and(|d| d.bound_at.get().is_some()) }
    /// This session's last plain step left sparse logits (the bounded head or the plain split
    /// head): a full-logit reader projects the whole head again. For checks of the path a step
    /// took; a bounded verify pass keeps its own mark (Draft::bound_at).
    pub fn step_was_sparse(&self) -> bool { self.valid && self.logits_ready && self.head.split_input().is_some() }
    /// Same argmax ids, one compact cohort read instead of full vocabulary downloads. A step
    /// that took the bounded head or the plain split head holds no full logits: logits() and
    /// decode_top_k project the whole head again from the session's retained input row.
    pub fn step_batch_greedy(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let inputs = tokens.iter().map(std::slice::from_ref).collect::<Vec<_>>();
        // Only the ids are read: the tensor split may split the head by vocabulary parts, and
        // one Spark may take the bounded head (bounded_step).
        let source = Self::run_rows_inner(b, sessions, &inputs, true, false, true)?;
        let model = sessions[0].model;
        let count = sessions.len();
        let result = source.read(
            || {
                let pool = model.pool.borrow();
                let work = &pool
                    .decode
                    .get(&count.next_power_of_two())
                    .ok_or("missing pooled decode workspace")?
                    .work;
                b.read_argmax_results(&work.head_result, count)
                    .map(|v| v.into_iter().map(|v| v.0).collect())
            },
            || {
                sessions
                    .iter_mut()
                    .map(|s| {
                        s.head
                            .local_max(b)?
                            .map(|v| v.token)
                            .ok_or_else(|| "empty vocabulary".into())
                    })
                    .collect()
            },
        );
        if result.is_err() {
            for s in sessions {
                s.valid = false;
            }
        }
        result
    }
    pub(super) fn run_pooled(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        output: bool,
        ids_only: bool,
    ) -> Result<()> {
        let model = sessions[0].model;
        let fallback = model.pp_front_split().is_none() && sessions[0].pool_graph_fallback;
        let ids = crate::flash_next_rows::padded_slots(
            &sessions.iter().map(|s| s.slot).collect::<Vec<_>>(),
        )?;
        let width = ids.len();
        let (mut pool, table, capacity) = {
            let mut p = model.pool.borrow_mut();
            (
                p.decode.remove(&width),
                p.table.device_address(),
                p.capacity,
            )
        };
        if pool.is_none() {
            pool = Some(Pooled::new(b, model, capacity, width)?);
        }
        let mut pool = pool.ok_or("missing pooled decode workspace")?;
        // All allocation/slot validation precedes hash and history changes.
        let words: Vec<_> = ids.iter().map(|&id| f32::from_bits(id)).collect();
        b.write_vector(&mut pool.slots, &words)?;
        pool.work.status.clear(b, width, 0)?;
        pool.work.status.pad_rows(
            b,
            width,
            if sessions.len() == 16 {
                u16::MAX
            } else {
                (1u16 << sessions.len()) - 1
            },
        )?;
        for s in sessions.iter_mut() {
            s.valid = false;
            s.logits_ready = false;
            if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
        }
        let result = (|| {
            let started = std::time::Instant::now();
            let mut input = [[0i64; 3]; 16];
            for (row, (s, token)) in sessions.iter_mut().zip(tokens).enumerate() {
                let prev = s.hash.advance_history(token[0]);
                input[row] = [token[0] as i64, prev[0], prev[1]];
            }
            b.write_vector_bytes(&mut pool.frontend[0], as_bytes(&input[..width]))?;
            let rows: Vec<_> = input
                .iter()
                .take(sessions.len())
                .map(|&[token, h0, h1]| {
                    spark_gpu_probe::frontend_math::rows(token as u32, &mut [h0, h1], &pool.params)
                })
                .collect();
            // Every row's page reads start before layer 0 is launched.
            for r in &rows {
                model.prefill_reader.prefetch_ple_rows(r);
            }
            let step = SlotStep {
                slots: pool.slots.device_address(),
                table,
                layer: 0,
                rows: width as u32,
            };
            // Pipeline front: the tail graph stops at `split`; `fin` runs the head after
            // the back's rows return (otherwise `end` is 48).
            let split = model.pp_front_split();
            let end = split.unwrap_or(LAYERS);
            // When head_after_tail holds, the head of this step's reader runs behind the tail.
            // Every reader gets the whole head's argmax.
            let later = crate::flash_next_rows::head_after_tail(split.is_some(), width, model.head_bounds.is_some());
            let bounded = crate::flash_next_rows::bounded_step(later, output, ids_only, sessions.len(), width,
                sessions.iter().all(|s| s.head.retains_input()));
            let tail = if later { Finish::Later } else { Finish::Head { bounded: false } };
            if pool.graph[b.arm_index()].is_none() {
                if fallback {
                    let arm = b.arm_index();
                    if pool.capture_retry[arm].attempt() {
                        pool.graph[arm] = crate::flash_next_rows::capture_pair(|head| unsafe {
                            b.try_capture_batch(if head { 1 } else { end - 1 }, width, || {
                                pool.enqueue(b, model, sessions[0], step, if head { 0..1 } else { 1..end }, if head { Finish::Later } else { tail })
                            })
                        })?;
                        if pool.graph[arm].is_none() {
                            pool.capture_retry[arm].refused();
                            engine_model::events::note_line("pooled_graph_fallback", &format!("flash_next pooled_graph_fallback width={width} retry_backoff=true"));
                        } else { pool.capture_retry[arm].captured(); }
                    }
                } else {
                    let head = unsafe {
                        b.capture_batch(1, width, || pool.enqueue(b, model, sessions[0], step, 0..1, Finish::Later))?
                    };
                    let tail = unsafe {
                        b.capture_batch(end - 1, width, || {
                            pool.enqueue(b, model, sessions[0], step, 1..end, tail)
                        })?
                    };
                    pool.graph[b.arm_index()] = Some((head, tail));
                }
            }
            if split.is_some() && pool.fin[b.arm_index()].is_none() {
                let fin = unsafe {
                    b.capture_batch(1, width, || pool.enqueue(b, model, sessions[0], step, LAYERS..LAYERS, Finish::Head { bounded: false }))?
                };
                pool.fin[b.arm_index()] = Some(fin);
            }
            if let Some((head, _)) = &pool.graph[b.arm_index()] {
                unsafe { b.replay_decode(head)?; }
            } else {
                pool.enqueue(b, model, sessions[0], step, 0..1, Finish::Later)?;
            }
            {
                // Layer 0 runs while the host reads the rows; layer 1 is their first use.
                let (host, device) = &mut pool.staged;
                let packed = host.bytes_mut()?;
                packed.fill(0); // inactive padded rows have no file reads
                for (row, rows) in rows.into_iter().enumerate() {
                    for (head, id) in rows.into_iter().enumerate() {
                        let offset = (row * 16 + head) * 90;
                        model
                            .prefill_reader
                            .ple_row(id, &mut packed[offset..offset + 90])?;
                    }
                }
                // Outside capture; its stream event protects pinned-source reuse.
                b.write_prompt_bytes(device, host, 0, width * 16 * 90)?;
            }
            let (calls, time) = model.frontend_host.get();
            model
                .frontend_host
                .set((calls + 1, time + started.elapsed()));
            if let Some((_, tail)) = &pool.graph[b.arm_index()] {
                unsafe { b.replay_decode(tail)?; }
            } else {
                pool.enqueue(b, model, sessions[0], step, 1..end, tail)?;
            }
            if later {
                pool.enqueue(b, model, sessions[0], step, LAYERS..LAYERS, Finish::Head { bounded })?;
            }
            if split.is_some() {
                let meta: Vec<[u32; 4]> = sessions
                    .iter()
                    .map(|s| [s.slot, s.capacity as u32, s.pos as u32, 1])
                    .collect();
                model.pp_exchange(b, super::super::pp::KIND_POOLED, &meta, &mut pool.work.residual, width)?;
                unsafe {
                    b.replay_decode(pool.fin[b.arm_index()].as_ref().ok_or("missing pipeline head graph")?)?;
                }
            }
            // Queue session output copies while the graph is still running.
            if output {
                for (row, s) in sessions.iter_mut().enumerate() {
                    if bounded {
                        s.head.publish_bound(b, &pool.work.head_result, &pool.work.mixed, row)?;
                    } else {
                        s.head.publish_next(b, &pool.work.head_logits, &pool.work.head_result, row)?;
                    }
                }
            }
            b.synchronize()?;
            for row in 0..sessions.len() {
                if pool.work.status.row_failed_after_readback(row)? {
                    return Err("Flash-Next pooled routing rejected a row; reset cohort".into());
                }
            }
            for ((s, input), row) in sessions.iter_mut().zip(tokens).zip(0..) {
                if s.draft.is_none() {
                    continue;
                }
                // Kept rows (queued prompt tiles) precede this row in the history.
                if split.is_some() {
                    s.ingest_kept(b)?;
                }
                if s.defer_feed {
                    if s.draft.as_ref().is_some_and(|d| d.backlog_len() >= super::mtp::DEFER_ROWS) {
                        s.flush_draft_feed(b)?;
                    }
                    let pos = s.pos;
                    s.draft.as_mut().unwrap().defer(b, pos, input[0], &pool.work.residual, row)?;
                } else {
                    s.flush_draft_feed(b)?;
                    let pos = s.pos;
                    s.draft.as_mut().unwrap().observe(b, model, pos, input, &pool.work.residual, row)?;
                }
            }
            for s in sessions.iter_mut() {
                for state in s.gdn_state.iter_mut().flatten() {
                    state.commit_next();
                }
                for state in s.qsa_state.iter_mut().flatten() {
                    state.commit_next();
                }
                s.pos += 1;
            }
            Ok(())
        })();
        model.pool.borrow_mut().decode.insert(width, pool);
        if result.is_ok() {
            for s in sessions {
                s.valid = true;
                s.logits_ready = output;
                if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
            }
        }
        result
    }
}
impl<'a, 'w> NextSession<'a, 'w> {
    /// Pipeline back: one pooled decode step over layers [split, 48) for the front's
    /// cohort (same padded width; this rank's own slots), residual in and out.
    pub(in super::super) fn pp_back_pooled(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        width: usize,
        residual: &mut [f32],
    ) -> Result<()> {
        let model = sessions[0].model;
        let split = model.pp_split.get();
        let ids = crate::flash_next_rows::padded_slots(
            &sessions.iter().map(|s| s.slot).collect::<Vec<_>>(),
        )?;
        if ids.len() != width || residual.len() != width * W || !(2..LAYERS).contains(&split) {
            return Err("pipeline pooled width mismatch".into());
        }
        // The front's bound (pos + 1 <= capacity per chat), checked here before any state
        // changes: the back's KV writes are sized by its own sessions, not the front's.
        let one = [0u32];
        crate::flash_next_rows::validate(
            &sessions.iter().map(|s| s.pos).collect::<Vec<_>>(),
            &vec![&one[..]; sessions.len()],
            &sessions.iter().map(|s| s.capacity).collect::<Vec<_>>(),
            model.checkpoint.config.vocab,
        )?;
        let (mut pool, table, capacity) = {
            let mut p = model.pool.borrow_mut();
            (p.decode.remove(&width), p.table.device_address(), p.capacity)
        };
        if pool.is_none() {
            pool = Some(Pooled::new(b, model, capacity, width)?);
        }
        let mut pool = pool.ok_or("missing pooled decode workspace")?;
        let words: Vec<_> = ids.iter().map(|&id| f32::from_bits(id)).collect();
        let prepared = b
            .write_vector(&mut pool.slots, &words)
            .and_then(|()| pool.work.status.clear(b, width, 0))
            .and_then(|()| {
                pool.work.status.pad_rows(
                    b,
                    width,
                    if sessions.len() == 16 { u16::MAX } else { (1u16 << sessions.len()) - 1 },
                )
            });
        if let Err(error) = prepared {
            model.pool.borrow_mut().decode.insert(width, pool);
            return Err(error);
        }
        for s in sessions.iter_mut() {
            s.valid = false;
            s.logits_ready = false;
            if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
        }
        let result = (|| {
            b.write_vector_range(&mut pool.work.residual, 0, residual)?;
            let step = SlotStep {
                slots: pool.slots.device_address(),
                table,
                layer: 0,
                rows: width as u32,
            };
            if pool.back[b.arm_index()].is_none() {
                let graph = unsafe {
                    b.capture_batch(model.pp_end.get() - split, width, || {
                        pool.enqueue(b, model, sessions[0], step, split..model.pp_end.get(), Finish::Position)
                    })?
                };
                pool.back[b.arm_index()] = Some(graph);
            }
            unsafe {
                b.replay_decode(pool.back[b.arm_index()].as_ref().ok_or("missing pipeline back graph")?)?;
            }
            b.synchronize()?;
            for row in 0..sessions.len() {
                if pool.work.status.row_failed_after_readback(row)? {
                    return Err("Flash-Next pooled routing rejected a row; reset cohort".into());
                }
            }
            for s in sessions.iter_mut() {
                for state in s.gdn_state.iter_mut().flatten() {
                    state.commit_next();
                }
                for state in s.qsa_state.iter_mut().flatten() {
                    state.commit_next();
                }
                s.pos += 1;
            }
            b.read_vector_range_into(&pool.work.residual, 0, residual)
        })();
        model.pool.borrow_mut().decode.insert(width, pool);
        if result.is_ok() {
            for s in sessions {
                s.valid = true;
            }
        }
        result
    }
}
fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
