//! One layer traversal for a prompt tile or independent chat rows.
use super::*;
use spark_gpu_probe::driver::Fp8RoutingFlag;
#[path = "next_prompt_route.rs"]
mod prompt_route;

pub(super) struct Workspace<'a> {
    /// VADE_MTP_GRAPH: per verify shape (rows, chats, chats whose GDN window stays inside
    /// its chunk) one retained executable per segment (layers 1-12, 13-24, 25-36, 37-47),
    /// oldest first; shapes seen once (eager); counters (graphed passes, in-place updates,
    /// instantiations, eager passes).
    verify_graphs: Vec<(VerifyShape, [Option<spark_gpu_probe::driver::DecodeGraph<'a>>; 4])>,
    verify_seen: Vec<VerifyShape>,
    verify_counts: [usize; 4],
    prompt: (Vec<f32>, Vec<f32>),
    prompt_packed: Option<DeviceVector<'a>>,
    prompt_route_packed: Option<DeviceVector<'a>>,
    mixed16_ready: bool,
    pub(super) qsa_batch: Option<super::super::qsa::QsaBatch<'a>>,
    pub(super) rows: usize,
    pub(super) capacity: usize,
    pub(super) hc: HcScratch<'a>,
    gdn: GdnScratch<'a>,
    qsa: QsaScratch<'a>,
    pub(super) residual: DeviceVector<'a>,
    pub(super) mixed: DeviceVector<'a>,
    pub(super) block: DeviceVector<'a>,
    gate: DeviceVector<'a>,
    up: DeviceVector<'a>,
    expert_out: DeviceVector<'a>,
    router: DeviceVector<'a>,
    head_input: DeviceVector<'a>,
    pub(super) head_logits: DeviceVector<'a>,
    pub(super) head_partials: DeviceVector<'a>,
    pub(super) head_result: DeviceVector<'a>,
    /// Only the private greedy verification API permits sparse head logits.
    pub(super) bound_head: bool,
    shared_gate: DeviceVector<'a>,
    lexical: DeviceVector<'a>,
    ids: DeviceVector<'a>,
    weights: DeviceVector<'a>,
    assignments: DeviceVector<'a>,
    recurrence: DeviceVector<'a>,
    /// flash_hybrid_copy rows of a cohort verify layer (gdn::cohort_copies): 4 per chat.
    copy_rows: DeviceVector<'a>,
    /// Parameter tables of a cohort verify layer's GDN launches (gdn::cohort_kernels).
    gdn_launches: DeviceVector<'a>,
    compact_mid: DeviceVector<'a>,
    /// BF16 copy of `mixed` for the prompt expert gate (fp4_prefill_tc_a16).
    pub(super) mixed16: DeviceVector<'a>,
    bounds: DeviceVector<'a>,
    pub(super) status: Fp8RoutingFlag<'a>,
    /// Prompt layers: the MoE output (k16_finish) is formed inside the next attention mix's
    /// fused combine instead (moe_at leaves its parameters in `deferred_finish`).
    defer_finish: bool,
    deferred_finish: Option<spark_gpu_probe::fp8_abi::MoeFinishParams>,
}
impl<'a> Workspace<'a> {
    pub(super) fn replay_gdn(
        &mut self,
        b: &Backend<'a>,
        model: &NextModel<'a>,
        states: &mut [Option<GdnState<'a>>],
        rollback: &mtp::Rollback<'a>,
        rows: usize,
        base: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        for (li, state) in states.iter_mut().enumerate() {
            if li % 4 == 3 || !layers.contains(&li) {
                continue;
            }
            let owner = state.take().ok_or("MTP live state missing")?;
            let mut g = Gdn::load_next(b, &model.weights, li, model.gdn_proj[li], Some(owner))?;
            let tape = rollback.inputs[li].as_ref().ok_or("MTP projection tape missing")?;
            let result = if rollback.shallow {
                g.replay_history(b, tape, rows)
            } else {
                g.replay_next(b, &mut self.gdn, tape, rows, &mut self.recurrence, base)
            };
            *state = Some(g.into_state());
            result?;
        }
        Ok(())
    }
    pub(super) fn reservation_bytes(capacity: usize, rows: usize, mtp_depth: usize, frontend: bool) -> usize {
        // Match new's head row bound; retain the old 80-row floor for default admission.
        let head_rows = rows.min(80.max(16 * (mtp_depth + 1)));
        let own: usize = [
            rows * W,
            rows * H,
            rows * H,
            rows * 640,
            rows * 640,
            rows * 11 * H,
            rows * EXPERTS,
            rows,
            rows * H,
            rows * 10,
            rows * 10,
            rows * 30,
            // GDN descriptor words: one chat (6) or a cohort verify table (6 per chat).
            6 * rows.min(COHORT_TABLE_CHATS),
            head_rows * H,
            head_rows * 248320,
            head_rows * 192 * 3,
            head_rows * 3,
            rows * 10 * 320,
            rows * H / 2,
            spark_gpu_probe::compact::words(rows),
            gdn::COPY_ROWS_PER_CHAT * gdn::COPY_ROW_WORDS * rows.min(COHORT_TABLE_CHATS),
            gdn::COHORT_LAUNCH_BYTES / 4 * rows.min(COHORT_TABLE_CHATS),
        ]
        .iter()
        .map(|n| (n * 4).div_ceil(64) * 64)
        .sum();
        // World-one plan: 48 value heads and 16 key heads.
        let plan = RankPlan::new(0, 1).expect("world-one plan");
        own + crate::flash_prompt_options::cached().packed_workspace_bytes(rows, frontend)
            + crate::flash_prompt_options::cached().route_workspace_bytes(rows)
            + crate::flash_next_workspace::bytes(&crate::flash_next_workspace::hc_lengths(W, 320, H, rows))
            + crate::flash_next_workspace::gdn_owned_lengths(spark_gpu_probe::next_math::gdn_lengths(H, 48, 16, rows))
                .iter()
                .map(|n| (n * 4).div_ceil(64) * 64)
                .sum::<usize>()
            + QsaScratch::reservation_bytes(&plan, rows, capacity, false)
            + Ple::DEVICE_BYTES
            + Head::top_k_reservation_bytes(248320)
            + (248320 + 55296) * 4
            + rows * 4
            + 64
    }
    pub(super) fn new(
        b: &Backend<'a>,
        model: &NextModel<'a>,
        capacity: usize,
        rows: usize,
    ) -> Result<Self> {
        if !(1..=crate::flash_next_rows::prompt_tile()).contains(&rows) {
            return Err("Flash-Next workspace rows exceed the prompt tile".into());
        }
        let mut v = b
            .vector_arena(&[
                rows * W,
                rows * H,
                rows * H,
                rows * 640,
                rows * 640,
                rows * 11 * H,
                rows * EXPERTS,
                rows,
                rows * H,
                rows * 10,
                rows * 10,
                rows * 30,
                6 * rows.min(COHORT_TABLE_CHATS),
                rows.min(16 * (model.mtp_depth + 1)) * H,
                rows.min(16 * (model.mtp_depth + 1)) * 248320,
                rows.min(16 * (model.mtp_depth + 1)) * 192 * 3,
                rows.min(16 * (model.mtp_depth + 1)) * 3,
                rows * 10 * 320,
                rows * H / 2,
                spark_gpu_probe::compact::words(rows),
                gdn::COPY_ROWS_PER_CHAT * gdn::COPY_ROW_WORDS * rows.min(COHORT_TABLE_CHATS),
                gdn::COHORT_LAUNCH_BYTES / 4 * rows.min(COHORT_TABLE_CHATS),
            ])?
            .into_iter();
        let status = Fp8RoutingFlag::new(b, rows)?;
        let prompt = (vec![0.0; rows * W], vec![0.0; rows * H]);
        let prompt_packed = if crate::flash_next_prefill::gpu_prompt() && model.prompt_frontend {Some(b.vector(rows*377)?)}else{None};
        let prompt_route_packed = {let bytes=crate::flash_prompt_options::cached().route_workspace_bytes(rows);if bytes>0 {Some(b.vector(bytes/4)?)}else{None}};
        let hc = HcScratch::next_rows(b, rows)?;
        let gdn = Gdn::next_scratch(b, &model.weights, rows, hc.normalized_workspace())?;
        let mut result = Self {
            verify_graphs: Vec::new(),
            verify_seen: Vec::new(),
            verify_counts: [0; 4],
            prompt,
            prompt_packed, prompt_route_packed,
            mixed16_ready:false,
            qsa_batch: None,
            status,
            defer_finish: false,
            deferred_finish: None,
            rows,
            capacity,
            hc, gdn,
            qsa: QsaScratch::new(b, &model.weights.plan, rows, capacity.max(rows))?,
            residual: v.next().unwrap(),
            mixed: v.next().unwrap(),
            block: v.next().unwrap(),
            gate: v.next().unwrap(),
            up: v.next().unwrap(),
            expert_out: v.next().unwrap(),
            router: v.next().unwrap(),
            shared_gate: v.next().unwrap(),
            lexical: v.next().unwrap(),
            ids: v.next().unwrap(),
            weights: v.next().unwrap(),
            assignments: v.next().unwrap(),
            recurrence: v.next().unwrap(),
            head_input: v.next().unwrap(),
            head_logits: v.next().unwrap(),
            head_partials: v.next().unwrap(),
            head_result: v.next().unwrap(),
            bound_head: false,
            compact_mid: v.next().unwrap(),
            // BF16 gate A rows (rows * H / 2 words; the retired compact_down
            // slot was rows * 10 * H).
            mixed16: v.next().unwrap(),
            bounds: v.next().unwrap(),
            copy_rows: v.next().unwrap(),
            gdn_launches: v.next().unwrap(),
        };
        // One chat's descriptor by default; a cohort verify pass widens it for its table.
        result.recurrence.resize_within_capacity(6)?;
        result.qsa_batch = Some(super::super::qsa::QsaBatch::workspace(
            &model.weights.plan,
            rows,
            &result.expert_out,
        )?);
        // Fund both batched uploads plus layer descriptors before state mutation.
        b.reserve_exchange(rows * (W + H) + (1 << 18))?;
        Ok(result)
    }
    fn moe(
        &mut self,
        b: &Backend<'a>,
        model: &NextModel<'a>,
        li: usize,
        count: usize,
        mask: Option<spark_gpu_probe::next_math::SlotStep>,
    ) -> Result<()> {
        self.moe_at(b, &model.moe[li], count, mask, true, None)
    }
    pub(super) fn draft_moe(
        &mut self,
        b: &Backend<'a>,
        addr: &MoeAddr<'a>,
        input: &DeviceVector<'a>,
        output: &mut DeviceVector<'a>,
    ) -> Result<()> {
        self.status.clear(b, 1, 0)?;
        b.copy_range(input, &mut self.mixed, 0, 0, H)?;
        self.moe_at(b, addr, 1, None, false, None)?;
        // No host sync here: the caller reads the routing status after the step's argmax
        // readback (draft_routing_failed), as the cohort draft path does.
        b.copy_range(&self.block, output, 0, 0, H)
    }
    /// The one-row draft MoE's routing verdict; only after a readback that follows it.
    pub(super) fn draft_routing_failed(&self) -> Result<bool> {
        self.status.failed_after_readback()
    }
    /// Cohort draft MoE (VADE_MTP_BATCH): `count` drafting chats' rows from `mixed`
    /// into `block` on the decode kernels; the caller clears and reads the status.
    pub(super) fn draft_moe_rows(&mut self, b: &Backend<'a>, addr: &MoeAddr<'a>, count: usize) -> Result<()> {
        self.moe_at(b, addr, count, None, false, None)
    }
    fn moe_at(
        &mut self,
        b: &Backend<'a>,
        addr: &MoeAddr<'a>,
        count: usize,
        mask: Option<spark_gpu_probe::next_math::SlotStep>,
        prefill: bool, tp: Option<super::tp::TpMoe<'_, 'a>>,
    ) -> Result<()> {
        use gdn::Arg::{Ptr, U32};
        use spark_gpu_probe::{fp8_abi::MoeFinishParams, kernel_params::FlashMoeRoute};
        let mixed16_ready=std::mem::take(&mut self.mixed16_ready);
        let dense = count >= 64;
        // Decode (M<=16) also runs router + shared gate/up in one group launch: nothing
        // between here and the shared expert touches shared_gate/gate/up.
        let fused = dense || count <= blocks::DECODE_ROWS;
        let wide = count > 16 && mask.is_none() && prefill;
        if fused {
            blocks::next_group(
                b,
                self.mixed.device_address(),
                H,
                count,
                [
                    (addr.router, self.router.device_address(), EXPERTS),
                    (addr.shared_gate, self.shared_gate.device_address(), 1),
                    (addr.shared[0], self.gate.device_address(), 640),
                    (addr.shared[1], self.up.device_address(), 640),
                ],
            )?;
        } else {
            blocks::next_rows(
                b,
                addr.router,
                self.mixed.device_address(),
                self.router.device_address(),
                EXPERTS,
                H,
                count,
            )?;
        }
        let mut route = FlashMoeRoute {
            logits: self.router.device_address(),
            ids: self.ids.device_address(),
            weights: self.weights.device_address(),
            tokens: count as u32,
            norm_topk: 1,
        };
        // Pooled decode: the route launch's _mask entry applies flash_next_route_mask's
        // stores in the route's CTA (below) instead of a launch of its own.
        let mask_route = !dense && fused && mask.is_some() && b.has_flash("flash_next_compact_route_silu_mask");
        // One decode token: flash_next_route_compact1_silu runs the route in the compact
        // route's CTA (below); its other CTAs apply the shared expert's silu.
        let one_route = count == 1 && (mask.is_none() || mask_route) && fused && b.has_flash("flash_next_route_compact1_silu");
        if !one_route {
            unsafe {
                b.launch_flash(
                    if b.has_flash("flash_moe_route_fast") { "flash_moe_route_fast" } else { "flash_moe_route" },
                    count as u32,
                    128,
                    &mut [(&mut route as *mut FlashMoeRoute).cast()],
                )?;
            }
        }
        if let Some(mut step) = mask.filter(|_| !mask_route) {
            let (mut ids, mut weights) = (self.ids.device_address(), self.weights.device_address());
            unsafe {
                b.launch_flash(
                    "flash_next_route_mask",
                    (count * 10).div_ceil(128) as u32,
                    128,
                    &mut [
                        (&mut ids as *mut u64).cast(),
                        (&mut weights as *mut u64).cast(),
                        (&mut step as *mut spark_gpu_probe::next_math::SlotStep).cast(),
                    ],
                )?;
            }
        }
        // Decode: flash_next_compact_route_ids writes these words in the route's CTA.
        let route_ids = one_route || (!dense && b.has_flash("flash_next_compact_route_ids"));
        // Decode with the shared gate/up already projected (`fused`): the route launch's
        // CTAs >= 1 also apply the shared expert's silu (flash_next_compact_route_silu).
        let silu_route = one_route || (route_ids && fused && b.has_flash("flash_next_compact_route_silu"));
        // The step the _mask entries read (mask_route implies silu_route: !dense and fused).
        let mut mask_step = mask.filter(|_| mask_route);
        // Decode (<= 16 rows): the route launch's CTAs >= 1 also write the shared down's dense
        // hi/lo planes of the silu'd gate (one tile row: 20 CTAs), offered to its pack below:
        // pooled rows (_pack_mask entries) and verify / draft rows (flash_next_compact_route_pack).
        let pack_route = silu_route
            && !dense
            && count <= 16
            && if mask_route {
                b.has_flash("flash_next_compact_route_pack_mask")
            } else {
                mask.is_none() && !one_route && b.has_flash("flash_next_compact_route_pack")
            };
        let mut dense_planes = if pack_route { b.flash_dense_input()? } else { 0 };
        if !route_ids {
            gdn::launch(
                b,
                "flash_moe_assignments",
                count * 10,
                &mut [
                    Ptr(self.ids.device_address()),
                    Ptr(self.weights.device_address()),
                    Ptr(self.assignments.device_address()),
                    U32(count as u32),
                ],
            )?;
        }
        // Compact expert bounds and inverse pairs. GU reads original token
        // rows directly; only actual BF16 middle/down rows occupy memory.
        use spark_gpu_probe::fp8_abi::{GroupedParams, GroupedRouteParams};
        let pairs = count * 10;
        let tasks = spark_gpu_probe::compact::tasks(count);
        let base = self.bounds.device_address();
        let mut grouped = GroupedRouteParams {
            error: self.status.device_address(),
            pairs: self.assignments.device_address(),
            counts: base,
            starts: base + 512 * 4,
            pair_to_row: base + (512 + 513) * 4,
            sorted_row: base + ((512 + 513 + pairs) * 4) as u64,
            token_pairs: base + ((512 + 513 + 2 * pairs) * 4) as u64,
            tasks: base + ((512 + 513 + 3 * pairs) * 4) as u64,
            token_count: 0,
            output_row: 0,
            experts: 512,
            tokens: count as u32,
            pair_count: pairs as u32,
            row_capacity: pairs as u32,
        };
        let mut params = GroupedParams {
            route: grouped,
            weights: addr.experts.device_address() + tp.map_or(0, |t| t.table()),
            input: self.mixed.device_address(),
            input_scales: 0,
            output: self.compact_mid.device_address() + tp.map_or(0, |t| t.mid()),
            cols: H as u32,
            activation_quantization: 0,
            output_shard: 0,
        };
        let planes = if wide {
            1
        } else {
            count.div_ceil(16).min(8) as u32
        };
        unsafe {
            if dense {
                prompt_route::launch(b, grouped, count,
                    self.prompt_route_packed.as_ref().filter(|_| prefill && crate::flash_prompt_options::Options::kernel_rows(count)))?;
            } else if one_route {
                let (mut gate, mut up, mut n) = (self.gate.device_address(), self.up.device_address(), 640u32);
                let mut args: Vec<*mut core::ffi::c_void> = vec![
                    (&mut route as *mut FlashMoeRoute).cast(),
                    (&mut grouped as *mut GroupedRouteParams).cast(),
                    (&mut gate as *mut u64).cast(),
                    (&mut up as *mut u64).cast(),
                    (&mut n as *mut u32).cast(),
                ];
                if let Some(step) = mask_step.as_mut() {
                    args.push((step as *mut spark_gpu_probe::next_math::SlotStep).cast());
                }
                if pack_route {
                    args.push((&mut dense_planes as *mut u64).cast());
                }
                b.launch_flash(
                    if pack_route {
                        "flash_next_route_compact1_pack_mask"
                    } else if mask_route {
                        "flash_next_route_compact1_silu_mask"
                    } else {
                        "flash_next_route_compact1_silu"
                    },
                    1 + if pack_route { 20 } else { 640u32.div_ceil(256) },
                    256,
                    &mut args,
                )?;
            } else if silu_route {
                let (mut ids, mut weights) = (self.ids.device_address(), self.weights.device_address());
                let (mut gate, mut up, mut n) = (self.gate.device_address(), self.up.device_address(), (count * 640) as u32);
                let mut args: Vec<*mut core::ffi::c_void> = vec![
                    (&mut grouped as *mut GroupedRouteParams).cast(),
                    (&mut ids as *mut u64).cast(),
                    (&mut weights as *mut u64).cast(),
                    (&mut gate as *mut u64).cast(),
                    (&mut up as *mut u64).cast(),
                    (&mut n as *mut u32).cast(),
                ];
                if let Some(step) = mask_step.as_mut() {
                    args.push((step as *mut spark_gpu_probe::next_math::SlotStep).cast());
                }
                if pack_route {
                    args.push((&mut dense_planes as *mut u64).cast());
                }
                b.launch_flash(
                    if pack_route && mask_route {
                        "flash_next_compact_route_pack_mask"
                    } else if pack_route {
                        "flash_next_compact_route_pack"
                    } else if mask_route {
                        "flash_next_compact_route_silu_mask"
                    } else {
                        "flash_next_compact_route_silu"
                    },
                    1 + if pack_route { 20 } else { (count * 640).div_ceil(256) as u32 },
                    256,
                    &mut args,
                )?;
            } else if route_ids {
                let (mut ids, mut weights) = (self.ids.device_address(), self.weights.device_address());
                b.launch_flash(
                    "flash_next_compact_route_ids",
                    1,
                    256,
                    &mut [
                        (&mut grouped as *mut GroupedRouteParams).cast(),
                        (&mut ids as *mut u64).cast(),
                        (&mut weights as *mut u64).cast(),
                    ],
                )?;
            } else {
                b.launch_flash(
                    "flash_next_compact_route",
                    1,
                    256,
                    &mut [(&mut grouped as *mut GroupedRouteParams).cast()],
                )?;
            }
            if wide {
                // The prompt gate reads its A rows as RN BF16 pairs from mixed16.
                let groups = count * H / 8;
                if !mixed16_ready {gdn::launch_threads(
                    b,
                    "fp4_prefill_tc_a16",
                    groups,
                    256,
                    &mut [
                        Ptr(self.mixed.device_address()),
                        Ptr(self.mixed16.device_address()),
                        U32(groups as u32),
                    ],
                )?;}
            }
            if wide {
                params.input = self.mixed16.device_address();
            }
            b.launch_flash_grid(
                if wide {
                    "fp4_prefill_tc_gate"
                } else if count == 1 && bw_entries(b).1 {
                    // One token: same bytes, the one-wave register budget (fp4_experts.rs).
                    "fp4_bw_gate_c1"
                } else {
                    "fp4_bw_gate"
                },
                // Prompt gate: only x < width/128 (<= 5) are live N128 tiles (was 20).
                if wide { 5 } else { tp.map_or(40, |t| t.gate_ctas(40)) },
                tasks as u32,
                planes,
                if wide { 256 } else { 64 },
                if wide { spark_gpu_probe::kernel_params::FP4_PREFILL_GATE_SHARED_BYTES } else { 0 },
                &mut [(&mut params as *mut GroupedParams).cast()],
            )?;
            if let Some(t) = tp { t.swap_mid(b, &mut self.compact_mid, count)?; } params.input = self.compact_mid.device_address();
            params.output = self.expert_out.device_address() + tp.map_or(0, |t| t.routed());
            // One token: down may start while the gate runs; fp4_bw_down_pdl waits
            // (griddepcontrol) before reading compact_mid, so bytes are unchanged.
            // Off by default: VADE_BW_PDL=1 turns it on. At 8 rows the overlap
            // costs time (DRAM-bound), so it is never used there.
            let pdl = !wide && count == 1 && bw_entries(b).0;
            if pdl {
                b.programmatic_next();
            }
            b.launch_flash_grid(
                if wide {
                    "fp4_prefill_tc_down"
                } else if pdl {
                    "fp4_bw_down_pdl"
                } else {
                    "fp4_bw_down"
                },
                // Prompt down: x < 4 live (two M64 planes x two first N128 tiles: was 80).
                if wide { 4 } else { tp.map_or(160, |t| t.down_ctas(160)) },
                tasks as u32,
                planes,
                if wide { 256 } else { 64 },
                0,
                &mut [(&mut params as *mut GroupedParams).cast()],
            )?;
        }
        let x = self.mixed.device_address();
        if !fused {
            for (w, y, rows, cols) in [
                (addr.shared_gate, self.shared_gate.device_address(), 1, H),
                (addr.shared[0], self.gate.device_address(), 640, H),
                (addr.shared[1], self.up.device_address(), 640, H),
            ] {
                blocks::next_rows(b, w, x, y, rows, cols, count)?;
            }
        }
        if !silu_route {
            gdn::launch(
                b,
                if b.has_flash("flash_moe_silu_fast") { "flash_moe_silu_fast" } else { "flash_moe_silu" },
                count * 640,
                &mut [
                    Ptr(self.gate.device_address()),
                    Ptr(self.up.device_address()),
                    U32((count * 640) as u32),
                ],
            )?;
        }
        if pack_route {
            b.offer_dense_planes(self.gate.device_address(), 640, count, dense_planes);
        }
        // Shared is the eleventh row in each token's partials, after experts.
        blocks::next_rows_strided(
            b,
            addr.shared[2],
            self.gate.device_address(),
            self.expert_out.device_address() + (10 * H * 4) as u64 - tp.map_or(0, |t| t.routed()),
            H,
            640,
            count,
            11 * H,
        )?;
        let mut finish = MoeFinishParams {
            error: self.status.device_address(),
            ids: self.ids.device_address(),
            weights: self.weights.device_address(),
            partials: self.expert_out.device_address() + tp.map_or(0, |t| t.routed()),
            shared_gate: self.shared_gate.device_address(),
            output: self.block.device_address() + tp.map_or(0, |t| t.out()),
            activation_quantization: 1,
            column_start: 0,
            columns: H as u32,
        };
        if self.defer_finish && tp.is_none() {
            self.deferred_finish = Some(finish);
            return Ok(());
        }
        unsafe {
            b.launch_flash_rows(
                "flash_next_k16_finish",
                tp.map_or(H.div_ceil(256) as u32, |t| t.finish_ctas()),
                count as u32,
                256,
                0,
                &mut [(&mut finish as *mut MoeFinishParams).cast()],
            )?;
        } tp.map_or(Ok(()), |t| t.swap_out(b, &mut self.block, count))
    }
    fn dump_routes(&self, b: &Backend<'_>, count: usize) -> Result<Vec<Vec<(usize, f32)>>> {
        let ids = b.read_vector_range(&self.ids, 0, count * 10)?;
        let weights = b.read_vector_range(&self.weights, 0, count * 10)?;
        Ok(ids
            .chunks_exact(10)
            .zip(weights.chunks_exact(10))
            .map(|(ids, weights)| {
                let mut pairs: Vec<_> = ids
                    .iter()
                    .zip(weights)
                    .map(|(id, &w)| (id.to_bits() as usize, w))
                    .collect();
                pairs.sort_by_key(|v| v.0);
                pairs
            })
            .collect())
    }
}

impl<'a> super::NextModel<'a> {
    /// The model's one prompt / verify / replay workspace, at least `rows` rows for a
    /// session of `capacity` (allocated at the full prompt tile); give it back with
    /// return_work. Scratch is expendable: finish outstanding users before replacing it.
    pub(super) fn take_work(&self, b: &Backend<'a>, capacity: usize, rows: usize) -> Result<Workspace<'a>> {
        let held = self.work.borrow_mut().take();
        match held {
            Some(w) if w.rows >= rows && w.capacity >= capacity => Ok(w),
            held => {
                b.synchronize()?;
                drop(held);
                Workspace::new(b, self, capacity, crate::flash_next_rows::prompt_tile().max(rows))
            }
        }
    }
    pub(super) fn return_work(&self, work: Workspace<'a>) {
        *self.work.borrow_mut() = Some(work);
    }
    /// Allocate the prompt workspace for sessions of `capacity` now (it is funded in
    /// shared_reservation_bytes), so the first prompt does not pay for it in its TTFT.
    pub fn prepare_prompt_workspace(&self, b: &Backend<'a>, capacity: usize) -> Result<()> {
        let work = self.take_work(b, capacity, crate::flash_next_rows::prompt_tile())?;
        self.return_work(work);
        Ok(())
    }
}
impl<'a, 'w> NextSession<'a, 'w> {
    /// VADE_MTP_DEFER_FEED: deferred drafter rows enter the draft history (observe_rows
    /// tiles, position order) through the prompt workspace's scratch.
    pub(super) fn flush_draft_feed(&mut self, b: &Backend<'a>) -> Result<()> {
        if self.draft.as_ref().is_none_or(|d| d.backlog_len() == 0) {
            return Ok(());
        }
        let model = self.model;
        let mut work = model.take_work(b, self.capacity, crate::flash_next_rows::prompt_tile())?;
        let result = Self::flush_into(b, model, self.draft.as_mut().unwrap(), &mut work);
        model.return_work(work);
        result
    }
    pub(super) fn flush_into(b: &Backend<'a>, model: &NextModel<'a>, d: &mut mtp::Draft<'a, 'w>, work: &mut Workspace<'a>)
        -> Result<()> {
        let Some((entry, tokens, hidden)) = d.take_backlog() else { return Ok(()) };
        let result = d.observe_rows(b, model, &work.mixed16, &mut work.hc, &mut work.qsa, entry, &tokens, &hidden, 0);
        d.restore_backlog(hidden);
        result
    }
    pub(super) fn run_rows(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        output: bool,
    ) -> Result<crate::flash_next_rows::ResultSource> {
        Self::run_rows_inner(b, sessions, tokens, output, false, false)
    }
    pub(super) fn run_rows_inner(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        output: bool,
        verify: bool,
        greedy_head: bool,
    ) -> Result<crate::flash_next_rows::ResultSource> {
        let spans = crate::flash_next_rows::validate(
            &sessions.iter().map(|s| s.pos).collect::<Vec<_>>(),
            tokens,
            &sessions.iter().map(|s| s.capacity).collect::<Vec<_>>(),
            sessions
                .first()
                .map_or(0, |s| s.model.checkpoint.config.vocab),
        )?;
        let model = sessions[0].model;
        if sessions
            .iter()
            .any(|s| !s.valid || !std::ptr::eq(s.model, model))
        {
            return Err("Flash-Next cohort needs valid sessions of the same model".into());
        }
        for s in sessions.iter() {
            crate::flash_next_rows::check_owners(&s.gdn_state, &s.qsa_state, s.model.state_range())?;
        }
        let count = spans.last().unwrap().end;
        let source = crate::flash_next_rows::ResultSource::select_for_policy(
            !verify && tokens.iter().all(|t| t.len() == 1),
            // Serving chooses one path for the whole scheduled cohort and installs
            // that same policy on every session. Read it once, without an OFF scan.
            sessions[0].allow_pooled_decode && !model.tp_on(),
            || {
                sessions.iter().all(|s| {
                    s.dump.is_none()
                        && s.layer_bytes.is_none()
                        && (3..LAYERS).step_by(4).all(|li| {
                            s.indexer.select(li, s.pos, s.pos + 1) == IndexSelection::Sparse
                        })
                })
            },
        );
        if source == crate::flash_next_rows::ResultSource::Pooled {
            Self::run_pooled(b, sessions, tokens, output)?;
            return Ok(source);
        }
        if count > crate::flash_next_rows::prompt_tile() {
            return Err("Flash-Next execution tile exceeds the prompt tile".into());
        }
        // VADE_W2_PP=1 front: layers [split, 48) run on the back rank (MTP verify with VADE_W2_PP_MTP=1).
        let part = match model.pp_front_split().or(model.tp_part(b, sessions, &spans, verify)?) {
            None => Part::Whole,
            Some(_) if verify && !model.pp_mtp() => return Err("VADE_W2_PP serves plain decoding only (VADE_W2_PP_MTP=0)".into()),
            Some(split) => Part::Front(split),
        };
        // All admission and workspace allocation failures precede state mutation.
        let owner = crate::flash_next_rows::workspace_owner(sessions.iter().map(|s| s.capacity))?;
        let capacity = sessions[owner].capacity;
        let mut work = model.take_work(b, capacity, count)?;
        work.bound_head = greedy_head && verify && sessions.len() == 1 && (2..=4).contains(&count)
            && model.head_bounds.is_some()
            && sessions.iter().all(|s| s.dump.is_none() && s.layer_bytes.is_none());
        if let Err(error) = work.status.clear(b, count, 0) {
            model.return_work(work);
            return Err(error);
        }
        for s in sessions.iter_mut() {
            s.valid = false;
            s.logits_ready = false;
            if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
        }
        // VADE_MTP_GRAPH=0: the eager build of `execute`, which holds no segment code.
        // VADE_W2_PP stages capture their own layers' segments; each closes before the exchange.
        let result = if verify && mtp_graph() {
            Self::execute::<true>(b, &mut work, sessions, tokens, &spans, output, verify, part).inspect_err(|_| b.abort_segment())
        } else {
            Self::execute::<false>(b, &mut work, sessions, tokens, &spans, output, verify, part)
        };
        let result = result
            .and_then(|()| {
                // One completion boundary, including advance-only prompt tiles.
                // Sticky mapped status covers all 48 layers without a MoE readback.
                b.synchronize()?;
                if work.status.failed_after_readback()? || model.tp_failed() {
                    Err(if model.tp_failed() { "tensor split mailbox timed out or its proxy failed" } else { "Flash-Next device routing rejected a row; reset cohort" }.into())
                } else {
                    Ok(())
                }
            });
        model.return_work(work);
        if result.is_ok() {
            for (s, t) in sessions.iter_mut().zip(tokens) {
                s.pos += t.len();
                if verify {
                    s.sync_pool_state(b)?;
                } else {
                    s.sync_pool_position(b)?;
                }
                s.valid = true;
                s.logits_ready = output;
            }
        }
        result.map(|()| source)
    }
    fn check_ops(
        b: &Backend<'_>,
        sessions: &[&mut Self],
        spans: &[std::ops::Range<usize>],
        stage: &str,
        v: &DeviceVector<'_>,
        stride: usize,
    ) -> Result<()> {
        use std::io::Write;
        for (s, span) in sessions.iter().zip(spans) {
            if let Some(path) = &s.layer_bytes {
                let bytes: Vec<_> = b
                    .read_vector_range(v, span.start * stride, span.len() * stride)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path.join(format!("{stage}.f32")))
                    .map_err(err)?
                    .write_all(&bytes)
                    .map_err(err)?;
            }
        }
        Ok(())
    }
    /// GRAPH: built with the VADE_MTP_GRAPH segment captures; `execute::<false>` is the eager pass.
    #[allow(clippy::too_many_arguments)]
    fn execute<const GRAPH: bool>(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        spans: &[std::ops::Range<usize>],
        output: bool,
        verify: bool,
        part: Part<'_>,
    ) -> Result<()> {
        let model = sessions[0].model;
        let count = spans.last().unwrap().end;
        // Layers this rank runs: all of them (world 1), [0, split) on the VADE_W2_PP front
        // (the back's rows come back before the tail), [split, 48) on the back (no tail).
        let (first, end) = match part {
            Part::Whole => (0, LAYERS),
            Part::Front(split) => (0, split),
            Part::Back(split, _) => (split, model.pp_end.get()),
        };
        work.residual.resize_within_capacity(count * W)?;
        work.mixed.resize_within_capacity(count * H)?;
        work.block.resize_within_capacity(count * H)?;
        if let Part::Back(_, input) = part {
            model.tp_residual_in(b, &mut work.residual, input, count)?;
        }
        // Hash every row first (same advance order; spans are consecutive from
        // row 0), then run the per-row frontend (PLE hints, preads, dequant,
        // embedding) for the whole tile; dump records keep their row order.
        if first == 0 {
        let mut frontend = Vec::with_capacity(count);
        for (s, inputs) in sessions.iter_mut().zip(tokens) {
            for &token in inputs.iter() {
                frontend.push((token, s.hash.advance(token)));
            }
        }
        if frontend.len() != count {
            return Err("missing hashed prefill row".into());
        }
        let gpu=crate::flash_next_prefill::gpu_prompt() && !verify
            && !sessions.iter().any(|s|s.layer_bytes.is_some() || s.dump.is_some());
        if gpu {
            let dedup=crate::flash_next_prefill::gpu_prompt_dedup();
            let packed=if dedup {model.prefill_reader.frontend_packed_dedup(&frontend)?}
                else {model.prefill_reader.frontend_packed(&frontend)?};
            let input=work.prompt_packed.as_mut().ok_or("missing packed prompt workspace")?;
            b.write_vector_bytes(input,&packed)?;
            gdn::launch_threads(b,if dedup {"flash_next_prompt_frontend_dedup"}else{"flash_next_prompt_frontend"},count*H,256,&mut[
                gdn::Arg::Ptr(input.device_address()),gdn::Arg::Ptr(model.weights.device_address(EMBED)?),
                gdn::Arg::Ptr(work.residual.device_address()),gdn::Arg::Ptr(work.lexical.device_address()),
                gdn::Arg::U32(count as u32),gdn::Arg::F32(model.ple_global),
            ])?;
        } else {
        model.prefill_reader.frontend_prompt(&frontend, model.ple_global, &mut work.prompt)?;
        for ((s, inputs), span) in sessions.iter_mut().zip(tokens).zip(spans) {
            for (&token, row) in inputs.iter().zip(span.clone()) {
                if let Some(d) = &mut s.dump {
                    d.push(format!("token {} {token}", s.pos));
                }
                s.rec("embed", &work.prompt.0[row * W..(row + 1) * W]);
            }
        }
        {
            let (x, e) = &work.prompt;
            b.write_vector(&mut work.residual, &x[..count * W])?;
            b.write_vector(&mut work.lexical.slice(0..count * H)?, &e[..count * H])?;
        }
        }
        }
        // No session reads the residual between a MoE combine and the next layer's mix:
        // that combine then runs inside the mix's norm (combine_mix_tiled).
        let unobserved = !sessions.iter().any(|s| s.layer_bytes.is_some() || s.dump.is_some());
        let mut pending_moe: Option<usize> = None;
        // VADE_MTP_GRAPH (default on): a verify pass whose shape was seen before captures its stage's layers after
        // the first as segments (one Spark: 1-47 in four; VADE_W2_PP: clipped at the split, 24 eager on the back),
        // launching each (updated in place when its topology holds) while the host captures the next. The captured
        // work is the eager work in the same order. A shape's first pass runs eagerly; the oldest shape retires.
        let graphed = if GRAPH && verify && unobserved && !model.tp_host_pass(count) {
            let shape = VerifyShape {
                rows: count,
                chats: sessions.len(),
                shallow: sessions.iter().zip(spans).enumerate()
                    .filter(|(_, (s, span))| gdn::shallow_verify(s.pos, span.len()))
                    .fold(0, |mask, (i, _)| mask | 1u32 << i),
            };
            if let Some(i) = work.verify_graphs.iter().position(|(k, _)| *k == shape) {
                Some(i)
            } else if let Some(i) = work.verify_seen.iter().position(|k| *k == shape) {
                work.verify_seen.remove(i);
                if work.verify_graphs.len() == VERIFY_SHAPES {
                    work.verify_graphs.remove(0);
                }
                work.verify_graphs.push((shape, Default::default()));
                Some(work.verify_graphs.len() - 1)
            } else {
                if work.verify_seen.len() == VERIFY_SHAPES {
                    work.verify_seen.remove(0);
                }
                work.verify_seen.push(shape);
                work.verify_counts[3] += 1;
                None
            }
        } else {
            None
        };
        let mut open: Option<usize> = None;
        // Passes above 16 rows come only from a 32-row tick (VADE_VERIFY_ROWS): stage PLE at once.
        let ple_rows = blocks::PLE_PASS_ROWS;
        for li in first..end { model.tp_post(b, sessions, spans, &work.residual, count, verify, li)?;
            if let Some(entry) = graphed.filter(|_| li % VERIFY_SEGMENT == 1) {
                if let Some(slot) = open.take() {
                    let updated = unsafe { b.end_segment(&mut work.verify_graphs[entry].1[slot])? };
                    work.verify_counts[if updated { 1 } else { 2 }] += 1;
                }
                if b.begin_segment(VERIFY_SEGMENT.min(end - li), count)? {
                    open = Some(li / VERIFY_SEGMENT);
                }
            }
            if li == 1 && verify && count <= ple_rows && sessions.iter().all(|s| s.dump.is_none()) {
                // Key/value projections and gates once for the pass (row kernels keep each
                // row's one-row sums); every row then convolves in order after its snapshot.
                let base = work.residual.device_address();
                let (input, gated) = sessions[0].ple.stage_pass(b, &work.lexical, base, count)?;
                for (s, span) in sessions.iter_mut().zip(spans) {
                    for row in span.clone() {
                        let saved = &mut s.saved[row - span.start];
                        s.ple.copy_history(b, &mut saved.ple)?;
                        s.ple.select_history(&mut saved.ple);
                        let at = |a: u64, width: usize| a + (row * width * 4) as u64;
                        s.ple.convolve_row(b, at(input, W), at(base, W), at(gated, W))?;
                    }
                }
            } else if li == 1 {
                for (s, span) in sessions.iter_mut().zip(spans) {
                    let tile = if !verify && s.dump.is_none() {
                        blocks::PLE_TOKENS
                    } else {
                        1
                    };
                    if tile == blocks::PLE_TOKENS {
                        // Project four tiles per pass over the weights; gate and convolve
                        // the same 64-row tiles in the same order.
                        for slice in span.clone().step_by(blocks::PLE_SLICE) {
                            let slice_end = (slice + blocks::PLE_SLICE).min(span.end);
                            s.ple.project_slice(
                                b,
                                &work.lexical.slice(slice * H..slice_end * H)?,
                                slice_end - slice,
                            )?;
                            for first in (slice..slice_end).step_by(tile) {
                                let end = (first + tile).min(slice_end);
                                s.ple.finish_tile(
                                    b,
                                    &mut work.residual.slice(first * W..end * W)?,
                                    first - slice,
                                    end - first,
                                )?;
                            }
                        }
                        continue;
                    }
                    for first in span.clone().step_by(tile) {
                        let end = (first + tile).min(span.end);
                        let n = end - first;
                        for row in first..end {
                            s.rec("layer1.ple_e", &work.prompt.1[row * H..(row + 1) * H]);
                        }
                        let at = first * H;
                        if verify {
                            let saved = &mut s.saved[first - span.start];
                            s.ple.copy_history(b, &mut saved.ple)?;
                            s.ple.select_history(&mut saved.ple);
                        }
                        s.ple.project_next(
                            b,
                            &mut work.residual.slice(first * W..end * W)?,
                            &work.lexical.slice(at..at + n * H)?,
                            n,
                        )?;
                        if s.dump.is_some() {
                            s.rec(
                                "layer1.ple",
                                &b.read_vector_range(&work.residual, first * W, W)?,
                            );
                        }
                    }
                }
            }
            if let Some(prev) = pending_moe.take() {
                let finish = work.deferred_finish.take();
                sessions[0].attn_hc[li].combine_mix_tiled(
                    &sessions[0].mlp_hc[prev],
                    b,
                    &mut work.hc,
                    &mut work.residual,
                    &work.block,
                    &work.mixed,
                    count,
                    finish,
                )?;
            } else {
                sessions[0].attn_hc[li].mix_tiled(
                    b,
                    &mut work.hc,
                    &work.residual,
                    &work.mixed,
                    count,
                )?;
            }
            Self::check_ops(
                b,
                sessions,
                spans,
                &format!("layer{li}.hc_attn"),
                &work.mixed,
                H,
            )?;
            if li % 4 == 3
                && sessions
                    .iter()
                    .all(|s| s.indexer.select(li, s.pos, s.pos + 1) == IndexSelection::Sparse)
            {
                let mut attention = sessions
                    .iter_mut()
                    .map(|s| {
                        Qsa::load_next_rows(
                            b,
                            &model.weights,
                            li,
                            s.capacity,
                            model.qsa_proj[li],
                            s.qsa_state[li].take(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let batch = work.qsa_batch.as_mut().unwrap();
                attention[0].project_batch(b, batch, &work.mixed, count)?;
                for (q, span) in attention.iter_mut().zip(spans) {
                    work.qsa.use_norm(model.qsa_norms[li].as_ref().unwrap())?;
                    q.attend_batch_row(b, batch, span.start, &mut work.qsa, span.len())?;
                }
                attention[0].finish_batch(b, batch, &mut work.block, count)?;
                for (s, q) in sessions.iter_mut().zip(attention) {
                    s.qsa_state[li] = Some(q.into_state());
                }
            } else if li % 4 == 3 {
                // Custom dense hooks share the same projection schedule. Sparse
                // is the model/default path, and its projections are pooled above.
                for (s, span) in sessions.iter_mut().zip(spans) {
                    let input = work.mixed.slice(span.start * H..span.end * H)?;
                    work.qsa.use_norm(model.qsa_norms[li].as_ref().unwrap())?;
                    let mut q = Qsa::load_next_rows(
                        b,
                        &model.weights,
                        li,
                        s.capacity,
                        model.qsa_proj[li],
                        s.qsa_state[li].take(),
                    )?;
                    q.project(b, &mut work.qsa, &input, span.len())?;
                    if s.indexer.select(li, s.pos, s.pos + 1) == IndexSelection::Dense {
                        q.attend_dense(b, &mut work.qsa)?;
                    } else {
                        q.attend(b, &mut work.qsa)?;
                    }
                    b.copy_range(
                        work.qsa.output(),
                        &mut work.block,
                        0,
                        span.start * H,
                        span.len() * H,
                    )?;
                    s.qsa_state[li] = Some(q.into_state());
                }
            } else {
                let mut attention = sessions
                    .iter_mut()
                    .map(|s| {
                        Gdn::load_next(
                            b,
                            &model.weights,
                            li,
                            model.gdn_proj[li],
                            s.gdn_state[li].take(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                work.gdn
                    .copy_norm(b, model.gdn_norms[li].as_ref().unwrap())?;
                attention[0].project_next(b, &work.gdn, &work.mixed, count)?;
                // Every session's call fully fused (prompt rows, no verify): the
                // gated scans write the out projection's packed input directly.
                if !verify && count >= 64 && spans.iter().all(|span| gdn::fuses_fully(span.len())) {
                    let tile = b.flash_dense_tile(attention[0].out_cols(), count)?;
                    work.gdn.packed_out.set(Some((tile, count)));
                }
                // VADE_GDN_COHORT: a multi-chat verify pass whose windows all run the rows kernel
                // uploads one descriptor table and gates every row in one launch (same words).
                let cohort = verify && sessions.len() > 1 && sessions.len() <= COHORT_TABLE_CHATS
                    && gdn::cohort_verify() && gdn::chunk_form()
                    && spans.iter().all(|span| span.len() <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT);
                if cohort {
                    let table: Vec<f32> = attention.iter().flat_map(|g| g.descriptor_words()).collect();
                    work.recurrence.resize_within_capacity(table.len())?;
                    b.write_vector(&mut work.recurrence, &table)?;
                    work.gdn.defer_gate.set(true);
                }
                // gdn::cohort_copies: every chat's entry history and tape, before any of this layer's scans.
                let copies = cohort && gdn::cohort_copies();
                // gdn::cohort_kernels: every chat's prepare, rows and history in one launch each.
                let kernels = cohort && gdn::cohort_kernels() && gdn::cohort_kernel_entries(b);
                let scans = (|| -> Result<()> {
                if copies {
                    let (mut wide, mut narrow) = (Vec::with_capacity(2 * sessions.len()), Vec::with_capacity(2 * sessions.len()));
                    for ((s, g), span) in sessions.iter_mut().zip(attention.iter()).zip(spans) {
                        let rollback = s.rollback.as_mut().ok_or("MTP rollback missing")?;
                        wide.push(g.history_copy_row(rollback.gdn[li].as_ref().ok_or("MTP entry state missing")?)?);
                        let [qkv, a, gb] = rollback.inputs[li].as_mut().ok_or("MTP GDN tape missing")?.capture_rows(&work.gdn, span.start, span.len())?;
                        wide.push(qkv); narrow.extend([a, gb]);
                    }
                    gdn::hybrid_copy_rows(b, &mut work.copy_rows, &[&wide, &narrow])?;
                    work.gdn.entry_copied.set(true);
                }
                if kernels { *work.gdn.cohort.borrow_mut() = Some(gdn::CohortLaunches::default()); }
                for (chat, ((s, g), span)) in sessions.iter_mut().zip(attention.iter_mut()).zip(spans).enumerate() {
                    let pos = s.pos;
                    let table = cohort.then(|| work.recurrence.device_address() + (chat * 24) as u64);
                    if verify {
                        let rollback = s.rollback.as_mut().ok_or("MTP rollback missing")?;
                        rollback.shallow = gdn::shallow_verify(pos, span.len());
                        let tape = rollback.inputs[li].as_mut().ok_or("MTP GDN tape missing")?;
                        // With gdn::cohort_copies the layer's tapes and entry histories are already copied.
                        let tape = if copies { None } else if gdn::fused_verify_copy(b, &work.recurrence) {
                            Some(tape.capture_ranges(&work.gdn, span.start, span.len())?)
                        } else {
                            tape.capture(b, &work.gdn, span.start, span.len())?;
                            None
                        };
                        g.verify_next(
                            b,
                            &work.gdn,
                            span.start,
                            span.len(),
                            &mut work.recurrence,
                            rollback.gdn[li].as_mut().ok_or("MTP entry state missing")?,
                            pos,
                            tape,
                            table,
                        )?;
                    } else {
                        g.recur_next(b, &work.gdn, span.start, span.len(), &mut work.recurrence, pos)?;
                    }
                }
                if kernels { gdn::launch_cohort(b, &work.gdn, &mut work.gdn_launches)?; }
                Ok(())
                })();
                work.gdn.entry_copied.set(false);
                work.gdn.cohort.borrow_mut().take();
                if cohort {
                    work.gdn.defer_gate.set(false);
                    work.recurrence.resize_within_capacity(6)?;
                }
                scans?;
                if cohort {
                    attention[0].gate_rows(b, &work.gdn, 0, count)?;
                }
                attention[0].finish_next(b, &work.gdn, &work.block, count)?;
                for (s, g) in sessions.iter_mut().zip(attention) {
                    s.gdn_state[li] = Some(g.into_state());
                }
            }
            Self::check_ops(
                b,
                sessions,
                spans,
                &format!("layer{li}.attention"),
                &work.block,
                H,
            )?;
            for (s, span) in sessions.iter_mut().zip(spans) {
                if s.dump.is_some() {
                    s.rec(
                        &format!("layer{li}.attn"),
                        &b.read_vector_range(&work.block, span.start * H, span.len() * H)?,
                    );
                }
            }
            work.mixed16_ready=false;
            if !unobserved {
                sessions[0].attn_hc[li].combine_tiled(
                    b,
                    &mut work.hc,
                    &mut work.residual,
                    &work.block,
                    count,
                )?;
                Self::check_ops(
                    b,
                    sessions,
                    spans,
                    &format!("layer{li}.combine_attn"),
                    &work.residual,
                    W,
                )?;
                sessions[0].mlp_hc[li].mix_tiled(
                    b,
                    &mut work.hc,
                    &work.residual,
                    &work.mixed,
                    count,
                )?;
            } else if !verify && crate::flash_prompt_options::cached().cold8k_r2 && crate::flash_prompt_options::Options::kernel_rows(count) {
                work.mixed16_ready=sessions[0].mlp_hc[li].combine_mix_tiled_a16(
                    &sessions[0].attn_hc[li],b,&mut work.hc,&mut work.residual,&work.block,&work.mixed,&work.mixed16,count)?;
            } else {
                // The attention combine and the MLP mix's norm share one residual pass.
                sessions[0].mlp_hc[li].combine_mix_tiled(
                    &sessions[0].attn_hc[li],
                    b,
                    &mut work.hc,
                    &mut work.residual,
                    &work.block,
                    &work.mixed,
                    count,
                    None,
                )?;
            }
            Self::check_ops(
                b,
                sessions,
                spans,
                &format!("layer{li}.hc_mlp"),
                &work.mixed,
                H,
            )?;
            // A combine deferred below also takes over this layer's k16_finish.
            work.defer_finish = unobserved && li >= 1 && li + 1 < end && !model.tp_plain(li, count);
            work.moe_at(b, &model.moe[li], count, None, !verify, model.tp_moe(li, count))?;
            work.defer_finish = false;
            Self::check_ops(
                b,
                sessions,
                spans,
                &format!("layer{li}.moe"),
                &work.block,
                H,
            )?;
            let routing = if sessions.iter().any(|s| s.dump.is_some()) {
                work.dump_routes(b, count)?
            } else {
                Vec::new()
            };
            // Layer 1 starts with the PLE residual update and the last layer feeds the final
            // mixer: those combines run here.
            if unobserved && li >= 1 && li + 1 < end && !model.tp_plain(li, count) {
                pending_moe = Some(li);
            } else {
                sessions[0].mlp_hc[li].combine_tiled(
                    b,
                    &mut work.hc,
                    &mut work.residual,
                    &work.block,
                    count,
                )?;
            }
            for (s, span) in sessions.iter_mut().zip(spans) {
                if let Some(layers) = &mut s.layer_bytes {
                    use std::io::Write;
                    let values =
                        b.read_vector_range(&work.residual, span.start * W, span.len() * W)?;
                    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(layers.join(format!("layer{li}.f32")))
                        .map_err(err)?
                        .write_all(&bytes)
                        .map_err(err)?;
                }
                if let Some(d) = &mut s.dump {
                    d.push(dump::moe_line(li, &routing[span.start]));
                }
                if s.dump.is_some() {
                    s.rec(
                        &format!("layer{li}.x"),
                        &b.read_vector_range(&work.residual, span.start * W, span.len() * W)?,
                    );
                }
            }
        }
        // A captured segment closes before a pipeline stage hands its rows on (exchange, return).
        if let (Some(entry), Some(slot)) = (graphed, open.take()) {
            let updated = unsafe { b.end_segment(&mut work.verify_graphs[entry].1[slot])? };
            work.verify_counts[if updated { 1 } else { 2 }] += 1;
        }
        if graphed.is_some() {
            work.verify_counts[0] += 1;
            let passes = work.verify_counts[0];
            if passes.is_power_of_two() || passes % 256 == 0 {
                let [_, updated, instantiated, eager] = work.verify_counts;
                eprintln!("flash_next mtp_graph passes={passes} eager_passes={eager} segments_updated={updated} segments_instantiated={instantiated} shapes={} deferred_frees={} rows={count} chats={}", work.verify_graphs.len(), b.deferred_frees(), sessions.len());
            }
        }
        match part {
            Part::Back(..) => return Ok(()),
            Part::Front(_) => {
                let meta: Vec<[u32; 4]> = sessions
                    .iter()
                    .zip(spans)
                    .map(|(s, span)| [s.slot, s.capacity as u32, s.pos as u32, span.len() as u32])
                    .collect();
                // VADE_W2_PP_MTP=1: a chat at position 0 drops rows a previous chat on its slot left.
                if model.pp_mtp() {
                    sessions.iter().filter(|s| s.pos == 0).for_each(|s| model.pp_drop_kept(s.slot));
                }
                // VADE_W2_PP_ASYNC=1: a tile without projection is queued; the back runs it while
                // this rank runs the next tile. With VADE_W2_PP_MTP=1 the back keeps the rows of
                // chats that draft; they come back with the next synchronous reply and enter each
                // chat's draft history before its next observe (observe_kept).
                if !output && model.pp_async() {
                    let keep = sessions.iter().zip(tokens).zip(spans).filter(|((s, _), _)| model.pp_mtp() && s.draft.is_some())
                        .map(|((s, t), span)| super::pp::KeepPart { slot: s.slot, pos: s.pos, tokens: t.to_vec(), start: span.start })
                        .collect::<Vec<_>>();
                    // Past KEEP_ROWS retained rows the tile runs synchronously below instead.
                    if keep.is_empty() || model.pp_keep_room(count) {
                        return model.pp_post(b, &meta, &work.residual, count, Some(keep).filter(|k| !k.is_empty()));
                    }
                }
                Self::pp_return_and_observe(b, model, sessions, work, output, verify, &meta, count)?;
            }
            Part::Whole => (),
        }
        for ((s, input), span) in sessions.iter_mut().zip(tokens).zip(spans) {
            if verify {
                for i in 1..span.len() {
                    b.copy_range(
                        &work.residual,
                        &mut s.saved[i].last,
                        (span.start + i - 1) * W,
                        0,
                        W,
                    )?;
                }
            }
            // Any span of two or more rows (prompt tiles of chats admitted
            // above the drafting cohort, verify rows) enters the draft history in observe_rows
            // tiles (the history `observe` writes); VADE_MTP_TILE_ALL=0 keeps tiles for
            // VADE_MTP_BATCH chats only.
            let tiled = s.batch_drafts || (input.len() > 1 && super::mtp::tile_all());
            if let Some(d) = &mut s.draft {
                Self::flush_into(b, model, d, work)?;
                if (matches!(part, Part::Front(_)) || model.tp_on()) && model.pp_mtp() {
                    Self::observe_kept(b, model, d, work, s.slot, s.pos, s.batch_drafts)?;
                }
                if tiled {
                    d.observe_rows(b, model, &work.mixed16, &mut work.hc, &mut work.qsa, s.pos, input,
                        &work.residual, span.start)?;
                } else {
                    d.observe(b, model, s.pos, input, &work.residual, span.start)?;
                }
            }
        }
        if output {
            sessions[0]
                .mixer
                .mix_tiled(b, &mut work.hc, &work.residual, &work.mixed, count)?;
            if verify {
                if work.bound_head {
                    Head::project_bound_rows(b, model.weights.device_address("lm_head.weight")?, model.vocabulary(), count,
                        &work.mixed, &mut work.head_logits, &mut work.head_partials, &mut work.head_result,
                        model.head_bounds.as_ref().ok_or("head bounds missing")?)?;
                } else { Head::project_next_rows(
                    b,
                    model.weights.device_address("lm_head.weight")?,
                    model.vocabulary(),
                    count,
                    &work.mixed,
                    &mut work.head_logits,
                    &mut work.head_partials,
                    &mut work.head_result,
                )?; }
                for (s, span) in sessions.iter_mut().zip(spans) {
                    if work.bound_head {
                        let d = s.draft.as_mut().ok_or("MTP disabled")?;
                        b.copy_range(&work.mixed, d.bound_input.as_mut().ok_or("missing retained head inputs")?,
                            span.start * H, 0, span.len() * H)?;
                        d.bound_at.set(Some((s.pos + span.len(), span.len())));
                    }
                    s.head
                        .publish_next(b, &work.head_logits, &work.head_result, span.end - 1)?;
                    b.copy_range(
                        &work.head_logits,
                        &mut s.draft.as_mut().ok_or("MTP disabled")?.verification,
                        span.start * model.vocabulary(),
                        0,
                        span.len() * model.vocabulary(),
                    )?;
                }
                return Ok(());
            }
            work.head_input.resize_within_capacity(sessions.len() * H)?;
            for (row, span) in spans.iter().enumerate() {
                b.copy_range(
                    &work.mixed,
                    &mut work.head_input,
                    (span.end - 1) * H,
                    row * H,
                    H,
                )?;
            }
            Head::project_next_batch(
                b,
                &mut sessions.iter_mut().map(|s| &mut s.head).collect::<Vec<_>>(),
                &work.head_input,
                &mut work.head_logits,
                &mut work.head_partials,
                &mut work.head_result,
            )?;
            for (row, s) in sessions.iter_mut().enumerate() {
                if s.dump.is_some() {
                    s.rec("final", &b.read_vector_range(&work.head_input, row * H, H)?);
                }
                if let Some(d) = &mut s.dump {
                    d.push(dump::top_line(&s.head.logits(b)?));
                }
            }
        }
        Ok(())
    }
}

/// Which layers `execute` runs on this rank (VADE_W2_PP=1 splits them at `split`).
#[derive(Clone, Copy)]
pub(super) enum Part<'p> {
    Whole,
    Front(usize),
    Back(usize, &'p [f32]),
}
#[path = "next_pool.rs"]
mod pool;
#[path = "next_execution_pp.rs"]
mod pp_rows;
pub(super) use pool::Pool;
/// Layers per captured verify segment (VADE_MTP_GRAPH): segments start at layers 1, 13, 25, 37.
const VERIFY_SEGMENT: usize = 12;
/// Verify shapes with retained segments (VADE_MTP_GRAPH), and shapes remembered as seen once.
const VERIFY_SHAPES: usize = 6;
/// Chats a cohort verify pass's GDN descriptor table holds (VADE_GDN_COHORT).
const COHORT_TABLE_CHATS: usize = 32;
/// A verify pass's capture key: its launches depend on the row count, the chats and, per chat,
/// whether its GDN window stays inside one chunk (gdn::shallow_verify).
#[derive(Clone, Copy, PartialEq, Eq)]
struct VerifyShape {
    rows: usize,
    chats: usize,
    shallow: u32,
}
/// VADE_MTP_GRAPH: verify passes run as captured segments (default on; VADE_MTP_GRAPH=0 runs them
/// eagerly). Read once.
fn mtp_graph() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_MTP_GRAPH").is_ok_and(|v| v == "0"))
}
/// (fp4_bw_down_pdl + PDL allowed, fp4_bw_gate_c1 entry) for the process's PTX.
fn bw_entries(b: &Backend<'_>) -> (bool, bool) {
    static ON: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let pdl = std::env::var("VADE_BW_PDL").is_ok_and(|v| v == "1");
        let c1 = b.has_flash("fp4_bw_gate_c1");
        (pdl && c1 && b.has_flash("fp4_bw_down_pdl"), c1)
    })
}

#[cfg(test)]
#[test]
fn mtp_reservation_preserves_default_and_funds_deeper_heads() {
    let reserved = |depth| Workspace::reservation_bytes(32768, 4096, depth, false);
    assert_eq!(reserved(3), reserved(4)); // The previous 80-row reservation.
    assert_eq!(reserved(7) - reserved(3), 48 * (H + 248320 + 192 * 3 + 3) * 4);
    println!("workspace reservation capacity=32768 rows=4096 depth3={} depth7={}", reserved(3), reserved(7));
}
