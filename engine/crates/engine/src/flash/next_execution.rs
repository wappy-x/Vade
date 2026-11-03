//! One layer traversal for a prompt tile or independent chat rows.
use super::*;
use spark_gpu_probe::driver::Fp8RoutingFlag;
use crate::flash_next_prefill::Half;
#[path = "next_prompt_route.rs"]
mod prompt_route;

pub(super) struct Workspace<'a> {
    /// VADE_MTP_GRAPH and flash_tp_switch::plain_segments: per shape (verify or plain, rows, chats, chats whose GDN
    /// window stays inside its chunk) one retained executable per segment (layers 1-2, 3-12, 13-24, 25-36, 37-47),
    /// oldest first; shapes seen once (eager); counters (graphed passes, in-place updates,
    /// instantiations, eager passes).
    verify_graphs: Vec<(VerifyShape, [Option<spark_gpu_probe::driver::DecodeGraph<'a>>; 5])>,
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
    /// A greedy verify pass may leave sparse head logits (verify_batch_greedy). A plain step has
    /// its own rules (flash_next_rows::bounded_step, head_split::plain_eligible).
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
            let mut g = model.gdn_load(b, li, Some(owner))?;
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
        // pooled and one-row passes (_pack_mask entries; one row: C1, empty mask), verify / draft rows (_compact_route_pack).
        let pack_route = silu_route && !dense && count <= 16
            && if mask_route || one_route {
                b.has_flash(if one_route { "flash_next_route_compact1_pack_mask" } else { "flash_next_compact_route_pack_mask" })
            } else {
                mask.is_none() && b.has_flash("flash_next_compact_route_pack")
            };
        // route_ids checks rows before slots or table: zero rows keep every assignment, SiLU and plane order.
        if one_route && pack_route && mask_step.is_none() { mask_step = Some(spark_gpu_probe::next_math::SlotStep { slots: 0, table: 0, layer: 0, rows: 0 }); }
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
            // Off by default: decode_fuse on or VADE_BW_PDL=1 turns it on. At 8 rows the
            // overlap costs time (DRAM-bound), so it is never used there.
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
        // Shared is the eleventh row in each token's partials, after experts (the tensor split:
        // this rank's rows, the columns its finish reads).
        let shared = self.expert_out.device_address() + (10 * H * 4) as u64 - tp.map_or(0, |t| t.routed());
        let (weight, shared, rows) = tp.map_or((addr.shared[2], shared, H), |t| t.shared_rows(addr.shared[2], shared, 640));
        blocks::next_rows_strided(b, weight, self.gate.device_address(), shared, rows, 640, count, 11 * H)?;
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
                tp.map_or(256, |t| t.finish_threads()),
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
        Self::run_rows_inner(b, sessions, tokens, output, false, false).inspect_err(|_| if let Some(s) = sessions.first() { s.model.tp_seq_abandon(b) })
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
        if sessions.iter().any(|s| !s.valid || !std::ptr::eq(s.model, model)) {
            return Err("Flash-Next cohort needs valid sessions of the same model".into());
        }
        // Every row this pass writes has its KV pages before any kernel is queued.
        for (s, t) in sessions.iter_mut().zip(tokens) {
            crate::flash_next_rows::check_owners(&s.gdn_state, &s.qsa_state, s.model.state_range())?;
            s.fit_kv(b, s.pos + t.len())?;
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
            Self::run_pooled(b, sessions, tokens, output, greedy_head)?;
            return Ok(source);
        }
        if count > crate::flash_next_rows::prompt_tile() {
            return Err("Flash-Next execution tile exceeds the prompt tile".into());
        }
        // Pipeline front: layers [split, 48) run on the back rank (MTP verify included).
        let part = match model.pp_front_split().or(model.tp_part(b, sessions, &spans, verify, output)?) {
            None => Part::Whole,
            Some(_) if verify && !model.pp_mtp() => return Err("this pipeline serves plain decoding only (MTP verify=0)".into()),
            Some(split) => Part::Front(split),
        };
        model.tp_gdn_frame(b, sessions, count)?;
        // All admission and workspace allocation failures precede state mutation.
        let owner = crate::flash_next_rows::workspace_owner(sessions.iter().map(|s| s.capacity))?;
        let capacity = sessions[owner].capacity;
        let mut work = model.take_work(b, capacity, count)?;
        work.bound_head = Self::bound_head(model, sessions, count, greedy_head, verify, work.head_result.len());
        // Rank 0 decides each pass whether its plain greedy head splits; tp_post sends it to rank 1.
        model.tp_set_head_pass(greedy_head && output && !verify && matches!(part, Part::Whole)
            && head_split::plain_eligible(model, sessions, count, &work));
        if let Err(error) = work.status.clear(b, count, 0) {
            model.return_work(work);
            return Err(error);
        }
        for s in sessions.iter_mut() {
            s.valid = false;
            s.logits_ready = false;
            if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
            s.head.take_split();
        }
        // VADE_MTP_GRAPH=0: the eager build of `execute` for verify passes, which holds no segment code.
        // Pipeline stages capture their own layers' segments; each closes before the exchange.
        let result = if verify && mtp_graph() || crate::flash_tp_switch::plain_segments(model.tp_on(), verify, b.decode_readback_active(), tokens.iter().map(|t| t.len())) {
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
        // A posted verify group: its tail and positions follow in verify_finish (overlap).
        if result.is_ok() && model.pp_posted(false) { return Ok(source); }
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
    /// The single-chat verify head projects only rows above the retained bound.
    fn bound_head(model: &NextModel<'a>, sessions: &[&mut Self], count: usize, greedy_head: bool, verify: bool, result: usize) -> bool {
        greedy_head && verify && sessions.len() == 1 && (head_split::eligible(model, sessions.len(), count, verify, result)
            || (1..=crate::flash_head_options::RETAINED_ROWS).contains(&count) && model.head_bounds.is_some())
            && sessions.iter().all(|s| s.dump.is_none() && s.layer_bytes.is_none())
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
    /// The per-row frontend of a pass that starts at layer 0, for every row of `spans`: hash
    /// each row in order, then PLE hints, preads, dequant and embedding into the residual and
    /// lexical rows; dump records keep their row order. An unobserved verify pass reads its
    /// embedding rows only and returns its rows: layer 1 reads the PLE rows first, so
    /// `frontend_ple` reads them after layer 0's launches, while the GPU runs layer 0.
    pub(super) fn frontend(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        spans: &[std::ops::Range<usize>],
        verify: bool,
    ) -> Result<Option<Vec<(u32, Vec<u64>)>>> {
        let model = sessions[0].model;
        let count = spans.last().unwrap().end;
        let mut frontend = Vec::with_capacity(count);
        for (s, inputs) in sessions.iter_mut().zip(tokens) {
            let inside = s.inside_row();
            for (row, &token) in inputs.iter().enumerate() {
                if inside == Some(row) { s.inside_hash(); }
                frontend.push((token, s.hash.advance(token)));
            }
        }
        if frontend.len() != count {
            return Err("missing hashed prefill row".into());
        }
        let unobserved = !sessions.iter().any(|s| s.layer_bytes.is_some() || s.dump.is_some());
        let gpu = crate::flash_next_prefill::gpu_prompt() && !verify && unobserved;
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
        let half = if verify && unobserved { Half::Embedding } else { Half::Both };
        model.prefill_reader.frontend_prompt(&frontend, model.ple_global, &mut work.prompt, half)?;
        for ((s, inputs), span) in sessions.iter_mut().zip(tokens).zip(spans) {
            for (&token, row) in inputs.iter().zip(span.clone()) {
                if let Some(d) = &mut s.dump {
                    d.push(format!("token {} {token}", s.pos));
                }
                s.rec("embed", &work.prompt.0[row * W..(row + 1) * W]);
            }
        }
        b.write_vector(&mut work.residual, &work.prompt.0[..count * W])?;
        if half == Half::Embedding {
            return Ok(Some(frontend));
        }
        b.write_vector(&mut work.lexical.slice(0..count * H)?, &work.prompt.1[..count * H])?;
        }
        Ok(None)
    }
    /// A verify pass's PLE rows (`frontend`'s second half), read and uploaded before layer 1.
    fn frontend_ple(b: &Backend<'a>, work: &mut Workspace<'a>, model: &NextModel<'a>, rows: &[(u32, Vec<u64>)]) -> Result<()> {
        model.prefill_reader.frontend_prompt(rows, model.ple_global, &mut work.prompt, Half::Ple)?;
        b.write_vector(&mut work.lexical.slice(0..rows.len() * H)?, &work.prompt.1[..rows.len() * H])
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
        // Layers this rank runs: all of them (world 1), [0, split) on the pipeline front
        // (the back's rows come back before the tail), [split, 48) on the back (no tail).
        let (first, end) = match part {
            Part::Whole => (0, LAYERS),
            Part::Front(split) => (0, split),
            Part::Back(split, _) => (split, model.tp_back_end().unwrap_or(model.pp_end.get())),
        };
        work.residual.resize_within_capacity(count * W)?;
        work.mixed.resize_within_capacity(count * H)?;
        work.block.resize_within_capacity(count * H)?;
        if let Part::Back(_, input) = part {
            model.tp_residual_in(b, &mut work.residual, input, count)?;
        }
        // Hash every row first (same advance order; spans are consecutive from
        // row 0), then run the per-row frontend (PLE hints, preads, dequant,
        // embedding) for the whole tile; dump records keep their row order. A
        // verify pass's PLE rows wait for the top of layer 1 (`deferred`).
        let mut deferred = if first == 0 { Self::frontend(b, work, sessions, tokens, spans, verify)? } else { None };
        if end <= 1 && let Some(rows) = deferred.take() { Self::frontend_ple(b, work, model, &rows)?; }
        // No session reads the residual between a MoE combine and the next layer's mix:
        // that combine then runs inside the mix's norm (combine_mix_tiled).
        let unobserved = !sessions.iter().any(|s| s.layer_bytes.is_some() || s.dump.is_some());
        let mut pending_moe: Option<usize> = None;
        // VADE_MTP_GRAPH (default on): a verify pass (or a plain one-token pass of the tensor split, flash_tp_switch::plain_segments, under its own shape) whose shape was seen before captures its stage's layers after
        // the first as segments (one Spark: 1-47 in five; the pipeline: clipped at the split, 24 eager on the back),
        // launching each (updated in place when its topology holds) while the host captures the next. The captured
        // work is the eager work in the same order. A shape's first pass runs eagerly; the oldest shape retires.
        let graphed = if GRAPH && unobserved && !model.tp_host_pass(count) {
            let shape = VerifyShape {
                verify, rows: count,
                chats: sessions.len(),
                shallow: if !verify { 0 } else { sessions.iter().zip(spans).enumerate()
                    .filter(|(_, (s, span))| gdn::shallow_verify(s.pos, span.len()))
                    .fold(0, |mask, (i, _)| mask | 1u32 << i) },
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
        for li in first..end { model.tp_seq_layer(b, sessions, li, count)?; model.tp_post(b, sessions, spans, &work.residual, count, verify, li)?;
            if li == 1 && let Some(rows) = deferred.take() { Self::frontend_ple(b, work, model, &rows)?; }
            if let Some((entry, start)) = graphed.zip(segment_start(li)) {
                if let Some(slot) = open.take() {
                    let updated = unsafe { b.end_segment(&mut work.verify_graphs[entry].1[slot])? };
                    work.verify_counts[if updated { 1 } else { 2 }] += 1;
                }
                let next = (li + 1..end).find(|&l| segment_start(l).is_some()).unwrap_or(end);
                if b.begin_segment(next - li, count)? {
                    open = Some(start);
                }
            }
            Self::layer_stage(b, work, sessions, spans, verify, li, end, unobserved, &mut pending_moe)?;
        }
        model.tp_seq_layer(b, sessions, end, count)?;
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
                eprintln!("flash_next segment_graph verify={verify} passes={passes} eager_passes={eager} segments_updated={updated} segments_instantiated={instantiated} shapes={} deferred_frees={} rows={count} chats={}", work.verify_graphs.len(), b.deferred_frees(), sessions.len());
            }
        }
        match part {
            Part::Back(..) => return Self::split_head_back(b, sessions, work, spans, count, verify),
            Part::Front(_) => {
                let meta: Vec<[u32; 4]> = sessions
                    .iter()
                    .zip(spans)
                    .map(|(s, span)| [s.slot, super::pp::capacity_word(s.capacity, s.kv_rows()), s.pos as u32, span.len() as u32])
                    .collect();
                // A chat at position 0 drops rows a previous chat on its slot left.
                if model.pp_mtp() {
                    sessions.iter().filter(|s| s.pos == 0).for_each(|s| model.pp_drop_kept(s.slot));
                }
                // A tile without projection is queued; the back runs it while
                // this rank runs the next tile. The back keeps the rows of
                // chats that draft; they come back with the next synchronous reply and enter each
                // chat's draft history before its next observe (observe_kept).
                if !output && model.pp_async() {
                    let keep = sessions.iter().zip(tokens).zip(spans).filter(|((s, _), _)| model.pp_mtp() && s.draft.is_some())
                        .map(|((s, t), span)| super::pp::KeepPart { slot: s.slot, pos: s.pos, tokens: t.to_vec(), start: span.start })
                        .collect::<Vec<_>>();
                    // Past KEEP_ROWS kept rows the drafters first observe the oldest returned rows (observe_returned); no room freed: synchronous tile.
                    if !keep.is_empty() && !model.pp_keep_room(count) { Self::observe_returned(b, model, sessions, work, count)?; }
                    if keep.is_empty() || model.pp_keep_room(count) {
                        return model.pp_post(b, &meta, &work.residual, count, Some(keep).filter(|k| !k.is_empty())).and_then(|()| model.tp_seq_posted());
                    }
                }
                Self::pp_return_and_observe(b, model, sessions, work, output, verify, &meta, count)?;
                if model.pp_posted(false) { return Ok(()); }
            }
            Part::Whole => model.tp_seq_drain()?,
        }
        Self::execute_tail(b, work, sessions, tokens, spans, output, verify, matches!(part, Part::Front(_)))
    }
    /// The rest of a pass once its rows are on this rank: draft history, then the head.
    /// `front`: the pipeline front (its kept rows enter the draft history first).
    #[allow(clippy::too_many_arguments)]
    fn execute_tail(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        spans: &[std::ops::Range<usize>],
        output: bool,
        verify: bool,
        front: bool,
    ) -> Result<()> {
        let model = sessions[0].model;
        let count = spans.last().unwrap().end;
        for ((s, input), span) in sessions.iter_mut().zip(tokens).zip(spans) {
            s.inside_last(b, &work.residual, span.start)?;
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
                if (front || model.tp_on()) && model.pp_mtp() {
                    Self::observe_kept(b, model, d, work, s.slot, s.pos, s.batch_drafts, None)?;
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
                if Self::split_head_front(b, sessions, work, spans, count)? { return Ok(()); }
                if work.bound_head {
                    Head::project_bound_rows(b, model.weights.device_address("lm_head.weight")?, 0, model.vocabulary(), count,
                        &work.mixed, &mut work.head_logits, &mut work.head_partials, &mut work.head_result,
                        model.head_bounds.as_ref().ok_or("head bounds missing")?)?;
                } else { Head::project_next_rows(b, model.weights.device_address("lm_head.weight")?, model.vocabulary(), count,
                    &work.mixed, &mut work.head_logits, &mut work.head_partials, &mut work.head_result)?; }
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
                return Self::split_head_after(b, model, sessions.len(), work, count);
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
            if model.tp_head_pass() {
                return Self::split_plain_front(b, sessions, work);
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

/// Which layers `execute` runs on this rank (the pipeline splits them at `split`).
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
#[path = "next_execution_overlap.rs"]
mod overlap;
#[path = "next_stage.rs"]
mod stage;
#[path = "next_split_bank.rs"]
mod split_bank;
#[path = "next_split.rs"]
mod split;
#[path = "head_split.rs"]
mod head_split;
pub(super) use pool::Pool;
/// Layers per captured verify segment (VADE_MTP_GRAPH): segments start at layers 1, 3, 13, 25, 37.
const VERIFY_SEGMENT: usize = 12;
/// The slot of the verify segment starting at layer `li`, if one does. On every topology a
/// segment also starts at layer 3: the pass's first segment launch then waits on a two-layer
/// capture only, and the host captures each later segment while the GPU runs the one before
/// (the same launches in the same order, so the same words).
fn segment_start(li: usize) -> Option<usize> {
    if li % VERIFY_SEGMENT == 1 { Some(li / VERIFY_SEGMENT + usize::from(li > 1)) } else { (li == 3).then_some(1) }
}
/// Verify shapes with retained segments (VADE_MTP_GRAPH), and shapes remembered as seen once.
const VERIFY_SHAPES: usize = 6;
/// Chats a cohort verify pass's GDN descriptor table holds (VADE_GDN_COHORT).
const COHORT_TABLE_CHATS: usize = 32;
/// A segmented pass's capture key: verify or plain (their launches differ), the row count, the
/// chats and, per verify chat, whether its GDN window stays inside one chunk (gdn::shallow_verify).
#[derive(Clone, Copy, PartialEq, Eq)]
struct VerifyShape {
    verify: bool, rows: usize,
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
        let pdl = spark_gpu_probe::decode_fuse::on() || std::env::var("VADE_BW_PDL").is_ok_and(|v| v == "1");
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
