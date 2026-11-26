//! The weights of the driver (`mix::Weights`) on a loaded checkpoint.
//!
//! A matrix that the checkpoint stores as BF16 stays with the owner of the weights, and a product runs there on
//! the stored bytes (`ResidentProduct`). The picked expert of a stack is a word range of the stack. Its product
//! with one input row reads the packed stack where the compute has a packed product for the shape, and the widened
//! stack else; both give the same words. Every other
//! weight is small and is held as host words, read from the weight files one time, in the layout that its name
//! in `mix::Weight` gives. A matrix that is stored as 32-bit values, and a projection whose rows change order,
//! are host words too; their product is the plain chain product of the compute (`mix::chain_product`).
//!
//! Layouts made here from the stored form:
//! - Fixed taps by lag: `[channel, lag]`, and lag 0 meets the current token. The description says which stored
//!   tap that is.
//! - The projection of token-made taps: row `head * taps + lag`.
//! - The message queries: for each loop layer the query, key, value and expert row, from the two stored tensors.
use super::{ConvOperands, InputNorms, Operands, Resident, load::Files};
use crate::{Result, bounded_decoder::Compute, loop_decoder::mix::{Message, Weight, Weights, chain_product, check_runs, gathered_runs}};
use engine_model::{plan::{description::{ConvTaps, TapOrder}, loop_load::LoopCheckpoint, loop_packed_unit::UnitShape}, safetensors::Dtype, shapes::Proj};
use std::{cell::Cell,collections::BTreeMap,ops::Range};

/// One gate operand tuple for `ResidentProduct::resident_prefill_gate_fuse`.
pub type GateFuseEntry<'a, V, Val> = (
    &'a V, Range<usize>, &'a V, Range<usize>, &'a Val, &'a Val, &'a Val, &'a Val, Option<&'a [f32]>,
);

/// A compute that can run a product on a matrix that the owner of the weights holds as value `V`.
pub trait ResidentProduct<V>: Compute {
    type RoutePlan;
    fn draft_head_fold(&self,_op:spark_gpu_probe::draft_ops::DraftOp,_head:&V,_fold:&V,_input:&Self::Value)->Result<Self::Value> {
        Err("combined draft head unavailable".into())
    }
    fn draft_panel(&self,_op:spark_gpu_probe::draft_ops::DraftOp,_weight:&V,_x:&Self::Value,_base:Option<&Self::Value>)->Result<Self::Value> {
        Err("draft panel products unavailable".into())
    }
    /// `words` of `weight` hold `rows x columns` BF16 values, two a word, row-major. The result is `rows` 32-bit
    /// words: one chain a row over the columns in rising order from +0.0, each product rounded before its add,
    /// the sum not rounded.
    fn resident_product(&self, weight: &V, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value) -> Result<Self::Value>;
    /// `resident_product` for `inputs` rows of `x` (`[inputs][columns]`). The result is `[inputs][rows]`, row-major,
    /// and each word is the word of `resident_product` on its input row. The body here runs one product a row and
    /// joins them. A compute with a launch for many rows gives its own body with the same words.
    fn resident_product_rows(&self, weight: &V, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value, inputs: usize, _decode: bool) -> Result<Self::Value> {
        if inputs == 0 || inputs.checked_mul(columns) != Some(self.words(x)) { return Err("resident product: the input differs from the input rows x the columns of the weight".into()); }
        if inputs == 1 { return self.resident_product(weight, words, rows, columns, x); }
        let outs = (0..inputs).map(|input| self.resident_product(weight, words.clone(), rows, columns, &self.range(x, input * columns, columns)?)).collect::<Result<Vec<_>>>()?;
        self.join(&outs.iter().collect::<Vec<_>>())
    }
    /// The products of the expert runs of a step of rows with no gather (`Weights::product_runs`). `runs` holds the
    /// matrix of each run in launch order: its value, its words, its rows and its count of launch rows; all runs have
    /// the same rows and `columns` columns. Launch row `i` reads row `input[i]` of `x` (row `i` without a table); row
    /// `j` of the result is launch row `output[j]` (row `j` without a table). The tables passed `mix::check_runs`.
    /// None: this compute has no such body for these runs, and the caller gathers. The body here gives None.
    fn resident_product_runs(&self, runs: &[(&V, Range<usize>, usize, usize)], x: &Self::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<Option<Self::Value>> {
        let _ = (runs, x, columns, input, output);
        Ok(None)
    }
    /// `words` of `weight` hold the unit of one packed matrix (`loop_packed_unit`). The result is the result of
    /// `resident_product` on the same matrix widened to BF16, gate rows first, word for word. None: this compute
    /// has no packed product for this shape now, and the caller reads the widened matrix.
    fn packed_product(&self, _weight: &V, _words: Range<usize>, _unit: UnitShape, _x: &Self::Value) -> Result<Option<Self::Value>> { Ok(None) }
    /// The packed products of several matrices of one shape, each `units[e]` of `weight`, in one call: matrix `e` times
    /// `x` (`shared`) or times words `e * columns ..` of `x`. The result is the products back to back, each the words
    /// of `packed_product` on its matrix. None: no grouped product for this shape now; the caller takes one a matrix.
    fn packed_group(&self, _weight: &V, _units: &[Range<usize>], _unit: UnitShape, _x: &Self::Value, _shared: bool) -> Result<Option<Self::Value>> { Ok(None) }
    /// `resident_product` of each matrix of `matrices` (all `rows x columns`) on its own input row `xs[g]`, in one
    /// call: the results back to back, each the words of `resident_product` on its matrix. None: this compute has no
    /// such call for these matrices now; the caller takes one matrix at a time.
    fn resident_group(&self, _matrices: &[(&V, Range<usize>)], _rows: usize, _columns: usize, _xs: &[&Self::Value]) -> Result<Option<Self::Value>> { Ok(None) }
    /// `packed_group` on the experts that the route picked, with their numbers read where the compute holds them:
    /// `stack` holds the units of all `experts` experts back to back, `ids` holds the picked numbers (whole 32-bit
    /// floats, in pick order). The host reads no number. None: no such product here; the caller reads the numbers.
    fn packed_picked(&self, _stack: &V, _experts: usize, _unit: UnitShape, _ids: &Self::Value, _x: &Self::Value, _shared: bool, _token_rows: usize) -> Result<Option<Self::Value>> { Ok(None) }
    /// Apply the capped activation inside a picked projection of paired rows.
    #[allow(clippy::too_many_arguments)]
    fn packed_gate(&self,_stack:&V,_experts:usize,_unit:UnitShape,_ids:&Self::Value,_x:&Self::Value,_shared:bool,_caps:[f32;4])->Result<Option<Self::Value>> {Ok(None)}
    /// Picked packed products reading strided route rows with optional input row repeat and optional fused gate.
    #[allow(clippy::too_many_arguments)]
    fn packed_picked_strided(&self,_stack:&V,_experts:usize,_unit:UnitShape,_ids:&Self::Value,_picks:usize,_picks_per_row:usize,_id_stride:usize,_x:&Self::Value,_repeat:usize,_caps:Option<[f32;4]>)->Result<Option<Self::Value>> {Ok(None)}
    #[allow(clippy::too_many_arguments)]
    fn packed_picked_strided_compact(&self,stack:&V,experts:usize,unit:UnitShape,ids:&Self::Value,picks:usize,picks_per_row:usize,id_stride:usize,x:&Self::Value,repeat:usize,caps:Option<[f32;4]>,_compact:Option<&Self::Value>)->Result<Option<Self::Value>> {
        self.packed_picked_strided(stack,experts,unit,ids,picks,picks_per_row,id_stride,x,repeat,caps)
    }
    /// The products of the expert runs of a step of rows from their packed units, with no gather: `runs` holds the
    /// stack value, the words of the unit and the count of launch rows of each run in launch order, every unit of the
    /// shape `unit`. The tables are those of `resident_product_runs`. The result is the result of
    /// `resident_product_runs` on the same matrices widened to BF16, gate rows first, word for word. None: no such
    /// body here for these runs; the caller reads the widened matrices.
    fn packed_product_runs(&self, _runs: &[(&V, Range<usize>, usize)], _unit: UnitShape, _x: &Self::Value, _input: Option<&[usize]>, _output: Option<&[usize]>) -> Result<Option<Self::Value>> { Ok(None) }
    /// Rows `ids` of the table in `words` of `table` (`rows x columns` BF16 values, two a word, row-major), each value
    /// as its 32-bit word: `[ids][columns]`. None: no such copy here now.
    fn resident_rows(&self, _table: &V, _words: Range<usize>, _rows: usize, _columns: usize, _ids: &[u32]) -> Result<Option<Self::Value>> { Ok(None) }
    /// Fused multi-output BF16 product over `parts` contiguous row blocks of `weight[words]`.
    #[allow(clippy::too_many_arguments)]
    fn resident_product_rows_split(&self, _weight: &V, _words: Range<usize>, _parts: &[usize], _columns: usize, _x: &Self::Value, _inputs: usize, _decode: bool) -> Result<Option<Vec<Self::Value>>> { Ok(None) }
    /// Dual-input BF16 product over `[x1[t], x2[t]]` (`[inputs, columns/2]` each) with optional BF16 output rounding.
    #[allow(clippy::too_many_arguments)]
    fn resident_product_rows_pair(&self, _weight: &V, _words: Range<usize>, _rows: usize, _columns: usize, _x1: &Self::Value, _x2: &Self::Value, _inputs: usize, _decode: bool, _round_out: bool) -> Result<Option<Self::Value>> { Ok(None) }
    /// Grouped dual-input narrow BF16 product over up to 3 matrices of one shape (`GateDown` Q/K/V) with optional BF16 output rounding.
    #[allow(clippy::too_many_arguments)]
    fn resident_group_rows_pair(&self, _matrices: &[(&V, Range<usize>)], _rows: usize, _columns: usize, _pairs: &[(&Self::Value, &Self::Value)], _inputs: usize, _round_out: bool) -> Result<Option<Self::Value>> { Ok(None) }
    /// Device-side deterministic MoE route grouping (`MOE_GROUP_ROUTE`), returning the weights and the route plan.
    fn device_route_group(&self, _route: &Self::Value, _rows: usize, _picks: usize, _experts: usize) -> Result<Option<(Self::Value, Self::RoutePlan)>> { Ok(None) }
    /// Device-routed tensor-core MoE expert stack GEMM (`PLAIN_TC_ROWS` Mode 1).
    #[allow(clippy::too_many_arguments)]
    fn device_routed_products(&self, _stack: &V, _experts: usize, _rows: usize, _columns: usize, _route: &Self::RoutePlan, _x: &Self::Value, _down: bool) -> Result<Option<Self::Value>> { Ok(None) }
    /// Device-routed tensor-core packed MoE expert stack GEMM (`PLAIN_PACKED_TC_ROWS`, kind 106).
    #[allow(clippy::too_many_arguments)]
    fn device_routed_packed_products(&self, _stack: &V, _experts: usize, _unit: UnitShape, _route: &Self::RoutePlan, _x: &Self::Value, _down: bool) -> Result<Option<Self::Value>> { Ok(None) }
    /// The gated gate-and-up form of `device_routed_packed_products` (`PLAIN_PACKED_TC_ROWS` with `CappedGate`).
    #[allow(clippy::too_many_arguments)]
    fn device_routed_packed_gate(&self, _stack: &V, _experts: usize, _unit: UnitShape, _route: &Self::RoutePlan, _x: &Self::Value, _caps: [f32; 4]) -> Result<Option<Self::Value>> { Ok(None) }
    /// The prefill `Down` products of `device_routed_packed_products` and their `RouteFold` over `weights`
    /// (`[tokens][picks]`) in one call: `[tokens][rows]`, the same words, with the down words kept as BF16 between the two.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_packed_down_fold(&self, _stack: &V, _experts: usize, _unit: UnitShape, _route: &Self::RoutePlan, _x: &Self::Value, _weights: &Self::Value, _picks: usize) -> Result<Option<Self::Value>> { Ok(None) }
    /// `device_routed_packed_gate` then `device_routed_packed_down_fold` in one call, the gated words kept as the BF16
    /// values that the down product reads: the same words. `gate` and `down` are the stack, its experts and its unit.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_packed_experts(&self, _gate: (&V, usize, UnitShape), _down: (&V, usize, UnitShape), _route: &Self::RoutePlan, _x: &Self::Value, _caps: [f32; 4], _weights: &Self::Value, _picks: usize) -> Result<Option<Self::Value>> { Ok(None) }
    /// Prefill: `device_routed_packed_experts` with optional packed-BF16 input (`half_in`), skipping `TC_MODE_PACK`.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_packed_experts_half_in(&self, _gate: (&V, usize, UnitShape), _down: (&V, usize, UnitShape), _route: &Self::RoutePlan, _x: &Self::Value, _caps: [f32; 4], _weights: &Self::Value, _picks: usize) -> Result<Option<Self::Value>> { Ok(None) }
    /// Prefill: `device_routed_packed_experts` fused with `AfterExperts ShortConv` + `next_ring`, with optional packed-BF16 `x`.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_packed_experts_close(
        &self, _gate: (&V, usize, UnitShape), _down: (&V, usize, UnitShape), _route: &Self::RoutePlan,
        _x: &Self::Value, _caps: [f32; 4], _weights: &Self::Value, _picks: usize,
        _residual: &Self::Value, _fixed: &[f32], _ring: &Self::Value, _taps: usize, _first_slot: usize, _bound: f32, _half_in: bool,
    ) -> Result<Option<(Self::Value, Self::Value)>> { Ok(None) }
    /// Prefill fused `GateOuter + GateBlend + RoundedNorm` for 1..=3 stacked `[GateForget, GateGain]` gates.
    #[allow(clippy::too_many_arguments)]
    fn resident_gate_outer_blend_norm(
        &self, _entries: &[(&V, Range<usize>, &Self::Value, &Self::Value, &Self::Value, Option<&[f32]>)],
        _rows: usize, _dim: usize, _k: usize, _eps: f32, _half_out: bool,
    ) -> Result<Option<Vec<Self::Value>>> { Ok(None) }
    /// Prefill BF16-input dense product (`HALF_IN`, skips `TC_MODE_PACK`).
    fn resident_product_rows_half_in(&self, _weight: &V, _words: Range<usize>, _rows: usize, _columns: usize, _x: &Self::Value, _inputs: usize) -> Result<Option<Self::Value>> { Ok(None) }
    /// Prefill BF16-input multi-output split product (`HALF_IN`, skips `TC_MODE_PACK`).
    fn resident_product_rows_split_half_in(&self, _weight: &V, _words: Range<usize>, _parts: &[usize], _columns: usize, _x: &Self::Value, _inputs: usize) -> Result<Option<Vec<Self::Value>>> { Ok(None) }
    /// Prefill BF16-input dual-half product (`TC_MODE_NARROW_HALF` for `rows <= 128`).
    #[allow(clippy::too_many_arguments)]
    fn resident_product_rows_pair_half_in(&self, _weight: &V, _words: Range<usize>, _rows: usize, _columns: usize, _x1: &Self::Value, _x2: &Self::Value, _inputs: usize) -> Result<Option<Self::Value>> { Ok(None) }
    /// Prefill `FLASH_WINDOW_ATTN` emitting packed BF16 output (`FW_HALF_OUT`).
    fn window_rows_half(&self, _a: &crate::bounded_decoder::WindowOne<'_, Self::Value>, _first_position: u64) -> Result<Option<Self::Value>> { Ok(None) }
    /// Prefill `MIX_ONE_POST_HALF_OUT`: packed BF16 post-norm output.
    #[allow(clippy::too_many_arguments)]
    fn mix_rows_normed_half(
        &self, _rows: usize, _dim: usize, _scale: f32, _eps: f32, _post_eps: f32,
        _transpose: bool, _keep_normed: bool, _keep_mixed: bool,
        _queries: &[f32], _list: &[&Self::Value], _stream: &Self::Value,
        _post_weight: Option<&[f32]>, _post_weight_per_query: bool,
    ) -> Result<Option<(Option<Self::Value>, Option<Self::Value>, Self::Value)>> { Ok(None) }
    /// Prefill `MIX_LIST_NORM_HALF`: FP32 messages + packed BF16 gate-normed messages.
    fn mix_prefill_normed_half(&self, _rows: usize, _dim: usize, _scale: f32, _eps: f32, _queries: &[f32], _list: &[&Self::Value]) -> Result<Option<(Self::Value, Self::Value)>> { Ok(None) }
    /// Prefill `PG_OP_MSG_META`: compact lazy `(msg_weights, msg_nscales)` replacing `MIX_LIST_NORM_HALF`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_msg_meta(&self, _rows: usize, _dim: usize, _scale: f32, _eps: f32, _queries: &[f32], _list: &[&Self::Value]) -> Result<Option<(Self::Value, Self::Value)>> { Ok(None) }
    /// Prefill `PG_OP_FULL`: fused `GateDown + GateOuter + GateBlend + RoundedNorm` for `1..=3` gates over lazy sources.
    #[allow(clippy::too_many_arguments)]
    fn resident_prefill_gate_fuse(
        &self, _sources: &[&Self::Value],
        _entries: &[GateFuseEntry<'_, V, Self::Value>],
        _rows: usize, _dim: usize, _k: usize, _eps: f32, _half_out: bool,
    ) -> Result<Option<Vec<Self::Value>>> { Ok(None) }
}
#[cfg(target_os = "linux")]
impl<'b, 'd> ResidentProduct<spark_gpu_probe::driver::DeviceVector<'d>> for crate::bounded_decoder::Device<'b, 'd> {
    type RoutePlan = spark_gpu_probe::driver::MoeRoutePlan<'d>;
    fn draft_head_fold(&self,op:spark_gpu_probe::draft_ops::DraftOp,head:&spark_gpu_probe::driver::DeviceVector<'d>,fold:&spark_gpu_probe::driver::DeviceVector<'d>,input:&Self::Value)->Result<Self::Value> {
        self.combined_head_fold(op,head,fold,input)
    }
    fn draft_panel(&self,op:spark_gpu_probe::draft_ops::DraftOp,weight:&spark_gpu_probe::driver::DeviceVector<'d>,x:&Self::Value,base:Option<&Self::Value>)->Result<Self::Value> {
        self.panel_projection(op,weight,x,base)
    }
    fn resident_product(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value) -> Result<Self::Value> {
        crate::bounded_decoder::Device::resident_product(self, weight, words, rows, columns, x)
    }
    fn resident_product_rows(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value, inputs: usize, decode: bool) -> Result<Self::Value> {
        crate::bounded_decoder::Device::resident_product_rows(self, weight, words, rows, columns, x, inputs, decode)
    }
    fn resident_product_runs(&self, runs: &[(&spark_gpu_probe::driver::DeviceVector<'d>, Range<usize>, usize, usize)], x: &Self::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_product_runs(self, runs, x, columns, input, output)
    }
    fn packed_product(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, unit: UnitShape, x: &Self::Value) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_product(self, weight, words, unit, x)
    }
    fn packed_group(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, units: &[Range<usize>], unit: UnitShape, x: &Self::Value, shared: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_group(self, weight, units, unit, x, shared)
    }
    fn packed_gate(&self,stack:&spark_gpu_probe::driver::DeviceVector<'d>,experts:usize,unit:UnitShape,ids:&Self::Value,x:&Self::Value,shared:bool,caps:[f32;4])->Result<Option<Self::Value>> {
        self.packed_picked(stack,experts,unit,ids,x,shared,shared,Some(caps))
    }
    fn packed_picked_strided(&self,stack:&spark_gpu_probe::driver::DeviceVector<'d>,experts:usize,unit:UnitShape,ids:&Self::Value,picks:usize,picks_per_row:usize,id_stride:usize,x:&Self::Value,repeat:usize,caps:Option<[f32;4]>)->Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_picked_strided(self,stack,experts,unit,ids,picks,picks_per_row,id_stride,x,repeat,caps,false)
    }
    fn packed_picked_strided_compact(&self,stack:&spark_gpu_probe::driver::DeviceVector<'d>,experts:usize,unit:UnitShape,ids:&Self::Value,picks:usize,picks_per_row:usize,id_stride:usize,x:&Self::Value,repeat:usize,caps:Option<[f32;4]>,compact:Option<&Self::Value>)->Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_picked_strided_compact(self,stack,experts,unit,ids,picks,picks_per_row,id_stride,x,repeat,caps,false,compact)
    }
    fn resident_group(&self, matrices: &[(&spark_gpu_probe::driver::DeviceVector<'d>, Range<usize>)], rows: usize, columns: usize, xs: &[&Self::Value]) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_group(self, matrices, rows, columns, xs)
    }
    fn packed_picked(&self, stack: &spark_gpu_probe::driver::DeviceVector<'d>, experts: usize, unit: UnitShape, ids: &Self::Value, x: &Self::Value, shared: bool, token_rows: usize) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_picked(self, stack, experts, unit, ids, x, shared,token_rows == 1,None)
    }
    fn packed_product_runs(&self, runs: &[(&spark_gpu_probe::driver::DeviceVector<'d>, Range<usize>, usize)], unit: UnitShape, x: &Self::Value, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_product_runs(self, runs, unit, x, input, output)
    }
    fn resident_rows(&self, table: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, ids: &[u32]) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_rows(self, table, words, rows, columns, ids)
    }
    fn resident_product_rows_split(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, parts: &[usize], columns: usize, x: &Self::Value, inputs: usize, decode: bool) -> Result<Option<Vec<Self::Value>>> {
        crate::bounded_decoder::Device::resident_product_rows_split(self, weight, words, parts, columns, x, inputs, decode)
    }
    fn resident_product_rows_pair(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x1: &Self::Value, x2: &Self::Value, inputs: usize, decode: bool, round_out: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_product_rows_pair(self, weight, words, rows, columns, x1, x2, inputs, decode, round_out)
    }
    fn resident_group_rows_pair(&self, matrices: &[(&spark_gpu_probe::driver::DeviceVector<'d>, Range<usize>)], rows: usize, columns: usize, pairs: &[(&Self::Value, &Self::Value)], inputs: usize, round_out: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_group_rows_pair(self, matrices, rows, columns, pairs, inputs, round_out)
    }
    fn device_route_group(&self, route: &Self::Value, rows: usize, picks: usize, experts: usize) -> Result<Option<(Self::Value, Self::RoutePlan)>> {
        crate::bounded_decoder::Device::device_route_group(self, route, rows, picks, experts)
    }
    fn device_routed_products(&self, stack: &spark_gpu_probe::driver::DeviceVector<'d>, experts: usize, rows: usize, columns: usize, route: &Self::RoutePlan, x: &Self::Value, down: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::device_routed_products(self, stack, experts, rows, columns, route, x, down)
    }
    fn device_routed_packed_products(&self, stack: &spark_gpu_probe::driver::DeviceVector<'d>, experts: usize, unit: UnitShape, route: &Self::RoutePlan, x: &Self::Value, down: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::device_routed_packed_products(self, stack, experts, unit, route, x, down)
    }
    fn device_routed_packed_gate(&self, stack: &spark_gpu_probe::driver::DeviceVector<'d>, experts: usize, unit: UnitShape, route: &Self::RoutePlan, x: &Self::Value, caps: [f32; 4]) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::device_routed_packed_gate(self, stack, experts, unit, route, x, caps)
    }
    fn device_routed_packed_down_fold(&self, stack: &spark_gpu_probe::driver::DeviceVector<'d>, experts: usize, unit: UnitShape, route: &Self::RoutePlan, x: &Self::Value, weights: &Self::Value, picks: usize) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::device_routed_packed_down_fold(self, stack, experts, unit, route, x, weights, picks)
    }
    fn device_routed_packed_experts(&self, gate: (&spark_gpu_probe::driver::DeviceVector<'d>, usize, UnitShape), down: (&spark_gpu_probe::driver::DeviceVector<'d>, usize, UnitShape), route: &Self::RoutePlan, x: &Self::Value, caps: [f32; 4], weights: &Self::Value, picks: usize) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::device_routed_packed_experts(self, gate, down, route, x, caps, weights, picks)
    }
    fn device_routed_packed_experts_half_in(&self, gate: (&spark_gpu_probe::driver::DeviceVector<'d>, usize, UnitShape), down: (&spark_gpu_probe::driver::DeviceVector<'d>, usize, UnitShape), route: &Self::RoutePlan, x: &Self::Value, caps: [f32; 4], weights: &Self::Value, picks: usize) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::device_routed_packed_experts_form(self, gate, down, route, x, caps, weights, picks, true)
    }
    fn device_routed_packed_experts_close(
        &self, gate: (&spark_gpu_probe::driver::DeviceVector<'d>, usize, UnitShape), down: (&spark_gpu_probe::driver::DeviceVector<'d>, usize, UnitShape),
        route: &Self::RoutePlan, x: &Self::Value, caps: [f32; 4], weights: &Self::Value, picks: usize,
        residual: &Self::Value, fixed: &[f32], ring: &Self::Value, taps: usize, first_slot: usize, bound: f32, half_in: bool,
    ) -> Result<Option<(Self::Value, Self::Value)>> {
        crate::bounded_decoder::Device::device_routed_packed_experts_close(self, gate, down, route, x, caps, weights, picks, residual, fixed, ring, taps, first_slot, bound, half_in)
    }
    fn resident_gate_outer_blend_norm(
        &self, entries: &[(&spark_gpu_probe::driver::DeviceVector<'d>, Range<usize>, &Self::Value, &Self::Value, &Self::Value, Option<&[f32]>)],
        rows: usize, dim: usize, k: usize, eps: f32, half_out: bool,
    ) -> Result<Option<Vec<Self::Value>>> {
        crate::bounded_decoder::Device::resident_gate_outer_blend_norm(self, entries, rows, dim, k, eps, half_out)
    }
    fn resident_product_rows_half_in(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value, inputs: usize) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_product_rows_half_in(self, weight, words, rows, columns, x, inputs)
    }
    fn resident_product_rows_split_half_in(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, parts: &[usize], columns: usize, x: &Self::Value, inputs: usize) -> Result<Option<Vec<Self::Value>>> {
        crate::bounded_decoder::Device::resident_product_rows_split_half_in(self, weight, words, parts, columns, x, inputs)
    }
    fn resident_product_rows_pair_half_in(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x1: &Self::Value, x2: &Self::Value, inputs: usize) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_product_rows_pair_half_in(self, weight, words, rows, columns, x1, x2, inputs)
    }
    fn window_rows_half(&self, a: &crate::bounded_decoder::WindowOne<'_, Self::Value>, first_position: u64) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::record_window_rows_half(self, a, first_position)
    }
    fn mix_rows_normed_half(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: f32,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &[f32], list: &[&Self::Value], stream: &Self::Value,
        post_weight: Option<&[f32]>, post_weight_per_query: bool,
    ) -> Result<Option<(Option<Self::Value>, Option<Self::Value>, Self::Value)>> {
        crate::bounded_decoder::Device::try_mix_rows_normed_half(self, rows, dim, scale, eps, post_eps, transpose, keep_normed, keep_mixed, queries, list, stream, post_weight, post_weight_per_query)
    }
    fn mix_prefill_normed_half(&self, rows: usize, dim: usize, scale: f32, eps: f32, queries: &[f32], list: &[&Self::Value]) -> Result<Option<(Self::Value, Self::Value)>> {
        crate::bounded_decoder::Device::try_mix_prefill_normed_half(self, rows, dim, scale, eps, queries, list)
    }
    fn prefill_msg_meta(&self, rows: usize, dim: usize, scale: f32, eps: f32, queries: &[f32], list: &[&Self::Value]) -> Result<Option<(Self::Value, Self::Value)>> {
        crate::bounded_decoder::Device::try_prefill_msg_meta(self, rows, dim, scale, eps, queries, list)
    }
    fn resident_prefill_gate_fuse(
        &self, sources: &[&Self::Value],
        entries: &[GateFuseEntry<'_, spark_gpu_probe::driver::DeviceVector<'d>, Self::Value>],
        rows: usize, dim: usize, k: usize, eps: f32, half_out: bool,
    ) -> Result<Option<Vec<Self::Value>>> {
        crate::bounded_decoder::Device::resident_prefill_gate_fuse(self, sources, entries, rows, dim, k, eps, half_out)
    }
}

enum Entry {
    /// Host words in the layout of the weight name. A matrix has its rows and columns.
    Host(Vec<f32>, Option<(usize, usize)>),
    /// A whole BF16 matrix of the owner of the weights, by the name of its tensor.
    Resident { name: String, rows: usize, columns: usize },
}
/// The weights of one loaded checkpoint for the driver. `V` is the value type of the owner of the weights.
pub struct LoadedWeights<'a, V> { operands: &'a Operands<V>, entries: BTreeMap<Weight, Entry>,
    /// A matrix that a stack of the operands holds: the stack and its words in the stack.
    stacked: BTreeMap<Weight, (usize, Range<usize>)>, decode:Cell<bool> }

fn size(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| "loop weights: a size exceeds the address space".into()) }
/// Fixed taps in the layout of the arm. `row_is_tap`: the stored tensor is `[tap, channel]`; else `[channel, tap]`.
fn taps_by_lag(stored: &[f32], channels: usize, taps: usize, order: TapOrder, row_is_tap: bool) -> Result<Vec<f32>> {
    if channels.checked_mul(taps) != Some(stored.len()) { return Err("loop weights: stored taps differ from channels x taps".into()); }
    let mut out = vec![0.0; stored.len()];
    for channel in 0..channels {
        for lag in 0..taps {
            let tap = match order { TapOrder::OldestFirst => taps - 1 - lag, TapOrder::NewestFirst => lag };
            out[channel * taps + lag] = if row_is_tap { stored[tap * channels + channel] } else { stored[channel * taps + tap] };
        }
    }
    Ok(out)
}
fn order(taps: ConvTaps) -> TapOrder { match taps { ConvTaps::Fixed { order } | ConvTaps::FixedAndToken { order, .. } => order } }

impl<'a, V> LoadedWeights<'a, V> {
    /// Read the small weights from `files` and name every weight of the driver. `operands` are the loaded tensors.
    pub fn new(checkpoint: &LoopCheckpoint, files: &Files, operands: &'a Operands<V>) -> Result<Self> {
        let d = &checkpoint.description;
        let hidden = size(d.hidden)?;
        let mut entries = BTreeMap::new();
        let host = |r: &Resident<V>| files.words(&r.tensor);
        // A BF16 matrix stays with the owner of the weights. Any other matrix is small: host words.
        let matrix = |r: &Resident<V>| -> Result<Entry> {
            match (r.tensor.dtype, &r.tensor.shape[..]) {
                (Dtype::Bf16, &[rows, columns]) => Ok(Entry::Resident { name: r.tensor.name.clone(), rows: size(rows)?, columns: size(columns)? }),
                (_, &[rows, columns]) => Ok(Entry::Host(host(r)?, Some((size(rows)?, size(columns)?)))),
                _ => Err(format!("loop weights: {} is no matrix", r.tensor.name)),
            }
        };
        let m = operands.model(d)?;
        // A BF16 table stays with the owner of the weights: a step copies its rows there (`stored_rows`), so the host
        // holds no 32-bit copy of it. Any other table is host words.
        let embedding = match (m.embedding.tensor.dtype, &m.embedding.tensor.shape[..]) {
            (Dtype::Bf16, &[_, columns]) if columns == d.hidden && columns % 2 == 0 => matrix(m.embedding)?,
            _ => Entry::Host(host(m.embedding)?, None),
        };
        entries.insert(Weight::Embedding, embedding);
        entries.insert(Weight::EmbeddingNorm, Entry::Host(host(m.embedding_norm)?, None));
        entries.insert(Weight::LastNorm, Entry::Host(host(m.final_norm)?, None));
        entries.insert(Weight::MixOutput, Entry::Host(host(m.mix_output)?, None));
        entries.insert(Weight::Head, matrix(m.head)?);
        if let (Some(first), Some(second)) = (m.message_queries, m.message_kv_queries) {
            // Rows `2j` and `2j + 1` of the first tensor: the query and the expert row of loop layer `j`. Of the
            // second: its key and its value row.
            let (first, second) = (host(first)?, host(second)?);
            if first.len() != second.len() || !first.len().is_multiple_of(2 * hidden) { return Err("loop weights: the two message query tensors differ from two rows a loop layer".into()); }
            let mut merged = Vec::with_capacity(2 * first.len());
            for (qe, kv) in first.chunks(2 * hidden).zip(second.chunks(2 * hidden)) {
                merged.extend_from_slice(&qe[..hidden]); merged.extend_from_slice(kv); merged.extend_from_slice(&qe[hidden..]);
            }
            entries.insert(Weight::Messages, Entry::Host(merged, None));
        }
        for (i, layer) in d.layers.iter().enumerate() {
            let l = operands.layer(d, i as u64)?;
            let a = layer.attention;
            for (weight, r) in [(Weight::Query(i), l.query), (Weight::Key(i), l.key), (Weight::Value(i), l.value), (Weight::Relative(i), l.relative_projection),
                (Weight::Sink(i), l.sink), (Weight::OutputGate(i), l.output_gate), (Weight::Output(i), l.output), (Weight::Router(i), l.router)] { entries.insert(weight, matrix(r)?); }
            for (weight, r) in [(Weight::QueryClip(i), l.query_clip), (Weight::RelativeClip(i), l.relative_clip), (Weight::RelativeTable(i), l.relative_table),
                (Weight::ExpertNorm(i), l.expert_norm), (Weight::RouterBias(i), l.router_bias), (Weight::MixExperts(i), l.mix_experts)] { entries.insert(weight, Entry::Host(host(r)?, None)); }
            if let Some(r) = l.mix_attention { entries.insert(Weight::MixAttention(i), Entry::Host(host(r)?, None)); }
            let (query, key, value) = match l.input_norms { InputNorms::One(n) => (n, n, n), InputNorms::Separate { query, key, value } => (query, key, value) };
            for (weight, r) in [(Weight::QueryNorm(i), query), (Weight::KeyNorm(i), key), (Weight::ValueNorm(i), value)] { entries.insert(weight, Entry::Host(host(r)?, None)); }
            let (channels, taps, heads) = (size(a.kv_heads * a.head_dim)?, size(a.key_value_conv.kernel)?, size(a.kv_heads)?);
            for (fixed_name, made_name, conv) in [(Weight::KeyTaps(i), Weight::KeyTapProjection(i), &l.key_conv), (Weight::ValueTaps(i), Weight::ValueTapProjection(i), &l.value_conv)] {
                match (conv, a.key_value_conv.taps) {
                    (ConvOperands::Fixed(r), ConvTaps::Fixed { order }) => { entries.insert(fixed_name, Entry::Host(taps_by_lag(&host(r)?, channels, taps, order, false)?, None)); }
                    (ConvOperands::Token { fixed, projection }, ConvTaps::FixedAndToken { order, .. }) => {
                        entries.insert(fixed_name, Entry::Host(taps_by_lag(&host(fixed)?, channels, taps, order, true)?, None));
                        // With the newest tap first a stored row `head * taps + tap` is row `head * taps + lag` already:
                        // a BF16 projection stays with the owner of the weights, as stored.
                        if order == TapOrder::NewestFirst && projection.tensor.dtype == Dtype::Bf16 {
                            if projection.tensor.shape != [(heads * taps) as u64, hidden as u64] { return Err(format!("loop weights: {} differs from heads x taps rows", projection.tensor.name)); }
                            entries.insert(made_name, matrix(projection)?);
                            continue;
                        }
                        // A stored row is `head * taps + tap`. The driver reads row `head * taps + lag`.
                        let stored = host(projection)?;
                        if stored.len() != heads * taps * hidden { return Err(format!("loop weights: {} differs from heads x taps rows", projection.tensor.name)); }
                        let mut by_lag = Vec::with_capacity(stored.len());
                        for head in 0..heads {
                            for lag in 0..taps {
                                let tap = match order { TapOrder::OldestFirst => taps - 1 - lag, TapOrder::NewestFirst => lag };
                                by_lag.extend_from_slice(&stored[(head * taps + tap) * hidden..(head * taps + tap + 1) * hidden]);
                            }
                        }
                        entries.insert(made_name, Entry::Host(by_lag, Some((heads * taps, hidden))));
                    }
                    _ => return Err(format!("loop weights: layer {i}: the stored taps differ from the taps of the description")),
                }
            }
            let stream_taps = size(d.conv.kernel)?;
            entries.insert(Weight::ConvAfterAttention(i), Entry::Host(taps_by_lag(&host(l.conv_after_attention)?, hidden, stream_taps, order(d.conv.taps), false)?, None));
            entries.insert(Weight::ConvAfterExperts(i), Entry::Host(taps_by_lag(&host(l.conv_after_experts)?, hidden, stream_taps, order(d.conv.taps), false)?, None));
            if let Some(g) = &l.gates {
                for (message, gate) in [(Message::Query, &g.query), (Message::Key, &g.key), (Message::Value, &g.value), (Message::Expert, &g.expert)] {
                    entries.insert(Weight::GateDown(i, message), matrix(gate.down)?);
                    entries.insert(Weight::GateForget(i, message), matrix(gate.forget)?);
                    entries.insert(Weight::GateGain(i, message), matrix(gate.gain)?);
                }
            }
        }
        let by_name = entries.iter().filter_map(|(w, e)| match e { Entry::Resident { name, .. } => Some((name.as_str(), *w)), Entry::Host(..) => None }).collect::<BTreeMap<_, _>>();
        let by_name = &by_name;
        let stacked = operands.stacks().iter().enumerate()
            .flat_map(|(i, stack)| stack.members.iter().filter_map(move |(name, words)| by_name.get(name.as_str()).map(|w| (*w, (i, words.clone()))))).collect();
        Ok(Self { operands, entries, stacked, decode:Cell::new(true) })
    }
    /// Where `weights` are two or more matrices that one stack holds one after another in this order, with the same
    /// columns: the stack value, the words of all of them, the rows of each and the columns. None otherwise.
    fn run(&self, weights: &[Weight]) -> Result<Option<(&V, Range<usize>, Vec<usize>, usize)>> {
        let mut found = Vec::with_capacity(weights.len());
        for weight in weights {
            let (Some((stack, words)), Some(Entry::Resident { rows, columns, .. })) = (self.stacked.get(weight), self.entries.get(weight)) else { return Ok(None) };
            found.push((*stack, words.clone(), *rows, *columns));
        }
        if found.len() < 2 || found.windows(2).any(|p| p[0].0 != p[1].0 || p[0].1.end != p[1].1.start || p[0].3 != p[1].3) { return Ok(None); }
        let ((stack, first, _, columns), last) = (&found[0], &found[found.len() - 1].1);
        let value = &self.operands.stacks().get(*stack).ok_or("loop weights: a stack is missing")?.value;
        Ok(Some((value, first.start..last.end, found.iter().map(|f| f.2).collect(), *columns)))
    }
    /// Every weight name but the experts, and true for a weight that is held as host words.
    pub fn names(&self) -> impl Iterator<Item = (Weight, bool)> + '_ { self.entries.iter().map(|(w, e)| (*w, matches!(e, Entry::Host(..)))) }
    /// The rows and columns of a weight that is read through a product. None for a weight that is read as words.
    pub fn product_shape(&self, weight: Weight) -> Result<Option<(usize, usize)>> {
        Ok(match (self.matrix(weight)?, self.entries.get(&weight)) {
            (Some((_, _, rows, columns)), _) => Some((rows, columns)),
            (None, Some(Entry::Host(_, shape))) => *shape,
            (None, _) => return Err(format!("loop weights: the checkpoint holds no weight {weight:?}")),
        })
    }
    /// The value, the word range, the rows and the columns of a matrix that stays with the owner of the weights.
    /// None for a weight that is held as host words.
    pub fn matrix(&self, weight: Weight) -> Result<Option<(&V, Range<usize>, usize, usize)>> {
        let expert = |layer: usize, proj: Proj, expert: usize| -> Result<Option<(&V, Range<usize>, usize, usize)>> {
            let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
            let m = self.operands.expert(layer, proj, expert as u64)?;
            Ok(Some((m.value, m.words, size(m.rows)?, size(m.columns)?)))
        };
        match weight {
            Weight::GateUp(layer, e) => expert(layer, Proj::GateUp, e),
            Weight::Down(layer, e) => expert(layer, Proj::Down, e),
            _ => match self.entries.get(&weight) {
                Some(Entry::Resident { name, rows, columns }) => {
                    let words = rows.checked_mul(*columns).filter(|n| n.is_multiple_of(2)).ok_or_else(|| format!("loop weights: {name} does not fill whole pairs of BF16 values"))? / 2;
                    Ok(Some((&self.operands.named(name)?.value, 0..words, *rows, *columns)))
                }
                _ => Ok(None),
            },
        }
    }
    /// The host words of a weight that is read as words (`Weights::words`).
    pub fn host_words(&self, weight: Weight) -> Result<&[f32]> {
        match self.entries.get(&weight) {
            Some(Entry::Host(words, _)) => Ok(words),
            Some(Entry::Resident { .. }) => Err(format!("loop weights: {weight:?} stays with the owner of the weights: it is read through a product")),
            None if matches!(weight, Weight::GateUp(..) | Weight::Down(..)) => Err(format!("loop weights: {weight:?} stays with the owner of the weights: it is read through a product")),
            None => Err(format!("loop weights: the checkpoint holds no weight {weight:?}")),
        }
    }
    /// `Weights::product_runs` from the packed units: every run an expert of one projection of one layer whose packed
    /// unit has the columns `columns`. None where a run is no such expert, or the compute has no packed rows body.
    fn packed_runs<C: ResidentProduct<V>>(&self, compute: &C, runs: &[(Weight, usize)], x: &C::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<Option<C::Value>> {
        let (layer, proj) = match runs.first() { Some(&(Weight::GateUp(layer, _), _)) => (layer, Proj::GateUp), Some(&(Weight::Down(layer, _), _)) => (layer, Proj::Down), _ => return Ok(None) };
        let layer_word = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let (mut units, mut shape): (Vec<(&V, Range<usize>, usize)>, Option<UnitShape>) = (Vec::with_capacity(runs.len()), None);
        for &(weight, count) in runs {
            let expert = match (weight, proj) { (Weight::GateUp(l, e), Proj::GateUp) | (Weight::Down(l, e), Proj::Down) if l == layer => e, _ => return Ok(None) };
            let Some(packed) = self.operands.packed_expert(layer_word, proj, expert as u64)? else { return Ok(None) };
            if packed.unit.columns != columns || shape.is_some_and(|s| s != packed.unit) || units.first().is_some_and(|first| !std::ptr::eq(first.0, packed.value)) { return Ok(None); }
            shape = Some(packed.unit);
            units.push((packed.value, packed.words, count));
        }
        match shape { Some(unit) => compute.packed_product_runs(&units, unit, x, input, output), None => Ok(None) }
    }
    /// True when `weight` is a resident BF16 matrix whose column count is a multiple of `TC_STAGE` (64).
    fn is_resident_tc(&self, weight: Weight) -> bool {
        matches!(self.entries.get(&weight), Some(Entry::Resident { rows, columns, .. })
            if *rows <= spark_gpu_probe::tc_rows::TC_ROWS_MAX && spark_gpu_probe::tc_rows::tc_shape_admitted(*rows, *columns, 1))
    }
    /// Bytes of the host words. They are host memory beside the device bytes of the ledger.
    pub fn host_bytes(&self) -> usize { self.entries.values().map(|e| match e { Entry::Host(w, _) => std::mem::size_of_val(&w[..]), Entry::Resident { .. } => 0 }).sum() }
}
impl<C: ResidentProduct<V>, V> Weights<C> for LoadedWeights<'_, V> {
    type RoutePlan = C::RoutePlan;
    fn arithmetic_phase(&self,decode:bool)->bool {self.decode.replace(decode)}
    fn words(&self, weight: Weight) -> Result<&[f32]> { self.host_words(weight) }
    fn stored_rows(&self, compute: &C, weight: Weight, tokens: &[u32]) -> Result<Option<C::Value>> {
        let Some(Entry::Resident { name, rows, columns }) = self.entries.get(&weight) else { return Ok(None) };
        let words = rows.checked_mul(*columns).filter(|n| n.is_multiple_of(2)).ok_or_else(|| format!("loop weights: {name} does not fill whole pairs of BF16 values"))? / 2;
        // The table is not held as host words: no copy of the device is a refusal, not a fallback.
        compute.resident_rows(&self.operands.named(name)?.value, 0..words, *rows, *columns, tokens)?
            .map(Some).ok_or_else(|| format!("loop weights: {weight:?} stays on the device and the compute gave no copy of its rows now"))
    }
    fn product(&self, compute: &C, weight: Weight, x: &C::Value) -> Result<C::Value> {
        // An expert: the packed stack first. A compute without a packed product for the shape gives None.
        if let Weight::GateUp(layer, expert) | Weight::Down(layer, expert) = weight {
            let (layer, proj) = (u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?, if matches!(weight, Weight::Down(..)) { Proj::Down } else { Proj::GateUp });
            if let Some(packed) = self.operands.packed_expert(layer, proj, expert as u64)?
                && let Some(out) = compute.packed_product(packed.value, packed.words, packed.unit, x)? { return Ok(out); }
        }
        if let Some((value, words, rows, columns)) = self.matrix(weight)? { return compute.resident_product_rows(value, words, rows, columns, x, 1,self.decode.get()); }
        match self.entries.get(&weight) {
            Some(Entry::Host(words, _)) => chain_product(compute, words, x),
            _ => Err(format!("loop weights: the checkpoint holds no weight {weight:?}")),
        }
    }
    /// Matrices that stay with the owner of the weights, all of one shape, go to the compute in one call; any other case
    /// (a host matrix, an expert, two shapes) gives None and the caller takes one product a matrix.
    fn group_products(&self, compute: &C, weights: &[Weight], xs: &[&C::Value]) -> Result<Option<C::Value>> {
        if !self.decode.get() || weights.len() < 2 || xs.len() != weights.len() { return Ok(None); }
        let mut matrices = Vec::with_capacity(weights.len());
        let mut shape = None;
        for &weight in weights {
            if matches!(weight, Weight::GateUp(..) | Weight::Down(..)) { return Ok(None); }
            let Some((value, words, rows, columns)) = self.matrix(weight)? else { return Ok(None) };
            if shape.is_some_and(|s| s != (rows, columns)) { return Ok(None); }
            shape = Some((rows, columns));
            matrices.push((value, words));
        }
        let Some((rows, columns)) = shape else { return Ok(None) };
        if xs.iter().any(|x| compute.words(x) != columns) { return Err("loop weights: an input of a grouped product differs from the columns".into()); }
        compute.resident_group(&matrices, rows, columns, xs)
    }
    /// A matrix that stays with the owner of the weights goes to the compute with all its input rows. A matrix of
    /// host words runs one chain product a row.
    fn product_rows(&self, compute: &C, weight: Weight, x: &C::Value, rows: usize) -> Result<C::Value> {
        if rows == 0 || !compute.words(x).is_multiple_of(rows) { return Err("a product for rows holds no whole count of input rows".into()); }
        // One row is `product`: an expert reads its packed stack there.
        if rows == 1 && (self.decode.get() || matches!(weight,Weight::GateUp(..)|Weight::Down(..))) { return self.product(compute, weight, x); }
        if let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? { return compute.resident_product_rows(value, words, matrix_rows, columns, x, rows,self.decode.get()); }
        let columns = compute.words(x) / rows;
        let outs = (0..rows).map(|row| self.product(compute, weight, &compute.range(x, row * columns, columns)?)).collect::<Result<Vec<_>>>()?;
        compute.join(&outs.iter().collect::<Vec<_>>())
    }
    /// The expert runs go to the compute with no gather: from the packed units first, where every run is a unit of one
    /// packed stack of one shape and the compute has the packed rows tile for it; else from the widened matrices when
    /// every run is a matrix that stays with the owner of the weights, with one shape. Any other case, or a compute
    /// with neither body, takes the body with gathers. All forms give the same words.
    fn product_runs(&self, compute: &C, runs: &[(Weight, usize)], x: &C::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<C::Value> {
        check_runs(compute.words(x), columns, runs, input, output)?;
        if let Some(value) = self.packed_runs(compute, runs, x, columns, input, output)? { return Ok(value); }
        let mut matrices = Vec::with_capacity(runs.len());
        for &(weight, count) in runs {
            match self.matrix(weight)? {
                Some((value, words, rows, matrix_columns)) if matrix_columns == columns && matrices.first().is_none_or(|first: &(&V, Range<usize>, usize, usize)| first.2 == rows) => matrices.push((value, words, rows, count)),
                _ => return gathered_runs(self, compute, runs, x, columns, input, output),
            }
        }
        match compute.resident_product_runs(&matrices, x, columns, input, output)? {
            Some(value) => Ok(value),
            None => gathered_runs(self, compute, runs, x, columns, input, output),
        }
    }
    /// Matrices that a stack holds one after another run as one product over all their rows. A row is one chain over
    /// the same input, so each part holds the words of its own product.
    fn stack_products(&self, compute: &C, weights: &[Weight], x: &C::Value) -> Result<Vec<C::Value>> {
        let Some((value, words, rows, columns)) = self.run(weights)? else { return weights.iter().map(|&weight| self.product(compute, weight, x)).collect() };
        let y = compute.resident_product_rows(value, words, rows.iter().sum(), columns, x, 1,self.decode.get())?;
        let mut start = 0;
        rows.iter().map(|&n| { let part = compute.range(&y, start, n); start += n; part }).collect()
    }
    fn stack_product_rows(&self, compute: &C, weights: &[Weight], x: &C::Value, rows: usize) -> Result<Option<C::Value>> {
        let Some((value, words, parts, columns)) = self.run(weights)? else { return Ok(None) };
        if rows == 0 || rows.checked_mul(columns) != Some(compute.words(x)) { return Err("loop weights: the input of a stacked product differs from its rows".into()); }
        compute.resident_product_rows(value, words, parts.iter().sum(), columns, x, rows,self.decode.get()).map(Some)
    }
    fn stack_products_rows(&self, compute: &C, weights: &[Weight], x: &C::Value, rows: usize) -> Result<Option<Vec<C::Value>>> {
        let Some((value, words, parts, columns)) = self.run(weights)? else { return Ok(None) };
        if rows == 0 || rows.checked_mul(columns) != Some(compute.words(x)) { return Err("loop weights: the input of a stacked product differs from its rows".into()); }
        compute.resident_product_rows_split(value, words, &parts, columns, x, rows, self.decode.get())
    }
    fn product_rows_pair(&self, compute: &C, weight: Weight, x1: &C::Value, x2: &C::Value, rows: usize) -> Result<Option<C::Value>> {
        let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? else { return Ok(None); };
        compute.resident_product_rows_pair(value, words, matrix_rows, columns, x1, x2, rows, self.decode.get(), false)
    }
    fn product_rows_pair_rounded(&self, compute: &C, weight: Weight, x1: &C::Value, x2: &C::Value, rows: usize) -> Result<Option<C::Value>> {
        let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? else { return Ok(None); };
        compute.resident_product_rows_pair(value, words, matrix_rows, columns, x1, x2, rows, self.decode.get(), true)
    }
    fn group_product_rows_pair_rounded(&self, compute: &C, weights: &[Weight], pairs: &[(&C::Value, &C::Value)], rows: usize) -> Result<Option<C::Value>> {
        if !self.decode.get() || weights.len() < 2 || weights.len() != pairs.len() { return Ok(None); }
        let mut matrices = Vec::with_capacity(weights.len());
        let mut shape = None;
        for &weight in weights {
            if matches!(weight, Weight::GateUp(..) | Weight::Down(..)) { return Ok(None); }
            let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? else { return Ok(None) };
            if shape.is_some_and(|s| s != (matrix_rows, columns)) { return Ok(None); }
            shape = Some((matrix_rows, columns));
            matrices.push((value, words));
        }
        let Some((matrix_rows, columns)) = shape else { return Ok(None) };
        if !columns.is_multiple_of(2) || pairs.iter().any(|(a, b)| compute.words(a) != rows * (columns / 2) || compute.words(b) != rows * (columns / 2)) { return Ok(None); }
        compute.resident_group_rows_pair(&matrices, matrix_rows, columns, pairs, rows, true)
    }
    fn device_route_group(&self, compute: &C, layer: usize, route: &C::Value, rows: usize, picks: usize, experts: usize) -> Result<Option<(C::Value, Self::RoutePlan)>> {
        // Admit device routing when both projections have either a packed stack supported by
        // `PLAIN_PACKED_TC_ROWS` or a widened BF16 stack supported by `PLAIN_TC_ROWS` Mode 1.
        let layer = u32::try_from(layer).map_err(|_| "loop weights: layer exceeds its range")?;
        for proj in [Proj::GateUp, Proj::Down] {
            if let Some((_, n_experts, unit)) = self.operands.packed_experts(layer, proj)? {
                let wide = unit.codes == engine_model::plan::loop_load::ExpertCodes::Mxfp8;
                let pairs = unit.order.is_some();
                if n_experts == experts && spark_gpu_probe::tc_rows::tc_packed_moe_shape_admitted(unit.rows, unit.columns, wide, pairs) {
                    continue;
                }
            }
            let role = super::StoredRole::Expert { proj, part: super::ExpertPart::Codes };
            if self.operands.find(super::Place::Layer(layer), role)?.is_none() {
                self.operands.packed_stack(layer, proj)?.ok_or("loop weights: expert stack missing")?;
                return Ok(None);
            }
        }
        compute.device_route_group(route, rows, picks, experts)
    }
    fn device_routed_products(&self, compute: &C, first: Weight, route: &Self::RoutePlan, x: &C::Value) -> Result<Option<C::Value>> {
        let (layer, proj) = match first { Weight::GateUp(layer, 0) => (layer, Proj::GateUp), Weight::Down(layer, 0) => (layer, Proj::Down), _ => return Ok(None) };
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let down = proj == Proj::Down;
        if let Some((stack, experts, unit)) = self.operands.packed_experts(layer, proj)?
            && let Some(out) = compute.device_routed_packed_products(stack, experts, unit, route, x, down)?
        {
            return Ok(Some(out));
        }
        let (stack, experts, rows, columns) = self.operands.expert_stack(layer, proj)?;
        compute.device_routed_products(stack, experts, rows, columns, route, x, down)
    }
    fn device_routed_gate(&self, compute: &C, layer: usize, route: &Self::RoutePlan, x: &C::Value, caps: [f32; 4]) -> Result<Option<C::Value>> {
        // Only the packed gate-and-up stack (mixed gate and up rows) has the gated launch; else the two launches.
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some((stack, experts, unit)) = self.operands.packed_experts(layer, Proj::GateUp)? else { return Ok(None) };
        compute.device_routed_packed_gate(stack, experts, unit, route, x, caps)
    }
    fn device_routed_down_fold(&self, compute: &C, layer: usize, route: &Self::RoutePlan, x: &C::Value, weights: &C::Value, picks: usize) -> Result<Option<C::Value>> {
        // Only the packed down stack has the half launch; else the product and the fold of before.
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some((stack, experts, unit)) = self.operands.packed_experts(layer, Proj::Down)? else { return Ok(None) };
        compute.device_routed_packed_down_fold(stack, experts, unit, route, x, weights, picks)
    }
    fn device_routed_experts(&self, compute: &C, layer: usize, route: &Self::RoutePlan, x: &C::Value, caps: [f32; 4], weights: &C::Value, picks: usize) -> Result<Option<C::Value>> {
        // Only two packed stacks (gate-and-up with mixed rows, down) have the joined launches; else the calls of before.
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some(gate) = self.operands.packed_experts(layer, Proj::GateUp)? else { return Ok(None) };
        let Some(down) = self.operands.packed_experts(layer, Proj::Down)? else { return Ok(None) };
        compute.device_routed_packed_experts(gate, down, route, x, caps, weights, picks)
    }
    fn device_routed_experts_half_in(&self, compute: &C, layer: usize, route: &Self::RoutePlan, x: &C::Value, caps: [f32; 4], weights: &C::Value, picks: usize) -> Result<Option<C::Value>> {
        if self.decode.get() && (picks == 0 || compute.words(weights) / picks <= spark_gpu_probe::few_product::FEW_MAX) { return Ok(None); }
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some(gate) = self.operands.packed_experts(layer, Proj::GateUp)? else { return Ok(None) };
        let Some(down) = self.operands.packed_experts(layer, Proj::Down)? else { return Ok(None) };
        compute.device_routed_packed_experts_half_in(gate, down, route, x, caps, weights, picks)
    }
    fn device_routed_experts_close(
        &self, compute: &C, layer: usize, route: &Self::RoutePlan, x: &C::Value, caps: [f32; 4],
        weights: &C::Value, picks: usize, residual: &C::Value, fixed: &[f32], ring: &C::Value,
        taps: usize, first_slot: usize, bound: f32,
    ) -> Result<Option<(C::Value, C::Value)>> {
        if self.decode.get() { return Ok(None); }
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some(gate) = self.operands.packed_experts(layer, Proj::GateUp)? else { return Ok(None) };
        let Some(down) = self.operands.packed_experts(layer, Proj::Down)? else { return Ok(None) };
        compute.device_routed_packed_experts_close(gate, down, route, x, caps, weights, picks, residual, fixed, ring, taps, first_slot, bound, false)
    }
    fn device_routed_experts_close_half_in(
        &self, compute: &C, layer: usize, route: &Self::RoutePlan, x: &C::Value, caps: [f32; 4],
        weights: &C::Value, picks: usize, residual: &C::Value, fixed: &[f32], ring: &C::Value,
        taps: usize, first_slot: usize, bound: f32,
    ) -> Result<Option<(C::Value, C::Value)>> {
        if self.decode.get() { return Ok(None); }
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some(gate) = self.operands.packed_experts(layer, Proj::GateUp)? else { return Ok(None) };
        let Some(down) = self.operands.packed_experts(layer, Proj::Down)? else { return Ok(None) };
        compute.device_routed_packed_experts_close(gate, down, route, x, caps, weights, picks, residual, fixed, ring, taps, first_slot, bound, true)
    }
    fn gate_outer_blend_norm(
        &self, compute: &C, layer: usize,
        entries: &[(Message, &C::Value, &C::Value, &C::Value, Option<&[f32]>)],
        rows: usize, eps: f32, half_out: bool,
    ) -> Result<Option<Vec<C::Value>>> {
        if (self.decode.get() && rows <= spark_gpu_probe::few_product::FEW_MAX)
            || entries.is_empty() || entries.len() > spark_gpu_probe::tc_rows::GATE_OUTER_GATES_MAX
        {
            return Ok(None);
        }
        let mut resolved = Vec::with_capacity(entries.len());
        let mut shape = None;
        for &(which, inner, kept, msg, norm_w) in entries {
            let Some((value, words, parts, k)) = self.run(&[Weight::GateForget(layer, which), Weight::GateGain(layer, which)])? else {
                return Ok(None);
            };
            if parts.len() != 2 || parts[0] != parts[1] { return Ok(None); }
            let dim = parts[0];
            if shape.is_some_and(|s| s != (dim, k)) { return Ok(None); }
            shape = Some((dim, k));
            resolved.push((value, words, inner, kept, msg, norm_w));
        }
        let Some((dim, k)) = shape else { return Ok(None); };
        compute.resident_gate_outer_blend_norm(&resolved, rows, dim, k, eps, half_out)
    }
    fn prefers_half_attention(&self, layer: usize) -> bool {
        let base = [Weight::Query(layer), Weight::Relative(layer), Weight::Sink(layer), Weight::OutputGate(layer), Weight::Key(layer), Weight::Value(layer)];
        if !base.into_iter().all(|w| self.is_resident_tc(w)) { return false; }
        for tap in [Weight::KeyTapProjection(layer), Weight::ValueTapProjection(layer)] {
            if self.entries.contains_key(&tap) && !self.is_resident_tc(tap) { return false; }
        }
        true
    }
    fn prefers_half_experts(&self, layer: usize) -> bool {
        if !self.is_resident_tc(Weight::Router(layer)) { return false; }
        let Ok(layer_u32) = u32::try_from(layer) else { return false };
        for proj in [Proj::GateUp, Proj::Down] {
            let Ok(Some((_, _, unit))) = self.operands.packed_experts(layer_u32, proj) else { return false };
            let wide = unit.codes == engine_model::plan::loop_load::ExpertCodes::Mxfp8;
            let pairs = unit.order.is_some();
            if !spark_gpu_probe::tc_rows::tc_packed_moe_shape_admitted(unit.rows, unit.columns, wide, pairs) { return false; }
        }
        true
    }
    fn prefers_half_messages(&self, start: usize, end: usize) -> bool {
        if start >= end { return false; }
        for layer in start..end {
            for which in [Message::Query, Message::Key, Message::Value, Message::Expert] {
                let down = Weight::GateDown(layer, which);
                let ok_down = matches!(self.entries.get(&down), Some(Entry::Resident { rows, columns, .. })
                    if spark_gpu_probe::tc_rows::tc_shape_admitted(*rows, *columns, 1)
                        && (*columns / 2).is_multiple_of(spark_gpu_probe::tc_rows::TC_BK));
                if !ok_down { return false; }
                let Ok(Some((_, _, parts, k))) = self.run(&[Weight::GateForget(layer, which), Weight::GateGain(layer, which)]) else { return false };
                if parts.len() != 2 || parts[0] != parts[1] || !spark_gpu_probe::tc_rows::gate_outer_admitted(1, parts[0], k, 1, spark_gpu_probe::tc_rows::TC_THREADS) {
                    return false;
                }
            }
        }
        true
    }
    fn prefers_lazy_messages(&self, start: usize, end: usize) -> bool {
        use spark_gpu_probe::prefill_gate;
        if self.decode.get() || start >= end { return false; }
        for layer in start..end {
            for which in [Message::Query, Message::Key, Message::Value, Message::Expert] {
                let down = Weight::GateDown(layer, which);
                let Some(Entry::Resident { rows: k, columns, .. }) = self.entries.get(&down) else { return false };
                let Ok(Some((_, _, parts, ko))) = self.run(&[Weight::GateForget(layer, which), Weight::GateGain(layer, which)]) else { return false };
                if parts.len() != 2 || parts[0] != parts[1] || *k != ko || *columns != 2 * parts[0]
                    || !prefill_gate::full_admitted(1, parts[0], *k, 1, 1, prefill_gate::PG_FULL_THREADS)
                {
                    return false;
                }
            }
        }
        true
    }
    fn prefill_msg_meta(&self, compute: &C, rows: usize, dim: usize, scale: f32, eps: f32, queries: &[f32], list: &[&C::Value]) -> Result<Option<(C::Value, C::Value)>> {
        if self.decode.get() { return Ok(None); }
        compute.prefill_msg_meta(rows, dim, scale, eps, queries, list)
    }
    fn prefill_gate_fuse(
        &self, compute: &C, layer: usize, sources: &[&C::Value],
        entries: &[(Message, &C::Value, &C::Value, &C::Value, &C::Value, Option<&[f32]>)],
        rows: usize, eps: f32, half_out: bool,
    ) -> Result<Option<Vec<C::Value>>> {
        if self.decode.get() || entries.is_empty() || entries.len() > spark_gpu_probe::prefill_gate::PG_GATES_MAX {
            return Ok(None);
        }
        let mut resolved = Vec::with_capacity(entries.len());
        let mut shape = None;
        for &(which, kept_val, kept_half, mw, mn, norm_w) in entries {
            let Some((wd_val, wd_words, kd, cols)) = self.matrix(Weight::GateDown(layer, which))? else { return Ok(None); };
            let Some((wo_val, wo_words, parts, ko)) = self.run(&[Weight::GateForget(layer, which), Weight::GateGain(layer, which)])? else {
                return Ok(None);
            };
            if parts.len() != 2 || parts[0] != parts[1] || kd != ko || cols != 2 * parts[0] { return Ok(None); }
            let dim = parts[0];
            if shape.is_some_and(|s| s != (dim, ko)) { return Ok(None); }
            shape = Some((dim, ko));
            resolved.push((wd_val, wd_words, wo_val, wo_words, kept_val, kept_half, mw, mn, norm_w));
        }
        let Some((dim, k)) = shape else { return Ok(None); };
        compute.resident_prefill_gate_fuse(sources, &resolved, rows, dim, k, eps, half_out)
    }
    fn prefers_half_output(&self, layer: usize) -> bool {
        !self.decode.get() && self.is_resident_tc(Weight::Output(layer))
    }
    fn window_rows_half(&self, compute: &C, a: &crate::bounded_decoder::WindowOne<'_, C::Value>, first_position: u64) -> Result<Option<C::Value>> {
        if self.decode.get() { return Ok(None); }
        compute.window_rows_half(a, first_position)
    }
    fn mix_rows_normed_half(
        &self, compute: &C, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: f32,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &[f32], list: &[&C::Value], stream: &C::Value,
        post_weight: Option<&[f32]>, post_weight_per_query: bool,
    ) -> Result<Option<(Option<C::Value>, Option<C::Value>, C::Value)>> {
        if self.decode.get() && rows <= spark_gpu_probe::few_product::FEW_MAX { return Ok(None); }
        compute.mix_rows_normed_half(rows, dim, scale, eps, post_eps, transpose, keep_normed, keep_mixed, queries, list, stream, post_weight, post_weight_per_query)
    }
    fn mix_prefill_normed_half(&self, compute: &C, rows: usize, dim: usize, scale: f32, eps: f32, queries: &[f32], list: &[&C::Value]) -> Result<Option<(C::Value, C::Value)>> {
        if self.decode.get() { return Ok(None); }
        compute.mix_prefill_normed_half(rows, dim, scale, eps, queries, list)
    }
    fn product_rows_half_in(&self, compute: &C, weight: Weight, x: &C::Value, rows: usize) -> Result<Option<C::Value>> {
        if self.decode.get() && rows <= spark_gpu_probe::few_product::FEW_MAX { return Ok(None); }
        let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? else { return Ok(None) };
        compute.resident_product_rows_half_in(value, words, matrix_rows, columns, x, rows)
    }
    fn stack_products_rows_half_in(&self, compute: &C, weights: &[Weight], x: &C::Value, rows: usize) -> Result<Option<Vec<C::Value>>> {
        if self.decode.get() && rows <= spark_gpu_probe::few_product::FEW_MAX { return Ok(None); }
        let Some((value, words, parts, columns)) = self.run(weights)? else { return Ok(None) };
        if rows == 0 || !columns.is_multiple_of(2) || rows.checked_mul(columns / 2) != Some(compute.words(x)) {
            return Err("loop weights: the half input of a stacked product differs from its rows".into());
        }
        compute.resident_product_rows_split_half_in(value, words, &parts, columns, x, rows)
    }
    fn product_rows_pair_half_in(&self, compute: &C, weight: Weight, x1: &C::Value, x2: &C::Value, rows: usize) -> Result<Option<C::Value>> {
        if self.decode.get() && rows <= spark_gpu_probe::few_product::FEW_MAX { return Ok(None); }
        let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? else { return Ok(None) };
        compute.resident_product_rows_pair_half_in(value, words, matrix_rows, columns, x1, x2, rows)
    }
    fn products(&self, compute: &C, weights: &[Weight], x: &C::Value, shared: bool) -> Result<Option<C::Value>> {
        // Experts of one projection of one layer, from one packed stack.
        let (layer, proj) = match weights.first() {
            Some(&Weight::GateUp(layer, _)) => (layer, Proj::GateUp),
            Some(&Weight::Down(layer, _)) => (layer, Proj::Down),
            _ => return Ok(None),
        };
        let layer_word = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let (mut value, mut units, mut shape): (Option<&V>, _, Option<UnitShape>) = (None, Vec::with_capacity(weights.len()), None);
        for &weight in weights {
            let expert = match (weight, proj) { (Weight::GateUp(l, e), Proj::GateUp) | (Weight::Down(l, e), Proj::Down) if l == layer => e, _ => return Ok(None) };
            let Some(packed) = self.operands.packed_expert(layer_word, proj, expert as u64)? else { return Ok(None); };
            if value.is_some_and(|v| !std::ptr::eq(v, packed.value)) || shape.is_some_and(|s| s != packed.unit) { return Ok(None); }
            (value, shape) = (Some(packed.value), Some(packed.unit));
            units.push(packed.words);
        }
        match (value, shape) { (Some(value), Some(unit)) => compute.packed_group(value, &units, unit, x, shared), _ => Ok(None) }
    }
    fn picked_gate(&self,compute:&C,layer:usize,ids:&C::Value,x:&C::Value,shared:bool,caps:[f32;4])->Result<Option<C::Value>> {
        let layer=u32::try_from(layer).map_err(|_|"loop weights: layer exceeds its range")?;
        let Some((stack,experts,unit))=self.operands.packed_experts(layer,Proj::GateUp)? else {return Ok(None)};
        compute.packed_gate(stack,experts,unit,ids,x,shared,caps)
    }
    fn picked_products(&self, compute: &C, first: Weight, ids: &C::Value, x: &C::Value, shared: bool, token_rows: usize) -> Result<Option<C::Value>> {
        // The whole packed stack of the projection and layer of `first`; the compute reads the picked numbers.
        let (layer, proj) = match first { Weight::GateUp(layer, 0) => (layer, Proj::GateUp), Weight::Down(layer, 0) => (layer, Proj::Down), _ => return Ok(None) };
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some((stack, experts, unit)) = self.operands.packed_experts(layer, proj)? else { return Ok(None); };
        compute.packed_picked(stack, experts, unit, ids, x, shared, token_rows)
    }
    fn picked_products_strided(&self, compute: &C, first: Weight, route: &C::Value, rows: usize, picks: usize, id_stride: usize, x: &C::Value, repeat: usize, caps: Option<[f32; 4]>) -> Result<Option<C::Value>> {
        self.picked_products_strided_compact(compute, first, route, rows, picks, id_stride, x, repeat, caps, None)
    }
    fn picked_products_strided_compact(&self, compute: &C, first: Weight, route: &C::Value, rows: usize, picks: usize, id_stride: usize, x: &C::Value, repeat: usize, caps: Option<[f32; 4]>, compact: Option<&C::Value>) -> Result<Option<C::Value>> {
        let (layer, proj) = match first { Weight::GateUp(layer, 0) => (layer, Proj::GateUp), Weight::Down(layer, 0) => (layer, Proj::Down), _ => return Ok(None) };
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some((stack, experts, unit)) = self.operands.packed_experts(layer, proj)? else { return Ok(None); };
        compute.packed_picked_strided_compact(stack, experts, unit, route, rows * picks, picks, id_stride, x, repeat, caps, compact)
    }
}
#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "packed_tests.rs"]
mod packed_tests;
