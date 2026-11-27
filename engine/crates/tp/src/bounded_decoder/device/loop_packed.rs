//! The product of one packed expert matrix with one input row, for the driver of the block family with a loop.
//!
//! The matrix is a range of words of a resident vector: the unit of one expert in a packed stack
//! (`engine_model::plan::loop_packed_unit`: the stored codes, the stored block scales and a weight table). The
//! output is `rows` 32-bit words, gate rows first: word for word the output of `resident_product` on the same
//! matrix widened to BF16 (`Backend::project_packed_into`, kernel body `plain_packed`).
//!
//! The packed kernel takes whole stages of columns, 16-byte aligned operands and recorded launches. For any other
//! call this file gives None and the caller reads the widened matrix, so no shape and no state is refused here.
//!
//! The expert runs of a step of rows (`packed_product_runs`, `Weights::product_runs`): the packed rows tile
//! (`Backend::project_packed_rows_grouped_into`, kernel body `plain_packed_rows`) reads each run's unit and gives the
//! words of the BF16 rows tile on the widened matrix, gate rows first. So a step of many rows needs no widened stack.
use super::*;
use engine_model::plan::{loop_load::ExpertCodes, loop_packed_unit::UnitShape};

impl<'b,'d> Device<'b,'d> {
    pub(crate) fn packed_product(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,unit:UnitShape,x:&Value<'b,'d>)->Result<Option<Value<'b,'d>>> {
        let (wide,pairs)=(unit.codes==ExpertCodes::Mxfp8,unit.order.is_some());
        if !self.backend.packed_product_ready(unit.rows,unit.columns,wide,pairs) {return Ok(None);}
        // The unit of the load plan is the unit that the kernel reads: the same parts at the same bytes.
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,spark_gpu_probe::packed_product::layout(unit.rows,unit.columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && words.len().checked_mul(4)==Some(plan.bytes),"packed product: the unit of the load plan differs from the unit of the kernel")?;
        need(x.words==unit.columns,"packed product: the input differs from the columns of the weight")?;
        self.ready(x)?;
        let matrix=weight.resident_range(words)?;
        if !matrix.device_address().is_multiple_of(16) || !x.vector.device_address().is_multiple_of(16) {return Ok(None);}
        let mut output=self.backend.vector(unit.rows)?;
        self.backend.project_packed_into(unit.rows,unit.columns,wide,pairs,&x.vector,&matrix,&mut output)?;
        self.value(output,unit.rows).map(Some)
    }
    /// The packed products of several matrices of one shape in ONE launch (`Backend::project_packed_group_into`):
    /// `units` are word ranges of `weight`, the units of matrices of the shape `unit`. `x` is the input of all
    /// (`shared`) or the inputs back to back. The result: the products back to back, each the words of `packed_product`
    /// on its matrix. None where `packed_product` would give None for one of them.
    pub(crate) fn packed_group(&self,weight:&DeviceVector<'d>,units:&[std::ops::Range<usize>],unit:UnitShape,x:&Value<'b,'d>,shared:bool)->Result<Option<Value<'b,'d>>> {
        let (wide,pairs,picks)=(unit.codes==ExpertCodes::Mxfp8,unit.order.is_some(),units.len());
        if picks==0 || picks>spark_gpu_probe::packed_product::GROUP_MAX || !self.backend.packed_product_ready(unit.rows,unit.columns,wide,pairs) {return Ok(None);}
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,spark_gpu_probe::packed_product::layout(unit.rows,unit.columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && units.iter().all(|w|w.len().checked_mul(4)==Some(plan.bytes)),"packed group: the unit of the load plan differs from the unit of the kernel")?;
        need(x.words==if shared {unit.columns} else {unit.columns*picks},"packed group: the input differs from the columns of the weights")?;
        self.ready(x)?;
        let matrices=units.iter().map(|w|weight.resident_range(w.clone())).collect::<std::result::Result<Vec<_>,_>>()?;
        if matrices.iter().any(|m|!m.device_address().is_multiple_of(16)) || !x.vector.device_address().is_multiple_of(16) {return Ok(None);}
        let mut output=self.backend.vector(unit.rows*picks)?;
        self.backend.project_packed_group_into(unit.rows,unit.columns,wide,pairs,&x.vector,shared,&matrices.iter().collect::<Vec<_>>(),&mut output)?;
        self.value(output,unit.rows*picks).map(Some)
    }
    /// The packed products of the experts that the route picked, in ONE launch, with the expert numbers read on the
    /// device (`Backend::project_packed_picked_into`): `stack` holds the units of all `experts` experts of the shape
    /// `unit` back to back; `ids` is the route output's first `picks` words (FP32 whole numbers, in pick order). The
    /// result: the words of `packed_group` with the units of those experts. The host reads no expert number. None where
    /// `packed_group` would give None.
    pub(crate) fn packed_picked(&self,stack:&DeviceVector<'d>,experts:usize,unit:UnitShape,ids:&Value<'b,'d>,x:&Value<'b,'d>,shared:bool,one_row:bool,caps:Option<[f32;4]>)->Result<Option<Value<'b,'d>>> {
        let picks = ids.words;
        self.packed_picked_strided(stack, experts, unit, ids, picks, picks, picks, x, if shared { 0 } else { 1 }, caps, one_row)
    }
    /// Picked packed products reading strided route rows (`id_stride >= picks_per_row`) with optional input row
    /// repeat (`repeat == 0` shares one input row; `repeat >= 1` maps pick `j` to input row `j / repeat`).
    /// `one_row`: the products of one decoded token, which read the one-row expert panels of the stack when it has them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn packed_picked_strided(
        &self, stack: &DeviceVector<'d>, experts: usize, unit: UnitShape,
        ids: &Value<'b, 'd>, picks: usize, picks_per_row: usize, id_stride: usize,
        x: &Value<'b, 'd>, repeat: usize, caps: Option<[f32; 4]>, one_row: bool,
    ) -> Result<Option<Value<'b, 'd>>> {
        self.packed_picked_strided_compact(stack, experts, unit, ids, picks, picks_per_row, id_stride, x, repeat, caps, one_row, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn packed_picked_strided_compact(
        &self, stack: &DeviceVector<'d>, experts: usize, unit: UnitShape,
        ids: &Value<'b, 'd>, picks: usize, picks_per_row: usize, id_stride: usize,
        x: &Value<'b, 'd>, repeat: usize, caps: Option<[f32; 4]>, one_row: bool, compact: Option<&Value<'b, 'd>>,
    ) -> Result<Option<Value<'b, 'd>>> {
        let (wide, pairs) = (unit.codes == ExpertCodes::Mxfp8, unit.order.is_some());
        if caps.is_some_and(|v| !pairs || !v.iter().all(|c| c.is_finite() && *c > 0.0)) { return Ok(None); }
        if picks == 0 || picks > spark_gpu_probe::packed_product::PICKS_MAX || picks_per_row == 0 || !picks.is_multiple_of(picks_per_row) || id_stride < picks_per_row { return Ok(None); }
        if repeat > 0 && !picks.is_multiple_of(repeat) { return Ok(None); }
        if !self.backend.packed_picked_ready(unit.rows, unit.columns, wide, pairs) { return Ok(None); }
        let (plan, tile) = (unit.layout().map_err(|e| e.to_string())?, spark_gpu_probe::packed_product::layout(unit.rows, unit.columns, wide));
        need((plan.scales, plan.table, plan.bytes) == (tile.scales, tile.table, tile.bytes) && experts.checked_mul(plan.bytes) == stack.len().checked_mul(4), "packed picks: the stack of the load plan differs from the units of the kernel")?;
        let input_rows = picks.checked_div(repeat).unwrap_or(1);
        need(x.words == input_rows * unit.columns, "packed picks: the input differs from the columns of the weights")?;
        need(ids.words == (picks / picks_per_row) * id_stride, "packed picks: route extent differs from strided layout")?;
        self.ready(x)?; self.ready(ids)?;
        if let Some(cmp) = compact {
            need(cmp.words == picks * spark_gpu_probe::fast_ops::ROUTER_PICK_COMPACT_ENTRY_WORDS, "packed picks: compact dispatch extent differs")?;
            self.ready(cmp)?;
            if cmp.vector.is_segmented() || !cmp.vector.device_address().is_multiple_of(16) { return Ok(None); }
        }
        if !stack.device_address().is_multiple_of(16) || !x.vector.device_address().is_multiple_of(16) || ids.vector.is_segmented() || x.vector.is_segmented() { return Ok(None); }
        let mut output = self.backend.vector(unit.rows * picks)?;
        let panels = if one_row && repeat <= 1 { self.expert_panels.get(&stack.device_address()) } else { None };
        self.backend.project_packed_picked_strided_compact_into(unit.rows, unit.columns, wide, pairs, &x.vector, repeat, stack, experts, &ids.vector, picks, picks_per_row, id_stride, &mut output, caps, one_row, panels, compact.map(|c| &*c.vector))?;
        let words = if caps.is_some() { unit.rows * picks / 2 } else { unit.rows * picks };
        let value = self.value(output, unit.rows * picks)?;
        if caps.is_some() { self.range(&value, 0, words).map(Some) } else { Ok(Some(value)) }
    }
    /// The products of the expert runs of a step of rows from their packed units, with no gather
    /// (`ResidentProduct::packed_product_runs`): `runs` holds (stack, unit words, launch rows) of each run in launch
    /// order, every unit of the shape `unit`. Launch row `i` reads row `input[i]` of `x` (row `i` without a table); row
    /// `j` of the result is launch row `output[j]` (row `j` without a table). The words are those of
    /// `resident_product_runs` on the widened matrices: one chain a row over the columns, the same chain as the rows
    /// tile. The runs go in as few grouped launches as the recorder's stage takes. None where a launch cannot take them
    /// (a segmented operand, a shape outside the tile, launches not recorded); then nothing was launched.
    pub(crate) fn packed_product_runs(&self,runs:&[(&DeviceVector<'d>,std::ops::Range<usize>,usize)],unit:UnitShape,x:&Value<'b,'d>,input:Option<&[usize]>,output:Option<&[usize]>)->Result<Option<Value<'b,'d>>> {
        let (wide,pairs,rows,columns)=(unit.codes==ExpertCodes::Mxfp8,unit.order.is_some(),unit.rows,unit.columns);
        let n=runs.iter().map(|run|run.2).sum::<usize>();
        if rows==0 || columns==0 || n==0 || !x.words.is_multiple_of(columns) || x.vector.len()!=x.words || x.vector.is_segmented() {return Ok(None);}
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,spark_gpu_probe::packed_product::layout(rows,columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && runs.iter().all(|run|run.2>0 && run.1.len().checked_mul(4)==Some(plan.bytes)),"packed runs: the unit of the load plan differs from the unit of the kernel")?;
        self.ready(x)?;
        let sources=x.words/columns;
        // The input row and the result row of each launch row (the result table is the inverse of `output`).
        let word=|row:usize|u32::try_from(row).map_err(|_|"packed runs: a row of the runs exceeds the ABI".to_owned());
        let in_words=match input {Some(rows)=>rows.iter().map(|&row|word(row)).collect::<Result<Vec<_>>>()?,None=>(0..n).map(word).collect::<Result<Vec<_>>>()?};
        let mut out_words=(0..n).map(word).collect::<Result<Vec<_>>>()?;
        if let Some(rows)=output {for (j,&i) in rows.iter().enumerate() {*out_words.get_mut(i).ok_or("packed runs: an output row of the runs is outside the launch rows")?=word(j)?;}}
        need(in_words.len()==n && output.is_none_or(|rows|rows.len()==n) && in_words.iter().all(|&row|(row as usize)<sources),"packed runs: a table of the runs differs from the launch rows")?;
        let units=runs.iter().map(|run|run.0.resident_range(run.1.clone())).collect::<std::result::Result<Vec<_>,_>>()?;
        // A run of more launch rows than the tile takes goes in pieces of the same unit, in order.
        let most=spark_gpu_probe::staged_product::ROWS_INPUTS_MAX;
        let pieces=runs.iter().enumerate().flat_map(|(r,run)|(0..run.2).step_by(most).map(move |from|(r,(run.2-from).min(most)))).collect::<Vec<_>>();
        // The launches: consecutive pieces while the staged tables of one launch fit, all asked before the first launch.
        let mut launches:Vec<std::ops::Range<usize>>=Vec::new();
        for at in 0..pieces.len() {
            let grown=launches.last().map(|last|last.start..at+1).filter(|range|{
                let members=range.clone().map(|p|(&units[pieces[p].0],pieces[p].1)).collect::<Vec<_>>();
                self.backend.packed_rows_grouped_ready(rows,columns,wide,pairs,&members,&x.vector,true)
            });
            match grown {Some(range)=>*launches.last_mut().expect("a launch was found")=range,None=>{
                if !self.backend.packed_rows_grouped_ready(rows,columns,wide,pairs,&[(&units[pieces[at].0],pieces[at].1)],&x.vector,true) {return Ok(None);}
                launches.push(at..at+1);
            }}
        }
        let total=n.checked_mul(rows).ok_or("packed runs: the result of the runs is too large")?;
        let whole=self.backend.vector(total)?.into_resident_shared()?;
        let mut first=0;
        for launch in launches {
            let members=launch.clone().map(|p|(&units[pieces[p].0],pieces[p].1)).collect::<Vec<_>>();
            let count=members.iter().map(|member|member.1).sum::<usize>();
            let span=first..first+count;
            self.backend.project_packed_rows_grouped_into(rows,columns,wide,pairs,rows,&members,&x.vector,Some(&in_words[span.clone()]),&mut whole.resident_range(0..total)?,Some(&out_words[span]))?;
            first+=count;
        }
        Ok(Some(self.shared_value(Rc::new(whole),total)?))
    }
}
