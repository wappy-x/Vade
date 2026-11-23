//! The product of one packed expert matrix with one input row, for the driver of the block family with a loop.
//!
//! The matrix is a range of words of a resident vector: the unit of one expert in a packed stack
//! (`engine_model::plan::loop_packed_unit`: the stored codes, the stored block scales and a weight table). The
//! output is `rows` 32-bit words, gate rows first: word for word the output of `resident_product` on the same
//! matrix widened to BF16 (`Backend::project_packed_into`, kernel body `plain_packed`).
//!
//! The packed kernel takes whole stages of columns, 16-byte aligned operands and recorded launches. For any other
//! call this file gives None and the caller reads the widened matrix, so no shape and no state is refused here.
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
    pub(crate) fn packed_picked(&self,stack:&DeviceVector<'d>,experts:usize,unit:UnitShape,ids:&Value<'b,'d>,x:&Value<'b,'d>,shared:bool)->Result<Option<Value<'b,'d>>> {
        let (wide,pairs,picks)=(unit.codes==ExpertCodes::Mxfp8,unit.order.is_some(),ids.words);
        if picks==0 || picks>spark_gpu_probe::packed_product::GROUP_MAX || !self.backend.packed_picked_ready(unit.rows,unit.columns,wide,pairs) {return Ok(None);}
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,spark_gpu_probe::packed_product::layout(unit.rows,unit.columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"packed picks: the stack of the load plan differs from the units of the kernel")?;
        need(x.words==if shared {unit.columns} else {unit.columns*picks},"packed picks: the input differs from the columns of the weights")?;
        self.ready(x)?;self.ready(ids)?;
        if !stack.device_address().is_multiple_of(16) || !x.vector.device_address().is_multiple_of(16) {return Ok(None);}
        let mut output=self.backend.vector(unit.rows*picks)?;
        self.backend.project_packed_picked_into(unit.rows,unit.columns,wide,pairs,&x.vector,shared,stack,experts,&ids.vector,&mut output)?;
        self.value(output,unit.rows*picks).map(Some)
    }
}
