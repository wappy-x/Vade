//! Checked plain FP32 operands through the existing loaded matrix owner.
use super::*;
use crate::ordered_projection::{Error,Shape};
impl Backend<'_> {
    pub(super) fn check_ordered_projection(&self,matrix: &Matrix<'_>,input: &DeviceVector<'_>,output: &DeviceVector<'_>) -> Result<(),Error> {
        if !ptr::eq(matrix.buffer._context,self.context) || !ptr::eq(input.buffer._context,self.context)
            || !ptr::eq(output.buffer._context,self.context) {return Err(Error::Context);}
        let shape=Shape::new(matrix.rows,matrix.cols)?;
        shape.layout(matrix.scalar_f32 && !matrix.packed && !matrix.is_fp8() && !matrix.fp8_fragments,
            matrix.stride,[matrix.row_offset,matrix.column_offset],matrix.buffer.bytes)?;
        if input.is_segmented() || output.is_segmented() {return Err(Error::Layout);}
        if input.len!=shape.cols() || output.len!=shape.rows() {return Err(Error::Extent);}
        shape.operands((matrix.buffer.address,matrix.buffer.bytes),(input.buffer.address,input.buffer.bytes),
            (output.buffer.address,output.buffer.bytes))
    }
    /// The resident BF16 adapter accepts only complete unblocked matrices.
    /// The shared projection keeps the existing hybrid-model contract.
    pub fn project_bf16_resident(&self,matrix:&Matrix<'_>,input:&DeviceVector<'_>,output:&mut DeviceVector<'_>)->Result<(),String> {
        if matrix.packed_blocked {return Err("native BF16 projection requires a complete unblocked packed matrix".into());}
        self.matvec_bf16_device(matrix,input,output)
    }
    /// Logical FP32 primitive only. This does not qualify effective runtime loads.
    /// Complete work before any borrowed operand can leave this call.
    pub fn project_f32_ordered(&self,matrix: &Matrix<'_>,input: &DeviceVector<'_>,output: &mut DeviceVector<'_>) -> Result<(),Error> {
        self.check_ordered_projection(matrix,input,output)?;
        // Retain the matrix allocation in a checked immutable vector view.
        // The resident operation owns its launch and returns completed rows.
        let weight=DeviceVector {buffer:matrix.weight_view(0).map_err(Error::Device)?,
            len:matrix.rows*matrix.cols,managed:false,#[cfg(debug_assertions)] projection_scratch:Default::default()};
        let shape=crate::causal_ops::Shape {op:crate::causal_ops::Op::Project,heads:0,dim:0,taps:0,
            rows:matrix.rows as u32,cols:matrix.cols as u32,epsilon:0.0,bound:0.0};
        let rows=self.causal_operation(shape,[Some(input),None,None,None],Some(&weight)).map_err(Error::Device)?;
        let launch=self.copy_prefix(&rows,output,matrix.rows).map_err(Error::Device);
        let complete=self.synchronize().map_err(Error::Device);
        launch.and(complete)
    }
}
