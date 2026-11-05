//! Safe resident adapters. Checkpoint arithmetic qualification remains open.
use super::*;
use spark_gpu_probe::driver::{Backend,Matrix,Bf16Vector,ResidentNorm,NormOutput,ResidentMhc,MhcPass};

pub struct ResidentProjection<'b,'d> {backend: &'b Backend<'d>,matrix: Matrix<'d>,topology: Topology,origin: u64}
impl ProjectionOperands {
    pub fn upload<'b,'d>(&self,backend: &'b Backend<'d>) -> Result<ResidentProjection<'b,'d>> {
        backend.require_resident_stream()?;
        let [rows,cols]=self.shape;
        // Packing changes byte placement only. A local matrix retains complete K.
        let matrix=backend.upload_native_bf16(rows,cols,1024*1024,|offset,out| {
            let end=offset.checked_add(out.len()).ok_or("projection read extent overflow")?;
            out.copy_from_slice(self.weight.get(offset..end).ok_or("projection read exceeds selected rows")?);Ok(())
        })?;
        backend.synchronize()?;
        Ok(ResidentProjection {backend,matrix,topology:self.topology,origin:self.origin})
    }
}
impl<'d> ResidentProjection<'_,'d> {
    pub fn topology(&self) -> Topology {self.topology}
    pub fn row_origin(&self) -> u64 {self.origin}
    /// Return the local output rows only. The layer's declared gather must
    /// finish before a consumer can treat these rows as a replicated tensor.
    pub fn apply(&self,input: &Bf16Vector<'_>) -> Result<Bf16Vector<'d>> {
        let b=self.backend;b.require_resident_stream()?;
        let mut output=b.vector(self.matrix.rows())?;
        let launched=b.project_bf16_resident(&self.matrix,input.vector(),&mut output);
        let completed=b.synchronize();launched.and(completed)?;
        // The existing projection already rounds. This typed copy retains its
        // public BF16 invariant without exposing the private vector constructor.
        b.round_bf16_lanes(&output)
    }
}
pub struct ResidentResidualBlock<'b,'d> {mhc: ResidentMhc<'b,'d>,norm: ResidentNorm<'b,'d>}
pub struct ResidualBlockPass<'b,'d> {mhc: MhcPass<'b,'d>,normalized: NormOutput<'d>}
impl ResidualBlockOperands {
    pub fn upload<'b,'d>(&self,backend: &'b Backend<'d>) -> Result<ResidentResidualBlock<'b,'d>> {
        // Both immutable owners use the same backend and checkpoint operands.
        let mhc=self.mhc.upload(backend)?;let norm=self.norm.upload(backend)?;
        Ok(ResidentResidualBlock {mhc,norm})
    }
}
impl<'b,'d> ResidentResidualBlock<'b,'d> {
    pub fn begin(&self,residual: Bf16Vector<'d>) -> Result<ResidualBlockPass<'b,'d>> {
        let mhc=self.mhc.begin(residual)?;let normalized=self.norm.apply(mhc.collapsed())?;
        Ok(ResidualBlockPass {mhc,normalized})
    }
}
impl<'d> ResidualBlockPass<'_,'d> {
    /// Inspect the completed prefix without releasing its residual or coefficients.
    pub fn mhc(&self) -> &MhcPass<'_,'d> {&self.mhc}
    pub fn normalized(&self) -> &NormOutput<'d> {&self.normalized}
    pub fn finish(self,block: Bf16Vector<'d>) -> Result<Bf16Vector<'d>> {self.mhc.finish(block)}
}
