use super::*;
use crate::prompt_route::{Build,Shape,THREADS};
/// Private metadata for one route epoch. No plane can be reused while this owner lives.
pub struct PromptRoute<'a> {shape:Shape,ids:Rc<DeviceVector<'a>>,planes:[DeviceVector<'a>;9]}
impl<'a> PromptRoute<'a> {
    pub fn shape(&self)->Shape {self.shape}
    pub fn ids(&self)->&DeviceVector<'a> {&self.ids}
    pub fn counts(&self)->&DeviceVector<'a> {&self.planes[0]}
    pub fn starts(&self)->&DeviceVector<'a> {&self.planes[1]}
    pub fn order(&self)->&DeviceVector<'a> {&self.planes[2]}
    pub fn pairs(&self)->&DeviceVector<'a> {&self.planes[3]}
    pub fn grouped(&self)->&DeviceVector<'a> {&self.planes[4]}
    pub fn placed(&self)->&DeviceVector<'a> {&self.planes[5]}
    pub fn work128(&self)->&DeviceVector<'a> {&self.planes[6]}
    pub fn work64(&self)->&DeviceVector<'a> {&self.planes[7]}
    /// Static workspace quote, including every metadata/work plane and page rounding.
    pub fn capacity_bytes(&self)->usize {self.shape.workspace_bytes().expect("admitted prompt route")}
    /// Actual allocation owners, for a queued consumer's retention ledger.
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {self.ids.allocations(out);for plane in &self.planes {plane.allocations(out);}}
}
impl<'a> Backend<'a> {
    /// Build all route tables on the current ordered stream. This method reads no
    /// device value and does not synchronize. Keep the owner through every consumer.
    pub fn prompt_route(&self,ids:Rc<DeviceVector<'a>>,shape:Shape)->Result<PromptRoute<'a>,String> {
        self.require_resident_stream()?;
        let lengths=shape.lengths().ok_or("prompt route dimensions are outside the admitted range")?;
        if ids.is_segmented() || ids.len()!=shape.pairs() {return Err("prompt route ID extent differs".into());}
        self.check_vector(&ids,shape.pairs())?;
        self.require_arm_kernels(&["prompt_route_count","prompt_route_tables"])?;
        let mut made=Vec::with_capacity(lengths.len());
        for words in lengths {let mut plane=self.vector(words)?;self.skip_fill(&mut plane)?;made.push(plane);}
        let planes:[DeviceVector<'a>;9]=made.try_into().map_err(|_|"prompt route plane count differs")?;
        let [counts,starts,order,pairs,grouped,placed,work128,work64,status]=planes.each_ref().map(|v|v.device_address());
        let mut args=Build {ids:ids.device_address(),counts,starts,order,pairs,grouped,placed,work128,work64,status,shape,pad:0};
        // Every output is a fresh private vector; the input and all outputs have
        // exact checked spans. The two launches are ordered on this same stream.
        self.unrecorded(|| {
            self.launch_checked_expert_payload("prompt_route_count",shape.experts,1,THREADS,0,&mut args)?;
            self.launch_checked_expert_payload("prompt_route_tables",shape.experts,1,THREADS,0,&mut args)
        })?;
        crate::prompt_launches::record("prompt_route_count");crate::prompt_launches::record("prompt_route_tables");
        Ok(PromptRoute {shape,ids,planes})
    }
}
