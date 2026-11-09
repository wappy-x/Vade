//! A replay owns every captured address. Inputs and outputs have separate slots.
use super::*;
use crate::resident_program::Program;
/// Destruction retires the executable before its retained storage. One stream
/// orders input copies, replay and fresh immutable output copies on every call.
pub struct ResidentProgram<'b,'a> {graph:DecodeGraph<'a>,backend:&'b Backend<'a>,inputs:Vec<DeviceVector<'a>>,outputs:Vec<DeviceVector<'a>>,_weights:Vec<Option<DeviceVector<'a>>>}
impl<'a> Backend<'a> {
    pub fn resident_program(&self,program:&Program,weights:&[Option<&DeviceVector<'a>>])->Result<ResidentProgram<'_,'a>,String> {
        self.require_resident_stream()?;
        if self.resident_launches_active() {return Err("standalone resident program requires launch recording off".into());}
        if !self.context.resident_queued.get() {return Err("resident program requires an owned queue".into());}
        let slots=program.validate()?;if weights.len()!=program.nodes.len() {return Err("resident program weight count differs".into());}
        self.require_arm_kernels(&["causal_operation","project_f32_ordered"])?;
        let mut retained=Vec::new();
        for (node,weight) in program.nodes.iter().zip(weights) {
            let (_,words,_,_)=node.shape.lengths().ok_or("resident program shape differs")?;
            retained.push(match weight {Some(w) if words>0=>{self.check_vector(w,words)?;Some(w.resident_range(0..words)?)},None if words==0=>None,_=>return Err("resident program weight extent differs".into())});
        }
        let inputs=program.inputs.iter().map(|&n|self.vector(n)).collect::<Result<Vec<_>,_>>()?;
        let mut outputs=slots[inputs.len()..].iter().map(|&n|self.vector(n)).collect::<Result<Vec<_>,_>>()?;
        // SAFETY: Validation permits only earlier exact contiguous slots. The
        // callback allocates no GPU storage and reads no host/device results.
        // Every captured input, weight and output allocation moves into the
        // returned owner. Replay cannot outlive them or expose mutable outputs.
        let graph=unsafe {self.capture_batch_bounded(program.nodes.len()*2,|| {
            for (index,node) in program.nodes.iter().enumerate() {
                let (prior,current)=outputs.split_at_mut(index);let output=&mut current[0];
                let arguments=node.inputs.map(|slot|slot.map(|slot|if slot<inputs.len() {&inputs[slot]} else {&prior[slot-inputs.len()]}));
                self.resident_causal_into(node.shape,arguments,retained[index].as_ref(),output)?;
            }
            Ok(())
        })}?;
        Ok(ResidentProgram {graph,backend:self,inputs,outputs,_weights:retained})
    }
}
impl<'a> ResidentProgram<'_,'a> {
    /// Results are fresh copies. A second replay cannot overwrite a prior result.
    pub fn run(&mut self,inputs:&[&DeviceVector<'_>])->Result<DeviceVector<'a>,String> {
        self.run_inner(inputs,false)
    }
    /// Append initialized positive-zero lanes to each shorter immutable input.
    /// This is useful for a fixed-capacity graph whose live count is an operand.
    pub fn run_padded(&mut self,inputs:&[&DeviceVector<'_>])->Result<DeviceVector<'a>,String> {self.run_inner(inputs,true)}
    fn run_inner(&mut self,inputs:&[&DeviceVector<'_>],padded:bool)->Result<DeviceVector<'a>,String> {
        let b=self.backend;b.require_resident_stream()?;
        if b.resident_launches_active() {return Err("standalone resident program requires launch recording off".into());}
        if !b.context.resident_queued.get() {return Err("resident replay requires an owned queue".into());}
        if inputs.len()!=self.inputs.len() {return Err("resident replay input count differs".into());}
        // Validate all inputs before enqueuing the first copy.
        for (input,slot) in inputs.iter().zip(&self.inputs) {b.check_vector(input,input.len())?;if input.len()>slot.len() || (!padded && input.len()!=slot.len()) {return Err("resident replay input extent differs".into());}if input.is_segmented() {return Err("resident replay input must be contiguous".into());}}
        for (input,slot) in inputs.iter().zip(&mut self.inputs) {if input.len()<slot.len() {b.zero_vector_range(slot,input.len()..slot.len())?;}b.copy_prefix(input,slot,input.len())?;}
        // SAFETY: This owner retains every allocation used by graph. The caller
        // cannot access its mutable slots. Prior input copies and subsequent
        // output copies share the capture's ordered stream, including errors.
        unsafe {b.replay_decode(&self.graph)?;}
        let source=self.outputs.last().ok_or("resident program has no output")?;let mut output=b.vector(source.len())?;b.copy_prefix(source,&mut output,source.len())?;
        Ok(output)
    }
}
