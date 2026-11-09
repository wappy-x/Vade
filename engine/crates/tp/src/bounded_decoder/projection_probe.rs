use super::{Result,device::Device};
use spark_gpu_probe::driver::DeviceVector;

#[derive(Default)]
pub(super) struct Probe {pub weight:String,pub expected:Option<u32>,pub active:bool,pub output:Option<Vec<f32>>}

impl<'b,'d> Device<'b,'d> {
    pub fn fixture_projection_probe(&self,weight:&str,expected:Option<u32>) {
        *self.probe.borrow_mut()=Some(Probe {weight:weight.to_owned(),expected,..Default::default()});
    }
    pub fn fixture_start_projection_probe(&self) {if let Some(probe)=self.probe.borrow_mut().as_mut() {probe.active=true;}}
    pub fn fixture_projection_receipt(&self)->Result<Option<(String,Vec<f32>)>> {
        let probe=self.probe.borrow();let Some(probe)=probe.as_ref() else {return Ok(None);};
        let output=probe.output.as_ref().ok_or("warm projection marker was not reached during decode")?;
        if output.is_empty() || output.iter().any(|value|!value.is_finite()) {return Err("warm projection marker has invalid output".into());}
        if let Some(expected)=probe.expected {if output.iter().any(|value|value.to_bits()!=expected) {return Err("warm projection marker differs from selected image contract".into());}}
        Ok(Some((format!("weight={} expected_bits={:?} words={} decode_projection=true\n",probe.weight,probe.expected,output.len()),output.clone())))
    }
    pub(super) fn observe_projection(&self,name:&str,output:&DeviceVector<'d>)->Result<()> {
        let needed=self.probe.borrow().as_ref().is_some_and(|probe|probe.active && probe.output.is_none() && probe.weight==name);
        if needed {let output=self.backend.read_vector_values(output)?;
            self.probe.borrow_mut().as_mut().ok_or("warm projection probe disappeared")?.output=Some(output);}
        Ok(())
    }
}
