//! Synthetic files for the isolated projection probe. No checkpoint evidence.
use engine_model::{plan::{ConfigPlan,TextDescription,description::*,layer::{LayerPlan,Topology}},
    shapes::Ffn};
use std::{fs::{File,OpenOptions},io::Write,path::Path};
pub type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;

pub fn plan(hidden: u64,parts: u32,rank: u32) -> Result<LayerPlan> {
    make(&description(hidden)?,parts,rank)
}
pub fn description(hidden: u64) -> Result<TextDescription> {
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json"))? else {return Err("expected deferred fixture".into());};
    d.hidden=hidden;d.max_positions=16;d.residual=SinkhornDescription {streams:2,iterations:4,epsilon:1e-5};
    d.layers=vec![DecoderDescription {mixer:TokenMixer::Kda(KdaDescription {heads:4,head_dim:3,conv:4,gate_lower_bound:-5.0}),
        ffn:Ffn::Dense {width:12}}];
    d.storage.layers.clear();d.storage.excluded.clear();
    Ok(d)
}
fn make(d: &TextDescription,parts: u32,rank: u32) -> Result<LayerPlan> {Ok(LayerPlan::new(d,0,1,Topology {parts,rank})?)}
pub fn write(path: &Path,name: &str,dtype: &str,rows: u64,cols: u64,bytes: &[u8]) -> Result<File> {
    let header=format!("{{{name:?}:{{\"dtype\":{dtype:?},\"shape\":[{rows},{cols}],\"data_offsets\":[0,{}]}}}}",bytes.len());
    let mut file=OpenOptions::new().read(true).write(true).create_new(true).open(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;file.write_all(header.as_bytes())?;file.write_all(bytes)?;
    file.sync_all()?;Ok(file)
}
pub fn values(rows: usize,cols: usize) -> (Vec<f32>,Vec<f32>) {
    let mut x=(0..cols).map(|j| ((j*13%29) as f32-14.0)*0.0625).collect::<Vec<_>>();
    let mut w=(0..rows*cols).map(|j| ((j*7%31) as f32-15.0)*0.03125).collect::<Vec<_>>();
    if cols>=3 && rows>=4 {
        x[..3].copy_from_slice(&[1.0,f32::from_bits(0x3f7ffffe),1.0]);
        w[..cols*4].fill(0.0);
        w[..3].copy_from_slice(&[-1.0,f32::from_bits(0x3f800001),0.0]);
        w[cols..cols+3].copy_from_slice(&[16777216.0,1.0,-16777216.0]);
        w[2*cols..2*cols+3].copy_from_slice(&[-16777216.0,16777216.0,1.0]);
        w[3*cols]=f32::from_bits(1);w[3*cols+1]= -0.0;
    }
    (w,x)
}
pub fn bytes(values: &[f32]) -> Vec<u8> {values.iter().flat_map(|v| v.to_le_bytes()).collect()}
