//! The host twin with a record of the calls a device answers with memory work: when launch recording is paused
//! and resumed, when a group's spare buffers are drained, and each append of expanded rows with the room it asks
//! for. The words are the host's. `fail_extend` makes one append fail, as a refused device allocation would.
use spark_tp::{bounded_decoder::{Compute,Host},Result};
use std::cell::{Cell,RefCell};
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Event {
    /// `rows_unrecorded(pause)`.
    Pause(bool),
    /// `rows_keep`: a device drains here and frees what was dropped since the last drain.
    Keep,
    /// `rows_extend`: whether earlier rows were handed in, how many, the words of the new rows, the room asked behind them.
    Extend {held:bool,rows:usize,words:usize,room:usize}}
#[derive(Default)]
pub struct Watch {pub events:RefCell<Vec<Event>>,paused:Cell<bool>,
    /// The append with this number (from 0) fails. None: none fails.
    pub fail_extend:Cell<Option<usize>>,extends:Cell<usize>}
impl Watch {
    pub fn take(&self)->Vec<Event> {std::mem::take(&mut *self.events.borrow_mut())}
    /// The appends so far: `fail_extend` counts from here.
    pub fn extends(&self)->usize {self.extends.get()}
}
impl Compute for Watch {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    fn rows_unrecorded(&self,pause:bool)->Result<bool> {self.events.borrow_mut().push(Event::Pause(pause));Ok(self.paused.replace(pause))}
    fn rows_keep(&self)->Result<()> {self.events.borrow_mut().push(Event::Keep);Ok(())}
    fn rows_extend(&self,held:Option<&Vec<f32>>,rows:usize,new:&Vec<f32>,width:usize,room:usize)->Result<Vec<f32>> {
        self.events.borrow_mut().push(Event::Extend {held:held.is_some(),rows,words:new.len(),room});
        let number=self.extends.replace(self.extends.get()+1);
        if self.fail_extend.get()==Some(number) {return Err("watched device refused the expanded rows".into());}
        Host.rows_extend(held,rows,new,width,room)
    }
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Vec<f32>,_:usize)->Result<Option<Vec<u32>>> {Ok(None)}
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,s:spark_gpu_probe::causal_ops::Shape,x:[Option<&Vec<f32>>;4],reserve:usize,expected:f32)->Result<Vec<f32>> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:spark_gpu_probe::causal_ops::Shape,q:&Vec<f32>,l:&Vec<f32>,n:&Vec<f32>,slots:usize)->Result<Vec<f32>> {Host.latent_absorption(name,s,q,l,n,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,q:&Vec<f32>,l:&Vec<f32>,n:&Vec<f32>)->Result<Vec<f32>> {Host.latent_absorption_rows(name,s,q,l,n)}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,live:usize)->Result<()> {Host.prepare_beside(c,live)}
    fn select_known(&self,ids:&Vec<f32>,count:&Vec<f32>,row:&[f32])->Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->Result<()> {Ok(())}
    fn prompt_seeds(&self)->bool {false}
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {Host.upload(v)}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {Host.read(v)}
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->Result<Vec<f32>> {Host.range(v,start,len)}
    fn words(&self,v:&Vec<f32>)->usize {v.len()}
    fn join(&self,parts:&[&Vec<f32>])->Result<Vec<f32>> {Host.join(parts)}
    fn zeros(&self,len:usize)->Result<Vec<f32>> {Host.zeros(len)}
    fn operation(&self,s:spark_gpu_probe::causal_ops::Shape,x:[Option<&Vec<f32>>;4],w:&[f32])->Result<Vec<f32>> {Host.operation(s,x,w)}
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Vec<f32>,w:&[f32])->Result<Vec<f32>> {Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&Vec<f32>)->Result<(Self::Pass,Vec<f32>)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&Vec<f32>)->Result<Vec<f32>> {Host.finish(p,x)}
    fn mx(&self,p:&spark_tp::mxfp8_projection::Mxfp8Projection<'_>,x:&Vec<f32>)->Result<Vec<f32>> {Host.mx(p,x)}
    fn round(&self,x:&Vec<f32>)->Result<Vec<f32>> {Host.round(x)}
    fn mean(&self,x:&Vec<f32>,s:u32,h:u32)->Result<Vec<f32>> {Host.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Vec<f32>>;5],w:&[f32])->Result<Vec<f32>> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Vec<f32>,m:usize,w:&[f32])->Result<Vec<f32>> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&Vec<f32>,m:usize)->Result<(Self::Pass,Vec<f32>)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&Vec<f32>,m:usize)->Result<Vec<f32>> {Host.rows_finish(p,x,m)}
}
