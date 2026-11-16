//! GATHERED ROWS: in a prompt group the gate and up step of a routed layer reads every token's row once for each of
//! its picked experts. Before, the rows were copied into expert order first (`PermuteRows`: one copy of a row a pick)
//! and the compute read the copy (`Compute::rows_experts`). Now the compute is asked first to read the rows where they
//! lie, through the table of that copy (`Compute::rows_experts_gathered`); a compute that refuses gets the copy as it
//! was. Every form must give the bits of the group that copies.
use spark_tp::bounded_decoder::{Compute,Decoder,HeadRows,Host};
use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan,Topology}},safetensors::Dtype};
use spark_gpu_probe::{bf16::rounded,causal_ops::Shape,prompt_ops::{Op as Row,Shape as RowShape}};
use std::cell::Cell;
#[allow(dead_code,reason="the shared fixture has more forms than this test uses")]
#[path="support/attention_checkpoint.rs"] mod fixture;
type V=Vec<f32>;
/// What the compute does when it is asked to read the gate rows through the table.
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
enum Form {
    /// It refuses (an older kernel image, a group outside the dense path): the caller copies the rows.
    Copies,
    /// It reads each row where it lies, as the device does.
    Gathers,
    /// It reads through the table shifted by one row: wrong on purpose, to show that the table decides the output.
    Shifted,
}
/// The host compute as a pair's device for a prompt group. The grouped expert passes are answered with a stand-in for
/// the expert products: finished BF16 words that depend on the source row, on the expert and on the lane, so a row read
/// for the wrong token or made by the wrong expert cannot pass. Every form uses the same stand-in; the group that
/// copies its rows is the reference.
struct Gathering<'c> {c:&'c CheckpointAdmission,form:Form,
    /// Times the caller asked for the gathered form, and times the compute answered it.
    asked:Cell<usize>,answered:Cell<usize>,
    /// Gate and up passes on copied rows (`rows_experts` with a second list), and every `PermuteRows` the caller ran.
    copied:Cell<usize>,permutes:Cell<usize>}
impl<'c> Gathering<'c> {
    fn new(c:&'c CheckpointAdmission,form:Form)->Self {Self {c,form,asked:Cell::new(0),answered:Cell::new(0),copied:Cell::new(0),permutes:Cell::new(0)}}
    /// (output rows, input columns) of a matrix.
    fn extent(&self,name:&str)->(usize,usize) {let shape=&self.c.headers().weights().get(name).unwrap().logical().shape;(shape[0] as usize,shape[1] as usize)}
    /// The stand-in for one list of expert products: output row `row` reads the source row of its slot (`earlier + at`).
    fn experts(&self,list:&[(&str,usize)],source:&V,places:&[f32])->V {
        let n=list.iter().map(|v|v.1).sum::<usize>();assert!(n>0 && places.len()==3*n,"three numbers an output row");
        let (out,width)=self.extent(list[0].0);assert_eq!(source.len(),n*width,"one source row a slot");
        let mut owner=Vec::with_capacity(n);
        for (name,count) in list {assert_eq!(self.extent(name),(out,width));owner.extend(std::iter::repeat_n((name.bytes().map(usize::from).sum::<usize>()%13) as f32/16.0,*count));}
        let mut value=vec![0.0;n*out];
        for (row,entry) in places.as_chunks::<3>().0.iter().enumerate() {let slot=entry[0] as usize+entry[2] as usize;assert!(entry[2]<entry[1] && slot<n);
            for lane in 0..out {value[row*out+lane]=rounded(source[slot*width+lane%width]*0.5+owner[slot]+(lane%7) as f32/32.0);}}
        value
    }
    /// Gate and up of one group from rows in expert order: the two lists meet as the device's SwiGLU stand-in.
    fn both(&self,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,source:&V,places:&[f32])->V {
        let a=self.experts(first,source,places);
        match second {Some((list,_))=>{let c=self.experts(list,source,places);a.iter().zip(&c).map(|(a,c)|rounded(a*0.5+c*0.25)).collect()},None=>a}
    }
}
impl Compute for Gathering<'_> {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn rows_experts(&self,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,source:&V,places:&[f32])->spark_tp::Result<Option<V>> {
        if second.is_some() {self.copied.set(self.copied.get()+1);}
        Ok(Some(self.both(first,second,source,places)))
    }
    fn rows_experts_gathered(&self,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,source:&V,order:&[f32],places:&[f32])->spark_tp::Result<Option<V>> {
        // What the device form needs: gate and up lists over the same rows, one row a token in `source`, and for every
        // row the lists read a whole row number inside `source`.
        self.asked.set(self.asked.get()+1);
        assert!(second.is_some(),"only the gate and up step reads rows that were not copied");
        let n=first.iter().map(|v|v.1).sum::<usize>();let (_,width)=self.extent(first[0].0);
        assert_eq!(order.len(),n,"one table word a row the lists read");
        assert!(width>0 && source.len().is_multiple_of(width),"the source holds whole rows");let tokens=source.len()/width;
        assert!(n>tokens,"a token's row is read once a pick: the copy this form saves is larger than the source");
        assert!(order.iter().all(|&row|row>=0.0 && row.fract()==0.0 && (row as usize)<tokens),"every table word names a source row");
        let shift=match self.form {Form::Copies=>return Ok(None),Form::Gathers=>0,Form::Shifted=>1};
        self.answered.set(self.answered.get()+1);
        // Row r of the lists is source row order[r]: the words the copy would hold, never stored by the caller.
        let mut rows=Vec::with_capacity(n*width);
        for &row in order {let row=(row as usize+shift)%tokens;rows.extend_from_slice(&source[row*width..(row+1)*width]);}
        Ok(Some(self.both(first,second,&rows,places)))
    }
    fn prompt(&self,s:RowShape,x:[Option<&V>;5],w:&[f32])->spark_tp::Result<V> {
        if s.op==Row::PermuteRows {self.permutes.set(self.permutes.get()+1);}
        Host.prompt(s,x,w)
    }
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Self::Value,_:usize)->spark_tp::Result<Option<Vec<u32>>> {Ok(None)}
    fn latent_absorption_chats(&self,name:&str,s:RowShape,g:&[spark_tp::bounded_decoder::ChatGather<'_,Self::Value>],stride:usize,
        q:&Self::Value,n:&Self::Value)->spark_tp::Result<Option<Self::Value>> {Host.latent_absorption_chats(name,s,g,stride,q,n)}
    fn select_known(&self,ids:&V,count:&V,row:&[f32])->spark_tp::Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->spark_tp::Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,c:&CheckpointAdmission,_:usize)->spark_tp::Result<()> {self.prepare(c)}
    fn upload(&self,v:&[f32])->spark_tp::Result<V> {Host.upload(v)}
    fn read(&self,v:&V)->spark_tp::Result<V> {Host.read(v)}
    fn words(&self,v:&V)->usize {v.len()}
    fn range(&self,v:&V,start:usize,len:usize)->spark_tp::Result<V> {Host.range(v,start,len)}
    fn join(&self,v:&[&V])->spark_tp::Result<V> {Host.join(v)}
    fn zeros(&self,n:usize)->spark_tp::Result<V> {Host.zeros(n)}
    fn operation(&self,s:Shape,x:[Option<&V>;4],w:&[f32])->spark_tp::Result<V> {Host.operation(s,x,w)}
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,w:&[f32])->spark_tp::Result<V> {Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V)->spark_tp::Result<(Self::Pass,V)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&V)->spark_tp::Result<V> {Host.finish(p,x)}
    fn mx(&self,p:&spark_tp::mxfp8_projection::Mxfp8Projection<'_>,x:&V)->spark_tp::Result<V> {Host.mx(p,x)}
    fn round(&self,x:&V)->spark_tp::Result<V> {Host.round(x)}
    fn mean(&self,x:&V,s:u32,h:u32)->spark_tp::Result<V> {Host.mean(x,s,h)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,m:usize,w:&[f32])->spark_tp::Result<V> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V,m:usize)->spark_tp::Result<(Self::Pass,V)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&V,m:usize)->spark_tp::Result<V> {Host.rows_finish(p,x,m)}
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,s:Shape,x:[Option<&V>;4],reserve:usize,expected:f32)->spark_tp::Result<V> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:Shape,q:&V,l:&V,n:&V,slots:usize)->spark_tp::Result<V> {Host.latent_absorption(name,s,q,l,n,slots)}
    fn latent_absorption_rows(&self,name:&str,s:RowShape,q:&V,l:&V,n:&V)->spark_tp::Result<V> {Host.latent_absorption_rows(name,s,q,l,n)}
}
fn bits(rows:&[Vec<f32>])->Vec<Vec<u32>> {rows.iter().map(|row|row.iter().map(|v|v.to_bits()).collect()).collect()}
/// Two prompt groups of one chat (5 rows from an empty state, then 3 rows behind them): every row's logits.
fn two_groups(compute:&Gathering<'_>)->Vec<Vec<u32>> {
    let decoder=Decoder::new(compute.c,compute,8).unwrap();
    let (first,state)=decoder.prefill_rows(&[1,7,2,3,4],&decoder.zero_state().unwrap(),HeadRows::All).unwrap();
    let (second,next)=decoder.prefill_rows(&[5,6,1],&state,HeadRows::All).unwrap();assert_eq!(next.position,8);
    let mut all=bits(&first);all.extend(bits(&second));assert_eq!(all.len(),8);all
}
#[test]
fn a_prompt_group_reads_its_gate_rows_through_the_table_or_copies_them() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let d=c.admission.description();
    let routed=(0..d.layers.len()).filter(|&layer|LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).unwrap().steps.iter().any(|step|step.kind==Kind::ExpertSum)).count();
    assert!(routed>0);let groups=2;
    // The group that copies: the compute refuses the gathered form, so every routed layer of every group copies its
    // gate rows and runs gate and up on the copy. It is asked once a routed layer and group, before any copy.
    let copies=Gathering::new(&c.admission,Form::Copies);let expected=two_groups(&copies);
    assert_eq!((copies.asked.get(),copies.answered.get(),copies.copied.get()),(groups*routed,0,groups*routed));
    assert!((1..expected.len()).all(|row|expected[row]!=expected[0]),"the rows of the group differ");
    // The gathered form answered as the device answers it: ONE answer a routed layer and group, no gate and up pass on
    // copied rows, exactly one `PermuteRows` fewer a routed layer and group (the gate copy), and every row keeps its bits.
    let gathers=Gathering::new(&c.admission,Form::Gathers);
    assert_eq!(two_groups(&gathers),expected,"a group that reads its gate rows through the table must keep every row's bits");
    assert_eq!((gathers.asked.get(),gathers.answered.get(),gathers.copied.get()),(groups*routed,groups*routed,0));
    assert_eq!(copies.permutes.get()-gathers.permutes.get(),groups*routed,"the gate copy, and only it, is not made");
    // Wrong on purpose: the table shifted by one row changes the rows, so the table decides what a row reads.
    let shifted=Gathering::new(&c.admission,Form::Shifted);let wrong=two_groups(&shifted);
    assert_eq!(shifted.answered.get(),groups*routed);
    assert!(wrong.iter().zip(&expected).any(|(got,want)|got!=want),"a wrong table must change the output");
}
