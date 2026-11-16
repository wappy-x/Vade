//! GROUP FOLD: in a prompt group the grouped down step of a routed layer waits for its fold, the shared row stays on its
//! rank, and the compute gets the down step and the fold as one stage (`Compute::rows_experts_fold`): the device places
//! and folds each rank's own rows and exchanges the folded rows, one row a token. A compute that refuses the stage gets
//! the down step and the fold as they were. Every form must give the bits of the group that does not fold.
use spark_tp::bounded_decoder::{Compute,Decoder,HeadRows,Host};
use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan,Topology}},safetensors::Dtype};
use spark_gpu_probe::{bf16::rounded,causal_ops::{Op,Shape},prompt_ops::{Op as Row,Shape as RowShape}};
use std::{cell::{Cell,RefCell},collections::BTreeSet};
#[allow(dead_code,reason="the shared fixture has more forms than this test uses")]
#[path="support/attention_checkpoint.rs"] mod fixture;
type V=Vec<f32>;
/// What the compute does with a fold stage.
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
enum Stage {
    /// It folds as the device does: the down rows, then the wrapped fold of them with the kept shared rows.
    Folds,
    /// It refuses (a shape outside the swept kernels): the caller keeps the two steps.
    Refuses,
    /// It answers with zeros: wrong on purpose, to show that the answer is the layer's output.
    Zeros,
}
/// The host compute as a pair's device for a prompt group. The grouped expert passes are answered (`rows_experts`), as
/// the device answers them, with a stand-in for the expert products: finished BF16 words that depend on the source row,
/// on the expert and on the lane, so a row placed for the wrong (token, pick) or made by the wrong expert cannot pass.
/// The shared expert's down projection is a stand-in too (`shared` names it), so the compute can keep it "on its rank":
/// a host compute is one rank, so its own rows are the whole rows. Every form below uses the same stand-ins; the group
/// that does not fold is the reference.
struct Grouping<'c> {c:&'c CheckpointAdmission,shared:BTreeSet<String>,
    /// `folds_groups`: the compute folds the rows of a prompt group.
    folds:bool,
    /// The shared row stays on its rank (`rows_project_local`).
    local:bool,
    stage:Stage,
    /// Times a pass asked `folds_groups`: the one flag a routed layer of a prompt group pays.
    flags:Cell<usize>,
    /// Shared rows kept on their rank; shared rows made with their exchange (`rows_project`).
    kept:Cell<usize>,gathered:Cell<usize>,
    /// Fold stages asked, and their rows.
    stages:Cell<usize>,rows:Cell<usize>,
    /// Down steps the caller ran apart from their fold (`rows_experts` without a second list), and folds it ran itself.
    downs:Cell<usize>,folds_apart:Cell<usize>,
    /// LOOKAHEAD: the compute leaves the link step of the routed gate and up open (`rows_experts_open`).
    ahead:bool,
    /// The middle of the open exchange: what its close returns.
    open:RefCell<Option<V>>,
    /// Opens, closes, and closes that were told the work queued ahead failed.
    opens:Cell<usize>,closes:Cell<usize>,failed:Cell<usize>,
    /// What the caller queued between an open and its close, in order.
    between:RefCell<Vec<&'static str>>,
    /// Wrong on purpose: the close answers zeros. Fails on purpose: the local rows fail while an exchange is open.
    zero_close:bool,fails:bool,
    /// GATHERED ROWS under LOOKAHEAD: the compute opens the exchange on rows it reads through the expert order table
    /// (`rows_experts_open_gathered`), as the device does. `shifted`: wrong on purpose, the table read one row off.
    gathers:bool,shifted:bool,
    /// Opens on rows read through the table, and every `PermuteRows` the caller ran.
    opens_gathered:Cell<usize>,permutes:Cell<usize>,
    /// WIDE COHORT counters coexist with the independent LOOKAHEAD fixture.
    wide:u8,shared_gates:Cell<[usize;2]>,shared_folds:Cell<[usize;2]>,lone_gates:Cell<usize>}
impl<'c> Grouping<'c> {
    fn new(c:&'c CheckpointAdmission,shared:&BTreeSet<String>,folds:bool,local:bool,stage:Stage)->Self {
        Self {c,shared:shared.clone(),folds,local,stage,flags:Cell::new(0),kept:Cell::new(0),gathered:Cell::new(0),stages:Cell::new(0),rows:Cell::new(0),downs:Cell::new(0),folds_apart:Cell::new(0),
            ahead:false,open:RefCell::new(None),opens:Cell::new(0),closes:Cell::new(0),failed:Cell::new(0),between:RefCell::new(Vec::new()),zero_close:false,fails:false,gathers:false,shifted:false,opens_gathered:Cell::new(0),permutes:Cell::new(0),wide:0,shared_gates:Cell::new([0;2]),shared_folds:Cell::new([0;2]),lone_gates:Cell::new(0)}
    }
    fn sharing(mut self,wide:u8)->Self {self.wide=wide;self}
    /// The same compute, which leaves the link step of the routed gate and up open, as the device does for a group it folds.
    fn ahead(mut self)->Self {self.ahead=true;self}
    /// The same compute, which also opens that exchange on rows it reads through the expert order table.
    fn gathers(mut self)->Self {self.ahead=true;self.gathers=true;self}
    /// No exchange starts while one is open: the device fails the transport for it.
    fn none_open(&self,what:&str) {assert!(self.open.borrow().is_none(),"{what} while an exchange is open");}
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
    /// The stand-in for the shared expert's down projection of `m` rows.
    fn shared_rows(&self,name:&str,x:&V,m:usize)->V {
        let (out,width)=self.extent(name);assert_eq!(x.len(),m*width);
        (0..m*out).map(|at|rounded(x[at/out*width+at%out%width]*0.5+(at%out%5) as f32/64.0)).collect()
    }
}
impl Compute for Grouping<'_> {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn rows_experts(&self,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,source:&V,places:&[f32])->spark_tp::Result<Option<V>> {
        self.none_open("an expert exchange");
        let a=self.experts(first,source,places);
        Ok(Some(match second {
            Some((list,_))=>{let c=self.experts(list,source,places);a.iter().zip(&c).map(|(a,c)|rounded(a*0.5+c*0.25)).collect()},
            None=>{self.downs.set(self.downs.get()+1);a}}))
    }
    fn rows_experts_gathered(&self,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,source:&V,order:&[f32],places:&[f32])->spark_tp::Result<Option<V>> {
        if self.wide==0 {return Ok(None);}
        // The rows the lists read are the rows of `source` the order table names: the copy `PermuteRows` would make.
        let (_,width)=self.extent(first[0].0);assert!(source.len().is_multiple_of(width));
        let copied=order.iter().flat_map(|&row|{let row=row as usize;assert!(row<source.len()/width);source[row*width..(row+1)*width].iter().copied()}).collect::<V>();
        self.lone_gates.set(self.lone_gates.get()+1);
        self.rows_experts(first,second,&copied,places)
    }
    fn rows_experts_gathered_wide(&self,sets:&[(&[(&str,usize)],&[(&str,usize)],f32,&V,&[f32],&[f32])])->spark_tp::Result<Option<Vec<V>>> {
        if self.wide<2 {return Ok(None);}
        assert!(sets.len()>=2,"the shared form is asked for two or more runs");
        let before=self.lone_gates.get();
        let made=sets.iter().map(|&(first,second,bound,source,order,places)|Ok(self.rows_experts_gathered(first,Some((second,bound)),source,order,places)?.expect("the stand-in takes every gate"))).collect::<spark_tp::Result<Vec<V>>>()?;
        self.lone_gates.set(before);let [calls,rows]=self.shared_gates.get();self.shared_gates.set([calls+1,rows+sets.len()]);
        Ok(Some(if self.wide==3 {made.into_iter().map(|rows|vec![0.0;rows.len()]).collect()} else {made}))
    }
    fn rows_experts_fold_wide(&self,sets:&[(&[(&str,usize)],&V,&[f32],&V,&V,Option<&V>,Shape,usize)])->spark_tp::Result<Option<Vec<V>>> {
        if self.wide<2 {return Ok(None);}
        assert!(sets.len()>=2,"the shared form is asked for two or more runs");
        let (stages,rows)=(self.stages.get(),self.rows.get());
        let made=sets.iter().map(|&(first,source,places,ids,weights,shared,fold,m)|Ok(self.rows_experts_fold(first,source,places,ids,weights,shared,fold,m)?.expect("the stand-in folds every run"))).collect::<spark_tp::Result<Vec<V>>>()?;
        self.stages.set(stages);self.rows.set(rows);let [calls,runs]=self.shared_folds.get();self.shared_folds.set([calls+1,runs+sets.len()]);
        Ok(Some(made))
    }
    fn rows_experts_open(&self,first:&[(&str,usize)],second:(&[(&str,usize)],f32),source:&V,places:&[f32])->spark_tp::Result<bool> {
        if !self.ahead || !self.folds {return Ok(false);}
        let middle=self.rows_experts(first,Some(second),source,places)?.expect("the stand-in answers every list");
        *self.open.borrow_mut()=Some(middle);self.opens.set(self.opens.get()+1);Ok(true)
    }
    fn rows_experts_open_gathered(&self,first:&[(&str,usize)],second:(&[(&str,usize)],f32),source:&V,order:&[f32],places:&[f32])->spark_tp::Result<bool> {
        if !self.gathers || !self.ahead || !self.folds {return Ok(false);}
        // What the device form needs: one row a token in `source`, and for every row the lists read a row inside it.
        let n=first.iter().map(|v|v.1).sum::<usize>();let (_,width)=self.extent(first[0].0);
        assert_eq!(order.len(),n,"one table word a row the lists read");
        assert!(width>0 && source.len().is_multiple_of(width),"the source holds whole rows");let tokens=source.len()/width;
        assert!(n>tokens && order.iter().all(|&row|row>=0.0 && row.fract()==0.0 && (row as usize)<tokens),"every table word names a source row");
        // Row r of the lists is source row order[r]: the words the copy would hold, never stored by the caller.
        let mut rows=Vec::with_capacity(n*width);
        for &row in order {let row=(row as usize+usize::from(self.shifted))%tokens;rows.extend_from_slice(&source[row*width..(row+1)*width]);}
        let open=self.rows_experts_open(first,second,&rows,places)?;
        if open {self.opens_gathered.set(self.opens_gathered.get()+1);}
        Ok(open)
    }
    fn rows_experts_close(&self,failed:bool)->spark_tp::Result<V> {
        let middle=self.open.borrow_mut().take().expect("a close follows its open");self.closes.set(self.closes.get()+1);
        if failed {self.failed.set(self.failed.get()+1);return Err("the exchange failed with the work queued ahead".into());}
        Ok(if self.zero_close {vec![0.0;middle.len()]} else {middle})
    }
    fn rows_project(&self,_:&CheckpointAdmission,name:&str,x:&V,m:usize,_:bool)->spark_tp::Result<Option<V>> {
        // Not a shared row: the caller makes the projection on the host compute, which is not an exchange of this stand-in.
        if !self.shared.contains(name) {return Ok(None);}
        self.none_open("a projection's exchange");
        self.gathered.set(self.gathered.get()+1);Ok(Some(self.shared_rows(name,x,m)))
    }
    fn rows_project_local(&self,_:&CheckpointAdmission,name:&str,x:&V,m:usize,_:bool)->spark_tp::Result<Option<V>> {
        if !self.folds || !self.local {return Ok(None);}
        assert!(self.shared.contains(name),"only the shared row of a fold stays on its rank: {name}");
        if self.open.borrow().is_some() {if self.fails {return Err("the local rows failed".into());}self.between.borrow_mut().push("local");}
        self.kept.set(self.kept.get()+1);Ok(Some(self.shared_rows(name,x,m)))
    }
    fn folds_groups(&self,m:usize)->bool {assert!(m>0);self.flags.set(self.flags.get()+1);self.folds}
    fn rows_experts_fold(&self,first:&[(&str,usize)],source:&V,places:&[f32],ids:&V,weights:&V,shared:Option<&V>,fold:Shape,m:usize)->spark_tp::Result<Option<V>> {
        // What the device form needs: the down step's experts and their rows, the middle in the grouped order, the place
        // of every (token, pick), every token's picks and weights, this rank's rows of the shared row, and the fold shape
        // of one token's whole output.
        self.none_open("a fold stage");
        let picks=fold.rows as usize;assert!(self.folds && picks>0 && m>0);assert_eq!(fold.op,Op::ExpertFold);
        assert_eq!(first.iter().map(|v|v.1).sum::<usize>(),m*picks,"the experts' rows are every (token, pick)");
        assert_eq!(places.len(),3*m*picks,"one place a (token, pick)");
        assert_eq!((ids.len(),weights.len()),(m*picks,m*picks));
        let (out,width)=self.extent(first[0].0);
        assert_eq!(source.len(),m*picks*width,"the middle is one row a (token, pick)");
        assert_eq!(fold.cols as usize,out,"the fold makes one whole output row a token");
        assert_eq!(fold.taps!=0,shared.is_some());
        let shared=shared.expect("the fixture's fold reads the shared row");
        assert_eq!(shared.len(),m*out,"the kept shared row is one row a token");
        self.stages.set(self.stages.get()+1);self.rows.set(self.rows.get()+m);
        match self.stage {
            Stage::Refuses=>Ok(None),
            Stage::Zeros=>Ok(Some(vec![0.0;m*out])),
            Stage::Folds=>{let down=self.experts(first,source,places);
                Ok(Some(Host.prompt(RowShape::wrapped(fold,m).unwrap(),[Some(ids),Some(weights),Some(&down),Some(shared),None],&[])?))},
        }
    }
    fn prompt(&self,s:RowShape,x:[Option<&V>;5],w:&[f32])->spark_tp::Result<V> {
        if s.op==Row::Wrapped && s.inner.op==Op::ExpertFold {self.folds_apart.set(self.folds_apart.get()+1);}
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
    fn operation(&self,s:Shape,x:[Option<&V>;4],w:&[f32])->spark_tp::Result<V> {
        if self.open.borrow().is_some() {self.between.borrow_mut().push(if s.op==Op::SwiGlu {"swiglu"} else {"other"});}
        Host.operation(s,x,w)
    }
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
fn two_groups(compute:&Grouping<'_>)->Vec<Vec<u32>> {
    let decoder=Decoder::new(compute.c,compute,8).unwrap();
    let (first,state)=decoder.prefill_rows(&[1,7,2,3,4],&decoder.zero_state().unwrap(),HeadRows::All).unwrap();
    let (second,next)=decoder.prefill_rows(&[5,6,1],&state,HeadRows::All).unwrap();assert_eq!(next.position,8);
    let mut all=bits(&first);all.extend(bits(&second));assert_eq!(all.len(),8);all
}
/// The shared expert's down matrix of every routed layer (the last input of the layer's fold), and the routed layers.
fn shared_downs(c:&CheckpointAdmission)->(BTreeSet<String>,usize) {
    let d=c.description();let mut shared=BTreeSet::new();let mut routed=0usize;
    for layer in 0..d.layers.len() {
        let plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).unwrap();
        let Some(sum)=plan.steps.iter().find(|step|step.kind==Kind::ExpertSum) else {continue;};
        routed+=1;assert_eq!(sum.inputs.len(),4,"the fixture's fold reads the shared row");
        let down=plan.steps.iter().find(|step|step.kind==Kind::Project && step.outputs[0].name==sum.inputs[3]).expect("the shared row is one projection");
        shared.insert(down.weights[0].name.clone());
    }
    assert!(routed>0 && shared.len()==routed);(shared,routed)
}
/// LOOKAHEAD: a prompt group whose compute leaves the link step of the routed gate and up open queues, between the
/// open and the close, the shared expert's SwiGLU and this rank's rows of its down projection, and nothing else. Every
/// row keeps the bits of the group that does not fold. The close always follows the open, also when the work queued
/// ahead fails. What this does NOT run: the device's stream mark, link step and buffers. Those need a pair.
#[test]
fn a_prompt_group_queues_the_shared_steps_behind_its_open_gate_and_up_exchange() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let (shared,routed)=shared_downs(&c.admission);let groups=2;
    let expected=two_groups(&Grouping::new(&c.admission,&shared,false,true,Stage::Folds));
    // The exchange left open: one open and one close a routed layer of a group, and between them the SwiGLU, then the
    // local rows. The fold stage, the kept shared rows and the bits are those of the folded group.
    let ahead=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).ahead();
    assert_eq!(two_groups(&ahead),expected,"a group with work queued ahead must keep every row's bits");
    assert_eq!((ahead.opens.get(),ahead.closes.get(),ahead.failed.get()),(groups*routed,groups*routed,0));
    assert_eq!(*ahead.between.borrow(),["swiglu","local"].repeat(groups*routed),"only the shared expert's two steps are queued ahead, in their order");
    assert_eq!((ahead.stages.get(),ahead.rows.get(),ahead.kept.get(),ahead.flags.get()),(groups*routed,8*routed,groups*routed,groups*routed),"the rows made ahead are the fold's shared rows: none is made twice");
    assert_eq!((ahead.downs.get(),ahead.folds_apart.get(),ahead.gathered.get()),(0,0,0));
    // A compute that keeps no rows local: the SwiGLU alone is queued ahead, and the projection keeps its exchange,
    // after the close (`none_open` in the stand-in would fail it between).
    let gathered=Grouping::new(&c.admission,&shared,true,false,Stage::Folds).ahead();
    assert_eq!(two_groups(&gathered),expected,"a group whose shared row was gathered must keep every row's bits");
    assert_eq!((gathered.opens.get(),gathered.closes.get()),(groups*routed,groups*routed));
    assert_eq!(*gathered.between.borrow(),["swiglu"].repeat(groups*routed));
    assert_eq!((gathered.stages.get(),gathered.kept.get(),gathered.gathered.get(),gathered.downs.get()),(0,0,groups*routed,groups*routed));
    // The stage refused after work was queued ahead: the shared row is made again with its exchange, as without the cut.
    let refused=Grouping::new(&c.admission,&shared,true,true,Stage::Refuses).ahead();
    assert_eq!(two_groups(&refused),expected,"a refused stage must keep every row's bits");
    assert_eq!((refused.opens.get(),refused.closes.get(),refused.kept.get(),refused.gathered.get()),(groups*routed,groups*routed,groups*routed,groups*routed));
    // A compute that does not open (it folds, as before the cut): no open, no close, nothing queued ahead.
    let closed=Grouping::new(&c.admission,&shared,true,true,Stage::Folds);
    assert_eq!(two_groups(&closed),expected);
    assert_eq!((closed.opens.get(),closed.closes.get(),closed.between.borrow().len()),(0,0,0));
    // Wrong on purpose: a close that answers zeros changes the rows, so the closed value is the layer's middle.
    let mut zeros=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).ahead();zeros.zero_close=true;
    let wrong=two_groups(&zeros);
    assert!(wrong.iter().zip(&expected).all(|(got,want)|got!=want),"the closed middle was not used");
    // The work queued ahead fails: the close still runs, told so, the group fails with the work's own error, and no
    // exchange is left open.
    let mut failing=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).ahead();failing.fails=true;
    let decoder=Decoder::new(failing.c,&failing,8).unwrap();
    let error=decoder.prefill_rows(&[1,7,2,3,4],&decoder.zero_state().unwrap(),HeadRows::All).err().expect("the group fails");
    assert_eq!(error,"the local rows failed");
    assert_eq!((failing.opens.get(),failing.closes.get(),failing.failed.get()),(1,1,1));assert!(failing.open.borrow().is_none());
    // A pass that keeps its rows for a verdict is not a prompt group: it never opens.
    let kept=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).ahead();
    let decoder=Decoder::new(kept.c,&kept,8).unwrap();
    let (rows,_,_)=decoder.prefill_rows_open(&[1,7,2,3,4],&decoder.zero_state().unwrap(),HeadRows::All).unwrap();
    assert_eq!(bits(&rows),expected[..5]);assert_eq!((kept.opens.get(),kept.closes.get()),(0,0));
}
/// GATHERED ROWS under LOOKAHEAD: a compute that opens the gate-and-up exchange on rows it reads through the expert
/// order table gets that form first. The gate copy (`PermuteRows`) is then not made, the same two steps are queued
/// between the open and the close, and every row keeps its bits. A compute that opens only on copied rows keeps the copy.
#[test]
fn a_prompt_group_opens_its_gate_and_up_exchange_on_rows_read_through_the_table() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let (shared,routed)=shared_downs(&c.admission);let groups=2;
    let expected=two_groups(&Grouping::new(&c.admission,&shared,false,true,Stage::Folds));
    // The reference of this test: the exchange left open on COPIED rows (seat LAP's form).
    let copied=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).ahead();
    assert_eq!(two_groups(&copied),expected);
    assert_eq!((copied.opens.get(),copied.opens_gathered.get(),copied.closes.get()),(groups*routed,0,groups*routed));
    // Both together: one open on rows read through the table and one close a routed layer of a group; between them
    // the SwiGLU, then the local rows; exactly the gate copy is not made; every row keeps its bits.
    let both=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).gathers();
    assert_eq!(two_groups(&both),expected,"a group that opens on rows read through the table must keep every row's bits");
    assert_eq!((both.opens.get(),both.opens_gathered.get(),both.closes.get(),both.failed.get()),(groups*routed,groups*routed,groups*routed,0));
    assert_eq!(*both.between.borrow(),["swiglu","local"].repeat(groups*routed),"only the shared expert's two steps are queued ahead, in their order");
    assert_eq!((both.stages.get(),both.kept.get(),both.downs.get(),both.folds_apart.get(),both.gathered.get()),(groups*routed,groups*routed,0,0,0));
    assert_eq!(copied.permutes.get()-both.permutes.get(),groups*routed,"the gate copy, and only it, is not made");
    // Wrong on purpose: the table read one row off changes the rows, so the table decides what a row reads.
    let mut shifted=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).gathers();shifted.shifted=true;
    let wrong=two_groups(&shifted);assert_eq!(shifted.opens_gathered.get(),groups*routed);
    assert!(wrong.iter().zip(&expected).any(|(got,want)|got!=want),"a wrong table must change the output");
    // The work queued ahead fails: the close still runs, told so, and no exchange is left open.
    let mut failing=Grouping::new(&c.admission,&shared,true,true,Stage::Folds).gathers();failing.fails=true;
    let decoder=Decoder::new(failing.c,&failing,8).unwrap();
    let error=decoder.prefill_rows(&[1,7,2,3,4],&decoder.zero_state().unwrap(),HeadRows::All).err().expect("the group fails");
    assert_eq!(error,"the local rows failed");
    assert_eq!((failing.opens_gathered.get(),failing.closes.get(),failing.failed.get()),(1,1,1));assert!(failing.open.borrow().is_none());
}
#[test]
fn a_prompt_group_folds_its_grouped_down_rows_at_the_fold_step_or_keeps_both_steps() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let (shared,routed)=shared_downs(&c.admission);
    let groups=2;
    // The group that does not fold: the down step and the fold apart, the shared row with its exchange. One flag a routed layer.
    let apart=Grouping::new(&c.admission,&shared,false,true,Stage::Folds);let expected=two_groups(&apart);
    assert_eq!((apart.flags.get(),apart.stages.get(),apart.kept.get()),(groups*routed,0,0),"a group that does not fold asks one flag a routed layer and nothing else");
    assert_eq!((apart.downs.get(),apart.folds_apart.get(),apart.gathered.get()),(groups*routed,groups*routed,groups*routed));
    assert!((1..expected.len()).all(|row|expected[row]!=expected[0]),"the rows of the group differ");
    // The stage answered as the device answers it: ONE stage a routed layer with every row of the group, the shared row
    // kept on its rank (no exchange of it), no down step and no fold apart, and every row keeps its bits.
    let folded=Grouping::new(&c.admission,&shared,true,true,Stage::Folds);
    assert_eq!(two_groups(&folded),expected,"a folded group must keep every row's bits");
    assert_eq!((folded.stages.get(),folded.rows.get(),folded.kept.get(),folded.flags.get()),(groups*routed,8*routed,groups*routed,groups*routed));
    assert_eq!((folded.downs.get(),folded.folds_apart.get(),folded.gathered.get()),(0,0,0),"the stage is the down step and the fold; the shared row makes no exchange");
    // The stage refused: the shared row is made again with its exchange, then the down step and the fold as they were.
    let refused=Grouping::new(&c.admission,&shared,true,true,Stage::Refuses);
    assert_eq!(two_groups(&refused),expected,"a refused stage must keep every row's bits");
    assert_eq!((refused.stages.get(),refused.kept.get()),(groups*routed,groups*routed));
    assert_eq!((refused.downs.get(),refused.folds_apart.get(),refused.gathered.get()),(groups*routed,groups*routed,groups*routed));
    // The shared row did not stay on its rank (the compute refused that): no stage is asked, both steps run as they were.
    let gathered=Grouping::new(&c.admission,&shared,true,false,Stage::Folds);
    assert_eq!(two_groups(&gathered),expected,"a group whose shared row was gathered must keep every row's bits");
    assert_eq!((gathered.stages.get(),gathered.kept.get(),gathered.flags.get()),(0,0,groups*routed));
    assert_eq!((gathered.downs.get(),gathered.folds_apart.get(),gathered.gathered.get()),(groups*routed,groups*routed,groups*routed));
    // Wrong on purpose: a stage answered with zeros changes the rows, so the stage's answer is the layer's output.
    let zeros=Grouping::new(&c.admission,&shared,true,true,Stage::Zeros);let wrong=two_groups(&zeros);
    assert_eq!((zeros.stages.get(),zeros.folds_apart.get()),(groups*routed,0));
    assert!(wrong.iter().zip(&expected).all(|(got,want)|got!=want),"the folded output was not used");
    // A pass that keeps its rows for a verdict (the check rows of drafting) is not a prompt group: it never asks the flag.
    let open=|compute:&Grouping<'_>| {
        let decoder=Decoder::new(compute.c,compute,8).unwrap();
        let (rows,_,_)=decoder.prefill_rows_open(&[1,7,2,3,4],&decoder.zero_state().unwrap(),HeadRows::All).unwrap();bits(&rows)
    };
    let kept=Grouping::new(&c.admission,&shared,true,true,Stage::Zeros);
    assert_eq!(open(&kept),expected[..5],"a pass with kept rows keeps the bits of the group that does not fold");
    assert_eq!((kept.flags.get(),kept.stages.get(),kept.kept.get()),(0,0,0),"a pass with kept rows pays nothing for the group fold");
}
/// WIDE COHORT: a prompt group above PROMPT_GROUP_MAX rows runs its routed half a run of at most that many rows at a
/// time, and asks the compute to share the expert product launches of all runs: ONE shared gate call and ONE shared fold
/// call a routed layer, each with every run as a row set that holds what the run's own call holds. The shared forms, the
/// forms of one run (a compute that refuses to share) and the same rows read as two groups give the same bits.
#[test]
fn the_runs_of_a_wide_group_share_the_expert_product_launches_and_keep_their_bits() {
    use spark_tp::bounded_decoder::PROMPT_GROUP_MAX;
    let rows=PROMPT_GROUP_MAX+24;let context=(rows+8) as u64;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,context);let d=c.admission.description();
    let mut shared=BTreeSet::new();let mut routed=0usize;
    for layer in 0..d.layers.len() {
        let plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).unwrap();
        let Some(sum)=plan.steps.iter().find(|step|step.kind==Kind::ExpertSum) else {continue;};
        routed+=1;
        let down=plan.steps.iter().find(|step|step.kind==Kind::Project && step.outputs[0].name==sum.inputs[3]).expect("the shared row is one projection");
        shared.insert(down.weights[0].name.clone());
    }
    assert!(routed>0);
    let tokens=(0..rows).map(|i|((i*7+3)%16) as u32).collect::<Vec<_>>();
    // The last row's logits and the state after the read, as bits.
    let read=|compute:&Grouping<'_>,width:usize| {
        let decoder=Decoder::new(compute.c,compute,context).unwrap();let mut state=decoder.zero_state().unwrap();let mut last=Vec::new();
        for chunk in tokens.chunks(width) {let (mut logits,next)=decoder.prefill_rows(chunk,&state,HeadRows::Last).unwrap();state=next;last=logits.pop().unwrap();}
        let mut all=vec![last.iter().map(|v|v.to_bits()).collect::<Vec<_>>()];
        for layer in &state.layers {for value in layer.values() {all.push(value.iter().map(|v|v.to_bits()).collect());}}
        all
    };
    let compute=|wide:u8|Grouping::new(&c.admission,&shared,true,true,Stage::Folds).sharing(wide);
    // Two groups (1,024 rows, then the rest): no shared call; one gate call and one fold stage a routed layer and group.
    let apart=compute(2);let expected=read(&apart,PROMPT_GROUP_MAX);
    assert_eq!((apart.shared_gates.get(),apart.shared_folds.get()),([0;2],[0;2]),"groups of 1,024 rows or fewer share nothing");
    assert_eq!((apart.lone_gates.get(),apart.stages.get()),(2*routed,2*routed));
    // One wide group: one shared gate call and one shared fold call a routed layer, two runs each; no call of one run.
    let wide=compute(2);
    assert_eq!(read(&wide,rows),expected,"a wide group that shares its expert launches must keep every bit");
    assert_eq!((wide.shared_gates.get(),wide.shared_folds.get()),([routed,2*routed],[routed,2*routed]));
    assert_eq!((wide.lone_gates.get(),wide.stages.get(),wide.downs.get(),wide.folds_apart.get()),(0,0,0,0),"every expert step of a wide group went through the shared forms");
    // A compute that does not share: each run takes its own gate call and its own fold stage, and every bit is kept.
    let lone=compute(1);
    assert_eq!(read(&lone,rows),expected,"a wide group whose compute does not share must keep every bit");
    assert_eq!((lone.shared_gates.get(),lone.shared_folds.get(),lone.lone_gates.get(),lone.stages.get()),([0;2],[0;2],2*routed,2*routed));
    // Wrong on purpose: shared gate calls answered with zeros change the rows, so the shared answer is what the runs use.
    let zeros=compute(3);
    assert_ne!(read(&zeros,rows)[0],expected[0],"the shared gate answer was not used");
    assert_eq!(zeros.shared_gates.get(),[routed,2*routed]);
}
/// GROUP FOLD twin (no device): the words of the device path's own order, from the host references of its kernels. Each
/// rank's expert rows lie as the expert launch group leaves them (an expert's rows contiguous, feature-major). Main:
/// both ranks' blocks are exchanged, the tile place of two parts sets every (token, pick) row whole, the wrapped fold
/// folds the whole rows with the gathered shared row. The cut: each rank places its own block (the tile place with ONE
/// part), folds its own rows with its own rows of the shared row, the folded rows are exchanged and `RankRows` sets a
/// token's rows of the two ranks side by side. One token, a few, a group of 16 and one past 64; picks whose slot order
/// is not their ID order; rows that differ by token, pick and rank. What this does NOT run: the device function, the
/// kernels and the exchange themselves. Those need a GPU: the list sweeps of `kernel-bench --reading-probe`
/// (`list-move`, `list-route-fold`) and the served answers on a pair.
#[test]
fn rows_placed_and_folded_on_each_rank_then_gathered_equal_the_gathered_rows_placed_and_folded() {
    use spark_gpu_probe::prompt_ops::{place_table,reference};
    let (picks,experts,ranks)=(8usize,288usize,2usize);
    let word=|seed:usize|rounded(((seed.wrapping_mul(2654435761))%2001) as f32/250.0-4.0);
    for (m,local) in [(1usize,32usize),(3,32),(16,64),(70,32)] {
        let (total,rows)=(local*ranks,m*picks);
        let ids=(0..rows).map(|at|((at/picks*29+at%picks*101+11)%experts) as f32).collect::<Vec<_>>();
        let weights=(0..rows).map(|at|word(at*7+3)/8.0).collect::<Vec<_>>();
        // Finished BF16 words, as the expert products store them: one whole row a (token, pick), and one shared row a token.
        let down=(0..rows*total).map(|at|word(at+17)).collect::<Vec<_>>();
        let shared=(0..m*total).map(|at|word(at*5+3)).collect::<Vec<_>>();
        for token in 0..m {let row=&ids[token*picks..(token+1)*picks];assert!(row.windows(2).any(|pair|pair[0]>pair[1]));
            assert!((0..picks).all(|pick|!row[..pick].contains(&row[pick])));}
        // The grouping of the decoder (`experts_group`): rows of one expert contiguous, in ascending expert order.
        let mut groups=std::collections::BTreeMap::<usize,Vec<usize>>::new();
        for (row,&id) in ids.iter().enumerate() {groups.entry(id as usize).or_default().push(row);}
        let mut placed=vec![0.0f32;3*rows];let mut blocks=vec![vec![f32::NAN;rows*local];ranks];let mut at=0usize;
        for members in groups.values() {let count=members.len();
            for (position,&row) in members.iter().enumerate() {placed[3*row..3*row+3].copy_from_slice(&[at as f32,count as f32,position as f32]);
                for (rank,block) in blocks.iter_mut().enumerate() {for feature in 0..local {block[at*local+feature*count+position]=down[row*total+rank*local+feature];}}}
            at+=count;}
        assert!(at==rows && blocks.iter().all(|block|block.iter().all(|v|v.is_finite())));
        let table=place_table(&placed).expect("a real expert grouping has a tile table");
        let place=|parts:usize| {let mut s=RowShape::new(Row::ExpertPlaceTile,1).unwrap();s.rows=rows as u32;s.cols=(local*parts) as u32;s.dim=local as u32;s.taps=parts as u32;s.slots=(rows*local) as u32;s};
        let fold=|cols:usize|RowShape::wrapped(Shape {op:Op::ExpertFold,rows:picks as u32,cols:cols as u32,taps:1,heads:0,dim:0,epsilon:0.0,bound:0.0},m).unwrap();
        // Main: exchange both ranks' rows, place them whole, fold.
        let exchange=blocks.concat();
        let whole=reference(RowShape {first:1,..place(ranks)},[&exchange,&table,&[],&[],&[]],&[]).unwrap();
        assert!(whole.iter().map(|v|v.to_bits()).eq(down.iter().map(|v|v.to_bits())),"m={m}: the placed rows are the (token, pick) rows");
        let expected=reference(fold(total),[&ids,&weights,&whole,&shared,&[]],&[]).unwrap();assert_eq!(expected.len(),m*total);
        // The cut: each rank places and folds its own rows, then one exchange of the folded rows.
        let folded=|swap:bool|(0..ranks).map(|rank| {
            let own=reference(place(1),[&blocks[rank],&table,&[],&[],&[]],&[]).unwrap();assert_eq!(own.len(),rows*local);
            let from=if swap {ranks-1-rank} else {rank};
            let plane=(0..m).flat_map(|token|shared[token*total+from*local..token*total+(from+1)*local].iter().copied()).collect::<Vec<_>>();
            reference(fold(local),[&ids,&weights,&own,&plane,&[]],&[]).unwrap()
        }).collect::<Vec<_>>().concat();
        let gather=|blocks:&[f32]|reference(RowShape::rank_rows(m,local,ranks,false).unwrap(),[blocks,&[],&[],&[],&[]],&[]).unwrap();
        let got=gather(&folded(false));
        assert!(expected.iter().map(|v|v.to_bits()).eq(got.iter().map(|v|v.to_bits())),"m={m} local={local}: fold then gather differs from gather then fold");
        // Wrong on purpose: the other rank's rows of the shared row cannot pass.
        let wrong=gather(&folded(true));
        assert!(!expected.iter().map(|v|v.to_bits()).eq(wrong.iter().map(|v|v.to_bits())),"m={m}: the wrong shared rows passed");
        if m>1 {assert!((1..m).all(|token|expected[..total]!=expected[token*total..(token+1)*total]));}
    }
}
