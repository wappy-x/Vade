//! Device projections, checks and picks for speculative proposals.
use super::*;

impl<'b,'d> crate::loop_decoder::draft::Lanes for Device<'b,'d> {
    fn draft_sum(&self,x:&Value<'b,'d>,y:&Value<'b,'d>)->Result<Value<'b,'d>> {
        use spark_gpu_probe::draft_ops::DraftOp::AddRound;
        let words=x.words;
        if !self.backend.resident_queue_active() || words==0 || words>1<<24 || words!=y.words
            || x.vector.len()!=words || y.vector.len()!=words {return self.bf16_sum(x,y);}
        self.ready(x)?;self.ready(y)?;
        let out=self.backend.draft_lanes(AddRound(words),[Some(&x.vector),Some(&y.vector),None])?;
        self.value(out,words)
    }
    fn draft_norm(&self,x:&Value<'b,'d>,rows:usize,width:usize,eps:f32,weight:&Value<'b,'d>)->Result<Value<'b,'d>> {
        self.ready(x)?;self.ready(weight)?;
        need(rows.checked_mul(width)==Some(x.words) && weight.words==width,"draft norm shape differs")?;
        if self.backend.resident_launches_active()
            && !x.vector.is_segmented() && !weight.vector.is_segmented()
            && spark_gpu_probe::fast_ops::norm_admitted(rows,width,true,spark_gpu_probe::fast_ops::NORM_THREADS)
        {
            let mut out=self.backend.vector(x.words)?;
            self.backend.fast_norm_into(rows,width,true,eps,false,&x.vector,Some(&weight.vector),&mut out)?;
            return self.value(out,x.words);
        }
        let mut s=shape(Op::RoundedNorm);
        s.heads=rows.try_into().map_err(|_|"draft norm rows exceed ABI")?;s.dim=width.try_into().map_err(|_|"draft norm width exceeds ABI")?;
        s.taps=1;s.epsilon=eps;
        let out=self.backend.causal_operation(s,[Some(&x.vector),None,None,None],Some(&weight.vector))?;
        self.value(out,x.words)
    }
    fn lanes(&self,op:spark_gpu_probe::draft_ops::DraftOp,x:&Value<'b,'d>,y:Option<&Value<'b,'d>>)->Result<Value<'b,'d>> {
        self.ready(x)?;if let Some(y)=y {self.ready(y)?;}
        let out=self.backend.draft_lanes(op,[Some(&x.vector),y.map(|v|v.vector.as_ref()),None])?;
        let words=out.len();self.value(out,words)
    }
    fn lanes3(&self,op:spark_gpu_probe::draft_ops::DraftOp,x:&Value<'b,'d>,y:&Value<'b,'d>,z:&Value<'b,'d>)->Result<Value<'b,'d>> {
        self.ready(x)?;self.ready(y)?;self.ready(z)?;
        let out=self.backend.draft_lanes(op,[Some(&x.vector),Some(&y.vector),Some(&z.vector)])?;
        let words=out.len();self.value(out,words)
    }
    fn greedy_on_device(&self,logits:&Value<'b,'d>)->Result<Value<'b,'d>> {
        use spark_gpu_probe::draft_ops::DraftOp::{PickFinish,PickPartials};
        let partials=self.lanes(PickPartials(logits.words),logits,None)?;
        self.lanes(PickFinish(partials.words/3),&partials,None)
    }
}

impl<'b,'d> Device<'b,'d> {
    pub(crate) fn combined_head_fold(&self,op:spark_gpu_probe::draft_ops::DraftOp,head:&DeviceVector<'d>,fold:&DeviceVector<'d>,input:&Value<'b,'d>)->Result<Value<'b,'d>> {
        self.ready(input)?;
        let out=self.backend.draft_lanes(op,[Some(head),Some(fold),Some(&input.vector)])?;
        let words=out.len();self.value(out,words)
    }
    pub(crate) fn panel_projection(&self,op:spark_gpu_probe::draft_ops::DraftOp,weight:&DeviceVector<'d>,x:&Value<'b,'d>,base:Option<&Value<'b,'d>>)->Result<Value<'b,'d>> {
        self.ready(x)?;
        if let Some(v)=base {self.ready(v)?;}
        let out=self.backend.draft_lanes(op,[Some(weight),Some(&x.vector),base.map(|v|v.vector.as_ref())])?;
        let words=out.len();self.value(out,words)
    }
}
