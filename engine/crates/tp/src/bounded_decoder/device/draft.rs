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
        if self.backend.inkling_image_present()
            && !x.vector.is_segmented() && !weight.vector.is_segmented()
            && spark_gpu_probe::fast_ops::norm_admitted(rows,width,true,spark_gpu_probe::fast_ops::NORM_THREADS)
        {
            let mut out=self.backend.vector(x.words)?;
            self.backend.draft_fast_norm_into(rows,width,eps,rows>=64,&x.vector,&weight.vector,&mut out)?;
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
    fn flash_attn_batch(
        &self,q:&Value<'b,'d>,k_new:&Value<'b,'d>,v_new:&Value<'b,'d>,
        chats:&[(&Value<'b,'d>,&Value<'b,'d>,usize,usize,usize)],
        heads:usize,kv:usize,width:usize,q_rows:usize,new_rows:usize,
        kv_bf16:bool,k_new_bf16:bool,causal_base:usize,row_off:usize,
    )->Result<Value<'b,'d>> {
        self.ready(q)?;self.ready(k_new)?;self.ready(v_new)?;
        let mut dev_chats=Vec::with_capacity(chats.len());
        for &(ck,cv,rel_start,hist_keys,splits_b) in chats {
            self.ready(ck)?;self.ready(cv)?;
            dev_chats.push((ck.vector.as_ref(),cv.vector.as_ref(),rel_start,hist_keys,splits_b));
        }
        let out=self.backend.draft_flash_attn_batch(
            &q.vector,&k_new.vector,&v_new.vector,&dev_chats,
            heads,kv,width,q_rows,new_rows,kv_bf16,k_new_bf16,causal_base,row_off,
        )?;
        let words=out.len();self.value(out,words)
    }
    fn flash_attn_merge_batch(
        &self,part:&Value<'b,'d>,splits:&[usize],heads:usize,q_rows:usize,max_splits:usize,
    )->Result<Value<'b,'d>> {
        self.ready(part)?;
        let out=self.backend.draft_flash_attn_merge_batch(&part.vector,splits,heads,q_rows,max_splits)?;
        let words=out.len();self.value(out,words)
    }
    fn scatter_kv_batch(
        &self,k_src:&Value<'b,'d>,v_src:&Value<'b,'d>,total_rows:usize,kv_u32_stride:usize,v_is_fp32:bool,
        chats:&mut [(&mut Value<'b,'d>,&mut Value<'b,'d>,usize,usize,usize)],
    )->Result<bool> {
        self.ready(k_src)?;self.ready(v_src)?;
        let mut dev_chats=Vec::with_capacity(chats.len());
        for (dk,dv,dst_row,src_row,rows_b) in chats.iter_mut() {
            let Some(kv)=Rc::get_mut(&mut dk.vector) else {return Ok(false);};
            let Some(vv)=Rc::get_mut(&mut dv.vector) else {return Ok(false);};
            dev_chats.push((kv,vv,*dst_row,*src_row,*rows_b));
        }
        self.backend.draft_scatter_kv_batch(&k_src.vector,&v_src.vector,total_rows,kv_u32_stride,v_is_fp32,&mut dev_chats)
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
