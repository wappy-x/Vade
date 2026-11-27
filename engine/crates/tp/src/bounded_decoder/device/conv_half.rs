//! The output product of a prompt step in the half form, with the convolution that reads it: the device wrapper.
use super::*;

impl<'b,'d> Device<'b,'d> {
    /// The output product of attention and the convolution after attention of a prompt step, with the product in
    /// the half form (`tc_rows::TC_BIG_HALF_OUT`, `fast_ops::SHORT_CONV_HALF_IN_WORD`): the convolution rounds each
    /// product word where it reads it, so it reads the rounded words from half the bytes. `x` is the packed input
    /// of the product, `s` the shape of the convolution, `first` the first ring slot of the step.
    /// Gives `(the convolution output, the ring after the step)`: the words of `resident_product_rows_half_in`
    /// then `try_short_conv_prefill`. None when the pair is not taken: then no launch was made.
    #[allow(clippy::too_many_arguments,clippy::type_complexity)]
    pub(crate) fn try_output_conv_half(
        &self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,width:usize,columns:usize,x:&Value<'b,'d>,rows:usize,
        s:Shape,first:usize,addend:Option<&Value<'b,'d>>,ring:&Value<'b,'d>,fixed:&[f32],
    )->Result<Option<(Value<'b,'d>,Value<'b,'d>)>> {
        use spark_gpu_probe::{block_stream,fast_ops};
        let taps=s.taps as usize;
        if !self.loop_prefill.get() || !self.backend.resident_launches_active() || s.heads!=1 || s.dim as usize!=width
            || s.cols&block_stream::CONV_TOKEN_TAPS!=0 || (s.cols&block_stream::CONV_ADDEND!=0)!=addend.is_some()
            || !columns.is_multiple_of(2) || weight.is_segmented() || rows==0 || taps==0
            || [Some(x),addend,Some(ring)].iter().flatten().any(|v|v.vector.is_segmented() || v.vector.len()!=v.words)
            || width.checked_mul(columns)!=words.len().checked_mul(2) || x.words!=rows*(columns/2)
            || addend.is_some_and(|a|a.words!=rows*width) || ring.words!=taps*width || fixed.len()!=width*taps
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_SHORT_CONV_WIN,rows,width,taps,fast_ops::STREAM_OPS_THREADS)
            || !(2..=fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || rows<taps || first>=taps
        {
            return Ok(None);
        }
        let matrix=weight.resident_range(words)?;
        if !self.backend.tc_big_half_out_ready(width,columns,rows,first,taps,&x.vector,&matrix) {return Ok(None);}
        let fixed_dev=self.parameters.borrow_mut().resolve(fixed,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("output convolution half: missing fixed taps")?;
        if fixed_dev.is_segmented() {return Ok(None);}
        self.ready(x)?;self.ready(ring)?;
        if let Some(addend)=addend {self.ready(addend)?;}
        let stamp=self.cost.start(self.backend);
        let mut half=self.backend.vector(rows*width/2)?;
        let mut next_ring=self.backend.vector(taps*width)?;
        self.backend.project_tc_big_half_out_into(width,columns,rows,first,taps,&x.vector,&matrix,&mut half,&mut next_ring)?;
        let mut out=self.backend.vector(rows*width)?;
        self.backend.fast_short_conv_half_into(rows,width,taps,s.cols,s.bound,&half,addend.map(|v|&*v.vector),&fixed_dev,(&ring.vector,first),&mut out)?;
        let made=(self.value(out,rows*width)?,self.value(next_ring,taps*width)?);
        self.cost.record(||"output_conv_half".into(),stamp,self.backend,fixed.len()*4);
        Ok(Some(made))
    }
}
