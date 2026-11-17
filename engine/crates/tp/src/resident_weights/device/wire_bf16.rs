//! Lossless transport for outputs whose BF16 round already completed.
use super::*;
impl<'b,'d> Weights<'b,'d> {
        /// The caller proves that each input word is already a widened BF16 word.
        /// The encoder checks this again on the device; it never rounds to fit.
        pub(super) fn exchange_bf16(&mut self,b:&Backend<'d>,source:&DeviceVector<'_>,tokens:usize,width:usize,row_major:bool,finite:bool)->Result<DeviceVector<'d>> {
            let parts=self.topology.parts as usize;
            if tokens.checked_mul(width)!=Some(source.len()) {return Err("BF16 wire source extent differs".into());}
            let whole=parts.checked_mul(width).ok_or("BF16 wire extent overflow")?;
            let open=self.gather_mapped_open_wire(b,&[source],false,true)?;
            self.gather_mapped_close(b,open,false)?;
            let stamp=self.cost.start(b);
            let exchange=self.exchange.as_ref().ok_or("BF16 wire exchange is absent")?;
            let mut shape=spark_gpu_probe::prompt_ops::Shape::new(spark_gpu_probe::prompt_ops::Op::WireUnpackBf16,if row_major {1} else {tokens}).ok_or("wire unpack shape")?;
            shape.rows=u32::try_from(if row_major {whole} else {parts}).map_err(|_|"wire rows exceed ABI")?;
            shape.cols=u32::try_from(if row_major {tokens} else {width}).map_err(|_|"wire columns exceed ABI")?;
            shape.first=u32::from(row_major);shape.slots=u32::from(finite);
            let output=b.prompt_operation(shape,[Some(exchange),None,None,None,None],None)?;
            crate::exchange_trace::reader();self.exchange_read(b)?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);
            spark_gpu_probe::prompt_launches::record("prompt_wire_bf16_exchange");Ok(output)
        }
}
