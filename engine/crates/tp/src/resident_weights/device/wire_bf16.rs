//! Lossless transport for outputs whose BF16 round already completed.
use super::*;
impl<'b,'d> Weights<'b,'d> {
        /// Any failed pack invalidates this service, even when no receive was
        /// posted ahead. Peers blocked in the gather observe the link failure.
        pub(super) fn gather_bf16_parts(&mut self,b:&Backend<'d>,local:&[&DeviceVector<'_>],format:impl Fn(usize)->Format)->Result<()> {
            let result=self.gather_mapped_open_wire(b,local,false,format)
                .and_then(|open|self.gather_mapped_close(b,open,false));
            wire_packet::finish(self.ring.as_mut(),result)
        }
        /// The caller proves that each input word is already a widened BF16 word.
        /// The encoder checks this again on the device; it never rounds to fit.
        pub(super) fn exchange_bf16(&mut self,b:&Backend<'d>,source:&DeviceVector<'_>,tokens:usize,width:usize,row_major:bool,finite:bool)->Result<DeviceVector<'d>> {
            let parts=self.topology.parts as usize;
            if tokens.checked_mul(width)!=Some(source.len()) {return Err("BF16 wire source extent differs".into());}
            self.gather_bf16_parts(b,&[source],|_|Format::Bf16)?;
            let stamp=self.cost.start(b);
            let exchange=self.exchange.as_ref().ok_or("BF16 wire exchange is absent")?;
            let shape=wire_packet::unpack_shape(tokens,width,parts,row_major,finite)?;
            let output=b.prompt_operation(shape,[Some(exchange),None,None,None,None],None)?;
            crate::exchange_trace::reader();self.exchange_read(b)?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);
            spark_gpu_probe::prompt_launches::record("prompt_wire_bf16_exchange");Ok(output)
        }
}
