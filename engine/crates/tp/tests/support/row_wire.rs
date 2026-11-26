// Two ranks with the same header, gather and completion order as the service.
struct Wire {topology:Topology,controller:bool,tx:SyncSender<Vec<u8>>,rx:Receiver<Vec<u8>>}
impl Exchange for Wire {
    fn topology(&self)->Topology {self.topology}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {
        self.tx.send(local.iter().flat_map(|v|v.to_le_bytes()).collect()).map_err(|e|e.to_string())?;
        let other=self.rx.recv().map_err(|e|e.to_string())?;
        let peer=other.as_chunks::<4>().0.iter().map(|v|f32::from_le_bytes(*v)).collect::<Vec<_>>();
        let mut out=vec![0.0;rows];let r=spark_tp::rows(rows,self.topology.rank as usize,self.topology.parts as usize);out[r].copy_from_slice(local);
        let r=spark_tp::rows(rows,1-self.topology.rank as usize,self.topology.parts as usize);out[r].copy_from_slice(&peer);Ok(out)
    }
    fn command(&mut self,v:&mut [u8])->Result<()> {
        if self.controller {self.tx.send(v.to_vec()).map_err(|e|e.to_string())} else {let p=self.rx.recv().map_err(|e|e.to_string())?;v.copy_from_slice(&p);Ok(())}
    }
    fn complete(&mut self)->Result<()> {self.tx.send(Vec::new()).map_err(|e|e.to_string())?;assert!(self.rx.recv().map_err(|e|e.to_string())?.is_empty());Ok(())}
}
