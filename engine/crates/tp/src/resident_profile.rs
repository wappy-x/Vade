//! Optional wall, driver, and existing CUDA-event counters for the resident probe.
#![forbid(unsafe_code)]
#[cfg(target_os="linux")]
mod device {
    use std::{cell::RefCell,collections::BTreeMap,time::Instant};
    use spark_gpu_probe::driver::{Backend,BackendStats};
    pub(crate) struct Stamp {time:Instant,before:BackendStats}
    pub(crate) struct Cost {enabled:bool,rows:RefCell<BTreeMap<String,[u128;11]>>}
    impl Cost {
        pub(crate) fn new(enabled:bool)->Self {Self {enabled,rows:RefCell::new(BTreeMap::new())}}
        pub(crate) fn enabled(&self)->bool {self.enabled}
        pub(crate) fn start(&self,b:&Backend<'_>)->Option<Stamp> {self.enabled.then(||Stamp {time:Instant::now(),before:b.stats()})}
        pub(crate) fn record(&self,name:impl FnOnce()->String,stamp:Option<Stamp>,b:&Backend<'_>,bytes:usize) {
            let Some(stamp)=stamp else {return;};let wall=stamp.time.elapsed().as_nanos();let s=b.stats().since(stamp.before);
            let values=[1,wall,s.device_kernel_ns[..12].iter().sum::<u128>()+s.device_kernel_ns[13],s.device_kernel_ns[12],
                s.allocation_ns,s.stream_wait_ns,s.host_launch_ns,s.host_profile_ns,s.output_download_ns,s.weight_read_upload_ns,bytes as u128];
            let mut rows=self.rows.borrow_mut();let row=rows.entry(name()).or_default();for (a,v) in row.iter_mut().zip(values) {*a+=v;}
        }
        pub(crate) fn host(&self,name:&str,ns:u128) {if self.enabled {let mut rows=self.rows.borrow_mut();let row=rows.entry(name.into()).or_default();row[0]+=1;row[1]+=ns;}}
        pub(crate) fn report(&self) {
            for (name,v) in std::mem::take(&mut *self.rows.borrow_mut()) {eprintln!("RESIDENT DETAIL name={name} calls={} wall_ns={} gpu_ns={} copy_ns={} allocation_ns={} wait_ns={} launch_ns={} profile_ns={} download_ns={} upload_ns={} weight_bytes={}",v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10]);}
        }
    }
}
#[cfg(target_os="linux")]
pub(crate) use device::Cost;
