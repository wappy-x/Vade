//! One decode mHC boundary on CPU cores. The same FP32 operations in the same order as
//! mhc_boundary_rms, mhc_boundary_dots and mhc_boundary_finish: rounded products, no fused
//! multiply-add, one ascending chain of adds per sum, each chain started at 0.0. A CPU core
//! adds with a shorter latency than a GPU thread, so the two serial chains of a boundary end
//! sooner here. Every plane must equal the kernels' plane bit for bit; `tests` pins this
//! module to `mhc_reference`, and the device gate compares it with the kernels.
use crate::{bf16::rounded,mhc_params::Shape,mhc_reference,norm_boundary::NormShape};
use std::{cell::Cell,sync::{atomic::{AtomicBool,AtomicU32,Ordering},Arc,Mutex,RwLock},time::{Duration,Instant}};

/// Coefficient rows one core folds together. The rows of a group are independent chains in
/// adjacent vector lanes; the group size changes no value.
const GROUPS:[usize;4]=[8,12,16,24];
fn group_rows(count:usize,cores:usize)->Result<usize,String> {
    if cores==0 {return Err("host mHC requires at least one core".into());}
    let need=count.div_ceil(cores);Ok(GROUPS.into_iter().find(|&group|group>=need).unwrap_or(GROUPS[GROUPS.len()-1]))
}

/// The parameters of one boundary in the order the folds read them. A group holds its rows
/// feature-major (the rows' weights of one feature are adjacent), padded with zero rows.
pub struct HostBoundary {shape:Shape,norm:NormShape,group:usize,groups:Vec<Vec<f32>>,base:Vec<f32>,scale:[f32;3],norm_weight:Vec<f32>}
/// Every plane the three kernels write for one token.
#[derive(Clone,Debug,Default,PartialEq)]
pub struct Planes {pub residual:Vec<f32>,pub mixing:Vec<f32>,pub coefficients:Vec<f32>,pub collapsed:Vec<f32>,pub partials:Vec<f32>,pub output:Vec<f32>}

impl HostBoundary {
    /// `cores`: the cores that fold coefficient rows (one group each when the count allows).
    pub fn new(shape:Shape,norm:NormShape,weights:&[f32],base:&[f32],scale:&[f32],norm_weight:&[f32],cores:usize)->Result<Self,String> {
        shape.validate()?;norm.check_weights(norm_weight)?;let (wide,count)=(shape.wide(),shape.coefficients());
        if shape.rows!=1 || cores==0 || norm.width()!=shape.hidden as usize || weights.len()!=count*wide || base.len()!=count || scale.len()!=3
            || weights.iter().chain(base).chain(scale).any(|value|!value.is_finite()) {return Err("host mHC boundary parameter geometry or value differs".into());}
        let group=group_rows(count,cores)?;
        let groups=(0..count.div_ceil(group)).map(|k|{let mut lanes=vec![0.0f32;wide*group];
            for g in 0..group.min(count-k*group) {let row=&weights[(k*group+g)*wide..(k*group+g+1)*wide];for (i,&w) in row.iter().enumerate() {lanes[i*group+g]=w;}}lanes}).collect();
        Ok(Self {shape,norm,group,groups,base:base.to_vec(),scale:[scale[0],scale[1],scale[2]],norm_weight:norm_weight.to_vec()})
    }
    pub fn shape(&self)->Shape {self.shape}
    /// Folds a caller may give to separate cores.
    pub fn groups(&self)->usize {self.groups.len()}
    pub fn parameter_bytes(shape:Shape,cores:usize)->Result<usize,String> {
        shape.validate()?;let count=shape.coefficients();let group=group_rows(count,cores)?;let groups=count.div_ceil(group);
        let words=groups.checked_mul(group).and_then(|rows|rows.checked_mul(shape.wide())).and_then(|weights|weights.checked_add(count))
            .and_then(|words|words.checked_add(shape.hidden as usize)).ok_or("host mHC parameter word count overflow")?;
        words.checked_mul(std::mem::size_of::<f32>()).and_then(|bytes|groups.checked_mul(std::mem::size_of::<Vec<f32>>()).and_then(|headers|bytes.checked_add(headers)))
            .and_then(|bytes|bytes.checked_add(std::mem::size_of::<Self>())).ok_or_else(||"host mHC parameter byte count overflow".into())
    }
    pub fn host_bytes(&self)->Result<usize,String> {
        self.groups.iter().map(|group|group.capacity()).chain([self.base.capacity(),self.norm_weight.capacity()]).try_fold(0usize,|bytes,words| {
            words.checked_mul(std::mem::size_of::<f32>()).and_then(|size|bytes.checked_add(size)).ok_or_else(||"host mHC parameter byte count overflow".to_owned())
        }).and_then(|bytes|self.groups.capacity().checked_mul(std::mem::size_of::<Vec<f32>>()).and_then(|size|bytes.checked_add(size))
            .and_then(|bytes|bytes.checked_add(std::mem::size_of::<Self>())).ok_or_else(||"host mHC parameter byte count overflow".to_owned()))
    }
    fn touch_group(&self,index:usize) {std::hint::black_box(self.touch(index));}
    /// The first boundary of a token: the residual is the rounded input (mhc_boundary_rms with no previous plane).
    pub fn residual_first(&self,input:&[f32],residual:&mut [f32]) {for (out,&value) in residual.iter_mut().zip(input) {*out=rounded(value);}}
    /// A later boundary: the residual is the last boundary's combine (mhc_boundary::combine, same order).
    pub fn residual_after(&self,previous:&[f32],coefficients:&[f32],block:&[f32],residual:&mut [f32]) {
        let (n,h)=(self.shape.streams as usize,self.shape.hidden as usize);
        // Each feature's sum starts at 0.0 and takes the sources in ascending order, as combine() does.
        // A source's loop over the features carries no value, so it runs in vector lanes.
        // The rounded block is the same for every destination: round it once.
        let block:Vec<f32>=block.iter().map(|&b|rounded(b)).collect();
        for (destination,out) in residual.chunks_exact_mut(h).enumerate() {
            out.fill(0.0);
            for (source,last) in previous.chunks_exact(h).enumerate().take(n) {add_scaled(out,last,rounded(coefficients[2*n+source*n+destination]));}
            round_combine(out,&block,rounded(coefficients[n+destination]));
        }
    }
    fn residual_range(&self,input:&ResidualInput,first:usize,mut output:&mut [f32]) {
        let (streams,hidden)=(self.shape.streams as usize,self.shape.hidden as usize);let mut position=first;
        while !output.is_empty() {
            let (destination,feature)=(position/hidden,position%hidden);let words=output.len().min(hidden-feature);let (part,rest)=output.split_at_mut(words);part.fill(0.0);
            for source in 0..streams {let start=source*hidden+feature;add_scaled(part,&input.previous[start..start+words],rounded(input.coefficients[2*streams+source*streams+destination]));}
            round_combine(part,&input.block[feature..feature+words],rounded(input.coefficients[streams+destination]));position+=words;output=rest;
        }
    }
    /// 1 / RMS of the residual: one ascending chain over every feature (mhc_reference::inverse).
    pub fn inverse(&self,residual:&[f32])->f32 {mhc_reference::inverse(self.shape,residual)}
    /// The raw coefficients of group `k`: each row one ascending chain of w * (x * inverse).
    pub fn fold(&self,k:usize,residual:&[f32],inverse:f32,raw:&mut [f32]) {
        let lanes=&self.groups[k];let first=k*self.group;let rows=raw.len().min(self.group);
        match self.group {
            8=>raw[..rows].copy_from_slice(&fold_lanes::<8>(lanes,residual,inverse)[..rows]),
            12=>raw[..rows].copy_from_slice(&fold_lanes::<12>(lanes,residual,inverse)[..rows]),
            16=>raw[..rows].copy_from_slice(&fold_lanes::<16>(lanes,residual,inverse)[..rows]),
            _=>raw[..rows].copy_from_slice(&fold_lanes::<24>(lanes,residual,inverse)[..rows]),
        }
        let n=self.shape.streams as usize;
        for (g,value) in raw[..rows].iter_mut().enumerate() {let i=first+g;let group=if i<n {0} else if i<2*n {1} else {2};*value=*value*self.scale[group]+self.base[i];}
    }
    /// Read one word of every cache line of group `k`, so its fold finds the weights in this core's cache.
    pub fn touch(&self,k:usize)->u32 {self.groups[k].iter().step_by(16).fold(0u32,|sum,value|sum.wrapping_add(value.to_bits()))}
    /// The rows of group `k` inside the coefficient plane.
    pub fn rows(&self,k:usize)->std::ops::Range<usize> {k*self.group..((k+1)*self.group).min(self.shape.coefficients())}
    /// After every fold: activation, collapse and the ordinary norm (mhc_boundary_finish).
    pub fn finish(&self,inverse:f32,planes:&mut Planes) {
        let (n,h)=(self.shape.streams as usize,self.shape.hidden as usize);
        scale_into(&mut planes.mixing,&planes.residual,inverse);
        mhc_reference::activate(self.shape,&mut planes.coefficients);
        // mhc_boundary::collapse for every feature: the sum starts at 0.0, ascending source order, one final round.
        planes.collapsed.fill(0.0);
        for (source,stream) in planes.residual.chunks_exact(h).enumerate().take(n) {add_scaled(&mut planes.collapsed,stream,planes.coefficients[source]);}
        round_each(&mut planes.collapsed);
        norm_planes(self.norm,&planes.collapsed,&self.norm_weight,&mut planes.partials,&mut planes.output);
    }
    pub fn planes(&self)->Planes {let (wide,h)=(self.shape.wide(),self.shape.hidden as usize);
        Planes {residual:vec![0.0;wide],mixing:vec![0.0;wide],coefficients:vec![0.0;self.shape.coefficients()],collapsed:vec![0.0;h],partials:vec![0.0;h.div_ceil(128)],output:vec![0.0;h]}}
    /// The whole boundary on the calling core. `previous`: the last boundary's residual and coefficients.
    pub fn run(&self,previous:Option<(&[f32],&[f32])>,source:&[f32],planes:&mut Planes) {
        match previous {None=>self.residual_first(source,&mut planes.residual),Some((residual,coefficients))=>self.residual_after(residual,coefficients,source,&mut planes.residual)}
        let inverse=self.inverse(&planes.residual);
        for k in 0..self.groups() {let rows=self.rows(k);self.fold(k,&planes.residual,inverse,&mut planes.coefficients[rows]);}
        self.finish(inverse,planes);
    }
}
// The three loops below stay out of line: inside a larger body the compiler kept them scalar
// (29 us for one residual on an X925 core; 12 us out of line). They change no value.
/// sum[i] += weight * x[i]. One rounded product and one add per feature; no value is carried between features.
#[inline(never)]
fn add_scaled(sum:&mut [f32],x:&[f32],weight:f32) {for (sum,&x) in sum.iter_mut().zip(x) {*sum+=weight*x;}}
/// out[i] = rounded(rounded(out[i]) + rounded(post * block[i])), the tail of mhc_boundary::combine. `block` is rounded already.
#[inline(never)]
fn round_combine(out:&mut [f32],block:&[f32],post:f32) {for (out,&b) in out.iter_mut().zip(block) {*out=rounded(rounded(*out)+rounded(post*b));}}
#[inline(never)]
fn round_each(values:&mut [f32]) {for value in values.iter_mut() {*value=rounded(*value);}}
/// `G` independent ascending chains. Lane g adds w[g] * (x * inverse) feature by feature.
#[inline(never)]
fn fold_lanes<const G:usize>(lanes:&[f32],residual:&[f32],inverse:f32)->[f32;G] {
    let mut sums=[0.0f32;G];
    for (row,&x) in lanes.as_chunks::<G>().0.iter().zip(residual) {
        let normalized=x*inverse;
        for g in 0..G {sums[g]+=row[g]*normalized;}
    }
    sums
}
/// The resident ordinary norm tree (the tail of mhc_boundary_finish): 32 strided lanes per
/// 128-feature chunk, one fixed pairwise tree per chunk, the same tree over the chunk sums.
#[inline(never)]
pub fn norm_planes(s:NormShape,x:&[f32],weight:&[f32],partials:&mut [f32],output:&mut [f32]) {
    fn sum(mut lanes:[f32;32])->f32 {for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32 {lanes[lane]=old[lane]+old[lane^distance];}}lanes[0]}
    // Lane l of a chunk adds the squares of features l, l+32, l+64, l+96 in that order.
    for (partial,chunk) in partials.iter_mut().zip(x.chunks(128)) {let mut lanes=[0.0f32;32];for row in chunk.chunks(32) {for (lane,&v) in lanes.iter_mut().zip(row) {*lane+=v*v;}}*partial=sum(lanes);}
    let mut lanes=[0.0;32];for (lane,sum) in lanes.iter_mut().enumerate() {let mut i=lane;while i<partials.len() {*sum+=partials[i];i+=32;}}
    let scale=1.0/(sum(lanes)/s.width() as f32+s.epsilon()).sqrt();
    norm_output(s,x,weight,scale,output);
}
#[inline(never)]
fn norm_output(s:NormShape,x:&[f32],weight:&[f32],scale:f32,output:&mut [f32]) {for ((out,&x),&w) in output.iter_mut().zip(x).zip(weight) {*out=s.weight(x*scale,w);}}
/// out[i] = x[i] * k.
#[inline(never)]
fn scale_into(out:&mut [f32],x:&[f32],k:f32) {for (out,&x) in out.iter_mut().zip(x) {*out=x*k;}}

/// Helper cores for the coefficient folds. Each helper spins on a sequence word while a
/// decode step runs, so a fold starts within a few hundred nanoseconds of its request.
pub struct Pool {shared:Arc<Shared>,helpers:Vec<std::thread::JoinHandle<()>>,preparing:Cell<bool>,pub last_ns:Cell<[u64;4]>}
#[derive(Default)]
struct ResidualInput {previous:Vec<f32>,coefficients:Vec<f32>,block:Vec<f32>}
struct Shared {sequence:AtomicU32,done:AtomicU32,ready:AtomicU32,operation:AtomicU32,stop:AtomicBool,inverse:AtomicU32,residual:RwLock<Vec<f32>>,input:RwLock<ResidualInput>,parts:Vec<Mutex<Vec<f32>>>,boundary:Mutex<Option<Arc<HostBoundary>>>,error:Mutex<Option<String>>,raw:Vec<AtomicU32>}
impl Pool {
    /// `helpers` extra cores; the caller's core folds group 0 and every group past the helpers.
    pub fn new(helpers:usize,coefficients:usize)->Self {
        let shared=Arc::new(Shared {sequence:AtomicU32::new(0),done:AtomicU32::new(0),ready:AtomicU32::new(0),operation:AtomicU32::new(0),stop:AtomicBool::new(false),inverse:AtomicU32::new(0),
            residual:RwLock::new(Vec::new()),input:RwLock::new(ResidualInput::default()),parts:(0..helpers).map(|_|Mutex::new(Vec::new())).collect(),boundary:Mutex::new(None),error:Mutex::new(None),raw:(0..coefficients).map(|_|AtomicU32::new(0)).collect()});
        let helpers:Vec<_>=(1..=helpers).map(|helper|{let shared=shared.clone();std::thread::spawn(move||{
            #[cfg(target_os="linux")] if let Err(error)=crate::driver::prefer_fast_core(helper) {if let Ok(mut saved)=shared.error.lock() {*saved=Some(error);}}
            shared.ready.fetch_add(1,Ordering::Release);
            let (mut seen,mut idle)=(0u32,Instant::now());
            loop {
                let now=shared.sequence.load(Ordering::Acquire);
                if now==seen {if shared.stop.load(Ordering::Acquire) {return;}
                    if idle.elapsed()>=Duration::from_millis(2) {std::thread::park_timeout(Duration::from_millis(2));}else {std::hint::spin_loop();}continue;}
                seen=now;idle=Instant::now();
                let result=(||->Result<(),String> {
                    let boundary=shared.boundary.lock().map_err(|_|"host mHC boundary lock is poisoned")?.clone().ok_or("host mHC helper has no boundary")?;
                    match shared.operation.load(Ordering::Acquire) {
                        0=>{if helper<boundary.groups() {boundary.touch_group(helper);}},
                        1=>{let cores=shared.parts.len()+1;let wide=boundary.shape().wide();let (first,end)=(wide*helper/cores,wide*(helper+1)/cores);
                            let input=shared.input.read().map_err(|_|"host mHC input lock is poisoned")?;let mut part=shared.parts[helper-1].lock().map_err(|_|"host mHC residual part lock is poisoned")?;
                            part.resize(end-first,0.0);boundary.residual_range(&input,first,&mut part);},
                        2=>{if helper<boundary.groups() {let residual=shared.residual.read().map_err(|_|"host mHC residual lock is poisoned")?;
                            let rows=boundary.rows(helper);let mut raw=[0.0f32;24];boundary.fold(helper,&residual,f32::from_bits(shared.inverse.load(Ordering::Acquire)),&mut raw[..rows.len()]);
                            for (word,value) in shared.raw[rows].iter().zip(raw) {word.store(value.to_bits(),Ordering::Relaxed);}}},
                        _=>return Err("host mHC helper operation differs".into()),
                    }Ok(())
                })();
                if let Err(error)=result {if let Ok(mut saved)=shared.error.lock() {*saved=Some(error);}}
                shared.done.fetch_add(1,Ordering::Release);
            }
        })}).collect();
        while shared.ready.load(Ordering::Acquire)!=helpers.len() as u32 {std::hint::spin_loop();}
        Self {shared,helpers,preparing:Cell::new(false),last_ns:Cell::new([0;4])}
    }
    fn check(&self)->Result<(),String> {match &*self.shared.error.lock().map_err(|_|"host mHC helper error lock is poisoned")? {Some(error)=>Err(error.clone()),None=>Ok(())}}
    fn dispatch(&self,boundary:&Arc<HostBoundary>,operation:u32)->Result<(),String> {
        self.check()?;*self.shared.boundary.lock().map_err(|_|"host mHC boundary lock is poisoned")?=Some(boundary.clone());
        self.shared.operation.store(operation,Ordering::Release);self.shared.done.store(0,Ordering::Release);self.shared.sequence.fetch_add(1,Ordering::Release);
        for helper in &self.helpers {helper.thread().unpark();}Ok(())
    }
    fn prepared(&self)->Result<(),String> {
        if self.preparing.replace(false) {while self.shared.done.load(Ordering::Acquire)!=self.helpers.len() as u32 {std::hint::spin_loop();}}
        self.check()
    }
    pub fn prepare(&self,boundary:&Arc<HostBoundary>)->Result<(),String> {
        self.prepared()?;self.dispatch(boundary,0)?;self.preparing.set(true);
        for group in (0..boundary.groups()).filter(|&group|group==0 || group>self.helpers.len()) {boundary.touch_group(group);}Ok(())
    }
    /// The whole boundary: the residual, the RMS chain and group 0 on the calling core, the other groups on the helpers.
    pub fn run(&self,boundary:&Arc<HostBoundary>,previous:Option<(&[f32],&[f32])>,source:&[f32],planes:&mut Planes)->Result<(),String> {
        let inverse=self.fold_boundary(boundary,previous,source,planes)?;let clock=Instant::now();boundary.finish(inverse,planes);
        let mut ns=self.last_ns.get();ns[3]=clock.elapsed().as_nanos() as u64;self.last_ns.set(ns);Ok(())
    }
    pub fn fold_boundary(&self,boundary:&Arc<HostBoundary>,previous:Option<(&[f32],&[f32])>,source:&[f32],planes:&mut Planes)->Result<f32,String> {
        self.check()?;let shared=&*self.shared;let count=boundary.shape().coefficients();
        if count>shared.raw.len() || boundary.group>24 {return Err("host mHC pool is too small for this boundary".into());}
        let clock=Instant::now();let mut ns=[0u64;4];
        if let Some((last,coefficients))=previous.filter(|_|!self.helpers.is_empty()) {
            self.prepared()?;
            {let mut input=shared.input.write().map_err(|_|"host mHC input lock is poisoned")?;
                input.previous.clear();input.previous.extend_from_slice(last);input.coefficients.clear();input.coefficients.extend_from_slice(coefficients);
                input.block.clear();input.block.extend(source.iter().map(|&value|rounded(value)));}
            self.dispatch(boundary,1)?;let cores=self.helpers.len()+1;let wide=boundary.shape().wide();
            {let input=shared.input.read().map_err(|_|"host mHC input lock is poisoned")?;boundary.residual_range(&input,0,&mut planes.residual[..wide/cores]);}
            while shared.done.load(Ordering::Acquire)!=self.helpers.len() as u32 {std::hint::spin_loop();}self.check()?;
            for (index,part) in shared.parts.iter().enumerate() {let first=wide*(index+1)/cores;let end=wide*(index+2)/cores;
                let part=part.lock().map_err(|_|"host mHC residual part lock is poisoned")?;planes.residual[first..end].copy_from_slice(&part);}
        } else {match previous {None=>boundary.residual_first(source,&mut planes.residual),Some((last,coefficients))=>boundary.residual_after(last,coefficients,source,&mut planes.residual)}}
        ns[0]=clock.elapsed().as_nanos() as u64;let inverse=boundary.inverse(&planes.residual);ns[1]=clock.elapsed().as_nanos() as u64-ns[0];
        self.prepared()?;
        let helped=self.helpers.len().min(boundary.groups().saturating_sub(1));
        if helped>0 {
            std::mem::swap(&mut *shared.residual.write().map_err(|_|"host mHC residual lock is poisoned")?,&mut planes.residual);
            shared.inverse.store(inverse.to_bits(),Ordering::Release);self.dispatch(boundary,2)?;
        }
        {let lent=if helped>0 {Some(shared.residual.read().map_err(|_|"host mHC residual lock is poisoned")?)} else {None};
            let residual=lent.as_ref().map_or(&planes.residual[..],|lent|&lent[..]);
            for k in (0..boundary.groups()).filter(|&k|k==0 || k>helped) {let rows=boundary.rows(k);boundary.fold(k,residual,inverse,&mut planes.coefficients[rows]);}}
        if helped>0 {
            // Every helper answers every request, so the count is the helper count.
            while shared.done.load(Ordering::Acquire)!=self.helpers.len() as u32 {std::hint::spin_loop();}
            for k in 1..=helped {for i in boundary.rows(k) {planes.coefficients[i]=f32::from_bits(shared.raw[i].load(Ordering::Relaxed));}}
            std::mem::swap(&mut *shared.residual.write().map_err(|_|"host mHC residual lock is poisoned")?,&mut planes.residual);
        }
        self.check()?;ns[2]=clock.elapsed().as_nanos() as u64-ns[0]-ns[1];self.last_ns.set(ns);Ok(inverse)
    }
}
impl Drop for Pool {
    fn drop(&mut self) {self.shared.stop.store(true,Ordering::Release);for helper in &self.helpers {helper.thread().unpark();}for helper in self.helpers.drain(..) {let _=helper.join();}}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mhc_boundary,norm_boundary::WeightType};
    fn value(seed:&mut u64)->f32 {*seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);((*seed>>40) as f32/(1u64<<24) as f32)*2.0-1.0}
    /// The existing host reference of one token, plane by plane (bounded_decoder::host::Host::begin and finish).
    fn reference(shape:Shape,norm:NormShape,weights:&[f32],base:&[f32],scale:&[f32],norm_weight:&[f32],previous:Option<(&[f32],&[f32])>,source:&[f32])->Planes {
        let (n,h,wide,count)=(shape.streams as usize,shape.hidden as usize,shape.wide(),shape.coefficients());
        let residual:Vec<f32>=match previous {None=>source.iter().map(|&v|rounded(v)).collect(),
            Some((last,coefficients))=>(0..wide).map(|j|mhc_boundary::combine(last,coefficients,rounded(source[j%h]),n,h,j/h,j%h)).collect()};
        let (mut mixing,mut coefficients)=(vec![0.0;wide],vec![0.0;count]);
        mhc_reference::prepare(shape,&residual,weights,base,scale,&mut mixing,&mut coefficients);
        let collapsed:Vec<f32>=(0..h).map(|j|mhc_boundary::collapse(&residual,&coefficients,n,h,j)).collect();
        let (mut partials,mut output)=(vec![0.0;h.div_ceil(128)],vec![0.0;h]);norm_planes(norm,&collapsed,norm_weight,&mut partials,&mut output);
        Planes {residual,mixing,coefficients,collapsed,partials,output}
    }
    fn bits(planes:&Planes)->Vec<u32> {[&planes.residual,&planes.mixing,&planes.coefficients,&planes.collapsed,&planes.partials,&planes.output].into_iter().flatten().map(|v|v.to_bits()).collect()}
    fn case(streams:u32,hidden:u32,helpers:usize,cores:usize,special:bool) {
        let shape=Shape {rows:1,streams,hidden,iterations:20,rms_eps:1e-6,eps:1e-6};let (wide,count)=(shape.wide(),shape.coefficients());
        let mut seed=u64::from(streams)<<32|u64::from(hidden)|(helpers as u64)<<48;
        let weights:Vec<f32>=(0..count*wide).map(|_|value(&mut seed)*0.05).collect();let base:Vec<f32>=(0..count).map(|_|value(&mut seed)).collect();
        let scale=[0.01f32,0.02,0.03];let norm=NormShape::new(hidden as usize,1e-6,WeightType::Bf16).unwrap();
        let norm_weight:Vec<f32>=(0..hidden).map(|_|rounded(value(&mut seed)+1.5)).collect();
        // Special inputs: signed zeros, subnormals, tiny and huge finite values, values that round up in BF16.
        let specials=[0.0f32,-0.0,f32::MIN_POSITIVE,-f32::MIN_POSITIVE,1e-45,-1e-45,1e-30,-1e-30,3.0e18,-3.0e18,1.0,-1.0,f32::from_bits(0x3f80_8000),f32::from_bits(0x3f80_7fff),f32::from_bits(0x3f81_8000),65504.0];
        let mut input=|len:usize|(0..len).map(|i|if special && i%3!=1 {specials[(i/3+i)%specials.len()]} else {value(&mut seed)*4.0}).collect::<Vec<f32>>();
        let first=input(wide);let blocks=[input(hidden as usize),input(hidden as usize)];
        let boundary=Arc::new(HostBoundary::new(shape,norm,&weights,&base,&scale,&norm_weight,cores).unwrap());let pool=Pool::new(helpers,count);
        let mut want=reference(shape,norm,&weights,&base,&scale,&norm_weight,None,&first);
        let (mut alone,mut pooled)=(boundary.planes(),boundary.planes());
        boundary.run(None,&first,&mut alone);pool.run(&boundary,None,&first,&mut pooled).unwrap();
        assert_eq!(bits(&alone),bits(&want),"first boundary, one core: {streams}x{hidden}");assert_eq!(bits(&pooled),bits(&want),"first boundary, pool: {streams}x{hidden}");
        for block in &blocks {
            let next=reference(shape,norm,&weights,&base,&scale,&norm_weight,Some((&want.residual,&want.coefficients)),block);
            let (mut a,mut p)=(boundary.planes(),boundary.planes());
            boundary.run(Some((&alone.residual,&alone.coefficients)),block,&mut a);pool.run(&boundary,Some((&pooled.residual,&pooled.coefficients)),block,&mut p).unwrap();
            assert_eq!(bits(&a),bits(&next),"later boundary, one core: {streams}x{hidden}");assert_eq!(bits(&p),bits(&next),"later boundary, pool: {streams}x{hidden}");
            (want,alone,pooled)=(next,a,p);
        }
    }
    /// Host timing probe (run with --ignored --nocapture): the parts of one boundary on this core.
    #[test]
    #[ignore]
    fn time_the_parts_on_this_core() {
        let shape=Shape {rows:1,streams:4,hidden:4096,iterations:20,rms_eps:1e-6,eps:1e-6};let (wide,count)=(shape.wide(),shape.coefficients());let mut seed=7u64;
        let weights:Vec<f32>=(0..count*wide).map(|_|value(&mut seed)*0.05).collect();let base:Vec<f32>=(0..count).map(|_|value(&mut seed)).collect();
        let norm=NormShape::new(4096,1e-6,WeightType::Bf16).unwrap();let norm_weight:Vec<f32>=(0..4096).map(|_|rounded(value(&mut seed)+1.5)).collect();
        let boundary=HostBoundary::new(shape,norm,&weights,&base,&[0.01,0.02,0.03],&norm_weight,3).unwrap();
        let first:Vec<f32>=(0..wide).map(|_|value(&mut seed)).collect();let block:Vec<f32>=(0..4096).map(|_|value(&mut seed)).collect();
        let (mut a,mut b)=(boundary.planes(),boundary.planes());boundary.run(None,&first,&mut a);
        let mut ns=[0u128;4];let reps=2000;
        for _ in 0..reps {
            let t=std::time::Instant::now();boundary.residual_after(&a.residual,&a.coefficients,std::hint::black_box(&block),&mut b.residual);ns[0]+=t.elapsed().as_nanos();
            let t=std::time::Instant::now();let inverse=boundary.inverse(&b.residual);ns[1]+=t.elapsed().as_nanos();
            let t=std::time::Instant::now();for k in 0..boundary.groups() {let rows=boundary.rows(k);boundary.fold(k,&b.residual,inverse,&mut b.coefficients[rows]);}ns[2]+=t.elapsed().as_nanos();
            let t=std::time::Instant::now();boundary.finish(inverse,&mut b);ns[3]+=t.elapsed().as_nanos();
        }
        println!("HOST PARTS us: residual={:.2} rms_chain={:.2} folds_one_core={:.2} finish={:.2}",ns[0] as f64/1e3/reps as f64,ns[1] as f64/1e3/reps as f64,ns[2] as f64/1e3/reps as f64,ns[3] as f64/1e3/reps as f64);
    }
    #[test]
    #[ignore]
    fn time_parallel_residual_on_fast_cores() {
        #[cfg(target_os="linux")] crate::driver::prefer_fast_cores().unwrap();
        let shape=Shape {rows:1,streams:4,hidden:4096,iterations:20,rms_eps:1e-6,eps:1e-6};let (wide,count)=(shape.wide(),shape.coefficients());let mut seed=7u64;
        let weights:Vec<f32>=(0..count*wide).map(|_|value(&mut seed)*0.05).collect();let base:Vec<f32>=(0..count).map(|_|value(&mut seed)).collect();
        let norm=NormShape::new(4096,1e-6,WeightType::Bf16).unwrap();let norm_weight:Vec<f32>=(0..4096).map(|_|rounded(value(&mut seed)+1.5)).collect();
        let boundary=Arc::new(HostBoundary::new(shape,norm,&weights,&base,&[0.01,0.02,0.03],&norm_weight,3).unwrap());let pool=Pool::new(2,count);
        #[cfg(target_os="linux")] crate::driver::prefer_fast_core(0).unwrap();
        let first:Vec<f32>=(0..wide).map(|_|value(&mut seed)).collect();let block:Vec<f32>=(0..4096).map(|_|value(&mut seed)).collect();
        let (mut previous,mut expected,mut actual)=(boundary.planes(),boundary.planes(),boundary.planes());boundary.run(None,&first,&mut previous);
        boundary.run(Some((&previous.residual,&previous.coefficients)),&block,&mut expected);let mut serial=Vec::new();let mut parallel=Vec::new();let mut totals=Vec::new();
        for _ in 0..2000 {
            let clock=Instant::now();boundary.residual_after(&previous.residual,&previous.coefficients,std::hint::black_box(&block),&mut actual.residual);serial.push(clock.elapsed().as_nanos() as u64);
            pool.prepare(&boundary).unwrap();let clock=Instant::now();pool.run(&boundary,Some((&previous.residual,&previous.coefficients)),&block,&mut actual).unwrap();totals.push(clock.elapsed().as_nanos() as u64);parallel.push(pool.last_ns.get()[0]);
        }
        assert_eq!(bits(&actual),bits(&expected));serial.sort_unstable();parallel.sort_unstable();totals.sort_unstable();let middle=serial.len()/2;
        println!("HOST PARALLEL samples={} cores=3 serial_residual_us={:.3} parallel_residual_us={:.3} boundary_us={:.3} every_plane_bit_equal=true",serial.len(),serial[middle] as f64/1e3,parallel[middle] as f64/1e3,totals[middle] as f64/1e3);
    }
    #[test]
    fn every_plane_equals_the_reference_on_the_model_shape() {for (helpers,cores) in [(0,1),(1,2),(2,3),(3,4),(2,1),(5,6)] {case(4,4096,helpers,cores,false);case(4,4096,helpers,cores,true);}}
    #[test]
    fn every_plane_equals_the_reference_on_swept_shapes() {
        for streams in [1u32,2,3,4,5,7,8] {for hidden in [1u32,2,3,5,15,16,17,31,32,33,127,128,129,255,256,257,1023,1024,1025] {
            for (helpers,cores) in [(0,1),(2,3),(1,5)] {case(streams,hidden,helpers,cores,false);case(streams,hidden,helpers,cores,true);}
        }}
    }
}
