//! Plan-owned recurrence: ordered FP32 steps with state retained per prompt tile.
//! Independent sessions share launches; verification snapshots retain token order.
use engine_model::{plan::ModelPlan, safetensors::Dtype, shapes::{Mixer, Part, Place, Residual, Store}, split::Owners};
use std::ops::Range;
type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenStepError { HistoryShape, Execution(String) }
impl std::fmt::Display for TokenStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::HistoryShape => f.write_str("GDN fused history requires nonzero precision and whole aligned value/key head groups"), Self::Execution(error) => f.write_str(error) }
    }
}
impl std::error::Error for TokenStepError {}
impl From<String> for TokenStepError { fn from(error: String) -> Self { Self::Execution(error) } }
impl From<TokenStepError> for String { fn from(error: TokenStepError) -> Self { error.to_string() } }

#[derive(Debug, Clone, PartialEq)]
pub struct StepBinding { keys: Range<u32>, values: Range<u32>, repeat: u32, precision: u32, epsilon: f32 }
impl StepBinding {
    pub fn whole(plan: &ModelPlan, layer: usize) -> Result<Self> {
        let Mixer::Linear(s) = plan.shape.layers.get(layer).ok_or("GDN layer missing")?.mixer else { return Err("GDN layer kind".into()); };
        Self::new(plan, layer, 0..s.key_heads, 0..s.value_heads)
    }
    pub fn owned(plan: &ModelPlan, layer: usize, owner: &Owners) -> Result<Self> {
        Self::new(plan, layer, owner.key_heads.clone(), owner.value_heads.clone())
    }
    fn new(plan: &ModelPlan, layer: usize, keys: Range<u64>, values: Range<u64>) -> Result<Self> {
        if plan.shape.residual != Residual::Single {
            return Err("GDN step binding requires single-residual hybrid semantics".into());
        }
        let Mixer::Linear(s) = plan.shape.layers.get(layer).ok_or("GDN layer missing")?.mixer else { return Err("GDN layer kind".into()); };
        crate::arms::Limits::gdn(s)?;
        let repeat = s.value_heads / s.key_heads;
        if !(1..=4).contains(&repeat) || keys.start >= keys.end || keys.end > s.key_heads
            || values.start >= values.end || values.end > s.value_heads
            || values.start != keys.start * repeat || values.end != keys.end * repeat
            || s.value_heads > u32::MAX as u64 / 16384 { return Err("GDN requires whole owned head groups".into()); }
        let epsilon = plan.epsilon as f32;
        if !epsilon.is_finite() || epsilon <= 0.0 { return Err("GDN epsilon".into()); }
        let mut precision = 1;
        for (suffix, bit, shape) in [(".A_log", 2, s.value_heads), (".dt_bias", 4, s.value_heads), (".norm.weight", 8, 128)] {
            let mut tensors = plan.shape.tensors.iter().filter(|t| t.role.place == Place::Layer(layer as u32) && t.name.ends_with(suffix));
            let t = tensors.next().ok_or("GDN parameter missing")?;
            let part = if bit == 8 { Part::LinearNorm } else { Part::LinearHeads };
            if tensors.next().is_some() || t.shape != [shape] || t.role.part != part || t.role.store != Store::Weight { return Err("GDN parameter shape/role".into()); }
            match t.dtype { Dtype::Bf16 => {}, Dtype::F32 => precision |= bit, _ => return Err("GDN parameter precision".into()) }
        }
        Ok(Self { keys: keys.start as u32..keys.end as u32, values: values.start as u32..values.end as u32,
            repeat: repeat as u32, precision, epsilon })
    }
    pub fn geometry(&self) -> [u32; 5] { [self.keys.end-self.keys.start, self.values.end-self.values.start, self.keys.start, self.values.start, self.precision] }
    /// Separate checkpoint Q|K|V, z, a and b row extents in local coordinates.
    pub fn projection_rows(&self) -> [usize; 4] {
        let [keys, heads, ..] = self.geometry();
        [(2*keys+heads) as usize*128, heads as usize*128, heads as usize, heads as usize]
    }
    /// Whole owned groups already validate repeats 1..=4; native proof is separate.
    pub fn execution_ready(&self) -> Result<()> {
        Ok(())
    }
    pub fn history_ready(&self) -> std::result::Result<(), TokenStepError> {
        let [keys, heads, key_start, value_start, precision] = self.geometry();
        if keys == 0 || heads == 0 || heads % keys != 0 || precision == 0
            || heads / keys != self.repeat || value_start != key_start * self.repeat {
            return Err(TokenStepError::HistoryShape);
        }
        Ok(())
    }
    /// Prefill/chunks/replay/verification preserve identical steps in token order.
    /// # Safety
    /// Weights are retained local conv/A_log/dt/norm views with these dtypes and
    /// owners in this backend. Each row has distinct state/history and output;
    /// live prompt spans may keep their state on chip between tokens. Descriptors are
    /// GdnDecodeRow records, with immutable state/history addresses and optional
    /// output_state snapshots. All allocations survive stream/graph replay.
    #[cfg(target_os = "linux")]
    pub unsafe fn run(&self, b: &spark_gpu_probe::driver::Backend<'_>, scratch: &mut crate::flash::gdn::GdnScratch<'_>,
        weights: [u64; 4], rows: &spark_gpu_probe::driver::DeviceVector<'_>, offset: usize, count: usize) -> Result<()> {
        self.execution_ready()?;
        scratch.step_batch_range(b, weights, rows, offset, count, self.epsilon, self.geometry())
    }
}

/// Only a live prompt's identical state/history descriptors form a token span.
/// Verification snapshots and independent chats retain their single-token path.
fn prompt_span(rows: &[f32]) -> bool {
    !rows.is_empty() && rows.len().is_multiple_of(6) && rows.as_chunks::<6>().0.iter().all(|r|
        r[..4].iter().zip(&rows[..4]).all(|(a,b)| a.to_bits()==b.to_bits())
            && r[4].to_bits()==0 && r[5].to_bits()==0)
}

fn live_span(rows: &[f32]) -> usize {
    rows.as_chunks::<6>().0.iter().take_while(|r| r.iter().zip(&rows[..6]).all(|(a,b)| a.to_bits()==b.to_bits())).count()
}

/// Repeated state/history addresses imply ordered tokens or snapshots, not chats.
fn independent_chats(rows: &[f32]) -> bool {
    if rows.len() < 12 || !rows.len().is_multiple_of(6) { return false; }
    let mut states = std::collections::BTreeSet::new();
    let mut histories = std::collections::BTreeSet::new();
    rows.as_chunks::<6>().0.iter().all(|r| {
        let address = |i: usize| u64::from(r[i].to_bits()) | (u64::from(r[i+1].to_bits()) << 32);
        address(4) == 0 && address(0) != 0 && address(2) != 0
            && states.insert(address(0)) && histories.insert(address(2))
    })
}

/// Read-only span preparation, ordered recurrence, then history publication.
#[cfg(target_os = "linux")]
pub(crate) struct TokenStep<'a> { work: Vec<spark_gpu_probe::driver::DeviceVector<'a>> }
#[cfg(target_os = "linux")]
impl<'a> TokenStep<'a> {
    pub(crate) fn new(b: &spark_gpu_probe::driver::Backend<'a>, heads: usize) -> Result<Self> {
        Ok(Self { work:b.vector_arena(&[heads*128,heads*128,heads*128,heads,heads,heads*128])? })
    }
    pub(crate) fn run(&mut self, binding: &StepBinding, b: &spark_gpu_probe::driver::Backend<'_>, scratch: &mut crate::flash::gdn::GdnScratch<'_>,
        weights: [u64;4], rows: &spark_gpu_probe::driver::DeviceVector<'_>, descriptors: &[f32], offset: usize, count: usize, history_outputs: &[u64]) -> std::result::Result<(), TokenStepError> {
        use spark_gpu_probe::{kernel_params::{GdnPrepare,GdnPrepareHist,GdnHistory,GdnGate},gdn_params::{GdnStepBatch,GdnStepGateBatch}};
        binding.execution_ready()?;
        binding.history_ready()?;
        let [keys,heads,key_start,value_start,precision] = binding.geometry(); let width=(2*keys+heads) as usize*128; let hd=heads as usize*128;
        if count==0 || rows.len()!=count*6 || descriptors.len()!=rows.len() || !rows.device_address().is_multiple_of(8) || weights.contains(&0) || self.work[0].len()<hd || !history_outputs.is_empty() && history_outputs.len()!=count { return Err(TokenStepError::Execution("token-step descriptors or workspace".into())); }
        if history_outputs.is_empty() && independent_chats(descriptors) {
            let address=|v:&spark_gpu_probe::driver::DeviceVector<'_>,n:usize| v.slice(offset*n..(offset+count)*n).map(|s| s.device_address());
            let [q,k,v,beta,g,_core]=scratch.token_workspace(offset,count)?;
            let x=address(&scratch.qkv,width)?;
            // Two launches: conv-prepare publishes each row's history, and the
            // step gates its own core rows through shared memory.
            let mut prepare=GdnPrepareHist { rows:rows.device_address(), snapshot:0, prepare:GdnPrepare {
                x:x as _,w:weights[0] as _,history:std::ptr::null(),a:address(&scratch.a,heads as usize)? as _,b:address(&scratch.b,heads as usize)? as _,
                alog:weights[1] as _,dt:weights[2] as _,q:q as _,k:k as _,v:v as _,beta:beta as _,g:g as _,
                key_heads:keys,value_heads:heads,key_start,value_start,tokens:1,precision,scale:0 } };
            let mut step=GdnStepGateBatch {q,k,v,beta,g,rows:rows.device_address(),z:address(&scratch.z,hd)?,norm:weights[3],
                out:address(&scratch.gated,hd)?,heads,count:count as u32,precision,epsilon:binding.epsilon};
            unsafe {
                b.launch_flash_rows("flash_gdn_conv_prepare_hist",(keys*32).div_ceil(128),count as u32,128,0,&mut [(&mut prepare as *mut GdnPrepareHist).cast()])?;
                b.launch_flash_shared("flash_gdn_step_gate_batch",heads*count as u32,128,0,&mut [(&mut step as *mut GdnStepGateBatch).cast()])?;
            }
            return Ok(());
        } else {
        let mut begin = 0;
        while begin < count {
            // Each live prompt span keeps its ordered recurrence and history.
            let tail = &descriptors[begin*6..];
            let span = if history_outputs.is_empty() && prompt_span(&tail[..6]) {
                live_span(tail)
            } else { 1 };
            let index=offset+begin; let address=|v:&spark_gpu_probe::driver::DeviceVector<'_>,n:usize| v.slice(index*n..(index+span)*n).map(|s| s.device_address());
            let history = u64::from(descriptors[begin*6+2].to_bits()) | (u64::from(descriptors[begin*6+3].to_bits())<<32);
            let [q,k,v,beta,g,_] = if span>1 { scratch.token_workspace(index,span)? }
                else { [0,1,2,3,4,5].map(|i| self.work[i].device_address()) };
            let core = scratch.token_workspace(index,span)?[5];
            let x=address(&scratch.qkv,width)?;
            let mut prepare=GdnPrepare { x:x as _,w:weights[0] as _,history:history as _,a:address(&scratch.a,heads as usize)? as _,b:address(&scratch.b,heads as usize)? as _,
                alog:weights[1] as _,dt:weights[2] as _,q:q as _,k:k as _,v:v as _,beta:beta as _,g:g as _,key_heads:keys,value_heads:heads,key_start,value_start,tokens:span as u32,precision,scale:0 };
            unsafe {
                if span==1 {
                    // Two launches: conv-prepare publishes this token's history
                    // (and any snapshot copy); the step gates its own core row.
                    let mut fused=GdnPrepareHist {rows:0,snapshot:history_outputs.get(begin).copied().unwrap_or(0),prepare};
                    b.launch_flash_shared("flash_gdn_conv_prepare_hist",(keys*32).div_ceil(128),128,0,&mut [(&mut fused as *mut GdnPrepareHist).cast()])?;
                    let mut step=GdnStepGateBatch {q,k,v,beta,g,rows:rows.device_address()+(begin*24) as u64,
                        z:address(&scratch.z,hd)?,norm:weights[3],out:address(&scratch.gated,hd)?,heads,count:1,precision,epsilon:binding.epsilon};
                    b.launch_flash_shared("flash_gdn_step_gate_batch",heads,128,0,&mut [(&mut step as *mut GdnStepGateBatch).cast()])?;
                } else {
                    b.launch_flash_shared("flash_gdn_conv_prepare_tokens",(keys*span as u32*32).div_ceil(128),128,0,&mut [(&mut prepare as *mut GdnPrepare).cast()])?;
                    let mut step=GdnStepBatch {q,k,v,beta,g,rows:rows.device_address()+(begin*24) as u64,
                        out:core,sessions:0,heads,count:span as u32};
                    // A scan's columns are independent; one warp per CTA spreads
                    // long recurrences across SMs without changing a column's sums.
                    b.launch_flash_shared("flash_gdn_scan_exact",heads*4,32,0,&mut [(&mut step as *mut GdnStepBatch).cast()])?;
                    let mut update=GdnHistory {x:x as _,history:history as _,rows:history_outputs.get(begin).copied().unwrap_or(0),channels:width as u32,tokens:span as u32,precision:1};
                    b.launch_flash_shared("flash_gdn_history",(width as u32).div_ceil(128),128,0,&mut [(&mut update as *mut GdnHistory).cast()])?;
                    // Gate this span's independent token-major core rows; the
                    // single-token spans above gate inside their steps instead.
                    let mut gate=GdnGate {x:core as _,z:address(&scratch.z,hd)? as _,norm:weights[3] as _,
                        output:address(&scratch.gated,hd)? as _,heads:heads*span as u32,tokens:1,precision,packed:0,epsilon:binding.epsilon,sessions:0};
                    b.launch_flash_shared("flash_gdn_gate",(heads*span as u32*32).div_ceil(128),128,0,&mut [(&mut gate as *mut GdnGate).cast()])?;
                }
            }
            begin += span;
        }
        }
        Ok(())
    }
}

#[cfg(test)] #[test] fn prompt_span_requires_live_identical_descriptors() {
    let row=[f32::from_bits(0x7fc00001),0.0,f32::from_bits(0x7fc00002),0.0,0.0,0.0];
    for n in [1,2,63,64,65,255,256,257,8192] { assert!(prompt_span(&row.repeat(n))); }
    for word in 0..6 { let mut rows=row.repeat(2); rows[6+word]=f32::from_bits(1); assert!(!prompt_span(&rows)); }
    assert!(!prompt_span(&[]) && !prompt_span(&row[..5]));
    let chats=[1,0,2,0,0,0,3,0,4,0,0,0].map(f32::from_bits);
    assert!(independent_chats(&chats));
    let spans = [row.repeat(256), chats[6..].repeat(128)].concat(); assert_eq!(live_span(&spans), 256); assert_eq!(live_span(&spans[256*6..]), 128);
    for word in [0,2,4] { let mut rows=chats; rows[6+word]=rows[word]; if word==4 { rows[10]=f32::from_bits(1); } assert!(!independent_chats(&rows)); }
    assert!(!independent_chats(&row.repeat(2)) && !independent_chats(&chats[..6]));
}

// This module is nested in Decode; its child can observe the target's private hidden row.
#[cfg(target_os="linux")] #[path="mtp_step.rs"] mod mtp_step;

#[cfg(test)] #[test] fn fused_history_rejects_incomplete_shapes() {
    for (keys,heads,precision,start,repeat,ok) in [(2,2,1,0,1,true),(2,6,1,0,3,true),(2,4,1,0,2,true),(2,8,15,0,4,true),(2,4,1,4,2,true),(0,4,1,0,2,false),(2,0,1,0,2,false),(2,5,1,0,2,false),(2,4,0,0,2,false),(2,4,1,1,2,false),(2,4,1,0,3,false)] { let binding=StepBinding { keys: (start/2)..(start/2+keys), values:start..(start+heads), precision,repeat,epsilon:1e-6 }; assert_eq!(binding.history_ready(),if ok {Ok(())} else {Err(TokenStepError::HistoryShape)}); } }
