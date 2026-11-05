//! A session-owned program with immutable operators and causal cache geometry.
use super::*;

/// The private program is validated once. A position change only adjusts cache
/// extents. Callers cannot change its operations, weights, settings, or topology.
pub struct LayerSchedule {plan:LayerPlan,maximum:u64}
impl LayerSchedule {
    pub fn new(d:&TextDescription,layer:usize,topology:Topology)->Result<Self> {
        Ok(Self {plan:LayerPlan::new(d,layer,1,topology)?,maximum:d.max_positions.min(u32::MAX as u64)})
    }
    /// `context` includes this token. Earlier rows occupy `context - 1` slots.
    /// Check the bound before changing any part of the program.
    pub fn at(&mut self,context:u64)->Result<&LayerPlan> {
        need(context>0 && context<=self.maximum,"invalid cached layer context")?;
        if context==self.plan.context {return Ok(&self.plan);}
        if let TokenMixer::LatentAttention(a)=self.plan.settings.decoder.mixer {
            let elements=context.checked_mul(a.kv_rank.max(a.indexer.dim));
            need(elements.is_some_and(|n|n<=isize::MAX as u64/4),"cached layer buffer extent overflow")?;
            for buffer in &mut self.plan.inputs {
                if matches!(buffer.name.as_str(),"latent_cache"|"index_keys"|"index_gates") {buffer.shape[0]=context-1;}
            }
            for step in &mut self.plan.steps {match step.kind {
                Kind::CacheAppend=>step.outputs[0].shape[0]=context,
                Kind::Pool=>{step.outputs[0].shape[0]=context;step.outputs[1].shape[0]=context;step.outputs[2].shape[0]=context/a.indexer.pool;},
                Kind::ScorePools=>step.outputs[0].shape[0]=context/a.indexer.pool,
                _=>{},
            }}
        }
        self.plan.context=context;Ok(&self.plan)
    }
}
