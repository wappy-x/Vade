//! Checked local ownership and explicit positions for the shared BF16 causal row.
use engine_model::{plan::Rope, shapes::AttentionShape};
use spark_gpu_probe::kernel_params::{DenseAttention, QsaPrepare, DENSE_VALUE_GROUP};
use std::ops::Range;
type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
pub struct FullAttentionBinding { query: Range<u32>, key: Range<u32>, epsilon: f32 }
impl FullAttentionBinding {
    pub fn new(shape: AttentionShape, rope: &Rope, epsilon: f32,
        query: Range<u32>, key: Range<u32>) -> Result<Self> {
        crate::arms::Limits::attention(shape,rope)?;
        if shape.index.is_some() || !shape.gated || !rope.theta.is_finite() || rope.theta <= 0.0
            || rope.sections.len() != 3 || rope.sections.iter().try_fold(0u64, |a,&b| a.checked_add(b)) != Some(rope.rotary_dim/2)
            || !rope.interleaved || !epsilon.is_finite() || epsilon <= 0.0
            || shape.kv_heads == 0 || shape.query_heads % shape.kv_heads != 0
            || query.start >= query.end || key.start >= key.end
            || u64::from(query.end) > shape.query_heads || u64::from(key.end) > shape.kv_heads {
            return Err("unsupported full attention geometry".into());
        }
        let group = shape.query_heads / shape.kv_heads;
        if group == 0 || group > u64::from(DENSE_VALUE_GROUP)
            || u64::from(query.start) != u64::from(key.start)*group
            || u64::from(query.end) != u64::from(key.end)*group {
            return Err("full attention requires complete local query/KV groups".into());
        }
        Ok(Self { query, key, epsilon })
    }
    /// Per-cache K or V byte extent; physical storage is [position,local KV,256] BF16.
    pub fn kv_bytes(&self, capacity: u32) -> Result<u64> {
        u64::from(capacity).checked_mul(u64::from(self.key.end-self.key.start)*512).ok_or("KV extent overflow".into())
    }
    /// Buffers: Q/gate, projected K, projected V, normalized Q, gated output,
    /// BF16 K cache, BF16 V cache, FP32 scores (null is allowed for short rows).
    /// Norms are complete 256-element BF16 vectors; frequencies are 32 FP32
    /// values derived from the plan's theta. Text uses one position on all axes.
    pub fn row(&self, past: u32, position: u32, capacity: u32, norms: [u64;2], frequencies: u64,
        buffers: [u64;8]) -> Result<(QsaPrepare, DenseAttention)> {
        let visible = past.checked_add(1).filter(|&v| v <= capacity).ok_or("causal row exceeds KV capacity")?;
        let offset = position.checked_sub(past).ok_or("RoPE position precedes cache coordinate")?;
        let (heads, keys) = (self.query.end-self.query.start, self.key.end-self.key.start);
        let score_bytes = u64::from(heads).checked_mul(u64::from(capacity)).and_then(|n| n.checked_mul(4)).ok_or("score extent overflow")?;
        let [qg,k,v,q,out,kc,vc,scores] = buffers;
        let spans = [(qg,u64::from(heads)*2048,4),(k,u64::from(keys)*1024,4),(v,u64::from(keys)*1024,4),
            (q,u64::from(heads)*1024,4),(out,u64::from(heads)*1024,4),(kc,self.kv_bytes(capacity)?,16),
            (vc,self.kv_bytes(capacity)?,16),(norms[0],512,2),(norms[1],512,2),(frequencies,128,4)];
        if spans.iter().any(|&(p,n,a)| p == 0 || p%a != 0 || p.checked_add(n).is_none())
            || heads.checked_add(keys).is_none() || keys.checked_mul(visible.div_ceil(32)).is_none()
            || visible > 512 && (scores == 0 || scores%4 != 0 || scores.checked_add(score_bytes).is_none()) {
            return Err("invalid full attention address spans".into());
        }
        Ok((QsaPrepare { query_gate:qg,key:k,value:v,query_weight:norms[0],key_weight:norms[1],query:q,
            keys:kc,values:vc,heads,kv_heads:keys,tokens:1,past,position_offset:offset,frequencies,
            precision:1,epsilon:self.epsilon,replay_position:0 },
            DenseAttention { query:q,query_gate:qg,output:out,rows:0,scores:if visible>512 {scores} else {0},
                replay_position:0,keys:kc,values:vc,query_heads:heads,key_heads:keys,count:1,
                score_stride:capacity,score_tiles:visible.div_ceil(32),past }))
    }
}
