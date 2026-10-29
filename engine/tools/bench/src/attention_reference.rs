//! Independent causal attention oracle with BF16 KV and explicit text positions.
//! Host transcendental functions specify semantics, not CUDA approximate bits.
use crate::fp8_reference::{encode_bf16, round_bf16 as r, widen_bf16 as bf};
pub type Result<T> = std::result::Result<T, &'static str>;
pub const DIM: usize = 256;

fn warp_sum(mut lanes: [f32;32]) -> f32 {
    for distance in [16,8,4,2,1] { let previous=lanes; for lane in 0..32 { lanes[lane]=previous[lane]+previous[lane^distance]; } }
    lanes[0]
}
/// Zero-centered full-head RMSNorm, then rotate only channels 0..64.
pub fn prepare_head(input: &[f32], norm: &[u16], frequencies: &[f32], position: u32, epsilon: f32) -> Result<Vec<f32>> {
    if input.len()!=DIM || norm.len()!=DIM || frequencies.len()!=32 || !epsilon.is_finite() || epsilon<=0.0
        || input.iter().chain(frequencies).any(|x| !x.is_finite()) || norm.iter().any(|&x| !bf(x).is_finite()) {
        return Err("attention head operands");
    }
    let mut lanes=[0.0;32];
    for lane in 0..32 { for part in 0..8 { let x=input[lane+part*32]; lanes[lane]+=x*x; } }
    let scale=1.0/(warp_sum(lanes)/256.0+epsilon).sqrt();
    let mut out: Vec<_>=input.iter().zip(norm).map(|(&x,&w)| r(x*scale*(1.0+bf(w)))).collect();
    for i in 0..32 {
        let angle=position as f32*frequencies[i];
        let turns=(f64::from(angle)/std::f64::consts::TAU) as i64;
        let angle=(f64::from(angle)-turns as f64*std::f64::consts::TAU) as f32;
        let (s,c)=(r(angle.sin()),r(angle.cos())); let (a,b)=(out[i],out[i+32]);
        out[i]=r(r(a*c)-r(b*s)); out[i+32]=r(r(b*c)+r(a*s));
    }
    if out.iter().any(|x| !x.is_finite()) { return Err("attention head overflow"); }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct AttentionReference {
    heads: usize, keys_per_row: usize, epsilon: f32, frequencies: Vec<f32>,
    pub keys: Vec<u16>, pub values: Vec<u16>, pub positions: Vec<u32>,
}
impl AttentionReference {
    pub fn new(heads: usize, keys: usize, theta: f32, epsilon: f32) -> Result<Self> {
        if heads==0 || keys==0 || !heads.is_multiple_of(keys) || heads/keys>16 || heads.checked_mul(512).is_none()
            || !theta.is_finite() || theta<=0.0 || !epsilon.is_finite() || epsilon<=0.0 { return Err("attention geometry"); }
        Ok(Self { heads,keys_per_row:keys,epsilon,frequencies:(0..32).map(|i| 1.0/theta.powf(i as f32/32.0)).collect(),
            keys:Vec::new(),values:Vec::new(),positions:Vec::new() })
    }
    /// One row is the only arithmetic entry for every scheduling phase.
    /// Q/gate is [local head,512]; K/V are [local KV head,256].
    pub fn step(&mut self, qg: &[f32], k: &[f32], v: &[f32], norms: [&[u16];2], position: u32) -> Result<Vec<f32>> {
        let width=self.keys_per_row*DIM;
        if qg.len()!=self.heads*512 || k.len()!=width || v.len()!=width || norms.iter().any(|n| n.len()!=DIM)
            || qg.iter().chain(k).chain(v).any(|x| !x.is_finite()) || (position as usize)<self.positions.len()
            || self.keys.len()!=self.positions.len()*width || self.values.len()!=self.keys.len() { return Err("attention row operands"); }
        let mut query=Vec::new(); let mut keys=Vec::new();
        for head in 0..self.heads { query.extend(prepare_head(&qg[head*512..head*512+DIM],norms[0],&self.frequencies,position,self.epsilon)?); }
        for head in 0..self.keys_per_row { keys.extend(prepare_head(&k[head*DIM..(head+1)*DIM],norms[1],&self.frequencies,position,self.epsilon)?.into_iter().map(encode_bf16)); }
        let values: Vec<_>=v.iter().copied().map(encode_bf16).collect();
        if values.iter().any(|&x| !bf(x).is_finite()) { return Err("attention value overflow"); }
        let visible=self.positions.len()+1; let mut output=Vec::with_capacity(self.heads*DIM);
        for head in 0..self.heads {
            let key_head=head/(self.heads/self.keys_per_row); let mut scores=Vec::with_capacity(visible);
            for pos in 0..visible {
                let cache=if pos+1==visible { &keys[..] } else { &self.keys[pos*width..(pos+1)*width] };
                let mut lanes=[0.0;32];
                for lane in 0..32 { for part in 0..8 { let i=lane+part*32; lanes[lane]+=query[head*DIM+i]*bf(cache[key_head*DIM+i]); } }
                scores.push(r(warp_sum(lanes))*(1.0/16.0));
            }
            let maximum=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
            let probabilities: Vec<_>=scores.iter().map(|&x| (x-maximum).exp()).collect();
            let denominator=probabilities.iter().fold(0.0f32,|sum,&x| sum+x);
            let weights: Vec<_>=probabilities.iter().map(|&x| r(x/denominator)).collect();
            for feature in 0..DIM {
                let mut sum=0.0;
                for (pos,&weight) in weights.iter().enumerate() {
                    let cache=if pos+1==visible { &values[..] } else { &self.values[pos*width..(pos+1)*width] };
                    sum+=weight*bf(cache[key_head*DIM+feature]);
                }
                output.push(r(r(sum)*r(1.0/(1.0+(-qg[head*512+DIM+feature]).exp()))));
            }
        }
        if output.iter().any(|x| !x.is_finite()) { return Err("attention output overflow"); }
        self.keys.extend(keys); self.values.extend(values); self.positions.push(position);
        Ok(output)
    }
}
