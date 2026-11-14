//! Independent scalar host recurrence for projected synthetic rows, never serving.
//! Host exp/log/sqrt specify this oracle; CUDA approximations need native proof.
use crate::fp8_reference::{round_bf16 as bf, widen_bf16};
type Result<T> = std::result::Result<T, &'static str>;
const D: usize = 128;
#[derive(Clone, Copy)]
pub enum Parameter<'a> { Bf16(&'a [u16]), F32(&'a [f32]) }
impl Parameter<'_> {
    fn len(self) -> usize { match self { Self::Bf16(v) => v.len(), Self::F32(v) => v.len() } }
    fn at(self, i: usize) -> f32 { match self { Self::Bf16(v) => widen_bf16(v[i]), Self::F32(v) => v[i] } }
}
#[derive(Clone, Copy)]
pub struct Layout { pub keys: usize, pub values: usize, pub repeat: usize, pub key_start: usize, pub value_start: usize }
impl Layout {
    fn width(self) -> Result<usize> {
        if self.keys == 0 || self.values == 0 || !(1..=4).contains(&self.repeat)
            || self.keys.checked_mul(self.repeat) != Some(self.values)
            || self.key_start.checked_mul(self.repeat) != Some(self.value_start)
            || self.values > u32::MAX as usize/(D*D) || self.key_start.checked_add(self.keys).is_none()
            || self.value_start.checked_add(self.values).is_none() { return Err("GDN owned groups"); }
        self.keys.checked_mul(2).and_then(|n| n.checked_add(self.values)).and_then(|n| n.checked_mul(D)).ok_or("GDN width overflow")
    }
}
pub struct Weights<'a> { pub conv: &'a [u16], pub alog: Parameter<'a>, pub bias: Parameter<'a>, pub norm: Parameter<'a>, pub epsilon: f32 }
pub struct Row<'a> { pub qkv: &'a [f32], pub z: &'a [f32], pub a: &'a [f32], pub b: &'a [f32] }
#[derive(Clone, Debug, PartialEq)]
pub struct State { pub cells: Vec<f32>, pub history: Vec<u16> }
impl State {
    pub fn zero(layout: Layout) -> Result<Self> {
        let history = layout.width()?.checked_mul(4).ok_or("GDN history overflow")?;
        let cells = layout.values.checked_mul(D*D).ok_or("GDN state overflow")?;
        Ok(Self { cells: vec![0.0; cells], history: vec![0; history] })
    }
    /// Stable receipt over exact state/history storage bits; clone is fork/restore.
    pub fn hashes(&self) -> [u64; 2] {
        fn hash(bytes: impl Iterator<Item=u8>) -> u64 { bytes.fold(0xcbf29ce484222325, |h,b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3)) }
        [hash(self.cells.iter().flat_map(|v| v.to_bits().to_le_bytes())), hash(self.history.iter().flat_map(|v| v.to_le_bytes()))]
    }
}
fn sigmoid(x: f32) -> f32 { 1.0/(1.0+(-x).exp()) }
fn squares(row: &[f32]) -> f32 {
    let mut lanes = [0.0;32];
    for lane in 0..32 { for feature in 0..4 { let x=row[lane+feature*32]; lanes[lane] += x*x; } }
    for distance in [16,8,4,2,1] { let prior=lanes; for lane in 0..32 { lanes[lane]=prior[lane]+prior[lane ^ distance]; } }
    lanes[0]
}
/// BF16 conv/history and projected boundaries, FP32 recurrent cells; SiLU z gate.
pub fn step(layout: Layout, w: &Weights<'_>, state: &mut State, row: Row<'_>) -> Result<Vec<f32>> {
    let width=layout.width()?; let hd=layout.values.checked_mul(D).ok_or("GDN extent overflow")?;
    if w.conv.len()!=width*4 || w.alog.len()!=layout.values || w.bias.len()!=layout.values || w.norm.len()!=D
        || !w.epsilon.is_finite() || w.epsilon<=0.0 || row.qkv.len()!=width || row.z.len()!=hd
        || row.a.len()!=layout.values || row.b.len()!=layout.values || state.cells.len()!=hd*D || state.history.len()!=width*4
        || row.qkv.iter().chain(row.z).chain(row.a).chain(row.b).chain(&state.cells).any(|x|!x.is_finite())
        || w.conv.iter().chain(&state.history).any(|&x|!widen_bf16(x).is_finite())
        || [w.alog,w.bias,w.norm].iter().any(|p|(0..p.len()).any(|i|!p.at(i).is_finite())) { return Err("GDN operands"); }
    let mut next=state.clone(); let mut convolved=vec![0.0;width];
    for (channel,y) in convolved.iter_mut().enumerate() {
        let mut sum=0.0;
        for tap in 0..4 { let x=if tap==3 { bf(row.qkv[channel]) } else { widen_bf16(state.history[channel*4+tap+1]) }; sum += x*widen_bf16(w.conv[channel*4+tap]); }
        let x=bf(sum); *y=bf(x*sigmoid(x));
        next.history[channel*4..channel*4+3].copy_from_slice(&state.history[channel*4+1..channel*4+4]);
        next.history[channel*4+3]=(bf(row.qkv[channel]).to_bits()>>16) as u16;
    }
    let mut output=vec![0.0;hd];
    for head in 0..layout.values {
        let key=(head+layout.value_start)/layout.repeat-layout.key_start;
        let query=&convolved[key*D..(key+1)*D]; let keys=&convolved[(layout.keys+key)*D..(layout.keys+key+1)*D];
        let qs=1.0/(squares(query)+1e-6).sqrt(); let ks=1.0/(squares(keys)+1e-6).sqrt();
        let a=bf(row.a[head])+w.bias.at(head); let softplus=if a>20.0 { a } else { a.exp().ln_1p() };
        let decay=(-w.alog.at(head).exp()*softplus).exp(); let beta=bf(sigmoid(bf(row.b[head])));
        let mut core=[0.0;D];
        for (feature,y) in core.iter_mut().enumerate() {
            let mut prediction=0.0;
            for (coordinate,&k) in keys.iter().enumerate() { let at=head*D*D+coordinate*D+feature; next.cells[at]=state.cells[at]*decay; prediction += (k*ks)*next.cells[at]; }
            let delta=(convolved[(2*layout.keys+head)*D+feature]-prediction)*beta;
            for coordinate in 0..D { let at=head*D*D+coordinate*D+feature; next.cells[at] += (keys[coordinate]*ks)*delta; *y += ((query[coordinate]*qs)*(1.0/(D as f32).sqrt()))*next.cells[at]; }
            *y=bf(*y);
        }
        let scale=1.0/(squares(&core)/D as f32+w.epsilon).sqrt();
        for feature in 0..D { let normalized=bf(bf(core[feature]*scale)*w.norm.at(feature)); let z=bf(row.z[head*D+feature]); output[head*D+feature]=bf((normalized*sigmoid(z))*z); }
    }
    if output.iter().chain(&next.cells).any(|x|!x.is_finite()) { return Err("GDN arithmetic overflow"); }
    *state=next; Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_history_and_restore() {
        assert_eq!(Parameter::F32(&[1.0001]).at(0).to_bits(),1.0001f32.to_bits()); assert_ne!(Parameter::F32(&[1.0001]).at(0).to_bits(),bf(1.0001).to_bits());
        for repeat in [2,4] { let l=Layout {keys:1,values:repeat,repeat,key_start:3,value_start:3*repeat}; let width=l.width().unwrap(); let conv=vec![0x3e80;width*4]; let p=vec![0u16;repeat]; let norm=vec![0x3f80;D]; let f=vec![0.0;repeat];
            let w=Weights {conv:&conv,alog:Parameter::Bf16(&p),bias:Parameter::Bf16(&p),norm:Parameter::Bf16(&norm),epsilon:1e-6}; let wf=Weights {alog:Parameter::F32(&f),norm:Parameter::F32(&vec![1.0;D]),..w}; let z=vec![0.5;repeat*D]; let x:Vec<_>=(0..width).map(|i|(i%17) as f32/32.0).collect(); let mut s=State::zero(l).unwrap(); let mut fp=s.clone(); let mut hashes=Vec::new();
            for _ in 0..7 { let before=s.clone(); let y=step(l,&w,&mut s,Row {qkv:&x,z:&z,a:&f,b:&f}).unwrap(); assert_eq!(y,step(l,&wf,&mut fp,Row {qkv:&x,z:&z,a:&f,b:&f}).unwrap()); let mut fork=before; assert_eq!(y,step(l,&w,&mut fork,Row {qkv:&x,z:&z,a:&f,b:&f}).unwrap()); assert_eq!(fork.hashes(),s.hashes()); hashes.push(s.hashes()); } assert!(hashes.windows(2).all(|h|h[0]!=h[1])); assert_eq!(fp,s);
            let before=s.clone(); assert!(step(l,&w,&mut s,Row {qkv:&[],z:&z,a:&f,b:&f}).is_err()); assert_eq!(s,before);
        }
    }
}
