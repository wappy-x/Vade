//! Checked storage for replicated embedding, stream mean, and stored MXFP8 head.
//! These candidate primitive edges do not admit a decoder program.
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape { hidden: u32, vocab: u32, streams: u32 }
impl Shape {
    pub fn new(hidden: u64, vocab: u64, streams: u64) -> Result<Self, String> {
        if hidden == 0 || !hidden.is_multiple_of(32) || vocab == 0 || vocab > 1 << 24 || streams == 0 {
            return Err("invalid text edge dimensions".into());
        }
        let dim = |v| u32::try_from(v).map_err(|_| "text edge dimension exceeds ABI".to_string());
        let s = Self {hidden:dim(hidden)?,vocab:dim(vocab)?,streams:dim(streams)?};
        if hidden.checked_mul(streams).is_none_or(|n| n > u32::MAX as u64) {
            return Err("text residual exceeds ABI".into());
        }
        s.weight_bytes()?; Ok(s)
    }
    pub fn hidden(self) -> usize { self.hidden as usize }
    pub fn vocab(self) -> usize { self.vocab as usize }
    pub fn streams(self) -> usize { self.streams as usize }
    pub fn residual(self) -> usize { self.hidden() * self.streams() }
    pub fn weight_bytes(self) -> Result<[usize;3],String> {
        let n = self.hidden().checked_mul(self.vocab()).ok_or("text weight size overflow")?;
        Ok([n.checked_mul(2).ok_or("embedding size overflow")?,n,n/32])
    }
    pub fn token(self, token: u32) -> Result<(),String> {
        if token as usize >= self.vocab() {Err("text token outside vocabulary".into())} else {Ok(())}
    }
}
/// Offsets into one immutable arena. Byte lengths follow Shape and stored types:
/// BF16 embedding, E4M3 head, and one UE8M0 scale for each head row's 32 columns.
/// Norm weights keep their original stored dtype and bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormStorage { Bf16, F32 }
impl NormStorage {pub fn bytes(self)->usize {match self {Self::Bf16=>2,Self::F32=>4}}}
#[derive(Debug, Clone, Copy)]
pub struct Layout { pub shape: Shape, pub embedding: usize, pub head: usize, pub scales: usize,
    pub norm: usize, pub norm_storage: NormStorage, pub epsilon: f32 }
impl Layout {
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    pub fn ranges(self, arena_bytes: usize) -> Result<[Range<usize>;4],String> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {return Err("invalid text norm epsilon".into());}
        let offsets = [self.embedding,self.head,self.scales,self.norm]; let w = self.shape.weight_bytes()?;
        let norm_bytes = self.shape.hidden().checked_mul(self.norm_storage.bytes()).ok_or("text norm size overflow")?;
        let sizes = [w[0],w[1],w[2],norm_bytes];
        let mut ranges = [0..0,0..0,0..0,0..0];
        for i in 0..4 {
            let end = offsets[i].checked_add(sizes[i]).ok_or("text weight offset overflow")?;
            if end > arena_bytes || offsets[i] % [2,16,1,self.norm_storage.bytes()][i] != 0 {return Err("text weight span is invalid".into());}
            ranges[i] = offsets[i]..end;
            if ranges[..i].iter().any(|r| r.start < end && offsets[i] < r.end) {return Err("text weight spans overlap".into());}
        }
        Ok(ranges)
    }
}
/// Selection of complete finite logits. Equal maxima keep the lowest token ID.
pub fn greedy(logits: &[f32]) -> Result<u32,String> {
    if logits.is_empty() || logits.len() > u32::MAX as usize || logits.iter().any(|x| !x.is_finite()) {
        return Err("text logits must be finite and nonempty".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {if logits[i] > logits[best] {best = i;}}
    Ok(best as u32)
}
