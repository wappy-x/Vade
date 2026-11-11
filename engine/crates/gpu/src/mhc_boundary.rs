//! BF16 mHC boundaries with FP32 lanes. These do not select a framework backend.
//! Source: CAST-CONTRACT.md, saved forward lines 328 and 1336-1347.
use crate::bf16::rounded;

/// Round a resident FP32 vector to BF16 values in FP32 lanes. Input and output
/// ranges are disjoint and each contains `words` four-byte elements.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct LaneCast { pub input:u64, pub output:u64, pub words:u32 }

/// Collapse uses FP32 pre coefficients. Round the completed sum once.
pub fn collapse(input: &[f32], pre: &[f32], streams: usize, hidden: usize, feature: usize) -> f32 {
    let mut value=0.0f32;
    for src in 0..streams { value+=pre[src]*input[src*hidden+feature]; }
    rounded(value)
}
/// Round post and comb coefficients first. Round the residual dot and block
/// product separately. Round their final sum. The source stream order is fixed.
pub fn combine(input: &[f32], coefficients: &[f32], block: f32,
    streams: usize, hidden: usize, destination: usize, feature: usize) -> f32 {
    let mut residual=0.0f32;
    for src in 0..streams {
        residual+=rounded(coefficients[2*streams+src*streams+destination])*input[src*hidden+feature];
    }
    let block=rounded(rounded(coefficients[streams+destination])*block);
    rounded(rounded(residual)+block)
}

/// Check the host upload before it receives a BF16 lane type. Signed zero is
/// preserved. Device arithmetic can still produce nonfinite BF16 results.
pub fn check_bf16(values: &[f32]) -> Result<(), &'static str> {
    if values.is_empty() || values.iter().any(|&v| !v.is_finite() || rounded(v).to_bits()!=v.to_bits()) {
        return Err("BF16 lanes require nonempty, finite, exact BF16 values");
    }
    Ok(())
}

/// FP32 physical spans. A BF16 lane still occupies four bytes. This descriptor
/// never represents packed BF16 or authorizes a cache storage dtype.
#[derive(Debug, Clone, Copy)]
pub struct Span { pub address: u64, pub words: usize }
/// Reject short views and aliases before a launch. Inputs are disjoint too.
pub fn check_spans(spans: &[Span], lengths: &[usize]) -> Result<(), &'static str> {
    if spans.len()!=lengths.len() { return Err("mHC plane count differs"); }
    for (i,(span,&words)) in spans.iter().zip(lengths).enumerate() {
        if words==0 || span.words!=words || span.address==0 || span.address%4!=0 { return Err("mHC plane extent or alignment differs"); }
        let bytes=u64::try_from(words).ok().and_then(|n| n.checked_mul(4)).ok_or("mHC plane extent overflow")?;
        let end=span.address.checked_add(bytes).ok_or("mHC plane address overflow")?;
        for other in &spans[..i] {
            let other_end=other.address.checked_add((other.words as u64)*4).ok_or("mHC plane address overflow")?;
            if span.address<other_end && other.address<end { return Err("mHC planes overlap"); }
        }
    }
    Ok(())
}
