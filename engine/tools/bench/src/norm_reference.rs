//! Host trace of fixed order=2, block128 hybrid norms. No CUDA or checkpoint I/O.
//! This specifies ordering and BF16 boundaries; host sqrt is not a native bit proof.
use crate::fp8_reference::{round_bf16, widen_bf16};
pub type Result<T> = std::result::Result<T, &'static str>;
pub const THREADS: usize = 128;
pub const ORDER: u32 = 2;

#[derive(Debug, Clone, PartialEq)]
pub struct SquareTrace { pub lanes: [f32; 32], pub warp: [[f32; 32]; 5], pub sum: f32 }
#[derive(Debug, Clone, PartialEq)]
pub struct Normalized { pub hidden: Vec<f32>, pub output: Vec<f32>, pub traces: Vec<SquareTrace> }

/// Original feature coordinates of one independent adjacent accumulation chain.
pub fn chain_indices(width: usize, lane: usize, group: usize, adjacent: usize) -> Result<Vec<usize>> {
    if width == 0 || width > 5120 || lane >= 32 || group >= 16 || adjacent >= 4 {
        return Err("norm chain dimensions");
    }
    Ok((((lane + group*32)*4 + adjacent)..width).step_by(2048).collect())
}

pub fn square_trace(input: &[f32]) -> Result<SquareTrace> {
    let width = input.len();
    if width == 0 || width > 5120 || input.iter().any(|x| !x.is_finite()) { return Err("norm row dimensions or nonfinite input"); }
    let mut lanes = [0.0; 32];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        let mut partials = [0.0; 16];
        for (group, partial) in partials.iter_mut().enumerate() {
            let mut values = [0.0; 4];
            for (adjacent, value) in values.iter_mut().enumerate() {
                for i in chain_indices(width, lane, group, adjacent)? { *value += input[i]*input[i]; }
            }
            *partial = ((values[0]+values[1])+values[2])+values[3];
        }
        for half in [8,4,2,1] { for i in 0..half { partials[i] += partials[i+half]; } }
        *sum = partials[0];
    }
    let mut warp = [[0.0; 32]; 5];
    let mut current = lanes;
    for (stage, distance) in [16,8,4,2,1].into_iter().enumerate() {
        let previous = current;
        for lane in 0..32 { current[lane] = previous[lane]+previous[lane ^ distance]; }
        warp[stage] = current;
    }
    if !current[0].is_finite() { return Err("norm square overflow"); }
    Ok(SquareTrace { lanes, warp, sum: current[0] })
}

/// Single BF16 residual first; weight is zero-centered BF16, output rounds once.
/// The same row function handles input, post-attention, next-layer and final norms.
pub fn normalize(input: &[f32], residual: Option<&[f32]>, weights: &[u16], epsilon: f32) -> Result<Normalized> {
    let width = weights.len();
    if width == 0 || width > 5120 || input.is_empty() || input.len() % width != 0
        || residual.is_some_and(|r| r.len() != input.len() || r.iter().any(|x| !x.is_finite()))
        || !epsilon.is_finite() || epsilon <= 0.0 || input.iter().any(|x| !x.is_finite())
        || weights.iter().any(|&w| !widen_bf16(w).is_finite()) { return Err("norm operands or epsilon"); }
    let hidden: Vec<_> = input.iter().enumerate().map(|(i,&x)| residual.map_or(x, |r| round_bf16(x+r[i]))).collect();
    let mut output = Vec::with_capacity(input.len());
    let mut traces = Vec::with_capacity(input.len()/width);
    for row in hidden.chunks_exact(width) {
        let trace = square_trace(row)?;
        let scale = 1.0/(trace.sum*(1.0/width as f32)+epsilon).sqrt();
        for (&x,&w) in row.iter().zip(weights) {
            let value = round_bf16(x*scale*(widen_bf16(w)+1.0));
            if !value.is_finite() { return Err("norm output overflow"); }
            output.push(value);
        }
        traces.push(trace);
    }
    Ok(Normalized { hidden, output, traces })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_rows_and_boundaries() {
        for h in [2048,3072] { let x: Vec<_> = (0..h).map(|i| if i%7==0 { 128.0 } else { (i%31) as f32/32.0 }).collect(); let w=vec![0;h]; let solo=normalize(&x,None,&w,1e-6).unwrap(); assert_ne!(x.iter().fold(0.0f32,|s,v|s+v*v).to_bits(),solo.traces[0].sum.to_bits()); let mut batch=vec![0.0;h]; batch.extend_from_slice(&x); batch.extend(vec![32.0;h]); let pooled=normalize(&batch,None,&w,1e-6).unwrap(); assert_eq!(pooled.traces[1],solo.traces[0]); assert_eq!(&pooled.output[h..2*h],solo.output); let mut seen=vec![0;h]; for l in 0..32 { for g in 0..16 { for j in 0..4 { for i in chain_indices(h,l,g,j).unwrap() { seen[i]+=1; } } } } assert!(seen.iter().all(|&n|n==1)); }
        let r=normalize(&[16777216.0,1.0,-16777216.0,2.0],Some(&[-16777216.0,0.00390625,16777216.0,-1.0]),&[0;4],1e-6).unwrap(); assert_eq!(r.hidden,[0.0,1.0,0.0,1.0]); assert_eq!(r.traces[0].sum,2.0); assert_eq!(r.output[1].to_bits(),round_bf16(1.0/(0.5f32+1e-6).sqrt()).to_bits());
        assert!(normalize(&[f32::NAN],None,&[0],1e-6).is_err()); assert!(normalize(&[1.0],Some(&[]),&[0],1e-6).is_err()); assert!(square_trace(&[f32::MAX]).is_err()); assert!(normalize(&[1.0],None,&[0],0.0).is_err());
    }
}
