//! Shared conservative envelope for the optional exact target-head screen.
//!
//! Preconditions: K<=2560, BF16 weights and two BF16 input planes, FP32
//! accumulations with the canonical four fixed K ranges. Each range has <=640
//! terms per plane. Inputs to this function are upward-rounded L2 norms.
//! No partial sum can overflow when S=(||W||+||Q||)*(||hi||+||lo||)
//! is <= MAX_MAGNITUDE: Cauchy bounds every product/partial by S, leaving 16x
//! headroom. Otherwise return infinity, retaining every uncertain row.
//!
//! RELATIVE exceeds gamma_4096, covering both heads' range accumulations and
//! folds. ABSOLUTE covers <=65536 FP32 product/add underflow losses, each<=2^-126.
//! Operand FTZ also matters: a tiny BF16 operand times a huge operand need not
//! have a tiny product. By ||v||1<=sqrt(2560)||v||2, the 256*2^-126 term bounds
//! both heads' possible BF16 operand flushes, including the two input planes.
//! Nonfinite/invalid norms or overflow never justify pruning.
pub const RELATIVE: f32 = 0.00025;
pub const ABSOLUTE: f32 = 65536.0 * f32::MIN_POSITIVE;
pub const OPERAND_FTZ: f32 = 256.0 * f32::MIN_POSITIVE;
pub const MAX_MAGNITUDE: f32 = f32::MAX / 16.0;

#[inline(always)]
pub fn envelope(error: f32, weights: f32, effective: f32, planes: f32,
    mul_up: impl Fn(f32, f32) -> f32, add_up: impl Fn(f32, f32) -> f32) -> f32 {
    if [error,weights,effective,planes].iter().any(|x| !x.is_finite() || *x<0.0) {
        return f32::INFINITY;
    }
    let magnitude=mul_up(weights,planes);
    if !magnitude.is_finite() || magnitude>MAX_MAGNITUDE { return f32::INFINITY; }
    let operand_loss=mul_up(OPERAND_FTZ,add_up(weights,planes));
    let quantization=mul_up(error,effective);
    let rounding=add_up(mul_up(RELATIVE,magnitude),add_up(ABSOLUTE,operand_loss));
    add_up(quantization,rounding)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn up(x:f64)->f32 { let v=x as f32; if (v as f64)<x {v.next_up()}else{v} }
    fn norm(v:&[f32])->f32 {up(v.iter().fold(0.0f64,|sum,&x|(sum+(x as f64)*(x as f64)).next_up()).sqrt().next_up())}
    fn ftz(x:f32,on:bool)->f32 {if on && x.abs()<f32::MIN_POSITIVE {0.0f32.copysign(x)}else{x}}
    fn dot(w:&[f32],h:&[f32],l:&[f32],flush:bool)->f32 {
        let mut parts=[0.0f32;4];
        for (part,out) in parts.iter_mut().enumerate() {
            let(mut hi,mut lo)=(0.0f32,0.0f32);
            for i in w.len()*part/4..w.len()*(part+1)/4 {
                let a=ftz(w[i],flush);
                hi=ftz(ftz(a*ftz(h[i],flush),flush)+hi,flush);
                lo=ftz(ftz(a*ftz(l[i],flush),flush)+lo,flush);
            }
            *out=ftz(hi+lo,flush);
        }
        ftz(ftz(parts[0]+parts[1],flush)+ftz(parts[2]+parts[3],flush),flush)
    }
    fn winner(rows:impl Iterator<Item=(usize,f32)>)->Result<usize,()> {
        let mut best=(usize::MAX,f32::NEG_INFINITY);
        for (id,v) in rows {if !v.is_finite(){return Err(());} if v>best.1 || v==best.1 && id<best.0 {best=(id,v);}}
        if best.0==usize::MAX {Err(())}else{Ok(best.0)}
    }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self)->u32 {self.0^=self.0<<13;self.0^=self.0>>7;self.0^=self.0<<17;(self.0>>32) as u32}
        fn bf16(&mut self,exponent:i32)->f32 {
            let bits=self.next();let exponent=(exponent+127).clamp(0,254) as u32;
            f32::from_bits((bits&1)<<31 | exponent<<23 | ((bits>>1)&127)<<16)
        }
    }
    #[test]
    fn ten_thousand_adversarial_rows_preserve_canonical_argmax() {
        let mut rng=Rng(0xc411_2026_1022_0001);let(mut pruned,mut uncertain,mut invalid)=(0,0,0);
        for case in 0..10000 {
            let n=if case%100==0 {2560}else{128};
            let (we,xe)=match case%7 {0=>(0,0),1=>(-126,120),2=>(120,-126),3=>(100,100),4=>(-126,-126),_=>((rng.next()%240) as i32-125,(rng.next()%240) as i32-125)};
            let h:Vec<_>=(0..n).map(|_|rng.bf16(xe)).collect();let l:Vec<_>=h.iter().map(|&x|x*0.00390625).collect();
            let planes=up(norm(&h) as f64+norm(&l) as f64);
            let effective=planes;
            let first:Vec<_>=(0..n).map(|_|rng.bf16(we)).collect();
            let mut canonical=Vec::new();let mut screen=Vec::new();let mut errors=Vec::new();
            for id in 0..4 {
                let mut w=if id<2 {first.clone()} else {(0..n).map(|_|rng.bf16(we)).collect()};
                // Exact ties and one-BF16-ulp perturbations exercise lower-ID tie policy.
                if id==1 && case%2==0 {w[0]=f32::from_bits(w[0].to_bits().wrapping_add(1<<16));}
                if case%257==0 && id==3 {w[0]=f32::NAN;}
                if case%263==0 && id==2 {w[0]=f32::INFINITY;}
                let q:Vec<_>=w.iter().map(|&v|f32::from_bits(v.to_bits()&!0xf_0000)).collect();
                let e=up(w.iter().zip(&q).fold(0.0f64,|sum,(&a,&b)|{let d=(a as f64-b as f64).abs().next_up();(sum+d*d).next_up()}).sqrt().next_up());
                let weights=up(norm(&w) as f64+norm(&q) as f64);
                let error=envelope(e,weights,effective,planes,|a,b|up(a as f64*b as f64),|a,b|up(a as f64+b as f64));
                let a=dot(&w,&h,&l,case%2==0);let b=dot(&q,&h,&l,case%3==0);
                if error.is_finite() {assert!(a.is_finite() && b.is_finite(),"finite precondition case{case}");assert!((a as f64-b as f64).abs()<=error as f64,"envelope case{case}");}
                else {uncertain+=1;}
                canonical.push(a);screen.push(b);errors.push(error);
            }
            let lower=(0..4).filter(|&i|screen[i].is_finite()).map(|i|screen[i] as f64-errors[i] as f64).fold(f64::NEG_INFINITY,f64::max);
            let keep:Vec<_>=(0..4).filter(|&i|!screen[i].is_finite() || screen[i] as f64+errors[i] as f64>=lower).collect();
            pruned+=4-keep.len();let a=winner(canonical.iter().copied().enumerate());let b=winner(keep.into_iter().map(|i|(i,canonical[i])));
            if a.is_err(){invalid+=1;}assert_eq!(a,b,"bounded argmax case{case}");
        }
        assert!(pruned>0 && uncertain>0 && invalid>0);
        println!("10000 cases PASS;pruned{pruned};uncertain rows{uncertain};nonfinite cases{invalid}");
    }
    #[test]
    fn unsafe_bounds_force_canonical_retention() {
        for bad in [f32::NAN,f32::INFINITY,-1.0] {
            assert!(envelope(bad,1.0,1.0,1.0,|a,b|a*b,|a,b|a+b).is_infinite());
        }
        assert!(envelope(0.0,f32::MAX,1.0,1.0,|a,b|a*b,|a,b|a+b).is_infinite());
        assert!(envelope(0.0,1.0,0.0,0.0,|a,b|a*b,|a,b|a+b)>=ABSOLUTE);
    }
}
