//! Calibrated NVFP4 and weight-only FP8 operations. Host and device share math.
use crate::next_math::{bf16,exp};
#[inline(always)]
pub fn round(x:f32)->f32 {f32::from_bits(bf16(x)<<16)}
#[inline(always)]
pub fn e2m1(n:u8)->f32 {
    let magnitude=u32::from(n&7);
    let bits=if magnitude<2 {magnitude*0x3f00_0000} else {((magnitude/2+126)<<23)|((magnitude&1)<<22)};
    f32::from_bits(bits|(u32::from(n&8)<<28))
}
#[inline(always)]
pub fn e4m3(b:u8)->f32 {
    let e=(b>>3)&15;let f=b&7;
    let v=if e==0 {f as f32/512.0} else if b&127==127 {f32::NAN}
        else {f32::from_bits(((u32::from(e)+120)<<23)|(u32::from(f)<<20))};
    if b&128==0 {v} else {-v}
}
/// Nearest-even finite encoding. Distances are exact for adjacent FP formats.
pub fn nearest_e4m3(x:f32)->u8 {
    let sign=if x.is_sign_negative() {128} else {0};let x=x.abs();let (mut lo,mut hi)=(0u8,126u8);
    while lo<hi {let m=lo+(hi-lo)/2;if e4m3(m)<x {lo=m+1;} else {hi=m;}}
    if lo>0 {let a=x-e4m3(lo-1);let b=e4m3(lo)-x;if a<b || a==b && (lo-1)%2==0 {lo-=1;}}
    sign|lo
}
pub fn nearest_e2m1(x:f32)->u8 {
    let sign=if x.is_sign_negative() {8} else {0};let x=x.abs();if x>=6.0 {return sign|7;}let mut best=0;
    for n in 1..8 {let a=(x-e2m1(best)).abs();let b=(x-e2m1(n)).abs();if b<a || b==a && n%2==0 {best=n;}}
    sign|best
}
#[repr(u32)]
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Op {Quantize,NvProject,Fp8Project,SwiGlu,Fold,QuantizeRows,NvRows,Fp8Rows}
#[repr(C)]
#[derive(Debug,Clone,Copy)]
pub struct Shape {pub op:Op,pub rows:u32,pub cols:u32,pub block_rows:u32,pub block_cols:u32,pub tokens:u32,
    pub global:f32,pub activation:f32,pub bound:f32}
impl Shape {
    pub fn lengths(self)->Option<([usize;4],usize,usize,usize,usize)> {
        let (r,k,br,bc)=(self.rows as usize,self.cols as usize,self.block_rows as usize,self.block_cols as usize);
        if r==0 || k==0 || !self.global.is_finite() || self.global<=0.0 || !self.activation.is_finite()
            || self.activation<=0.0 || !self.bound.is_finite() {return None;}
        let rk=r.checked_mul(k)?;
        let m=self.tokens as usize;if m==0 || m>4 {return None;}
        let result=match self.op {
            Op::Quantize if k%16==0=>([k,0,0,0],0,0,k/16,17),
            Op::NvProject if k%16==0=>([k/16*17,0,0,0],rk/2,rk/16,r,1),
            Op::Fp8Project if br>0 && bc>0=>([k,0,0,0],rk,r.div_ceil(br).checked_mul(k.div_ceil(bc))?.checked_mul(4)?,r,1),
            Op::SwiGlu if self.bound>0.0=>([r,r,0,0],0,0,r,1),
            Op::Fold if k<=1024=>([rk,k,k,r],0,0,r,1),
            Op::QuantizeRows if k%16==0=>([m.checked_mul(k)?,0,0,0],0,0,m.checked_mul(k/16)?,17),
            Op::NvRows if k%16==0=>([m.checked_mul(k/16)?.checked_mul(17)?,0,0,0],rk/2,rk/16,r.checked_mul(m)?,1),
            Op::Fp8Rows if br>0 && bc>0=>([m.checked_mul(k)?,0,0,0],rk,r.div_ceil(br).checked_mul(k.div_ceil(bc))?.checked_mul(4)?,r.checked_mul(m)?,1),
            _=>return None,

        };
        if result.0.into_iter().chain([result.1,result.2,result.3.checked_mul(result.4)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
        Some(result)
    }
}
/// A complete warp can schedule fewer independent long NVFP4 rows per block.
/// The inactive upper lanes never obtain an output slice.
pub fn output_index(op:Op,thread:usize)->Option<usize> {
    if op==Op::NvProject {let lane=thread%32;if lane>=8 {None} else {Some(thread/32*8+lane)}} else {Some(thread)}
}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Arguments {pub input:[u64;4],pub weight:u64,pub scales:u64,pub output:u64,pub shape:Shape}
#[inline(always)]
fn nv_row(s:Shape,index:usize,x:&[f32],w:&[u8],scales:&[u8])->f32 {
    let k=s.cols as usize;
    let global=s.global*s.activation;let mut sum=0.0f32;
    for block in 0..k/16 {
        let a=&x[block*17..block*17+17];let fold=(e4m3(scales[index*(k/16)+block])*a[16])*global;
        for pair in 0..8 {
            let packed=w[index*(k/2)+block*8+pair];
            sum+=(e2m1(packed&15)*a[2*pair])*fold;
            sum+=(e2m1(packed>>4)*a[2*pair+1])*fold;
        }
    }round(sum)
}
#[inline(always)]
fn fp8_row(s:Shape,index:usize,x:&[f32],w:&[u8],scales:&[u8])->f32 {
    let k=s.cols as usize;
    let mut sum=0.0f32;let br=s.block_rows as usize;let bc=s.block_cols as usize;let blocks=k.div_ceil(bc);
    for block in 0..blocks {let at=((index/br)*blocks+block)*4;
        let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
        for j in block*bc..((block+1)*bc).min(k) {sum+=(e4m3(w[index*k+j])*scale)*round(x[j]);}
    }round(sum)
}
/// Quantized blocks contain sixteen exact decoded E2M1 values and an E4M3 scale.
/// Fold follows ascending expert IDs. It rounds each contribution and addition.
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[u8],scales:&[u8],out:&mut[f32]) {
    let k=s.cols as usize;
    match s.op {
        Op::Quantize|Op::QuantizeRows=>{
            let row=&x[0][index*16..index*16+16];let mut maximum=0.0f32;
            for &v in row {maximum=maximum.max(round(v).abs());}
            let scale=nearest_e4m3((maximum/s.activation)/6.0);let block=e4m3(scale);
            for pair in 0..8 {
                let q=|v|if block==0.0 {if round(v).is_sign_negative() {8} else {0}} else {nearest_e2m1((round(v)/s.activation)/block)};
                out[2*pair]=e2m1(q(row[2*pair]));out[2*pair+1]=e2m1(q(row[2*pair+1]));
            }out[16]=block;
        },
        Op::NvProject=>{
            let global=s.global*s.activation;let mut sum=0.0f32;
            for block in 0..k/16 {
                let a=&x[0][block*17..block*17+17];let fold=(e4m3(scales[index*(k/16)+block])*a[16])*global;
                for pair in 0..8 {
                    let packed=w[index*(k/2)+block*8+pair];
                    sum+=(e2m1(packed&15)*a[2*pair])*fold;
                    sum+=(e2m1(packed>>4)*a[2*pair+1])*fold;
                }
            }out[0]=round(sum);
        },
        Op::Fp8Project=>{
            let mut sum=0.0f32;let br=s.block_rows as usize;let bc=s.block_cols as usize;let blocks=k.div_ceil(bc);
            for block in 0..blocks {let at=((index/br)*blocks+block)*4;
                let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
                for j in block*bc..((block+1)*bc).min(k) {sum+=(e4m3(w[index*k+j])*scale)*round(x[0][j]);}
            }out[0]=round(sum);
        },
        Op::NvRows=>{
            // Adjacent token lanes read the same immutable weight row. Each
            // owns one accumulator and keeps the scalar feature fold.
            let m=s.tokens as usize;let token=index%m;let row=index/m;let stride=k/16*17;
            out[0]=nv_row(s,row,&x[0][token*stride..(token+1)*stride],w,scales);
        },
        Op::Fp8Rows=>{
            let m=s.tokens as usize;let token=index%m;let row=index/m;
            out[0]=fp8_row(s,row,&x[0][token*k..(token+1)*k],w,scales);
        },
        Op::SwiGlu=>{let g=x[0][index].min(s.bound);let u=x[1][index].clamp(-s.bound,s.bound);
            out[0]=round(round(g*(1.0/(1.0+exp(-g))))*u);},
        Op::Fold=>{
            let mut previous=-1.0f32;let mut sum=0.0;
            for _ in 0..k {let mut chosen=0;let mut id=f32::INFINITY;
                for slot in 0..k {if x[1][slot]>previous && x[1][slot]<id {chosen=slot;id=x[1][slot];}}
                sum=round(sum+round(x[0][chosen*s.rows as usize+index]*x[2][chosen]));previous=id;
            }out[0]=round(sum+x[3][index]);
        },
    }
}
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(s:Shape,x:[&[f32];4],w:&[u8],scales:&[u8])->Result<Vec<f32>,String> {
    let (lengths,wb,sb,units,words)=s.lengths().ok_or("invalid expert operation shape")?;
    if x.iter().zip(lengths).any(|(a,n)|a.len()!=n) || w.len()!=wb || scales.len()!=sb
        || x.iter().flat_map(|v|v.iter()).any(|v|!v.is_finite()) {return Err("expert operation operand extent or value differs".into());}
    if matches!(s.op,Op::Quantize|Op::QuantizeRows) || matches!(s.op,Op::Fp8Project|Op::Fp8Rows) {
        if x[0].iter().any(|&v|!round(v).is_finite()) {return Err("expert source overflows BF16".into());}
    }
    if matches!(s.op,Op::Quantize|Op::QuantizeRows) && x[0].chunks_exact(16).any(|b|
        !(b.iter().fold(0.0f32,|a,&v|a.max(round(v).abs()))/s.activation/6.0).is_finite()) {
        return Err("expert block scale exceeds the calibrated range".into());
    }
    if matches!(s.op,Op::NvProject|Op::NvRows) && !(s.global*s.activation).is_finite() {return Err("expert combined global scale is not finite".into());}
    if matches!(s.op,Op::NvProject|Op::NvRows) && (scales.iter().any(|&b|b>=127) || x[0].chunks_exact(17).any(|b|
        b[..16].iter().any(|&v|e2m1(nearest_e2m1(v)).to_bits()!=v.to_bits())
        || b[16].is_sign_negative() || e4m3(nearest_e4m3(b[16])).to_bits()!=b[16].to_bits())) {
        return Err("invalid NVFP4 decoded activation or weight scale".into());
    }
    if matches!(s.op,Op::Fp8Project|Op::Fp8Rows) && (w.iter().any(|b|b&127==127) || scales.chunks_exact(4).any(|b|
        {let v=f32::from_le_bytes(b.try_into().unwrap());!v.is_finite() || v<=0.0})) {return Err("invalid FP8 block weight or scale".into());}
    if s.op==Op::Fold && (x[1].iter().any(|&v|v<0.0 || v>=16_777_216.0 || v.fract()!=0.0)
        || (0..x[1].len()).any(|i|x[1][i+1..].contains(&x[1][i]))) {return Err("invalid or duplicate expert fold IDs".into());}
    let mut out=vec![0.0;units*words];for (i,o) in out.chunks_exact_mut(words).enumerate() {unit(s,i,x,w,scales,o);}
    if out.iter().any(|v|!v.is_finite()) {return Err("expert operation output is not finite".into());}Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn s(op:Op)->Shape {Shape {op,tokens:1,rows:2,cols:16,block_rows:1,block_cols:16,global:1.0,activation:1.0,bound:10.0}}
    #[test]
    fn e2m1_bits_match_all_codes_and_signs() {
        let values=[0.0f32,0.5,1.0,1.5,2.0,3.0,4.0,6.0];
        for n in 0..=255u8 {let v=values[(n&7) as usize];let old=if n&8==0 {v} else {-v};
            assert_eq!(e2m1(n).to_bits(),old.to_bits(),"nibble {n}");
        }
    }
    #[test]
    fn output_lanes_cover_each_row_once_at_warp_and_tail_boundaries() {
        for op in [Op::Quantize,Op::NvProject,Op::NvRows,Op::Fp8Project,Op::SwiGlu,Op::Fold] {
            for units in [1usize,7,8,9,15,16,17,31,32,33,1024,4097] {
                let active=if op==Op::NvProject {8} else {32};
                let rows=(0..units.div_ceil(active)*32).filter_map(|i|output_index(op,i)).filter(|&i|i<units).collect::<Vec<_>>();
                assert_eq!(rows,(0..units).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn exact_encodings_ties_and_signed_zero() {
        for b in 0..=254u8 {if b&127!=127 {assert_eq!(nearest_e4m3(e4m3(b)),b);}}
        for n in 0..16 {assert_eq!(nearest_e2m1(e2m1(n)),n);}
        for (x,n) in [(0.25,0),(0.75,2),(1.25,2),(1.75,4),(2.5,4),(3.5,6),(5.0,6)] {
            assert_eq!(nearest_e2m1(x),n);assert_eq!(nearest_e2m1(-x),n|8);
        }
        assert_eq!(nearest_e4m3(1.0625),0x38);assert_eq!(nearest_e4m3(1.1875),0x3a);
    }
    #[test]
    fn calibrated_bytes_and_global_fold() {
        let mut x=[0.0;16];x[..8].copy_from_slice(&[6.0,0.25,0.75,1.25,1.75,2.5,3.5,5.0]);x[8]=-0.0;
        let q=reference(s(Op::Quantize),[&x,&[],&[],&[]],&[],&[]).unwrap();
        let expected=[6.0f32,0.0,1.0,1.0,2.0,2.0,4.0,4.0,-0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0];
        assert_eq!(q.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.map(f32::to_bits));
        let mut shape=s(Op::NvProject);shape.global=2.0;
        let v=reference(shape,[&q,&[],&[],&[]],&[0x22;16],&[0x38;2]).unwrap();assert_eq!(v,[40.0;2]);
        let mut shape=s(Op::Quantize);shape.activation=2.0;
        let scaled=reference(shape,[&x,&[],&[],&[]],&[],&[]).unwrap();assert_eq!(scaled[..16].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),q[..16].iter().map(|v|v.to_bits()).collect::<Vec<_>>());assert_eq!(scaled[16],0.5);
    }
    #[test]
    fn decoded_activation_bits_match_the_original_numeric_code_oracle() {
        for width in [16,32,128] {for activation in [0.125,1.0,4.0] {
            let mut shape=s(Op::Quantize);shape.cols=width;shape.activation=activation;
            let input=(0..width as usize).map(|i|if i%7==0 {-0.0} else {(i as i32%31-15) as f32*0.0625}).collect::<Vec<_>>();
            let actual=reference(shape,[&input,&[],&[],&[]],&[],&[]).unwrap();
            for (row,lanes) in input.chunks_exact(16).zip(actual.chunks_exact(17)) {
                let maximum=row.iter().fold(0.0f32,|a,&v|a.max(round(v).abs()));
                let scale=nearest_e4m3((maximum/activation)/6.0);let block=e4m3(scale);
                let old=|v:f32|if block==0.0 {if round(v).is_sign_negative() {8} else {0}} else {nearest_e2m1((round(v)/activation)/block)};
                for j in 0..16 {assert_eq!(lanes[j].to_bits(),e2m1(old(row[j])).to_bits());}
                assert_eq!(lanes[16].to_bits(),e4m3(scale).to_bits());
            }
        }}
    }
    #[test]
    fn decoded_activations_keep_the_original_packed_ascending_feature_fold() {
        for rows in [1,17,65] {for cols in [16,32,128] {
            let mut shape=s(Op::NvProject);shape.rows=rows;shape.cols=cols;shape.global=0.75;shape.activation=0.125;
            let k=cols as usize;let packed=(0..k/16*9).map(|i|if i%9==8 {0x20+(i%31) as u8} else {((i*37+9)%256) as u8}).collect::<Vec<_>>();
            let x=packed.chunks_exact(9).flat_map(|b|(0..16).map(|j|e2m1(if j%2==0 {b[j/2]&15} else {b[j/2]>>4})).chain([e4m3(b[8])])).collect::<Vec<_>>();
            let w=(0..rows as usize*k/2).map(|i|((i*53+17)%256) as u8).collect::<Vec<_>>();
            let scales=(0..rows as usize*k/16).map(|i|0x10+(i%64) as u8).collect::<Vec<_>>();
            let actual=reference(shape,[&x,&[],&[],&[]],&w,&scales).unwrap();
            for (row,&value) in actual.iter().enumerate() {let global=shape.global*shape.activation;let mut sum=0.0f32;
                for block in 0..k/16 {let a=&packed[block*9..block*9+9];let fold=(e4m3(scales[row*(k/16)+block])*e4m3(a[8]))*global;
                    for j in 0..16 {let packed=w[row*(k/2)+block*8+j/2];let input=a[j/2];
                        let weight=e2m1(if j%2==0 {packed&15} else {packed>>4});let input=e2m1(if j%2==0 {input&15} else {input>>4});sum+=(weight*input)*fold;
                    }
                }assert_eq!(value.to_bits(),round(sum).to_bits(),"rows={rows} cols={cols} row={row}");
            }
        }}
    }
    #[test]
    fn decoded_planes_reject_old_extents_and_non_format_values() {
        let shape=s(Op::NvProject);let mut x=vec![0.0;17];let weight=[0x22;16];let scales=[0x38;2];
        assert!(reference(shape,[&x[..9],&[],&[],&[]],&weight,&scales).is_err());
        x[0]=0.25;assert!(reference(shape,[&x,&[],&[],&[]],&weight,&scales).is_err());
        x[0]=-0.0;x[16]=-0.0;assert!(reference(shape,[&x,&[],&[],&[]],&weight,&scales).is_err());
        x[16]=0.0;assert!(reference(shape,[&x,&[],&[],&[]],&weight,&scales).is_ok());
    }
    #[test]
    fn hoisted_fp8_scales_keep_each_feature_add_at_partial_blocks() {
        for rows in [1,3,17] {for cols in [1,17,129,515] {for br in [1,3,64] {for bc in [1,7,128,1024] {
            let mut shape=s(Op::Fp8Project);shape.rows=rows;shape.cols=cols;shape.block_rows=br;shape.block_cols=bc;
            let (r,k)=(rows as usize,cols as usize);
            let x=(0..k).map(|i|if i%11==0 {-0.0} else {(i as i32%23-11) as f32*0.037}).collect::<Vec<_>>();
            let w=(0..r*k).map(|i|((i*53+17)%127) as u8|(((i/7)%2) as u8*128)).collect::<Vec<_>>();
            let grid=k.div_ceil(bc as usize);let scales=(0..r.div_ceil(br as usize)*grid)
                .flat_map(|i|(0.125+(i%17) as f32*0.03125).to_le_bytes()).collect::<Vec<_>>();
            let got=reference(shape,[&x,&[],&[],&[]],&w,&scales).unwrap();
            for (row,&value) in got.iter().enumerate() {let mut sum=0.0f32;
                for j in 0..k {let at=((row/br as usize)*grid+j/bc as usize)*4;
                    let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
                    sum+=(e4m3(w[row*k+j])*scale)*round(x[j]);
                }assert_eq!(value.to_bits(),round(sum).to_bits(),"row={row} cols={cols} br={br} bc={bc}");
            }
        }}}}
    }
    #[test]
    fn fp8_partial_blocks_and_fixed_fold_order() {
        let mut shape=s(Op::Fp8Project);shape.rows=3;shape.cols=3;shape.block_rows=2;shape.block_cols=2;
        let scales=[1.0f32,2.0,3.0,4.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(reference(shape,[&[1.0;3],&[],&[],&[]],&[0x38;9],&scales).unwrap(),[4.0,4.0,10.0]);
        let mut shape=s(Op::Fold);shape.rows=1;shape.cols=3;
        assert_eq!(reference(shape,[&[256.0,-256.0,1.0],&[0.0,2.0,1.0],&[1.0;3],&[0.0]],&[],&[]).unwrap(),[0.0]);
        assert!(reference(shape,[&[0.0;3],&[1.0;3],&[1.0;3],&[0.0]],&[],&[]).is_err());
        assert!(reference(s(Op::Quantize),[&[f32::NAN;16],&[],&[],&[]],&[],&[]).is_err());
    }
    #[test]
    fn grouped_rows_keep_scalar_quantization_folds_and_scale_tails() {
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        for tokens in 1..=4 {for cols in [16,32,96] {
            let base=Shape {rows:33,cols,tokens:1,..s(Op::NvProject)};let k=cols as usize;let r=base.rows as usize;
            let x=(0..tokens as usize*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();
            let q=reference(Shape {op:Op::QuantizeRows,tokens,..base},[&x,&[],&[],&[]],&[],&[]).unwrap();
            let w=(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>();let scales=(0..r*k/16).map(|i|0x28+(i%24) as u8).collect::<Vec<_>>();
            let got=reference(Shape {op:Op::NvRows,tokens,..base},[&q,&[],&[],&[]],&w,&scales).unwrap();
            for token in 0..tokens as usize {
                let scalar_q=reference(Shape {op:Op::Quantize,..base},[&x[token*k..(token+1)*k],&[],&[],&[]],&[],&[]).unwrap();
                assert_eq!(bits(&q[token*k/16*17..(token+1)*k/16*17]),bits(&scalar_q));
                let expected=reference(base,[&scalar_q,&[],&[],&[]],&w,&scales).unwrap();
                for row in 0..r {assert_eq!(got[row*tokens as usize+token].to_bits(),expected[row].to_bits());}
            }
        }}
        for tokens in 1..=4 {let base=Shape {op:Op::Fp8Project,rows:5,cols:33,block_rows:2,block_cols:16,tokens:1,..s(Op::Fp8Project)};
            let k=base.cols as usize;let r=base.rows as usize;let x=(0..tokens as usize*k).map(|i|if i%9==0 {-0.0} else {(i as i32%23-11) as f32/16.0}).collect::<Vec<_>>();
            let w=(0..r*k).map(|i|(i%120+1) as u8).collect::<Vec<_>>();let scales=(0..9).flat_map(|i|(0.25+(i as f32)/16.0).to_le_bytes()).collect::<Vec<_>>();
            let got=reference(Shape {op:Op::Fp8Rows,tokens,..base},[&x,&[],&[],&[]],&w,&scales).unwrap();
            for token in 0..tokens as usize {let expected=reference(base,[&x[token*k..(token+1)*k],&[],&[],&[]],&w,&scales).unwrap();for row in 0..r {assert_eq!(got[row*tokens as usize+token].to_bits(),expected[row].to_bits());}}
        }
        assert!(Shape {tokens:5,..s(Op::NvRows)}.lengths().is_none());
    }

}
