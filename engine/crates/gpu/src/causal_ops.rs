//! Fixed single-token KDA and dense operations. Shapes come from the layer recipe.
//! Host and device use these same operations and FP32 lane layouts.
use crate::next_math::{exp,bf16};
#[inline(always)]
fn round(x:f32)->f32 {f32::from_bits(bf16(x)<<16)}
#[inline(always)]
fn sigmoid(x:f32)->f32 {1.0/(1.0+exp(-x))}
#[inline(always)]
fn root(x:f32)->f32 {
    #[cfg(target_arch="nvptx64")] {crate::qsa_common::root(x)}
    #[cfg(not(target_arch="nvptx64"))] {x.sqrt()}
}
#[repr(u32)]
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Op {Conv,Prepare,Step,GatedNorm,SwiGlu,Project,KeyNorm,Pool,PoolScores,Select,Gather,Scores,Softmax,Values,Route,ExpertFold,Nvfp4,StreamMean,Mxfp8Quantize,UnpackKv,StepColumns,StepLayout,ConvChannels,ConvLayout,ProjectRows,PoolChannels,PoolLayout,Mxfp8QuantizePositiveZero}
#[repr(C)]
#[derive(Debug,Clone,Copy)]
pub struct Shape {pub op:Op,pub heads:u32,pub dim:u32,pub taps:u32,pub rows:u32,pub cols:u32,pub epsilon:f32,pub bound:f32}
impl Shape {
    pub fn lengths(self)->Option<([usize;4],usize,usize,usize)> {
        let (h,d,t,r,k)=(self.heads as usize,self.dim as usize,self.taps as usize,self.rows as usize,self.cols as usize);
        if !self.epsilon.is_finite() || !self.bound.is_finite() {return None;}
        let width=h.checked_mul(d)?;
        let (input,weight,units,words)=match self.op {
            Op::Conv if h>0 && d>0 && t>0 => ([3usize.checked_mul(width)?,3usize.checked_mul(t-1)?.checked_mul(width)?,0,0],3usize.checked_mul(width)?.checked_mul(t)?,3,width.checked_mul(t)?),
            Op::ConvChannels if h>0 && d>0 && t>0 => ([3usize.checked_mul(width)?,3usize.checked_mul(t-1)?.checked_mul(width)?,0,0],3usize.checked_mul(width)?.checked_mul(t)?,3usize.checked_mul(width)?,t),
            Op::ConvLayout if h>0 && d>0 && t>0 => ([3usize.checked_mul(width)?.checked_mul(t)?,0,0,0],0,3usize.checked_mul(width)?.checked_mul(t)?,1),
            Op::Prepare if h>0 && d>0 && d<=1024 && self.bound<0.0 => ([3usize.checked_mul(width)?,width,h,0],h.checked_add(width)?,h,4usize.checked_mul(d)?.checked_add(1)?),
            Op::Step if h>0 && d>0 && d<=1024 => ([h.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,h.checked_mul(d)?.checked_mul(d)?,0,0],0,h,d.checked_mul(d)?.checked_add(d)?),
            Op::StepColumns if h>0 && d>0 && d<=1024 => ([h.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,width.checked_mul(d)?,0,0],0,width,d.checked_add(1)?),
            Op::StepLayout if h>0 && d>0 && d<=1024 => ([width.checked_mul(d.checked_add(1)?)?,0,0,0],0,width.checked_mul(d.checked_add(1)?)?,1),
            Op::GatedNorm if h>0 && d>0 && d<=1024 && self.epsilon>0.0 => ([width,width,0,0],d,h,d),
            Op::SwiGlu if r>0 && self.bound>0.0 => ([r,r,0,0],0,r,1),
            Op::StreamMean if h>0 && d>0 => ([width,0,0,0],0,d,1),
            Op::Mxfp8Quantize|Op::Mxfp8QuantizePositiveZero if k>0 && k%32==0 && r<=2 => ([k,0,0,0],0,k,1),
            Op::UnpackKv if r>0 && h>0 && d>0 && k>0 && t<=r =>
                ([t.checked_mul(h)?.checked_mul(d.checked_add(k)?)?,0,0,0],0,r.checked_mul(h)?.checked_mul(d.checked_add(k)?)?,1),
            Op::Nvfp4 if r>0 && k>0 && k%16==0 => ([k,r.checked_mul(k/16)?,0,0],r.checked_mul(k)?,r,1),
            Op::Project if r>0 && k>0 => ([k,0,0,0],r.checked_mul(k)?,r,1),
            Op::ProjectRows if r>0 && k>0 && h>0 && h<=4 => ([h.checked_mul(k)?,0,0,0],r.checked_mul(k)?,r,h),
            _=>crate::latent_ops::lengths(self)?,
        };
        if input.into_iter().chain([weight,units.checked_mul(words)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
        Some((input,weight,units,words))
    }
}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Arguments {pub input:[u64;4],pub weight:u64,pub output:u64,pub shape:Shape}
/// Fixed 32-lane feature tree, independent of token count and topology.
fn square_sum(x:&[f32])->f32 {
    let mut lanes=[0.0f32;32];
    for (lane,sum) in lanes.iter_mut().enumerate() {let mut i=lane;while i<x.len() {*sum+=x[i]*x[i];i+=32;}}
    for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32-distance {lanes[lane]+=old[lane+distance];}}
    lanes[0]
}
/// Write one complete disjoint output unit. The caller validates all extents.
/// Convolution units are planes. Other KDA units are complete heads.
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    let (heads,d,taps)=(s.heads as usize,s.dim as usize,s.taps as usize);let width=heads*d;
    match s.op {
        Op::Conv=>{
            let plane=index;let history=(taps-1)*width;
            for ch in 0..width {
                let mut sum=0.0;
                for tap in 0..taps {let v=if tap+1==taps {x[0][plane*width+ch]} else {x[1][plane*history+tap*width+ch]};
                    sum+=v*w[(plane*width+ch)*taps+tap];}
                out[ch]=round(sum*sigmoid(sum));
                for tap in 0..taps-1 {out[width+tap*width+ch]=if tap+2==taps {x[0][plane*width+ch]} else {x[1][plane*history+(tap+1)*width+ch]};}
            }
        },
        Op::ConvChannels=>{
            let plane=index/width;let ch=index%width;let history=(taps-1)*width;let mut sum=0.0;
            for tap in 0..taps {let v=if tap+1==taps {x[0][plane*width+ch]} else {x[1][plane*history+tap*width+ch]};
                sum+=v*w[(plane*width+ch)*taps+tap];}
            out[0]=round(sum*sigmoid(sum));
            for tap in 0..taps-1 {out[tap+1]=if tap+2==taps {x[0][plane*width+ch]} else {x[1][plane*history+(tap+1)*width+ch]};}
        },
        Op::ConvLayout=>{
            let plane=index/(taps*width);let lane=index%(taps*width);let tap=lane/width;let ch=lane%width;
            out[0]=x[0][(plane*width+ch)*taps+tap];
        },
        Op::Prepare=>{
            let base=index*d;let q=&x[0][base..base+d];let k=&x[0][width+base..width+base+d];
            let qnorm=root(square_sum(q)+1e-6);let knorm=root(square_sum(k)+1e-6);let scale=1.0/root(d as f32);
            let rate=exp(w[index]);
            for j in 0..d {out[j]=(q[j]/qnorm)*scale;out[d+j]=k[j]/knorm;out[2*d+j]=x[0][2*width+base+j];
                out[3*d+j]=s.bound*sigmoid(rate*(x[1][base+j]+w[heads+base+j]));}
            out[4*d]=round(sigmoid(x[2][index]));
        },
        Op::Step=>{
            let p=&x[0][index*(4*d+1)..(index+1)*(4*d+1)];let state=&x[1][index*d*d..(index+1)*d*d];
            for key in 0..d {let decay=exp(p[3*d+key]);for value in 0..d {out[key*d+value]=state[key*d+value]*decay;}}
            for value in 0..d {
                let mut prediction=0.0;for key in 0..d {prediction+=out[key*d+value]*p[d+key];}
                let delta=(p[2*d+value]-prediction)*p[4*d];
                for key in 0..d {out[key*d+value]+=p[d+key]*delta;}
                let mut sum=0.0;for key in 0..d {sum+=out[key*d+value]*p[key];}out[d*d+value]=round(sum);
            }
        },
        Op::StepColumns=>{
            // Each value column is independent. Keep every key fold in the
            // original ascending order and round only the final output lane.
            let head=index/d;let value=index%d;let p=&x[0][head*(4*d+1)..(head+1)*(4*d+1)];
            let state=&x[1][head*d*d..(head+1)*d*d];
            for key in 0..d {let decay=exp(p[3*d+key]);out[key]=state[key*d+value]*decay;}
            let mut prediction=0.0;for key in 0..d {prediction+=out[key]*p[d+key];}
            let delta=(p[2*d+value]-prediction)*p[4*d];
            for key in 0..d {out[key]+=p[d+key]*delta;}
            let mut sum=0.0;for key in 0..d {sum+=out[key]*p[key];}out[d]=round(sum);
        },
        Op::StepLayout=>{
            // Restore the public per-head state/output layout by copying bits.
            let head=index/(d*d+d);let lane=index%(d*d+d);
            let (value,key)=if lane<d*d {(lane%d,lane/d)} else {(lane-d*d,d)};
            out[0]=x[0][(head*d+value)*(d+1)+key];
        },
        Op::GatedNorm=>{
            let base=index*d;let row=&x[0][base..base+d];let scale=1.0/root(square_sum(row)/d as f32+s.epsilon);
            for j in 0..d {out[j]=round(((row[j]*scale)*w[j])*sigmoid(x[1][base+j]));}
        },
        Op::SwiGlu=>{let gate=x[0][index].min(s.bound);let up=x[1][index].clamp(-s.bound,s.bound);
            out[0]=round(round(gate*sigmoid(gate))*up);},
        Op::StreamMean=>{let mut sum=0.0;for stream in 0..heads {sum+=x[0][stream*d+index];}out[0]=round(sum/heads as f32);},
        Op::Mxfp8Quantize=>{out[0]=crate::mxfp8_quantize::lane(x[0],index,s.rows);},
        Op::Mxfp8QuantizePositiveZero=>{out[0]=crate::mxfp8_quantize::positive_zero_lane(x[0],index,s.rows);},
        Op::UnpackKv=>{let values=s.cols as usize;let keys=s.rows as usize*heads*d;
            let (row,feature)=if index<keys {(index/d,index%d)} else {let i=index-keys;(i/values,d+i%values)};
            out[0]=if row<taps*heads {x[0][row*(d+values)+feature]} else {0.0};
        },
        Op::Nvfp4=>{let cols=s.cols as usize;let mut sum=0.0;for group in 0..cols/16 {let block=x[1][index*cols/16+group];
            for j in group*16..(group+1)*16 {sum+=(x[0][j]*w[index*cols+j])*block;}}out[0]=round(sum);},
        Op::Project=>{let cols=s.cols as usize;let mut sum=0.0;for j in 0..cols {sum+=x[0][j]*w[index*cols+j];}out[0]=sum;},
        Op::ProjectRows=>{let cols=s.cols as usize;let mut sum=[0.0f32;4];
            for j in 0..cols {let weight=w[index*cols+j];for token in 0..4 {if token<heads {sum[token]+=x[0][token*cols+j]*weight;}}}
            out[..heads].copy_from_slice(&sum[..heads]);},
        _=>crate::latent_ops::unit(s,index,x,w,out),
    }
}
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(s:Shape,x:[&[f32];4],w:&[f32])->Result<Vec<f32>,String> {
    let (lengths,weight,units,words)=s.lengths().ok_or("invalid causal operation shape")?;
    if x.iter().zip(lengths).any(|(v,n)|v.len()!=n) || w.len()!=weight
        || x.iter().flat_map(|v|v.iter()).chain(w).any(|v|!v.is_finite()) {return Err("causal operation operand extent or value differs".into());}
    let mut out=vec![0.0;units*words];for (i,row) in out.chunks_exact_mut(words).enumerate() {unit(s,i,x,w,row);}
    if out.iter().any(|v|!v.is_finite()) {return Err("causal operation output is not finite".into());}Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn shape(op:Op)->Shape {Shape {op,heads:1,dim:2,taps:2,rows:2,cols:2,epsilon:1e-5,bound:-5.0}}
    #[test]
    fn kv_layout_keeps_slot_head_order_signed_zero_and_padding() {
        let s=Shape {op:Op::UnpackKv,heads:2,dim:2,taps:2,rows:3,cols:3,epsilon:0.0,bound:0.0};
        let input=[1.0,-0.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0];
        let got=reference(s,[&input,&[],&[],&[]],&[]).unwrap();
        let expected=[1.0f32,-0.0,6.0,7.0,11.0,12.0,16.0,17.0,0.0,0.0,0.0,0.0,
            3.0,4.0,5.0,8.0,9.0,10.0,13.0,14.0,15.0,18.0,19.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0];
        assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert!(Shape {taps:4,..s}.lengths().is_none());
        assert_eq!(reference(Shape {taps:0,..s},[&[],&[],&[],&[]],&[]).unwrap(),vec![0.0;30]);
    }
    #[test]
    fn causal_convolution_keeps_oldest_first_history_and_bf16_outputs() {
        let s=shape(Op::Conv);let x=[1.0,2.0,3.0,4.0,5.0,6.0];let old=[7.0,8.0,9.0,10.0,11.0,12.0];
        let weights=[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0];
        let v=reference(s,[&x,&old,&[],&[]],&weights).unwrap();
        for plane in 0..3 {for j in 0..2 {let value=x[plane*2+j];assert_eq!(v[plane*4+j].to_bits(),round(value*sigmoid(value)).to_bits());assert_eq!(v[plane*4+2+j],value);}}
        assert!(reference(s,[&x,&old[..5],&[],&[]],&weights).is_err());
    }
    #[test]
    fn independent_convolution_channels_keep_history_and_output_bits() {
        for dim in [1,2,128] {for taps in [1,2,4] {let s=Shape {heads:3,dim,taps,..shape(Op::Conv)};let width=(s.heads*dim) as usize;let t=taps as usize;
            let x=(0..3*width).map(|i|(i as i32%19-9) as f32/8.0).collect::<Vec<_>>();
            let w=(0..3*width*t).map(|i|(i as i32%13-6) as f32/16.0).collect::<Vec<_>>();
            let mut old=(0..3*(t-1)*width).map(|i|if i%7==0 {-0.0} else {(i as i32%17-8) as f32/8.0}).collect::<Vec<_>>();
            for _ in 0..3 {
                let expected=reference(s,[&x,&old,&[],&[]],&w).unwrap();
                let channels=reference(Shape {op:Op::ConvChannels,..s},[&x,&old,&[],&[]],&w).unwrap();
                let got=reference(Shape {op:Op::ConvLayout,..s},[&channels,&[],&[],&[]],&[]).unwrap();
                assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                old=(0..3).flat_map(|p|got[p*t*width+width..(p+1)*t*width].iter().copied()).collect();
            }
        }}
    }
    #[test]
    fn recurrence_reads_immutable_state_and_rounds_only_its_output() {
        let s=shape(Op::Step);let p=[0.25,0.5,0.5,0.25,2.0,4.0,0.0,0.0,0.5];let old=[0.0;4];
        let next=reference(s,[&p,&old,&[],&[]],&[]).unwrap();assert_eq!(&next[..4],&[0.5,1.0,0.25,0.5]);
        assert_eq!(&next[4..],&[0.25,0.5]);assert_eq!(old,[0.0;4]);
        assert_eq!(next,reference(s,[&p,&old,&[],&[]],&[]).unwrap());
        assert_ne!(next,reference(s,[&p,&next[..4],&[],&[]],&[]).unwrap());
    }
    #[test]
    fn independent_recurrence_columns_keep_every_state_and_output_bit() {
        for d in [1,2,8,128] {let s=Shape {heads:3,dim:d,..shape(Op::Step)};let n=d as usize;
            let mut p=vec![0.0;3*(4*n+1)];let mut state=vec![0.0;3*n*n];
            for (i,v) in p.iter_mut().enumerate() {*v=(i as i32%17-8) as f32/32.0;}
            for (i,v) in state.iter_mut().enumerate() {*v=if i%7==0 {-0.0} else {(i as i32%29-14) as f32/64.0};}
            for _ in 0..3 {let expected=reference(s,[&p,&state,&[],&[]],&[]).unwrap();
                let columns=reference(Shape {op:Op::StepColumns,..s},[&p,&state,&[],&[]],&[]).unwrap();
                let got=reference(Shape {op:Op::StepLayout,..s},[&columns,&[],&[],&[]],&[]).unwrap();
                assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                state=(0..3).flat_map(|h|got[h*(n*n+n)..h*(n*n+n)+n*n].iter().copied()).collect();
            }
        }
        assert!(Shape {dim:1025,..shape(Op::StepColumns)}.lengths().is_none());
        assert!(Shape {heads:u32::MAX,..shape(Op::StepLayout)}.lengths().is_none());
    }
    #[test]
    fn sigmoid_beta_rounds_before_recurrence_and_invalid_shapes_refuse() {
        let s=shape(Op::Prepare);let v=reference(s,[&[1.0;6],&[0.0;2],&[0.1],&[]],&[0.0;3]).unwrap();
        assert_eq!(v[8].to_bits(),round(sigmoid(0.1)).to_bits());assert_ne!(v[8].to_bits(),sigmoid(0.1).to_bits());
        let mut bad=s;bad.dim=u32::MAX;assert!(bad.lengths().is_none());bad=s;bad.bound=0.0;assert!(bad.lengths().is_none());
        assert!(reference(s,[&[f32::NAN;6],&[0.0;2],&[0.1],&[]],&[0.0;3]).is_err());
    }
    #[test]
    fn grouped_plain_projection_keeps_each_ascending_feature_fold() {
        for tokens in 1..=4 {for cols in [1,17,128] {let s=Shape {op:Op::ProjectRows,heads:tokens,rows:7,cols,..shape(Op::Project)};
            let x=(0..tokens as usize*cols as usize).map(|i|if i%11==0 {-0.0} else {(i as i32%19-9) as f32/8.0}).collect::<Vec<_>>();
            let w=(0..7*cols as usize).map(|i|(i as i32%13-6) as f32/16.0).collect::<Vec<_>>();
            let got=reference(s,[&x,&[],&[],&[]],&w).unwrap();
            for token in 0..tokens as usize {let k=cols as usize;let expected=reference(Shape {op:Op::Project,..s},[&x[token*k..(token+1)*k],&[],&[],&[]],&w).unwrap();for row in 0..7 {assert_eq!(got[row*tokens as usize+token].to_bits(),expected[row].to_bits());}}
        }}
    }
}
