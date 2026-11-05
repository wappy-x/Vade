//! Fixed-order pooled latent attention and corrected expert operations.
//! The causal kernel owns each disjoint output unit through stream completion.
use crate::causal_ops::{Op,Shape};
use crate::next_math::{exp,bf16};
fn round(x:f32)->f32 {f32::from_bits(bf16(x)<<16)}
fn sigmoid(x:f32)->f32 {1.0/(1.0+exp(-x))}
fn root(x:f32)->f32 {
    #[cfg(target_arch="nvptx64")] {crate::qsa_common::root(x)}
    #[cfg(not(target_arch="nvptx64"))] {x.sqrt()}
}
pub fn lengths(s:Shape)->Option<([usize;4],usize,usize,usize)> {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    // All products are checked again by the shared ABI length bound.
    match s.op {
        Op::KeyNorm if d>0 && s.epsilon>0.0=>Some(([d,0,0,0],2*d,1,d)),
        Op::Pool if r>0 && d>0 && t>0=>Some(([(r-1).checked_mul(d)?,(r-1).checked_mul(d)?,d,d],t.checked_mul(d)?,1,(2*r).checked_mul(d)?.checked_add((r/t).checked_mul(d)?)?)),
        Op::PoolChannels if r>0 && d>0 && t>0=>Some(([(r-1).checked_mul(d)?,(r-1).checked_mul(d)?,d,d],t.checked_mul(d)?,d,(2*r).checked_add(r/t)?)),
        Op::PoolLayout if r>0 && d>0 && t>0=>{let words=(2*r).checked_mul(d)?.checked_add((r/t).checked_mul(d)?)?;Some(([words,0,0,0],0,words,1))},
        Op::PoolScores if h>0 && d>0=>Some(([h.checked_mul(d)?,r.checked_mul(d)?,h,0],0,r.max(1),1)),
        Op::Select if d>0 && t>0 && t%d==0 && k<=1=>{let slots=t.checked_add(if k!=0 {d-1} else {0})?;Some(([r/d,0,0,0],0,1,slots+1))},
        Op::Gather if r>0 && d>0 && k>0=>Some(([r.checked_mul(d)?,k,1,0],0,k,d)),
        Op::Scores if h>0 && d>0 && r>0=>Some(([h.checked_mul(d)?,r.checked_mul(h)?.checked_mul(d)?,1,0],0,h,r)),
        Op::Softmax if h>0 && r>0=>Some(([h.checked_mul(r)?,1,0,0],0,h,r)),
        Op::Values if h>0 && d>0 && r>0=>Some(([h.checked_mul(r)?,r.checked_mul(h)?.checked_mul(d)?,1,0],0,h,d)),
        Op::Route if h>0 && d>0 && d<=h && r>0 && r%h==0 && r/h>=2 && k>0 && k<=d*(r/h) && t<=1 && s.bound>0.0=>Some(([r,0,0,0],r,1,2*k+2*r+2*h)),
        Op::ExpertFold if r>0 && k>0 && t<=1=>Some(([r,r,r.checked_mul(k)?,if t!=0 {k} else {0}],0,k,1)),
        _=>None,
    }
}
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    match s.op {
        Op::KeyNorm=>{
            let mut sum=0.0;for &v in x[0] {sum+=v;}let mean=sum/d as f32;
            let mut squares=0.0;for &v in x[0] {let a=v-mean;squares+=a*a;}let inv=1.0/root(squares/d as f32+s.epsilon);
            for j in 0..d {out[j]=round(((x[0][j]-mean)*inv)*w[j]+w[d+j]);}
        },
        Op::Pool=>{
            let length=r*d;let (keys,rest)=out.split_at_mut(length);let (gates,pooled)=rest.split_at_mut(length);
            keys[..(r-1)*d].copy_from_slice(x[0]);keys[(r-1)*d..].copy_from_slice(x[2]);
            gates[..(r-1)*d].copy_from_slice(x[1]);gates[(r-1)*d..].copy_from_slice(x[3]);
            for group in 0..r/t {for j in 0..d {
                let mut maximum=f32::NEG_INFINITY;for slot in 0..t {maximum=maximum.max(gates[(group*t+slot)*d+j]+w[slot*d+j]);}
                let mut sum=0.0;for slot in 0..t {sum+=exp(gates[(group*t+slot)*d+j]+w[slot*d+j]-maximum);}
                let mut value=0.0;for slot in 0..t {let p=round(exp(gates[(group*t+slot)*d+j]+w[slot*d+j]-maximum)/sum);
                    value+=round(p*keys[(group*t+slot)*d+j]);}pooled[group*d+j]=round(value);
            }}
        },
        Op::PoolChannels=>{
            let j=index;let (keys,rest)=out.split_at_mut(r);let (gates,pooled)=rest.split_at_mut(r);
            for slot in 0..r-1 {keys[slot]=x[0][slot*d+j];gates[slot]=x[1][slot*d+j];}
            keys[r-1]=x[2][j];gates[r-1]=x[3][j];
            for group in 0..r/t {
                let mut maximum=f32::NEG_INFINITY;for slot in 0..t {maximum=maximum.max(gates[group*t+slot]+w[slot*d+j]);}
                let mut sum=0.0;for slot in 0..t {sum+=exp(gates[group*t+slot]+w[slot*d+j]-maximum);}
                let mut value=0.0;for slot in 0..t {let p=round(exp(gates[group*t+slot]+w[slot*d+j]-maximum)/sum);
                    value+=round(p*keys[group*t+slot]);}pooled[group]=round(value);
            }
        },
        Op::PoolLayout=>{
            let width=2*r+r/t;let length=r*d;
            let (slot,j,plane)=if index<length {(index/d,index%d,0)} else if index<2*length {let at=index-length;(at/d,at%d,r)}
                else {let at=index-2*length;(at/d,at%d,2*r)};
            out[0]=x[0][j*width+plane+slot];
        },
        Op::PoolScores=>{
            let mut score=0.0;if index<r {for head in 0..h {let mut dot=0.0;for j in 0..d {dot+=x[0][head*d+j]*x[1][index*d+j];}
                score+=(x[2][head]*(1.0/root(h as f32)))*(dot*(1.0/root(d as f32))).max(0.0);}}
            out[0]=score;
        },
        Op::Select=>{
            let groups=r/d;let take=(t/d).min(groups);let slots=out.len()-1;
            out.fill(-1.0);
            for slot in 0..take {let mut id=usize::MAX;let mut best=f32::NEG_INFINITY;
                for group in 0..groups {if out[..slot].contains(&(group as f32)) {continue;}let score=x[0][group];
                    if score>best || score==best && group<id {best=score;id=group;}}
                out[slot]=id as f32;
            }
            // Sort selected pool IDs, then expand backward without scratch.
            for i in 1..take {let value=out[i];let mut j=i;while j>0 && out[j-1]>value {out[j]=out[j-1];j-=1;}out[j]=value;}
            for i in (0..take).rev() {let group=out[i] as usize;for slot in 0..d {out[i*d+slot]=(group*d+slot) as f32;}}
            let mut count=take*d;if k!=0 {for token in groups*d..r {out[count]=token as f32;count+=1;}}
            out[count..slots].fill(-1.0);out[slots]=count as f32;
        },
        Op::Gather=>{let count=x[2][0] as usize;
            if index<count {let row=x[1][index] as usize;out.copy_from_slice(&x[0][row*d..(row+1)*d]);} else {out.fill(0.0);}},
        Op::Scores=>{
            let count=x[2][0] as usize;out.fill(0.0);
            for slot in 0..count {let mut sum=0.0;for j in 0..d {sum+=x[0][index*d+j]*x[1][(slot*h+index)*d+j];}
                out[slot]=round(round(sum)*(1.0/root(d as f32)));}
        },
        Op::Softmax=>{
            let count=x[1][0] as usize;out.fill(0.0);if count==0 {return;}let row=&x[0][index*r..(index+1)*r];
            let mut maximum=f32::NEG_INFINITY;for &v in &row[..count] {maximum=maximum.max(v);}
            let mut denominator=0.0;for &v in &row[..count] {denominator+=exp(v-maximum);}
            for slot in 0..count {out[slot]=round(exp(row[slot]-maximum)/denominator);}
        },
        Op::Values=>{
            let count=x[2][0] as usize;for j in 0..d {let mut sum=0.0;for slot in 0..count {sum+=x[0][index*r+slot]*x[1][(slot*h+index)*d+j];}out[j]=round(sum);}
        },
        Op::Route=>{
            let (selected,scratch)=out.split_at_mut(2*k);let (scores,scratch)=scratch.split_at_mut(r);let (choice,scratch)=scratch.split_at_mut(r);
            let (group_scores,kept)=scratch.split_at_mut(h);let width=r/h;
            for expert in 0..r {scores[expert]=sigmoid(x[0][expert]);choice[expert]=scores[expert]+w[expert];}
            for group in 0..h {let (mut a,mut b)=(f32::NEG_INFINITY,f32::NEG_INFINITY);
                for &v in &choice[group*width..(group+1)*width] {if v>a {b=a;a=v;} else if v>b {b=v;}}group_scores[group]=a+b;}
            kept.fill(0.0);for _ in 0..d {let mut id=0;let mut best=f32::NEG_INFINITY;for group in 0..h {if group_scores[group]>best {best=group_scores[group];id=group;}}
                kept[id]=1.0;group_scores[id]=f32::NEG_INFINITY;}
            let mut sum=0.0;for slot in 0..k {let mut id=0;let mut best=f32::NEG_INFINITY;
                for expert in 0..r {if kept[expert/width]!=0.0 && choice[expert]>best {best=choice[expert];id=expert;}}
                selected[slot]=id as f32;selected[k+slot]=scores[id];sum+=scores[id];choice[id]=f32::NEG_INFINITY;}
            for v in &mut selected[k..] {if t!=0 {*v/=sum+1e-20;}*v*=s.bound;}
            out[2*k..].fill(0.0);
        },
        Op::ExpertFold=>{
            let mut value=0.0;let mut previous=None;
            for _ in 0..r {let mut selected=0;let mut best=f32::INFINITY;
                for slot in 0..r {if previous.is_none_or(|v|x[0][slot]>v) && x[0][slot]<best {selected=slot;best=x[0][slot];}}
                value=round(value+round(x[2][selected*k+index]*x[1][selected]));previous=Some(best);}
            out[0]=if t!=0 {round(value+x[3][index])} else {value};
        },
        _=>{},
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_ops::reference;
    fn shape(op:Op)->Shape {Shape {op,heads:2,dim:2,taps:2,rows:3,cols:1,epsilon:1e-6,bound:1.0}}
    #[test]
    fn learned_pool_retains_causal_tail_and_selects_lower_pool_ids_on_ties() {
        let s=shape(Op::Pool);let v=reference(s,[&[1.0,2.0,3.0,4.0],&[0.0;4],&[5.0,6.0],&[0.0;2]],&[0.0;4]).unwrap();
        assert_eq!(&v[..6],&[1.0,2.0,3.0,4.0,5.0,6.0]);assert_eq!(&v[12..],&[2.0,3.0]);
        let mut s=shape(Op::Select);s.rows=5;let selected=reference(s,[&[1.0,1.0],&[],&[],&[]],&[]).unwrap();
        assert_eq!(selected,[0.0,1.0,4.0,3.0]);
    }
    #[test]
    fn independent_pool_channels_restore_all_state_and_keep_slot_folds() {
        for dim in [1u32,2,17,128] {for taps in [1u32,2,4] {for rows in [1u32,3,4,5,17] {
            let s=Shape {op:Op::Pool,dim,taps,rows,..shape(Op::Pool)};let (d,r,t)=(dim as usize,rows as usize,taps as usize);
            let a=(0..(r-1)*d).map(|i|if i%11==0 {-0.0} else {(i as i32%19-9) as f32*0.031}).collect::<Vec<_>>();
            let b=(0..(r-1)*d).map(|i|(i as i32%17-8) as f32*0.047).collect::<Vec<_>>();
            let c=(0..d).map(|i|(i as i32%7-3) as f32*0.019).collect::<Vec<_>>();
            let e=(0..d).map(|i|(i as i32%13-6) as f32*0.021).collect::<Vec<_>>();
            let w=(0..t*d).map(|i|(i as i32%23-11) as f32*0.073).collect::<Vec<_>>();
            let expected=reference(s,[&a,&b,&c,&e],&w).unwrap();
            let channels=reference(Shape {op:Op::PoolChannels,..s},[&a,&b,&c,&e],&w).unwrap();
            let actual=reference(Shape {op:Op::PoolLayout,..s},[&channels,&[],&[],&[]],&[]).unwrap();
            assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"dim={d} rows={r} taps={t}");
        }}}
    }
    #[test]
    fn corrected_router_keeps_choice_bias_out_of_selected_weights() {
        let mut s=shape(Op::Route);s.rows=4;s.cols=2;s.dim=1;s.taps=1;
        let out=reference(s,[&[0.0;4],&[],&[],&[]],&[0.0;4]).unwrap();assert_eq!(&out[..4],&[0.0,1.0,0.5,0.5]);
        let out=reference(s,[&[0.0;4],&[],&[],&[]],&[0.0,0.0,2.0,0.0]).unwrap();assert_eq!(&out[..4],&[2.0,3.0,0.5,0.5]);
    }
}
