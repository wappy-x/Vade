//! Elementwise stream operations used by the one-row block replay.
use crate::{block_ops::Shape,block_stream::{CONV_ADDEND,CONV_ROUND_SUM,CONV_TOKEN_TAPS},fixed_math::rn};

/// The shared tanh polynomial with each small-input coefficient immediate. In a packed
/// tile LLVM otherwise indexes the eight coefficients through a 48-byte local frame.
#[inline(always)]
pub(crate) fn gate_tanh(x:f32)->f32 {
    let a=x.abs();
    if !(a<crate::fixed_math::TANH_SERIES_BELOW) {return crate::fixed_math::tanh(x);}
    let z=a*a;
    let mut p=-443861162.0f32/1856156927625.0;
    p=core::f32::math::mul_add(p,z,6404582.0/10854718875.0);
    p=core::f32::math::mul_add(p,z,-929569.0/638512875.0);
    p=core::f32::math::mul_add(p,z,21844.0/6081075.0);
    p=core::f32::math::mul_add(p,z,-1382.0/155925.0);
    p=core::f32::math::mul_add(p,z,62.0/2835.0);
    p=core::f32::math::mul_add(p,z,-17.0/315.0);
    p=core::f32::math::mul_add(p,z,2.0/15.0);
    p=core::f32::math::mul_add(p,z,-1.0/3.0);
    core::f32::math::mul_add(a*z,p,a).copysign(x)
}

/// Visit ring rows in rising storage order with wraparound. Lag still decreases
/// from the oldest tap to the current token, in the order of block_stream::unit.
#[inline(always)]
pub(crate) fn convolution(s:Shape,index:usize,x:[&[f32];4],w:&[f32])->f32 {
    let taps=s.taps as usize;
    if taps>=1 && taps<=CONV_GROUP {return convolution_grouped(s,index,x,w);}
    let slot=x[3][0] as usize;
    if slot>=taps || slot as f32!=x[3][0] {return f32::NAN;}
    let width=(s.heads*s.dim) as usize;
    let mut row=if slot+1==taps {0} else {slot+1};
    let mut sum=0.0;
    for lag in (0..taps).rev() {
        let mut tap=w[index*taps+lag];
        if s.cols&CONV_TOKEN_TAPS!=0 {tap+=rn(x[1][(index/s.dim as usize)*taps+lag]);}
        sum+=tap*rn(x[0][row*width+index]);
        row=if row+1==taps {0} else {row+1};
    }
    let y=rn((sum+rn(x[0][slot*width+index]))*s.bound);
    if s.cols&CONV_ADDEND==0 {y} else if s.cols&CONV_ROUND_SUM==0 {x[2][index]+y} else {rn(x[2][index]+y)}
}

/// `convolution` with no addend when the ring row of the current token (`slot`) does not hold its input yet: that
/// input is `current`, and the same launch writes it into the row. Every other row, tap and the made taps are read
/// as `convolution` reads them, and the products, sums and roundings run in the same order: the same word.
#[inline(always)]
pub(crate) fn convolution_current(s:Shape,index:usize,x:[&[f32];4],w:&[f32],slot:usize,current:f32)->f32 {
    let taps=s.taps as usize;
    let width=(s.heads*s.dim) as usize;
    let token=s.cols&CONV_TOKEN_TAPS!=0;
    let mut row=if slot+1==taps {0} else {slot+1};
    let mut sum=0.0;
    for lag in (0..taps).rev() {
        let mut tap=w[index*taps+lag];
        if token {tap+=rn(x[1][(index/s.dim as usize)*taps+lag]);}
        let value=if row==slot {current} else {x[0][row*width+index]};
        sum+=tap*rn(value);
        row=if row+1==taps {0} else {row+1};
    }
    rn((sum+rn(current))*s.bound)
}

/// Taps of the convolution whose reads go out together (`convolution_grouped`); the model's short convolution has 4.
const CONV_GROUP:usize=4;
/// Entry `index` of four values: a select, so a runtime index needs no local frame.
#[inline(always)]
fn pick(v:&[f32;CONV_GROUP],i:usize)->f32 {match i {0=>v[0],1=>v[1],2=>v[2],_=>v[3]}}
/// `convolution` for at most `CONV_GROUP` taps. Every ring row, tap weight, token tap and the addend are read before the
/// slot word is known (all lie inside the checked operands: `convolution` reads every ring row too), so the reads go out
/// together instead of one after the other. Then the same products, sums and roundings in the same order: the same word.
#[inline(always)]
fn convolution_grouped(s:Shape,index:usize,x:[&[f32];4],w:&[f32])->f32 {
    let taps=s.taps as usize;
    let width=(s.heads*s.dim) as usize;
    let token=s.cols&CONV_TOKEN_TAPS!=0;
    let (mut ring,mut tap)=([0.0f32;CONV_GROUP],[0.0f32;CONV_GROUP]);
    let mut k=0;
    while k<CONV_GROUP {
        if k<taps {
            ring[k]=x[0][k*width+index];
            tap[k]=w[index*taps+k];
            if token {tap[k]+=rn(x[1][(index/s.dim as usize)*taps+k]);}
        }
        k+=1;
    }
    let addend=if s.cols&CONV_ADDEND==0 {0.0} else {x[2][index]};
    let slot=x[3][0] as usize;
    if slot>=taps || slot as f32!=x[3][0] {return f32::NAN;}
    let mut row=if slot+1==taps {0} else {slot+1};
    let mut sum=0.0;
    k=0;
    while k<CONV_GROUP {
        if k<taps {
            let lag=taps-1-k;
            sum+=pick(&tap,lag)*rn(pick(&ring,row));
            row=if row+1==taps {0} else {row+1};
        }
        k+=1;
    }
    let y=rn((sum+rn(pick(&ring,slot)))*s.bound);
    if s.cols&CONV_ADDEND==0 {y} else if s.cols&CONV_ROUND_SUM==0 {addend+y} else {rn(addend+y)}
}
