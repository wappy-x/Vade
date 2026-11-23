//! Block arms of the stream: `ShortConv`, `MixBlend`, `GateBlend`, `PlainNorm`, `RoundedNorm`, `RowDots`. Safe shared code
//! for the host and the GPU.
//!
//! Shape fields: `h` = heads, `d` = dim, `t` = taps, `r` = rows, `k` = cols. Counts are 32-bit words.
//! - `ShortConv`: `h` heads of `d` channels, `t` taps; `k` = form word: bit 0 token-made taps given, bit 1 addend
//!   given, bit 2 round after the addend; `bound` = the reciprocal of the divisor (above zero). Inputs: the ring
//!   [t*h*d], slot then channel; the token-made taps [h*t] or empty, head then lag; the addend [h*d] or empty; the
//!   step word [1] = ring slot of the current token. Weight: the fixed taps [h*d*t], channel then lag. Units `h*d`,
//!   words 1.
//! - `MixBlend`: `h` queries, `k` vectors in the stack, `d` = width; `bound` = the scale (above zero). Inputs: the
//!   dots [h*k], the normed stack [k*d]. Units `h*d`, words 1.
//! - `GateBlend`: `d` = width. Inputs: the forget row then the gain row [2*d], the kept vector [d], the message [d].
//!   Units `d`, words 1.
//! - `PlainNorm`: `h` rows of `d` values; `epsilon` = the epsilon of the norm (above zero). Input: [h*d]. The mean
//!   of squares uses the 32-lane tree of `causal_ops`. The result stays 32-bit. Units `h * d / p`, words `p`, where
//!   `p` = `block_ops::norm_part(d)`: 32 words when the width is a multiple of 32, else the whole row.
//! - `RoundedNorm`: `h` rows of `d` values; `t` = 1 if a weight is given, else 0; `epsilon` = the epsilon of the
//!   norm (above zero). Input: [h*d], FP32, read as it is. Weight: [t*d], one weight row for all rows. The mean of
//!   squares uses the tree of the norm kernel (`block_ops::norm_square_sum`). The result is BF16. Units and words
//!   as `PlainNorm`.
//! - `RowDots`: `r` weight rows and `h` input rows of `k` values, the fields of `Op::ProjectRows`. Input: [h*k].
//!   Weight: [r*k]. Units `r*h`, words 1: unit `row * h + line` is the dot of weight row `row` with input row `line`.
//!   The output has the layout and the words of `Op::ProjectRows`, with one GPU thread a dot and any count of input rows.
//!
//! A unit of a norm is one part of one row, so many GPU threads write a row. Each of them makes the sum of the
//! squares of the whole row itself, in the one fixed order: the scale has the same bits in every part.
//!
//! Number rules. Every value is one FP32 word. A product and its sum are two rounded steps, never a fused step.
//! An input that its producer holds as BF16 is read through `rn`; on a word that is BF16 already `rn` changes
//! nothing. Thus a dense product can leave its FP32 sum in memory, and the bytes are those of "round, then use".
//! `exp`, `tanh` and `rn` come from `fixed_math`.
//!
//! What each arm computes, for one output word:
//! - `ShortConv`, channel `c` of head `c / d`: the tap of lag `l` is the fixed tap, plus `rn` of the token-made
//!   tap of the head when bit 0 is set. `sum` adds `tap(l) * rn(ring[slot - l])` from +0.0, the oldest lag first and
//!   lag 0 last. Then `y = rn((sum + rn(ring[slot])) * bound)`. With bit 1 the result is `addend + y`, and bit 2
//!   rounds that sum. The addend is FP32 and is not rounded when it is read. Lag 0 is the current token. The row of
//!   lag `l` is `(slot - l) mod t`. A ring that starts as zeros gives the tokens before the text the value zero.
//! - `MixBlend`, query `q` and channel `c`: the scores are `dot[q][i] * bound`. The largest score is taken out, each
//!   weight is `exp(score - largest)` divided by the sum of these in rising `i` from +0.0, and the result adds
//!   `weight(i) * normed[i][c]` in rising `i` from +0.0. No value is rounded.
//! - `GateBlend`: `(1 - tanh(forget[c])) * kept[c] + tanh(gain[c]) * message[c]`: two products, then one sum.
//! - `RowDots`, one weight row and one input row: `input[j] * weight[j]` added in rising `j` from +0.0. A product
//!   and its add are two rounded steps. No value is rounded.
//! - `PlainNorm`, one row: `x[j] * (1 / sqrt(squares / d + epsilon))`, with `squares` on the 32-lane tree.
//! - `RoundedNorm`, one row: `rn((x[j] * scale) * weight[j])`, or `rn(x[j] * scale)` with no weight, where
//!   `scale = 1 / sqrt(squares / d + epsilon)` and `squares` is on the tree of the norm kernel. The input is not
//!   rounded when it is read, and the result is rounded one time.
//!
//! The two norms have two sum trees on purpose. A row of 128 values or less gives the same sum on both. Above that
//! width the sums can differ in the last bit, so a rounded `PlainNorm` is not a `RoundedNorm`.
//!
//! Source of the math (read for the math, no code copied): the public Transformers release v5.24.0, file
//! `models/inkling2_loop/modeling_inkling2_loop.py`. Short convolutions: lines 285-455. The two sums of the stream:
//! lines 892-910. Mix and message: lines 145-173 and 828-853. Gate: lines 79-96. Plain norm: lines 67-76.
//! The source leaves its sum orders to its library. The orders here are the engine's own.
use crate::{block_ops::{Lengths,Op,Shape,norm_part,norm_squares,root,row_squares,split},fixed_math::{exp,rn,tanh}};
/// Bits of the form word of `ShortConv` (`cols`).
pub const CONV_TOKEN_TAPS:u32=1;
pub const CONV_ADDEND:u32=2;
pub const CONV_ROUND_SUM:u32=4;
/// The place of the tap of lag `lag` in a channel's row of fixed taps and in a head's row of token-made taps.
/// Lag 0 is the current token, and the contract stores a row by lag. One function holds this layout fact, so
/// another stored order is a change of this line only.
#[inline(always)]
pub const fn tap_place(lag:usize,_taps:usize)->usize {lag}
/// Steps `0..n` of one `RowDots` chain, after the steps before them gave `sum`: `sum += v(j) * w(j)` in rising `j`,
/// each product rounded before its add, 8 steps a turn (the steps of the `RowDots` unit, so the same bits; host
/// test). The staged form of the entry calls it on each tile of the two rows in shared memory.
#[inline(always)]
pub fn dot_steps(mut sum:f32,n:usize,v:impl Fn(usize)->f32,w:impl Fn(usize)->f32)->f32 {
    let whole=n/8*8;let mut j=0;
    while j<whole {
        let a=[v(j),v(j+1),v(j+2),v(j+3),v(j+4),v(j+5),v(j+6),v(j+7)];let b=[w(j),w(j+1),w(j+2),w(j+3),w(j+4),w(j+5),w(j+6),w(j+7)];
        sum+=a[0]*b[0];sum+=a[1]*b[1];sum+=a[2]*b[2];sum+=a[3]*b[3];sum+=a[4]*b[4];sum+=a[5]*b[5];sum+=a[6]*b[6];sum+=a[7]*b[7];
        j+=8;
    }
    while j<n {sum+=v(j)*w(j);j+=1;}
    sum
}
/// Word counts of an arm of this file. `None` refuses the shape.
#[inline]
pub fn lengths(s:Shape)->Option<Lengths> {
    let (h,d,t,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize);
    match s.op {
        // Bit 2 rounds the sum with the addend, so it needs bit 1. A bit above bit 2 has no meaning.
        Op::ShortConv if h>0 && d>0 && t>0 && s.bound>0.0 && s.cols<2*CONV_ROUND_SUM && (s.cols&CONV_ROUND_SUM==0 || s.cols&CONV_ADDEND!=0)=>{
            let width=h.checked_mul(d)?;let taps=width.checked_mul(t)?;
            Some(([taps,if s.cols&CONV_TOKEN_TAPS!=0 {h.checked_mul(t)?} else {0},if s.cols&CONV_ADDEND!=0 {width} else {0},1],taps,width,1))
        },
        Op::MixBlend if h>0 && d>0 && k>0 && s.bound>0.0=>Some(([h.checked_mul(k)?,k.checked_mul(d)?,0,0],0,h.checked_mul(d)?,1)),
        Op::GateBlend if d>0=>Some(([d.checked_mul(2)?,d,d,0],0,d,1)),
        Op::PlainNorm if h>0 && d>0 && s.epsilon>0.0=>Some(([h.checked_mul(d)?,0,0,0],0,h.checked_mul(d/norm_part(d))?,norm_part(d))),
        Op::RowDots if h>0 && s.rows>0 && k>0 && d==0 && t==0 && s.epsilon==0.0 && s.bound==0.0=>{
            let rows=s.rows as usize;
            Some(([h.checked_mul(k)?,0,0,0],rows.checked_mul(k)?,rows.checked_mul(h)?,1))
        },
        Op::RoundedNorm if h>0 && d>0 && t<=1 && s.epsilon>0.0=>Some(([h.checked_mul(d)?,0,0,0],t*d,h.checked_mul(d/norm_part(d))?,norm_part(d))),
        _=>None,
    }
}
/// Write one output unit of an arm of this file. The caller has checked every extent with `lengths`.
#[inline]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    let (h,d,t,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize);
    match s.op {
        Op::ShortConv=>{
            // The slot comes from data. A word that names no slot of the ring reads no row: the unit gives a NaN,
            // and the host refuses a result that is not finite.
            let word=x[3][0];let slot=word as usize;
            if slot>=t || slot as f32!=word {out[0]=f32::NAN;return;}
            let width=h*d;let head=index/d;let token=s.cols&CONV_TOKEN_TAPS!=0;
            let mut sum=0.0f32;
            for lag in (0..t).rev() {
                let mut tap=w[index*t+tap_place(lag,t)];
                if token {tap+=rn(x[1][head*t+tap_place(lag,t)]);}
                sum+=tap*rn(x[0][((slot+t-lag)%t)*width+index]);
            }
            // Ruling R8: the source divides by a number, and its device form is one product with the reciprocal.
            let y=rn((sum+rn(x[0][slot*width+index]))*s.bound);
            out[0]=if s.cols&CONV_ADDEND==0 {y} else if s.cols&CONV_ROUND_SUM==0 {x[2][index]+y} else {rn(x[2][index]+y)};
        },
        Op::MixBlend=>{
            let (query,channel)=split(index,d);let dots=&x[0][query*k..(query+1)*k];
            let mut largest=f32::NEG_INFINITY;for &dot in dots {largest=largest.max(dot*s.bound);}
            let mut total=0.0f32;for &dot in dots {total+=exp(dot*s.bound-largest);}
            let mut sum=0.0f32;for (i,&dot) in dots.iter().enumerate() {sum+=(exp(dot*s.bound-largest)/total)*x[1][i*d+channel];}
            out[0]=sum;
        },
        Op::GateBlend=>{out[0]=(1.0-tanh(x[0][index]))*x[1][index]+tanh(x[0][d+index])*x[2][index];},
        // One dot: one chain over the width. The steps are written 8 at a time, in rising order.
        Op::RowDots=>{
            let (row,line)=split(index,h);
            let (weights,rest)=w[row*k..(row+1)*k].as_chunks::<8>();let (values,last)=x[0][line*k..(line+1)*k].as_chunks::<8>();
            let mut sum=0.0f32;
            for (a,b) in values.iter().zip(weights) {
                sum+=a[0]*b[0];sum+=a[1]*b[1];sum+=a[2]*b[2];sum+=a[3]*b[3];sum+=a[4]*b[4];sum+=a[5]*b[5];sum+=a[6]*b[6];sum+=a[7]*b[7];
            }
            for (&a,&b) in last.iter().zip(rest) {sum+=a*b;}
            out[0]=sum;
        },
        // A unit is one part of one row. The sum of the squares is that of the whole row.
        Op::PlainNorm=>{let part=norm_part(d);let (line,first)=split(index,d/part);normed(s,line,first*part,x,w,row_squares(&x[0][line*d..(line+1)*d]),out)},
        Op::RoundedNorm=>{let part=norm_part(d);let (line,first)=split(index,d/part);normed(s,line,first*part,x,w,norm_squares(&x[0][line*d..(line+1)*d]),out)},
        _=>{},
    }
}
/// Words of row `line` of `PlainNorm` or `RoundedNorm` from place `first` on, from the sum of the squares of the row.
#[inline(always)]
pub fn normed(s:Shape,line:usize,first:usize,x:[&[f32];4],w:&[f32],squares:f32,out:&mut[f32]) {
    let (d,t)=(s.dim as usize,s.taps as usize);
    let scale=1.0/root(squares/d as f32+s.epsilon);
    let values=&x[0][line*d+first..line*d+first+out.len()];
    if s.op==Op::PlainNorm {for (word,&value) in out.iter_mut().zip(values) {*word=value*scale;}}
    else if t==0 {for (word,&value) in out.iter_mut().zip(values) {*word=rn(value*scale);}}
    else {for ((word,&value),&weight) in out.iter_mut().zip(values).zip(&w[first..first+values.len()]) {*word=rn((value*scale)*weight);}}
}
