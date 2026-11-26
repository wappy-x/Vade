//! Test support of the attention half and of the token step: each block arm that these launch, written again from
//! its written contract, and a backend that runs them in place of the shared arms.
//!
//! The contract is the lab design of the device step (its arm table and its sum orders), the header of each arm
//! file under `gpu/src/`, and the lab rulings on the rounded norm and on a division by a number. The forms here are
//! test code only and share no line with the arms. So a run on `Hand` is this seat's own reading of the contract:
//! its words are the hand values of the driver tests. Where an arm has its body, the tests compare the two.
use crate::{Result,bounded_decoder::{ChatGather,Compute,Host},mxfp8_projection::Mxfp8Projection};
use engine_model::plan::admission::CheckpointAdmission;
use spark_gpu_probe::{causal_ops::{Op,Shape},fixed_math::{exp,rn,sigmoid,tanh},norm_boundary::NormShape};
use std::{cell::{Cell,RefCell},collections::BTreeMap};

/// Lanes of every fixed sum tree, and how the lanes meet: at the distances 16, 8, 4, 2, 1 each lane below the
/// distance adds its upper partner.
const LANES:usize=32;
fn joined(mut lanes:[f32;LANES])->f32 {
    let mut distance=LANES/2;
    while distance>0 {for lane in 0..distance {lanes[lane]+=lanes[lane+distance];}distance/=2;}
    lanes[0]
}
/// A sum on the 32 lanes: term `i` goes to lane `i mod 32`, each lane adds in rising `i` from +0.0.
fn lane_sum(terms:impl Iterator<Item=f32>)->f32 {let mut lanes=[0.0f32;LANES];for (i,term) in terms.enumerate() {lanes[i%LANES]+=term;}joined(lanes)}
/// The squares of one clip head: one 32-lane tree over the whole head.
pub fn clip_squares(x:&[f32])->f32 {lane_sum(x.iter().map(|v|v*v))}
/// The squares of a rounded norm, in the tree of the norm kernel: chunks of 128 values, a 32-lane tree in each
/// chunk, then the chunk sums on 32 lanes by chunk number.
pub fn norm_squares(x:&[f32])->f32 {lane_sum(x.chunks(128).map(|chunk|lane_sum(chunk.iter().map(|v|v*v))))}
/// A sum over keys by distance: blocks of 1,024 distances, a 32-lane tree in each block by distance, then the block
/// sums as one chain from +0.0.
pub fn distance_sum(n:usize,term:impl Fn(usize)->f32)->f32 {
    let mut total=0.0f32;
    for first in (0..n).step_by(1024) {total+=lane_sum((first..n.min(first+1024)).map(&term));}
    total
}
/// Value `j` of a packed row: value `2i` is the low half of word `i`, value `2i + 1` its high half.
pub fn unpacked(row:&[f32],j:usize)->f32 {let bits=row[j/2].to_bits();f32::from_bits(if j.is_multiple_of(2) {bits<<16} else {bits&0xffff_0000})}
/// A step word as a count: a whole number that is not above `most`.
fn count(word:f32,most:usize)->Result<usize> {
    let n=word as usize;if word>=0.0 && n as f32==word && n<=most {Ok(n)} else {Err(format!("hand arm: the step word {word} is no count up to {most}"))}
}
/// The row of distance `dist` in a ring or a plane of `rows` rows.
fn row_of(slot:usize,dist:usize,rows:usize)->usize {(slot+rows-dist)%rows}

/// One rounded norm row: `rn((x * scale) * weight)`, or `rn(x * scale)`.
pub fn rounded_norm(x:&[f32],weight:Option<&[f32]>,epsilon:f32)->Vec<f32> {
    let scale=1.0/(norm_squares(x)/x.len() as f32+epsilon).sqrt();
    x.iter().enumerate().map(|(j,&v)|match weight {Some(weight)=>rn((v*scale)*weight[j]),None=>rn(v*scale)}).collect()
}
/// One clip head. The input is read as BF16. The factor is the threshold times the 32-bit reciprocal of the length
/// plus 1e-6, and never above 1. One round after the clip, one more after the position factor.
pub fn head_clip(x:&[f32],scale:Option<&[f32]>,factor:Option<f32>,threshold:f32)->Vec<f32> {
    let y=x.iter().enumerate().map(|(j,&v)|match scale {Some(scale)=>rn(v)*scale[j],None=>rn(v)}).collect::<Vec<_>>();
    let reciprocal=1.0/(clip_squares(&y).sqrt()+1e-6);let clip=(threshold*reciprocal).min(1.0);
    y.iter().map(|&v|{let clipped=rn(v*clip);match factor {Some(factor)=>rn(clipped*factor),None=>clipped}}).collect()
}
/// The relative term of one head and one distance: a chain over the state, rounded once.
pub fn rel_bias(state:&[f32],table:&[f32],extent:usize,dist:usize)->f32 {
    let mut sum=0.0f32;for (j,&v) in state.iter().enumerate() {sum+=rn(v)*table[j*extent+dist];}rn(sum)
}
/// One output word of a short convolution. `ring` is `[taps][width]`, `fixed` is the channel's taps by lag, `made`
/// the token-made taps of the channel's head by lag. The oldest lag adds first, then the input comes back in.
#[allow(clippy::too_many_arguments)]
pub fn short_conv(ring:&[f32],width:usize,channel:usize,slot:usize,fixed:&[f32],made:Option<&[f32]>,reciprocal:f32,addend:Option<f32>,round_sum:bool)->f32 {
    let taps=fixed.len();let mut sum=0.0f32;
    for lag in (0..taps).rev() {
        let tap=match made {Some(made)=>fixed[lag]+rn(made[lag]),None=>fixed[lag]};
        sum+=tap*rn(ring[row_of(slot,lag,taps)*width+channel]);
    }
    let y=rn((sum+rn(ring[slot*width+channel]))*reciprocal);
    match addend {None=>y,Some(addend) if round_sum=>rn(addend+y),Some(addend)=>addend+y}
}
/// The cap `tanh(x * r) * c`, with `r` the reciprocal of the cap `c`.
pub fn cap(x:f32,reciprocal:f32,c:f32)->f32 {tanh(x*reciprocal)*c}

/// The hand form of one launch. `None`: not an arm that this file holds.
pub fn arm(s:Shape,x:[&[f32];4],w:&[f32])->Result<Option<Vec<f32>>> {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    let sized=|ok:bool|->Result<()> {if ok {Ok(())} else {Err(format!("hand arm {:?}: an extent or a field differs from the contract",s.op))}};
    let plane=|plane:&[f32]|->Result<usize> {sized(t>0 && h%t==0 && d%2==0 && r>0 && plane.len()==r*(h/t)*(d/2))?;Ok((h/t)*(d/2))};
    Ok(Some(match s.op {
        Op::RoundedNorm=>{
            sized(h>0 && d>0 && t<=1 && s.epsilon>0.0 && x[0].len()==h*d && w.len()==t*d && r==0 && k==0)?;
            x[0].chunks_exact(d).flat_map(|row|rounded_norm(row,(t==1).then_some(w),s.epsilon)).collect()
        },
        Op::HeadClip=>{
            sized(h>0 && d>0 && t<=1 && k<=1 && r==0 && s.bound>0.0 && s.epsilon==0.0 && x[0].len()==h*d && x[1].len()==k && w.len()==t*h*d)?;
            x[0].chunks_exact(d).enumerate().flat_map(|(head,row)|head_clip(row,(t==1).then(||&w[head*d..(head+1)*d]),(k==1).then(||x[1][0]),s.bound)).collect()
        },
        Op::RelBias=>{
            sized(h>0 && d>0 && k>0 && t==0 && r==0 && s.bound==0.0 && s.epsilon==0.0 && x[0].len()==h*d && w.len()==d*k)?;
            (0..h*k).map(|unit|rel_bias(&x[0][(unit/k)*d..(unit/k+1)*d],w,k,unit%k)).collect()
        },
        Op::ShortConv=>{
            let width=h*d;
            sized(width>0 && t>0 && s.bound>0.0 && k<8 && (k&4==0 || k&2!=0) && x[0].len()==t*width && x[1].len()==if k&1!=0 {h*t} else {0}
                && x[2].len()==if k&2!=0 {width} else {0} && x[3].len()==1 && w.len()==width*t)?;
            let slot=count(x[3][0],t-1)?;
            (0..width).map(|c|short_conv(x[0],width,c,slot,&w[c*t..(c+1)*t],(k&1!=0).then(||&x[1][(c/d)*t..(c/d+1)*t]),s.bound,(k&2!=0).then(||x[2][c]),k&4!=0)).collect()
        },
        Op::WindowScores=>{
            let row_words=plane(x[1])?;
            sized(h>0 && d>0 && k>0 && s.bound>0.0 && s.epsilon==0.0 && x[0].len()==h*d && x[2].len()==h*k && x[3].len()==2)?;
            let (n,slot)=(count(x[3][0],r)?,count(x[3][1],r-1)?);
            (0..h*r).map(|unit|{
                let (head,dist)=(unit/r,unit%r);if dist>=n {return 0.0;}
                let start=row_of(slot,dist,r)*row_words+(head/t)*(d/2);let key=&x[1][start..start+d/2];
                let mut sum=0.0f32;for j in 0..d {sum+=rn(x[0][head*d+j])*unpacked(key,j);}
                rn(rn(rn(sum)*s.bound)+if dist<k {x[2][head*k+dist]} else {0.0})
            }).collect()
        },
        Op::SinkSoftmax=>{
            sized(h>0 && r>0 && d==0 && t==0 && k==0 && s.bound==0.0 && s.epsilon==0.0 && x[0].len()==h*r && x[1].len()==h && x[2].len()==1)?;
            let n=count(x[2][0],r)?;
            (0..h).flat_map(|head|{
                let (scores,sink)=(&x[0][head*r..head*r+n],rn(x[1][head]));
                let top=scores.iter().fold(sink,|top,&score|if score>top {score} else {top});
                let total=distance_sum(n,|dist|exp(scores[dist]-top))+exp(sink-top);
                (0..r).map(|dist|if dist<n {rn(exp(scores[dist]-top)/total)} else {0.0}).collect::<Vec<_>>()
            }).collect()
        },
        Op::WindowValues=>{
            let row_words=plane(x[1])?;
            sized(h>0 && d>0 && k==0 && s.bound==0.0 && s.epsilon==0.0 && x[0].len()==h*r && x[2].len()==h*d && x[3].len()==2)?;
            let (n,slot)=(count(x[3][0],r)?,count(x[3][1],r-1)?);
            (0..h*d).map(|unit|{
                let (head,channel)=(unit/d,unit%d);
                let sum=distance_sum(n,|dist|{let start=row_of(slot,dist,r)*row_words+(head/t)*(d/2);x[0][head*r+dist]*unpacked(&x[1][start..start+d/2],channel)});
                rn(rn(sum)*sigmoid(rn(x[2][unit])))
            }).collect()
        },
        // The cap of the logits: `epsilon` holds the reciprocal of the cap.
        Op::Softcap=>{sized(r>0 && s.bound>0.0 && s.epsilon>0.0 && x[0].len()==r)?;x[0].iter().map(|&v|cap(rn(v),s.epsilon,s.bound)).collect()},
        // The expert activation: the weight holds the reciprocals of the gate cap and of the up cap.
        Op::CappedGate=>{
            sized(r>0 && d>0 && s.epsilon>0.0 && s.bound>0.0 && x[0].len()==r*2*d && w.len()==2)?;
            x[0].chunks_exact(2*d).flat_map(|pick|(0..d).map(|i|{let (gate,up)=(rn(pick[i]),rn(pick[d+i]));rn((cap(gate,w[0],s.epsilon)*sigmoid(gate))*cap(up,w[1],s.bound))}).collect::<Vec<_>>()).collect()
        },
        _=>return Ok(None),
    }))
}
/// Which launches a `Hand` backend takes from the hand forms.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Mode {
    /// Every arm that this file holds. A run in this mode gives the hand values.
    All,
    /// Only a launch that the host refuses. A run in this mode is the host run where every arm has its body.
    Refused,
}
/// The host backend with the hand forms of this file. It counts the launches of each operation, those that ran on
/// a hand form, and the calls of the norm of the backend.
///
/// The rounded norm of this family is arm `RoundedNorm`, and no part of the driver may call `Compute::norm`: its
/// host form rounds twice and its device form rounds its input. `norms` counts such calls, and a test holds it at 0.
/// `fail_at`: the launch with this number (counted from 1) fails, for the tests of a step that stops half way.
///
/// `direct`: with a list, the backend takes the direct convolution of a decode cohort (`Compute::short_conv_direct`)
/// from its hand form and writes the rows of each call into the list; with none it refuses the call, as the host
/// does. `direct_refused`: the direct call with this number (counted from 1) is refused.
pub struct Hand {pub mode:Mode,pub launches:RefCell<BTreeMap<u32,usize>>,pub by_hand:RefCell<BTreeMap<u32,usize>>,pub norms:Cell<usize>,pub fail_at:Cell<usize>,
    pub direct:RefCell<Option<Vec<usize>>>,pub direct_refused:Cell<usize>}
impl Hand {
    pub fn new(mode:Mode)->Self {Self {mode,launches:RefCell::default(),by_hand:RefCell::default(),norms:Cell::new(0),fail_at:Cell::new(0),direct:RefCell::new(None),direct_refused:Cell::new(0)}}
    pub fn count(&self,op:Op)->usize {self.launches.borrow().get(&(op as u32)).copied().unwrap_or(0)}
    pub fn launches(&self)->usize {self.launches.borrow().values().sum()}
}
impl Compute for Hand {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,logits:&Vec<f32>,rows:usize)->Result<Option<Vec<u32>>> {Host.rows_greedy(logits,rows)}
    fn has_latent_absorption(&self,name:&str)->bool {Host.has_latent_absorption(name)}
    fn gather_absorbed(&self,s:Shape,x:[Option<&Vec<f32>>;4],reserve:usize,expected:f32)->Result<Vec<f32>> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:Shape,query:&Vec<f32>,latent:&Vec<f32>,count:&Vec<f32>,slots:usize)->Result<Vec<f32>> {Host.latent_absorption(name,s,query,latent,count,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,query:&Vec<f32>,latent:&Vec<f32>,selected:&Vec<f32>)->Result<Vec<f32>> {Host.latent_absorption_rows(name,s,query,latent,selected)}
    fn latent_absorption_chats(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,gathers:&[ChatGather<'_,Vec<f32>>],stride:usize,query:&Vec<f32>,selected:&Vec<f32>)->Result<Option<Vec<f32>>> {
        Host.latent_absorption_chats(name,s,gathers,stride,query,selected)
    }
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {Host.upload(v)}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {Host.read(v)}
    fn select_known(&self,ids:&Vec<f32>,count:&Vec<f32>,row:&[f32])->Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->Result<()> {Host.prompt_end()}
    fn prompt_seeds(&self)->bool {Host.prompt_seeds()}
    fn prepare_beside(&self,c:&CheckpointAdmission,live:usize)->Result<()> {Host.prepare_beside(c,live)}
    fn words(&self,v:&Vec<f32>)->usize {v.len()}
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->Result<Vec<f32>> {Host.range(v,start,len)}
    fn join(&self,parts:&[&Vec<f32>])->Result<Vec<f32>> {Host.join(parts)}
    fn zeros(&self,len:usize)->Result<Vec<f32>> {Host.zeros(len)}
    fn operation(&self,s:Shape,x:[Option<&Vec<f32>>;4],w:&[f32])->Result<Vec<f32>> {
        *self.launches.borrow_mut().entry(s.op as u32).or_default()+=1;
        if self.launches()==self.fail_at.get() {return Err("hand backend: this launch was told to fail".into());}
        let slices=x.map(|v|v.map_or(&[][..],Vec::as_slice));
        let host=if self.mode==Mode::Refused {match Host.operation(s,x,w) {Ok(out)=>return Ok(out),Err(why)=>Some(why)}} else {None};
        match arm(s,slices,w)? {
            Some(out)=>{
                *self.by_hand.borrow_mut().entry(s.op as u32).or_default()+=1;
                if out.iter().all(|v|v.is_finite()) {Ok(out)} else {Err(format!("hand arm {:?}: a result is not finite",s.op))}
            },
            None=>match host {Some(why)=>Err(why),None=>Host.operation(s,x,w)},
        }
    }
    /// The direct convolution of a decode cohort, from its written contract (`bounded_decoder::ShortConvDirect`): row
    /// `r` reads the ring of its chat with the row's own slot taken by its input, the hand form of arm `ShortConv`
    /// gives its words, and `every` holds that ring of each row. As the device form, one call takes 2 to
    /// `DIRECT_CHATS_MAX` rows; any other count is refused.
    fn short_conv_direct(&self,p:&crate::bounded_decoder::ShortConvDirect<'_,Vec<f32>>)->Result<Option<(Vec<f32>,Vec<f32>)>> {
        let Some(call)=self.direct.borrow_mut().as_mut().map(|rows|{rows.push(p.rows);rows.len()}) else {return Ok(None)};
        let (s,rows)=(p.shape,p.rows);let (heads,taps,width)=(s.heads as usize,s.taps as usize,(s.heads*s.dim) as usize);
        if !(2..=spark_gpu_probe::fast_ops::DIRECT_CHATS_MAX).contains(&rows) || call==self.direct_refused.get() {return Ok(None);}
        if p.inputs.len()!=rows*width || p.chat_rings.len()!=rows || p.chat_rings.iter().any(|&(ring,own)|ring.len()!=taps*width || own>=taps)
            || p.token.is_some_and(|token|token.len()!=rows*heads*taps) || p.addend.is_some_and(|addend|addend.len()!=rows*width) {
            return Err("hand direct convolution: an operand differs from the rows of the call".into());
        }
        let (mut output,mut every)=(Vec::with_capacity(rows*width),Vec::with_capacity(rows*taps*width));
        for (row,&(ring,own)) in p.chat_rings.iter().enumerate() {
            let mut ring=ring.clone();ring[own*width..(own+1)*width].copy_from_slice(&p.inputs[row*width..(row+1)*width]);
            let token=p.token.map_or(&[][..],|token|&token[row*heads*taps..(row+1)*heads*taps]);
            let addend=p.addend.map_or(&[][..],|addend|&addend[row*width..(row+1)*width]);
            output.extend(arm(s,[&ring,token,addend,&[own as f32]],p.fixed)?.ok_or("hand direct convolution: the shape is no ShortConv arm")?);
            every.extend(ring);
        }
        Ok(Some((output,every)))
    }
    fn norm(&self,name:&str,s:NormShape,x:&Vec<f32>,w:&[f32])->Result<Vec<f32>> {self.norms.set(self.norms.get()+1);Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>)->Result<(Self::Pass,Vec<f32>)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&Vec<f32>)->Result<Vec<f32>> {Host.finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Vec<f32>)->Result<Vec<f32>> {Host.mx(p,x)}
    fn round(&self,x:&Vec<f32>)->Result<Vec<f32>> {Host.round(x)}
    fn mean(&self,x:&Vec<f32>,streams:u32,hidden:u32)->Result<Vec<f32>> {Host.mean(x,streams,hidden)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Vec<f32>>;5],w:&[f32])->Result<Vec<f32>> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&Vec<f32>,m:usize,w:&[f32])->Result<Vec<f32>> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>,m:usize)->Result<(Self::Pass,Vec<f32>)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&Vec<f32>,m:usize)->Result<Vec<f32>> {Host.rows_finish(p,x,m)}
}
/// True when the host arm of the logit cap has the ruled form: it takes the reciprocal of the cap in `epsilon` and
/// does not divide by the cap. With a word in `epsilon` that is not the reciprocal of the cap, only the ruled form
/// follows that word.
pub fn softcap_is_ruled()->bool {
    let s=Shape {op:Op::Softcap,heads:0,dim:0,taps:0,rows:1,cols:0,epsilon:0.25,bound:2.0};
    Host.operation(s,[Some(&vec![1.0]),None,None,None],&[]).is_ok_and(|out|out[0].to_bits()==cap(1.0,0.25,2.0).to_bits())
}
/// The bits of a row, for a comparison that tells +0.0 from -0.0.
pub fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|v.to_bits()).collect()}
