//! Token runs for a prompt group larger than the swept range of the reading kernels.
//!
//! The reading kernels of `prompt_admission`, `softmax_rows` and `select_admission` are admitted, and swept on the device, for
//! 1 to `GROUP_MAX` tokens. A larger group used to take `prompt_rows` for these operations, one thread a unit, about ten times
//! slower. Where every token of an operation is independent and every input and the output are token-major, the group runs the
//! same reading kernel over runs of at most `GROUP_MAX` tokens instead. Each run is a tuple the sweeps hold. It reads the words
//! of its own tokens and writes the output words of its own tokens. So each token gets the launch that a group of the run's size
//! gives it, and the bits are those of the whole operation.
//!
//! Operations with state between tokens (the convolution history, the recurrent update) are not here: the decoder runs the
//! convolution in runs with its history, and the recurrent kernel takes any token count.
use crate::prompt_admission::GROUP_MAX;
use crate::prompt_ops::{Op,Shape};
/// One run of tokens: its shape, the word offset of its tokens in each input, and the word offset of its output.
#[derive(Debug,Clone,Copy)]
pub struct Run {pub shape:Shape,pub inputs:[usize;5],pub output:usize}
/// True when a group of this shape (1 to `GROUP_MAX` tokens) has a reading kernel whose tokens are independent.
fn reading(s:Shape)->bool {
    s.op==Op::Wrapped && crate::prompt_admission::fast_kernel(s).is_some() || crate::softmax_rows::admitted(s) || crate::select_admission::admitted(s)
}
/// The runs of a group past `GROUP_MAX` tokens, in token order, when every run has a reading kernel. None: the operation keeps
/// its one launch (a group inside the swept range, an operation with no reading kernel, or a layout that is not token-major).
pub fn runs(shape:Shape)->Option<Vec<Run>> {
    let m=shape.tokens as usize;
    if m<=GROUP_MAX || !matches!(shape.op,Op::Wrapped|Op::SoftmaxRows|Op::SelectRows) {return None;}
    let (lengths,_,units,words)=shape.lengths()?;
    // Token-major: each input and the output hold the same number of words for every token.
    if lengths.iter().any(|n|n%m!=0) || units%m!=0 {return None;}
    let per=lengths.map(|n|n/m);let out=(units/m).checked_mul(words)?;
    let mut list=Vec::with_capacity(m.div_ceil(GROUP_MAX));let mut first=0usize;
    while first<m {
        let n=(m-first).min(GROUP_MAX);let mut sub=shape;sub.tokens=n as u32;
        // A selection's causal context is that of the run's first token.
        if shape.op==Op::SelectRows {sub.first=shape.first.checked_add(u32::try_from(first).ok()?)?;}
        let (sub_lengths,_,sub_units,sub_words)=sub.lengths()?;
        if !reading(sub) || sub_lengths!=per.map(|p|p*n) || sub_words!=words || sub_units.checked_mul(sub_words)!=out.checked_mul(n) {return None;}
        list.push(Run {shape:sub,inputs:per.map(|p|p*first),output:out.checked_mul(first)?});first+=n;
    }
    Some(list)
}
/// The row runs of an mHC boundary past `GROUP_MAX` rows whose `GROUP_MAX`-row form takes the token-tiled project: (first row,
/// the run's shape), runs of `GROUP_MAX` rows and a shorter last one. Each run is a row count a group of that size already
/// takes: the tile from `MHC_MIN_ROWS` rows, the one-block project below it. None for any other shape, which keeps one launch.
pub fn mhc_row_runs(s:crate::mhc_params::Shape)->Option<Vec<(u32,crate::mhc_params::Shape)>> {
    let max=GROUP_MAX as u32;
    if s.rows<=max || !crate::prompt_admission::mhc_tile_admitted(crate::mhc_params::Shape {rows:max,..s}) {return None;}
    Some((0..s.rows).step_by(GROUP_MAX).map(|first|(first,crate::mhc_params::Shape {rows:(s.rows-first).min(max),..s})).collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_ops::{Op as Causal,Shape as CausalShape};
    use crate::prompt_ops::reference;
    fn wrapped(op:Causal,heads:u32,dim:u32,taps:u32,rows:u32,cols:u32,epsilon:f32,bound:f32,tokens:usize)->Shape {
        Shape::wrapped(CausalShape {op,heads,dim,taps,rows,cols,epsilon,bound},tokens).unwrap()
    }
    fn values(n:usize,salt:usize)->Vec<f32> {(0..n).map(|i|((i*37+salt*11)%113) as f32/64.0-0.875).collect()}
    /// The whole operation on the host reference against its runs joined: the same words.
    fn same(shape:Shape,x:[Vec<f32>;5],w:&[f32]) {
        let whole=reference(shape,[&x[0],&x[1],&x[2],&x[3],&x[4]],w).unwrap();
        let list=runs(shape).expect("runs");let (lengths,_,units,words)=shape.lengths().unwrap();
        assert_eq!(whole.len(),units*words);let mut joined=vec![f32::NAN;whole.len()];let mut tokens=0;
        for run in &list {
            let (l,_,u,wd)=run.shape.lengths().unwrap();
            assert!(run.shape.tokens as usize<=GROUP_MAX && run.shape.tokens>0);
            let part:Vec<&[f32]>=(0..5).map(|i|if lengths[i]==0 {&x[i][..0]} else {&x[i][run.inputs[i]..run.inputs[i]+l[i]]}).collect();
            let got=reference(run.shape,[part[0],part[1],part[2],part[3],part[4]],w).unwrap();
            assert_eq!(got.len(),u*wd);joined[run.output..run.output+got.len()].copy_from_slice(&got);tokens+=run.shape.tokens as usize;
        }
        assert_eq!(tokens,shape.tokens as usize);
        assert_eq!(whole.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),joined.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"{shape:?}");
    }
    #[test]
    fn a_group_inside_the_swept_range_keeps_one_launch() {
        for m in [1usize,2,255,256] {
            assert!(runs(wrapped(Causal::Route,1,1,1,288,8,0.0,2.5,m)).is_none());
            assert!(runs(wrapped(Causal::ExpertFold,0,0,1,8,4096,0.0,0.0,m)).is_none());
        }
        // No reading kernel for this shape (16 experts): one launch at any size.
        assert!(runs(wrapped(Causal::Route,1,1,1,16,8,0.0,2.5,1024)).is_none());
        // The convolution carries history between tokens: the decoder runs it in runs, not this module.
        let mut conv=Shape::new(Op::ConvRows,1024).unwrap();conv.heads=64;conv.dim=128;conv.taps=4;
        assert!(crate::prompt_admission::fast_kernel(Shape {tokens:256,..conv}).is_some() && runs(conv).is_none());
    }
    #[test]
    fn runs_cover_each_token_once_in_order_with_swept_shapes() {
        for m in [257usize,300,511,512,513,768,1023,1024] {
            let shape=wrapped(Causal::Route,1,1,1,288,8,0.0,2.5,m);let list=runs(shape).unwrap();
            assert_eq!(list.len(),m.div_ceil(GROUP_MAX));
            let (lengths,_,units,words)=shape.lengths().unwrap();let (mut at,mut out)=(0usize,0usize);
            for run in &list {
                let n=run.shape.tokens as usize;assert!((1..=GROUP_MAX).contains(&n));
                assert_eq!(crate::prompt_admission::fast_kernel(run.shape),Some("prompt_route"));
                assert_eq!(run.inputs[0],at*(lengths[0]/m));assert_eq!(run.output,out);at+=n;out+=n*(units/m)*words;
            }
            assert_eq!((at,out),(m,units*words));
        }
    }
    #[test]
    fn route_and_fold_runs_equal_the_whole_operation_on_the_host() {
        for m in [257usize,520,1024] {
            // Route: 288 router logits a token and the bias; 8 ids then 8 weights out.
            let route=wrapped(Causal::Route,1,1,1,288,8,0.0,2.5,m);let (l,w,_,_)=route.lengths().unwrap();
            same(route,[values(l[0],1),vec![],vec![],vec![],vec![]],&values(w,2));
            // Fold: the 8 routed rows of 4096, their weights and the shared row.
            let fold=wrapped(Causal::ExpertFold,0,0,1,8,4096,0.0,0.0,m);let (l,w,_,_)=fold.lengths().unwrap();
            // Ids are small whole numbers; the fold only weights rows, it does not index by id.
            let ids=(0..l[0]).map(|i|(i%8) as f32).collect::<Vec<_>>();
            same(fold,[ids,values(l[1],3),values(l[2],4),values(l[3],5),vec![]],&values(w,6));
        }
    }
    #[test]
    fn prepare_and_gated_norm_runs_equal_the_whole_operation_on_the_host() {
        let m=300usize;
        let prepare=wrapped(Causal::Prepare,64,128,0,0,0,1e-6,-5.0,m);let (l,w,_,_)=prepare.lengths().unwrap();
        same(prepare,[values(l[0],1),values(l[1],2),values(l[2],3),values(l[3],4),vec![]],&values(w,5));
        let gated=wrapped(Causal::GatedNorm,64,128,0,0,0,1e-6,0.0,m);let (l,w,_,_)=gated.lengths().unwrap();
        same(gated,[values(l[0],6),values(l[1],7),values(l[2],8),values(l[3],9),vec![]],&values(w,10));
    }
    #[test]
    fn selection_runs_keep_each_token_its_own_context() {
        // Pools of 4, budget 2048, tail kept: 2051 slots. The group starts at context 3000 and holds 600 tokens.
        let (m,first)=(600usize,3000u32);let mut s=Shape::new(Op::SelectRows,m).unwrap();
        s.dim=4;s.taps=2048;s.cols=1;s.slots=2051;s.first=first+1;s.rows=(first+m as u32)/4;
        assert!(crate::select_admission::admitted(Shape {tokens:256,..s}),"the model's selection shape must have its reading kernel");
        let list=runs(s).unwrap();assert_eq!(list.len(),3);
        assert_eq!(list.iter().map(|r|(r.shape.tokens,r.shape.first)).collect::<Vec<_>>(),[(256,first+1),(256,first+257),(88,first+513)]);
        let (l,_,_,_)=s.lengths().unwrap();
        same(s,[(0..l[0]).map(|i|((i*7919)%1009) as f32/1009.0).collect(),vec![],vec![],vec![],vec![]],&[]);
    }
    #[test]
    fn mhc_row_runs_cover_the_rows_with_the_launch_each_run_size_takes() {
        let shape=|rows:u32,streams:u32,hidden:u32|crate::mhc_params::Shape {rows,streams,hidden,iterations:20,rms_eps:1e-6,eps:1e-6};
        for rows in [1u32,128,256] {assert!(mhc_row_runs(shape(rows,4,4096)).is_none());}
        // Not the model's boundary: no tile at 256 rows, so no runs.
        assert!(mhc_row_runs(shape(1024,2,4096)).is_none() && mhc_row_runs(shape(1024,4,2048)).is_none());
        for rows in [257u32,300,512,640,1000,1024] {
            let runs=mhc_row_runs(shape(rows,4,4096)).unwrap();let mut at=0;
            for (first,sub) in &runs {
                assert_eq!(*first,at);assert!((1..=256).contains(&sub.rows));at+=sub.rows;
                // The tile from 128 rows; a shorter last run keeps the one-block project, as a group of that size does.
                assert_eq!(crate::prompt_admission::mhc_tile_admitted(*sub),sub.rows>=128);
            }
            assert_eq!(at,rows);assert_eq!(runs.len(),(rows as usize).div_ceil(256));
        }
    }
    #[test]
    fn softmax_runs_equal_the_whole_operation_on_the_host() {
        // The model's attention: 64 heads, 2051 slots. Three tokens past one run.
        let m=GROUP_MAX+3;let mut s=Shape::new(Op::SoftmaxRows,m).unwrap();s.heads=64;s.cols=64;s.slots=2051;
        if !crate::softmax_rows::admitted(Shape {tokens:256,..s}) {return;}
        let (l,_,_,_)=s.lengths().unwrap();let words=2052usize;
        // Each token's selection row: ids then the count in the last word.
        let mut selected=vec![0.0f32;l[1]];for token in 0..m {for slot in 0..64 {selected[token*words+slot]=slot as f32;}selected[token*words+2051]=64.0;}
        same(s,[values(l[0],1),selected,vec![],vec![],vec![]],&[]);
    }
}
