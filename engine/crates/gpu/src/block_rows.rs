//! Many rows of one Inkling block arm in one launch: the host half of the rows entry of the arms (Ruling R6g; the
//! entry `resident_launch_block_rows` is in the Inkling image, `kernels-inkling`). No entry of the general image reads this file.
//!
//! A rows launch of `rows` rows runs the arm of `s` on each row. Unit `index` of the launch is unit
//! `index % units` of row `index / units`, and it writes `words` words at `index * words`, so the output holds the
//! output of each row in turn. Input `n` holds the words of each row in turn, or, when bit `n` of `shared` is set,
//! the words of ONE row that every row reads whole (a weight-like input of a step). The weight is always shared.
//!
//! List form (bit `LIST` of the mask, for `RowDots` and `MixBlend` only): the vectors of the arm (`RowDots` input 0,
//! `MixBlend` input 1) lie as one joined list of all rows, vector `e` of row `r` at `(e * rows + r) * width`, so a
//! step of rows reads the block list as it is made and needs no swap before each mix. Row `r` gives the words of the
//! arm on the vectors of row `r` in rising `e`: the same values in the same order, so the same bits (host test). The
//! input has the same count as when each row holds its own vectors; it is not a shared input.
//!
//! Every function here is pure and takes only counts, so the kernel entry reads its slices by the same counts as
//! the host checks them.
use crate::block_ops::{Op,Shape,checked_lengths,split,unit as arm_unit};

/// The most rows of one rows launch.
pub const ROWS_MAX:usize=1<<16;

/// True when input `n` is a shared input: bit `n` of `shared` is set.
#[inline(always)]
pub fn is_shared(shared:u32,n:usize)->bool {shared>>n&1==1}

/// Bit 4 of the mask: the vectors of the arm are read in list form (module text).
pub const LIST:u32=1<<4;

/// The input that the list form reads as a joined list, and the words of one vector: `RowDots` input 0 (`cols`
/// words), `MixBlend` input 1 (`dim` words). None: the arm has no list form.
#[inline(always)]
pub fn listed(s:Shape)->Option<(usize,usize)> {
    match s.op {Op::RowDots=>Some((0,s.cols as usize)),Op::MixBlend=>Some((1,s.dim as usize)),_=>None}
}

/// The counts of a rows launch: the words of each input, of the weight, the units of all rows and the words of a
/// unit. None: the arm has no counts, `rows` is 0 or above `ROWS_MAX`, the mask names an input past the fourth or an
/// input the arm does not read, the list form is asked of an arm without one or with its listed input shared, or a
/// count exceeds what the device ABI holds.
#[inline(always)]
pub fn lengths(s:Shape,rows:usize,shared:u32)->Option<([usize;4],usize,usize,usize)> {
    let (x,weight,units,words)=checked_lengths(s)?;
    if rows==0 || rows>ROWS_MAX || shared>=LIST<<1 || (0..4).any(|n|is_shared(shared,n) && x[n]==0) {return None;}
    if shared&LIST!=0 && listed(s).is_none_or(|(n,_)|is_shared(shared,n)) {return None;}
    let extent=|n:usize|if is_shared(shared,n) {Some(x[n])} else {rows.checked_mul(x[n])};
    let lengths=[extent(0)?,extent(1)?,extent(2)?,extent(3)?];
    let all=rows.checked_mul(units)?;
    if lengths.into_iter().chain([all.checked_mul(words)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
    Some((lengths,weight,all,words))
}

/// Write unit `index` of a rows launch. `x` holds the inputs of all rows (the counts of `lengths`); the caller has
/// checked them, and `index` is below the units of all rows.
#[inline(always)]
pub fn unit(s:Shape,rows:usize,shared:u32,index:usize,x:[&[f32];4],w:&[f32],out:&mut [f32]) {
    let Some((one,_,units,_))=checked_lengths(s) else {return;};
    if units==0 || rows==0 {return;}
    let (row,inner)=(index/units,index%units);
    let part=|n:usize|if is_shared(shared,n) {x[n]} else {&x[n][row*one[n]..(row+1)*one[n]]};
    if shared&LIST!=0 {list_unit(s,rows,shared,index,x,w,out);return;}
    arm_unit(s,inner,[part(0),part(1),part(2),part(3)],w,out)
}

/// Unit `index` of a rows launch in list form: the words of `RowDots` or `MixBlend` (`block_stream::unit`) with
/// vector `e` of row `row` read at `(e * rows + row) * width` of the joined list `x[listed]`.
#[inline(always)]
fn list_unit(s:Shape,rows:usize,shared:u32,index:usize,x:[&[f32];4],w:&[f32],out:&mut [f32]) {
    let Some((one,_,units,_))=checked_lengths(s) else {return;};
    let (h,d,k)=(s.heads as usize,s.dim as usize,s.cols as usize);
    let (row,inner)=(index/units,index%units);
    let vector=|e:usize,width:usize,n:usize|{let at=(e*rows+row)*width;&x[n][at..at+width]};
    match s.op {
        // One dot: one chain over the width in rising columns from +0.0, as `RowDots`.
        Op::RowDots=>{
            let (query,e)=split(inner,h);
            let (weights,rest)=w[query*k..(query+1)*k].as_chunks::<8>();let (values,last)=vector(e,k,0).as_chunks::<8>();
            let mut sum=0.0f32;
            for (a,b) in values.iter().zip(weights) {
                sum+=a[0]*b[0];sum+=a[1]*b[1];sum+=a[2]*b[2];sum+=a[3]*b[3];sum+=a[4]*b[4];sum+=a[5]*b[5];sum+=a[6]*b[6];sum+=a[7]*b[7];
            }
            for (&a,&b) in last.iter().zip(rest) {sum+=a*b;}
            out[0]=sum;
        },
        // The scores, their largest, the exp sum and the blend in rising list place, as `MixBlend`.
        Op::MixBlend=>{
            use crate::fixed_math::exp;
            let (query,channel)=split(inner,d);
            let own=if is_shared(shared,0) {x[0]} else {&x[0][row*one[0]..(row+1)*one[0]]};let dots=&own[query*k..(query+1)*k];
            let mut largest=f32::NEG_INFINITY;for &dot in dots {largest=largest.max(dot*s.bound);}
            let mut total=0.0f32;for &dot in dots {total+=exp(dot*s.bound-largest);}
            let mut sum=0.0f32;for (i,&dot) in dots.iter().enumerate() {sum+=(exp(dot*s.bound-largest)/total)*vector(i,d,1)[channel];}
            out[0]=sum;
        },
        _=>{},
    }
}

/// The words of a rows launch on the host: `causal_ops::reference` of each row in turn. The counts of the inputs
/// and of the weight must be those of `lengths`.
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(s:Shape,rows:usize,shared:u32,x:[&[f32];4],w:&[f32])->Result<Vec<f32>,String> {
    let (lengths,weight,_,_)=lengths(s,rows,shared).ok_or("invalid block rows shape")?;
    if (0..4).any(|n|x[n].len()!=lengths[n]) || w.len()!=weight {return Err("block rows operand length differs".into());}
    let (one,_,_,_)=checked_lengths(s).ok_or("invalid block rows shape")?;
    let mut out=Vec::new();
    for row in 0..rows {
        // List form: the row's own vectors, gathered from the joined list in rising place, then the arm as today.
        let gathered=listed(s).filter(|_|shared&LIST!=0).map(|(n,width)|(n,(0..one[n]/width).flat_map(|e|&x[n][(e*rows+row)*width..(e*rows+row+1)*width]).copied().collect::<Vec<f32>>()));
        let part=|n:usize|match &gathered {Some((m,g)) if *m==n=>g.as_slice(),_=>if is_shared(shared,n) {x[n]} else {&x[n][row*one[n]..(row+1)*one[n]]}};
        out.extend(crate::causal_ops::reference(s,[part(0),part(1),part(2),part(3)],w)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_ops::Op;
    fn value(i:usize,scale:f32)->f32 {((i*37%101) as f32-50.0)*scale}
    fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|v.to_bits()).collect()}
    /// Each arm with a small shape. The step words of an arm are counts and slots, so each row gets them from `words`.
    fn arms()->[Shape;14] {
        let s=|op,heads,dim,taps,rows,cols,epsilon,bound|Shape {op,heads,dim,taps,rows,cols,epsilon,bound};
        [s(Op::HeadClip,2,4,1,0,1,0.0,1.0),s(Op::RelBias,2,4,0,0,8,0.0,0.0),s(Op::ShortConv,2,3,4,0,3,0.0,0.5),
         s(Op::WindowScores,2,4,1,4,4,0.0,0.5),s(Op::SinkSoftmax,2,0,0,4,0,0.0,0.0),s(Op::WindowValues,2,4,1,4,0,0.0,0.0),
         s(Op::MixBlend,1,2,0,0,2,0.0,0.5),s(Op::GateBlend,0,2,0,0,0,0.0,0.0),s(Op::CappedGate,0,2,0,1,0,4.0,25.0),
         s(Op::RouteFold,0,2,0,2,0,0.0,0.0),s(Op::Softcap,0,0,0,2,0,0.03125,32.0),s(Op::PlainNorm,1,2,0,0,0,1.0e-6,0.0),
         s(Op::RoundedNorm,1,2,1,0,0,1.0e-6,0.0),s(Op::RowDots,2,0,0,3,5,0.0,0.0)]
    }
    fn words(s:Shape,n:usize,row:usize,count:usize)->Vec<f32> {
        match (s.op,n) {
            (Op::ShortConv,3)=>vec![(row%s.taps as usize) as f32],
            (Op::SinkSoftmax,2)=>vec![(1+row%s.rows as usize) as f32],
            (Op::WindowScores|Op::WindowValues,3)=>vec![(1+row%s.rows as usize) as f32,(row%s.rows as usize) as f32],
            _=>(0..count).map(|i|value(i*5+n*17+row*29+3,0.0625)).collect(),
        }
    }
    #[test]
    fn each_row_of_a_rows_launch_equals_the_arm_on_that_row_with_and_without_shared_inputs() {
        let mut masked=0;
        for s in arms() {
            assert!(crate::block_ops::is_block(s.op));
            let (per,weights,units,size)=checked_lengths(s).unwrap();
            let w=(0..weights).map(|i|value(i+11,0.125)+2.0).collect::<Vec<_>>();
            for mask in 0u32..16 {
                if (0..4).any(|n|is_shared(mask,n) && per[n]==0) {assert!(lengths(s,5,mask).is_none(),"{:?} mask {mask}",s.op);continue;}
                for m in [1usize,2,5,129] {
                    let row=|n:usize,r:usize|words(s,n,if is_shared(mask,n) {0} else {r},per[n]);
                    let x:[Vec<f32>;4]=core::array::from_fn(|n|if is_shared(mask,n) {row(n,0)} else {(0..m).flat_map(|r|row(n,r)).collect()});
                    assert_eq!(lengths(s,m,mask),Some((core::array::from_fn(|n|x[n].len()),weights,m*units,size)),"{:?} mask {mask} rows {m}",s.op);
                    let got=reference(s,m,mask,[&x[0],&x[1],&x[2],&x[3]],&w).unwrap();
                    // The unit of the kernel entry, unit by unit, gives the words of the host reference.
                    let mut units_out=vec![f32::NAN;m*units*size];
                    for index in 0..m*units {unit(s,m,mask,index,[&x[0],&x[1],&x[2],&x[3]],&w,&mut units_out[index*size..(index+1)*size]);}
                    assert_eq!(bits(&units_out),bits(&got),"{:?} mask {mask} rows {m}",s.op);
                    for r in 0..m {
                        let parts:[Vec<f32>;4]=core::array::from_fn(|n|row(n,r));
                        let one=crate::causal_ops::reference(s,[&parts[0],&parts[1],&parts[2],&parts[3]],&w).unwrap();
                        assert_eq!(bits(&one),bits(&got[r*units*size..(r+1)*units*size]),"{:?} mask {mask} rows {m} row {r}",s.op);
                    }
                    // An input one word short or one word long is refused, shared or not, and so is a wrong weight.
                    for n in 0..4 {if per[n]>0 {for change in [-1isize,1] {
                        let mut wrong=x.clone();if change<0 {wrong[n].pop();} else {wrong[n].push(0.0);}
                        assert!(reference(s,m,mask,[&wrong[0],&wrong[1],&wrong[2],&wrong[3]],&w).is_err());
                    }}}
                    if weights>0 {assert!(reference(s,m,mask,[&x[0],&x[1],&x[2],&x[3]],&w[1..]).is_err());}
                }
                masked+=usize::from(mask!=0);
            }
        }
        assert!(masked>=30,"only {masked} masks ran");
    }
    /// The list form gives, bit for bit, the words of today's rows launch on the same vectors laid out row by row,
    /// with every mask of the other inputs, unit by unit as the kernel entry writes them.
    #[test]
    fn the_list_form_equals_the_rows_launch_on_each_rows_own_vectors() {
        let mut runs=0;
        for s in arms().into_iter().filter(|&s|listed(s).is_some()).chain([
            Shape {op:Op::RowDots,heads:3,dim:0,taps:0,rows:4,cols:2304,epsilon:0.0,bound:0.0},
            Shape {op:Op::MixBlend,heads:2,dim:2304,taps:0,rows:0,cols:3,epsilon:0.0,bound:0.0625},
            Shape {op:Op::RowDots,heads:1,dim:0,taps:0,rows:2,cols:13,epsilon:0.0,bound:0.0}]) {
            let (n,width)=listed(s).unwrap();
            let (per,weights,units,size)=checked_lengths(s).unwrap();
            let list=per[n]/width;
            let w=(0..weights).map(|i|value(i+11,0.125)+2.0).collect::<Vec<_>>();
            for mask in (0u32..16).filter(|&m|!is_shared(m,n) && (0..4).all(|i|!is_shared(m,i) || per[i]>0)) {
                for m in [1usize,2,5,129] {
                    let row=|i:usize,r:usize|words(s,i,if is_shared(mask,i) {0} else {r},per[i]);
                    let plain:[Vec<f32>;4]=core::array::from_fn(|i|if is_shared(mask,i) {row(i,0)} else {(0..m).flat_map(|r|row(i,r)).collect()});
                    // The same vectors as a joined list: vector e of row r at (e * rows + r) * width.
                    let mut joined=plain.clone();
                    for r in 0..m {for e in 0..list {
                        joined[n][(e*m+r)*width..(e*m+r+1)*width].copy_from_slice(&plain[n][r*per[n]+e*width..r*per[n]+(e+1)*width]);
                    }}
                    let today=reference(s,m,mask,[&plain[0],&plain[1],&plain[2],&plain[3]],&w).unwrap();
                    let got=reference(s,m,mask|LIST,[&joined[0],&joined[1],&joined[2],&joined[3]],&w).unwrap();
                    assert_eq!(bits(&got),bits(&today),"{:?} mask {mask} rows {m}",s.op);
                    let mut units_out=vec![f32::NAN;m*units*size];
                    for index in 0..m*units {unit(s,m,mask|LIST,index,[&joined[0],&joined[1],&joined[2],&joined[3]],&w,&mut units_out[index*size..(index+1)*size]);}
                    assert_eq!(bits(&units_out),bits(&today),"{:?} mask {mask} rows {m} units",s.op);
                    if m>1 && list>1 {assert_ne!(bits(&joined[n]),bits(&plain[n]),"{:?} rows {m}: the test lays out no list",s.op);}
                    runs+=1;
                }
            }
        }
        assert_eq!(runs,28,"list runs: 3 RowDots shapes x 1 mask x 4 row counts, 2 MixBlend shapes x 2 masks x 4");
    }
    #[test]
    fn a_rows_launch_refuses_what_it_does_not_hold() {
        for s in arms() {
            assert!(lengths(s,0,0).is_none() && lengths(s,ROWS_MAX+1,0).is_none(),"{:?}",s.op);
            assert!(lengths(s,ROWS_MAX,0).is_some(),"{:?}",s.op);
            for high in [32u32,33,1<<31] {assert!(lengths(s,4,high).is_none(),"{:?} mask {high}",s.op);}
            // The list form only for the two arms that have one, and never with the listed input shared.
            match listed(s) {
                None=>assert!(lengths(s,4,LIST).is_none() && lengths(s,4,LIST|1).is_none(),"{:?}",s.op),
                Some((n,_))=>assert!(lengths(s,4,LIST).is_some() && lengths(s,4,LIST|1<<n).is_none(),"{:?}",s.op),
            }
            assert!(lengths(Shape {epsilon:f32::NAN,..s},4,0).is_none() && lengths(Shape {bound:f32::INFINITY,..s},4,0).is_none(),"{:?}",s.op);
        }
        // An operation that is not a block arm has no rows launch here.
        for op in [Op::Prepare,Op::StreamMean,Op::Project,Op::PackBf16] {
            let s=Shape {op,heads:2,dim:4,taps:0,rows:3,cols:4,epsilon:0.0,bound:0.0};
            assert!(lengths(s,4,0).is_none(),"{op:?}");
        }
    }
}
