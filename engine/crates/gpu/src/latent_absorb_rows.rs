//! Grouped latent attention. Each token keeps the scalar SUM-ORDER v1 arithmetic. One tile kernel runs the four fold steps.
use crate::prompt_ops::{Op,Shape};
#[cfg(test)] mod tests;
pub fn admitted(op:Op)->bool {matches!(op,Op::AbsorbQuery|Op::AbsorbScores|Op::AbsorbSoftmax|Op::AbsorbValues|Op::AbsorbProject)}
/// `cols` is latent width; `rows` is cache length; `first` is score capacity;
/// `slots` is selection capacity. Softmax alone uses `cols` for score capacity.
pub fn lengths(s:Shape)->Option<([usize;5],usize,usize,usize)> {
    let (m,h,d,v,r,k,c,n)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize,s.first as usize,s.slots as usize);
    if m==0 || h==0 || d==0 || v==0 || r==0 || k==0 || c==0 || c>n || c>r {return None;}
    let selected=m.checked_mul(n.checked_add(1)?)?;let mh=m.checked_mul(h)?;
    if s.op==Op::AbsorbSoftmax {return (k==c).then_some(([mh.checked_mul(c)?,selected,0,0,0],0,mh,c));}
    if k%2!=0 {return None;}let weights=h.checked_mul(d.checked_add(v)?)?.checked_mul(k)?/2;
    match s.op {
        Op::AbsorbQuery=>Some(([mh.checked_mul(d)?,0,0,0,0],weights,mh.checked_mul(k)?,1)),
        Op::AbsorbScores=>Some(([mh.checked_mul(k)?,r.checked_mul(k)?,selected,0,0],0,mh.checked_mul(c)?,1)),
        Op::AbsorbValues=>Some(([mh.checked_mul(c)?,r.checked_mul(k)?,selected,0,0],0,mh.checked_mul(k)?,1)),
        Op::AbsorbProject=>Some(([mh.checked_mul(k)?,selected,0,0,0],weights,mh.checked_mul(v)?,1)),
        _=>None,
    }
}
/// The grouped kernel's tile: a block owns TILE_A x TILE_B dot products of one outer index (latent_absorption_tile).
/// A thread owns 2 A rows x 4 B rows for the whole reduction.
pub const TILE_A:u32=32;pub const TILE_B:u32=64;pub const TILE_THREADS:u32=256;
/// (outer count, A rows, B rows) of one operation. Query and project: a head, tokens, features.
/// Scores: a token, heads, selected slots. Values: a token, heads, latent features.
pub fn tile(s:Shape)->Option<(u32,u32,u32)> {
    Some(match s.op {
        Op::AbsorbQuery=>(s.heads,s.tokens,s.cols),Op::AbsorbProject=>(s.heads,s.tokens,s.taps),
        Op::AbsorbScores=>(s.tokens,s.heads,s.first),Op::AbsorbValues=>(s.tokens,s.heads,s.cols),_=>return None,
    })
}
/// B row `j` (0..4) of the thread with column index `tx` (0..16) inside its tile. Values keeps a thread's four
/// features adjacent, because one staged load holds four adjacent features of a cache row.
pub const fn tile_b_row(values:bool,tx:u32,j:u32)->u32 {if values {tx*4+j} else {tx+16*j}}
pub fn launch(s:Shape)->Option<(&'static str,u32,u32)> {
    if s.op==Op::AbsorbSoftmax {return Some(("prompt_softmax_rows",s.tokens.checked_mul(s.heads)?,128));}
    let (outer,a,b)=tile(s)?;
    Some(("latent_absorption_tile",outer.checked_mul(a.div_ceil(TILE_A))?.checked_mul(b.div_ceil(TILE_B))?,TILE_THREADS))
}
#[cfg(not(target_arch="nvptx64"))]
pub fn unit(s:Shape,index:usize,x:[&[f32];5],w:&[f32],out:&mut [f32]) {
    if s.op==Op::AbsorbSoftmax {crate::prompt_ops::unit(Shape {op:Op::SoftmaxRows,..s},index,x,w,out);return;}
    let (h,d,v,k,c,n)=(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,s.first as usize,s.slots as usize);
    let width=match s.op {Op::AbsorbScores=>c,Op::AbsorbProject=>v,_=>k};
    let token=index/(h*width);let head=index/width%h;let feature=index%width;
    let weight=|j:usize|{let bits=w[j/2].to_bits();f32::from_bits(if j.is_multiple_of(2) {bits<<16} else {bits&0xffff0000})};
    if s.op==Op::AbsorbProject && x[1][token*(n+1)+n]==0.0 {out[0]=0.0;return;}
    let selected=if matches!(s.op,Op::AbsorbScores|Op::AbsorbValues) {&x[2][token*(n+1)..(token+1)*(n+1)]} else {&[]};
    if s.op==Op::AbsorbScores && feature>=selected[n] as usize {out[0]=0.0;return;}
    let terms=match s.op {Op::AbsorbQuery=>d,Op::AbsorbValues=>selected[n] as usize,_=>k};
    let mut strips=[0.0f32;32];
    for j in 0..terms {
        let (a,b)=match s.op {
            Op::AbsorbQuery=>(x[0][(token*h+head)*d+j],weight(head*(d+v)*k+feature*d+j)),
            Op::AbsorbScores=>(x[0][(token*h+head)*k+j],x[1][selected[feature] as usize*k+j]),
            Op::AbsorbValues=>(x[0][(token*h+head)*c+j],x[1][selected[j] as usize*k+feature]),
            Op::AbsorbProject=>(x[0][(token*h+head)*k+j],weight((head*(d+v)+d+feature)*k+j)),
            _=>return,
        };let product=a*b;strips[(j/8)%32]+=product;
    }
    for distance in [16,8,4,2,1] {for lane in 0..distance {strips[lane]+=strips[lane+distance];}}
    out[0]=match s.op {Op::AbsorbScores=>strips[0]*(1.0/crate::causal_ops::root(d as f32)),_=>crate::latent_absorb::round(strips[0])};
}
