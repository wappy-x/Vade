//! Grouped latent attention. Each token keeps the word function of `absorb_tc` (the tensor-core fold), as the step form
//! does. One kernel runs the four fold steps.
use crate::prompt_ops::{Op,Shape};
#[cfg(test)] mod tests;
#[cfg(test)] mod rows_form_tests;
pub fn admitted(op:Op)->bool {matches!(op,Op::AbsorbQuery|Op::AbsorbScores|Op::AbsorbSoftmax|Op::AbsorbValues|Op::AbsorbProject|Op::AbsorbScoresPacked|Op::AbsorbValuesPacked)}
pub fn packed(op:Op)->bool {matches!(op,Op::AbsorbScoresPacked|Op::AbsorbValuesPacked)}
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
        Op::AbsorbScores|Op::AbsorbScoresPacked=>Some(([mh.checked_mul(k)?,r.checked_mul(k)?/(if packed(s.op) {2} else {1}),selected,0,0],0,mh.checked_mul(c)?,1)),
        Op::AbsorbValues|Op::AbsorbValuesPacked=>Some(([mh.checked_mul(c)?,r.checked_mul(k)?/(if packed(s.op) {2} else {1}),selected,0,0],0,mh.checked_mul(k)?,1)),
        Op::AbsorbProject=>Some(([mh.checked_mul(k)?,selected,0,0,0],weights,mh.checked_mul(v)?,1)),
        _=>None,
    }
}
/// The grouped kernel's launch: `absorb_tc::Form::blocks` blocks of 256 threads (one outer index, 128 M rows and 32 N
/// columns a block). The kernel finds its words from the block index and the shape alone. A check pass (1 to
/// `SMALL_TOKENS` tokens) takes the ROWS FORM (`rows_form`) when its shape fits it.
pub fn launch(s:Shape)->Option<(&'static str,u32,u32)> {
    if packed(s.op) {return None;}
    if s.op==Op::AbsorbSoftmax {return Some(("prompt_softmax_rows",s.tokens.checked_mul(s.heads)?,128));}
    let f=crate::absorb_tc::Form::rows(s)?;
    // A selected row is addressed by a 32-bit word offset in the kernel's staged chunk.
    if f.rows.checked_mul(f.k)?>=crate::absorb_tc::NO_ROW as usize {return None;}
    if let Some(rows)=rows_form(f) {return Some(rows);}
    let blocks=f.outer().checked_mul(f.m().div_ceil(crate::absorb_tc::M_BLOCK))?.checked_mul(f.n().div_ceil(crate::absorb_tc::N_BLOCK))?;
    Some(("latent_absorption_tile",u32::try_from(blocks).ok()?,crate::absorb_tc::THREADS))
}
/// THE ROWS FORM of the grouped kernel (`kernels/src/latent_rows.rs`): the word of `latent_absorption_tile`, bit for bit
/// (device proof: `kernel-bench --latent-rows`, every output word of the four operations against the tile of the same
/// image), on the bodies of the one-row step. Scores and values of each token run the step form's attention body (the four
/// ranges of a sum side by side, `Form::step_blocks` blocks); query and project run the tile's small body in an entry that
/// holds only the `ROWS_DENSE_STAGE` shared bytes it stages. Shared bytes of the rows form's query and project: every
/// step of B at once (`SMALL_STEP_BYTES` a step), 32 steps (512 terms).
pub const ROWS_DENSE_STAGE:usize=32*crate::absorb_tc::SMALL_STEP_BYTES;
/// (kernel, blocks, threads) of the rows form for this launch, or None: keep the tile. `f` is an unpacked operation of
/// 1 to `SMALL_TOKENS` tokens whose selected rows fit 32-bit word offsets (checked by `launch`).
pub fn rows_form(f:crate::absorb_tc::Form)->Option<(&'static str,u32,u32)> {
    use crate::absorb_tc::{SMALL_STEP_BYTES,SMALL_TOKENS,THREADS,steps};
    if f.packed || f.tokens==0 || f.tokens>SMALL_TOKENS || f.table==0 {return None;}
    if f.attention() {return Some(("latent_rows_attention",u32::try_from(f.step_blocks()).ok()?,THREADS));}
    if steps(f.terms(0)).checked_mul(SMALL_STEP_BYTES)?>ROWS_DENSE_STAGE {return None;}
    Some(("latent_rows_dense",u32::try_from(f.blocks()).ok()?,THREADS))
}
/// THE THIRD FORM of the grouped kernel (`kernels/src/latent_tile3.rs`): the word of `latent_absorption_tile`, bit for
/// bit (device proof: `kernel-bench --latent3`, every output word of the four operations against the first form of the
/// same image). Only the way a body gets its operands differs: the B operand is staged once with 16-byte loads, the
/// A operand is read at a register plus a constant one step ahead, and for values the two operands of the
/// tensor-core step change sides so that a lane reads a cache row with one 16-byte load.
/// Token rows from which the third form is launched (below it the first form stays).
pub const THIRD_MIN_TOKENS:usize=17;
pub const THIRD_THREADS:u32=256;
/// (kernel, blocks, threads) of the third form for this launch, or None: keep `launch`. The kernels take the model's
/// shape only (32 heads, 512 latent features, 256 key features, value rows in whole tiles of 16) and 16-byte aligned
/// operands, and trap on anything else, so every condition they check is checked here first. `inputs`, `weight` and
/// `output` are the device addresses of the launch.
pub fn third(s:Shape,inputs:[u64;5],weight:u64,output:u64)->Option<(&'static str,u32,u32)> {
    use crate::absorb_tc::{Form,Kind};
    if !admitted(s.op) || s.op==Op::AbsorbSoftmax {return None;}
    let f=Form::rows(s)?;
    if f.tokens<THIRD_MIN_TOKENS || f.h!=32 || f.k!=512 || f.table==0 {return None;}
    // Values address a selected row by a 32-bit byte offset (2,048 bytes a row).
    if f.rows.checked_mul(2048)?>=crate::absorb_tc::NO_ROW as usize {return None;}
    let tokens=u32::try_from(f.tokens).ok()?;
    match f.kind {
        Kind::Scores if f.packed && (inputs[0]|inputs[1])&15==0=>Some(("latent_scores3_packed",tokens,384)),
        Kind::Values if f.packed && (inputs[1]|output)&15==0=>Some(("latent_values3_full_park",tokens,THIRD_THREADS)),
        Kind::Scores if (inputs[0]|inputs[1])&15==0=>Some(("latent_scores3",tokens,THIRD_THREADS)),
        Kind::Values if (inputs[1]|output)&15==0=>Some(("latent_values3",tokens.checked_mul(4)?,THIRD_THREADS)),
        Kind::Query if f.d==256 && (inputs[0]|weight)&15==0=>Some(("latent_weights3",32*tokens.div_ceil(32),THIRD_THREADS)),
        Kind::Project if f.v.is_multiple_of(16) && (inputs[0]|weight)&15==0=>Some(("latent_weights3_panel",32*tokens.div_ceil(32),THIRD_THREADS)),
        _=>None,
    }
}
#[cfg(not(target_arch="nvptx64"))]
pub fn unit(s:Shape,index:usize,x:[&[f32];5],w:&[f32],out:&mut [f32]) {
    if s.op==Op::AbsorbSoftmax {crate::prompt_ops::unit(Shape {op:Op::SoftmaxRows,..s},index,x,w,out);return;}
    use crate::absorb_tc::{Form,Kind,word};
    let Some(f)=Form::rows(s) else {return;};
    // An output unit is [token][head][feature]; the word function names it by its outer index and its place in the product.
    let width=f.m();let (token,head,feature)=(index/(f.h*width),index/width%f.h,index%width);
    let (outer,n)=if f.attention() {(token,head)} else {(head,token)};
    let (latent,selected):(&[f32],&[f32])=match f.kind {Kind::Scores|Kind::Values=>(x[1],x[2]),Kind::Project=>(&[],x[1]),Kind::Query=>(&[],&[])};
    out[0]=word(f,outer,feature,n,x[0],latent,selected,w);
}
