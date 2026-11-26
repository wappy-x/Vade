//! The host check of every operand of one launch of the Inkling block arms (entries `resident_launch_block_arms` and
//! `resident_launch_block_rows`), as one pure function with a test for each refusal (Ruling R6h, rule 3). The launch
//! (`Backend::resident_causal_into`) calls it before it records or uploads a command; a refusal is an error, never a
//! launch.
use crate::block_ops::Lengths;
/// One device buffer as the check sees it: its first byte, its words, and two facts the backend knows.
#[derive(Clone,Copy,Debug)]
pub struct Operand {pub address:u64,pub words:usize,pub contiguous:bool,pub own_context:bool}
/// The words a launch reads and writes: the counts of the arm (`block_ops::checked_lengths`) or of its rows
/// (`block_rows::lengths`). Each input and the weight must hold exactly its count (none when the count is 0), the
/// output exactly units x words; every buffer contiguous, of this context, 4-byte aligned; the output not empty and
/// overlapping no input and not the weight. `capturing`: a stream capture runs, where a launch table could not live
/// until a replay, so it is refused (`Backend::arm_capture`: every capture of `capture_with_allowance` and of
/// `verify_segment`; device test `an_arm_inside_each_graph_capture_is_refused`).
pub fn operands(counts:Option<Lengths>,inputs:[Option<Operand>;4],weight:Option<Operand>,output:Operand,capturing:bool)->Result<Lengths,String> {
    if capturing {return Err("a block arm is never launched inside a stream capture".into());}
    let (lengths,weights,units,words)=counts.ok_or("invalid block arm shape")?;
    let all=units.checked_mul(words).ok_or("block arm output overflow")?;
    if all==0 {return Err("a block arm launch writes no word".into());}
    let fits=|operand:Option<Operand>,count:usize,what:&str|->Result<(),String> {match operand {
        None if count==0=>Ok(()),
        Some(o) if count>0 && o.words==count=>if !o.contiguous {Err(format!("block arm {what} is not contiguous"))} else if !o.own_context {Err(format!("block arm {what} belongs to another context"))}
            else if o.address==0 {Err(format!("block arm {what} has no address"))}
            else if o.address%4!=0 {Err(format!("block arm {what} is not aligned"))}
            else if end_of(o).is_none() {Err(format!("block arm {what} ends past the address space"))} else {Ok(())},
        _=>Err(format!("block arm {what} count differs")),
    }};
    for (n,(operand,count)) in inputs.into_iter().zip(lengths).enumerate() {fits(operand,count,["input 0","input 1","input 2","input 3"][n])?;}
    fits(weight,weights,"weight")?;fits(Some(output),all,"output")?;
    // Every operand passed `fits`, so its end is known (review REVARM1 finding 3: no unchecked sum).
    let span=|o:Operand|end_of(o).map(|end|(o.address,end)).ok_or("block arm operand ends past the address space");
    let (start,end)=span(output)?;
    for operand in inputs.into_iter().flatten().chain(weight) {let (a,b)=span(operand)?;if start<b && a<end {return Err("block arm output overlaps an operand".into());}}
    Ok((lengths,weights,units,words))
}
/// The first byte after an operand, or None when its bytes would pass the end of the address space.
fn end_of(o:Operand)->Option<u64> {(o.words as u64).checked_mul(4).and_then(|bytes|o.address.checked_add(bytes))
}
#[cfg(test)] mod tests {
    use super::*;
    use crate::causal_ops::{Op,Shape};
    const ROW:usize=64;
    fn norm()->Shape {Shape {op:Op::RoundedNorm,heads:2,dim:ROW as u32,taps:1,rows:0,cols:0,epsilon:1.0e-6,bound:0.0}}
    fn at(address:u64,words:usize)->Operand {Operand {address,words,contiguous:true,own_context:true}}
    fn check(input:Option<Operand>,weight:Option<Operand>,output:Operand)->Result<Lengths,String> {operands(norm().lengths(),[input,None,None,None],weight,output,false)}
    #[test] fn a_good_launch_passes_and_gives_the_counts() {
        assert_eq!(check(Some(at(0x1000,2*ROW)),Some(at(0x2000,ROW)),at(0x3000,2*ROW)),Ok(norm().lengths().unwrap()));
        let rows=crate::block_rows::lengths(norm(),3,0);
        assert!(operands(rows,[Some(at(0x1000,6*ROW)),None,None,None],Some(at(0x2000,ROW)),at(0x4000,6*ROW),false).is_ok());
    }
    #[test] fn each_bad_operand_is_refused() {
        let (x,w,y)=(at(0x1000,2*ROW),at(0x2000,ROW),at(0x3000,2*ROW));
        let refused=|r:Result<Lengths,String>,text:&str|assert!(r.as_ref().is_err_and(|e|e.contains(text)),"{r:?} wants {text}");
        // Bad extents: one word short or long, an operand the arm does not read, a missing one.
        refused(check(Some(at(0x1000,2*ROW-1)),Some(w),y),"input 0 count differs");
        refused(check(Some(x),Some(at(0x2000,ROW+1)),y),"weight count differs");
        refused(check(Some(x),Some(w),at(0x3000,2*ROW+1)),"output count differs");
        refused(check(None,Some(w),y),"input 0 count differs");
        refused(operands(norm().lengths(),[Some(x),Some(at(0x5000,1)),None,None],Some(w),y,false),"input 1 count differs");
        // Zero sizes: a shape with no rows has no counts; an output of no word.
        refused(operands(Shape {heads:0,..norm()}.lengths(),[Some(x),None,None,None],Some(w),y,false),"invalid block arm shape");
        refused(operands(Some(([0,0,0,0],0,0,1)),[None;4],None,at(0x3000,0),false),"writes no word");
        // A buffer of another kind: segmented, of another context, not aligned.
        refused(check(Some(Operand {contiguous:false,..x}),Some(w),y),"input 0 is not contiguous");
        refused(check(Some(x),Some(Operand {own_context:false,..w}),y),"weight belongs to another context");
        refused(check(Some(x),Some(w),Operand {address:0x3002,..y}),"output is not aligned");
        // Overlap of the output with an input or the weight.
        refused(check(Some(x),Some(w),at(0x1000+4*ROW as u64,2*ROW)),"overlaps");
        refused(check(Some(x),Some(w),at(0x2000-4,2*ROW)),"overlaps");
        // Inside a stream capture (the finding of review REVUNSAFE, HIGH body 1).
        refused(operands(norm().lengths(),[Some(x),None,None,None],Some(w),y,true),"stream capture");
        // A shape that is no arm has no arm counts.
        refused(operands(crate::block_ops::checked_lengths(Shape {op:Op::Project,..norm()}),[Some(x),None,None,None],Some(w),y,false),"invalid block arm shape");
        // Review REVARM1 finding 3: an operand at address 0, one whose bytes pass the end of the address space, and
        // counts whose output words overflow.
        refused(check(Some(at(0,2*ROW)),Some(w),y),"input 0 has no address");
        refused(check(Some(x),Some(w),at(0,2*ROW)),"output has no address");
        refused(check(Some(x),Some(at(u64::MAX-3,ROW)),y),"weight ends past the address space");
        refused(check(Some(x),Some(w),at(u64::MAX-4*ROW as u64+1,2*ROW)),"output ends past the address space");
        refused(operands(Some(([0,0,0,0],0,usize::MAX,2)),[None;4],None,y,false),"output overflow");
        // An operand that ends one word before the end of the address space passes.
        assert!(check(Some(x),Some(w),at(u64::MAX-8*ROW as u64-3,2*ROW)).is_ok());
    }
    /// The shared-sum norms of read arm 1 at the row counts around `WHOLE_ROWS` and at 1,024 rows, width 2,304: the
    /// counts pass with exact extents and are refused one word off (review REVARM1 finding 3).
    #[test] fn the_norms_of_many_rows_pass_with_their_exact_counts() {
        for heads in [63u32,64,65,1024] {
            let s=Shape {heads,dim:2304,..norm()};let n=heads as usize*2304;
            let (x,w,y)=(at(0x10_0000,n),at(0x100_0000,2304),at(0x200_0000,n));
            assert_eq!(operands(s.lengths(),[Some(x),None,None,None],Some(w),y,false),Ok(s.lengths().unwrap()),"{heads} rows");
            assert!(operands(s.lengths(),[Some(at(0x10_0000,n-1)),None,None,None],Some(w),y,false).is_err(),"{heads} rows");
            assert!(operands(s.lengths(),[Some(x),None,None,None],Some(w),at(0x200_0000,n+1),false).is_err(),"{heads} rows");
        }
    }
}
