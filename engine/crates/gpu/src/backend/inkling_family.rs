//! The Inkling kernel family: the third image (`sm_121a`, crate `spark-inkling-kernels`, Ruling R6h), loaded when one
//! of its entries is first launched or at a preload. Its entries are the Inkling 2 Mini block arms and the packed product; no other model
//! launches them, and the general image does not hold them.
use super::*;

/// Entries that live in the Inkling image: the kernel names of its kinds (host test).
pub(crate) fn in_inkling_image(name:&str)->bool {matches!(name,"resident_launch_block_arms"|"resident_launch_block_rows"|"resident_launch_plain_staged"|"resident_launch_plain_staged_rows"|"resident_launch_plain_staged_narrow"|"resident_launch_plain_packed"|"resident_launch_plain_packed_group"|"resident_launch_plain_packed_picked"|"resident_launch_block_scores"|"resident_launch_block_softmax")}

/// The entry of a rows launch of a block arm: `WindowScores` with its key plane shared takes its own entry
/// (`block_scores::stage`), `SinkSoftmax` with its scores per row its own (`block_attention::softmax_rows_stage`);
/// every other rows launch takes the rows entry. One choice for the kernel check of
/// `resident_causal_into` and for `block_arm` (review REVSCORES, LOW 2).
pub(super) fn rows_kind(shape:crate::causal_ops::Shape,count:usize,shared:u32)->crate::resident_launch::Kind {
    use crate::resident_launch::Kind;
    if crate::block_scores::stage(shape,count,shared).is_some() {Kind::BlockScores}
    else if crate::block_attention::softmax_rows_stage(shape,count,shared).is_some() {Kind::BlockSoftmax}
    else {Kind::BlockRows}
}

/// The backend's state for the Inkling image.
pub(super) struct InklingState<'a> {
    /// The image, loaded when one of its entries is first launched.
    module:std::cell::OnceCell<Module<'a>>,
    /// True while `capture_with_allowance` captures the stream (`capture_batch`, `capture_batch_bounded`,
    /// `capture_decode`). The general flag `context.capturing` is set only by `verify_segment`, and its meaning for the
    /// other families stays as it is; the arms read both (`arm_capture`).
    capturing:Cell<bool>,
}
impl InklingState<'_> {
    pub(super) fn new()->Self {Self {module:std::cell::OnceCell::new(),capturing:Cell::new(false)}}
}
impl<'a> Backend<'a> {
    pub(super) fn inkling_module(&self)->Result<&Module<'a>,String> {
        if let Some(module)=self.inkling.module.get() {return Ok(module);}
        let ptx=crate::inkling_ptx().ok_or("this build has no Inkling image: run `cargo kernels-inkling` before the host build")?;
        let module=Module::load_second(self.context,ptx)?;
        Ok(self.inkling.module.get_or_init(||module))
    }
    /// True when this build carries the Inkling image.
    pub fn inkling_image_present(&self)->bool {crate::inkling_ptx().is_some()}
    /// Loads the Inkling image now when this build carries it, so the first arm launch after a start does not pay for
    /// its PTX compile. The same module and entries: no word changes.
    pub fn inkling_preload(&self)->Result<(),String> {if self.inkling_image_present() {self.inkling_module()?;} Ok(())}
    /// Marks a graph capture of `capture_with_allowance` (`on`) or its end; gives the mark before, to put back.
    pub(super) fn mark_graph_capture(&self,on:bool)->bool {self.inkling.capturing.replace(on)}
    /// True when any stream capture runs: a block arm is then refused, since its one-command table could not live
    /// until a replay (review REVIMAGE finding 1).
    pub(super) fn arm_capture(&self)->bool {self.inkling.capturing.get() || self.context.capturing.get()}
    /// One block-arm operation of `resident_causal_into` (Ruling R6h rule 1: the whole launch lives in this Inkling
    /// host file). Here, in this one function: every operand is checked (`block_check::operands`: exact counts of the
    /// arm or of its rows, this context, contiguous, aligned, the output disjoint, no capture), the command is built
    /// from the checked addresses, and then it is recorded, or uploaded as a table of one aligned Command and launched
    /// at once. Returns that table (None when recorded): the caller keeps it through the completion it waits for.
    /// The unsafe block below rests only on what this function checked (review REVIMAGE-2 finding 1).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_arm(&self,shape:crate::causal_ops::Shape,row_launch:Option<(usize,u32)>,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,output:&DeviceVector<'_>,units:usize,args:crate::causal_ops::Arguments)->Result<Option<DeviceVector<'a>>,String> {
        use crate::resident_launch::Kind;
        let operand=|v:&DeviceVector<'_>|crate::block_check::Operand {address:v.device_address(),words:v.len(),contiguous:!v.is_segmented(),own_context:self.check_vector(v,v.len()).is_ok()};
        let counts=match row_launch {Some((count,shared))=>crate::block_rows::lengths(shape,count,shared),None=>crate::block_ops::checked_lengths(shape)};
        let (_,_,checked_units,_)=crate::block_check::operands(counts,input.map(|v|v.map(operand)),weight.map(operand),operand(output),self.arm_capture())?;
        if checked_units!=units {return Err("block arm units differ from the checked counts".into());}
        let addresses=input.map(|v|v.map_or(4,|v|v.device_address()));
        // Every field of the shape in the command is the checked shape (review REVPACK finding 3), the floats by their bits.
        let same_shape=|a:crate::causal_ops::Shape,b:crate::causal_ops::Shape|a.op==b.op && a.heads==b.heads && a.dim==b.dim && a.taps==b.taps && a.rows==b.rows && a.cols==b.cols && a.epsilon.to_bits()==b.epsilon.to_bits() && a.bound.to_bits()==b.bound.to_bits();
        if args.input!=addresses || args.weight!=weight.map_or(4,|v|v.device_address()) || args.output!=output.device_address() || !same_shape(args.shape,shape) {return Err("block arm command differs from its checked operands".into());}
        let mut command=crate::resident_launch_encode::causal(args);
        let (kind,(blocks,threads))=match row_launch {
            Some((count,shared))=>{
                command.put(crate::block_ops::device::ROWS_AT,&(count as u32).to_ne_bytes());command.put(crate::block_ops::device::SHARED_AT,&shared.to_ne_bytes());
                // WindowScores at rows with its key plane shared (the bias shared or each row's own), and SinkSoftmax at
                // rows with its scores per row: each its own entry (rows_kind, block_scores::stage, softmax_rows_stage).
                match (rows_kind(shape,count,shared),crate::block_scores::stage(shape,count,shared),crate::block_attention::softmax_rows_stage(shape,count,shared)) {
                    (Kind::BlockScores,Some((blocks,threads)),_)=>(Kind::BlockScores,(u32::try_from(blocks).map_err(|_|"block scores grid overflow")?,threads as u32)),
                    (Kind::BlockSoftmax,_,Some(blocks))=>(Kind::BlockSoftmax,(u32::try_from(blocks).map_err(|_|"block softmax grid overflow")?,crate::block_ops::SUM_THREADS as u32)),
                    _=>(Kind::BlockRows,(u32::try_from(units.div_ceil(128)).map_err(|_|"block rows grid overflow")?,128)),
                }
            },
            None=>(Kind::BlockArms,crate::block_ops::geometry(shape,units)),
        };
        command.kind=kind as u32;
        if self.record_resident_block(kind,command,[blocks,1,1,threads,0])? {return Ok(None);}
        let mut words=vec![f32::from_bits(command.kind),0.0,0.0,0.0];
        words.extend(command.payload.as_chunks::<4>().0.iter().map(|b|f32::from_bits(u32::from_ne_bytes(*b))));
        let table=self.upload_vector(&words)?;
        let name=kind.kernel();
        if !in_inkling_image(name) || table.len()!=crate::inkling_launch::COMMAND_WORDS || table.device_address()%16!=0 {return Err("block arm command table is not one aligned command of an Inkling entry".into());}
        let (mut address,mut index)=(table.device_address(),0u64);
        // SAFETY: every fact the entry's `run` relies on is checked above in this function: `name` is an entry of the
        // Inkling image; `table` is a live private vector of this context holding exactly one initialized, 16-byte
        // aligned Command of that entry's kind, built from `args`, whose addresses are those of the operands that
        // block_check::operands proved (exact counts of the arm or of its rows, this context, contiguous, aligned, the
        // output disjoint, no capture); the grid is the one block_ops::geometry or the rows launch gives for those
        // counts. The operands are borrowed by the caller and the table is returned to it, both through completion.
        unsafe {self.launch_flash(name,blocks,threads,&mut [parameter(&mut address),parameter(&mut index)])}?;
        Ok(Some(table))
    }
    /// The module that holds the entry `name`: the tensor-core image, the Inkling image, or the general image.
    pub(super) fn module_for(&self,name:&str)->Result<&Module<'a>,String> {
        if tc_family::in_tc_image(name) {self.tc_module()} else if in_inkling_image(name) {self.inkling_module()} else {Ok(&self.module)}
    }
}
#[cfg(test)] mod tests {
    use crate::{inkling_launch::{BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_SOFTMAX},resident_launch::Kind};
    #[test] fn the_inkling_entries_are_the_kernels_of_its_kinds() {
        for kind in Kind::ALL {
            assert_eq!(super::in_inkling_image(kind.kernel()),[BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_SOFTMAX].contains(&(kind as u32)),"{}",kind.kernel());
        }
        assert!(!super::in_inkling_image("causal_operation") && !super::in_inkling_image("resident_launch_causal"));
    }
    /// The one choice of the rows entry (review REVSOFTMAX, LOW 1): WindowScores with its key plane shared takes
    /// kind 97, SinkSoftmax with its scores per row kind 98, every other rows launch the rows entry.
    #[test] fn a_rows_launch_takes_the_entry_of_its_stage() {
        use crate::causal_ops::{Op,Shape};
        let softmax=Shape {op:Op::SinkSoftmax,heads:24,dim:0,taps:0,rows:8192,cols:0,epsilon:0.0,bound:0.0};
        let scores=Shape {op:Op::WindowScores,heads:24,dim:128,taps:3,rows:1024,cols:1024,epsilon:0.0,bound:0.088_388_35};
        assert_eq!(super::rows_kind(scores,1024,0b0010),Kind::BlockScores);
        assert_eq!(super::rows_kind(scores,1024,0),Kind::BlockRows);
        assert_eq!(super::rows_kind(softmax,1024,0),Kind::BlockSoftmax);
        assert_eq!(super::rows_kind(softmax,1024,0b0110),Kind::BlockSoftmax);
        // Shared scores, a launch above the rows limit, another arm: the rows entry.
        assert_eq!(super::rows_kind(softmax,1024,0b0001),Kind::BlockRows);
        assert_eq!(super::rows_kind(softmax,crate::block_rows::ROWS_MAX+1,0),Kind::BlockRows);
        assert_eq!(super::rows_kind(Shape {op:Op::WindowValues,..scores},1024,0b0010),Kind::BlockRows);
    }
    /// Review REVIMAGE finding 1: an arm launched inside each graph capture is refused with an error, no launch, and
    /// the same arm launches after the capture.
    #[test]
    #[ignore = "skipped: no GPU; requires CUDA and the Inkling image"]
    fn an_arm_inside_each_graph_capture_is_refused() {
        use crate::causal_ops::{Op,Shape};
        crate::driver::with_backend(0,crate::layer_ptx().unwrap(),usize::MAX,|b| {
            let norm=Shape {op:Op::PlainNorm,heads:1,dim:64,taps:0,rows:0,cols:0,epsilon:1.0e-6,bound:0.0};
            let x=b.upload_vector(&[0.5f32;64])?;let position=b.upload_vector(&[0.0f32;4])?;let y=std::cell::RefCell::new(b.vector(64)?);
            let arm=||b.resident_causal_into(norm,[Some(&x),None,None,None],None,&mut y.borrow_mut());
            let refused=|graph:Result<crate::driver::DecodeGraph<'_>,String>,name:&str|->Result<(),String> {
                match graph {Err(e) if e.contains("stream capture")=>Ok(()),Err(e)=>Err(format!("{name}: other error {e}")),Ok(_)=>Err(format!("{name}: the arm was captured"))}
            };
            // SAFETY: each capture records nothing (the arm is refused before any launch, so the capture fails or is
            // empty) and every buffer the closure names (x, y, position) lives to the end of this test.
            let graphs=unsafe {[b.capture_batch(1,1,arm),b.capture_batch_bounded(64,arm),b.capture_decode(&position,0,1,1,arm)]};
            for (graph,name) in graphs.into_iter().zip(["capture_batch","capture_batch_bounded","capture_decode"]) {refused(graph,name)?;}
            if b.arm_capture() {return Err("the capture mark stayed set".into());}
            arm()
        }).unwrap();
    }
}
