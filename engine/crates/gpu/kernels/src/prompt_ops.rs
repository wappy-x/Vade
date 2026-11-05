//! Layer-major prompt rows. Each thread owns its own output words.
use crate::prompt_ops::{Arguments,Op,StepStore,step_column,thread_unit,unit};
/// One value column inside the StepRows output. Adjacent lanes touch adjacent
/// words, so one warp shares cache lines.
struct Column {out:*mut f32,base:usize,stride:usize,outputs:usize,step:usize}
impl StepStore for Column {
    #[inline(always)]
    fn get(&self,key:usize)->f32 {
        // SAFETY: `base+key*stride` stays in this thread's own state column. See prompt_rows.
        unsafe {*self.out.add(self.base+key*self.stride)}
    }
    #[inline(always)]
    fn set(&mut self,key:usize,value:f32) {
        // SAFETY: Only this thread touches this column. See prompt_rows.
        unsafe {*self.out.add(self.base+key*self.stride)=value;}
    }
    #[inline(always)]
    fn output(&mut self,token:usize,value:f32) {
        // SAFETY: Only this thread touches this column's output word. See prompt_rows.
        unsafe {*self.out.add(self.outputs+token*self.step)=value;}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn prompt_rows(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents
    // here, with same-context contiguous owners that live through stream
    // completion. Inputs are immutable. The private output overlaps no input and
    // is initialized earlier on the stream. thread_unit is a bijection on units,
    // so this guarded thread creates only its own disjoint output unit. A StepRows
    // thread owns one value column of the state and of the outputs, by raw offsets
    // inside the checked output extent. Empty inputs use aligned dangling addresses.
    unsafe {
        let thread=crate::index();if thread as u64>=p.units {return;}let words=p.words as usize;
        let x=core::array::from_fn(|n|core::slice::from_raw_parts(p.input[n] as *const f32,p.lengths[n] as usize));
        if p.shape.op==Op::StepRows {
            let (d,width)=(p.shape.dim as usize,p.shape.heads as usize*p.shape.dim as usize);let head=thread/d;let value=thread%d;
            let mut store=Column {out:p.output as *mut f32,base:head*d*d+value,stride:d,outputs:width*d+head*d+value,step:width};
            step_column(p.shape,thread,x,&mut store);return;
        }
        let index=thread_unit(p.shape,thread);
        let weight=core::slice::from_raw_parts(p.weight as *const f32,p.weights as usize);
        let output=core::slice::from_raw_parts_mut((p.output as *mut f32).add(index*words),words);
        unit(p.shape,index,x,weight,output);
    }
}
