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
    // completion. Inputs are immutable. The private output overlaps no input. It is zeroed earlier on the stream for every op
    // except a Transpose of a swept shape (prompt_writes), whose output may be unwritten: that branch makes its one word in a local and
    // stores it through the raw output pointer, and no reference over the output is built before the store. thread_unit is a bijection on units,
    // so this guarded thread creates only its own disjoint output unit. A StepRows
    // thread owns one value column of the state and of the outputs, by raw offsets
    // inside the checked output extent. A ScoresRows thread owns one output word and a
    // ValuesRows thread owns 8 words of one unit. Empty inputs use aligned dangling
    // addresses. ScoresRows and ValuesRows read counts and row IDs from their selected
    // input, so Shape alone does not bound those reads: each is checked against its
    // extent before use and a bad one traps, so every raw read stays inside the inputs.
    // A ScoresRows table entry is data too: it must name a live slot of its token and that
    // slot's row, or it traps, so entries from one correct table own distinct output words.
    unsafe {
        let thread=crate::index();if thread as u64>=p.units {return;}let words=p.words as usize;
        let x=core::array::from_fn(|n|core::slice::from_raw_parts(p.input[n] as *const f32,p.lengths[n] as usize));
        if p.shape.op==Op::StepRows {
            let (d,width)=(p.shape.dim as usize,p.shape.heads as usize*p.shape.dim as usize);let head=thread/d;let value=thread%d;
            let mut store=Column {out:p.output as *mut f32,base:head*d*d+value,stride:d,outputs:width*d+head*d+value,step:width};
            step_column(p.shape,thread,x,&mut store);return;
        }
        // Plain rows: each output keeps its ascending sum, with four terms read per line.
        if p.shape.op==Op::ProjectRows && p.shape.cols%4==0 && p.shape.cols>0 && (p.input[0]|p.weight)&15==0 {
            use core::arch::asm;
            macro_rules! load16 {($pointer:expr)=>{{
                let (first,second,third,fourth):(u32,u32,u32,u32);
                asm!("ld.global.v4.u32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,third=out(reg32)third,fourth=out(reg32)fourth,pointer=in(reg64)$pointer,options(nostack,readonly));
                [f32::from_bits(first),f32::from_bits(second),f32::from_bits(third),f32::from_bits(fourth)]
            }};}
            let s=p.shape;let (m,k)=(s.tokens as usize,s.cols as usize);let (row,token)=(thread/m,thread%m);
            let input=(p.input[0] as *const f32).add(token*k);let weight=(p.weight as *const f32).add(row*k);
            let mut sum=0.0f32;let mut j=0;
            while j<k {
                let (a,b)=(load16!(input.add(j).cast::<u8>()),load16!(weight.add(j).cast::<u8>()));
                sum+=a[0]*b[0];sum+=a[1]*b[1];sum+=a[2]*b[2];sum+=a[3]*b[3];j+=4;
            }
            *(p.output as *mut f32).add(thread*words)=sum;return;
        }
        // One chain for each token, head and slot. A warp holds adjacent tokens of one slot and
        // head, so its lanes share that key's lines. Each slot keeps its own ascending sum.
        if p.shape.score_lines() && (p.input[0]|p.input[1])&15==0 {
            use core::arch::asm;
            macro_rules! load16 {($pointer:expr)=>{{
                let (first,second,third,fourth):(u32,u32,u32,u32);
                asm!("ld.global.v4.u32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,third=out(reg32)third,fourth=out(reg32)fourth,pointer=in(reg64)$pointer,options(nostack,readonly));
                [f32::from_bits(first),f32::from_bits(second),f32::from_bits(third),f32::from_bits(fourth)]
            }};}
            let s=p.shape;let (m,h,d,t,k,slots)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,s.slots as usize);
            // With a table sorted by row, a warp holds tokens that read the same key. Counts, row
            // IDs and table entries are data. A bad one traps, as an indexed read does.
            let entries=s.first as usize;
            let (token,head,slot,listed)=if entries>0 {let entry=(p.input[3] as *const f32).add((thread%entries)*3);
                (*entry.add(1) as usize,thread/entries,*entry.add(2) as usize,Some(*entry as usize))} else {(thread%m,(thread/m)%h,thread/(m*h),None)};
            if token>=m || head>=h {panic!();}
            let selected=(p.input[2] as *const f32).add(token*(slots+1));
            let count=*selected.add(slots) as usize;if count>k.min(slots) {panic!();}
            if slot>=count {if listed.is_some() {panic!();}return;}
            let id=*selected.add(slot) as usize;if id>=s.rows as usize || listed.is_some_and(|row|row!=id) {panic!();}
            let index=token*h+head;
            let query=(p.input[0] as *const f32).add(index*d);
            let mut sum=0.0f32;let mut j=0;
            if s.op==Op::ScoresRowsPacked {
                // Eight features a line: four packed words, each half widened to the f32 the unpacked row held.
                // score_lines admits packed shapes only with dim a multiple of 8, so every load is whole.
                let key=(p.input[1] as *const u8).add((id*h*(d+t)/2+head*(d+t)/2)*4);
                while j<d {
                    let (a,b,c)=(load16!(query.add(j).cast::<u8>()),load16!(query.add(j+4).cast::<u8>()),load16!(key.add(j*2)));
                    sum+=a[0]*f32::from_bits(c[0].to_bits()<<16);sum+=a[1]*f32::from_bits(c[0].to_bits()&0xffff_0000);
                    sum+=a[2]*f32::from_bits(c[1].to_bits()<<16);sum+=a[3]*f32::from_bits(c[1].to_bits()&0xffff_0000);
                    sum+=b[0]*f32::from_bits(c[2].to_bits()<<16);sum+=b[1]*f32::from_bits(c[2].to_bits()&0xffff_0000);
                    sum+=b[2]*f32::from_bits(c[3].to_bits()<<16);sum+=b[3]*f32::from_bits(c[3].to_bits()&0xffff_0000);j+=8;
                }
            } else {
                let key=(p.input[1] as *const f32).add(id*h*(d+t)+head*(d+t));
                while j<d {
                    let (a,b)=(load16!(query.add(j).cast::<u8>()),load16!(key.add(j).cast::<u8>()));
                    sum+=a[0]*b[0];sum+=a[1]*b[1];sum+=a[2]*b[2];sum+=a[3]*b[3];j+=4;
                }
            }
            *(p.output as *mut f32).add(index*words+slot)=crate::prompt_ops::round(crate::prompt_ops::round(sum)*(1.0/crate::prompt_ops::root(d as f32)));
            return;
        }
        // Eight channels stay in registers and a warp reads one head's value row by the line.
        // Each channel still adds its slots in ascending order, and rounds once at the end.
        let (unit_thread,block)=if let Some(blocks)=p.shape.value_blocks() {(thread/blocks,thread%blocks)} else {(thread,0)};
        if p.shape.value_blocks().is_some() && p.input[1]&15==0 {
            use core::arch::asm;
            macro_rules! load16 {($pointer:expr)=>{{
                let (first,second,third,fourth):(u32,u32,u32,u32);
                asm!("ld.global.v4.u32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,third=out(reg32)third,fourth=out(reg32)fourth,pointer=in(reg64)$pointer,options(nostack,readonly));
                [f32::from_bits(first),f32::from_bits(second),f32::from_bits(third),f32::from_bits(fourth)]
            }};}
            let s=p.shape;let (h,d,t,k,slots)=(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,s.slots as usize);
            let index=thread_unit(s,unit_thread);let (token,head)=(index/h,index%h);let row=h*(d+t);
            let selected=(p.input[2] as *const f32).add(token*(slots+1));let count=*selected.add(slots) as usize;
            // Counts and row IDs are data. A bad one traps, as an indexed read does.
            if count>k.min(slots) {panic!();}
            let weights=(p.input[0] as *const f32).add(index*k);let values=p.input[1] as *const f32;let offset=head*(d+t)+d+block*8;
            let (mut low,mut high)=([0.0f32;4],[0.0f32;4]);let mut slot=0;
            while slot<count {
                let id=*selected.add(slot) as usize;if id>=s.rows as usize {panic!();}
                let weight=*weights.add(slot);
                if s.op==Op::ValuesRowsPacked {
                    // One 16-byte line holds all eight channels packed; each half widens to its unpacked f32.
                    let c=load16!((values as *const u8).add((id*row/2+offset/2)*4));
                    low[0]+=weight*f32::from_bits(c[0].to_bits()<<16);low[1]+=weight*f32::from_bits(c[0].to_bits()&0xffff_0000);
                    low[2]+=weight*f32::from_bits(c[1].to_bits()<<16);low[3]+=weight*f32::from_bits(c[1].to_bits()&0xffff_0000);
                    high[0]+=weight*f32::from_bits(c[2].to_bits()<<16);high[1]+=weight*f32::from_bits(c[2].to_bits()&0xffff_0000);
                    high[2]+=weight*f32::from_bits(c[3].to_bits()<<16);high[3]+=weight*f32::from_bits(c[3].to_bits()&0xffff_0000);
                } else {
                    let line=values.add(id*row+offset);
                    let (a,b)=(load16!(line.cast::<u8>()),load16!(line.add(4).cast::<u8>()));
                    low[0]+=weight*a[0];low[1]+=weight*a[1];low[2]+=weight*a[2];low[3]+=weight*a[3];
                    high[0]+=weight*b[0];high[1]+=weight*b[1];high[2]+=weight*b[2];high[3]+=weight*b[3];
                }
                slot+=1;
            }
            let output=(p.output as *mut f32).add(index*words+block*8);
            let mut j=0;while j<4 {*output.add(j)=crate::prompt_ops::round(low[j]);*output.add(4+j)=crate::prompt_ops::round(high[j]);j+=1;}
            return;
        }
        if block!=0 || (p.shape.score_lines() && thread/(p.shape.tokens as usize*p.shape.heads as usize)!=0) {return;}
        let index=thread_unit(p.shape,unit_thread);
        let weight=core::slice::from_raw_parts(p.weight as *const f32,p.weights as usize);
        // Transpose may run on an output nobody has written (prompt_writes): its one word a unit is made in a local and stored through the
        // raw pointer, so no reference over the output exists before the store. Every other op keeps the zeroed output and the slice.
        if p.shape.op==crate::prompt_ops::Op::Transpose {
            let mut word=[0.0f32;1];unit(p.shape,index,x,weight,&mut word);
            *(p.output as *mut f32).add(index*words)=word[0];return;
        }
        let output=core::slice::from_raw_parts_mut((p.output as *mut f32).add(index*words),words);
        unit(p.shape,index,x,weight,output);
    }
}
