//! Lossless packet representation. No floating-point operation changes a word.
use crate::prompt_ops::{Arguments,Op};
use core::arch::asm;
#[export_name="prompt_wire_bf16"]
pub extern "ptx-kernel" fn entry(p:Arguments) {
    // SAFETY: prompt_operation_output validates Shape::lengths, contiguous same-context
    // owners and disjoint input/output ranges. Pack owns input 2*i,2*i+1 and output i;
    // unpack owns output i and reads the half for its bijective rank/transpose index.
    // The host queues owners through stream completion. Mapped pack writes finish
    // before host transport reads; unpack follows both rail completions. Each output
    // word has one writer. No output reference is formed over unwritten memory.
    // The transpose form uses 32x33 shared words. All threads reach its barrier;
    // stores and reads cover the same live tile coordinates, including edge tiles.
    unsafe {
        let x=p.input[0] as *const u32;let out=p.output as *mut u32;let s=p.shape;
        if s.op==Op::WireUnpackBf16 && s.first!=0 {
            let (rows,cols)=(s.rows as usize,s.cols as usize);
            let tile=core::arch::nvptx::_block_idx_x() as usize;let tid=core::arch::nvptx::_thread_idx_x() as usize;
            if core::arch::nvptx::_block_dim_x()!=128 {core::arch::nvptx::trap();}
            let (r0,c0)=(tile/cols.div_ceil(32)*32,tile%cols.div_ceil(32)*32);let shared=crate::ptx::shared::<4224>();
            let mut at=tid;while at<1024 {
                let (r,c)=(at/32,at%32);if r0+r<rows && c0+c<cols {
                    let source=(r0+r)*cols+c0+c;let v=*x.add(source/2);let bits=if source%2==0 {v<<16} else {v&0xffff0000};
                    if s.slots!=0 && bits&0x7f800000==0x7f800000 {core::arch::nvptx::trap();}
                    asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(shared+((r*33+c)*4) as u32),v=in(reg32)bits);
                }at+=128;
            }
            asm!("bar.sync 0;");
            at=tid;while at<1024 {
                let (c,r)=(at/32,at%32);if r0+r<rows && c0+c<cols {
                    let bits:u32;asm!("ld.shared.u32 {v}, [{a}];",v=out(reg32)bits,a=in(reg32)(shared+((r*33+c)*4) as u32));
                    *out.add((c0+c)*rows+r0+r)=bits;
                }at+=128;
            }return;
        }
        let i=crate::index();if i as u64>=p.units {return;}
        let bits=if s.op==Op::WirePackBf16 {
            let a=*x.add(2*i);let b=*x.add(2*i+1);
            if (a|b)&0xffff!=0 {core::arch::nvptx::trap();}
            (a>>16)|(b&0xffff0000)
        } else if s.op==Op::WireUnpackBf16 {
            let (r,k)=(s.rows as usize,s.cols as usize);
            let source=if s.first!=0 {(i%r)*k+i/r} else {crate::prompt_ops::rank_row(s,i/k)*k+i%k};
            let v=*x.add(source/2);let bits=if source%2==0 {v<<16} else {v&0xffff0000};
            if s.slots!=0 && bits&0x7f800000==0x7f800000 {core::arch::nvptx::trap();}bits
        } else {core::arch::nvptx::trap();};
        *out.add(i)=bits;
    }
}
