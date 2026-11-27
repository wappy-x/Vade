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
    // The transpose form uses 32x33 dynamic shared words (4,224 bytes, checked here via
    // %dynamic_smem_size); linear pack/unpack use 0 bytes shared memory. All threads
    // reach the transpose barrier; stores and reads cover the same live tile coordinates.
    unsafe {
        let x=p.input[0] as *const u32;let out=p.output as *mut u32;let s=p.shape;
        if s.op==Op::WireUnpackBf16 && s.first!=0 {
            let (rows,cols,rank_stride)=(s.rows as usize,s.cols as usize,s.dim as usize);
            let local_rows=if rank_stride==0 {rows} else {if s.taps==0 || rows%(s.taps as usize)!=0 {core::arch::nvptx::trap();} rows/(s.taps as usize)};
            let tile=core::arch::nvptx::_block_idx_x() as usize;let tid=core::arch::nvptx::_thread_idx_x() as usize;
            let dynamic:u32;
            asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
            if core::arch::nvptx::_block_dim_x()!=128 || dynamic<4224 {core::arch::nvptx::trap();}
            let (r0,c0)=(tile/cols.div_ceil(32)*32,tile%cols.div_ceil(32)*32);let shared=0u32;
            if (cols&1)==0 {
                // Read each packed BF16 u32 word once for two adjacent columns (2x fewer mapped-memory loads).
                let mut at=tid;while at<512 {
                    let (r,c)=(at/16,(at%16)*2);let rg=r0+r;if rg<rows && c0+c<cols {
                        let source=if rank_stride==0 {rg*cols+c0+c} else {(rg/local_rows)*(rank_stride*2)+(rg%local_rows)*cols+c0+c};
                        let v:u32;asm!("ld.global.nc.u32 {v}, [{p}];",v=out(reg32)v,p=in(reg64)x.add(source/2),options(nostack,readonly));
                        let b0=v<<16;let b1=v&0xffff0000;
                        if s.slots!=0 && (b0&0x7f800000==0x7f800000 || b1&0x7f800000==0x7f800000) {core::arch::nvptx::trap();}
                        asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(shared+((r*33+c)*4) as u32),v=in(reg32)b0);
                        if c0+c+1<cols {asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(shared+((r*33+c+1)*4) as u32),v=in(reg32)b1);}
                    }at+=128;
                }
            } else {
                let mut at=tid;while at<1024 {
                    let (r,c)=(at/32,at%32);let rg=r0+r;if rg<rows && c0+c<cols {
                        let source=if rank_stride==0 {rg*cols+c0+c} else {(rg/local_rows)*(rank_stride*2)+(rg%local_rows)*cols+c0+c};
                        let v=*x.add(source/2);let bits=if source%2==0 {v<<16} else {v&0xffff0000};
                        if s.slots!=0 && bits&0x7f800000==0x7f800000 {core::arch::nvptx::trap();}
                        asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(shared+((r*33+c)*4) as u32),v=in(reg32)bits);
                    }at+=128;
                }
            }
            asm!("bar.sync 0;");
            let mut at=tid;while at<1024 {
                let (c,r)=(at/32,at%32);if r0+r<rows && c0+c<cols {
                    let bits:u32;asm!("ld.shared.u32 {v}, [{a}];",v=out(reg32)bits,a=in(reg32)(shared+((r*33+c)*4) as u32));
                    *out.add((c0+c)*rows+r0+r)=bits;
                }at+=128;
            }return;
        }
        let tid=crate::index();
        let stride=(core::arch::nvptx::_grid_dim_x() as usize)*(core::arch::nvptx::_block_dim_x() as usize);
        let units=p.units as usize;
        if s.op==Op::WirePackBf16 {
            if (units&3)==0 && (p.input[0]&15)==0 && (p.output&15)==0 {
                let quads=units/4;
                let mut q=tid;
                while q<quads {
                    let (a0,b0,a1,b1):(u32,u32,u32,u32);
                    let (a2,b2,a3,b3):(u32,u32,u32,u32);
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",
                        a=out(reg32)a0,b=out(reg32)b0,c=out(reg32)a1,d=out(reg32)b1,p=in(reg64)x.add(8*q),options(nostack,readonly));
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",
                        a=out(reg32)a2,b=out(reg32)b2,c=out(reg32)a3,d=out(reg32)b3,p=in(reg64)x.add(8*q+4),options(nostack,readonly));
                    if (a0|b0|a1|b1|a2|b2|a3|b3)&0xffff!=0 {core::arch::nvptx::trap();}
                    let p0=(a0>>16)|(b0&0xffff0000);let p1=(a1>>16)|(b1&0xffff0000);
                    let p2=(a2>>16)|(b2&0xffff0000);let p3=(a3>>16)|(b3&0xffff0000);
                    asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",
                        p=in(reg64)out.add(4*q),a=in(reg32)p0,b=in(reg32)p1,c=in(reg32)p2,d=in(reg32)p3,options(nostack));
                    q+=stride;
                }
                return;
            }
            let mut i=tid;
            while i<units {
                let a=*x.add(2*i);let b=*x.add(2*i+1);
                if (a|b)&0xffff!=0 {core::arch::nvptx::trap();}
                *out.add(i)=(a>>16)|(b&0xffff0000);
                i+=stride;
            }
            return;
        }
        if s.op==Op::WireUnpackBf16 {
            let (r,k,rank_stride)=(s.rows as usize,s.cols as usize,s.dim as usize);
            if s.first==0 && (k&7)==0 && (rank_stride&3)==0 && (p.input[0]&15)==0 && (p.output&15)==0 {
                let octs=units/8;
                let k_pow2=k.is_power_of_two();
                let k_shift=k.trailing_zeros();
                let k_mask=k.wrapping_sub(1);
                let r_pow2=r.is_power_of_two();
                let r_shift=r.trailing_zeros();
                let r_mask=r.wrapping_sub(1);
                let rank_stride2=rank_stride*2;
                let mut o=tid;
                while o<octs {
                    let base=8*o;
                    let (row,col)=if k_pow2 {(base>>k_shift,base&k_mask)} else {(base/k,base%k)};
                    let source=if rank_stride==0 {
                        crate::prompt_ops::rank_row(s,row)*k+col
                    } else if r_pow2 {
                        (row&r_mask)*rank_stride2+(row>>r_shift)*k+col
                    } else {
                        (row%r)*rank_stride2+(row/r)*k+col
                    };
                    let (v0,v1,v2,v3):(u32,u32,u32,u32);
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",
                        a=out(reg32)v0,b=out(reg32)v1,c=out(reg32)v2,d=out(reg32)v3,p=in(reg64)x.add(source/2),options(nostack,readonly));
                    let b0=v0<<16;let b1=v0&0xffff0000;
                    let b2=v1<<16;let b3=v1&0xffff0000;
                    let b4=v2<<16;let b5=v2&0xffff0000;
                    let b6=v3<<16;let b7=v3&0xffff0000;
                    if s.slots!=0 && ((b0&0x7f800000)==0x7f800000 || (b1&0x7f800000)==0x7f800000
                        || (b2&0x7f800000)==0x7f800000 || (b3&0x7f800000)==0x7f800000
                        || (b4&0x7f800000)==0x7f800000 || (b5&0x7f800000)==0x7f800000
                        || (b6&0x7f800000)==0x7f800000 || (b7&0x7f800000)==0x7f800000) {
                        core::arch::nvptx::trap();
                    }
                    asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",
                        p=in(reg64)out.add(base),a=in(reg32)b0,b=in(reg32)b1,c=in(reg32)b2,d=in(reg32)b3,options(nostack));
                    asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",
                        p=in(reg64)out.add(base+4),a=in(reg32)b4,b=in(reg32)b5,c=in(reg32)b6,d=in(reg32)b7,options(nostack));
                    o+=stride;
                }
                return;
            }
            if s.first==0 && (k&3)==0 && (rank_stride&1)==0 && (p.input[0]&7)==0 && (p.output&15)==0 {
                let quads=units/4;
                let mut q=tid;
                while q<quads {
                    let base=4*q;
                    let row=base/k;
                    let source=if rank_stride==0 {crate::prompt_ops::rank_row(s,row)*k+base%k} else {(row%r)*(rank_stride*2)+(row/r)*k+base%k};
                    let (v0,v1):(u32,u32);
                    asm!("ld.global.nc.v2.u32 {{{a},{b}}}, [{p}];",a=out(reg32)v0,b=out(reg32)v1,p=in(reg64)x.add(source/2),options(nostack,readonly));
                    let b0=v0<<16;let b1=v0&0xffff0000;
                    let b2=v1<<16;let b3=v1&0xffff0000;
                    if s.slots!=0 && (b0&0x7f800000==0x7f800000 || b1&0x7f800000==0x7f800000 || b2&0x7f800000==0x7f800000 || b3&0x7f800000==0x7f800000) {
                        core::arch::nvptx::trap();
                    }
                    asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",
                        p=in(reg64)out.add(base),a=in(reg32)b0,b=in(reg32)b1,c=in(reg32)b2,d=in(reg32)b3,options(nostack));
                    q+=stride;
                }
                return;
            }
            let mut i=tid;
            while i<units {
                let row=i/k;
                let source=if s.first!=0 {(i%r)*k+i/r} else if rank_stride==0 {crate::prompt_ops::rank_row(s,row)*k+i%k} else {(row%r)*(rank_stride*2)+(row/r)*k+i%k};
                let v=*x.add(source/2);let bits=if source%2==0 {v<<16} else {v&0xffff0000};
                if s.slots!=0 && bits&0x7f800000==0x7f800000 {core::arch::nvptx::trap();}
                *out.add(i)=bits;
                i+=stride;
            }
            return;
        }
        core::arch::nvptx::trap();
    }
}
