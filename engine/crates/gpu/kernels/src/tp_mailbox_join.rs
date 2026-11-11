//! The joined put, in its own module so the existing kernels keep their code.
//! One launch does for one exchange message what several launches did: the BF16 round of this
//! rank's rows (where the lane asks), the join of the message parts, the fill of the gathered
//! output (own rows at their final places, zero elsewhere), and the put of the message to the
//! send slot. Every word that reaches the slot or the output is the word those launches wrote.
use core::arch::asm;
use core::arch::nvptx::{_block_dim_x,_block_idx_x,_grid_dim_x,_thread_idx_x};
use crate::{kernel_params::HybridCopyRow,resident_launch_abi::{Command,Kind},tp_put_join::{quad,ROUND}};

/// `table` has `rows` (low 32 bits) rows; bit 32 of `rows` allows four-word moves. The first
/// `slot_rows` rows fill the send slot and the others fill the gathered output. A row moves its
/// words (low 31 bits of `words`) from its source, or zero when the source is null, to its
/// destination; bit 31 asks for the BF16 round of each word first (crate::bf16::rounded, the
/// round kernel's function). `padding` holds the units before the row in its space and `units`
/// holds the totals (slot low, output high). Block 0 moves the slot rows and publishes the
/// message length `bytes` and the sequence; the other blocks move the output rows.
/// A nonfinite message word sets `error` and nothing is published, as the finite put does.
#[no_mangle]
pub extern "ptx-kernel" fn tp_mailbox_put_join(table:*const HybridCopyRow,rows:u64,units:u64,slot_rows:u64,len_word:*mut u32,bytes:u64,seq_word:*mut u32,seq:u64,error:*mut u32) {
    put_join(table,rows,units,slot_rows,len_word,bytes,seq_word,seq,error)
}
#[no_mangle]
pub extern "ptx-kernel" fn resident_launch_put_join(table:*const Command,index:u64) {
    // SAFETY: The checked recorder writes nine u64 words of a PutJoin command in its private
    // aligned table. That table covers index and is uploaded before this replay on the same
    // stream; the command's row table is uploaded with the same piece. A command of another
    // kind traps here. The body then has exactly the extents of the direct entry above.
    unsafe {
        let c=&*table.add(index as usize);if c.kind!=Kind::PutJoin as u32 {core::arch::nvptx::trap();}let w=c.payload.as_ptr().cast::<u64>();
        put_join(*w as _,*w.add(1),*w.add(2),*w.add(3),*w.add(4) as _,*w.add(5),*w.add(6) as _,*w.add(7),*w.add(8) as _);
    }
}
#[inline(always)]
fn put_join(table:*const HybridCopyRow,rows:u64,units:u64,slot_rows:u64,len_word:*mut u32,bytes:u64,seq_word:*mut u32,seq:u64,error:*mut u32) {
    // SAFETY: Backend::tp_mailbox_put_join admits this body. tp_put_join::table and encode make
    // the table: slot rows first, at least one, then at least one output row; every source range
    // lies inside one retained contiguous part; slot rows lie inside the send slot (mapped host
    // memory) and are the message in order; output rows are disjoint and together are one whole
    // retained output, which overlaps no part. A null source is never read. `padding` is the
    // exact count of units before each row in its space under tp_put_join::quad, the rule this
    // body uses, and `units` holds the two totals; so a unit below its total lies in one row of
    // its space, at or after the row the search finds. The grid has block 0 and at least one
    // output block. `len_word`, `seq_word` and `error` are aligned mapped header words. Table,
    // parts, output and mailbox live through ordered completion.
    // One writer a word: in its space a thread takes units first, first+stride, ...
    // (tp_put_wide::first and stride over that space's blocks), which are disjoint among threads
    // and cover every unit (host test tp_put_join::tests at every message shape and launch
    // shape). Four-word moves run only under tp_put_join::quad: the count divides and both ends
    // are 16-byte aligned.
    // Publication is the one-block put's: one block owns the slot, its barrier orders every slot
    // store and the error store before thread 0 publishes, sequence word last. The output is
    // read by later launches of the stream, after this launch ends.
    unsafe {
        let threads=_block_dim_x();let slot_rows=slot_rows as usize;let slot=_block_idx_x()==0;
        let (first_row,count,total,blocks,block)=if slot {(0,slot_rows,units as u32 as usize,1,0)}
            else {(slot_rows,rows as u32 as usize-slot_rows,(units>>32) as usize,_grid_dim_x()-1,_block_idx_x()-1)};
        let (table,wide,step)=(table.add(first_row),rows>>32&1!=0,crate::tp_put_wide::stride(blocks,threads));let mut valid=true;
        let mut unit=crate::tp_put_wide::first(block,_thread_idx_x(),threads);
        if unit<total {
            let (mut low,mut high)=(0,count);
            while high-low>1 {let middle=(low+high)/2;if (*table.add(middle)).padding as usize<=unit {low=middle;} else {high=middle;}}
            let mut row=low;
            loop {
                let entry=&*table.add(row);let (from,to,n,round)=(entry.source as *const u32,entry.destination as *mut u32,(entry.words&!ROUND) as usize,entry.words&ROUND!=0);
                let four=quad(entry.source,entry.destination,n,wide);let at=unit-entry.padding as usize;
                if at>=if four {n/4} else {n} {row+=1;continue;}
                if four {
                    let (mut a,mut b,mut c,mut d)=(0u32,0u32,0u32,0u32);
                    if !from.is_null() {asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)from.add(4*at));}
                    if round {a=bf16(a);b=bf16(b);c=bf16(c);d=bf16(d);}
                    valid&=(a&0x7f80_0000!=0x7f80_0000)&(b&0x7f80_0000!=0x7f80_0000)&(c&0x7f80_0000!=0x7f80_0000)&(d&0x7f80_0000!=0x7f80_0000);
                    asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",p=in(reg64)to.add(4*at),a=in(reg32)a,b=in(reg32)b,c=in(reg32)c,d=in(reg32)d);
                } else {
                    let mut value=if from.is_null() {0} else {*from.add(at)};if round {value=bf16(value);}
                    valid&=value&0x7f80_0000!=0x7f80_0000;*to.add(at)=value;
                }
                unit+=step;if unit>=total {break;}
            }
        }
        if !slot {return;}
        if !valid {asm!("st.release.sys.global.u32 [{p}], {v};",p=in(reg64)error,v=in(reg32)1u32);}
        crate::ptx::sync();
        if _thread_idx_x()==0 {
            let failed:u32;asm!("ld.acquire.sys.global.u32 {v}, [{p}];",v=out(reg32)failed,p=in(reg64)error);
            if failed==0 {asm!("fence.sc.sys;");*len_word=bytes as u32;asm!("st.release.sys.global.u32 [{p}], {v};",p=in(reg64)seq_word,v=in(reg32)seq as u32);}
        }
    }
}
/// The round kernel's function on raw bits.
#[inline(always)]
fn bf16(bits:u32)->u32 {crate::bf16::rounded(f32::from_bits(bits)).to_bits()}
