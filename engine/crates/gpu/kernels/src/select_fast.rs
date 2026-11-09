//! Pool selection for one or more tokens (`Op::SelectRows`), one block of 1024 threads per token. The generic code
//! keeps the best `take` pools in a binary heap in global memory with ONE thread. The order it uses is total (score
//! descending, then pool id ascending; -0.0 and +0.0 are equal scores), so the chosen set and the ascending list of
//! ids are unique and any way of finding them writes the same bytes. Here the cut-off score is found by a four-pass
//! radix select on the sortable score bits, ties at the cut-off go to the lowest ids, and an ordered block scan
//! writes the ascending ids. A NaN score has no order and the generic code defines the bytes then, so a token
//! with any NaN score is handed to the original `select_all` / `select_top` on thread 0.
use core::arch::asm;
use crate::prompt_ops::{Arguments,Op};

/// Threads per block this kernel is written for.
pub(crate) const THREADS:usize=1024;

#[no_mangle]
pub extern "ptx-kernel" fn prompt_select_rows(p:Arguments) {
    // SAFETY: The host (backend/prompt_ops.rs, select_fast::admitted) checked Shape::lengths and launches blocks of
    // THREADS threads, one block per token strided by the grid. Scores are read for groups <= rows (else the
    // block traps, as the generic slice would panic); the output unit of a token has slots+1 words and every write
    // below is at an index below slots+1 (selected positions < take, tail and fill < slots, the count at slots), each by one
    // thread. Shared memory is this kernel's own 1,280-byte array. The NaN flag and the histogram bins are written only by
    // atomic shared OR and ADD (any number of threads, no plain store races); every other word has ONE named writer (thread 0,
    // lane 31 of a warp, or the thread of that index) between barriers. Every thread reaches every barrier (loops depend on the
    // shape only; the NaN branch has none inside). The NaN branch runs the generic functions on thread 0 over the same slices.
    // Ids: the host admits only contexts below 2^24 (select_admission), where every pool id and expanded id is an exact f32 integer,
    // as the old heap's f32 id storage requires.
    // Every shared read follows a write: the 256 histogram words are zeroed by threads 0..255 of the 1,024 before the barrier
    // that precedes any add or read; PREFIX, REM and ABOVE are written by thread 0 before a barrier before any read; each of
    // the 32 warps writes its own scan word before the barrier that precedes the first-level scan, which reads exactly those 32
    // words (the block always has 1,024 threads, checked at entry); TOTAL is written by lane 31 of warp 0 before the barrier that
    // precedes its read. A block with fewer threads traps.
    unsafe {
        let s=p.shape;
        if s.op!=Op::SelectRows || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (d,budget,r,slots,m)=(s.dim as usize,s.taps as usize,s.rows as usize,s.slots as usize,s.tokens as usize);
        let words=slots+1;let take=budget/d;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;let warp=tid/32;
        let smem=crate::ptx::shared::<1280>();
        const HIST:u32=0;const FLAG:u32=1024;const PREFIX:u32=1028;const REM:u32=1032;const ABOVE:u32=1036;const TOTAL:u32=1040;const WARPS:u32=1088;
        macro_rules! ld {($a:expr)=>{{let v:u32;asm!("ld.shared.u32 {v}, [{a}];",v=out(reg32)v,a=in(reg32)(smem+$a));v}}}
        macro_rules! st {($a:expr,$v:expr)=>{asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(smem+$a),v=in(reg32)$v)}}
        macro_rules! bar {()=>{asm!("bar.sync 0;")}}
        macro_rules! key {($v:expr)=>{{let mut bits=($v).to_bits();if bits==0x8000_0000 {bits=0;}if bits>>31==1 {!bits} else {bits|0x8000_0000}}}}
        // Exclusive scan of one u32 per thread across the block; returns (exclusive prefix, block total).
        macro_rules! scan {($v:expr)=>{{
            let own:u32=$v;let mut x=own;let mut o=1u32;
            while o<32 {let y:u32;asm!("shfl.sync.up.b32 {y}, {x}, {o}, 0, -1;",y=out(reg32)y,x=in(reg32)x,o=in(reg32)o,options(nomem,nostack));if lane>=o as usize {x+=y;}o*=2;}
            if lane==31 {st!(WARPS+(warp as u32)*4,x);}
            bar!();
            if warp==0 {
                let t=ld!(WARPS+(lane as u32)*4);let mut w=t;let mut o=1u32;
                while o<32 {let y:u32;asm!("shfl.sync.up.b32 {y}, {x}, {o}, 0, -1;",y=out(reg32)y,x=in(reg32)w,o=in(reg32)o,options(nomem,nostack));if lane>=o as usize {w+=y;}o*=2;}
                st!(WARPS+(lane as u32)*4,w-t);
                if lane==31 {st!(TOTAL,w);}
            }
            bar!();
            let base=ld!(WARPS+(warp as u32)*4);let total=ld!(TOTAL);
            bar!();
            (x-own+base,total)
        }}}
        let mut token=core::arch::nvptx::_block_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize;
        while token<m {
            let context=s.first as usize+token;let groups=context/d;
            if groups>r {core::arch::nvptx::trap();}
            let scores=(p.input[0] as *const f32).add(token*r);
            let out=(p.output as *mut f32).add(token*words);
            if tid==0 {st!(FLAG,0u32);}
            bar!();
            let mut i=tid;while i<groups {let v=*scores.add(i);if v!=v {asm!("red.shared.or.b32 [{a}], 1;",a=in(reg32)(smem+FLAG));}i+=THREADS;}
            bar!();
            let nan=ld!(FLAG)!=0;
            let tail=if s.cols!=0 {context-groups*d} else {0};
            if nan {
                if tid==0 {
                    let row=core::slice::from_raw_parts(scores,r);let o=core::slice::from_raw_parts_mut(out,words);
                    if take>=groups {crate::prompt_ops::select_all(s,token,row,o);} else {crate::prompt_ops::select_top(s,token,row,o);}
                }
            } else if take>=groups {
                let count=groups*d+tail;
                i=tid;while i<count {*out.add(i)=i as f32;i+=THREADS;}
                i=count+tid;while i<slots {*out.add(i)=-1.0;i+=THREADS;}
                if tid==0 {*out.add(slots)=count as f32;}
            } else {
                if tid==0 {st!(PREFIX,0u32);st!(REM,take as u32);st!(ABOVE,0u32);}
                bar!();
                let mut pass=0u32;
                while pass<4 {
                    if tid<256 {st!(HIST+(tid as u32)*4,0u32);}
                    bar!();
                    let prefix=ld!(PREFIX);let shift=24-8*pass;
                    i=tid;while i<groups {
                        let k:u32=key!(*scores.add(i));
                        if pass==0 || (k>>(shift+8))==prefix {asm!("red.shared.add.u32 [{a}], 1;",a=in(reg32)(smem+HIST+((k>>shift)&255)*4));}
                        i+=THREADS;
                    }
                    bar!();
                    if tid==0 {
                        let rem=ld!(REM);let mut cum=0u32;let mut b=255u32;
                        loop {let h=ld!(HIST+b*4);if cum+h>=rem || b==0 {break;}cum+=h;b-=1;}
                        st!(REM,rem-cum);st!(ABOVE,ld!(ABOVE)+cum);st!(PREFIX,(prefix<<8)|b);
                    }
                    bar!();
                    pass+=1;
                }
                let cut=ld!(PREFIX);let need=ld!(REM);
                let (mut carry_eq,mut carry_selected)=(0u32,0u32);
                let mut base=0;
                while base<groups {
                    let g=base+tid;let live=g<groups;
                    let k:u32=if live {key!(*scores.add(g))} else {0};
                    let greater=live && k>cut;let equal=live && k==cut;
                    let (eq_before,eq_total)=scan!(equal as u32);
                    let chosen=greater || (equal && carry_eq+eq_before<need);
                    let (before,total)=scan!(chosen as u32);
                    if chosen {let at=((carry_selected+before) as usize)*d;let mut j=0;while j<d {*out.add(at+j)=(g*d+j) as f32;j+=1;}}
                    carry_eq+=eq_total;carry_selected+=total;
                    base+=THREADS;
                }
                let count=take*d+tail;
                i=tid;while i<tail {*out.add(take*d+i)=(groups*d+i) as f32;i+=THREADS;}
                i=count+tid;while i<slots {*out.add(i)=-1.0;i+=THREADS;}
                if tid==0 {*out.add(slots)=count as f32;}
            }
            bar!();
            token+=stride;
        }
    }
}
