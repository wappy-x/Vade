use crate::prompt_route_abi::{Build,THREADS};
/// Count each expert without assigning slots by arrival order.
#[no_mangle]
pub extern "ptx-kernel" fn prompt_route_count(p:Build) {
    // SAFETY: the backend checks the admitted dimensions, exact contiguous ID span,
    // private counts[experts] and status[1], and one 256-thread block per expert.
    // Owners live through completion. Each count and the status have one writer.
    // All threads reach every barrier; shared indices stay within 512 words.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        if _block_dim_x()!=THREADS || !p.shape.admitted() {core::arch::asm!("trap;");}
        let e=_block_idx_x();if e>=p.shape.experts {return;}
        let tid=_thread_idx_x() as usize;let ids=p.ids as *const f32;let n=p.shape.pairs();let mut count=0u32;let mut bad=0u32;
        let mut at=tid;while at<n {
            let value=*ids.add(at);count+=u32::from(value==e as f32);
            if e==0 {
                if !(value>=0.0 && value<p.shape.experts as f32 && value==(value as u32) as f32) {bad=1;}
                let slot=at%p.shape.picks as usize;let mut prev=0;while prev<slot {if *ids.add(at-slot+prev)==value {bad=1;}prev+=1;}
            }
            at+=THREADS as usize;
        }
        let shared=crate::ptx::shared::<2048>();crate::ptx::store(shared+(4*tid) as u32,f32::from_bits(count));crate::ptx::store(shared+(4*(tid+256)) as u32,f32::from_bits(bad));crate::ptx::sync();
        let mut width=128usize;while width>0 {
            if tid<width {
                let a=crate::ptx::load(shared+(4*tid) as u32).to_bits()+crate::ptx::load(shared+(4*(tid+width)) as u32).to_bits();
                let b=crate::ptx::load(shared+(4*(tid+256)) as u32).to_bits()|crate::ptx::load(shared+(4*(tid+256+width)) as u32).to_bits();
                crate::ptx::store(shared+(4*tid) as u32,f32::from_bits(a));crate::ptx::store(shared+(4*(tid+256)) as u32,f32::from_bits(b));
            }
            crate::ptx::sync();width/=2;
        }
        if tid==0 {*(p.counts as *mut u32).add(e as usize)=crate::ptx::load(shared).to_bits();if e==0 {*(p.status as *mut u32)=crate::ptx::load(shared+1024).to_bits();}}
    }
}
/// Stable scatter and compact tile maps. A ballot ranks each matching original slot.
#[no_mangle]
pub extern "ptx-kernel" fn prompt_route_tables(p:Build) {
    // SAFETY: the backend checks every admitted plane extent and keeps all owners
    // through completion. The preceding ordered count launch writes every count
    // and status. Invalid IDs trap before any table is used. One block owns one
    // expert; its stable rank assigns each pair exactly one sorted slot. Expert
    // intervals and work intervals are disjoint. The last expert fills only the
    // unused work tail. All lanes take every ballot and all threads every barrier.
    // Each work-map capacity is experts + ceil(pairs/tile), bounding base + live.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        if _block_dim_x()!=THREADS || !p.shape.admitted() || *(p.status as *const u32)!=0 {core::arch::asm!("trap;");}
        let e=_block_idx_x();if e>=p.shape.experts {return;}
        let tid=_thread_idx_x() as usize;let lane=tid%32;let warp=tid/32;let s=p.shape;let n=s.pairs();let counts=p.counts as *const u32;
        let shared=crate::ptx::shared::<64>();
        if tid==0 {
            let mut earlier=0u32;let mut w128=0u32;let mut w64=0u32;let mut other=0;
            while other<e {let count=*counts.add(other as usize);if count>s.rows {core::arch::asm!("trap;");}earlier+=count;w128+=count.div_ceil(128);w64+=count.div_ceil(64);other+=1;}
            let count=*counts.add(e as usize);if count>s.rows || earlier+count>n as u32 {core::arch::asm!("trap;");}
            for (i,v) in [earlier,count,w128,w64].into_iter().enumerate() {crate::ptx::store(shared+(4*i) as u32,f32::from_bits(v));}
            *(p.starts as *mut u32).add(e as usize)=earlier;
            if e+1==s.experts {if earlier+count!=n as u32 {core::arch::asm!("trap;");}*(p.starts as *mut u32).add(s.experts as usize)=earlier+count;}
        }
        crate::ptx::sync();let first=crate::ptx::load(shared).to_bits();let count=crate::ptx::load(shared+4).to_bits();let w128=crate::ptx::load(shared+8).to_bits();let w64=crate::ptx::load(shared+12).to_bits();
        for (tile,base,address) in [(128u32,w128,p.work128),(64,w64,p.work64)] {
            let live=count.div_ceil(tile);let mut at=tid as u32;while at<live {let target=(address as *mut crate::prompt_route_abi::Tile).add((base+at) as usize);*target=crate::prompt_route_abi::Tile {expert:e,tile:at};at+=THREADS;}
            if e+1==s.experts {let mut at=(base+live) as usize+tid;while at<s.work_capacity(tile) {*(address as *mut crate::prompt_route_abi::Tile).add(at)=crate::prompt_route_abi::EMPTY;at+=THREADS as usize;}}
        }
        let ids=p.ids as *const f32;let mut before=0u32;let mut chunk=0usize;
        while chunk<n {
            let pair=chunk+tid;let yes=u32::from(pair<n && *ids.add(pair)==e as f32);let mask:u32;
            core::arch::asm!("{{ .reg .pred p; setp.ne.u32 p, {yes}, 0; vote.ballot.sync.b32 {mask}, p, 0xffffffff; }}",yes=in(reg32) yes,mask=out(reg32) mask);
            if lane==0 {crate::ptx::store(shared+16+(4*warp) as u32,f32::from_bits(mask.count_ones()));}crate::ptx::sync();
            let mut position=before+(mask&((1u32<<lane)-1)).count_ones();let mut total=0u32;
            let mut w=0;while w<8 {let v=crate::ptx::load(shared+16+4*w).to_bits();if w<(warp as u32) {position+=v;}total+=v;w+=1;}
            if yes!=0 {
                let sorted=first+position;if position>=count || sorted>=n as u32 {core::arch::asm!("trap;");}
                *(p.order as *mut f32).add(sorted as usize)=(pair/s.picks as usize) as f32;*(p.pairs as *mut f32).add(sorted as usize)=pair as f32;
                for (col,v) in [first as f32,count as f32,position as f32].into_iter().enumerate() {
                    *(p.grouped as *mut f32).add(3*sorted as usize+col)=v;*(p.placed as *mut f32).add(3*pair+col)=v;
                }
                *(p.grouped as *mut f32).add(3*n+sorted as usize)=sorted as f32;*(p.placed as *mut f32).add(3*n+sorted as usize)=pair as f32;
            }
            before+=total;crate::ptx::sync();chunk+=THREADS as usize;
        }
        if before!=count {core::arch::asm!("trap;");}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn prompt_route_native(p:crate::prompt_route_abi::NativeBuild) {
    // SAFETY: the backend checks count immutable NativeSet headers, experts
    // immutable NvPart templates and each route's exact admitted metadata spans.
    // Optional parts/work/activation outputs are private, disjoint and fully
    // bounded. Each expert block owns its interleaved part records, sorted row
    // intervals and compact work interval. Only the last block fills the unused
    // work tail. The ordered route launches finish first; all owners survive the
    // consumers. All threads reach the one shared barrier. No pointer comes from
    // an ID: every weight pointer is copied from a checked resident template.
    // The host passes the route work map built at tokens, so prefixes stay disjoint.
    // PAIRED PARTS (merge 1): the expert's interval is ceil(sum of its set counts
    // / tokens) records, its prefix the same sum over every lower expert, reduced
    // by all 256 threads in this block's shared words (every thread reaches both
    // barriers; merge is uniform). Each record names part e*count. The bound
    // experts + ceil(pairs / tokens) covers every interval (host capacity).
    // DIRECT PLACE (rowmap, with a table): each expert block stores, for its own
    // sorted rows first..first+count of each set (disjoint across experts), the
    // address of the row's original pair in the set's private output; the pair
    // is read from the route's immutable pairs plane and must be a whole number
    // below the set's pairs (else trap), so the address lies inside that output.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        use crate::prompt_route_abi::{NativeSet,Tile,MAX_SETS,MAX_EXPERTS,MAX_ROWS,MAX_PICKS};
        if _block_dim_x()!=THREADS || p.count==0 || p.count>MAX_SETS || p.experts==0 || p.experts>MAX_EXPERTS
            || p.work!=0 && !(p.tokens==64 || p.tokens==128) || p.table!=0 && (p.rows==0 || p.rows%128!=0) || p.merge>1 || p.merge==1 && p.work==0 {core::arch::asm!("trap;");}
        let e=_block_idx_x();if e>=p.experts {return;}
        let tid=_thread_idx_x() as usize;let shared=crate::ptx::shared::<1024>();let sets=p.sets as *const NativeSet;
        if p.merge==1 {
            // Sum over experts x < e of ceil(sum over sets of count[x] / tokens), one strided share a thread.
            let mut sum=0u32;let mut x=tid as u32;
            while x<e {let mut t=0u32;let mut set=0;while set<p.count {let c=*((*sets.add(set as usize)).counts as *const u32).add(x as usize);if c>MAX_ROWS {core::arch::asm!("trap;");}t+=c;set+=1;}sum+=t.div_ceil(p.tokens);x+=THREADS;}
            crate::ptx::store(shared+(4*tid) as u32,f32::from_bits(sum));crate::ptx::sync();
            if tid==0 {
                let mut prefix=0u32;let mut i=0;while i<THREADS as usize {prefix+=crate::ptx::load(shared+(4*i) as u32).to_bits();i+=1;}
                if prefix>p.capacity {core::arch::asm!("trap;");}
                crate::ptx::store(shared,f32::from_bits(prefix));
            }
        } else if tid==0 {
            let mut prefix=0u32;
            if p.work!=0 {let mut set=0;while set<p.count {
                let s=*sets.add(set as usize);let mut lo=0u32;let mut hi=s.work_capacity;
                while lo<hi {let mid=lo+(hi-lo)/2;let item=*(s.work as *const Tile).add(mid as usize);if item.expert<e {lo=mid+1;} else {hi=mid;}}
                prefix+=lo;set+=1;
            }}
            if prefix>p.capacity {core::arch::asm!("trap;");}crate::ptx::store(shared,f32::from_bits(prefix));
        }
        crate::ptx::sync();let mut next=crate::ptx::load(shared).to_bits();let template=*(p.templates as *const crate::tc_abi::NvPart).add(e as usize);
        let mut total=0u32;let mut set=0;while set<p.count {
            let s=*sets.add(set as usize);let first=*(s.starts as *const u32).add(e as usize);let count=*(s.counts as *const u32).add(e as usize);
            if s.pairs==0 || s.pairs>MAX_ROWS*MAX_PICKS || count>MAX_ROWS || first>s.pairs || count>s.pairs-first || s.first>u32::MAX-s.pairs {core::arch::asm!("trap;");}
            if p.table!=0 && tid==0 {
                if s.output==0 {core::arch::asm!("trap;");}
                let made=crate::tc_abi::NvPart {output:s.output+4*u64::from(first)*u64::from(p.rows),first:s.first+first,tokens:count,..template};
                *(p.table as *mut crate::tc_abi::NvPart).add((e*p.count+set) as usize)=made;
            }
            if s.activations!=0 {let mut at=tid;while at<count as usize {*(s.activations as *mut f32).add(first as usize+at)=template.activation;at+=THREADS as usize;}}
            if p.rowmap!=0 {
                if p.table==0 || s.order==0 || s.output==0 {core::arch::asm!("trap;");}
                let mut at=tid;while at<count as usize {
                    let i=first as usize+at;let v=*(s.order as *const f32).add(i);
                    if !(v>=0.0 && v<s.pairs as f32 && v==(v as u32) as f32) {core::arch::asm!("trap;");}
                    *(p.rowmap as *mut u64).add(s.first as usize+i)=s.output+4*u64::from(v as u32)*u64::from(p.rows);at+=THREADS as usize;
                }
            }
            if p.work!=0 && p.merge==0 {
                let live=count.div_ceil(p.tokens);if next>p.capacity || live>p.capacity-next {core::arch::asm!("trap;");}
                let mut at=tid as u32;while at<live {*(p.work as *mut crate::tc_abi::NvTile).add((next+at) as usize)=crate::tc_abi::NvTile {part:e*p.count+set,tile:at};at+=THREADS;}next+=live;
            }
            total+=count;set+=1;
        }
        if p.merge==1 {
            let live=total.div_ceil(p.tokens);if next>p.capacity || live>p.capacity-next {core::arch::asm!("trap;");}
            let mut at=tid as u32;while at<live {*(p.work as *mut crate::tc_abi::NvTile).add((next+at) as usize)=crate::tc_abi::NvTile {part:e*p.count,tile:at};at+=THREADS;}next+=live;
        }
        if p.work!=0 && e+1==p.experts {let mut at=next as usize+tid;while at<p.capacity as usize {*(p.work as *mut crate::tc_abi::NvTile).add(at)=crate::tc_abi::NvTile {part:u32::MAX,tile:u32::MAX};at+=THREADS as usize;}}
    }
}
