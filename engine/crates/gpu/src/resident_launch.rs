//! Stable indirect launch ABI. Padding is initialized and never copied from Rust structs.
#[repr(u32)]
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub enum Kind {Causal=1,Expert,Plain,Mx,Round,Copy,Put,Take,Prepare,Collapse,Combine,RmsPartials,RmsApply,Elementwise,CopyWords,ZeroWords,Mixed,NvStream,CausalFast=19,NvTile=20,DeviceRoute=21,TakeScatter=22,PutWide=23,PlainBf16=26,MhcBoundaryRms=0,MhcBoundaryDots=24,MhcBoundaryFinish=25}
#[cfg(not(target_arch="nvptx64"))]
impl Kind {
    pub const ALL:[Self;27]=[Self::PutWide,Self::Causal,Self::Expert,Self::Plain,Self::Mx,Self::Round,Self::Copy,Self::Put,Self::Take,Self::Prepare,Self::Collapse,Self::Combine,Self::RmsPartials,Self::RmsApply,Self::Elementwise,Self::CopyWords,Self::ZeroWords,Self::Mixed,Self::NvStream,Self::CausalFast,Self::NvTile,Self::DeviceRoute,Self::TakeScatter,Self::PlainBf16,Self::MhcBoundaryRms,Self::MhcBoundaryDots,Self::MhcBoundaryFinish];
    pub fn kernel(self)->&'static str {match self {
        Self::Causal=>"resident_launch_causal",Self::Expert=>"resident_launch_expert",Self::Plain=>"resident_launch_plain",Self::Mx=>"resident_launch_mx",
        Self::Round=>"resident_launch_round",Self::Copy=>"resident_launch_copy",Self::Put=>"resident_launch_put",Self::Take=>"resident_launch_take",
        Self::Prepare=>"resident_launch_prepare",Self::Collapse=>"resident_launch_collapse",Self::Combine=>"resident_launch_combine",
        Self::RmsPartials=>"resident_launch_rms_partials",Self::RmsApply=>"resident_launch_rms_apply",Self::Elementwise=>"resident_launch_elementwise",
        Self::CopyWords=>"resident_launch_copy_words",Self::ZeroWords=>"resident_launch_zero_words",Self::Mixed=>"resident_launch_mixed",
        Self::NvStream=>"resident_launch_nv_stream",Self::CausalFast=>"resident_launch_causal_fast",Self::NvTile=>"resident_launch_nv_tile",
        Self::DeviceRoute=>"resident_launch_device_route",Self::TakeScatter=>"resident_launch_take_scatter",Self::PutWide=>"resident_launch_put_wide",Self::PlainBf16=>"resident_launch_plain_bf16",
        Self::MhcBoundaryRms=>"resident_launch_mhc_boundary_rms",Self::MhcBoundaryDots=>"resident_launch_mhc_boundary_dots",Self::MhcBoundaryFinish=>"resident_launch_mhc_boundary_finish",
    }}
}
#[repr(C,align(16))]
#[derive(Clone,Copy)]
pub struct Command {pub kind:u32,pub reserved:[u32;3],pub payload:[u8;1024]}
#[cfg(not(target_arch="nvptx64"))]
impl Command {
    pub fn new(kind:Kind)->Self {Self {kind:kind as u32,reserved:[0;3],payload:[0;1024]}}
    pub fn put(&mut self,offset:usize,bytes:&[u8]) {self.payload[offset..offset+bytes.len()].copy_from_slice(bytes);}
    pub fn word(&mut self,index:usize,value:u64) {self.put(index*8,&value.to_ne_bytes());}
}
const _:()={assert!(core::mem::size_of::<Command>()==1040);assert!(core::mem::offset_of!(Command,payload)==16);};
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn command_words_keep_raw_bits_and_initialized_padding() {
        let mut c=Command::new(Kind::CopyWords);c.word(0,0x80000000_00000001);c.word(1,u64::MAX);
        assert_eq!(&c.payload[..8],&0x80000000_00000001u64.to_ne_bytes());
        assert_eq!(&c.payload[8..16],&u64::MAX.to_ne_bytes());assert_eq!(c.reserved,[0;3]);
        assert!(c.payload[16..].iter().all(|&v|v==0));
    }
    #[test] fn every_original_struct_fits_the_aligned_payload() {
        use core::mem::{size_of,align_of};
        macro_rules! fits {($($t:ty),*)=>{$(assert!(size_of::<$t>()<=1024);assert!(align_of::<$t>()<=16);)*}}
        fits!(crate::causal_ops::Arguments,crate::expert_ops::Arguments,crate::expert_ops::LaunchArguments,crate::next_params::MxLayer,crate::next_params::ResidentProjectionBatch,
            crate::mhc_boundary::LaneCast,crate::kernel_params::HybridCopy,crate::mhc_params::Prepare,
            crate::mhc_params::Collapse,crate::mhc_params::Combine,crate::mhc_params::Boundary,crate::kernel_params::ModelRms,crate::kernel_params::ModelElementwise);
    }
}

/// Successfully submitted resident replay nodes, not graph-capture construction.
/// Bytes are logical destination bytes; copy traffic also reads that many bytes.
#[cfg(not(target_arch="nvptx64"))]
#[derive(Clone,Copy,Default,Debug,PartialEq,Eq)]
pub struct CopyCounts {pub launches:[u64;3],pub bytes:[u64;3],pub merged_zero_launches:u64,pub merged_zero_bytes:u64,pub stage_full_flushes:u64,pub piece_evictions:u64,pub captures_new_key:u64,pub captures_seen_key:u64}
#[cfg(not(target_arch="nvptx64"))]
impl CopyCounts {
    pub fn since(self,before:Self)->Self {Self {launches:std::array::from_fn(|i|self.launches[i].saturating_sub(before.launches[i])),
        bytes:std::array::from_fn(|i|self.bytes[i].saturating_sub(before.bytes[i])),merged_zero_launches:self.merged_zero_launches.saturating_sub(before.merged_zero_launches),merged_zero_bytes:self.merged_zero_bytes.saturating_sub(before.merged_zero_bytes),stage_full_flushes:self.stage_full_flushes.saturating_sub(before.stage_full_flushes),piece_evictions:self.piece_evictions.saturating_sub(before.piece_evictions),captures_new_key:self.captures_new_key.saturating_sub(before.captures_new_key),captures_seen_key:self.captures_seen_key.saturating_sub(before.captures_seen_key)}}
    pub fn line(self)->String {format!("resident_launch_copy={} resident_launch_copy_bytes={} resident_launch_copy_words={} resident_launch_copy_words_bytes={} resident_launch_zero_words={} resident_launch_zero_words_bytes={} resident_merged_zero_launches={} resident_merged_zero_bytes={} stage_full_flushes={} piece_evictions={} captures_new_key={} captures_seen_key={}",
        self.launches[0],self.bytes[0],self.launches[1],self.bytes[1],self.launches[2],self.bytes[2],self.merged_zero_launches,self.merged_zero_bytes,self.stage_full_flushes,self.piece_evictions,self.captures_new_key,self.captures_seen_key)}
}

/// The recorder's table area. Zero rows are the first bytes, and every flush's tables follow them.
#[cfg(not(target_arch="nvptx64"))]
pub mod stage {
    pub const BYTES:usize=1<<20;pub const ZERO_ROWS:usize=256;pub const BASE:usize=ZERO_ROWS*24;
    /// A table that does not fit this flush's area forces a flush. This depends only on the tables
    /// of this flush, never on earlier flushes, so the same commands always cut at the same place.
    pub fn full(staged:usize,bytes:usize)->bool {BASE+staged+bytes>BYTES}
    #[cfg(test)] mod tests {
        use super::*;
        /// One recorded command: its grid id and its table bytes.
        type Call=(u32,usize);
        /// The recorder's cutting, over repeated steps. `next` is the area test. Returns each step's piece keys.
        fn keys(steps:usize,step:&[Call],waits:&[usize],mut next:impl FnMut(usize,usize)->bool,mut flushed:impl FnMut(usize))->Vec<Vec<Vec<u32>>> {
            let mut out=Vec::new();let (mut piece,mut staged)=(Vec::new(),0usize);
            for _ in 0..steps {
                let mut pieces=Vec::new();
                for (at,&(grid,bytes)) in step.iter().enumerate() {
                    if next(staged,bytes) && !piece.is_empty() {pieces.push(std::mem::take(&mut piece));flushed(staged);staged=0;}
                    piece.push(grid);staged+=bytes.div_ceil(16)*16;
                    if waits.contains(&at) {pieces.push(std::mem::take(&mut piece));flushed(staged);staged=0;}
                }
                out.push(pieces);
            }out
        }
        fn new_keys_after_warmup(stream:&[Vec<Vec<u32>>])->usize {
            let mut seen=std::collections::BTreeSet::new();let mut late=0;
            for (step,pieces) in stream.iter().enumerate() {for piece in pieces {if seen.insert(piece.clone()) && step>0 {late+=1;}}}late
        }
        /// A step shaped like a plain decode: 11 latent rounds, each a wait, with real gather families.
        /// Copies has more table bytes in each gather than the join it replaces.
        fn step(table:usize)->(Vec<Call>,Vec<usize>) {
            let mut calls=Vec::new();let mut waits=Vec::new();
            for round in 0..11u32 {for job in 0..20u32 {calls.push((round*100+job,table));}waits.push(calls.len()-1);}
            (calls,waits)
        }
        /// The rule the head and 15cf49b97 used: the offset carries over and wraps.
        fn carried(calls:&[Call],waits:&[usize],steps:usize)->Vec<Vec<Vec<u32>>> {
            let base=std::cell::Cell::new(BASE);
            keys(steps,calls,waits,|staged,bytes|{
                let full=base.get()+staged+bytes>BYTES;
                if full && base.get()+bytes>BYTES {base.set(BASE);}
                full
            },|staged|base.set(base.get()+staged))
        }
        #[test] fn the_key_stream_over_63_steps_has_no_new_key_after_the_first_step() {
            for table in [96,240,1024,4096] {
                let (calls,waits)=step(table);
                let stream=keys(63,&calls,&waits,full,|_|{});
                assert_eq!(new_keys_after_warmup(&stream),0,"table {table}");
                assert!(stream.windows(2).all(|p|p[0]==p[1]),"table {table}: a step cut differently");
            }
        }
        /// The control: the carried offset makes new keys, and more of them with bigger tables.
        /// This is the captures finding: same tokens, a new cut once the offset reaches the end.
        #[test] fn the_carried_offset_makes_history_dependent_keys_and_grows_with_table_bytes() {
            let new=|table:usize|{let (calls,waits)=step(table);new_keys_after_warmup(&carried(&calls,&waits,63))};
            assert!(new(1024)>0,"the control must fail");assert!(new(4096)>new(1024));
            // Same step, later in the stream: the cut moves.
            let (calls,waits)=step(4096);let stream=carried(&calls,&waits,63);assert!(stream.windows(2).any(|p|p[0]!=p[1]));
        }
        #[test] fn only_a_table_too_big_for_this_flush_forces_a_cut() {
            assert!(!full(0,BYTES-BASE));assert!(full(0,BYTES-BASE+1));assert!(!full(BYTES-BASE-16,16));assert!(full(BYTES-BASE-16,17));
        }
    }
}

/// Whether a captured piece's key was captured before. Print only: nothing reads this to decide.
#[cfg(not(target_arch="nvptx64"))]
pub mod capture_keys {
    use std::collections::BTreeSet;
    /// At most this many hashes, about 2 MiB. A long service must not grow without limit for a print.
    pub const MOST:usize=1<<16;
    #[derive(Default)]
    pub struct Seen {hashes:BTreeSet<u64>}
    /// A fixed-seed hash of the grid words (FNV-1a), the same in every process and on both ranks.
    pub fn hash(grids:impl Iterator<Item=[u32;6]>)->u64 {
        let mut h=0xcbf29ce484222325u64;
        for grid in grids {for word in grid {for byte in word.to_le_bytes() {h^=u64::from(byte);h=h.wrapping_mul(0x100000001b3);}}}h
    }
    impl Seen {
        /// True when this key is new. Past MOST, an unseen key reports new and is not kept, so a
        /// later repeat of it also reports new: captures_new_key can read high after MOST keys.
        pub fn note(&mut self,key:u64)->bool {
            if self.hashes.contains(&key) {return false;}
            if self.hashes.len()<MOST {self.hashes.insert(key);}true
        }
        pub fn len(&self)->usize {self.hashes.len()}
        pub fn is_empty(&self)->bool {self.hashes.is_empty()}
    }
    #[cfg(test)] mod tests {
        use super::*;
        use std::collections::VecDeque;
        /// The recorder's capture path in small: a limited table, oldest out, one count per capture.
        struct Table {live:VecDeque<u64>,limit:usize,seen:Seen,new:u64,again:u64,captures:u64,evictions:u64}
        impl Table {
            fn new(limit:usize)->Self {Self {live:VecDeque::new(),limit,seen:Seen::default(),new:0,again:0,captures:0,evictions:0}}
            fn replay(&mut self,key:u64) {
                if let Some(at)=self.live.iter().position(|k|*k==key) {let k=self.live.remove(at).unwrap();self.live.push_back(k);return;}
                if self.live.len()>=self.limit {self.live.pop_front();self.evictions+=1;}
                self.live.push_back(key);self.captures+=1;
                if self.seen.note(key) {self.new+=1;} else {self.again+=1;}
            }
        }
        #[test] fn the_counts_sum_to_the_captures_and_separate_new_keys_from_evicted_ones() {
            // A cycle larger than the table: the first pass is all new, later passes are all seen again.
            let mut t=Table::new(4);for _ in 0..3 {for key in 0..6 {t.replay(key);}}
            assert_eq!((t.captures,t.new,t.again),(18,6,12));assert_eq!(t.new+t.again,t.captures);
            // Keys that never repeat: all new, and the evictions alone could not tell these cases apart.
            let mut u=Table::new(4);for key in 0..18 {u.replay(key);}
            assert_eq!((u.captures,u.new,u.again),(18,18,0));assert_eq!(u.new+u.again,u.captures);assert_eq!(u.evictions,t.evictions);
            // Everything fits: only the first pass captures.
            let mut v=Table::new(8);for _ in 0..3 {for key in 0..6 {v.replay(key);}}assert_eq!((v.captures,v.new,v.again),(6,6,0));
        }
        #[test] fn the_set_stops_growing_at_its_bound_and_the_counts_still_sum() {
            let mut t=Table::new(4);for key in 0..(MOST as u64+10) {t.replay(key);}
            assert_eq!(t.seen.len(),MOST);assert_eq!(t.new+t.again,t.captures);assert_eq!(t.again,0);
        }
        #[test] fn the_hash_is_fixed_and_sees_every_field_and_the_order() {
            let a=[[1,2,3,4,5,6],[7,8,9,10,11,12]];
            assert_eq!(hash(a.into_iter()),hash(a.into_iter()));
            assert_ne!(hash(a.into_iter()),hash(a.into_iter().rev()));
            for field in 0..6 {let mut b=a;b[1][field]+=1;assert_ne!(hash(a.into_iter()),hash(b.into_iter()));}
            assert_ne!(hash(a.into_iter()),hash(a[..1].iter().copied()));
        }
    }
}
