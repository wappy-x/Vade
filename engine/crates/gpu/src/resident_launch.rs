//! Stable indirect launch ABI. Padding is initialized and never copied from Rust structs.
#[repr(u32)]
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
pub enum Kind {Causal=1,Expert,Plain,Mx,Round,Copy,Put,Take,Prepare,Collapse,Combine,RmsPartials,RmsApply,Elementwise,CopyWords,ZeroWords}
#[cfg(not(target_arch="nvptx64"))]
impl Kind {
    pub fn kernel(self)->&'static str {match self {
        Self::Causal=>"resident_launch_causal",Self::Expert=>"resident_launch_expert",Self::Plain=>"resident_launch_plain",Self::Mx=>"resident_launch_mx",
        Self::Round=>"resident_launch_round",Self::Copy=>"resident_launch_copy",Self::Put=>"resident_launch_put",Self::Take=>"resident_launch_take",
        Self::Prepare=>"resident_launch_prepare",Self::Collapse=>"resident_launch_collapse",Self::Combine=>"resident_launch_combine",
        Self::RmsPartials=>"resident_launch_rms_partials",Self::RmsApply=>"resident_launch_rms_apply",Self::Elementwise=>"resident_launch_elementwise",
        Self::CopyWords=>"resident_launch_copy_words",Self::ZeroWords=>"resident_launch_zero_words",
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
        fits!(crate::causal_ops::Arguments,crate::expert_ops::Arguments,crate::expert_ops::LaunchArguments,crate::next_params::MxLayer,
            crate::mhc_boundary::LaneCast,crate::kernel_params::HybridCopy,crate::mhc_params::Prepare,
            crate::mhc_params::Collapse,crate::mhc_params::Combine,crate::kernel_params::ModelRms,crate::kernel_params::ModelElementwise);
    }
}
