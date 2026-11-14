//! Numeric field serialization preserves every original ABI field and zeroes padding.
use crate::resident_launch::{Command,Kind};
macro_rules! fields {($c:ident,$t:ty,$p:expr,$($f:ident),+)=>{{let p=$p;$( $c.put(core::mem::offset_of!($t,$f),&p.$f.to_ne_bytes()); )+}}}
pub(super) fn causal(p:crate::causal_ops::Arguments)->Command {
    use crate::causal_ops::{Arguments,Shape};let mut c=Command::new(Kind::Causal);
    for (i,v) in p.input.into_iter().enumerate(){c.word(i,v);}fields!(c,Arguments,p,weight,output);
    let base=core::mem::offset_of!(Arguments,shape);let mut s=Command::new(Kind::Causal);
    s.put(core::mem::offset_of!(Shape,op),&(p.shape.op as u32).to_ne_bytes());fields!(s,Shape,p.shape,heads,dim,taps,rows,cols,epsilon,bound);
    c.put(base,&s.payload[..core::mem::size_of::<Shape>()]);c
}
fn expert_arguments(p:crate::expert_ops::Arguments)->Command {
    use crate::expert_ops::{Arguments,Shape};let mut c=Command::new(Kind::Expert);
    for (i,v) in p.input.into_iter().enumerate(){c.word(i,v);}fields!(c,Arguments,p,weight,scales,output);
    let base=core::mem::offset_of!(Arguments,shape);let mut s=Command::new(Kind::Expert);
    s.put(core::mem::offset_of!(Shape,op),&(p.shape.op as u32).to_ne_bytes());fields!(s,Shape,p.shape,rows,cols,block_rows,block_cols,tokens,global,activation,bound,role);
    c.put(base,&s.payload[..core::mem::size_of::<Shape>()]);c
}
pub(super) fn expert(p:crate::expert_ops::LaunchArguments)->Command {let mut c=expert_arguments(p.single);fields!(c,crate::expert_ops::LaunchArguments,p,batch,count);c}
pub(super) fn causal_fast(p:crate::causal_ops::Arguments)->Command {let mut c=causal(p);c.kind=Kind::CausalFast as u32;c}
pub(super) fn nv_stream(p:crate::expert_ops::LaunchArguments)->Command {let mut c=expert(p);c.kind=Kind::NvStream as u32;c}
pub(super) fn nv_tile(p:crate::expert_ops::LaunchArguments)->Command {let mut c=expert(p);c.kind=Kind::NvTile as u32;c}
/// The pair split's raw quarter launch: the expert payload, and the quarters a row gets in the word after it.
pub(super) fn nv_quarters(p:crate::expert_ops::LaunchArguments,quarters:u64)->Command {
    let mut c=expert(p);c.kind=Kind::NvQuarters as u32;c.word(crate::resident_launch::NV_QUARTERS_WORD,quarters);c
}
pub(super) fn quarter_fold(p:crate::pair_split::QuarterFold)->Command {
    let mut c=Command::new(Kind::QuarterFold);fields!(c,crate::pair_split::QuarterFold,p,ids,weights,scales,acc,shared,output,experts,rows,quarters,flags,pool);c
}
pub(super) fn device_route(p:crate::device_route::Build)->Command {
    let mut c=Command::new(Kind::DeviceRoute);fields!(c,crate::device_route::Build,p,table,ids,input,quantized,quantizers,projects,output,experts,picks,phases,down,shared,raw);c
}
pub(super) fn mx(p:crate::next_params::MxLayer)->Command {
    use crate::next_params::{MxLayer,MxProjection};let mut c=Command::new(Kind::Mx);
    for (i,p) in p.projections.into_iter().enumerate(){let mut s=Command::new(Kind::Mx);fields!(s,MxProjection,p,weights,scales,input,output,rows,cols,tokens,input_bf16,round);c.put(i*core::mem::size_of::<MxProjection>(),&s.payload[..core::mem::size_of::<MxProjection>()]);}
    for (i,v) in p.ends.into_iter().enumerate(){c.put(core::mem::offset_of!(MxLayer,ends)+i*4,&v.to_ne_bytes());}
    fields!(c,MxLayer,p,count);c
}
pub(super) fn mixed(p:crate::next_params::ResidentProjectionBatch,kind:Kind)->Command {
    let mut c=mx(p.layer);c.kind=kind as u32;fields!(c,crate::next_params::ResidentProjectionBatch,p,plain,round);c
}
pub(super) fn prepare(p:crate::mhc_params::Prepare)->Command {
    use crate::mhc_params::{Prepare,Shape};let mut c=Command::new(Kind::Prepare);fields!(c,Prepare,p,input,weights,base,scale,normalized,coefficients,collapsed);
    let mut s=Command::new(Kind::Prepare);fields!(s,Shape,p.shape,rows,streams,hidden,iterations,rms_eps,eps);
    c.put(core::mem::offset_of!(Prepare,shape),&s.payload[..core::mem::size_of::<Shape>()]);c
}
pub(super) fn round(p:crate::mhc_boundary::LaneCast)->Command {let mut c=Command::new(Kind::Round);fields!(c,crate::mhc_boundary::LaneCast,p,input,output,words);c}
pub(super) fn boundary(p:crate::mhc_params::Boundary,kind:Kind)->Command {
    use crate::mhc_params::{Boundary,Shape};let mut c=Command::new(kind);
    fields!(c,Boundary,p,input,previous_input,previous_coefficients,residual,weights,base,scale,mixing_normalized,coefficients,collapsed,
        norm_weight,norm_partials,output,norm_epsilon,norm_bf16,prefetch,prefetch_bytes,prefetch_policy,prefetch_reserved);
    let mut s=Command::new(kind);fields!(s,Shape,p.shape,rows,streams,hidden,iterations,rms_eps,eps);
    c.put(core::mem::offset_of!(Boundary,shape),&s.payload[..core::mem::size_of::<Shape>()]);c
}
/// D1: the split chain's Boundary payload, then the strip copy, the scratch plane and the weight form.
pub(super) fn fused(p:crate::mhc_params::FusedBoundary)->Command {
    use crate::mhc_params::FusedBoundary;let mut c=boundary(p.boundary,Kind::MhcBoundaryFused);
    fields!(c,FusedBoundary,p,strips,scratch,bf16,reserved);c
}
pub(super) fn collapse(p:crate::mhc_params::Collapse)->Command {let mut c=Command::new(Kind::Collapse);fields!(c,crate::mhc_params::Collapse,p,input,coefficients,output,rows,streams,hidden);c}
pub(super) fn combine(p:crate::mhc_params::Combine)->Command {let mut c=Command::new(Kind::Combine);fields!(c,crate::mhc_params::Combine,p,input,coefficients,block,output,rows,streams,hidden);c}
pub(super) fn rms(p:crate::kernel_params::ModelRms,kind:Kind)->Command {let mut c=Command::new(kind);fields!(c,crate::kernel_params::ModelRms,p,input,weight,partials,output,width,rows,epsilon);c}
pub(super) fn elementwise(p:crate::kernel_params::ModelElementwise)->Command {let mut c=Command::new(Kind::Elementwise);fields!(c,crate::kernel_params::ModelElementwise,p,input,peer,output,elements,operation,scalar);c}
pub(super) fn copy(p:crate::kernel_params::HybridCopy)->Command {let mut c=Command::new(Kind::Copy);c.word(0,p.source as u64);c.word(1,p.destination as u64);fields!(c,crate::kernel_params::HybridCopy,p,words,padding,rows);c}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn boundary_replay_keeps_deferred_planes_shape_and_shadow_bits() {
        use crate::mhc_params::{Boundary,Shape};
        let shape=Shape {rows:1,streams:4,hidden:4096,iterations:20,rms_eps:1e-6,eps:1e-5};
        let p=Boundary {input:16,previous_input:32,previous_coefficients:48,residual:64,weights:80,base:96,scale:112,
            mixing_normalized:128,coefficients:144,collapsed:160,norm_weight:176,norm_partials:192,output:208,shape,
            norm_epsilon:-0.0,norm_bf16:1,prefetch:224,prefetch_bytes:7<<20,prefetch_policy:1,prefetch_reserved:0};
        for kind in [Kind::MhcBoundaryRms,Kind::MhcBoundaryDots,Kind::MhcBoundaryFinish] {
            let c=boundary(p,kind);assert_eq!(c.kind,kind as u32);assert_eq!(c.reserved,[0;3]);
            let planes=(1..=13u64).flat_map(|i|(i*16).to_ne_bytes()).collect::<Vec<_>>();assert_eq!(&c.payload[..104],planes);
            let shape=[1u32,4,4096,20,1e-6f32.to_bits(),1e-5f32.to_bits()].into_iter().flat_map(u32::to_ne_bytes).collect::<Vec<_>>();
            assert_eq!(&c.payload[104..128],shape);assert_eq!(&c.payload[128..136],&[0u8,0,0,128,1,0,0,0]);
            assert_eq!(&c.payload[136..144],&224u64.to_ne_bytes());assert_eq!(&c.payload[144..152],&(7u64<<20).to_ne_bytes());
            assert_eq!(&c.payload[152..156],&1u32.to_ne_bytes());assert!(c.payload[156..].iter().all(|&b|b==0));
        }
    }
    #[test] fn device_route_replay_preserves_addresses_bounds_and_zero_padding() {
        use crate::device_route::Build;
        let p=Build {table:16,ids:32,input:48,quantized:64,quantizers:80,projects:96,output:112,experts:1024,picks:8,phases:2,down:0,shared:1,raw:0};
        let command=device_route(p);assert_eq!(command.kind,21);assert_eq!(Kind::DeviceRoute.kernel(),"resident_launch_device_route");
        for (word,address) in [16u64,32,48,64,80,96,112].into_iter().enumerate() {assert_eq!(&command.payload[word*8..word*8+8],&address.to_ne_bytes());}
        for (word,value) in [1024u32,8,2,0,1].into_iter().enumerate() {assert_eq!(&command.payload[56+word*4..60+word*4],&value.to_ne_bytes());}
        assert!(command.payload[76..].iter().all(|&byte|byte==0));assert_eq!(command.reserved,[0;3]);
        // The pair split's raw word is the sixth 32-bit field; nothing follows it.
        let raw=device_route(Build {down:1,phases:1,raw:2,..p});
        assert_eq!(&raw.payload[76..80],&2u32.to_ne_bytes());assert!(raw.payload[80..].iter().all(|&byte|byte==0));
    }
    #[test] fn nv_stream_replay_keeps_the_original_expert_payload() {
        use crate::expert_ops::{Arguments,LaunchArguments,Shape,Op};
        let shape=Shape {op:Op::NvProject,rows:1024,cols:4096,block_rows:0,block_cols:0,tokens:1,global:1.0,activation:2.0,bound:0.0,role:0};
        let p=LaunchArguments {single:Arguments {input:[16,32,48,64],weight:80,scales:96,output:112,shape},batch:128,count:16};
        let old=expert(p);let new=nv_stream(p);
        assert_eq!(new.kind,Kind::NvStream as u32);assert_eq!(new.payload,old.payload);assert_eq!(new.reserved,[0;3]);
        assert_eq!(Kind::NvStream.kernel(),"resident_launch_nv_stream");
    }
    #[test] fn causal_fast_replay_keeps_the_causal_payload() {
        use crate::causal_ops::{Arguments,Shape,Op};
        let shape=Shape {op:Op::Softmax,heads:32,dim:128,taps:0,rows:2051,cols:0,epsilon:0.0,bound:0.0};
        let p=Arguments {input:[16,32,0,0],weight:4,output:64,shape};
        let old=causal(p);let new=causal_fast(p);
        assert_eq!(new.kind,Kind::CausalFast as u32);assert_eq!(new.payload,old.payload);assert_eq!(new.reserved,[0;3]);
        assert_eq!(Kind::CausalFast.kernel(),"resident_launch_causal_fast");
        assert_eq!(core::mem::size_of::<Arguments>(),80);
        let alignment=core::mem::align_of::<Arguments>();
        assert_eq!(core::mem::offset_of!(Command,payload)%alignment,0);
        assert_eq!(core::mem::size_of::<Command>()%alignment,0);
        assert_eq!(new.payload.as_ptr() as usize%alignment,0);
        for (index,value) in [16u64,32,0,0,4,64].into_iter().enumerate() {assert_eq!(u64_at(&new,index*8),value);}
        for (index,value) in [Op::Softmax as u32,32,128,0,2051,0,0,0].into_iter().enumerate() {assert_eq!(u32_at(&new,48+index*4),value);}
        assert!(new.payload[80..].iter().all(|&byte|byte==0));
    }
    #[test] fn pair_split_replays_keep_every_field_and_the_quarter_word() {
        use crate::expert_ops::{Arguments,LaunchArguments,Shape,Op};use crate::pair_split::QuarterFold;use core::mem::size_of;
        let shape=Shape {op:Op::NvProject,rows:4096,cols:1024,block_rows:crate::expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:1.0,activation:2.0,bound:0.0,role:crate::expert_ops::NV_SECOND};
        let p=LaunchArguments {single:Arguments {input:[16,32,48,64],weight:80,scales:96,output:112,shape},batch:128,count:8};
        assert_eq!((size_of::<LaunchArguments>(),crate::resident_launch::NV_QUARTERS_WORD*8),(112,112));
        for quarters in [4u64,2,1] {
            let (old,new)=(expert(p),nv_quarters(p,quarters));
            assert_eq!((new.kind,new.reserved),(Kind::NvQuarters as u32,[0;3]));assert_eq!(new.payload[..112],old.payload[..112]);
            assert_eq!(u64_at(&new,112),quarters);assert!(new.payload[120..].iter().all(|&byte|byte==0));
        }
        let fold=quarter_fold(QuarterFold {ids:16,weights:32,scales:48,acc:64,shared:80,output:96,experts:8,rows:4096,quarters:2,flags:13,pool:288});
        assert_eq!((fold.kind,fold.reserved),(Kind::QuarterFold as u32,[0;3]));
        for (word,address) in [16u64,32,48,64,80,96].into_iter().enumerate() {assert_eq!(u64_at(&fold,word*8),address);}
        for (word,value) in [8u32,4096,2,13,288].into_iter().enumerate() {assert_eq!(u32_at(&fold,48+word*4),value);}
        assert!(fold.payload[68..].iter().all(|&byte|byte==0));
        assert_eq!((Kind::NvQuarters as u32,Kind::QuarterFold as u32,Kind::ReduceTake as u32),(39,40,41));
        assert_eq!([Kind::NvQuarters,Kind::QuarterFold,Kind::ReduceTake].map(Kind::kernel),["resident_launch_nv_quarters","resident_launch_quarter_fold","resident_launch_reduce_take"]);
    }
    #[test] fn nv_tile_replay_keeps_the_original_expert_payload() {
        use crate::expert_ops::{Arguments,LaunchArguments,Shape,Op};
        let shape=Shape {op:Op::NvProject,rows:1024,cols:4096,block_rows:0,block_cols:0,tokens:1,global:1.0,activation:2.0,bound:0.0,role:0};
        let p=LaunchArguments {single:Arguments {input:[16,32,48,64],weight:80,scales:96,output:112,shape},batch:128,count:16};
        let old=expert(p);let new=nv_tile(p);
        assert_eq!(Kind::NvStream as u32,18);assert_eq!(Kind::CausalFast as u32,19);assert_eq!(Kind::NvTile as u32,20);
        assert_eq!(new.kind,Kind::NvTile as u32);assert_eq!(new.payload,old.payload);assert_eq!(new.reserved,[0;3]);
        assert_eq!(Kind::NvTile.kernel(),"resident_launch_nv_tile");
    }
    fn u64_at(c:&Command,at:usize)->u64 {u64::from_ne_bytes(c.payload[at..at+8].try_into().unwrap())}
    fn u32_at(c:&Command,at:usize)->u32 {u32::from_ne_bytes(c.payload[at..at+4].try_into().unwrap())}
    #[test] fn mixed_projection_abi_keeps_all_descriptors_and_flags() {
        use crate::next_params::{MxProjection,MxLayer,ResidentProjectionBatch};use core::mem::{offset_of,size_of};
        let mut p=ResidentProjectionBatch {round:1,..Default::default()};
        for i in 0..9 {p.push(MxProjection {weights:16*(i+1),scales:32,input:48,output:64,rows:(i+1) as u32,cols:32,tokens:1,input_bf16:0,round:0},i%2==0).unwrap();}
        for kind in [Kind::Mixed,Kind::MixedBf16,Kind::PlainBatch] {
        let c=mixed(p,kind);assert_eq!(c.kind,kind as u32);assert_eq!(c.reserved,[0;3]);assert_eq!(offset_of!(ResidentProjectionBatch,layer),0);
        for (i,v) in p.layer.projections.iter().enumerate() {let at=i*size_of::<MxProjection>();assert_eq!(u64_at(&c,at),v.weights);assert_eq!(u64_at(&c,at+24),v.output);assert_eq!(u32_at(&c,at+40),v.tokens);assert_eq!(u32_at(&c,at+52),0);}
        for (i,v) in p.layer.ends.iter().enumerate() {assert_eq!(u32_at(&c,offset_of!(MxLayer,ends)+i*4),*v);}
        assert_eq!(u32_at(&c,offset_of!(MxLayer,count)),9);assert_eq!(u32_at(&c,offset_of!(ResidentProjectionBatch,plain)),p.plain);assert_eq!(u32_at(&c,offset_of!(ResidentProjectionBatch,round)),1);
        assert!(c.payload[size_of::<ResidentProjectionBatch>()..].iter().all(|&v|v==0));}
    }
    #[test] fn original_numeric_abis_and_zero_padding_are_serialized_explicitly() {
        use core::mem::{offset_of,size_of};
        let shape=crate::causal_ops::Shape {op:crate::causal_ops::Op::Step,heads:3,dim:7,taps:4,rows:2,cols:6,epsilon:-0.0,bound:-1.0};
        let p=crate::causal_ops::Arguments {input:[16,32,48,64],weight:80,output:96,shape};let c=causal(p);
        assert_eq!(u64_at(&c,offset_of!(crate::causal_ops::Arguments,output)),96);assert_eq!(u64_at(&c,24),64);
        let at=offset_of!(crate::causal_ops::Arguments,shape);assert_eq!(u32_at(&c,at),shape.op as u32);assert_eq!(u32_at(&c,at+24),(-0.0f32).to_bits());
        assert!(c.payload[size_of::<crate::causal_ops::Arguments>()..].iter().all(|&v|v==0));
        let p=crate::expert_ops::Arguments {input:[16,32,48,64],weight:80,scales:96,output:112,shape:crate::expert_ops::Shape {op:crate::expert_ops::Op::NvProject,rows:3,cols:16,block_rows:1,block_cols:16,tokens:1,global:0.5,activation:2.0,bound:-0.0,role:0}};
        let c=expert(crate::expert_ops::LaunchArguments {single:p,batch:128,count:3});assert_eq!(u64_at(&c,offset_of!(crate::expert_ops::Arguments,output)),112);assert_eq!(u32_at(&c,offset_of!(crate::expert_ops::Arguments,shape)+32),(-0.0f32).to_bits());
        assert_eq!(u64_at(&c,offset_of!(crate::expert_ops::LaunchArguments,batch)),128);assert_eq!(u32_at(&c,offset_of!(crate::expert_ops::LaunchArguments,count)),3);
        assert_eq!(u32_at(&c,offset_of!(crate::expert_ops::Arguments,shape)+offset_of!(crate::expert_ops::Shape,tokens)),1);
        let mut p=crate::next_params::MxLayer::default();p.push(crate::next_params::MxProjection {weights:16,scales:32,input:48,output:64,rows:3,cols:32,tokens:1,input_bf16:0,round:0}).unwrap();let c=mx(p);
        assert_eq!(u64_at(&c,24),64);assert_eq!(u32_at(&c,offset_of!(crate::next_params::MxLayer,count)),1);
        for (i,v) in p.ends.iter().enumerate(){assert_eq!(u32_at(&c,offset_of!(crate::next_params::MxLayer,ends)+i*4),*v);}
        let p=crate::mhc_params::Prepare {input:16,weights:32,base:48,scale:64,normalized:80,coefficients:96,collapsed:112,shape:crate::mhc_params::Shape {rows:1,streams:2,hidden:3,iterations:4,rms_eps:1e-6,eps:1e-5}};let c=prepare(p);
        assert_eq!(u64_at(&c,40),96);assert_eq!(u64_at(&c,offset_of!(crate::mhc_params::Prepare,collapsed)),112);
        assert_eq!(u32_at(&c,offset_of!(crate::mhc_params::Prepare,shape)+offset_of!(crate::mhc_params::Shape,iterations)),4);assert!(c.payload[size_of::<crate::mhc_params::Prepare>()..].iter().all(|&v|v==0));
        let c=round(crate::mhc_boundary::LaneCast {input:16,output:32,words:9});assert_eq!(u64_at(&c,8),32);assert_eq!(u32_at(&c,16),9);
        let c=collapse(crate::mhc_params::Collapse {input:16,coefficients:32,output:48,rows:1,streams:2,hidden:3});assert_eq!(u32_at(&c,32),3);assert!(c.payload[36..40].iter().all(|&v|v==0));
        let c=combine(crate::mhc_params::Combine {input:16,coefficients:32,block:48,output:64,rows:1,streams:2,hidden:3});assert_eq!(u64_at(&c,24),64);assert_eq!(u32_at(&c,40),3);
        let c=rms(crate::kernel_params::ModelRms {input:16,weight:32,partials:48,output:64,width:5,rows:1,epsilon:-0.0},Kind::RmsApply);assert_eq!(u32_at(&c,40),(-0.0f32).to_bits());assert_eq!(u32_at(&c,44),0);
        let c=elementwise(crate::kernel_params::ModelElementwise {input:16,peer:32,output:48,elements:5,operation:4,scalar:-0.0});assert_eq!(u32_at(&c,32),(-0.0f32).to_bits());
        let c=copy(crate::kernel_params::HybridCopy {source:16 as _,destination:32 as _,words:5,padding:0,rows:48});assert_eq!(u64_at(&c,0),16);assert_eq!(u64_at(&c,8),32);assert_eq!(u64_at(&c,24),48);
    }
}
