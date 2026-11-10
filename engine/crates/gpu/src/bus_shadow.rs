//! Prefetch metadata for idle tail rows of a recorded scatter receive.
#[cfg(not(target_arch="nvptx64"))]
use crate::resident_launch::{Command,Kind};
pub const OFFSET:usize=768;
#[cfg(not(target_arch="nvptx64"))]
pub const BUDGET:u64=8<<20;
#[cfg(not(target_arch="nvptx64"))]
pub const BLOCKS:u32=8;
#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub struct Window {pub weight:u64,pub bytes:u64,pub ids:u64,pub experts:u32,pub picks:u32}
#[cfg(not(target_arch="nvptx64"))]
fn word(bytes:&[u8],at:usize)->Option<u64> {Some(u64::from_ne_bytes(bytes.get(at..at+8)?.try_into().ok()?))}
#[cfg(not(target_arch="nvptx64"))]
fn number(bytes:&[u8],at:usize)->Option<u32> {Some(u32::from_ne_bytes(bytes.get(at..at+4)?.try_into().ok()?))}
#[cfg(not(target_arch="nvptx64"))]
pub fn mx(command:&Command)->Option<Window> {
    use crate::next_params::{MxLayer,MxProjection};
    if command.kind!=Kind::Mx as u32 || number(&command.payload,core::mem::offset_of!(MxLayer,count))?!=1
        || number(&command.payload,core::mem::offset_of!(MxProjection,tokens))?!=1 {return None;}
    let rows=number(&command.payload,core::mem::offset_of!(MxProjection,rows))?;
    let cols=number(&command.payload,core::mem::offset_of!(MxProjection,cols))?;
    let bytes=u64::from(rows).checked_mul(u64::from(cols))?.min(BUDGET);
    (bytes!=0).then_some(Window {weight:word(&command.payload,0)?,bytes,ids:0,experts:0,picks:0})
}
#[cfg(not(target_arch="nvptx64"))]
pub fn prior_route(command:&Command,window:Window)->bool {
    use crate::device_route::Build;
    command.kind==Kind::DeviceRoute as u32
        && word(&command.payload,core::mem::offset_of!(Build,ids))==Some(window.ids)
        && number(&command.payload,core::mem::offset_of!(Build,experts))==Some(window.experts)
        && number(&command.payload,core::mem::offset_of!(Build,picks))==Some(window.picks)
}
#[cfg(not(target_arch="nvptx64"))]
pub fn arm(command:&mut Command,grid:&mut [u32;5],window:Window,stage_base:u64,staged:&[u8])->bool {
    if command.kind!=Kind::TakeScatter as u32 || command.reserved!=[0;3] || grid[0]==0 || grid[1]==0 || grid[2]!=1
        || grid[3]!=256 || window.weight==0 || window.weight%8!=0 || window.bytes==0 || window.bytes>BUDGET {return false;}
    let Some(rows)=grid[1].checked_add(BLOCKS.div_ceil(grid[0])).filter(|&rows|rows<=65535) else {return false;};
    if window.ids!=0 {
        if window.picks==0 || window.experts==0 || window.picks>window.experts || window.ids%4!=0 {return false;}
        let Some(end)=window.ids.checked_add(u64::from(window.picks)*4) else {return false;};
        let Some(offset)=word(&command.payload,0).and_then(|address|address.checked_sub(stage_base)).and_then(|at|usize::try_from(at).ok()) else {return false;};
        let Some(table)=staged.get(offset..).and_then(|bytes|bytes.get(..grid[1] as usize*24)) else {return false;};
        for row in table.chunks_exact(24) {
            let Some(target)=word(row,8) else {return false;};let Some(words)=number(row,16) else {return false;};
            let Some(target_end)=target.checked_add(u64::from(words)*4) else {return false;};
            if target<end && window.ids<target_end {return false;}
        }
    } else if window.experts!=0 || window.picks!=0 {return false;}
    command.reserved=[grid[1],u32::from(window.ids!=0)+1,0];
    for (index,value) in [window.weight,window.bytes,window.ids].into_iter().enumerate() {command.put(OFFSET+index*8,&value.to_ne_bytes());}
    command.put(OFFSET+24,&window.experts.to_ne_bytes());command.put(OFFSET+28,&window.picks.to_ne_bytes());
    grid[1]=rows;true
}
#[cfg(not(target_arch="nvptx64"))]
static ARMED:[std::sync::atomic::AtomicU64;2]=[const {std::sync::atomic::AtomicU64::new(0)};2];
#[cfg(not(target_arch="nvptx64"))]
pub fn record(routed:bool) {ARMED[usize::from(routed)].fetch_add(1,std::sync::atomic::Ordering::Relaxed);}
#[cfg(not(target_arch="nvptx64"))]
pub fn counts()->[u64;2] {ARMED.each_ref().map(|value|value.load(std::sync::atomic::Ordering::Relaxed))}
const _:()={assert!(OFFSET%16==0);assert!(OFFSET+core::mem::size_of::<Window>()<=1024);};
#[cfg(test)] mod tests {
    use super::*;
    fn command()->Command {let mut value=Command::new(Kind::TakeScatter);value.word(0,0x1000);value}
    fn window()->Window {Window {weight:0x8000,bytes:4096,ids:0,experts:0,picks:0}}
    #[test] fn only_tail_rows_change_the_grid_and_original_payload() {
        for blocks in [1,2,3,8,32,96] {for rows in [1,2,17,127] {
            let mut call=command();let original=call.payload;let mut grid=[blocks,rows,1,256,0];
            assert!(arm(&mut call,&mut grid,window(),0,&[]));assert_eq!(call.payload[..OFFSET],original[..OFFSET]);
            assert_eq!([grid[0],grid[2],grid[3],grid[4]],[blocks,1,256,0]);assert_eq!(call.reserved[0],rows);
            assert!(grid[1]>rows);let tails=(grid[1]-rows)*blocks;assert!(tails>=BLOCKS && tails<BLOCKS+blocks);
            assert!(!arm(&mut call,&mut grid,window(),0,&[]));
        }}
    }
    #[test] fn routed_ids_must_not_overlap_any_receive_row() {
        let plan=crate::scatter_copy::Plan::checked(64,64,&[[0,0,8],[8,16,8]]).unwrap();
        let staged=plan.encode(0x4000,0x2000).unwrap();
        for ids in [0x1fe4,0x2000,0x201c,0x2024,0x2040,0x205c] {
            let mut call=command();let mut grid=[1,2,1,256,0];let target=Window {ids,experts:16,picks:8,..window()};
            assert!(!arm(&mut call,&mut grid,target,0x1000,&staged),"ids={ids:x}");assert_eq!(grid[1],2);assert_eq!(call.reserved,[0;3]);
        }
        for ids in [0x1fe0,0x2020,0x2060,0x3000] {
            let mut call=command();let mut grid=[1,2,1,256,0];
            assert!(arm(&mut call,&mut grid,Window {ids,experts:16,picks:8,..window()},0x1000,&staged));
        }
    }
    #[test] fn invalid_spans_and_grids_are_refused_without_mutation() {
        for grid in [[0,1,1,256,0],[1,65535,1,256,0],[1,1,2,256,0],[1,1,1,128,0]] {
            let mut call=command();let mut actual=grid;assert!(!arm(&mut call,&mut actual,window(),0,&[]));assert_eq!(actual,grid);
        }
        for target in [Window {weight:1,..window()},Window {bytes:0,..window()},Window {bytes:BUDGET+1,..window()},Window {ids:u64::MAX-3,picks:8,experts:16,..window()}] {
            assert!(!arm(&mut command(),&mut [1,1,1,256,0],target,0,&[]));
        }
        assert!(!arm(&mut command(),&mut [1,1,1,256,0],Window {ids:0x2000,picks:8,experts:16,..window()},0x1001,&[0;24]));
    }
    #[test] fn tail_lines_cover_the_exact_bounded_prefetch_set() {
        for picks in [1u64,8] {for bytes in [1u64,127,128,129,1<<20] {for tails in [8u64,9,32] {
            let lines=crate::mhc_params::shadow_lines(bytes);let mut visits=vec![0u8;(lines*picks) as usize];
            for thread in 0..tails*256 {let mut line=thread;while line<lines*picks {assert!((line%lines)*128<bytes);visits[line as usize]+=1;line+=tails*256;}}
            assert!(visits.iter().all(|&visits|visits==1));
        }}}
    }
    #[test] fn route_readiness_matches_the_same_initialized_id_owner() {
        use crate::device_route::Build;
        let mut prior=Command::new(Kind::DeviceRoute);let target=Window {ids:0x2000,experts:16,picks:8,..window()};
        prior.put(core::mem::offset_of!(Build,ids),&target.ids.to_ne_bytes());prior.put(core::mem::offset_of!(Build,experts),&16u32.to_ne_bytes());prior.put(core::mem::offset_of!(Build,picks),&8u32.to_ne_bytes());
        assert!(prior_route(&prior,target));assert!(!prior_route(&prior,Window {ids:0x3000,..target}));
        assert!(!prior_route(&prior,Window {experts:15,..target}));assert!(!prior_route(&command(),target));
    }
    #[test] fn static_target_is_only_one_decode_projection_and_is_capped() {
        use crate::next_params::{MxLayer,MxProjection};let mut target=Command::new(Kind::Mx);
        target.put(0,&0x8000u64.to_ne_bytes());target.put(core::mem::offset_of!(MxProjection,rows),&4096u32.to_ne_bytes());
        target.put(core::mem::offset_of!(MxProjection,cols),&4096u32.to_ne_bytes());target.put(core::mem::offset_of!(MxProjection,tokens),&1u32.to_ne_bytes());
        target.put(core::mem::offset_of!(MxLayer,count),&1u32.to_ne_bytes());assert_eq!(mx(&target).unwrap().bytes,BUDGET);
        target.put(core::mem::offset_of!(MxProjection,tokens),&2u32.to_ne_bytes());assert!(mx(&target).is_none());
        target.put(core::mem::offset_of!(MxProjection,tokens),&1u32.to_ne_bytes());target.put(core::mem::offset_of!(MxLayer,count),&2u32.to_ne_bytes());assert!(mx(&target).is_none());
    }
}
