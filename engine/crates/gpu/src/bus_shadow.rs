//! Prefetch metadata for idle tail rows of a recorded scatter receive.
#[cfg(not(target_arch="nvptx64"))]
use crate::resident_launch::{Command,Kind};
pub const OFFSET:usize=768;
#[cfg(not(target_arch="nvptx64"))]
/// Bytes of weight a recorded take's idle tail rows prefetch for the next stage (both window families).
/// 6 MB. At rc.9 (retest-take-tail, pair one) 4 MB was exact and faster than 8 MB at every cell, and 0.2 to 0.3 ms
/// faster than 2 MB at c4158. Since the mixed BF16 batch and the block FP8 launches read on the lookahead loaders
/// (2026-11-12) a one-chat plain step is faster with 6 MB than with 4 MB; the numbers are in the merge line.
pub const BUDGET:u64=6<<20;
#[cfg(not(target_arch="nvptx64"))]
pub const BLOCKS:u32=8;
/// Blocks of a side-branch prefetch launch (kind ShadowSide); each block is 256 threads and each thread holds one
/// 128-byte line request. The side launch runs beside the take it is placed in front of and beside the launches after
/// it, and their own reads wait behind its requests: in a traced one-row step a waiting take beside an 8-block
/// prefetch ended 13.8 us after the peer's publish, against 6.6 us with no prefetch beside it, and the mHC boundary
/// lasted 30 us against 15 us (engine/docs/message-hand-over.md). Two blocks: a served plain step of 41.147 ms against
/// 41.345 ms with eight, one run each on one pair. Prefetch only: no value can change with this number.
#[cfg(not(target_arch="nvptx64"))]
pub const SIDE_BLOCKS:u32=2;
#[cfg(not(target_arch="nvptx64"))]
const _:()=assert!(SIDE_BLOCKS>=1 && SIDE_BLOCKS<=BLOCKS);
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
/// The smallest weight span worth a side prefetch of its own (a smaller projection streams in under 10 us).
#[cfg(not(target_arch="nvptx64"))]
pub const MIN_TARGET:u64=2<<20;
/// More one-token static projections whose weights a side-branch prefetch can read ahead while the stretch in front of
/// them runs: an ordered BF16 projection, and one NVFP4 or FP8 projection job. Side form only: in a take's tail the
/// take would last as long as the prefetch. Shape and address come from the recorded command itself.
#[cfg(not(target_arch="nvptx64"))]
pub fn more(command:&Command)->Option<Window> {
    use crate::expert_ops::{Arguments,LaunchArguments,Op,Shape};
    use core::mem::offset_of;
    let (weight,bytes)=if command.kind==Kind::PlainBf16 as u32 {
        (word(&command.payload,0)?,word(&command.payload,24)?.checked_mul(word(&command.payload,32)?)?.checked_mul(2)?)
    } else if command.kind==Kind::Expert as u32 || command.kind==Kind::Fp8Tile as u32 {
        let single=offset_of!(LaunchArguments,single);let shape=single+offset_of!(Arguments,shape);
        if word(&command.payload,offset_of!(LaunchArguments,batch))?!=0 || number(&command.payload,shape+offset_of!(Shape,tokens))?!=1 {return None;}
        let cells=u64::from(number(&command.payload,shape+offset_of!(Shape,rows))?).checked_mul(u64::from(number(&command.payload,shape+offset_of!(Shape,cols))?))?;
        let op=number(&command.payload,shape+offset_of!(Shape,op))?;
        let bytes=if op==Op::NvProject as u32 {cells/2} else if op==Op::Fp8Project as u32 || op==Op::Fp8Serial as u32 {cells} else {return None;};
        (word(&command.payload,single+offset_of!(Arguments,weight))?,bytes)
    } else {return None;};
    (weight!=0 && weight%8==0 && bytes>=MIN_TARGET).then_some(Window {weight,bytes:bytes.min(BUDGET),ids:0,experts:0,picks:0})
}
/// The first projection of a one-row BF16 projection batch (the attention matrices that read a boundary's output), as a
/// side-branch window of `BUDGET` bytes at most. The batch's blocks start at this projection's first rows. Before the
/// boundary was one launch, the tails of its dots and finish launches prefetched these weights (CHAIN-SHADOW); the
/// one-launch boundary has no tail, so the batch gets a side window in front of the take that precedes that boundary.
/// Side form only, as `more`. Shape and address come from the recorded command itself; an MXFP8 weight is a byte a
/// cell, a plain BF16 weight two.
#[cfg(not(target_arch="nvptx64"))]
pub fn batch(command:&Command)->Option<Window> {
    use crate::next_params::{MxLayer,MxProjection,ResidentProjectionBatch};
    use core::mem::offset_of;
    if command.kind!=Kind::MixedBf16 as u32 || number(&command.payload,offset_of!(MxLayer,count))?==0
        || number(&command.payload,offset_of!(MxProjection,tokens))?!=1 {return None;}
    let cells=u64::from(number(&command.payload,offset_of!(MxProjection,rows))?).checked_mul(u64::from(number(&command.payload,offset_of!(MxProjection,cols))?))?;
    let bytes=if number(&command.payload,offset_of!(ResidentProjectionBatch,plain))?&1!=0 {cells.checked_mul(2)?} else {cells};
    let weight=word(&command.payload,0)?;
    (weight!=0 && weight%8==0 && bytes>=MIN_TARGET).then_some(Window {weight,bytes:bytes.min(BUDGET),ids:0,experts:0,picks:0})
}
/// Most token rows of a ROW launch that gets a side window (`rows`): the row groups of a decode step of several chats and
/// the check rows of a drafting pass. A larger group is a prompt group; it keeps its own path.
#[cfg(not(target_arch="nvptx64"))]
pub const ROW_TOKENS:u32=8;
/// ROW SHADOW (seat CHATS4): the first weights of a row launch of 2 to `ROW_TOKENS` token rows, as a side-branch window of
/// `BUDGET` bytes at most, as `more` and `batch` give a one-token launch: an MXFP8 row projection (`mx_rows_small*`,
/// `mx_rows_tile`), a mixed row stage (`flash_resident_projection_bf16_rows`) and the fold rows (`..._fold_rows`, `_f32`) by
/// their first projection, a block-FP8 row group (`fp8_rows_*`) by the first part of its table, read from the staged
/// bytes the command names. Side form only. Prefetch only: no value can change.
#[cfg(not(target_arch="nvptx64"))]
pub fn rows(command:&Command,stage_base:u64,staged:&[u8])->Option<Window> {
    use crate::expert_ops::{GroupArguments,GroupEntry,Shape};
    use crate::next_params::{MxLayer,MxProjection,ResidentProjectionBatch};
    use core::mem::offset_of;
    let kind=command.kind;let is=|k:Kind|kind==k as u32;
    let tokens_ok=|t:u32|(2..=ROW_TOKENS).contains(&t);
    let (weight,bytes)=if is(Kind::MxRows2) || is(Kind::MxRows4) || is(Kind::MxRows8) || is(Kind::MxRows8w) || is(Kind::MxRowsTile) {
        if !tokens_ok(number(&command.payload,offset_of!(MxProjection,tokens))?) {return None;}
        let cells=u64::from(number(&command.payload,offset_of!(MxProjection,rows))?).checked_mul(u64::from(number(&command.payload,offset_of!(MxProjection,cols))?))?;
        (word(&command.payload,offset_of!(MxProjection,weights))?,cells)
    } else if is(Kind::MixedRows) || is(Kind::FoldRows) || is(Kind::FoldRowsF32) {
        if number(&command.payload,offset_of!(MxLayer,count))?==0 || !tokens_ok(number(&command.payload,offset_of!(MxProjection,tokens))?) {return None;}
        let cells=u64::from(number(&command.payload,offset_of!(MxProjection,rows))?).checked_mul(u64::from(number(&command.payload,offset_of!(MxProjection,cols))?))?;
        // A word of the first projection: 4 bytes (FP32 fold rows), 2 (a plain BF16 projection) or 1 (an MXFP8 one).
        let width=if is(Kind::FoldRowsF32) {4} else if is(Kind::FoldRows) || number(&command.payload,offset_of!(ResidentProjectionBatch,plain))?&1!=0 {2} else {1};
        (word(&command.payload,0)?,cells.checked_mul(width)?)
    } else if is(Kind::Fp8RowsSmall) || is(Kind::Fp8RowsMulti) || is(Kind::Fp8RowsFew) || is(Kind::Fp8RowsShare) {
        if number(&command.payload,offset_of!(GroupArguments,count))?==0 {return None;}
        let at=usize::try_from(word(&command.payload,offset_of!(GroupArguments,table))?.checked_sub(stage_base)?).ok()?;
        let entry=staged.get(at..)?.get(..core::mem::size_of::<GroupEntry>())?;
        if !tokens_ok(number(entry,offset_of!(GroupEntry,shape)+offset_of!(Shape,tokens))?.max(2)) {return None;}
        (word(entry,offset_of!(GroupEntry,weight))?,u64::from(number(entry,offset_of!(GroupEntry,weight_bytes))?))
    } else {return None;};
    (weight!=0 && weight%8==0 && bytes>=MIN_TARGET).then_some(Window {weight,bytes:bytes.min(BUDGET),ids:0,experts:0,picks:0})
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
        || grid[3]!=256 || window.weight==0 || !window.weight.is_multiple_of(8) || window.bytes==0 || window.bytes>BUDGET {return false;}
    let Some(rows)=grid[1].checked_add(BLOCKS.div_ceil(grid[0])).filter(|&rows|rows<=65535) else {return false;};
    if window.ids!=0 {
        if window.picks==0 || window.experts==0 || window.picks>window.experts || !window.ids.is_multiple_of(4) {return false;}
        let Some(end)=window.ids.checked_add(u64::from(window.picks)*4) else {return false;};
        let Some(offset)=word(&command.payload,0).and_then(|address|address.checked_sub(stage_base)).and_then(|at|usize::try_from(at).ok()) else {return false;};
        let Some(table)=staged.get(offset..).and_then(|bytes|bytes.get(..grid[1] as usize*24)) else {return false;};
        for row in table.as_chunks::<24>().0 {
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
/// The window as a launch of its own on a side branch (recorder kind ShadowSide), for the take `take` with grid `grid`:
/// the checks are `arm`'s, on a copy of the take, so a window a tail could not take makes no side command either.
#[cfg(not(target_arch="nvptx64"))]
pub fn side(take:&Command,grid:[u32;5],window:Window,stage_base:u64,staged:&[u8])->Option<Command> {
    let (mut probe,mut dimensions)=(*take,grid);
    if !arm(&mut probe,&mut dimensions,window,stage_base,staged) {return None;}
    let mut command=Command::new(Kind::ShadowSide);
    for (index,value) in [window.weight,window.bytes,window.ids].into_iter().enumerate() {command.put(OFFSET+index*8,&value.to_ne_bytes());}
    command.put(OFFSET+24,&window.experts.to_ne_bytes());command.put(OFFSET+28,&window.picks.to_ne_bytes());Some(command)
}
/// WAIT COVER (seat DRAFTR): a static window as a side launch beside a PLAIN take (recorder kind Take), the hand-off of
/// the rows path that has no scatter table and so no tail rows. The window's checks are `arm`'s for a window with no
/// route IDs: an aligned weight, 1 to `BUDGET` bytes. A window with route IDs gets none here: its ID plane is not
/// checked against this take's rows. Prefetch only: the side launch writes nothing and no value can change.
#[cfg(not(target_arch="nvptx64"))]
pub fn side_plain(take:&Command,window:Window)->Option<Command> {
    if take.kind!=Kind::Take as u32 || window.ids!=0 || window.experts!=0 || window.picks!=0
        || window.weight==0 || !window.weight.is_multiple_of(8) || window.bytes==0 || window.bytes>BUDGET {return None;}
    let mut command=Command::new(Kind::ShadowSide);
    for (index,value) in [window.weight,window.bytes,0].into_iter().enumerate() {command.put(OFFSET+index*8,&value.to_ne_bytes());}
    command.put(OFFSET+24,&0u32.to_ne_bytes());command.put(OFFSET+28,&0u32.to_ne_bytes());Some(command)
}
#[cfg(not(target_arch="nvptx64"))]
static ARMED:[std::sync::atomic::AtomicU64;2]=[const {std::sync::atomic::AtomicU64::new(0)};2];
/// The words of a take that name its message: the address of its arrival word, then the message number. A scatter take
/// keeps them in words 1 and 2 (`tp_mailbox_take_scatter`), a plain take in words 6 and 7 (`tp_mailbox_take_shared`).
#[cfg(not(target_arch="nvptx64"))]
fn message(take:&Command)->Option<(u64,u64)> {
    let at=if take.kind==Kind::TakeScatter as u32 {8} else if take.kind==Kind::Take as u32 {48} else {return None;};
    Some((word(&take.payload,at)?,word(&take.payload,at+8)?))
}
/// Two takes of ONE exchange: one kind, the SAME message number, DIFFERENT arrival words. A mailbox has one receive
/// channel for each other rank, so above two ranks an exchange has one take for each of them, back to back. A pair has
/// one channel: there no two takes are of one exchange. The launch recorder puts a prefetch in front of the first take
/// of an exchange (`resident_launch_shadow::take_for`).
#[cfg(not(target_arch="nvptx64"))]
pub fn one_exchange(a:&Command,b:&Command)->bool {
    a.kind==b.kind && matches!((message(a),message(b)),(Some((x,s)),Some((y,t))) if s==t && x!=y)
}
#[cfg(not(target_arch="nvptx64"))]
pub fn record(routed:bool) {ARMED[usize::from(routed)].fetch_add(1,std::sync::atomic::Ordering::Relaxed);}
#[cfg(not(target_arch="nvptx64"))]
pub fn counts()->[u64;2] {ARMED.each_ref().map(|value|value.load(std::sync::atomic::Ordering::Relaxed))}
const _:()={assert!(OFFSET.is_multiple_of(16));assert!(OFFSET+core::mem::size_of::<Window>()<=1024);};
#[cfg(test)] mod tests {
    /// The prefetch moves to the first take only inside one exchange: the same message number on different arrival
    /// words. A pair's mailbox has one arrival word a slot, so two takes there never match.
    #[test] fn takes_of_one_exchange_share_the_message_number_on_different_arrival_words() {
        use crate::resident_launch::{Command,Kind};
        let take=|kind:Kind,arrival:u64,number:u64| {let at=if kind==Kind::Take {6} else {1};let mut command=Command::new(kind);command.word(at,arrival);command.word(at+1,number);command};
        for kind in [Kind::TakeScatter,Kind::Take] {
            let (a,b,c,d)=(take(kind,0x100,7),take(kind,0x108,7),take(kind,0x100,8),take(kind,0x100,7));
            assert!(super::one_exchange(&a,&b));
            assert!(!super::one_exchange(&a,&c),"another message number is another exchange");
            assert!(!super::one_exchange(&a,&d),"one arrival word is one channel: a pair");
        }
        assert!(!super::one_exchange(&take(Kind::TakeScatter,0x100,7),&take(Kind::Take,0x108,7)),"two kinds are not one exchange");
        let put=Command::new(Kind::Put);
        assert!(!super::one_exchange(&put,&put),"only a take waits for a message");
    }
    #[test] fn more_targets_are_one_job_static_projections_of_two_megabytes_or_more() {
        use crate::expert_ops::{Arguments,LaunchArguments,Op,Shape};use core::mem::offset_of;
        let mut plain=Command::new(Kind::PlainBf16);plain.word(0,0x4000_0000);plain.word(3,1536);plain.word(4,4096);
        let w=more(&plain).unwrap();assert_eq!((w.weight,w.bytes,w.ids,w.experts,w.picks),(0x4000_0000,BUDGET,0,0,0));
        plain.word(3,128);assert!(more(&plain).is_none(),"1 MB is under the least target");
        plain.word(3,512);assert_eq!(more(&plain).unwrap().bytes,512*4096*2);plain.word(0,0x4000_0004);assert!(more(&plain).is_none());
        let single=offset_of!(LaunchArguments,single);let shape=single+offset_of!(Arguments,shape);
        let mut nv=Command::new(Kind::Expert);nv.put(single+offset_of!(Arguments,weight),&0x5000_0000u64.to_ne_bytes());
        nv.put(shape+offset_of!(Shape,op),&(Op::NvProject as u32).to_ne_bytes());nv.put(shape+offset_of!(Shape,rows),&2048u32.to_ne_bytes());
        nv.put(shape+offset_of!(Shape,cols),&4096u32.to_ne_bytes());nv.put(shape+offset_of!(Shape,tokens),&1u32.to_ne_bytes());
        assert_eq!(more(&nv).unwrap().bytes,2048*4096/2);
        nv.put(shape+offset_of!(Shape,op),&(Op::Fp8Project as u32).to_ne_bytes());assert_eq!(more(&nv).unwrap().bytes,BUDGET);
        for op in [Op::Quantize,Op::SwiGlu,Op::Fold,Op::NvRows] {nv.put(shape+offset_of!(Shape,op),&(op as u32).to_ne_bytes());assert!(more(&nv).is_none());}
        nv.put(shape+offset_of!(Shape,op),&(Op::NvProject as u32).to_ne_bytes());nv.put(shape+offset_of!(Shape,tokens),&2u32.to_ne_bytes());assert!(more(&nv).is_none());
        nv.put(shape+offset_of!(Shape,tokens),&1u32.to_ne_bytes());nv.put(offset_of!(LaunchArguments,batch),&8u64.to_ne_bytes());assert!(more(&nv).is_none());
        assert!(more(&Command::new(Kind::Mx)).is_none() && more(&Command::new(Kind::Mixed)).is_none());
    }
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
        // 8, 9 and 32: tail rows of a take. SIDE_BLOCKS: the blocks of a side launch (side_window has the same stride rule).
        for picks in [1u64,8] {for bytes in [1u64,127,128,129,1<<20] {for tails in [u64::from(SIDE_BLOCKS),1,2,8,9,32] {
            let lines=crate::mhc_params::shadow_lines(bytes);let mut visits=vec![0u8;(lines*picks) as usize];
            for thread in 0..tails*256 {let mut line=thread;while line<lines*picks {assert!((line%lines)*128<bytes);visits[line as usize]+=1;line+=tails*256;}}
            assert!(visits.iter().all(|&visits|visits==1));
        }}}
    }
    #[test] fn a_plain_take_gets_a_static_side_window_and_never_a_routed_one() {
        let take=Command::new(Kind::Take);let w=window();
        let made=side_plain(&take,w).unwrap();assert_eq!(made.kind,Kind::ShadowSide as u32);
        // The side command carries the same window words as the scatter take's side command.
        let scatter=side(&command(),[6,1,1,256,0],w,0,&[]).unwrap();assert_eq!(made.payload[OFFSET..OFFSET+32],scatter.payload[OFFSET..OFFSET+32]);
        assert_eq!(made.payload[..OFFSET],Command::new(Kind::ShadowSide).payload[..OFFSET]);
        // Refused: another take kind, a routed window, a bad weight or size.
        assert!(side_plain(&command(),w).is_none(),"a scatter take keeps `side`");
        for bad in [Window {ids:0x2000,experts:16,picks:8,..w},Window {experts:16,..w},Window {picks:8,..w},Window {weight:0,..w},Window {weight:0x8004,..w},Window {bytes:0,..w},Window {bytes:BUDGET+1,..w}] {
            assert!(side_plain(&take,bad).is_none());
        }
        assert!(side_plain(&take,Window {bytes:BUDGET,..w}).is_some());
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
    #[test] fn batch_target_is_the_first_projection_of_a_one_row_bf16_batch() {
        use crate::next_params::{MxLayer,MxProjection,ResidentProjectionBatch};use core::mem::offset_of;
        let mut target=Command::new(Kind::MixedBf16);
        target.put(0,&0x8000u64.to_ne_bytes());target.put(offset_of!(MxProjection,rows),&4096u32.to_ne_bytes());
        target.put(offset_of!(MxProjection,cols),&4096u32.to_ne_bytes());target.put(offset_of!(MxProjection,tokens),&1u32.to_ne_bytes());
        target.put(offset_of!(MxLayer,count),&3u32.to_ne_bytes());
        let w=batch(&target).unwrap();assert_eq!((w.weight,w.bytes,w.ids,w.experts,w.picks),(0x8000,BUDGET,0,0,0));
        // An MXFP8 matrix is a byte a cell; a plain BF16 matrix (bit 0 of `plain`) is two.
        target.put(offset_of!(MxProjection,rows),&1024u32.to_ne_bytes());target.put(offset_of!(MxProjection,cols),&2048u32.to_ne_bytes());assert_eq!(batch(&target).unwrap().bytes,2<<20);
        target.put(offset_of!(ResidentProjectionBatch,plain),&1u32.to_ne_bytes());assert_eq!(batch(&target).unwrap().bytes,4<<20);
        target.put(offset_of!(ResidentProjectionBatch,plain),&6u32.to_ne_bytes());assert_eq!(batch(&target).unwrap().bytes,2<<20);
        // Refused: under the least target, more than one source row, an empty batch, an unaligned weight, another kind.
        target.put(offset_of!(MxProjection,rows),&256u32.to_ne_bytes());assert!(batch(&target).is_none());
        target.put(offset_of!(MxProjection,rows),&1024u32.to_ne_bytes());target.put(offset_of!(MxProjection,tokens),&2u32.to_ne_bytes());assert!(batch(&target).is_none());
        target.put(offset_of!(MxProjection,tokens),&1u32.to_ne_bytes());target.put(offset_of!(MxLayer,count),&0u32.to_ne_bytes());assert!(batch(&target).is_none());
        target.put(offset_of!(MxLayer,count),&1u32.to_ne_bytes());target.put(0,&0x8004u64.to_ne_bytes());assert!(batch(&target).is_none());
        target.put(0,&0x8000u64.to_ne_bytes());assert!(batch(&target).is_some());
        for kind in [Kind::Mixed,Kind::PlainBatch,Kind::Mx] {let mut other=target;other.kind=kind as u32;assert!(batch(&other).is_none());}
        assert!(more(&target).is_none() && mx(&target).is_none(),"the batch window is its own family");
        // The window passes the take's checks as every side window does, and makes a side command.
        let w=batch(&target).unwrap();assert_eq!(side(&command(),[6,1,1,256,0],w,0,&[]).unwrap().kind,Kind::ShadowSide as u32);
    }
    #[test] fn row_targets_are_launches_of_two_to_eight_rows_by_their_first_weights() {
        use crate::expert_ops::{GroupArguments,GroupEntry,Shape};use crate::next_params::{MxLayer,MxProjection,ResidentProjectionBatch};use core::mem::offset_of;
        // An MXFP8 row projection: a byte a cell, 2 to ROW_TOKENS rows.
        let mut mx_rows=Command::new(Kind::MxRows8);
        mx_rows.put(offset_of!(MxProjection,weights),&0x8000u64.to_ne_bytes());mx_rows.put(offset_of!(MxProjection,rows),&4096u32.to_ne_bytes());
        mx_rows.put(offset_of!(MxProjection,cols),&4096u32.to_ne_bytes());
        for tokens in 2..=ROW_TOKENS {mx_rows.put(offset_of!(MxProjection,tokens),&tokens.to_ne_bytes());let w=rows(&mx_rows,0,&[]).unwrap();assert_eq!((w.weight,w.bytes,w.ids,w.experts,w.picks),(0x8000,BUDGET,0,0,0));}
        for tokens in [0u32,1,9,64] {mx_rows.put(offset_of!(MxProjection,tokens),&tokens.to_ne_bytes());assert!(rows(&mx_rows,0,&[]).is_none(),"{tokens} rows");}
        mx_rows.put(offset_of!(MxProjection,tokens),&4u32.to_ne_bytes());mx_rows.put(offset_of!(MxProjection,rows),&256u32.to_ne_bytes());assert!(rows(&mx_rows,0,&[]).is_none(),"1 MB is under the least target");
        mx_rows.put(offset_of!(MxProjection,rows),&1024u32.to_ne_bytes());assert_eq!(rows(&mx_rows,0,&[]).unwrap().bytes,4<<20);
        mx_rows.put(offset_of!(MxProjection,weights),&0x8004u64.to_ne_bytes());assert!(rows(&mx_rows,0,&[]).is_none(),"unaligned weight");
        for kind in [Kind::MxRows2,Kind::MxRows4,Kind::MxRows8w,Kind::MxRowsTile] {let mut other=mx_rows;other.kind=kind as u32;other.put(0,&0x8000u64.to_ne_bytes());assert_eq!(rows(&other,0,&[]).unwrap().bytes,4<<20);}
        // The one-token families do not take a row launch, and a row window is no one-token window.
        let mut one=mx_rows;one.kind=Kind::Mx as u32;one.put(0,&0x8000u64.to_ne_bytes());assert!(rows(&one,0,&[]).is_none());
        let mut row_mx=mx_rows;row_mx.put(0,&0x8000u64.to_ne_bytes());assert!(mx(&row_mx).is_none() && more(&row_mx).is_none() && batch(&row_mx).is_none());
        // A mixed row stage and the fold rows, by their first projection: 1, 2 or 4 bytes a cell.
        let mut stage=Command::new(Kind::MixedRows);
        stage.put(0,&0x9000u64.to_ne_bytes());stage.put(offset_of!(MxProjection,rows),&1024u32.to_ne_bytes());stage.put(offset_of!(MxProjection,cols),&2048u32.to_ne_bytes());
        stage.put(offset_of!(MxProjection,tokens),&3u32.to_ne_bytes());stage.put(offset_of!(MxLayer,count),&2u32.to_ne_bytes());
        assert_eq!(rows(&stage,0,&[]).unwrap().bytes,2<<20);
        stage.put(offset_of!(ResidentProjectionBatch,plain),&1u32.to_ne_bytes());assert_eq!(rows(&stage,0,&[]).unwrap().bytes,4<<20);
        let mut fold=stage;fold.kind=Kind::FoldRows as u32;fold.put(offset_of!(ResidentProjectionBatch,plain),&0u32.to_ne_bytes());assert_eq!(rows(&fold,0,&[]).unwrap().bytes,4<<20);
        fold.kind=Kind::FoldRowsF32 as u32;fold.put(offset_of!(MxProjection,rows),&144u32.to_ne_bytes());fold.put(offset_of!(MxProjection,cols),&4096u32.to_ne_bytes());
        assert_eq!(rows(&fold,0,&[]).unwrap().bytes,144*4096*4);
        stage.put(offset_of!(MxLayer,count),&0u32.to_ne_bytes());assert!(rows(&stage,0,&[]).is_none(),"an empty stage");
        stage.put(offset_of!(MxLayer,count),&1u32.to_ne_bytes());stage.put(offset_of!(MxProjection,tokens),&1u32.to_ne_bytes());assert!(rows(&stage,0,&[]).is_none(),"one row");
        // A block-FP8 row group: the first part of the staged table the command names.
        let mut entry=GroupEntry {shape:Shape {op:crate::expert_ops::Op::Fp8Rows,rows:1024,cols:4096,block_rows:128,block_cols:128,tokens:4,global:1.0,activation:1.0,bound:0.0,role:0},
            weight:0xa000,scales:0xb000,first:0,first_row:0,input_words:0,weight_bytes:4<<20,scale_bytes:0,pad:0};
        // The staged bytes: 16 bytes of another table, then the entry's fields that the window reads.
        let as_bytes=|e:&GroupEntry|{let mut v=vec![0u8;16+core::mem::size_of::<GroupEntry>()];
            for (at,b) in [(offset_of!(GroupEntry,shape)+offset_of!(Shape,tokens),e.shape.tokens.to_ne_bytes().to_vec()),(offset_of!(GroupEntry,weight),e.weight.to_ne_bytes().to_vec()),
                (offset_of!(GroupEntry,weight_bytes),e.weight_bytes.to_ne_bytes().to_vec())] {v[16+at..16+at+b.len()].copy_from_slice(&b);}v};
        let base=0x10_0000u64;let mut group=Command::new(Kind::Fp8RowsMulti);
        group.put(offset_of!(GroupArguments,table),&(base+16).to_ne_bytes());group.put(offset_of!(GroupArguments,count),&2u32.to_ne_bytes());
        let w=rows(&group,base,&as_bytes(&entry)).unwrap();assert_eq!((w.weight,w.bytes),(0xa000,4<<20));
        for kind in [Kind::Fp8RowsSmall,Kind::Fp8RowsFew,Kind::Fp8RowsShare] {let mut other=group;other.kind=kind as u32;assert!(rows(&other,base,&as_bytes(&entry)).is_some());}
        assert!(rows(&group,base,&as_bytes(&entry)[..40]).is_none(),"a table past the staged bytes");
        assert!(rows(&group,base+32,&as_bytes(&entry)).is_none(),"a table below the stage");
        entry.shape.tokens=16;assert!(rows(&group,base,&as_bytes(&entry)).is_none(),"a prompt-sized part");
        entry.shape.tokens=1;assert!(rows(&group,base,&as_bytes(&entry)).is_some(),"parts of one row in a row group");
        group.put(offset_of!(GroupArguments,count),&0u32.to_ne_bytes());assert!(rows(&group,base,&as_bytes(&entry)).is_none(),"an empty group");
        // The window passes the take's checks as every side window does, and makes a side command.
        assert_eq!(side(&command(),[6,1,1,256,0],w,0,&[]).unwrap().kind,Kind::ShadowSide as u32);
    }
}
