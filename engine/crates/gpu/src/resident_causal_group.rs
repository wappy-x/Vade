//! One launch for 2 to 8 same-shape recurrent calls. The recorder asks `plan` after chain fusion.
//! A block row runs one member command through that command's own one-chat body.
use crate::causal_ops::{Arguments,Op,Shape};
use crate::resident_launch::{Command,Kind};
const COMMAND_SIZE:usize=1040;
const MAX_MEMBERS:u32=8;
const MAIN:u8=0;
const SIDE:u8=1;
const _:()={
    assert!(core::mem::size_of::<Command>()==COMMAND_SIZE);
    assert!(core::mem::offset_of!(Arguments,weight)==32);
    assert!(core::mem::offset_of!(Arguments,output)==40);
    assert!(core::mem::offset_of!(Arguments,shape)==48);
    assert!(core::mem::size_of::<Shape>()==32);
};
/// One recorded call, as the recorder sees it.
pub(crate) struct Launch<'a> {pub command:&'a Command,pub blocks:u32,pub rows:u32,pub planes:u32,pub threads:u32,pub shared:u32,pub lane:u8}
/// What to append to the stage, and how many trailing calls the group replaces.
pub(crate) struct Plan {pub offset:u32,pub append:Vec<u8>,pub n:u32,pub member_kind:u32,pub blocks:u32,pub threads:u32,pub shared:u32,pub lane:u8,pub replace:u8}
fn u32_at(bytes:&[u8],at:usize)->Option<u32> {Some(u32::from_ne_bytes(bytes.get(at..at+4)?.try_into().ok()?))}
fn u64_at(bytes:&[u8],at:usize)->Option<u64> {Some(u64::from_ne_bytes(bytes.get(at..at+8)?.try_into().ok()?))}
fn op_of(n:u32)->Option<Op> {[Op::Conv,Op::ConvParts,Op::Step,Op::StepPlanar,Op::Prepare,Op::GatedNorm].into_iter().find(|op|*op as u32==n)}
fn op_at(payload:&[u8],arguments:usize)->Option<Op> {op_of(u32_at(payload,arguments+core::mem::offset_of!(Arguments,shape))?) }
fn arguments_at(payload:&[u8],at:usize)->Option<Arguments> {
    let shape_at=at+core::mem::offset_of!(Arguments,shape);
    let shape=Shape {op:op_of(u32_at(payload,shape_at)?)?,heads:u32_at(payload,shape_at+4)?,dim:u32_at(payload,shape_at+8)?,taps:u32_at(payload,shape_at+12)?,
        rows:u32_at(payload,shape_at+16)?,cols:u32_at(payload,shape_at+20)?,epsilon:f32::from_bits(u32_at(payload,shape_at+24)?),bound:f32::from_bits(u32_at(payload,shape_at+28)?)};
    Some(Arguments {input:[u64_at(payload,at)?,u64_at(payload,at+8)?,u64_at(payload,at+16)?,u64_at(payload,at+24)?],weight:u64_at(payload,at+32)?,output:u64_at(payload,at+40)?,shape})
}
fn push_range(ranges:&mut Vec<(u64,u64)>,addr:u64,count:usize)->Option<()> {
    if count==0 {return Some(());}
    let end=addr.checked_add(u64::try_from(count).ok()?.checked_mul(4)?)?;ranges.push((addr,end));Some(())
}
/// Reads and writes of one command. A chain contributes all three of its argument blocks. Empty lengths are not ranges.
fn spans(command:&Command)->Option<(Vec<(u64,u64)>,Vec<(u64,u64)>)> {
    let offsets:&[usize]=if command.kind==Kind::CausalChain as u32 {&[0,256,512]} else {&[0]};
    let (mut reads,mut writes)=(Vec::new(),Vec::new());
    for &at in offsets {
        let args=arguments_at(&command.payload,at)?;
        let (inputs,weight,units,words)=args.shape.lengths()?;
        for (addr,count) in args.input.into_iter().zip(inputs) {push_range(&mut reads,addr,count)?;}
        push_range(&mut reads,args.weight,weight)?;
        push_range(&mut writes,args.output,units.checked_mul(words)?)?;
    }
    Some((reads,writes))
}
fn overlaps(a:(u64,u64),b:(u64,u64))->bool {a.0<b.1 && b.0<a.1}
fn clashes(writes:&[(u64,u64)],reads:&[(u64,u64)],other_writes:&[(u64,u64)])->bool {
    writes.iter().any(|write| reads.iter().chain(other_writes).any(|other| overlaps(*write,*other)))
}
/// True when no write of one command overlaps a read or a write of the other. Shared read-only weights are allowed.
pub(crate) fn independent(members:&[Command],incoming:&Command)->bool {
    let Some((incoming_reads,incoming_writes))=spans(incoming) else {return false;};
    members.iter().all(|member| spans(member).is_some_and(|(reads,writes)| !clashes(&incoming_writes,&reads,&writes) && !clashes(&writes,&incoming_reads,&incoming_writes)))
}
fn recurrent(command:&Command)->Option<u32> {
    if command.kind==Kind::CausalChain as u32 {return Some(command.kind);}
    if command.kind!=Kind::CausalFast as u32 {return None;}
    matches!(op_at(&command.payload,0),Some(Op::Conv|Op::ConvParts|Op::Step|Op::StepPlanar)).then_some(command.kind)
}
fn same_body(left:&Command,right:&Command)->bool {left.kind==right.kind && (left.kind!=Kind::CausalFast as u32 || op_at(&left.payload,0)==op_at(&right.payload,0))}
fn command_bytes(command:&Command)->[u8;COMMAND_SIZE] {
    let mut out=[0u8;COMMAND_SIZE];out[..4].copy_from_slice(&command.kind.to_ne_bytes());
    for (i,word) in command.reserved.iter().enumerate() {out[4+i*4..8+i*4].copy_from_slice(&word.to_ne_bytes());}
    out[16..].copy_from_slice(&command.payload);out
}
fn command_from(bytes:&[u8])->Option<Command> {
    if bytes.len()<COMMAND_SIZE {return None;}
    let mut command=Command::new(Kind::CausalFast);command.kind=u32_at(bytes,0)?;
    for (i,word) in command.reserved.iter_mut().enumerate() {*word=u32_at(bytes,4+i*4)?;}
    command.payload.copy_from_slice(&bytes[16..COMMAND_SIZE]);Some(command)
}
fn fits(staged:&[u8],append:&[u8],offset:usize)->bool {offset.is_multiple_of(16) && u32::try_from(offset).is_ok() && !crate::resident_launch::stage::full(staged.len(),append.len())}
/// `None` leaves the two calls as separate launches. The incoming call must be on the main lane.
pub(crate) fn plan(previous:Launch<'_>,incoming:Launch<'_>,staged:&[u8])->Option<Plan> {
    if incoming.lane!=MAIN || previous.lane==SIDE || incoming.rows!=1 || incoming.planes!=1 {return None;}
    if incoming.blocks==0 || incoming.blocks!=previous.blocks || incoming.threads!=128 || previous.threads!=128 || incoming.shared!=0 || previous.shared!=0 {return None;}
    let body=recurrent(incoming.command)?;
    if previous.command.kind==Kind::CausalGroup as u32 {extend(previous,incoming,staged,body)} else {fresh(previous,incoming,staged,body)}
}
fn fresh(previous:Launch<'_>,incoming:Launch<'_>,staged:&[u8],body:u32)->Option<Plan> {
    if previous.rows!=1 || previous.planes!=1 || recurrent(previous.command)?!=body || !same_body(previous.command,incoming.command) {return None;}
    if !independent(std::slice::from_ref(previous.command),incoming.command) {return None;}
    let mut append=command_bytes(previous.command).to_vec();append.extend_from_slice(&command_bytes(incoming.command));
    let offset=staged.len();if !fits(staged,&append,offset) {return None;}
    Some(Plan {offset:offset as u32,append,n:2,member_kind:body,blocks:incoming.blocks,threads:128,shared:0,lane:previous.lane,replace:2})
}
fn extend(previous:Launch<'_>,incoming:Launch<'_>,staged:&[u8],body:u32)->Option<Plan> {
    if previous.rows<2 || previous.rows>=MAX_MEMBERS || previous.planes!=1 {return None;}
    let n=u32_at(&previous.command.payload,8)?;let member_kind=u32_at(&previous.command.payload,12)?;
    let offset=u32_at(&previous.command.payload,16)? as usize;let blocks=u32_at(&previous.command.payload,20)?;
    if n!=previous.rows || blocks!=previous.blocks || member_kind!=body {return None;}
    let table=staged.get(offset..offset+n as usize*COMMAND_SIZE)?.to_vec();
    let mut members=Vec::with_capacity(n as usize);
    for chunk in table.chunks(COMMAND_SIZE) {let member=command_from(chunk)?;if member.kind!=member_kind || !same_body(&member,incoming.command) {return None;}members.push(member);}
    if !independent(&members,incoming.command) {return None;}
    let added=command_bytes(incoming.command);
    let (new_offset,append)=if offset+table.len()==staged.len() {(offset,added.to_vec())} else {let mut append=table;append.extend_from_slice(&added);(staged.len(),append)};
    if !fits(staged,&append,new_offset) {return None;}
    Some(Plan {offset:new_offset as u32,append,n:n+1,member_kind,blocks,threads:128,shared:0,lane:previous.lane,replace:1})
}
/// An allocation can submit the calls the first plan saw. Re-read after it succeeds.
/// The first check avoids allocating a stage for calls that cannot group. With an existing stage no allocation or second plan is needed.
pub(crate) fn after_stage<T>(mut plan:impl FnMut()->Option<T>,ensure:impl FnOnce()->Result<bool,String>)->Result<Option<T>,String> {
    let before=plan();if before.is_none() {return Ok(None);}
    if ensure()? {Ok(plan())} else {Ok(before)}
}
/// A nested flush may have installed a stage used by captured graphs.
pub(crate) fn install_stage<T>(slot:&mut Option<T>,stage:T) {slot.get_or_insert(stage);}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_ops::reference;
    struct Recorded {command:Command,blocks:u32,rows:u32,planes:u32,threads:u32,shared:u32,lane:u8}
    fn launch_of(call:&Recorded)->Launch<'_> {Launch {command:&call.command,blocks:call.blocks,rows:call.rows,planes:call.planes,threads:call.threads,shared:call.shared,lane:call.lane}}
    /// The recorder's order: chain fusion, then `plan`, then one arithmetic count for the whole group.
    fn record(calls:&mut Vec<Recorded>,staged:&mut Vec<u8>,heavy:&mut usize,command:Command,blocks:u32,lane:u8,arithmetic:bool) {
        if calls.len()>=2 && lane!=2 {
            if let Some((fused,fused_blocks))=crate::resident_causal_chain::fuse(&calls[calls.len()-2].command,&calls[calls.len()-1].command,&command) {
                calls.truncate(calls.len()-2);*heavy=heavy.saturating_sub(2);
                calls.push(Recorded {command:fused,blocks:fused_blocks,rows:1,planes:1,threads:128,shared:0,lane});*heavy+=1;absorb(calls,staged,heavy);return;
            }
        }
        calls.push(Recorded {command,blocks,rows:1,planes:1,threads:128,shared:0,lane});
        if arithmetic {*heavy+=1;absorb(calls,staged,heavy);}
    }
    fn absorb(calls:&mut Vec<Recorded>,staged:&mut Vec<u8>,heavy:&mut usize) {
        if calls.len()<2 {return;}
        let n=calls.len();
        let Some(plan)=plan(launch_of(&calls[n-2]),launch_of(&calls[n-1]),staged) else {return;};
        staged.extend_from_slice(&plan.append);
        let mut command=Command::new(Kind::CausalGroup);command.put(8,&plan.n.to_ne_bytes());command.put(12,&plan.member_kind.to_ne_bytes());command.put(16,&plan.offset.to_ne_bytes());command.put(20,&plan.blocks.to_ne_bytes());
        let grid_blocks=plan.blocks;let rows=plan.n;let threads=plan.threads;let shared=plan.shared;let lane=plan.lane;
        if plan.replace==1 {calls.pop();let previous=calls.last_mut().unwrap();previous.command=command;previous.rows=rows;previous.blocks=grid_blocks;previous.threads=threads;previous.shared=shared;previous.lane=lane;}
        else {let keep=calls.len()-plan.replace as usize;calls.truncate(keep);calls.push(Recorded {command,blocks:grid_blocks,rows,planes:1,threads,shared,lane});}
        *heavy=heavy.saturating_sub(1);
    }
    #[test]
    fn stage_pressure_flush_invalidates_the_plan_and_keeps_the_installed_stage() {
        use std::cell::{Cell,RefCell};
        let calls=RefCell::new(vec![step(0,32),step(1,32)]);let stage=RefCell::new(None);
        let queued=Cell::new(0);let old_stage=73u64;
        let plan_now=|| {let calls=calls.borrow();if calls.len()<2 {return None;}
            let launch=|command|Launch {command,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN};
            plan(launch(&calls[0]),launch(&calls[1]),&[])};
        let got=after_stage(plan_now,|| {
            // The budget refuses a new allocation. A nested flush uses a pooled stage.
            queued.set(calls.borrow().len());calls.borrow_mut().clear();*stage.borrow_mut()=Some(old_stage);
            install_stage(&mut stage.borrow_mut(),99);Ok(true)
        }).unwrap();
        assert!(got.is_none(),"a submitted pair must not be replaced by the saved plan");
        assert_eq!(queued.get(),2);assert_eq!(*stage.borrow(),Some(old_stage),"retain the stage named by captured graphs");
    }
    #[test]
    fn an_existing_stage_needs_only_one_plan() {
        let mut plans=0;assert_eq!(after_stage(||{plans+=1;Some(7)},||Ok(false)).unwrap(),Some(7));assert_eq!(plans,1);
    }
    #[test]
    fn stage_failure_stops_planning_and_ineligible_calls_allocate_nothing() {
        use std::cell::Cell;
        let calls=Cell::new(0);
        let failed=after_stage(||{calls.set(calls.get()+1);Some(1)},||Err("stage budget refused".into()));
        assert_eq!(failed.unwrap_err(),"stage budget refused");assert_eq!(calls.get(),1);
        let got=after_stage(||None::<usize>,||panic!("ineligible calls must not allocate")).unwrap();assert_eq!(got,None);
    }
    fn members(staged:&[u8],command:&Command)->Vec<Command> {
        let n=u32::from_ne_bytes(command.payload[8..12].try_into().unwrap()) as usize;let offset=u32::from_ne_bytes(command.payload[16..20].try_into().unwrap()) as usize;
        (0..n).map(|i| command_from(&staged[offset+i*COMMAND_SIZE..offset+(i+1)*COMMAND_SIZE]).unwrap()).collect()
    }
    /// Far enough apart that a 32-head step's state and output do not touch the next chat.
    const STEP_STRIDE:u64=0x40_0000;
    fn step(chat:u32,blocks:u32)->Command {
        let base=0x1_0000_0000u64+u64::from(chat)*STEP_STRIDE;
        crate::resident_launch_encode::causal_fast(Arguments {input:[base,base+0x2_0000,4,4],weight:4,output:base+0x8_0000,shape:Shape {op:Op::StepPlanar,heads:blocks,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}})
    }
    fn conv(chat:u32,weight:u64)->Command {
        let base=0x2_0000_0000u64+u64::from(chat)*0x10_0000;
        crate::resident_launch_encode::causal_fast(Arguments {input:[base,base+0x100,base+0x200,base+0x300],weight,output:base+0x1_000,shape:Shape {op:Op::ConvParts,heads:2,dim:1,taps:2,rows:0,cols:0,epsilon:0.0,bound:0.0}})
    }
    #[test]
    fn shared_weight_is_independent_and_a_shared_scratch_is_not() {
        let (left,right)=(conv(0,0x5000),conv(1,0x5000));
        assert!(independent(std::slice::from_ref(&left),&right),"a shared read-only weight is still independent");
        let mut shared=conv(1,0x5000);shared.word(5,left.payload[40..48].try_into().map(u64::from_ne_bytes).unwrap());
        assert!(!independent(std::slice::from_ref(&left),&shared),"a scratch one chat writes and another writes is not independent");
        let mut reads_state=conv(1,0x5000);reads_state.word(0,left.payload[40..48].try_into().map(u64::from_ne_bytes).unwrap());
        assert!(!independent(std::slice::from_ref(&left),&reads_state),"a buffer one chat writes and another reads is not independent");
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        record(&mut calls,&mut staged,&mut heavy,left,96,MAIN,true);record(&mut calls,&mut staged,&mut heavy,right,96,MAIN,true);
        assert_eq!(calls.len(),1);assert_eq!(calls[0].rows,2);
        record(&mut calls,&mut staged,&mut heavy,shared,96,MAIN,true);
        assert_eq!(calls.len(),2,"the overlapping call stays its own launch");assert_eq!(calls[0].rows,2);assert_eq!(calls[1].command.kind,Kind::CausalFast as u32);
    }
    #[test]
    fn chats_at_different_positions_keep_their_own_commands() {
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        for chat in 0..4u32 {record(&mut calls,&mut staged,&mut heavy,step(chat,32),32,MAIN,true);}
        assert_eq!((calls.len(),calls[0].rows,heavy),(1,4,1));
        let grouped=members(&staged,&calls[0].command);
        for (chat,member) in grouped.iter().enumerate() {
            let plain=step(chat as u32,32);assert_eq!(member.payload,plain.payload,"chat {chat} keeps its own addresses");
            if chat!=0 {assert_ne!(u64::from_ne_bytes(member.payload[40..48].try_into().unwrap()),u64::from_ne_bytes(grouped[0].payload[40..48].try_into().unwrap()),"chat {chat} does not share chat 0's output");}
        }
    }
    #[test]
    fn unequal_blocks_and_context_ops_do_not_group() {
        let (left,right)=(step(0,32),step(1,32));
        assert!(plan(Launch {command:&left,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},Launch {command:&right,blocks:16,rows:1,planes:1,threads:128,shared:0,lane:MAIN},&[]).is_none());
        let softmax=crate::resident_launch_encode::causal_fast(Arguments {input:[0x1000,0x2000,4,4],weight:4,output:0x3000,shape:Shape {op:Op::Softmax,heads:32,dim:0,taps:0,rows:8,cols:0,epsilon:0.0,bound:0.0}});
        let other=crate::resident_launch_encode::causal_fast(Arguments {input:[0x4000,0x5000,4,4],weight:4,output:0x6000,shape:Shape {op:Op::Softmax,heads:32,dim:0,taps:0,rows:64,cols:0,epsilon:0.0,bound:0.0}});
        assert!(plan(Launch {command:&softmax,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},Launch {command:&other,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},&[]).is_none());
    }
    #[test]
    fn one_call_stays_its_own_launch() {
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        let command=step(0,32);record(&mut calls,&mut staged,&mut heavy,command,32,MAIN,true);
        assert_eq!(calls.len(),1);assert_eq!(calls[0].command.kind,Kind::CausalFast as u32);assert_eq!(calls[0].command.payload,command.payload);assert_eq!(heavy,1);
        record(&mut calls,&mut staged,&mut heavy,Command::new(Kind::Copy),1,MAIN,false);record(&mut calls,&mut staged,&mut heavy,step(1,32),32,MAIN,true);
        assert_eq!(calls.len(),3,"a call between two chats keeps each chat on its own launch");
        assert!(calls.iter().all(|call| call.command.kind!=Kind::CausalGroup as u32));
    }
    #[test]
    fn joining_and_leaving_changes_only_the_count() {
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);let mut rows=Vec::new();
        for n in [2u32,3,2] {
            for chat in 0..n {record(&mut calls,&mut staged,&mut heavy,step(chat+n*10,32),32,MAIN,true);}
            rows.push(calls.last().unwrap().rows);
            record(&mut calls,&mut staged,&mut heavy,Command::new(Kind::Copy),1,MAIN,false);
        }
        assert_eq!(rows,[2,3,2]);assert_eq!(heavy,3);
    }
    #[test]
    fn a_ninth_chat_starts_a_new_group() {
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        for chat in 0..9u32 {record(&mut calls,&mut staged,&mut heavy,step(chat,32),32,MAIN,true);}
        assert_eq!(calls.len(),2);assert_eq!(calls[0].rows,8);assert_eq!(calls[1].command.kind,Kind::CausalFast as u32);assert_eq!(heavy,2);
    }
    #[test]
    fn prepare_stays_ungrouped_and_eight_chains_become_one_group() {
        let prepare=|chat:u32| {let base=0x3_0000_0000u64+u64::from(chat)*0x20_0000;crate::resident_launch_encode::causal_fast(Arguments {input:[base,base+0x1000,base+0x2000,4],weight:base+0x3000,output:base+0x4000,shape:Shape {op:Op::Prepare,heads:1,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:-1.0}})};
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        record(&mut calls,&mut staged,&mut heavy,prepare(0),32,MAIN,true);record(&mut calls,&mut staged,&mut heavy,prepare(1),32,MAIN,true);
        assert_eq!(calls.len(),2,"prepare stays ungrouped so chain fusion can still see it");assert!(calls.iter().all(|call| call.command.kind==Kind::CausalFast as u32));
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        for chat in 0..8u32 {
            let base=0x4_0000_0000u64+u64::from(chat)*0x20_0000;let step_out=base+0x10_0000;let norm_in=step_out+128*128*4;
            let prepared=crate::resident_launch_encode::causal_fast(Arguments {input:[base,base+0x1000,base+0x2000,4],weight:base+0x3000,output:base+0x8000,shape:Shape {op:Op::Prepare,heads:1,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:-1.0}});
            let stepped=crate::resident_launch_encode::causal_fast(Arguments {input:[base+0x8000,base+0x4_0000,4,4],weight:4,output:step_out,shape:Shape {op:Op::StepPlanar,heads:1,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}});
            let norm=crate::resident_launch_encode::causal_fast(Arguments {input:[norm_in,base+0x18_0000,4,4],weight:base+0x19_0000,output:base+0x1a_0000,shape:Shape {op:Op::GatedNorm,heads:1,dim:128,taps:0,rows:0,cols:0,epsilon:1e-6,bound:0.0}});
            record(&mut calls,&mut staged,&mut heavy,prepared,32,MAIN,true);record(&mut calls,&mut staged,&mut heavy,stepped,1,MAIN,true);record(&mut calls,&mut staged,&mut heavy,norm,32,MAIN,true);
        }
        assert_eq!(calls.len(),1);assert_eq!(calls[0].command.kind,Kind::CausalGroup as u32);assert_eq!(calls[0].rows,8);assert_eq!(heavy,1);
        assert!(members(&staged,&calls[0].command).iter().all(|member| member.kind==Kind::CausalChain as u32));
    }
    fn step_values(chat:u32)->(Vec<f32>,Vec<f32>,Vec<f32>) {
        let (d,heads)=(128usize,1usize);let mark=0.05+chat as f32*0.01;
        let prepared=vec![mark;heads*(4*d+1)];let state=vec![0.02+chat as f32*0.001;heads*d*d];
        let output=reference(Shape {op:Op::StepPlanar,heads:1,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0},[&prepared,&state,&[],&[]],&[]).unwrap();
        (prepared,state,output)
    }
    #[test]
    fn each_chat_of_a_group_matches_its_plain_step_and_a_swapped_table_does_not() {
        use std::collections::BTreeMap;
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);let mut memory=BTreeMap::new();let mut plain=Vec::new();
        for chat in 0..4u32 {
            let command=step(chat,1);let args=arguments_at(&command.payload,0).unwrap();let (prepared,state,output)=step_values(chat);
            memory.insert(args.input[0],prepared);memory.insert(args.input[1],state);plain.push(output);
            record(&mut calls,&mut staged,&mut heavy,command,1,MAIN,true);
        }
        let execute=|member:&Command| {let args=arguments_at(&member.payload,0).unwrap();
            (args.output,reference(args.shape,[&memory[&args.input[0]],&memory[&args.input[1]],&[],&[]],&[]).unwrap())};
        let grouped=members(&staged,&calls[0].command);
        for (chat,member) in grouped.iter().enumerate() {
            let (address,output)=execute(member);assert_eq!(address,0x1_0000_0000+chat as u64*STEP_STRIDE+0x8_0000);
            assert_eq!(output.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),plain[chat].iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        }
        // Keep chat zero's output address but substitute chat one's inputs. Address-driven execution must catch this.
        let mut wrong=grouped[0];wrong.word(0,u64_at(&grouped[1].payload,0).unwrap());wrong.word(1,u64_at(&grouped[1].payload,8).unwrap());
        let (address,output)=execute(&wrong);assert_eq!(address,u64_at(&grouped[0].payload,40).unwrap());
        assert_ne!(output,plain[0],"wrong input pointers cannot pass by using the loop index");
    }
    #[test]
    fn a_join_stays_first_and_a_side_call_does_not_group() {
        let (left,right)=(step(0,32),step(1,32));
        let joined=plan(Launch {command:&left,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:2},Launch {command:&right,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},&[]).unwrap();
        assert_eq!(joined.lane,2,"the group's wait is the first call's wait");
        assert!(plan(Launch {command:&left,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},Launch {command:&right,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:2},&[]).is_none());
        assert!(plan(Launch {command:&left,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:SIDE},Launch {command:&right,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},&[]).is_none());
        assert!(plan(Launch {command:&left,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN},Launch {command:&right,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:SIDE},&[]).is_none());
    }
    /// The member table lives in the recorder's fixed stage area. A table that does not fit leaves the calls as they are.
    #[test]
    fn a_table_that_does_not_fit_the_stage_area_makes_no_group() {
        use crate::resident_launch::stage::{BASE,BYTES};
        let room=BYTES-BASE;let launch=|command|Launch {command,blocks:32,rows:1,planes:1,threads:128,shared:0,lane:MAIN};
        let (left,right)=(step(0,32),step(1,32));
        let fit=plan(launch(&left),launch(&right),&vec![0u8;room-2*COMMAND_SIZE]).unwrap();
        assert_eq!(fit.offset as usize+fit.append.len(),room,"two members fill the area to its last byte");
        assert!(plan(launch(&left),launch(&right),&vec![0u8;room-2*COMMAND_SIZE+16]).is_none(),"16 bytes less room and the pair stays two launches");
        // Room for one pair and 16 bytes more: the third chat cannot join, and the fourth cannot start a group with it.
        let (mut calls,mut staged,mut heavy)=(Vec::new(),vec![0u8;room-2*COMMAND_SIZE-16],0usize);
        for chat in 0..4u32 {record(&mut calls,&mut staged,&mut heavy,step(chat,32),32,MAIN,true);}
        assert_eq!(calls.iter().map(|call|call.rows).collect::<Vec<_>>(),[2,1,1]);assert_eq!(heavy,3);
        assert_eq!(staged.len(),room-16,"nothing is staged for the calls that stay alone");
        assert!(calls[1..].iter().all(|call|call.command.kind==Kind::CausalFast as u32));
    }
    /// A group grows in place only while its table is the last one staged. Behind another table it moves whole to the end.
    #[test]
    fn a_group_behind_another_table_moves_whole_before_it_grows() {
        let (mut calls,mut staged,mut heavy)=(Vec::new(),Vec::new(),0usize);
        for chat in 0..2u32 {record(&mut calls,&mut staged,&mut heavy,step(chat,32),32,MAIN,true);}
        staged.extend_from_slice(&[0xa5;48]);
        record(&mut calls,&mut staged,&mut heavy,step(2,32),32,MAIN,true);
        assert_eq!((calls.len(),calls[0].rows,heavy),(1,3,1));
        let offset=u32_at(&calls[0].command.payload,16).unwrap() as usize;
        assert_eq!(offset,2*COMMAND_SIZE+48,"the grown table starts after the other table");assert_eq!(staged.len(),offset+3*COMMAND_SIZE);
        assert_eq!(&staged[2*COMMAND_SIZE..offset],&[0xa5u8;48][..],"the other table keeps its bytes");
        for (chat,member) in members(&staged,&calls[0].command).iter().enumerate() {assert_eq!(member.payload,step(chat as u32,32).payload,"chat {chat} keeps its own command");}
        // The next chat finds the table last again and is appended in place.
        record(&mut calls,&mut staged,&mut heavy,step(3,32),32,MAIN,true);
        assert_eq!(u32_at(&calls[0].command.payload,16).unwrap() as usize,offset);assert_eq!((calls[0].rows,staged.len()),(4,offset+4*COMMAND_SIZE));
    }
}
