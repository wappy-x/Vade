//! Grouped recurrent launch: 2 to 8 chats, one `resident_launch_causal_group`, against one `resident_launch_causal_fast` a chat.
//! Old image and this image must match. A table with two commands swapped must follow the swapped order, not the plain order.
#![forbid(unsafe_code)]
use super::{Arguments, Buffer, Gpu, H, Result, Shape, CONV_PARTS, GUARD, STEP_PLANAR, STEP, CONV, command, launch_replay, pattern, ptr};
use std::time::Instant;

const FAST: u32 = 19;
const GROUP: u32 = 64;

struct ShapeCase { op: u32, heads: u32, dim: u32, taps: u32, blocks: u32, name: &'static str }

fn shapes() -> [ShapeCase; 4] {
    [ShapeCase { op: STEP_PLANAR, heads: 32, dim: 128, taps: 0, blocks: 32, name: "step-planar" },
     ShapeCase { op: CONV_PARTS, heads: 32, dim: 128, taps: 4, blocks: 96, name: "conv-parts" },
     ShapeCase { op: STEP, heads: 17, dim: 128, taps: 0, blocks: 17, name: "step" },
     ShapeCase { op: CONV, heads: 17, dim: 128, taps: 4, blocks: 51, name: "conv" }]
}

fn words_of(case: &ShapeCase) -> (Vec<usize>, usize, usize) {
    let (h,d,t)=(case.heads as usize,case.dim as usize,case.taps as usize);let w=h*d;
    match case.op {
        STEP_PLANAR|STEP=>(vec![h*(4*d+1),h*d*d,0,0],0,h*(d*d+d)),
        CONV_PARTS=>(vec![w,w,w,3*(t-1)*w],3*w*t,3*w*t),
        CONV=>(vec![3*w,3*(t-1)*w,0,0],3*w*t,3*w*t),
        _=>unreachable!(),
    }
}

struct Chat { args: Arguments, output: Buffer, command: u64 }

fn make_chat(g: &mut Gpu, case: &ShapeCase, chat: u32) -> Result<Chat> {
    let (lens, weight_len, total) = words_of(case);
    let seed = 4100 + u64::from(chat) * 17 + u64::from(case.op);
    let inputs: Vec<Buffer> = lens.iter().enumerate().map(|(i, &count)| Buffer::new(g, &pattern(0, count.max(1), seed + i as u64))).collect::<Result<_>>()?;
    let weight = Buffer::new(g, &pattern(0, weight_len.max(1), seed + 9))?;
    let output = Buffer::new(g, &vec![f32::NAN; total])?;
    let mut input = [4u64; 4];
    for (i, buffer) in inputs.iter().enumerate() { if lens[i] > 0 { input[i] = buffer.address; } }
    let args = Arguments { input, weight: if weight_len == 0 { 4 } else { weight.address }, output: output.address, shape: Shape { op: case.op, heads: case.heads, dim: case.dim, taps: case.taps, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 } };
    let command = g.upload(&command(&args, FAST))?;
    Ok(Chat { args, output, command })
}

fn group_bytes(member_table: u64, n: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; 1040];
    bytes[..4].copy_from_slice(&GROUP.to_le_bytes());
    bytes[16..24].copy_from_slice(&member_table.to_le_bytes());
    bytes[24..28].copy_from_slice(&n.to_le_bytes());
    bytes[28..32].copy_from_slice(&FAST.to_le_bytes());
    bytes
}

fn launch_group(g: &Gpu, kernel: H, table: u64, blocks: u32, n: u32) -> Result<()> {
    let (mut table, mut index) = (table, 0u64);
    g.launch(kernel, [blocks, n, 1], 128, 0, &mut [ptr(&mut table), ptr(&mut index)])
}

fn body(buffer: &Buffer, g: &Gpu) -> Result<Vec<u8>> {
    let bytes = buffer.read(g)?;
    if bytes[..16] != GUARD || bytes[bytes.len() - 16..] != GUARD { return Err("CAUSALGROUP guard changed".into()); }
    Ok(bytes[16..bytes.len() - 16].to_vec())
}

fn launch_each(g: &Gpu, kernel: H, chats: &[Chat], blocks: u32) -> Result<()> {
    for chat in chats { launch_replay(g, kernel, chat.command, (blocks, 128))?; }
    Ok(())
}

fn read_all(g: &Gpu, chats: &[Chat]) -> Result<Vec<Vec<u8>>> { chats.iter().map(|chat| body(&chat.output, g)).collect() }

fn poison_all(g: &Gpu, chats: &[Chat]) -> Result<()> { for chat in chats { chat.output.poison(g)?; } Ok(()) }

fn owner_of_slot(slot: usize) -> usize { if slot < 2 { 1 - slot } else { slot } }

pub(super) fn run(g: &mut Gpu, baseline: H, candidate: H) -> Result<()> {
    let base_fast = g.function(baseline, "resident_launch_causal_fast")?;
    let cand_fast = g.function(candidate, "resident_launch_causal_fast")?;
    let grouped = g.function(candidate, "resident_launch_causal_group")?;
    let started = Instant::now();
    let (mut comparisons, mut output_bytes) = (0u64, 0u64);
    for case in shapes() {
        for n in 2..=8u32 {
            let mut chats = Vec::new();
            for chat in 0..n { chats.push(make_chat(g, &case, chat)?); }
            poison_all(g, &chats)?;
            launch_each(g, base_fast, &chats, case.blocks)?;
            g.sync()?;
            let plain = read_all(g, &chats)?;
            if plain[0] == plain[1] { return Err(format!("CAUSALGROUP control chats match op={} chats={n}", case.name)); }
            poison_all(g, &chats)?;
            let members = chats.iter().flat_map(|chat| command(&chat.args, FAST)).collect::<Vec<_>>();
            let member_table = g.upload(&members)?;
            let group = g.upload(&group_bytes(member_table, n))?;
            launch_group(g, grouped, group, case.blocks, n)?;
            g.sync()?;
            let got = read_all(g, &chats)?;
            for (chat, (left, right)) in plain.iter().zip(&got).enumerate() {
                if left != right {
                    let at = left.iter().zip(right).position(|(p, q)| p != q);
                    return Err(format!("CAUSALGROUP BYTE FAIL op={} chats={n} chat={chat} arm=old-fast first_diff_byte={at:?}", case.name));
                }
                comparisons += 1; output_bytes += left.len() as u64;
            }
            poison_all(g, &chats)?;
            launch_each(g, cand_fast, &chats, case.blocks)?;
            g.sync()?;
            let same = read_all(g, &chats)?;
            for (chat, (left, right)) in plain.iter().zip(&same).enumerate() {
                if left != right {
                    let at = left.iter().zip(right).position(|(p, q)| p != q);
                    return Err(format!("CAUSALGROUP BYTE FAIL op={} chats={n} chat={chat} arm=same-image first_diff_byte={at:?}", case.name));
                }
                comparisons += 1; output_bytes += left.len() as u64;
            }
            // The swapped commands write fresh buffers. Slot 0 carries chat 1's inputs. A kernel that
            // ignored the table and followed the plain chat index would leave these buffers empty.
            let (_, _, total) = words_of(&case);
            let mut fresh = Vec::new();
            for _ in 0..n { let buffer = Buffer::new(g, &vec![0.0; total])?; buffer.poison(g)?; fresh.push(buffer); }
            let swapped = (0..n as usize).map(|slot| Arguments { output: fresh[slot].address, ..chats[owner_of_slot(slot)].args }).collect::<Vec<_>>();
            let swapped_table = g.upload(&swapped.iter().flat_map(|args| command(args, FAST)).collect::<Vec<_>>())?;
            let swapped_group = g.upload(&group_bytes(swapped_table, n))?;
            launch_group(g, grouped, swapped_group, case.blocks, n)?;
            g.sync()?;
            let wrong = fresh.iter().map(|buffer| body(buffer, g)).collect::<Result<Vec<_>>>()?;
            let swapped_order = (0..n as usize).map(|slot| plain[owner_of_slot(slot)].clone()).collect::<Vec<_>>();
            if wrong[0] == plain[0] { return Err(format!("CAUSALGROUP WRONG TABLE matched the plain order op={} chats={n}", case.name)); }
            if wrong != swapped_order { return Err(format!("CAUSALGROUP WRONG TABLE did not match the swapped order op={} chats={n}", case.name)); }
            println!("CAUSALGROUP BYTES PASS op={} chats={n} blocks={}", case.name, case.blocks);
            println!("CAUSALGROUP WRONG TABLE PASS op={} chats={n}", case.name);
            g.clear()?;
        }
    }
    chain_screen(g,baseline,candidate)?;
    println!("CAUSALGROUP PASS comparisons={comparisons} output_bytes={output_bytes} elapsed_s={:.1}", started.elapsed().as_secs_f64());
    Ok(())
}

/// Three original argument blocks and all intermediate outputs stay private to a chat.
struct Chain { args:[Arguments;3], outputs:[Buffer;3] }
impl Chain {
    fn member(&self)->Vec<u8> {
        let mut bytes=vec![0u8;1040];bytes[..4].copy_from_slice(&33u32.to_le_bytes());
        for (offset,args) in [0,256,512].into_iter().zip(&self.args) {bytes[16+offset..16+offset+80].copy_from_slice(&command(args,FAST)[16..96]);}bytes
    }
    fn read(&self,g:&Gpu)->Result<Vec<Vec<u8>>> {self.outputs.iter().map(|b|body(b,g)).collect()}
    fn poison(&self,g:&Gpu)->Result<()> {for b in &self.outputs {b.poison(g)?;}Ok(())}
}
fn chain(g:&mut Gpu,chat:u32,heads:u32,kind:u32)->Result<Chain> {
    let h=heads as usize;let d=128;let seed=9000+u64::from(chat)*23;
    let mut buffer=|words,at|Buffer::new(g,&pattern(kind,words,seed+at));
    let qkv=buffer(3*h*d,0)?;let gate=buffer(h*d,1)?;let decay=buffer(h,2)?;let weight=buffer(h+h*d,3)?;
    let state=buffer(h*d*d,4)?;let norm_gate=buffer(h*d,5)?;let norm_weight=buffer(d,6)?;
    let prepared=buffer(h*(4*d+1),7)?;let stepped=buffer(h*(d*d+d),8)?;let normalized=buffer(h*d,9)?;
    let shape=Shape {op:super::PREPARE,heads,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:-1.0};
    let args=[Arguments {input:[qkv.address,gate.address,decay.address,4],weight:weight.address,output:prepared.address,shape},
        Arguments {input:[prepared.address,state.address,4,4],weight:4,output:stepped.address,shape:Shape {op:STEP_PLANAR,bound:0.0,..shape}},
        Arguments {input:[stepped.address+(h*d*d*4) as u64,norm_gate.address,4,4],weight:norm_weight.address,output:normalized.address,shape:Shape {op:super::GATED_NORM,epsilon:1e-6,bound:0.0,..shape}}];
    Ok(Chain {args,outputs:[prepared,stepped,normalized]})
}
fn chain_screen(g:&mut Gpu,baseline:H,candidate:H)->Result<()> {
    let old=g.function(baseline,"resident_launch_causal_chain")?;let new=g.function(candidate,"resident_launch_causal_group")?;
    let mut comparisons=0;
    for heads in [1u32,17,32] {for kind in [0u32,2,6,9] {for n in 2..=8u32 {
        let chats=(0..n).map(|chat|chain(g,chat,heads,kind)).collect::<Result<Vec<_>>>()?;
        for chat in &chats {chat.poison(g)?;let table=g.upload(&chat.member())?;launch_replay(g,old,table,(heads,128))?;}g.sync()?;
        let expected=chats.iter().map(|chat|chat.read(g)).collect::<Result<Vec<_>>>()?;
        for chat in &chats {chat.poison(g)?;}
        let members=g.upload(&chats.iter().flat_map(Chain::member).collect::<Vec<_>>())?;
        let mut bytes=group_bytes(members,n);bytes[28..32].copy_from_slice(&33u32.to_le_bytes());let group=g.upload(&bytes)?;
        launch_group(g,new,group,heads,n)?;g.sync()?;
        let actual=chats.iter().map(|chat|chat.read(g)).collect::<Result<Vec<_>>>()?;
        if expected!=actual {return Err(format!("CAUSALGROUP CHAIN BYTE FAIL heads={heads} pattern={kind} chats={n}"));}comparisons+=n*3;
        // Redirect every output and its dependent input to fresh storage. Swap only the input owners.
        let mut fresh=Vec::new();let mut swapped=Vec::new();
        for slot in 0..n as usize {
            let outputs=[Buffer::new(g,&vec![0.0;heads as usize*513])?,Buffer::new(g,&vec![0.0;heads as usize*16512])?,Buffer::new(g,&vec![0.0;heads as usize*128])?];
            for out in &outputs {out.poison(g)?;}
            let mut args=chats[owner_of_slot(slot)].args;args[0].output=outputs[0].address;args[1].input[0]=outputs[0].address;
            args[1].output=outputs[1].address;args[2].input[0]=outputs[1].address+u64::from(heads)*16384*4;args[2].output=outputs[2].address;
            let redirected=Chain {args,outputs};swapped.extend(redirected.member());fresh.push(redirected);
        }
        let table=g.upload(&swapped)?;let mut bytes=group_bytes(table,n);bytes[28..32].copy_from_slice(&33u32.to_le_bytes());let group=g.upload(&bytes)?;
        launch_group(g,new,group,heads,n)?;g.sync()?;
        let wrong=fresh.iter().map(|chat|chat.read(g)).collect::<Result<Vec<_>>>()?;
        let wanted=(0..n as usize).map(|i|expected[owner_of_slot(i)].clone()).collect::<Vec<_>>();
        if wrong!=wanted {return Err(format!("CAUSALGROUP CHAIN SWAP FAIL heads={heads} pattern={kind} chats={n}"));}
        if kind==0 && wrong[0]==expected[0] {return Err("CAUSALGROUP CHAIN wrong-on-purpose control did not fail".into());}
        println!("CAUSALGROUP CHAIN PASS heads={heads} pattern={kind} chats={n} outputs=3 wrong_table=detected");g.clear()?;
    }}}
    println!("CAUSALGROUP CHAIN PASS comparisons={comparisons}");Ok(())
}
