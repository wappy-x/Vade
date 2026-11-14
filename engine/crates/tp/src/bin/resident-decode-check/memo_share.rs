//! One box, small checkpoint, the real served stack (Factory, Split, Device, resident queue): what a chat pays when
//! another chat opens while it decodes.
//!
//! Chat A has a 4,158-token prompt and decodes 12 steps. Chat B (1,040 tokens) opens after A's step 3 and chat C
//! (127 tokens) after A's step 8; B and C read their prompts and decode between A's steps, as the served scheduler
//! runs them. Each step reports the memo rows it found and the rows it had to expand again, the memo's bytes and
//! limit, the answers made ahead of the selection read and those used, and the host time.
//!
//! Three arms, each in its own backend. SOLO: every chat alone, for the logits and for A's counts. CONTROL
//! (Device::set_memo_share(false)): every session that opens resets the memo, the form before MEMO-SHARE: A's first
//! step after each open must find no row and expand its selection again, or this check can prove nothing. SHARED:
//! A's found and added rows and its answers made ahead are those of the solo run at every step, and the limit does
//! not move at an open.
//! NINTH: chat A, seven short chats, one of them ends, a ninth session selects: A's asked list stays (the list of the
//! chat that ended goes), so A's next step makes the answers ahead of its solo run.
//! SLICED: one chat whose prompt comes as one token and then the rest, so its cache's first span is one row: with the
//! fix every gather after the first step finds the chat's own asked list; the control (the list stored under the
//! gather's output, the first build's fault) must miss.
//! Logits of every chat are equal to its solo run in both arms, bit for bit. The small checkpoint's rows are
//! kilobytes where the model's are megabytes: the times here show the direction, not the model's milliseconds.
use super::*;
use super::first_step::{checkpoint_typed,prompt_tokens,OneRank};
use spark_tp::{bounded_decoder::Device,row_split::{Factory,Split}};
use spark_serve::{Engine,Session};

const A_STEPS:usize=12;
/// One decode step: chat, its step number, memo rows found and added, memo bytes and limit after it, answers made
/// ahead and used, host ms.
#[derive(Clone,Copy,PartialEq,Debug)]
struct Step {chat:char,at:usize,found:u64,added:u64,bytes:usize,limit:usize,ahead:u64,used:u64,ms:f64}
type Bits=Vec<Vec<u32>>;
fn token(step:usize)->u32 {1+(step as u32*5)%15}
/// One decode step of one chat, with the memo's counts around it.
#[allow(clippy::too_many_arguments)]
fn one<S:Session>(d:&Device<'_,'_>,name:&str,s:&mut S,chat:char,at:usize,out:&mut Bits,steps:&mut Vec<Step>,lines:&mut String)->Result<()> {
    let (m0,a0)=(d.memo_counts(),d.ahead_counts());let t=std::time::Instant::now();let l=s.step(token(at))?;let ms=t.elapsed().as_secs_f64()*1000.0;
    let (m1,a1)=(d.memo_counts(),d.ahead_counts());out.push(l.iter().map(|v|v.to_bits()).collect());
    let one=Step {chat,at,found:m1.2-m0.2,added:m1.3-m0.3,bytes:m1.0,limit:m1.1,ahead:a1[0]-a0[0],used:a1[1]-a0[1],ms};
    let line=format!("MEMO_SHARE arm={name} chat={chat} step={at} rows_found={} rows_added={} memo_bytes={} memo_limit={} ahead_made={} ahead_used={} host_ms={ms:.3}\n",
        one.found,one.added,one.bytes,one.limit,one.ahead,one.used);
    print!("{line}");lines.push_str(&line);steps.push(one);Ok(())
}
/// Chats A, B and C in served order (`others` false: A alone). Returns each chat's logits, every decode step, and
/// the memo's (bytes, limit) before and after each open.
#[allow(clippy::type_complexity)]
fn arm(c:&super::select_known::attention::Checkpoint,name:&str,share:bool,others:bool,lines:&mut String)->Result<([Bits;3],Vec<Step>,Vec<[(usize,usize);2]>)> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        b.configure_vector_pool(1<<30);let d=Device::new(b);d.set_memo_share(share);
        let split=Split::new(&d,OneRank)?;let service=Factory::new(&c.admission,split,4200,vec![0])?;service.negotiate_available_bytes(1usize<<40)?;
        let mut logits:[Bits;3]=Default::default();let mut steps=Vec::new();let mut opens=Vec::new();
        let state=|d:&Device<'_,'_>|{let m=d.memo_counts();(m.0,m.1)};
        let mut step=|s:&mut _,chat:char,at:usize,out:&mut Bits,lines:&mut String|one(&d,name,s,chat,at,out,&mut steps,lines);
        let mut a=service.session()?;logits[0].push(a.prefill(&prompt_tokens(4158))?.iter().map(|v|v.to_bits()).collect());
        for at in 0..4 {step(&mut a,'A',at,&mut logits[0],lines)?;}
        if !others {for at in 4..A_STEPS {step(&mut a,'A',at,&mut logits[0],lines)?;}drop(step);drop(a);service.shutdown()?;return Ok((logits,steps,opens));}
        let before=state(&d);let mut second=service.session()?;opens.push([before,state(&d)]);
        for at in 4..6 {step(&mut a,'A',at,&mut logits[0],lines)?;}
        logits[1].push(second.prefill(&prompt_tokens(1040))?.iter().map(|v|v.to_bits()).collect());
        for round in 0..3 {step(&mut a,'A',6+round,&mut logits[0],lines)?;step(&mut second,'B',round,&mut logits[1],lines)?;}
        let before=state(&d);let mut third=service.session()?;opens.push([before,state(&d)]);
        step(&mut a,'A',9,&mut logits[0],lines)?;step(&mut second,'B',3,&mut logits[1],lines)?;
        logits[2].push(third.prefill(&prompt_tokens(127))?.iter().map(|v|v.to_bits()).collect());
        for round in 0..2 {step(&mut a,'A',10+round,&mut logits[0],lines)?;step(&mut second,'B',4+round,&mut logits[1],lines)?;step(&mut third,'C',round,&mut logits[2],lines)?;}
        for (open,[before,after]) in opens.iter().enumerate() {
            let line=format!("MEMO_SHARE arm={name} open={} memo_bytes_before={} memo_bytes_after={} memo_limit_before={} memo_limit_after={}\n",
                ['B','C'][open],before.0,after.0,before.1,after.1);
            print!("{line}");lines.push_str(&line);
        }
        drop(step);drop(third);drop(second);drop(a);service.shutdown()?;Ok((logits,steps,opens))
    })
}
/// Nine sessions and eight lists. Chat A selects, then seven short chats open and select, so A's asked list is the one
/// written longest ago. One short chat ends and a ninth session selects: the list of the chat that ended must go, not
/// A's. Returns A's logits, every step, and the most asked lists a projection held before and after the ninth.
#[allow(clippy::type_complexity)]
fn ninth(c:&super::select_known::attention::Checkpoint,lines:&mut String)->Result<(Bits,Vec<Step>,[usize;2])> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        b.configure_vector_pool(1<<30);let d=Device::new(b);let split=Split::new(&d,OneRank)?;
        let service=Factory::new(&c.admission,split,4200,vec![0])?;service.negotiate_available_bytes(1usize<<40)?;
        let mut logits:Bits=Vec::new();let mut other:Bits=Vec::new();let mut steps=Vec::new();
        let mut a=service.session()?;logits.push(a.prefill(&prompt_tokens(4158))?.iter().map(|v|v.to_bits()).collect());
        for at in 0..2 {one(&d,"ninth",&mut a,'A',at,&mut logits,&mut steps,lines)?;}
        let mut short=Vec::new();
        for _ in 0..7 {
            let mut s=service.session()?;s.prefill(&prompt_tokens(127))?;one(&d,"ninth",&mut s,'F',0,&mut other,&mut steps,lines)?;short.push(s);
        }
        let full=d.asked_lists();drop(short.pop());
        let mut last=service.session()?;last.prefill(&prompt_tokens(127))?;one(&d,"ninth",&mut last,'N',0,&mut other,&mut steps,lines)?;
        let after=d.asked_lists();
        one(&d,"ninth",&mut a,'A',2,&mut logits,&mut steps,lines)?;
        drop(last);drop(short);drop(a);service.shutdown()?;Ok((logits,steps,[full,after]))
    })
}
/// A chat whose prompt comes in two slices, one token and then the rest. Its cache's first span is ONE row, so a
/// selection that does not hold row 0 starts past the first span. `by_output`: the asked list is stored under the
/// first span of the gather's output (the fault of the first build, the control). Returns the logits and, for each
/// decode step, the gathers that looked for the session's asked list and those that found it.
fn sliced(c:&super::select_known::attention::Checkpoint,name:&str,by_output:bool,lines:&mut String)->Result<(Bits,Vec<[u64;2]>)> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        b.configure_vector_pool(1<<30);let d=Device::new(b);d.set_asked_by_output(by_output);let split=Split::new(&d,OneRank)?;
        let service=Factory::new(&c.admission,split,4200,vec![0])?;service.negotiate_available_bytes(1usize<<40)?;
        let prompt=prompt_tokens(4158);let mut s=service.session()?;
        if s.prefill_slice(&prompt[..1],false,None,prompt.len())?.is_none() {return Err("memo-share: the first prompt slice gave no logits".into());}
        let first=s.prefill_slice(&prompt[1..],true,None,prompt.len())?.ok_or("memo-share: the last prompt slice gave no logits")?;
        let mut logits:Bits=vec![first.iter().map(|v|v.to_bits()).collect()];let mut counts=Vec::new();
        for at in 0..6 {
            let before=d.asked_counts();logits.push(s.step(token(at))?.iter().map(|v|v.to_bits()).collect());let after=d.asked_counts();
            let step=[after[0]-before[0],after[1]-before[1]];
            let line=format!("MEMO_SHARE arm={name} chat=S step={at} gathers_that_looked={} found_their_list={}\n",step[0],step[1]);
            print!("{line}");lines.push_str(&line);counts.push(step);
        }
        drop(s);service.shutdown()?;Ok((logits,counts))
    })
}
/// One chat alone: prompt, then `count` decode steps.
fn alone(c:&super::select_known::attention::Checkpoint,prompt:usize,count:usize)->Result<Bits> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        b.configure_vector_pool(1<<30);let d=Device::new(b);let split=Split::new(&d,OneRank)?;
        let service=Factory::new(&c.admission,split,4200,vec![0])?;service.negotiate_available_bytes(1usize<<40)?;
        let mut s=service.session()?;let mut out:Bits=vec![s.prefill(&prompt_tokens(prompt))?.iter().map(|v|v.to_bits()).collect()];
        for at in 0..count {out.push(s.step(token(at))?.iter().map(|v|v.to_bits()).collect());}
        drop(s);service.shutdown()?;Ok(out)
    })
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut lines=String::new();
    // The memo holds expanded rows. An absorbed latent layer (BF16 kv_b_proj) makes none, so the check runs on the expanded
    // path: kv_b_proj as F32, outside latent absorption's admission (as select-known runs its expanded arm).
    let c=checkpoint_typed(11,4200,true);
    let (solo,solo_steps,_)=arm(&c,"solo",true,false,&mut lines)?;
    let (solo_b,solo_c)=(alone(&c,1040,6)?,alone(&c,127,2)?);
    let (control,control_steps,control_opens)=arm(&c,"control",false,true,&mut lines)?;
    let (shared,shared_steps,shared_opens)=arm(&c,"shared",true,true,&mut lines)?;
    for (name,got) in [("control",&control),("shared",&shared)] {
        if got[0]!=solo[0] {return Err(format!("memo-share: chat A's logits in the {name} arm differ from its solo run"));}
        if got[1]!=solo_b || got[2]!=solo_c {return Err(format!("memo-share: chat B's or C's logits in the {name} arm differ from the solo run"));}
    }
    let of=|steps:&[Step],chat:char|steps.iter().filter(|s|s.chat==chat).copied().collect::<Vec<_>>();
    let (solo_a,control_a,shared_a)=(of(&solo_steps,'A'),of(&control_steps,'A'),of(&shared_steps,'A'));
    // The control must show the reset: A's first step after each open finds no row and expands its selection again.
    for at in [4usize,9] {
        if control_a[at].found!=0 || control_a[at].added<=solo_a[at].added {
            return Err(format!("memo-share: the control arm shows no memo reset at A's step {at} ({:?}); the check proves nothing",control_a[at]));
        }
    }
    // Shared: a chat that opens takes no row of A and does not move the limit.
    for (mine,alone) in shared_a.iter().zip(&solo_a) {
        if (mine.found,mine.added)!=(alone.found,alone.added) {return Err(format!("memo-share: A's rows differ from its solo run at step {}: {mine:?} against {alone:?}",mine.at));}
        // The answers made ahead of a selection read come from A's own asked list, also between the steps of B and C.
        if (mine.ahead,mine.used)!=(alone.ahead,alone.used) {
            return Err(format!("memo-share: A's answers ahead differ from its solo run at step {}: {mine:?} against {alone:?}",mine.at));
        }
    }
    if shared_opens.iter().any(|[before,after]|after.1!=before.1 || after.0<before.0) {return Err(format!("memo-share: an open moved the limit or took rows: {shared_opens:?}"));}
    // Nine sessions: the list of a chat that ended goes, the list of the chat that waited longest stays.
    let (ninth_logits,ninth_steps,lists)=ninth(&c,&mut lines)?;
    if ninth_logits[..]!=solo[0][..ninth_logits.len()] {return Err("memo-share: chat A's logits beside eight other sessions differ from its solo run".into());}
    if lists!=[8,8] {return Err(format!("memo-share: the nine-session case did not hold eight asked lists before and after the ninth: {lists:?}"));}
    let waited=ninth_steps.iter().rfind(|s|s.chat=='A').copied().ok_or("memo-share: chat A has no step in the nine-session case")?;
    if (waited.found,waited.added,waited.ahead,waited.used)!=(solo_a[2].found,solo_a[2].added,solo_a[2].ahead,solo_a[2].used) {
        return Err(format!("memo-share: after a ninth session A's step differs from its solo run: {waited:?} against {:?}",solo_a[2]));
    }
    // A prompt in two slices (first span of one row). From its second step on, every gather finds the list this chat
    // wrote at the step before. The control (list stored under the gather output) must miss, or this proves nothing.
    let (control_sliced,control_counts)=sliced(&c,"sliced-control",true,&mut lines)?;
    let (fixed_sliced,fixed_counts)=sliced(&c,"sliced",false,&mut lines)?;
    if control_sliced!=fixed_sliced {return Err("memo-share: the sliced chat's logits differ between the control and the fix".into());}
    let total=|counts:&[[u64;2]]|counts.iter().skip(1).fold([0u64;2],|sum,step|[sum[0]+step[0],sum[1]+step[1]]);
    let (control_sum,fixed_sum)=(total(&control_counts),total(&fixed_counts));
    if fixed_sum[0]==0 || fixed_counts.iter().skip(1).any(|step|step[1]!=step[0]) {
        return Err(format!("memo-share: the sliced chat did not find its asked list at every gather after its first step: {fixed_counts:?}"));
    }
    if control_sum[1]>=control_sum[0] {
        return Err(format!("memo-share: the control of the sliced case shows no missed list ({control_counts:?}); the check proves nothing"));
    }
    let sliced_as_whole=u8::from(fixed_sliced[..]==solo[0][..fixed_sliced.len()]);
    let median=|steps:&[Step]|{let mut v=steps.iter().filter(|s|![0,4,9].contains(&s.at)).map(|s|s.ms).collect::<Vec<_>>();v.sort_by(f64::total_cmp);v[v.len()/2]};
    let again=|steps:&[Step]|steps[4].added+steps[9].added;
    let used=|steps:&[Step]|{let made=steps.iter().map(|s|s.ahead).sum::<u64>();(steps.iter().map(|s|s.used).sum::<u64>(),made)};
    let line=format!("MEMO_SHARE COMPLETE logits_equal_to_solo=1 control_rows_found_after_open={},{} control_rows_added_after_open={},{} shared_rows_added_after_open={},{} \
        solo_rows_added_at_those_steps={},{} a_step_ms_after_open control={:.3},{:.3} shared={:.3},{:.3} a_steady_ms control={:.3} shared={:.3} solo={:.3} \
        rows_expanded_again control={} shared={} a_ahead_used_of_made solo={:?} control={:?} shared={:?} control_limit={:?} shared_limit={:?} \
        ninth_session_lists={lists:?} a_ahead_after_ninth={}_of_{} \
        sliced_control_found={}_of_{} sliced_found={}_of_{} sliced_logits_equal_to_one_slice={sliced_as_whole} scope=one_box_fixture_not_served_grade\n",
        control_a[4].found,control_a[9].found,control_a[4].added,control_a[9].added,shared_a[4].added,shared_a[9].added,solo_a[4].added,solo_a[9].added,
        control_a[4].ms,control_a[9].ms,shared_a[4].ms,shared_a[9].ms,median(&control_a),median(&shared_a),median(&solo_a),
        again(&control_a)-again(&solo_a),again(&shared_a)-again(&solo_a),used(&solo_a),used(&control_a),used(&shared_a),
        control_opens.iter().map(|o|(o[0].1,o[1].1)).collect::<Vec<_>>(),shared_opens.iter().map(|o|(o[0].1,o[1].1)).collect::<Vec<_>>(),
        waited.used,waited.ahead,control_sum[1],control_sum[0],fixed_sum[1],fixed_sum[0]);
    print!("{line}");lines.push_str(&line);
    write(&out.join("MEMO_SHARE.txt"),lines.as_bytes())?;write(&out.join("COMPLETE"),b"MEMO SHARE PASS\n")
}
