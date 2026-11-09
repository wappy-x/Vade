#![forbid(unsafe_code)]
use super::{Arguments,Buffer,Fixture,Gpu,H,Result,Shape,STEP,STEP_COLUMNS,STEP_LAYOUT,arguments,command,fixture,launch_causal,launch_replay,median,pattern};
use std::time::Instant;

fn cases()->Vec<(u32,u32)> {[1,2,3,17,63,64].into_iter().flat_map(|heads|(0..10).map(move |kind|(heads,kind))).collect()}
fn shape(heads:u32)->Shape {Shape {op:STEP,heads,dim:128,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}}
fn split_args(source:&Fixture,columns:&Buffer,output:&Buffer)->[Arguments;2] {
    let mut first=arguments(source,columns);first.shape.op=STEP_COLUMNS;
    let mut last=arguments(source,output);last.shape.op=STEP_LAYOUT;last.input=[columns.address,4,4,4];[first,last]
}
fn split(gpu:&Gpu,kernel:H,args:&mut [Arguments;2])->Result<()> {
    let heads=args[0].shape.heads;launch_causal(gpu,kernel,&mut args[0],(heads,128))?;
    launch_causal(gpu,kernel,&mut args[1],(heads*17,128))
}
fn compare(expected:&[u8],actual:&[u8],heads:u32,kind:u32,tick:u32,arm:&str)->Result<()> {
    if actual.len()<32 || actual[..16]!=super::GUARD || actual[actual.len()-16..]!=super::GUARD {return Err(format!("STEPDIRECT GUARD FAIL heads={heads} pattern={kind} tick={tick} arm={arm}"));}
    if expected!=actual {
        let byte=expected.iter().zip(actual).position(|(left,right)|left!=right).unwrap_or(expected.len().min(actual.len()));
        let word=byte.saturating_sub(16)/4;let head=word/16512;let lane=word%16512;
        return Err(format!("STEPDIRECT BYTE FAIL heads={heads} pattern={kind} tick={tick} arm={arm} byte={byte} head={head} plane={} word={lane}",if lane<16384 {"state"} else {"output"}));
    }
    Ok(())
}
pub(super) fn run(gpu:&mut Gpu,baseline:H,candidate:H)->Result<()> {
    let base_fast=gpu.function(baseline,"causal_fast")?;let candidate_fast=gpu.function(candidate,"causal_fast")?;
    let base_replay=gpu.function(baseline,"resident_launch_causal_fast")?;let candidate_replay=gpu.function(candidate,"resident_launch_causal_fast")?;
    let started=Instant::now();let mut comparisons=0u64;let mut compared_bytes=0u64;
    for (heads,kind) in cases() {
        let mut source=fixture(gpu,shape(heads),kind,0,9000+u64::from(heads)*10+u64::from(kind))?;
        if kind==9 {for index in 0..2 {let words=if index==0 {heads as usize*513} else {heads as usize*16384};source.inputs[index]=Buffer::new(gpu,&pattern(9,words,1))?;}}
        let words=source.units*source.words;
        let columns=Buffer::new(gpu,&vec![0.0;words])?;let expected=Buffer::new(gpu,&vec![0.0;words])?;
        let direct=Buffer::new(gpu,&vec![0.0;words])?;let replay=Buffer::new(gpu,&vec![0.0;words])?;let composed=Buffer::new(gpu,&vec![0.0;words])?;
        let mut base_args=split_args(&source,&columns,&expected);let mut old_args=split_args(&source,&columns,&composed);
        let mut direct_args=arguments(&source,&direct);let replay_args=arguments(&source,&replay);
        let table=gpu.upload(&command(&replay_args,19))?;
        let prepared=source.inputs[0].read(gpu)?;let mut old_state=source.inputs[1].read(gpu)?;
        for tick in 0..3 {
            for output in [&columns,&expected,&direct,&replay,&composed] {output.poison(gpu)?;}
            split(gpu,base_fast,&mut base_args)?;launch_causal(gpu,candidate_fast,&mut direct_args,(heads,128))?;
            launch_replay(gpu,candidate_replay,table,(heads,128))?;split(gpu,candidate_fast,&mut old_args)?;gpu.sync()?;
            let reference=expected.read(gpu)?;
            compare(&reference,&reference,heads,kind,tick,"baseline-guards")?;
            for (name,output) in [("direct",&direct),("replay",&replay),("candidate-split",&composed)] {
                compare(&reference,&output.read(gpu)?,heads,kind,tick,name)?;comparisons+=1;compared_bytes+=(words*4) as u64;
            }
            compare(&prepared,&source.inputs[0].read(gpu)?,heads,kind,tick,"prepared-input")?;
            compare(&old_state,&source.inputs[1].read(gpu)?,heads,kind,tick,"old-state-input")?;
            let payload=&reference[16..reference.len()-16];let mut next=Vec::with_capacity(heads as usize*16384*4);
            for head in payload.chunks_exact(16512*4) {next.extend_from_slice(&head[..16384*4]);}
            gpu.upload_at(source.inputs[1].address,&next)?;old_state=[super::GUARD.as_slice(),&next,super::GUARD.as_slice()].concat();
        }
        println!("STEPDIRECT BYTES PASS heads={heads} dim=128 pattern={kind} ticks=3 arms=direct,replay,candidate-split guards=PASS inputs=UNCHANGED");
        gpu.clear()?;
    }
    println!("STEPDIRECT BYTES SUMMARY cases={} comparisons={comparisons} output_bytes={compared_bytes}",cases().len());
    for copies in [1usize,24] {
        let mut cells=Vec::new();
        for index in 0..copies {
            let source=fixture(gpu,shape(64),0,0,10000+index as u64)?;let words=source.units*source.words;
            let columns=Buffer::new(gpu,&vec![0.0;words])?;let old=Buffer::new(gpu,&vec![0.0;words])?;let new=Buffer::new(gpu,&vec![0.0;words])?;
            let [first,last]=split_args(&source,&columns,&old);let next=arguments(&source,&new);
            cells.push((gpu.upload(&command(&first,19))?,gpu.upload(&command(&last,19))?,gpu.upload(&command(&next,19))?));
        }
        let mut times=[Vec::new(),Vec::new()];
        for repeat in 0..11 {for ordinal in 0..2 {
            let arm=if repeat%2==0 {ordinal} else {1-ordinal};
            let launch=|index:usize|->Result<()> {let cell=cells[index%copies];if arm==0 {launch_replay(gpu,base_replay,cell.0,(64,128))?;launch_replay(gpu,base_replay,cell.1,(64*17,128))} else {launch_replay(gpu,candidate_replay,cell.2,(64,128))}};
            for index in 0..copies {launch(index)?;}gpu.sync()?;
            let began=Instant::now();for index in 0..480 {launch(index)?;}gpu.sync()?;
            times[arm].push(began.elapsed().as_secs_f64()*1e6/480.0);
        }}
        let old=median(times[0].clone());let new=median(times[1].clone());
        println!("STEPDIRECT TIME heads=64 dim=128 copies={copies} split_us={old:.3} direct_us={new:.3} saved_us={:.3} saved_ms_per_34={:.6} speedup={:.6}",old-new,(old-new)*34.0/1000.0,old/new);
        gpu.clear()?;
    }
    println!("STEPDIRECT PASS elapsed_s={:.3}",started.elapsed().as_secs_f64());Ok(())
}
#[cfg(test)] mod tests {
    #[test] fn covers_every_direct_shape_and_special_value_case() {
        let cases=super::cases();assert_eq!(cases.len(),60);
        for heads in [1,2,3,17,63,64] {for kind in 0..10 {assert_eq!(cases.iter().filter(|&&case|case==(heads,kind)).count(),1);}}
        assert!(cases.iter().all(|&(heads,_)|super::shape(heads).dim==128));
        assert_eq!(super::pattern(9,10,1).iter().map(|value|value.to_bits()).collect::<Vec<_>>(),vec![0x7fc1_2345,0xffc2_3456,0x7f81_2345,0x7f80_0000,0xff80_0000,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_0000]);
    }
    #[test] fn mismatch_control_reports_the_plane_and_word() {
        let expected=[super::super::GUARD.as_slice(),&vec![0u8;16512*4],super::super::GUARD.as_slice()].concat();let mut actual=expected.clone();actual[16+16384*4]=1;
        let error=super::compare(&expected,&actual,1,0,0,"control").unwrap_err();assert!(error.contains("plane=output word=16384"));
    }
}
