//! Recorder failure retirement, piece bounds and standalone-program exclusion.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{driver::with_backend,causal_ops::{Shape,Op},resident_program::Program,GraphAllowance};
fn same_error<T>(result:Result<T>,expected:&str)->Result<()> {
    match result {Err(error) if error==expected=>Ok(()),Err(error)=>Err(format!("recorder failure changed: {error}")),Ok(_)=>Err("failed recorder accepted a consumer".into())}
}
fn recorder_budget_failure_is_permanent()->Result<()> {
    let mut checked=false;
    let result=with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let mut input=b.upload_vector(&[1.0])?;let mut output=b.vector(1)?;let mut slot=b.exchange_slot(1)?;b.exchange_begin(&mut slot,&input)?;b.synchronize()?;
        b.set_graph_storage_budget(0)?;let queue=b.resident_queue()?;b.enable_resident_launches()?;
        b.zero_vector(&mut input)?;
        let error=b.exchange_begin(&mut slot,&input).err().ok_or("zero graph budget accepted an exchange before its producer")?;
        if !error.contains("graph storage") {return Err(format!("unexpected recorder failure: {error}"));}
        same_error(b.read_vector_values(&input),&error)?;same_error(b.exchange_wait(&mut slot),&error)?;
        same_error(b.round_bf16_lanes(&input),&error)?;
        same_error(b.copy_prefix(&input,&mut output,1),&error)?;
        same_error(b.synchronize(),&error)?;
        same_error(b.zero_vector(&mut input),&error)?;
        if b.fatal_error().as_deref()!=Some(error.as_str()) {return Err("recorder failure was not retained".into());}
        drop(queue);same_error(b.resident_queue(),&error)?;checked=true;Ok(())
    });
    if !checked {return result.and(Err("recorder failure checks did not finish".into()));}
    if result.is_ok() {return Err("failed recorder backend retired without an error".into());}Ok(())
}
fn recorder_chain_exceeds_graph_limit()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let expected=[1.0f32,-0.0,3.0,-4.0].repeat(8);let input=b.upload_vector(&expected)?;
        let queue=b.resident_queue()?;b.enable_resident_launches()?;let before=b.stats();
        let mut current=input;
        for _ in 0..GraphAllowance::FLASH_MAX_NODES*2+17 {current=b.round_bf16_lanes(&current)?.into_vector();}
        let got=b.read_vector_values(&current)?;
        if got.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err("long recorder chain changed bits".into());}
        if b.stats().since(before).graph_launches<3 {return Err("long recorder chain was not split at the graph limit".into());}
        drop(queue);Ok(())
    })
}
fn standalone_program_rejects_active_recording()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let shape=Shape {op:Op::StreamMean,heads:2,dim:1,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
        let plan=Program::operation(shape)?;let queue=b.resident_queue()?;
        let mut program=b.resident_program(&plan,&[None])?;let input=b.upload_vector(&[1.0,3.0])?;
        let output=program.run(&[&input])?;if b.read_vector_values(&output)?!=[2.0] {return Err("standalone warmup differs".into());}
        let next=b.upload_vector(&[5.0,7.0])?;b.synchronize()?;b.enable_resident_launches()?;
        let error="standalone resident program requires launch recording off";
        same_error(b.resident_program(&plan,&[None]),error)?;
        same_error(program.run(&[&next]),error)?;same_error(program.run_padded(&[&next]),error)?;
        if b.read_vector_values(&output)?!=[2.0] {return Err("rejected replay changed its prior output".into());}
        drop(queue);Ok(())
    })
}
fn recorder_exchange_failure_rejects_readout()->Result<()> {
    let mut checked=false;
    let result=with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let mut input=b.upload_vector(&[1.0])?;let mut slot=b.exchange_slot(1)?;b.synchronize()?;
        let queue=b.resident_queue()?;b.enable_resident_launches()?;b.zero_vector(&mut input)?;
        b.exchange_begin(&mut slot,&input)?;
        if b.exchange_wait(&mut slot)?!=[0.0] {return Err("exchange copied ahead of recorded zero".into());}
        // Keep an older completed download available when a later producer fails.
        b.exchange_begin(&mut slot,&input)?;b.synchronize()?;
        b.set_graph_storage_budget(b.graph_storage_bytes())?;
        let fence=b.fence()?;let side=b.frontend_stream()?;let next=b.upload_vector(&[1.0,2.0])?;b.round_bf16_lanes(&next)?;
        let error=b.read_vector_values(&next).err().ok_or("graph budget accepted a new recorded piece")?;
        if !error.contains("graph storage") {return Err(format!("unexpected recorder failure: {error}"));}
        same_error(b.exchange_wait(&mut slot),&error)?;
        same_error(b.exchange_begin(&mut slot,&input),&error)?;
        same_error(b.read_vector_values(&next),&error)?;
        same_error(b.wait_fence(fence),&error)?;same_error(b.wait_moe_readiness(),&error)?;same_error(b.frontend_join(&side),&error)?;
        same_error(b.read_argmax_status(&next,&next,1),&error)?;same_error(b.read_argmax_results(&next,1),&error)?;
        drop(queue);checked=true;Ok(())
    });
    if !checked {return result.and(Err("exchange failure checks did not finish".into()));}
    if result.is_ok() {return Err("failed exchange backend retired without an error".into());}Ok(())
}
fn recorder_replay_after_poison_is_rejected()->Result<()> {
    let mut checked=false;
    let result=with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let shape=Shape {op:Op::StreamMean,heads:2,dim:1,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
        let plan=Program::operation(shape)?;let queue=b.resident_queue()?;
        let mut program=b.resident_program(&plan,&[None])?;let mut input=b.upload_vector(&[1.0,3.0])?;
        let output=program.run(&[&input])?;
        if b.read_vector_values(&output)?!=[2.0] {return Err("poison replay warmup differs".into());}
        b.set_graph_storage_budget(b.graph_storage_bytes())?;b.enable_resident_launches()?;b.zero_vector(&mut input)?;
        let error=b.read_vector_values(&input).err().ok_or("graph budget accepted replay poison producer")?;
        if !error.contains("graph storage") {return Err(format!("unexpected recorder failure: {error}"));}
        same_error(program.run(&[&input]),&error)?;same_error(program.run_padded(&[&input]),&error)?;
        same_error(b.read_vector_values(&output),&error)?;drop(queue);checked=true;Ok(())
    });
    if !checked {return result.and(Err("poison replay checks did not finish".into()));}
    if result.is_ok() {return Err("poisoned replay backend retired without an error".into());}Ok(())
}
pub fn check(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    recorder_budget_failure_is_permanent()?;recorder_chain_exceeds_graph_limit()?;standalone_program_rejects_active_recording()?;recorder_exchange_failure_rejects_readout()?;recorder_replay_after_poison_is_rejected()?;
    let receipt="RESIDENT RECORDER PASS recorder_budget_failure_is_permanent=true later_read=true later_consumer=true later_flush=true later_queue=true recorder_chain_exceeds_graph_limit=true standalone_program_rejects_active_recording=true recorder_exchange_failure_rejects_readout=true recorder_replay_after_poison_is_rejected=true exact_bits=true\n";
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
