//! Recorder failure retirement, piece bounds and standalone-program exclusion.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{driver::with_backend,causal_ops::{Shape,Op},resident_program::Program,GraphAllowance};
#[path="persistent_step.rs"] mod persistent_step;
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
fn recorder_retains_small_graph_working_set()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let inputs=(1..=160).map(|blocks|b.upload_vector(&vec![1.25;blocks*256])).collect::<Result<Vec<_>>>()?;b.synchronize()?;
        let queue=b.resident_queue()?;b.enable_resident_launches()?;
        for repeat in 0..2 {
            let before=b.stats();
            for input in &inputs {
                let output=b.round_bf16_lanes(input)?.into_vector();let got=b.read_vector_values(&output)?;
                if got.iter().any(|value|value.to_bits()!=1.25f32.to_bits()) {return Err("small graph working set changed bits".into());}
            }
            if repeat==1 && b.stats().since(before).graph_captures!=0 {return Err("small graph working set was recaptured".into());}
        }
        if b.resident_copy_counts().piece_evictions!=0 {return Err("small graph working set was evicted".into());}
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
fn recorder_mailbox_preserves_mixed_capture_failure()->Result<()> {
    let mut checked=false;
    let result=with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let mailbox=b.tp_mailbox(8)?;let input=b.upload_vector(&[1.0;32])?;let weight=b.upload_vector(&[1.0;64])?;let mut output=b.vector(2)?;
        b.synchronize()?;b.set_graph_storage_budget(0)?;let queue=b.resident_queue()?;b.enable_resident_launches()?;
        b.resident_projection_batch(&[spark_gpu_probe::driver::ResidentProjection::Plain {rows:2,cols:32,weight:&weight,input:&input}],&mut output,false)?;
        let error=b.read_vector_values(&output).err().ok_or("graph budget accepted mixed mailbox producer")?;
        if !error.contains("graph storage") {return Err(format!("unexpected mixed recorder failure: {error}"));}
        for _ in 0..2 {
            same_error(b.tp_mailbox_take(&mailbox,&mut output,1,2,1,1,1,1),&error)?;
            same_error(b.tp_mailbox_take_from(&mailbox,&mut output,1,2,1,1,0,1,1),&error)?;
            same_error(b.tp_mailbox_put_finite(&mailbox,&output,1,2,1,0,1),&error)?;
            same_error(b.tp_mailbox_put(&mailbox,&output,1,2,1,0,1),&error)?;
        }
        drop(queue);checked=true;Ok(())
    });
    if !checked {return result.and(Err("mixed mailbox failure checks did not finish".into()));}
    if result.is_ok() {return Err("failed mixed mailbox backend retired without an error".into());}Ok(())
}
/// Recorded joins keep their copy rows in the recorder's area. They must not end
/// the piece, and the area must start again when it is full.
fn recorder_join_keeps_one_piece()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let host=(0..5019usize).map(|i|(0..1+i%7).map(|j|((i*131+j*17)%1000) as f32*0.25-100.0).collect::<Vec<_>>()).collect::<Vec<_>>();
        let sources=host.iter().map(|values|b.upload_vector(values)).collect::<std::result::Result<Vec<_>,_>>()?;b.synchronize()?;
        // Twelve joins of 5000 rows hold 1.44 MB of copy rows: more than the 1 MiB area.
        let counts=[1usize,2,3,46,64,300,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,46,2];
        // Parts of unequal length: the long ones are cut into rows of one length.
        let long=(0..70_001usize).map(|i|((i*7)%2003) as f32*0.5-300.0).collect::<Vec<_>>();let wide=b.upload_vector(&long)?;b.synchronize()?;
        let mut seen:Option<Vec<Vec<u32>>>=None;
        for queued in [false,true] {
            let queue=if queued {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
            let before=b.stats();let mut table=None;let mut outputs=Vec::new();
            for (round,&count) in counts.iter().enumerate() {
                let parts=sources[round..round+count].iter().map(|v|(v,v.len())).collect::<Vec<_>>();
                let joined=b.resident_join(&parts,&mut table,0)?;
                // A later kernel of the same piece reads the joined words.
                outputs.push(b.round_bf16_lanes(&joined)?.into_vector());outputs.push(joined);
            }
            if queued && b.stats().since(before).graph_launches!=1 {return Err(format!("recorded joins ended {} pieces before the area was full",b.stats().since(before).graph_launches));}
            let mut bits=Vec::new();for v in &outputs {bits.push(b.read_vector_values(v)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>());}
            let launches=b.stats().since(before).graph_launches;
            if queued && launches!=2 {return Err(format!("recorded joins used {launches} pieces, expected 2"));}
            if !queued && launches!=0 {return Err("ordinary joins replayed a graph".into());}
            for (round,&count) in counts.iter().enumerate() {
                let expected=host[round..round+count].iter().flatten().map(|v|v.to_bits()).collect::<Vec<_>>();
                if bits[2*round+1]!=expected {return Err(format!("join {round} of {count} rows changed bits queued={queued}"));}
            }
            let uneven=b.resident_join(&[(&sources[3],sources[3].len()),(&wide,70_001),(&sources[6],sources[6].len()),(&wide,129),(&wide,40_000)],&mut table,0)?;
            let expected=host[3].iter().chain(&long).chain(&host[6]).chain(&long[..129]).chain(&long[..40_000]).map(|v|v.to_bits()).collect::<Vec<_>>();
            if b.read_vector_values(&uneven)?.iter().map(|v|v.to_bits()).ne(expected.iter().copied()) {return Err(format!("join of unequal parts changed bits queued={queued}"));}
            if let Some(first)=&seen&& first!=&bits {return Err("recorded joins differ from ordinary joins".into());}
            seen=Some(bits);drop(queue);
        }
        Ok(())
    })
}
/// A recorded standalone norm replays its three kernels inside the piece and
/// matches the ordinary norm bit for bit, for both weight precisions.
fn recorder_norm_stays_in_the_piece()->Result<()> {
    use spark_gpu_probe::norm_boundary::{NormShape,WeightType};
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for (width,kind) in [(1usize,WeightType::F32),(127,WeightType::Bf16),(128,WeightType::F32),(1536,WeightType::Bf16),(4097,WeightType::F32)] {
            let shape=NormShape::new(width,1e-6,kind)?;
            let weight=(0..width).map(|i|0.5+(i%13) as f32*0.125).collect::<Vec<_>>();
            let values=(0..width).map(|i|if i%11==0 {-0.0} else {((i*37)%257) as f32*0.0625-8.0}).collect::<Vec<_>>();
            let norm=b.resident_norm(shape,&weight)?;let source=b.upload_vector(&values)?;b.synchronize()?;
            let mut seen:Option<Vec<u32>>=None;
            for queued in [false,true] {
                let queue=if queued {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
                let before=b.stats();let input=b.round_bf16_lanes(&source)?;let first=norm.apply(&input)?;
                // A second norm of the first output must enter the same piece.
                let again=b.round_bf16_lanes(first.vector())?;let second=norm.apply(&again)?;
                if queued && b.stats().since(before).graph_launches!=0 {return Err("a recorded norm ended its piece".into());}
                let mut bits=b.read_vector_values(first.vector())?.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
                bits.extend(b.read_vector_values(second.vector())?.iter().map(|v|v.to_bits()));
                let launches=b.stats().since(before).graph_launches;
                if queued && launches!=1 {return Err(format!("recorded norms used {launches} pieces, expected 1"));}
                if let Some(first)=&seen&& first!=&bits {return Err(format!("recorded norm differs from the ordinary norm width={width}"));}
                seen=Some(bits);drop(queue);
            }
        }
        Ok(())
    })
}
/// Zero fills of fresh outputs share one kernel at the start of the piece. A
/// zero fill of words that an earlier command of the piece wrote keeps its place.
fn recorder_zero_fills_keep_their_order()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let sizes=[1usize,127,128,129,4352,32768,32769,7,4096];
        let pattern=|n:usize|(0..n).map(|i|f32::from_bits(0x7fc0_0000|(i as u32&0xffff))).collect::<Vec<_>>();
        let source=b.upload_vector(&[3.0,-0.0,5.5,-7.25,1.0,2.0,4.0,8.0])?;
        let queue=b.resident_queue()?;b.enable_resident_launches()?;
        for round in 0..3 {
            // Dirty the allocator first: later fresh outputs reuse these words.
            for &n in &sizes {let dirty=b.upload_vector(&pattern(n))?;drop(dirty);}b.synchronize()?;
            let before=b.stats();let mut fresh=Vec::new();
            for &n in &sizes {let mut v=b.vector(n)?;b.zero_vector(&mut v)?;fresh.push(b.round_bf16_lanes(&v)?.into_vector());fresh.push(v);}
            // Written then cleared: the clear must stay after the copy.
            let mut late=b.vector(8)?;b.copy_prefix(&source,&mut late,8)?;b.zero_vector(&mut late)?;
            // Cleared then written: the hoisted clear must stay before the copy.
            let mut early=b.vector(8)?;b.zero_vector(&mut early)?;b.copy_prefix(&source,&mut early,5)?;
            if b.stats().since(before).graph_launches!=0 {return Err("zero fills ended the recorded piece".into());}
            for (index,v) in fresh.iter().enumerate() {if b.read_vector_values(v)?.iter().any(|v|v.to_bits()!=0) {return Err(format!("fresh output {index} is not zero in round {round}"));}}
            if b.stats().since(before).graph_launches!=1 {return Err("zero fills used more than one piece".into());}
            if b.read_vector_values(&late)?.iter().any(|v|v.to_bits()!=0) {return Err("a late zero fill ran before its writer".into());}
            let got=b.read_vector_values(&early)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            let expected=[3.0f32,-0.0,5.5,-7.25,1.0,0.0,0.0,0.0].map(f32::to_bits);
            if got!=expected {return Err("a hoisted zero fill ran after its writer".into());}
        }
        drop(queue);Ok(())
    })
}
/// The pool trim releases spares from the front until the room fits under the cap, does
/// nothing when the pool is off or has the room, and leaves the other spares for reuse.
fn pool_trim_releases_spares_down_to_the_room()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,6<<30,|b| {
        let live=b.allocated_bytes();let sizes=(0..12usize).map(|i|(25+i)*1_000_000).collect::<Vec<_>>();
        let bytes=|range:std::ops::Range<usize>|sizes[range].iter().sum::<usize>()*4;
        // Pool off: a dropped vector is freed, and the trim has nothing to do.
        drop(b.vector(sizes[0])?);b.synchronize()?;b.trim_vector_spares(1<<30);
        if b.allocated_bytes()!=live {return Err("pool off: a spare was kept or the trim changed the ledger".into());}
        b.configure_vector_pool(1<<30);
        drop(sizes.iter().map(|&n|b.vector(n)).collect::<std::result::Result<Vec<_>,_>>()?);b.synchronize()?;
        if b.allocated_bytes()!=live+bytes(0..12) {return Err(format!("the pool kept {} of {} bytes",b.allocated_bytes()-live,bytes(0..12)));}
        // 1.464 GB of spares under a 2.0 GB cap: no room is missing, nothing leaves.
        b.trim_vector_spares(500_000_000);
        if b.allocated_bytes()!=live+bytes(0..12) {return Err("the trim released spares though the room was there".into());}
        // 1 GiB of room: the first five spares (540 MB) leave, 924 MB stay.
        b.trim_vector_spares(1<<30);
        if b.allocated_bytes()!=live+bytes(5..12) {return Err(format!("after the trim {} spare bytes stay, expected {}",b.allocated_bytes()-live,bytes(5..12)));}
        // A spare that stayed serves again; a size that left is allocated anew.
        let kept=b.vector(sizes[11])?;if b.allocated_bytes()!=live+bytes(5..12) {return Err("a kept spare was not reused".into());}
        let fresh=b.vector(sizes[0])?;if b.allocated_bytes()!=live+bytes(5..12)+sizes[0]*4 {return Err("a released size was not allocated anew".into());}
        drop((kept,fresh));b.synchronize()?;Ok(())
    })
}
/// Held rows are appended in place only past every live view of their storage: a prefix, a
/// suffix, a view inside a suffix, a rollback prefix and a competing branch each keep their
/// words. With the recorder off and on. An append in place keeps the device address of the held rows.
fn held_rows_append_only_past_every_live_view()->Result<()> {
    use spark_tp::bounded_decoder::{Compute,Device,Value};
    // One append with room for 4 words, and whether it copied the held rows to another place.
    fn extend<'b,'d>(d:&Device<'b,'d>,held:Option<&Value<'b,'d>>,rows:usize,new:&[f32])->Result<(Value<'b,'d>,bool)> {
        let new=d.upload(new)?;let value=d.rows_extend(held,rows,&new,1,4)?;
        let copied=match held {Some(held)=>held.device_address()!=value.device_address(),None=>true};Ok((value,copied))
    }
    fn same<'b,'d>(d:&Device<'b,'d>,v:&Value<'b,'d>,want:&[f32],what:&str)->Result<()> {
        if d.read(v)?.iter().map(|x|x.to_bits()).ne(want.iter().map(|x|x.to_bits())) {Err(format!("held rows: {what}"))} else {Ok(())}
    }
    for recording in [false,true] {
        with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            let queue=if recording {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
            let d=Device::new(b);let fail=|what:&str|format!("held rows, recorder {}: {what}",if recording {"on"} else {"off"});
            let (full,_)=extend(&d,None,0,&[1.0,2.0,3.0,4.0])?;
            let prefix=d.range(&full,0,2)?;let suffix=d.range(&full,2,2)?;let nested=d.range(&suffix,1,1)?;drop(full);
            // The suffix reads words 2..4: an append behind the prefix must copy.
            let (first,copied)=extend(&d,Some(&prefix),2,&[9.0,9.5])?;if !copied {return Err(fail("an append wrote under a live suffix view"));}
            same(&d,&suffix,&[3.0,4.0],"the suffix changed")?;same(&d,&nested,&[4.0],"the view inside the suffix changed")?;same(&d,&first,&[1.0,2.0,9.0,9.5],"the copied append differs")?;
            // The view inside the suffix alone still reads word 3.
            drop(suffix);let (second,copied)=extend(&d,Some(&prefix),2,&[8.0,8.5])?;if !copied {return Err(fail("an append wrote under a live nested view"));}
            same(&d,&nested,&[4.0],"the nested view changed")?;same(&d,&second,&[1.0,2.0,8.0,8.5],"the second copied append differs")?;
            // Only the prefix lives: the append is in place and the prefix keeps its words.
            drop((nested,first,second));let (third,copied)=extend(&d,Some(&prefix),2,&[7.0,7.5])?;if copied {return Err(fail("a free append copied"));}
            same(&d,&third,&[1.0,2.0,7.0,7.5],"the in-place append differs")?;same(&d,&prefix,&[1.0,2.0],"the prefix changed")?;
            // A rollback prefix of three rows, the longer state gone: in place behind it.
            let rolled=d.range(&third,0,3)?;drop(third);let (fourth,copied)=extend(&d,Some(&rolled),3,&[6.0])?;if copied {return Err(fail("an append behind a rollback prefix copied"));}
            same(&d,&fourth,&[1.0,2.0,7.0,6.0],"the append behind the rollback prefix differs")?;
            // Two branches from one state: the first appends in place, the second must copy.
            let (one,copied)=extend(&d,Some(&fourth),4,&[5.0])?;if copied {return Err(fail("the first branch copied"));}
            let (two,copied)=extend(&d,Some(&fourth),4,&[4.5])?;if !copied {return Err(fail("the second branch wrote under the first"));}
            same(&d,&one,&[1.0,2.0,7.0,6.0,5.0],"the first branch changed")?;same(&d,&two,&[1.0,2.0,7.0,6.0,4.5],"the second branch differs")?;same(&d,&fourth,&[1.0,2.0,7.0,6.0],"the shared state changed")?;
            // Past the room the rows are copied; a value that does not start its storage is copied too.
            let (long,copied)=extend(&d,Some(&one),5,&[0.5,1.5,2.5,3.5])?;if !copied {return Err(fail("an append past the room did not copy"));}
            same(&d,&long,&[1.0,2.0,7.0,6.0,5.0,0.5,1.5,2.5,3.5],"the append past the room differs")?;
            let middle=d.range(&long,1,8)?;drop(long);let (moved,copied)=extend(&d,Some(&middle),8,&[4.0])?;if !copied {return Err(fail("a view that does not start its storage was appended in place"));}
            same(&d,&moved,&[2.0,7.0,6.0,5.0,0.5,1.5,2.5,3.5,4.0],"the append behind a middle view differs")?;
            drop((prefix,rolled,fourth,one,two,middle,moved));drop(queue);b.synchronize()?;Ok(())
        })?;
    }
    Ok(())
}
pub fn check(out:&Path)->Result<()> {

    fs::create_dir(out).map_err(|e|e.to_string())?;
    recorder_mailbox_preserves_mixed_capture_failure()?;recorder_budget_failure_is_permanent()?;recorder_chain_exceeds_graph_limit()?;standalone_program_rejects_active_recording()?;recorder_exchange_failure_rejects_readout()?;recorder_replay_after_poison_is_rejected()?;recorder_join_keeps_one_piece()?;recorder_norm_stays_in_the_piece()?;recorder_zero_fills_keep_their_order()?;pool_trim_releases_spares_down_to_the_room()?;held_rows_append_only_past_every_live_view()?;
    recorder_retains_small_graph_working_set()?;persistent_step::check(&out.join("two-rank"))?;
    let receipt="RESIDENT RECORDER PASS recorder_mailbox_preserves_mixed_capture_failure=true mailbox_take=true mailbox_put_finite=true recorder_budget_failure_is_permanent=true later_read=true later_consumer=true later_flush=true later_queue=true recorder_chain_exceeds_graph_limit=true standalone_program_rejects_active_recording=true recorder_exchange_failure_rejects_readout=true recorder_replay_after_poison_is_rejected=true recorder_join_keeps_one_piece=true recorder_norm_stays_in_the_piece=true recorder_zero_fills_keep_their_order=true pool_trim_releases_spares_down_to_the_room=true held_rows_append_only_past_every_live_view=true exact_bits=true\n";
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
