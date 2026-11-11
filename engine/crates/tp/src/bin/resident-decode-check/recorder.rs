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
        for _ in 0..GraphAllowance::FLASH_MAX_NODES*64+17 {current=b.round_bf16_lanes(&current)?.into_vector();}
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
        for (width,kind) in [(1usize,WeightType::F32),(33,WeightType::Bf16),(127,WeightType::Bf16),(128,WeightType::F32),(1536,WeightType::Bf16),(4097,WeightType::F32)] {
            let shape=NormShape::new(width,1e-6,kind)?;
            let weight=(0..width).map(|i|0.5+(i%13) as f32*0.125).collect::<Vec<_>>();
            let values=(0..width).map(|i|if width==33 {f32::from_bits([0x80000000,0x7f810000,0xffc10000,0x7f800000,1,0x3f808000,0xbf818000][i%7])} else if i%11==0 {-0.0} else {((i*37)%257) as f32*0.0625-8.0}).collect::<Vec<_>>();
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
                let address=first.vector().device_address();let owned=first.into_vector();
                if owned.device_address()!=address || b.read_vector_values(&owned)?.iter().map(|v|v.to_bits()).ne(bits[..width].iter().copied()) {return Err("norm ownership changed storage or bits".into());}
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
fn causal_chain_keeps_bits()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for heads in [1usize,3,17,64] {
            let dim=128usize;let width=heads*dim;
            let values=|words:usize,seed:usize|(0..words).map(|index|if index%17==0 {-0.0} else {((index*37+seed)%257) as f32/256.0-0.5}).collect::<Vec<_>>();
            let qkv=b.upload_vector(&values(3*width,1))?;let gate=b.upload_vector(&values(width,2))?;let beta=b.upload_vector(&values(heads,3))?;
            let prepare_weight=b.upload_vector(&values(heads+width,4))?;let norm_weight=b.upload_vector(&values(dim,5))?;
            let mut state=values(heads*dim*dim,6);
            for repeat in 0..3 {
                let prior=b.upload_vector(&state)?;b.synchronize()?;let mut seen:Option<Vec<Vec<u32>>>=None;
                for queued in [false,true] {
                    let queue=if queued {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
                    let shape=Shape {op:Op::Prepare,heads:heads as u32,dim:dim as u32,taps:0,rows:0,cols:0,epsilon:0.0,bound:-0.5};
                    let prepared=b.causal_operation(shape,[Some(&qkv),Some(&gate),Some(&beta),None],Some(&prepare_weight))?;
                    let step=b.causal_operation(Shape {op:Op::StepPlanar,..shape},[Some(&prepared),Some(&prior),None,None],None)?.into_resident_shared()?;
                    let vector=step.resident_range(heads*dim*dim..step.len())?;
                    let normalized=b.causal_operation(Shape {op:Op::GatedNorm,epsilon:1e-6,..shape},[Some(&vector),Some(&gate),None,None],Some(&norm_weight))?;
                    let mut bits=Vec::new();for output in [&prepared,&step,&normalized] {bits.push(b.read_vector_values(output)?.iter().map(|value|value.to_bits()).collect::<Vec<_>>());}
                    if let Some(first)=seen.as_ref() && first!=&bits {return Err(format!("causal-chain replay changed bits heads={heads} repeat={repeat}"));}
                    state=bits[1][..heads*dim*dim].iter().map(|&bits|f32::from_bits(bits)).collect();seen=Some(bits);drop(queue);
                }
            }
        }
        Ok(())
    })
}
/// A fused causal chain is ONE command for the ramp, when it is recorded and when a flush cuts the parts. After a wait the
/// parts hold 1 and then 4 commands. Here they are [norm] and [chain, norm, norm, norm]. The first pass captures both parts
/// at its flush. From the second pass on, each must leave when recording reaches its limit (2 early parts a pass, none
/// late), with the bits of the first pass. Before the count was the same in both places the second part never left early.
fn chain_counts_once_in_the_ramp()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let (heads,dim)=(3usize,128usize);let width=heads*dim;
        let values=|words:usize,seed:usize|(0..words).map(|index|((index*37+seed)%257) as f32/256.0-0.5).collect::<Vec<_>>();
        let qkv=b.upload_vector(&values(3*width,1))?;let gate=b.upload_vector(&values(width,2))?;let beta=b.upload_vector(&values(heads,3))?;
        let prepare_weight=b.upload_vector(&values(heads+width,4))?;let norm_weight=b.upload_vector(&values(dim,5))?;
        let prior=b.upload_vector(&values(heads*dim*dim,6))?;let plain=b.upload_vector(&values(width,7))?;
        let queue=b.resident_queue()?;b.enable_resident_launches()?;let mut seen:Option<Vec<Vec<u32>>>=None;
        for pass in 0..4 {
            b.synchronize()?;let before=b.resident_ramp_counts();
            let shape=Shape {op:Op::Prepare,heads:heads as u32,dim:dim as u32,taps:0,rows:0,cols:0,epsilon:0.0,bound:-0.5};
            let norm=|input:&_|b.causal_operation(Shape {op:Op::GatedNorm,epsilon:1e-6,..shape},[Some(input),Some(&gate),None,None],Some(&norm_weight));
            let first=norm(&plain)?;
            let prepared=b.causal_operation(shape,[Some(&qkv),Some(&gate),Some(&beta),None],Some(&prepare_weight))?;
            let step=b.causal_operation(Shape {op:Op::StepPlanar,..shape},[Some(&prepared),Some(&prior),None,None],None)?.into_resident_shared()?;
            let vector=step.resident_range(heads*dim*dim..step.len())?;
            let chained=norm(&vector)?;let second=norm(&first)?;let third=norm(&second)?;let fourth=norm(&third)?;
            let mut bits=Vec::new();for output in [&first,&step,&chained,&fourth] {bits.push(b.read_vector_values(output)?.iter().map(|value|value.to_bits()).collect::<Vec<_>>());}
            let ramp=b.resident_ramp_counts();let (early,late)=(ramp[0]-before[0],ramp[2]-before[2]);
            if let Some(first)=seen.as_ref() {
                if first!=&bits {return Err(format!("the ramp changed bits pass={pass}"));}
                if early!=2 || late!=0 {return Err(format!("a causal chain did not count once in the ramp: pass={pass} early_parts={early} late_parts={late}"));}
            }
            seen=Some(bits);
        }
        drop(queue);Ok(())
    })
}
fn pointwise_replay_keeps_bits()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let values=(0..65536u32).map(|bits|f32::from_bits(bits<<16)).chain((0..4097u32).map(|bits|f32::from_bits(bits.wrapping_mul(104729)))).collect::<Vec<_>>();
        let source=b.upload_vector(&values)?;b.synchronize()?;let mut seen=None;
        for queued in [false,true] {
            let queue=if queued {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
            let rounded=b.round_bf16_lanes(&source)?.into_vector().into_resident_shared()?;
            let mut copied=b.vector(rounded.len()+7)?;b.zero_vector(&mut copied)?;b.copy_range(&rounded,&mut copied,3,7,rounded.len()-3)?;
            let prefix=rounded.resident_range(0..127)?;let tail=rounded.resident_range(65535..rounded.len())?;
            let joined=b.resident_join(&[(&prefix,prefix.len()),(&tail,tail.len())],&mut None,0)?;
            let again=b.round_bf16_lanes(&copied)?.into_vector();
            let mut bits=Vec::new();for output in [&rounded,&copied,&joined,&again] {bits.push(b.read_vector_values(output)?.iter().map(|value|value.to_bits()).collect::<Vec<_>>());}
            if let Some(first)=seen.as_ref() && first!=&bits {return Err("pointwise copy/round replay changed raw bits".into());}
            seen=Some(bits);drop(queue);
        }
        Ok(())
    })
}
fn retained_joins_keep_bits()->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let words=[0,0x80000000,0x7f810000,0x7fc12345,0x3f800001,0xff800000,1,0xffffffff];
        let values=words.map(f32::from_bits);let owner=b.upload_vector(&values)?.into_resident_shared()?;
        let left=owner.resident_range(1..4)?;let right=owner.resident_range(4..8)?;
        let queue=b.resident_queue()?;b.enable_resident_launches()?;let before=b.resident_copy_counts();
        let joined=b.resident_join(&[(&left,3),(&right,4)],&mut None,0)?;
        if joined.device_address()!=left.device_address() {return Err("contiguous join copied storage".into());}
        drop((left,right,owner));
        if b.read_vector_values(&joined)?.iter().map(|value|value.to_bits()).ne(words[1..].iter().copied()) {return Err("retained join changed bits".into());}
        if b.resident_copy_counts().since(before).launches!=[0;3] {return Err("retained join launched a copy".into());}
        let left=joined.resident_range(0..3)?;let right=joined.resident_range(4..7)?;
        let disjoint=b.resident_join(&[(&right,3),(&left,3)],&mut None,0)?;
        let expected=words[5..].iter().chain(&words[1..4]);
        if b.read_vector_values(&disjoint)?.iter().map(|value|value.to_bits()).ne(expected.copied()) {return Err("noncontiguous join changed bits".into());}
        drop(queue);Ok(())
    })
}
/// A column-tile prompt group of few tokens launches `nv_cols` DIRECTLY, as the other prompt tile kernels are launched: its grid
/// follows the token count of the group, so a recorded launch cut a piece of its own (a graph capture) for every group size, and
/// the group table it uploaded was never read. With recording on, such a group records nothing, replays nothing and captures
/// nothing, and it writes the bits of the same call with recording off. Cases: the model's two expert shapes, 8 parts a token
/// (the picks of a token), 15 tokens and 2 tokens. The line also carries the time of a call that is followed by a stream wait.
fn column_groups_launch_directly(lines:&mut String)->Result<()> {
    use spark_gpu_probe::{driver::{Backend,DeviceVector},expert_ops::{self,Op as Expert,Shape as Group}};
    const CALLS:usize=126;const PICKS:usize=8;const SHAPES:[(u32,u32);2]=[(1024,4096),(2048,2048)];
    fn mixed(index:usize,salt:usize)->u32 {
        let mut value=(index as u32).wrapping_add((salt as u32+1).wrapping_mul(0x9e37_79b9));
        value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
    }
    fn words(bytes:Vec<u8>)->Vec<f32> {bytes.chunks(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()}
    fn part(rows:u32,cols:u32,part:usize)->Group {
        Group {op:Expert::NvRows,rows,cols,block_rows:expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:0.75+(part%PICKS) as f32/16.0,activation:0.125+(part%PICKS) as f32/32.0,bound:0.0}
    }
    // One part a (token, pick), each of one token: the list is not dense, so the group takes the column jobs.
    fn group<'a>(b:&Backend<'a>,(rows,cols):(u32,u32),tokens:usize,owners:&[(DeviceVector<'a>,DeviceVector<'a>)],input:&DeviceVector<'a>,lut:&DeviceVector<'a>)->Result<DeviceVector<'a>> {
        let parts=(0..tokens*PICKS).map(|at|(part(rows,cols,at),Some((&owners[at%PICKS].0,&owners[at%PICKS].1)),at)).collect::<Vec<_>>();
        b.expert_group(&parts,input,cols as usize/16*17,lut)
    }
    fn bits(values:&[f32])->Vec<u32> {values.iter().map(|value|value.to_bits()).collect()}
    with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<31,|b| {
        b.configure_vector_pool(1<<30);
        let lut=b.upload_vector(&expert_ops::decode_lut())?;let mut owners=Vec::new();let mut cases=Vec::new();
        for (rows,cols) in SHAPES {
            // Any byte is a pair of codes: the owners are read as column tiles and need no meaning beyond that.
            let (codes,scales)=(rows as usize*cols as usize/2,rows as usize*cols as usize/16);let mut set=Vec::new();
            for pick in 0..PICKS {set.push((b.upload_vector(&words((0..codes).map(|i|mixed(i,pick) as u8).collect()))?,
                b.upload_vector(&words((0..scales).map(|i|if i%17==0 {0} else {0x28+(mixed(i,pick+31)%64) as u8}).collect()))?));}
            owners.push(set);
            for tokens in [15usize,2] {
                let mut joined=Vec::new();
                for at in 0..tokens*PICKS {
                    let source=(0..cols as usize).map(|i|match (i+at)%13 {0=>0.0,1=>-0.0,_=>((mixed(i,at)%47) as i32-23) as f32/17.0}).collect::<Vec<_>>();
                    joined.extend(expert_ops::reference(Group {op:Expert::QuantizeRows,block_rows:1,..part(rows,cols,at)},[&source,&[],&[],&[]],&[],&[])?);
                }
                cases.push((owners.len()-1,tokens,b.upload_vector(&joined)?));
            }
        }
        // Recording off: the bits to keep.
        let mut plain=Vec::new();
        for (shape,tokens,input) in &cases {plain.push(bits(&b.read_vector_values(&group(b,SHAPES[*shape],*tokens,&owners[*shape],input,&lut)?)?));}
        b.synchronize()?;let queue=b.resident_queue()?;b.enable_resident_launches()?;let mut failure=None;
        for ((shape,tokens,input),plain) in cases.iter().zip(&plain) {
            let (rows,cols)=SHAPES[*shape];let before=b.stats();
            // The first call of a group size is where a recorded launch pays its capture.
            let started=std::time::Instant::now();let first=group(b,(rows,cols),*tokens,&owners[*shape],input,&lut)?;b.synchronize()?;let first_us=started.elapsed().as_secs_f64()*1e6;
            if bits(&b.read_vector_values(&first)?)!=*plain {return Err(format!("column group bits differ with recording on rows={rows} cols={cols} tokens={tokens}"));}
            drop(first);b.synchronize()?;let mut times=Vec::with_capacity(CALLS);let mut last=None;
            for _ in 0..CALLS {
                let started=std::time::Instant::now();last=Some(group(b,(rows,cols),*tokens,&owners[*shape],input,&lut)?);b.synchronize()?;
                times.push(started.elapsed().as_secs_f64()*1e6);
            }
            let d=b.stats().since(before);let recorded=d.recorded_kinds.iter().sum::<u128>();times.sort_by(f64::total_cmp);
            let counts=format!("recorded_commands={recorded} graph_replays={} graph_captures={}",d.graph_launches,d.graph_captures);
            let line=format!("RECORDER_COLUMN_GROUP rows={rows} cols={cols} tokens={tokens} jobs={} calls={} {counts} first_call_us={first_us:.1} \
                call_and_wait_us_median={:.1} call_and_wait_us_min={:.1}\n",tokens*PICKS,CALLS+1,times[CALLS/2],times[0]);
            print!("{line}");lines.push_str(&line);
            if bits(&b.read_vector_values(&last.ok_or("no column group ran")?)?)!=*plain {return Err(format!("column group bits differ after repeats rows={rows} cols={cols} tokens={tokens}"));}
            if (recorded!=0 || d.graph_launches!=0 || d.graph_captures!=0) && failure.is_none() {
                failure=Some(format!("a column group of few tokens entered the recorder rows={rows} cols={cols} tokens={tokens}: {counts}"));}
        }
        drop(queue);failure.map_or(Ok(()),Err)
    })
}
/// Distinct prompt tails must not leave standalone programs behind while recording is paused.
fn prompt_rows_do_not_retain_programs()->Result<()> {
    use spark_tp::bounded_decoder::{Decoder,Device,HeadRows};
    let checkpoint=super::select_known::attention::Checkpoint::new_with_capacity(engine_model::safetensors::Dtype::Bf16,true,64);
    if !checkpoint.admission.description().layers.iter().any(|layer|matches!(layer.ffn,engine_model::shapes::Ffn::Dense {..})) {
        return Err("prompt program fixture needs a dense SwiGlu layer".into());
    }
    let mut expected=Vec::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for recorded in [false,true] {
            let device=Device::new(b);let decoder=Decoder::new(&checkpoint.admission,&device,64)?;
            // Preparation enables recording on an existing queue. Prepare first so the control stays off.
            let queue=b.resident_queue()?;if recorded {b.enable_resident_launches()?;}
            for (at,length) in [17usize,31].into_iter().enumerate() {
                let tokens=(0..length).map(|i|1+(i as u32*5)%15).collect::<Vec<_>>();
                let state=decoder.zero_state()?;let (logits,_)=decoder.prefill_rows(&tokens,&state,HeadRows::Last)?;b.synchronize()?;
                let count=device.resident_program_count();let bits=logits.into_iter().flatten().map(f32::to_bits).collect::<Vec<_>>();
                println!("PROMPT PROGRAMS recorded={recorded} tail={length} programs={count}");
                if recorded {
                    if count!=0 {return Err(format!("prompt tail {length} retained {count} standalone programs"));}
                    if bits!=expected[at] {return Err("prompt program exclusion changed logit bits".into());}
                } else {
                    if count==0 {return Err("prompt program positive control captured no program".into());}
                    expected.push(bits);
                }
            }
            drop(queue);
        }Ok(())
    })
}
pub fn rider(kind:&str,out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    match kind {"ownership"=>{retained_joins_keep_bits()?;recorder_norm_stays_in_the_piece()?;},
        "pointwise"=>pointwise_replay_keeps_bits()?,"norm-tail"=>recorder_norm_stays_in_the_piece()?,
        "prompt-programs"=>prompt_rows_do_not_retain_programs()?,
        "causal-chain"=>{causal_chain_keeps_bits()?;chain_counts_once_in_the_ramp()?;},_=>return Err("unknown resident rider fixture".into())}
    let receipt=format!("RESIDENT RIDER PASS kind={kind} eager_vs_recorded=true exact_bits=true\n");
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
pub fn check(out:&Path)->Result<()> {
    pointwise_replay_keeps_bits()?;
    retained_joins_keep_bits()?;

    fs::create_dir(out).map_err(|e|e.to_string())?;
    recorder_mailbox_preserves_mixed_capture_failure()?;recorder_budget_failure_is_permanent()?;recorder_chain_exceeds_graph_limit()?;standalone_program_rejects_active_recording()?;recorder_exchange_failure_rejects_readout()?;recorder_replay_after_poison_is_rejected()?;recorder_join_keeps_one_piece()?;recorder_norm_stays_in_the_piece()?;recorder_zero_fills_keep_their_order()?;pool_trim_releases_spares_down_to_the_room()?;held_rows_append_only_past_every_live_view()?;
    recorder_retains_small_graph_working_set()?;prompt_rows_do_not_retain_programs()?;persistent_step::check(&out.join("two-rank"))?;
    let mut column_lines=String::new();let columns=column_groups_launch_directly(&mut column_lines);super::write(&out.join("COLUMN-GROUPS"),column_lines.as_bytes())?;columns?;
    let receipt="RESIDENT RECORDER PASS recorder_mailbox_preserves_mixed_capture_failure=true mailbox_take=true mailbox_put_finite=true recorder_budget_failure_is_permanent=true later_read=true later_consumer=true later_flush=true later_queue=true recorder_chain_exceeds_graph_limit=true standalone_program_rejects_active_recording=true recorder_exchange_failure_rejects_readout=true recorder_replay_after_poison_is_rejected=true recorder_join_keeps_one_piece=true recorder_norm_stays_in_the_piece=true recorder_zero_fills_keep_their_order=true pool_trim_releases_spares_down_to_the_room=true held_rows_append_only_past_every_live_view=true column_groups_launch_directly=true exact_bits=true\n";
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
