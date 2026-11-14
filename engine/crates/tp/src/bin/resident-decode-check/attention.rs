//! Baseline phase timing with checkpoint-derived attention geometry.
use super::*;
pub fn measure(root:&Path,out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,causal_ops::{Shape,Op,reference}};
    use engine_model::plan::description::TokenMixer;
    fs::create_dir(out).map_err(|e|e.to_string())?;let checkpoint=open_checkpoint(root)?;
    let d=checkpoint.description();let attention=d.layers.iter().filter_map(|l|match &l.mixer {TokenMixer::LatentAttention(a)=>Some(a),_=>None}).collect::<Vec<_>>();
    let a=attention.first().ok_or("checkpoint has no latent attention")?;let layers=attention.len();
    let mut report=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|b| {
        let _queue=b.resident_queue()?;
        for rows in [16u32,47,78] {for op in [Op::PoolScores,Op::Scores,Op::ScoresSlots,Op::Softmax,Op::Values,Op::ValuesChannels] {
            let index=op==Op::PoolScores;let heads=u32::try_from(if index {a.indexer.heads} else {a.heads}).map_err(|_|"head count exceeds ABI")?;
            let dim=u32::try_from(if index {a.indexer.dim} else if matches!(op,Op::Values|Op::ValuesChannels) {a.value_dim} else {a.key_dim}).map_err(|_|"attention dimension exceeds ABI")?;
            let shape=Shape {op,heads,dim,taps:0,rows:if index {rows/a.indexer.pool as u32} else {rows},cols:0,epsilon:0.0,bound:0.0};
            let (lengths,_,_,_)=shape.lengths().ok_or("attention timing shape differs")?;
            let mut data=lengths.map(|n|(0..n).map(|i|(i as i32%37-18) as f32/64.0).collect::<Vec<_>>());
            if matches!(op,Op::Scores|Op::ScoresSlots|Op::Values|Op::ValuesChannels) {data[2][0]=rows as f32;} else if op==Op::Softmax {data[1][0]=rows as f32;}
            if matches!(op,Op::Values|Op::ValuesChannels) {for v in &mut data[0] {*v=1.0/rows as f32;}}
            let original=Shape {op:match op {Op::ScoresSlots=>Op::Scores,Op::ValuesChannels=>Op::Values,_=>op},..shape};
            let expected=reference(original,data.each_ref().map(Vec::as_slice),&[])?;
            let inputs=data.iter().map(|v|if v.is_empty() {Ok(None)} else {b.upload_vector(v).map(Some)}).collect::<Result<Vec<_>>>()?;
            let args=std::array::from_fn(|i|inputs[i].as_ref());
            for _ in 0..3 {let result=b.causal_operation(shape,args,None)?;b.synchronize()?;std::hint::black_box(result);}
            b.set_profiling(true);b.synchronize()?;let before=b.stats();let started=std::time::Instant::now();let mut last=None;
            for _ in 0..16 {last=Some(b.causal_operation(shape,args,None)?);}
            b.synchronize()?;let wall=started.elapsed().as_secs_f64();let stats=b.stats().since(before);b.set_profiling(false);
            let got=b.read_vector_values(&last.ok_or("attention probe has no output")?)?;
            if got.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err(format!("baseline attention differs from original reference {op:?}"));}
            let gpu=stats.device_kernel_ns.iter().sum::<u128>() as f64/16.0/1e6;
            report+=&format!("RESIDENT ATTENTION BASELINE op={op:?} context={rows} heads={heads} dim={dim} layers={layers} repeats=16 gpu_ms_per_layer={gpu:.9} gpu_ms_per_token={:.9} wall_ms_per_layer={:.9} exact_bits=true\n",gpu*layers as f64,wall/16.0*1e3);
        }}Ok(())
    })?;
    write(&out.join("RECEIPT"),report.as_bytes())?;print!("{report}");Ok(())
}
