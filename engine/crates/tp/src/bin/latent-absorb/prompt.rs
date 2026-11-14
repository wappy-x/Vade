//! Prompt-only component timing. The scalar arm follows the screened decoder calls.
use super::*;
pub(super) fn run(b:&Backend<'_>,cases:&[Case<'_>],samples:usize)->Result<()> {
    for context in [15usize,127,1040,4158] {
        let m=context.min(256);let first=context-m;
        let mut times=Vec::new();let mut devices=Vec::new();
        for sample in 0..samples+1 {
            let mut elapsed=[0u128;8];let mut outputs=Vec::new();
            let prepared=cases.iter().map(|case| {
                let latent=b.upload_vector(&data(context*case.shape.cols as usize,89))?.into_resident_shared()?;
                let query=b.upload_vector(&data(m*(case.shape.heads*case.shape.dim) as usize,97))?.into_resident_shared()?;
                Ok((latent,query))
            }).collect::<Result<Vec<_>>>()?;b.synchronize()?;
            let start=b.gpu_mark()?;let wall=Instant::now();
            for (case,(latent,queries)) in cases.iter().zip(&prepared) {for token in 0..m {
                let live=(first+token+1).min(case.shape.rows as usize);
                let t=Instant::now();let ids=b.upload_vector(&(0..live).map(|j|j as f32).collect::<Vec<_>>())?;elapsed[0]+=t.elapsed().as_nanos();
                let mut gather=Rows::new(Row::PermuteRows,1).ok_or("prompt gather shape")?;gather.rows=live as u32;gather.cols=case.shape.cols;gather.first=context as u32;
                let t=Instant::now();let selected=b.prompt_operation(gather,[Some(latent),Some(&ids),None,None,None],None)?;elapsed[1]+=t.elapsed().as_nanos();
                let width=(case.shape.heads*case.shape.dim) as usize;let query=queries.resident_range(token*width..(token+1)*width)?;
                let t=Instant::now();let count=b.upload_vector(&[live as f32])?;elapsed[2]+=t.elapsed().as_nanos();
                let s=Shape {rows:live as u32,..case.shape};
                let t=Instant::now();let q=b.causal_operation(s,[Some(&query),None,None,None],Some(&case.weight))?;elapsed[3]+=t.elapsed().as_nanos();
                let t=Instant::now();let scores=b.causal_operation(Shape {op:Op::AbsorbScores,..s},[Some(&q),Some(&selected),Some(&count),None],None)?;elapsed[4]+=t.elapsed().as_nanos();
                let t=Instant::now();let probability=b.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..s},[Some(&scores),Some(&count),None,None],None)?;elapsed[5]+=t.elapsed().as_nanos();
                let t=Instant::now();let u=b.causal_operation(Shape {op:Op::AbsorbValues,..s},[Some(&probability),Some(&selected),Some(&count),None],None)?;elapsed[6]+=t.elapsed().as_nanos();
                let t=Instant::now();let output=b.causal_operation(Shape {op:Op::AbsorbProject,..s},[Some(&u),None,None,None],Some(&case.weight))?;elapsed[7]+=t.elapsed().as_nanos();
                outputs.push(output);
            }}
            let end=b.gpu_mark()?;let enqueue=wall.elapsed().as_nanos();b.synchronize()?;let total=wall.elapsed().as_nanos();let gpu=start.ns_until(&end)?;
            println!("PROMPT_SCALAR context={context} tokens={m} layers={} sample={sample} enqueue_ns={enqueue} wall_ns={total} gpu_ns={gpu} call_ns={elapsed:?} call_order=ids,gather,count,query,scores,softmax,values,project",cases.len());
            if sample>0 {times.push(total);devices.push(gpu);}std::hint::black_box(outputs);
        }
        println!("PROMPT_SCALAR_MEDIAN context={context} tokens={m} layers={} samples={samples} wall_ns={} gpu_ns={} full_prompt=false",cases.len(),median(&mut times),median(&mut devices));
    }Ok(())
}
