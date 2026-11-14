//! Grouped prompt twin and component timing against the screened scalar arithmetic.
use super::*;
fn program<'a>(b:&Backend<'a>,s:Rows,q:&DeviceVector<'_>,cache:&DeviceVector<'_>,selected:&DeviceVector<'_>,weight:&DeviceVector<'_>,timing:&mut [u128;5])->Result<Vec<DeviceVector<'a>>> {
    let t=Instant::now();let q=b.prompt_operation(s,[Some(q),None,None,None,None],Some(weight))?;timing[0]+=t.elapsed().as_nanos();
    let t=Instant::now();let scores=b.prompt_operation(Rows {op:Row::AbsorbScores,..s},[Some(&q),Some(cache),Some(selected),None,None],None)?;timing[1]+=t.elapsed().as_nanos();
    let t=Instant::now();let prob=b.prompt_operation(Rows {op:Row::AbsorbSoftmax,cols:s.first,..s},[Some(&scores),Some(selected),None,None,None],None)?;timing[2]+=t.elapsed().as_nanos();
    let t=Instant::now();let u=b.prompt_operation(Rows {op:Row::AbsorbValues,..s},[Some(&prob),Some(cache),Some(selected),None,None],None)?;timing[3]+=t.elapsed().as_nanos();
    let t=Instant::now();let out=b.prompt_operation(Rows {op:Row::AbsorbProject,..s},[Some(&u),Some(selected),None,None,None],Some(weight))?;timing[4]+=t.elapsed().as_nanos();
    Ok(vec![q,scores,prob,u,out])
}
/// GPU time of each grouped step, by events between the steps. The events disturb the pipeline: read the parts as shares.
fn parts<'a>(b:&Backend<'a>,s:Rows,q:&DeviceVector<'_>,cache:&DeviceVector<'_>,selected:&DeviceVector<'_>,weight:&DeviceVector<'_>,ns:&mut [u64;5])->Result<()> {
    let m0=b.gpu_mark()?;let q=b.prompt_operation(s,[Some(q),None,None,None,None],Some(weight))?;
    let m1=b.gpu_mark()?;let scores=b.prompt_operation(Rows {op:Row::AbsorbScores,..s},[Some(&q),Some(cache),Some(selected),None,None],None)?;
    let m2=b.gpu_mark()?;let prob=b.prompt_operation(Rows {op:Row::AbsorbSoftmax,cols:s.first,..s},[Some(&scores),Some(selected),None,None,None],None)?;
    let m3=b.gpu_mark()?;let u=b.prompt_operation(Rows {op:Row::AbsorbValues,..s},[Some(&prob),Some(cache),Some(selected),None,None],None)?;
    let m4=b.gpu_mark()?;let out=b.prompt_operation(Rows {op:Row::AbsorbProject,..s},[Some(&u),Some(selected),None,None,None],Some(weight))?;
    let m5=b.gpu_mark()?;b.synchronize()?;std::hint::black_box(out);
    for (at,(from,to)) in [(&m0,&m1),(&m1,&m2),(&m2,&m3),(&m3,&m4),(&m4,&m5)].into_iter().enumerate() {ns[at]+=from.ns_until(to)? as u64;}Ok(())
}
/// `spread` sets how the selected-row counts of the tokens differ; with `full` the last token takes every slot.
fn twin(b:&Backend<'_>,s:Rows,weight:&[f32],spread:usize,full:bool)->Result<()> {
    use spark_gpu_probe::prompt_ops::reference;
    let (m,h,d,k,c,n,r)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.cols as usize,s.first as usize,s.slots as usize,s.rows as usize);
    let q=data(m*h*d,31);let cache=data(r*k,37);let mut selected=vec![-1.0;m*(n+1)];
    for token in 0..m {let count=if token==0 {0} else if full && token+1==m {c} else {(token*spread+3)%(c+1)};
        for j in 0..count {selected[token*(n+1)+j]=((j*13+token)%r) as f32;}
        if count>1 {selected[token*(n+1)+1]=selected[token*(n+1)];}selected[token*(n+1)+n]=count as f32;
    }
    let a=reference(s,[&q,&[],&[],&[],&[]],weight)?;
    let score=reference(Rows {op:Row::AbsorbScores,..s},[&a,&cache,&selected,&[],&[]],&[])?;
    let prob=reference(Rows {op:Row::AbsorbSoftmax,cols:s.first,..s},[&score,&selected,&[],&[],&[]],&[])?;
    let u=reference(Rows {op:Row::AbsorbValues,..s},[&prob,&cache,&selected,&[],&[]],&[])?;
    let out=reference(Rows {op:Row::AbsorbProject,..s},[&u,&selected,&[],&[],&[]],weight)?;
    let (query_device,cache_device,selected_device,weight_device)=(b.upload_vector(&q)?.into_resident_shared()?,b.upload_vector(&cache)?,b.upload_vector(&selected)?,b.upload_vector(weight)?);
    let got=program(b,s,&query_device,&cache_device,&selected_device,&weight_device,&mut [0;5])?;
    let expected=[a,score,prob,u,out];let mut batch=Vec::new();
    for (plane,(want,got)) in expected.iter().zip(&got).enumerate() {
        let got=bits(b,got)?;let diff=want.iter().zip(&got).filter(|(a,b)|a.to_bits()!=**b).count();
        println!("PROMPT_TWIN tokens={m} heads={h} key={d} latent={k} slots={n} context={r} plane={plane} words={} differing={diff}",want.len());
        if diff!=0 {return Err("grouped prompt CPU/GPU twin differs".into());}batch.push(got);
    }
    for token in 0..m {
        let count=selected[token*(n+1)+n] as usize;let mut taken=vec![0.0;c*k];
        for j in 0..count {let id=selected[token*(n+1)+j] as usize;taken[j*k..(j+1)*k].copy_from_slice(&cache[id*k..(id+1)*k]);}
        let latent=b.upload_vector(&taken)?;let count=b.upload_vector(&[count as f32])?;let query=query_device.resident_range(token*h*d..(token+1)*h*d)?;
        let s=Shape {op:Op::AbsorbQuery,heads:s.heads,dim:s.dim,taps:s.taps,rows:c as u32,cols:s.cols,epsilon:0.0,bound:0.0};
        let q=b.causal_operation(s,[Some(&query),None,None,None],Some(&weight_device))?;
        let scores=b.causal_operation(Shape {op:Op::AbsorbScores,..s},[Some(&q),Some(&latent),Some(&count),None],None)?;
        let p=b.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..s},[Some(&scores),Some(&count),None,None],None)?;
        let u=b.causal_operation(Shape {op:Op::AbsorbValues,..s},[Some(&p),Some(&latent),Some(&count),None],None)?;
        let out=b.causal_operation(Shape {op:Op::AbsorbProject,..s},[Some(&u),None,None,None],Some(&weight_device))?;
        for (plane,v) in [q,scores,p,u,out].iter().enumerate() {let got=bits(b,v)?;
            if got.as_slice()!=&batch[plane][token*got.len()..(token+1)*got.len()] {return Err(format!("grouped/scalar GPU differs token={token} plane={plane}"));}
        }
    }
    println!("PROMPT_SCALAR_EQUAL tokens={m} heads={h} key={d} latent={k} slots={n} context={r} planes=5 differing=0");Ok(())
}
pub(super) fn run(b:&Backend<'_>,cases:&[Case<'_>],samples:usize)->Result<()> {
    for (m,h,d,v,k) in [(1usize,1usize,17usize,9usize,2usize),(3,3,17,9,10),(5,5,33,7,34),(17,4,64,32,512)] {
        let wide=data(h*(d+v)*k,43);let weight=wide.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect::<Vec<_>>();
        twin(b,Rows {op:Row::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:31,cols:k as u32,first:19,slots:23,..Rows::new(Row::AbsorbQuery,m).ok_or("prompt twin shape")?},&weight,7,false)?;
    }
    // The model's widths (key 256, value 256, latent 512, 2,051 slots) at the head counts of four, two and one rank.
    // 16 heads is a partial tile (a tile holds 32 heads); 64 heads is two tiles. Rows are picked out of order from a
    // context above the slot bound, and the last token takes every slot.
    for (m,h) in [(64usize,16usize),(32,32),(8,64)] {
        let (d,v,k)=(256usize,256usize,512usize);
        let wide=data(h*(d+v)*k,47);let weight=wide.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect::<Vec<_>>();
        twin(b,Rows {op:Row::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:4158,cols:k as u32,first:2051,slots:2051,..Rows::new(Row::AbsorbQuery,m).ok_or("prompt twin shape")?},&weight,331,true)?;
    }
    for m in [3usize,17] {let c=&cases[0];twin(b,Rows {heads:c.shape.heads,dim:c.shape.dim,taps:c.shape.taps,rows:31,cols:c.shape.cols,first:19,slots:23,..Rows::new(Row::AbsorbQuery,m).ok_or("prompt twin shape")?},&c.raw_weight,7,false)?;}
    for context in [15usize,127,1040,4158] {
        let m=context.min(256);let first=context-m;let mut times=Vec::new();let mut devices=Vec::new();
        for sample in 0..samples+1 {
            let mut calls=[0u128;5];let mut upload_ns=0u128;let mut outputs=Vec::new();
            let prepared=cases.iter().map(|case| {
                let cache=b.upload_vector(&data(context*case.shape.cols as usize,89))?;
                let q=b.upload_vector(&data(m*(case.shape.heads*case.shape.dim) as usize,97))?;
                Ok((cache,q))
            }).collect::<Result<Vec<_>>>()?;b.synchronize()?;
            let start=b.gpu_mark()?;let wall=Instant::now();
            for (case,(cache,q)) in cases.iter().zip(&prepared) {
                let n=case.shape.rows as usize;let c=n.min(context);let mut selected=vec![-1.0;m*(n+1)];
                for token in 0..m {let live=(first+token+1).min(n);for j in 0..live {selected[token*(n+1)+j]=j as f32;}selected[token*(n+1)+n]=live as f32;}
                let t=Instant::now();let selected=b.upload_vector(&selected)?;upload_ns+=t.elapsed().as_nanos();
                let s=Rows {heads:case.shape.heads,dim:case.shape.dim,taps:case.shape.taps,rows:context as u32,cols:case.shape.cols,first:c as u32,slots:n as u32,..Rows::new(Row::AbsorbQuery,m).ok_or("prompt grouped shape")?};
                outputs.push(program(b,s,q,cache,&selected,&case.weight,&mut calls)?);
            }
            let end=b.gpu_mark()?;let enqueue=wall.elapsed().as_nanos();b.synchronize()?;let total=wall.elapsed().as_nanos();let gpu=start.ns_until(&end)?;
            println!("PROMPT_GROUPED context={context} tokens={m} layers={} sample={sample} enqueue_ns={enqueue} wall_ns={total} gpu_ns={gpu} upload_ns={upload_ns} call_ns={calls:?} call_order=query,scores,softmax,values,project",cases.len());
            if sample>0 {times.push(total);devices.push(gpu);}std::hint::black_box(outputs);
        }
        println!("PROMPT_GROUPED_MEDIAN context={context} tokens={m} layers={} samples={samples} wall_ns={} gpu_ns={} full_prompt=false",cases.len(),median(&mut times),median(&mut devices));
        let mut ns=[0u64;5];
        for case in cases {
            let cache=b.upload_vector(&data(context*case.shape.cols as usize,89))?;let q=b.upload_vector(&data(m*(case.shape.heads*case.shape.dim) as usize,97))?;
            let n=case.shape.rows as usize;let c=n.min(context);let mut selected=vec![-1.0;m*(n+1)];
            for token in 0..m {let live=(first+token+1).min(n);for j in 0..live {selected[token*(n+1)+j]=j as f32;}selected[token*(n+1)+n]=live as f32;}
            let selected=b.upload_vector(&selected)?;b.synchronize()?;
            let s=Rows {heads:case.shape.heads,dim:case.shape.dim,taps:case.shape.taps,rows:context as u32,cols:case.shape.cols,first:c as u32,slots:n as u32,..Rows::new(Row::AbsorbQuery,m).ok_or("prompt grouped shape")?};
            parts(b,s,&q,&cache,&selected,&case.weight,&mut ns)?;
        }
        println!("PROMPT_PARTS context={context} tokens={m} layers={} query_ns={} scores_ns={} softmax_ns={} values_ns={} project_ns={} event_perturbed=true",cases.len(),ns[0],ns[1],ns[2],ns[3],ns[4]);
    }
    println!("PROMPT_GROUPED_KERNELS {}",spark_gpu_probe::prompt_launches::line());Ok(())
}
