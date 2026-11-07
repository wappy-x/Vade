#[cfg(target_os="linux")]
mod device {
    use spark_gpu_probe::{causal_ops::{Shape,Op},driver::{with_backend,Backend,DeviceVector},layer_ptx};
    type Result<T>=std::result::Result<T,String>;
    fn equal(label:&str,left:&[f32],right:&[f32])->Result<()> {
        if left.len()!=right.len() {return Err(format!("{label}: length {} != {}",left.len(),right.len()));}
        if let Some(index)=left.iter().zip(right).position(|(a,b)|a.to_bits()!=b.to_bits()) {return Err(format!("{label}: first differing word {index}: {:08x} != {:08x}",left[index].to_bits(),right[index].to_bits()));}Ok(())
    }
    fn old<'a>(backend:&Backend<'a>,shape:Shape,parts:&[&DeviceVector<'a>],query:&DeviceVector<'a>)->Result<[DeviceVector<'a>;3]> {
        let mut table=None;let width=shape.heads as usize*(shape.dim+shape.cols) as usize;
        let storage=backend.resident_join(&parts.iter().map(|part|(*part,part.len())).collect::<Vec<_>>(),&mut table,shape.rows as usize*width)?.into_resident_shared()?;
        let packed=storage.resident_range(0..shape.taps as usize*width)?;
        let unpack=backend.causal_operation(Shape {op:Op::UnpackKv,..shape},[if shape.taps==0 {None} else {Some(&packed)},None,None,None],None)?.into_resident_shared()?;
        let keys=shape.rows as usize*shape.heads as usize*shape.dim as usize;
        let key_view=unpack.resident_range(0..keys)?;let value_view=unpack.resident_range(keys..unpack.len())?;
        let count=backend.upload_vector(&[shape.taps as f32])?;
        let score_shape=Shape {op:Op::ScoresSlots,cols:0,taps:0,..shape};
        let scores=backend.causal_operation(score_shape,[Some(query),Some(&key_view),Some(&count),None],None)?;
        let probabilities=backend.causal_operation(Shape {op:Op::Softmax,..score_shape},[Some(&scores),Some(&count),None,None],None)?;
        let output=backend.causal_operation(Shape {op:Op::ValuesChannels,dim:shape.cols,..score_shape},[Some(&probabilities),Some(&value_view),Some(&count),None],None)?;
        Ok([scores,probabilities,output])
    }
    fn median(values:&mut[u128])->u128 {values.sort_unstable();values[values.len()/2]}
    pub fn run()->Result<()> {
        let mut changed=[1.0f32];changed[0]=f32::from_bits(changed[0].to_bits()^1);
        if equal("negative control",&[1.0],&changed).is_ok() {return Err("comparator accepted a changed bit".into());}
        with_backend(0,layer_ptx()?,usize::MAX,|backend| {
            backend.configure_vector_pool(1<<30);let queue=backend.resident_queue()?;
            let mut comparisons=0usize;let mut checked_bytes=0usize;
            let args=std::env::args().skip(1).collect::<Vec<_>>();let recorded=args.iter().any(|arg|arg=="--record");let padded=args.iter().any(|arg|arg=="--padded");let special=args.iter().any(|arg|arg=="--special");
            let args=args.iter().filter(|arg|!matches!(arg.as_str(),"--record"|"--padded"|"--special")).map(|arg|arg.parse::<u32>().map_err(|_|"invalid row bound".to_string())).collect::<Result<Vec<_>>>()?;
            if recorded {backend.enable_resident_launches()?;}
            println!("MODE recorded={recorded} special={special}");
            let first=args.first().copied().unwrap_or(1);let last=args.get(1).copied().unwrap_or(2051);
            if first==0 || first>last || last>2051 {return Err("row sweep must be inside 1..2051".into());}
            let extents=if special {[1,2,3,31,32,33,63,64,65,127,128,129,255,256,511,512,513,1023,1024,1025,1151,1152,1153,2047,2048,2049,2051].into_iter().filter(|extent|*extent>=first && *extent<=last).collect::<Vec<_>>()} else {(first..=last).collect()};
            for pattern in if special {vec![1,2,3]} else {vec![0]} {
            println!("PATTERN {pattern} name={}",super::patterns::name(pattern));
            let (heads,key_dim,value_dim)=(64u32,256u32,256u32);
            let raw=(0..2051usize*heads as usize*(key_dim+value_dim) as usize).map(|index|super::patterns::word(pattern,index,false)).collect::<Vec<_>>();
            let owner=backend.upload_vector(&raw)?.into_resident_shared()?;
            let memo=backend.upload_vector(&raw.iter().rev().copied().collect::<Vec<_>>())?.into_resident_shared()?;
            let query=backend.upload_vector(&(0..heads as usize*key_dim as usize).map(|index|super::patterns::word(pattern,index,true)).collect::<Vec<_>>())?;
            for padded in if special {vec![false,true]} else {vec![padded]} {for &extent in &extents {
                let rows=if padded {2051} else {extent};let live=extent;
                let width=heads as usize*(key_dim+value_dim) as usize;
                for reverse in [false,true] {
                    let shape=Shape {op:Op::ScoresIndirect,heads,dim:key_dim,cols:value_dim,rows,taps:live,epsilon:0.0,bound:0.0};
                    let mut parts=Vec::new();let mut first=0usize;
                    while first<live as usize {let count=(live as usize-first).min(if reverse {1+first%7} else {4}).min(if first<1024 {1024-first} else {usize::MAX});
                        let source=if first<1024 {&owner} else {&memo};parts.push(source.resident_range(first*width..(first+count)*width)?);first+=count;}
                    if reverse {parts.reverse();}
                    let input=parts.iter().collect::<Vec<_>>();
                    let baseline=old(backend,shape,&input,&query)?;let candidate=backend.resident_row_attention(shape,&input,&query)?;
                    for (plane,(left,right)) in baseline.iter().zip(&candidate).enumerate() {let left=backend.read_vector_values(left)?;let right=backend.read_vector_values(right)?;
                        equal(&format!("h{heads}/k{key_dim}/v{value_dim}/r{rows}/live{live}/plane{plane}"),&left,&right)?;comparisons+=1;checked_bytes+=left.len()*4;}
                    println!("BYTE PASS pattern={pattern} heads={heads} key={key_dim} value={value_dim} rows={rows} live={live} reverse={reverse} planes=3");
                }
                if !special && !padded && matches!(rows,1041|2051) {
                    let shape=Shape {op:Op::ScoresIndirect,heads,dim:key_dim,cols:value_dim,rows,taps:rows,epsilon:0.0,bound:0.0};
                    let mut parts=Vec::new();for first in (0..rows as usize).step_by(4) {let source=if first<1024 {&owner} else {&memo};parts.push(source.resident_range(first*width..(first+4).min(rows as usize)*width)?);}
                    let input=parts.iter().collect::<Vec<_>>();let mut baseline=Vec::new();let mut candidate=Vec::new();
                    for repeat in 0..7 {for candidate_first in [repeat%2==0,repeat%2!=0] {
                        backend.synchronize()?;let start=std::time::Instant::now();
                        let output=if candidate_first {backend.resident_row_attention(shape,&input,&query)?} else {old(backend,shape,&input,&query)?};
                        backend.synchronize()?;let elapsed=start.elapsed().as_nanos();drop(output);
                        if candidate_first {candidate.push(elapsed);} else {baseline.push(elapsed);}
                    }}
                    let before=median(&mut baseline);let after=median(&mut candidate);
                    println!("TIMING rows={rows} repeats=7 alternating=true baseline_ns={before} candidate_ns={after} speedup={:.6} eleven_layer_saved_ms={:.6} scope=wall_reserved_join_unpack_scores_softmax_values cache=warm_mixed_owners_no_flush",before as f64/after as f64,(before as f64-after as f64)*11.0/1e6);
                }
            }}
            let invalid=Shape {op:Op::ScoresIndirect,heads:64,dim:256,cols:256,rows:1,taps:1,epsilon:0.0,bound:0.0};
            let row=backend.upload_vector(&[1.0,2.0,3.0])?;
            if backend.resident_row_attention(invalid,&[&row],&query).is_ok() {return Err("ragged row owner accepted".into());}
            if backend.resident_row_attention(invalid,&[],&query).is_ok() {return Err("missing row owner accepted".into());}
            let fake=backend.upload_vector(&[0.0,0.0])?;
            if backend.causal_operation(invalid,[Some(&query),Some(&fake),None,None],None).is_ok() {return Err("forged address table accepted".into());}
            for op in [Op::ScoresTable,Op::ValuesTable] {
                if backend.causal_operation(Shape {op,..invalid},[Some(&query),Some(&fake),None,None],None).is_ok() {return Err("forged plane table accepted".into());}
            }
            let width=heads as usize*(key_dim+value_dim) as usize;let query_words=heads as usize*key_dim as usize;
            let shifted_row=backend.upload_vector(&[&[0.0][..],&raw[..width]].concat())?.into_resident_shared()?;
            let misaligned_row=shifted_row.resident_range(1..width+1)?;let aligned_row=owner.resident_range(0..width)?;
            let query_host=backend.read_vector_values(&query)?;
            let shifted_query=backend.upload_vector(&[&[0.0][..],query_host.as_slice()].concat())?.into_resident_shared()?;
            let misaligned_query=shifted_query.resident_range(1..query_words+1)?;
            for (part,operand) in [(&misaligned_row,&query),(&aligned_row,&misaligned_query)] {
                let before=backend.stats().attention_dispatches;
                let candidate=backend.resident_row_attention(invalid,&[part],operand)?;
                let baseline=old(backend,invalid,&[part],operand)?;
                for (left,right) in baseline.iter().zip(&candidate) {equal("unaligned scalar fallback",&backend.read_vector_values(left)?,&backend.read_vector_values(right)?)?;}
                if backend.stats().attention_dispatches!=before {return Err("unaligned operands reached the table kernel".into());}
            }
            let dispatch=backend.stats().attention_dispatches;
            if dispatch[0]==0 || dispatch[1]==0 {return Err("table kernels were not accepted".into());}
            println!("TABLE_DISPATCH scores={} values={} unaligned_fallback_cases=2 forged_plane_tables_rejected=true",dispatch[0],dispatch[1]);
            }
            drop(queue);backend.synchronize()?;
            println!("COMPLETE PASS comparisons={comparisons} bytes={checked_bytes} comparator_negative=PASS malformed_owners=PASS forged_table_rejected=PASS");Ok(())
        })
    }
}
#[cfg(any(target_os="linux",test))]
#[path="latent-rows/patterns.rs"] mod patterns;
fn main() {
    #[cfg(target_os="linux")] if let Err(error)=device::run() {eprintln!("FAIL {error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))] eprintln!("latent-rows requires a CUDA device on Linux");
}
