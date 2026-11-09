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
        let packed=if shape.taps==0 {None} else {Some(storage.resident_range(0..shape.taps as usize*width)?)};
        let unpack=backend.causal_operation(Shape {op:Op::UnpackKv,..shape},[packed.as_ref(),None,None,None],None)?.into_resident_shared()?;
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
    fn head_split_prompt(backend:&Backend<'_>)->Result<usize> {
        use spark_gpu_probe::prompt_ops::{Shape as Rows,Op as Row};
        let (heads,dim,value,latent)=(64usize,256usize,256usize,16usize);let width=heads*(dim+value);
        let weights=(0..width*latent).map(|index|(index as i32%31-15) as f32/64.0).collect::<Vec<_>>();
        let weight=backend.upload_vector(&weights)?;let mut comparisons=0;
        for tokens in [1usize,3,16] {
            let raw=(0..tokens*latent).map(|index|(index as i32%23-11) as f32/32.0).collect::<Vec<_>>();let input=backend.upload_vector(&raw)?;
            let mut shape=Rows::new(Row::ProjectRows,tokens).ok_or("projection fixture shape")?;shape.rows=width as u32;shape.cols=latent as u32;
            let full=backend.prompt_operation(shape,[Some(&input),None,None,None,None],Some(&weight))?;
            let full=backend.round_bf16_lanes(&full)?.into_vector();let full_raw=backend.read_vector_values(&full)?;
            for rank in 0..2 {
                let local_weight=backend.upload_vector(&weights[rank*width/2*latent..(rank+1)*width/2*latent])?;
                let local=backend.prompt_operation(Rows {rows:width as u32/2,..shape},[Some(&input),None,None,None,None],Some(&local_weight))?;
                let local=backend.round_bf16_lanes(&local)?.into_vector();
                equal("owned projection rows",&full_raw[rank*width/2*tokens..(rank+1)*width/2*tokens],&backend.read_vector_values(&local)?)?;comparisons+=1;
            }
        }
        for tokens in [1usize,3,16] {for selected_rows in [16usize,1041,2051] {
            let raw=(0..selected_rows*width).map(|index|super::patterns::word(0,index,false)).collect::<Vec<_>>();
            let query=(0..tokens*heads*dim).map(|index|super::patterns::word(0,index,true)).collect::<Vec<_>>();
            let selected=(0..tokens).flat_map(|token|(0..selected_rows).map(move |slot|((slot+token)%selected_rows) as f32).chain(std::iter::once(selected_rows as f32))).collect::<Vec<_>>();
            let owner=backend.upload_vector(&raw)?;let queries=backend.upload_vector(&query)?;let selection=backend.upload_vector(&selected)?;
            let mut shape=Rows::new(Row::ScoresRows,tokens).ok_or("prompt fixture shape")?;
            shape.heads=heads as u32;shape.dim=dim as u32;shape.taps=value as u32;shape.rows=selected_rows as u32;shape.cols=selected_rows as u32;shape.slots=selected_rows as u32;
            let evaluate=|shape:Rows,owner:&DeviceVector<'_>,queries:&DeviceVector<'_>|->Result<Vec<Vec<f32>>> {
                let scores=backend.prompt_operation(shape,[Some(queries),Some(owner),Some(&selection),None,None],None)?;
                let probabilities=backend.prompt_operation(Rows {op:Row::SoftmaxRows,..shape},[Some(&scores),Some(&selection),None,None,None],None)?;
                let attended=backend.prompt_operation(Rows {op:Row::ValuesRows,..shape},[Some(&probabilities),Some(owner),Some(&selection),None,None],None)?;
                [&scores,&probabilities,&attended].into_iter().map(|plane|backend.read_vector_values(plane)).collect()
            };
            let full=evaluate(shape,&owner,&queries)?;
            for rank in 0..2 {
                let local_raw=raw.chunks_exact(width).flat_map(|row|row[rank*width/2..(rank+1)*width/2].iter().copied()).collect::<Vec<_>>();
                let local_query=query.chunks_exact(heads*dim).flat_map(|row|row[rank*heads/2*dim..(rank+1)*heads/2*dim].iter().copied()).collect::<Vec<_>>();
                let local_owner=backend.upload_vector(&local_raw)?;let local_queries=backend.upload_vector(&local_query)?;
                let local=evaluate(Rows {heads:heads as u32/2,..shape},&local_owner,&local_queries)?;
                for plane in 0..3 {
                    let row=full[plane].len()/tokens;
                    let expected=full[plane].chunks_exact(row).flat_map(|words|words[rank*row/2..(rank+1)*row/2].iter().copied()).collect::<Vec<_>>();
                    equal("prompt owned head plane",&expected,&local[plane])?;comparisons+=1;
                }
            }
            println!("HEAD_SPLIT_PROMPT tokens={tokens} rows={selected_rows} differing_words=0");
        }}
        Ok(comparisons)
    }
    fn head_split_proof()->Result<()> {
        with_backend(0,layer_ptx()?,usize::MAX,|backend| {
            backend.configure_vector_pool(1<<30);let queue=backend.resident_queue()?;backend.enable_resident_launches()?;
            let mut comparisons=head_split_prompt(backend)?;
            for pattern in [0,1,2,3] {
                let width=64*512;let local_width=width/2;
                let raw=(0..2051*width).map(|index|super::patterns::word(pattern,index,false)).collect::<Vec<_>>();
                let query_raw=(0..64*256).map(|index|super::patterns::word(pattern,index,true)).collect::<Vec<_>>();
                let owner=backend.upload_vector(&raw)?.into_resident_shared()?;
                let memo=backend.upload_vector(&raw)?.into_resident_shared()?;
                let query=backend.upload_vector(&query_raw)?;
                let owners=(0..2).map(|rank|backend.upload_vector(&raw.chunks_exact(width).flat_map(|row|row[rank*local_width..(rank+1)*local_width].iter().copied()).collect::<Vec<_>>())?.into_resident_shared()).collect::<Result<Vec<_>>>()?;
                let memos=(0..2).map(|rank|backend.upload_vector(&raw.chunks_exact(width).flat_map(|row|row[rank*local_width..(rank+1)*local_width].iter().copied()).collect::<Vec<_>>())?.into_resident_shared()).collect::<Result<Vec<_>>>()?;
                for (rows,live) in [(16,0),(16,1),(16,16),(1041,1041),(1151,1151),(1152,1152),(1153,1153),(2051,1041),(2051,2049),(2051,2051)] {
                    for repeated in [false,true] {
                        let selected=(0..live as usize).map(|slot|if repeated {(slot/2+1151)%2051} else {slot}).collect::<Vec<_>>();
                        let shape=Shape {op:Op::ScoresIndirect,heads:64,dim:256,cols:256,rows,taps:live,epsilon:0.0,bound:0.0};
                        let parts=selected.iter().enumerate().map(|(slot,&row)|if slot%3==0 {&memo} else {&owner}.resident_range(row*width..(row+1)*width)).collect::<Result<Vec<_>>>()?;
                        let reference=old(backend,shape,&parts.iter().collect::<Vec<_>>(),&query)?;
                        let mut attended=Vec::new();
                        for rank in 0..2 {
                            let local_parts=selected.iter().enumerate().map(|(slot,&row)|if slot%3==0 {&memos[rank]} else {&owners[rank]}.resident_range(row*local_width..(row+1)*local_width)).collect::<Result<Vec<_>>>()?;
                            let local_query=backend.upload_vector(&query_raw[rank*32*256..(rank+1)*32*256])?;
                            let local_shape=Shape {heads:32,..shape};
                            let output=if spark_gpu_probe::causal_ops::indirect_attention_proven(local_shape) {
                                backend.resident_row_attention(local_shape,&local_parts.iter().collect::<Vec<_>>(),&local_query)?
                            } else {old(backend,local_shape,&local_parts.iter().collect::<Vec<_>>(),&local_query)?};
                            for plane in 0..3 {let full=backend.read_vector_values(&reference[plane])?;let words=full.len()/2;
                                equal(&format!("split pattern={pattern} rows={rows} live={live} repeated={repeated} rank={rank} plane={plane}"),&full[rank*words..(rank+1)*words],&backend.read_vector_values(&output[plane])?)?;comparisons+=1;}
                            attended.push(output.into_iter().nth(2).ok_or("attended plane is absent")?);
                        }
                        let joined=backend.resident_join(&attended.iter().map(|value|(value,value.len())).collect::<Vec<_>>(),&mut None,0)?;
                        equal("gathered head proof",&backend.read_vector_values(&reference[2])?,&backend.read_vector_values(&joined)?)?;comparisons+=1;
                        println!("HEAD_SPLIT_PROOF pattern={pattern} rows={rows} live={live} repeated={repeated} differing_words=0");
                    }
                }
            }
            drop(queue);backend.synchronize()?;
            println!("HEAD_SPLIT_PROOF COMPLETE comparisons={comparisons} differing_words=0 coverage=attention_planes_packed_owner_views_mixed_owners_empty_tail_repeated_limit_crossings scope=synthetic_not_model_gate");Ok(())
        })
    }
    fn head_split_premise()->Result<()> {
        with_backend(0,layer_ptx()?,usize::MAX,|backend| {
            backend.configure_vector_pool(1<<30);let queue=backend.resident_queue()?;
            if std::env::args().any(|arg|arg=="--record") {backend.enable_resident_launches()?;}
            for (cell,rows) in [(1040,1041u32),(4158,2051u32)] {
                let shape=Shape {op:Op::ScoresIndirect,heads:64,dim:256,cols:256,rows,taps:rows,epsilon:0.0,bound:0.0};
                let width=shape.heads as usize*(shape.dim+shape.cols) as usize;
                let raw=(0..rows as usize*width).map(|index|super::patterns::word(0,index,false)).collect::<Vec<_>>();
                let query_raw=(0..shape.heads as usize*shape.dim as usize).map(|index|super::patterns::word(0,index,true)).collect::<Vec<_>>();
                let owner=backend.upload_vector(&raw)?;let query=backend.upload_vector(&query_raw)?;
                let reference=backend.resident_row_attention(shape,&[&owner],&query)?;
                let fixture=backend.fixture_head_attention(shape,&[&owner],&query)?;
                for plane in 0..3 {equal("fixture versus production",&backend.read_vector_values(&reference[plane])?,&backend.read_vector_values(&fixture[plane])?)?;}
                let mut owners=Vec::new();let mut queries=Vec::new();let mut outputs=Vec::new();
                let local=Shape {heads:shape.heads/2,..shape};let local_width=width/2;
                for rank in 0..2usize {
                    let packed=raw.chunks_exact(width).flat_map(|row|row[rank*local_width..(rank+1)*local_width].iter().copied()).collect::<Vec<_>>();
                    owners.push(backend.upload_vector(&packed)?);
                    let query_words=query_raw.len()/2;queries.push(backend.upload_vector(&query_raw[rank*query_words..(rank+1)*query_words])?);
                    let output=backend.fixture_head_attention(local,&[&owners[rank]],&queries[rank])?;
                    for plane in 0..3 {
                        let full=backend.read_vector_values(&reference[plane])?;let words=full.len()/2;
                        equal("local head plane",&full[rank*words..(rank+1)*words],&backend.read_vector_values(&output[plane])?)?;
                    }
                    outputs.push(output);
                }
                println!("PREMISE_BYTES cell=c{cell} rows={rows} full_heads={} local_heads={} full_kv_bytes={} local_kv_bytes={} source=executed_shape_times_FP32_plane_width hardware_counter=false byte_diffs=0",shape.heads,local.heads,raw.len()*4,raw.len()*2);
                if std::env::args().any(|arg|arg=="--head-split-profile") {continue;}
                let mut full_times=Vec::new();let mut local_times=Vec::new();let mut full_gpu=Vec::new();let mut local_gpu=Vec::new();
                for repeat in 0..12 {for split in [repeat%2==0,repeat%2!=0] {
                    backend.synchronize()?;let timer=std::time::Instant::now();let start=backend.gpu_mark()?;
                    let rank=repeat%2;
                    let output=if split {backend.fixture_head_attention(local,&[&owners[rank]],&queries[rank])?} else {backend.resident_row_attention(shape,&[&owner],&query)?};
                    let end=backend.gpu_mark()?;backend.synchronize()?;let wall=timer.elapsed().as_nanos();let gpu=start.ns_until(&end)?;drop(output);
                    if repeat>=2 {if split {local_times.push(wall);local_gpu.push(gpu);} else {full_times.push(wall);full_gpu.push(gpu);}}
                }}
                let mut table=None;let mut gather_times=Vec::new();let mut gather_gpu=Vec::new();
                for repeat in 0..12 {
                    backend.synchronize()?;let timer=std::time::Instant::now();let start=backend.gpu_mark()?;
                    let gathered=backend.resident_join(&[(&outputs[0][2],outputs[0][2].len()),(&outputs[1][2],outputs[1][2].len())],&mut table,reference[2].len())?;
                    let end=backend.gpu_mark()?;backend.synchronize()?;let wall=timer.elapsed().as_nanos();let gpu=start.ns_until(&end)?;
                    equal("gathered attended",&backend.read_vector_values(&reference[2])?,&backend.read_vector_values(&gathered)?)?;
                    if repeat>=2 {gather_times.push(wall);gather_gpu.push(gpu);}
                }
                let full=median(&mut full_times);let split=median(&mut local_times);let gather=median(&mut gather_times);
                let full_event=median(&mut full_gpu);let split_event=median(&mut local_gpu);let gather_event=median(&mut gather_gpu);
                println!("PREMISE_TIMING cell=c{cell} samples=10 warmup=2 full_wall_ns={full} local_wall_ns={split} gather_local_wall_ns={gather} full_event_ns={full_event} local_event_ns={split_event} gather_local_event_ns={gather_event} eleven_layer_wall_saving_ms={:.6} eleven_layer_event_saving_ms={:.6} transport_measured=false",(full as f64-split as f64-gather as f64)*11.0/1e6,(full_event as f64-split_event as f64-gather_event as f64)*11.0/1e6);
            }
            drop(queue);backend.synchronize()?;println!("PREMISE COMPLETE byte_diffs=0 scope=kernel_bench_not_model_gate");Ok(())
        })
    }
    pub fn run()->Result<()> {
        if std::env::args().any(|arg|arg=="--head-split-proof") {return head_split_proof();}
        if std::env::args().any(|arg|arg=="--head-split-premise") {return head_split_premise();}
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
