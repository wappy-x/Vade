use std::{fs,path::Path,time::Instant};
use spark_gpu_probe::{device_route,driver::{Backend,DeviceVector,with_backend},expert_ops::{Op,Shape}};
use spark_tp::Result;
fn bytes(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect()}
fn raw<'d>(backend:&Backend<'d>,values:&[u8])->Result<DeviceVector<'d>> {
    let mut padded=values.to_vec();padded.resize(values.len().div_ceil(4)*4,0);
    let mut value=backend.vector(padded.len()/4)?;backend.write_vector_bytes(&mut value,&padded)?;value.into_resident_shared()
}
fn input(cols:usize,picks:usize,down:bool,rank:usize)->Vec<f32> {
    let patterns=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x80800000,0x3f808000,0x3f818000,0xbf808000,0x3f800000,0xbf800000,0x3e800000];
    (0..cols*if down {picks} else {1}).map(|at|if at%3==0 {f32::from_bits(patterns[(at+rank)%patterns.len()])} else {(at as i32%31-15) as f32/16.0}).collect()
}
fn patterns(experts:usize)->Vec<Vec<usize>> {
    let mut patterns=vec![(0..8).collect(),(0..8).rev().collect(),(experts-8..experts).collect(),vec![0,7,1,6,2,5,3,4]];
    if experts>=16 {patterns.push((0..8).map(|slot|slot*2).collect());patterns.push((0..8).map(|slot|slot*2+1).collect());}patterns
}
pub fn run(out:&Path,full:bool)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let started=Instant::now();let mut receipt=String::new();let mut compared=0u64;
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<29,|backend| {
        let mut shapes=if full {device_route::sweep()} else {vec![(1,512,1),(33,1024,2),(1024,4096,2),(2048,2048,1)]}.into_iter().map(|(rows,cols,phases)|(rows,cols,phases,16usize)).collect::<Vec<_>>();
        for experts in [8,9,255,256,257,1024] {for phases in 1..=2 {shapes.push((1,512,phases,experts));}}
        for rank in 0..2 {for &(rows,cols,phases,experts) in &shapes {for calibrated in [false,true] {for down in [false,true].into_iter().filter(|&down|!down || phases==1) {
            let picks=8usize;
            let mut owners=Vec::new();let mut shapes=Vec::new();
            // The same matrices again as column tiles when the storage rule admits the shape: the route then projects with
            // `nv_cols` and must give the bytes of the row-major reference.
            let tiled=spark_gpu_probe::expert_ops::nv_tile_layout_admits(rows as usize,cols as usize);let mut tiled_owners=Vec::new();
            for phase in 0..phases as usize {for expert in 0..experts {
                let shape=Shape {op:Op::NvProject,rows,cols,block_rows:0,block_cols:16,tokens:1,
                    global:if expert%2==0 {0.5} else {1.0},activation:if calibrated {0.125*(1+(expert+phase)%7) as f32} else {0.25},bound:0.0,
                    // A down table holds SECOND matrices (quarter chains), as the resident weights build it. The route then runs
                    // the quarter-chain decode kernel, and the device's descriptors must carry the role (review finding NEW-1).
                    role:if down {spark_gpu_probe::expert_ops::NV_SECOND} else {spark_gpu_probe::expert_ops::NV_FIRST}};
                let weight=(0..rows as usize*cols as usize/2).map(|at|(at+expert*19+rank*rows as usize*13+phase*7) as u8).collect::<Vec<_>>();
                let scales=(0..rows as usize*cols as usize/16).map(|at|[0x00,0x80,0x38,0x30,0x28,0x40,0x01,0x81][(at+expert+phase+rank)%8]).collect::<Vec<_>>();
                owners.push((raw(backend,&weight)?,raw(backend,&scales)?));shapes.push(shape);
                if tiled {tiled_owners.push((raw(backend,&spark_gpu_probe::expert_ops::nv_tile_codes(&weight,rows as usize,cols as usize))?,raw(backend,&spark_gpu_probe::expert_ops::nv_tile_scales(&scales,rows as usize,cols as usize))?));}
            }}
            let parts=shapes.iter().zip(&owners).map(|(&shape,(weight,scales))|(shape,weight,scales)).collect::<Vec<_>>();
            // TENSOR-CORE REBASE: a row-major table takes the host route (its tile kernel keeps the rc.8 chain), so only the
            // column-tile table is a device route. The reference stays the row-major batch (the generic body, the contract's
            // words): the column-tile route must give its bytes.
            let table=backend.device_route_table(&parts,experts as u32,picks as u32,phases,down)?;
            if table.is_some() {return Err("a row-major route table was admitted after the tensor-core rebase".into());}
            let tiled_shapes=shapes.iter().map(|&shape|Shape {block_rows:spark_gpu_probe::expert_ops::NV_TILE_ROWS,..shape}).collect::<Vec<_>>();
            let tiled_parts=tiled_shapes.iter().zip(&tiled_owners).map(|(&shape,(weight,scales))|(shape,weight,scales)).collect::<Vec<_>>();
            let tiled_table=if tiled {Some(backend.device_route_table(&tiled_parts,experts as u32,picks as u32,phases,down)?.ok_or("column-tile fixture shape not admitted")?)} else {None};
            let source=backend.upload_vector(&input(cols as usize,picks,down,rank))?.into_resident_shared()?;
            for ids in patterns(experts) {
                let device_ids=backend.upload_vector(&ids.iter().enumerate().map(|(slot,&id)|if slot==0 && id==0 && !calibrated {-0.0} else {id as f32}).collect::<Vec<_>>())?;
                let mut sorted=ids.iter().copied().enumerate().collect::<Vec<_>>();sorted.sort_by_key(|&(_,id)|id);
                let mut quantized=Vec::new();let mut slots=Vec::new();
                for phase in 0..phases as usize {for &(slot,expert) in &sorted {
                    let shape=shapes[phase*experts+expert];let source=if down {source.resident_range(slot*cols as usize..(slot+1)*cols as usize)?} else {source.resident_range(0..cols as usize)?};
                    quantized.push(backend.expert_operation(Shape {op:Op::Quantize,..shape},[Some(&source),None,None,None],None,None)?);slots.push((phase,slot,expert));
                }}
                let jobs=slots.iter().zip(&quantized).map(|(&(phase,_,expert),quantized)|{let at=phase*experts+expert;(shapes[at],quantized,&owners[at].0,&owners[at].1)}).collect::<Vec<_>>();
                let reference=backend.expert_projection_batch(&jobs)?;let reference=backend.read_vector_values(&reference)?;
                let mut expected=vec![0.0;reference.len()];
                for (job,&(phase,slot,_)) in slots.iter().enumerate() {expected[(phase*picks+slot)*rows as usize..(phase*picks+slot+1)*rows as usize].copy_from_slice(&reference[job*rows as usize..(job+1)*rows as usize]);}
                for (column_tiles,table) in table.as_ref().map(|table|(false,table)).into_iter().chain(tiled_table.as_ref().map(|table|(true,table))) {for recorded in [false,true] {
                    let queue=if recorded {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
                    for repeat in 0..3 {
                        let before=backend.stats();let retained=backend.device_route_projection(table,&device_ids,&source)?;
                        let output=backend.device_route_projection(table,&device_ids,&source)?;
                        if recorded && backend.stats().since(before).stream_wait_calls!=0 {return Err("device route waited before its consumer".into());}
                        let (retained,actual)=backend.read_resident_pair(&retained,&output)?;
                        compared+=backend.check_device_route_descriptors(table,&ids,&source,&output)? as u64;
                        if bytes(&retained)!=bytes(&expected) {return Err("reusing route scratch changed an earlier output".into());}
                        if bytes(&actual)!=bytes(&expected) {let at=actual.iter().zip(&expected).position(|(a,b)|a.to_bits()!=b.to_bits()).ok_or("route byte extent differs")?;
                            return Err(format!("route differs rank={rank} rows={rows} cols={cols} phases={phases} calibrated={calibrated} column_tiles={column_tiles} recorded={recorded} repeat={repeat} word={at} old={:08x} new={:08x}",expected[at].to_bits(),actual[at].to_bits()));}
                        compared+=actual.len() as u64*8;
                    }drop(queue);
                }}
            }
            // ROUTE ROWS: T rows of one step in one launch through the host call (`device_route_projection_rows`), against the
            // same rows one token at a time through the one-token route above. Rows cycle the ID patterns, so experts repeat
            // across rows (pattern 1 is pattern 0 reversed). ID strides: a plane of picks, and the route kernel's own 594-word rows.
            let mut rows_cases=0;let mut rows_waits=[0u128;2];
            if let Some(table)=tiled_table.as_ref() {
                let lists=patterns(experts);let (rows,cols,per)=(rows as usize,cols as usize,cols as usize*if down {picks} else {1});
                for tokens in [1usize,2,3,8] {for stride in [picks,594] {
                    let ids=(0..tokens).map(|token|lists[(token+tokens)%lists.len()].clone()).collect::<Vec<_>>();
                    let mut plane=vec![f32::NAN;tokens*stride];
                    for (token,row) in ids.iter().enumerate() {for (slot,&id) in row.iter().enumerate() {plane[token*stride+slot]=id as f32;}}
                    let plane=backend.upload_vector(&plane)?;
                    let source=backend.upload_vector(&(0..tokens).flat_map(|token|input(cols,picks,down,rank+token)).collect::<Vec<_>>())?.into_resident_shared()?;
                    let mut expected=vec![0.0;tokens*picks*phases as usize*rows];
                    for (token,row) in ids.iter().enumerate() {
                        let one_ids=backend.upload_vector(&row.iter().map(|&id|id as f32).collect::<Vec<_>>())?;
                        let one=backend.device_route_projection(table,&one_ids,&source.resident_range(token*per..(token+1)*per)?)?;let one=backend.read_vector_values(&one)?;
                        for phase in 0..phases as usize {for slot in 0..picks {
                            let (to,from)=(((phase*tokens+token)*picks+slot)*rows,(phase*picks+slot)*rows);expected[to..to+rows].copy_from_slice(&one[from..from+rows]);
                        }}
                    }
                    for recorded in [false,true] {
                        let queue=if recorded {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
                        // The rows build is a direct launch: it ends the recorded piece before it. The first time a piece of a new
                        // shape is seen its capture waits once (as for any new piece), so the no-wait check starts at the second repeat.
                        for repeat in 0..3 {
                            let before=backend.stats();let retained=backend.device_route_projection_rows(table,&plane,stride,&source,tokens)?;
                            let output=backend.device_route_projection_rows(table,&plane,stride,&source,tokens)?;
                            let waits=backend.stats().since(before).stream_wait_calls;if recorded {rows_waits[usize::from(repeat>0)]+=waits;}
                            if recorded && repeat>0 && waits!=0 {return Err(format!("route rows waited before its consumer: {waits} waits at repeat {repeat}"));}
                            let (retained,actual)=backend.read_resident_pair(&retained,&output)?;
                            compared+=backend.check_device_route_rows_descriptors(table,&ids,&source,&output)? as u64;
                            if bytes(&retained)!=bytes(&expected) {return Err("reusing route rows scratch changed an earlier output".into());}
                            if bytes(&actual)!=bytes(&expected) {let at=actual.iter().zip(&expected).position(|(a,b)|a.to_bits()!=b.to_bits()).ok_or("route rows byte extent differs")?;
                                return Err(format!("route rows differ rank={rank} rows={rows} cols={cols} phases={phases} calibrated={calibrated} down={down} tokens={tokens} stride={stride} recorded={recorded} repeat={repeat} word={at} one_token={:08x} rows={:08x}",expected[at].to_bits(),actual[at].to_bits()));}
                            compared+=actual.len() as u64*8;
                        }drop(queue);
                    }
                    rows_cases+=1;
                }}
            }
            receipt.push_str(&format!("DEVICE ROUTE PASS rank={rank} rows={rows} cols={cols} phases={phases} experts={experts} down={down} calibrated={calibrated} column_tiles_too={tiled} id_patterns={} repeats=3 direct_and_recorded=true scratch_reuse_before_drain=true route_rows_cases={rows_cases} route_rows_recorded_waits_first_repeat={} later_repeats={}\n",patterns(experts).len(),rows_waits[0],rows_waits[1]));
        }}}}
        Ok(())
    })?;
    receipt.push_str(&format!("DEVICE ROUTE COMPLETE compared_bytes={compared} differing_bytes=0 seconds={:.6}\n",started.elapsed().as_secs_f64()));
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
pub fn refuse(kind:&str)->Result<()> {
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<29,|backend| {
        let shape=Shape {op:Op::NvProject,rows:1,cols:512,block_rows:0,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0};
        let owners=(0..8).map(|_|Ok((raw(backend,&[0x22;256])?,raw(backend,&[0x38;32])?))).collect::<Result<Vec<_>>>()?;
        let parts=owners.iter().map(|(weight,scales)|(shape,weight,scales)).collect::<Vec<_>>();let table=backend.device_route_table(&parts,8,8,1,true)?.ok_or("refusal table not admitted")?;
        let mut ids=(0..8).map(|id|id as f32).collect::<Vec<_>>();ids[7]=match kind {"duplicate"=>0.0,"negativezero"=>-0.0,"nan"=>f32::NAN,"infinity"=>f32::INFINITY,"negative"=>-1.0,"fraction"=>1.5,"outside"=>8.0,_=>return Err("unknown route refusal".into())};
        let ids=backend.upload_vector(&ids)?;let input=backend.upload_vector(&vec![0.0;512*8])?;
        let output=backend.device_route_projection(&table,&ids,&input)?;backend.read_vector_values(&output)?;
        Err("INVALID ROUTE WAS NOT REFUSED".into())
    })
}
