//! FOLD ROWS: the plain fold batch for 2 to 8 source rows (`flash_resident_projection_fold_rows` for BF16 weight patterns,
//! `flash_resident_projection_fold_rows_f32` for FP32 weights) against the forms the service runs for ONE source row, taken
//! from the BASELINE image: `flash_resident_projection_plain_bf16` (the BF16 fold batch) and `flash_resident_projection_batch`
//! with plain FP32 jobs (and `project_f32_ordered`, the router gate's one-row fold, when no round is asked).
//! Bytes: every output word of every source row of a group equals that row alone, for the model's fold shapes, ragged
//! shapes and seeded random shapes, 2 to 8 rows, both weight types, with and without the BF16 round; the same from NaN-filled
//! shared memory (the poisoned entries). Controls that must FAIL the same comparison: a row against another row's reference,
//! and a wrong-on-purpose launch (the group's rows read in reverse order). A neighbor's input is then changed: every other
//! row must keep its bytes and the changed row must follow its own new reference.
//! Time: the three groups of a step (router gate, narrow recurrent group, index group) at 2, 3, 4, 6 and 8 rows, device graph
//! stamps, rotating weight copies so every call streams from memory.
//! Run: kernel-bench --fold-rows <base ptx> <candidate ptx> [bytes|time|both]. Safe code only.
#![forbid(unsafe_code)]
use super::*;
use super::super::{capture_graph,sample,TIMER};

fn bf16_bits(values:&[f32])->Vec<u8> {values.iter().flat_map(|v|((v.to_bits()>>16) as u16).to_le_bytes()).collect()}
/// Weights of each job (FP32 words or BF16 patterns) and `[tokens][cols]` FP32 inputs.
struct Operands {weights:Vec<u64>,inputs:Vec<u64>}
fn operands(g:&mut Gpu,shapes:&[(usize,usize)],tokens:usize,bf16:bool,seed:u64,kinds:(u32,u32))->Result<Operands> {
    let (mut weights,mut inputs)=(vec![],vec![]);
    for (i,&(rows,cols)) in shapes.iter().enumerate() {
        let s=seed*31+i as u64;let values=pattern(kinds.0,rows*cols,s);
        weights.push(g.upload(&if bf16 {bf16_bits(&values)} else {le(&values)})?);
        inputs.push(g.upload(&le(&pattern(kinds.1,cols*tokens,s^5)))?);
    }
    Ok(Operands {weights,inputs})
}
/// One launch's batch and its poisoned, guarded outputs (one buffer a job).
struct Arm {batch:Batch,outputs:Vec<Buffer>}
/// The candidate: one descriptor a matrix with all its source rows; `rows.div_ceil(4)` blocks a descriptor.
/// `reverse` (the wrong-on-purpose launch) hands the kernel the source rows in reverse order.
fn group(g:&mut Gpu,shapes:&[(usize,usize)],o:&Operands,tokens:usize,bf16:bool,round:bool,reverse:Option<&[u64]>)->Result<Arm> {
    let mut batch=Batch {round:round as u32,..Batch::default()};let mut outputs=vec![];
    for (i,&(rows,cols)) in shapes.iter().enumerate() {
        let out=Buffer::new(g,&vec![f32::NAN;rows*tokens])?;out.poison(g)?;
        batch.layer.projections[i]=MxProjection {weights:o.weights[i],scales:if bf16 {2} else {4},input:reverse.map_or(o.inputs[i],|r|r[i]),output:out.address,
            rows:rows as u32,cols:cols as u32,tokens:tokens as u32,input_bf16:0,round:0,mode:0};
        batch.layer.ends[i+1]=batch.layer.ends[i]+rows.div_ceil(4) as u32;batch.plain|=1<<i;outputs.push(out);
    }
    batch.layer.count=shapes.len() as u32;Ok(Arm {batch,outputs})
}
/// The one-row batch of source row `token`: fold blocks (BF16 fold batch) or 128-row blocks (the FP32 batch).
fn one_row(g:&mut Gpu,shapes:&[(usize,usize)],o:&Operands,token:usize,bf16:bool,round:bool)->Result<Arm> {
    let mut batch=Batch {round:round as u32,..Batch::default()};let mut outputs=vec![];
    for (i,&(rows,cols)) in shapes.iter().enumerate() {
        let out=Buffer::new(g,&vec![f32::NAN;rows])?;out.poison(g)?;
        batch.layer.projections[i]=MxProjection {weights:o.weights[i],scales:if bf16 {2} else {4},input:o.inputs[i]+(token*cols*4) as u64,output:out.address,
            rows:rows as u32,cols:cols as u32,tokens:1,input_bf16:0,round:0,mode:0};
        batch.layer.ends[i+1]=batch.layer.ends[i]+if bf16 {rows.div_ceil(4)} else {rows.div_ceil(128)} as u32;batch.plain|=1<<i;outputs.push(out);
    }
    batch.layer.count=shapes.len() as u32;Ok(Arm {batch,outputs})
}
fn launch_arm(g:&Gpu,f:H,arm:&Arm,threads:u32)->Result<()> {let mut b=arm.batch;g.launch(f,[b.layer.ends[b.layer.count as usize],1,1],threads,0,&mut [ptr(&mut b)])}
/// The data words of a guarded output; an error when a guard changed.
fn words(g:&Gpu,out:&Buffer,what:&str)->Result<Vec<u8>> {
    let bytes=out.read(g)?;
    if bytes[..16]!=GUARD || bytes[bytes.len()-16..]!=GUARD {return Err(format!("FOLDROWS guard changed {what}"));}
    Ok(bytes[16..bytes.len()-16].to_vec())
}
struct Entries {rows:[H;2],poisoned:[H;2],fold_bf16:H,batch_f32:H,scalar_f32:H,rows_bf16:H}
struct Counts {comparisons:u64,bytes:u64,poisoned:u64,controls:u64,neighbors:u64,scalar:u64}
/// One (shapes, rows, weight type, round, patterns) case: bytes, poisoned bytes, the two failing controls, the neighbor change.
#[allow(clippy::too_many_arguments)]
fn case(g:&mut Gpu,e:&Entries,n:&mut Counts,label:&str,shapes:&[(usize,usize)],tokens:usize,bf16:bool,round:bool,seed:u64,kinds:(u32,u32))->Result<()> {
    let at=usize::from(!bf16);let what=|more:&str|format!("case={label} shapes={shapes:?} rows_in_group={tokens} bf16={bf16} round={round} patterns={kinds:?} {more}");
    let o=operands(g,shapes,tokens,bf16,seed,kinds)?;
    let (cand,poisoned)=(group(g,shapes,&o,tokens,bf16,round,None)?,group(g,shapes,&o,tokens,bf16,round,None)?);
    launch_arm(g,e.rows[at],&cand,160)?;launch_arm(g,e.poisoned[at],&poisoned,160)?;g.sync()?;
    let mut got=vec![];
    for (i,out) in cand.outputs.iter().enumerate() {
        let bytes=words(g,out,&what("candidate"))?;
        if bytes!=words(g,&poisoned.outputs[i],&what("poisoned"))? {return Err(format!("FOLDROWS POISON FAIL (a slot word was read that no producer wrote) {}",what("")));}
        n.poisoned+=1;got.push(bytes);
    }
    // Each source row alone, from the baseline image. `wants[token][job]`.
    let mut wants=vec![];
    for token in 0..tokens {
        let reference=one_row(g,shapes,&o,token,bf16,round)?;
        launch_arm(g,if bf16 {e.fold_bf16} else {e.batch_f32},&reference,if bf16 {160} else {128})?;g.sync()?;
        let mut row=vec![];for (i,out) in reference.outputs.iter().enumerate() {row.push(words(g,out,&what(&format!("one row job={i}")))?);}
        if !bf16 && !round {
            // The router gate's own one-row entry (fold geometry) gives the same words as the FP32 batch.
            for (i,&(rows,cols)) in shapes.iter().enumerate() {
                let out=Buffer::new(g,&vec![f32::NAN;rows])?;out.poison(g)?;
                launch_g(g,e.scalar_f32,o.weights[i],o.inputs[i]+(token*cols*4) as u64,out.address,rows,cols,plain_geo(rows,cols))?;g.sync()?;
                if words(g,&out,&what("scalar fold"))?!=row[i] {return Err(format!("FOLDROWS ONE-ROW FORMS DIFFER {}",what(&format!("token={token} job={i}"))));}
                n.scalar+=1;
            }
        }
        wants.push(row);
    }
    for token in 0..tokens {for (i,&(rows,_)) in shapes.iter().enumerate() {
        let slice=&got[i][token*rows*4..(token+1)*rows*4];
        if slice!=&wants[token][i][..] {
            let word=slice.chunks(4).zip(wants[token][i].chunks(4)).position(|(a,b)|a!=b).unwrap();
            let bits=|v:&[u8]|u32::from_le_bytes(v[word*4..word*4+4].try_into().unwrap());
            return Err(format!("FOLDROWS BYTE FAIL {} token={token} job={i} row={word} one_row={:08x} group={:08x}",what(""),bits(&wants[token][i]),bits(slice)));
        }
        n.comparisons+=1;n.bytes+=(rows*4) as u64;
    }}
    // The controls only where the source rows differ (random values; the other pattern kinds repeat values on purpose).
    if kinds==(0,0) {
        // Control 1: the same comparison against ANOTHER row's reference must fail.
        for token in 0..tokens {for (i,&(rows,_)) in shapes.iter().enumerate() {
            if got[i][token*rows*4..(token+1)*rows*4]==wants[(token+1)%tokens][i][..] {return Err(format!("FOLDROWS CONTROL DID NOT FAIL (another row's reference) {}",what(&format!("token={token} job={i}"))));}
            n.controls+=1;
        }}
        // Control 2, wrong on purpose: a launch that reads the group's rows in reverse order must fail the comparison.
        let mut reversed=vec![];
        for (i,&(_,cols)) in shapes.iter().enumerate() {
            let bytes=g.read(o.inputs[i],tokens*cols*4)?;let mut back=Vec::with_capacity(bytes.len());
            for token in (0..tokens).rev() {back.extend_from_slice(&bytes[token*cols*4..(token+1)*cols*4]);}
            reversed.push(g.upload(&back)?);
        }
        let wrong=group(g,shapes,&o,tokens,bf16,round,Some(&reversed))?;launch_arm(g,e.rows[at],&wrong,160)?;g.sync()?;
        for (i,&(rows,_)) in shapes.iter().enumerate() {
            let bytes=words(g,&wrong.outputs[i],&what("wrong on purpose"))?;
            if bytes==got[i] {return Err(format!("FOLDROWS CONTROL DID NOT FAIL (rows in reverse order) {}",what(&format!("job={i}"))));}
            // And the wrong launch is still each row's own chain: row t of it is row tokens-1-t of the group.
            for token in 0..tokens {if bytes[token*rows*4..(token+1)*rows*4]!=wants[tokens-1-token][i][..] {return Err(format!("FOLDROWS REVERSED ROWS DIFFER {}",what(&format!("token={token} job={i}"))));}}
            n.controls+=1;
        }
        // Neighbor change: new input words for ONE source row; every other row keeps its bytes, the changed row follows its own reference.
        let changed=tokens/2;
        for (i,&(_,cols)) in shapes.iter().enumerate() {g.upload_at(o.inputs[i]+(changed*cols*4) as u64,&le(&pattern(0,cols,seed*131+977+i as u64)))?;}
        let after=group(g,shapes,&o,tokens,bf16,round,None)?;launch_arm(g,e.rows[at],&after,160)?;
        let reference=one_row(g,shapes,&o,changed,bf16,round)?;launch_arm(g,if bf16 {e.fold_bf16} else {e.batch_f32},&reference,if bf16 {160} else {128})?;g.sync()?;
        for (i,&(rows,_)) in shapes.iter().enumerate() {
            let bytes=words(g,&after.outputs[i],&what("after a neighbor changed"))?;let own=words(g,&reference.outputs[i],&what("changed row alone"))?;
            for token in 0..tokens {
                let (now,before)=(&bytes[token*rows*4..(token+1)*rows*4],&got[i][token*rows*4..(token+1)*rows*4]);
                if token==changed {
                    if now!=&own[..] || now==before {return Err(format!("FOLDROWS CHANGED ROW FAIL {}",what(&format!("token={token} job={i}"))));}
                } else if now!=before {return Err(format!("FOLDROWS NEIGHBOR FAIL (a row's bits follow its neighbor) {}",what(&format!("token={token} changed={changed} job={i}"))));}
                n.neighbors+=1;
            }
        }
    }
    Ok(())
}
pub(crate) fn run(base_path:&str,candidate_path:&str,mode:&str)->Result<()> {
    let (a,b)=(std::fs::read(base_path).map_err(|e|e.to_string())?,std::fs::read(candidate_path).map_err(|e|e.to_string())?);
    println!("FOLDROWS baseline_sha256={} candidate_sha256={} mode={mode}",sha256::digest(&a),sha256::digest(&b));
    let mut gpu=Gpu::new()?;let (ma,mb)=(gpu.module(&a)?,gpu.module(&b)?);let tm=gpu.module(TIMER.as_bytes())?;
    let g=&mut gpu;let timer=g.function(tm,"kb_stamp")?;
    let e=Entries {rows:[g.function(mb,"flash_resident_projection_fold_rows")?,g.function(mb,"flash_resident_projection_fold_rows_f32")?],
        poisoned:[g.function(mb,"flash_resident_projection_fold_rows_poisoned")?,g.function(mb,"flash_resident_projection_fold_rows_f32_poisoned")?],
        fold_bf16:g.function(ma,"flash_resident_projection_plain_bf16")?,batch_f32:g.function(ma,"flash_resident_projection_batch")?,
        scalar_f32:g.function(ma,"project_f32_ordered")?,rows_bf16:g.function(ma,"flash_resident_projection_bf16_rows")?};
    let started=Instant::now();
    // The three groups of a decode step, rank-local: the router gate (FP32), the narrow recurrent group and the index group (BF16).
    let router=vec![(144usize,4096usize)];let narrow=vec![(128,4096),(32,4096),(128,4096)];let index=vec![(64,4096),(64,4096),(16,4096)];
    if mode!="time" {
        let mut cases:Vec<(String,Vec<(usize,usize)>)>=vec![
            ("router".into(),router.clone()),("router_logical".into(),vec![(288,4096)]),("narrow".into(),narrow.clone()),("index".into(),index.clone()),
            ("narrow_logical".into(),vec![(256,4096),(64,4096),(256,4096)]),("nine".into(),vec![(16,4096);9]),
            // Ragged last blocks, short last chunks, the narrowest and the widest rows of the envelope.
            ("one_row_matrix".into(),vec![(1,1024)]),("ragged".into(),vec![(5,1056),(6,1024),(7,2080)]),("short_chunks".into(),vec![(9,1120),(3,1184),(2,1312)]),
            ("wide".into(),vec![(3,8192),(130,8192)]),("tall".into(),vec![(512,1024)]),
        ];
        let mut rng=Rng(0x51ab_7c3d_9e02_4f17);
        for index in 0..36 {
            let jobs=1+(rng.next()%4) as usize;
            cases.push((format!("random{index}"),(0..jobs).map(|_|(1+(rng.next()%if index%6==0 {512} else {150}) as usize,32*(32+(rng.next()%225) as usize))).collect()));
        }
        let mut n=Counts {comparisons:0,bytes:0,poisoned:0,controls:0,neighbors:0,scalar:0};
        for (index,(label,shapes)) in cases.iter().enumerate() {
            for bf16 in [true,false] {for tokens in 2..=8usize {for round in [false,true] {
                for &kinds in &[(0u32,0u32),(5,5),(2,2),(6,7),(8,0)] {
                    if index>=6 && kinds!=(0,0) && kinds!=(6,7) {continue;}
                    if index>=11 && (kinds!=(0,0) || (round && tokens%2==1)) {continue;}
                    case(g,&e,&mut n,label,shapes,tokens,bf16,round,500+index as u64,kinds)?;g.clear()?;
                }
            }}}
            if index%8==0 {println!("FOLDROWS case {index} ({label}) ok comparisons={} elapsed_s={:.1}",n.comparisons,started.elapsed().as_secs_f64());}
        }
        println!("FOLDROWS BYTES PASS cases={} comparisons={} output_bytes={} poisoned_equal={} one_row_scalar_checks={} controls_failed_as_they_must={} neighbor_checks={} elapsed_s={:.1}",
            cases.len(),n.comparisons,n.bytes,n.poisoned,n.scalar,n.controls,n.neighbors,started.elapsed().as_secs_f64());
    }
    if mode!="bytes" {
        for (label,shapes,bf16) in [("router",&router,false),("narrow",&narrow,true),("index",&index,true)] {
            let weight_bytes:usize=shapes.iter().map(|&(r,c)|r*c*if bf16 {2} else {4}).sum();
            let copies=(256usize<<20).div_ceil(weight_bytes).clamp(2,96);
            for tokens in [2usize,3,4,6,8] {
                // Arms, each a list of launches (entry, threads, batch) a weight copy:
                // one_row: the one-row form of ONE source row (the floor). fold_loop: that form once a source row (the service's form at
                // 2 rows, and at 3 for BF16). current: the service's form at this row count. fold_rows: the candidate, one launch.
                let mut arms:Vec<(&str,Vec<Vec<(H,u32,Batch)>>)>=vec![("one_row",vec![]),("fold_loop",vec![]),("current",vec![]),("fold_rows",vec![])];
                for copy in 0..copies {
                    let o=operands(g,shapes,tokens,bf16,9000+copy as u64,(0,0))?;
                    let scalar=|g:&mut Gpu,token:usize|->Result<Vec<(H,u32,Batch)>> {
                        // FP32: one `project_f32_ordered` launch a matrix. It takes five scalars, so it is timed as its batch twin:
                        // the BF16 fold batch has the same blocks and body; for FP32 the fold entry is launched through `launch_g` below.
                        let arm=one_row(g,shapes,&o,token,bf16,false)?;Ok(vec![(if bf16 {e.fold_bf16} else {e.batch_f32},if bf16 {160} else {128},arm.batch)])
                    };
                    arms[0].1.push(scalar(g,0)?);
                    let mut looped=vec![];for token in 0..tokens {looped.extend(scalar(g,token)?);}
                    arms[1].1.push(looped);
                    // The service today: FP32 3 to 8 rows one mixed batch of (matrix, row) jobs; FP32 2 rows two fold launches (timed in
                    // the scalar arm below); BF16 2 and 3 rows the fold batch of (matrix, row) jobs; BF16 4, 6, 8 rows the row entry
                    // with parts of 4, 3, 4 rows.
                    let mut current=vec![];
                    if !bf16 || tokens<=3 {
                        let mut batch=Batch::default();
                        for (i,&(rows,cols)) in shapes.iter().enumerate() {for token in 0..tokens {
                            if batch.layer.count==9 {current.push((if bf16 {e.fold_bf16} else {e.batch_f32},if bf16 {160} else {128},batch));batch=Batch::default();}
                            let out=Buffer::new(g,&vec![f32::NAN;rows])?;let at=batch.layer.count as usize;
                            batch.layer.projections[at]=MxProjection {weights:o.weights[i],scales:if bf16 {2} else {4},input:o.inputs[i]+(token*cols*4) as u64,output:out.address,rows:rows as u32,cols:cols as u32,tokens:1,input_bf16:0,round:0,mode:0};
                            batch.layer.ends[at+1]=batch.layer.ends[at]+if bf16 {rows.div_ceil(4)} else {rows.div_ceil(128)} as u32;batch.plain|=1<<at;batch.layer.count+=1;
                        }}
                        if batch.layer.count!=0 {current.push((if bf16 {e.fold_bf16} else {e.batch_f32},if bf16 {160} else {128},batch));}
                    } else {
                        let part=if tokens==6 {3} else {4};let mut batch=Batch::default();
                        for (i,&(rows,cols)) in shapes.iter().enumerate() {for view in 0..tokens/part {
                            if batch.layer.count==9 {current.push((e.rows_bf16,128,batch));batch=Batch::default();}
                            let out=Buffer::new(g,&vec![f32::NAN;rows*part])?;let at=batch.layer.count as usize;
                            batch.layer.projections[at]=MxProjection {weights:o.weights[i],scales:2,input:o.inputs[i]+(view*part*cols*4) as u64,output:out.address,rows:rows as u32,cols:cols as u32,tokens:part as u32,input_bf16:0,round:0,mode:0};
                            batch.layer.ends[at+1]=batch.layer.ends[at]+(rows.div_ceil(128)*part) as u32;batch.plain|=1<<at;batch.layer.count+=1;
                        }}
                        if batch.layer.count!=0 {current.push((e.rows_bf16,128,batch));}
                    }
                    arms[2].1.push(current);
                    let cand=group(g,shapes,&o,tokens,bf16,false,None)?;arms[3].1.push(vec![(e.rows[usize::from(!bf16)],160,cand.batch)]);
                }
                // The FP32 router's own one-row fold entry, once a source row (the service's two-row form after the selector).
                let scalar_fold:Vec<Vec<(u64,u64,u64)>>=if bf16 {vec![]} else {
                    let mut all=vec![];for copy in 0..copies {let o=operands(g,shapes,tokens,false,9000+copy as u64,(0,0))?;let mut v=vec![];
                        for token in 0..tokens {let out=Buffer::new(g,&vec![f32::NAN;shapes[0].0])?;v.push((o.weights[0],o.inputs[0]+(token*shapes[0].1*4) as u64,out.address));}all.push(v);}all};
                let times=g.upload(&[0;16])?;let mut samples=vec![Vec::new();arms.len()+1];
                {
                    let g=&*g;let mut graphs=Vec::new();
                    for (_,per_copy) in &arms {graphs.push(capture_graph(g,|| {
                        let mut stamp=times;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
                        for launches in per_copy {for (f,threads,batch) in launches {let mut b=*batch;g.launch(*f,[b.layer.ends[b.layer.count as usize],1,1],*threads,0,&mut [ptr(&mut b)])?;}}
                        stamp=times+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;Ok(())
                    })?);}
                    if !bf16 {graphs.push(capture_graph(g,|| {
                        let mut stamp=times;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
                        for copy in &scalar_fold {for &(w,x,y) in copy {launch_g(g,e.scalar_f32,w,x,y,shapes[0].0,shapes[0].1,plain_geo(shapes[0].0,shapes[0].1))?;}}
                        stamp=times+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;Ok(())
                    })?);}
                    for _ in 0..3 {for graph in &graphs {sample(graph,times)?;}}
                    for repeat in 0..11 {for k in 0..graphs.len() {let a=(k+repeat)%graphs.len();samples[a].push(sample(&graphs[a],times)?.1/copies as f64);}}
                }
                for (i,(name,_)) in arms.iter().enumerate() {
                    println!("FOLDROWS TIME group={label} rows_in_group={tokens} arm={name} us={:.3} min={:.3} max={:.3} copies={copies} clock=graph_stamps",
                        median(samples[i].clone()),samples[i].iter().cloned().fold(f64::MAX,f64::min),samples[i].iter().cloned().fold(0.0,f64::max));
                }
                if !bf16 {let s=&samples[arms.len()];println!("FOLDROWS TIME group={label} rows_in_group={tokens} arm=scalar_fold_loop us={:.3} min={:.3} max={:.3} copies={copies} clock=graph_stamps",
                    median(s.clone()),s.iter().cloned().fold(f64::MAX,f64::min),s.iter().cloned().fold(0.0,f64::max));}
                g.clear()?;
            }
        }
    }
    println!("FOLDROWS DONE elapsed_s={:.1}",started.elapsed().as_secs_f64());
    Ok(())
}
