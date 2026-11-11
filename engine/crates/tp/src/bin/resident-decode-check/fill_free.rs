//! One box, no model: each skipped fill runs through its production backend call under three modes (the head's zero fill, a
//! poison fill, no fill) and the outputs must match bit for bit. A word that no writer stored differs between zero and poison.
//! Eager and recorded paths. The no-fill arm follows dirty arms, so it starts from used pool bytes. It also prints zero-fill counts.
use std::{fs,path::Path};
use engine_model::plan::layer::Topology;
use spark_gpu_probe::{causal_ops::{Op as CausalOp,Shape as CausalShape},driver::{Backend,DeviceVector,ResidentProjection,with_backend},expert_ops::{Op,Shape},fill_free::{Mode,set_mode},mxfp8_projection::Shape as MxShape,norm_boundary::WeightType};
use spark_tp::{Result,resident_weights::{Rows,gather_stage}};
const MODES:[Mode;3]=[Mode::Zero,Mode::Poison,Mode::Skip];
fn raw<'d>(backend:&Backend<'d>,values:&[u8])->Result<DeviceVector<'d>> {
    let mut padded=values.to_vec();padded.resize(values.len().div_ceil(4)*4,0);
    let mut value=backend.vector(padded.len()/4)?;backend.write_vector_bytes(&mut value,&padded)?;value.into_resident_shared()
}
fn activation(cols:usize,seed:usize)->Vec<f32> {
    let patterns=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x80800000,0x3f808000,0x3f818000,0xbf808000,0x3f800000,0xbf800000,0x3e800000];
    (0..cols).map(|at|if at%3==0 {f32::from_bits(patterns[(at+seed)%patterns.len()])} else {((at+seed) as i32%31-15) as f32/16.0}).collect()
}
struct Tally {cases:usize,words:usize,lines:String}
/// Run one call under each mode and compare. `need_none`: the no-fill arm must run no zero fill at all.
fn arms(backend:&Backend<'_>,queued:bool,tally:&mut Tally,name:&str,need_none:bool,mut work:impl FnMut()->Result<Vec<f32>>)->Result<()> {
    backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
    let mut first:Option<Vec<u32>>=None;let (mut counts,mut zero_arm)=((0,0,0),(0,0,0));
    for mode in MODES {
        set_mode(mode);let before=backend.stats();let got=work();let after=backend.stats().since(before);set_mode(Mode::Skip);
        let got=got?.iter().map(|value|value.to_bits()).collect::<Vec<_>>();
        if mode==Mode::Skip {counts=(after.zero_fill_launches,after.zero_fill_rows,after.zero_fill_words);}
        if mode==Mode::Zero {zero_arm=(after.zero_fill_launches,after.zero_fill_rows,after.zero_fill_words);}
        match &first {
            None=>first=Some(got),
            Some(zero)=>if *zero!=got {
                let at=zero.iter().zip(&got).position(|(a,b)|a!=b);
                return Err(format!("FILL FREE BYTE FAIL {name} mode={mode:?} words={} first_word={at:?}",got.len()));
            },
        }
    }
    drop(queue);let words=first.map_or(0,|bits|bits.len());
    if need_none && counts!=(0,0,0) {return Err(format!("FILL FREE COUNT FAIL {name}: the no-fill arm ran zero fills {counts:?}"));}
    tally.cases+=1;tally.words+=words;tally.lines.push_str(&format!("FILL FREE PASS {name} words={words} modes=zero,poison,skip zero_arm_fill_launches={} rows={} words={} skip_arm_fill_launches={} rows={} words={}\n",zero_arm.0,zero_arm.1,zero_arm.2,counts.0,counts.1,counts.2));Ok(())
}
fn pass<'d>(backend:&Backend<'d>,tally:&mut Tally,out:&Path,queued:bool)->Result<()> {
    let tag=if queued {"recorded"} else {"eager"};
    // Control: with no writer, each mode must show what it stored. The comparison below proves nothing unless poison really lands.
    for (mode,word) in [(Mode::Zero,0u32),(Mode::Poison,spark_gpu_probe::fill_free::POISON)] {
        backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
        for words in [1usize,17,4096,40000] {
            set_mode(mode);let mut output=backend.vector(words)?;let stored=backend.skip_fill(&mut output);set_mode(Mode::Skip);stored?;
            let got=backend.read_vector_values(&output)?;
            if got.len()!=words || got.iter().any(|value|value.to_bits()!=word) {return Err(format!("FILL FREE CONTROL FAIL {tag} mode={mode:?} words={words}"));}
            tally.lines.push_str(&format!("FILL FREE CONTROL PASS {tag} mode={mode:?} words={words} every_word={word:#010x}\n"));
        }
        drop(queue);
    }
    // NV activation quantization (production widths and the swept edges), one job and a job table.
    for cols in [16usize,128,2048,4096,8192,16384] {
        let input=backend.upload_vector(&activation(cols,cols))?;
        let shape=Shape {op:Op::Quantize,rows:1,cols:cols as u32,block_rows:0,block_cols:16,tokens:1,global:1.0,activation:0.125,bound:0.0};
        arms(backend,queued,tally,&format!("{tag} quantize cols={cols}"),true,||backend.read_vector_values(&backend.expert_operation(shape,[Some(&input),None,None,None],None,None)?))?;
        if cols%128==0 {
            let jobs=(0..8).map(|job|(Shape {activation:0.125*(1+job) as f32,..shape},&input)).collect::<Vec<_>>();
            arms(backend,queued,tally,&format!("{tag} quantize batch cols={cols} jobs=8"),true,||backend.read_vector_values(&backend.expert_quantize_batch(&jobs)?.0))?;
        }
    }
    // FP8 tile: the real rank-local decode shapes, and one job table.
    for (rows,cols) in [(1024usize,4096usize),(2048,2048),(2048,16384),(8192,1536),(768,4096),(288,4096)] {
        let shape=Shape {op:Op::Fp8Serial,rows:rows as u32,cols:cols as u32,block_rows:128,block_cols:128,tokens:1,global:1.0,activation:1.0,bound:0.0};
        if !spark_gpu_probe::expert_ops::fp8_tile_admits(&shape) {return Err(format!("FILL FREE FP8 shape is not a tile shape rows={rows} cols={cols}"));}
        let weight=raw(backend,&(0..rows*cols).map(|at|{let magnitude=0x20+((at*7)%79) as u8;magnitude|if at%5==0 {128} else {0}}).collect::<Vec<_>>())?;
        let scales=raw(backend,&(0..rows.div_ceil(128)*cols.div_ceil(128)).flat_map(|at|(0.001953125*(1+at%7) as f32).to_le_bytes()).collect::<Vec<_>>())?;
        let input=backend.upload_vector(&activation(cols,rows))?;
        arms(backend,queued,tally,&format!("{tag} fp8 tile rows={rows} cols={cols}"),true,||backend.read_vector_values(&backend.expert_operation(shape,[Some(&input),None,None,None],Some(&weight),Some(&scales))?))?;
        let jobs=(0..2).map(|_|(shape,&input,&weight,&scales)).collect::<Vec<_>>();
        arms(backend,queued,tally,&format!("{tag} fp8 batch rows={rows} cols={cols} jobs=2"),false,||backend.read_vector_values(&backend.expert_projection_batch(&jobs)?))?;
    }
    // NV tile: the route table of a gate and up pair and of a down, and the direct job table.
    for (rows,cols,phases,down) in [(1024u32,4096u32,2u32,false),(2048,2048,1,true)] {
        let experts=16usize;let (mut owners,mut shapes)=(Vec::new(),Vec::new());
        for phase in 0..phases as usize {for expert in 0..experts {
            shapes.push(Shape {op:Op::NvProject,rows,cols,block_rows:0,block_cols:16,tokens:1,global:if expert%2==0 {0.5} else {1.0},activation:0.125*(1+(expert+phase)%7) as f32,bound:0.0});
            owners.push((raw(backend,&(0..rows as usize*cols as usize/2).map(|at|(at+expert*19+phase*7) as u8).collect::<Vec<_>>())?,
                raw(backend,&(0..rows as usize*cols as usize/16).map(|at|[0x00,0x80,0x38,0x30,0x28,0x40,0x01,0x81][(at+expert+phase)%8]).collect::<Vec<_>>())?));
        }}
        let parts=shapes.iter().zip(&owners).map(|(&shape,(weight,scales))|(shape,weight,scales)).collect::<Vec<_>>();
        let table=backend.device_route_table(&parts,experts as u32,8,phases,down)?.ok_or("fill free route table not admitted")?;
        let source=backend.upload_vector(&activation(cols as usize*if down {8} else {1},rows as usize))?.into_resident_shared()?;
        let ids=backend.upload_vector(&[0.0,7.0,1.0,6.0,2.0,5.0,3.0,4.0])?;
        arms(backend,queued,tally,&format!("{tag} nv tile route rows={rows} cols={cols} phases={phases} down={down}"),true,||backend.read_vector_values(&backend.device_route_projection(&table,&ids,&source)?))?;
        if !down {
            let quantized=(0..8).map(|at|backend.expert_operation(Shape {op:Op::Quantize,..shapes[at]},[Some(&source),None,None,None],None,None)).collect::<Result<Vec<_>>>()?;
            let jobs=(0..8).map(|at|(shapes[at],&quantized[at],&owners[at].0,&owners[at].1)).collect::<Vec<_>>();
            arms(backend,queued,tally,&format!("{tag} nv batch rows={rows} cols={cols} jobs=8"),false,||backend.read_vector_values(&backend.expert_projection_batch(&jobs)?))?;
        }
    }
    // MX quantization keeps its fill (its body forms output slices). The mixed batch skips.
    for cols in [32usize,4096,16384] {
        let input=backend.upload_vector(&activation(cols,cols))?;
        let shape=CausalShape {op:CausalOp::Mxfp8QuantizePositiveZero,heads:0,dim:0,taps:0,rows:0,cols:cols as u32,epsilon:0.0,bound:0.0};
        arms(backend,queued,tally,&format!("{tag} mx quantize keeps its fill cols={cols}"),false,||backend.read_vector_values(&backend.causal_operation(shape,[Some(&input),None,None,None],None)?))?;
    }
    for (case,(cols,mx_rows,plain_rows)) in [(32usize,[1usize,5,9],[1usize,7,129]),(4096,[4096,3072,1024],[64,32,64])].into_iter().enumerate() {
        let decoded=backend.upload_vector(spark_gpu_probe::dynamic_mxfp8::Activation::from_f32_positive_zero(&activation(cols,case))?.decoded())?;
        let source=backend.upload_vector(&activation(cols,case))?;let (mut matrices,mut plain)=(Vec::new(),Vec::new());
        for job in 0..3 {
            let shape=MxShape::new(mx_rows[job],cols,WeightType::F32)?;
            let mut bytes=(0..shape.weight_bytes()).map(|index|{let magnitude=0x20+((index*7+job*3)%79) as u8;magnitude|if index%5==0 {128} else {0}}).collect::<Vec<_>>();
            bytes.extend((0..shape.scale_bytes()).map(|index|120+((index+job)%12) as u8));
            let path=out.join(format!("{tag}-case{case}-mx{job}.raw"));super::write(&path,&bytes)?;let file=fs::File::open(&path).map_err(|error|error.to_string())?;
            let ranges=[spark_gpu_probe::load::WeightRange {file_offset:0,destination:0,bytes:bytes.len(),row_bytes:0,row_stride:0}];
            let arena=backend.load_weight_arena(bytes.len(),&[spark_gpu_probe::load::WeightFile {file:&file,ranges:&ranges}])?;
            matrices.push(backend.resident_mxfp8_projection(&arena,shape)?);
            plain.push(backend.upload_vector(&(0..plain_rows[job]*cols).map(|index|(index as i32%23-11) as f32/32.0).collect::<Vec<_>>())?);
        }
        let total=mx_rows.iter().sum::<usize>()+plain_rows.iter().sum::<usize>();
        let jobs=(0..6).map(|job|if job%2==0 {ResidentProjection::Mxfp8 {matrix:&matrices[job/2],input:&decoded}} else {let job=job/2;ResidentProjection::Plain {rows:plain_rows[job],cols,weight:&plain[job],input:&source}}).collect::<Vec<_>>();
        for round in [false,true] {
            arms(backend,queued,tally,&format!("{tag} mixed batch cols={cols} round={round} jobs=6"),true,||{let mut output=backend.vector(total)?;backend.resident_projection_batch(&jobs,&mut output,round)?;backend.read_vector_values(&output)})?;
        }
    }
    // The exchange keeps its fill: a failed take can leave some or all peer words unwritten and queued consumers read them. These cases only guard that path's bits.
    for (rank,counts) in [(0usize,vec![2304usize,1024,1024]),(1,vec![2304,1024,1024]),(0,vec![16384]),(1,vec![16384]),(0,vec![1,17,129])] {
        let words=counts.iter().sum::<usize>();
        let mailbox=backend.tp_mailbox(words*4)?;let mut host=mailbox.take_host().ok_or("fill free fixture host already taken")?;
        let local=counts.iter().enumerate().map(|(job,&count)|backend.upload_vector(&activation(count,job+rank))).collect::<Result<Vec<_>>>()?;
        let peer=counts.iter().enumerate().flat_map(|(job,&count)|activation(count,job+1-rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        let rows=counts.iter().map(|&count|Rows {start:rank*count,count,total:count*2,split:true}).collect::<Vec<_>>();
        let mut sequence=0u64;
        arms(backend,queued,tally,&format!("{tag} exchange keeps its fill rank={rank} words={words}"),false,||{
            host.deliver(sequence+1,&peer)?;let output=gather_stage(backend,&mailbox,&mut sequence,Topology {rank:rank as u32,parts:2},&local,&rows)?;
            let mut values=Vec::new();for value in &output {values.extend(backend.read_vector_values(value)?);}
            backend.synchronize()?;mailbox.require_success()?;Ok(values)
        })?;
    }
    Ok(())
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut tally=Tally {cases:0,words:0,lines:String::new()};
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<32,|backend| {
        pass(backend,&mut tally,out,false)?;pass(backend,&mut tally,out,true)
    })?;
    let line=format!("{}FILL FREE COMPLETE cases={} compared_words={} differing_words=0 paths=eager,recorded\n",tally.lines,tally.cases,tally.words);
    super::write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
