//! Native NV row geometry screen. Saved bytes compare independent builds.
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Op,Shape}};
fn raw(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect()}
fn upload<'a>(backend:&Backend<'a>,bytes:&[u8])->Result<DeviceVector<'a>> {
    backend.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())
}
fn mixed(index:usize,job:usize)->u32 {
    let mut value=(index as u32).wrapping_add((job as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=format!("NV_GROUP native_rows={} physical_threads=32 timing=complete_public_call_including_allocation_zero_and_table conditioning_outside_timer=true\n",expert_ops::NV_ROWS_PER_WARP);
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let cache=backend.upload_vector(&vec![1.0;64<<20])?;
        for (rows,cols,max_jobs) in [(1u32,16u32,16usize),(7,32,16),(8,48,16),(9,64,16),(15,128,16),(16,16,16),(17,32,16),(31,48,16),(32,64,16),(33,128,16),(65,16,16),(1024,4096,16),(2048,2048,8)] {
            let (mut shapes,mut inputs,mut weights,mut scales,mut expected)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
            for job in 0..max_jobs {
                let shape=Shape {op:Op::NvProject,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75+job as f32/16.0,activation:0.125+job as f32/32.0,bound:0.0,role:0};
                let source=(0..cols as usize).map(|index|match (index+job)%13 {0=>0.0,1=>-0.0,2=>f32::from_bits(1),3=>-f32::from_bits(1),_=>((mixed(index,job)%47) as i32-23) as f32/17.0}).collect::<Vec<_>>();
                let quantized=expert_ops::reference(Shape {op:Op::Quantize,..shape},[&source,&[],&[],&[]],&[],&[])?;
                let bytes=(0..rows as usize*cols as usize/2).map(|index|mixed(index,job) as u8).collect::<Vec<_>>();
                let scale=(0..rows as usize*cols as usize/16).map(|index|if index%17==0 {0} else {0x28+(mixed(index,job+31)%64) as u8}).collect::<Vec<_>>();
                expected.push(expert_ops::reference(shape,[&quantized,&[],&[],&[]],&bytes,&scale)?);
                inputs.push(backend.upload_vector(&quantized)?);weights.push(upload(backend,&bytes)?);scales.push(upload(backend,&scale)?);shapes.push(shape);
            }
            let jobs=(0..max_jobs).map(|job|(shapes[job],&inputs[job],&weights[job],&scales[job])).collect::<Vec<_>>();
            for count in [1,2,8,16].into_iter().filter(|&count|count<=max_jobs) {
                let expected=raw(&expected[..count].concat());
                for repeat in 1..=3 {
                    let output=backend.expert_projection_batch(&jobs[..count])?;let actual=raw(&backend.read_vector_values(&output)?);
                    if actual!=expected {return Err(format!("NV GROUP BITS DIFFER rows={rows} cols={cols} jobs={count} repeat={repeat}"));}
                    super::write(&out.join(format!("batch-r{rows}-k{cols}-j{count}-{repeat}.f32")),&actual)?;
                }
                if rows>=1024 {for cold in [false,true] {for sample in 1..=5 {
                    if cold {let conditioned=backend.round_bf16_lanes(&cache)?;backend.synchronize()?;drop(conditioned);}
                    backend.synchronize()?;let started=Instant::now();let output=backend.expert_projection_batch(&jobs[..count])?;
                    let ns=started.elapsed().as_nanos();if raw(&backend.read_vector_values(&output)?)!=expected {return Err("timed NV group bytes differ".into());}
                    receipt.push_str(&format!("NV_GROUP_MICRO rows={rows} cols={cols} jobs={count} cold_cache={cold} sample={sample} ns={ns} conditioning_bytes=268435456 exact=true\n"));
                }}}
            }
            for job in 0..2 {let (shape,input,weight,scales)=jobs[job];
                let output=backend.expert_operation(shape,[Some(input),None,None,None],Some(weight),Some(scales))?;
                let actual=raw(&backend.read_vector_values(&output)?);if actual!=raw(&expected[job]) {return Err("single NV group bytes differ".into());}
                super::write(&out.join(format!("single-r{rows}-k{cols}-j{job}.f32")),&actual)?;
            }
            receipt.push_str(&format!("NV_GROUP_PASS rows={rows} cols={cols} max_jobs={max_jobs} repeats=3 raw_bits=true distinct_rows_and_jobs=true\n"));
            super::write(&out.join(format!("RECEIPT-r{rows}-k{cols}")),receipt.as_bytes())?;
        }
        edges(backend,out,&mut receipt)?;all_codes(backend,out,&mut receipt)?;column_tiles(backend,out,&mut receipt)?;row_major_groups(backend,&mut receipt)
    })?;super::write(&out.join("RECEIPT"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
fn edges(backend:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    for global in [1,0x0000_8000,0x0001_8000,0x007f_ffff,0x0080_0000,0x3f80_8000,0x3f81_8000,0x7f7f_0000,0x7f7f_ffff] {for case in 0..4 {
        let shape=Shape {op:Op::NvProject,rows:33,cols:16,block_rows:1,block_cols:16,tokens:1,global:f32::from_bits(global),activation:1.0,bound:0.0,role:0};
        let mut source=vec![if case==1 {-0.0} else {0.0};17];source[16]=1.0;
        if case>=2 {source[0]=if case==3 {6.0} else {1.0};source[1]=if case==3 {6.0} else {-0.0};}
        let mut weight=vec![0;33*8];for row in 0..33 {weight[row*8]=if case==3 {0xa2} else {0x02};}let scales=vec![0x38;33];
        let host=match expert_ops::reference(shape,[&source,&[],&[],&[]],&weight,&scales) {
            Ok(values)=>Some(raw(&values)),Err(error) if error=="expert operation output is not finite"=>None,Err(error)=>return Err(error),
        };
        let input=backend.upload_vector(&source)?;let weight=upload(backend,&weight)?;let scales=upload(backend,&scales)?;let jobs=[(shape,&input,&weight,&scales);2];let mut previous=None;
        for repeat in 1..=3 {
            let output=backend.expert_projection_batch(&jobs)?;let bytes=raw(&backend.read_vector_values(&output)?);
            if let Some(host)=&host&& bytes!=host.repeat(2) {return Err(format!("NV group finite edge differs global={global:08x} case={case}"));}
            if let Some(previous)=&previous {if previous!=&bytes {return Err("NV group edge repeat differs".into());}}else {previous=Some(bytes.clone());}
            super::write(&out.join(format!("edge-{global:08x}-case{case}-{repeat}.f32")),&bytes)?;
        }
        receipt.push_str(&format!("NV_GROUP_EDGE global_bits={global:08x} case={case} host_finite={} repeats=3 exact=true\n",host.is_some()));
    }}Ok(())
}
pub(super) fn all_codes(backend:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let shape=Shape {op:Op::NvProject,rows:65536,cols:16,block_rows:1,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0};
    let bytes=(0..65536usize).flat_map(|row|[row as u8;8]).collect::<Vec<_>>();
    let scale=(0..65536usize).map(|row|(row/256) as u8).collect::<Vec<_>>();
    let weight=upload(backend,&bytes)?;let scales=upload(backend,&scale)?;
    for pattern in [0,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_8000,0x3f81_8000,0x7f7f_0000,0x7f80_0000,0xff80_0000,0x7fc0_1234] {
        let mut source=vec![f32::from_bits(pattern);17];source[16]=1.0;let input=backend.upload_vector(&source)?;let mut previous=None;
        for repeat in 1..=3 {
            let output=backend.expert_operation(shape,[Some(&input),None,None,None],Some(&weight),Some(&scales))?;
            let actual=backend.read_vector_values(&output)?;
            for (row,&value) in actual.iter().enumerate() {let mut expected=[0.0];expert_ops::unit(shape,row,[&source,&[],&[],&[]],&bytes,&scale,&mut expected);
                if expected[0].is_finite() && value.to_bits()!=expected[0].to_bits() {return Err(format!("NV all-code finite row differs input={pattern:08x} row={row}"));}
            }
            let bytes=raw(&actual);if let Some(previous)=&previous {if previous!=&bytes {return Err("NV all-code repeated raw bits differ".into());}} else {previous=Some(bytes.clone());}
            super::write(&out.join(format!("allcodes-input{pattern:08x}-{repeat}.f32")),&bytes)?;
        }
        receipt.push_str(&format!("NV_GROUP_ALL_CODES input_bits={pattern:08x} paired_weight_codes=256 scale_codes=256 repeats=3 finite_host_bits=true nonfinite_raw_bytes_saved=true\n"));
    }Ok(())
}
/// COLUMN-TILE storage (expert_ops::NV_TILE_ROWS): the same logical matrices repacked on the host must give the row-major host
/// reference bit for bit through every product path: the decode batch (`nv_cols`), the single scalar body, a dense prompt group
/// (`nv_rows_cols`) and a prompt group of few tokens a part (`nv_cols` jobs). The launch counters prove which kernel ran.
fn column_tiles(backend:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let lut=backend.upload_vector(&expert_ops::decode_lut())?;
    let count=|name:&str|->u64 {spark_gpu_probe::prompt_launches::line().split_whitespace().find_map(|word|word.strip_prefix(&format!("{name}="))).and_then(|value|value.parse().ok()).unwrap_or(0)};
    for (rows,cols,max_jobs) in [(128u32,64u32,16usize),(128,4096,16),(256,512,16),(512,4096,16),(1024,4096,16),(2048,2048,8),(4096,2048,4)] {
        if !expert_ops::nv_tile_layout_admits(rows as usize,cols as usize) {return Err("column tile case is outside the storage rule".into());}
        let stride=cols as usize/16*17;let mut repacked=0usize;
        let (mut shapes,mut inputs,mut weights,mut scales,mut expected,mut flats,mut planes)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
        let source=|job:usize,token:usize|(0..cols as usize).map(|index|match (index+job+3*token)%13 {0=>0.0,1=>-0.0,2=>f32::from_bits(1),3=>-f32::from_bits(1),_=>((mixed(index+97*token,job)%47) as i32-23) as f32/17.0}).collect::<Vec<_>>();
        for job in 0..max_jobs {
            let flat=Shape {op:Op::NvProject,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75+job as f32/16.0,activation:0.125+job as f32/32.0,bound:0.0,role:0};
            let shape=Shape {block_rows:expert_ops::NV_TILE_ROWS,..flat};
            let quantized=expert_ops::reference(Shape {op:Op::Quantize,..flat},[&source(job,0),&[],&[],&[]],&[],&[])?;
            let bytes=(0..rows as usize*cols as usize/2).map(|index|mixed(index,job) as u8).collect::<Vec<_>>();
            let scale=(0..rows as usize*cols as usize/16).map(|index|if index%17==0 {0} else {0x28+(mixed(index,job+31)%64) as u8}).collect::<Vec<_>>();
            expected.push(expert_ops::reference(flat,[&quantized,&[],&[],&[]],&bytes,&scale)?);
            inputs.push(backend.upload_vector(&quantized)?);
            // The loader's path: row-major planes on the device, then the device repack. Its bytes must be the host repack's.
            let (row_codes,row_scales)=(upload(backend,&bytes)?,upload(backend,&scale)?);
            let (mut tile_codes,mut tile_scales)=(backend.vector(bytes.len()/4)?,backend.vector(scale.len()/4)?);
            backend.nv_tile_repack(&row_codes,&row_scales,&mut tile_codes,&mut tile_scales,rows as usize,cols as usize)?;backend.synchronize()?;
            if raw(&backend.read_vector_values(&tile_codes)?)!=expert_ops::nv_tile_codes(&bytes,rows as usize,cols as usize)
                || raw(&backend.read_vector_values(&tile_scales)?)!=expert_ops::nv_tile_scales(&scale,rows as usize,cols as usize) {
                return Err(format!("NV REPACK BYTES DIFFER rows={rows} cols={cols} job={job}"));
            }
            repacked+=bytes.len()+scale.len();
            weights.push(tile_codes);scales.push(tile_scales);
            shapes.push(shape);flats.push(flat);planes.push((bytes,scale));
        }
        let jobs=(0..max_jobs).map(|job|(shapes[job],&inputs[job],&weights[job],&scales[job])).collect::<Vec<_>>();
        for jobs_in in [1,2,8,16].into_iter().filter(|&n|n<=max_jobs) {
            let want=raw(&expected[..jobs_in].concat());
            for repeat in 1..=3 {
                let output=backend.expert_projection_batch(&jobs[..jobs_in])?;let actual=raw(&backend.read_vector_values(&output)?);
                if actual!=want {return Err(format!("NV COLUMN BITS DIFFER decode rows={rows} cols={cols} jobs={jobs_in} repeat={repeat}"));}
                super::write(&out.join(format!("columns-batch-r{rows}-k{cols}-j{jobs_in}-{repeat}.f32")),&actual)?;
            }
        }
        for job in 0..2.min(max_jobs) {let (shape,input,weight,scale)=jobs[job];
            let output=backend.expert_operation(shape,[Some(input),None,None,None],Some(weight),Some(scale))?;
            if raw(&backend.read_vector_values(&output)?)!=raw(&expected[job]) {return Err(format!("NV COLUMN BITS DIFFER single rows={rows} cols={cols} job={job}"));}
        }
        // Prompt groups: token rows quantized at each part's own scale, then one group launch. Dense lists take the tile kernel;
        // lists of few tokens a part take one decode job a (part, token).
        let (mut tile_runs,mut job_runs)=(0u64,0u64);
        for (case,tokens) in [vec![8usize,3,9],vec![5,16],vec![1,1,1,1],vec![1,2,1],vec![2,2]].into_iter().enumerate() {
            if tokens.len()>max_jobs {continue;}
            let total=tokens.iter().sum::<usize>();let mut joined=Vec::with_capacity(total*stride);let mut want=Vec::new();let mut first=0;
            let mut parts_shape=Vec::new();
            for (part,&m) in tokens.iter().enumerate() {
                let rows_source=(0..m).flat_map(|token|source(part,first+token)).collect::<Vec<_>>();
                let quantized=expert_ops::reference(Shape {op:Op::QuantizeRows,tokens:m as u32,..flats[part]},[&rows_source,&[],&[],&[]],&[],&[])?;
                want.extend(expert_ops::reference(Shape {op:Op::NvRows,tokens:m as u32,..flats[part]},[&quantized,&[],&[],&[]],&planes[part].0,&planes[part].1)?);
                joined.extend(quantized);parts_shape.push((Shape {op:Op::NvRows,tokens:m as u32,..shapes[part]},first));first+=m;
            }
            let input=backend.upload_vector(&joined)?;
            let parts=parts_shape.iter().enumerate().map(|(part,&(shape,first_row))|(shape,Some((&weights[part],&scales[part])),first_row)).collect::<Vec<_>>();
            // TENSOR-CORE REBASE: a dense list takes the tensor-core image when this build has one (counter fp4_cohort).
            let before=(count("fp4_cohort"),count("nv_cols"));
            for repeat in 1..=2 {
                let output=backend.expert_group(&parts,&input,stride,&lut)?;backend.synchronize()?;
                if raw(&backend.read_vector_values(&output)?)!=raw(&want) {return Err(format!("NV COLUMN BITS DIFFER prompt rows={rows} cols={cols} case={case} tokens={tokens:?} repeat={repeat}"));}
            }
            tile_runs+=count("fp4_cohort")-before.0;job_runs+=count("nv_cols")-before.1;
        }
        // The tile kernel takes at most 2,048 rows and whole 128-column steps (nv_rows); other dense lists keep the scalar body.
        if backend.tc_image_present() && tile_runs==0 {return Err(format!("the tensor-core prompt group did not run rows={rows} cols={cols}"));}
        if job_runs==0 {return Err(format!("the column prompt jobs did not run rows={rows} cols={cols}"));}
        receipt.push_str(&format!("NV_COLUMN_PASS rows={rows} cols={cols} max_jobs={max_jobs} decode_batches=1,2,8,16 single=true prompt_tile_launches={tile_runs} prompt_job_launches={job_runs} device_repack_bytes={repacked} device_repack_equals_host=true raw_bits=true\n"));
    }Ok(())
}
/// ROW-MAJOR storage in a prompt group (review finding B1 of the tensor-core rebase): a matrix outside the column-tile storage rule
/// keeps row-major bytes, and its token rows in a group must give the contract word of the reference (`expert_ops::nv_word`), for both
/// word functions, exactly as its one-row product does. The last BF16 rounding hides most order differences (about one word in a
/// thousand moves on random inputs), so the first rows of every matrix are SEARCHED on the host: rows whose rc.8 chain word differs
/// from the contract word for the part's first token. A kernel that kept the rc.8 chain fails on those rows.
fn row_major_groups(backend:&Backend<'_>,receipt:&mut String)->Result<()> {
    let lut=backend.upload_vector(&expert_ops::decode_lut())?;
    let source=|job:usize,token:usize,cols:usize|(0..cols).map(|index|match (index+job+3*token)%13 {0=>0.0,1=>-0.0,2=>f32::from_bits(1),_=>((mixed(index+97*token,job)%47) as i32-23) as f32/17.0}).collect::<Vec<_>>();
    // The rc.8 chain of one weight row on one quantized token row (17 words a block): one FP32 chain over all columns.
    let rc8=|shape:Shape,x:&[f32],bytes:&[u8],scale:&[u8]|->u32 {
        let global=shape.global*shape.activation;let mut sum=0.0f32;
        for block in 0..shape.cols as usize/16 {let fold=(expert_ops::e4m3(scale[block])*x[block*17+16])*global;
            for e in 0..16 {let byte=bytes[block*8+e/2];sum+=(expert_ops::e2m1(if e%2==0 {byte&15} else {byte>>4})*x[block*17+e])*fold;}}
        expert_ops::round(sum).to_bits()
    };
    let (mut checked,mut launches,mut telling)=(0usize,0usize,0usize);
    for (rows,cols,role) in [(9u32,64u32,expert_ops::NV_FIRST),(33,256,expert_ops::NV_FIRST),(33,256,expert_ops::NV_SECOND),(127,1024,expert_ops::NV_FIRST),(65,1024,expert_ops::NV_SECOND),(7,4096,expert_ops::NV_SECOND)] {
        let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
        let flat=|job:usize|Shape {op:Op::NvProject,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75+job as f32/16.0,activation:0.125+job as f32/32.0,bound:0.0,role};
        for (case,tokens) in [vec![3usize,1,5],vec![2,2],vec![17],vec![1,1,1]].into_iter().enumerate() {
            let (mut joined,mut want,mut first,mut shapes,mut planes)=(Vec::new(),Vec::new(),0usize,Vec::new(),Vec::new());
            for (part,&m) in tokens.iter().enumerate() {
                let rows_source=(0..m).flat_map(|token|source(part,first+token+case,k)).collect::<Vec<_>>();
                let quantized=expert_ops::reference(Shape {op:Op::QuantizeRows,tokens:m as u32,..flat(part)},[&rows_source,&[],&[],&[]],&[],&[])?;
                // The matrix: up to four searched rows first, then seeded rows.
                let (mut bytes,mut scale,mut seed,mut found)=(Vec::with_capacity(r*k/2),Vec::with_capacity(r*k/16),0usize,0usize);
                let one=Shape {rows:1,..flat(part)};
                while bytes.len()<r*k/2 {
                    let row_bytes=(0..k/2).map(|index|mixed(index,1000*part+seed+500) as u8).collect::<Vec<_>>();
                    let row_scale=(0..k/16).map(|index|if mixed(index,1000*part+seed+7).is_multiple_of(17) {0} else {0x18+(mixed(index,1000*part+seed+531)%96) as u8}).collect::<Vec<_>>();
                    seed+=1;
                    let tells=expert_ops::reference(one,[&quantized[..stride],&[],&[],&[]],&row_bytes,&row_scale)?[0].to_bits()!=rc8(one,&quantized[..stride],&row_bytes,&row_scale);
                    if found<4.min(r) && !tells && seed<60_000 {continue;}
                    if tells {found+=1;}
                    bytes.extend(row_bytes);scale.extend(row_scale);
                }
                telling+=found;
                want.extend(expert_ops::reference(Shape {op:Op::NvRows,tokens:m as u32,..flat(part)},[&quantized,&[],&[],&[]],&bytes,&scale)?);
                joined.extend(quantized);shapes.push((Shape {op:Op::NvRows,tokens:m as u32,..flat(part)},first));first+=m;
                planes.push((upload(backend,&bytes)?,upload(backend,&scale)?));
            }
            let input=backend.upload_vector(&joined)?;
            let parts=shapes.iter().enumerate().map(|(part,&(shape,first_row))|(shape,Some((&planes[part].0,&planes[part].1)),first_row)).collect::<Vec<_>>();
            let output=backend.expert_group(&parts,&input,stride,&lut)?;backend.synchronize()?;
            let got=backend.read_vector_values(&output)?;
            if let Some(at)=got.iter().zip(&want).position(|(a,b)|a.to_bits()!=b.to_bits()) {
                return Err(format!("NV ROW-MAJOR GROUP BITS DIFFER rows={rows} cols={cols} role={role} case={case} tokens={tokens:?} word={at} device={:08x} reference={:08x}",got[at].to_bits(),want[at].to_bits()));
            }
            // Alone: every (part, token) as its own one-row product through the generic body.
            let mut at=0usize;
            for (part,&m) in tokens.iter().enumerate() {for token in 0..m {
                let row=&joined[(shapes[part].1+token)*stride..(shapes[part].1+token+1)*stride];let x=backend.upload_vector(row)?;
                let alone=backend.read_vector_values(&backend.expert_operation(flat(part),[Some(&x),None,None,None],Some(&planes[part].0),Some(&planes[part].1))?)?;
                for (row,word) in alone.iter().enumerate() {if word.to_bits()!=got[at+row*m+token].to_bits() {
                    return Err(format!("NV ROW-MAJOR ALONE DIFFERS FROM GROUP rows={rows} cols={cols} role={role} case={case} part={part} token={token} row={row}"));}}
            }at+=r*m;}
            checked+=want.len();launches+=1;
        }
    }
    if telling==0 {return Err("NV ROW-MAJOR GROUP: the search found no row that tells the rc.8 chain from the contract word".into());}
    receipt.push_str(&format!("NV_ROW_MAJOR_GROUP_PASS shapes=6 roles=first,second token_lists=4 group_launches={launches} words={checked} rows_that_tell_rc8_from_contract={telling} group_equals_reference=true alone_equals_group=true\n"));Ok(())
}
