//! Exhaustive finite dispatch envelope, plus legacy guard boundaries.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{Op,Shape}};
const COLS:[u32;11]=[16,32,48,64,128,256,512,1024,2048,4096,4112];
fn mixed(index:usize,job:usize)->u32 {let mut value=(index as u32).wrapping_add((job as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)}
fn raw(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect()}
fn upload<'a>(b:&Backend<'a>,bytes:&[u8],offset:usize)->Result<DeviceVector<'a>> {
    let mut values=vec![0.0;offset];values.extend(bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}));
    b.upload_vector(&values)?.into_resident_shared()
}
fn shape(rows:u32,cols:u32,job:usize)->Shape {Shape {op:Op::NvProject,rows,cols,block_rows:1,block_cols:16,tokens:1,
    global:0.5+(mixed(job,91)%65536) as f32/65536.0,activation:0.125+job as f32/32.0,bound:0.0,role:0}}
fn family(b:&Backend<'_>,out:&Path,cols:u32,rows:&[u32],offset:usize,receipt:&mut String)->Result<()> {
    let maximum=*rows.iter().max().ok_or("empty NV family")? as usize;
    let (mut inputs,mut weights,mut scales)=(Vec::new(),Vec::new(),Vec::new());
    for job in 0..16 {
        let input=(0..cols as usize/16).flat_map(|block|{
            let mut values=(0..16).map(|feature|spark_gpu_probe::expert_ops::e2m1((mixed(block*16+feature,job)&15) as u8)).collect::<Vec<_>>();
            values.push(if block%19==0 {0.0} else {1.0});values
        }).collect::<Vec<_>>();
        let weight=(0..maximum*cols as usize/2).map(|at|mixed(at,job) as u8).collect::<Vec<_>>();
        let scale=(0..maximum*cols as usize/16).map(|at|if at%37==0 {0} else {0x20+(mixed(at,job+31)%64) as u8}).collect::<Vec<_>>();
        inputs.push(b.upload_vector(&input)?);weights.push(upload(b,&weight,offset)?);scales.push(upload(b,&scale,offset)?);
    }
    for &r in rows {
        let weight_words=(r as usize*cols as usize/2).div_ceil(4);let scale_words=(r as usize*cols as usize/16).div_ceil(4);
        let w=weights.iter().map(|v|v.resident_range(offset..offset+weight_words)).collect::<Result<Vec<_>>>()?;
        let s=scales.iter().map(|v|v.resident_range(offset..offset+scale_words)).collect::<Result<Vec<_>>>()?;
        if w.iter().any(|v|v.device_address()%8!=(offset*4) as u64) {return Err("NV-b alignment class differs".into());}
        let jobs=(0..16).map(|job|(shape(r,cols,job),&inputs[job],&w[job],&s[job])).collect::<Vec<_>>();
        for count in [1,2,8,16] {
            let output=b.expert_projection_batch(&jobs[..count])?;let values=b.read_vector_values(&output)?;
            super::write(&out.join(format!("r{r}-k{cols}-j{count}-o{offset}.f32")),&raw(&values))?;
            receipt.push_str(&format!("NV_B_CASE rows={r} cols={cols} jobs={count} offset_words={offset} bytes={}\n",values.len()*4));
        }
    }Ok(())
}
fn counterexample(b:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let input=b.upload_vector(&[1.0;85])?;let w=upload(b,&[0x22;80],0)?;let s=upload(b,&[0x38;10],0)?;
    let shape=Shape {global:1.0,activation:1.0,..shape(2,80,0)};
    let one=b.expert_operation(shape,[Some(&input),None,None,None],Some(&w),Some(&s))?;
    let values=b.read_vector_values(&one)?;if values!=[80.0;2] {return Err("NV-b solo counterexample differs from80".into());}
    super::write(&out.join("counterexample-single.f32"),&raw(&values))?;
    let output=b.expert_projection_batch(&[(shape,&input,&w,&s);3])?;let values=b.read_vector_values(&output)?;
    if values!=[80.0;6] {return Err("NV-b batch counterexample differs from80".into());}
    super::write(&out.join("counterexample-three.f32"),&raw(&values))?;
    receipt.push_str("NV_B_COUNTEREXAMPLE rows=2 cols=80 jobs=1,3 expected=80 exact=true\n");Ok(())
}
fn counts(b:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let mut input=vec![1.0;17];input[16]=1.0;let input=b.upload_vector(&input)?;let w=upload(b,&[0x22;8],0)?;let s=upload(b,&[0x38;1],0)?;
    for count in [3usize,17,257,65536] {
        let jobs=(0..count).map(|job|(Shape {activation:1.0,..shape(1,16,job)},&input,&w,&s)).collect::<Vec<_>>();
        let output=b.expert_projection_batch(&jobs)?;let values=b.read_vector_values(&output)?;
        if values.len()!=count {return Err("NV-b fallback count extent differs".into());}
        super::write(&out.join(format!("fallback-count{count}.f32")),&raw(&values))?;
        receipt.push_str(&format!("NV_B_LEGACY_COUNT rows=1 cols=16 jobs={count} bytes={}\n",values.len()*4));
    }Ok(())
}
#[allow(clippy::possible_missing_else)] // dense one-line guards.
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b|{
        let rows=(1..=257).collect::<Vec<u32>>();
        for cols in COLS {let mut family_rows=rows.clone();if cols==4096 {family_rows.push(1024);}if cols==2048 {family_rows.push(2048);}
            for offset in [0usize,1] {family(b,out,cols,&family_rows,offset,&mut receipt)?;}
            super::write(&out.join(format!("RECEIPT-k{cols}")),receipt.as_bytes())?;
        }
        for offset in [0usize,1] {family(b,out,16,&[258,1023,1025,2047,2049,65535,65537],offset,&mut receipt)?;}
        counterexample(b,out,&mut receipt)?;counts(b,out,&mut receipt)?;
        super::nv_group_rows::all_codes(b,out,&mut receipt)?;Ok(())
    })?;super::write(&out.join("RECEIPT"),receipt.as_bytes())?;println!("NV_B_SWEEP_COMPLETE cases={} all_codes=true",receipt.lines().count());Ok(())
}
