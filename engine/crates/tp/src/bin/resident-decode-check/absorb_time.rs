//! Kernel kit, plain BF16 format: time of the four fold operations of the absorbed latent attention (query, scores,
//! values, project) through the public calls of the served path, at the model's shape. The prompt form
//! (`Backend::prompt_operation`) at 1, 2, 4, 8, 256 and 1,024 token rows, and the step form (`Backend::causal_operation`)
//! of one decode row. This mode uses only calls that exist before and after the tensor-core switch, so the same file
//! times the old words on the base commit and the new words on the switch commit. It checks no words.
#![forbid(unsafe_code)]
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},prompt_ops::{Op,Shape},causal_ops::{Op as StepOp,Shape as StepShape}};
fn mixed(index:usize,salt:usize)->u32 {
    let mut value=(index as u32).wrapping_add((salt as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
/// A BF16 value near one, either sign.
fn near_one(index:usize,salt:usize)->f32 {let h=mixed(index,salt);f32::from_bits(((h>>16)&0x8000|(124+(h>>8)%6)<<7|h&127)<<16)}
fn plane(len:usize,salt:usize)->Vec<f32> {(0..len).map(|i|near_one(i,salt)).collect()}
/// A probability-like BF16 value in [0, 1).
fn share(index:usize,salt:usize)->f32 {f32::from_bits((((mixed(index,salt)>>8) as f32/16_777_216.0/64.0).to_bits()+0x8000)&0xffff_0000)}
/// The selection of a prompt group as the bounded decoder makes it: token `t` sees `rows - m + t + 1` cache rows and
/// keeps up to 512 pools of 4 rows spread over them, in an order that is not sorted, then the last rows.
pub(super) fn selection(m:usize,rows:usize,slots:usize,first:usize)->Vec<f32> {
    let mut out=vec![0.0f32;m*(slots+1)];
    for t in 0..m {
        let context=rows-m+t+1;let pools=(context/4).min(512);let row=&mut out[t*(slots+1)..(t+1)*(slots+1)];let mut count=0;
        for p in 0..pools {let pool=(p*389+t*7)%pools*(context/4)/pools;
            for j in 0..4 {if count<first {row[count]=(pool*4+j) as f32;count+=1;}}}
        for tail in (pools*4..context).rev().take(3) {if count<first {row[count]=tail as f32;count+=1;}}
        row[slots]=count as f32;
    }
    out
}
fn median(backend:&Backend<'_>,run:&mut dyn FnMut()->Result<()>)->Result<f64> {
    let mut times=Vec::new();
    for repeat in 0..10 {backend.synchronize()?;let started=Instant::now();run()?;backend.synchronize()?;if repeat>=3 {times.push(started.elapsed().as_secs_f64()*1e3);}}
    times.sort_by(|a,b|a.partial_cmp(b).unwrap());Ok(times[times.len()/2])
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("ABSORB_TIME timing=complete_public_call shape=heads32_key256_value256_latent512_slots2051\n");
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        let (h,d,v,k,slots)=(32usize,256usize,256usize,512usize,2051usize);
        let weight=backend.upload_vector(&(0..h*(d+v)*k/2).map(|i|f32::from_bits(near_one(2*i,31).to_bits()>>16|near_one(2*i+1,31).to_bits()&0xffff_0000)).collect::<Vec<_>>())?;
        // The prompt form. The cache holds 8,192 rows: the last groups of an 8,192-token read.
        let rows=8192usize;let first=slots.min(rows);let latent=backend.upload_vector(&plane(rows*k,41))?;
        for m in [1usize,2,4,8,256,1024] {
            let mut s=Shape::new(Op::AbsorbQuery,m).ok_or("absorbed shape")?;
            s.heads=h as u32;s.dim=d as u32;s.taps=v as u32;s.cols=k as u32;s.rows=rows as u32;s.slots=slots as u32;s.first=first as u32;
            let selected=backend.upload_vector(&selection(m,rows,slots,first))?;
            let query=backend.upload_vector(&plane(m*h*d,51))?;let absorbed=backend.upload_vector(&plane(m*h*k,52))?;
            let probability=backend.upload_vector(&(0..m*h*first).map(|i|share(i,53)).collect::<Vec<_>>())?;let mixed_latent=backend.upload_vector(&plane(m*h*k,54))?;
            let mut times=[0f64;5];
            let calls:[(Op,[Option<&DeviceVector<'_>>;5],Option<&DeviceVector<'_>>);5]=[
                (Op::AbsorbQuery,[Some(&query),None,None,None,None],Some(&weight)),
                (Op::AbsorbScores,[Some(&absorbed),Some(&latent),Some(&selected),None,None],None),
                (Op::AbsorbSoftmax,[Some(&probability),Some(&selected),None,None,None],None),
                (Op::AbsorbValues,[Some(&probability),Some(&latent),Some(&selected),None,None],None),
                (Op::AbsorbProject,[Some(&mixed_latent),Some(&selected),None,None,None],Some(&weight))];
            for (at,(op,input,w)) in calls.into_iter().enumerate() {
                let shape=if op==Op::AbsorbSoftmax {Shape {op,cols:first as u32,..s}} else {Shape {op,..s}};
                times[at]=median(backend,&mut ||{drop(backend.prompt_operation(shape,input,w)?);Ok(())})?;
            }
            let line=format!("ABSORB TIME PROMPT tokens={m} cache_rows={rows} query_ms={:.4} scores_ms={:.4} softmax_ms={:.4} values_ms={:.4} project_ms={:.4} four_folds_ms={:.4}\n",
                times[0],times[1],times[2],times[3],times[4],times[0]+times[1]+times[3]+times[4]);
            eprint!("{line}");receipt.push_str(&line);
        }
        // The step form: one decode row at a full selection (2,051 gathered rows), and at a short one.
        for live in [2051usize,256,16] {
            let s=StepShape {op:StepOp::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:live as u32,cols:k as u32,epsilon:0.0,bound:0.0};
            let gathered=backend.upload_vector(&plane(live*k,61))?;let count=backend.upload_vector(&[live as f32])?;
            let query=backend.upload_vector(&plane(h*d,62))?;let absorbed=backend.upload_vector(&plane(h*k,63))?;
            let probability=backend.upload_vector(&(0..h*live).map(|i|share(i,64)).collect::<Vec<_>>())?;let mixed_latent=backend.upload_vector(&plane(h*k,65))?;
            let mut times=[0f64;4];
            let calls:[(StepOp,[Option<&DeviceVector<'_>>;4],Option<&DeviceVector<'_>>);4]=[
                (StepOp::AbsorbQuery,[Some(&query),None,None,None],Some(&weight)),
                (StepOp::AbsorbScores,[Some(&absorbed),Some(&gathered),Some(&count),None],None),
                (StepOp::AbsorbValues,[Some(&probability),Some(&gathered),Some(&count),None],None),
                (StepOp::AbsorbProject,[Some(&mixed_latent),None,None,None],Some(&weight))];
            for (at,(op,input,w)) in calls.into_iter().enumerate() {
                times[at]=median(backend,&mut ||{drop(backend.causal_operation(StepShape {op,..s},input,w)?);Ok(())})?;
            }
            let line=format!("ABSORB TIME STEP rows=1 selected={live} query_ms={:.4} scores_ms={:.4} values_ms={:.4} project_ms={:.4} four_folds_ms={:.4}\n",
                times[0],times[1],times[2],times[3],times.iter().sum::<f64>());
            eprint!("{line}");receipt.push_str(&line);
        }
        Ok(())
    })?;
    receipt.push_str("ABSORB_TIME COMPLETE\n");
    fs::write(out.join("ABSORB_TIME"),receipt).map_err(|error|error.to_string())
}
