//! Compare the column schedule with the original complete-head device fold.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{causal_ops::{Op,Shape},driver::with_backend};
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for dim in [1,2,8,128] {let d=dim as usize;let heads=3usize;
            let s=Shape {op:Op::Step,heads:heads as u32,dim,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
            let mut p=vec![0.0;heads*(4*d+1)];let mut state=vec![0.0;heads*d*d];
            for (i,v) in p.iter_mut().enumerate() {*v=(i as i32%17-8) as f32/32.0;}
            for (i,v) in state.iter_mut().enumerate() {*v=if i%7==0 {-0.0} else {(i as i32%29-14) as f32/64.0};}
            let prepared=b.upload_vector(&p)?;
            for repeat in 1..=3 {let prior=b.upload_vector(&state)?;
                let expected=b.causal_operation(s,[Some(&prepared),Some(&prior),None,None],None)?;
                let columns=b.causal_operation(Shape {op:Op::StepColumns,..s},[Some(&prepared),Some(&prior),None,None],None)?;
                let actual=b.causal_operation(Shape {op:Op::StepLayout,..s},[Some(&columns),None,None,None],None)?;
                let expected=b.read_vector_values(&expected)?;let got=b.read_vector_values(&actual)?;
                if got.iter().map(|v|v.to_bits()).collect::<Vec<_>>()!=expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>() {return Err(format!("recurrence columns differ dim={dim} repeat={repeat}"));}
                super::write(&out.join(format!("d{dim}-run{repeat}.f32")),&super::floats(&got)?)?;
                state=(0..heads).flat_map(|h|got[h*(d*d+d)..h*(d*d+d)+d*d].iter().copied()).collect();
                receipt.push_str(&format!("RECURRENCE DEVICE PASS dim={dim} repeat={repeat} state_and_output_bits={}\n",got.len()));
            }
        }Ok(())
    })?;super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
