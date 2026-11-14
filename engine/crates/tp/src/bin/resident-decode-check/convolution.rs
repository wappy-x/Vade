//! Prove independent channels against complete-plane convolution and history.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{causal_ops::{Op,Shape},driver::with_backend};
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for dim in [1,2,128] {for taps in [1,2,4] {let heads=3;let width=(heads*dim) as usize;let t=taps as usize;
            let s=Shape {op:Op::Conv,heads,dim,taps,rows:0,cols:0,epsilon:0.0,bound:0.0};
            let x=(0..3*width).map(|i|(i as i32%19-9) as f32/8.0).collect::<Vec<_>>();
            let w=(0..3*width*t).map(|i|(i as i32%13-6) as f32/16.0).collect::<Vec<_>>();
            let mut old=(0..3*(t-1)*width).map(|i|if i%7==0 {-0.0} else {(i as i32%17-8) as f32/8.0}).collect::<Vec<_>>();
            let source=b.upload_vector(&x)?;let weight=b.upload_vector(&w)?;
            for repeat in 1..=3 {
                let history=if old.is_empty() {None} else {Some(b.upload_vector(&old)?)};
                let input=[Some(&source),history.as_ref(),None,None];
                let expected=b.causal_operation(s,input,Some(&weight))?;
                let channels=b.causal_operation(Shape {op:Op::ConvChannels,..s},input,Some(&weight))?;
                let result=b.causal_operation(Shape {op:Op::ConvLayout,..s},[Some(&channels),None,None,None],None)?;
                let expected=b.read_vector_values(&expected)?;let got=b.read_vector_values(&result)?;
                if got.iter().map(|v|v.to_bits()).collect::<Vec<_>>()!=expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>() {return Err(format!("convolution channels differ dim={dim} taps={taps} repeat={repeat}"));}
                super::write(&out.join(format!("d{dim}-t{taps}-run{repeat}.f32")),&super::floats(&got)?)?;
                old=(0..3).flat_map(|p|got[p*t*width+width..(p+1)*t*width].iter().copied()).collect();
                receipt.push_str(&format!("CONVOLUTION DEVICE PASS dim={dim} taps={taps} repeat={repeat} output_and_history_bits={}\n",got.len()));
            }
        }}Ok(())
    })?;super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
