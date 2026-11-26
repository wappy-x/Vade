//! Device checks of the production fold-row launch boundary. Run with --ignored on CUDA.
#![cfg(target_os="linux")]
use spark_gpu_probe::layer_ptx;
use spark_gpu_probe::driver::{with_backend,ResidentProjection};

#[test]
#[ignore = "requires CUDA and the compiled layer image"]
fn fold_rows_refuses_wrong_bounds_before_writing_device_output() {
    with_backend(0,layer_ptx().unwrap(),64<<20,|b| {
        let weight=b.upload_vector(&vec![0.5;4*1024])?.into_resident_shared()?;
        let input=b.upload_vector(&vec![1.;2*1024])?.into_resident_shared()?;
        let short_weight=weight.resident_range(0..4*1024-1)?;
        let short_input=input.resident_range(0..2*1024-1)?;
        let unaligned=input.resident_range(1..2*1024)?;
        let job=|w,i,rows,cols|ResidentProjection::Plain {rows,cols,weight:w,input:i};
        let mut output=b.upload_vector(&[37.;8])?;
        let mut count=0;
        for (jobs,tokens,expected) in [
            (vec![],2,"count differs"),
            ((0..10).map(|_|job(&weight,&input,4,1024)).collect(),2,"count differs"),
            (vec![job(&weight,&input,4,1024)],1,"count differs"),
            (vec![job(&weight,&input,4,1024)],9,"count differs"),
            (vec![job(&short_weight,&input,4,1024)],2,"vector"),
            (vec![job(&weight,&short_input,4,1024)],2,"vector"),
            (vec![job(&weight,&unaligned,4,1024)],2,"envelope"),
            (vec![job(&weight,&input,4,1023)],2,"vector"),
            (vec![job(&weight,&input,0,1024)],2,"vector"),
            (vec![job(&weight,&input,4,1024),ResidentProjection::PlainBf16 {rows:4,cols:1024,weight:&weight,input:&input}],2,"one precision"),
        ] {
            let error=b.resident_projection_fold_rows(&jobs,&mut output,false,tokens).unwrap_err();
            assert!(error.contains(expected),"expected {expected}: {error}");
            assert_eq!(b.read_vector_values(&output)?,vec![37.;8]);count+=1;
        }
        let mut short=b.upload_vector(&[37.;7])?;
        assert!(b.resident_projection_fold_rows(&[job(&weight,&input,4,1024)],&mut short,false,2).unwrap_err().contains("output is short"));
        assert_eq!(b.read_vector_values(&short)?,vec![37.;7]);count+=1;
        // An admitted input span aliases the output at its first eight words.
        let mut alias=input.resident_range(0..8)?;
        assert!(b.resident_projection_fold_rows(&[job(&weight,&input,4,1024)],&mut alias,false,2).unwrap_err().contains("aliases"));count+=1;
        for round in [false,true] {
            b.resident_projection_fold_rows(&[job(&weight,&input,4,1024)],&mut output,round,2)?;
            assert_eq!(b.read_vector_values(&output)?,vec![512.;8]);
        }
        let packed=b.upload_vector(&vec![f32::from_bits(0x3f00_3f00);2*1024])?;
        b.resident_projection_fold_rows(&[ResidentProjection::PlainBf16 {rows:4,cols:1024,weight:&packed,input:&input}],&mut output,false,2)?;
        assert_eq!(b.read_vector_values(&output)?,vec![512.;8]);
        println!("FOLD ROWS BOUNDARY PASS refusals={count} valid_controls=3 output_guard=unchanged");Ok(())
    }).unwrap();
}
