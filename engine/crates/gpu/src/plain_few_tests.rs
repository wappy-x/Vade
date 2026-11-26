use super::*;
use crate::next_params::{ResidentProjectionBatch,MxProjection};
#[test]
fn few_row_fp32_fold_preserves_each_rows_bytes_and_refuses_other_batches() {
    for m in 2..=4u32 {
        let (rows,cols)=(144usize,4096usize);let mut batch=ResidentProjectionBatch {plain:(1<<m)-1,..Default::default()};batch.layer.count=m;
        for t in 0..m as usize {batch.layer.projections[t]=MxProjection {weights:256,scales:4,input:512+(t*cols*4) as u64,output:1_000_000+(t*rows*4) as u64,rows:rows as u32,cols:cols as u32,tokens:1,..Default::default()};}
        assert_eq!(batch_fold(&batch),m==2,"three/four rows retain the base batch and its byte contract");
        let weights=(0..rows*cols).map(|i|if i%4==0 {4096.0} else {(i%7) as f32/32.0}).collect::<Vec<_>>();
        let mut inputs=(0..m as usize*cols).map(|i|((i*17%31) as f32-15.0)/16.0).collect::<Vec<_>>();
        let mut shape=crate::prompt_ops::Shape::new(crate::prompt_ops::Op::ProjectRows,m as usize).unwrap();shape.rows=rows as u32;shape.cols=cols as u32;
        let row=crate::causal_ops::Shape {op:crate::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
        for _ in 0..2 {
            let together=crate::prompt_ops::reference(shape,[&inputs,&[],&[],&[],&[]],&weights).unwrap();
            for t in 0..m as usize {let alone=crate::causal_ops::reference(row,[&inputs[t*cols..(t+1)*cols],&[],&[],&[]],&weights).unwrap();
                for r in 0..rows {assert_eq!(alone[r].to_bits(),together[r*m as usize+t].to_bits());}}
            inputs[cols..2*cols].fill(-3.25);
        }
        let mut other=batch;other.round=1;assert!(!batch_fold(&other));
        other=batch;other.layer.projections[1].weights+=16;assert!(!batch_fold(&other));
        other=batch;other.layer.projections[1].input+=4;assert!(!batch_fold(&other));
        for count in [0,1,5,6,8,9] {other=batch;other.layer.count=count;assert!(!batch_fold(&other));}
    }
}
