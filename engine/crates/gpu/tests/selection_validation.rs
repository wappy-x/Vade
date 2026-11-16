//! Compact selection checks retain per-token causal bounds and ignore padding.
use spark_gpu_probe::prompt_ops::{Op,Shape,reference,selection_validation_status};

#[test]
fn compact_status_rejects_bad_live_words_and_keeps_unused_padding() {
    for context in [1usize,7,2048,8192,32768,262144] {
        for slots in [1usize,7,2051] {
            let count=context.min(slots);let mut row=vec![f32::NAN;slots+1];
            for (i,v) in row[..count].iter_mut().enumerate() {*v=(context-count+i) as f32;}
            row[slots]=count as f32;assert_eq!(selection_validation_status(&row,context),0);
            for bad in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,-1.,0.5,context as f32] {
                let mut wrong=row.clone();wrong[0]=bad;
                assert_eq!(selection_validation_status(&wrong,context),1,"context={context} slots={slots} id={bad}");
            }
            for bad in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,-1.,0.5,(count+1) as f32] {
                let mut wrong=row.clone();wrong[slots]=bad;
                assert_eq!(selection_validation_status(&wrong,context),2,"context={context} slots={slots} count={bad}");
            }
            row[slots]= -0.;assert_eq!(selection_validation_status(&row,context),0);
        }
    }
}

#[test]
fn validation_uses_each_tokens_context_and_writes_one_status_each() {
    let mut s=Shape::new(Op::CheckSelectionRows,5).unwrap();s.first=2;s.slots=3;
    let rows=[0.,1.,f32::NAN,2., 0.,2.,f32::NAN,2., 0.,4.,f32::NAN,2., 0.,1.,2.,4., f32::NAN,f32::NAN,f32::NAN,0.];
    assert_eq!(reference(s,[&rows,&[],&[],&[],&[]],&[]).unwrap(),[0.,0.,1.,2.,0.]);
    assert!(reference(s,[&rows[..19],&[],&[],&[],&[]],&[]).is_err());
    assert_eq!(Op::CheckSelectionRows as u32,27,"kernel bench ABI");
    for (first,slots,tokens) in [(0,3,5),(2,0,5),(u32::MAX,3,5),(2,3,65537)] {
        s.first=first;s.slots=slots;s.tokens=tokens;assert!(s.lengths().is_none());
    }
}
