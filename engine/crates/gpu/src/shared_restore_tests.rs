//! The rejected broad variant must fail these restored three/four-row contracts. The block-FP8 rows of the contract were moved once since, by
//! SHARE-B, to a NEW entry with its own byte proof and launch timing (see the note in the test).
use super::*;
#[test]
fn three_and_four_rows_keep_the_base_entries_grids_and_byte_order() {
    for m in [3,4] {
        for (rows,cols) in [(2048,8192),(2048,12288),(4096,4096),(6144,4096)] {
            assert_eq!(mx(rows,cols,m,256,512,768),Mx::Small("mx_rows_small4",4));
        }
        // SHARE-B (2026-11-16): the timed single block-FP8 tables take the new entry `fp8_rows_share` at 3 and 4 rows (one copy and one decode
        // of a weight for all rows; launch times in `fp8_small::share_wins`, same bytes in `kernel-bench --fp8-small {OLD} {X} share`). The tables
        // that were not timed (ONE part of 1,024 x 4,096) or where it loses (256 x 4,096) keep the base entries and grids.
        for (rows,cols,want) in [(1024,4096,"fp8_rows_few"),(2048,2048,"fp8_rows_share"),
            (2048,16384,"fp8_rows_share"),(8192,1536,"fp8_rows_share"),(768,4096,"fp8_rows_share"),(256,4096,"fp8_rows_small")] {
            let parts=[fp8_rows::Part {rows,cols,tokens:m,block_rows:128,block_cols:128,weight:256,scales:512}];
            let got=group(&Group {nv:&[],fp8:&parts,all_nv:false,all_fp8:true,words:1,cumulative:true,shared_rows:false,input:768});
            let expected=match want {
                "fp8_rows_few"=>(want,rows.div_ceil(16) as u32,128,0),
                "fp8_rows_share"=>(want,rows.div_ceil(16) as u32,192,30912),
                _=>(want,(rows.div_ceil(16)*m) as u32,160,30720),
            };
            assert_eq!(got,Some(expected),"{rows}x{cols} m={m}");
        }
        // The same entry/geometry means the same device body. Also retain the host
        // word contract: every row equals its solo computation, even after its neighbor changes.
        let (rows,cols)=(7,128);let weights=(0..rows*cols).map(|i|((i*13%31) as f32-15.0)/64.0).collect::<Vec<_>>();
        let mut input=(0..m*cols).map(|i|((i*7%17) as f32-8.0)/32.0).collect::<Vec<_>>();
        let mut shape=Shape::new(Op::ProjectRows,m).unwrap();shape.rows=rows as u32;shape.cols=cols as u32;
        let solo=crate::causal_ops::Shape {op:crate::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
        for _ in 0..2 {
            let all=crate::prompt_ops::reference(shape,[&input,&[],&[],&[],&[]],&weights).unwrap();
            for t in 0..m {let one=crate::causal_ops::reference(solo,[&input[t*cols..(t+1)*cols],&[],&[],&[]],&weights).unwrap();
                for r in 0..rows {assert_eq!(all[r*m+t].to_bits(),one[r].to_bits());}}
            input[cols..2*cols].fill(-11.25);
        }
    }
}
