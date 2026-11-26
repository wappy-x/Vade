//! The rejected broad variant must fail these restored three/four-row contracts.
use super::*;
#[test]
fn three_and_four_rows_keep_the_base_entries_grids_and_byte_order() {
    for m in [3,4] {
        for (rows,cols) in [(2048,8192),(2048,12288),(4096,4096),(6144,4096)] {
            assert_eq!(mx(rows,cols,m,256,512,768),Mx::Small("mx_rows_small4",4));
        }
        for (rows,cols,want) in [(1024,4096,"fp8_rows_few"),(2048,2048,"fp8_rows_few"),
            (2048,16384,"fp8_rows_multi"),(8192,1536,"fp8_rows_multi"),(768,4096,"fp8_rows_small"),(256,4096,"fp8_rows_small")] {
            let parts=[fp8_rows::Part {rows,cols,tokens:m,block_rows:128,block_cols:128,weight:256,scales:512}];
            let got=group(&Group {nv:&[],fp8:&parts,all_nv:false,all_fp8:true,words:1,cumulative:true,shared_rows:false,input:768});
            // TENSOR-CORE REBASE (block FP8): the three entries keep their names (each is a recorded launch kind) and run
            // the one row body with one geometry: a block a tile of 16 rows and up to 8 token rows (`fp8_small`).
            let expected=(want,rows.div_ceil(16) as u32,fp8_small::threads(&parts),fp8_small::shared(&parts));
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
