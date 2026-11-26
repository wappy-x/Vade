//! FOLD ROWS on the host: the block schedule of `plain_warp::rows_fold_rows`, walked with the kernel's own index functions.
use super::*;
/// The kernel's block on the CPU. Slots start as NaN (the bench's poisoned entry does the same on the device), so a read of
/// a word that no producer wrote shows. `word` is the slot word of (layout, source row, block weight row, column): the kernel
/// uses `fold_rows_word`; a test may pass a wrong one on purpose.
fn twin(weights:&[f32],input:&[f32],rows:usize,cols:usize,tokens:usize,word:fn(usize,usize,usize,usize)->usize)->Vec<f32> {
    let span=fold_token_span(tokens);
    if span < tokens {
        return input.chunks(span*cols).flat_map(|tile|twin(weights,tile,rows,cols,tile.len()/cols,word)).collect();
    }
    let layout=fold_rows_layout(tokens);let ch=fold_rows_chunk(layout);let slot_words=FOLD_ROWS*layout*fold_rows_pitch(layout);
    assert!(2*slot_words*4<=FOLD_ROWS_BYTES);
    let mut output=vec![f32::NAN;tokens*rows];
    for first_row in (0..rows).step_by(FOLD_ROWS) {
        let mut sums=[0.0f32;32];
        for c in 0..cols.div_ceil(ch) {
            let (first,span)=(c*ch,(cols-c*ch).min(ch));let mut slot=vec![f32::NAN.to_bits();slot_words];
            // Producers: each loads its weight vectors once and makes one product vector a source row.
            for p in 0..FOLD_THREADS-32 {for j in 0..8 {
                let (_,t,r,q)=fold_rows_vector(layout,p,j);
                if q*4<span && first_row+r<rows && t<tokens {for e in 0..4 {
                    let product=weights[(first_row+r)*cols+first+q*4+e]*input[t*cols+first+q*4+e];
                    slot[word(layout,t,r,q*4+e)]=product.to_bits();
                }}
            }}
            // The folding warp: lane `t * FOLD_ROWS + r` adds its own chain in column order.
            for (lane,sum) in sums.iter_mut().enumerate().take(FOLD_ROWS*layout) {
                let (t,r)=(lane/FOLD_ROWS,lane%FOLD_ROWS);
                if t<tokens && first_row+r<rows {for col in 0..span {*sum+=f32::from_bits(slot[fold_rows_word(layout,t,r,col)]);}}
            }
        }
        for (lane,sum) in sums.iter().enumerate().take(FOLD_ROWS*layout) {
            let (t,r)=(lane/FOLD_ROWS,lane%FOLD_ROWS);if t<tokens && first_row+r<rows {output[t*rows+first_row+r] = *sum;}
        }
    }
    output
}
fn alone(weights:&[f32],input:&[f32],rows:usize,cols:usize)->Vec<f32> {
    let row=crate::causal_ops::Shape {op:crate::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
    crate::causal_ops::reference(row,[input,&[],&[],&[]],weights).unwrap()
}
fn operands(rows:usize,cols:usize,tokens:usize,seed:usize)->(Vec<f32>,Vec<f32>) {
    // Mixed magnitudes and signs, so a changed add order or a shared accumulator changes bits.
    let weights=(0..rows*cols).map(|i|if (i+seed)%5==0 {4096.0} else {((i*7+seed*3)%23) as f32/32.0-0.34375}).collect::<Vec<_>>();
    let input=(0..tokens*cols).map(|i|(((i*17+seed)%31) as f32-15.0)/16.0+if i%97==0 {1.0e-3} else {0.0}).collect::<Vec<_>>();
    (weights,input)
}
#[test]
fn every_slot_word_has_one_producer_and_one_reader_for_every_layout() {
    for layout in [2usize,4,8] {
        let (ch,pitch)=(fold_rows_chunk(layout),fold_rows_pitch(layout));let quads=ch/4;
        assert_eq!((ch*layout,pitch),(1024,ch+4));
        assert!(2*FOLD_ROWS*layout*pitch*4<=FOLD_ROWS_BYTES,"layout {layout}: two slots fit the block's shared bytes");
        let mut writers=vec![0u32;FOLD_ROWS*layout*pitch];let mut weight_of=std::collections::BTreeMap::new();
        for p in 0..FOLD_THREADS-32 {for j in 0..8 {
            let (k,t,r,q)=fold_rows_vector(layout,p,j);
            assert!(k<8/layout && t<layout && r<FOLD_ROWS && q<quads,"layout {layout} p {p} j {j}");
            // One loaded weight vector serves all the source rows of its (weight row, vector): the slot k names one (r, q).
            assert_eq!(*weight_of.entry((p,k)).or_insert((r,q)),(r,q));
            for e in 0..4 {writers[fold_rows_word(layout,t,r,q*4+e)]+=1;}
        }}
        for t in 0..layout {for r in 0..FOLD_ROWS {
            // A chain owns one pitch of the slot: every column written once, the pad never, and no other chain's word.
            let base=fold_rows_word(layout,t,r,0);assert_eq!(base,(t*FOLD_ROWS+r)*pitch);
            assert!(writers[base..base+ch].iter().all(|&n|n==1) && writers[base+ch..base+pitch].iter().all(|&n|n==0),"layout {layout} chain {t} {r}");
        }}
        // A warp of producers reads neighbouring vectors of one weight row (coalesced loads).
        for warp in 0..(FOLD_THREADS-32)/32 {for k in 0..8/layout {
            let (_,_,r0,q0)=fold_rows_vector(layout,warp*32,k*layout);
            for lane in 0..32 {assert_eq!(fold_rows_vector(layout,warp*32+lane,k*layout),(k,0,r0,q0+lane));}
        }}
    }
    assert_eq!((fold_rows_layout(2),fold_rows_layout(3),fold_rows_layout(4),fold_rows_layout(5),fold_rows_layout(8)),(2,4,4,8,8));
}
#[test]
fn a_row_in_a_group_of_2_to_16_equals_the_row_alone_and_ignores_its_neighbors() {
    // The model's fold shapes (router gate, narrow BF16 groups), and shapes with a ragged last block and a short last chunk.
    for (seed,(rows,cols)) in [(144usize,4096usize),(128,4096),(64,4096),(32,4096),(16,4096),(6,1024),(5,1056),(1,8192),(9,2080)].into_iter().enumerate() {
        for tokens in 2..=FOLD_ROWS_MAX_TOKENS {
            if rows>32 && !matches!(tokens,2|3|4|8) {continue;}
            assert!(fold_rows_admitted(rows,cols,256,512,tokens));
            let (weights,mut input)=operands(rows,cols,tokens,seed);
            let together=twin(&weights,&input,rows,cols,tokens,fold_rows_word);
            for t in 0..tokens {
                let one=alone(&weights,&input[t*cols..(t+1)*cols],rows,cols);
                for r in 0..rows {assert_eq!(together[t*rows+r].to_bits(),one[r].to_bits(),"{rows}x{cols} tokens {tokens} token {t} row {r}");}
            }
            // Change one neighbor: every other row keeps its bits, the changed row moves.
            let changed=tokens/2;input[changed*cols..(changed+1)*cols].iter_mut().enumerate().for_each(|(i,v)|*v=-3.25+(i%13) as f32/8.0);
            let after=twin(&weights,&input,rows,cols,tokens,fold_rows_word);
            for t in 0..tokens {
                let same=(0..rows).all(|r|after[t*rows+r].to_bits()==together[t*rows+r].to_bits());
                assert_eq!(same,t!=changed,"{rows}x{cols} tokens {tokens}: token {t} after token {changed} changed");
            }
        }
    }
}
#[test]
fn wrong_on_purpose_forms_fail_the_same_check() {
    let (rows,cols,tokens)=(144,4096,4);let (weights,input)=operands(rows,cols,tokens,3);
    let good=twin(&weights,&input,rows,cols,tokens,fold_rows_word);
    let equal=|got:&[f32]|(0..tokens).all(|t|{let one=alone(&weights,&input[t*cols..(t+1)*cols],rows,cols);(0..rows).all(|r|got[t*rows+r].to_bits()==one[r].to_bits())});
    assert!(equal(&good));
    // Two source rows share one slot pitch (a shared scratch buffer): the check must fail.
    let shared=twin(&weights,&input,rows,cols,tokens,|layout,t,r,col|fold_rows_word(layout,t%2,r,col));
    assert!(!equal(&shared),"a shared slot pitch must change bits");
    // A chain read in another column order: the check must fail.
    let reordered=twin(&weights,&input,rows,cols,tokens,|layout,t,r,col|fold_rows_word(layout,t,r,col^1));
    assert!(!equal(&reordered),"a changed chain order must change bits");
}
#[test]
fn the_shape_rule_takes_the_fold_shapes_at_2_to_16_rows_and_nothing_else() {
    for (rows,cols) in [(144,4096),(288,4096),(128,4096),(64,4096),(32,4096),(16,4096),(1,1024),(512,8192)] {
        for tokens in 0..=17 {assert_eq!(fold_rows_shape(rows,cols,tokens),(2..=16).contains(&tokens),"{rows}x{cols} tokens {tokens}");}
        assert_eq!(plain_geometry(rows,cols).1 as usize,FOLD_THREADS);
        assert!(fold_rows_admitted(rows,cols,256,512,4) && !fold_rows_admitted(rows,cols,264,512,4) && !fold_rows_admitted(rows,cols,256,516,4));
    }
    // The other plain shapes of the step: long tables with short rows, and anything past the envelope.
    for (rows,cols) in [(4096,128),(8192,128),(2048,1536),(4096,1536),(16384,512),(32768,512),(513,4096),(144,1000),(0,4096),(144,8224),(144,16384)] {
        for tokens in 2..=8 {assert!(!fold_rows_shape(rows,cols,tokens),"{rows}x{cols} tokens {tokens}");}
    }
}
#[test]
fn a_fold_rows_batch_keeps_the_one_row_fold_grid_and_refuses_other_row_counts() {
    use crate::next_params::{MxProjection,ResidentProjectionBatch};
    let p=|rows:u32,tokens:u32|MxProjection {weights:256,scales:4,input:512,output:1024,rows,cols:4096,tokens,input_bf16:0,round:0,mode:0};
    let mut batch=ResidentProjectionBatch::default();
    for (rows,tokens) in [(144,2),(128,8),(5,3)] {batch.push_fold_rows(p(rows,tokens)).unwrap();}
    assert_eq!((batch.layer.count,batch.plain,&batch.layer.ends[..4]),(3,0b111,&[0,36,68,70][..]));
    // The grid does not grow with the source rows: a block runs all of them.
    for job in 0..3 {let d=batch.layer.projections[job];assert_eq!(batch.layer.ends[job+1]-batch.layer.ends[job],plain_geometry(d.rows as usize,d.cols as usize).0);}
    for tokens in 9..=16 {let mut tiled=ResidentProjectionBatch::default();tiled.push_fold_rows(p(5,tokens)).unwrap();assert_eq!(tiled.layer.ends[1],4);}
    for tokens in [0,1,17,32] {assert!(ResidentProjectionBatch::default().push_fold_rows(p(144,tokens)).is_err(),"tokens {tokens}");}
    let mut rounded=p(144,2);rounded.round=1;assert!(ResidentProjectionBatch::default().push_fold_rows(rounded).is_err());
}
