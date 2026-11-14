//! Packet placement preserves raw words and the exchange's finite check.
use spark_gpu_probe::{prompt_ops::{self,Op,Shape},prompt_admission};
fn run(s:Shape,x:&[f32])->Vec<f32> {prompt_ops::reference(s,[x,&[],&[],&[],&[]],&[]).unwrap()}
fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|v.to_bits()).collect()}
fn transpose(x:&[f32],rows:usize,cols:usize)->Vec<f32> {
    let mut s=Shape::new(Op::Transpose,1).unwrap();s.rows=rows as u32;s.cols=cols as u32;run(s,x)
}
#[test]
fn rank_packets_equal_the_old_two_transposes_for_every_group_size() {
    for tokens in 1..=1024 {for parts in [1,2,3,4,6] {for width in [1,3,4,12] {
        let s=Shape::rank_rows(tokens,width,parts,false).unwrap();
        let input=(0..tokens*width*parts).map(|i|f32::from_bits([0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,0x7fc01234,i as u32][i%8])).collect::<Vec<_>>();
        let old=input.chunks_exact(tokens*width).flat_map(|packet|transpose(packet,tokens,width)).collect::<Vec<_>>();
        let expected=transpose(&old,parts*width,tokens);
        assert_eq!(bits(&run(s,&input)),bits(&expected),"tokens={tokens} parts={parts} width={width}");
        let admitted=width.is_multiple_of(4);
        assert_eq!(prompt_admission::permute_admitted(s,16,32),admitted);
        if admitted {assert_eq!(prompt_admission::permute_blocks(s) as usize,tokens*parts);}
        assert!(!prompt_admission::permute_admitted(s,4,32));
    }}}
}
#[test]
fn vector_copy_rows_cover_every_word_once_including_the_tail() {
    for tokens in [1,3,16,17,257,1024] {for parts in [1,2,3,4,6] {for width in [4,12,128,144,1024,2048,4096,8192] {
        let s=Shape::rank_rows(tokens,width,parts,true).unwrap();let mut seen=vec![false;tokens*parts];
        for row in 0..tokens*parts {let source=prompt_ops::rank_row(s,row);assert!(!seen[source]);seen[source]=true;
            assert_eq!((source/tokens,source%tokens),(row%parts,row/parts));}
        let mut copied=vec![0u8;width];let threads=prompt_admission::MOVE_THREADS as usize;
        for tid in 0..threads {let mut base=0;while base<width/4 {for j in 0..4 {let v=base+j*threads+tid;
            if v<width/4 {for word in &mut copied[v*4..v*4+4] {*word+=1;}}
        }base+=4*threads;}}
        assert!(copied.iter().all(|&n|n==1));
    }}}
}
#[test]
fn invalid_extents_and_nonfinite_exchange_words_are_refused() {
    for (m,w,p) in [(0,1,1),(1,0,1),(1,1,0),(usize::MAX,1,1),(1,usize::MAX,1),(1,1,usize::MAX),(65537,1,1),(65536,65536,1)] {
        assert!(Shape::rank_rows(m,w,p,true).is_none());
    }
    let s=Shape::rank_rows(3,4,2,true).unwrap();
    assert!(prompt_ops::reference(s,[&[0.0;23],&[],&[],&[],&[]],&[]).is_err());
    for word in [0x7f800000,0xff800000,0x7fc01234] {for at in 0..24 {
        let mut x=vec![0.0;24];x[at]=f32::from_bits(word);
        assert!(std::panic::catch_unwind(||run(s,&x)).is_err());
    }}
    let x=[0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff].repeat(4).into_iter().map(f32::from_bits).collect::<Vec<_>>();
    assert_eq!(bits(&run(s,&x)),bits(&run(Shape {first:0,..s},&x)));
}
