use super::*;
use crate::{prompt_ops::reference,causal_ops::{Shape as One,Op as Scalar,reference as scalar}};
fn data(n:usize,seed:usize)->Vec<f32> {(0..n).map(|i|crate::causal_ops::round(((i*37+seed)%109) as f32/64.0-0.75)).collect()}
fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
#[test]
fn prompt_groups_match_scalar_planes_with_empty_duplicate_and_reordered_selections() {
    for m in [1usize,3,5,17] {for h in [1usize,3,4,5] {for k in [2usize,10,32,64] {
        let (d,v,r,c,n)=(17usize,9usize,31usize,13usize,19usize);
        let q=data(m*h*d,11);let latent=data(r*k,23);let mut selected=vec![-1.0;m*(n+1)];
        for token in 0..m {let count=(token*7)%(c+1);for j in 0..count {selected[token*(n+1)+j]=((j*17+token)%r) as f32;}if count>1 {selected[token*(n+1)+1]=selected[token*(n+1)];}selected[token*(n+1)+n]=count as f32;}
        let wide=data(h*(d+v)*k,29);let weight=wide.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect::<Vec<_>>();
        let s=Shape {op:Op::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:r as u32,cols:k as u32,first:c as u32,slots:n as u32,..Shape::new(Op::AbsorbQuery,m).unwrap()};
        let a=reference(s,[&q,&[],&[],&[],&[]],&weight).unwrap();
        let b=reference(Shape {op:Op::AbsorbScores,..s},[&a,&latent,&selected,&[],&[]],&[]).unwrap();
        let p=reference(Shape {op:Op::AbsorbSoftmax,cols:c as u32,..s},[&b,&selected,&[],&[],&[]],&[]).unwrap();
        let u=reference(Shape {op:Op::AbsorbValues,..s},[&p,&latent,&selected,&[],&[]],&[]).unwrap();
        let out=reference(Shape {op:Op::AbsorbProject,..s},[&u,&selected,&[],&[],&[]],&weight).unwrap();
        for token in 0..m {
            let count=selected[token*(n+1)+n] as usize;let mut taken=vec![0.0;c*k];
            for j in 0..count {let id=selected[token*(n+1)+j] as usize;taken[j*k..(j+1)*k].copy_from_slice(&latent[id*k..(id+1)*k]);}
            let s=One {op:Scalar::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:c as u32,cols:k as u32,epsilon:0.0,bound:0.0};
            let aa=scalar(s,[&q[token*h*d..(token+1)*h*d],&[],&[],&[]],&weight).unwrap();
            let bb=scalar(One {op:Scalar::AbsorbScores,..s},[&aa,&taken,&[count as f32],&[]],&[]).unwrap();
            let pp=scalar(One {op:Scalar::Softmax,taps:0,cols:0,..s},[&bb,&[count as f32],&[],&[]],&[]).unwrap();
            let uu=scalar(One {op:Scalar::AbsorbValues,..s},[&pp,&taken,&[count as f32],&[]],&[]).unwrap();
            let oo=scalar(One {op:Scalar::AbsorbProject,..s},[&uu,&[],&[],&[]],&weight).unwrap();
            for (plane,(whole,one)) in [(&a,&aa),(&b,&bb),(&p,&pp),(&u,&uu),(&out,&oo)].into_iter().enumerate() {
                assert_eq!(bits(&whole[token*one.len()..(token+1)*one.len()]),bits(one),"m={m} h={h} k={k} token={token} plane={plane}");
            }
        }
    }}}
}
/// The grouped kernel finds its words from the block index alone (`absorb_tc::Form::block`). Walk the same arithmetic
/// on the host: every output unit of each fold step has exactly one owner, and no block lies outside the launch grid.
#[test]
fn the_grid_owns_every_output_unit_exactly_once() {
    use crate::absorb_tc::{Form,WARPS,TILES,THREADS,c_place};
    for (m,h,k,v,c) in [(1u32,1u32,2u32,9u32,13u32),(3,5,34,9,1),(17,4,64,33,40),(33,32,512,256,70),(256,2,32,32,33),(2,33,130,131,300)] {
        for op in [Op::AbsorbQuery,Op::AbsorbScores,Op::AbsorbValues,Op::AbsorbProject] {
            let s=Shape {op,heads:h,dim:17,taps:v,rows:c.max(31),cols:k,first:c,slots:c+6,..Shape::new(op,m as usize).unwrap()};
            let (name,blocks,threads)=launch(s).unwrap();assert_eq!((name,threads),("latent_absorption_tile",THREADS));
            let f=Form::rows(s).unwrap();assert_eq!(blocks as usize,f.blocks());
            let mut owners=vec![0u8;f.outer()*f.m()*f.n()];
            for block in 0..f.blocks() {let (outer,m0,n0)=f.block(block);assert!(outer<f.outer());
                for warp in 0..WARPS {for tile in 0..TILES {for lane in 0..32 {for reg in 0..4 {let (mm,nn)=c_place(lane,reg);let (row,col)=(m0+warp*16+mm,n0+tile*8+nn);
                    if row<f.m() && col<f.n() {owners[(outer*f.m()+row)*f.n()+col]+=1;}}}}}
            }
            assert!(owners.iter().all(|&n|n==1),"{op:?} m={m} h={h} k={k}");
        }
    }
}
/// The third form takes the model's shape from 17 token rows with 16-byte aligned operands, and nothing else: its
/// kernels trap on any other launch, so the host must decline first.
#[test]
fn the_third_form_takes_the_model_shape_only() {
    let model=|op,tokens:usize|Shape {op,heads:32,dim:256,taps:256,rows:8192,cols:512,first:2051,slots:2051,..Shape::new(op,tokens).unwrap()};
    let a=[0x1000u64,0x2000,0x3000,4,4];
    for (op,name,blocks) in [(Op::AbsorbQuery,"latent_weights3",32*32),(Op::AbsorbScores,"latent_scores3",1024),(Op::AbsorbValues,"latent_values3",4096),(Op::AbsorbProject,"latent_weights3_panel",32*32)] {
        assert_eq!(third(model(op,1024),a,0x4000,0x5000),Some((name,blocks,THIRD_THREADS)),"{op:?}");
        // Whole blocks of 32 tokens for the two weight operations, a block a token for scores, four for values.
        let partial=third(model(op,33),a,0x4000,0x5000).unwrap();
        assert_eq!(partial.1,match op {Op::AbsorbScores=>33,Op::AbsorbValues=>132,_=>64},"{op:?}");
        assert_eq!(third(model(op,THIRD_MIN_TOKENS),a,0x4000,0x5000).map(|t|t.0),Some(name),"{op:?}");
        assert_eq!(third(model(op,THIRD_MIN_TOKENS-1),a,0x4000,0x5000),None,"{op:?}");
        // Another head count, latent width or key width: the first form.
        assert_eq!(third(Shape {heads:16,..model(op,1024)},a,0x4000,0x5000),None,"{op:?}");
        assert_eq!(third(Shape {cols:256,..model(op,1024)},a,0x4000,0x5000),None,"{op:?}");
        // The first form still takes every launch the third takes.
        assert_eq!(launch(model(op,1024)).unwrap().0,"latent_absorption_tile");
    }
    assert_eq!(third(Shape {dim:128,..model(Op::AbsorbQuery,1024)},a,0x4000,0x5000),None);
    assert_eq!(third(Shape {taps:250,..model(Op::AbsorbProject,1024)},a,0x4000,0x5000),None);
    assert_eq!(third(model(Op::AbsorbSoftmax,1024),a,0x4000,0x5000),None);
    // Alignment: each kernel's own operands.
    assert_eq!(third(model(Op::AbsorbScores,1024),[0x1004,0x2000,0x3000,4,4],0x4000,0x5000),None);
    assert_eq!(third(model(Op::AbsorbScores,1024),[0x1000,0x2008,0x3000,4,4],0x4000,0x5000),None);
    assert_eq!(third(model(Op::AbsorbValues,1024),[0x1004,0x2000,0x3000,4,4],0x4000,0x5000).map(|t|t.0),Some("latent_values3"));
    assert_eq!(third(model(Op::AbsorbValues,1024),a,0x4000,0x5004),None);
    assert_eq!(third(model(Op::AbsorbQuery,1024),a,0x4002,0x5000),None);
    assert_eq!(third(model(Op::AbsorbProject,1024),[0x1008,0x2000,0x3000,4,4],0x4000,0x5000),None);
    // A cache too long for a 32-bit byte offset.
    assert_eq!(third(Shape {rows:1<<21,..model(Op::AbsorbValues,1024)},a,0x4000,0x5000),None);
}
/// Packed halves are the existing high-half operand, including nonzero low halves in the unpacked source. Counts
/// include empty, short and full selections. This checks the CPU word and the new ABI's actual half-sized extent.
#[test]
fn packed_cache_has_the_unpacked_word_and_rejects_an_incomplete_cache() {
    use crate::prompt_ops::reference;
    for m in [1usize,5,17,33] {
        let (h,k,c,r)=(3usize,32usize,17usize,31usize);
        let latent:Vec<f32>=(0..r*k).map(|i|f32::from_bits(if i%11==0 {0x8000_1234} else {(0x3e80_1234+(i%127) as u32*65536) | ((i%2) as u32)<<31})).collect();
        let packed:Vec<f32>=latent.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect();
        let mut selected=vec![-1.0;m*(c+1)];
        for t in 0..m {
            let count=[0,1,16,17][t%4];
            for j in 0..count {selected[t*(c+1)+j]=((t*7+j*3)%r) as f32;}
            selected[t*(c+1)+c]=count as f32;
        }
        for (old,new,width) in [(Op::AbsorbScores,Op::AbsorbScoresPacked,k),(Op::AbsorbValues,Op::AbsorbValuesPacked,c)] {
            let x:Vec<f32>=(0..m*h*width).map(|i|((i*29%97) as f32-48.0)/64.0).collect();
            let s=Shape {heads:h as u32,dim:16,taps:16,rows:r as u32,cols:k as u32,first:c as u32,slots:c as u32,..Shape::new(old,m).unwrap()};
            let a=reference(s,[&x,&latent,&selected,&[],&[]],&[]).unwrap();
            let b=reference(Shape {op:new,..s},[&x,&packed,&selected,&[],&[]],&[]).unwrap();
            assert_eq!(bits(&a),bits(&b),"{new:?} tokens={m}");
            assert!(reference(Shape {op:new,..s},[&x,&packed[..packed.len()-1],&selected,&[],&[]],&[]).is_err());
        }
    }
}
#[test]
fn packed_launches_are_checked_and_never_fall_back_to_an_unpacked_kernel() {
    let model=|op,tokens|Shape {heads:32,dim:256,taps:256,rows:8192,cols:512,first:2051,slots:2051,..Shape::new(op,tokens).unwrap()};
    let a=[0x1000,0x2000,0x3000,4,4];
    for (op,name,threads) in [(Op::AbsorbScoresPacked,"latent_scores3_packed",384),(Op::AbsorbValuesPacked,"latent_values3_full_park",256)] {
        let s=model(op,1024);
        assert_eq!(s.lengths().unwrap().0[1],8192*512/2);
        assert_eq!(third(s,a,4,0x4000),Some((name,1024,threads)));
        assert!(launch(s).is_none());
        assert!(third(model(op,16),a,4,0x4000).is_none());
        assert!(third(Shape {heads:16,..s},a,4,0x4000).is_none());
    }
}
