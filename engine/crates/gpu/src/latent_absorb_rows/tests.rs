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
/// The grouped kernel finds its pairs from the block index alone. Walk the same arithmetic on the host:
/// every output unit of each fold step has exactly one owner, and no block lies outside the launch grid.
#[test]
fn the_tile_grid_owns_every_output_unit_exactly_once() {
    for (m,h,k,v,c) in [(1u32,1u32,2u32,9u32,13u32),(3,5,34,9,1),(17,4,64,33,40),(33,32,512,256,70),(256,2,32,32,33)] {
        for op in [Op::AbsorbQuery,Op::AbsorbScores,Op::AbsorbValues,Op::AbsorbProject] {
            let s=Shape {op,heads:h,dim:17,taps:v,rows:c.max(31),cols:k,first:c,slots:c+6,..Shape::new(op,m as usize).unwrap()};
            let (name,blocks,threads)=launch(s).unwrap();assert_eq!((name,threads),("latent_absorption_tile",TILE_THREADS));
            let (outer,a,b)=tile(s).unwrap();let (ta,tb)=(a.div_ceil(TILE_A),b.div_ceil(TILE_B));assert_eq!(blocks,outer*ta*tb);
            let mut owners=vec![0u8;(outer*a*b) as usize];
            for block in 0..blocks {let (o,a0,b0)=(block/(ta*tb),block/tb%ta*TILE_A,block%tb*TILE_B);assert!(o<outer);
                for thread in 0..TILE_THREADS {for i in 0..2 {for j in 0..4 {let (row,col)=(a0+thread/16*2+i,b0+tile_b_row(op==Op::AbsorbValues,thread%16,j));
                    if row<a && col<b {owners[((o*a+row)*b+col) as usize]+=1;}}}}
            }
            assert!(owners.iter().all(|&n|n==1),"{op:?} m={m} h={h} k={k}");
        }
    }
}
