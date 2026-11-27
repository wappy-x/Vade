use spark_engine::arms;
#[allow(dead_code)] #[path="../src/hybrid/attention.rs"] mod binding;
#[allow(dead_code)] #[path="../../../tools/bench/src/fp8_reference.rs"] mod fp8_reference;
#[allow(dead_code)] #[path="../../../tools/bench/src/attention_reference.rs"] mod reference;
#[test]
fn causal_local_rows_and_reference() {
    use engine_model::{shapes::AttentionShape,plan::Rope}; use binding::FullAttentionBinding as B;
    let rope=Rope{theta:1e7,rotary_dim:64,sections:vec![11,11,10],interleaved:true}; let norms=vec![0;256]; let addresses=[4096,131072,262144,393216,524288,1048576,8388608,16777216];
    for ratio in [8,16] { let shape=AttentionShape{query_heads:ratio*2,kv_heads:2,head_dim:256,rotary_dim:64,gated:true,index:None};
        for rank in 0..2 { let b=B::new(shape,&rope,1e-6,(rank*ratio) as u32..((rank+1)*ratio) as u32,rank as u32..rank as u32+1).unwrap(); for past in [0,510,511,512,513,2048] { let (p,a)=b.row(past,past+7,4096,[64,576],2048,addresses).unwrap(); assert_eq!((p.precision,p.position_offset,a.query_heads,a.key_heads,a.count),(1,7,ratio as u32,1,1)); assert_eq!(a.scores!=0,past+1>512); assert_eq!(a.score_tiles,(past+1).div_ceil(32)); } assert_eq!(b.kv_bytes(4096).unwrap(),4096*512); assert!(b.row(4096,4096,4096,[64,576],2048,addresses).is_err()); }
        assert!(B::new(shape,&rope,1e-6,1..ratio as u32,0..1).is_err()); let mut indexed=shape; indexed.gated=false; assert!(B::new(indexed,&rope,1e-6,0..ratio as u32,0..1).is_err());
        indexed=shape; indexed.index=Some(engine_model::shapes::IndexShape{heads:4,kv_heads:1,dim:128,budget:2048,compression:4}); assert!(B::new(indexed,&rope,1e-6,0..ratio as u32,0..1).is_err()); let binding=B::new(shape,&rope,1e-6,0..ratio as u32,0..1).unwrap(); let mut unaligned=addresses; unaligned[6]+=2; assert!(binding.row(512,512,4096,[64,576],2048,unaligned).is_err()); unaligned=addresses; unaligned[7]=0; assert!(binding.row(512,512,4096,[64,576],2048,unaligned).is_err());
        let mut r=reference::AttentionReference::new(ratio as usize,1,1e7,1e-6).unwrap(); let q=vec![0.0;ratio as usize*512]; let k=vec![1.0;256]; let v=vec![2.0;256]; assert!(r.step(&q,&k,&v,[&norms,&norms],0).unwrap().iter().all(|&x| x==1.0)); let mut fork=r.clone(); for p in 1..5 { assert_eq!(r.step(&q,&k,&v,[&norms,&norms],p).unwrap(),fork.step(&q,&k,&v,[&norms,&norms],p).unwrap()); } assert_eq!((r.keys,r.values,r.positions),(fork.keys,fork.values,fork.positions));
    }
    let x:Vec<_>=(0..256).map(|i| (i%13) as f32/16.0).collect(); let f=vec![1.0;32]; let a=reference::prepare_head(&x,&norms,&f,0,1e-6).unwrap(); let b=reference::prepare_head(&x,&norms,&f,513,1e-6).unwrap(); assert_eq!(&a[64..],&b[64..]); assert_ne!(&a[..64],&b[..64]);
}
