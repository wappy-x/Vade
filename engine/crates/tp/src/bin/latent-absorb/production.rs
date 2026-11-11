//! Independent checkpoint-loader and head-ownership check against the scalar order.
use super::*;
pub(super) fn check<'b,'d>(b:&'b Backend<'d>,checkpoint:&CheckpointAdmission)->Result<()> {
    use spark_gpu_probe::causal_ops::reference;
    let (layer,a)=checkpoint.description().layers.iter().enumerate().find_map(|(i,l)|match l.mixer {TokenMixer::LatentAttention(a)=>Some((i,a)),_=>None}).ok_or("no latent fixture layer")?;
    let plan=LayerPlan::new(checkpoint.description(),layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
    let name=plan.steps.iter().find(|s|s.kind==Kind::ExpandKv).ok_or("no expansion")?.weights[0].name.clone();
    let u=checkpoint.upload(&name).map_err(|e|e.to_string())?;if u.dtype!=Dtype::Bf16 {return Err("production twin needs BF16 weights".into());}
    let (h,d,v,k)=(a.heads as usize,a.key_dim as usize,a.value_dim as usize,a.kv_rank as usize);let live=17usize;
    // The oracle indexes original checkpoint rows. It does not repeat the loader transpose.
    let source=u.weight.as_chunks::<2>().0;
    let weight=|head:usize,feature:usize,j:usize|f32::from_bits((u16::from_le_bytes(source[(head*(d+v)+feature)*k+j]) as u32)<<16);
    let query=data(h*d,67);let latent=data(live*k,71);let count=[live as f32];
    let s=Shape {op:Op::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,cols:k as u32,rows:live as u32,epsilon:0.0,bound:0.0};
    let q=(0..h*k).map(|index|{let head=index/k;let j=index%k;
        spark_gpu_probe::latent_absorb::round(fold(&(0..d).map(|feature|query[head*d+feature]*weight(head,feature,j)).collect::<Vec<_>>(),1.0))
    }).collect::<Vec<_>>();
    let scores=reference(Shape {op:Op::AbsorbScores,..s},[&q,&latent,&count,&[]],&[])?;
    let probability=reference(Shape {op:Op::Softmax,taps:0,cols:0,..s},[&scores,&count,&[],&[]],&[])?;
    let values=reference(Shape {op:Op::AbsorbValues,..s},[&probability,&latent,&count,&[]],&[])?;
    let expected=(0..h*v).map(|index|{let head=index/v;let feature=index%v;
        let sum=fold(&(0..k).map(|j|values[head*k+j]*weight(head,d+feature,j)).collect::<Vec<_>>(),1.0);
        spark_gpu_probe::latent_absorb::round(sum)
    }).collect::<Vec<_>>();
    let latent=b.upload_vector(&latent)?;let count=b.upload_vector(&count)?;
    for parts in [1u32,2,4] {for rank in 0..parts {
        let topology=Topology {rank,parts};let layout=spark_tp::sparse_heads::Layout::new(a.heads,a.key_dim,a.value_dim,topology)?;
        let local_query=b.upload_vector(&layout.local_queries(&query)?)?;let shape=Shape {heads:layout.local_heads() as u32,..s};
        let wanted=&expected[layout.attended_rows()];
        for load in 0..2 {
            let weights=spark_tp::resident_weights::Weights::load_local_fixture(checkpoint,b,topology,std::slice::from_ref(&name))?;
            if !weights.has_absorption(&name) {return Err("production absorption was not admitted".into());}
            // The second load stores the heads x rows vectors in a larger slot bound, as a decode step under the budget does.
            let slots=shape.rows as usize+if load==0 {0} else {333};
            let output=weights.absorb(&name,b,shape,&local_query,&latent,&count,slots)?;let got=bits(b,&output)?;
            if got.len()!=wanted.len() || got.iter().zip(wanted).any(|(&a,&b)|a!=b.to_bits()) {return Err(format!("production twin failed parts={parts} rank={rank} load={load}"));}
            println!("PRODUCTION_TWIN parts={parts} rank={rank} load={load} slots={slots} words={} differing=0 physical_sparks=1",got.len());
        }
    }}Ok(())
}
