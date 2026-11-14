use std::{path::Path,time::Instant};
use engine_model::{plan::{ConfigPlan,admission::CheckpointAdmission,description::TokenMixer,layer::{Kind,LayerPlan,Topology}},safetensors::{Dtype,read_bounded},shapes::{CONFIG_BYTES,HEADER_BYTES}};
use spark_gpu_probe::{causal_ops::{Shape,Op},driver::{Backend,DeviceVector},prompt_ops::{Shape as Rows,Op as Row}};
#[path="production.rs"] mod production;
#[path="prompt.rs"] mod prompt;
#[path="grouped.rs"] mod grouped;
type Result<T>=std::result::Result<T,String>;
struct Case<'a> {layer:usize,shape:Shape,weight:DeviceVector<'a>,original:DeviceVector<'a>,latent:DeviceVector<'a>,query:DeviceVector<'a>,packed:DeviceVector<'a>,raw_weight:Vec<f32>,raw_latent:Vec<f32>,raw_query:Vec<f32>}
fn open(root:&Path)->Result<CheckpointAdmission> {
    let config=read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|e|e.to_string())?;
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&config).map_err(|e|e.to_string())? else {return Err("deferred checkpoint required".into());};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|e|e.to_string())?;
    d.open_headers(root,&index).map_err(|e|e.to_string())
}
fn data(n:usize,seed:usize)->Vec<f32> {
    let mut state=seed as u64+1;(0..n).map(|_| {state^=state<<13;state^=state>>7;state^=state<<17;
        let value=((state>>32) as i32 as f32)/(i32::MAX as f32)*0.25;
        f32::from_bits(spark_gpu_probe::next_math::bf16(value)<<16)}).collect()
}
fn load<'a>(b:&Backend<'a>,c:&CheckpointAdmission,limit:usize)->Result<Vec<Case<'a>>> {
    let mut cases=Vec::new();
    for (layer,description) in c.description().layers.iter().enumerate() {
        let TokenMixer::LatentAttention(spec)=description.mixer else {continue;};
        if cases.len()==limit {break;}
        if spec.rotary_dim!=0 || spec.heads%2!=0 {return Err("fixture requires divisible local heads and no rotary dimensions".into());}
        let (heads,dim,cols,value)=(spec.heads/2,spec.key_dim,spec.kv_rank,spec.value_dim);
        let rows=spec.indexer.token_budget+if spec.indexer.select_tail {spec.indexer.pool-1} else {0};
        let plan=LayerPlan::new(c.description(),layer,1,Topology {rank:0,parts:2}).map_err(|e|e.to_string())?;
        let name=&plan.steps.iter().find(|s|s.kind==Kind::ExpandKv).ok_or("no expansion")?.weights[0].name;
        let weight=c.upload_rows(name,0..heads*(dim+value)).map_err(|e|e.to_string())?;
        if weight.dtype!=Dtype::Bf16 {return Err("fixture requires BF16 kv projection".into());}

        let original=weight.weight.as_chunks::<4>().0.iter().map(|v|f32::from_bits(u32::from_le_bytes(*v))).collect::<Vec<_>>();
        let original=b.upload_vector(&original)?;
        let wide=weight.weight.as_chunks::<2>().0.iter().map(|v|f32::from_bits((u16::from_le_bytes(*v) as u32)<<16)).collect::<Vec<_>>();
        let mut reordered=wide.clone();let (h,d,k,v)=(heads as usize,dim as usize,cols as usize,value as usize);
        for head in 0..h {for j in 0..k {for feature in 0..d {
            reordered[head*(d+v)*k+j*d+feature]=wide[(head*(d+v)+feature)*k+j];
        }}}
        let raw_weight=reordered.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect::<Vec<_>>();
        let raw_latent=data((rows*cols) as usize,layer*2);let raw_query=data((heads*dim) as usize,layer*2+1);
        let latent=b.upload_vector(&raw_latent)?.into_resident_shared()?;let query=b.upload_vector(&raw_query)?;let packed_weight=b.upload_vector(&raw_weight)?;
        let wide=b.upload_vector(&wide)?;let mut expansion=Rows::new(Row::ProjectRows,rows as usize).ok_or("prompt shape")?;
        expansion.rows=(heads*(dim+value)) as u32;expansion.cols=cols as u32;
        let expanded=b.prompt_operation(expansion,[Some(&latent),None,None,None,None],Some(&wide))?;
        let expanded=b.round_bf16_lanes(&expanded)?.into_vector();
        let mut transpose=Rows::new(Row::Transpose,1).ok_or("transpose shape")?;transpose.rows=expansion.rows;transpose.cols=rows as u32;
        let expanded=b.prompt_operation(transpose,[Some(&expanded),None,None,None,None],None)?;
        let pack=Shape {op:Op::PackBf16,heads:0,dim:0,taps:0,rows:(expanded.len()/2) as u32,cols:0,epsilon:0.0,bound:0.0};
        let packed=b.causal_operation(pack,[Some(&expanded),None,None,None],None)?.into_resident_shared()?;
        b.synchronize()?;
        let shape=Shape {op:Op::AbsorbQuery,heads:heads as u32,dim:dim as u32,taps:value as u32,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
        println!("LOAD layer={layer} heads={heads} key={dim} value={value} latent={cols} slots={rows} weight={name}");
        cases.push(Case {layer,shape,weight:packed_weight,original,latent,query,packed,raw_weight,raw_latent,raw_query});
    }
    if cases.is_empty() {return Err("no latent layers".into());}Ok(cases)
}
fn candidate<'a>(b:&Backend<'a>,case:&Case<'a>,rows:usize)->Result<Vec<DeviceVector<'a>>> {
    let shape=Shape {rows:rows as u32,..case.shape};let latent=case.latent.resident_range(0..rows*shape.cols as usize)?;
    let count=b.upload_vector(&[rows as f32])?;
    let q=b.causal_operation(shape,[Some(&case.query),None,None,None],Some(&case.weight))?;
    let scores=b.causal_operation(Shape {op:Op::AbsorbScores,..shape},[Some(&q),Some(&latent),Some(&count),None],None)?;
    let probabilities=b.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..shape},[Some(&scores),Some(&count),None,None],None)?;
    let u=b.causal_operation(Shape {op:Op::AbsorbValues,..shape},[Some(&probabilities),Some(&latent),Some(&count),None],None)?;
    let output=b.causal_operation(Shape {op:Op::AbsorbProject,..shape},[Some(&u),None,None,None],Some(&case.weight))?;
    Ok(vec![q,scores,probabilities,u,output])
}
fn profile<'a>(b:&Backend<'a>,case:&Case<'a>,rows:usize)->Result<()> {
    let shape=Shape {rows:rows as u32,..case.shape};let latent=case.latent.resident_range(0..rows*shape.cols as usize)?;let count=b.upload_vector(&[rows as f32])?;b.synchronize()?;
    let a=b.gpu_mark()?;
    let q=b.causal_operation(shape,[Some(&case.query),None,None,None],Some(&case.weight))?;let c=b.gpu_mark()?;
    let scores=b.causal_operation(Shape {op:Op::AbsorbScores,..shape},[Some(&q),Some(&latent),Some(&count),None],None)?;let d=b.gpu_mark()?;
    let probabilities=b.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..shape},[Some(&scores),Some(&count),None,None],None)?;let e=b.gpu_mark()?;
    let u=b.causal_operation(Shape {op:Op::AbsorbValues,..shape},[Some(&probabilities),Some(&latent),Some(&count),None],None)?;let f=b.gpu_mark()?;
    let output=b.causal_operation(Shape {op:Op::AbsorbProject,..shape},[Some(&u),None,None,None],Some(&case.weight))?;let g=b.gpu_mark()?;
    b.synchronize()?;std::hint::black_box(output);
    println!("PARTS layer={} live={rows} query_ns={} scores_ns={} softmax_ns={} values_ns={} project_ns={} event_perturbed=true",case.layer,a.ns_until(&c)?,c.ns_until(&d)?,d.ns_until(&e)?,e.ns_until(&f)?,f.ns_until(&g)?);Ok(())
}
fn baseline<'a>(b:&Backend<'a>,case:&Case<'a>,rows:usize)->Result<Vec<DeviceVector<'a>>> {
    let shape=Shape {op:Op::ScoresIndirect,taps:rows as u32,cols:case.shape.taps,..case.shape};
    let width=shape.heads as usize*(shape.dim+shape.cols) as usize/2;
    // The live tail's newest token must be expanded on rc.8. Time that producer and consume
    // its actual result. All earlier rows are already cached. This charges no other cache miss.
    let latent=case.latent.resident_range((rows-1)*case.shape.cols as usize..rows*case.shape.cols as usize)?;
    let fresh=b.project_bf16(width*2,case.shape.cols as usize,&latent,&case.original)?;
    let fresh=b.round_bf16_lanes(&fresh)?.into_vector();
    let fresh=b.causal_operation(Shape {op:Op::PackBf16,heads:0,dim:0,taps:0,rows:width as u32,cols:0,epsilon:0.0,bound:0.0},[Some(&fresh),None,None,None],None)?;
    let mut parts=Vec::new();for first in (0..rows-1).step_by(4) {parts.push(case.packed.resident_range(first*width..(first+4).min(rows-1)*width)?);}
    let mut input=parts.iter().collect::<Vec<_>>();input.push(&fresh);
    Ok(b.resident_packed_attention(shape,&input,&case.query)?.into_iter().collect())
}
fn bits(b:&Backend<'_>,v:&DeviceVector<'_>)->Result<Vec<u32>> {Ok(b.read_vector_values(v)?.into_iter().map(f32::to_bits).collect())}
fn twin<'a>(b:&Backend<'a>,case:&Case<'a>,rows:usize)->Result<()> {
    use spark_gpu_probe::causal_ops::reference;
    let s=Shape {rows:rows as u32,..case.shape};let latent=&case.raw_latent[..rows*s.cols as usize];
    let q=reference(s,[&case.raw_query,&[],&[],&[]],&case.raw_weight)?;
    let scores=reference(Shape {op:Op::AbsorbScores,..s},[&q,latent,&[rows as f32],&[]],&[])?;
    let probabilities=reference(Shape {op:Op::Softmax,taps:0,cols:0,..s},[&scores,&[rows as f32],&[],&[]],&[])?;
    let u=reference(Shape {op:Op::AbsorbValues,..s},[&probabilities,latent,&[rows as f32],&[]],&[])?;
    let output=reference(Shape {op:Op::AbsorbProject,..s},[&u,&[],&[],&[]],&case.raw_weight)?;
    let device=candidate(b,case,rows)?;
    for (plane,(want,got)) in [q,scores,probabilities,u,output].iter().zip(&device).enumerate() {
        let got=bits(b,got)?;let count=want.iter().zip(&got).filter(|(a,b)|a.to_bits()!=**b).count();
        println!("TWIN layer={} rows={rows} plane={plane} words={} differing={count}",case.layer,want.len());
        if count!=0 {return Err("CPU/GPU twin differs".into());}
    }Ok(())
}
fn fold(terms:&[f32],weight:f32)->f32 {
    let mut strips=[0.0f32;32];for (j,&term) in terms.iter().enumerate() {let product=term*weight;strips[(j/8)%32]+=product;}
    for distance in [16,8,4,2,1] {for lane in 0..distance {strips[lane]+=strips[lane+distance];}}strips[0]
}
fn conformance(b:&Backend<'_>)->Result<()> {
    let mut cases=0;let mut words=0;let mut vectors=Vec::new();let mut seed=0x83a1_c43f_17u64;
    for n in (0usize..=600).chain([4096,16384,16385]) {
        let terms=(0..n).map(|_| {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;f32::from_bits(seed as u32)}).collect::<Vec<_>>();vectors.push(terms);
        let finite=(0..n).map(|j|{let bits=0x3f000000|((j.wrapping_mul(2654435761).wrapping_add(40503)) as u32&0x7fffff);f32::from_bits(bits|if (j+1)%3==0 {0x80000000} else {0})}).collect::<Vec<_>>();vectors.push(finite);
    }
    for special in [vec![0.0,-0.0],vec![-0.0;5],vec![f32::from_bits(1);257],vec![f32::MAX,-f32::MAX],
        vec![f32::INFINITY],vec![f32::NEG_INFINITY],vec![f32::INFINITY,f32::NEG_INFINITY],vec![f32::from_bits(0x7fc01234)]] {vectors.push(special);}
    for terms in vectors {let n=terms.len();let rows=n.max(1);
        let source=b.upload_vector(if n==0 {&[0.0]} else {&terms})?;
        let latent=b.upload_vector(&(0..rows).flat_map(|_|[1.0,-0.5]).collect::<Vec<_>>())?;
        let count=b.upload_vector(&[n as f32])?;
        let shape=Shape {op:Op::AbsorbValues,heads:1,dim:1,taps:1,rows:rows as u32,cols:2,epsilon:0.0,bound:0.0};
        let output=b.causal_operation(shape,[Some(&source),Some(&latent),Some(&count),None],None)?;
        let got=b.read_vector_values(&output)?;let r=spark_gpu_probe::latent_absorb::round;let want=[r(fold(&terms,1.0)),r(fold(&terms,-0.5))];
        for (i,(&a,&c)) in got.iter().zip(&want).enumerate() {if a.to_bits()!=c.to_bits() && !(a.is_nan() && c.is_nan()) {
            return Err(format!("order conformance n={n} feature={i} got={:08x} want={:08x}",a.to_bits(),c.to_bits()));
        }}cases+=1;words+=got.len();
    }
    println!("ORDER_CONFORMANCE cases={cases} words={words} differing=0 NaN_payloads=exempt order=SUM-ORDER-v1");Ok(())
}
fn median(v:&mut [u128])->u128 {v.sort_unstable();v[v.len()/2]}
pub fn run()->Result<()> {
    let args=std::env::args().collect::<Vec<_>>();let root=args.get(1).ok_or("usage: latent-absorb CHECKPOINT [layers] [samples]")?;
    let limit=args.get(2).map(|v|v.parse()).transpose().map_err(|_|"invalid layers")?.unwrap_or(usize::MAX);
    let samples=args.get(3).map(|v|v.parse()).transpose().map_err(|_|"invalid samples")?.unwrap_or(6usize);
    let checkpoint=open(Path::new(root))?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|b| {
        b.configure_vector_pool(2<<30);let queue=b.resident_queue()?;
        println!("SCOPE real_checkpoint_weights synthetic_BF16_activations local_heads parts=2 capacity_context=32768 standalone_component=true device_wire=false order=SUM-ORDER-v1");
        if args.get(4).is_some_and(|v|v=="prompt") {let cases=load(b,&checkpoint,limit)?;prompt::run(b,&cases,samples)?;drop(queue);b.synchronize()?;return Ok(());}
        if args.get(4).is_some_and(|v|v=="grouped") {let cases=load(b,&checkpoint,limit)?;grouped::run(b,&cases,samples)?;drop(queue);b.synchronize()?;return Ok(());}
        conformance(b)?;production::check(b,&checkpoint)?;let cases=load(b,&checkpoint,limit)?;
        for rows in [16usize,128,1041,2051] {twin(b,&cases[0],rows.min(cases[0].shape.rows as usize))?;}
        if args.get(4).is_some_and(|v|v=="recorded") {b.enable_resident_launches()?;println!("RECORDER enabled=true");}
        for cell in [15usize,127,1040,4158] {
            profile(b,&cases[0],(cell+1).min(cases[0].shape.rows as usize))?;
            let mut wall=[Vec::new(),Vec::new()];let mut event=[Vec::new(),Vec::new()];let mut previous:Option<Vec<Vec<u32>>>=None;
            for sample in 0..samples+2 {for arm in if sample%2==0 {[0,1]} else {[1,0]} {
                b.synchronize()?;let time=Instant::now();let start=b.gpu_mark()?;let mut outputs=Vec::new();
                for case in &cases {let rows=(cell+1).min(case.shape.rows as usize);
                    outputs.push(if arm==0 {baseline(b,case,rows)?} else {candidate(b,case,rows)?});}
                let end=b.gpu_mark()?;b.synchronize()?;let elapsed=time.elapsed().as_nanos();let gpu=start.ns_until(&end)?;
                if sample>=2 {wall[arm].push(elapsed);event[arm].push(gpu);}
                println!("SAMPLE cell=c{cell} sample={sample} arm={arm} wall_ns={elapsed} event_ns={gpu}");
                let words=outputs.iter().map(|planes|bits(b,planes.last().unwrap())).collect::<Result<Vec<_>>>()?;
                if arm==1 {if let Some(want)=&previous {if want!=&words {return Err("candidate repeat output differs".into());}}else {previous=Some(words);}}
                if sample==0 && arm==0 {for (case,old) in cases.iter().zip(&outputs) {let live=(cell+1).min(case.shape.rows as usize);let new=candidate(b,case,live)?;
                    for (plane,(a,c)) in old.iter().zip([&new[1],&new[2],&new[4]]).enumerate() {let old=bits(b,a)?;let new=bits(b,c)?;
                        let old=if plane<2 {old.chunks_exact(case.shape.rows as usize).flat_map(|row|row[..live].iter().copied()).collect::<Vec<_>>()} else {old};
                        if old.len()!=new.len() {return Err("reset-price extent differs".into());}
                        let diff=old.iter().zip(&new).filter(|(a,b)|a!=b).count();
                        println!("RESET_PRICE cell=c{cell} layer={} plane={plane} words={} differing={diff} unused_zero_slots=omitted",case.layer,old.len());
                    }
                }}
            }}
            let old=median(&mut wall[0]);let new=median(&mut wall[1]);let og=median(&mut event[0]);let ng=median(&mut event[1]);
            println!("RESULT cell=c{cell} layers={} samples={samples} baseline_wall_ns={old} candidate_wall_ns={new} saving_wall_ms={:.6} baseline_event_ns={og} candidate_event_ns={ng} saving_event_ms={:.6} baseline_new_rows=1 full_step=false",cases.len(),(old as f64-new as f64)/1e6,(og as f64-ng as f64)/1e6);
        }
        drop(queue);b.synchronize()?;Ok(())
    })
}
