//! Independent scalar-versus-grouped GPU row bits. No checkpoint or policy is selected.
#![forbid(unsafe_code)]
use super::{Result,write,floats,sha256};
use std::path::Path;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Shape,Op}};
fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {let mut v=b.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;b.write_vector_bytes(&mut v,bytes)?;b.synchronize()?;Ok(v)}
fn equal(a:&[f32],b:&[f32])->Result<()> {if a.len()!=b.len() || a.iter().zip(b).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err(format!("grouped prompt output differs from scalar bits lengths={}/{} first={:?}",a.len(),b.len(),a.iter().zip(b).enumerate().find(|(_, (a,b))|a.to_bits()!=b.to_bits()).map(|(at,(a,b))|(at,a.to_bits(),b.to_bits()))));}Ok(())}
fn shape(op:Op,rows:u32,cols:u32,tokens:u32)->Shape {Shape {op,rows,cols,tokens,block_rows:2,block_cols:16,global:0.75,activation:1.5,bound:0.0}}
fn packed(b:&Backend<'_>,s:Shape,output:&mut String)->Result<()> {
    let (r,k,m)=(s.rows as usize,s.cols as usize,s.tokens as usize);eprintln!("PROMPT FIXTURE op={:?} rows={r} cols={k} tokens={m}",s.op);
    let x=(0..m*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();
    let input=b.upload_vector(&x)?;let nv=s.op==Op::NvRows;
    let w=if nv {(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>()} else {(0..r*k).map(|i|(i%120+1) as u8).collect()};
    let scales=if nv {(0..r*k/16).map(|i|0x28+(i%24) as u8).collect::<Vec<_>>()} else {(0..r.div_ceil(2)*k.div_ceil(16)).flat_map(|i|(0.25+(i%9) as f32/16.0).to_le_bytes()).collect()};
    let weight=raw(b,&w)?;let scale=raw(b,&scales)?;
    let quantized=if nv {Some(b.expert_operation(Shape {op:Op::QuantizeRows,..s},[Some(&input),None,None,None],None,None)?)} else {None};
    let source=quantized.as_ref().unwrap_or(&input);let got=b.expert_operation(s,[Some(source),None,None,None],Some(&weight),Some(&scale))?;let got=b.read_vector_values(&got)?;
    let source_host=if nv {expert_ops::reference(Shape {op:Op::QuantizeRows,..s},[&x,&[],&[],&[]],&[],&[])?} else {x.clone()};
    let expected=expert_ops::reference(s,[&source_host,&[],&[],&[]],&w,&scales)?;equal(&got,&expected)?;
    for token in 0..m {let row=b.upload_vector(&x[token*k..(token+1)*k])?;
        let q=if nv {Some(b.expert_operation(Shape {op:Op::Quantize,tokens:1,..s},[Some(&row),None,None,None],None,None)?)} else {None};
        let scalar=b.expert_operation(Shape {op:if nv {Op::NvProject} else {Op::Fp8Project},tokens:1,..s},[Some(q.as_ref().unwrap_or(&row)),None,None,None],Some(&weight),Some(&scale))?;
        let scalar=b.read_vector_values(&scalar)?;let column=(0..r).map(|i|got[i*m+token]).collect::<Vec<_>>();equal(&scalar,&column)?;
    }
    output.push_str(&format!("op={:?} rows={r} cols={k} tokens={m} sha256={} scalar_bits=true host_bits=true\n",s.op,sha256::digest(&floats(&got)?)));Ok(())
}
fn plain(b:&Backend<'_>,m:usize,k:usize,output:&mut String)->Result<()> {
    use spark_gpu_probe::causal_ops::{self,Shape,Op};
    eprintln!("PROMPT FIXTURE op=ProjectRows rows=7 cols={k} tokens={m}");
    let s=Shape {op:Op::ProjectRows,heads:m as u32,dim:0,taps:0,rows:7,cols:k as u32,epsilon:0.0,bound:0.0};
    let x=(0..m*k).map(|i|if i%11==0 {-0.0} else {(i as i32%19-9) as f32/7.3}).collect::<Vec<_>>();let w=(0..7*k).map(|i|(i as i32%13-6) as f32/9.1).collect::<Vec<_>>();
    let input=b.upload_vector(&x)?;let weight=b.upload_vector(&w)?;let got=b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?;let got=b.read_vector_values(&got)?;
    equal(&got,&causal_ops::reference(s,[&x,&[],&[],&[]],&w)?).map_err(|e|format!("grouped versus host: {e}"))?;
    for token in 0..m {let row=b.upload_vector(&x[token*k..(token+1)*k])?;let scalar=b.causal_operation(Shape {op:Op::Project,..s},[Some(&row),None,None,None],Some(&weight))?;
        equal(&b.read_vector_values(&scalar)?,&(0..7).map(|r|got[r*m+token]).collect::<Vec<_>>()).map_err(|e|format!("scalar versus grouped token={token}: {e}"))?;}
    output.push_str(&format!("op=ProjectRows rows=7 cols={k} tokens={m} sha256={} scalar_bits=true host_bits=true\n",sha256::digest(&floats(&got)?)));Ok(())
}
/// Products whose fused multiply-add differs from a rounded product then add.
/// A contracted grouped kernel fails this against the one-row pinned kernel.
fn plain_unfused(b:&Backend<'_>,m:usize,output:&mut String)->Result<()> {
    use spark_gpu_probe::causal_ops::{self,Shape,Op};
    let a=f32::from_bits(0x3f800001);let c=f32::from_bits(0x3f7ffffe);
    let w=[-1.0,a,0.0,16777216.0,1.0,-16777216.0,-16777216.0,16777216.0,1.0];
    let x=(0..m).flat_map(|_|[1.0,c,1.0]).collect::<Vec<_>>();
    let s=Shape {op:Op::ProjectRows,heads:m as u32,dim:0,taps:0,rows:3,cols:3,epsilon:0.0,bound:0.0};
    let input=b.upload_vector(&x)?;let weight=b.upload_vector(&w)?;let got=b.read_vector_values(&b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?)?;
    equal(&got,&causal_ops::reference(s,[&x,&[],&[],&[]],&w)?).map_err(|e|format!("unfused grouped versus host: {e}"))?;
    if (0..m).any(|token|got[token].to_bits()!=0.0f32.to_bits()) {return Err("grouped plain projection fused a multiply and add".into());}
    for token in 0..m {let row=b.upload_vector(&x[token*3..token*3+3])?;let scalar=b.causal_operation(Shape {op:Op::Project,..s},[Some(&row),None,None,None],Some(&weight))?;
        equal(&b.read_vector_values(&scalar)?,&(0..3).map(|r|got[r*m+token]).collect::<Vec<_>>()).map_err(|e|format!("unfused scalar versus grouped token={token}: {e}"))?;}
    output.push_str(&format!("op=ProjectRows unfused rows=3 cols=3 tokens={m} sha256={} scalar_bits=true host_bits=true\n",sha256::digest(&floats(&got)?)));Ok(())
}
fn micro(b:&Backend<'_>,text:&mut String)->Result<()> {
    let s=shape(Op::NvRows,1024,4096,4);let (r,k,m)=(s.rows as usize,s.cols as usize,s.tokens as usize);
    let x=(0..m*k).map(|i|(i as i32%37-18) as f32/8.0).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
    let q=b.expert_operation(Shape {op:Op::QuantizeRows,..s},[Some(&input),None,None,None],None,None)?;
    let weight=raw(b,&(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>())?;let scale=raw(b,&vec![0x38;r*k/16])?;
    let mut scalar_inputs=Vec::new();for token in 0..m {scalar_inputs.push(b.upload_vector(&expert_ops::reference(Shape {op:Op::Quantize,tokens:1,..s},[&x[token*k..(token+1)*k],&[],&[],&[]],&[],&[])?)?);}
    b.set_profiling(true);
    for repeat in 1..=3 {let before=b.stats();let mut expected=Vec::new();for row in &scalar_inputs {expected.push(b.expert_operation(Shape {op:Op::NvProject,tokens:1,..s},[Some(row),None,None,None],Some(&weight),Some(&scale))?);}
        let scalar=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();let before=b.stats();
        let grouped=b.expert_operation(s,[Some(&q),None,None,None],Some(&weight),Some(&scale))?;let grouped_ns=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();
        let got=b.read_vector_values(&grouped)?;for (token,row) in expected.iter().enumerate() {equal(&b.read_vector_values(row)?,&(0..r).map(|row|got[row*m+token]).collect::<Vec<_>>())?;}
        text.push_str(&format!("MICRO repeat={repeat} rows={r} cols={k} tokens={m} scalar_gpu_ns={scalar} grouped_gpu_ns={grouped_ns} scalar_seconds_per_row={:.9} grouped_seconds_per_row={:.9} exact=true\n",scalar as f64/1e9/m as f64,grouped_ns as f64/1e9/m as f64));
    }b.set_profiling(false);Ok(())
}
fn micro_fp8(b:&Backend<'_>,text:&mut String)->Result<()> {
    let s=Shape {block_rows:128,block_cols:128,..shape(Op::Fp8Rows,1024,4096,4)};
    let (r,k,m)=(s.rows as usize,s.cols as usize,s.tokens as usize);
    let x=(0..m*k).map(|i|(i as i32%37-18) as f32/8.0).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
    let weight=raw(b,&(0..r*k).map(|i|(i%120+1) as u8).collect::<Vec<_>>())?;
    let scale=raw(b,&vec![1.0f32;r.div_ceil(128)*k.div_ceil(128)].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
    let inputs=(0..m).map(|token|b.upload_vector(&x[token*k..(token+1)*k])).collect::<Result<Vec<_>>>()?;
    b.set_profiling(true);
    for repeat in 1..=3 {let before=b.stats();let mut expected=Vec::new();for row in &inputs {
        expected.push(b.expert_operation(Shape {op:Op::Fp8Project,tokens:1,..s},[Some(row),None,None,None],Some(&weight),Some(&scale))?);}
        let scalar=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();let before=b.stats();
        let grouped=b.expert_operation(s,[Some(&input),None,None,None],Some(&weight),Some(&scale))?;
        let grouped_ns=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();let got=b.read_vector_values(&grouped)?;
        for (token,row) in expected.iter().enumerate() {equal(&b.read_vector_values(row)?,&(0..r).map(|row|got[row*m+token]).collect::<Vec<_>>())?;}
        text.push_str(&format!("MICRO_FP8 repeat={repeat} rows={r} cols={k} tokens={m} scalar_gpu_ns={scalar} grouped_gpu_ns={grouped_ns} exact=true\n"));
    }b.set_profiling(false);Ok(())
}
/// Every prompt kernel operation against the shared host code, signed zeros included.
fn row_ops(b:&Backend<'_>,bits:&mut String)->Result<()> {
    use spark_gpu_probe::{causal_ops::{Op as C,Shape as Inner},prompt_ops::{Op as R,Shape as S,reference}};
    let v=|n:usize,seed:usize,scale:f32|(0..n).map(|i|if (i+seed)%19==0 {-0.0} else {((i+seed)*31%53) as f32*scale-26.0*scale}).collect::<Vec<_>>();
    let base=|op,m|S::new(op,m).ok_or_else(||"prompt fixture row count".to_string());
    let mut cases:Vec<(S,Vec<Vec<f32>>,Vec<f32>)>=Vec::new();
    let inner=Inner {op:C::Project,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
    for m in [1usize,2,5,33,130] {
        let (h,d,t)=(4usize,16usize,4usize);let w=h*d;
        cases.push((S {rows:7,cols:48,..base(R::ProjectRows,m)?},vec![v(m*48,0,0.031)],v(7*48,3,0.017)));
        cases.push((S {rows:m as u32,cols:9,..base(R::Transpose,1)?},vec![v(m*9,1,0.1)],vec![]));
        cases.push((S {dim:64,cols:1,..base(R::NormApply,m)?},vec![v(m*64,2,0.1)],v(64,5,0.03)));
        cases.push((S {dim:64,cols:0,..base(R::NormApply,m)?},vec![v(m*64,2,0.1)],v(64,5,0.03)));
        let conv=S {heads:h as u32,dim:d as u32,taps:t as u32,..base(R::ConvRows,m)?};let x=vec![v(m*w,0,0.0625),v(m*w,7,0.0625),v(m*w,13,0.0625),v(3*(t-1)*w,11,0.03125)];
        cases.push((conv,x.clone(),v(3*w*t,5,0.0078125)));cases.push((S {op:R::HistoryRows,..conv},x,vec![]));
        let step=S {heads:h as u32,dim:d as u32,..base(R::StepRows,m)?};let stride=4*d+1;let mut x=vec![v(m*h*stride,0,0.03125),v(h*d*d,9,0.015625)];
        let decay=reference(S {op:R::StepDecay,..step},[&x[0],&[],&[],&[],&[]],&[])?;cases.push((S {op:R::StepDecay,..step},vec![x[0].clone()],vec![]));x.push(decay);
        cases.push((step,x,vec![]));
        let prep=Inner {op:C::Prepare,heads:h as u32,dim:d as u32,taps:0,bound:-5.0,epsilon:1e-5,..inner};
        cases.push((S::wrapped(prep,m).ok_or("prompt fixture")?,vec![v(m*3*w,1,0.0625),v(m*w,2,0.0625),v(m*h,3,0.0625)],v(h+w,4,0.01)));
        let gated=Inner {op:C::GatedNorm,heads:h as u32,dim:d as u32,epsilon:1e-5,..inner};
        cases.push((S::wrapped(gated,m).ok_or("prompt fixture")?,vec![v(m*w,1,0.0625),v(m*w,2,0.0625)],v(d,4,0.01)));
        let key=Inner {op:C::KeyNorm,dim:128,epsilon:1e-5,..inner};cases.push((S::wrapped(key,m).ok_or("prompt fixture")?,vec![v(m*128,1,0.0625)],v(256,4,0.01)));
        let mean=Inner {op:C::StreamMean,heads:4,dim:64,..inner};cases.push((S::wrapped(mean,m).ok_or("prompt fixture")?,vec![v(m*256,1,0.0625)],vec![]));
        let route=Inner {op:C::Route,heads:1,dim:1,taps:1,rows:32,cols:8,bound:2.5,epsilon:1e-5};cases.push((S::wrapped(route,m).ok_or("prompt fixture")?,vec![v(m*32,1,0.0625)],v(32,4,0.01)));
        let fold=Inner {op:C::ExpertFold,taps:1,rows:8,cols:64,..inner};let ids=(0..m*8).map(|i|((i%8)*3+i/8%2) as f32).collect::<Vec<_>>();
        cases.push((S::wrapped(fold,m).ok_or("prompt fixture")?,vec![ids,v(m*8,2,0.03),v(m*8*64,3,0.0625),v(m*64,4,0.0625)],vec![]));
        let (pool,first)=(4usize,3usize);let r=(first+m-1)/pool;
        if r>0 {cases.push((S {heads:4,dim:16,taps:pool as u32,rows:r as u32,first:first as u32,..base(R::PoolScoresRows,m)?},vec![v(m*w,1,0.125),v(r*16,2,0.0625),v(m*4,3,0.25)],vec![]));}
        for tail in [0u32,1] {for budget in [8usize,16,2048] {let slots=budget+if tail!=0 {pool-1} else {0};
            let s=S {dim:pool as u32,taps:budget as u32,rows:r as u32,cols:tail,first:first as u32,slots:slots as u32,..base(R::SelectRows,m)?};
            cases.push((s,vec![(0..m*r).map(|i|if i%3==0 {f32::NEG_INFINITY} else {(i*7%13) as f32*0.125}).collect()],vec![]));}}
        // Live prefixes through shared expanded rows, including reordered slots.
        let (heads,kd,vd,first)=(3usize,8usize,4usize,2usize);let total=first+m;let row=heads*(kd+vd);let slots=7usize;let cap=slots.min(total);
        let e=v(total*row,1,0.0625);let q=v(m*heads*kd,3,0.125);let mut selected=vec![-1.0f32;m*(slots+1)];
        for token in 0..m {let context=first+token;let count=context.min(slots);for slot in 0..count {selected[token*(slots+1)+slot]=(context-1-slot) as f32;}selected[token*(slots+1)+slots]=count as f32;}
        let att=S {heads:heads as u32,dim:kd as u32,taps:vd as u32,rows:total as u32,cols:cap as u32,slots:slots as u32,..base(R::ScoresRows,m)?};
        let scores=reference(att,[&q,&e,&selected,&[],&[]],&[])?;let prob=reference(S {op:R::SoftmaxRows,..att},[&scores,&selected,&[],&[],&[]],&[])?;
        cases.push((att,vec![q,e.clone(),selected.clone()],vec![]));cases.push((S {op:R::SoftmaxRows,..att},vec![scores,selected.clone()],vec![]));cases.push((S {op:R::ValuesRows,..att},vec![prob,e,selected],vec![]));
        let table=(0..m*3).map(|i|((i*7)%(m*2)) as f32).collect::<Vec<_>>();cases.push((S {rows:(m*3) as u32,cols:32,first:(m*2) as u32,..base(R::PermuteRows,1)?},vec![v(m*2*32,6,0.1),table],vec![]));
    }
    // A fused multiply-add gives a nonzero first output for these operands.
    let a=f32::from_bits(0x3f800001);let c=f32::from_bits(0x3f7ffffe);
    for m in [1usize,3,40] {let shape=S {rows:3,cols:3,..base(R::ProjectRows,m)?};let x=(0..m).flat_map(|_|[1.0,c,1.0]).collect::<Vec<_>>();
        let w=[-1.0,a,0.0,16777216.0,1.0,-16777216.0,-16777216.0,16777216.0,1.0];
        let got=b.read_vector_values(&b.prompt_operation(shape,[Some(&b.upload_vector(&x)?),None,None,None,None],Some(&b.upload_vector(&w)?))?)?;
        if (0..m).any(|token|got[token].to_bits()!=0.0f32.to_bits()) {return Err("prompt plain rows fused a multiply and add".into());}
        cases.push((shape,vec![x],w.to_vec()));}
    for (shape,inputs,weight) in &cases {
        let empty:&[f32]=&[];let host=reference(*shape,std::array::from_fn(|i|inputs.get(i).map_or(empty,Vec::as_slice)),weight)?;
        let owners=inputs.iter().map(|x|if x.is_empty() {Ok(None)} else {b.upload_vector(x).map(Some)}).collect::<Result<Vec<_>>>()?;
        let w=if weight.is_empty() {None} else {Some(b.upload_vector(weight)?)};
        let x=std::array::from_fn(|i|owners.get(i).and_then(Option::as_ref));
        let got=b.read_vector_values(&b.prompt_operation(*shape,x,w.as_ref())?)?;
        equal(&got[..host.len()],&host).map_err(|e|format!("prompt op {:?} tokens={}: {e}",shape.op,shape.tokens))?;
        bits.push_str(&format!("op={:?} inner={:?} tokens={} words={} sha256={} host_bits=true\n",shape.op,shape.inner.op,shape.tokens,host.len(),sha256::digest(&floats(&host)?)));
    }Ok(())
}
/// A whole-layer launch equals each expert's own launch, for unequal token groups.
fn experts(b:&Backend<'_>,bits:&mut String)->Result<()> {
    let (r,k)=(9usize,48usize);let tokens=[1usize,2,5,1,33];let n=tokens.iter().sum::<usize>();
    let x=(0..n*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
    let mut raws=Vec::new();let mut hosts=Vec::new();
    for e in 0..tokens.len() {let w=(0..r*k/2).map(|i|((i+e)*17+31) as u8).collect::<Vec<_>>();let s=(0..r*k/16).map(|i|0x28+((i+e)%24) as u8).collect::<Vec<_>>();raws.push((raw(b,&w)?,raw(b,&s)?));hosts.push((w,s));}
    let shape=|e:usize,op,m:usize|Shape {op,rows:r as u32,cols:k as u32,block_rows:1,block_cols:16,tokens:m as u32,global:0.75+e as f32/8.0,activation:1.5-e as f32/16.0,bound:0.0};
    let mut row=0;let mut q=Vec::new();let mut p=Vec::new();
    for (e,&m) in tokens.iter().enumerate() {q.push((shape(e,Op::QuantizeRows,m),None,row));p.push((shape(e,Op::NvRows,m),Some((&raws[e].0,&raws[e].1)),row));row+=m;}
    let lut=b.upload_vector(&expert_ops::decode_lut())?;
    let packed=b.expert_group(&q,&input,k,&lut)?;let got=b.read_vector_values(&b.expert_group(&p,&packed,k/16*17,&lut)?)?;
    let quantized=b.read_vector_values(&packed)?;let mut expected=Vec::new();let mut expected_q=Vec::new();let mut row=0;
    for (e,&m) in tokens.iter().enumerate() {let source=&x[row*k..(row+m)*k];
        let host_q=expert_ops::reference(shape(e,Op::QuantizeRows,m),[source,&[],&[],&[]],&[],&[])?;
        expected.extend(expert_ops::reference(shape(e,Op::NvRows,m),[&host_q,&[],&[],&[]],&hosts[e].0,&hosts[e].1)?);expected_q.extend(host_q);row+=m;}
    equal(&quantized,&expected_q).map_err(|e|format!("expert group quantize: {e}"))?;equal(&got,&expected).map_err(|e|format!("expert group: {e}"))?;
    // FP8 token rows through the table, from BF16-rounded lanes.
    let (fr,fk)=(5usize,33usize);let w=(0..fr*fk).map(|i|((i*53+17)%127) as u8).collect::<Vec<_>>();let scales=(0..3*3).flat_map(|i|(0.25+(i as f32)/16.0).to_le_bytes()).collect::<Vec<_>>();
    let f=Shape {op:Op::Fp8Rows,rows:fr as u32,cols:fk as u32,block_rows:2,block_cols:16,tokens:7,global:1.0,activation:1.0,bound:0.0};
    let source=(0..7*fk).map(|i|if i%9==0 {-0.0} else {(i as i32%23-11) as f32*0.037}).collect::<Vec<_>>();
    let rounded=b.round_bf16_lanes(&b.upload_vector(&source)?)?;let (wo,so)=(raw(b,&w)?,raw(b,&scales)?);
    let got8=b.read_vector_values(&b.expert_group(&[(f,Some((&wo,&so)),0)],rounded.vector(),fk,&lut)?)?;
    equal(&got8,&expert_ops::reference(f,[&source,&[],&[],&[]],&w,&scales)?).map_err(|e|format!("expert group FP8: {e}"))?;
    bits.push_str(&format!("expert_group tokens={tokens:?} sha256={} fp8={} host_bits=true\n",sha256::digest(&floats(&got)?),sha256::digest(&floats(&got8)?)));Ok(())
}
/// Unsafe boundary refusal: an expert group input from another context.
fn boundaries(b:&Backend<'_>,bits:&mut String)->Result<()> {
    let refused=|what:&str,result:std::result::Result<usize,String>,wanted:&str|->Result<()> {match result {
        Err(error) if error.contains(wanted)=>Ok(()),other=>Err(format!("{what} was not refused: {other:?}"))}};
    let lut=b.upload_vector(&expert_ops::decode_lut())?;
    let q=Shape {op:Op::QuantizeRows,rows:1,cols:16,block_rows:1,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0};
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|other| {
        let input=other.upload_vector(&[0.5f32;16])?;other.synchronize()?;
        refused("expert_group_foreign_input",b.expert_group(&[(q,None,0)],&input,16,&lut).map(|v|v.len()),"context")
    })?;
    let own=b.upload_vector(&[0.5f32;16])?;b.read_vector_values(&b.expert_group(&[(q,None,0)],&own,16,&lut)?)?;
    bits.push_str("boundaries expert_group_foreign_input refused=true\n");Ok(())
}
/// Weight-pass scaling on real weight geometry. It sets the group targets.
fn micro_groups(b:&Backend<'_>,text:&mut String)->Result<()> {
    let (r,k)=(1024usize,4096usize);let weight=raw(b,&(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>())?;let scale=raw(b,&vec![0x38;r*k/16])?;
    let fp8=raw(b,&(0..r*k).map(|i|(i%120+1) as u8).collect::<Vec<_>>())?;let fp8s=raw(b,&vec![1.0f32;r.div_ceil(128)*k.div_ceil(128)].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
    let plain=b.upload_vector(&(0..(r/4)*k).map(|i|(i as i32%13-6) as f32/9.1).collect::<Vec<_>>())?;let lut=b.upload_vector(&expert_ops::decode_lut())?;
    b.set_profiling(true);
    for m in [1usize,2,3,4,8,16,32,64,128,256,512] {
        let x=(0..m*k).map(|i|(i as i32%37-18) as f32/8.0).collect::<Vec<_>>();let input=b.upload_vector(&x)?;b.synchronize()?;
        let nv=Shape {op:if m==1 {Op::NvProject} else {Op::NvRows},rows:r as u32,cols:k as u32,tokens:m as u32,block_rows:1,block_cols:16,global:0.75,activation:1.5,bound:0.0};
        let f8=Shape {op:if m==1 {Op::Fp8Project} else {Op::Fp8Rows},block_rows:128,block_cols:128,..nv};
        let mut best=[u128::MAX;6];let rounded=b.round_bf16_lanes(&input)?;
        for _ in 0..3 {
            let before=b.stats();let q=b.expert_operation(Shape {op:Op::QuantizeRows,..nv},[Some(&input),None,None,None],None,None)?;b.synchronize()?;best[0]=best[0].min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());
            let before=b.stats();let out=b.expert_operation(nv,[Some(&q),None,None,None],Some(&weight),Some(&scale))?;b.synchronize()?;best[1]=best[1].min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);
            let group=Shape {op:Op::NvRows,..nv};let before=b.stats();let out=b.expert_group(&[(group,Some((&weight,&scale)),0)],&q,k/16*17,&lut)?;b.synchronize()?;best[4]=best[4].min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);
            let before=b.stats();let out=b.expert_group(&[(Shape {op:Op::Fp8Rows,..f8},Some((&fp8,&fp8s)),0)],rounded.vector(),k,&lut)?;b.synchronize()?;best[5]=best[5].min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);
            let before=b.stats();let out=b.expert_operation(f8,[Some(&input),None,None,None],Some(&fp8),Some(&fp8s))?;b.synchronize()?;best[2]=best[2].min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);
            let mut s=spark_gpu_probe::prompt_ops::Shape::new(spark_gpu_probe::prompt_ops::Op::ProjectRows,m).ok_or("micro rows")?;s.rows=(r/4) as u32;s.cols=k as u32;
            let before=b.stats();let out=b.prompt_operation(s,[Some(&input),None,None,None,None],Some(&plain))?;b.synchronize()?;best[3]=best[3].min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);
        }
        let per=|ns:u128|ns as f64/1e3/m as f64;let macs=(r*k*m) as f64;
        text.push_str(&format!("MICRO_GROUP m={m} quantize_us_per_row={:.3} nv_us_per_row={:.3} nv_gmac_s={:.1} group_gmac_s={:.1} fp8_us_per_row={:.3} fp8_gmac_s={:.1} fp8_group_gmac_s={:.1} plain256_us_per_row={:.3} plain_gmac_s={:.1}\n",per(best[0]),per(best[1]),macs/best[1] as f64,macs/best[4] as f64,per(best[2]),macs/best[2] as f64,macs/best[5] as f64,per(best[3]),(r/4*k*m) as f64/best[3] as f64));
    }b.set_profiling(false);Ok(())
}
pub(super) fn run(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut bits=String::new();let mut timing=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,16<<30,|b| {
        for repeat in 1..=3 {bits.push_str(&format!("repeat={repeat}\n"));for m in 1..=4 {for k in [16,32,96] {packed(b,shape(Op::NvRows,33,k,m),&mut bits)?;}
            packed(b,shape(Op::Fp8Rows,5,33,m),&mut bits)?;for k in [1,17,128] {plain(b,m as usize,k,&mut bits)?;}plain_unfused(b,m as usize,&mut bits)?;}}
        row_ops(b,&mut bits)?;experts(b,&mut bits)?;boundaries(b,&mut bits)?;
        micro(b,&mut timing)?;micro_fp8(b,&mut timing)?;micro_groups(b,&mut timing)
    })?;
    write(&out.join("BITS"),bits.as_bytes())?;write(&out.join("TIMING"),timing.as_bytes())?;write(&out.join("COMPLETE"),b"PROMPT ROW DEVICE PASS\n")?;print!("{timing}");println!("PROMPT ROW DEVICE PASS");Ok(())
}
