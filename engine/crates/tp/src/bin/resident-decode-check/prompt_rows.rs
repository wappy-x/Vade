//! Independent scalar-versus-grouped GPU row bits. No checkpoint or policy is selected.
#![forbid(unsafe_code)]
use super::{Result,write,floats,sha256};
use std::path::Path;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Shape,Op}};
fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {let mut v=b.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;b.write_vector_bytes(&mut v,bytes)?;b.synchronize()?;Ok(v)}
fn equal(a:&[f32],b:&[f32])->Result<()> {if a.len()!=b.len() || a.iter().zip(b).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err(format!("grouped prompt output differs from scalar bits lengths={}/{} first={:?}",a.len(),b.len(),a.iter().zip(b).enumerate().find(|(_, (a,b))|a.to_bits()!=b.to_bits()).map(|(at,(a,b))|(at,a.to_bits(),b.to_bits()))));}Ok(())}
fn shape(op:Op,rows:u32,cols:u32,tokens:u32)->Shape {Shape {op,rows,cols,tokens,block_rows:2,block_cols:16,global:0.75,activation:1.5,bound:0.0,role:0}}
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
    let v=|n:usize,seed:usize,scale:f32|(0..n).map(|i|if (i + seed).is_multiple_of(19) {-0.0} else {((i+seed)*31%53) as f32*scale-26.0*scale}).collect::<Vec<_>>();
    let base=|op,m|S::new(op,m).ok_or_else(||"prompt fixture row count".to_string());
    let mut cases:Vec<(S,Vec<Vec<f32>>,Vec<f32>)>=Vec::new();
    let inner=Inner {op:C::Project,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
    for m in [1usize,2,5,33,130] {
        let (h,d,t)=(4usize,16usize,4usize);let w=h*d;
        cases.push((S {rows:7,cols:48,..base(R::ProjectRows,m)?},vec![v(m*48,0,0.031)],v(7*48,3,0.017)));
        // BF16 rows: the same weights rounded, packed two patterns a word. Real shapes of the dropped twins too.
        for (r,k) in [(7usize,48usize),(16384,512),(2048,1536),(4096,128)] {if r>7 && m>2 {continue;}
            let wr=v(r*k,3,0.017).iter().map(|&x|f32::from_bits(x.to_bits().wrapping_add(0x7fff+((x.to_bits()>>16)&1))&0xffff_0000)).collect::<Vec<_>>();
            let wp=wr.as_chunks::<2>().0.iter().map(|p|f32::from_bits((p[0].to_bits()>>16)|(p[1].to_bits()&0xffff_0000))).collect::<Vec<_>>();
            cases.push((S {rows:r as u32,cols:k as u32,..base(R::ProjectRowsBf16,m)?},vec![v(m*k,0,0.031)],wp));}
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
        for (heads,kd,vd) in [(3usize,8usize,4usize),(3,8,8),(2,16,16),(5,4,32),(3,6,2)] {let first=2usize;let total=first+m;let row=heads*(kd+vd);let slots=7usize;let cap=slots.min(total);
        let e=v(total*row,1,0.0625);let q=v(m*heads*kd,3,0.125);let mut selected=vec![-1.0f32;m*(slots+1)];
        for token in 0..m {let context=first+token;let count=context.min(slots);for slot in 0..count {selected[token*(slots+1)+slot]=(context-1-slot) as f32;}selected[token*(slots+1)+slots]=count as f32;}
        let att=S {heads:heads as u32,dim:kd as u32,taps:vd as u32,rows:total as u32,cols:cap as u32,slots:slots as u32,..base(R::ScoresRows,m)?};
        let scores=reference(att,[&q,&e,&selected,&[],&[]],&[])?;let prob=reference(S {op:R::SoftmaxRows,..att},[&scores,&selected,&[],&[],&[]],&[])?;
        let mut order=Vec::new();for token in 0..m {let row=&selected[token*(slots+1)..(token+1)*(slots+1)];for (slot,&id) in row[..row[slots] as usize].iter().enumerate() {order.push((id as u32,token as u32,slot as u32));}}order.sort_unstable();
        cases.push((S {first:order.len() as u32,..att},vec![q.clone(),e.clone(),selected.clone(),order.iter().flat_map(|o|[o.0 as f32,o.1 as f32,o.2 as f32]).collect()],vec![]));
        let mut map=vec![-1.0f32;total*m];for o in &order {map[o.0 as usize*m+o.1 as usize]=o.2 as f32;}
        cases.push((S {op:R::ScoresMap,..att},vec![q.clone(),e.clone(),selected.clone(),map],vec![]));
        // A table shorter than its tokens: only the first token has a live slot.
        if m>1 {let mut short=vec![-1.0f32;m*(slots+1)];short[0]=0.0;for token in 0..m {short[token*(slots+1)+slots]=if token==0 {1.0} else {0.0};}
            cases.push((S {first:1,..att},vec![q.clone(),e.clone(),short,vec![0.0,0.0,0.0]],vec![]));}
        cases.push((att,vec![q.clone(),e.clone(),selected.clone()],vec![]));cases.push((S {op:R::SoftmaxRows,..att},vec![scores,selected.clone()],vec![]));cases.push((S {op:R::ValuesRows,..att},vec![prob.clone(),e.clone(),selected.clone()],vec![]));
        // The same rows packed: BF16-round every word (as producers leave them), then two patterns a word.
        // All geometries here have even kd and vd, so every one has a packed form.
        let er=e.iter().map(|&x|f32::from_bits(x.to_bits().wrapping_add(0x7fff+((x.to_bits()>>16)&1))&0xffff_0000)).collect::<Vec<_>>();
        let ep=er.as_chunks::<2>().0.iter().map(|p|f32::from_bits((p[0].to_bits()>>16)|(p[1].to_bits()&0xffff_0000))).collect::<Vec<_>>();
        let mut pmap=vec![-1.0f32;total*m];for o in &order {pmap[o.0 as usize*m+o.1 as usize]=o.2 as f32;}
        cases.push((S {op:R::ScoresMapPacked,..att},vec![q.clone(),ep.clone(),selected.clone(),pmap],vec![]));
        cases.push((S {op:R::ScoresRowsPacked,first:order.len() as u32,..att},vec![q.clone(),ep.clone(),selected.clone(),order.iter().flat_map(|o|[o.0 as f32,o.1 as f32,o.2 as f32]).collect()],vec![]));
        cases.push((S {op:R::ScoresRowsPacked,..att},vec![q,ep.clone(),selected.clone()],vec![]));
        cases.push((S {op:R::ValuesRowsPacked,..att},vec![prob,ep,selected],vec![]));}
        let table=(0..m*3).map(|i|((i*7)%(m*2)) as f32).collect::<Vec<_>>();cases.push((S {rows:(m*3) as u32,cols:32,first:(m*2) as u32,..base(R::PermuteRows,1)?},vec![v(m*2*32,6,0.1),table],vec![]));
    }
    // A fused multiply-add gives a nonzero first output for these operands.
    let a=f32::from_bits(0x3f800001);let c=f32::from_bits(0x3f7ffffe);
    for m in [1usize,3,40] {let shape=S {rows:3,cols:3,..base(R::ProjectRows,m)?};let x=(0..m).flat_map(|_|[1.0,c,1.0]).collect::<Vec<_>>();
        let w=[-1.0,a,0.0,16777216.0,1.0,-16777216.0,-16777216.0,16777216.0,1.0];
        let got=b.read_vector_values(&b.prompt_operation(shape,[Some(&b.upload_vector(&x)?),None,None,None,None],Some(&b.upload_vector(&w)?))?)?;
        if (0..m).any(|token|got[token].to_bits()!=0.0f32.to_bits()) {return Err("prompt plain rows fused a multiply and add".into());}
        cases.push((shape,vec![x],w.to_vec()));
        // The same operands through the path that reads four terms per line.
        let shape=S {rows:3,cols:4,..base(R::ProjectRows,m)?};let x=(0..m).flat_map(|_|[1.0,c,1.0,0.0]).collect::<Vec<_>>();
        let w=[-1.0,a,0.0,0.0,16777216.0,1.0,-16777216.0,0.0,-16777216.0,16777216.0,1.0,0.0];
        let got=b.read_vector_values(&b.prompt_operation(shape,[Some(&b.upload_vector(&x)?),None,None,None,None],Some(&b.upload_vector(&w)?))?)?;
        if (0..m).any(|token|got[token].to_bits()!=0.0f32.to_bits()) {return Err("prompt plain rows fused on the line path".into());}
        cases.push((shape,vec![x],w.to_vec()));}
    for (shape,inputs,weight) in &cases {
        let empty:&[f32]=&[];let host=reference(*shape,std::array::from_fn(|i|inputs.get(i).map_or(empty,Vec::as_slice)),weight)?;
        let owners=inputs.iter().map(|x|if x.is_empty() {Ok(None)} else {b.upload_vector(x).map(Some)}).collect::<Result<Vec<_>>>()?;
        let w=if weight.is_empty() {None} else {Some(b.upload_vector(weight)?)};
        let x=std::array::from_fn(|i|owners.get(i).and_then(Option::as_ref));
        let got=b.read_vector_values(&b.prompt_operation(*shape,x,w.as_ref())?)?;
        equal(&got[..host.len()],&host).map_err(|e|format!("prompt op {:?} tokens={}: {e}",shape.op,shape.tokens))?;
        bits.push_str(&format!("op={:?} inner={:?} tokens={} words={} sha256={} host_bits=true\n",shape.op,shape.inner.op,shape.tokens,host.len(),sha256::digest(&floats(&host)?)));
    }
    // NOFILL-1 (review of ac51ae021): Transpose through the production dispatch on fresh output storage. No clear runs for these shapes (the
    // count of skipped fills rises by one each), and (1,1), (1,16), (16,1) and (1,288) have no tile, so the generic kernel writes the output
    // through raw stores. Bits equal the host reference.
    let skipped=||spark_gpu_probe::prompt_launches::line().split(' ').find_map(|f|f.strip_prefix("prompt_fill_skipped=")?.parse::<u64>().ok()).unwrap_or(0);
    for (r,k) in [(1u32,1u32),(1,16),(16,1),(1,288),(3,6144),(6144,3)] {
        let shape=S {rows:r,cols:k,..base(R::Transpose,1)?};let x=v((r*k) as usize,4,0.1);
        let host=reference(shape,[&x,&[],&[],&[],&[]],&[])?;let input=b.upload_vector(&x)?;b.synchronize()?;
        let before=skipped();let out=b.prompt_operation(shape,[Some(&input),None,None,None,None],None)?;b.synchronize()?;
        if skipped()!=before+1 {return Err(format!("transpose {r}x{k}: the fill was not skipped ({before} to {})",skipped()));}
        equal(&b.read_vector_values(&out)?[..host.len()],&host).map_err(|e|format!("fresh-storage transpose {r}x{k}: {e}"))?;
        bits.push_str(&format!("fresh_transpose rows={r} cols={k} words={} sha256={} host_bits=true\n",host.len(),sha256::digest(&floats(&host)?)));
    }
    Ok(())
}
/// A whole-layer launch equals each expert's own launch, for unequal token groups.
/// NVFP4 rows through the launch the model uses, with uneven token counts of equal parts.
/// These shapes take lane K's tiles; the host reference holds them to the same bits.
fn nv_tiles(b:&Backend<'_>,bits:&mut String)->Result<()> {
    let lut=b.upload_vector(&expert_ops::decode_lut())?;
    for (r,k,tokens) in [(130usize,256usize,vec![1usize,7,13,8,9]),(16,128,vec![1,3]),(129,1024,vec![128,2,65]),(130,256,vec![300,1,700]),(65,128,vec![1024,3])] {
        let n=tokens.iter().sum::<usize>();
        let x=(0..n*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
        let mut raws=Vec::new();let mut hosts=Vec::new();
        for e in 0..tokens.len() {let w=(0..r*k/2).map(|i|((i+e)*17+31) as u8).collect::<Vec<_>>();let s=(0..r*k/16).map(|i|0x28+((i+e)%24) as u8).collect::<Vec<_>>();raws.push((raw(b,&w)?,raw(b,&s)?));hosts.push((w,s));}
        let shape=|e:usize,op,m:usize|Shape {op,rows:r as u32,cols:k as u32,block_rows:1,block_cols:16,tokens:m as u32,global:0.75+e as f32/8.0,activation:1.5-e as f32/16.0,bound:0.0,role:0};
        let mut row=0;let mut q=Vec::new();let mut p=Vec::new();
        for (e,&m) in tokens.iter().enumerate() {q.push((shape(e,Op::QuantizeRows,m),None,row));p.push((shape(e,Op::NvRows,m),Some((&raws[e].0,&raws[e].1)),row));row+=m;}
        let packed=b.expert_group(&q,&input,k,&lut)?;let got=b.read_vector_values(&b.expert_group(&p,&packed,k/16*17,&lut)?)?;
        let mut expected=Vec::new();let mut row=0;
        for (e,&m) in tokens.iter().enumerate() {let source=&x[row*k..(row+m)*k];
            let host_q=expert_ops::reference(shape(e,Op::QuantizeRows,m),[source,&[],&[],&[]],&[],&[])?;
            expected.extend(expert_ops::reference(shape(e,Op::NvRows,m),[&host_q,&[],&[],&[]],&hosts[e].0,&hosts[e].1)?);row+=m;}
        equal(&got,&expected).map_err(|e|format!("nv tiles rows={r} cols={k}: {e}"))?;
        bits.push_str(&format!("nv_tiles rows={r} cols={k} tokens={tokens:?} sha256={} host_bits=true\n",sha256::digest(&floats(&got)?)));
    }
    println!("NV TILE COUNTS {}",spark_gpu_probe::prompt_launches::line());Ok(())
}
fn experts(b:&Backend<'_>,bits:&mut String)->Result<()> {
    let (r,k)=(9usize,48usize);let tokens=[1usize,2,5,1,33];let n=tokens.iter().sum::<usize>();
    let x=(0..n*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
    let mut raws=Vec::new();let mut hosts=Vec::new();
    for e in 0..tokens.len() {let w=(0..r*k/2).map(|i|((i+e)*17+31) as u8).collect::<Vec<_>>();let s=(0..r*k/16).map(|i|0x28+((i+e)%24) as u8).collect::<Vec<_>>();raws.push((raw(b,&w)?,raw(b,&s)?));hosts.push((w,s));}
    let shape=|e:usize,op,m:usize|Shape {op,rows:r as u32,cols:k as u32,block_rows:1,block_cols:16,tokens:m as u32,global:0.75+e as f32/8.0,activation:1.5-e as f32/16.0,bound:0.0,role:0};
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
    let f=Shape {op:Op::Fp8Rows,rows:fr as u32,cols:fk as u32,block_rows:2,block_cols:16,tokens:7,global:1.0,activation:1.0,bound:0.0,role:0};
    let source=(0..7*fk).map(|i|if i%9==0 {-0.0} else {(i as i32%23-11) as f32*0.037}).collect::<Vec<_>>();
    let rounded=b.round_bf16_lanes(&b.upload_vector(&source)?)?;let (wo,so)=(raw(b,&w)?,raw(b,&scales)?);
    let got8=b.read_vector_values(&b.expert_group(&[(f,Some((&wo,&so)),0)],rounded.vector(),fk,&lut)?)?;
    equal(&got8,&expert_ops::reference(f,[&source,&[],&[],&[]],&w,&scales)?).map_err(|e|format!("expert group FP8: {e}"))?;
    // Whole tiles take word loads and hardware decode: every finite code, signed zeros, two block widths.
    let mut fast=String::new();
    for (fk,bc,m) in [(64usize,16u32,1u32),(64,16,7),(64,32,16),(128,64,17),(4096,128,33),(64,16,1024)] {
        let fr=6usize;let w=(0..fr*fk).map(|i|match ((i*53+17)%256) as u8 {0x7f|0xff=>1,code=>code}).collect::<Vec<_>>();let blocks=fk.div_ceil(bc as usize);
        let scales=(0..3*blocks).flat_map(|i|(0.25+(i as f32)/16.0).to_le_bytes()).collect::<Vec<_>>();
        let f=Shape {op:Op::Fp8Rows,rows:fr as u32,cols:fk as u32,block_rows:2,block_cols:bc,tokens:m,global:1.0,activation:1.0,bound:0.0,role:0};
        let source=(0..m as usize*fk).map(|i|if i%9==0 {-0.0} else {(i as i32%23-11) as f32*0.037}).collect::<Vec<_>>();
        let rounded=b.round_bf16_lanes(&b.upload_vector(&source)?)?;let (wo,so)=(raw(b,&w)?,raw(b,&scales)?);
        let got=b.read_vector_values(&b.expert_group(&[(f,Some((&wo,&so)),0)],rounded.vector(),fk,&lut)?)?;
        equal(&got,&expert_ops::reference(f,[&source,&[],&[],&[]],&w,&scales)?).map_err(|e|format!("expert group FP8 tiles cols={fk} block={bc} tokens={m}: {e}"))?;
        fast.push_str(&format!(" {fk}x{bc}x{m}={}",&sha256::digest(&floats(&got)?)[..12]));
    }
    bits.push_str(&format!("expert_group tokens={tokens:?} sha256={} fp8={} fp8_tiles={fast} host_bits=true\n",sha256::digest(&floats(&got)?),sha256::digest(&floats(&got8)?)));Ok(())
}
/// Unsafe boundary refusal: an expert group input from another context.
fn boundaries(b:&Backend<'_>,bits:&mut String)->Result<()> {
    let refused=|what:&str,result:std::result::Result<usize,String>,wanted:&str|->Result<()> {match result {
        Err(error) if error.contains(wanted)=>Ok(()),other=>Err(format!("{what} was not refused: {other:?}"))}};
    let lut=b.upload_vector(&expert_ops::decode_lut())?;
    let q=Shape {op:Op::QuantizeRows,rows:1,cols:16,block_rows:1,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0};
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|other| {
        let input=other.upload_vector(&[0.5f32;16])?;other.synchronize()?;
        refused("expert_group_foreign_input",b.expert_group(&[(q,None,0)],&input,16,&lut).map(|v|v.len()),"context")
    })?;
    let own=b.upload_vector(&[0.5f32;16])?;b.read_vector_values(&b.expert_group(&[(q,None,0)],&own,16,&lut)?)?;
    bits.push_str("boundaries expert_group_foreign_input refused=true\n");Ok(())
}
/// A bad row ID or count must trap on the device. A trap ends the context, so each case
/// runs in its own process, and that process reports the failed cleanup after its PASS line.
/// A trap kills the context, so normal teardown fails after a PASS. Exit here with success, so
/// the exit status says PASS or not: a bad word that was accepted returns an error and exits non-zero.
fn trapped()->Result<()> {use std::io::Write;let _=std::io::stdout().flush();std::process::exit(0)}
pub fn bad_selection(kind:&str)->Result<()> {
    use spark_gpu_probe::prompt_ops::{Op as R,Shape as S};
    // An exchange with a non-finite word: tiles, the word path, and the expert placement.
    if let Some((r,k,place))=match kind {"nan-tile"=>Some((8usize,4usize,false)),"nan-word"=>Some((6,4,false)),"nan-place"|"nan-unread"=>Some((3,4,true)),_=>None} {
        return with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            let mut x=vec![0.5f32;if place {8} else {r*k}];x[if kind=="nan-unread" {3} else if place {1} else {3}]=f32::INFINITY;let e=b.upload_vector(&x)?;
            let s=if place {S {rows:3,cols:4,dim:2,taps:2,slots:4,first:1,..S::new(R::ExpertPlace,1).ok_or("bad place shape")?}} else {S {rows:r as u32,cols:k as u32,first:1,..S::new(R::Transpose,1).ok_or("bad transpose shape")?}};
            // Without `first` the same value passes, so the refusal is the check itself.
            let open=S {first:0,..s};let table=b.upload_vector(&[0.0f32,1.0,1.0,0.0,2.0,0.0,0.0,1.0,0.0])?;b.synchronize()?;
            let second=place.then_some(&table);
            b.read_vector_values(&b.prompt_operation(open,[Some(&e),second,None,None,None],None)?)?;
            let got=b.prompt_operation(s,[Some(&e),second,None,None,None],None).and_then(|v|b.read_vector_values(&v));
            if got.is_ok() {return Err(format!("{kind} was accepted"));}
            println!("BAD SELECTION TRAP PASS {kind}");trapped()
        });
    }
    let (m,h,d,t,slots)=(1usize,2usize,8usize,8usize,4usize);let row=h*(d+t);
    // Table entries are (row, token, slot). The selection they are checked against is valid.
    // Map words: rows 3 x 1 token. The slot must be live and must hold that row.
    let map=match kind {"map-dead"=>Some([3.0f32,-1.0,-1.0]),"map-row"=>Some([1.0,-1.0,-1.0]),_=>None};
    let table=match kind {"table-row"=>Some([1.0f32,0.0,0.0]),"table-token"=>Some([0.0,1.0,0.0]),"table-slot"=>Some([0.0,0.0,3.0]),_=>None};
    let (selected,op)=match kind {
        "id-scores"=>([0.0f32,1.0,3.0,2.0,4.0],R::ScoresRows),"id-values"=>([0.0,1.0,3.0,2.0,4.0],R::ValuesRows),
        "count-scores"=>([0.0,1.0,2.0,0.0,5.0],R::ScoresRows),"count-values"=>([0.0,1.0,2.0,0.0,5.0],R::ValuesRows),
        _ if table.is_some()=>([0.0,1.0,2.0,0.0,3.0],R::ScoresRows),
        _ if map.is_some()=>([0.0,1.0,2.0,0.0,3.0],R::ScoresMap),
        _=>return Err("bad selection case".into()),
    };
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let e=b.upload_vector(&vec![0.5f32;3*row])?;
        let first=if op!=R::ValuesRows {b.upload_vector(&vec![0.25f32;m*h*d])?} else {b.upload_vector(&vec![0.25f32;m*h*slots])?};
        let selected=b.upload_vector(&selected)?;let order=table.or(map).map(|t|b.upload_vector(&t)).transpose()?;b.synchronize()?;
        let s=S {heads:h as u32,dim:d as u32,taps:t as u32,rows:3,cols:slots as u32,slots:slots as u32,first:u32::from(table.is_some()),..S::new(op,m).ok_or("bad selection shape")?};
        let got=b.prompt_operation(s,[Some(&first),Some(&e),Some(&selected),order.as_ref(),None],None).and_then(|v|b.read_vector_values(&v));
        if got.is_ok() {return Err(format!("{kind} was accepted"));}
        println!("BAD SELECTION TRAP PASS {kind}");trapped()
    })
}
/// Attention value rows at real geometry: 2051 selected rows of a 64-head layer.
fn micro_values(b:&Backend<'_>,text:&mut String)->Result<()> {
    use spark_gpu_probe::prompt_ops::{Op as R,Shape as S};
    let (m,h,d,slots)=(64usize,64usize,256usize,2051usize);let k=slots;
    b.set_profiling(true);
    for t in [252usize,256] {
        let row=h*(d+t);let e=(0..slots*row).map(|i|((i*31)%53) as f32*0.01-0.26).collect::<Vec<_>>();let e=b.upload_vector(&e)?;
        let p=(0..m*h*k).map(|i|((i*7)%97) as f32*1e-5).collect::<Vec<_>>();let p=b.upload_vector(&p)?;
        let mut selected=vec![0.0f32;m*(slots+1)];for token in 0..m {for slot in 0..slots {selected[token*(slots+1)+slot]=slot as f32;}selected[token*(slots+1)+slots]=slots as f32;}
        let selected=b.upload_vector(&selected)?;b.synchronize()?;
        let s=S {heads:h as u32,dim:d as u32,taps:t as u32,rows:slots as u32,cols:k as u32,slots:slots as u32,..S::new(R::ValuesRows,m).ok_or("micro values shape")?};
        let mut best=u128::MAX;for _ in 0..3 {let before=b.stats();let out=b.prompt_operation(s,[Some(&p),Some(&e),Some(&selected),None,None],None)?;b.synchronize()?;best=best.min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);}
        text.push_str(&format!("MICRO_VALUES m={m} heads={h} taps={t} slots={slots} us={:.1} gmac_s={:.1}\n",best as f64/1e3,(m*h*t*slots) as f64/best as f64));
    }
    // 510 columns stay on prompt_rows; the others run in tiles.
    for (m,k) in [(m,510usize),(m,512),(256,512)] {
        let r=16384usize;let w=(0..r*k).map(|i|((i*17)%89) as f32*0.001-0.04).collect::<Vec<_>>();let w=b.upload_vector(&w)?;
        let x=(0..m*k).map(|i|((i*7)%97) as f32*0.01-0.4).collect::<Vec<_>>();let x=b.upload_vector(&x)?;b.synchronize()?;
        let s=S {rows:r as u32,cols:k as u32,..S::new(R::ProjectRows,m).ok_or("micro plain shape")?};
        let mut best=u128::MAX;for _ in 0..3 {let before=b.stats();let out=b.prompt_operation(s,[Some(&x),None,None,None,None],Some(&w))?;b.synchronize()?;best=best.min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);}
        text.push_str(&format!("MICRO_PLAIN m={m} rows={r} cols={k} us={:.1} gmac_s={:.1}\n",best as f64/1e3,(m*r*k) as f64/best as f64));
    }
    // Tokens that select different rows, with threads by slot, by row, and in tiles over all pairs.
    for (m,rows) in [(m,3000usize),(256,3000),(256,8192),(256,16384)] {
        let (d,t)=(256usize,256usize);let row=h*(d+t);let e=(0..rows*row).map(|i|((i*31)%53) as f32*0.01-0.26).collect::<Vec<_>>();let e=b.upload_vector(&e)?;
        let q=(0..m*h*d).map(|i|((i*7)%97) as f32*0.01-0.4).collect::<Vec<_>>();let q=b.upload_vector(&q)?;
        let mut selected=vec![0.0f32;m*(slots+1)];let mut order=Vec::new();
        for token in 0..m {let mut ids=(0..slots).map(|slot|((slot*13+token*29)%rows) as u32).collect::<Vec<_>>();ids.sort_unstable();ids.dedup();
            for (slot,&id) in ids.iter().enumerate() {selected[token*(slots+1)+slot]=id as f32;order.push((id,token as u32,slot as u32));}selected[token*(slots+1)+slots]=ids.len() as f32;}
        order.sort_unstable();let table=b.upload_vector(&order.iter().flat_map(|o|[o.0 as f32,o.1 as f32,o.2 as f32]).collect::<Vec<_>>())?;
        let selected=b.upload_vector(&selected)?;b.synchronize()?;
        let s=S {heads:h as u32,dim:d as u32,taps:t as u32,rows:rows as u32,cols:k as u32,slots:slots as u32,..S::new(R::ScoresRows,m).ok_or("micro scores shape")?};
        let mut map=vec![-1.0f32;rows*m];for o in &order {map[o.0 as usize*m+o.1 as usize]=o.2 as f32;}let map=b.upload_vector(&map)?;b.synchronize()?;
        let mut first=None;
        for (name,shape,fourth) in [("slot",s,None),("row",S {first:order.len() as u32,..s},Some(&table)),("tile",S {op:R::ScoresMap,..s},Some(&map))] {
            let mut best=u128::MAX;let mut bits=Vec::new();for _ in 0..3 {let before=b.stats();let out=b.prompt_operation(shape,[Some(&q),Some(&e),Some(&selected),fourth,None],None)?;b.synchronize()?;best=best.min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());bits=b.read_vector_values(&out)?;drop(out);}
            let bits=bits.iter().map(|v|v.to_bits()).collect::<Vec<_>>();if *first.get_or_insert_with(||bits.clone())!=bits {return Err(format!("scores order {name} changed bits"));}
            text.push_str(&format!("MICRO_SCORES_MIXED order={name} m={m} rows={rows} pairs={} us={:.1} gmac_s={:.1}\n",order.len(),best as f64/1e3,(order.len()*h*d) as f64/best as f64));
        }
    }
    for (d,t) in [(254usize,258usize),(256,256)] {
        let row=h*(d+t);let e=(0..slots*row).map(|i|((i*31)%53) as f32*0.01-0.26).collect::<Vec<_>>();let e=b.upload_vector(&e)?;
        let q=(0..m*h*d).map(|i|((i*7)%97) as f32*0.01-0.4).collect::<Vec<_>>();let q=b.upload_vector(&q)?;
        let mut selected=vec![0.0f32;m*(slots+1)];for token in 0..m {for slot in 0..slots {selected[token*(slots+1)+slot]=slot as f32;}selected[token*(slots+1)+slots]=slots as f32;}
        let selected=b.upload_vector(&selected)?;b.synchronize()?;
        let s=S {heads:h as u32,dim:d as u32,taps:t as u32,rows:slots as u32,cols:k as u32,slots:slots as u32,..S::new(R::ScoresRows,m).ok_or("micro scores shape")?};
        let mut best=u128::MAX;for _ in 0..3 {let before=b.stats();let out=b.prompt_operation(s,[Some(&q),Some(&e),Some(&selected),None,None],None)?;b.synchronize()?;best=best.min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);}
        text.push_str(&format!("MICRO_SCORES m={m} heads={h} dim={d} slots={slots} us={:.1} gmac_s={:.1}\n",best as f64/1e3,(m*h*d*slots) as f64/best as f64));
    }
    Ok(())
}
/// Weight-pass scaling on real weight geometry. It sets the group targets.
fn micro_groups(b:&Backend<'_>,text:&mut String)->Result<()> {
    let (r,k)=(1024usize,4096usize);let weight=raw(b,&(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>())?;let scale=raw(b,&vec![0x38;r*k/16])?;
    let fp8=raw(b,&(0..r*k).map(|i|(i%120+1) as u8).collect::<Vec<_>>())?;let fp8s=raw(b,&vec![1.0f32;r.div_ceil(128)*k.div_ceil(128)].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
    let plain=b.upload_vector(&(0..(r/4)*k).map(|i|(i as i32%13-6) as f32/9.1).collect::<Vec<_>>())?;let lut=b.upload_vector(&expert_ops::decode_lut())?;
    b.set_profiling(true);
    for m in [1usize,2,3,4,8,16,32,64,128,256,512] {
        let x=(0..m*k).map(|i|(i as i32%37-18) as f32/8.0).collect::<Vec<_>>();let input=b.upload_vector(&x)?;b.synchronize()?;
        let nv=Shape {op:if m==1 {Op::NvProject} else {Op::NvRows},rows:r as u32,cols:k as u32,tokens:m as u32,block_rows:1,block_cols:16,global:0.75,activation:1.5,bound:0.0,role:0};
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
/// The recurrent update at real shape, against the host and for its rate. One extra head
/// size with a small column shows the same bits on an edge of the block kernel.
fn micro_step(b:&Backend<'_>,text:&mut String)->Result<()> {
    use spark_gpu_probe::prompt_ops::{Op as R,Shape as S,reference};
    b.set_profiling(true);
    // Transpose at the real exchange shape: tiles, then one word off, which takes the older path.
    let (r,k)=(32768usize,256usize);let x=(0..r*k).map(|i|i as f32).collect::<Vec<_>>();let mut w=vec![0.0f32];w.extend_from_slice(&x);
    let (aligned,whole)=(b.upload_vector(&x)?,b.upload_vector(&w)?.into_resident_shared()?);let off=whole.resident_range(1..w.len())?;b.synchronize()?;
    let s=S {rows:r as u32,cols:k as u32,..S::new(R::Transpose,1).ok_or("micro transpose shape")?};
    for (name,input) in [("tile",&aligned),("word",&off)] {
        let mut best=u128::MAX;for _ in 0..3 {let before=b.stats();let out=b.prompt_operation(s,[Some(input),None,None,None,None],None)?;b.synchronize()?;best=best.min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());drop(out);}
        text.push_str(&format!("MICRO_TRANSPOSE path={name} rows={r} cols={k} us={:.1} words_per_us={:.0}\n",best as f64/1e3,(r*k) as f64*1e3/best as f64));
    }
    for (m,h,d) in [(256usize,64usize,128usize),(64,64,128),(3,5,4),(7,2,32),(40,3,96),(5,2,160)] {
        let stride=4*d+1;let s=S {heads:h as u32,dim:d as u32,..S::new(R::StepRows,m).ok_or("micro step shape")?};
        let p=(0..m*h*stride).map(|i|((i*7)%97) as f32*0.003-0.1).collect::<Vec<_>>();let state=(0..h*d*d).map(|i|((i*13)%89) as f32*0.002-0.07).collect::<Vec<_>>();
        let decay=reference(S {op:R::StepDecay,..s},[&p,&[],&[],&[],&[]],&[])?;
        let (px,sx,dx)=(b.upload_vector(&p)?,b.upload_vector(&state)?,b.upload_vector(&decay)?);b.synchronize()?;
        let mut best=u128::MAX;let mut got=Vec::new();for _ in 0..3 {let before=b.stats();let out=b.prompt_operation(s,[Some(&px),Some(&sx),Some(&dx),None,None],None)?;b.synchronize()?;best=best.min(b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>());got=b.read_vector_values(&out)?;drop(out);}
        equal(&got,&reference(s,[&p,&state,&decay,&[],&[]],&[])?).map_err(|e|format!("step m={m} h={h} d={d}: {e}"))?;
        text.push_str(&format!("MICRO_STEP m={m} heads={h} dim={d} us={:.1} gmac_s={:.1}\n",best as f64/1e3,(2*m*h*d*d) as f64/best as f64));
    }
    b.set_profiling(false);Ok(())
}
/// Review K-3. The line, tile and word-load paths need aligned operands. Shifted by one word
/// the same shape takes the older path, on the device, so the two must give equal bits. That
/// holds for non-finite inputs and NaN weight codes too, where no host reference is exact.
fn path_agreement(b:&Backend<'_>)->Result<()> {
    use spark_gpu_probe::prompt_ops::{Op as R,Shape as S};
    let bare=|v:&[f32]|b.upload_vector(v);
    let shifted=|v:&[f32]|{let mut w=vec![0.0f32];w.extend_from_slice(v);b.upload_vector(&w)?.into_resident_shared()?.resident_range(1..w.len())};
    let same=|name:&str,a:Vec<f32>,c:Vec<f32>|if a.len()!=c.len() || a.iter().zip(&c).any(|(a,c)|a.to_bits()!=c.to_bits()) {Err(format!("path bits differ: {name}"))} else {Ok(())};
    let special=|i:usize,scale:f32|match i%11 {0=>f32::NAN,1=>f32::INFINITY,2=>f32::NEG_INFINITY,3=>-0.0,4=>1e-40,_=>((i%37) as f32-18.0)*scale};
    for (name,bad) in [("finite",false),("non-finite",true)] {
        let value=|n:usize,scale:f32|(0..n).map(|i|if bad {special(i,scale)} else {((i%37) as f32-18.0)*scale}).collect::<Vec<_>>();
        // Plain rows: one word off takes prompt_rows; aligned takes the tile.
        for (r,k,m) in [(70usize,64usize,67usize),(70,64,1024)] {let (x,w)=(value(m*k,0.125),value(r*k,0.0625));let s=S {rows:r as u32,cols:k as u32,..S::new(R::ProjectRows,m).ok_or("agreement shape")?};
        let tile=b.prompt_operation(s,[Some(&bare(&x)?),None,None,None,None],Some(&bare(&w)?))?;let old=b.prompt_operation(s,[Some(&shifted(&x)?),None,None,None,None],Some(&bare(&w)?))?;
        same(&format!("plain {name} tokens={m}"),b.read_vector_values(&tile)?,b.read_vector_values(&old)?)?;}
        // Scores with and without the table, and Values. Row-major key and value rows.
        for m in [5usize,1024] {let (h,d,t,slots,rows)=(3usize,16usize,16usize,6usize,9usize);let row=h*(d+t);
        let q=value(m*h*d,0.125);let e=value(rows*row,0.0625);let p=value(m*h*slots,0.25);let mut selected=vec![0.0f32;m*(slots+1)];let mut order=Vec::new();
        for token in 0..m {for slot in 0..slots {let id=(slot*2+token)%rows;selected[token*(slots+1)+slot]=id as f32;order.push((id as u32,token as u32,slot as u32));}selected[token*(slots+1)+slots]=slots as f32;}
        order.sort_unstable();let table=order.iter().flat_map(|o|[o.0 as f32,o.1 as f32,o.2 as f32]).collect::<Vec<_>>();
        let scores=S {heads:h as u32,dim:d as u32,taps:t as u32,rows:rows as u32,cols:slots as u32,slots:slots as u32,..S::new(R::ScoresRows,m).ok_or("agreement shape")?};
        let selected=bare(&selected)?;let table=bare(&table)?;
        let old=b.read_vector_values(&b.prompt_operation(scores,[Some(&bare(&q)?),Some(&shifted(&e)?),Some(&selected),None,None],None)?)?;
        same(&format!("scores {name}"),b.read_vector_values(&b.prompt_operation(scores,[Some(&bare(&q)?),Some(&bare(&e)?),Some(&selected),None,None],None)?)?,old.clone())?;
        let ordered=S {first:order.len() as u32,..scores};
        same(&format!("scores table {name}"),b.read_vector_values(&b.prompt_operation(ordered,[Some(&bare(&q)?),Some(&bare(&e)?),Some(&selected),Some(&table),None],None)?)?,old.clone())?;
        let mut pairs=vec![-1.0f32;rows*m];for o in &order {pairs[o.0 as usize*m+o.1 as usize]=o.2 as f32;}let pairs=bare(&pairs)?;let mapped=S {op:R::ScoresMap,..scores};
        same(&format!("scores tile {name}"),b.read_vector_values(&b.prompt_operation(mapped,[Some(&bare(&q)?),Some(&bare(&e)?),Some(&selected),Some(&pairs),None],None)?)?,old.clone())?;
        same(&format!("scores map shifted {name}"),b.read_vector_values(&b.prompt_operation(mapped,[Some(&bare(&q)?),Some(&shifted(&e)?),Some(&selected),Some(&pairs),None],None)?)?,old.clone())?;
        same(&format!("scores table shifted {name}"),b.read_vector_values(&b.prompt_operation(ordered,[Some(&bare(&q)?),Some(&shifted(&e)?),Some(&selected),Some(&table),None],None)?)?,old)?;
        let values=S {op:R::ValuesRows,..scores};
        same(&format!("values {name}"),b.read_vector_values(&b.prompt_operation(values,[Some(&bare(&p)?),Some(&bare(&e)?),Some(&selected),None,None],None)?)?,b.read_vector_values(&b.prompt_operation(values,[Some(&bare(&p)?),Some(&shifted(&e)?),Some(&selected),None,None],None)?)?)?;
        }
    }
    // Transpose: tiles against the older path, at edge sizes.
    for (r,k) in [(4usize,4usize),(8,4),(4,12),(64,256),(1024,64)] {
        let x=(0..r*k).map(|i|f32::from_bits(0x3f80_0000^(i as u32).wrapping_mul(2654435761))).collect::<Vec<_>>();let s=S {rows:r as u32,cols:k as u32,..S::new(R::Transpose,1).ok_or("agreement shape")?};
        same(&format!("transpose {r}x{k}"),b.read_vector_values(&b.prompt_operation(s,[Some(&bare(&x)?),None,None,None,None],None)?)?,b.read_vector_values(&b.prompt_operation(s,[Some(&shifted(&x)?),None,None,None,None],None)?)?)?;
    }
    // FP8: every code, NaN codes 0x7f and 0xff included, word loads against the table path.
    let lut=b.upload_vector(&expert_ops::decode_lut())?;let (fr,fk,m)=(8usize,64usize,7u32);
    let codes=(0..fr*fk).map(|i|((i*53+17)%256) as u8).collect::<Vec<_>>();let blocks=fk/16;let scales=(0..4*blocks).flat_map(|i|(0.25+(i as f32)/16.0).to_le_bytes()).collect::<Vec<_>>();
    let f=Shape {op:Op::Fp8Rows,rows:fr as u32,cols:fk as u32,block_rows:2,block_cols:16,tokens:m,global:1.0,activation:1.0,bound:0.0,role:0};
    let source=(0..m as usize*fk).map(|i|((i as i32)%23-11) as f32*0.037).collect::<Vec<_>>();let rounded=b.round_bf16_lanes(&b.upload_vector(&source)?)?;let scale=raw(b,&scales)?;
    let mut off=vec![0u8;4];off.extend_from_slice(&codes);let off=raw(b,&off)?.into_resident_shared()?;let off=off.resident_range(1..off.len())?;
    let words=b.read_vector_values(&b.expert_group(&[(f,Some((&raw(b,&codes)?,&scale)),0)],rounded.vector(),fk,&lut)?)?;
    let table=b.read_vector_values(&b.expert_group(&[(f,Some((&off,&scale)),0)],rounded.vector(),fk,&lut)?)?;
    same("fp8 nan codes",words,table)
}
/// A group's peak covers only that group, and the whole run's peak stays true.
fn group_peak(b:&Backend<'_>)->Result<()> {
    b.synchronize()?;let big=b.vector(1<<26)?;drop(big);b.synchronize()?;let start=b.accounted_bytes();
    let whole=b.begin_group_peak();
    let small=b.vector(1<<16)?;let held=b.accounted_bytes();drop(small);
    let group=b.end_group_peak(whole);
    if group<held || group>start.max(held)+(1<<20) || whole<(1<<28) || b.peak_accounted_bytes()<whole {return Err(format!("group peak differs: start={start} held={held} group={group} whole={whole}"));}
    Ok(())
}
pub(super) fn run(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut bits=String::new();let mut timing=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,16<<30,|b| {
        for repeat in 1..=3 {bits.push_str(&format!("repeat={repeat}\n"));for m in 1..=4 {for k in [16,32,96] {packed(b,shape(Op::NvRows,33,k,m),&mut bits)?;}
            packed(b,shape(Op::Fp8Rows,5,33,m),&mut bits)?;for k in [1,17,128] {plain(b,m as usize,k,&mut bits)?;}plain_unfused(b,m as usize,&mut bits)?;}}
        row_ops(b,&mut bits)?;experts(b,&mut bits)?;nv_tiles(b,&mut bits)?;boundaries(b,&mut bits)?;group_peak(b)?;path_agreement(b)?;
        micro_step(b,&mut timing)?;micro(b,&mut timing)?;micro_fp8(b,&mut timing)?;micro_groups(b,&mut timing)?;micro_values(b,&mut timing)
    })?;
    write(&out.join("BITS"),bits.as_bytes())?;write(&out.join("TIMING"),timing.as_bytes())?;write(&out.join("COMPLETE"),b"PROMPT ROW DEVICE PASS\n")?;print!("{timing}");println!("PROMPT ROW DEVICE PASS");Ok(())
}
