//! One box, no model: the selection row known on the host against the real selection kernel's
//! readback, bit for bit, plus the wait and download counts of a step's 11 selection rounds.
use super::*;
use spark_gpu_probe::{causal_ops::{Op,Shape as CausalShape},driver::Backend,prompt_ops::{Op as PromptOp,Shape},select_admission::known_selection};
use spark_tp::{bounded_decoder::{Compute,Device,Value},row_split::{Exchange,Factory,Split}};
use spark_serve::{Engine,Session};
use engine_model::{plan::{description::{TokenMixer,TextDescription},layer::Topology},safetensors::Dtype};
#[allow(dead_code)]
#[path="../../../tests/support/attention_checkpoint.rs"] mod attention;
/// The service's exchange on one box: one rank, no peer.
struct OneRank;
impl Exchange for OneRank {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],_:usize)->Result<Vec<f32>> {Ok(local.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
/// `latents` latent layers after the first KDA layer, with this pool budget. Weights are small, not a model.
fn checkpoint(latents:usize,budget:u64,pool:u64,capacity:u64)->attention::Checkpoint {
    attention::Checkpoint::build_with(Dtype::Bf16,true,capacity,false,|d:&mut TextDescription| {
        let mut latent=d.layers[1].clone();if let TokenMixer::LatentAttention(a)=&mut latent.mixer {a.indexer.token_budget=budget;a.indexer.pool=pool;}
        d.layers.truncate(1);for _ in 0..latents {d.layers.push(latent.clone());}
    })
}
#[derive(Default)]
struct Step {dtoh:u128,waits:u128,drains:u128,selection:[u64;2],new_buffers:usize,new_bytes:usize,logits:Vec<u32>}
fn prompt_tokens(length:usize)->Vec<u32> {(0..length).map(|i|1+(i as u32*5)%15).collect()}
/// Prefill, then plain decode steps, through Factory, Split and the decoder over a real Device.
fn served_arm<'b,'d>(d:&Device<'b,'d>,b:&Backend<'d>,c:&attention::Checkpoint,capacity:usize,shortcut:bool,prompt:&[u32],steps:usize)->Result<Vec<Step>> {
    d.set_selection_shortcut(shortcut);
    let split=Split::new(d,OneRank)?;let service=Factory::new(&c.admission,split,capacity,vec![0])?;service.negotiate_available_bytes(1usize<<40)?;
    let mut out=Vec::new();{
        let mut session=service.session()?;let mut token=1u32;
        if !prompt.is_empty() {let logits=session.prefill(prompt)?;out.push(Step {logits:bits(&logits),..Default::default()});}
        for at in 0..steps {
            let (before,selection,pool)=(b.stats(),d.selection_counts(),b.vector_pool_stats());let logits=session.step(token)?;
            let (after,done,grown)=(b.stats().since(before),d.selection_counts(),b.vector_pool_stats());
            out.push(Step {dtoh:after.dtoh_calls,waits:after.stream_wait_calls,drains:after.budget_drains,selection:[done[0]-selection[0],done[1]-selection[1]],
                new_buffers:grown.allocated_count-pool.allocated_count,new_bytes:grown.allocated_bytes-pool.allocated_bytes,logits:bits(&logits)});
            token=1+(at as u32*5)%15;
        }
    }service.shutdown()?;d.set_selection_shortcut(true);Ok(out)
}
/// Both arms, logits bit for bit, on the small boundary checkpoint.
fn served_boundary<'b,'d>(d:&Device<'b,'d>,b:&Backend<'d>,lines:&mut String)->Result<usize> {
    let mut bytes=0;let c=checkpoint(11,4,2,8);
    let readback=served_arm(d,b,&c,8,false,&[],8)?;let known=served_arm(d,b,&c,8,true,&[],8)?;
    for (at,(a,k)) in readback.iter().zip(&known).enumerate() {
        if a.logits!=k.logits {return Err(format!("served logits differ at boundary step {at}"));}bytes+=a.logits.len()*4;
        let expected=if at<5 {11} else {0};if k.selection[1]!=expected || a.selection[1]!=0 {return Err(format!("boundary step {at}: readback {:?} known {:?}",a.selection,k.selection));}
    }
    let line=format!("SCREEN served_boundary steps={} logits_equal=1 known_at_contexts=1..5 real_at_contexts=6..8\n",readback.len());print!("{line}");lines.push_str(&line);Ok(bytes)
}
/// One arm at one prompt length, in a fresh backend so its peak is its own.
fn length_arm(c:&attention::Checkpoint,budget:usize,capacity:usize,shortcut:bool,prompt:&[u32],steps:usize)->Result<(Vec<Step>,usize)> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,budget,|b| {
        b.configure_vector_pool(1<<30);let d=Device::new(b);let out=served_arm(&d,b,c,capacity,shortcut,prompt,steps)?;Ok((out,b.peak_accounted_bytes()))
    })
}
/// Peak bytes and new buffers per prompt length, both arms. Small weights: the shape of the
/// rule, not the model's bytes. The prompt near 2,040 crosses 2,051, and runs once at the
/// readback arm's own peak as its budget, so only drains can make it fit.
fn lengths(lines:&mut String)->Result<usize> {
    let mut bytes=0;let capacity=2100usize;let c=checkpoint(11,2048,4,capacity as u64);
    let unlimited=1usize<<34;
    for (length,steps) in [(15usize,10usize),(127,10),(1040,10),(2040,24)] {
        let prompt=prompt_tokens(length);
        let (readback,readback_peak)=length_arm(&c,unlimited,capacity,false,&prompt,steps)?;
        let (known,known_peak)=length_arm(&c,unlimited,capacity,true,&prompt,steps)?;
        for (at,(a,k)) in readback.iter().zip(&known).enumerate() {if a.logits!=k.logits {return Err(format!("length {length} logits differ at {at}"));}bytes+=a.logits.len()*4;}
        for (at,k) in known.iter().enumerate().skip(1) {
            let context=length+at;let expected=if context<=2051 {22} else {0};
            if k.selection[1]!=expected {return Err(format!("length {length} context {context}: known selections {} expected {expected}",k.selection[1]));}
        }
        let new=|arm:&[Step],range:std::ops::Range<usize>|arm[range].iter().map(|s|s.new_buffers).sum::<usize>();
        let later=known.len()-3;
        let line=format!("SCREEN length={length} peak_bytes readback={readback_peak} known={known_peak} new_buffers_steps_1_2 readback={} known={} new_buffers_from_step_3 readback={} known={} new_bytes_from_step_3 readback={} known={} waits_last readback={} known={} dtoh_last readback={} known={} selection_known_last={} steps={steps} logits_equal=1\n",
            new(&readback,1..3),new(&known,1..3),new(&readback,3..readback.len()),new(&known,3..known.len()),readback[3..].iter().map(|s|s.new_bytes).sum::<usize>(),known[3..].iter().map(|s|s.new_bytes).sum::<usize>(),readback[later+2].waits,known[later+2].waits,readback[later+2].dtoh,known[later+2].dtoh,known[later+2].selection[1]);
        print!("{line}");lines.push_str(&line);
        if length==2040 {
            let budget=readback_peak;let (held,held_peak)=length_arm(&c,budget,capacity,true,&prompt,steps)?;
            for (at,(a,k)) in readback.iter().zip(&held).enumerate() {if a.logits!=k.logits {return Err(format!("budgeted logits differ at {at}"));}bytes+=a.logits.len()*4;}
            let drains=held.iter().map(|s|s.drains).sum::<u128>();
            let crossed=held.iter().enumerate().skip(1).filter(|(at,s)|length+at>2051 && s.selection[1]==0).count();
            if crossed==0 {return Err("budgeted run did not cross 2,051".into());}
            let line=format!("SCREEN budgeted length=2040 budget={budget} peak_bytes={held_peak} budget_drains={drains} steps_above_2051={crossed} logits_equal=1\n");print!("{line}");lines.push_str(&line);
        }
    }Ok(bytes)
}
/// The budget refuses, a buffer is deferred, and a drain makes room. With nothing deferred it stays refused.
fn budget_drain(lines:&mut String)->Result<()> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        let words=16usize<<20;let bytes=words*4;b.set_allocation_budget(b.accounted_bytes()+bytes+bytes/2)?;let queue=b.resident_queue()?;b.enable_resident_launches()?;
        let mut first=b.vector(words)?;b.zero_vector(&mut first)?;drop(first);
        let before=b.stats();let mut second=b.vector(words)?;let drains=b.stats().since(before).budget_drains;
        if drains!=1 {return Err(format!("the budget refusal drained {drains} times, not once"));}
        b.zero_vector(&mut second)?;if b.read_vector_values(&second)?.iter().any(|v|v.to_bits()!=0) {return Err("the retried buffer differs".into());}
        let before=b.stats();let refused=b.vector(words).is_err();let again=b.stats().since(before).budget_drains;
        if !refused || again!=0 {return Err(format!("with nothing deferred: refused {refused}, drains {again}"));}
        drop(second);drop(queue);
        let line="SCREEN budget_drain refusal_drained=1 retried_buffer_equal=1 refused_with_nothing_deferred=1 drains=0\n".to_string();print!("{line}");lines.push_str(&line);Ok(())
    })
}
/// A NaN through a real step, on the decoder over &Split: Gather writes zero rows under the source's
/// lineage and ExpandKv keeps their projections. The retry must equal a run that never saw the NaN.
fn model_nan_retry(backend:&Backend<'_>,lines:&mut String)->Result<usize> {
    let c=checkpoint(11,2048,4,64);let tokens=[1u32,3,2,5,4,7,6,9,8];let at=7usize;let limit=1usize<<28;
    #[derive(Clone,Copy,PartialEq)] enum Arm {Clean,Fixed,Control}
    let run=|arm:Arm|->Result<(Vec<Vec<u32>>,u64,usize)> {
        let d=Device::new(backend);if arm==Arm::Control {d.screen_keep_unverified(true);}
        let split=Split::new(&d,OneRank)?;let decoder=spark_tp::bounded_decoder::Decoder::new(&c.admission,&split,64)?;let mut state=decoder.zero_state()?;let mut out=Vec::new();
        for token in &tokens[..at] {state=decoder.step(*token,&state,|_,_|Ok(()))?.2;}
        // Every old row now misses, as after an eviction, so the refused step keeps them.
        d.clear_projection_memo()?;d.set_projection_memo_limit(limit)?;let (mut kept,mut left)=(0,0);
        if arm!=Arm::Clean {
            let before=d.memo_counts().3;d.screen_nan_selection(3);
            match decoder.step(tokens[at],&state,|_,_|Ok(())) {Err(error) if error.contains(NAN_ERROR)=>{},Err(error)=>return Err(format!("wrong NaN error: {error}")),Ok(_)=>return Err("a NaN score reached a result".into())}
            kept=d.memo_counts().3-before;left=d.memo_counts().0;
        }
        for token in &tokens[at..] {let (logits,_,next)=decoder.step(*token,&state,|_,_|Ok(()))?;out.push(bits(&logits));state=next;}
        Ok((out,kept,left))
    };
    let (clean,_,_)=run(Arm::Clean)?;let (fixed,kept,left)=run(Arm::Fixed)?;let (control,control_kept,control_left)=run(Arm::Control)?;
    if kept==0 {return Err("the refused step kept no row, so the test proves nothing".into());}
    // Rows of layers whose count already passed a check are clean and may stay. The refused layer's may not.
    if left>=control_left {return Err(format!("the refusal discarded nothing: {left} bytes against the control's {control_left}"));}
    if fixed!=clean {return Err("the retry differs from a run that never saw the NaN".into());}
    if control==clean {return Err(format!("the control did not fail: left {control_left}, so this test cannot catch poison"));}
    let line=format!("SCREEN model_nan_retry rows_kept_in_refused_step={kept} memo_bytes_after_refusal={left} retry_equal_to_clean=1 control_rows_kept={control_kept} control_memo_bytes={control_left} control_retry_differs=1\n");print!("{line}");lines.push_str(&line);
    Ok(fixed.iter().map(|v|v.len()*4).sum())
}
/// NaN, Gather, ExpandKv's keep, refusal, clean retry: the retry finds no row from the refused selection.
fn memo_window<'b,'d>(b:&'b Backend<'d>,scope:bool)->Result<()> {
    let d=Device::new(b);let width=32768usize;let dim=512usize;let context=8usize;let s=shape(context);let row=known_selection(s).ok_or("memo case needs a known row")?;
    d.set_projection_memo_limit(2*spark_tp::bounded_decoder::SLAB_ROWS*width*4)?;
    let source=d.upload(&vec![0.25;context*dim])?;let op=CausalShape {op:Op::Gather,heads:0,dim:dim as u32,taps:0,rows:context as u32,cols:context as u32,epsilon:0.0,bound:0.0};
    let clean=d.upload(&[0.5,0.25])?;let bad=d.upload(&[f32::NAN,0.25])?;let logits=d.upload(&[1.0])?;
    // A row kept before the window, under its own lineage.
    let (older_in,older_out)=(d.upload(&vec![0.125;dim])?,d.upload(&vec![7.0;width])?);d.remember_projections("kv",true,&[(&older_in,&older_out)])?;
    let window=|scores:&Value<'b,'d>|->Result<Value<'b,'d>> {
        let (_,ids,count)=select(&d,s,scores)?;d.select_known(&ids,&count,&row)?;let view=d.range(&ids,0,context)?;
        let gathered=d.operation(op,[Some(&source),Some(&view),Some(&count),None],&[])?;d.range(&gathered,0,dim)
    };
    let kept=d.upload(&vec![9.0;width])?;
    let refused=if scope {
        d.work(|| {let tainted=window(&bad)?;d.remember_projections("kv",true,&[(&tainted,&kept)])?;
            if d.projection_origin("kv",&tainted,true)?.is_none() {return Err("window row was not kept".to_string());}Ok(())})
    } else {
        let tainted=window(&bad)?;d.remember_projections("kv",true,&[(&tainted,&kept)])?;
        if d.projection_origin("kv",&tainted,true)?.is_none() {return Err("window row was not kept".into());}
        d.read(&logits).map(|_|())
    };
    match refused {Err(error) if error.contains(NAN_ERROR)=>{},other=>return Err(format!("the window was not refused: {other:?}"))}
    // The clean retry reads the same source, so its key is the same. It must miss, then keep the real row.
    let retry=window(&clean)?;
    if d.projection_origin("kv",&retry,true)?.is_some() {return Err("a clean retry found a row from the refused selection".into());}
    if bits(&d.read(&d.projection_origin("kv",&older_in,true)?.ok_or("a row from before the window was lost")?)?)!=vec![7.0f32.to_bits();width] {return Err("older memo row changed".into());}
    let real=d.upload(&vec![3.0;width])?;d.remember_projections("kv",true,&[(&retry,&real)])?;
    // This check passes, so the row stays.
    if d.read(&logits)?!=vec![1.0] {return Err("read after retry differs".into());}
    let found=d.projection_origin("kv",&retry,true)?.ok_or("a verified row was discarded")?;
    if bits(&d.read(&found)?)!=vec![3.0f32.to_bits();width] {return Err("verified memo row differs".into());}Ok(())
}
const NAN_ERROR:&str="selection score is not a number";
fn shape(context:usize)->Shape {let mut s=Shape::new(PromptOp::SelectRows,1).unwrap();s.dim=4;s.taps=2048;s.cols=1;s.slots=2051;s.first=context as u32;s.rows=(context/4) as u32;s}
struct Random(u64);
impl Random {fn next(&mut self)->u64 {self.0^=self.0<<13;self.0^=self.0>>7;self.0^=self.0<<17;self.0}}
/// Finite and infinite scores, including every kind of tie. Kinds 0 and 2 are the random rows.
fn scores(kind:usize,n:usize,r:&mut Random)->Vec<f32> {
    (0..n).map(|i|{let v=r.next();match kind {
        0=>f32::from_bits(v as u32&0x7f7fffff|((v>>40&1) as u32)<<31),1=>0.5,2=>(v%3) as f32,3=>if i%2==0 {0.0} else {-0.0},
        4=>if i%2==0 {f32::INFINITY} else {f32::NEG_INFINITY},5=>f32::from_bits(v as u32&0x007fffff),_=>if i%2==0 {f32::MAX} else {f32::MIN},
    }}).collect()
}
fn bits(v:&[f32])->Vec<u32> {v.iter().map(|x|x.to_bits()).collect()}
type Row<'b,'d>=(Value<'b,'d>,Value<'b,'d>,Value<'b,'d>);
fn select<'b,'d>(d:&Device<'b,'d>,s:Shape,x:&Value<'b,'d>)->Result<Row<'b,'d>> {
    let result=d.prompt(s,[Some(x),None,None,None,None],&[])?;let ids=d.range(&result,0,2051)?;let count=d.range(&result,2051,1)?;Ok((result,ids,count))
}
/// Readback path, known path and the known path's own kernel output must be one row.
fn compare<'b,'d>(d:&Device<'b,'d>,s:Shape,x:&[f32])->Result<usize> {
    let context=s.first as usize;let upload=d.upload(x)?;
    let (_,ids,count)=select(d,s,&upload)?;let (a,b)=Compute::read_pair(d,&ids,&count)?;let device=[a,b].concat();
    let known=known_selection(s);
    match known.as_ref() {
        Some(row)=>{
            if bits(&device)!=bits(row) {return Err(format!("known row differs from the readback at context {context}"));}
            let (result,ids,count)=select(d,s,&upload)?;d.select_known(&ids,&count,row)?;
            let (a,b)=Compute::read_pair(d,&ids,&count)?;if bits(&[a,b].concat())!=bits(row) {return Err(format!("known path returned another row at context {context}"));}
            // A view made before select_known has no known row: this read is the kernel's own bytes, and checks the count.
            let own=d.read(&d.range(&result,1,2051)?)?;if bits(&own)!=bits(&row[1..]) {return Err(format!("kernel output differs at context {context}"));}
        },
        None=>{
            if context<=2051 {return Err(format!("context {context} should be known"));}
            let expected=spark_gpu_probe::prompt_ops::reference(s,[x,&[],&[],&[],&[]],&[])?;
            if bits(&device)!=bits(&expected) {return Err(format!("selection differs from the host's select_top at context {context}"));}
        },
    }
    Ok(device.len()*4)
}
/// Gather through both paths: the same rows, and the known path reads nothing.
fn gather<'b,'d>(d:&Device<'b,'d>,s:Shape,x:&[f32],r:&mut Random)->Result<usize> {
    let context=s.first as usize;let row=known_selection(s).ok_or("gather case needs a known row")?;let dim=8usize;
    let source=(0..context*dim).map(|_|f32::from_bits(r.next() as u32&0x7f7fffff)).collect::<Vec<_>>();let source=d.upload(&source)?;let upload=d.upload(x)?;
    let op=CausalShape {op:Op::Gather,heads:0,dim:dim as u32,taps:0,rows:context as u32,cols:context.min(2051) as u32,epsilon:0.0,bound:0.0};
    let mut out=Vec::new();
    for known in [false,true] {
        let (_,ids,count)=select(d,s,&upload)?;if known {d.select_known(&ids,&count,&row)?;}
        let view=d.range(&ids,0,op.cols as usize)?;
        let value=d.operation(op,[Some(&source),Some(&view),Some(&count),None],&[])?;out.push(bits(&d.read(&value)?));
    }
    if out[0]!=out[1] {return Err(format!("gather differs at context {context}"));}Ok(out[0].len()*4)
}
/// A NaN score: ids unchanged, count -2.0, and the request fails before its result is read.
fn nan<'b,'d>(d:&Device<'b,'d>,s:Shape,x:&[f32],at:usize)->Result<usize> {
    let mut x=x.to_vec();x[at]=f32::NAN;let upload=d.upload(&x)?;let row=known_selection(s).ok_or("NaN case needs a known row")?;
    let (_,ids,count)=select(d,s,&upload)?;let (a,b)=Compute::read_pair(d,&ids,&count)?;
    if b[0].to_bits()!=(-2.0f32).to_bits() || bits(&a)!=bits(&row[..2051]) {return Err(format!("NaN row differs at context {}",s.first));}
    // Refused at the next real read, and at the end of a scope with no read.
    let logits=d.upload(&[1.0,2.0])?;let (_,ids,count)=select(d,s,&upload)?;d.select_known(&ids,&count,&row)?;
    let (a,b)=Compute::read_pair(d,&ids,&count)?;if bits(&a)!=bits(&row[..2051]) || b[0]!=s.first as f32 {return Err("known path changed before its check".into());}
    match d.read(&logits) {Err(error) if error.contains(NAN_ERROR)=>{},Err(error)=>return Err(format!("wrong NaN error: {error}")),Ok(_)=>return Err("a NaN score reached a result".into())}
    match d.work(||{let (_,ids,count)=select(d,s,&upload)?;d.select_known(&ids,&count,&row)}) {Err(error) if error.contains(NAN_ERROR)=>{},Err(error)=>return Err(format!("wrong NaN error at scope end: {error}")),Ok(())=>return Err("a NaN score passed the end of a scope".into())}
    // The refusal left nothing behind: the next clean row is accepted.
    let clean=vec![0.5;s.rows as usize];let upload=d.upload(&clean)?;let (_,ids,count)=select(d,s,&upload)?;d.select_known(&ids,&count,&row)?;let fresh=d.upload(&[3.0])?;
    if d.read(&fresh)?!=vec![3.0] {return Err("read after a refusal differs".into());}let _=(ids,count);Ok(2052*4)
}
/// One step's selection work: 11 rounds, each with Gather's pair and ExpandKv's reread, then the logits.
/// A step's real upload volume between drains (head trace: 3,832,114 bytes in 398 calls). The cursor
/// fills, the logits read drains once, and no uploaded word changes. Warm on the second call.
fn step_uploads<'b,'d>(d:&Device<'b,'d>,b:&Backend<'d>,lines:&mut String,r:&mut Random)->Result<()> {
    let (context,calls,words)=(47usize,398usize,2407usize);let s=shape(context);let row=known_selection(s).ok_or("upload case needs a known row")?;let dim=8usize;
    let x=scores(0,context/4,r);let upload=d.upload(&x)?;let source=d.upload(&vec![0.25;context*dim])?;
    let op=CausalShape {op:Op::Gather,heads:0,dim:dim as u32,taps:0,rows:context as u32,cols:context as u32,epsilon:0.0,bound:0.0};
    let payload=(0..calls).map(|i|(0..words).map(|at|f32::from_bits(((i*7919+at) as u32).wrapping_mul(2654435761)&0x7f7fffff)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mut arms=Vec::new();
    for known in [false,true] {
        // Each arm's own: a value read once answers later reads from its cache.
        let logits=d.upload(&vec![1.5;154880])?;b.synchronize()?;let mut counted=[0u128;2];let mut values=Vec::new();
        d.work(|| {
            let before=b.stats();
            for round in 0..11 {
                for i in 0..calls {if i*11/calls==round {values.push(d.upload(&payload[i])?);}}
                let (_,ids,count)=select(d,s,&upload)?;if known {d.select_known(&ids,&count,&row)?;}
                let view=d.range(&ids,0,context)?;d.operation(op,[Some(&source),Some(&view),Some(&count),None],&[])?;Compute::read_pair(d,&ids,&count)?;
            }
            let out=d.read(&logits)?;let after=b.stats().since(before);counted=[after.dtoh_calls,after.stream_wait_calls];Ok(out)
        })?;
        for (value,want) in values.iter().zip(&payload) {if bits(&d.read(value)?)!=bits(want) {return Err("an uploaded word changed under the upload volume".into());}}
        arms.push(counted);
    }
    if arms[0][0]!=34 || arms[1][0]!=2 {return Err(format!("upload step DtoH differs: {arms:?}"));}
    let line=format!("SCREEN upload_volume bytes_a_step={} calls={calls} readback dtoh_calls={} stream_waits={} known dtoh_calls={} stream_waits={} uploaded_words_equal=1\n",calls*words*4,arms[0][0],arms[0][1],arms[1][0],arms[1][1]);print!("{line}");lines.push_str(&line);Ok(())
}
fn step_counts<'b,'d>(d:&Device<'b,'d>,b:&Backend<'d>,context:usize,known:bool,r:&mut Random)->Result<[u128;4]> {
    let s=shape(context);let row=known_selection(s).ok_or("step case needs a known row")?;let dim=8usize;
    let x=scores(0,context/4,r);let upload=d.upload(&x)?;let source=d.upload(&vec![0.25;context*dim])?;
    let op=CausalShape {op:Op::Gather,heads:0,dim:dim as u32,taps:0,rows:context as u32,cols:context as u32,epsilon:0.0,bound:0.0};
    let logits=d.upload(&vec![1.5;154880])?;b.synchronize()?;
    let mut counted=[0u128;2];let selection=d.selection_counts();
    let got=d.work(|| {
        let before=b.stats();
        for _ in 0..11 {
            let (_,ids,count)=select(d,s,&upload)?;if known {d.select_known(&ids,&count,&row)?;}
            let view=d.range(&ids,0,context)?;d.operation(op,[Some(&source),Some(&view),Some(&count),None],&[])?;
            let (a,c)=Compute::read_pair(d,&ids,&count)?;if c[0]!=context as f32 || bits(&a)!=bits(&row[..2051]) {return Err("step row differs".to_string());}
        }
        let out=d.read(&logits)?;let after=b.stats().since(before);counted=[after.dtoh_calls,after.stream_wait_calls];Ok(out)
    })?;
    if got.len()!=154880 {return Err("step logits length differs".into());}
    let done=d.selection_counts();
    Ok([counted[0],counted[1],(done[0]-selection[0]) as u128,(done[1]-selection[1]) as u128])
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let (mut cases,mut bytes,mut nans)=(0usize,0usize,0usize);let mut r=Random(0x9e3779b97f4a7c15);
    let mut lines=String::new();
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let d=Device::new(backend);
        // Every context to the edge, each with the next kind of scores, both sides of the boundary.
        let mut kind=0usize;
        for context in 1..=2056usize {let s=shape(context);let x=scores(kind%7,context/4,&mut r);kind+=1;bytes+=compare(&d,s,&x)?;cases+=1;}
        // Every kind at the edges, and many random rows there, and above the boundary.
        for context in [4,5,7,8,127,128,1040,2047,2048,2049,2050,2051,2052,2053,2055,2056,3000,4158] {
            let s=shape(context);
            for kind in 0..7 {for _ in 0..40 {let x=scores(kind,context/4,&mut r);bytes+=compare(&d,s,&x)?;cases+=1;}}
        }
        for context in [2051,2052] {let s=shape(context);for i in 0..1000 {let x=scores(if i%2==0 {0} else {2},context/4,&mut r);bytes+=compare(&d,s,&x)?;cases+=1;}}
        // Recorded as well as eager.
        for context in [4,16,47,127,1040,2051,2052] {let s=shape(context);
            d.work(|| {for kind in 0..7 {let x=scores(kind,context/4,&mut r);bytes+=compare(&d,s,&x)?;cases+=1;}Ok(())})?;
        }
        for context in [4,16,47,127,1040,2051] {let s=shape(context);
            for kind in 0..7 {let x=scores(kind,context/4,&mut r);bytes+=gather(&d,s,&x,&mut r)?;cases+=1;}
            d.work(|| {let x=scores(0,context/4,&mut r);bytes+=gather(&d,s,&x,&mut r)?;cases+=1;Ok(())})?;
        }
        for context in [4,5,47,127,1040,2048,2051] {let s=shape(context);let groups=context/4;
            for at in [0,groups/2,groups-1] {let x=scores(0,groups,&mut r);bytes+=nan(&d,s,&x,at)?;nans+=1;cases+=1;}
        }
        for scope in [false,true] {memo_window(backend,scope)?;cases+=1;}
        // Before and after, 11 selection rounds and the logits.
        for context in [47,1040,2051] {
            // The first run of a new piece shape captures its graph, and a capture drains the stream once.
            // So each arm runs twice: the second run is the warm count, and the first is printed too.
            let head_first=step_counts(&d,backend,context,false,&mut r)?;let head=step_counts(&d,backend,context,false,&mut r)?;
            let known_first=step_counts(&d,backend,context,true,&mut r)?;let known=step_counts(&d,backend,context,true,&mut r)?;
            lines.push_str(&format!("FIRST context={context} readback={head_first:?} known={known_first:?}\n"));println!("FIRST context={context} readback={head_first:?} known={known_first:?}");
            if head!=[34,23,22,0] {return Err(format!("readback step counters differ: {head:?}"));}
            if known!=[2,1,0,22] {return Err(format!("known step counters differ: {known:?}"));}
            let line=format!("COUNTS context={context} arm=readback dtoh_calls={} stream_waits={} selection_waits={} selection_known={}\nCOUNTS context={context} arm=known dtoh_calls={} stream_waits={} selection_waits={} selection_known={}\n",head[0],head[1],head[2],head[3],known[0],known[1],known[2],known[3]);
            print!("{line}");lines.push_str(&line);
        }
        for _ in 0..2 {step_uploads(&d,backend,&mut lines,&mut r)?;}
        bytes+=served_boundary(&d,backend,&mut lines)?;
        bytes+=model_nan_retry(backend,&mut lines)?;
        Ok(())
    })?;
    budget_drain(&mut lines)?;
    bytes+=lengths(&mut lines)?;
    let line=format!("SELECT KNOWN PASS cases={cases} bytes={bytes} differing_bytes=0 nan_cases={nans} memo_retry=eager,scope,model served=factory,split,decoder budgeted_cross=1 budget_drain=1 upload_volume=1 boundary=2051_known_2052_real ties=1 signed_zeros=1 infinities=1 subnormals=1 paths=eager,recorded scope=one_device_no_model\n");
    write(&out.join("COMPLETE"),format!("{lines}{line}").as_bytes())?;print!("{line}");Ok(())
}
