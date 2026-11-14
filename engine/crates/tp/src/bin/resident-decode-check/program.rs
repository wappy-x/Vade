//! Direct ownership, replay and arithmetic checks for retained causal pieces.
use super::*;
pub fn check(out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,resident_program::Program,causal_ops::{Shape,Op,reference}};
    fs::create_dir(out).map_err(|e|e.to_string())?;
    with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|b| {
        let s=Shape {op:Op::Step,heads:2,dim:4,taps:3,rows:4,cols:4,epsilon:1e-6,bound:-1.0};
        let plan=Program::operation(s)?;
        if b.resident_program(&plan,&[None,None]).is_ok() {return Err("graph accepted an unowned queue".into());}
        let _queue=b.resident_queue()?;
        for op in [Op::Conv,Op::Step,Op::Pool,Op::Project] {
            let shape=Shape {op,..s};let plan=Program::operation(shape)?;let (lengths,wn,_,_)=shape.lengths().ok_or("fixture shape differs")?;
            let weight=(0..wn).map(|i|(i as f32-7.0)/256.0).collect::<Vec<_>>();let w=if wn>0 {Some(b.upload_vector(&weight)?.into_resident_shared()?)} else {None};
            let weights=plan.nodes.iter().map(|n|if n.shape.lengths().unwrap().1>0 {w.as_ref()} else {None}).collect::<Vec<_>>();
            let mut program=b.resident_program(&plan,&weights)?;drop(w);
            let mut saved=None;
            for repeat in 0..3 {
                let data=lengths.map(|n|(0..n).map(|i|if i%17==0 {-0.0} else {((i+repeat) as f32-7.0)/128.0}).collect::<Vec<_>>());
                let expected=reference(shape,data.each_ref().map(|v|v.as_slice()),&weight)?;
                let inputs=data.iter().filter(|v|!v.is_empty()).map(|v|b.upload_vector(v)).collect::<Result<Vec<_>>>()?;
                let before=b.stats();let output=program.run(&inputs.iter().collect::<Vec<_>>())?;let stats=b.stats().since(before);
                if stats.graph_launches!=1 || stats.stream_wait_calls!=0 {return Err("steady replay added a capture or wait".into());}
                let got=b.read_vector_values(&output)?;if got.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err(format!("resident program differs op={op:?} repeat={repeat}"));}
                if repeat==0 {saved=Some((output,got));} else {let (old,bits)=saved.as_ref().unwrap();if b.read_vector_values(old)?.iter().map(|v|v.to_bits()).ne(bits.iter().map(|v|v.to_bits())) {return Err("replay overwrote a retained output".into());}}
            }
            if program.run(&[]).is_ok() {return Err("graph accepted an invalid input count".into());}
        }
{
            use spark_gpu_probe::resident_program::Node;
            let (h,k,v,capacity)=(3u32,5u32,7u32,16u32);
            let score=Shape {op:Op::ScoresSlots,heads:h,dim:k,taps:0,rows:capacity,cols:0,epsilon:0.0,bound:0.0};
            let softmax=Shape {op:Op::Softmax,..score};let value=Shape {op:Op::ValuesChannels,dim:v,..score};
            let plan=Program {inputs:vec![(h*k) as usize,(capacity*h*k) as usize,(capacity*h*v) as usize,1],nodes:vec![
                Node {shape:score,inputs:[Some(0),Some(1),Some(3),None]},Node {shape:softmax,inputs:[Some(4),Some(3),None,None]},Node {shape:value,inputs:[Some(5),Some(2),Some(3),None]}]};
            let mut program=b.resident_program(&plan,&[None,None,None])?;
            for rows in [16u32,1,7,16,1] {for live in [0,rows] {
                let data=[(0..h*k).map(|i|(i as i32%7-3) as f32/16.0).collect::<Vec<_>>(),
                    (0..rows*h*k).map(|i|if i%11==0 {-0.0} else {(i as i32%13-6) as f32/16.0}).collect(),
                    (0..rows*h*v).map(|i|if i%17==0 {-0.0} else {(i as i32%19-9) as f32/16.0}).collect(),vec![live as f32]];
                let score=Shape {op:Op::Scores,rows,..score};let scores=reference(score,[&data[0],&data[1],&data[3],&[]],&[])?;
                let probabilities=reference(Shape {op:Op::Softmax,..score},[&scores,&data[3],&[],&[]],&[])?;
                let expected=reference(Shape {op:Op::Values,dim:v,..score},[&probabilities,&data[2],&data[3],&[]],&[])?;
                let inputs=data.iter().map(|v|b.upload_vector(v)).collect::<Result<Vec<_>>>()?;
                let output=program.run_padded(&inputs.iter().collect::<Vec<_>>())?;
                if b.read_vector_values(&output)?.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err("padded attention replay differs from original ordered attention".into());}
            }}
        }
        Ok(())
    })?;
    let receipt="RESIDENT PROGRAM PASS repeats=3 chains=4 prior_outputs_immutable=true weight_owner_retained=true one_replay=true no_steady_wait=true invalid_inputs=true padded_attention=true shrinking_rows=true zero_live_rows=true exact_bits=true\n";
    write(&out.join("RECEIPT"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
