use super::*;
use spark_tp::resident_weights::{Rows,merge_argmax,gather_argmax};
use engine_model::plan::layer::Topology;

pub fn check(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut cases=0usize;let mut mailbox_cases=0usize;let mut bytes=0usize;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<28,|backend| {
        for queued in [false,true] {for count in [1,3,31,32,1023,1024,1025,77440] {for variant in 0..8 {
            let mut local=vec![vec![-1.0;count];2];
            match variant {
                0=>{local[0][count-1]=7.0;local[1][0]=7.0;},
                1=>{local[0].fill(-0.0);local[1].fill(0.0);},
                2=>{local[0][count-1]=f32::from_bits(1);local[1][0]=f32::MIN_POSITIVE;},
                3=>{local[0][0]=-f32::MAX;local[1][count-1]=f32::MAX;},
                4=>{local[0][0]=7.0;local[1][count-1]=f32::NAN;},
                5=>{local[0][0]=7.0;local[1][count-1]=f32::NEG_INFINITY;},
                6=>{local[0][0]=7.0;local[1][count-1]=f32::INFINITY;},
                _=>{local[0][0]=f32::from_bits(0x80000001);local[1][count-1]=f32::from_bits(1);},
            }
            let vectors=local.iter().map(|values|backend.upload_vector(values)).collect::<Result<Vec<_>>>()?;
            backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
            let mut partials=backend.vector(count.div_ceil(1024)*3)?;let mut result=backend.vector(6)?;
            for (row,vector) in vectors.iter().enumerate() {backend.enqueue_argmax_device(vector,&mut partials,&mut result,row)?;}
            let values=backend.read_argmax_raw(&result,2,|values|Ok(values.to_vec()))?;bytes+=values.len()*4;
            for (part,values) in local.iter().zip(values.chunks_exact(3)) {
                let finite=part.iter().all(|value|value.is_finite());
                if values[2]!=f32::from(u8::from(finite)) {return Err("compact argmax validity word differs".into());}
                if finite {let expected=part.iter().copied().enumerate().fold((-f32::INFINITY,u32::MAX),|best,(id,value)|spark_gpu_probe::next_math::better(best,(value,id as u32)));
                    if values[0].to_bits()!=expected.0.to_bits() || values[1]!=expected.1 as f32 {return Err("compact argmax local result bytes differ".into());}
                }
            }
            let rows=[Rows {start:0,count,total:count*2,split:true},Rows {start:count,count,total:count*2,split:true}];
            let selected=merge_argmax(&values,&rows);
            if local.iter().flatten().any(|value|!value.is_finite()) {
                if selected.is_ok() {return Err(format!("compact argmax accepted nonfinite logits queued={queued} count={count} variant={variant}"));}
            } else {
                let expected=local.iter().flatten().copied().enumerate().fold((-f32::INFINITY,u32::MAX),|best,(id,value)|spark_gpu_probe::next_math::better(best,(value,id as u32))).1;
                if selected?!=expected {return Err(format!("compact argmax differs queued={queued} count={count} variant={variant}"));}
                let reversed=[values[3],values[4],values[5],values[0],values[1],values[2]];
                if merge_argmax(&reversed,&[rows[1],rows[0]])?!=expected {return Err("compact argmax depends on part arrival order".into());}
            }
            if values.iter().all(|value|value.is_finite()) {for rank in 0..2 {
                let local=&values[rank*3..rank*3+3];let peer=&values[(1-rank)*3..(1-rank)*3+3];
                let local=backend.upload_vector(local)?;let mailbox=backend.tp_mailbox(4096)?;let mut host=mailbox.take_host().ok_or("compact argmax mailbox host is absent")?;
                host.deliver(1,&peer.iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>())?;let mut sequence=0;
                let selected=gather_argmax(backend,&mailbox,&mut sequence,Topology {rank:rank as u32,parts:2},&local,&rows);bytes+=24;
                if selected!=merge_argmax(&values,&rows) {return Err("compact argmax mailbox result differs".into());}
                mailbox.require_success()?;let mut sent=Vec::new();host.read_send(sequence,&mut sent)?;
                if sent!=values[rank*3..rank*3+3].iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>() {return Err("compact argmax mailbox payload bytes differ".into());}
                mailbox_cases+=1;
            }}
            drop(queue);cases+=1;
        }}}
        Ok(())
    })?;
    let line=format!("ARGMAX PASS cases={cases} mailbox_cases={mailbox_cases} result_bytes={bytes} differing_ids=0 paths=eager,recorded validity=all_parts global_origins=true signed_zeros=true subnormals=true\n");
    write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
