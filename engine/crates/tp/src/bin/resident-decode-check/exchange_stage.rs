use super::*;
use engine_model::plan::layer::Topology;
use spark_tp::resident_weights::{gather_stage,Rows};
fn payload(count:usize,job:usize,rank:usize)->Vec<f32> {
    let bits=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000];
    (0..count).map(|at|f32::from_bits(bits[(at+job*3+rank*7)%bits.len()])).collect()
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut cases=0usize;let mut bytes=0usize;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let lengths=[1,2,3,7,15,16,17,127,128,129,256,1024,4096];
        for queued in [false,true] {for rank in 0..2 {for jobs in [1,2,3,8,16,24,32] {
            let count=(0..jobs).map(|job|lengths[(job+jobs)%lengths.len()]).collect::<Vec<_>>();let words=count.iter().sum::<usize>();
            let mailbox=backend.tp_mailbox(words*4)?;let mut host=mailbox.take_host().ok_or("exchange stage fixture host already taken")?;
            let local=count.iter().enumerate().map(|(job,&count)|backend.upload_vector(&payload(count,job,rank))).collect::<Result<Vec<_>>>()?;
            let peer=count.iter().enumerate().flat_map(|(job,&count)|payload(count,job,1-rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
            let rows=count.iter().map(|&count|Rows {start:rank*count,count,total:count*2,split:true}).collect::<Vec<_>>();
            backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
            let mut sequence=0;
            for _ in 0..3 {
                host.deliver(sequence+1,&peer)?;let output=gather_stage(backend,&mailbox,&mut sequence,Topology {rank:rank as u32,parts:2},&local,&rows)?;
                for (job,(&count,value)) in count.iter().zip(&output).enumerate() {
                    let got=backend.read_vector_values(value)?.iter().map(|value|value.to_bits()).collect::<Vec<_>>();
                    let expected=(0..2).flat_map(|origin|payload(count,job,origin)).map(f32::to_bits).collect::<Vec<_>>();
                    if got!=expected {return Err(format!("grouped exchange differs queued={queued} rank={rank} jobs={jobs} job={job}"));}bytes+=got.len()*4;
                }
                backend.synchronize()?;mailbox.require_success()?;let mut sent=Vec::new();host.read_send(sequence,&mut sent)?;
                let expected=count.iter().enumerate().flat_map(|(job,&count)|payload(count,job,rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
                if sent!=expected {return Err("grouped exchange published a different local payload".into());}bytes+=sent.len();cases+=1;
            }
            drop(queue);
        }}}
        Ok(())
    })?;
    let line=format!("EXCHANGE STAGE PASS cases={cases} bytes={bytes} differing_bytes=0 ranks=0,1 paths=eager,recorded parts_per_stage=1,2,3,8,16,24,32 finite_extremes=1 signed_zeros=1 subnormals=1\n");write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
