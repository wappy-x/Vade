use super::*;
use engine_model::plan::layer::Topology;
use spark_tp::resident_weights::{gather_stage,gather_stage_rounds,Rows};
/// `owed`: the stage still owes BF16 rounds, so the largest finite words (which round to infinity) are left out.
fn payload(count:usize,job:usize,rank:usize,owed:bool)->Vec<f32> {
    let bits=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000];
    let ties=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f808000,0xbf818000,0x3f807fff,0x7e7f7fff,0x3f800000,0xbf808001];
    (0..count).map(|at|f32::from_bits(if owed {ties[(at+job*3+rank*7)%ties.len()]} else {bits[(at+job*3+rank*7)%bits.len()]})).collect()
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut cases=0usize;let mut bytes=0usize;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let lengths=[1,2,3,7,15,16,17,127,128,129,256,1024,4096];
        // `owed`: every second part reaches the gather unrounded with its flag, as a projection stage passes it.
        for owed in [false,true] {for queued in [false,true] {for rank in 0..2 {for jobs in [1,2,3,8,16,24,32] {
            let count=(0..jobs).map(|job|lengths[(job+jobs)%lengths.len()]).collect::<Vec<_>>();let words=count.iter().sum::<usize>();
            let mailbox=backend.tp_mailbox(words*4)?;let mut host=mailbox.take_host().ok_or("exchange stage fixture host already taken")?;
            let rounds=(0..jobs).map(|job|owed && job%2==0).collect::<Vec<_>>();
            let sent=|job:usize,origin:usize|payload(count[job],job,origin,owed).into_iter().map(|value|if rounds[job] {spark_gpu_probe::bf16::rounded(value)} else {value}).collect::<Vec<_>>();
            let local=count.iter().enumerate().map(|(job,&count)|backend.upload_vector(&payload(count,job,rank,owed))).collect::<Result<Vec<_>>>()?;
            let peer=(0..jobs).flat_map(|job|sent(job,1-rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
            let rows=count.iter().map(|&count|Rows {start:rank*count,count,total:count*2,split:true}).collect::<Vec<_>>();
            backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
            let mut sequence=0;
            for _ in 0..3 {
                host.deliver(sequence+1,&peer)?;let topology=Topology {rank:rank as u32,parts:2};
                let output=if owed {gather_stage_rounds(backend,&mailbox,&mut sequence,topology,&local,&rounds,&rows)?} else {gather_stage(backend,&mailbox,&mut sequence,topology,&local,&rows)?};
                for (job,value) in output.iter().enumerate() {
                    let got=backend.read_vector_values(value)?.iter().map(|value|value.to_bits()).collect::<Vec<_>>();
                    let expected=(0..2).flat_map(|origin|sent(job,origin)).map(f32::to_bits).collect::<Vec<_>>();
                    if got!=expected {return Err(format!("grouped exchange differs owed={owed} queued={queued} rank={rank} jobs={jobs} job={job}"));}bytes+=got.len()*4;
                }
                backend.synchronize()?;mailbox.require_success()?;let mut published=Vec::new();host.read_send(sequence,&mut published)?;
                let expected=(0..jobs).flat_map(|job|sent(job,rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
                if published!=expected {return Err("grouped exchange published a different local payload".into());}bytes+=published.len();cases+=1;
            }
            // The parts are sources only: an owed round is never written back to them.
            for (job,value) in local.iter().enumerate() {if backend.read_vector_values(value)?.iter().map(|v|v.to_bits()).ne(payload(count[job],job,rank,owed).iter().map(|v|v.to_bits())) {return Err("grouped exchange changed a local part".into());}}
            drop(queue);
        }}}}
        let (more,checked)=kda_owned(backend)?;cases+=more;bytes+=checked;
        Ok(())
    })?;
    let line=format!("EXCHANGE STAGE PASS cases={cases} bytes={bytes} differing_bytes=0 ranks=0,1 paths=eager,recorded parts_per_stage=1,2,3,8,16,24,32 owed_rounds=0,1 finite_extremes=1 signed_zeros=1 subnormals=1 bf16_ties=1\n");write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
/// The KDA frontiers at their rc.8 dimensions. The all-local frontier must not
/// advance the sequence. Only f_a/g_a and the final gated plane cross ranks.
fn kda_owned(b:&spark_gpu_probe::driver::Backend<'_>)->Result<(usize,usize)> {
    use spark_tp::resident_weights::gather_stage_selected;
    let mut cases=0;let mut bytes=0;
    for queued in [false,true] {for rank in 0..2 {for owed in [false,true] {
        let mailbox=b.tp_mailbox(1<<20)?;let mut host=mailbox.take_host().ok_or("KDA exchange host is absent")?;
        let queue=if queued {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
        let topology=Topology {rank:rank as u32,parts:2};let mut sequence=0;
        for (count,owned) in [(vec![4096,4096,4096,64,64,32],vec![true,true,true,false,false,true]),(vec![4096,4096],vec![true,true]),(vec![4096],vec![false])] {
            let words=|job,origin|payload(count[job],job,origin,owed).into_iter().map(|v|if owed {spark_gpu_probe::bf16::rounded(v)} else {v}).collect::<Vec<_>>();
            let local=count.iter().enumerate().map(|(job,&n)|b.upload_vector(&payload(n,job,rank,owed))).collect::<Result<Vec<_>>>()?;
            let peer=(0..count.len()).filter(|&job|!owned[job]).flat_map(|job|words(job,1-rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
            let before=sequence;if !peer.is_empty() {host.deliver(sequence+1,&peer)?;}
            let rows=count.iter().map(|&n|Rows {start:rank*n,count:n,total:2*n,split:true}).collect();
            let got=gather_stage_selected(b,&mailbox,&mut sequence,topology,local,vec![owed;count.len()],rows,owned.clone())?;
            for (job,value) in got.iter().enumerate() {
                let expected=if owned[job] {words(job,rank)} else {[words(job,0),words(job,1)].concat()};
                let actual=b.read_vector_values(value)?;
                if actual.iter().map(|v|v.to_bits()).ne(expected.iter().map(|v|v.to_bits())) {return Err(format!("KDA selected exchange differs rank={rank} queued={queued} owed={owed} job={job}"));}
                bytes+=actual.len()*4;
            }
            b.synchronize()?;mailbox.require_success()?;
            if peer.is_empty() {if sequence!=before {return Err("local KDA frontier sent a message".into());}}
            else {
                if sequence!=before+1 {return Err("KDA frontier message count differs".into());}
                let mut sent=Vec::new();host.read_send(sequence,&mut sent)?;
                let expected=(0..count.len()).filter(|&job|!owned[job]).flat_map(|job|words(job,rank)).flat_map(f32::to_le_bytes).collect::<Vec<_>>();
                if sent!=expected {return Err("KDA frontier wire bytes differ".into());}bytes+=sent.len();
            }cases+=1;
        }
        if sequence!=2 {return Err("KDA gather sequence differs".into());}drop(queue);
    }}}
    println!("KDA SELECTED EXCHANGE PASS cases={cases} bytes={bytes} differing_bytes=0 first_payload_bytes=512 gated_payload_bytes=16384 messages=2");Ok((cases,bytes))
}
