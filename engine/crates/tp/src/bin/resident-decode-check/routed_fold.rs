use super::*;
use engine_model::plan::layer::Topology;
use spark_gpu_probe::causal_ops::{Op,Shape};
use spark_tp::resident_weights::{fold_then_gather,Rows};
/// LOCAL FOLD on one device, no model: for each rank, the fold of this rank's rows before the exchange
/// (fold_then_gather, the peer's folded rows delivered by the host) against the fold of the gathered rows on the same
/// device. The whole output and the published message must have every bit of the gathered fold.
fn word(seed:usize,special:bool)->f32 {
    let bits=[0,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_8000,0x3f80_7fff,0x3f81_8000,0xbf80_8000,0x3f7f_ffff,0x4000_0001,0xc0ff_ffff,0x477f_e000,0xc77f_e000,0x3380_0000,0xb380_0000u32];
    if special && seed%3!=1 {f32::from_bits(bits[(seed/3)%bits.len()])} else {((seed*2654435761usize)%2001) as f32/250.0-4.0}
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut cases=0usize;let mut words=0usize;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        for queued in [false,true] {for (picks,total) in [(8usize,4096usize),(8,256),(1,128),(5,384),(32,128)] {for taps in [0u32,1] {for special in [false,true] {
            let count=total/2;
            let ids=(0..picks).map(|pick|((pick*37+11)%288) as f32).collect::<Vec<_>>();
            let weights=(0..picks).map(|pick|spark_gpu_probe::bf16::rounded(if special && pick%3==0 {-0.0} else {word(pick*7+1,false)/8.0})).collect::<Vec<_>>();
            let raw=(0..picks*total).map(|at|word(at+17,special)).collect::<Vec<_>>();
            let shared=(0..total).map(|at|spark_gpu_probe::bf16::rounded(word(at*5+3,special))).collect::<Vec<_>>();
            let fold=Shape {op:Op::ExpertFold,heads:0,dim:0,taps,rows:picks as u32,cols:total as u32,epsilon:0.0,bound:0.0};
            // The fold of the gathered rows: what both ranks hold today behind the exchange of the down rows.
            let (device_ids,device_weights)=(backend.upload_vector(&ids)?,backend.upload_vector(&weights)?);
            let device_shared=if taps!=0 {Some(backend.upload_vector(&shared)?.into_resident_shared()?)} else {None};
            let gathered=backend.upload_vector(&raw.iter().map(|&value|spark_gpu_probe::bf16::rounded(value)).collect::<Vec<_>>())?;
            let whole=backend.causal_operation(fold,[Some(&device_ids),Some(&device_weights),Some(&gathered),device_shared.as_ref()],None)?;
            let expected=backend.read_vector_values(&whole)?;
            for rank in 0..2usize {
                let rows=Rows {start:rank*count,count,total,split:true};
                let own=(0..picks).flat_map(|pick|raw[pick*total+rank*count..pick*total+(rank+1)*count].iter().copied()).collect::<Vec<_>>();
                let down=backend.upload_vector(&own)?;
                let mailbox=backend.tp_mailbox(total*4)?;let mut host=mailbox.take_host().ok_or("routed fold fixture host already taken")?;
                backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
                let mut sequence=0;
                for _ in 0..3 {
                    let peer=expected[(1-rank)*count..(2-rank)*count].iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>();
                    host.deliver(sequence+1,&peer)?;
                    let output=fold_then_gather(backend,&mailbox,&mut sequence,Topology {rank:rank as u32,parts:2},&device_ids,&device_weights,&down,device_shared.as_ref(),fold,rows)?;
                    let got=backend.read_vector_values(&output)?;
                    if got.iter().map(|value|value.to_bits()).ne(expected.iter().map(|value|value.to_bits())) {
                        return Err(format!("routed fold differs queued={queued} picks={picks} total={total} taps={taps} special={special} rank={rank}"));
                    }
                    backend.synchronize()?;mailbox.require_success()?;let mut published=Vec::new();host.read_send(sequence,&mut published)?;
                    let sent=expected[rank*count..(rank+1)*count].iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>();
                    if published!=sent {return Err(format!("routed fold published other rows queued={queued} picks={picks} total={total} taps={taps} special={special} rank={rank}"));}
                    words+=got.len()+published.len()/4;cases+=1;
                }
                // The down rows are a source only: the round is never written back.
                if backend.read_vector_values(&down)?.iter().map(|value|value.to_bits()).ne(own.iter().map(|value|value.to_bits())) {return Err("routed fold changed its down rows".into());}
                drop(queue);
            }
        }}}}
        Ok(())
    })?;
    let line=format!("ROUTED FOLD PASS cases={cases} compared_words={words} differing_words=0 ranks=0,1 paths=eager,recorded picks=1,5,8,32 shared_row=0,1 signed_zeros=1 bf16_ties=1\n");
    write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
