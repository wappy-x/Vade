#![forbid(unsafe_code)]

#[cfg(target_os="linux")]
mod device {
    use std::{sync::atomic::{AtomicBool,Ordering},time::{Duration,Instant}};
    use spark_gpu_probe::{driver::Backend,tp_mailbox_host::{ERROR,SEND_LEN,SEND_SEQ},driver::Shadow};

    fn pattern(sequence:usize,index:usize)->u32 {
        let value=(sequence as u32).wrapping_mul(0x9e3779b9)^(index as u32).wrapping_mul(0x85ebca6b);
        [value,0,0x80000000,0x7fc01234,0x7f800000,1,0x007fffff][index%7]
    }
    fn answer(sequence:usize,value:u32)->u32 {value.rotate_left(sequence as u32%31)^0xa55a39c3}

    fn publication(backend:&Backend<'_>,words:usize,wide:bool,shadow:&Shadow<'_>)->Result<(),String> {
        let rounds=64usize;
        let mailbox=backend.tp_mailbox(words*4)?;
        let mut host=mailbox.take_host().ok_or("mailbox host already taken")?;
        host.set_wide(wide);host.set_take_pause(128);
        let layout=mailbox.layout();
        let mut inputs=Vec::new();let mut outputs=Vec::new();
        for sequence in 1..=rounds {
            inputs.push(backend.upload_vector(&(0..words).map(|index|f32::from_bits(pattern(sequence,index))).collect::<Vec<_>>())?);
            outputs.push(backend.upload_vector(&vec![f32::from_bits(0xdeadbeef);words])?);
        }
        backend.synchronize()?;
        let stop=AtomicBool::new(false);
        std::thread::scope(|scope|->Result<(),String> {
            let worker=scope.spawn(||->Result<(),String> {
                let memory=host.words();
                let result=(|| {
                    for sequence in 1..=rounds {
                        let slot=sequence%layout.slots;let deadline=Instant::now()+Duration::from_secs(3);
                        while memory[SEND_SEQ+slot].load(Ordering::Acquire)!=sequence as u32 {
                            if stop.load(Ordering::Acquire) || Instant::now()>=deadline {return Err("publication worker timed out".to_string());}
                            std::hint::spin_loop();
                        }
                        if memory[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=words*4 {return Err("published length differs".into());}
                        let from=layout.send_slot(slot)+layout.frame;let to=layout.recv_slot(0,slot)+layout.frame;
                        for index in 0..words {
                            let actual=memory[from+index].load(Ordering::Relaxed);
                            if actual!=pattern(sequence,index) {return Err(format!("send publication differs sequence={sequence} word={index}"));}
                            memory[to+index].store(answer(sequence,actual),Ordering::Relaxed);
                        }
                        memory[layout.recv_seq(0,slot)].store(sequence as u32,Ordering::Release);
                    }
                    Ok(())
                })();
                if result.is_err() {memory[ERROR].store(1,Ordering::Release);}
                result
            });
            let queued=(|| {
                let queue=backend.resident_queue()?;
                for sequence in 1..=rounds {
                    backend.tp_mailbox_put(&mailbox,&inputs[sequence-1],1,words,words,0,sequence as u64)?;
                    backend.chain_shadow_prefetch(shadow)?;
                    backend.tp_mailbox_take(&mailbox,&mut outputs[sequence-1],1,words,words,0,sequence as u64,2_000_000_000)?;
                }
                drop(queue);backend.synchronize()
            })();
            if queued.is_err() {stop.store(true,Ordering::Release);mailbox.fail();}
            let served=worker.join().map_err(|_|"publication worker panicked".to_string())?;
            queued?;served?;mailbox.require_success()
        })?;
        for (index,output) in outputs.iter().enumerate() {
            let sequence=index+1;
            for (word,value) in backend.read_vector_values(output)?.iter().enumerate() {
                if value.to_bits()!=answer(sequence,pattern(sequence,word)) {return Err(format!("receive publication differs sequence={sequence} word={word}"));}
            }
        }
        println!("PUBLICATION PASS words={words} rounds={rounds} wide={wide} compared_bytes={} differing_words=0",rounds*words*4*2);
        Ok(())
    }

    pub fn run()->Result<(),String> {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
            let data=(0..65536).map(|index|f32::from_bits(pattern(97,index))).collect::<Vec<_>>();
            let weights=backend.upload_vector(&data)?;
            let range=backend.shadow_vector_range(&weights,0,weights.len()*4)?;
            for blocks in [1,2,4,64] {for evict_last in [false,true] {
                let shadow=Shadow {rms:None,dots:Some(range),finish:None,blocks,evict_last};
                for bytes in [1,127,128,129,1023,65535] {
                    let tail=backend.shadow_vector_range(&weights,1,bytes)?;
                    backend.chain_shadow_prefetch(&Shadow {dots:Some(tail),..shadow})?;
                }
                backend.synchronize()?;
                println!("PREFETCH PASS blocks={blocks} evict_last={evict_last} tail_cases=6");
            }}
            for blocks in [0,65] {
                if backend.chain_shadow_prefetch(&Shadow {rms:None,dots:Some(range),finish:None,blocks,evict_last:false}).is_ok() {return Err("invalid prefetch grid was accepted".into());}
            }
            if backend.shadow_vector_range(&weights,weights.len()*4,1).is_ok() {return Err("out of bounds prefetch was accepted".into());}
            let shadow=Shadow {rms:None,dots:Some(range),finish:None,blocks:2,evict_last:false};
            for words in [1,3,4,31,32,33,4096,16384] {for wide in [false,true] {publication(backend,words,wide,&shadow)?;}}
            if backend.read_vector_values(&weights)?.iter().map(|value|value.to_bits()).ne(data.iter().map(|value|value.to_bits())) {return Err("prefetch changed weight bytes".into());}
            println!("CONDUCTOR_GPU_CHECK PASS scope=publication_and_prefetch_not_model_gate");Ok(())
        })
    }
}

fn main() {
    #[cfg(target_os="linux")]
    if let Err(error)=device::run() {eprintln!("{error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("conductor GPU check needs Linux and CUDA");std::process::exit(1);}
}
