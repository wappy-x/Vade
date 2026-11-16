//! Device proof of the lookahead fence's deferred allocation boundary.
use super::*;

#[test]
#[ignore="requires CUDA; poisons recycled allocations on a separate stream"]
fn deferred_fence_reuse_survives_poison_while_later_work_runs() {
    with_backend(0,crate::layer_ptx().unwrap(),512<<20,|b| {
        // The poison stream shares the allocator but has its own upload staging.
        // Wait for its event, not Backend::synchronize: a side-stream wait must
        // not clear the context's pending flag while the main stream is busy.
        let poison=Backend::load(b.context,crate::layer_ptx()?,MatvecMode::Vec8,0.0)?;
        b.configure_vector_pool(0);
        let mut delay=b.vector(16<<20)?;let mut delay_out=b.vector(16<<20)?;
        b.zero_vector(&mut delay)?;b.zero_vector(&mut delay_out)?;b.synchronize()?;
        for wrong in [false,false,false,true] {
            b.context.clear_reserves();
            let early=b.upload_vector(&vec![19.;1024])?;
            let late=b.upload_vector(&vec![23.;8192])?;
            let mut early_out=b.vector(1024)?;let mut late_out=b.vector(8192)?;
            let spare=b.vector(8192)?;let spare_address=spare.device_address();
            b.synchronize()?;drop(spare);
            let queue=b.resident_queue()?;
            let early_address=early.device_address();let late_address=late.device_address();
            b.copy_range(&early,&mut early_out,0,0,1024)?;drop(early);
            let (fence,mark)=b.fence_frees()?;
            // More than 8 GiB of ordered copies keep the late consumer behind
            // the fence while the small poison writes finish on the other stream.
            for _ in 0..128 {b.copy_range(&delay,&mut delay_out,0,0,16<<20)?;}
            b.copy_range(&late,&mut late_out,0,0,8192)?;drop(late);
            let late_total=b.context.deferred_total.get();
            let start=Instant::now();
            // Wrong on purpose: claim that the later drop preceded the fence.
            // Its allocation stays in the pool; no memory is returned to CUDA.
            b.wait_fence_frees(fence,if wrong {late_total} else {mark})?;
            let mut recycled_early=poison.vector(1024)?;
            let mut recycled_late=poison.vector(8192)?;
            let mut recycled_second=if wrong {Some(poison.vector(8192)?)} else {None};
            assert_eq!(recycled_early.device_address(),early_address,"the freed allocation must really be poisoned");
            if wrong {
                let mut actual=[recycled_late.device_address(),recycled_second.as_ref().unwrap().device_address()];actual.sort_unstable();
                let mut expected=[spare_address,late_address];expected.sort_unstable();
                assert_eq!(actual,expected,"the wrong mark must release the still-needed allocation");
            } else {assert_eq!(recycled_late.device_address(),spare_address,"the later allocation must remain deferred");}
            poison.write_vector(&mut recycled_early,&vec![-77.;1024])?;
            poison.write_vector(&mut recycled_late,&vec![-99.;8192])?;
            if let Some(second)=&mut recycled_second {poison.write_vector(second,&vec![-99.;8192])?;}
            Event::ready(&poison.stream)?.wait(false)?;
            let poison_us=start.elapsed().as_micros();
            let early_values=b.read_vector_values(&early_out)?;
            let late_values=b.read_vector_values(&late_out)?;
            assert_eq!(early_values,vec![19.;1024],"a completed consumer changed");
            if wrong {assert_eq!(late_values,vec![-99.;8192],"the wrong-mark control must expose early reuse");}
            else {assert_eq!(late_values,vec![23.;8192],"a consumer after the fence read poison");}
            println!("DEFERRED POISON wrong_mark={wrong} early_words=1024 late_words=8192 poison_us={poison_us} pass=true");
            drop(queue);
        }
        println!("DEFERRED POISON PASS valid=3 wrong_mark_detected=1");Ok(())
    }).unwrap();
}
