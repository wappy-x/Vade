//! Cache append ownership and raw-byte screen. No checkpoint is loaded.
use super::*;
use spark_tp::bounded_decoder::{Compute,Device,Value};

fn same<'b,'d>(d:&Device<'b,'d>,value:&Value<'b,'d>,want:&[f32])->Result<()> {
    if want.len()<2 {return Err("cache fixture requires two fresh partial views".into());}
    // A full-range clone keeps known host words. Two proper subranges instead
    // force fresh device reads, including the retained prefix after growth.
    let prefix=d.range(value,0,want.len()-1)?;let tail=d.range(value,want.len()-1,1)?;
    let mut actual=d.read(&prefix)?;actual.extend(d.read(&tail)?);
    if actual.iter().map(|v|v.to_bits()).ne(want.iter().map(|v|v.to_bits())) {return Err("cache append changed raw words".into());}Ok(())
}
pub fn check(out:&Path)->Result<()> {
    fs::create_dir_all(out).map_err(|e|e.to_string())?;
    for recording in [false,true] {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            let queue=if recording {let q=b.resident_queue()?;b.enable_resident_launches()?;Some(q)} else {None};
            let d=Device::new(b);let empty=d.upload(&[])?;let row=d.upload(&[-0.0,f32::from_bits(1)])?;
            let first=d.cache_append(&empty,&row,0,2,8)?;let address=first.device_address();
            let second_row=d.upload(&[f32::INFINITY,-17.0])?;
            let second=d.cache_append(&first,&second_row,1,2,8)?;
            if second.device_address()!=address {return Err("cache append copied within its room".into());}
            same(&d,&first,&[-0.0,f32::from_bits(1)])?;
            same(&d,&second,&[-0.0,f32::from_bits(1),f32::INFINITY,-17.0])?;
            // A second branch cannot write over the first branch's live suffix.
            let suffix=d.range(&second,2,2)?;drop(second);
            let alternative=d.upload(&[7.0,9.0])?;let branch=d.cache_append(&first,&alternative,1,2,8)?;
            if branch.device_address()==address {return Err("cache append overwrote a retained suffix".into());}
            same(&d,&suffix,&[f32::INFINITY,-17.0])?;
            same(&d,&branch,&[-0.0,f32::from_bits(1),7.0,9.0])?;
            drop((branch,suffix));
            let rollback=d.cache_append(&first,&alternative,1,2,8)?;
            if rollback.device_address()!=address {return Err("cache append did not reuse a released tail".into());}
            // Exhaustion copies. Every still-live prefix remains unchanged.
            let mut current=rollback;let mut want=vec![-0.0,f32::from_bits(1),7.0,9.0];
            for rows in 2..8 {
                let next=d.cache_append(&current,&alternative,rows,2,8)?;
                if rows<5 && next.device_address()!=current.device_address() {return Err("cache append grew before exhausting room".into());}
                if rows==5 && next.device_address()==current.device_address() {return Err("cache append did not grow past its room".into());}
                current=next;want.extend([7.0,9.0]);same(&d,&current,&want)?;
            }
            same(&d,&first,&[-0.0,f32::from_bits(1)])?;
            let exact=d.cache_append(&current,&alternative,8,2,0)?;want.extend([7.0,9.0]);same(&d,&exact,&want)?;
            if exact.device_address()==current.device_address() {return Err("zero-room cache append retained spare capacity".into());}
            if d.cache_append(&first,&alternative,2,2,8).is_ok() {return Err("cache append accepted a wrong extent".into());}
            drop((exact,current,first,empty,row,alternative,second_row));drop(d);drop(queue);b.synchronize()?;Ok(())
        })?;
    }
    let receipt="CACHE APPEND PASS raw_bits=true recorder_off_and_on=true prefix=true suffix=true competing_branch=true rollback=true reuse=true growth=true zero_room=true wrong_extent_refused=true no_model=true\n";
    write(&out.join("RECEIPT"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
