//! Workspace ownership, shrink, growth and queued reuse device fixture.
use super::*;
pub fn check(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|b| {
        let _queue=b.resident_queue()?;b.enable_resident_launches()?;let mut storage=None;
        // A retained full value and its range must both stay immutable.
        let mut old=b.resident_workspace(257,&mut storage)?;let old_address=old.device_address();
        let data=(0..257).map(|i|if i%5==0 {-0.0} else {i as f32/16.0}).collect::<Vec<_>>();
        let input=b.upload_vector(&data)?;b.copy_prefix(&input,&mut old,257)?;
        let alias=old.resident_range(7..200)?;
        let mut next=b.resident_workspace(129,&mut storage)?;
        if next.device_address()==old_address {return Err("workspace reused a published owner".into());}
        b.zero_vector(&mut next)?;
        if b.read_vector_values(&old)?.iter().map(|x|x.to_bits()).ne(data.iter().map(|x|x.to_bits())) {return Err("workspace changed a retained value".into());}
        if b.read_vector_values(&alias)?.iter().map(|x|x.to_bits()).ne(data[7..200].iter().map(|x|x.to_bits())) {return Err("workspace changed a retained range".into());}
        drop(old);drop(alias);let address=next.device_address();drop(next);
        let before=b.stats();let mut resized=b.resident_workspace(17,&mut storage)?;
        if resized.device_address()!=address || resized.len()!=17 || b.stats().since(before).allocation_ns!=0 {return Err("private workspace shrink allocated or changed its address".into());}
        b.zero_vector(&mut resized)?;drop(resized);
        // A suffix alone must retain its root after its parent value disappears.
        // Repeat with a nested suffix whose intermediate view also disappears.
        for nested in [false,true] {
            let mut parent=b.resident_workspace(257,&mut storage)?;let address=parent.device_address();
            b.copy_prefix(&input,&mut parent,257)?;
            let suffix=parent.resident_range(71..257)?;
            let (alias,start)=if nested {let child=suffix.resident_range(43..186)?;drop(suffix);(child,114)} else {(suffix,71)};
            drop(parent);
            let mut replacement=b.resident_workspace(257,&mut storage)?;
            if replacement.device_address()==address {return Err("workspace reused an allocation held only by a suffix or nested view".into());}
            b.zero_vector(&mut replacement)?;
            if b.read_vector_values(&alias)?.iter().map(|x|x.to_bits()).ne(data[start..257].iter().map(|x|x.to_bits())) {return Err("workspace overwrote a parent-dropped suffix or nested view".into());}
            drop(alias);let address=replacement.device_address();drop(replacement);
            let private=b.resident_workspace(257,&mut storage)?;
            if private.device_address()!=address {return Err("workspace did not reuse after the final nested view disappeared".into());}drop(private);
        }
        // Reuse while old GPU consumers are pending is safe on the same stream.
        for repeat in 0..32 {
            let len=17+repeat%13;let mut value=b.resident_workspace(len,&mut storage)?;
            let data=(0..len).map(|i|if i%7==0 {-0.0} else {(i+repeat) as f32/32.0}).collect::<Vec<_>>();
            let input=b.upload_vector(&data)?;b.copy_prefix(&input,&mut value,len)?;
            let mut saved=b.vector(len)?;b.copy_prefix(&value,&mut saved,len)?;drop(value);
            let mut reused=b.resident_workspace(len,&mut storage)?;b.zero_vector(&mut reused)?;drop(reused);
            if b.read_vector_values(&saved)?.iter().map(|x|x.to_bits()).ne(data.iter().map(|x|x.to_bits())) {return Err("queued workspace reuse reordered a consumer".into());}
        }
        let mut grown=b.resident_workspace(1025,&mut storage)?;b.zero_vector(&mut grown)?;
        if grown.len()!=1025 || b.read_vector_values(&grown)?.iter().any(|x|x.to_bits()!=0) {return Err("workspace growth extent differs".into());}
        if b.resident_workspace(0,&mut storage).is_ok() {return Err("workspace accepted an empty shape".into());}
        drop(grown);drop(storage);
        Ok(())
    })?;
    workspace_rejects_poison()?;
    let text="RESIDENT WORKSPACE PASS retained_value=true retained_range=true suffix_only=true nested_suffix_only=true shrink_reuses=true growth=true ordered_pending_consumers=32 zero_extent_refused=true exact_raw_bits=true poison_preserved=true no_new_kernel=true\n";
    write(&out.join("RECEIPT"),text.as_bytes())?;print!("{text}");Ok(())
}
/// The allocation helper must not bypass the first permanent recorder error.
fn workspace_rejects_poison()->Result<()> {
    let mut checked=false;
    let result=spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let mut storage=None;let mut value=b.resident_workspace(17,&mut storage)?;b.synchronize()?;
        b.set_graph_storage_budget(0)?;let queue=b.resident_queue()?;b.enable_resident_launches()?;b.zero_vector(&mut value)?;
        let error=b.read_vector_values(&value).err().ok_or("workspace poison fixture admitted a zero-budget graph")?;
        if !error.contains("graph storage") {return Err(format!("workspace poison fixture failed unexpectedly: {error}"));}
        drop(value);
        for words in [7,257,0] {match b.resident_workspace(words,&mut storage) {
            Err(actual) if actual==error=>(),Err(actual)=>return Err(format!("workspace replaced its first recorder error: {actual}")),
            Ok(_)=>return Err("workspace reused or allocated after permanent failure".into())}}
        drop(storage);drop(queue);checked=true;Ok(())
    });
    if !checked {return result.and(Err("workspace poison checks did not finish".into()));}
    if result.is_ok() {return Err("workspace poisoned backend retired without an error".into());}Ok(())
}
