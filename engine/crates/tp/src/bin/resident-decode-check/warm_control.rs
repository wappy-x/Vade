use super::Result;

fn marker_image(source:&str)->Result<String> {
    let start=source.find(".visible .entry resident_launch_zero_words(").ok_or("missing recorded zero entry")?;
    let body=start+source[start..].find('{').ok_or("missing recorded zero body")?;
    let mut depth=0usize;let mut end=None;
    for (offset,character) in source[body..].char_indices() {match character {'{'=>depth+=1,'}'=>{depth-=1;if depth==0 {end=Some(body+offset+1);break;}},_=>{}}}
    let end=end.ok_or("unterminated recorded zero body")?;
    let stores=source[body..end].lines().filter(|line|line.trim_start().starts_with("st.b32") && line.trim_end().ends_with(", 0;")).collect::<Vec<_>>();
    if stores.len()!=1 {return Err("marker requires exactly one recorded zero store".into());}
    let old=stores[0];let replacement=old.replace(", 0;",", 1065353216;");
    let mut image=String::new();image.push_str(&source[..body]);image.push_str(&source[body..end].replacen(old,&replacement,1));image.push_str(&source[end..]);Ok(image)
}

#[cfg(target_os="linux")]
pub fn check(out:&super::Path)->Result<()> {
    use super::{fs,write,sha256};
    use spark_gpu_probe::{driver::{Backend,with_backend},causal_ops::{Shape,Op},resident_program::Program};
    fn plane(backend:&Backend<'_>)->Result<Vec<f32>> {
        let mut target=backend.upload_vector(&[7.0;128])?;let queue=backend.resident_queue()?;backend.enable_resident_launches()?;
        let before=backend.stats();backend.zero_vector(&mut target)?;let output=backend.read_vector_values(&target)?;
        if backend.stats().since(before).graph_launches==0 || backend.graph_storage_bytes()==0 {return Err("marker control did not execute a captured piece".into());}
        drop(queue);Ok(output)
    }
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let baseline=spark_gpu_probe::layer_ptx()?.to_bytes();let text=std::str::from_utf8(baseline).map_err(|error|error.to_string())?;let wrong=marker_image(text)?;
    let baseline_hash=sha256::digest(baseline);let wrong_hash=sha256::digest(wrong.as_bytes());
    if baseline_hash==wrong_hash {return Err("marker image hash did not change".into());}
    write(&out.join("wrong-marker.ptx"),wrong.as_bytes())?;
    write(&out.join("IMAGE_HASHES"),format!("baseline={baseline_hash}\nwrong={wrong_hash}\nmarker_f32_bits=1065353216\n").as_bytes())?;
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        for round in 1..=3 {for (label,image,expected) in [("wrong",wrong.as_bytes(),1.0f32),("right",baseline,0.0),("baseline-again",baseline,0.0)] {
            backend.fixture_reset_resident()?;backend.select_arm(false)?;backend.fixture_replace_full_candidate(image)?;backend.select_arm(true)?;
            let output=plane(backend)?;
            if output.len()!=128 || output.iter().any(|value|value.to_bits()!=expected.to_bits()) {return Err(format!("selected module marker differs round={round} image={label}"));}
            let bytes=output.iter().flat_map(|value|value.to_le_bytes()).collect::<Vec<_>>();write(&out.join(format!("{round}-{label}.f32")),&bytes)?;
            let equals_baseline=bytes==vec![0;128*4];
            if equals_baseline!=(label!="wrong") {return Err("must-fail marker verdict is wrong".into());}
            println!("WARM CONTROL round={round} image={label} sha256={} expected_marker_bits={} comparison={} marker_bytes_checked=512",sha256::digest(image),expected.to_bits(),if equals_baseline {"PASS"} else {"FAIL"});
        }}
        backend.fixture_reset_resident()?;
        let queue=backend.resident_queue()?;
        if backend.fixture_reset_resident().is_ok() || backend.fixture_replace_full_candidate(baseline).is_ok() {return Err("warm control admitted a live queue".into());}
        let shape=Shape {op:Op::StreamMean,heads:2,dim:1,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
        let program=backend.resident_program(&Program::operation(shape)?,&[None])?;drop(queue);
        if backend.fixture_reset_resident().is_ok() || backend.fixture_replace_full_candidate(baseline).is_ok() {return Err("warm control admitted an externally retained graph".into());}
        drop(program);backend.fixture_reset_resident()?;Ok(())
    })?;
    let mut preserved=false;
    let poisoned=with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let mut target=backend.upload_vector(&[7.0;128])?;backend.set_graph_storage_budget(0)?;
        let queue=backend.resident_queue()?;backend.enable_resident_launches()?;backend.zero_vector(&mut target)?;
        let error=backend.synchronize().err().ok_or("zero graph budget did not fail")?;drop(queue);
        let actual=backend.fixture_reset_resident().err().ok_or("warm reset erased a permanent failure")?;
        if actual!=error {return Err(format!("warm reset changed first failure: {error}; {actual}"));}preserved=true;Ok(())
    });
    if !preserved || poisoned.is_ok() {return Err("warm permanent-failure control did not finish".into());}
    let receipt="WARM CONTROL PASS wrong_FAIL_right_PASS_baseline_PASS_rounds=3 one_main_context=true same_names_and_grids=true module_marker=true queue_refused=true external_graph_refused=true poison_preserved=true model_loaded=false projection_memo_tested=false full_model_trust_unblocked=false\n";
    write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn marker_changes_one_store_in_one_entry_only() {
        let source=".visible .entry untouched() { st.b32 [%r0], 0; }\n.visible .entry resident_launch_zero_words(.param .u64 table) {\n st.b32 [%r1], 0;\n}\n.visible .entry after() { ret; }";
        let changed=marker_image(source).unwrap();assert_eq!(changed.matches("1065353216").count(),1);
        assert!(changed.starts_with(".visible .entry untouched() { st.b32 [%r0], 0; }"));assert!(changed.ends_with(".visible .entry after() { ret; }"));
        assert!(marker_image("no entry").is_err());assert!(marker_image(&source.replace("st.b32 [%r1], 0;","st.b32 [%r1], 0;\n st.b32 [%r2], 0;")).is_err());
    }
}
