use super::*;
#[allow(dead_code)]
#[path="../resident-layer-check/fixture.rs"] mod fixture;
struct Temp(std::path::PathBuf);
impl Temp {fn new()->Self {
    static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
    let p=std::env::temp_dir().join(format!("vade-checkpoint-prefix-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
    fs::create_dir(&p).unwrap();Self(p)
}}
impl Drop for Temp {fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();}}
#[test]
fn selected_checkpoint_rows_feed_the_real_prefix_without_a_layer_admission() {
    let temp=Temp::new();let d=fixture::description();let root=temp.0.join("checkpoint");
    let index=fixture::checkpoint(&root,&d).unwrap();let c=d.open_headers(&root,&index).unwrap();
    let a=Prefix::new(&c,Topology {rank:0,parts:2}).unwrap();let b=Prefix::new(&c,Topology {rank:1,parts:2}).unwrap();
    assert!(a.next.contains("Project"));assert!(a.row(d.vocab as u32).is_err());
    let expected=fixture::payload(&a.embedding,&fixture::headers(&d)[&a.embedding]);
    for token in [0,1,d.vocab as u32-1] {
        let row=a.row(token).unwrap();let start=token as usize*a.width*2;
        assert_eq!(row,&expected[start..start+a.width*2]);
        let prefix=a.expected(token).unwrap();
        assert_eq!(prefix[0],oracle::embedding(&row).repeat(d.residual.streams as usize));
        assert_eq!(prefix,b.expected(token).unwrap());
        assert_eq!(prefix[4].len(),d.hidden as usize);
        assert!(prefix.iter().flatten().all(|v|v.is_finite()));
    }
    let bad=Topology {rank:2,parts:2};assert!(Prefix::new(&c,bad).is_err());
    let (device,required)=oracle::budget(&a.block).unwrap();assert!(required>device);
    assert!(device>=a.block.mhc().weights().len()*4);
}
#[test]
fn prefix_receipts_reject_partial_changed_and_mismatched_inputs() {
    let temp=Temp::new();let a=temp.0.join("a");let b=temp.0.join("b");
    let ids=[2,3];
    for p in [&a,&b] {
        fs::create_dir(p).unwrap();
        write(&p.join("IDENTITY"),b"host comparator fixture; no device proof\n").unwrap();
        write(&p.join("SCOPE"),SCOPE).unwrap();write(&p.join("INPUT_IDS"),b"2\n3\n").unwrap();
        write(&p.join("PLANE_WORDS"),b"1 1 1 1 1\n").unwrap();
        for name in plane_files(&ids) {write(&p.join(name),&1.0f32.to_le_bytes()).unwrap();}
        seal(p,&ids).unwrap();
    }
    compare(&a,&b).unwrap();
    fs::write(b.join("0-input-norm.f32"),2.0f32.to_le_bytes()).unwrap();assert!(compare(&a,&b).is_err());
    fs::write(b.join("0-input-norm.f32"),1.0f32.to_le_bytes()).unwrap();
    fs::write(b.join("INPUT_IDS"),b"2\n4\n").unwrap();assert!(compare(&a,&b).is_err());
    fs::write(b.join("INPUT_IDS"),b"2\n3\n").unwrap();
    fs::write(b.join("0-input-norm.f32"),[]).unwrap();assert!(complete(&b).is_err());
    fs::write(b.join("0-input-norm.f32"),1.0f32.to_le_bytes()).unwrap();
    fs::write(b.join("PLANE_WORDS"),b"0 1 1 1 1\n").unwrap();assert!(complete(&b).is_err());
    fs::write(b.join("PLANE_WORDS"),b"1 1 1 1 1\n").unwrap();
    fs::write(b.join("COMPLETE"),b"FAILED\n").unwrap();assert!(compare(&a,&b).is_err());
}
#[test]
#[allow(clippy::excessive_precision)] // literal constants stay as written.
fn input_norm_keeps_the_pre_weight_cast_and_stored_fp32_bits() {
    use spark_gpu_probe::{bf16,norm_boundary::{NormShape,WeightType}};
    let w=1.0000009536743164;let shape=NormShape::new(1,1.0,WeightType::F32).unwrap();
    assert_eq!(oracle::norm(shape,&[1.0],&[w]),[bf16::rounded(1.0/2.0f32.sqrt())*w]);
    assert_ne!(oracle::norm(shape,&[1.0],&[w]),oracle::norm(shape,&[1.0],&[bf16::rounded(w)]));
}
#[test]
fn prompt_files_refuse_empty_invalid_and_overlarge_id_sets() {
    let temp=Temp::new();let path=temp.0.join("ids");
    for content in [String::new(),"-1".into(),"4294967296".into(),"nan".into(),"1 ".repeat(4097)] {
        fs::write(&path,content).unwrap();assert!(prompt(&path).is_err());
    }
    fs::write(&path,b"0\n17\n").unwrap();assert_eq!(prompt(&path).unwrap(),[0,17]);
}
