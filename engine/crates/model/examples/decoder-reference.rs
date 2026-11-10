//! Checkpoint-backed causal stack reference. This executable never calls CUDA.
#[path="layer-reference/checkpoint.rs"] mod checkpoint;
#[path="layer-reference/receipt.rs"] mod receipt;
#[path="decoder-reference/fixture.rs"] mod fixture;
use engine_model::{Result,plan::{ConfigPlan,decoder_reference::Decoder,layer::reference::{Data,Tensor}},safetensors};
use std::{fs,path::Path};
fn invalid(message: impl Into<String>) -> engine_model::Error {std::io::Error::new(std::io::ErrorKind::InvalidData,message.into()).into()}
fn main() {if let Err(e)=run() {eprintln!("decoder-reference: {e}");std::process::exit(1);}}
fn run() -> Result<()> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("generate") if args.len()==4=>generate(Path::new(&args[1]),Path::new(&args[2]),&args[3]),
        Some("verify") if args.len()==2=>receipt::verify(Path::new(&args[1])),
        Some("compare") if args.len()==3=>compare(Path::new(&args[1]),Path::new(&args[2])),
        Some("self-test") if args.len()==2=>self_test(Path::new(&args[1])),
        _=>Err(invalid("usage: decoder-reference generate MODEL NEW_OUT IDS_CSV | verify DIR | compare EXPECTED ACTUAL | self-test NEW_OUT")),
    }
}
fn self_test(out: &Path) -> Result<()> {
    fs::create_dir(out)?; let model=out.join("model"); fixture::create(&model)?;
    generate(&model,&out.join("first"),"1,3,0,4,2")?;
    generate(&model,&out.join("repeat"),"1,3,0,4,2")?;
    compare(&out.join("first"),&out.join("repeat"))?;
    println!("STACK_SELF_TEST_OK synthetic=true device_pass=false"); Ok(())
}
fn compare(expected: &Path,actual: &Path) -> Result<()> {
    receipt::compare(expected,actual)?;
    let first=safetensors::read_bounded(&expected.join("TOKENS.tsv"),64<<20)?;
    let second=safetensors::read_bounded(&actual.join("TOKENS.tsv"),64<<20)?;
    if first!=second {return Err(invalid("decoder reference token IDs differ"));}
    println!("STACK_TOKEN_IDS_EQUAL device_pass=false"); Ok(())
}
fn write_tensor(out: &Path,rows: &mut String,name: &str,tensor: &Tensor) -> Result<()> {
    let (dtype,bytes)=match &tensor.data {
        Data::F32(values)=>("F32",values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
        Data::U32(values)=>("U32",values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    };
    let file=format!("{name}.safetensors"); let encoded=receipt::encode(dtype,&tensor.shape,&bytes);
    fs::write(out.join(&file),&encoded)?;
    rows.push_str(&format!("{file}\t{dtype}\t{:?}\t{}\t{}\n",tensor.shape,bytes.len(),engine_hash::digest(&encoded))); Ok(())
}
fn generate(root: &Path,out: &Path,ids: &str) -> Result<()> {
    let ids=ids.split(',').map(|s| s.parse::<u32>().map_err(|_| invalid("invalid token ID"))).collect::<Result<Vec<_>>>()?;
    if ids.is_empty() || ids.len()>16 {return Err(invalid("reference needs 1 through 16 text IDs"));}
    let config=safetensors::read_bounded(&root.join("config.json"),engine_model::shapes::CONFIG_BYTES)?;
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&config).map_err(|e| invalid(e.to_string()))? else {
        return Err(invalid("reference needs deferred text operation data"));
    };
    let cp=checkpoint::Checkpoint::open(root,d.storage.clone())?;
    let mut decoder=Decoder::new(&d,&cp,ids.len() as u64)?;
    fs::create_dir(out)?; fs::write(out.join("config.json"),&config)?;
    let profile=format!("profile=logical-f32-v1\nscope=causal-decoder-stack\nsource_revision={}\nconfig_sha256={}\nindex_sha256={}\nids={ids:?}\nlayers={}\ninput=stored embedding row widened to FP32 and copied to each residual stream\nstate=zero initially; carry every layer state between text IDs\nlayer_isolation=false; each decoder receives the preceding decoder output\nhead=ascending FP32 stream mean; scalar RMS with stored decoded gain; decoded full-vocabulary projection\nselection=lowest token ID for finite maximum ties\ncasts=no checkpoint BF16 activation casts or activation quantization\nexecution=host-only; not checkpoint backend parity\n",
        include_str!("../../../REVISION").trim(),engine_hash::digest(&config),cp.index_sha256,d.layers.len());
    fs::write(out.join("PROFILE.txt"),profile)?; fs::write(out.join("POLICY.txt"),receipt::POLICY)?;
    let mut rows=String::from("file\tdtype\tshape\tbytes\tsha256\n"); let mut selections=String::from("context\tinput_id\tnext_id\n");
    for &id in &ids {
        let outputs=decoder.append_observed(&[id],|context,layer,result| {
            for (name,tensor) in &result.tensors {write_tensor(out,&mut rows,&format!("token{context}.layer{layer}.{name}"),tensor)?;}
            let trace=result.trace.iter().map(|e| format!("{}\t{:?}\t{:?}\t{:?}\n",e.id,e.kind,e.inputs,e.outputs)).collect::<String>();
            fs::write(out.join(format!("token{context}.layer{layer}.trace.tsv")),trace)?;
            println!("STACK_LAYER context={context} layer={layer} device_pass=false"); Ok(())
        })?;
        for output in outputs {
            for (name,tensor) in [("residual",&output.residual),("hidden",&output.hidden),("normalized",&output.normalized),("logits",&output.logits)] {
                write_tensor(out,&mut rows,&format!("token{}.text.{name}",output.context),tensor)?;
            }
            selections.push_str(&format!("{}\t{}\t{}\n",output.context,id,output.next_token));
            println!("STACK_TOKEN context={} input_id={id} next_id={} device_pass=false",output.context,output.next_token);
        }
    }
    fs::write(out.join("TOKENS.tsv"),selections)?; fs::write(out.join("TENSORS.tsv"),rows)?;
    fs::write(out.join("WEIGHTS.tsv"),format!("tensor\tdtype\tshape\tstart\tend\tshard\tsha256\n{}",
        cp.receipts.borrow().values().map(|s| s.clone()+"\n").collect::<String>()))?;
    fs::write(out.join("FP4-BANDS.tsv"),format!("tensor\telements\tmax_abs_dequant_round_error\tmax_relative_dequant_round_error\tmax_ulp_error\n{}",
        cp.bands.borrow().values().map(|s| s.clone()+"\n").collect::<String>()))?;
    receipt::seal(out)?; receipt::verify(out)?;
    println!("STACK_REFERENCE_READY profile=logical-f32-v1 device_pass=false"); Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test]
    fn checkpoint_stack_receipts_repeat_and_reject_a_changed_input() {
        let time=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("vade-stack-reference-{}-{time}",std::process::id()));
        self_test(&root).unwrap();
        generate(&root.join("model"),&root.join("different"),"1,3,0,4,1").unwrap();
        assert!(compare(&root.join("first"),&root.join("different")).is_err());
        fs::remove_file(root.join("repeat/SHA256SUMS")).unwrap();
        fs::write(root.join("repeat/TOKENS.tsv"),"changed IDs\n").unwrap();
        receipt::seal(&root.join("repeat")).unwrap();
        assert!(compare(&root.join("first"),&root.join("repeat")).is_err());
        assert!(generate(&root.join("model"),&root.join("first"),"1").is_err());
        assert!(generate(&root.join("model"),&root.join("invalid"),"5").is_err());
        assert!(!root.join("invalid/SHA256SUMS").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
