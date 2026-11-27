//! Hash and compare complete, named reference planes. Refuse unknown policies.
use engine_model::Result;
use super::invalid;
use std::{collections::BTreeSet,fs,path::{Path,Component}};
pub const POLICY: &str="profile=logical-f32-v1\nmax_absolute_error=0\nmax_relative_error=0\nfinite_required=true\nbyte_rule=all tensor bytes equal; include signed zero, integer IDs, history, and state\nrepeat_rule=3 fresh processes per rank; both rank results equal in global tensor order\nphysical_gpu_acceptance=unqualified; this policy accepts only the stated scalar profile\nfp4_reason=stored E2M1 times E4M3 is exact in FP32; global multiply rounds once\nquantization_loss=not a decoder output tolerance; original prequantized weights are absent\n";
fn safe(name: &str) -> Result<()> {
    if name.is_empty() || Path::new(name).components().count()!=1 || Path::new(name).components().any(|c| !matches!(c,Component::Normal(_))) || name.contains(['\n','\r','\t']) {
        return Err(invalid("invalid receipt path"));
    } Ok(())
}
fn read(root: &Path,name: &str) -> Result<Vec<u8>> {
    safe(name)?;
    if !fs::symlink_metadata(root.join(name))?.file_type().is_file() { return Err(invalid("receipt needs a regular file")); }
    engine_model::safetensors::read_bounded(&root.join(name),64<<20)
}
pub fn encode(dtype: &str,shape: &[u64],payload: &[u8]) -> Vec<u8> {
    let mut header=format!("{{\"value\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[0,{}]}}}}",payload.len()).into_bytes();
    // Canonical metadata padding. It does not change tensor values.
    header.resize(header.len().div_ceil(256)*256,b' ');
    let mut bytes=(header.len() as u64).to_le_bytes().to_vec(); bytes.extend(header); bytes.extend(payload); bytes
}
fn payload(bytes: &[u8],fields: &[String]) -> Result<Vec<u8>> {
    let header=engine_model::safetensors::Header::read(&mut std::io::Cursor::new(bytes),Default::default())?;
    let tensor=header.tensors.get("value").ok_or_else(|| invalid("missing reference value tensor"))?;
    let dtype=match tensor.dtype {engine_model::safetensors::Dtype::F32=>"F32",engine_model::safetensors::Dtype::U32=>"U32",_=>return Err(invalid("invalid reference dtype"))};
    if header.tensors.len()!=1 || !header.metadata.is_empty() || dtype!=fields[1] || format!("{:?}",tensor.shape)!=fields[2]
        || tensor.byte_len().to_string()!=fields[3] { return Err(invalid("reference header differs from manifest")); }
    Ok(bytes[(header.data_start+tensor.start) as usize..(header.data_start+tensor.end) as usize].to_vec())
}
pub fn seal(root: &Path) -> Result<()> {
    let mut names=fs::read_dir(root)?.map(|e| e.map(|e| e.file_name())).collect::<std::io::Result<Vec<_>>>()?; names.sort();
    let mut sums=String::new();
    for name in names { let name=name.to_str().ok_or_else(|| invalid("invalid receipt name"))?; safe(name)?;
        if name=="SHA256SUMS" { return Err(invalid("receipt is already sealed")); }
        sums.push_str(&format!("{}  {name}\n",engine_hash::digest(&read(root,name)?))); }
    fs::write(root.join("SHA256SUMS"),sums)?; Ok(())
}
pub fn verify(root: &Path) -> Result<()> {
    let bytes=read(root,"SHA256SUMS")?; let text=std::str::from_utf8(&bytes).map_err(|_| invalid("checksum list is not UTF-8"))?; let mut seen=BTreeSet::new();
    for row in text.lines() {
        let (hash,name)=row.split_once("  ").ok_or_else(|| invalid("invalid checksum row"))?; safe(name)?;
        if hash.len()!=64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) || !seen.insert(name) { return Err(invalid("invalid or duplicate checksum")); }
        if engine_hash::digest(&read(root,name)?)!=hash { return Err(invalid(format!("checksum differs: {name}"))); }
    }
    if seen.is_empty() { return Err(invalid("empty checksum list")); }
    for entry in fs::read_dir(root)? { let name=entry?.file_name().into_string().map_err(|_| invalid("invalid receipt name"))?;
        if name!="SHA256SUMS" && !seen.contains(name.as_str()) { return Err(invalid("unlisted receipt file")); } }
    println!("REFERENCE_SHA256_OK files={}",seen.len()); Ok(())
}
pub fn compare(expected: &Path,actual: &Path) -> Result<()> {
    verify(expected)?; verify(actual)?;
    if read(expected,"POLICY.txt")?!=POLICY.as_bytes() || read(actual,"POLICY.txt")?!=POLICY.as_bytes() {
        return Err(invalid("comparison policy differs"));
    }
    if read(expected,"PROFILE.txt")?!=read(actual,"PROFILE.txt")? || read(expected,"WEIGHTS.tsv")?!=read(actual,"WEIGHTS.tsv")? {
        return Err(invalid("reference input or weight identity differs"));
    }
    let manifest=|root: &Path| -> Result<Vec<Vec<String>>> {
        let bytes=read(root,"TENSORS.tsv")?; let text=std::str::from_utf8(&bytes).map_err(|_| invalid("tensor list is not UTF-8"))?;
        let mut lines=text.lines();
        if lines.next()!=Some("file\tdtype\tshape\tbytes\tsha256") { return Err(invalid("invalid tensor manifest header")); }
        let mut rows=Vec::new(); let mut names=BTreeSet::new();
        for row in lines { let fields=row.split('\t').map(str::to_owned).collect::<Vec<_>>();
            if fields.len()!=5 || !names.insert(fields[0].clone()) { return Err(invalid("invalid or duplicate tensor row")); }
            safe(&fields[0])?; rows.push(fields);
        } Ok(rows)
    };
    let rows=manifest(expected)?; let other=manifest(actual)?;
    if rows.len()!=other.len() { return Err(invalid("tensor manifest extent differs")); }
    let mut count=0; let mut equal=true; let mut maximum=0f64;
    for (fields,other) in rows.iter().zip(&other) {
        if fields[..4]!=other[..4] { return Err(invalid("tensor name, dtype, shape, or extent differs")); }
        let a=read(expected,&fields[0])?; let b=read(actual,&fields[0])?;
        if engine_hash::digest(&a)!=fields[4] || engine_hash::digest(&b)!=other[4] { return Err(invalid("tensor row hash differs")); }
        let a=payload(&a,fields)?; let b=payload(&b,other)?;
        if a.len()%4!=0 || b.len()!=a.len() { return Err(invalid("tensor byte extent differs")); }
        let mut max_error=0f64;
        match fields[1].as_str() { "F32" => for (av,bv) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
            let (av,bv)=(f32::from_le_bytes(*av),f32::from_le_bytes(*bv));
            if !av.is_finite() || !bv.is_finite() { return Err(invalid("nonfinite tensor output")); }
            max_error=max_error.max((av as f64-bv as f64).abs());
        }, "U32"=>(), _=>return Err(invalid("unknown output dtype")) }
        if a!=b { println!("TENSOR_DIFFERENCE file={} max_absolute_error={max_error:.17e} bytes_equal=false",fields[0]); equal=false; }
        maximum=maximum.max(max_error); count+=1;
    }
    if count==0 { return Err(invalid("no tensor outputs")); }
    println!("SCALAR_REFERENCE_COMPARE tensors={count} max_absolute_error={maximum:.17e} bytes_equal={equal} device_pass=false");
    if !equal { return Err(invalid("tensor bytes differ from the scalar profile")); } Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64,Ordering};
    static NEXT: AtomicU64=AtomicU64::new(0);
    fn fixture(value: f32,policy: &str) -> std::path::PathBuf {
        let root=std::env::temp_dir().join(format!("vade-layer-receipt-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&root).unwrap(); let bytes=value.to_le_bytes();
        fs::write(root.join("POLICY.txt"),policy).unwrap(); fs::write(root.join("PROFILE.txt"),"fixed test input").unwrap();
        fs::write(root.join("WEIGHTS.tsv"),"fixed test weights").unwrap(); let encoded=encode("F32",&[1],&bytes);
        fs::write(root.join("output.safetensors"),&encoded).unwrap();
        fs::write(root.join("TENSORS.tsv"),format!("file\tdtype\tshape\tbytes\tsha256\noutput.safetensors\tF32\t[1]\t4\t{}\n",engine_hash::digest(&encoded))).unwrap();
        seal(&root).unwrap(); root
    }
    #[test] fn reject_receipt_path_escape() { for name in ["", "../x", "/x", "a\nb", "./x", "a/b"] { assert!(safe(name).is_err()); } assert!(safe("layer0.token0.output.bin").is_ok()); }
    #[test] fn reject_header_shape_dtype_and_truncated_payload() {
        let fields=["x.safetensors","F32","[1]","4","unused"].map(str::to_owned);
        let value=1f32.to_le_bytes();
        assert!(payload(&encode("F32",&[1,1],&value),&fields).is_err());
        assert!(payload(&encode("U32",&[1],&value),&fields).is_err());
        let mut truncated=encode("F32",&[1],&value); truncated.pop(); assert!(payload(&truncated,&fields).is_err());
        assert_eq!(payload(&encode("F32",&[1],&value),&fields).unwrap(),value);
    }
    #[test] fn accept_equal_and_reject_numeric_difference() {
        let expected=fixture(1.0,POLICY); let equal=fixture(1.0,POLICY); let different=fixture(1.0001,POLICY);
        compare(&expected,&equal).unwrap(); assert!(compare(&expected,&different).is_err());
    }
    #[test] fn reject_signed_zero_change_and_nonfinite_values() {
        let positive=fixture(0.0,POLICY); let negative=fixture(-0.0,POLICY);
        assert_eq!(compare(&positive,&negative).unwrap_err().to_string(),"tensor bytes differ from the scalar profile");
        let nan=fixture(f32::NAN,POLICY); assert_eq!(compare(&nan,&nan).unwrap_err().to_string(),"nonfinite tensor output");
    }
    #[test] fn reject_changed_policy_corruption_and_missing_files() {
        let expected=fixture(1.0,POLICY); let changed=fixture(1.0,"max_absolute_error=1\n"); assert!(compare(&expected,&changed).is_err());
        let corrupt=fixture(1.0,POLICY); fs::write(corrupt.join("output.safetensors"),2f32.to_le_bytes()).unwrap(); assert!(verify(&corrupt).is_err());
        let missing=fixture(1.0,POLICY); fs::rename(missing.join("output.safetensors"),missing.join("moved.safetensors")).unwrap(); assert!(verify(&missing).is_err());
    }
}
