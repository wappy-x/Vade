#![forbid(unsafe_code)]
use std::{collections::{BTreeMap,BTreeSet},fs::{self,File},io::{self,Write},path::{Component,Path,PathBuf}};
#[path="../../../cli/src/sha256.rs"] mod sha256;
type Result<T> = io::Result<T>;
fn require(condition:bool,message:&str)->Result<()> {if condition {Ok(())} else {Err(io::Error::other(message))}}
fn scan(root:&Path)->Result<Vec<PathBuf>> {
    require(fs::symlink_metadata(root)?.file_type().is_dir(),"receipt root is not a directory")?;
    let mut files=Vec::new();
    for entry in fs::read_dir(root)? {let entry=entry?;let kind=entry.file_type()?;
        if kind.is_dir() {files.extend(scan(&entry.path())?);} else {require(kind.is_file(),"receipt contains a symlink or special file")?;files.push(entry.path());}
    }files.sort();Ok(files)
}
fn relative(root:&Path,path:&Path)->Result<String> {
    let value=path.strip_prefix(root).map_err(io::Error::other)?.to_str().ok_or_else(||io::Error::other("non-UTF8 receipt path"))?;
    require(!value.contains(['\n','\r','\t']),"receipt path has a control character")?;Ok(value.into())
}
fn safe_path(root:&Path,value:&str)->Result<PathBuf> {
    require(!value.is_empty() && Path::new(value).components().all(|part|matches!(part,Component::Normal(_))),"manifest path is not relative")?;
    Ok(root.join(value))
}
fn digest(path:&Path)->Result<(u64,String)> {let file=File::open(path)?;let bytes=file.metadata()?.len();Ok((bytes,sha256::digest_reader(file)?))}
fn total(root:&Path)->Result<u64> {scan(root)?.into_iter().try_fold(0u64,|bytes,path|Ok(bytes+fs::metadata(path)?.len()))}
fn save_new(path:&Path,bytes:&[u8])->Result<()> {let mut file=File::options().write(true).create_new(true).open(path)?;file.write_all(bytes)?;file.sync_all()}
#[derive(Clone)] struct Entry {path:String,bytes:u64,hash:String,reference:String,reference_bytes:u64,reference_hash:String,equal:bool,keep:bool,setting:Option<String>}
fn compact(root:&Path)->Result<()> {
    require(!root.join("LOGITS-SHA256.tsv").exists(),"retention manifest already exists; use verify")?;
    require(root.join("RUN-FINISHED").is_file() && root.join("TABLE.tsv").is_file() && root.join("SCREEN-STATUS").is_file(),"full screen and checker must finish before retention")?;
    let config=fs::read_to_string(root.join("CONFIG"))?;let config=config.split_whitespace().collect::<Vec<_>>();require(config.len()==6,"invalid screen CONFIG")?;
    let ranks=[config[3],config[4]];
    for rank in ranks {require(fs::read_to_string(root.join(rank).join("run.exit"))?.trim()=="0","rank did not exit successfully")?;}
    let variants=fs::read_to_string(root.join("VARIANTS.tsv"))?;let variants=variants.lines().map(|line|line.split('\t').collect::<Vec<_>>()).collect::<Vec<_>>();
    require(!variants.is_empty() && variants.iter().all(|fields|fields.len()==5),"invalid variant manifest")?;let first=variants[0][0];require(variants[0][4]=="head","first variant is not the head")?;
    let table=fs::read_to_string(root.join("TABLE.tsv"))?;let mut failed_cuts=BTreeSet::new();
    for line in table.lines().filter(|line|!line.starts_with('#')) {let fields=line.split('\t').collect::<Vec<_>>();if fields[0]=="cut" {continue;}
        require(fields.len()>=2 && ["PASS","FAIL"].contains(&fields[1]),"unrecognized exactness verdict")?;if fields[1]=="FAIL" {failed_cuts.insert(fields[0]);}
    }
    let mut failed_settings=variants.iter().filter(|fields|failed_cuts.contains(fields[4])).map(|fields|fields[0].to_owned()).collect::<BTreeSet<_>>();
    let files=scan(root)?;let before=files.iter().try_fold(0u64,|bytes,path|Ok::<_,io::Error>(bytes+fs::metadata(path)?.len()))?;
    let mut references=BTreeMap::<String,(u64,String)>::new();let mut entries=Vec::new();
    for path in files.iter().filter(|path|path.file_name().unwrap().to_string_lossy().ends_with("-logits.f32")) {
        let name=relative(root,path)?;let parts=name.split('/').collect::<Vec<_>>();let filename=parts.last().unwrap();
        require(filename.trim_end_matches("-logits.f32").parse::<usize>().is_ok(),"invalid logits filename")?;
        let (job,setting,canonical)=if parts.len()==5 && ranks.contains(&parts[0]) {
            require(variants.iter().any(|fields|fields[0]==parts[1]),"unknown logits setting")?;
            let job=if parts[2]=="warmup" {"glm-tokens64"} else {parts[2]};
            (job,Some(parts[1].to_owned()),parts[0]==ranks[0] && parts[1]==first && parts[2]!="warmup" && parts[3]=="run1")
        } else if ["golden","reference"].contains(&parts[0]) {
            let job=parts.iter().copied().find(|part|["glm-tokens64","out127","out1040","outlong"].contains(part)).ok_or_else(||io::Error::other("unknown reference logits job"))?;
            (job,None,false)
        } else {return Err(io::Error::other(format!("unexpected logits path {name}")));};
        let reference=format!("{}/{first}/{job}/run1/{filename}",ranks[0]);
        if !references.contains_key(&reference) {references.insert(reference.clone(),digest(&safe_path(root,&reference)?)?);}
        let (reference_bytes,reference_hash)=&references[&reference];let (bytes,hash)=digest(path)?;
        let equal=bytes==*reference_bytes && hash==*reference_hash;
        if !equal {if let Some(setting)=&setting {failed_settings.insert(setting.clone());}}
        entries.push(Entry {path:name,bytes,hash,reference,reference_bytes:*reference_bytes,reference_hash:reference_hash.clone(),equal,keep:canonical,setting});
    }
    require(!entries.is_empty(),"no logits files found")?;
    let mut manifest=String::from("path\tbytes\tsha256\treference\treference_bytes\treference_sha256\texact\tretain\n");
    for entry in &mut entries {entry.keep |= !entry.equal || entry.setting.as_ref().is_some_and(|setting|failed_settings.contains(setting));
        manifest.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",entry.path,entry.bytes,entry.hash,entry.reference,entry.reference_bytes,entry.reference_hash,entry.equal,entry.keep));
    }
    save_new(&root.join("LOGITS-SHA256.tsv"),manifest.as_bytes())?;
    save_new(&root.join("LOGITS-MANIFEST.sha256"),format!("{}  LOGITS-SHA256.tsv\n",sha256::sha(manifest.as_bytes())).as_bytes())?;
    let mut verdicts=String::new();for name in ["TABLE.tsv","PAIRED.tsv","DIFFERENCES.txt","HEAD-LIVE.tsv","SCREEN-STATUS"] {
        let (_,hash)=digest(&root.join(name))?;verdicts.push_str(&format!("{hash}  {name}\n"));
    }save_new(&root.join("FULL-BYTE-VERDICT.sha256"),verdicts.as_bytes())?;
    let mut removed=0u64;let mut kept=0usize;
    for entry in &entries {if entry.keep {kept+=1;} else {require(entry.equal,"refusing to delete failed logits")?;let path=safe_path(root,&entry.path)?;
        require(fs::metadata(&path)?.len()==entry.bytes,"logits size changed during retention")?;fs::remove_file(path)?;removed+=entry.bytes;}}
    verify(root)?;
    fs::write(root.join("RETENTION-SOURCE.rs"),include_str!("flywheel-retain.rs").replace("../../../cli/src/sha256.rs","RETENTION-SHA256.rs").replace("flywheel-retain.rs","RETENTION-SOURCE.rs"))?;
    fs::write(root.join("RETENTION-SHA256.rs"),include_str!("../../../cli/src/sha256.rs"))?;
    let failed_settings=failed_settings.into_iter().collect::<Vec<_>>().join(",");let mut size=total(root)?;
    for _ in 0..5 {
        fs::write(root.join("RETENTION.txt"),format!("policy=full-byte-check-then-sha256; canonical-head-once; retain-failed-variants\nfull_byte_verdict=FULL-BYTE-VERDICT.sha256\nlogits_files={}\nretained_logits_files={kept}\nremoved_logits_bytes={removed}\nreceipt_bytes_before={before}\nreceipt_bytes_after={size}\nfailed_settings={failed_settings}\nfull_logit_recheck=retained_hashes_against_canonical; original driver needs restored full files\n",entries.len()))?;
        let actual=total(root)?;if actual==size {break;}size=actual;
        if size==0 {return Err(io::Error::other("empty retained receipt"));}
    }
    println!("RETENTION PASS logits={} kept={kept} removed_bytes={removed} receipt_bytes={}",entries.len(),total(root)?);Ok(())
}
fn verify(root:&Path)->Result<()> {
    let files=scan(root)?;let manifest=fs::read_to_string(root.join("LOGITS-SHA256.tsv"))?;
    let hash=fs::read_to_string(root.join("LOGITS-MANIFEST.sha256"))?;require(hash==format!("{}  LOGITS-SHA256.tsv\n",sha256::sha(manifest.as_bytes())),"retention manifest hash differs")?;
    for line in fs::read_to_string(root.join("FULL-BYTE-VERDICT.sha256"))?.lines() {
        let (hash,name)=line.split_once("  ").ok_or_else(||io::Error::other("invalid verdict digest record"))?;
        require(digest(&safe_path(root,name)?)?.1==hash,"full-byte verdict file changed")?;
    }
    let mut known=BTreeSet::new();let mut references=BTreeMap::<String,(u64,String)>::new();
    for (index,line) in manifest.lines().enumerate() {let fields=line.split('\t').collect::<Vec<_>>();if index==0 {require(line=="path\tbytes\tsha256\treference\treference_bytes\treference_sha256\texact\tretain","invalid hash manifest header")?;continue;}
        require(fields.len()==8 && ["true","false"].contains(&fields[6]) && ["true","false"].contains(&fields[7]),"invalid hash manifest row")?;
        require(known.insert(fields[0]),"duplicate hash manifest path")?;
        let bytes=fields[1].parse::<u64>().map_err(io::Error::other)?;let reference_bytes=fields[4].parse::<u64>().map_err(io::Error::other)?;
        if !references.contains_key(fields[3]) {references.insert(fields[3].into(),digest(&safe_path(root,fields[3])?)?);}
        require(references[fields[3]]==(reference_bytes,fields[5].into()),"canonical head bytes or hash differ")?;
        let equal=bytes==reference_bytes && fields[2]==fields[5];require(equal==(fields[6]=="true"),"saved equality verdict differs")?;
        let path=safe_path(root,fields[0])?;
        if fields[7]=="true" {require(digest(&path)?==(bytes,fields[2].into()),"retained logits differ")?;} else {require(equal && !path.exists(),"pruned logits were not equal or are still present")?;}
    }
    for path in files.iter().filter(|path|path.file_name().unwrap().to_string_lossy().ends_with("-logits.f32")) {require(known.contains(relative(root,path)?.as_str()),"unlisted logits remain")?;}
    Ok(())
}
/// The authorized receipt root: the RECEIPTS line of the file named by VADE_KIT_HOSTS (see tools/kit/hosts.example).
fn receipt_root()->Result<String> {
    let path=std::env::var("VADE_KIT_HOSTS").map_err(|_|io::Error::other("set VADE_KIT_HOSTS to the private hosts file"))?;
    fs::read_to_string(&path)?.lines().find_map(|line|match line.split('#').next().unwrap_or("").split_whitespace().collect::<Vec<_>>()[..] {["RECEIPTS",root]=>Some(root.to_owned()),_=>None})
        .ok_or_else(||io::Error::other("the hosts file has no RECEIPTS line"))
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let result=(|| {require(args.len()==2,"usage: flywheel-retain compact|verify RECEIPT")?;let root=fs::canonicalize(&args[1])?;
        let allowed=receipt_root()?;require(!allowed.is_empty() && root.starts_with(&allowed),"outside authorized flywheel receipts")?;
        match args[0].as_str() {"compact"=>compact(&root),"verify"=>verify(&root),_=>Err(io::Error::other("unknown retention action"))}
    })();if let Err(error)=result {eprintln!("RETENTION FAIL {error}");std::process::exit(1);}
}
#[cfg(test)] mod tests {
    use super::*;
    fn fixture()->PathBuf {
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();static FIXTURES:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);let serial=FIXTURES.fetch_add(1,std::sync::atomic::Ordering::Relaxed);let root=std::env::temp_dir().join(format!("flywheel-retain-{}-{nonce}-{serial}",std::process::id()));fs::create_dir(&root).unwrap();
        fs::write(root.join("CONFIG"),"base test revision left right short\n").unwrap();fs::write(root.join("VARIANTS.tsv"),"head-00\thead\t\t0\thead\np1-good\thead\t\t1\tgood\np1-bad\thead\t\t1\tbad\n").unwrap();
        fs::write(root.join("TABLE.tsv"),"cut\texactness\ngood\tPASS\nbad\tFAIL\n").unwrap();for name in ["RUN-FINISHED","SCREEN-STATUS","PAIRED.tsv","DIFFERENCES.txt","HEAD-LIVE.tsv"] {fs::write(root.join(name),"fixture\n").unwrap();}
        for rank in ["left","right"] {fs::create_dir(root.join(rank)).unwrap();fs::write(root.join(rank).join("run.exit"),"0\n").unwrap();
            for setting in ["head-00","p1-good","p1-bad"] {for job in ["warmup","glm-tokens64","out127"] {let directory=root.join(rank).join(setting).join(job).join("run1");fs::create_dir_all(&directory).unwrap();
                fs::write(directory.join("0-logits.f32"),if setting=="p1-bad" {1.0f32} else {0.0f32}.to_le_bytes()).unwrap();fs::write(directory.join("OUTPUT_IDS"),"1\n").unwrap();}}
        }root
    }
    #[test] fn compaction_preserves_failed_variants_and_verifies_removed_hashes() {
        let root=fixture();compact(&root).unwrap();verify(&root).unwrap();
        assert!(root.join("left/head-00/glm-tokens64/run1/0-logits.f32").exists());assert!(!root.join("right/head-00/glm-tokens64/run1/0-logits.f32").exists());
        assert!(!root.join("left/p1-good/glm-tokens64/run1/0-logits.f32").exists());assert!(root.join("left/p1-good/glm-tokens64/run1/OUTPUT_IDS").exists());
        assert!(root.join("right/p1-bad/glm-tokens64/run1/0-logits.f32").exists());assert!(compact(&root).is_err());
        fs::write(root.join("left/head-00/glm-tokens64/run1/0-logits.f32"),2.0f32.to_le_bytes()).unwrap();assert!(verify(&root).is_err());fs::remove_dir_all(root).unwrap();
    }
    #[test] fn an_unreported_mismatch_keeps_the_whole_variant() {
        let root=fixture();fs::write(root.join("left/p1-good/glm-tokens64/run1/0-logits.f32"),3.0f32.to_le_bytes()).unwrap();compact(&root).unwrap();
        assert!(root.join("right/p1-good/out127/run1/0-logits.f32").exists());verify(&root).unwrap();fs::remove_dir_all(root).unwrap();
    }
    #[test] fn incomplete_run_does_not_delete_logits() {
        let root=fixture();fs::remove_file(root.join("RUN-FINISHED")).unwrap();assert!(compact(&root).is_err());assert!(root.join("left/p1-good/glm-tokens64/run1/0-logits.f32").exists());fs::remove_dir_all(root).unwrap();
    }
}
