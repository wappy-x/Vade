use super::{Path,PathBuf,Result,fs,sha256};
use std::collections::{BTreeSet,VecDeque};

pub struct Image {pub name:String,pub bytes:Result<Vec<u8>>,pub probe:Option<(String,Option<u32>)>,pub geometry:Option<spark_gpu_probe::fixture_image::Geometry>}
pub struct Plan {pub images:Vec<Image>,pub identity:String}
pub struct Arm {pub name:String,pub image:usize,pub baseline:bool}

impl Plan {
    pub fn read(path:&Path)->Result<Self> {
        let text=fs::read_to_string(path).map_err(|error|error.to_string())?;
        let root=path.parent().unwrap_or_else(||Path::new("."));
        let mut images=Vec::new();let mut identity=String::new();let mut names=BTreeSet::new();
        for line in text.lines().map(str::trim).filter(|line|!line.is_empty() && !line.starts_with('#')) {
            let fields=line.split_whitespace().collect::<Vec<_>>();
            if !matches!(fields.len(),2|4|5|7) {return Err("warm image row needs LABEL IMAGE [PROJECTION_WEIGHT EXPECTED_HEX_BITS_OR_any] [ENTRY GRID_SCALE THREADS]".into());}
            let probe=if matches!(fields.len(),4|7) {
                if fields[2].len()>300 || !fields[2].bytes().all(|byte|byte.is_ascii_alphanumeric() || byte==b'_' || byte==b'.') {return Err("invalid warm projection weight".into());}
                let expected=if fields[3]=="any" {None} else {let bits=u32::from_str_radix(fields[3],16).map_err(|_|"invalid warm marker bits")?;
                    if !f32::from_bits(bits).is_finite() {return Err("warm marker must be finite".into());}Some(bits)};
                Some((fields[2].to_owned(),expected))
            } else {None};
            let geometry=if matches!(fields.len(),5|7) {let offset=fields.len()-3;
                Some(spark_gpu_probe::fixture_image::Geometry::new(fields[offset].to_owned(),fields[offset+1].parse().map_err(|_|"invalid warm grid scale")?,fields[offset+2].parse().map_err(|_|"invalid warm threads")?)?)
            } else {None};
            let name=fields[0];
            if name.is_empty() || !name.bytes().all(|byte|byte.is_ascii_alphanumeric() || byte==b'-' || byte==b'_')
                || name=="final-baseline" || name.starts_with("recovery-") || !names.insert(name.to_owned()) {return Err("invalid or duplicate warm image label".into());}
            if images.is_empty() && name!="baseline" {return Err("warm image list must start with baseline IMAGE".into());}
            let image=root.join(fields[1]);
            let bytes=(|| {let size=fs::metadata(&image).map_err(|error|error.to_string())?.len();
                if size==0 || size>64<<20 {return Err("warm image exceeds its 64 MiB bound".into());}
                fs::read(&image).map_err(|error|error.to_string())})();
            identity.push_str(&format!("warm_image={name} sha256={}\n",bytes.as_ref().map_or_else(|_|"unavailable".to_string(),|bytes|sha256::digest(bytes))));
            identity.push_str(&format!("projection_probe={probe:?} geometry={geometry:?}\n"));
            images.push(Image {name:name.to_owned(),bytes,probe,geometry});
            if images.len()>17 {return Err("warm image list exceeds 16 candidates".into());}
        }
        if images.len()<2 {return Err("warm image list requires a baseline and a candidate".into());}
        if let Err(error)=&images[0].bytes {return Err(format!("warm baseline image: {error}"));}
        Ok(Self {images,identity})
    }
    pub fn arms(&self)->VecDeque<Arm> {
        let mut arms=self.images.iter().enumerate().map(|(image,value)|Arm {name:value.name.clone(),image,baseline:image==0}).collect::<VecDeque<_>>();
        arms.push_back(Arm {name:"final-baseline".into(),image:0,baseline:true});arms
    }
}
pub fn recovery(name:&str)->Arm {Arm {name:format!("recovery-{name}"),image:0,baseline:true}}
pub fn gate(jobs:&[(Vec<u32>,usize,PathBuf,usize)])->Result<()> {
    if ![15,127].into_iter().all(|length|jobs.iter().any(|(input,_,_,_)|input.len()==length))
        || jobs.iter().any(|(_,tokens,_,repeats)|*tokens!=64 || *repeats!=3) {return Err("warm gate requires 15-token and 127-token prompts, each with 64 outputs and 3 sessions".into());}
    let mut outputs=BTreeSet::new();
    if jobs.iter().any(|(_,_,out,_)|!outputs.insert(out)) {return Err("duplicate warm job output".into());}Ok(())
}
pub fn compare_exact(reference:&Path,candidate:&Path,tokens:usize)->Result<()> {
    if super::verify(reference)?.len()!=tokens || super::verify(candidate)?.len()!=tokens {return Err("warm output token count differs".into());}
    for file in ["INPUT_IDS".to_string()].into_iter().chain((0..tokens).map(|token|format!("{token}-logits.f32"))).chain(["PROMPT_LOGITS".to_string(),"OUTPUT_IDS".to_string()]) {
        let left=fs::read(reference.join(&file)).map_err(|error|error.to_string())?;let right=fs::read(candidate.join(&file)).map_err(|error|error.to_string())?;
        if left!=right {let byte=left.iter().zip(&right).position(|(left,right)|left!=right).unwrap_or(left.len().min(right.len()));
            return Err(format!("{file} first_differing_byte={byte} baseline={:?} candidate={:?}",left.get(byte),right.get(byte)));}
    }
    println!("WARM COMPARE PASS tokens={tokens} complete_logits=true prompt_hashes=true ids=true");Ok(())
}
#[cfg(target_os="linux")]
pub fn report(out:&Path,arm:&str,repeats:usize)->Result<()> {
    let mut rates=Vec::new();
    for repeat in 1..=repeats {let text=fs::read_to_string(out.join(arm).join(format!("run{repeat}/TIMING"))).map_err(|error|error.to_string())?;
        let rate=text.split_whitespace().find_map(|word|word.strip_prefix("decode_tokens_per_second=")).ok_or("missing warm decode rate")?.parse::<f64>().map_err(|error|error.to_string())?;
        if !rate.is_finite() || rate<=0.0 {return Err("invalid warm decode rate".into());}rates.push(rate);}
    rates.sort_by(f64::total_cmp);println!("WARM RATE arm={arm} job={} sessions={repeats} median_decode_tokens_per_second={:.9}",out.display(),rates[rates.len()/2]);Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn manifest_identity_recovery_and_rejection_are_explicit() {
        let root=std::env::temp_dir().join(format!("warm-plan-{}",std::process::id()));fs::create_dir(&root).unwrap();
        fs::write(root.join("base.ptx"),b"baseline").unwrap();fs::write(root.join("candidate.ptx"),b"candidate").unwrap();
        let path=root.join("images.tsv");fs::write(&path,"baseline base.ptx\nnext candidate.ptx\nmissing absent.cubin\n").unwrap();
        let plan=Plan::read(&path).unwrap();assert_eq!(plan.images.len(),3);assert!(plan.images[2].bytes.is_err());
        assert!(plan.identity.contains("warm_image=next sha256="));assert!(plan.identity.contains("warm_image=missing sha256=unavailable"));
        let mut arms=plan.arms();assert!(arms.pop_front().unwrap().baseline);assert_eq!(arms.pop_front().unwrap().image,1);
        arms.push_front(recovery("next"));let recovered=arms.pop_front().unwrap();assert!(recovered.baseline);assert_eq!(recovered.image,0);
        assert_eq!(arms.pop_front().unwrap().name,"missing");assert_eq!(arms.pop_front().unwrap().name,"final-baseline");
        for text in ["next candidate.ptx", "baseline base.ptx\nnext candidate.ptx\nnext candidate.ptx", "baseline base.ptx\n../escape candidate.ptx", "baseline base.ptx\nfinal-baseline candidate.ptx"] {fs::write(&path,text).unwrap();assert!(Plan::read(&path).is_err());}
        fs::remove_dir_all(root).unwrap();
    }
    #[test] fn standard_gate_rejects_shortened_sessions_and_duplicate_outputs() {
        let jobs=vec![(vec![1;15],64,PathBuf::from("short"),3),(vec![2;127],64,PathBuf::from("long"),3)];assert!(gate(&jobs).is_ok());
        let mut invalid=jobs.clone();invalid[1].1=32;assert!(gate(&invalid).is_err());
        let mut invalid=jobs.clone();invalid[1].3=1;assert!(gate(&invalid).is_err());
        let mut invalid=jobs.clone();invalid[1].2=invalid[0].2.clone();assert!(gate(&invalid).is_err());assert!(gate(&jobs[..1]).is_err());
    }
    #[test] fn marker_contract_is_bound_into_the_pair_identity() {
        let root=std::env::temp_dir().join(format!("warm-marker-plan-{}",std::process::id()));fs::create_dir(&root).unwrap();fs::write(root.join("base.ptx"),b"baseline").unwrap();
        let path=root.join("images.tsv");fs::write(&path,"baseline base.ptx layer.weight any\nwrong base.ptx layer.weight 3f800000\n").unwrap();
        let plan=Plan::read(&path).unwrap();assert_eq!(plan.images[1].probe,Some(("layer.weight".into(),Some(1.0f32.to_bits()))));assert!(plan.identity.contains("layer.weight"));
        fs::write(&path,"baseline base.ptx\nnext base.ptx layer.weight any project_f32_ordered 4 32\n").unwrap();
        let plan=Plan::read(&path).unwrap();assert_eq!(plan.images[1].geometry.as_ref().unwrap().dimensions(128,128).unwrap(),(512,32));
        for marker in ["7fc00000","invalid"] {fs::write(&path,format!("baseline base.ptx layer.weight any\nwrong base.ptx layer.weight {marker}\n")).unwrap();assert!(Plan::read(&path).is_err());}
        fs::remove_dir_all(root).unwrap();
    }
    #[test] fn byte_compare_rejects_equal_prefix_with_missing_outputs() {
        let root=std::env::temp_dir().join(format!("warm-compare-{}",std::process::id()));fs::create_dir(&root).unwrap();
        let reference=root.join("reference");let candidate=root.join("candidate");fs::create_dir(&reference).unwrap();fs::create_dir(&candidate).unwrap();
        for directory in [&reference,&candidate] {fs::write(directory.join("INPUT_IDS"),b"1\n").unwrap();fs::write(directory.join("PROMPT_LOGITS"),b"hash\n").unwrap();fs::write(directory.join("OUTPUT_IDS"),b"0\n").unwrap();fs::write(directory.join("0-logits.f32"),1.0f32.to_le_bytes()).unwrap();}
        assert!(compare_exact(&reference,&candidate,1).is_ok());assert!(compare_exact(&reference,&candidate,2).is_err());
        fs::write(candidate.join("0-logits.f32"),2.0f32.to_le_bytes()).unwrap();assert!(compare_exact(&reference,&candidate,1).unwrap_err().contains("0-logits.f32 first_differing_byte=2"));fs::remove_dir_all(root).unwrap();
    }
}
