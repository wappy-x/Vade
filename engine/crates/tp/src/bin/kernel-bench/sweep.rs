use super::*;
use std::process::Command;
pub(super) fn label(value:&str)->Result<()> {
    if value.is_empty() || value.len()>80 || !value.bytes().all(|byte|byte.is_ascii_alphanumeric() || b"-_".contains(&byte)) {return Err("variant label requires 1..80 ASCII letters, digits, - or _".into());}Ok(())
}
#[cfg_attr(not(target_os="linux"),allow(dead_code))]
pub(super) struct Variant {pub label:String,pub image:PathBuf,pub geometry:Option<spark_gpu_probe::fixture_image::Geometry>}
pub(super) fn geometry(words:&[&str])->Result<Option<spark_gpu_probe::fixture_image::Geometry>> {
    if words.is_empty() {return Ok(None);}let (mut kernel,mut scale,mut threads)=(None,1,0);let mut seen=BTreeSet::new();
    for word in words {let (key,value)=word.split_once('=').ok_or("geometry requires key=value")?;
        if !seen.insert(key) {return Err("duplicate geometry key".into());}
        match key {"kernel"=>kernel=Some(value.to_owned()),"grid_scale"=>scale=value.parse().map_err(|_|"invalid grid_scale")?,
            "threads"=>threads=value.parse().map_err(|_|"invalid thread count")?,_=>return Err("unknown geometry key".into())}
    }
    spark_gpu_probe::fixture_image::Geometry::new(kernel.ok_or("geometry requires kernel=ENTRY")?,scale,threads).map(Some)
}
#[cfg_attr(not(target_os="linux"),allow(dead_code))]
pub(super) fn variants(path:&Path)->Result<Vec<Variant>> {
    let text=fs::read_to_string(path).map_err(|error|error.to_string())?;if text.len()>1<<20 {return Err("variant manifest exceeds 1 MiB".into());}
    let mut seen=BTreeSet::new();let mut output=Vec::new();
    for line in text.lines().filter(|line|!line.trim().is_empty() && !line.starts_with('#')) {
        let fields=line.split_whitespace().collect::<Vec<_>>();if fields.len()<2 {return Err("variant manifest requires LABEL IMAGE [geometry] per line".into());}
        label(fields[0])?;if !seen.insert(fields[0]) || output.len()>=256 {return Err("duplicate variant or more than 256 variants".into());}
        let image=PathBuf::from(fields[1]);let image=if image.is_absolute() {image} else {path.parent().unwrap_or(Path::new(".")).join(image)};
        output.push(Variant {label:fields[0].into(),image,geometry:geometry(&fields[2..])?});
    }if output.is_empty() {return Err("empty sweep manifest".into());}Ok(output)
}
pub(super) fn build(arguments:&[String])->Result<()> {
    if arguments.len()!=4 || !matches!(arguments[3].as_str(),"ptx"|"cubin") {return Err("build-sweep SPEC BUILDER OUT ptx|cubin".into());}
    let specification=fs::read_to_string(&arguments[0]).map_err(|error|error.to_string())?;
    if specification.len()>1<<20 {return Err("sweep specification exceeds 1 MiB".into());}
    let out=Path::new(&arguments[2]);fs::create_dir(out).map_err(|error|error.to_string())?;
    let out=fs::canonicalize(out).map_err(|error|error.to_string())?;let mut names=BTreeSet::new();let mut manifest=String::new();
    for line in specification.lines().filter(|line|!line.trim().is_empty() && !line.starts_with('#')) {
        let fields=line.split_whitespace().collect::<Vec<_>>();label(fields[0])?;
        if !names.insert(fields[0]) || names.len()>256 {return Err("duplicate sweep label or more than 256 variants".into());}
        let geometry_words=fields[1..].iter().copied().filter(|word|word.starts_with("kernel=") || word.starts_with("grid_scale=") || word.starts_with("threads=")).collect::<Vec<_>>();
        geometry(&geometry_words)?;
        let filename=format!("{}.{}",fields[0],arguments[3]);let image=out.join(&filename);
        let log=fs::OpenOptions::new().create_new(true).write(true).open(out.join(format!("{}.build.log",fields[0]))).map_err(|error|error.to_string())?;
        let started=Instant::now();let result=Command::new(&arguments[1]).arg(&image).args(&fields[1..]).stdout(log.try_clone().map_err(|error|error.to_string())?).stderr(log).status().map_err(|error|error.to_string())?;
        if !result.success() {return Err(format!("variant {} build failed: {result}",fields[0]));}
        let bytes=fs::read(&image).map_err(|error|error.to_string())?;spark_gpu_probe::fixture_image::validate(&bytes)?;
        println!("BUILD PASS label={} seconds={:.6} sha256={} parameters={:?}",fields[0],started.elapsed().as_secs_f64(),sha256::digest(&bytes),&fields[1..]);
        fs::write(out.join(format!("{}.params",fields[0])),fields[1..].join(" ")).map_err(|error|error.to_string())?;
        manifest.push_str(&format!("{} {filename} {}\n",fields[0],geometry_words.join(" ")));
    }
    if names.is_empty() {return Err("empty build sweep".into());}
    fs::write(out.join("variants.tsv"),manifest).map_err(|error|error.to_string())?;Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn labels_reject_path_traversal_and_whitespace() {
        for value in ["","../escape","with space","a/b","a.b"] {assert!(label(value).is_err());}
        assert!(label("warp32-strip8_4").is_ok());
    }
    #[test] fn manifest_paths_are_relative_to_manifest_and_duplicates_refuse() {
        let root=std::env::temp_dir().join(format!("kernel-bench-manifest-{}",std::process::id()));fs::create_dir(&root).unwrap();
        let path=root.join("variants.tsv");fs::write(&path,"first first.cubin\nsecond second.ptx\n").unwrap();
        let parsed=variants(&path).unwrap();assert_eq!(parsed[0].label,"first");assert_eq!(parsed[0].image,root.join("first.cubin"));
        fs::write(&path,"first a\nfirst b\n").unwrap();assert!(variants(&path).is_err());
        fs::write(&path,"first a ignored\n").unwrap();assert!(variants(&path).is_err());fs::remove_dir_all(root).unwrap();
    }
    #[cfg(unix)]
    #[test] fn sweep_builder_produces_all_parameter_variants_without_a_gpu() {
        use std::os::unix::fs::PermissionsExt;
        let root=std::env::temp_dir().join(format!("kernel-bench-builder-{}",std::process::id()));fs::create_dir(&root).unwrap();
        let builder=root.join("builder.sh");fs::write(&builder,"#!/bin/sh\nset -eu\ncp \"$2\" \"$1\"\n").unwrap();fs::set_permissions(&builder,fs::Permissions::from_mode(0o700)).unwrap();
        let image=root.join("reference.ptx");fs::write(&image,".version 8.0\n.target sm_121\n.visible .entry demo() {}\n").unwrap();
        let spec=root.join("spec.tsv");fs::write(&spec,format!("warp8 {} rows=8 kernel=expert_group grid_scale=4 threads=128\nwarp4 {} rows=4\n",image.display(),image.display())).unwrap();
        let out=root.join("built folder");build(&[spec.display().to_string(),builder.display().to_string(),out.display().to_string(),"ptx".into()]).unwrap();
        let variants=variants(&out.join("variants.tsv")).unwrap();assert_eq!(variants.len(),2);
        let geometry=variants[0].geometry.as_ref().unwrap();assert_eq!(geometry.kernel(),"expert_group");assert_eq!(geometry.dimensions(2,32).unwrap(),(8,128));
        assert!(variants[1].geometry.is_none());assert!(fs::read_to_string(out.join("warp8.params")).unwrap().contains("rows=8"));
        assert_eq!(fs::read(&variants[0].image).unwrap(),fs::read(&image).unwrap());assert!(out.join("warp4.build.log").is_file());
        assert!(build(&[spec.display().to_string(),builder.display().to_string(),out.display().to_string(),"ptx".into()]).is_err());
        let relocated=root.join("moved folder");fs::rename(&out,&relocated).unwrap();
        let moved=super::variants(&relocated.join("variants.tsv")).unwrap();
        assert_eq!(moved[0].image,relocated.join("warp8.ptx"));assert_eq!(moved[1].image,relocated.join("warp4.ptx"));
        for variant in moved {assert_eq!(fs::read(variant.image).unwrap(),fs::read(&image).unwrap());}
        fs::remove_dir_all(root).unwrap();
    }
    #[test] fn geometry_words_are_checked_before_a_launch() {
        assert!(geometry(&["kernel=expert_group","grid_scale=32","threads=128"]).unwrap().is_some());
        assert!(geometry(&["grid_scale=32"]).is_err());assert!(geometry(&["kernel=expert_group","threads=33"]).is_err());
        assert!(geometry(&["kernel=expert_group","kernel=other"]).is_err());assert!(geometry(&["ignored"]).is_err());
    }
}
