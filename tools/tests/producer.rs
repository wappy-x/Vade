#![cfg(unix)]
mod common;
use common::*;
use std::{fs,collections::BTreeMap,path::PathBuf,process::Command};
struct Producer { temp: Temp, output: PathBuf, env: BTreeMap<String,String> }
impl Producer {
    fn new() -> Self {
        let temp = Temp::new("bundle producer"); let root = &temp.0;
        write(&root.join("scripts/make-bundle.sh"),fs::read(vade_tools::root().join("scripts/make-bundle.sh")).unwrap());
        for name in ["LICENSE","NOTICE","engine/LICENSE","engine/NOTICE","engine/licenses/LICENSE",
            "engine/crates/tokenizer/UNICODE-NOTICE.txt","engine/crates/tokenizer/QWEN-NOTICE.txt","cli/Cargo.toml","harness/Cargo.toml"] {
            write(&root.join("fixture").join(name),"fixture notice or manifest\n");
        }
        write(&root.join("fixture/engine/crates/cuda.rs"),"fn cuInit(flags: u32);\n");
        for name in ["uname","git","cargo","rustup","cc","readelf","tar","zstd","sha256sum","rustc-probe","mv"] {
            executable(&root.join("bin").join(name),&format!("#!/bin/sh\nexec {} {} \"$@\"\n",quote(env!("CARGO_BIN_EXE_bundle-fixture")),quote(name)));
        }
        let output = root.join("published");
        let env = [("PATH",format!("{}:{}",root.join("bin").display(),std::env::var("PATH").unwrap())),
            ("TEST_ROOT",root.display().to_string()),("TEST_OUTPUT",output.display().to_string()),
            ("TEST_REAL_MV",which("mv")),("TEST_REAL_TAR",which("tar"))].into_iter().map(|(k,v)|(k.into(),v)).collect();
        Self {temp,output,env}
    }
    fn produce(&self, mode: &str, succeeds: bool) {
        let result = run(Command::new("sh").arg(self.temp.0.join("scripts/make-bundle.sh")).arg(&self.output).envs(&self.env).env("TEST_MODE",mode),&self.temp.0);
        assert_eq!(result.status.success(),succeeds,"{}",diagnostic(&result));
        assert!(!self.temp.0.join("published.lock").exists());
        for directory in [&self.temp.0,&self.temp.0.join(".local")] {
            if directory.exists() { assert!(fs::read_dir(directory).unwrap().all(|p| {let s = p.unwrap().file_name(); let s = s.to_string_lossy(); !s.starts_with(".bundle-publish.") && !s.starts_with("bundle-build.") })); }
        }
    }
    fn check(&self) {
        let result = Command::new(env!("CARGO_BIN_EXE_vade-tools")).arg("check-bundle").arg(&self.output)
            .args([self.temp.0.to_str().unwrap(),&std::env::var("HOME").unwrap(),"bundle-build."]).envs(&self.env).output().unwrap();
        assert!(result.status.success(),"{}",diagnostic(&result)); assert_eq!(fs::read_dir(&self.output).unwrap().count(),2);
    }
    fn files(&self) -> BTreeMap<String,Vec<u8>> { fs::read_dir(&self.output).unwrap().map(|p| { let p = p.unwrap(); (p.file_name().to_str().unwrap().into(),fs::read(p.path()).unwrap()) }).collect() }
}
#[test] fn layout_manifest_hashes_and_compiler_path_sanitization() { let p = Producer::new(); p.produce("",true); p.check(); }
#[test] fn existing_pair_and_empty_directory_are_preserved() {
    let p = Producer::new(); p.produce("",true); let before = p.files(); p.produce("",false); assert_eq!(p.files(),before);
    fs::remove_dir_all(&p.output).unwrap(); fs::create_dir(&p.output).unwrap(); p.produce("",false); assert!(p.files().is_empty());
}
#[test] fn dangling_output_symlink_is_preserved() {
    let p = Producer::new(); std::os::unix::fs::symlink(p.temp.0.join("absent"),&p.output).unwrap(); p.produce("",false); assert!(p.output.is_symlink());
}
#[test] fn competing_producer_cannot_publish() { let p = Producer::new(); p.produce("concurrent",true); p.check(); }
#[test] fn destination_appearing_during_build_is_preserved() {
    for mode in ["destination","empty-destination","rename-race","empty-rename-race"] {
        let p = Producer::new(); p.produce(mode,false);
        if matches!(mode,"destination"|"rename-race") { assert_eq!(fs::read_to_string(p.output.join("existing")).unwrap(),"keep me"); }
        else { assert!(p.files().is_empty()); }
    }
}
#[test] fn refusals_and_failure_cleanup() {
    for mode in ["dirty","build-fails","interrupt","path-leak","remapped-path-leak","wrong-arch","rpath"] {
        let p = Producer::new(); p.produce(mode,false); assert!(!p.output.exists(),"{mode}");
    }
}
