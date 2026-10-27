#![cfg(unix)]
mod common;
use common::*;
use std::{collections::BTreeMap,fs,process::Command};
use vade_tools::{bundle,sha256};
const REV: &str = "1234567890abcdef1234567890abcdef12345678";
struct Fixture { files: BTreeMap<String,Vec<u8>>, manifest: BTreeMap<String,String> }
impl Fixture {
    fn new() -> Self {
        let mut files = BTreeMap::new();
        for name in ["LICENSE","NOTICE","engine/LICENSE","engine/NOTICE","engine/licenses/LICENSE",
            "engine/crates/tokenizer/UNICODE-NOTICE.txt","engine/crates/tokenizer/QWEN-NOTICE.txt"] { files.insert(name.into(),b"notice\n".to_vec()); }
        for name in ["flash-serve","flash.ptx","vade-harness","vade"] { files.insert(name.into(),format!("portable {name}").into_bytes()); }
        let mut manifest: BTreeMap<String,String> = [("format","vade-launch-bundle-v1"),("target","aarch64-unknown-linux-gnu"),
            ("engine_sha",REV),("cli_sha",REV),("harness_sha",REV)].into_iter().map(|(k,v)|(k.into(),v.into())).collect();
        for (key,name) in [("engine","flash-serve"),("ptx","flash.ptx"),("harness","vade-harness"),("vade","vade")] {
            manifest.insert(format!("{key}_sha256"),sha256::digest(&files[name]));
        } Self {files,manifest}
    }
    fn manifest(&self) -> Vec<u8> { self.manifest.iter().map(|(k,v)|format!("{k}={v}\n")).collect::<String>().into_bytes() }
    fn tar(&self, manifest: Option<Vec<u8>>, extra: &[(&str,&[u8])], mode: u64, kind: u8) -> Vec<u8> {
        let mut files = self.files.clone(); files.insert("launcher-bundle.conf".into(),manifest.unwrap_or_else(|| self.manifest()));
        let mut tar = Vec::new();
        for (name,data) in files.iter().map(|(n,d)|(n.as_str(),d.as_slice())).chain(extra.iter().copied()) {
            let mut h = [0;512]; h[..name.len()].copy_from_slice(name.as_bytes());
            let kind = if name == "vade" {kind} else {b'0'}; let size = if matches!(kind,0|b'0') {data.len()} else {0};
            h[100..108].copy_from_slice(format!("{:07o}\0",if name == "vade" {mode} else {0o755}).as_bytes());
            h[124..136].copy_from_slice(format!("{size:011o}\0").as_bytes()); h[156] = kind;
            h[148..156].fill(b' '); let sum: u64 = h.iter().map(|&b|u64::from(b)).sum();
            h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes()); tar.extend(h);
            tar.extend_from_slice(&data[..size]); tar.resize(tar.len()+(512-size%512)%512,0);
        } tar.resize(tar.len()+1024,0); tar
    }
    fn valid(&self) -> Vec<u8> { self.tar(None,&[],0o755,b'0') }
}
fn check(raw: &[u8], forbidden: &[String]) -> Result<(),String> { bundle::check_tar(raw,&REV[..7],forbidden).map_err(|e|e.to_string()) }
fn error(raw: Vec<u8>, expected: &str) { assert_eq!(check(&raw,&[]).unwrap_err(),expected); }
#[test] fn valid_archive_is_checked_without_extraction() {
    let t = Temp::new("bundle checker"); let archive = format!("vade-launch-bundle-v1-{}-aarch64.tar.zst",&REV[..7]);
    let mut raw = Fixture::new().valid(); raw.resize(raw.len()+1024*1024,0); write(&t.0.join(&archive),&raw);
    write(&t.0.join("SHA256SUMS"),format!("{}  {archive}\n",sha256::digest(&raw)));
    executable(&t.0.join("bin/zstd"),"#!/bin/sh\nfor arg do file=$arg; done\ncat \"$file\"\n");
    let before = fs::read_dir(&t.0).unwrap().count();
    let out = Command::new(env!("CARGO_BIN_EXE_vade-tools")).arg("check-bundle").arg(&t.0)
        .env("PATH",format!("{}:{}",t.0.join("bin").display(),std::env::var("PATH").unwrap())).output().unwrap();
    assert!(out.status.success(),"{}",diagnostic(&out)); assert_eq!(fs::read_dir(&t.0).unwrap().count(),before);
    assert_eq!(fs::read(t.0.join(&archive)).unwrap(),raw);
}
#[test] fn checksum_rejects_corruption_and_malformed_lines() {
    let t = Temp::new("bundle corruption"); let name = format!("vade-launch-bundle-v1-{}-aarch64.tar.zst",&REV[..7]);
    write(&t.0.join(&name),b"corruption"); let valid = format!("{}  {name}\n","0".repeat(64)); write(&t.0.join("SHA256SUMS"),&valid);
    assert_eq!(bundle::check(&t.0,&[]).unwrap_err().to_string(),"compressed archive hash mismatch");
    for invalid in [String::new(),valid.trim_end().into(),valid.repeat(2),valid.replace("  "," ")] { assert!(bundle::checksum(&invalid).is_err()); }
}
#[test] fn archive_rejects_duplicate_unsafe_and_unexpected_members() {
    let f = Fixture::new();
    for (name,expected) in [("LICENSE","duplicate archive member"),("LICENSE/","regular archive member ends in '/': LICENSE/"),
        ("/absolute","unsafe archive member: /absolute"),("../escape","unsafe archive member: ../escape"),("stray","unexpected bundle member")] {
        error(f.tar(None,&[(name,b"stray")],0o755,b'0'),expected);
    }
}
#[test] fn archive_rejects_a_trailing_slash_on_the_only_executable_payload() {
    for kind in [0,b'0'] {
        let mut f = Fixture::new(); let data = f.files.remove("vade").unwrap(); f.files.insert("vade/".into(),data);
        let mut raw = f.tar(None,&[],0o755,b'0');
        // Change the actual vade/ header type, and retain a valid checksum.
        let mut offset = 0;
        while raw[offset] != 0 {
            let h = &mut raw[offset..offset+512];
            let size = usize::from_str_radix(std::str::from_utf8(&h[124..135]).unwrap(),8).unwrap();
            if &h[..6] == b"vade/\0" { h[156] = kind; h[148..156].fill(b' '); let sum: u64 = h.iter().map(|&b|u64::from(b)).sum();
                h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes()); break; }
            offset += 512+size.div_ceil(512)*512;
        }
        error(raw,"regular archive member ends in '/': vade/");
    }
}
#[test] fn archive_requires_payload_and_licenses() {
    for (name,expected) in [("vade","missing bundle member"),("engine/licenses/LICENSE","missing engine licenses")] {
        let mut f = Fixture::new(); f.files.remove(name); error(f.valid(),expected);
    }
}
#[test] fn archive_rejects_links_and_required_directories() {
    for (kind,expected) in [(b'2',"unsafe archive member: vade"),(b'1',"unsafe archive member: vade"),(b'5',"bundle member is not a regular file")] {
        error(Fixture::new().tar(None,&[],0o755,kind),expected);
    }
}
#[test] fn archive_errors_precede_manifest_errors() {
    let mut f = Fixture::new(); error(f.tar(Some(b"malformed".to_vec()),&[("LICENSE",b"duplicate")],0o755,b'0'),"duplicate archive member");
    f.files.remove("vade"); error(f.tar(Some(b"malformed".to_vec()),&[],0o755,b'0'),"missing bundle member");
}
#[test] fn manifest_fields_and_source_identity_are_checked() {
    for (key,value,expected) in [("format","unknown","wrong bundle format or target"),("target","unknown","wrong bundle format or target"),
        ("engine_sha","invalid","invalid source revision"),("engine_sha",&"f".repeat(40),"invalid source revision"),
        ("cli_sha",&"f".repeat(40),"source revisions differ"),("harness_sha",&"f".repeat(40),"source revisions differ"),("extra","field","invalid manifest fields")] {
        let mut f = Fixture::new(); f.manifest.insert(key.into(),value.into()); error(f.valid(),expected);
    }
    let mut f = Fixture::new(); let mut duplicate = f.manifest(); duplicate.extend_from_slice(format!("engine_sha={REV}\n").as_bytes());
    error(f.tar(Some(duplicate),&[],0o755,b'0'),"invalid manifest fields"); f.manifest.remove("format"); error(f.valid(),"invalid manifest fields");
}
#[test] fn payload_hashes_modes_and_producer_paths_are_checked() {
    let mut f = Fixture::new(); f.files.get_mut("flash.ptx").unwrap().extend_from_slice(b"corruption"); error(f.valid(),"manifest hash mismatch: flash.ptx");
    let f = Fixture::new(); for mode in [0o644,0o700,0o750] { error(f.tar(None,&[],mode,b'0'),"missing executable mode: vade"); }
    assert!(check(&f.valid(),&[]).is_ok());
    for path in ["portable","aarch64-unknown-linux-gnu"] { assert_eq!(check(&f.valid(),&[path.into()]).unwrap_err(),"producer path retained: flash-serve"); }
}
#[test] fn tar_rejects_bad_headers_and_truncated_payloads() {
    let f = Fixture::new(); let mut bad = f.valid(); bad[0] ^= 1; assert!(check(&bad,&[]).is_err());
    let raw = f.valid(); assert!(check(&raw[..600],&[]).is_err());
    let mut f = Fixture::new(); let long = format!("{}producer-path{}","a".repeat(65530),"b".repeat(100));
    f.files.insert("flash-serve".into(),long.into_bytes()); f.manifest.insert("engine_sha256".into(),sha256::digest(&f.files["flash-serve"]));
    assert_eq!(check(&f.valid(),&["producer-path".into()]).unwrap_err(),"producer path retained: flash-serve");
}
