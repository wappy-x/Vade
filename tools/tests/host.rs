#![cfg(unix)]
mod common;
use common::*;
use std::{fs,process::Command};
use vade_tools::{json,docs,cuda,fetch,sha256};
#[test] fn documentation_links_cover_anchors_assets_titles_and_encoded_paths() {
    let t = Temp::new("doc links");
    write(&t.0.join("a.md"),"# Heading!\n# Heading!\n## <em>Café</em> `x_y`\n<a id='explicit'></a>\n[ok](#heading-1) [ok](#café-x_y) [ok](#explicit)\n[space](<sub/file%20name.md#child>)\n[title](sub/file%20name.md#child \"title\")\n<img src='asset.svg'>\n[reference]: sub/file%20name.md#child\n[net](https://example.invalid/a)\n[net](mailto:reader@example.invalid)\n[net](//example.invalid/a)\n[bad](gone.md)\n[bad](#absent)\n");
    write(&t.0.join("sub/file name.md"),"# Child\n"); write(&t.0.join("asset.svg"),"<svg/>");
    let issues = docs::check(&t.0,&["a.md".into()]).unwrap();
    assert_eq!(issues,["a.md:13: missing path: gone.md","a.md:14: missing anchor: #absent"]);
}
#[test] fn json_round_trips_escapes_surrogates_and_numbers_and_rejects_bad_input() {
    let text = "{\"a\":[null,true,false,-1.2e+3,\"\\uD83D\\uDE00\\n\\t\\\\\\\"\"],\"b\":\"é\"}";
    let value = json::parse(text).unwrap(); assert_eq!(json::parse(&value.encode()).unwrap(),value);
    assert_eq!(value.get("a").get("4").text().unwrap(),"😀\n\t\\\"");
    for invalid in ["01","-","1.","1e","[1,]","{\"a\":1,\"a\":2}","\"\\uD800\"","\"\\uDC00\"","\"\n\"","true false"] { assert!(json::parse(invalid).is_err(),"{invalid}"); }
}
#[test] fn cuda_symbols_are_unique_and_all_stubs_abort() {
    let names = cuda::symbols("fn cuInit(a: u32); fn cuInit(a:u32); fn cuMemAlloc_v2 (); fn custom();");
    assert_eq!(names.iter().map(String::as_str).collect::<Vec<_>>(),["cuInit","cuMemAlloc_v2"]);
    let source = cuda::source(&names); assert_eq!(source.matches("panic!(").count(),2); assert_eq!(source.matches("pub extern").count(),2);
}
const REV: &str = "1234567890abcdef1234567890abcdef12345678";
fn metadata() -> String {
    format!("{{\"sha\":\"{REV}\",\"siblings\":[{{\"rfilename\":\"config.json\",\"size\":6,\"blobId\":\"ce013625030ba8dba906f756967f9e9ca394464a\"}},{{\"rfilename\":\"sub/weights\",\"size\":6,\"lfs\":{{\"sha256\":\"{}\"}}}}]}}",sha256::digest(b"hello\n"))
}
#[test] fn model_metadata_and_hashes_reject_changes_and_unsafe_paths() {
    let t = Temp::new("model hashes"); write(&t.0.join("file"),b"hello\n");
    let entries = fetch::entries(&metadata(),REV).unwrap();
    for entry in &entries { assert_eq!(fetch::verify(&t.0.join("file"),entry).unwrap(),sha256::digest(b"hello\n")); }
    write(&t.0.join("file"),b"wrong\n"); for entry in &entries { assert!(fetch::verify(&t.0.join("file"),entry).is_err()); }
    for path in ["../escape","/absolute","a/../b",".cache/huggingface/token","a\\b"] {
        assert!(fetch::entries(&metadata().replace("\"config.json\"",&json::quote(path)),REV).is_err(),"{path}");
    }
    assert!(fetch::entries(&metadata(),&"f".repeat(40)).is_err());
}
#[test] fn model_fetch_reuses_files_and_keeps_the_manifest_on_failure() {
    let t = Temp::new("model fetch"); let model = t.0.join("model");
    write(&t.0.join("metadata"),metadata()); write(&t.0.join("payload"),b"hello\n");
    executable(&t.0.join("bin/curl"),r#"#!/bin/sh
for arg do case $arg in *'/api/models/'*) cat "$TEST_ROOT/metadata"; exit ;; esac; done
while [ "$#" -gt 0 ]; do if [ "$1" = --output ]; then shift; echo download >> "$TEST_ROOT/downloads"; cp "$TEST_ROOT/payload" "$1"; exit; fi; shift; done
exit 1
"#);
    let invoke = || Command::new(env!("CARGO_BIN_EXE_vade-tools")).args(["fetch-model","org/model",REV]).arg(&model)
        .env("TEST_ROOT",&t.0).env("PATH",format!("{}:{}",t.0.join("bin").display(),std::env::var("PATH").unwrap())).output().unwrap();
    for _ in 0..2 { let out = invoke(); assert!(out.status.success(),"{}",diagnostic(&out)); assert!(String::from_utf8_lossy(&out.stdout).contains("verified 2 files"));
        assert_eq!(fs::read_to_string(t.0.join("downloads")).unwrap().lines().count(),2); }
    let manifest = fs::read(t.0.join("model.sha256")).unwrap();
    fs::File::options().read(true).write(true).open(t.0.join("model.fetch/lock")).unwrap().try_lock().unwrap();
    write(&model.join("extra"),"extra"); assert!(!invoke().status.success()); assert_eq!(fs::read(t.0.join("model.sha256")).unwrap(),manifest);
    fs::remove_file(model.join("extra")).unwrap(); write(&model.join("config.json"),"wrong\n"); assert!(invoke().status.success());
    write(&t.0.join("payload"),"wrong\n"); write(&model.join("config.json"),"wrong\n"); assert!(!invoke().status.success());
    fs::File::options().read(true).write(true).open(t.0.join("model.fetch/lock")).unwrap().try_lock().unwrap();
    assert_eq!(fs::read(t.0.join("model.sha256")).unwrap(),manifest);
}
