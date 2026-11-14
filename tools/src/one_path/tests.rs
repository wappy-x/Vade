use super::*;
fn fixture(source: &str) -> Vec<Finding> { scan(source,"fixture","decode","engine/fixture.rs") }
#[test]
fn env_reads_fail_fixture() {
    let hits=fixture(include_str!("../../tests/fixtures/one-path/env-fail.rs.txt"));
    assert_eq!(hits.len(),11,"{hits:#?}");
    assert!(hits.iter().all(|f|f.key.rule=="env-read"));
    assert_eq!(hits[0].line,5);
    assert_eq!(hits[0].key.item,"decode::run");
    assert_eq!(check(&hits,&BTreeMap::new()).len(),11);
}
#[test]
fn env_reads_pass_fixture() {
    assert!(fixture(include_str!("../../tests/fixtures/one-path/env-pass.rs.txt")).is_empty());
}
#[test]
fn topology_branches_fail_fixture() {
    let hits=fixture(include_str!("../../tests/fixtures/one-path/topology-fail.rs.txt"));
    assert_eq!(hits.len(),13,"{hits:#?}");
    assert!(hits.iter().all(|f|f.key.rule=="topology-branch"));
    assert_eq!(hits[0].line,2);
    assert_eq!(check(&hits,&BTreeMap::new()).len(),13);
}
#[test]
fn topology_data_pass_fixture() {
    assert!(fixture(include_str!("../../tests/fixtures/one-path/topology-pass.rs.txt")).is_empty());
}
#[test]
fn tensor_rank_names_do_not_select_topology_paths() {
    for name in ["lora_rank","q_lora_rank","kv_lora_rank","tensor_rank","TENSOR_RANK","MAX_TENSOR_RANK","config.q_lora_rank","shape.tensor_rank()","r#kv_lora_rank"] {
        for statement in [
            format!("if {name} > 0 {{ choose(); }}"),
            format!("while {name} > 0 {{ choose(); }}"),
            format!("match {name} {{ 0 => first(), _ => second() }}"),
            format!("match value {{ _ if {name} > 0 => first(), _ => second() }}"),
            format!("let choice = matches!({name}, 1 | 2);"),
            format!("let choice = {name} > 0;"),
            format!("let choice = 0 < {name};"),
        ] {
            let hits=fixture(&format!("fn run() {{ {statement} }}"));
            assert!(hits.is_empty(),"{statement}: {hits:#?}");
        }
    }
}
#[test]
fn tensor_rank_exception_does_not_hide_other_topology_names() {
    for name in ["rank","world","tp","pp","sparks","state.rank()","group.world_size()","tensor_parallel_rank","lora_process_rank","q_lora_ranks","notlora_rank","nontensor_rank","q_lora_rank_world","tensor_rank_topology"] {
        for statement in [
            format!("if {name} > 0 {{ choose(); }}"),
            format!("if config.q_lora_rank > 0 && {name} > 0 {{ choose(); }}"),
            format!("if {name} > 0 && shape.len() <= MAX_TENSOR_RANK {{ choose(); }}"),
            format!("while kv_lora_rank > 0 && {name} > 0 {{ choose(); }}"),
            format!("match (tensor_rank, {name}) {{ _ => choose() }}"),
            format!("match value {{ _ if q_lora_rank > 0 && {name} > 0 => first(), _ => second() }}"),
            format!("let choice = matches!((TENSOR_RANK, {name}), (_, 1 | 2));"),
            format!("let choice = {name} > MAX_TENSOR_RANK;"),
            format!("let choice = q_lora_rank < {name};"),
        ] {
            let hits=fixture(&format!("fn run() {{ {statement} }}"));
            assert_eq!(hits.len(),1,"{statement}: {hits:#?}");
            assert_eq!(hits[0].key.rule,"topology-branch");
            assert_eq!(hits[0].key.item,"decode::run");
        }
    }
}
#[test]
fn literals_comments_and_lifetimes_keep_lines_and_items() {
    let source=r####"
/* std::env::var("unused"); /* if world == 2 {} */ */
mod nested {
    const TEXT: &str = r###"std::env::vars(); if world > 1 {}"###;
    impl<'a> Session<'a> {
        fn run(&'a self) {
            let _ = '\''; let _ = '界'; let _ = b'"'; let _ = c"if world == 2 {}";
            let _ = br##"std::env::vars_os()"##;
            let _ = ::std::env::var_os("MODE");
        }
    }
}
"####;
    let hits=fixture(source); assert_eq!(hits.len(),1,"{hits:#?}");
    assert_eq!(hits[0].line,9); assert_eq!(hits[0].key.item,"decode::nested::Session::run");
}
#[test]
fn allowances_follow_items_and_reject_new_sites_and_stale_rows() {
    let source="impl Worker { fn run() { std::env::vars(); } }";
    let hits=fixture(source); let allowed=allowlist("env-read fixture decode::Worker::run 1").unwrap();
    assert!(check(&hits,&allowed).is_empty());
    assert!(check(&fixture(&format!("\n\n{source}")),&allowed).is_empty());
    assert_eq!(check(&fixture(&source.replace("vars();","vars(); std::env::vars_os();")),&allowed).len(),2);
    assert_eq!(check(&fixture(&source.replace("run()","other()")),&allowed).len(),2);
    assert_eq!(check(&scan(source,"other","decode","engine/other.rs"),&allowed).len(),2);
    assert_eq!(check(&[],&allowed).len(),1);
    for bad in ["env-read fixture run 0","other fixture run 1","env-read fixture run 1\nenv-read fixture run 1","env-read fixture run","env-read fixture run nope"] {
        assert!(allowlist(bad).is_err(),"{bad}");
    }
}
#[test]
fn modules_impls_traits_constants_and_inner_items_have_separate_keys() {
    let source="mod nested { impl Read for First { fn run() { std::env::vars(); fn inner() { std::env::vars(); } } } impl Second { fn run() { std::env::vars(); } } static VALUE: usize = { std::env::vars().count() }; }";
    let hits=fixture(source);
    assert_eq!(hits.iter().map(|f|f.key.item.as_str()).collect::<Vec<_>>(),[
        "decode::nested::First@Read::run","decode::nested::First@Read::run::inner","decode::nested::Second::run","decode::nested::VALUE",
    ]);
    assert_eq!(module(Path::new("src/bin/runner/main.rs")),"bin::runner::main");
    assert_eq!(module(Path::new("src/plan/mod.rs")),"plan");
    assert_eq!(package("[package]\nname = \"test-engine\"\n[dependencies]"),Some("test-engine".into()));
}
#[test]
fn standard_library_import_aliases_and_globs_are_checked() {
    for source in [
        "use std::env::*; fn f() { vars(); }",
        "use std::*; fn f() { env::var_os(\"MODE\"); }",
        "use std as rust; use rust::env as process; fn f() { process::var(\"MODE\"); }",
        "extern crate std as rust; use rust::env::{self as process}; fn f() { process::vars_os(); }",
        "fn f() { let read = std::env::var::<&str>; read(\"MODE\"); }",
        "fn f() { ::std::env::r#vars(); }",
    ] {let hits=fixture(source); assert_eq!(hits.len(),1,"{source}: {hits:#?}");}
    let source="use std::env as read; fn outer() { read::vars(); } mod nested { use std::env::var as read; fn inner() { read(\"MODE\"); } } mod other { use std::fs as read; }";
    assert_eq!(fixture(source).len(),2);
}
#[test]
fn inventory_checks_new_source_and_uses_the_nearest_package() {
    struct Tree(std::path::PathBuf);
    impl Drop for Tree {fn drop(&mut self) {let _=fs::remove_dir_all(&self.0);}}
    let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let tree=Tree(crate::root().join(format!(".local/one-path-tests/{}-{nonce}",std::process::id())));
    fs::create_dir_all(&tree.0).unwrap();
    crate::output(Command::new("git").args(["init","--quiet"]).arg(&tree.0)).unwrap();
    let write=|file: &str, source: &str| {let path=tree.0.join(file); fs::create_dir_all(path.parent().unwrap()).unwrap(); fs::write(path,source).unwrap();};
    write(".gitignore","target/\n.local/\n");
    write("engine/outer/Cargo.toml","[package]\nname = \"outer\"\n");
    write("engine/outer/nested/Cargo.toml","[package]\nname = \"nested\"\n");
    write("engine/outer/src/lib.rs","fn run() { std::env::vars(); }");
    write("engine/outer/nested/src/lib.rs","fn run() { if world > 1 { alternate(); } }");
    write("engine/outer/target/generated.rs","fn run() { std::env::vars(); }");
    write("engine/outer/.local/private.rs","fn run() { std::env::vars(); }");
    write("engine/tools/probe.rs","fn run() { std::env::var_os(\"MODE\"); }");
    write("other/source.rs","fn run() { std::env::vars(); }");
    crate::output(Command::new("git").current_dir(&tree.0).args(["add","engine/outer/src/lib.rs"])).unwrap();
    let hits=inventory(&tree.0).unwrap(); assert_eq!(hits.len(),3,"{hits:#?}");
    assert_eq!(hits.iter().map(|f|f.key.krate.as_str()).collect::<BTreeSet<_>>(),BTreeSet::from(["nested","outer","probe"]));
    assert!(hits.iter().all(|f|f.key.item=="run"));
    write("tools/one-path-allowlist.txt",""); assert!(run(&tree.0).is_err());
    let mut allowed=String::new();
    for (key,count) in counts(&hits) {allowed.push_str(&format!("{} {} {} {count}\n",key.rule,key.krate,key.item));}
    write("tools/one-path-allowlist.txt",&allowed); assert!(run(&tree.0).is_ok());
    write("engine/outer/src/lib.rs","fn run() { std::env::vars(); std::env::vars_os(); }");
    assert!(run(&tree.0).is_err());
    fs::remove_file(tree.0.join("engine/outer/src/lib.rs")).unwrap();
    assert_eq!(inventory(&tree.0).unwrap().len(),2);
}
#[test]
fn engine_source_uses_one_path_allowlist() {
    run(&crate::root()).unwrap();
}
