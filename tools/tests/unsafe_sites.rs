//! Source fixtures for the unsafe rule. These tests do not run GPU code.
#![forbid(unsafe_code)]
use std::{collections::BTreeMap,fs,process::Command};
use vade_tools::{Result,unsafe_sites::{self,Allowlist,Site,allowlist,inventory,violations}};
#[cfg(unix)]
mod common;
fn sites(path: &str, source: &str, gpu: bool) -> Result<Vec<Site>> {
    unsafe_sites::sites(if gpu { "engine/crates/gpu" } else { "tools" },path,source,gpu)
}
const RULING: &str = "ruling 2026-11-02 11:54";
fn approved(sites: &[Site]) -> Allowlist { sites.iter().map(|s| (s.key.clone(),format!("frozen {RULING}"))).collect() }
fn render(sites: &[Site]) -> String { unsafe_sites::render(sites,&approved(sites)) }
#[test]
fn repository_unsafe_sites() {
    unsafe_sites::check(&vade_tools::root()).unwrap();
}
#[test]
fn allowlist_matches_tree() {
    let root = vade_tools::root(); let sites = inventory(&root).unwrap();
    let allowed = allowlist(&fs::read_to_string(root.join("tools/unsafe-allowlist.txt")).unwrap()).unwrap();
    let issues = unsafe_sites::drift(&sites,&allowed);
    assert!(issues.is_empty(),"{}\nThe allowlist is frozen. A refresh needs an orchestrator ruling.",issues.join("\n"));
    println!("unsafe allowlist matches tree: {} sites; {} entries",sites.len(),allowed.len());
}

const GPU: &str = "engine/crates/gpu/src/fixture.rs";
const HOST: &str = "tools/src/fixture.rs";
const GOOD: &str = "fn launch() {\n    // SAFETY: The caller owns the buffer until this call returns.\n    unsafe { call(); }\n}\n";
fn check_fixture(path: &str, source: &str, gpu: bool) -> Vec<String> { violations(&sites(path,source,gpu).unwrap(),&BTreeMap::new()) }
fn fails_with(issues: &[String], text: &str) { assert!(issues.iter().any(|s| s.contains(text)),"missing {text}: {issues:?}"); }
#[test]
fn new_block_is_only_allowed_in_gpu_crate() {
    assert!(check_fixture(GPU,GOOD,true).is_empty());
    fails_with(&check_fixture(HOST,GOOD,false),"outside the GPU crate");
}
#[test]
fn new_declarations_fail_and_legacy_declarations_pass() {
    for source in ["unsafe fn old() {}","unsafe impl Send for Old {}","unsafe extern \"C\" { fn old(); }","unsafe trait Old {}","#[unsafe(no_mangle)] fn old() {}"] {
        let old = sites(HOST,source,false).unwrap(); assert_eq!(old.len(),1);
        fails_with(&violations(&old,&BTreeMap::new()),"outside the GPU crate");
        assert!(violations(&old,&approved(&old)).is_empty());
        fails_with(&check_fixture(GPU,source,true),if source.starts_with("unsafe fn") { "needs an allowlist ruling line" } else { "only a new unsafe block" });
    }
    let old = sites(HOST,"struct Old { call: unsafe extern \"C\" fn(*mut u8) -> i32 }",false).unwrap();
    assert_eq!(old.len(),1); assert!(violations(&old,&approved(&old)).is_empty());
}
#[test]
fn new_unsafe_function_needs_a_ruling_even_when_the_allowlist_matches() {
    let found = sites(HOST,"unsafe fn added() {}",false).unwrap();
    let listed = allowlist(&unsafe_sites::render(&found,&BTreeMap::new())).unwrap();
    assert!(unsafe_sites::drift(&found,&listed).is_empty());
    fails_with(&violations(&found,&listed),"needs an allowlist ruling line");
    fails_with(&violations(&found,&BTreeMap::new()),"needs an allowlist ruling line");
    let ruled = allowlist(&format!("{}\t{RULING}\n",found[0].key)).unwrap();
    assert!(violations(&found,&ruled).is_empty());
    assert!(unsafe_sites::drift(&found,&ruled).is_empty());
    // A ruling for a different item does not approve this function.
    let other = sites(HOST,"unsafe fn other() {}",false).unwrap();
    fails_with(&violations(&other,&ruled),"needs an allowlist ruling line");
}

const KERNEL: &str = "// SAFETY: The caller keeps the device buffer live for the launch.\n#[no_mangle]\npub unsafe extern \"ptx-kernel\" fn launch() {\n    // SAFETY: The launch owns the buffer and this thread stays within its bounds.\n    unsafe { call(); }\n}\n";
fn listed_kernel(crate_path: &str, source: &str) -> (Vec<Site>,Allowlist) {
    let path = format!("{crate_path}/src/fixture.rs");
    let gpu = matches!(crate_path,"engine/crates/gpu"|"engine/crates/gpu/kernels");
    let found = unsafe_sites::sites(crate_path,&path,source,gpu).unwrap();
    let allowed = allowlist(&unsafe_sites::render(&found,&BTreeMap::new())).unwrap();
    assert!(unsafe_sites::drift(&found,&allowed).is_empty()); (found,allowed)
}
#[test]
fn new_ptx_kernel_with_safety_notes_passes_in_each_gpu_crate() {
    for crate_path in ["engine/crates/gpu","engine/crates/gpu/kernels"] {
        for source in [KERNEL.into(),KERNEL.replace("#[no_mangle]","#[unsafe(no_mangle)]"),
            KERNEL.replace("pub unsafe","pub(crate) unsafe"),format!("mod entries {{\n{KERNEL}}}")] {
            let (found,allowed) = listed_kernel(crate_path,&source);
            assert!(violations(&found,&allowed).is_empty(),"{source}");
        }
    }
    for crate_path in ["tools","engine/crates/tp/kernels","engine/crates/gpu/kernels/nested"] {
        let (found,allowed) = listed_kernel(crate_path,KERNEL);
        fails_with(&violations(&found,&allowed),"needs an allowlist ruling line");
    }
    for source in [KERNEL.replace("ptx-kernel","C"),"type Callback = unsafe extern \"ptx-kernel\" fn();".into()] {
        let (found,allowed) = listed_kernel("engine/crates/gpu",&source);
        fails_with(&violations(&found,&allowed),"needs an allowlist ruling line");
    }
}
#[test]
fn new_ptx_kernel_without_an_entry_or_block_note_fails() {
    for source in [KERNEL.replacen("SAFETY:","Reason:",1),KERNEL.replacen("#[no_mangle]","\n#[no_mangle]",1),
        KERNEL.replace("The caller keeps the device buffer live for the launch.","")] {
        let (found,allowed) = listed_kernel("engine/crates/gpu/kernels",&source);
        fails_with(&violations(&found,&allowed),"ptx-kernel entry needs a SAFETY:");
        let ruled: Allowlist = found.iter().map(|s| (s.key.clone(),RULING.into())).collect();
        fails_with(&violations(&found,&ruled),"ptx-kernel entry needs a SAFETY:");
    }
    let source = KERNEL.replace("    // SAFETY:","    // Reason:");
    let (found,allowed) = listed_kernel("engine/crates/gpu/kernels",&source);
    fails_with(&violations(&found,&allowed),"unsafe block needs a SAFETY:");
}
#[test]
fn two_new_blocks_in_one_ptx_kernel_fail_even_when_listed() {
    let source = KERNEL.replace("    unsafe { call(); }","    unsafe { call(); }\n    // SAFETY: The same buffer stays live.\n    unsafe { other(); }");
    for crate_path in ["engine/crates/gpu","engine/crates/gpu/kernels"] {
        let (found,allowed) = listed_kernel(crate_path,&source);
        fails_with(&violations(&found,&allowed),"more than one new unsafe block");
        let ruled: Allowlist = found.iter().map(|s| (s.key.clone(),RULING.into())).collect();
        fails_with(&violations(&found,&ruled),"more than one new unsafe block");
        let separate = format!("{KERNEL}{}",KERNEL.replace("launch","other"));
        let (found,allowed) = listed_kernel(crate_path,&separate);
        assert!(violations(&found,&allowed).is_empty());
    }
}
#[test]
fn legacy_function_and_impl_names_do_not_pin_signatures() {
    for (before,after) in [
        ("unsafe fn old() {}","unsafe fn old(x: u64) -> u64 { x }"),
        ("pub unsafe extern \"ptx-kernel\" fn old(x: u64,elements: u32) {}",
         "pub unsafe extern \"ptx-kernel\" fn old(x: u64,elements: u32,planes: u64) {}"),
        ("unsafe impl<T> Send for Old<T> {}","unsafe impl<U: Send> Send for Old<U> where U: Copy {}"),
        ("impl<T> Old<T> { unsafe fn call() {} }","impl<U: Copy> Old<U> where U: Send { unsafe fn call(x: u64) {} }")
    ] {
        let old = sites(HOST,before,false).unwrap(); let allowed = allowlist(&render(&old)).unwrap();
        let edited = sites(HOST,after,false).unwrap();
        assert!(violations(&edited,&allowed).is_empty(),"{before} -> {after}");
        assert!(unsafe_sites::drift(&edited,&allowed).is_empty());
        let renamed = sites(HOST,&after.replace("old(","new(").replace("Old","New"),false).unwrap();
        fails_with(&violations(&renamed,&allowed),"outside the GPU crate");
        let moved = unsafe_sites::sites("harness",HOST,after,false).unwrap();
        fails_with(&violations(&moved,&allowed),"outside the GPU crate");
    }
}
#[test]
fn impl_keys_ignore_function_return_types_in_generic_bounds() {
    for bound in ["Fn() -> u32","Fn() -> Vec<u32>","Fn() -> [u32; { 2 } ]","Fn(fn() -> u32) -> u64"] {
        let source = format!("impl<F: {bound}> Tr for X<F> {{ fn call() {{ unsafe {{ run(); }} }} }}");
        let found = sites(HOST,&source,false).unwrap(); assert_eq!(found.len(),1);
        assert_eq!(found[0].item,"impl:Tr for X::fn:call");
        let edited = sites(HOST,&source.replace("u32","u64"),false).unwrap();
        let allowed = allowlist(&render(&found)).unwrap();
        assert!(unsafe_sites::drift(&edited,&allowed).is_empty());
        assert!(violations(&edited,&allowed).is_empty());
    }
}
#[test]
fn moved_legacy_blocks_get_one_refresh_message() {
    let source = "fn old() { unsafe { call(); } }";
    let old = sites(HOST,source,false).unwrap(); let allowed = allowlist(&render(&old)).unwrap();
    for (path,source) in [(HOST,source.replace("fn old","fn new")),("tools/src/moved.rs",source.into())] {
        let moved = sites(path,&source,false).unwrap();
        let issues = violations(&moved,&allowed);
        assert_eq!(issues,["possible moved legacy block, refresh-unsafe-allowlist needs an orchestrator ruling"]);
        let drift = unsafe_sites::drift(&moved,&allowed); assert_eq!(drift.len(),2);
        fails_with(&drift,"missing from allowlist"); fails_with(&drift,"stale unsafe allowlist entry");
        assert!(violations(&moved,&allowlist(&render(&moved)).unwrap()).is_empty());
    }
    let old = sites(GPU,"fn old() { unsafe { first(); } unsafe { second(); } }",true).unwrap();
    let gpu_allowed = allowlist(&render(&old)).unwrap();
    let split = sites(GPU,"fn old() { unsafe { first(); } } fn new() { unsafe { second(); } }",true).unwrap();
    assert_eq!(violations(&split,&gpu_allowed),["possible moved legacy block, refresh-unsafe-allowlist needs an orchestrator ruling"]);
    let mixed = sites(HOST,"fn new() { unsafe { call(); } } unsafe fn added() {}",false).unwrap();
    let issues = violations(&mixed,&allowed); assert_eq!(issues.len(),2);
    fails_with(&issues,"moved legacy block, refresh-unsafe-allowlist needs an orchestrator ruling");
    fails_with(&issues,"new unsafe fn outside the GPU crate");
    let grown = sites(HOST,"fn new() { unsafe { call(); } unsafe { other(); } }",false).unwrap();
    fails_with(&violations(&grown,&allowed),"new unsafe block outside the GPU crate");
}
#[test]
fn one_new_block_per_wrapper() {
    assert!(check_fixture(GPU,GOOD,true).is_empty());
    let two = format!("{}{}",GOOD.trim_end().trim_end_matches('}'),"\n// SAFETY: The call uses the same live buffer.\nunsafe { other(); }\n}");
    fails_with(&check_fixture(GPU,&two,true),"more than one new unsafe block");
    let separate = format!("{GOOD}{}",GOOD.replace("launch","other"));
    assert!(check_fixture(GPU,&separate,true).is_empty());
    let nested = "fn launch() {\n// SAFETY: Both calls use a live buffer.\nunsafe {\n// SAFETY: The buffer is still live.\nunsafe { other(); }\n}\n}";
    fails_with(&check_fixture(GPU,nested,true),"more than one new unsafe block");
}
#[test]
fn each_new_block_needs_an_adjacent_nonempty_contract() {
    for source in [GOOD.replace("SAFETY:","Reason:"),GOOD.replace("    unsafe","\n    unsafe"),GOOD.replace("The caller owns the buffer until this call returns.",""),
        "fn launch() { unsafe { call(); } }".into(),"fn launch() {\n// SAFETY: The buffer is live.\nlet x = 1;\nunsafe { call(); }\n}".into(),
        "fn launch() { /* SAFETY: The buffer is live. */ unsafe { call(); } }".into()] {
        fails_with(&check_fixture(GPU,&source,true),"directly above it");
    }
    for source in [GOOD.to_string(),GOOD.replace("    unsafe","    let result = unsafe"),
        GOOD.replace("    unsafe","    // The buffer does not alias the output.\n    unsafe"),
        GOOD.replace("// SAFETY: The caller owns the buffer until this call returns.","/* SAFETY: The caller owns the buffer.\n       The buffer stays live. */")] {
        assert!(check_fixture(GPU,&source,true).is_empty(),"{source}");
    }
}
#[test]
fn block_order_shifts_do_not_skip_contract_checks() {
    for crate_path in ["engine/crates/gpu","engine/crates/gpu/kernels"] {
        let path = format!("{crate_path}/src/fixture.rs");
        let old = unsafe_sites::sites(crate_path,&path,GOOD,true).unwrap(); let frozen = approved(&old);
        for (source,line) in [
            (GOOD.replace("fn launch() {\n","fn launch() {\n    unsafe { new_no_note(); }\n"),2),
            (GOOD.replace("    unsafe { call(); }","    unsafe { call(); }\n    unsafe { new_no_note(); }"),4)
        ] {
            let found = unsafe_sites::sites(crate_path,&path,&source,true).unwrap();
            let listed = allowlist(&unsafe_sites::render(&found,&frozen)).unwrap();
            assert!(unsafe_sites::drift(&found,&listed).is_empty());
            let message = format!("{path}:{line}: fn:launch: unsafe block needs a SAFETY:");
            fails_with(&violations(&found,&frozen),&message);
            fails_with(&violations(&found,&listed),&message);
            let noted = source.replace("    unsafe { new_no_note(); }","    // SAFETY: The caller owns the new buffer.\n    unsafe { new_no_note(); }");
            let found = unsafe_sites::sites(crate_path,&path,&noted,true).unwrap();
            assert!(violations(&found,&listed).is_empty());
        }
    }
}
#[test]
fn legacy_entries_do_not_allow_more_blocks_in_the_same_item() {
    let source = GOOD;
    let old = sites(GPU,source,true).unwrap(); let allowed = approved(&old);
    assert!(violations(&old,&allowed).is_empty());
    let one = source.replace("    unsafe { call(); }","    unsafe { call(); }\n// SAFETY: The buffer is live.\nunsafe { new(); }");
    assert!(violations(&sites(GPU,&one,true).unwrap(),&allowed).is_empty());
    let grown = one.replace("unsafe { new(); }","unsafe { new(); }\n// SAFETY: The buffer is live.\nunsafe { other(); }");
    fails_with(&violations(&sites(GPU,&grown,true).unwrap(),&allowed),"more than one new unsafe block");
    let host = sites(HOST,source,false).unwrap(); let allowed = approved(&host);
    let grown = "fn launch() { unsafe { old(); } unsafe { old(); } }";
    fails_with(&violations(&sites(HOST,grown,false).unwrap(),&allowed),"outside the GPU crate");
    let changed = sites(HOST,&source.replace("call()","new()"),false).unwrap();
    assert!(violations(&changed,&allowed).is_empty());
    fails_with(&violations(&[],&allowed),"stale unsafe allowlist entry");
}
#[test]
fn allowlist_is_stable_across_source_lines_and_rejects_invalid_entries() {
    let old = sites(HOST,GOOD,false).unwrap(); let allowed = approved(&old);
    let moved = sites(HOST,&format!("\n\n{GOOD}"),false).unwrap(); assert!(violations(&moved,&allowed).is_empty());
    let removed = sites(HOST,&GOOD.replace("SAFETY:","Reason:"),false).unwrap();
    assert!(violations(&removed,&allowed).is_empty());
    let text = format!("# Legacy source sites\n{}\tfrozen {RULING}\n",old[0].key); assert_eq!(allowlist(&text).unwrap(),allowed);
    assert!(allowlist(&format!("{text}{}\n",old[0].key)).is_err());
    for text in ["fixture.rs:12", "tools\tfixture.rs\tfn:launch\tblock:0", "../tools\tfixture.rs\tfn:launch\tblock:1",
        "tools\t../fixture.rs\tfn:launch\tblock:1", "tools\tfixture.rs\tfn:launch\tunknown:1"] { assert!(allowlist(text).is_err()); }
    assert!(allowlist(&format!("{text}{}\truling 2026-11-02 12:00\n",old[0].key)).is_err());
    for bad in ["approved","ruling","ruling 2026-13-02 11:54","ruling 2026-11-00 11:54","ruling 2026-11-02 24:00","ruling 2026-11-02 11:60"] {
        assert!(unsafe_sites::ruling(bad).is_err());
        assert!(allowlist(&format!("{}\t{bad}\n",old[0].key)).is_err());
    }
    unsafe_sites::ruling(RULING).unwrap();
}
#[test]
fn frozen_rows_require_the_fixed_freeze_ruling() {
    let found = sites(HOST,"unsafe fn added() {}",false).unwrap();
    let key = &found[0].key;
    let frozen = allowlist(&format!("{key}\tfrozen {RULING}\n")).unwrap();
    assert!(violations(&found,&frozen).is_empty());
    for ruling in ["ruling 2026-12-31 23:59","ruling 2026-11-02 11:53","ruling 2026-11-02 11:55","ruling 2026-11-01 11:54"] {
        unsafe_sites::ruling(ruling).unwrap();
        let error = allowlist(&format!("{key}\tfrozen {ruling}\n")).unwrap_err().to_string();
        assert!(error.contains("frozen ruling 2026-11-02 11:54"),"{error}");
        let ruled = allowlist(&format!("{key}\t{ruling}\n")).unwrap();
        assert!(unsafe_sites::drift(&found,&ruled).is_empty());
        assert!(violations(&found,&ruled).is_empty());
    }
}
#[test]
fn allowlist_drift_finds_missing_and_stale_entries() {
    let current = sites(HOST,GOOD,false).unwrap(); let allowed = allowlist(&render(&current)).unwrap();
    assert!(unsafe_sites::drift(&current,&allowed).is_empty());
    fails_with(&unsafe_sites::drift(&current,&BTreeMap::new()),"missing from allowlist");
    fails_with(&unsafe_sites::drift(&[],&allowed),"stale unsafe allowlist entry");
}
#[test]
fn lexer_skips_comments_literals_raw_names_and_keeps_lifetimes() {
    let source = r#####"
// unsafe { fake() }
/* unsafe fn fake() {} /* nested unsafe impl Send for X {} */ */
fn safe<'a>(x: &'a str) {
    let text = "unsafe { fake() } \\\"";
    let raw = r###"unsafe fn fake() { \" }"###;
    let bytes = br#"unsafe impl Send for X {}"#;
    let c = c"unsafe { fake() }";
    let raw_c = cr##"unsafe fn fake() {}"##;
    let ch = '\''; let brace = b'{'; let unicode = 'é'; let escape = '\u{7b}';
    let r#unsafe = 1;
}
"#####;
    assert!(sites(HOST,source,false).unwrap().is_empty());
    fails_with(&check_fixture(HOST,&format!("{source}\nfn real() {{ unsafe {{ call(); }} }}"),false),"outside the GPU crate");
    for source in ["/* unclosed", "let x = r###\"unclosed", "fn f() {", "fn f() {]"] { assert!(sites(HOST,source,false).is_err()); }
}
#[test]
fn named_scopes_cover_impls_modules_macros_and_const_generics() {
    let source = "mod nested { impl Device { fn launch<const N: usize>() {\n// SAFETY: The device owns the buffer.\nunsafe { call(); }\n} } }";
    let found = sites(GPU,source,true).unwrap(); assert!(found[0].item.contains("mod:nested::impl:Device::fn:launch"));
    assert!(violations(&found,&BTreeMap::new()).is_empty());
    let source = "macro_rules! kernel { ($name:ident) => { fn $name() {\n// SAFETY: The caller owns the buffer.\nunsafe { call(); }\n} }; }";
    assert!(check_fixture(GPU,source,true).is_empty());
    let top = "// SAFETY: This pointer is valid.\nconst VALUE: u8 = unsafe { call() };";
    fails_with(&check_fixture(GPU,top,true),"named kernel wrapper");
    let signature = "fn launch(f: impl FnOnce()) -> impl Copy {\n// SAFETY: The buffer is live.\nunsafe { call(); }\n}";
    let found = sites(GPU,signature,true).unwrap(); assert_eq!(found[0].item,"fn:launch");
    assert!(violations(&found,&BTreeMap::new()).is_empty());
    assert_eq!(sites(HOST,"type Callback = unsafe fn();",false).unwrap()[0].item,"type:Callback");
    assert_eq!(sites(GPU,&GOOD.replace("fn launch","const fn launch"),true).unwrap()[0].item,"fn:launch");
    assert_eq!(sites(HOST,"#[unsafe(no_mangle)] pub unsafe extern \"C\" fn launch() {}",false).unwrap()[0].item,"fn:launch");
}

#[cfg(unix)]
#[test]
fn inventory_checks_new_files_and_uses_the_nearest_crate() {
    let t = common::Temp::new("unsafe inventory");
    let mut git = Command::new("git"); git.current_dir(&t.0).args(["init","-q"]);
    for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("GIT_") { git.env_remove(key); } }
    vade_tools::output(&mut git).unwrap();
    common::write(&t.0.join(".gitignore"),"ignored.rs\ntarget/\n");
    common::write(&t.0.join("engine/crates/gpu/Cargo.toml"),"[package]\nname = \"gpu\"\n");
    common::write(&t.0.join(GPU),GOOD);
    common::write(&t.0.join("ignored.rs"),"unsafe fn ignored() {}");
    common::write(&t.0.join("target/generated.rs"),"unsafe fn ignored() {}");
    let found = inventory(&t.0).unwrap(); assert_eq!(found.len(),1);
    assert!(violations(&found,&BTreeMap::new()).is_empty());
    // A tracked path remains in Git's inventory after its working copy is deleted.
    let mut add = Command::new("git"); add.current_dir(&t.0).args(["add",GPU]);
    for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("GIT_") { add.env_remove(key); } }
    vade_tools::output(&mut add).unwrap(); fs::remove_file(t.0.join(GPU)).unwrap();
    assert!(inventory(&t.0).unwrap().is_empty());
    common::write(&t.0.join(GPU),GOOD);
    common::write(&t.0.join("host source.rs"),GOOD);
    common::write(&t.0.join("engine/crates/gpu/kernels/Cargo.toml"),"[package]\nname = \"kernels\"\n");
    common::write(&t.0.join("engine/crates/gpu/kernels/src/fixture.rs"),GOOD);
    let found = inventory(&t.0).unwrap(); assert_eq!(found.len(),3);
    let issues = violations(&found,&BTreeMap::new()); assert_eq!(issues.len(),1);
    fails_with(&issues,"host source.rs");
    for crate_path in ["engine/crates/gpu/build-kernels","engine/crates/tp/kernels","engine/crates/gpu/kernels/nested"] {
        common::write(&t.0.join(crate_path).join("Cargo.toml"),"[package]\nname = \"other\"\n");
        common::write(&t.0.join(crate_path).join("src/fixture.rs"),GOOD);
    }
    let found = inventory(&t.0).unwrap(); let issues = violations(&found,&BTreeMap::new()); assert_eq!(issues.len(),4);
    for path in ["gpu/build-kernels","tp/kernels","gpu/kernels/nested"] { fails_with(&issues,path); }
    std::os::unix::fs::symlink("host source.rs",t.0.join("linked.rs")).unwrap();
    assert!(inventory(&t.0).unwrap_err().to_string().contains("symbolic link"));
}

#[cfg(unix)]
#[test]
fn inventory_render_preserves_rulings_and_does_not_approve_new_sites() {
    let t = common::Temp::new("unsafe render");
    let mut git = Command::new("git"); git.current_dir(&t.0).args(["init","-q"]);
    for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("GIT_") { git.env_remove(key); } }
    vade_tools::output(&mut git).unwrap();
    common::write(&t.0.join("tools/Cargo.toml"),"[package]\nname = \"tools\"\n");
    common::write(&t.0.join(HOST),"unsafe fn old() {}");
    let found = inventory(&t.0).unwrap(); let first = render(&found);
    common::write(&t.0.join("tools/unsafe-allowlist.txt"),&first);
    assert!(unsafe_sites::drift(&found,&allowlist(&first).unwrap()).is_empty());
    assert_eq!(first,unsafe_sites::render(&found,&allowlist(&first).unwrap()));
    common::write(&t.0.join(HOST),"unsafe fn old(x: u64) {}");
    unsafe_sites::check(&t.0).unwrap();
    common::write(&t.0.join(HOST),"unsafe fn new() {}");
    assert!(unsafe_sites::check(&t.0).unwrap_err().to_string().contains("outside the GPU crate"));
    let new = inventory(&t.0).unwrap(); let text = unsafe_sites::render(&new,&allowlist(&first).unwrap());
    let allowed = allowlist(&text).unwrap(); assert!(unsafe_sites::drift(&new,&allowed).is_empty());
    fails_with(&violations(&new,&allowed),"needs an allowlist ruling line");
    fs::remove_file(t.0.join(HOST)).unwrap();
    assert!(allowlist(&unsafe_sites::render(&inventory(&t.0).unwrap(),&allowed)).unwrap().is_empty());
}

#[cfg(unix)]
#[test]
fn command_uses_target_tree_allowlist_and_names_read_errors() {
    let t = common::Temp::new("unsafe check root");
    let mut git = Command::new("git"); git.current_dir(&t.0).args(["init","-q"]);
    for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("GIT_") { git.env_remove(key); } }
    vade_tools::output(&mut git).unwrap();
    common::write(&t.0.join("tools/Cargo.toml"),"[package]\nname = \"tools\"\n");
    common::write(&t.0.join(HOST),"unsafe fn old() {}");
    let path = t.0.join("tools/unsafe-allowlist.txt");
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-tools")); command.arg("check-unsafe-sites").arg(&t.0);
    let out = common::run(&mut command,&t.0); assert!(!out.status.success());
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(error.contains("cannot read") && error.contains(path.to_str().unwrap()),"{error}");
    common::write(&path,render(&inventory(&t.0).unwrap()));
    let out = common::run(&mut command,&t.0); assert!(out.status.success(),"{}",common::diagnostic(&out));
    // A valid kernel exception still needs a row. Equality also rejects a stale row.
    common::write(&t.0.join("engine/crates/gpu/Cargo.toml"),"[package]\nname = \"gpu\"\n");
    common::write(&t.0.join(GPU),GOOD);
    let out = common::run(&mut command,&t.0); assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("missing from allowlist"));
    let old = allowlist(&fs::read_to_string(&path).unwrap()).unwrap();
    common::write(&path,unsafe_sites::render(&inventory(&t.0).unwrap(),&old));
    let out = common::run(&mut command,&t.0); assert!(out.status.success(),"{}",common::diagnostic(&out));
    fs::remove_file(t.0.join(GPU)).unwrap();
    let out = common::run(&mut command,&t.0); assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("stale unsafe allowlist entry"));
    let found = inventory(&t.0).unwrap();
    common::write(&path,unsafe_sites::render(&found,&BTreeMap::new()));
    let out = common::run(&mut command,&t.0); assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs an allowlist ruling line"));
    common::write(&path,render(&found));
    let out = common::run(&mut command,&t.0); assert!(out.status.success(),"{}",common::diagnostic(&out));
    common::write(&path,"");
    let out = common::run(&mut command,&t.0); assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("new unsafe fn outside the GPU crate"));
}
