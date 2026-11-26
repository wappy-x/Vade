mod common;
use common::*;
use std::{fs, process::Command};
use vade_tools::release_source::{canonical, prepare};

#[test]
fn comments_and_blank_lines_do_not_move_release_tokens() {
    let plain = "fn f<'a>(x: &'a str)->usize { x.len()+1 }";
    let comment = "//! module docs\n\nfn /* outer /* nested */ end */ f<'a>(x: &'a str)->usize {\n// line\nx.len()+/* value */1\n}";
    assert_eq!(canonical(plain).unwrap(), canonical(comment).unwrap());
    assert_ne!(canonical(plain).unwrap(), canonical(&plain.replace("+1", "+2")).unwrap());
    assert_ne!(canonical("a&&b").unwrap(), canonical("a& &b").unwrap());
}

#[test]
fn literals_lifetimes_and_operators_keep_their_bytes() {
    let literals = [r##"r#"/* raw // */"#"##, r###"br##"// bytes \""##"###,
        r##"cr#"/* C */"#"##, r#""// escaped \" /*""#, "'é'", r"'\u{1f600}'", r"b'\x2f'",
        "1.25e-3f64", "1.", "0xff_u8", "'static", "r#type"];
    for literal in literals { assert_eq!(canonical(literal).unwrap(), format!("{literal}\n")); }
    for malformed in ["/*", "/* /* */", "\"unfinished", "r##\"unfinished\"#"] { assert!(canonical(malformed).is_err()); }
    assert_eq!(canonical("a::b->c..=d<<=e").unwrap(), "a\n::\nb\n->\nc\n..=\nd\n<<=\ne\n");
    assert_eq!(canonical("0xfe-1 + 1.+2").unwrap(), canonical("0xfe/*gap*/-1 + 1./*gap*/+2").unwrap());
}

#[test]
fn preparation_keeps_originals_and_removes_the_revision_input() {
    let t = Temp::new("release source"); let source = t.0.join("source");
    write(&source.join("engine/Cargo.toml"), "[workspace]\n");
    write(&source.join("engine/REVISION"), "a".repeat(40));
    write(&source.join("engine/src/lib.rs"), "// original license\npub fn f() {}\n");
    write(&source.join("engine/target/output"), "ignored");
    let out = t.0.join("prepared"); prepare(&source, &out).unwrap();
    assert!(fs::read_to_string(source.join("engine/src/lib.rs")).unwrap().contains("original license"));
    assert_eq!(fs::read_to_string(out.join("engine/REVISION")).unwrap(), "unknown\n");
    assert_eq!(fs::read_to_string(out.join("engine/src/lib.rs")).unwrap(), "pub\nfn\nf\n(\n)\n{\n}\n");
    assert!(!out.join("engine/target").exists());
    assert!(prepare(&source, &out).is_err());
    assert!(prepare(&source, &source.join("nested")).is_err());
    write(&source.join(".git"), "gitdir: elsewhere");
    assert!(prepare(&source, &t.0.join("checkout")).is_err());
}

#[test]
fn comment_only_commits_build_identical_executables_with_fixed_locations() {
    let t = Temp::new("comment commits"); let repo = t.0.join("repo"); fs::create_dir(&repo).unwrap();
    let git = |args: &[&str]| {
        let result = Command::new("git").arg("-C").arg(&repo).args(args)
            .env("GIT_AUTHOR_NAME", "Test").env("GIT_AUTHOR_EMAIL", "test@example.invalid")
            .env("GIT_COMMITTER_NAME", "Test").env("GIT_COMMITTER_EMAIL", "test@example.invalid").output().unwrap();
        assert!(result.status.success(), "{}", diagnostic(&result)); result.stdout
    };
    git(&["init", "--quiet"]);
    write(&repo.join("engine/Cargo.toml"), "[workspace]\n");
    write(&repo.join("engine/REVISION"), "$Format:%H$\n");
    write(&repo.join(".gitattributes"), "engine/REVISION export-subst\n");
    let code = "fn main() { println!(\"{}:{}\", line!(), column!()); }\n";
    let mut binaries = Vec::new();
    for (name, source) in [("base", code.to_owned()), ("comment", format!("//! changed docs\n\n// extra line\n{code}"))] {
        write(&repo.join("engine/main.rs"), source); git(&["add", "."]); git(&["commit", "--quiet", "-m", name]);
        let archive = t.0.join(format!("{name}.tar")); fs::write(&archive, git(&["archive", "HEAD"])).unwrap();
        let extracted = t.0.join(format!("{name}-archive")); fs::create_dir(&extracted).unwrap();
        let result = Command::new("tar").arg("-xf").arg(archive).arg("-C").arg(&extracted).output().unwrap();
        assert!(result.status.success(), "{}", diagnostic(&result));
        let prepared = t.0.join(name); prepare(&extracted, &prepared).unwrap();
        let binary = prepared.join("flash-serve");
        let mut command = Command::new("rustc");
        command.args(["+1.99.0", "--edition=2024", "--crate-name", "flash_serve", "-C", "strip=symbols", "-C", "opt-level=3"])
            .arg(format!("--remap-path-prefix={}=/vade", prepared.display()))
            .arg(prepared.join("engine/main.rs")).arg("-o").arg(&binary);
        if cfg!(target_os = "macos") { command.args(["-C", "link-arg=-Wl,-no_uuid"]); }
        let result = command.output().unwrap(); assert!(result.status.success(), "{}", diagnostic(&result));
        binaries.push(fs::read(binary).unwrap());
    }
    assert_eq!(binaries[0], binaries[1]);
}
