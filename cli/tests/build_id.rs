//! The launcher's build-id probe inside real `git archive` extractions: a
//! temporary repository, then this repository's own committed attribute.
#[path = "../src/launch/build_id.rs"]
mod build_id;

use build_id::Probe;
use std::{fs, path::{Path, PathBuf}, process::{Command, Stdio}};

fn scratch(name: &str) -> PathBuf {
    let path = fs::canonicalize(std::env::temp_dir()).unwrap()
        .join(format!("vade-build-id-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn git(repository: &Path, args: &[&str]) -> String {
    let output = Command::new("git").arg("-C").arg(repository).args(args).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}

/// `git archive <revision> [paths] | tar -x`, as a downloaded tarball would arrive.
fn extract(repository: &Path, revision: &str, paths: &[&str], destination: &Path) {
    let mut archive = Command::new("git").arg("-C").arg(repository).arg("archive").arg(revision).args(paths)
        .stdout(Stdio::piped()).spawn().unwrap();
    let tar = Command::new("tar").arg("-x").arg("-C").arg(destination)
        .stdin(archive.stdout.take().unwrap()).status().unwrap();
    assert!(archive.wait().unwrap().success() && tar.success());
}

#[test]
fn probe_reads_the_substituted_id_from_a_git_archive_extraction() {
    let repository = scratch("repository");
    for (name, body) in [(".gitattributes", ".vade-build-id export-subst\n"), (build_id::FILE, "$Format:%H$\n"),
        ("engine/Cargo.toml", "engine"), ("harness/Cargo.toml", "harness"), ("rust-toolchain.toml", "toolchain")] {
        fs::create_dir_all(repository.join(name).parent().unwrap()).unwrap();
        fs::write(repository.join(name), body).unwrap();
    }
    git(&repository, &["init", "-q"]);
    git(&repository, &["add", "."]);
    git(&repository, &["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgSign=false",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Fixture"]);
    let revision = git(&repository, &["rev-parse", "HEAD"]).trim().to_owned();
    assert_eq!(build_id::probe(&repository), Probe::Git);

    let tree = scratch("extracted");
    extract(&repository, "HEAD", &[], &tree);
    assert!(!tree.join(".git").exists());
    assert_eq!(fs::read_to_string(tree.join(build_id::FILE)).unwrap(), format!("{revision}\n"));
    assert_eq!(build_id::probe(&tree), Probe::Archive(revision.clone()));
    // Build output after extraction is neither inventory nor an edit.
    fs::create_dir_all(tree.join("engine/target")).unwrap();
    fs::write(tree.join("engine/target/kernels.ptx"), "generated").unwrap();
    let files = build_id::archive_files(&tree).unwrap();
    assert_eq!(files, ["engine/Cargo.toml", "harness/Cargo.toml", "rust-toolchain.toml"]);
    assert_eq!(build_id::archive_id(build_id::probe(&tree), &tree, &files), (revision.clone(), false));

    // An edited extraction and an unsubstituted export both build as unknown.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    fs::write(tree.join("engine/Cargo.toml"), "edited").unwrap();
    assert_eq!(build_id::archive_id(build_id::probe(&tree), &tree, &files), (build_id::UNKNOWN.into(), true));
    fs::write(tree.join(build_id::FILE), "$Format:%H$\n").unwrap();
    assert_eq!(build_id::probe(&tree), Probe::Unknown);
    assert!(build_id::NOTICE.contains("archive") && build_id::NOTICE.contains("git clone"));
    fs::remove_dir_all(repository).unwrap();
    fs::remove_dir_all(tree).unwrap();
}

#[test]
fn this_repository_stamps_its_archives_with_the_commit_id() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    assert!(fs::read_to_string(root.join(".gitattributes")).unwrap().lines().any(|l| l == ".vade-build-id export-subst"));
    let Probe::Git = build_id::probe(root) else {
        // Testing from an archive: the file itself must already be substituted.
        assert!(matches!(build_id::probe(root), Probe::Archive(_)), "unsubstituted {} outside Git", build_id::FILE);
        return;
    };
    assert_eq!(fs::read_to_string(root.join(build_id::FILE)).unwrap(), "$Format:%H$\n");
    let tree = scratch("this-repository");
    extract(root, "HEAD", &[build_id::FILE], &tree);
    let revision = git(root, &["rev-parse", "HEAD"]).trim().to_owned();
    assert_eq!(build_id::probe(&tree), Probe::Archive(revision));
    fs::remove_dir_all(tree).unwrap();
}
