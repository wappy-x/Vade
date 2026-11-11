//! Give the `vade` program its commit, which `vade --version` prints. A Git
//! checkout answers through Git. A `git archive` export (the release bundle
//! and the cross build) answers through `.vade-build-id`. Anything else has no
//! commit and says so; the CLI never prints a commit that it cannot name.
use std::{path::Path, process::Command};

#[path = "src/launch/build_id.rs"]
#[allow(dead_code)] // The CLI uses the rest of this file.
mod build_id;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    Command::new("git").arg("-C").arg(root).args(args).output().ok()
        .filter(|output| output.status.success()).and_then(|output| String::from_utf8(output.stdout).ok())
}

/// Build again when the commit or the tracked state moves. The ref file can be
/// absent (a packed ref), and then Cargo runs this script on every build.
fn watch(root: &Path) {
    let head = git(root, &["symbolic-ref", "--quiet", "HEAD"]);
    for name in ["HEAD", "index", "packed-refs"].into_iter().chain(head.as_deref().map(str::trim)) {
        if let Some(path) = git(root, &["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={}", root.join(path.trim()).display());
        }
    }
    for name in ["src", "Cargo.toml", "Cargo.lock"] { println!("cargo:rerun-if-changed={name}"); }
}

fn main() {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest).parent().expect("cli/ has a parent folder");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", root.join(build_id::FILE).display());
    // A CLI folder copied alone sits in a folder that is not this tree.
    let probe = root.join("cli/build.rs").is_file().then(|| build_id::probe(root));
    let (commit, modified) = match probe {
        Some(build_id::Probe::Git) => {
            watch(root);
            let status = git(root, &["--no-optional-locks", "status", "--porcelain", "--untracked-files=normal"]);
            (git(root, &["rev-parse", "HEAD"]), status.is_none_or(|status| !status.is_empty()))
        }
        Some(build_id::Probe::Archive(id)) => (Some(id), false),
        _ => (None, false),
    };
    let commit = commit.map(|sha| sha.trim().to_owned()).filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()));
    println!("cargo:rustc-env=VADE_CLI_COMMIT={}", commit.as_deref().unwrap_or(""));
    println!("cargo:rustc-env=VADE_CLI_MODIFIED={modified}");
}
