use std::{env, fs, path::{Path, PathBuf}, process::Command};
#[path = "../gpu/ptx_stamp.rs"]
#[allow(dead_code)] // Kernel-builder methods share the variant definition.
mod ptx_stamp;

fn archive_dirty(root: &Path) -> bool {
    fn newer(dir: &Path, root: &Path, revision: std::time::SystemTime) -> std::io::Result<bool> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            // Managed CLI builds can share generated output through a target symlink.
            if (kind.is_dir() || kind.is_symlink()) && matches!(entry.file_name().to_str(),
                Some("target" | "artifacts" | "provenance" | "__pycache__" | ".git")) { continue; }
            if kind.is_symlink() { return Ok(true); }
            if kind.is_dir() {
                // Generated Cargo/kernel output is not archive source. Directory
                // mtimes themselves change merely because the build ran.
                if newer(&path, root, revision)? { return Ok(true); }
            } else if kind.is_file() {
                if matches!(path.extension().and_then(|s| s.to_str()), Some("ptx" | "pyc"))
                    || entry.file_name() == ".DS_Store"
                    || (entry.file_name() == "Cargo.lock" && dir != root) { continue; }
                if entry.metadata()?.modified()? > revision { return Ok(true); }
            } else { return Ok(true); }
        }
        Ok(false)
    }
    fs::metadata(root.join("REVISION")).and_then(|m| m.modified())
        .and_then(|revision| newer(root, root, revision)).unwrap_or(true)
}

fn build_identity(root: &Path) {
    // Deliberately absent: refresh even when only HEAD, the index or an
    // untracked source file changed. Cargo must not reuse a stale clean stamp.
    println!("cargo:rerun-if-changed={}/build-identity-refresh", env::var("OUT_DIR").unwrap());
    let git = |args: &[&str]| Command::new("git").current_dir(root).args(args).output()
        .ok().filter(|out| out.status.success()).and_then(|out| String::from_utf8(out.stdout).ok());
    // The engine is also a tracked workspace under the monorepo's engine/.
    // Archives copied inside unrelated checkouts must keep their own REVISION.
    let monorepo = git(&["rev-parse", "--show-prefix"])
        .is_some_and(|prefix| prefix.trim() == "engine/")
        && git(&["ls-files", "--error-unmatch", "--", "Cargo.toml"]).is_some();
    let (sha, dirty, source) = if root.join(".git").exists() || monorepo {
        (git(&["rev-parse", "HEAD"]), git(&["status", "--porcelain", "--untracked-files=normal"])
            .is_none_or(|status| !status.is_empty()), "git")
    } else {
        // Release exports have no Git directory. Preserve their source revision
        // rather than accidentally stamping an enclosing checkout's HEAD.
        let manifest = fs::read(root.join("EXPORT-MANIFEST.json")).ok()
            .and_then(|bytes| engine_model::json::parse(&bytes, Default::default()).ok());
        let object = manifest.as_ref().and_then(|v| v.object().ok());
        if let Some(object) = object {
            (object.get("source_revision").and_then(|v| v.string().ok()).map(str::to_owned),
                object.get("preview") == Some(&engine_model::json::Value::Bool(true)), "export")
        } else {
            (fs::read_to_string(root.join("REVISION")).ok(), archive_dirty(root), "archive")
        }
    };
    let sha = sha.as_deref().map(str::trim).filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()));
    println!("cargo:rustc-env=VADE_BUILD_GIT_SHA={}", sha.unwrap_or("unknown"));
    println!("cargo:rustc-env=VADE_BUILD_DIRTY={}", dirty || sha.is_none());
    println!("cargo:rustc-env=VADE_BUILD_SOURCE={source}");
}

#[cfg(test)] #[test] fn archive_edits_are_dirty() {
    let root = env::temp_dir().join(format!("vade-archive-{}", std::process::id())); fs::create_dir_all(root.join("target")).unwrap(); let stamp = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
    for name in ["REVISION", "lib.rs", "target/generated.rs"] { fs::write(root.join(name), b"source").unwrap(); fs::File::open(root.join(name)).unwrap().set_times(fs::FileTimes::new().set_modified(if name.starts_with("target") { stamp + std::time::Duration::from_secs(1) } else { stamp })).unwrap(); }
    assert!(!archive_dirty(&root)); fs::write(root.join("lib.rs"), b"edited").unwrap(); assert!(archive_dirty(&root)); fs::remove_file(root.join("REVISION")).unwrap(); assert!(archive_dirty(&root));
    fs::remove_dir_all(root).unwrap(); }

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    build_identity(&manifest.join("../.."));
    let gpu = manifest.join("../gpu");
    let variant = ptx_stamp::host_variant().unwrap_or_else(|error| panic!("{error}"));
    let ptx = variant.ptx_path(&gpu, false);
    println!("cargo:rerun-if-changed={}", ptx.display());
    let mut bytes = match fs::read(&ptx) {
        Ok(bytes) if !bytes.is_empty() => {
            ptx_stamp::verify(&gpu, &ptx, &bytes, variant).unwrap_or_else(|error|
                panic!("PTX validation failed: {error}; run `{}` first", variant.build_command()));
            bytes
        },
        _ if matches!(variant, ptx_stamp::Variant::Production)
            && env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") => Vec::new(),
        _ => panic!("missing or empty kernel PTX at {}; run `{}` first", ptx.display(), variant.build_command()),
    };
    bytes.push(0);
    fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kernels.ptx"), bytes).unwrap();
}
