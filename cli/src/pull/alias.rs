//! Short model aliases (`inkling`, `qwen`) and read-only Hugging Face cache discovery.
//!
//! Never writes into `~/.cache/huggingface/` (or `$HF_HOME`). When a snapshot
//! stores files as symlinks into a sibling `blobs/` directory, live starts can
//! materialize a zero-copy hardlink view inside `$VADE_HOME/hf-cache/` so the
//! engine's checkpoint-containment checks pass without copying gigabytes.
use std::{
    collections::BTreeSet,
    fs,
    os::unix::fs::MetadataExt,
    path::{Component, Path, PathBuf},
};

/// Pinned repository for Inkling 2 Mini.
pub(crate) const INKLING_REPO: &str = "thinkingmachines/Inkling-2-Mini";
/// Pinned commit for Inkling 2 Mini.
pub(crate) const INKLING_REV: &str = "b85c3cfa615fc413cc9af3379f491b963f3b586a";
/// Pinned repository for Qwen3.8 Flash-Next FP4-Local.
pub(crate) const QWEN_REPO: &str = "nvidia/Qwen3.8-Flash-Next-FP4-Local";
/// Pinned commit for Qwen3.8 Flash-Next FP4-Local.
pub(crate) const QWEN_REV: &str = "b9ce18a6887e18c089fdea8509a33e63429a3300";

/// One pinned model alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Alias {
    pub short: &'static str,
    pub names: &'static [&'static str],
    pub repo: &'static str,
    pub revision: &'static str,
    pub loop_text: bool,
}

pub(crate) const ALIASES: &[Alias] = &[
    Alias {
        short: "inkling",
        names: &["inkling", "inkling-2-mini"],
        repo: INKLING_REPO,
        revision: INKLING_REV,
        loop_text: true,
    },
    Alias {
        short: "qwen",
        names: &["qwen", "qwen3.8"],
        repo: QWEN_REPO,
        revision: QWEN_REV,
        loop_text: false,
    },
];

/// Resolve a short alias (case-insensitive) to its pinned repository and commit.
pub(crate) fn resolve(name: &str) -> Option<Alias> {
    ALIASES
        .iter()
        .copied()
        .find(|alias| alias.names.iter().any(|n| n.eq_ignore_ascii_case(name)))
}

/// Look up an alias by its canonical `<org>/<name>` repository string.
pub(crate) fn by_repo(repo: &str) -> Option<Alias> {
    ALIASES
        .iter()
        .copied()
        .find(|alias| alias.repo.eq_ignore_ascii_case(repo))
}

/// Whether `name` is an Inkling short alias (case-insensitive).
pub(crate) fn is_inkling(name: &str) -> bool {
    resolve(name).is_some_and(|alias| alias.loop_text)
}

/// Hugging Face Hub cache directory: `$HF_HOME/hub`, else `<home>/.cache/huggingface/hub`.
pub(crate) fn hf_hub_dir(home: &Path) -> PathBuf {
    hf_hub_dir_with(home, std::env::var_os("HF_HOME").as_deref())
}

fn hf_hub_dir_with(home: &Path, hf_home: Option<&std::ffi::OsStr>) -> PathBuf {
    if let Some(value) = hf_home.filter(|v| !v.is_empty()) {
        let base = match value.to_str() {
            Some("~") => home.to_path_buf(),
            Some(rest) if rest.starts_with("~/") => home.join(&rest[2..]),
            _ => PathBuf::from(value),
        };
        return base.join("hub");
    }
    home.join(".cache/huggingface/hub")
}

/// HF cache folder for `<org>/<name>`: `models--<org>--<name>`.
pub(crate) fn hf_repo_dir(hub: &Path, repo: &str) -> PathBuf {
    hub.join(format!("models--{}", repo.replace('/', "--")))
}

/// Candidate snapshot folders under `<hub>/models--<org>--<name>/snapshots/`,
/// with `preferred_rev` first when present, then any others in sorted order.
pub(crate) fn hf_snapshots(hub: &Path, repo: &str, preferred_rev: Option<&str>) -> Vec<(PathBuf, String)> {
    let snapshots = hf_repo_dir(hub, repo).join("snapshots");
    let Ok(entries) = fs::read_dir(&snapshots) else { return Vec::new() };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(rev) = name.to_str() else { continue };
        if rev.starts_with('.') { continue; }
        let path = entry.path();
        if path.is_dir() {
            found.push((path, rev.to_owned()));
        }
    }
    found.sort_by(|a, b| {
        let a_pref = preferred_rev == Some(a.1.as_str());
        let b_pref = preferred_rev == Some(b.1.as_str());
        b_pref.cmp(&a_pref).then_with(|| a.1.cmp(&b.1))
    });
    found
}

/// Read-only check that `snapshot` is a complete HF Hub snapshot whose metadata
/// and weight files exist as regular nonempty files inside `snapshot` or the
/// sibling `blobs/` directory of the same `models--...` repo folder.
/// Returns total shard bytes (main + optional draft) when valid.
pub(crate) fn valid_hf_snapshot_bytes(snapshot: &Path) -> Option<u64> {
    let snap_canon = fs::canonicalize(snapshot).ok()?;
    if !snap_canon.is_dir() { return None; }
    let repo_dir = snap_canon.parent()?.parent()?;
    let blobs_canon = fs::canonicalize(repo_dir.join("blobs")).ok();
    let allowed = |path: &Path| -> Option<fs::Metadata> {
        let target = fs::canonicalize(path).ok()?;
        let confined = target.starts_with(&snap_canon)
            || blobs_canon.as_ref().is_some_and(|blobs| target.starts_with(blobs));
        if !confined { return None; }
        let meta = fs::metadata(&target).ok()?;
        (meta.is_file() && meta.len() > 0).then_some(meta)
    };
    allowed(&snap_canon.join("config.json"))?;
    let index_path = snap_canon.join("model.safetensors.index.json");
    allowed(&index_path)?;
    let index_bytes = engine_model::safetensors::read_bounded(&index_path, engine_model::shapes::HEADER_BYTES).ok()?;
    let doc = engine_model::json::parse(&index_bytes, engine_model::json::Limits::default()).ok()?;
    let weight_map = doc.object().ok()?.get("weight_map")?.object().ok()?;
    if weight_map.is_empty() { return None; }
    let mut shards = BTreeSet::new();
    for value in weight_map.values() {
        let name = value.string().ok()?;
        if !name.ends_with(".safetensors")
            || name.contains('\\')
            || name.chars().any(char::is_control)
            || !Path::new(name).components().all(|part| matches!(part, Component::Normal(_)))
        {
            return None;
        }
        shards.insert(name.to_owned());
    }
    let mut total = 0u64;
    for shard in &shards {
        total = total.checked_add(allowed(&snap_canon.join(shard))?.len())?;
    }
    if snap_canon.join("mtp.safetensors").exists() {
        total = total.checked_add(allowed(&snap_canon.join("mtp.safetensors"))?.len())?;
    }
    Some(total)
}

/// Find a valid cached HF Hub snapshot for `repo` (read-only), preferring `preferred_rev`.
pub(crate) fn find_in_hf_cache(home: &Path, repo: &str, preferred_rev: Option<&str>) -> Option<(PathBuf, String)> {
    find_in_hf_hub(&hf_hub_dir(home), repo, preferred_rev)
}

pub(crate) fn find_in_hf_hub(hub: &Path, repo: &str, preferred_rev: Option<&str>) -> Option<(PathBuf, String)> {
    hf_snapshots(hub, repo, preferred_rev)
        .into_iter()
        .find(|(path, _)| valid_hf_snapshot_bytes(path).is_some())
}

/// Create a zero-copy hardlink view of an HF Hub snapshot under `vade_root/hf-cache/<rev>/<repo>`.
/// Never writes into the Hugging Face cache. Every regular file (including symlinked blobs)
/// becomes a direct regular hardlink inside the view directory so the engine's strict
/// checkpoint-containment checks pass without copying weight files.
pub(crate) fn materialize_hf_snapshot(snapshot: &Path, vade_root: &Path, repo: &str, rev: &str) -> Result<PathBuf, String> {
    let snap_canon = fs::canonicalize(snapshot)
        .map_err(|_| format!("Hugging Face cache snapshot {} is unavailable.", snapshot.display()))?;
    let repo_dir = snap_canon.parent().and_then(Path::parent)
        .ok_or("Hugging Face cache snapshot layout is unexpected.")?;
    let blobs_canon = fs::canonicalize(repo_dir.join("blobs")).ok();
    let safe_rev = rev.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) && !rev.starts_with('.');
    if !safe_rev {
        return Err("Hugging Face cache revision name is invalid.".into());
    }
    let view = vade_root.join("hf-cache").join(rev).join(repo);
    fs::create_dir_all(&view)
        .map_err(|error| crate::text::file_error("Cannot create Hugging Face cache view", &view, &error))?;
    let view = fs::canonicalize(&view)
        .map_err(|_| "Cannot resolve Hugging Face cache view folder.".to_string())?;
    let mut pending = vec![(snap_canon.clone(), view.clone(), 0usize)];
    let mut seen = 0usize;
    while let Some((src_dir, dst_dir, depth)) = pending.pop() {
        if depth > 4 { continue; }
        let entries = fs::read_dir(&src_dir)
            .map_err(|error| crate::text::file_error("Cannot read Hugging Face cache snapshot", &src_dir, &error))?;
        for entry in entries {
            seen += 1;
            if seen > 10_000 { return Err("Hugging Face cache snapshot has too many entries.".into()); }
            let entry = entry.map_err(|_| "Cannot read a Hugging Face cache snapshot entry.".to_string())?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') { continue; }
            let src = entry.path();
            let dst = dst_dir.join(&name);
            let sym_meta = fs::symlink_metadata(&src)
                .map_err(|_| "Cannot inspect a Hugging Face cache snapshot entry.".to_string())?;
            if sym_meta.is_dir() {
                fs::create_dir_all(&dst)
                    .map_err(|error| crate::text::file_error("Cannot create Hugging Face cache subfolder", &dst, &error))?;
                pending.push((src, dst, depth + 1));
                continue;
            }
            let target = fs::canonicalize(&src)
                .map_err(|_| format!("Hugging Face cache entry {} is broken.", src.display()))?;
            let confined = target.starts_with(&snap_canon)
                || blobs_canon.as_ref().is_some_and(|blobs| target.starts_with(blobs));
            if !confined {
                return Err(format!("Hugging Face cache entry {} points outside the repository cache.", name.to_string_lossy()));
            }
            let target_meta = fs::metadata(&target)
                .map_err(|_| format!("Cannot inspect Hugging Face cache file {}.", target.display()))?;
            if !target_meta.is_file() { continue; }
            if let Ok(existing) = fs::symlink_metadata(&dst) {
                if existing.is_file() && existing.dev() == target_meta.dev() && existing.ino() == target_meta.ino() {
                    continue;
                }
                fs::remove_file(&dst)
                    .map_err(|error| crate::text::file_error("Cannot refresh Hugging Face cache view entry", &dst, &error))?;
            }
            fs::hard_link(&target, &dst).map_err(|error| {
                if error.kind() == std::io::ErrorKind::CrossesDevices {
                    format!("Cannot link Hugging Face cache snapshot across filesystems ({} -> {}). Run vade pull {} to keep a copy under ~/models.",
                        target.display(), dst.display(), by_repo(repo).map_or(repo, |a| a.short))
                } else {
                    crate::text::file_error("Cannot link Hugging Face cache file", &dst, &error)
                }
            })?;
        }
    }
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn aliases_resolve_case_insensitively() {
        for name in ["inkling", "Inkling", "INKLING-2-MINI", "inkling-2-mini"] {
            let alias = resolve(name).expect(name);
            assert_eq!((alias.repo, alias.revision, alias.loop_text), (INKLING_REPO, INKLING_REV, true));
            assert!(is_inkling(name));
        }
        for name in ["qwen", "Qwen", "qwen3.8", "QWEN3.8"] {
            let alias = resolve(name).expect(name);
            assert_eq!((alias.repo, alias.revision, alias.loop_text), (QWEN_REPO, QWEN_REV, false));
            assert!(!is_inkling(name));
        }
        assert_eq!(resolve("unknown"), None);
        assert_eq!(resolve("thinkingmachines/Inkling-2-Mini"), None);
        assert_eq!(by_repo(INKLING_REPO).unwrap().short, "inkling");
    }

    #[test]
    fn hf_hub_prefers_hf_home_and_pinned_revision_and_materializes_blob_symlinks() {
        let base = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-hf-alias-{}", crate::text::random_hex(8).unwrap()));
        let home = base.join("home");
        let vade_root = home.join(".vade");
        let custom_hf = base.join("custom-hf");
        assert_eq!(hf_hub_dir_with(&home, None), home.join(".cache/huggingface/hub"));
        assert_eq!(hf_hub_dir_with(&home, Some(std::ffi::OsStr::new("~/hf"))), home.join("hf/hub"));
        assert_eq!(hf_hub_dir_with(&home, Some(custom_hf.as_os_str())), custom_hf.join("hub"));

        let hub = home.join(".cache/huggingface/hub");
        let repo_dir = hf_repo_dir(&hub, INKLING_REPO);
        let blobs = repo_dir.join("blobs");
        let other_snap = repo_dir.join("snapshots/0000000000000000000000000000000000000000");
        let pinned_snap = repo_dir.join("snapshots").join(INKLING_REV);
        fs::create_dir_all(&blobs).unwrap();
        // Build a real unserved loop checkpoint in a scratch dir, move files to blobs/, and symlink them in snapshots/.
        for snap in [&other_snap, &pinned_snap] {
            let stage = base.join(format!("stage-{}", snap.file_name().unwrap().to_string_lossy()));
            crate::launch::test_write_unserved_checkpoint(&stage, None);
            fs::create_dir_all(snap).unwrap();
            for entry in fs::read_dir(&stage).unwrap() {
                let entry = entry.unwrap();
                let name = entry.file_name();
                let blob = blobs.join(format!("{}-{}", snap.file_name().unwrap().to_string_lossy(), name.to_string_lossy()));
                fs::rename(entry.path(), &blob).unwrap();
                let rel = Path::new("../../blobs").join(blob.file_name().unwrap());
                symlink(&rel, snap.join(name)).unwrap();
            }
            fs::remove_dir_all(&stage).unwrap();
        }
        // Pinned revision wins even though "0000..." sorts earlier alphabetically.
        let (found, rev) = find_in_hf_hub(&hub, INKLING_REPO, Some(INKLING_REV)).unwrap();
        assert_eq!(rev, INKLING_REV);
        assert_eq!(fs::canonicalize(&found).unwrap(), fs::canonicalize(&pinned_snap).unwrap());
        assert!(valid_hf_snapshot_bytes(&found).unwrap() > 0);
        // Raw snapshot fails engine_model containment because symlinks point to ../../blobs/.
        assert!(engine_model::plan::inkling_index::inspect(&found).is_err());
        // Materializing creates a zero-copy hardlink view inside VADE_HOME that passes real engine inspection,
        // while leaving the HF cache untouched (zero files added to repo_dir).
        let before_entries: usize = fs::read_dir(&pinned_snap).unwrap().count();
        let view = materialize_hf_snapshot(&found, &vade_root, INKLING_REPO, &rev).unwrap();
        assert!(view.starts_with(fs::canonicalize(&vade_root).unwrap()));
        assert!(engine_model::plan::inkling_index::inspect(&view).is_ok());
        assert_eq!(fs::read_dir(&pinned_snap).unwrap().count(), before_entries);
        // Idempotent second call reuses existing hardlinks.
        let view2 = materialize_hf_snapshot(&found, &vade_root, INKLING_REPO, &rev).unwrap();
        assert_eq!(view, view2);
        // Hardlink shares the exact same inode as the HF blob (zero copy).
        let blob_ino = fs::metadata(fs::canonicalize(pinned_snap.join("config.json")).unwrap()).unwrap().ino();
        let view_ino = fs::metadata(view.join("config.json")).unwrap().ino();
        assert_eq!(blob_ino, view_ino);
        fs::remove_dir_all(&base).unwrap();
    }
}
