//! A file that `vade pull` hashed on this Spark needs no second hash. The engine
//! skips a file only with its own private stamp, and pull writes none, so a rank
//! that never served hashed the whole checkpoint at install. This writes those
//! stamps from the pull's records, for files that have not changed since: the
//! same claim the engine's own stamp makes. A file with no record gets no stamp,
//! and the engine hashes it as before. Nothing here decides which weights run.
//! The tests below ask the engine's own reader, so a change of its format fails
//! a test here, and it costs a user only one more hash.
use std::{collections::BTreeMap, fs, os::unix::{ffi::OsStrExt, fs::MetadataExt}, path::Path};
use crate::pull::Verified;

const INDEX: &str = "model.safetensors.index.json";
/// The engine's stamp folder under the cache base (`vade-manifest-v1`).
const STAMPS: &str = "vade-manifest-v1";

/// The bytes of one engine stamp: the canonical path, then what identifies the
/// file, then the hash it was checked against.
fn stamp(path: &Path, meta: &fs::Metadata, sha256: &str) -> Option<String> {
    Some(format!("{}\0{} {} {} {} {} {} {} {sha256}\n", path.to_str()?, meta.dev(), meta.ino(), meta.len(), meta.mtime(), meta.mtime_nsec(), meta.ctime(), meta.ctime_nsec()))
}

/// Write the stamps for `model` under `base` and return how many. Any failure
/// leaves the engine to hash, so it is never a reason to stop an install.
pub(super) fn seed(model: &Path, base: &Path) -> Result<usize, String> {
    let root = model.canonicalize().map_err(|_| "Cannot read the checkpoint.")?;
    let files = crate::pull::verified(&root);
    if files.is_empty() { return Ok(0); }
    let stamps = base.join(STAMPS);
    crate::text::private_dir(&stamps)?;
    let mut written = 0;
    for (name, file) in &files {
        // The metadata that the record check compared, so the stamp is that state.
        let path = root.join(name).canonicalize().map_err(|_| "Cannot read a checkpoint file.")?;
        if !path.starts_with(&root) { continue; }
        let Some(text) = stamp(&path, &file.meta, &file.sha256) else { continue };
        crate::text::atomic_write(&stamps.join(crate::sha256::sha(path.as_os_str().as_bytes())), &text)?;
        written += 1;
    }
    list(&root, &files)?;
    Ok(written)
}

/// FP8 keeps its list in the checkpoint, and the engine trusts a stamp only with
/// that list and its size file. Write the two as the engine would, when the
/// engine has not. NVFP4 lists are the operator's and part of the cross-rank
/// identity, so this never writes one.
fn list(root: &Path, files: &BTreeMap<String, Verified>) -> Result<(), String> {
    let manifest = root.join("manifest.sha256");
    let format = engine_model::flash::nvfp4::detect(root).map_err(|error| error.to_string())?;
    if format == engine_model::flash::WeightFormat::Nvfp4 || fs::symlink_metadata(&manifest).is_ok() { return Ok(()); }
    let index = fs::read_to_string(root.join(INDEX)).map_err(|_| "Cannot read the model shard index.")?;
    let shards = super::model::indexed_shards(&index)?;
    let row = |name: &str| Some(format!("{}  {name}\n", files.get(name)?.sha256));
    let Some(rows) = std::iter::once(INDEX).chain(shards.iter().map(String::as_str)).map(row).collect::<Option<String>>() else { return Ok(()) };
    let mut named = BTreeMap::new();
    for name in std::iter::once(INDEX).chain(shards.iter().map(String::as_str)) {
        let Some(file) = files.get(name) else { return Ok(()) };
        named.insert(name, file.meta.len());
    }
    let lengths: String = named.iter().map(|(name, size)| format!("{size}  {name}\n")).collect();
    // Both create only, as vade pull leaves a folder that others can read, which
    // atomic_write refuses. The list goes last: without it, the engine ignores the
    // size file. An old size file stays, and the engine then hashes and replaces it.
    crate::text::create_private(&root.join("manifest.sha256.sizes"), &format!("{}\n{lengths}", crate::sha256::sha(rows.as_bytes())))?;
    crate::text::create_private(&manifest, &rows).map(drop)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The files of a two-shard FP8-style checkpoint in `model`, a folder with
    /// the mode that `vade pull` gives it under umask 022: other users can read it.
    fn files(model: &Path, config: &str) {
        use std::os::unix::fs::PermissionsExt;
        fs::create_dir_all(model).unwrap();
        fs::set_permissions(model, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(model.join("config.json"), config).unwrap();
        let header = br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend(header); shard.extend([1, 2]);
        for name in ["a.safetensors", "b.safetensors"] { fs::write(model.join(name), &shard).unwrap(); }
        fs::write(model.join(INDEX), r#"{"metadata":{"total_size":4},"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}"#).unwrap();
    }

    /// That checkpoint with a finished pull's records.
    fn checkpoint(root: &Path, config: &str) -> PathBuf {
        let model = root.join("model");
        files(&model, config);
        crate::text::private_dir(&model.join(crate::pull::STATE)).unwrap();
        crate::pull::record_for_test(&model, &[INDEX, "a.safetensors", "b.safetensors", "config.json"]);
        model
    }

    /// A folder for the test whose parent folders no other user can write. The
    /// engine uses its stamp cache only below such folders. A shared temporary
    /// folder such as `/tmp` on Linux fails that rule, so the test then uses the
    /// folder of its own test binary.
    pub(in crate::worker) fn private_base() -> PathBuf {
        let owner = fs::metadata(std::env::current_exe().unwrap()).unwrap().uid();
        let usable = |path: &Path| path.ancestors().all(|folder| fs::symlink_metadata(folder)
            .is_ok_and(|m| m.is_dir() && m.mode() & 0o022 == 0 && [0, owner].contains(&m.uid())));
        let temporary = fs::canonicalize(std::env::temp_dir()).unwrap();
        if usable(&temporary) { return temporary; }
        let binary = fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
        let folder = binary.parent().unwrap().to_path_buf();
        assert!(usable(&folder), "no private folder for the seed test: {} and {} are shared", temporary.display(), folder.display());
        folder
    }

    /// The child test process sets the cache folder that the engine reads. Its
    /// current folder is the checkpoint. In `engine`, a copy with no pull records,
    /// it hashes every file and writes the two files as the engine writes them.
    #[test]
    fn the_engine_reuses_every_file_a_pull_checked() {
        let _guard = crate::runner::process_test_lock();
        let root = private_base().join(format!("vade-seed-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let config = crate::launch::test_config(32768);
        let model = checkpoint(&root, &config);
        files(&root.join("engine"), &config);
        let base = root.join("cache");
        crate::text::private_dir(&base).unwrap();
        assert_eq!(seed(&model, &base).unwrap(), 4);
        let list = fs::read_to_string(model.join("manifest.sha256")).unwrap();
        assert!(list.starts_with(&format!("{}  {INDEX}\n", crate::sha256::digest_reader(fs::File::open(model.join(INDEX)).unwrap()).unwrap())), "{list}");
        assert_eq!(list.lines().count(), 3);
        let mut child = crate::runner::Exec::new("seeded checkpoint fixture", std::env::current_exe().unwrap().to_str().unwrap(),
            vec!["--exact".into(), "worker::seed::tests::seeded_child".into(), "--nocapture".into()]);
        child.env.insert("XDG_CACHE_HOME".into(), base.to_string_lossy().into_owned());
        child.cwd = Some(model.clone());
        child.cancellable = false;
        use crate::runner::Runner;
        let output = crate::runner::SystemRunner { dry: false }.run(&child).unwrap();
        assert!(output.success && output.stdout.contains("hashed=0 reused=3"), "{}", output.stdout);
        for name in ["manifest.sha256", "manifest.sha256.sizes"] {
            assert_eq!(fs::read(model.join(name)).unwrap(), fs::read(root.join("engine").join(name)).unwrap(), "{name}");
        }
        // A file that changes gets no stamp from the old record.
        fs::write(model.join("b.safetensors"), [9; 64]).unwrap();
        assert!(!crate::pull::verified(&model).contains_key("b.safetensors"));
        assert!(crate::pull::verified(&model).contains_key("a.safetensors"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn seeded_child() {
        let model = std::env::current_dir().unwrap();
        if !model.join("manifest.sha256.sizes").exists() { return; }
        let plan = engine_model::flash::manifest::ReceiptPlan::open(&model).unwrap();
        let receipt = plan.cached().unwrap().expect("the engine reuses the seeded stamps");
        println!("hashed={} reused={}", receipt.hashed_bytes, receipt.reused_files);
        engine_model::flash::manifest::ReceiptPlan::open(&model.join("../engine")).unwrap().finish().unwrap();
    }

    #[test]
    fn no_pull_record_or_an_unfinished_pull_writes_nothing() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-seed-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let model = checkpoint(&root, &crate::launch::test_config(32768));
        let base = root.join("cache");
        crate::text::private_dir(&base).unwrap();
        fs::write(model.join(crate::pull::STATE).join("updating"), "commit\n").unwrap();
        assert_eq!(seed(&model, &base).unwrap(), 0);
        fs::remove_file(model.join(crate::pull::STATE).join("updating")).unwrap();
        fs::remove_file(model.join(crate::pull::RECORD)).unwrap();
        assert_eq!(seed(&model, &base).unwrap(), 0);
        assert!(!model.join("manifest.sha256").exists() && !base.join(STAMPS).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_nvfp4_checkpoint_gets_stamps_and_never_a_list() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-seed-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let model = checkpoint(&root, &crate::launch::test_mixed_config(32768));
        let base = root.join("cache");
        crate::text::private_dir(&base).unwrap();
        assert_eq!(seed(&model, &base).unwrap(), 4);
        assert!(!model.join("manifest.sha256").exists() && !model.join("manifest.sha256.sizes").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
