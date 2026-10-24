//! Payload identity for loading and serving. Private verification stamps avoid reading
//! unchanged files again; full verification remains available for media scrubs.
use super::{Checkpoint, WeightFormat, checkpoint_limits};
use crate::{Result, invalid, safetensors};
use std::{collections::{BTreeMap, BTreeSet}, fs::{self, File}, path::{Path, PathBuf}, sync::atomic::{AtomicUsize, Ordering}};

use engine_hash as sha256;
pub const HASH_BUFFER_BYTES: usize = sha256::FILE_BUFFER_BYTES;

pub struct Verification {
    pub files: usize,
    pub bytes: u64,
    pub hashed_bytes: u64,
    pub reused_files: usize,
    /// Identifies the expected hashes as well as the files checked against them.
    pub manifest_sha256: String,
    root: std::path::PathBuf,
}

/// Accept standard sha256sum/shasum output: 64 hex digits, two spaces (or
/// space-star), then one safe relative path. Duplicate rows are refused.
pub fn verify(checkpoint: &Checkpoint, root: &Path, manifest: &Path) -> Result<Verification> {
    verify_files(checkpoint, root, manifest, true)
}

/// Prepare the loader's verification cache without loading weights. A cache
/// miss still hashes every byte against the supplied manifest; unchanged files
/// reuse only the same private identity-bound stamps as normal startup.
pub fn verify_cached(checkpoint: &Checkpoint, root: &Path, manifest: &Path) -> Result<Verification> {
    verify_files(checkpoint, root, manifest, false)
}

fn verify_files(checkpoint: &Checkpoint, root: &Path, manifest: &Path, full: bool) -> Result<Verification> {
    let root = root.canonicalize()?;
    let mixed = checkpoint.config.mixed.is_some();
    // Mixed storage is declared by config.json. Without publisher hashes, retain
    // bounded header, payload-size and index reconciliation; invent no hashes.
    let missing = if mixed {
        match fs::symlink_metadata(manifest) {
            Ok(_) => false,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => return Err(e.into()),
        }
    } else {
        false
    };
    if missing {
        let shards = checkpoint
            .index
            .inspect_shards(&root, checkpoint_limits(checkpoint.config.format, checkpoint.config.mixed.is_some()))?;
        return Ok(Verification {
            files: shards.len(),
            bytes: shards.values().map(|h| h.file_bytes).sum(),
            hashed_bytes: 0,
            reused_files: 0,
            manifest_sha256: String::new(),
            root,
        });
    }
    let (hashes, manifest_sha256) = read_manifest(manifest, mixed)?;
    let mut required = checkpoint.index.weight_map.values().cloned().collect::<BTreeSet<_>>();
    required.insert(INDEX.into());
    if !mixed { for name in ["config.json", "hf_quant_config.json",
        "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
        required.insert(name.into());
    } }
    if required.iter().any(|name| !hashes.contains_key(name)) {
        return Err(invalid(if mixed { "mixed manifest must hash every shard and index" }
            else { "NVFP4 manifest must hash every shard, config, index and tokenizer/template file" }));
    }
    let (_, bytes, hashed_bytes, reused_files) = hash_files(&root, &hashes, full, true)?;
    Ok(Verification { files: hashes.len(), bytes, hashed_bytes, reused_files, manifest_sha256, root })
}

fn read_manifest(manifest: &Path, mixed: bool) -> Result<(BTreeMap<String, String>, String)> {
    let bytes = safetensors::read_bounded(manifest, 1024 * 1024)
        .map_err(|e| invalid(format!("cannot read SHA-256 manifest: {e}")))?;
    let text =
        std::str::from_utf8(&bytes).map_err(|_| invalid("SHA-256 manifest is not UTF-8"))?;
    let mut hashes = BTreeMap::new();
    for line in text.lines().filter(|line| !line.is_empty()) {
        if line.len() < 67
            || !line.as_bytes()[..64].iter().all(u8::is_ascii_hexdigit)
            || !matches!(&line.as_bytes()[64..66], b"  " | b" *")
        {
            return Err(invalid("invalid SHA-256 manifest row"));
        }
        let name = &line[66..];
        if mixed {
            safetensors::validate_relative_name(name)?;
        } else {
            let mut parts = Path::new(name).components();
            if !matches!(parts.next(), Some(std::path::Component::Normal(_)))
                || parts.next().is_some()
                || name.contains(['/', '\\', '\0', '\r', '\n'])
            {
                return Err(invalid("unsafe SHA-256 manifest filename"));
            }
        }
        if hashes
            .insert(name.to_owned(), line[..64].to_ascii_lowercase())
            .is_some()
        {
            return Err(invalid("unsafe or duplicate SHA-256 manifest filename"));
        }
    }
    Ok((hashes, sha256::digest(&bytes)))
}
type Hashes = (BTreeMap<String, String>, u64, u64, usize);
fn hash_files(root: &Path, hashes: &BTreeMap<String, String>, full: bool, strict: bool) -> Result<Hashes> {
    let cache = cache_directory();
    let entries = hashes.iter().collect::<Vec<_>>();
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(8).min(entries.len());
    let results = std::thread::scope(|scope| {
        let handles = (0..workers).map(|_| {
                scope.spawn(|| -> Result<Hashes> {
            let (mut bytes, mut hashed, mut reused) = (0u64, 0u64, 0usize);
            let mut found = BTreeMap::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(name, expected)) = entries.get(i) else { break; };
                let (hash, size, hit) = hash_file(root, name, expected, cache.as_deref(), full, strict)?;
                found.insert(name.clone(), hash);
                bytes = bytes.checked_add(size).ok_or_else(|| invalid("manifest byte total overflow"))?;
                if hit { reused += 1; } else { hashed += size; }
            }
            Ok((found, bytes, hashed, reused))
        })
            }).collect::<Vec<_>>();
        handles.into_iter().map(|handle| {
                handle.join().unwrap_or_else(|_| Err(invalid("manifest hash worker panicked")))
            })
            .collect::<Vec<_>>()
    });
    let (mut bytes, mut hashed_bytes, mut reused_files) = (0u64, 0u64, 0usize);
    let mut found = BTreeMap::new();
    for result in results {
        let (files, size, hashed, reused) = result?;
        found.extend(files);
        bytes = bytes.checked_add(size).ok_or_else(|| invalid("manifest byte total overflow"))?;
        hashed_bytes += hashed;
        reused_files += reused;
    }
    Ok((found, bytes, hashed_bytes, reused_files))
}

const INDEX: &str = "model.safetensors.index.json";

/// Goldens pin: index sha256sum row first, then sorted shard rows, each with LF.
pub fn checkpoint_sha256(hashes: &BTreeMap<String, String>) -> String {
    sha256::digest(manifest_text(hashes).as_bytes())
}
fn manifest_text(hashes: &BTreeMap<String, String>) -> String {
    let mut text = String::new();
    if let Some(hash) = hashes.get(INDEX) { text.push_str(&format!("{hash}  {INDEX}\n")); }
    for (name, hash) in hashes.iter().filter(|(name, _)| name.as_str() != INDEX) {
        text.push_str(&format!("{hash}  {name}\n"));
    }
    text
}

#[derive(Debug)]
pub struct Receipt {
    pub sha256: String,
    pub bytes: u64,
    pub hashed_bytes: u64,
    pub reused_files: usize,
}

/// Captures the index and file identities before serving starts. Only small
/// metadata is read here; a cold payload is hashed by the caller's background task.
#[derive(Debug)]
pub struct ReceiptPlan {
    root: PathBuf,
    manifest: PathBuf,
    manifest_sha256: Option<String>,
    hashes: BTreeMap<String, String>,
    snapshots: BTreeMap<String, (u64, Vec<u8>)>,
    stale: bool,
    pub format: WeightFormat,
    mixed: bool,
}
impl ReceiptPlan {
    pub fn open(root: &Path) -> Result<Self> {
        let root = root.canonicalize()?;
        let format = super::nvfp4::detect(&root)?;
        let mixed = if format == WeightFormat::Nvfp4 {
            let bytes = safetensors::read_bounded(&root.join("config.json"), 1024 * 1024)?;
            let doc = crate::json::parse(&bytes, crate::json::Limits::default())?;
            doc.object()?
                .get("quantization_config")
                .map(|q| {
                    q.object().map(|q| {
                        !q.contains_key("config_groups")
                            && q.get("quant_method").and_then(|v| v.string().ok())
                                == Some("modelopt")
                    })
                })
                .transpose()?
                .unwrap_or(false)
        } else {
            false
        };
        let index_path = root.join(INDEX);
        let index_before = fs::metadata(&index_path).ok();
        let (names, index_hash) = if index_path.try_exists()? {
            let limits = checkpoint_limits(format, mixed);
            let bytes = safetensors::read_bounded(&index_path, limits.header_bytes)?;
            let index = if mixed {
                safetensors::Index::parse_mixed(&bytes, limits)?
            } else {
                safetensors::Index::parse(&bytes, limits)?
            };
            let names = index.weight_map.into_values().collect::<BTreeSet<_>>();
            if index_before.as_ref().map(|m| identity(&index_path, m, ""))
                != Some(identity(&index_path, &fs::metadata(&index_path)?, ""))
            {
                return Err(invalid("checkpoint index changed while reading"));
            }
            (names, Some(sha256::digest(&bytes)))
        } else {
            (BTreeSet::from(["model.safetensors".into()]), None)
        };
        if names.is_empty() {
            return Err(invalid("checkpoint has no weight files"));
        }
        let manifest = if format == WeightFormat::Nvfp4 {
            std::env::var_os("FLASH_NVFP4_MANIFEST")
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("manifest.sha256"))
        } else {
            root.join("manifest.sha256")
        };
        // Serving manifests are a cache written by us. An NVFP4 manifest is
        // operator-supplied verification input and keeps its strict contract.
        let expected = (|| -> Result<_> {
            if !manifest.try_exists()? {
                return Ok((BTreeMap::new(), None));
            }
            let (hashes, digest) = read_manifest(&manifest, mixed)?;
            if index_hash
                .as_ref()
                .is_some_and(|hash| hashes.get(INDEX) != Some(hash))
            {
                return Err(invalid("checkpoint manifest index SHA-256 mismatch"));
            }
            if names.iter().any(|name| !hashes.contains_key(name)) {
                return Err(invalid("checkpoint manifest is missing a shard"));
            }
            Ok((hashes, Some(digest)))
        })();
        let (expected, manifest_sha256, stale) = match expected {
            Ok((hashes, digest)) => (hashes, digest, false),
            Err(error) if format != WeightFormat::Nvfp4 => {
                eprintln!("checkpoint_hash state=failed-stale error={error}");
                (BTreeMap::new(), None, true)
            }
            Err(error) => return Err(error),
        };
        let mut hashes = BTreeMap::new();
        for name in names {
            if name.contains(['\r', '\n']) {
                return Err(invalid("checkpoint filename cannot be a sha256sum row"));
            }
            let hash = expected.get(&name).cloned();
            hashes.insert(name, hash.unwrap_or_default());
        }
        if let Some(hash) = index_hash {
            hashes.insert(INDEX.into(), hash);
        }
        let mut snapshots = BTreeMap::new();
        for name in hashes.keys() {
            let path = root.join(name).canonicalize()?;
            if !path.starts_with(&root) {
                return Err(invalid("checkpoint file escapes checkpoint"));
            }
            let meta = fs::metadata(&path)?;
            if !meta.is_file() {
                return Err(invalid("checkpoint entry is not a regular file"));
            }
            snapshots.insert(name.clone(), (meta.len(), identity(&path, &meta, "")));
        }
        let mut plan = Self {
            root,
            manifest,
            manifest_sha256,
            hashes,
            snapshots,
            stale,
            format,
            mixed,
        };
        if let Err(error) = plan.validate_sizes() {
            eprintln!("checkpoint_hash state=failed-stale error={error}");
            plan.invalidate();
        }
        plan.unchanged()?;
        Ok(plan)
    }
    fn sizes_path(&self) -> PathBuf {
        let mut path = self.manifest.as_os_str().to_owned();
        path.push(".sizes");
        path.into()
    }
    fn validate_sizes(&self) -> Result<()> {
        let path = self.sizes_path();
        if !path.try_exists()? {
            // A missing sibling is an incomplete cache, not a weight error.
            // Supplied NVFP4 manifests can precede their serving size cache.
            return if self.manifest_sha256.is_some() && self.format != WeightFormat::Nvfp4 {
                Err(invalid("checkpoint manifest sizes missing"))
            } else { Ok(()) };
        }
        let bytes = safetensors::read_bounded(&path, 1024 * 1024)?;
        let text = std::str::from_utf8(&bytes).map_err(|_| invalid("manifest sizes are not UTF-8"))?;
        let mut rows = text.lines();
        if rows.next() != self.manifest_sha256.as_deref() {
            return Err(invalid("manifest sizes digest mismatch"));
        }
        let mut sizes = BTreeMap::new();
        for row in rows {
            let (size, name) = row.split_once("  ").ok_or_else(|| invalid("invalid manifest size row"))?;
            let size = size.parse::<u64>().map_err(|_| invalid("invalid manifest size"))?;
            if sizes.insert(name, size).is_some() { return Err(invalid("duplicate manifest size row")); }
        }
        if self.snapshots.iter().any(|(name, (size, _))| sizes.get(name.as_str()) != Some(size)) {
            return Err(invalid("checkpoint manifest file size mismatch"));
        }
        Ok(())
    }
    pub fn is_stale(&self) -> bool { self.stale }
    fn invalidate(&mut self) {
        self.stale = true;
        if self.format != WeightFormat::Nvfp4 {
            self.manifest_sha256 = None;
            for (name, hash) in &mut self.hashes {
                if name != INDEX { hash.clear(); }
            }
        }
    }
    fn unchanged(&self) -> Result<()> {
        for (name, (_, before)) in &self.snapshots {
            let path = self.root.join(name).canonicalize()?;
            if !path.starts_with(&self.root) || identity(&path, &fs::metadata(&path)?, "") != *before {
                return Err(invalid(format!("checkpoint file changed during receipt: {name}")));
            }
        }
        Ok(())
    }

    /// A cache miss returns immediately, including a same-size replaced shard.
    pub fn cached(&self) -> Result<Option<Receipt>> {
        if self.stale { return Ok(None); }
        let Some(cache) = cache_directory() else { return Ok(None); };
        for (name, expected) in &self.hashes {
            if expected.is_empty() { return Ok(None); }
            let path = self.root.join(name).canonicalize()?;
            let key = identity(&path, &fs::metadata(&path)?, expected);
            let stamp = cache.join(sha256::digest(path.as_os_str().as_encoded_bytes()));
            if !stamp_matches(&stamp, &key) { return Ok(None); }
        }
        self.unchanged()?;
        self.persist(&self.hashes);
        let bytes = self.snapshots.values().try_fold(0u64, |total, (size, _)| total.checked_add(*size))
            .ok_or_else(|| invalid("checkpoint byte total overflow"))?;
        Ok(Some(Receipt { sha256: checkpoint_sha256(&self.hashes),
            bytes, hashed_bytes: 0, reused_files: self.hashes.len() }))
    }

    pub fn finish(self) -> Result<Receipt> {
        self.finish_with(|| {})
    }

    /// Notify shared health metadata if payload verification discovers stale
    /// expected hashes. File changes during this load remain real failures.
    pub fn finish_with(mut self, on_stale: impl FnOnce()) -> Result<Receipt> {
        self.unchanged()?;
        let strict = self.format == WeightFormat::Nvfp4;
        let (mut hashes, mut bytes, mut hashed_bytes, mut reused_files) =
            hash_files(&self.root, &self.hashes, self.stale, strict)?;
        if !strict && self.hashes.iter().any(|(name, hash)| !hash.is_empty() && hashes.get(name) != Some(hash)) {
            self.invalidate();
            on_stale();
            self.unchanged()?;
            let fresh = hash_files(&self.root, &self.hashes, true, false)?;
            hashes = fresh.0;
            bytes = fresh.1;
            hashed_bytes += fresh.2;
            reused_files = fresh.3;
        }
        self.unchanged()?;
        self.persist(&hashes);
        self.unchanged()?;
        Ok(Receipt { sha256: checkpoint_sha256(&hashes), bytes, hashed_bytes, reused_files })
    }

    fn persist(&self, hashes: &BTreeMap<String, String>) {
        let result = (|| -> Result<()> {
            let text = manifest_text(hashes);
            if self.stale && self.format != WeightFormat::Nvfp4 {
                publish_file(&self.manifest, text.as_bytes(), true)?;
            } else if self.manifest_sha256.is_none() {
                publish_file(&self.manifest, text.as_bytes(), false)?;
            }
            // Preserve extra NVFP4 config/tokenizer rows and bind sizes to the
            // actual manifest, including when another rank seeded it first.
            let (expected, digest) = read_manifest(&self.manifest, self.mixed)?;
            if hashes.iter().any(|(name, hash)| expected.get(name) != Some(hash)) {
                return Err(invalid("checkpoint manifest changed during receipt"));
            }
            let mut sizes = format!("{digest}\n");
            for (name, (size, _)) in &self.snapshots { sizes.push_str(&format!("{size}  {name}\n")); }
            publish_file(&self.sizes_path(), sizes.as_bytes(), self.stale)?;
            self.validate_persisted_sizes(&digest)
        })();
        if result.is_err() { eprintln!("checkpoint_manifest persistence_failed=true; verified receipt remains available"); }
    }
    fn validate_persisted_sizes(&self, digest: &str) -> Result<()> {
        let bytes = safetensors::read_bounded(&self.sizes_path(), 1024 * 1024)?;
        let mut expected = format!("{digest}\n");
        for (name, (size, _)) in &self.snapshots { expected.push_str(&format!("{size}  {name}\n")); }
        if bytes != expected.as_bytes() { return Err(invalid("persisted manifest sizes differ")); }
        Ok(())
    }
}
// Atomic publication: only a stale server-written cache permits replacement.
// Concurrent ranks verifying the same immutable checkpoint publish equal rows.
fn publish_file(path: &Path, bytes: &[u8], replace: bool) -> std::io::Result<()> {
    use std::io::Write;
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    let tmp = path.with_extension(format!("{}-{}.tmp", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let result = (|| {
        let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        if replace { return fs::rename(&tmp, path); }
        match fs::hard_link(&tmp, path) {
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            result => result,
        }
    })();
    let _ = fs::remove_file(tmp);
    result
}

// Stamps are an optimization only. Missing, malformed, inaccessible or insecure
// cache storage falls back to hashing, and cannot suppress a hash mismatch.
#[cfg(unix)]
fn private(metadata: &fs::Metadata, directory: bool, owner: u32) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.uid() == owner && metadata.mode() & 0o077 == 0
        && if directory { metadata.is_dir() } else { metadata.is_file() }
}

#[cfg(unix)]
fn cache_directory() -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    if !base.is_absolute() { return None; }
    let path = base.join("vade-manifest-v1");
    fs::DirBuilder::new().recursive(true).mode(0o700).create(&path).ok()?;
    // A fresh file tells us the effective owner without FFI. An inaccessible or
    // foreign private directory cannot supply a usable verification cache.
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    let probe = path.join(format!(".owner-{}-{}", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let file = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&probe).ok()?;
    let owner = file.metadata().ok().map(|m| m.uid());
    drop(file);
    fs::remove_file(&probe).ok()?;
    let owner = owner?;
    if !private(&fs::symlink_metadata(&path).ok()?, true, owner) { return None; }
    // Protect pathname opens from directory replacement by another account.
    // Symlinked or shared writable cache ancestors simply disable the cache.
    for ancestor in path.ancestors().skip(1) {
        let metadata = fs::symlink_metadata(ancestor).ok()?;
        if !metadata.is_dir() || metadata.mode() & 0o022 != 0
            || ![0, owner].contains(&metadata.uid()) { return None; }
    }
    Some(path)
}

#[cfg(not(unix))]
fn cache_directory() -> Option<PathBuf> { None }

#[cfg(unix)]
fn identity(path: &Path, metadata: &fs::Metadata, expected: &str) -> Vec<u8> {
    use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};
    let mut value = path.as_os_str().as_bytes().to_vec();
    value.extend_from_slice(format!("\0{} {} {} {} {} {} {} {expected}\n", metadata.dev(), metadata.ino(),
        metadata.len(), metadata.mtime(), metadata.mtime_nsec(), metadata.ctime(), metadata.ctime_nsec()).as_bytes());
    value
}

#[cfg(not(unix))]
fn identity(path: &Path, metadata: &fs::Metadata, expected: &str) -> Vec<u8> {
    format!("{path:?} {} {:?} {expected}", metadata.len(), metadata.modified()).into_bytes()
}

#[cfg(unix)]
fn stamp_matches(path: &Path, identity: &[u8]) -> bool {
    use std::{io::Read, os::unix::fs::MetadataExt};
    let Some(owner) = path.parent().and_then(|p| fs::symlink_metadata(p).ok()).map(|m| m.uid()) else { return false; };
    let Ok(file) = File::open(path) else { return false; };
    if !file.metadata().is_ok_and(|m| private(&m, false, owner) && m.len() <= 65536) { return false; }
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes).is_ok() && bytes == identity
}

#[cfg(not(unix))]
fn stamp_matches(_: &Path, _: &[u8]) -> bool { false }

#[cfg(unix)]
fn save_stamp(path: &Path, identity: &[u8]) -> std::io::Result<()> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    let tmp = path.with_extension(format!("{}-{}.tmp", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let mut file = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
    let result = (|| { file.write_all(identity)?; file.sync_all()?; fs::rename(&tmp, path) })();
    if result.is_err() { let _ = fs::remove_file(tmp); }
    result
}

#[cfg(not(unix))]
fn save_stamp(_: &Path, _: &[u8]) -> std::io::Result<()> { Ok(()) }

fn hash_file(root: &Path, name: &str, expected: &str, cache: Option<&Path>, full: bool, strict: bool) -> Result<(String, u64, bool)> {
    let path = root.join(name).canonicalize()?;
    if !path.starts_with(root) { return Err(invalid("manifest file escapes checkpoint")); }
    let file = File::open(&path)?;
    let before = file.metadata()?;
    if !before.is_file() { return Err(invalid("manifest entry is not a regular file")); }
    let key = identity(&path, &before, expected);
    // The name of a stamp identifies the canonical path; its contents bind the
    // expected SHA-256 and all metadata, including ctime to catch restored mtimes.
    let stamp = cache.map(|cache| cache.join(sha256::digest(path.as_os_str().as_encoded_bytes())));
    let hit = !expected.is_empty() && !full && stamp.as_ref().is_some_and(|stamp| stamp_matches(stamp, &key));
    let hash = if hit { expected.to_owned() } else {
        let result = sha256::digest_file_with_stats(&file);
        match result {
            Ok(digest) => {
                eprintln!("checkpoint_hash_file io_mode={} direct_bytes={} buffered_bytes={} bytes={}",
                    digest.io_mode, digest.direct_bytes, digest.buffered_bytes, before.len());
                digest.sha256
            }
            Err(error) => {
                let policy = if cfg!(target_os = "linux") { "direct-or-bounded-buffer" } else { "portable-buffered" };
                eprintln!("checkpoint_hash_file state=failed io_policy={policy} error={error}");
                return Err(error.into());
            }
        }
    };
    if strict && !expected.is_empty() && hash != expected {
        return Err(invalid(format!("SHA-256 mismatch: {name}")));
    }
    if identity(&path, &file.metadata()?, expected) != key
        || identity(&path, &fs::metadata(root.join(name))?, expected) != key {
        return Err(invalid(format!("manifest file changed during verification: {name}")));
    }
    if !hit {
        if let Some(stamp) = stamp { let _ = save_stamp(&stamp, &identity(&path, &before, &hash)); }
    }
    Ok((hash, before.len(), hit))
}

impl Checkpoint {
    pub fn prepare_load(&self, root: &Path) -> Result<()> {
        if self.config.format != WeightFormat::Nvfp4 { return Ok(()); }
        self.config.format.require_opt_in(WeightFormat::opt_in()?)?;
        let verification = self.manifest.get_or_init(|| {
            let path = std::env::var_os("FLASH_NVFP4_MANIFEST")
                .map(std::path::PathBuf::from).unwrap_or_else(|| root.join("manifest.sha256"));
            let start = std::time::Instant::now();
            let result = (|| {
                let full = match std::env::var("FLASH_VERIFY_FULL").as_deref() {
                    Err(std::env::VarError::NotPresent) | Ok("0") => false,
                    Ok("1") => true,
                    _ => return Err(invalid("FLASH_VERIFY_FULL must be 0 or 1")),
                };
                verify_files(self, root, &path, full)
            })().map_err(|e| e.to_string());
            if let Ok(v) = &result {
                eprintln!("nvfp4_manifest files={} bytes={} sha256={} elapsed_ms={:.3} hashed_bytes={} reused_files={} hash_buffer_bytes_per_worker_max={} max_workers=8",
                    v.files, v.bytes, v.manifest_sha256, start.elapsed().as_secs_f64() * 1000.0, v.hashed_bytes, v.reused_files, HASH_BUFFER_BYTES);
            }
            result
        }).as_ref().map_err(|e| invalid(e.clone()))?;
        if verification.root != root.canonicalize()? {
            return Err(invalid("NVFP4 load root differs from verified manifest"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::Config;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!("flash-manifest-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir_all(root.join("tensors/sub-dir_1")).unwrap();
            let header = br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header); bytes.extend([1, 2]);
            fs::write(root.join("tensors/sub-dir_1/a.safetensors"), &bytes).unwrap();
            fs::write(root.join("a.safetensors"), bytes).unwrap();
            fs::write(root.join(INDEX), br#"{"metadata":{"total_size":2},"weight_map":{"a":"a.safetensors"}}"#).unwrap();
            for name in ["config.json", "hf_quant_config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
                fs::write(root.join(name), b"{}").unwrap();
            }
            Self(root)
        }
        fn checkpoint(&self, mixed: bool) -> Checkpoint {
            // These tests isolate file verification from architecture validation.
            let model = crate::plan::ModelPlan::from_config(include_bytes!("../../tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json")).unwrap();
            let config = Config {
                format: WeightFormat::Nvfp4,
                mixed: mixed.then_some(model),
                hidden: 1,
                layers: vec![],
                query_heads: 1,
                kv_heads: 1,
                head_dim: 1,
                linear_key_heads: 1,
                linear_value_heads: 1,
                linear_key_dim: 1,
                linear_value_dim: 1,
                linear_conv: 1,
                experts: 1,
                top_k: 1,
                renormalize_top_k: false,
                expert_width: 1,
                shared_width: 1,
                branches: 1,
                mix_rank: 1,
                vocab: 1,
                max_positions: 1,
                eos_token: 0,
                epsilon: 1e-6,
                rope_theta: 10000.,
                rotary_dim: 1,
                mrope_sections: vec![],
                index_heads: 1,
                index_dim: 1,
                index_budget: 1,
                index_compression: 1,
                ple_layers: vec![],
                ngram_size: 1,
                ngram_heads: 1,
                ple_dim: 1,
                ple_conv: 1,
                ngram_parts: 1,
                mtp_layers: 0,
                block: [128, 128],
            };
            let index = safetensors::Index::open(
                &self.0.join(INDEX),
                checkpoint_limits(config.format, config.mixed.is_some()),
            )
            .unwrap();
            let shards = index
                .inspect_shards(&self.0, checkpoint_limits(config.format, config.mixed.is_some()))
                .unwrap();
            Checkpoint {
                config,
                index,
                shards,
                manifest: Default::default(),
            }
        }
        fn manifest(&self, legacy: bool) -> PathBuf {
            let mut names = vec![INDEX, "a.safetensors"];
            if legacy { names.extend(["config.json", "hf_quant_config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]); }
            let text = names.iter().map(|name| {
                    format!("{}  {name}\n", sha256::digest(&fs::read(self.0.join(name)).unwrap()))
                }).collect::<String>();
            let path = self.0.join("manifest.sha256"); fs::write(&path, text).unwrap(); path
        }
    }
    impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

    #[test]
    fn mixed_receipt_without_sidecar_accepts_real_scale_csf_index() {
        use std::fmt::Write;
        let f = Fixture::new();
        fs::remove_file(f.0.join("hf_quant_config.json")).unwrap();
        let config = r#"{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,"ignore":[],"quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf"},"model.language_model.layers.0.self_attn.q_proj":{"quant_algo":"MXFP8","group_size":32}}}}"#;
        fs::write(f.0.join("config.json"), config).unwrap();
        assert_eq!(super::super::nvfp4::detect(&f.0).unwrap(), WeightFormat::Nvfp4);
        let shard = "tensors/sub-dir_1/model-00001-of-00021.safetensors";
        fs::write(f.0.join(shard), []).unwrap();
        let mut index = String::from("{\"metadata\":{},\"weight_map\":{");
        let suffixes = ["weight", "input_scale", "weight_scale_2", "weight_scale.nvfp4_csf_fixed", "weight_scale.nvfp4_csf_exceptions"];
        for n in 0..375_487 {
            let (module, suffix) = (n / 5, suffixes[n % 5]);
            let projection = ["gate_proj", "up_proj", "down_proj"][module % 3];
            write!(index, "{}\n  \"model.language_model.layers.{}.mlp.experts.{}.{projection}.{suffix}\": \"{shard}\"",
                if n == 0 { "" } else { "," }, module / (512 * 3), module / 3 % 512).unwrap();
        }
        index.push_str("}}");
        assert!(
            ((40 << 20)..(64 << 20)).contains(&index.len()),
            "{}",
            index.len()
        );
        assert!(safetensors::Index::parse(index.as_bytes(), checkpoint_limits(WeightFormat::Fp8, false)).is_err());
        assert_eq!(
            safetensors::Index::parse_mixed(index.as_bytes(), checkpoint_limits(WeightFormat::Nvfp4, true))
                .unwrap()
                .weight_map
                .len(),
            375_487
        );
        fs::write(f.0.join(INDEX), index).unwrap();
        let plan = ReceiptPlan::open(&f.0).unwrap();
        assert_eq!(plan.format, WeightFormat::Nvfp4);
        assert_eq!(plan.hashes.len(), 2);
        fs::write(f.0.join("config.json"), config.replace("\"group_size\":16", "\"group_size\":32")).unwrap();
        assert!(ReceiptPlan::open(&f.0).is_err());
        fs::write(f.0.join("config.json"), "{}").unwrap();
        assert_eq!(super::super::nvfp4::detect(&f.0).unwrap(), WeightFormat::Fp8);
    }

    #[test]
    fn mixed_optional_manifest_and_legacy_required_contract() {
        let f = Fixture::new();
        let mixed = f.checkpoint(true);
        let legacy = f.checkpoint(false);
        let path = f.0.join("manifest.sha256");
        fs::remove_file(f.0.join("hf_quant_config.json")).unwrap();
        let v = verify(&mixed, &f.0, &path).unwrap();
        assert_eq!((v.files, v.hashed_bytes), (1, 0));
        assert!(v.manifest_sha256.is_empty());
        assert!(verify(&legacy, &f.0, &path).is_err());
        f.manifest(false);
        assert!(verify(&mixed, &f.0, &path).unwrap().hashed_bytes > 0);
        assert!(verify(&legacy, &f.0, &path).is_err());
        fs::write(f.0.join("hf_quant_config.json"), b"{}").unwrap();
        f.manifest(true);
        assert_eq!(verify(&legacy, &f.0, &path).unwrap().files, 8);
        assert!(verify(&mixed, &f.0, &path).is_ok());
        fs::write(f.0.join("a.safetensors"), b"corrupt").unwrap();
        assert!(verify(&mixed, &f.0, &path).is_err());
        assert!(verify(&legacy, &f.0, &path).is_err());
        fs::remove_file(path).unwrap();
        assert!(verify(&mixed, &f.0, &f.0.join("manifest.sha256")).is_err());
    }

    #[test]
    fn manifest_paths_and_coverage() {
        let f = Fixture::new(); let c = f.checkpoint(true); let path = f.manifest(false);
        let valid = fs::read_to_string(&path).unwrap();
        for name in ["model part.safetensors", "模型.safetensors", "model+copy.safetensors"] {
            fs::write(&path, format!("{}  {name}\n", "0".repeat(64))).unwrap();
            assert!(read_manifest(&path, false).is_ok());
        }
        for name in ["../a", "/a", "a/../b", "a//b", "a/./b", "a\\b", "a b", "é", ""] {
            fs::write(&path, format!("{}  {name}\n", "0".repeat(64))).unwrap();
            assert!(read_manifest(&path, true).is_err(), "accepted {name:?}");
        }
        fs::write(&path, format!("{valid}{valid}")).unwrap(); assert!(read_manifest(&path, true).is_err());
        fs::write(&path, valid.lines().next().unwrap()).unwrap(); assert!(verify(&c, &f.0, &path).is_err());
        fs::write(&path, valid.replace(&sha256::digest(&fs::read(f.0.join(INDEX)).unwrap()), &"0".repeat(64))).unwrap();
        assert!(verify(&c, &f.0, &path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn install_verification_reuses_hashes_and_refuses_same_size_changed_bytes() {
        use std::os::unix::fs::PermissionsExt;
        let f = Fixture::new();
        let checkpoint = f.checkpoint(true);
        let manifest = f.manifest(false);
        verify_cached(&checkpoint, &f.0, &manifest).unwrap();
        let file = f.0.join("a.safetensors");
        let expected = sha256::digest(&fs::read(&file).unwrap());
        let cache = f.0.join("private-cache");
        fs::create_dir(&cache).unwrap();
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o700)).unwrap();
        let root = f.0.canonicalize().unwrap();
        assert!(!hash_file(&root, "a.safetensors", &expected, Some(&cache), false, true).unwrap().2);
        assert!(hash_file(&root, "a.safetensors", &expected, Some(&cache), false, true).unwrap().2);
        let before = fs::metadata(&file).unwrap();
        let mut changed = fs::read(&file).unwrap();
        *changed.last_mut().unwrap() ^= 1;
        fs::write(&file, changed).unwrap();
        File::options().write(true).open(&file).unwrap().set_times(
            fs::FileTimes::new().set_modified(before.modified().unwrap())).unwrap();
        assert!(hash_file(&root, "a.safetensors", &expected, Some(&cache), false, true).is_err());
        assert!(verify_cached(&checkpoint, &f.0, &manifest).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn nested_symlink_escape_rejected_by_manifests_and_metadata_maps() {
        let f = Fixture::new(); let outside = Fixture::new(); let c = f.checkpoint(true);
        let path = f.manifest(false);
        std::os::unix::fs::symlink(&outside.0, f.0.join("escape")).unwrap();
        let mut text = fs::read_to_string(&path).unwrap();
        text.push_str(&format!("{}  escape/config.json\n", sha256::digest(b"{}")));
        fs::write(&path, text).unwrap(); assert!(verify(&c, &f.0, &path).is_err());
        assert!(safetensors::validate_model_path(&f.0, "escape/missing.safetensors").is_err());
        assert!(safetensors::validate_model_path(&f.0, "tensors/sub-dir_1/a.safetensors").is_ok());
        assert!(safetensors::validate_model_path(&f.0, "tensors/missing/a.safetensors").is_ok());
        std::os::unix::fs::symlink(f.0.join("tensors"), f.0.join("inside")).unwrap();
        assert!(safetensors::validate_model_path(&f.0, "inside/sub-dir_1/a.safetensors").is_ok());
    }
}
