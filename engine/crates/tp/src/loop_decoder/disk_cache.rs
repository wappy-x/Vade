//! Persistent, 100% bit-exact NVMe disk prefix cache for `loop_decoder` on `2 MiB` VMM KV pages.
//!
//! Design properties:
//! - Strict identity binding: checkpoint `Identity::fingerprint()`, engine/format version tag,
//!   exact target visit layouts (ring/straight/window/taps/dims), optional draft model layout,
//!   authorization/salt scope, exact position, prompt SHA-256, and exact token-by-token verification.
//! - Integrity: whole-manifest SHA-256 (prelude + header + prompt tokens + logits + chunk CRC table)
//!   plus per-chunk IEEE CRC-32 across every byte of the streamed tensor payload.
//! - Crash safety: writes to `<entry>.tmp.<pid>.<seq>`, `fsync`, atomic `rename`, directory `fsync`.
//! - Direct NVMe streaming: multi-worker Linux `O_DIRECT` (`FileExt::read_at` / sequential `4 MiB`
//!   direct writes) overlaps NVMe queue depth, slicing-by-8 CRC-32, LE conversion, and sub-chunk
//!   GPU transfers with near-zero unified-RAM page-cache footprint (transparent buffered fallback
//!   on non-Linux hosts or filesystems without `O_DIRECT`).
//! - Bounded LRU eviction: hardware-derived default budget (50% of free filesystem space via POSIX
//!   `df -kP`, or explicit `--disk-cache-bytes`), zero arbitrary token-length caps (Rule 17).
//! - Zero `unsafe`, zero third-party crates, zero hot-path overhead on RAM hits.
use crate::{
    Result,
    bounded_decoder::Compute,
    loop_decoder::{
        draft::DraftChat,
        state::{self, Layout, State},
    },
};
use engine_model::mtp_draft::MtpConfig;
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[allow(clippy::duplicate_mod)] // shared stdlib SHA-256 file included intentionally across crates.
#[path = "../../../../tools/export/src/sha256.rs"]
mod sha256;
mod crc32;
mod direct_io;
mod format;
#[cfg(test)]
mod tests;

use direct_io::{AlignedBuf, DirectReader, DirectSequentialWriter};
use format::{
    CHUNK_BYTES, CHUNK_WORDS, DraftDesc, ENGINE_TAG, FLAG_HAS_DRAFT, FOOTER_FIXED_BYTES,
    FOOTER_MAGIC, PRELUDE_BYTES, Prelude, VisitDesc, encode_header, expected_payload_bytes,
    manifest_sha256, prefix_64_hash, prompt_sha256, read_and_verify_envelope, scope_sha256, to_hex,
};

/// Sub-chunk upload granularity (`1 MiB` = 1/4 of the `4 MiB` pinned CUDA upload staging ring)
/// so host-to-pinned memcpy overlaps with asynchronous GPU H2D + D2D copies without growing staging.
const UPLOAD_SUBCHUNK_WORDS: usize = CHUNK_WORDS / 4;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Restored prefix payload from NVMe, ready to open into an active `Chat`.
pub(super) struct DiskRestored<V> {
    pub state: State<V>,
    pub draft: Option<DraftChat<V>>,
    pub tokens: Vec<u32>,
    pub next_logits: Vec<f32>,
}

#[derive(Debug, Clone)]
struct DiskEntry {
    path: PathBuf,
    file_bytes: u64,
    ino: u64,
    lru_seq: u64,
    mtime_ns: u128,
    scope: Option<String>,
    position: usize,
    prompt_sha256: [u8; 32],
    first_token: u32,
    last_token: u32,
    prefix_64_hash: u64,
    has_logits: bool,
    has_draft: bool,
}

#[derive(Debug, Clone)]
struct ForeignFile {
    path: PathBuf,
    file_bytes: u64,
    ino: u64,
    lru_seq: u64,
    mtime_ns: u128,
}

/// Persistent NVMe prefix cache directory manager for one loaded loop-decoder checkpoint.
pub struct DiskPrefixCache {
    dir: PathBuf,
    budget_bytes: u64,
    checkpoint_fingerprint: String,
    vocab_rows: u32,
    context: usize,
    visits: Vec<VisitDesc>,
    draft_desc: Option<DraftDesc>,
    entries: Vec<DiskEntry>,
    foreign: Vec<ForeignFile>,
    total_bytes: u64,
    lru_clock: u64,
    last_dir_mtime: Option<SystemTime>,
}

struct TempFileGuard(Option<PathBuf>);
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn system_time_ns(t: SystemTime) -> u128 {
    t.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
}

fn file_mtime_ns(path: &Path) -> u128 {
    fs::metadata(path).and_then(|m| m.modified()).map(system_time_ns).unwrap_or(0)
}

fn file_ino(meta: &fs::Metadata) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = meta;
        0
    }
}

fn file_meta_stats(path: &Path) -> (u128, u64) {
    fs::metadata(path)
        .map(|m| (m.modified().map(system_time_ns).unwrap_or(0), file_ino(&m)))
        .unwrap_or((0, 0))
}

fn remove_file_if_same_ino(path: &Path, expected_ino: u64) {
    #[cfg(unix)]
    if expected_ino != 0 && fs::metadata(path).ok().map(|m| file_ino(&m)) != Some(expected_ino) {
        return;
    }
    let _ = fs::remove_file(path);
}

/// Hardware-derived default disk budget (50% of free filesystem space + existing cache files) via
/// POSIX `df -kP`, requiring zero `unsafe` syscalls and imposing zero arbitrary caps (Rule 17).
pub fn resolve_budget_bytes(dir: &Path, explicit: Option<u64>, existing_cache_bytes: u64) -> Result<u64> {
    if let Some(bytes) = explicit {
        if bytes == 0 {
            return Err("disk prefix cache budget must be positive".into());
        }
        return Ok(bytes);
    }
    let avail = filesystem_available_bytes(dir)?;
    let pool = avail.saturating_add(existing_cache_bytes);
    let half = pool / 2;
    if half == 0 {
        return Err(format!("filesystem hosting {} has no free space for disk prefix cache", dir.display()));
    }
    Ok(half)
}

fn filesystem_available_bytes(dir: &Path) -> Result<u64> {
    let out = Command::new("df")
        .arg("-kP")
        .arg(dir)
        .output()
        .map_err(|e| format!("failed to query free disk space for {}: {e}", dir.display()))?;
    if !out.status.success() {
        return Err(format!("df -kP {} exited with {}", dir.display(), out.status));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .skip(1)
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("df -kP {} returned no data line", dir.display()))?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(format!("unexpected df -kP output: {line}"));
    }
    // POSIX `df -kP`: Filesystem 1024-blocks Used Available Capacity Mounted-on.
    // Parse from the right relative to Capacity (`N%`) so filesystem names with spaces stay safe.
    let cap_idx = fields
        .iter()
        .rposition(|f| f.ends_with('%'))
        .ok_or_else(|| format!("df -kP output missing capacity column: {line}"))?;
    let avail_kib: u64 = fields
        .get(cap_idx.checked_sub(1).ok_or("df -kP output missing available column")?)
        .ok_or("df -kP output missing available column")?
        .parse()
        .map_err(|_| format!("invalid available KiB in df -kP output: {line}"))?;
    Ok(avail_kib.saturating_mul(1024))
}

fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}

fn touch_mtime(path: &Path) {
    if let Ok(file) = OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

fn is_stale_temp(name: &str, path: &Path) -> bool {
    let Some((_, tail)) = name.split_once(".tmp.") else {
        return false;
    };
    let pid_str = tail.split('.').next().unwrap_or("");
    let Ok(pid) = pid_str.parse::<u32>() else {
        return true;
    };
    if pid == std::process::id() {
        return true;
    }
    #[cfg(target_os = "linux")]
    {
        let _ = path;
        !Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .is_some_and(|age| age.as_secs() > 3600)
    }
}

fn save_worker_count(expected_chunks: usize) -> usize {
    let cpus = std::thread::available_parallelism().map_or(2, |n| n.get());
    cpus.clamp(1, 8).min(expected_chunks.max(1))
}

fn read_worker_count(chunk_bytes: usize, chunk_count: usize) -> usize {
    let cpus = std::thread::available_parallelism().map_or(2, |n| n.get());
    // Keep total host read+decode scratch bounded (~192 MiB) even for larger custom chunk sizes.
    let per_worker = chunk_bytes.max(1).saturating_mul(3);
    let max_by_mem = (192 * 1024 * 1024 / per_worker).max(1);
    cpus.clamp(1, 12).min(max_by_mem).min(chunk_count.max(1))
}

impl DiskPrefixCache {
    /// Open or create a persistent disk prefix cache directory bound to the loaded checkpoint and layouts.
    pub fn open(
        dir: PathBuf,
        explicit_budget_bytes: Option<u64>,
        checkpoint_fingerprint: String,
        layouts: &[Layout],
        vocab_rows: usize,
        context: usize,
        draft_config: Option<&MtpConfig>,
    ) -> Result<Self> {
        Self::open_with_kv_bf16(dir, explicit_budget_bytes, checkpoint_fingerprint, layouts, vocab_rows, context, draft_config, false)
    }

    /// Open or create a persistent disk prefix cache directory with explicit draft KV BF16 packing mode.
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_kv_bf16(
        dir: PathBuf,
        explicit_budget_bytes: Option<u64>,
        checkpoint_fingerprint: String,
        layouts: &[Layout],
        vocab_rows: usize,
        context: usize,
        draft_config: Option<&MtpConfig>,
        draft_kv_bf16: bool,
    ) -> Result<Self> {
        fs::create_dir_all(&dir).map_err(|e| format!("cannot create disk prefix cache dir {}: {e}", dir.display()))?;
        let vocab_rows = u32::try_from(vocab_rows).map_err(|_| "vocab_rows exceeds u32")?;
        let visits = layouts.iter().map(VisitDesc::from_layout).collect::<Result<Vec<_>>>()?;
        let draft_desc = draft_config.map(|cfg| DraftDesc::from_config(cfg, draft_kv_bf16)).transpose()?;
        let mut cache = Self {
            dir,
            budget_bytes: u64::MAX,
            checkpoint_fingerprint,
            vocab_rows,
            context,
            visits,
            draft_desc,
            entries: Vec::new(),
            foreign: Vec::new(),
            total_bytes: 0,
            lru_clock: 0,
            last_dir_mtime: None,
        };
        cache.scan_dir(true)?;
        cache.budget_bytes = resolve_budget_bytes(&cache.dir, explicit_budget_bytes, cache.total_bytes)?;
        cache.evict_to_budget(0, None);
        eprintln!(
            "disk_prefix_cache ready dir={} budget_bytes={} entries={} foreign_files={} disk_bytes={}",
            cache.dir.display(),
            cache.budget_bytes,
            cache.entries.len(),
            cache.foreign.len(),
            cache.total_bytes
        );
        Ok(cache)
    }

    pub fn dir(&self) -> &Path { &self.dir }
    pub fn budget_bytes(&self) -> u64 { self.budget_bytes }
    pub fn total_bytes(&self) -> u64 { self.total_bytes }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    fn entry_filename(&self, scope: Option<&str>, position: usize, prompt_hash: &[u8; 32]) -> String {
        let ckpt = &self.checkpoint_fingerprint[..self.checkpoint_fingerprint.len().min(16)];
        let scope_hex = to_hex(&scope_sha256(scope)[..8]);
        let prompt_hex = to_hex(&prompt_hash[..16]);
        format!("{ckpt}_{scope_hex}_{position:010}_{prompt_hex}.vdpc")
    }

    fn next_lru(&mut self) -> u64 {
        self.lru_clock = self.lru_clock.saturating_add(1);
        self.lru_clock
    }

    /// Incrementally synchronize in-memory directory index with on-disk `.vdpc` files.
    ///
    /// Reuses already-parsed metadata when `(path, file_bytes, ino)` is unchanged (avoiding
    /// multi-MB header re-reads and SHA-256 passes on every lookup), discovers files written by
    /// peer replicas without relying on coarse directory `mtime` ticks, drops peer-evicted files,
    /// and promotes LRU sequence numbers when peer hits advance file `mtime`.
    fn scan_dir(&mut self, clean_temps: bool) -> Result<()> {
        let dir_mtime = fs::metadata(&self.dir).and_then(|m| m.modified()).ok();
        let initial_scan = self.lru_clock == 0;
        let mut old_entries: HashMap<PathBuf, DiskEntry> =
            self.entries.drain(..).map(|e| (e.path.clone(), e)).collect();
        let mut old_foreign: HashMap<PathBuf, ForeignFile> =
            self.foreign.drain(..).map(|f| (f.path.clone(), f)).collect();
        let mut scanned_entries = Vec::new();
        let mut scanned_foreign = Vec::new();
        let mut lru_updates: Vec<(u128, bool, usize)> = Vec::new();
        let mut total_bytes = 0u64;
        let read_dir = fs::read_dir(&self.dir)
            .map_err(|e| format!("cannot read disk cache dir {}: {e}", self.dir.display()))?;
        for item in read_dir {
            let Ok(entry) = item else { continue; };
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.contains(".tmp.") {
                if clean_temps && is_stale_temp(&name, &path) {
                    let _ = fs::remove_file(&path);
                }
                continue;
            }
            if !name.ends_with(".vdpc") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue; };
            if !meta.is_file() {
                continue;
            }
            let file_bytes = meta.len();
            let mtime_ns = meta.modified().map(system_time_ns).unwrap_or(0);
            let ino = file_ino(&meta);

            // Fast O(1) reuse for already-verified inodes (zero file open, zero SHA-256).
            if let Some(mut existing) = old_entries.remove(&path) {
                if existing.file_bytes == file_bytes && existing.ino == ino && ino != 0 {
                    let touched = mtime_ns > existing.mtime_ns;
                    existing.mtime_ns = mtime_ns;
                    total_bytes = total_bytes.saturating_add(file_bytes);
                    let idx = scanned_entries.len();
                    scanned_entries.push(existing);
                    if initial_scan || touched {
                        lru_updates.push((mtime_ns, true, idx));
                    }
                    continue;
                }
            } else if let Some(mut existing) = old_foreign.remove(&path)
                && existing.file_bytes == file_bytes
                && existing.ino == ino
                && ino != 0
            {
                let touched = mtime_ns > existing.mtime_ns;
                existing.mtime_ns = mtime_ns;
                total_bytes = total_bytes.saturating_add(file_bytes);
                let idx = scanned_foreign.len();
                scanned_foreign.push(existing);
                if initial_scan || touched {
                    lru_updates.push((mtime_ns, false, idx));
                }
                continue;
            }

            let mut file = match File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            // Re-stat the opened fd so size, mtime, and inode match the exact verified file.
            let Ok(open_meta) = file.metadata() else { continue; };
            if !open_meta.is_file() {
                continue;
            }
            let file_bytes = open_meta.len();
            let mtime_ns = open_meta.modified().map(system_time_ns).unwrap_or(0);
            let ino = file_ino(&open_meta);
            match read_and_verify_envelope(&mut file, false) {
                Ok((_prelude, header, _crcs)) => {
                    let compatible = header.engine_tag == ENGINE_TAG
                        && header.checkpoint_fingerprint == self.checkpoint_fingerprint
                        && header.vocab_rows == self.vocab_rows
                        && header.visits == self.visits
                        && (header.position as usize) <= self.context
                        && match (&header.draft, &self.draft_desc) {
                            (Some(d), Some(mine)) => d == mine,
                            (Some(_), None) => true, // Target state still usable when restarted with --mtp 0.
                            (None, _) => true,
                        };
                    total_bytes = total_bytes.saturating_add(file_bytes);
                    if compatible {
                        let idx = scanned_entries.len();
                        scanned_entries.push(DiskEntry {
                            path,
                            file_bytes,
                            ino,
                            lru_seq: 0,
                            mtime_ns,
                            scope: header.scope,
                            position: header.position as usize,
                            prompt_sha256: header.prompt_sha256,
                            first_token: header.first_token,
                            last_token: header.last_token,
                            prefix_64_hash: header.prefix_64_hash,
                            has_logits: header.logits_len > 0,
                            has_draft: header.draft.is_some(),
                        });
                        lru_updates.push((mtime_ns, true, idx));
                    } else {
                        let idx = scanned_foreign.len();
                        scanned_foreign.push(ForeignFile { path, file_bytes, ino, lru_seq: 0, mtime_ns });
                        lru_updates.push((mtime_ns, false, idx));
                    }
                }
                Err(reason) => {
                    eprintln!("disk_prefix_cache_corrupt path={} reason={reason} action=removed_on_scan", path.display());
                    remove_file_if_same_ino(&path, ino);
                }
            }
        }
        // Assign monotonic LRU sequence numbers (oldest mtime first) for newly discovered or
        // peer-touched entries while preserving relative order of untouched local entries.
        lru_updates.sort_by_key(|&(m, _, _)| m);
        for (_, is_entry, idx) in lru_updates {
            let seq = self.next_lru();
            if is_entry {
                scanned_entries[idx].lru_seq = seq;
            } else {
                scanned_foreign[idx].lru_seq = seq;
            }
        }
        self.entries = scanned_entries;
        self.foreign = scanned_foreign;
        self.total_bytes = total_bytes;
        self.last_dir_mtime = dir_mtime;
        Ok(())
    }

    fn sync_from_disk(&mut self) {
        let _ = self.scan_dir(false);
    }

    fn remove_entry_by_path(&mut self, path: &Path, delete_ino: Option<u64>) {
        if let Some(idx) = self.entries.iter().position(|e| e.path == path) {
            let removed = self.entries.swap_remove(idx);
            self.total_bytes = self.total_bytes.saturating_sub(removed.file_bytes);
        } else if let Some(idx) = self.foreign.iter().position(|f| f.path == path) {
            let removed = self.foreign.swap_remove(idx);
            self.total_bytes = self.total_bytes.saturating_sub(removed.file_bytes);
        }
        if let Some(ino) = delete_ino {
            remove_file_if_same_ino(path, ino);
            self.last_dir_mtime = fs::metadata(&self.dir).and_then(|m| m.modified()).ok();
        }
    }

    /// Evict oldest entries (across both compatible and foreign `.vdpc` files) until
    /// `total_bytes + incoming_bytes <= budget_bytes`. Never evicts `protect`.
    fn evict_to_budget(&mut self, incoming_bytes: u64, protect: Option<&Path>) {
        let mut evicted_any = false;
        while self.total_bytes.saturating_add(incoming_bytes) > self.budget_bytes {
            let oldest_entry = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| protect.is_none_or(|p| e.path != p))
                .min_by_key(|(_, e)| (e.lru_seq, e.mtime_ns))
                .map(|(i, e)| (e.lru_seq, e.mtime_ns, true, i));
            let oldest_foreign = self
                .foreign
                .iter()
                .enumerate()
                .filter(|(_, f)| protect.is_none_or(|p| f.path != p))
                .min_by_key(|(_, f)| (f.lru_seq, f.mtime_ns))
                .map(|(i, f)| (f.lru_seq, f.mtime_ns, false, i));
            let victim = match (oldest_entry, oldest_foreign) {
                (Some(a), Some(b)) => Some(if (a.0, a.1) <= (b.0, b.1) { a } else { b }),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            let Some((_, _, is_entry, idx)) = victim else { break; };
            let (path, bytes, ino) = if is_entry {
                let e = self.entries.swap_remove(idx);
                (e.path, e.file_bytes, e.ino)
            } else {
                let f = self.foreign.swap_remove(idx);
                (f.path, f.file_bytes, f.ino)
            };
            self.total_bytes = self.total_bytes.saturating_sub(bytes);
            remove_file_if_same_ino(&path, ino);
            evicted_any = true;
            eprintln!(
                "disk_prefix_cache_evicted path={} freed_bytes={} disk_bytes={} budget_bytes={}",
                path.display(),
                bytes,
                self.total_bytes,
                self.budget_bytes
            );
        }
        if evicted_any {
            self.last_dir_mtime = fs::metadata(&self.dir).and_then(|m| m.modified()).ok();
        }
    }

    /// Persist a committed target `State` + optional `DraftChat` + prompt tokens + logits atomically to NVMe.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn save<C: Compute>(
        &mut self,
        compute: &C,
        layouts: &[Layout],
        state: &State<C::Value>,
        draft: Option<(&DraftChat<C::Value>, &MtpConfig)>,
        scope: Option<&str>,
        tokens: &[u32],
        next_logits: &[f32],
    ) -> Result<()> {
        if tokens.is_empty() || state.position as usize != tokens.len() || state.visits.len() != layouts.len() {
            return Err("disk cache save: state position or visit count mismatch".into());
        }
        if next_logits.len() > self.vocab_rows as usize || next_logits.iter().any(|v| !v.is_finite()) {
            return Err("disk cache save: invalid logits".into());
        }
        for visit in &state.visits {
            if visit.planes.is_some()
                || visit.next.iter().any(Option::is_some)
                || visit.every.iter().any(Option::is_some)
                || visit.written.is_some()
            {
                return Err("disk cache save: cannot persist unfinished step".into());
            }
        }
        // Synchronize with peer replicas before checking dedup or enforcing shared disk budget.
        self.sync_from_disk();

        let prompt_hash = prompt_sha256(tokens);
        let has_draft = draft.is_some();
        let final_path = self.dir.join(self.entry_filename(scope, tokens.len(), &prompt_hash));

        // Skip redundant disk write when an identical or richer (with draft) entry is already persisted.
        if let Some(idx) = self.entries.iter().position(|e| {
            e.scope.as_deref() == scope
                && e.position == tokens.len()
                && e.prompt_sha256 == prompt_hash
                && e.has_logits >= !next_logits.is_empty()
                && e.has_draft >= has_draft
        }) && self.entries[idx].path.exists()
        {
            let seq = self.next_lru();
            self.entries[idx].lru_seq = seq;
            touch_mtime(&self.entries[idx].path);
            self.entries[idx].mtime_ns = file_mtime_ns(&self.entries[idx].path);
            return Ok(());
        }

        let draft_desc = match draft {
            Some((chat, cfg)) => Some(DraftDesc::from_config(cfg, chat.is_kv_bf16())?),
            None => None,
        };
        let header_bytes = encode_header(
            &self.checkpoint_fingerprint,
            scope,
            self.vocab_rows,
            state.position,
            &prompt_hash,
            &self.visits,
            draft_desc.as_ref(),
            tokens,
            next_logits,
        )?;
        let payload_bytes = expected_payload_bytes(layouts, state.position, draft_desc.as_ref())?;
        let chunk_count = u32::try_from(payload_bytes.div_ceil(CHUNK_BYTES as u64))
            .map_err(|_| "disk cache chunk_count exceeds u32")?;
        let footer_bytes = u64::from(chunk_count)
            .checked_mul(4)
            .and_then(|n| n.checked_add(FOOTER_FIXED_BYTES))
            .ok_or("disk cache footer overflow")?;
        let total_file_bytes = PRELUDE_BYTES
            .checked_add(header_bytes.len() as u64)
            .and_then(|n| n.checked_add(payload_bytes))
            .and_then(|n| n.checked_add(footer_bytes))
            .ok_or("disk cache file size overflow")?;

        if total_file_bytes > self.budget_bytes {
            eprintln!(
                "disk_prefix_cache_skip reason=exceeds_budget tokens={} file_bytes={} budget_bytes={}",
                tokens.len(),
                total_file_bytes,
                self.budget_bytes
            );
            return Ok(());
        }

        let existing_same_bytes = self.entries.iter().find(|e| e.path == final_path).map_or(0, |e| e.file_bytes);
        let net_added = total_file_bytes.saturating_sub(existing_same_bytes);
        self.evict_to_budget(net_added, Some(&final_path));

        let prelude = Prelude {
            flags: if has_draft { FLAG_HAS_DRAFT } else { 0 },
            header_bytes: header_bytes.len() as u64,
            payload_bytes,
            chunk_bytes: CHUNK_BYTES as u32,
            chunk_count,
        };
        let prelude_raw = prelude.encode();

        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self.dir.join(format!(
            "{}.tmp.{}.{seq}",
            final_path.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        let mut tmp_guard = TempFileGuard(Some(tmp_path.clone()));
        let started = Instant::now();

        let mut writer = DirectSequentialWriter::create_new(&tmp_path)
            .map_err(|e| format!("cannot create disk cache temp file {}: {e}", tmp_path.display()))?;
        writer.write_all(&prelude_raw).map_err(|e| format!("write prelude failed: {e}"))?;
        writer.write_all(&header_bytes).map_err(|e| format!("write header failed: {e}"))?;

        let draft_export = match draft {
            Some((chat, cfg)) => Some(chat.export_parts(compute, cfg)?),
            None => None,
        };
        let (chunk_crcs, mut writer) = stream_gpu_to_file(
            compute,
            layouts,
            state,
            draft_export.as_ref(),
            writer,
            chunk_count as usize,
        )?;
        if chunk_crcs.len() != chunk_count as usize {
            return Err("disk cache save: chunk count mismatch".into());
        }
        let manifest = manifest_sha256(&prelude_raw, &header_bytes, &chunk_crcs);
        let mut footer = Vec::with_capacity(footer_bytes as usize);
        for crc in &chunk_crcs {
            footer.extend_from_slice(&crc.to_le_bytes());
        }
        footer.extend_from_slice(&manifest);
        footer.extend_from_slice(FOOTER_MAGIC);
        writer.write_all(&footer).map_err(|e| format!("write footer failed: {e}"))?;
        writer.finish().map_err(|e| format!("direct flush/fsync failed: {e}"))?;

        fs::rename(&tmp_path, &final_path)
            .map_err(|e| format!("atomic rename {} -> {} failed: {e}", tmp_path.display(), final_path.display()))?;
        tmp_guard.0 = None;
        sync_dir(&self.dir);

        self.remove_entry_by_path(&final_path, None);
        let (mtime_ns, ino) = file_meta_stats(&final_path);
        let lru_seq = self.next_lru();
        self.entries.push(DiskEntry {
            path: final_path.clone(),
            file_bytes: total_file_bytes,
            ino,
            lru_seq,
            mtime_ns,
            scope: scope.map(str::to_owned),
            position: tokens.len(),
            prompt_sha256: prompt_hash,
            first_token: tokens[0],
            last_token: *tokens.last().unwrap(),
            prefix_64_hash: prefix_64_hash(tokens),
            has_logits: !next_logits.is_empty(),
            has_draft,
        });
        self.total_bytes = self.total_bytes.saturating_add(total_file_bytes);
        // Re-sync after rename to account for any peer files written concurrently while streaming,
        // keep this newly written entry strictly newest, and enforce the shared budget.
        self.sync_from_disk();
        let newest_seq = self.next_lru();
        if let Some(entry) = self.entries.iter_mut().find(|e| e.path == final_path) {
            entry.lru_seq = newest_seq;
        }
        self.evict_to_budget(0, Some(&final_path));
        self.last_dir_mtime = fs::metadata(&self.dir).and_then(|m| m.modified()).ok();

        let elapsed = started.elapsed().as_secs_f64();
        let mib_s = if elapsed > 0.0 { (total_file_bytes as f64 / (1024.0 * 1024.0)) / elapsed } else { 0.0 };
        eprintln!(
            "disk_prefix_cache_saved tokens={} has_draft={has_draft} file_bytes={total_file_bytes} save_ms={:.3} throughput_mib_s={mib_s:.1} entries={} disk_bytes={} budget_bytes={}",
            tokens.len(),
            elapsed * 1000.0,
            self.entries.len(),
            self.total_bytes,
            self.budget_bytes
        );
        Ok(())
    }

    /// Find and restore the longest verified matching prefix from NVMe that strictly beats
    /// the best in-RAM prefix candidate (`spark_serve::disk_prefix_min_tokens()`).
    /// Any corrupt, truncated, or tampered entry is logged, removed, and skipped cleanly.
    pub(super) fn restore<C: Compute>(
        &mut self,
        compute: &C,
        layouts: &[Layout],
        draft_config: Option<&MtpConfig>,
        scope: Option<&str>,
        prompt: &[u32],
        target_tokens: usize,
    ) -> Result<Option<DiskRestored<C::Value>>> {
        let min_tokens = spark_serve::disk_prefix_min_tokens();
        if prompt.is_empty() || min_tokens >= prompt.len() {
            return Ok(None);
        }
        self.sync_from_disk();
        let p_first = prompt[0];
        let p_hash64 = prefix_64_hash(prompt);

        // Select candidates passing O(1) scope, length, boundary, and fast prefix filters.
        let mut candidates: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.scope.as_deref() == scope
                    && e.position > min_tokens
                    && e.position <= prompt.len()
                    && e.position <= self.context
                    && (e.position < prompt.len() || e.has_logits)
                    && e.first_token == p_first
                    && e.last_token == prompt[e.position - 1]
                    && (e.position < 64 || e.prefix_64_hash == p_hash64)
            })
            .map(|(i, _)| i)
            .collect();
        // Longest prefix first, preferring entries with draft state and newest LRU.
        candidates.sort_by(|&a, &b| {
            let (ea, eb) = (&self.entries[a], &self.entries[b]);
            eb.position
                .cmp(&ea.position)
                .then(eb.has_draft.cmp(&ea.has_draft))
                .then(eb.lru_seq.cmp(&ea.lru_seq))
        });

        // Snapshot candidate metadata in priority order so SHA-256 is computed lazily only until
        // the longest matching entry restores cleanly.
        let ordered_candidates: Vec<(PathBuf, usize, [u8; 32])> = candidates
            .into_iter()
            .map(|idx| {
                let e = &self.entries[idx];
                (e.path.clone(), e.position, e.prompt_sha256)
            })
            .collect();

        let mut sha_cache: Vec<(usize, [u8; 32])> = Vec::new();
        for (path, pos, expected) in ordered_candidates {
            let actual = match sha_cache.iter().find(|&&(l, _)| l == pos) {
                Some(&(_, h)) => h,
                None => {
                    let h = prompt_sha256(&prompt[..pos]);
                    sha_cache.push((pos, h));
                    h
                }
            };
            if actual != expected {
                continue;
            }
            let started = Instant::now();
            match self.try_load_file(compute, layouts, draft_config, scope, prompt, target_tokens, &path) {
                Ok(Some((state, draft, tokens, next_logits, file_bytes))) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    let restore_ms = elapsed * 1000.0;
                    let mib_s = if elapsed > 0.0 { (file_bytes as f64 / (1024.0 * 1024.0)) / elapsed } else { 0.0 };
                    touch_mtime(&path);
                    let seq = self.next_lru();
                    if let Some(entry) = self.entries.iter_mut().find(|e| e.path == path) {
                        entry.lru_seq = seq;
                        entry.mtime_ns = file_mtime_ns(&path);
                    }
                    eprintln!(
                        "disk_prefix_cache_hit prompt_tokens={} restored_tokens={} has_draft={} file_bytes={file_bytes} restore_ms={restore_ms:.3} throughput_mib_s={mib_s:.1}",
                        prompt.len(),
                        tokens.len(),
                        draft.is_some(),
                    );
                    return Ok(Some(DiskRestored { state, draft, tokens, next_logits }));
                }
                Ok(None) => {
                    // File vanished concurrently (e.g. evicted by another process) or incompatible.
                    self.remove_entry_by_path(&path, None);
                }
                Err(CorruptOrIo::Corrupt(reason, ino)) => {
                    eprintln!(
                        "disk_prefix_cache_corrupt path={} reason={reason} action=removed_and_miss",
                        path.display()
                    );
                    self.remove_entry_by_path(&path, Some(ino));
                }
                Err(CorruptOrIo::Backend(reason)) => {
                    return Err(reason);
                }
            }
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn try_load_file<C: Compute>(
        &self,
        compute: &C,
        layouts: &[Layout],
        draft_config: Option<&MtpConfig>,
        scope: Option<&str>,
        prompt: &[u32],
        target_tokens: usize,
        path: &Path,
    ) -> std::result::Result<Option<(State<C::Value>, Option<DraftChat<C::Value>>, Vec<u32>, Vec<f32>, u64)>, CorruptOrIo> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(CorruptOrIo::Corrupt(format!("open failed: {e}"), 0)),
        };
        let meta = file.metadata().map_err(|e| CorruptOrIo::Corrupt(format!("stat failed: {e}"), 0))?;
        let file_bytes = meta.len();
        let ino = file_ino(&meta);
        let (prelude, header, chunk_crcs) =
            read_and_verify_envelope(&mut file, true).map_err(|r| CorruptOrIo::Corrupt(r, ino))?;

        // Full strict compatibility and exact token-by-token verification (zero collision risk).
        if header.engine_tag != ENGINE_TAG
            || header.checkpoint_fingerprint != self.checkpoint_fingerprint
            || header.scope.as_deref() != scope
            || header.vocab_rows != self.vocab_rows
            || header.visits != self.visits
        {
            return Ok(None);
        }
        let pos = header.position as usize;
        if pos == 0 || pos > prompt.len() || pos > self.context || (pos == prompt.len() && header.next_logits.is_empty()) {
            return Ok(None);
        }
        if header.tokens.len() != pos || header.tokens.as_slice() != &prompt[..pos] {
            return Ok(None);
        }
        let expected_payload = expected_payload_bytes(layouts, header.position, header.draft.as_ref())
            .map_err(|r| CorruptOrIo::Corrupt(r, ino))?;
        if expected_payload != prelude.payload_bytes {
            return Err(CorruptOrIo::Corrupt(format!(
                "payload size mismatch: header={} expected={expected_payload}",
                prelude.payload_bytes
            ), ino));
        }
        let restore_draft_cfg = match (&header.draft, draft_config, &self.draft_desc) {
            (Some(file_draft), Some(cfg), Some(expected)) => {
                if file_draft != expected {
                    return Err(CorruptOrIo::Corrupt("draft layout mismatch".into(), ino));
                }
                Some((cfg, file_draft.kv_width as usize != cfg.kv_width()))
            }
            (Some(file_draft), Some(cfg), None) => {
                let expected = DraftDesc::from_config(cfg, false).map_err(CorruptOrIo::Backend)?;
                if file_draft != &expected {
                    return Err(CorruptOrIo::Corrupt("draft layout mismatch".into(), ino));
                }
                Some((cfg, false))
            }
            _ => None,
        };

        // Pre-allocate target State (2 MiB CUDA VMM straight planes) and optional DraftChat OUTSIDE `compute.work`.
        let mut state = state::alloc_for_restore(compute, layouts, header.position, target_tokens)
            .map_err(CorruptOrIo::Backend)?;
        let mut draft = match restore_draft_cfg {
            Some((cfg, kv_bf16)) => Some(
                DraftChat::alloc_for_restore(compute, cfg, kv_bf16, pos, target_tokens, self.context)
                    .map_err(CorruptOrIo::Backend)?,
            ),
            None => None,
        };

        // Stream Section 3 Tensor Payload via concurrent O_DIRECT read_at + CRC-32 + GPU upload.
        let payload_offset = PRELUDE_BYTES + prelude.header_bytes;
        let reader = DirectReader::from_file(file);
        stream_file_to_gpu(
            compute,
            layouts,
            &mut state,
            draft.as_mut().zip(restore_draft_cfg.map(|(cfg, _)| cfg)),
            &reader,
            payload_offset,
            prelude.payload_bytes,
            prelude.chunk_bytes as usize,
            &chunk_crcs,
            ino,
        )?;

        Ok(Some((state, draft, header.tokens, header.next_logits, file_bytes)))
    }
}

enum CorruptOrIo {
    Corrupt(String, u64),
    Backend(String),
}

/// Read a single tensor slice `[start_word .. start_word + count_words)` from `compute` in bounded
/// `CHUNK_WORDS` (`4 MiB`) pieces without growing pinned staging or mutating `v.known`.
fn read_tensor_slice<C: Compute>(
    compute: &C,
    v: &C::Value,
    start_word: usize,
    count_words: usize,
    mut on_words: impl FnMut(Vec<f32>) -> Result<()>,
) -> Result<()> {
    if count_words == 0 {
        return Ok(());
    }
    let total = compute.words(v);
    if start_word.checked_add(count_words).is_none_or(|end| end > total) {
        return Err("disk cache save: tensor slice exceeds source words".into());
    }
    let mut done = 0usize;
    while done < count_words {
        let take = (count_words - done).min(CHUNK_WORDS);
        let pos = start_word + done;
        let words = if pos == 0 && take < total {
            let view = compute.range(v, 0, take)?;
            let copy = compute.rows_append(None, 0, &view, 1)?;
            compute.read(&copy)?
        } else {
            let view = compute.range(v, pos, take)?;
            compute.read(&view)?
        };
        if words.len() != take {
            return Err(format!("disk cache save: expected {take} words, read {}", words.len()));
        }
        on_words(words)?;
        done += take;
    }
    Ok(())
}

/// Pipelined GPU -> parallel LE-encode + CRC-32 workers -> sequential O_DIRECT NVMe writer.
fn stream_gpu_to_file<C: Compute>(
    compute: &C,
    layouts: &[Layout],
    state: &State<C::Value>,
    draft: Option<&super::draft::DraftExport<'_, C::Value>>,
    mut writer_sink: DirectSequentialWriter,
    expected_chunks: usize,
) -> Result<(Vec<u32>, DirectSequentialWriter)> {
    let w = save_worker_count(expected_chunks);

    let mut in_txs = Vec::with_capacity(w);
    let mut in_rxs = Vec::with_capacity(w);
    let mut out_txs = Vec::with_capacity(w);
    let mut out_rxs = Vec::with_capacity(w);
    let mut worker_buf_txs = Vec::with_capacity(w);
    let mut worker_buf_rxs = Vec::with_capacity(w);
    for _ in 0..w {
        let (itx, irx) = mpsc::sync_channel::<(usize, Vec<f32>)>(2);
        let (otx, orx) = mpsc::sync_channel::<(usize, u32, Vec<u8>)>(2);
        let (btx, brx) = mpsc::sync_channel::<Vec<u8>>(2);
        for _ in 0..2 {
            let _ = btx.send(vec![0u8; CHUNK_BYTES]);
        }
        in_txs.push(itx);
        in_rxs.push(irx);
        out_txs.push(otx);
        out_rxs.push(orx);
        worker_buf_txs.push(btx);
        worker_buf_rxs.push(brx);
    }

    std::thread::scope(|scope| {
        for ((irx, otx), brx) in in_rxs.into_iter().zip(out_txs).zip(worker_buf_rxs) {
            scope.spawn(move || {
                while let Ok((idx, words)) = irx.recv() {
                    let Ok(mut buf) = brx.recv() else {
                        break;
                    };
                    buf.resize(words.len() * 4, 0);
                    for (dst, &val) in buf.as_chunks_mut::<4>().0.iter_mut().zip(&words) {
                        *dst = val.to_bits().to_le_bytes();
                    }
                    drop(words);
                    let crc = crc32::checksum(&buf);
                    if otx.send((idx, crc, buf)).is_err() {
                        break;
                    }
                }
            });
        }

        let writer = scope.spawn(move || -> Result<(Vec<u32>, DirectSequentialWriter)> {
            let mut crcs = Vec::with_capacity(expected_chunks);
            for idx in 0..expected_chunks {
                let Ok((got_idx, crc, buf)) = out_rxs[idx % w].recv() else {
                    return Err("disk cache writer stopped before receiving all chunks".into());
                };
                if got_idx != idx {
                    return Err("disk cache writer chunk order mismatch".into());
                }
                writer_sink
                    .write_all(&buf)
                    .map_err(|e| format!("disk cache payload write failed: {e}"))?;
                crcs.push(crc);
                let _ = worker_buf_txs[idx % w].send(buf);
            }
            Ok((crcs, writer_sink))
        });

        let gpu_read = compute.work(|| {
            let paused = compute.rows_unrecorded(true)?;
            let res = (|| -> Result<()> {
                let mut chunk_idx = 0usize;
                let mut cur: Vec<f32> = Vec::with_capacity(CHUNK_WORDS);

                let mut push_words = |words: Vec<f32>| -> Result<()> {
                    // Zero-copy fast path for full chunk-aligned tensor reads (99%+ of long-context KV).
                    if cur.is_empty() && words.len() == CHUNK_WORDS {
                        in_txs[chunk_idx % w]
                            .send((chunk_idx, words))
                            .map_err(|_| "disk cache hash worker stopped".to_owned())?;
                        chunk_idx += 1;
                        return Ok(());
                    }
                    let mut src = 0usize;
                    while src < words.len() {
                        let space = CHUNK_WORDS - cur.len();
                        let take = (words.len() - src).min(space);
                        cur.extend_from_slice(&words[src..src + take]);
                        src += take;
                        if cur.len() == CHUNK_WORDS {
                            let full = std::mem::replace(&mut cur, Vec::with_capacity(CHUNK_WORDS));
                            in_txs[chunk_idx % w]
                                .send((chunk_idx, full))
                                .map_err(|_| "disk cache hash worker stopped".to_owned())?;
                            chunk_idx += 1;
                        }
                    }
                    Ok(())
                };

                for (visit, layout) in state.visits.iter().zip(layouts) {
                    let plane_words = state::prefix_plane_words(layout, state.position);
                    read_tensor_slice(compute, &visit.keys, 0, plane_words, &mut push_words)?;
                    read_tensor_slice(compute, &visit.values, 0, plane_words, &mut push_words)?;
                    let kv_ring = layout.kv_taps * layout.kv_channels;
                    let after_ring = layout.taps * layout.channels;
                    read_tensor_slice(compute, &visit.rings[0], 0, kv_ring, &mut push_words)?;
                    read_tensor_slice(compute, &visit.rings[1], 0, kv_ring, &mut push_words)?;
                    read_tensor_slice(compute, &visit.rings[2], 0, after_ring, &mut push_words)?;
                    read_tensor_slice(compute, &visit.rings[3], 0, after_ring, &mut push_words)?;
                }
                if let Some(d) = draft {
                    for &(keys, vals, offset, words) in &d.layers {
                        read_tensor_slice(compute, keys, offset, words, &mut push_words)?;
                        read_tensor_slice(compute, vals, offset, words, &mut push_words)?;
                    }
                    if let Some(feat) = d.feature {
                        read_tensor_slice(compute, feat, 0, compute.words(feat), &mut push_words)?;
                    }
                }
                if !cur.is_empty() {
                    in_txs[chunk_idx % w]
                        .send((chunk_idx, cur))
                        .map_err(|_| "disk cache hash worker stopped on final chunk".to_owned())?;
                    chunk_idx += 1;
                }
                if chunk_idx != expected_chunks {
                    return Err(format!("disk cache save produced {chunk_idx} chunks, expected {expected_chunks}"));
                }
                Ok(())
            })();
            let resumed = compute.rows_unrecorded(paused);
            res.and(resumed.map(|_| ()))
        });

        drop(in_txs);
        let written = writer.join().map_err(|_| "disk cache writer thread panicked".to_owned())?;
        gpu_read?;
        written
    })
}

/// Pipelined concurrent O_DIRECT NVMe reader + CRC-32 + LE-to-f32 workers -> streaming GPU upload.
#[allow(clippy::too_many_arguments)]
fn stream_file_to_gpu<C: Compute>(
    compute: &C,
    layouts: &[Layout],
    state: &mut State<C::Value>,
    draft: Option<(&mut DraftChat<C::Value>, &MtpConfig)>,
    reader: &DirectReader,
    payload_offset: u64,
    payload_bytes: u64,
    chunk_bytes: usize,
    expected_crcs: &[u32],
    ino: u64,
) -> std::result::Result<(), CorruptOrIo> {
    let chunk_count = expected_crcs.len();
    let w = read_worker_count(chunk_bytes, chunk_count);

    let mut out_txs = Vec::with_capacity(w);
    let mut out_rxs = Vec::with_capacity(w);
    let mut f32_rec_txs = Vec::with_capacity(w);
    let mut f32_rec_rxs = Vec::with_capacity(w);
    for _ in 0..w {
        let (otx, orx) = mpsc::sync_channel::<std::result::Result<(usize, Vec<f32>), String>>(2);
        let (ftx, frx) = mpsc::sync_channel::<Vec<f32>>(2);
        for _ in 0..2 {
            let _ = ftx.send(Vec::with_capacity(chunk_bytes / 4));
        }
        out_txs.push(otx);
        out_rxs.push(orx);
        f32_rec_txs.push(ftx);
        f32_rec_rxs.push(frx);
    }

    // Collect mutable destination tensors and their exact word counts in Section 3 order.
    let mut targets: Vec<(&mut C::Value, usize)> = Vec::new();
    for (visit, layout) in state.visits.iter_mut().zip(layouts) {
        let plane_words = state::prefix_plane_words(layout, state.position);
        let kv_ring = layout.kv_taps * layout.kv_channels;
        let after_ring = layout.taps * layout.channels;
        targets.push((&mut visit.keys, plane_words));
        targets.push((&mut visit.values, plane_words));
        let [r0, r1, r2, r3] = &mut visit.rings;
        targets.push((r0, kv_ring));
        targets.push((r1, kv_ring));
        targets.push((r2, after_ring));
        targets.push((r3, after_ring));
    }
    if let Some((draft_chat, cfg)) = draft {
        let pos = state.position as usize;
        let kv_width = super::draft::kv_row_words(cfg, draft_chat.is_kv_bf16());
        let win = cfg.sliding_window_size.saturating_sub(1);
        let (kv_slots, feature_slot) = draft_chat.restore_targets_mut();
        for (i, slot) in kv_slots.iter_mut().enumerate() {
            let Some((keys, vals)) = slot.as_mut() else {
                return Err(CorruptOrIo::Backend("allocated draft restore missing KV slot".into()));
            };
            let rows = if cfg.is_local(i) { pos.min(win) } else { pos };
            let words = rows * kv_width;
            targets.push((keys, words));
            targets.push((vals, words));
        }
        if let Some(feat) = feature_slot {
            targets.push((feat, cfg.hidden_size));
        }
    }

    let io_buf_cap = DirectReader::buffer_capacity(chunk_bytes);

    std::thread::scope(|scope| {
        // Concurrent O_DIRECT read_at + in-cache CRC-32 + LE-to-f32 verification workers.
        // Each worker owns its reusable page-aligned read buffer and handles chunks
        // `worker_id, worker_id + w, ...` with zero mutexes and full NVMe queue depth.
        for (worker_id, (otx, frx)) in out_txs.into_iter().zip(f32_rec_rxs).enumerate() {
            scope.spawn(move || {
                let mut io_buf = AlignedBuf::new(io_buf_cap);
                let mut idx = worker_id;
                while idx < chunk_count {
                    let rel_off = (idx as u64) * (chunk_bytes as u64);
                    let take = payload_bytes.saturating_sub(rel_off).min(chunk_bytes as u64) as usize;
                    let chunk = match reader.read_exact_chunk(payload_offset + rel_off, take, &mut io_buf) {
                        Ok(slice) => slice,
                        Err(e) => {
                            let _ = otx.send(Err(format!("payload read failed at chunk {idx}: {e}")));
                            break;
                        }
                    };
                    let got = crc32::checksum(chunk);
                    let want = expected_crcs[idx];
                    if got != want {
                        let _ = otx.send(Err(format!(
                            "payload chunk {idx} crc32 mismatch: got={got:#010x} expected={want:#010x}"
                        )));
                        break;
                    }
                    let Ok(mut words) = frx.recv() else { break; };
                    words.resize(take / 4, 0.0);
                    for (dst, src) in words.iter_mut().zip(chunk.as_chunks::<4>().0) {
                        *dst = f32::from_bits(u32::from_le_bytes(*src));
                    }
                    if otx.send(Ok((idx, words))).is_err() {
                        break;
                    }
                    idx += w;
                }
            });
        }

        // Main GPU thread: consume verified `f32` chunks in order and stream into VMM planes via `state_fill`.
        let result = compute.work(|| -> Result<()> {
            let paused = compute.rows_unrecorded(true)?;
            let fill_res = (|| -> std::result::Result<(), CorruptOrIo> {
                let mut t_idx = 0usize;
                let mut t_off = 0usize;
                for idx in 0..chunk_count {
                    let msg = out_rxs[idx % w]
                        .recv()
                        .map_err(|_| CorruptOrIo::Corrupt("verification worker stopped early".into(), ino))?;
                    let (got_idx, words) = msg.map_err(|r| CorruptOrIo::Corrupt(r, ino))?;
                    if got_idx != idx {
                        return Err(CorruptOrIo::Corrupt("restore chunk order mismatch".into(), ino));
                    }
                    let mut pos = 0usize;
                    while pos < words.len() {
                        while t_idx < targets.len() && t_off >= targets[t_idx].1 {
                            t_idx += 1;
                            t_off = 0;
                        }
                        if t_idx >= targets.len() {
                            // Remaining verified bytes belong to a draft payload skipped on a `--mtp 0` server.
                            break;
                        }
                        let need_words = targets[t_idx].1 - t_off;
                        let take = (words.len() - pos).min(need_words).min(UPLOAD_SUBCHUNK_WORDS);
                        let uploaded = compute
                            .upload(&words[pos..pos + take])
                            .map_err(CorruptOrIo::Backend)?;
                        compute
                            .state_fill(targets[t_idx].0, t_off, &uploaded)
                            .map_err(CorruptOrIo::Backend)?;
                        pos += take;
                        t_off += take;
                    }
                    let _ = f32_rec_txs[idx % w].send(words);
                }
                while t_idx < targets.len() && t_off >= targets[t_idx].1 {
                    t_idx += 1;
                    t_off = 0;
                }
                if t_idx < targets.len() {
                    return Err(CorruptOrIo::Corrupt("payload ended before filling all target tensors".into(), ino));
                }
                Ok(())
            })();
            let resumed = compute.rows_unrecorded(paused);
            match (fill_res, resumed) {
                (Ok(()), Ok(_)) => Ok(()),
                (Err(CorruptOrIo::Corrupt(e, _)), _) => Err(format!("CORRUPT:{e}")),
                (Err(CorruptOrIo::Backend(e)), _) => Err(format!("BACKEND:{e}")),
                (Ok(()), Err(e)) => Err(format!("BACKEND:{e}")),
            }
        });

        // Explicitly drop channels before scope exit so workers unblock immediately on error.
        drop(out_rxs);
        drop(f32_rec_txs);

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                if let Some(msg) = e.strip_prefix("CORRUPT:") {
                    Err(CorruptOrIo::Corrupt(msg.to_owned(), ino))
                } else if let Some(msg) = e.strip_prefix("BACKEND:") {
                    Err(CorruptOrIo::Backend(msg.to_owned()))
                } else {
                    Err(CorruptOrIo::Backend(e))
                }
            }
        }
    })
}
