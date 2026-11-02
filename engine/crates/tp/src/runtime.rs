use crate::Result;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

static HEAP: AtomicUsize = AtomicUsize::new(0);
static HEAP_PEAK: AtomicUsize = AtomicUsize::new(0);
pub struct TrackedAllocator;
fn acquired(bytes: usize) {
    HEAP_PEAK.fetch_max(HEAP.fetch_add(bytes, Relaxed) + bytes, Relaxed);
}
unsafe impl GlobalAlloc for TrackedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            acquired(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            acquired(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(p, layout);
        }
        HEAP.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(p, layout, new_size) };
        if !out.is_null() {
            if new_size > layout.size() {
                acquired(new_size - layout.size());
            } else {
                HEAP.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        out
    }
}
pub fn reset_heap_peak() {
    HEAP_PEAK.store(HEAP.load(Relaxed), Relaxed);
}
pub fn heap_peak() -> usize {
    HEAP_PEAK.load(Relaxed)
}

/// Optional affinity for this thread only; no host-wide settings are changed.
pub fn pin() -> Result<()> {
    static WATCH: std::sync::Once = std::sync::Once::new();
    WATCH.call_once(|| {
        if std::env::var("TP_WATCH_STDIN").as_deref() == Ok("1") {
            std::thread::spawn(|| {
                use std::io::Read;
                let mut bytes = [0u8; 16];
                loop {
                    match std::io::stdin().read(&mut bytes) {
                        Ok(0) => break,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
                        Err(_) => break,
                        Ok(_) => (),
                    }
                }
                // Rank zero needs time to flush terminal HTTP errors after a
                // launcher closes stdin in response to another rank failing.
                std::thread::sleep(std::time::Duration::from_secs(1));
                std::process::exit(1);
            });
        }
        if let Ok(deadline) = std::env::var("TP_DEADLINE_UNIX")
            .unwrap_or_default()
            .parse::<u64>()
        {
            std::thread::spawn(move || {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                std::thread::sleep(std::time::Duration::from_secs(deadline.saturating_sub(now)));
                eprintln!("rank lease expired");
                std::process::exit(1);
            });
        }
    });
    if let Ok(cpu) = std::env::var("TP_CPU") {
        let cpu = cpu.parse::<usize>().map_err(|e| e.to_string())?;
        pin_cpu(cpu)?;
    }
    Ok(())
}
pub fn pin_cpu(cpu: usize) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn sched_setaffinity(pid: i32, bytes: usize, mask: *const u64) -> i32;
        }
        let mut mask = [0u64; 16];
        if cpu >= mask.len() * 64 {
            return Err("CPU index outside affinity mask".into());
        }
        mask[cpu / 64] = 1 << (cpu % 64);
        if unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) } != 0 {
            return Err(format!(
                "thread affinity: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
    Ok(())
}

pub fn run_hash() -> u64 {
    let args = std::env::args().collect::<Vec<_>>();
    let program = args
        .first()
        .and_then(|p| std::path::Path::new(p).file_name())
        .and_then(|p| p.to_str())
        .unwrap_or_default();
    let auxiliary = match (
        std::env::var_os("TP_AUX_LISTEN"),
        std::env::var_os("TP_AUX_NEXT"),
    ) {
        (Some(_), Some(_)) => "aux",
        (None, None) => "single",
        _ => "invalid",
    };
    let mut hash = 0xcbf29ce484222325u64;
    // Rank and neighbour endpoints vary; names, arguments and settings agree.
    // verify_artifacts separately compares the actual execution contents.
    for value in [program, auxiliary]
        .into_iter()
        .chain(args.iter().skip(5).map(String::as_str))
        .chain(
            [
                "TP_RUN_ID",
                "TP_TOPOLOGY",
                "TP_SERVE_HYBRID",
                "TP_SERVE_SEQUENCE_PREFILL",
                "TP_SERVE_GENERATION_TIMEOUT_MS",
                "SPARK_KV_BF16",
                "TP_RDMA",
                "TP_RDMA_SMALL",
                "SEQPAR_ATTENTION_WEIGHT",
                "SEQPAR_OVERLAP",
                "SEQPAR_SERIAL_COMPUTE",
                "SEQPAR_RDMA",
                "SPARK_ADAPTIVE_PREFILL",
            ]
            .map(|k| std::env::var(k).unwrap_or_default())
            .iter()
            .map(String::as_str),
        )
    {
        for b in (value.len() as u64)
            .to_le_bytes()
            .into_iter()
            .chain(value.bytes())
        {
            hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

pub fn host_memory() -> (usize, usize) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let bytes = |key: &str| {
        status
            .lines()
            .find_map(|line| {
                line.strip_prefix(key)?
                    .split_whitespace()
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .unwrap_or(0)
            * 1024
    };
    (bytes("VmRSS:"), bytes("VmHWM:"))
}

#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;

#[path = "pipeline_settings.rs"]
mod pipeline_settings;

fn pipeline_identity_names(format: u8) -> Vec<&'static str> {
    let mut names = identity_names(format);
    names.extend(pipeline_settings::names());
    names
}

// Keep legacy sidecar identity separate from the mixed loader's config contract.
fn identity_names(format: u8) -> Vec<&'static str> {
    let mut names = vec![
        "executable",
        "loaded PTX",
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
        "tokenizer_config.json",
    ];
    if format == 1 {
        names.push("hf_quant_config.json");
    }
    if format != 0 {
        names.push(MANIFEST_ROW);
    }
    names
}

const MANIFEST_ROW: &str = "NVFP4 SHA-256 manifest";

/// The checksum list of this rank: the path a start names, else the checkpoint's own file.
fn manifest_path(root: &std::path::Path) -> std::path::PathBuf {
    std::env::var_os("FLASH_NVFP4_MANIFEST")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("manifest.sha256"))
}

fn artifact_format(root: &std::path::Path) -> Result<u8> {
    let nvfp4 = engine_model::cartridge::Kind::detect(root).ok()
        == Some(engine_model::cartridge::Kind::FlashNext)
        && engine_model::flash::nvfp4::detect(root).map_err(|e| e.to_string())?
            == engine_model::flash::WeightFormat::Nvfp4;
    let mixed = nvfp4
        && engine_model::flash::Config::open(&root.join("config.json"))
            .map_err(|e| e.to_string())?
            .mixed
            .is_some();
    Ok(if mixed { 2 } else { u8::from(nvfp4) })
}

/// Transport-independent identity of the actual PTX buffers and checkpoint metadata.
/// Optional-file absence is encoded explicitly, exactly as on the native rank path.
fn artifact_fingerprints(
    ptx: &std::ffi::CStr,
    root: &std::path::Path,
    ab_ptx: Option<&std::ffi::CStr>,
    format: u8,
) -> Result<Vec<u8>> {
    use std::{fs::File, io::ErrorKind};
    let manifest = manifest_path(root);
    let names = identity_names(format);
    let mut local = vec![0u8; names.len() * 64];
    // /proc/self/exe pins the running inode even if its pathname was replaced.
    #[cfg(target_os = "linux")]
    let executable = std::path::PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let executable = std::env::current_exe().map_err(|e| format!("TP executable identity: {e}"))?;
    for (i, name) in names.iter().enumerate() {
        let digest = if i == 1 {
            let a = sha256::digest(ptx.to_bytes());
            if let Some(x) = ab_ptx {
                sha256::digest(format!("VADE_AB_PTX:{a}:{}", sha256::digest(x.to_bytes())).as_bytes())
            } else { a }
        } else {
            let path = if i == 0 {
                executable.clone()
            } else if *name == MANIFEST_ROW {
                manifest.clone()
            } else {
                root.join(name)
            };
            match File::open(path) {
                Ok(file) => sha256::digest_reader(file)
                    .map_err(|e| format!("TP identity read {name}: {e}"))?,
                // Unsharded checkpoints have no index; optional config absence
                // is an explicit all-zero marker, distinct from any SHA-256.
                Err(e)
                    if ((3..6).contains(&i) || format == 2 && *name == MANIFEST_ROW)
                        && e.kind() == ErrorKind::NotFound =>
                {
                    continue;
                }
                Err(e) => return Err(format!("TP identity open {name}: {e}")),
            }
        };
        local[i * 64..(i + 1) * 64].copy_from_slice(digest.as_bytes());
    }
    Ok(local)
}

/// Refuse a pipeline link before either rank loads weights or serves requests.
/// The caller passes the same PTX buffers it will load, not paths to reread.
/// Like native rank verification, this compares manifests, not tensor payloads.
/// Two different manifests refuse. If only one rank has a manifest, that is a
/// difference in history, not in content: `share_manifest` sets it on both.
/// `prefix_restore`: the link is a tensor split, where an unset prefix cache means on.
pub fn verify_pipeline_artifacts(
    stream: &mut std::net::TcpStream,
    ptx: &std::ffi::CStr,
    root: &std::path::Path,
    ab_ptx: Option<&std::ffi::CStr>,
    experiments: spark_serve::mtp_batch::Experiments,
    prefix_restore: bool,
    limit: std::time::Duration,
) -> Result<()> {
    let format = artifact_format(root)?;
    let mut local = artifact_fingerprints(ptx, root, ab_ptx, format)?;
    local.extend(pipeline_settings::fingerprints(experiments, prefix_restore, |name| std::env::var_os(name))?);
    exchange_pipeline_identity(stream, format, &local, &manifest_path(root), limit)?;
    eprintln!("pipeline content identity passed (executable, loaded PTX, checkpoint metadata, numerical settings; weight payloads not hashed)");
    Ok(())
}

fn exchange_pipeline_identity(
    stream: &mut std::net::TcpStream,
    format: u8,
    local: &[u8],
    manifest: &std::path::Path,
    limit: std::time::Duration,
) -> Result<()> {
    use std::io::{Read, Write};
    let io = |e| format!("pipeline artifact handshake: {e}; refusing startup");
    stream.set_read_timeout(Some(limit)).map_err(io)?;
    stream.set_write_timeout(Some(limit)).map_err(io)?;
    // Fixed, versioned header; agree on the layout before exchanging bounded rows.
    let mut header = *b"VADART03\0\0\0\0\0\0\0\0";
    header[8] = format;
    header[12..].copy_from_slice(&(local.len() as u32).to_le_bytes());
    stream.write_all(&header).map_err(io)?;
    let mut peer_header = [0u8; 16];
    stream.read_exact(&mut peer_header).map_err(io)?;
    if peer_header[..8] != header[..8] || peer_header[9..12] != header[9..12] {
        return Err("pipeline artifact protocol version differs between ranks; refusing startup".into());
    }
    if peer_header[8..] != header[8..] {
        return Err("pipeline checkpoint formats differ between ranks; refusing startup".into());
    }
    stream.write_all(local).map_err(io)?;
    let mut peer = vec![0u8; local.len()];
    stream.read_exact(&mut peer).map_err(io)?;
    let unset = |row: &[u8]| row.iter().all(|byte| *byte == 0);
    let mut one_sided = None;
    for (i, name) in pipeline_identity_names(format).iter().enumerate() {
        let span = i * 64..(i + 1) * 64;
        if local[span.clone()] != peer[span.clone()] {
            // Every other row must still agree before any manifest moves.
            if format == 2 && *name == MANIFEST_ROW && unset(&local[span.clone()]) != unset(&peer[span.clone()]) {
                one_sided = Some(span);
                continue;
            }
            let local_value = pipeline_settings::describe(name, &local[span.clone()]);
            let peer_value = pipeline_settings::describe(name, &peer[span]);
            return Err(format!("pipeline content identity mismatch: {name} local={local_value}, peer={peer_value}; refusing startup"));
        }
    }
    match one_sided {
        Some(span) if unset(&local[span.clone()]) => take_manifest(stream, manifest, &peer[span]),
        Some(span) => share_manifest(stream, manifest, &local[span]),
        None => Ok(()),
    }
}

/// An earlier start can leave a manifest on one rank only: rank 0 of a start
/// with no manifest path writes its receipt into the checkpoint, and a rank
/// can hold a publisher's list that the other rank does not. Such a rank
/// sends its manifest, so both ranks start with the same one. The other rank
/// hashes its own weight files against it during the load and refuses on a
/// difference, so stale rows stop that rank; they never pass as a result.
/// The rank that sends waits for the reply, so a refusal comes before any load.
fn share_manifest(stream: &mut std::net::TcpStream, manifest: &std::path::Path, digest: &[u8]) -> Result<()> {
    use std::io::{Read, Write};
    let io = |e| format!("pipeline manifest handoff: {e}; refusing startup");
    let bytes = engine_model::safetensors::read_bounded(manifest, engine_model::flash::manifest::MANIFEST_BYTES)
        .map_err(|e| format!("pipeline manifest handoff: {e}; refusing startup"))?;
    if sha256::digest(&bytes).as_bytes() != digest {
        return Err("pipeline manifest changed during the identity exchange; refusing startup".into());
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes()).map_err(io)?;
    stream.write_all(&bytes).map_err(io)?;
    let mut saved = [0u8; 1];
    stream.read_exact(&mut saved).map_err(io)?;
    eprintln!("pipeline manifest set on the peer: {}", pipeline_settings::describe(MANIFEST_ROW, digest));
    Ok(())
}

/// The peer's side of `share_manifest`: the bytes must have the digest the
/// peer's row gave, and they go to this rank's own manifest path.
fn take_manifest(stream: &mut std::net::TcpStream, manifest: &std::path::Path, digest: &[u8]) -> Result<()> {
    use std::io::{Read, Write};
    let io = |e| format!("pipeline manifest handoff: {e}; refusing startup");
    let mut length = [0u8; 4];
    stream.read_exact(&mut length).map_err(io)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > engine_model::flash::manifest::MANIFEST_BYTES {
        return Err("pipeline manifest from the peer is too large; refusing startup".into());
    }
    let mut bytes = vec![0u8; length];
    stream.read_exact(&mut bytes).map_err(io)?;
    if sha256::digest(&bytes).as_bytes() != digest {
        return Err("pipeline manifest from the peer differs from its SHA-256; refusing startup".into());
    }
    engine_model::flash::manifest::adopt(manifest, &bytes)
        .map_err(|e| format!("pipeline manifest from the peer cannot be saved: {e}; refusing startup"))?;
    stream.write_all(&[1]).map_err(io)?;
    eprintln!("pipeline manifest set from the peer: {}; this rank checks its weight files against it during the load",
        pipeline_settings::describe(MANIFEST_ROW, digest));
    Ok(())
}

/// Compare artifacts before any rank loads weights. The PTX slice is the exact
/// buffer subsequently passed to CUDA; embedded layer PTX is covered by the exe.
/// Checkpoint files must remain immutable, as required by the model loader.
/// Metadata agreement does not validate the contents of tensor payloads.
pub fn verify_artifacts(
    ring: &mut crate::transport::Ring,
    ptx: &std::ffi::CStr,
    root: &std::path::Path,
) -> Result<()> {
    verify_artifacts_with_ab(ring, ptx, root, None)
}
pub fn verify_artifacts_with_ab(
    ring: &mut crate::transport::Ring,
    ptx: &std::ffi::CStr,
    root: &std::path::Path,
    ab_ptx: Option<&std::ffi::CStr>,
) -> Result<()> {
    if ring.world() == 1 {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let format = artifact_format(root)?;
    let names = identity_names(format);
    // Agree on layout as well as variable row width before exchanging bytes.
    let mut formats = vec![0u8; ring.world()];
    formats[ring.rank()] = format;
    ring.all_gather(&mut formats, 1)?;
    if formats.iter().any(|format| *format != formats[0]) {
        return Err("TP checkpoint formats differ between ranks; refusing startup".into());
    }
    let row = names.len() * 64;
    let mut fingerprints = vec![0u8; row * ring.world()];
    fingerprints[ring.rank() * row..(ring.rank() + 1) * row]
        .copy_from_slice(&artifact_fingerprints(ptx, root, ab_ptx, format)?);
    ring.all_gather(&mut fingerprints, row)?;
    for rank in 1..ring.world() {
        for (i, name) in names.iter().enumerate() {
            let offset = i * 64;
            if fingerprints[offset..offset + 64]
                != fingerprints[rank * row + offset..rank * row + offset + 64]
            {
                return Err(format!(
                    "TP content identity mismatch: {name} differs between ranks 0 and {rank}; refusing startup"
                ));
            }
        }
    }
    eprintln!(
        "TP content identity passed rank={} world={} (executable, loaded PTX, checkpoint metadata; weight payloads not hashed) elapsed_ms={:.3} hash_buffer_bytes=65536 identity_bytes={}",
        ring.rank(),
        ring.world(),
        started.elapsed().as_secs_f64() * 1000.,
        fingerprints.len()
    );
    Ok(())
}

#[cfg(test)]
mod identity_tests {
    use std::{io::{Read, Write}, net::{TcpListener, TcpStream}, path::Path, time::Duration};

    fn experiments() -> spark_serve::mtp_batch::Experiments {
        spark_serve::mtp_batch::Experiments::startup_with(2, true, |_| Ok(None)).unwrap()
    }

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let front = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (back, _) = listener.accept().unwrap();
        (front, back)
    }

    fn exchange(a_format: u8, a: Vec<u8>, b_format: u8, b: Vec<u8>) -> (super::Result<()>, super::Result<()>) {
        let (mut front, mut back) = pair();
        std::thread::scope(|scope| {
            let peer = scope.spawn(|| super::exchange_pipeline_identity(&mut back, b_format, &b, Path::new("unused"), Duration::from_secs(2)));
            let result = super::exchange_pipeline_identity(&mut front, a_format, &a, Path::new("unused"), Duration::from_secs(2));
            (result, peer.join().unwrap())
        })
    }

    #[test]
    fn pipeline_identity_agrees_and_refuses_each_changed_artifact_on_both_peers() {
        for format in 0..=2 {
            let names = super::pipeline_identity_names(format);
            let a = vec![b'a'; names.len() * 64];
            let (front, back) = exchange(format, a.clone(), format, a.clone());
            assert!(front.is_ok() && back.is_ok());
            for (i, name) in names.iter().enumerate() {
                let mut b = a.clone();
                b[i * 64] = b'b';
                let (front, back) = exchange(format, a.clone(), format, b);
                for error in [front.unwrap_err(), back.unwrap_err()] {
                    assert!(error.contains(name) && error.contains("refusing startup"), "{error}");
                }
            }
        }
    }

    #[test]
    fn pipeline_identity_refuses_formats_even_with_equal_row_width() {
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let (front, back) = exchange(a, vec![0; 512], b, vec![0; 512]);
            assert!(front.unwrap_err().contains("formats differ"));
            assert!(back.unwrap_err().contains("formats differ"));
        }
    }

    #[test]
    fn pipeline_identity_refuses_numerical_settings_with_identical_artifacts() {
        let names = super::pipeline_settings::names();
        assert_eq!(names.len(), names.iter().collect::<std::collections::BTreeSet<_>>().len());
        for setting in ["VADE_QSA_TC", "VADE_GDN_CHUNK", "VADE_FLASHNEXT_MTP_DEPTH",
            "VADE_VERIFY_MX4", "VADE_FLASHNEXT_DRAFT_FP4", "VADE_MTP_SAMPLED"] {
            let mut a = vec![b'a'; super::identity_names(2).len() * 64];
            let mut b = a.clone();
            a.extend(super::pipeline_settings::fingerprints(experiments(), true, |_| None).unwrap());
            b.extend(super::pipeline_settings::fingerprints(experiments(), true, |name|
                (name == setting).then(|| "0".into())).unwrap());
            let (front, back) = exchange(2, a, 2, b);
            for error in [front.unwrap_err(), back.unwrap_err()] {
                assert!(error.contains(setting) && error.contains("refusing startup"), "{error}");
            }
        }
    }

    #[test]
    fn pipeline_identity_compares_parsed_prompt_values_with_identical_builds() {
        for setting in spark_engine::flash_prompt_options::Options::NAMES {
            let identity = |value: Option<&str>| {
                let mut bytes = vec![b'a'; super::identity_names(2).len() * 64];
                bytes.extend(super::pipeline_settings::fingerprints(experiments(), true, |name| {
                    if name == setting { value.map(Into::into) }
                    // Both recipes must remain valid when FRONTEND_GPU is off.
                    else if setting == "VADE_PROMPT_FRONTEND_GPU" && name == "VADE_PROMPT_FRONTEND_DEDUP" { Some("0".into()) }
                    else { None }
                }).unwrap());
                bytes
            };
            for (left, right) in [(None, Some("0")), (Some("1"), Some("0"))] {
                let (front, back) = exchange(2, identity(left), 2, identity(right));
                let front = front.unwrap_err();
                let back = back.unwrap_err();
                assert!(front.contains(setting) && front.contains("local=1 (on), peer=0 (off)") && front.contains("refusing startup"), "{front}");
                assert!(back.contains(setting) && back.contains("local=0 (off), peer=1 (on)") && back.contains("refusing startup"), "{back}");
            }
            for (left, right) in [(None, Some("1")), (None, None), (Some("0"), Some("0"))] {
                let (front, back) = exchange(2, identity(left), 2, identity(right));
                assert!(front.is_ok() && back.is_ok());
            }
        }
    }

    #[test]
    fn pipeline_identity_compares_parsed_cohort_recipe_on_both_peers() {
        let identity = |values: &[(&str, &str)]| {
            let parsed = spark_serve::mtp_batch::Experiments::startup_with(2, true, |name|
                Ok(values.iter().find(|(key, _)| *key == name).map(|(_, value)| value.to_string()))).unwrap();
            let mut bytes = vec![b'a'; super::identity_names(2).len() * 64];
            bytes.extend(super::pipeline_settings::fingerprints(parsed, true, |_| None).unwrap());
            bytes
        };
        for setting in ["VADE_MTP_C8", "VADE_MTP_COHORT_ADAPT", "VADE_C8_GUARD_ROBUST"] {
            let (front, back) = exchange(2, identity(&[]), 2, identity(&[(setting, "1")]));
            assert!(front.is_ok() && back.is_ok(), "{setting}: {front:?}, {back:?}");
        }
        let (front, back) = exchange(2, identity(&[]), 2, identity(&[("VADE_POOL_GRAPH_FALLBACK", "0")]));
        assert!(front.is_ok() && back.is_ok());
        for (setting, expected) in [
            ("VADE_MTP_C8", "VADE_MTP_BATCH/VADE_MTP_C8 (effective)"),
            ("VADE_MTP_BATCH", "VADE_MTP_BATCH/VADE_MTP_C8 (effective)"),
            ("VADE_MTP_COHORT_ADAPT", "VADE_MTP_COHORT_ADAPT"),
            ("VADE_POOL_GRAPH_FALLBACK", "VADE_POOL_GRAPH_FALLBACK"),
            ("VADE_C8_GUARD_ROBUST", "VADE_C8_GUARD_ROBUST")] {
            let (front, back) = exchange(2, identity(&[(setting, "1")]), 2, identity(&[(setting, "0")]));
            let (front, back) = (front.unwrap_err(), back.unwrap_err());
            for error in [&front, &back] {
                assert!(error.contains(expected) && error.contains("refusing startup"), "{error}");
                assert!(!error.contains("sha256:"), "parsed setting should be readable: {error}");
            }
        }
    }

    #[test]
    fn pipeline_identity_accepts_configured_recipe_with_back_frontend_suppressed() {
        use spark_engine::flash_prompt_options::Options;
        let configured = Options::parse(|_| Ok(None)).unwrap();
        // NextModel::disable_prompt_frontend sets this role flag to false only
        // after verification. Exercise the actual workspace reservation policy
        // with both roles: configured ON, front allocates, back allocates none.
        assert!(configured.packed_workspace_bytes(8192, true) > 0);
        assert_eq!(configured.packed_workspace_bytes(8192, false), 0);
        let mut front = vec![b'a'; super::identity_names(2).len() * 64];
        let mut back = front.clone();
        front.extend(super::pipeline_settings::fingerprints(experiments(), true, |_| None).unwrap());
        back.extend(super::pipeline_settings::fingerprints(experiments(), true, |name|
            Options::NAMES.contains(&name).then(|| "1".into())).unwrap());
        let (front, back) = exchange(2, front, 2, back);
        assert!(front.is_ok() && back.is_ok());
    }

    #[test]
    fn pipeline_identity_rejects_invalid_prompt_recipe_before_exchange() {
        for setting in spark_engine::flash_prompt_options::Options::NAMES {
            let error = super::pipeline_settings::fingerprints(experiments(), true, |name|
                (name == setting).then(|| "yes".into())).unwrap_err();
            assert!(error.contains(setting) && error.contains("must be 0 or 1"));
        }
        let error = super::pipeline_settings::fingerprints(experiments(), true, |name|
            (name == "VADE_PROMPT_FRONTEND_GPU").then(|| "0".into())).unwrap_err();
        assert!(error.contains("requires"));
    }

    #[test]
    fn pipeline_identity_stalled_peer_times_out_without_serving() {
        let (mut front, _back) = pair();
        let local = vec![0; super::pipeline_identity_names(2).len() * 64];
        let error = super::exchange_pipeline_identity(&mut front, 2, &local, Path::new("unused"), Duration::from_millis(30)).unwrap_err();
        assert!(error.contains("pipeline artifact handshake") && error.contains("refusing startup"), "{error}");
    }

    #[test]
    fn pipeline_identity_chain_admits_only_after_tail_agreement() {
        for ranks in 2..=4 {
            for mismatch in [false, true] {
                let mut edges: Vec<_> = (1..ranks).map(|_| {
                    let (near, far) = pair();
                    (Some(near), Some(far))
                }).collect();
                let mut workers = Vec::new();
                let local = vec![b'a'; super::pipeline_identity_names(2).len() * 64];
                let mut front = edges[0].0.take().unwrap();
                for rank in 1..ranks {
                    let mut incoming = edges[rank - 1].1.take().unwrap();
                    let outgoing = edges.get_mut(rank).and_then(|edge| edge.0.take());
                    let mut local = local.clone();
                    if mismatch && outgoing.is_none() { local[64] = b'b'; }
                    workers.push(std::thread::spawn(move || -> super::Result<()> {
                        // Same downstream-first ordering as run_back, before weight load.
                        if let Some(mut outgoing) = outgoing {
                            super::exchange_pipeline_identity(&mut outgoing, 2, &local, Path::new("unused"), Duration::from_secs(2))?;
                        }
                        super::exchange_pipeline_identity(&mut incoming, 2, &local, Path::new("unused"), Duration::from_secs(2))
                    }));
                }
                let front_result = super::exchange_pipeline_identity(&mut front, 2, &local, Path::new("unused"), Duration::from_secs(2));
                let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
                assert_eq!(front_result.is_err(), mismatch, "{ranks} ranks: {front_result:?}");
                for result in results { assert_eq!(result.is_err(), mismatch, "{ranks} ranks: {result:?}"); }
            }
        }
    }

    #[test]
    fn pipeline_identity_refuses_old_protocol_and_truncated_record() {
        for truncated in [false, true] {
            let (mut front, mut back) = pair();
            let peer = std::thread::spawn(move || {
                let mut header = [0u8; 16];
                back.read_exact(&mut header).unwrap();
                if !truncated { header[7] = b'0'; }
                back.write_all(&header).unwrap();
                // A matching header followed by EOF is never successful agreement.
            });
            let result = super::exchange_pipeline_identity(&mut front, 2, &vec![0; 448], Path::new("unused"), Duration::from_secs(2));
            assert!(result.unwrap_err().contains("refusing startup"));
            peer.join().unwrap();
        }
    }

    #[test]
    fn fingerprints_cover_actual_ptx_and_manifest_bytes() {
        let root = std::env::temp_dir().join(format!("pipeline-artifacts-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
        let _cleanup = Cleanup(root.clone());
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        std::fs::write(root.join("manifest.sha256"), b"checkpoint-a").unwrap();
        let a = super::artifact_fingerprints(c"ptx-a", &root, None, 2).unwrap();
        let ptx = super::artifact_fingerprints(c"ptx-b", &root, None, 2).unwrap();
        assert_ne!(&a[64..128], &ptx[64..128]);
        assert_eq!(&a[128..], &ptx[128..]);
        std::fs::write(root.join("manifest.sha256"), b"checkpoint-b").unwrap();
        let manifest = super::artifact_fingerprints(c"ptx-a", &root, None, 2).unwrap();
        assert_eq!(&a[..384], &manifest[..384]);
        assert_ne!(&a[384..], &manifest[384..]);
        for mut b in [ptx, manifest] {
            let mut a = a.clone();
            let settings = super::pipeline_settings::fingerprints(experiments(), true, |_| None).unwrap();
            a.extend_from_slice(&settings);
            b.extend_from_slice(&settings);
            let (front, back) = exchange(2, a, 2, b);
            assert!(front.is_err() && back.is_err());
        }
    }

    /// One rank's checkpoint folder of a chosen weight format, removed at the end of a test.
    struct Folder(std::path::PathBuf);
    impl Drop for Folder {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    impl std::ops::Deref for Folder {
        type Target = std::path::Path;
        fn deref(&self) -> &std::path::Path { &self.0 }
    }
    const FP8_QUANTIZATION: &str = r#"{"quant_method":"fp8","activation_scheme":"dynamic","weight_per_tensor":false,"act_per_tensor":false,"weight_block_size":[128,128]}"#;
    const MIXED_QUANTIZATION: &str = r#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,"ignore":[],"quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf"},"model.language_model.layers.0.self_attn.q_proj":{"quant_algo":"MXFP8","group_size":32}}}"#;
    fn checkpoint(quantization: &str, weights: &[u8]) -> Folder {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!("pipeline-checkpoint-{}-{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir(&root).unwrap();
        let folder = Folder(root.canonicalize().unwrap());
        let layers = (0..48).map(|i| if i % 4 == 3 { "\"full_attention\"" } else { "\"linear_attention\"" }).collect::<Vec<_>>().join(",");
        std::fs::write(folder.join("config.json"), format!(r#"{{"model_type":"qwen4_exp","text_config":{{
            "model_type":"qwen4_exp_text","hidden_act":"silu","output_gate_type":"sigmoid",
            "mamba_ssm_dtype":"float32","attention_bias":false,"tie_word_embeddings":false,
            "mtp_use_dedicated_embeddings":false,"hidden_size":2560,"num_hidden_layers":48,
            "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
            "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,
            "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_experts":512,
            "num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
            "hc_count":4,"hc_lowrank":320,"vocab_size":248320,"max_position_embeddings":262144,
            "eos_token_id":248044,"rms_norm_eps":0.000001,"partial_rotary_factor":0.25,
            "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,
            "indexer_compress_ratio":4,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,
            "ple_embed_dim":2560,"ple_conv_kernel_size":4,"split_ngram_parts":128,
            "mtp_num_hidden_layers":1,"layer_types":[{layers}],
            "rope_parameters":{{"rope_type":"default","mrope_interleaved":true,"rope_theta":10000000,
                "partial_rotary_factor":0.25,"mrope_section":[11,11,10]}},
            "mtp":{{"num_hidden_layers":1,"layer_types":["full_attention"],"hybrid":true,
                "mtp_use_hidden_state_from_layer":null,"rope_theta":10000000}}
            }},"quantization_config":{quantization}}}"#)).unwrap();
        std::fs::write(folder.join("model.safetensors"), weights).unwrap();
        folder
    }

    /// What rank 0 does in the background of a start with no list path set, and rank 1 never does.
    fn rank_zero_receipt(root: &std::path::Path) {
        engine_model::flash::manifest::ReceiptPlan::open(root).unwrap().finish().unwrap();
    }

    /// Both ranks run the identity step of a start, as next_batched_service.rs calls it.
    fn start(rank0: &std::path::Path, rank1: &std::path::Path) -> (super::Result<()>, super::Result<()>) {
        let (mut front, mut back) = pair();
        std::thread::scope(|scope| {
            let peer = scope.spawn(|| super::verify_pipeline_artifacts(&mut back, c"ptx", rank1, None, experiments(), true, Duration::from_secs(5)));
            let result = super::verify_pipeline_artifacts(&mut front, c"ptx", rank0, None, experiments(), true, Duration::from_secs(5));
            (result, peer.join().unwrap())
        })
    }

    fn list(root: &std::path::Path) -> Option<Vec<u8>> { std::fs::read(root.join("manifest.sha256")).ok() }

    #[test]
    fn fp4_local_starts_after_an_fp8_run_with_the_list_on_both_ranks() {
        let (fp4, fp8) = ([checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4")],
            [checkpoint(FP8_QUANTIZATION, b"fp8"), checkpoint(FP8_QUANTIZATION, b"fp8")]);
        // CARD 1 of the rented pairs: a fresh pair, no list on either rank, the identity passes.
        let (front, back) = start(&fp4[0], &fp4[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert_eq!((list(&fp4[0]), list(&fp4[1])), (None, None));
        rank_zero_receipt(&fp4[0]);
        // The FP8 run: the ring's receipt goes in its own checkpoint, on rank 0 only.
        let (front, back) = start(&fp8[0], &fp8[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        rank_zero_receipt(&fp8[0]);
        // The later FP4-Local start.
        let (front, back) = start(&fp4[0], &fp4[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert!(list(&fp4[0]).is_some());
        assert_eq!(list(&fp4[0]), list(&fp4[1]));
    }

    #[test]
    fn a_second_fp4_local_start_needs_no_fp8_run_to_fail_or_to_pass() {
        let fp4 = [checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4")];
        rank_zero_receipt(&fp4[0]);
        let (front, back) = start(&fp4[0], &fp4[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert_eq!(list(&fp4[0]), list(&fp4[1]));
        // The same list on both ranks: the next start compares and sends nothing.
        let (front, back) = start(&fp4[0], &fp4[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
    }

    #[test]
    fn rank_zero_takes_the_list_when_only_rank_one_holds_it() {
        let fp4 = [checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4")];
        rank_zero_receipt(&fp4[1]);
        let (front, back) = start(&fp4[0], &fp4[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert!(list(&fp4[1]).is_some());
        assert_eq!(list(&fp4[0]), list(&fp4[1]));
    }

    #[test]
    fn fp8_starts_after_fp4_local_and_reads_no_list() {
        let (fp4, fp8) = ([checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4")],
            [checkpoint(FP8_QUANTIZATION, b"fp8"), checkpoint(FP8_QUANTIZATION, b"fp8")]);
        rank_zero_receipt(&fp4[0]);
        // One rank holds an FP8 receipt: no row reads it, so nothing moves and the start passes.
        rank_zero_receipt(&fp8[0]);
        let (front, back) = start(&fp8[0], &fp8[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert_eq!(list(&fp8[1]), None);
        // Also from a checkpoint folder with a stale list that is not the FP8 receipt.
        std::fs::write(fp8[1].join("manifest.sha256"), b"not a list").unwrap();
        let (front, back) = start(&fp8[0], &fp8[1]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert_eq!(list(&fp4[1]), None);
    }

    /// Rows of one rank's start with a chosen manifest row, for the cases a real list cannot reach.
    fn rows(manifest: Option<&[u8]>) -> Vec<u8> {
        let names = super::pipeline_identity_names(2);
        let mut local = vec![b'a'; names.len() * 64];
        let at = names.iter().position(|name| *name == super::MANIFEST_ROW).unwrap();
        local[at * 64..(at + 1) * 64].copy_from_slice(&manifest.map_or([0; 64], |bytes| {
            super::sha256::digest(bytes).into_bytes().try_into().unwrap()
        }));
        local
    }

    fn link(front: (&[u8], &Path), back: (&[u8], &Path)) -> (super::Result<()>, super::Result<()>) {
        let (mut near, far) = pair();
        std::thread::scope(|scope| {
            let peer = scope.spawn(move || { let mut far = far; super::exchange_pipeline_identity(&mut far, 2, back.0, back.1, Duration::from_secs(5)) });
            let result = super::exchange_pipeline_identity(&mut near, 2, front.0, front.1, Duration::from_secs(5));
            (result, peer.join().unwrap())
        })
    }

    fn valid_list(name: &str) -> Vec<u8> { format!("{}  {name}\n", "a".repeat(64)).into_bytes() }

    #[test]
    fn a_rank_with_a_named_list_path_takes_the_list_there_and_leaves_the_checkpoint_alone() {
        // The vade route: the start names a list in the state folder of each rank.
        let (rank0, rank1) = (checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4"));
        let list = valid_list("model.safetensors");
        let (sent, taken) = (rank0.join("state-list"), rank1.join("state-list"));
        std::fs::write(&sent, &list).unwrap();
        let (front, back) = link((&rows(Some(&list)), &sent), (&rows(None), &taken));
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        assert_eq!(std::fs::read(&taken).unwrap(), list);
        assert!(!rank1.join("manifest.sha256").exists());
    }

    #[test]
    fn two_different_lists_and_any_other_row_still_refuse_and_nothing_moves() {
        let (rank0, rank1) = (checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4"));
        let (one, other) = (valid_list("model.safetensors"), valid_list("other.safetensors"));
        let (sent, taken) = (rank0.join("list"), rank1.join("list"));
        std::fs::write(&sent, &one).unwrap();
        std::fs::write(&taken, &other).unwrap();
        let (front, back) = link((&rows(Some(&one)), &sent), (&rows(Some(&other)), &taken));
        assert!(front.unwrap_err().contains("mismatch: NVFP4 SHA-256 manifest local=sha256:"));
        assert!(back.unwrap_err().contains("mismatch: NVFP4 SHA-256 manifest local=sha256:"));
        assert_eq!(std::fs::read(&taken).unwrap(), other);
        // One rank holds the list, and a config row differs: that row refuses, no list moves.
        std::fs::remove_file(&taken).unwrap();
        let (mut changed, unset) = (rows(None), rows(Some(&one)));
        changed[2 * 64] = b'b';
        let (front, back) = link((&unset, &sent), (&changed, &taken));
        assert!(front.unwrap_err().contains("mismatch: config.json"));
        assert!(back.unwrap_err().contains("mismatch: config.json"));
        assert!(!taken.exists());
    }

    #[test]
    fn a_list_that_fails_a_check_stops_both_ranks_before_any_load() {
        let (rank0, rank1) = (checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4"));
        let sent = rank0.join("list");
        // Not rows of a list: the taking rank refuses, and the sending rank sees no reply.
        let bad = b"not a list".to_vec();
        std::fs::write(&sent, &bad).unwrap();
        let taken = rank1.join("list");
        let (front, back) = link((&rows(Some(&bad)), &sent), (&rows(None), &taken));
        assert!(front.unwrap_err().contains("pipeline manifest handoff"));
        assert!(back.unwrap_err().contains("cannot be saved"));
        assert!(!taken.exists());
        // A folder that does not exist has the same result.
        let list = valid_list("model.safetensors");
        std::fs::write(&sent, &list).unwrap();
        let (front, back) = link((&rows(Some(&list)), &sent), (&rows(None), &rank1.join("no-folder/list")));
        assert!(front.unwrap_err().contains("pipeline manifest handoff"));
        assert!(back.unwrap_err().contains("cannot be saved"));
        // The file on disk no longer has the digest of the row.
        let (front, back) = link((&rows(Some(b"before")), &sent), (&rows(None), &taken));
        assert!(front.unwrap_err().contains("changed during the identity exchange"));
        assert!(back.is_err());
        assert!(!taken.exists());
    }

    #[test]
    fn a_taking_rank_refuses_bytes_that_do_not_have_the_digest_of_the_row() {
        let rank1 = checkpoint(MIXED_QUANTIZATION, b"fp4");
        let (taken, list) = (rank1.join("list"), valid_list("model.safetensors"));
        let row = super::sha256::digest(&list).into_bytes();
        for (length, body) in [(list.len() as u32, valid_list("other.safetensors")), (1 << 21, Vec::new())] {
            let (mut near, mut far) = pair();
            near.write_all(&length.to_le_bytes()).unwrap();
            near.write_all(&body).unwrap();
            far.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let error = super::take_manifest(&mut far, &taken, &row).unwrap_err();
            assert!(error.contains("differs from its SHA-256") || error.contains("too large"), "{error}");
            assert!(!taken.exists());
        }
    }

    #[test]
    fn a_chain_moves_a_list_over_each_link_and_two_lists_meet_on_a_link() {
        // Tail link first, as run_back does. The front holds the list: the next rank takes it.
        let ranks = [checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4"), checkpoint(MIXED_QUANTIZATION, b"fp4")];
        rank_zero_receipt(&ranks[0]);
        for (a, b) in [(1, 2), (0, 1)] {
            let (front, back) = start(&ranks[a], &ranks[b]);
            assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        }
        assert_eq!(list(&ranks[0]), list(&ranks[1]));
        assert_eq!(list(&ranks[2]), None);
        // The next start reaches the next rank.
        for (a, b) in [(1, 2), (0, 1)] {
            let (front, back) = start(&ranks[a], &ranks[b]);
            assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        }
        assert_eq!(list(&ranks[0]), list(&ranks[2]));
        // The tail holds another list: its link gives it to the middle rank, and the front refuses it.
        let ranks = [checkpoint(MIXED_QUANTIZATION, b"one"), checkpoint(MIXED_QUANTIZATION, b"one"), checkpoint(MIXED_QUANTIZATION, b"two")];
        rank_zero_receipt(&ranks[0]);
        rank_zero_receipt(&ranks[2]);
        let (front, back) = start(&ranks[1], &ranks[2]);
        assert!(front.is_ok() && back.is_ok(), "{front:?} {back:?}");
        let (front, back) = start(&ranks[0], &ranks[1]);
        assert!(front.unwrap_err().contains("mismatch: NVFP4 SHA-256 manifest"));
        assert!(back.unwrap_err().contains("mismatch: NVFP4 SHA-256 manifest"));
    }

    #[test]
    fn mixed_layout_identity_needs_no_legacy_sidecar() {
        let names = super::identity_names(2);
        assert!(!names.contains(&"hf_quant_config.json"));
        assert!(names.contains(&"config.json") && names.contains(&"model.safetensors.index.json"));
        assert!(names.contains(&"NVFP4 SHA-256 manifest"));
        assert!(super::identity_names(1).contains(&"hf_quant_config.json"));
        assert!(!super::identity_names(0).contains(&"NVFP4 SHA-256 manifest"));
    }
}
