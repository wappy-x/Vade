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
        names.push("NVFP4 SHA-256 manifest");
    }
    names
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
    let manifest = std::env::var_os("FLASH_NVFP4_MANIFEST")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("manifest.sha256"));
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
            } else if *name == "NVFP4 SHA-256 manifest" {
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
                    if ((3..6).contains(&i) || format == 2 && *name == "NVFP4 SHA-256 manifest")
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
pub fn verify_pipeline_artifacts(
    stream: &mut std::net::TcpStream,
    ptx: &std::ffi::CStr,
    root: &std::path::Path,
    ab_ptx: Option<&std::ffi::CStr>,
    experiments: spark_serve::mtp_batch::Experiments,
    limit: std::time::Duration,
) -> Result<()> {
    let format = artifact_format(root)?;
    let mut local = artifact_fingerprints(ptx, root, ab_ptx, format)?;
    local.extend(pipeline_settings::fingerprints(experiments, |name| std::env::var_os(name))?);
    exchange_pipeline_identity(stream, format, &local, limit)?;
    eprintln!("pipeline content identity passed (executable, loaded PTX, checkpoint metadata, numerical settings; weight payloads not hashed)");
    Ok(())
}

fn exchange_pipeline_identity(
    stream: &mut std::net::TcpStream,
    format: u8,
    local: &[u8],
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
    for (i, name) in pipeline_identity_names(format).iter().enumerate() {
        let span = i * 64..(i + 1) * 64;
        if local[span.clone()] != peer[span.clone()] {
            let local_value = pipeline_settings::describe(name, &local[span.clone()]);
            let peer_value = pipeline_settings::describe(name, &peer[span]);
            return Err(format!("pipeline content identity mismatch: {name} local={local_value}, peer={peer_value}; refusing startup"));
        }
    }
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
    use std::{io::{Read, Write}, net::{TcpListener, TcpStream}, time::Duration};

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
            let peer = scope.spawn(|| super::exchange_pipeline_identity(&mut back, b_format, &b, Duration::from_secs(2)));
            let result = super::exchange_pipeline_identity(&mut front, a_format, &a, Duration::from_secs(2));
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
            a.extend(super::pipeline_settings::fingerprints(experiments(), |_| None).unwrap());
            b.extend(super::pipeline_settings::fingerprints(experiments(), |name|
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
                bytes.extend(super::pipeline_settings::fingerprints(experiments(), |name| {
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
            bytes.extend(super::pipeline_settings::fingerprints(parsed, |_| None).unwrap());
            bytes
        };
        for setting in ["VADE_W2_PP_C8", "VADE_MTP_C8", "VADE_MTP_COHORT_ADAPT", "VADE_C8_GUARD_ROBUST"] {
            let (front, back) = exchange(2, identity(&[]), 2, identity(&[(setting, "1")]));
            assert!(front.is_ok() && back.is_ok(), "{setting}: {front:?}, {back:?}");
        }
        let (front, back) = exchange(2, identity(&[]), 2, identity(&[("VADE_POOL_GRAPH_FALLBACK", "0")]));
        assert!(front.is_ok() && back.is_ok());
        for (setting, expected) in [("VADE_W2_PP_C8", "VADE_W2_PP_C8"),
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
            if setting == "VADE_W2_PP_C8" {
                assert!(front.contains("local=1, peer=0"), "{front}");
                assert!(back.contains("local=0, peer=1"), "{back}");
            }
        }
        // Even an explicit plain override cannot hide a different parsed route gate.
        let (front, back) = exchange(2, identity(&[("VADE_W2_PP_C8", "1"), ("VADE_MTP_BATCH", "0")]),
            2, identity(&[("VADE_W2_PP_C8", "0"), ("VADE_MTP_BATCH", "0")]));
        assert!(front.unwrap_err().contains("VADE_W2_PP_C8"));
        assert!(back.unwrap_err().contains("VADE_W2_PP_C8"));
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
        front.extend(super::pipeline_settings::fingerprints(experiments(), |_| None).unwrap());
        back.extend(super::pipeline_settings::fingerprints(experiments(), |name|
            Options::NAMES.contains(&name).then(|| "1".into())).unwrap());
        let (front, back) = exchange(2, front, 2, back);
        assert!(front.is_ok() && back.is_ok());
    }

    #[test]
    fn pipeline_identity_rejects_invalid_prompt_recipe_before_exchange() {
        for setting in spark_engine::flash_prompt_options::Options::NAMES {
            let error = super::pipeline_settings::fingerprints(experiments(), |name|
                (name == setting).then(|| "yes".into())).unwrap_err();
            assert!(error.contains(setting) && error.contains("must be 0 or 1"));
        }
        let error = super::pipeline_settings::fingerprints(experiments(), |name|
            (name == "VADE_PROMPT_FRONTEND_GPU").then(|| "0".into())).unwrap_err();
        assert!(error.contains("requires"));
    }

    #[test]
    fn pipeline_identity_stalled_peer_times_out_without_serving() {
        let (mut front, _back) = pair();
        let local = vec![0; super::pipeline_identity_names(2).len() * 64];
        let error = super::exchange_pipeline_identity(&mut front, 2, &local, Duration::from_millis(30)).unwrap_err();
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
                            super::exchange_pipeline_identity(&mut outgoing, 2, &local, Duration::from_secs(2))?;
                        }
                        super::exchange_pipeline_identity(&mut incoming, 2, &local, Duration::from_secs(2))
                    }));
                }
                let front_result = super::exchange_pipeline_identity(&mut front, 2, &local, Duration::from_secs(2));
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
            let result = super::exchange_pipeline_identity(&mut front, 2, &vec![0; 448], Duration::from_secs(2));
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
            let settings = super::pipeline_settings::fingerprints(experiments(), |_| None).unwrap();
            a.extend_from_slice(&settings);
            b.extend_from_slice(&settings);
            let (front, back) = exchange(2, a, 2, b);
            assert!(front.is_err() && back.is_err());
        }
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
