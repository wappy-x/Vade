//! Private Linux helper. Its request arrives over SSH stdin, never shell argv.
use crate::unix_file::{AnchoredPath, OpenMode, same_regular_file};
use crate::unix_flags::{LOCK_EX, LOCK_NB};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

mod gpu;
mod model;
mod prepare;
use gpu::{gpu_empty, gpu_settled};
mod process;
use process::*;

type Fields = BTreeMap<String, String>;
const MAX_REQUEST: u64 = 1024 * 1024;
const MAX_LOG: u64 = 64 * 1024;
const TERM: i32 = 15;
const KILL: i32 = 9;

unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
    fn setsid() -> i32;
    fn getpid() -> i32;
    fn kill(pid: i32, signal: i32) -> i32;
    fn signal(number: i32, handler: usize) -> usize;
    fn syscall(number: std::ffi::c_long, ...) -> std::ffi::c_long;
    fn getrlimit(resource: i32, limits: *mut MemoryLimits) -> i32;
}

#[repr(C)]
struct MemoryLimits {
    soft: u64,
    hard: u64,
}

fn check_memlock() -> Result<(), String> {
    // Linux aarch64 and x86_64 use RLIMIT_MEMLOCK=8 and unsigned 64-bit
    // rlim_t. Check the actual helper session, not a different SSH route.
    if cfg!(target_os = "linux") {
        let mut limits = MemoryLimits { soft: 0, hard: 0 };
        if unsafe { getrlimit(8, &mut limits) } != 0 {
            return Err("Cannot inspect locked-memory limits in this launch session.".into());
        }
        if limits.soft != u64::MAX || limits.hard != u64::MAX {
            return Err("RDMA launch needs unlimited soft and hard memlock in this SSH session. Use an administrator-prepared SSH alias, or set SEQPAR_RDMA=0 on the controller for TCP over the existing network. Vade does not change system limits.".into());
        }
    }
    Ok(())
}

/// Called only by the hidden `__worker` subcommand.
pub fn run() -> Result<bool, String> {
    if !cfg!(target_os = "linux") {
        return Err("The machine helper needs Linux. Run vade from the controller.".into());
    }
    let mut input = String::new();
    std::io::stdin()
        .take(MAX_REQUEST + 1)
        .read_to_string(&mut input)
        .map_err(|_| "Could not read the machine request. Retry vade status.".to_string())?;
    if input.len() as u64 > MAX_REQUEST {
        return Err("Machine request is too large. Check the configuration.".into());
    }
    let request = crate::text::parse(&input)
        .map_err(|_| "Machine request is invalid. Reinstall vade.".to_string())?;
    let response = handle(&request)?;
    std::io::stdout()
        .write_all(crate::text::encode(&response).as_bytes())
        .map_err(|_| "The reply was interrupted. Run vade status to recover.".to_string())?;
    Ok(!response.contains_key("error"))
}

/// Launch helper: durable identity exists before any engine instruction executes.
/// The parent holds the stable store lock. A lost parent never makes an unjournaled engine.
pub fn engine() -> Result<(), String> {
    if !cfg!(target_os = "linux") {
        return Err("Engine launch requires Linux.".into());
    }
    let mut input = String::new();
    std::io::stdin()
        .take(MAX_REQUEST + 1)
        .read_to_string(&mut input)
        .map_err(|_| "Cannot read launch request.")?;
    if input.len() as u64 > MAX_REQUEST {
        return Err("Launch request is too large.".into());
    }
    let request = crate::text::parse(&input)?;
    let token = required(&request, "token")?;
    if !is_hash(token) {
        return Err("Invalid launch token.".into());
    }
    let root = expanded_path(required(&request, "state_dir")?)?;
    let run = root.join(token);
    let active = read_fields(&root.join("active.conf"))?.ok_or("Launch intent is missing.")?;
    let initial = read_fields(&run.join("journal.conf"))?.ok_or("Launch journal is missing.")?;
    if active.get("token").map(String::as_str) != Some(token)
        || initial.get("token").map(String::as_str) != Some(token)
        || initial.get("phase").map(String::as_str) != Some("starting")
        || initial.contains_key("pid")
    {
        return Err("Launch intent changed. Nothing executed.".into());
    }
    let executable = expanded_path(required(&request, "engine")?)?;
    let arguments = arguments(&request)?;
    let mut identity =
        snapshot(unsafe { getpid() } as u32)?.ok_or("Cannot inspect launch helper identity.")?;
    identity.exe = fs::canonicalize(&executable).map_err(|_| "Launch binary is missing.")?;
    identity.command = command_bytes(&executable, &arguments);
    let mut record = initial;
    identity.write_to(&mut record);
    crate::text::atomic_write(&run.join("journal.conf"), &crate::text::encode(&record))?;
    // exec preserves PID and birth identity; the journal already names the exact binary/argv.
    let error = Command::new(&executable)
        .args(arguments)
        .stdin(Stdio::null())
        .exec();
    Err(format!("Engine exec failed: {error}"))
}

struct Store {
    root: PathBuf,
    run: PathBuf,
    token: String,
    _lock: File,
}

impl Store {
    fn open(request: &Fields) -> Result<Self, String> {
        let store = Self::lock(request)?;
        store.check_owner()?;
        Ok(store)
    }

    fn lock(request: &Fields) -> Result<Self, String> {
        let token = required(request, "token")?;
        if !is_hash(token) {
            return Err("Ownership token is invalid. Run vade status.".into());
        }
        let root = expanded_path(required(request, "state_dir")?)?;
        if root.parent().is_none() {
            return Err("Machine state_dir must be a dedicated directory.".into());
        }
        crate::text::private_dir(&root)?;
        let lock = private_open(&root.join(".lock"), false, true)?;
        // A permanent inode serializes probe/start/stop, including across tokens.
        if unsafe { flock(lock.as_raw_fd(), LOCK_EX | LOCK_NB) } != 0 {
            return Err("Another vade operation is using this machine. Retry shortly.".into());
        }
        Ok(Self {
            run: root.join(token),
            root,
            token: token.to_owned(),
            _lock: lock,
        })
    }

    fn active(&self) -> PathBuf {
        self.root.join("active.conf")
    }
    fn journal(&self) -> PathBuf {
        self.run.join("journal.conf")
    }
    fn log(&self) -> PathBuf {
        self.run.join("server.log")
    }
    fn env(&self) -> PathBuf {
        self.run.join("env.conf")
    }

    fn check_owner(&self) -> Result<(), String> {
        if let Some(owner) = read_fields(&self.active())? {
            if owner.get("token") != Some(&self.token) {
                return Err("This machine has another recorded owner. Stop it with its original launcher first.".into());
            }
        }
        Ok(())
    }

    fn load(&self) -> Result<Option<Fields>, String> {
        let record = read_fields(&self.journal())?;
        if let Some(record) = &record {
            if record.get("token") != Some(&self.token) {
                return Err("The saved run file belongs to another launch. Leave these files in place; see README Recovery.".into());
            }
        } else if self.active().exists() {
            return Err(
                "The machine's saved process file is missing. Leave these files in place; see README Recovery."
                    .into(),
            );
        }
        Ok(record)
    }

    fn save(&self, record: &Fields) -> Result<(), String> {
        crate::text::atomic_write(&self.journal(), &crate::text::encode(record))
    }

    fn ensure_available(&self) -> Result<(), String> {
        if self.active().exists() {
            return Err(
                "This machine is already reserved. Run vade status, then vade down.".into(),
            );
        }
        let entries = fs::read_dir(&self.root).map_err(|_| {
            "Cannot inspect machine state. Check state_dir permissions.".to_string()
        })?;
        for entry in entries {
            let entry =
                entry.map_err(|_| "Cannot inspect machine state. Run vade doctor.".to_string())?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if is_hash(name) {
                let kind = entry
                    .file_type()
                    .map_err(|_| "Cannot inspect saved run files. Leave these files in place; see README Recovery.".to_string())?;
                if !kind.is_dir()
                    || fs::read_dir(entry.path())
                        .map_err(|_| "Cannot read saved run files. Leave these files in place; see README Recovery.".to_string())?
                        .next()
                        .is_some()
                {
                    return Err("A previous machine run needs cleanup. Use its original vade down; if it cannot finish, see README Recovery.".into());
                }
            }
        }
        Ok(())
    }
}

fn handle(request: &Fields) -> Result<Fields, String> {
    let operation = required(request, "op")?;
    if operation == "warm-ptx" { return prepare::run(request); }
    if operation == "platform" {
        check_pidfd()?;
        check_driver()?;
        return Ok(fields(&[("pidfd", "ok"), ("cuda_driver", "ok")]));
    }
    // Installation can check staged weights before acquiring any GPU lease.
    if operation == "model" {
        let path = expanded_path(required(request, "model_path")?)?;
        let cached = if let Some(stamp) = request.get("model_cache") {
            model::cached(&path, &expanded_path(stamp)?)?
        } else { check_model(&path)?; false };
        return Ok(fields(&[("weights", "present"), ("inventory_cache", if cached { "hit" } else { "checked" })]));
    }
    // Use the same expansion as engine argv, without creating ownership state.
    if operation == "paths" {
        let ptx = expanded_path(required(request, "ptx_path")?)?;
        let model = expanded_path(required(request, "model_path")?)?;
        return Ok(fields(&[
            ("ptx_path", &ptx.to_string_lossy()),
            ("model_path", &model.to_string_lossy()),
            ("install_prepare", "warm-ptx-v1"),
        ]));
    }
    if !matches!(
        operation,
        "probe" | "start" | "inspect" | "listener" | "stop" | "clean" | "logs" | "gpu"
    ) {
        return Err("Unknown machine operation. Reinstall vade.".into());
    }
    let store = if operation == "start" {
        let store = Store::lock(request)?;
        if let Err(error) = store.check_owner() {
            // Only a confirmed absent token directory proves this request could
            // not have started an engine. Existing/uncertain files stay owned.
            if fs::symlink_metadata(&store.run)
                .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
            {
                return Ok(fields(&[
                    ("error_code", "start_not_written"),
                    ("phase", "absent"),
                    ("error", &error),
                ]));
            }
            return Err(error);
        }
        store
    } else {
        Store::open(request)?
    };
    match operation {
        "probe" => {
            store.ensure_available()?;
            check_pidfd()?;
            if request.get("rdma").map(String::as_str) == Some("1") {
                check_memlock()?;
            }
            gpu_empty()?;
            Ok(fields(&[
                ("gpu", "empty"),
                ("pidfd", "ok"),
                ("phase", "absent"),
            ]))
        }
        "gpu" => {
            gpu_settled()?;
            Ok(fields(&[("gpu", "empty")]))
        }
        "start" => start(&store, request),
        "inspect" => inspect(&store),
        "listener" => {
            let record = store.load()?.ok_or("No recorded server.")?;
            let identity =
                Identity::from_record(&record)?.ok_or("Server identity is incomplete.")?;
            let port = required(request, "port")?
                .parse::<u16>()
                .ok()
                .filter(|p| *p >= 1024)
                .ok_or("Invalid listener port.")?;
            if !verify(&identity)? || !owns_listener(identity.pid, port)? || !verify(&identity)? {
                return Err("The recorded server does not own the loopback listener.".into());
            }
            Ok(fields(&[("listener", "owned")]))
        }
        "stop" => stop(&store),
        "clean" => clean(&store),
        "logs" => {
            store.load()?;
            Ok(fields(&[("log", &read_log(&store)?)]))
        }
        _ => unreachable!(),
    }
}

fn start(store: &Store, request: &Fields) -> Result<Fields, String> {
    store.ensure_available()?;
    check_pidfd()?;
    // Only tensor-parallel RDMA startup needs this prerequisite. The local
    // single-Spark workflow shares this helper without using RDMA.
    if request.get("env.SEQPAR_RDMA").map(String::as_str) == Some("1") {
        check_memlock()?;
    }
    let engine = expanded_path(required(request, "engine")?)?;
    let cwd = expanded_path(required(request, "cwd")?)?;
    let ptx = expanded_path(required(request, "ptx_path")?)?;
    let model = expanded_path(required(request, "model_path")?)?;
    model::cached(&model, &store.root.join("model-inventory.stamp"))?;
    check_hash(&engine, required(request, "binary_sha256")?)?;
    check_hash(&ptx, required(request, "ptx_sha256")?)?;
    let engine_identity = fs::canonicalize(&engine)
        .map_err(|_| "Engine binary is missing. Run vade install.".to_string())?;
    let arguments = arguments(request)?;
    let expected_command = command_bytes(&engine, &arguments);
    let mut environment = environment(request)?;
    persistent_cuda_cache(&store.root, required(request, "ptx_sha256")?, &mut environment)?;
    let mut private_environment = environment.clone();
    // The SSH session may itself carry credentials. Capture inherited private
    // values for redaction, while respecting explicit removals and overrides.
    for (name, value) in std::env::vars_os() {
        let name = name.to_string_lossy();
        if sensitive_environment_name(&name)
            && !environment.contains_key(&format!("env.{name}"))
            && !environment.contains_key(&format!("env_unset.{name}"))
        {
            private_environment.insert(
                format!("redact.secret.{}", hex(name.as_bytes())),
                value.to_string_lossy().into_owned(),
            );
        }
    }
    if private_environment.iter().any(|(key, value)| {
        sensitive_environment_name(key)
            && !value.is_empty()
            && arguments.iter().any(|arg| arg.contains(value))
    }) {
        return Err("A private value appeared in engine arguments. Put it in the private environment instead.".into());
    }
    gpu_empty()?;
    crate::text::private_dir(&store.run)?;
    // SSH disconnects must not interrupt the commit of the recovery journal.
    let _signals = IgnoreDisconnect::new();
    let mut record = fields(&[("token", &store.token), ("phase", "starting")]);
    store.save(&record)?;
    let prepared = (|| {
        crate::text::atomic_write(
            &store.active(),
            &crate::text::encode(&fields(&[("token", &store.token)])),
        )?;
        let log = private_open(&store.log(), true, false)?;
        // Only this separate private file contains environment values. Journals
        // and replies never contain them; it permits redacting future logs.
        crate::text::atomic_write(&store.env(), &crate::text::encode(&private_environment))?;
        let executable = std::env::current_exe().map_err(|_| "Cannot find the launch helper.")?;
        let mut command = Command::new(executable);
        command
            .arg("__engine")
            .current_dir(cwd)
            .stdin(Stdio::piped());
        command
            .stdout(
                log.try_clone()
                    .map_err(|_| "Cannot open the private engine log.".to_string())?,
            )
            .stderr(log);
        // Detached ranks must not inherit numerical switches or an EOF watcher.
        for (name, _) in std::env::vars_os() {
            if ["FLASH_", "SPARK_", "TP_", "SEQPAR_"]
                .iter()
                .any(|prefix| name.to_string_lossy().starts_with(prefix))
            {
                command.env_remove(name);
            }
        }
        for (key, value) in &environment {
            if let Some(name) = key.strip_prefix("env.") {
                command.env(name, value);
            }
            if let Some(name) = key.strip_prefix("env_unset.") {
                command.env_remove(name);
            }
        }
        command.env("TP_WATCH_STDIN", "0");
        // The helper is single threaded. This pre-exec closure invokes only
        // libc operations: no allocation, locks, or filesystem helpers.
        unsafe {
            command.pre_exec(|| {
                for number in [1, 2, 13, TERM] {
                    signal(number, 0);
                }
                if setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        command
            .spawn()
            .map_err(|_| "Engine could not start. Run vade logs, then vade down.".to_string())
    })();
    let mut child = match prepared {
        Ok(child) => child,
        Err(error) => {
            record.insert("phase".into(), "stopped".into());
            store.save(&record)?;
            return Err(error);
        }
    };
    let result = (|| {
        let mut launch = request.clone();
        launch.insert("engine".into(), engine.to_string_lossy().into_owned());
        launch.insert(
            "state_dir".into(),
            store.root.to_string_lossy().into_owned(),
        );
        launch.insert("token".into(), store.token.clone());
        child
            .stdin
            .take()
            .ok_or("Launch helper stdin is missing.")?
            .write_all(crate::text::encode(&launch).as_bytes())
            .map_err(|_| "Cannot send the launch helper its private request.")?;
        let mut previous = None;
        for _ in 0..100 {
            if child
                .try_wait()
                .map_err(|_| "Cannot inspect the new engine.".to_string())?
                .is_some()
            {
                return Err("Engine exited during startup. Run vade logs, then vade down.".into());
            }
            let identity = snapshot(child.id())?.ok_or_else(|| {
                "Engine exited during startup. Run vade logs, then vade down.".to_string()
            })?;
            if identity.exe == engine_identity && identity.command == expected_command {
                if previous.as_ref() == Some(&identity) {
                    record.insert("phase".into(), "running".into());
                    store.save(&record)?;
                    return Ok(reply(&record, true));
                }
                // Bridge recovery before waiting for a second stable snapshot.
                // A lost SSH acknowledgement can recover this owned identity.
                identity.write_to(&mut record);
                store.save(&record)?;
            }
            previous = Some(identity);
            std::thread::sleep(Duration::from_millis(20));
        }
        Err("Engine identity did not stabilize. Run vade logs, then vade down.".into())
    })();
    if result.is_err() {
        // This unreaped child cannot have its PID reused. Rollback does not use
        // records or signal any other process, even when journal writes failed.
        rollback_child(&mut child)?;
        record.insert("phase".into(), "stopped".into());
        store.save(&record)?;
    }
    result
}

fn owns_listener(pid: u32, port: u16) -> Result<bool, String> {
    // Socket inventory is readiness evidence only; it never selects signal targets.
    let table =
        fs::read_to_string("/proc/net/tcp").map_err(|_| "Cannot inspect loopback listeners.")?;
    let address = format!("0100007F:{port:04X}");
    let inodes = table
        .lines()
        .skip(1)
        .filter_map(|line| {
            let columns = line.split_whitespace().collect::<Vec<_>>();
            if columns.len() > 9 && columns[1] == address && columns[3] == "0A" {
                Some(format!("socket:[{}]", columns[9]))
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>();
    let descriptors = fs::read_dir(format!("/proc/{pid}/fd"))
        .map_err(|_| "Cannot inspect owned server sockets.")?;
    for entry in descriptors {
        let path = entry
            .map_err(|_| "Cannot inspect owned server descriptor.")?
            .path();
        if let Ok(link) = fs::read_link(path) {
            if inodes.contains(link.to_string_lossy().as_ref()) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn inspect(store: &Store) -> Result<Fields, String> {
    let Some(record) = store.load()? else {
        return Ok(reply(&Fields::new(), false));
    };
    let alive = match Identity::from_record(&record)? {
        Some(identity) => verify(&identity)?,
        None if record.get("phase").is_some_and(|v| v == "stopped") => false,
        None => return Err("Start was interrupted before the process was saved. Leave these files in place; see README Recovery.".into()),
    };
    Ok(reply(&record, alive))
}

fn stop(store: &Store) -> Result<Fields, String> {
    let Some(mut record) = store.load()? else {
        return Ok(reply(&Fields::new(), false));
    };
    match Identity::from_record(&record)? {
        Some(identity) => stop_identity(&identity)?,
        None if record.get("phase").is_some_and(|v| v == "stopped") => {}
        None => return Err(
            "The saved startup file is incomplete, so stopping is unsafe. Leave these files in place; see README Recovery."
                .into(),
        ),
    }
    record.insert("phase".into(), "stopped".into());
    store.save(&record)?;
    Ok(reply(&record, false))
}

fn clean(store: &Store) -> Result<Fields, String> {
    let Some(record) = store.load()? else {
        return Ok(reply(&Fields::new(), false));
    };
    if record.get("phase").map(String::as_str) != Some("stopped") {
        return Err("Cleanup needs a stopped engine. Run vade down again.".into());
    }
    if let Some(identity) = Identity::from_record(&record)? {
        if verify(&identity)? {
            return Err("The recorded engine is still running. Run vade down again.".into());
        }
    }
    for entry in
        fs::read_dir(&store.run).map_err(|_| "Cannot inspect stopped run files.".to_string())?
    {
        let entry = entry.map_err(|_| "Cannot inspect stopped run files.".to_string())?;
        if !matches!(
            entry.file_name().to_str(),
            Some("journal.conf" | "env.conf" | "server.log")
        ) {
            return Err(
                "Stopped run has extra files. Leave these files in place; see README Recovery."
                    .into(),
            );
        }
    }
    let log = read_log(store)?;
    remove_own_file(&store.log())?;
    remove_own_file(&store.env())?;
    // Remove the marker first while holding the permanent lock. If journal
    // removal fails, ensure_available still notices and preserves that journal.
    store.check_owner()?;
    remove_own_file(&store.active())?;
    remove_own_file(&store.journal())?;
    fs::remove_dir(&store.run).map_err(|_| {
        "Stopped run has extra files. Leave these files in place; see README Recovery.".to_string()
    })?;
    Ok(fields(&[
        ("phase", "absent"),
        ("alive", "false"),
        ("log", &log),
    ]))
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char)
    -> *mut std::ffi::c_void;
    fn dlclose(handle: *mut std::ffi::c_void) -> i32;
}
fn check_driver() -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        // Initialization checks the existing driver without creating a GPU context.
        let library = unsafe { dlopen(c"libcuda.so.1".as_ptr(), 2) };
        if library.is_null() {
            return Err("CUDA driver library libcuda.so.1 is unavailable.".into());
        }
        let symbol = unsafe { dlsym(library, c"cuInit".as_ptr()) };
        let result = if symbol.is_null() {
            -1
        } else {
            let initialize: unsafe extern "C" fn(u32) -> i32 =
                unsafe { std::mem::transmute(symbol) };
            unsafe { initialize(0) }
        };
        unsafe {
            dlclose(library);
        }
        if result != 0 {
            return Err(format!("CUDA driver initialization failed ({result})."));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    Err("CUDA driver preflight needs Linux.".into())
}

struct IgnoreDisconnect(Vec<(i32, usize)>);
impl IgnoreDisconnect {
    fn new() -> Self {
        Self(
            [1, 2, 13, TERM]
                .into_iter()
                .map(|number| (number, unsafe { signal(number, 1) }))
                .collect(),
        )
    }
}
impl Drop for IgnoreDisconnect {
    fn drop(&mut self) {
        for (number, previous) in &self.0 {
            unsafe {
                signal(*number, *previous);
            }
        }
    }
}

// CUDA keys entries by device/driver internally. Keep the directory across
// launches and installs with identical PTX; never put it in a disposable run.
fn persistent_cuda_cache(root: &Path, ptx_hash: &str, environment: &mut Fields) -> Result<(), String> {
    if !is_hash(ptx_hash) { return Err("Invalid PTX cache identity; run vade install.".into()); }
    let path = root.join("cuda-cache").join(ptx_hash);
    crate::text::private_dir(&path)?;
    environment.insert("env.CUDA_CACHE_PATH".into(), path.to_string_lossy().into_owned());
    environment.insert("env.CUDA_CACHE_DISABLE".into(), "0".into());
    environment.insert("env.CUDA_CACHE_MAXSIZE".into(), "1073741824".into());
    for name in ["CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT"] {
        environment.remove(&format!("env.{name}"));
        environment.insert(format!("env_unset.{name}"), "1".into());
    }
    Ok(())
}

fn check_hash(path: &Path, expected: &str) -> Result<(), String> {
    if !is_hash(expected) {
        return Err("Artifact manifest needs SHA-256 hashes. Run vade install.".into());
    }
    let path = path
        .to_str()
        .ok_or_else(|| "Artifact path is not valid text. Check the configuration.".to_string())?;
    let output = query("sha256sum", &["--", path], Duration::from_secs(30))?;
    if output.split_whitespace().next() != Some(expected) {
        return Err("Engine or kernel hash differs from the build manifest. Run vade install before starting.".into());
    }
    Ok(())
}

// The official FP4-Local index is about 45 MiB (many expert tensors).
// Bound reads at 64 MiB while accepting the complete current weight map.
const MAX_MODEL_INDEX_BYTES: u64 = 64 * 1024 * 1024;

fn check_model(path: &Path) -> Result<(), String> {
    let entries = fs::read_dir(path).map_err(|_| {
        "Model weights are missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".to_string()
    })?;
    if !path.join("config.json").is_file() || !path.join("tokenizer.json").is_file() {
        return Err("Model configuration or tokenizer is missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".into());
    }
    let index_path = path.join("model.safetensors.index.json");
    match File::open(&index_path) {
        Ok(index) => {
            let mut text = String::new();
            index
                .take(MAX_MODEL_INDEX_BYTES + 1)
                .read_to_string(&mut text)
                .map_err(|_| {
                    "Cannot read the model shard index. Check model permissions.".to_string()
                })?;
            if text.len() as u64 > MAX_MODEL_INDEX_BYTES {
                return Err(
                    "Model shard index exceeds 64 MiB. Check the configured model directory and INSTALL.md step 3.".into(),
                );
            }
            for shard in indexed_shards(&text)? {
                let metadata = fs::metadata(path.join(shard)).map_err(|_| {
                    "A model shard is missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".to_string()
                })?;
                if !metadata.is_file() || metadata.len() == 0 {
                    return Err("A model shard is empty or invalid. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".into());
                }
            }
            return Ok(());
        }
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                && fs::symlink_metadata(&index_path)
                    .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) => {}
        Err(_) => return Err("Cannot read the model shard index. Check model permissions.".into()),
    }
    let mut weights = false;
    for entry in entries {
        let entry = entry
            .map_err(|_| "Cannot read the model directory. Check its permissions.".to_string())?;
        if entry
            .path()
            .extension()
            .is_some_and(|suffix| suffix == "safetensors")
            && entry
                .metadata()
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() != 0)
        {
            weights = true;
        }
    }
    if !weights {
        return Err(
            "Model weights are missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".into(),
        );
    }
    Ok(())
}

// Official safetensors indexes name every shard in JSON string values. This
// small scanner decodes those strings, including escapes, without a JSON crate.
fn indexed_shards(index: &str) -> Result<BTreeSet<String>, String> {
    let invalid = || {
        "Model shard index is invalid. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights."
            .to_string()
    };
    let mut characters = index.trim().chars().peekable();
    if characters.peek() != Some(&'{') {
        return Err(invalid());
    }
    let mut structure = Vec::new();
    let mut shards = BTreeSet::new();
    let mut weight_map = false;
    while let Some(character) = characters.next() {
        match character {
            '{' | '[' => structure.push(character),
            '}' | ']' => {
                if structure.pop() != Some(if character == '}' { '{' } else { '[' }) {
                    return Err(invalid());
                }
            }
            '"' => {
                let mut value = String::new();
                loop {
                    let character = characters.next().ok_or_else(invalid)?;
                    match character {
                        '"' => break,
                        '\\' => {
                            let escaped = characters.next().ok_or_else(invalid)?;
                            value.push(match escaped {
                                '"' | '\\' | '/' => escaped,
                                'b' => '\u{8}',
                                'f' => '\u{c}',
                                'n' => '\n',
                                'r' => '\r',
                                't' => '\t',
                                'u' => {
                                    let mut code = 0;
                                    for _ in 0..4 {
                                        code = code * 16
                                            + characters
                                                .next()
                                                .and_then(|c| c.to_digit(16))
                                                .ok_or_else(invalid)?;
                                    }
                                    char::from_u32(code).ok_or_else(invalid)?
                                }
                                _ => return Err(invalid()),
                            });
                        }
                        character if character <= '\u{1f}' => return Err(invalid()),
                        character => value.push(character),
                    }
                }
                let mut following = characters.clone();
                while following
                    .peek()
                    .is_some_and(|character| character.is_whitespace())
                {
                    following.next();
                }
                let is_key = following.peek() == Some(&':');
                weight_map |= is_key && value == "weight_map";
                if !is_key && value.ends_with(".safetensors") {
                    if !Path::new(&value)
                        .components()
                        .all(|part| matches!(part, std::path::Component::Normal(_)))
                        || value.contains('\\')
                        || value.contains('\0')
                    {
                        return Err(invalid());
                    }
                    shards.insert(value);
                }
            }
            _ => {}
        }
    }
    if !structure.is_empty() || !weight_map || shards.is_empty() {
        return Err(invalid());
    }
    Ok(shards)
}

fn command_bytes(engine: &Path, arguments: &[String]) -> Vec<u8> {
    let mut command = engine.as_os_str().as_encoded_bytes().to_vec();
    command.push(0);
    for argument in arguments {
        command.extend_from_slice(argument.as_bytes());
        command.push(0);
    }
    command
}

fn query(program: &str, arguments: &[&str], timeout: Duration) -> Result<String, String> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| {
            format!("{program} is unavailable. Run vade doctor to check prerequisites.")
        })?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(format!(
                        "{program} failed. Run vade doctor to check this machine."
                    ));
                }
                let mut output = String::new();
                child
                    .stdout
                    .take()
                    .ok_or_else(|| "Machine check output was unavailable.".to_string())?
                    .take(MAX_REQUEST)
                    .read_to_string(&mut output)
                    .map_err(|_| {
                        "Could not read machine check output. Retry vade doctor.".to_string()
                    })?;
                return Ok(output);
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                // This query is still an owned unreaped child, so rollback is safe.
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{program} did not finish. Retry vade doctor."));
            }
        }
    }
}

fn read_log(store: &Store) -> Result<String, String> {
    let secrets = read_fields(&store.env())?.unwrap_or_default();
    let mut log = match private_open(&store.log(), false, false) {
        Ok(file) => file,
        Err(_) if !store.log().exists() => return Ok(String::new()),
        Err(error) => return Err(error),
    };
    let length = log
        .metadata()
        .map_err(|_| "Cannot inspect the private engine log.".to_string())?
        .len();
    log.seek(SeekFrom::Start(length.saturating_sub(MAX_LOG)))
        .map_err(|_| "Cannot read the private engine log.".to_string())?;
    let mut bytes = Vec::new();
    log.take(MAX_LOG)
        .read_to_end(&mut bytes)
        .map_err(|_| "Cannot read the private engine log.".to_string())?;
    let text = String::from_utf8_lossy(&bytes);
    // A tail can begin halfway through a secret-bearing line: discard its
    // partial first line before applying complete-line redaction.
    let text = if length > MAX_LOG {
        text.split_once('\n').map(|(_, rest)| rest).unwrap_or("")
    } else {
        &text
    };
    Ok(redact_log(text, &secrets))
}

fn redact_log(text: &str, environment: &Fields) -> String {
    let mut result = String::new();
    for line in text.lines() {
        // These exact server diagnostics contain neither paths nor credentials.
        let auth_diagnostic = matches!(line,
            "FAIL --allow-remote requires --api-key-file PATH"
            | "WARNING: --api-key-file is group- or world-readable");
        if sensitive_name(line) && !auth_diagnostic {
            result.push_str("[private environment line omitted]\n");
            continue;
        }
        let mut line = line.to_owned();
        for (key, value) in environment {
            if sensitive_environment_name(key) && !value.is_empty() {
                line = line.replace(value, "[redacted]");
            }
        }
        for character in line.chars() {
            if !character.is_control() || character == '\t' {
                result.push(character);
            }
        }
        result.push('\n');
    }
    result
}

fn sensitive_name(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    [
        "secret",
        "password",
        "credential",
        "private_key",
        "api_key",
        "api-key",
        "authorization",
        "auth_token",
        "access_token",
        "bearer ",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

fn sensitive_environment_name(value: &str) -> bool {
    let name = value
        .strip_prefix("env.")
        .or_else(|| value.strip_prefix("env_unset."))
        .unwrap_or(value)
        .to_ascii_uppercase();
    // This is a file path, not the credential; serving receives the path in argv.
    if name == "VADE_API_KEY_FILE" { return false; }
    sensitive_name(&name)
        || [
            "_TOKEN",
            "_KEY",
            "_PASSWORD",
            "_SECRET",
            "_CREDENTIAL",
            "_CREDENTIALS",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
        || matches!(name.as_str(), "TOKEN" | "KEY" | "PASSWORD" | "SECRET")
}

fn arguments(request: &Fields) -> Result<Vec<String>, String> {
    let mut indexed = BTreeMap::new();
    for (key, value) in request {
        if let Some(index) = key.strip_prefix("arg.") {
            let index = index
                .parse::<usize>()
                .map_err(|_| "Engine argument index is invalid.".to_string())?;
            if index >= 256 || indexed.insert(index, expand_home(value)?).is_some() {
                return Err("Engine argument list is invalid. Check the configuration.".into());
            }
        }
    }
    if indexed.keys().copied().ne(0..indexed.len()) {
        return Err("Engine argument list has a gap. Reinstall vade.".into());
    }
    Ok(indexed.into_values().collect())
}

fn environment(request: &Fields) -> Result<Fields, String> {
    let mut environment = Fields::new();
    for (key, value) in request {
        let (name, unset) = match key.strip_prefix("env.") {
            Some(name) => (name, false),
            None => match key.strip_prefix("env_unset.") {
                Some(name) => (name, true),
                None => continue,
            },
        };
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            || name.as_bytes()[0].is_ascii_digit()
            || value.contains('\0')
            || (unset && value != "1")
        {
            return Err("Engine environment is invalid. Check the configuration.".into());
        }
        if request.contains_key(&format!("env.{}", name))
            && request.contains_key(&format!("env_unset.{}", name))
        {
            return Err(
                "An engine variable is both set and unset. Check the configuration.".into(),
            );
        }
        environment.insert(key.clone(), value.clone());
    }
    Ok(environment)
}

fn expand_home(value: &str) -> Result<String, String> {
    if value.contains('\0') {
        return Err("A machine path contains an invalid character.".into());
    }
    let suffix = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("{home}/"));
    if let Some(suffix) = suffix {
        let home = std::env::var("HOME")
            .map_err(|_| "Machine HOME is unavailable. Check its SSH environment.".to_string())?;
        return Ok(Path::new(&home).join(suffix).to_string_lossy().into_owned());
    }
    if matches!(value, "~" | "{home}") {
        return std::env::var("HOME")
            .map_err(|_| "Machine HOME is unavailable. Check its SSH environment.".into());
    }
    Ok(value.to_owned())
}

fn expanded_path(value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(expand_home(value)?);
    if !path.is_absolute() {
        return Err(
            "Machine paths must be absolute or start with ~/. Check the configuration.".into(),
        );
    }
    Ok(path)
}

fn private_open(path: &Path, exclusive: bool, create: bool) -> Result<File, String> {
    // Keep the precheck for the existing symlink-specific error message. The
    // descriptor walk and O_NOFOLLOW enforce refusal even if this lookup races.
    crate::text::reject_symlinks(path)?;
    let open_error = || {
        "Cannot open a private machine state file. Check permissions; see README Recovery."
            .to_string()
    };
    let anchor = AnchoredPath::new(path).map_err(|_| open_error())?;
    let before = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => return Err(open_error()),
    };
    let mode = if exclusive {
        OpenMode::CreateNew
    } else if create {
        OpenMode::Create
    } else {
        OpenMode::Read
    };
    let file = anchor.open(mode).map_err(|_| open_error())?;
    let metadata = file
        .metadata()
        .map_err(|_| "Cannot inspect machine state file permissions.".to_string())?;
    let now = fs::symlink_metadata(path)
        .map_err(|_| "Cannot inspect machine state file permissions.".to_string())?;
    if !same_regular_file(&now, &metadata)
        || before
            .as_ref()
            .is_some_and(|before| !same_regular_file(before, &metadata))
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(
            "Machine state file is not private. Restore mode 600 before continuing.".into(),
        );
    }
    Ok(file)
}

fn read_fields(path: &Path) -> Result<Option<Fields>, String> {
    let file = match private_open(path, false, false) {
        Ok(file) => file,
        Err(_)
            if fs::symlink_metadata(path)
                .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let mut input = String::new();
    file.take(MAX_REQUEST + 1)
        .read_to_string(&mut input)
        .map_err(|_| "Cannot read the machine's saved run file. Leave these files in place; see README Recovery.".to_string())?;
    if input.len() as u64 > MAX_REQUEST {
        return Err("The machine's saved run file is too large. Leave these files in place; see README Recovery.".into());
    }
    crate::text::parse(&input)
        .map(Some)
        .map_err(|_| "The machine's saved run file is invalid. Leave these files in place; see README Recovery.".into())
}

fn remove_own_file(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            fs::remove_file(path).map_err(|_| {
                "Could not finish machine cleanup. Retain the record and retry vade down.".into()
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => Err(
            "Cleanup found a changed state path. Leave these files in place; see README Recovery."
                .into(),
        ),
    }
}

fn required<'a>(fields: &'a Fields, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .filter(|value| !value.is_empty())
        .map(String::as_str)
        .ok_or_else(|| {
            format!("Machine request needs {key}. Check the configuration or reinstall vade.")
        })
}
fn fields(entries: &[(&str, &str)]) -> Fields {
    entries
        .iter()
        .map(|(key, value)| ((*key).into(), (*value).into()))
        .collect()
}
fn reply(record: &Fields, alive: bool) -> Fields {
    let mut response = fields(&[
        (
            "phase",
            record.get("phase").map(String::as_str).unwrap_or("absent"),
        ),
        ("alive", if alive { "true" } else { "false" }),
    ]);
    if let Some(pid) = record.get("pid") {
        response.insert("pid".into(), pid.clone());
    }
    response
}
fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn unhex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("The saved process identity is invalid. Leave these files in place; see README Recovery.".into());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|bytes| {
            let digit = |byte: u8| {
                if byte.is_ascii_digit() {
                    byte - b'0'
                } else {
                    byte.to_ascii_lowercase() - b'a' + 10
                }
            };
            Ok((digit(bytes[0]) << 4) | digit(bytes[1]))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_open_accepts_regular_files_and_refuses_nonregular_or_linked_files() {
        let root = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!(
                "vade-private-open-{}",
                crate::text::random_hex(12).unwrap()
            ));
        crate::text::private_dir(&root).unwrap();
        let target = root.join("state");
        private_open(&target, true, false).unwrap();
        private_open(&target, false, false).unwrap();
        private_open(&target, false, true).unwrap();
        assert!(private_open(&target, true, false).is_err());
        let error = private_open(&root, false, false).unwrap_err();
        assert_eq!(
            error,
            "Machine state file is not private. Restore mode 600 before continuing."
        );
        std::os::unix::fs::symlink(&target, root.join("link")).unwrap();
        assert!(private_open(&root.join("link"), false, false).is_err());
        std::os::unix::fs::symlink(&root, root.join("parent-link")).unwrap();
        assert!(private_open(&root.join("parent-link/state"), false, false).is_err());
        fs::hard_link(&target, root.join("alias")).unwrap();
        assert!(private_open(&target, false, false).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stat_handles_parentheses_and_zombies() {
        let mut columns = vec!["0"; 20];
        columns[0] = "S";
        columns[19] = "12345";
        assert_eq!(
            stat_start(&format!("42 (engine (rank)) {}", columns.join(" "))).unwrap(),
            Some("12345".into())
        );
        columns[0] = "Z";
        assert_eq!(
            stat_start(&format!("42 (engine) {}", columns.join(" "))).unwrap(),
            None
        );
    }
    #[test]
    fn logs_hide_environment_and_terminal_controls() {
        let env = fields(&[("env.SEQPAR_CONTROL_SECRET", "private-value")]);
        assert_eq!(
            redact_log(
                "ready private-value\nSEQPAR_CONTROL_SECRET=x\nrate \u{1b}[2J\n",
                &env
            ),
            "ready [redacted]\n[private environment line omitted]\nrate [2J\n"
        );
        let warning = "WARNING: --api-key-file is group- or world-readable";
        let refusal = "FAIL --allow-remote requires --api-key-file PATH";
        assert_eq!(redact_log(&format!("{warning}\n{refusal}\n{warning} private-value"), &env), format!("{warning}\n{refusal}\n[private environment line omitted]\n"));
        for name in [
            "HF_TOKEN",
            "GITHUB_TOKEN",
            "AWS_SESSION_TOKEN",
            "VADE_API_TOKEN",
            "OTHER_KEY",
        ] {
            let env = fields(&[(&format!("env.{name}"), "private-value")]);
            assert_eq!(
                redact_log("completion_tokens=42 private-value", &env),
                "completion_tokens=42 [redacted]\n"
            );
        }
    }
    #[test]
    fn arguments_are_numeric_contiguous_and_binary_identity_roundtrips() {
        assert!(arguments(&fields(&[("arg.1", "missing zero")])).is_err());
        let bytes = b"engine\0arg\xff";
        assert_eq!(unhex(&hex(bytes)).unwrap(), bytes);
        assert!(unhex("a").is_err());
        assert!(environment(&fields(&[("env.KEY", "one"), ("env_unset.KEY", "1")])).is_err());
    }
    #[test]
    fn shard_index_and_mocked_identity_fail_closed() {
        let shards = indexed_shards(r#"{"metadata":{},"weight_map":{"a":"shard-1.safetensors","b":"shard-2\u002esafetensors"}}"#).unwrap();
        assert_eq!(shards.len(), 2);
        assert!(shards.contains("shard-2.safetensors"));
        assert!(indexed_shards(r#"{"weight_map":{"a":"../outside.safetensors"}}"#).is_err());
        assert!(indexed_shards(r#"{"weight_map":{"a":"missing.safetensors"}"#).is_err());
        let original = Identity {
            pid: 42,
            start: "100".into(),
            boot: "boot-a".into(),
            exe: PathBuf::from("/engine"),
            command: b"engine\0".to_vec(),
        };
        assert!(verify_snapshot(&original, "boot-a", Some(&original)).unwrap());
        assert!(!verify_snapshot(&original, "boot-a", None).unwrap());
        let reboot = verify_snapshot(&original, "boot-b", None).unwrap_err();
        assert!(reboot.contains("README Recovery") && !reboot.contains("Preserve"));
        let mut replacement = original.clone();
        replacement.start = "101".into();
        let changed = verify_snapshot(&original, "boot-a", Some(&replacement)).unwrap_err();
        assert!(changed.contains("Nothing was signalled") && changed.contains("README Recovery"));
    }
    #[test]
    fn stopped_recovery_is_idempotent_and_foreign_owner_is_preserved() {
        let _process_guard = crate::runner::process_test_lock();
        let base = fs::canonicalize(std::env::temp_dir()).unwrap();
        let root = base.join(format!(
            "vade-worker-{}",
            crate::text::random_hex(12).unwrap()
        ));
        let token = "a".repeat(64);
        let mut request = fields(&[
            ("op", "inspect"),
            ("token", &token),
            ("state_dir", root.to_str().unwrap()),
        ]);
        let store = Store::open(&request).unwrap();
        crate::text::private_dir(&store.run).unwrap();
        store
            .save(&fields(&[("token", &token), ("phase", "stopped")]))
            .unwrap();
        crate::text::atomic_write(
            &store.active(),
            &crate::text::encode(&fields(&[("token", &token)])),
        )
        .unwrap();
        crate::text::atomic_write(
            &store.env(),
            &crate::text::encode(&fields(&[("env.API_KEY", "hidden")])),
        )
        .unwrap();
        private_open(&store.log(), true, false)
            .unwrap()
            .write_all(b"ready hidden\n")
            .unwrap();
        drop(store);
        assert_eq!(handle(&request).unwrap()["alive"], "false");
        request.insert("op".into(), "stop".into());
        assert_eq!(handle(&request).unwrap()["phase"], "stopped");
        request.insert("op".into(), "clean".into());
        assert_eq!(handle(&request).unwrap()["log"], "ready [redacted]\n");
        assert_eq!(handle(&request).unwrap()["phase"], "absent");
        let foreign = fields(&[("token", &"b".repeat(64))]);
        crate::text::atomic_write(&root.join("active.conf"), &crate::text::encode(&foreign))
            .unwrap();
        assert!(handle(&request).is_err());
        assert_eq!(
            read_fields(&root.join("active.conf")).unwrap(),
            Some(foreign.clone())
        );
        request.insert("op".into(), "start".into());
        let refused = handle(&request).unwrap();
        assert_eq!(refused["error_code"], "start_not_written");
        assert_eq!(refused["phase"], "absent");
        assert!(!root.join(&token).exists());
        assert_eq!(
            read_fields(&root.join("active.conf")).unwrap(),
            Some(foreign)
        );
        crate::text::private_dir(&root.join(&token)).unwrap();
        let interrupted = crate::text::encode(&fields(&[("token", &token), ("phase", "starting")]));
        crate::text::atomic_write(&root.join(&token).join("journal.conf"), &interrupted).unwrap();
        assert!(handle(&request).is_err());
        assert_eq!(
            fs::read_to_string(root.join(&token).join("journal.conf")).unwrap(),
            interrupted
        );
        // With our original owner restored, incomplete startup still refuses
        // both inspect and stop, and tells the operator where the steps are.
        crate::text::atomic_write(
            &root.join("active.conf"),
            &crate::text::encode(&fields(&[("token", &token)])),
        )
        .unwrap();
        for operation in ["inspect", "stop"] {
            request.insert("op".into(), operation.into());
            let error = handle(&request).unwrap_err();
            assert!(
                error.contains("README Recovery")
                    && !error.contains("journal")
                    && !error.contains("Preserve")
            );
        }
        let model = root.join("model");
        fs::create_dir(&model).unwrap();
        for name in ["config.json", "tokenizer.json", "shard-1.safetensors"] {
            fs::write(model.join(name), "{}").unwrap();
        }
        fs::write(
            model.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}}"#,
        )
        .unwrap();
        let request = fields(&[("op", "model"), ("model_path", model.to_str().unwrap())]);
        assert!(handle(&request).is_err());
        fs::write(model.join("shard-2.safetensors"), "{}").unwrap();
        assert_eq!(handle(&request).unwrap()["weights"], "present");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn large_fp4_index_is_bounded_and_still_checks_every_shard() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-large-index-{}", crate::text::random_hex(12).unwrap()));
        fs::create_dir(&root).unwrap();
        for name in ["config.json", "tokenizer.json", "one.safetensors"] {
            fs::write(root.join(name), "{}").unwrap();
        }
        let index = root.join("model.safetensors.index.json");
        let mut file = File::create(&index).unwrap();
        file.write_all(b"{").unwrap();
        let padding = vec![b' '; 1024 * 1024];
        for _ in 0..33 { file.write_all(&padding).unwrap(); }
        file.write_all(br#""weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#).unwrap();
        drop(file);
        assert!(check_model(&root).unwrap_err().contains("shard is missing"));
        fs::write(root.join("two.safetensors"), "{}").unwrap();
        check_model(&root).unwrap();
        File::options().write(true).open(&index).unwrap().set_len(MAX_MODEL_INDEX_BYTES + 1).unwrap();
        assert!(check_model(&root).unwrap_err().contains("exceeds 64 MiB"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn shared_machine_state_is_refused_without_any_writes() {
        let root = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!(
                "vade-shared-{}",
                crate::text::random_hex(12).unwrap()
            ));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(root.join("keep"), "Other job.").unwrap();
        let request = fields(&[
            ("op", "start"),
            ("token", &"a".repeat(64)),
            ("state_dir", root.to_str().unwrap()),
        ]);
        assert!(handle(&request).unwrap_err().contains("shared"));
        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o755
        );
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        assert_eq!(fs::read_to_string(root.join("keep")).unwrap(), "Other job.");
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
#[test]
fn cuda_cache_survives_runs_and_is_keyed_by_ptx() {
    let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-cuda-cache-{}", crate::text::random_hex(8).unwrap()));
    let mut env = Fields::new();
    persistent_cuda_cache(&root, &"a".repeat(64), &mut env).unwrap();
    let first = env["env.CUDA_CACHE_PATH"].clone();
    fs::write(Path::new(&first).join("cached-cubin"), "cache").unwrap();
    persistent_cuda_cache(&root, &"a".repeat(64), &mut env).unwrap();
    assert_eq!(env["env.CUDA_CACHE_PATH"], first);
    assert!(Path::new(&first).join("cached-cubin").is_file());
    assert_eq!(env["env.CUDA_CACHE_DISABLE"], "0");
    persistent_cuda_cache(&root, &"b".repeat(64), &mut env).unwrap();
    assert_ne!(env["env.CUDA_CACHE_PATH"], first);
    fs::remove_dir_all(root).unwrap();
}
