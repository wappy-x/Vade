//! Preserve the gate wrapper's receipts; flash-bench owns the comparison math.
//! This is the legacy fixture protocol. Its native checks and HTTP cases do not
//! establish FP4-Local coverage merely because the wrapper accepts world one.
use super::{TempDir, exec_inherited, exec_logged, executable, interrupted, sha256};
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Default)]
struct Options {
    bench: String,
    ptx: String,
    dump: String,
    url: String,
    bodies: String,
    fixtures: String,
    groups: String,
    wait: String,
    logs: Vec<String>,
    world: usize,
    topology: String,
    server_only: bool,
    selftest: bool,
    require_poison: bool,
    native: Vec<String>,
}
fn parse(args: &[String]) -> Result<Option<Options>, &'static str> {
    let mut o = Options {
        groups: "256,512".into(),
        wait: "0".into(),
        world: 2,
        ..Options::default()
    };
    let mut i = 0;
    let mut requested_world = None;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(None),
            "--" => {
                o.native = args[i + 1..].to_vec();
                break;
            }
            "--server-only" => o.server_only = true,
            "--selftest" => o.selftest = true,
            "--require-poison" => o.require_poison = true,
            "--world" => {
                requested_world = Some(match args.get(i + 1).map(String::as_str) {
                    Some("1") => 1,
                    Some("2") => 2,
                    Some("4") => 4,
                    _ => return Err("FAIL inputs: world must be 1, 2 or 4"),
                });
                i += 1;
            }
            "--server-logs" => {
                let start = i + 1;
                while args.get(i + 1).is_some_and(|arg| !arg.starts_with('-') && !arg.is_empty()) {
                    i += 1;
                }
                if start > i {
                    return Err("FAIL inputs: --server-logs needs logs in rank order");
                }
                o.logs = args[start..i + 1].to_vec();
            }
            flag @ ("--bench" | "--ptx" | "--dump-dir" | "--url" | "--bodies" | "--fixtures"
            | "--groups" | "--wait-server") => {
                let value = args
                    .get(i + 1)
                    .filter(|s| !s.is_empty())
                    .ok_or("FAIL inputs: option needs a value")?;
                match flag {
                    "--bench" => o.bench = value.clone(),
                    "--ptx" => o.ptx = value.clone(),
                    "--dump-dir" => o.dump = value.clone(),
                    "--url" => o.url = value.clone(),
                    "--bodies" => o.bodies = value.clone(),
                    "--fixtures" => o.fixtures = value.clone(),
                    "--groups" => o.groups = value.clone(),
                    "--wait-server" => o.wait = value.clone(),
                    _ => unreachable!(),
                }
                i += 1;
            }
            _ => return Err("FAIL inputs: unknown argument (see --help)"),
        }
        i += 1;
    }
    if !o.native.is_empty() {
        let native_world = match o.native.len() {
            5 => requested_world.unwrap_or(2),
            7 if o.native[5] == "--world" => match o.native[6].as_str() {
                "1" => 1,
                "2" => 2,
                "4" => 4,
                _ => return Err("FAIL inputs: world must be 1, 2 or 4"),
            },
            _ => return Err("FAIL inputs: require MODEL FIXTURES LISTEN NEXT RANK [--world 1|2|4]"),
        };
        if requested_world.is_some_and(|world| world != native_world) {
            return Err("FAIL inputs: conflicting world sizes");
        }
        o.world = native_world;
        if !o.native[4].parse::<usize>().is_ok_and(|rank| rank < o.world) {
            return Err("FAIL inputs: rank must be in 0..world");
        }
        if o.native.len() == 5 && o.world != 2 {
            o.native.extend(["--world".into(), o.world.to_string()]);
        }
    } else {
        o.world = requested_world.unwrap_or(2);
    }
    if o.world == 4 {
        o.topology = match std::env::var("TP_TOPOLOGY") {
            Ok(value) if matches!(value.as_str(), "ring" | "path") => value,
            Err(std::env::VarError::NotPresent) => "ring".into(),
            _ => return Err("FAIL inputs: world 4 needs TP_TOPOLOGY=path or ring"),
        };
    }
    Ok(Some(o))
}
fn error(message: &str, status: i32) -> i32 {
    crate::output::stderr_line(format_args!("{message}"));
    status
}
pub(super) fn run(args: &[String]) -> i32 {
    let o = match parse(args) {
        Ok(None) => {
            crate::public_output(HELP);
            return 0;
        }
        Ok(Some(o)) => o,
        Err(e) => return error(e, 2),
    };
    if o.bench.is_empty() || !executable(Path::new(&o.bench)) {
        return error("FAIL inputs: require executable --bench", 2);
    }
    if o.selftest {
        if o.require_poison {
            return error("FAIL inputs: selftest cannot require native poison", 2);
        }
        return selftest(&o.bench).unwrap_or_else(|e| error(&e, 1));
    }
    // GNU timeout's job is now handled by the Rust child deadline below.
    if !o.url.is_empty() && o.logs.len() != o.world {
        return error("FAIL inputs: --url requires one --server-logs entry per rank, in rank order", 2);
    }
    if o.wait.is_empty() || !o.wait.bytes().all(|b| b.is_ascii_digit()) {
        return error("FAIL inputs: invalid server wait", 2);
    }
    let mut receipt = Receipt {
        started: Instant::now(),
        failed: false,
        poison: "SKIP",
    };
    crate::output::report(format_args!("case rows bytes_compared bytes_different result"), true);
    let mut status = execute(&o, &mut receipt).unwrap_or_else(|e| error(&e, 1));
    let elapsed = receipt.started.elapsed().as_secs();
    if elapsed >= 900 {
        crate::output::report(format_args!("runtime 0 0 0 FAIL"), true);
        crate::output::stderr_line(format_args!("FAIL total runtime reached 900-second hang cap"));
        receipt.failed = true;
    }
    if o.require_poison && receipt.poison == "SKIP" {
        crate::output::stderr_line(format_args!(
            "FAIL poison: --require-poison needs a completed native poison PASS; SKIP is insufficient"
        ));
        receipt.poison = "FAIL";
        receipt.failed = true;
    }
    if status == 0 {
        status = i32::from(receipt.failed);
    } else if status != 2 {
        status = 1;
    }
    crate::output::stderr_line(format_args!(
        "determinism complete elapsed_seconds={elapsed} target_seconds=480 target_met={} hang_cap_seconds=900 poison={} result={status}",
        if elapsed < 480 { "yes" } else { "no" },
        receipt.poison
    ));
    status
}
struct Receipt {
    started: Instant,
    failed: bool,
    poison: &'static str,
}
impl Receipt {
    fn deadline(&self) -> Result<Instant, String> {
        let left = 900u64
            .checked_sub(self.started.elapsed().as_secs())
            .filter(|n| *n > 0)
            .ok_or("FAIL total runtime exceeded 900-second hang cap")?;
        Ok(Instant::now() + Duration::from_secs(left))
    }
}
fn expired_timeout(message: &str, log: Option<&Path>) -> Result<i32, String> {
    // The shell expands remaining before redirecting the timeout command. An
    // empty interval then fails inside GNU timeout without starting the bench.
    // Retain its C-locale log so the usual seal/incomplete checks still run.
    crate::output::stderr_line(format_args!("{message}"));
    let output = "timeout: invalid time interval ''\nTry 'timeout --help' for more information.\n";
    if let Some(path) = log {
        fs::write(path, output).map_err(|e| e.to_string())?;
    } else {
        crate::output::stderr(format_args!("{output}"));
    }
    Ok(125)
}
fn seal(log: &str, rank: &str, world: usize) -> bool {
    let expected = format!("registrations sealed before weights rank={rank} world={world} rdma={}", world > 1);
    log.split_terminator('\n').any(|line| line == expected)
}
fn complete(log: &str) -> bool {
    log.split_terminator('\n')
        .any(|l| l.starts_with("flash_gate complete "))
}
fn exact_row(log: &str, rank: &str, expected: &str) -> bool {
    let case = expected.split(' ').next().unwrap();
    let prefix = format!("logits-rank-{rank}-{case} ");
    let full = format!("logits-rank-{rank}-{expected}");
    log.split_terminator('\n')
        .filter(|l| l.starts_with(&prefix))
        .count()
        == 1
        && log.split_terminator('\n').filter(|l| *l == full).count() == 1
}
fn forward(log: &str, world: usize) {
    for line in log.split_terminator('\n') {
        if (0..world).any(|rank| line.starts_with(&format!("logits-rank-{rank}-"))) {
            crate::output::report(format_args!("{line}"), true);
        }
    }
    // The shell runs two awk passes. Keep stdout rows before stderr details
    // even when the caller captures both streams into one measurement log.
    for line in log.split_terminator('\n') {
        if line.starts_with("first_difference")
            || line.starts_with("FAIL")
            || [
                "complete",
                "mtp",
                "batch-mixed-16",
                "mixed-16-bound",
                "skip",
            ]
            .iter()
            .any(|p| line.starts_with(&format!("flash_gate {p}")))
        {
            crate::output::stderr_line(format_args!("{line}"));
        }
    }
}
// A world-four reference is local to this topology, ordered rank assignment,
// build and fixture set. Keep this private sidecar beside the raw captures;
// neither its existence nor a REF row constitutes a passing qualification.
fn provenance(o: &Options, dump: &Path, rank: &str, run_id: &str) -> Result<(), String> {
    let mut fields = crate::text::Fields::from([
        ("version".into(), "1".into()),
        ("world".into(), o.world.to_string()),
        ("rank".into(), rank.into()),
        ("topology".into(), o.topology.clone()),
        ("run_id".into(), run_id.into()),
        ("listen".into(), o.native[2].clone()),
        ("next".into(), o.native[3].clone()),
        ("groups".into(), o.groups.clone()),
        ("mtp_depths".into(), "0,3,15".into()),
    ]);
    let hash = |path: &Path| {
        sha256::digest_reader(File::open(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    };
    fields.insert("bench_sha256".into(), hash(Path::new(&o.bench))?);
    fields.insert("ptx_sha256".into(), hash(Path::new(&o.ptx))?);
    for (index, fixture) in o.native[1].split(',').enumerate() {
        fields.insert(format!("fixture.{index}.sha256"), hash(Path::new(fixture))?);
    }
    for name in [
        "SEQPAR_MEMORY_BYTES", "SEQPAR_RDMA", "SEQPAR_OVERLAP", "SEQPAR_SERIAL_COMPUTE",
        "TP_RDMA", "TP_RDMA_SMALL", "TP_CPU", "TP_AUX_LISTEN", "TP_AUX_NEXT",
        "SPARK_KV_BF16", "FLASH_BF16_EXPERTS", "SPARK_ADAPTIVE_PREFILL",
    ] {
        fields.insert(format!("env.{name}"), std::env::var(name).unwrap_or_else(|_| "<unset>".into()));
    }
    fs::write(dump.join(format!("world-4-rank-{rank}.conf")), crate::text::encode(&fields))
        .map_err(|e| e.to_string())
}
fn execute(o: &Options, r: &mut Receipt) -> Result<i32, String> {
    let (rank, fixtures);
    if !o.server_only {
        if o.ptx.is_empty()
            || !Path::new(&o.ptx).is_file()
            || o.dump.is_empty()
            || o.native.len() < 5
        {
            return Ok(error(
                "FAIL inputs: require PTX, new dump directory and MODEL FIXTURES ADDR0 ADDR1 RANK",
                2,
            ));
        }
        rank = o.native[4].as_str();
        fixtures = o.native[1].as_str();
        // Bash read ignores trailing IFS separators; validation of numerical
        // group sizes belongs to flash-bench, as it does in the shell gate.
        let parts = o
            .groups
            .split('\n')
            .next()
            .unwrap()
            .splitn(3, ',')
            .collect::<Vec<_>>();
        if parts.len() < 2
            || parts[..2].iter().any(|v| v.is_empty())
            || parts.get(2).is_some_and(|v| !matches!(*v, "" | ","))
        {
            return Ok(error("FAIL inputs: need two groups", 2));
        }
        let dump = Path::new(&o.dump);
        if dump.exists() {
            return Ok(error("FAIL inputs: dump directory must be new", 2));
        }
        let preflight = TempDir::new("vade-gate-plan")?;
        let plan_file = preflight.path.join("stdout");
        let status = super::exec_stdout(Command::new(&o.bench).args(["--gate-dry-run", "--model", &o.native[0],
            &o.world.to_string(), rank, fixtures, &o.groups]), &plan_file, Some(r.deadline()?))?;
        if status != 0 { return Ok(status); }
        let plan = fs::read_to_string(plan_file).map_err(|e| e.to_string())?;
        let width = plan.split_whitespace().find_map(|s| s.strip_prefix("row_bytes="))
            .and_then(|s| s.parse::<usize>().ok()).filter(|&n| n != 0 && n % 4 == 0 && n.checked_mul(768).is_some())
            .ok_or("FAIL native: missing model-plan vocabulary width")?;
        if !plan.starts_with("gate_plan model=") || plan.lines().count() != 1
            || !plan.split_whitespace().any(|s| s == format!("world={}", o.world)) || !plan.split_whitespace().any(|s| s == format!("rank={rank}")) {
            return Err("FAIL native: invalid model-plan receipt".into());
        }
        fs::create_dir_all(dump).map_err(|e| e.to_string())?;
        fs::write(dump.join("gate-plan.conf"), &plan).map_err(|e| e.to_string())?;
        let run_id = std::env::var("TP_RUN_ID")
            .ok()
            .filter(|s| !s.is_empty())
            .ok_or("set a fresh common TP_RUN_ID on all ranks")?;
        if o.world == 4 {
            provenance(o, dump, rank, &run_id)?;
        }
        r.poison = "FAIL";
        let tmp = TempDir::new("vade-gate-capabilities")?;
        let capability_file = tmp.path.join("stdout");
        let status = super::exec_stdout(
            Command::new(&o.bench).arg("--gate-capabilities"),
            &capability_file,
            None,
        )?;
        if status != 0 {
            return Ok(status);
        }
        let capability = fs::read_to_string(capability_file).map_err(|e| e.to_string())?;
        let capability = capability.trim_end_matches('\n');
        crate::output::stderr_line(format_args!("{capability}"));
        let (outcome, hook) = match capability {
            "flash_gate routing_reject_per_row=0 detector=cargo-feature:flash-routing-per-row routing_reject_width=0"
            | "flash_gate routing_reject_per_row=0 detector=cargo-feature:flash-routing-per-row routing_reject_width=1" =>
            {
                crate::output::stderr_line(format_args!(
                    "flash_gate skip poison: gate per-row capability explicitly off (--no-default-features); use --require-poison for merge receipts"
                ));
                ("SKIP", "0")
            }
            "flash_gate routing_reject_per_row=1 detector=cargo-feature:flash-routing-per-row routing_reject_width=0" => {
                ("PASS", "1:3")
            }
            "flash_gate routing_reject_per_row=1 detector=cargo-feature:flash-routing-per-row routing_reject_width=1" => {
                ("PASS", "1:3:16")
            }
            _ => {
                return Ok(error(
                    "FAIL native: missing compiled per-row routing capability",
                    1,
                ));
            }
        };
        let expected = if outcome == "PASS" { format!("720 {} 0 PASS", 720 * width) } else { "0 0 0 SKIP".into() };
        for depth in [0, 3, 15] {
            let path = dump.join(format!("mtp-{depth}.log"));
            let path_arg = (o.world == 4 && o.topology == "path")
                .then_some("--path");
            let status = match r.deadline() {
                Ok(deadline) => exec_logged(
                    Command::new(&o.bench)
                        .arg(&o.ptx)
                        .args(&o.native)
                        .args(path_arg)
                        .args([
                            "--mtp",
                            &depth.to_string(),
                            "--gate",
                            &o.dump,
                            "--prefill-group",
                            parts[0],
                            "--gate-group",
                            parts[1],
                        ])
                        .env(
                            "VADE_TEST_ROUTING_REJECT",
                            if depth == 0 { hook } else { "0" },
                        )
                        .env("TP_RUN_ID", format!("{run_id}-mtp-{depth}")),
                    &path,
                    Some(deadline),
                )?,
                Err(message) => expired_timeout(&message, Some(&path))?,
            };
            let log = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            forward(&log, o.world);
            if !log.lines().any(|line| line == plan.trim_end()) { return Err("FAIL native: capture model-plan receipt differs".into()); }
            if !seal(&log, rank, o.world) {
                crate::output::stderr_line(format_args!(
                    "FAIL native MTP {depth}: missing rank={rank} world={} RDMA seal line (old binaries fail even when bytes match)", o.world
                ));
                r.failed = true;
            }
            if !complete(&log) {
                crate::output::report(format_args!("logits-rank-{rank}-mtp-{depth}-incomplete 0 0 0 FAIL"), true);
                let lines = log.split_terminator('\n').collect::<Vec<_>>();
                for line in &lines[lines.len().saturating_sub(12)..] {
                    crate::output::stderr_line(format_args!("{line}"));
                }
                r.failed = true;
            }
            r.failed |= status != 0;
            if depth == 0 {
                let poison_row = format!("batch-mixed-16-poison {expected}");
                if status == 0 && complete(&log) && exact_row(&log, rank, &poison_row) {
                    r.poison = outcome;
                }
                for row in [
                    format!("batch-mixed 577 {} 0 PASS", 577 * width),
                    format!("batch-mixed-16 768 {} 0 PASS", 768 * width),
                    poison_row,
                ] {
                    if !exact_row(&log, rank, &row) {
                        crate::output::stderr_line(format_args!("FAIL native: expected exactly one logits-rank-{rank}-{row}"));
                        r.failed = true;
                    }
                }
                if dump.join("reference").is_dir() || o.world == 4 {
                    let count = manifest(dump, rank, width)?;
                    if count != 192 {
                        crate::output::report(format_args!("logits-rank-{rank}-reference-manifest {count} 0 0 FAIL"), true);
                        crate::output::stderr_line(format_args!("FAIL reference manifest: expected 192 rows, found {count}"));
                        r.failed = true;
                    }
                }
            }
            if interrupted() != 0 {
                return Ok(1);
            }
        }
        crate::output::stderr_line(format_args!(
            "native_phase complete elapsed_seconds={} target_seconds=480",
            r.started.elapsed().as_secs()
        ));
        crate::output::stderr_line(format_args!(
            "logits_finished: GPU model released; rank zero may now use the running server URL"
        ));
    } else {
        if !o.native.is_empty() || o.fixtures.is_empty() || o.url.is_empty() {
            return Ok(error("FAIL inputs: server-only needs fixtures and URL", 2));
        }
        rank = "0";
        fixtures = &o.fixtures;
    }
    if rank == "0" && !o.url.is_empty() {
        let tmp = TempDir::new("vade-gate-health")?;
        let mut wait = o.wait.clone();
        loop {
            let log = tmp.path.join("health.log");
            let health = super::exec_stdout(
                Command::new("curl")
                    .args(["--noproxy", "*", "--fail", "--silent", "--max-time", "2"])
                    .arg(format!(
                        "{}/health",
                        o.url.strip_suffix('/').unwrap_or(&o.url)
                    )),
                &log,
                None,
            )?;
            if health == 0 {
                if o.world == 4 {
                    let body = fs::read_to_string(&log).map_err(|e| e.to_string())?;
                    if crate::json::string(&body, "status").as_deref() != Some("ready")
                        || crate::json::path_number(&body, &["vade", "numerics", "world_size"]) != Some(4.0)
                        || crate::json::path_string(&body, &["vade", "numerics", "topology"]).as_deref()
                            != Some(o.topology.as_str())
                    {
                        return Ok(error("FAIL server health: require ready world 4 with the selected TP_TOPOLOGY", 1));
                    }
                }
                break;
            }
            if !wait.parse::<i64>().is_ok_and(|n| n > 0) {
                return Ok(error("FAIL server health unavailable", 1));
            }
            if interrupted() != 0 {
                return Ok(1);
            }
            std::thread::sleep(Duration::from_secs(1));
            // Bash test compares decimal integers, but the subsequent shell
            // arithmetic treats a leading zero as octal. Preserve that edge.
            let radix = if wait.len() > 1 && wait.starts_with('0') {
                8
            } else {
                10
            };
            wait = (i64::from_str_radix(&wait, radix)
                .map_err(|_| "FAIL invalid server wait arithmetic")?
                - 1)
            .to_string();
            r.deadline()?;
        }
        for (rank, log) in o.logs.iter().enumerate() {
            if !Path::new(log).is_file()
                || !fs::read_to_string(log).is_ok_and(|v| seal(&v, &rank.to_string(), o.world))
            {
                crate::output::stderr_line(format_args!(
                    "FAIL flash-serve: missing rank={rank} world={} RDMA seal line (old binaries fail even when bytes match)", o.world
                ));
                r.failed = true;
            }
        }
        let mut command = Command::new(&o.bench);
        command.args(["--gate-server", &o.url, fixtures]);
        if !o.bodies.is_empty() {
            command.args(["--bodies", &o.bodies]);
        }
        let status = match r.deadline() {
            Ok(deadline) => exec_inherited(&mut command, Some(deadline))?,
            Err(message) => expired_timeout(&message, None)?,
        };
        r.failed |= status != 0;
    }
    Ok(i32::from(r.failed))
}
fn selftest(bench: &str) -> Result<i32, String> {
    let tmp = TempDir::new("vade-dump-selftest")?;
    let reference = tmp.path.join("reference.f32");
    let copied = tmp.path.join("copied.f32");
    let mut bytes = vec![0; 993280];
    fs::write(&reference, &bytes).map_err(|e| e.to_string())?;
    bytes[47] = 1;
    fs::write(&copied, &bytes).map_err(|e| e.to_string())?;
    let log = tmp.path.join("result");
    let status = exec_logged(
        Command::new(bench)
            .arg("--gate-compare")
            .args([reference.as_os_str(), copied.as_os_str()])
            .args(["285", "7", "63"]),
        &log,
        None,
    )?;
    if interrupted() != 0 {
        return Ok(128 + interrupted());
    }
    let output = fs::read_to_string(log).map_err(|e| e.to_string())?;
    crate::output::report(format_args!("{output}"), false);
    if status != 1
        || !output
            .split_terminator('\n')
            .any(|l| l == "logits-dump-compare 1 993280 1 FAIL")
        || !output.contains("prompt=285 row=7 position=63 vocabulary=11 byte=3")
    {
        return Ok(error(
            "FAIL selftest: corrupted dump was not rejected with its location",
            1,
        ));
    }
    crate::output::report(format_args!(
        "selftest PASS: expected FAIL and exit 1 observed; 993280 bytes compared, 1 byte differs"
    ), true);
    Ok(0)
}
fn files(dir: &Path, rank: &str, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    // GNU find's default -P does not follow even a command-line symlink.
    if fs::symlink_metadata(dir)
        .map_err(|e| e.to_string())?
        .file_type()
        .is_symlink()
    {
        return Ok(());
    }
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_dir() {
            files(&entry.path(), rank, paths)?;
        } else if kind.is_file()
            && entry
                .file_name()
                .as_bytes()
                .ends_with(format!("-rank-{rank}.f32").as_bytes())
        {
            paths.push(entry.path());
        }
    }
    Ok(())
}
fn manifest(dump: &Path, rank: &str, width: usize) -> Result<usize, String> {
    let mut paths = Vec::new();
    files(&dump.join("reference"), rank, &mut paths)?;
    paths.sort_by(|a, b| a.as_os_str().as_bytes().cmp(b.as_os_str().as_bytes()));
    let mut out = File::create(dump.join(format!("reference-rank-{rank}.sha256")))
        .map_err(|e| e.to_string())?;
    if paths.is_empty() {
        // xargs without -r invokes sha256sum once on empty stdin.
        writeln!(out, "{}  -", sha256::sha(b"")).map_err(|e| e.to_string())?;
        return Ok(1);
    }
    for path in &paths {
        if fs::metadata(path).map_err(|e| e.to_string())?.len() != width as u64 { return Err("FAIL reference manifest: vocabulary size differs".into()); }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() != width || bytes.as_chunks::<4>().0.iter().any(|&v| !f32::from_le_bytes(v).is_finite()) {
            return Err("FAIL reference manifest: incomplete or nonfinite vocabulary row".into());
        }
        let hash = sha256::sha(&bytes);
        let name = path
            .strip_prefix(dump)
            .map_err(|e| e.to_string())?
            .as_os_str()
            .as_bytes();
        let escaped = name.contains(&b'\\') || name.contains(&b'\n');
        if escaped {
            out.write_all(b"\\").map_err(|e| e.to_string())?;
        }
        write!(out, "{hash}  ").map_err(|e| e.to_string())?;
        for &b in name {
            let bytes: &[u8] = match b {
                b'\\' => b"\\\\",
                b'\n' => b"\\n",
                _ => std::slice::from_ref(&b),
            };
            out.write_all(bytes).map_err(|e| e.to_string())?;
        }
        out.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    Ok(paths.len())
}

const HELP: &str = r#"Usage: vade test determinism --bench BIN --ptx PTX --dump-dir DIR \
  [--groups N,N] [--require-poison] [--url URL --server-logs LOG0 LOG1 [LOG2 LOG3] --bodies REQUEST.json,... --wait-server SECONDS] -- \
  MODEL FIXTURE.json[,FIXTURE.json...] ADDR0 ADDR1 RANK [--world 1|2|4]
Or: vade test determinism --server-only [--world 1|2|4] --bench BIN --url URL --server-logs LOG0 LOG1 [LOG2 LOG3] \
  --fixtures FIXTURE.json,... [--bodies REQUEST.json,...]
Or: vade test determinism --selftest --bench BIN

Launch all ranks concurrently with identical inputs and a fresh external
TP_RUN_ID, private SEQPAR_CONTROL_SECRET, SEQPAR_RDMA=1, TP_TOPOLOGY=ring or path,
SPARK_PROFILE=0 and memory budget.
World 2 remains the default: ADDR0/ADDR1 are the two shared pair endpoints.
World 1 uses the generic native adapter with rank 0 and a non-RDMA startup seal.
Native preflight inspects ModelPlan headers and binds model/world/fixtures/groups to captures.
World 4: ADDR0/ADDR1 are this rank's LISTEN/NEXT endpoints, rank is 0..3.
For world 4, TP_TOPOLOGY=path adds --path to flash-bench; ring is the default.
Every rank needs its own new dump directory. World-four runs save a private
world-4-rank-R.conf sidecar with topology, runtime settings and input hashes.
Keep goldens separate by world, topology, ordered rank assignment and settings;
see cli/docs/test.md for the world-four capture and promotion procedure.
No hosts, model paths or server lifecycle are inferred. The caller owns all
rank PIDs and, after the logit runs release the GPU, starts the usual server.
With --url, rank zero waits up to --wait-server seconds (default 0) for /health.
Rank zero requires all flash-serve startup logs via --server-logs in rank order, readable
locally after health succeeds. Capture stdout and stderr together on each rank.
All three MTP logs on each rank and every server log must contain that rank's
RDMA seal line for the selected world. An old binary without it fails even when tokens and logit bytes match.
Use the server's root URL; --bodies uses /v1/chat/completions with retained
request bodies in fixture order. Without bodies, /v1/completions receives the
exact fixture prompt_token_ids. No endpoint is added. TLS uses a local tunnel.
Order fixtures and matching bodies 8192,2048,285: the HTTP gate completes each
prompt's cold/prime/hit/cohorts before the mixed cohort, keeping 285 newest.

Fixtures: exactly three prompts of 285/2048/8192 tokens with at least 64 rows;
the first 64 frozen labels drive the continuation and follow-up captures.
Groups default to 256,512: distinct multiples of 64 within 512. They limit row
groups inside a prefill call while attention keeps its existing 64-token tile.

Logit cases: cold batch-one MTP-0 reference; homogeneous batches with independent state
2/8/16 through forward_batch; both prefill groups; snapshot/restore cache replay;
MTP-3 cold and replay with actual proposals, accepted-prefix commit and rollback.
Each MTP case reports proposal/acceptance counts and fails if no proposal is accepted.
Row launches (rows-4/8/16, verification depth 15): each prompt is restored into
fresh sessions that verify the same 63 teacher-forced decode inputs in launches
of k + 1 rows for k = 3, 7 and 15; every row is compared with the one-row reference.
Follow-up: 63 decode inputs, canonicalize, fresh exact restore, then one suffix
token against a cold prefill of the identical prompt plus 64 frozen labels.
Native batch-mixed cycles widths 3..16: 577 rows, 573122560 bytes/rank against each lane's solo position; HTTP mixes three prompts.
Native batch-mixed-16 cycles 8192/2048/285 with lane r advanced by r frozen
inputs, then compares 48 full-width passes to that lane's solo positions:
768 rows, 762839040 bytes/rank. batch-mixed-16-poison uses the env-only hook
VADE_TEST_ROUTING_REJECT=1:3 (legacy) or 1:3:16 (width-aware), deferred through
setup, requires row 3 to fail inside the first sixteen-row pass, retains its
slot as padding, and compares 720 survivor rows (715161600 bytes/rank) to the
no-hook mixed-16 captures. --gate-capabilities reports the compiled per-row
API via the engine Cargo feature flash-routing-per-row; detection is printed.
Default flash-bench builds report routing_reject_per_row=1 and must run poison.
A build with --no-default-features reports per_row=0 and the poison cell is
0 0 0 SKIP; it turns off only the gate's per-row capability (per-row rejection in
the engine and server stays on) and is not an aggregate comparison arm.
The summary always carries poison=SKIP|PASS|FAIL. Input errors exit 2.
--require-poison makes SKIP a FAIL and exits 1, including server-only runs;
the merge-receipt form is vade test determinism --require-poison
followed by the normal rank arguments above.
Both forms use sixteen rows for all 48 calls. The poison step compares 15
survivor rows (14899200 bytes/rank); 47 subsequent calls retain all sixteen
slots and compare 705 survivor rows (700262400 bytes/rank).
Both extra native cells share a 120-second work cap (checked between calls;
the 900-second invocation timeout still bounds hangs), reuse one weight copy,
and cannot exceed the device peak already reached by the existing cases.
Extra dump bytes are 762839040/rank, or 1478000640/rank with poison enabled.
Homogeneous cohorts prefill one row and restore its snapshot into the others.
Only the lead's prompt-end output is captured; all rows' 63 actual decode
outputs are compared. A copied prompt-end output is never a PASS comparison.
These are independent-axis cases, each compared to the same reference. Every
complete vocabulary row is compared byte-for-byte, including signed zero.
Dump format: DIR/CASE/prompt-NNNNN/session-NNNNN/position-NNNNN-rank-R.f32
(full vocabulary, little-endian FP32). DIR must be new on each rank. Capture
logs and reference-rank-R.sha256 stay there for later baseline comparison.

Server cases per prompt: cache=false cold; cache=true prime then hit; 2/8/16
simultaneous requests with the same cached body, temperature 0, seed 0, plain
drafting, max_tokens 64. Compare IDs, finish reason and prompt/completion/total
usage. A natural EOS is charged in usage but its ID is not streamed.
Cached-token counts deliberately vary; cold must reuse zero and every
hit/cohort must report reuse. Missing replies, IDs or usage fail closed.

One table shape: case rows bytes_compared bytes_different result. Logit rows
are positions; server rows are responses. First differences go to stderr.
Reference captures print REF with zero compared bytes and do not count as PASS.
Any FAIL or incomplete run exits nonzero. Requires curl for --url and the extended flash-bench. Rust handles deadlines
and reference SHA-256 manifests without GNU timeout or sha256sum. Whole invocation is limited to
900 seconds as a hang cap; elapsed time is reported against the 480-second
target without failing on that target. A timeout fails. --help does no work.
--selftest needs no GPU: copy a full dump, flip one byte, and require the same
native comparator to exit 1 with FAIL and prompt/row/position/vocabulary detail.
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn poison_evidence_must_be_present_unique_and_exact() {
        let expected = "batch-mixed-16-poison 720 715161600 0 PASS";
        let good = format!("logits-rank-0-{expected}\n");
        assert!(exact_row(&good, "0", expected));
        for log in [String::new(), good.repeat(2), good.replace("720", "719"),
            good.replace("PASS", "SKIP"), good.replace("rank-0", "rank-1")] {
            assert!(!exact_row(&log, "0", expected));
        }
        for (world, rank, valid) in [("1","0",true),("1","1",false),("2","1",true),("4","3",true),("3","0",false)] {
            let args = ["--","model","fixtures","listen","next",rank,"--world",world].map(str::to_owned);
            assert_eq!(parse(&args).is_ok(),valid);
        }
        assert!(seal("registrations sealed before weights rank=0 world=1 rdma=false\n","0",1));
        assert!(!seal("registrations sealed before weights rank=0 world=2 rdma=false\n","0",2));
    }
    #[test]
    fn poison_cannot_be_certified_by_comparator_selftest() {
        let args = vec!["--bench".into(), std::env::current_exe().unwrap().to_string_lossy().into_owned(),
            "--selftest".into(), "--require-poison".into()];
        assert!(parse(&args).unwrap().unwrap().require_poison);
        assert_eq!(run(&args), 2);
    }
}
