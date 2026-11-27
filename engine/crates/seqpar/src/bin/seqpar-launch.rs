//! Run finite prefill jobs on loopback processes or discovered Spark neighbours.
use std::{io::{self, BufRead, Write}, net::Ipv4Addr, process::{Child, Command, Stdio}, time::{Duration, Instant}};
use spark_seqpar::{discovery, Result};

const USAGE: &str = "usage: seqpar-launch HOST... -- TOKENS_FILE MAX_NEW_TOKENS\n       seqpar-launch --local N -- TOKENS_FILE MAX_NEW_TOKENS\n       seqpar-launch discover\nList SSH aliases in physical neighbour order; the final host decodes.\nSet SEQPAR_MODEL_DIR; remote jobs also require SEQPAR_REMOTE_DIR.\nSEQPAR_TOPOLOGY=auto (default), ring (require closing link), or path.\nSEQPAR_PTX selects the kernel file; SEQPAR_PORT selects the first rank port.";

fn quote(s: &str) -> String { format!("'{}'", s.replace('\'', "'\\''")) }
fn remote_path(s: &str) -> String {
    if let Some(s) = s.strip_prefix("~/") { format!("\"$HOME\"/{}", quote(s)) }
    else { quote(s) }
}
fn ssh(host: &str, command: &str) -> Command {
    let mut c = Command::new("ssh");
    c.args(["-o", "BatchMode=yes", host, command]); c
}
struct Jobs(Vec<Child>, Vec<std::thread::JoinHandle<io::Result<()>>>);
impl Drop for Jobs {
    fn drop(&mut self) {
        for child in &mut self.0 { drop(child.stdin.take()); }
        let until = Instant::now() + Duration::from_secs(2);
        while self.0.iter_mut().any(|c| matches!(c.try_wait(), Ok(None))) && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20));
        }
        for child in &mut self.0 {
            if matches!(child.try_wait(), Ok(None)) { let _ = child.kill(); }
            let _ = child.wait();
        }
        for reader in self.1.drain(..) {
            match reader.join() {
                Ok(Ok(())) => (),
                Ok(Err(error)) => eprintln!("rank stderr: {error}"),
                Err(_) => eprintln!("rank stderr reader panicked"),
            }
        }
    }
}
fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|s| matches!(s.as_str(), "--help" | "-h")) {
        println!("{USAGE}");
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "discover") {
        for link in discovery::links()? { println!("{} {}/{} {}", link.interface, link.address, link.prefix, link.mbps); }
        return Ok(());
    }
    let split = args.iter().position(|s| s == "--").ok_or(USAGE)?;
    let hosts = &args[..split];
    let tail = &args[split + 1..];
    let local = hosts.first().is_some_and(|s| s == "--local");
    let world = if local {
        if hosts.len() != 2 { return Err(USAGE.into()); }
        hosts[1].parse::<usize>().map_err(|_| "invalid local rank count")?
    } else {
        if hosts.iter().any(|s| s.starts_with('-') || s.contains(char::is_whitespace)) { return Err("invalid SSH alias".into()); }
        hosts.len()
    };
    if world == 0 || tail.len() != 2 { return Err("need positive rank count and TOKENS_FILE MAX_NEW_TOKENS".into()); }
    let directory = if local { String::new() } else {
        remote_path(&std::env::var("SEQPAR_REMOTE_DIR").map_err(|_| "set SEQPAR_REMOTE_DIR to the remote build directory")?)
    };
    let model = std::env::var("SEQPAR_MODEL_DIR").map_err(|_| "set SEQPAR_MODEL_DIR to the checkpoint directory")?;
    let secret = spark_seqpar::control::control_secret()?;
    if !local && secret.as_ref().is_some_and(|value| value.contains(&b'\n')) {
        return Err("SEQPAR_CONTROL_SECRET cannot contain a newline when launching over SSH".into());
    }
    if !local && world > 1 && secret.is_none() { return Err("multi-host launch requires SEQPAR_CONTROL_SECRET".into()); }
    let ptx = std::env::var("SEQPAR_PTX").unwrap_or_else(|_| "target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx".into());
    let port = std::env::var("SEQPAR_PORT").unwrap_or_else(|_| "29580".into()).parse::<u16>().map_err(|e| e.to_string())?;
    let last_rank = u16::try_from(world - 1).map_err(|_| "rank count exceeds TCP port range")?;
    if port == 0 || port.checked_add(last_rank).is_none() { return Err("SEQPAR_PORT range must fit nonzero TCP ports".into()); }
    let mut links = Vec::new();
    for host in hosts.iter().filter(|_| !local) {
        let result = ssh(host, &format!("cd {directory} && target/release/seqpar-launch discover")).output().map_err(|e| e.to_string())?;
        if !result.status.success() { return Err(format!("discovery failed on {host}: {}", String::from_utf8_lossy(&result.stderr))); }
        let mut found = Vec::new();
        for line in String::from_utf8_lossy(&result.stdout).lines() {
            let (address, prefix) = line.split_whitespace().nth(1).and_then(|s| s.split_once('/')).ok_or("invalid discovery reply")?;
            found.push((address.parse::<Ipv4Addr>().map_err(|e| e.to_string())?, prefix.parse::<u8>().map_err(|e| e.to_string())?));
        }
        links.push(found);
    }
    let requested_topology = std::env::var("SEQPAR_TOPOLOGY").unwrap_or_else(|_| "auto".into());
    if !matches!(requested_topology.as_str(), "auto" | "ring" | "path") {
        return Err("SEQPAR_TOPOLOGY must be auto, ring, or path".into());
    }
    let mut path = requested_topology == "path";
    let mut listen = vec![format!("{}:0", Ipv4Addr::LOCALHOST); world];
    let mut next = listen.clone();
    for rank in 0..world {
        let peer = (rank + 1) % world;
        if local {
            listen[rank] = format!("{}:{}", Ipv4Addr::LOCALHOST, port + rank as u16);
            next[rank] = format!("{}:{}", Ipv4Addr::LOCALHOST, port + peer as u16);
            continue;
        }
        if world == 1 { break; }
        let address = links[rank].iter().find_map(|&(a,p)| links[peer].iter().find_map(|&(b,q)| {
            let mask = if p == 0 {0} else {u32::MAX << (32-p.min(32))};
            (p == q && a != b && u32::from(a)&mask == u32::from(b)&mask).then_some(b)
        }));
        if peer == 0 && (path || (requested_topology == "auto" && address.is_none())) { path = true; continue; }
        let address = address.ok_or_else(|| format!("no direct QSFP link {} -> {}; list hosts in physical neighbor order", hosts[rank], hosts[peer]))?;
        let peer_port = port.checked_add(peer as u16).ok_or("port overflow")?;
        next[rank] = format!("{address}:{peer_port}");
        listen[peer] = next[rank].clone();
    }
    let run_id = format!("{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
    let env = ["SEQPAR_RDMA", "SEQPAR_ATTENTION_WEIGHT", "SEQPAR_OVERLAP", "SEQPAR_SERIAL_COMPUTE", "SEQPAR_MEMORY_BYTES", "SPARK_PROFILE", "SPARK_ADAPTIVE_PREFILL"]
        .iter().map(|key| match std::env::var(key) {
            Ok(value) => format!("export {key}={}; ", quote(&value)),
            Err(_) => format!("unset {key}; "),
        }).collect::<String>();
    let mut jobs = Jobs(Vec::new(), Vec::new());
    let topology = if path { "path" } else { "ring" };
    let executable = std::env::current_exe().map_err(|e| e.to_string())?.with_file_name("seqpar-generate");
    eprintln!("launch world={world} topology={topology} placement={} run={run_id}", if local { "loopback".into() } else { hosts.join(",") });
    for rank in 0..world {
        let mut command = if local {
            let mut command = Command::new(&executable);
            command.args([&ptx, &model, &rank.to_string(), &world.to_string(), &listen[rank], &next[rank], &tail[0], &tail[1]])
                .env("SEQPAR_RUN_ID", &run_id).env("SEQPAR_WATCH_STDIN", "1")
                .env("SEQPAR_TOPOLOGY", topology).env("SEQPAR_RDMA", "0");
            command
        } else {
            let authenticate = if secret.is_some() { "IFS= read -r SEQPAR_CONTROL_SECRET || exit 1; export SEQPAR_CONTROL_SECRET; " } else { "" };
            let command = format!("{authenticate}{env}export SEQPAR_RUN_ID={} SEQPAR_WATCH_STDIN=1 SEQPAR_TOPOLOGY={topology}; cd {directory} && exec target/release/seqpar-generate {} {} {rank} {world} {} {} {} {}", quote(&run_id),remote_path(&ptx),remote_path(&model),quote(&listen[rank]),quote(&next[rank]),remote_path(&tail[0]),quote(&tail[1]));
            ssh(&hosts[rank], &command)
        };
        jobs.0.push(command.stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?);
        if let Some(secret) = secret.as_ref().filter(|_| !local) {
            let stdin = jobs.0.last_mut().unwrap().stdin.as_mut().ok_or("missing rank stdin")?;
            stdin.write_all(secret).and_then(|_| stdin.write_all(b"\n")).map_err(|e| e.to_string())?;
        }
        let stderr = jobs.0.last_mut().unwrap().stderr.take().unwrap();
        jobs.1.push(std::thread::spawn(move || {
            let mut reader = io::BufReader::new(stderr);
            let mut line = Vec::new();
            while reader.read_until(b'\n', &mut line)? != 0 {
                io::stderr().lock().write_all(&line)?;
                line.clear();
            }
            Ok(())
        }));
    }
    loop {
        let mut running = false;
        for child in &mut jobs.0 {
            match child.try_wait().map_err(|e|e.to_string())? {
                Some(status) if !status.success() => return Err("rank failed; stopping remaining ranks".into()),
                Some(_) => (), None => running = true,
            }
        }
        if !running { return Ok(()); }
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn main() { if let Err(error) = run() { eprintln!("FAIL {error}"); std::process::exit(1); } }
