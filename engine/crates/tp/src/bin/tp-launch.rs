//! Discover direct QSFP neighbours and supervise a distributed launch over SSH.
use spark_tp::{discovery, Result};
use std::{io::Write, net::Ipv4Addr, process::{Child, Command, Stdio}, time::{Duration, Instant}};

#[path = "tp-launch/net_plan.rs"]
mod net_plan;

fn quote(s: &str) -> String { format!("'{}'", s.replace('\'', "'\\''")) }
fn remote_path(s: &str) -> String {
    if s.starts_with('/') { quote(s) }
    else { format!("\"$HOME\"/{}", quote(s.strip_prefix("~/").unwrap_or(s))) }
}
fn ssh(host: &str, command: &str) -> Command {
    let mut c = ssh_command();
    c.args([host, command]);
    c
}
fn ssh_command() -> Command {
    let mut c = Command::new("ssh");
    c.args(["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]);
    c
}
fn run() -> Result<()> {
    let mut a = std::env::args().skip(1).collect::<Vec<_>>();
    if a.as_slice() == ["discover"] {
        for link in discovery::links()? {
            println!("{} {}/{} {}", link.interface, link.address, link.prefix, link.mbps);
        }
        return Ok(());
    }
    if a.first().map(String::as_str) == Some("net-plan") {
        return net_plan::run(&a[1..]);
    }
    let remote = if a.first().map(String::as_str) == Some("--dir") {
        if a.len() < 2 { return Err("--dir needs a remote build directory".into()); }
        let directory = a.remove(1); a.remove(0); directory
    } else { std::env::var("TP_REMOTE_DIR").map_err(|_| "set TP_REMOTE_DIR or pass --dir REMOTE_DIRECTORY")? };
    if remote.is_empty() { return Err("remote directory cannot be empty".into()); }
    let directory = remote_path(&remote);
    let split = a.iter().position(|s| s == "--").ok_or("usage: tp-launch [--dir DIRECTORY] run HOST... -- PROGRAM ARGS...")?;
    if split < 2 || !matches!(a[0].as_str(), "run" | "local") { return Err("use run HOST... or local HOST before --".into()); }
    let local = a[0] == "local";
    let mut hosts = a[1..split].to_vec();
    if hosts.iter().any(|h| h.is_empty() || h.starts_with('-') || h.contains(char::is_whitespace)) { return Err("invalid SSH alias".into()); }
    let mut args = &a[split + 1..];
    if local {
        if hosts.len() != 1 || args.is_empty() { return Err("usage: tp-launch local HOST -- RANKS PROGRAM ARGS...".into()); }
        let count = args[0].parse::<usize>().map_err(|e| e.to_string())?;
        if !(1..=64).contains(&count) { return Err("local rank count must be in 1..=64".into()); }
        hosts = vec![hosts[0].clone(); count]; args = &args[1..];
    }
    let world = hosts.len();
    if world > 64 { return Err("at most 64 ranks are supported".into()); }
    let secret = spark_seqpar::control::control_secret()?;
    if secret.as_ref().is_some_and(|value| value.contains(&b'\n')) {
        return Err("SEQPAR_CONTROL_SECRET cannot contain a newline when launching over SSH".into());
    }
    if secret.is_none() && hosts.iter().any(|host| host != &hosts[0]) {
        return Err("multi-host launch requires SEQPAR_CONTROL_SECRET".into());
    }
    if !args.first().is_some_and(|p| matches!(p.as_str(), "tp-generate" | "tp-bench" | "tp-hf-check" | "tp-serve")) {
        return Err("program must be tp-generate, tp-bench, tp-hf-check, or tp-serve".into());
    }
    let mut links: Vec<Vec<(Ipv4Addr, u8)>> = Vec::new();
    for (index, host) in hosts.iter().enumerate() {
        if local || world == 1 { links.push(Vec::new()); continue; }
        if let Some(previous) = hosts[..index].iter().position(|h| h == host) { links.push(links[previous].clone()); continue; }
        let output = ssh(host, &format!("cd {directory} && exec target/release/tp-launch discover")).output().map_err(|e| e.to_string())?;
        if !output.status.success() { return Err(format!("{host} discovery failed: {}", String::from_utf8_lossy(&output.stderr))); }
        let mut found = Vec::new();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let (ip, prefix) = line.split_whitespace().nth(1).and_then(|s| s.split_once('/')).ok_or("invalid discovery response")?;
            let prefix = prefix.parse::<u8>().map_err(|e| e.to_string())?;
            if prefix > 32 { return Err("invalid discovery IPv4 prefix".into()); }
            found.push((ip.parse::<Ipv4Addr>().map_err(|e| e.to_string())?, prefix));
        }
        links.push(found);
    }
    let port = std::env::var("TP_PORT").unwrap_or_else(|_| "29470".into()).parse::<u16>().map_err(|e| e.to_string())?;
    if port == 0 { return Err("TP_PORT must be nonzero".into()); }
    let mut listen = vec![format!("{}:{port}", Ipv4Addr::LOCALHOST); world];
    let mut next = listen.clone();
    let mut aux_listen = listen.clone(); let mut aux_next = listen.clone();
    let mut dual = world > 1 && !local && std::env::var("TP_DUAL_RAIL").as_deref() != Ok("0");
    let mut path = match std::env::var("TP_TOPOLOGY").as_deref() {
        Ok("path") => true, Ok("ring") | Err(_) => false, _ => return Err("TP_TOPOLOGY must be ring or path".into()),
    };
    if world > 1 {
        for rank in 0..world {
            let peer = (rank + 1) % world;
            if peer == 0 && path { continue; }
            let mut remotes = if hosts[rank] == hosts[peer] { vec![Ipv4Addr::LOCALHOST] }
            else { links[rank].iter().flat_map(|&(a, p)| links[peer].iter().filter_map(move |&(b, q)| {
                let mask = if p == 0 { 0 } else { u32::MAX << (32 - p) };
                (p == q && u32::from(a) & mask == u32::from(b) & mask && a != b).then_some(b)
            })).collect::<Vec<_>>() };
            remotes.sort_unstable(); remotes.dedup();
            let Some(&remote) = remotes.first() else {
                if peer == 0 { path = true; continue; }
                return Err(format!("no direct QSFP subnet for {} -> {}; list hosts in physical neighbour order", hosts[rank], hosts[peer]));
            };
            let peer_port = port.checked_add(peer as u16).ok_or("port overflow")?;
            next[rank] = format!("{remote}:{peer_port}"); listen[peer] = next[rank].clone();
            if dual {
                if let Some(remote) = remotes.get(1) {
                    let p = peer_port.checked_add(512).ok_or("auxiliary port overflow")?;
                    aux_next[rank] = format!("{remote}:{p}"); aux_listen[peer] = aux_next[rank].clone();
                } else { dual = false; }
            }
        }
    }
    let shared = (0..world).any(|rank| world > 1 && hosts[rank] == hosts[(rank + 1) % world] && !(path && rank + 1 == world));
    let topology = if path { "path" } else { "ring" };
    eprintln!("launch world={world} topology={topology} rails={} hosts={}", if dual { 2 } else { 1 }, hosts.join(","));
    let tail = args[1..].iter().map(|s| quote(s)).collect::<Vec<_>>().join(" ");
    let run_id = format!("{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos());
    let mut children = Children(Vec::new(), args[0] == "tp-serve");
    for (rank, host) in hosts.iter().enumerate() {
        let mut environment = ["TP_CPU", "TP_BUSY_POLL_US", "TP_DEADLINE_UNIX", "TP_SERVE_HYBRID", "TP_SERVE_GENERATION_TIMEOUT_MS", "TP_SERVE_MAX_ACTIVE", "TP_SERVE_SEQUENCE_PREFILL", "SPARK_KV_BF16", "SPARK_PROFILE", "TP_RDMA", "TP_RDMA_SMALL",
            "SEQPAR_RDMA", "SEQPAR_ATTENTION_WEIGHT", "SEQPAR_OVERLAP", "SEQPAR_SERIAL_COMPUTE", "SEQPAR_MEMORY_BYTES", "SPARK_ADAPTIVE_PREFILL"]
            .into_iter().filter_map(|key| std::env::var(key).ok().map(|value| {
                let same_host = hosts[..rank].iter().filter(|h| *h == host).count();
                let value = if key == "TP_CPU" { value.parse::<usize>().ok().and_then(|n| n.checked_add(same_host)).map(|n| n.to_string()).unwrap_or(value) } else { value };
                format!("export {key}={}; ", quote(&value))
            })).collect::<String>();
        if std::env::var_os("SEQPAR_RDMA").is_none() {
            if let Ok(mode) = std::env::var("TP_RDMA") {
                environment.push_str(&format!("export SEQPAR_RDMA={}; ", quote(&mode)));
            }
        }
        if shared { environment.push_str("export TP_RDMA=0 SEQPAR_RDMA=0; "); }
        environment.push_str("unset TP_AUX_LISTEN TP_AUX_NEXT; ");
        if dual { environment.push_str(&format!("export TP_AUX_LISTEN={} TP_AUX_NEXT={}; ", quote(&aux_listen[rank]), quote(&aux_next[rank]))); }
        // SSH encrypts stdin; keep the secret out of process arguments and logs.
        let authenticate = if secret.is_some() { "IFS= read -r SEQPAR_CONTROL_SECRET || exit 1; export SEQPAR_CONTROL_SECRET; " } else { "" };
        let command = format!("{authenticate}{environment}export TP_TOPOLOGY={topology} TP_WATCH_STDIN=1 TP_RUN_ID={run_id}; cd {directory} && exec target/release/{} {rank} {world} {} {} {tail}", args[0], quote(&listen[rank]), quote(&next[rank]));
        let mut child = ssh(host, &command).stdin(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        if let Some(secret) = &secret {
            let stdin = child.stdin.as_mut().ok_or("missing rank stdin")?;
            stdin.write_all(secret).and_then(|_| stdin.write_all(b"\n")).map_err(|e| e.to_string())?;
        }
        children.0.push(child);
    }
    loop {
        let mut running = false;
        for child in &mut children.0 {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(status) if !status.success() => return Err("rank failed; stopping remaining ranks".into()),
                Some(_) => (), None => running = true,
            }
        }
        if !running { return Ok(()); }
        std::thread::sleep(Duration::from_millis(20));
    }
}
struct Children(Vec<Child>, bool);
impl Drop for Children {
    fn drop(&mut self) {
        if self.1 {
            // A failed peer closes the collective socket. Give rank zero time
            // to send its HTTP error before the stdin watcher forces exit.
            let deadline = Instant::now() + Duration::from_secs(1);
            while self.0.iter_mut().any(|c| matches!(c.try_wait(), Ok(None))) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        for child in &mut self.0 { drop(child.stdin.take()); }
        let deadline = Instant::now() + Duration::from_secs(2);
        while self.0.iter_mut().any(|c| matches!(c.try_wait(), Ok(None))) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        for child in &mut self.0 {
            if matches!(child.try_wait(), Ok(None)) { let _ = child.kill(); }
            let _ = child.wait();
        }
    }
}
fn main() { if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); } }
