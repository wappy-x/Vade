//! Runtime QSFP discovery; no hostnames, addresses, or device names are compiled in.
use std::{fs, net::Ipv4Addr, process::Command};
use super::Result;

#[derive(Clone, Debug)]
pub struct Link { pub interface: String, pub address: Ipv4Addr, pub prefix: u8, pub mbps: u64 }
pub fn links() -> Result<Vec<Link>> {
    let mut result = Vec::new();
    for entry in fs::read_dir("/sys/class/net").map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let speed = fs::read_to_string(entry.path().join("speed")).unwrap_or_default().trim().parse::<u64>().unwrap_or(0);
        if speed < 100_000 { continue; }
        let output = Command::new("ip").args(["-o", "-4", "addr", "show", "dev", &name]).output().map_err(|e| e.to_string())?;
        if !output.status.success() { return Err("ip address discovery failed".into()); }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let words = line.split_whitespace().collect::<Vec<_>>();
            if let Some(i) = words.iter().position(|w| *w == "inet") {
                if let Some((address, prefix)) = words.get(i + 1).and_then(|s| s.split_once('/')) {
                    let prefix: u8 = prefix.parse().map_err(|_| "invalid interface prefix")?;
                    if prefix > 32 { return Err("invalid interface IPv4 prefix".into()); }
                    result.push(Link { interface: name.clone(), address: address.parse().map_err(|_| "invalid interface IPv4")?,
                        prefix, mbps: speed });
                }
            }
        }
    }
    result.sort_by(|a, b| a.interface.cmp(&b.interface));
    Ok(result)
}
