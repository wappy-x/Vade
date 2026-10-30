//! Extra preflight checks share the launcher's runner and SSH options.
use crate::{checks::Report, cli::{self, Command}, config::Machine, runner::{self, Runner}, text};
use std::path::Path;

mod model;
pub(crate) mod fabric;

/// Identify the command that is in use, without starting another copy.
pub(crate) fn binary(report: &mut Report) -> Result<(), String> {
    report.check("vade binary", std::env::current_exe().map_err(|_| "Cannot find the running vade binary.".to_string()),
        |path| format!("{}; vade {}", path.display(), env!("CARGO_PKG_VERSION")),
        "Install vade and put its folder on PATH, then run vade doctor.")?;
    Ok(())
}

/// Show the four settings that affect the first launch.
/// `cache`: None when the switch is unset and the topology is not known.
pub(crate) fn defaults(report: &mut Report, context: u32, output: u32, mtp: Option<u32>, cache: Result<Option<bool>, String>) -> Result<(), String> {
    generation_defaults(report, context, output, mtp)?;
    report.check("default prefix cache", cache, |enabled| match enabled { Some(true) => "on", Some(false) => "off", None => UNSET_CACHE }.into(),
        "Set VADE_FLASHNEXT_PREFIX_CACHE to 0 or 1, then run vade doctor.")?;
    Ok(())
}

pub(crate) fn generation_defaults(report: &mut Report, context: u32, output: u32, mtp: Option<u32>) -> Result<(), String> {
    report.ok("default context", &format!("{context} tokens"))?;
    report.ok("default output", &format!("{output} tokens"))?;
    report.ok("default drafting", &match mtp {
        Some(0) => "off (MTP 0)".into(),
        Some(depth) => format!("on (MTP {depth})"),
        None => "on (engine default)".into(),
    })?;
    Ok(())
}

/// Read the same MTP setting as `vade up`.
pub(crate) fn start_options() -> Result<cli::UpOptions, String> {
    match cli::parse(&["up".into()])?.command {
        Command::Up(options) => Ok(options),
        _ => unreachable!(),
    }
}

/// What the engine does with the prefix cache switch unset.
const UNSET_CACHE: &str = "unset: on for one Spark and the tensor split, off on the layer chain";

/// The retained-prefix setting passed to the engine (None: unset, see UNSET_CACHE).
pub(crate) fn prefix_cache() -> Result<Option<bool>, String> {
    match std::env::var("VADE_FLASHNEXT_PREFIX_CACHE") {
        Ok(value) if value == "1" => Ok(Some(true)),
        Ok(value) if value == "0" => Ok(Some(false)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        _ => Err("VADE_FLASHNEXT_PREFIX_CACHE must be 0 or 1.".into()),
    }
}

pub(crate) fn local_model(report: &mut Report, folder: &Path) -> Result<(), String> { model::local(report, folder) }

/// The prerequisite sample and ownership state already read by the controller.
#[derive(Clone, Copy)]
pub(crate) struct RemoteState {
    pub owned: bool,
    pub memory: Option<(u64, u64)>,
    pub split: bool,
    /// The prefix cache default for the selected topology, when known (Session::prefix_restore).
    pub cache_default: Option<bool>,
}

/// One address and TCP port that this rank must bind.
pub(crate) struct Port { pub number: u16, pub address: String }
impl Port {
    pub(crate) fn new(address: &str, number: u16) -> Self {
        let address = address.parse::<std::net::IpAddr>().map(|address| address.to_canonical().to_string()).unwrap_or_else(|_| address.into());
        Self { number, address }
    }
    pub(crate) fn http(number: u16) -> Self { Self::new("127.0.0.1", number) }
}

/// Check files and listener ports through the same SSH path as `vade up`.
/// No helper binary or installed engine is needed on the Spark.
pub(crate) fn machine<R: Runner>(report: &mut Report, runner: &mut R, machine: &Machine, ports: &[Port], dry: bool, state: RemoteState) -> Result<(), String> {
    let name = |check: &str| format!("{} {check}", machine.name);
    if dry {
        for check in ["model folder", "config.json", "tokenizer.json", "safetensors", "model index", "model disk", "install disk", "default prefix cache"] {
            report.plan(&name(check), "read file metadata through the configured SSH destination")?;
        }
        for port in ports { report.plan(&name(&format!("port {}", port.number)), &format!("check TCP listeners at {} on the Spark", port.address))?; }
        report.plan(&name("model memory"), "compare the prerequisite memory sample with the model file sizes")?;
        return Ok(());
    }
    let command = format!("{}\n{}", model::script(&machine.model_path), extra_script(machine, ports));
    let result = runner.run(&runner::ssh(&machine.ssh, command, &name("doctor files")))
        .and_then(|output| if output.success { text::parse(&output.stdout) } else { Err("The file and port check did not finish over SSH.".into()) });
    let mut fields = match result {
        Ok(fields) => fields,
        Err(error) => {
            report.fail(&name("file and port check"), &error,
                &format!("Run ssh {} true, then run vade doctor.", machine.ssh))?;
            return Ok(());
        }
    };
    model::remote(runner, machine, &mut fields);
    match fields.get("prefix_cache").map(String::as_str) {
        Some("0") => report.ok(&name("default prefix cache"), "off in the Spark SSH session")?,
        Some("1") => report.ok(&name("default prefix cache"), "on in the Spark SSH session")?,
        Some("unset") => report.ok(&name("default prefix cache"), match state.cache_default {
            Some(true) => "unset in the Spark SSH session: on for this topology",
            Some(false) => "unset in the Spark SSH session: off on the layer chain",
            None => "unset in the Spark SSH session: on for one Spark and the tensor split, off on the layer chain",
        })?,
        _ => report.fail(&name("default prefix cache"), "The Spark prefix cache setting is invalid or unavailable.",
            "Set VADE_FLASHNEXT_PREFIX_CACHE to 0 or 1 in the Spark SSH session, then run vade doctor.")?,
    }
    model::rows(report, &machine.name, &machine.model_path, &fields)?;
    let memory_name = name("model memory");
    match state.memory {
        None => report.fail(&memory_name, "The Spark free memory sample is unavailable.",
            "Check /proc/meminfo through the configured SSH destination, then run vade doctor.")?,
        Some((total, available)) if total == 0 || available > total => report.fail(&memory_name,
            "The Spark reported invalid total or available memory.",
            "Check /proc/meminfo through the configured SSH destination, then run vade doctor.")?,
        Some((total, available)) if state.owned => report.skip(&memory_name,
            &format!("{}; a recorded engine can hold this memory; use vade status", crate::detect::memory_summary(total, available)))?,
        // A full checkpoint size does not give each rank's tensor share or
        // context cost. Only engine startup can check those bounds.
        Some((total, available)) if state.split => {
            if available == 0 {
                report.fail(&memory_name, "No free memory is available on the Spark.",
                    "Close other GPU applications, then run vade doctor before vade up.")?;
            } else {
                report.ok(&memory_name, &format!("{}; startup checks the tensor share and context", crate::detect::memory_summary(total, available)))?;
            }
        }
        Some((total, available)) => match model::bytes(&fields) {
            Ok(bytes) => {
                report.check(&memory_name, crate::detect::check_memory(total, available, bytes),
                    |_| format!("{}; checkpoint {} bytes fits with working room before context", crate::detect::memory_summary(total, available), bytes),
                    "Close other GPU applications, then run vade doctor before vade up.")?;
            }
            Err(_) => report.skip(&memory_name, "needs a complete safetensors inventory")?,
        },
    }
    for (key, label, path) in [("model", "model disk", &machine.model_path), ("install", "install disk", &machine.stage_dir)] {
        let bytes = fields.get(&format!("{key}_disk_kb")).and_then(|value| value.trim().parse::<u64>().ok()).and_then(|kb| kb.checked_mul(1024));
        match bytes {
            Some(bytes) if bytes >= 1 << 30 => report.ok(&name(label), &format!("{bytes} bytes free on the volume for {path}"))?,
            Some(bytes) => report.fail(&name(label), &format!("Only {bytes} bytes are free on the volume for {path}."),
                "Free at least 1 GiB on that volume, then run vade doctor.")?,
            None => report.fail(&name(label), "Cannot read free disk space.", "Check the model and install folder permissions, then run vade doctor.")?,
        }
    }
    for required in ports {
        let port = required.number;
        match fields.get(&format!("port_{port}")).map(String::as_str) {
            Some("free") => report.ok(&name(&format!("port {port}")), "free on the Spark")?,
            Some("busy") if state.owned => report.skip(&name(&format!("port {port}")), "a recorded engine can hold this port; use vade status")?,
            Some("busy") => report.fail(&name(&format!("port {port}")), "A program listens on this port.",
                "Close that program or change the port in the config, then run vade doctor.")?,
            Some("unverified") => report.fail(&name(&format!("port {port}")), "Cannot confirm whether the IPv6 listener also holds IPv4.",
                "Run ss -ltn -e on the Spark and check the listener's v6only setting, then run vade doctor.")?,
            _ => report.fail(&name(&format!("port {port}")), "Cannot read TCP listeners on the Spark.",
                "Install the ss command on the Spark, then run vade doctor.")?,
        }
    }
    Ok(())
}

fn extra_script(machine: &Machine, ports: &[Port]) -> String {
    let mut script = String::from("vade_disk() {\n  vade_disk_path=$2\n  while ! test -d \"$vade_disk_path\"; do\n    case \"$vade_disk_path\" in /) break;; */*) vade_disk_path=${vade_disk_path%/*}; test -n \"$vade_disk_path\" || vade_disk_path=/;; *) vade_disk_path=.; break;; esac\n  done\n  vade_free=$(df -Pk \"$vade_disk_path\" 2>/dev/null | awk 'NR == 2 {print $4}')\n  printf '%s_disk_kb=%s\\n' \"$1\" \"$vade_free\"\n}\n");
    for (key, path) in [("model", &machine.model_path), ("install", &machine.stage_dir)] {
        script += &format!("vade_disk {key} {}\n", runner::remote_path(path));
    }
    script += "if vade_listeners=$(ss -H -ltn -e 2>/dev/null); then\n";
    for required in ports {
        let port = required.number;
        script += &format!("  vade_bind={}\n", runner::quote(&required.address));
        script += "  case \"$vade_bind\" in *:*) ;; *[!0-9.]*|'') vade_bind=$(getent ahostsv4 \"$vade_bind\" 2>/dev/null | awk 'NR == 1 {print $1}');; esac\n";
        // Extended ss output identifies IPv6-only wildcard listeners. If it
        // omits that flag, report the uncertainty instead of a false conflict.
        script += &format!("  if test -z \"$vade_bind\"; then printf 'port_{port}=unknown\\n'; else printf '%s\\n' \"$vade_listeners\" | awk -v port={port} -v bind=\"$vade_bind\" '$4 ~ (\":\" port \"$\") {{addr=$4; sub(/:[0-9]+$/, \"\", addr); sub(/^\\[/, \"\", addr); sub(/\\]$/, \"\", addr); if (addr ~ /^::ffff:[0-9]+\\./) sub(/^::ffff:/, \"\", addr); ipv4=(index(addr, \":\") == 0); bind4=(index(bind, \":\") == 0); wildcard=(addr == \"*\" || addr == \"::\"); if (bind4 && wildcard) {{if ($0 ~ /v6only:0/) found=1; else if ($0 !~ /v6only:1/) uncertain=1}} else if (addr == bind || wildcard || bind == \"::\" || (ipv4 && bind4 && (addr == \"0.0.0.0\" || bind == \"0.0.0.0\"))) found=1}} END {{print \"port_\" port \"=\" (found ? \"busy\" : uncertain ? \"unverified\" : \"free\")}}'; fi\n");
    }
    script += "else\n";
    for port in ports { script += &format!("  printf 'port_{}=unknown\\n'\n", port.number); }
    script += "fi\n";
    script += "if test -z \"${VADE_FLASHNEXT_PREFIX_CACHE+set}\"; then printf 'prefix_cache=unset\\n'; else case \"$VADE_FLASHNEXT_PREFIX_CACHE\" in 0) printf 'prefix_cache=0\\n';; 1) printf 'prefix_cache=1\\n';; *) printf 'prefix_cache=invalid\\n';; esac; fi\n";
    script
}

#[cfg(test)]
mod tests;
