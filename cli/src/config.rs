use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::cli::{UpOptions, identifier};

pub const MAX_FP4_SPARKS: usize = 24;
/// Independent one-Spark servers behind one local gateway.
pub const MAX_REPLICAS: usize = 16;

#[derive(Clone, Debug)]
pub struct Config {
    pub engine_source: PathBuf,
    pub prebuilt_dir: Option<PathBuf>,
    pub harness_source: PathBuf,
    pub cli_source: PathBuf,
    pub stage_dir: PathBuf,
    pub stage_root: PathBuf,
    pub activation: PathBuf,
    pub state_dir: PathBuf,
    pub model: String,
    pub model_repo: String,
    pub model_revision: String,
    pub local_port: u16,
    pub serve_port: u16,
    pub tp_port: u16,
    pub health_timeout_secs: u64,
    pub build_manifest: PathBuf,
    pub machines: Vec<Machine>,
}

#[derive(Clone, Debug)]
pub struct Machine {
    pub name: String,
    pub ssh: String,
    /// An already-running independent engine, used only by `vade verify`.
    pub verify_url: Option<String>,
    pub fabric: BTreeMap<String, String>,
    pub model_path: String,
    pub engine_source: String,
    pub cli_source: String,
    pub stage_dir: String,
    pub state_dir: String,
}

pub fn config_path() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("VADE_CONFIG") {
        if path.is_empty() {
            return Err("VADE_CONFIG is empty. Set it to your config file, or unset it.".into());
        }
        return absolute_local(PathBuf::from(path));
    }
    absolute_local(user_config_dir()?.join("config"))
}

fn user_config_dir() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("XDG_CONFIG_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join("vade"));
    }
    let home = home()?;
    if cfg!(target_os = "macos") {
        Ok(home.join("Library/Application Support/vade"))
    } else {
        Ok(home.join(".config/vade"))
    }
}

fn home() -> Result<PathBuf, String> {
    env::var_os("HOME")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            "Your home directory is unavailable. Set HOME, or use absolute paths in VADE_CONFIG."
                .into()
        })
}

impl Config {
    pub fn validate_control_socket(&self) -> Result<(), String> {
        // OpenSSH runs inside state_dir; only the basename enters sockaddr_un.
        // Reserve 17 bytes for its temporary suffix and one for the NUL.
        let socket = crate::state::Session::socket_path(Path::new("."), "0000000000000000");
        if socket.as_os_str().len() > 86 {
            return Err("The generated SSH control socket name is too long. Update Vade.".into());
        }
        Ok(())
    }

    pub fn load() -> Result<Self, String> {
        let path = config_path()?;
        let text = fs::read_to_string(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                format!("Config is missing: {}. Copy cli/examples/fp4-local.conf for FP4-Local (or cli/examples/vade.conf for legacy FP8), edit the placeholders, and set VADE_CONFIG. See INSTALL.md.", crate::text::quoted(&path.to_string_lossy()))
            } else {
                format!("Cannot read config {}. Check its path and permissions.", crate::text::quoted(&path.to_string_lossy()))
            }
        })?;
        let config = Self::parse(&text, &path)?;
        if config.validate_model_revision().is_err() {
            crate::output::stderr_line(format_args!("Warning: model_revision is missing or mutable; inspection and shutdown remain available. Set a verified immutable revision before vade install or vade up; see INSTALL.md."));
        }
        Ok(config)
    }

    pub fn validate_model_revision(&self) -> Result<(), String> {
        if crate::text::hex(&self.model_revision, 40)
            || self.model_revision.strip_prefix("local-sha256-").is_some_and(|hash| crate::text::hex(hash, 64))
        {
            return Ok(());
        }
        let start = if self.model.ends_with("-fp8") { "vade up --fp8-legacy" } else { "vade up" };
        Err(format!("Set model_revision to a verified immutable 40-hex publisher revision or local-sha256-<64-hex manifest digest>; missing values, placeholders such as REPLACE_WITH_VERIFIED_IMMUTABLE_REVISION, and mutable tags are refused for installation and startup. See INSTALL.md, then run {start}."))
    }

    pub fn parse(text: &str, path: &Path) -> Result<Self, String> {
        let base = path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let (globals, rows) = parse_sections(text)?;
        let mut globals = Values::new(globals);
        let state_dir = local_path(
            &globals.take("state_dir").unwrap_or_else(|| "state".into()),
            base,
        )?;
        let stage_root = local_path(
            &globals.take("stage_dir").unwrap_or_else(|| "bin".into()),
            base,
        )?;
        let activation = match globals.take("build_manifest") {
            Some(value) => local_path(&value, base)?,
            None => state_dir.join("build.conf"),
        };
        let model = globals.take("model").unwrap_or_else(|| "qwen3.8-flash-next-fp4-local".into());
        let model_repo = globals.take("model_repo").unwrap_or_else(|| if model == "qwen3.8-flash-next-fp8" {
            "Qwen/Qwen3.8-Flash-Next-FP8".into()
        } else { "nvidia/Qwen3.8-Flash-Next-FP4-Local".into() });
        let config = Self {
            prebuilt_dir: globals.take("prebuilt_dir").map(|p| local_path(&p, base)).transpose()?,
            engine_source: local_path(
                &globals
                    .take("engine_source")
                    .unwrap_or_else(|| "engine-source".into()),
                base,
            )?,
            harness_source: local_path(
                &globals
                    .take("harness_source")
                    .unwrap_or_else(|| "harness-source".into()),
                base,
            )?,
            cli_source: local_path(
                &globals
                    .take("cli_source")
                    .unwrap_or_else(|| "cli-source".into()),
                base,
            )?,
            stage_dir: stage_root.clone(),
            stage_root,
            activation: activation.clone(),
            model,
            model_repo,
            model_revision: globals
                .take("model_revision")
                .unwrap_or_default(),
            local_port: port(&mut globals, "local_port", 18080)?,
            serve_port: port(&mut globals, "serve_port", 18080)?,
            tp_port: integer(&mut globals, "tp_port", 29960, 1024, 65534)? as u16,
            health_timeout_secs: integer(&mut globals, "health_timeout_secs", 360, 1, 3600)?,
            build_manifest: activation,
            state_dir,
            machines: Vec::new(),
        };
        config.validate_control_socket()?;
        globals.finish("global")?;
        if !safe_model(&config.model) || !safe_model_repo(&config.model_repo) {
            return Err(
                "Use a plain model name and an official owner/model repository in the config."
                    .into(),
            );
        }
        if !config.model_revision.is_empty()
            && (!safe_model(&config.model_revision) || matches!(config.model_revision.as_str(), "." | ".."))
        {
            return Err("Use a plain model_revision value in the config; see INSTALL.md.".into());
        }
        if config.serve_port == config.tp_port || config.serve_port == config.tp_port + 1 {
            return Err("Set serve_port apart from tp_port and tp_port + 1 in the config.".into());
        }
        let mut config = config;
        for (name, values) in rows {
            let mut values = Values::new(values);
            let ssh = values.take("ssh").ok_or_else(|| {
                format!("Machine {name} needs ssh = your-ssh-alias or user@host in the config.")
            })?;
            if !ssh_destination(&ssh) {
                return Err(format!(
                    "Machine {name}: set ssh to an SSH alias, host name or IPv4 address, optionally as user@host, without options or shell symbols."
                ));
            }
            let fabric_keys: Vec<String> = values
                .0
                .keys()
                .filter(|key| key.starts_with("fabric."))
                .cloned()
                .collect();
            let mut fabric = BTreeMap::new();
            for key in fabric_keys {
                let peer = key.strip_prefix("fabric.").unwrap();
                let address = values.take(&key).unwrap();
                if !identifier(peer) || peer == name || !fabric_address(&address) {
                    return Err(format!(
                        "Machine {name}: set fabric.<peer> to its own address on that link, without a port."
                    ));
                }
                fabric.insert(peer.into(), address);
            }
            let machine = Machine {
                verify_url: values.take("verify_url"),
                model_path: remote_path(
                    values
                        .take("model_path")
                        .unwrap_or_else(|| format!("~/models/{}", config.model_repo)),
                    &name,
                    "model_path",
                )?,
                engine_source: remote_path(
                    values
                        .take("engine_source")
                        .unwrap_or_else(|| "~/vade-source".into()),
                    &name,
                    "engine_source",
                )?,
                cli_source: remote_path(
                    values
                        .take("cli_source")
                        .unwrap_or_else(|| "~/vade-cli-source".into()),
                    &name,
                    "cli_source",
                )?,
                stage_dir: remote_path(
                    values
                        .take("stage_dir")
                        .unwrap_or_else(|| "~/vade-stage".into()),
                    &name,
                    "stage_dir",
                )?,
                state_dir: remote_path(
                    values
                        .take("state_dir")
                        .unwrap_or_else(|| "~/.local/state/vade".into()),
                    &name,
                    "state_dir",
                )?,
                name,
                ssh,
                fabric,
            };
            values.finish(&format!("machine {}", machine.name))?;
            config.machines.push(machine);
        }
        if config.machines.is_empty() {
            return Err(
                "Add at least one [machine name] section to your config. See cli/examples/fp4-local.conf and INSTALL.md."
                    .into(),
            );
        }
        let mut aliases = BTreeSet::new();
        for machine in &config.machines {
            if !aliases.insert(&machine.ssh) {
                return Err("Give each machine a distinct ssh destination in the config.".into());
            }
            for peer in machine.fabric.keys() {
                if !config.machines.iter().any(|machine| machine.name == *peer) {
                    return Err(format!(
                        "Machine {} refers to an unknown fabric peer. Add [machine {peer}] or fix that link.",
                        machine.name
                    ));
                }
            }
        }
        Ok(config)
    }

    // Resolve the activation record once per command. All reads use this generation.
    pub fn resolve_generation(&mut self) -> Result<(), String> {
        crate::text::reject_symlinks(&self.activation)?;
        let data = match fs::read_to_string(&self.activation) {
            Ok(data) => data,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_) => return Err("Cannot read install activation. Check its permissions.".into()),
        };
        let fields = crate::text::parse(&data)?;
        if let Some(id) = fields.get("generation") {
            if fields.get("activation_version").map(String::as_str) != Some("1")
                || !crate::text::hex(id, 24)
            {
                return Err(
                    "Invalid install generation. Leave the activation record in place.".into(),
                );
            }
            let generation = self.stage_root.join("builds").join(id);
            crate::text::reject_symlinks(&generation)?;
            self.build_manifest = generation.join("build.conf");
            crate::state::Manifest::load(&self.build_manifest)?;
            self.stage_dir = generation;
        }
        Ok(())
    }

    pub fn selected(&self, options: &UpOptions) -> Result<Vec<Machine>, String> {
        let selected = self.selected_machines(options)?;
        // Independent replicas never talk to each other, so they need no fabric links.
        if !options.replicas { self.check_links(&selected)?; }
        Ok(selected)
    }
    fn selected_machines(&self, options: &UpOptions) -> Result<Vec<Machine>, String> {
        if options.fp4_local && self.model.ends_with("-fp8") {
            return Err("Your config selects legacy FP8. Run vade up --fp8-legacy, or select the FP4-Local checkpoint in VADE_CONFIG and run vade up.".into());
        }
        if !options.fp4_local && self.model.ends_with("-fp4-local") {
            return Err("Your config selects FP4-Local. Run vade up without --fp8-legacy, or select a legacy FP8 config with VADE_CONFIG.".into());
        }
        let requested = if options.hosts.is_empty() {
            options.sparks.unwrap_or(if options.fp4_local { 1 } else { 2 })
        } else { options.hosts.len() };
        if options.fp4_local && !(1..=MAX_FP4_SPARKS).contains(&requested) {
            return Err("FP4-Local supports 1..24 Sparks. Choose --sparks 1..24 or list that many distinct --hosts.".into());
        }
        let selected = if !options.hosts.is_empty() {
            options.hosts.iter().map(|name| {
                self.machines.iter().find(|machine| machine.name == *name).cloned()
                    .ok_or_else(|| format!("Machine {name} is not in your config. Check --hosts or add its machine section."))
            }).collect::<Result<Vec<_>, _>>()?
        } else {
            let count = options.sparks.unwrap_or(if options.fp4_local { 1 } else { 2 });
            if count > self.machines.len() {
                return Err(format!(
                    "Your config has {} machine{}. Choose fewer supported machines with --sparks or add machine sections, then retry {}. For one Spark, select an FP4-Local config with VADE_CONFIG and run vade up.",
                    self.machines.len(), if self.machines.len() == 1 { "" } else { "s" },
                    if options.fp4_local { "vade up" } else { "vade up --fp8-legacy" }
                ));
            }
            self.machines[..count].to_vec()
        };
        if !options.fp4_local && !matches!(selected.len(), 2 | 4 | 6) {
            return Err("Legacy FP8 needs 2, 4, or 6 machines. Run vade up --fp8-legacy --sparks 2, 4, or 6, or list that many --hosts. For one Spark, select an FP4-Local config with VADE_CONFIG and run vade up.".into());
        }
        let names: BTreeSet<&str> = selected
            .iter()
            .map(|machine| machine.name.as_str())
            .collect();
        if names.len() != selected.len() {
            return Err("Choose each machine only once in --hosts.".into());
        }
        Ok(selected)
    }
    fn check_links(&self, selected: &[Machine]) -> Result<(), String> {
        for link in selected.windows(2) {
            let a = &link[0];
            let b = &link[1];
            let address_a = a.fabric.get(&b.name);
            let address_b = b.fabric.get(&a.name);
            if address_a.is_none() || address_b.is_none() {
                return Err(format!(
                    "Add fabric.{} under {} and fabric.{} under {} for their link, or reorder --hosts to follow connected machines.",
                    b.name, a.name, a.name, b.name
                ));
            }
            if address_a == address_b {
                return Err(format!(
                    "Machines {} and {} need different addresses on their fabric link. Fix those fabric entries.",
                    a.name, b.name
                ));
            }
        }
        Ok(())
    }
}

type Section = BTreeMap<String, String>;
type MachineRows = Vec<(String, Section)>;

fn parse_sections(text: &str) -> Result<(Section, MachineRows), String> {
    let mut global = Section::new();
    let mut machines: MachineRows = Vec::new();
    let mut current: Option<usize> = None;
    let mut global_seen = false;
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let error = |message: &str| {
            format!(
                "Config line {}: {message} Fix that line and try again.",
                index + 1
            )
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.chars().any(char::is_control) {
            return Err(error("use plain text without control characters."));
        }
        if line.starts_with('[') {
            let section = line
                .strip_prefix('[')
                .and_then(|line| line.strip_suffix(']'))
                .ok_or_else(|| error("close the section with ]."))?
                .trim();
            if section == "global" {
                if global_seen {
                    return Err(error("keep only one [global] section."));
                }
                global_seen = true;
                current = None;
            } else if let Some(name) = section.strip_prefix("machine ").map(str::trim) {
                if !identifier(name) || machines.iter().any(|(existing, _)| existing == name) {
                    return Err(error(
                        "use a unique [machine name] section with a plain name.",
                    ));
                }
                machines.push((name.into(), Section::new()));
                current = Some(machines.len() - 1);
            } else {
                return Err(error("use [global] or [machine name]."));
            }
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| error("write key = value."))?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty()
            || !key
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-'))
            || value.is_empty()
        {
            return Err(error("use a plain key and a nonempty value."));
        }
        let value = if value.starts_with('"') || value.starts_with('\'') {
            if value.len() < 2 || value.as_bytes().first() != value.as_bytes().last() {
                return Err(error("close the quoted value."));
            }
            &value[1..value.len() - 1]
        } else {
            value
        };
        if value.is_empty() || value.chars().any(char::is_control) {
            return Err(error("use a nonempty value without control characters."));
        }
        let target = match current {
            Some(index) => &mut machines[index].1,
            None => &mut global,
        };
        if target.insert(key.into(), value.into()).is_some() {
            return Err(error("use each key only once in a section."));
        }
    }
    Ok((global, machines))
}

struct Values(Section);

impl Values {
    fn new(values: Section) -> Self {
        Self(values)
    }
    fn take(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }
    fn finish(self, section: &str) -> Result<(), String> {
        match self.0.first_key_value() {
            Some((key, _)) => Err(format!(
                "Unknown config key {key} in {section}. Check examples/vade.conf for supported keys."
            )),
            None => Ok(()),
        }
    }
}

fn local_path(value: &str, base: &Path) -> Result<PathBuf, String> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err("Use a nonempty local path without control characters in the config.".into());
    }
    let path = if value == "~" {
        home()?
    } else if let Some(suffix) = value.strip_prefix("~/") {
        home()?.join(suffix)
    } else {
        if value.starts_with('~') {
            return Err(
                "Use ~/ for your home directory, or an absolute local path, in the config.".into(),
            );
        }
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            base.join(path)
        }
    };
    absolute_local(path)
}

fn absolute_local(path: PathBuf) -> Result<PathBuf, String> {
    let path = if path.is_absolute() {
        path
    } else {
        env::current_dir()
            .map_err(
                |_| "Cannot read the current folder. Change to an existing folder, then try again.",
            )?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

pub fn remote_path(value: String, machine: &str, key: &str) -> Result<String, String> {
    if value.is_empty()
        || value == "/"
        || value == "~"
        || value == "."
        || !(value.starts_with('/') || value.starts_with("~/"))
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b'/' | b'~'))
        || value.split('/').any(|part| part == "..")
        || (value.contains('~') && (!value.starts_with("~/") || value[1..].contains('~')))
    {
        return Err(format!(
            "Machine {machine}: set {key} to a plain absolute or home-relative path without shell symbols or .. ."
        ));
    }
    Ok(value)
}

fn safe_alias(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

/// An SSH destination: an ~/.ssh/config alias, host name or IPv4 address,
/// optionally preceded by `user@`. Never an option or shell text.
pub fn ssh_destination(value: &str) -> bool {
    let (user, host) = value.split_once('@').map_or((None, value), |(user, host)| (Some(user), host));
    user.is_none_or(|user| {
        !user.is_empty()
            && user.len() <= 32
            && (user.as_bytes()[0].is_ascii_alphanumeric() || user.starts_with('_'))
            && user.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
    }) && safe_alias(host)
}

pub fn fabric_address(value: &str) -> bool {
    value.parse::<std::net::IpAddr>().is_ok() || (!value.contains(':') && safe_alias(value))
}

fn safe_model(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

fn safe_model_repo(value: &str) -> bool {
    let mut parts = value.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(model), None) if safe_model(owner) && safe_model(model) && owner != "." && owner != ".." && model != "." && model != "..")
}

fn port(values: &mut Values, key: &str, default: u64) -> Result<u16, String> {
    Ok(integer(values, key, default, 1024, u16::MAX as u64)? as u16)
}

fn integer(
    values: &mut Values,
    key: &str,
    default: u64,
    min: u64,
    max: u64,
) -> Result<u64, String> {
    let Some(value) = values.take(key) else {
        return Ok(default);
    };
    if value.bytes().all(|c| c.is_ascii_digit()) {
        if let Ok(number) = value.parse::<u64>() {
            if (min..=max).contains(&number) {
                return Ok(number);
            }
        }
    }
    Err(format!(
        "Set config {key} to a whole number from {min} to {max}."
    ))
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    const PAIR: &str = "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nstate_dir = /state\nengine_source = engine\nharness_source = harness\ncli_source = cli\n[machine machine-a]\nssh = machine-a-alias\nfabric.machine-b = fabric-a\nmodel_path = ~/models/model\n[machine machine-b]\nssh = machine-b-alias\nfabric.machine-a = fabric-b\nmodel_path = ~/models/model\n";
    fn parse(text: &str) -> Result<Config, String> { Config::parse(text, Path::new("test-config/config")) }
    #[test]
    fn selections_keep_config_or_user_order() {
        let config = parse(PAIR).unwrap();
        assert_eq!(config.model, "qwen3.8-flash-next-fp4-local");
        assert_eq!(config.engine_source, env::current_dir().unwrap().join("test-config/engine"));
        assert_eq!(config.selected(&UpOptions::default()).unwrap()[0].name, "machine-a");
        let reversed = UpOptions { hosts: vec!["machine-b".into(), "machine-a".into()], ..UpOptions::default() };
        assert_eq!(config.selected(&reversed).unwrap()[0].name, "machine-b");
        assert!(config.selected(&UpOptions { sparks: Some(4), ..UpOptions::default() }).is_err());
    }
    #[test]
    fn reject_unknown_duplicate_and_shell_values() {
        for invalid in [
            PAIR.replace("ssh = machine-a-alias", "ssh = -oProxyCommand=bad"),
            PAIR.replace("~/models/model", "~/models/../model"),
            PAIR.replace("fabric-a", "fabric-a:9000"),
            PAIR.replace("harness_source = harness", "unsupported = value"),
            PAIR.replace("ssh = machine-a-alias", "ssh = machine-a-alias\nssh = duplicate"),
            PAIR.replace("ssh = machine-a-alias", "ssh = user@-oProxyCommand=bad"),
            PAIR.replace("ssh = machine-a-alias", "ssh = -user@machine-a"),
            PAIR.replace("ssh = machine-a-alias", "ssh = @machine-a"),
            PAIR.replace("ssh = machine-a-alias", "ssh = user@"),
            PAIR.replace("ssh = machine-a-alias", "ssh = user@host@machine-a"),
            PAIR.replace("ssh = machine-a-alias", "ssh = user name@machine-a"),
            PAIR.replace("ssh = machine-a-alias", "ssh = user@machine-a:22"),
            PAIR.replace("ssh = machine-a-alias", "ssh = user@$(command)"),
            PAIR.replace("machine-b-alias", "machine-a-alias"),
            PAIR.replace("fabric.machine-b", "fabric.unknown"),
        ] { assert!(parse(&invalid).is_err()); }
        for invalid in [
            "relative;command", "relative$(command)", "relative`command`", "~/../model", "~other/model", "relative\ncommand",
        ] { assert!(remote_path(invalid.into(), "machine-a", "model_path").is_err()); }
    }
    #[test]
    fn ssh_accepts_alias_host_ip_or_user_at_host_for_every_rank() {
        for (a, b) in [("machine-a-alias", "machine-b-alias"), ("user@192.0.2.10", "user@192.0.2.11"), ("user@127.0.0.1", "other_user@machine-b.example"), ("192.0.2.10", "_svc@machine-b")] {
            let config = parse(&PAIR.replace("ssh = machine-a-alias", &format!("ssh = {a}")).replace("ssh = machine-b-alias", &format!("ssh = {b}"))).unwrap();
            let selected = config.selected(&UpOptions { sparks: Some(2), ..UpOptions::default() }).unwrap();
            assert_eq!([selected[0].ssh.as_str(), selected[1].ssh.as_str()], [a, b]);
        }
        assert!(parse(&PAIR.replace("machine-a-alias", "user@machine-b-alias").replace("ssh = machine-b-alias", "ssh = user@machine-b-alias")).is_err());
    }
    #[test]
    fn revision_pin_is_required_for_startup_without_blocking_config_recovery() {
        for revision in ["", "main", "REPLACE_WITH_VERIFIED_IMMUTABLE_REVISION"] {
            let config = parse(&PAIR.replace("model_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", &if revision.is_empty() { String::new() } else { format!("model_revision = {revision}\n") })).unwrap();
            assert!(config.validate_model_revision().unwrap_err().contains("immutable"));
        }
        for revision in ["a".repeat(40), format!("local-sha256-{}", "b".repeat(64))] {
            assert!(parse(&PAIR.replace("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &revision)).unwrap().validate_model_revision().is_ok());
        }
    }
    #[test]
    fn disconnected_selection_needs_both_link_addresses() {
        let config = parse(&PAIR.replace("fabric.machine-b = fabric-a\n", "")).unwrap();
        assert!(config.selected(&UpOptions { sparks: Some(2), ..UpOptions::default() }).unwrap_err().contains("Add fabric."));
    }
    #[test]
    fn normalize_local_paths_and_validate_remote_endpoints() {
        let config = parse(&PAIR.replace("engine_source = engine", "engine_source = ./nested/../engine")).unwrap();
        assert_eq!(config.engine_source, env::current_dir().unwrap().join("test-config/engine"));
        assert!(remote_path("relative/model".into(), "machine-a", "model_path").is_err());
        assert!(remote_path("/models/model".into(), "machine-a", "model_path").is_ok());
        assert!(fabric_address(&std::net::Ipv6Addr::LOCALHOST.to_string()));
        assert!(parse(&PAIR.replace("[global]", "[global]\nserve_port = 29961")).is_err());
        for (bytes, valid) in [(81, true), (86, true), (87, true), (290, true)] { let state = format!("/{}", "s".repeat(bytes - 30)); assert_eq!(parse(&PAIR.replace("state_dir = /state", &format!("state_dir = {state}"))).is_ok(), valid); }
    }
}
