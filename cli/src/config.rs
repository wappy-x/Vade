//! The controller config file: a `[global]` section and one `[machine NAME]`
//! section per Spark. [`Config::load`] reads `VADE_CONFIG` or the user config
//! directory and validates every path, SSH alias, fabric address and port.
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::cli::{UpOptions, identifier};

/// The most Sparks one FP4-Local launch may chain.
pub const MAX_FP4_SPARKS: usize = 24;
/// Independent one-Spark servers behind one local gateway.
pub const MAX_REPLICAS: usize = 16;

/// The parsed controller config: source checkouts, staging and state folders,
/// the model, ports and the machines.
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
    /// Every `[group NAME]` section, in config order.
    pub groups: Vec<Group>,
    /// The group `VADE_GROUP` selects. It sets `state_dir` and `local_port`.
    pub group: Option<Group>,
    /// The folder whose lock serializes `up`, `down`, `install` and `update`
    /// for every group: the configured `state_dir`.
    pub lock_dir: PathBuf,
}

/// One `[machine NAME]` section: how to reach the Spark and where its files are.
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

/// One `[group NAME]` section: machines that run one server together. Each
/// group has its own ownership record and local port, so two groups can run at
/// the same time. Groups with the same `layout` must not share a machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub name: String,
    /// Machine names in rank order. Rank 0, the first, answers HTTP.
    pub hosts: Vec<String>,
    pub local_port: u16,
    /// The set of groups that runs at the same time; empty for the default set.
    pub layout: String,
}

impl Group {
    /// The local ports this group can hold: its gateway or tunnel port and,
    /// for replicas, one tunnel port per machine after it.
    fn ports(&self) -> std::ops::RangeInclusive<usize> {
        usize::from(self.local_port)..=usize::from(self.local_port) + self.hosts.len()
    }
    fn port_label(&self) -> String {
        format!("{}..{}", self.ports().start(), self.ports().end())
    }
}

/// Whether the environment selects the controller: `VADE_CONFIG`, or
/// `VADE_GROUP`, which names a group of the controller config.
pub fn controller_selected() -> bool {
    env::var_os("VADE_CONFIG").is_some() || env::var_os("VADE_GROUP").is_some()
}

/// `VADE_CONFIG`, or `config` in the user config directory (`$XDG_CONFIG_HOME/vade`,
/// else `~/Library/Application Support/vade` on macOS and `~/.config/vade`).
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
    /// Refuse a generated SSH control socket name that would not fit in `sockaddr_un`.
    pub fn validate_control_socket(&self) -> Result<(), String> {
        // OpenSSH runs inside state_dir; only the basename enters sockaddr_un.
        // Reserve 17 bytes for its temporary suffix and one for the NUL.
        let socket = crate::state::Session::socket_path(Path::new("."), "0000000000000000");
        if socket.as_os_str().len() > 86 {
            return Err("The generated SSH control socket name is too long. Update Vade.".into());
        }
        Ok(())
    }

    /// Read and parse the file at [`config_path`]; a missing file names the examples
    /// to copy.
    pub fn load() -> Result<Self, String> {
        let path = config_path()?;
        let text = fs::read_to_string(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound && env::var_os("VADE_CONFIG").is_none() {
                format!("Vade is not set up yet: there is no local setup and no controller config at {}. Run vade launch on a Spark, or copy cli/examples/fp4-local.conf there (or set VADE_CONFIG to it) to drive Sparks from this computer. See INSTALL.md.", crate::text::quoted(&path.to_string_lossy()))
            } else if error.kind() == std::io::ErrorKind::NotFound {
                format!("VADE_CONFIG names a missing file: {}. Check the path, or copy cli/examples/fp4-local.conf there (cli/examples/vade.conf for FP8) and edit the placeholders. See INSTALL.md.", crate::text::quoted(&path.to_string_lossy()))
            } else if error.kind() == std::io::ErrorKind::InvalidData {
                format!("Config {} is not UTF-8 text. Save it as plain UTF-8 text, then retry.", crate::text::quoted(&path.to_string_lossy()))
            } else {
                format!("Cannot read config {}. Check its path and permissions.", crate::text::quoted(&path.to_string_lossy()))
            }
        })?;
        let mut config = Self::parse(&text, &path)?;
        match env::var("VADE_GROUP") {
            Ok(name) if name.is_empty() => return Err("VADE_GROUP is empty. Set it to a [group name] from your config, or unset it.".into()),
            Ok(name) => config.select_group(&name)?,
            Err(env::VarError::NotPresent) => {}
            Err(env::VarError::NotUnicode(_)) => return Err("Set VADE_GROUP to a plain [group name] from your config.".into()),
        }
        if config.validate_model_revision().is_err() {
            crate::output::stderr_line(format_args!("Warning: model_revision is missing or mutable; inspection and shutdown remain available. Set a verified immutable revision before vade install or vade up; see INSTALL.md."));
        }
        Ok(config)
    }

    /// Require a pinned model revision: a 40-digit commit or `local-sha256-<64 hex>`.
    pub fn validate_model_revision(&self) -> Result<(), String> {
        if crate::text::is_lower_hex(&self.model_revision, 40)
            || self.model_revision.strip_prefix("local-sha256-").is_some_and(|hash| crate::text::is_lower_hex(hash, 64))
        {
            return Ok(());
        }
        let start = if self.model.ends_with("-fp8") { "vade up --fp8" } else { "vade up" };
        Err(format!("Set model_revision to a verified immutable 40-hex publisher revision or local-sha256-<64-hex manifest digest>; missing values, placeholders such as REPLACE_WITH_VERIFIED_IMMUTABLE_REVISION, and mutable tags are refused for installation and startup. See INSTALL.md, then run {start}."))
    }

    /// Parse config text; relative local paths resolve against `path`'s folder.
    pub fn parse(text: &str, path: &Path) -> Result<Self, String> {
        let base = path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let (globals, rows, group_rows) = parse_sections(text)?;
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
            lock_dir: state_dir.clone(),
            state_dir,
            machines: Vec::new(),
            groups: Vec::new(),
            group: None,
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
        for (name, values) in group_rows {
            let mut values = Values::new(values);
            let hosts: Vec<String> = values.take("hosts").ok_or_else(|| format!(
                "Group {name} needs hosts = machine names in rank order, separated by commas."
            ))?.split(',').map(|host| host.trim().to_owned()).collect();
            let mut names = BTreeSet::new();
            for host in &hosts {
                if !names.insert(host.as_str()) || !config.machines.iter().any(|machine| machine.name == *host) {
                    return Err(format!("Group {name}: list each configured machine once in hosts, in rank order. Add [machine {host}] or fix the name."));
                }
            }
            let group = Group {
                local_port: port(&mut values, "local_port", config.local_port.into())?,
                layout: values.take("layout").unwrap_or_default(),
                name,
                hosts,
            };
            if !group.layout.is_empty() && !identifier(&group.layout) {
                return Err(format!("Group {}: set layout to a plain name.", group.name));
            }
            if usize::from(u16::MAX - group.local_port) < group.hosts.len() {
                return Err(format!("Group {}: replica tunnels need the local ports after local_port. Choose a lower local_port.", group.name));
            }
            values.finish(&format!("group {}", group.name))?;
            // The groups of one layout run at the same time: no shared machine or local port.
            for other in config.groups.iter().filter(|other| other.layout == group.layout) {
                if let Some(host) = group.hosts.iter().find(|host| other.hosts.contains(host)) {
                    return Err(format!(
                        "Groups {} and {} share machine {host} in the same layout. Give one of them another layout = name, or remove the machine from one group.",
                        other.name, group.name
                    ));
                }
                if group.ports().start() <= other.ports().end() && other.ports().start() <= group.ports().end() {
                    return Err(format!(
                        "Groups {} and {} in the same layout need separate local ports. Set local_port so ports {} and {} do not overlap.",
                        other.name, group.name, other.port_label(), group.port_label()
                    ));
                }
            }
            config.groups.push(group);
        }
        Ok(config)
    }

    /// The ownership folder of group `name`: `state_dir/groups/NAME`.
    pub fn group_state_dir(&self, name: &str) -> PathBuf {
        self.lock_dir.join("groups").join(name)
    }

    /// Every group that may own a server: each `[group]` of this config and
    /// each folder under `state_dir/groups`, so a group renamed or removed
    /// while its server runs still counts.
    pub fn group_record_names(&self) -> Result<Vec<String>, String> {
        let unreadable = "Cannot list the group ownership folders under state_dir/groups; keep them for inspection.";
        let mut names: Vec<String> = self.groups.iter().map(|group| group.name.clone()).collect();
        match fs::read_dir(self.lock_dir.join("groups")) {
            Ok(entries) => for entry in entries {
                let entry = entry.map_err(|_| unreadable)?;
                if let Some(name) = entry.file_name().to_str().filter(|name| identifier(name)) { names.push(name.into()); }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(unreadable.into()),
        }
        names.sort();
        names.dedup();
        Ok(names)
    }

    /// The step that stops the server of group `name` with this config, also
    /// when the config has no `[group name]` section any more.
    pub fn group_down_step(&self, name: &str) -> String {
        if self.groups.iter().any(|group| group.name == name) {
            format!("run VADE_GROUP={name} vade down with its config")
        } else {
            format!("restore [group {name}] in its config, then run VADE_GROUP={name} vade down")
        }
    }

    /// Select the group `name`: its machines in rank order, its own ownership
    /// folder under `state_dir` and its own local port.
    pub fn select_group(&mut self, name: &str) -> Result<(), String> {
        let group = self.groups.iter().find(|group| group.name == name).cloned().ok_or_else(|| {
            let names: Vec<&str> = self.groups.iter().map(|group| group.name.as_str()).collect();
            if names.is_empty() {
                format!("VADE_GROUP selects group {}, but the config has no [group name] sections. Unset VADE_GROUP, or add the group. See docs/eight-sparks.md.", crate::text::argument(name))
            } else {
                format!("VADE_GROUP selects group {}, which is not in the config. Choose one of: {}.", crate::text::argument(name), names.join(", "))
            }
        })?;
        self.state_dir = self.group_state_dir(&group.name);
        self.local_port = group.local_port;
        self.group = Some(group);
        Ok(())
    }

    /// Resolve the activation record once per command. All reads use this generation.
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
                || !crate::text::is_lower_hex(id, 24)
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

    /// The machines `vade up` uses for these options, with their fabric links
    /// checked unless they are independent one-Spark replicas.
    pub fn selected(&self, options: &UpOptions) -> Result<Vec<Machine>, String> {
        let selected = self.selected_machines(options)?;
        // Independent replicas never talk to each other, so they need no fabric links.
        // A group must be an arc of the cabling in rank order, each host linked
        // to the next; --hosts and --sparks need the same path.
        if !options.replicas {
            match &self.group {
                Some(group) => crate::topology::check_arc(&format!("Group {}", group.name), &selected)?,
                None => self.check_links(&selected)?,
            }
        } else if options.replica_sparks > 1 {
            // A --pairs replica is a tensor split: its two machines share a fabric link.
            for (index, pair) in selected.chunks(options.replica_sparks).enumerate() {
                crate::topology::check_arc(&format!("Pair {}", index + 1), pair)?;
            }
        }
        Ok(selected)
    }
    fn selected_machines(&self, options: &UpOptions) -> Result<Vec<Machine>, String> {
        let from_group;
        let options = match &self.group {
            None => options,
            Some(group) => {
                // `vade update` restarts a running group with its own hosts.
                if (!options.hosts.is_empty() && options.hosts != group.hosts) || (options.sparks.is_some() && !options.replicas) {
                    return Err(format!("VADE_GROUP={} names the machines and their rank order. Remove --sparks and --hosts, or unset VADE_GROUP.", group.name));
                }
                let count = if options.hosts.is_empty() { options.sparks } else { Some(options.hosts.len()) };
                if options.replicas && count != Some(group.hosts.len()) {
                    // Name only commands the parser accepts for this group size.
                    // FP4-Local runs on 1..24 Sparks, so a larger group has no command.
                    let size = group.hosts.len();
                    let machines = crate::text::count(size, "machine", "machines");
                    if size > MAX_FP4_SPARKS {
                        return Err(format!("Group {} has {machines}. FP4-Local supports 1..{MAX_FP4_SPARKS} Sparks. Shorten the group, or unset VADE_GROUP.", group.name));
                    }
                    let mut runs = Vec::new();
                    if (2..=MAX_REPLICAS).contains(&size) { runs.push(format!("vade up --replicas {size} for one replica on each")); }
                    if size % 2 == 0 && (2..=MAX_REPLICAS).contains(&(size / 2)) { runs.push(format!("vade up --replicas {} --pairs for a pair each", size / 2)); }
                    if size % 2 == 0 && size / 2 == 1 { runs.push("vade up without --replicas for one tensor-split pair".into()); }
                    if runs.is_empty() { runs.push("vade up without --replicas".into()); }
                    return Err(format!("Group {} has {machines}. Run {}, or unset VADE_GROUP.", group.name, runs.join(", or ")));
                }
                from_group = UpOptions { hosts: group.hosts.clone(), sparks: None, ..options.clone() };
                &from_group
            }
        };
        if options.fp4_local && self.model.ends_with("-fp8") {
            return Err("Your config selects FP8. Run vade up --fp8, or select the FP4-Local checkpoint in VADE_CONFIG and run vade up.".into());
        }
        if !options.fp4_local && self.model.ends_with("-fp4-local") {
            return Err("Your config selects FP4-Local. Run vade up without --fp8, or select an FP8 config with VADE_CONFIG.".into());
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
                    .ok_or_else(|| format!("Machine {name} is not in your config. Give --hosts the NAME of a [machine NAME] section; vade config lists them."))
            }).collect::<Result<Vec<_>, _>>()?
        } else {
            let count = options.sparks.unwrap_or(if options.fp4_local { 1 } else { 2 });
            if count > self.machines.len() {
                return Err(format!(
                    "Your config has {} machine{}. Choose fewer supported machines with --sparks or add machine sections, then retry {}. For one Spark, select an FP4-Local config with VADE_CONFIG and run vade up.",
                    self.machines.len(), if self.machines.len() == 1 { "" } else { "s" },
                    if options.fp4_local { "vade up" } else { "vade up --fp8" }
                ));
            }
            self.machines[..count].to_vec()
        };
        if !options.fp4_local && !matches!(selected.len(), 2 | 4 | 6) {
            return Err("FP8 needs 2, 4, or 6 machines. Run vade up --fp8 --sparks 2, 4, or 6, or list that many --hosts. For one Spark, select an FP4-Local config with VADE_CONFIG and run vade up.".into());
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
    pub(crate) fn check_links(&self, selected: &[Machine]) -> Result<(), String> {
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

fn parse_sections(text: &str) -> Result<(Section, MachineRows, MachineRows), String> {
    let mut global = Section::new();
    let mut machines: MachineRows = Vec::new();
    let mut groups: MachineRows = Vec::new();
    // The open section: a machine (false) or a group (true), by index.
    let mut current: Option<(bool, usize)> = None;
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
                current = Some((false, machines.len() - 1));
            } else if let Some(name) = section.strip_prefix("group ").map(str::trim) {
                if !identifier(name) || groups.iter().any(|(existing, _)| existing == name) {
                    return Err(error("use a unique [group name] section with a plain name."));
                }
                groups.push((name.into(), Section::new()));
                current = Some((true, groups.len() - 1));
            } else {
                return Err(error("use [global], [machine name] or [group name]."));
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
            Some((false, index)) => &mut machines[index].1,
            Some((true, index)) => &mut groups[index].1,
            None => &mut global,
        };
        if target.insert(key.into(), value.into()).is_some() {
            return Err(error("use each key only once in a section."));
        }
    }
    Ok((global, machines, groups))
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
                "Unknown config key {key} in {section}. Check cli/examples/fp4-local.conf (or cli/examples/vade.conf for FP8) for supported keys."
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

/// Accept a plain absolute or `~/` path for a machine setting: no shell
/// symbols and no `..`.
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

/// Whether `value` is an IP address or a safe host alias for the fabric.
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
    /// Eight machines in two rings of four (s1..s4, s5..s8) plus `groups`.
    fn eight(groups: &str) -> Result<Config, String> {
        let mut text = PAIR.split("[machine").next().unwrap().to_string();
        for index in 1..=8 {
            let (ring, offset) = ((index - 1) / 4 * 4, (index - 1) % 4);
            let (next, previous) = (ring + (offset + 1) % 4 + 1, ring + (offset + 3) % 4 + 1);
            text += &format!("[machine s{index}]\nssh = s{index}-ssh\nfabric.s{next} = to-s{next}-from-s{index}\nfabric.s{previous} = to-s{previous}-from-s{index}\n");
        }
        parse(&(text + groups))
    }
    const LAYOUTS: &str = "[group quad-a]\nhosts = s1,s2,s3,s4\nlayout = quads\n[group quad-b]\nhosts = s5,s6,s7,s8\nlocal_port = 18090\nlayout = quads\n\
        [group pair-1]\nhosts = s1,s2\nlayout = doubles\n[group pair-2]\nhosts = s3,s4\nlocal_port = 18083\nlayout = doubles\n\
        [group pair-3]\nhosts = s5,s6\nlocal_port = 18086\nlayout = doubles\n[group pair-4]\nhosts = s7,s8\nlocal_port = 18089\nlayout = doubles\n\
        [group one-1]\nhosts = s1\nlayout = singles\n[group one-8]\nhosts = s8\nlocal_port = 18082\nlayout = singles\n";
    #[test]
    fn groups_express_quads_doubles_and_singles_on_eight_machines() {
        let mut config = eight(LAYOUTS).unwrap();
        assert_eq!(config.groups.len(), 8);
        assert_eq!(config.groups[1], Group { name: "quad-b".into(), hosts: vec!["s5".into(), "s6".into(), "s7".into(), "s8".into()], local_port: 18090, layout: "quads".into() });
        // Without VADE_GROUP nothing changes.
        assert_eq!(config.selected(&UpOptions::default()).unwrap().len(), 1);
        let base = config.state_dir.clone();
        config.select_group("quad-b").unwrap();
        assert_eq!((config.state_dir.clone(), config.local_port, config.lock_dir.clone()), (base.join("groups/quad-b"), 18090, base.clone()));
        let names = |machines: Vec<Machine>| machines.into_iter().map(|machine| machine.name).collect::<Vec<_>>();
        assert_eq!(names(config.selected(&UpOptions::default()).unwrap()), ["s5", "s6", "s7", "s8"]);
        assert_eq!(names(config.selected(&UpOptions { replicas: true, sparks: Some(4), ..UpOptions::default() }).unwrap()).len(), 4);
        assert!(config.selected(&UpOptions { replicas: true, sparks: Some(2), ..UpOptions::default() }).unwrap_err().contains("Group quad-b has 4 machines. Run vade up --replicas 4 for one replica on each, or vade up --replicas 2 --pairs for a pair each, or unset VADE_GROUP."));
        // Every hint is a command the parser accepts: no --replicas 1 for a pair or a single machine.
        for (name, hint) in [("pair-1", "Group pair-1 has 2 machines. Run vade up --replicas 2 for one replica on each, or vade up without --replicas for one tensor-split pair, or unset VADE_GROUP."),
            ("one-1", "Group one-1 has 1 machine. Run vade up without --replicas, or unset VADE_GROUP.")] {
            let mut small = eight(LAYOUTS).unwrap();
            small.select_group(name).unwrap();
            let error = small.selected(&UpOptions { replicas: true, replica_sparks: 2, sparks: Some(4), ..UpOptions::default() }).unwrap_err();
            assert_eq!(error, hint);
            assert!(!error.contains("--replicas 1") && crate::cli::parse(&["up".into(), "--replicas".into(), "1".into()]).is_err());
        }
        // For every group size, each hinted command parses; a group over 24 machines gets no command.
        for size in 1..=32 {
            let mut sized = eight(LAYOUTS).unwrap();
            sized.groups.push(Group { name: "sized".into(), hosts: (1..=size).map(|index| format!("m{index}")).collect(), local_port: 18100, layout: "sized".into() });
            sized.select_group("sized").unwrap();
            let error = sized.selected(&UpOptions { replicas: true, replica_sparks: 2, sparks: Some(size + 2), ..UpOptions::default() }).unwrap_err();
            if size > MAX_FP4_SPARKS {
                assert_eq!(error, format!("Group sized has {size} machines. FP4-Local supports 1..24 Sparks. Shorten the group, or unset VADE_GROUP."));
                continue;
            }
            let runs = error.split_once(". Run ").and_then(|(_, runs)| runs.strip_suffix(", or unset VADE_GROUP.")).unwrap_or_else(|| panic!("{error}"));
            for run in runs.split(", or ") {
                let words: Vec<String> = run.split(' ').skip(1).take_while(|word| !["for", "without"].contains(word)).map(String::from).collect();
                assert!(words[0] == "up" && crate::cli::parse(&words).is_ok(), "size {size}: {run}");
            }
        }
        assert!(crate::cli::parse(&["up", "--replicas", "13", "--pairs"].map(String::from)).is_err());
        for options in [UpOptions { sparks: Some(2), ..UpOptions::default() }, UpOptions { hosts: vec!["s1".into()], ..UpOptions::default() }] {
            assert!(config.selected(&options).unwrap_err().contains("Remove --sparks and --hosts"));
        }
        // vade update restarts a running group with its saved hosts.
        let own: Vec<String> = ["s5", "s6", "s7", "s8"].map(String::from).to_vec();
        assert!(config.selected(&UpOptions { hosts: own.clone(), ..UpOptions::default() }).is_ok());
        assert!(config.selected(&UpOptions { hosts: own, replicas: true, ..UpOptions::default() }).is_ok());
        // FP8 on a group of four keeps its own count rule.
        config.model = "qwen3.8-flash-next-fp8".into();
        assert_eq!(config.selected(&UpOptions { fp4_local: false, ..UpOptions::default() }).unwrap().len(), 4);
        let mut config = eight(LAYOUTS).unwrap();
        config.select_group("one-8").unwrap();
        assert_eq!(names(config.selected(&UpOptions::default()).unwrap()), ["s8"]);
        config.select_group("pair-3").unwrap();
        assert_eq!(names(config.selected(&UpOptions::default()).unwrap()), ["s5", "s6"]);
        assert!(config.select_group("nope").unwrap_err().contains("Choose one of: quad-a, quad-b, pair-1"));
        assert!(eight("").unwrap().select_group("quad-a").unwrap_err().contains("no [group name] sections"));
    }
    #[test]
    fn a_group_must_be_an_arc_in_cable_order() {
        let mut config = eight("[group cross]\nhosts = s1,s2,s3,s4,s5,s6,s7,s8\n[group shuffled]\nhosts = s1,s3,s2,s4\nlocal_port = 18100\nlayout = other\n").unwrap();
        config.select_group("cross").unwrap();
        assert!(config.selected(&UpOptions::default()).unwrap_err().contains("Group cross is not an arc of the cabling: s4 and s5 have no fabric link"));
        config.select_group("shuffled").unwrap();
        assert!(config.selected(&UpOptions::default()).unwrap_err().contains("hosts = s1,s2,s3,s4"));
        // The same machines in --hosts cable order only need a path.
        let config = eight("").unwrap();
        assert!(config.selected(&UpOptions { hosts: ["s2", "s3", "s4", "s1"].map(String::from).to_vec(), ..UpOptions::default() }).is_ok());
    }
    #[test]
    fn the_eight_spark_example_starts_every_group() {
        let examples = [
            (include_str!("../examples/eight-sparks.conf"), vec!["ring of 8: spark-1, spark-2, spark-3, spark-4, spark-5, spark-6, spark-7, spark-8"], ["eight", "quads", "doubles", "doubles-b", "singles"]),
        ];
        for (text, cabling, layouts) in examples {
            let config = parse(text).unwrap();
            assert_eq!(crate::topology::cabling(&config.machines), cabling);
            assert_eq!(config.groups.iter().map(|group| group.layout.as_str()).collect::<BTreeSet<_>>(), layouts.into());
            assert_eq!(config.groups.iter().filter(|group| group.layout == "singles").count(), 8);
            assert!(crate::topology::check_whole_ring(&config.machines).is_ok());
            // Every group of the example passes the arc check vade up applies.
            for group in &config.groups {
                let mut config = config.clone();
                config.select_group(&group.name).unwrap();
                assert!(config.selected(&UpOptions::default()).is_ok(), "{}", group.name);
            }
        }
    }
    #[test]
    fn groups_in_one_layout_cannot_share_machines_or_ports() {
        for (groups, expected) in [
            ("[group a]\nhosts = s1,s2\n[group b]\nhosts = s2,s3\nlocal_port = 18090\n", "share machine s2"),
            ("[group a]\nhosts = s1,s2\n[group b]\nhosts = s3,s4\nlocal_port = 18082\n", "ports 18080..18082 and 18082..18084"),
            ("[group a]\nhosts = s1,s1\n", "list each configured machine once"),
            ("[group a]\nhosts = s1,s9\n", "Add [machine s9]"),
            ("[group a]\nlocal_port = 18090\n", "needs hosts"),
            ("[group a]\nhosts = s1\nlayout = two words\n", "plain name"),
            ("[group a]\nhosts = s1\nwatts = 1\n", "Unknown config key watts in group a"),
            ("[group a]\nhosts = s1\n[group a]\nhosts = s2\n", "unique [group name]"),
            ("[group a]\nhosts = s1,s2\nlocal_port = 65535\n", "Choose a lower local_port"),
        ] { assert!(eight(groups).unwrap_err().contains(expected), "{groups}: {:?}", eight(groups).err()); }
        assert!(eight("[group a]\nhosts = s1,s2\nlayout = x\n[group b]\nhosts = s2,s3\nlayout = y\n").is_ok());
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
