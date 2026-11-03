//! Controller records: the build [`Manifest`], the running [`Session`] with one
//! [`Rank`] per Spark, and the operation [`Lock`] that serializes lifecycle
//! commands.
use crate::cli::UpOptions;
use crate::config::{Config, Machine};
use crate::text::{self, Fields, required};
use crate::unix_file::HeldLock;
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

/// The staged files one machine runs: worker helper, engine binary and PTX,
/// with the SHA-256 recorded when they were built.
#[derive(Clone, Debug)]
pub struct Artifact {
    pub worker: String,
    pub binary: String,
    pub ptx: String,
    pub binary_sha256: String,
    pub ptx_sha256: String,
    pub worker_sha256: String,
}

/// The activated build manifest (`version=1`): source revisions, per-machine
/// artifacts and the engine features the CLI may rely on.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub engine_sha: String,
    pub cli_sha: String,
    pub harness_sha256: Option<String>,
    pub supports_output: bool,
    pub supports_fp4_chain: bool,
    pub artifacts: std::collections::BTreeMap<String, Artifact>,
}

impl Manifest {
    /// Read the manifest at `path`; a missing file asks for `vade install`.
    pub fn load(path: &Path) -> Result<Self, String> {
        text::reject_symlinks(path)?;
        let data = fs::read_to_string(path)
            .map_err(|_| "Build manifest is missing. Run vade install first.")?;
        Self::parse(&data)
    }
    /// Parse and validate a manifest record; unknown versions are refused.
    pub fn parse(data: &str) -> Result<Self, String> {
        let fields = text::parse(data)?;
        if required(&fields, "version")? != "1" {
            return Err("Unknown build manifest version. Run vade install.".into());
        }
        let engine_sha = required(&fields, "engine_sha")?.to_string();
        let cli_sha = required(&fields, "cli_sha")?.to_string();
        if !text::is_lower_hex(&engine_sha, 40) || !text::is_lower_hex(&cli_sha, 40) {
            return Err("Build manifest needs full source revisions. Run vade install.".into());
        }
        let harness_sha256 = fields.get("harness_sha256").cloned();
        if harness_sha256
            .as_ref()
            .is_some_and(|hash| !text::is_lower_hex(hash, 64))
        {
            return Err("Harness SHA256 is invalid. Run vade install.".into());
        }
        let supports_output = boolean(required(&fields, "supports_output")?)?;
        let supports_fp4_chain = fields.get("supports_fp4_chain")
            .map(|value| boolean(value)).transpose()?.unwrap_or(false);
        let mut artifacts = std::collections::BTreeMap::new();
        for key in fields
            .keys()
            .filter(|key| key.starts_with("machine.") && key.ends_with(".binary"))
        {
            let name = &key[8..key.len() - 7];
            let prefix = format!("machine.{name}");
            artifacts.insert(name.into(), Artifact::from_fields(&fields, &prefix)?);
        }
        if artifacts.is_empty() {
            return Err("Build manifest has no machines. Run vade install.".into());
        }
        Ok(Self {
            engine_sha,
            cli_sha,
            harness_sha256,
            supports_output,
            supports_fp4_chain,
            artifacts,
        })
    }
    /// The `key=value` record that [`Manifest::parse`] reads back.
    pub fn encode(&self) -> String {
        let mut fields = Fields::from([
            ("version".into(), "1".into()),
            ("engine_sha".into(), self.engine_sha.clone()),
            ("cli_sha".into(), self.cli_sha.clone()),
            ("supports_output".into(), self.supports_output.to_string()),
            ("supports_fp4_chain".into(), self.supports_fp4_chain.to_string()),
        ]);
        if let Some(hash) = &self.harness_sha256 {
            fields.insert("harness_sha256".into(), hash.clone());
        }
        for (name, artifact) in &self.artifacts {
            artifact.write(&mut fields, &format!("machine.{name}"));
        }
        text::encode(&fields)
    }
}

impl Artifact {
    fn from_fields(fields: &Fields, prefix: &str) -> Result<Self, String> {
        let get = |key: &str| required(fields, &format!("{prefix}.{key}")).map(str::to_string);
        let result = Self {
            worker: get("worker")?,
            binary: get("binary")?,
            ptx: get("ptx")?,
            binary_sha256: get("binary_sha256")?,
            ptx_sha256: get("ptx_sha256")?,
            worker_sha256: get("worker_sha256")?,
        };
        for value in [
            &result.binary_sha256,
            &result.ptx_sha256,
            &result.worker_sha256,
        ] {
            if !text::is_lower_hex(value, 64) {
                return Err(
                    "Artifact SHA256 is invalid. Run vade install to rebuild the manifest.".into(),
                );
            }
        }
        for value in [&result.worker, &result.binary, &result.ptx] {
            validate_remote_path(value)?;
        }
        Ok(result)
    }
    fn write(&self, fields: &mut Fields, prefix: &str) {
        for (key, value) in [
            ("worker", &self.worker),
            ("binary", &self.binary),
            ("ptx", &self.ptx),
            ("binary_sha256", &self.binary_sha256),
            ("ptx_sha256", &self.ptx_sha256),
            ("worker_sha256", &self.worker_sha256),
        ] {
            fields.insert(format!("{prefix}.{key}"), value.clone());
        }
    }
}

/// One machine in a session: its config, staged artifacts, fabric endpoints and
/// whether a start was attempted there.
#[derive(Clone, Debug)]
pub struct Rank {
    pub machine: Machine,
    pub artifact: Artifact,
    pub listen: String,
    pub next: String,
    pub attempted: bool,
}

/// The controller's ownership record (`session.conf`): the token that proves
/// ownership, the phase, the launch settings and every rank.
#[derive(Clone, Debug)]
pub struct Session {
    pub token: String,
    pub phase: String,
    pub launch_fingerprint: Option<String>,
    pub model: String,
    pub build: String,
    pub cli_build: String,
    pub supports_output: bool,
    pub mtp: u32,
    pub mtp_explicit: bool,
    pub fp4_local: bool,
    pub context: u32,
    pub output: u32,
    pub chats: Option<usize>,
    /// Concurrent chats the engine reported at ready; never part of the launch request.
    pub admitted_chats: Option<usize>,
    pub fast_prompts: bool,
    pub memory_bytes: Option<u64>,
    pub cache_bytes: Option<u64>,
    /// The engine's decode_fuse option, passed in the engine arguments.
    pub decode_fuse: bool,
    pub local_port: u16,
    pub serve_port: u16,
    pub ranks: Vec<Rank>,
    /// Independent servers behind a local gateway that owns local_port.
    pub replicas: bool,
    /// Sparks in each replica: 1, or 2 for `--pairs` (each replica is a tensor-split pair).
    pub replica_sparks: usize,
    /// The local gateway's PID and `ps` start time, recorded before it can serve.
    pub gateway: Option<(u32, String)>,
}

impl Session {
    /// A new session for `vade up` from the config, options and build manifest.
    /// `token` must be 64 hex digits; limits are checked here.
    pub fn new(
        config: &Config,
        options: &UpOptions,
        manifest: &Manifest,
        token: String,
    ) -> Result<Self, String> {
        if !text::is_lower_hex(&token, 64) {
            return Err("A full ownership token is required. Retry vade up.".into());
        }
        if options.output > options.context {
            return Err(
                "Output must fit in context. Set --output no larger than --context.".into(),
            );
        }
        if options.chats.is_some_and(|chats| !(1..=32).contains(&chats)) {
            return Err("Set --chats to a whole number from 1 to 32.".into());
        }
        let memory_bytes = match options.memory_bytes {
            Some(bytes) => Some(bytes),
            None => environment_bytes("SEQPAR_MEMORY_BYTES")?,
        };
        let cache_bytes = match options.cache_bytes {
            Some(bytes) => Some(bytes),
            None => environment_bytes("FLASH_SERVE_KV_BYTES")?,
        };
        if memory_bytes == Some(0)
            || matches!((memory_bytes, cache_bytes), (Some(memory), Some(cache)) if cache >= memory)
        {
            return Err("Memory must be positive and cache must be smaller than memory.".into());
        }
        let machines = config.selected(options)?;
        // Each group of Sparks serves one model: every machine, or one replica.
        let group = if options.replicas { options.replica_sparks.max(1) } else { machines.len() };
        if machines.len() % group != 0 {
            return Err("Each pair needs two machines. Give --replicas N --pairs 2N machines.".into());
        }
        if options.replicas && usize::from(u16::MAX - config.local_port) < machines.len() {
            return Err("Replica tunnels need the local ports after local_port. Choose a lower local_port.".into());
        }
        if options.fp4_local && machines.len() > 2 && !manifest.supports_fp4_chain {
            return Err("This staged engine does not declare FP4 layer-chain support. Run vade install from current source before choosing more than 2 Sparks, or use --sparks 1 or 2. Nothing started.".into());
        }
        let mut ranks = Vec::new();
        for (index, machine) in machines.iter().enumerate() {
            let (first, at) = (index - index % group, index % group);
            let members = &machines[first..first + group];
            let (listen, next) = if members.len() == 1 {
                ("127.0.0.1:0".into(), "127.0.0.1:0".into())
            } else {
                let prev = if at == 0 {
                    &members[1]
                } else {
                    &members[at - 1]
                };
                let listen = if options.fp4_local && members.len() > 2 && at == 0 {
                    "127.0.0.1:0".into()
                } else { address(
                    machine.fabric.get(&prev.name).ok_or(
                        "A fabric link is missing. Add both ends of each selected link to config.",
                    )?,
                    config.tp_port,
                ) };
                let next_machine = &members[(at + 1) % members.len()];
                let next = if options.fp4_local && members.len() > 2 && at + 1 == members.len() {
                    "127.0.0.1:0".into()
                } else if at + 1 == members.len() && members.len() > 2 {
                    ranks
                        .get(first)
                        .map(|r: &Rank| r.listen.clone())
                        .unwrap_or_default()
                } else {
                    address(
                        next_machine.fabric.get(&machine.name).ok_or(
                            "A fabric link is missing. Add both ends of each selected link to config.",
                        )?,
                        config.tp_port,
                    )
                };
                (listen, next)
            };
            let artifact = manifest
                .artifacts
                .get(&machine.name)
                .ok_or_else(|| format!("{} has no staged build. Run vade install.", machine.name))?
                .clone();
            if let Some(first) = ranks.first() {
                if artifact.binary_sha256 != first.artifact.binary_sha256
                    || artifact.ptx_sha256 != first.artifact.ptx_sha256
                {
                    return Err(format!(
                        "{} and {} have different engine or PTX hashes. Nothing started; rebuild all selected machines from the same source with vade install.",
                        machine.name, first.machine.name
                    ));
                }
            }
            ranks.push(Rank {
                machine: machine.clone(),
                artifact,
                listen,
                next,
                attempted: false,
            });
        }
        Ok(Self {
            token,
            phase: "starting".into(),
            launch_fingerprint: None,
            model: config.model.clone(),
            build: manifest.engine_sha.clone(),
            cli_build: manifest.cli_sha.clone(),
            supports_output: manifest.supports_output,
            mtp: options.mtp,
            mtp_explicit: options.mtp_explicit || !options.fp4_local,
            fp4_local: options.fp4_local,
            context: options.context,
            output: options.output,
            chats: options.chats,
            admitted_chats: None,
            fast_prompts: options.fast_prompts,
            memory_bytes,
            cache_bytes,
            decode_fuse: options.decode_fuse.unwrap_or(options.fp4_local && group == 1),
            local_port: config.local_port,
            serve_port: config.serve_port,
            ranks,
            replicas: options.replicas,
            replica_sparks: if options.replicas { group } else { 1 },
            gateway: None,
        })
    }
    /// Sparks that serve one model together: every rank, or one replica.
    pub fn group_len(&self) -> usize {
        if self.replicas { self.replica_sparks } else { self.ranks.len() }
    }
    /// Ranks that cooperate on one model over the fabric: a chain, or `--pairs` replicas.
    pub fn chained(&self) -> bool {
        self.group_len() > 1
    }
    /// Two FP4-Local Sparks on one model: the tensor split, the only two-Spark path.
    pub fn two_sparks(&self) -> bool {
        self.fp4_local && self.group_len() == 2
    }
    /// Whether the engine surely has a prefix restore here (one Spark, replicas, the two-Spark
    /// split), so an unset VADE_FLASHNEXT_PREFIX_CACHE means on. On three or more Sparks the
    /// saved settings do not name the transport: the layer chain has none, the split in parts on
    /// four Sparks over RDMA has one; the engine's own report decides.
    pub fn prefix_restore(&self) -> bool {
        !self.chained() || self.two_sparks()
    }
    /// The engine's own rank and world size for this machine.
    pub fn engine_slot(&self, index: usize) -> (usize, usize) {
        (index % self.group_len(), self.group_len())
    }
    /// Ranks serving HTTP: rank zero, or the first rank of every replica.
    pub fn servers(&self) -> std::iter::StepBy<std::ops::Range<usize>> {
        (0..self.ranks.len()).step_by(self.group_len().max(1))
    }
    /// Local tunnel port for a serving rank; replicas use the ports after the gateway's.
    pub fn tunnel_port(&self, index: usize) -> u16 {
        if self.replicas { self.local_port + 1 + (index / self.replica_sparks) as u16 } else { self.local_port }
    }
    /// The SSH control socket for a serving rank's tunnel; replicas get one each.
    pub fn tunnel_socket(&self, dir: &Path, index: usize) -> PathBuf {
        if self.replicas { dir.join(format!("tunnel-{}-{index}.sock", &self.token[..16])) } else { self.socket(dir) }
    }
    /// The drafting depth for display: the explicit MTP value or "engine default".
    pub fn mtp_label(&self) -> String {
        if self.mtp_explicit { self.mtp.to_string() } else { "engine default".into() }
    }
    /// The engine's positional arguments for the rank at `index`.
    pub fn engine_args(&self, index: usize) -> Vec<String> {
        let rank = &self.ranks[index];
        let (engine_rank, size) = self.engine_slot(index);
        let mut args = vec![engine_rank.to_string(), size.to_string(),
            rank.listen.clone(), rank.next.clone(), rank.artifact.ptx.clone(),
            rank.machine.model_path.clone(), self.context.to_string(),
            format!("127.0.0.1:{}", self.serve_port), "0".into()];
        if self.fp4_local { args.push("--fp4-local".into()); }
        if self.mtp_explicit { args.extend(["--mtp".into(), self.mtp.to_string()]); }
        if self.decode_fuse || size == 1 { args.extend(["--decode-fuse".into(), if self.decode_fuse { "on" } else { "off" }.into()]); }
        args
    }
    /// `TP_TOPOLOGY` for this session: "ring" for a two-Spark chain, else "path".
    pub fn topology(&self) -> &'static str {
        if self.group_len() == 2 {
            "ring"
        } else {
            "path"
        }
    }
    /// The local URL that chat and status use.
    pub fn endpoint(&self) -> String {
        format!("http://localhost:{}", self.local_port)
    }
    /// The SSH control socket for the controller tunnel.
    pub fn socket(&self, dir: &Path) -> PathBuf {
        Self::socket_path(dir, &self.token)
    }
    /// The control socket name derived from an ownership token.
    pub fn socket_path(dir: &Path, token: &str) -> PathBuf {
        dir.join(format!("tunnel-{}.sock", &token[..16]))
    }
    /// Atomically write `session.conf` in `dir`.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        text::atomic_write(&dir.join("session.conf"), &self.encode())
    }
    /// Read `session.conf` from `dir`, or `None` when no engine is owned.
    pub fn load(dir: &Path) -> Result<Option<Self>, String> {
        let path = dir.join("session.conf");
        text::reject_symlinks(&path)?;
        match fs::read_to_string(path) {
            Ok(data) => Self::parse(&data).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) => {
                Err("Cannot read the ownership record. Check permissions; keep the record.".into())
            }
        }
    }
    /// The `key=value` record that [`Session::load`] reads back.
    pub fn encode(&self) -> String {
        let mut fields = Fields::new();
        for (key, value) in [
            ("version", "1".into()),
            ("token", self.token.clone()),
            ("phase", self.phase.clone()),
            ("model", self.model.clone()),
            ("build", self.build.clone()),
            ("cli_build", self.cli_build.clone()),
            ("supports_output", self.supports_output.to_string()),
            ("mtp", self.mtp.to_string()),
            ("mtp_explicit", self.mtp_explicit.to_string()),
            ("fp4_local", self.fp4_local.to_string()),
            ("context", self.context.to_string()),
            ("output", self.output.to_string()),
            ("fast_prompts", self.fast_prompts.to_string()),
            ("decode_fuse", self.decode_fuse.to_string()),
            ("local_port", self.local_port.to_string()),
            ("serve_port", self.serve_port.to_string()),
            ("count", self.ranks.len().to_string()),
        ] {
            fields.insert(key.into(), value);
        }
        if let Some(fingerprint) = &self.launch_fingerprint {
            fields.insert("launch_fingerprint".into(), fingerprint.clone());
        }
        for (key, value) in [("memory_bytes", self.memory_bytes), ("cache_bytes", self.cache_bytes)] {
            if let Some(bytes) = value {
                fields.insert(key.into(), bytes.to_string());
            }
        }
        for (key, value) in [("chats", self.chats), ("admitted_chats", self.admitted_chats)] {
            if let Some(chats) = value {
                fields.insert(key.into(), chats.to_string());
            }
        }
        // Chain records keep their original fields and launch fingerprints.
        if self.replicas {
            fields.insert("replicas".into(), "true".into());
        }
        if self.replica_sparks > 1 {
            fields.insert("replica_sparks".into(), self.replica_sparks.to_string());
        }
        if let Some((pid, started)) = &self.gateway {
            fields.insert("gateway_pid".into(), pid.to_string());
            fields.insert("gateway_started".into(), started.clone());
        }
        for (i, rank) in self.ranks.iter().enumerate() {
            let p = format!("rank.{i}");
            rank.artifact.write(&mut fields, &p);
            for (key, value) in [
                ("name", &rank.machine.name),
                ("ssh", &rank.machine.ssh),
                ("model_path", &rank.machine.model_path),
                ("engine_source", &rank.machine.engine_source),
                ("cli_source", &rank.machine.cli_source),
                ("stage_dir", &rank.machine.stage_dir),
                ("state_dir", &rank.machine.state_dir),
                ("listen", &rank.listen),
                ("next", &rank.next),
            ] {
                fields.insert(format!("{p}.{key}"), value.clone());
            }
            fields.insert(format!("{p}.attempted"), rank.attempted.to_string());
        }
        text::encode(&fields)
    }
    fn parse(data: &str) -> Result<Self, String> {
        let f = text::parse(data)?;
        let get = |key: &str| required(&f, key).map(str::to_string);
        if get("version")? != "1" {
            return Err("Unknown ownership record version. Keep it and run vade doctor.".into());
        }
        let token = get("token")?;
        if !text::is_lower_hex(&token, 64) {
            return Err("Ownership token is invalid. Keep the record for manual review.".into());
        }
        let phase = get("phase")?;
        if !["starting", "running", "stopping"].contains(&phase.as_str()) {
            return Err("Ownership phase is invalid. Keep the record for manual review.".into());
        }
        let count: usize = number(&f, "count")?;
        let fp4_local = f.get("fp4_local").map(|value| boolean(value)).transpose()?.unwrap_or(false);
        if !(if fp4_local { (1..=crate::config::MAX_FP4_SPARKS).contains(&count) } else { [2, 4, 6].contains(&count) }) {
            return Err(
                "Ownership machine count is invalid. Keep the record for manual review.".into(),
            );
        }
        let mut ranks = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        let mut aliases = std::collections::BTreeSet::new();
        for i in 0..count {
            let p = format!("rank.{i}");
            let rget = |key: &str| get(&format!("{p}.{key}"));
            let ssh = rget("ssh")?;
            if !crate::config::ssh_destination(&ssh) {
                return Err(
                    "Recorded ssh destination is invalid. Keep the record for manual review.".into(),
                );
            }
            let name = rget("name")?;
            if !crate::cli::identifier(&name)
                || !names.insert(name.clone())
                || !aliases.insert(ssh.clone())
            {
                return Err("Ownership rank names or aliases are invalid. Keep the record for manual review.".into());
            }
            let machine = Machine {
                name,
                ssh,
                verify_url: None,
                fabric: Default::default(),
                model_path: rget("model_path")?,
                engine_source: rget("engine_source")?,
                cli_source: rget("cli_source")?,
                stage_dir: rget("stage_dir")?,
                state_dir: rget("state_dir")?,
            };
            validate_remote_path(&machine.state_dir)?;
            ranks.push(Rank {
                machine,
                artifact: Artifact::from_fields(&f, &p)?,
                listen: rget("listen")?,
                next: rget("next")?,
                attempted: boolean(&rget("attempted")?)?,
            });
        }
        let session = Self {
            token,
            phase,
            launch_fingerprint: f.get("launch_fingerprint").cloned(),
            model: get("model")?,
            build: get("build")?,
            cli_build: get("cli_build")?,
            supports_output: boolean(&get("supports_output")?)?,
            mtp: number(&f, "mtp")?,
            mtp_explicit: f.get("mtp_explicit").map(|value| boolean(value)).transpose()?.unwrap_or(true),
            fp4_local,
            context: number(&f, "context")?,
            output: number(&f, "output")?,
            chats: f.get("chats").map(|_| number(&f, "chats")).transpose()?,
            admitted_chats: f.get("admitted_chats").map(|_| number(&f, "admitted_chats")).transpose()?,
            fast_prompts: f.get("fast_prompts").map(|value| boolean(value)).transpose()?.unwrap_or(false),
            memory_bytes: f.get("memory_bytes").map(|_| number(&f, "memory_bytes")).transpose()?,
            cache_bytes: f.get("cache_bytes").map(|_| number(&f, "cache_bytes")).transpose()?,
            decode_fuse: f.get("decode_fuse").map(|value| boolean(value)).transpose()?.unwrap_or(false),
            local_port: number(&f, "local_port")?,
            serve_port: number(&f, "serve_port")?,
            ranks,
            replicas: f.get("replicas").map(|value| boolean(value)).transpose()?.unwrap_or(false),
            replica_sparks: f.get("replica_sparks").map(|_| number(&f, "replica_sparks")).transpose()?.unwrap_or(1),
            gateway: match (f.get("gateway_pid"), f.get("gateway_started")) {
                (Some(_), Some(started)) => Some((number(&f, "gateway_pid")?, started.clone())),
                _ => None,
            },
        };
        if session.launch_fingerprint.as_ref().is_some_and(|value| !text::is_lower_hex(value, 64))
            || !text::is_lower_hex(&session.build, 40)
            || !text::is_lower_hex(&session.cli_build, 40)
            || session.local_port == 0
            || session.serve_port == 0
            || (session.replicas && (!session.fp4_local || usize::from(u16::MAX - session.local_port) < session.ranks.len()))
            || session.replica_sparks == 0
            || (session.replica_sparks > 1 && (!session.replicas || session.ranks.len() % session.replica_sparks != 0))
            || session.mtp > 3
            || !(4096..=262144).contains(&session.context)
            || !(1024..=session.context).contains(&session.output)
            || [session.chats, session.admitted_chats].into_iter().flatten().any(|chats| !(1..=32).contains(&chats))
            || (!session.supports_output && session.output != session.context.min(8192))
            || session.memory_bytes == Some(0)
            || matches!((session.memory_bytes, session.cache_bytes), (Some(memory), Some(cache)) if cache >= memory)
        {
            return Err(
                "Ownership build or port is invalid. Keep the record for manual review.".into(),
            );
        }
        Ok(session)
    }
}

/// Concurrent chats from the engine's last ready line (FP4-Local `flashnext
/// ready ... active=N`); None when the log does not report it.
pub fn admitted_chats(log: &str) -> Option<usize> {
    let line = log.lines().rev().find(|line| line.contains("flashnext ready "))?;
    line.split_whitespace().find_map(|word| word.strip_prefix("active="))?
        .parse().ok().filter(|chats| (1..=32).contains(chats))
}

fn environment_bytes(name: &str) -> Result<Option<u64>, String> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) => value
            .parse()
            .map(Some)
            .map_err(|_| format!("Set {name} to a whole number of bytes that fits in 64 bits.")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        _ => Err(format!("Set {name} to a whole number of bytes that fits in 64 bits.")),
    }
}

/// A required numeric field of a saved record.
pub fn number<T: std::str::FromStr>(f: &Fields, key: &str) -> Result<T, String> {
    required(f, key)?
        .parse()
        .map_err(|_| format!("Invalid {key} in record. Keep it for manual review."))
}
fn boolean(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err("Invalid boolean in record. Keep it for manual review.".into()),
    }
}
/// Accept an absolute, `~/` or `{home}/` remote path with no `..` component and
/// no control characters.
pub fn validate_remote_path(value: &str) -> Result<(), String> {
    let rest = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("{home}/"))
        .or_else(|| value.strip_prefix('/'));
    if rest.is_none_or(|r| {
        r.is_empty() || r.split('/').any(|part| part == "..") || r.chars().any(|ch| ch.is_control())
    }) {
        return Err("Remote path must be absolute or start with ~/ and contain no parent traversal. Check config or manifest.".into());
    }
    Ok(())
}
fn address(host: &str, port: u16) -> String {
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// The error when another lifecycle command holds the lock.
const LOCK_BUSY: &str = "Vade is already changing this engine. Wait, then run vade status.";

/// The exclusive operation lock on a state directory, held until dropped.
pub struct Lock {
    _lock: HeldLock,
}
impl Lock {
    /// Take the lock without waiting; another running lifecycle command is an error.
    pub fn acquire(dir: &Path) -> Result<Self, String> {
        text::private_dir(dir)?;
        let path = dir.join(".lock");
        text::reject_symlinks(&path)?;
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options
            .open(path)
            .map_err(|_| "Cannot open operation lock. Check state directory permissions.")?;
        // The file stays open for this value's lifetime; the inode is never removed.
        HeldLock::try_take(file).ok().flatten().map(|_lock| Self { _lock }).ok_or_else(|| LOCK_BUSY.into())
    }
    /// Whether another lifecycle command (such as a vade launch that still
    /// loads the model) holds the lock now. The kernel drops the lock when
    /// its process ends, so a launch that died does not count.
    pub fn held(dir: &Path) -> bool {
        matches!(Self::acquire(dir), Err(error) if error == LOCK_BUSY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unix_file::ChildCopy;

    #[test]
    fn a_child_with_a_copy_of_the_lock_file_does_not_keep_the_lock() {
        let _guard = crate::runner::process_test_lock();
        let base = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-state-lock-{}", text::random_hex(8).unwrap()));
        fs::create_dir(&base).unwrap();
        let root = base.join(".vade");
        let lock = Lock::acquire(&root).unwrap();
        assert!(Lock::acquire(&root).is_err() && Lock::held(&root));
        let mut child = ChildCopy::of(&lock._lock);
        drop(lock);
        let again = Lock::acquire(&root);
        let running = child.running();
        drop(child);
        fs::remove_dir_all(&base).unwrap();
        assert!(running, "the child must still hold its copy");
        assert!(again.is_ok(), "the copy in a child kept the lock");
    }
}
