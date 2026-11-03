//! Local launch defaults saved in the Vade home's `config` file: model, port,
//! context and output limits, and the context step-down after an engine
//! memory refusal.
use super::*;

/// Smaller contexts tried, in order, after an engine memory refusal.
const STEP_DOWN: [u32; 4] = [32768, 16384, 8192, 4096];

/// The engine's context refusal ("Requested size does not fit." whose Asked:
/// line names `CONTEXT=<n>`), or None for any other stop. Weight and allocator
/// refusals say "CONTEXT and OUTPUT not supplied"; a smaller context cannot help.
pub(crate) fn memory_refusal(log: &str) -> Option<String> {
    let lines: Vec<&str> = log.lines().collect();
    let start = lines.iter().position(|line| line.contains("Requested size does not fit."))?;
    let asked = lines.get(start + 1)?;
    if !asked.trim_start().starts_with("Asked:") || !asked.contains("CONTEXT=") { return None; }
    let refusal = lines[start..].iter().take(8).copied().collect::<Vec<_>>().join("\n");
    refusal.contains("Available").then_some(refusal)
}

/// The saved local defaults (`format=vade-owner-v1`): checkpoint, limits, port
/// and the optional engine controls a later `vade up` must reuse.
#[derive(Clone)]
pub(super) struct Config {
    pub model: PathBuf,
    pub model_bytes: u64,
    pub context: u32,
    pub output: u32,
    pub port: u16,
    mtp: Option<u32>,
    chats: Option<usize>,
    memory_bytes: Option<u64>,
    cache_bytes: Option<u64>,
    decode_fuse: bool,
    stage: PathBuf,
    state: PathBuf,
}

/// The API port a first launch tries before it takes any free loopback port.
pub const DEFAULT_PORT: u16 = 18080;

impl Config {
    /// First-run defaults for `model`: the standard FP4-Local context (capped by the
    /// checkpoint) and its default output.
    pub fn new(
        root: &Path,
        model: &model::Model,
    ) -> Result<Self, String> {
        // Same default as vade up; the engine's admission may lower it at start.
        let context = crate::cli::FP4_CONTEXT.min(model.context_limit);
        if context < 4096 {
            return Err("The model has less than 4096 tokens of context; this launcher needs a larger supported context.".into());
        }
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, DEFAULT_PORT))
            .or_else(|_| std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)))
            .map_err(|_| "Cannot reserve a local chat port.")?;
        Ok(Self {
            mtp: None, chats: None, memory_bytes: None, cache_bytes: None, decode_fuse: false,
            model: model.path.clone(),
            model_bytes: model.bytes,
            context,
            output: crate::cli::fp4_output(context),
            port: listener
                .local_addr()
                .map_err(|_| "Cannot read the local chat port.")?
                .port(),
            stage: root.join("stage"),
            state: root.join("state"),
        })
    }

    /// Read `root/config`, or `None` before the first launch; unknown formats and
    /// out-of-range values are refused.
    pub fn load(root: &Path) -> Result<Option<Self>, String> {
        let Some(fields) = read_fields(&root.join("config"))? else {
            return Ok(None);
        };
        if required(&fields, "format")? != "vade-owner-v1" {
            return Err("The saved Vade config has an unknown format.".into());
        }
        let number = |key: &str| -> Result<u64, String> {
            required(&fields, key)?
                .parse()
                .map_err(|_| format!("Saved config {key} is invalid."))
        };
        let context = number("context_tokens")?;
        let output = number("output_tokens")?;
        let port = number("port")?;
        let model = PathBuf::from(required(&fields, "model_path")?);
        let stage = PathBuf::from(required(&fields, "stage_dir")?);
        let state = PathBuf::from(required(&fields, "state_dir")?);
        if !(4096..=262144).contains(&context)
            || output == 0
            || output > context || output > u64::from(crate::cli::FP4_OUTPUT_LIMIT)
            || !(1024..=65535).contains(&port)
            || !model.is_absolute()
            || stage != root.join("stage")
            || state != root.join("state")
        {
            return Err("Saved defaults are invalid; preserve the config for inspection.".into());
        }
        let optional = |key: &str, maximum: u64| -> Result<Option<u64>, String> {
            fields.get(key).map(|value| value.parse::<u64>().ok().filter(|n| *n <= maximum)
                .ok_or_else(|| format!("Saved config {key} is invalid; run vade doctor."))).transpose()
        };
        let chats = optional("chats", 32)?;
        if optional("memory_bytes", u64::MAX)? == Some(0) { return Err("Saved memory budget must be positive; run vade doctor.".into()); }
        if chats == Some(0) { return Err("Saved config chats must be positive; run vade doctor.".into()); }
        Ok(Some(Self {
            mtp: optional("mtp", 3)?.map(|n| n as u32),
            chats: chats.map(|n| n as usize),
            memory_bytes: optional("memory_bytes", u64::MAX)?,
            cache_bytes: optional("cache_bytes", u64::MAX)?,
            decode_fuse: optional("decode_fuse", 1)? == Some(1),
            model,
            model_bytes: number("model_bytes")?,
            context: context as u32,
            output: output as u32,
            port: port as u16,
            stage,
            state,
        }))
    }

    /// Refuse options a local launch cannot honour: FP8, more than one Spark,
    /// `--chats` or a nonzero `--cache-bytes`.
    pub fn validate_options(options: &UpOptions) -> Result<(), String> {
        if !options.fp4_local {
            return Err("FP8 needs two or four linked Sparks and a controller config. Run vade setup --fp8, or read docs/install-fp8.md.".into());
        }
        if options.sparks.is_some_and(|n| n != 1) || !options.hosts.is_empty() {
            return Err("Local launch serves FP4-Local on one Spark. More Sparks need a controller config that names them. Run vade setup --hosts spark-a,spark-b, or read docs/INSTALL.md#two-and-more-sparks.".into());
        }
        Self::supported_controls(options.chats, options.cache_bytes)
    }

    fn supported_controls(chats: Option<usize>, cache: Option<u64>) -> Result<(), String> {
        if chats.is_some() {
            return Err("Local FP4-Local does not support --chats; omit it (remove saved chats from the local config after vade down); the engine chooses concurrency from available memory.".into());
        }
        if cache.is_some_and(|bytes| bytes != 0) {
            return Err("Local FP4-Local supports only --cache-bytes 0 to disable prefix retention; omit the flag or use 0, then run vade up.".into());
        }
        Ok(())
    }

    fn overrides(&mut self, options: &UpOptions) {
        if options.supplied_options.contains("--context") {
            self.context = options.context;
            if !options.output_explicit { self.output = self.output.min(self.context); }
        }
        if options.output_explicit { self.output = options.output; }
        if options.mtp_explicit { self.mtp = Some(options.mtp); }
        if let Some(bytes) = options.memory_bytes { self.memory_bytes = Some(bytes); }
        if let Some(bytes) = options.cache_bytes { self.cache_bytes = Some(bytes); }
        if let Some(on) = options.decode_fuse { self.decode_fuse = on; }
    }

    /// This saved config with `options` applied as a start applies them, before
    /// the checkpoint and resource checks; the dry-run preview reads its limits.
    pub fn planned(&self, options: &UpOptions) -> Self {
        let mut planned = self.clone();
        planned.overrides(options);
        planned
    }

    /// The configuration these options request, if the running engine already
    /// uses exactly those limits.
    pub fn check_running_options(&self, options: &UpOptions, record: &Fields) -> Result<Self, String> {
        Self::validate_options(options)?;
        Self::supported_controls(self.chats, self.cache_bytes)?;
        let mut requested = self.clone();
        requested.overrides(options);
        requested.check_running_limits(record)?;
        Ok(requested)
    }

    /// The limits recorded with an owned engine, compared before reuse.
    pub fn launch_limits(&self) -> Fields {
        Fields::from([
            ("context_tokens".into(), self.context.to_string()),
            ("output_tokens".into(), self.output.to_string()),
            // Distinguish engine defaults from older ownership records that
            // never captured these controls, so reuse cannot guess live values.
            ("mtp".into(), self.mtp.map(|n| n.to_string()).unwrap_or_else(|| "engine_default".into())),
            ("memory_bytes".into(), self.memory_bytes.map(|n| n.to_string()).unwrap_or_else(|| "engine_default".into())),
            ("decode_fuse".into(), if self.decode_fuse { "on" } else { "off" }.into()),
        ])
    }

    /// Fail unless the owned engine's record has these exact limits.
    pub fn check_running_limits(&self, record: &Fields) -> Result<(), String> {
        // Records from before decode_fuse ran with it off.
        let recorded = |key: &str| record.get(key).map(String::as_str).or((key == "decode_fuse").then_some("off"));
        if self.launch_limits().iter().any(|(key, value)| recorded(key) != Some(value.as_str())) {
            return Err("The owned engine has different or unrecorded launch limits; run vade down, then repeat vade up with the requested flags.".into());
        }
        Ok(())
    }

    /// Apply explicit options for this start, then check them against the
    /// checkpoint and this Spark.
    pub fn apply(&mut self, options: &UpOptions, model: &model::Model, identity: &detect::Identity) -> Result<(), String> {
        Self::validate_options(options)?;
        self.overrides(options);
        self.model_bytes = model.bytes;
        self.check_resources(model, identity)
    }

    /// Check the context and output limits against the checkpoint, and the model
    /// against this Spark's memory.
    pub fn check_resources(&self, model: &model::Model, identity: &detect::Identity) -> Result<(), String> {
        Self::supported_controls(self.chats, self.cache_bytes)?;
        if self.context > model.context_limit {
            return Err(format!("Requested context {} exceeds this checkpoint's {} token limit; run vade up --context {}.",
                self.context, model.context_limit, model.context_limit));
        }
        if self.output > self.context || self.output > 16384 {
            return Err("Output must fit context and the FP4-Local engine's 16384 token limit; run vade up --output 8192 (or a smaller value).".into());
        }
        if self.memory_bytes == Some(0) || matches!((self.memory_bytes, self.cache_bytes), (Some(memory), Some(cache)) if cache >= memory) {
            return Err("Memory budget must be positive and larger than the KV pool; run vade up with corrected --memory-bytes and --cache-bytes.".into());
        }
        // Detection already refused memory that cannot hold the checkpoint plus
        // the 8 GiB working reserve; the engine's admission funds the context.
        identity.check_resources(model.bytes, 0)
    }

    /// After the engine refuses a default or saved context for memory, lower it
    /// to the next standard size; callers never lower an explicit --context.
    pub fn step_down(&mut self, explicit_output: bool) -> Option<u32> {
        let next = STEP_DOWN.into_iter().find(|n| *n < self.context)?;
        self.context = next;
        self.output = if explicit_output { self.output.min(next) } else { crate::cli::fp4_output(next) };
        Some(next)
    }

    /// Add the engine environment for MTP, the memory budget and disabled prefix
    /// retention to a worker start request.
    pub fn engine_env(&self, fields: &mut Fields) {
        for (key, value) in [("MTP", self.mtp.map(u64::from)), ("SEQPAR_MEMORY_BYTES", self.memory_bytes)] {
            if let Some(value) = value { fields.insert(format!("env.{key}"), value.to_string()); }
        }
        // The FP4 service bypasses FLASH_SERVE_KV_BYTES. Its own switch is
        // authoritative, including when the caller inherited a cache opt-in.
        if self.cache_bytes == Some(0) {
            fields.insert("env.VADE_FLASHNEXT_PREFIX_CACHE".into(), "0".into());
        }
    }

    /// The prefix-retention switch to record: "0" when disabled by `--cache-bytes 0`,
    /// else `VADE_FLASHNEXT_PREFIX_CACHE` if set to 0 or 1.
    pub fn prefix_cache(&self) -> Result<Option<String>, String> {
        if self.cache_bytes == Some(0) { return Ok(Some("0".into())); }
        match std::env::var("VADE_FLASHNEXT_PREFIX_CACHE") {
            Ok(value) if matches!(value.as_str(), "0" | "1") => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            _ => Err("VADE_FLASHNEXT_PREFIX_CACHE must be 0 or 1; unset it or correct it, then run vade up.".into()),
        }
    }

    /// Fail when disabled retention is saved but the owned engine does not record it.
    pub fn check_running_cache(&self, record: &Fields) -> Result<(), String> {
        if self.cache_bytes == Some(0) && record.get("prefix_cache").map(String::as_str) != Some("0") {
            return Err("The owned engine has no record of disabled prefix retention; run vade down, then vade up --cache-bytes 0 to apply the saved limit.".into());
        }
        Ok(())
    }

    /// Save first-run defaults; existing defaults are never overwritten.
    pub fn save(&self, root: &Path) -> Result<(), String> {
        if root.join("config").exists() {
            return Err("Vade defaults already exist; they were left unchanged.".into());
        }
        text::atomic_write(&root.join("config"), &self.display())
    }

    /// The saved `key=value` form of these defaults.
    pub fn display(&self) -> String {
        let mut fields = Fields::from([
            ("format".into(), "vade-owner-v1".into()),
            (
                "model_path".into(),
                self.model.to_string_lossy().into_owned(),
            ),
            ("model_bytes".into(), self.model_bytes.to_string()),
            ("context_tokens".into(), self.context.to_string()),
            ("output_tokens".into(), self.output.to_string()),
            ("port".into(), self.port.to_string()),
            (
                "stage_dir".into(),
                self.stage.to_string_lossy().into_owned(),
            ),
            (
                "state_dir".into(),
                self.state.to_string_lossy().into_owned(),
            ),
        ]);
        for (key, value) in [("mtp", self.mtp.map(u64::from)), ("chats", self.chats.map(|n| n as u64)),
            ("memory_bytes", self.memory_bytes), ("cache_bytes", self.cache_bytes),
            ("decode_fuse", self.decode_fuse.then_some(1))] {
            if let Some(value) = value { fields.insert(key.into(), value.to_string()); }
        }
        text::encode(&fields)
    }
    /// The loopback URL of the local engine.
    /// The saved drafting depth: `None` means the engine chooses.
    pub fn mtp(&self) -> Option<u32> { self.mtp }
    /// Whether the engine runs the decode step as a chain.
    pub fn decode_fuse(&self) -> bool { self.decode_fuse }
    /// Whether drafting is on: off only when the saved setup says `--mtp 0`.
    pub fn drafting(&self) -> bool { self.mtp != Some(0) }
    pub fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}
