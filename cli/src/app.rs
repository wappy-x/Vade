//! The controller: one or more Sparks driven over SSH from a config file
//! (`VADE_CONFIG` or the user config directory). [`App`] runs each lifecycle
//! command; every subprocess goes through a [`Runner`], so the tests drive the
//! whole flow with a fake.
//!
//! - `deployment`: source sync, immutable builds, activation and updates.
//! - `prebuilt`: copy a verified local bundle instead of building.
//! - `diagnostics`: `vade doctor` prerequisite and harness checks.
//! - `health`: the controller tunnel, health probes and the readiness wait.
//! - `plan`: resolve launch paths, fabric, artifacts and the worker environment.
//! - `resident`: rebuild the requested launch and reuse an identical running one.
//! - `replicas`: `vade up --replicas N` servers behind one local gateway.
//! - `status`: `vade status --json` observations for a controller record.
mod deployment;
mod prebuilt;
mod diagnostics;
mod health;
pub(crate) mod events;
mod status;
mod plan;
mod resident;
mod replicas;
use deployment::REMOTE_STORAGE_GUARD;
/// The pinned Rust toolchains of a source build.
pub(crate) use diagnostics::{HOST_TOOLCHAIN, KERNEL_TOOLCHAIN};
use crate::cli::{Command, UpOptions};
use crate::config::{Config, Machine};
use crate::runner::{self, Exec, Output, Runner, quote, remote_path};
use crate::state::{Artifact, Manifest, Rank, Session};
use crate::style::{self, Stream, Token};
use crate::text::{self, Fields};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

/// The controller for one config. With `dry`, commands are printed, not run.
pub struct App<R> {
    pub config: Config,
    pub runner: R,
    pub dry: bool,
}

impl<R: Runner> App<R> {
    fn start_command(&self) -> &'static str {
        if self.config.model == "qwen3.8-flash-next-fp8" { "vade up --fp8" } else { "vade up" }
    }

    /// Run one lifecycle command against the configured machines.
    pub fn execute(&mut self, command: Command) -> Result<(), String> {
        if self.dry && !crate::checks::json() {
            crate::output::line(format_args!("Dry run. No commands run, files written, or connections opened."))?;
        }
        if !matches!(command, Command::Install | Command::Update) {
            self.config.resolve_generation()?;
        }
        match command {
            Command::Menu => self.menu(),
            Command::Up(options) => self.up(&options),
            Command::Down => self.down(),
            Command::Status => self.status(),
            Command::Logs => self.logs(),
            Command::Chat(options) => self.chat(&options),
            Command::Sessions(action) => self.sessions(&action),
            Command::Doctor => self.doctor(),
            Command::DoctorFabric { ring } => self.doctor_fabric(ring),
            Command::Install => self.install(),
            Command::Update => self.update(),
            _ => Err("Run vade --help to choose a command.".into()),
        }
    }

    fn menu(&mut self) -> Result<(), String> {
        match Session::load(&self.config.state_dir)? {
            Some(session) => {
                let connection = self.connection_ok(&session)?;
                let ready = self.local_health(&session)?
                    && connection
                    && session.phase == "running";
                let (mark, state) = if self.dry {
                    (style::paint(Token::Dim, "○", Stream::Out), "recorded (health not checked)")
                } else if ready {
                    (style::paint(Token::Ok, "●", Stream::Out), "ready")
                } else {
                    (style::paint(Token::Warn, "▲", Stream::Out), "recorded and needs attention")
                };
                let machines: Vec<_> = session.ranks.iter().map(|rank| rank.machine.name.as_str()).collect();
                crate::output::line(format_args!("{mark} Vade is {state} on {}.", machines.join(", ")))?;
            }
            None => {
                let owned = self.group_records()?;
                if owned.is_empty() {
                    crate::output::line(format_args!("{} Vade is stopped.", style::paint(Token::Dim, "○", Stream::Out)))?;
                } else {
                    // Groups still run: never headline this as stopped.
                    crate::output::line(format_args!("{} Vade has no server of its own; {} group(s) still own one.", style::paint(Token::Dim, "○", Stream::Out), owned.len()))?;
                    self.print_group_records(&owned)?;
                }
            }
        }
        crate::output::text(format_args!("{}", crate::status::next_steps(&[
            ("Start", self.start_command()), ("Chat", "vade chat"), ("Stop", "vade down"), ("Check", "vade doctor")])))
    }

    /// `vade up`: reuse an identical running launch, or start every rank, wait for
    /// readiness and record the session.
    pub fn up(&mut self, options: &UpOptions) -> Result<(), String> {
        self.config.validate_model_revision()?;
        self.config.validate_control_socket()?;
        let rdma = rdma_enabled()?;
        let existing = Session::load(&self.config.state_dir)?;
        let mut session = self.requested_session(options, existing.as_ref(), rdma)?;
        if let Some(existing) = existing {
            if !resident::same_launch(&existing, &session) {
                return Err("Vade already has an ownership record with different launch settings or build. Run vade down before changing it.".into());
            }
            return self.already_ready(&existing);
        }
        let rdma = session.chained() && rdma;
        let transport = if session.replicas { "replicas (independent one-Spark servers behind a local gateway; no fabric)" } else { transport_name(session.ranks.len(), rdma) };
        let group = self.config.group.as_ref().map(|group| ("Group", format!("{} (own ownership record; local port {})", group.name, group.local_port)));
        crate::output::text(format_args!("{} {}\n{}", style::paint(Token::Heading, "Launch plan:", Stream::Out), text::terminal_text(&session.model), style::table(&group.into_iter().chain([
            ("Machines", session.ranks.iter().map(|rank| rank.machine.name.as_str()).collect::<Vec<_>>().join(", ")),
            ("Context", format!("{} tokens", session.context)),
            ("Output limit", format!("{} tokens", session.output)),
            ("Drafting request", if session.mtp_explicit { format!("MTP {}", session.mtp) } else { "engine default".into() }),
            ("Concurrent chats", session.chats.map_or_else(|| "engine-admitted".into(), |count| count.to_string())),
            ("Transport", transport.into()),
            ("Endpoint", format!("{}/v1", session.endpoint())),
        ]).collect::<Vec<_>>(), 2)))?;
        self.resolve_fabric(&mut session)?;
        self.runner.port_free(session.local_port)?;
        if session.replicas {
            for index in session.servers() {
                self.runner.port_free(session.tunnel_port(index))?;
            }
        }
        // Verify helpers before executing them, then check paths and every GPU
        // before committing launch intent.
        for rank in &session.ranks {
            self.verify_artifacts(rank)?;
        }
        self.resolve_rank_paths(&mut session)?;
        for rank in &session.ranks {
            self.worker(&session, rank, "probe", Fields::new())?;
        }
        if !self.dry {
            crate::output::line(format_args!(
                "Starting {} on {} over {}. Waiting for the model to load.",
                session.model,
                crate::text::count(session.ranks.len(), "Spark", "Sparks"),
                transport
            ))?;
        }
        if !self.dry {
            session.save(&self.config.state_dir)?;
        }
        let result = (|| {
            let env = self.environment(&session)?;
            for index in 0..session.ranks.len() {
                if self.runner.interrupted() {
                    return Err(
                        "Startup was interrupted. Stopping the engines this launch started.".into(),
                    );
                }
                // Intent goes to disk before SSH: a lost reply is recovered from the remote journal.
                session.ranks[index].attempted = true;
                if !self.dry {
                    session.save(&self.config.state_dir)?;
                }
                let rank = &session.ranks[index];
                let mut request = env.clone();
                request.insert("engine".into(), rank.artifact.binary.clone());
                request.insert(
                    "cwd".into(),
                    rank.artifact
                        .binary
                        .rsplit_once('/')
                        .map(|(dir, _)| dir)
                        .unwrap_or("/")
                        .to_string(),
                );
                request.insert("binary_sha256".into(), rank.artifact.binary_sha256.clone());
                request.insert("ptx_path".into(), rank.artifact.ptx.clone());
                request.insert("ptx_sha256".into(), rank.artifact.ptx_sha256.clone());
                request.insert("model_path".into(), rank.machine.model_path.clone());
                let args = session.engine_args(index);
                for (i, arg) in args.iter().enumerate() {
                    request.insert(format!("arg.{i}"), arg.clone());
                }
                let (engine_rank, size) = session.engine_slot(index);
                request.insert("env.FLASH_SERVE_RANK".into(), engine_rank.to_string());
                request.insert("env.FLASH_SERVE_WORLD_SIZE".into(), size.to_string());
                let reply = self
                    .worker(&session, rank, "start", request)
                    .map_err(|error| self.startup_error(&session, index, &error))?;
                if reply.get("error_code").map(String::as_str) == Some("start_not_written") {
                    let error = format!("{}: {}", rank.machine.name, reply["error"]);
                    session.ranks[index].attempted = false;
                    if !self.dry {
                        session.save(&self.config.state_dir)?;
                    }
                    return Err(error);
                }
            }
            self.wait_ready(&session, true, self.config.health_timeout_secs)?;
            self.start_tunnels(&session)?;
            if session.replicas {
                self.start_gateway(&mut session)?;
            }
            self.wait_ready(&session, false, 20)?;
            session.admitted_chats = self.admitted_chats(&session);
            session.phase = "running".into();
            if !self.dry {
                session.save(&self.config.state_dir)?;
            }
            if self.dry {
                crate::output::line(format_args!(
                    "Start planned for {} on {}. Run the same command without --dry-run to start it.",
                    session.model,
                    crate::text::count(session.ranks.len(), "Spark", "Sparks")
                ))?;
            } else {
                crate::output::line(format_args!(
                    "{} ready on {}{} at {}: context {} tokens, drafting depth {}.\n{}",
                    session.model,
                    crate::text::count(session.ranks.len(), "Spark", "Sparks"),
                    if session.replicas { " (independent replicas behind the gateway)" } else { "" },
                    session.endpoint(),
                    session.context,
                    session.mtp_label(),
                    crate::status::next_steps(&[("Chat", "vade chat"), ("Stop", "vade down")]).trim_end(),
                ))?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            if !self.dry {
                crate::output::stderr_line(format_args!("Start did not finish. Cleaning up this launch."));
                if let Err(cleanup) = self.cleanup(&mut session) {
                    crate::output::cleanup_failed();
                    return Err(format!("{error} Cleanup could not finish: {cleanup} Leave saved ownership files in place; retry vade down or see Recovery in cli/README.md."));
                }
                crate::output::stderr_line(format_args!("Cleanup finished. The engines and the tunnel this launch started are closed."));
            }
            return Err(error);
        }
        Ok(())
    }

    /// FP4-Local admits chats from context and memory and reports the count
    /// once at ready; record it so status can show it after the log rolls.
    fn admitted_chats(&mut self, session: &Session) -> Option<usize> {
        let rank = session.ranks.first().filter(|_| !self.dry)?;
        let log = self.worker(session, rank, "logs", Fields::new()).ok()?;
        crate::state::admitted_chats(log.get("log")?)
    }

    fn worker(
        &mut self,
        session: &Session,
        rank: &Rank,
        op: &str,
        mut request: Fields,
    ) -> Result<Fields, String> {
        request.insert("op".into(), op.into());
        request.insert("token".into(), session.token.clone());
        request.insert("state_dir".into(), rank.machine.state_dir.clone());
        if op == "probe" {
            request.insert("rdma".into(), if session.chained() && rdma_enabled()? { "1" } else { "0" }.into());
        }
        let mut exec = runner::ssh(
            &rank.machine.ssh,
            format!("exec {} __worker", remote_path(&rank.artifact.worker)),
            &format!("{} {op}", rank.machine.name),
        );
        exec.stdin = text::encode(&request).into_bytes();
        exec.timeout = if session.phase == "starting" && matches!(op, "inspect" | "logs") {
            Duration::from_secs(5)
        } else {
            Duration::from_secs(90)
        };
        let output = self
            .run_owned_command(
                &exec,
                matches!(op, "inspect" | "stop" | "gpu" | "clean") && session.phase == "stopping",
            )
            .map_err(|error| format!("Transport: {}: {error}", rank.machine.name))?;
        if self.dry {
            return Ok(Fields::new());
        }
        if !output.success && output.stdout.trim().is_empty() {
            // The pre-start probe runs before any local or remote ownership record exists.
            return Err(if op == "probe" {
                format!("Transport: {} did not reply to the pre-start check; nothing was started or recorded. Retry vade up; the retained logs show the SSH error.", rank.machine.name)
            } else {
                format!("Transport: {} did not reply; its ownership record remains.", rank.machine.name)
            });
        }
        let fields = text::parse(&output.stdout)?;
        if op == "start"
            && fields.get("error_code").map(String::as_str) == Some("start_not_written")
            && fields.get("phase").map(String::as_str) == Some("absent")
            && fields.contains_key("error")
        {
            return Ok(fields);
        }
        if !output.success || fields.contains_key("error") {
            let message = fields
                .get("error")
                .map(String::as_str)
                .unwrap_or("Machine check failed. Check SSH and prerequisites with vade doctor.");
            return Err(format!("{}: {message}", rank.machine.name));
        }
        if matches!(op, "probe" | "start") {
            if let Some(note) = fields.get("gpu_note") {
                crate::output::line(format_args!("{}: {}", rank.machine.name, text::terminal_text(note)))?;
            }
        }
        Ok(fields)
    }

    fn verify_artifacts(&mut self, rank: &Rank) -> Result<(), String> {
        self.verify_staged(&rank.machine, &rank.artifact)
    }

    /// Hash the staged engine, PTX and helper on `machine` against `a`.
    pub(super) fn verify_staged(&mut self, machine: &Machine, a: &Artifact) -> Result<(), String> {
        let command = format!(
            "sha256sum {} {} {}",
            remote_path(&a.binary),
            remote_path(&a.ptx),
            remote_path(&a.worker)
        );
        let output = self.checked(
            runner::ssh(
                &machine.ssh,
                command,
                &format!("{} artifact hashes", machine.name),
            ),
            "Cannot verify staged artifacts. Run vade install, then try vade up.",
        )?;
        if !self.dry {
            let hashes = output
                .stdout
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .collect::<Vec<_>>();
            if hashes
                != [
                    a.binary_sha256.as_str(),
                    a.ptx_sha256.as_str(),
                    a.worker_sha256.as_str(),
                ]
            {
                return Err(format!(
                    "{}: staged hashes differ from the manifest. Nothing started; run vade install.",
                    machine.name
                ));
            }
        }
        Ok(())
    }

    fn run_owned_command(&mut self, exec: &Exec, cleanup: bool) -> Result<Output, String> {
        if cleanup {
            self.runner.cleanup(exec)
        } else {
            self.runner.run(exec)
        }
    }

    fn verify_listener(&mut self, session: &Session) -> Result<(), String> {
        for index in session.servers() {
            let reply = self.worker(
                session,
                &session.ranks[index],
                "listener",
                Fields::from([("port".into(), session.serve_port.to_string())]),
            )?;
            if !self.dry && reply.get("listener").map(String::as_str) != Some("owned") {
                return Err(format!(
                    "{} does not own the health listener. Run vade logs, then vade down.",
                    if session.replicas { session.ranks[index].machine.name.as_str() } else { "Rank zero" }
                ));
            }
        }
        Ok(())
    }

    fn down(&mut self) -> Result<(), String> {
        let Some(mut session) = Session::load(&self.config.state_dir)? else {
            let owned = self.group_records()?;
            if owned.is_empty() {
                crate::output::line(format_args!("Vade is already stopped. Nothing to stop."))?;
            } else {
                crate::output::line(format_args!("Vade has no server of its own to stop; groups still own servers."))?;
                self.print_group_records(&owned)?;
            }
            return Ok(());
        };
        self.cleanup(&mut session)?;
        crate::output::line(format_args!("Vade is stopped. The engines and the tunnel that vade started are closed."))?;
        Ok(())
    }

    fn cleanup(&mut self, session: &mut Session) -> Result<(), String> {
        // Cleanup must remain possible after an INT/TERM cancelled startup.
        // This only changes command cancellation; exact identity checks still
        // complete on every reachable rank before the first signal.
        session.phase = "stopping".into();
        if !self.dry {
            session.save(&self.config.state_dir)?;
        }
        // Validate all reachable ranks first; an identity refusal still blocks all signals.
        let mut unavailable = std::collections::BTreeSet::new();
        let mut errors = Vec::new();
        for i in 0..session.ranks.len() {
            if !session.ranks[i].attempted {
                continue;
            }
            let found = match self.worker(session, &session.ranks[i], "inspect", Fields::new()) {
                Ok(found) => found,
                Err(error) if error.starts_with("Transport:") => {
                    unavailable.insert(i);
                    errors.push(error);
                    continue;
                }
                Err(error) => return Err(error),
            };
            if !self.dry && found.get("phase").map(String::as_str) == Some("absent") {
                session.ranks[i].attempted = false;
                session.save(&self.config.state_dir)?;
            }
        }
        self.stop_gateway(session)?;
        for index in session.servers() {
            let socket = session.tunnel_socket(&self.config.state_dir, index);
            text::reject_symlinks(&socket)?;
            if self.dry || socket.exists() {
                let checked = self
                    .runner
                    .cleanup(&self.tunnel_at(session, index, Some("check")))?;
                if !self.dry && !checked.success {
                    remove_dead_socket(&socket)?;
                    crate::output::stderr_line(format_args!("The controller tunnel was already closed."));
                }
            }
            if self.dry || socket.exists() {
                let stopped = self
                    .runner
                    .cleanup(&self.tunnel_at(session, index, Some("exit")))?;
                if !stopped.success && !self.dry {
                    return Err(
                        "The owned tunnel did not stop. Keep the record and run vade down again."
                            .into(),
                    );
                }
            }
        }
        for (i, rank) in session
            .ranks
            .iter()
            .enumerate()
            .filter(|(_, r)| r.attempted)
        {
            if unavailable.contains(&i) {
                continue;
            }
            for op in ["stop", "gpu"] {
                if let Err(error) = self.worker(session, rank, op, Fields::new()) {
                    unavailable.insert(i);
                    errors.push(error);
                    break;
                }
            }
        }
        for i in 0..session.ranks.len() {
            if !session.ranks[i].attempted || unavailable.contains(&i) {
                continue;
            }
            let result = match self.worker(session, &session.ranks[i], "clean", Fields::new()) {
                Ok(result) => result,
                Err(error) => {
                    errors.push(error);
                    continue;
                }
            };
            if !self.dry {
                if let Some(log) = result.get("log") {
                    text::atomic_write(
                        &self.config.state_dir.join("logs").join(format!(
                            "{}-{}.log",
                            &session.token[..16],
                            session.ranks[i].machine.name
                        )),
                        log,
                    )?;
                }
                session.ranks[i].attempted = false;
                session.save(&self.config.state_dir)?;
            }
        }
        if !errors.is_empty() {
            return Err(errors.join("\n"));
        }
        if !self.dry {
            fs::remove_file(self.config.state_dir.join("session.conf")).map_err(|_| "Stopped, but cannot remove the ownership record. Check permissions and run vade down again.")?;
        } else {
            crate::output::line(format_args!("# remove the ownership record after confirmed cleanup"))?;
        }
        Ok(())
    }

    fn already_ready(&mut self, session: &Session) -> Result<(), String> {
        if session.phase != "running" || session.ranks.iter().any(|rank| !rank.attempted) {
            return Err("Vade has an unfinished ownership record. Run vade status, then vade down.".into());
        }
        self.check_starting_ranks(session)?;
        self.verify_listener(session)?;
        let tunnel = self.connection_ok(session)?;
        if !self.dry && (!tunnel || !self.local_health(session)?) {
            return Err("The recorded engine or tunnel is unhealthy. Run vade status, then vade down; nothing restarted.".into());
        }
        crate::output::line(format_args!("Vade is already running at {}/v1 (build {}). Chat: vade chat", session.endpoint(), session.build))
    }

    /// Without `VADE_GROUP`, the groups whose ownership record exists, also a
    /// group no longer in the config.
    pub(crate) fn group_records(&self) -> Result<Vec<String>, String> {
        if self.config.group.is_some() { return Ok(Vec::new()); }
        Ok(self.config.group_record_names()?.into_iter().filter(|name| self.config.group_state_dir(name).join("session.conf").is_file()).collect())
    }

    /// Name the groups that still own a server, and any of them with no
    /// `[group]` section.
    fn print_group_records(&self, owned: &[String]) -> Result<(), String> {
        crate::output::line(format_args!("Groups with an ownership record: {}. Run VADE_GROUP=NAME vade status to see one, VADE_GROUP=NAME vade down to stop one.", owned.join(", ")))?;
        let missing: Vec<&str> = owned.iter().filter(|name| self.config.groups.iter().all(|group| &group.name != *name)).map(String::as_str).collect();
        if !missing.is_empty() {
            crate::output::line(format_args!("No [group] section for {}: restore it in the controller config, then use VADE_GROUP=NAME vade status or vade down.", missing.join(", ")))?;
        }
        Ok(())
    }

    fn status(&mut self) -> Result<(), String> {
        let Some(session) = Session::load(&self.config.state_dir)? else {
            let owned = self.group_records()?;
            if owned.is_empty() {
                crate::output::text(format_args!("{}", crate::status::stopped(self.start_command())))?;
            } else {
                // Groups still run: never headline this as stopped.
                crate::output::line(format_args!("{} Vade has no server of its own; {} group(s) still own one.", style::paint(Token::Dim, "○", Stream::Out), owned.len()))?;
                self.print_group_records(&owned)?;
            }
            return Ok(());
        };
        // Rows print as each check finishes, in one label column.
        let row = |label: &str, value: &str| crate::output::text(format_args!("{}", style::table_at(&[(label, value.to_owned())], 2, 21)));
        crate::output::line(format_args!("{} is {}.", style::paint(Token::Title, "Vade", Stream::Out), session.phase))?;
        row("Machines", &session.ranks.iter().map(|r| r.machine.name.as_str()).collect::<Vec<_>>().join(", "))?;
        if session.replicas { row("Mode", "independent one-Spark replicas behind the local gateway")?; }
        row("Model", &session.model)?;
        row("Build", &session.build)?;
        row("MTP", &session.mtp_label())?;
        row("Context", &format!("{} tokens", session.context))?;
        row("Output limit", &format!("{} tokens{}", session.output,
            if session.supports_output { "" } else { " (engine fixed limit; harness limit applied)" }))?;
        row("Concurrent chats", &chats_label(&session))?;
        row("Memory budget", &session.memory_bytes.map(|bytes| format!("{bytes} bytes/rank")).unwrap_or_else(|| "automatic".into()))?;
        let mut all = session.ranks.iter().all(|r| r.attempted);
        for rank in session.ranks.iter().filter(|r| r.attempted) {
            match self.worker(&session, rank, "inspect", Fields::new()) {
                Ok(fields) => {
                    let alive = fields.get("alive").is_some_and(|v| v == "true");
                    all &= alive;
                    if !self.dry && !alive { crate::output::operation_failed(); }
                    row(&rank.machine.name, &if self.dry {
                        "not checked".to_owned()
                    } else if alive {
                        style::paint(Token::Ok, "running", Stream::Out)
                    } else {
                        style::paint(Token::Warn, "stopped", Stream::Out)
                    })?;
                }
                Err(error) => {
                    all = false;
                    crate::output::operation_failed();
                    row(&rank.machine.name, &text::terminal_text(&error))?;
                }
            }
        }
        if !self.dry && session.phase == "starting" && !all {
            return Err("The engine stopped, or never started, on one or more Sparks. Run vade logs, then vade down before another start.".into());
        }
        let tunnel = self.connection_ok(&session)?;
        if session.replicas {
            let state = if self.dry { "not checked" } else { match self.gateway_alive(&session)? { Some(true) => "running", Some(false) => "stopped", None => "not started" } };
            row("Gateway", &format!("{state} (log {})", text::terminal_text(&self.gateway_log(&session).to_string_lossy())))?;
        }
        let health_response = self.local_health_response(&session)?;
        let health =
            !self.dry && health_response.success && ready(&health_response.stdout, &session.model);
        if !self.dry && !(health && all && tunnel) { crate::output::operation_failed(); }
        let speed = if health {
            ["tok_s", "tokens_per_second", "tokens_per_sec"]
                .iter()
                .find_map(|key| crate::json::number(&health_response.stdout, key))
                .filter(|value| *value >= 0.0)
        } else {
            None
        };
        row("Controller connection", if self.dry { "not checked" } else if tunnel { "up" } else { "down" })?;
        row("Health", &if self.dry {
            "not checked".to_owned()
        } else if health && all && tunnel {
            style::paint(Token::Ok, "ready", Stream::Out)
        } else {
            style::paint(Token::Warn, "needs attention", Stream::Out)
        })?;
        row("Speed", &speed.map(|value| format!("{value:.1} tok/s"))
            .unwrap_or_else(|| if self.dry { "not checked" } else { "not reported by this engine" }.into()))?;
        if session.chained() {
            row("Multi-Spark path", &health.then(|| health::MultiSparkPath::from_health(&health_response.stdout)).flatten().map(|path| path.label())
                .unwrap_or_else(|| if self.dry { "not checked" } else { "not reported by this engine" }.into()))?;
        }
        let reported = if health { self.engine_prefix_cache(&session)? } else { None };
        row("Retained prompt cache", &retention_label(&session, reported))?;
        if health {
            for line in self.engine_events(&session, &health_response.stdout)? { crate::output::line(format_args!("  {line}"))?; }
        }
        if !self.dry && !(health && all && tunnel) {
            return Err("Vade needs attention. Run vade logs or vade doctor; use vade down before another start.".into());
        }
        crate::output::text(format_args!("\n{}", crate::status::next_steps(&[("Chat", "vade chat"), ("Stop", "vade down")])))
    }

    fn logs(&mut self) -> Result<(), String> {
        if let Some(session) = Session::load(&self.config.state_dir)? {
            for rank in session.ranks.iter().filter(|r| r.attempted) {
                let fields = self.worker(&session, rank, "logs", Fields::new())?;
                crate::output::line(format_args!("{}:", rank.machine.name))?;
                if let Some(log) = fields.get("log") {
                    crate::output::line(format_args!("{}", text::terminal_text(log)))?;
                }
            }
        } else {
            let dir = self.config.state_dir.join("logs");
            let mut paths = fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|e| e == "log"))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            paths.sort_by_key(|p| fs::metadata(p).and_then(|m| m.modified()).ok());
            if paths.is_empty() {
                crate::output::line(format_args!("No saved logs yet. Start: {}", self.start_command()))?;
            }
            for path in paths.into_iter().rev().take(6).rev() {
                text::reject_symlinks(&path)?;
                if let Ok(log) = fs::read_to_string(&path) {
                    crate::output::line(format_args!(
                        "{}:\n{}",
                        text::terminal_text(&path.file_name().unwrap_or_default().to_string_lossy()),
                        text::terminal_text(&log)
                    ))?;
                }
            }
        }
        Ok(())
    }

    fn chat(&mut self, options: &crate::cli::ChatOptions) -> Result<(), String> {
        let Some(session) = Session::load(&self.config.state_dir)? else {
            let owned = self.group_records()?;
            if owned.is_empty() {
                return Err(format!("Vade is stopped. Run {}, then vade chat.", self.start_command()));
            }
            return Err(format!("Vade has no server of its own; groups own servers: {}. Run VADE_GROUP=NAME vade chat for one of them.", owned.join(", ")));
        };
        if session.phase != "running" || session.ranks.iter().any(|r| !r.attempted) {
            return Err(
                "Engine startup or cleanup is unfinished. Run vade status before chat.".into(),
            );
        }
        let requested = self.requested_session(&UpOptions::default(), Some(&session), rdma_enabled()?)?;
        if !resident::same_launch(&session, &requested) {
            return Err("Vade already has an ownership record with different launch settings or build. Run vade down before changing it.".into());
        }
        self.verify_harness()?;
        for rank in &session.ranks {
            let fields = self.worker(&session, rank, "inspect", Fields::new())?;
            if !self.dry && fields.get("alive").map(String::as_str) != Some("true") {
                return Err("The engine stopped on one of the Sparks. Run vade status, then vade down.".into());
            }
        }
        if !self.connection_ok(&session)? {
            return Err("The SSH tunnel from this controller is down. Run vade status before you chat.".into());
        }
        if !self.local_health(&session)? && !self.dry {
            return Err("The engine is not ready. Run vade status or vade doctor.".into());
        }
        let workspace = std::env::current_dir().map_err(
            |_| "Current folder is unavailable. Open an existing folder, then run vade chat.",
        )?;
        let mut args = vec![
            "agent".into(),
            "--endpoint".into(),
            session.endpoint(),
            "--model".into(),
            session.model.clone(),
            "--workspace".into(),
            workspace.to_string_lossy().into_owned(),
            "--context-tokens".into(),
            session.context.to_string(),
        ];
        args.extend(crate::launch::chat_session_arguments(options));
        args.extend(crate::launch::chat_state_arguments()?);
        let mut exec = Exec::new(
            "Vade chat",
            &self.config.stage_dir.join("vade-harness").to_string_lossy(),
            args,
        );
        // Workspace checks need the macOS sandbox; match the local launcher.
        if cfg!(target_os = "linux") { exec.args.push("--no-check".into()); }
        exec.cwd = Some(workspace);
        clear_cloud_auth(&mut exec);
        exec.env.insert("VADE_CHAT_SETUP".into(), crate::launch::chat_setup(session.ranks.len(), !(session.mtp_explicit && session.mtp == 0)));
        if !self.runner.interactive(&exec)? {
            return Err("Chat ended with an error. Run vade doctor for the next step.".into());
        }
        Ok(())
    }

    fn sessions(&mut self, action: &[String]) -> Result<(), String> {
        self.verify_harness()?;
        let exec = crate::launch::sessions_exec(&self.config.stage_dir.join("vade-harness").to_string_lossy(), action)?;
        if !self.runner.interactive(&exec)? {
            return Err(crate::launch::SESSIONS_FAILED.into());
        }
        Ok(())
    }

    fn checked(&mut self, exec: Exec, message: &str) -> Result<Output, String> {
        let output = self.runner.run(&exec)?;
        if !output.success {
            if exec.args.iter().any(|arg| arg.contains(REMOTE_STORAGE_GUARD)) {
                if let Some(detail) = output.stdout.lines().find_map(|line| line.strip_prefix("VADE_STORAGE_ERROR ")) {
                    if let Some((kind, path)) = detail.split_once(' ') {
                        let host = exec.args.iter().rev().nth(1).map(String::as_str).unwrap_or("the remote machine");
                        return Err(if kind == "private" {
                            format!("Directory {path} on {host} is open to other users. Run chmod 700 {} there, then run vade install again.", quote(path))
                        } else {
                            format!("Cannot use build folder {path} on {host}. Choose a new folder owned by your account, with no symlinks or access for other users, then run vade install again.")
                        });
                    }
                }
            }
            return Err(message.into());
        }
        Ok(output)
    }
}

fn clear_cloud_auth(exec: &mut Exec) {
    crate::auth::forward(exec);
    exec.unset.extend(
        [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
            "ANTHROPIC_API_KEY",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
}
fn remove_dead_socket(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    let refusal =
        "Tunnel ownership could not be checked. Leave its files in place; see Recovery in cli/README.md.";
    text::private_dir(path.parent().ok_or(refusal)?)?;
    text::reject_symlinks(path)?;
    let original = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        _ => return Err(refusal.into()),
    };
    if !crate::control_socket::closed(path) {
        return Err(refusal.into());
    }
    match fs::symlink_metadata(path) {
        Ok(now) if now.file_type().is_socket() && now.dev() == original.dev() && now.ino() == original.ino() => {
            fs::remove_file(path).map_err(|_| "Cannot remove the closed tunnel's socket. Check state_dir permissions and retry vade down.".into())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => Err(refusal.into()),
    }
}


/// Concurrency for status: the engine's admitted count at ready, plus any request.
fn chats_label(session: &Session) -> String {
    let requested = session.chats.map(|chats| format!("; requested {chats}")).unwrap_or_default();
    match session.admitted_chats {
        // The ready line comes from the first replica; every replica is the same engine.
        Some(chats) if session.replicas => {
            let replicas = session.servers().count();
            format!("{chats} per replica, {} across {replicas} replicas (engine-admitted at this context{requested})", chats * replicas)
        }
        Some(chats) => format!("{chats} (engine-admitted at this context{requested})"),
        None => format!("engine-admitted from context and memory (count not reported{requested})"),
    }
}

fn ready(input: &str, model: &str) -> bool {
    // Parse just the top-level string fields needed for health, with no substring matches.
    crate::json::string(input, "status").as_deref() == Some("ready")
        && crate::json::string(input, "model").as_deref() == Some(model)
}

/// Status text for prefix retention: the engine's report when it answered,
/// otherwise what the saved launch settings select.
pub(crate) fn retention_label(session: &Session, reported: Option<bool>) -> String {
    match (reported, session.cache_bytes) {
        (Some(true), _) => "on (engine report)".into(),
        (Some(false), _) => "off (engine report)".into(),
        (None, Some(0)) => "off (--cache-bytes 0)".into(),
        (None, _) if session.fp4_local && session.prefix_restore() => "on by default; engine not checked".into(),
        (None, _) if session.fp4_local => "off by default on the layer chain; engine not checked".into(),
        (None, Some(bytes)) => format!("{bytes} bytes/rank"),
        (None, None) => "automatic".into(),
    }
}

/// One wording for the selected transport in launch plans, install and doctor.
pub(crate) fn transport_name(machines: usize, rdma: bool) -> &'static str {
    if machines == 1 { "local (one Spark; no fabric)" } else if rdma { "RDMA" } else { "TCP" }
}

fn rdma_enabled() -> Result<bool, String> {
    match std::env::var("SEQPAR_RDMA") {
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Err(std::env::VarError::NotPresent) => Ok(true),
        _ => Err("Set SEQPAR_RDMA to 0 for TCP over any reachable interface, or 1 for required RDMA. Unset it to keep the RDMA default.".into()),
    }
}

/// `~/.cargo/bin/<name>` when it exists, else the bare name for `PATH`.
pub(crate) fn rust_tool(name: &str) -> String {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|home| home.join(".cargo/bin").join(name))
        .filter(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| name.into())
}

#[cfg(test)]
mod tests;
