//! The controller tunnel, engine health probes and the startup readiness wait.
use super::*;

impl<R: Runner> App<R> {
    pub(super) fn tunnel_at(&self, session: &Session, index: usize, operation: Option<&str>) -> Exec {
        let socket = session.tunnel_socket(Path::new("."), index);
        let mut args = vec![
            "-o".into(),
            "BatchMode=yes".into(),
            "-o".into(),
            "ConnectTimeout=10".into(),
            "-o".into(),
            "ServerAliveInterval=10".into(),
            "-o".into(),
            "ServerAliveCountMax=2".into(),
            "-o".into(),
            "PermitLocalCommand=no".into(),
            "-S".into(),
            socket.to_string_lossy().into_owned(),
        ];
        if let Some(op) = operation {
            args.extend(["-O".into(), op.into()]);
        } else {
            args.extend([
                "-o".into(),
                "ExitOnForwardFailure=yes".into(),
                "-o".into(),
                "ControlPersist=no".into(),
                "-M".into(),
                "-fNT".into(),
                "-L".into(),
                format!(
                    "127.0.0.1:{}:127.0.0.1:{}",
                    session.tunnel_port(index), session.serve_port
                ),
            ]);
        }
        args.push(session.ranks[index].machine.ssh.clone());
        let mut exec = Exec::new(
            &format!("controller tunnel {}", operation.unwrap_or("start")),
            "ssh",
            args,
        );
        // OpenSSH opens its control socket relative to this private directory.
        // Start, status and stop all use the same base without changing our cwd.
        exec.cwd = Some(self.config.state_dir.clone());
        exec
    }

    pub(super) fn remote_health_at(&self, session: &Session, index: usize) -> Exec {
        runner::ssh(
            &session.ranks[index].machine.ssh,
            format!(
                "curl --noproxy '*' --fail --silent --show-error --max-time 5 {}",
                quote(&format!("http://127.0.0.1:{}/health", session.serve_port))
            ),
            "engine health",
        )
    }
    pub(super) fn local_health(&mut self, session: &Session) -> Result<bool, String> {
        let output = self.local_health_response(session)?;
        Ok(!self.dry && output.success && ready(&output.stdout, &session.model))
    }
    pub(super) fn local_health_response(&mut self, session: &Session) -> Result<Output, String> {
        self.verify_listener(session)?;
        self.runner.run(&Exec::new(
            "controller health",
            "curl",
            vec![
                "--noproxy".into(),
                "*".into(),
                "--fail".into(),
                "--silent".into(),
                "--show-error".into(),
                "--max-time".into(),
                "5".into(),
                format!("{}/health", session.endpoint()),
            ],
        ))
    }
    /// The engine's own prefix-retention switch, from its capabilities.
    pub(super) fn engine_prefix_cache(&mut self, session: &Session) -> Result<Option<bool>, String> {
        let output = self.runner.run(&Exec::new("controller capabilities", "curl", vec![
            "--noproxy".into(), "*".into(), "--fail".into(), "--silent".into(), "--show-error".into(),
            "--max-time".into(), "5".into(), format!("{}/v1/capabilities", session.endpoint()),
        ]))?;
        if !output.success { return Ok(None); }
        Ok(match crate::evals::json::parse(&output.stdout) {
            Ok(value) => match value["prefix_cache"] { crate::evals::json::Value::Bool(on) => Some(on), _ => None },
            Err(_) => None,
        })
    }
    pub(super) fn check_starting_ranks(&mut self, session: &Session) -> Result<(), String> {
        for (index, rank) in session
            .ranks
            .iter()
            .enumerate()
            .filter(|(_, r)| r.attempted)
        {
            let found = self.worker(session, rank, "inspect", Fields::new())?;
            if self.dry || found.get("alive").is_some_and(|v| v == "true") {
                continue;
            }
            return Err(self.startup_error(session, index, "exited during startup."));
        }
        Ok(())
    }

    pub(super) fn startup_error(&mut self, session: &Session, index: usize, cause: &str) -> String {
        let rank = &session.ranks[index];
        let log = match self.worker(session, rank, "logs", Fields::new()) {
            Ok(fields) => {
                let mut lines = fields
                    .get("log")
                    .map(String::as_str)
                    .unwrap_or("")
                    .lines()
                    .rev()
                    .take(20)
                    .collect::<Vec<_>>();
                lines.reverse();
                if lines.is_empty() {
                    "No engine log lines were recorded.".into()
                } else {
                    lines.join("\n")
                }
            }
            Err(error) => format!("Could not read the engine log: {error}"),
        };
        format!(
            "{} (rank {index}) {cause}\nLast engine log lines:\n{log}",
            rank.machine.name
        )
    }

    pub(super) fn wait_ready(&mut self, session: &Session, remote: bool, seconds: u64) -> Result<(), String> {
        let began = Instant::now();
        let diagnostic = self.config.state_dir.join(if remote { "last-engine-readiness.log" } else { "last-tunnel-readiness.log" });
        let deadline = began + Duration::from_secs(seconds);
        let mut progress = runner::Progress::new(if remote { "waiting for engine readiness" } else { "waiting for the controller tunnel" });
        // Remote readiness covers every serving rank: rank zero, or each replica.
        let mut pending = if remote { session.servers().collect::<Vec<_>>() } else { vec![0] };
        while self.dry || Instant::now() < deadline {
            if self.runner.interrupted() {
                return Err("Startup interrupted while waiting for readiness.".into());
            }
            self.check_starting_ranks(session)?;
            let mut exec = if remote {
                self.remote_health_at(session, pending[0])
            } else {
                Exec::new(
                    "controller health",
                    "curl",
                    vec![
                        "--noproxy".into(),
                        "*".into(),
                        "--fail".into(),
                        "--silent".into(),
                        "--show-error".into(),
                        "--max-time".into(),
                        "5".into(),
                        format!("{}/health", session.endpoint()),
                    ],
                )
            };
            exec.timeout = exec
                .timeout
                .min(deadline.saturating_duration_since(Instant::now()));
            // Connection refusal is expected while the owned process loads.
            // Startup errors still include the owned engine's retained log.
            exec.transient_probe = true;
            exec.probe_log = Some(diagnostic.clone());
            let output = self.runner.run(&exec)?;
            self.check_starting_ranks(session)?;
            let healthy = output.success && ready(&output.stdout, &session.model);
            if self.dry || healthy {
                pending.remove(0);
                if !pending.is_empty() {
                    continue;
                }
                self.verify_listener(session)?;
                if !self.dry { let _ = fs::remove_file(&diagnostic); }
                return Ok(());
            }
            // A replica gateway that exits after it started will never answer.
            if !remote && session.replicas && self.gateway_alive(session)? == Some(false) {
                return Err(self.gateway_exited(session));
            }
            progress.report(began.elapsed());
            self.runner.pause(
                Duration::from_secs(2).min(deadline.saturating_duration_since(Instant::now())),
            );
        }
        Err(format!(
            "{} did not become ready. Last probe stdout/stderr: {}. Run vade logs, then vade doctor.",
            if remote {
                "The engine"
            } else {
                "The controller tunnel"
            },
            text::terminal_text(&diagnostic.to_string_lossy())
        ))
    }
}

/// The path the engine's receipt names in a health reply (`vade.numerics.path`): the tensor split
/// or the layer pipeline and, for the split, its levels. None: the reply does not name one (one
/// Spark, or an older engine).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TwoSparkPath { pub path: String, pub projection_level: Option<f64>, pub wire: Option<String>, pub head: Option<String> }
impl TwoSparkPath {
    pub(crate) fn from_health(health: &str) -> Option<Self> {
        let split = |key| crate::json::path_string(health, &["vade", "numerics", "tensor_split", key]);
        Some(Self {
            path: crate::json::path_string(health, &["vade", "numerics", "path"])?,
            projection_level: crate::json::path_number(health, &["vade", "numerics", "tensor_split", "projection_level"]),
            wire: split("wire"),
            head: split("head"),
        })
    }
    /// One status line: `tensor split (projection level 3, write wire, head split by vocabulary halves)`.
    pub(crate) fn label(&self) -> String {
        match self.path.as_str() {
            "tensor_split" => format!("tensor split (projection level {}, {} wire, {})",
                self.projection_level.map_or("unknown".into(), |level| format!("{level}")),
                text::terminal_text(self.wire.as_deref().unwrap_or("unknown")),
                match self.head.as_deref() { Some("vocabulary_halves") => "head split by vocabulary halves".into(), Some("whole") => "whole head".into(),
                    other => format!("head {}", text::terminal_text(other.unwrap_or("unknown"))) }),
            "layer_pipeline" => "layer pipeline (VADE_W2_PATH=pipeline, or TCP)".into(),
            other => text::terminal_text(other).into_owned(),
        }
    }
}
