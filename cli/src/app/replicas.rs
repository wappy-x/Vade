//! `vade up --replicas N`: independent servers (one Spark each, or a tensor-split
//! pair each with `--pairs`), one tunnel to each server's first rank, and a local
//! vade-gateway on local_port. The gateway is an owned process: it is signaled
//! only while its PID still has the recorded start time and command.
use super::*;

impl<R: Runner> App<R> {
    /// Every owned tunnel answers and a replica gateway is still the recorded process.
    pub(super) fn connection_ok(&mut self, session: &Session) -> Result<bool, String> {
        let mut ok = true;
        for index in session.servers().filter(|&index| tunneled(session, index)) {
            ok &= self.runner.run(&self.tunnel_at(session, index, Some("check")))?.success;
        }
        if session.replicas && ok && !self.dry {
            ok = self.gateway_alive(session)? == Some(true);
        }
        Ok(ok)
    }

    pub(super) fn start_tunnels(&mut self, session: &Session) -> Result<(), String> {
        for index in session.servers().filter(|&index| tunneled(session, index)) {
            self.checked(self.tunnel_at(session, index, None), &format!("The controller tunnel could not start. Check rank {index}'s ssh entry and local_port; run vade status."))?;
            self.checked(
                self.tunnel_at(session, index, Some("check")),
                "The tunnel did not respond. Run vade status, then vade down.",
            )?;
        }
        Ok(())
    }

    pub(super) fn gateway_log(&self, session: &Session) -> std::path::PathBuf {
        self.config.state_dir.join("logs").join(format!("{}-gateway.log", &session.token[..16]))
    }

    /// Start the staged harness's vade-gateway in front of every replica tunnel.
    pub(super) fn start_gateway(&mut self, session: &mut Session) -> Result<(), String> {
        self.verify_harness()?;
        let mut args = vec!["vade-gateway".into(), "--bind".into(), format!("127.0.0.1:{}", session.local_port)];
        for index in session.servers() {
            args.extend(["--upstream".into(), format!("http://127.0.0.1:{}", session.tunnel_port(index))]);
        }
        // Pairs keep a chat's prefix cache: the gateway gives each chat a stable
        // prompt_cache_key and cache:true, so a later turn on its pair restores it.
        if session.chained() {
            args.push("--stable-prefix".into());
        }
        let mut exec = Exec::new("replica gateway", &self.config.stage_dir.join("vade-harness").to_string_lossy(), args);
        clear_cloud_auth(&mut exec);
        // up probes this gateway's /health without a key and starts engines without one,
        // so no gateway token, upstream or key file may come in from the shell; client
        // credentials then pass through unchanged (the gateway's no-key behavior).
        for key in ["VADE_GATEWAY_TOKEN", "VADE_GATEWAY_UPSTREAM", "VADE_API_KEY_FILE"] {
            exec.env.remove(key);
            exec.unset.push(key.into());
        }
        let pid = self.runner.spawn(&exec, &self.gateway_log(session))?;
        if self.dry {
            return Ok(());
        }
        // A gateway that already exited is a zombie until `up` ends: ps still answers
        // for its PID, with `<defunct>` as the command. Record only the owned command.
        let port = session.local_port;
        let started = self.process_identity(pid)?.filter(|(_, command)| owned_gateway(command, port))
            .map(|(started, _)| started).ok_or_else(|| self.gateway_exited(session))?;
        session.gateway = Some((pid, started));
        session.save(&self.config.state_dir)
    }

    pub(super) fn gateway_exited(&self, session: &Session) -> String {
        format!(
            "The replica gateway exited at startup; its log is {}. Run vade down, then vade install to stage a harness with vade-gateway, then vade up.",
            crate::text::terminal_text(&self.gateway_log(session).to_string_lossy())
        )
    }

    /// One `ps` column of `pid`, or None once the process is gone. Each column is read
    /// alone under LC_ALL=C: the start date's words and field count follow the locale
    /// otherwise, so no split between date and command can depend on it.
    fn process_field(&mut self, label: &str, column: &str, pid: u32) -> Result<Option<String>, String> {
        let mut exec = Exec::new(label, "ps", vec!["-o".into(), format!("{column}="), "-p".into(), pid.to_string()]);
        exec.env.insert("LC_ALL".into(), "C".into());
        exec.cancellable = false;
        // ps fails once the process is gone; that answer is expected, not an error log.
        exec.transient_probe = true;
        let output = self.runner.run(&exec)?;
        let value = output.stdout.lines().next().unwrap_or("").trim().to_owned();
        Ok(Some(value).filter(|value| output.success && !value.is_empty()))
    }

    /// Start time (whitespace-normalized) and command line of `pid`.
    fn process_identity(&mut self, pid: u32) -> Result<Option<(String, String)>, String> {
        let Some(started) = self.process_field("gateway identity", "lstart", pid)? else { return Ok(None) };
        let started = started.split_whitespace().collect::<Vec<_>>().join(" ");
        Ok(self.process_field("gateway command", "command", pid)?.map(|command| (started, command)))
    }

    /// None without a recorded gateway; Some(false) after exit or PID reuse.
    pub(super) fn gateway_alive(&mut self, session: &Session) -> Result<Option<bool>, String> {
        let Some((pid, started)) = session.gateway.clone() else { return Ok(None) };
        let port = session.local_port;
        Ok(Some(self.process_identity(pid)?.is_some_and(|(start, command)| {
            start == started.split_whitespace().collect::<Vec<_>>().join(" ") && owned_gateway(&command, port)
        })))
    }

    /// Stop exactly the recorded gateway; an exited or reused PID is never signaled.
    pub(super) fn stop_gateway(&mut self, session: &mut Session) -> Result<(), String> {
        let Some((pid, _)) = session.gateway.clone() else { return Ok(()) };
        for signal in ["-TERM", "-KILL"] {
            if self.gateway_alive(session)? != Some(true) {
                break;
            }
            self.runner.cleanup(&Exec::new("replica gateway stop", "kill", vec![signal.into(), pid.to_string()]))?;
            for _ in 0..50 {
                if self.dry || self.gateway_alive(session)? != Some(true) {
                    break;
                }
                self.runner.pause(Duration::from_millis(100));
            }
        }
        if !self.dry && self.gateway_alive(session)? == Some(true) {
            return Err("The replica gateway did not stop. Keep the record and run vade down again.".into());
        }
        session.gateway = None;
        if !self.dry {
            session.save(&self.config.state_dir)?;
        }
        Ok(())
    }
}

/// The command line `vade up` gave this session's gateway (never `<defunct>`).
fn owned_gateway(command: &str, port: u16) -> bool {
    format!("{command} ").contains(&format!(" vade-gateway --bind 127.0.0.1:{port} "))
}
