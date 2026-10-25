//! Prerequisite checks, model guidance and harness diagnostics.
use super::*;
use crate::checks::Report;

/// What one Spark's prerequisite script found.
#[derive(Default)]
pub(super) struct Probe {
    /// Whether the model files were found; `None` when an earlier failure
    /// stopped the script before it looked.
    present: Option<bool>,
    memory: Option<(u64, u64)>,
    disk_kb: Option<u64>,
    gpu: Option<String>,
    rdma: bool,
    /// The failed prerequisite: `disk`, a missing or failing tool, `rdma-*`,
    /// `driver` (the GPU query failed), `ssh` (the script never started) or
    /// `script` (it stopped without naming a check).
    missing: Option<String>,
    /// The message `vade install` stops with for `missing`.
    error: Option<String>,
}

/// One row for a whole stage whose own output was captured: `planned` in a
/// dry run, `passed` when it passed, its error when it failed.
fn stage(report: &mut Report, dry: bool, name: &str, result: Result<(), String>, passed: &str, planned: &str, next: &str) -> Result<(), String> {
    match result {
        Ok(()) if dry => report.plan(name, planned),
        Ok(()) => report.ok(name, passed),
        Err(error) => report.fail(name, &error, next),
    }
}
impl<R: Runner> App<R> {
    /// `vade doctor` from a controller: one row per check (this computer's
    /// tools, then SSH, nvidia-smi, disk, free memory and model files on each
    /// Spark, the fabric, the local port and the engine), then PASS or FAIL.
    /// A failed check never stops the rest.
    pub(super) fn doctor(&mut self) -> Result<(), String> {
        let transport = if self.config.machines.len() > 1 { rdma_enabled() } else { Ok(false) };
        self.doctor_with(transport)
    }
    /// [`Self::doctor`] with the SEQPAR_RDMA selection already read.
    pub(super) fn doctor_with(&mut self, transport: Result<bool, String>) -> Result<(), String> {
        let mut report = Report::new("controller", self.dry);
        let (result, _) = crate::output::buffered(|| self.local_checks());
        stage(&mut report, self.dry, "controller tools", result, "rustc, cargo, rsync, curl and ssh found", "rustc, cargo, rsync, curl, ssh -V",
            "Install Rust, rsync, curl and OpenSSH, then run vade doctor.")?;
        let machines = self.config.machines.clone();
        // An invalid SEQPAR_RDMA stops every Spark probe and the fabric check;
        // report it once and skip what it hides.
        if let Err(error) = &transport {
            report.fail("SEQPAR_RDMA", error, "Set SEQPAR_RDMA=0 for TCP or 1 for RDMA (or unset it), then run vade doctor.")?;
        }
        for machine in &machines {
            if transport.is_err() {
                for check in ["ssh", "nvidia-smi", "disk", "free memory", "model files"] { report.skip(&format!("{} {check}", machine.name), "needs a valid SEQPAR_RDMA")?; }
                continue;
            }
            let (probe, _) = crate::output::buffered(|| self.probe_machine(machine, self.config.prebuilt_dir.is_none()));
            self.machine_rows(&mut report, machine, probe)?;
        }
        match transport {
            _ if machines.len() == 1 => report.skip("fabric", "one Spark; no fabric")?,
            Err(_) => report.skip("fabric", "needs a valid SEQPAR_RDMA")?,
            Ok(rdma) => {
                let (result, _) = crate::output::buffered(|| self.network_checks(&machines));
                stage(&mut report, self.dry, "fabric", result, if rdma { "every fabric link answered ping" } else { "TCP; engine startup checks the addresses" },
                    "link addresses in the config", "Check both link addresses and connectivity, then run vade doctor.")?;
            }
        }
        let session = match Session::load(&self.config.state_dir) {
            Ok(session) => session,
            Err(error) => {
                report.fail("ownership", &error, "Preserve the state folder; run vade status, then vade down.")?;
                return report.finish();
            }
        };
        let port = self.config.local_port;
        if self.dry {
            report.plan(&format!("port {port}"), "free on this computer, or held by the vade tunnel")?;
        } else if session.is_some() {
            report.skip(&format!("port {port}"), "held by the vade tunnel")?;
        } else {
            report.check(&format!("port {port}"), self.runner.port_free(port), |_| "free on this computer".into(),
                &format!("Close the program listening on port {port}, or change local_port in the config, then run vade doctor."))?;
        }
        if session.is_none() && !self.dry {
            let available = self.verify_harness().and_then(|()| self.checked(Exec::new("harness doctor availability",
                &self.config.stage_dir.join("vade-harness").to_string_lossy(), vec!["doctor".into(), "--help".into()]),
                "Harness doctor is unavailable. Run vade install to build it.").map(drop));
            report.check("harness", available, |_| "installed and matches its saved hash".into(), "Run vade install, then vade doctor.")?;
            report.skip("engine health", "stopped; start: vade up")?;
        } else {
            let (result, _) = crate::output::buffered(|| self.harness_doctor(false));
            stage(&mut report, self.dry, "engine health", result, "ranks alive, tunnel up, /health ready, harness turns passed", "ranks, tunnel, /health and two small harness turns",
                "Run vade status, then vade logs.")?;
        }
        report.finish()
    }

    /// The rows for one Spark's probe: SSH, nvidia-smi, disk, free memory and
    /// model files. An earlier failure skips the rows it hides.
    fn machine_rows(&self, report: &mut Report, machine: &Machine, probe: Result<Probe, String>) -> Result<(), String> {
        let name = |check: &str| format!("{} {check}", machine.name);
        let rerun = "then run vade doctor.";
        if self.dry {
            report.plan(&name("ssh"), &format!("ssh {} runs one read-only check script", machine.ssh))?;
            for check in ["nvidia-smi", "disk", "free memory", "model files"] { report.plan(&name(check), "read by that script")?; }
            return Ok(());
        }
        let probe = match probe {
            Ok(probe) => probe,
            Err(error) => {
                report.fail(&name("ssh"), &error, &format!("Run ssh {} true to see the SSH error, {rerun}", machine.ssh))?;
                for check in ["nvidia-smi", "disk", "free memory", "model files"] { report.skip(&name(check), "needs SSH")?; }
                return Ok(());
            }
        };
        let missing = probe.missing.as_deref();
        if missing == Some("ssh") {
            report.fail(&name("ssh"), &probe.error.clone().unwrap_or_default(), &format!("Run ssh {} true to see the SSH error, {rerun}", machine.ssh))?;
            for check in ["nvidia-smi", "disk", "free memory", "model files"] { report.skip(&name(check), "needs SSH")?; }
            return Ok(());
        }
        report.ok(&name("ssh"), &format!("reached {}", machine.ssh))?;
        match missing {
            Some("script") => {
                report.fail(&name("check script"), &probe.error.clone().unwrap_or_default(), &format!("Rerun vade doctor; if this repeats, check the login shell with ssh {} true.", machine.ssh))?;
                for check in ["nvidia-smi", "disk", "free memory", "model files"] { report.skip(&name(check), "not checked; the script stopped early")?; }
                return Ok(());
            }
            Some(tool) if tool != "disk" && !tool.starts_with("rdma") => {
                let error = probe.error.clone().unwrap_or_default();
                if tool == "nvidia-smi" || tool == "driver" {
                    report.fail(&name("nvidia-smi"), &error, &format!("Run ssh {} nvidia-smi to inspect the driver, {rerun}", machine.ssh))?;
                } else {
                    // The script looks for cargo and rustc before nvidia-smi.
                    if matches!(tool, "cargo" | "rustc") { report.skip(&name("nvidia-smi"), &format!("not checked; the script stopped at {tool}"))?; }
                    else { report.ok(&name("nvidia-smi"), "present")?; }
                    report.fail(&name("tools"), &error, &format!("Install {tool} on {}, {rerun}", machine.name))?;
                }
                for check in ["disk", "free memory", "model files"] { report.skip(&name(check), "needs the tool checks")?; }
                return Ok(());
            }
            _ => report.ok(&name("nvidia-smi"), probe.gpu.as_deref().unwrap_or("driver answered"))?,
        }
        let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let free = probe.disk_kb.map_or_else(|| "free space unknown".into(), |kb| format!("{:.1} GiB free in the home folder", gib(kb.saturating_mul(1024))));
        if missing == Some("disk") {
            report.fail(&name("disk"), &probe.error.clone().unwrap_or(free), &format!("Free disk space on {}, {rerun}", machine.name))?;
        } else {
            report.ok(&name("disk"), &free)?;
        }
        match probe.memory {
            Some((total, available)) => report.ok(&name("free memory"), &format!("{:.1} GiB available of {:.1} GiB (GPU and CPU share this memory)", gib(available), gib(total)))?,
            None => report.skip(&name("free memory"), "/proc/meminfo unreadable")?,
        }
        if let Some(error) = probe.error.as_ref().filter(|_| missing.is_some_and(|m| m.starts_with("rdma"))) {
            report.fail(&name("rdma"), error, "Have the administrator prepare RDMA, or set SEQPAR_RDMA=0 for TCP, then run vade doctor.")?;
        }
        match probe.present {
            Some(true) => report.ok(&name("model files"), &format!("{} (config, tokenizer, weights; vade install checks every SHA256)", machine.model_path))?,
            Some(false) => report.fail(&name("model files"), &format!("missing or incomplete at {}", machine.model_path),
                &format!("Copy {} at {} there (INSTALL.md step 3), then run vade install.", self.config.model_repo, self.config.model_revision))?,
            None => report.skip(&name("model files"), &format!("not checked; the script stopped at {}", missing.unwrap_or("an earlier check")))?,
        }
        Ok(())
    }

    pub(super) fn local_checks(&mut self) -> Result<(), String> {
        self.local_checks_for_build(true)
    }
    pub(super) fn local_checks_for_build(&mut self, source_build: bool) -> Result<(), String> {
        for (program, args) in [
            (rust_tool("rustc"), vec!["--version".into()]),
            (rust_tool("cargo"), vec!["--version".into()]),
            ("rsync".into(), vec!["--version".into()]),
            ("curl".into(), vec!["--version".into()]),
        ] {
            if !source_build && (program == rust_tool("rustc") || program == rust_tool("cargo")) { continue; }
            self.checked(Exec::new(&format!("local {program}"),&program,args),"A controller prerequisite is unavailable. Install Rust, rsync, and curl, then run vade doctor.")?;
        }
        // -V only prints OpenSSH's version; it never connects.
        self.checked(
            Exec::new("local SSH", "ssh", vec!["-V".into()]),
            "OpenSSH is unavailable. Install SSH, then run vade doctor.",
        )?;
        crate::output::line(format_args!(
            "Controller tools: {}.",
            if self.dry { "checks planned" } else { "ready" }
        ))?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn machine_checks(&mut self, machine: &Machine) -> Result<bool, String> {
        self.machine_checks_for_build(machine, self.config.prebuilt_dir.is_none())
    }
    /// Run one Spark's read-only prerequisite script over SSH and read what
    /// it found. Only a failed SSH command or unreadable output is an `Err`;
    /// a failed prerequisite is `missing` plus the message install prints.
    fn probe_machine(&mut self, machine: &Machine, source_build: bool) -> Result<Probe, String> {
        let rdma = self.config.machines.len() > 1 && rdma_enabled()?;
        // `reached` proves SSH ran the script; `missing` and `failed` name the
        // check that stopped it.
        let command = format!(
            r#"set -eu
printf 'reached=1\n'
export PATH="$HOME/.cargo/bin:$PATH"
for tool in {compiler_tools} nvidia-smi sha256sum rsync curl {ping}; do
  command -v "$tool" >/dev/null 2>&1 || {{ printf 'missing=%s\n' "$tool"; exit 1; }}
done
{compiler_versions}
gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) || {{ printf 'failed=driver\n'; exit 1; }}
printf 'gpu=%s\n' "$(printf '%s\n' "$gpu" | head -n 1)"
available=$(df -Pk "$HOME" | awk 'NR == 2 {{print $4}}')
printf 'disk_kb=%s\n' "$available"
awk '/^MemTotal:/ {{print "mem_total_kb=" $2}} /^MemAvailable:/ {{print "mem_available_kb=" $2}}' /proc/meminfo 2>/dev/null || true
test "$available" -ge {disk_kb} || {{ printf 'missing=disk\n'; exit 1; }}
if test {rdma} = 1; then
  command -v ldconfig >/dev/null 2>&1 && ldconfig -p | /usr/bin/grep -Fq 'libibverbs.so.1' || {{ printf 'missing=rdma-library\n'; exit 1; }}
  rdma_device=0
  for device in /dev/infiniband/uverbs*; do
    if test -r "$device" && test -w "$device"; then rdma_device=1; break; fi
  done
  test "$rdma_device" = 1 || {{ printf 'missing=rdma-device\n'; exit 1; }}
  test "$(ulimit -Sl)" = unlimited && test "$(ulimit -Hl)" = unlimited || {{ printf 'missing=rdma-memlock\n'; exit 1; }}
fi
if test -d {model} && test -f {model}/config.json && test -f {model}/tokenizer.json && {{ test -f {model}/model.safetensors.index.json || find {model} -maxdepth 1 -name '*.safetensors' -print -quit | /usr/bin/grep -q .; }}; then
  printf 'weights=present\n'
else
  printf 'weights=missing\n'
fi
printf 'rust=ok\ndriver=ok\n'
"#,
            compiler_tools = if source_build { "cargo rustc" } else { "" },
            compiler_versions = if source_build {
                "cargo --version >/dev/null 2>&1 || { printf 'failed=cargo\\n'; exit 1; }\nrustc --version >/dev/null 2>&1 || { printf 'failed=rustc\\n'; exit 1; }"
            } else { "" },
            disk_kb = if source_build { 10485760 } else { 1048576 },
            model = remote_path(&machine.model_path),
            rdma = u8::from(rdma),
            ping = if rdma { "ping" } else { "" }
        );
        let output = self.runner.run(&runner::ssh(
            &machine.ssh,
            command,
            &format!("{} prerequisites", machine.name),
        ))?;
        if self.dry {
            crate::output::line(format_args!("{}: prerequisite checks planned.", machine.name))?;
            return Ok(Probe { present: Some(true), ..Probe::default() });
        }
        let fields = text::parse(&output.stdout)?;
        let kib = |key: &str| fields.get(key).and_then(|v| v.parse::<u64>().ok()).and_then(|v| v.checked_mul(1024));
        let mut probe = Probe {
            present: fields.get("weights").map(|v| v == "present"),
            memory: kib("mem_total_kb").zip(kib("mem_available_kb")),
            disk_kb: fields.get("disk_kb").and_then(|v| v.parse().ok()),
            gpu: fields.get("gpu").map(|gpu| text::terminal_text(gpu).into_owned()).filter(|gpu| !gpu.is_empty()),
            rdma,
            missing: None,
            error: None,
        };
        if !output.success {
            let failed = fields.get("failed").map(String::as_str);
            let missing = match (fields.get("missing"), failed) {
                (Some(tool), _) => tool.as_str(),
                (None, Some(failed)) => failed,
                // No `reached`: SSH itself (or the login shell) failed first.
                _ if !fields.contains_key("reached") => "ssh",
                _ => "script",
            };
            probe.missing = Some(missing.into());
            probe.error = Some(match fields.get("missing").map(String::as_str) {
                Some("disk") => format!(
                    "{} needs at least {} GiB free for {}. Free disk space, then run vade install.",
                    machine.name, if source_build { 10 } else { 1 }, if source_build { "a source build" } else { "prebuilt installation" }
                ),
                Some(reason @ ("rdma-library" | "rdma-device" | "rdma-memlock")) => {
                    let needed = match reason {
                        "rdma-library" => "the OS libibverbs.so.1 library and ldconfig",
                        "rdma-device" => "a readable and writable RDMA verbs device",
                        _ => "unlimited soft and hard memlock in this SSH session",
                    };
                    format!("{} needs {needed} for RDMA. Have the administrator prepare RDMA, or set SEQPAR_RDMA=0 for TCP over the existing network, then rerun this command. Vade does not change system settings.", machine.name)
                }
                Some(tool)
                    if ["cargo", "rustc", "nvidia-smi", "sha256sum", "rsync", "ping", "curl"]
                        .contains(&tool) =>
                {
                    format!(
                        "{} is missing {tool}. Install that prerequisite, then run vade doctor.",
                        machine.name
                    )
                }
                _ => match missing {
                    "driver" => format!("{} has nvidia-smi, but its GPU query failed. Run ssh {} nvidia-smi to see the driver error, then run vade doctor.", machine.name, machine.ssh),
                    tool @ ("cargo" | "rustc") => format!("{} has {tool}, but {tool} --version failed there. Repair that Rust toolchain, then run vade doctor.", machine.name),
                    "ssh" => format!("ssh {} did not run the check script. Run ssh {} true to see the SSH error, then run vade doctor.", machine.ssh, machine.ssh),
                    _ => format!("{}'s check script stopped before naming a failed check. Rerun vade doctor; if this repeats, check the login shell with ssh {} true.", machine.name, machine.ssh),
                },
            });
        }
        Ok(probe)
    }

    pub(super) fn machine_checks_for_build(&mut self, machine: &Machine, source_build: bool) -> Result<bool, String> {
        let probe = self.probe_machine(machine, source_build)?;
        if self.dry { return Ok(true); }
        if let Some(error) = probe.error { return Err(error); }
        let (present, rdma) = (probe.present == Some(true), probe.rdma);
        let memory = probe.memory.map_or_else(String::new, |(total, available)| format!("; {}", crate::detect::memory_summary(total, available)));
        crate::output::line(format_args!(
            "{}: driver and install disk ready{memory}; transport: {}{}; model files {}.",
            machine.name,
            transport_name(self.config.machines.len(), rdma),
            if rdma { " (prerequisites checked)" } else { "" },
            if present {
                "found (inventory check follows)"
            } else {
                "missing"
            }
        ))?;
        if !present {
            self.model_copy_instructions(machine)?;
        }
        Ok(present)
    }

    pub(super) fn model_copy_instructions(&self, machine: &Machine) -> Result<(), String> {
        crate::output::line(format_args!(
            "{}: the official checkpoint is missing or incomplete.\n  Model: {}\n  Revision: {}\n  Destination: {}\nFollow INSTALL.md step 3: copy this verified checkpoint revision to the configured destination on each machine. Verify the source and every destination copy against a trusted full-file SHA256 manifest, including supporting files, then rerun vade install.",
            machine.name,
            self.config.model_repo,
            self.config.model_revision,
            machine.model_path
        ))?;
        Ok(())
    }

    pub(super) fn network_checks(&mut self, machines: &[Machine]) -> Result<(), String> {
        let rdma = self.config.machines.len() > 1 && rdma_enabled()?;
        let mut count = 0;
        for machine in machines {
            for peer_name in machine.fabric.keys() {
                let Some(peer) = machines.iter().find(|m| m.name == *peer_name) else {
                    continue;
                };
                let Some(address) = peer.fabric.get(&machine.name) else {
                    return Err("A fabric link has only one end in config. Add its reciprocal fabric entry.".into());
                };
                if rdma {
                    self.checked(runner::ssh(&machine.ssh,format!("ping -n -c 1 -W 2 {} >/dev/null",quote(address)),&format!("{} to {} fabric",machine.name,peer.name)),
                        &format!("Fabric from {} to {} did not reply. Check both link addresses and connectivity, then run vade doctor.",machine.name,peer.name))?;
                }
                count += 1;
            }
        }
        if !rdma {
            crate::output::line(format_args!(
                "TCP: skipping ICMP reachability checks for {count} directed links. \
                 Transport listeners may not be running yet; engine startup checks the configured addresses and ports."
            ))?;
            return Ok(());
        }
        crate::output::line(format_args!(
            "Fabric: {count} directed link {}. Transport ports are checked by engine startup.",
            if self.dry {
                "checks planned"
            } else {
                "checks passed"
            }
        ))?;
        Ok(())
    }

    pub(super) fn harness_doctor(&mut self, installing: bool) -> Result<(), String> {
        self.verify_harness()?;
        let session = Session::load(&self.config.state_dir)?;
        if !self.dry {
            if let Some(session) = &session {
                if session.phase != "running" {
                    return Err("Engine startup or cleanup is unfinished. Run vade status, then vade down before doctor.".into());
                }
                for rank in session.ranks.iter().filter(|r| r.attempted) {
                    let fields = self.worker(session, rank, "inspect", Fields::new())?;
                    if fields.get("alive").map(String::as_str) != Some("true") {
                        return Err(
                            "An engine rank is stopped. Run vade status, then vade down.".into(),
                        );
                    }
                }
                if !self.connection_ok(session)? {
                    return Err("The owned tunnel is down. Run vade status before engine doctor.".into());
                }
                if !self.local_health(session)? {
                    return Err(
                        "Engine health is not ready. Run vade logs, then vade status.".into(),
                    );
                }
            } else {
                self.checked(
                    Exec::new(
                        "harness doctor availability",
                        &self.config.stage_dir.join("vade-harness").to_string_lossy(),
                        vec!["doctor".into(), "--help".into()],
                    ),
                    "Harness doctor is unavailable. Run vade install to build it.",
                )?;
                crate::output::line(format_args!("Harness: available. Engine check needs vade up, then vade doctor."))?;
                return if installing {
                    Ok(())
                } else {
                    Err("Machine checks finished. Start vade up, then run vade doctor for the engine and cache check.".into())
                };
            }
        }
        let endpoint = session
            .as_ref()
            .map(Session::endpoint)
            .unwrap_or_else(|| format!("http://localhost:{}", self.config.local_port));
        let model = session
            .as_ref()
            .map(|s| s.model.clone())
            .unwrap_or_else(|| self.config.model.clone());
        let mut exec = Exec::new(
            "harness doctor",
            &self.config.stage_dir.join("vade-harness").to_string_lossy(),
            vec![
                "doctor".into(),
                endpoint,
                "--model".into(),
                model,
                "--max-seconds".into(),
                "120".into(),
            ],
        );
        // The harness budget covers both turns; leave time for setup and exit.
        exec.timeout = Duration::from_secs(150);
        clear_cloud_auth(&mut exec);
        let output = self.runner.run(&exec)?;
        if !self.dry && !output.success {
            crate::output::operation_failed();
            crate::output::line(format_args!("Run this for the harness cache diagnostics:"))?;
            exec.preview()?;
            // The error names the command itself: doctor's report keeps only
            // this message, not the lines printed above.
            return Err(format!("Harness doctor needs attention. Start with vade status; then run {} to see its cache diagnostics.", exec.shell_line()));
        }
        crate::output::line(format_args!(
            "Harness doctor: {}.",
            if self.dry {
                "planned (two small test turns)"
            } else {
                "passed"
            }
        ))?;
        Ok(())
    }

    pub(super) fn verify_harness(&mut self) -> Result<(), String> {
        let expected = if self.dry {
            None
        } else {
            text::private_dir(&self.config.stage_dir)?;
            text::reject_symlinks(&self.config.stage_dir.join("vade-harness"))?;
            Some(Manifest::load(&self.config.build_manifest)?.harness_sha256
                .ok_or("The staged harness has no saved hash. Run vade install before chat or doctor.")?)
        };
        let output = self.checked(
            Exec::new(
                "harness hash",
                "shasum",
                vec![
                    "-a".into(),
                    "256".into(),
                    self.config
                        .stage_dir
                        .join("vade-harness")
                        .to_string_lossy()
                        .into_owned(),
                ],
            ),
            "Cannot check the staged harness. Run vade install before chat or doctor.",
        )?;
        if let Some(expected) = expected {
            let hashes = output
                .stdout
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .collect::<Vec<_>>();
            if hashes != [expected.as_str()] {
                return Err("The staged harness differs from its saved hash. Run vade install before chat or doctor.".into());
            }
        }
        Ok(())
    }

}
