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
    /// Why SSH stopped, when `missing` is `ssh` and ssh said why.
    ssh_stop: Option<runner::SshStop>,
    /// Whether the pinned toolchains a source build needs are installed;
    /// `None` when the script did not check them.
    toolchains: Option<bool>,
    /// Compute processes on the GPU and its highest power draw in watts.
    gpu_use: Option<(usize, Option<f64>)>,
}

/// The host toolchain every source build pins, as in `rust-toolchain.toml`.
pub(crate) const HOST_TOOLCHAIN: &str = "1.99.0";
/// The nightly toolchain `cargo kernels` drives for the NVPTX build.
pub(crate) const KERNEL_TOOLCHAIN: &str = "nightly-2026-10-01";
/// The Rust install on one machine, as in docs/INSTALL.md (Install Rust). A source
/// build needs it; the bundle route does not.
pub(crate) const RUST_STEPS: &str = concat!(
    "curl --proto '=https' --tlsv1.2 -sSf -o /tmp/vade-rustup-init.sh https://sh.rustup.rs && ",
    "sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path && ",
    "export PATH=\"$HOME/.cargo/bin:$PATH\" && ",
    "rustup toolchain install 1.99.0 --profile minimal --no-self-update && ",
    "rustup toolchain install nightly-2026-10-01 --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda",
);
/// The `failed=` values of the source-build toolchain checks.
const TOOLCHAIN_FAILURES: [&str; 3] = ["rustup", "toolchain", "kernel-toolchain"];

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
        self.doctor_with(rdma_enabled())
    }
    /// [`Self::doctor`] with the SEQPAR_RDMA selection already read.
    pub(super) fn doctor_with(&mut self, transport: Result<bool, String>) -> Result<(), String> {
        let mut report = Report::new("controller", self.dry);
        crate::doctor::binary(&mut report)?;
        let mut arguments = vec!["up".into()];
        if self.config.model.ends_with("-fp8") { arguments.push("--fp8".into()); }
        let options = match crate::cli::parse(&arguments) {
            Ok(crate::cli::Cli { command: Command::Up(options), .. }) => {
                crate::doctor::generation_defaults(&mut report, options.context, options.output, options.mtp_explicit.then_some(options.mtp))?;
                Some(options)
            }
            Err(error) => { report.fail("launch defaults", &error, "Correct MTP, then run vade doctor.")?; None }
            _ => unreachable!(),
        };
        // The bundle route builds nothing, so it needs no Rust.
        let builds = !self.config.installed_bundle;
        let (result, _) = crate::output::buffered(|| self.local_checks_for_build(builds));
        stage(&mut report, self.dry, "controller tools", result, if builds { "rustc, cargo, rsync, curl and ssh found" } else { "rsync, curl and ssh found" },
            if builds { "rustc, cargo, rsync, curl, ssh -V" } else { "rsync, curl, ssh -V" },
            if builds { "Install Rust, rsync, curl and OpenSSH, then run vade doctor." } else { "Install rsync, curl and OpenSSH, then run vade doctor." })?;
        let session = match Session::load(&self.config.state_dir) {
            Ok(session) => session,
            Err(error) => {
                report.fail("ownership", &error, "Preserve the state folder; run vade status, then vade down.")?;
                return report.finish();
            }
        };
        let machines = if let Some(session) = &session {
            session.ranks.iter().map(|rank| rank.machine.clone()).collect()
        } else if let Some(options) = &options {
            match self.config.selected(options) {
                Ok(machines) => machines,
                Err(error) => { report.fail("launch topology", &error, "Correct the config, then run vade doctor.")?; Vec::new() }
            }
        } else { Vec::new() };
        let replicas = session.as_ref().is_some_and(|session| session.replicas);
        let split = session.as_ref().map_or(machines.len() > 1, Session::chained);
        let fp4_local = session.as_ref().map_or(options.as_ref().is_some_and(|options| options.fp4_local), |session| session.fp4_local);
        let cache_default = session.as_ref().filter(|session| session.fp4_local).map(Session::prefix_restore);
        let transport = if split { transport } else { Ok(false) };
        let owned = session.is_some();
        // Doctor reads the active manifest only when one exists; a fresh controller has none.
        let built = self.config.build_manifest.is_file().then(|| Manifest::load(&self.config.build_manifest));
        // A split run needs a valid SEQPAR_RDMA before its probes can run.
        // Report it once and skip what it hides.
        if let Err(error) = &transport {
            report.fail("SEQPAR_RDMA", error, "Set SEQPAR_RDMA=0 for TCP or 1 for RDMA (or unset it), then run vade doctor.")?;
        }
        for (index, machine) in machines.iter().enumerate() {
            if transport.is_err() {
                for check in ["ssh", "nvidia-smi", "GPU use", "disk", "free memory", "model files", "staged build"] { report.skip(&format!("{} {check}", machine.name), "needs a valid SEQPAR_RDMA")?; }
                continue;
            }
            let (probe, _) = crate::output::buffered(|| self.probe_machine_with_rdma(machine, self.config.prebuilt_dir.is_none(), transport.as_ref().copied().unwrap_or(false)));
            let memory = probe.as_ref().ok().and_then(|probe| probe.memory);
            let reached = probe.as_ref().is_ok_and(|probe| probe.missing.as_deref() != Some("ssh"));
            self.machine_rows(&mut report, machine, probe, owned)?;
            // The files vade up runs: the recorded rank's, else the active build's.
            let staged = match (&session, &built) { (Some(session), _) => Ok(Some(Some(session.ranks[index].artifact.clone()))),
                (None, None) => Ok(None), (None, Some(Err(error))) => Err(error.clone()),
                (None, Some(Ok(manifest))) => Ok(Some(manifest.artifacts.get(&machine.name).cloned())) };
            let row = format!("{} staged build", machine.name);
            match staged {
                _ if self.dry => report.plan(&row, "hash the engine, PTX and helper against the build manifest")?,
                _ if !reached => report.skip(&row, "needs SSH")?,
                Err(error) => report.fail(&row, &error, "Run vade install, then run vade doctor.")?,
                Ok(None) => report.skip(&row, "no build manifest; run vade install")?,
                Ok(Some(None)) => report.fail(&row, "The build manifest has no build for this machine.", "Run vade install, then run vade doctor.")?,
                Ok(Some(Some(artifact))) => {
                    let (result, _) = crate::output::buffered(|| self.verify_staged(machine, &artifact));
                    report.check(&row, result, |_| "engine, PTX and helper match the build manifest".into(), "Run vade install, then run vade doctor.")?;
                }
            }
            let mut ports = Vec::new();
            if session.as_ref().map_or(index == 0, |session| session.servers().any(|server| server == index)) {
                ports.push(crate::doctor::Port::http(session.as_ref().map_or(self.config.serve_port, |session| session.serve_port)));
            }
            let starts_path = fp4_local || session.as_ref().is_some_and(|session| session.topology() == "path");
            if split && !(starts_path && index == 0) {
                let listen = session.as_ref().map(|session| session.ranks[index].listen.clone()).or_else(|| {
                    let peer = if index == 0 { machines.get(1) } else { machines.get(index - 1) }?;
                    machine.fabric.get(&peer.name).map(|host| format!("{host}:{}", self.config.tp_port))
                });
                if let Some((host, port)) = listen.as_deref().and_then(|listen| listen.rsplit_once(':')) {
                    if let Ok(port) = port.parse::<u16>() {
                        if port != 0 {
                            ports.push(crate::doctor::Port::new(host.trim_start_matches('[').trim_end_matches(']'), port));
                            if !fp4_local {
                                if let Some(data) = port.checked_add(1) { ports.push(crate::doctor::Port::new(host.trim_start_matches('[').trim_end_matches(']'), data)); }
                            }
                        }
                    }
                }
            }
            crate::doctor::machine(&mut report, &mut self.runner, machine, &ports, self.dry,
                crate::doctor::RemoteState { owned, memory, split, cache_default })?;
        }
        match transport {
            _ if machines.is_empty() => report.skip("fabric", "needs a valid launch topology")?,
            _ if !split => report.skip("fabric", if replicas { "independent replicas; no fabric" } else { "one Spark; no fabric" })?,
            Err(_) => report.skip("fabric", "needs a valid SEQPAR_RDMA")?,
            Ok(rdma) => {
                let (result, _) = crate::output::buffered(|| match &session {
                    Some(session) => self.session_network_checks(session, rdma),
                    None => self.config.check_links(&machines).and_then(|()| self.network_checks_with_rdma(&machines, rdma)),
                });
                stage(&mut report, self.dry, "fabric", result, if rdma { "every fabric link answered ping" } else { "TCP; engine startup checks the addresses" },
                    "link addresses in the config", "Check both link addresses and connectivity, then run vade doctor.")?;
            }
        }
        let port = session.as_ref().map_or(self.config.local_port, |session| session.local_port);
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

    /// `vade doctor --fabric`: the cabling shape, every group's arc, and each
    /// Spark's ports, cables and clock, for every machine in the config.
    /// `ring` expects one closed ring of every machine at one line rate.
    pub(super) fn doctor_fabric(&mut self, ring: bool) -> Result<(), String> {
        let mut report = Report::new("controller", self.dry);
        match rdma_enabled() {
            Ok(rdma) => crate::doctor::fabric::run(&mut report, &mut self.runner, &self.config, rdma, ring, self.dry)?,
            Err(error) => report.fail("SEQPAR_RDMA", &error, "Set SEQPAR_RDMA=0 for TCP or 1 for RDMA (or unset it), then run vade doctor --fabric.")?,
        }
        report.finish()
    }

    /// The rows for one Spark's probe: SSH, nvidia-smi, disk, free memory and
    /// model files. An earlier failure skips the rows it hides.
    fn machine_rows(&self, report: &mut Report, machine: &Machine, probe: Result<Probe, String>, owned: bool) -> Result<(), String> {
        let name = |check: &str| format!("{} {check}", machine.name);
        let rerun = "then run vade doctor.";
        if self.dry {
            report.plan(&name("ssh"), &format!("ssh {} runs one read-only check script", machine.ssh))?;
            for check in ["nvidia-smi", "GPU use", "disk", "free memory", "model files"] { report.plan(&name(check), "read by that script")?; }
            return Ok(());
        }
        let probe = match probe {
            Ok(probe) => probe,
            Err(error) => {
                report.fail(&name("ssh"), &error, &format!("Run ssh {} true to see the SSH error, {rerun}", machine.ssh))?;
                for check in ["nvidia-smi", "GPU use", "disk", "free memory", "model files"] { report.skip(&name(check), "needs SSH")?; }
                return Ok(());
            }
        };
        let missing = probe.missing.as_deref();
        if missing == Some("ssh") {
            match &probe.ssh_stop {
                Some(stop) => report.fail(&name("ssh"), &stop.error, &stop.fix)?,
                None => report.fail(&name("ssh"), &probe.error.clone().unwrap_or_default(), &format!("Run ssh {} true to see the SSH error, {rerun}", machine.ssh))?,
            }
            for check in ["nvidia-smi", "GPU use", "disk", "free memory", "model files"] { report.skip(&name(check), "needs SSH")?; }
            return Ok(());
        }
        report.ok(&name("ssh"), &format!("reached {}", machine.ssh))?;
        match missing {
            Some("script") => {
                report.fail(&name("check script"), &probe.error.clone().unwrap_or_default(), &format!("Rerun vade doctor; if this repeats, check the login shell with ssh {} true.", machine.ssh))?;
                for check in ["nvidia-smi", "GPU use", "disk", "free memory", "model files"] { report.skip(&name(check), "not checked; the script stopped early")?; }
                return Ok(());
            }
            Some(tool) if tool != "disk" && !tool.starts_with("rdma") && !TOOLCHAIN_FAILURES.contains(&tool) => {
                let error = probe.error.clone().unwrap_or_default();
                if tool == "nvidia-smi" || tool == "driver" {
                    report.fail(&name("nvidia-smi"), &error, &format!("Run ssh {} nvidia-smi to inspect the driver, {rerun}", machine.ssh))?;
                } else {
                    // The script looks for cargo and rustc before nvidia-smi.
                    if matches!(tool, "cargo" | "rustc") { report.skip(&name("nvidia-smi"), &format!("not checked; the script stopped at {tool}"))?; }
                    else { report.ok(&name("nvidia-smi"), "present")?; }
                    let next = if matches!(tool, "cargo" | "rustc") { format!("Run those commands on {}, {rerun}", machine.name) } else { format!("Install {tool} on {}, {rerun}", machine.name) };
                    report.fail(&name("tools"), &error, &next)?;
                }
                let reason = if tool == "driver" { "needs a supported GPU and driver" } else { "needs the tool checks" };
                for check in ["GPU use", "disk", "free memory", "model files"] { report.skip(&name(check), reason)?; }
                return Ok(());
            }
            _ => report.ok(&name("nvidia-smi"), probe.gpu.as_deref().unwrap_or("driver answered"))?,
        }
        // The rule vade up applies: any listed compute process, or 30 W or more, is a busy GPU.
        let busy = "Stop that job with its own launcher; vade up refuses a busy GPU. Then run vade doctor.";
        match probe.gpu_use {
            _ if owned => report.skip(&name("GPU use"), "a recorded engine can hold the GPU; use vade status")?,
            Some((0, Some(watts))) if watts < crate::worker::BUSY_POWER_WATTS => report.ok(&name("GPU use"), &format!("no compute processes; {watts:.1} W"))?,
            Some((0, Some(watts))) => report.fail(&name("GPU use"), &format!("The GPU draws {watts:.1} W with no compute process listed."), busy)?,
            Some((0, None)) => report.fail(&name("GPU use"), "nvidia-smi did not report the GPU power draw, so vade up cannot judge the GPU idle.",
                &format!("Run ssh {} nvidia-smi to inspect the driver, {rerun}", machine.ssh))?,
            Some((apps, _)) => report.fail(&name("GPU use"), &format!("{apps} compute process(es) use the GPU."), busy)?,
            None => report.skip(&name("GPU use"), "nvidia-smi did not list compute processes")?,
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
        } else if probe.rdma && missing.is_none_or(|missing| TOOLCHAIN_FAILURES.contains(&missing)) {
            report.ok(&name("rdma"), "verbs device, library and memory lock limit are ready")?;
        } else if probe.rdma {
            report.skip(&name("rdma"), "needs the earlier prerequisite checks")?;
        }
        match probe.present {
            Some(true) => report.ok(&name("model files"), &format!("{} (config, tokenizer, weights; vade install checks every SHA256)", machine.model_path))?,
            Some(false) => report.fail(&name("model files"), &format!("missing or incomplete at {}", machine.model_path),
                &format!("Copy {} at {} there (docs/INSTALL.md step 3), then run vade install.", self.config.model_repo, self.config.model_revision))?,
            None => report.skip(&name("model files"), &format!("not checked; the script stopped at {}", missing.unwrap_or("an earlier check")))?,
        }
        match probe.toolchains {
            Some(true) => report.ok(&name("toolchains"), &format!("Rust {HOST_TOOLCHAIN} and {KERNEL_TOOLCHAIN} with the NVPTX target for the source build"))?,
            Some(false) => report.fail(&name("toolchains"), &probe.error.clone().unwrap_or_default(),
                &format!("Install the pinned toolchains on {} (docs/INSTALL.md), or set prebuilt_dir to a matching bundle, {rerun}", machine.name))?,
            None => {}
        }
        Ok(())
    }

    pub(super) fn local_checks_for_build(&mut self, source_build: bool) -> Result<(), String> {
        for (program, args) in [
            (rust_tool("rustc"), vec!["--version".into()]),
            (rust_tool("cargo"), vec!["--version".into()]),
            ("rsync".into(), vec!["--version".into()]),
            ("curl".into(), vec!["--version".into()]),
        ] {
            let rust = program == rust_tool("rustc") || program == rust_tool("cargo");
            if !source_build && rust { continue; }
            let message = if rust { format!("This setup builds from source, so this computer needs Rust. Run {RUST_STEPS}, then run vade doctor.") }
                else if source_build { "A controller prerequisite is unavailable. Install Rust, rsync, and curl, then run vade doctor.".to_owned() }
                else { "A controller prerequisite is unavailable. Install rsync and curl, then run vade doctor.".to_owned() };
            let label = format!("local {program}");
            // A tool that is not installed cannot start, so the runner has its own error.
            self.checked(Exec::new(&label, &program, args), &message).map_err(|error| if error.starts_with(&format!("Cannot run {label}.")) { message.clone() } else { error })?;
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
        self.probe_machine_with_rdma(machine, source_build, rdma)
    }
    fn probe_machine_with_rdma(&mut self, machine: &Machine, source_build: bool, rdma: bool) -> Result<Probe, String> {
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
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) && printf 'gpu_apps=%s\n' "$(printf '%s\n' "$apps" | awk 'NF {{n++}} END {{print n + 0}}')"
power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null) && printf 'gpu_watts=%s\n' "$(printf '%s\n' "$power" | awk '$1 ~ /^[0-9]+([.][0-9]+)?$/ {{n++; if ($1 + 0 > max) max = $1 + 0}} END {{if (n) print max}}')"
printf 'cuda=%s\n' "$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1)"
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
{toolchains}
printf 'rust=ok\ndriver=ok\n'
"#,
            // A source build pins the host toolchain, and `cargo kernels` asks rustup
            // for the nightly NVPTX toolchain unless KERNEL_BIN names one. Read the
            // installed lists only: a toolchain name never starts a download here.
            toolchains = if source_build { format!(r#"if test -z "${{KERNEL_BIN-}}"; then
  export RUSTUP_AUTO_INSTALL=0
  command -v rustup >/dev/null 2>&1 || {{ printf 'failed=rustup\n'; exit 1; }}
  rustup toolchain list 2>/dev/null | /usr/bin/grep -q '^{host}-' || {{ printf 'failed=toolchain\n'; exit 1; }}
  vade_kernel=$(rustup component list --toolchain {KERNEL_TOOLCHAIN} --installed 2>/dev/null) || {{ printf 'failed=kernel-toolchain\n'; exit 1; }}
  for vade_component in rust-src llvm-tools llvm-bitcode-linker rust-std-nvptx64-nvidia-cuda; do
    printf '%s\n' "$vade_kernel" | awk -v c="$vade_component" '$1 == c || index($1, c "-") == 1 {{f = 1}} END {{exit !f}}' || {{ printf 'failed=kernel-toolchain\n'; exit 1; }}
  done
  printf 'toolchains=ok\n'
fi"#, host = HOST_TOOLCHAIN.replace('.', "[.]")) } else { String::new() },
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
            ssh_stop: None,
            gpu_use: fields.get("gpu_apps").and_then(|apps| apps.parse().ok())
                .map(|apps| (apps, fields.get("gpu_watts").and_then(|watts| watts.parse::<f64>().ok()).filter(|watts| watts.is_finite()))),
            toolchains: (fields.get("toolchains").map(String::as_str) == Some("ok")).then_some(true)
                .or_else(|| fields.get("failed").filter(|failed| TOOLCHAIN_FAILURES.contains(&failed.as_str())).map(|_| false)),
        };
        // The same GPU and driver rules as local detection, on what the script read.
        let gpu_name = fields.get("gpu").map(|gpu| gpu.rsplit_once(',').map_or(gpu.as_str(), |(name, _)| name).trim().to_owned());
        let driver = match (gpu_name, fields.get("cuda")) {
            _ if !output.success => None,
            (Some(name), _) if !crate::detect::is_gb10(&name) => Some(format!(
                "{} reports the GPU {}; Vade needs an NVIDIA GB10 (DGX Spark). Check that the config names a Spark, then run vade doctor.",
                machine.name, text::terminal_text(&name))),
            (_, Some(cuda)) if !crate::detect::cuda_supported(cuda) => Some(format!(
                "{} has an NVIDIA driver with CUDA {}; this Vade build needs CUDA 13.0 or newer. Have the administrator update the driver, then run vade doctor.",
                machine.name, if cuda.is_empty() { "support unknown".into() } else { text::terminal_text(cuda) })),
            _ => None,
        };
        if let Some(error) = driver {
            probe.missing = Some("driver".into());
            probe.error = Some(error);
        }
        if !output.success {
            let failed = fields.get("failed").map(String::as_str);
            let missing = match (fields.get("missing"), failed) {
                (Some(tool), _) => tool.as_str(),
                (None, Some(failed)) => failed,
                // No `reached`: SSH itself (or the login shell) failed first.
                _ if !fields.contains_key("reached") => "ssh",
                _ => "script",
            };
            let missing_ssh = missing == "ssh";
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
                Some(tool @ ("cargo" | "rustc")) => format!("{} is missing {tool}, and a source build there needs Rust. Run {RUST_STEPS}, then run vade doctor.", machine.name),
                Some(tool)
                    if ["nvidia-smi", "sha256sum", "rsync", "ping", "curl"]
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
                    "rustup" => format!("{} has no rustup. A source build needs rustup with Rust {HOST_TOOLCHAIN} and {KERNEL_TOOLCHAIN} for the GPU kernels. Install them there (docs/INSTALL.md), or set prebuilt_dir to a matching bundle, then run vade doctor.", machine.name),
                    "toolchain" => format!("{} has no Rust {HOST_TOOLCHAIN} toolchain. Run rustup toolchain install {HOST_TOOLCHAIN} --profile minimal --no-self-update there, or set prebuilt_dir to a matching bundle, then run vade doctor.", machine.name),
                    "kernel-toolchain" => format!("{} lacks the {KERNEL_TOOLCHAIN} toolchain for the GPU kernels. Run rustup toolchain install {KERNEL_TOOLCHAIN} --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda there, or set prebuilt_dir to a matching bundle, then run vade doctor.", machine.name),
                    "ssh" => format!("ssh {} did not run the check script. Run ssh {} true to see the SSH error, then run vade doctor.", machine.ssh, machine.ssh),
                    _ => format!("{}'s check script stopped before naming a failed check. Rerun vade doctor; if this repeats, check the login shell with ssh {} true.", machine.name, machine.ssh),
                },
            });
            if missing_ssh {
                probe.ssh_stop = self.ssh_stop(&machine.ssh);
                if let Some(stop) = &probe.ssh_stop { probe.error = Some(format!("{} {}", stop.error, stop.fix)); }
            }
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
            "{}: the official checkpoint is missing or incomplete.\n  Model: {}\n  Revision: {}\n  Destination: {}\nFollow docs/INSTALL.md step 3: copy this verified checkpoint revision to the configured destination on each machine. Verify the source and every destination copy against a trusted full-file SHA256 manifest, including supporting files, then rerun vade install.",
            machine.name,
            self.config.model_repo,
            self.config.model_revision,
            machine.model_path
        ))?;
        Ok(())
    }

    pub(super) fn network_checks(&mut self, machines: &[Machine]) -> Result<(), String> {
        let rdma = self.config.machines.len() > 1 && rdma_enabled()?;
        self.network_checks_with_rdma(machines, rdma)
    }
    fn network_checks_with_rdma(&mut self, machines: &[Machine], rdma: bool) -> Result<(), String> {
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

    /// Saved ranks retain their actual link endpoints, not the config's fabric map.
    fn session_network_checks(&mut self, session: &Session, rdma: bool) -> Result<(), String> {
        for (index, rank) in session.ranks.iter().enumerate() {
            let closes_ring = !session.fp4_local && session.topology() == "ring";
            // Peers stay inside a --pairs replica; the next pair is not a peer.
            let peer = session.ranks.get(index + 1).filter(|_| (index + 1) % session.group_len() != 0)
                .or_else(|| closes_ring.then(|| session.ranks.first()).flatten());
            let Some(peer) = peer else { continue; };
            let (host, port) = rank.next.rsplit_once(':').ok_or("The saved fabric endpoint is invalid. Preserve the state folder, then run vade status.")?;
            if port.parse::<u16>().ok().is_none_or(|port| port == 0) || host.is_empty() {
                return Err("The saved fabric endpoint is invalid. Preserve the state folder, then run vade status.".into());
            }
            if rdma {
                self.checked(runner::ssh(&rank.machine.ssh, format!("ping -n -c 1 -W 2 {} >/dev/null", quote(host.trim_start_matches('[').trim_end_matches(']'))),
                    &format!("{} to {} fabric", rank.machine.name, peer.machine.name)),
                    &format!("Fabric from {} to {} did not reply. Check the saved link addresses, then run vade doctor.", rank.machine.name, peer.machine.name))?;
            }
        }
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
                .ok_or("The staged harness has no saved hash. Run vade install before chat, sessions or doctor.")?)
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
            "Cannot check the staged harness. Run vade install before chat, sessions or doctor.",
        )?;
        if let Some(expected) = expected {
            let hashes = output
                .stdout
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .collect::<Vec<_>>();
            if hashes != [expected.as_str()] {
                return Err("The staged harness differs from its saved hash. Run vade install before chat, sessions or doctor.".into());
            }
        }
        Ok(())
    }

}
