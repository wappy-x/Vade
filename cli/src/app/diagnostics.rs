//! Prerequisite checks, model guidance and harness diagnostics.
use super::*;
impl<R: Runner> App<R> {
    pub(super) fn doctor(&mut self) -> Result<(), String> {
        self.local_checks()?;
        let machines = self.config.machines.clone();
        for machine in &machines {
            self.machine_checks(machine)?;
        }
        self.network_checks(&machines)?;
        self.harness_doctor(false)
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

    pub(super) fn machine_checks(&mut self, machine: &Machine) -> Result<bool, String> {
        self.machine_checks_for_build(machine, self.config.prebuilt_dir.is_none())
    }
    pub(super) fn machine_checks_for_build(&mut self, machine: &Machine, source_build: bool) -> Result<bool, String> {
        let rdma = self.config.machines.len() > 1 && rdma_enabled()?;
        let command = format!(
            r#"set -eu
export PATH="$HOME/.cargo/bin:$PATH"
for tool in {compiler_tools} nvidia-smi sha256sum rsync curl {ping}; do
  command -v "$tool" >/dev/null 2>&1 || {{ printf 'missing=%s\n' "$tool"; exit 1; }}
done
{compiler_versions}
nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null
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
            compiler_versions = if source_build { "cargo --version >/dev/null\nrustc --version >/dev/null" } else { "" },
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
            return Ok(true);
        }
        let fields = text::parse(&output.stdout)?;
        if !output.success {
            return Err(match fields.get("missing").map(String::as_str) {
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
                _ => format!(
                    "{} failed its Rust or CUDA driver check. Check SSH and nvidia-smi there, then run vade doctor.",
                    machine.name
                ),
            });
        }
        let present = fields.get("weights").is_some_and(|v| v == "present");
        let kib = |key: &str| fields.get(key).and_then(|v| v.parse::<u64>().ok()).and_then(|v| v.checked_mul(1024));
        let memory = match (kib("mem_total_kb"), kib("mem_available_kb")) {
            (Some(total), Some(available)) => format!("; {}", crate::detect::memory_summary(total, available)),
            _ => String::new(),
        };
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
            return Err("Harness doctor needs attention. Start with vade status; then run the printed harness doctor command to see its cache diagnostics.".into());
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
