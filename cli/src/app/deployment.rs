//! Source synchronization, immutable builds, activation and updates.
use super::*;

/// Busy means a listed compute app or power draw at or above 30 W.
/// Utilization alone never blocks; a stale counter is reported, not obeyed.
pub(super) const BUILD_GPU_CHECK: &str = r#"set -eu
applications=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || utilization=''
test -z "$applications"
printf '%s\n' "$power" | awk 'NF {n++; if ($1 !~ /^[0-9]+([.][0-9]+)?$/) bad=1; else if ($1 + 0 >= 30) busy=1} END {exit (n == 0 || bad || busy)}'
printf '%s\n' "$utilization" | awk 'NF && $1 ~ /^[0-9]+$/ && $1 + 0 > max {max = $1 + 0} END {if (max > 0) printf "stale_utilization=%d\n", max}'
"#;
/// A `source = installed-bundle` config run by a vade with no bundle beside it.
pub(super) const NO_INSTALLED_BUNDLE: &str = "This config uses the installed Vade bundle, and this vade program is not in a bundle. Run the vade from the install command, or set engine_source, harness_source and cli_source in the config.";
impl<R: Runner> App<R> {
    fn source_revision(&mut self, path: &Path, label: &str) -> Result<String, String> {
        let folder = path.to_string_lossy().into_owned();
        let status=self.checked(Exec::new(&format!("{label} clean source"),"git",vec!["-C".into(),folder.clone(),"status".into(),"--porcelain".into(),"--untracked-files=normal".into()]),
            &format!("Cannot inspect {label} source. Set its local source path to a Git checkout in config."))?;
        if !self.dry && !status.stdout.trim().is_empty() {
            return Err(format!(
                "{label} source has local changes. Commit or preserve them before install/update; nothing was overwritten."
            ));
        }
        let output = self.checked(
            Exec::new(
                &format!("{label} revision"),
                "git",
                vec!["-C".into(), folder, "rev-parse".into(), "HEAD".into()],
            ),
            &format!("Cannot read {label} revision. Check its source path in config."),
        )?;
        if self.dry {
            return Ok("0".repeat(40));
        }
        let revision = output.stdout.trim();
        if !text::is_lower_hex(revision, 40) {
            return Err(format!(
                "{label} source revision is invalid. Use a Git checkout with a full revision."
            ));
        }
        Ok(revision.into())
    }

    fn sync_source(
        &mut self,
        source: &Path,
        machine: &Machine,
        destination: &str,
        label: &str,
    ) -> Result<(), String> {
        self.checked(
            runner::ssh(
                &machine.ssh,
                format!("{REMOTE_STORAGE_GUARD}\nvade_private_dir {} new\n", remote_path(destination)),
                &format!("{} source directory", machine.name),
            ),
            "Cannot create a fresh private machine source directory. Check its ownership, permissions, and symlink-free ancestors.",
        )?;
        // No deletion: source checkouts and immutable staged builds are never removed.
        let tracked = self.checked(
            Exec::new(
                &format!("{label} tracked files"),
                "git",
                vec![
                    "-C".into(),
                    source.to_string_lossy().into_owned(),
                    "ls-files".into(),
                    "-z".into(),
                    "--cached".into(),
                ],
            ),
            "Cannot list tracked source files. Check the configured Git checkout.",
        )?;
        // Enter exactly the directory prepared above. Neither old rsync nor
        // protected-argument rsync has to expand a tilde. Implied parent
        // directories use the receiver's umask. Archive mode otherwise
        // reapplies shared source permissions.
        let mut exec = runner::copy_to(
            &format!("{} {label} source copy", machine.name),
            &machine.ssh,
            destination,
            ["--from0", "--files-from=-", "--exclude=target", "--exclude=.git", "--exclude=*.local"].map(String::from).to_vec(),
            vec![format!("{}/", source.display())],
        );
        exec.stdin = tracked.stdout.into_bytes();
        exec.timeout = Duration::from_secs(600);
        self.checked(exec,"Source copy did not finish. Check SSH, disk space, and the local source path; run vade install again.")?;
        Ok(())
    }

    /// Build every staged machine: one worker thread per machine when the
    /// runner allows it, otherwise in machine order. Each worker's output is
    /// printed in machine order, and the first failing machine fails install.
    fn build_machines(&mut self, builds: &[SourceBuild], engine_sha: &str, warm_ptx: bool, prepare_model: bool) -> Result<Vec<(Artifact, bool)>, String> {
        for job in builds {
            crate::output::line(format_args!("{}: {} engine and helper.", job.machine.name, if self.dry { "plan to build" } else { "building" }))?;
        }
        let workers = if builds.len() > 1 { builds.iter().map(|_| self.runner.worker()).collect::<Option<Vec<_>>>() } else { None };
        let Some(workers) = workers else {
            return builds.iter().map(|job| self.build_machine(job, engine_sha, warm_ptx, prepare_model)).collect();
        };
        let results = std::thread::scope(|scope| {
            let handles = builds.iter().zip(workers).map(|(job, runner)| {
                let (config, dry) = (self.config.clone(), self.dry);
                scope.spawn(move || crate::output::buffered(|| App { config, runner, dry }.build_machine(job, engine_sha, warm_ptx, prepare_model)))
            }).collect::<Vec<_>>();
            handles.into_iter().map(|handle| handle.join()).collect::<Vec<_>>()
        });
        let mut built = Vec::new();
        for result in results {
            let (result, printed) = result.map_err(|_| "Install worker stopped unexpectedly; inspect retained logs.")?;
            crate::output::text(format_args!("{printed}"))?;
            built.push(result?);
        }
        Ok(built)
    }

    fn build_machine(&mut self, job: &SourceBuild, engine_sha: &str, warm_ptx: bool, prepare_model: bool) -> Result<(Artifact, bool), String> {
        let SourceBuild { machine, borrowed, storage, remote_engine, remote_cli, build } = job;
        let artifact = Artifact {
            worker: format!("{build}/vade"),
            binary: format!("{build}/flash-serve"),
            ptx: format!("{build}/flash.ptx"),
            binary_sha256: "0".repeat(64),
            ptx_sha256: "0".repeat(64),
            worker_sha256: "0".repeat(64),
        };
        let mut helper = runner::ssh(&machine.ssh, format!(r#"{storage}
export PATH="$HOME/.cargo/bin:$PATH"
unset CARGO_TARGET_DIR RUSTC_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_TARGET RUSTUP_TOOLCHAIN RUSTC RUSTDOC
export RUSTUP_TOOLCHAIN=1.99.0
cd {cli}
cli_cache=$(cd {cli_cache} && pwd -P)
ln -s "$cli_cache" target
CARGO_TARGET_DIR="$cli_cache" cargo build --release --offline --bin vade
vade_private_tree "$cli_cache"
vade_private_dir {destination} new
cp "$cli_cache/release/vade" {worker}
chmod 700 {worker}
"#, cli=remote_path(remote_cli), cli_cache=remote_path(&format!("{}/build-cache/cli",machine.cli_source)),
            destination=remote_path(build), worker=remote_path(&artifact.worker)), &format!("{} helper build", machine.name));
        helper.timeout = Duration::from_secs(900);
        self.checked(helper, "Helper build failed; check the pinned Rust toolchain and retained logs.")?;
        // Reuse compiler outputs, then copy into a fresh immutable build.
        // The target link also supports kernel scripts with relative paths.
        let script = format!(
            r#"{storage}
export PATH="$HOME/.cargo/bin:$PATH"
unset CARGO_TARGET_DIR SPARK_LAYER_PTX SPARK_CUBIN RUSTC_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_TARGET RUSTUP_TOOLCHAIN RUSTC RUSTDOC
export RUSTUP_TOOLCHAIN=1.99.0
vade_private_dir {engine} existing
engine_cache=$(cd {engine_cache} && pwd -P)
cd {engine}
printf '%s\n' {engine_revision} > REVISION
ln -s "$engine_cache" target
export CARGO_TARGET_DIR="$engine_cache"
cargo kernels
cargo build --release --offline --workspace --exclude spark-rust-kernels
vade_private_tree "$engine_cache"
destination={destination}
vade_private_dir "$destination" existing
cp "$engine_cache/release/flash-serve" "$destination/flash-serve"
cp "$engine_cache/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx" "$destination/flash.ptx"
chmod 700 "$destination/flash-serve" "$destination/vade"
chmod 600 "$destination/flash.ptx"
sha256sum "$destination/flash-serve" "$destination/flash.ptx" "$destination/vade"
"#,
            engine = remote_path(remote_engine),
            engine_revision = quote(engine_sha),
            engine_cache =
                remote_path(&format!("{}/build-cache/engine", machine.engine_source)),
            destination = remote_path(build)
        );
        let mut exec = runner::ssh(
            &machine.ssh,
            script,
            &format!("{} build and stage", machine.name),
        );
        exec.timeout = Duration::from_secs(3600);
        let model_check = self.model_check(machine, &artifact.worker);
        let (output, checked) = self.runner.parallel(&exec, &model_check)?;
        if !output.success { return Err("Machine build did not finish. Inspect retained logs and the pinned Rust/CUDA toolchain; run vade install again.".into()); }
        let mut artifact = artifact;
        if !self.dry {
            let hashes = artifact_hashes(&output.stdout)?;
            artifact.binary_sha256 = hashes[0].clone();
            artifact.ptx_sha256 = hashes[1].clone();
            artifact.worker_sha256 = hashes[2].clone();
        }
        let present = self.model_result(machine, checked)?;
        if warm_ptx && !*borrowed {
            self.prepare_install(machine, &artifact, prepare_model && present)?;
        }
        Ok((artifact, present))
    }

    /// What `vade install` will do, for `--dry-run`: where, which files, the
    /// model on each Spark and where state lives. The commands follow it.
    pub(super) fn install_plan(&self) -> String {
        let config = &self.config;
        let shown = |path: &Path| text::terminal_text(&path.to_string_lossy()).into_owned();
        let names: Vec<&str> = config.machines.iter().map(|machine| machine.name.as_str()).collect();
        let sparks = if names.len() == 1 { "Spark" } else { "Sparks" };
        let files = match &config.prebuilt_dir {
            Some(dir) if config.installed_bundle => format!("the installed Vade bundle in {}", shown(dir)),
            None if config.installed_bundle => NO_INSTALLED_BUNDLE.into(),
            Some(dir) => format!("a bundle in {} if one matches the engine and CLI revisions, else a source build on each Spark", shown(dir)),
            None => "a source build on each Spark (the config sets no prebuilt_dir)".into(),
        };
        let models = config.machines.iter().map(|machine| format!("{} {}", machine.name, text::terminal_text(&machine.model_path))).collect::<Vec<_>>().join("; ");
        let rows = [
            ("Where", format!("{} {sparks} over SSH: {}", names.len(), names.join(", "))),
            ("Files", files),
            ("Source", if config.installed_bundle { "none; the bundle holds every program".into() } else { format!("{} (committed files only)", shown(&config.engine_source)) }),
            ("Toolchain", if config.installed_bundle { "none; the bundle is prebuilt".into() } else { format!("Rust {} and {} on each Spark, for a source build", diagnostics::HOST_TOOLCHAIN, diagnostics::KERNEL_TOOLCHAIN) }),
            ("Model", models),
            ("State", format!("{} and {} here; stage_dir and state_dir on each Spark", shown(&config.state_dir), shown(&config.stage_root))),
            ("Steps", "check tools, SSH, GPUs, disks and models; copy or build; build chat here; warm the GPU startup cache; switch to the new build".into()),
            ("Next", text::command("install")),
        ];
        format!("Plan for vade install:\n{}", style::table(&rows, 2))
    }

    pub(super) fn install(&mut self) -> Result<(), String> {
        self.config.validate_model_revision()?;
        self.config.validate_control_socket()?;
        rdma_enabled()?;
        if self.dry && !crate::checks::json() {
            crate::output::text(format_args!("{}Commands it would run:\n", self.install_plan()))?;
        }
        if !self.dry {
            text::private_dir(&self.config.stage_root)?;
            text::private_dir(
                self.config
                    .activation
                    .parent()
                    .ok_or("Activation record needs a directory.")?,
            )?;
        }
        let existing = Session::load(&self.config.state_dir)?;
        let has_running_record = existing.is_some();
        // Every ownership record of this config, the selected one and each
        // other group's, also a group no longer in the config: their ranks
        // are owned engines, not outside jobs.
        let mut owned: Vec<Session> = existing.iter().cloned().collect();
        let groups = self.config.group_record_names()?;
        for dir in std::iter::once(self.config.lock_dir.clone()).chain(groups.iter().map(|name| self.config.group_state_dir(name))) {
            if dir != self.config.state_dir { owned.extend(Session::load(&dir)?); }
        }
        let install_id = if self.dry {
            "preview".into()
        } else {
            text::random_hex(12)?
        };
        let engine_source = self.config.engine_source.clone();
        let harness_source = self.config.harness_source.clone();
        let cli_source = self.config.cli_source.clone();
        // With no checkout, the bundle names its own revisions and is the only source.
        let bundled = self.config.installed_bundle;
        let [engine_sha, harness_sha, cli_sha] = if bundled {
            crate::prebuilt::revisions(self.config.prebuilt_dir.as_deref().ok_or(NO_INSTALLED_BUNDLE)?)?
        } else {
            [self.source_revision(&engine_source, "engine")?, self.source_revision(&harness_source, "harness")?, self.source_revision(&cli_source, "CLI")?]
        };
        let selected = crate::prebuilt::Bundle::find(self.config.prebuilt_dir.as_deref(), &engine_sha, &cli_sha)
            .and_then(|bundle| {
                let host = bundle.as_ref().map(|bundle| if bundled { bundle.launcher_host().map(Some) } else { bundle.host(&harness_sha) }).transpose()?.flatten();
                Ok((bundle, host))
            });
        let (mut bundle, mut host_bundle) = match selected {
            // No source to fall back on, so a bundle fault stops the install.
            Err(error) if bundled => return Err(format!("Cannot use the installed Vade bundle: {} Run the install command again.", text::terminal_text(&error))),
            Ok((None, _)) if self.config.prebuilt_dir.is_some() => {
                // A set prebuilt_dir that serves nothing is named, so a source build is never a surprise.
                let dir = self.config.prebuilt_dir.clone().unwrap_or_default();
                let shown = text::terminal_text(&dir.to_string_lossy()).into_owned();
                crate::output::line(format_args!("{}", if dir.is_dir() {
                    format!("No prebuilt bundle in {shown} matches engine {engine_sha} and CLI {cli_sha}; building from source, which needs Rust on each Spark.")
                } else {
                    format!("prebuilt_dir {shown} is not a folder; building from source, which needs Rust on each Spark.")
                }))?;
                (None, None)
            }
            Ok(bundle) => bundle,
            Err(error) => {
                crate::output::line(format_args!("Prebuilt bundle rejected: {}; building from source, which needs Rust on each Spark.", text::terminal_text(&error)))?;
                (None, None)
            }
        };
        self.local_checks_for_build(host_bundle.is_none())?;
        let previous = if bundle.is_some() {
            let mut config = self.config.clone();
            config.resolve_generation().ok().and_then(|_| {
                text::reject_symlinks(&config.build_manifest).ok()?;
                let data = fs::read_to_string(&config.build_manifest).ok()?;
                Some((Manifest::parse(&data).ok()?, text::parse(&data).ok()?))
            }).filter(|(m, _)| m.engine_sha == engine_sha && m.cli_sha == cli_sha)
        } else { None };
        let machines = self.config.machines.clone();
        let mut weights_present = true;
        for machine in &machines {
            if !self.dry {
                if runner::this_computer(&machine.ssh) { crate::output::line(format_args!("Checking {} on this computer.", machine.name))?; }
                else { crate::output::line(format_args!("Checking SSH to {}.", machine.name))?; }
            }
            weights_present &= self.machine_checks_for_build(machine, bundle.is_none())?;
        }
        if let Some(found) = &bundle {
            let floors = found.floors();
            let mut misfit = None;
            for machine in &machines {
                if let Some(cause) = self.bundle_misfit(machine, &floors)? {
                    misfit = Some(format!("{}: the bundle does not fit: {cause}", machine.name));
                    break;
                }
            }
            // No checkout here, so no source to build. Every other route builds on
            // every Spark: all ranks must run the same bytes (see the check below).
            match misfit {
                Some(misfit) if bundled => return Err(format!("{misfit}. A source build needs Rust on each Spark and a checkout on this computer. Update that Spark, or run git clone {}, set engine_source, harness_source and cli_source in the config in place of source, and follow INSTALL.md#install-rust-source-builds-only.", crate::repository::VADE_REPO_URL)),
                Some(misfit) => {
                    crate::output::line(format_args!("{misfit}; every Spark builds from source, which needs Rust on each Spark."))?;
                    (bundle, host_bundle) = (None, None);
                    self.local_checks_for_build(true)?;
                    for machine in &machines { weights_present &= self.machine_checks_for_build(machine, true)?; }
                }
                None => (),
            }
        }
        self.network_checks(&machines)?;
        // An installed bundle is one build of this CLI's engine, so it has all four.
        let supports_output =
            self.dry || bundled || source_contains(&engine_source, "FLASH_SERVE_OUTPUT_TOKENS")?;
        // Record the startup contract from the same checked source revision
        // used for every staged engine. Older manifests conservatively omit it.
        let supports_fp4_chain = self.dry || bundled
            || source_contains(&engine_source, "next_service::run_pp(rank, world,")?;
        let supports_warm_ptx = self.dry || bundled || source_contains(&engine_source, "warm_ptx_arg")?;
        let supports_prepare_model = self.dry || bundled || source_contains(&engine_source, "prepare_model_for_install")?;
        let mut manifest = Manifest {
            engine_sha: engine_sha.clone(),
            cli_sha: cli_sha.clone(),
            harness_sha256: None,
            supports_output,
            supports_fp4_chain,
            artifacts: BTreeMap::new(),
        };
        let mut builds = Vec::new();
        for machine in &machines {
            let borrowed = owned.iter().any(|session| {
                session.ranks.iter().any(|rank| {
                    rank.machine.name == machine.name && rank.machine.ssh == machine.ssh
                })
            });
            if !borrowed {
                let checked = self.checked(runner::ssh(&machine.ssh,BUILD_GPU_CHECK.into(),&format!("{} build GPU check",machine.name)),
                    &format!("{} GPU is in use or unavailable. Finish its current job with its original launcher before install; nothing was stopped.",machine.name))?;
                if let Some(v) = checked.stdout.lines().find_map(|line| line.strip_prefix("stale_utilization="))
                    .and_then(|v| v.parse::<u32>().ok()).filter(|v| (1..=100).contains(v)) {
                    crate::output::line(format_args!("{}: {}", machine.name, crate::worker::stale_line(v)))?;
                }
            }
            if let Some(bundle) = &bundle {
                let reusable = previous.as_ref()
                    .filter(|(_, fields)| {
                        fields.get(&format!("machine.{}.install_ssh", machine.name)) == Some(&machine.ssh)
                            && fields.get(&format!("machine.{}.install_stage_dir", machine.name)) == Some(&machine.stage_dir)
                    })
                    .and_then(|(m, _)| m.artifacts.get(&machine.name))
                    .filter(|a| [&a.binary_sha256, &a.ptx_sha256, &a.worker_sha256] == bundle.hashes.each_ref());
                let installed = if let Some(artifact) = reusable {
                    self.reuse_prebuilt(machine, bundle, artifact).map(|present| (artifact.clone(), present))
                } else {
                    let build = format!("{}/builds/{engine_sha}-{cli_sha}-{install_id}-prebuilt", machine.stage_dir);
                    self.install_prebuilt(machine, bundle, &build)
                };
                match installed {
                    Ok((artifact, present)) => {
                        weights_present &= present;
                        if supports_warm_ptx && !borrowed {
                            self.prepare_install(machine, &artifact, supports_prepare_model && present)?;
                        }
                        manifest.artifacts.insert(machine.name.clone(), artifact);
                        continue;
                    }
                    Err(error) if !bundled && error.starts_with("Prebuilt artifact rejected:") => {
                        crate::output::line(format_args!("{}: {}; building from source, which needs Rust there.", machine.name, text::terminal_text(&error)))?;
                        weights_present &= self.machine_checks_for_build(machine, true)?;
                    }
                    Err(error) => return Err(error),
                }
            }
            // Preserve the monorepo layout for CLI path dependencies while
            // keeping each install's source pair and published builds immutable.
            let remote_revision = format!(
                "{}/revisions/{engine_sha}-{cli_sha}-{install_id}",
                machine.engine_source
            );
            let remote_engine = format!("{remote_revision}/engine");
            let remote_cli = format!("{remote_revision}/cli");
            let storage = remote_build_storage(machine);
            self.checked(
                runner::ssh(
                    &machine.ssh,
                    format!("{storage}\nvade_private_dir {} new\n", remote_path(&remote_revision)),
                    &format!("{} build directories", machine.name),
                ),
                "Cannot prepare build folders on this machine. Use folders owned by your account, with no symlinks or access for other users, then run vade install again.",
            )?;
            if !self.dry { crate::output::line(format_args!("Copying the source to {}.", machine.name))?; }
            self.sync_source(&engine_source, machine, &remote_engine, "engine")?;
            self.sync_source(&cli_source, machine, &remote_cli, "CLI")?;
            // The remote CLI build resolves its `../harness` path dependency here.
            self.sync_source(&harness_source, machine, &format!("{remote_revision}/harness"), "harness")?;
            builds.push(SourceBuild { machine: machine.clone(), borrowed, storage,
                remote_engine, remote_cli, build: format!("{}/builds/{engine_sha}-{cli_sha}-{install_id}", machine.stage_dir) });
        }
        // Every source copy is staged; the machines now compile at once.
        for (job, (artifact, present)) in builds.iter().zip(self.build_machines(&builds, &engine_sha, supports_warm_ptx, supports_prepare_model)?) {
            weights_present &= present;
            manifest.artifacts.insert(job.machine.name.clone(), artifact);
        }
        // A machine that rejected the bundle may have rebuilt different bytes.
        // Validate the complete generation before publishing any local state.
        if !self.dry {
            if let Some((first_name, first)) = manifest.artifacts.iter().next() {
                for (name, artifact) in &manifest.artifacts {
                    if artifact.binary_sha256 != first.binary_sha256
                        || artifact.ptx_sha256 != first.ptx_sha256
                        || artifact.worker_sha256 != first.worker_sha256
                    {
                        return Err(format!(
                            "{first_name} and {name} have different artifact hashes (engine, PTX, or helper). The active generation is unchanged. Remove prebuilt_dir to rebuild every machine from source, or repair the prebuilt bundle and retry vade install."
                        ));
                    }
                }
            }
        }
        if host_bundle.is_some() {
            crate::output::line(format_args!("Local tools: copying verified prebuilt harness and CLI."))?;
        }
        let mut build = Exec::new(
            "harness build",
            &rust_tool("cargo"),
            vec![
                "build".into(),
                "--release".into(),
                "--offline".into(),
                "--manifest-path".into(),
                harness_source
                    .join("Cargo.toml")
                    .to_string_lossy()
                    .into_owned(),
                "--bin".into(),
                "vade-harness".into(),
                "--target-dir".into(),
                harness_source.join("target").to_string_lossy().into_owned(),
            ],
        );
        build.unset.extend(
            [
                "CARGO_TARGET_DIR",
                "SPARK_LAYER_PTX",
                "RUSTC_WRAPPER",
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_BUILD_TARGET",
                "RUSTUP_TOOLCHAIN",
                "RUSTC",
                "RUSTDOC",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        build.cwd = Some(harness_source.clone());
        build.timeout = Duration::from_secs(900);
        if host_bundle.is_none() {
            self.checked(build,"Harness build failed. Check its local source and Rust toolchain, then run vade install.")?;
        }
        let built = host_bundle.as_ref().map(|host| host.harness.clone())
            .unwrap_or_else(|| harness_source.join("target/release/vade-harness"));
        let mut cli_build = Exec::new(
            "controller CLI build",
            &rust_tool("cargo"),
            vec![
                "build".into(),
                "--release".into(),
                "--offline".into(),
                "--manifest-path".into(),
                cli_source.join("Cargo.toml").to_string_lossy().into_owned(),
                "--bin".into(),
                "vade".into(),
                "--target-dir".into(),
                cli_source.join("target").to_string_lossy().into_owned(),
            ],
        );
        cli_build.cwd = Some(cli_source.clone());
        cli_build.timeout = Duration::from_secs(900);
        cli_build.unset.extend(
            [
                "CARGO_TARGET_DIR",
                "RUSTC_WRAPPER",
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_BUILD_TARGET",
                "RUSTUP_TOOLCHAIN",
                "RUSTC",
                "RUSTDOC",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        if host_bundle.is_none() {
            self.checked(cli_build,"The controller CLI build failed. Check cli_source and the Rust toolchain, then run vade install.")?;
        }
        let cli_built = host_bundle.as_ref().map(|host| host.cli.clone())
            .unwrap_or_else(|| cli_source.join("target/release/vade"));
        if self.dry {
            crate::output::line(format_args!(
                "# atomically stage {} at {}; save mode-0600 SHA256 build manifest at {}",
                built.display(),
                self.config.stage_dir.join("vade-harness").display(),
                self.config.build_manifest.display()
            ))?;
            crate::output::line(format_args!(
                "# atomically stage {} at {}",
                cli_built.display(),
                self.config.stage_dir.join("vade").display()
            ))?;
        } else {
            let generation = self.config.stage_root.join("builds").join(&install_id);
            text::private_dir(&generation)?;
            stage_local(&built, &generation.join("vade-harness"))?;
            stage_local(&cli_built, &generation.join("vade"))?;
            self.config.stage_dir = generation.clone();
            self.config.build_manifest = generation.join("build.conf");
            let local_hashes = self.checked(
                Exec::new(
                    "controller binary hashes",
                    "shasum",
                    vec![
                        "-a".into(),
                        "256".into(),
                        self.config
                            .stage_dir
                            .join("vade-harness")
                            .to_string_lossy()
                            .into_owned(),
                        self.config
                            .stage_dir
                            .join("vade")
                            .to_string_lossy()
                            .into_owned(),
                    ],
                ),
                "Cannot hash the controller binaries. Install shasum, then run vade install.",
            )?;
            let hashes = local_hashes
                .stdout
                .lines()
                .filter_map(|l| l.split_whitespace().next())
                .filter(|h| text::is_lower_hex(h, 64))
                .collect::<Vec<_>>();
            if hashes.len() != 2 {
                return Err(
                    "Controller binaries did not return two SHA256 hashes. Run vade install again.".into(),
                );
            }
            if host_bundle.as_ref().is_some_and(|host| hashes != host.hashes.each_ref().map(String::as_str)) {
                return Err("Staged host artifact SHA256 mismatch; activation was not changed.".into());
            }
            let mut fields = text::parse(&manifest.encode())?;
            // A machine name alone does not identify where its bytes were installed.
            // Older manifests lack these fields and safely take the fresh-copy path.
            for machine in &machines {
                fields.insert(format!("machine.{}.install_ssh", machine.name), machine.ssh.clone());
                fields.insert(format!("machine.{}.install_stage_dir", machine.name), machine.stage_dir.clone());
            }
            fields.insert("harness_source_sha".into(), harness_sha);
            fields.insert("harness_sha256".into(), hashes[0].into());
            fields.insert("local_cli_sha256".into(), hashes[1].into());
            text::atomic_write(&self.config.build_manifest, &text::encode(&fields))?;
            Manifest::load(&self.config.build_manifest)?;
            // A failure before this single write preserves the active generation.
            text::atomic_write(
                &self.config.activation,
                &text::encode(&Fields::from([
                    ("activation_version".into(), "1".into()),
                    ("generation".into(), install_id.clone()),
                ])),
            )?;
            crate::output::line(format_args!(
                "CLI for this generation: {}",
                self.config.stage_dir.join("vade").display()
            ))?;
        }
        crate::output::line(format_args!(
            "{} {}. Models {}.",
            if self.dry {
                "Install planned for build"
            } else {
                "Installed build"
            },
            engine_sha,
            if weights_present {
                "were already present"
            } else {
                "need the verified-copy and SHA256 checks above"
            }
        ))?;
        // The build stays staged for the next run; the exit status still says the model is not ready.
        if !self.dry && !weights_present {
            return Err("The model is missing or incomplete on at least one Spark, so vade up cannot start yet. Copy it as shown above, then run vade install again.".into());
        }
        match self.harness_doctor(true) {
            Err(error) if has_running_record => {
                crate::output::line(format_args!("Build staged. Current engine doctor: {error}"))?;
                Ok(())
            }
            result => result,
        }
    }

    pub(super) fn update(&mut self) -> Result<(), String> {
        if self.config.installed_bundle {
            return Err("This setup uses the installed Vade bundle, so it has no checkout to update. For a newer Vade, run the install command again: curl -fsSL https://vade.lol/install.sh | sh".into());
        }
        let sources = [
            ("engine", self.config.engine_source.clone()),
            ("harness", self.config.harness_source.clone()),
            ("CLI", self.config.cli_source.clone()),
        ];
        // Check every checkout before the first pull so one dirty source stops the whole update.
        for (label, path) in &sources {
            self.source_revision(path, label)?;
        }
        for (label, path) in &sources {
            let folder = path.to_string_lossy().into_owned();
            let remotes = self.checked(
                Exec::new(
                    &format!("{label} remotes"),
                    "git",
                    vec!["-C".into(), folder.clone(), "remote".into()],
                ),
                "Cannot inspect source remotes. Check the configured checkout.",
            )?;
            if !self.dry && remotes.stdout.trim().is_empty() {
                crate::output::line(format_args!("{label}: local source, no remote to pull."))?;
                continue;
            }
            let mut exec = Exec::new(
                &format!("{label} update"),
                "git",
                vec!["-C".into(), folder, "pull".into(), "--ff-only".into()],
            );
            exec.timeout = Duration::from_secs(600);
            self.checked(exec,&format!("{label} could not update without a conflict. Resolve its branch or upstream, then run vade update; local changes are preserved."))?;
        }
        self.install()?;
        if let Some(session) = Session::load(&self.config.state_dir)? {
            let options = super::resident::recorded_options(&session);
            // A stop that came with the answer keeps the engine running.
            if !self.dry && self.runner.confirm_restart()? && !crate::runner::interrupted() {
                self.down()?;
                return self.up(&options);
            }
            // A group restarts under its own VADE_GROUP.
            let group = self.config.group.as_ref().map(|group| format!("VADE_GROUP={} ", group.name)).unwrap_or_default();
            crate::output::line(format_args!(
                "Running engine kept. Restart when ready:\n  {group}vade down\n  {group}vade up --hosts {}{}{} --context {} --output {}{}{}{}{}{}",
                options.hosts.join(","),
                if options.replicas { format!(" --replicas {}{}", options.hosts.len() / options.replica_sparks.max(1), if options.replica_sparks > 1 { " --pairs" } else { "" }) } else { String::new() },
                if options.mtp_explicit { format!(" --mtp {}", options.mtp) } else { String::new() },
                options.context,
                options.output,
                options.chats
                    .map(|chats| format!(" --chats {chats}"))
                    .unwrap_or_default(),
                if options.fast_prompts {
                    " --fast-prompts"
                } else {
                    ""
                },
                options.memory_bytes.map(|bytes| format!(" --memory-bytes {bytes}")).unwrap_or_default(),
                options.cache_bytes.map(|bytes| format!(" --cache-bytes {bytes}")).unwrap_or_default(),
                if options.fp4_local { " --fp4-local" } else { " --fp8" }
            ))?;
        } else {
            crate::output::line(format_args!("Start the new build: {}", self.start_command()))?;
        }
        Ok(())
    }

}

/// This bootstrap runs before a trusted remote helper exists. Fixed system tools
/// keep the checks independent of PATH. Every descent starts at a checked parent;
/// only root or this account can rename it or change its permissions thereafter.
/// A private leaf alone is insufficient when an ancestor belongs to another user.
pub(super) const REMOTE_STORAGE_GUARD: &str = r#"set -eu
umask 077
vade_uid=$(/usr/bin/id -u)
vade_storage_fail() {
  printf 'VADE_STORAGE_ERROR %s %s\n' "${1:-invalid}" "${vade_location:-${vade_path:-unknown}}"
  exit 1
}
vade_check_directory() {
  vade_location=$(/bin/pwd -P)
  if test "$1" != .; then vade_location=${vade_location%/}/${1#./}; fi
  test ! -L "$1" && test -d "$1" || vade_storage_fail
  vade_metadata=$(/usr/bin/stat -c '%u %a' -- "$1") || vade_storage_fail
  vade_owner=${vade_metadata%% *}
  vade_mode=${vade_metadata#* }
  case "$vade_owner" in ''|*[!0-9]*) vade_storage_fail ;; esac
  case "$vade_mode" in ''|*[!0-7]*) vade_storage_fail ;; esac
  test "$vade_owner" = 0 || test "$vade_owner" = "$vade_uid" || vade_storage_fail
  test $((0$vade_mode & 0022)) -eq 0 || vade_storage_fail
  if test "$2" = private; then
    test "$vade_owner" = "$vade_uid" || vade_storage_fail
    test $((0$vade_mode & 0077)) -eq 0 || vade_storage_fail private
  fi
}
vade_private_dir() (
  vade_path=$1
  vade_action=$2
  case "$vade_action" in create|existing|new) ;; *) vade_storage_fail ;; esac
  case "$vade_path" in /*) ;; *) vade_storage_fail ;; esac
  while test "${vade_path%/}" != "$vade_path"; do vade_path=${vade_path%/}; done
  test -n "$vade_path" || vade_storage_fail
  cd /
  vade_check_directory . ancestor
  vade_remaining=${vade_path#/}
  while test -n "$vade_remaining"; do
    vade_component=${vade_remaining%%/*}
    if test "$vade_component" = "$vade_remaining"; then
      vade_remaining=
    else
      vade_remaining=${vade_remaining#*/}
    fi
    case "$vade_component" in '') continue ;; .|..) vade_storage_fail ;; esac
    vade_location=$(/bin/pwd -P)
    vade_location=${vade_location%/}/$vade_component
    if test "$vade_action" = new && test -z "$vade_remaining"; then
      /bin/mkdir -m 700 -- "./$vade_component" || vade_storage_fail
    elif test ! -e "./$vade_component" && test ! -L "./$vade_component"; then
      test "$vade_action" = create || vade_storage_fail
      /bin/mkdir -m 700 -- "./$vade_component" || vade_storage_fail
    fi
    vade_check_directory "./$vade_component" ancestor
    cd -P "./$vade_component"
  done
  vade_check_directory . private
)
vade_private_tree() (
  vade_private_dir "$1" existing
  # Cargo may hard-link its own regular artifacts, but no cache entry may
  # redirect a build outside this protected tree or allow another user to write.
  vade_unsafe=$(/usr/bin/find "$1" \( \( ! -type d -a ! -type f \) -o ! -user "$vade_uid" -o -perm -0020 -o -perm -0002 \) -print -quit) || vade_storage_fail
  test -z "$vade_unsafe" || vade_storage_fail
)
"#;

fn remote_build_storage(machine: &Machine) -> String {
    let mut script = String::from(REMOTE_STORAGE_GUARD);
    for root in [&machine.engine_source, &machine.cli_source, &machine.stage_dir] {
        script.push_str(&format!("vade_private_dir {} create\n", remote_path(root)));
    }
    for directory in [
        format!("{}/revisions", machine.engine_source),
        format!("{}/build-cache", machine.engine_source),
        format!("{}/build-cache/engine", machine.engine_source),
        format!("{}/build-cache", machine.cli_source),
        format!("{}/build-cache/cli", machine.cli_source),
        format!("{}/builds", machine.stage_dir),
    ] {
        script.push_str(&format!("vade_private_dir {} create\n", remote_path(&directory)));
    }
    for cache in [
        format!("{}/build-cache/engine", machine.engine_source),
        format!("{}/build-cache/cli", machine.cli_source),
    ] {
        script.push_str(&format!("vade_private_tree {}\n", remote_path(&cache)));
    }
    script
}

fn artifact_hashes(output: &str) -> Result<[String; 3], String> {
    let hashes = output
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|value| text::is_lower_hex(value, 64))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    hashes.try_into().map_err(|_| "Build did not return all three artifact SHA256s. Run vade install again; no manifest was changed.".into())
}
fn source_contains(root: &Path, needle: &str) -> Result<bool, String> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)
            .map_err(|_| "Cannot inspect engine source. Check engine_source in config.")?
        {
            let entry = entry.map_err(|_| "Cannot read engine source directory.")?;
            let file_type = entry
                .file_type()
                .map_err(|_| "Cannot inspect engine source file.")?;
            if file_type.is_dir() {
                if !["target", ".git"].contains(&entry.file_name().to_string_lossy().as_ref()) {
                    pending.push(entry.path());
                }
            } else if file_type.is_file() && entry.file_name() == "flash-serve.rs" {
                if fs::read_to_string(entry.path())
                    .map_err(|_| "Cannot read engine source. Check permissions.")?
                    .contains(needle)
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
#[test]
fn chain_capability_marker_matches_the_real_engine_startup_contract() {
    let engine = Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine");
    assert!(source_contains(&engine, "next_service::run_pp(rank, world,").unwrap());
}

#[cfg(test)]
#[test]
fn staging_errors_name_the_program_that_failed() {
    let dir = Path::new("target").join(format!("vade-stage-local-{}", text::random_hex(8).unwrap()));
    for name in ["vade", "vade-harness"] {
        let error = stage_local(&dir.join("missing-build-output"), &dir.join("generation").join(name)).unwrap_err();
        assert_eq!(error, format!("Cannot stage {name}. Check its build output, disk space and stage_dir permissions."));
    }
    assert!(fs::read_dir(dir.join("generation")).unwrap().next().is_none(), "no temporary copy is left behind");
    fs::remove_dir_all(dir).unwrap();
}

/// Copy one controller program (the harness or the CLI) into a generation; errors name it.
fn stage_local(source: &Path, destination: &Path) -> Result<(), String> {
    let what = destination.file_name().map_or_else(|| "program".into(), |name| name.to_string_lossy().into_owned());
    let parent = destination
        .parent()
        .ok_or_else(|| format!("Staged {what} needs a parent directory. Check stage_dir in config."))?;
    text::private_dir(parent)?;
    text::reject_symlinks(destination)?;
    let temp = parent.join(format!(".stage-{}", text::random_hex(12)?));
    let result = (|| {
        fs::copy(source, &temp).map_err(
            |_| format!("Cannot stage {what}. Check its build output, disk space and stage_dir permissions."),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&temp, fs::Permissions::from_mode(0o700))
                .map_err(|_| format!("Cannot protect staged {what}."))?;
        }
        fs::File::open(&temp)
            .and_then(|f| f.sync_all())
            .map_err(|_| format!("Cannot sync staged {what}. Check disk space."))?;
        fs::rename(&temp, destination)
            .map_err(|_| format!("Cannot publish staged {what}. Check stage_dir permissions."))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

/// One machine whose engine and CLI sources are staged and ready to compile.
struct SourceBuild { machine: Machine, borrowed: bool, storage: String, remote_engine: String, remote_cli: String, build: String }
