//! Pinned source builds, immutable release publication and artifact validation.
use super::*;

impl<R: Runner> Local<R> {
    pub(super) fn acquire_engine(&mut self, setup: &mut Fields) -> Result<Fields, String> {
        let generations = self.root.join("generations");
        text::private_dir(&generations)?;
        let id = if let Some(id) = setup.get("generation") {
            id.clone()
        } else {
            let id = text::random_hex(12)?;
            setup.insert("generation".into(), id.clone());
            self.save_setup(setup)?;
            id
        };
        let target = generations.join(&id);
        if target.exists() {
            let generation = self.generation(&id)?;
            self.activate(&id)?;
            return Ok(generation);
        }
        let stage = generations.join(format!("{id}.part"));
        text::private_dir(&stage)?;
        let mut manifest = Fields::from([
            ("version".into(), "1".into()),
            ("generation".into(), id.clone()),
        ]);
        let release = setup.get("release").filter(|s| !s.is_empty()).cloned();
        if let Some(release) = release {
            let release = PathBuf::from(release);
            let supplied = read_fields(&release.join("release.conf"))?
                .ok_or("Release artifact needs release.conf.")?;
            if required(&supplied, "version")? != "1"
                || required(&supplied, "target")? != "linux-aarch64-gb10"
            {
                return Err("Release has the wrong format or target.".into());
            }
            for name in ["engine", "harness", "ptx"] {
                let filename = match name {
                    "engine" => "spark-serve",
                    "harness" => "vade-harness",
                    _ => "kernels.ptx",
                };
                let expected = required(&supplied, &format!("{name}_sha256"))?;
                self.verify_hash(&release.join(filename), expected)?;
                copy(&release.join(filename), &stage.join(filename))?;
                self.verify_hash(&stage.join(filename), expected)?;
                manifest.insert(format!("{name}_sha256"), expected.into());
            }
            for name in ["engine_revision", "harness_revision"] {
                let revision = required(&supplied, name)?;
                if !text::is_lower_hex(revision, 40) {
                    return Err("Release source revision must be a full Git SHA.".into());
                }
                manifest.insert(name.into(), revision.into());
            }
        } else {
            for (label, repository, binary) in [
                (
                    "engine",
                    crate::repository::VADE_REPO_URL,
                    "spark-serve",
                ),
                (
                    "harness",
                    crate::repository::VADE_REPO_URL,
                    "vade-harness",
                ),
            ] {
                let source = self.source(setup, label, repository)?;
                let mut build = Exec::new(
                    &format!("{label} build"),
                    &crate::app::rust_tool("cargo"),
                    if label == "engine" {
                        vec!["kernels".into()]
                    } else {
                        vec![
                            "build".into(),
                            "--release".into(),
                            "--offline".into(),
                            "--bin".into(),
                            binary.into(),
                        ]
                    },
                );
                build.cwd = Some(source.clone());
                build.timeout = Duration::from_secs(1800);
                clean_build_environment(&mut build);
                let built = Instant::now();
                self.checked(build.clone())?;
                crate::output::line(format_args!(
                    "{label} {}: {:.3} s",
                    if label == "engine" {
                        "kernels build"
                    } else {
                        "build"
                    },
                    built.elapsed().as_secs_f64()
                ))?;
                if label == "engine" {
                    build.args = vec![
                        "build".into(),
                        "--release".into(),
                        "--offline".into(),
                        "-p".into(),
                        "spark-serve".into(),
                    ];
                    let built = Instant::now();
                    self.checked(build)?;
                    crate::output::line(format_args!("Engine host build: {:.3} s", built.elapsed().as_secs_f64()))?;
                    let ptx =
                        source.join("target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx");
                    copy(&ptx, &stage.join("kernels.ptx"))?;
                    manifest.insert("ptx_sha256".into(), self.hash(&stage.join("kernels.ptx"))?);
                }
                copy(
                    &source.join("target/release").join(binary),
                    &stage.join(binary),
                )?;
                manifest.insert(format!("{label}_sha256"), self.hash(&stage.join(binary))?);
                manifest.insert(
                    format!("{label}_revision"),
                    setup[&format!("{label}_revision")].clone(),
                );
            }
        }
        text::atomic_write(&stage.join("build.conf"), &text::encode(&manifest))?;
        fs::rename(&stage, &target).map_err(|_| "Cannot publish engine generation.")?;
        sync_dir(&generations)?;
        let generation = self.generation(&id)?;
        self.activate(&id)?;
        Ok(generation)
    }

    pub(super) fn source(
        &mut self,
        setup: &mut Fields,
        label: &str,
        repository: &str,
    ) -> Result<PathBuf, String> {
        let sources = self.root.join("sources");
        text::private_dir(&sources)?;
        let supplied = setup
            .get(&format!("{label}_source"))
            .filter(|s| !s.is_empty())
            .cloned();
        let checkout = supplied
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| sources.join(format!("{label}-checkout")));
        if !checkout.exists() {
            if supplied.is_some() {
                return Err(format!(
                    "Saved local {label} source is missing: {}. Restore that checkout, or explicitly choose a fresh VADE_HOME without --{label}-source for a network fetch. Nothing fetched.",
                    checkout.display()
                ));
            }
            self.checked(Exec::new(
                &format!("fetch {label}"),
                "git",
                vec![
                    "clone".into(),
                    "--depth".into(),
                    "1".into(),
                    repository.into(),
                    checkout.to_string_lossy().into_owned(),
                ],
            ))?;
        }
        text::reject_symlinks(&checkout)?;
        let component = checkout.join(label);
        let source = if component.join("Cargo.toml").is_file() {
            component
        } else {
            checkout
        };
        text::reject_symlinks(&source)?;
        let folder = source.to_string_lossy().into_owned();
        let head = self
            .checked(Exec::new(
                &format!("{label} revision"),
                "git",
                vec![
                    "-C".into(),
                    folder.clone(),
                    "rev-parse".into(),
                    "HEAD".into(),
                ],
            ))?
            .stdout
            .trim()
            .to_owned();
        if !text::is_lower_hex(&head, 40) {
            return Err("Source needs a full commit SHA.".into());
        }
        let key = format!("{label}_revision");
        if let Some(revision) = setup.get(&key)
            && revision != &head {
            return Err("Source revision changed during setup. Restore its pinned revision or select a fresh VADE_HOME.".into());
        }
        let clean = self.checked(Exec::new(
            &format!("{label} clean source"),
            "git",
            vec![
                "-C".into(),
                folder.clone(),
                "status".into(),
                "--porcelain".into(),
            ],
        ))?;
        if !clean.stdout.trim().is_empty() {
            return Err(
                "Source has local changes. Use a clean checkout; no changes were discarded.".into(),
            );
        }
        setup.insert(key, head.clone());
        self.save_setup(setup)?;
        // Build a committed snapshot, without writing in a supplied checkout.
        let snapshot = sources.join(format!("{label}-{head}"));
        if !snapshot.exists() {
            let layout = self.checked(Exec::new(
                &format!("{label} source layout"),
                "git",
                vec![
                    "-C".into(),
                    folder,
                    "rev-parse".into(),
                    "--show-toplevel".into(),
                    "--show-prefix".into(),
                ],
            ))?;
            let mut layout = layout.stdout.lines();
            let repository_root = layout.next().ok_or("Cannot locate source repository root.")?;
            if !Path::new(repository_root).is_absolute() {
                return Err("Source repository root must be absolute.".into());
            }
            let prefix = layout.next().unwrap_or("").trim_end_matches('/');
            let tree = if prefix.is_empty() {
                head.clone()
            } else {
                format!("{head}:{prefix}")
            };
            let archive = sources.join(format!("{label}-{head}.tar"));
            self.checked(Exec::new(
                &format!("archive {label}"),
                "git",
                vec![
                    "-C".into(),
                    repository_root.into(),
                    "archive".into(),
                    "--format=tar".into(),
                    format!("--output={}", archive.display()),
                    tree,
                ],
            ))?;
            let partial = snapshot.with_extension("part");
            if partial.exists() {
                text::reject_symlinks(&partial)?;
                fs::remove_dir_all(&partial)
                    .map_err(|_| "Cannot clear interrupted source extraction.")?;
            }
            text::private_dir(&partial)?;
            self.checked(Exec::new(
                &format!("extract {label}"),
                "tar",
                vec![
                    "-xf".into(),
                    archive.to_string_lossy().into_owned(),
                    "-C".into(),
                    partial.to_string_lossy().into_owned(),
                ],
            ))?;
            if label == "engine" {
                text::atomic_write(&partial.join("REVISION"), &format!("{head}\n"))?;
            }
            fs::rename(&partial, &snapshot).map_err(|_| "Cannot publish source snapshot.")?;
            fs::remove_file(archive).map_err(|_| "Cannot remove source archive.")?;
        }
        Ok(snapshot)
    }

    pub(super) fn generation(&mut self, id: &str) -> Result<Fields, String> {
        if !text::is_lower_hex(id, 24) {
            return Err("Invalid generation identity. Keep the record.".into());
        }
        let directory = self.root.join("generations").join(id);
        let mut manifest =
            read_fields(&directory.join("build.conf"))?.ok_or("Generation manifest is missing.")?;
        if required(&manifest, "generation")? != id || required(&manifest, "version")? != "1" {
            return Err("Generation manifest identity differs.".into());
        }
        for (name, file) in [
            ("engine", "spark-serve"),
            ("harness", "vade-harness"),
            ("ptx", "kernels.ptx"),
        ] {
            let path = directory.join(file);
            self.verify_hash(&path, required(&manifest, &format!("{name}_sha256"))?)?;
            if name != "ptx"
                && fs::metadata(&path)
                    .map_err(|_| "Cannot inspect executable.")?
                    .permissions()
                    .mode()
                    & 0o111
                    == 0
            {
                return Err("Engine/harness artifact is not executable. Fix the supplied artifact before acquisition.".into());
            }
            manifest.insert(name.into(), path.to_string_lossy().into_owned());
        }
        Ok(manifest)
    }

}
