//! Pinned Qwen3-8B weights for the legacy local bootstrap: a supplied folder is
//! imported, the published revision is reused or the files are downloaded, and
//! every file is checked against its pinned hash before it is published.
use super::*;

impl<R: Runner> Local<R> {
    fn check_inventory(&self, checkpoint: &Path) -> Result<(), String> {
        text::reject_symlinks(checkpoint)?;
        for entry in fs::read_dir(checkpoint).map_err(|_| "Cannot inspect model inventory.")? {
            let path = entry.map_err(|_| "Cannot inspect model entry.")?.path();
            if path.extension().is_some_and(|ext| {
                matches!(ext.to_str(), Some("safetensors" | "json" | "jinja" | "txt"))
            }) && !MODEL_FILES
                .iter()
                .any(|(name, _)| path.file_name().is_some_and(|file| file == *name))
            {
                return Err("Model directory has an unpinned payload. Keep it for inspection; no server started.".into());
            }
        }
        Ok(())
    }

    fn verify_model(&mut self, checkpoint: &Path) -> Result<PathBuf, String> {
        let started = Instant::now();
        self.check_inventory(checkpoint)?;
        let mut total_bytes = 0u64;
        for &(filename, expected) in MODEL_FILES {
            let path = checkpoint.join(filename);
            self.verify_hash(&path, expected)?;
            total_bytes += fs::metadata(&path)
                .map_err(|_| "Cannot inspect verified file.")?
                .len();
            crate::output::line(format_args!("{filename}: verified SHA256 {expected}; {total_bytes} bytes verified so far"))?;
        }
        crate::output::line(format_args!(
            "Model: {total_bytes} verified bytes; download 0.000 s; verification {:.3} s; supplied directory read-only",
            started.elapsed().as_secs_f64()
        ))?;
        Ok(checkpoint.into())
    }

    pub(super) fn supplied_model(&mut self, checkpoint: &Path) -> Result<PathBuf, String> {
        // The engine derives its API model ID from the checkpoint directory name.
        if canonical_model_name(checkpoint) {
            return self.verify_model(checkpoint);
        }
        let models = self.root.join("models");
        let published = models.join(MODEL_REVISION).join("Qwen3-8B");
        if published.exists() {
            return self.acquire_model(false);
        }
        self.verify_model(checkpoint)?;
        let started = Instant::now();
        let stage = models
            .join(format!("{MODEL_REVISION}.part"))
            .join("Qwen3-8B");
        text::private_dir(&stage)?;
        self.check_inventory(&stage)?;
        for &(filename, expected) in MODEL_FILES {
            let path = stage.join(filename);
            if path.exists() {
                self.verify_hash(&path, expected)?;
                continue;
            }
            let partial = path.with_file_name(format!("{filename}.part"));
            copy(&checkpoint.join(filename), &partial)?;
            self.verify_hash(&partial, expected)?;
            fs::OpenOptions::new()
                .write(true)
                .open(&partial)
                .and_then(|f| f.sync_all())
                .map_err(|_| "Cannot sync imported weight payload. Check free space and filesystem health; retain the source weights and retry.")?;
            fs::rename(&partial, &path).map_err(|_| "Cannot publish imported weight payload. Check destination directory permissions and free space; retain the source weights.")?;
        }
        crate::output::line(format_args!(
            "Local weight import: {:.3} s; canonical Qwen3-8B directory; no downloads",
            started.elapsed().as_secs_f64()
        ))?;
        // Every required file now exists; acquisition only rehashes and publishes it.
        self.acquire_model(false)
    }

    pub(super) fn acquire_model(&mut self, allow_download: bool) -> Result<PathBuf, String> {
        let models = self.root.join("models");
        text::private_dir(&models)?;
        let final_dir = models.join(MODEL_REVISION);
        let already_published = final_dir.exists();
        let stage = if already_published {
            final_dir.clone()
        } else {
            models.join(format!("{MODEL_REVISION}.part"))
        };
        let checkpoint = stage.join("Qwen3-8B");
        text::private_dir(&checkpoint)?;
        self.check_inventory(&checkpoint)?;
        let started = Instant::now();
        let mut download_seconds = 0.0;
        let mut verify_seconds = 0.0;
        let mut total_bytes = 0u64;
        for &(filename, expected) in MODEL_FILES {
            let path = checkpoint.join(filename);
            text::reject_symlinks(&path)?;
            let verify_start = Instant::now();
            if path.exists() {
                self.verify_hash(&path, expected)?;
                verify_seconds += verify_start.elapsed().as_secs_f64();
                total_bytes += fs::metadata(&path)
                    .map_err(|_| "Cannot inspect verified file.")?
                    .len();
                crate::output::line(format_args!("{filename}: verified; {total_bytes} bytes verified so far"))?;
                continue;
            }
            if already_published {
                return Err("Published model is incomplete. Keep it for inspection and choose a fresh VADE_HOME.".into());
            }
            if !allow_download {
                return Err("Local weight import is incomplete. Retry with the original weights source; no downloads were attempted.".into());
            }
            let part = path.with_extension(format!(
                "{}.part",
                path.extension().unwrap_or_default().to_string_lossy()
            ));
            text::reject_symlinks(&part)?;
            let timer = Instant::now();
            let complete_part = part.exists() && self.verify_hash(&part, expected).is_ok();
            verify_seconds += timer.elapsed().as_secs_f64();
            if !complete_part {
                let url = format!(
                    "https://huggingface.co/Qwen/Qwen3-8B/resolve/{MODEL_REVISION}/{filename}"
                );
                let head = self.checked(curl("model size", vec!["--head".into(), url.clone()]))?;
                let size = header_size(&head.stdout).ok_or(
                    "Official server did not report model file size. Retry before downloading.",
                )?;
                let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                if have >= size && have > 0 {
                    // A complete but invalid part cannot be fixed with an append.
                    fs::rename(
                        &part,
                        part.with_extension(format!("bad-{}", text::random_hex(4)?)),
                    )
                    .map_err(|_| "Cannot quarantine invalid partial download. Check model-directory permissions; keep the partial file for inspection.")?;
                }
                let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                crate::output::line(format_args!(
                    "{filename}: {have}/{size} bytes present; {} bytes remaining",
                    size.saturating_sub(have)
                ))?;
                let mut download = curl(
                    "model download",
                    vec![
                        "--retry".into(),
                        "3".into(),
                        "--continue-at".into(),
                        "-".into(),
                        "--output".into(),
                        part.to_string_lossy().into_owned(),
                        url,
                    ],
                );
                download.timeout = Duration::from_secs(6 * 3600);
                let timer = Instant::now();
                self.checked(download)?;
                download_seconds += timer.elapsed().as_secs_f64();
            }
            let timer = Instant::now();
            if let Err(error) = self.verify_hash(&part, expected) {
                // Preserve bad bytes for review, but let the next run download a fresh part.
                fs::rename(
                    &part,
                    part.with_extension(format!("bad-{}", text::random_hex(4)?)),
                )
                .map_err(|_| "Cannot quarantine failed payload. Check model-directory permissions; keep the failed file for inspection.")?;
                return Err(error);
            }
            verify_seconds += timer.elapsed().as_secs_f64();
            fs::OpenOptions::new()
                .write(true)
                .open(&part)
                .and_then(|f| f.sync_all())
                .map_err(|_| "Cannot sync verified model payload. Check free space and filesystem health; keep downloaded files for retry.")?;
            fs::rename(&part, &path).map_err(|_| "Cannot publish verified model file. Check model-directory permissions and free space; keep downloaded files for retry.")?;
            total_bytes += fs::metadata(&path)
                .map_err(|_| "Cannot inspect verified file.")?
                .len();
            crate::output::line(format_args!("{filename}: verified; {total_bytes} bytes verified so far"))?;
        }
        // Save provenance after every required byte has passed. Reuse still rehashes all files.
        text::atomic_write(
            &checkpoint.join("verified.conf"),
            &text::encode(&Fields::from([
                ("repository".into(), "Qwen/Qwen3-8B".into()),
                ("revision".into(), MODEL_REVISION.into()),
                ("bytes".into(), total_bytes.to_string()),
            ])),
        )?;
        if !already_published {
            fs::rename(&stage, &final_dir)
                .map_err(|_| "Cannot publish verified model revision. Check model-directory permissions and free space; keep the verified files.")?;
            sync_dir(&models)?;
        }
        crate::output::line(format_args!(
            "Model: {total_bytes} verified bytes; download {download_seconds:.3} s; verification {verify_seconds:.3} s; total {:.3} s",
            started.elapsed().as_secs_f64()
        ))?;
        Ok(final_dir.join("Qwen3-8B"))
    }
}
