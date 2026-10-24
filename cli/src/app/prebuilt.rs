//! Copy a verified local bundle; only the helper precedes the parallel jobs.
use super::*;
impl<R: Runner> App<R> {
    pub(super) fn prepare_install(&mut self, machine: &Machine, artifact: &Artifact, model: bool) -> Result<(), String> {
        // Engine and helper revisions are independently selected. Ask the verified
        // staged helper through an operation older helpers already understand.
        let mut support = runner::ssh(&machine.ssh, format!("exec {} __worker", remote_path(&artifact.worker)),
            &format!("{} preparation support", machine.name));
        support.stdin = text::encode(&Fields::from([
            ("op".into(), "paths".into()), ("ptx_path".into(), artifact.ptx.clone()),
            ("model_path".into(), machine.model_path.clone()),
        ])).into_bytes();
        let response = self.checked(support, "Cannot check the staged helper's preparation support; the previous activation was kept.")?;
        if !self.dry && text::parse(&response.stdout)?.get("install_prepare").map(String::as_str) != Some("warm-ptx-v1") {
            crate::output::line(format_args!("{}: helper does not advertise install preparation; continuing ordinary installation.", machine.name))?;
            return Ok(());
        }
        let mut exec = runner::ssh(&machine.ssh, format!("exec {} __worker", remote_path(&artifact.worker)),
            &format!("{} install preparation", machine.name));
        let mut fields = Fields::from([
            ("op".into(), "warm-ptx".into()), ("state_dir".into(), machine.state_dir.clone()),
            ("engine".into(), artifact.binary.clone()), ("binary_sha256".into(), artifact.binary_sha256.clone()),
            ("ptx_path".into(), artifact.ptx.clone()), ("ptx_sha256".into(), artifact.ptx_sha256.clone()),
        ]);
        if model { fields.insert("model_path".into(), machine.model_path.clone()); }
        exec.stdin = text::encode(&fields).into_bytes();
        exec.timeout = Duration::from_secs(330);
        let output = self.runner.run(&exec)?;
        if !output.success {
            // The helper's own refusal (a busy GPU, a hash mismatch) is the cause; name it first.
            return Err(match text::parse(&output.stdout).ok().and_then(|fields| fields.get("error").cloned()) {
                Some(cause) => format!("{}: {} Install cache preparation failed; the previous activation was kept.",
                    machine.name, text::terminal_text(&cause)),
                None => "Install cache preparation failed; the previous activation was kept. Retry install after checking the engine and model files.".into(),
            });
        }
        if self.dry { return Ok(()); }
        let fields = text::parse(&output.stdout)?;
        if fields.get("ptx_cache").map(String::as_str) != Some("warm") {
            return Err("Install cache preparation returned no verified completion; the previous activation was kept.".into());
        }
        let timings = ["elapsed_ms", "ptx_warmup_ms", "model_verify_ms", "model_hashed_bytes", "model_reused_files"]
            .into_iter().filter_map(|key| fields.get(key).map(|value| format!(" {key}={}", text::terminal_text(value))))
            .collect::<String>();
        crate::output::line(format_args!("{}: install caches prepared{}.", machine.name, timings))
    }
    fn bundle_copy(&self, bundle: &crate::prebuilt::Bundle, machine: &Machine, destination: &str, files: &[&str]) -> Exec {
        let ssh = std::iter::once("ssh".into()).chain(runner::ssh_options()).map(|s| quote(&s)).collect::<Vec<_>>().join(" ");
        let mut exec = Exec::new(&format!("{} prebuilt copy", machine.name), "rsync", vec![
            "-a".into(), "--chmod=go-rwx".into(),
            format!("--rsync-path=umask 077 && cd {} && exec rsync", remote_path(destination)), "-e".into(), ssh,
        ]);
        exec.args.extend(files.iter().map(|name| bundle.dir.join(name).to_string_lossy().into_owned()));
        exec.args.push(format!("{}:./", machine.ssh));
        exec.timeout = Duration::from_secs(600);
        exec
    }
    pub(super) fn model_check(&self, machine: &Machine, worker: &str) -> Exec {
        let mut exec = runner::ssh(&machine.ssh, format!("exec {} __worker", remote_path(worker)), &format!("{} complete model check", machine.name));
        exec.stdin = text::encode(&Fields::from([
            ("op".into(), "model".into()), ("model_path".into(), machine.model_path.clone()),
            ("model_cache".into(), format!("{}/model-inventory.stamp", machine.state_dir)),
        ])).into_bytes();
        exec.timeout = Duration::from_secs(600);
        exec
    }
    pub(super) fn model_result(&mut self, machine: &Machine, checked: Output) -> Result<bool, String> {
        if self.dry { return Ok(true); }
        if !checked.success { self.model_copy_instructions(machine)?; return Ok(false); }
        crate::output::line(format_args!("{}: all model shards present{}.", machine.name,
            if checked.stdout.contains("inventory_cache=hit") { " (inventory cache hit)" } else { "" }))?;
        Ok(true)
    }
    pub(super) fn reuse_prebuilt(&mut self, machine: &Machine, bundle: &crate::prebuilt::Bundle, artifact: &Artifact) -> Result<bool, String> {
        let mut verify = String::from("set -eu\n");
        for (path, hash) in [&artifact.binary, &artifact.ptx, &artifact.worker].into_iter().zip(&bundle.hashes) {
            verify.push_str(&format!("test \"$(sha256sum {} | cut -d ' ' -f 1)\" = {}\n", remote_path(path), quote(hash)));
        }
        self.checked(runner::ssh(&machine.ssh, verify, &format!("{} existing prebuilt hashes", machine.name)), "Prebuilt artifact rejected: installed SHA256 mismatch")?;
        let checked = self.runner.run(&self.model_check(machine, &artifact.worker))?;
        crate::output::line(format_args!("{}: exact prebuilt already installed; hashes verified.", machine.name))?;
        self.model_result(machine, checked)
    }
    pub(super) fn install_prebuilt(&mut self, machine: &Machine, bundle: &crate::prebuilt::Bundle, build: &str) -> Result<(Artifact, bool), String> {
        crate::output::line(format_args!("{}: copying verified prebuilt engine and helper.", machine.name))?;
        let storage = format!("{REMOTE_STORAGE_GUARD}\nvade_private_dir {} create\nvade_private_dir {} create\nvade_private_dir {} new\n",
            remote_path(&machine.stage_dir), remote_path(&format!("{}/builds", machine.stage_dir)), remote_path(build));
        self.checked(runner::ssh(&machine.ssh, storage, &format!("{} prebuilt directory", machine.name)), "Cannot prepare prebuilt stage; check its ownership and permissions.")?;
        self.checked(self.bundle_copy(bundle, machine, build, &["vade"]), "Cannot copy prebuilt helper; check SSH and free space.")?;
        let worker = format!("{build}/vade");
        self.checked(runner::ssh(&machine.ssh, format!("set -eu\ntest \"$(sha256sum {} | cut -d ' ' -f 1)\" = {}\nchmod 700 {}\n", remote_path(&worker), quote(&bundle.hashes[2]), remote_path(&worker)),
            &format!("{} prebuilt helper hash", machine.name)), "Prebuilt artifact rejected: helper SHA256 mismatch")?;
        let copy = self.bundle_copy(bundle, machine, build, &["flash-serve", "flash.ptx"]);
        let check = self.model_check(machine, &worker);
        let (copied, checked) = self.runner.parallel(&copy, &check)?;
        if !copied.success { return Err("Prebuilt copy failed; no activation changed. Check SSH and free space.".into()); }
        let mut verify = String::from("set -eu\n");
        for (name, hash) in crate::prebuilt::FILES.iter().zip(&bundle.hashes) {
            verify.push_str(&format!("test \"$(sha256sum {} | cut -d ' ' -f 1)\" = {}\n", remote_path(&format!("{build}/{name}")), quote(hash)));
        }
        verify.push_str(&format!("chmod 700 {} {}\nchmod 600 {}\n", remote_path(&format!("{build}/flash-serve")), remote_path(&worker), remote_path(&format!("{build}/flash.ptx"))));
        self.checked(runner::ssh(&machine.ssh, verify, &format!("{} prebuilt hashes", machine.name)), "Prebuilt artifact rejected: copied SHA256 mismatch")?;
        let present = self.model_result(machine, checked)?;
        Ok((Artifact { worker, binary: format!("{build}/flash-serve"), ptx: format!("{build}/flash.ptx"),
            binary_sha256: bundle.hashes[0].clone(), ptx_sha256: bundle.hashes[1].clone(), worker_sha256: bundle.hashes[2].clone() }, present))
    }
}
