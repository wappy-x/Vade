use super::*;
use crate::runner::{Exec, Output, Runner, SystemRunner};
use std::{os::unix::fs::{symlink, PermissionsExt}, path::PathBuf, sync::atomic::{AtomicU64, Ordering}, time::Duration};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct CapturedSsh { program: PathBuf, calls: usize, maximum: usize }
impl Runner for CapturedSsh {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        assert_eq!(exec.program, "ssh");
        assert_eq!(&exec.args[..crate::runner::ssh_options().len()], &crate::runner::ssh_options());
        let mut local = exec.clone();
        local.program = self.program.to_string_lossy().into_owned();
        let output = SystemRunner { dry: false }.run(&local)?;
        self.calls += 1;
        self.maximum = self.maximum.max(output.stdout.len());
        Ok(output)
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { panic!("Unexpected interactive command.") }
    fn port_free(&mut self, _: u16) -> Result<(), String> { panic!("Unexpected local listener check.") }
    fn pause(&mut self, _: Duration) { panic!("Unexpected server wait.") }
}

fn fixture() -> (PathBuf, crate::config::Machine, CapturedSsh) {
    let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-doctor-model-r2-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let model = root.join("checkpoint ' files");
    fs::create_dir_all(&model).unwrap();
    super::tests::fixture(&model);
    let program = root.join("fake-ssh");
    fs::write(&program, "#!/bin/sh\nfor vade_arg; do vade_command=$vade_arg; done\nexec /bin/sh -c \"$vade_command\"\n").unwrap();
    fs::set_permissions(&program, fs::Permissions::from_mode(0o700)).unwrap();
    let machine = crate::config::Machine {
        name: "spark-a".into(), ssh: "user@spark-a".into(), verify_url: None, fabric: BTreeMap::new(), fabric_aux: BTreeMap::new(),
        model_path: model.to_string_lossy().into_owned(), engine_source: "~/source/engine".into(), cli_source: "~/source/cli".into(),
        stage_dir: "~/install".into(), state_dir: "~/state".into(),
    };
    (model, machine, CapturedSsh { program, calls: 0, maximum: 0 })
}

fn read(machine: &crate::config::Machine, runner: &mut CapturedSsh) -> Fields {
    let output = runner.run(&crate::runner::ssh(&machine.ssh, script(&machine.model_path), "doctor initial fixture")).unwrap();
    assert!(output.success);
    let mut fields = crate::text::parse(&output.stdout).unwrap();
    remote(runner, machine, &mut fields);
    fields
}

fn report(fields: &Fields) -> (Result<(), String>, String) {
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || {
        let mut report = Report::new("remote", false);
        rows(&mut report, "", "fixture", fields)?;
        report.finish()
    });
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    (result, printed)
}

#[test]
fn remote_metadata_above_eight_mib_uses_bounded_system_runner_capture() {
    let _guard = crate::runner::process_test_lock();
    let (folder, machine, mut runner) = fixture();
    // A 45 MiB index exercises the full size of current large checkpoints.
    // Whitespace keeps this test about capture, independent of node limits.
    let mut index = String::with_capacity(45 * 1024 * 1024 + 128);
    index.push('{');
    index.extend(std::iter::repeat_n(' ', 45 * 1024 * 1024));
    index.push_str("\"weight_map\":{\"a\":\"part-1.safetensors\",\"b\":\"part-2.safetensors\"}}");
    fs::write(folder.join("model.safetensors.index.json"), index).unwrap();
    let fields = read(&machine, &mut runner);
    let (result, printed) = report(&fields);
    result.unwrap();
    assert!(printed.contains("2 files, 40 bytes"), "{printed}");
    assert!(runner.calls > 2);
    assert!(runner.maximum <= CHUNK_BYTES as usize * 2);
    assert!((runner.maximum as u64) < crate::runner::MAX_CAPTURE_BYTES);
}

#[test]
fn doctor_accepts_internal_metadata_links_and_nested_shard_links() {
    let _guard = crate::runner::process_test_lock();
    let (folder, machine, mut runner) = fixture();
    let metadata = folder.join("metadata");
    let nested = folder.join("nested weights");
    fs::create_dir(&metadata).unwrap();
    fs::create_dir(&nested).unwrap();
    for (key, file) in JSON_FILES {
        fs::rename(folder.join(file), metadata.join(file)).unwrap();
        symlink(Path::new("metadata").join(file), folder.join(file)).unwrap();
        assert!(folder.join(file).is_file(), "{key}");
    }
    symlink("../part-1.safetensors", nested.join("one.safetensors")).unwrap();
    fs::rename(folder.join("part-2.safetensors"), nested.join("two.safetensors")).unwrap();
    fs::write(metadata.join("model.safetensors.index.json"), "{\"weight_map\":{\"a\":\"nested weights/one.safetensors\",\"b\":\"nested weights/two.safetensors\"}}").unwrap();
    // Launch ignores unreferenced tensor files. Doctor must do the same.
    fs::write(folder.join("unused.safetensors"), []).unwrap();
    super::tests::captured(&folder).0.unwrap();
    let fields = read(&machine, &mut runner);
    report(&fields).0.unwrap();
    let shards = inventory(&fields).unwrap();
    assert_eq!(shards.len(), 2);
    assert_eq!(shards.values().copied().sum::<u64>(), 40);
}

#[test]
fn doctor_accepts_a_root_shard_link_inside_the_checkpoint() {
    let _guard = crate::runner::process_test_lock();
    let (folder, machine, mut runner) = fixture();
    fs::rename(folder.join("part-1.safetensors"), folder.join("payload")).unwrap();
    symlink("payload", folder.join("part-1.safetensors")).unwrap();
    super::tests::captured(&folder).0.unwrap();
    report(&read(&machine, &mut runner)).0.unwrap();
}

#[test]
fn doctor_rejects_metadata_and_shard_links_outside_the_checkpoint() {
    let _guard = crate::runner::process_test_lock();
    for file in ["config.json", "tokenizer.json", "model.safetensors.index.json", "part-1.safetensors"] {
        let (folder, machine, mut runner) = fixture();
        let outside = folder.parent().unwrap().join("outside");
        fs::rename(folder.join(file), &outside).unwrap();
        symlink("../outside", folder.join(file)).unwrap();
        assert!(super::tests::captured(&folder).0.is_err(), "{file}");
        let (result, printed) = report(&read(&machine, &mut runner));
        assert!(result.is_err(), "{file}: {printed}");
    }
}

#[test]
fn doctor_rejects_index_paths_that_escape_the_checkpoint() {
    let _guard = crate::runner::process_test_lock();
    for name in ["../part-1.safetensors", "/part-1.safetensors", "nested/../../part-1.safetensors", "bad\\\\part.safetensors"] {
        let (folder, machine, mut runner) = fixture();
        let index = format!("{{\"weight_map\":{{\"a\":\"{name}\"}}}}");
        fs::write(folder.join("model.safetensors.index.json"), index).unwrap();
        assert!(super::tests::captured(&folder).0.is_err(), "{name}");
        let (result, printed) = report(&read(&machine, &mut runner));
        assert!(result.is_err(), "{name}: {printed}");
    }
}
