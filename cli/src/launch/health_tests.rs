use super::*;
use crate::runner::{Output, SystemRunner};

struct LocalCurl {
    home: PathBuf,
    payload: PathBuf,
    probe: Option<Exec>,
}

impl LocalCurl {
    fn local_exec(&self, exec: &Exec) -> Exec {
        let mut local = exec.clone();
        // Exercise real curl's default-config handling without opening a socket.
        *local.args.last_mut().unwrap() = format!("file://{}", self.payload.display());
        for name in ["HOME", "CURL_HOME", "XDG_CONFIG_HOME"] {
            local.unset.retain(|key| key != name);
            local
                .env
                .insert(name.into(), self.home.to_string_lossy().into_owned());
        }
        local
    }
}

impl Runner for LocalCurl {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        assert_eq!(exec.program, "curl");
        assert_eq!(exec.label, "local health");
        assert_eq!(exec.args.last().unwrap(), "http://127.0.0.1:18080/health");
        self.probe = Some(exec.clone());
        SystemRunner { dry: false }.run(&self.local_exec(exec))
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> {
        unreachable!()
    }
    fn port_free(&mut self, _: u16) -> Result<(), String> {
        unreachable!()
    }
    fn pause(&mut self, _: Duration) {
        unreachable!()
    }
}

#[test]
fn health_probe_ignores_default_curl_configuration() {
    let _guard = crate::runner::process_test_lock();
    let base = fs::canonicalize(std::env::temp_dir())
        .unwrap()
        .join(format!("vade-health-curl-{}", text::random_hex(8).unwrap()));
    struct Remove(PathBuf);
    impl Drop for Remove {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _remove = Remove(base.clone());
    let home = base.join("home");
    let root = base.join("owner");
    fs::create_dir_all(&home).unwrap();
    text::private_dir(&root).unwrap();
    fs::write(home.join(".curlrc"), "write-out = \"\\n%{http_code}\\n\"\n").unwrap();
    let payload = home.join("health.json");
    let ready = r#"{"status":"ready","model":"fixture-model"}"#;
    fs::write(&payload, ready).unwrap();
    text::atomic_write(
        &root.join("config"),
        &text::encode(&Fields::from([
            ("format".into(), "vade-owner-v1".into()),
            (
                "model_path".into(),
                home.join("model").to_string_lossy().into_owned(),
            ),
            ("model_bytes".into(), "32".into()),
            ("context_tokens".into(), "8192".into()),
            ("output_tokens".into(), "8192".into()),
            ("port".into(), "18080".into()),
            (
                "stage_dir".into(),
                root.join("stage").to_string_lossy().into_owned(),
            ),
            (
                "state_dir".into(),
                root.join("state").to_string_lossy().into_owned(),
            ),
        ])),
    )
    .unwrap();
    let config = Config::load(&root).unwrap().unwrap();
    let mut launcher = Launcher {
        root,
        home: home.clone(),
        runner: LocalCurl {
            home,
            payload,
            probe: None,
        },
        detector: Native,
    };
    assert_eq!(
        launcher.health(&config).unwrap().as_deref(),
        Some("fixture-model")
    );
    let probe = launcher.runner.probe.as_ref().unwrap();
    assert_eq!(probe.args.first().map(String::as_str), Some("-q"));

    // Prove the fixture config is active without curl's first-argument guard.
    let mut unguarded = launcher.runner.local_exec(probe);
    unguarded.args.remove(0);
    let output = SystemRunner { dry: false }.run(&unguarded).unwrap();
    assert!(output.success);
    assert_eq!(output.stdout, format!("{ready}\n000\n"));
    assert!(crate::json::string(&output.stdout, "status").is_none());
}
