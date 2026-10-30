//! Actual CLI chain previews; no SSH, GPU or ownership writes.
use std::{fs, path::PathBuf, process::Command};
struct Fixture(PathBuf);
impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap()
            .join(format!("vade-chain-{}-{label}", std::process::id()));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command.env("VADE_HOME", self.0.join("home"))
            .env("VADE_CONFIG", self.0.join("config"))
            .env_remove("VADE_API_KEY_FILE").env_remove("VADE_API_KEY");
        command
    }
}
impl Drop for Fixture { fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); } }
fn safe(bytes: &[u8]) -> &str { std::str::from_utf8(bytes).unwrap() }

#[test]
fn fp4_chain_preview_covers_size_order_transport_and_local_refusals() {
    let f = Fixture::new("chain-preview");
    let config = f.0.join("config");
    let mut source = "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nstate_dir = state\nmodel = qwen3.8-flash-next-fp4-local\n".to_owned();
    for index in 0..24 {
        source.push_str(&format!("[machine n{index}]\nssh = fixture-n{index}-no-network\nmodel_path = /models/fixture\n"));
        if index > 0 { source.push_str(&format!("fabric.n{} = 10.20.{}.2\n", index - 1, index - 1)); }
        if index < 23 { source.push_str(&format!("fabric.n{} = 10.20.{index}.1\n", index + 1)); }
    }
    fs::write(&config, &source).unwrap();
    for count in [3, 4, 6, 24] {
        for rdma in ["0", "1"] {
            let output = f.command().args(["up", "--fp4-local", "--sparks", &count.to_string(), "--dry-run"])
                .env("VADE_CONFIG", &config).env("SEQPAR_RDMA", rdma).env_remove("MTP").output().unwrap();
            assert!(output.status.success(), "{}", safe(&output.stderr));
            let stdout = safe(&output.stdout);
            assert!(stdout.contains(&format!("on {count} Sparks")));
            assert_eq!(stdout.matches(&format!("# arg.1={count}\n")).count(), count);
            assert!(!stdout.contains("env.VADE_W2_PP="), "preview must keep private environment values hidden");
            assert!(stdout.contains("Run the same command without --dry-run"));
            assert!(!f.0.join("state").exists() && !f.0.join("home").exists());
        }
    }
    let output = f.command().args(["up", "--fp4-local", "--hosts", "n3,n2,n1,n0", "--dry-run"])
        .env("VADE_CONFIG", &config).output().unwrap();
    assert!(output.status.success(), "{}", safe(&output.stderr));
    let stdout = safe(&output.stdout);
    let positions = [3, 2, 1, 0].map(|rank| stdout.find(&format!("fixture-n{rank}-no-network")).unwrap());
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]), "host order changed in command preview");
    for args in [vec!["--sparks", "25"], vec!["--hosts", "n0,n2,n3"], vec!["--hosts", "n0,n1,n0"]] {
        let output = f.command().args(["up", "--fp4-local", "--dry-run"]).args(args)
            .env("VADE_CONFIG", &config).output().unwrap();
        assert!(!output.status.success());
        assert!(!safe(&output.stdout).contains("$ "), "invalid selection reached command preview");
        assert!(!f.0.join("state").exists());
    }
}
