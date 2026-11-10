//! The proof client must bypass saved endpoint discovery and bench output flags.
use std::process::Command;

#[test]
fn proof_help_and_mock_loop_need_no_installation() {
    for option in ["--help", "--self-test"] {
        let output = Command::new(env!("CARGO_BIN_EXE_vade"))
            .args(["bench", "--first-tokens", option])
            .env("VADE_HOME", "absent-first-tokens-installation")
            .env_remove("VADE_CONFIG").output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains(if option == "--help" { "first-tokens prepare" } else { "SELFTEST PASS" }), "{text}");
    }
}
