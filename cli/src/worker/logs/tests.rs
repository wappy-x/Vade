//! Log redaction preserves public diagnostics and removes sensitive values.
use super::*;

#[test]
fn logs_hide_environment_and_terminal_controls() {
    let env = fields(&[("env.SEQPAR_CONTROL_SECRET", "private-value")]);
    assert_eq!(
        redact_log(
            "ready private-value\nSEQPAR_CONTROL_SECRET=x\nrate \u{1b}[2J\n",
            &env
        ),
        "ready [redacted]\n[private environment line omitted]\nrate [2J\n"
    );
    let warning = "WARNING: --api-key-file is group- or world-readable";
    let refusal = "FAIL --allow-remote requires --api-key-file PATH";
    assert_eq!(redact_log(&format!("{warning}\n{refusal}\n{warning} private-value"), &env), format!("{warning}\n{refusal}\n[private environment line omitted]\n"));
    for name in [
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "AWS_SESSION_TOKEN",
        "VADE_API_TOKEN",
        "OTHER_KEY",
    ] {
        let env = fields(&[(&format!("env.{name}"), "private-value")]);
        assert_eq!(
            redact_log("completion_tokens=42 private-value", &env),
            "completion_tokens=42 [redacted]\n"
        );
    }
}
