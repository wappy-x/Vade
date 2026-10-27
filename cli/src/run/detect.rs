//! Host prerequisite checks for the legacy single-Spark bootstrap.
use super::*;

impl<R: Runner> Local<R> {
    pub(super) fn detect(
        &mut self,
        owned: bool,
        artifact: bool,
        supplied_weights: bool,
    ) -> Result<(), String> {
        let script = r#"set -u
printf 'os=%s\narch=%s\n' "$(uname -s)" "$(uname -m)"
missing=''
for tool in curl sha256sum nvidia-smi sh; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done
printf 'missing=%s\n' "$missing"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader,nounits) || exit 1
  printf 'gpu=%s\n' "$gpu"
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 1
  power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits) || exit 1
  utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || utilization=''
  if [ -n "$apps" ] || ! printf '%s\n' "$power" | awk 'NF {n++; if ($1 !~ /^[0-9]+([.][0-9]+)?$/) bad=1; else if ($1 + 0 >= 30) busy=1} END {exit (n == 0 || bad || busy)}'; then
    printf 'busy=yes\n'
  else
    printf 'busy=no\n'
    printf '%s\n' "$utilization" | awk 'NF && $1 ~ /^[0-9]+$/ && $1 + 0 > max {max = $1 + 0} END {if (max > 0) printf "stale_utilization=%d\n", max}'
  fi
fi
awk '/^MemAvailable:/ {printf "ram_bytes=%.0f\n", $2 * 1024}' /proc/meminfo 2>/dev/null
probe=$1
while [ ! -d "$probe" ]; do probe=$(dirname "$probe"); done
df -Pk "$probe" | awk 'NR==2 {printf "disk_bytes=%.0f\n", $4 * 1024}'
"#;
        let output = self.checked(Exec::new(
            "Spark prerequisites",
            "sh",
            vec![
                "-c".into(),
                script.into(),
                "vade-detect".into(),
                self.root.to_string_lossy().into_owned(),
            ],
        ))?;
        if self.dry {
            return Ok(());
        }
        let fields = text::parse(&output.stdout)?;
        let mut missing = Vec::new();
        if fields.get("os").map(String::as_str) != Some("Linux") {
            missing.push("Linux".into());
        }
        if fields.get("arch").map(String::as_str) != Some("aarch64") {
            missing.push("aarch64".into());
        }
        if let Some(tools) = fields.get("missing").filter(|s| !s.trim().is_empty()) {
            missing.push(format!("tools:{tools}"));
        }
        let gpu = fields.get("gpu").map(String::as_str).unwrap_or("");
        let parts = gpu.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() != 3
            || !parts[0].contains("GB10")
            || !parts[1].bytes().all(|b| b.is_ascii_digit() || b == b'.')
            || parts[1].is_empty()
        {
            missing.push("one NVIDIA GB10 with a working driver".into());
        }
        let ram = fields
            .get("ram_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let gpu_free = parts
            .get(2)
            .and_then(|s| s.parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024));
        // GB10 uses unified RAM; nvidia-smi may return N/A for memory.free.
        if parts.get(2).is_some_and(|s| {
            gpu_free.is_none() && !matches!(*s, "N/A" | "[N/A]" | "[Not Supported]")
        }) {
            missing.push("readable GPU free memory".into());
        }
        if !owned {
            if fields.get("busy").map(String::as_str) != Some("no") {
                missing.push("idle GPU (another compute job is active)".into());
            } else if let Some(v) = fields.get("stale_utilization").and_then(|v| v.parse::<u32>().ok()).filter(|v| (1..=100).contains(v)) {
                crate::output::line(format_args!("{}", crate::worker::stale_line(v)))?;
            }
            if ram < 24 * 1024 * 1024 * 1024
                || gpu_free.is_some_and(|b| b < 24 * 1024 * 1024 * 1024)
            {
                missing.push("24 GiB available unified/GPU memory".into());
            }
        }
        let disk = fields
            .get("disk_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let model_present = self.root.join("models").join(MODEL_REVISION).is_dir();
        let disk_need = if model_present || supplied_weights {
            1024 * 1024 * 1024
        } else {
            25 * 1024 * 1024 * 1024
        };
        if disk < disk_need {
            missing.push(format!("{disk_need} free disk bytes"));
        }
        if !missing.is_empty() {
            return Err(format!(
                "Missing prerequisites: {}. Nothing downloaded.",
                missing.join(", ")
            ));
        }
        if !artifact {
            let tools = self.runner.run(&Exec::new("source build prerequisites", "sh", vec!["-c".into(), r#"PATH="$HOME/.cargo/bin:$PATH"; export PATH; missing=''; for tool in git cc tar rustup cargo; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done; printf 'missing=%s\n' "$missing"; rustup run 1.99.0 rustc --version >/dev/null 2>&1 && echo stable=ok; rustup run nightly-2026-10-01 rustc --version >/dev/null 2>&1 && echo nightly=ok; rustup component list --toolchain nightly-2026-10-01 --installed 2>/dev/null"#.into()]))?;
            let tools = tools.stdout;
            let first = tools.lines().next().unwrap_or("");
            if first != "missing=" {
                missing.push(format!("build tools ({first})"));
            }
            for (needle, description) in [
                ("stable=ok", "Rust 1.99.0"),
                ("nightly=ok", "nightly-2026-10-01"),
                ("rust-src", "nightly rust-src"),
                ("llvm-tools", "nightly llvm-tools"),
                ("llvm-bitcode-linker", "nightly llvm-bitcode-linker"),
            ] {
                if !tools.contains(needle) {
                    missing.push(description.into());
                }
            }
            let target = self.runner.run(&Exec::new(
                "NVPTX prerequisite",
                &crate::app::rust_tool("rustup"),
                vec![
                    "target".into(),
                    "list".into(),
                    "--toolchain".into(),
                    "nightly-2026-10-01".into(),
                    "--installed".into(),
                ],
            ))?;
            if !target.success || !target.stdout.lines().any(|s| s == "nvptx64-nvidia-cuda") {
                missing.push("nightly nvptx64-nvidia-cuda target".into());
            }
        }
        crate::output::line(format_args!(
            "Detected: {}; available RAM {ram} bytes; GPU free {} bytes; disk {disk} bytes",
            text::terminal_text(gpu), gpu_free.unwrap_or(ram)
        ))?;
        if !missing.is_empty() {
            return Err(format!(
                "Missing prerequisites: {}. Nothing downloaded. Install prerequisites through your usual user/admin process, then retry.",
                missing.join(", ")
            ));
        }
        let executable = std::env::current_exe().map_err(|_| "Cannot locate vade. Run the installed binary from an existing directory.")?;
        let mut platform = Exec::new(
            "safe process prerequisites",
            &executable.to_string_lossy(),
            vec!["__worker".into()],
        );
        platform.stdin = b"op=platform\n".to_vec();
        let output = self.checked(platform)?;
        if text::parse(&output.stdout)?
            .get("pidfd")
            .map(String::as_str)
            != Some("ok")
        {
            return Err("Safe Linux process handles are unavailable. Nothing downloaded. Check the supported Spark prerequisites in INSTALL.md.".into());
        }
        Ok(())
    }
}
