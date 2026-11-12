//! Stop a child's process group, and never any other process.
//!
//! The standard library signals one process only, and this crate adds no
//! unsafe code, so a group stop runs the system tool as
//! `/bin/kill -KILL -- -<pgid>`. The `--` is required. POSIX `kill` reads a
//! negative operand after `--` as a process group id. macOS `/bin/kill` (BSD)
//! and Linux `kill` (procps-ng, util-linux) then send SIGKILL to each process
//! in that group. When no such group exists, the tool signals nothing and
//! exits with a nonzero status. Without `--`, procps-ng 4.0.4 on Linux stopped
//! every process of the user, not the group. A target of `-1` means every
//! process the user may signal, and `0` means our own group, so those targets
//! are never passed.

use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;

/// Stop a child that leads its own process group (spawned with
/// `process_group(0)`), then the rest of its group. The direct child is
/// stopped first, without the system tool. The group stop then runs only
/// when [`group_kill_args`] accepts the target. A child that does not lead a
/// group has no group with its pid, so the group stop then finds nothing.
pub(crate) fn kill_group(child: &mut Child) {
    let _ = child.kill();
    #[cfg(unix)]
    if let Some(args) = group_kill_args(child.id(), std::process::id(), own_group()) {
        let _ = Command::new("/bin/kill").args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
}

/// The `/bin/kill` arguments that stop process group `group`, or None when
/// the target is not safe. Refused: 0 and 1 (the `0` and `-1` targets), a
/// value too large for a pid, our own pid, our own group, and any target
/// when our own group is not known.
pub(crate) fn group_kill_args(group: u32, own_pid: u32, own_group: Option<u32>) -> Option<[String; 3]> {
    let own_group = own_group?;
    if group <= 1 || group > i32::MAX as u32 || group == own_pid || group == own_group {
        return None;
    }
    Some(["-KILL".into(), "--".into(), format!("-{group}")])
}

/// Our own process group id, read once. None when it cannot be read.
pub(crate) fn own_group() -> Option<u32> {
    static GROUP: OnceLock<Option<u32>> = OnceLock::new();
    *GROUP.get_or_init(read_own_group)
}

/// Linux: field 5 of `/proc/self/stat`, after the `)` that ends the name.
#[cfg(target_os = "linux")]
fn read_own_group() -> Option<u32> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    stat.rsplit_once(')')?.1.split_whitespace().nth(2)?.parse().ok()
}

/// Other systems: `ps -o pgid= -p <our pid>`.
#[cfg(not(target_os = "linux"))]
fn read_own_group() -> Option<u32> {
    let output = Command::new("/bin/ps").args(["-o", "pgid=", "-p", &std::process::id().to_string()])
        .stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    if !output.status.success() { return None; }
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests;
