use super::*;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

fn alive(pid: u32) -> bool {
    Command::new("/bin/kill").args(["-0", &pid.to_string()]).stderr(Stdio::null()).status().unwrap().success()
}

fn gone_soon(pid: u32) -> bool {
    let start = Instant::now();
    while alive(pid) {
        if start.elapsed() > Duration::from_secs(10) { return false; }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

#[test]
fn group_stop_argv_ends_options_and_targets_a_negative_group() {
    assert_eq!(group_kill_args(4242, 100, Some(100)), Some(["-KILL".into(), "--".into(), "-4242".into()]));
    for group in [2, 4242, i32::MAX as u32] {
        let args = group_kill_args(group, 100, Some(100)).unwrap();
        assert_eq!(args[0], "-KILL");
        assert_eq!(args[1], "--", "the group operand must follow --");
        let target: i64 = args[2].parse().unwrap();
        assert_eq!(target, -i64::from(group));
        assert!(-target > 1, "never 0 or -1: {args:?}");
    }
}

#[test]
fn group_stop_refuses_unsafe_targets() {
    for group in [0, 1, i32::MAX as u32 + 1, u32::MAX] {
        assert_eq!(group_kill_args(group, 100, Some(100)), None, "group {group}");
    }
    assert_eq!(group_kill_args(100, 100, Some(7)), None, "our own pid");
    assert_eq!(group_kill_args(7, 100, Some(7)), None, "our own group");
    assert_eq!(group_kill_args(4242, 100, None), None, "our own group is not known");
}

#[cfg(unix)]
#[test]
fn group_stop_never_targets_our_own_group() {
    let own = own_group().expect("own process group must be readable");
    let output = Command::new("ps").args(["-o", "pgid=", "-p", &std::process::id().to_string()]).output().unwrap();
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim().parse::<u32>().unwrap(), own);
    let me = std::process::id();
    assert_eq!(group_kill_args(own, me, own_group()), None);
    assert_eq!(group_kill_args(me, me, own_group()), None);
}

#[cfg(unix)]
#[test]
fn kill_group_stops_the_child_and_its_group() {
    use std::os::unix::process::CommandExt;
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "sleep 30 & echo $!; wait"]).stdin(Stdio::null()).stdout(Stdio::piped()).process_group(0);
    let mut child = command.spawn().unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    let grandchild: u32 = line.trim().parse().unwrap();
    assert!(alive(grandchild));
    kill_group(&mut child);
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(child.wait().unwrap().signal(), Some(9));
    assert!(gone_soon(grandchild), "grandchild {grandchild} must stop with its group");
}

#[cfg(unix)]
#[test]
fn kill_group_of_a_child_in_our_group_stops_only_that_child() {
    let spawn = || Command::new("sleep").arg("30").stdin(Stdio::null()).spawn().unwrap();
    let (mut target, mut other) = (spawn(), spawn());
    kill_group(&mut target);
    let _ = target.wait();
    assert!(alive(other.id()), "a process of our own group must survive");
    let _ = other.kill();
    let _ = other.wait();
}
