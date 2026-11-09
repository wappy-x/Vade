//! Standalone Mac diagnostic for the session owner's follow-up, not a Cargo
//! test or an external-agent launch. Compile locally with rustc, then execute.
//! Only this executable is spawned. It opens/removes its own adjacent fixture
//! lock and prints counts, never paths or user content. A failure demonstrates
//! transient flock retention across concurrent spawn despite close-on-exec.
//! Pass --explicit-unlock to compare owner release via LOCK_UN. The production
//! regression is session::tests::session_lock_release_is_independent_of_duplicate_descriptor_lifetime.
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::process::{Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
}
fn main() {
    let explicit_unlock = match std::env::args().nth(1).as_deref() {
        Some("--child") => return,
        Some("--explicit-unlock") => true,
        None => false,
        _ => panic!("expected --explicit-unlock or no arguments"),
    };
    let file = std::env::current_exe()
        .unwrap()
        .with_extension("fixture-lock");
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let worker = std::thread::spawn(move || {
        let exe = std::env::current_exe().unwrap();
        let mut n = 0;
        while !worker_stop.load(Ordering::Relaxed) && n < 2000 {
            assert!(
                Command::new(&exe)
                    .arg("--child")
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .unwrap()
                    .success()
            );
            n += 1;
        }
        n
    });
    let mut misses = 0;
    let mut acquisitions = 0;
    for _ in 0..100000 {
        let open = || {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&file)
                .unwrap()
        };
        let first = open();
        assert_ne!(unsafe { fcntl(first.as_raw_fd(), 1) } & 1, 0);
        if unsafe { flock(first.as_raw_fd(), 2 | 4) } != 0 {
            continue;
        }
        acquisitions += 1;
        if explicit_unlock {
            assert_eq!(unsafe { flock(first.as_raw_fd(), 8) }, 0);
        }
        drop(first);
        let second = open();
        if unsafe { flock(second.as_raw_fd(), 2 | 4) } != 0 {
            misses += 1;
        } else if explicit_unlock {
            assert_eq!(unsafe { flock(second.as_raw_fd(), 8) }, 0);
        }
    }
    stop.store(true, Ordering::Relaxed);
    println!(
        "explicit_unlock={explicit_unlock} acquisitions={acquisitions} immediate_reacquire_failures={misses} mock_spawns={}",
        worker.join().unwrap()
    );
    std::fs::remove_file(file).unwrap();
}
