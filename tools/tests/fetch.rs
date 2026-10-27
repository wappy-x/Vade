#![cfg(unix)]
mod common;
use common::*;
use std::{fs::{self,File},io::{BufRead,BufReader,Write},net::TcpListener,process::{Child,Command,Stdio},sync::mpsc,thread,time::{Duration,Instant}};
const REV: &str = "1234567890abcdef1234567890abcdef12345678";
fn setup(t: &Temp) {
    write(&t.0.join("metadata"),format!("{{\"sha\":\"{REV}\",\"siblings\":[{{\"rfilename\":\"weights\",\"size\":6,\"lfs\":{{\"sha256\":\"{}\"}}}}]}}",vade_tools::sha256::digest(b"hello\n")));
    write(&t.0.join("payload"),"hello\n");
}
fn command(t: &Temp, wrapper: bool) -> Command {
    let mut c = if wrapper { let mut c = Command::new("sh"); c.arg(vade_tools::root().join("ops/fetch-model.sh")); c }
        else { let mut c = Command::new(env!("CARGO_BIN_EXE_vade-tools")); c.arg("fetch-model"); c };
    c.args(["org/model",REV]).arg(t.0.join("model")).env("TEST_ROOT",&t.0).env("VADE_TOOLS",env!("CARGO_BIN_EXE_vade-tools"))
        .env("PATH",format!("{}:{}",t.0.join("bin").display(),std::env::var("PATH").unwrap())); c
}
fn until(label: &str, mut condition: impl FnMut() -> bool) {
    let start = Instant::now(); while !condition() { assert!(start.elapsed() < Duration::from_secs(10),"fetch fixture timed out: {label}"); thread::sleep(Duration::from_millis(10)); }
}
fn lock(t: &Temp) -> File { File::options().read(true).write(true).open(t.0.join("model.fetch/lock")).unwrap() }
struct Running(Child,std::path::PathBuf);
impl Drop for Running { fn drop(&mut self) { let _ = fs::write(&self.1,""); let _ = self.0.kill(); let _ = self.0.wait(); } }
#[test] fn interrupted_fetch_stops_its_downloader_and_recovers_without_unlocking_early() {
    for (wrapper,signal) in [(false,"-TERM"),(true,"-TERM"),(false,"-KILL"),(true,"-KILL")] {
        let t = Temp::new("interrupted fetch"); setup(&t);
        executable(&t.0.join("bin/curl"),r#"#!/bin/sh
for arg do case $arg in *'/api/models/'*) cat "$TEST_ROOT/metadata"; exit ;; esac; done
while [ "$#" -gt 0 ]; do if [ "$1" = --output ]; then shift; part=$1; fi; shift; done
if [ ! -f "$TEST_ROOT/release" ]; then
  stop() {
    : > "$TEST_ROOT/stopping"
    while [ ! -f "$TEST_ROOT/release" ]; do sleep 0.05; done
    : > "$TEST_ROOT/stopped"
    exit 143
  }
  trap stop HUP INT TERM
  printf hel > "$part"
  echo "$$" > "$TEST_ROOT/downloader"
  while :; do sleep 0.05; done
fi
cp "$TEST_ROOT/payload" "$part"
"#);
        let mut owner = Running(command(&t,wrapper).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap(),t.0.join("release"));
        until("download start",|| t.0.join("downloader").exists()); let downloader = fs::read_to_string(t.0.join("downloader")).unwrap();
        let held = lock(&t); assert!(held.try_lock().is_err());
        assert!(Command::new("kill").args([signal,&owner.0.id().to_string()]).status().unwrap().success());
        until("owner exit",|| owner.0.try_wait().unwrap().is_some());
        until("downloader signal",|| t.0.join("stopping").exists());
        assert!(held.try_lock().is_err(),"curl still owns the lock during signal cleanup");
        let denied = run(&mut command(&t,false),&t.0); assert!(!denied.status.success());
        assert!(String::from_utf8_lossy(&denied.stderr).contains("another fetch may be running"));
        write(&t.0.join("release"),"");
        until("lock release",|| held.try_lock().is_ok()); drop(held);
        assert!(t.0.join("stopped").exists());
        assert!(!Command::new("kill").args(["-0",downloader.trim()]).stderr(Stdio::null()).status().unwrap().success());
        let out = run(&mut command(&t,wrapper),&t.0); assert!(out.status.success(),"{}",diagnostic(&out));
        assert_eq!(fs::read(t.0.join("model/weights")).unwrap(),b"hello\n"); lock(&t).try_lock().unwrap();
    }
}
struct Stall { url: String, stop: Option<mpsc::Sender<()>>, task: Option<thread::JoinHandle<()>> }
impl Stall {
    fn start(body: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap(); listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}",listener.local_addr().unwrap()); let (send,recv) = mpsc::channel();
        let task = thread::spawn(move || {
            let start = Instant::now();
            let mut tcp = loop { match listener.accept() { Ok((s,_)) => break s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && start.elapsed() < Duration::from_secs(5) => thread::sleep(Duration::from_millis(5)),
                _ => return } };
            tcp.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut reader = BufReader::new(tcp.try_clone().unwrap());
            loop { let mut line = String::new(); match reader.read_line(&mut line) { Ok(0)|Err(_) => return, Ok(_) if line == "\r\n" => break, _ => () } }
            if body { tcp.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nh").unwrap(); }
            let _ = recv.recv_timeout(Duration::from_secs(5));
        }); Self {url,stop:Some(send),task:Some(task)}
    }
}
impl Drop for Stall { fn drop(&mut self) { let _ = self.stop.take().unwrap().send(()); self.task.take().unwrap().join().unwrap(); } }
#[test] fn stalled_headers_and_body_fail_and_release_the_fetch_lock() {
    for body in [false,true] {
        let t = Temp::new("stalled fetch"); setup(&t); let stall = Stall::start(body);
        // Check the production bounds, then shorten them only in this fixture.
        // Real curl must report the timeout against a local peer that stays open.
        executable(&t.0.join("bin/curl"),r#"#!/bin/sh
speed=; limit=; max=; part=
while [ "$#" -gt 0 ]; do
  case $1 in --speed-time) shift; speed=$1 ;; --speed-limit) shift; limit=$1 ;; --max-time) shift; max=$1 ;; --output) shift; part=$1 ;; esac
  shift
done
[ "$speed" = 120 ] && [ "$limit" = 1 ] || exit 90
if [ -z "$part" ]; then
  [ "$max" = 120 ] || exit 91
  if [ "$STALL_BODY" = 1 ]; then cat "$TEST_ROOT/metadata"; exit; fi
  exec "$REAL_CURL" --disable --fail -sS --max-time 1 "$STALL_URL"
fi
[ -z "$max" ] || exit 92
exec "$REAL_CURL" --disable --fail -sS --speed-time 1 --speed-limit 1 --output "$part" "$STALL_URL"
"#);
        let out = run(command(&t,false).env("REAL_CURL",which("curl")).env("STALL_URL",&stall.url).env("STALL_BODY",if body {"1"} else {"0"}),&t.0);
        drop(stall); assert!(!out.status.success(),"{}",diagnostic(&out));
        assert!(String::from_utf8_lossy(&out.stderr).contains("(28)"),"{}",diagnostic(&out));
        lock(&t).try_lock().unwrap(); assert!(!t.0.join("model.sha256").exists());
        if body { assert_eq!(fs::read(t.0.join("model.fetch/download-0.part")).unwrap(),b"h"); }
    }
}
