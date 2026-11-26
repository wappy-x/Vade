#![allow(dead_code)]
use std::{fs,path::{Path,PathBuf},process::{Command,Output,Stdio},sync::atomic::{AtomicUsize,Ordering},time::{Duration,Instant}};
static NEXT: AtomicUsize = AtomicUsize::new(0);
pub struct Temp(pub PathBuf);
impl Temp {
    pub fn new(name: &str) -> Self {
        let root = vade_tools::root().join(".local/tool-tests"); fs::create_dir_all(&root).unwrap();
        let path = root.join(format!("{name} {} {}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&path).unwrap(); Self(path.canonicalize().unwrap())
    }
}
impl Drop for Temp { fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); } }
pub fn write(path: &Path, data: impl AsRef<[u8]>) { fs::create_dir_all(path.parent().unwrap()).unwrap(); fs::write(path,data).unwrap(); }
pub fn executable(path: &Path, data: &str) {
    use std::os::unix::fs::PermissionsExt;
    write(path,data); fs::set_permissions(path,fs::Permissions::from_mode(0o755)).unwrap();
}
pub fn which(name: &str) -> String {
    std::env::split_paths(&std::env::var_os("PATH").unwrap()).map(|p|p.join(name)).find(|p|p.is_file()).unwrap().to_str().unwrap().into()
}
pub fn quote(s: &str) -> String { format!("'{}'",s.replace('\'',"'\\''")) }
// Bound each real shell test. Use files so a full pipe cannot block the child.
pub fn run(command: &mut Command, temp: &Path) -> Output {
    let out = temp.join("stdout"); let err = temp.join("stderr");
    let mut child = command.stdout(Stdio::from(fs::File::create(&out).unwrap())).stderr(Stdio::from(fs::File::create(&err).unwrap())).spawn().unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() { break status; }
        if start.elapsed() > Duration::from_secs(30) { child.kill().unwrap(); child.wait().unwrap(); panic!("shell fixture exceeded 30 seconds"); }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output { status, stdout:fs::read(out).unwrap(),stderr:fs::read(err).unwrap() }
}
pub fn diagnostic(out: &Output) -> String { format!("{}\n{}\n{}",out.status,String::from_utf8_lossy(&out.stdout),String::from_utf8_lossy(&out.stderr)) }
