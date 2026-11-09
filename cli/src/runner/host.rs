//! Whether an SSH destination names this computer. A controller that is also
//! one of its configured Sparks runs that Spark's commands as local child
//! processes: a fresh Spark has no SSH key for itself, and SSH to itself is
//! not needed.
use std::collections::BTreeMap;
use std::net::IpAddr;

/// What identifies this computer.
#[derive(Clone, Debug, Default)]
pub(crate) struct Host {
    /// The host name, as the kernel reports it.
    pub name: Option<String>,
    /// This account; a destination with another user is not local.
    pub user: Option<String>,
    /// Own addresses. The real host instead tests each address with a bind.
    pub addresses: Vec<IpAddr>,
    /// `ssh -G` answers: destination to resolved (host name, user). The real
    /// host asks `ssh -G`, which reads the SSH config and opens no connection.
    pub aliases: BTreeMap<String, (String, String)>,
    /// Ask this computer: bind to an address, run `ssh -G`. False for fakes.
    pub real: bool,
}

impl Host {
    /// Whether `name` is this computer: localhost, a loopback or own address,
    /// or this host name. A short name matches a full name with the same
    /// first label.
    fn named(&self, name: &str) -> bool {
        let name = name.trim_start_matches('[').trim_end_matches(']').trim_end_matches('.').to_ascii_lowercase();
        if name == "localhost" {
            return true;
        }
        if let Ok(ip) = name.parse::<IpAddr>() {
            return ip.is_loopback()
                || self.addresses.contains(&ip)
                || (self.real && !ip.is_unspecified() && !ip.is_multicast() && std::net::UdpSocket::bind((ip, 0)).is_ok());
        }
        let Some(own) = self.name.as_deref().map(str::to_ascii_lowercase).filter(|own| !own.is_empty()) else { return false };
        let short = |value: &str| value.split('.').next().unwrap_or_default().to_owned();
        name == own || ((!name.contains('.') || !own.contains('.')) && short(&name) == short(&own))
    }

    /// The host name and user SSH would use for `destination`, or None.
    fn resolve(&self, destination: &str) -> Option<(String, String)> {
        if !self.real {
            return self.aliases.get(destination).cloned();
        }
        ssh_resolve(destination)
    }

    /// The facts of the computer this process runs on. Only a Linux computer
    /// can be a Spark rank; elsewhere only loopback names match.
    fn this() -> Self {
        let user = std::env::var("USER").or_else(|_| std::env::var("LOGNAME")).ok().filter(|user| !user.is_empty());
        if !cfg!(target_os = "linux") {
            return Self { user, ..Self::default() };
        }
        let name = std::fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|name| name.trim().to_owned()).filter(|name| !name.is_empty());
        Self { name, user, real: true, ..Self::default() }
    }
}

/// Whether `destination` (an alias, host name or address, optionally
/// `user@`) names this computer and this account, by the facts in `host`.
pub(crate) fn names(destination: &str, host: &Host) -> bool {
    let (user, name) = destination.rsplit_once('@').map_or((None, destination), |(user, name)| (Some(user), name));
    let (name, user) = match host.resolve(destination) {
        Some((name, user)) => (name, Some(user)),
        None => (name.to_owned(), user.map(str::to_owned)),
    };
    host.named(&name) && user.is_none_or(|user| host.user.as_deref() == Some(user.as_str()))
}

/// `ssh -G destination`: the resolved `hostname` and `user`. It only reads
/// the SSH config. It is bounded to 5 s; a failure means "not resolved".
fn ssh_resolve(destination: &str) -> Option<(String, String)> {
    use std::process::{Command, Stdio};
    if destination.starts_with('-') {
        return None;
    }
    let mut child = Command::new("ssh").args(["-G", destination]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < until => std::thread::sleep(std::time::Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let mut text = String::new();
    std::io::Read::read_to_string(&mut child.stdout.take()?, &mut text).ok()?;
    let value = |key: &str| text.lines().find_map(|line| line.strip_prefix(key)?.strip_prefix(' ').map(str::trim).map(str::to_owned));
    Some((value("hostname")?, value("user")?))
}

#[cfg(test)]
thread_local! {
    static FAKE: std::cell::RefCell<Option<Host>> = const { std::cell::RefCell::new(None) };
}

/// Use `host` instead of this computer's facts on this test thread until the
/// guard drops.
#[cfg(test)]
pub(crate) fn fake(host: Host) -> FakeGuard {
    FAKE.with(|fake| *fake.borrow_mut() = Some(host));
    FakeGuard
}
#[cfg(test)]
pub(crate) struct FakeGuard;
#[cfg(test)]
impl Drop for FakeGuard {
    fn drop(&mut self) {
        FAKE.with(|fake| *fake.borrow_mut() = None);
    }
}

/// Whether `destination` names this computer. The answer is kept for the
/// life of the process, so one command sees one answer for each machine.
pub(crate) fn this_computer(destination: &str) -> bool {
    #[cfg(test)]
    if let Some(answer) = FAKE.with(|fake| fake.borrow().as_ref().map(|host| names(destination, host))) {
        return answer;
    }
    static HOST: std::sync::OnceLock<Host> = std::sync::OnceLock::new();
    static ANSWERS: std::sync::Mutex<BTreeMap<String, bool>> = std::sync::Mutex::new(BTreeMap::new());
    let mut answers = ANSWERS.lock().unwrap_or_else(|error| error.into_inner());
    *answers.entry(destination.to_owned()).or_insert_with(|| names(destination, HOST.get_or_init(Host::this)))
}
