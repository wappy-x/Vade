//! Public machine labels. SSH destinations stay in private config and records.
use super::Config;

#[derive(Clone, Default)]
pub(crate) struct PublicNames { replacements: Vec<(String, String)>, active: bool }

/// An IPv4 address, with dots or hyphens, at the start of this text.
fn address(value: &str) -> Option<(usize, bool)> {
    let bytes = value.as_bytes();
    let mut at = 0;
    let mut separator = 0;
    let mut local = false;
    for part in 0..4 {
        let start = at;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) { at += 1; }
        if at == start || at - start > 3 { return None; }
        let number = value[start..at].parse::<u8>().ok()?;
        if part == 0 { local = number == 127; }
        if part != 3 {
            let next = *bytes.get(at)?;
            if part == 0 { separator = next; }
            if !matches!(next, b'.' | b'-') || next != separator { return None; }
            at += 1;
        }
    }
    Some((at, local))
}

fn hide_addresses(value: &str) -> String {
    let mut out = String::new();
    let mut rest = value;
    while !rest.is_empty() {
        if let Some((end, local)) = address(rest).filter(|_| !out.ends_with(|ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))) {
            out.push_str(if local { "localhost" } else { "private-address" });
            rest = &rest[end..];
        } else {
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    out
}

/// A short machine name must not contain an address in either common form.
pub(crate) fn hostname(value: &str) -> bool {
    crate::cli::identifier(value) && !value.char_indices().any(|(at, _)| address(&value[at..]).is_some())
}

impl PublicNames {
    /// Labels before setup can reach a box. A plain SSH name is safe to show.
    pub(crate) fn hosts(hosts: &[String]) -> Self {
        let mut names = Self { active: true, ..Self::default() };
        for (index, host) in hosts.iter().enumerate() {
            let label = if hostname(host) { host.clone() } else { format!("spark-{}", index + 1) };
            names.add(host, &label);
        }
        names
    }

    pub(crate) fn config(config: &Config) -> Self {
        let mut names = Self::default();
        names.machines(&config.machines);
        names
    }

    pub(crate) fn controller(config: &Config) -> Self {
        let mut names = Self::config(config);
        if let Ok(Some(session)) = crate::state::Session::load(&config.state_dir) { names.session(&session); }
        for group in config.group_record_names().unwrap_or_default() {
            if let Ok(Some(session)) = crate::state::Session::load(&config.group_state_dir(&group)) { names.session(&session); }
        }
        names
    }

    pub(crate) fn machines(&mut self, machines: &[super::Machine]) {
        self.active = true;
        for (index, machine) in machines.iter().enumerate() {
            let label = if hostname(&machine.name) { machine.name.clone() } else { format!("spark-{}", index + 1) };
            self.add(&machine.name, &label);
            self.add(&machine.ssh, &label);
            for address in machine.fabric.values() { self.add(address, "cable-address"); }
        }
    }

    pub(crate) fn session(&mut self, session: &crate::state::Session) {
        self.machines(&session.ranks.iter().map(|rank| rank.machine.clone()).collect::<Vec<_>>());
    }

    fn add(&mut self, value: &str, label: &str) {
        let host = value.rsplit('@').next().unwrap_or(value);
        if value == label || (hostname(value) && !value.contains('@')) { return; }
        let mut values = vec![value.to_owned()];
        if !hostname(host) { values.extend([host.to_owned(), host.replace('.', "-")]); }
        for value in values {
            if !value.is_empty() && value != label && !self.replacements.iter().any(|(old, _)| old == &value) {
                // A controller endpoint must still refer to this computer.
                let label = if value.parse::<std::net::Ipv4Addr>().is_ok_and(|ip| ip.is_loopback()) { "localhost" } else { label };
                self.replacements.push((value, label.to_owned()));
            }
        }
        // Replace the complete destination before its host part.
        self.replacements.sort_by_key(|(value, _)| std::cmp::Reverse(value.len()));
    }

    pub(crate) fn text(&self, value: &str) -> String {
        if !self.active { return value.to_owned(); }
        let mut out = String::new();
        let mut rest = value;
        while !rest.is_empty() {
            let found = self.replacements.iter().filter_map(|(old, label)| {
                rest.match_indices(old).find(|(at, _)| !old.ends_with(|ch: char| ch.is_ascii_digit()) || !rest.as_bytes().get(at + old.len()).is_some_and(u8::is_ascii_digit))
                    .map(|(at, _)| (at, old, label))
            }).min_by_key(|(at, _, _)| *at);
            let Some((at, old, label)) = found else { out.push_str(rest); break };
            out.push_str(&rest[..at]);
            out.push_str(label);
            rest = &rest[at + old.len()..];
        }
        // Tool output can also name a cable or a resolved address absent from config.
        hide_addresses(&out)
    }

    /// Protect streamed output and the error that main prints after this scope.
    pub(crate) fn run<T>(&self, run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        crate::output::with_names(self.clone(), run).map_err(|error| self.text(&error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn destinations_and_both_address_forms_use_labels_without_changing_private_input() {
        let hosts = ["user@192.0.2.1", "user@192.0.2.10"].map(String::from);
        let names = PublicNames::hosts(&hosts);
        let input = "user@192.0.2.10 192.0.2.1 192-0-2-10 192.0.2.99 192-0-2-99 192.0.2.100";
        assert_eq!(names.text(input), "spark-2 spark-1 spark-2 private-address private-address private-address");
        assert_eq!(hosts[0], "user@192.0.2.1");
        assert_eq!(hosts[1], "user@192.0.2.10");
        let plain = "vade 0.1.0, context 32768, λ, set-4-2-2-1, layout-4-2-2";
        assert_eq!(names.text(plain), plain);
    }
    #[test]
    fn names_reject_addresses_and_accept_short_hostnames() {
        for name in ["192.0.2.10", "192-0-2-10", "ip-192-0-2-10", "user@192.0.2.10", ""] { assert!(!hostname(name)); }
        for name in ["box-one", "spark-1", "box_2"] { assert!(hostname(name)); }
    }
}
