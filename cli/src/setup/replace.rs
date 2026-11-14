//! The one saved setup. A list of Sparks is a new answer, and it wins over what
//! exists: the local setup and the controller config move aside under free
//! names, and nothing is deleted. All checks run first, so a refusal leaves every
//! file as it was. A running engine stops first through `vade down`, so only the
//! ownership record decides what stops.
use super::walk::{run_step, step_command, LocalSetup};
use crate::{config::Config, output, state::Session, text};
use std::path::{Path, PathBuf};

/// The first words of a config that `vade setup` wrote.
pub(super) const WRITTEN: &str = "# Written by vade setup. Start: vade up";

/// A controller config that is in place.
pub(super) struct Controller {
    pub path: PathBuf,
    pub aside: PathBuf,
    /// The file as it is. A file that does not parse still moves aside.
    pub text: String,
    pub config: Option<Config>,
    pub running: bool,
}

/// What exists before a new answer takes its place.
pub(super) struct Held {
    pub local: Option<LocalSetup>,
    pub controller: Option<Controller>,
}

impl Held {
    /// Read what exists. `local` says that the local setup is the selected one.
    pub(super) fn read(path: &Path, local: bool, root: &Path) -> Result<Self, String> {
        let local = if local { Some(LocalSetup::find(root)?) } else { None };
        if !path.is_file() { return Ok(Self { local, controller: None }); }
        let text = std::fs::read_to_string(path).map_err(|_| format!("Cannot read {}. Check its permissions.", text::quoted(&path.to_string_lossy())))?;
        let config = Config::load().ok();
        let mut running = false;
        if let Some(config) = config.as_ref().filter(|config| config.group.is_none()) {
            running = Session::load(&config.state_dir)?.is_some();
            for name in config.group_record_names()? {
                if Session::load(&config.group_state_dir(&name))?.is_some() {
                    return Err(format!("Group {name} still runs, so {}, then run the same command again.", config.group_down_step(&name)));
                }
            }
        }
        let aside = free(path, "before");
        Ok(Self { local, controller: Some(Controller { path: path.to_owned(), aside, text, config, running }) })
    }

    /// Whether a group selection owns these machines. `vade up` then keeps its own refusal.
    pub(super) fn grouped(&self) -> bool { self.loaded().is_some_and(|config| config.group.is_some()) }

    fn loaded(&self) -> Option<&Config> { self.controller.as_ref()?.config.as_ref() }

    /// The checkpoint to keep when a list names the Sparks of this config and no
    /// checkpoint flag: the same Sparks keep their model.
    pub(super) fn checkpoint(&self, hosts: &[String]) -> Option<bool> {
        let config = self.loaded()?;
        let mut sparks = sparks(config);
        let mut given = hosts.to_vec();
        sparks.sort_unstable();
        given.sort_unstable();
        (sparks == given).then(|| config.model.ends_with("-fp8"))
    }

    /// Whether this config already holds `hosts` in this order, with this
    /// checkpoint. In a config by hand, a selection that passes its checks
    /// counts too: an eight-Spark config holds many chains.
    pub(super) fn same(&self, hosts: &[String], fp8: Option<bool>) -> bool {
        let (Some(config), Some(controller)) = (self.loaded(), &self.controller) else { return false };
        if fp8.is_some_and(|fp8| config.model.ends_with("-fp8") != fp8) { return false; }
        if controller.text.starts_with(WRITTEN) { return sparks(config) == hosts; }
        let mut up = vec!["--hosts".to_owned(), hosts.join(",")];
        if config.model.ends_with("-fp8") { up.insert(0, "--fp8".to_owned()); }
        super::walk::check_start(config, &up).is_ok()
    }

    /// Whether `text`, the new config, says the same as this one.
    pub(super) fn same_text(&self, text: &str) -> bool { self.controller.as_ref().is_some_and(|controller| controller.text == text) }

    /// The one line for this change. `sparks` counts the new Sparks.
    pub(super) fn line(&self, sparks: usize, dry: bool) -> Option<String> {
        let old = match (&self.local, &self.controller) {
            (Some(_), _) => "the one-Spark setup".to_owned(),
            (None, Some(Controller { config: Some(config), .. })) => format!("the {}-Spark setup", word(config.machines.len())),
            (None, Some(controller)) => format!("the setup in {}", text::quoted(&controller.path.to_string_lossy())),
            (None, None) => return None,
        };
        let moves: Vec<(&Path, &Path)> = self.local.iter().map(|local| (local.file.as_path(), local.aside.as_path()))
            .chain(self.controller.iter().map(|controller| (controller.path.as_path(), controller.aside.as_path()))).collect();
        let places = moves.iter().map(|(_, aside)| text::quoted(&aside.to_string_lossy())).collect::<Vec<_>>().join(" and ");
        let now = if dry { format!("Setup would move the old setup to {places}.") } else { format!("The old setup is now in {places}.") };
        Some(format!("Replacing {old} with {} {}. {now}", word(sparks), if sparks == 1 { "Spark" } else { "Sparks" }))
    }

    /// Stop a running engine of what moves aside, with `vade down`, as a person would.
    pub(super) fn stop(&self, program: &Path, yes: bool) -> Result<(), String> {
        if self.local.as_ref().is_some_and(|local| local.running) {
            output::line(format_args!("Stopping the one-Spark engine."))?;
            down(program, &[], yes)?;
        }
        if let Some(controller) = self.controller.as_ref().filter(|controller| controller.running) {
            output::line(format_args!("Stopping the {}-Spark engine.", controller.config.as_ref().map_or_else(|| "old".to_owned(), |config| word(config.machines.len()))))?;
            down(program, &[("VADE_CONFIG".to_owned(), controller.path.to_string_lossy().into_owned())], yes)?;
        }
        Ok(())
    }

    /// Move each file aside. A failure puts back what has moved.
    pub(super) fn move_aside(&self, root: &Path) -> Result<(), String> {
        if let Some(local) = &self.local { local.move_aside(root)?; }
        if let Some(controller) = &self.controller {
            let lock = controller.config.as_ref().map(|config| crate::state::Lock::acquire(&config.lock_dir)).transpose();
            let _lock = match lock {
                Ok(lock) => lock,
                Err(error) => { self.restore(); return Err(error); }
            };
            if let Err(error) = std::fs::rename(&controller.path, &controller.aside) {
                drop(error);
                self.restore();
                return Err(format!("Cannot move {} aside. Check its permissions, then run the same command again.", text::quoted(&controller.path.to_string_lossy())));
            }
        }
        Ok(())
    }

    /// Put each file back. A failure leaves it at the path that the line printed.
    pub(super) fn restore(&self) {
        if let Some(local) = &self.local { local.restore(); }
        if let Some(controller) = &self.controller { let _ = std::fs::rename(&controller.aside, &controller.path); }
    }
}

/// The ssh values of a config, in rank order.
fn sparks(config: &Config) -> Vec<String> { config.machines.iter().map(|machine| machine.ssh.clone()).collect() }

/// Run `vade down`. The file stays in place when it fails, so the command can run again.
fn down(program: &Path, env: &[(String, String)], yes: bool) -> Result<(), String> {
    let args = ["down".to_owned()];
    let (passed, stopped) = run_step(step_command(program, &args, env, yes), &crate::runner::interrupted)
        .map_err(|_| "Cannot run vade down. Check that the vade program still exists, then run the same command again.")?;
    if passed && !stopped { return Ok(()); }
    Err("vade down did not finish, so the old setup stays in place. Fix the cause it printed above, then run the same command again.".into())
}

/// The number as a word in a sentence, up to the largest cabled group we name.
pub(super) fn word(count: usize) -> String {
    const WORDS: [&str; 8] = ["one", "two", "three", "four", "five", "six", "seven", "eight"];
    WORDS.get(count.wrapping_sub(1)).map_or_else(|| count.to_string(), |word| (*word).to_owned())
}

/// The first free name beside `path`: `config.before`, then `config.before-2`.
pub(super) fn free(path: &Path, tag: &str) -> PathBuf {
    let name = path.file_name().map_or_else(|| "config".to_owned(), |name| name.to_string_lossy().into_owned());
    (1..).map(|number| path.with_file_name(if number == 1 { format!("{name}.{tag}") } else { format!("{name}.{tag}-{number}") }))
        .find(|candidate| std::fs::symlink_metadata(candidate).is_err()).unwrap_or_else(|| path.with_file_name(format!("{name}.{tag}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(local: bool, sparks: usize) -> Held {
        let machines: String = (0..sparks).map(|index| format!("[machine m{index}]\nssh = host-{index}\n")).collect();
        let text = format!("[global]\n{machines}");
        let config = (sparks > 0).then(|| Config::parse(&text, Path::new("/home/you/config")).unwrap());
        Held {
            local: local.then(|| LocalSetup { file: "/home/you/.vade/config".into(), aside: "/home/you/.vade/config.local".into(), running: false }),
            controller: config.map(|config| Controller { path: "/home/you/config".into(), aside: "/home/you/config.before".into(), text, config: Some(config), running: false }),
        }
    }

    #[test]
    fn the_replace_sentence_names_the_old_and_new_shape() {
        assert_eq!(held(false, 2).line(4, false).unwrap(), "Replacing the two-Spark setup with four Sparks. The old setup is now in \"/home/you/config.before\".");
        assert_eq!(held(true, 0).line(2, false).unwrap(), "Replacing the one-Spark setup with two Sparks. The old setup is now in \"/home/you/.vade/config.local\".");
        assert_eq!(held(false, 4).line(1, false).unwrap(), "Replacing the four-Spark setup with one Spark. The old setup is now in \"/home/you/config.before\".");
        assert_eq!(held(true, 2).line(4, true).unwrap(), "Replacing the one-Spark setup with four Sparks. Setup would move the old setup to \"/home/you/.vade/config.local\" and \"/home/you/config.before\".");
        assert_eq!(held(false, 0).line(2, false), None);
    }

    #[test]
    fn a_file_that_does_not_parse_is_named_by_its_path() {
        let controller = Controller { path: "/home/you/config".into(), aside: "/home/you/config.before".into(), text: "not a config".into(), config: None, running: false };
        assert_eq!(Held { local: None, controller: Some(controller) }.line(2, false).unwrap(),
            "Replacing the setup in \"/home/you/config\" with two Sparks. The old setup is now in \"/home/you/config.before\".");
    }

    #[test]
    fn the_stop_is_vade_down_with_the_file_of_that_setup() {
        use std::os::unix::fs::PermissionsExt;
        let _guard = crate::runner::process_test_lock();
        let dir = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-stop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let program = dir.join("vade");
        let write = |code: i32| {
            std::fs::write(&program, format!("#!/bin/sh\nprintf '%s %s\n' \"$1\" \"${{VADE_CONFIG:-none}}\" >> '{}/calls'\nexit {code}\n", dir.display())).unwrap();
            std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        };
        write(0);
        let mut both = held(true, 2);
        both.local.as_mut().unwrap().running = true;
        both.controller.as_mut().unwrap().running = true;
        both.stop(&program, true).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("calls")).unwrap(), "down none\ndown /home/you/config\n");
        // Nothing runs, so nothing stops.
        std::fs::remove_file(dir.join("calls")).unwrap();
        held(true, 2).stop(&program, true).unwrap();
        assert!(!dir.join("calls").exists());
        write(1);
        assert!(both.stop(&program, true).unwrap_err().starts_with("vade down did not finish"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A lock that cannot be taken stops the move. The file that moved first goes back.
    #[test]
    fn a_lock_failure_on_the_controller_puts_the_local_setup_back() {
        let dir = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-move-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let root = dir.join("root");
        crate::text::private_dir(&root).unwrap();
        std::fs::write(root.join("config"), "saved\n").unwrap();
        // A file in place of a folder, so the lock in that folder cannot be taken.
        std::fs::write(dir.join("not-a-folder"), "").unwrap();
        let text = format!("[global]\nstate_dir = {}\n[machine a]\nssh = host-a\n", dir.join("not-a-folder/state").display());
        let path = dir.join("controller");
        std::fs::write(&path, &text).unwrap();
        let config = Config::parse(&text, &path).unwrap();
        let held = Held {
            local: Some(LocalSetup::find(&root).unwrap()),
            controller: Some(Controller { path: path.clone(), aside: dir.join("controller.before"), text, config: Some(config), running: false }),
        };
        assert!(held.move_aside(&root).is_err());
        assert_eq!(std::fs::read_to_string(root.join("config")).unwrap(), "saved\n");
        assert!(!root.join("config.local").exists() && path.is_file() && !dir.join("controller.before").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn numbers_read_as_words_up_to_eight() {
        assert_eq!((word(1), word(8), word(9), word(24)), ("one".to_owned(), "eight".to_owned(), "9".to_owned(), "24".to_owned()));
    }

    #[test]
    fn a_free_name_never_takes_a_file_that_exists() {
        let dir = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-free-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("config");
        assert_eq!(free(&path, "before"), dir.join("config.before"));
        std::fs::write(dir.join("config.before"), "").unwrap();
        assert_eq!(free(&path, "before"), dir.join("config.before-2"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_hand_config_keeps_its_chains_and_a_written_config_only_its_own() {
        let machines = "[machine a]\nssh = host-a\nfabric.b = 192.0.2.1\n[machine b]\nssh = host-b\nfabric.a = 192.0.2.2\nfabric.c = 192.0.2.3\n[machine c]\nssh = host-c\nfabric.b = 192.0.2.4\n";
        let make = |text: String| {
            let config = Config::parse(&text, Path::new("/home/you/config")).unwrap();
            Held { local: None, controller: Some(Controller { path: "/home/you/config".into(), aside: "/home/you/config.before".into(), text, config: Some(config), running: false }) }
        };
        let (a, b, c) = ("host-a".to_owned(), "host-b".to_owned(), "host-c".to_owned());
        let hand = make(format!("[global]\n{machines}"));
        assert!(hand.same(&[a.clone(), b.clone()], None) && hand.same(&[a.clone(), b.clone(), c.clone()], None));
        assert!(!hand.same(&[a.clone(), c.clone()], None), "no cable in the config");
        let written = make(format!("{WRITTEN} --hosts host-a,host-b,host-c\n[global]\n{machines}"));
        assert!(written.same(&[a.clone(), b.clone(), c.clone()], None) && !written.same(&[a.clone(), b.clone()], None));
        assert!(!written.same(&[c.clone(), b.clone(), a.clone()], None));
        assert!(!written.same(&[a.clone(), b.clone(), c.clone()], Some(true)), "another checkpoint is another setup");
        assert_eq!(written.checkpoint(&[c, a, b]), Some(false));
        assert_eq!(written.checkpoint(&["host-z".to_owned()]), None);
    }
}
