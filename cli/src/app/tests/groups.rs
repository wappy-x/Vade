//! [group] sections: status, down, install and update see every group
//! ownership record, also one no longer in the config.
use super::*;
use crate::{evals::json, status::Snapshot};

#[test]
fn update_restart_hints_keep_the_selected_group() {
    let mut f = Fixture::fp4(1, false);
    f.app.config.lock_dir = f.app.config.state_dir.clone();
    f.app.config.groups.push(crate::config::Group { name: "solo".into(), hosts: vec!["a".into()], local_port: 18090, layout: "singles".into() });
    f.app.config.select_group("solo").unwrap();
    f.app.up(&fp4_options(&[])).unwrap();
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.update().unwrap());
    let output = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(output.contains("\n  VADE_GROUP=solo vade down\n  VADE_GROUP=solo vade up --hosts a "), "{output}");
}

#[test]
fn status_without_a_group_names_the_groups_that_own_a_server() {
    let mut f = Fixture::fp4(4, false);
    f.app.config.lock_dir = f.app.config.state_dir.clone();
    for (name, hosts) in [("left", ["a", "b"]), ("right", ["c", "d"])] {
        f.app.config.groups.push(crate::config::Group { name: name.into(), hosts: hosts.map(String::from).to_vec(), local_port: 18080, layout: name.into() });
    }
    let record = f.app.config.state_dir.join("groups/right");
    fs::create_dir_all(&record).unwrap(); fs::write(record.join("session.conf"), "owned\n").unwrap();
    let status = |f: &mut Fixture| { let capture = crate::output::testing::Capture::default(); crate::output::with_writer(capture.clone(), || f.app.status()).unwrap(); String::from_utf8(capture.bytes.borrow().clone()).unwrap() };
    let printed = status(&mut f);
    assert!(printed.starts_with("○ Vade has no server of its own; 1 group(s) still own one.\n") && !printed.contains("Vade is stopped"), "{printed}");
    assert!(printed.contains("Groups with an ownership record: right. Run VADE_GROUP=NAME vade status to see one, VADE_GROUP=NAME vade down to stop one."), "{printed}");
    // Plain vade and vade chat name the groups too, and never say stopped.
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.menu()).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.starts_with("○ Vade has no server of its own; 1 group(s) still own one.\nGroups with an ownership record: right.") && !printed.contains("stopped"), "{printed}");
    let error = f.app.chat(&crate::cli::ChatOptions::default()).unwrap_err();
    assert_eq!(error, "Vade has no server of its own; groups own servers: right. Run VADE_GROUP=NAME vade chat for one of them.");
    // Plain vade down names them too, and vade status --json lists them.
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.down()).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.starts_with("Vade has no server of its own to stop; groups still own servers.\nGroups with an ownership record: right."), "{printed}");
    let mut report = Snapshot::new("configured", false);
    f.app.status_snapshot(&mut report).unwrap();
    let value = json::parse(&report.finish(Duration::ZERO).0).unwrap();
    assert_eq!((value["phase"].as_str(), value["groups"].as_array().map(Vec::len)), (Some("stopped"), Some(1)), "{value:?}");
    // A group removed from the config while its server runs is still named.
    let record = f.app.config.state_dir.join("groups/old");
    fs::create_dir_all(&record).unwrap(); fs::write(record.join("session.conf"), "owned\n").unwrap();
    let printed = status(&mut f);
    assert!(printed.contains("Groups with an ownership record: old, right.") && printed.contains("No [group] section for old: restore it"), "{printed}");
    f.app.config.select_group("left").unwrap();
    assert!(!status(&mut f).contains("Groups with an ownership record"));
    // With no group record left, plain status is stopped again.
    let mut f = Fixture::fp4(4, false);
    f.app.config.lock_dir = f.app.config.state_dir.clone();
    assert!(status(&mut f).starts_with("○ Vade is stopped."));
    assert!(f.app.chat(&crate::cli::ChatOptions::default()).unwrap_err().starts_with("Vade is stopped. Run "));
}

#[test]
fn install_treats_every_group_record_as_owned_engines() {
    let mut f = Fixture::fp4(4, false);
    f.app.config.lock_dir = f.app.config.state_dir.clone();
    for (name, hosts, port) in [("left", ["a", "b"], 18080), ("right", ["c", "d"], 18090)] {
        f.app.config.groups.push(crate::config::Group { name: name.into(), hosts: hosts.map(String::from).to_vec(), local_port: port, layout: "pairs".into() });
    }
    // Group right runs; install runs with group left selected.
    let mut right = f.app.config.clone();
    right.select_group("right").unwrap();
    let session = Session::new(&right, &fp4_options(&[]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    session.save(&right.state_dir).unwrap();
    f.app.config.select_group("left").unwrap();
    f.app.runner.calls.clear();
    f.app.execute(Command::Install).unwrap();
    let checked: Vec<&str> = f.app.runner.calls.iter().map(|exec| exec.label.as_str()).filter(|label| label.ends_with("build GPU check")).collect();
    assert_eq!(checked, ["a build GPU check", "b build GPU check"], "the running group's engines are owned, not outside jobs");
    // Still owned after group right leaves the config: its record folder counts.
    f.app.config.groups.retain(|group| group.name != "right");
    f.app.runner.calls.clear();
    f.app.execute(Command::Install).unwrap();
    let checked: Vec<&str> = f.app.runner.calls.iter().map(|exec| exec.label.as_str()).filter(|label| label.ends_with("build GPU check")).collect();
    assert_eq!(checked, ["a build GPU check", "b build GPU check"], "a removed group's engines are owned, not outside jobs");
}
