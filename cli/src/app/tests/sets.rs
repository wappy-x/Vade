//! Saved inventory and server sets. FakeRunner opens no connection.
use super::*;
use crate::{evals::json::{self, Value}, status::Snapshot};

fn planned(count: usize) -> Fixture {
    let mut f = Fixture::fp4(count, false);
    let names = f.app.config.machines.iter().map(|machine| machine.name.clone()).collect::<Vec<_>>();
    f.app.config.groups = crate::sets::plan(&names);
    f.app.config.lock_dir = f.app.config.state_dir.clone();
    if let [first, _, .., last] = names.as_slice() {
        f.app.config.machines.first_mut().unwrap().fabric.insert(last.clone(), "closing-first".into());
        f.app.config.machines.last_mut().unwrap().fabric.insert(first.clone(), "closing-last".into());
    }
    f
}

fn snapshot(f: &mut Fixture) -> Value {
    f.app.runner.calls.clear();
    f.app.dry = true;
    let mut report = Snapshot::new("configured", true);
    f.app.status_snapshot(&mut report).unwrap();
    assert!(f.app.runner.calls.is_empty(), "saved set labels need no device probe");
    let (encoded, code) = report.finish(Duration::ZERO);
    assert_eq!(code, 0);
    json::parse(&encoded).unwrap()
}

fn printed(f: &mut Fixture) -> String {
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.status()).unwrap();
    String::from_utf8(capture.bytes.borrow().clone()).unwrap()
}

#[test]
fn status_reports_stopped_and_saved_sets_for_each_inventory() {
    for count in [1, 2, 4, 6, 8] {
        let mut f = planned(count);
        let value = snapshot(&mut f);
        assert_eq!(value["phase"].as_str(), Some("stopped"));
        assert!(value["server_sets"].as_array().unwrap().is_empty());
        let text = printed(&mut f);
        if count == 8 {
            assert!(text.contains("Cabling: ring of 8:"), "{text}");
            assert!(value["cabling"][0].as_str().unwrap().starts_with("ring of 8:"));
        }
        for layout in value["layouts"].as_array().unwrap() {
            assert!(text.contains(layout["name"].as_str().unwrap()) && text.contains(layout["shape"].as_str().unwrap()), "{text}");
        }
        // Save a server from the fake inventory. Its ranks, not the current
        // selection, supply its set label.
        let options = fp4_options(&["--sparks", &count.to_string()]);
        let session = Session::new(&f.app.config, &options, &Manifest::load(&f.app.config.build_manifest).unwrap(), "0".repeat(64)).unwrap();
        session.save(&f.app.config.state_dir).unwrap();
        let value = snapshot(&mut f);
        let server = &value["server_sets"][0];
        assert_eq!(server["machines"].as_array().unwrap().len(), count);
        assert_eq!(server["sets"].as_array().unwrap().len(), 1);
        assert!(printed(&mut f).contains(server["sets"][0].as_str().unwrap()));
    }
}

#[test]
fn a_selected_set_and_replica_sets_follow_saved_ranks() {
    let mut f = planned(8);
    let group = f.app.config.groups.iter().find(|set| set.layout == "layout-4-2-2").unwrap().clone();
    f.app.config.select_group(&group.name).unwrap();
    let session = Session::new(&f.app.config, &fp4_options(&[]), &Manifest::load(&f.app.config.build_manifest).unwrap(), "0".repeat(64)).unwrap();
    session.save(&f.app.config.state_dir).unwrap();
    let value = snapshot(&mut f);
    assert!(value["server_sets"][0]["sets"].as_array().unwrap().contains(&Value::from(group.name.clone())));
    assert_eq!(value["server_sets"][0]["machines"].as_array().unwrap().len(), group.hosts.len());
    for group in &mut f.app.config.groups { group.hosts.reverse(); }
    let value = snapshot(&mut f);
    assert!(value["server_sets"][0]["sets"].as_array().unwrap().is_empty());
    assert!(printed(&mut f).contains("Server 1 set: unlisted"));

    let mut f = planned(8);
    let options = fp4_options(&["--replicas", "4", "--pairs"]);
    let session = Session::new(&f.app.config, &options, &Manifest::load(&f.app.config.build_manifest).unwrap(), "0".repeat(64)).unwrap();
    session.save(&f.app.config.state_dir).unwrap();
    let value = snapshot(&mut f);
    let servers = value["server_sets"].as_array().unwrap();
    assert_eq!(servers.len(), 4);
    for (index, server) in servers.iter().enumerate() {
        assert_eq!(server["machines"].as_array().unwrap().len(), 2);
        assert_eq!(server["machines"][0].as_str(), Some(session.ranks[index * 2].machine.name.as_str()));
        assert_eq!(server["local_port"].as_f64(), Some(f64::from(session.tunnel_port(index * 2))));
    }
    assert!(printed(&mut f).contains("Server 4 set:"));
}

#[test]
fn status_lists_group_servers_without_selecting_or_probing_them() {
    let mut f = planned(8);
    for group in f.app.config.groups.iter().filter(|set| set.layout == "layout-4-2-2") {
        let mut selected = f.app.config.clone();
        selected.select_group(&group.name).unwrap();
        let session = Session::new(&selected, &fp4_options(&[]), &Manifest::load(&selected.build_manifest).unwrap(), "0".repeat(64)).unwrap();
        session.save(&selected.state_dir).unwrap();
    }
    let value = snapshot(&mut f);
    let records = value["group_server_sets"].as_array().unwrap();
    assert_eq!(records.len(), 3);
    let text = printed(&mut f);
    for record in records {
        let name = record["group"].as_str().unwrap();
        assert!(text.contains(&format!("Group {name} saved sets")), "{text}");
        assert!(record["server_sets"][0]["sets"].as_array().unwrap().contains(&Value::from(name)));
    }
}
