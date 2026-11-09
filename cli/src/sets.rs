//! Host-only set plans and status labels. A set is one config group. A layout
//! lists sets that can run together. These records do not select an engine path.
use crate::{config::{Config, Group}, evals::json::{self, Value}, state::Session};

/// Make contiguous sets in cable order: one quad then pairs, quads, or all
/// machines together. Smaller inventories use the same partition rules.
pub(crate) fn plan(names: &[String]) -> Vec<Group> {
    let mut shapes = Vec::new();
    for (first, rest) in [(4, 2), (4, 4), (names.len().max(1), names.len().max(1))] {
        let mut sizes = Vec::new();
        let mut left = names.len();
        let mut size = first;
        while left > 0 {
            sizes.push(left.min(size));
            left -= left.min(size);
            size = rest;
        }
        if !sizes.is_empty() && !shapes.contains(&sizes) { shapes.push(sizes); }
    }
    let mut groups = Vec::new();
    for sizes in shapes {
        let shape = sizes.iter().map(usize::to_string).collect::<Vec<_>>().join("-");
        let mut offset = 0;
        for (index, size) in sizes.into_iter().enumerate() {
            groups.push(Group {
                name: format!("set-{shape}-{}", index + 1),
                hosts: names[offset..offset + size].to_vec(),
                local_port: 18080 + u16::try_from(offset + index).expect("setup inventory fits local ports"),
                layout: format!("layout-{shape}"),
            });
            offset += size;
        }
    }
    groups
}

/// Write the groups that setup adds to its controller inventory.
pub(crate) fn sections(names: &[String]) -> String {
    let mut out = String::new();
    for group in plan(names) {
        out += &format!("\n[group {}]\nhosts = {}\nlocal_port = {}\nlayout = {}\n",
            group.name, group.hosts.join(","), group.local_port, group.layout);
    }
    out
}

fn layouts(config: &Config) -> Vec<(&str, Vec<&Group>)> {
    let mut layouts: Vec<(&str, Vec<&Group>)> = Vec::new();
    for group in &config.groups {
        match layouts.iter_mut().find(|(name, _)| *name == group.layout) {
            Some((_, sets)) => sets.push(group),
            None => layouts.push((&group.layout, vec![group])),
        }
    }
    layouts
}

fn layout_name(name: &str) -> &str { if name.is_empty() { "default" } else { name } }
fn shape(sets: &[&Group]) -> String { sets.iter().map(|set| set.hosts.len().to_string()).collect::<Vec<_>>().join(" + ") }
fn strings(names: &[String]) -> Value { Value::Array(names.iter().map(|name| name.as_str().into()).collect()) }

/// The inventory's layouts, with each set's rank order and local port.
pub(crate) fn inventory_json(config: &Config) -> Value {
    Value::Array(layouts(config).into_iter().map(|(name, sets)| json::object(vec![
        ("name", layout_name(name).into()),
        ("shape", shape(&sets).into()),
        ("sets", Value::Array(sets.into_iter().map(|set| json::object(vec![
            ("name", set.name.as_str().into()), ("machines", strings(&set.hosts)),
            ("local_port", Value::Int(set.local_port.into())),
        ])).collect())),
    ])).collect())
}

/// The same inventory for setup and plain status. It needs no device probe.
pub(crate) fn inventory_text(config: &Config) -> String {
    let mut out = String::new();
    for cable in crate::topology::cabling(&config.machines) { out += &format!("  Cabling: {cable}\n"); }
    for (name, sets) in layouts(config) {
        out += &format!("  Layout {}: {}\n", layout_name(name), shape(&sets));
        for set in sets {
            out += &format!("    Set {}: {} (local port {})\n", set.name, set.hosts.join(", "), set.local_port);
        }
    }
    if !config.groups.is_empty() { out += "  Use VADE_GROUP=NAME with up, status, chat, logs and down to select a set.\n"; }
    out
}

/// Set names must match the saved rank order. A changed config must not
/// relabel an old server with the newly selected group's machines.
fn matches(config: &Config, names: &[String]) -> Vec<String> {
    config.groups.iter().filter(|set| set.hosts == names).map(|set| set.name.clone()).collect()
}

/// Each server's actual set, including every independent replica.
pub(crate) fn servers_json(config: &Config, session: &Session) -> Value {
    Value::Array(session.ranks.chunks(session.group_len().max(1)).enumerate().map(|(index, ranks)| {
        let names: Vec<String> = ranks.iter().map(|rank| rank.machine.name.clone()).collect();
        json::object(vec![
            ("server", (index + 1).into()), ("sets", strings(&matches(config, &names))),
            ("machines", strings(&names)), ("local_port", Value::Int(session.tunnel_port(index * session.group_len().max(1)).into())),
        ])
    }).collect())
}

pub(crate) fn servers_text(config: &Config, session: &Session) -> String {
    let mut out = String::new();
    for (index, ranks) in session.ranks.chunks(session.group_len().max(1)).enumerate() {
        let names: Vec<String> = ranks.iter().map(|rank| rank.machine.name.clone()).collect();
        let sets = matches(config, &names);
        let label = if sets.is_empty() { "unlisted".into() } else { sets.join(", ") };
        out += &format!("  Server {} set: {label} ({})\n", index + 1, names.join(", "));
    }
    out
}

/// Read group ownership records without a live check. Keep an unreadable
/// record visible, including a group that has left the config.
pub(crate) fn group_servers(config: &Config, owned: &[String]) -> (String, Value) {
    let mut out = String::new();
    let mut records = Vec::new();
    for name in owned {
        match Session::load(&config.group_state_dir(name)) {
            Ok(Some(session)) => {
                out += &format!("  Group {name} saved sets ({}; health not checked):\n{}", session.phase, servers_text(config, &session));
                records.push(json::object(vec![
                    ("group", name.as_str().into()), ("phase", session.phase.as_str().into()),
                    ("server_sets", servers_json(config, &session)), ("error", Value::Null),
                ]));
            }
            result => {
                let error = result.err().unwrap_or_else(|| "The ownership record is no longer present. Run vade status again.".into());
                out += &format!("  Group {name} saved sets: cannot read the ownership record. Use VADE_GROUP={name} vade status.\n");
                records.push(json::object(vec![("group", name.as_str().into()), ("server_sets", Value::Null), ("error", error.into())]));
            }
        }
    }
    (out, Value::Array(records))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn fake_inventories_keep_every_machine_once_per_layout() {
        for (count, expected) in [(1, vec!["1"]), (2, vec!["2"]), (4, vec!["4"]),
            (6, vec!["4 + 2", "6"]), (8, vec!["4 + 2 + 2", "4 + 4", "8"])] {
            let names: Vec<String> = (0..count).map(|index| format!("m{index}")).collect();
            let mut text = String::from("[global]\n");
            for name in &names { text += &format!("[machine {name}]\nssh = {name}\n"); }
            text += &sections(&names);
            let config = Config::parse(&text, Path::new("/fixture/config")).unwrap();
            let value = inventory_json(&config);
            let actual: Vec<&str> = value.as_array().unwrap().iter().map(|layout| layout["shape"].as_str().unwrap()).collect();
            assert_eq!(actual, expected, "inventory of {count}");
            for (_, sets) in layouts(&config) {
                assert_eq!(sets.iter().flat_map(|set| set.hosts.clone()).collect::<Vec<_>>(), names);
            }
            let printed = inventory_text(&config);
            for shape in expected { assert!(printed.contains(&format!(": {shape}\n")), "{printed}"); }
        }
        assert!(plan(&[]).is_empty());
    }
}
