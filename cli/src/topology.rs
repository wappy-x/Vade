//! The cabling shape the config describes. Two machines are linked when each
//! names the other in a `fabric.<peer>` entry. From those links this module
//! finds rings, paths, arcs (machines that follow the cables, each linked to
//! the next) and their cable order. It reads no network.
use crate::config::Machine;

/// Whether both ends of the link between `a` and `b` are in the config.
pub fn linked(a: &Machine, b: &Machine) -> bool {
    a.fabric.contains_key(&b.name) && b.fabric.contains_key(&a.name)
}

/// A cable order of `machines` that closes into a ring, as indexes into the
/// list. It starts at the first machine and tries the listed order first.
/// `None` means no ring uses every machine once.
pub fn ring_order(machines: &[Machine]) -> Option<Vec<usize>> {
    if machines.is_empty() {
        return None;
    }
    let mut order = vec![0];
    let mut used = vec![false; machines.len()];
    used[0] = true;
    // A search budget keeps a dense or odd config from a long search.
    let mut budget = 100_000usize;
    extend(machines, &mut order, &mut used, &mut budget, true).then_some(order)
}

/// A cable order of `machines` in which each machine is linked to the next,
/// as indexes into the list: an arc of the cabling. It tries the first
/// machine and the listed order first. `None` means no arc uses every machine
/// once.
pub fn arc_order(machines: &[Machine]) -> Option<Vec<usize>> {
    let mut budget = 100_000usize;
    (0..machines.len()).find_map(|start| {
        let mut order = vec![start];
        let mut used = vec![false; machines.len()];
        used[start] = true;
        extend(machines, &mut order, &mut used, &mut budget, false).then_some(order)
    })
}

fn extend(machines: &[Machine], order: &mut Vec<usize>, used: &mut [bool], budget: &mut usize, close: bool) -> bool {
    let last = *order.last().unwrap();
    if order.len() == machines.len() {
        return !close || last == order[0] || linked(&machines[last], &machines[order[0]]);
    }
    for step in 1..machines.len() {
        let next = (last + step) % machines.len();
        if used[next] || !linked(&machines[last], &machines[next]) {
            continue;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        used[next] = true;
        order.push(next);
        if extend(machines, order, used, budget, close) {
            return true;
        }
        order.pop();
        used[next] = false;
    }
    false
}

/// Refuse a list of machines that is not an arc of the cabling in list
/// order: each machine must be linked to the next. A link from the last back
/// to the first is allowed, not needed, so any run of neighbors on a ring is
/// an arc. The error names the missing link and, when one exists, an order
/// that follows the cables.
pub fn check_arc(label: &str, machines: &[Machine]) -> Result<(), String> {
    if let Some(pair) = machines.windows(2).find(|pair| !linked(&pair[0], &pair[1])) {
        let (a, b) = (&pair[0], &pair[1]);
        let fix = match arc_order(machines) {
            Some(order) => format!(
                "These machines follow the cables in another order. List them in cable order: hosts = {}.",
                order.iter().map(|&index| machines[index].name.as_str()).collect::<Vec<_>>().join(",")
            ),
            None => format!(
                "If the cable is in place, add fabric.{b} under [machine {a}] and fabric.{a} under [machine {b}]. Otherwise choose machines that are neighbors on the cables.",
                a = a.name,
                b = b.name
            ),
        };
        return Err(format!("{label} is not an arc of the cabling: {} and {} have no fabric link in the config. {fix}", a.name, b.name));
    }
    if let Some(pair) = machines.windows(2).find(|pair| pair[0].fabric.get(&pair[1].name) == pair[1].fabric.get(&pair[0].name)) {
        return Err(format!(
            "{label}: machines {} and {} need different addresses on their fabric link. Fix those fabric entries.",
            pair[0].name, pair[1].name
        ));
    }
    for pair in machines.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        match (a.fabric_aux.get(&b.name), b.fabric_aux.get(&a.name)) {
            (None, None) => (),
            (Some(aux_a), Some(aux_b)) if aux_a != aux_b && Some(aux_a) != b.fabric.get(&a.name) && Some(aux_b) != a.fabric.get(&b.name) => (),
            (Some(_), Some(_)) => return Err(format!("{label}: machines {} and {} need different addresses on their fabric_aux link.", a.name, b.name)),
            _ => return Err(format!("{label}: add both fabric_aux.{} under {} and fabric_aux.{} under {}, or remove it.", b.name, a.name, a.name, b.name)),
        }
    }
    Ok(())
}

/// The shape of an arc that passed [`check_arc`], for a report row.
pub fn arc_text(machines: &[Machine]) -> String {
    let names = machines.iter().map(|machine| machine.name.as_str()).collect::<Vec<_>>().join(", ");
    match machines {
        [] => "no machines".into(),
        [one] => format!("one Spark, no cable: {}", one.name),
        [first, .., last] if machines.len() > 2 && linked(last, first) => format!("arc of {} in rank order: {names}; the ring closes", machines.len()),
        _ => format!("arc of {} in rank order: {names}", machines.len()),
    }
}

/// The `--ring` expectation for the whole config: every machine has exactly
/// two linked neighbors, and the links close into one ring that holds every
/// machine. On success, the ring in cable order.
pub fn check_whole_ring(machines: &[Machine]) -> Result<String, String> {
    let parts = cabling(machines);
    let neighbors = |machine: &Machine| machines.iter().filter(|other| other.name != machine.name && linked(machine, other)).map(|other| other.name.as_str()).collect::<Vec<_>>();
    let odd: Vec<(&Machine, Vec<&str>)> = machines.iter().map(|machine| (machine, neighbors(machine))).filter(|(_, near)| near.len() != 2).collect();
    if odd.is_empty() && parts.len() == 1 {
        return Ok(format!("{}; every Spark has two neighbors", parts[0]));
    }
    if odd.is_empty() {
        return Err(format!("The cabling is {} rings, not one: {}. Cable every Spark into one ring, then fix the fabric entries.", parts.len(), parts.join("; ")));
    }
    let ends: Vec<&str> = odd.iter().filter(|(_, near)| near.len() == 1).map(|(machine, _)| machine.name.as_str()).collect();
    if parts.len() == 1 && odd.len() == ends.len() && parts[0].starts_with("path of") {
        let (a, b) = (ends[0], ends[1]);
        return Err(format!(
            "The ring does not close: {}. Add the cable between {a} and {b}, then add fabric.{b} under [machine {a}] and fabric.{a} under [machine {b}].",
            parts[0]
        ));
    }
    let rows: Vec<String> = odd.iter().map(|(machine, near)| match near.as_slice() {
        [] => format!("{} has none", machine.name),
        _ => format!("{} has {} ({})", machine.name, near.len(), near.join(", ")),
    }).collect();
    Err(format!("A ring needs exactly two neighbors per Spark in the config: {}. Cabling: {}.", rows.join("; "), parts.join("; ")))
}

/// One line for each connected part of the configured cabling, in config
/// order: a ring, a path, a pair, one machine with no link, or another shape.
pub fn cabling(machines: &[Machine]) -> Vec<String> {
    let mut seen = vec![false; machines.len()];
    let mut lines = Vec::new();
    for start in 0..machines.len() {
        if seen[start] {
            continue;
        }
        let mut part = vec![start];
        seen[start] = true;
        let mut next = 0;
        while next < part.len() {
            let current = part[next];
            for (index, machine) in machines.iter().enumerate() {
                if !seen[index] && linked(&machines[current], machine) {
                    seen[index] = true;
                    part.push(index);
                }
            }
            next += 1;
        }
        let members: Vec<Machine> = part.iter().map(|&index| machines[index].clone()).collect();
        let degree = |machine: &Machine| members.iter().filter(|other| linked(machine, other)).count();
        let names = |order: &[usize]| order.iter().map(|&index| members[index].name.as_str()).collect::<Vec<_>>().join(", ");
        let ends: Vec<usize> = (0..members.len()).filter(|&index| degree(&members[index]) == 1).collect();
        let inner = members.iter().all(|machine| degree(machine) <= 2);
        lines.push(if members.len() == 1 {
            format!("no link: {}", members[0].name)
        } else if inner && ends.is_empty() && ring_order(&members).is_some() {
            format!("ring of {}: {}", members.len(), names(&ring_order(&members).unwrap()))
        } else if inner && ends.len() == 2 {
            let mut order = vec![ends[0]];
            while let Some(index) = (0..members.len()).find(|index| !order.contains(index) && linked(&members[*order.last().unwrap()], &members[*index])) {
                order.push(index);
            }
            if members.len() == 2 { format!("pair: {}", names(&order)) } else { format!("path of {}: {}", members.len(), names(&order)) }
        } else {
            format!("linked, neither a ring nor a path: {}", names(&(0..members.len()).collect::<Vec<_>>()))
        });
    }
    lines
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Machines named by `names`, linked by `links` given as name pairs.
    fn machines(names: &[&str], links: &[(&str, &str)]) -> Vec<Machine> {
        names.iter().map(|name| {
            let mut fabric = BTreeMap::new();
            for (a, b) in links {
                if a == name { fabric.insert(b.to_string(), format!("{a}-to-{b}")); }
                if b == name { fabric.insert(a.to_string(), format!("{b}-to-{a}")); }
            }
            Machine { name: name.to_string(), ssh: format!("{name}-ssh"), verify_url: None, fabric, fabric_aux: BTreeMap::new(), model_path: "~/m".into(),
                engine_source: "~/e".into(), cli_source: "~/c".into(), stage_dir: "~/s".into(), state_dir: "~/t".into() }
        }).collect()
    }
    const EIGHT: [&str; 8] = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
    fn ring(names: &[&'static str]) -> Vec<(&'static str, &'static str)> {
        (0..names.len()).map(|index| (names[index], names[(index + 1) % names.len()])).collect()
    }

    #[test]
    fn two_rings_of_four_are_two_rings() {
        let links = [ring(&EIGHT[..4]), ring(&EIGHT[4..])].concat();
        let all = machines(&EIGHT, &links);
        assert_eq!(cabling(&all), ["ring of 4: s1, s2, s3, s4", "ring of 4: s5, s6, s7, s8"]);
        assert!(check_arc("Group left", &all[..4]).is_ok());
        assert_eq!(arc_text(&all[..4]), "arc of 4 in rank order: s1, s2, s3, s4; the ring closes");
        assert!(check_arc("Group right", &all[4..]).is_ok());
        let error = check_arc("Group all", &all).unwrap_err();
        assert!(error.contains("s4 and s5 have no fabric link") && error.contains("add fabric.s5 under [machine s4]"), "{error}");
        let error = check_whole_ring(&all).unwrap_err();
        assert!(error.starts_with("The cabling is 2 rings, not one: ring of 4: s1, s2, s3, s4; ring of 4: s5"), "{error}");
    }

    #[test]
    fn every_arc_of_a_ring_of_eight_passes_and_the_ring_closes() {
        let all = machines(&EIGHT, &ring(&EIGHT));
        assert_eq!(cabling(&all), ["ring of 8: s1, s2, s3, s4, s5, s6, s7, s8"]);
        assert_eq!(check_whole_ring(&all).unwrap(), "ring of 8: s1, s2, s3, s4, s5, s6, s7, s8; every Spark has two neighbors");
        // Every run of 2, 4 or 8 neighbors, from every start and in both directions, is an arc.
        for size in [2, 4, 8] {
            for start in 0..8 {
                let arc: Vec<Machine> = (0..size).map(|step| all[(start + step) % 8].clone()).collect();
                assert!(check_arc("Group arc", &arc).is_ok(), "{size} from s{}", start + 1);
                let back: Vec<Machine> = arc.iter().rev().cloned().collect();
                assert!(check_arc("Group arc", &back).is_ok(), "{size} back from s{}", start + 1);
            }
        }
        // An arc of four crosses cable 8 (s8 - s1) as well as any other cable.
        let crossing = vec![all[6].clone(), all[7].clone(), all[0].clone(), all[1].clone()];
        assert_eq!(arc_text(&crossing), "arc of 4 in rank order: s7, s8, s1, s2");
        assert_eq!(arc_text(&all), "arc of 8 in rank order: s1, s2, s3, s4, s5, s6, s7, s8; the ring closes");
        assert_eq!(cabling(&all[..4]), ["path of 4: s1, s2, s3, s4"]);
        // Four Sparks with a gap are not an arc.
        let gap = vec![all[0].clone(), all[1].clone(), all[3].clone(), all[4].clone()];
        let error = check_arc("Group gap", &gap).unwrap_err();
        assert!(error.contains("Group gap is not an arc of the cabling: s2 and s4") && error.contains("choose machines that are neighbors"), "{error}");
    }

    #[test]
    fn a_ring_with_a_missing_or_extra_cable_names_it() {
        // Cable 8 (s8 - s1) missing: a path of eight.
        let open = machines(&EIGHT, &ring(&EIGHT)[..7]);
        let error = check_whole_ring(&open).unwrap_err();
        assert!(error.starts_with("The ring does not close: path of 8: s1, s2,") && error.contains("Add the cable between s1 and s8"), "{error}");
        // A ninth link across the ring gives two Sparks three neighbors.
        let extra = machines(&EIGHT, &[ring(&EIGHT), vec![("s1", "s5")]].concat());
        let error = check_whole_ring(&extra).unwrap_err();
        assert!(error.contains("s1 has 3 (s2, s5, s8); s5 has 3 (s1, s4, s6)"), "{error}");
        // A Spark with no cable in the config.
        let lone = machines(&EIGHT, &ring(&EIGHT[..7]));
        let error = check_whole_ring(&lone).unwrap_err();
        assert!(error.contains("s8 has none") && error.contains("no link: s8"), "{error}");
        // The six-Spark ring passes the same check.
        assert!(check_whole_ring(&machines(&EIGHT[..6], &ring(&EIGHT[..6]))).unwrap().starts_with("ring of 6: s1"));
    }

    #[test]
    fn a_group_listed_out_of_cable_order_gets_the_cable_order() {
        let all = machines(&EIGHT[..4], &ring(&EIGHT[..4]));
        let listed = vec![all[0].clone(), all[2].clone(), all[1].clone(), all[3].clone()];
        let error = check_arc("Group left", &listed).unwrap_err();
        assert!(error.contains("s1 and s3 have no fabric link") && error.contains("hosts = s1,s2,s3,s4"), "{error}");
        let reversed: Vec<Machine> = all.iter().rev().cloned().collect();
        assert!(check_arc("Group left", &reversed).is_ok());
        // On a ring of eight, the arc order may start at another machine.
        let eight = machines(&EIGHT, &ring(&EIGHT));
        let listed = vec![eight[1].clone(), eight[3].clone(), eight[2].clone()];
        assert!(check_arc("Group three", &listed).unwrap_err().contains("hosts = s2,s3,s4"));
        let listed = vec![eight[2].clone(), eight[0].clone(), eight[1].clone()];
        assert!(check_arc("Group three", &listed).unwrap_err().contains("hosts = s3,s2,s1"));
    }

    #[test]
    fn pairs_single_machines_and_odd_shapes() {
        let all = machines(&["a", "b", "c", "d", "e"], &[("a", "b"), ("c", "d"), ("c", "e"), ("d", "e")]);
        assert_eq!(cabling(&all), ["pair: a, b", "ring of 3: c, d, e"]);
        assert!(check_arc("Group pair", &all[..2]).is_ok());
        assert_eq!(arc_text(&all[..2]), "arc of 2 in rank order: a, b");
        assert!(check_arc("Group one", &all[..1]).is_ok());
        assert_eq!(arc_text(&all[..1]), "one Spark, no cable: a");
        assert!(check_whole_ring(&all[..2]).unwrap_err().contains("a has 1 (b); b has 1 (a)"));
        let star = machines(&["h", "x", "y", "z", "w"], &[("h", "x"), ("h", "y"), ("h", "z")]);
        assert_eq!(cabling(&star), ["linked, neither a ring nor a path: h, x, y, z", "no link: w"]);
        assert!(ring_order(&star[..4]).is_none());
        assert!(arc_order(&star[..4]).is_none());
    }

    #[test]
    fn a_link_needs_two_different_addresses() {
        let mut all = machines(&["a", "b"], &[("a", "b")]);
        all[1].fabric.insert("a".into(), "a-to-b".into());
        assert!(check_arc("Group pair", &all).unwrap_err().contains("different addresses"));
    }

    #[test]
    fn a_dense_config_finds_a_ring_quickly() {
        let names: Vec<String> = (0..24).map(|index| format!("m{index}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut links = Vec::new();
        for a in 0..refs.len() { for b in a + 1..refs.len() { links.push((refs[a], refs[b])); } }
        let all = machines(&refs, &links);
        assert_eq!(ring_order(&all).unwrap(), (0..24).collect::<Vec<_>>());
        assert_eq!(arc_order(&all).unwrap(), (0..24).collect::<Vec<_>>());
    }
}
