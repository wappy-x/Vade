//! The cabling shape the config describes. Two machines are linked when each
//! names the other in a `fabric.<peer>` entry. From those links this module
//! finds rings, paths and the cable order of a ring. It reads no network.
use crate::config::Machine;

/// Whether both ends of the link between `a` and `b` are in the config.
pub fn linked(a: &Machine, b: &Machine) -> bool {
    a.fabric.contains_key(&b.name) && b.fabric.contains_key(&a.name)
}

/// The first neighbors in list order, including the last and the first, that
/// have no link. `None` means the list closes into a ring.
pub fn ring_gap(machines: &[Machine]) -> Option<(&Machine, &Machine)> {
    (0..machines.len())
        .map(|index| (&machines[index], &machines[(index + 1) % machines.len()]))
        .find(|(a, b)| a.name != b.name && !linked(a, b))
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
    extend(machines, &mut order, &mut used, &mut budget).then_some(order)
}

fn extend(machines: &[Machine], order: &mut Vec<usize>, used: &mut [bool], budget: &mut usize) -> bool {
    let last = *order.last().unwrap();
    if order.len() == machines.len() {
        return last == order[0] || linked(&machines[last], &machines[order[0]]);
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
        if extend(machines, order, used, budget) {
            return true;
        }
        order.pop();
        used[next] = false;
    }
    false
}

/// Refuse a list of machines that does not close into a ring in list order.
/// The error names the missing link and, when one exists, the ring order.
pub fn check_ring(label: &str, machines: &[Machine]) -> Result<(), String> {
    if let Some((a, b)) = ring_gap(machines) {
        let fix = match ring_order(machines) {
            Some(order) => format!(
                "These machines form a ring in another order. List them in cable order: hosts = {}.",
                order.iter().map(|&index| machines[index].name.as_str()).collect::<Vec<_>>().join(",")
            ),
            None => format!(
                "If the cable is in place, add fabric.{b} under [machine {a}] and fabric.{a} under [machine {b}]. Otherwise list the hosts in cable order.",
                a = a.name,
                b = b.name
            ),
        };
        return Err(format!("{label} is not a ring: {} and {} have no fabric link in the config. {fix}", a.name, b.name));
    }
    for (index, a) in machines.iter().enumerate() {
        let b = &machines[(index + 1) % machines.len()];
        if a.name != b.name && a.fabric.get(&b.name) == b.fabric.get(&a.name) {
            return Err(format!(
                "{label}: machines {} and {} need different addresses on their fabric link. Fix those fabric entries.",
                a.name, b.name
            ));
        }
    }
    Ok(())
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
            Machine { name: name.to_string(), ssh: format!("{name}-ssh"), verify_url: None, fabric, model_path: "~/m".into(),
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
        assert!(check_ring("Group left", &all[..4]).is_ok());
        assert!(check_ring("Group right", &all[4..]).is_ok());
        let error = check_ring("Group all", &all).unwrap_err();
        assert!(error.contains("s4 and s5 have no fabric link") && error.contains("add fabric.s5 under [machine s4]"), "{error}");
    }

    #[test]
    fn one_ring_of_eight_is_one_ring_with_paths_of_four() {
        let all = machines(&EIGHT, &ring(&EIGHT));
        assert_eq!(cabling(&all), ["ring of 8: s1, s2, s3, s4, s5, s6, s7, s8"]);
        assert!(check_ring("Group all", &all).is_ok());
        // Four neighbors of a ring of eight are a path: no closing cable.
        let error = check_ring("Group left", &all[..4]).unwrap_err();
        assert!(error.contains("s4 and s1 have no fabric link"), "{error}");
        assert_eq!(cabling(&all[..4]), ["path of 4: s1, s2, s3, s4"]);
    }

    #[test]
    fn a_ring_listed_out_of_cable_order_gets_the_cable_order() {
        let all = machines(&EIGHT[..4], &ring(&EIGHT[..4]));
        let listed = vec![all[0].clone(), all[2].clone(), all[1].clone(), all[3].clone()];
        let error = check_ring("Group left", &listed).unwrap_err();
        assert!(error.contains("s1 and s3 have no fabric link") && error.contains("hosts = s1,s2,s3,s4"), "{error}");
        let reversed: Vec<Machine> = all.iter().rev().cloned().collect();
        assert!(check_ring("Group left", &reversed).is_ok());
    }

    #[test]
    fn pairs_single_machines_and_odd_shapes() {
        let all = machines(&["a", "b", "c", "d", "e"], &[("a", "b"), ("c", "d"), ("c", "e"), ("d", "e")]);
        assert_eq!(cabling(&all), ["pair: a, b", "ring of 3: c, d, e"]);
        assert!(check_ring("Group pair", &all[..2]).is_ok());
        assert!(check_ring("Group one", &all[..1]).is_ok());
        let star = machines(&["h", "x", "y", "z", "w"], &[("h", "x"), ("h", "y"), ("h", "z")]);
        assert_eq!(cabling(&star), ["linked, neither a ring nor a path: h, x, y, z", "no link: w"]);
        assert!(ring_order(&star[..4]).is_none());
    }

    #[test]
    fn a_link_needs_two_different_addresses() {
        let mut all = machines(&["a", "b"], &[("a", "b")]);
        all[1].fabric.insert("a".into(), "a-to-b".into());
        assert!(check_ring("Group pair", &all).unwrap_err().contains("different addresses"));
    }

    #[test]
    fn a_dense_config_finds_a_ring_quickly() {
        let names: Vec<String> = (0..24).map(|index| format!("m{index}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut links = Vec::new();
        for a in 0..refs.len() { for b in a + 1..refs.len() { links.push((refs[a], refs[b])); } }
        let all = machines(&refs, &links);
        assert_eq!(ring_order(&all).unwrap(), (0..24).collect::<Vec<_>>());
    }
}
