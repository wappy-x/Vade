//! `repo-digest-v1`: a synthetic repository digest of 241,865 bytes (aimed at about 60K
//! tokens; the live card gives the measured prompt tokens), the same bytes on every run. Five facts sit at fixed places: near
//! the start, in the middle, near the end, and two counts over the whole file.
//! The answers live in the fixture (`expected/ANSWERS.txt`); `answers()` gives
//! the same lines so the runner can prove the fixture still matches.

const FILES: usize = 87;
const FUNCS_PER_FILE: usize = 7;
const DEPRECATED: usize = 23;
const PORT_FILE: usize = 4;
const LEDGER_FILE: usize = 43;
const RETRY_FILE: usize = 83;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick<'a>(&mut self, list: &[&'a str]) -> &'a str { list[(self.next() as usize) % list.len()] }
}

const CRATES: &[&str] = &["core", "store", "query", "sched", "codec", "audit", "cache", "index"];
const VERBS: &[&str] = &["merge", "split", "load", "flush", "scan", "probe", "align", "shift", "pack", "trim", "fold", "rank"];
const NOUNS: &[&str] = &["cursor", "segment", "bucket", "frame", "record", "window", "column", "token", "slot", "page", "range", "tile"];
const TYPES: &[&str] = &["u64", "usize", "u32"];

fn file_name(i: usize, rng: &mut Rng) -> String {
    match i {
        PORT_FILE => "crates/net/src/listener.rs".into(),
        LEDGER_FILE => "crates/ledger/src/reconcile.rs".into(),
        RETRY_FILE => "crates/sync/src/backoff.rs".into(),
        _ => format!("crates/{}/src/{}_{}_{i:03}.rs", rng.pick(CRATES), rng.pick(VERBS), rng.pick(NOUNS)),
    }
}

pub fn generate() -> String {
    let mut rng = Rng(0x7469_6465_7761_7465);
    // Spread the deprecated marks over fixed function slots, never in the fact files.
    let total = FILES * FUNCS_PER_FILE;
    let mut marked = vec![false; total];
    let mut placed = 0;
    while placed < DEPRECATED {
        let slot = (rng.next() as usize) % total;
        let file = slot / FUNCS_PER_FILE;
        if marked[slot] || matches!(file, PORT_FILE | LEDGER_FILE | RETRY_FILE) { continue; }
        marked[slot] = true;
        placed += 1;
    }
    let mut out = String::new();
    out.push_str("# Repository digest: tidewater\n\n");
    out.push_str("This digest holds every source file of the tidewater repository, one section per file.\n");
    out.push_str("Each section starts with a `## File:` line that gives the path from the repository root.\n\n");
    for f in 0..FILES {
        let name = file_name(f, &mut rng);
        out.push_str(&format!("## File: {name}\n\n```rust\n"));
        out.push_str(&format!("//! Module {f}: generated helpers for the tidewater pipeline.\n\n"));
        if f == PORT_FILE { out.push_str("/// The port the harbor listener binds when no setting is given.\npub const HARBOR_PORT: u16 = 47113;\n\n"); }
        if f == RETRY_FILE { out.push_str("/// How many times a sync job retries before it gives up.\npub const RETRY_LIMIT: u32 = 19;\n\n"); }
        for g in 0..FUNCS_PER_FILE {
            let slot = f * FUNCS_PER_FILE + g;
            if f == LEDGER_FILE && g == FUNCS_PER_FILE / 2 {
                out.push_str("/// Brings the ledger back in line with the settled batches.\npub fn reconcile_ledger(entries: &[u64]) -> u64 {\n    let total: u64 = entries.iter().sum();\n    settle_batch_v2(total)\n}\n\n");
            }
            let (verb, noun, ty) = (rng.pick(VERBS), rng.pick(NOUNS), rng.pick(TYPES));
            let fname = format!("{verb}_{noun}_{slot}");
            let (a, b, c) = (rng.next() % 997, rng.next() % 61 + 1, rng.next() % 13 + 2);
            if marked[slot] { out.push_str("#[deprecated(note = \"use the batch form\")]\n"); }
            out.push_str(&format!("/// {} the {noun} values and return the {verb}ed result.\n", capital(verb)));
            out.push_str(&format!("pub fn {fname}(input: &[{ty}], seed: {ty}) -> {ty} {{\n"));
            out.push_str(&format!("    let mut acc: {ty} = seed.wrapping_add({a});\n"));
            out.push_str(&format!("    for (i, v) in input.iter().enumerate() {{\n        if i % {c} == 0 {{\n            acc = acc.wrapping_mul({b}).wrapping_add(*v);\n        }} else {{\n            acc ^= v.rotate_left({});\n        }}\n    }}\n", c + 1));
            out.push_str(&format!("    acc / {b}\n}}\n\n"));
        }
        out.push_str("```\n\n");
    }
    out
}

pub fn answers() -> String {
    format!("1: crates/net/src/listener.rs 47113\n2: settle_batch_v2\n3: 19\n4: {DEPRECATED}\n5: {FILES}\n")
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn digest_is_fixed_and_counts_hold() {
        let d = generate();
        assert_eq!(d, generate());
        assert_eq!(d.matches("#[deprecated").count(), DEPRECATED);
        assert_eq!(d.matches("\n## File: ").count(), FILES);
        assert_eq!(d.matches("HARBOR_PORT").count(), 1);
        assert_eq!(d.matches("reconcile_ledger").count(), 1);
        assert_eq!(d.matches("settle_batch_v2").count(), 1);
        assert_eq!(d.len(), 241_865, "the fixture README and task title name this size");
    }
}
