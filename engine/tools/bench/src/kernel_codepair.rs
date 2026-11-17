//! CODE-PAIR: the machine code of two kernel images, kernel by kernel.
//! Run: kernel-bench --mx-tc <old ptx> <new ptx> code.
//!
//! Each image is compiled by the box's `ptxas` with the arguments of `scripts/make-cubin.sh` (`--gpu-name=<the image's
//! own target line> -m64 -O3`). Then the code section of every kernel of the new image is compared byte for byte with the
//! old image's: EQUAL, DIFFER, or NEW (GONE: a kernel only the old image has). The other sections are counted by class.
//! Measure only: nothing is loaded, nothing is launched.
//!
//! Use: the receipt of a target, compiler or source change. The kernels listed DIFFER or NEW are the ones a device
//! proof must cover again; a kernel listed EQUAL runs the same machine code as before when the image is precompiled.
//! (An image loaded as text is compiled by the driver, whose compiler this tool cannot run.)
//! An entry the offline compiler refuses for its static shared data is taken out of that image's text and listed as
//! NOT COMPARED.
#![forbid(unsafe_code)]
use super::{Result, sha256};
use std::{collections::BTreeMap, path::Path, process::Command, time::Instant};

// The one compiler the token proof covers. Keep equal to `crates/gpu/cubin.rs` and `scripts/make-cubin.sh`.
const PINNED_PTXAS_VERSION: &str = "13.0.88";
const PINNED_PTXAS_SHA256: &str = "a1941a04ca4fd233b2fbe50c625b1e72b3d5f79ebe80209a272c85482dfbb487";

/// One section of an image. A section without bytes in the file (type 8, "no bits": the static shared arrays) has
/// `bytes` empty and its extent in `size`; `same` compares type, flags, alignment and size as well as the bytes.
struct Section { name: String, kind: u32, flags: u64, align: u64, size: u64, bytes: Vec<u8> }
impl Section {
    fn same(&self, other: &Section) -> bool { self.kind == other.kind && self.flags == other.flags && self.align == other.align && self.size == other.size && self.bytes == other.bytes }
}
/// Every section of an ELF64 little-endian image.
fn sections(image: &[u8]) -> Result<Vec<Section>> {
    let u16_at = |at: usize| -> Result<usize> { image.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).ok_or_else(|| "short ELF image".to_string()) };
    let u32_at = |at: usize| -> Result<u32> { image.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| "short ELF image".to_string()) };
    let u64_at = |at: usize| -> Result<u64> { image.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| "short ELF image".to_string()) };
    if image.len() < 64 || &image[..6] != b"\x7fELF\x02\x01" { return Err("not an ELF64 little-endian image".into()); }
    let (table, size, mut count, mut names) = (u64_at(0x28)? as usize, u16_at(0x3a)?, u16_at(0x3c)?, u16_at(0x3e)?);
    if size != 64 { return Err("ELF section header size differs".into()); }
    // More sections than the header field holds: the first header carries the count and the name table's index.
    if count == 0 { count = u64_at(table + 0x20)? as usize; }
    if names == 0xffff { names = u32_at(table + 0x28)? as usize; }
    let (strings_at, strings_len) = (u64_at(table + names * 64 + 0x18)? as usize, u64_at(table + names * 64 + 0x20)? as usize);
    let strings = image.get(strings_at..strings_at + strings_len).ok_or("ELF name table outside the image")?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = table + i * 64;
        let (name, kind, flags, offset, size, align) = (u32_at(at)? as usize, u32_at(at + 4)?, u64_at(at + 8)?, u64_at(at + 0x18)? as usize, u64_at(at + 0x20)?, u64_at(at + 0x30)?);
        let start = name.min(strings.len());
        let end = strings[start..].iter().position(|&b| b == 0).unwrap_or(0);
        let name = String::from_utf8_lossy(&strings[start..start + end]).into_owned();
        // Type 8 (no bits) has no bytes in the file: its extent is `size`.
        let bytes = if kind == 8 { vec![] } else { image.get(offset..offset + size as usize).ok_or("ELF section outside the image")?.to_vec() };
        out.push(Section { name, kind, flags, align, size, bytes });
    }
    Ok(out)
}
/// The names a PTX text gives: (its entries, its other functions, declared or defined).
fn text_names(text: &str) -> (Vec<String>, Vec<String>) {
    let (mut entries, mut functions) = (vec![], vec![]);
    for line in text.lines() {
        let line = line.trim_start_matches(".visible ").trim_start_matches(".weak ").trim_start_matches(".extern ");
        if let Some(rest) = line.strip_prefix(".entry ") {
            entries.push(rest.split(|c: char| c == '(' || c.is_whitespace()).next().unwrap_or("").to_string());
        } else if let Some(rest) = line.strip_prefix(".func ") {
            // `.func NAME(` or `.func (.param ... retval) NAME(`.
            let rest = rest.trim_start();
            let rest = if rest.starts_with('(') { rest.split_once(')').map_or("", |(_, tail)| tail).trim_start() } else { rest };
            let name = rest.split(|c: char| c == '(' || c == ';' || c.is_whitespace()).next().unwrap_or("");
            if !name.is_empty() { functions.push(name.to_string()); }
        }
    }
    (entries, functions)
}
/// The code sections of an image against the names of the text it was compiled from: every entry has exactly one code
/// section, no code section is named twice, and every code section is an entry or a function of the text.
fn names_check(label: &str, text: &str, image: &[Section]) -> bool {
    let (entries, functions) = text_names(text);
    let code: Vec<&str> = image.iter().filter(|s| class(&s.name) == "code").map(|s| &s.name[6..]).collect();
    let count = |name: &str| code.iter().filter(|c| **c == name).count();
    let missing: Vec<&String> = entries.iter().filter(|e| count(e) == 0).collect();
    let mut duplicate: Vec<&str> = code.iter().copied().filter(|c| count(c) > 1).collect();
    duplicate.sort_unstable(); duplicate.dedup();
    let mut twice: Vec<&String> = entries.iter().filter(|e| entries.iter().filter(|o| o == e).count() > 1).collect();
    twice.sort_unstable(); twice.dedup();
    let extra: Vec<&str> = code.iter().copied().filter(|c| !entries.iter().any(|e| e == c) && !functions.iter().any(|f| f == c)).collect();
    let ok = missing.is_empty() && duplicate.is_empty() && twice.is_empty() && extra.is_empty();
    println!("CODE PAIR NAMES {} {label}: entries_in_text={} code_sections={} of_them_entries={} entries_without_code={missing:?} code_sections_named_twice={duplicate:?} entries_named_twice={twice:?} code_sections_not_in_text={extra:?}",
        if ok { "PASS" } else { "FAIL" }, entries.len(), code.len(), code.iter().filter(|c| entries.iter().any(|e| e == *c)).count());
    ok
}
/// The PTX text without the entry `name` (from its `.visible .entry` line to the closing brace of its body).
fn strip_entry(text: &str, name: &str) -> Option<String> {
    let start = text.find(&format!("\n.visible .entry {name}("))? + 1;
    let end = start + text[start..].find("\n}\n")? + 3;
    Some(format!("{}{}", &text[..start], &text[end..]))
}
/// One ptxas run. Ok(seconds), or Err(the entry it refuses for its shared data), or a hard error.
fn ptxas_once(ptxas: &str, gpu: &str, source: &Path, out: &Path) -> Result<std::result::Result<f64, String>> {
    let t0 = Instant::now();
    let run = Command::new(ptxas).arg(format!("--gpu-name={gpu}")).args(["-m64", "-O3", "-o"]).arg(out).arg(source).output().map_err(|e| e.to_string())?;
    let seconds = t0.elapsed().as_secs_f64();
    if run.status.success() { return Ok(Ok(seconds)); }
    let message = String::from_utf8_lossy(&run.stderr).into_owned();
    let first = message.lines().next().unwrap_or("").to_string();
    println!("CODE PAIR compile --gpu-name={gpu}: exit={:?} seconds={seconds:.1} message=\"{first}\"", run.status.code());
    if let Some(at) = first.find("Entry function '") {
        let rest = &first[at + 16..];
        if let Some(end) = rest.find('\'') && first.contains("too much shared data") { return Ok(Err(rest[..end].to_string())); }
    }
    Err(format!("ptxas failed for {gpu}: {}", message.lines().take(6).collect::<Vec<_>>().join(" | ")))
}
fn class(name: &str) -> &'static str {
    if name.starts_with(".text.") { "code" } else if name.starts_with(".nv.info") { "info" } else if name.starts_with(".nv.constant") { "constants" }
    else if name.starts_with(".nv.shared") { "shared" } else if name.starts_with(".rel") { "relocations" } else { "other" }
}
struct Image { sections: Vec<Section>, removed: Vec<String>, gpu: String, names: bool }
/// One image compiled for its own target (or for `other`: its one target line changed), entries the compiler refuses
/// taken out.
fn compile(ptxas: &str, dir: &Path, label: &str, path: &str, other: Option<&str>) -> Result<Image> {
    let mut text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let targets: Vec<&str> = text.lines().filter_map(|line| line.strip_prefix(".target ")).collect();
    let [target] = targets[..] else { return Err(format!("{path}: not exactly one .target line")) };
    let mut gpu = target.split(|c: char| c == ',' || c.is_whitespace()).next().unwrap_or("").to_string();
    if let Some(other) = other {
        text = text.replacen(&format!("\n.target {gpu}\n"), &format!("\n.target {other}\n"), 1);
        gpu = other.to_string();
    }
    println!("CODE PAIR {label} image={path} text_sha256={} target={gpu} entries={} bytes={}", sha256::digest(text.as_bytes()), text.matches("\n.visible .entry ").count(), text.len());
    let (source, out) = (dir.join(format!("{label}.ptx")), dir.join(format!("{label}.cubin")));
    let mut removed = vec![];
    loop {
        std::fs::write(&source, &text).map_err(|e| e.to_string())?;
        match ptxas_once(ptxas, &gpu, &source, &out)? {
            Ok(seconds) => { println!("CODE PAIR {label} compile --gpu-name={gpu} -m64 -O3: exit=Some(0) seconds={seconds:.1} entries_taken_out={}", removed.len()); break; }
            Err(entry) => {
                if removed.len() >= 12 { return Err("more than 12 entries refused: stop".into()); }
                text = strip_entry(&text, &entry).ok_or_else(|| format!("entry {entry} not found in the text"))?;
                println!("CODE PAIR NOT COMPARED {label} entry={entry}: the offline compiler refuses it (too much static shared data); taken out of the {label} text");
                removed.push(entry);
            }
        }
    }
    let image = std::fs::read(&out).map_err(|e| e.to_string())?;
    println!("CODE PAIR {label} machine code bytes={} sha256={} elf_flags={:#010x} entries_taken_out={:?}", image.len(), sha256::digest(&image), u32::from_le_bytes(image[48..52].try_into().unwrap()), removed);
    let sections = sections(&image)?;
    let names = names_check(label, &text, &sections);
    Ok(Image { sections, removed, gpu, names })
}
/// The kernels two images do not share: (only in the first, only in the second).
fn cover(first: &[Section], second: &[Section]) -> (Vec<String>, Vec<String>) {
    let code = |image: &[Section]| image.iter().filter(|s| class(&s.name) == "code").map(|s| s.name[6..].to_string()).collect::<Vec<_>>();
    let (a, b) = (code(first), code(second));
    (a.iter().filter(|n| !b.contains(n)).cloned().collect(), b.iter().filter(|n| !a.contains(n)).cloned().collect())
}
/// What the target-alone check found: the old text compiled for the new target (the third image) against the old image.
/// `differ`: kernels with other machine code. `missing`: kernels of the old image the third image does not hold (not
/// compared). `extra`: kernels only the third image holds. `removed`: entries taken out of the third text.
#[derive(Default)]
struct Alone { differ: Vec<String>, missing: Vec<String>, extra: Vec<String>, removed: Vec<String> }
impl Alone {
    /// Kernels only the third image holds that no entry the old target refused explains (strict: every one of them).
    fn unexplained(&self, strict: bool, removed_old: &[String]) -> Vec<String> { self.extra.iter().filter(|n| strict || !removed_old.contains(n)).cloned().collect() }
    /// "The target alone changes no kernel" is proven only when every kernel of the old image was compared and is equal.
    fn proven(&self, strict: bool, removed_old: &[String]) -> bool { self.differ.is_empty() && self.missing.is_empty() && self.unexplained(strict, removed_old).is_empty() && !(strict && !self.removed.is_empty()) }
}
/// Why a receipt fails (none: it passes). No compiler in here, so the rules have tests.
fn reasons(strict: bool, pinned: bool, names: bool, removed_old: &[String], removed_new: &[String], alone: Option<&Alone>) -> Vec<String> {
    let mut reasons = vec![];
    if !pinned { reasons.push("the compiler is not the pinned binary".to_string()); }
    if !names { reasons.push("the code sections of an image do not match the names of its text one to one".to_string()); }
    if let Some(alone) = alone {
        if !alone.differ.is_empty() { reasons.push(format!("a kernel has other machine code under the new target {:?}", alone.differ)); }
        if !alone.missing.is_empty() { reasons.push(format!("kernels of the old image are not in the old text compiled for the new target, so the target-alone check did not compare them {:?}", alone.missing)); }
        let unexplained = alone.unexplained(strict, removed_old);
        if !unexplained.is_empty() { reasons.push(format!("kernels are only in the old text compiled for the new target, so the target-alone check did not compare them {unexplained:?}")); }
        if strict && !alone.removed.is_empty() { reasons.push(format!("entries were taken out of the old text compiled for the new target {:?}", alone.removed)); }
    }
    if strict && !(removed_old.is_empty() && removed_new.is_empty()) { reasons.push(format!("entries were taken out (old {removed_old:?}, new {removed_new:?})")); }
    reasons
}
/// `strict`: the run also fails when an entry had to be taken out of any of the three images, or when the old image and
/// the old text compiled for the new target do not hold the same kernels (the receipt of a head that lands).
/// It always fails when the compiler is not the pinned binary, when the code sections of an image do not match the
/// names of its text one to one, or when a kernel of the old image was not compared by the target-alone check.
/// Kernels that differ between the old and the new image are a list, not a failure.
pub(crate) fn run(old_path: &str, new_path: &str, strict: bool) -> Result<()> {
    let mut found = None;
    for path in ["/usr/local/cuda/bin/ptxas", "/usr/local/cuda-13.0/bin/ptxas", "/usr/bin/ptxas", "ptxas"] {
        if let Ok(out) = Command::new(path).arg("--version").output() && out.status.success() { found = Some((path, String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("").to_string())); break; }
    }
    let Some((ptxas, version)) = found else { return Err("no ptxas on this box".into()); };
    let binary = std::fs::read(ptxas).map(|b| sha256::digest(&b)).unwrap_or_else(|e| format!("unreadable ({e})"));
    let pinned = binary == PINNED_PTXAS_SHA256;
    println!("CODE PAIR ptxas={ptxas} version_line=\"{version}\" sha256={binary} pinned_version={PINNED_PTXAS_VERSION} pinned_sha256={PINNED_PTXAS_SHA256} is_the_pinned_binary={pinned}");
    let dir = std::env::temp_dir().join(format!("code-pair-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let result = (|| -> Result<()> {
        let (old, new) = (compile(ptxas, &dir, "old", old_path, None)?, compile(ptxas, &dir, "new", new_path, None)?);
        let short = |d: &[u8]| sha256::digest(d)[..16].to_string();
        // The one section of that name (None: absent, or named twice; `names_check` fails an image with a code name twice).
        fn find<'a>(image: &'a Image, name: &str) -> Option<&'a Section> { let mut all = image.sections.iter().filter(|s| s.name == name); let first = all.next()?; all.next().is_none().then_some(first) }
        // Code, one line a kernel of the new image.
        let (mut equal, mut differ, mut fresh) = (vec![], vec![], vec![]);
        for section in new.sections.iter().filter(|s| class(&s.name) == "code") {
            let kernel = &section.name[6..];
            let before = find(&old, &section.name);
            let verdict = match before { None => "NEW", Some(b) if b.same(section) => "EQUAL", _ => "DIFFER" };
            println!("CODE PAIR kernel {kernel} bytes_old={} bytes_new={} old={} new={} {verdict}", before.map_or("none".to_string(), |b| b.bytes.len().to_string()), section.bytes.len(), before.map_or("none".to_string(), |b| short(&b.bytes)), short(&section.bytes));
            match verdict { "EQUAL" => equal.push(kernel), "DIFFER" => differ.push(kernel), _ => fresh.push(kernel) }
        }
        let gone: Vec<&str> = old.sections.iter().filter(|s| class(&s.name) == "code" && !new.sections.iter().any(|t| t.name == s.name)).map(|s| &s.name[6..]).collect();
        // The other sections, by class; each one that differs by name. A section without bytes (the shared arrays) is
        // compared by its declared size, type, flags and alignment.
        let mut tally: BTreeMap<&str, [usize; 4]> = Default::default();
        let (mut listed, mut shared_bytes) = (0, [0u64; 2]);
        for section in new.sections.iter().filter(|s| class(&s.name) != "code") {
            if class(&section.name) == "shared" { shared_bytes[1] += section.size; }
            let slot = tally.entry(class(&section.name)).or_default();
            match find(&old, &section.name) {
                None => slot[2] += 1,
                Some(b) if b.same(section) => slot[0] += 1,
                Some(b) => { slot[1] += 1; if listed < 80 { listed += 1; println!("CODE PAIR section differs class={} name={} size_old={} size_new={} type_old={} type_new={} flags_old={:#x} flags_new={:#x} align_old={} align_new={}", class(&section.name), section.name, b.size, section.size, b.kind, section.kind, b.flags, section.flags, b.align, section.align); } }
            }
        }
        for section in old.sections.iter().filter(|s| class(&s.name) != "code") {
            if class(&section.name) == "shared" { shared_bytes[0] += section.size; }
            if !new.sections.iter().any(|t| t.name == section.name) { tally.entry(class(&section.name)).or_default()[3] += 1; }
        }
        for (kind, [same, changed, added, removed]) in &tally { println!("CODE PAIR class={kind} equal={same} differ={changed} only_new={added} only_old={removed}"); }
        println!("CODE PAIR shared sections: declared bytes old={} new={} (compared by size, type, flags and alignment; they hold no bytes in the file)", shared_bytes[0], shared_bytes[1]);
        println!("CODE PAIR DIFFER {} {:?}", differ.len(), differ);
        println!("CODE PAIR NEW {} {:?}", fresh.len(), fresh);
        println!("CODE PAIR GONE {} {:?}", gone.len(), gone);
        // The target alone, when the targets differ: the OLD text compiled for the new image's target. Every kernel of
        // it against the old image's code (same text, two targets), and an entry the compiler refused for the old target
        // against the new image. (The new text cannot be compiled for the old target when it holds an instruction that
        // target does not have; so the text both targets can hold is the old one.)
        let mut alone = None;
        let mut names = old.names && new.names;
        if old.gpu != new.gpu {
            let again = compile(ptxas, &dir, "old-as-new-target", old_path, Some(&new.gpu))?;
            names &= again.names;
            let (mut same, mut other) = (0usize, vec![]);
            for section in again.sections.iter().filter(|s| class(&s.name) == "code") {
                match find(&old, &section.name) { Some(b) if b.same(section) => same += 1, Some(_) => other.push(section.name[6..].to_string()), None => {} }
            }
            // Both ways: a kernel of the old image the third image lost is as uncompared as one only the third holds.
            let (missing, extra) = cover(&old.sections, &again.sections);
            println!("CODE PAIR TARGET ALONE the old text compiled for {} against the same text compiled for {}: kernels_old={} kernels_retargeted={} equal={same} differ={} {:?} only_old_not_compared={:?} only_retargeted_not_compared={:?} entries_taken_out={:?}", new.gpu, old.gpu, same + other.len() + missing.len(), same + other.len() + extra.len(), other.len(), other, missing, extra, again.removed);
            for entry in &old.removed {
                let name = format!(".text.{entry}");
                let verdict = match (find(&again, &name), find(&new, &name)) { (Some(a), Some(b)) if a.same(b) => "EQUAL", (Some(_), Some(_)) => "DIFFER", _ => "NOT COMPARED" };
                println!("CODE PAIR kernel {entry} old text compiled for {}: {verdict} (against the new image)", new.gpu);
            }
            alone = Some(Alone { differ: other, missing, extra, removed: again.removed });
        }
        println!("CODE PAIR RESULT kernels_new={} equal={} differ={} new={} gone={} not_compared_old={:?} not_compared_new={:?}", equal.len() + differ.len() + fresh.len(), equal.len(), differ.len(), fresh.len(), gone.len(), old.removed, new.removed);
        // The verdict of the receipt itself (the lists above are information; the device proofs cover them).
        let reasons = reasons(strict, pinned, names, &old.removed, &new.removed, alone.as_ref());
        let target_alone = alone.as_ref().is_none_or(|a| a.proven(strict, &old.removed));
        println!("CODE PAIR VERDICT {} strict={strict} pinned_compiler={pinned} names_one_to_one={names} target_alone_changes_no_kernel={target_alone} entries_taken_out_old={} new={} old_as_new_target={} reasons={reasons:?}", if reasons.is_empty() { "PASS" } else { "FAIL" }, old.removed.len(), new.removed.len(), alone.as_ref().map_or(0, |a| a.removed.len()));
        if reasons.is_empty() { Ok(()) } else { Err(format!("CODE PAIR FAIL: {}", reasons.join("; "))) }
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn code(names: &[&str]) -> Vec<Section> { names.iter().map(|n| Section { name: format!(".text.{n}"), kind: 1, flags: 6, align: 128, size: 4, bytes: vec![1, 2, 3, 4] }).collect() }
    fn list(names: &[&str]) -> Vec<String> { names.iter().map(|n| n.to_string()).collect() }

    #[test] fn cover_names_the_kernels_only_one_image_holds() {
        let mut old = code(&["alpha", "beta"]);
        old.push(Section { name: ".nv.shared.beta".into(), kind: 8, flags: 3, align: 16, size: 65536, bytes: vec![] });
        assert_eq!(cover(&old, &code(&["alpha", "beta"])), (vec![], vec![]));
        assert_eq!(cover(&old, &code(&["alpha"])), (list(&["beta"]), vec![]));
        assert_eq!(cover(&old, &code(&["alpha", "beta", "gamma"])), (vec![], list(&["gamma"])));
    }
    #[test] fn a_clean_receipt_passes_and_each_basic_fault_fails() {
        let clean = Alone::default();
        assert!(reasons(true, true, true, &[], &[], Some(&clean)).is_empty());
        assert!(reasons(true, true, true, &[], &[], None).is_empty());
        assert!(clean.proven(true, &[]));
        assert_eq!(reasons(true, false, true, &[], &[], Some(&clean)).len(), 1);
        assert_eq!(reasons(true, true, false, &[], &[], Some(&clean)).len(), 1);
        let differ = Alone { differ: list(&["alpha"]), ..Default::default() };
        assert_eq!(reasons(false, true, true, &[], &[], Some(&differ)).len(), 1);
        assert!(!differ.proven(false, &[]));
    }
    // The reviewer's case (REVC 14:04): the old and the new image compile whole, the old text compiled for the new target
    // loses an entry. The kernel was never compared, so no mode may pass.
    #[test] fn a_kernel_the_third_image_lost_fails_in_both_modes() {
        let lost = Alone { missing: list(&["beta"]), removed: list(&["beta"]), ..Default::default() };
        for strict in [true, false] {
            assert!(reasons(strict, true, true, &[], &[], Some(&lost)).iter().any(|r| r.contains("did not compare") && r.contains("beta")), "strict={strict}");
            assert!(!lost.proven(strict, &[]), "strict={strict}");
        }
        assert!(reasons(true, true, true, &[], &[], Some(&lost)).iter().any(|r| r.contains("taken out of the old text compiled for the new target")));
        // Lost without a refusal (a compiler that drops a kernel silently): still not compared.
        let silent = Alone { missing: list(&["beta"]), ..Default::default() };
        assert!(!reasons(true, true, true, &[], &[], Some(&silent)).is_empty() && !reasons(false, true, true, &[], &[], Some(&silent)).is_empty());
    }
    #[test] fn a_kernel_only_the_third_image_holds_needs_an_entry_the_old_target_refused_and_never_passes_strict() {
        let extra = Alone { extra: list(&["absorb"]), ..Default::default() };
        let refused = list(&["absorb"]);
        assert!(reasons(false, true, true, &refused, &[], Some(&extra)).is_empty());
        assert!(extra.proven(false, &refused));
        assert!(!reasons(false, true, true, &[], &[], Some(&extra)).is_empty());
        assert!(!extra.proven(false, &[]));
        assert!(reasons(true, true, true, &refused, &[], Some(&extra)).len() >= 2);
        assert!(!extra.proven(true, &refused));
    }
    #[test] fn strict_refuses_any_entry_taken_out_of_any_of_the_three_images() {
        let clean = Alone::default();
        assert!(!reasons(true, true, true, &list(&["a"]), &[], Some(&clean)).is_empty());
        assert!(!reasons(true, true, true, &[], &list(&["a"]), Some(&clean)).is_empty());
        // Both targets refuse the same entry: in neither image, so not missing and not extra; strict still refuses.
        let both = Alone { removed: list(&["a"]), ..Default::default() };
        assert!(!reasons(true, true, true, &list(&["a"]), &list(&["a"]), Some(&both)).is_empty());
        assert!(!both.proven(true, &list(&["a"])));
        assert!(reasons(false, true, true, &list(&["a"]), &list(&["a"]), Some(&both)).is_empty());
    }
}
