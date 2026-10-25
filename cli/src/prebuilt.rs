//! An exact-source, owner-provided Linux bundle. Missing is a source fallback;
//! a corrupt matching bundle is rejected before falling back to source.
use std::{fs::{self, File}, io::Read, path::{Path, PathBuf}};
use crate::{sha256, text};
/// A verified prebuilt Linux bundle for one engine and CLI revision, with the
/// SHA-256 of each file in [`FILES`] order.
pub struct Bundle { pub dir: PathBuf, pub hashes: [String; 3], fields: text::Fields }
/// The bundle's verified host-side CLI and harness for this computer.
pub struct HostBundle { pub cli: PathBuf, pub harness: PathBuf, pub hashes: [String; 2] }
/// The Spark files every bundle carries.
pub const FILES: [&str; 3] = ["flash-serve", "flash.ptx", "vade"];
/// The release launcher bundle's manifest (`vade-launch-bundle-v1`).
const LAUNCHER: &str = "launcher-bundle.conf";
impl Bundle {
    /// The bundle in `root/<engine>-<cli>` after checking its manifest and hashes;
    /// `None` when there is none, an error when a matching one is corrupt.
    /// An unpacked release launcher bundle (`launcher-bundle.conf`) carries the
    /// same Spark files and serves too, in that folder or the matching one.
    pub fn find(root: Option<&Path>, engine: &str, cli: &str) -> Result<Option<Self>, String> {
        let Some(root) = root else { return Ok(None); };
        let mut dir = root.join(format!("{engine}-{cli}"));
        let launcher = |dir: &Path| dir.join(LAUNCHER).try_exists().map_err(|_| "Cannot inspect prebuilt_dir.");
        if !dir.try_exists().map_err(|_| "Cannot inspect prebuilt_dir.")? {
            if !launcher(root)? { return Ok(None); }
            dir = root.to_path_buf();
        }
        let release = launcher(&dir)?;
        let mut manifest = String::new();
        File::open(dir.join(if release { LAUNCHER } else { "manifest.conf" })).and_then(|f| f.take(4097).read_to_string(&mut manifest))
            .map_err(|_| "Matching prebuilt bundle has no readable manifest.conf; repair it or remove it to build from source.")?;
        if manifest.len() > 4096 { return Err("Prebuilt manifest exceeds 4 KiB.".into()); }
        let mut fields = text::parse(&manifest)?;
        if release {
            if fields.get("engine_sha").map(String::as_str) != Some(engine) || fields.get("cli_sha").map(String::as_str) != Some(cli) {
                // Another release's bundle: build this checkout's revisions from source.
                if dir == root { return Ok(None); }
            }
            // Same files under the launcher's names; it has no host pair.
            let mut mapped = text::Fields::new();
            for (key, value) in fields {
                let key = match key.as_str() {
                    "format" if value == "vade-launch-bundle-v1" => { mapped.insert(key, "vade-prebuilt-v1".into()); continue; }
                    "format" => key,
                    "engine_sha256" => "flash-serve.sha256".into(),
                    "ptx_sha256" => "flash.ptx.sha256".into(),
                    "vade_sha256" => "vade.sha256".into(),
                    "target" | "engine_sha" | "cli_sha" => key,
                    _ => continue,
                };
                mapped.insert(key, value);
            }
            fields = mapped;
        }
        for (key, expected) in [("format", "vade-prebuilt-v1"), ("target", "aarch64-unknown-linux-gnu"), ("engine_sha", engine), ("cli_sha", cli)] {
            if fields.get(key).map(String::as_str) != Some(expected) {
                return Err(format!("Prebuilt {key} does not match this source/platform. Repair the matching bundle; nothing installed."));
            }
        }
        let mut hashes = Vec::new();
        for name in FILES {
            let expected = fields.get(&format!("{name}.sha256")).filter(|s| text::hex(s, 64))
                .ok_or("Prebuilt manifest is missing an artifact SHA256.")?;
            let path = dir.join(name);
            if !fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() > 0) {
                return Err(format!("Prebuilt {name} is missing or empty."));
            }
            let actual = sha256::digest_reader(File::open(&path).map_err(|_| "Cannot read prebuilt artifact.")?)
                .map_err(|_| "Cannot hash prebuilt artifact.")?;
            if actual != *expected { return Err(format!("Prebuilt {name} SHA256 mismatch; nothing installed.")); }
            if name != "flash.ptx" {
                let mut header = [0; 20];
                File::open(&path).and_then(|mut f| f.read_exact(&mut header)).map_err(|_| "Prebuilt executable is truncated.")?;
                if &header[..6] != b"\x7fELF\x02\x01" || header[18..20] != [183, 0] {
                    return Err("Prebuilt executable must be Linux AArch64 ELF64.".into());
                }
            }
            hashes.push(actual);
        }
        Ok(Some(Self { dir, hashes: hashes.try_into().unwrap(), fields }))
    }
    /// The host executables for this platform and harness revision, if the bundle
    /// has any; mismatched or corrupt ones are errors.
    pub fn host(&self, harness: &str) -> Result<Option<HostBundle>, String> {
        if !self.fields.keys().any(|key| key.starts_with("host-") || key == "host_target") {
            return Ok(None);
        }
        if host_target() == "unsupported" || self.fields.get("host_target").map(String::as_str) != Some(host_target())
            || self.fields.get("harness_sha").map(String::as_str) != Some(harness) {
            return Err("Prebuilt host target or harness revision does not match.".into());
        }
        let mut hashes = Vec::new();
        for name in ["host-vade-harness", "host-vade"] {
            let expected = self.fields.get(&format!("{name}.sha256")).filter(|s| text::hex(s, 64))
                .ok_or("Prebuilt host artifact SHA256 is missing.")?;
            let path = self.dir.join(name);
            if !fs::metadata(&path).is_ok_and(|meta| meta.is_file()) {
                return Err("Prebuilt host artifact is missing.".into());
            }
            let actual = sha256::digest_reader(File::open(&path).map_err(|_| "Cannot read prebuilt host artifact.")?)
                .map_err(|_| "Cannot hash prebuilt host artifact.")?;
            if actual != *expected { return Err("Prebuilt host artifact SHA256 mismatch.".into()); }
            let mut header = [0; 20];
            File::open(&path).and_then(|mut f| f.read_exact(&mut header)).map_err(|_| "Prebuilt host executable is truncated.")?;
            if !host_executable(&header) { return Err("Prebuilt host executable has the wrong platform.".into()); }
            hashes.push(actual);
        }
        Ok(Some(HostBundle { cli: self.dir.join("host-vade"), harness: self.dir.join("host-vade-harness"),
            hashes: hashes.try_into().unwrap() }))
    }
}
/// This computer's Rust target triple, or "unsupported".
pub(crate) fn host_target() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        _ => "unsupported",
    }
}
fn host_executable(header: &[u8; 20]) -> bool {
    let arm = cfg!(target_arch = "aarch64");
    if cfg!(target_os = "macos") {
        header[..4] == [0xcf, 0xfa, 0xed, 0xfe]
            && u32::from_le_bytes(header[4..8].try_into().unwrap()) == if arm { 0x0100_000c } else { 0x0100_0007 }
    } else if cfg!(target_os = "linux") {
        &header[..6] == b"\x7fELF\x02\x01" && u16::from_le_bytes(header[18..20].try_into().unwrap()) == if arm { 183 } else { 62 }
    } else { false }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_bundle_checks_hashes_target_and_source_before_use() {
        let root = std::env::temp_dir().join(format!("vade-prebuilt-{}", text::random_hex(8).unwrap()));
        let engine = "a".repeat(40); let cli = "b".repeat(40);
        assert!(Bundle::find(Some(&root), &engine, &cli).unwrap().is_none());
        let dir = root.join(format!("{engine}-{cli}")); fs::create_dir_all(&dir).unwrap();
        let mut manifest = format!("format=vade-prebuilt-v1\ntarget=aarch64-unknown-linux-gnu\nengine_sha={engine}\ncli_sha={cli}\n");
        for name in FILES {
            let mut bytes = vec![0; 32]; bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183;
            fs::write(dir.join(name), &bytes).unwrap();
            manifest.push_str(&format!("{name}.sha256={}\n", sha256::sha(&bytes)));
        }
        fs::write(dir.join("manifest.conf"), &manifest).unwrap();
        assert!(Bundle::find(Some(&root), &engine, &cli).unwrap().is_some());
        assert!(Bundle::find(Some(&root), &"c".repeat(40), &cli).unwrap().is_none());
        fs::write(dir.join("manifest.conf"), manifest.replace("aarch64", "x86_64")).unwrap();
        assert!(Bundle::find(Some(&root), &engine, &cli).err().unwrap().contains("platform"));
        fs::write(dir.join("manifest.conf"), manifest).unwrap();
        fs::write(dir.join("flash.ptx"), b"changed").unwrap();
        assert!(Bundle::find(Some(&root), &engine, &cli).err().unwrap().contains("SHA256 mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unpacked_release_launcher_bundle_installs_the_same_spark_files() {
        let root = std::env::temp_dir().join(format!("vade-launch-bundle-{}", text::random_hex(8).unwrap()));
        fs::create_dir_all(&root).unwrap();
        let engine = "a".repeat(40); let cli = "b".repeat(40);
        let mut conf = format!("format=vade-launch-bundle-v1\ntarget=aarch64-unknown-linux-gnu\nengine_sha={engine}\ncli_sha={cli}\nharness_sha={cli}\n");
        for (name, key) in [("flash-serve", "engine_sha256"), ("flash.ptx", "ptx_sha256"), ("vade", "vade_sha256"), ("vade-harness", "harness_sha256")] {
            let mut bytes = vec![0; 32]; bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183; bytes[31] = name.len() as u8;
            fs::write(root.join(name), &bytes).unwrap();
            conf.push_str(&format!("{key}={}\n", sha256::sha(&bytes)));
        }
        fs::write(root.join(LAUNCHER), &conf).unwrap();
        // prebuilt_dir may name the unpacked folder itself.
        let bundle = Bundle::find(Some(&root), &engine, &cli).unwrap().unwrap();
        assert_eq!(bundle.dir, root);
        assert_eq!(bundle.hashes[0], sha256::sha(&fs::read(root.join("flash-serve")).unwrap()));
        assert!(bundle.host(&cli).unwrap().is_none(), "no host pair: the controller builds its own CLI and harness");
        // Another revision's bundle there means: build from source.
        assert!(Bundle::find(Some(&root), &"c".repeat(40), &cli).unwrap().is_none());
        fs::write(root.join("vade"), b"changed").unwrap();
        assert!(Bundle::find(Some(&root), &engine, &cli).err().unwrap().contains("SHA256 mismatch"));
        fs::write(root.join(LAUNCHER), conf.replace("vade-launch-bundle-v1", "vade-launch-bundle-v9")).unwrap();
        assert!(Bundle::find(Some(&root), &engine, &cli).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
