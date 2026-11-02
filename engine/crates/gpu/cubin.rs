//! Optional release-producer input. The installed binary authenticates this image.
use std::{collections::BTreeMap, env, fs, io, path::Path};
#[allow(dead_code)]
#[path = "../../tools/export/src/sha256.rs"]
mod sha256;

// The one compiler the token id proof covers. Keep these equal to scripts/make-cubin.sh.
// Another compiler can emit other GPU code, so it needs a new proof and a change here.
const PTXAS_VERSION: &str = "13.0.88";
const PTXAS_SHA256: &str = "a1941a04ca4fd233b2fbe50c625b1e72b3d5f79ebe80209a272c85482dfbb487";
const PTXAS_ARGS: &str = "--gpu-name=sm_121 -m64 -O3";

fn invalid(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }

fn validate(ptx: &[u8], image: &[u8], manifest: &str) -> io::Result<i32> {
    let mut fields = BTreeMap::new();
    for line in manifest.lines().filter(|line| !line.is_empty()) {
        let (key, value) = line.split_once('=').ok_or_else(|| invalid("invalid cubin manifest"))?;
        if fields.insert(key, value).is_some() { return Err(invalid("duplicate cubin manifest field")); }
    }
    if fields.len() != 8 || fields.get("format") != Some(&"vade-cubin-v1")
        || fields.get("target") != Some(&"sm_121") {
        return Err(invalid("cubin manifest must describe vade-cubin-v1 for sm_121"));
    }
    if fields.get("ptxas_version") != Some(&PTXAS_VERSION) || fields.get("ptxas_sha256") != Some(&PTXAS_SHA256)
        || fields.get("ptxas_args") != Some(&PTXAS_ARGS) {
        return Err(invalid("cubin was not compiled by the pinned ptxas and arguments"));
    }
    if fields.get("ptx_sha256").copied() != Some(sha256::digest(ptx).as_str()) {
        return Err(invalid("cubin PTX SHA256 differs from this build's exact kernel bytes"));
    }
    if fields.get("cubin_sha256").copied() != Some(sha256::digest(image).as_str()) {
        return Err(invalid("cubin SHA256 differs from its producer manifest"));
    }
    // Deliberately accept only the vetted ptxas CUDA ELF ABI/target encoding.
    // Numeric SM alone is insufficient: sm_121a uses 0x0a007902, whereas the
    // supported sm_121 image uses 0x06007902. Unknown ABI/features fail closed.
    if image.len() < 64 || &image[..9] != b"\x7fELF\x02\x01\x01\x41\x08"
        || u16::from_le_bytes([image[16], image[17]]) != 2
        || u16::from_le_bytes([image[18], image[19]]) != 190
        || u32::from_le_bytes(image[20..24].try_into().unwrap()) != 1
        || u32::from_le_bytes(image[48..52].try_into().unwrap()) != 0x06007902
        || u16::from_le_bytes([image[52], image[53]]) != 64 {
        return Err(invalid("cubin requires CUDA ELF64 ABI 8 and exact sm_121 flags 0x06007902"));
    }
    fields.get("min_driver_api").and_then(|value| value.parse::<i32>().ok())
        .filter(|version| *version >= 13000 && *version < 100000)
        .ok_or_else(|| invalid("cubin needs its producer's CUDA driver API floor (at least 13000)"))
}

pub fn prepare(out: &Path, ptx: &[u8]) -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=SPARK_CUBIN");
    let (image, minimum, stamp) = if let Some(path) = env::var_os("SPARK_CUBIN") {
        let path = std::path::PathBuf::from(path);
        let manifest = path.with_extension("cubin.conf");
        println!("cargo:rerun-if-changed={}", path.display());
        println!("cargo:rerun-if-changed={}", manifest.display());
        let image = fs::read(&path)?;
        let minimum = validate(ptx, &image, &fs::read_to_string(manifest)?)?;
        let stamp = format!("cubin_sha256={}", sha256::digest(&image));
        (image, minimum, stamp)
    } else { (Vec::new(), 0, String::new()) };
    fs::write(out.join("kernel.cubin"), image)?;
    fs::write(out.join("cubin_stamp.rs"), format!("{stamp:?}\n"))?;
    fs::write(out.join("cubin_driver.rs"), format!("{minimum}i32\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<u8>, String) {
        let image = include_bytes!("fixtures/sm_121.elf-header").to_vec();
        let manifest = format!("format=vade-cubin-v1\ntarget=sm_121\nptx_sha256={}\ncubin_sha256={}\nmin_driver_api=13000\nptxas_version={PTXAS_VERSION}\nptxas_sha256={PTXAS_SHA256}\nptxas_args={PTXAS_ARGS}\n",
            sha256::digest(b"exact PTX"), sha256::digest(&image));
        (image, manifest)
    }
    #[test]
    fn producer_image_is_bound_to_exact_ptx_bytes_and_digest() {
        let (mut image, manifest) = fixture();
        assert_eq!(validate(b"exact PTX", &image, &manifest).unwrap(), 13000);
        assert!(validate(b"changed PTX", &image, &manifest).is_err());
        image[63] ^= 1;
        assert!(validate(b"exact PTX", &image, &manifest).is_err());
    }
    #[test]
    fn producer_manifest_refuses_ambiguity_and_wrong_architecture() {
        let (image, manifest) = fixture();
        for bad in [manifest.replace("sm_121", "sm_120"), manifest.replace("13000", "0"),
            format!("{manifest}target=sm_121\n"), format!("{manifest}unknown=value\n")] {
            assert!(validate(b"exact PTX", &image, &bad).is_err());
        }
        let mut wrong = image.clone();
        wrong[49] = 120;
        let wrong_manifest = manifest.replace(&sha256::digest(&image), &sha256::digest(&wrong));
        assert!(validate(b"exact PTX", &wrong, &wrong_manifest).is_err());
    }
    #[test]
    fn producer_refuses_any_compiler_but_the_pinned_one() {
        let (image, manifest) = fixture();
        for bad in [manifest.replace(PTXAS_VERSION, "13.0.48"), manifest.replace(PTXAS_SHA256, &"0".repeat(64)),
            manifest.replace(PTXAS_ARGS, "--gpu-name=sm_121 -m64 -O2"),
            manifest.lines().filter(|line| !line.starts_with("ptxas_args=")).collect::<Vec<_>>().join("\n")] {
            assert!(validate(b"exact PTX", &image, &bad).is_err());
        }
    }
    #[test]
    fn producer_rejects_compiler_sm_121a_header_even_with_matching_digest() {
        let (image, manifest) = fixture();
        let other = include_bytes!("fixtures/sm_121a.elf-header");
        let manifest = manifest.replace(&sha256::digest(&image), &sha256::digest(other));
        assert!(validate(b"exact PTX", other, &manifest).unwrap_err().to_string().contains("exact sm_121"));
    }
    #[test]
    fn producer_rejects_unvetted_elf_abi_and_features_even_with_matching_digest() {
        let (image, manifest) = fixture();
        for byte in [7, 8, 16, 18, 20, 48, 49, 50, 51, 52] {
            let mut other = image.clone();
            other[byte] ^= 1;
            let manifest = manifest.replace(&sha256::digest(&image), &sha256::digest(&other));
            assert!(validate(b"exact PTX", &other, &manifest).is_err(), "byte {byte}");
        }
    }
}
