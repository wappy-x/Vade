//! Host-only install failures. Fixtures contain synthetic bytes, never runnable programs.
use crate::{prebuilt::{self, Bundle}, sha256, state::{Artifact, Manifest}};
use std::{collections::BTreeMap, fs, io::{self, Read}, path::{Path, PathBuf}, sync::atomic::{AtomicU64, Ordering}, time::{SystemTime, UNIX_EPOCH}};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("vade-host-edges-{}-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&root).unwrap();
        let fixture = Self(root);
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/host_edges").join(name);
        copy_fixture(&source, &fixture.0);
        fixture
    }
    fn bundle(&self) -> Result<Option<Bundle>, String> { Bundle::find(Some(&self.0), "engine", "cli") }
}
impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn copy_fixture(source: &Path, target: &Path) {
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap(); let name = entry.file_name(); let kind = entry.file_type().unwrap();
        let destination = target.join(if name == "flash.ptx.fixture" { std::ffi::OsStr::new("flash.ptx") } else { &name });
        if kind.is_dir() { fs::create_dir(&destination).unwrap(); copy_fixture(&entry.path(), &destination); }
        else { assert!(kind.is_file(), "fixture must contain only regular files and directories"); fs::copy(entry.path(), destination).unwrap(); }
    }
}
fn bundle_error(name: &str, expected: &str) {
    let fixture = Fixture::new(name);
    let error = fixture.bundle().err().unwrap_or_else(|| panic!("{name} unexpectedly passed"));
    assert!(error.contains(expected), "{name}: {error}");
}

#[test]
fn install_manifest_absent_and_oversized_fail_before_payload_use() {
    let fixture = Fixture::new("valid");
    assert!(fixture.bundle().unwrap().is_some());
    for (case, expected) in [("manifest-absent", "no readable manifest"), ("manifest-large", "exceeds 4 KiB")] {
        bundle_error(case, expected);
    }
}

#[test]
fn install_payload_requires_a_complete_lowercase_sha256() {
    for case in ["hash-absent", "hash-invalid"] { bundle_error(case, "missing an artifact SHA256"); }
}

#[test]
fn install_payload_absent_and_empty_have_a_specific_error() {
    for case in ["payload-absent", "payload-empty"] { bundle_error(case, "flash-serve is missing or empty"); }
}

#[test]
fn install_matching_hash_does_not_accept_short_or_wrong_machine_executables() {
    for (case, expected) in [("payload-short", "executable is truncated"), ("payload-arch", "Linux AArch64 ELF64")] {
        bundle_error(case, expected);
    }
}

#[test]
fn install_host_bundle_requires_identity_when_only_a_host_hash_is_present() {
    let fixture = Fixture::new("host-identity");
    let bundle = fixture.bundle().unwrap().unwrap();
    for harness in ["harness", "another-revision", ""] {
        let error = bundle.host(harness).err().expect("incomplete host identity must fail");
        assert_eq!(error, "Prebuilt host target or harness revision does not match.");
    }
    let valid = Fixture::new("valid");
    assert!(valid.bundle().unwrap().unwrap().host("harness").unwrap().is_none());
}

#[test]
fn install_host_bundle_rejects_missing_hash_missing_file_and_corrupt_file() {
    assert_ne!(prebuilt::host_target(), "unsupported", "test fixtures support the CLI's four host targets");
    for (case, expected) in [
        ("host-hash-absent", "host artifact SHA256 is missing"),
        ("host-payload-absent", "host artifact is missing"),
        ("host-hash-mismatch", "host artifact SHA256 mismatch"),
    ] {
        let fixture = Fixture::new(&format!("{case}-{}", prebuilt::host_target()));
        let bundle = fixture.bundle().unwrap().unwrap();
        let error = bundle.host("harness").err().unwrap_or_else(|| panic!("{case} unexpectedly passed"));
        assert!(error.contains(expected), "{case}: {error}");
    }
}

struct InterruptedChunks<'a> { remaining: &'a [u8], state: u64, interrupt: bool }
impl Read for InterruptedChunks<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.interrupt = !self.interrupt;
        if self.interrupt { return Err(io::ErrorKind::Interrupted.into()); }
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let count = output.len().min(self.remaining.len()).min((self.state >> 32) as usize % 97 + 1);
        output[..count].copy_from_slice(&self.remaining[..count]); self.remaining = &self.remaining[count..]; Ok(count)
    }
}

#[test]
fn model_hash_retries_interrupted_reads_without_skipping_or_repeating_bytes() {
    let mut state = 0x56ad_e123_7102_bacdu64;
    let bytes: Vec<u8> = (0..2049).map(|_| { state = state.wrapping_mul(6364136223846793005).wrapping_add(1); (state >> 32) as u8 }).collect();
    for length in [0, 1, 55, 56, 63, 64, 65, 127, 128, 1024, 2049] {
        let expected = sha256::sha(&bytes[..length]);
        for seed in 0..8 {
            let reader = InterruptedChunks { remaining: &bytes[..length], state: seed, interrupt: false };
            assert_eq!(sha256::digest_reader(reader).unwrap(), expected, "length {length}, seed {seed}");
        }
    }
}

#[test]
fn model_hash_propagates_terminal_read_errors_after_partial_input() {
    struct Fails { calls: usize, kind: io::ErrorKind }
    impl Read for Fails {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            match self.calls { 1 => { output[..3].copy_from_slice(b"abc"); Ok(3) }, 2 => Err(self.kind.into()), _ => panic!("read again after terminal error") }
        }
    }
    for kind in [io::ErrorKind::PermissionDenied, io::ErrorKind::UnexpectedEof, io::ErrorKind::TimedOut, io::ErrorKind::InvalidData] {
        let mut reader = Fails { calls: 0, kind };
        assert_eq!(sha256::digest_reader(&mut reader).unwrap_err().kind(), kind);
        assert_eq!(reader.calls, 2);
    }
}

#[test]
fn model_hash_returns_after_first_eof_without_polling_the_reader_again() {
    struct Eof { calls: usize }
    impl Read for Eof {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            self.calls += 1; assert_eq!(self.calls, 1, "a completed reader must not spin"); Ok(0)
        }
    }
    let mut reader = Eof { calls: 0 };
    assert_eq!(sha256::digest_reader(&mut reader).unwrap(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(reader.calls, 1);
}

#[test]
fn install_manifest_round_trips_seeded_artifacts_and_rejects_corrupt_fields() {
    let mut seed = 0xa25c_109bu64;
    for count in 1..9 {
        let mut artifacts = BTreeMap::new();
        for index in 0..count {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let hash = format!("{seed:016x}").repeat(4);
            artifacts.insert(format!("machine_{index}"), Artifact {
                worker: format!("/fixture/{index}/vade"), binary: format!("/fixture/{index}/engine"), ptx: format!("/fixture/{index}/kernel"),
                worker_sha256: hash.clone(), binary_sha256: hash.clone(), ptx_sha256: hash,
            });
        }
        let manifest = Manifest { engine_sha: "a".repeat(40), cli_sha: "b".repeat(40), harness_sha256: Some("c".repeat(64)), supports_output: true, supports_fp4_chain: true, artifacts };
        let encoded = manifest.encode();
        assert_eq!(Manifest::parse(&encoded).unwrap().encode(), encoded);
        for (from, to, expected) in [
            ("version=1", "version=9", "Unknown build manifest version"),
            ("harness_sha256=", "harness_sha256=z", "Harness SHA256 is invalid"),
            ("binary_sha256=", "binary_sha256=z", "Artifact SHA256 is invalid"),
            ("worker=/fixture/0/vade", "worker=/fixture/../vade", "no parent traversal"),
        ] {
            let error = Manifest::parse(&encoded.replacen(from, to, 1)).err().expect("corrupt manifest must fail");
            assert!(error.contains(expected), "{error}");
        }
    }
}
