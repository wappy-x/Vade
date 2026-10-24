//! Compile the CLI's embedded flag definitions for the requested ABI, without linking.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const HELP: &str = "vade test unix-flags-abi [--target TARGET]\nCompile all twelve raw Unix flag assertions without linking or running target code.\nDefaults to the selected rustc's host. RUSTC selects the compiler.\nSupported: aarch64/x86_64 Linux GNU and macOS; target Rust libraries must be installed.\n";
const NAMES: [&str; 12] = [
    "O_NOFOLLOW",
    "O_DIRECTORY",
    "O_CLOEXEC",
    "O_NONBLOCK",
    "O_CREAT",
    "O_EXCL",
    "O_RDONLY",
    "O_RDWR",
    "F_GETFL",
    "F_SETFL",
    "LOCK_EX",
    "LOCK_NB",
];

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn run(args: &[String]) -> Result<i32, String> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        crate::public_output(HELP);
        return Ok(0);
    }
    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let target = match args {
        [] => {
            let output = Command::new(&compiler)
                .arg("-vV")
                .output()
                .map_err(|error| format!("cannot query rustc host: {error}"))?;
            if !output.status.success() {
                return Err("rustc host query failed".into());
            }
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .find_map(|line| line.strip_prefix("host: "))
                .ok_or("rustc did not report its host target")?
                .to_owned()
        }
        [flag, target] if flag == "--target" => target.clone(),
        _ => return Err(format!("invalid ABI check arguments; {HELP}")),
    };
    // Independent ABI snapshot: Linux v6.12 arch/arm64/include/uapi/asm/fcntl.h
    // and include/uapi/asm-generic/fcntl.h; macOS XNU bsd/sys/fcntl.h. The preserved
    // libc 0.2.189 compile assertions paired these values before shell removal.
    let first_six = match target.as_str() {
        "aarch64-unknown-linux-gnu" => [0x8000, 0x4000, 0x80000, 0x800, 0x40, 0x80],
        "x86_64-unknown-linux-gnu" => [0x20000, 0x10000, 0x80000, 0x800, 0x40, 0x80],
        "aarch64-apple-darwin" | "x86_64-apple-darwin" => {
            [0x100, 0x100000, 0x1000000, 0x4, 0x200, 0x800]
        }
        _ => return Err(format!("unsupported ABI check target: {target}")),
    };
    let mut expected = first_six.to_vec();
    expected.extend_from_slice(&[0, 2, 3, 4, 2, 4]);
    let root =
        std::env::temp_dir().join(format!("vade-flags-abi-{}", crate::text::random_hex(12)?));
    fs::create_dir(&root)
        .map_err(|error| format!("cannot create ABI scratch directory: {error}"))?;
    let scratch = Scratch(root);
    // Check exactly the definitions built into this CLI, using target cfgs
    // selected by rustc rather than the cfgs of the host running the driver.
    fs::write(
        scratch.0.join("unix_flags.rs"),
        include_str!("../unix_flags.rs"),
    )
    .map_err(|error| format!("cannot write ABI definitions: {error}"))?;
    let mut source = "#![no_std]\nmod unix_flags;\nconst _: () = {\n".to_owned();
    for (name, value) in NAMES.iter().zip(expected) {
        source.push_str(&format!(
            "assert!(unix_flags::{name} == {value}, \"{name} ABI mismatch\");\n"
        ));
    }
    source.push_str("};\n");
    let input = scratch.0.join("check.rs");
    fs::write(&input, source).map_err(|error| format!("cannot write ABI assertions: {error}"))?;
    let status = Command::new(&compiler)
        .args([
            "--edition=2024",
            "--crate-name=vade_unix_flags_abi",
            "--crate-type=lib",
            "--emit=metadata",
            "--target",
        ])
        .arg(&target)
        .arg(&input)
        .arg("-o")
        .arg(scratch.0.join("check.rmeta"))
        .status()
        .map_err(|error| format!("cannot compile ABI assertions: {error}"))?;
    if !status.success() {
        return Err(format!("Unix flag ABI compilation failed for {target}"));
    }
    crate::output::report(format_args!("PASS unix-flags-abi {target} constants=12 compile-only"), true);
    Ok(0)
}
