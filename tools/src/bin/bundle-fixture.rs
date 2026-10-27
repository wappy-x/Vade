//! Simulated build tools for the real shell bundle producer's regression tests.
#![forbid(unsafe_code)]
use std::{env,fs,path::{Path,PathBuf},process::{Command,ExitCode},io::Write};
use vade_tools::{Result,sha256};
fn variable(key: &str) -> String { env::var(key).unwrap() }
fn write(path: &Path, data: impl AsRef<[u8]>) -> Result<()> { fs::create_dir_all(path.parent().unwrap())?; fs::write(path,data)?; Ok(()) }
fn command(mut c: Command) -> Result<()> { std::io::stdout().write_all(&vade_tools::output(&mut c)?)?; Ok(()) }
fn tool(name: &str, args: &[String]) -> Result<()> {
    let root = PathBuf::from(variable("TEST_ROOT")); let output = PathBuf::from(variable("TEST_OUTPUT")); let mode = env::var("TEST_MODE").unwrap_or_default();
    match name {
        "uname" => println!("{}",if args == ["-s"] { "Linux" } else { "aarch64" }),
        "git" => match args[0].as_str() {
            "status" => if mode == "dirty" { print!(" M tracked"); },
            "rev-parse" => println!("1234567890abcdef1234567890abcdef12345678"), "show" => println!("1700000000"),
            "archive" => { let mut c = Command::new(variable("TEST_REAL_TAR")); c.current_dir(root.join("fixture")).args(["-cf","-","."]); command(c)?; },
            _ => return Err("unexpected git invocation".into()) },
        "rustup" => (),
        "cc" => write(Path::new(&args[args.iter().position(|a| a == "-o").unwrap()+1]),b"link-only stub")?,
        "cargo" => {
            if mode == "build-fails" { std::process::exit(42); }
            if mode == "interrupt" {
                let parent = String::from_utf8(vade_tools::output(Command::new("ps").args(["-o","ppid=","-p",&std::process::id().to_string()]))?)?;
                vade_tools::output(Command::new("kill").args(["-TERM",parent.trim()]))?; return Ok(());
            }
            let target = if args.iter().any(|s| s == "kernels") {
                if matches!(mode.as_str(),"destination"|"empty-destination") {
                    fs::create_dir(&output)?; if mode == "destination" { fs::write(output.join("existing"),"keep me")?; }
                }
                if mode == "concurrent" {
                    let r = Command::new("sh").arg(root.join("scripts/make-bundle.sh")).arg(&output).output()?;
                    assert!(!r.status.success() && String::from_utf8_lossy(&r.stderr).contains("reserved"));
                }
                env::current_dir()?.join("target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx")
            } else { PathBuf::from(variable("CARGO_TARGET_DIR")).join("release").join(&args[args.iter().position(|s| s == "--bin").unwrap()+1]) };
            let current = env::current_dir()?;
            vade_tools::output(Command::new(variable("RUSTC_WRAPPER")).arg(root.join("bin/rustc-probe")).args(["--crate-name","probe"])
                .env("CARGO_MANIFEST_DIR",&current).env("CARGO_MANIFEST_PATH",current.join("Cargo.toml")).env("TEST_ARTIFACT",target))?;
        },
        "rustc-probe" => {
            let source = variable("VADE_BUNDLE_SOURCE"); let work = variable("VADE_BUNDLE_BUILD");
            assert!(variable("CARGO_MANIFEST_DIR").starts_with("./")); assert!(variable("CARGO_MANIFEST_PATH").starts_with("./"));
            assert_eq!(&args[args.len()-3..],&[format!("--remap-path-prefix={}=/build-home",variable("HOME")),
                format!("--remap-path-prefix={work}=/vade-build"),format!("--remap-path-prefix={source}=.")]);
            let mut data = format!("portable artifact {}",variable("CARGO_MANIFEST_DIR"));
            if mode == "path-leak" { data.push_str(&work); }
            if mode == "remapped-path-leak" { data.push_str(Path::new(&work).file_name().unwrap().to_str().unwrap()); }
            write(Path::new(&variable("TEST_ARTIFACT")),data)?;
        },
        "readelf" => if args[0] == "-h" { println!("Class: ELF64\nMachine: {}",if mode == "wrong-arch" { "X86-64" } else { "AArch64" }); }
            else { println!("(NEEDED) Shared library: [libcuda.so.1]"); if mode == "rpath" { println!("(RUNPATH) producer-path"); } },
        "tar" => {
            let mut c = Command::new(variable("TEST_REAL_TAR"));
            if args.iter().any(|a| a == "-xf") { c.args(args); }
            else { let position = args.iter().position(|a| a == "-cf").unwrap(); c.arg("--format=ustar").args(&args[position..]); }
            command(c)?;
        },
        "zstd" => if args.iter().any(|a| a == "-d") { std::io::stdout().write_all(&fs::read(args.last().unwrap())?)?; }
            else { fs::copy(&args[args.iter().position(|a| a == "-10").unwrap()+1],&args[args.iter().position(|a| a == "-o").unwrap()+1])?; },
        "sha256sum" => if args[0] == "-c" {
            for line in fs::read_to_string(&args[1])?.lines() { let (digest,file) = line.split_once("  ").unwrap();
                assert_eq!(sha256::digest_reader(fs::File::open(file)?)?,digest); println!("{file}: OK"); }
        } else { println!("{}  {}",sha256::digest_reader(fs::File::open(&args[0])?)?,args[0]); },
        "mv" => {
            if matches!(mode.as_str(),"rename-race"|"empty-rename-race") { fs::create_dir(&output)?;
                if mode == "rename-race" { fs::write(output.join("existing"),"keep me")?; } }
            if cfg!(target_os = "macos") { let source = &args[args.len()-2]; let dest = &args[args.len()-1];
                if fs::symlink_metadata(dest).is_err() { fs::rename(source,dest)?; }
            } else { let mut c = Command::new(variable("TEST_REAL_MV")); c.args(args); command(c)?; }
        },
        _ => return Err(format!("unexpected fake build tool: {name}").into())
    } Ok(())
}
fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    match tool(&args[0],&args[1..]) { Ok(()) => ExitCode::SUCCESS, Err(e) => { eprintln!("fixture: {e}"); ExitCode::FAILURE } }
}
