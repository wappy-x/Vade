use std::{fs, path::PathBuf, process::Command, time::{SystemTime, UNIX_EPOCH}};
#[allow(dead_code)]
#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;

struct Scratch(PathBuf);
impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

#[test]
fn producer_validation_and_io_errors_exit_with_one_plain_diagnostic() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scratch = Scratch(std::env::temp_dir().join(format!("gpu-build-errors-{}-{}",
        std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos())));
    fs::create_dir(&scratch.0).unwrap();
    let executable = scratch.0.join("gpu-build");
    let compiled = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2021", "-o"]).arg(&executable).arg(manifest.join("build.rs"))
        .output().unwrap();
    assert!(compiled.status.success(), "{}", String::from_utf8_lossy(&compiled.stderr));
    let invalid = scratch.0.join("invalid.cubin");
    fs::write(&invalid, include_bytes!("../fixtures/sm_121.elf-header")).unwrap();
    fs::write(invalid.with_extension("cubin.conf"), "bad\nmanifest\n").unwrap();
    for (path, expected) in [(invalid, "invalid cubin manifest"),
        (scratch.0.join("absent\nimage.cubin"), "No such file")] {
        let output = Command::new(&executable)
            .env("CARGO_MANIFEST_DIR", &manifest).env("OUT_DIR", &scratch.0)
            .env("CARGO_CFG_TARGET_OS", "macos").env("SPARK_PTX_VARIANT", "production")
            .env("SPARK_LAYER_PTX", scratch.0.join("absent.ptx")).env("SPARK_CUBIN", path)
            .env("RUST_BACKTRACE", "1").output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(stderr.lines().count(), 1, "{stderr}");
        assert!(stderr.starts_with("GPU build: Prebuilt CUDA image: "), "{stderr}");
        assert!(stderr.contains(expected), "{stderr}");
        assert!(!stderr.contains("panicked") && !stderr.contains("backtrace"), "{stderr}");
    }
}

// The packager reads this stamp in the binary, so it must exist exactly when an image does.
#[test]
fn engine_stamp_names_the_embedded_image_and_is_empty_without_one() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scratch = Scratch(std::env::temp_dir().join(format!("gpu-build-stamp-{}-{}",
        std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos())));
    fs::create_dir(&scratch.0).unwrap();
    let executable = scratch.0.join("gpu-build");
    let compiled = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2021", "-o"]).arg(&executable).arg(manifest.join("build.rs")).output().unwrap();
    assert!(compiled.status.success(), "{}", String::from_utf8_lossy(&compiled.stderr));
    let image = include_bytes!("../fixtures/sm_121.elf-header");
    let cubin = scratch.0.join("flash.cubin");
    fs::write(&cubin, image).unwrap();
    // This host build sees no PTX, so the bound PTX bytes are empty.
    fs::write(cubin.with_extension("cubin.conf"), format!("format=vade-cubin-v1\ntarget=sm_121\nptx_sha256={}\ncubin_sha256={}\n\
        min_driver_api=13000\nptxas_version=13.0.88\nptxas_sha256=a1941a04ca4fd233b2fbe50c625b1e72b3d5f79ebe80209a272c85482dfbb487\n\
        ptxas_args=--gpu-name=sm_121 -m64 -O3\n", sha256::digest(b""), sha256::digest(image))).unwrap();
    let build = |cubin: Option<&PathBuf>| {
        let mut command = Command::new(&executable);
        command.env("CARGO_MANIFEST_DIR", &manifest).env("OUT_DIR", &scratch.0).env("CARGO_CFG_TARGET_OS", "macos")
            .env("SPARK_PTX_VARIANT", "production").env("SPARK_LAYER_PTX", scratch.0.join("absent.ptx")).env_remove("SPARK_CUBIN");
        if let Some(cubin) = cubin { command.env("SPARK_CUBIN", cubin); }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        fs::read_to_string(scratch.0.join("cubin_stamp.rs")).unwrap()
    };
    assert_eq!(build(Some(&cubin)), format!("\"cubin_sha256={}\"\n", sha256::digest(image)));
    assert_eq!(build(None), "\"\"\n");
}
