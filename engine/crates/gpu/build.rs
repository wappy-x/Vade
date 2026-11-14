use std::{env, fs, path::PathBuf};
#[allow(dead_code)] // Kernel-builder methods share the variant definition.
mod ptx_stamp;
mod cubin;

fn main() {
    if let Err(error) = run() {
        // Paths and OS errors may contain line breaks. Keep producer diagnostics
        // to one plain line, with no panic or backtrace framing.
        eprintln!("GPU build: {}", error.split_whitespace().collect::<Vec<_>>().join(" "));
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing CARGO_MANIFEST_DIR")?);
    let variant = ptx_stamp::host_variant().map_err(|error| error.to_string())?;
    let ptx = env::var_os("SPARK_LAYER_PTX").map(PathBuf::from).unwrap_or_else(||
        variant.ptx_path(&manifest, true));
    println!("cargo:rerun-if-env-changed=SPARK_LAYER_PTX");
    println!("cargo:rerun-if-changed={}", ptx.display());
    let linux = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let mut bytes = match fs::read(&ptx) {
        Ok(bytes) if !bytes.is_empty() => {
            ptx_stamp::verify(&manifest, &ptx, &bytes, variant).map_err(|error|
                format!("PTX validation failed: {error}; run `{}` first", variant.build_command()))?;
            bytes
        },
        _ if matches!(variant, ptx_stamp::Variant::Production) && !linux => Vec::new(),
        _ => return Err(format!("missing or empty layer PTX at {}; run `{}` first", ptx.display(), variant.build_command())),
    };
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    cubin::prepare(&out, &bytes).map_err(|error| format!("Prebuilt CUDA image: {error}"))?;
    bytes.push(0);
    fs::write(out.join("layer.ptx"), bytes).map_err(|error| format!("write embedded PTX: {error}"))?;
    // The second image: the tensor-core family (`cargo kernels --tc`, sm_121a). A Linux build needs it, like the first
    // image: prompt groups of the routed experts run on it, and a build without it would serve them on the generic
    // one-thread kernel with no error. Off Linux (no device) it may be absent. A stale one fails every build.
    let tc = ptx_stamp::Variant::Tc.ptx_path(&manifest, true);
    println!("cargo:rerun-if-changed={}", tc.display());
    let mut tc_bytes = match fs::read(&tc) {
        Ok(bytes) if !bytes.is_empty() => {
            ptx_stamp::verify(&manifest, &tc, &bytes, ptx_stamp::Variant::Tc).map_err(|error|
                format!("tensor-core PTX validation failed: {error}; run `cargo kernels --tc` first"))?;
            bytes
        },
        _ if !linux => Vec::new(),
        _ => return Err(format!("missing or empty tensor-core PTX at {}; run `cargo kernels --tc` first", tc.display())),
    };
    tc_bytes.push(0);
    fs::write(out.join("tc.ptx"), tc_bytes).map_err(|error| format!("write embedded tensor-core PTX: {error}"))
}
