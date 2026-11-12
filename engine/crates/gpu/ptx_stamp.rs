//! Bind a built PTX artifact to its build variant, kernel sources and launch ABI.
use std::{env, fs, io, path::{Path, PathBuf}};
#[allow(clippy::duplicate_mod)] // the shared file is included on purpose.
#[path = "../../tools/export/src/sha256.rs"]
mod sha256;

const PTX_VERSION: &str = "8.8";
#[derive(Clone, Copy)]
/// `Tc` is not a variant of the host build: it is the second image (the tensor-core family, crate `spark-tc-kernels`),
/// built beside the production image and loaded as a second module.
#[derive(PartialEq)]
pub enum Variant { Production, Fp4, Tc }
impl Variant {
    pub fn features(self) -> &'static str {
        match self { Self::Production => "tensor-cores", Self::Fp4 => "fp4,tensor-cores", Self::Tc => "" }
    }
    fn target_cpu(self) -> &'static str {
        match self { Self::Production => "sm_121", Self::Fp4 | Self::Tc => "sm_121a" }
    }
    pub fn flags(self) -> &'static str {
        match self {
            Self::Production => "-C target-cpu=sm_121 -C overflow-checks=off",
            Self::Fp4 | Self::Tc => "-C target-cpu=sm_121a -C target-feature=+ptx88 -C overflow-checks=off",
        }
    }
    pub fn target_dir(self) -> &'static str {
        match self { Self::Production => "../../target", Self::Fp4 => "../../target/fp4", Self::Tc => "../../target/tc" }
    }
    pub fn ptx_path(self, root: &Path, layer: bool) -> PathBuf {
        let name = match (self, layer) {
            (Self::Production, false) => "spark_rust_kernels.ptx",
            (Self::Production, true) => "spark_layer_kernels.ptx",
            (Self::Fp4, false) => "spark_rust_kernels_fp4.ptx",
            (Self::Fp4, true) => "spark_layer_kernels_fp4.ptx",
            // The crate's own output name, and the stamped copy the host embeds as its second image.
            (Self::Tc, false) => "spark_tc_kernels.ptx",
            (Self::Tc, true) => "spark_tc_layer.ptx",
        };
        root.join(self.target_dir()).join("nvptx64-nvidia-cuda/release").join(name)
    }
    pub fn build_command(self) -> &'static str {
        match self { Self::Production => "cargo kernels", Self::Fp4 => "cargo kernels --fp4", Self::Tc => "cargo kernels --tc" }
    }
    /// The kernel crate of the image.
    pub fn package(self) -> &'static str {
        match self { Self::Production | Self::Fp4 => "spark-rust-kernels", Self::Tc => "spark-tc-kernels" }
    }
    fn identity(self) -> String {
        format!("features={}\ntarget-cpu={}\nptx-version={PTX_VERSION}\nrustflags={}\n", self.features(), self.target_cpu(), self.flags())
    }
}
pub fn host_variant() -> io::Result<Variant> {
    println!("cargo:rerun-if-env-changed=SPARK_PTX_VARIANT");
    match env::var("SPARK_PTX_VARIANT") {
        Err(env::VarError::NotPresent) => Ok(Variant::Production),
        Ok(value) if value == "production" => Ok(Variant::Production),
        Ok(value) if value == "fp4" => Ok(Variant::Fp4),
        _ => Err(io::Error::new(io::ErrorKind::InvalidInput, "SPARK_PTX_VARIANT must be production or fp4")),
    }
}
fn validate_variant(bytes: &[u8], variant: Variant) -> io::Result<()> {
    let text = std::str::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    for (directive, expected) in [(".version ", PTX_VERSION), (".target ", variant.target_cpu())] {
        let actual = text.lines().find_map(|line| line.trim().strip_prefix(directive))
            .and_then(|value| value.split(|c: char| c.is_whitespace() || c == ',').next());
        if actual != Some(expected) {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("PTX variant mismatch: expected {directive}{expected}, got {actual:?}")));
        }
    }
    Ok(())
}

fn source_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() { visit(&path, files)?; }
            else if path.extension().is_some_and(|e| e == "rs") { files.push(path); }
        }
        Ok(())
    }
    let mut files = vec![root.join("kernels/Cargo.toml"), root.join("kernel_params.rs"), root.join("src/fp8_abi.rs"), root.join("src/fp4_abi.rs"),
        root.join("src/projection_params.rs"), root.join("src/gdn_params.rs"), root.join("src/next_params.rs"), root.join("src/router_abi.rs"), root.join("src/mhc_params.rs"),
        root.join("src/mhc_boundary.rs"), root.join("src/tp_payload.rs"), root.join("src/expert_ops.rs"), root.join("src/tc_contract.rs"), root.join("src/tc_abi.rs"), root.join("src/next_math.rs"),
        root.join("src/causal_ops.rs"), root.join("src/latent_ops.rs"), root.join("src/latent_absorb.rs"), root.join("src/latent_absorb_rows.rs"), root.join("src/prompt_ops.rs"), root.join("src/mxfp8_quantize.rs"),
        root.join("src/resident_launch.rs"), root.join("src/device_route.rs"), root.join("src/bus_shadow.rs"), root.join("src/tp_put_wide.rs"), root.join("src/tp_put_join.rs"), root.join("src/pair_split.rs"), root.join("src/plain_admission.rs"), root.join("src/prompt_admission.rs"), root.join("src/compact.rs"),
        root.join("src/ordered_projection_kernel.rs"), root.join("src/frontend_math.rs"), root.join("src/quantize_sweep.rs"),
        root.join("build-kernels/src/main.rs"), root.join("ptx_stamp.rs")];
    visit(&root.join("kernels/src"), &mut files)?;
    files.sort();
    Ok(files)
}
/// The sources that make the tensor-core image: its crate, the kernel files it includes by path, the shared ABI and
/// contract files, and the build tools.
fn tc_source_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = vec![root.join("kernels-tc/Cargo.toml"), root.join("kernels/src/ptx.rs"), root.join("kernels/src/fp4_cohort.rs"),
        root.join("src/next_math.rs"), root.join("src/tc_contract.rs"), root.join("src/tc_abi.rs"),
        root.join("build-kernels/src/main.rs"), root.join("ptx_stamp.rs")];
    for entry in fs::read_dir(root.join("kernels-tc/src"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "rs") { files.push(path); }
    }
    files.sort();
    Ok(files)
}
fn files_of(root: &Path, variant: Variant) -> io::Result<Vec<PathBuf>> {
    if variant == Variant::Tc { tc_source_files(root) } else { source_files(root) }
}
fn digest(root: &Path, files: &[PathBuf], ptx: &[u8], variant: Variant) -> io::Result<String> {
    let mut input = variant.identity().into_bytes();
    for path in files {
        let name = path.strip_prefix(root).unwrap().to_string_lossy();
        let bytes = fs::read(path)?;
        input.extend_from_slice(&(name.len() as u64).to_le_bytes());
        input.extend_from_slice(name.as_bytes());
        input.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        input.extend_from_slice(&bytes);
    }
    input.extend_from_slice(&(ptx.len() as u64).to_le_bytes());
    input.extend_from_slice(ptx);
    Ok(sha256::digest(&input))
}
#[allow(dead_code)] // Used by `cargo kernels`; shared by the host build scripts.
pub fn stamp(root: &Path, ptx: &Path, variant: Variant) -> io::Result<()> {
    let bytes = fs::read(ptx)?;
    if bytes.is_empty() { return Err(io::Error::new(io::ErrorKind::InvalidData, "empty PTX")); }
    validate_variant(&bytes, variant)?;
    let bytes = if variant == Variant::Tc { bytes } else { cap_registers(bytes)? };
    fs::write(ptx, &bytes)?;
    fs::write(ptx.with_extension("ptx.sha256"),
        format!("{}sha256={}\n", variant.identity(), digest(root, &files_of(root, variant)?, &bytes, variant)?))
}
/// Four-warp, 32 KiB flash-prefill tiles fit three per SM (12 warps) only at
/// <= 168 registers; unbounded ptxas takes 170-174, which leaves two per SM.
/// Rust has no launch-bounds attribute, so the stamp adds `.maxnreg`.
/// flash_bf16_mma_latency: 128 threads x 128 registers = four CTAs (16 warps) per SM, so the
/// 192-CTA persistent grid is one wave (unbounded ptxas takes 134-136 -> three per SM).
/// fp4_prefill_tc_down (256 threads): 128 keeps two CTAs (16 warps) per SM.
/// mx_rows_tile (384 threads): 160 registers fit one 12-warp block per SM (unbounded ptxas takes 200: no 384-thread block would launch).
#[allow(dead_code)]
// The project block can have 1024 threads; 64 registers allow that block to launch.
const REGISTER_CAPS: [(&str, u32); 23] = [("mx_rows_tile", 160), ("resident_launch_mx", 40), ("mhc_boundary_project", 64),
    // Absorbed attention values: 1024-thread blocks, eager and recorded (the recorded entry also holds the 128-thread
    // scores arm). 64 registers is the most a 1024-thread block can launch with; the driver takes 40 and 48 today.
    ("latent_absorption_values", 64), ("resident_launch_absorb_attention", 64),
    ("flash_bf16_mma_latency", 128), ("flash_bf16_mma_latency_group", 128), ("gemm_fp8_block128", 128), ("fp8_routed_gate", 128), ("fp8_routed_up", 128),
    ("fp8_routed_down", 128), ("fp4_prefill_tc_down", 128), ("prefill_attention_swizzled_packed_bf16", 168),
    ("prefill_attention_swizzled_bf16", 168), ("prefill_attention_gqa4_fragments", 168), ("prefill_attention_gqa4_bf16", 168),
    // Chunk scan: one 512-thread CTA per value head needs <= 128 registers.
    ("flash_next_gdn_chunk_scan", 128), ("flash_next_gdn_chunk_scan_cv", 128),
    // Tensor-core index scores (VADE_QSA_TC): 128 registers fit four 24 KB CTAs per SM.
    ("flash_qsa_scores_tc", 128), ("flash_qsa_scores_tc_m", 128), ("flash_qsa_scores_tc_batch", 128),
    // Warp-slot TC attention scores: 24 KB Q planes, four CTAs per SM.
    ("flash_qsa_attention_scores_tc_w", 128),
    // Fused TC prompt attention: three CTAs per SM at <= 168.
    ("flash_qsa_attention_fused_tc", 168)];
#[allow(dead_code)]
fn cap_registers(bytes: Vec<u8>) -> io::Result<Vec<u8>> {
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidData, e);
    let mut text = String::from_utf8(bytes).map_err(|_| invalid("PTX is not UTF-8".into()))?;
    for (name, registers) in REGISTER_CAPS {
        let entry = text.find(&format!(".entry {name}(")).ok_or_else(|| invalid(format!("missing entry {name}")))?;
        // Performance directives go between the parameter list and the body.
        let body = entry + text[entry..].find("\n{").ok_or_else(|| invalid(format!("no body for {name}")))?;
        if !text[entry..body].contains(".maxnreg") { text.insert_str(body + 1, &format!(".maxnreg {registers}\n")); }
    }
    Ok(text.into_bytes())
}
pub fn verify(root: &Path, ptx: &Path, bytes: &[u8], variant: Variant) -> io::Result<()> {
    validate_variant(bytes, variant)?;
    let files = files_of(root, variant)?;
    println!("cargo:rerun-if-changed={}", root.join(if variant == Variant::Tc { "kernels-tc/src" } else { "kernels/src" }).display());
    for file in &files { println!("cargo:rerun-if-changed={}", file.display()); }
    let stamp = ptx.with_extension("ptx.sha256");
    println!("cargo:rerun-if-changed={}", stamp.display());
    let expected = format!("{}sha256={}\n", variant.identity(), digest(root, &files, bytes, variant)?);
    if fs::read_to_string(stamp)? != expected {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "stale PTX, kernel ABI or build variant"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernel_external_modules_are_part_of_the_ptx_identity() {
        let manifest=Path::new(env!("CARGO_MANIFEST_DIR"));
        let root=[manifest.to_path_buf(),manifest.join(".."),manifest.join("../gpu")].into_iter()
            .find(|p|p.join("ptx_stamp.rs").is_file() && p.join("kernels/src/lib.rs").is_file()).unwrap().canonicalize().unwrap();
        let bound=source_files(&root).unwrap().into_iter().map(|p|p.canonicalize().unwrap()).collect::<Vec<_>>();
        let kernel=root.join("kernels/src/lib.rs");let source=fs::read_to_string(&kernel).unwrap();let mut missing=Vec::new();
        for declaration in source.split("#[path").skip(1) {
            let Some(relative)=declaration.split('"').nth(1) else {continue;};
            let file=kernel.parent().unwrap().join(relative).canonicalize().unwrap();
            if file.starts_with(root.join("src")) && !bound.contains(&file) {missing.push(file.strip_prefix(&root).unwrap().to_owned());}
        }
        assert!(missing.is_empty(),"kernel source files absent from PTX identity: {missing:?}");
    }
    #[test]
    fn shared_cast_and_expert_sources_are_part_of_the_ptx_identity() {
        let manifest=Path::new(env!("CARGO_MANIFEST_DIR"));
        // This module is also included by serving build-metadata tests.
        let candidates=[manifest.to_path_buf(),manifest.join(".."),manifest.join("../gpu")];
        let root=candidates.iter().find(|p| p.join("ptx_stamp.rs").is_file() && p.join("src/mhc_boundary.rs").is_file())
            .expect("GPU source root").canonicalize().unwrap();let root=root.as_path();
        let source=root.join("src/mhc_boundary.rs");
        assert!(source_files(root).unwrap().contains(&source));
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let scratch=std::env::temp_dir().join(format!("vade-mhc-stamp-{}-{nonce}",std::process::id()));
        for file in source_files(root).unwrap() {
            let copy=scratch.join(file.strip_prefix(root).unwrap());
            fs::create_dir_all(copy.parent().unwrap()).unwrap();fs::copy(file,copy).unwrap();
        }
        for path in ["src/mhc_boundary.rs","src/expert_ops.rs","src/next_math.rs","src/resident_launch.rs","src/device_route.rs","src/tp_put_wide.rs","src/tp_put_join.rs","src/pair_split.rs","src/plain_admission.rs","src/prompt_admission.rs",
            "src/compact.rs","src/ordered_projection_kernel.rs","src/frontend_math.rs"] {
        let shared=scratch.join(path);let original=fs::read(&shared).unwrap();
        // A host cast edit must invalidate either previously stamped image.
        for variant in [Variant::Production,Variant::Fp4] {
            fs::write(&shared,&original).unwrap();
            let bytes=format!(".version {PTX_VERSION}\n.target {}\n",variant.target_cpu()).into_bytes();
            let ptx=scratch.join("image.ptx");fs::write(&ptx,&bytes).unwrap();
            let stamp=format!("{}sha256={}\n",variant.identity(),digest(&scratch,&source_files(&scratch).unwrap(),&bytes,variant).unwrap());
            fs::write(ptx.with_extension("ptx.sha256"),stamp).unwrap();
            verify(&scratch,&ptx,&bytes,variant).unwrap();
            fs::write(&shared,[original.as_slice(),b"\n// changed cast\n"].concat()).unwrap();
            assert!(verify(&scratch,&ptx,&bytes,variant).unwrap_err().to_string().contains("stale PTX"));
        }
        fs::write(&shared,&original).unwrap();
        }
        fs::remove_dir_all(scratch).unwrap();
    }
}
