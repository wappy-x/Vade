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
// Every image targets `sm_121a`, the GB10's tensor-core target. The production image had `sm_121` until the MXFP8 word
// took the block-scaled tensor-core step, which `sm_121` refuses (driver error 218). Lead's ruling of 2026-11-16 12:25
// allows the switch; the receipt is run `yard-11` (`engine/docs/tc-dense-contract.md`, section "The image target"): the
// pinned offline compiler gives byte-equal machine code for every compared kernel of the image under the two targets.
impl Variant {
    pub fn features(self) -> &'static str {
        match self { Self::Production => "tensor-cores", Self::Fp4 => "fp4,tensor-cores", Self::Tc => "" }
    }
    fn target_cpu(self) -> &'static str {
        match self { Self::Production | Self::Fp4 | Self::Tc => "sm_121a" }
    }
    pub fn flags(self) -> &'static str {
        match self { Self::Production | Self::Fp4 | Self::Tc => "-C target-cpu=sm_121a -C target-feature=+ptx88 -C overflow-checks=off" }
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
        root.join("src/mhc_boundary.rs"), root.join("src/tp_payload.rs"), root.join("src/expert_ops.rs"), root.join("src/tc_contract.rs"), root.join("src/tc_f8.rs"), root.join("src/tc_abi.rs"), root.join("src/next_math.rs"),
        root.join("src/causal_ops.rs"), root.join("src/recurrent_rows.rs"), root.join("src/latent_ops.rs"), root.join("src/latent_absorb.rs"), root.join("src/latent_absorb_rows.rs"), root.join("src/absorb_tc.rs"), root.join("src/prompt_ops.rs"), root.join("src/mxfp8_quantize.rs"),
        root.join("src/resident_launch.rs"), root.join("src/device_route.rs"), root.join("src/prompt_route.rs"), root.join("src/bus_shadow.rs"), root.join("src/tp_put_wide.rs"), root.join("src/tp_put_join.rs"), root.join("src/pair_split.rs"), root.join("src/plain_admission.rs"), root.join("src/prompt_admission.rs"), root.join("src/compact.rs"),
        root.join("src/ordered_projection_kernel.rs"), root.join("src/frontend_math.rs"), root.join("src/quantize_sweep.rs"), root.join("src/wide_weights.rs"), root.join("src/head_bound.rs"),
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
/// resident_launch_mx_rows_tile: the recorded wrapper of the same body needs the same cap. Without it the wrapper does not launch at all
/// (CUDA 701 on one Spark, `kernel-bench --mx-small`, run shareb-mx-wrap-07; a decode step of 8 chats stopped at its first graph).
/// latent_absorption_tile (256 threads): 128 keeps two CTAs (16 warps) per SM; unbounded it runs one. Measured on the
/// device with the same output bytes: the cap alone 1.10 to 1.19 times the speed, with the low-register form of the
/// kernel 1.22 to 1.39 (docs/latent-absorption.md). 112 and 85 spill and are slower than no cap.
/// resident_launch_mx (128 threads): the recorded one-job MXFP8 entry holds two bodies, the scalar wave body (it takes 40
/// registers) and the tensor-core row body (`mx_tc::block`, about 64). The cap was 40 (12 blocks an SM for the scalar
/// body); under it the tensor-core body spills to stack memory and the served one-row step lost 0.89 ms (traced run
/// `onerow-tr-1`: 91.0 us a launch against 67.0 us on main, local memory 704 against 576 bytes a thread). 64 keeps 8 blocks
/// an SM and no spill. Measured (`kernel-bench --mx-tc cap`, run `onerow-cap-1`, us a launch cold / after a 6 MiB
/// prefix, 2048 x 8192): tensor-core body cap 40 82.3 / 66.5, 48 80.8 / 58.4, 56 79.9 / 54.5, 64 80.0 / 54.6, none
/// 79.8 / 54.1; the scalar output head in the same entry (77,440 x 4,096) 1416.7 at 40, 1399.0 at 64, 1396.9 with none.
/// A scalar one-job layer launch is slower above 40 (77.6 to 81.3 us cold): no layer matrix is on the scalar word.
#[allow(dead_code)]
// The project block can have 1024 threads; 64 registers allow that block to launch.
const REGISTER_CAPS: [(&str, u32); 32] = [("mx_rows_tile", 160), ("resident_launch_mx_rows_tile", 160), ("resident_launch_mx", 64), ("mhc_boundary_project", 64), ("mhc_boundary_block", 128),
    // Absorbed attention on the tensor cores: 256-thread blocks, the prompt entry, the three step entries and the two
    // recorded entries. 128 registers keep two blocks (16 warps) an SM; a warp carries 48 accumulator registers.
    ("latent_absorption_tile", 128), ("latent_absorption", 128), ("latent_absorption_scores", 128),
    ("latent_absorption_values", 128), ("resident_launch_absorb_dense", 128), ("resident_launch_absorb_attention", 128),
    // The rows form's scores and values of a check pass (RB3, P7), direct and recorded: 168 registers uncapped (one block an
    // SM). Measured (kernel-bench --latent-rows, run rb3-k3, us a launch at cache 1,024): values of 4 rows 52.7 -> 37.1,
    // of 8 rows 77.0 -> 67.9 with the cap; of 1 and 2 rows 29.8 -> 32.6 and 30.0 -> 33.4. Its dense entry takes 106, no cap.
    ("latent_rows_attention", 128), ("resident_launch_latent_rows_attention", 128),
    // Recorded row selection uses the same 1024-thread bound.
    ("resident_launch_prompt_select_rows", 64),
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
    /// The recorded one-job MXFP8 entry also holds the tensor-core row body (about 64 registers): a lower cap spills it.
    #[test]
    fn the_one_job_mx_entry_cap_fits_the_tensor_core_body_and_eight_blocks() {
        let cap=REGISTER_CAPS.iter().find(|(name,_)|*name=="resident_launch_mx").expect("the one-job MXFP8 entry has a register cap").1;
        assert!(cap>=64 && 8*128*cap<=65_536,"cap {cap}");
    }
    #[test]
    fn latent_tile_cap_keeps_two_blocks_an_sm() {
        // An SM has 65,536 registers (the other entries of the table). Two 256-thread blocks need 128 or fewer a thread.
        let cap=REGISTER_CAPS.iter().find(|(name,_)|*name=="latent_absorption_tile").expect("the latent tile has a register cap").1;
        assert!(2*256*cap<=65_536 && 3*256*cap>65_536,"cap {cap}");
        // The stamp writes the cap between the entry's parameter list and its body.
        let image:String=REGISTER_CAPS.iter().map(|(name,_)|format!(".visible .entry {name}(\n\t.param .b64 p\n)\n{{\n\tret;\n}}\n")).collect();
        let capped=String::from_utf8(cap_registers(image.into_bytes()).unwrap()).unwrap();
        let entry=capped.find(".entry latent_absorption_tile(").unwrap();let body=entry+capped[entry..].find("\n{").unwrap();
        assert!(capped[entry..body].ends_with(&format!(")\n.maxnreg {cap}")),"{}",&capped[entry..body]);
    }
    #[test]
    fn kernel_external_modules_are_part_of_the_ptx_identity() {
        let manifest=Path::new(env!("CARGO_MANIFEST_DIR"));
        let root=[manifest.to_path_buf(),manifest.join(".."),manifest.join("../gpu")].into_iter()
            .find(|p|p.join("ptx_stamp.rs").is_file() && p.join("kernels/src/lib.rs").is_file()).unwrap().canonicalize().unwrap();
        let bound=source_files(&root).unwrap().into_iter().map(|p|p.canonicalize().unwrap()).collect::<Vec<_>>();
        // Every file of the kernel crate, not only its root: an include by path in a nested module is part of the image too.
        let kernels=root.join("kernels/src");let mut missing=Vec::new();
        for kernel in bound.iter().filter(|p|p.starts_with(&kernels) && p.extension().is_some_and(|e|e=="rs")) {
            let source=fs::read_to_string(kernel).unwrap();
            for declaration in source.split("#[path").skip(1) {
                let Some(relative)=declaration.split('"').nth(1) else {continue;};
                let Ok(file)=kernel.parent().unwrap().join(relative).canonicalize() else {continue;};
                if file.starts_with(root.join("src")) && !bound.contains(&file) {missing.push(file.strip_prefix(&root).unwrap().to_owned());}
            }
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
