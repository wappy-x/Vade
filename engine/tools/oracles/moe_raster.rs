//! Host ownership/byte oracle for a grid-only transformation; no CUDA emulation.
//! Source equality binds the unchanged arithmetic to the exact integration base:
//! the down-writes tree (a116627) the raster change was integrated onto.
use std::process::Command;

const BASE: &str = "a116627f8764bdd34528b383a2c77ad1e750757c";
// Historical comparison tree; later work edits both compared files again (A staging, one CTA per token in
// flash_moe_route), and their own byte gates prove those changes, not this oracle.
const RASTER: &str = "500c408538851de99a7da4d9b7aa707c535be017";

fn tree(rev: &str, path: &str) -> Result<String, String> {
    let old = Command::new("git")
        .args(["show", &format!("{rev}:{path}")])
        .output()
        .map_err(|e| e.to_string())?;
    if !old.status.success() {
        return Err("cannot read pinned baseline".into());
    }
    String::from_utf8(old.stdout).map_err(|e| e.to_string())
}

fn baseline(path: &str) -> Result<String, String> {
    tree(BASE, path)
}

#[test]
fn kernel_body_is_identical_after_coordinate_substitution() -> Result<(), String> {
    let old = baseline("engine/crates/gpu/kernels/src/fp4_prefill_tc.rs")?;
    let new = tree(RASTER, "engine/crates/gpu/kernels/src/fp4_prefill_tc.rs")?
        .replace("    let bx = _block_idx_x() as usize;\n", "")
        .replace("(bx % 4) * 32", "_block_idx_z() as usize * 32")
        .replace("(bx / 4) * 128", "_block_idx_x() as usize * 128");
    let code = |s: &str| {
        s.lines()
            .filter(|s| !s.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(code(&old), code(&new), "CTA arithmetic/epilogue changed");
    Ok(())
}

#[test]
fn host_launch_changes_are_wide_only_decode_is_identical() -> Result<(), String> {
    // The integration retired the dead `compact_down` field; its arena slot stays.
    let old = baseline("engine/crates/engine/src/flash/next_execution.rs")?
        .replace("    compact_down: DeviceVector<'a>,\n", "")
        .replace(
            "            compact_down: v.next().unwrap(),\n            bounds: v.next().unwrap(),",
            "            bounds: v.nth(1).unwrap(),",
        );
    let new = tree(RASTER, "engine/crates/engine/src/flash/next_execution.rs")?
        .replace(
            "let planes = if wide {\n            1",
            "let planes = if wide {\n            4",
        )
        .replace(
            "                if wide { 20 } else { 40 },\n                tasks as u32,",
            "                if wide { 5 } else { 40 },\n                tasks as u32,",
        )
        .replace(
            "                if wide { 80 } else { 160 },\n                tasks as u32,",
            "                if wide { 20 } else { 160 },\n                tasks as u32,",
        );
    let compact = |s: &str| {
        s.lines()
            .filter(|s| !s.trim_start().starts_with("//"))
            .flat_map(str::chars)
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
    };
    assert_eq!(
        compact(&old),
        compact(&new),
        "non-grid host execution changed"
    );
    Ok(())
}

fn output(m: usize, n: usize, adjacent: bool) -> Vec<u32> {
    let mut out = vec![u32::MAX; m * n];
    for first in (0..m).step_by(128) {
        for linear in 0..n / 128 * 4 {
            // Baseline grid order has N in X, planes in Z; X folds plane into X.
            let (plane, tile) = if adjacent {
                (linear % 4, linear / 4)
            } else {
                (linear / (n / 128), linear % (n / 128))
            };
            let mb = first + plane * 32;
            if mb >= m {
                continue;
            }
            // Match the actual epilogue's warp/J/I/lane and pair ownership.
            for warp in 0..4 {
                for j in 0..4 {
                    for i in 0..2 {
                        for lane in 0..32 {
                            for r in [mb + i * 16 + lane / 4, mb + i * 16 + lane / 4 + 8] {
                                if r >= m {
                                    continue;
                                }
                                for col in 0..2 {
                                    let c = tile * 128 + (warp * 4 + j) * 8 + lane % 4 * 2 + col;
                                    let at = r * n + c;
                                    assert_eq!(out[at], u32::MAX, "overlapping output owner");
                                    // Arbitrary distinct output bits: covers every live byte owner.
                                    out[at] = at as u32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(out.iter().all(|&v| v != u32::MAX), "missing output owner");
    out
}

#[test]
fn routed_8k_bytes_chunks_512_2048_and_tails() {
    for chunk in [512, 2048] {
        let mut launches = 0;
        for first in (0..8192).step_by(chunk) {
            let mut counts = [0usize; 512];
            // Ten distinct experts per token; skew also crosses M128 task bounds.
            for token in first..first + chunk {
                for slot in 0..10 {
                    let e = if token % 7 == 0 {
                        slot
                    } else {
                        (token * 17 + slot * 41) % 512
                    };
                    counts[e] += 1;
                }
            }
            assert_eq!(counts.iter().sum::<usize>(), chunk * 10);
            for m in counts {
                for n in [640, 2560] {
                    assert_eq!(
                        output(m, n, false),
                        output(m, n, true),
                        "chunk={chunk} m={m} n={n}"
                    );
                }
            }
            launches += 48;
        }
        assert_eq!(launches, 48 * 8192 / chunk);
    }
    for m in [
        0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2048,
    ] {
        for n in [640, 2560] {
            assert_eq!(output(m, n, false), output(m, n, true));
        }
    }
}
