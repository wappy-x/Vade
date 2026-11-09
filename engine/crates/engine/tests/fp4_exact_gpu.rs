#![cfg(target_os = "linux")]
//! Fresh-GB10 gate: fp4_exact_project/fp4_exact_swiglu vs the CPU oracle
//! (project_ref/swiglu_ref), bit-for-bit, on synthetic tensors with the
//! FP4-Local expert shapes and CSF-encoded scales (exceptions included).
//! Run on a Spark/cloud GB10 after `cargo kernels` (see the job report).
use engine_model::flash::nvfp4::{csf_expand, f32_to_bf16, project_ref, swiglu_ref};
use spark_gpu_probe::driver::{Backend, DeviceVector, with_backend};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
    }
    fn byte(&mut self) -> u8 { (self.next() >> 32) as u8 }
    fn bf16(&mut self) -> u16 { f32_to_bf16((self.next() % 2000) as f32 / 500.0 - 2.0) }
}

fn csf_encode(scales: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<u32>) {
    let (scale_cols, fixed_row) = (cols / 16, cols / 16 * 8 + 16);
    let mut fixed = vec![0u8; rows / 16 * fixed_row];
    let mut exceptions = Vec::new();
    for row in 0..rows {
        let (strip, i, base) = (row / 16, row % 16, scales[row * scale_cols].min(240));
        fixed[strip * fixed_row + i] = base;
        for c in 0..scale_cols {
            let code = scales[row * scale_cols + c];
            let delta = code as i32 - base as i32;
            let nibble = if (0..=15).contains(&delta) { delta as u8 }
                else { exceptions.push(((code as u32) << 24) | (row * scale_cols + c) as u32); 0 };
            let at = strip * fixed_row + 16 + i * (scale_cols / 2) + c / 2;
            fixed[at] |= if c % 2 == 0 { nibble } else { nibble << 4 };
        }
    }
    (fixed, exceptions)
}

struct Matrix<'x> {
    packed: Vec<u8>, scales: Vec<u8>, global: f32,
    dev_packed: DeviceVector<'x>, dev_scales: DeviceVector<'x>,
}

fn matrix<'x>(b: &Backend<'x>, rng: &mut Rng, rows: usize, cols: usize) -> Result<Matrix<'x>, String> {
    let packed: Vec<_> = (0..rows * cols / 2).map(|_| rng.byte()).collect();
    let raw: Vec<_> = (0..rows * (cols / 16)).map(|_| (rng.next() % 127) as u8).collect();
    let (fixed, exceptions) = csf_encode(&raw, rows, cols);
    assert!(!exceptions.is_empty());
    let scales = csf_expand(&fixed, &exceptions, rows, cols).map_err(|e| e.to_string())?;
    assert_eq!(scales, raw);
    let mut dev_packed = b.vector(packed.len().div_ceil(4))?;
    b.write_vector_bytes(&mut dev_packed, &packed)?;
    let mut dev_scales = b.vector(scales.len().div_ceil(4))?;
    b.write_vector_bytes(&mut dev_scales, &scales)?;
    Ok(Matrix { packed, scales, global: 0.0078125, dev_packed, dev_scales })
}

fn project(b: &Backend<'_>, m: &Matrix<'_>, input: &DeviceVector<'_>, tokens: usize,
    rows: usize, cols: usize) -> Result<Vec<u32>, String> {
    let mut out = b.vector(tokens * rows)?;
    unsafe {
        b.fp4_exact_project(input, &m.dev_packed, &m.dev_scales, m.global,
            tokens, rows, cols, &mut out)?;
    }
    Ok(b.read_vector_values(&out)?.into_iter().map(f32::to_bits).collect())
}

fn pack_bf16(bits: &[u16]) -> Vec<f32> {
    bits.chunks_exact(2).map(|p| f32::from_bits(p[0] as u32 | ((p[1] as u32) << 16))).collect()
}

#[test]
#[ignore = "skipped: no GPU; run explicitly with --ignored on a qualified GPU"]
fn fp4_exact_expert_matches_cpu_oracle_bit_for_bit() -> Result<(), String> {
    if std::env::var_os("VADE_FP4_EXACT_GPU").is_none() {
        return Err("explicit GPU gate requires VADE_FP4_EXACT_GPU".into());
    }
    with_backend(0, spark_gpu_probe::layer_ptx()?, 256 * 1024 * 1024, |b| {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let gate = matrix(b, &mut rng, 640, 2560)?;
        let up = matrix(b, &mut rng, 640, 2560)?;
        let down = matrix(b, &mut rng, 2560, 640)?;
        for tokens in [1usize, 16, 128] {
            let input: Vec<_> = (0..tokens * 2560).map(|_| rng.bf16()).collect();
            let mut dev_in = b.vector(tokens * 1280)?;
            b.write_vector(&mut dev_in, &pack_bf16(&input))?;
            let (gate_bits, up_bits) = (project(b, &gate, &dev_in, tokens, 640, 2560)?,
                project(b, &up, &dev_in, tokens, 640, 2560)?);
            let (mut gate_ref, mut up_ref) = (vec![0.0; tokens * 640], vec![0.0; tokens * 640]);
            project_ref(&input, &gate.packed, &gate.scales, gate.global, 640, 2560, &mut gate_ref)
                .map_err(|e| e.to_string())?;
            project_ref(&input, &up.packed, &up.scales, up.global, 640, 2560, &mut up_ref)
                .map_err(|e| e.to_string())?;
            assert_eq!(gate_bits, gate_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "gate tokens={tokens}");
            assert_eq!(up_bits, up_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "up tokens={tokens}");
            let (mut dev_gate, mut dev_up, mut dev_mid) =
                (b.vector(tokens * 640)?, b.vector(tokens * 640)?, b.vector(tokens * 320)?);
            b.write_vector(&mut dev_gate, &gate_ref)?;
            b.write_vector(&mut dev_up, &up_ref)?;
            unsafe { b.fp4_exact_swiglu(&dev_gate, &dev_up, tokens * 320, &mut dev_mid)?; }
            let mid_gpu = b.read_vector_values(&dev_mid)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            let middle: Vec<_> = gate_ref.iter().zip(&up_ref).map(|(&g, &u)| swiglu_ref(g, u)).collect();
            assert_eq!(mid_gpu, pack_bf16(&middle).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "swiglu tokens={tokens}");
            let down_bits = project(b, &down, &dev_mid, tokens, 2560, 640)?;
            let mut down_ref = vec![0.0; tokens * 2560];
            project_ref(&middle, &down.packed, &down.scales, down.global, 2560, 640, &mut down_ref)
                .map_err(|e| e.to_string())?;
            assert_eq!(down_bits, down_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "down tokens={tokens}");
            println!("PASS fp4 exact expert tokens={tokens} gate/up [640x2560] down [2560x640] bit-exact");
        }
        Ok(())
    })
}
