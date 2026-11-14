//! Official ModelOpt NVFP4: adjacent E2M1 values, E4M3 scales per 16
//! values and one FP32 multiplier per projection. No weight requantization.
use super::{WeightFormat, Checkpoint, PLE_PREFIX};
use crate::{Result, invalid, json::{self, Value, field}, safetensors};
use std::{collections::BTreeMap, path::Path};

type Object = BTreeMap<String, Value>;
pub const BLOCK: usize = 16;

fn require(ok: bool, setting: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("unsupported NVFP4 setting: {setting}"))) }
}

fn validate_layers(layers: &Object, mtp_algo: &str) -> Result<()> {
    require(layers.len() == 50, "quantized layer count")?;
    for layer in 0..48 {
        let name = format!("model.language_model.layers.{layer}.mlp.experts");
        let entry = field(layers, &name)?.object()?;
        require(field(entry, "quant_algo")?.string()? == "NVFP4"
            && field(entry, "group_size")?.u64()? == BLOCK as u64, "expert quantization")?;
    }
    let mtp = field(layers, "mtp.layers.0.mlp.experts")?.object()?;
    require(field(mtp, "quant_algo")?.string()? == mtp_algo
        && field(mtp, "group_size")?.u64()? == 128, "MTP FP8 blocks")?;
    let ple = field(layers, &format!("{PLE_PREFIX}.ngram_embedding"))?.object()?;
    require(field(ple, "quant_algo")?.string()? == "FP8", "PLE per-tensor FP8")
}

/// Shared discovery/startup contract. Any present sidecar must satisfy the
/// complete legacy NVFP4 schema, including for mixed FP4-Local checkpoints.
/// Without a sidecar, mixed FP4-Local declares storage in config.json instead.
pub fn detect(root: &Path) -> Result<WeightFormat> {
    let path = root.join("hf_quant_config.json");
    match std::fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let config = root.join("config.json");
            if !config.try_exists()? { return Ok(WeightFormat::Fp8); }
            let bytes = safetensors::read_bounded(&config, 1024 * 1024)?;
            let doc = json::parse(&bytes, json::Limits::default())?;
            let Some(q) = doc.object()?.get("quantization_config") else { return Ok(WeightFormat::Fp8); };
            let q = q.object()?;
            if q.get("quant_method").map(Value::string).transpose()? != Some("modelopt") {
                return Ok(WeightFormat::Fp8);
            }
            return super::quantization_format(q);
        },
        Err(e) => return Err(e.into()),
        Ok(_) => (),
    }
    let bytes = safetensors::read_bounded(&path, 1024 * 1024)?;
    let doc = json::parse(&bytes, json::Limits::default())?;
    let q = field(doc.object()?, "quantization")?.object()?;
    require(field(q, "quant_algo")?.string()? == "MIXED_PRECISION"
        && field(q, "group_size")?.u64()? == BLOCK as u64
        && field(q, "kv_cache_quant_algo")? == &Value::Null, "sidecar precision/cache")?;
    validate_layers(field(q, "quantized_layers")?.object()?, "FP8_BLOCK_SCALES")?;
    Ok(WeightFormat::Nvfp4)
}

pub(super) fn validate_config(q: &Object) -> Result<()> {
    require(field(q, "quant_algo")?.string()? == "MIXED_PRECISION", "config precision")?;
    validate_layers(field(q, "quantized_layers")?.object()?, "FP8_PB_WO")?;
    let groups = field(q, "config_groups")?.object()?;
    require(groups.len() == 3, "config groups")?;
    for (name, bits, group, dynamic, targets) in [
        ("group_0", 4, Some(16), false, (0..48).map(|i|
            format!("model.language_model.layers.{i}.mlp.experts")).collect::<Vec<_>>()),
        ("group_1", 8, Some(128), true, vec!["mtp.layers.0.mlp.experts".into()]),
        ("group_2", 8, None, false, vec![format!("{PLE_PREFIX}.ngram_embedding")]),
    ] {
        let g = field(groups, name)?.object()?;
        let mut actual = field(g, "targets")?.array()?.iter()
            .map(|v| Ok(v.string()?.to_owned())).collect::<Result<Vec<_>>>()?;
        let mut expected = targets;
        actual.sort_unstable(); expected.sort_unstable();
        require(actual == expected, "quantization targets")?;
        for (role, is_dynamic) in [("weights", false), ("input_activations", dynamic)] {
            if role == "input_activations" && group.is_none() {
                require(!g.contains_key(role), "PLE activations")?;
                continue;
            }
            let v = field(g, role)?.object()?;
            require(field(v, "num_bits")?.u64()? == bits
                && field(v, "type")?.string()? == "float"
                && field(v, "dynamic")?.boolean()? == is_dynamic, "operand precision")?;
            if let Some(size) = group {
                require(field(v, "group_size")?.u64()? == size, "operand block size")?;
            } else {
                require(!v.contains_key("group_size"), "PLE per-tensor weights")?;
            }
        }
    }
    Ok(())
}

/// E2M1 includes signed zero and subnormal +/-0.5, with no NaN/Inf codes.
pub fn e2m1(nibble: u8) -> f32 {
    let value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nibble & 7) as usize];
    if nibble & 8 != 0 { -value } else { value }
}

pub fn e4m3(byte: u8) -> f32 {
    let exponent = (byte >> 3) & 15;
    let mantissa = byte & 7;
    let value = if exponent == 0 { mantissa as f32 / 512.0 }
        else if exponent == 15 && mantissa == 7 { f32::NAN }
        else { f32::from_bits(((exponent as u32 + 120) << 23) | ((mantissa as u32) << 20)) };
    if byte & 128 != 0 { -value } else { value }
}

pub fn dequant_value(byte: u8, high: bool, scale: u8, global: f32) -> f32 {
    // E2M1 * E4M3 is exactly representable in FP32. Only the multiplication
    // by the stored FP32 global scale rounds; no BF16 rounding belongs here.
    e2m1(if high { byte >> 4 } else { byte & 15 }) * e4m3(scale) * global
}

/// Caller-owned output avoids an expanded resident copy. Matrix K must be a
/// multiple of 16, so each row's scale blocks follow the same flat indexing.
pub fn dequantize(packed: &[u8], scales: &[u8], global: f32, output: &mut [f32]) -> Result<()> {
    require(output.len().is_multiple_of(BLOCK) && output.len() / 2 == packed.len()
        && output.len() / BLOCK == scales.len(), "dequantization lengths")?;
    require(global.is_finite() && global > 0.0, "global weight scale")?;
    require(scales.iter().all(|&s| s < 127), "nonnegative finite block scales")?;
    for (i, value) in output.iter_mut().enumerate() {
        *value = dequant_value(packed[i / 2], i % 2 != 0, scales[i / BLOCK], global);
    }
    Ok(())
}

/// Invalid CSF container data, before any scale bytes reach the kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsfError {
    Shape,
    FixedLength { expected: usize, actual: usize },
    Base { row: usize, value: u8 },
    ExceptionPosition { position: usize, scales: usize },
}
impl std::fmt::Display for CsfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape => write!(f, "invalid CSF matrix shape"),
            Self::FixedLength { expected, actual } =>
                write!(f, "CSF fixed length {actual}, expected {expected}"),
            Self::Base { row, value } => write!(f, "CSF row {row} base {value} exceeds 240"),
            Self::ExceptionPosition { position, scales } =>
                write!(f, "CSF exception position {position} outside {scales} scales"),
        }
    }
}
impl std::error::Error for CsfError {}

/// Losslessly expand FP4-Local CSF into E4M3 bytes `[rows, cols/16]`;
/// `cols` is the unpacked weight K dimension, not the scale column count.
/// Fixed U8 shape is `[rows/16, 16*(1+(cols/16)/2)]`. Each slab has 16
/// row bases (<=240), then row-major unsigned codes, low nibble first.
/// Sorted U32 exceptions store the flat position in low 24 bits and the
/// replacement byte in high 8 bits. Expansion preserves every byte pattern;
/// arithmetic consumers validate their own finite/nonnegative requirements.
pub fn csf_expand(fixed: &[u8], exceptions: &[u32], rows: usize, cols: usize)
    -> std::result::Result<Vec<u8>, CsfError> {
    if rows == 0 || !rows.is_multiple_of(BLOCK) || cols == 0 || !cols.is_multiple_of(2 * BLOCK) {
        return Err(CsfError::Shape);
    }
    let scale_cols = cols / BLOCK;
    let count = rows.checked_mul(scale_cols).ok_or(CsfError::Shape)?;
    let fixed_row = scale_cols.checked_mul(8).and_then(|n| n.checked_add(BLOCK))
        .ok_or(CsfError::Shape)?;
    let expected = (rows / BLOCK).checked_mul(fixed_row).ok_or(CsfError::Shape)?;
    if fixed.len() != expected {
        return Err(CsfError::FixedLength { expected, actual: fixed.len() });
    }
    for (slab, fixed) in fixed.chunks_exact(fixed_row).enumerate() {
        for (i, &value) in fixed[..BLOCK].iter().enumerate() {
            if value > 240 { return Err(CsfError::Base { row: slab * BLOCK + i, value }); }
        }
    }
    let mut scales = vec![0u8; count];
    for (slab, fixed) in fixed.chunks_exact(fixed_row).enumerate() {
        for i in 0..BLOCK {
            for c in 0..scale_cols {
                let byte = fixed[BLOCK + i * (scale_cols / 2) + c / 2];
                let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                scales[(slab * BLOCK + i) * scale_cols + c] = fixed[i] + code;
            }
        }
    }
    for &entry in exceptions {
        let position = (entry & 0x00ff_ffff) as usize;
        if position >= count { return Err(CsfError::ExceptionPosition { position, scales: count }); }
        scales[position] = (entry >> 24) as u8;
    }
    Ok(scales)
}

/// Widen BF16 bits exactly; every BF16 value is representable in FP32.
pub fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

/// Round FP32 to BF16 bits, nearest-even. Inputs are finite by contract.
pub fn f32_to_bf16(value: f32) -> u16 {
    ((value.to_bits().wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1))) >> 16) as u16
}

#[allow(clippy::excessive_precision, clippy::manual_clamp)] // literal constants stay as written; clamp panics and treats NaN differently.
pub(crate) fn exp_ref(t: f32) -> f32 {
    // FMA-only exp so the CPU oracle and the PTX kernel agree bit-for-bit:
    // range-reduce by ln2 (hi/lo split), degree-7 Taylor on [-ln2/2, ln2/2],
    // rescale by 2^n from raw exponent bits. Clamping t to [-87, 87] keeps
    // n in [-126, 126]; beyond it, 1 + exp(t) rounds identically either way.
    let t = if t > 87.0 { 87.0 } else if t < -87.0 { -87.0 } else { t };
    let n = t * core::f32::consts::LOG2_E + 12_582_912.0 - 12_582_912.0;
    let r = (-n).mul_add(0.693145751953125, t);
    let r = (-n).mul_add(0.0000014286068, r);
    let mut p = 1.0 / 5040.0f32;
    for c in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = p.mul_add(r, c);
    }
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// SwiGLU middle element: BF16(silu(gate) * up), silu(g) = g / (1 + exp(-g)).
pub fn swiglu_ref(gate: f32, up: f32) -> u16 {
    f32_to_bf16(gate / (1.0 + exp_ref(-gate)) * up)
}

/// CPU FP32 oracle for one NVFP4 projection, W4A16: BF16 activations,
/// dequantized FP32 weights, one fused multiply-add per element in strictly
/// ascending K order. `input_scale` from the checkpoint calibrates W4A4
/// activation quantization only and is never applied to W4A16 weights.
/// Input is row-major BF16 bits [tokens, cols]; output [tokens, rows].
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
pub fn project_ref(input: &[u16], packed: &[u8], scales: &[u8], global: f32,
    rows: usize, cols: usize, output: &mut [f32]) -> Result<()> {
    require(cols > 0 && cols.is_multiple_of(BLOCK) && input.len().is_multiple_of(cols)
        && packed.len() == rows * cols / 2 && scales.len() == rows * cols / BLOCK
        && output.len() == input.len() / cols * rows, "projection reference lengths")?;
    require(global.is_finite() && global > 0.0, "global weight scale")?;
    require(scales.iter().all(|&s| s < 127), "nonnegative finite block scales")?;
    for (token, x) in input.chunks_exact(cols).enumerate() {
        for row in 0..rows {
            let mut sum = 0.0f32;
            for k in 0..cols {
                let at = row * cols + k;
                sum = bf16_to_f32(x[k])
                    .mul_add(dequant_value(packed[at / 2], !at.is_multiple_of(2), scales[at / BLOCK], global), sum);
            }
            output[token * rows + row] = sum;
        }
    }
    Ok(())
}

impl Checkpoint {
    /// Bounded, exact CPU reference for one logical row segment; never reads
    /// the entire expert matrix or expands weights in the inference loader.
    pub fn nvfp4_row(&self, root: &Path, prefix: &str, row: u64, column: u64, output: &mut [f32]) -> Result<()> {
        require(self.config.format == WeightFormat::Nvfp4, "reference weight format")?;
        let tensor = self.tensor(&format!("{prefix}.weight"))?;
        require(tensor.shape.len() == 2 && tensor.dtype == safetensors::Dtype::U8, "reference packed matrix")?;
        let columns = tensor.shape[1] * 2;
        require(row < tensor.shape[0] && column.is_multiple_of(BLOCK as u64)
            && output.len().is_multiple_of(BLOCK)
            && column.checked_add(output.len() as u64).is_some_and(|end| end <= columns), "reference row range")?;
        let mut packed = vec![0; output.len() / 2];
        let mut scales = vec![0; output.len() / BLOCK];
        let mut global = [0; 4];
        self.read_range(root, &format!("{prefix}.weight"), (row * columns + column) / 2, &mut packed)?;
        self.read_range(root, &format!("{prefix}.weight_scale"), (row * columns + column) / BLOCK as u64, &mut scales)?;
        self.read_range(root, &format!("{prefix}.weight_scale_2"), 0, &mut global)?;
        dequantize(&packed, &scales, f32::from_le_bytes(global), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_scale_precedes_global_scale_without_intermediate_rounding() {
        let global = f32::from_bits(0x3f800001);
        // Captured from today's dequantizer; reassociating the scales changes one ULP.
        assert_eq!(dequant_value(0x77, false, 0x3c, global).to_bits(), 0x41100001);
        assert_eq!(dequant_value(0x77, true, 0x3c, global).to_bits(), 0x41100001);
        let folded = e2m1(7) * (e4m3(0x3c) * global);
        assert_eq!(folded.to_bits(), 0x41100002);
    }
    #[test]
    fn format_values_and_bounds() {
        for n in 0..16 { assert_eq!(e2m1(n).abs(), [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(n & 7) as usize]); }
        assert_eq!(e2m1(8).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e4m3(1), 1.0 / 512.0); assert_eq!(e4m3(126), 448.0); assert!(e4m3(127).is_nan());
        let mut out = [0.; 16]; dequantize(&[0x21; 8], &[56], 2., &mut out).unwrap();
        assert_eq!(&out[..2], &[1., 2.]); assert!(dequantize(&[0; 7], &[56], 1., &mut out).is_err());
        assert!(dequantize(&[0; 8], &[127], 1., &mut out).is_err()); assert!(WeightFormat::Nvfp4.require_opt_in(false).is_err());
    }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
        }
        fn byte(&mut self) -> u8 { (self.next() >> 32) as u8 }
        fn bf16(&mut self) -> u16 { f32_to_bf16((self.next() % 2000) as f32 / 500.0 - 2.0) }
    }

    /// Test-side CSF encoder: unsigned row windows, with outliers as u24 exceptions.
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

    #[test]
    fn csf_roundtrip_real_shapes_and_rejects_garbage() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for (rows, packed_cols, fixed_shape, scale_cols) in
            [(640, 1280, [40, 1296], 160), (2560, 320, [160, 336], 40)] {
            let cols = packed_cols * 2;
            let scales: Vec<_> = (0..rows * scale_cols).map(|_| rng.byte()).collect();
            let (fixed, exceptions) = csf_encode(&scales, rows, cols);
            assert_eq!(fixed.len(), fixed_shape[0] * fixed_shape[1]);
            assert!(!exceptions.is_empty(), "outliers must produce exceptions");
            assert_eq!(csf_expand(&fixed, &exceptions, rows, cols).unwrap(), scales);
            assert!(matches!(csf_expand(&fixed[..fixed.len() - 1], &exceptions, rows, cols),
                Err(CsfError::FixedLength { .. })));
            assert_eq!(csf_expand(&fixed, &[(rows * scale_cols) as u32], rows, cols),
                Err(CsfError::ExceptionPosition { position: rows * scale_cols, scales: rows * scale_cols }));
            // Independent fixed slabs: alternating columns 0/15, distinct row bases.
            let mut fixed = vec![0xf0; fixed_shape[0] * fixed_shape[1]];
            for slab in 0..fixed_shape[0] { for i in 0..16 {
                fixed[slab * fixed_shape[1] + i] = ((slab * 16 + i) % 241) as u8;
            } }
            let out = csf_expand(&fixed, &[], rows, cols).unwrap();
            assert_eq!(out.len(), rows * scale_cols);
            for r in 0..rows { for c in 0..scale_cols {
                assert_eq!(out[r * scale_cols + c], (r % 241) as u8 + if c % 2 == 0 { 0 } else { 15 });
            } }
        }
        // Hand-built two slabs, four scale columns: every nibble, row and slab differ.
        let fixed = [
            0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
            240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0,
            0xff, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let out = csf_expand(&fixed, &[], 32, 64).unwrap();
        for r in 0..16 { for c in 0..4 {
            assert_eq!(out[r * 4 + c], (r * 16 + (r % 4) * 4 + c) as u8);
        } }
        assert_eq!(&out[64..68], &[255, 255, 240, 248]);
        let mut expected = out.clone(); expected[0] = 0x80; expected[127] = 0xff;
        assert_eq!(csf_expand(&fixed, &[0x8000_0000, 0xff00_007f], 32, 64).unwrap(), expected);
        let mut invalid = fixed; invalid[48 + 15] = 241;
        assert_eq!(csf_expand(&invalid, &[], 32, 64), Err(CsfError::Base { row: 31, value: 241 }));
        for (rows, cols) in [(0, 64), (17, 64), (16, 0), (16, 16), (usize::MAX - 15, 64)] {
            assert_eq!(csf_expand(&[], &[], rows, cols), Err(CsfError::Shape));
        }
    }

    #[test]
    fn project_ref_matches_dequantized_matmul_and_is_batch_invariant() {
        let mut rng = Rng(0x243f6a8885a308d3);
        for (rows, cols) in [(640, 2560), (2560, 640)] {
            let packed: Vec<_> = (0..rows * cols / 2).map(|_| rng.byte()).collect();
            let raw: Vec<_> = (0..rows * (cols / 16)).map(|_| (rng.next() % 127) as u8).collect();
            let (fixed, exceptions) = csf_encode(&raw, rows, cols);
            let scales = csf_expand(&fixed, &exceptions, rows, cols).unwrap();
            let global = 0.0078125f32;
            let input: Vec<_> = (0..3 * cols).map(|_| rng.bf16()).collect();
            let mut got = vec![0.0; 3 * rows];
            project_ref(&input, &packed, &scales, global, rows, cols, &mut got).unwrap();
            let mut dense = vec![0.0f32; rows * cols];
            dequantize(&packed, &scales, global, &mut dense).unwrap();
            for token in 0..3 { for row in 0..rows {
                let mut sum = 0.0f32;
                for k in 0..cols {
                    sum = bf16_to_f32(input[token * cols + k]).mul_add(dense[row * cols + k], sum);
                }
                assert_eq!(got[token * rows + row].to_bits(), sum.to_bits());
            } }
            let mut solo = vec![0.0; rows];
            project_ref(&input[cols..2 * cols], &packed, &scales, global, rows, cols, &mut solo).unwrap();
            assert_eq!(solo.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got[rows..2 * rows].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn swiglu_ref_is_correctly_rounded_or_one_ulp() {
        for gi in -40..=40 { for ui in -4..=4 {
            let (g, u) = (gi as f64 / 4.0, ui as f64 / 2.0);
            let exact = g / (1.0 + (-g).exp()) * u;
            let got = bf16_to_f32(swiglu_ref(g as f32, u as f32)) as f64;
            let ulp = if exact == 0.0 { 1e-38 } else { exact.abs() * 0.00390625 };
            assert!((got - exact).abs() <= ulp, "g={g} u={u} got={got} exact={exact}");
        } }
        assert_eq!(swiglu_ref(0.0, 3.0), 0);
        assert_eq!(bf16_to_f32(swiglu_ref(30.0, 2.0)), 60.0);
        let tail = bf16_to_f32(swiglu_ref(-30.0, 2.0));
        assert!(tail < 0.0 && tail.abs() < 1e-11, "silu(-30)*2={tail}");
    }

    #[test]
    #[ignore = "requires VADE_NVFP4_MODEL_DIR pointing to the official checkpoint"]
    fn real_checkpoint_reference() {
        let root = std::path::PathBuf::from(std::env::var_os("VADE_NVFP4_MODEL_DIR").expect("checkpoint directory"));
        let c = Checkpoint::inspect(&root).unwrap(); assert_eq!(c.index.weight_map.len(), 299545);
        for layer in [0, 23, 47] { for expert in [0, 511] { for proj in ["gate_proj", "up_proj", "down_proj"] {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}");
            let t = c.tensor(&format!("{p}.weight")).unwrap(); let k = t.shape[1] as usize * 2;
            for row in [0, t.shape[0] / 2, t.shape[0] - 1] {
                let mut out = vec![0.; k]; c.nvfp4_row(&root, &p, row, 0, &mut out).unwrap();
                let (mut w, mut s, mut g) = (vec![0; k / 2], vec![0; k / 16], [0; 4]);
                c.read_range(&root, &format!("{p}.weight"), row * k as u64 / 2, &mut w).unwrap();
                c.read_range(&root, &format!("{p}.weight_scale"), row * k as u64 / 16, &mut s).unwrap();
                c.read_range(&root, &format!("{p}.weight_scale_2"), 0, &mut g).unwrap();
                for i in 0..k {
                    let n = (w[i / 2] >> (4 * (i % 2))) & 15; let e = (n >> 1) & 3; let m = n & 1;
                    let a = if e == 0 { m as f64 / 2. } else { (1. + m as f64 / 2.) * 2f64.powi(e as i32 - 1) };
                    let b = s[i / 16]; let e = (b >> 3) & 15; let m = b & 7;
                    let b = if e == 0 { m as f64 / 512. } else { (1. + m as f64 / 8.) * 2f64.powi(e as i32 - 7) };
                    let expected = ((if n & 8 != 0 { -a } else { a }) * b * f32::from_le_bytes(g) as f64) as f32;
                    assert_eq!(out[i].to_bits(), expected.to_bits(), "{p} row={row} column={i}");
                }
            }
        } } }
    }
}
