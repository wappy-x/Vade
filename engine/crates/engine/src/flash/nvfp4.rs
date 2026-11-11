//! Borrowed device descriptors for the exact packed checkpoint layout. Dense
//! BF16, FP8 MTP and FP8 PLE keep their existing loaders and descriptors.
use super::Weights;
use engine_model::{flash::{WeightFormat, nvfp4, plan::Selection}, safetensors::Dtype};
use spark_gpu_probe::kernel_params::NvFp4Matrix;

type Result<T> = std::result::Result<T, String>;

pub struct Matrix<'w, 'a> {
    pub params: NvFp4Matrix,
    weights: &'w Weights<'a>,
    prefix: String,
}
impl<'w, 'a> Matrix<'w, 'a> {
    pub fn load(weights: &'w Weights<'a>, prefix: &str) -> Result<Self> {
        if weights.mixed { return Err("mixed NVFP4 requires the typed CSF-aware storage hook".into()); }
        if weights.format != WeightFormat::Nvfp4 || !prefix.starts_with("model.language_model.layers.")
            || !prefix.contains(".mlp.experts.") {
            return Err("NVFP4 descriptor requires a main-model routed expert".into());
        }
        let down = prefix.ends_with(".down_proj");
        let shape = if down { [2560, 640] } else { [640, 2560] };
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let global_name = format!("{prefix}.weight_scale_2");
        let input_name = format!("{prefix}.input_scale");
        for (name, dtype, shape) in [
            (&weight_name, Dtype::U8, vec![shape[0], shape[1] / 2]),
            (&scale_name, Dtype::F8E4M3, vec![shape[0], shape[1] / 16]),
            (&global_name, Dtype::F32, vec![]),
            (&input_name, Dtype::F32, vec![]),
        ] {
            let t = weights.tensor(name)?;
            let selected = weights.plan.selection_for(name, weights.format).map_err(|e| e.to_string())?;
            if t.tensor.dtype != dtype || t.tensor.shape != shape || t.selection != selected {
                return Err(format!("NVFP4 descriptor dtype/shape/ownership mismatch: {name}"));
            }
        }
        let weight = weights.tensor(&weight_name)?;
        let (rows, cols, row_start, column_start) = match &weight.selection {
            Selection::Rows(ranges) if !down && ranges.len() == 1 =>
                (ranges[0].end - ranges[0].start, shape[1], ranges[0].start, 0),
            Selection::Columns(columns) if down =>
                (shape[0], (columns.end - columns.start) * 2, 0, columns.start * 2),
            _ => return Err("NVFP4 descriptor has invalid rank-local layout".into()),
        };
        if rows == 0 || cols == 0 || cols % 16 != 0
            || weight.bytes as u64 != rows * cols / 2
            || weights.tensor(&scale_name)?.bytes as u64 != rows * cols / 16 {
            return Err("NVFP4 packed weight/scale lengths disagree".into());
        }
        let params = NvFp4Matrix {
            weight: weights.device_address(&weight_name)?, block_scales: weights.device_address(&scale_name)?,
            global_scale: weights.device_address(&global_name)?, input_scale: weights.device_address(&input_name)?,
            rows: rows as u32, cols: cols as u32, row_start: row_start as u32, column_start: column_start as u32,
        };
        if !params.weight.is_multiple_of(16) || !params.block_scales.is_multiple_of(16)
            || !params.global_scale.is_multiple_of(4) || !params.input_scale.is_multiple_of(4) {
            return Err("NVFP4 descriptor has unaligned device storage".into());
        }
        Ok(Self { params, weights, prefix: prefix.into() })
    }

    /// Bounded logical inspection; never allocates or retains an expanded matrix.
    pub fn value_f32(&self, local_element: usize) -> Result<f32> {
        if local_element >= self.params.rows as usize * self.params.cols as usize {
            return Err("NVFP4 element exceeds local matrix".into());
        }
        let mut packed = [0];
        let mut scale = [0];
        let mut global = [0; 4];
        self.weights.read(&format!("{}.weight", self.prefix), local_element / 2, &mut packed)?;
        self.weights.read(&format!("{}.weight_scale", self.prefix), local_element / 16, &mut scale)?;
        self.weights.read(&format!("{}.weight_scale_2", self.prefix), 0, &mut global)?;
        let global = f32::from_le_bytes(global);
        if scale[0] >= 127 || !global.is_finite() || global <= 0.0 {
            return Err("NVFP4 weight has invalid block/global scale".into());
        }
        Ok(nvfp4::dequant_value(packed[0], !local_element.is_multiple_of(2), scale[0], global))
    }
}
