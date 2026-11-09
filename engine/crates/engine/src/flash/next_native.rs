//! Native projection storage/dispatch and three-launch Flash-Next GDN layer.
use spark_gpu_probe::{
    driver::{Backend, DeviceVector},
    next_params::{MxLayer, MxProjection},
};
type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone, Copy, Default)]
pub struct Matrix {
    pub weights: u64,
    pub scales: u64,
    pub rows: u32,
    pub cols: u32,
}
pub(crate) struct Weight<'a> {
    pub weights: DeviceVector<'a>,
    pub scales: DeviceVector<'a>,
    pub matrix: Matrix,
}
impl<'a> Weight<'a> {
    pub fn load(b: &Backend<'a>, w: &[u8], s: &[u8], rows: usize, cols: usize) -> Result<Self> {
        if rows == 0
            || cols == 0
            || cols % 32 != 0
            || w.len() != rows * cols
            || s.len() != rows * cols / 32
        {
            return Err("native MXFP8 matrix/scales shape mismatch".into());
        }
        for (block, &scale) in w.chunks_exact(32).zip(s) {
            if scale == 255
                || block.iter().any(|&c| {
                    c & 127 == 127
                        || scale > 246
                            && !engine_model::flash::mxfp8::dequant_value(c, scale).is_finite()
                })
            {
                return Err("MXFP8 dequantization is not finite".into());
            }
        }
        let mut weights = b.vector(w.len().div_ceil(4))?;
        let mut scales = b.vector(s.len().div_ceil(4))?;
        b.write_vector_bytes(&mut weights, w)?;
        b.write_vector_bytes(&mut scales, s)?;
        let matrix = Matrix {
            weights: weights.device_address(),
            scales: scales.device_address(),
            rows: rows as u32,
            cols: cols as u32,
        };
        Ok(Self {
            weights,
            scales,
            matrix,
        })
    }
}
/// All operands are session-owned; at most nine independent matrices per phase.
pub fn project(
    b: &Backend<'_>,
    matrices: &[(Matrix, u64, u64)],
    tokens: usize,
    input_bf16: bool,
    round: bool,
) -> Result<()> {
    let mut layer = MxLayer::default();
    for &(m, input, output) in matrices {
        layer
            .push(MxProjection {
                weights: m.weights,
                scales: m.scales,
                input,
                output,
                rows: m.rows,
                cols: m.cols,
                tokens: tokens as u32,
                input_bf16: u32::from(input_bf16),
                round: u32::from(round),
            })
            .map_err(str::to_string)?;
    }
    if layer.count == 0 {
        return Err("empty MXFP8 phase".into());
    }
    // M 9..16 general rows run four output rows per warp (same values, see
    // flash_next_mxfp8_bits_r4): its grid is rows.div_ceil(16) blocks per projection.
    let mut name = layer.projections[0].kernel();
    if name == "flash_next_mxfp8_bits" {
        for i in 0..layer.count as usize {
            layer.ends[i + 1] = layer.ends[i] + layer.projections[i].rows.div_ceil(16);
        }
        name = "flash_next_mxfp8_bits_r4";
    }
    unsafe {
        b.launch_flash_shared(
            name,
            layer.ends[layer.count as usize],
            128,
            0,
            &mut [(&mut layer as *mut MxLayer).cast()],
        )
    }
}
