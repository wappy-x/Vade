//! Exercise the packed expert kernel with a partial final tile, repeated inputs, scattered
//! outputs and guard words before, between and after the output rows.
use engine_model::{plan::loop_load::ExpertCodes, shapes::Proj};
use spark_gpu_probe::driver::{Backend, DeviceVector};
use spark_tp::{Result, bounded_decoder::{Compute, Device}, loop_decoder::operands::Operands};

pub fn run<'d>(backend: &Backend<'d>, operands: &Operands<DeviceVector<'d>>, layers: usize) -> Result<()> {
    use spark_tp::loop_decoder::operands::weights::ResidentProduct;
    let compute = Device::new(backend);
    const INPUTS: usize = 5;
    const COUNT: usize = 65;
    const GUARD: usize = 16;
    let poison = f32::from_bits(0x7fc0_dead);
    for layer in [0, u32::try_from(layers.checked_sub(1).ok_or("checkpoint has no layers")?).map_err(|_| "layer exceeds its range")?] {
        for proj in [Proj::GateUp, Proj::Down] {
            let packed = operands.packed_expert(layer, proj, 0)?.ok_or("packed guard requires a packed expert")?;
            let widened = operands.expert(layer, proj, 0)?;
            let unit = packed.unit;
            let (rows, cols) = (unit.rows, unit.columns);
            let stride = rows + 4;
            let len = (COUNT - 1) * stride + rows;
            let host: Vec<_> = (0..INPUTS * cols).map(|i| ((i * 37 % 59) as f32 - 29.0) / 16.0).collect();
            let input = backend.upload_vector(&host)?.into_resident_shared()?;
            let x = compute.upload(&host)?;
            let mut reference = Vec::new();
            for source in 0..INPUTS {
                let values = compute.work(|| {
                    let row = compute.range(&x, source * cols, cols)?;
                    let y = ResidentProduct::resident_product(&compute, widened.value, widened.words.clone(), rows, cols, &row)?;
                    compute.read(&y)
                })?;
                reference.push(values);
            }
            let unit_value = packed.value.resident_range(packed.words)?;
            let mut expected = vec![poison; GUARD + len + GUARD];
            let whole = backend.upload_vector(&expected)?.into_resident_shared()?;
            let inputs: Vec<_> = (0..COUNT).map(|i| ((3 * i) % INPUTS) as u32).collect();
            let outputs: Vec<_> = (0..COUNT).map(|i| ((7 * i) % COUNT) as u32).collect();
            compute.work(|| {
                let mut output = whole.resident_range(GUARD..GUARD + len)?;
                backend.project_packed_rows_grouped_into(rows, cols, unit.codes == ExpertCodes::Mxfp8,
                    unit.order.is_some(), stride, &[(&unit_value, COUNT)], &input, Some(&inputs), &mut output, Some(&outputs))
            })?;
            for (&source, &destination) in inputs.iter().zip(&outputs) {
                let at = GUARD + destination as usize * stride;
                expected[at..at + rows].copy_from_slice(&reference[source as usize]);
            }
            let actual = backend.read_vector_values(&whole)?;
            let different = actual.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if actual.len() != expected.len() || different != 0 { return Err(format!("packed guard: layer {layer} {proj:?}: {different} differing words, read {} of {}", actual.len(), expected.len())); }
            eprintln!("LOOP PACKED GUARDS layer={layer} projection={proj:?} codes={:?} output_words={} guard_words={} different={different}", unit.codes, COUNT * rows, expected.len() - COUNT * rows);
        }
    }
    Ok(())
}
