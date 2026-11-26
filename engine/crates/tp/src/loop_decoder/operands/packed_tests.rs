//! The packed expert stacks of a loaded checkpoint, on the made-up checkpoints of the driver tests.
//!
//! A compute with a packed product (the host twin of the device kernel, `UnitShape::product`) reads the experts
//! through `LoadedWeights::product`. Each product must give the words that the fixture gives, which reads the same
//! stored bytes by its own code; and a compute that takes only some shapes must give the same words through the
//! widened stack for the others.
use super::{*, tests::{bits, folder}};
use crate::loop_decoder::{fixture::Fixture, mix::support::Stand, operands::load::{self, Store}};
use engine_model::{flash::nvfp4::{bf16_to_f32, f32_to_bf16}, plan::{loop_load::{ExpertCodes, PartKind}, loop_widen}, shapes::Ffn};
use spark_gpu_probe::packed_product;
use std::{cell::Cell, os::unix::fs::FileExt};

/// Host bytes as the owner of the weights, of a type of their own: `Stand` has a packed product for them.
struct Units(Vec<u8>);
struct UnitStore;
impl Store for UnitStore {
    type Value = Units;
    fn create(&mut self, words: usize) -> Result<Units> { Ok(Units(vec![0xa5; words * 4])) }
    fn write(&mut self, value: &mut Units, word: usize, bytes: &[u8]) -> Result<()> { value.0[word * 4..word * 4 + bytes.len()].copy_from_slice(bytes); Ok(()) }
    fn read(&mut self, value: &Units, word: usize, out: &mut [u8]) -> Result<()> { out.copy_from_slice(&value.0[word * 4..word * 4 + out.len()]); Ok(()) }
}
thread_local! {
    /// Products that read a packed unit, and products that read a widened matrix, on this test thread.
    static PACKED: Cell<usize> = const { Cell::new(0) };
    static WIDENED: Cell<usize> = const { Cell::new(0) };
    /// True: take only the shapes that the device kernel takes (`packed_product::shape_admitted`).
    static DEVICE_SHAPES: Cell<bool> = const { Cell::new(false) };
    /// Grouped calls that `Stand` took on this test thread.
    static GROUPED: Cell<usize> = const { Cell::new(0) };
}
impl ResidentProduct<Units> for Stand {
    type RoutePlan = ();
    fn resident_product(&self, weight: &Units, words: Range<usize>, rows: usize, columns: usize, x: &Vec<f32>) -> Result<Vec<f32>> {
        let matrix = weight.0[words.start * 4..words.end * 4].as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect::<Vec<_>>();
        assert_eq!(matrix.len(), rows * columns);
        WIDENED.set(WIDENED.get() + 1);
        chain_product(self, &matrix, x)
    }
    fn resident_rows(&self, table: &Units, words: Range<usize>, rows: usize, columns: usize, ids: &[u32]) -> Result<Option<Vec<f32>>> {
        Ok(Some(super::tests::host_rows(&table.0[words.start * 4..words.end * 4], rows, columns, ids)))
    }
    fn packed_product(&self, weight: &Units, words: Range<usize>, unit: UnitShape, x: &Vec<f32>) -> Result<Option<Vec<f32>>> {
        if DEVICE_SHAPES.get() && !packed_product::shape_admitted(unit.rows, unit.columns, unit.codes == ExpertCodes::Mxfp8, unit.order.is_some()) { return Ok(None); }
        let mut out = vec![f32::NAN; unit.rows];
        unit.product(&weight.0[words.start * 4..words.end * 4], x, &mut out).map_err(|e| e.to_string())?;
        PACKED.set(PACKED.get() + 1);
        Ok(Some(out))
    }
    fn packed_group(&self, weight: &Units, units: &[Range<usize>], unit: UnitShape, x: &Vec<f32>, shared: bool) -> Result<Option<Vec<f32>>> {
        if DEVICE_SHAPES.get() && !packed_product::shape_admitted(unit.rows, unit.columns, unit.codes == ExpertCodes::Mxfp8, unit.order.is_some()) { return Ok(None); }
        assert_eq!(x.len(), if shared { unit.columns } else { unit.columns * units.len() });
        let mut all = Vec::with_capacity(unit.rows * units.len());
        for (e, words) in units.iter().enumerate() {
            let input = if shared { &x[..] } else { &x[e * unit.columns..(e + 1) * unit.columns] };
            let mut out = vec![f32::NAN; unit.rows];
            unit.product(&weight.0[words.start * 4..words.end * 4], input, &mut out).map_err(|e| e.to_string())?;
            all.extend(out);
        }
        GROUPED.set(GROUPED.get() + 1);
        Ok(Some(all))
    }
}
/// Every expert product of the fixture through the loaded weights. Returns (products, through a packed unit,
/// through a widened matrix).
fn expert_products(fixture: &Fixture) -> (usize, usize, usize) {
    let (folder, checkpoint) = folder(fixture);
    let loaded = load::load(&checkpoint, &folder.0, &mut UnitStore, 2).unwrap();
    let files = Files::open(&checkpoint, &folder.0).unwrap();
    let weights = LoadedWeights::new(&checkpoint, &files, &loaded.operands).unwrap();
    let (c, d, hidden) = (Stand::new(), &fixture.description, fixture.description.hidden as usize);
    let (before, mut products) = ((PACKED.get(), WIDENED.get()), 0);
    for (i, layer) in d.layers.iter().enumerate() {
        let Ffn::Experts(e) = layer.ffn else { panic!("a fixture layer has routed experts") };
        for expert in 0..e.count as usize {
            for weight in [Weight::GateUp(i, expert), Weight::Down(i, expert)] {
                let expected = <Fixture as Weights<Stand>>::words(fixture, weight).unwrap();
                let columns = if matches!(weight, Weight::GateUp(..)) { hidden } else { expected.len() / hidden };
                // An input with BF16 values of both signs, another for each weight.
                let x = (0..columns).map(|k| ((k * 37 + expected.len() + expert) % 23) as f32 * 0.0625 - 0.6875).collect::<Vec<_>>();
                let got = weights.product(&c, weight, &x).unwrap_or_else(|e| panic!("{weight:?}: {e}"));
                assert_eq!(bits(&got), bits(&<Fixture as Weights<Stand>>::product(fixture, &c, weight, &x).unwrap()), "{weight:?}");
                assert_eq!(got.len(), expected.len() / columns, "{weight:?}");
                products += 1;
            }
        }
    }
    // A weight that is no expert never reads a packed unit.
    let (packed, widened) = (PACKED.get() - before.0, WIDENED.get() - before.1);
    let x = vec![0.5f32; hidden];
    weights.product(&c, Weight::Router(0), &x).unwrap();
    assert_eq!(PACKED.get() - before.0, packed);
    (products, packed, widened)
}

#[test]
fn every_expert_product_reads_its_packed_unit_and_equals_the_fixture() {
    for fixture in [Fixture::small(), Fixture::wide()] {
        let (products, packed, widened) = expert_products(&fixture);
        assert!(products > 0);
        assert_eq!((packed, widened), (products, 0));
    }
}
#[test]
fn a_shape_that_the_packed_product_does_not_take_reads_the_widened_stack() {
    DEVICE_SHAPES.set(true);
    for fixture in [Fixture::small(), Fixture::wide()] {
        let (products, packed, widened) = expert_products(&fixture);
        assert_eq!(packed + widened, products);
        // The fixtures are narrower than a stage of the device kernel in at least one projection.
        assert!(widened > 0, "{products} {packed}");
    }
    DEVICE_SHAPES.set(false);
}
#[test]
fn a_loaded_unit_holds_the_stored_bytes_at_the_places_of_the_kernel_layout() {
    let fixture = Fixture::wide();
    let (folder, checkpoint) = folder(&fixture);
    let loaded = load::load(&checkpoint, &folder.0, &mut UnitStore, 3).unwrap();
    let shard = checkpoint.shards.keys().next().unwrap();
    let file = std::fs::File::open(folder.0.join(shard)).unwrap();
    let stored = |offset: u64, bytes: u64| { let mut out = vec![0u8; bytes as usize]; file.read_exact_at(&mut out, offset).unwrap(); out };
    let (mut units, mut forms) = (0, std::collections::BTreeSet::new());
    for (i, layer) in checkpoint.description.layers.iter().enumerate() {
        let Ffn::Experts(e) = layer.ffn else { panic!("a fixture layer has routed experts") };
        for proj in [Proj::GateUp, Proj::Down] {
            let (widened, packed) = (checkpoint.load.experts(i as u32, proj).unwrap(), checkpoint.load.packed_experts(i as u32, proj).unwrap());
            for expert in 0..e.count {
                let part = loaded.operands.packed_expert(i as u32, proj, expert).unwrap().unwrap();
                let (unit, bytes) = (part.unit, &part.value.0[part.words.start * 4..part.words.end * 4]);
                // The layout of the load plan is the layout of the kernel for every unit, also for a shape that the kernel does not take.
                let (plan, wide) = (unit.layout().unwrap(), unit.codes == ExpertCodes::Mxfp8);
                let tile = packed_product::layout(unit.rows, unit.columns, wide);
                assert_eq!((plan.scales, plan.table, plan.bytes, bytes.len()), (tile.scales, tile.table, tile.bytes, tile.bytes));
                assert_eq!((unit.code_stride(), unit.scale_stride(), unit.table_bytes()), (packed_product::code_stride(unit.columns, wide), packed_product::scale_stride(unit.columns, wide), packed_product::table_bytes(wide)));
                // The stored codes and block scales of this expert, byte for byte, at their places in tiles, and zeros
                // everywhere else before the table. For a shape the kernel takes, the places are those of its copies.
                let (codes, scales) = (packed.matrix_part(PartKind::Codes, expert).unwrap(), packed.matrix_part(PartKind::BlockScale, expert).unwrap());
                let (codes, scales) = (stored(codes.0, codes.1), stored(scales.0, scales.1));
                let ((cs, ss), mut expected) = ((unit.code_stride(), unit.scale_stride()), vec![0u8; plan.table]);
                assert_eq!((codes.len(), scales.len()), (unit.rows * cs, unit.rows * ss));
                for row in 0..unit.rows {
                    for b in 0..cs { expected[unit.code_at(row, b)] = codes[row * cs + b]; }
                    for b in 0..ss { expected[plan.scales + unit.scale_at(row, b)] = scales[row * ss + b]; }
                }
                assert!(bytes[..plan.table] == expected[..], "layer {i} {proj:?} expert {expert}");
                // The widened matrix of the same expert is the widened form of these bytes.
                let other = loaded.operands.expert(i as u32, proj, expert).unwrap();
                assert_eq!((other.rows as usize, other.columns as usize), (unit.rows, unit.columns));
                assert_eq!(widened.matrix_part(PartKind::Codes, expert).unwrap(), packed.matrix_part(PartKind::Codes, expert).unwrap());
                units += 1; forms.insert(wide);
            }
        }
    }
    assert!(units > 0 && forms.len() == 2, "{units} {forms:?}");
    // The units are in the fingerprint of the load: one unit line a packed matrix.
    assert_eq!(loaded.units.iter().filter(|u| u.name.ends_with(engine_model::plan::loop_load::PACKED_NAME)).count(), units);
    let verified = load::verify(&loaded, &mut UnitStore, 2).unwrap();
    assert_eq!((verified.units, verified.different.len()), (loaded.units.len(), 0));
}
/// The places of the host fill (`UnitShape::code_at`, `scale_at`) are the places of the kernel's copies
/// (`packed_product::code_copy`, `scale_copy`) for shapes the kernel takes, and the two layouts agree.
#[test]
fn the_host_places_of_a_unit_are_the_places_of_the_kernel_copies() {
    let mut compared = 0;
    for (codes, wide, column_list) in [(ExpertCodes::Nvfp4, false, [64usize, 192, 2304]), (ExpertCodes::Mxfp8, true, [128, 1152, 2304])] {
        for columns in column_list {
            for rows in [1usize, 31, 32, 33, 70, 2304] {
                let unit = UnitShape { codes, rows, columns, order: None };
                assert!(packed_product::shape_admitted(rows, columns, wide, false));
                let (plan, tile) = (unit.layout().unwrap(), packed_product::layout(rows, columns, wide));
                assert_eq!((plan.scales, plan.table, plan.bytes), (tile.scales, tile.table, tile.bytes));
                let step = packed_product::code_stage(wide);
                for row in 0..rows {
                    for stage in 0..packed_product::stages(columns) {
                        for part in 0..step / 16 { assert_eq!(unit.code_at(row, stage * step + part * 16), packed_product::code_copy(row, stage, part, columns, wide)); }
                        assert_eq!(unit.scale_at(row, if wide { (stage & !1) * 2 } else { stage * 4 }), packed_product::scale_copy(row, stage, columns, wide));
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 100_000, "{compared}");
}
#[test]
fn the_round_and_the_power_of_the_kernel_are_those_of_the_widen_step() {
    // The round of the kernel against the round of form A1, on patterns around every half way point and on small values.
    let mut compared = 0;
    for high in (0u32..0x7f80).step_by(7).chain([0, 1, 0x007f, 0x0080, 0x3f80, 0x7f7e]) {
        for low in [0u32, 1, 0x7fff, 0x8000, 0x8001, 0xffff] {
            for sign in [0u32, 0x8000_0000] {
                let value = f32::from_bits(sign | (high << 16) | low);
                if !value.is_finite() { continue; }
                assert_eq!(packed_product::round_bf16(value).to_bits(), bf16_to_f32(f32_to_bf16(value)).to_bits(), "{:#x}", value.to_bits());
                compared += 1;
            }
        }
    }
    assert!(compared > 50_000, "{compared}");
    // Value byte 0x38 is 1.0, so the weight of form A1 for it is the power of two of the scale byte.
    for scale in 0..=255u8 { assert_eq!(packed_product::power(u32::from(scale)).to_bits(), loop_widen::mxfp8_weight(0x38, scale).to_bits(), "scale {scale}"); }
}
#[test]
fn a_grouped_call_gives_the_words_of_one_call_a_matrix_for_every_layer_and_both_projections() {
    for fixture in [Fixture::small(), Fixture::wide()] {
        let (folder, checkpoint) = folder(&fixture);
        let loaded = load::load(&checkpoint, &folder.0, &mut UnitStore, 2).unwrap();
        let files = Files::open(&checkpoint, &folder.0).unwrap();
        let weights = LoadedWeights::new(&checkpoint, &files, &loaded.operands).unwrap();
        let (c, d, hidden) = (Stand::new(), &fixture.description, fixture.description.hidden as usize);
        let before = GROUPED.get();
        let mut groups = 0;
        for (i, layer) in d.layers.iter().enumerate() {
            let Ffn::Experts(e) = layer.ffn else { panic!("a fixture layer has routed experts") };
            // The picks in a falling order, as a router gives them: not the stored order.
            let ids = (0..e.count as usize).rev().collect::<Vec<_>>();
            let gate_up = ids.iter().map(|&x| Weight::GateUp(i, x)).collect::<Vec<_>>();
            let x = (0..hidden).map(|k| ((k * 29 + i) % 19) as f32 * 0.125 - 1.125).collect::<Vec<_>>();
            let one = gate_up.iter().flat_map(|&w| weights.product(&c, w, &x).unwrap()).collect::<Vec<_>>();
            let all = weights.products(&c, &gate_up, &x, true).unwrap().expect("the stand groups every shape");
            assert_eq!(bits(&all), bits(&one), "layer {i} gate-and-up");
            let down = ids.iter().map(|&x| Weight::Down(i, x)).collect::<Vec<_>>();
            let inner = one.len() / ids.len() / 2;
            let xs = (0..inner * ids.len()).map(|k| ((k * 31 + i) % 23) as f32 * 0.0625 - 0.6875).collect::<Vec<_>>();
            let one = down.iter().enumerate().flat_map(|(p, &w)| weights.product(&c, w, &xs[p * inner..(p + 1) * inner].to_vec()).unwrap()).collect::<Vec<_>>();
            let all = weights.products(&c, &down, &xs, false).unwrap().expect("the stand groups every shape");
            assert_eq!(bits(&all), bits(&one), "layer {i} down");
            // A group that mixes the projections, or the layers, is not taken.
            assert!(weights.products(&c, &[Weight::GateUp(i, 0), Weight::Down(i, 1)], &x, true).unwrap().is_none());
            if i > 0 { assert!(weights.products(&c, &[Weight::GateUp(i, 0), Weight::GateUp(i - 1, 0)], &x, true).unwrap().is_none()); }
            groups += 2;
        }
        assert_eq!(GROUPED.get() - before, groups);
    }
}

#[test]
fn packed_rows_quads_match_checkpoint_code_and_scale_offsets() {
    for codes in [ExpertCodes::Nvfp4, ExpertCodes::Mxfp8] {
        let wide = codes == ExpertCodes::Mxfp8;
        for rows in [1, 31, 32, 33, 65] {
            let unit = UnitShape { rows, columns: 256, codes, order: None };
            let layout = unit.layout().unwrap();
            for row in 0..rows {
                for column in (0..unit.columns).step_by(4) {
                    let (code, scale) = packed_product::rows_quad(row, column, unit.columns, wide);
                    let bytes = if wide { 4 } else { 2 };
                    assert_eq!(code, unit.code_at(row, column * bytes / 4));
                    assert_eq!(code + bytes - 1, unit.code_at(row, (column + 3) * bytes / 4));
                    assert_eq!(scale, unit.scale_at(row, column / unit.block()));
                    assert!(code.is_multiple_of(bytes) && code + bytes <= layout.scales);
                    assert!(layout.scales + scale < layout.table);
                }
            }
        }
    }
}
