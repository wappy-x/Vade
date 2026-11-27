//! Form A2 of an expert product on the real weight files: a product that reads the stored 4-bit or 8-bit codes
//! gives the words of the product on the widened BF16 matrix of form A1. Ignored: it reads the public model.
use engine_model::{flash::nvfp4::bf16_to_f32, plan::loop_load::{self, LoadTensor, PartKind}, shapes::{Ffn, Proj}};
use std::{fs::File, os::unix::fs::FileExt, path::PathBuf};

fn local(folder: &str) -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.local").join(folder) }
/// An input row with 24 bits in each value and both signs, or one of BF16 values.
fn input(columns: usize, bf16: bool) -> Vec<f32> {
    (0..columns as u32).map(|k| { let v = (k.wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0;
        if bf16 { f32::from_bits(v.to_bits() & 0xffff_0000) } else { v } }).collect()
}

/// The public model, or a link to it, in `.local/inkling-checkpoint` at the repository root.
/// `cargo test -p engine-model --test loop_packed -- --ignored --nocapture`.
#[test]
#[ignore = "reads the public model folder from .local/inkling-checkpoint"]
fn packed_products_equal_the_products_on_the_widened_matrices() {
    let root = local("inkling-checkpoint");
    assert!(root.join("config.json").is_file(), "put the public model folder, or a link to it, in .local/inkling-checkpoint for the ignored host test");
    let checkpoint = loop_load::read_folder(&root).unwrap();
    let d = &checkpoint.description;
    let read = |t: &LoadTensor, kind: PartKind, matrix: u64| -> Option<Vec<u8>> {
        let part = t.part(kind)?;
        let (offset, bytes) = t.matrix_part(kind, matrix).unwrap();
        let mut out = vec![0u8; bytes as usize];
        File::open(root.join(&part.shard)).unwrap().read_exact_at(&mut out, offset).unwrap();
        Some(out)
    };
    let float = |bytes: Option<Vec<u8>>| bytes.map(|b| f32::from_le_bytes(b.as_slice().try_into().unwrap()));
    // Bytes that the expert products of one token read: every visit picks `selected` experts of its layer.
    let (mut widened_token, mut packed_token) = (0u64, 0u64);
    for visit in &d.visits {
        let Ffn::Experts(e) = d.layers[visit.layer as usize].ffn else { panic!("a layer has no experts") };
        for proj in [Proj::GateUp, Proj::Down] {
            let t = checkpoint.load.experts(visit.layer as u32, proj).unwrap();
            widened_token += e.selected * t.matrix_bytes().unwrap();
            packed_token += e.selected * (t.matrix_part(PartKind::Codes, 0).unwrap().1 + t.matrix_part(PartKind::BlockScale, 0).unwrap().1);
        }
    }
    println!("PACKED TOKEN visits={} widened_bytes={widened_token} packed_bytes={packed_token}", d.visits.len());
    let last = d.layers.len() as u32 - 1;
    let (mut matrices, mut rows_compared, mut different) = (0u64, 0u64, 0u64);
    // The first layer, the last 4-bit layer, the first 8-bit layer and the last layer: the first, second and last expert.
    for layer in [0, 19, 20, last] {
        let Ffn::Experts(e) = d.layers[layer as usize].ffn else { panic!("a layer has no experts") };
        for expert in [0, 1, e.count - 1] {
            for proj in [Proj::GateUp, Proj::Down] {
                let t = checkpoint.load.experts(layer, proj).unwrap();
                let (codes, scales) = (read(t, PartKind::Codes, expert).unwrap(), read(t, PartKind::BlockScale, expert).unwrap());
                let (amax, divisor) = (float(read(t, PartKind::ExpertScale, expert)), float(read(t, PartKind::ScaleDivisor, expert)));
                let mut widened = vec![0u8; t.matrix_bytes().unwrap() as usize];
                t.widen(&codes, &scales, amax, divisor, &mut widened).unwrap();
                let packed = t.packed(&codes, &scales, amax, divisor).unwrap();
                let (rows, columns) = (packed.rows, packed.columns);
                let mut unequal = 0u64;
                for bf16 in [false, true] {
                    let (x, mut out) = (input(columns, bf16), vec![f32::NAN; rows]);
                    packed.product(&x, &mut out).unwrap();
                    // Form A1: the chain of the BF16 row product on the widened matrix.
                    let expected = widened.chunks(columns * 2).map(|row| row.as_chunks::<2>().0.iter().zip(&x).fold(0.0f32, |sum, (w, x)| sum + bf16_to_f32(u16::from_le_bytes(*w)) * x));
                    unequal += out.iter().zip(expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count() as u64;
                    rows_compared += rows as u64;
                }
                println!("PACKED layer={layer} expert={expert} matrix={proj:?} form={} rows={rows} columns={columns} inputs=2 different={unequal} widened_bytes={} packed_bytes={}",
                    if amax.is_some() { "4-bit" } else { "8-bit" }, widened.len(), packed.bytes());
                matrices += 1; different += unequal;
            }
        }
    }
    println!("PACKED SUM matrices={matrices} rows_compared={rows_compared} different={different}");
    assert_eq!(different, 0);
    assert!(matrices == 24 && packed_token < widened_token);
}
