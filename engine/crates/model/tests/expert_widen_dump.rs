//! Widened expert weights against the values that the reference model holds after its load.
//! Ignored by default: the dump files are private. They stay outside the
//! tracked tree, in the ignored folder `.local/capped-reference/dumps/` at
//! the repository root. A link is enough.
//!
//! The folder holds `dump-experts-index.json`. For each layer and expert of
//! the index it holds the stored bytes of the two expert matrices and the
//! BF16 values of the loaded reference parameters.
//!
//! The test reads no environment variable. It widens the stored bytes and
//! counts the values that equal the loaded values bit for bit. The loaded
//! gate-and-up matrix holds the gate rows first and then the up rows. The
//! stored matrix keeps its own order.
use engine_model::{capped_moe::STORED_GATE_UP_ROWS, expert_widen::{Stored, StoredMatrix, widen}, json};
use std::path::Path;

fn read(folder: &Path, name: &str) -> Option<Vec<u8>> { std::fs::read(folder.join(name)).ok() }
fn first_f32(bytes: &[u8]) -> f32 { f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) }
fn words(bytes: &[u8]) -> Vec<u16> { bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect() }

/// Widen one stored matrix of the dump. The packed 4-bit format has an `amax` file. The 8-bit format has none.
fn widened(folder: &Path, stem: &str, rows: usize, columns: usize) -> Vec<u16> {
    let (codes, scales) = (read(folder, &format!("{stem}.u8")).expect("stored codes"), read(folder, &format!("{stem}_scale.u8")).expect("stored scales"));
    let data = match read(folder, &format!("{stem}_amax.f32")) {
        Some(amax) => Stored::Packed4 { codes: &codes, scales: &scales, amax: first_f32(&amax),
            denominator: first_f32(&read(folder, &format!("{stem}_outer_scale_denominator.f32")).expect("denominator")) },
        None => Stored::Float8 { values: &codes, scales: &scales },
    };
    widen(&StoredMatrix { rows, columns, data }).unwrap()
}

#[test]
#[ignore = "requires dumps of loaded reference weights outside the repository"]
fn widened_experts_equal_the_loaded_reference_values() {
    let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/capped-reference/dumps");
    assert!(folder.is_dir(), "put dumps in .local/capped-reference/ for the ignored host test");
    let index = json::parse(&std::fs::read(folder.join("dump-experts-index.json")).unwrap(), json::Limits::default()).unwrap();
    let index = index.object().unwrap();
    let list = |name: &str| -> Vec<usize> { index[name].array().unwrap().iter().map(|v| v.u64().unwrap() as usize).collect() };
    let shape_of = |file: &str| -> (usize, usize) {
        let entry = index["files"].array().unwrap().iter().map(|v| v.object().unwrap()).find(|o| o["file"].string().unwrap() == file).expect("file in the index");
        let shape = entry["shape"].array().unwrap();
        (shape[0].u64().unwrap() as usize, shape[1].u64().unwrap() as usize)
    };
    let (mut equal_all, mut total, mut experts) = (0usize, 0usize, 0usize);
    for layer in list("layers") {
        for expert in list("experts") {
            let gate_up_file = format!("loaded-L{layer}-E{expert}.gate_up_proj.bf16");
            let down_file = format!("loaded-L{layer}-E{expert}.down_proj.bf16");
            let ((rows13, hidden), (rows2, width)) = (shape_of(&gate_up_file), shape_of(&down_file));
            assert_eq!((rows13, rows2), (2 * width, hidden));
            let loaded13 = words(&read(&folder, &gate_up_file).expect("loaded gate and up"));
            let loaded2 = words(&read(&folder, &down_file).expect("loaded down"));
            let got13 = widened(&folder, &format!("stored-L{layer}-E{expert}.w13_weight"), rows13, hidden);
            let got2 = widened(&folder, &format!("stored-L{layer}-E{expert}.w2_weight"), rows2, width);
            assert_eq!((loaded13.len(), loaded2.len()), (got13.len(), got2.len()));
            // Loaded row i is the gate row i. Loaded row width + i is the up row i.
            let mut equal13 = 0usize;
            for i in 0..width {
                for (loaded_row, stored_row) in [(i, STORED_GATE_UP_ROWS.gate(i, width)), (width + i, STORED_GATE_UP_ROWS.up(i, width))] {
                    equal13 += loaded13[loaded_row * hidden..(loaded_row + 1) * hidden].iter()
                        .zip(&got13[stored_row * hidden..(stored_row + 1) * hidden]).filter(|(a, b)| a == b).count();
                }
            }
            let equal2 = loaded2.iter().zip(&got2).filter(|(a, b)| a == b).count();
            println!("EXPERT layer {layer} expert {expert}: gate-and-up equal {equal13}/{}; down equal {equal2}/{}", got13.len(), got2.len());
            equal_all += equal13 + equal2; total += got13.len() + got2.len(); experts += 1;
        }
    }
    println!("SUMMARY experts {experts}: widened values equal to the loaded reference values {equal_all}/{total}");
    assert!(experts > 0);
    assert_eq!(equal_all, total, "a widened value differs from the loaded reference value");
}
