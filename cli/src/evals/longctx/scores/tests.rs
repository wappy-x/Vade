//! Native score-capture row layout, truncation and identity tests.
use super::*;
fn native_fixture(vocab: usize) -> (crate::test_gate::TempDir, Value, Vec<u8>, Vec<u8>) {
    let dir = crate::test_gate::TempDir::new("vade-native-capture-test").unwrap();
    let inputs: Vec<u8> = [7u32, 11].into_iter().flat_map(u32::to_le_bytes).collect();
    let logits: Vec<u8> = (0..2 * vocab).flat_map(|i| ((i % vocab) as f32).to_le_bytes()).collect();
    let case = json::parse(&format!(r#"{{"id":"sample","length":2,"prompt_sha256":"frozen","input_ids_sha256":"{}"}}"#, crate::sha256::sha(&inputs))).unwrap();
    let path = dir.path.join("sample");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("metadata.json"), format!(r#"{{"path":"prefill_all","prompt_tokens":2,"prompt_sha256":"frozen","vocab_size":{vocab}}}"#)).unwrap();
    std::fs::write(path.join("logits.f32"), &logits).unwrap();
    std::fs::write(path.join("input_ids.u32"), &inputs).unwrap();
    (dir, case, logits, inputs)
}
#[test]
fn native_capture_row_bytes_follow_declared_vocab() {
    for vocab in [3, 129] {
        let (dir, case, logits, _) = native_fixture(vocab);
        assert_eq!(logits.len(), 2 * vocab * 4);
        let rows = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(rows.len(), 2);
        for (position, row) in rows.iter().enumerate() {
            assert_eq!(row.position, position);
            assert_eq!(row.input_id, [7, 11][position]);
            assert_eq!(row.top1_id.to_string(), (vocab - 1).to_string());
            assert_eq!(row.full_logits.as_ref().unwrap(), &(0..vocab).map(|i| i as f64).collect::<Vec<_>>());
        }
    }
}
#[test]
fn native_capture_refuses_truncated_missing_or_excess_bytes() {
    let (dir, case, logits, inputs) = native_fixture(3);
    let path = dir.path.join("sample");
    for (name, original) in [("logits.f32", &logits), ("input_ids.u32", &inputs)] {
        std::fs::write(path.join(name), &original[..original.len() - 1]).unwrap();
        let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
        assert_eq!(result.err().as_deref(), Some("truncated native dump"));
        let mut extra = original.clone();
        extra.push(0);
        std::fs::write(path.join(name), extra).unwrap();
        let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
        assert_eq!(result.err().as_deref(), Some("native dump size or input IDs differ"));
        std::fs::remove_file(path.join(name)).unwrap();
        assert_eq!(score_rows(&dir.path, &case, 2).err().as_deref(), Some("FileNotFoundError"));
        std::fs::write(path.join(name), original).unwrap();
    }
    std::fs::remove_file(path.join("metadata.json")).unwrap();
    assert_eq!(score_rows(&dir.path, &case, 2).err().as_deref(), Some("FileNotFoundError"));
}
#[test]
fn native_capture_refuses_nonfinite_logits_and_changed_input_ids() {
    let (dir, case, logits, mut inputs) = native_fixture(3);
    let path = dir.path.join("sample");
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = logits.clone();
        bad[..4].copy_from_slice(&value.to_le_bytes());
        std::fs::write(path.join("logits.f32"), bad).unwrap();
        let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
        assert_eq!(result.err().as_deref(), Some("nonfinite native logits"));
    }
    std::fs::write(path.join("logits.f32"), logits).unwrap();
    inputs[0] ^= 1;
    std::fs::write(path.join("input_ids.u32"), inputs).unwrap();
    let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
    assert_eq!(result.err().as_deref(), Some("native dump size or input IDs differ"));
}
