use engine_model::safetensors::{Index, Limits, TensorFile};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "engine-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn shard(&self, name: &str, tensor: &str, payload: &[u8]) {
        let header = format!(
            r#"{{"{tensor}":{{"dtype":"U8","shape":[{}],"data_offsets":[0,{}]}}}}"#,
            payload.len(),
            payload.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(payload);
        fs::write(self.0.join(name), bytes).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn index(size: u64, map: &str) -> Index {
    Index::parse(
        format!(r#"{{"metadata":{{"total_size":{size}}},"weight_map":{map}}}"#).as_bytes(),
        Limits::default(),
    )
    .unwrap()
}

#[test]
fn tensor_reads_preserve_bounds_and_byte_order() {
    let f = Fixture::new();
    f.shard("a.safetensors", "a", &[17, 5, 233, 0, 128]);
    let reader = TensorFile::open(&f.0.join("a.safetensors"), Limits::default()).unwrap();
    let mut bytes = [0; 3];
    reader.read_range("a", 1, &mut bytes).unwrap();
    assert_eq!(bytes, [5, 233, 0]);
    assert!(reader.read_range("a", 3, &mut bytes).is_err());
    assert!(reader.read_range("a", u64::MAX, &mut bytes).is_err());
    assert!(reader.read_range("absent", 0, &mut []).is_err());
    reader.read_range("a", 5, &mut []).unwrap();
    assert!(reader.read_range("a", 6, &mut []).is_err());
}

#[cfg(unix)]
#[test]
fn concurrent_reads_use_retained_inode_and_independent_offsets() {
    let f = Fixture::new();
    let payload: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
    f.shard("a.safetensors", "a", &payload);
    let reader = TensorFile::open(&f.0.join("a.safetensors"), Limits::default()).unwrap();
    fs::rename(f.0.join("a.safetensors"), f.0.join("retained.safetensors")).unwrap();
    f.shard("a.safetensors", "a", &[0; 8192]);
    std::thread::scope(|scope| {
        for lane in 0..4 {
            let reader = &reader;
            let payload = &payload;
            scope.spawn(move || {
                for i in 0..64 {
                    let offset = (i * 101 + lane * 307) % (8192 - 511);
                    let mut bytes = [0; 511];
                    reader.read_range("a", offset as u64, &mut bytes).unwrap();
                    assert_eq!(bytes, payload[offset..offset + bytes.len()]);
                }
            });
        }
    });
}

#[test]
fn index_reconciles_every_tensor_and_total() {
    let f = Fixture::new();
    f.shard("a.safetensors", "a", &[1, 2]);
    f.shard("b.safetensors", "b", &[3]);
    let map = r#"{"a":"a.safetensors","b":"b.safetensors"}"#;
    assert_eq!(
        index(3, map)
            .inspect_shards(&f.0, Limits::default())
            .unwrap()
            .len(),
        2
    );
    assert!(
        index(4, map)
            .inspect_shards(&f.0, Limits::default())
            .is_err()
    );
    assert!(
        index(3, r#"{"b":"a.safetensors","a":"b.safetensors"}"#)
            .inspect_shards(&f.0, Limits::default())
            .is_err()
    );
    assert!(
        index(2, r#"{"missing":"a.safetensors"}"#)
            .inspect_shards(&f.0, Limits::default())
            .is_err()
    );
    assert!(
        index(2, r#"{"a":"absent.safetensors"}"#)
            .inspect_shards(&f.0, Limits::default())
            .is_err()
    );
}

#[cfg(unix)]
#[test]
fn shard_symlink_cannot_escape_checkpoint_root() {
    let f = Fixture::new();
    let outside = Fixture::new();
    outside.shard("a.safetensors", "a", &[1, 2]);
    std::os::unix::fs::symlink(outside.0.join("a.safetensors"), f.0.join("a.safetensors")).unwrap();
    assert!(
        index(2, r#"{"a":"a.safetensors"}"#)
            .inspect_shards(&f.0, Limits::default())
            .is_err()
    );

}

#[test]
fn truncation_after_open_surfaces_as_error() {
    let f = Fixture::new();
    f.shard("a.safetensors", "a", &[1, 2]);
    let path = f.0.join("a.safetensors");
    let reader = TensorFile::open(&path, Limits::default()).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(8)
        .unwrap();
    assert!(reader.read_range("a", 0, &mut [0; 2]).is_err());
}

#[test]
fn fp8_storage_2d_and_3d_preserves_shape_dtype_and_payload() {
    use engine_model::safetensors::Dtype;
    for shape in [vec![129, 257], vec![2, 129, 257]] {
        let f = Fixture::new();
        let payload: Vec<u8> = (0..shape.iter().product::<usize>())
            .map(|i| i as u8)
            .collect();
        let header = format!(
            r#"{{"w":{{"dtype":"F8_E4M3","shape":{shape:?},"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(&payload);
        for name in [
            "fp8.safetensors",
            "model part.safetensors",
            "模型.safetensors",
            "model+copy.safetensors",
        ] {
            let path = f.0.join(name);
            fs::write(&path, &bytes).unwrap();
            let parsed = index(payload.len() as u64, &format!(r#"{{"w":"{name}"}}"#));
            parsed.inspect_shards(&f.0, Limits::default()).unwrap();
            let reader = TensorFile::open(&path, Limits::default()).unwrap();
            let tensor = &reader.header().tensors["w"];
            assert_eq!(tensor.dtype, Dtype::F8E4M3);
            assert_eq!(
                tensor.shape,
                shape.iter().map(|&n| n as u64).collect::<Vec<_>>()
            );
            assert_eq!(tensor.byte_len(), payload.len() as u64);
            let mut actual = vec![0; payload.len()];
            reader.read_range("w", 0, &mut actual).unwrap();
            assert_eq!(actual, payload); // Includes every FP8 bit pattern, without conversion.
            fs::remove_file(path).unwrap();
        }
    }
    assert!(Dtype::parse("F8_E5M2").is_err());
}
#[test]
fn fp8_column_shards_use_global_block_scale_coordinates() {
    use engine_model::{flash::plan::Selection, safetensors::{Dtype, Tensor}};
    let weight = Tensor { dtype: Dtype::F8E4M3, shape: vec![129, 640], start: 13, end: 13 + 129 * 640 };
    let scales = Tensor { dtype: Dtype::Bf16, shape: vec![2, 5], start: 7, end: 27 };
    let selected = Selection::Columns(128..384);
    let scale_selection = Selection::Columns(1..3);
    assert_eq!(selected.bytes(&weight), 33024);
    assert_eq!(scale_selection.bytes(&scales), 8);
    for (local, global, scale_local) in [
        (0, 128, 0), (127, 255, 0), (128, 256, 2), (255, 383, 2),
        (256, 768, 0), (32767, 81663, 2), (32768, 82048, 4), (33023, 82303, 6),
    ] {
        assert_eq!(selected.global_byte_offset(&weight, local).unwrap(), global);
        assert_eq!(selected.local_byte_offset(&weight, global), Some(local));
        let cell = (global / 640 / 128) * 5 + (global % 640) / 128;
        assert_eq!(scale_selection.local_byte_offset(&scales, cell * 2), Some(scale_local));
        assert_eq!(scale_selection.local_byte_offset(&scales, cell * 2 + 1), Some(scale_local + 1));
    }
    assert_eq!(selected.local_byte_offset(&weight, 127), None);
    assert_eq!(selected.local_byte_offset(&weight, 384), None);
    assert!(selected.global_byte_offset(&weight, 33024).is_err());
    assert_eq!(selected.spans(&weight).nth(128), Some((82048, 256)));
}

#[test]
fn fp8_stacked_expert_rows_and_empty_owners_preserve_bytes() {
    use engine_model::{flash::plan::{RankPlan, Selection}, safetensors::{Dtype, Tensor}};
    let tensor = Tensor { dtype: Dtype::F8E4M3, shape: vec![3, 129, 257], start: 0, end: 99459 };
    let selected = Selection::Rows(vec![0..1, 2..3]);
    assert_eq!(selected.bytes(&tensor), 66306);
    assert_eq!(selected.spans(&tensor).collect::<Vec<_>>(), [(0, 33153), (66306, 33153)]);
    for (local, global) in [(33152, 33152), (33153, 66306), (66305, 99458)] {
        assert_eq!(selected.global_byte_offset(&tensor, local).unwrap(), global);
        assert_eq!(selected.local_byte_offset(&tensor, global), Some(local));
    }
    assert_eq!(selected.local_byte_offset(&tensor, 33153), None);
    for slot in 5..8 {
        let plan = RankPlan::new(slot, 8).unwrap();
        let selection = plan.selection("model.language_model.layers.0.mlp.experts.0.gate_proj.weight").unwrap();
        assert_eq!(selection.bytes(&tensor), 0);
        assert_eq!(selection.spans(&tensor).count(), 0);
        assert!(selection.global_byte_offset(&tensor, 0).is_err());
        assert_eq!(selection.local_byte_offset(&tensor, 0), None);
    }
}
