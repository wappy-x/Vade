//! Cache the existing shard-presence check by filesystem inventory identity.
//! This does not replace publisher checksum verification or the engine's hashes.
use super::*;
use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};
fn inventory(path: &Path) -> Result<String, String> {
    let root = fs::canonicalize(path).map_err(|_| "Cannot inspect model directory.")?;
    let mut pending = vec![(root.clone(), 0)];
    let mut records = Vec::new();
    while let Some((path, depth)) = pending.pop() {
        if depth > 32 || records.len() + pending.len() > 100_000 { return Err("Model inventory exceeds limits.".into()); }
        let meta = fs::metadata(&path).map_err(|_| "Cannot inspect model inventory; check missing files and permissions.")?;
        let mut row = path.as_os_str().as_bytes().to_vec();
        row.extend_from_slice(format!("\0{}:{}:{}:{}:{}:{}:{}:{}", meta.dev(), meta.ino(), meta.len(), meta.mode(),
            meta.mtime(), meta.mtime_nsec(), meta.ctime(), meta.ctime_nsec()).as_bytes());
        records.push(row);
        if meta.is_dir() {
            if path != root && fs::symlink_metadata(&path).map_err(|_| "Cannot inspect model directory.")?.file_type().is_symlink() {
                return Err("Model inventory contains a directory symlink; use a resolved model directory.".into());
            }
            for entry in fs::read_dir(&path).map_err(|_| "Cannot list model inventory.")? {
                let entry = entry.map_err(|_| "Cannot list model inventory.")?;
                pending.push((entry.path(), depth + 1));
            }
        }
    }
    records.sort_unstable();
    crate::sha256::digest_reader(records.iter().flat_map(|row| row.iter().copied().chain([0])).collect::<Vec<_>>().as_slice())
        .map_err(|_| "Cannot hash model inventory.".into())
}
pub(super) fn cached(path: &Path, stamp: &Path) -> Result<bool, String> {
    let digest = inventory(path)?;
    let expected = format!("vade-model-inventory-v1\n{digest}\n");
    if fs::read_to_string(stamp).is_ok_and(|text| text == expected) { return Ok(true); }
    check_model(path)?;
    if inventory(path)? != digest { return Err("Model changed during shard check; retry install after the copy finishes.".into()); }
    crate::text::atomic_write(stamp, &expected)?;
    Ok(false)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stamp_hits_and_invalidates_on_nested_changes_and_corruption() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-inventory-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let model = root.join("model"); fs::create_dir_all(model.join("shards")).unwrap();
        for name in ["config.json", "tokenizer.json", "shards/a.safetensors"] { fs::write(model.join(name), "{}").unwrap(); }
        let index = model.join("model.safetensors.index.json");
        fs::write(&index, r#"{"weight_map":{"w":"shards/a.safetensors"}}"#).unwrap();
        let stamp = root.join("stamp");
        assert!(!cached(&model, &stamp).unwrap()); assert!(cached(&model, &stamp).unwrap());
        fs::write(&index, r#"{"weight_map":{"w":"shards/b.safetensors"}}"#).unwrap();
        assert!(cached(&model, &stamp).is_err());
        fs::write(model.join("shards/b.safetensors"), "{}").unwrap();
        assert!(!cached(&model, &stamp).unwrap()); assert!(cached(&model, &stamp).unwrap());
        fs::write(&stamp, "invalid").unwrap(); assert!(!cached(&model, &stamp).unwrap());
        fs::remove_file(model.join("shards/b.safetensors")).unwrap(); assert!(cached(&model, &stamp).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
