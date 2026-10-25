//! Model directory checks: every shard named by the safetensors index must be
//! present ([`check_model`]), and [`cached`] skips a repeat check while the
//! filesystem inventory identity is unchanged. This does not replace publisher
//! checksum verification or the engine's hashes.
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
    Ok(crate::sha256::sha(&records.iter().flat_map(|row| row.iter().copied().chain([0])).collect::<Vec<_>>()))
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

/// The official FP4-Local index is about 45 MiB (many expert tensors).
/// Bound reads at 64 MiB while accepting the complete current weight map.
pub(super) const MAX_MODEL_INDEX_BYTES: u64 = 64 * 1024 * 1024;

pub(super) fn check_model(path: &Path) -> Result<(), String> {
    let entries = fs::read_dir(path).map_err(|_| {
        "Model weights are missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".to_string()
    })?;
    if !path.join("config.json").is_file() || !path.join("tokenizer.json").is_file() {
        return Err("Model configuration or tokenizer is missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".into());
    }
    let index_path = path.join("model.safetensors.index.json");
    match File::open(&index_path) {
        Ok(index) => {
            let mut text = String::new();
            index
                .take(MAX_MODEL_INDEX_BYTES + 1)
                .read_to_string(&mut text)
                .map_err(|_| {
                    "Cannot read the model shard index. Check model permissions.".to_string()
                })?;
            if text.len() as u64 > MAX_MODEL_INDEX_BYTES {
                return Err(
                    "Model shard index exceeds 64 MiB. Check the configured model directory and INSTALL.md step 3.".into(),
                );
            }
            for shard in indexed_shards(&text)? {
                let metadata = fs::metadata(path.join(shard)).map_err(|_| {
                    "A model shard is missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".to_string()
                })?;
                if !metadata.is_file() || metadata.len() == 0 {
                    return Err("A model shard is empty or invalid. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".into());
                }
            }
            return Ok(());
        }
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                && fs::symlink_metadata(&index_path)
                    .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) => {}
        Err(_) => return Err("Cannot read the model shard index. Check model permissions.".into()),
    }
    let mut weights = false;
    for entry in entries {
        let entry = entry
            .map_err(|_| "Cannot read the model directory. Check its permissions.".to_string())?;
        if entry
            .path()
            .extension()
            .is_some_and(|suffix| suffix == "safetensors")
            && entry
                .metadata()
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() != 0)
        {
            weights = true;
        }
    }
    if !weights {
        return Err(
            "Model weights are missing. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights.".into(),
        );
    }
    Ok(())
}

/// Official safetensors indexes name every shard in JSON string values. This
/// small scanner decodes those strings, including escapes, without a JSON crate.
pub(super) fn indexed_shards(index: &str) -> Result<BTreeSet<String>, String> {
    let invalid = || {
        "Model shard index is invalid. Check the configured model directory; for Flash-Next, follow INSTALL.md step 3 for verified weights."
            .to_string()
    };
    let mut characters = index.trim().chars().peekable();
    if characters.peek() != Some(&'{') {
        return Err(invalid());
    }
    let mut structure = Vec::new();
    let mut shards = BTreeSet::new();
    let mut weight_map = false;
    while let Some(character) = characters.next() {
        match character {
            '{' | '[' => structure.push(character),
            '}' | ']' => {
                if structure.pop() != Some(if character == '}' { '{' } else { '[' }) {
                    return Err(invalid());
                }
            }
            '"' => {
                let mut value = String::new();
                loop {
                    let character = characters.next().ok_or_else(invalid)?;
                    match character {
                        '"' => break,
                        '\\' => {
                            let escaped = characters.next().ok_or_else(invalid)?;
                            value.push(match escaped {
                                '"' | '\\' | '/' => escaped,
                                'b' => '\u{8}',
                                'f' => '\u{c}',
                                'n' => '\n',
                                'r' => '\r',
                                't' => '\t',
                                'u' => {
                                    let mut code = 0;
                                    for _ in 0..4 {
                                        code = code * 16
                                            + characters
                                                .next()
                                                .and_then(|c| c.to_digit(16))
                                                .ok_or_else(invalid)?;
                                    }
                                    char::from_u32(code).ok_or_else(invalid)?
                                }
                                _ => return Err(invalid()),
                            });
                        }
                        character if character <= '\u{1f}' => return Err(invalid()),
                        character => value.push(character),
                    }
                }
                let mut following = characters.clone();
                while following
                    .peek()
                    .is_some_and(|character| character.is_whitespace())
                {
                    following.next();
                }
                let is_key = following.peek() == Some(&':');
                weight_map |= is_key && value == "weight_map";
                if !is_key && value.ends_with(".safetensors") {
                    if !Path::new(&value)
                        .components()
                        .all(|part| matches!(part, std::path::Component::Normal(_)))
                        || value.contains('\\')
                        || value.contains('\0')
                    {
                        return Err(invalid());
                    }
                    shards.insert(value);
                }
            }
            _ => {}
        }
    }
    if !structure.is_empty() || !weight_map || shards.is_empty() {
        return Err(invalid());
    }
    Ok(shards)
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
