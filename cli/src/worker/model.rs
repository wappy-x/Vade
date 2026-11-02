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
        "Model weights are missing. Check the configured model directory; for Flash-Next, follow docs/INSTALL.md step 3 for verified weights.".to_string()
    })?;
    if !path.join("config.json").is_file() || !path.join("tokenizer.json").is_file() {
        return Err("Model configuration or tokenizer is missing. Check the configured model directory; for Flash-Next, follow docs/INSTALL.md step 3 for verified weights.".into());
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
                    "Model shard index exceeds 64 MiB. Check the configured model directory and docs/INSTALL.md step 3.".into(),
                );
            }
            for shard in indexed_shards(&text)? {
                let metadata = fs::metadata(path.join(shard)).map_err(|_| {
                    "A model shard is missing. Check the configured model directory; for Flash-Next, follow docs/INSTALL.md step 3 for verified weights.".to_string()
                })?;
                if !metadata.is_file() || metadata.len() == 0 {
                    return Err("A model shard is empty or invalid. Check the configured model directory; for Flash-Next, follow docs/INSTALL.md step 3 for verified weights.".into());
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
            "Model weights are missing. Check the configured model directory; for Flash-Next, follow docs/INSTALL.md step 3 for verified weights.".into(),
        );
    }
    Ok(())
}

/// Official safetensors indexes name every shard in JSON string values. This
/// small scanner decodes those strings, including escapes, without a JSON crate.
pub(super) fn indexed_shards(index: &str) -> Result<BTreeSet<String>, String> {
    let invalid = || {
        "Model shard index is invalid. Check the configured model directory; for Flash-Next, follow docs/INSTALL.md step 3 for verified weights."
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

/// Where an FP4-Local engine keeps its checkpoint receipt, chosen by the
/// controller's `model_receipt` field. Without one the engine writes the
/// receipt into the checkpoint on rank 0 only, and a chain whose ranks differ
/// in that history moves the file between them before the load.
/// A checksum list supplied with the checkpoint is copied in at every start,
/// so its hashes stay the strict expectation on every rank, in both modes.
/// Without one, `private` keeps one receipt per checkpoint path in this
/// helper's state folder across starts, so a different checkpoint never
/// inherits another's hashes; `pipeline` starts every rank of a chain from no
/// receipt, and a supplied list goes in as [`chain_manifest`] gives it. Older
/// controllers send no field and keep the engine's default.
pub(crate) fn receipt(root: &Path, model: &Path, request: &Fields, environment: &mut Fields) -> Result<(), String> {
    let pipeline = match request.get("model_receipt").map(String::as_str) {
        None => return Ok(()),
        Some("private") => false,
        Some("pipeline") => true,
        Some(_) => return Err("Unknown checkpoint receipt mode. Install the same Vade on the controller and Sparks.".into()),
    };
    // Other formats keep their own receipt contracts in the checkpoint.
    if !engine_model::flash::Config::open(&model.join("config.json")).is_ok_and(|config| config.mixed.is_some()) {
        return Ok(());
    }
    let name = if pipeline { "pipeline-manifest.sha256".into() } else {
        let checkpoint = fs::canonicalize(model).map_err(|_| "Cannot inspect model directory.")?;
        format!("model-manifest-{}.sha256", &crate::sha256::sha(checkpoint.as_os_str().as_bytes())[..16])
    };
    let path = root.join(&name);
    let supplied = crate::launch::supplied_manifest(model)?;
    if pipeline || supplied.is_some() {
        for file in [path.clone(), root.join(format!("{name}.sizes"))] {
            crate::text::reject_symlinks(&file)?;
            match fs::remove_file(&file) {
                Ok(()) => (),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                Err(_) => return Err("Cannot clear the previous checkpoint receipt; check the state folder.".into()),
            }
        }
    }
    if let Some(manifest) = supplied {
        let manifest = if pipeline { chain_manifest(&manifest).unwrap_or(manifest) } else { manifest };
        crate::text::atomic_write(&path, &manifest)?;
    }
    environment.remove("env_unset.FLASH_NVFP4_MANIFEST");
    environment.insert("env.FLASH_NVFP4_MANIFEST".into(), path.to_string_lossy().into_owned());
    Ok(())
}

/// A chain compares the bytes of this file between ranks, so two correct lists
/// of a different shape refuse to start: `vade pull` lists every file, and a
/// rank with no list gets the engine's own. This gives the engine's shape: the
/// index row, then the weight files by name. Equal weights give equal bytes on
/// every rank, and a different hash still stops the chain. A row that is not
/// valid gives `None`, so the engine sees the list as it is and refuses it.
fn chain_manifest(supplied: &str) -> Option<String> {
    const INDEX: &str = "model.safetensors.index.json";
    let mut rows = std::collections::BTreeMap::new();
    for row in supplied.lines().filter(|row| !row.is_empty()) {
        let (hash, mark, name) = (row.get(..64)?, row.get(64..66)?, row.get(66..)?);
        if !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) || !matches!(mark, "  " | " *") { return None; }
        if (name == INDEX || name.ends_with(".safetensors")) && rows.insert(name, hash.to_ascii_lowercase()).is_some() { return None; }
    }
    let index = rows.remove(INDEX)?;
    Some(std::iter::once(format!("{index}  {INDEX}\n")).chain(rows.iter().map(|(name, hash)| format!("{hash}  {name}\n"))).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chain_ranks_with_a_full_list_and_a_short_list_write_the_same_bytes() {
        let list = include_str!("../../examples/qwen3.8-flash-next-fp4-local.sha256");
        let full = list.replace("  ./", "  ");
        let chain = chain_manifest(&full).unwrap();
        assert_eq!(chain.lines().count(), 22);
        assert!(chain.starts_with(&full.lines().find(|row| row.ends_with("  model.safetensors.index.json")).map(|row| format!("{row}\n")).unwrap()));
        // The digest that the engine's own list had on a real pair: the same bytes.
        assert_eq!(crate::sha256::sha(chain.as_bytes()), "d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716");
        // The list from one rank, as the other rank sees it, gives the same file again.
        assert_eq!(chain_manifest(&chain).as_deref(), Some(chain.as_str()));
        assert_eq!(chain_manifest(&full.replace("  ", " *")).as_deref(), Some(chain.as_str()));
        assert_eq!(chain_manifest(&full.to_ascii_uppercase().replace(".SAFETENSORS", ".safetensors").replace("MODEL.safetensors.INDEX.JSON", "model.safetensors.index.json").replace("TENSORS/MODEL", "tensors/model").replace("-OF-", "-of-")).as_deref(), Some(chain.as_str()));
        // Another weight file is still another list, so the chain stops.
        let first = full.lines().find(|row| row.ends_with(".safetensors")).unwrap();
        let changed = full.replacen(&first[..64], &"0".repeat(64), 1);
        assert_ne!(chain_manifest(&changed).unwrap(), chain);
        // A list the engine would refuse reaches the engine as it is.
        for bad in ["not a row", &format!("{}", "z".repeat(70)), &format!("{full}{first}\n")] { assert_eq!(chain_manifest(bad), None, "{bad}"); }
        assert_eq!(chain_manifest(&full.replace("model.safetensors.index.json", "other.json")), None);
    }

    #[test]
    fn a_chain_start_writes_the_short_list_and_a_private_start_keeps_the_full_one() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-receipt-{}", crate::text::random_hex(8).unwrap()));
        let state = root.join("state");
        crate::text::private_dir(&state).unwrap();
        let [leftover, pulled] = ["leftover/model", "pulled/model"].map(|name| {
            let model = root.join(name); fs::create_dir_all(&model).unwrap();
            fs::write(model.join("config.json"), crate::launch::test_mixed_config(32768)).unwrap();
            model
        });
        let list = include_str!("../../examples/qwen3.8-flash-next-fp4-local.sha256");
        let short = chain_manifest(&list.replace("  ./", "  ")).unwrap();
        fs::write(leftover.join("manifest.sha256"), &short).unwrap();
        fs::write(root.join("pulled/model.sha256"), list).unwrap();
        let start = |model: &Path, mode: &str| {
            let mut environment = Fields::new();
            receipt(&state, model, &fields(&[("model_receipt", mode)]), &mut environment).unwrap();
            fs::read_to_string(&environment["env.FLASH_NVFP4_MANIFEST"]).unwrap()
        };
        assert_eq!(start(&leftover, "pipeline"), start(&pulled, "pipeline"));
        assert_eq!(start(&pulled, "private"), list.replace("  ./", "  "));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fp4_local_receipts_stay_private_and_chain_ranks_start_without_one() {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-receipt-{}", crate::text::random_hex(8).unwrap()));
        let state = root.join("state");
        crate::text::private_dir(&state).unwrap();
        let [a, b] = ["a", "b"].map(|name| {
            let model = root.join(name); fs::create_dir_all(&model).unwrap();
            fs::write(model.join("config.json"), crate::launch::test_mixed_config(32768)).unwrap();
            model
        });
        let unset = fields(&[("env_unset.FLASH_NVFP4_MANIFEST", "1")]);
        let mode = |name: &str| fields(&[("model_receipt", name)]);
        let start = |model: &Path, name: &str| {
            let mut environment = unset.clone();
            receipt(&state, model, &mode(name), &mut environment).unwrap();
            assert_eq!(environment.len(), 1);
            PathBuf::from(&environment["env.FLASH_NVFP4_MANIFEST"])
        };
        // An older controller sends no mode: the engine default stays.
        let mut environment = unset.clone();
        receipt(&state, &a, &Fields::new(), &mut environment).unwrap();
        assert_eq!(environment, unset);
        // Each checkpoint keeps its own receipt across starts: A -> B -> A.
        let private = start(&a, "private");
        assert!(private.starts_with(&state) && !private.exists());
        fs::write(&private, "receipt of a\n").unwrap();
        assert_eq!(start(&a, "private"), private);
        let other = start(&b, "private");
        assert_ne!(other, private);
        assert!(!other.exists());
        fs::write(&other, "receipt of b\n").unwrap();
        assert_eq!(start(&a, "private"), private);
        assert_eq!(fs::read_to_string(&private).unwrap(), "receipt of a\n");
        // A chain rank drops the receipt its previous start left behind.
        for name in ["pipeline-manifest.sha256", "pipeline-manifest.sha256.sizes"] { fs::write(state.join(name), "old").unwrap(); }
        assert_eq!(start(&a, "pipeline"), state.join("pipeline-manifest.sha256"));
        assert!(!state.join("pipeline-manifest.sha256").exists() && !state.join("pipeline-manifest.sha256.sizes").exists());
        // A supplied checksum list replaces either receipt, in the engine's row form.
        let hash = "a".repeat(64);
        let mut sibling = a.as_os_str().to_owned();
        sibling.push(".sha256");
        fs::write(&sibling, format!("{hash}  ./model.safetensors\n")).unwrap();
        for name in ["private", "pipeline"] {
            assert_eq!(fs::read_to_string(start(&a, name)).unwrap(), format!("{hash}  model.safetensors\n"));
        }
        assert!(receipt(&state, &a, &mode("shared"), &mut environment).is_err());
        // FP8 receipts stay where the engine keeps them.
        fs::write(a.join("config.json"), crate::launch::test_config(32768)).unwrap();
        let mut environment = unset.clone();
        receipt(&state, &a, &mode("pipeline"), &mut environment).unwrap();
        assert_eq!(environment, unset);
        fs::remove_dir_all(root).unwrap();
    }
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
