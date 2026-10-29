//! Model file checks read JSON and file sizes. They do not read tensor data.
use crate::{checks::Report, text::{Fields, to_hex}};
use engine_model::json::{self, Value};
use std::{collections::{BTreeMap, BTreeSet}, fs, io::Read, path::{Component, Path}};

const JSON_LIMIT: u64 = 64 * 1024 * 1024;
// Three inline files use at most 6 MiB of hex. The inventory uses at most
// 1 MiB. Larger files travel in 4 MiB chunks (8 MiB of hex), below the
// runner's 16 MiB stdout limit on every command.
const INLINE_LIMIT: u64 = 1024 * 1024;
const CHUNK_BYTES: u64 = 4 * 1024 * 1024;
const _: () = assert!(INLINE_LIMIT * 7 + 64 * 1024 < crate::runner::MAX_CAPTURE_BYTES);
const _: () = assert!(CHUNK_BYTES * 2 < crate::runner::MAX_CAPTURE_BYTES);

/// Report the files in a local checkpoint folder.
pub(super) fn local(report: &mut Report, folder: &Path) -> Result<(), String> {
    let mut fields = Fields::new();
    let root = fs::canonicalize(folder).ok().filter(|root| root.is_dir());
    let valid_folder = root.is_some();
    fields.insert("model_folder".into(), if valid_folder { "present" } else { "missing" }.into());
    if !valid_folder {
        for key in ["config", "tokenizer", "index"] { fields.insert(format!("model_{key}"), "unreadable".into()); }
        inventory_fields(&mut fields, &[], false);
        return rows(report, "", &folder.to_string_lossy(), &fields);
    }
    let root = root.expect("checked folder");
    for (key, name) in JSON_FILES {
        let path = root.join(name);
        let path = fs::canonicalize(&path).ok().filter(|path| path.starts_with(&root));
        let state = match path.as_ref().map(fs::metadata).transpose() {
            Ok(Some(metadata)) if metadata.is_file() && metadata.len() > 0 && metadata.len() <= JSON_LIMIT => {
                match fs::File::open(path.as_ref().expect("checked path")).and_then(|file| {
                    let mut bytes = Vec::new();
                    file.take(JSON_LIMIT + 1).read_to_end(&mut bytes)?;
                    Ok(bytes)
                }) {
                    Ok(bytes) if bytes.len() as u64 <= JSON_LIMIT => {
                        fields.insert(format!("model_{key}_hex"), to_hex(&bytes));
                        "present"
                    }
                    _ => "unreadable",
                }
            }
            Ok(None) if !root.join(name).exists() => "missing",
            _ => "unreadable",
        };
        fields.insert(format!("model_{key}"), state.into());
    }
    let mut shards = Vec::new();
    let mut complete = true;
    let names = json_field(&fields, "index").and_then(|index| index_names(&index));
    let names = match names {
        Ok(names) => names,
        Err(_) => {
            let mut names = BTreeSet::new();
            match fs::read_dir(&root) {
            Ok(entries) => for entry in entries {
                let Ok(entry) = entry else { complete = false; continue; };
                let name = entry.file_name();
                let Some(name) = name.to_str() else { complete = false; continue; };
                if shard_name(name) { names.insert(name.to_string()); }
            },
            Err(_) => complete = false,
            }
            names
        }
    };
    for name in names {
        let Some(path) = fs::canonicalize(root.join(&name)).ok().filter(|path| path.starts_with(&root)) else { continue; };
        if let Ok(metadata) = fs::metadata(&path) {
            if metadata.is_file() && fs::File::open(&path).is_ok() { shards.push((name, metadata.len())); }
        }
    }
    shards.sort_by(|a, b| a.0.cmp(&b.0));
    inventory_fields(&mut fields, &shards, complete);
    rows(report, "", &folder.to_string_lossy(), &fields)
}

/// A read-only Linux script for the existing SSH path. Hex keeps JSON and file
/// names inside the CLI's one-line record format.
pub(super) fn script(folder: &str) -> String {
    format!(r#"{start}
if test "$vade_model_ok" = 1 && test -d "$vade_model"; then printf 'model_folder=present\n'; else vade_model_ok=0; printf 'model_folder=missing\n'; fi
vade_model_json() {{
  vade_key=$1
  vade_file=$2
  if test "$vade_model_ok" != 1; then printf 'model_%s=unreadable\n' "$vade_key"
  elif ! test -e "$vade_file"; then printf 'model_%s=missing\n' "$vade_key"
  elif vade_model_file "$vade_file"; then
    vade_size=$(vade_model_size "$vade_resolved")
    if test "$vade_size" -gt 0 && test "$vade_size" -le {limit}; then
      printf 'model_%s=present\nmodel_%s_size=%s\n' "$vade_key" "$vade_key" "$vade_size"
      if test "$vade_size" -le {inline}; then
        printf 'model_%s_hex=' "$vade_key"
        head -c {inline_read} "$vade_resolved" | od -An -v -tx1 | tr -d ' \n'
        printf '\n'
      fi
    else printf 'model_%s=unreadable\n' "$vade_key"; fi
  else printf 'model_%s=unreadable\n' "$vade_key"; fi
}}
vade_model_json config "$vade_model/config.json"
vade_model_json tokenizer "$vade_model/tokenizer.json"
vade_model_json index "$vade_model/model.safetensors.index.json"
vade_count=0
vade_bytes=0
vade_complete=$vade_model_ok
vade_inventory_bytes=0
if test "$vade_model_ok" = 1; then
for vade_file in "$vade_model"/*.safetensors "$vade_model"/.[!.]*.safetensors "$vade_model"/..?*.safetensors; do
  test -e "$vade_file" || test -L "$vade_file" || continue
  if ! vade_model_file "$vade_file"; then vade_complete=0; continue; fi
  vade_size=$(vade_model_size "$vade_resolved")
  vade_shard_name=${{vade_file##*/}}
  vade_inventory_bytes=$((vade_inventory_bytes + ${{#vade_shard_name}} * 2 + 80))
  if test "$vade_inventory_bytes" -gt {inline}; then vade_complete=0; break; fi
  printf 'model_shard.%s=' "$vade_count"
  printf '%s' "${{vade_file##*/}}" | od -An -v -tx1 | tr -d ' \n'
  printf ':%s\n' "$vade_size"
  vade_count=$((vade_count + 1))
  vade_bytes=$((vade_bytes + vade_size))
done
fi
printf 'model_shards=%s\nmodel_bytes=%s\nmodel_complete=%s\n' "$vade_count" "$vade_bytes" "$vade_complete"
"#, start=script_start(folder), limit=JSON_LIMIT, inline=INLINE_LIMIT, inline_read=INLINE_LIMIT + 1)
}

const JSON_FILES: [(&str, &str); 3] = [("config", "config.json"), ("tokenizer", "tokenizer.json"), ("index", "model.safetensors.index.json")];

fn script_start(folder: &str) -> String {
    format!(r#"LC_ALL=C
export LC_ALL
vade_model=$(realpath -- {folder} 2>/dev/null)
vade_model_ok=0
if test -n "$vade_model" && test -d "$vade_model"; then vade_model_ok=1; fi
vade_model_size() {{ stat -c %s -- "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }}
vade_model_file() {{
  test "$vade_model_ok" = 1 || return 1
  vade_resolved=$(realpath -- "$1" 2>/dev/null) || return 1
  case "$vade_resolved" in "$vade_model"/*) ;; *) return 1;; esac
  test -f "$vade_resolved" && test -r "$vade_resolved"
}}
"#, folder=crate::runner::remote_path(folder))
}

/// Complete large metadata reads and inspect the index-named paths. Each
/// command has a bounded reply below the runner's stdout limit. A file error
/// stays on its model row so disk, memory and port checks can still run.
pub(super) fn remote<R: crate::runner::Runner>(runner: &mut R, machine: &crate::config::Machine, fields: &mut Fields) {
    for (key, file) in JSON_FILES {
        if fields.get(&format!("model_{key}")).map(String::as_str) != Some("present") || fields.contains_key(&format!("model_{key}_hex")) { continue; }
        let result = read_remote_json(runner, machine, file, fields.get(&format!("model_{key}_size")));
        match result {
            Ok(value) => { fields.insert(format!("model_{key}_hex"), value); }
            Err(error) => { fields.insert(format!("model_{key}_error"), error); }
        }
    }
    let Ok(names) = json_field(fields, "index").and_then(|index| index_names(&index)) else { return; };
    // The first reply includes root-level shards. Read nested or truncated
    // entries only when the index names them. Unused files cannot fail a
    // checkpoint that the launcher accepts.
    let mut shards = BTreeMap::new();
    for (key, value) in fields.iter() {
        if !key.starts_with("model_shard.") { continue; }
        if let Some((name, bytes)) = value.split_once(':').and_then(|(name, bytes)| unhex(name).ok().zip(bytes.parse::<u64>().ok())) {
            if names.contains(&name) { shards.insert(name, bytes); }
        }
    }
    let missing = names.into_iter().filter(|name| !shards.contains_key(name)).collect::<Vec<_>>();
    let mut offset = 0;
    while offset < missing.len() {
        let mut command = script_start(&machine.model_path);
        let first = offset;
        while let Some(name) = missing.get(offset) {
            // A path this long cannot resolve on the supported hosts.
            if name.len() > 16 * 1024 { offset += 1; continue; }
            let quoted = crate::runner::quote(name);
            if command.len() + quoted.len() > 24 * 1024 && offset > first { break; }
            command += &format!("if vade_model_file \"$vade_model\"/{quoted}; then printf 'model_shard.{}={}:%s\\n' \"$(vade_model_size \"$vade_resolved\")\"; fi\n", offset - first, to_hex(name.as_bytes()));
            offset += 1;
        }
        let result = runner.run(&crate::runner::ssh(&machine.ssh, command, &format!("{} doctor indexed files", machine.name)))
            .and_then(|output| if output.success { crate::text::parse(&output.stdout) } else { Err("Cannot read indexed model files through SSH.".into()) });
        if let Ok(reply) = result {
            for value in reply.values() {
                if let Some((name, bytes)) = value.split_once(':').and_then(|(name, bytes)| unhex(name).ok().zip(bytes.parse::<u64>().ok())) {
                    if missing[first..offset].contains(&name) { shards.insert(name, bytes); }
                }
            }
        }
    }
    fields.retain(|key, _| !key.starts_with("model_shard."));
    inventory_fields(fields, &shards.into_iter().collect::<Vec<_>>(), true);
}

fn read_remote_json<R: crate::runner::Runner>(runner: &mut R, machine: &crate::config::Machine, file: &str, size: Option<&String>) -> Result<String, String> {
    let size = size.and_then(|value| value.parse::<u64>().ok()).filter(|&size| size > 0 && size <= JSON_LIMIT)
        .ok_or_else(|| format!("The {file} size record is missing or outside its limit."))?;
    let mut encoded = String::with_capacity(size as usize * 2);
    for offset in (0..size).step_by(CHUNK_BYTES as usize) {
        let command = format!("{}\nif vade_model_file \"$vade_model/{file}\"; then dd if=\"$vade_resolved\" bs=1048576 skip={} count=4 2>/dev/null | od -An -v -tx1 | tr -d ' \\n'; fi\n", script_start(&machine.model_path), offset / (1024 * 1024));
        let output = runner.run(&crate::runner::ssh(&machine.ssh, command, &format!("{} doctor {file} chunk", machine.name)))?;
        let expected = (size - offset).min(CHUNK_BYTES) as usize * 2;
        if !output.success || output.stdout.len() != expected || !output.stdout.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("Cannot read a complete bounded chunk of {file}; check SSH and file permissions, then run vade doctor."));
        }
        encoded.push_str(&output.stdout);
    }
    Ok(encoded)
}

/// Report a remote inventory. Invalid or absent fields fail the relevant row.
pub(super) fn rows(report: &mut Report, prefix: &str, folder: &str, fields: &Fields) -> Result<(), String> {
    let label = |check: &str| if prefix.is_empty() { check.to_string() } else { format!("{prefix} {check}") };
    let fix = "Set the model path to a complete checkpoint folder, then run vade doctor.";
    if fields.get("model_folder").is_some_and(|value| value == "present") {
        report.ok(&label("model folder"), folder)?;
    } else {
        report.fail(&label("model folder"), &format!("The model folder is missing or unreadable at {folder}."), fix)?;
    }
    for (key, file) in [("config", "config.json"), ("tokenizer", "tokenizer.json")] {
        report.check(&label(file), json_field(fields, key), |_| format!("{file} is valid JSON"),
            &format!("Copy a complete {file} from the same checkpoint, then run vade doctor."))?;
    }
    let inventory = inventory(fields);
    report.check(&label("safetensors"), inventory.as_ref().map_err(Clone::clone), |shards| {
        let bytes: u64 = shards.values().copied().sum();
        format!("{} files, {bytes} bytes", shards.len())
    }, "Copy all nonempty safetensors files from the checkpoint, then run vade doctor.")?;
    let result = json_field(fields, "index").and_then(|index| match &inventory {
        Ok(shards) => check_index(&index, shards),
        Err(_) if fields.get("model_shards").map(String::as_str) == Some("0") && fields.get("model_complete").map(String::as_str) == Some("1") => check_index(&index, &BTreeMap::new()),
        Err(error) => Err(error.clone()),
    });
    report.check(&label("model index"), result, |_| "every named shard is present and nonempty".into(),
        "Copy the index and all its named shards from the same checkpoint, then run vade doctor.")?;
    Ok(())
}

fn inventory_fields(fields: &mut Fields, shards: &[(String, u64)], complete: bool) {
    for (index, (name, bytes)) in shards.iter().enumerate() {
        fields.insert(format!("model_shard.{index}"), format!("{}:{bytes}", to_hex(name.as_bytes())));
    }
    fields.insert("model_shards".into(), shards.len().to_string());
    let total = shards.iter().try_fold(0u64, |total, (_, bytes)| total.checked_add(*bytes));
    fields.insert("model_bytes".into(), total.unwrap_or(0).to_string());
    fields.insert("model_complete".into(), u8::from(complete && total.is_some()).to_string());
}

fn inventory(fields: &Fields) -> Result<BTreeMap<String, u64>, String> {
    let count = fields.get("model_shards").and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0 && count <= fields.len()).ok_or("No safetensors files were found.")?;
    if fields.get("model_complete").map(String::as_str) != Some("1") { return Err("Some safetensors files are unreadable or leave the checkpoint folder.".into()); }
    let mut shards = BTreeMap::new();
    let mut total = 0u64;
    for index in 0..count {
        let (name, bytes) = fields.get(&format!("model_shard.{index}")).and_then(|value| value.split_once(':'))
            .ok_or("The safetensors inventory is incomplete.")?;
        let name = unhex(name)?;
        let bytes = bytes.trim().parse::<u64>().ok().filter(|&bytes| bytes > 0).ok_or("A safetensors file is empty.")?;
        if !shard_name(&name) || shards.insert(name, bytes).is_some() { return Err("A safetensors file name is invalid or repeated.".into()); }
        total = total.checked_add(bytes).ok_or("The safetensors size exceeds the supported range.")?;
    }
    if fields.get("model_bytes").and_then(|value| value.parse::<u64>().ok()) != Some(total) { return Err("The safetensors size record is incomplete.".into()); }
    Ok(shards)
}

/// The checked total from regular, readable, nonempty tensor files.
pub(super) fn bytes(fields: &Fields) -> Result<u64, String> {
    inventory(fields).map(|shards| shards.values().copied().sum())
}

fn json_field(fields: &Fields, key: &str) -> Result<Value, String> {
    if let Some(error) = fields.get(&format!("model_{key}_error")) { return Err(error.clone()); }
    if fields.get(&format!("model_{key}")).map(String::as_str) != Some("present") {
        return Err(format!("The {key} JSON file is missing, empty, unreadable, or larger than {JSON_LIMIT} bytes."));
    }
    let encoded = fields.get(&format!("model_{key}_hex")).filter(|value| value.len() as u64 <= JSON_LIMIT * 2)
        .ok_or_else(|| format!("The {key} JSON record is missing or too large."))?;
    let decoded = unhex(encoded)?;
    match json::parse(decoded.as_bytes(), json::Limits { bytes: JSON_LIMIT as usize, nodes: 2_000_000, ..Default::default() }) {
        Ok(value @ Value::Object(_)) => Ok(value),
        _ => Err(format!("The {key} file does not contain a valid JSON object.")),
    }
}

fn check_index(index: &Value, shards: &BTreeMap<String, u64>) -> Result<(), String> {
    for name in index_names(index)? {
        if !shards.contains_key(&name) { return Err(format!("The model index names a missing shard: {name}.")); }
    }
    Ok(())
}

fn shard_name(value: &str) -> bool {
    value.ends_with(".safetensors") && !value.contains('\\') && !value.chars().any(char::is_control) && value != ".safetensors"
        && Path::new(value).components().all(|part| matches!(part, Component::Normal(_)))
}

fn index_names(index: &Value) -> Result<BTreeSet<String>, String> {
    let Some(Value::Object(map)) = index.object().ok().and_then(|object| object.get("weight_map")) else { return Err("The model index has no weight_map object.".into()); };
    if map.is_empty() { return Err("The model index has no tensor entries.".into()); }
    map.values().map(|value| value.string().ok().filter(|name| shard_name(name)).map(str::to_string)
        .ok_or_else(|| "The model index has an invalid shard name.".into())).collect()
}

fn unhex(value: &str) -> Result<String, String> {
    let digit = |byte: u8| match byte { b'0'..=b'9' => Some(byte - b'0'), b'a'..=b'f' => Some(byte - b'a' + 10), b'A'..=b'F' => Some(byte - b'A' + 10), _ => None };
    if !value.len().is_multiple_of(2) { return Err("The model file record has invalid text.".into()); }
    let bytes = value.as_bytes().as_chunks::<2>().0.iter().map(|[high, low]| digit(*high).zip(digit(*low)).map(|(a, b)| a * 16 + b))
        .collect::<Option<Vec<_>>>().ok_or("The model file record has invalid text.")?;
    String::from_utf8(bytes).map_err(|_| "The model file record is not UTF-8 text.".into())
}

#[cfg(test)]
#[path = "model_remote_tests.rs"]
mod remote_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn folder() -> std::path::PathBuf {
        let folder = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-doctor-model-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir_all(&folder).unwrap();
        folder
    }
    pub(super) fn fixture(folder: &Path) {
        fs::write(folder.join("config.json"), "{\"hidden_size\":16}").unwrap();
        fs::write(folder.join("tokenizer.json"), "{\"model\":{}}").unwrap();
        fs::write(folder.join("part-1.safetensors"), [0u8; 16]).unwrap();
        fs::write(folder.join("part-2.safetensors"), [0u8; 24]).unwrap();
        fs::write(folder.join("model.safetensors.index.json"), "{\"weight_map\":{\"a\":\"part-1.safetensors\",\"b\":\"part-2.safetensors\"}}").unwrap();
    }
    pub(super) fn captured(folder: &Path) -> (Result<(), String>, String) {
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || {
            let mut report = Report::new("local", false);
            local(&mut report, folder)?;
            report.finish()
        });
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        (result, printed)
    }
    #[test]
    fn local_model_lists_count_and_bytes_without_tensor_reads() {
        let folder = folder();
        fixture(&folder);
        let (result, printed) = captured(&folder);
        result.unwrap();
        assert!(printed.contains("2 files, 40 bytes"), "{printed}");
        assert!(printed.contains("every named shard is present and nonempty"), "{printed}");
    }
    #[test]
    fn local_model_reports_bad_json_and_missing_shard() {
        let folder = folder();
        fixture(&folder);
        fs::write(folder.join("tokenizer.json"), "{").unwrap();
        fs::write(folder.join("model.safetensors.index.json"), "{\"weight_map\":{\"a\":\"absent.safetensors\"}}").unwrap();
        let (result, printed) = captured(&folder);
        assert!(result.is_err());
        assert!(printed.contains("tokenizer file does not contain a valid JSON object"), "{printed}");
        assert!(printed.contains("missing shard: absent.safetensors"), "{printed}");
        assert!(printed.contains("Copy a complete tokenizer.json"), "{printed}");
    }
    #[test]
    fn remote_script_matches_the_local_inventory_with_quoted_path() {
        let _guard = crate::runner::process_test_lock();
        let folder = folder().join("model ' spaced");
        fs::create_dir(&folder).unwrap();
        fixture(&folder);
        let output = std::process::Command::new("sh").arg("-c").arg(script(folder.to_str().unwrap())).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let fields = crate::text::parse(&String::from_utf8(output.stdout).unwrap()).unwrap();
        let shards = inventory(&fields).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards.values().copied().sum::<u64>(), 40);
        check_index(&json_field(&fields, "index").unwrap(), &shards).unwrap();
        json_field(&fields, "config").unwrap();
        json_field(&fields, "tokenizer").unwrap();
    }
    #[test]
    fn empty_weights_and_index_paths_fail() {
        let folder = folder();
        fixture(&folder);
        fs::write(folder.join("part-1.safetensors"), []).unwrap();
        let (result, printed) = captured(&folder);
        assert!(result.is_err());
        assert!(printed.contains("safetensors file is empty"), "{printed}");
        let shards = BTreeMap::from([("part-1.safetensors".into(), 16)]);
        let value = json::parse(b"{\"weight_map\":{\"a\":\"../part-1.safetensors\"}}", json::Limits::default()).unwrap();
        assert!(check_index(&value, &shards).is_err());
    }
    #[test]
    fn missing_index_fails_before_startup() {
        let folder = folder();
        fs::write(folder.join("config.json"), "{}").unwrap();
        fs::write(folder.join("tokenizer.json"), "{}").unwrap();
        fs::write(folder.join("model.safetensors"), [0u8; 16]).unwrap();
        let (result, printed) = captured(&folder);
        assert!(result.is_err());
        assert!(printed.contains("FAIL  model index"), "{printed}");
        assert!(printed.contains("Copy the index and all its named shards"), "{printed}");
    }
    #[test]
    fn model_metadata_rejects_non_json_numbers() {
        for name in ["config.json", "tokenizer.json", "model.safetensors.index.json"] {
            let folder = folder();
            fixture(&folder);
            fs::write(folder.join(name), "{\"value\":NaN}").unwrap();
            let (result, printed) = captured(&folder);
            assert!(result.is_err(), "{name}: {printed}");
            assert!(printed.contains("does not contain a valid JSON object"), "{name}: {printed}");
        }
    }
    #[test]
    fn tokenizer_with_a_large_vocabulary_has_one_complete_json_check() {
        use std::fmt::Write;
        let folder = folder();
        fixture(&folder);
        let mut tokenizer = String::from("{\"model\":{\"vocab\":{");
        for index in 0..10_000 {
            if index > 0 { tokenizer.push(','); }
            write!(tokenizer, "\"token{index}\":{index}").unwrap();
        }
        tokenizer.push_str("}}}");
        fs::write(folder.join("tokenizer.json"), tokenizer).unwrap();
        let (result, printed) = captured(&folder);
        result.unwrap();
        assert!(printed.contains("tokenizer.json is valid JSON"), "{printed}");
    }
}
