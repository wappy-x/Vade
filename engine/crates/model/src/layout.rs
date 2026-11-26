//! Read a header fixture and a written loader audit. This is not a load plan.
//! Model names, tensor roles and operation gaps stay in the audit data.
use crate::{Result, invalid, json::{self, Value, field}, safetensors::{self, Header, Index, Limits}};
use std::{collections::{BTreeMap, BTreeSet}, io::{self, Cursor, Read, Seek, SeekFrom}, path::Path};

const FILE_LIMIT: usize = 1024 * 1024;
const TOTAL_LIMIT: usize = 16 * FILE_LIMIT;

/// A written audit joined to checked fixture metadata. It cannot enable execution.
#[derive(Debug)]
pub struct LayoutReport { pub text: String }

fn read(root: &Path, name: &str, budget: &mut usize) -> Result<Vec<u8>> {
    let path = root.join(name).canonicalize()?;
    if !path.starts_with(root) { return Err(invalid("fixture path escapes its directory")); }
    let bytes = safetensors::read_bounded(&path, FILE_LIMIT.min(*budget))?;
    *budget = budget.checked_sub(bytes.len()).ok_or_else(|| invalid("fixture byte limit exceeded"))?;
    Ok(bytes)
}

/// The reader exposes a declared length for header checks. It has no payload.
struct HeaderOnly { prefix: Cursor<Vec<u8>>, length: u64 }
impl Read for HeaderOnly {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if !bytes.is_empty() && self.prefix.position() >= self.prefix.get_ref().len() as u64 {
            return Err(io::Error::other("fixture payload read refused"));
        }
        self.prefix.read(bytes)
    }
}
impl Seek for HeaderOnly {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::End(n) => self.length.checked_add_signed(n),
            SeekFrom::Current(n) => self.prefix.position().checked_add_signed(n),
        }.ok_or_else(|| io::Error::other("invalid fixture seek"))?;
        self.prefix.set_position(position);
        Ok(position)
    }
}

fn header(bytes: &[u8]) -> Result<Header> {
    let doc = json::parse(bytes, json::Limits { bytes: FILE_LIMIT, ..Default::default() })?;
    let mut payload = 0;
    for (name, value) in doc.object()? {
        if name == "__metadata__" { continue; }
        let offsets = field(value.object()?, "data_offsets")?.array()?;
        if offsets.len() != 2 { return Err(invalid("fixture tensor needs two offsets")); }
        payload = payload.max(offsets[1].u64()?);
    }
    let mut prefix = (bytes.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(bytes);
    let length = (prefix.len() as u64).checked_add(payload).ok_or_else(|| invalid("fixture length overflow"))?;
    Header::read(&mut HeaderOnly { prefix: Cursor::new(prefix), length }, Limits {
        header_bytes: FILE_LIMIT, ..Default::default()
    })
}

fn text<'a>(o: &'a BTreeMap<String, Value>, name: &str) -> Result<&'a str> { field(o, name)?.string() }

impl LayoutReport {
    /// Read config, index, audit and `<shard>.header.json` files only.
    /// Header shapes can be synthetic. The audit must state its evidence limits.
    pub fn open_fixture(root: &Path) -> Result<Self> {
        let root = root.canonicalize()?;
        let mut budget = TOTAL_LIMIT;
        let config = read(&root, "config.json", &mut budget)?;
        let config_doc = json::parse(&config, json::Limits { bytes: FILE_LIMIT, ..Default::default() })?;
        let config_obj = config_doc.object()?;
        let index = Index::parse_mixed(&read(&root, "model.safetensors.index.json", &mut budget)?, Limits {
            header_bytes: FILE_LIMIT, ..Default::default()
        })?;
        let audit = json::parse(&read(&root, "audit.json", &mut budget)?, json::Limits {
            bytes: FILE_LIMIT, ..Default::default()
        })?;
        let audit = audit.object()?;
        let roles = field(audit, "tensors")?.object()?;
        let mut tensors = BTreeMap::new();
        let mut total = 0u64;
        for shard in index.weight_map.values().collect::<BTreeSet<_>>() {
            let h = header(&read(&root, &format!("{shard}.header.json"), &mut budget)?)?;
            for (name, tensor) in h.tensors {
                if index.weight_map.get(&name) != Some(shard) {
                    return Err(invalid("fixture header tensor is absent from index or in the wrong shard"));
                  }
                total = total.checked_add(tensor.byte_len()).ok_or_else(|| invalid("fixture tensor size overflow"))?;
                if tensors.insert(name, tensor).is_some() { return Err(invalid("duplicate fixture tensor")); }
            }
        }
        if !tensors.keys().eq(index.weight_map.keys()) { return Err(invalid("fixture index differs from header tensors")); }
        if index.size_declared && total != index.total_size { return Err(invalid("fixture index total_size differs from headers")); }
        if !tensors.keys().eq(roles.keys()) { return Err(invalid("fixture audit differs from header tensors")); }
        let mut out = String::from("inspection=header-fixture; execution=not-qualified\n");
        out.push_str(&format!("source={:?}\nscope={:?}\n", text(audit, "source")?, text(audit, "scope")?));
        out.push_str("gaps=written-audit; storage-checks=current-loader-parser\n");
        out.push_str("header-checks=byte-geometry; logical-scale-pairing=not-validated\n");
        out.push_str(&format!("model_type={:?}\n", text(config_obj, "model_type")?));
        match crate::plan::ModelPlan::from_config(&config) {
            Ok(_) => out.push_str("config_plan=accepted; execution=not-qualified\n"),
            Err(e) => out.push_str(&format!("gap=config-plan reason={e:?}\n")),
        }
        // Keep the full schema failure. Check each declared recipe as well, so a
        // missing outer field does not hide the rest of the storage gaps.
        if let Some(q) = config_obj.get("quantization_config") {
            match q.object().and_then(crate::flash::mixed::Mixed::parse) {
                Ok(_) => out.push_str("mixed_schema=accepted; execution=not-qualified\n"),
                Err(e) => out.push_str(&format!("gap=mixed-schema reason={e:?}\n")),
            }
            let recipes = q.object().and_then(|q| field(q, "quantized_layers")?.object().map(|layers| (q, layers)));
            if let Ok((q, layers)) = recipes {
                let mut probe = BTreeMap::from([
                    ("quant_method".into(), Value::String("modelopt".into())),
                    ("quant_algo".into(), Value::String("MIXED_PRECISION".into())),
                    ("kv_cache_quant_algo".into(), Value::Null),
                    ("ignore".into(), Value::Array(Vec::new())),
                ]);
                if let Some(exclude) = q.get("exclude_modules") { probe.insert("exclude_modules".into(), exclude.clone()); }
                for (name, recipe) in layers {
                    probe.insert("quantized_layers".into(), Value::Object(BTreeMap::from([(name.clone(), recipe.clone())])));
                    let result = match crate::flash::mixed::Mixed::parse(&probe) {
                        Ok(_) => "recipe-schema-accepted; tensors-not-qualified".to_owned(),
                        Err(e) => format!("gap: {e}"),
                    };
                    out.push_str(&format!("recipe={name:?} declared={recipe:?} isolated_check={result:?}\n"));
                }
            } else if let Err(e) = recipes {
                out.push_str(&format!("gap=mixed-recipes reason={e:?}\n"));
            }
        }
        for op in field(audit, "operations")?.array()? {
            let op = op.object()?;
            out.push_str(&format!("op={:?} evidence={:?} gap={:?}\n",
                text(op, "name")?, text(op, "evidence")?, text(op, "gap")?));
        }
        for (name, tensor) in &tensors {
            let role = roles[name].object()?;
            out.push_str(&format!("tensor={name:?} dtype={:?} shape={:?} role={:?} loader={:?} gap={:?}\n",
                tensor.dtype, tensor.shape, text(role, "role")?, text(role, "loader")?, text(role, "gap")?));
        }
        out.push_str(&format!("fixture_tensors={} declared_payload_bytes={total} payload_bytes_read=0\n", tensors.len()));
        Ok(Self { text: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn header_reader_has_no_payload() {
        let bytes = br#"{"w":{"dtype":"U8","shape":[16,16],"data_offsets":[0,256]}}"#;
        let h = header(bytes).unwrap();
        assert_eq!(h.tensors["w"].shape, [16,16]);
        assert_eq!(h.file_bytes - h.data_start, 256);
        let mut reader = HeaderOnly { prefix: Cursor::new(vec![]), length: 256 };
        assert!(reader.read(&mut [0]).is_err());
    }
    #[test]
    fn malformed_header_is_rejected() {
        for bytes in [
            r#"{"w":{"dtype":"U8","shape":[16,16],"data_offsets":[0,255]}}"#,
            r#"{"w":{"dtype":"U8","shape":[16,16],"data_offsets":[1,257]}}"#,
            r#"{"w":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,1]}}"#,
            r#"{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,18446744073709551615]}}"#,
        ] { assert!(header(bytes.as_bytes()).is_err()); }
    }
}
