//! Mixed storage recipes for metadata inspection. No device storage is admitted.
use crate::{Result, invalid, json::{Value, field}, shapes::CONFIG_BYTES};
use std::collections::BTreeMap;
type Object = BTreeMap<String, Value>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleEncoding { Direct, Csf, Csf32 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightStorage {
    Plain,
    Mxfp8,
    Fp8Block { block: [u64; 2] },
    Nvfp4 { encoding: ScaleEncoding, activation_scale: bool },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageMap { pub excluded: Vec<String>, pub layers: BTreeMap<String, WeightStorage> }
fn require(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn block(o: &Object) -> Result<Option<Vec<u64>>> {
    o.get("weight_block_size").map(|v| v.array()?.iter().map(Value::u64).collect()).transpose()
}
impl StorageMap {
    /// Accept config.json's quantization object or hf_quant_config.json's wrapper.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let doc = crate::json::parse(bytes, crate::json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let o = doc.object()?;
        Self::parse(o.get("quantization").or_else(|| o.get("quantization_config")).unwrap_or(&doc))
    }
    pub(super) fn parse(value: &Value) -> Result<Self> {
        let o = value.object()?;
        require(field(o, "quant_method")?.string()? == "modelopt"
            && field(o, "quant_algo")?.string()? == "MIXED_PRECISION", "unsupported mixed storage method")?;
        if let Some(v) = o.get("kv_cache_quant_algo") { require(v == &Value::Null, "unsupported KV cache recipe")?; }
        let mut excluded = Vec::new();
        for key in ["exclude_modules", "ignore"] {
            if let Some(v) = o.get(key) { for p in v.array()? { excluded.push(p.string()?.to_owned()); } }
        }
        let mut layers = BTreeMap::new();
        for (name, value) in field(o, "quantized_layers")?.object()? {
            require(!name.is_empty(), "empty storage target")?;
            // Vision is outside this text description. Do not inspect its recipes.
            if name.starts_with("model.visual.") { continue; }
            let r = value.object()?;
            let group = r.get("group_size").map(Value::u64).transpose()?;
            let block = block(r)?;
            let encoding = r.get("weight_scale_encoding").map(Value::string).transpose()?;
            let algo = field(r, "quant_algo")?.string()?;
            require(encoding.is_none() || matches!(algo, "NVFP4" | "W4A16_NVFP4"), "scale encoding requires FP4 storage")?;
            let storage = match algo {
                "MXFP8" => {
                    require(group.is_none_or(|g| g == 32) && block.as_deref().is_none_or(|b| b == [1,32])
                        && (group.is_some() || block.is_some()), "invalid MXFP8 block")?;
                    WeightStorage::Mxfp8
                }
                "FP8_PB_WO" => {
                    let b = block.ok_or_else(|| invalid("missing FP8 weight block"))?;
                    require(b.len() == 2 && b.iter().all(|&v| v > 0 && v <= 4096) && group.is_none(), "invalid FP8 weight block")?;
                    WeightStorage::Fp8Block { block: [b[0], b[1]] }
                }
                "NVFP4" | "W4A16_NVFP4" => {
                    require(group == Some(16) && block.is_none(), "invalid NVFP4 group")?;
                    let encoding = match encoding { None => ScaleEncoding::Direct, Some("csf") => ScaleEncoding::Csf,
                        Some("csf_32x2_v1") => ScaleEncoding::Csf32, Some(_) => return Err(invalid("unsupported scale encoding")) };
                    WeightStorage::Nvfp4 { encoding, activation_scale: algo == "NVFP4" }
                }
                _ => return Err(invalid(format!("unsupported quant_algo {algo}"))),
            };
            layers.insert(name.clone(), storage);
        }
        require(!layers.is_empty(), "empty text storage map")?;
        Ok(Self { excluded, layers })
    }
    pub fn storage(&self, module: &str) -> WeightStorage {
        if self.excluded.iter().any(|p| glob(p.as_bytes(), module.as_bytes())) { return WeightStorage::Plain; }
        self.layers.iter().filter(|(p, _)| module == p.as_str()
            || module.strip_prefix(p.as_str()).is_some_and(|s| s.starts_with('.')))
            .max_by_key(|(p, _)| p.len()).map_or(WeightStorage::Plain, |(_, &s)| s)
    }
}
fn glob(pattern: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t, mut star, mut retry) = (0, 0, None, 0);
    while t < text.len() {
        if p < pattern.len() && pattern[p] != b'*' && pattern[p] == text[t] { p += 1; t += 1; }
        else if p < pattern.len() && pattern[p] == b'*' { star = Some(p); p += 1; retry = t; }
        else if let Some(s) = star { retry += 1; t = retry; p = s + 1; }
        else { return false; }
    }
    while p < pattern.len() && pattern[p] == b'*' { p += 1; }
    p == pattern.len()
}
