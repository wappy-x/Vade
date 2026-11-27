//! Storage groups of a quantization sidecar that lists tensor names. Host metadata only.
//! Each group gives one stored format and the names stored in it. A scale tensor is not
//! listed: it goes with a listed tensor and takes its name with a suffix.
//!
//! Reference: transformers v5.24.0. This file follows these lines and holds its own code.
//! - `core_model_loading.py:772-816`: the two expert formats and their scale layout.
//! - `conversion_mapping.py:2658-2675`: the names of the scale tensors, and the row
//!   interleave of the gate-and-up matrix.
use crate::{Result, invalid, json::{self, Value, field}, safetensors::{self, Dtype}, shapes::CONFIG_BYTES};
use super::{description::LoopTextDescription, storage::{ScaleEncoding, StorageMap, WeightStorage}};
use std::{collections::{BTreeMap, BTreeSet}, path::Path};
type Object = BTreeMap<String, Value>;

pub const SIDECAR_FILE: &str = "hf_quant_config.json";
const METHOD: &str = "inkling2";
const ROOT_KEYS: &[&str] = &["model","mtp","quant_method"];
const GROUP_KEYS: &[&str] = &["compute_dtype","fwd_output_dtype","output_dtype","required_activation_quantization",
    "tensors","weight_block_shape","weight_scale_format","weight_transform"];
/// A quantized tensor name holds its stored layer number after this text.
const LAYER_MARK: &str = ".layers.";
const SCALE: &str = "_scale";
const AMAX: &str = "_amax";
const DENOMINATOR: &str = "_outer_scale_denominator";
/// Values of one row that share one scale.
const NVFP4_BLOCK: u64 = 16;
const MXFP8_BLOCK: u64 = 32;
/// The released sidecar writes `[32, 32]` for the 8-bit block. The stored layout is one
/// scale for 32 values of one row, so both spellings mean the same here. The scale
/// shape that `GroupKind::companions` gives is the check against the file.
const MXFP8_BLOCK_SPELLINGS: [[u64; 2]; 2] = [[1,MXFP8_BLOCK],[MXFP8_BLOCK,MXFP8_BLOCK]];

/// Which half of a byte holds the code of the even column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleOrder { LowFirst }
/// 4-bit codes: the low half of a byte is the even column. Not yet checked by a run.
/// Reference: `core_model_loading.py:777-778`.
pub const NVFP4_NIBBLES: NibbleOrder = NibbleOrder::LowFirst;
/// Row order of a stored matrix that holds the gate rows and the up rows of an expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRows { Interleaved }
/// Stored row `2i` is gate row `i` and stored row `2i + 1` is up row `i`. Not yet checked by a run.
/// Reference: `conversion_mapping.py:2658-2667`.
pub const GATE_UP_ROWS: ExpertRows = ExpertRows::Interleaved;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageGroups {
    pub method: String,
    /// Groups of the main weight file, in file order.
    pub model: Vec<StorageGroup>,
    /// Groups of the draft model file. Empty when the sidecar has none.
    pub draft: Vec<StorageGroup>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageGroup { pub kind: GroupKind, pub tensors: Vec<String> }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    /// Stored as is. `forward` is the type of the result of a product with this tensor.
    Plain { dtype: Dtype, forward: Dtype },
    /// A stack of matrices. U8 codes, two 4-bit codes a byte. One F8_E4M3 scale for 16
    /// values of one row. One F32 `amax` for each matrix and one F32 denominator.
    Nvfp4,
    /// A stack of matrices. F8_E4M3 values. One U8 power-of-two scale for 32 values of one row.
    Mxfp8,
}
/// A scale tensor that goes with one listed tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Companion { pub name: String, pub dtype: Dtype, pub shape: Vec<u64> }

fn require(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn known(o: &Object, what: &str, keys: &[&str]) -> Result<()> {
    match o.keys().find(|k| !keys.contains(&k.as_str())) {
        Some(key) => Err(invalid(format!("unknown {what} field {key}"))),
        None => Ok(()),
    }
}
/// The sidecar also says how the maker turns the input of a product into the group's
/// format. This reader checks that recipe and keeps none of it: the group kind implies it.
fn activation(v: &Value, block: u64, strings: &[(&str,&str)]) -> Result<()> {
    let o = v.object().map_err(|_| invalid("a quantized group needs required_activation_quantization"))?;
    known(o,"activation",&strings.iter().map(|&(key,_)| key).chain(["block_size"]).collect::<Vec<_>>())?;
    require(field(o,"block_size")?.u64()? == block,"unsupported activation block_size")?;
    for &(key,value) in strings {
        let actual = field(o,key)?.string()?;
        if actual != value { return Err(invalid(format!("unsupported activation {key} {actual}"))); }
    }
    Ok(())
}
fn group(v: &Value) -> Result<StorageGroup> {
    let o = v.object()?;
    known(o,"storage group",GROUP_KEYS)?;
    let text = |key: &str| -> Result<&str> { field(o,key)?.string() };
    let block = field(o,"weight_block_shape")?.array()?.iter().map(Value::u64).collect::<Result<Vec<_>>>()?;
    let recipe = field(o,"required_activation_quantization")?;
    let format = o.get("weight_scale_format").map(Value::string).transpose()?;
    let forward = o.get("fwd_output_dtype").map(Value::string).transpose()?;
    let plain = |dtype: Dtype| -> Result<GroupKind> {
        require(block.is_empty() && format.is_none() && recipe == &Value::Null,"a cast group has a block, a scale format or an activation recipe")?;
        let forward = match forward { None => dtype, Some("float32") => Dtype::F32, Some("bfloat16") => Dtype::Bf16,
            Some(other) => return Err(invalid(format!("unsupported fwd_output_dtype {other}"))) };
        Ok(GroupKind::Plain { dtype,forward })
    };
    let kind = match (text("compute_dtype")?,text("output_dtype")?,text("weight_transform")?) {
        ("float32","float32","cast") => plain(Dtype::F32)?,
        ("bfloat16","bfloat16","cast") => plain(Dtype::Bf16)?,
        ("nvfp4","uint8","nvfp4_nearest") => {
            require(block == [1,NVFP4_BLOCK] && format == Some("amax_denominator") && forward.is_none(),"unsupported 4-bit block, scale format or forward type")?;
            activation(recipe,NVFP4_BLOCK,&[("amax_type","per_token"),("format","nvfp4_e2m1"),("rounding","nearest"),
                ("scale_dtype","float8_e4m3fn"),("scale_selection_rule","fouroversix_mae")])?;
            GroupKind::Nvfp4
        }
        ("mxfp8","float8_e4m3fn","mxfp8_nearest") => {
            require(MXFP8_BLOCK_SPELLINGS.iter().any(|b| block == b) && format.is_none() && forward.is_none(),"unsupported 8-bit block, scale format or forward type")?;
            activation(recipe,MXFP8_BLOCK,&[("format","mxfp8_e4m3"),("rounding","nearest"),("scale_dtype","ue8m0")])?;
            GroupKind::Mxfp8
        }
        (compute,output,transform) => return Err(invalid(format!(
            "unsupported storage group: compute_dtype {compute}, output_dtype {output}, weight_transform {transform}"))),
    };
    let tensors = field(o,"tensors")?.array()?.iter().map(|v| v.string().map(str::to_owned)).collect::<Result<Vec<_>>>()?;
    require(!tensors.is_empty() && tensors.iter().all(|name| !name.is_empty()),"a storage group has no tensor or an empty name")?;
    Ok(StorageGroup { kind,tensors })
}
fn section(v: &Value, what: &str) -> Result<Vec<StorageGroup>> {
    let o = v.object()?;
    known(o,what,&["parameter_groups"])?;
    let groups = field(o,"parameter_groups")?.array()?.iter().map(group).collect::<Result<Vec<_>>>()?;
    require(!groups.is_empty(),"a sidecar section has no storage group")?;
    let mut seen = BTreeSet::new();
    for name in groups.iter().flat_map(|g| &g.tensors) {
        if !seen.insert(name.as_str()) { return Err(invalid(format!("tensor {name} is in two storage groups"))); }
    }
    Ok(groups)
}
fn layer_of(name: &str) -> Option<usize> {
    name.split_once(LAYER_MARK)?.1.split_once('.')?.0.parse().ok()
}

impl StorageGroups {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let o = doc.object()?;
        known(o,"sidecar",ROOT_KEYS)?;
        let method = field(o,"quant_method")?.string()?;
        if method != METHOD { return Err(invalid(format!("unsupported sidecar quant_method {method}"))); }
        Ok(Self { method: method.to_owned(), model: section(field(o,"model")?,"model section")?,
            draft: o.get("mtp").map(|v| section(v,"mtp section")).transpose()?.unwrap_or_default() })
    }
    /// The group that lists this exact name in the main weight file.
    pub fn group_of(&self, name: &str) -> Option<&StorageGroup> {
        self.model.iter().find(|g| g.tensors.iter().any(|t| t == name))
    }
    /// Listed names of each group of the main weight file, in file order.
    pub fn counts(&self) -> Vec<usize> { self.model.iter().map(|g| g.tensors.len()).collect() }
    fn quantized(&self) -> impl Iterator<Item = &StorageGroup> {
        self.model.iter().filter(|g| !matches!(g.kind,GroupKind::Plain { .. }))
    }
    /// Storage of the quantized tensors of each stored layer. `Plain` for a layer that has none.
    pub fn layer_storage(&self, d: &LoopTextDescription) -> Result<Vec<WeightStorage>> {
        let mut found = vec![None; d.layers.len()];
        for g in self.quantized() {
            for name in &g.tensors {
                let layer = layer_of(name).filter(|&i| i < found.len())
                    .ok_or_else(|| invalid(format!("quantized tensor {name} names no layer of config.json")))?;
                let (storage,count) = found[layer].get_or_insert((g.kind.storage(),0usize));
                if *storage != g.kind.storage() { return Err(invalid(format!("layer {layer} has tensors in two storage formats"))); }
                *count += 1;
            }
        }
        require(found.iter().flatten().map(|&(_,count)| count).collect::<BTreeSet<_>>().len() <= 1,
            "layers differ in their count of quantized tensors")?;
        Ok(found.into_iter().map(|f| f.map_or(WeightStorage::Plain,|(storage,_)| storage)).collect())
    }
    /// The storage map, after the cross-check with config.json. The key is the full tensor name.
    pub fn storage_map(&self, d: &LoopTextDescription) -> Result<StorageMap> {
        match &d.declared_storage {
            None => require(self.quantized().next().is_none(),"the sidecar has quantized groups and config.json declares no storage")?,
            Some(s) if s.method == "modelopt" && s.algo == "NVFP4" => require(self.model.iter().any(|g| g.kind == GroupKind::Nvfp4),
                "config.json declares NVFP4 storage and the sidecar has no 4-bit group")?,
            Some(s) => return Err(invalid(format!("unsupported declared storage: quant_method {}, quant_algo {}",s.method,s.algo))),
        }
        self.layer_storage(d)?;
        Ok(StorageMap { excluded: Vec::new(), layers: self.quantized()
            .flat_map(|g| g.tensors.iter().map(|name| (name.clone(),g.kind.storage()))).collect() })
    }
}
impl GroupKind {
    pub fn stored_dtype(&self) -> Dtype {
        match *self { Self::Plain { dtype,.. } => dtype, Self::Nvfp4 => Dtype::U8, Self::Mxfp8 => Dtype::F8E4M3 }
    }
    /// The file stores no activation scale tensor for a 4-bit group.
    pub fn storage(&self) -> WeightStorage {
        match self { Self::Plain { .. } => WeightStorage::Plain, Self::Mxfp8 => WeightStorage::Mxfp8,
            Self::Nvfp4 => WeightStorage::Nvfp4 { encoding: ScaleEncoding::AmaxDenominator, activation_scale: false } }
    }
    /// The scale tensors that go with one listed tensor, from its stored shape.
    pub fn companions(&self, name: &str, stored_shape: &[u64]) -> Result<Vec<Companion>> {
        let (block,dtype) = match self { Self::Plain { .. } => return Ok(Vec::new()),
            Self::Nvfp4 => (NVFP4_BLOCK,Dtype::F8E4M3), Self::Mxfp8 => (MXFP8_BLOCK,Dtype::U8) };
        let &[matrices,rows,stored] = stored_shape else { return Err(invalid(format!("quantized tensor {name} is not a stack of matrices"))); };
        // A byte of 4-bit codes holds two values.
        let columns = if *self == Self::Nvfp4 { stored.checked_mul(2).ok_or_else(|| invalid("column count overflow"))? } else { stored };
        if columns == 0 || !columns.is_multiple_of(block) { return Err(invalid(format!("tensor {name}: columns do not fill complete scale blocks"))); }
        let mut out = vec![Companion { name: format!("{name}{SCALE}"), dtype, shape: vec![matrices,rows,columns / block] }];
        if *self == Self::Nvfp4 {
            out.push(Companion { name: format!("{name}{AMAX}"), dtype: Dtype::F32, shape: vec![matrices] });
            out.push(Companion { name: format!("{name}{DENOMINATOR}"), dtype: Dtype::F32, shape: vec![1] });
        }
        Ok(out)
    }
}
/// Read config.json and the sidecar with the bounded reader and cross-check them.
/// No weight file is opened.
pub fn read_folder(root: &Path) -> Result<(LoopTextDescription, StorageGroups, StorageMap)> {
    let d = LoopTextDescription::from_config(&safetensors::read_bounded(&root.join("config.json"),CONFIG_BYTES)?)
        .map_err(|e| invalid(format!("config.json: {e}")))?;
    let groups = StorageGroups::from_json(&safetensors::read_bounded(&root.join(SIDECAR_FILE),CONFIG_BYTES)?)
        .map_err(|e| invalid(format!("{SIDECAR_FILE}: {e}")))?;
    let map = groups.storage_map(&d)?;
    Ok((d,groups,map))
}
