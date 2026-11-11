//! Metadata and full-row contract shared by native gate adapters; no payload reads.
use engine_model::{plan::ModelPlan, shapes::Residual};
use std::{fs, path::Path};
#[path = "../../../../tools/export/src/sha256.rs"]
mod sha256;
type Result<T> = std::result::Result<T, String>;

pub struct GatePlan {
    pub model: ModelPlan,
    pub world: usize,
    pub rank: usize,
    pub row_bytes: usize,
    identity: String,
}
impl GatePlan {
    pub fn inspect(root: &Path, world: usize, rank: usize) -> Result<Self> {
        let (model, index, headers) = ModelPlan::inspect_parts(root).map_err(|e| e.to_string())?;
        let allowed: &[usize] = match model.shape.residual {
            Residual::Single => &[1, 2], Residual::Streams { .. } => &[2, 4],
        };
        if !allowed.contains(&world) || rank >= world { return Err("gate model/world/rank unsupported".into()); }
        let row_bytes = usize::try_from(model.shape.vocab).ok().and_then(|n| n.checked_mul(4))
            .filter(|&n| n != 0 && n <= isize::MAX as usize).ok_or("invalid gate vocabulary size")?;
        // Ordered metadata, including physical offsets and shard names; no absolute roots.
        let identity = sha256::digest(format!("gate-plan-v1\n{model:?}\n{index:?}\n{headers:?}").as_bytes());
        Ok(Self { model, world, rank, row_bytes, identity })
    }
    pub fn fixture(&self, prompt: &[u32], rows: &[u32]) -> Result<()> {
        if ![285, 2048, 8192].contains(&prompt.len()) || rows.len() < 64
            || prompt.len().checked_add(rows.len()).is_none_or(|n| n as u64 > self.model.shape.max_positions)
            || prompt.iter().chain(rows).any(|&t| u64::from(t) >= self.model.shape.vocab) {
            return Err("gate needs valid 285/2048/8192 fixtures and at least 64 continuation rows".into());
        }
        Ok(())
    }
    fn prefix(&self) -> String {
        format!("gate_plan model={} world={} rank={} row_bytes={} ", self.identity, self.world, self.rank, self.row_bytes)
    }
    pub fn receipt(&self, tokens: &[u8], groups: [usize; 2]) -> String {
        format!("{}fixtures={} groups={},{}\n", self.prefix(), sha256::digest(tokens), groups[0], groups[1])
    }
    /// A run's stamp is immutable across depths and cases. REF is never a PASS.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn bind(&self, dir: &Path, receipt: &str) -> Result<()> {
        if !receipt.starts_with(&self.prefix()) { return Err("gate plan receipt differs".into()); }
        let path = dir.join("gate-plan.conf");
        if path.exists() {
            if fs::read_to_string(path).map_err(|e| e.to_string())? != receipt {
                return Err("gate model/world/fixtures/groups differ from captures".into());
            }
        } else {
            use std::io::Write;
            fs::OpenOptions::new().write(true).create_new(true).open(path)
                .and_then(|mut f| f.write_all(receipt.as_bytes())).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn check_dumps(&self, reference: &Path, actual: &Path) -> Result<()> {
        let stamp = |file: &Path| file.ancestors().skip(1).find_map(|p| {
            let path = p.join("gate-plan.conf"); path.exists().then_some(path)
        }).ok_or_else(|| "missing gate plan capture receipt".to_owned())
            .and_then(|p| engine_model::safetensors::read_bounded(&p, 4096).map_err(|e| e.to_string()));
        let expected = stamp(reference)?; let observed = stamp(actual)?;
        if expected != observed || !expected.starts_with(self.prefix().as_bytes()) {
            return Err("gate dump model/world/rank/fixtures/groups differ".into());
        }
        Ok(())
    }
}
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn row(logits: Option<&[f32]>, width: usize) -> Result<Vec<u8>> {
    let logits = logits.ok_or("missing full-row gate logits")?;
    if logits.len().checked_mul(4) != Some(width) { return Err("incomplete gate logits".into()); }
    let bytes = logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    finite_row(&bytes, width)?;
    Ok(bytes)
}
pub fn finite_row(bytes: &[u8], width: usize) -> Result<()> {
    if width == 0 || !width.is_multiple_of(4) || bytes.len() != width
        || bytes.as_chunks::<4>().0.iter().any(|v| !f32::from_le_bytes(*v).is_finite()) {
        return Err("gate needs a finite, complete vocabulary FP32 row".into());
    }
    Ok(())
}
