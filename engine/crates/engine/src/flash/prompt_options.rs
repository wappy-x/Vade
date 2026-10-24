//! Default-ON prompt switches, validated once before model allocation.
//! Unset/"1" enables, "0" disables; malformed values fail startup. DEDUP requires
//! FRONTEND_GPU. COLD8K_R2=0 restores the preceding prompt kernel paths and scratch.
use std::sync::OnceLock;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub frontend_gpu: bool,
    pub frontend_dedup: bool,
    pub gemm_u32: bool,
    pub cold8k_r2: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self { frontend_gpu: true, frontend_dedup: true, gemm_u32: true, cold8k_r2: true }
    }
}
static OPTIONS: OnceLock<Result<Options, String>> = OnceLock::new();
impl Options {
    pub const NAMES: [&str; 4] = ["VADE_PROMPT_FRONTEND_GPU", "VADE_PROMPT_FRONTEND_DEDUP", "VADE_PROMPT_GEMM_U32", "VADE_PROMPT_COLD8K_R2"];
    pub fn configured_values(self) -> [bool; Self::NAMES.len()] {
        [self.frontend_gpu,self.frontend_dedup,self.gemm_u32,self.cold8k_r2]
    }
    pub fn parse(mut value: impl FnMut(&str) -> Result<Option<String>, String>) -> Result<Self, String> {
        let d = Self::default();
        let mut flags = [d.frontend_gpu, d.frontend_dedup, d.gemm_u32, d.cold8k_r2];
        for (flag, name) in flags.iter_mut().zip(Self::NAMES) {
            *flag = match value(name)?.as_deref() {
                None => *flag,
                Some("1") => true,
                Some("0") => false,
                _ => return Err(format!("{name} must be 0 or 1")),
            };
        }
        let [frontend_gpu, frontend_dedup, gemm_u32, cold8k_r2] = flags;
        if frontend_dedup && !frontend_gpu {
            return Err("VADE_PROMPT_FRONTEND_DEDUP=1 requires VADE_PROMPT_FRONTEND_GPU=1".into());
        }
        Ok(Self {frontend_gpu, frontend_dedup, gemm_u32, cold8k_r2})
    }
    /// Eligibility uses the complete matrix row count, including pooled prefill.
    /// Small matrices retain main; every kernel keeps its fixed operation order.
    pub fn kernel_rows(tokens: usize) -> bool { (2048..=32768).contains(&tokens) }
    /// Fixed segment counts plus one validated u32 expert ID per pair; align as DeviceVector.
    pub fn route_workspace_bytes(self, rows: usize) -> usize {
        if self.cold8k_r2 && Self::kernel_rows(rows) {(((rows*10).div_ceil(10240)*512+rows*10)*4).div_ceil(64)*64} else {0}
    }
    /// DeviceVector allocations are rounded to64bytes. Back ranks need no input expansion.
    pub fn packed_workspace_bytes(self, rows: usize, frontend: bool) -> usize {
        if self.frontend_gpu && frontend { (rows * 1508).div_ceil(64) * 64 } else { 0 }
    }
}
/// flash-serve calls this before inspecting the checkpoint; direct model loaders also
/// call it before any allocation. A malformed environment remains a cached startup error.
pub fn load() -> Result<Options, String> {
    OPTIONS.get_or_init(|| Options::parse(|name| match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(format!("{name} must be 0 or 1")),
    })).clone()
}
/// Execution performs no environment lookup. Model startup always calls `load`.
/// Standalone CPU readers that have not loaded a model retain the CPU path;
/// they have no initialized GPU frontend to consume packed inputs.
pub fn cached() -> Options {
    OPTIONS.get().and_then(|v| v.as_ref().ok()).copied().unwrap_or(Options {
        frontend_gpu: false, frontend_dedup: false, gemm_u32: false, cold8k_r2: false,
    })
}
#[cfg(test)]
mod tests {
    use super::Options;
    #[test]
    fn strict_prompt_switches_and_dependency() {
        assert_eq!(Options::parse(|_| Ok(None)).unwrap(), Options::default());
        assert_eq!(Options::parse(|_| Ok(Some("1".into()))).unwrap(), Options {
            frontend_gpu: true, frontend_dedup: true, gemm_u32: true, cold8k_r2: true,
        });
        assert_eq!(Options::parse(|_| Ok(Some("0".into()))).unwrap(), Options {
            frontend_gpu: false, frontend_dedup: false, gemm_u32: false, cold8k_r2: false,
        });
        // Every mix of unset, explicit OFF and explicit ON, including dependencies.
        for mask in 0..81 {
            let parsed = Options::parse(|name| {
                let i = Options::NAMES.iter().position(|v| *v == name).unwrap();
                Ok(match mask / 3usize.pow(i as u32) % 3 { 0 => None, 1 => Some("0".into()), _ => Some("1".into()) })
            });
            let enabled: [bool; 4] = std::array::from_fn(|i| match mask / 3usize.pow(i as u32) % 3 { 0 => true, 1 => false, _ => true });
            if enabled[1] && !enabled[0] { assert!(parsed.unwrap_err().contains("requires")); }
            else { let v = parsed.unwrap(); assert_eq!([v.frontend_gpu,v.frontend_dedup,v.gemm_u32,v.cold8k_r2], enabled); }
        }
        for bad in Options::NAMES {
            for value in ["", "yes", "true", "2", " 1", "1 "] {
                assert!(Options::parse(|name| Ok(Some(if name == bad {value} else {"0"}.into()))).unwrap_err().contains(bad));
            }
            assert!(Options::parse(|name| if name == bad {Err(format!("{name} must be 0 or 1"))} else {Ok(None)}).unwrap_err().contains(bad));
        }
    }
    #[test]
    fn route_scratch_matches_fixed_segments_and_only_enabled_prompt_workspaces() {
        let on=Options{cold8k_r2:true,..Options::default()};
        for(rows,bytes)in [(1,0),(63,0),(64,0),(511,0),(512,0),(1024,0),(1025,0),(2047,0),(2048,86016),(8192,344064),(8193,346176),(32768,1376256),(32769,0)]{
            assert_eq!(on.route_workspace_bytes(rows),bytes);
            assert_eq!(Options{cold8k_r2:false,..Options::default()}.route_workspace_bytes(rows),0);
        }
    }
    #[test]
    fn packed_workspace_is_funded_only_for_enabled_frontends() {
        let on = Options::default();
        let off = Options::parse(|_| Ok(Some("0".into()))).unwrap();
        for (rows, bytes) in [(1,1536),(65,98048),(8192,12353536)] {
            assert_eq!(on.packed_workspace_bytes(rows,true),bytes);
            assert_eq!(on.packed_workspace_bytes(rows,false),0);
            assert_eq!(off.packed_workspace_bytes(rows,true),0);
        }
    }
}
