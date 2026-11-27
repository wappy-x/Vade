//! Typed launches on the backend's ordered stream; caller owns device storage.
use crate::{driver::Backend, qsa::contract::*};
use std::ffi::c_void;
fn arg<T>(p: &mut T) -> *mut c_void { (p as *mut T).cast() }
fn extent(rows: u32, past: u32) -> Result<(), String> {
    if rows == 0 || rows as usize > CHUNK || past as usize + rows as usize > LIMIT { return Err("QSA rows must be 1..=1024, context <=1M".into()); }
    Ok(())
}
impl Backend<'_> {
    /// # Safety
    /// Addresses obey Prepare's layouts, do not alias, and remain live until
    /// synchronization. Ring holds the preceding partial group; tables are FP32.
    pub unsafe fn qsa_prepare(&self, mut p: Prepare) -> Result<(), String> {
        extent(p.rows, p.past)?;
        if !p.epsilon.is_finite() || p.epsilon <= 0.0 { return Err("invalid QSA epsilon".into()); }
        let groups = (p.past + p.rows) / 4 - p.past / 4;
        self.launch_flash("qsa_fp32_prepare", (p.rows * 4 + groups).div_ceil(128), 128, &mut [arg(&mut p)])?;
        self.launch_flash("qsa_fp32_ring", 4, 128, &mut [arg(&mut p)])
    }
    /// # Safety
    /// Addresses obey Select's layouts, are disjoint, and remain live until sync.
    /// Cache contains all completed groups; queries are normalized and rotated.
    pub unsafe fn qsa_select(&self, mut p: Select) -> Result<(), String> {
        extent(p.rows, p.past)?;
        if p.stride < (p.past + p.rows) / 4 { return Err("QSA score stride too small".into()); }
        if p.stride > 0 { self.launch_flash_rows("qsa_fp32_scores", p.stride.div_ceil(128), p.rows, 128, 0, &mut [arg(&mut p)])?; }
        self.launch_flash("qsa_fp32_select", p.rows, 256, &mut [arg(&mut p)])
    }
    /// # Safety
    /// Addresses obey Attention's layouts and remain live until sync. Selection
    /// is the causal, ascending output of qsa_select; K/V include every position.
    pub unsafe fn qsa_attention(&self, mut p: Attention) -> Result<(), String> {
        extent(p.rows, 0)?;
        self.launch_flash("qsa_fp32_attention_scores", (p.rows * 24 * WIDTH as u32).div_ceil(128), 128, &mut [arg(&mut p)])?;
        self.launch_flash("qsa_fp32_softmax", (p.rows * 24).div_ceil(32), 32, &mut [arg(&mut p)])?;
        self.launch_flash("qsa_fp32_values", p.rows * 48, 128, &mut [arg(&mut p)])
    }
}
