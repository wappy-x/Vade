//! Kernel kit, part 4, capture side. DIAGNOSTIC ONLY: it changes no arithmetic and nothing unless a check tool turns
//! it on from its own command line (`nv_capture_configure`, the option `--nv-capture=` of `resident-decode-check`).
//! No served path and no environment variable turns it on. It saves, for NVFP4 expert launches number SKIP,
//! SKIP + STRIDE, ... (CALLS of them) of this process, everything the launch read and what it wrote: the part table,
//! the quantized activation rows, every part's weight and scale bytes as they sit on the device, and the output words.
//! One folder a call. `resident-decode-check nv-replay` runs a folder again on one box. A capture reads device memory
//! back (it waits for the stream), so a capture run is never a timing run.
use super::*;
use crate::expert_ops::Shape;
use std::sync::Mutex;

struct State {dir:std::path::PathBuf,skip:usize,stride:usize,calls:usize,seen:usize}
static STATE:Mutex<Option<State>>=Mutex::new(None);
/// Turn the capture on for this process: launches `skip`, `skip + stride`, ... (`calls` of them) are saved under `dir`.
pub fn nv_capture_configure(dir:std::path::PathBuf,skip:usize,stride:usize,calls:usize)->Result<(),String> {
    if stride==0 || calls==0 {return Err("NV capture needs a stride and a call count above zero".into());}
    *STATE.lock().map_err(|_|"NV capture state is poisoned")?=Some(State {dir,skip,stride,calls,seen:0});Ok(())
}
fn raw(values:&[f32])->Vec<u8> {values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect()}
/// One part of a captured call: its shape, first input row, and owners.
pub(super) type Part<'p,'v>=(Shape,&'p DeviceVector<'v>,&'p DeviceVector<'v>,usize);

impl<'a> Backend<'a> {
    /// True when the next NVFP4 launch is one to save (and counts it).
    /// True while a check tool has a capture configured. It does not count a call (`nv_capture_wanted` does).
    pub(super) fn nv_capture_active(&self)->bool {STATE.lock().ok().is_some_and(|guard|guard.is_some())}
    pub(super) fn nv_capture_wanted(&self)->Option<std::path::PathBuf> {
        let mut guard=STATE.lock().ok()?;let s=guard.as_mut()?;
        let at=s.seen;s.seen+=1;
        if at<s.skip || (at-s.skip)%s.stride!=0 || (at-s.skip)/s.stride>=s.calls {return None;}
        Some(s.dir.join(format!("call-{at:05}")))
    }
    /// Saves one call. `kind` is "group" (`expert_group`: `input` holds every part's rows, `stride` words a row) or
    /// "batch" (`expert_projection_batch`: `inputs` holds one row a part).
    pub(super) fn nv_capture_save(&self,dir:&std::path::Path,kind:&str,parts:&[Part<'_,'_>],inputs:&[&DeviceVector<'_>],stride:usize,output:&DeviceVector<'_>)->Result<(),String> {
        let io=|e:std::io::Error|format!("NV capture: {e}");
        std::fs::create_dir_all(dir).map_err(io)?;
        let mut meta=format!("format=vade-nv-capture-v1\nkind={kind}\nstride={stride}\nparts={}\ninputs={}\ntile_rows={}\n",parts.len(),inputs.len(),crate::expert_ops::NV_TILE_ROWS);
        for (i,(s,weight,scales,first_row)) in parts.iter().enumerate() {
            meta+=&format!("part {i} op={} rows={} cols={} block_rows={} block_cols={} tokens={} global={:08x} activation={:08x} bound={:08x} role={} first_row={first_row}\n",
                s.op as u32,s.rows,s.cols,s.block_rows,s.block_cols,s.tokens,s.global.to_bits(),s.activation.to_bits(),s.bound.to_bits(),s.role);
            std::fs::write(dir.join(format!("w-{i}.bin")),raw(&self.read_vector_values(weight)?)).map_err(io)?;
            std::fs::write(dir.join(format!("s-{i}.bin")),raw(&self.read_vector_values(scales)?)).map_err(io)?;
        }
        for (i,input) in inputs.iter().enumerate() {std::fs::write(dir.join(format!("input-{i}.bin")),raw(&self.read_vector_values(input)?)).map_err(io)?;}
        std::fs::write(dir.join("output.bin"),raw(&self.read_vector_values(output)?)).map_err(io)?;
        std::fs::write(dir.join("META"),meta).map_err(io)
    }
}
