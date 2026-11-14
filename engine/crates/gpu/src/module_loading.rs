//! Shared, safe module-load control flow. CUDA calls stay in the driver adapter.
#![forbid(unsafe_code)]
use std::ffi::CStr;

pub(crate) struct Image<'a> {
    pub bytes: &'a [u8],
    pub ptx: &'a [u8],
    pub minimum_driver: i32,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Loaded<H> { pub handle: H, pub receipt: Option<usize> }

pub(crate) struct Failure<H> { pub handle: Option<H>, pub error: String }

pub(crate) trait Driver {
    type Handle: Copy;
    fn device_version(&mut self) -> Result<((i32, i32), i32), String>;
    fn cubin(&mut self, image: &[u8]) -> (i32, Option<Self::Handle>);
    fn ptx(&mut self, ptx: &CStr, receipt: bool) -> Result<Loaded<Self::Handle>, Failure<Self::Handle>>;
    fn unload(&mut self, handle: Self::Handle) -> Result<(), String>;
    fn error(&self, code: i32) -> String;
    fn notice(&mut self, message: &str);
}

fn failed<D: Driver>(driver: &mut D, failure: Failure<D::Handle>) -> String {
    if let Some(handle) = failure.handle {
        if let Err(cleanup) = driver.unload(handle) {
            return format!("{}; {cleanup}", failure.error);
        }
    }
    failure.error
}

/// Every fallback uses the caller's original CStr, including its terminator.
/// A failed load relinquishes any partial handle before another load is tried.
pub(crate) fn load<D: Driver>(driver: &mut D, ptx: &CStr, image: Option<Image<'_>>,
    receipt: bool, force_ptx: bool) -> Result<Loaded<D::Handle>, String> {
    if let Some(image) = image.filter(|image| !image.bytes.is_empty() && !receipt && !force_ptx) {
        let (device, version) = driver.device_version()?;
        if crate::cubin_matches(ptx.to_bytes(), image.ptx, device, version, image.minimum_driver) {
            let (code, handle) = driver.cubin(image.bytes);
            if code == 0 {
                let handle = handle.ok_or("prebuilt cubin load returned no module")?;
                driver.notice("gpu_backend module=prebuilt_cubin target=sm_121");
                return Ok(Loaded { handle, receipt: None });
            }
            let error = driver.error(code);
            if let Some(handle) = handle {
                driver.unload(handle).map_err(|cleanup| format!("{error}; {cleanup}"))?;
            }
            // Only incompatible image codes may retry using PTX. Resource and
            // context failures must remain visible to the caller.
            if !matches!(code, 200 | 209 | 218 | 222) { return Err(error); }
            driver.notice(&format!("gpu_backend module=ptx reason=incompatible_prebuilt_cubin code={code}"));
        }
    }
    driver.ptx(ptx, receipt).map_err(|failure| failed(driver, failure))
}

#[cfg(test)]
#[path = "module_loading_tests.rs"]
mod tests;
