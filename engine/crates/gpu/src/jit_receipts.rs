//! Opt-in driver observations. No compiler target, register cap or optimization overrides.
use super::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

struct Kernel {
    name: String,
    module: usize,
    // NUM_REGS, LOCAL_SIZE_BYTES, SHARED_SIZE_BYTES, MAX_THREADS, PTX, BINARY.
    attributes: [c_int; 6],
    launches: BTreeMap<(u32, u32), c_int>,
}
pub(super) struct Receipts {
    file: File,
    kernels: Vec<Kernel>,
    handles: BTreeMap<usize, usize>,
    logs: Vec<Vec<String>>,
}
impl Receipts {
    pub(super) fn configured() -> Result<Option<Box<RefCell<Self>>>, String> {
        let Some(path) = std::env::var_os("VADE_JIT_RECEIPTS") else { return Ok(None); };
        let file = File::create(&path).map_err(|e| format!("VADE_JIT_RECEIPTS {path:?}: {e}"))?;
        let mut receipts = Self { file, kernels: Vec::new(), handles: BTreeMap::new(), logs: Vec::new() };
        receipts.write()?;
        Ok(Some(Box::new(RefCell::new(receipts))))
    }
    #[allow(clippy::manual_dangling_ptr)] // the FFI value 1 is not the type alignment.
    pub(super) fn load(&mut self, raw: &mut Handle, ptx: &CStr) -> Result<usize, String> {
        let mut info = vec![0u8; 4 * 1024 * 1024];
        let mut error = vec![0u8; 4 * 1024 * 1024];
        // CU_JIT_{INFO_LOG_BUFFER, INFO_LOG_BUFFER_SIZE_BYTES,
        // ERROR_LOG_BUFFER, ERROR_LOG_BUFFER_SIZE_BYTES, LOG_VERBOSE}.
        // Scalar option values are encoded directly as void*, not pointers to scalars.
        let mut options = [3, 4, 5, 6, 12];
        let mut values = [info.as_mut_ptr().cast(), info.len() as *mut c_void,
            error.as_mut_ptr().cast(), error.len() as *mut c_void, 1usize as *mut c_void];
        let result = unsafe { cuModuleLoadDataEx(raw, ptx.as_ptr().cast(), options.len() as u32,
            options.as_mut_ptr(), values.as_mut_ptr()) };
        let module = self.logs.len();
        let mut lines = Vec::new();
        for (kind, bytes) in [("info", info), ("error", error)] {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            lines.extend(String::from_utf8_lossy(&bytes[..end]).lines().map(|s| format!("{kind}: {s}")));
            if end >= bytes.len() - 1 { lines.push(format!("{kind}: BUFFER MAY BE TRUNCATED")); }
        }
        if lines.is_empty() { lines.push("info/error: empty (driver cache may suppress compiler logs)".into()); }
        self.logs.push(lines);
        self.write()?;
        if let Err(e) = check(result, "cuModuleLoadDataEx") { return Err(format!("{e}; JIT logs in VADE_JIT_RECEIPTS")); }
        Ok(module)
    }
    pub(super) fn resolved(&mut self, raw: Handle, name: &str, module: usize) -> Result<(), String> {
        if let Some(&i) = self.handles.get(&(raw as usize)) {
            if self.kernels[i].module == module { return Ok(()); }
        }
        let mut attributes = [0; 6];
        for (value, attribute) in attributes.iter_mut().zip([4, 3, 1, 0, 5, 6]) {
            unsafe { check(cuFuncGetAttribute(value, attribute, raw), "cuFuncGetAttribute")
                .map_err(|e| format!("{name}: {e}"))?; }
        }
        self.handles.insert(raw as usize, self.kernels.len());
        self.kernels.push(Kernel { name: name.into(), module, attributes, launches: BTreeMap::new() });
        self.write()
    }
    pub(super) fn launched(&mut self, raw: Handle, threads: u32, shared: u32) -> Result<(), String> {
        let i = *self.handles.get(&(raw as usize)).ok_or("JIT receipt missing resolved kernel")?;
        if self.kernels[i].launches.contains_key(&(threads, shared)) { return Ok(()); }
        let mut blocks = 0;
        unsafe { check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut blocks, raw,
            c_int::try_from(threads).map_err(|_| "block size exceeds driver ABI")?, shared as usize),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor")?; }
        self.kernels[i].launches.insert((threads, shared), blocks);
        self.write()
    }
    fn write(&mut self) -> Result<(), String> {
        let mut text = String::from("# Driver maximum residency, not achieved occupancy. '-' means resolved but never launched.\n");
        text.push_str("kernel\tregs\tlocal B\tsmem B\tblock threads\tdynamic smem B\tblocks per SM\tmax threads\tPTX version\tbinary version\tJIT log lines\n");
        for k in &self.kernels {
            let [regs, local, smem, max, ptx, binary] = k.attributes;
            let prefix = format!("{}\t{regs}\t{local}\t{smem}", k.name);
            let suffix = format!("{max}\t{ptx}\t{binary}\tmodule {} lines 1..{}", k.module, self.logs[k.module].len());
            if k.launches.is_empty() { let _ = writeln!(text, "{prefix}\t-\t-\t-\t{suffix}"); }
            for (&(threads, shared), blocks) in &k.launches {
                let _ = writeln!(text, "{prefix}\t{threads}\t{shared}\t{blocks}\t{suffix}");
            }
        }
        for (module, lines) in self.logs.iter().enumerate() {
            for (i, line) in lines.iter().enumerate() { let _ = writeln!(text, "# JIT {module}:{} {line}", i + 1); }
        }
        self.file.seek(SeekFrom::Start(0)).and_then(|_| self.file.write_all(text.as_bytes()))
            .and_then(|_| self.file.set_len(text.len() as u64))
            .map_err(|e| format!("VADE_JIT_RECEIPTS write: {e}"))
    }
}
