//! Read-only PLE mappings; one-step WILLNEED hints, no table device copy.
use super::gdn::{self, Arg::Ptr};
use super::{EMBED, NextModel, Result, err};
use crate::flash_next_input as input;
use engine_model::flash::{
    Checkpoint, PLE_PART_ROWS, PLE_ROWS,
    mixed::{Plan, ScaleView},
};
use spark_gpu_probe::driver::{Backend, DeviceVector, ExchangeSlot, FrontendStream};
use std::{collections::BTreeMap, ffi::c_void, fs::File, os::fd::AsRawFd, path::Path, sync::Arc};
unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn sysconf(name: i32) -> i64;
}
struct Mapping {
    address: usize,
    len: usize,
}
impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            munmap(self.address as _, self.len);
        }
    }
}
struct Span {
    map: Arc<Mapping>,
    start: u64,
}
pub(in crate::flash) struct Tables {
    spans: Vec<[Span; 2]>,
    page: usize,
}
impl Tables {
    pub fn bases(&self) -> Vec<u64> {
        self.spans
            .iter()
            .flat_map(|p| p.iter().map(|s| s.map.address as u64 + s.start))
            .collect()
    }
    pub fn load(b: &Backend<'_>, cp: &Checkpoint, plan: &Plan, root: &Path) -> Result<Self> {
        if !b.pageable_host_page_tables()? {
            return Err("FP4 PLE requires pageable host page tables (attribute 100=1)".into());
        }
        let root = root.canonicalize().map_err(err)?;
        let mut files: BTreeMap<String, Arc<Mapping>> = BTreeMap::new();
        let mut spans = Vec::new();
        for shard in 0..PLE_ROWS / PLE_PART_ROWS {
            let row = plan.ple_row(shard as u32, 0).map_err(err)?;
            let ScaleView::E4m3(scale) = &row.matrix.scales else {
                return Err("PLE requires direct scales".into());
            };
            let mut pair = Vec::new();
            for name in [&row.matrix.weight.name, &scale.name] {
                let filename = &cp.index.weight_map[name];
                let header = &cp.shards[filename];
                if !files.contains_key(filename) {
                    let path = root.join(filename).canonicalize().map_err(err)?;
                    if !path.starts_with(&root) {
                        return Err("PLE shard escapes checkpoint".into());
                    }
                    let file = File::open(path).map_err(err)?;
                    let len = file.metadata().map_err(err)?.len() as usize;
                    if len as u64 != header.file_bytes || len == 0 {
                        return Err("PLE shard changed after inspection".into());
                    }
                    let address =
                        unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) }
                            as usize;
                    if address == usize::MAX {
                        return Err(std::io::Error::last_os_error().to_string());
                    }
                    files.insert(filename.clone(), Arc::new(Mapping { address, len }));
                }
                pair.push(Span {
                    map: files[filename].clone(),
                    start: header.data_start + cp.tensor(name).map_err(err)?.start,
                });
            }
            spans.push(pair.try_into().ok().unwrap());
        }
        let page = unsafe { sysconf(30) } as usize;
        if page < 4096 || !page.is_power_of_two() {
            return Err("invalid PLE page size".into());
        }
        Ok(Self { spans, page })
    }
    fn advise(&self, rows: &[u64]) {
        for &row in rows {
            for (p, stride) in self.spans[(row / PLE_PART_ROWS) as usize]
                .iter()
                .zip([80, 10])
            {
                let at = p.start as usize + (row % PLE_PART_ROWS) as usize * stride;
                let first = at / self.page * self.page;
                unsafe {
                    madvise(
                        (p.map.address + first) as _,
                        (at + stride - first).div_ceil(self.page) * self.page,
                        3,
                    );
                }
            }
        }
    }
}
pub(super) struct Feed<'a> {
    side: FrontendStream<'a>,
    pub ids: DeviceVector<'a>,
    hist: DeviceVector<'a>,
    params: DeviceVector<'a>,
    rows: DeviceVector<'a>,
    bases: DeviceVector<'a>,
    output: ExchangeSlot<'a>,
    activation: ExchangeSlot<'a>,
}
fn bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
impl<'a> Feed<'a> {
    pub fn new(b: &Backend<'a>, m: &NextModel<'a>) -> Result<Self> {
        let mut params = b.vector(70)?;
        let (mult, sizes, offsets, _) = &m.hash;
        b.write_vector_bytes(
            &mut params,
            &bytes(&[mult.as_slice(), sizes, offsets].concat()),
        )?;
        let bases = m
            .tables
            .as_ref()
            .ok_or("frontend tables not loaded")?
            .bases();
        let mut table = b.vector(bases.len() * 2)?;
        b.write_vector_bytes(&mut table, bytes(&bases))?;
        let mut feed = Self {
            side: b.frontend_stream()?,
            ids: input::ids(b)?,
            hist: b.vector(4)?,
            params,
            rows: b.vector(32)?,
            bases: table,
            output: b.exchange_slot(input::IDS_WORDS)?,
            activation: b.exchange_slot(2560)?,
        };
        feed.reset(b)?;
        Ok(feed)
    }
    pub fn reset(&mut self, b: &Backend<'_>) -> Result<()> {
        b.synchronize()?;
        b.write_vector_bytes(&mut self.hist, bytes(&[248044i64; 2]))
    }
    pub fn begin(
        &mut self,
        b: &Backend<'a>,
        m: &NextModel<'a>,
        x: &DeviceVector<'a>,
        token: Option<u32>,
    ) -> Result<()> {
        input::begin(b, &mut self.ids, token)?;
        gdn::launch(
            b,
            "flash_next_feed",
            10240,
            &mut [
                Ptr(self.ids.device_address()),
                Ptr(self.hist.device_address()),
                Ptr(self.params.device_address()),
                Ptr(self.rows.device_address()),
                Ptr(m.weights.device_address(EMBED)?),
                Ptr(x.device_address()),
            ],
        )
    }
    pub fn gather(
        &mut self,
        b: &Backend<'a>,
        m: &NextModel<'a>,
        lexical: &DeviceVector<'a>,
    ) -> Result<()> {
        let (mut rows, mut bases, mut out, mut global) = (
            self.rows.device_address(),
            self.bases.device_address(),
            lexical.device_address(),
            m.ple_global,
        );
        unsafe {
            b.frontend_launch(
                &self.side,
                "flash_next_ple",
                &mut [
                    (&mut rows as *mut u64).cast(),
                    (&mut bases as *mut u64).cast(),
                    (&mut out as *mut u64).cast(),
                    (&mut global as *mut f32).cast(),
                ],
            )
        }
    }
    pub fn join(&self, b: &Backend<'a>) -> Result<()> {
        b.frontend_join(&self.side)
    }
    // A context-wide layer-0 read would retire the PLE side stream too early.
    pub fn activation(&mut self, b: &Backend<'a>, v: &DeviceVector<'_>) -> Result<Vec<f32>> {
        b.exchange_begin(&mut self.activation, v)?;
        Ok(b.exchange_wait(&mut self.activation)?.to_vec())
    }
    pub fn publish(&mut self, b: &Backend<'a>) -> Result<()> {
        gdn::launch(
            b,
            "flash_next_predict",
            32,
            &mut [
                Ptr(self.ids.device_address()),
                Ptr(self.hist.device_address()),
                Ptr(self.params.device_address()),
            ],
        )?;
        b.exchange_begin(&mut self.output, &self.ids)
    }
    pub fn read(&mut self, b: &Backend<'a>, m: &NextModel<'a>) -> Result<u32> {
        let v = b.exchange_wait(&mut self.output)?;
        if v[2] != 1.0 || !v[..3].iter().all(|x| x.is_finite()) {
            return Err("nonfinite logits".into());
        }
        let rows: [u64; 16] = std::array::from_fn(|h| {
            v[4 + 2 * h].to_bits() as u64 | ((v[5 + 2 * h].to_bits() as u64) << 32)
        });
        m.tables
            .as_ref()
            .ok_or("frontend tables not loaded")?
            .advise(&rows);
        Ok(v[1] as u32)
    }
}
