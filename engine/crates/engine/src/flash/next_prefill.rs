//! Resolve immutable checkpoint row descriptors once, outside prompt submission.
//! Retained files avoid per-row plan scans, canonicalize/open/stat and allocation.
use engine_model::flash::{
    Checkpoint, PLE_PART_ROWS, PLE_ROWS,
    mixed::{Plan, ScaleView},
    nvfp4, refcpu,
};
use crate::flash_next_read_plan::{self as read_plan, Request};
use std::{collections::BTreeMap, fs::File, os::unix::fs::FileExt, path::Path, sync::Arc};
type Result<T> = std::result::Result<T, String>;
const H: usize = 2560;
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const EMBED: &str = "model.language_model.embed_tokens.weight";
/// Staged bytes per prompt row: 16 PLE rows (80 B nibbles + 10 B scales), then
/// the BF16 embedding row.
const STAGED: usize = 16 * 90 + H * 2;
/// Experimental prompt input expansion on the GPU; default remains the CPU path.
pub(crate) fn gpu_prompt() -> bool {
    crate::flash_prompt_options::cached().frontend_gpu
}
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn gpu_prompt_dedup() -> bool {
    crate::flash_prompt_options::cached().frontend_dedup
}
/// Read-ahead hint budget per call in requested bytes (VADE_PLE_READ_AHEAD_BYTES,
/// default 64 MiB: three 2048-token chunks of PLE and embedding rows).
fn read_ahead_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("VADE_PLE_READ_AHEAD_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 << 20)
    })
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
#[derive(Clone)]
struct Span {
    file: Arc<File>,
    /// Index of `file` in `Reader::files` (read-plan file id).
    id: u32,
    start: u64,
    len: u64,
}
impl Span {
    fn read(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err("Flash prefill read exceeds tensor".into());
        }
        self.file
            .read_exact_at(output, self.start + offset)
            .map_err(err)
    }
    /// POSIX_FADV_WILLNEED: the kernel starts reading the pages now and returns;
    /// a later read finds them cached or in flight. A hint: bytes are unchanged.
    fn advise(&self, offset: u64, len: u64) {
        if offset.checked_add(len).is_some_and(|end| end <= self.len) {
            read_plan::advise(&self.file, self.start + offset, len);
        }
    }
    /// The same bounds as `read`, as a read-plan request.
    fn request(&self, offset: u64, len: usize, dest: usize) -> Result<Request> {
        if offset
            .checked_add(len as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err("Flash prefill read exceeds tensor".into());
        }
        Ok(Request { file: self.id, at: self.start + offset, len: len as u32, dest })
    }
}
/// One prompt tile's frontend output computed ahead on the read-ahead thread,
/// keyed by the exact (token, PLE rows) list and global scale it was made from.
struct Prestaged {
    key: Vec<(u32, Vec<u64>)>,
    global: u32,
    residual: Vec<f32>,
    lexical: Vec<f32>,
}
#[derive(Clone)]
pub(crate) struct Reader {
    ple: Vec<[Span; 2]>,
    embed: Span,
    files: Vec<Arc<File>>,
    prestaged: Arc<std::sync::Mutex<Option<Prestaged>>>,
    /// Launch-thread prompt frontends finished so far; read-ahead work waits for
    /// the tile it was queued before, so it never competes with that tile's reads.
    finished: Arc<(std::sync::Mutex<u64>, std::sync::Condvar)>,
}
impl Reader {
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn load(cp: &Checkpoint, plan: &Plan, root: &Path) -> Result<Self> {
        let root = root.canonicalize().map_err(err)?;
        let mut files = BTreeMap::<String, (u32, Arc<File>)>::new();
        let mut list = Vec::new();
        let mut span = |name: &str| -> Result<Span> {
            let tensor = cp.tensor(name).map_err(err)?;
            let shard = &cp.index.weight_map[name];
            let header = &cp.shards[shard];
            if !files.contains_key(shard) {
                let path = root.join(shard).canonicalize().map_err(err)?;
                if !path.starts_with(&root) {
                    return Err("shard escapes checkpoint".into());
                }
                let file = File::open(path).map_err(err)?;
                if file.metadata().map_err(err)?.len() != header.file_bytes {
                    return Err("shard changed after inspection".into());
                }
                let file = Arc::new(file);
                files.insert(shard.clone(), (list.len() as u32, file.clone()));
                list.push(file);
            }
            let (id, file) = files[shard].clone();
            Ok(Span {
                file,
                id,
                start: header.data_start + tensor.start,
                len: tensor.byte_len(),
            })
        };
        let embed = span(EMBED)?;
        let mut ple = Vec::new();
        for shard in 0..PLE_ROWS / PLE_PART_ROWS {
            let row = plan.ple_row(shard as u32, 0).map_err(err)?;
            let ScaleView::E4m3(scale) = &row.matrix.scales else {
                return Err("PLE requires direct scales".into());
            };
            if row.packed_bytes != 80 || row.scale_bytes != 10 {
                return Err("unexpected PLE row shape".into());
            }
            ple.push([span(&row.matrix.weight.name)?, span(&scale.name)?]);
        }
        Ok(Self { ple, embed, files: list, prestaged: Default::default(), finished: Default::default() })
    }
    /// Start the page reads of every listed row before any pread waits: hashed
    /// n-gram rows are random in a 28.8 GB table the page cache cannot hold, and
    /// serial cold preads cost one NVMe round trip each (C1 decode: ~28 per step,
    /// 5.7 ms). With the hints the set costs about one round trip.
    pub(crate) fn prefetch_ple_rows(&self, rows: &[u64]) {
        for &row in rows {
            if row >= PLE_ROWS {
                continue;
            }
            let pair = &self.ple[(row / PLE_PART_ROWS) as usize];
            let local = row % PLE_PART_ROWS;
            pair[0].advise(local * 80, 80);
            pair[1].advise(local * 10, 10);
        }
    }
    /// MTP drafts: page-cache hints for rows a later frontend reads (each
    /// row's 16 PLE rows and its embedding row, as `frontend_rows` requests
    /// them). Never blocks on I/O; a hint: bytes are unchanged.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn hint_rows(&self, rows: &[(u32, Vec<u64>)]) {
        let mut requests = Vec::with_capacity(rows.len() * 33);
        for (i, (token, ple)) in rows.iter().enumerate() {
            if self.requests(i, *token, ple, &mut requests).is_err() {
                return;
            }
        }
        read_plan::Plan::new(requests).hint(&self.files, usize::MAX);
    }
    /// Background read-ahead of the prompt continuing at `hash`'s position
    /// (`tokens[0]` is the tile about to run on the launch thread). A
    /// long-lived thread hashes a copy of the n-gram state, computes the
    /// frontend of the predicted next tile `next` (the same reads and the same
    /// math as `frontend_rows`) into the prestage slot, then hints the rows after
    /// it (nearest first, up to the byte budget). The caller only enqueues; the
    /// launch thread uses a prestaged tile only when its (token, rows) list and
    /// global scale are identical, so a wrong prediction changes nothing.
    pub(crate) fn read_ahead(
        &self,
        mut hash: refcpu::PleHash,
        tokens: Vec<u32>,
        next: std::ops::Range<usize>,
        global: f32,
    ) {
        if tokens.is_empty() {
            return;
        }
        let reader = self.clone();
        let queued = self.finished.0.lock().map_or(0, |n| *n);
        read_plan::background(Box::new(move |latest| {
            // Start once the launch thread's current tile has its rows (bounded wait).
            let (count, done) = &*reader.finished;
            if let Ok(n) = count.lock() {
                let _ = done.wait_timeout_while(n, std::time::Duration::from_secs(10), |n| *n <= queued);
            }
            reader.read_ahead_job(&mut hash, &tokens, next, global, latest)
        }));
    }
    fn read_ahead_job(
        &self,
        hash: &mut refcpu::PleHash,
        tokens: &[u32],
        next: std::ops::Range<usize>,
        global: f32,
        latest: bool,
    ) {
        let rows: Vec<(u32, Vec<u64>)> = tokens.iter().map(|&t| (t, hash.advance(t))).collect();
        let next = next.start.min(rows.len())..next.end.min(rows.len());
        // A newer job means the launch thread has moved on: hints only.
        if latest && !next.is_empty() && !gpu_prompt() {
            let key = rows[next.clone()].to_vec();
            let n = key.len();
            let (mut residual, mut lexical) = (vec![0f32; n * H * 4], vec![0f32; n * H]);
            if self
                .frontend_rows_with(&key, global, &mut residual, &mut lexical, read_plan::readers())
                .is_ok()
                && let Ok(mut slot) = self.prestaged.lock()
            {
                *slot = Some(Prestaged { key, global: global.to_bits(), residual, lexical });
            }
        }
        let budget = read_ahead_budget();
        let mut requests = Vec::new();
        for (i, (token, ple)) in rows.iter().enumerate().skip(next.end) {
            if (i - next.end) * STAGED >= budget || self.requests(i, *token, ple, &mut requests).is_err() {
                break;
            }
        }
        read_plan::Plan::new(requests).hint(&self.files, usize::MAX);
    }
    /// Official packed nibbles followed by direct E4M3 scales, in head order.
    /// The same bounded preads serve prompt dequantization and decode staging.
    pub(crate) fn ple_row(&self, row: u64, output: &mut [u8]) -> Result<()> {
        if row >= PLE_ROWS || output.len() != 90 {
            return Err("unexpected PLE row or staging shape".into());
        }
        let pair = &self.ple[(row / PLE_PART_ROWS) as usize];
        let local = row % PLE_PART_ROWS;
        pair[0].read(local * 80, &mut output[..80])?;
        pair[1].read(local * 10, &mut output[80..])
    }
    /// The per-row reference (one pread per row): the test oracle for the plan.
    #[cfg(test)]
    pub(crate) fn token(
        &self,
        token: u32,
        rows: &[u64],
        global: f32,
        residual: &mut [f32],
        lexical: &mut [f32],
    ) -> Result<()> {
        if rows.len() != 16 || residual.len() != H * 4 || lexical.len() != H {
            return Err("unexpected prefill frontend shape".into());
        }
        let (mut staged, mut raw) = ([0u8; 90], [0u8; H * 2]);
        for (h, &row) in rows.iter().enumerate() {
            self.ple_row(row, &mut staged)?;
            refcpu::ple_row_nvfp4(
                &staged[..80],
                &staged[80..],
                global,
                &mut lexical[h * 160..(h + 1) * 160],
            )
            .map_err(err)?;
        }
        self.embed
            .read(u64::from(token) * (H * 2) as u64, &mut raw)?;
        for (o, bytes) in residual[..H].iter_mut().zip(raw.chunks_exact(2)) {
            *o = nvfp4::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        for stream in 1..4 {
            residual.copy_within(0..H, stream * H);
        }
        Ok(())
    }
    /// The reads of one prompt row as read-plan requests into `STAGED` bytes
    /// at row `i`: per head the 80 B nibbles then the 10 B scales (`ple_row`
    /// layout), then the embedding row (`token`'s read).
    fn requests(&self, i: usize, token: u32, rows: &[u64], out: &mut Vec<Request>) -> Result<()> {
        if rows.len() != 16 {
            return Err("unexpected prefill frontend shape".into());
        }
        let base = i * STAGED;
        for (h, &row) in rows.iter().enumerate() {
            if row >= PLE_ROWS {
                return Err("unexpected PLE row or staging shape".into());
            }
            let pair = &self.ple[(row / PLE_PART_ROWS) as usize];
            let local = row % PLE_PART_ROWS;
            out.push(pair[0].request(local * 80, 80, base + h * 90)?);
            out.push(pair[1].request(local * 10, 10, base + h * 90 + 80)?);
        }
        out.push(self.embed.request(u64::from(token) * (H * 2) as u64, H * 2, base + 16 * 90)?);
        Ok(())
    }
    /// `token`'s math on staged bytes: dequantize the 16 PLE rows in head order,
    /// widen the BF16 embedding and copy it to the other three streams.
    fn decode(staged: &[u8], global: f32, residual: &mut [f32], lexical: &mut [f32]) -> Result<()> {
        for h in 0..16 {
            let row = &staged[h * 90..(h + 1) * 90];
            refcpu::ple_row_nvfp4(&row[..80], &row[80..], global, &mut lexical[h * 160..(h + 1) * 160])
                .map_err(err)?;
        }
        let raw = &staged[16 * 90..STAGED];
        for (o, bytes) in residual[..H].iter_mut().zip(raw.chunks_exact(2)) {
            *o = nvfp4::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        for stream in 1..4 {
            residual.copy_within(0..H, stream * H);
        }
        Ok(())
    }
    /// Read compact PLE rows; embeddings are already resident in the model.
    /// Preserve the original offset-sorted read plan and per-row hash order.
    pub(crate) fn frontend_packed(&self, rows:&[(u32,Vec<u64>)]) -> Result<Vec<u8>> {
        let n=rows.len();
        let mut staged=vec![0u8;n*1444];
        let mut requests=Vec::with_capacity(n*32);
        for (i,(token,hashes)) in rows.iter().enumerate() {
            if hashes.len()!=16 {return Err("unexpected packed frontend hashes".into());}
            self.embed.request(u64::from(*token)*(H*2) as u64,H*2,0)?; // bounds only
            staged[i*4..i*4+4].copy_from_slice(&token.to_le_bytes());
            for (h,&row) in hashes.iter().enumerate() {
                if row>=PLE_ROWS {return Err("unexpected packed frontend PLE row".into());}
                let pair=&self.ple[(row/PLE_PART_ROWS) as usize];
                let local=row%PLE_PART_ROWS;
                let at=n*4+i*1440+h*90;
                requests.push(pair[0].request(local*80,80,at)?);
                requests.push(pair[1].request(local*10,10,at+80)?);
            }
        }
        read_plan::Plan::new(requests).read(&self.files,read_plan::readers(),&mut staged).map_err(err)?;
        for row in staged[n*4..].chunks_exact(90) {
            if row[80..].iter().any(|&s|s>=127) {return Err("nonnegative finite block scales".into());}
        }
        let (count,done)=&*self.finished;
        if let Ok(mut n)=count.lock() {*n+=1;done.notify_all();}
        Ok(staged)
    }
    /// Index each distinct official PLE row once, in first-use order. This
    /// removes repeated read-plan requests; arithmetic still runs per output row.
    pub(crate) fn frontend_packed_dedup(&self, rows:&[(u32,Vec<u64>)]) -> Result<Vec<u8>> {
        let n=rows.len();
        let mut staged=vec![0u8;n*68];
        let mut unique=Vec::new();
        let mut indices=BTreeMap::new();
        for (i,(token,hashes)) in rows.iter().enumerate() {
            if hashes.len()!=16 {return Err("unexpected packed frontend hashes".into());}
            self.embed.request(u64::from(*token)*(H*2) as u64,H*2,0)?;
            staged[i*4..i*4+4].copy_from_slice(&token.to_le_bytes());
            for (h,&row) in hashes.iter().enumerate() {
                if row>=PLE_ROWS {return Err("unexpected packed frontend PLE row".into());}
                let index=match indices.entry(row) {
                    std::collections::btree_map::Entry::Occupied(e)=>*e.get(),
                    std::collections::btree_map::Entry::Vacant(e)=>{
                        let index=u32::try_from(unique.len()).map_err(err)?;
                        unique.push(row);e.insert(index);index
                    }
                };
                let at=n*4+(i*16+h)*4;
                staged[at..at+4].copy_from_slice(&index.to_le_bytes());
            }
        }
        staged.resize(n*68+unique.len()*90,0);
        let mut requests=Vec::with_capacity(unique.len()*2);
        for (i,&row) in unique.iter().enumerate() {
            let pair=&self.ple[(row/PLE_PART_ROWS) as usize];
            let local=row%PLE_PART_ROWS;
            requests.push(pair[0].request(local*80,80,n*68+i*90)?);
            requests.push(pair[1].request(local*10,10,n*68+i*90+80)?);
        }
        read_plan::Plan::new(requests).read(&self.files,read_plan::readers(),&mut staged).map_err(err)?;
        for row in staged[n*68..].chunks_exact(90) {
            if row[80..].iter().any(|&s|s>=127) {return Err("nonnegative finite block scales".into());}
        }
        let (count,done)=&*self.finished;
        if let Ok(mut n)=count.lock() {*n+=1;done.notify_all();}
        Ok(staged)
    }
    /// Prompt-tile frontend. All rows' reads go through one offset-sorted,
    /// merged read plan (VADE_PLE_READERS readers, each hinting its extents
    /// before its preads), landing every row's bytes in its own staging slot;
    /// then each row runs `token`'s math on exactly the bytes its own preads
    /// would return, in contiguous row blocks on scoped threads. Rows write only
    /// their own output rows, so the bytes do not change.
    pub(crate) fn frontend_rows(
        &self,
        rows: &[(u32, Vec<u64>)],
        global: f32,
        residual: &mut [f32],
        lexical: &mut [f32],
    ) -> Result<()> {
        self.frontend_rows_with(rows, global, residual, lexical, read_plan::readers())
    }
    /// The launch thread's prompt frontend into the workspace staging: a tile
    /// prestaged from the identical (token, rows) list and global scale is
    /// taken (swapped in when the staging has its length, else copied into the
    /// leading rows); otherwise `frontend_rows` runs here. Rows past the tile
    /// are never uploaded or read.
    pub(crate) fn frontend_prompt(
        &self,
        rows: &[(u32, Vec<u64>)],
        global: f32,
        prompt: &mut (Vec<f32>, Vec<f32>),
    ) -> Result<()> {
        let n = rows.len();
        let ready = self.prestaged.lock().ok().and_then(|mut slot| {
            slot.take_if(|p| p.global == global.to_bits() && p.key == rows)
        });
        let result = match ready {
            Some(p) if prompt.0.len() >= n * H * 4 && prompt.1.len() >= n * H => {
                if p.residual.len() == prompt.0.len() && p.lexical.len() == prompt.1.len() {
                    (prompt.0, prompt.1) = (p.residual, p.lexical);
                } else {
                    prompt.0[..n * H * 4].copy_from_slice(&p.residual);
                    prompt.1[..n * H].copy_from_slice(&p.lexical);
                }
                Ok(())
            }
            _ => self.frontend_rows(rows, global, &mut prompt.0, &mut prompt.1),
        };
        let (count, done) = &*self.finished;
        if let Ok(mut n) = count.lock() {
            *n += 1;
            done.notify_all();
        }
        result
    }
    fn frontend_rows_with(
        &self,
        rows: &[(u32, Vec<u64>)],
        global: f32,
        residual: &mut [f32],
        lexical: &mut [f32],
        readers: usize,
    ) -> Result<()> {
        let n = rows.len();
        if residual.len() < n * H * 4 || lexical.len() < n * H {
            return Err("prefill frontend rows exceed staging".into());
        }
        let (residual, lexical) = (&mut residual[..n * H * 4], &mut lexical[..n * H]);
        let mut requests = Vec::with_capacity(n * 33);
        for (i, (token, hashes)) in rows.iter().enumerate() {
            self.requests(i, *token, hashes, &mut requests)?;
        }
        let mut staged = vec![0u8; n * STAGED];
        read_plan::Plan::new(requests)
            .read(&self.files, readers, &mut staged)
            .map_err(err)?;
        let threads = (n / 64).clamp(1, 8);
        let block = n.div_ceil(threads).max(1);
        let decode_block = |staged: &[u8], x: &mut [f32], e: &mut [f32]| -> Result<()> {
            for ((s, x), e) in staged
                .chunks_exact(STAGED)
                .zip(x.chunks_exact_mut(H * 4))
                .zip(e.chunks_exact_mut(H))
            {
                Self::decode(s, global, x, e)?;
            }
            Ok(())
        };
        if threads == 1 {
            return decode_block(&staged, residual, lexical);
        }
        std::thread::scope(|scope| {
            let jobs: Vec<_> = staged
                .chunks(block * STAGED)
                .zip(residual.chunks_mut(block * H * 4))
                .zip(lexical.chunks_mut(block * H))
                .map(|((s, x), e)| scope.spawn(move || decode_block(s, x, e)))
                .collect();
            jobs.into_iter().try_for_each(|job| {
                job.join()
                    .map_err(|_| "prefill frontend thread panicked".to_string())?
            })
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    const LOCAL: u64 = 1536;
    const VOCAB: u64 = 384;
    fn table(name: &str, bytes: usize, seed: u64, modulus: u64) -> Arc<File> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("vade-prefill-{}-{unique}-{name}", std::process::id()));
        let mut x = seed;
        let data: Vec<u8> = (0..bytes)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % modulus) as u8
            })
            .collect();
        File::create(&path).unwrap().write_all(&data).unwrap();
        let file = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        Arc::new(file)
    }
    /// 128 shards whose nibble and scale tensors sit at shifted (and partly
    /// shared) starts in two files, the embedding in a third: the same layout
    /// questions a checkpoint poses (several tensors per file, unaligned starts).
    fn reader() -> Reader {
        let files = vec![
            table("nibbles", (LOCAL * 80) as usize + 4 * 997, 11, 256),
            // Nonnegative finite E4M3 scales (0x7f is NaN).
            table("scales", (LOCAL * 10) as usize + 3 * 7, 13, 0x7f),
            table("embed", (VOCAB * 5120) as usize, 17, 256),
        ];
        let span = |id: u32, start: u64, len: u64| Span { file: files[id as usize].clone(), id, start, len };
        let ple = (0..PLE_ROWS / PLE_PART_ROWS)
            .map(|s| [span(0, (s % 4) * 997, LOCAL * 80), span(1, (s % 3) * 7, LOCAL * 10)])
            .collect();
        Reader { ple, embed: span(2, 0, VOCAB * 5120), files: files.clone(), prestaged: Default::default(), finished: Default::default() }
    }
    #[test]
    fn plan_frontend_matches_per_row_token_reads() {
        let reader = reader();
        let shards = PLE_ROWS / PLE_PART_ROWS;
        for (seed, n) in [(1u64, 1usize), (2, 63), (3, 64), (4, 65), (5, 700)] {
            let mut x = seed.wrapping_mul(0x2545_f491_4f6c_dd1d);
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let rows: Vec<(u32, Vec<u64>)> = (0..n)
                .map(|i| {
                    let token = if i % 5 == 0 { 7 } else { (next() % VOCAB) as u32 };
                    let ple = (0..16)
                        .map(|h| match (i + h) % 3 {
                            0 => 3 * PLE_PART_ROWS + 42,                                 // repeated row
                            1 => (next() % shards) * PLE_PART_ROWS + next() % 32,     // neighbours
                            _ => (next() % shards) * PLE_PART_ROWS + next() % LOCAL,  // sparse
                        })
                        .collect();
                    (token, ple)
                })
                .collect();
            let (mut want_x, mut want_e) = (vec![0f32; n * H * 4], vec![0f32; n * H]);
            for (i, (token, ple)) in rows.iter().enumerate() {
                reader
                    .token(*token, ple, 0.37, &mut want_x[i * H * 4..(i + 1) * H * 4], &mut want_e[i * H..(i + 1) * H])
                    .unwrap();
            }
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            let packed = reader.frontend_packed(&rows).unwrap();
            assert_eq!(packed.len(), n * 1444);
            for (i, (token, hashes)) in rows.iter().enumerate() {
                assert_eq!(&packed[i * 4..i * 4 + 4], &token.to_le_bytes());
                for (h, &row) in hashes.iter().enumerate() {
                    let mut want = [0u8; 90];
                    reader.ple_row(row, &mut want).unwrap();
                    let at = n * 4 + i * 1440 + h * 90;
                    assert_eq!(&packed[at..at + 90], &want);
                }
            }
            let compact = reader.frontend_packed_dedup(&rows).unwrap();
            assert_eq!(&compact[..n*4], &packed[..n*4]);
            for (i,(_,hashes)) in rows.iter().enumerate() {
                for h in 0..hashes.len() {
                    let at=n*4+(i*16+h)*4;
                    let index=u32::from_le_bytes(compact[at..at+4].try_into().unwrap()) as usize;
                    assert_eq!(&compact[n*68+index*90..n*68+(index+1)*90], &packed[n*4+i*1440+h*90..n*4+i*1440+(h+1)*90]);
                }
            }
            assert_eq!(compact,reader.frontend_packed_dedup(&rows).unwrap());
            for readers in [1, 4, 16] {
                // Oversized staging, as in the workspace: rows past n stay untouched.
                let (mut x, mut e) = (vec![-1f32; (n + 3) * H * 4], vec![-1f32; (n + 3) * H]);
                reader.frontend_rows_with(&rows, 0.37, &mut x, &mut e, readers).unwrap();
                assert!(bits(&x[..n * H * 4]) == bits(&want_x), "residual n {n} readers {readers}");
                assert!(bits(&e[..n * H]) == bits(&want_e), "lexical n {n} readers {readers}");
                assert!(x[n * H * 4..].iter().chain(&e[n * H..]).all(|&v| v == -1.0));
            }
            // Read-ahead and per-row hints change no bytes.
            reader.read_ahead(hash(), vec![1, 2, 3], 1..3, 0.37);
            let mut prompt = (vec![0f32; n * H * 4], vec![0f32; n * H]);
            reader.frontend_prompt(&rows, 0.37, &mut prompt).unwrap();
            assert!(bits(&prompt.0) == bits(&want_x) && bits(&prompt.1) == bits(&want_e), "frontend_prompt n {n}");
            reader.prefetch_ple_rows(&rows[0].1);
            reader.hint_rows(&rows);
            let (mut x, mut e) = (vec![0f32; n * H * 4], vec![0f32; n * H]);
            reader.frontend_rows(&rows, 0.37, &mut x, &mut e).unwrap();
            assert!(bits(&x) == bits(&want_x) && bits(&e) == bits(&want_e), "default readers n {n}");
        }
    }
    fn hash() -> refcpu::PleHash {
        // Rows below 1000 stay inside the synthetic shard-0 tensors.
        refcpu::PleHash::new(vec![3, 5, 7], vec![1000; 16], vec![0; 16], 9)
    }
    fn oracle(reader: &Reader, rows: &[(u32, Vec<u64>)], global: f32) -> (Vec<u32>, Vec<u32>) {
        let (mut x, mut e) = (vec![0f32; rows.len() * H * 4], vec![0f32; rows.len() * H]);
        for (i, (token, ple)) in rows.iter().enumerate() {
            reader
                .token(*token, ple, global, &mut x[i * H * 4..(i + 1) * H * 4], &mut e[i * H..(i + 1) * H])
                .unwrap();
        }
        (x.iter().map(|v| v.to_bits()).collect(), e.iter().map(|v| v.to_bits()).collect())
    }
    /// A prestaged next tile is taken only for the identical (token, rows) list
    /// and global scale, swapped in at the staging length or copied into the
    /// leading rows of a longer staging; its bytes equal per-row `token`.
    #[test]
    fn prestaged_tile_equals_token_and_is_taken_only_on_exact_key() {
        let reader = reader();
        let tokens: Vec<u32> = (0..200u32).map(|i| (i * 37 + i / 7) % VOCAB as u32).collect();
        let mut h = hash();
        let rows: Vec<(u32, Vec<u64>)> = tokens.iter().map(|&t| (t, h.advance(t))).collect();
        let next = 64..192;
        let want = oracle(&reader, &rows[next.clone()], 0.37);
        let bits = |p: &(Vec<f32>, Vec<f32>), n: usize| {
            (
                p.0[..n * H * 4].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                p.1[..n * H].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            )
        };
        let taken = |reader: &Reader| reader.prestaged.lock().unwrap().is_none();
        for staging in [128usize, 300] {
            // Not the latest job: hints only, nothing prestaged.
            reader.read_ahead_job(&mut hash(), &tokens, next.clone(), 0.37, false);
            assert!(taken(&reader));
            reader.read_ahead_job(&mut hash(), &tokens, next.clone(), 0.37, true);
            assert!(!taken(&reader));
            // Wrong key (one row shifted) and wrong global: computed, slot kept.
            let mut prompt = (vec![0f32; staging * H * 4], vec![0f32; staging * H]);
            reader.frontend_prompt(&rows[63..191], 0.37, &mut prompt).unwrap();
            assert!(bits(&prompt, 128) == oracle(&reader, &rows[63..191], 0.37));
            reader.frontend_prompt(&rows[next.clone()], 0.5, &mut prompt).unwrap();
            assert!(bits(&prompt, 128) == oracle(&reader, &rows[next.clone()], 0.5));
            assert!(!taken(&reader));
            // Exact key: taken, identical bytes, staging length unchanged.
            let mut prompt = (vec![-1f32; staging * H * 4], vec![-1f32; staging * H]);
            reader.frontend_prompt(&rows[next.clone()], 0.37, &mut prompt).unwrap();
            assert!(taken(&reader));
            assert!(bits(&prompt, 128) == want, "staging {staging}");
            assert_eq!((prompt.0.len(), prompt.1.len()), (staging * H * 4, staging * H));
        }
        // Ranges past the tokens are clipped; an empty next tile prestages nothing.
        reader.read_ahead_job(&mut hash(), &tokens, 300..400, 0.37, true);
        assert!(taken(&reader));
    }
    #[test]
    fn plan_frontend_rejects_what_token_rejects() {
        let reader = reader();
        let (mut x, mut e) = (vec![0f32; H * 4], vec![0f32; H]);
        let bad_row = vec![(1u32, vec![PLE_ROWS; 16])];
        assert!(reader.frontend_rows_with(&bad_row, 1.0, &mut x, &mut e, 4).is_err());
        let past_tensor = vec![(1u32, vec![LOCAL; 16])];
        assert!(reader.frontend_rows_with(&past_tensor, 1.0, &mut x, &mut e, 4).is_err());
        let short = vec![(1u32, vec![0; 15])];
        assert!(reader.frontend_rows_with(&short, 1.0, &mut x, &mut e, 4).is_err());
        let past_vocab = vec![(VOCAB as u32, vec![0; 16])];
        assert!(reader.frontend_rows_with(&past_vocab, 1.0, &mut x, &mut e, 4).is_err());
        assert!(reader.token(VOCAB as u32, &[0; 16], 1.0, &mut x, &mut e).is_err());
    }
}
