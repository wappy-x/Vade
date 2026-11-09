//! Rank-owned vocabulary and n-gram rows. Only requested stored bytes cross ranks.
use super::Weights;
use engine_model::flash::{partition, PLE_PART_ROWS, PLE_PREFIX, PLE_ROWS};
use spark_gpu_probe::driver::{Backend, DeviceVector, PromptBuffer};
use spark_gpu_probe::kernel_params::{HybridCopy, HybridCopyRow};
use std::{ffi::c_void, ops::Range};

type Result<T> = std::result::Result<T, String>;
pub const HIDDEN: usize = 2560;
pub const STREAMS: usize = 4;
pub const EMBEDDING_BYTES: usize = HIDDEN * 2;
pub const LEXICAL_BYTES: usize = 16 * 160;
const EOS: i64 = 248044;
const VOCABULARY: u64 = 248320;
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

/// Two token IDs belong to the lexical lookup, independently of PLE convolution.
#[derive(Clone)]
pub struct Rows {
    multipliers: [i64; 3],
    sizes: [i64; 16],
    offsets: [i64; 16],
    history: [i64; 2],
    scale: f32,
    rank: usize,
    vocabulary: Vec<Range<u64>>,
    lexical: Vec<Range<u64>>,
}

fn integers<const N: usize>(weights: &Weights<'_>, suffix: &str) -> Result<[i64; N]> {
    let name = format!("{PLE_PREFIX}.{suffix}");
    if weights.tensor(&name)?.bytes != N * 8 { return Err(format!("unexpected n-gram buffer {suffix}")); }
    let mut bytes = vec![0; N * 8];
    weights.read(&name, 0, &mut bytes)?;
    Ok(std::array::from_fn(|i| i64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap())))
}

impl Rows {
    pub fn new(weights: &Weights<'_>) -> Result<Self> {
        let multipliers = integers(weights, "layer_multipliers")?;
        let sizes = integers(weights, "ngram_heads_vocab_sizes")?;
        let offsets = integers(weights, "ngram_heads_offsets")?;
        if sizes.iter().zip(offsets).any(|(&size, offset)| size <= 0 || offset < 0
            || offset.checked_add(size).is_none_or(|end| end as u64 > PLE_ROWS)) {
            return Err("invalid checkpoint n-gram head ranges".into());
        }
        let scale = weights.value_f32(&format!("{PLE_PREFIX}.ngram_embedding.weight_scale"), 0)?;
        let ranges = |total| (0..weights.plan.ranks).map(|rank|
            partition(total, rank, weights.plan.ranks).map_err(|e| e.to_string())).collect::<Result<Vec<_>>>();
        Ok(Self { multipliers, sizes, offsets, history: [EOS; 2], scale,
            rank: weights.plan.rank as usize, vocabulary: ranges(VOCABULARY)?, lexical: ranges(PLE_ROWS)? })
    }
    pub fn reset(&mut self) { self.history = [EOS; 2]; }
    pub fn history(&self) -> [i64; 2] { self.history }
    pub(super) fn restore_history(&mut self, history: [i64; 2]) { self.history = history; }

    /// Signed wrapping I64 is the checkpoint's hash arithmetic. EOS can see
    /// its preceding segment; the following token starts with EOS padding.
    pub fn advance(&mut self, token: u32) -> Result<[u64; 16]> {
        if token as u64 >= VOCABULARY { return Err("token outside vocabulary".into()); }
        let previous = if self.history[0] == EOS { [EOS; 2] } else { self.history };
        let bigram = (token as i64).wrapping_mul(self.multipliers[0])
            ^ previous[0].wrapping_mul(self.multipliers[1]);
        let trigram = bigram ^ previous[1].wrapping_mul(self.multipliers[2]);
        let rows = std::array::from_fn(|head| {
            let hash = if head < 8 { bigram } else { trigram };
            (hash.rem_euclid(self.sizes[head]) + self.offsets[head]) as u64
        });
        self.history = [token as i64, previous[0]];
        Ok(rows)
    }

    /// Capture one decode token or a bounded prefill tile. Each request reads
    /// one BF16 row and exactly sixteen FP8 rows across all owners. No table is
    /// expanded, copied, or restricted to a predetermined set of token IDs.
    pub fn capture(&mut self, weights: &Weights<'_>, tokens: &[u32]) -> Result<RowBatch> {
        self.capture_rows(weights, tokens, None)
    }
    pub(super) fn capture_device(&mut self, weights: &Weights<'_>, token: u32, destination: u64) -> Result<RowBatch> {
        self.capture_rows(weights, &[token], Some(destination))
    }
    pub(super) fn capture_prompt(&mut self, weights: &Weights<'_>, tokens: &[u32], offset: usize) -> Result<RowBatch> {
        self.capture_rows(weights, tokens, Some(offset as u64))
    }
    fn capture_rows(&mut self, weights: &Weights<'_>, tokens: &[u32], destination: Option<u64>) -> Result<RowBatch> {
        if weights.plan.rank as usize != self.rank || weights.plan.ranks as usize != self.vocabulary.len() {
            return Err("row lookup and resident rank differ".into());
        }
        if tokens.is_empty() || tokens.len() > 64 { return Err("row gather accepts 1..=64 unpadded tokens".into()); }
        if tokens.iter().any(|&t| t as u64 >= VOCABULARY) { return Err("token outside vocabulary".into()); }
        let previous_history = self.history;
        let result = self.capture_inner(weights, tokens, destination);
        if result.is_err() { self.history = previous_history; }
        result
    }

    fn capture_inner(&mut self, weights: &Weights<'_>, tokens: &[u32], destination: Option<u64>) -> Result<RowBatch> {
        let mut requests = Vec::with_capacity(tokens.len() * 17);
        for (position, &token) in tokens.iter().enumerate() {
            let owner = self.vocabulary.iter().position(|r| r.contains(&(token as u64))).ok_or("embedding owner missing")?;
            requests.push(Request { owner, position, head: 16, row: token as u64, offset: 0 });
            for (head, row) in self.advance(token)?.into_iter().enumerate() {
                let owner = self.lexical.iter().position(|r| r.contains(&row)).ok_or("n-gram owner missing")?;
                requests.push(Request { owner, position, head, row, offset: 0 });
            }
        }
        requests.sort_by_key(|r| r.owner);
        let mut offsets = vec![0; self.vocabulary.len() + 1];
        let mut locations = vec![[0; 17]; tokens.len()];
        let mut length = 0;
        for request in &mut requests {
            request.offset = length;
            locations[request.position][request.head] = length;
            let count = if request.head == 16 { EMBEDDING_BYTES } else { 160 };
            offsets[request.owner + 1] += count;
            length += count;
        }
        for rank in 1..offsets.len() { offsets[rank] += offsets[rank - 1]; }
        let mut wire = vec![0; length];
        let mut copies = Vec::new();
        for request in requests.iter().filter(|r| r.owner == self.rank) {
            let (name, global, count) = if request.head == 16 {
                (EMBEDDING.to_owned(), request.row * EMBEDDING_BYTES as u64, EMBEDDING_BYTES)
            } else {
                (format!("{PLE_PREFIX}.ngram_embedding.shard_{}.weight", request.row / PLE_PART_ROWS),
                    (request.row % PLE_PART_ROWS) * 160, 160)
            };
            let local = weights.tensor(&name)?.local_byte_offset(global)?;
            if let Some(destination) = destination {
                copies.push(HybridCopyRow { source: weights.row_address(&name, local, count)?,
                    destination: destination + request.offset as u64, words: (count / 4) as u32, padding: 0 });
            } else { weights.read(&name, local, &mut wire[request.offset..request.offset + count])?; }
        }
        Ok(RowBatch { wire, offsets, locations, scale: self.scale, copies })
    }
}

struct Request { owner: usize, position: usize, head: usize, row: u64, offset: usize }

/// The same deterministic layout is constructed by every rank. Populate only
/// the local byte interval, then all-gather those intervals without summation.
pub struct RowBatch {
    wire: Vec<u8>,
    offsets: Vec<usize>,
    locations: Vec<[usize; 17]>,
    scale: f32,
    copies: Vec<HybridCopyRow>,
}
impl RowBatch {
    pub(super) fn take_prompt_copies(&mut self) -> Vec<HybridCopyRow> { std::mem::take(&mut self.copies) }
    pub(super) fn fill_prompt(&mut self, bytes: &[u8], offset: usize, rank: usize) {
        let (start, end) = (self.offsets[rank], self.offsets[rank + 1]);
        self.wire[start..end].copy_from_slice(&bytes[offset + start..offset + end]);
    }
    /// Local rows are packed during lookahead. After the ordered all-gather,
    /// patch only remote rows into the same embedding/lexical byte planes.
    pub(super) fn pack_prompt(&self, bytes: &mut [u8], base: usize, first: usize,
        group: usize, rank: usize, local: bool) {
        for (t, locations) in self.locations.iter().enumerate() {
            for (head, &source) in locations.iter().enumerate() {
                let owned = (self.offsets[rank]..self.offsets[rank + 1]).contains(&source);
                if owned != local { continue; }
                let (destination, count) = if head == 16 {
                    (base + (first + t) * EMBEDDING_BYTES, EMBEDDING_BYTES)
                } else {
                    (base + group * EMBEDDING_BYTES + (first + t) * LEXICAL_BYTES + head * 160, 160)
                };
                bytes[destination..destination + count].copy_from_slice(&self.wire[source..source + count]);
            }
        }
    }
    /// Queue all locally owned immutable rows before one readback, then gather
    /// rank-major blocks once. Each session retains its own n-gram history.
    pub(super) fn gather_device(b: &Backend<'_>, batches: &mut [Self], rank: usize,
        scratch: &mut DeviceVector<'_>, table: &mut DeviceVector<'_>,
        exchange: impl FnOnce(&mut [u8], &[usize]) -> Result<()>) -> Result<()> {
        let first = batches.first().ok_or("empty Flash row batch")?;
        let bytes = first.wire.len();
        let ranks = first.offsets.len() - 1;
        if rank >= ranks || batches.iter().any(|v| v.wire.len() != bytes || v.offsets.len() != ranks + 1) {
            return Err("Flash row batches differ in shape".into());
        }
        scratch.resize_within_capacity(batches.len() * bytes / 4)?;
        let copies = batches.iter().flat_map(|v| v.copies.iter().copied()).collect::<Vec<_>>();
        if !copies.is_empty() {
            let raw = unsafe { std::slice::from_raw_parts(copies.as_ptr().cast::<u8>(), std::mem::size_of_val(copies.as_slice())) };
            b.write_vector_bytes(table, raw)?;
            let mut p = HybridCopy { source: std::ptr::null(), destination: std::ptr::null_mut(),
                words: 0, padding: 0, rows: table.device_address() };
            unsafe { b.launch_flash_rows("flash_hybrid_copy", (EMBEDDING_BYTES / 4).div_ceil(128) as u32,
                copies.len() as u32, 128, 0, &mut [(&mut p as *mut HybridCopy).cast()])?; }
            let values = b.read_vector_values(scratch)?;
            let raw = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
            for (i, v) in batches.iter_mut().enumerate() {
                let (start, end) = (v.offsets[rank], v.offsets[rank + 1]);
                v.wire[start..end].copy_from_slice(&raw[i * bytes + start..i * bytes + end]);
            }
        }
        let mut offsets = vec![0; ranks + 1];
        for r in 0..ranks {
            offsets[r + 1] = offsets[r] + batches.iter().map(|v| v.offsets[r + 1] - v.offsets[r]).sum::<usize>();
        }
        let mut wire = vec![0; *offsets.last().unwrap()];
        for r in 0..ranks {
            let mut at = offsets[r];
            for v in batches.iter() {
                let local = &v.wire[v.offsets[r]..v.offsets[r + 1]];
                wire[at..at + local.len()].copy_from_slice(local);
                at += local.len();
            }
        }
        exchange(&mut wire, &offsets)?;
        for r in 0..ranks {
            let mut at = offsets[r];
            for v in batches.iter_mut() {
                let local = &mut v.wire[v.offsets[r]..v.offsets[r + 1]];
                local.copy_from_slice(&wire[at..at + local.len()]);
                at += local.len();
            }
        }
        Ok(())
    }
    pub fn wire(&mut self) -> &mut [u8] { &mut self.wire }
    pub fn offsets(&self) -> &[usize] { &self.offsets }
    pub fn len(&self) -> usize { self.locations.len() }
    pub fn is_empty(&self) -> bool { self.locations.is_empty() }
    pub fn token(&self, position: usize) -> Result<TokenRows> {
        let locations = self.locations.get(position).ok_or("token outside gathered tile")?;
        let mut out = TokenRows { embedding: [0; EMBEDDING_BYTES], ple: [0; LEXICAL_BYTES], scale: self.scale };
        out.embedding.copy_from_slice(&self.wire[locations[16]..locations[16] + EMBEDDING_BYTES]);
        for (head, &at) in locations[..16].iter().enumerate() {
            out.ple[head*160..head*160+160].copy_from_slice(&self.wire[at..at+160]);
        }
        Ok(out)
    }
    /// Pass the engine transport's raw-byte all-gather. A tile is complete only
    /// after this synchronous boundary returns successfully on every rank.
    pub fn gather(&mut self, exchange: impl FnOnce(&mut [u8], &[usize]) -> Result<()>) -> Result<()> {
        exchange(&mut self.wire, &self.offsets)
    }
}

pub struct TokenRows {
    pub embedding: [u8; EMBEDDING_BYTES],
    pub ple: [u8; LEXICAL_BYTES],
    pub scale: f32,
}

/// One token's raw rows and FP32 outputs, reused across prompt/decode positions,
/// plus one group's raw rows for prompt tiles.
pub struct FrontScratch<'a> {
    embedding_raw: DeviceVector<'a>,
    lexical_raw: DeviceVector<'a>,
    pub residual: DeviceVector<'a>,
    pub lexical: DeviceVector<'a>,
    tokens: usize,
}
impl<'a> FrontScratch<'a> {
    pub const DEVICE_BYTES: usize = EMBEDDING_BYTES + LEXICAL_BYTES + (STREAMS + 1) * HIDDEN * 4;
    pub fn new(backend: &Backend<'a>) -> Result<Self> { Self::with_tokens(backend, 1) }
    /// Raw row buffers for `tokens` consecutive prompt tokens.
    pub fn with_tokens(backend: &Backend<'a>, tokens: usize) -> Result<Self> {
        Ok(Self { embedding_raw: backend.vector(tokens * EMBEDDING_BYTES / 4)?,
            lexical_raw: backend.vector(tokens * LEXICAL_BYTES / 4)?,
            residual: backend.vector(STREAMS * HIDDEN)?, lexical: backend.vector(HIDDEN)?, tokens })
    }
    pub fn device_bytes(&self) -> usize { self.tokens * (EMBEDDING_BYTES + LEXICAL_BYTES) + (STREAMS + 1) * HIDDEN * 4 }
    /// Expand `tokens` gathered embedding rows, uploaded together, straight into
    /// consecutive W-wide residual rows of `hidden`: per token the same kernel
    /// and values as `expand`, without an upload and a copy per token.
    pub fn expand_rows(&mut self, backend: &Backend<'_>, rows: &[u8], hidden: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > self.tokens || rows.len() != tokens * EMBEDDING_BYTES || hidden.len() < tokens * STREAMS * HIDDEN {
            return Err("embedding rows do not match the front workspace".into());
        }
        backend.write_vector_bytes(&mut self.embedding_raw, rows)?;
        self.expand_prompt(backend, hidden, tokens)
    }
    pub(super) fn expand_staged(&mut self, backend: &Backend<'_>, slot: &PromptBuffer<'_>,
        offset: usize, hidden: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > self.tokens || hidden.len() < tokens * STREAMS * HIDDEN {
            return Err("embedding rows do not match the front workspace".into());
        }
        backend.write_prompt_bytes(&mut self.embedding_raw, slot, offset, tokens * EMBEDDING_BYTES)?;
        self.expand_prompt(backend, hidden, tokens)
    }
    fn expand_prompt(&self, backend: &Backend<'_>, hidden: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        for t in 0..tokens {
            let mut raw = self.embedding_raw.device_address() + (t * EMBEDDING_BYTES) as u64;
            let mut output = hidden.device_address() + (t * STREAMS * HIDDEN * 4) as u64;
            let mut width = HIDDEN as u32;
            unsafe { backend.launch_flash("flash_embedding_expand", (STREAMS * HIDDEN / 128) as u32, 128,
                &mut [(&mut raw as *mut u64).cast::<c_void>(), (&mut output as *mut u64).cast(), (&mut width as *mut u32).cast()])?; }
        }
        Ok(())
    }
    /// Upload `tokens` gathered lexical rows together; `lexical_row` addresses each.
    pub fn upload_lexical_rows(&mut self, backend: &Backend<'_>, rows: &[u8], tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > self.tokens || rows.len() != tokens * LEXICAL_BYTES {
            return Err("lexical rows do not match the front workspace".into());
        }
        backend.write_vector_bytes(&mut self.lexical_raw, rows)
    }
    pub(super) fn upload_staged_lexical(&mut self, backend: &Backend<'_>, slot: &PromptBuffer<'_>,
        offset: usize, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > self.tokens { return Err("lexical rows do not match the front workspace".into()); }
        backend.write_prompt_bytes(&mut self.lexical_raw, slot, offset, tokens * LEXICAL_BYTES)
    }
    pub fn lexical_row(&self, token: usize) -> u64 { self.lexical_raw.device_address() + (token * LEXICAL_BYTES) as u64 }
    pub fn expand(&mut self, backend: &Backend<'_>, rows: &TokenRows) -> Result<()> {
        backend.write_vector_bytes(&mut self.embedding_raw, &rows.embedding)?;
        backend.write_vector_bytes(&mut self.lexical_raw, &rows.ple)?;
        let mut raw = self.embedding_raw.device_address();
        let mut output = self.residual.device_address();
        let mut width = HIDDEN as u32;
        unsafe { backend.launch_flash("flash_embedding_expand", (STREAMS * HIDDEN / 128) as u32, 128,
            &mut [(&mut raw as *mut u64).cast::<c_void>(), (&mut output as *mut u64).cast(), (&mut width as *mut u32).cast()])?; }
        raw = self.lexical_raw.device_address();
        output = self.lexical.device_address();
        let mut scale = rows.scale;
        unsafe { backend.launch_flash("flash_ple_decode", (HIDDEN / 128) as u32, 128,
            &mut [(&mut raw as *mut u64).cast::<c_void>(), (&mut output as *mut u64).cast(), (&mut scale as *mut f32).cast()]) }
    }
}
