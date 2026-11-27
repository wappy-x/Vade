//! On-disk binary format for bit-exact loop-decoder KV prefix cache snapshots.
//!
//! Layout:
//!   1. Fixed 64-byte Prelude (`MAGIC`, `FORMAT_VERSION`, flags, `header_bytes`, `payload_bytes`,
//!      `chunk_bytes`, `chunk_count`).
//!   2. Metadata Header (`engine_tag`, `checkpoint_fingerprint`, `scope`, `vocab_rows`, `position`,
//!      `prompt_sha256`, target visit layouts, optional draft config descriptor, `logits_len`,
//!      exact `prompt_tokens`, exact `next_logits`).
//!   3. Tensor Payload (`payload_bytes` streamed in `chunk_bytes` chunks: target straight/ring K+V
//!      planes, 4 target conv rings per visit, optional draft local/global K+V planes + feature).
//!   4. Integrity Footer (`chunk_crc32[chunk_count]`, `file_sha256` over Prelude || Header ||
//!      chunk_crc32 bytes, `FOOTER_MAGIC`).
use crate::loop_decoder::state::{self, Layout};
use engine_model::mtp_draft::MtpConfig;
use std::io::{self, Read, Seek, SeekFrom};

pub(super) const MAGIC: &[u8; 8] = b"VADEDPC1";
pub(super) const FOOTER_MAGIC: &[u8; 8] = b"DPC1END!";
pub(super) const FORMAT_VERSION: u32 = 1;
pub(super) const ENGINE_TAG: &str = "vade.loop_decoder.kv_disk_cache.v1";
pub(super) const PRELUDE_BYTES: u64 = 64;
pub(super) const FOOTER_FIXED_BYTES: u64 = 40; // 32-byte SHA-256 + 8-byte footer magic
pub(super) const FLAG_HAS_DRAFT: u32 = 1;
/// Payload streaming chunk size: matches `spark_gpu_probe::staging::UPLOAD_BYTES` (4 MiB = 2 VMM pages).
pub(super) const CHUNK_BYTES: usize = 4 * 1024 * 1024;
pub(super) const CHUNK_WORDS: usize = CHUNK_BYTES / 4;
/// Upper bound on `chunk_bytes` accepted from an untrusted file on disk (64 MiB).
const MAX_ACCEPTED_CHUNK_BYTES: u32 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VisitDesc {
    pub ring: bool,
    pub window: u32,
    /// Fixed ring row count (`window + spare_rows`) for ring visits; 0 for straight planes so a
    /// snapshot remains valid across different `--context` settings while `position <= context`.
    pub ring_rows: u32,
    pub row_words: u32,
    pub kv_channels: u32,
    pub kv_taps: u32,
    pub channels: u32,
    pub taps: u32,
}

impl VisitDesc {
    pub(super) fn from_layout(l: &Layout) -> Result<Self, String> {
        let u = |v: usize, name: &str| u32::try_from(v).map_err(|_| format!("disk cache layout {name} overflow"));
        Ok(Self {
            ring: l.ring,
            window: if l.ring { u(l.window, "window")? } else { 0 },
            ring_rows: if l.ring { u(l.rows, "ring_rows")? } else { 0 },
            row_words: u(l.row_words, "row_words")?,
            kv_channels: u(l.kv_channels, "kv_channels")?,
            kv_taps: u(l.kv_taps, "kv_taps")?,
            channels: u(l.channels, "channels")?,
            taps: u(l.taps, "taps")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DraftDesc {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub kv_width: u32,
    pub sliding_window_size: u32,
    pub block_size: u32,
    pub is_local: Vec<u8>,
    pub target_layer_ids: Vec<u32>,
}

impl DraftDesc {
    pub(super) fn from_config(cfg: &MtpConfig, kv_bf16: bool) -> Result<Self, String> {
        let u = |v: usize, name: &str| u32::try_from(v).map_err(|_| format!("disk cache draft {name} overflow"));
        let num_layers = u(cfg.num_layers, "num_layers")?;
        let is_local = (0..cfg.num_layers).map(|i| u8::from(cfg.is_local(i))).collect();
        let target_layer_ids = cfg.target_layer_ids.iter().map(|&id| u(id, "target_layer_id")).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            num_layers,
            hidden_size: u(cfg.hidden_size, "hidden_size")?,
            kv_width: u(super::super::draft::kv_row_words(cfg, kv_bf16), "kv_width")?,
            sliding_window_size: u(cfg.sliding_window_size, "sliding_window_size")?,
            block_size: u(cfg.block_size, "block_size")?,
            is_local,
            target_layer_ids,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Prelude {
    pub flags: u32,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub chunk_bytes: u32,
    pub chunk_count: u32,
}

impl Prelude {
    pub(super) fn encode(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..8].copy_from_slice(MAGIC);
        out[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out[16..24].copy_from_slice(&self.header_bytes.to_le_bytes());
        out[24..32].copy_from_slice(&self.payload_bytes.to_le_bytes());
        out[32..36].copy_from_slice(&self.chunk_bytes.to_le_bytes());
        out[36..40].copy_from_slice(&self.chunk_count.to_le_bytes());
        out
    }

    pub(super) fn decode(raw: &[u8; 64], file_len: u64) -> Result<Self, String> {
        if &raw[0..8] != MAGIC {
            return Err("invalid magic".into());
        }
        let version = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(format!("unsupported format version {version}"));
        }
        let flags = u32::from_le_bytes(raw[12..16].try_into().unwrap());
        if flags & !FLAG_HAS_DRAFT != 0 {
            return Err(format!("unknown flags {flags:#x}"));
        }
        let header_bytes = u64::from_le_bytes(raw[16..24].try_into().unwrap());
        let payload_bytes = u64::from_le_bytes(raw[24..32].try_into().unwrap());
        let chunk_bytes = u32::from_le_bytes(raw[32..36].try_into().unwrap());
        let chunk_count = u32::from_le_bytes(raw[36..40].try_into().unwrap());
        if raw[40..64].iter().any(|&b| b != 0) {
            return Err("nonzero reserved prelude bytes".into());
        }
        if header_bytes == 0 || payload_bytes == 0 || payload_bytes % 4 != 0 {
            return Err("invalid header or payload byte count".into());
        }
        if chunk_bytes == 0 || chunk_bytes % 4 != 0 || chunk_bytes > MAX_ACCEPTED_CHUNK_BYTES {
            return Err(format!("invalid chunk_bytes {chunk_bytes}"));
        }
        let expected_chunks = payload_bytes.div_ceil(u64::from(chunk_bytes));
        if expected_chunks != u64::from(chunk_count) {
            return Err(format!("chunk_count mismatch: header={chunk_count} expected={expected_chunks}"));
        }
        let footer_bytes = u64::from(chunk_count)
            .checked_mul(4)
            .and_then(|n| n.checked_add(FOOTER_FIXED_BYTES))
            .ok_or("footer byte count overflow")?;
        let total = PRELUDE_BYTES
            .checked_add(header_bytes)
            .and_then(|n| n.checked_add(payload_bytes))
            .and_then(|n| n.checked_add(footer_bytes))
            .ok_or("file byte count overflow")?;
        if total != file_len {
            return Err(format!("file length mismatch: actual={file_len} expected={total}"));
        }
        Ok(Self { flags, header_bytes, payload_bytes, chunk_bytes, chunk_count })
    }
}

pub(super) struct ParsedHeader {
    pub engine_tag: String,
    pub checkpoint_fingerprint: String,
    pub scope: Option<String>,
    pub vocab_rows: u32,
    pub position: u64,
    pub prompt_sha256: [u8; 32],
    pub visits: Vec<VisitDesc>,
    pub draft: Option<DraftDesc>,
    pub logits_len: u32,
    pub first_token: u32,
    pub last_token: u32,
    pub prefix_64_hash: u64,
    pub tokens: Vec<u32>,
    pub next_logits: Vec<f32>,
}

/// Fast 64-bit FNV-1a hash of up to the first 64 prompt tokens (used only as an O(1) candidate pre-filter;
/// exact SHA-256 and token-by-token comparison always authorize every hit).
pub(super) fn prefix_64_hash(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in tokens.iter().take(64) {
        h = (h ^ u64::from(t)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Streaming reader over `&[u32]` as little-endian bytes so SHA-256 needs zero heap allocation.
struct U32LeReader<'a> {
    tokens: &'a [u32],
    byte_pos: usize,
}

impl Read for U32LeReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let total = self.tokens.len() * 4;
        if self.byte_pos >= total || buf.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        while written < buf.len() && self.byte_pos < total {
            let word = self.tokens[self.byte_pos / 4].to_le_bytes();
            let sub = self.byte_pos % 4;
            let take = (4 - sub).min(buf.len() - written);
            buf[written..written + take].copy_from_slice(&word[sub..sub + take]);
            written += take;
            self.byte_pos += take;
        }
        Ok(written)
    }
}

pub(super) fn hex_to_32(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err("invalid sha256 hex length".into());
    }
    let mut out = [0u8; 32];
    let nibble = |b: u8| match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    };
    for (dst, pair) in out.iter_mut().zip(hex.as_bytes().as_chunks::<2>().0) {
        let hi = nibble(pair[0]).ok_or("invalid sha256 hex digit")?;
        let lo = nibble(pair[1]).ok_or("invalid sha256 hex digit")?;
        *dst = (hi << 4) | lo;
    }
    Ok(out)
}

pub(super) fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

pub(super) fn prompt_sha256(tokens: &[u32]) -> [u8; 32] {
    let hex = super::sha256::digest_reader(U32LeReader { tokens, byte_pos: 0 })
        .expect("in-memory u32 slice read cannot fail");
    hex_to_32(&hex).expect("sha256 digest is 64 lowercase hex chars")
}

pub(super) fn scope_sha256(scope: Option<&str>) -> [u8; 32] {
    let mut data = Vec::with_capacity(1 + scope.map_or(0, str::len));
    data.push(u8::from(scope.is_some()));
    if let Some(s) = scope {
        data.extend_from_slice(s.as_bytes());
    }
    hex_to_32(&super::sha256::digest(&data)).expect("sha256 digest is 64 lowercase hex chars")
}

pub(super) fn manifest_sha256(prelude: &[u8; 64], header: &[u8], chunk_crcs: &[u32]) -> [u8; 32] {
    let reader = prelude
        .as_slice()
        .chain(header)
        .chain(U32LeReader { tokens: chunk_crcs, byte_pos: 0 });
    let hex = super::sha256::digest_reader(reader).expect("in-memory manifest read cannot fail");
    hex_to_32(&hex).expect("sha256 digest is 64 lowercase hex chars")
}

/// Exact payload byte length for a target state at `position` plus optional `DraftDesc`.
pub(super) fn expected_payload_bytes(layouts: &[Layout], position: u64, draft: Option<&DraftDesc>) -> Result<u64, String> {
    let mut words = 0u64;
    for l in layouts {
        let plane = state::prefix_plane_words(l, position) as u64;
        let kv_ring = (l.kv_taps as u64).checked_mul(l.kv_channels as u64).ok_or("kv ring words overflow")?;
        let after_ring = (l.taps as u64).checked_mul(l.channels as u64).ok_or("after ring words overflow")?;
        let visit = plane
            .checked_mul(2)
            .and_then(|n| n.checked_add(kv_ring.checked_mul(2)?))
            .and_then(|n| n.checked_add(after_ring.checked_mul(2)?))
            .ok_or("target payload words overflow")?;
        words = words.checked_add(visit).ok_or("payload words overflow")?;
    }
    if let Some(d) = draft {
        let pos = usize::try_from(position).map_err(|_| "position exceeds host usize")?;
        let win = (d.sliding_window_size as usize).saturating_sub(1);
        for &local in &d.is_local {
            let rows = if local != 0 { pos.min(win) } else { pos } as u64;
            let layer_words = rows
                .checked_mul(u64::from(d.kv_width))
                .and_then(|n| n.checked_mul(2))
                .ok_or("draft layer words overflow")?;
            words = words.checked_add(layer_words).ok_or("payload words overflow")?;
        }
        if d.num_layers > 0 {
            words = words.checked_add(u64::from(d.hidden_size)).ok_or("draft feature words overflow")?;
        }
    }
    words.checked_mul(4).ok_or_else(|| "payload bytes overflow".to_owned())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_header(
    checkpoint_fingerprint: &str,
    scope: Option<&str>,
    vocab_rows: u32,
    position: u64,
    prompt_hash: &[u8; 32],
    visits: &[VisitDesc],
    draft: Option<&DraftDesc>,
    tokens: &[u32],
    next_logits: &[f32],
) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(256 + tokens.len() * 4 + next_logits.len() * 4);
    let push_str = |out: &mut Vec<u8>, s: &str| -> Result<(), String> {
        let len = u32::try_from(s.len()).map_err(|_| "header string overflow")?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        Ok(())
    };
    push_str(&mut out, ENGINE_TAG)?;
    push_str(&mut out, checkpoint_fingerprint)?;
    match scope {
        None => {
            out.push(0);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        Some(s) => {
            out.push(1);
            push_str(&mut out, s)?;
        }
    }
    out.extend_from_slice(&vocab_rows.to_le_bytes());
    out.extend_from_slice(&position.to_le_bytes());
    out.extend_from_slice(prompt_hash);
    let visit_count = u32::try_from(visits.len()).map_err(|_| "visit count overflow")?;
    out.extend_from_slice(&visit_count.to_le_bytes());
    for v in visits {
        out.push(u8::from(v.ring));
        out.extend_from_slice(&v.window.to_le_bytes());
        out.extend_from_slice(&v.ring_rows.to_le_bytes());
        out.extend_from_slice(&v.row_words.to_le_bytes());
        out.extend_from_slice(&v.kv_channels.to_le_bytes());
        out.extend_from_slice(&v.kv_taps.to_le_bytes());
        out.extend_from_slice(&v.channels.to_le_bytes());
        out.extend_from_slice(&v.taps.to_le_bytes());
    }
    match draft {
        None => out.push(0),
        Some(d) => {
            out.push(1);
            out.extend_from_slice(&d.num_layers.to_le_bytes());
            out.extend_from_slice(&d.hidden_size.to_le_bytes());
            out.extend_from_slice(&d.kv_width.to_le_bytes());
            out.extend_from_slice(&d.sliding_window_size.to_le_bytes());
            out.extend_from_slice(&d.block_size.to_le_bytes());
            out.extend_from_slice(&d.is_local);
            let tlen = u32::try_from(d.target_layer_ids.len()).map_err(|_| "draft target layers overflow")?;
            out.extend_from_slice(&tlen.to_le_bytes());
            for &id in &d.target_layer_ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    let logits_len = u32::try_from(next_logits.len()).map_err(|_| "logits length overflow")?;
    out.extend_from_slice(&logits_len.to_le_bytes());
    for &t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    for &v in next_logits {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(out)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("header offset overflow")?;
        let slice = self.buf.get(self.pos..end).ok_or("truncated header")?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).map(str::to_owned).map_err(|_| "invalid UTF-8 in header".into())
    }
}

pub(super) fn decode_header(raw: &[u8], keep_arrays: bool) -> Result<ParsedHeader, String> {
    let mut cur = Cursor { buf: raw, pos: 0 };
    let engine_tag = cur.string()?;
    let checkpoint_fingerprint = cur.string()?;
    let scope_present = cur.u8()?;
    let scope_str = cur.string()?;
    let scope = match scope_present {
        0 if scope_str.is_empty() => None,
        1 => Some(scope_str),
        _ => return Err("invalid scope flag".into()),
    };
    let vocab_rows = cur.u32()?;
    let position = cur.u64()?;
    if position == 0 {
        return Err("header position is zero".into());
    }
    let expected_prompt_sha256: [u8; 32] = cur.take(32)?.try_into().unwrap();
    let visit_count = cur.u32()? as usize;
    let mut visits = Vec::with_capacity(visit_count.min(1024));
    for _ in 0..visit_count {
        let ring_b = cur.u8()?;
        if ring_b > 1 {
            return Err("invalid visit ring flag".into());
        }
        visits.push(VisitDesc {
            ring: ring_b == 1,
            window: cur.u32()?,
            ring_rows: cur.u32()?,
            row_words: cur.u32()?,
            kv_channels: cur.u32()?,
            kv_taps: cur.u32()?,
            channels: cur.u32()?,
            taps: cur.u32()?,
        });
    }
    let has_draft = cur.u8()?;
    let draft = match has_draft {
        0 => None,
        1 => {
            let num_layers = cur.u32()?;
            let hidden_size = cur.u32()?;
            let kv_width = cur.u32()?;
            let sliding_window_size = cur.u32()?;
            let block_size = cur.u32()?;
            let is_local = cur.take(num_layers as usize)?.to_vec();
            if is_local.iter().any(|&b| b > 1) {
                return Err("invalid draft local layer flag".into());
            }
            let tlen = cur.u32()? as usize;
            let mut target_layer_ids = Vec::with_capacity(tlen.min(128));
            for _ in 0..tlen {
                target_layer_ids.push(cur.u32()?);
            }
            Some(DraftDesc { num_layers, hidden_size, kv_width, sliding_window_size, block_size, is_local, target_layer_ids })
        }
        _ => return Err("invalid has_draft flag".into()),
    };
    let logits_len = cur.u32()?;
    let pos_usize = usize::try_from(position).map_err(|_| "header position exceeds host usize")?;
    let tok_bytes_len = pos_usize.checked_mul(4).ok_or("header token bytes overflow")?;
    let tok_bytes = cur.take(tok_bytes_len)?;
    let logit_bytes_len = (logits_len as usize).checked_mul(4).ok_or("header logit bytes overflow")?;
    let logit_bytes = cur.take(logit_bytes_len)?;
    if cur.pos != raw.len() {
        return Err("trailing bytes in header".into());
    }
    let first_token = u32::from_le_bytes(tok_bytes[0..4].try_into().unwrap());
    let last_token = u32::from_le_bytes(tok_bytes[tok_bytes.len() - 4..].try_into().unwrap());
    let prefix_64_hash = {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for chunk in tok_bytes.as_chunks::<4>().0.iter().take(64) {
            h = (h ^ u64::from(u32::from_le_bytes(*chunk))).wrapping_mul(0x0100_0000_01b3);
        }
        h
    };
    let (tokens, next_logits) = if keep_arrays {
        let tokens: Vec<u32> = tok_bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        if prompt_sha256(&tokens) != expected_prompt_sha256 {
            return Err("header prompt_sha256 mismatch".into());
        }
        let next_logits: Vec<f32> = logit_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from_le_bytes(*b)))
            .collect();
        if next_logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logit in header".into());
        }
        (tokens, next_logits)
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(ParsedHeader {
        engine_tag,
        checkpoint_fingerprint,
        scope,
        vocab_rows,
        position,
        prompt_sha256: expected_prompt_sha256,
        visits,
        draft,
        logits_len,
        first_token,
        last_token,
        prefix_64_hash,
        tokens,
        next_logits,
    })
}

/// Read and authenticate Prelude + Metadata Header + Footer (verifying exact file size, magic,
/// footer magic, and whole-manifest SHA-256) without reading the multi-GB tensor payload.
pub(super) fn read_and_verify_envelope(
    file: &mut std::fs::File,
    keep_arrays: bool,
) -> Result<(Prelude, ParsedHeader, Vec<u32>), String> {
    let file_len = file.metadata().map_err(|e| format!("stat failed: {e}"))?.len();
    if file_len < PRELUDE_BYTES + FOOTER_FIXED_BYTES {
        return Err(format!("file too short ({file_len} bytes)"));
    }
    file.seek(SeekFrom::Start(0)).map_err(|e| format!("seek start failed: {e}"))?;
    let mut prelude_raw = [0u8; 64];
    file.read_exact(&mut prelude_raw).map_err(|e| format!("read prelude failed: {e}"))?;
    let prelude = Prelude::decode(&prelude_raw, file_len)?;
    let header_len = usize::try_from(prelude.header_bytes).map_err(|_| "header exceeds host usize")?;
    let mut header_raw = vec![0u8; header_len];
    file.read_exact(&mut header_raw).map_err(|e| format!("read header failed: {e}"))?;
    let footer_offset = PRELUDE_BYTES
        .checked_add(prelude.header_bytes)
        .and_then(|n| n.checked_add(prelude.payload_bytes))
        .ok_or("footer offset overflow")?;
    file.seek(SeekFrom::Start(footer_offset)).map_err(|e| format!("seek footer failed: {e}"))?;
    let footer_len = (prelude.chunk_count as usize)
        .checked_mul(4)
        .and_then(|n| n.checked_add(FOOTER_FIXED_BYTES as usize))
        .ok_or("footer length overflow")?;
    let mut footer_raw = vec![0u8; footer_len];
    file.read_exact(&mut footer_raw).map_err(|e| format!("read footer failed: {e}"))?;
    if &footer_raw[footer_len - 8..] != FOOTER_MAGIC {
        return Err("invalid footer magic".into());
    }
    let expected_sha: [u8; 32] = footer_raw[footer_len - 40..footer_len - 8].try_into().unwrap();
    let chunk_crcs: Vec<u32> = footer_raw[..footer_len - 40]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    if manifest_sha256(&prelude_raw, &header_raw, &chunk_crcs) != expected_sha {
        return Err("manifest sha256 mismatch".into());
    }
    let header = decode_header(&header_raw, keep_arrays)?;
    if (prelude.flags & FLAG_HAS_DRAFT != 0) != header.draft.is_some() {
        return Err("prelude has_draft flag disagrees with header".into());
    }
    Ok((prelude, header, chunk_crcs))
}
