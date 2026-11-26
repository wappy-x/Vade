//! Header-only safetensors inspection. Tensor payload is never allocated here.
use crate::{
    Result, invalid,
    json::{self, Value, field},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path},
};

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub header_bytes: usize,
    pub tensors: usize,
    pub rank: usize,
    pub shards: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            header_bytes: 16 * 1024 * 1024,
            tensors: 100_000,
            rank: 16,
            shards: 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    F8E4M3,
    U8,
    I8,
    U16,
    I16,
    F16,
    Bf16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}
impl Dtype {
    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "BOOL" => Self::Bool,
            "F8_E4M3" => Self::F8E4M3,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "F32" => Self::F32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F64" => Self::F64,
            _ => return Err(invalid(format!("unsupported tensor dtype {name}"))),
        })
    }
    pub fn bytes(self) -> u64 {
        match self {
            Self::Bool | Self::F8E4M3 | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte offsets relative to the beginning of the data section; end exclusive.
    pub start: u64,
    pub end: u64,
}
impl Tensor {
    pub fn byte_len(&self) -> u64 {
        self.end - self.start
    }
}

#[derive(Debug)]
pub struct Header {
    pub tensors: BTreeMap<String, Tensor>,
    pub metadata: BTreeMap<String, String>,
    pub header_bytes: u64,
    pub data_start: u64,
    pub file_bytes: u64,
}
impl Header {
    pub fn open(path: &Path, limits: Limits) -> Result<Self> {
        let mut file = File::open(path)?;
        Self::read(&mut file, limits)
    }
    /// Reads only the length prefix and bounded header, after determining file size.
    pub fn read<R: Read + Seek>(reader: &mut R, limits: Limits) -> Result<Self> {
        let file_bytes = reader.seek(SeekFrom::End(0))?;
        if file_bytes < 8 {
            return Err(invalid("truncated safetensors length prefix"));
        }
        reader.seek(SeekFrom::Start(0))?;
        let mut prefix = [0; 8];
        reader.read_exact(&mut prefix)?;
        let header_bytes = u64::from_le_bytes(prefix);
        if header_bytes == 0 || header_bytes > limits.header_bytes as u64 {
            return Err(invalid("safetensors header limit exceeded or empty header"));
        }
        let data_start = header_bytes
            .checked_add(8)
            .ok_or_else(|| invalid("safetensors header offset overflow"))?;
        if data_start > file_bytes {
            return Err(invalid("truncated safetensors header"));
        }
        let length = usize::try_from(header_bytes)
            .map_err(|_| invalid("header does not fit address space"))?;
        let mut bytes = vec![0; length];
        reader.read_exact(&mut bytes)?;
        if bytes.first() != Some(&b'{') {
            return Err(invalid("safetensors header must start with '{'"));
        }
        let document = json::parse(
            &bytes,
            json::Limits {
                bytes: limits.header_bytes,
                nodes: limits.tensors.saturating_mul(limits.rank.saturating_add(8)).max(json::Limits::default().nodes),
                ..Default::default()
            },
        )?;
        let object = document.object()?;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                for (key, value) in value.object()? {
                    metadata.insert(key.clone(), value.string()?.to_owned());
                }
                continue;
            }
            if name.is_empty() || tensors.len() >= limits.tensors {
                return Err(invalid("empty tensor name or tensor count limit exceeded"));
            }
            let properties = value.object()?;
            if properties.len() != 3 {
                return Err(invalid(format!(
                    "tensor {name} has unknown or missing properties"
                )));
            }
            let dtype = Dtype::parse(field(properties, "dtype")?.string()?)?;
            let dimensions = field(properties, "shape")?.array()?;
            if dimensions.len() > limits.rank {
                return Err(invalid("tensor rank limit exceeded"));
            }
            let shape = dimensions
                .iter()
                .map(Value::u64)
                .collect::<Result<Vec<_>>>()?;
            // Reject dimension-product overflow even for a later zero dimension.
            let elements = shape.iter().try_fold(1u64, |n, &d| {
                n.checked_mul(d)
                    .ok_or_else(|| invalid("tensor element count overflow"))
            })?;
            let expected_bytes = elements
                .checked_mul(dtype.bytes())
                .ok_or_else(|| invalid("tensor byte count overflow"))?;
            let offsets = field(properties, "data_offsets")?.array()?;
            if offsets.len() != 2 {
                return Err(invalid("tensor needs exactly two data offsets"));
            }
            let start = offsets[0].u64()?;
            let end = offsets[1].u64()?;
            if end < start || end > file_bytes - data_start {
                return Err(invalid(format!(
                    "tensor {name} offset outside file or reversed"
                )));
            }
            if end - start != expected_bytes {
                return Err(invalid(format!(
                    "tensor {name} shape/dtype byte count differs from data offsets"
                )));
            }
            tensors.insert(
                name.clone(),
                Tensor {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        let mut intervals = tensors
            .values()
            .map(|t| (t.start, t.end))
            .collect::<Vec<_>>();
        intervals.sort_unstable();
        let mut cursor = 0;
        for (start, end) in intervals {
            if start != cursor {
                return Err(invalid(
                    "tensor data contains overlap, gap, or misplaced empty tensor",
                ));
            }
            cursor = end;
        }
        if cursor != file_bytes - data_start {
            return Err(invalid("unclaimed trailing tensor data"));
        }
        Ok(Self {
            tensors,
            metadata,
            header_bytes,
            data_start,
            file_bytes,
        })
    }
}

/// Retains the file descriptor used to validate metadata. Reads are explicitly
/// bounded by the caller's buffer and one validated tensor range. The caller
/// must keep checkpoint files immutable; metadata validation is not a hash.
pub struct TensorFile {
    file: File,
    header: Header,
    /// The same file opened for direct reads that bypass the page cache (Linux
    /// O_DIRECT), or None where the file system refuses them.
    direct: Option<File>,
}
/// Reads of at least this many bytes go to `direct`; smaller ones use the page cache.
const DIRECT_READ_MIN: usize = 1 << 20;
impl TensorFile {
    pub fn open(path: &Path, limits: Limits) -> Result<Self> {
        Self::from_file(File::open(path)?, limits)
    }
    pub fn from_file(mut file: File, limits: Limits) -> Result<Self> {
        if !file.metadata()?.is_file() {
            return Err(invalid("tensor source must be a regular file"));
        }
        let header = Header::read(&mut file, limits)?;
        let direct = direct_reader(&file);
        Ok(Self { file, header, direct })
    }
    pub fn header(&self) -> &Header {
        &self.header
    }
    /// Retained descriptor for direct I/O into caller-owned GPU-visible memory.
    pub fn file(&self) -> &File {
        &self.file
    }
    pub fn read_range(&self, name: &str, byte_offset: u64, output: &mut [u8]) -> Result<()> {
        let tensor = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| invalid("tensor is absent from file"))?;
        let end = byte_offset
            .checked_add(output.len() as u64)
            .ok_or_else(|| invalid("tensor read length overflow"))?;
        if end > tensor.byte_len() {
            return Err(invalid("read extends beyond tensor range"));
        }
        let absolute = self
            .header
            .data_start
            .checked_add(tensor.start)
            .and_then(|n| n.checked_add(byte_offset))
            .ok_or_else(|| invalid("tensor read file offset overflow"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            // The same bytes either way: a failed direct read is read again through the page cache.
            if output.len() >= DIRECT_READ_MIN
                && let Some(direct) = &self.direct
                && read_aligned(direct, absolute, output).is_ok()
            {
                return Ok(());
            }
            self.file.read_exact_at(output, absolute)?;
        }
        #[cfg(not(unix))]
        {
            let mut file = self.file.try_clone()?;
            file.seek(SeekFrom::Start(absolute))?;
            file.read_exact(output)?;
        }
        Ok(())
    }
}

/// The file again, opened for direct reads (O_DIRECT through /proc/self/fd);
/// None where the file system refuses them.
#[cfg(target_os = "linux")]
fn direct_reader(file: &File) -> Option<File> {
    use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
    #[cfg(target_arch = "aarch64")]
    const O_DIRECT: i32 = 0x10000;
    #[cfg(not(target_arch = "aarch64"))]
    const O_DIRECT: i32 = 0x4000;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .ok()
}
#[cfg(not(target_os = "linux"))]
fn direct_reader(_file: &File) -> Option<File> {
    None
}

/// Reads `output.len()` bytes at `absolute` with block-aligned file offsets,
/// lengths and buffer, as direct reads need: each piece lands in this thread's
/// aligned scratch and only the asked bytes are copied out. A short read past
/// the end of the file is fine while the asked bytes are covered.
#[cfg(unix)]
fn read_aligned(file: &File, absolute: u64, output: &mut [u8]) -> std::io::Result<()> {
    use std::{cell::RefCell, io::ErrorKind, os::unix::fs::FileExt};
    const BLOCK: usize = 4096;
    const PIECE: usize = 8 << 20;
    thread_local! {static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };}
    SCRATCH.with_borrow_mut(|scratch| {
        scratch.resize(PIECE + 2 * BLOCK, 0);
        let base = scratch.as_ptr().align_offset(BLOCK);
        let window = scratch
            .get_mut(base..base.saturating_add(PIECE + BLOCK))
            .ok_or(ErrorKind::OutOfMemory)?;
        let mut done = 0;
        while done < output.len() {
            let at = absolute + done as u64;
            let skip = (at % BLOCK as u64) as usize;
            let want = (output.len() - done).min(PIECE);
            let span = (skip + want).div_ceil(BLOCK) * BLOCK;
            let got = match file.read_at(&mut window[..span], at - skip as u64) {
                Ok(got) => got,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            if got <= skip {
                return Err(ErrorKind::UnexpectedEof.into());
            }
            let take = (got - skip).min(want);
            output[done..done + take].copy_from_slice(&window[skip..skip + take]);
            done += take;
        }
        Ok(())
    })
}

/// Refuses nonregular/oversized files before allocation and detects growth beyond
/// the limit. On serving targets the open itself is nonblocking: inspecting a
/// FIFO must not wait for a writer, including if it replaces a checked path.
pub fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(invalid("metadata must be a regular file"));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(any(
        all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")),
        target_os = "macos",
    ))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Audited O_NONBLOCK values also used by the launcher. Rust's open
        // supplies O_CLOEXEC; regular files are unaffected by O_NONBLOCK.
        #[cfg(target_os = "linux")]
        const O_NONBLOCK: i32 = 0x800;
        #[cfg(target_os = "macos")]
        const O_NONBLOCK: i32 = 0x4;
        options.custom_flags(O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(invalid("metadata must be a regular file"));
    }
    if metadata.len() > limit as u64 {
        return Err(invalid("metadata file byte limit exceeded"));
    }
    let mut bytes = Vec::new();
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(invalid("metadata file grew beyond byte limit"));
    }
    Ok(bytes)
}

#[derive(Debug)]
pub struct Index {
    pub total_size: u64,
    pub weight_map: BTreeMap<String, String>,
    pub(crate) size_declared: bool,
}
fn exact_index_size(value: &Value) -> Result<u64> {
    if let Ok(size) = value.u64() { return Ok(size); }
    let Value::Number(text) = value else { return Err(invalid("expected JSON number")); };
    let bad = || invalid("index total_size must be an exact nonnegative integer");
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() { return Ok(0); }
    if negative { return Err(bad()); }
    let significant = digits.trim_end_matches('0');
    let exponent = exponent.parse::<i64>().map_err(|_| bad())?;
    let power = exponent.checked_sub(fraction.len() as i64)
        .and_then(|n| n.checked_add((digits.len() - significant.len()) as i64)).ok_or_else(bad)?;
    // Decimal/scientific forms retain their exact-integer ceiling. Integer
    // literals above that ceiling already returned through Value::u64.
    if !(0..=16).contains(&power) || significant.len() + power as usize > 16 { return Err(bad()); }
    let size = significant.parse::<u64>().map_err(|_| bad())?
        .checked_mul(10u64.pow(power as u32)).ok_or_else(bad)?;
    if size > 1u64 << 53 { return Err(bad()); }
    Ok(size)
}
impl Index {
    pub fn open(path: &Path, limits: Limits) -> Result<Self> {
        Self::parse(&read_bounded(path, limits.header_bytes)?, limits)
    }
    pub fn parse(bytes: &[u8], limits: Limits) -> Result<Self> {
        Self::parse_names(bytes, limits, validate_shard_name)
    }
    /// Nested shard names are reserved for ModelOpt mixed NVFP4 checkpoints.
    pub fn parse_mixed(bytes: &[u8], limits: Limits) -> Result<Self> {
        Self::parse_names(bytes, limits, validate_mixed_shard_name)
    }
    fn parse_names(bytes: &[u8], limits: Limits, validate: fn(&str) -> Result<()>) -> Result<Self> {
        let document = json::parse(
            bytes,
            json::Limits {
                bytes: limits.header_bytes,
                nodes: limits.tensors.saturating_add(16).max(json::Limits::default().nodes),
                ..Default::default()
            },
        )?;
        let object = document.object()?;
        if object.len() != 2 {
            return Err(invalid("index requires only metadata and weight_map"));
        }
        let metadata = field(object, "metadata")?.object()?;
        // Some official indexes serialize integer byte counts with a .0 suffix.
        let total_size = metadata.get("total_size").map(exact_index_size).transpose()?.unwrap_or(0);
        let map = field(object, "weight_map")?.object()?;
        if map.is_empty() || map.len() > limits.tensors {
            return Err(invalid("empty or excessive index tensor count"));
        }
        let mut weight_map = BTreeMap::new();
        let mut shards = BTreeSet::new();
        for (name, shard) in map {
            if name.is_empty() {
                return Err(invalid("empty indexed tensor name"));
            }
            let shard = shard.string()?;
            validate(shard)?;
            shards.insert(shard.to_owned());
            if shards.len() > limits.shards {
                return Err(invalid("shard count limit exceeded"));
            }
            weight_map.insert(name.clone(), shard.to_owned());
        }
        Ok(Self {
            total_size,
            weight_map,
            size_declared: metadata.contains_key("total_size"),
        })
    }
    /// Reconcile every index entry with every on-disk tensor, not just expected names.
    pub fn inspect_shards(&self, root: &Path, limits: Limits) -> Result<BTreeMap<String, Header>> {
        let root = root.canonicalize()?;
        let names = self.weight_map.values().collect::<BTreeSet<_>>();
        let mut paths = Vec::with_capacity(names.len());
        for name in &names {
            let path = root.join(name).canonicalize()?;
            if !path.starts_with(&root) {
                return Err(invalid("shard symlink escapes checkpoint directory"));
            }
            paths.push(path);
        }
        // Headers are independent reads and parses (about 1 s serial for 21 shards);
        // checks below still run in shard-name order.
        let headers = std::thread::scope(|scope| {
            let parsing = paths.iter().map(|path| scope.spawn(move || Header::open(path, limits)))
                .collect::<Vec<_>>();
            parsing.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect::<Vec<_>>()
        });
        let mut shards = BTreeMap::new();
        let mut observed = BTreeMap::new();
        let mut total = 0u64;
        for (name, header) in names.into_iter().zip(headers) {
            let header = header?;
            for (tensor, descriptor) in &header.tensors {
                if self.weight_map.get(tensor) != Some(name) {
                    return Err(invalid(
                        "shard tensor is absent from index or mapped to a different shard",
                    ));
                }
                if observed.insert(tensor.clone(), name.clone()).is_some() {
                    return Err(invalid("tensor occurs in multiple shards"));
                }
                total = total
                    .checked_add(descriptor.byte_len())
                    .ok_or_else(|| invalid("checkpoint total size overflow"))?;
            }
            shards.insert(name.clone(), header);
        }
        if observed != self.weight_map {
            return Err(invalid(
                "index weight map differs from actual shard tensors",
            ));
        }
        if self.size_declared && total != self.total_size {
            return Err(invalid("index total_size differs from actual tensor bytes"));
        }
        Ok(shards)
    }
}

pub(crate) fn validate_relative_name(name: &str) -> Result<()> {
    if name.len() > 4096
        || name.split('/').count() > 32
        || name.split('/').any(|part| {
            part.is_empty()
                || matches!(part, "." | "..")
                || !part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
    {
        return Err(invalid("unsafe relative checkpoint filename"));
    }
    Ok(())
}

pub(crate) fn validate_shard_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || !name.ends_with(".safetensors")
        || name.contains(['/', '\\', '\0'])
    {
        return Err(invalid(
            "shard must be a single relative .safetensors filename",
        ));
    }
    Ok(())
}

pub(crate) fn validate_mixed_shard_name(name: &str) -> Result<()> {
    validate_relative_name(name)?;
    if !name.ends_with(".safetensors") {
        return Err(invalid("shard must end in .safetensors"));
    }
    Ok(())
}
/// Metadata-only maps may name absent payloads, but existing ancestors must stay
/// inside the model root. Actual shard inspection requires the complete path.
pub(crate) fn validate_model_path(root: &Path, name: &str) -> Result<()> {
    validate_relative_name(name)?;
    let root = root.canonicalize()?;
    let mut path = root.join(name);
    loop {
        match std::fs::symlink_metadata(&path) {
            Ok(_) => return if path.canonicalize()?.starts_with(&root) { Ok(()) }
                else { Err(invalid("checkpoint path escapes model root")) },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => { path.pop(); }
            Err(e) => return Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    #[cfg(unix)]
    fn aligned_reads_return_the_bytes_of_ordinary_reads() {
        use std::{io::Write, os::unix::fs::FileExt};
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("vade-aligned-read-{}-{nanos}", std::process::id()));
        let bytes = (0..(9usize << 20) + 4321).map(|i| (i * 131 % 251) as u8).collect::<Vec<_>>();
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        let file = File::open(&path).unwrap();
        let len = bytes.len() as u64;
        for (offset, count) in [(0, 1), (0, 4096), (1, 4095), (4095, 2), (12345, 1 << 20), (4096, 8 << 20),
            (777, (8 << 20) + 9000), (len - 5000, 5000), (len - 1, 1), (0, bytes.len())] {
            let mut aligned = vec![0u8; count];
            read_aligned(&file, offset, &mut aligned).unwrap();
            let mut plain = vec![0u8; count];
            file.read_exact_at(&mut plain, offset).unwrap();
            assert_eq!(aligned, plain, "offset {offset} count {count}");
        }
        let mut past = vec![0u8; 2];
        assert!(read_aligned(&file, len - 1, &mut past).is_err());
        std::fs::remove_file(&path).unwrap();
    }
    fn fixture(header: &str, payload: usize) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.resize(bytes.len() + payload, 0);
        bytes
    }
    fn read(header: &str, payload: usize) -> Result<Header> {
        Header::read(
            &mut Cursor::new(fixture(header, payload)),
            Limits::default(),
        )
    }
    #[test]
    fn ordinary_scalar_empty_and_metadata() {
        let header = read(r#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[12,16]},"z":{"dtype":"U8","shape":[0],"data_offsets":[16,16]},"__metadata__":{"format":"pt"}}"#, 16).unwrap();
        assert_eq!(header.tensors["a"].byte_len(), 12);
        assert_eq!(header.metadata["format"], "pt");
    }
    #[test]
    fn bounds_checked_before_allocation() {
        let bytes = u64::MAX.to_le_bytes().to_vec();
        assert!(Header::read(&mut Cursor::new(bytes), Limits::default()).is_err());
        for n in 0..8 {
            assert!(Header::read(&mut Cursor::new(vec![0; n]), Limits::default()).is_err());
        }
        let mut bytes = 100u64.to_le_bytes().to_vec();
        bytes.extend(b"{}");
        assert!(Header::read(&mut Cursor::new(bytes), Limits::default()).is_err());
    }
    #[test]
    fn invalid_layouts() {
        for header in [
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,3]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,6]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[1,5]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[4,0]}}"#,
            r#"{"a":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F64","shape":[18446744073709551615],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F8_UNKNOWN","shape":[4],"data_offsets":[0,4]}}"#,
            r#"{"__metadata__":{"format":3}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4],"extra":1}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
        ] {
            assert!(read(header, 4).is_err(), "accepted {header}");
        }
        assert!(read("{}", 1).is_err());
    }
    #[test]
    fn index_paths_and_totals() {
        for name in [
            "../a.safetensors",
            "/a.safetensors",
            "dir/../a.safetensors",
            "dir//a.safetensors",
            "./a.safetensors",
            "dir/./a.safetensors",
            "nested/a.safetensors",
            "dir\\a.safetensors",
            "a.bin",
        ] {
            assert!(validate_shard_name(name).is_err());
            let bytes = format!(r#"{{"metadata":{{}},"weight_map":{{"w":{name:?}}}}}"#);
            assert!(Index::parse(bytes.as_bytes(), Limits::default()).is_err());
            if name != "nested/a.safetensors" {
                assert!(Index::parse_mixed(bytes.as_bytes(), Limits::default()).is_err());
            }
        }
        assert!(validate_shard_name("model-00001-of-00005.safetensors").is_ok());
        let nested = br#"{"metadata":{},"weight_map":{"w":"tensors/sub-dir_1/model-00001-of-00021.safetensors"}}"#;
        assert!(Index::parse(nested, Limits::default()).is_err());
        assert!(Index::parse_mixed(nested, Limits::default()).is_ok());
        assert!(Index::parse(br#"{"metadata":{"total_size":18446744073709551616},"weight_map":{"a":"a.safetensors"}}"#, Limits::default()).is_err());
    }
}
