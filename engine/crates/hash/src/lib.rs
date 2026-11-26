//! Safe, std-only streaming SHA-256 for checkpoint verification.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../../../tools/export/src/sha256.rs"]
mod scalar;
#[cfg(unix)]
mod file;

/// Maximum stream and aligned-file I/O buffer storage per verification worker.
/// Excludes the compressor's small, fixed-size state and padding storage.
pub const FILE_BUFFER_BYTES: usize = 65536 + 1048576 + 4095;

pub fn digest(data: &[u8]) -> String {
    digest_reader(data).expect("slice reads cannot fail")
}

/// Hash every input byte with bounded scratch space and standard SHA-256.
pub fn digest_reader(reader: impl std::io::Read) -> std::io::Result<String> {
    scalar::digest_reader(reader)
}

/// Digest and actual I/O byte counts, including any bounded buffered tail.
#[derive(Debug)]
pub struct FileDigest {
    pub sha256: String,
    pub direct_bytes: u64,
    pub buffered_bytes: u64,
    pub io_mode: &'static str,
}

/// Hash the opened file without changing its position on Unix. Unix reads are
/// bounded by the captured file length; callers must check file identity before
/// and after hashing to reject concurrent changes. Linux requires direct I/O for
/// large files, allowing at most 64 KiB buffered per file. Other platforms use
/// portable buffered reads and make no page-cache guarantee.
pub fn digest_file(file: &std::fs::File) -> std::io::Result<String> {
    Ok(digest_file_with_stats(file)?.sha256)
}

/// Hash with actual I/O counts using [`digest_file`]'s length and identity contract.
pub fn digest_file_with_stats(file: &std::fs::File) -> std::io::Result<FileDigest> {
    #[cfg(unix)]
    {
        let mut reader = file::Reader::new(file)?;
        let sha256 = digest_reader(&mut reader)?;
        Ok(reader.finish(sha256))
    }
    #[cfg(not(unix))]
    {
        Ok(FileDigest { sha256: digest_reader(file)?, direct_bytes: 0,
            buffered_bytes: file.metadata()?.len(), io_mode: "buffered-portable" })
    }
}

#[cfg(test)]
mod tests;
