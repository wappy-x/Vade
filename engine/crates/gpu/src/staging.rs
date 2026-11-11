//! Shared sizes for the existing pinned upload allocators.
pub const UPLOAD_BYTES: usize = 4 * 1024 * 1024;
pub const FILE_CHUNK: usize = 8 * 1024 * 1024;
pub const FILE_READERS: usize = 4;
pub const FILE_BUFFERS: usize = 2;
pub const WEIGHT_ARENA_BYTES: usize = FILE_CHUNK * FILE_READERS * FILE_BUFFERS;
