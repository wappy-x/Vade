//! Opt-in NVFP4 weights with BF16 operands. Checkpoint storage is never expanded.
//! This W4A16 path does not implement the checkpoint's W4A4 activation quantizer.

/// Row-major E2M1: low nibble is the even column. Scales are row-major E4M3,
/// one byte per sixteen columns; global_scale points to one checkpoint F32.
/// A weight is BF16-RNE((E2M1 * E4M3) * global_scale).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Matrix {
    pub weights: u64,
    pub scales: u64,
    pub global_scale: u64,
}

/// One rank's expert channels, in checkpoint order. Width zero owns no channels.
/// Gate/up are [width,2560], down [2560,width]; widths are multiples of sixteen.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Expert {
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
    pub width: u32,
    pub reserved: u32,
}

/// Packed BF16 A fragments use the existing flash_bf16_input layout.
/// Output is row-major [tokens,rows] F32; rows%8=0, cols%16=0.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProjectParams {
    pub matrix: Matrix,
    pub input: u64,
    pub output: u64,
    pub tokens: u32,
    pub rows: u32,
    pub cols: u32,
    pub reserved: u32,
}
