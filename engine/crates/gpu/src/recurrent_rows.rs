//! Checked shape and launch arguments for one recurrent prompt layer.
pub const DIM: usize = 128;
pub const TAPS: usize = 4;
pub const MAX_TOKENS: u32 = 8;
pub const THREADS: u32 = DIM as u32;
pub const RECORD_KIND: u32 = 81;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Shape {
    pub tokens: u32,
    pub heads: u32,
    pub bound: f32,
    pub epsilon: f32,
}

impl Shape {
    /// The short-row shapes with the same fast Prepare, Conv and GatedNorm controls.
    pub fn admitted(self) -> bool {
        (2..=MAX_TOKENS).contains(&self.tokens) && matches!(self.heads, 16 | 32 | 64)
            && self.bound.is_finite() && self.bound < 0.0
            && self.epsilon.is_finite() && self.epsilon > 0.0
    }

    /// Input, weight and output word counts. Prepared words and decays are kept only for prefix acceptance.
    pub fn lengths(self, keep: bool) -> Option<([usize; 7], [usize; 3], [usize; 5])> {
        if !self.admitted() { return None; }
        let heads = self.heads as usize;
        let tokens = self.tokens as usize;
        let width = heads * DIM;
        let rows = tokens * width;
        let history = 3 * (TAPS - 1) * width;
        let state = width * DIM;
        let prepared = tokens * heads * (4 * DIM + 1);
        let saved = usize::from(keep);
        Some(([rows, rows, rows, history, rows, tokens * heads, state],
              [3 * TAPS * width, heads + width, DIM],
              [history, state, rows, saved * prepared, saved * rows]))
    }
}

/// All addresses are for f32 words. Inputs are Q, K, V, history, forget, beta and state.
/// The separate output gate has the same extent as Q. Weights are Conv, Prepare and GatedNorm.
/// Outputs are next history, next state, gated rows, prepared and decay. The last two are both
/// zero addresses when the caller does not need them.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Arguments {
    pub shape: Shape,
    pub inputs: [u64; 7],
    pub gate: u64,
    pub weights: [u64; 3],
    pub outputs: [u64; 5],
}

#[cfg(not(target_arch = "nvptx64"))]
impl Arguments {
    /// Serialize the fields for the aligned command payload without Rust padding.
    pub fn payload(self) -> [u8; 144] {
        let mut bytes = [0; 144];
        let shape = [self.shape.tokens, self.shape.heads, self.shape.bound.to_bits(), self.shape.epsilon.to_bits()];
        for (dst, word) in bytes[..16].as_chunks_mut::<4>().0.iter_mut().zip(shape) {
            dst.copy_from_slice(&word.to_ne_bytes());
        }
        let addresses = self.inputs.into_iter().chain([self.gate]).chain(self.weights).chain(self.outputs);
        for (dst, word) in bytes[16..].as_chunks_mut::<8>().0.iter_mut().zip(addresses) {
            dst.copy_from_slice(&word.to_ne_bytes());
        }
        bytes
    }
}

const _: () = {
    assert!(core::mem::size_of::<Arguments>() == 144);
    assert!(core::mem::offset_of!(Arguments, inputs) == 16);
    assert!(core::mem::offset_of!(Arguments, gate) == 72);
    assert!(core::mem::offset_of!(Arguments, weights) == 80);
    assert!(core::mem::offset_of!(Arguments, outputs) == 104);
};
