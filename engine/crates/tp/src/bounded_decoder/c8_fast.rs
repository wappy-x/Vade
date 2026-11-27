//! Batched multi-chat decode request descriptors (`KvFinishDirect`, `ShortConvDirect`) for `Compute`.
use spark_gpu_probe::causal_ops::Shape;

/// Operands of `Compute::kv_finish_direct`: multi-chat decode (`2..=16` chats, 1 row/chat, `head_dim == 128`,
/// `taps` in `2..=4`) fusing Key (`ShortConv` + `HeadClip` + `PackBf16`) and Value (`ShortConv` + `RoundedNorm` +
/// `PackBf16`) with direct per-chat KV plane row writes and `every` ring output in one `FAST_NORM` launch.
pub struct KvFinishDirect<'a, V> {
    pub rows: usize,
    pub kv_heads: usize,
    pub taps: usize,
    pub plane_rows: usize,
    pub key_bound: f32,
    pub val_eps: f32,
    pub key_scale: f32,
    pub val_scale: f32,
    pub key_in: &'a V,
    pub key_fixed: &'a [f32],
    pub key_made: Option<&'a V>,
    pub val_in: &'a V,
    pub val_fixed: &'a [f32],
    pub val_made: Option<&'a V>,
}

/// Per-chat rings and mutable KV planes for `Compute::kv_finish_direct`.
pub type KvDirectTarget<'a, V> = (&'a V, &'a V, usize, &'a mut V, &'a mut V, usize);

/// Operands of `Compute::short_conv_direct`: multi-chat decode (`2..=16` chats, 1 row/chat, `taps` in `2..=4`)
/// reading each chat's ring directly without a `row_rings` join copy, writing output and per-chat next rings.
pub struct ShortConvDirect<'a, V> {
    pub shape: Shape,
    pub rows: usize,
    pub inputs: &'a V,
    pub token: Option<&'a V>,
    pub addend: Option<&'a V>,
    pub fixed: &'a [f32],
    pub chat_rings: &'a [(&'a V, usize)],
}
