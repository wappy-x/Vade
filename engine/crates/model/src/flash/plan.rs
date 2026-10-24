//! Text-only ownership in original tensor coordinates; payloads pack these ranges.
use super::{Checkpoint, WeightFormat, PLE_PART_ROWS, PLE_ROWS, partition};
use crate::{Result, invalid, safetensors::Tensor};
use std::ops::Range;

/// Official FP8 text payload per rank for the default two-rank plan (no MTP).
/// Excludes allocation alignment; all twelve 640-row index projections replicate.
/// Used by load-check's checkpoint assertion, not serving memory reservations.
pub const TWO_RANK_WEIGHT_BYTES: u64 = 91_709_334_682;
/// Operation classes used by the rank execution plan's opt-in prefill probe.
pub const PREFILL_PROFILE_CLASSES: [&str; 11] = ["dense_qkv_o_gemm", "gdn_recurrence", "qsa_index_attention",
    "moe_route", "moe_gather", "moe_experts", "moe_shared", "norms_hc_mix", "lm_head", "transport_collectives", "other"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    All,
    Rows(Vec<Range<u64>>),
    Columns(Range<u64>),
    Absent,
}
impl Selection {
    pub fn bytes(&self, tensor: &Tensor) -> u64 {
        match self {
            Self::All => tensor.byte_len(),
            Self::Absent => 0,
            Self::Rows(ranges) => {
                ranges.iter().map(|r| r.end - r.start).sum::<u64>() * row_bytes(tensor)
            }
            Self::Columns(range) => {
                tensor.shape[0] * (range.end - range.start) * tensor.dtype.bytes()
            }
        }
    }

    /// Original tensor-relative source offset and length; destination is packed
    /// in this iteration order. Column selections retain their global stride.
    pub fn spans<'a>(&'a self, tensor: &'a Tensor) -> Box<dyn Iterator<Item = (u64, u64)> + 'a> {
        match self {
            Self::All => Box::new(std::iter::once((0, tensor.byte_len()))),
            Self::Absent => Box::new(std::iter::empty()),
            Self::Rows(ranges) => {
                let stride = row_bytes(tensor);
                Box::new(
                    ranges
                        .iter()
                        .filter(|r| !r.is_empty())
                        .map(move |r| (r.start * stride, (r.end - r.start) * stride)),
                )
            }
            Self::Columns(range) => {
                let scalar = tensor.dtype.bytes();
                Box::new(
                    (0..tensor.shape[0])
                        .filter(|_| !range.is_empty())
                        .map(move |row| {
                            (
                                (row * tensor.shape[1] + range.start) * scalar,
                                (range.end - range.start) * scalar,
                            )
                        }),
                )
            }
        }
    }

    /// Translate a packed allocation offset back to its original tensor offset.
    /// FP8 scale lookup must use this global coordinate, including column shards.
    pub fn global_byte_offset(&self, tensor: &Tensor, local_byte: u64) -> Result<u64> {
        if local_byte >= self.bytes(tensor) {
            return Err(invalid("byte offset outside selected tensor"));
        }
        match self {
            Self::All => Ok(local_byte),
            Self::Columns(range) => {
                let width = (range.end - range.start) * tensor.dtype.bytes();
                Ok(
                    (local_byte / width * tensor.shape[1] + range.start) * tensor.dtype.bytes()
                        + local_byte % width,
                )
            }
            Self::Rows(_) => {
                let mut local = local_byte;
                for (offset, length) in self.spans(tensor) {
                    if local < length {
                        return Ok(offset + local);
                    }
                    local -= length;
                }
                Err(invalid("byte offset outside selected rows"))
            }
            Self::Absent => Err(invalid("tensor is absent on this rank")),
        }
    }

    /// Inverse mapping, used to locate a block scale belonging to a global FP8 value.
    pub fn local_byte_offset(&self, tensor: &Tensor, global_byte: u64) -> Option<u64> {
        if global_byte >= tensor.byte_len() {
            return None;
        }
        match self {
            Self::All => Some(global_byte),
            Self::Absent => None,
            Self::Columns(range) => {
                let scalar = tensor.dtype.bytes();
                let row = global_byte / (tensor.shape[1] * scalar);
                let column = global_byte / scalar % tensor.shape[1];
                range.contains(&column).then(|| {
                    (row * (range.end - range.start) + column - range.start) * scalar
                        + global_byte % scalar
                })
            }
            Self::Rows(_) => {
                let mut local = 0;
                for (offset, length) in self.spans(tensor) {
                    if (offset..offset + length).contains(&global_byte) {
                        return Some(local + global_byte - offset);
                    }
                    local += length;
                }
                None
            }
        }
    }
}

fn row_bytes(tensor: &Tensor) -> u64 {
    tensor.shape[1..].iter().product::<u64>() * tensor.dtype.bytes()
}
fn scaled(range: &Range<u64>, scale: u64) -> Range<u64> {
    range.start * scale..range.end * scale
}
fn grouped(range: &Range<u64>, per_group: u64) -> Range<u64> {
    if range.is_empty() {
        0..0
    } else {
        range.start / per_group..range.end.div_ceil(per_group)
    }
}

#[derive(Debug, Clone)]
pub struct RankPlan {
    pub mtp: bool,
    /// Main FP8 down projections may own complete output rows instead of K columns.
    pub expert_output_rows: Option<Range<u64>>,
    pub rank: u64,
    pub ranks: u64,
    pub query_heads: Range<u64>,
    pub kv_heads: Range<u64>,
    pub value_heads: Range<u64>,
    pub key_heads: Range<u64>,
    pub shared_channels: Range<u64>,
    pub vocabulary: Range<u64>,
    pub ple_rows: Range<u64>,
}
impl RankPlan {
    pub fn mixed_selection(&self, name: &str) -> Result<Selection> {
        if self.rank != 0 || self.ranks != 1 { return Err(invalid("mixed Flash storage requires world 1")); }
        Ok(if name.starts_with("model.visual.") || name.starts_with("mtp.") && !self.mtp {
            Selection::Absent
        } else { Selection::All })
    }
    /// Pair packet and workspace capabilities are independent of link topology.
    pub fn is_pair(&self) -> bool { self.ranks == 2 }

    pub fn new(rank: u64, ranks: u64) -> Result<Self> {
        let query_heads = partition(24, rank, ranks)?;
        let value_heads = partition(48, rank, ranks)?;
        Ok(Self {
            mtp: false,
            expert_output_rows: None,
            rank,
            ranks,
            // Keep every KV head used by the local queries, including replicas:
            // at four ranks, queries 0..6 and 6..12 both need KV head 0.
            kv_heads: grouped(&query_heads, 12),
            key_heads: grouped(&value_heads, 3),
            query_heads,
            value_heads,
            // BF16 decode consumes four columns at once; multirow projection
            // stages eight. Keep both the packed row stride and K aligned.
            shared_channels: scaled(&partition(80, rank, ranks)?, 8),
            vocabulary: partition(248320, rank, ranks)?,
            ple_rows: partition(PLE_ROWS, rank, ranks)?,
        })
    }

    /// Divide five 128-wide blocks, rotating remainder ownership by expert ID.
    /// At two ranks even experts are 3/2 blocks and odd experts are 2/3 blocks.
    /// Global channel ranges stay ordered by rank; larger counts may be empty.
    pub fn expert_blocks(&self, expert: u64) -> Range<u64> {
        crate::placement::rotating_blocks(5, self.rank, self.ranks, expert)
    }

    pub fn selection(&self, name: &str) -> Result<Selection> {
        self.selection_for(name, WeightFormat::Fp8)
    }

    /// NVFP4 keeps the FP8 128-channel ownership, expressed in each tensor's
    /// stored coordinates: two values/byte and eight 16-value scales/block.
    pub fn selection_for(&self, name: &str, format: WeightFormat) -> Result<Selection> {
        let rows = |range| Selection::Rows(vec![range]);
        if name.starts_with("model.visual.") || name.starts_with("mtp.") && !self.mtp {
            return Ok(Selection::Absent);
        }
        if name == "lm_head.weight" || name == "model.language_model.embed_tokens.weight" {
            return Ok(rows(self.vocabulary.clone()));
        }
        if let Some((_, suffix)) = name.split_once(".mlp.experts.") {
            let (expert, projection) = suffix
                .split_once('.')
                .ok_or_else(|| invalid("invalid expert tensor name"))?;
            let expert = expert
                .parse::<u64>()
                .map_err(|_| invalid("invalid expert id"))?;
            if expert >= 512 {
                return Err(invalid("expert id exceeds Flash configuration"));
            }
            let (projection, storage) = projection
                .split_once('.')
                .ok_or_else(|| invalid("invalid expert projection"))?;
            let nvfp4 = format == WeightFormat::Nvfp4 && name.starts_with("model.language_model.layers.");
            if projection == "down_proj" && !nvfp4 && name.starts_with("model.language_model.layers.") {
                if let Some(output) = &self.expert_output_rows {
                    return Ok(rows(match storage {
                        "weight" => output.clone(),
                        "weight_scale_inv" => output.start / 128..output.end / 128,
                        _ => return Err(invalid("unknown expert storage")),
                    }));
                }
            }
            let scale = if nvfp4 {
                if !matches!(projection, "gate_proj" | "up_proj" | "down_proj") {
                    return Err(invalid("unknown NVFP4 expert projection"));
                }
                match storage {
                    "weight" => if projection == "down_proj" { 64 } else { 128 },
                    "weight_scale" => if projection == "down_proj" { 8 } else { 128 },
                    "weight_scale_2" | "input_scale" => return Ok(if self.expert_blocks(expert).is_empty() {
                        Selection::Absent
                    } else { Selection::All }),
                    _ => return Err(invalid("unknown NVFP4 expert storage")),
                }
            } else {
                match storage {
                    "weight" => 128,
                    "weight_scale_inv" => 1,
                    _ => return Err(invalid("unknown expert storage")),
                }
            };
            let channels = scaled(&self.expert_blocks(expert), scale);
            return Ok(match projection {
                "gate_proj" | "up_proj" => rows(channels),
                "down_proj" => Selection::Columns(channels),
                _ => return Err(invalid("unknown expert projection")),
            });
        }
        if let Some((_, suffix)) = name.split_once(".ngram_embedding.shard_") {
            let part = suffix
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("invalid PLE part"))?
                .parse::<u64>()
                .map_err(|_| invalid("invalid PLE part"))?;
            if part >= 128 {
                return Err(invalid("PLE part exceeds Flash configuration"));
            }
            let start = part * PLE_PART_ROWS;
            let from = self.ple_rows.start.max(start);
            let to = self.ple_rows.end.min(start + PLE_PART_ROWS);
            return Ok(if from < to {
                rows(from - start..to - start)
            } else {
                Selection::Absent
            });
        }
        if let Some((_, projection)) = name.split_once(".mlp.shared_expert.") {
            return Ok(match projection {
                "gate_proj.weight" | "up_proj.weight" => rows(self.shared_channels.clone()),
                "down_proj.weight" => Selection::Columns(self.shared_channels.clone()),
                _ => return Err(invalid("unknown shared expert projection")),
            });
        }
        if let Some((_, part)) = name.split_once(".self_attn.") {
            return Ok(match part {
                "q_proj.weight" => rows(scaled(&self.query_heads, 512)),
                "k_proj.weight" | "v_proj.weight" => rows(scaled(&self.kv_heads, 256)),
                "o_proj.weight" => Selection::Columns(scaled(&self.query_heads, 256)),
                "q_norm.weight" | "k_norm.weight" => rows(partition(256, self.rank, self.ranks)?),
                // Every output is a complete K reduction. Replicate this small
                // projection to keep index preparation on the ordered stream,
                // removing its per-layer gather without regrouping any sum.
                "indexer.index_qk_proj.weight" => rows(0..640),
                "indexer.q_layernorm.weight" | "indexer.k_layernorm.weight" => {
                    rows(partition(128, self.rank, self.ranks)?)
                }
                _ => return Err(invalid(format!("unknown indexed attention tensor {name}"))),
            });
        }
        if let Some((_, part)) = name.split_once(".linear_attn.") {
            let keys = scaled(&self.key_heads, 128);
            let values = scaled(&self.value_heads, 128);
            return Ok(match part {
                "in_proj_qkv.weight" | "conv1d.weight" => Selection::Rows(vec![
                    keys.clone(),
                    keys.start + 2048..keys.end + 2048,
                    values.start + 4096..values.end + 4096,
                ]),
                "in_proj_z.weight" => rows(values),
                "out_proj.weight" => Selection::Columns(values),
                "A_log" | "dt_bias" | "in_proj_a.weight" | "in_proj_b.weight" => {
                    rows(self.value_heads.clone())
                }
                "norm.weight" => rows(partition(128, self.rank, self.ranks)?),
                _ => return Err(invalid(format!("unknown linear attention tensor {name}"))),
            });
        }
        // Validated HC, routing controls, hash constants and PLE projections.
        Ok(Selection::All)
    }

    pub fn budget(&self, checkpoint: &Checkpoint) -> Result<Budget> {
        let mut budget = Budget::default();
        for (name, tensor) in checkpoint.tensors() {
            let selection = if checkpoint.config.mixed.is_some() { self.mixed_selection(name)? }
                else { self.selection_for(name, checkpoint.config.format)? };
            let bytes = selection.bytes(tensor);
            budget.weights += bytes;
            if name.contains(".mlp.experts.") {
                budget.experts += bytes;
            } else if name.contains(".ngram_embedding.shard_") {
                budget.ple += bytes;
            } else {
                budget.dense += bytes;
            }
        }
        Ok(budget)
    }
}

#[derive(Debug, Default)]
pub struct Budget {
    pub weights: u64,
    pub experts: u64,
    pub ple: u64,
    pub dense: u64,
}
