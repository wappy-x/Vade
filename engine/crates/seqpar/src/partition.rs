//! Contiguous token partitions; later tokens can be assigned smaller slices.
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct Partition {
    offsets: Vec<usize>,
}

impl Partition {
    /// Both the 128-row projection and 32-row attention tiles remain complete.
    pub const ALIGNMENT: usize = 128;

    // GB10 Qwen3-8B isolated compute, 8,192/28,672/38,912 tokens,
    // two through four ranks. Charge each actual chunk's prefix packing.
    const ROW_MS: f64 = 0.38982;
    const ROW_PREFIX_MS: f64 = 0.000009979;
    const SHORT_ROW_PREFIX_MS: f64 = 0.000011193;
    const PREFIX_INIT_MS: f64 = 0.014647;
    // A short tail still streams a chunk's weights through every projection.
    const SHORT_TAIL_ROW_MS: f64 = 0.1512;
    const LONG_TAIL_ROW_MS: f64 = 0.06694;
    const PACK_PREFIX_MS: f64 = 0.0036703;
    const SMALL_CHUNK_ROW_MS: f64 = 0.01419;

    /// Convert the adaptive kernel's per-key work to the endpoint weight of
    /// the quadratic cumulative cost. The ordinary short-prompt path is unchanged.
    pub fn prefill_weight(tokens: usize) -> f64 {
        let slope = if tokens >= 32768 { Self::ROW_PREFIX_MS / Self::ROW_MS }
            else if tokens >= 6144 { 0.0000121767 / 0.384896 } else { 0.000075 };
        slope * tokens as f64
    }

    /// Balance row work, causal attention/packing and prefix initialization.
    /// Long prompts include the local-span chunk-size transition.
    pub fn prefill(tokens: usize, world: usize, attention_weight: f64) -> Result<Self, String> {
        let mut partition = Self::balanced_aligned(tokens, world, attention_weight, Self::ALIGNMENT)?;
        if attention_weight == 0.0 || world == 1 { return Ok(partition); }
        if tokens >= 6144 { return Self::prefill_chunks(tokens, world, attention_weight); }
        let cumulative = |x: usize| {
            let x = x as f64;
            x + attention_weight * x * x / (2.0 * tokens.max(1) as f64)
        };
        let initialization = 0.00896 / 0.3754;
        for _ in 0..world.saturating_mul(4) {
            let mut changed = false;
            for rank in 1..world {
                let (start, end) = (partition.offsets[rank - 1], partition.offsets[rank + 1]);
                let target = cumulative(start) + cumulative(end) + initialization * end as f64;
                let difference = |x: usize| 2.0 * cumulative(x) + initialization * x as f64 - target;
                let (mut low, mut high) = (start / Self::ALIGNMENT + 1, (end - 1) / Self::ALIGNMENT);
                while low < high {
                    let middle = low + (high - low) / 2;
                    if difference(middle * Self::ALIGNMENT) < 0.0 { low = middle + 1; } else { high = middle; }
                }
                let mut boundary = low * Self::ALIGNMENT;
                if boundary > start + Self::ALIGNMENT
                    && difference(boundary - Self::ALIGNMENT).abs() < difference(boundary).abs()
                { boundary -= Self::ALIGNMENT; }
                changed |= boundary != partition.offsets[rank];
                partition.offsets[rank] = boundary;
            }
            if !changed { break; }
        }
        Ok(partition)
    }

    fn prefill_chunks(tokens: usize, world: usize, attention_weight: f64) -> Result<Self, String> {
        let tiles = tokens.div_ceil(Self::ALIGNMENT);
        let position = |tile: usize| (tile * Self::ALIGNMENT).min(tokens);
        let cost = |start: usize, end: usize| {
            let (start, end) = (position(start), position(end));
            let local = end - start;
            if tokens < 32768 {
                // Preserve the calibrated short/medium-context policy.
                let tail = if local < 6144 { (local.div_ceil(512) * 512 - local) as f64 * 0.12 / 0.384896 } else { 0.0 };
                let slope = if local < 6144 { 0.00001900298 / 0.0000121767 } else { 1.0 };
                let rows = local as f64 * if local >= 20480 { 1.0 + 0.036 / 0.384896 } else { 1.0 };
                return rows + tail
                    + attention_weight / tokens as f64 * slope
                        * (end as f64 * end as f64 - start as f64 * start as f64) / 2.0
                    + 0.0133023 / 0.384896 * end as f64;
            }
            let chunk = spark_engine::context::sequence_chunk_size(local, end, 4096, 4096, 1024, 12288, 512);
            let slope = attention_weight / tokens as f64 * if local < 6144 {
                Self::SHORT_ROW_PREFIX_MS / Self::ROW_PREFIX_MS
            } else { 1.0 };
            let rows = local as f64 * if chunk == 1024 {
                1.0 + Self::SMALL_CHUNK_ROW_MS / Self::ROW_MS
            } else { 1.0 };
            let tail = (local.div_ceil(512) * 512 - local) as f64
                * if local < 6144 { Self::SHORT_TAIL_ROW_MS } else { Self::LONG_TAIL_ROW_MS } / Self::ROW_MS;
            let chunks = local.div_ceil(chunk);
            let packed = (chunks - 1) * start + chunk * chunks * (chunks - 1) / 2 + end;
            rows + tail + slope * (end as f64 * end as f64 - start as f64 * start as f64) / 2.0
                + Self::PREFIX_INIT_MS / Self::ROW_MS * end as f64
                + Self::PACK_PREFIX_MS / Self::ROW_MS * packed as f64
        };
        // Crossing 6,144 local rows changes the chunk size and can lower cost.
        // Minimax over tile boundaries handles that discontinuity without a
        // monotonic-cost assumption, including projection-tail padding.
        let mut previous = vec![f64::INFINITY; tiles + 1];
        previous[0] = 0.0;
        let mut boundaries = vec![vec![0usize; tiles + 1]; world];
        for rank in 1..=world {
            let mut next = vec![f64::INFINITY; tiles + 1];
            for end in rank..=tiles - (world - rank) {
                for start in rank - 1..end {
                    let maximum = previous[start].max(cost(start, end));
                    if maximum < next[end] {
                        next[end] = maximum;
                        boundaries[rank - 1][end] = start;
                    }
                }
            }
            previous = next;
        }
        let mut offsets = vec![0; world + 1];
        offsets[world] = tokens;
        let mut end = tiles;
        for rank in (0..world).rev() {
            end = boundaries[rank][end];
            offsets[rank] = position(end);
        }
        Self::from_offsets(offsets)
    }

    pub fn from_offsets(offsets: Vec<usize>) -> Result<Self, String> {
        if offsets.len() < 2 || offsets[0] != 0
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err("partition requires monotonic ranges starting at zero".into());
        }
        Ok(Self { offsets })
    }

    pub fn equal(tokens: usize, world: usize) -> Result<Self, String> {
        Self::balanced(tokens, world, 0.0)
    }

    /// Cost per token at position x is 1 + attention_weight * x / tokens.
    /// The weight is measured last-position attention cost / projection+FFN cost.
    /// Zero gives equal token counts. No measurement is implied by this model.
    pub fn balanced(tokens: usize, world: usize, attention_weight: f64) -> Result<Self, String> {
        if world == 0 || !attention_weight.is_finite() || attention_weight < 0.0 {
            return Err("world must be positive and attention weight finite and nonnegative".into());
        }
        let total = tokens as f64;
        let cost = |x: usize| {
            let x = x as f64;
            x + attention_weight * x * x / (2.0 * total.max(1.0))
        };
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(world.checked_add(1).ok_or("partition rank count overflow")?).map_err(|e| e.to_string())?;
        offsets.push(0);
        for rank in 1..world {
            if attention_weight == 0.0 {
                offsets.push((tokens as u128 * rank as u128 / world as u128) as usize);
                continue;
            }
            let target = cost(tokens) * rank as f64 / world as f64;
            let (mut low, mut high) = (0, tokens);
            while low < high {
                let middle = low + (high - low) / 2;
                if cost(middle) < target { low = middle + 1; } else { high = middle; }
            }
            offsets.push(low);
        }
        offsets.push(tokens);
        Self::from_offsets(offsets)
    }

    /// Keep nonempty slices on projection-chunk boundaries. Only the final
    /// slice may have a partial trailing chunk; short prompts need fewer ranks.
    pub fn balanced_aligned(tokens: usize, world: usize, attention_weight: f64, alignment: usize) -> Result<Self, String> {
        if alignment == 0 || world == 0 || tokens < world {
            return Err("aligned partition needs positive alignment and at least one token per rank".into());
        }
        if world > 1 && tokens / world < alignment {
            return Err("prompt shorter than world * alignment; use fewer ranks".into());
        }
        let mut partition = Self::balanced(tokens, world, attention_weight)?;
        for rank in 1..world {
            let minimum = partition.offsets[rank - 1] + alignment;
            let maximum = ((tokens - 1) / alignment - (world - rank - 1)) * alignment;
            let boundary = partition.offsets[rank].min(maximum);
            let rounded = (boundary / alignment + usize::from(boundary % alignment >= alignment.div_ceil(2))) * alignment;
            partition.offsets[rank] = rounded.clamp(minimum, maximum);
        }
        Ok(partition)
    }

    pub fn offsets(&self) -> &[usize] { &self.offsets }
    pub fn range(&self, rank: usize) -> Range<usize> { self.offsets[rank]..self.offsets[rank + 1] }
    pub fn tokens(&self) -> usize { *self.offsets.last().unwrap() }
    pub fn world(&self) -> usize { self.offsets.len() - 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_balanced_ranges_cover_prompt() {
        let p = Partition::balanced(28_672, 4, 2.0).unwrap();
        assert_eq!(p.range(0).start, 0);
        assert_eq!(p.range(3).end, 28_672);
        assert!(p.range(0).len() > p.range(3).len());
        let cost = |x: usize| x as f64 + (x as f64).powi(2) / 28_672.0;
        let costs = (0..4).map(|r| cost(p.range(r).end) - cost(p.range(r).start)).collect::<Vec<_>>();
        assert!(costs.iter().all(|c| (c - costs[0]).abs() < 4.0));
        assert_eq!(Partition::equal(3, 5).unwrap().offsets, [0, 0, 1, 1, 2, 3]);
    }
}
