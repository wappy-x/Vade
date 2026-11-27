//! Gemma's global layers on a kept prompt. The wide prefill (`Backend::prefill_attention_wide`) must give a row the same
//! attention output wherever its call starts: at 0 (a cold prompt) or at the end of a kept prompt of any length.
//! The host tests run a twin of its softmax and values passes on the bounds and word addresses the kernels use
//! (`wide_weights`). The last test runs the kernels themselves and needs a Spark.
use spark_gpu_probe::{bf16::rounded, wide_weights::{block_end, read_end, word, write_end, BLOCK}};

const HEADS: usize = 2;
fn mix(a: usize, b: usize, c: usize) -> u32 {
    let mut x = (a as u64 * 0x9e37_79b9 + b as u64 * 0x85eb_ca6b + c as u64 * 0xc2b2_ae35 + 1).wrapping_mul(0x2545_f491_4f6c_dd1d);
    x ^= x >> 29; (x.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40) as u32
}
fn half(v: f32) -> u16 { (rounded(v).to_bits() >> 16) as u16 }
fn full(w: u16) -> f32 { f32::from_bits((w as u32) << 16) }
// A token's Q, K and V do not depend on the call that reads it: scores and values are functions of absolute positions.
fn score(row: usize, head: usize, key: usize) -> f32 { full(half(mix(row, head, key) as f32 / (1 << 21) as f32 - 4.0)) }
fn value(head: usize, key: usize) -> f32 { full(half(mix(key, head, 77) as f32 / (1 << 23) as f32 + 0.5)) }
/// The BF16 weights of one absolute row over its own keys: the full-row softmax of `attention_bounded_softmax`, form 2.
fn weights(row: usize, head: usize) -> Vec<u16> {
    let peak = (0..=row).map(|key| score(row, head, key)).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = (0..=row).map(|key| (score(row, head, key) - peak).exp()).sum();
    (0..=row).map(|key| half((score(row, head, key) - peak).exp() / sum)).collect()
}
/// What a row's output must be: its own keys only, in key order.
fn exact(row: usize, head: usize) -> f32 {
    weights(row, head).iter().enumerate().fold(0.0, |sum, (key, &w)| sum + full(w) * value(head, key))
}
/// Host twin of `Backend::bounded_attention_device` on the wide global path: tiles of 256 rows, each a softmax pass that
/// writes the weight tile and a values pass that reads it back with no causal mask. `tile` is the weight scratch as earlier
/// work left it. Returns each row's output and the count of weight words read that this call's softmax did not write.
fn call(position: usize, rows: usize, tile: &mut [u16]) -> (Vec<f32>, usize) {
    let capacity = (position + rows).div_ceil(64) * 64;
    assert!(tile.len() >= 256 * HEADS * capacity);
    let (mut out, mut unwritten) = (vec![0.0; rows * HEADS], 0);
    for start in (0..rows).step_by(256) {
        let (position, n) = (position + start, (rows - start).min(256));
        let mut fresh = vec![false; tile.len()];
        for row in 0..n { for head in 0..HEADS {
            let own = weights(position + row, head);
            for key in 0..write_end(position, n, row) {
                let at = word(row, head, HEADS, capacity, key);
                tile[at] = if key < own.len() {own[key]} else {0}; fresh[at] = true;
            }
        }}
        for block in (0..n).step_by(BLOCK) {
            let end = block_end(position, n, block);
            for row in block..(block + BLOCK).min(n) { for head in 0..HEADS {
                let mut sum = 0.0f32;
                for key in 0..read_end(position, n, block) {
                    let at = word(row, head, HEADS, capacity, key);
                    unwritten += usize::from(!fresh[at]);
                    // The V row of every key below `end` is loaded, also the keys after this row.
                    sum += full(tile[at]) * if key < end {value(head, key)} else {0.0};
                }
                out[(start + row) * HEADS + head] = sum;
            }}
        }
    }
    (out, unwritten)
}
fn scratch(kept: usize, more: usize, fill: u16) -> Vec<u16> { vec![fill; 256 * HEADS * (kept + more).div_ceil(64) * 64] }
// (kept prompt, continuation). The first is the shortest that takes the wide path off a tile edge.
const CASES: [(usize, usize); 9] = [(1, 256), (63, 256), (65, 256), (100, 300), (288, 512), (1000, 256), (0, 256), (64, 300), (256, 512)];

#[test]
fn a_kept_prompt_of_any_length_gives_each_row_its_own_keys_only() {
    for (kept, more) in CASES {
        // Nothing clears the scratch: here every word starts as the weight 1.0 that some earlier layer left.
        let (out, unwritten) = call(kept, more, &mut scratch(kept, more, 0x3f80));
        for row in 0..more { for head in 0..HEADS {
            let (got, want) = (out[row * HEADS + head], exact(kept + row, head));
            assert_eq!(got.to_bits(), want.to_bits(),
                "kept prompt of {kept} tokens, continuation of {more}: token {} head {head} is {got}, must be {want}", kept + row);
        }}
        assert_eq!(unwritten, 0, "kept prompt of {kept} tokens, continuation of {more}: weight words read but not written by this call");
    }
}
#[test]
fn the_read_after_a_kept_prompt_equals_the_cold_read() {
    for (kept, more) in CASES {
        let mut tile = scratch(kept, more, 0);
        // One scratch, as in a session: the cold read of the whole prompt leaves its weights for the next call.
        let cold = call(0, kept + more, &mut tile).0;
        let warm = call(kept, more, &mut tile).0;
        for (at, (w, c)) in warm.iter().zip(&cold[kept * HEADS..]).enumerate() {
            assert_eq!(w.to_bits(), c.to_bits(),
                "kept prompt of {kept} tokens, continuation of {more}: token {} head {} is {w} after the kept prompt, {c} cold", kept + at / HEADS, at % HEADS);
        }
    }
}
#[test]
fn a_call_on_a_tile_edge_writes_the_words_it_wrote_before() {
    // The softmax bound before the kept-prompt fix: the row's own next multiple of 64 keys.
    let before = |position: usize, row: usize| (position + row + 1).div_ceil(64) * 64;
    for position in 0..=1300 { for rows in [1, 2, 31, 63, 64, 65, 100, 127, 128, 200, 255, 256] { for row in 0..rows {
        let (now, read) = (write_end(position, rows, row), read_end(position, rows, row / BLOCK * BLOCK));
        // Never fewer words than before, never past what the values pass reads, never past the tile.
        assert!(before(position, row) <= now && now <= read && read <= (position + rows).div_ceil(64) * 64, "position {position} rows {rows} row {row}");
        // A cold prompt, and a kept prompt of a multiple of 64 tokens: the same words, so the same answer.
        if position.is_multiple_of(64) { assert_eq!(now, before(position, row), "position {position} rows {rows} row {row}"); }
    }}}
}
#[test]
fn weight_words_are_distinct_and_inside_the_tile() {
    let (heads, capacity) = (3, 320);
    let mut seen = vec![false; 256 * heads * capacity];
    for row in 0..256 { for head in 0..heads { for key in 0..capacity {
        let at = word(row, head, heads, capacity, key);
        assert!(!std::mem::replace(&mut seen[at], true), "row {row} head {head} key {key}");
    }}}
}

/// The kernels themselves, on both head widths the wide path takes: the global layers of Gemma 4 31B (4 kv heads of 512), then
/// 16 kv heads of 256. Each case reads `kept + more` tokens cold, then the last `more` tokens again at position `kept`, on the
/// scratch the cold read left. Off a tile edge the kept lengths cover both ends (1 and 63 past a multiple of 64) and the
/// middle. Prints one line a case, then asserts.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires CUDA"]
fn device_read_after_a_kept_prompt_equals_the_cold_read() {
    use spark_gpu_probe::driver::with_backend;
    const DEVICE: [(usize, usize); 8] = [(288, 512), (1, 256), (63, 256), (100, 300), (1000, 256), (0, 512), (256, 512), (320, 300)];
    with_backend(0, spark_gpu_probe::layer_ptx().unwrap(), usize::MAX, |b| {
        b.require_layer_kernels()?; b.require_batch_kernels()?;
        let (heads, most) = (32, 1280);
        let unit = |i: usize, salt: usize| full(half((mix(i, salt, 5) % 65) as f32 / 64.0 - 0.5));
        let mut scores = b.vector(256 * heads * most)?;
        let mut wrong = Vec::new();
        for (kv, dim) in [(4, 512), (16, 256)] {
            let bf16 = |salt: usize| (0..most * kv * dim).flat_map(|i| half(unit(i, salt + dim)).to_le_bytes()).collect::<Vec<_>>();
            let query = (0..most * heads * dim).map(|i| unit(i, 1 + dim)).collect::<Vec<_>>();
            let (mut keys, mut values) = (b.vector(most * kv * dim / 2)?, b.vector(most * kv * dim / 2)?);
            b.write_vector_bytes(&mut keys, &bf16(2))?; b.write_vector_bytes(&mut values, &bf16(3))?;
            b.write_vector(&mut scores, &vec![0.0; 256 * heads * most])?;
            for (kept, more) in DEVICE {
                let all = kept + more;
                let (q_cold, q_warm) = (b.upload_vector(&query[..all * heads * dim])?, b.upload_vector(&query[kept * heads * dim..all * heads * dim])?);
                let (mut cold, mut warm) = (b.vector(all * heads * dim)?, b.vector(more * heads * dim)?);
                b.bounded_attention_device(&q_cold, &keys, &values, &mut scores, &mut cold, heads, kv, dim, 0, all, 0, most, 1.0)?;
                let cold = b.read_vector_values(&cold)?;
                b.bounded_attention_device(&q_warm, &keys, &values, &mut scores, &mut warm, heads, kv, dim, kept, more, 0, most, 1.0)?;
                let read = b.read_vector_values(&warm)?;
                let rows = (0..more).filter(|row| (0..heads * dim).any(|i|
                    read[row * heads * dim + i].to_bits() != cold[(kept + row) * heads * dim + i].to_bits())).collect::<Vec<_>>();
                let hash = read.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
                let start = std::time::Instant::now();
                for _ in 0..20 { b.bounded_attention_device(&q_warm, &keys, &values, &mut scores, &mut warm, heads, kv, dim, kept, more, 0, most, 1.0)?; }
                b.synchronize()?;
                eprintln!("wide_prefill dim={dim} kept={kept} more={more} rows_differing_from_cold={} first={:?} finite={} warm_hash={hash:016x} ms_per_call={:.3}",
                    rows.len(), rows.first(), read.iter().all(|v| v.is_finite()), start.elapsed().as_secs_f64() * 50.0);
                if !rows.is_empty() { wrong.push((dim, kept, more, rows.len())); }
            }
        }
        assert!(wrong.is_empty(), "(dim, kept, continuation, rows that differ from the cold read): {wrong:?}");
        Ok(())
    }).unwrap();
}
