use super::*;

// Hand values use a hidden size of 4, so the score scale is 4^-0.5 = 0.5.
// A vector with one nonzero number `a` has mean(x^2) = a^2 / 4. Its plain norm
// is 2 or -2 in that place. The norm epsilon of 1e-6 moves that 2 by
// 4e-6 / a^2 at most, so each `a` below is 2 or more and the bounds are 2e-6.
const H: usize = 4;
const BOUND: f32 = 2e-6;

fn close(actual: &[f32], expected: &[f32], bound: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= bound, "{a} != {b}, bound {bound}"); }
}
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }

#[test]
fn plain_norm_hand_values() {
    // mean([3,4,0,0]^2) = 25/4. With an epsilon of 11/4 the divisor is sqrt(9) = 3.
    close(&plain_norm(&[3.0, 4.0, 0.0, 0.0], 2.75).unwrap(), &[1.0, 4.0 / 3.0, 0.0, 0.0], 2e-7);
    // With the mix epsilon the divisor is sqrt(6.25) = 2.5, to 1 part in 10^7.
    close(&plain_norm(&[3.0, 4.0, 0.0, 0.0], NORM_EPSILON).unwrap(), &[1.2, 1.6, 0.0, 0.0], 1e-6);
    // A zero vector stays zero: the epsilon keeps the divisor above zero.
    assert_eq!(bits(&plain_norm(&[0.0; 4], NORM_EPSILON).unwrap()), bits(&[0.0; 4]));
}

#[test]
fn the_sum_of_squares_uses_the_32_lane_tree() {
    // Lane 0 holds 4096^2 = 2^24. Lanes 1, 3, 5 and 7 hold 1 each. The tree
    // joins lanes 1 and 5, then 3 and 7, then those two sums: 4. Then lane 0:
    // 2^24 + 4. A chain in rising index adds 1 to 2^24 four times, and each
    // sum rounds back to 2^24.
    let x = [4096.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(square_sum(&x), 16_777_220.0);
    let mut chain = 0.0f32;
    for v in x { chain += v * v; }
    assert_eq!(chain, 16_777_216.0);
    // The plain norm uses that sum.
    let scale = 1.0 / (16_777_220.0f32 / 8.0 + NORM_EPSILON).sqrt();
    assert_eq!(bits(&plain_norm(&x, NORM_EPSILON).unwrap()), bits(&x.map(|v| v * scale)));
    // Element 32 shares lane 0 with element 0: 2^24 + 1 rounds to 2^24 there.
    let mut long = [0.0f32; 34];
    (long[0], long[32], long[33]) = (4096.0, 1.0, 1.0);
    assert_eq!(square_sum(&long), 16_777_216.0);
}

#[test]
fn the_softmax_uses_the_shared_exponential() {
    // Entries [2,0,0,0] and [0,2,0,0] norm to 2 in one place. The query
    // [3,1,0,0] gives the scores 3 and 1 after the scale of 0.5. The softmax
    // takes the largest out first: exp(0) and exp(-2), a chain sum, one division each.
    let got = mix(H, &[3.0, 1.0, 0.0, 0.0], &[&[2.0, 0.0, 0.0, 0.0]], &[0.0, 2.0, 0.0, 0.0]).unwrap();
    let norm = plain_norm(&[2.0, 0.0, 0.0, 0.0], NORM_EPSILON).unwrap()[0];
    let scores = [3.0 * norm * 0.5, 1.0 * norm * 0.5];
    let (first, second) = (fixed_math::exp(scores[0] - scores[0]), fixed_math::exp(scores[1] - scores[0]));
    let sum = first + second;
    assert_eq!(bits(&got.weights[0]), bits(&[first / sum, second / sum]));
    close(&got.weights[0], &[0.880797, 0.119203], BOUND);
}

#[test]
fn two_entry_mix_blends_the_normed_entries() {
    // Normed entries: [2,0,0,0] and [0,-2,0,0]. Scores: 0.5 * (q . n) = [ln 3, 0].
    // Softmax of [ln 3, 0] = [3/4, 1/4]. Blend: 3/4 * [2,0,0,0] + 1/4 * [0,-2,0,0].
    let l = 3.0f32.ln();
    let mixed = mix(H, &[l, 0.0, 0.0, 0.0], &[&[3.0, 0.0, 0.0, 0.0]], &[0.0, -5.0, 0.0, 0.0]).unwrap();
    close(&mixed.weights[0], &[0.75, 0.25], 1e-6);
    close(&mixed.values[0], &[1.5, -0.5, 0.0, 0.0], BOUND);
    assert_eq!((mixed.weights.len(), mixed.values.len()), (1, 1));
}

#[test]
fn three_queries_give_three_results_in_row_order() {
    // Row 0 is the case above. Row 1 scores [0, ln 3]: weights [1/4, 3/4].
    // Row 2 is zero: equal weights.
    let l = 3.0f32.ln();
    let queries = [l, 0.0, 0.0, 0.0, 0.0, -l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mixed = mix(H, &queries, &[&[3.0, 0.0, 0.0, 0.0]], &[0.0, -5.0, 0.0, 0.0]).unwrap();
    assert_eq!(mixed.values.len(), 3);
    close(&mixed.values[0], &[1.5, -0.5, 0.0, 0.0], BOUND);
    close(&mixed.values[1], &[0.5, -1.5, 0.0, 0.0], BOUND);
    close(&mixed.values[2], &[1.0, -1.0, 0.0, 0.0], BOUND);
    close(&mixed.weights[1], &[0.25, 0.75], 1e-6);
    close(&mixed.weights[2], &[0.5, 0.5], 1e-6);
}

#[test]
fn end_mix_reads_four_blocks_and_the_stream() {
    // Normed entries: 2e0, 2e1, -2e2, 2e3 and the stream -2e0.
    // Scores with q = [ln 2, 0, ln 2, 0]: [ln 2, 0, -ln 2, 0, -ln 2].
    // Exponentials: [2, 1, 1/2, 1, 1/2], sum 5. Weights: [0.4, 0.2, 0.1, 0.2, 0.1].
    // Blend: [0.4*2 - 0.1*2, 0.2*2, -0.1*2, 0.2*2].
    let l = 2.0f32.ln();
    let blocks: [&[f32]; 4] = [&[4.0, 0.0, 0.0, 0.0], &[0.0, 2.0, 0.0, 0.0], &[0.0, 0.0, -3.0, 0.0], &[0.0, 0.0, 0.0, 8.0]];
    let stream = [-2.0, 0.0, 0.0, 0.0];
    let query = [l, 0.0, l, 0.0];
    close(&end_mix(H, &query, &blocks, &stream).unwrap(), &[0.6, 0.4, -0.2, 0.4], BOUND);
    close(&mix(H, &query, &blocks, &stream).unwrap().weights[0], &[0.4, 0.2, 0.1, 0.2, 0.1], 1e-6);
    // The end mix takes one query.
    assert!(end_mix(H, &[0.0; 8], &blocks, &stream).is_err());
}

#[test]
fn the_mix_reads_normed_entries_not_raw_ones() {
    // 1000 times the block gives the same normed entry, so the same result.
    let l = 3.0f32.ln();
    let small = mix(H, &[l, 0.0, 0.0, 0.0], &[&[3.0, 0.0, 0.0, 0.0]], &[0.0, -5.0, 0.0, 0.0]).unwrap();
    let large = mix(H, &[l, 0.0, 0.0, 0.0], &[&[3000.0, 0.0, 0.0, 0.0]], &[0.0, -5.0, 0.0, 0.0]).unwrap();
    close(&large.values[0], &small.values[0], BOUND);
}

#[test]
fn a_zero_entry_keeps_its_softmax_share() {
    // A zero block norms to zero. With a zero query it still takes half of
    // the weight: the result is half of the normed stream.
    let mixed = mix(H, &[0.0; 4], &[&[0.0; 4]], &[2.0, 0.0, 0.0, 0.0]).unwrap();
    close(&mixed.weights[0], &[0.5, 0.5], 1e-7);
    close(&mixed.values[0], &[1.0, 0.0, 0.0, 0.0], BOUND);
}

#[test]
fn no_finished_block_gives_the_normed_stream() {
    let mixed = mix(H, &[0.3, -0.7, 0.1, 5.0], &[], &[0.0, 0.0, 6.0, 0.0]).unwrap();
    assert_eq!(bits(&mixed.weights[0]), bits(&[1.0]));
    close(&mixed.values[0], &[0.0, 0.0, 2.0, 0.0], BOUND);
}

// Deterministic test numbers in [-1, 1).
fn numbers(count: usize, seed: &mut u32) -> Vec<f32> {
    (0..count).map(|_| { *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); (*seed >> 8) as f32 / 8_388_608.0 - 1.0 }).collect()
}

#[test]
fn dense_vectors_agree_with_a_wide_recomputation() {
    // The same formula in f64, written from the README. Bound: f32 round-off.
    let (hidden, mut seed) = (6usize, 7u32);
    let queries = numbers(2 * hidden, &mut seed);
    let entries: Vec<Vec<f32>> = (0..4).map(|_| numbers(hidden, &mut seed).iter().map(|v| v * 9.0).collect()).collect();
    let blocks: Vec<&[f32]> = entries[..3].iter().map(Vec::as_slice).collect();
    let mixed = mix(hidden, &queries, &blocks, &entries[3]).unwrap();
    let normed: Vec<Vec<f64>> = entries.iter().map(|e| {
        let mean = e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / hidden as f64;
        e.iter().map(|&v| f64::from(v) / (mean + 1e-6).sqrt()).collect()
    }).collect();
    for (row, query) in queries.chunks_exact(hidden).enumerate() {
        let scores: Vec<f64> = normed.iter().map(|n| query.iter().zip(n).map(|(&q, &v)| f64::from(q) * v).sum::<f64>() / (hidden as f64).sqrt()).collect();
        let sum: f64 = scores.iter().map(|s| s.exp()).sum();
        for d in 0..hidden {
            let expected: f64 = scores.iter().zip(&normed).map(|(s, n)| s.exp() / sum * n[d]).sum();
            assert!((f64::from(mixed.values[row][d]) - expected).abs() <= 4e-6, "row {row} element {d}");
        }
        let total: f32 = mixed.weights[row].iter().sum();
        assert!((total - 1.0).abs() <= 1e-6);
    }
    // A second call gives the same bits.
    let again = mix(hidden, &queries, &blocks, &entries[3]).unwrap();
    assert_eq!(bits(&again.values[0]), bits(&mixed.values[0]));
    assert_eq!(bits(&again.weights[1]), bits(&mixed.weights[1]));
}

#[test]
fn bad_shapes_and_values_are_refused() {
    let stream = [1.0, 0.0, 0.0, 0.0];
    assert!(mix(0, &[], &[], &[]).is_err());
    assert!(mix(H, &[], &[], &stream).is_err());
    assert!(mix(H, &[0.0; 5], &[], &stream).is_err());
    assert!(mix(H, &[0.0; 4], &[&[1.0, 2.0, 3.0]], &stream).is_err());
    assert!(mix(H, &[0.0; 4], &[], &stream[..3]).is_err());
    assert!(mix(H, &[f32::NAN, 0.0, 0.0, 0.0], &[], &stream).is_err());
    assert!(mix(H, &[0.0; 4], &[&[f32::INFINITY, 0.0, 0.0, 0.0]], &stream).is_err());
    assert!(mix(H, &[0.0; 4], &[], &[f32::MAX, f32::MAX, 0.0, 0.0]).is_err());
    assert!(plain_norm(&[], NORM_EPSILON).is_err());
    assert!(plain_norm(&[1.0], 0.0).is_err());
}
