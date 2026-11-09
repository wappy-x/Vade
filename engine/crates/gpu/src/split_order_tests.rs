//! Host model of the FP8 K-split sums on two and four ranks. It repeats the
//! kernels' order of FP32 additions on the CPU: K16 tile chains per canonical
//! range, the four-part merge, the rank fold and the routed down block fold.
//! The tile product is a fixed CPU function, so the model proves orders, not
//! tensor-core bits.
use crate::fp8_abi::{middle_expert_end, EXPERT_FOLD_RANKS};
use crate::kernel_params::{Bf16Rows, bf16_cols_ranges, bf16_cols_word, canonical_span, fold_ranks, shard_ranges, FOLD_RANKS_LIMIT};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits + 0x7fff + (bits >> 16 & 1)) & 0xffff_0000)
}
/// One K16 MMA step of one output: c plus a fixed-order dot of 16 products.
fn tile(c: f32, x: &[f32], w: &[f32]) -> f32 {
    c + x.iter().zip(w).fold(0.0f32, |s, (a, b)| a.mul_add(*b, s))
}
/// One shard's output: H and L chains per range, P = H + L, then the merge
/// (p0 + p1) + (p2 + p3); a part outside the shard's ranges gives -0.0.
fn shard(x: &[f32], w: &[f32], ranges: u32) -> f32 {
    let (hi, lo): (Vec<f32>, Vec<f32>) = x.iter().map(|&v| (bf16(v), bf16(v - bf16(v)))).unzip();
    let part = |part| canonical_span(x.len() / 16, part, ranges).map_or(-0.0, |(start, end)| {
        let (mut h, mut l) = (0.0f32, 0.0f32);
        for t in start..end {
            h = tile(h, &hi[t * 16..t * 16 + 16], &w[t * 16..t * 16 + 16]);
            l = tile(l, &lo[t * 16..t * 16 + 16], &w[t * 16..t * 16 + 16]);
        }
        h + l
    });
    (part(0) + part(1)) + (part(2) + part(3))
}
/// Main's rank fold: rank 0, then add ranks 1..N in sequence.
fn sequential(partials: &[f32]) -> f32 {
    partials[1..].iter().fold(partials[0], |sum, value| sum + value)
}
/// Every output of a [rows, k] projection split by K over `ranks`.
fn split(x: &[f32], w: &[Vec<f32>], ranks: usize, fixed: bool) -> Vec<u32> {
    let k = x.len();
    w.iter().map(|row| {
        let partials = (0..ranks).map(|r| {
            let cols = r * k / ranks..(r + 1) * k / ranks;
            shard(&x[cols.clone()], &row[cols.clone()], if fixed { shard_ranges(cols.len(), k) } else { 0 })
        }).collect::<Vec<_>>();
        if fixed { fold_ranks(ranks, |r| partials[r]) } else { sequential(&partials) }.to_bits()
    }).collect()
}
fn problem(k: usize, rows: usize, seed: u64) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut g = Lcg(seed);
    let x = (0..k).map(|_| g.next() * 3.0).collect();
    (x, (0..rows).map(|_| (0..k).map(|_| bf16(g.next() * 0.05)).collect()).collect())
}
fn differ(a: &[u32], b: &[u32]) -> usize { a.iter().zip(b).filter(|(a, b)| a != b).count() }

/// The cause: on main, two and four ranks group the same K sum differently.
/// The fix: four ranks give the two-rank bits for every output.
#[test]
fn k_split_projections_match_two_ranks_only_with_the_fixed_tree() {
    // Attention o_proj and GDN out_proj (K 6144) and the shared expert down (K 640).
    for (k, seed) in [(6144, 1), (6144, 2), (640, 3)] {
        let (x, w) = problem(k, 256, seed);
        let two = split(&x, &w, 2, false);
        assert_eq!(two, split(&x, &w, 2, true), "K {k}: the fix must keep the two-rank bits");
        let main_four = differ(&two, &split(&x, &w, 4, false));
        let fixed_four = differ(&two, &split(&x, &w, 4, true));
        println!("k_split K={k} outputs=256 differ_from_two_ranks: main_four={main_four} fixed_four={fixed_four}");
        assert!(main_four > 0, "K {k}: main's four-rank order must differ (the cause)");
        assert_eq!(fixed_four, 0, "K {k}: fixed order must equal two ranks");
    }
}

/// fp8.rs project_tile: K128 block dots add in ascending order inside each
/// owner's group; the TP2 entry folds first + rest at split_k, GENERAL folds
/// groups in owner order and adds 0.0 per trailing empty shard.
fn routed_down(blocks: &[f32; 5], expert: usize, owners: usize, general: bool) -> u32 {
    let (mut sums, mut first, mut owner) = (0.0f32, 0.0f32, 0);
    let split_k = if general { 0 } else { middle_expert_end(expert, 0, 2) };
    for (kb, block) in blocks.iter().enumerate() {
        if split_k != 0 && kb == split_k { first = sums; sums = 0.0; }
        if general {
            while owner + 1 < owners && kb == middle_expert_end(expert, owner, owners) {
                if owner == 0 { first = sums; } else { first += sums; }
                sums = 0.0;
                owner += 1;
            }
        }
        sums += block;
    }
    if general {
        if owner != 0 { sums = first + sums; }
        for _ in owner + 1..owners { sums += 0.0; }
        return sums.to_bits();
    }
    (first + sums).to_bits()
}

#[test]
fn routed_down_block_fold_follows_the_pair_split() {
    let mut g = Lcg(7);
    let (mut main_four, mut fixed_four, mut total) = (0, 0, 0);
    for expert in 0..512 {
        for _ in 0..16 {
            let blocks = [(); 5].map(|_| g.next() * 0.3);
            let two = routed_down(&blocks, expert, 2, false);
            assert_eq!(two, routed_down(&blocks, expert, 2, true), "GENERAL at two ranks equals the TP2 fold");
            main_four += usize::from(two != routed_down(&blocks, expert, 4, true));
            fixed_four += usize::from(two != routed_down(&blocks, expert, EXPERT_FOLD_RANKS, true));
            total += 1;
        }
    }
    println!("routed_down outputs={total} differ_from_two_ranks: main_four={main_four} fixed_four={fixed_four}");
    assert!(main_four > 0);
    assert_eq!(fixed_four, 0);
}

/// Bit-exact order checks: the pair tree, the fold levels and the shard spans.
#[test]
fn fold_and_span_orders_are_exact() {
    let mut g = Lcg(11);
    for count in 1..=64 {
        for _ in 0..64 {
            let v = (0..count).map(|_| g.next() * 1.0e3).collect::<Vec<_>>();
            // In-place pairwise passes: step 1, 2, 4, ...
            let mut p = v.clone();
            let mut step = 1;
            while step < count {
                let mut i = 0;
                while i + step < count { p[i] += p[i + step]; i += 2 * step; }
                step *= 2;
            }
            assert_eq!(fold_ranks(count, |r| v[r]).to_bits(), p[0].to_bits(), "count {count}");
        }
    }
    let v = [1.0e8f32, 1.0, -1.0e8, 1.0, 3.0, 5.0];
    assert_eq!(fold_ranks(2, |r| v[r]).to_bits(), (v[0] + v[1]).to_bits());
    assert_eq!(fold_ranks(3, |r| v[r]).to_bits(), ((v[0] + v[1]) + v[2]).to_bits());
    assert_eq!(fold_ranks(4, |r| v[r]).to_bits(), ((v[0] + v[1]) + (v[2] + v[3])).to_bits());
    assert_eq!(fold_ranks(6, |r| v[r]).to_bits(), (((v[0] + v[1]) + (v[2] + v[3])) + (v[4] + v[5])).to_bits());
    // The largest count the 16 levels hold: rank 65534 has 15 trailing ones at most.
    let count = FOLD_RANKS_LIMIT - 1;
    let mut p = (0..count).map(|r| (r % 7) as f32 - 2.5).collect::<Vec<_>>();
    let expected = { let mut step = 1; while step < count { let mut i = 0; while i + step < count { p[i] += p[i + step]; i += 2 * step; } step *= 2; } p[0] };
    assert_eq!(fold_ranks(count, |r| (r % 7) as f32 - 2.5).to_bits(), expected.to_bits());
    for tiles in 0..512usize {
        for part in 0..4 {
            // Ranges 0 (four) are main's canonical::range, unchanged.
            assert_eq!(canonical_span(tiles, part, 0), Some((tiles * part / 4, tiles * (part + 1) / 4)));
        }
        // Each half of a two-rank shard holds that shard's two ranges exactly.
        if tiles % 2 == 0 {
            let half = tiles / 2;
            let two = (0..4).map(|p| canonical_span(tiles, p, 0).unwrap()).collect::<Vec<_>>();
            let four = [0, 1].map(|h| [0, 2].map(|p| canonical_span(half, p, 2).unwrap())
                .map(|(s, e)| (s + h * half, e + h * half)));
            assert_eq!(four.concat(), two);
            for p in [1, 3] { assert_eq!(canonical_span(half, p, 2), None); }
        }
    }
    // -0.0 is the identity for every FP32 value, including both zeros.
    for x in [0.0f32, -0.0, 1.5, -2.0e-38, f32::MAX] { assert_eq!((x + -0.0).to_bits(), x.to_bits()); }
    // The FP8 geometry: o_proj and out_proj K 6144, shared down K 640.
    for whole in [6144, 640] {
        assert_eq!([1, 2, 4, 8].map(|w| shard_ranges(whole / w, whole)), [0, 0, 2, 0]);
        assert_eq!(shard_ranges(whole / 6, whole), 0);
    }
}

/// flash_prefill.rs flash_bf16_mma128 walks K in K32 stages (two K16 tiles) and
/// closes a range when tile + 1 == canonical::end, then moves to canonical::next;
/// its merge is left = p0 (+ p1), right = p2 (+ p3), left + right. decode_rows'
/// paired split block adds l0 + l2. Both must give shard()'s bits, which the
/// two-rank shard gives, for four ranges and for two.
fn wide_walk(x: &[f32], w: &[f32], ranges: u32) -> f32 {
    let kt = x.len() / 16;
    let end = |part| if ranges == 0 { kt * (part + 1) / 4 } else { canonical_span(kt, part, ranges).map_or(usize::MAX, |s| s.1) };
    let next = |part| part + if ranges == 2 { 2 } else { 1 };
    let (hi, lo): (Vec<f32>, Vec<f32>) = x.iter().map(|&v| (bf16(v), bf16(v - bf16(v)))).unzip();
    let (mut h, mut l, mut range, mut left, mut right) = (0.0f32, 0.0f32, 0, 0.0f32, 0.0f32);
    let mut k = 0;
    while k < kt {
        for z in 0..2 {
            let t = k + z;
            h = tile(h, &hi[t * 16..t * 16 + 16], &w[t * 16..t * 16 + 16]);
            l = tile(l, &lo[t * 16..t * 16 + 16], &w[t * 16..t * 16 + 16]);
            if t + 1 == end(range) {
                let value = h + l;
                match range { 0 => left = value, 1 => left += value, 2 => right = value, _ => right += value }
                (h, l, range) = (0.0, 0.0, next(range));
            }
        }
        k += 2;
    }
    left + right
}
#[test]
fn two_range_prompt_tile_and_paired_decode_keep_the_shard_bits() {
    // Shard K of the FP8 geometry on four ranks (1536 and 160) and on two (3072 and 320).
    for (k, seed) in [(1536, 21), (160, 22), (3072, 23), (320, 24), (64, 25)] {
        let (x, w) = problem(k, 64, seed);
        for row in &w {
            for ranges in [0, 2] {
                if ranges == 2 && (k / 16) % 2 != 0 { continue; }
                assert_eq!(wide_walk(&x, row, ranges).to_bits(), shard(&x, row, ranges).to_bits(), "K {k} ranges {ranges}");
            }
            // Paired decode: (l0 + -0.0) + (l2 + -0.0) is l0 + l2 for the two-range parts.
            let parts = [0, 2].map(|p| canonical_span(k / 16, p, 2).map_or(-0.0, |(s, e)| {
                let (hi, lo): (Vec<f32>, Vec<f32>) = x.iter().map(|&v| (bf16(v), bf16(v - bf16(v)))).unzip();
                let (mut h, mut l) = (0.0f32, 0.0f32);
                for t in s..e { h = tile(h, &hi[t * 16..t * 16 + 16], &row[t * 16..t * 16 + 16]); l = tile(l, &lo[t * 16..t * 16 + 16], &row[t * 16..t * 16 + 16]); }
                h + l
            }));
            assert_eq!((parts[0] + parts[1]).to_bits(), shard(&x, row, 2).to_bits(), "K {k} paired decode");
        }
    }
}

/// The carry stack fold_ranks used before its levels moved to registers.
fn carry_stack(count: usize, value: impl Fn(usize) -> f32) -> f32 {
    let (mut sums, mut sizes, mut depth) = ([0.0f32; 16], [0usize; 16], 0);
    for rank in 0..count {
        sums[depth] = value(rank);
        sizes[depth] = 1;
        depth += 1;
        while depth > 1 && sizes[depth - 2] == sizes[depth - 1] { sums[depth - 2] += sums[depth - 1]; sizes[depth - 2] *= 2; depth -= 1; }
    }
    if depth == 0 { return 0.0; }
    let mut sum = sums[depth - 1];
    while depth > 1 { depth -= 1; sum = sums[depth - 1] + sum; }
    sum
}

/// The register levels make the carry stack's additions for every rank count:
/// signed zeros, subnormals, infinities, NaN payloads and cancelling values.
#[test]
fn register_levels_give_the_carry_stack_bits() {
    let special = [0.0f32, -0.0, f32::MIN_POSITIVE / 8.0, -f32::MIN_POSITIVE / 3.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234), f32::from_bits(0xffa0_0001), 1.0e8, -1.0e8, 1.0, f32::MAX, -f32::MAX, 3.0e-3];
    let mut g = Lcg(29);
    for count in (0..=257).chain([1023, 1024, 1025, 32767, 32768, FOLD_RANKS_LIMIT - 1]) {
        for round in 0..8 {
            let v = (0..count).map(|r| if round % 2 == 0 { special[(r * 7 + round) % special.len()] }
                else if round == 3 { -0.0 } else { g.next() * 1.0e3 }).collect::<Vec<_>>();
            assert_eq!(fold_ranks(count, |r| v[r]).to_bits(), carry_stack(count, |r| v[r]).to_bits(), "count {count} round {round}");
        }
    }
    // All -0.0 stays -0.0: no level adds a +0.0.
    for count in 1..=64 { assert_eq!(fold_ranks(count, |_| -0.0).to_bits(), (-0.0f32).to_bits(), "count {count}"); }
}

/// Two ranks launch main's bytes: the descriptor keeps its size, and a shard
/// with four ranges keeps K as its cols word.
#[test]
fn two_rank_launches_keep_main_descriptors() {
    assert_eq!(std::mem::size_of::<Bf16Rows>(), 80);
    for cols in [160u32, 320, 640, 1536, 2560, 3072, 6144, 10240] {
        assert_eq!(bf16_cols_word(cols, 0), cols);
        for ranges in [0, 1, 2] { assert_eq!(bf16_cols_ranges(bf16_cols_word(cols, ranges)), (cols, ranges)); }
        // Only four-rank shards take a ranges word.
        let whole = cols as usize * 2;
        assert_eq!([2, 4, 8].map(|w| shard_ranges(whole / w, whole)), [0, 2, 0]);
    }
}
