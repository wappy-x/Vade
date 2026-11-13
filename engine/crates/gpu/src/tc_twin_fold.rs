//! Kernel kit: the CPU twin (reference) of QUARTER-FOLD v1, the expert fold of a second matrix that is split by
//! input columns, and of the REDUCE that joins the owners' partial sums.
//!
//! The function, for one token row and one output feature:
//! - `a[e][q]` = the quarter accumulator of routed expert e's down matrix (`tc_twin::chain` over quarter q, before the
//!   epilogue); `k[e] = fl(global_e * activation_e)` = that matrix's epilogue scale; `w[e]` = its route weight;
//!   `s[q]` = the quarter accumulator of the shared expert's down matrix, when the layer has one.
//! - `Q[q] = +0.0`; for each routed expert in fold order `Q[q] = fl(Q[q] + fl(fl(a[e][q] * k[e]) * w[e]))`; then
//!   `Q[q] = fl(Q[q] + s[q])`.
//! - `joined = fl(fl(Q[0] + Q[1]) + fl(Q[2] + Q[3]))`; `word = bf16(joined)`: the ONE BF16 round of the sublayer output.
//!
//! Every product and sum is one binary32 operation, round to nearest even. Never fused, never reordered: Rust does
//! not contract a multiply and an add on the CPU, and the tests below hold this file to an integer model of the two
//! operations. The order depends on the column quarter and the expert id only. So one owner (four quarters), two
//! owners (two quarters each) and four owners (one each) give the same bits: an owner sends the join of its own
//! quarters, and every owner joins all partials in owner order. Std only. No unsafe code.

use crate::tc_twin::Rng;

/// The BF16 round of a sublayer output, as FP32 bits: nearest even at 16 bits; it can give infinity; -0.0 stays -0.0.
/// A NaN gives the canonical quiet NaN of `cvt.rn.bf16.f32` (`kernels/src/bf16.rs`), so a NaN term gives a NaN word.
pub fn bf16_word(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x7fff_ffff > 0x7f80_0000 { 0x7fff_0000 } else { b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000 }
}

/// One routed expert of a token row: its id, the four quarter accumulators of its down matrix (FP32 bits), its
/// epilogue scale and its route weight.
#[derive(Clone, Copy, Debug)]
pub struct FoldExpert { pub id: u32, pub acc: [u32; 4], pub k: f32, pub w: f32 }

/// The slots of a token row in the order the fold visits them. This is the rule of today's fold (`latent_ops`,
/// `Op::ExpertFold`), kept bit for bit: each visit takes the smallest id above the last visited id, and the first slot
/// that holds it. When no id is left above (two slots hold one id), the visit takes slot 0 again. The router never
/// picks one expert twice, so a served row is visited in ascending id, each slot once.
pub fn fold_order(ids: &[u32]) -> Vec<usize> {
    let mut order = Vec::with_capacity(ids.len());
    let mut previous: Option<u64> = None;
    for _ in 0..ids.len() {
        let (mut selected, mut best) = (0usize, u64::MAX);
        for (slot, &id) in ids.iter().enumerate() {
            if previous.is_none_or(|v| u64::from(id) > v) && u64::from(id) < best { selected = slot; best = u64::from(id); }
        }
        order.push(selected);
        previous = Some(best);
    }
    order
}

/// `Q[q]`: the fold of one quarter. `order` = `fold_order` of the experts' ids.
pub fn fold_quarter(experts: &[FoldExpert], order: &[usize], shared: Option<[u32; 4]>, q: usize) -> u32 {
    let mut v = 0.0f32;
    for &slot in order {
        let e = &experts[slot];
        let scaled = f32::from_bits(e.acc[q]) * e.k;
        let term = scaled * e.w;
        v += term;
    }
    if let Some(s) = shared { v += f32::from_bits(s[q]); }
    v.to_bits()
}

/// The fixed join of 1, 2 or 4 partial sums: neighbors first. 4 -> fl(fl(p0 + p1) + fl(p2 + p3)); 2 -> fl(p0 + p1).
pub fn join(partials: &[u32]) -> u32 {
    assert!(partials.len().is_power_of_two() && partials.len() <= 4, "a join takes 1, 2 or 4 partial sums");
    if partials.len() == 1 { return partials[0]; }
    let half = partials.len() / 2;
    (f32::from_bits(join(&partials[..half])) + f32::from_bits(join(&partials[half..]))).to_bits()
}

/// What owner `owner` of `parts` equal owners sends in the REDUCE: the join of its own quarters. With one owner this
/// is the whole join; with four it is the quarter itself.
pub fn partial(q: [u32; 4], owner: usize, parts: usize) -> u32 {
    assert!(parts.is_power_of_two() && parts <= 4 && owner < parts, "quarters split over 1, 2 or 4 owners");
    let own = 4 / parts;
    join(&q[owner * own..][..own])
}

/// The REDUCE as every owner computes it: the join of all partials in owner order. The first operand of each add is
/// the lower owner's partial on every owner, so all owners run the same operation on the same bits.
pub fn reduce(partials: &[u32]) -> u32 { join(partials) }

/// The result of the fold for one output feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fold { pub q: [u32; 4], pub joined: u32, pub word: u32 }

/// QUARTER-FOLD v1 for one token row and one output feature.
pub fn quarter_fold(experts: &[FoldExpert], shared: Option<[u32; 4]>) -> Fold {
    let ids: Vec<u32> = experts.iter().map(|e| e.id).collect();
    let order = fold_order(&ids);
    let q: [u32; 4] = std::array::from_fn(|q| fold_quarter(experts, &order, shared, q));
    let joined = join(&q);
    Fold { q, joined, word: bf16_word(f32::from_bits(joined)) }
}

/// A second matrix with no fold (the o projections, a dense down): the quarter join, then the epilogue of today,
/// `bf16(fl(acc * k))`. Returns (joined accumulator, word).
pub fn single(acc: [u32; 4], k: f32) -> (u32, u32) {
    let joined = join(&acc);
    (joined, bf16_word(f32::from_bits(joined) * k))
}

/// One word of the REDUCE of two owners, as `tp_reduce_take` computes it: fl(lower + higher), where `lower` is the
/// lower owner's partial on both owners; then times `scale` when given (the epilogue of a second matrix with no
/// fold); then the BF16 round when `round`.
pub fn reduce_word(lower: u32, higher: u32, scale: Option<f32>, round: bool) -> u32 {
    let mut sum = f32::from_bits(lower) + f32::from_bits(higher);
    if let Some(scale) = scale { sum *= scale; }
    if round { bf16_word(sum) } else { sum.to_bits() }
}

// ------------------------------------------------------------------ hard inputs
/// One fold input and the name of its class.
pub struct FoldCase { pub class: &'static str, pub experts: Vec<FoldExpert>, pub shared: Option<[u32; 4]> }
/// The hard classes of the fold. Every run covers each of them `n / FOLD_CLASSES` times, with 1 to 8 experts, with
/// and without a shared expert: ordinary values; any FP32 bits; signed zeros; subnormal accumulators; a pair of
/// experts that cancels inside every quarter; a cancelled inner join; a cancelled outer join; one NaN; one infinity;
/// opposite infinities; a sum that rounds to BF16 infinity; one id in two slots; ids out of slot order; a sum on a
/// BF16 tie.
pub const FOLD_CLASSES: usize = 14;
pub fn hard_folds(n: usize, seed: u64) -> Vec<FoldCase> {
    let mut r = Rng(seed);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let count = 1 + (i / FOLD_CLASSES) % 8;
        let with_shared = (i / FOLD_CLASSES / 8).is_multiple_of(2);
        let mut experts: Vec<FoldExpert> = (0..count).map(|e| FoldExpert { id: 3 * e as u32 + r.below(3) as u32, acc: std::array::from_fn(|_| ordinary(&mut r, 6)), k: f32::from_bits(ordinary(&mut r, -5)), w: f32::from_bits(ordinary(&mut r, -2) & 0x7fff_ffff) }).collect();
        let mut shared: Option<[u32; 4]> = with_shared.then(|| std::array::from_fn(|_| ordinary(&mut r, 0)));
        let unit = |experts: &mut Vec<FoldExpert>| for e in experts.iter_mut() { e.k = 1.0; e.w = 1.0; };
        let class = match i % FOLD_CLASSES {
            0 => "ordinary values",
            1 => { for e in &mut experts { for v in &mut e.acc { *v = r.next_u64() as u32; } e.k = f32::from_bits(r.next_u64() as u32); e.w = f32::from_bits(r.next_u64() as u32); } if let Some(s) = &mut shared { for v in s { *v = r.next_u64() as u32; } } "any FP32 bits" }
            2 => { for e in &mut experts { for v in &mut e.acc { *v = (r.below(2) as u32) << 31; } } if let Some(s) = &mut shared { for v in s { *v = (r.below(2) as u32) << 31; } } "signed zeros" }
            3 => { unit(&mut experts); for e in &mut experts { for v in &mut e.acc { *v = r.next_u64() as u32 & 0x807f_ffff; } } if let Some(s) = &mut shared { for v in s { *v = r.next_u64() as u32 & 0x807f_ffff; } } "subnormal accumulators" }
            4 => { let first = experts[0]; experts.truncate(1); experts.push(FoldExpert { id: first.id + 1, acc: first.acc.map(|a| a ^ 0x8000_0000), ..first }); shared = shared.map(|_| [0; 4]); "a pair cancels inside every quarter" }
            5 => { unit(&mut experts); experts.truncate(1); shared = None; let a = experts[0].acc; experts[0].acc = [a[0], a[0] ^ 0x8000_0000, a[2], a[2] ^ 0x8000_0000]; "a cancelled inner join" }
            6 => { unit(&mut experts); experts.truncate(1); shared = None; let a = experts[0].acc; experts[0].acc = [a[0], a[1], a[0] ^ 0x8000_0000, a[1] ^ 0x8000_0000]; "a cancelled outer join" }
            7 => { let (e, q) = (r.below(count as u64) as usize, r.below(4) as usize); experts[e].acc[q] = 0x7f80_0000 | ((r.below(2) as u32) << 31) | (1 + r.below(0x7f_ffff) as u32); "one NaN" }
            8 => { let (e, q) = (r.below(count as u64) as usize, r.below(4) as usize); experts[e].acc[q] = 0x7f80_0000 | ((r.below(2) as u32) << 31); "one infinity" }
            9 => { let e = r.below(count as u64) as usize; let (q0, q1) = (r.below(4) as usize, r.below(4) as usize); let sign = (r.below(2) as u32) << 31;
                if q0 == q1 { experts.push(FoldExpert { id: 99, acc: [0; 4], k: experts[e].k, w: experts[e].w }); let last = experts.len() - 1; experts[last].acc[q0] = 0x7f80_0000 | sign; experts[e].acc[q0] = 0xff80_0000 ^ sign; }
                else { experts[e].acc[q0] = 0x7f80_0000 | sign; experts[e].acc[q1] = 0xff80_0000 ^ sign; }
                "opposite infinities" }
            10 => { unit(&mut experts); experts.truncate(1); shared = None; let sign = (r.below(2) as u32) << 31; let q = r.below(4) as usize; experts[0].acc = [0; 4]; experts[0].acc[q] = sign | 0x7f7f_8000 | (r.next_u64() as u32 & 0x7fff); "rounds to BF16 infinity" }
            11 => { if count > 1 { let from = r.below(count as u64 - 1) as usize; experts[from + 1].id = experts[from].id; } "one id in two slots" }
            12 => { for slot in (1..count).rev() { let other = r.below(slot as u64 + 1) as usize; experts.swap(slot, other); } "ids out of slot order" }
            _ => { unit(&mut experts); experts.truncate(1); shared = None; let base = (ordinary(&mut r, 0) & 0xffff_0000) | 0x8000; let tiny = (base & 0xff80_0000).wrapping_sub(24 << 23);
                let half = tiny ^ ((r.below(2) as u32) << 31);
                // base is on a BF16 tie and `tiny` is half of its last FP32 place: the tree join moves off the tie, a serial sum stays on it.
                experts[0].acc = [base, if r.below(2) == 0 { tiny ^ 0x8000_0000 } else { 0 }, half, half]; "a sum on a BF16 tie" }
        };
        out.push(FoldCase { class, experts, shared });
    }
    out
}
/// A finite FP32 value of either sign with its exponent within 8 of `exponent`.
fn ordinary(r: &mut Rng, exponent: i32) -> u32 {
    let e = (127 + exponent + r.below(17) as i32 - 8) as u32;
    ((r.below(2) as u32) << 31) | (e << 23) | (r.next_u64() as u32 & 0x7f_ffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    // An integer model of the two binary32 operations (round to nearest even), sharing no code with the twin above.
    // value = mag * 2^exp. Zero has mag 0.
    fn parts(b: u32) -> (bool, u128, i32) {
        let (e, m) = ((b >> 23 & 255) as i32, (b & 0x7f_ffff) as u128);
        if e == 0 { (b >> 31 != 0, m, -149) } else { (b >> 31 != 0, m | 1 << 23, e - 150) }
    }
    fn pack(neg: bool, mag: u128, exp: i32) -> u32 {
        let sign = (neg as u32) << 31;
        if mag == 0 { return sign; }
        let top = 127 - mag.leading_zeros() as i32 + exp;
        let low = if top < -126 { -149 } else { top - 23 };
        let shift = low - exp;
        let mut kept = if shift <= 0 { mag << (-shift) as u32 } else if shift >= 128 { 0 } else { mag >> shift as u32 };
        if shift > 0 {
            let (rest, half) = if shift > 128 { (0, 1) } else if shift == 128 { (mag, 1u128 << 127) } else { (mag & ((1u128 << shift as u32) - 1), 1u128 << (shift as u32 - 1)) };
            if rest > half || (rest == half && kept & 1 == 1) { kept += 1; }
        }
        // A carry out of the significand moves into the exponent field by itself; so does a subnormal that became normal.
        let field = if top < -126 { 0 } else { (top + 127) as u32 };
        let bits = (field << 23).wrapping_add(kept as u32).wrapping_sub(if top < -126 { 0 } else { 1 << 23 });
        if bits >= 0x7f80_0000 { sign | 0x7f80_0000 } else { sign | bits }
    }
    const NAN: u32 = 0x7fc0_0000;
    fn is_nan(b: u32) -> bool { b & 0x7fff_ffff > 0x7f80_0000 }
    fn is_inf(b: u32) -> bool { b & 0x7fff_ffff == 0x7f80_0000 }
    fn soft_mul(a: u32, b: u32) -> u32 {
        let sign = (a ^ b) & 0x8000_0000;
        if is_nan(a) || is_nan(b) { return NAN; }
        if is_inf(a) || is_inf(b) { return if a << 1 == 0 || b << 1 == 0 { NAN } else { sign | 0x7f80_0000 }; }
        let ((_, ma, ea), (_, mb, eb)) = (parts(a), parts(b));
        pack(sign != 0, ma * mb, ea + eb)
    }
    fn soft_add(a: u32, b: u32) -> u32 {
        if is_nan(a) || is_nan(b) { return NAN; }
        if is_inf(a) || is_inf(b) { return if is_inf(a) && is_inf(b) && a != b { NAN } else if is_inf(a) { a } else { b }; }
        if a << 1 == 0 && b << 1 == 0 { return a & b; }
        let ((na, ma, ea), (nb, mb, eb)) = (parts(a), parts(b));
        let (hi, lo) = if ea >= eb { ((na, ma, ea), (nb, mb, eb)) } else { ((nb, mb, eb), (na, ma, ea)) };
        // Past 100 bits of distance the low operand is under half a unit of any result: it acts as one sticky unit.
        let distance = (hi.2 - lo.2).min(100);
        let low_mag = if hi.2 - lo.2 > 100 { (lo.1 != 0) as u128 } else { lo.1 };
        let (x, y) = ((hi.1 << distance as u32) as i128, low_mag as i128);
        let sum = if hi.0 { -x } else { x } + if lo.0 { -y } else { y };
        if sum == 0 { return 0; }
        pack(sum < 0, sum.unsigned_abs(), hi.2 - distance)
    }
    fn same(a: u32, b: u32) -> bool { a == b || (is_nan(a) && is_nan(b)) }
    fn soft_join(p: &[u32]) -> u32 { if p.len() == 1 { p[0] } else { soft_add(soft_join(&p[..p.len() / 2]), soft_join(&p[p.len() / 2..])) } }
    /// The spec's pseudocode on the integer model, with its own copy of the visiting order.
    fn soft_fold(case: &FoldCase) -> ([u32; 4], u32) {
        let n = case.experts.len();
        let mut order = Vec::new();
        let mut last: Option<u32> = None;
        for _ in 0..n {
            let next = case.experts.iter().map(|e| e.id).filter(|&id| last.is_none_or(|l| id > l)).min();
            order.push(next.map_or(0, |id| case.experts.iter().position(|e| e.id == id).unwrap()));
            last = Some(next.unwrap_or(u32::MAX));
        }
        let q: [u32; 4] = std::array::from_fn(|q| {
            let mut v = 0u32;
            for &slot in &order { let e = &case.experts[slot]; v = soft_add(v, soft_mul(soft_mul(e.acc[q], e.k.to_bits()), e.w.to_bits())); }
            if let Some(s) = case.shared { v = soft_add(v, s[q]); }
            v
        });
        (q, soft_join(&q))
    }

    #[test]
    fn the_integer_model_equals_the_cpu_operations() {
        let mut r = Rng(0xf01d);
        let pick = |r: &mut Rng| match r.below(6) { 0 => r.next_u64() as u32, 1 => r.next_u64() as u32 & 0x807f_ffff, 2 => (r.below(2) as u32) << 31, 3 => 0x7f00_0000 | (r.next_u64() as u32 & 0x80ff_ffff), 4 => ordinary(r, 0), _ => ordinary(r, -20) };
        for i in 0..400_000 {
            let (a, mut b) = (pick(&mut r), pick(&mut r));
            // Near-cancelling and far-apart pairs, which a random pair almost never is.
            if i % 5 == 0 { b = (a ^ 0x8000_0000).wrapping_add(r.below(5) as u32).wrapping_sub(2); }
            let (x, y) = (f32::from_bits(a), f32::from_bits(b));
            assert!(same((x + y).to_bits(), soft_add(a, b)), "{a:08x} + {b:08x}: {:08x} {:08x}", (x + y).to_bits(), soft_add(a, b));
            assert!(same((x * y).to_bits(), soft_mul(a, b)), "{a:08x} * {b:08x}: {:08x} {:08x}", (x * y).to_bits(), soft_mul(a, b));
        }
        // Ties to even, the largest finite value, the smallest subnormal, signed zeros.
        assert_eq!(soft_add(0x3f80_0000, 0x3380_0000), 0x3f80_0000);
        assert_eq!(soft_add(0x3f80_0001, 0x3380_0000), 0x3f80_0002);
        assert_eq!(soft_add(0x7f7f_ffff, 0x7f7f_ffff), 0x7f80_0000);
        assert_eq!(soft_mul(0x0000_0001, 0x3f00_0000), 0);
        assert_eq!(soft_mul(0x0000_0003, 0x3f00_0000), 2);
        assert_eq!(soft_add(0x8000_0000, 0x8000_0000), 0x8000_0000);
        assert_eq!(soft_add(0x8000_0000, 0), 0);
        assert_eq!(soft_add(0x3f80_0000, 0xbf80_0000), 0);
    }

    // The generator of the spec's vectors (the SUM-ORDER-SPEC term): magnitude in [0.5, 1), sign set when (i + s) % 3 == 0.
    fn term(i: u64, s: u64) -> f32 {
        let b = 0x3f00_0000 | ((i * 2_654_435_761 + s * 40_503) & 0x7f_ffff) as u32;
        f32::from_bits(if (i + s).is_multiple_of(3) { b | 0x8000_0000 } else { b })
    }
    /// The vectors of the spec (`quarter_fold_vectors.txt`, made by the plain reference beside the spec), bit for bit:
    /// the four `Q`, the two partial sums of a two-owner split, the joined sum and the output word.
    #[test]
    fn the_spec_vectors() {
        let rows: [(u64, u64, bool, [u32; 4], [u32; 2], u32, u32); 4] = [
            (8, 1, true, [0x3dad5638, 0x401ee505, 0xbfdc9fa6, 0x3ed06e4b], [0x40244fb7, 0xbfa88413], 0x3fa01b5b, 0x3fa00000),
            (8, 2, false, [0xbf386fd7, 0x3fe29752, 0xbf17f76d, 0xbf2c62fd], [0x3f865f66, 0xbfa22d35], 0xbe5e6e78, 0xbe5e0000),
            (1, 3, true, [0xbf29b970, 0x3f711e79, 0x3f9c41c1, 0xbf5b2d60], [0x3e8eca12, 0x3ebaac44], 0x3f24bb2b, 0x3f250000),
            (3, 4, true, [0x3e883750, 0x3fa9e83f, 0xbfa266ea, 0x3f032621], [0x3fcbf613, 0xbf41a7b3], 0x3f564473, 0x3f560000),
        ];
        for (n, salt, with_shared, q, sent, joined, word) in rows {
            let experts: Vec<FoldExpert> = (0..n).map(|e| FoldExpert { id: e as u32, acc: std::array::from_fn(|q| (term(4 * e + q as u64, salt) * 64.0).to_bits()), k: term(100 + e, salt) * 0.03125, w: term(200 + e, salt).abs() * 0.25 }).collect();
            let shared = with_shared.then(|| std::array::from_fn(|q| term(300 + q as u64, salt).to_bits()));
            let fold = quarter_fold(&experts, shared);
            assert_eq!(fold, Fold { q, joined, word }, "{n} experts, salt {salt}");
            assert_eq!([partial(fold.q, 0, 2), partial(fold.q, 1, 2)], sent, "{n} experts, salt {salt}");
        }
        let minus_zero: Vec<FoldExpert> = (0..8).map(|id| FoldExpert { id, acc: [0x8000_0000; 4], k: 1.0, w: 1.0 }).collect();
        let fold = quarter_fold(&minus_zero, None);
        assert_eq!((fold.joined, fold.word), (0, 0), "a fold of only -0.0 terms is +0.0");
        let cancel = quarter_fold(&[FoldExpert { id: 0, acc: [1.0f32, -1.0, 0.5, -0.5].map(f32::to_bits), k: 1.0, w: 1.0 }], None);
        assert_eq!((cancel.joined, cancel.word), (0, 0));
        let acc: [u32; 4] = std::array::from_fn(|q| (term(q as u64, 9) * 1024.0).to_bits());
        assert_eq!(acc, [0xc4058fef, 0x443d09a0, 0x44748351, 0xc42bfd02]);
        assert_eq!(single(acc, 0.00390625 * term(7, 9)), (0x44000000, 0x3f8a0000));
        for (x, want) in [(0x3f808000u32, 0x3f800000u32), (0x3f818000, 0x3f820000), (0x3f808001, 0x3f810000), (0x7f7fffff, 0x7f800000), (0x80008000, 0x80000000)] {
            assert_eq!(bf16_word(f32::from_bits(x)), want, "BF16 round of {x:08x}");
        }
    }

    /// Every hard class is present, the twin equals the integer model on every case, and each class keeps its law.
    #[test]
    fn the_fold_equals_the_integer_model_on_hard_inputs() {
        let cases = hard_folds(FOLD_CLASSES * 16 * 400, 0xd2);
        let mut names: Vec<&str> = cases.iter().map(|c| c.class).collect();
        names.sort(); names.dedup();
        assert_eq!(names.len(), FOLD_CLASSES);
        let mut tied = 0usize;
        for (i, case) in cases.iter().enumerate() {
            let fold = quarter_fold(&case.experts, case.shared);
            let (q, joined) = soft_fold(case);
            for (k, (&got, &want)) in fold.q.iter().zip(&q).enumerate() { assert!(same(got, want), "case {i} ({}) quarter {k}: {got:08x} {want:08x}", case.class); }
            assert!(same(fold.joined, joined), "case {i} ({}): {:08x} {joined:08x}", case.class, fold.joined);
            assert_eq!(is_nan(fold.word), is_nan(joined), "case {i} ({})", case.class);
            if is_nan(fold.word) { assert_eq!(fold.word, 0x7fff_0000); }
            match case.class {
                "signed zeros" => assert_eq!((fold.q, fold.word), ([0; 4], 0), "case {i}: only zero terms give +0.0"),
                "a pair cancels inside every quarter" => assert_eq!(fold.q, [0; 4], "case {i}"),
                "a cancelled inner join" | "a cancelled outer join" => assert_eq!(fold.joined, 0, "case {i} ({})", case.class),
                "one NaN" | "opposite infinities" => assert_eq!(fold.word, 0x7fff_0000, "case {i} ({})", case.class),
                "one infinity" => assert_eq!(fold.word & 0x7fff_ffff, 0x7f80_0000, "case {i}"),
                "rounds to BF16 infinity" => { assert!(fold.joined & 0x7fff_ffff < 0x7f80_0000); assert_eq!(fold.word & 0x7fff_ffff, 0x7f80_0000, "case {i}"); }
                "subnormal accumulators" => assert!(fold.q.iter().all(|&v| (v >> 23) & 255 <= 4), "case {i}: a sum of at most 9 subnormals stays under 2^-122"),
                "one id in two slots" => { let ids: Vec<u32> = case.experts.iter().map(|e| e.id).collect(); let order = fold_order(&ids); let mut seen = order.clone(); seen.sort(); seen.dedup(); if seen.len() < ids.len() { tied += 1; assert_eq!(*order.last().unwrap(), 0, "case {i}: the visit past the last id takes slot 0"); } }
                "ids out of slot order" => { let ids: Vec<u32> = case.experts.iter().map(|e| e.id).collect(); let visited: Vec<u32> = fold_order(&ids).iter().map(|&s| ids[s]).collect(); assert!(visited.windows(2).all(|w| w[0] < w[1]), "case {i}: ascending id, each slot once"); }
                _ => {}
            }
        }
        assert!(tied > 1_000, "{tied}");
    }

    /// The same bits on 1, 2 and 4 owners: each owner sends the join of its own quarters; the REDUCE joins the
    /// partials in owner order.
    #[test]
    fn one_two_and_four_owners_give_the_same_bits() {
        for (i, case) in hard_folds(FOLD_CLASSES * 16 * 100, 0x124).iter().enumerate() {
            let fold = quarter_fold(&case.experts, case.shared);
            for parts in [1usize, 2, 4] {
                let sent: Vec<u32> = (0..parts).map(|owner| partial(fold.q, owner, parts)).collect();
                assert!(same(reduce(&sent), fold.joined), "case {i} ({}), {parts} owners", case.class);
            }
            // The pair's exchange, word by word: the unrounded sum is the join, the rounded one is the output word.
            let (low, high) = (partial(fold.q, 0, 2), partial(fold.q, 1, 2));
            assert!(same(reduce_word(low, high, None, false), fold.joined), "case {i} ({})", case.class);
            assert_eq!(reduce_word(low, high, None, true), fold.word, "case {i} ({})", case.class);
            // A second matrix with no fold: its scale enters after the join.
            let acc = case.experts[0].acc;
            assert_eq!(reduce_word(partial(acc, 0, 2), partial(acc, 1, 2), Some(case.experts[0].k), true), single(acc, case.experts[0].k).1, "case {i} ({})", case.class);
        }
    }

    /// The fold is a NEW function. The inputs here tell it from each form it must not become: a left-to-right join,
    /// a fold after the join, and today's fold (a BF16 round of every expert word and of every step).
    #[test]
    fn the_hard_inputs_tell_the_fold_from_its_neighbors() {
        let f = f32::from_bits;
        let (mut serial_sum, mut serial_word, mut late_sum, mut today_word, mut tie_word) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let cases = hard_folds(FOLD_CLASSES * 16 * 200, 0x9e);
        for case in &cases {
            let fold = quarter_fold(&case.experts, case.shared);
            if is_nan(fold.joined) { continue; }
            let serial = (((f(fold.q[0]) + f(fold.q[1])) + f(fold.q[2])) + f(fold.q[3])).to_bits();
            serial_sum += (serial != fold.joined) as usize;
            serial_word += (bf16_word(f(serial)) != fold.word) as usize;
            tie_word += (case.class == "a sum on a BF16 tie" && bf16_word(f(serial)) != fold.word) as usize;
            // A fold after the join: each expert's four quarters are joined first, then the experts are folded.
            let ids: Vec<u32> = case.experts.iter().map(|e| e.id).collect();
            let order = fold_order(&ids);
            let (mut late, mut today) = (0.0f32, 0.0f32);
            for &slot in &order {
                let e = &case.experts[slot];
                late += (f(join(&e.acc)) * e.k) * e.w;
                today = f(bf16_word(today + f(bf16_word(f(bf16_word(f(join(&e.acc)) * e.k)) * e.w))));
            }
            if let Some(s) = case.shared { late += f(join(&s)); today = f(bf16_word(today + f(bf16_word(f(join(&s)))))); }
            late_sum += (!same(late.to_bits(), fold.joined)) as usize;
            today_word += (!same(today.to_bits(), fold.word)) as usize;
        }
        assert!(serial_sum > cases.len() / 20 && late_sum > cases.len() / 20, "{serial_sum} {late_sum} of {}", cases.len());
        assert!(serial_word > 100 && tie_word > 100 && today_word > 1_000, "{serial_word} {tie_word} {today_word} of {}", cases.len());
        // The join order by hand: (1 + 2^-24) + (2^-24 - 1) = 2^-24; left to right it is 0.
        let q = [1.0f32, 2f32.powi(-24), 2f32.powi(-24), -1.0].map(f32::to_bits);
        assert_eq!(f(join(&q)), 2f32.powi(-24));
        assert_eq!(((f(q[0]) + f(q[1])) + f(q[2])) + f(q[3]), 0.0);
        assert_eq!([partial(q, 0, 2), partial(q, 1, 2)].map(f), [1.0, 2f32.powi(-24) - 1.0]);
        assert_eq!(partial(q, 2, 4), q[2]);
        assert_eq!(partial(q, 0, 1), join(&q));
    }
}
