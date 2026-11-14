//! Host side of main's fp4_bw expert entries for the world-2/4 rank adapter (next_gpu.rs):
//! one token's compact route tables, built with the device code's own compact functions.
use crate::Result;

/// Route words: error, 10 (token, expert, weight) pairs, then compact::words(1) tables.
pub const BW_PAIRS: usize = 1;
pub const BW_TABLES: usize = BW_PAIRS + 30;
/// Main's compact route tables for one token, built on the host with the device code's own
/// compact::validate/count/prefix/order (same counts, starts, tasks and slot maps as
/// flash_next_compact_route). Returns the words and their params at device address `base`.
pub fn bw_route(
    routes: &[(u32, f32)],
    base: u64,
) -> Result<(Vec<u32>, spark_gpu_probe::fp8_abi::GroupedRouteParams)> {
    use spark_gpu_probe::{compact, fp8_abi::GroupedRouteParams};
    if routes.len() != 10 {
        return Err("fp4_bw route needs ten experts".into());
    }
    let mut words = vec![0u32; BW_TABLES + compact::words(1)];
    for (slot, &(id, w)) in routes.iter().enumerate() {
        words[BW_PAIRS + slot * 3..BW_PAIRS + slot * 3 + 3]
            .copy_from_slice(&[0f32, id as f32, w].map(f32::to_bits));
    }
    let at = |base: u64, word: usize| base + (word * 4) as u64;
    let params = |base: u64| GroupedRouteParams {
        error: at(base, 0),
        pairs: at(base, BW_PAIRS),
        counts: at(base, BW_TABLES),
        starts: at(base, BW_TABLES + 512),
        pair_to_row: at(base, BW_TABLES + 1025),
        sorted_row: at(base, BW_TABLES + 1035),
        token_pairs: at(base, BW_TABLES + 1045),
        tasks: at(base, BW_TABLES + 1055),
        token_count: 0,
        output_row: 0,
        experts: 512,
        tokens: 1,
        pair_count: 10,
        row_capacity: 10,
    };
    let host = params(words.as_mut_ptr() as u64);
    // SAFETY: every span of `host` lies inside `words` (compact::words(1) layout).
    unsafe {
        compact::validate(host, 0);
        if host.failed(0) {
            return Err("invalid fp4_bw route".into());
        }
        for expert in 0..512 {
            compact::count(host, expert);
        }
        compact::prefix(host);
        for expert in 0..512 {
            compact::order(host, expert);
        }
    }
    Ok((words, params(base)))
}

/// bw_route for `rows.len()` tokens of one chunk (rank chunk prefill): error words (one per
/// token), 10 pairs per token, then compact::words(tokens) tables, built the same way.
/// Pair `token * 10 + slot`; fp4_bw_down writes slot `token * 11 + rank`.
/// After compact::validate accepts every token, compact::count is a histogram of the pairs
/// and compact::order one ascending pass over them (each expert's rows stay in pair order,
/// every word is written once): those two run here in O(pairs) instead of 512 scans of all
/// pairs each, writing the same words (bw_route_rows_scan in the tests is the reference).
pub fn bw_route_rows(
    rows: &[Vec<(u32, f32)>],
    base: u64,
) -> Result<(Vec<u32>, spark_gpu_probe::fp8_abi::GroupedRouteParams)> {
    use spark_gpu_probe::{compact, fp8_abi::RoutingLayout};
    let (mut words, params) = route_rows_words(rows)?;
    let n = rows.len();
    let host = params(words.as_mut_ptr() as u64);
    // SAFETY: every span of `host` lies inside `words` (compact::words(n) layout).
    unsafe {
        for token in 0..n {
            compact::validate(host, token);
            if host.failed(token) {
                return Err("invalid fp4_bw route".into());
            }
        }
    }
    // Validated: every id is an integer below 512 and unique within its token.
    let at = |a: u64| ((a - host.error) / 4) as usize;
    let (counts, starts) = (at(host.counts), at(host.starts));
    let (pair_to_row, sorted_row, token_pairs) = (at(host.pair_to_row), at(host.sorted_row), at(host.token_pairs));
    for routes in rows {
        for &(id, _) in routes {
            words[counts + id as usize] += 1;
        }
    }
    // SAFETY: as above; the pointer is taken again after the indexed writes.
    unsafe { compact::prefix(params(words.as_mut_ptr() as u64)) };
    let mut next = words[starts..starts + 512].to_vec();
    for (token, routes) in rows.iter().enumerate() {
        for (slot, &(id, _)) in routes.iter().enumerate() {
            let pair = token * 10 + slot;
            let row = &mut next[id as usize];
            words[sorted_row + *row as usize] = pair as u32;
            *row += 1;
            let rank = RoutingLayout::FLASH.ordered_slot(slot, |s| routes[s].0);
            words[token_pairs + token * 10 + rank] = pair as u32;
            words[pair_to_row + pair] = (token * 11 + rank) as u32;
        }
    }
    Ok((words, params(base)))
}

/// Pairs and zeroed tables of bw_route_rows, and its params at a base address.
#[allow(clippy::type_complexity)]
fn route_rows_words(
    rows: &[Vec<(u32, f32)>],
) -> Result<(Vec<u32>, impl Fn(u64) -> spark_gpu_probe::fp8_abi::GroupedRouteParams)> {
    use spark_gpu_probe::{compact, fp8_abi::GroupedRouteParams};
    let n = rows.len();
    if n == 0 || rows.iter().any(|r| r.len() != 10) {
        return Err("fp4_bw route needs ten experts per token".into());
    }
    let tables = n + 30 * n;
    let mut words = vec![0u32; tables + compact::words(n)];
    for (token, routes) in rows.iter().enumerate() {
        for (slot, &(id, w)) in routes.iter().enumerate() {
            let at = n + (token * 10 + slot) * 3;
            words[at..at + 3].copy_from_slice(&[token as f32, id as f32, w].map(f32::to_bits));
        }
    }
    let at = |base: u64, word: usize| base + (word * 4) as u64;
    let params = move |base: u64| GroupedRouteParams {
        error: at(base, 0),
        pairs: at(base, n),
        counts: at(base, tables),
        starts: at(base, tables + 512),
        pair_to_row: at(base, tables + 1025),
        sorted_row: at(base, tables + 1025 + 10 * n),
        token_pairs: at(base, tables + 1025 + 20 * n),
        tasks: at(base, tables + 1025 + 30 * n),
        token_count: 0,
        output_row: 0,
        experts: 512,
        tokens: n as u32,
        pair_count: 10 * n as u32,
        row_capacity: 10 * n as u32,
    };
    Ok((words, params))
}

#[cfg(test)]
mod bw_route_tests {
    use super::*;
    /// The earlier bw_route_rows (compact::count/order per expert over all pairs).
    fn bw_route_rows_scan(rows: &[Vec<(u32, f32)>], base: u64) -> Result<(Vec<u32>, spark_gpu_probe::fp8_abi::GroupedRouteParams)> {
        use spark_gpu_probe::compact;
        let (mut words, params) = route_rows_words(rows)?;
        let host = params(words.as_mut_ptr() as u64);
        unsafe {
            for token in 0..rows.len() {
                compact::validate(host, token);
                if host.failed(token) {
                    return Err("invalid fp4_bw route".into());
                }
            }
            for expert in 0..512 {
                compact::count(host, expert);
            }
            compact::prefix(host);
            for expert in 0..512 {
                compact::order(host, expert);
            }
        }
        Ok((words, params(base)))
    }
    #[test]
    fn chunk_route_tables_equal_the_per_expert_scans() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |m: u64| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x % m };
        for n in [1usize, 2, 3, 7, 16, 31, 64, 65, 127, 128, 129, 200, 256] {
            for skew in [512u64, 40, 11] {
                let rows = (0..n).map(|_| {
                    let mut ids = Vec::new();
                    while ids.len() < 10 { let id = rnd(skew.max(10)) as u32 * (512 / skew.max(10) as u32); if !ids.contains(&id) { ids.push(id); } }
                    ids.into_iter().map(|id| (id, f32::from_bits(0x3c00_0000 + rnd(1 << 22) as u32))).collect::<Vec<_>>()
                }).collect::<Vec<_>>();
                let (a, pa) = bw_route_rows(&rows, 8192).unwrap();
                let (b, pb) = bw_route_rows_scan(&rows, 8192).unwrap();
                assert_eq!(a, b, "n {n} skew {skew}");
                let f = |p: spark_gpu_probe::fp8_abi::GroupedRouteParams| [p.error, p.pairs, p.counts, p.starts, p.pair_to_row, p.sorted_row, p.token_pairs, p.tasks,
                    p.tokens as u64, p.pair_count as u64, p.row_capacity as u64, p.experts as u64];
                assert_eq!(f(pa), f(pb));
                let mut bad = rows.clone();
                bad[n / 2][3].0 = bad[n / 2][4].0;
                assert!(bw_route_rows(&bad, 0).is_err() && bw_route_rows_scan(&bad, 0).is_err());
                bad[n / 2][3].0 = 512;
                assert!(bw_route_rows(&bad, 0).is_err());
            }
        }
    }
    #[test]
    fn one_token_route_tables_match_the_routed_slots() {
        let ids = [3u32, 17, 40, 255, 256, 300, 301, 420, 500, 511];
        let routes = ids.iter().map(|&id| (id, id as f32 / 4096.0)).collect::<Vec<_>>();
        let (w, p) = bw_route(&routes, 4096).unwrap();
        let at = |a: u64| (a - 4096) as usize / 4;
        assert_eq!(w[0], 0, "route accepted");
        assert_eq!(w[at(p.tasks)], 10, "one task per routed expert");
        for (slot, &id) in ids.iter().enumerate() {
            assert_eq!(w[at(p.counts) + id as usize], 1);
            assert_eq!(w[at(p.starts) + id as usize], slot as u32);
            assert_eq!(w[at(p.tasks) + 1 + slot * 2], id);
            assert_eq!(w[at(p.tasks) + 2 + slot * 2], 0);
            assert_eq!(w[at(p.sorted_row) + slot], slot as u32);
            // Ascending ids: the ranked output slot of pair p is p (token 0).
            assert_eq!(w[at(p.pair_to_row) + slot], slot as u32);
        }
        assert_eq!(w[at(p.starts) + 512], 10);
        assert!(bw_route(&routes[..9], 0).is_err());
        let mut twice = routes.clone();
        twice[1].0 = 3;
        assert!(bw_route(&twice, 0).is_err(), "duplicate expert rejected");
    }
    #[test]
    fn chunk_route_tables_extend_the_one_token_tables() {
        let ids = [3u32, 17, 40, 255, 256, 300, 301, 420, 500, 511];
        let one = ids.iter().map(|&id| (id, id as f32 / 4096.0)).collect::<Vec<_>>();
        assert_eq!(bw_route_rows(std::slice::from_ref(&one), 4096).unwrap().0, bw_route(&one, 4096).unwrap().0);
        let two = [17u32, 18, 40, 41, 256, 257, 300, 400, 499, 511]
            .iter()
            .map(|&id| (id, 1.0 / 16.0))
            .collect::<Vec<_>>();
        let (w, p) = bw_route_rows(&[one, two], 0).unwrap();
        let at = |a: u64| a as usize / 4;
        assert_eq!(&w[..2], &[0, 0], "both tokens accepted");
        assert_eq!(w[at(p.counts) + 17], 2, "a shared expert holds both tokens");
        assert_eq!(w[at(p.starts) + 512], 20);
        // Token 1's pairs address token 1's output slots (11 per token).
        for slot in 0..10 {
            assert!((11..21).contains(&w[at(p.pair_to_row) + 10 + slot]));
            assert!(w[at(p.pair_to_row) + slot] < 10);
        }
        assert!(bw_route_rows(&[], 0).is_err());
    }
}
