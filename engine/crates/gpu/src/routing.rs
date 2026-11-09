//! Startup-only routing injection shared by legacy, Flash and typed forwards.
use std::sync::{OnceLock, atomic::{AtomicBool, AtomicUsize, Ordering}};
pub(crate) static ROUTING_DEFERRED: AtomicBool = AtomicBool::new(false);
pub(crate) fn forced_routing_rejection(rows: usize, rank: Option<usize>) -> Option<usize> {
    // References stay uncounted, including before startup hook initialization.
    if ROUTING_DEFERRED.load(Ordering::Relaxed) { return None; }
    static TARGET: OnceLock<Option<(usize, usize, usize, u128)>> = OnceLock::new(); static SEEN: AtomicUsize = AtomicUsize::new(0);
    let (n, row, width, mask) = (*TARGET.get_or_init(|| {
        let value = std::env::var("VADE_TEST_ROUTING_REJECT").ok().filter(|v| v != "0")?;
        let target = (|| {
            let mut fields = value.split(':');
            let n = fields.next()?.parse().ok().filter(|&n| n > 0)?;
            let row = fields.next().unwrap_or("0").parse().ok()?; let width = fields.next().unwrap_or("0").parse().ok()?;
            let mask = fields.next().map_or(Some(u128::MAX), |v| v.parse().ok().filter(|&m| m != 0))?;
            if fields.next().is_some() { return None; } Some((n, row, width, mask))
        })();
        eprintln!("test_hook VADE_TEST_ROUTING_REJECT={value} {}", if target.is_some() { "armed" } else { "invalid, ignored" }); target
    }))?;
    if rows == 0 || (width != 0 && rows != width) || rank.is_some_and(|r| r >= 128 || mask & (1u128 << r) == 0) { return None; }
    (SEEN.fetch_add(1, Ordering::Relaxed) + 1 == n).then(|| { let row = row.min(rows - 1);
        eprintln!("test_hook forced routing rejection verdict={n} row={row} rows={rows}"); row }) }
/// Clear once; only an armed, matching hook adds a row write. No upload/readback.
pub fn clear(rows: usize, rank: Option<usize>, mut set: impl FnMut(Option<usize>) -> Result<(), String>) -> Result<(), String> {
    set(None)?; if let Some(row) = forced_routing_rejection(rows, rank) { set(Some(row))?; } Ok(()) }
pub fn rejected_rows(status: &[f32]) -> Vec<usize> { status.iter().enumerate().filter_map(|(i,v)| (v.to_bits()!=0).then_some(i)).collect() }
