//! Header-only mixed Flash inventory and world-1 memory receipt.
use engine_model::flash::mixed::Plan;
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 { return Err("usage: flash-storage <config.json> <tensor-map.json>".into()); }
    let p = Plan::open(Path::new(&a[1]), Path::new(&a[2]), 1)?;
    println!("world=1 mapped={}/{} layers={} mtp={} PLE={:?}", p.model.shape.tensors.len(), p.files.len(),
        p.model.shape.layers.len(), p.model.shape.draft.len(), p.model.shape.lexical);
    println!("class tensors bytes GiB");
    for (class, total) in &p.classes { println!("{class:?} {} {} {:.6}", total.tensors, total.bytes, total.bytes as f64 / (1u64 << 30) as f64); }
    // Explicit planning reserve, not a measurement of runtime workspace.
    let (budget, reserve) = (110_000_000_000u64, 4u64 << 30);
    println!("budget_bytes={budget} workspace_reserve_bytes={reserve}; FP32 KV+index; one session; no prefix snapshots");
    println!("PLE MTP context resident_bytes kv_bytes recurrent_bytes total_bytes headroom_bytes fits");
    for ple in [true, false] { for mtp in [false, true] { for tokens in [32768, 131072, 262144] {
        let resident = p.resident_bytes(ple, mtp);
        let (kv, recurrent) = p.cache_bytes(tokens, mtp)?;
        let total = resident + kv + recurrent + reserve;
        println!("{ple} {mtp} {tokens} {resident} {kv} {recurrent} {total} {} {}", budget as i128 - total as i128, total <= budget);
    } } }
    let row = p.ple_row(0, 0)?;
    println!("PLE disk hook: {} packed bytes + {} scale bytes per row, shared FP32 global; {} files retained",
        row.packed_bytes, row.scale_bytes, p.files.values().collect::<std::collections::BTreeSet<_>>().len());
    println!("Next EP/TP (no MTP/vision; embedding+PLE on disk): world rank resident disk experts kv32k state");
    for world in [1, 2, 4] { for rank in 0..world {
        let shard = engine_model::flash::shard::ShardPlan::new(&p.model.shape, &p.files, rank, world)?;
        let (kv, state) = shard.cache_bytes(32768)?;
        println!("{world} {rank} {} {} {} {kv} {state}", shard.budget.resident, shard.budget.disk, shard.budget.experts);
    } }
    Ok(())
}
