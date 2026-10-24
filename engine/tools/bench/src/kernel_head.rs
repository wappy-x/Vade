//! Vocabulary fixtures and production M16/experimental M32 slice launches.
use super::{Case, Fixture, Gpu, H, Result, check, params, ptr};
pub(super) fn prepare(g: &mut Gpu, module: H, c: &Case, v: &mut Fixture) -> Result<()> {
if c.kind == 1 && c.real {
    // Encode the same lossless fixture in place for both arms, outside timing.
    let encoder = g.function(module, "flash_next_head_lbf12_encode")?;
    for &weight in &v.ring {
        let mut flags = g.alloc(c.n / 8 * 16)?;
        unsafe { check((g.d.memset)(flags, 0, c.n / 8 * 16))?; }
        let mut bases = g.alloc(c.n * c.k / 32)?;
        let mut p = params::FlashBf16Pack { source: 0, weight, rows: c.n as u32, cols: c.k as u32 };
        g.launch(encoder, [(c.n * c.k / 8).div_ceil(128) as u32, 1, 1], 128, 0,
            &mut [ptr(&mut p), ptr(&mut flags), ptr(&mut bases)])?;
        v.ring_x.push(flags);
        v.scales_x.push(bases);
    }
    g.sync()?;
    // One warp encodes 8 rows x K32, hence N*K/8 threads. The fixture's
    // exponent span fits every slot, so verify all expected bits were produced.
    for &flags in &v.ring_x {
        let raw = g.read(flags, c.n / 8 * 16)?;
        let encoded: usize = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()).count_ones() as usize).sum();
        if encoded != c.n * c.k / 256 { return Err(format!("{} encoded {encoded} slots, expected {}", c.name, c.n * c.k / 256)); }
        eprintln!("head LBF12 coverage case={} encoded_slots={encoded} expected_slots={}", c.name, c.n * c.k / 256);
    }
}
    Ok(())
}
pub(super) fn launch(g: &Gpu, f: H, c: &Case, v: &mut Fixture, i: usize, arm: usize) -> Result<()> {
    // Production main uses M16 slices. HEAD32 shares each fragment
    // across all of a wider slice, preserving each row's K graph.
    let wide = arm == 1 && std::env::var("KBENCH_XKERNEL").is_ok_and(|n| n.starts_with("flash_next_head32"));
    for start in (0..c.m).step_by(if wide { 32 } else { 16 }) {
        let mut p = v.rows;
        p.weight = v.ring[i];
        p.tokens = (c.m - start).min(if wide { 32 } else { 16 }) as u32;
        p.input += (start / 8 * (c.k / 16) * 32 * 16) as u64;
        p.output += (start * c.n * 4) as u64;
        let mut partial = v.aux + (start * 192 * 3 * 4) as u64;
        let mut args = vec![ptr(&mut p), ptr(&mut partial)];
        let (mut flags, mut bases) = if c.real { (v.ring_x[i], v.scales_x[i]) } else { (0, 0) };
        if c.real { args.extend([ptr(&mut flags), ptr(&mut bases)]); }
        g.launch(f, v.grid, v.threads, v.shared, &mut args)?;
    }
    Ok(())
}
