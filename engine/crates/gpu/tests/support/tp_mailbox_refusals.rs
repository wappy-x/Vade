//! Shared CUDA refusal checks for the mailbox integration test.
use std::sync::atomic::Ordering;
use spark_gpu_probe::{driver::{with_backend, Backend, DeviceVector, TpMailbox}, layer_ptx,
    tp_mailbox_host::{TpMailboxHost, SEND_LEN, SEND_SEQ}};

fn words(host: &TpMailboxHost<'_>) -> Vec<u32> {
    host.words().iter().map(|w| w.load(Ordering::Acquire)).collect()
}
fn seed(b: &Backend<'_>, v: &mut DeviceVector<'_>, tag: u32) -> Result<Vec<u32>, String> {
    let bits: Vec<_> = (0..v.len()).map(|i| tag ^ (i as u32).rotate_left(11)).collect();
    b.write_vector(v, &bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>())?;
    Ok(bits)
}
fn unchanged(b: &Backend<'_>, m: &TpMailbox<'_>, host: &TpMailboxHost<'_>, before: &[u32],
    vectors: &[(&DeviceVector<'_>, &[u32])], label: &str) -> Result<(), String> {
    b.synchronize()?;
    if m.failed() || words(host) != before { return Err(format!("{label}: refusal changed mailbox words or sequence state")); }
    for &(v, bits) in vectors {
        let after: Vec<_> = b.read_vector_range(v, 0, v.len())?.into_iter().map(f32::to_bits).collect();
        if after != bits { return Err(format!("{label}: refusal changed vector words")); }
    }
    Ok(())
}
pub fn run() -> Result<(), String> {
    with_backend(0, layer_ptx()?, usize::MAX, |a| {
        let own = a.tp_mailbox(64)?;
        let host = own.take_host().ok_or("own host missing")?;
        let before = words(&host);
        let mut arena = a.vector_arena(&[16])?;
        let mut rows = arena.pop().ok_or("arena vector missing")?;
        let mut second = a.vector(16)?;
        let row_bits = seed(a, &mut rows, 0xdead_beef)?;
        let second_bits = seed(a, &mut second, 0x1234_5678)?;
        let mut cases = 0;
        with_backend(0, layer_ptx()?, usize::MAX, |b| {
            let other = b.tp_mailbox(64)?;
            let other_host = other.take_host().ok_or("other host missing")?;
            let other_before = words(&other_host);
            let mut foreign = b.vector(16)?;
            let mut foreign_second = b.vector(16)?;
            let foreign_bits = seed(b, &mut foreign, 0x7fc0_0123)?;
            let foreign_second_bits = seed(b, &mut foreign_second, 0x8000_0000)?;
            macro_rules! refuse {
                ($call:expr, $reason:expr, $label:expr) => {{
                    let error = $call.err().ok_or_else(|| format!("{}: launch was not refused", $label))?;
                    if !error.contains($reason) { return Err(format!("{}: wrong refusal: {error}", $label)); }
                    unchanged(a, &own, &host, &before, &[(&rows, &row_bits), (&second, &second_bits)], $label)?;
                    unchanged(b, &other, &other_host, &other_before,
                        &[(&foreign, &foreign_bits), (&foreign_second, &foreign_second_bits)], $label)?;
                    cases += 1;
                }};
            }
            let owner = "belong to another backend";
            refuse!(a.tp_mailbox_put(&other, &rows, 1, 16, 8, 0, 1), owner, "scalar put mailbox owner");
            refuse!(a.tp_mailbox_take(&other, &mut rows, 1, 16, 8, 8, 1, 1_000), owner, "scalar take mailbox owner");
            refuse!(a.tp_mailbox_put(&own, &foreign, 1, 16, 8, 0, 1), owner, "scalar put source owner");
            refuse!(a.tp_mailbox_take(&own, &mut foreign, 1, 16, 8, 8, 1, 1_000), owner, "scalar take target owner");
            refuse!(a.tp_mailbox_put_pair(&other, [(&rows, 16, 4, 0), (&second, 16, 4, 0)], 1, 1), owner, "pair put mailbox owner");
            refuse!(a.tp_mailbox_put_pair(&own, [(&rows, 16, 4, 0), (&foreign, 16, 4, 0)], 1, 1), owner, "pair put source owner");
            refuse!(a.tp_mailbox_put_pair(&own, [(&foreign, 16, 4, 0), (&second, 16, 4, 0)], 1, 1), owner, "pair put first source owner");
            refuse!(a.tp_mailbox_take_pair_from(&own, [(&mut foreign, 16, 4, 0), (&mut foreign_second, 16, 4, 0)], 1, 0, 1, 1_000), owner, "pair take target owner");
            refuse!(a.tp_mailbox_take_pair_from(&other, [(&mut rows, 16, 4, 0), (&mut second, 16, 4, 0)], 1, 0, 1, 1_000), owner, "pair take mailbox owner");
            // These valid arena views overlap. Shape and owner checks must pass before overlap is refused.
            let (mut x, mut y) = (rows.slice(0..8)?, rows.slice(7..15)?);
            refuse!(a.tp_mailbox_take_pair_from(&own, [(&mut x, 8, 4, 0), (&mut y, 8, 4, 0)], 1, 0, 1, 1_000), "targets overlap", "pair take overlap");
            // Check each part's bounds and the combined slot extent in both wrappers.
            for (xw, yw, xc, yc) in [(17, 16, 4, 4), (16, 17, 4, 4), (16, 16, 8, 9)] {
                refuse!(a.tp_mailbox_put_pair(&own, [(&rows, xw, xc, 0), (&second, yw, yc, 0)], 1, 1),
                    "outside the slot or the vectors", "pair put extent");
                refuse!(a.tp_mailbox_take_pair_from(&own, [(&mut rows, xw, xc, 0), (&mut second, yw, yc, 0)], 1, 0, 1, 1_000),
                    "outside the slot or the vectors", "pair take extent");
            }
            refuse!(a.tp_mailbox_take_from(&own, &mut rows, 1, 16, 8, 8, own.layout().world - 1, 1, 1_000),
                "channel outside the layout", "scalar take channel");
            refuse!(a.tp_mailbox_take_pair_from(&own, [(&mut rows, 16, 4, 0), (&mut second, 16, 4, 0)], 1, own.layout().world - 1, 1, 1_000),
                "channel outside the layout", "pair take channel");
            Ok(())
        })?;
        // A valid scalar put and a valid paired put still publish after every refusal.
        a.tp_mailbox_put(&own, &rows, 1, 16, 8, 0, 1)?;
        a.tp_mailbox_put_pair(&own, [(&rows, 16, 4, 0), (&second, 16, 4, 0)], 1, 2)?;
        a.synchronize()?;
        for (seq, bits) in [(1, row_bits[..8].to_vec()), (2, row_bits[..4].iter().chain(&second_bits[..4]).copied().collect())] {
            let l = own.layout();
            let slot = seq % l.slots;
            let after = words(&host);
            let start = l.send_slot(slot) + l.frame;
            if own.failed() || after[SEND_SEQ + slot] != seq as u32 || after[SEND_LEN + slot] != (bits.len() * 4) as u32
                || after[start..start + bits.len()] != bits {
                return Err("valid put after refusals did not publish its bytes and sequence".into());
            }
        }
        println!("PASS mailbox refusals={cases}; arena overlap; vector and mailbox words unchanged after each refusal; scalar and pair puts publish");
        Ok(())
    })
}
