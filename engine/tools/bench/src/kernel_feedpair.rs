//! FEED-PAIR: the decode feed kernel `flash_next_rows_feed` of two images on the same inputs, byte for byte, and
//! against the host arithmetic (`gpu/src/frontend_math.rs`).
//! Run: kernel-bench --mx-tc <old ptx> <new ptx> feed-pair
//!
//! Use: the device check of this kernel when the code list names it DIFFER though its source did not change (the
//! compiler gives a kernel other machine code when another kernel of the image changes). No other check launches it.
//!
//! The kernel writes, for every row of a decode group: the sixteen hashed table rows of the token (from the token,
//! two history values and 35 hash parameters) and the token's embedding as FP32, four times across the row. A row
//! whose slot id is 0 gets zeros and sixteen all-ones row numbers.
//! Inputs: widths 1 to 64; tokens from the first 4,096 and the last 16 of the table; histories that are token ids,
//! the start sentinel 248044, or any 64-bit value; any multipliers, moduli from 1 to 2^31 and offsets below 2^40;
//! slot ids 0 to 16. Both output buffers start from the same filler, so a word a kernel leaves out shows.
//! Control: one token changed in the new image's input must change that row and no other.
#![forbid(unsafe_code)]
use super::{Gpu, H, Result, sha256};
#[allow(dead_code)]
#[path = "../../../crates/gpu/src/frontend_math.rs"] mod frontend_math;

const W: usize = 10240;
const HIDDEN: usize = 2560;
const TABLE: usize = 248320;
const LOW: usize = 4096;
const HIGH: usize = 16;
const SENTINEL: i64 = 248044;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: u64) -> u64 { (self.next() >> 11) % n }
}
/// The kernel's by-value step: `gpu/src/next_math.rs` `SlotStep { slots: u64, table: u64, layer: u32, rows: u32 }`.
fn step(slots: u64, rows: usize) -> [u8; 24] {
    let mut out = [0; 24];
    out[..8].copy_from_slice(&slots.to_le_bytes());
    out[20..].copy_from_slice(&(rows as u32).to_le_bytes());
    out
}
struct Buffers { input: u64, params: u64, slots: u64, embed: u64, rows: [u64; 2], out: [u64; 2] }
fn launch(g: &Gpu, f: H, v: &Buffers, arm: usize, width: usize) -> Result<()> {
    let (mut input, mut params, mut rows, mut embed, mut out, mut step) = (v.input, v.params, v.rows[arm], v.embed, v.out[arm], step(v.slots, width));
    g.launch(f, [(width * W).div_ceil(128) as u32, 1, 1], 128, 0, &mut [(&mut input as *mut u64).cast(), (&mut params as *mut u64).cast(), (&mut rows as *mut u64).cast(), (&mut embed as *mut u64).cast(), (&mut out as *mut u64).cast(), step.as_mut_ptr().cast()])
}
fn le64(values: &[i64]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// Rows (of `row` bytes each) in which two buffers differ.
fn rows_that_differ(a: &[u8], b: &[u8], row: usize) -> Vec<usize> { a.chunks(row).zip(b.chunks(row)).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect() }

pub(crate) fn run(old_path: &str, new_path: &str) -> Result<()> {
    let (a, b) = (std::fs::read(old_path).map_err(|e| e.to_string())?, std::fs::read(new_path).map_err(|e| e.to_string())?);
    println!("FEED PAIR old_sha256={} new_sha256={} kernel=flash_next_rows_feed", sha256::digest(&a), sha256::digest(&b));
    let mut g = Gpu::new()?;
    let (ma, mb) = (g.module(&a)?, g.module(&b)?);
    let entries = [g.function(ma, "flash_next_rows_feed")?, g.function(mb, "flash_next_rows_feed")?];
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    // The whole table is allocated (the kernel indexes it by token id); only the rows the tokens can name are filled.
    let embed = g.alloc(TABLE * HIDDEN * 2)?;
    let low: Vec<u8> = (0..LOW * HIDDEN * 2).map(|_| r.next() as u8).collect();
    let high: Vec<u8> = (0..HIGH * HIDDEN * 2).map(|_| r.next() as u8).collect();
    g.upload_at(embed, &low)?;
    g.upload_at(embed + ((TABLE - HIGH) * HIDDEN * 2) as u64, &high)?;
    let table = |token: usize, col: usize| -> u16 {
        let (data, at) = if token < LOW { (&low, token) } else { (&high, token - (TABLE - HIGH)) };
        u16::from_le_bytes([data[(at * HIDDEN + col) * 2], data[(at * HIDDEN + col) * 2 + 1]])
    };
    let (mut pass, mut words, mut cases) = (true, 0usize, 0usize);
    for width in [1usize, 2, 3, 5, 8, 16, 17, 31, 64] {
        for round in 0..6 {
            let mut params = [0i64; 35];
            for (i, p) in params.iter_mut().enumerate() { *p = if i < 3 { r.next() as i64 } else if i < 19 { 1 + r.below(1 << 31) as i64 } else { r.below(1 << 40) as i64 }; }
            let token = |r: &mut Rng| -> i64 { match r.below(8) { 0 => 0, 1 => (TABLE - 1) as i64, 2 => (TABLE - HIGH) as i64 + r.below(HIGH as u64) as i64, _ => r.below(LOW as u64) as i64 } };
            let history = |r: &mut Rng| -> i64 { match r.below(4) { 0 => SENTINEL, 1 => r.next() as i64, _ => r.below(TABLE as u64) as i64 } };
            let mut input: Vec<[i64; 3]> = (0..width).map(|_| [token(&mut r), history(&mut r), history(&mut r)]).collect();
            // Round 0: every slot selected. Later rounds: about one row in four is not.
            let slots: Vec<u32> = (0..width).map(|_| if round > 0 && r.below(4) == 0 { 0 } else { 1 + r.below(16) as u32 }).collect();
            let filler = |bytes: usize| vec![0xa5u8; bytes];
            let v = Buffers {
                input: g.upload(&le64(&input.concat()))?, params: g.upload(&le64(&params))?,
                slots: g.upload(&slots.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>())?, embed,
                rows: [g.upload(&filler(width * 128))?, g.upload(&filler(width * 128))?], out: [g.upload(&filler(width * W * 4))?, g.upload(&filler(width * W * 4))?],
            };
            for (arm, entry) in entries.iter().enumerate() { launch(&g, *entry, &v, arm, width)?; }
            g.sync()?;
            let got: Vec<(Vec<u8>, Vec<u8>)> = (0..2).map(|arm| Ok((g.read(v.rows[arm], width * 128)?, g.read(v.out[arm], width * W * 4)?))).collect::<Result<_>>()?;
            // The host arithmetic.
            let (mut want_rows, mut want_out) = (Vec::with_capacity(width * 128), Vec::with_capacity(width * W * 4));
            for (row, [token, h0, h1]) in input.iter().enumerate() {
                if slots[row] == 0 {
                    want_rows.extend([0xffu8; 128]);
                    want_out.extend(std::iter::repeat_n(0u8, W * 4));
                } else {
                    for hashed in frontend_math::rows(*token as u32, &mut [*h0, *h1], &params) { want_rows.extend(hashed.to_le_bytes()); }
                    for col in 0..W { want_out.extend(((table(*token as usize, col % HIDDEN) as u32) << 16).to_le_bytes()); }
                }
            }
            let differ = [rows_that_differ(&got[0].0, &got[1].0, 128).len(), rows_that_differ(&got[0].1, &got[1].1, W * 4).len(), rows_that_differ(&got[1].0, &want_rows, 128).len(), rows_that_differ(&got[1].1, &want_out, W * 4).len()];
            // Control: another token in one selected row, on the new image only.
            let target = (0..width).map(|i| (i + width / 2) % width).find(|&i| slots[i] != 0).unwrap_or(0);
            input[target][0] = if input[target][0] == 1 { 2 } else { 1 };
            g.upload_at(v.input, &le64(&input.concat()))?;
            launch(&g, entries[1], &v, 1, width)?;
            g.sync()?;
            let moved = (rows_that_differ(&g.read(v.rows[1], width * 128)?, &got[1].0, 128), rows_that_differ(&g.read(v.out[1], width * W * 4)?, &got[1].1, W * 4));
            let control = slots[target] == 0 || moved.0 == [target] && moved.1 == [target];
            let ok = differ == [0; 4] && control;
            pass &= ok;
            words += width * (W + 32);
            cases += 1;
            println!("FEED PAIR {} width={width} round={round} rows_selected={} rows_old_ne_new={} embedding_rows_old_ne_new={} rows_new_ne_host={} embedding_rows_new_ne_host={} control_row={target} control_moved_rows={:?} control_moved_embedding_rows={:?}", if ok { "ok" } else { "FAIL" }, slots.iter().filter(|&&s| s != 0).count(), differ[0], differ[1], differ[2], differ[3], moved.0, moved.1);
        }
    }
    println!("FEED PAIR {} cases={cases} words_compared={words} (old image = new image = host arithmetic in every case; every control moved its row alone)", if pass { "PASS" } else { "FAIL" });
    if pass { Ok(()) } else { Err("FEED PAIR FAIL".into()) }
}
