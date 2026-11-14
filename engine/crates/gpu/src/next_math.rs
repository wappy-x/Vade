//! Per-output arithmetic shared by the Flash-Next row kernels and host probes.
#[inline(always)]
pub fn bf16(x: f32) -> u32 { x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) >> 16 }
#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")] { core::f32::math::mul_add(a, b, c) }
    #[cfg(not(target_arch = "nvptx64"))] { a.mul_add(b, c) }
}
#[inline(always)]
pub fn exp(t: f32) -> f32 {
    let t = if t > 87.0 { 87.0 } else if t < -87.0 { -87.0 } else { t };
    let n = t * core::f32::consts::LOG2_E + 12_582_912.0 - 12_582_912.0;
    let r = madd(-n, 0.693145751953125, t);
    let r = madd(-n, 0.0000014286068, r);
    let mut p = 1.0 / 5040.0f32;
    for c in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] { p = madd(p, r, c); }
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}
/// Inputs are packed BF16; weights are unchanged nibbles and expanded E4M3 scales.
/// # Safety
/// All pointers hold one complete input/weight/scale row, with cols divisible by 16.
#[inline(always)]
pub unsafe fn project(input: *const u32, weights: *const u8, scales: *const u8, global: f32, cols: usize) -> f32 {
    let mut sum = 0.0;
    for k in 0..cols {
        let word = unsafe { *input.add(k / 2) };
        let x = f32::from_bits(((if k % 2 == 0 { word } else { word >> 16 }) & 0xffff) << 16);
        let byte = unsafe { *weights.add(k / 2) } as u32;
        let nibble = if k % 2 == 0 { byte } else { byte >> 4 };
        let m = nibble & 7;
        let bits = (m + 252 - u32::from(m == 1)) & 0u32.wrapping_sub(u32::from(m != 0));
        let w = f32::from_bits((bits << 22) | ((nibble & 8) << 28));
        let s = unsafe { *scales.add(k / 16) } as u32;
        let (e, f) = ((s >> 3) & 15, s & 7);
        let sign = (s & 0x80) << 24;
        let scale = if e == 0 { f32::from_bits((f as f32 / 512.0).to_bits() | sign) }
            else if e == 15 && f == 7 { f32::from_bits(sign | 0x7fc00000) }
            else { f32::from_bits(sign | ((e + 120) << 23) | (f << 20)) };
        sum = madd(x, (w * scale) * global, sum);
    }
    sum
}
/// Ordered expert contributions, with no cross-row reduction or split K.
#[inline(always)]
pub fn accumulate(sum: f32, value: f32, weight: f32) -> f32 { sum + value * weight }

/// Rows execution folds BF16 down values with FP32 route weights by FMA.
#[inline(always)]
pub fn routed_add(sum: f32,value: f32,weight: f32) -> f32 { madd(value,weight,sum) }

/// Token-major ordered recurrence storage; the old chunk solve is never used.
pub fn gdn_lengths(hidden: usize, heads: usize, keys: usize, rows: usize) -> [usize; 20] {
    let hd = heads*128; let width = (2*keys+heads)*128;
    [width*rows, hd*rows, heads*rows, heads*rows, 1, hd*rows, hd*rows, hd*rows,
        heads*rows, heads*rows, hd*rows, hd*rows, hidden*rows, 128, 1, 1, 1, 1, 1, 1]
}
/// One recurrent value column. Constant indices retain the 128 cells in registers.
/// All prompt chunks and single tokens visit exactly the same two ascending sums.
#[inline(always)]
pub fn column(cells: &mut [f32; 128], q: &[f32], k: &[f32], v: f32, beta: f32, decay: f32) -> f32 {
    macro_rules! eight { ($i:ident, $body:block) => {
        { const $i: usize = 0; $body } { const $i: usize = 1; $body }
        { const $i: usize = 2; $body } { const $i: usize = 3; $body }
        { const $i: usize = 4; $body } { const $i: usize = 5; $body }
        { const $i: usize = 6; $body } { const $i: usize = 7; $body }
    }; }
    let mut prediction = 0.0;
    eight!(G, { eight!(P, {
        const I: usize = G*16+P*2;
        cells[I] *= decay; prediction += k[I]*cells[I];
        cells[I+1] *= decay; prediction += k[I+1]*cells[I+1];
    }); });
    let delta = (v-prediction)*beta;
    let mut value = 0.0;
    eight!(G, { eight!(P, {
        const I: usize = G*16+P*2;
        cells[I] += k[I]*delta; value += q[I]*cells[I];
        cells[I+1] += k[I+1]*delta; value += q[I+1]*cells[I+1];
    }); });
    value
}
#[inline(always)]
pub fn hc_mix(x: [f32; 4], gates: [f32; 4], sigmoid: impl Fn(f32) -> f32) -> f32 {
    let mut value = 0.0;
    for stream in 0..4 { value = hc_stream(value,x[stream],gates[stream],&sigmoid); }
    value*0.25
}
#[inline(always)]
pub fn hc_stream(sum: f32,x: f32,gate: f32,sigmoid: impl Fn(f32) -> f32) -> f32 { sum+x*sigmoid(gate) }
#[inline(always)]
pub fn hc_add(residual: f32, gate: f32, block: f32, sigmoid: impl Fn(f32) -> f32) -> f32 {
    residual + (2.0*sigmoid(gate*0.25))*block
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn next_rows_chunk_and_batch_math_is_bitwise_invariant() {
        assert_eq!(core::mem::size_of::<StateSlot>(),1168);
        assert_eq!(core::mem::size_of::<SlotStep>(),24);
        assert_eq!(core::mem::offset_of!(StateSlot,position),1160);
        // Every 64-row source tile is an in-place-safe permutation; each
        // aligned fragment pair represents the same row and consecutive K16s.
        let mut seen = vec![false; 64*2560];
        for row in 0..64 { for col in 0..2560 { let at=head_weight(row,col,2560);
            assert!(!seen[at]); seen[at]=true;
            assert_eq!(at/8, (row/8*80+col/32)*32+row%8*4+col%8/2);
        } }
        assert!(seen.iter().all(|v| *v));
        assert_eq!(better((0.0,9),(-0.0,2)).1,2);
        assert_eq!(better((7.0,9),(7.0,2)).1,2);
        assert_eq!(better((7.0,2),(f32::NAN,0)).1,2);
        let rounded=|x| f32::from_bits(bf16(x)<<16);
        let finish=|token: usize| crate::fp8_abi::FinishLayout::FLASH.routed_sum(true,
            |slot| (token as f32*0.001-0.5)*(slot as f32+0.37),
            |slot| (slot as f32+1.25)/65.0, rounded).to_bits();
        let expected: Vec<_>=(0..513).map(finish).collect();
        for chunk in [1,2,7,16,63,64,65,127,512] {
            let got: Vec<_>=(0..513).collect::<Vec<_>>().chunks(chunk).flat_map(|c| c.iter().map(|&i| finish(i))).collect();
            assert_eq!(got,expected);
        }
        for width in 1..=16 { for group in (0..513).collect::<Vec<_>>().chunks(width) {
            for &i in group.iter().rev() { assert_eq!(finish(i),expected[i]); }
        } }
        for rows in [1, 16, 64, 65, 512] { let l = gdn_lengths(2560, 48, 16, rows);
            assert_eq!((l[5], l[7], l[12], l[15]), (rows*6144, rows*6144, rows*2560, 1)); }
        let run = |chunk: usize, order: &[usize]| {
            let mut states = [[0.0; 128]; 16]; let mut out = vec![Vec::new(); 16];
            for span in (0..129).collect::<Vec<_>>().chunks(chunk) { for &chat in order { for &t in span {
                let q = [0.015625*(chat+1) as f32; 128]; let k = [0.03125; 128];
                let value = column(&mut states[chat], &q, &k, (t as f32-64.0)/128.0, 0.25, 0.9921875);
                let x = [value, -value, 0.5, -0.25]; let gates = [0.5, -0.5, 0.25, -0.25];
                let sig = |v: f32| 1.0/(1.0+exp(-v));
                let mixed = hc_mix(x, gates, sig); let residual = hc_add(value, gates[0], mixed, sig);
                let packed = vec![bf16(residual) | (bf16(mixed)<<16); 16];
                let weights = [0x21; 16]; let scales = [56; 2];
                let expert = unsafe { project(packed.as_ptr(), weights.as_ptr(), scales.as_ptr(), 0.5, 32) };
                let mut sum = 0.0;
                for i in 0..10 { sum = accumulate(sum, expert*(i+1) as f32, 0.125); }
                out[chat].push((value.to_bits(), residual.to_bits(), sum.to_bits()));
            } } }
            (states.map(|s| s.map(f32::to_bits)), out)
        };
        let order: Vec<_> = (0..16).collect(); let expected = run(1, &order);
        for chunk in [2, 7, 16, 63, 64, 65, 127, 512] { assert_eq!(run(chunk, &order), expected); }
        for batch in [1, 2, 8, 16] { for rows in order.chunks(batch) {
            let (_, out) = run(17, &rows.iter().rev().copied().collect::<Vec<_>>());
            for &row in rows { assert_eq!(out[row], expected.1[row]); }
        } }
    }
}

/// Resident head layout: one aligned 128-bit load contains two K16 fragments.
#[inline(always)]
pub fn head_weight(row: usize, col: usize, width: usize) -> usize {
    (((row/8*(width/32)+col/32)*32+row%8*4+col%8/2)*8)
        + col%32/16*4+col%2+col%16/8*2
}
#[inline(always)]
pub fn better(a: (f32, u32), b: (f32, u32)) -> (f32, u32) {
    if b.0 > a.0 || b.0 == a.0 && b.1 < a.1 { b } else { a }
}

/// Funded state allocations are addressed through slots; zero has no storage.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StateSlot { pub layers: [[u64;3];48], pub ple: u64, pub position: u32, pub capacity: u32 }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlotStep { pub slots: u64, pub table: u64, pub layer: u32, pub rows: u32 }
impl SlotStep {
    /// # Safety
    /// slots holds rows validated IDs in 0..=16; table holds seventeen records.
    #[inline(always)]
    pub unsafe fn selected(self, row: usize) -> Option<usize> {
        if row >= self.rows as usize { return None; }
        let id=unsafe { *(self.slots as *const u32).add(row) } as usize;
        if id==0 || id>16 { None } else { Some(id) }
    }
}
