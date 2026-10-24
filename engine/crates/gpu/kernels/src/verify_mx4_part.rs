//! VADE_VERIFY_MX4_PART=1 (default on): cache one canonical input range.
//! First-party partitioned implementation of exact MXFP8 verification projections.
//! Same BF16 bits, absolute K16 chains, and (p0+p1)+(p2+p3) merge.
//! Host supplies derived shared bytes; each CTA follows uniform range barriers, including idle warps.
use super::*;

/// Compact image of one absolute canonical K range ([tile][half][lane] 8 bytes; H = 2
/// adds tokens 4-7 256 bytes on, cta_input8's layout). Every thread follows the same
/// part loop, including warps whose final task is absent; the CTA barriers are uniform.
#[inline(always)]
unsafe fn cache_part<const H: usize>(p: Bf16Rows, at: u32, part: usize) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    let (first, end) = canonical::range(p.cols as usize / 16, part);
    let mut c = _thread_idx_x() as usize;
    while c < (end - first) * 32 * H {
        let (t, h, l) = (first + c / (32 * H), c / 32 % H, c % 32);
        let src = (p.input as *const u8).add(t * 512 + (h * 16 + l / 8 * 4 + l % 4) * 16
            + (l / 4 % 2) * p.input_plane_words as usize * 4);
        ptx::copy4(at + (c * 8) as u32, src.cast());
        ptx::copy4(at + (c * 8 + 4) as u32, src.add(8).cast());
        c += _block_dim_x() as usize;
    }
    ptx::commit(); ptx::wait(); ptx::sync();
}

/// Same REDUCTION-PLAN as the full cache. Only one canonical input range is resident:
/// 19,456 ring + 10,240 input bytes at K2560. All warps advance a CTA wave together.
/// # Safety
/// M3/4, all matrices MXFP8, K<=2560 and K%256==0, total rows>4096.
/// Grid192/block128; dynamic allocation is VERIFY_MX4_PART_SHARED (29,696B).
/// Other shapes return; they must never dispatch here.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_part_group(g: crate::kernel_params::Bf16Pairs, _partials: u64) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let m = group(g);
    let total: usize = (0..m.count).map(|j| rows_of(&m, j)).sum();
    if !(3..=4).contains(&m.a.tokens) || m.a.cols > 2560 || m.a.cols % 256 != 0
        || crate::kernel_params::latency_split(total, true) {
        return;
    }
    for j in 0..m.count { if pick(&m, j).weight & 1 == 0 { return; } }
    let groups: usize = (0..m.count).map(|j| rows_of(&m, j).div_ceil(8)).sum();
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let cin = (4 * crate::kernel_params::LATENCY_W8_RING) as u32;
    let sh = (warp * crate::kernel_params::LATENCY_W8_RING) as u32;
    let mut base = _block_idx_x() as usize * 4;
    while base < groups {
        let task = base + warp;
        let valid = task < groups;
        let (mut j, mut first) = (0, 0);
        while j < m.count && task >= first + rows_of(&m, j).div_ceil(8) {
            first += rows_of(&m, j).div_ceil(8); j += 1;
        }
        let p = if valid { pick(&m, j) } else { m.a };
        let row = if valid { (task - first) * 8 } else { 0 };
        let (mut left, mut right) = (ZERO, ZERO);
        for part in 0..4 {
            cache_part::<1>(m.a, cin, part);
            if valid {
                let z = if p.weight & 1 != 0 { cached_range::<9, 8, true, false, 1>(p, sh, row, part, lane, cin) }
                    else { pick_range::<16, 6, 5, 9, 8, 0, 0>(p, sh, row, part, lane) };
                match part { 0 => left = z, 1 => left = canonical::add(left, z),
                    2 => right = z, _ => right = canonical::add(right, z) }
            }
            ptx::sync(); // All range consumers finish before the next image overwrites it.
        }
        if valid { write(p, canonical::add(left, right), lane / 4, row + lane % 4 * 2); }
        base += 192 * 4;
    }
}
#[no_mangle]
pub static FLASH_VERIFY_MX4_PART_GROUP_SHARED: u32 = crate::kernel_params::VERIFY_MX4_PART_SHARED as u32;

/// M5-8 twin of flash_verify_mx4_part_group (H = 2): one canonical range of tokens 0-7
/// resident (K2560: 40 tiles x 512 bytes), range_w8's operands and merge, so each output
/// keeps the bytes of main's M5-8 entry (wide8) without its whole-input CTA copy.
/// # Safety
/// Every present matrix MXFP8, M5-8, K<=2560 and K%256==0, total rows>4096.
/// Grid192/block128; dynamic allocation is VERIFY_MX4_PART8_SHARED (39,936B).
/// Other shapes return; they must never dispatch here.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_part_group8(g: crate::kernel_params::Bf16Pairs, _partials: u64) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let m = group(g);
    let total: usize = (0..m.count).map(|j| rows_of(&m, j)).sum();
    if !(5..=8).contains(&m.a.tokens) || m.a.cols > 2560 || m.a.cols % 256 != 0
        || crate::kernel_params::latency_split(total, true) {
        return;
    }
    for j in 0..m.count { if rows_of(&m, j) != 0 && pick(&m, j).weight & 1 == 0 { return; } }
    let groups: usize = (0..m.count).map(|j| rows_of(&m, j).div_ceil(8)).sum();
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let cin = (4 * crate::kernel_params::LATENCY_W8_RING) as u32;
    let sh = (warp * crate::kernel_params::LATENCY_W8_RING) as u32;
    let mut base = _block_idx_x() as usize * 4;
    while base < groups {
        let task = base + warp;
        let valid = task < groups;
        let (mut j, mut first) = (0, 0);
        while j < m.count && task >= first + rows_of(&m, j).div_ceil(8) {
            first += rows_of(&m, j).div_ceil(8); j += 1;
        }
        let p = if valid { pick(&m, j) } else { m.a };
        let row = if valid { (task - first) * 8 } else { 0 };
        let (mut left, mut right) = (ZERO, ZERO);
        for part in 0..4 {
            cache_part::<2>(m.a, cin, part);
            if valid {
                let z = cached_range::<9, 8, true, false, 2>(p, sh, row, part, lane, cin);
                match part { 0 => left = z, 1 => left = canonical::add(left, z),
                    2 => right = z, _ => right = canonical::add(right, z) }
            }
            ptx::sync(); // All range consumers finish before the next image overwrites it.
        }
        if valid { write(p, canonical::add(left, right), lane / 4, row + lane % 4 * 2); }
        base += 192 * 4;
    }
}
#[no_mangle]
pub static FLASH_VERIFY_MX4_PART_GROUP8_SHARED: u32 = crate::kernel_params::VERIFY_MX4_PART8_SHARED as u32;
