//! BF16 gate/up decode with rank-sharded resident checkpoint FP8 weights.
use core::arch::{asm, nvptx::{_block_idx_x, _block_idx_z, _thread_idx_x}};
use crate::{fp8_accurate_gate::{mma, promote_pair}, ptx};

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Input([u32; 4]);

#[inline(always)]
fn address(shared: u32, row: usize, col: usize) -> u32 {
    shared + (row * 128 + (col ^ (row * 16))) as u32
}

#[inline(always)]
unsafe fn stage(shared: u32, w: *const u8, k: usize, kb: usize, lane: usize) {
    let (row, col) = (lane / 8, lane % 8 * 16);
    crate::fp8_bf16::copy_weight16(address(shared, row, col), w.add(row * k + kb + col));
    crate::fp8_bf16::copy_weight16(address(shared, row + 4, col), w.add((row + 4) * k + kb + col));
}

#[inline(always)]
unsafe fn fragment(shared: u32, row: usize, col: usize) -> u32 {
    let (lo, hi): (u16, u16);
    asm!("ld.shared.b16 {lo}, [{p}]; ld.shared.b16 {hi}, [{p}+8];",
        lo=out(reg16)lo, hi=out(reg16)hi, p=in(reg32)address(shared,row,col));
    lo as u32 | ((hi as u32) << 16)
}

#[inline(always)]
unsafe fn step(raw: u32, high: [u32; 4], scale: f32, sum: [f32; 4]) -> [f32; 4] {
    mma(high, [promote_pair(raw as u16, scale), promote_pair((raw >> 16) as u16, scale)], sum)
}

#[inline(always)]
unsafe fn tile(shared: u32, w: *const u8, ws: *const u16,
    x: *const u32, k: usize) -> [f32; 4] {
    let tid = _thread_idx_x() as usize % 128;
    let (lane, warp) = (tid % 32, tid / 32);
    let x = x.add(lane * 4);
    let raw_shared = shared + (warp * 4096) as u32;
    let mut sum = [0.0f32; 4];
    let chunks = (k / 128 + 3 - warp) / 4;
    let offset = _block_idx_x() as usize % chunks.max(1);
    let mut iteration = 0;
    let mut kb = offset * 512 + warp * 128;
    if kb < k {
        stage(raw_shared, w, k, kb, lane);
        ptx::commit();
        if chunks > 1 {
            let second = if offset + 1 == chunks { 0 } else { offset + 1 };
            stage(raw_shared + 1024, w, k, second * 512 + warp * 128, lane);
            ptx::commit();
            if chunks > 2 {
                let third = if second + 1 == chunks { 0 } else { second + 1 };
                stage(raw_shared + 2048, w, k, third * 512 + warp * 128, lane);
                ptx::commit();
                asm!("cp.async.wait_group 2;");
            } else { ptx::wait_one(); }
        } else { ptx::wait(); }
        ptx::warp_sync();
    }
    while kb < k {
        let current = raw_shared + ((iteration % 4) * 1024) as u32;
        let next_group = if offset + iteration + 1 >= chunks {
            offset + iteration + 1 - chunks
        } else { offset + iteration + 1 };
        if iteration + 3 < chunks {
            let ahead = if next_group + 1 == chunks { 0 } else { next_group + 1 };
            let ahead = if ahead + 1 == chunks { 0 } else { ahead + 1 };
            stage(raw_shared + (((iteration + 3) % 4) * 1024) as u32,
                w, k, ahead * 512 + warp * 128, lane);
            ptx::commit();
        }
        let scale = f32::from_bits((*ws.add(kb / 128) as u32) << 16);
        macro_rules! raw {
            ($z:expr) => { fragment(current, lane / 4, $z * 16 + lane % 4 * 2) };
        }
        let raw = [raw!(0), raw!(1), raw!(2), raw!(3), raw!(4), raw!(5), raw!(6), raw!(7)];
        macro_rules! part {
            ($z:expr) => {{
                let position = (kb / 16 + $z) * 128;
                let high = (*x.add(position).cast::<Input>()).0;
                sum = step(raw[$z], high, scale, sum);
            }};
        }
        part!(0); part!(1); part!(2); part!(3);
        part!(4); part!(5); part!(6); part!(7);
        if iteration + 3 < chunks { asm!("cp.async.wait_group 2;"); }
        else if iteration + 2 < chunks { ptx::wait_one(); } else { ptx::wait(); }
        ptx::warp_sync();
        iteration += 1;
        if iteration == chunks { break; }
        kb = next_group * 512 + warp * 128;
    }
    let reduction = shared + 16384;
    for component in 0..4 {
        ptx::store(reduction + ((tid * 4 + component) * 4) as u32, sum[component]);
    }
    ptx::sync();
    if warp == 0 {
        for component in 0..4 {
            let p = reduction + ((lane * 4 + component) * 4) as u32;
            sum[component] = (ptx::load(p) + ptx::load(p + 512))
                + (ptx::load(p + 1024) + ptx::load(p + 1536));
        }
    }
    sum
}

/// Routed BF16 gate/up, one task per selected expert. W is a descriptor table.
/// Two four-warp groups run gate and up concurrently in 36,864 shared bytes.
/// Packed input/output and routing metadata follow the grouped FFN ABI.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp8_grouped_gate_decode_accurate(p: crate::fp8_abi::GroupedParams) {
    let shared = ptx::shared::<36864>();
    let (w, x, y) = (p.weights as *const u8, p.input as *const u32, p.output as *mut f32);
    let (counts, starts) = (p.route.counts as *const u32, p.route.starts as *const u32);
    let task_count = p.route.tasks as *const u32;
    let tasks = task_count.add(1);
    let k = p.cols;

    let task = _block_idx_z() as usize;
    if task >= *task_count as usize { return; }
    let expert = *tasks.add(task * 2) as usize;
    let m = *counts.add(expert) as usize;
    let start = *starts.add(expert) as usize;
    let k = k as usize;
    let nb = _block_idx_x() as usize * 8;
    let tid = _thread_idx_x() as usize;
    let group = tid / 128;
    let table = w.cast::<u64>().add(expert * 7);
    let width = *table.add(6) as usize;
    if nb >= width { return; }
    let w = (*table.add(group * 2) as *const u8).add(nb * k);
    let ws = (*table.add(group * 2 + 1) as *const u16).add(nb / 128 * (k / 128));
    let x = x.add(start * k / 2);
    let sum = tile(shared + (group * 18432) as u32, w, ws, x, k);
    if tid < 32 {
        for component in 0..4 {
            ptx::store(shared + ((tid * 4 + component) * 4) as u32, sum[component]);
        }
    }
    ptx::sync();
    if tid >= 128 && tid < 160 {
        let lane = tid - 128;
        let row = lane / 4;
        let col = nb + lane % 4 * 2;
        let gate = shared + (lane * 16) as u32;
        let y = y.cast::<u32>().add(start * 640 / 2);
        if row < m { crate::fp8_bf16::store_activation(y, width, row, col, ptx::load(gate), ptx::load(gate + 4), sum[0], sum[1]); }
        else { crate::fp8_bf16::store_middle_pair(y, width, row, col, 0); }
        if row + 8 < m { crate::fp8_bf16::store_activation(y, width, row + 8, col, ptx::load(gate + 8), ptx::load(gate + 12), sum[2], sum[3]); }
        else { crate::fp8_bf16::store_middle_pair(y, width, row + 8, col, 0); }
    }
}
