//! Device-fed batch-one frontend; each session owns its history and row ids.
use crate::{frontend_math, index};
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_feed(
    ids: *const f32,
    hist: *mut [i64; 2],
    p: *const [i64; 35],
    rows: *mut [u64; 16],
    embed: *const u16,
    out: *mut f32,
) {
    let i = index();
    let token = *ids.add(1);
    if *ids.add(2) != 1.0
        || !(*ids).is_finite()
        || !token.is_finite()
        || token < 0.0
        || token >= 248320.0
    {
        core::arch::nvptx::trap();
    }
    if i == 0 {
        *rows = frontend_math::rows(token as u32, &mut *hist, &*p);
    }
    if i < 10240 {
        *out.add(i) = f32::from_bits((*embed.add(token as usize * 2560 + i % 2560) as u32) << 16);
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_predict(
    ids: *mut f32,
    hist: *const [i64; 2],
    p: *const [i64; 35],
) {
    if index() == 0 {
        let mut history = *hist;
        *(ids.add(4) as *mut [u64; 16]) =
            frontend_math::rows(*ids.add(1) as u32, &mut history, &*p);
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_ple(
    rows: *const u64,
    bases: *const u64,
    out: *mut f32,
    global: f32,
) {
    let i = index();
    if i >= 2560 {
        return;
    }
    let (h, c) = (i / 160, i % 160);
    let row = *rows.add(h);
    *out.add(i) = ple_element(row, c, bases, global);
}
#[inline(always)]
unsafe fn ple_element(row: u64, c: usize, bases: *const u64, global: f32) -> f32 {
    let shard = (row / 2500012) as usize;
    let r = row % 2500012;
    let (w, s) = (
        (*bases.add(shard * 2) + r * 80) as *const u8,
        (*bases.add(shard * 2 + 1) + r * 10) as *const u8,
    );
    let scale = *s.add(c / 16);
    if scale >= 127 {
        core::arch::nvptx::trap();
    }
    frontend_math::dequant(*w.add(c / 2), c % 2 != 0, scale, global)
}
/// Inputs carry each session's canonical history, so reordering/padding and
/// moving between prompt tiles and any pooled width cannot alias hash state.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_rows_feed(
    input: *const [i64; 3],
    p: *const [i64; 35],
    rows: *mut [u64; 16],
    embed: *const u16,
    out: *mut f32,
    step: crate::next_math::SlotStep,
) {
    let i = index();
    if i >= step.rows as usize * 10240 {
        return;
    }
    let (row, col) = (i / 10240, i % 10240);
    if step.selected(row).is_none() {
        *out.add(i) = 0.0;
        if col == 0 {
            *rows.add(row) = [u64::MAX; 16];
        }
        return;
    }
    let [token, h0, h1] = *input.add(row);
    if token < 0 || token >= 248320 {
        core::arch::nvptx::trap();
    }
    if col == 0 {
        *rows.add(row) = frontend_math::rows(token as u32, &mut [h0, h1], &*p);
    }
    *out.add(i) = f32::from_bits((*embed.add(token as usize * 2560 + col % 2560) as u32) << 16);
}
/// Compact host-staged rows use exactly the mapped-table dequantization order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_rows_ple_staged(
    rows: *const u64,
    packed: *const u8,
    out: *mut f32,
    global: f32,
) {
    let i = index();
    let (h, c) = (i / 160, i % 160);
    *out.add(i) = if *rows.add(h) == u64::MAX {
        0.0
    } else {
        let row = packed.add(h * 90);
        let scale = *row.add(80 + c / 16);
        if scale >= 127 {
            core::arch::nvptx::trap();
        }
        frontend_math::dequant(*row.add(c / 2), c % 2 != 0, scale, global)
    };
}

/// Compact prompt input: rows u32 token IDs, then [rows][16][90] official
/// PLE nibbles/scales. The host validates token bounds and finite scale codes.
/// One lane per hidden feature widens one embedding and writes all four streams.
/// PLE arithmetic is the same shared scalar helper used by the decode frontend.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_prompt_frontend(
    packed: *const u8, embed: *const u16, residual: *mut f32,
    lexical: *mut f32, rows: u32, global: f32,
) {
    let i=index();
    if i>=rows as usize*2560 {return;}
    let (row,col)=(i/2560,i%2560);
    let token=*(packed as *const u32).add(row) as usize;
    let x=f32::from_bits((*embed.add(token*2560+col) as u32)<<16);
    let out=residual.add(row*10240+col);
    *out=x; *out.add(2560)=x; *out.add(5120)=x; *out.add(7680)=x;
    let (head,c)=(col/160,col%160);
    let src=packed.add(rows as usize*4+row*1440+head*90);
    *lexical.add(i)=frontend_math::dequant(*src.add(c/2),c%2!=0,*src.add(80+c/16),global);
}

// Same per-element math; only the official PLE row address is indirect.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_prompt_frontend_dedup(
    packed: *const u8, embed: *const u16, residual: *mut f32,
    lexical: *mut f32, rows: u32, global: f32,
) {
    let i=index();
    if i>=rows as usize*2560 {return;}
    let (row,col)=(i/2560,i%2560);
    let token=*(packed as *const u32).add(row) as usize;
    let x=f32::from_bits((*embed.add(token*2560+col) as u32)<<16);
    let out=residual.add(row*10240+col);
    *out=x; *out.add(2560)=x; *out.add(5120)=x; *out.add(7680)=x;
    let (head,c)=(col/160,col%160);
    let index=*(packed as *const u32).add(rows as usize+row*16+head) as usize;
    let src=packed.add(rows as usize*68+index*90);
    *lexical.add(i)=frontend_math::dequant(*src.add(c/2),c%2!=0,*src.add(80+c/16),global);
}
