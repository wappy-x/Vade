//! One register-resident scan per chat/layer, with an exact entry snapshot.
use crate::{gdn_params::GdnStepBatch, next_math};

macro_rules! eight {
    ($i:ident, $body:block) => {{
        const $i: usize = 0;
        $body
    }
    {
        const $i: usize = 1;
        $body
    }
    {
        const $i: usize = 2;
        $body
    }
    {
        const $i: usize = 3;
        $body
    }
    {
        const $i: usize = 4;
        $body
    }
    {
        const $i: usize = 5;
        $body
    }
    {
        const $i: usize = 6;
        $body
    }
    {
        const $i: usize = 7;
        $body
    }};
}
macro_rules! coordinates {
    ($i:ident, $body:block) => {
        eight!(G, {
            eight!(P, {
                {
                    const $i: usize = G * 16 + P * 2;
                    $body
                }
                {
                    const $i: usize = G * 16 + P * 2 + 1;
                    $body
                }
            });
        });
    };
}

/// output_state names the entry snapshot here. The live state keeps its address.
/// Constant cell indices let LLVM scalarize the complete recurrence into registers.
/// # Safety
/// The descriptor contains live, disjoint state/snapshot arrays of heads*128*128
/// FP32 cells. Q/K/V/out cover count*heads*128 values; beta/g cover count*heads.
/// All buffers stay alive on the ordered stream and the grid covers heads*128.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_verify_scan(p: GdnStepBatch) {
    let output = crate::index();
    let width = p.heads as usize * 128;
    if output >= width {
        return;
    }
    let (head, feature) = (output / 128, output % 128);
    let row = &*(p.rows as *const crate::gdn_params::GdnDecodeRow);
    let state = (row.state as *mut f32).add(head * 16384 + feature);
    let snapshot = (row.output_state as *mut f32).add(head * 16384 + feature);
    let mut cells = [0.0; 128];
    coordinates!(I, {
        cells[I] = *state.add(I * 128);
        *snapshot.add(I * 128) = cells[I];
    });
    for token in 0..p.count as usize {
        let at = token * width + head * 128;
        let q = core::slice::from_raw_parts((p.q as *const f32).add(at), 128);
        let k = core::slice::from_raw_parts((p.k as *const f32).add(at), 128);
        let value = next_math::column(
            &mut cells,
            q,
            k,
            *(p.v as *const f32).add(token * width + output),
            *(p.beta as *const f32).add(token * p.heads as usize + head),
            crate::layer::exp(*(p.g as *const f32).add(token * p.heads as usize + head)),
        );
        *(p.out as *mut f32).add(token * width + output) = value;
    }
    coordinates!(I, {
        *state.add(I * 128) = cells[I];
    });
}
