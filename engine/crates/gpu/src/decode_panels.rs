//! One-row expert panels: the NVFP4 code bytes and scale pairs of a packed expert unit in MMA lane order.
//! (The dense one-row panels are the lossless tiles of `bf16_panels`, shared with the few-row products.)
/// Interleave the original NVFP4 code bytes and scale pairs in MMA lane order.
/// The checked unit retains its byte length and its original BF16 lookup table.
pub fn pack_expert(source: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    use crate::packed_product::{code_copy, layout, scale_copy};
    let shape = layout(rows, cols, false);
    assert!(rows.is_multiple_of(64) && cols.is_multiple_of(64) && source.len() == shape.bytes);
    let mut data = vec![0; source.len()];
    for panel in 0..rows/16 {
        for k in 0..cols/16 {
            let chunk = ((panel / 4 * (cols / 64) + k / 4) * 4) + panel % 4;
            let within = k % 4;
            for pair in 0..8 {
                let top = panel * 16 + pair;
                let bottom = top + 8;
                let code_start = |row| code_copy(row, k / 4, 0, cols, false) + within * 8;
                let (a,b) = (code_start(top),code_start(bottom));
                let scale = |row| source[shape.scales + scale_copy(row,k/4,cols,false) + within];
                data[shape.scales + chunk*64 + pair*8 + within*2..shape.scales + chunk*64 + pair*8 + within*2 + 2]
                    .copy_from_slice(&[scale(top),scale(bottom)]);
                for lane in 0..4 {
                    let p = chunk * 512 + (pair * 4 + lane) * 16 + within * 4;
                    data[p..p+4].copy_from_slice(&[source[a+lane],source[b+lane],source[a+lane+4],source[b+lane+4]]);
                }
            }
        }
    }
    data[shape.table..].copy_from_slice(&source[shape.table..]);
    data
}

/// Group 64 columns in the four weight registers of one native E2M1 MMA.
/// Each warp has a coalesced 16-byte load per lane and two scale words per row pair.
pub fn pack_draft(source:&[u8],rows:usize,cols:usize,factor:f32)->Vec<u8> {
    use crate::packed_product::{code_copy,layout,scale_copy};
    let shape=layout(rows,cols,false);
    assert!(rows.is_multiple_of(64) && cols.is_multiple_of(64) && source.len()==shape.bytes);
    let mut dst=vec![0;source.len()];
    for row in 0..rows {
        for chunk in 0..cols/64 {
            let stripe=(row/64*(cols/64)+chunk)*4+(row%64)/16;
            let pair=row%8;
            let bottom=(row%16)/8;
            let scales=shape.scales+scale_copy(row,chunk,cols,false);
            let dest=shape.scales+stripe*64+pair*8+bottom*4;
            dst[dest..dest+4].copy_from_slice(&source[scales..scales+4]);
            for lane in 0..4 {
                for half in 0..2 {
                    let code=code_copy(row,chunk,half,cols,false)+lane*4;
                    let dest=stripe*512+(pair*4+lane)*16+(half*2+bottom)*4;
                    dst[dest..dest+4].copy_from_slice(&source[code..code+4]);
                }
            }
        }
    }
    dst[shape.table..shape.table+4].copy_from_slice(&factor.to_le_bytes());
    dst
}

/// Pack a 4-bit expert unit into MMA-lane order and tag the unused E4M3 NaN scale row (`0x7f`)
/// so every packed kernel detects the panel layout in-place with zero extra VRAM.
pub fn pack_expert_tagged(source: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    use crate::packed_product::{NVFP4_PANEL_MAGIC_HI, NVFP4_PANEL_MAGIC_LO, layout};
    let shape = layout(rows, cols, false);
    assert!(rows.is_multiple_of(64) && cols.is_multiple_of(64) && source.len() == shape.bytes);
    let mut data = vec![0u8; source.len()];
    let stages = cols / 64;
    for group64 in 0..rows / 64 {
        for stage in 0..stages {
            for p in 0..4 {
                let panel = group64 * 4 + p;
                let chunk = (group64 * stages + stage) * 4 + p;
                let row_block = panel / 2;
                let row_in_block = (panel & 1) * 16;
                let code_tile = (row_block * stages + stage) * 1024;
                let scale_tile = shape.scales + (row_block * stages + stage) * 128;
                let dst_code_chunk = chunk * 512;
                let dst_scale_chunk = shape.scales + chunk * 64;
                for pair in 0..8 {
                    let top = row_in_block + pair;
                    let bot = top + 8;
                    let a_base = code_tile + top * 32;
                    let b_base = code_tile + bot * 32;
                    let st_base = scale_tile + top * 4;
                    let sb_base = scale_tile + bot * 4;
                    let ds_base = dst_scale_chunk + pair * 8;
                    for within in 0..4 {
                        data[ds_base + within * 2] = source[st_base + within];
                        data[ds_base + within * 2 + 1] = source[sb_base + within];
                        let a = a_base + within * 8;
                        let b = b_base + within * 8;
                        for lane in 0..4 {
                            let dst = dst_code_chunk + (pair * 4 + lane) * 16 + within * 4;
                            data[dst] = source[a + lane];
                            data[dst + 1] = source[b + lane];
                            data[dst + 2] = source[a + lane + 4];
                            data[dst + 3] = source[b + lane + 4];
                        }
                    }
                }
            }
        }
    }
    data[shape.table..].copy_from_slice(&source[shape.table..]);
    let tag = shape.table + 0x7f * 32;
    data[tag..tag + 4].copy_from_slice(&NVFP4_PANEL_MAGIC_LO.to_le_bytes());
    data[tag + 4..tag + 8].copy_from_slice(&NVFP4_PANEL_MAGIC_HI.to_le_bytes());
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed_product::{NVFP4_PANEL_MAGIC_HI, NVFP4_PANEL_MAGIC_LO, layout};

    #[test]
    fn tagged_pack_matches_pack_expert_and_sets_nan_row_tag() {
        let (rows, cols) = (128, 192);
        let shape = layout(rows, cols, false);
        let source: Vec<u8> = (0..shape.bytes).map(|i| (i.wrapping_mul(37) ^ (i >> 5)) as u8).collect();
        let plain = pack_expert(&source, rows, cols);
        let tagged = pack_expert_tagged(&source, rows, cols);
        let tag = shape.table + 0x7f * 32;
        assert_eq!(tagged.len(), plain.len());
        assert_eq!(&tagged[..tag], &plain[..tag]);
        assert_eq!(&tagged[tag + 8..], &plain[tag + 8..]);
        assert_eq!(u32::from_le_bytes(tagged[tag..tag + 4].try_into().unwrap()), NVFP4_PANEL_MAGIC_LO);
        assert_eq!(u32::from_le_bytes(tagged[tag + 4..tag + 8].try_into().unwrap()), NVFP4_PANEL_MAGIC_HI);
    }
}
