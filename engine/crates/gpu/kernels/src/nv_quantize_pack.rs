//! `nv_quantize_pack`: the activation quantizer and `tc_nv_pack` in ONE pass, for the tensor-core prompt groups.
//! Before: `expert_group` (`Op::QuantizeRows`, one thread a 16-value block) wrote 17 FP32 words a block into a
//! zero-filled buffer, and `tc_nv_pack` read those 68 bytes back to write 9 (once for gate and once more for up).
//! Here one thread a block reads its 16 FP32 values and writes the 8 code bytes and the scale byte directly.
//! The arithmetic is `expert_ops::quantize_packed`: the quantizer's own expressions, then the pack's code map, so every
//! byte equals the two-step path. A row's bytes depend on that row and its input scale only.
use crate::tc_abi::{NvQuantizePack,NV_QUANTIZE_PACK_THREADS};

#[no_mangle]
pub extern "ptx-kernel" fn nv_quantize_pack(p:NvQuantizePack) {
    // SAFETY: the host (`Backend::nv_rows_tc`, `Backend::nv_quantize_pack_bytes`) checks before the launch: `input` holds
    // rows * cols FP32 words, cols a multiple of 16; `activations` holds rows FP32 words; `codes` holds rows * cols / 2
    // bytes and `scales` rows * cols / 16 bytes; all four are 4-byte aligned, the two outputs are private to this launch
    // and disjoint from the inputs; all owners live through stream completion. Thread i owns block i of the
    // rows * cols / 16 blocks: it reads that block's 16 words (j < 16 in every `src.add(j)`) and its row's scale word, and writes that block's two code
    // words and one scale byte (one writer a byte; every byte of both outputs is written). Any other launch traps.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        if _block_dim_x() as u32!=NV_QUANTIZE_PACK_THREADS || p.cols==0 || p.cols%16!=0 || (p.input|p.codes|p.scales|p.activations)&3!=0 {
            core::arch::asm!("trap;");
        }
        let blocks=p.cols as usize/16;
        let i=_block_idx_x() as usize*NV_QUANTIZE_PACK_THREADS as usize+_thread_idx_x() as usize;
        if i>=p.rows as usize*blocks {return;}
        let (row,b)=(i/blocks,i%blocks);
        let src=(p.input as *const f32).add(row*p.cols as usize+b*16);
        // The block's 16 words are read where they lie, once for the block maximum and once for the codes (the input is
        // immutable through the launch, so both reads give the same word).
        let (lo,hi,scale)=crate::expert_ops::quantize_packed_with(|j|*src.add(j),*(p.activations as *const f32).add(row));
        let dst=(p.codes as *mut u32).add(row*(blocks*2)+b*2);
        *dst=lo;
        *dst.add(1)=hi;
        *(p.scales as *mut u8).add(row*blocks+b)=scale as u8;
    }
}
