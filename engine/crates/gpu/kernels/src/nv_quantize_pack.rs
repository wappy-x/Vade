//! `nv_quantize_pack`: the activation quantizer and `tc_nv_pack` in ONE pass, for the tensor-core prompt groups.
//! Before: `expert_group` (`Op::QuantizeRows`, one thread a 16-value block) wrote 17 FP32 words a block into a
//! zero-filled buffer, and `tc_nv_pack` read those 68 bytes back to write 9 (once for gate and once more for up).
//! Here one thread a block reads its 16 FP32 values and writes the 8 code bytes and the scale byte directly.
//! The arithmetic is `expert_ops::quantize_packed`: the quantizer's own expressions, then the pack's code map, so every
//! byte equals the two-step path. A row's bytes depend on that row and its input scale only.
//!
//! `nv_quantize_gather`: the same pass with the row read through an index. A prompt group gives every token's row to
//! each of its picked experts; before, `prompt_permute` copied the rows into expert order (8 copies of each row on the
//! model) and `nv_quantize_pack` read the copy. Here operand row r reads input row `index[r]` where it lies. The copy
//! moved words, so the 16 words a thread reads are the words it read from the copy, and every byte is equal.
//!
//! Both entries call ONE body (`blocks`, never inlined), so the image holds one instance of the quantizer's expressions,
//! as it did with one entry. MEASURED 2026-11-15: a second instance changed the compiled code of six other kernels of
//! the image (`causal_fast`, `causal_operation`, `expert_operation`, `mhc_boundary_rms`, `fp4_prefill_tc_down`,
//! `resident_launch_causal_group`); with one body every other kernel of the image keeps its code.
use crate::tc_abi::{NvQuantizeGather,NvQuantizePack,NV_QUANTIZE_PACK_THREADS};

/// The body of both entries: thread i owns block i of the rows * cols / 16 blocks. `gather` false: operand row r reads
/// input row r. `gather` true: operand row r reads input row `index[r]`.
#[inline(never)]
#[allow(clippy::too_many_arguments)] // the fields of the two entries' argument blocks, as words.
fn blocks(input:u64,codes:u64,scales:u64,activations:u64,index:u64,gather:bool,rows:u32,cols:u32,source_rows:u32) {
    // SAFETY: the only callers are the two kernel entries below, with the words of their argument blocks. Their hosts
    // (`Backend::nv_rows_tc`, `Backend::nv_quantize_pack_bytes`) check before the launch: `input` holds
    // source_rows * cols FP32 words, cols a multiple of 16; `activations` holds rows FP32 words; with `gather`, `index`
    // holds rows FP32 words; `codes` holds rows * cols / 2 bytes and `scales` rows * cols / 16 bytes; the two outputs
    // are private to this launch and disjoint from the inputs; all owners live through stream completion.
    // Thread i owns block i: it reads its row's scale word, its row's index word when rows are gathered, and that
    // block's 16 words of its input row (j < 16 in every `src.add(j)`), and writes that block's two code words and one
    // scale byte (one writer a byte; every byte of both outputs is written; two operand rows may READ one input row,
    // none writes it). The index word is data: it is read as `prompt_permute` reads its table (a float cast to an
    // index, NaN and negative to 0). An input row at or beyond `source_rows` traps before any read of the input
    // (without an index the row is below `rows`, which is `source_rows` there). Every address must be 4-byte aligned,
    // a gathered launch must have an index, and the block must have NV_QUANTIZE_PACK_THREADS threads: any other launch traps.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        if _block_dim_x() as u32!=NV_QUANTIZE_PACK_THREADS || cols==0 || cols%16!=0 || (input|codes|scales|activations|index)&3!=0 || (gather && index==0) {
            core::arch::asm!("trap;");
        }
        let blocks=cols as usize/16;
        let i=_block_idx_x() as usize*NV_QUANTIZE_PACK_THREADS as usize+_thread_idx_x() as usize;
        if i>=rows as usize*blocks {return;}
        let (row,b)=(i/blocks,i%blocks);
        let from=if gather {*(index as *const f32).add(row) as usize} else {row};
        if from>=source_rows as usize {core::arch::asm!("trap;");}
        let src=(input as *const f32).add(from*cols as usize+b*16);
        // The block's 16 words are read where they lie, once for the block maximum and once for the codes (the input is
        // immutable through the launch, so both reads give the same word).
        let (lo,hi,scale)=crate::expert_ops::quantize_packed_with(|j|*src.add(j),*(activations as *const f32).add(row));
        let dst=(codes as *mut u32).add(row*(blocks*2)+b*2);
        *dst=lo;
        *dst.add(1)=hi;
        *(scales as *mut u8).add(row*blocks+b)=scale as u8;
    }
}

/// FP32 rows straight to the packed operands: operand row r reads input row r (`tc_abi::NvQuantizePack`). The launch
/// contract is in `blocks`; this entry gives it no index and `rows` input rows, as it always read them.
#[no_mangle]
pub extern "ptx-kernel" fn nv_quantize_pack(p:NvQuantizePack) {blocks(p.input,p.codes,p.scales,p.activations,4,false,p.rows,p.cols,p.rows)}

/// GATHERED ROWS: operand row r reads input row `index[r]` (`tc_abi::NvQuantizeGather`). The launch contract is in
/// `blocks`; a launch without an index traps there.
#[no_mangle]
pub extern "ptx-kernel" fn nv_quantize_gather(p:NvQuantizeGather) {blocks(p.input,p.codes,p.scales,p.activations,p.index,true,p.rows,p.cols,p.source_rows)}
