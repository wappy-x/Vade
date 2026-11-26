//! The embedding rows of a step (`embed_rows`): the entry `resident_launch_embed_rows` of the Inkling image (crate
//! `kernels-inkling`, Ruling R6h), kind `inkling_launch::EMBED_ROWS`.
//!
//! One launch copies the stored BF16 rows of `count` tokens out of the embedding table that the device holds, each
//! value widened to its 32-bit value (the low half word zero): output row `r` is table row `ids[r]`. Widening changes
//! no value, so the words are those of the host copy of the table that the driver read before. No arithmetic.
//!
//! The host checks every extent before it records a launch (`embed_rows::check`), and stages the token numbers with
//! the command.
//!
//! Launch: one block of `embed_rows::THREADS` threads a token row; a thread takes every `THREADS`-th word of the row.

/// The replay of one recorded command of this kind: `table[index]` holds six words: the address of the table
/// (`vocab` rows of `width` BF16 values, two a word), of the output (`count x width` FP32 words), `vocab`, `width`,
/// the address of the token table (`count` u32 numbers, staged with the command) and `count`.
/// SAFETY (contract of the caller, the entry `resident_launch_embed_rows`): `table` and `index` are the arguments the
/// recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::gather_embed_rows_into, after embed_rows::check
    // passed for its words: the table is exactly vocab x width / 2 words (width even, 2 to 16,384), the output is
    // exactly count x width words, private and contiguous, meeting neither the table nor anything the command reads;
    // count is 1 to ROWS_MAX and every token number is below vocab. The token table is those host words, staged with
    // this command in the recorder's own stage area (record_resident_tabled) and written by nothing else before this
    // replay. Here the block is compared with count, the token table's alignment and every token number with vocab
    // before use, and width is checked even and nonzero; any failure traps. Reads: word `id x width / 2 + w` of the
    // table for w below width / 2, inside it. Writes: words `r x width + 2w` and `+ 1` of the output, each by one
    // thread of one block, inside it. No shared memory and no barrier.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::EMBED_ROWS { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (stored, output) = (*words as *const u32, *words.add(1) as *mut f32);
        let (vocab, width) = (*words.add(2) as usize, *words.add(3) as usize);
        let (ids, count) = (*words.add(4) as *const u32, *words.add(5) as usize);
        let row = core::arch::nvptx::_block_idx_x() as usize;
        if row >= count || width == 0 || width % 2 != 0 || ids.is_null() || (ids as usize) & 3 != 0 { core::arch::nvptx::trap(); }
        let id = ids.add(row).read() as usize;
        if id >= vocab { core::arch::nvptx::trap(); }
        let (half, from, out) = (width / 2, stored.add(id * (width / 2)), output.add(row * width));
        let mut w = core::arch::nvptx::_thread_idx_x() as usize;
        let step = core::arch::nvptx::_block_dim_x() as usize;
        while w < half {
            let pair = from.add(w).read();
            out.add(2 * w).write(f32::from_bits(pair << 16));
            out.add(2 * w + 1).write(f32::from_bits(pair & 0xffff_0000));
            w += step;
        }
    }
}
