//! Plain FP32 rows with deeper schedules. The arithmetic is the one of `serial_projection::plain_row!`:
//! each output is ONE ascending chain over the columns from 0.0, every product rounded by its own
//! `mul.rn.f32` before its `add.rn.f32`, no fused multiply-add. Only the loads change.
//!
//! `row_deep`: one thread per row; 8 vectors of weights and input are loaded for the next batch
//! while the chain runs over the current one, and the line 8 batches ahead is pulled toward L2.
//! `rows_warp`: one warp owns 32 consecutive rows. The warp copies a 32 x 32 tile (and the matching
//! 32 input values) into shared memory with coalesced 16-byte `cp.async` copies, two stages deep,
//! then each lane runs the chain of its own row from shared memory. The original loop made every
//! lane load a different row line per instruction, which the load unit serves one line at a time.
//! Shapes are chosen by `plain_admission`; other shapes keep the original loop.
use core::arch::asm;

const U: usize = 8;
const PD: usize = 8;
/// Columns per tile chunk, pipeline stages, and shared words per tile row (a 4-word pad keeps
/// 16-byte reads of eight neighbouring rows in distinct banks).
const CH: usize = 32;
pub(crate) const NS_BLOCK: usize = 2;
/// A lone warp (one-warp block) gets the whole static block: four times the stages.
pub(crate) const NS_ALONE: usize = 8;
const _: () = assert!(NS_ALONE * STAGE_BYTES == 4 * NS_BLOCK * STAGE_BYTES);
const SK: usize = CH + 4;
/// Bytes of one stage: 32 weight rows and one input row.
const STAGE_BYTES: usize = 33 * SK * 4;
/// Shared bytes of one warp, and of the 4-warp block.
pub(crate) const WARP_BYTES: usize = NS_BLOCK * STAGE_BYTES;
pub(crate) const BLOCK_BYTES: usize = 4 * WARP_BYTES;

pub(crate) fn row_deep<const BF16:bool>(w: *const f32, x: *const f32, cols: usize) -> f32 {
    // SAFETY: The caller passes a row of `cols` FP32 values, or BF16 patterns when BF16 is true, at `w` and `cols` input values at `x`,
    // both 16-byte aligned with cols a multiple of 4 (plain_admission::deep_admitted). Every
    // vector read lies inside those extents: batch loads run only while a full batch remains and
    // the tail loop reads one whole vector at a time. Prefetch addresses are bounded by cols and a
    // prefetch never faults. No memory is written.
    unsafe {
        macro_rules! ld4 {($p:expr)=>{{
            let (a,b,c,d):(f32,f32,f32,f32);
            asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$p);
            [a,b,c,d]
        }};}
        macro_rules! weight {($at:expr)=>{{
            if BF16 {
                let (a,b):(u32,u32);let address=w.cast::<u16>().add($at);
                asm!("ld.global.v2.u32 {{{a},{b}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,p=in(reg64)address);
                [f32::from_bits(a<<16),f32::from_bits(a&0xffff0000),f32::from_bits(b<<16),f32::from_bits(b&0xffff0000)]
            } else {ld4!(w.add($at))}
        }};}
        macro_rules! acc {($sum:ident,$w:expr,$x:expr)=>{{
            let product:f32;
            asm!("mul.rn.f32 {product}, {w}, {x};",product=out(reg32)product,w=in(reg32)$w,x=in(reg32)$x,options(nomem,nostack));
            asm!("add.rn.f32 {sum}, {sum}, {product};",sum=inout(reg32)$sum,product=in(reg32)product,options(nomem,nostack));
        }};}
        macro_rules! group {($q:expr)=>{{
            let mut a=[[0.0f32;4];U];let mut b=[[0.0f32;4];U];let mut i=0;
            while i<U {a[i]=weight!(($q+i)*4);i+=1;}
            i=0;while i<U {b[i]=ld4!(x.add(($q+i)*4));i+=1;}
            (a,b)
        }};}
        let quads = cols / 4;
        let mut sum = 0.0f32;
        let mut q = 0;
        if quads >= U {
            let mut cur = group!(0);
            while q + 2 * U <= quads {
                let ahead = (q + U * (PD + 1)) * 4;
                if ahead < cols {let address=if BF16 {w.cast::<u16>().add(ahead) as u64} else {w.add(ahead) as u64};asm!("prefetch.global.L2 [{a}];", a = in(reg64) address);}
                let next = group!(q + U);
                let mut i = 0;
                while i < U {
                    acc!(sum, cur.0[i][0], cur.1[i][0]); acc!(sum, cur.0[i][1], cur.1[i][1]);
                    acc!(sum, cur.0[i][2], cur.1[i][2]); acc!(sum, cur.0[i][3], cur.1[i][3]);
                    i += 1;
                }
                cur = next;
                q += U;
            }
            let mut i = 0;
            while i < U {
                acc!(sum, cur.0[i][0], cur.1[i][0]); acc!(sum, cur.0[i][1], cur.1[i][1]);
                acc!(sum, cur.0[i][2], cur.1[i][2]); acc!(sum, cur.0[i][3], cur.1[i][3]);
                i += 1;
            }
            q += U;
        }
        while q < quads {
            let a = weight!(q * 4);
            let b = ld4!(x.add(q * 4));
            acc!(sum, a[0], b[0]); acc!(sum, a[1], b[1]); acc!(sum, a[2], b[2]); acc!(sum, a[3], b[3]);
            q += 1;
        }
        sum
    }
}

/// Returns the dot product of row `first_row + lane` for every lane whose row is below `rows`; a lane whose row is not
/// below `rows` returns 0.0 and the caller must not store it. `smem` is this warp's own `WARP_BYTES` of shared memory.
pub(crate) fn rows_warp<const NS: usize>(weights: *const f32, input: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> f32 {
    // SAFETY: The caller reaches this with a whole warp (every lane, no early exit), plain_admission::
    // tile_admitted true (cols a multiple of 4 in 32..=8192, weights and input 16-byte aligned, rows * cols
    // within the swept envelope), first_row below rows and the same on all lanes, and `smem` the start of
    // WARP_BYTES private to this warp. Global copies and prefetches are made only for rows below `rows` and
    // columns below `cols`, so every global read stays in the matrix and the input. Every shared read follows
    // a write that the reading lane has waited for: a lane reads only slots of its OWN row (a row below `rows`,
    // so its copies were issued) and the shared input row, for columns below `cols` that the copy of this
    // chunk covered, after `cp.async.wait_group` and the warp barrier completed all copies of the stage. A lane
    // whose row is not below `rows` (a partial warp) issues no copy for it and reads nothing; it still
    // executes every copy issue, commit, wait and barrier of the warp. The barrier after each chunk keeps a
    // stage from being refilled while any lane still reads it. No global memory is written.
    unsafe {
        let lane = (core::arch::nvptx::_thread_idx_x() as usize) & 31;
        let live = first_row + lane < rows;
        macro_rules! issue {($chunk:expr,$stage:expr)=>{{
            let col = $chunk * CH + (lane % 8) * 4;
            let mut i = 0;
            while i < 8 {
                let row = i * 4 + lane / 8;
                let g = first_row + row;
                if g < rows && col < cols {
                    crate::ptx::copy16($stage + ((row * SK + (lane % 8) * 4) * 4) as u32, weights.add(g * cols + col).cast());
                }
                i += 1;
            }
            if lane < 8 && $chunk * CH + lane * 4 < cols {
                crate::ptx::copy16($stage + ((32 * SK + lane * 4) * 4) as u32, input.add($chunk * CH + lane * 4).cast());
            }
        }};}
        let chunks = cols.div_ceil(CH);
        let mut s = 0;
        while s + 1 < NS {
            if s < chunks { issue!(s, smem + (s * STAGE_BYTES) as u32); }
            crate::ptx::commit();
            s += 1;
        }
        let mut sum = 0.0f32;
        let mut c = 0;
        while c < chunks {
            let ahead = c + NS - 1;
            if ahead < chunks { issue!(ahead, smem + ((ahead % NS) * STAGE_BYTES) as u32); }
            crate::ptx::commit();
            let pc = (c + 4 * NS) * CH;
            let g = first_row + lane;
            if pc < cols && g < rows { asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights.add(g * cols + pc) as u64); }
            asm!("cp.async.wait_group {n};", n = const (NS - 1));
            asm!("bar.warp.sync 0xffffffff;");
            let stage = smem + ((c % NS) * STAGE_BYTES) as u32;
            macro_rules! quad {($q:expr)=>{{
                let w = crate::ptx::load4(stage + ((lane * SK + $q * 4) * 4) as u32);
                let x = crate::ptx::load4(stage + ((32 * SK + $q * 4) * 4) as u32);
                let mut e = 0;
                while e < 4 {
                    let product: f32;
                    asm!("mul.rn.f32 {product}, {w}, {x};", product = out(reg32) product, w = in(reg32) f32::from_bits(w[e]), x = in(reg32) f32::from_bits(x[e]), options(nomem, nostack));
                    asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) product, options(nomem, nostack));
                    e += 1;
                }
            }};}
            if !live {
                // Nothing was copied for this lane's row; read nothing.
            } else if (c + 1) * CH <= cols {
                // A whole chunk: constant trip count, so the shared loads of all 8 vectors issue before the chain.
                let mut q = 0;
                while q < CH / 4 { quad!(q); q += 1; }
            } else {
                let mut q = 0;
                while q < (cols - c * CH) / 4 { quad!(q); q += 1; }
            }
            asm!("bar.warp.sync 0xffffffff;");
            c += 1;
        }
        sum
    }
}

/// Fold schedule (plain_admission::FOLD_THREADS-thread blocks, FOLD_ROWS rows a block): warps 1..=4 make the rounded products,
/// one lane a row adds them. For matrices with few rows and long rows (the router gate, 144 x 4096) the one-warp tile ran the
/// copies, the waits and the chain of each row in ONE warp, one after the other; here the chain of chunk c runs while the
/// producers fetch chunk c+1. Each output is the same ONE ascending chain from 0.0 of the same products: every product is made
/// by its own `mul.rn.f32` (weight first, input second, as `plain_row!`), stored to shared memory as its bits, and added by its
/// own `add.rn.f32` in column order. No fused multiply-add.
pub(crate) const FOLD_CH: usize = 1024;
pub(crate) const FOLD_SLOTS: usize = 2;
/// Shared words per row of a slot: a 4-word pad puts the 16-byte reads of the FOLD_ROWS folding lanes in distinct banks.
const FOLD_SK: usize = FOLD_CH + 4;
const FOLD_SLOT_BYTES: usize = crate::plain_admission::FOLD_ROWS * FOLD_SK * 4;
pub(crate) const FOLD_BYTES: usize = FOLD_SLOTS * FOLD_SLOT_BYTES;
const _: () = assert!(FOLD_BYTES <= BLOCK_BYTES, "the fold ring must not enlarge the block's shared memory");
/// Vectors of one chunk each producer thread handles: FOLD_ROWS rows x FOLD_CH/4 vectors over 128 producer threads.
const FOLD_PRODUCERS: usize = crate::plain_admission::FOLD_THREADS - 32;
const FOLD_VECTORS: usize = crate::plain_admission::FOLD_ROWS * (FOLD_CH / 4) / FOLD_PRODUCERS;
const _: () = assert!(FOLD_VECTORS * FOLD_PRODUCERS == crate::plain_admission::FOLD_ROWS * (FOLD_CH / 4) && FOLD_CH % 16 == 0 && (FOLD_CH / 4) % FOLD_PRODUCERS == 0);
const _: () = assert!(FOLD_VECTORS == 8, "rows_fold fetches and stores exactly eight vectors a chunk");
const FOLD_FULL: u32 = 1;
const FOLD_EMPTY: u32 = FOLD_FULL + FOLD_SLOTS as u32;

/// Returns, on lane `r < FOLD_ROWS` of warp 0, the dot product of row `first_row + r` when that row is below `rows`; every other
/// thread returns 0.0 and the caller must not store it. `smem` is the block's FOLD_BYTES of shared memory.
pub(crate) fn rows_fold<const BF16:bool>(weights: *const f32, input: *const f32, cols: usize, first_row: usize, rows: usize, smem: u32) -> f32 {
    // SAFETY: The caller reaches this with every thread of a FOLD_THREADS-thread block (no early exit), with
    // plain_admission::fold_admitted true (cols a multiple of 4 in 32..=8192, weights and input 16-byte aligned, rows * cols
    // within the swept envelope; weights contain BF16 patterns when BF16 is true), first_row below rows and the same on all threads, and `smem` the start of FOLD_BYTES of
    // shared memory private to this block. Global reads: a producer loads a weight vector only for a row below `rows` and
    // columns below `cols` (a whole 16-byte vector, cols a multiple of 4), and an input vector only for columns below `cols`;
    // so every read stays in the matrix and the input. Shared memory: a slot is written only by producers between the empty
    // barrier of that slot (or the start, for the first FOLD_SLOTS chunks) and their arrival at its full barrier; it is read
    // only by a folding lane after the full barrier, and only the words of its own live row below the chunk's span, all of which
    // a producer stored for this chunk. The consumer warp arrives at the empty barrier of a slot only after its reads of that
    // slot, and only when a later chunk will refill it; producers wait at it exactly for those chunks, so arrivals and waits
    // pair up and no thread waits for a barrier nobody reaches. Every named barrier counts all FOLD_THREADS threads; the
    // warp-level syncs before each one reconverge a warp after its per-thread branches. No global memory is written.
    unsafe {
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let n = crate::plain_admission::FOLD_THREADS as u32;
        let chunks = cols.div_ceil(FOLD_CH);
        if tid < 32 {
            let live = tid < crate::plain_admission::FOLD_ROWS && first_row + tid < rows;
            let row_base = (tid.min(crate::plain_admission::FOLD_ROWS - 1) * FOLD_SK * 4) as u32;
            let mut sum = 0.0f32;
            macro_rules! add {($v:expr)=>{{
                asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) f32::from_bits($v), options(nomem, nostack));
            }};}
            // Groups of 16 products, two register sets used in turn: the next group's shared loads issue before the current
            // group's adds. The adds are the same adds in the same ascending order.
            macro_rules! group {($at:expr)=>{{let at=$at;[crate::ptx::load4(at),crate::ptx::load4(at+16),crate::ptx::load4(at+32),crate::ptx::load4(at+48)]}}}
            macro_rules! adds {($r:ident)=>{{
                add!($r[0][0]);add!($r[0][1]);add!($r[0][2]);add!($r[0][3]);add!($r[1][0]);add!($r[1][1]);add!($r[1][2]);add!($r[1][3]);
                add!($r[2][0]);add!($r[2][1]);add!($r[2][2]);add!($r[2][3]);add!($r[3][0]);add!($r[3][1]);add!($r[3][2]);add!($r[3][3]);
            }};}
            let mut c = 0;
            while c < chunks {
                let s = c % FOLD_SLOTS;
                asm!("bar.sync {id}, {n};", id = in(reg32) FOLD_FULL + s as u32, n = in(reg32) n);
                if live {
                    let span = (cols - c * FOLD_CH).min(FOLD_CH);
                    let at = smem + (s * FOLD_SLOT_BYTES) as u32 + row_base;
                    let groups = span / 16;
                    if groups > 0 {
                        let mut a = group!(at);
                        let mut g = 0usize;
                        while g + 1 < groups {
                            let b = group!(at + ((g + 1) * 64) as u32);
                            adds!(a);
                            a = if g + 2 < groups { group!(at + ((g + 2) * 64) as u32) } else { b };
                            adds!(b);
                            g += 2;
                        }
                        if g < groups { adds!(a); }
                    }
                    let mut j = groups * 16;
                    while j < span {
                        let v = crate::ptx::load4(at + (j * 4) as u32);
                        add!(v[0]); add!(v[1]); add!(v[2]); add!(v[3]);
                        j += 4;
                    }
                }
                asm!("bar.warp.sync 0xffffffff;");
                if c + FOLD_SLOTS < chunks { asm!("bar.arrive {id}, {n};", id = in(reg32) FOLD_EMPTY + s as u32, n = in(reg32) n); }
                c += 1;
            }
            if live { sum } else { 0.0 }
        } else {
            let p = tid - 32;
            macro_rules! ld4 {($a:expr)=>{{
                let (a,b,c,d):(f32,f32,f32,f32);
                asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$a);
                [a,b,c,d]
            }};}
            macro_rules! mul {($w:expr,$x:expr)=>{{
                let product: f32;
                asm!("mul.rn.f32 {product}, {w}, {x};", product = out(reg32) product, w = in(reg32) $w, x = in(reg32) $x, options(nomem, nostack));
                product.to_bits()
            }};}
            let mut c = 0;
            while c < chunks {
                let s = c % FOLD_SLOTS;
                if c >= FOLD_SLOTS { asm!("bar.sync {id}, {n};", id = in(reg32) FOLD_EMPTY + s as u32, n = in(reg32) n); }
                let first = c * FOLD_CH;
                let span = (cols - first).min(FOLD_CH);
                let slot = smem + (s * FOLD_SLOT_BYTES) as u32;
                // Vector v = p + k * FOLD_PRODUCERS of the chunk: row v / (FOLD_CH/4), vector q = v % (FOLD_CH/4) of that row, so a
                // warp reads 32 neighbouring vectors (512 bytes) of one row. All loads of the chunk issue before any product.
                let mut w = [[0.0f32; 4]; FOLD_VECTORS];
                let mut x = [[0.0f32; 4]; FOLD_VECTORS];
                macro_rules! fetch {($k:expr)=>{{
                    let v = p + $k * FOLD_PRODUCERS; let (r, q) = (v / (FOLD_CH / 4), v % (FOLD_CH / 4));
                    if q * 4 < span {
                        x[$k] = ld4!(input.add(first + q * 4));
                        if first_row + r < rows {
                            let at=(first_row+r)*cols+first+q*4;
                            w[$k]=if BF16 {
                                let (a,b):(u32,u32);
                                asm!("ld.global.v2.u32 {{{a},{b}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,p=in(reg64)weights.cast::<u16>().add(at));
                                [f32::from_bits(a<<16),f32::from_bits(a&0xffff0000),f32::from_bits(b<<16),f32::from_bits(b&0xffff0000)]
                            } else {ld4!(weights.add(at))};
                        }
                    }
                }};}
                fetch!(0); fetch!(1); fetch!(2); fetch!(3); fetch!(4); fetch!(5); fetch!(6); fetch!(7);
                macro_rules! put {($k:expr)=>{{
                    let v = p + $k * FOLD_PRODUCERS; let (r, q) = (v / (FOLD_CH / 4), v % (FOLD_CH / 4));
                    if q * 4 < span && first_row + r < rows {
                        let (a, b, cc, d) = (mul!(w[$k][0], x[$k][0]), mul!(w[$k][1], x[$k][1]), mul!(w[$k][2], x[$k][2]), mul!(w[$k][3], x[$k][3]));
                        asm!("st.shared.v4.b32 [{at}], {{{a}, {b}, {c}, {d}}};", at = in(reg32) slot + ((r * FOLD_SK + q * 4) * 4) as u32, a = in(reg32) a, b = in(reg32) b, c = in(reg32) cc, d = in(reg32) d);
                    }
                }};}
                put!(0); put!(1); put!(2); put!(3); put!(4); put!(5); put!(6); put!(7);
                asm!("bar.warp.sync 0xffffffff;");
                asm!("bar.arrive {id}, {n};", id = in(reg32) FOLD_FULL + s as u32, n = in(reg32) n);
                c += 1;
            }
            0.0
        }
    }
}

const _: () = assert!(crate::plain_admission::FOLD_ROWS_BYTES <= BLOCK_BYTES, "the fold-rows slots must not enlarge the block's shared memory");
/// FOLD ROWS (`plain_admission::fold_rows_admitted`): the fold schedule for 2 to 8 source rows in one block. Returns, on lane
/// `t * FOLD_ROWS + r` of warp 0, the dot product of weight row `first_row + r` with source row `t` when `t` is below `tokens`
/// and the row is below `rows`; every other thread returns 0.0 and the caller must not store it. `LAYOUT` is
/// `plain_admission::fold_rows_layout(tokens)` (2, 4 or 8). `input` holds `[tokens][cols]` FP32 words. `smem` is the block's
/// `plain_admission::FOLD_ROWS_BYTES` of shared memory.
///
/// Each output is the ONE ascending chain from 0.0 that `rows_fold` makes for that source row alone: every product is made by its
/// own `mul.rn.f32` (weight first, input second), stored to shared memory as its bits, and added by its own `add.rn.f32` in column
/// order. No fused multiply-add. What changes against `rows_fold` is only who shares a block: a producer loads a weight vector
/// ONCE and multiplies it into one product vector for each source row, and the folding warp runs `tokens * FOLD_ROWS` chains,
/// one lane a (source row, weight row). A chain has its own accumulator register and its own pitch of each slot
/// (`plain_admission::fold_rows_word`); no two chains share either.
pub(crate) fn rows_fold_rows<const BF16:bool, const LAYOUT: usize>(weights: *const f32, input: *const f32, cols: usize, first_row: usize, rows: usize, tokens: usize, smem: u32) -> f32 {
    // SAFETY: The caller reaches this with every thread of a FOLD_THREADS-thread block (no early exit), with
    // plain_admission::fold_rows_admitted true (cols a multiple of 32 in 1024..=8192, weights and input 16-byte aligned, rows * cols
    // within the swept envelope; weights contain BF16 patterns when BF16 is true), `tokens` in 2..=LAYOUT with LAYOUT 2, 4 or 8,
    // `tokens * cols` input words, first_row below rows and the same on all threads, and `smem` the start of FOLD_ROWS_BYTES of
    // shared memory private to this block (2 slots x FOLD_ROWS * LAYOUT chains x (1024 / LAYOUT + 4) words fit it for every LAYOUT).
    // Global reads: a producer loads a weight vector only for a row below `rows` and columns below `cols` (a whole 16-byte vector,
    // cols a multiple of 4), and an input vector only for a source row below `tokens` and columns below `cols`; so every read stays
    // in the matrix and the input. A producer issues the loads of chunk c + 1 before it makes the products of chunk c (LOAD
    // AHEAD, into the other of its two register sets); loads read only the immutable weights and input under the bounds above, so
    // they need no barrier, and the order of the shared-memory protocol below is the one-chunk order it was.
    // Shared memory: a slot is written only by producers between the empty barrier of that slot (or the
    // start, for the first FOLD_SLOTS chunks) and their arrival at its full barrier; a word is written by exactly one producer
    // (`fold_rows_vector` maps the producers and their eight product vectors onto the chunk's (source row, weight row, vector)
    // triples one to one for each LAYOUT; a host test walks it); it is read only by a folding lane after the full barrier, and only
    // the words of its own live chain below the chunk's span, all of which a producer stored for this chunk. The consumer warp
    // arrives at the empty barrier of a slot only after its reads of that slot, and only when a later chunk will refill it;
    // producers wait at it exactly for those chunks, so arrivals and waits pair up and no thread waits for a barrier nobody
    // reaches. Every named barrier counts all FOLD_THREADS threads; the warp-level syncs before each one reconverge a warp after
    // its per-thread branches. No global memory is written.
    unsafe {
        use crate::plain_admission::{fold_rows_chunk, fold_rows_pitch, fold_rows_vector, fold_rows_word, FOLD_ROWS};
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let n = crate::plain_admission::FOLD_THREADS as u32;
        let ch = fold_rows_chunk(LAYOUT);
        let slot_bytes = FOLD_ROWS * LAYOUT * fold_rows_pitch(LAYOUT) * 4;
        let chunks = cols.div_ceil(ch);
        if tid < 32 {
            let chains = FOLD_ROWS * LAYOUT;
            let (t, r) = (tid / FOLD_ROWS, tid % FOLD_ROWS);
            let live = tid < chains && t < tokens && first_row + r < rows;
            let chain = if tid < chains { tid } else { chains - 1 };
            let row_base = (fold_rows_word(LAYOUT, chain / FOLD_ROWS, chain % FOLD_ROWS, 0) * 4) as u32;
            let mut sum = 0.0f32;
            macro_rules! add {($v:expr)=>{{
                asm!("add.rn.f32 {sum}, {sum}, {product};", sum = inout(reg32) sum, product = in(reg32) f32::from_bits($v), options(nomem, nostack));
            }};}
            // Groups of 16 products, two register sets used in turn, as `rows_fold`: the same adds in the same ascending order.
            macro_rules! group {($at:expr)=>{{let at=$at;[crate::ptx::load4(at),crate::ptx::load4(at+16),crate::ptx::load4(at+32),crate::ptx::load4(at+48)]}}}
            macro_rules! adds {($r:ident)=>{{
                add!($r[0][0]);add!($r[0][1]);add!($r[0][2]);add!($r[0][3]);add!($r[1][0]);add!($r[1][1]);add!($r[1][2]);add!($r[1][3]);
                add!($r[2][0]);add!($r[2][1]);add!($r[2][2]);add!($r[2][3]);add!($r[3][0]);add!($r[3][1]);add!($r[3][2]);add!($r[3][3]);
            }};}
            let mut c = 0;
            while c < chunks {
                let s = c % FOLD_SLOTS;
                asm!("bar.sync {id}, {n};", id = in(reg32) FOLD_FULL + s as u32, n = in(reg32) n);
                if live {
                    let span = (cols - c * ch).min(ch);
                    let at = smem + (s * slot_bytes) as u32 + row_base;
                    let groups = span / 16;
                    if groups > 0 {
                        let mut a = group!(at);
                        let mut g = 0usize;
                        while g + 1 < groups {
                            let b = group!(at + ((g + 1) * 64) as u32);
                            adds!(a);
                            a = if g + 2 < groups { group!(at + ((g + 2) * 64) as u32) } else { b };
                            adds!(b);
                            g += 2;
                        }
                        if g < groups { adds!(a); }
                    }
                    let mut j = groups * 16;
                    while j < span {
                        let v = crate::ptx::load4(at + (j * 4) as u32);
                        add!(v[0]); add!(v[1]); add!(v[2]); add!(v[3]);
                        j += 4;
                    }
                }
                asm!("bar.warp.sync 0xffffffff;");
                if c + FOLD_SLOTS < chunks { asm!("bar.arrive {id}, {n};", id = in(reg32) FOLD_EMPTY + s as u32, n = in(reg32) n); }
                c += 1;
            }
            if live { sum } else { 0.0 }
        } else {
            let p = tid - 32;
            macro_rules! ld4 {($a:expr)=>{{
                let (a,b,c,d):(f32,f32,f32,f32);
                asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$a);
                [a,b,c,d]
            }};}
            macro_rules! mul {($w:expr,$x:expr)=>{{
                let product: f32;
                asm!("mul.rn.f32 {product}, {w}, {x};", product = out(reg32) product, w = in(reg32) $w, x = in(reg32) $x, options(nomem, nostack));
                product.to_bits()
            }};}
            // LOAD AHEAD (seat CHATS4): the global loads of chunk c + 1 issue before the products of chunk c are made, into the
            // other of two register sets (`a` and `b`, used in turn), so a producer has two chunks of loads in the memory system
            // and a chunk no longer waits a whole load round trip. The products, their slots, the barriers and the adds are
            // those of one chunk at a time, as before.
            // Product vector j of this producer: weight vector slot k, source row t, block weight row r, vector q of the chunk.
            // A warp reads 32 neighbouring vectors of one weight row. A weight vector is loaded once (at its first source row) and
            // used for every source row.
            let (mut wa, mut xa) = ([[0.0f32; 4]; 8], [[0.0f32; 4]; 8]);
            let (mut wb, mut xb) = ([[0.0f32; 4]; 8], [[0.0f32; 4]; 8]);
            macro_rules! fetch {($w:ident, $x:ident, $c:expr, $j:expr)=>{{
                let first = $c * ch; let span = (cols - first).min(ch);
                let (k, t, r, q) = fold_rows_vector(LAYOUT, p, $j);
                if q * 4 < span {
                    if t < tokens { $x[$j] = ld4!(input.add(t * cols + first + q * 4)); }
                    if t == 0 && first_row + r < rows {
                        let at=(first_row+r)*cols+first+q*4;
                        $w[k]=if BF16 {
                            let (a,b):(u32,u32);
                            asm!("ld.global.v2.u32 {{{a},{b}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,p=in(reg64)weights.cast::<u16>().add(at));
                            [f32::from_bits(a<<16),f32::from_bits(a&0xffff0000),f32::from_bits(b<<16),f32::from_bits(b&0xffff0000)]
                        } else {ld4!(weights.add(at))};
                    }
                }
            }};}
            macro_rules! fetch_all {($w:ident, $x:ident, $c:expr)=>{{
                fetch!($w, $x, $c, 0); fetch!($w, $x, $c, 1); fetch!($w, $x, $c, 2); fetch!($w, $x, $c, 3);
                fetch!($w, $x, $c, 4); fetch!($w, $x, $c, 5); fetch!($w, $x, $c, 6); fetch!($w, $x, $c, 7);
            }};}
            // Chunk c from one register set: wait for its slot, make and store its products, mark the slot full.
            macro_rules! produce {($w:ident, $x:ident, $c:expr)=>{{
                let c = $c;
                let s = c % FOLD_SLOTS;
                if c >= FOLD_SLOTS { asm!("bar.sync {id}, {n};", id = in(reg32) FOLD_EMPTY + s as u32, n = in(reg32) n); }
                let first = c * ch;
                let span = (cols - first).min(ch);
                let slot = smem + (s * slot_bytes) as u32;
                macro_rules! put {($j:expr)=>{{
                    let (k, t, r, q) = fold_rows_vector(LAYOUT, p, $j);
                    if q * 4 < span && first_row + r < rows && t < tokens {
                        let (a, b, cc, d) = (mul!($w[k][0], $x[$j][0]), mul!($w[k][1], $x[$j][1]), mul!($w[k][2], $x[$j][2]), mul!($w[k][3], $x[$j][3]));
                        asm!("st.shared.v4.b32 [{at}], {{{a}, {b}, {c}, {d}}};", at = in(reg32) slot + (fold_rows_word(LAYOUT, t, r, q * 4) * 4) as u32, a = in(reg32) a, b = in(reg32) b, c = in(reg32) cc, d = in(reg32) d);
                    }
                }};}
                put!(0); put!(1); put!(2); put!(3); put!(4); put!(5); put!(6); put!(7);
                asm!("bar.warp.sync 0xffffffff;");
                asm!("bar.arrive {id}, {n};", id = in(reg32) FOLD_FULL + s as u32, n = in(reg32) n);
            }};}
            fetch_all!(wa, xa, 0);
            let mut c = 0;
            while c < chunks {
                if c + 1 < chunks { fetch_all!(wb, xb, c + 1); }
                produce!(wa, xa, c);
                c += 1;
                if c >= chunks { break; }
                if c + 1 < chunks { fetch_all!(wa, xa, c + 1); }
                produce!(wb, xb, c);
                c += 1;
            }
            0.0
        }
    }
}
