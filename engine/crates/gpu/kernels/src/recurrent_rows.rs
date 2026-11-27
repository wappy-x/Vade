//! One CTA per head. Its warps prepare different tokens, then each thread
//! keeps its value column across the ordered recurrence.
use crate::causal_ops::{root, round, sigmoid};
use crate::next_math::exp;
use crate::ptx;
use crate::recurrent_rows::{Arguments, DIM, MAX_TOKENS, TAPS, THREADS};
use crate::strip_warp::{add, mul, unroll};
use core::arch::asm;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};

// The four vectors (Q, K, decay, target), beta and padding. Every vector
// starts at a 16-byte boundary for the four-word loads in the state folds.
const PREPARED_BYTES: u32 = ((4 * DIM + 4) * 4) as u32;
const STEPPED_BYTES: u32 = (DIM * 4) as u32;
const SHARED: usize = MAX_TOKENS as usize * (PREPARED_BYTES + STEPPED_BYTES) as usize;

/// Add one lane's four squares, then use the Prepare and GatedNorm tree at
/// distances 16, 8, 4, 2, 1. All 32 lanes take part.
macro_rules! square_sum {
    ($lane:expr,$source:expr) => {{
        let mut total = 0.0f32;
        unroll!(4, i, {
            let value = ($source)(i);
            total += value * value;
        });
        let mut distance = 16u32;
        while distance != 0 {
            let other: f32;
            asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",
                o=out(reg32)other,v=in(reg32)total,d=in(reg32)distance,options(nomem,nostack));
            if $lane + (distance as usize) < 32 { total += other; }
            distance /= 2;
        }
        let sum: f32;
        asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",
            o=out(reg32)sum,v=in(reg32)total,options(nomem,nostack));
        sum
    }};
}

#[no_mangle]
pub extern "ptx-kernel" fn recurrent_rows(p: Arguments) { body(p) }

#[inline(always)]
pub(crate) fn body(p: Arguments) {
    // SAFETY: The host validates Shape::lengths, the gate extent, contexts
    // and contiguity. The launch assigns one block per head and DIM threads
    // per block. Inputs stay live and immutable. Output owners are disjoint.
    // A warp owns its token's prepared shared words. Each lane writes four
    // features in strides of 32. Lane zero also writes beta. The first CTA
    // barrier precedes any read of those words. Each thread then owns its
    // state column and its recurrent feature of every token. The second
    // CTA barrier precedes any read of those features. Both barriers are
    // unconditional. Every warp uses the same token bounds within the warp,
    // so its shuffles are converged. Global stores are disjoint per head,
    // token and feature. The optional output stores use nonzero owners.
    unsafe {
        let s = p.shape;
        let keep = p.outputs[3] != 0;
        if s.lengths(keep).is_none() || _block_dim_x() != THREADS || _grid_dim_x() != s.heads
            || keep != (p.outputs[4] != 0) { core::arch::nvptx::trap(); }
        let head = _block_idx_x() as usize;
        let column_id = _thread_idx_x() as usize;
        let lane = column_id % 32;
        let warp = column_id / 32;
        let heads = s.heads as usize;
        let tokens = s.tokens as usize;
        let width = heads * DIM;
        let head_offset = head * DIM;
        let input = p.inputs.map(|addr| addr as *const f32);
        let [conv_weight, prep_weight, norm_weight] = p.weights.map(|addr| addr as *const f32);
        let [history_out, state_out, gated_out, prepared_out, decay_out] = p.outputs.map(|addr| addr as *mut f32);
        let gate = p.gate as *const f32;
        let base = ptx::shared::<SHARED>();
        let stepped = base + MAX_TOKENS * PREPARED_BYTES;
        let vector_bytes = STEPPED_BYTES;

        let old_or_row = |plane: usize, position: usize, feature: usize| -> f32 {
            let history = TAPS - 1;
            if position < history {
                *input[3].add((plane * history + position) * width + feature)
            } else {
                *input[plane].add((position - history) * width + feature)
            }
        };
        let convolve = |plane: usize, token: usize, feature: usize| -> f32 {
            let weight = conv_weight.add((plane * width + feature) * TAPS);
            let mut sum = 0.0f32;
            unroll!(4, tap, { sum += old_or_row(plane, token + tap, feature) * *weight.add(tap); });
            round(sum * sigmoid(sum))
        };
        let rate = exp(*prep_weight.add(head));
        let query_scale = 1.0 / root(DIM as f32);
        let mut token = warp;
        while token < tokens {
            let prepared = base + token as u32 * PREPARED_BYTES;
            let row = token * width + head_offset;
            let saved = (token * heads + head) * (4 * DIM + 1);
            let mut queries = [0.0f32; 4];
            let mut keys = [0.0f32; 4];
            unroll!(4, i, {
                let feature = head_offset + lane + 32 * i;
                queries[i] = convolve(0, token, feature);
                keys[i] = convolve(1, token, feature);
            });
            let qnorm = root(square_sum!(lane, |i| queries[i]) + 1e-6);
            let knorm = root(square_sum!(lane, |i| keys[i]) + 1e-6);
            unroll!(4, i, {
                let j = lane + 32 * i;
                let offset = (4 * j) as u32;
                let query = (queries[i] / qnorm) * query_scale;
                let key = keys[i] / knorm;
                let target = convolve(2, token, head_offset + j);
                let log_decay = s.bound * sigmoid(rate * (*input[4].add(row + j) + *prep_weight.add(heads + head_offset + j)));
                let factor = exp(log_decay);
                ptx::store(prepared + offset, query);
                ptx::store(prepared + vector_bytes + offset, key);
                ptx::store(prepared + 2 * vector_bytes + offset, factor);
                ptx::store(prepared + 3 * vector_bytes + offset, target);
                if keep {
                    *prepared_out.add(saved + j) = query;
                    *prepared_out.add(saved + DIM + j) = key;
                    *prepared_out.add(saved + 2 * DIM + j) = target;
                    *prepared_out.add(saved + 3 * DIM + j) = log_decay;
                    *decay_out.add(row + j) = factor;
                }
            });
            if lane == 0 {
                let beta = round(sigmoid(*input[5].add(token * heads + head)));
                ptx::store(prepared + 4 * vector_bytes, beta);
                if keep { *prepared_out.add(saved + 4 * DIM) = beta; }
            }
            token += 4;
        }
        ptx::sync();

        let mut column = [0.0f32; DIM];
        let state = input[6].add(head * DIM * DIM + column_id);
        unroll!(8, part, { unroll!(4, group, { unroll!(4, i, {
            let k = (part * 4 + group) * 4 + i;
            column[k] = *state.add(k * DIM);
        }); }); });

        token = 0;
        while token < tokens {
            let query = base + token as u32 * PREPARED_BYTES;
            let key = query + vector_bytes;
            let decay = query + 2 * vector_bytes;
            let target = ptx::load(query + 3 * vector_bytes + (column_id * 4) as u32);
            let beta = ptx::load(query + 4 * vector_bytes);

            // Each fold visits keys in ascending order, like prompt_rows_small.
            let mut prediction = 0.0f32;
            unroll!(8, part, { unroll!(4, group, {
                let first = (part * 4 + group) * 4;
                let offset = (first * 4) as u32;
                let factors = ptx::load4(decay + offset);
                let keys = ptx::load4(key + offset);
                unroll!(4, i, {
                    let value = mul!(column[first + i], f32::from_bits(factors[i]));
                    column[first + i] = value;
                    add!(prediction, mul!(value, f32::from_bits(keys[i])));
                });
            }); });
            let delta = mul!(target - prediction, beta);
            let mut sum = 0.0f32;
            unroll!(8, part, { unroll!(4, group, {
                let first = (part * 4 + group) * 4;
                let offset = (first * 4) as u32;
                let keys = ptx::load4(key + offset);
                let queries = ptx::load4(query + offset);
                unroll!(4, i, {
                    let mut value = column[first + i];
                    add!(value, mul!(f32::from_bits(keys[i]), delta));
                    column[first + i] = value;
                    add!(sum, mul!(value, f32::from_bits(queries[i])));
                });
            }); });
            ptx::store(stepped + token as u32 * STEPPED_BYTES + (column_id * 4) as u32, round(sum));
            token += 1;
        }
        let final_column = state_out.add(head * DIM * DIM + column_id);
        unroll!(8, part, { unroll!(4, group, { unroll!(4, i, {
            let k = (part * 4 + group) * 4 + i;
            *final_column.add(k * DIM) = column[k];
        }); }); });
        ptx::sync();

        token = warp;
        while token < tokens {
            let recurrent = stepped + token as u32 * STEPPED_BYTES;
            let row = token * width + head_offset;
            let read = |i| ptx::load(recurrent + ((lane + 32 * i) * 4) as u32);
            let scale = 1.0 / root(square_sum!(lane, read) / DIM as f32 + s.epsilon);
            unroll!(4, i, {
                let j = lane + 32 * i;
                *gated_out.add(row + j) = round(((read(i) * scale) * *norm_weight.add(j)) * sigmoid(*gate.add(row + j)));
            });
            token += 4;
        }

        let feature = head_offset + column_id;
        unroll!(3, plane, { unroll!(3, tap, {
            *history_out.add((plane * (TAPS - 1) + tap) * width + feature) = old_or_row(plane, tokens + tap, feature);
        }); });
    }
}
