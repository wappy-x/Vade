//! Few-row NVFP4 draft projections and fold reduction.
use crate::fixed_math::{exp, rn, tanh};
use crate::tc_ptx::{shfl_xor_f32, shfl_xor_u32};

/// Last `PIN_FOLD_PANELS` 16-row panels of the `200,064 x 256` fold matrix (~22.8 MB including bias)
/// stay L2-resident across consecutive Markov steps on GB10's 24 MiB L2; pinning the tail panels that
/// `HeadFold` finishes last keeps them warm in L2 right into Markov step 1 (`FoldPick` #1).
const PIN_FOLD_PANELS: u32 = 9_600;

macro_rules! l2_evict_first {
    () => {{
        let p: u64;
        core::arch::asm!("createpolicy.fractional.L2::evict_first.b64 {p}, 1.0;", p = out(reg64) p, options(nomem, nostack));
        p
    }};
}

macro_rules! l2_evict_last {
    () => {{
        let p: u64;
        core::arch::asm!("createpolicy.fractional.L2::evict_last.b64 {p}, 1.0;", p = out(reg64) p, options(nomem, nostack));
        p
    }};
}

macro_rules! ld_code_v4 {
    ($p:expr, $policy:expr) => {{
        let (a, b, c, d): (u32, u32, u32, u32);
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}], {q};",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
            p = in(reg64) $p, q = in(reg64) $policy, options(readonly, nostack)
        );
        [a, b, c, d]
    }};
}

macro_rules! ld_scale_u32 {
    ($p:expr, $policy:expr) => {{
        let a: u32;
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.L2::128B.u32 {a}, [{p}], {q};",
            a = out(reg32) a, p = in(reg64) $p, q = in(reg64) $policy, options(readonly, nostack)
        );
        a
    }};
}

macro_rules! ld_f32_nc {
    ($p:expr) => {{
        let a: f32;
        core::arch::asm!(
            "ld.global.nc.f32 {a}, [{p}];",
            a = out(reg32) a, p = in(reg64) $p, options(readonly, nostack)
        );
        a
    }};
}

macro_rules! ld_f32_hint {
    ($p:expr, $policy:expr) => {{
        let a: f32;
        core::arch::asm!(
            "ld.global.nc.L2::cache_hint.f32 {a}, [{p}], {q};",
            a = out(reg32) a, p = in(reg64) $p, q = in(reg64) $policy, options(readonly, nostack)
        );
        a
    }};
}

macro_rules! ld_u16_hint {
    ($p:expr, $policy:expr) => {{
        let a: u32;
        core::arch::asm!(
            "{{ .reg .u16 t; ld.global.nc.L2::cache_hint.u16 t, [{p}], {q}; cvt.u32.u16 {a}, t; }}",
            a = out(reg32) a, p = in(reg64) $p, q = in(reg64) $policy, options(readonly, nostack)
        );
        a
    }};
}

macro_rules! st_u32_hint {
    ($p:expr, $v:expr, $policy:expr) => {{
        core::arch::asm!(
            "st.global.L2::cache_hint.u32 [{p}], {v}, {q};",
            p = in(reg64) $p, v = in(reg32) $v, q = in(reg64) $policy, options(nostack)
        );
    }};
}

/// Pack and stream one 16x8 BF16-rounded tile to `[limit - skip, row_words]` packed `u32` BF16 pairs.
/// All 32 lanes of the warp must enter unconditionally (`shfl_xor_u32(..., 4)` pairs `group` with `group ^ 1`).
macro_rules! store_bf16_tile {
    (no_skip, $t_base:expr, $limit:expr, $h:expr, $row:expr, $a:expr, $row_words:expr, $pair:expr, $group:expr, $out:expr, $evict:expr) => {{
        let lo0 = rn(($h)[0]).to_bits() >> 16;
        let hi0 = shfl_xor_u32(lo0, 4);
        let lo2 = rn(($h)[2]).to_bits() >> 16;
        let hi2 = shfl_xor_u32(lo2, 4);
        let t0 = ($t_base) + ($pair) * 2;
        if (($group) & 1) == 0 && t0 < ($limit) && ($row) < ($a) {
            let dst = (($out) as *mut u32).add((t0 as usize) * ($row_words) + (($row) >> 1) as usize);
            let w0 = lo0 | (if ($row) + 1 < ($a) { hi0 << 16 } else { 0 });
            st_u32_hint!(dst, w0, $evict);
            if ($row) + 8 < ($a) {
                let w2 = lo2 | (if ($row) + 9 < ($a) { hi2 << 16 } else { 0 });
                st_u32_hint!(dst.add(4), w2, $evict);
            }
        }
        let lo1 = rn(($h)[1]).to_bits() >> 16;
        let hi1 = shfl_xor_u32(lo1, 4);
        let lo3 = rn(($h)[3]).to_bits() >> 16;
        let hi3 = shfl_xor_u32(lo3, 4);
        let t1 = t0 + 1;
        if (($group) & 1) == 0 && t1 < ($limit) && ($row) < ($a) {
            let dst = (($out) as *mut u32).add((t1 as usize) * ($row_words) + (($row) >> 1) as usize);
            let w1 = lo1 | (if ($row) + 1 < ($a) { hi1 << 16 } else { 0 });
            st_u32_hint!(dst, w1, $evict);
            if ($row) + 8 < ($a) {
                let w3 = lo3 | (if ($row) + 9 < ($a) { hi3 << 16 } else { 0 });
                st_u32_hint!(dst.add(4), w3, $evict);
            }
        }
    }};
    ($t_base:expr, $skip:expr, $limit:expr, $h:expr, $row:expr, $a:expr, $row_words:expr, $pair:expr, $group:expr, $out:expr, $evict:expr) => {{
        let lo0 = rn(($h)[0]).to_bits() >> 16;
        let hi0 = shfl_xor_u32(lo0, 4);
        let lo2 = rn(($h)[2]).to_bits() >> 16;
        let hi2 = shfl_xor_u32(lo2, 4);
        let t0 = ($t_base) + ($pair) * 2;
        if (($group) & 1) == 0 && t0 >= ($skip) && t0 < ($limit) && ($row) < ($a) {
            let dst = (($out) as *mut u32).add(((t0 - ($skip)) as usize) * ($row_words) + (($row) >> 1) as usize);
            let w0 = lo0 | (if ($row) + 1 < ($a) { hi0 << 16 } else { 0 });
            st_u32_hint!(dst, w0, $evict);
            if ($row) + 8 < ($a) {
                let w2 = lo2 | (if ($row) + 9 < ($a) { hi2 << 16 } else { 0 });
                st_u32_hint!(dst.add(4), w2, $evict);
            }
        }
        let lo1 = rn(($h)[1]).to_bits() >> 16;
        let hi1 = shfl_xor_u32(lo1, 4);
        let lo3 = rn(($h)[3]).to_bits() >> 16;
        let hi3 = shfl_xor_u32(lo3, 4);
        let t1 = t0 + 1;
        if (($group) & 1) == 0 && t1 >= ($skip) && t1 < ($limit) && ($row) < ($a) {
            let dst = (($out) as *mut u32).add(((t1 - ($skip)) as usize) * ($row_words) + (($row) >> 1) as usize);
            let w1 = lo1 | (if ($row) + 1 < ($a) { hi1 << 16 } else { 0 });
            st_u32_hint!(dst, w1, $evict);
            if ($row) + 8 < ($a) {
                let w3 = lo3 | (if ($row) + 9 < ($a) { hi3 << 16 } else { 0 });
                st_u32_hint!(dst.add(4), w3, $evict);
            }
        }
    }};
}

/// Consume one k=64 NVFP4 chunk using quad-interleaved OMMA.SF (`{0,0}` for even chunks, `{0,1}` for odd)
/// and paired 2-pass SFB (`{0,0}` then `{0,1}`), keeping all accumulators in scalar registers.
macro_rules! consume_chunk {
    (0, $in_base:expr, $chunk:expr, $pair:expr, $w:expr, $sa:expr, $acc:expr) => {{
        let input = ($in_base).add((($chunk) * 18) as usize);
        let sb = *input.add((16 + (($pair) & 1)) as usize);
        let (v0, v1) = (*input.add(($pair) as usize), *input.add((($pair) + 4) as usize));
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        core::arch::asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,0}};",
            d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
            a0 = in(reg32) ($w)[0], a1 = in(reg32) ($w)[1], a2 = in(reg32) ($w)[2], a3 = in(reg32) ($w)[3],
            b0 = in(reg32) v0, b1 = in(reg32) v1,
            c0 = in(reg32) ($acc)[0], c1 = in(reg32) ($acc)[1], c2 = in(reg32) ($acc)[2], c3 = in(reg32) ($acc)[3],
            sa = in(reg32) $sa, sb = in(reg32) sb, options(nomem, nostack)
        );
        let (v2, v3) = (*input.add((8 + ($pair)) as usize), *input.add((12 + ($pair)) as usize));
        let (e0, e1, e2, e3): (f32, f32, f32, f32);
        core::arch::asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,1}};",
            d0 = out(reg32) e0, d1 = out(reg32) e1, d2 = out(reg32) e2, d3 = out(reg32) e3,
            a0 = in(reg32) ($w)[0], a1 = in(reg32) ($w)[1], a2 = in(reg32) ($w)[2], a3 = in(reg32) ($w)[3],
            b0 = in(reg32) v2, b1 = in(reg32) v3,
            c0 = in(reg32) d0, c1 = in(reg32) d1, c2 = in(reg32) d2, c3 = in(reg32) d3,
            sa = in(reg32) $sa, sb = in(reg32) sb, options(nomem, nostack)
        );
        [e0, e1, e2, e3]
    }};
    (1, $in_base:expr, $chunk:expr, $pair:expr, $w:expr, $sa:expr, $acc:expr) => {{
        let input = ($in_base).add((($chunk) * 18) as usize);
        let sb = *input.add((16 + (($pair) & 1)) as usize);
        let (v0, v1) = (*input.add(($pair) as usize), *input.add((($pair) + 4) as usize));
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        core::arch::asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,1}}, {sb}, {{0,0}};",
            d0 = out(reg32) d0, d1 = out(reg32) d1, d2 = out(reg32) d2, d3 = out(reg32) d3,
            a0 = in(reg32) ($w)[0], a1 = in(reg32) ($w)[1], a2 = in(reg32) ($w)[2], a3 = in(reg32) ($w)[3],
            b0 = in(reg32) v0, b1 = in(reg32) v1,
            c0 = in(reg32) ($acc)[0], c1 = in(reg32) ($acc)[1], c2 = in(reg32) ($acc)[2], c3 = in(reg32) ($acc)[3],
            sa = in(reg32) $sa, sb = in(reg32) sb, options(nomem, nostack)
        );
        let (v2, v3) = (*input.add((8 + ($pair)) as usize), *input.add((12 + ($pair)) as usize));
        let (e0, e1, e2, e3): (f32, f32, f32, f32);
        core::arch::asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,1}}, {sb}, {{0,1}};",
            d0 = out(reg32) e0, d1 = out(reg32) e1, d2 = out(reg32) e2, d3 = out(reg32) e3,
            a0 = in(reg32) ($w)[0], a1 = in(reg32) ($w)[1], a2 = in(reg32) ($w)[2], a3 = in(reg32) ($w)[3],
            b0 = in(reg32) v2, b1 = in(reg32) v3,
            c0 = in(reg32) d0, c1 = in(reg32) d1, c2 = in(reg32) d2, c3 = in(reg32) d3,
            sa = in(reg32) $sa, sb = in(reg32) sb, options(nomem, nostack)
        );
        [e0, e1, e2, e3]
    }};
}

/// Project one 16-row NVFP4 panel against `inputs` packed activation rows with hoisted L2 cache policy,
/// exact `chunks == 4` fast path, 6-chunk software-pipelined scalar ring, and zero tail over-fetch.
macro_rules! project_panel {
    ($x:expr, $y:expr, $a:expr, $b:expr, $inputs:expr, $panel:expr, $lane:expr, $group:expr, $pair:expr, $pin_l2:expr) => {{
        let (x, y, a, b, inputs, panel, lane, group, pair, pin_l2) = ($x, $y, $a, $b, $inputs, $panel, $lane, $group, $pair, $pin_l2);
        let elements = (a as usize).next_multiple_of(64) * (b as usize);
        let scales = x.cast::<u8>().add(elements / 2);
        let factor = *scales.add(elements / 16).cast::<f32>();
        let chunks = b >> 6;
        let last = chunks - 1;
        let base_stripe = ((panel >> 2) * chunks) * 4 + (panel & 3);
        let code_base = x.cast::<u32>().add((base_stripe as usize) * 128 + (lane as usize) * 4);
        let scale_base = scales.cast::<u32>().add((base_stripe as usize) * 16 + ((group * 2 + (pair & 1)) as usize));
        let in_base = y.cast::<u32>().add((group.min(inputs - 1) * chunks * 18) as usize);
        let policy = if pin_l2 { l2_evict_last!() } else { l2_evict_first!() };
        let hi_pair = (pair >> 1) & 1;
        let mut acc = [0.0f32; 4];

        if chunks == 4 {
            let w0 = ld_code_v4!(code_base, policy);
            let w1 = ld_code_v4!(code_base.add(512), policy);
            let sa01 = ld_scale_u32!(scale_base.add((hi_pair * 64) as usize), policy);
            let w2 = ld_code_v4!(code_base.add(1024), policy);
            let w3 = ld_code_v4!(code_base.add(1536), policy);
            let sa23 = ld_scale_u32!(scale_base.add(((2 + hi_pair) * 64) as usize), policy);
            acc = consume_chunk!(0, in_base, 0, pair, w0, sa01, acc);
            acc = consume_chunk!(1, in_base, 1, pair, w1, sa01, acc);
            acc = consume_chunk!(0, in_base, 2, pair, w2, sa23, acc);
            acc = consume_chunk!(1, in_base, 3, pair, w3, sa23, acc);
        } else if chunks >= 6 {
            let mut w0 = ld_code_v4!(code_base, policy);
            let mut w1 = ld_code_v4!(code_base.add(512), policy);
            let mut sa01 = ld_scale_u32!(scale_base.add((hi_pair * 64) as usize), policy);
            let mut w2 = ld_code_v4!(code_base.add(1024), policy);
            let mut w3 = ld_code_v4!(code_base.add(1536), policy);
            let mut sa23 = ld_scale_u32!(scale_base.add(((2 + hi_pair) * 64) as usize), policy);
            let mut w4 = ld_code_v4!(code_base.add(2048), policy);
            let mut w5 = ld_code_v4!(code_base.add(2560), policy);
            let mut sa45 = ld_scale_u32!(scale_base.add(((4 + hi_pair) * 64) as usize), policy);

            let mut chunk = 0u32;
            while chunk + 12 <= chunks {
                acc = consume_chunk!(0, in_base, chunk, pair, w0, sa01, acc);
                w0 = ld_code_v4!(code_base.add(((chunk + 6) as usize) * 512), policy);
                acc = consume_chunk!(1, in_base, chunk + 1, pair, w1, sa01, acc);
                w1 = ld_code_v4!(code_base.add(((chunk + 7) as usize) * 512), policy);
                sa01 = ld_scale_u32!(scale_base.add(((chunk + 6 + hi_pair) as usize) * 64), policy);

                acc = consume_chunk!(0, in_base, chunk + 2, pair, w2, sa23, acc);
                w2 = ld_code_v4!(code_base.add(((chunk + 8) as usize) * 512), policy);
                acc = consume_chunk!(1, in_base, chunk + 3, pair, w3, sa23, acc);
                w3 = ld_code_v4!(code_base.add(((chunk + 9) as usize) * 512), policy);
                sa23 = ld_scale_u32!(scale_base.add(((chunk + 8 + hi_pair) as usize) * 64), policy);

                acc = consume_chunk!(0, in_base, chunk + 4, pair, w4, sa45, acc);
                w4 = ld_code_v4!(code_base.add(((chunk + 10) as usize) * 512), policy);
                acc = consume_chunk!(1, in_base, chunk + 5, pair, w5, sa45, acc);
                w5 = ld_code_v4!(code_base.add(((chunk + 11) as usize) * 512), policy);
                sa45 = ld_scale_u32!(scale_base.add(((chunk + 10 + hi_pair) as usize) * 64), policy);

                chunk += 6;
            }
            acc = consume_chunk!(0, in_base, chunk, pair, w0, sa01, acc);
            acc = consume_chunk!(1, in_base, chunk + 1, pair, w1, sa01, acc);
            acc = consume_chunk!(0, in_base, chunk + 2, pair, w2, sa23, acc);
            acc = consume_chunk!(1, in_base, chunk + 3, pair, w3, sa23, acc);
            acc = consume_chunk!(0, in_base, chunk + 4, pair, w4, sa45, acc);
            acc = consume_chunk!(1, in_base, chunk + 5, pair, w5, sa45, acc);
            chunk += 6;
            while chunk < chunks {
                let tw0 = ld_code_v4!(code_base.add((chunk as usize) * 512), policy);
                let sa = ld_scale_u32!(scale_base.add((last.min(chunk + hi_pair) as usize) * 64), policy);
                if chunk + 1 < chunks {
                    let tw1 = ld_code_v4!(code_base.add(((chunk + 1) as usize) * 512), policy);
                    acc = consume_chunk!(0, in_base, chunk, pair, tw0, sa, acc);
                    acc = consume_chunk!(1, in_base, chunk + 1, pair, tw1, sa, acc);
                } else {
                    acc = consume_chunk!(0, in_base, chunk, pair, tw0, sa, acc);
                }
                chunk += 2;
            }
        } else {
            let mut c = 0u32;
            while c < chunks {
                let w0 = ld_code_v4!(code_base.add((c as usize) * 512), policy);
                let sa = ld_scale_u32!(scale_base.add((last.min(c + hi_pair) as usize) * 64), policy);
                if c + 1 < chunks {
                    let w1 = ld_code_v4!(code_base.add(((c + 1) as usize) * 512), policy);
                    acc = consume_chunk!(0, in_base, c, pair, w0, sa, acc);
                    acc = consume_chunk!(1, in_base, c + 1, pair, w1, sa, acc);
                } else {
                    acc = consume_chunk!(0, in_base, c, pair, w0, sa, acc);
                }
                c += 2;
            }
        }
        [acc[0] * factor, acc[1] * factor, acc[2] * factor, acc[3] * factor]
    }};
}

/// Project one 16-row NVFP4 panel against `9..=16` packed activation rows in ONE pass over the weight panel
/// (`1x` DRAM weight reads for both 8-input MMA N-tiles), using a 2-chunk scalar ring (`w0, w1, sa01`)
/// to keep register pressure flat (`0` stack, `0` spill) while hiding DRAM latency across two MMA tiles.
macro_rules! project_panel_two {
    ($x:expr, $y:expr, $a:expr, $b:expr, $inputs:expr, $panel:expr, $lane:expr, $group:expr, $pair:expr, $pin_l2:expr) => {{
        let (x, y, a, b, inputs, panel, lane, group, pair, pin_l2) = ($x, $y, $a, $b, $inputs, $panel, $lane, $group, $pair, $pin_l2);
        let elements = (a as usize).next_multiple_of(64) * (b as usize);
        let scales = x.cast::<u8>().add(elements / 2);
        let factor = *scales.add(elements / 16).cast::<f32>();
        let chunks = b >> 6;
        let last = chunks - 1;
        let base_stripe = ((panel >> 2) * chunks) * 4 + (panel & 3);
        let code_base = x.cast::<u32>().add((base_stripe as usize) * 128 + (lane as usize) * 4);
        let scale_base = scales.cast::<u32>().add((base_stripe as usize) * 16 + ((group * 2 + (pair & 1)) as usize));
        let in_base0 = y.cast::<u32>().add((group * chunks * 18) as usize);
        let in_base1 = y.cast::<u32>().add(((8 + group.min(inputs - 9)) * chunks * 18) as usize);
        let policy = if pin_l2 { l2_evict_last!() } else { l2_evict_first!() };
        let hi_pair = (pair >> 1) & 1;
        let mut acc0 = [0.0f32; 4];
        let mut acc1 = [0.0f32; 4];

        if chunks >= 2 {
            let mut w0 = ld_code_v4!(code_base, policy);
            let mut w1 = ld_code_v4!(code_base.add(512), policy);
            let mut sa01 = ld_scale_u32!(scale_base.add((hi_pair * 64) as usize), policy);

            let mut chunk = 0u32;
            while chunk + 4 <= chunks {
                acc0 = consume_chunk!(0, in_base0, chunk, pair, w0, sa01, acc0);
                acc1 = consume_chunk!(0, in_base1, chunk, pair, w0, sa01, acc1);
                w0 = ld_code_v4!(code_base.add(((chunk + 2) as usize) * 512), policy);
                acc0 = consume_chunk!(1, in_base0, chunk + 1, pair, w1, sa01, acc0);
                acc1 = consume_chunk!(1, in_base1, chunk + 1, pair, w1, sa01, acc1);
                w1 = ld_code_v4!(code_base.add(((chunk + 3) as usize) * 512), policy);
                sa01 = ld_scale_u32!(scale_base.add(((chunk + 2 + hi_pair) as usize) * 64), policy);
                chunk += 2;
            }
            acc0 = consume_chunk!(0, in_base0, chunk, pair, w0, sa01, acc0);
            acc1 = consume_chunk!(0, in_base1, chunk, pair, w0, sa01, acc1);
            acc0 = consume_chunk!(1, in_base0, chunk + 1, pair, w1, sa01, acc0);
            acc1 = consume_chunk!(1, in_base1, chunk + 1, pair, w1, sa01, acc1);
            chunk += 2;
            if chunk < chunks {
                let tw0 = ld_code_v4!(code_base.add((chunk as usize) * 512), policy);
                let sa = ld_scale_u32!(scale_base.add((last as usize) * 64), policy);
                acc0 = consume_chunk!(0, in_base0, chunk, pair, tw0, sa, acc0);
                acc1 = consume_chunk!(0, in_base1, chunk, pair, tw0, sa, acc1);
            }
        } else {
            let w0 = ld_code_v4!(code_base, policy);
            let sa = ld_scale_u32!(scale_base, policy);
            acc0 = consume_chunk!(0, in_base0, 0u32, pair, w0, sa, acc0);
            acc1 = consume_chunk!(0, in_base1, 0u32, pair, w0, sa, acc1);
        }
        (
            [acc0[0] * factor, acc0[1] * factor, acc0[2] * factor, acc0[3] * factor],
            [acc1[0] * factor, acc1[1] * factor, acc1[2] * factor, acc1[3] * factor],
        )
    }};
}

/// Project one 16-row NVFP4 panel against `9..=PANEL_INPUTS_MAX` packed activation rows using `slabs` (a power of
/// 2 up to 16) warps a panel, neighbours in the grid (each warp handling 16-input waves strided by `slabs * 16`).
/// Up to 4 warps a panel sit in one 128-thread CTA, so they read each weight byte from DRAM once and from L1 after;
/// the host gives a small matrix a warp for each 16 inputs (one wave, 2 or 4 CTAs a panel that share L2).
/// Up to `PANEL_NARROW_INPUTS_MAX` inputs of a small matrix take warps of 8 inputs through `project_panel!`.
macro_rules! project_panel_wide {
    ($x:expr, $y:expr, $a:expr, $b:expr, $inputs:expr, $panel:expr, $slab:expr, $slabs:expr, $lane:expr, $group:expr, $pair:expr, $pin_l2:expr, $out:expr) => {{
        let (x, y, a, b, inputs, panel, slab, slabs, lane, group, pair, pin_l2, out) = ($x, $y, $a, $b, $inputs, $panel, $slab, $slabs, $lane, $group, $pair, $pin_l2, $out);
        let elements = (a as usize).next_multiple_of(64) * (b as usize);
        let scales = x.cast::<u8>().add(elements / 2);
        let chunks = b >> 6;
        let last = chunks - 1;
        let base_stripe = ((panel >> 2) * chunks) * 4 + (panel & 3);
        let code_base = x.cast::<u32>().add((base_stripe as usize) * 128 + (lane as usize) * 4);
        let scale_base = scales.cast::<u32>().add((base_stripe as usize) * 16 + ((group * 2 + (pair & 1)) as usize));
        let policy = if pin_l2 || inputs > slabs * 16 { l2_evict_last!() } else { l2_evict_first!() };
        let hi_pair = (pair >> 1) & 1;
        let row = panel * 16 + group;
        let stride_in = slabs * 16;
        macro_rules! store_slab {
            ($base:expr, $acc:expr, $factor:expr) => {{
                let t0 = ($base) + pair * 2;
                if t0 < inputs && row < a {
                    let dst = (out as *mut f32).add((t0 as usize) * (a as usize) + (row as usize));
                    *dst = rn(($acc)[0] * $factor);
                    if row + 8 < a { *dst.add(8) = rn(($acc)[2] * $factor); }
                }
                let t1 = t0 + 1;
                if t1 < inputs && row < a {
                    let dst = (out as *mut f32).add((t1 as usize) * (a as usize) + (row as usize));
                    *dst = rn(($acc)[1] * $factor);
                    if row + 8 < a { *dst.add(8) = rn(($acc)[3] * $factor); }
                }
            }};
        }
        let mut base_in = slab * 16;
        while base_in < inputs {
            let in0 = y.cast::<u32>().add(((base_in + group).min(inputs - 1) * chunks * 18) as usize);
            let in1 = y.cast::<u32>().add(((base_in + 8 + group).min(inputs - 1) * chunks * 18) as usize);
            let mut acc0 = [0.0f32; 4];
            let mut acc1 = [0.0f32; 4];
            let mut c = 0u32;
            while c < chunks {
                let sa = ld_scale_u32!(scale_base.add((last.min(c + hi_pair) as usize) * 64), policy);
                let w = ld_code_v4!(code_base.add((c as usize) * 512), policy);
                acc0 = consume_chunk!(0, in0, c, pair, w, sa, acc0);
                acc1 = consume_chunk!(0, in1, c, pair, w, sa, acc1);
                if c + 1 < chunks {
                    let w = ld_code_v4!(code_base.add(((c + 1) as usize) * 512), policy);
                    acc0 = consume_chunk!(1, in0, c + 1, pair, w, sa, acc0);
                    acc1 = consume_chunk!(1, in1, c + 1, pair, w, sa, acc1);
                }
                c += 2;
            }
            let factor = *scales.add(elements / 16).cast::<f32>();
            store_slab!(base_in, acc0, factor);
            if base_in + 8 < inputs {
                store_slab!(base_in + 8, acc1, factor);
            }
            base_in += stride_in;
        }
    }};
}

/// # Safety
/// SAFETY: The checked host launch supplies a live command and aligned, contiguous
/// panel, input and output extents from DraftOp::layout.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn inkling_draft_panel(table: *const crate::Command, index: u64) {
    // SAFETY: the ordinary launch has the Panel, PanelGated, Panels, or FoldPick extents checked by the caller.
    run::<false>(table, index)
}

/// # Safety
/// SAFETY: The checked HeadFold command supplies both live panels and their packed input spans.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn inkling_draft_head_fold(table: *const crate::Command, index: u64) {
    // SAFETY: only the checked HeadFold launch names this entry.
    run::<true>(table, index)
}

/// # Safety
/// COMBINED selects the checked HeadFold ABI. Otherwise the command is Panel, PanelGated, Panels, or FoldPick.
#[inline(always)]
unsafe fn run<const COMBINED: bool>(table: *const crate::Command, index: u64) {
    // SAFETY: block_dim_x is a multiple of 32 (32 for small-grid single-slab panels, else 128).
    // Each active warp owns a distinct 16-row tile. Padded rows cannot enter a pick.
    // All pointer arithmetic stays inside the extents verified by `DraftOp::layout` and `Backend::draft_lanes`.
    unsafe {
        let cmd = (*table.add(index as usize)).payload.as_ptr().cast::<u64>();
        let kind = *cmd as u32;
        let n = *cmd.add(5) as u32;
        let i = core::arch::nvptx::_block_idx_x() as u32 * core::arch::nvptx::_block_dim_x() as u32
            + core::arch::nvptx::_thread_idx_x() as u32;
        if i >= n { return; }
        if COMBINED && kind == 28 {
            let x = *cmd.add(1) as *const f32;
            let y = *cmd.add(2) as *const f32;
            let out = *cmd.add(4);
            let a = *cmd.add(6) as u32;
            let b = *cmd.add(7) as u32;
            let c0 = *cmd.add(8) as u32;
            let c = c0 & 0xffff;
            let shape = c0 >> 16;
            let slab_shift = shape & 7;
            let panel_slabs = 1u32 << slab_shift;
            let panel = i >> 5;
            let lane = i & 31;
            let group = lane >> 2;
            let pair = lane & 3;
            let p_idx = panel >> slab_shift;
            let s_idx = panel & (panel_slabs - 1);
            let pin_w = (a as usize).next_multiple_of(64) * (b as usize) <= (1 << 20);
            let row = p_idx * 16 + group;
            let row_words = (a as usize).div_ceil(2);
            let ev = l2_evict_first!();
            if c <= 8 || (shape & 8) != 0 {
                let first = s_idx * 8;
                if first < c {
                    let inputs = (c - first).min(8);
                    let h = project_panel!(x, y.add((first * (b >> 6) * 18) as usize), a, b, inputs, p_idx, lane, group, pair, pin_w);
                    let out_sub = (out as *mut u32).add((first as usize) * row_words) as u64;
                    store_bf16_tile!(no_skip, 0u32, inputs, h, row, a, row_words, pair, group, out_sub, ev);
                }
            } else {
                let elements = (a as usize).next_multiple_of(64) * (b as usize);
                let scales = x.cast::<u8>().add(elements / 2);
                let chunks = b >> 6;
                let last = chunks - 1;
                let base_stripe = ((p_idx >> 2) * chunks) * 4 + (p_idx & 3);
                let code_base = x.cast::<u32>().add((base_stripe as usize) * 128 + (lane as usize) * 4);
                let scale_base = scales.cast::<u32>().add((base_stripe as usize) * 16 + ((group * 2 + (pair & 1)) as usize));
                let policy = if pin_w || c > panel_slabs * 16 { l2_evict_last!() } else { ev };
                let hi_pair = (pair >> 1) & 1;
                let stride_in = panel_slabs * 16;
                let mut base_in = s_idx * 16;
                while base_in < c {
                    let in0 = y.cast::<u32>().add(((base_in + group).min(c - 1) * chunks * 18) as usize);
                    let in1 = y.cast::<u32>().add(((base_in + 8 + group).min(c - 1) * chunks * 18) as usize);
                    let mut acc0 = [0.0f32; 4];
                    let mut acc1 = [0.0f32; 4];
                    let mut ch = 0u32;
                    while ch < chunks {
                        let sa = ld_scale_u32!(scale_base.add((last.min(ch + hi_pair) as usize) * 64), policy);
                        let w = ld_code_v4!(code_base.add((ch as usize) * 512), policy);
                        acc0 = consume_chunk!(0, in0, ch, pair, w, sa, acc0);
                        acc1 = consume_chunk!(0, in1, ch, pair, w, sa, acc1);
                        if ch + 1 < chunks {
                            let w = ld_code_v4!(code_base.add(((ch + 1) as usize) * 512), policy);
                            acc0 = consume_chunk!(1, in0, ch + 1, pair, w, sa, acc0);
                            acc1 = consume_chunk!(1, in1, ch + 1, pair, w, sa, acc1);
                        }
                        ch += 2;
                    }
                    let factor = *scales.add(elements / 16).cast::<f32>();
                    let h0 = [acc0[0] * factor, acc0[1] * factor, acc0[2] * factor, acc0[3] * factor];
                    store_bf16_tile!(no_skip, base_in, c, h0, row, a, row_words, pair, group, out, ev);
                    if base_in + 8 < c {
                        let h1 = [acc1[0] * factor, acc1[1] * factor, acc1[2] * factor, acc1[3] * factor];
                        store_bf16_tile!(no_skip, base_in + 8, c, h1, row, a, row_words, pair, group, out, ev);
                    }
                    base_in += stride_in;
                }
            }
            return;
        }
        let x0 = *cmd.add(1) as *const f32;
        let y = *cmd.add(2) as *const f32;
        let out0 = *cmd.add(4);
        let a0 = *cmd.add(6) as u32;
        let b = *cmd.add(7) as u32;
        let c0 = *cmd.add(8) as u32;
        // SAFETY: operation 19 holds two or three panels of whole 64-row blocks back to back (rows a0, d, e)
        // and writes their outputs back to back without a stack-allocated indexable array.
        // The host (`draft_ops::PanelShape`) picks how warps share a 16-row panel and sends it above the input
        // count: bits 16..=18 the shift of the warps a panel, bit 19 warps of 8 inputs (else 16).
        let panels = !COMBINED && (kind == 13 || kind == 19);
        let (c, shape) = if panels { (c0 & 0xffff, c0 >> 16) } else { (c0, 0) };
        let slab_shift = shape & 7;
        let panel_slabs = 1u32 << slab_shift;
        let (kind, x, a, out, i) = if !COMBINED && kind == 19 {
            let d = *cmd.add(9) as u32;
            let e = *cmd.add(10) as u32;
            let mut weight = x0.cast::<u8>();
            let mut dst = out0 as *mut f32;
            let span0 = 2 * a0 * panel_slabs;
            if i < span0 {
                (13u32, x0, a0, out0, i)
            } else {
                weight = weight.add((a0 as usize) * (b as usize) / 2 + (a0 as usize) * (b as usize) / 16 + 8192);
                dst = dst.add((a0 as usize) * (c as usize));
                let i1 = i - span0;
                let span1 = 2 * d * panel_slabs;
                if i1 < span1 {
                    (13u32, weight.cast::<f32>(), d, dst as u64, i1)
                } else {
                    weight = weight.add((d as usize) * (b as usize) / 2 + (d as usize) * (b as usize) / 16 + 8192);
                    dst = dst.add((d as usize) * (c as usize));
                    (13u32, weight.cast::<f32>(), e, dst as u64, i1 - span1)
                }
            }
        } else {
            (kind, x0, a0, out0, i)
        };

        if (COMBINED && kind == 17) || (!COMBINED && (kind == 13 || kind == 14)) {
            use crate::tc_ptx::st_global_f32_hint;
            if !COMBINED && kind == 13 && (*cmd as u32) == 13 && *cmd.add(9) as u32 != 0 {
                if ld_f32_nc!(*cmd.add(3) as *const f32) != 0.0 {
                    if i == 0 { *(out as *mut f32).add((a as usize) * (c as usize)) = 1.0; }
                    return;
                }
            }
            if !COMBINED && kind == 14 && (c & 2) != 0 && ld_f32_nc!(y.add(((b >> 6) * 18) as usize)) != 0.0 {
                return;
            }
            let panel = i >> 5;
            let lane = i & 31;
            let group = lane >> 2;
            let pair = lane & 3;
            if !COMBINED && kind == 13 && c > 8 {
                let p_idx = panel >> slab_shift;
                let s_idx = panel & (panel_slabs - 1);
                let pin_w = (a as usize).next_multiple_of(64) * (b as usize) <= (1 << 20);
                if shape & 8 == 0 {
                    project_panel_wide!(x, y, a, b, c, p_idx, s_idx, panel_slabs, lane, group, pair, pin_w, out);
                } else if s_idx * 8 < c {
                    // Warps of 8 inputs (the host gives a panel a warp for each 8 inputs): each warp runs the 6-chunk
                    // pipelined 1..=8-input product on its own 8 inputs. Each output is the same chunk-ordered MMA
                    // chain as in the two-tile product, so every bit matches.
                    let first = s_idx * 8;
                    let inputs = (c - first).min(8);
                    let h = project_panel!(x, y.add((first * (b >> 6) * 18) as usize), a, b, inputs, p_idx, lane, group, pair, pin_w);
                    let row = p_idx * 16 + group;
                    let t0 = pair * 2;
                    if row < a {
                        if t0 < inputs {
                            let dst = (out as *mut f32).add(((first + t0) as usize) * (a as usize) + (row as usize));
                            *dst = rn(h[0]);
                            if row + 8 < a { *dst.add(8) = rn(h[2]); }
                        }
                        if t0 + 1 < inputs {
                            let dst = (out as *mut f32).add(((first + t0 + 1) as usize) * (a as usize) + (row as usize));
                            *dst = rn(h[1]);
                            if row + 8 < a { *dst.add(8) = rn(h[3]); }
                        }
                    }
                }
                if i == 0 && (*cmd as u32) == 13 && *cmd.add(9) as u32 != 0 {
                    *(out as *mut f32).add((a as usize) * (c as usize)) = 0.0;
                }
                return;
            }
            let batch = if !COMBINED && kind == 14 { ((c >> 2) & 63) + 1 } else { 1 };
            let slabs = if !COMBINED && kind == 14 { batch.div_ceil(8) } else { 1 };
            let (panel, base_b) = if !COMBINED && slabs > 1 { (panel / slabs, (panel % slabs) * 8) } else { (panel, 0) };
            let inputs = if COMBINED { (c & 63) >> 1 } else if kind == 13 { c } else { (batch - base_b).min(8) };
            let head_input = if COMBINED {
                *cmd.add(3) as *const f32
            } else if !COMBINED && base_b > 0 {
                y.add((base_b * (b >> 6) * 18) as usize)
            } else {
                y
            };
            let pin_first = if COMBINED {
                false
            } else if kind == 14 {
                slabs > 1 || panel >= a.div_ceil(16).saturating_sub(PIN_FOLD_PANELS)
            } else {
                kind == 13 && (a as usize).next_multiple_of(64) * (b as usize) <= (1 << 20)
            };
            let row = panel * 16 + group;
            let bf16_base = if COMBINED { (c >> 31) != 0 } else { kind == 14 && (c & 256) != 0 };
            // Prefetch `base_ptr` and `bias` DRAM loads for FoldPick and BF16 FoldPickBatch before the weight MMAs so
            // their ~300-cycle DRAM miss latency overlaps 100% with the 4 NVFP4 chunks + MMAs.
            let (pref_base0, pref_base1, pref_bias_lo, pref_bias_hi) = if !COMBINED && kind == 14 && ((batch == 1 && pair == 0) || (batch > 1 && bf16_base)) {
                let elements = (a as usize).next_multiple_of(64) * (b as usize);
                let bias = x.cast::<u8>().add(elements / 2 + elements / 16 + 8192).cast::<f32>();
                let base_ptr = *cmd.add(3) as *const f32;
                let evict_first = l2_evict_first!();
                let b_pol = if pin_first { l2_evict_last!() } else { evict_first };
                let slo = if row < a { ld_f32_hint!(bias.add(row as usize), b_pol) } else { 0.0 };
                let shi = if row + 8 < a { ld_f32_hint!(bias.add((row + 8) as usize), b_pol) } else { 0.0 };
                if batch == 1 {
                    let (blo, bhi) = if bf16_base {
                        let bu16 = base_ptr as *const u16;
                        let lo = if row < a { ld_u16_hint!(bu16.add(row as usize), evict_first) << 16 } else { 0 };
                        let hi = if row + 8 < a { ld_u16_hint!(bu16.add((row + 8) as usize), evict_first) << 16 } else { 0 };
                        (lo, hi)
                    } else {
                        let lo = if row < a { ld_f32_hint!(base_ptr.add(row as usize), evict_first).to_bits() } else { 0 };
                        let hi = if row + 8 < a { ld_f32_hint!(base_ptr.add((row + 8) as usize), evict_first).to_bits() } else { 0 };
                        (lo, hi)
                    };
                    (blo, bhi, slo, shi)
                } else {
                    let bu16 = base_ptr as *const u16;
                    let stride_u16 = (a as usize).next_multiple_of(2);
                    let s0 = base_b + pair * 2;
                    let s1 = s0 + 1;
                    let p0 = if s0 < batch {
                        let off = (s0 as usize) * stride_u16;
                        let lo = if row < a { ld_u16_hint!(bu16.add(off + row as usize), evict_first) } else { 0 };
                        let hi = if row + 8 < a { ld_u16_hint!(bu16.add(off + (row + 8) as usize), evict_first) } else { 0 };
                        lo | (hi << 16)
                    } else { 0 };
                    let p1 = if s1 < batch {
                        let off = (s1 as usize) * stride_u16;
                        let lo = if row < a { ld_u16_hint!(bu16.add(off + row as usize), evict_first) } else { 0 };
                        let hi = if row + 8 < a { ld_u16_hint!(bu16.add(off + (row + 8) as usize), evict_first) } else { 0 };
                        lo | (hi << 16)
                    } else { 0 };
                    (p0, p1, slo, shi)
                }
            } else {
                (0u32, 0u32, 0.0f32, 0.0f32)
            };
            let (head, head1) = if COMBINED && inputs > 8 {
                project_panel_two!(x, head_input, a, b, inputs, panel, lane, group, pair, pin_first)
            } else {
                (project_panel!(x, head_input, a, b, inputs, panel, lane, group, pair, pin_first), [0.0f32; 4])
            };

            if COMBINED || kind == 13 {
                let skip = u32::from(COMBINED);
                let evict_first = l2_evict_first!();
                if COMBINED && bf16_base {
                    let row_words = (a as usize).div_ceil(2);
                    if inputs > 8 {
                        store_bf16_tile!(8u32, skip, inputs, head1, row, a, row_words, pair, group, out, evict_first);
                    }
                    store_bf16_tile!(0u32, skip, inputs, head, row, a, row_words, pair, group, out, evict_first);
                } else {
                    macro_rules! store_tile {
                        ($t_base:expr, $h:expr) => {{
                            let t0 = $t_base + pair * 2;
                            if t0 >= skip && t0 < inputs && row < a {
                                let dst = (out as *mut f32).add(((t0 - skip) as usize) * (a as usize) + (row as usize));
                                if COMBINED {
                                    st_global_f32_hint(dst, rn($h[0]), evict_first);
                                    if row + 8 < a { st_global_f32_hint(dst.add(8), rn($h[2]), evict_first); }
                                } else {
                                    *dst = rn($h[0]);
                                    if row + 8 < a { *dst.add(8) = rn($h[2]); }
                                }
                            }
                            let t1 = t0 + 1;
                            if t1 >= skip && t1 < inputs && row < a {
                                let dst = (out as *mut f32).add(((t1 - skip) as usize) * (a as usize) + (row as usize));
                                if COMBINED {
                                    st_global_f32_hint(dst, rn($h[1]), evict_first);
                                    if row + 8 < a { st_global_f32_hint(dst.add(8), rn($h[3]), evict_first); }
                                } else {
                                    *dst = rn($h[1]);
                                    if row + 8 < a { *dst.add(8) = rn($h[3]); }
                                }
                            }
                        }};
                    }
                    if COMBINED && inputs > 8 {
                        store_tile!(8u32, head1);
                    }
                    store_tile!(0u32, head);
                    if !COMBINED && i == 0 && (*cmd as u32) == 13 && *cmd.add(9) as u32 != 0 {
                        *(out as *mut f32).add((a as usize) * (c as usize)) = 0.0;
                    }
                }
            }

            if COMBINED || kind == 14 {
                let total_panels = a.div_ceil(16);
                let pin_from = total_panels.saturating_sub(PIN_FOLD_PANELS);
                let (fold, columns, acc, comb_b_lo, comb_b_hi) = if COMBINED {
                    let width = (c & 0x7fff_ffff) >> 6;
                    let latent = head_input.add((inputs * (b >> 6) * 18) as usize);
                    let pin_fold = panel >= pin_from;
                    let (blo, bhi) = if pair == 0 {
                        let f_elems = (a as usize).next_multiple_of(64) * (width as usize);
                        let f_bias = y.cast::<u8>().add(f_elems / 2 + f_elems / 16 + 8192).cast::<f32>();
                        let b_pol = if pin_fold { l2_evict_last!() } else { l2_evict_first!() };
                        let lo = if row < a { ld_f32_hint!(f_bias.add(row as usize), b_pol) } else { 0.0 };
                        let hi = if row + 8 < a { ld_f32_hint!(f_bias.add((row + 8) as usize), b_pol) } else { 0.0 };
                        (lo, hi)
                    } else {
                        (0.0f32, 0.0f32)
                    };
                    (y, width, project_panel!(y, latent, a, width, 1, panel, lane, group, pair, pin_fold), blo, bhi)
                } else {
                    (x, b, head, pref_bias_lo, pref_bias_hi)
                };
                let elements = (a as usize).next_multiple_of(64) * (columns as usize);
                let bias = fold.cast::<u8>().add(elements / 2 + elements / 16 + 8192).cast::<f32>();
                let base_ptr = *cmd.add(3) as *const f32;
                let eps = f32::from_bits(*cmd.add(9) as u32);
                let bound = f32::from_bits(*cmd.add(10) as u32);
                let evict_first = l2_evict_first!();
                let prefetched_bias = COMBINED || batch == 1 || bf16_base;
                let b_lo = if prefetched_bias { comb_b_lo } else if row < a { *bias.add(row as usize) } else { 0.0 };
                let b_hi = if prefetched_bias { comb_b_hi } else if row + 8 < a { *bias.add((row + 8) as usize) } else { 0.0 };
                let subs = if !COMBINED && batch > 1 { 2u32 } else { 1u32 };
                let mut sub = 0u32;
                while sub < subs {
                    let slot = base_b + pair * 2 + sub;
                    let active = if !COMBINED && batch > 1 { slot < batch } else { pair == 0 };
                    let (v_lo, v_hi) = if sub == 0 { (acc[0], acc[2]) } else { (acc[1], acc[3]) };
                    let base_b = if !COMBINED && batch > 1 && !bf16_base { base_ptr.add((slot as usize) * (a as usize)) } else { base_ptr };
                    let p_bf16 = if sub == 0 { pref_base0 } else { pref_base1 };
                    let lo = if active && row < a {
                        let base = if COMBINED {
                            rn(head[0])
                        } else if batch == 1 {
                            f32::from_bits(pref_base0)
                        } else if bf16_base {
                            f32::from_bits(p_bf16 << 16)
                        } else {
                            ld_f32_hint!(base_b.add(row as usize), evict_first)
                        };
                        let biased = rn(v_lo + b_lo);
                        let sum = rn(biased + base);
                        if biased.is_finite() && sum.is_finite() { tanh(sum * eps) * bound } else { f32::NAN }
                    } else {
                        f32::NEG_INFINITY
                    };
                    let hi = if active && row + 8 < a {
                        let base = if COMBINED {
                            rn(head[2])
                        } else if batch == 1 {
                            f32::from_bits(pref_base1)
                        } else if bf16_base {
                            f32::from_bits(p_bf16 & 0xffff_0000)
                        } else {
                            ld_f32_hint!(base_b.add((row + 8) as usize), evict_first)
                        };
                        let biased = rn(v_hi + b_hi);
                        let sum = rn(biased + base);
                        if biased.is_finite() && sum.is_finite() { tanh(sum * eps) * bound } else { f32::NAN }
                    } else {
                        f32::NEG_INFINITY
                    };
                    let mut best = lo.max(hi);
                    let mut id = if hi > lo { row + 8 } else { row };
                    let mut valid = u32::from(!active || ((row >= a || lo.is_finite()) && (row + 8 >= a || hi.is_finite())));
                    macro_rules! fold_max {
                        ($dist:expr) => {{
                            let v = shfl_xor_f32(best, $dist);
                            let other = shfl_xor_u32(id, $dist);
                            valid &= shfl_xor_u32(valid, $dist);
                            if v > best || (v == best && other < id) { best = v; id = other; }
                        }};
                    }
                    fold_max!(16);
                    fold_max!(8);
                    fold_max!(4);
                    let mut sum = 0.0f32;
                    if (c & 1) != 0 {
                        if active && row < a {
                            sum = exp(lo - best);
                            if row + 8 < a { sum += exp(hi - best); }
                        }
                        sum += shfl_xor_f32(sum, 16);
                        sum += shfl_xor_f32(sum, 8);
                        sum += shfl_xor_f32(sum, 4);
                    }
                    if group == 0 && active {
                        let offset = if COMBINED {
                            let base_row = if bf16_base { (a as usize).div_ceil(2) } else { a as usize };
                            ((inputs - 1) as usize) * base_row + (panel as usize) * 4
                        } else if batch > 1 {
                            ((slot as usize) * (total_panels as usize) + (panel as usize)) * 4
                        } else {
                            (panel as usize) * 4
                        };
                        let dst = (out as *mut f32).add(offset);
                        *dst = best;
                        *dst.add(1) = id as f32;
                        *dst.add(2) = valid as f32;
                        *dst.add(3) = sum;
                    }
                    sub += 1;
                }
            }
        }
    }
}
