//! HC K320 has five ascending K16 tiles in each canonical range.
#[inline(always)]
pub(super) fn five<T>(load: impl Fn(usize) -> T, mut apply: impl FnMut(T), start: usize) {
    let v0 = load(start);
    let v1 = load(start + 1);
    let v2 = load(start + 2);
    let v3 = load(start + 3);
    let v4 = load(start + 4);
    apply(v0);
    apply(v1);
    apply(v2);
    apply(v3);
    apply(v4);
}

#[cfg(test)]
mod tests {
    use super::five;
    use crate::{kernel_params::flash_bf16_element, next_math};
    #[test]
    fn hc_up_full_rows_and_operand_ownership() {
        for m in [1usize, 2, 8, 16, 17, 65] {
            let mut owners = vec![0u8; m * 2560];
            for block in 0..320 {
                let feature = block * 8;
                assert_eq!(feature, block / 4 * 32 + block % 4 * 8);
                for mt in (0..m).step_by(16) {
                    for lane in 0..32 {
                        for stream in 0..4 {
                            let load = |kt| {
                                let a = ((mt / 16 * 20 + kt) * 32 + lane) * 4;
                                let b = flash_bf16_element(
                                    stream * 2560 + feature + lane / 4,
                                    kt * 16 + lane % 4 * 2,
                                    320,
                                    10240,
                                );
                                (
                                    core::array::from_fn::<_, 8, _>(|i| {
                                        a + i % 4 + i / 4 * m.div_ceil(16) * 20 * 128
                                    }),
                                    b,
                                )
                            };
                            for part in 0..4 {
                                let off: Vec<_> = (part * 5..part * 5 + 5).map(load).collect();
                                let mut on = Vec::new();
                                five(load, |v| on.push(v), part * 5);
                                assert_eq!(on, off); // Both A planes use these same offsets.
                            }
                        }
                        for dt in [0, 8] {
                            let t = mt + lane / 4 + dt;
                            for df in 0..2 {
                                if t < m {
                                    owners[t * 2560 + feature + lane % 4 * 2 + df] += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert!(owners.iter().all(|&v| v == 1));
            // A scalar host model checks the entire HC output byte vector.
            // Hardware MMA rounding is not emulated; identical operand traces
            // above and the unchanged MMA/add graph establish its equivalence.
            let run = |on| {
                (0..m * 2560)
                    .map(|out| {
                        let mut mixed = 0.0;
                        for stream in 0..4 {
                            let mut parts = [0.0; 4];
                            for (part, z) in parts.iter_mut().enumerate() {
                                let (mut hi, mut lo) = (0.0f32, 0.0f32);
                                let load = |kt| {
                                    core::array::from_fn::<_, 16, _>(|j| {
                                        let n = out.wrapping_mul(73) ^ (stream * 320 + kt * 16 + j);
                                        ((n % 127) as f32 - 63.0) / 128.0
                                    })
                                };
                                let mut apply = |v: [f32; 16]| {
                                    for x in v {
                                        hi = x.mul_add(0.5, hi);
                                        lo = x.mul_add(0.0009765625, lo);
                                    }
                                };
                                if on {
                                    five(load, &mut apply, part * 5);
                                } else {
                                    for kt in part * 5..part * 5 + 5 {
                                        apply(load(kt));
                                    }
                                }
                                *z = hi + lo;
                            }
                            let gate = (parts[0] + parts[1]) + (parts[2] + parts[3]);
                            mixed = next_math::hc_stream(mixed, stream as f32 - 1.5, gate, |v| {
                                1.0 / (1.0 + next_math::exp(-v))
                            });
                        }
                        (mixed * 0.25).to_bits()
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(run(true), run(false), "M={m}, complete H2560 rows");
        }
    }
}
