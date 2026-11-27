use super::*;
use crate::transport::Ring;
use spark_gpu_probe::prompt_ops::reference;

mod loopback {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test-support/loopback.rs"));
}

fn operands(rank: usize, part: usize, words: usize, format: Format) -> Vec<f32> {
    (0..words).map(|word| {
        let high = (0x3e00 + (rank * 113 + part * 53 + word) % 251) as u32;
        let low = if format == Format::Fp32 { (word as u32 & 0xffff) | 1 } else { 0 };
        f32::from_bits(match word { 0 => 0, 1 => 0x8000_0000, _ => (high << 16) | low })
    }).collect()
}

fn pack(words: &[f32]) -> Result<Vec<f32>> {
    if words.iter().any(|word| word.to_bits() & 0xffff != 0) {
        return Err("non-BF16 wire word".into());
    }
    reference(pack_shape(words.len())?, [words, &[], &[], &[], &[]], &[])
}

fn gather(ring: &mut Ring, own: Vec<f32>) -> Result<Vec<f32>> {
    let stride = own.len();
    let world = ring.world();
    let mut wire = vec![0.0; world * stride];
    wire[ring.rank() * stride..(ring.rank() + 1) * stride].copy_from_slice(&own);
    let result = ring.gather_rank_blocks(&mut wire, world, 1, stride);
    finish(Some(ring), result)?;
    Ok(wire)
}

fn with_group<T: Send>(world: usize, path: bool, run: impl Fn(&mut Ring) -> T + Sync) -> Vec<T> {
    loopback::run_ranks(world, 1, |rank, ports| {
        Ring::connect_options(rank, world, ports[rank], ports[(rank + 1) % world], path)
    }, |_, mut ring| run(&mut ring))
}

fn scatter(wire: &[f32], layout: &Layout) -> Vec<f32> {
    let mut grouped = vec![0.0; layout.words];
    for &[from, to, words] in &layout.copies {
        grouped[to..to + words].copy_from_slice(&wire[from..from + words]);
    }
    grouped
}

fn old_shape(tokens: usize, width: usize, world: usize, transpose: bool) -> Shape {
    if transpose {
        Shape { rows: (width * world) as u32, cols: tokens as u32, ..Shape::new(Op::Transpose, 1).unwrap() }
    } else {
        Shape::rank_rows(tokens, width, world, false).unwrap()
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn folded_and_mixed_packets_keep_every_fp32_coordinate_on_two_and_four_ranks() {
    let fold = [(8, Format::Bf16)];
    let mixed = [(6, Format::Fp32), (12, Format::Bf16), (10, Format::Bf16)];
    let tokens = 7;
    for (world, path) in [(2, false), (4, true)] {
        for parts in [&fold[..], &mixed[..]] {
            let transpose = parts.len() > 1;
            let plain = Layout::new(world, &parts.iter().map(|&(width, _)| (tokens * width, Format::Fp32)).collect::<Vec<_>>()).unwrap();
            let packed = Layout::new(world, &parts.iter().map(|&(width, format)| (tokens * width, format)).collect::<Vec<_>>()).unwrap();
            let checks = with_group(world, path, |ring| -> Result<usize> {
                let mut raw = Vec::new();
                let mut mixed = Vec::new();
                for (part, &(width, format)) in parts.iter().enumerate() {
                    let input = operands(ring.rank(), part, tokens * width, format);
                    raw.extend_from_slice(&input);
                    if format == Format::Bf16 { mixed.extend(pack(&input)?); }
                    else { mixed.extend(input); }
                }
                let raw = gather(ring, raw)?;
                let wire = gather(ring, mixed)?;
                let actual = scatter(&wire, &packed);
                let original = scatter(&raw, &plain);
                let mut wrong = packed.copies.clone();
                let first_bf16 = usize::from(transpose) * world;
                let (left, right) = wrong.split_at_mut(first_bf16 + 1);
                std::mem::swap(&mut left[first_bf16][0], &mut right[0][0]);
                let wrong = scatter(&wire, &Layout { copies: wrong, sections: Vec::new(), offsets: Vec::new(), widths: Vec::new(), stride: 0, words: packed.words });
                let mut compared = 0;
                for (part, &(width, format)) in parts.iter().enumerate() {
                    let prior = old_shape(tokens, width, world, transpose);
                    let candidate = if format == Format::Bf16 { unpack_shape(tokens, width, world, transpose, transpose)? } else { prior };
                    let expected = reference(prior, [&original[plain.sections[part].clone()], &[], &[], &[], &[]], &[])?;
                    let section = packed.sections[part].clone();
                    let got = reference(candidate, [&actual[section.clone()], &[], &[], &[], &[]], &[])?;
                    assert_eq!(bits(&got), bits(&expected));
                    assert!(bits(&got).contains(&0) && bits(&got).contains(&0x8000_0000));
                    if format == Format::Bf16 {
                        let direct_shape = unpack_shape_strided(tokens, width, world, transpose, transpose, packed.stride)?;
                        let direct_got = reference(direct_shape, [&wire[packed.direct_range(world, part)], &[], &[], &[], &[]], &[])?;
                        assert_eq!(bits(&direct_got), bits(&expected));
                    } else if !transpose {
                        let direct_shape = Shape::rank_rows_strided(tokens, width, world, false, packed.stride).unwrap();
                        let direct_got = reference(direct_shape, [&wire[packed.direct_range(world, part)], &[], &[], &[], &[]], &[])?;
                        assert_eq!(bits(&direct_got), bits(&expected));
                    }
                    if part == usize::from(transpose) {
                        let wrong = reference(candidate, [&wrong[section], &[], &[], &[], &[]], &[])?;
                        assert_ne!(bits(&got), bits(&wrong));
                    }
                    compared += got.len();
                }
                Ok(compared)
            });
            for check in checks {
                assert_eq!(check.unwrap(), tokens * world * parts.iter().map(|part| part.0).sum::<usize>());
            }
        }
    }
}

#[test]
fn a_non_bf16_word_fails_the_gather_on_every_rank() {
    for (world, path) in [(2, false), (4, true)] {
        let mut poisoned_column = vec![1.0; world];
        poisoned_column[1] = f32::from_bits(0x3f80_0001);
        for mixed in [false, true] {
            let checks = with_group(world, path, |ring| {
                let mut data = [1.0; 8];
                data[7] = poisoned_column[ring.rank()];
                let mut packet = Vec::new();
                if mixed {
                    packet = operands(ring.rank(), 0, 8, Format::Fp32);
                    packet.extend(pack(&[1.0; 8]).unwrap());
                }
                match pack(&data) {
                    Ok(words) => {packet.extend(words);gather(ring, packet).map(|_| ())},
                    Err(error) => finish(Some(ring), Err(error)),
                }
            });
            assert_eq!(checks[1].as_ref().unwrap_err(), "non-BF16 wire word");
            assert!(checks.iter().all(Result::is_err), "world={world} mixed={mixed} {checks:?}");
        }
    }
}
