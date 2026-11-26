use super::*;

fn separate(v: &[u32], rows: usize, width: usize, columns: usize, offset: usize) -> Vec<u32> {
    (0..rows).flat_map(|row| v[row * width + offset..row * width + offset + columns].iter().copied()).collect()
}

#[test]
fn packed_parts_match_separate_exchanges_on_each_rank() {
    for world in [2, 4] {
        for rows in [1, 3, 16, 32] {
            // Unequal, aligned, and scalar-only parts. Include raw NaN and signed-zero words.
            for widths in [[20, 12], [24, 8], [4, 28], [10240, 6144]] {
                let inputs: [Vec<u32>; 2] = std::array::from_fn(|part| (0..rows * widths[part]).map(|i| {
                    [0x8000_0000, 0x7fc0_0123, 0xffff_ffff, 0][i % 4] ^ (i as u32).rotate_left(11) ^ (part as u32).rotate_left(23)
                }).collect());
                let columns = widths.map(|w| w / world);
                let mut messages = Vec::new();
                for rank in 0..world {
                    let p = Pair::checked(rows, std::array::from_fn(|i| (inputs[i].len(), widths[i], columns[i], rank * columns[i])), usize::MAX).unwrap();
                    let packed: Vec<u32> = (0..p.words()).map(|word| { let (i, at) = p.locate(word); inputs[i][at] }).collect();
                    let separate: Vec<u32> = (0..2).flat_map(|i| separate(&inputs[i], rows, widths[i], columns[i], rank * columns[i])).collect();
                    assert_eq!(packed, separate);
                    if columns.iter().all(|c| c % 4 == 0) {
                        let quads: Vec<u32> = (0..p.words()).step_by(4).flat_map(|word| {
                            let (i, at) = p.locate(word); inputs[i][at..at + 4].iter().copied()
                        }).collect();
                        assert_eq!(quads, packed);
                    }
                    messages.push(packed);
                }
                for rank in 0..world {
                    let mut packed: [Vec<u32>; 2] = std::array::from_fn(|i| vec![0xdead_beef; inputs[i].len()]);
                    for i in 0..2 {
                        for row in 0..rows {
                            let start = row * widths[i] + rank * columns[i];
                            packed[i][start..start + columns[i]].copy_from_slice(&inputs[i][start..start + columns[i]]);
                        }
                    }
                    let mut separate = packed.clone();
                    for origin in (0..world).rev().filter(|&o| o != rank) {
                        let p = Pair::checked(rows, std::array::from_fn(|i| (inputs[i].len(), widths[i], columns[i], origin * columns[i])), usize::MAX).unwrap();
                        for (word, &bits) in messages[origin].iter().enumerate() { let (i, at) = p.locate(word); packed[i][at] = bits; }
                        let mut at = 0;
                        for i in 0..2 {
                            for row in 0..rows {
                                let start = row * widths[i] + origin * columns[i];
                                separate[i][start..start + columns[i]].copy_from_slice(&messages[origin][at..at + columns[i]]);
                                at += columns[i];
                            }
                        }
                    }
                    assert_eq!(packed, separate);
                    assert_eq!(packed, inputs);
                }
            }
        }
    }
}

#[test]
fn pair_checks_both_shapes_and_combined_capacity() {
    let parts = [(60, 20, 5, 15), (36, 12, 3, 9)];
    assert_eq!(Pair::checked(3, parts, 24).unwrap().words(), 24);
    assert!(Pair::checked(3, parts, 23).is_none());
    assert!(Pair::checked(0, parts, 24).is_none());
    for i in 0..2 {
        let mut bad = parts; bad[i].0 -= 1;
        assert!(Pair::checked(3, bad, 24).is_none());
        let mut bad = parts; bad[i].3 += 1;
        assert!(Pair::checked(3, bad, 24).is_none());
        let mut bad = parts; bad[i].2 = 0;
        assert!(Pair::checked(3, bad, 24).is_none());
        let mut bad = parts; bad[i].3 = usize::MAX;
        assert!(Pair::checked(3, bad, 24).is_none());
    }
    assert!(Pair::checked(usize::MAX, [(usize::MAX, 2, 1, 0); 2], usize::MAX).is_none());
    assert!(Pair::checked(1, [(usize::MAX, usize::MAX, usize::MAX, 0); 2], usize::MAX).is_none());
    assert!(Pair::checked(1, [(u32::MAX as usize, u32::MAX as usize, u32::MAX as usize, 0); 2], usize::MAX).is_none());
}
