use engine_model::csf::{Csf32, Quantization};

#[test]
fn legacy_csf_and_csf32_share_the_nvfp4_arithmetic_after_expansion() {
    use engine_model::flash::nvfp4;
    // Each row stores codes 0 and 1. The final position is an exception.
    let bases: Vec<u8> = (0..32).collect();
    let codes = [0x10; 32];
    let mut legacy = Vec::new();
    for slab in bases.as_chunks::<16>().0 { legacy.extend_from_slice(slab); legacy.extend_from_slice(&[0x10;16]); }
    let offsets = words(&[0,1]); let exceptions = words(&[0x7e3f]);
    let plane = Csf32 { rows: 32,columns: 2,quantization: Quantization::Nvfp4,bases: &bases,codes: &codes,
        page_bases: &[0;4],offsets: &offsets,exceptions: &exceptions };
    let scales = plane.expand().unwrap();
    assert_eq!(scales,nvfp4::csf_expand(&legacy,&[0x7e00_003f],32,32).unwrap());
    assert_eq!(&scales[..4],&[0,1,1,2]); assert_eq!(scales[63],0x7e);
    let packed = [0x21;512]; let global = f32::from_bits(0x3f800001);
    let (mut direct,mut decoded) = ([0.;1024],[0.;1024]);
    nvfp4::dequantize(&packed,&scales,global,&mut direct).unwrap();
    plane.dequantize_nvfp4(&packed,global,&mut decoded).unwrap();
    assert_eq!(direct.map(f32::to_bits),decoded.map(f32::to_bits));
}

#[derive(Clone)]
struct Fixture {
    rows: usize,
    columns: usize,
    quantization: Quantization,
    bases: Vec<u8>,
    codes: Vec<u8>,
    page_bases: Vec<u8>,
    offsets: Vec<u8>,
    exceptions: Vec<u8>,
}
impl Fixture {
    fn view(&self) -> Csf32<'_> {
        Csf32 { rows: self.rows, columns: self.columns, quantization: self.quantization,
            bases: &self.bases, codes: &self.codes, page_bases: &self.page_bases,
            offsets: &self.offsets, exceptions: &self.exceptions }
    }
    fn component(&mut self, index: usize) -> &mut Vec<u8> {
        match index { 0 => &mut self.bases, 1 => &mut self.codes, 2 => &mut self.page_bases,
            3 => &mut self.offsets, 4 => &mut self.exceptions, _ => unreachable!() }
    }
}
fn words(values: &[u16]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }

// Test-side encoder from _from_logical_components in the public FlashInfer
// flashinfer/experimental/b12x/loader/_csf_blocks.py. Sort logical block IDs
// by their source-defined tile coordinates before packing each field.
fn encode(rows: usize, columns: usize, quantization: Quantization, bases: Vec<u8>, values: &[u8]) -> Fixture {
    let bits = match quantization { Quantization::Nvfp4 => 4, Quantization::Mxfp4 => 1 };
    let mask = (1u8 << bits) - 1;
    let block_columns = columns.div_ceil(2);
    let mut order: Vec<_> = (0..rows / 32 * block_columns).collect();
    order.sort_by_key(|&block| {
        let (r, c) = (block / block_columns, block % block_columns);
        (r / 4, c / 2, c % 2, r % 4)
    });
    let mut result = Fixture { rows, columns, quantization, bases, codes: vec![],
        page_bases: vec![], offsets: vec![], exceptions: vec![] };
    let mut page = None;
    let mut page_start = 0;
    for block in order {
        let (br, bc) = (block / block_columns, block % block_columns);
        let group = br / 4 * columns.div_ceil(4) + bc / 2;
        if page != Some(group / 32) {
            page = Some(group / 32);
            page_start = result.exceptions.len();
            result.page_bases.extend_from_slice(&(page_start as u32).to_le_bytes());
            result.offsets.extend_from_slice(&0u16.to_le_bytes());
        }
        let width = (columns - bc * 2).min(2);
        let mut fields = Vec::new();
        for r in 0..32 { for c in 0..width {
            let row = br * 32 + r;
            let value = values[row * columns + bc * 2 + c];
            let delta = value.checked_sub(result.bases[row]).filter(|&v| v <= mask);
            fields.push(delta.unwrap_or(0));
            if delta.is_none() {
                let record = (u16::from(value) << 8) | (r * 2 + c) as u16;
                result.exceptions.extend_from_slice(&record.to_le_bytes());
            }
        } }
        for chunk in fields.chunks_exact(8 / bits) {
            result.codes.push(chunk.iter().enumerate().fold(0, |byte, (i, &field)| byte | field << (i * bits)));
        }
        let count = (result.exceptions.len() - page_start) / 2;
        result.offsets.extend_from_slice(&(count as u16).to_le_bytes());
    }
    result
}

// Hand expansion of matrix(32, 32, group_size=16, seed=0) in public vLLM
// tests/quantization/csf_fixtures.py. No Python code runs in this test.
fn public_vector() -> (Fixture, Vec<u8>) {
    let fixture = Fixture { rows: 32, columns: 2, quantization: Quantization::Nvfp4,
        bases: (20..52).collect(), codes: [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe].repeat(4),
        page_bases: vec![0; 4], offsets: words(&[0, 5]),
        exceptions: words(&[0x0000, 0x0701, 0x0e1f, 0x1520, 0x1c3f]) };
    let mut expected: Vec<_> = (0..64).map(|i| (20 + i / 2 + i % 16) as u8).collect();
    for (position, value) in [(0, 0), (1, 7), (31, 14), (32, 21), (63, 28)] { expected[position] = value; }
    (fixture, expected)
}

#[test]
fn public_nvfp4_vector_preserves_exact_scale_bytes() {
    let (fixture, expected) = public_vector();
    fixture.view().validate().unwrap();
    assert_eq!(fixture.view().expand().unwrap(), expected);
    let mut output = vec![0xd6; expected.len()];
    fixture.view().decode_into(&mut output).unwrap();
    assert_eq!(output, expected);
}

#[test]
fn public_mxfp4_vector_preserves_exact_scale_bytes() {
    // Same public generator: matrix(32, 256, group_size=32, seed=0).
    let fixture = Fixture { rows: 32, columns: 8, quantization: Quantization::Mxfp4,
        bases: (20..52).collect(), codes: vec![0xaa; 32], page_bases: vec![0; 4],
        offsets: words(&[0, 2, 2, 2, 5]),
        exceptions: words(&[0x0000, 0x1520, 0x0701, 0x0e1f, 0x1c3f]) };
    let mut expected: Vec<_> = (0..256).map(|i| (20 + i / 8 + i % 2) as u8).collect();
    for (position, value) in [(0, 0), (7, 7), (127, 14), (128, 21), (255, 28)] { expected[position] = value; }
    assert_eq!(fixture.view().expand().unwrap(), expected);
}

#[test]
fn both_quantizations_round_trip_all_byte_patterns_and_clipped_tiles() {
    for quantization in [Quantization::Nvfp4, Quantization::Mxfp4] {
        for (rows, columns) in [(32, 1), (32, 9), (64, 3), (128, 4), (160, 5)] {
            let bases: Vec<_> = (0..rows).map(|r| (r % 200) as u8).collect();
            let values: Vec<_> = (0..rows * columns).map(|i| ((i * 73 + 19) % 256) as u8).collect();
            let encoded = encode(rows, columns, quantization, bases, &values);
            assert_eq!(encoded.view().expand().unwrap(), values);
        }
    }
}

#[test]
fn selected_rows_equal_full_expansion_across_tiles_and_exceptions() {
    for quantization in [Quantization::Nvfp4,Quantization::Mxfp4] {
        for (rows,columns) in [(32,1),(64,3),(160,5),(256,129)] {
            let bases=(0..rows).map(|row|(row%200) as u8).collect();
            let values=(0..rows*columns).map(|index|((index*73+19)%256) as u8).collect::<Vec<_>>();
            let fixture=encode(rows,columns,quantization,bases,&values);
            for selected in [0..rows,0..1,rows-1..rows,1..rows-1,rows/2..rows] {
                assert_eq!(fixture.view().expand_rows(selected.clone()).unwrap(),values[selected.start*columns..selected.end*columns]);
            }
            for selected in [0..0,rows..rows+1,rows..rows-1] {assert!(fixture.view().expand_rows(selected).is_err());}
            let mut corrupt=fixture.clone();corrupt.offsets.pop();
            assert!(corrupt.view().expand_rows(0..1).is_err());
        }
    }
}

#[test]
fn clipped_tile_order_has_an_independent_hand_vector() {
    // block_order(160, 5) from the public source. Each logical block has a
    // distinct code. The last column stores one field per row, with no pad.
    let order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14];
    let mut fixture = Fixture { rows: 160, columns: 5, quantization: Quantization::Nvfp4,
        bases: vec![0; 160], codes: vec![], page_bases: vec![0; 4],
        offsets: vec![0; 32], exceptions: vec![] };
    for block in order {
        let width = if block % 3 == 2 { 1 } else { 2 };
        fixture.codes.extend(std::iter::repeat_n(block | block << 4, 16 * width));
    }
    let got = fixture.view().expand().unwrap();
    for row in 0..160 { for col in 0..5 {
        assert_eq!(got[row * 5 + col], (row / 32 * 3 + col / 2) as u8);
    } }
}

#[test]
fn mxfp4_fields_are_least_significant_bit_first() {
    let fixture = Fixture { rows: 32, columns: 1, quantization: Quantization::Mxfp4,
        bases: vec![254; 32], codes: vec![0x01, 0x80, 0x55, 0xaa],
        page_bases: vec![0; 4], offsets: vec![0; 4], exceptions: vec![] };
    let got = fixture.view().expand().unwrap();
    assert_eq!(&got[..16], &[255, 254, 254, 254, 254, 254, 254, 254,
        254, 254, 254, 254, 254, 254, 254, 255]);
    assert_eq!(&got[16..24], &[255, 254, 255, 254, 255, 254, 255, 254]);
    assert_eq!(&got[24..], &[254, 255, 254, 255, 254, 255, 254, 255]);
}

#[test]
fn exception_pages_follow_groups_with_clipped_rows() {
    let (rows, columns) = (32, 130);
    let values: Vec<_> = (0..rows * columns).map(|i| (128 + i % 128) as u8).collect();
    let fixture = encode(rows, columns, Quantization::Nvfp4, vec![0; rows], &values);
    assert_eq!(fixture.page_bases, [0u32, 8192].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
    assert_eq!(fixture.offsets.len(), (65 + 2) * 2);
    assert_eq!(fixture.view().expand().unwrap(), values);
    let mut malformed = fixture.clone();
    malformed.page_bases[4..8].copy_from_slice(&8190u32.to_le_bytes());
    assert!(malformed.view().validate().is_err());
    let mut malformed = fixture.clone();
    malformed.offsets[130..132].copy_from_slice(&1u16.to_le_bytes());
    assert!(malformed.view().validate().is_err());
}

#[test]
fn full_group_pages_end_at_the_page_boundary_and_reject_gaps() {
    for rows in [4096, 4224] {
        let values = vec![200; rows * 4];
        let fixture = encode(rows, 4, Quantization::Nvfp4, vec![0; rows], &values);
        let pages = rows.div_ceil(4096);
        assert_eq!(fixture.page_bases.len(), pages * 4);
        assert_eq!(fixture.offsets.len(), (rows / 32 * 2 + pages) * 2);
        assert_eq!(fixture.view().expand().unwrap(), values);
        if pages == 2 {
            assert_eq!(&fixture.page_bases[4..], &32768u32.to_le_bytes());
            for base in [32766u32, 32770] {
                let mut bad = fixture.clone();
                bad.page_bases[4..].copy_from_slice(&base.to_le_bytes());
                assert!(bad.view().validate().is_err());
            }
        }
    }
}

#[test]
fn every_component_length_is_checked() {
    let (fixture, _) = public_vector();
    for component in 0..5 {
        for longer in [false, true] {
            let mut bad = fixture.clone();
            if longer { bad.component(component).push(0); }
            else { bad.component(component).pop(); }
            assert!(bad.view().validate().is_err(), "component {component}, longer {longer}");
            assert!(bad.view().expand().is_err());
            let mut output = [0xd6; 64];
            assert!(bad.view().decode_into(&mut output).is_err());
            assert_eq!(output, [0xd6; 64]);
        }
    }
    let mut bad = fixture.clone();
    bad.exceptions.extend_from_slice(&0u16.to_le_bytes());
    assert!(bad.view().validate().is_err());
    let mut bad = fixture;
    bad.exceptions.truncate(bad.exceptions.len() - 2);
    assert!(bad.view().validate().is_err());
}

#[test]
fn shape_overflow_and_output_lengths_are_checked() {
    let (fixture, _) = public_vector();
    for (rows, columns) in [(0, 2), (1, 2), (33, 2), (32, 0), (32, usize::MAX), (usize::MAX - 31, 3)] {
        let mut bad = fixture.clone(); bad.rows = rows; bad.columns = columns;
        assert!(bad.view().validate().is_err());
        assert!(bad.view().expand().is_err());
    }
    for len in [0, 63, 65] {
        let mut output = vec![0xd6; len];
        assert!(fixture.view().decode_into(&mut output).is_err());
        assert_eq!(output, vec![0xd6; len]);
    }
    let mut oversized = fixture;
    oversized.columns = usize::MAX / 32;
    assert!(oversized.view().dequantize_nvfp4(&[], 1., &mut []).is_err());
}

#[test]
fn palettes_offsets_and_page_bases_are_checked() {
    let (fixture, _) = public_vector();
    let maximum = encode(32, 1, Quantization::Nvfp4, vec![240; 32], &[255; 32]);
    assert!(maximum.exceptions.is_empty());
    assert_eq!(maximum.view().expand().unwrap(), [255; 32]);
    for base in [241, 255] {
        let mut bad = fixture.clone(); bad.bases[31] = base;
        assert!(bad.view().validate().is_err());
    }
    for offsets in [[1, 5], [0, 4], [0, 6], [5, 0], [0, 65]] {
        let mut bad = fixture.clone(); bad.offsets = words(&offsets);
        assert!(bad.view().validate().is_err());
    }
    for base in [1u32, 2, u32::MAX] {
        let mut bad = fixture.clone(); bad.page_bases = base.to_le_bytes().to_vec();
        assert!(bad.view().validate().is_err());
    }
    let mut bad = encode(32, 1, Quantization::Mxfp4, vec![254; 32], &[254; 32]);
    bad.bases[0] = 255;
    assert!(bad.view().validate().is_err());
    let mut bad = encode(32, 4, Quantization::Nvfp4, vec![0; 32], &[200; 128]);
    assert_eq!(bad.offsets, words(&[0, 64, 128]));
    bad.offsets = words(&[0, 64, 63]);
    assert!(bad.view().validate().is_err());
}

#[test]
fn exception_positions_are_sorted_unique_and_within_the_block() {
    let (fixture, _) = public_vector();
    for exceptions in [
        [0x0000, 0x0700, 0x0e1f, 0x1520, 0x1c3f],
        [0x0001, 0x0700, 0x0e1f, 0x1520, 0x1c3f],
        [0x0000, 0x0701, 0x0e1f, 0x1520, 0x1c40],
        [0x0000, 0x0701, 0x0e1f, 0x1520, 0x1cff],
    ] {
        let mut bad = fixture.clone(); bad.exceptions = words(&exceptions);
        assert!(bad.view().validate().is_err());
    }
    let mut values = [56; 32]; values[5] = 128;
    let mut odd_column = encode(32, 1, Quantization::Nvfp4, vec![56; 32], &values);
    assert_eq!(odd_column.view().expand().unwrap(), values);
    odd_column.exceptions[0] = 11;
    assert!(odd_column.view().validate().is_err());
}

#[test]
fn nvfp4_dequantization_uses_existing_block_order_and_signed_zero() {
    let fixture = encode(32, 1, Quantization::Nvfp4, vec![56; 32], &[56; 32]);
    let mut output = vec![0.; 512];
    fixture.view().dequantize_nvfp4(&[0x21; 256], 2., &mut output).unwrap();
    for pair in output.as_chunks::<2>().0 { assert_eq!(pair, &[1., 2.]); }
    fixture.view().dequantize_nvfp4(&[0x80; 256], 2., &mut output).unwrap();
    for pair in output.as_chunks::<2>().0 {
        assert_eq!(pair[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(pair[1].to_bits(), (-0.0f32).to_bits());
    }
    let (fixture, expected) = public_vector();
    let packed: Vec<_> = (0..512).map(|i| i as u8).collect();
    let mut actual = vec![0.; 1024]; let mut reference = vec![0.; 1024];
    fixture.view().dequantize_nvfp4(&packed, 0.5, &mut actual).unwrap();
    engine_model::flash::nvfp4::dequantize(&packed, &expected, 0.5, &mut reference).unwrap();
    assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn nvfp4_dequantization_rejects_invalid_arithmetic_inputs() {
    let fixture = encode(32, 1, Quantization::Nvfp4, vec![56; 32], &[56; 32]);
    for global in [0., -1., f32::INFINITY, f32::NAN] {
        let mut output = [21.; 512];
        assert!(fixture.view().dequantize_nvfp4(&[0; 256], global, &mut output).is_err());
        assert_eq!(output, [21.; 512]);
    }
    for packed in [vec![0; 255], vec![0; 257]] {
        assert!(fixture.view().dequantize_nvfp4(&packed, 1., &mut [0.; 512]).is_err());
    }
    for size in [511, 513] {
        assert!(fixture.view().dequantize_nvfp4(&[0; 256], 1., &mut vec![0.; size]).is_err());
    }
    for value in [127, 128, 255] {
        let mut values = [56; 32]; values[0] = value;
        let invalid = encode(32, 1, Quantization::Nvfp4, vec![56; 32], &values);
        assert!(invalid.view().expand().is_ok());
        let mut output = [21.; 512];
        assert!(invalid.view().dequantize_nvfp4(&[0; 256], 1., &mut output).is_err());
        assert_eq!(output, [21.; 512]);
    }
    let mxfp4 = encode(32, 1, Quantization::Mxfp4, vec![56; 32], &[56; 32]);
    assert!(mxfp4.view().dequantize_nvfp4(&[0; 256], 1., &mut [0.; 512]).is_err());
}
