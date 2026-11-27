//! Fast pure-Rust slicing-by-8 IEEE CRC-32 (zero unsafe, zero third-party crates).
const POLY: u32 = 0xEDB8_8320;

const fn make_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if (c & 1) != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        tables[0][i] = c;
        i += 1;
    }
    let mut t = 1;
    while t < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[t - 1][i];
            tables[t][i] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
            i += 1;
        }
        t += 1;
    }
    tables
}

static TABLES: [[u32; 256]; 8] = make_tables();

/// Compute the IEEE CRC-32 checksum of `data`.
pub(super) fn checksum(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    let (chunks, tail) = data.as_chunks::<8>();
    for chunk in chunks {
        let low = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ crc;
        let high = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        crc = TABLES[7][(low & 0xFF) as usize]
            ^ TABLES[6][((low >> 8) & 0xFF) as usize]
            ^ TABLES[5][((low >> 16) & 0xFF) as usize]
            ^ TABLES[4][(low >> 24) as usize]
            ^ TABLES[3][(high & 0xFF) as usize]
            ^ TABLES[2][((high >> 8) & 0xFF) as usize]
            ^ TABLES[1][((high >> 16) & 0xFF) as usize]
            ^ TABLES[0][(high >> 24) as usize];
    }
    for &b in tail {
        crc = TABLES[0][((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

#[test]
fn ieee_crc32_matches_standard_vector_and_detects_single_bit_flip() {
    assert_eq!(checksum(b""), 0x0000_0000);
    assert_eq!(checksum(b"123456789"), 0xCBF4_3926);
    let mut buf: Vec<u8> = (0usize..1027).map(|i| (i.wrapping_mul(37) & 0xFF) as u8).collect();
    let expected = checksum(&buf);
    buf[513] ^= 1;
    assert_ne!(checksum(&buf), expected);
}
