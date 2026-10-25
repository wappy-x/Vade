//! First-party SHA-256, shared in design with engine/tools/export.
//!
//! [`Sha256`] hashes incremental input, [`sha`] hashes one byte slice and
//! [`digest_reader`] hashes a stream with bounded scratch space. All three
//! return the lowercase hex digest.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (word, bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }
    for i in 16..64 {
        let x = w[i - 15];
        let y = w[i - 2];
        let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
        let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choice = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(choice)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(majority);
        [h, g, f, e, d, c, b, a] = [g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2)];
    }
    for (word, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(value);
    }
}

/// Incremental SHA-256 state.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    pending: Vec<u8>,
    length: u64,
}
impl Sha256 {
    /// Start an empty hash.
    pub fn new() -> Self {
        Self { state: INITIAL, pending: Vec::with_capacity(64), length: 0 }
    }
    /// Append bytes to the hashed message.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if !self.pending.is_empty() {
            let n = (64 - self.pending.len()).min(data.len());
            self.pending.extend_from_slice(&data[..n]);
            data = &data[n..];
            if let Ok(block) = <&[u8; 64]>::try_from(self.pending.as_slice()) {
                compress(&mut self.state, block);
                self.pending.clear();
            }
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            compress(&mut self.state, block);
        }
        self.pending.extend_from_slice(rest);
    }
    /// Pad the message and return the lowercase hex digest.
    pub fn finalize(mut self) -> String {
        let mut tail = [0u8; 128];
        tail[..self.pending.len()].copy_from_slice(&self.pending);
        tail[self.pending.len()] = 0x80;
        let padded = if self.pending.len() < 56 { 64 } else { 128 };
        tail[padded - 8..padded].copy_from_slice(&self.length.wrapping_mul(8).to_be_bytes());
        for block in tail[..padded].as_chunks::<64>().0 {
            compress(&mut self.state, block);
        }
        self.state.iter().map(|v| format!("{v:08x}")).collect()
    }
}

/// Hash one byte slice.
pub fn sha(data: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(data);
    hash.finalize()
}

/// Hash a stream with bounded scratch space, including executables and metadata.
pub fn digest_reader(mut reader: impl std::io::Read) -> std::io::Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => return Ok(hash.finalize()),
            Ok(n) => hash.update(&buffer[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn digest_and_streaming_boundaries() {
        assert_eq!(
            sha(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest_reader(&b""[..]).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let bytes = vec![b'x'; 70000];
        let expected = "bca09f4a757d5571c7d9f3341d4301f3c391c090826acc1a3013c6bcb7c01722";
        assert_eq!(sha(&bytes), expected);
        assert_eq!(digest_reader(&bytes[..]).unwrap(), expected);
        assert_eq!(digest_reader(std::io::Read::chain(&bytes[..7], &bytes[7..])).unwrap(), expected);
        for width in [1, 3, 55, 56, 63, 64, 65] {
            let mut hash = Sha256::new();
            for chunk in bytes[..129].chunks(width) {
                hash.update(chunk);
            }
            assert_eq!(hash.finalize(), sha(&bytes[..129]));
        }
    }
}
