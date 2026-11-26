use super::*;
use std::io::{self, Read};

#[test]
fn known_hashes_and_unaligned_padding_boundaries() {
    assert_eq!(digest(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(digest(&vec![b'a'; 1_000_000]), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // Independent hashlib/OpenSSL oracle, including unaligned input and 64KiB framing.
    let bytes = (0..131_074).map(|i| ((i * 37 + i / 7) % 256) as u8).collect::<Vec<_>>();
    for (len, expected) in [
        (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (1, "bbf3f11cb5b43e700273a78d12de55e4a7eab741ed2abf13787a4d2dc832b8ec"),
        (55, "40963cdb73c98813265a56f166c073731f216c918281b441cfcf7839ae5f47d7"),
        (56, "66f065c99d7106fcb5553f51b12a8f661ecf11956181d84f20853df61815a050"),
        (63, "97e10f650385cd919943603b8c7eacd2078cc63165b95447cc6f0727d4d51019"),
        (64, "9a5377b648c3c2aafc02db4d8b3a36243f0cbd27dc05de6503a6a8293a48e0c3"),
        (65, "699e21fe42367f55d74b489614475c306b3250840f1679d32d4d67dc94138af2"),
        (119, "75488ef9df8b356662071b09a468a7836e734a89c68b538d1bb2829d44a0b860"),
        (120, "0b25fd0e92c4195fe9282ebc0d5a324c6ee20f49ac8b3e9829038efb2f4b3f8e"),
        (127, "38ca3c04169ac8898ed8c829b5e70800e8751307c08712e3c867a0a0caf9a896"),
        (128, "3b60c938037172ef46fd12eb3cc483ae09f472a68d0bdb2d6b653d049d1d8459"),
        (129, "5898b32fd5a002f2c06087a87737bf874721f30d4299caa96367be7549385cf5"),
        (65535, "8eedb0e734675eb6827dd39544651d73479eba1073d3ef05a487f225686dc902"),
        (65536, "6b9672670ccc3c614c246f539cc0ddf7e8cb81d264352c6541454d3f1321a876"),
        (65537, "ae7a8f7a1d12470eb4f9c58a1adddafe28733e2f9743ce37a2de94569bbf6e78"),
        (131071, "0de84705a5695d0a8f93c5dd34cb35a623c14558a03fc85ec228a50e7aa9c63d"),
        (131072, "6ba730473065b125939526e3b6b1e6b3e278dfebe9f0f230dc2792baf92e9407"),
        (131073, "716340e0278455d1f473d64146e7d802b237cac64ac281724215a6f658268639"),
    ] { assert_eq!(digest(&bytes[1..1 + len]), expected, "length {len}"); }
}

#[test]
fn fragmented_stream_retries_interruption_and_propagates_io_errors() {
    struct Fragmented { at: usize, interrupted: bool }
    impl Read for Fragmented {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if !self.interrupted { self.interrupted = true; return Err(io::ErrorKind::Interrupted.into()); }
            if self.at == 3 { return Ok(0); }
            out[0] = b"abc"[self.at]; self.at += 1; self.interrupted = false; Ok(1)
        }
    }
    assert_eq!(digest_reader(Fragmented { at: 0, interrupted: false }).unwrap(), digest(b"abc"));
    struct Broken;
    impl Read for Broken { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { Err(io::Error::from_raw_os_error(5)) } }
    assert_eq!(digest_reader(Broken).unwrap_err().raw_os_error(), Some(5));
}

struct Generated { remaining: u64, block: [u8; 65536] }
impl Generated {
    fn new(remaining: u64) -> Self { Self { remaining, block: std::array::from_fn(|i| (i % 251) as u8) } }
}
impl Read for Generated {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        assert!(out.len() <= 65536);
        let n = out.len().min(self.block.len()).min(self.remaining as usize);
        out[..n].copy_from_slice(&self.block[..n]); self.remaining -= n as u64; Ok(n)
    }
}

#[test]
fn framing_keeps_64_bit_length_beyond_four_gibibytes() {
    let bytes = (1u64 << 32) + 123;
    let mut blocks_seen = 0u64;
    let mut final_block = [0u8; 64];
    scalar::digest_reader_with(Generated::new(bytes), |_, blocks| {
        blocks_seen += (blocks.len() / 64) as u64;
        final_block.copy_from_slice(&blocks[blocks.len() - 64..]);
    }).unwrap();
    assert_eq!(blocks_seen, bytes / 64 + 2);
    assert_eq!(&final_block[56..], &(bytes * 8).to_be_bytes());
}

#[test]
#[ignore = "4 GiB streaming regression; run explicitly in release mode"]
fn generated_stream_beyond_four_gibibytes_matches_independent_oracle() {
    // Python hashlib, repeating bytes(i % 251 for i in range(65536)), then first123.
    assert_eq!(digest_reader(Generated::new((1u64 << 32) + 123)).unwrap(), "87293bc6285e6d8f4ed2ead2df971f1b4f044d88dd49ad2690c4513553477e06");
}
