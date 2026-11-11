//! Host-only refusal tests. All file reads use small checked-in fixtures.
use engine_model::{cartridge::{Cartridge, Kind}, safetensors::{self, Header, Index, Limits}};
use std::{io::{self, Cursor, Read, Seek, SeekFrom}, path::PathBuf};

const HEADER: &str = include_str!("fixtures/host_edges/header.json");
const INDEX: &[u8] = include_bytes!("fixtures/host_edges/index.json");
fn fixture(name: &str) -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/host_edges").join(name) }
fn tensor_bytes(header: &str, count: usize) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes()); bytes.resize(bytes.len() + count, 0); bytes
}
fn index(size: &str, map: &str, limits: Limits) -> engine_model::Result<Index> {
    Index::parse(format!(r#"{{"metadata":{{"total_size":{size}}},"weight_map":{map}}}"#).as_bytes(), limits)
}

#[test]
fn bounded_metadata_read_rejects_directory_missing_file_and_excess_bytes() {
    let path = fixture("index.json");
    assert_eq!(safetensors::read_bounded(&path, INDEX.len()).unwrap(), INDEX);
    for limit in [0, INDEX.len() - 1] {
        assert!(safetensors::read_bounded(&path, limit).unwrap_err().to_string().contains("byte limit"));
    }
    assert!(safetensors::read_bounded(&fixture(""), 1024).unwrap_err().to_string().contains("regular file"));
    assert!(safetensors::read_bounded(&fixture("absent.json"), 1024).is_err());
}

struct FaultReader { bytes: Cursor<Vec<u8>>, seek_calls: usize, read_calls: usize, fail_seek: usize, fail_read: usize, interrupted: bool, eof: bool }
impl Read for FaultReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.read_calls += 1;
        if self.read_calls == self.fail_read { return if self.eof { Ok(0) } else { Err(io::Error::other("fixture read fault")) }; }
        if self.interrupted { self.interrupted = false; return Err(io::ErrorKind::Interrupted.into()); }
        self.bytes.read(output)
    }
}
impl Seek for FaultReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.seek_calls += 1;
        if self.seek_calls == self.fail_seek { return Err(io::Error::other("fixture seek fault")); }
        self.bytes.seek(position)
    }
}
#[test]
fn header_reader_propagates_seek_read_and_eof_errors() {
    for (fail_seek, fail_read, reason) in [(1, 0, "seek"), (2, 0, "seek"), (0, 1, "read"), (0, 2, "read")] {
        let mut reader = FaultReader { bytes: Cursor::new(tensor_bytes(HEADER, 4)), seek_calls: 0, read_calls: 0, fail_seek, fail_read, interrupted: false, eof: false };
        let error = Header::read(&mut reader, Limits::default()).unwrap_err().to_string();
        assert!(error.contains(&format!("fixture {reason} fault")), "{error}");
    }
    for fail_read in [1, 2] {
        let mut reader = FaultReader { bytes: Cursor::new(tensor_bytes(HEADER, 4)), seek_calls: 0, read_calls: 0, fail_seek: 0, fail_read, interrupted: false, eof: true };
        assert!(Header::read(&mut reader, Limits::default()).is_err());
        assert_eq!(reader.read_calls, fail_read, "EOF must stop the reader");
    }
    let mut reader = FaultReader { bytes: Cursor::new(tensor_bytes(HEADER, 4)), seek_calls: 0, read_calls: 0, fail_seek: 0, fail_read: 0, interrupted: true, eof: false };
    assert_eq!(Header::read(&mut reader, Limits::default()).unwrap().tensors["a"].byte_len(), 4);
    assert_eq!(reader.bytes.position(), 8 + HEADER.len() as u64, "inspection must not read payload");
    for cut in [0, 7, 8, 8 + HEADER.len() - 1, 8 + HEADER.len()] {
        let bytes = tensor_bytes(HEADER, 4);
        assert!(Header::read(&mut Cursor::new(&bytes[..cut]), Limits::default()).is_err(), "cut {cut}");
    }
}

#[test]
fn header_resource_limits_accept_boundary_and_reject_one_over() {
    let accepted = Limits { header_bytes: HEADER.len(), tensors: 1, rank: 1, shards: 1 };
    assert!(Header::read(&mut Cursor::new(tensor_bytes(HEADER, 4)), accepted).is_ok());
    for (limits, reason) in [
        (Limits { header_bytes: HEADER.len() - 1, ..accepted }, "header limit"),
        (Limits { tensors: 0, ..accepted }, "tensor count"),
        (Limits { rank: 0, ..accepted }, "rank limit"),
    ] {
        let error = Header::read(&mut Cursor::new(tensor_bytes(HEADER, 4)), limits).unwrap_err().to_string();
        assert!(error.contains(reason), "{error}");
    }
    for (header, reason) in [
        (r#"{"":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#, "empty tensor name"),
        (r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0]}}"#, "exactly two"),
        (r#" {"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#, "must start"),
    ] {
        assert!(Header::read(&mut Cursor::new(tensor_bytes(header, 4)), Limits::default()).unwrap_err().to_string().contains(reason));
    }
}

#[test]
fn seeded_tensor_truncation_never_yields_a_valid_header() {
    let mut seed = 0x6704_128a_73cb_90efu64;
    for case in 0..48 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let columns = 1 + (seed as usize % 13);
        let rows = 1 + ((seed >> 32) as usize % 7);
        let payload = rows * columns * 2;
        let header = format!(r#"{{"t{case}":{{"dtype":"BF16","shape":[{rows},{columns}],"data_offsets":[0,{payload}]}}}}"#);
        let bytes = tensor_bytes(&header, payload);
        let parsed = Header::read(&mut Cursor::new(&bytes), Limits::default()).unwrap();
        assert_eq!(parsed.tensors[&format!("t{case}")].byte_len(), payload as u64);
        for cut in 0..bytes.len() {
            assert!(Header::read(&mut Cursor::new(&bytes[..cut]), Limits::default()).is_err(), "case {case}, cut {cut}");
        }
    }
}

#[test]
fn index_total_size_requires_an_exact_decimal_integer() {
    let map = r#"{"a":"a.safetensors"}"#;
    for (size, expected) in [("0", 0), ("-0", 0), ("-0.0", 0), ("0e9999999999999999999999999", 0), ("4.0", 4), ("4e1", 40), ("0.004e3", 4), ("4000e-3", 4), ("9007199254740992.0", 9007199254740992), ("18446744073709551615", u64::MAX)] {
        assert_eq!(index(size, map, Limits::default()).unwrap().total_size, expected, "{size}");
    }
    let mut accepted = Vec::new();
    for size in ["-1", "0.5", "1.0000000000000001", "9007199254740993.0", "9007199254740991.5", "18446744073709551616", "1e-1000", "1e99999999999999999999999", "true", "null", "\"4\""] {
        if let Ok(index) = index(size, map, Limits::default()) { accepted.push((size, index.total_size)); }
    }
    assert!(accepted.is_empty(), "accepted inexact or invalid sizes: {accepted:?}");
    assert!(index(&format!("0.{}1", "0".repeat(512)), map, Limits::default()).is_err());
    let mut seed = 0x127f_34ca_0862_d5b9u64;
    for _ in 0..128 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let expected = seed >> 32;
        for size in [format!("{expected}.0000"), format!("{expected}000e-3"), format!("{expected}00.00e-2")] {
            assert_eq!(index(&size, map, Limits::default()).unwrap().total_size, expected, "{size}");
        }
        assert!(index(&format!("{expected}.0000000000000001"), map, Limits::default()).is_err());
    }
}

#[test]
fn index_limits_count_tensors_and_unique_shards_independently() {
    let shared = r#"{"a":"a.safetensors","b":"a.safetensors"}"#;
    let separate = r#"{"a":"a.safetensors","b":"b.safetensors"}"#;
    let limits = Limits { tensors: 2, shards: 1, ..Limits::default() };
    assert_eq!(index("4", shared, limits).unwrap().weight_map.len(), 2);
    assert!(index("4", separate, limits).unwrap_err().to_string().contains("shard count"));
    assert!(index("4", shared, Limits { tensors: 1, ..limits }).unwrap_err().to_string().contains("tensor count"));
    for map in ["{}", r#"{"":"a.safetensors"}"#] { assert!(index("0", map, limits).is_err()); }
    assert!(Index::parse(INDEX, Limits { header_bytes: INDEX.len(), ..limits }).is_ok());
    assert!(Index::parse(INDEX, Limits { header_bytes: INDEX.len() - 1, ..limits }).is_err());
}

#[test]
fn mixed_index_rejects_trailing_slash_and_path_size_limits() {
    let parse = |name: &str| Index::parse_mixed(format!(r#"{{"metadata":{{}},"weight_map":{{"a":"{name}"}}}}"#).as_bytes(), Limits::default());
    for name in ["a.safetensors/", "dir/a.safetensors/", "dir//a.safetensors", ".hidden/../a.safetensors", "dir/./a.safetensors"] {
        assert!(parse(name).is_err(), "accepted {name}");
    }
    let name = format!("{}a.safetensors", "d/".repeat(31));
    assert!(parse(&name).is_ok());
    assert!(parse(&format!("d/{name}")).is_err());
    let name = format!("{}.safetensors", "a".repeat(4096 - ".safetensors".len()));
    assert!(parse(&name).is_ok());
    assert!(parse(&format!("a{name}")).is_err());
}

fn cartridge() -> Cartridge {
    Cartridge { kind: Kind::Qwen3, skipped_bytes: 0, state_bytes: 7, kv_bytes_per_token: 3, max_positions: 100,
        architecture: "synthetic".into(), precision: "synthetic".into(), weight_bytes: 11, config: None, reason: None,
        gemma4: None, weights_complete: false }
}
#[test]
fn cartridge_fit_rejects_context_and_memory_overflow_at_boundaries() {
    let c = cartridge();
    for context in [0, 101, u64::MAX] { assert!(c.fit(context, u64::MAX).is_err(), "context {context}"); }
    let fit = c.fit(100, u64::MAX).unwrap();
    assert_eq!(fit.kv_bytes, 300); assert_eq!(fit.required_bytes, 11 + 7 + 300 + fit.scratch_reserve_bytes);
    assert!(c.fit(100, fit.required_bytes).unwrap().fits);
    assert!(!c.fit(100, fit.required_bytes - 1).unwrap().fits);
    let mut c = cartridge(); c.kv_bytes_per_token = u64::MAX;
    assert!(c.fit(2, u64::MAX).err().unwrap().to_string().contains("KV size overflow"));
    for (weight, state) in [(u64::MAX, 0), (0, u64::MAX), (u64::MAX - 3, 0)] {
        let mut c = cartridge(); c.weight_bytes = weight; c.state_bytes = state;
        assert!(c.fit(1, u64::MAX).err().unwrap().to_string().contains("cartridge memory overflow"));
    }
}
