use engine_model::safetensors::{Header, Limits};
use std::path::PathBuf;
fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 1 {
        eprintln!("usage: inspect-tensor-file FILE");
        std::process::exit(2);
    }
    match Header::open(&PathBuf::from(&args[0]), Limits::default()) {
        Ok(header) => println!(
            "tensors={} payload_bytes={} header_bytes={}",
            header.tensors.len(),
            header.file_bytes - header.data_start,
            header.header_bytes
        ),
        Err(error) => {
            eprintln!("tensor file rejected: {error}");
            std::process::exit(1);
        }
    }
}
