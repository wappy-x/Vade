use engine_model::qwen3::Checkpoint;
use std::{path::PathBuf, time::Instant};
fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 2 && args[0] == "--layout-fixture" {
        match engine_model::layout::LayoutReport::open_fixture(&PathBuf::from(&args[1])) {
            Ok(report) => print!("{}", report.text),
            Err(error) => { eprintln!("fixture rejected: {error}"); std::process::exit(1); }
        }
        return;
    }
    if args.len() != 1 {
        eprintln!("usage: inspect-checkpoint CHECKPOINT_DIRECTORY");
        eprintln!("       inspect-checkpoint --layout-fixture FIXTURE_DIRECTORY");
        std::process::exit(2);
    }
    let start = Instant::now();
    match Checkpoint::inspect(&PathBuf::from(&args[0])) {
        Ok(c) => {
            println!("checkpoint={} BF16; validation=metadata-and-shape-only", if c.config.llama_rope.is_some() { "Llama dense" } else if c.config.fused_projections { "Phi3 dense" } else { "Qwen3" });
            println!(
                "shards={} tensors={} tensor_bytes={}",
                c.shards.len(),
                c.index.weight_map.len(),
                c.index.total_size
            );
            println!(
                "layers={} hidden={} intermediate={} query_heads={} kv_heads={} head_dim={}",
                c.config.layers,
                c.config.hidden_size,
                c.config.intermediate_size,
                c.config.attention_heads,
                c.config.kv_heads,
                c.config.head_dim
            );
            println!(
                "configured_max_positions={} vocab_size={}",
                c.config.max_positions, c.config.vocab_size
            );
            for (name, h) in &c.shards {
                println!(
                    "shard={name} header_bytes={} file_bytes={} tensors={}",
                    h.header_bytes,
                    h.file_bytes,
                    h.tensors.len()
                );
            }
            println!(
                "payload_bytes_read=0 inspection_us={}",
                start.elapsed().as_micros()
            );
        }
        Err(error) => {
            eprintln!("checkpoint rejected: {error}");
            std::process::exit(1);
        }
    }
}
