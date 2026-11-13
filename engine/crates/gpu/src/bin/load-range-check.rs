//! Focused staged-loader checks: exact rows, uneven TP ranks, bounds, cleanup.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use std::{ffi::CString, fs::OpenOptions, io::Write};
    use spark_gpu_probe::{driver::with_backend, load::{WeightRows, WeightShard}};
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() { return Err("usage: load-range-check PTX [TEMP_DIRECTORY]".into()); }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let path = args.get(1).map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir)
        .join(format!("vade-rows-{}", std::process::id()));
    let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path).map_err(|e| e.to_string())?;
    std::fs::remove_file(path).map_err(|e| e.to_string())?;
    const ROWS: usize = 1031;
    const COLS: usize = 4097;
    let word = |row, col| 0x3f00u16 + ((row * 31 + col * 7) % 255) as u16;
    file.write_all(&[0u8; 123]).map_err(|e| e.to_string())?;
    for row in 0..ROWS {
        let bytes = (0..COLS).flat_map(|col| word(row, col).to_le_bytes()).collect::<Vec<_>>();
        file.write_all(&bytes).map_err(|e| e.to_string())?;
    }
    file.sync_all().map_err(|e| e.to_string())?;
    // Reads must use the retained inode, including an unaligned final sector.
    with_backend(0, &ptx, 64 * 1024 * 1024, |backend| {
        for ranks in [1, 2, 3, 5] {
            for rank in 0..ranks {
                let rows = ROWS * rank / ranks..ROWS * (rank + 1) / ranks;
                let plan = [WeightRows { file_offset: 123, rows: rows.clone(), columns: COLS, packed: false, fp8: false }];
                let matrices = backend.load_bf16_shards(&[WeightShard { file: &file, matrices: &plan }])?;
                if backend.allocated_bytes() != rows.len() * COLS * 2 { return Err("rank allocated unowned rows".into()); }
                for row in [0, rows.len() / 2, rows.len() - 1] {
                    let actual = backend.shared_bf16_row(&matrices[0][0], row)?;
                    if actual.iter().enumerate().any(|(col, value)| value.to_bits() != (word(rows.start + row, col) as u32) << 16) {
                        return Err("rank row bytes differ".into());
                    }
                }
                drop(matrices);
                if backend.allocated_bytes() != 0 { return Err("rank arena leaked".into()); }
            }
        }
        let valid = [WeightRows { file_offset: 123, rows: 0..1, columns: COLS, packed: false, fp8: false }];
        let invalid = [WeightRows { file_offset: 123, rows: ROWS..ROWS + 1, columns: COLS, packed: false, fp8: false }];
        // The second arena is already allocating when the first upload fails.
        if backend.load_bf16_shards(&[WeightShard { file: &file, matrices: &invalid }, WeightShard { file: &file, matrices: &valid }]).is_ok()
            || backend.allocated_bytes() != 0 { return Err("error cleanup failed".into()); }
        println!("PASS staged rows=1031 columns=4097 ranks=1,2,3,5 exact_samples=33 error_cleanup=checked");
        Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("load-range-check requires Linux CUDA"); std::process::exit(1); }
}
