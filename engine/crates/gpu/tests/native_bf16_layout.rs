//! Check the byte addresses used by the native scalar projection kernels.
#[allow(dead_code)] #[path="../src/staging.rs"] mod staging;
#[allow(dead_code,unused_imports)] #[path="../src/load.rs"] mod load;

fn native_word(row: usize,col: usize,cols: usize) -> usize {
    // packed_dot_view and wide_row_dot load a Pair at this fragment address.
    let pair=(row/8*(cols/16)+col/16)*32+row%8*4+col%8/2;
    pair*4+col%2+col%16/8*2
}
fn word(bytes: &[u8],at: usize) -> u16 {u16::from_le_bytes(bytes[at*2..at*2+2].try_into().unwrap())}

#[test]
fn native_fragments_keep_every_selected_word_and_padded_row() {
    for cols in [16,48,64,128,192] {for rows in [1usize,7,8,9,32,64,65,129] {for origin in [0,32] {
        let stored=rows.div_ceil(8)*8;
        let mut raw=vec![0u8;stored*cols*2];
        for r in 0..rows {for c in 0..cols {
            let value=((origin+r)*cols+c) as u16;
            raw[(r*cols+c)*2..(r*cols+c+1)*2].copy_from_slice(&value.to_le_bytes());
        }}
        // The uploader can split only at complete row tiles.
        for chunk_rows in [8,24,stored] {
            let mut packed=raw.clone();
            for chunk in packed.chunks_mut(chunk_rows*cols*2) {load::pack_rows_layout(chunk,cols,true);}
            for r in 0..stored {for c in 0..cols {
                assert_eq!(word(&packed,native_word(r,c,cols)),word(&raw,r*cols+c),
                    "rows={rows} cols={cols} origin={origin} row={r} col={c}");
            }}
        }
    }}}
}
#[test]
fn blocked_fragments_have_a_different_native_address() {
    let (rows,cols)=(64,128);
    let raw=(0..rows*cols).flat_map(|v| (v as u16).to_le_bytes()).collect::<Vec<_>>();
    let mut packed=raw.clone();load::pack_rows_layout(&mut packed,cols,false);
    for r in 0..rows {for c in 0..cols {
        // Blocked MMA layout groups four K tiles within each eight-row group.
        let pair=((c/64*32+r/8*4+c%64/16)*32+r%8*4+c%8/2)*4+c%2+c%16/8*2;
        assert_eq!(word(&packed,pair),word(&raw,r*cols+c));
    }}
    assert_ne!(word(&packed,native_word(0,64,cols)),word(&raw,64));
}

#[cfg(target_os="linux")]
#[test]
#[ignore="requires CUDA"]
fn resident_native_projection_copies_global_features_on_both_launch_shapes() {
    use spark_gpu_probe::driver::with_backend;
    with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),usize::MAX,|b| {
        for cols in [48usize,128,192] {for rows in [1usize,9,32,64,65,129] {for origin in [0,32] {
            let raw=(0..rows*cols).flat_map(|i| {
                let value=if i%cols==(origin+i/cols)%cols {0x3f80u16} else {0};
                value.to_le_bytes()
            }).collect::<Vec<_>>();
            let matrix=b.upload_native_bf16(rows,cols,2,|at,out| {out.copy_from_slice(&raw[at..at+out.len()]);Ok(())})?;
            let values=(0..cols).map(|c| c as f32/128.0-0.75).collect::<Vec<_>>();
            let input=b.upload_bf16_lanes(&values)?;let mut out=b.vector(rows)?;
            if cols%64==0 {
                let blocked=b.upload_packed_bf16(rows,cols,2,|at,out| {out.copy_from_slice(&raw[at..at+out.len()]);Ok(())})?;
                assert!(b.project_bf16_resident(&blocked,input.vector(),&mut out).unwrap_err().contains("unblocked"));
            }
            b.project_bf16_resident(&matrix,input.vector(),&mut out)?;b.synchronize()?;
            assert_eq!(b.read_vector_values(&out)?,(0..rows).map(|r| values[(origin+r)%cols]).collect::<Vec<_>>());
            for r in 0..rows {assert_eq!(b.shared_bf16_row(&matrix,r)?,(0..cols).map(|c| if c==(origin+r)%cols {1.0} else {0.0}).collect::<Vec<_>>());}
        }}}
        Ok(())
    }).unwrap();
}
