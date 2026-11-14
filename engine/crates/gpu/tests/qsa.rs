use spark_gpu_probe::qsa::{*, contract::*};
fn data(n: usize) -> Vec<f32> { (0..n).map(|i| ((i * 17 + i / 131) % 103) as f32 / 128.0 - 0.4).collect() }
fn same(a: &[f32], b: &[f32]) { assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()); }
fn rope(p: usize) -> Rope { std::array::from_fn(|i| if i < 32 { (p as f32 * 0.03).cos() } else { (p as f32 * 0.03).sin() }) }
#[test]
fn cpu_contract() {
    let mut cache = IndexCache::default(); for p in 0..2055 { cache.append(&[p as f32 / 4096.0; DIM], &[0.0; DIM], &rope(p), 1e-6).unwrap(); }
    for n in [1, 3, 4, 2047, 2048, 2049, 2051, 2052, 2055] { let (_, s) = cache.decode(&[0.0; 512], n - 1).unwrap(); let expected: Vec<_> = (0..(n / 4 * 4).min(2048)).chain(n / 4 * 4..n).map(|p| p as u32).collect(); assert_eq!(s, expected); }
    let qs = [[0.0; 512]; 7]; assert_eq!(cache.prefill(&qs, 2048).unwrap(), (2048..2055).map(|p| cache.decode(&qs[0], p).unwrap().1).collect::<Vec<_>>());
    let queries = vec![[0.125; 512]; 17];
    let solo: Vec<_> = (0..17).map(|p| cache.decode(&queries[p], p+2038).unwrap().1).collect();
    for chunk in [1, 7, 16, 63, 64, 65, 127, 512] {
        let got: Vec<_> = queries.chunks(chunk).enumerate().flat_map(|(i, q)| cache.prefill(q, 2038+i*chunk).unwrap()).collect();
        assert_eq!(got, solo);
    }
    for batch in [1, 2, 8, 16] { for rows in (0..17).collect::<Vec<_>>().chunks(batch) {
        for &p in rows.iter().rev() { assert_eq!(cache.decode(&queries[p], p+2038).unwrap().1, solo[p]); }
    } }
    let q: [f32; 512] = std::array::from_fn(|i| [1.0, -1.0, 2.0, -2.0][i / DIM]); assert_eq!(score(&q, &[1.0; DIM]).to_bits(), (384.0 / 128.0f32.sqrt()).to_bits());
    let raw = [0.25; DIM]; let mut small = IndexCache::default(); for p in 0..4 { small.append(&raw, &[0.0; DIM], &rope(p), 1e-6).unwrap(); } same(&small.keys[0], &normalize_rotate(raw, &[0.0; DIM], &rope(0), 1e-6));
    let q = vec![0.0; 6144]; let kv = data(8 * 512); let out = sparse_attention(&q, &kv, &kv, &[1, 3, 6]).unwrap(); for h in 0..24 { for d in 0..256 { let mut expected = 0.0; for p in [1, 3, 6] { expected += (1.0 / 3.0) * kv[p * 512 + h / 12 * 256 + d]; } assert_eq!(out[h * 256 + d].to_bits(), expected.to_bits()); } }
    for i in 0..10400 { let x = -(i as f32) / 100.0; assert!((exponential(x) - x.exp()).abs() <= 2e-7 * x.exp() + f32::from_bits(2), "x={x}"); }
    let m = CacheBytes::for_context(LIMIT, 4).unwrap(); assert_eq!(m.indexer, LIMIT * 1536); assert_eq!(m.kv, LIMIT * 49152); assert_eq!(m.total(), LIMIT * 50688 + 27744); assert!(cache.decode(&[0.0; 512], 2055).is_err()); assert!(CacheBytes::for_context(LIMIT + 1, 4).is_err());
}
#[cfg(target_os = "linux")]
#[test]
#[ignore = "skipped: no GPU; requires CUDA; QSA_CONTEXT=8192,131072,1048576 selects the context"]
fn gpu_exact() {
    use spark_gpu_probe::{driver::with_backend, layer_ptx};
    let n = std::env::var("QSA_CONTEXT").unwrap_or("8192".into()).parse::<usize>().unwrap(); assert!((8..=LIMIT).contains(&n));
    with_backend(0, layer_ptx().unwrap(), usize::MAX, |b| {
        let zero = b.upload_vector(&[0.0; DIM])?; let ring = b.upload_vector(&[0.0; 512])?; let cache = b.vector(2 * DIM)?; let mut cpu = IndexCache::default(); let x = data(9 * 640); let mut past = 0;
        for rows in [3, 1, 5] { let input = b.upload_vector(&x[past * 640..(past + rows) * 640])?; let qr: Vec<_> = (past..past + rows).flat_map(rope).collect(); let gr: Vec<_> = (past / 4..(past + rows) / 4).flat_map(|g| rope(g * 4)).collect(); let qr = b.upload_vector(&qr)?; let gr = b.upload_vector(if gr.is_empty() { &[0.0] } else { &gr })?; let q = b.vector(rows * 512)?;
            unsafe { b.qsa_prepare(Prepare { qk: input.device_address(), query_rope: qr.device_address(), group_rope: gr.device_address(), qw: zero.device_address(), kw: zero.device_address(), ring: ring.device_address(), cache: cache.device_address(), query: q.device_address(), rows: rows as u32, past: past as u32, epsilon: 1e-6 })?; }
            let mut expected = Vec::new(); for p in past..past + rows { cpu.append(x[p * 640 + 512..(p + 1) * 640].try_into().unwrap(), &[0.0; DIM], &rope(p), 1e-6)?; let raw: [[f32; DIM]; HEADS] = std::array::from_fn(|h| x[p * 640 + h * DIM..p * 640 + (h + 1) * DIM].try_into().unwrap()); expected.extend(prepare_query(&raw, &[0.0; DIM], &rope(p), 1e-6)); } same(&expected, &b.read_vector(&q)?); past += rows;
        }
        same(&cpu.keys.iter().flatten().copied().collect::<Vec<_>>(), &b.read_vector(&cache)?);
        let compressed: Vec<[f32; DIM]> = data(n / 4 * DIM).as_chunks::<DIM>().0.to_vec(); let mut index = data(2 * 512); index[512..].fill(0.0); let dc = b.upload_vector(&compressed.iter().flatten().copied().collect::<Vec<_>>())?; let dq = b.upload_vector(&index)?; let scores = b.vector(2 * (n / 4))?; let ids = b.vector(2 * WIDTH)?; let counts = b.vector(2)?;
        unsafe { b.qsa_select(Select { query: dq.device_address(), cache: dc.device_address(), scores: scores.device_address(), selected: ids.device_address(), counts: counts.device_address(), rows: 2, past: (n - 2) as u32, stride: (n / 4) as u32 })?; }
        let actual_scores = b.read_vector_values(&scores)?; let actual_ids = b.read_vector_values(&ids)?; let actual_counts = b.read_vector_values(&counts)?; let mut selections = Vec::new();
        for r in 0..2 { let (expected, selected) = select_blocks(index[r * 512..(r + 1) * 512].try_into().unwrap(), &compressed, n - 1 + r)?; same(&expected, &actual_scores[r * (n / 4)..r * (n / 4) + expected.len()]); assert_eq!(actual_counts[r].to_bits() as usize, selected.len()); assert_eq!(actual_ids[r * WIDTH..r * WIDTH + selected.len()].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), selected); assert!(actual_ids[r * WIDTH + selected.len()..(r + 1) * WIDTH].iter().all(|x| x.to_bits() == u32::MAX)); selections.push(selected); }
        let kv = data(n * 512); let query = data(2 * 6144); let dk = b.upload_vector(&kv)?; let dq = b.upload_vector(&query)?; let weights = b.vector(2 * 24 * WIDTH)?; let out = b.vector(query.len())?;
        let a = Attention { query: dq.device_address(), keys: dk.device_address(), values: dk.device_address(), selected: ids.device_address(), counts: counts.device_address(), weights: weights.device_address(), output: out.device_address(), rows: 2 };
        let start = std::time::Instant::now(); unsafe { b.qsa_attention(a)?; } b.synchronize()?; eprintln!("QSA context={n} attention_two_rows_us={}", start.elapsed().as_micros()); let actual = b.read_vector(&out)?;
        for r in 0..2 { same(&sparse_attention(&query[r * 6144..(r + 1) * 6144], &kv, &kv, &selections[r])?, &actual[r * 6144..(r + 1) * 6144]); } Ok(())
    }).unwrap();
}

#[test]
fn nonfinite_scores_return_errors_and_equal_scores_keep_position_order() {
    let keys = vec![[1.0; DIM]; 513];
    assert!(select_blocks(&[f32::INFINITY; HEADS * DIM], &keys, 2052).is_err());
    let (_, selected) = select_blocks(&[-0.0; HEADS * DIM], &keys, 2052).unwrap();
    assert_eq!(selected, (0..2048).collect::<Vec<_>>());
}
