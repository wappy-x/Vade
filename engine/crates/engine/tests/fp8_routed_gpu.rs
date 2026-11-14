#![cfg(target_os = "linux")]
use spark_gpu_probe::{driver::{Backend, DeviceVector, with_backend}, fp8_abi::*};

fn routed(b: &Backend<'_>, descriptors: &[ExpertProjection], table: &DeviceVector<'_>,
    ids: &[usize], k: usize, bad: Option<usize>, shared: bool) -> Result<(Vec<u32>, Vec<u32>), String> {
    let (t, e, h, i) = (ids.len(), descriptors.len() / 3 - usize::from(shared), descriptors[0].cols as usize, descriptors[0].rows as usize);
    let pairs = t * k;
    let rows = pairs + 15 * pairs.min(e);
    let tasks = pairs.div_ceil(128) + pairs.min(e);
    let mut v = b.vector_arena(&[t, pairs*3, e, e+1, pairs, pairs, t, 1+2*tasks,
        rows, rows, t*h, t*(k+1)*h, t*h/4, t*h/128, rows*i*2,
        rows*i/4, rows*i/128, rows*h])?;
    b.write_vector(&mut v[0], &vec![0.0; t])?; // Newly invalid: no pre-set failure hook.
    let routes: Vec<_> = (0..t).flat_map(|token| (0..k).flat_map(move |lane|
        [token as f32, if bad == Some(token) && lane == k-1 { e as f32 }
            else { ((ids[token]*7+lane*3)%e) as f32 }, 1.0/k as f32])).collect();
    b.write_vector(&mut v[1], &routes)?;
    b.write_vector(&mut v[10], &ids.iter().flat_map(|&id|
        (0..h).map(move |col| ((id*13+col)%31+1) as f32/32.0)).collect::<Vec<_>>())?;
    b.write_vector(&mut v[11], &vec![f32::from_bits(0x7fc00123); t*(k+1)*h])?;
    let a = |n: usize| v[n].device_address();
    let route = GroupedRouteParams { error:a(0), pairs:a(1), counts:a(2), starts:a(3),
        pair_to_row:a(4), token_pairs:a(5), token_count:a(6), tasks:a(7),
        sorted_row:a(8), output_row:a(9), experts:e as u32, tokens:t as u32,
        pair_count:pairs as u32, row_capacity:rows as u32 };
    let grouped = GroupedParams { route, weights:table.device_address(), input:a(10),
        input_scales:0, output:a(11), cols:h as u32, activation_quantization:1, output_shard:0 };
    let p = RoutedParams::checked(grouped, RoutingLayout::checked(k as u32, i as u32).unwrap()).unwrap();
    // All operands are disjoint arena views or retained immutable weights/table.
    unsafe { b.fp8_routed_ffn(p, descriptors, i as u64, [a(12),a(13),a(14),a(15),a(16),a(17)], None)?; }
    Ok((b.read_vector_values(&v[11])?.into_iter().map(f32::to_bits).collect(),
        b.read_vector_values(&v[0])?.into_iter().map(f32::to_bits).collect()))
}

#[test]
#[ignore = "skipped: no GPU; run explicitly with --ignored on a qualified GPU"]
fn fp8_routed_clear_rejected_row_bytes() -> Result<(), String> {
    if std::env::var_os("VADE_TYPED_35B_CHECKPOINT").is_none() {
        return Err("explicit GPU gate requires VADE_TYPED_35B_CHECKPOINT".into());
    }
    with_backend(0, spark_gpu_probe::layer_ptx()?, 1024*1024*1024, |b| {
        let mut resident = Vec::new();
        let mut descriptors = Vec::new();
        for expert in 0..32 {
            let (mut weights, mut scales) = (b.vector(128*128/4)?, b.vector(1)?);
            b.write_vector_bytes(&mut weights, &vec![0x30+(expert%8) as u8; 128*128])?;
            b.write_vector_bytes(&mut scales, &0x3c00u16.to_le_bytes())?;
            let d = ExpertProjection::checked(weights.device_address(), scales.device_address(),128,128,0,0).unwrap();
            descriptors.extend([d; 3]); resident.extend([weights, scales]);
        }
        let bytes: Vec<_> = descriptors.iter().flat_map(|d| d.weights.to_le_bytes().into_iter()
            .chain(d.scales.to_le_bytes()).chain([d.rows,d.cols,d.scale_cols,d.first_k_block,
                d.first_n_block,d.reserved].into_iter().flat_map(u32::to_le_bytes))).collect();
        let mut table = b.vector(bytes.len()/4)?;
        b.write_vector_bytes(&mut table, &bytes)?;
        for k in [1,8,32] { for t in [1,2,3,4,5,17,256] {
            let ids: Vec<_> = (0..t).collect();
            let bad = t/2;
            let (clean, status) = routed(b, &descriptors, &table, &ids, k, None, false)?;
            assert!(status.iter().all(|&v| v == 0));
            let (poisoned, status) = routed(b, &descriptors, &table, &ids, k, Some(bad), false)?;
            let stride = (k+1)*128;
            assert!(poisoned[bad*stride..bad*stride+k*128].iter().all(|&v| v == 0), "t={t} k={k}: +0.0 required");
            assert!(poisoned[bad*stride+k*128..(bad+1)*stride].iter().all(|&v| v == 0x7fc00123), "shared slot untouched");
            assert_eq!(status, (0..t).map(|r| u32::from(r == bad)).collect::<Vec<_>>());
            let healthy: Vec<_> = ids.iter().copied().filter(|&id| id != bad).collect();
            let compact = if healthy.is_empty() { Vec::new() }
                else { routed(b, &descriptors, &table, &healthy, k, None, false)?.0 };
            for (row, &id) in healthy.iter().enumerate() {
                assert_eq!(&poisoned[id*stride..(id+1)*stride], &clean[id*stride..(id+1)*stride], "t={t} k={k} token={id}");
                assert_eq!(&poisoned[id*stride..(id+1)*stride], &compact[row*stride..(row+1)*stride]);
                assert!(clean[id*stride..id*stride+k*128].iter().all(|&v| f32::from_bits(v).is_finite()));
                assert!(clean[id*stride..id*stride+k*128].iter().any(|&v| v != 0));
            }
        } }
        for t in [1,2,3,4] { let ids: Vec<_> = (0..t).collect();
            let (batch,status) = routed(b,&descriptors,&table,&ids,8,None,true)?;
            assert!(status.iter().all(|&v| v==0));
            for row in 0..t { let single = routed(b,&descriptors,&table,&ids[row..row+1],8,None,true)?.0;
                assert_eq!(&batch[row*9*128..(row+1)*9*128],&single);
                assert!(single[8*128..].iter().all(|&v| f32::from_bits(v).is_finite())); }
        }
        Ok(())
    })
}

#[test]
#[ignore = "skipped: no GPU; run explicitly with --ignored on a qualified GPU"]
fn fp8_fused_shared_matches_separate_bit_for_bit() -> Result<(), String> {
    use spark_gpu_probe::load::{WeightDtype, WeightFile, WeightRange};
    use std::io::Write;
    if std::env::var_os("VADE_TYPED_35B_CHECKPOINT").is_none() {
        return Err("explicit GPU gate requires VADE_TYPED_35B_CHECKPOINT".into());
    }
    let path = std::env::temp_dir().join(format!("fp8-shared-{}-{}.bin", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).map_err(|e| e.to_string())?;
    let result = (|| {
        for (h,i) in [(2048,512),(3072,1024)] { // 35B and 122B shared-expert shapes.
            let mut bytes = Vec::new();
            let mut spans = Vec::new();
            for part in 0..6 {
                let (n,k) = if part%3 == 2 { (h,i) } else { (i,h) };
                let start = bytes.len();
                bytes.extend((0..n*k).map(|v| {
                    let code = 0x20 + ((v*13+v/k*7+part*11)%24) as u8;
                    code | if (v/k+v%k+part)%3 == 0 { 0x80 } else { 0 }
                }));
                let scales = bytes.len();
                bytes.extend((0..n*k/16384).flat_map(|v| (0x3c00u16+(v%5) as u16*32).to_le_bytes()));
                spans.push((start,scales,n,k));
            }
            file.set_len(0).map_err(|e| e.to_string())?;
            std::io::Seek::rewind(&mut file).map_err(|e| e.to_string())?;
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            with_backend(0, spark_gpu_probe::layer_ptx()?, 1024*1024*1024, |b| {
                let ranges = [WeightRange { file_offset:0, destination:0, bytes:bytes.len(), row_bytes:0, row_stride:0 }];
                let arena = b.load_weight_arena(bytes.len(), &[WeightFile { file:&file, ranges:&ranges }])?;
                let mut matrices = Vec::new();
                let mut projections = Vec::new();
                for &(start,scales,n,k) in &spans {
                    let weights = b.weight_matrix(&arena,start,n,k,WeightDtype::Fp8)?;
                    let scale = b.weight_matrix(&arena,scales,n/128,k/128,WeightDtype::Bf16)?;
                    projections.push(ExpertProjection::checked(weights.device_address(),scale.device_address(),n as u64,k as u64,0,0).unwrap());
                    matrices.push(weights.with_fp8_scales(scale)?);
                }
                // Routed and shared weights differ: a wrong shared descriptor cannot match.
                let descriptors: Vec<_> = (0..8).flat_map(|_| projections[..3].iter().copied())
                    .chain(projections[3..].iter().copied()).collect();
                let table_bytes: Vec<_> = descriptors.iter().flat_map(|d| d.weights.to_le_bytes().into_iter()
                    .chain(d.scales.to_le_bytes()).chain([d.rows,d.cols,d.scale_cols,d.first_k_block,
                        d.first_n_block,d.reserved].into_iter().flat_map(u32::to_le_bytes))).collect();
                let mut table = b.vector(table_bytes.len()/4)?;
                b.write_vector_bytes(&mut table,&table_bytes)?;
                for t in 1..=4 {
                    let ids: Vec<_> = (0..t).collect();
                    let (fused,status) = routed(b,&descriptors,&table,&ids,8,None,true)?;
                    assert!(status.iter().all(|&v| v==0));
                    let input = b.upload_vector(&(0..t).flat_map(|id|
                        (0..h).map(move |col| ((id*13+col)%31+1) as f32/32.0)).collect::<Vec<_>>())?;
                    let (mut gate,mut up,mut output) = (b.vector(t*i)?,b.vector(t*i)?,b.vector(t*h)?);
                    let mut scratch = b.vector(Backend::fixed_fp8_scratch_words([i,h],t)?
                        .max(Backend::fixed_fp8_scratch_words([h,i],t)?))?;
                    b.fixed_fp8_projection(&matrices[3],&input,&mut gate,&mut scratch,t,true)?;
                    unsafe { b.fixed_fp8_projection_reusing_input(&matrices[4],&input,&mut up,&mut scratch,t,true,true)?; }
                    let (mut g,mut u,mut count,mut divisor,mut precision) =
                        (gate.device_address(),up.device_address(),(t*i) as u32,1.0f32,3u32);
                    unsafe { b.launch_flash_shared("flash_silu",count.div_ceil(128),128,0,
                        &mut [(&mut g as *mut u64).cast(),(&mut u as *mut u64).cast(),(&mut count as *mut u32).cast(),
                            (&mut divisor as *mut f32).cast(),(&mut precision as *mut u32).cast()])?; }
                    b.fixed_fp8_projection(&matrices[5],&gate,&mut output,&mut scratch,t,false)?;
                    let separate: Vec<_> = b.read_vector_values(&output)?.into_iter().map(f32::to_bits).collect();
                    assert!(separate.iter().all(|&v| f32::from_bits(v).is_finite()));
                    assert!(separate.iter().any(|&v| f32::from_bits(v)!=0.0));
                    for row in 0..t { assert_eq!(&fused[(row*9+8)*h..(row*9+9)*h],
                        &separate[row*h..(row+1)*h], "h={h} i={i} t={t} row={row}"); }
                }
                Ok(())
            })?;
        }
        Ok(())
    })();
    drop(file);
    std::fs::remove_file(path).map_err(|e| e.to_string())?;
    result
}
