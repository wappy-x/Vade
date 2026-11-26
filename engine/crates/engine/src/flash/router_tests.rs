//! Device byte gates for the deferred corrected router and ungated finish.
//! These tests own all spans passed through the existing launch helper.
use engine_model::host_ops::{router::{Router, RouterKind},moe::ungated_shared};
use spark_gpu_probe::{driver::{Backend,with_backend},router_abi::{CorrectedSigmoidConfig,CorrectedSigmoidRoute,UngatedMoeFinish,ROUTER_THREADS}};
type Result<T> = std::result::Result<T,String>;
const POISON: u32 = 0x7fc00042;
fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
fn op(e: usize,k: usize,g: usize,keep: usize,norm: bool) -> Router {
    Router::from_config(format!(r#"{{"n_routed_experts":{e},"num_experts_per_tok":{k},"n_group":{g},"topk_group":{keep},
        "norm_topk_prob":{norm},"routed_scaling_factor":2.5,"scoring_func":"sigmoid","topk_method":"noaux_tc"}}"#).as_bytes()).unwrap()
}
fn route(b: &Backend<'_>, op: &Router, logits: &[f32], bias: &[f32], status: &[u32]) -> Result<(Vec<u32>,Vec<u32>,Vec<u32>)> {
    assert_eq!(op.kind(),RouterKind::CorrectedSigmoid);
    let (t,e,k) = (status.len(),op.experts(),op.selected());
    assert_eq!(logits.len(),t*e); assert_eq!(bias.len(),e);
    let log = b.upload_vector(logits)?;
    let bias = b.upload_vector(bias)?;
    let poison = f32::from_bits(POISON);
    let ids = b.upload_vector(&vec![poison;(t+2)*k])?;
    let weights = b.upload_vector(&vec![poison;(t+2)*k])?;
    let mut flags: Vec<_> = status.iter().map(|&v| f32::from_bits(v)).collect(); flags.extend([poison;2]);
    let flags = b.upload_vector(&flags)?;
    let config = CorrectedSigmoidConfig::checked(e as u32,k as u32,op.groups() as u32,op.selected_groups() as u32,op.normalize(),op.scale()).unwrap();
    let p = CorrectedSigmoidRoute { config,logits: log.device_address(),bias: bias.device_address(),ids: ids.device_address(),
        weights: weights.device_address(),status: flags.device_address(),tokens: t as u32 }.checked().unwrap();
    super::gdn::launch_batch(b,"corrected_sigmoid_route",t*ROUTER_THREADS as usize,p)?;
    let (ids,weights,flags) = (bits(&b.read_vector_values(&ids)?),bits(&b.read_vector_values(&weights)?),bits(&b.read_vector_values(&flags)?));
    assert_eq!(&ids[t*k..],vec![POISON;2*k]); assert_eq!(&weights[t*k..],vec![POISON;2*k]);
    assert_eq!(&flags[t..],[POISON;2]);
    Ok((ids[..t*k].to_vec(),weights[..t*k].to_vec(),flags[..t].to_vec()))
}

#[test]
#[ignore = "requires an authorized CUDA device; compare router bytes"]
fn corrected_router_bytes_and_row_partitions() {
    with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),usize::MAX,|b| {
        let saved = Router::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap();
        for op in [op(4,2,2,1,true),op(6,3,3,2,false),saved,op(1024,4,512,2,true)] {
            let (e,k,t) = (op.experts(),op.selected(),5);
            // Exact sigmoid hand values, saturation, and a non-binary denominator.
            let rows: Vec<_> = (0..t*e).map(|i| match (i/e)%5 {
                0 => 0.0,1 => if i%2 == 0 { 3.0f32.ln() } else { -3.0f32.ln() },
                2 => -1000.0,3 => 1000.0,_ => if i%2 == 0 { -50.0 } else { 0.0 },
            }).collect();
            let bias: Vec<_> = (0..e).map(|i| ((i*7)%11) as f32/8.0-1.0).collect();
            let expected: Vec<_> = rows.chunks(e).map(|r| op.route(r,&bias).unwrap()).collect();
            let expected_ids: Vec<_> = expected.iter().flat_map(|r| r.0.iter().map(|&id| id as u32)).collect();
            let expected_weights: Vec<_> = expected.iter().flat_map(|r| bits(&r.1)).collect();
            let baseline = route(b,&op,&rows,&bias,&vec![0;t])?;
            assert_eq!(baseline.0,expected_ids); assert_eq!(baseline.1,expected_weights); assert_eq!(baseline.2,vec![0;t]);
            for chunk in [1,2,3,5] {
                for _ in 0..3 {
                    let (mut ids,mut weights) = (Vec::new(),Vec::new());
                    for block in rows.chunks(e*chunk) {
                        let actual = route(b,&op,block,&bias,&vec![0;block.len()/e])?;
                        ids.extend(actual.0); weights.extend(actual.1); assert!(actual.2.iter().all(|&v| v == 0));
                    }
                    assert_eq!(ids,baseline.0); assert_eq!(weights,baseline.1);
                }
            }
            // All ties must choose lower group IDs, then lower expert IDs.
            let tied = route(b,&op,&vec![0.0;e],&vec![0.0;e],&[0])?;
            assert_eq!(tied.0,(0..k as u32).collect::<Vec<_>>());
            println!("router byte gate: experts={e} selected={k} rows={t} groups={} normalize={} PASS",op.groups(),op.normalize());
        }
        Ok(())
    }).unwrap();
}

#[test]
#[ignore = "requires an authorized CUDA device; reject invalid and sticky rows"]
fn corrected_router_rejects_nonfinite_and_preserves_sticky_status() {
    with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),usize::MAX,|b| {
        let op = op(4,2,2,1,true);
        for bad in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
            let rows = [0.0,0.0,0.0,0.0, bad,0.0,0.0,0.0, 0.0,0.0,0.0,0.0];
            let out = route(b,&op,&rows,&[0.0;4],&[0,0,9])?;
            assert_eq!(out.2,[0,1,9]); assert_eq!(&out.0[2..],[u32::MAX;4]);
            assert!(out.1[2..].iter().all(|&v| f32::from_bits(v).is_nan()));
            let bias = [0.0,0.0,bad,0.0];
            assert_eq!(route(b,&op,&[0.0;4],&bias,&[0])?.2,[1]);
        }
        for bias in [[f32::MAX;4],[-f32::MAX;4]] { assert_eq!(route(b,&op,&[0.0;4],&bias,&[0])?.2,[1]); }
        // A zero mask would admit the excluded group in this negative-score row.
        assert_eq!(route(b,&op,&[0.0;4],&[-2.0,-2.0,-1.0,-1.0],&[0])?.0,[2,3]);
        println!("router rejection byte gate: non-finite logits/bias, group overflow, sticky status, and negative mask PASS");
        Ok(())
    }).unwrap();
}

fn finish(b: &Backend<'_>, ids: &[usize], weights: &[f32], rows: &[f32], shared: &[f32], flags: &[u32], h: usize) -> Result<Vec<u32>> {
    let (t,k) = (flags.len(),ids.len()/flags.len());
    assert_eq!(weights.len(),t*k); assert_eq!(rows.len(),t*k*h); assert_eq!(shared.len(),t*h);
    let ids = b.upload_vector(&ids.iter().map(|&id| f32::from_bits(id as u32)).collect::<Vec<_>>())?;
    let weights = b.upload_vector(weights)?;
    let rows = b.upload_vector(rows)?;
    let shared = b.upload_vector(shared)?;
    let flags = b.upload_vector(&flags.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>())?;
    let output = b.upload_vector(&vec![f32::from_bits(POISON);(t+2)*h])?;
    let p = UngatedMoeFinish { ids: ids.device_address(),weights: weights.device_address(),partials: rows.device_address(),
        shared: shared.device_address(),output: output.device_address(),status: flags.device_address(),
        selected: k as u32,hidden: h as u32,tokens: t as u32 }.checked().unwrap();
    super::gdn::launch_batch(b,"ungated_moe_finish",t*h,p)?;
    let actual = bits(&b.read_vector_values(&output)?);
    assert_eq!(&actual[t*h..],vec![POISON;2*h]);
    Ok(actual[..t*h].to_vec())
}

#[test]
#[ignore = "requires an authorized CUDA device; compare ungated finish bytes"]
fn ungated_finish_bytes_and_row_partitions() {
    with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),usize::MAX,|b| {
        for h in [128,384,4096] {
            let (t,k) = (3,3);
            let ids: Vec<_> = (0..t).flat_map(|_| [2,0,1]).collect();
            for weight in [[1.0;3],[1.003,0.5,1.0]] {
                let weights: Vec<_> = (0..t).flat_map(|_| weight).collect();
                let rows: Vec<_> = (0..t*k*h).map(|i| {
                    let slot = (i/h)%k;
                    match i%4 { 0 => [-256.0,256.0,1.0][slot],1 => [-16777216.0,16777216.0,1.0][slot],
                        2 => [1.5,2.5,-1.0][slot],_ => -0.0 }
                }).collect();
                let shared: Vec<_> = (0..t*h).map(|i| if i%4 == 3 { -0.0 } else { 3.0 }).collect();
                let expected: Vec<_> = (0..t).flat_map(|token| {
                    bits(&ungated_shared(&ids[token*k..(token+1)*k],&weights[token*k..(token+1)*k],
                        &rows[token*k*h..(token+1)*k*h],&shared[token*h..(token+1)*h],spark_gpu_probe::bf16::rounded).unwrap())
                }).collect();
                for chunk in [1,2,3] {
                    for _ in 0..3 {
                        let mut actual = Vec::new();
                        for start in (0..t).step_by(chunk) {
                            let end = (start+chunk).min(t);
                            actual.extend(finish(b,&ids[start*k..end*k],&weights[start*k..end*k],&rows[start*k*h..end*k*h],
                                &shared[start*h..end*h],&vec![0;end-start],h)?);
                        }
                        assert_eq!(actual,expected);
                    }
                }
            }
            // Rejected rows must not consume IDs, weights, partials, or shared.
            assert_eq!(finish(b,&[usize::MAX;3],&[f32::NAN;3],&vec![f32::NAN;k*h],&vec![f32::NAN;h],&[1],h)?,vec![0;h]);
            println!("ungated finish byte gate: hidden={h} rows={t} selected={k} PASS");
        }
        Ok(())
    }).unwrap();
}
