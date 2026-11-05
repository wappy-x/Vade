//! Synthetic retained-checkpoint operand gate. This is not a full layer gate.
#![forbid(unsafe_code)]
use engine_model::plan::{admission::CheckpointAdmission,layer::{Kind,Topology,LayerPlan},state::StatePlan};
use spark_tp::{Result,layer_driver::LayerBoundary};
use spark_gpu_probe::{norm_boundary::NormShape,mhc_boundary};
use std::{collections::BTreeMap,fs,path::Path};
#[path="resident-layer-check/fixture.rs"] mod fixture;
#[allow(dead_code)]
#[path="../../../../crates/gpu/kernels/src/qsa_common.rs"] mod qsa_common;
#[allow(dead_code)]
#[path="../../../../crates/gpu/src/mhc_params.rs"] mod mhc_params;
#[allow(dead_code)]
#[path="../../../../crates/gpu/kernels/src/mhc_math.rs"] mod mhc_math;

fn write(path: &Path,bytes: &[u8]) -> Result<()> {fs::write(path,bytes).map_err(|e| e.to_string())}
fn bytes(v: &[f32]) -> Vec<u8> {v.iter().flat_map(|v| v.to_le_bytes()).collect()}
fn equal(name: &str,actual: &[u8],expected: &[u8]) -> Result<()> {
    if actual!=expected {return Err(format!("{name}: bytes differ actual={} expected={} first={:?}",actual.len(),expected.len(),actual.iter().zip(expected).position(|(a,b)| a!=b)));}Ok(())
}
fn sum(mut lanes: [f32;32]) -> f32 {
    for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32 {lanes[lane]=old[lane]+old[lane^distance];}}
    lanes[0]
}
fn norm(shape: NormShape,x: &[f32],weight: &[f32]) -> Vec<f32> {
    let mut partials=Vec::new();
    for chunk in x.chunks(128) {let mut lanes=[0.0;32];for lane in 0..32 {
        let mut i=lane;while i<chunk.len() {lanes[lane]+=chunk[i]*chunk[i];i+=32;}
    }partials.push(sum(lanes));}
    let mut lanes=[0.0;32];for lane in 0..32 {let mut i=lane;while i<partials.len() {lanes[lane]+=partials[i];i+=32;}}
    let scale=1.0/(sum(lanes)/shape.width() as f32+shape.epsilon()).sqrt();
    x.iter().zip(weight).map(|(&x,&w)| shape.weight(x*scale,w)).collect()
}
fn norm_cases() -> Vec<NormShape> {
    use spark_gpu_probe::norm_boundary::WeightType;
    [1,31,129,4097].into_iter().flat_map(|width| [WeightType::Bf16,WeightType::F32]
        .map(move |dtype| NormShape::new(width,1e-5,dtype).unwrap())).collect()
}
fn norm_weight(s: NormShape) -> Vec<f32> {
    let dtype=match s.weight_type() {spark_gpu_probe::norm_boundary::WeightType::Bf16=>engine_model::safetensors::Dtype::Bf16,
        spark_gpu_probe::norm_boundary::WeightType::F32=>engine_model::safetensors::Dtype::F32};
    fixture::values("input_layernorm.weight",s.width(),dtype)
}
fn reference(checkpoint: &CheckpointAdmission) -> Result<BTreeMap<String,Vec<f32>>> {
    let d=fixture::description();let t=Topology {parts:1,rank:0};let state=StatePlan::new(&d,t,4,isize::MAX as usize).map_err(|e| e.to_string())?;
    let mut out=BTreeMap::new();
    for layer in 0..d.layers.len() {let boundary=LayerBoundary::new(&d,checkpoint.headers(),&state,layer,3,t,4096)?;
        for (i,op) in boundary.program().steps.iter().enumerate().filter(|(_,s)| s.kind==Kind::MhcPrepare) {
            let p=boundary.prepare_residual_block_checkpoint(i,checkpoint)?;let m=p.mhc();let s=m.shape();
            let input=fixture::data(s.wide(),layer+i);let block=fixture::data(s.hidden as usize,layer+i+11);
            let mut normalized=vec![0.0;s.wide()];let mut coefficients=vec![0.0;s.coefficients()];
            mhc_math::prepare(mhc_params::Shape {rows:1,streams:s.streams,hidden:s.hidden,iterations:s.iterations,rms_eps:s.rms_eps,eps:s.eps},
                &input,m.weights(),m.base(),m.scale(),&mut normalized,&mut coefficients);
            let collapsed=(0..s.hidden as usize).map(|j| mhc_boundary::collapse(&input,&coefficients,s.streams as usize,s.hidden as usize,j)).collect::<Vec<_>>();
            let n=norm(p.norm().shape(),&collapsed,p.norm().weight());
            if op.id=="attn.prepare" {
                // Sparse copy matrices give an independent exact projection oracle.
                // They avoid an assumed tensor-core reduction tree for arbitrary weights.
                let q=boundary.program().steps.iter().position(|s| s.kind==Kind::Project).unwrap();
                let operands=boundary.prepare_bf16_projection_checkpoint(q,checkpoint)?;
                let expected=(0..operands.shape()[0]).map(|r| n[r%n.len()]).collect();
                out.insert(format!("{layer}-projection.f32"),expected);
            }
            out.insert(format!("{layer}-{}-norm.f32",op.id),n);
            let combined=(0..s.wide()).map(|j| mhc_boundary::combine(&input,&coefficients,block[j%s.hidden as usize],
                s.streams as usize,s.hidden as usize,j/s.hidden as usize,j%s.hidden as usize)).collect();
            out.insert(format!("{layer}-{}-combined.f32",op.id),combined);
        }
    }
    for (i,s) in norm_cases().into_iter().enumerate() {
        out.insert(format!("norm-{i}.f32"),norm(s,&fixture::data(s.width(),7),&norm_weight(s)));
    }Ok(out)
}
fn host(out: &Path) -> Result<()> {
    let d=fixture::description();let root=out.join("checkpoint");let index=fixture::checkpoint(&root,&d)?;
    let checkpoint=d.open_headers(&root,&index).map_err(|e| e.to_string())?;let expected=reference(&checkpoint)?;
    for (name,values) in &expected {write(&out.join(name),&bytes(values))?;}
    write(&out.join("HOST-COMPLETE"),b"RESIDENT-LAYER-OPERANDS HOST PASS\n")?;
    println!("RESIDENT-LAYER-OPERANDS HOST PASS buffers={} full_layer=false",expected.len());Ok(())
}
#[cfg(target_os="linux")]
fn device(expected: &Path,out: &Path,t: Topology) -> Result<()> {
    use spark_gpu_probe::{driver::{with_backend,NormOutput},layer_ptx};
    t.validate().map_err(|e| e.to_string())?;
    equal("host marker",&fs::read(expected.join("HOST-COMPLETE")).map_err(|e| e.to_string())?,b"RESIDENT-LAYER-OPERANDS HOST PASS\n")?;
    let d=fixture::description();let root=expected.join("checkpoint");
    let index=fs::read(root.join("model.safetensors.index.json")).map_err(|e| e.to_string())?;
    let checkpoint=d.open_headers(&root,&index).map_err(|e| e.to_string())?;
    // Recompute from the shipped checkpoint, then require the saved host bytes.
    // A changed checkpoint cannot silently select a different expected result.
    let oracle=reference(&checkpoint)?;
    for (name,values) in &oracle {equal(name,&fs::read(expected.join(name)).map_err(|e| e.to_string())?,&bytes(values))?;}
    let state=StatePlan::new(&d,t,4,isize::MAX as usize).map_err(|e| e.to_string())?;
    with_backend(0,layer_ptx()?,usize::MAX,|backend| {
        let mut count=0;
        for layer in 0..d.layers.len() {let boundary=LayerBoundary::new(&d,checkpoint.headers(),&state,layer,3,t,4096)?;
            if boundary.require_executable().is_ok() {return Err("partial operand gate admitted a full layer".into());}
            for (i,op) in boundary.program().steps.iter().enumerate().filter(|(_,s)| s.kind==Kind::MhcPrepare) {
                let p=boundary.prepare_residual_block_checkpoint(i,&checkpoint)?;let s=p.mhc().shape();let resident=p.upload(backend)?;
                let pass=resident.begin(backend.upload_bf16_lanes(&fixture::data(s.wide(),layer+i))?)?;
                let name=format!("{layer}-{}-norm.f32",op.id);
                let actual=backend.read_vector_values(pass.normalized().vector())?;
                write(&out.join(&name),&bytes(&actual))?;equal(&name,&bytes(&actual),&bytes(&oracle[&name]))?;count+=1;
                if op.id=="attn.prepare" {
                    let NormOutput::Bf16(input)=pass.normalized() else {return Err("fixture projection requires BF16 norm output".into());};
                    let q=boundary.program().steps.iter().position(|s| s.kind==Kind::Project).unwrap();
                    let p=boundary.prepare_bf16_projection_checkpoint(q,&checkpoint)?;
                    let origin=p.row_origin() as usize;let rows=p.shape()[0];let projection=p.upload(backend)?;
                    let actual=projection.apply(input)?;let actual=backend.read_vector_values(actual.vector())?;
                    let name=format!("{layer}-projection.f32");write(&out.join(&name),&bytes(&actual))?;
                    equal(&name,&bytes(&actual),&bytes(&oracle[&name][origin..origin+rows]))?;count+=1;
                }
                let combined=pass.finish(backend.upload_bf16_lanes(&fixture::data(s.hidden as usize,layer+i+11))?)?;
                let actual=backend.read_vector_values(combined.vector())?;let name=format!("{layer}-{}-combined.f32",op.id);
                write(&out.join(&name),&bytes(&actual))?;equal(&name,&bytes(&actual),&bytes(&oracle[&name]))?;count+=1;
            }
        }
        for (i,s) in norm_cases().into_iter().enumerate() {
            let resident=backend.resident_norm(s,&norm_weight(s))?;
            let input=backend.upload_bf16_lanes(&fixture::data(s.width(),7))?;
            let actual=resident.apply(&input)?;let actual=backend.read_vector_values(actual.vector())?;
            let name=format!("norm-{i}.f32");write(&out.join(&name),&bytes(&actual))?;
            equal(&name,&bytes(&actual),&bytes(&oracle[&name]))?;count+=1;
        }
        // Reject a wrong context and a wrong input extent before any norm launch.
        let shape=spark_gpu_probe::norm_boundary::NormShape::new(1,1e-5,spark_gpu_probe::norm_boundary::WeightType::F32)?;
        let resident=backend.resident_norm(shape,&[1.0000009536743164])?;
        if resident.apply(&backend.upload_bf16_lanes(&[1.0,1.0])?).is_ok() {return Err("norm accepted a wrong extent".into());}
        with_backend(0,layer_ptx()?,usize::MAX,|other| {
            if resident.apply(&other.upload_bf16_lanes(&[1.0])?).is_ok() {return Err("norm accepted a foreign context".into());}Ok(())
        })?;
        write(&out.join("TOPOLOGY"),format!("{} {}\n",t.rank,t.parts).as_bytes())?;
        write(&out.join("COMPLETE"),b"RESIDENT-LAYER-OPERANDS DEVICE PASS\n")?;
        println!("RESIDENT-LAYER-OPERANDS DEVICE PASS rank={} parts={} buffers={count} full_layer=false",t.rank,t.parts);Ok(())
    })
}
#[cfg(not(target_os="linux"))]
fn device(_: &Path,_: &Path,_: Topology) -> Result<()> {Err("resident device gate requires Linux with CUDA".into())}
fn topology(path: &Path) -> Result<Topology> {
    let text=fs::read_to_string(path.join("TOPOLOGY")).map_err(|e| e.to_string())?;
    let parts=text.split_whitespace().map(str::parse::<u32>).collect::<std::result::Result<Vec<_>,_>>().map_err(|e| e.to_string())?;
    if parts.len()!=2 {return Err("invalid saved operand topology".into());}
    let t=Topology {rank:parts[0],parts:parts[1]};t.validate().map_err(|e| e.to_string())?;Ok(t)
}
fn projection_rows(layer: usize,t: Topology) -> Result<(usize,usize,usize)> {
    let plan=LayerPlan::new(&fixture::description(),layer,3,t).map_err(|e| e.to_string())?;
    let w=&plan.steps.iter().find(|s| s.kind==Kind::Project).unwrap().weights[0];
    let (first,rows)=t.interval(&w.shape,w.split).map_err(|e| e.to_string())?.map_or((0,w.shape[0]),|(_,a,n)| (a,n));
    Ok((first as usize,rows as usize,w.shape[0] as usize))
}
fn compare(a: &Path,b: &Path) -> Result<()> {
    for p in [a,b] {equal("device marker",&fs::read(p.join("COMPLETE")).map_err(|e| e.to_string())?,b"RESIDENT-LAYER-OPERANDS DEVICE PASS\n")?;}
    let d=fixture::description();let mut count=0;let ta=topology(a)?;let tb=topology(b)?;
    // Prefixes are replicated. Projection slices are checked against the full
    // saved oracle on each rank; different rank slices must not compare equal.
    for layer in 0..d.layers.len() {for side in ["attn","ffn"] {for plane in ["norm","combined"] {
        let name=format!("{layer}-{side}.prepare-{plane}.f32");
        let left=fs::read(a.join(&name)).map_err(|e| e.to_string())?;let right=fs::read(b.join(&name)).map_err(|e| e.to_string())?;
        let words=if plane=="norm" {d.hidden} else {d.hidden*d.residual.streams};
        if left.len()!=words as usize*4 || right.len()!=words as usize*4 {return Err("operand comparison extent differs".into());}
        equal(&name,&left,&right)?;count+=1;
    }}}
    for (i,s) in norm_cases().into_iter().enumerate() {
        let name=format!("norm-{i}.f32");let left=fs::read(a.join(&name)).map_err(|e| e.to_string())?;
        let right=fs::read(b.join(&name)).map_err(|e| e.to_string())?;
        if left.len()!=s.width()*4 || right.len()!=s.width()*4 {return Err("norm comparison extent differs".into());}
        equal(&name,&left,&right)?;count+=1;
    }
    if ta==tb {
        for layer in 0..d.layers.len() {
            let (_,rows,_)=projection_rows(layer,ta)?;
            let name=format!("{layer}-projection.f32");let left=fs::read(a.join(&name)).map_err(|e| e.to_string())?;
            let right=fs::read(b.join(&name)).map_err(|e| e.to_string())?;
            if left.len()!=rows*4 || right.len()!=rows*4 {return Err("projection comparison extent differs".into());}
            equal(&name,&left,&right)?;count+=1;
        }
    }
    println!("RESIDENT-LAYER-OPERANDS COMPARE PASS buffers={count} full_layer=false");Ok(())
}
fn compare_split(full: &Path,shards: &[&str]) -> Result<()> {
    if shards.is_empty() {return Err("operand split comparison has no ranks".into());}
    let mut ranks=BTreeMap::new();
    for path in shards {let path=Path::new(path);let t=topology(path)?;
        if t.parts as usize!=shards.len() || ranks.insert(t.rank,path).is_some() {return Err("operand split ranks differ or repeat".into());}
        compare(full,path)?;
    }
    let d=fixture::description();let t=topology(full)?;
    for layer in 0..d.layers.len() {
        let (_,_,rows)=projection_rows(layer,t)?;
        let name=format!("{layer}-projection.f32");let full=fs::read(full.join(&name)).map_err(|e| e.to_string())?;
        let mut joined=Vec::new();
        for (rank,path) in &ranks {let part=fs::read(path.join(&name)).map_err(|e| e.to_string())?;
            let (first,local,_)=projection_rows(layer,topology(path)?)?;
            if part.len()!=local*4 {return Err(format!("rank {rank}: projection split extent differs"));}
            if local==rows {equal(&name,&part,&full)?;joined=part;}
            else {if first*4!=joined.len() {return Err("projection rank intervals have a gap or overlap".into());}joined.extend(part);}
        }
        if full.len()!=rows*4 {return Err("full projection extent differs".into());}
        equal(&name,&joined,&full)?;
    }
    println!("RESIDENT-LAYER-OPERANDS SPLIT PASS ranks={} full_layer=false",ranks.len());Ok(())
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let result=(|| {match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["host",out]=>{fs::create_dir(out).map_err(|e| e.to_string())?;host(Path::new(out))},
        ["device",expected,out,rank,parts]=>{
            let t=Topology {rank:rank.parse().map_err(|_| "invalid rank")?,parts:parts.parse().map_err(|_| "invalid rank count")?};
            t.validate().map_err(|e| e.to_string())?;fs::create_dir(out).map_err(|e| e.to_string())?;device(Path::new(expected),Path::new(out),t)
        },
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),
        ["compare-split",full,shards @ ..]=>compare_split(Path::new(full),shards),
        _=>Err("usage: resident-layer-check host NEW_OUT | device EXPECTED NEW_OUT RANK PARTS | compare RUN_A RUN_B | compare-split FULL RANK_RUN...".into()),
    }})();
    if let Err(e)=result {eprintln!("resident-layer-check: {e}");std::process::exit(1);}
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {fn new() -> Self {
        static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let path=std::env::temp_dir().join(format!("vade-resident-layer-check-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
        fs::create_dir(&path).unwrap();Self(path)
    }}
    impl Drop for Temp {fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();}}
    fn saved_run(root: &Path,oracle: &BTreeMap<String,Vec<f32>>,t: Topology) {
        fs::create_dir(root).unwrap();
        for (name,v) in oracle {
            let values=if name.ends_with("-projection.f32") {let layer=name.split('-').next().unwrap().parse().unwrap();
                let (first,rows,_)=projection_rows(layer,t).unwrap();&v[first..first+rows]} else {v.as_slice()};
            write(&root.join(name),&bytes(values)).unwrap();
        }
        write(&root.join("TOPOLOGY"),format!("{} {}\n",t.rank,t.parts).as_bytes()).unwrap();
        // Comparator fixture only. No device function runs in this host test.
        write(&root.join("COMPLETE"),b"RESIDENT-LAYER-OPERANDS DEVICE PASS\n").unwrap();
    }
    #[test]
    fn normalized_cast_precedes_weight_multiply_in_the_oracle() {
        let shape=NormShape::new(1,1.0,spark_gpu_probe::norm_boundary::WeightType::F32).unwrap();
        let expected=spark_gpu_probe::bf16::rounded(1.0/2.0f32.sqrt())*1.0000009536743164;
        assert_eq!(norm(shape,&[1.0],&[1.0000009536743164]),[expected]);
    }
    #[test]
    fn byte_comparator_rejects_missing_markers_bad_extents_and_wrong_rank_sets() {
        let temp=Temp::new();let d=fixture::description();let root=temp.0.join("checkpoint");
        let index=fixture::checkpoint(&root,&d).unwrap();let checkpoint=d.open_headers(&root,&index).unwrap();
        let oracle=reference(&checkpoint).unwrap();let full=temp.0.join("full");let a=temp.0.join("a");let b=temp.0.join("b");
        saved_run(&full,&oracle,Topology {rank:0,parts:1});saved_run(&a,&oracle,Topology {rank:0,parts:2});saved_run(&b,&oracle,Topology {rank:1,parts:2});
        let shards=[b.to_str().unwrap(),a.to_str().unwrap()];compare_split(&full,&shards).unwrap();compare(&a,&a).unwrap();
        assert!(compare_split(&full,&[]).is_err());assert!(compare_split(&full,&[shards[0]]).is_err());
        assert!(compare_split(&full,&[shards[0],shards[0]]).is_err());
        write(&a.join("0-projection.f32"),&[]).unwrap();assert!(compare(&a,&a).is_err());assert!(compare_split(&full,&shards).is_err());
        write(&a.join("0-projection.f32"),&bytes(&oracle["0-projection.f32"][..32])).unwrap();
        write(&a.join("norm-0.f32"),&[]).unwrap();assert!(compare(&a,&a).is_err());
        write(&a.join("norm-0.f32"),&bytes(&oracle["norm-0.f32"])).unwrap();
        write(&a.join("COMPLETE"),b"FAILED\n").unwrap();assert!(compare(&a,&a).is_err());
        write(&a.join("TOPOLOGY"),b"bad\n").unwrap();assert!(topology(&a).is_err());
    }
}
