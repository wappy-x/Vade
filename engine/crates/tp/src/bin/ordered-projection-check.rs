//! Exact byte probe for isolated, checkpoint-backed logical FP32 projections.
use engine_model::{plan::layer::profile::Profile,safetensors::{TensorFile,Limits}};
use spark_tp::ordered_projection::PlainF32Projection;
use std::{fs,path::Path};
#[path="../ordered_projection/fixture.rs"] mod fixture;
use fixture::Result;
const CASES: [(&str,u64,&str);5]=[("q-small",8,"q"),("q-wide",260,"q"),("down-tail",260,"attention"),
    ("low-rank",8,"forget"),("replicated",8,"forget_down")];
#[cfg(target_os="linux")]
fn read(path: &Path) -> Result<Vec<f32>> {
    let b=fs::read(path)?;if b.len()%4!=0 {return Err("invalid FP32 file extent".into());}
    Ok(b.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect())
}
fn host(root: &Path) -> Result<()> {
    fs::create_dir(root)?;
    for (case,hidden,id) in CASES {
        let p=fixture::plan(hidden,1,0)?;let step=p.steps.iter().position(|s| s.id==id).ok_or("missing fixture projection")?;
        let w=&p.steps[step].weights[0];let (weights,input)=fixture::values(w.shape[0] as usize,w.shape[1] as usize);
        let file=TensorFile::from_file(fixture::write(&root.join(format!("{case}.safetensors")),&w.name,"F32",w.shape[0],w.shape[1],
            &fixture::bytes(&weights))?,Limits::default())?;
        let b=PlainF32Projection::bind(&p,step,&file,Profile::LogicalF32V1)?;
        let expected=b.shape().reference(&b.read()?,&input)?;
        fs::write(root.join(format!("{case}.input")),fixture::bytes(&input))?;
        fs::write(root.join(format!("{case}.expected")),fixture::bytes(&expected))?;
    }
    fs::write(root.join("COMPLETE"),"logical-f32-v1 synthetic operands; device_pass=false\n")?;
    println!("HOST_OPERANDS cases={} device_pass=false",CASES.len());Ok(())
}
#[cfg(target_os="linux")]
fn device(expected: &Path,out: &Path,parts: u32,rank: u32) -> Result<()> {
    use spark_gpu_probe::{driver::with_backend,ordered_projection::Error};
    if fs::read_to_string(expected.join("COMPLETE"))? != "logical-f32-v1 synthetic operands; device_pass=false\n" {
        return Err("expected host marker differs".into());
    }
    fs::create_dir(out)?;
    let ptx=spark_gpu_probe::layer_ptx()?;
    let result=with_backend(0,ptx,64*1024*1024,|backend| {
        for (case,hidden,id) in CASES {
            let run=|| -> Result<()> {
                let p=fixture::plan(hidden,parts,rank)?;
                let step=p.steps.iter().position(|s| s.id==id).ok_or("missing fixture projection")?;
                let file=TensorFile::open(&expected.join(format!("{case}.safetensors")),Limits::default())?;
                let b=PlainF32Projection::bind(&p,step,&file,Profile::LogicalF32V1)?;
                let matrix=b.load(backend)?;
                // The matrix view keeps its arena after the loader's local handle drops.
                let input=backend.upload_vector(&read(&expected.join(format!("{case}.input")))?)?;
                let mut output=backend.vector(b.shape().rows())?;
                backend.project_f32_ordered(&matrix,&input,&mut output)?;
                let bytes=fixture::bytes(&backend.read_vector_values(&output)?);
                fs::write(out.join(format!("{case}.actual")),&bytes)?;
                fs::write(out.join(format!("{case}.rows")),format!("{} {} {}\n",b.row_origin(),b.shape().rows(),p.steps[step].weights[0].shape[0]))?;
                let oracle=fs::read(expected.join(format!("{case}.expected")))?;
                let start=b.row_origin() as usize*4;let end=start+bytes.len();
                if oracle.get(start..end)!=Some(bytes.as_slice()) {return Err(format!("projection bytes differ: {case}").into());}
                // A wrong input length must fail before any launch.
                let wrong=backend.vector(b.shape().cols()+1)?;
                if backend.project_f32_ordered(&matrix,&wrong,&mut output)!=Err(Error::Extent) {
                    return Err("wrong input extent was not refused".into());
                }
                let bf=backend.weight_matrix(&backend.load_weight_arena(b.shape().weight_bytes(),&[] )?,0,
                    b.shape().rows(),b.shape().cols(),spark_gpu_probe::load::WeightDtype::Bf16)?;
                if backend.project_f32_ordered(&bf,&input,&mut output)!=Err(Error::UnsupportedStorage) {
                    return Err("BF16 storage was not refused".into());
                }
                if fixture::bytes(&backend.read_vector_values(&output)?)!=bytes {
                    return Err("refused projection changed the prior output".into());
                }
                println!("PROJECTION_BYTES case={case} origin={} rows={} cols={} bytes_equal=true",b.row_origin(),b.shape().rows(),b.shape().cols());
                Ok(())
            };
            run().map_err(|e| e.to_string())?;
        }
        Ok(())
    });
    result?;
    fs::write(out.join("COMPLETE"),marker(parts,rank))?;
    println!("PROJECTION_PRIMITIVE_PASS parts={parts} rank={rank} cases={} M2=false M3=false",CASES.len());Ok(())
}
fn marker(parts: u32,rank: u32) -> String {
    format!("parts={parts} rank={rank} cases={} primitive_only=true source_revision={}\n",CASES.len(),include_str!("../../../../REVISION").trim())
}
fn interval(hidden: u64,id: &str,parts: u32,rank: u32) -> Result<(u64,u64,u64)> {
    let p=fixture::plan(hidden,parts,rank)?;let w=&p.steps.iter().find(|s| s.id==id).ok_or("missing projection")?.weights[0];
    let (start,rows)=p.topology.interval(&w.shape,w.split)?.map_or((0,w.shape[0]),|(_,start,rows)| (start,rows));
    Ok((start,rows,w.shape[0]))
}
fn validate(root: &Path) -> Result<(u32,u32)> {
    let m=fs::read_to_string(root.join("COMPLETE"))?;let words=m.split_whitespace().collect::<Vec<_>>();
    let parts=words.first().and_then(|s| s.strip_prefix("parts=")).ok_or("missing parts")?.parse()?;
    let rank=words.get(1).and_then(|s| s.strip_prefix("rank=")).ok_or("missing rank")?.parse()?;
    if m!=marker(parts,rank) {return Err("invalid completion marker".into());}
    for (case,hidden,id) in CASES {
        let (start,rows,total)=interval(hidden,id,parts,rank)?;
        if fs::read_to_string(root.join(format!("{case}.rows")))?!=format!("{start} {rows} {total}\n")
            || fs::read(root.join(format!("{case}.actual")))?.len() as u64!=rows*4 {return Err("invalid saved operand extent".into());}
    }
    Ok((parts,rank))
}
fn compare(a: &Path,b: &Path) -> Result<()> {
    if validate(a)?!=validate(b)? {return Err("run markers differ".into());}
    for (case,_,_) in CASES {for ext in ["rows","actual"] {
        if fs::read(a.join(format!("{case}.{ext}")))?!=fs::read(b.join(format!("{case}.{ext}")))? {return Err(format!("run bytes differ: {case}.{ext}").into());}
    }}
    println!("PROJECTION_COMPARE cases={} bytes_equal=true primitive_only=true",CASES.len());Ok(())
}
fn compare_split(full: &Path,ranks: &[&Path]) -> Result<()> {
    if validate(full)?!=(1,0) || ranks.is_empty() {return Err("split compare requires a complete full run".into());}
    let parts=u32::try_from(ranks.len())?;
    for (rank,root) in ranks.iter().enumerate() {
        if validate(root)?!=(parts,rank as u32) {return Err("split ranks are missing, repeated, or out of order".into());}
    }
    for (case,hidden,id) in CASES {
        let whole=fs::read(full.join(format!("{case}.actual")))?;
        for (rank,root) in ranks.iter().enumerate() {
            let (start,rows,_)=interval(hidden,id,parts,rank as u32)?;
            let bytes=fs::read(root.join(format!("{case}.actual")))?;
            if whole.get(start as usize*4..(start+rows) as usize*4)!=Some(bytes.as_slice()) {
                return Err(format!("split bytes differ: {case}").into());
            }
        }
    }
    println!("PROJECTION_SPLIT_COMPARE parts={parts} cases={} bytes_equal=true primitive_only=true",CASES.len());Ok(())
}
fn run() -> Result<()> {
    println!("SOURCE_REVISION {}",include_str!("../../../../REVISION").trim());
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    match a.first().map(String::as_str) {
        Some("host") if a.len()==2=>host(Path::new(&a[1])),
        Some("compare") if a.len()==3=>compare(Path::new(&a[1]),Path::new(&a[2])),
        Some("compare-split") if a.len()>=3=>compare_split(Path::new(&a[1]),&a[2..].iter().map(Path::new).collect::<Vec<_>>()),
        #[cfg(target_os="linux")]
        Some("device") if a.len()==5=>device(Path::new(&a[1]),Path::new(&a[2]),a[3].parse()?,a[4].parse()?),
        _=>Err("usage: ordered-projection-check host NEW_EXPECTED | device EXPECTED NEW_OUT PARTS RANK | compare RUN_A RUN_B | compare-split FULL RANK_RUN...".into()),
    }
}
fn main() {if let Err(e)=run() {eprintln!("ordered-projection-check: {e}");std::process::exit(1);}}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf,sync::atomic::{AtomicU64,Ordering}};
    static NEXT: AtomicU64=AtomicU64::new(0);
    fn root() -> PathBuf {
        let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p=std::env::temp_dir().join(format!("vade-projection-compare-{}-{stamp}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&p).unwrap();p
    }
    // These markers are synthetic test fixtures. They are never device evidence.
    fn fake(parts: u32,rank: u32) -> PathBuf {
        let root=root();
        for (case,hidden,id) in CASES {
            let (start,rows,total)=interval(hidden,id,parts,rank).unwrap();
            fs::write(root.join(format!("{case}.rows")),format!("{start} {rows} {total}\n")).unwrap();
            fs::write(root.join(format!("{case}.actual")),fixture::bytes(&(start..start+rows).map(|v| v as f32).collect::<Vec<_>>())).unwrap();
        }
        fs::write(root.join("COMPLETE"),marker(parts,rank)).unwrap();root
    }
    #[test] fn compare_requires_complete_markers_and_exact_extents() {
        let a=fake(1,0);let b=fake(1,0);compare(&a,&b).unwrap();
        fs::write(b.join("q-small.actual"),[0]).unwrap();assert!(compare(&a,&b).is_err());
        assert!(compare(&a,&root()).is_err());
        fs::write(a.join("COMPLETE"),"partial").unwrap();assert!(validate(&a).is_err());
    }
    #[test] fn split_compare_requires_every_rank_and_keeps_replicated_rows() {
        let full=fake(1,0);let a=fake(2,0);let b=fake(2,1);
        compare_split(&full,&[&a,&b]).unwrap();
        assert!(compare_split(&full,&[&a,&a]).is_err());
        assert!(compare_split(&full,&[&a]).is_err());
        fs::write(b.join("replicated.actual"),fixture::bytes(&[0.0,1.0,3.0])).unwrap();
        assert!(compare_split(&full,&[&a,&b]).is_err());
    }
}
