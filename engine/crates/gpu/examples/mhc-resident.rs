//! Shipped resident mHC byte probe. No weights or model admission are involved.
#![forbid(unsafe_code)]
use spark_gpu_probe::{mhc_params::Shape,mhc_boundary as boundary};
use std::{fs,path::Path};
#[allow(dead_code)]
#[path = "../kernels/src/qsa_common.rs"] mod qsa_common;
#[allow(dead_code)]
#[path = "../src/mhc_params.rs"] mod mhc_params;
#[allow(dead_code)]
#[path = "../kernels/src/mhc_math.rs"] mod mhc_math;

type Result<T>=std::result::Result<T,String>;
const LABELS: [&str;4]=["normalized","coefficients","collapsed","combined"];
fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| ((i*13+i/7+salt)%31) as f32/16.0-0.9375).collect()
}
fn cases() -> Vec<Shape> {
    // Small odd widths and the saved mHC dimensions. These are fixture shapes.
    [(1,1,1,1),(2,3,3,2),(3,7,5,3),(4,129,3,5),(4,4096,20,2)]
        .map(|(streams,hidden,iterations,rows)| Shape {rows,streams,hidden,iterations,rms_eps:1e-6,eps:1e-6}).to_vec()
}
struct Fixture {shape:Shape,input:Vec<f32>,weights:Vec<f32>,base:Vec<f32>,scale:Vec<f32>,block:Vec<f32>}
impl Fixture {
    fn new(shape: Shape) -> Self {
        Self {shape,input:data(shape.rows as usize*shape.wide(),3),
            weights:data(shape.coefficients()*shape.wide(),11).into_iter().map(|v| v/64.0).collect(),
            base:data(shape.coefficients(),7),scale:vec![0.25,0.5,0.125],block:data(shape.rows as usize*shape.hidden as usize,23)}
    }
    fn expected(&self) -> Vec<Vec<f32>> {
        let s=self.shape;let (t,w,p,n,h)=(s.rows as usize,s.wide(),s.coefficients(),s.streams as usize,s.hidden as usize);
        let mut normalized=vec![0.0;t*w];let mut coefficients=vec![0.0;t*p];let mut collapsed=vec![0.0;t*h];let mut combined=vec![0.0;t*w];
        for row in 0..t {
            let x=&self.input[row*w..(row+1)*w];let c=&mut coefficients[row*p..(row+1)*p];
            mhc_math::prepare(mhc_params::Shape {rows:1,streams:s.streams,hidden:s.hidden,iterations:s.iterations,rms_eps:s.rms_eps,eps:s.eps},
                x,&self.weights,&self.base,&self.scale,&mut normalized[row*w..(row+1)*w],c);
            for d in 0..h {
                collapsed[row*h+d]=boundary::collapse(x,c,n,h,d);
                for dst in 0..n {combined[row*w+dst*h+d]=boundary::combine(x,c,self.block[row*h+d],n,h,dst,d);}
            }
        }
        vec![normalized,coefficients,collapsed,combined]
    }
}
fn bytes(v: &[f32]) -> Vec<u8> {v.iter().flat_map(|x| x.to_le_bytes()).collect()}
fn write(path: &Path, value: &[u8]) -> Result<()> {fs::write(path,value).map_err(|e| format!("{}: {e}",path.display()))}
fn equal(label: &str, actual: &[u8], expected: &[u8]) -> Result<()> {
    if actual!=expected {return Err(format!("{label}: bytes differ actual={} expected={} first={:?}",actual.len(),expected.len(),actual.iter().zip(expected).position(|(a,b)| a!=b)));}Ok(())
}
fn host(out: &Path) -> Result<()> {
    for (index,s) in cases().into_iter().enumerate() {
        s.validate()?;let f=Fixture::new(s);let values=f.expected();
        for (label,value) in LABELS.iter().zip(values) {write(&out.join(format!("{index}-{label}.f32")),&bytes(&value))?;}
    }
    write(&out.join("SOURCE"),b"contract=synthetic resident mHC; fixed FP32 prepare and BF16 collapse/combine; not a checkpoint oracle\n")?;
    println!("MHC-RESIDENT HOST PASS cases={}",cases().len());Ok(())
}
#[cfg(target_os = "linux")]
fn device(expected: &Path, out: &Path) -> Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    // Check all expected files before opening CUDA. No zero-case pass is possible.
    let mut saved=Vec::new();
    for (index,s) in cases().into_iter().enumerate() {
        let counts=[s.rows as usize*s.wide(),s.rows as usize*s.coefficients(),s.rows as usize*s.hidden as usize,s.rows as usize*s.wide()];
        for (label,count) in LABELS.iter().zip(counts) {
            let path=expected.join(format!("{index}-{label}.f32"));let v=fs::read(&path).map_err(|e| format!("{}: {e}",path.display()))?;
            if v.len()!=count*4 {return Err("saved mHC expectation extent differs".into());}saved.push(v);
        }
    }
    with_backend(0,layer_ptx()?,usize::MAX,|b| {
        let source=b.upload_vector(&[0.0,-0.0,1.00390625,1.01171875,-1.00390625])?;
        let rounded=b.round_bf16_lanes(&source)?;
        equal("resident BF16 ties",&bytes(&b.read_vector_values(rounded.vector())?),&bytes(&[0.0,-0.0,1.0,1.015625,-1.0]))?;
        with_backend(0,layer_ptx()?,usize::MAX,|other| {
            let foreign=other.upload_vector(&[0.0])?;
            if b.round_bf16_lanes(&foreign).is_ok() {return Err("foreign context accepted".into());}Ok(())
        })?;
        for (index,s) in cases().into_iter().enumerate() {
            let f=Fixture::new(s);let resident=b.resident_mhc(s,&f.weights,&f.base,&f.scale)?;
            let run=|first:usize,rows:usize| -> Result<Vec<Vec<f32>>> {
                let mut shape=s;shape.rows=rows as u32;
                let resident=b.resident_mhc(shape,&f.weights,&f.base,&f.scale)?;
                let input=b.upload_bf16_lanes(&f.input[first*s.wide()..(first+rows)*s.wide()])?;
                let pass=resident.begin(input)?;
                let normalized=b.read_vector_values(pass.normalized())?;let coefficients=b.read_vector_values(pass.coefficients())?;
                let collapsed=b.read_vector_values(pass.collapsed().vector())?;
                let block=b.upload_bf16_lanes(&f.block[first*s.hidden as usize..(first+rows)*s.hidden as usize])?;
                let combined=pass.finish(block)?;let combined=b.read_vector_values(combined.vector())?;
                Ok(vec![normalized,coefficients,collapsed,combined])
            };
            // Reject bad input shape and lossy BF16 uploads before kernels run.
            if b.upload_bf16_lanes(&[1.00390625]).is_ok() {return Err("lossy BF16 upload accepted".into());}
            if s.rows as usize*s.wide()>1 && resident.begin(b.upload_bf16_lanes(&[0.0])?).is_ok() {return Err("short residual accepted".into());}
            let actual=run(0,s.rows as usize)?;
            // Save actual buffers before comparison. Failure must preserve evidence.
            for (label,value) in LABELS.iter().zip(&actual) {
                write(&out.join(format!("{index}-{label}.f32")),&bytes(value))?;
            }
            for (j,(label,value)) in LABELS.iter().zip(&actual).enumerate() {
                equal(&format!("case {index} {label}"),&bytes(value),&saved[index*4+j])?;
            }
            // Reverse row execution checks isolation from batching and row order.
            for row in (0..s.rows as usize).rev() {
                let solo=run(row,1)?;
                for (label,value) in LABELS.iter().zip(&solo) {
                    write(&out.join(format!("{index}-row-{row}-{label}.f32")),&bytes(value))?;
                }
                for (j,value) in solo.iter().enumerate() {
                    let stride=actual[j].len()/s.rows as usize;
                    equal("solo versus batched",&bytes(value),&bytes(&actual[j][row*stride..(row+1)*stride]))?;
                }
            }
            println!("MHC-RESIDENT BYTES case={index} streams={} hidden={} rows={} iterations={} PASS",s.streams,s.hidden,s.rows,s.iterations);
        }
        if let Some(error)=b.fatal_error() {return Err(error);}Ok(())
    })?;
    write(&out.join("COMPLETE"),b"MHC-RESIDENT DEVICE PASS\n")?;
    println!("MHC-RESIDENT DEVICE PASS cases={}; synthetic primitive and cast binding only",cases().len());Ok(())
}
#[cfg(not(target_os = "linux"))]
fn device(_: &Path,_: &Path) -> Result<()> {Err("device probe requires Linux and CUDA".into())}
fn compare(a: &Path,b: &Path) -> Result<()> {
    for path in [a,b] {
        let mark=fs::read(path.join("COMPLETE")).map_err(|e| e.to_string())?;
        equal("completion marker",&mark,b"MHC-RESIDENT DEVICE PASS\n")?;
    }
    for (index,s) in cases().iter().enumerate() {
        let counts=[s.rows as usize*s.wide(),s.rows as usize*s.coefficients(),s.rows as usize*s.hidden as usize,s.rows as usize*s.wide()];
        for (label,count) in LABELS.iter().zip(counts) {
        let name=format!("{index}-{label}.f32");
        let left=fs::read(a.join(&name)).map_err(|e| e.to_string())?;
        let right=fs::read(b.join(&name)).map_err(|e| e.to_string())?;
        if left.len()!=count*4 || right.len()!=count*4 {return Err(format!("{name}: comparison extent differs"));}
        equal(&name,&left,&right)?;
    }}
    println!("MHC-RESIDENT COMPARE PASS cases={}",cases().len());Ok(())
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let result=(|| {
        match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
            ["host",out]=>{fs::create_dir(out).map_err(|e| e.to_string())?;host(Path::new(out))},
            ["device",expected,out]=>{fs::create_dir(out).map_err(|e| e.to_string())?;device(Path::new(expected),Path::new(out))},
            ["compare",a,b]=>compare(Path::new(a),Path::new(b)),
            _=>Err("usage: mhc-resident host NEW_OUT | device EXPECTED NEW_OUT | compare RUN_A RUN_B".into()),
        }
    })();
    if let Err(error)=result {eprintln!("mhc-resident: {error}");std::process::exit(1);}
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let path=std::env::temp_dir().join(format!("vade-mhc-resident-{}-{nonce}",std::process::id()));
            fs::create_dir(&path).unwrap();Self(path)
        }
    }
    impl Drop for Temp {fn drop(&mut self) {let _=fs::remove_dir_all(&self.0);}}
    #[test]
    fn comparator_rejects_failed_empty_truncated_and_different_runs() {
        let a=Temp::new();let b=Temp::new();host(&a.0).unwrap();host(&b.0).unwrap();
        assert!(compare(&a.0,&b.0).is_err());
        for path in [&a.0,&b.0] {write(&path.join("COMPLETE"),b"MHC-RESIDENT DEVICE PASS\n").unwrap();}
        compare(&a.0,&b.0).unwrap();
        let name="0-collapsed.f32";
        let expected=fs::read(a.0.join(name)).unwrap();
        for path in [&a.0,&b.0] {write(&path.join(name),&[]).unwrap();}
        assert!(compare(&a.0,&b.0).unwrap_err().contains("extent"));
        for path in [&a.0,&b.0] {write(&path.join(name),&expected).unwrap();}
        let mut changed=expected.clone();changed[0]^=1;write(&b.0.join(name),&changed).unwrap();
        assert!(compare(&a.0,&b.0).unwrap_err().contains("bytes differ"));
    }
}
