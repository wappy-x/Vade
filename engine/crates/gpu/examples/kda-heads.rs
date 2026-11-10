//! Diagnostic only: complete KDA heads on one device. No model or transport gate.
#![forbid(unsafe_code)]
#[cfg(target_os="linux")]
mod device {
    use spark_gpu_probe::{causal_ops::{Shape,Op},driver::{with_backend,Backend,DeviceVector},layer_ptx};
    type Result<T>=std::result::Result<T,String>;
    fn pattern(n:usize,salt:usize)->Vec<f32> {(0..n).map(|i|((i*13+salt*17)%61) as f32/128.0-0.234375).collect()}
    fn half(x:&[f32],groups:usize,rank:usize)->Vec<f32> {
        let width=x.len()/groups;assert_eq!(width%2,0);
        x.chunks_exact(width).flat_map(|row|row[rank*width/2..(rank+1)*width/2].iter().copied()).collect()
    }
    fn crop(x:&[f32],op:Op,plane:usize,rank:usize)->Vec<f32> {
        match (op,plane) {
            (Op::ConvParts,4)=>half(x,3,rank),
            (Op::ConvParts,3)=>half(x,9,rank),
            (Op::ConvParts,5)=>half(x,12,rank),
            (Op::Prepare,0)=>half(x,3,rank),
            (Op::Prepare,4)=>{let mut v=half(&x[..64],1,rank);v.extend(half(&x[64..],1,rank));v},
            (Op::GatedNorm,4)=>x.to_vec(),
            (Op::StepPlanar,5)=>{let mut v=half(&x[..64*128*128],1,rank);v.extend(half(&x[64*128*128..],1,rank));v},
            _=>half(x,1,rank),
        }
    }
    struct Case<'d> {shape:Shape,inputs:Vec<Option<DeviceVector<'d>>>,weight:Option<DeviceVector<'d>>,output:DeviceVector<'d>}
    fn make<'d>(b:&Backend<'d>,shape:Shape,raw:&[Vec<f32>],weight:&[f32])->Result<Case<'d>> {
        let (_,_,units,words)=shape.lengths().ok_or("invalid diagnostic shape")?;
        Ok(Case {shape,inputs:raw.iter().map(|x|if x.is_empty() {Ok(None)} else {b.upload_vector(x).map(Some)}).collect::<Result<_>>()?,weight:if weight.is_empty() {None} else {Some(b.upload_vector(weight)?)},output:b.vector(units*words)?})
    }
    fn run(b:&Backend<'_>,c:&mut Case<'_>)->Result<()> {
        b.resident_causal_into(c.shape,std::array::from_fn(|i|c.inputs.get(i).and_then(Option::as_ref)),c.weight.as_ref(),&mut c.output)
    }
    fn median(v:&mut[u128])->u128 {v.sort_unstable();v[v.len()/2]}
    pub fn main()->Result<()> {
        let image=std::env::args().nth(1).map(|path|std::fs::read(path).map_err(|e|e.to_string()).and_then(|bytes|std::ffi::CString::new(bytes).map_err(|e|e.to_string()))).transpose()?;
        let out=std::env::args().nth(2).ok_or("usage: kda-heads PTX NEW_OUTPUT_DIR")?;std::fs::create_dir(&out).map_err(|e|e.to_string())?;
        with_backend(0,image.as_deref().unwrap_or(layer_ptx()?),usize::MAX,|b| {
            b.configure_vector_pool(1<<30);let queue=b.resident_queue()?;b.enable_resident_launches()?;
            for op in [Op::ConvParts,Op::Prepare,Op::StepPlanar,Op::GatedNorm] {
                let shape=Shape {op,heads:64,dim:128,taps:4,rows:0,cols:0,epsilon:1e-5,bound:-5.0};
                let (lengths,weights,_,_)=shape.lengths().ok_or("invalid full shape")?;
                let raw=lengths.iter().enumerate().map(|(i,&n)|pattern(n,i)).collect::<Vec<_>>();let weight=pattern(weights,7);
                let mut full=make(b,shape,&raw,&weight)?;run(b,&mut full)?;let reference=b.read_vector_values(&full.output)?;
                let bytes=reference.iter().flat_map(|x|x.to_bits().to_le_bytes()).collect::<Vec<_>>();
                std::fs::write(std::path::Path::new(&out).join(format!("{op:?}.f32")),bytes).map_err(|e|e.to_string())?;
                for rank in 0..2 {
                    let parts=raw.iter().enumerate().map(|(i,x)|if x.is_empty() {Vec::new()} else {crop(x,op,i,rank)}).collect::<Vec<_>>();
                    let w=if weight.is_empty() {Vec::new()} else {crop(&weight,op,4,rank)};
                    let mut local=make(b,Shape {heads:32,..shape},&parts,&w)?;run(b,&mut local)?;
                    let got=b.read_vector_values(&local.output)?;let expected=crop(&reference,op,5,rank);
                    if got.len()!=expected.len() || got.iter().zip(&expected).any(|(a,z)|a.to_bits()!=z.to_bits()) {return Err(format!("{op:?} rank={rank} byte mismatch"));}
                    println!("BYTES op={op:?} rank={rank} words={} differing_words=0",got.len());
                }
                // Rotate independent states and outputs to avoid benchmarking one cache-hot state.
                let mut arms=Vec::new();
                for heads in [64,32] {let mut cases=Vec::new();for salt in 0..34 {
                    let s=Shape {heads,..shape};let (lengths,weights,_,_)=s.lengths().unwrap();
                    let inputs=lengths.iter().enumerate().map(|(i,&n)|pattern(n,i+salt)).collect::<Vec<_>>();
                    cases.push(make(b,s,&inputs,&pattern(weights,7))?);
                }arms.push(cases);}
                let mut times=[Vec::new(),Vec::new()];
                for trial in 0..14 {for arm in if trial%2==0 {[0,1]} else {[1,0]} {
                    b.synchronize()?;let start=b.gpu_mark()?;
                    for case in &mut arms[arm] {run(b,case)?;}
                    let end=b.gpu_mark()?;b.synchronize()?;let ns=start.ns_until(&end)?;
                    if trial>=4 {times[arm].push(ns);}
                    println!("SAMPLE op={op:?} heads={} trial={trial} calls=34 gpu_ns={ns}",if arm==0 {64} else {32});
                }}
                let full=median(&mut times[0]);let local=median(&mut times[1]);
                println!("PRICE op={op:?} full_ns={full} local_ns={local} saved_ns={} scope=34_synthetic_calls_not_model",full as i128-local as i128);
            }
            drop(queue);b.synchronize()?;Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os="linux")]
    if let Err(error)=device::main() {eprintln!("KDA HEADS ERROR {error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("KDA HEADS requires Linux and CUDA");std::process::exit(1);}
}
