//! Synthetic resident text edge bytes. This is not a checkpoint norm reference.
#![forbid(unsafe_code)]
use spark_gpu_probe::{text_edges::{Shape,Layout,NormStorage,greedy},bf16};
use std::{fs,path::Path};
type Result<T> = std::result::Result<T,String>;
const LABELS:[&str;6] = ["embedding","mean","norm","logits","fp32-logits","selected"];
fn cases()->Vec<Shape>{[(32,9,1),(96,17,2),(256,9,3),(4096,9,4),(96,9,7)].into_iter().map(|(h,v,s)|Shape::new(h,v,s).unwrap()).collect()}
fn values(n:usize,salt:usize)->Vec<f32>{(0..n).map(|i|((i*17+salt)%37) as f32/16.0-1.0).collect()}
struct Fixture {shape:Shape,layout:Layout,stored:Vec<u8>,residual:Vec<f32>,normalized:Vec<f32>,norm_weights:Vec<f32>}
impl Fixture {
    fn new(shape:Shape)->Self {
        let sizes=shape.weight_bytes().unwrap();
        // Alternate the stored-dtype fixtures. This does not select model math.
        let norm_storage=if shape.streams()%2==0{NormStorage::F32}else{NormStorage::Bf16};
        let layout=Layout{shape,embedding:0,head:sizes[0],scales:sizes[0]+sizes[1],
            norm:sizes.iter().sum::<usize>().div_ceil(4)*4,norm_storage,epsilon:1e-5};
        let mut stored=Vec::new();
        for v in values(shape.vocab()*shape.hidden(),3){stored.extend_from_slice(&((v.to_bits()>>16) as u16).to_le_bytes());}
        for i in 0..sizes[1]{stored.push([0x30,0x38,0x40,0xb0,0xb8,0xc0][(i+i/shape.hidden())%6]);}
        for i in 0..sizes[2]{stored.push(126+(i%3) as u8);}
        stored.resize(layout.norm,0);
        let norm_weights=(0..shape.hidden()).map(|i|{
            let value=1.0+(i%7) as f32/128.0+0.00390625;
            match norm_storage{NormStorage::F32=>value,NormStorage::Bf16=>bf16::rounded(value)}
        }).collect::<Vec<_>>();
        for &v in &norm_weights{match norm_storage{
            NormStorage::F32=>stored.extend_from_slice(&v.to_le_bytes()),
            NormStorage::Bf16=>stored.extend_from_slice(&((v.to_bits()>>16) as u16).to_le_bytes()),
        }}
        let residual=values(shape.residual(),7);
        let normalized=values(shape.hidden(),11).into_iter().map(|v|v+0.00390625).collect();
        Self{shape,layout,stored,residual,normalized,norm_weights}
    }
    fn expected(&self)->Vec<Vec<u8>> {
        let s=self.shape;let h=s.hidden();let mut embedding=Vec::new();
        for _ in 0..s.streams(){for col in 0..h{let at=(2*h+col)*2;embedding.push(f32::from_bits(u32::from(u16::from_le_bytes(self.stored[at..at+2].try_into().unwrap()))<<16));}}
        let mean=(0..h).map(|col|{let mut sum=0.0f32;for stream in 0..s.streams(){sum+=self.residual[stream*h+col];}bf16::rounded(sum/s.streams() as f32)}).collect::<Vec<_>>();
        let mut partials=Vec::new();
        for chunk in mean.chunks(128){let mut lanes=[0.0f32;32];for (i,&x) in chunk.iter().enumerate(){lanes[i%32]+=x*x;}partials.push(butterfly(lanes));}
        let mut lanes=[0.0f32;32];for (i,&v) in partials.iter().enumerate(){lanes[i%32]+=v;}
        let inverse=1.0/(butterfly(lanes)/h as f32+self.layout.epsilon).sqrt();
        let norm=mean.iter().zip(&self.norm_weights).map(|(&x,&w)|{
            let product=bf16::rounded(x*inverse)*w;
            match self.layout.norm_storage{NormStorage::F32=>product,NormStorage::Bf16=>bf16::rounded(product)}
        }).collect::<Vec<_>>();
        let logits=self.project(&norm);let fp32=self.project(&self.normalized);
        vec![bytes(&embedding),bytes(&mean),bytes(&norm),bytes(&logits),bytes(&fp32),greedy(&logits).unwrap().to_le_bytes().to_vec()]
    }
    fn project(&self,input:&[f32])->Vec<f32>{
        let s=self.shape;let h=s.hidden();
        let mut logits=Vec::new();
        for row in 0..s.vocab(){
            let mut lanes=[0.0f32;32];
            for (lane,acc) in lanes.iter_mut().enumerate(){
                for tile in 0..h.div_ceil(256){for j in 0..8{let col=tile*256+lane*8+j;if col>=h{continue;}
                    let w=match self.stored[self.layout.head+row*h+col]{0x30=>0.5,0x38=>1.0,0x40=>2.0,0xb0=>-0.5,0xb8=>-1.0,0xc0=>-2.0,_=>unreachable!()};
                    let scale=match self.stored[self.layout.scales+row*(h/32)+col/32]{126=>0.5,127=>1.0,128=>2.0,_=>unreachable!()};
                    *acc=(w*scale as f32).mul_add(input[col],*acc);
                }}
            }
            for distance in [16,8,4,2,1]{let old=lanes;for lane in 0..32-distance{lanes[lane]+=old[lane+distance];}}
            logits.push(lanes[0]);
        }
        logits
    }
}
fn butterfly(mut lanes:[f32;32])->f32{for distance in [16,8,4,2,1]{let old=lanes;for i in 0..32{lanes[i]+=old[i^distance];}}lanes[0]}
fn bytes(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn write(path:&Path,value:&[u8])->Result<()>{fs::write(path,value).map_err(|e|format!("{}: {e}",path.display()))}
fn equal(name:&str,a:&[u8],b:&[u8])->Result<()>{if a!=b{Err(format!("{name}: bytes differ; actual={} expected={} first={:?}",a.len(),b.len(),a.iter().zip(b).position(|(a,b)|a!=b)))}else{Ok(())}}
fn host(out:&Path)->Result<()>{
    for (i,s) in cases().into_iter().enumerate(){let f=Fixture::new(s);for (label,value) in LABELS.iter().zip(f.expected()){write(&out.join(format!("{i}-{label}.bin")),&value)?;}}
    println!("TEXT-EDGE HOST PASS cases={}; synthetic primitive oracle",cases().len());Ok(())
}
fn expected(root:&Path)->Result<Vec<Vec<Vec<u8>>>>{
    cases().into_iter().enumerate().map(|(i,s)|{
        let sizes=[s.residual()*4,s.hidden()*4,s.hidden()*4,s.vocab()*4,s.vocab()*4,4];
        LABELS.iter().zip(sizes).map(|(label,size)|{let data=fs::read(root.join(format!("{i}-{label}.bin"))).map_err(|e|e.to_string())?;
            if data.len()!=size{return Err("text edge expected extent differs".into());}Ok(data)}).collect()
    }).collect()
}
#[cfg(target_os="linux")]
fn device(source:&Path,out:&Path)->Result<()>{
    use spark_gpu_probe::{driver::with_backend,layer_ptx,load::{WeightFile,WeightRange}};
    let saved=expected(source)?;
    with_backend(0,layer_ptx()?,64<<20,|b|{
        for (i,s) in cases().into_iter().enumerate(){
            let f=Fixture::new(s);let path=out.join(format!("{i}-synthetic-weights.bin"));write(&path,&f.stored)?;
            let file=fs::File::open(path).map_err(|e|e.to_string())?;
            let ranges=[WeightRange{file_offset:0,destination:0,bytes:f.stored.len(),row_bytes:0,row_stride:0}];
            let arena=b.load_weight_arena(f.stored.len(),&[WeightFile{file:&file,ranges:&ranges}])?;
            let edges=b.resident_text_edges(&arena,f.layout)?;drop(arena);
            if edges.embed(s.vocab() as u32).is_ok(){return Err("out-of-vocabulary token accepted".into());}
            let short=b.upload_vector(&[0.0])?;
            if edges.mean(&short).is_ok()||edges.norm(&short).is_ok()||edges.project_fp32(&short).is_ok(){return Err("short edge operand accepted".into());}
            let embedding=edges.embed(2)?;let residual=b.upload_vector(&f.residual)?;let mean=edges.mean(&residual)?;
            let norm=edges.norm(&mean)?;let logits=edges.project_fp32(&norm)?;
            let normalized=b.upload_vector(&f.normalized)?;let fp32=edges.project_fp32(&normalized)?;
            let logits=b.read_vector_values(&logits)?;
            let actual=vec![bytes(&b.read_vector_values(&embedding)?),bytes(&b.read_vector_values(&mean)?),bytes(&b.read_vector_values(&norm)?),
                bytes(&logits),bytes(&b.read_vector_values(&fp32)?),greedy(&logits)?.to_le_bytes().to_vec()];
            // Save all outputs before comparing, including a failed case.
            for (label,value) in LABELS.iter().zip(&actual){write(&out.join(format!("{i}-{label}.bin")),value)?;}
            for (j,(label,value)) in LABELS.iter().zip(&actual).enumerate(){equal(label,value,&saved[i][j])?;}
            println!("TEXT-EDGE BYTES case={i} hidden={} vocab={} streams={} PASS",s.hidden(),s.vocab(),s.streams());
        }
        if let Some(error)=b.fatal_error(){return Err(error);}Ok(())
    })?;
    write(&out.join("COMPLETE"),b"TEXT-EDGE DEVICE PASS\n")?;
    println!("TEXT-EDGE DEVICE PASS cases={}; candidate norm only; no decoder or model IDs",cases().len());Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path)->Result<()>{Err("text edge device probe requires Linux and CUDA".into())}
fn compare(left:&Path,right:&Path)->Result<()>{
    for root in [left,right]{equal("completion",&fs::read(root.join("COMPLETE")).map_err(|e|e.to_string())?,b"TEXT-EDGE DEVICE PASS\n")?;}
    for (a,b) in expected(left)?.iter().zip(expected(right)?){for (a,b) in a.iter().zip(b){equal("cross-run edge bytes",a,&b)?;}}
    println!("TEXT-EDGE COMPARE PASS cases={}",cases().len());Ok(())
}
fn main(){
    println!("TEXT-EDGE source={}",include_str!("../../../../REVISION").trim());
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let run=||->Result<()>{match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice(){
        ["host",out]=>{fs::create_dir(out).map_err(|e|e.to_string())?;host(Path::new(out))},
        ["device",source,out]=>{expected(Path::new(source))?;fs::create_dir(out).map_err(|e|e.to_string())?;device(Path::new(source),Path::new(out))},
        ["compare",left,right]=>compare(Path::new(left),Path::new(right)),
        _=>Err("usage: text-edge-check host NEW_DIR | device EXPECTED NEW_DIR | compare LEFT RIGHT".into()),
    }};
    if let Err(error)=run(){eprintln!("TEXT-EDGE FAIL {error}");std::process::exit(1);}
}
#[test]
fn synthetic_oracle_has_all_nonempty_planes(){
    for s in cases(){let f=Fixture::new(s);assert!(f.layout.ranges(f.stored.len()).is_ok());let output=f.expected();
        assert_eq!(output.iter().map(Vec::len).collect::<Vec<_>>(),[s.residual()*4,s.hidden()*4,s.hidden()*4,s.vocab()*4,s.vocab()*4,4]);}
}
#[test]
fn stored_norm_dtype_controls_only_the_product_boundary(){
    for s in cases(){let f=Fixture::new(s);let values=f.expected();
        let exact=values[2].chunks_exact(4).all(|b|u32::from_le_bytes(b.try_into().unwrap())&0xffff==0);
        assert_eq!(exact,f.layout.norm_storage==NormStorage::Bf16);
    }
}
