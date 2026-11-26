//! Device route tables against an independent stable host sort, with guarded planes.
#![forbid(unsafe_code)]
use super::{Gpu,H,Result,ptr,sha256};
use std::time::Instant;
#[allow(dead_code)] #[path="../../../crates/gpu/src/prompt_route.rs"] pub(super) mod abi;
pub(super) fn words(v:impl IntoIterator<Item=u32>)->Vec<u8> {v.into_iter().flat_map(u32::to_le_bytes).collect()}
pub(super) fn expected(r:&abi::Reference)->Vec<Vec<u8>> {
    vec![words(r.counts.iter().copied()),words(r.starts.iter().copied()),words(r.order.iter().map(|v|v.to_bits())),words(r.pairs.iter().map(|v|v.to_bits())),
        words(r.grouped.iter().map(|v|v.to_bits())),words(r.placed.iter().map(|v|v.to_bits())),words(r.work128.iter().flat_map(|t|[t.expert,t.tile])),words(r.work64.iter().flat_map(|t|[t.expert,t.tile])),words([0])]
}
pub(super) fn launch(g:&Gpu,count:H,tables:H,p:&mut abi::Build)->Result<()> {g.launch(count,[p.shape.experts,1,1],abi::THREADS,0,&mut [ptr(p)])?;g.launch(tables,[p.shape.experts,1,1],abi::THREADS,0,&mut [ptr(p)])}
pub(super) fn compare(g:&Gpu,buffers:&[u64],want:&[Vec<u8>])->Result<()> {
    for (plane,(buffer,want)) in buffers.iter().zip(want).enumerate() {
        let got=g.read(*buffer,want.len()+32)?;
        if got[..16]!=[0xa5;16] || got[16+want.len()..]!=[0xa5;16] {return Err(format!("PROMPT_ROUTE guard changed plane={plane}"));}
        if got[16..16+want.len()]!=*want {return Err(format!("PROMPT_ROUTE table differs plane={plane}"));}
    }Ok(())
}
pub(super) fn fixture(g:&mut Gpu,s:abi::Shape,ids:&[f32])->Result<(abi::Build,Vec<u64>,u64)> {
    let raw=words(ids.iter().map(|v|v.to_bits()));let mut guarded=vec![0xa5;raw.len()+32];guarded[16..16+raw.len()].copy_from_slice(&raw);let input=g.upload(&guarded)?;
    let mut owners=Vec::new();for len in s.lengths().ok_or("prompt route fixture shape")? {owners.push(g.upload(&vec![0xa5;len*4+32])?);}
    let [counts,starts,order,pairs,grouped,placed,work128,work64,status]:[u64;9]=owners.iter().map(|p|p+16).collect::<Vec<_>>().try_into().map_err(|_|"prompt route fixture planes")?;
    Ok((abi::Build {ids:input+16,counts,starts,order,pairs,grouped,placed,work128,work64,status,shape:s,pad:0},owners,input))
}
fn invalid(image:&[u8],which:usize)->Result<()> {
    let mut g=Gpu::new()?;let module=g.module(image)?;let count=g.function(module,"prompt_route_count")?;let tables=g.function(module,"prompt_route_tables")?;
    let s=abi::Shape {rows:2,picks:2,experts:8};let mut ids=[0.0,1.0,2.0,3.0];ids[3]=[f32::NAN,0.5,8.0,2.0,-1.0,f32::INFINITY][which];
    let (mut p,_,_)=fixture(&mut g,s,&ids)?;
    let rejected=launch(&g,count,tables,&mut p).and_then(|()|g.sync()).is_err();
    if !rejected {return Err(format!("PROMPT_ROUTE invalid ID accepted control={which}"));}
    println!("PROMPT_ROUTE INVALID_REJECTED control={which}");Ok(())
}
fn median(mut v:Vec<f64>)->f64 {v.sort_by(f64::total_cmp);v[v.len()/2]}
pub(crate) fn run(image_path:&str,mode:&str)->Result<()> {
    let image=std::fs::read(image_path).map_err(|e|e.to_string())?;
    if let Some(which)=mode.strip_prefix("bad") {return invalid(&image,which.parse::<usize>().map_err(|e|e.to_string())?);}
    println!("PROMPT_ROUTE image_sha256={}",sha256::digest(&image));
    let mut g=Gpu::new()?;let module=g.module(&image)?;let count=g.function(module,"prompt_route_count")?;let tables=g.function(module,"prompt_route_tables")?;
    let mut cases=0;let mut bytes=0;
    for rows in [1u32,2,7,31,32,33,127,128,129,256,1024] {for picks in [1u32,2,8] {for experts in [8u32,257,512,1024] {for pattern in 0..3u32 {
        let s=abi::Shape {rows,picks,experts};let ids=(0..rows).flat_map(|t|(0..picks).map(move |p|match pattern {0=>(t*17+p*7)%experts,1=>p,_=>(t/2*17+p*7)%experts} as f32)).collect::<Vec<_>>();
        let reference=abi::reference(&ids,s)?;let want=expected(&reference);let (mut args,owners,input)=fixture(&mut g,s,&ids)?;
        launch(&g,count,tables,&mut args)?;g.sync()?;compare(&g,&owners,&want)?;
        let mut wrong=ids.clone();if picks>1 {wrong.swap(0,1);} else {wrong[0]=(wrong[0]+1.0)%experts as f32;}
        g.upload_at(input+16,&words(wrong.iter().map(|v|v.to_bits())))?;launch(&g,count,tables,&mut args)?;g.sync()?;
        if compare(&g,&owners,&want).is_ok() {return Err("PROMPT_ROUTE wrong-ID control was not detected".into());}
        g.upload_at(input+16,&words(ids.iter().map(|v|v.to_bits())))?;launch(&g,count,tables,&mut args)?;g.sync()?;compare(&g,&owners,&want)?;
        let original=g.read(input,ids.len()*4+32)?;if original[..16]!=[0xa5;16] || original[16+ids.len()*4..]!=[0xa5;16] || original[16..16+ids.len()*4]!=words(ids.iter().map(|v|v.to_bits())) {return Err("PROMPT_ROUTE input changed".into());}
        let size=want.iter().map(Vec::len).sum::<usize>();cases+=1;bytes+=size;
        println!("PROMPT_ROUTE BYTE_PASS rows={rows} picks={picks} experts={experts} pattern={pattern} bytes={size} WRONG_ID_PASS REPEAT_PASS GUARDS_PASS");
        if rows==1024 && picks==8 && experts==512 {
            let (mut cpu,mut gpu)=(Vec::new(),Vec::new());
            for _ in 0..7 {for candidate in [false,true,true,false] {
                let start=Instant::now();
                if candidate {launch(&g,count,tables,&mut args)?;g.sync()?;} else {
                    let raw=g.read(input+16,ids.len()*4)?;let back=raw.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
                    let built=expected(&abi::reference(&back,s)?);for (&owner,plane) in owners.iter().zip(&built) {g.upload_at(owner+16,plane)?;}g.sync()?;
                }
                let elapsed=start.elapsed().as_secs_f64()*1e6;if candidate {gpu.push(elapsed);} else {cpu.push(elapsed);}compare(&g,&owners,&want)?;
            }}
            println!("PROMPT_ROUTE TIME rows={rows} picks={picks} experts={experts} pattern={pattern} host_twin_roundtrip_us={:.6} gpu_us={:.6} order=AXXA",median(cpu),median(gpu));
        }
        g.clear()?;
    }}}}
    drop(g);
    for which in 0..6 {
        let output=std::process::Command::new(std::env::current_exe().map_err(|e|e.to_string())?).args(["--prompt-route",image_path,&format!("bad{which}")]).output().map_err(|e|e.to_string())?;
        let stdout=String::from_utf8_lossy(&output.stdout);if !output.status.success() || !stdout.contains(&format!("INVALID_REJECTED control={which}")) {return Err(format!("PROMPT_ROUTE invalid control failed which={which} status={} stdout={} stderr={}",output.status,stdout,String::from_utf8_lossy(&output.stderr)));}print!("{stdout}");
    }
    println!("PROMPT_ROUTE PASS cases={cases} bytes={bytes} wrong_ids={cases} repeats={cases} invalid=6");Ok(())
}
