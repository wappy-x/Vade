//! Prompt frontend phase bench: the production CPU decode/expand plus two uploads
//! versus compact upload plus the GPU frontend. Read-plan construction/I/O is excluded.
use super::*;
use engine_model::flash::{nvfp4,refcpu};
fn bytes<T>(x:&[T])->&[u8] {unsafe {std::slice::from_raw_parts(x.as_ptr().cast(),std::mem::size_of_val(x))}}
pub(super) fn run(g:&mut Gpu, modules:[H;2], c:&Case, o:&Options, hashes:&[String;2], source:&str)->Result<()> {
    let n=c.m;
    let embed:Vec<u16>=(0..256*2560).map(|i|{
        let x=(((i*13+i/2560*29)%997) as f32-498.)/127.1;
        (x.to_bits()>>16) as u16
    }).collect();
    let mut packed=vec![0u8;n*1444];
    let mut staged=vec![0u8;n*6560];
    for r in 0..n {
        let token=(r*37%256) as u32;
        packed[r*4..r*4+4].copy_from_slice(&token.to_le_bytes());
        for h in 0..16 {
            let p=n*4+r*1440+h*90;
            for j in 0..80 {packed[p+j]=((r*17+h*71+j*13)%256) as u8;}
            for j in 0..10 {packed[p+80+j]=((r*7+h*23+j*11)%127) as u8;}
        }
        staged[r*6560..r*6560+1440].copy_from_slice(&packed[n*4+r*1440..n*4+(r+1)*1440]);
        staged[r*6560+1440..(r+1)*6560].copy_from_slice(bytes(&embed[token as usize*2560..(token as usize+1)*2560]));
    }
    let dedup=std::env::var("KBENCH_FRONTEND_DEDUP").is_ok_and(|v|v=="1");
    if dedup {
        let mut compact=vec![0u8;n*68];
        compact[..n*4].copy_from_slice(&packed[..n*4]);
        let mut map=std::collections::BTreeMap::<[u8;90],u32>::new();
        let mut unique=Vec::new();
        for row in 0..n*16 {
            let key:[u8;90]=packed[n*4+row*90..n*4+(row+1)*90].try_into().unwrap();
            let index=*map.entry(key).or_insert_with(||{let i=unique.len() as u32;unique.push(key);i});
            compact[n*4+row*4..n*4+row*4+4].copy_from_slice(&index.to_le_bytes());
        }
        for row in unique {compact.extend_from_slice(&row);}
        packed=compact;
    }
    let global=0.37f32;
    let cpu=|x:&mut [f32],e:&mut [f32]| {
        let workers=(n/64).clamp(1,8);
        let block=n.div_ceil(workers);
        std::thread::scope(|s| {
            for ((st,x),e) in staged.chunks(block*6560).zip(x.chunks_mut(block*10240)).zip(e.chunks_mut(block*2560)) {
                s.spawn(move || {
                    for ((row,x),e) in st.as_chunks::<6560>().0.iter().zip(x.as_chunks_mut::<10240>().0).zip(e.as_chunks_mut::<2560>().0) {
                        for h in 0..16 {
                            let p=&row[h*90..(h+1)*90];
                            refcpu::ple_row_nvfp4(&p[..80],&p[80..],global,&mut e[h*160..(h+1)*160]).unwrap();
                        }
                        for (out,&v) in x[..2560].iter_mut().zip(row[1440..].as_chunks::<2>().0) {
                            *out=nvfp4::bf16_to_f32(u16::from_le_bytes(v));
                        }
                        for stream in 1..4 {x.copy_within(0..2560,stream*2560);}
                    }
                });
            }
        });
    };
    let (mut x,mut e)=(vec![0f32;n*10240],vec![0f32;n*2560]);
    cpu(&mut x,&mut e);
    let expected=[sha256::digest(bytes(&x)),sha256::digest(bytes(&e))].concat();
    let input=g.alloc(packed.len())?;
    let weight=g.upload(bytes(&embed))?;
    let output=g.alloc(x.len()*4)?;
    let lexical=g.alloc(e.len()*4)?;
    let (rows,scale)=(n as u32,global);
    let entry=if dedup {"flash_next_prompt_frontend_dedup"}else{"flash_next_prompt_frontend"};
    let f=g.function(modules[1],entry)?;
    let invoke=|arm:usize,x:&mut [f32],e:&mut [f32]|->Result<f64>{
        let (mut input,mut weight,mut output,mut lexical,mut rows,mut scale)=(input,weight,output,lexical,rows,scale);
        let t=Instant::now();
        if arm==0 {
            cpu(x,e);g.upload_at(output,bytes(x))?;g.upload_at(lexical,bytes(e))?;
        }else{
            g.upload_at(input,&packed)?;
            g.launch(f,[(n*2560).div_ceil(256) as u32,1,1],256,0,&mut[
                ptr(&mut input),ptr(&mut weight),ptr(&mut output),ptr(&mut lexical),ptr(&mut rows),ptr(&mut scale),
            ])?;
        }
        g.sync()?;Ok(t.elapsed().as_secs_f64()*1e6)
    };
    let mut digests=Vec::new();
    for arm in 0..2 {
        invoke(arm,&mut x,&mut e)?;
        let got=[sha256::digest(&g.read(output,x.len()*4)?),sha256::digest(&g.read(lexical,e.len()*4)?)].concat();
        if got!=expected {return Err(format!("{} arm{arm} differs from CPU oracle expected={expected} got={got}",c.name));}
        digests.push(got);
    }
    let mut times:[Vec<f64>;2]=[Vec::new(),Vec::new()];
    while times[0].len()<9 || times.iter().any(|v|v.iter().sum::<f64>()<o.ms as f64*1000.) {
        let order=if times[0].len().is_multiple_of(2) {[0,1]}else{[1,0]};
        for arm in order {times[arm].push(invoke(arm,&mut x,&mut e)?);}
    }
    for arm in 0..2 {
        println!("{{\"case\":{:?},\"arm\":{:?},\"entry\":{:?},\"m\":{},\"median_us\":{:.6},\"samples\":{},\"sha256\":{:?},\"bytes_equal\":true,\"calls_per_prompt\":1,\"timing_basis\":\"host wall: decode+uploads vs packed upload+kernel, sync each sample\",\"file_reads_included\":false,\"cpu_oracle\":true,\"ptx_sha256\":{:?},\"bench_source_sha256\":{:?}}}",
            c.name,if arm==0 {"A"}else{"X"},if arm==0 {"CPU frontend + two uploads"}else{entry},n,median(&times[arm]),times[arm].len(),digests[arm],hashes[arm],source);
    }
    Ok(())
}
