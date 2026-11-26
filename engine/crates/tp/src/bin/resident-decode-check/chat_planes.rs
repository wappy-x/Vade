//! Check separate KV allocations, wrapped ring slots, untouched output guards and a short plane.
use spark_gpu_probe::causal_ops::{Op,Shape};
use spark_tp::Result;

pub fn guard()->Result<()> {
    const GUARD:u32=0x7fc0_dead;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let _queue=b.resident_queue()?;
        b.enable_resident_launches()?;
        let mut checked=0;
        for (count,view) in [(2usize,31u32),(4,32),(8,257),(16,1024)] {
            for op in [Op::WindowScores,Op::WindowValues] {
                let scores=op==Op::WindowScores;
                let s=Shape {op,heads:6,dim:32,taps:3,rows:view,cols:if scores {9} else {0},epsilon:0.0,bound:if scores {0.125} else {0.0}};
                let (lens,_,units,words)=s.lengths().ok_or("unadmitted plane fixture")?;
                let make=|n:usize,seed:usize|(0..n).map(|i|((i*13+seed*7)%67) as f32/32.0-1.0).collect::<Vec<_>>();
                let a=make(count*lens[0],1);
                let c=make(count*lens[2],7);
                let steps=(0..count).flat_map(|i|[(view-i as u32) as f32,((i*7)%view as usize) as f32]).collect::<Vec<_>>();
                let planes=(0..count).map(|i| {
                    let v=make(lens[1]*2,i+1);
                    v.as_chunks::<2>().0.iter().map(|pair|f32::from_bits((pair[0].to_bits()>>16)|(pair[1].to_bits()&0xffff0000))).collect::<Vec<_>>()
                }).collect::<Vec<_>>();
                let d=planes.iter().map(|v|b.upload_vector(v)).collect::<Result<Vec<_>>>()?;
                let mut p=d.iter().collect::<Vec<_>>();
                let (x,y,z)=(b.upload_vector(&a)?,b.upload_vector(&c)?,b.upload_vector(&steps)?);
                let n=count*units*words;
                let whole=b.upload_vector(&vec![f32::from_bits(GUARD);n+128])?.into_resident_shared()?;
                let mut out=whole.resident_range(64..64+n)?;
                let last=count-1;
                let short=b.upload_vector(&planes[last][1..])?;
                p[last]=&short;
                if b.block_planes_into(s,[&x,&y,&z],&p,&mut out).is_ok() {return Err("short final plane was accepted".into());}
                p[last]=&d[last];
                if !b.block_planes_into(s,[&x,&y,&z],&p,&mut out)? {return Err("per-chat fixture did not record".into());}
                b.flush_resident_launches()?;
                b.synchronize()?;
                let got=b.read_vector_values(&whole)?;
                if got[..64].iter().chain(&got[64+n..]).any(|v|v.to_bits()!=GUARD) {return Err("per-chat guard changed".into());}
                for i in 0..count {
                    let x=[&a[i*lens[0]..(i+1)*lens[0]],planes[i].as_slice(),&c[i*lens[2]..(i+1)*lens[2]],&steps[2*i..2*i+2]];
                    let expected=spark_gpu_probe::causal_ops::reference(s,x,&[])?;
                    let at=64+i*units*words;
                    if !expected.iter().zip(&got[at..]).all(|(a,b)|a.to_bits()==b.to_bits()) {return Err(format!("per-chat {op:?} mismatch at row {i} of {count}"));}
                }
                checked+=1;
            }
        }
        eprintln!("CHAT PLANES GUARDS PASS launches={checked}");
        Ok(())
    })
}
