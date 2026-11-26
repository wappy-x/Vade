//! Replays the absorbed attention softmax on the same score and count words.
use super::*;
use super::super::{capture_graph,sample,TIMER};

// The safe exponential used by the causal CPU reference, from gpu/src/next_math.rs.
fn exp(t:f32)->f32 {
    let t=t.clamp(-87.0,87.0);
    let n=t*core::f32::consts::LOG2_E+12_582_912.0-12_582_912.0;
    let r=(-n).mul_add(f32::from_bits(0x3f31_7200),t);
    let r=(-n).mul_add(0.0000014286068,r);
    let mut p=1.0/5040.0f32;
    for c in [1.0/720.0,1.0/120.0,1.0/24.0,1.0/6.0,0.5,1.0,1.0] {p=p.mul_add(r,c);}
    p*f32::from_bits(((n as i32+127) as u32)<<23)
}

fn reference(scores:&[f32],slots:usize,count:usize)->Vec<f32> {
    let mut out=vec![0.0;scores.len()];
    if count==0 {return out;}
    for (score,output) in scores.chunks_exact(slots).zip(out.chunks_exact_mut(slots)) {
        let maximum=score[..count].iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let terms=score[..count].iter().map(|&v|exp(v-maximum)).collect::<Vec<_>>();
        let denominator=terms.iter().sum::<f32>();
        for (v,term) in output.iter_mut().zip(terms) {
            let word=(term/denominator).to_bits();
            *v=f32::from_bits(word.wrapping_add(0x7fff+((word>>16)&1))&0xffff_0000);
        }
    }
    out
}

fn table(g:&mut Gpu,heads:u32,slots:usize,count:usize,x:&Buffer,out:&Buffer,entry:u32)->Result<u64> {
    let s=Shape {op:SOFTMAX,heads,dim:256,taps:0,rows:slots as u32,cols:0,epsilon:0.0,bound:0.0};
    let count=Buffer::new(g,&[count as f32])?;
    let a=Arguments {input:[x.address,count.address,4,4],weight:4,output:out.address,shape:s};
    g.upload(&command(&a,entry))
}

pub(super) fn run(g:&mut Gpu,base:H,fast:H)->Result<()> {
    let (mut cases,mut words)=(0,0);
    for slots in [1usize,2,31,32,33,97,98,127,128,129,159,160,255,256,257,512,1024,2048,2051,4096] {
        for count in [0,1,slots/2,slots.saturating_sub(1),slots] {for kind in [0,2,5,6,7,8] {
            let scores=pattern(kind,16*slots,97+slots as u64);
            let input=Buffer::new(g,&scores)?;
            let out=Buffer::new(g,&vec![f32::NAN;scores.len()])?;
            let expected=[GUARD.as_slice(),&le(&reference(&scores,slots,count)),GUARD.as_slice()].concat();
            for (entry,f,blocks) in [(1,base,1),(19,fast,16)] {
                out.poison(g)?;
                let cmd=table(g,16,slots,count,&input,&out,entry)?;
                launch_replay(g,f,cmd,(blocks,128))?;
                g.sync()?;
                let got=out.read(g)?;
                if let Some(at)=first_difference(&expected,&got) {
                    return Err(format!("SOFTMAX16 byte difference slots={slots} count={count} kind={kind} entry={entry} byte={at}"));
                }
                words+=scores.len();
            }
            cases+=1;
            g.clear()?;
        }}
    }
    println!("SOFTMAX16 WORDS PASS cases={cases} words={words}");

    let timer=|g:&mut Gpu| {let m=g.module(TIMER.as_bytes())?;g.function(m,"kb_stamp")};
    for slots in [98usize,112,128,144,160,256,512,1024,2051] {
        let stamp=timer(g)?;
        let ticks=g.upload(&[0;16])?;
        let mut times=[0.0;3];
        for (i,(heads,entry,f,blocks)) in [(16,1,base,1),(16,19,fast,16),(32,19,fast,32)].into_iter().enumerate() {
            let mut commands=Vec::new();
            for seed in 1..=16 {
                let scores=Buffer::new(g,&pattern(0,heads as usize*slots,seed))?;
                let out=Buffer::new(g,&vec![0.0;heads as usize*slots])?;
                commands.push(table(g,heads,slots,slots,&scores,&out,entry)?);
            }
            let graph=capture_graph(g,|| {
                let mut at=ticks;g.launch(stamp,[1,1,1],1,0,&mut [ptr(&mut at)])?;
                for &cmd in &commands {launch_replay(g,f,cmd,(blocks,128))?;}
                at+=8;g.launch(stamp,[1,1,1],1,0,&mut [ptr(&mut at)])
            })?;
            for _ in 0..3 {sample(&graph,ticks)?;}
            let mut samples=Vec::new();
            for _ in 0..9 {samples.push(sample(&graph,ticks)?.1/commands.len() as f64);}
            times[i]=median(samples);
        }
        println!("SOFTMAX16 TIME slots={slots} generic16_us={:.3} fast16_us={:.3} fast32_us={:.3}",times[0],times[1],times[2]);
        g.clear()?;
    }
    Ok(())
}
