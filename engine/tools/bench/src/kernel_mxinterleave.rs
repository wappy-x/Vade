//! Independent matrix block scheduling. Compare every output and time cold weights.
#![forbid(unsafe_code)]
use super::*;
use super::super::super::{capture_graph,sample,TIMER};
fn jobs(spec:&[(bool,usize,usize)])->Vec<Job> {spec.iter().map(|&(plain,rows,cols)|Job {plain,rows,cols}).collect()}
fn equal(g:&Gpu,a:&Built,b:&Built)->Result<()> {
    for ((pa,n),(pb,m)) in a.outputs.iter().zip(&b.outputs) {
        if n!=m {return Err("MXINTERLEAVE output extent differs".into());}
        let x=g.read(*pa,*n)?;let y=g.read(*pb,*m)?;
        if x!=y {return Err(format!("MXINTERLEAVE BYTE FAIL offset={:?}",x.iter().zip(&y).position(|(a,b)|a!=b)));}
        if y[..16]!=GUARD || y[y.len()-16..]!=GUARD {return Err("MXINTERLEAVE guard changed".into());}
    }Ok(())
}
pub(in super::super) fn run(g:&mut Gpu,(ma,mb):(H,H))->Result<()> {
    let one=[g.function(ma,"flash_resident_projection_bf16")?,g.function(mb,"flash_resident_projection_bf16")?];
    let rows=[g.function(ma,"flash_resident_projection_bf16_rows")?,g.function(mb,"flash_resident_projection_bf16_rows")?];
    let replay=[g.function(ma,"resident_launch_mixed_bf16")?,g.function(mb,"resident_launch_mixed_bf16")?];
    let mut rng=Rng(0x243f_6a88_85a3_08d3);let mut checked=0;
    let cases=[
        jobs(&[(false,4096,4096),(false,4096,4096),(false,4096,4096),(true,64,4096),(true,32,4096),(true,64,4096)]),
        jobs(&[(true,16,4096),(false,17,32),(false,33,288),(true,64,4096),(false,5,1056)]),
        jobs(&[(true,4096,128),(true,4096,128)]),
    ];
    for index in 0..67 {
        let random;
        let jobs=if index<cases.len() {&cases[index]} else {
            let count=1+(rng.next()%9) as usize;
            random=(0..count).map(|_|Job {plain:rng.next().is_multiple_of(2),rows:1+(rng.next()%300) as usize,cols:32*(1+(rng.next()%64) as usize)}).collect::<Vec<_>>();&random
        };
        for tokens in [1,2,3,4] {for round in [false,true] {for kinds in [(0,0),(2,2),(4,4),(5,5),(6,6)] {
            let fx=p1_build(g,jobs,tokens,round,700+index as u64,kinds)?;
            let shape=if tokens==1 {Shape::One(0)} else {Shape::Rows {spread:true}};
            let mut a=p1_arm(g,jobs,&fx,shape,tokens)?;let mut b=p1_arm(g,jobs,&fx,shape,tokens)?;
            let f=if tokens==1 {one} else {rows};
            launch_batch(g,f[0],&mut a.batch)?;launch_batch(g,f[1],&mut b.batch)?;g.sync()?;equal(g,&a,&b)?;
            if tokens==1 {
                // Kind 34 is MixedBf16 in gpu/src/resident_launch.rs. The table
                // encoder above matches the recorder's initialized payload.
                let ta=g.upload(&command(34,&batch_bytes(&a.batch)))?;let tb=g.upload(&command(34,&batch_bytes(&b.batch)))?;
                for (p,n) in &b.outputs {g.upload_at(*p+16,&vec![0xff;*n-32])?;}
                let blocks=a.batch.layer.ends[a.batch.layer.count as usize];
                launch_replay(g,replay[0],ta,blocks)?;launch_replay(g,replay[1],tb,blocks)?;g.sync()?;equal(g,&a,&b)?;
            }
            checked+=1;g.clear()?;
        }}}
    }
    println!("MXINTERLEAVE BYTES PASS cases={checked}");
    let timer_module=g.module(TIMER.as_bytes())?;let timer=g.function(timer_module,"kb_stamp")?;
    for (label,spec) in [("qkv",vec![(false,4096,4096);3]),
        ("qkv_plain",vec![(false,4096,4096),(false,4096,4096),(false,4096,4096),(true,64,4096),(true,32,4096),(true,64,4096)]),
        ("unequal",vec![(false,2048,4096),(false,4096,4096),(false,1024,4096)]),
        ("dense_gate_up",vec![(false,6144,4096);2]),
        ("plain",vec![(true,4096,128),(true,4096,128)])] {
        let jobs=jobs(&spec);let bytes:usize=jobs.iter().map(|j|if j.plain {j.rows*j.cols*2} else {j.rows*j.cols+j.rows*j.cols/32}).sum();
        let copies=(384usize<<20).div_ceil(bytes).clamp(2,96);let mut tables=Vec::new();let mut blocks=0;
        for copy in 0..copies {
            let mut fx=p1_build(g,&jobs,1,true,900+copy as u64,(0,0))?;
            // These timing groups all read the same completed source row.
            let input=fx.inputs[0];fx.inputs.fill(input);
            let a=p1_arm(g,&jobs,&fx,Shape::One(0),1)?;
            blocks=a.batch.layer.ends[a.batch.layer.count as usize];
            tables.push(g.upload(&command(34,&batch_bytes(&a.batch)))?);
        }
        let times=g.upload(&[0;16])?;
        let mut samples=[Vec::new(),Vec::new()];let mut events=[Vec::new(),Vec::new()];
        {
            // The stamps are inside each captured graph. Events around a graph
            // also count a host delay between event record and graph submission.
            let capture=|arm:usize|capture_graph(g,|| {
                let mut stamp=times;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
                for _ in 0..4 {for &table in &tables {launch_replay(g,replay[arm],table,blocks)?;}}
                stamp=times+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
                Ok(())
            });
            let graphs=[capture(0)?,capture(1)?];
            for _ in 0..3 {for graph in &graphs {sample(graph,times)?;}}
            for repeat in 0..15 {for k in 0..2 {
                let arm=(repeat+k)%2;let (event,inside)=sample(&graphs[arm],times)?;
                samples[arm].push(inside/(4*copies) as f64);events[arm].push(event/(4*copies) as f64);
            }}
        }
        println!("MXINTERLEAVE TIME clock=graph_stamps shape={label} main_us={:.3} candidate_us={:.3} bytes={bytes} copies={copies} main_samples={:?} candidate_samples={:?} main_event_samples={:?} candidate_event_samples={:?}",median(samples[0].clone()),median(samples[1].clone()),samples[0],samples[1],events[0],events[1]);
        g.clear()?;
    }
    println!("MXINTERLEAVE PASS");Ok(())
}
