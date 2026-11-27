//! Byte and guard proof for compact causal selection status. Safe CUDA harness.
use super::{Buffer,Gpu,H,Inner,PArgs,PShape,Result,GUARD,le,median,ptr};
use std::time::Instant;
fn launch(g:&Gpu,f:H,input:u64,output:u64,m:usize,slots:usize,first:usize)->Result<()> {
    let inner=Inner {op:5,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.,bound:0.};
    let shape=PShape {op:27,tokens:m as u32,heads:0,dim:0,taps:0,rows:0,cols:0,first:first as u32,slots:slots as u32,inner};
    let mut args=PArgs {input:[input,4,4,4,4],lengths:[(m*(slots+1)) as u64,0,0,0,0],weight:4,weights:0,output,units:m as u64,words:1,shape};
    g.launch(f,[m.div_ceil(128) as u32,1,1],128,0,&mut [ptr(&mut args)])
}
pub(super) fn run(g:&mut Gpu,module:H,time:bool)->Result<()> {
    let f=g.function(module,"prompt_rows")?;let mut cases=0;let mut controls=0;
    for m in [1usize,5,256,1024] {for slots in [1usize,7,2051] {for first in [1usize,8192,262144] {
        let mut table=vec![f32::NAN;m*(slots+1)];
        for (token,row) in table.chunks_exact_mut(slots+1).enumerate() {
            let context=first+token;let count=context.min(slots);
            for (slot,id) in row[..count].iter_mut().enumerate() {*id=(context-1-slot) as f32;}
            row[slots]=count as f32;
        }
        for fault in 0..14 {
            let mut input=table.clone();let token=m/2;let at=token*(slots+1);let mut want=vec![0.;m];
            if fault>0 {
                let count=fault<=7;let bad=match (fault-1)%7 {0=>f32::NAN,1=>f32::INFINITY,2=>f32::NEG_INFINITY,3=>-1.,4=>0.5,5=>(if count {slots.min(first+token)+1} else {first+token}) as f32,_=>u32::MAX as f32};
                input[at+if count {slots} else {0}]=bad;want[token]=if count {2.} else {1.};controls+=1;
            }
            let source=Buffer::new(g,&input)?;let out=Buffer::new(g,&vec![f32::NAN;m])?;out.poison(g)?;
            launch(g,f,source.address,out.address,m,slots,first)?;g.sync()?;
            let got=out.read(g)?;let expected=[GUARD.as_slice(),&le(&want),GUARD.as_slice()].concat();
            if got!=expected {return Err(format!("SELECTION CHECK differs m={m} slots={slots} first={first} fault={fault}"));}
            cases+=1;g.clear()?;
        }
    }}}
    println!("SELECTION CHECK BYTES PASS cases={cases} wrong_inputs_detected={controls} output_guards=unchanged");
    if !time {return Ok(());}
    for m in [256usize,1024] {
        let (slots,first)=(2051usize,8192usize);let mut table=vec![0.;m*(slots+1)];
        for row in table.chunks_exact_mut(slots+1) {for (i,id) in row[..slots].iter_mut().enumerate() {*id=i as f32;}row[slots]=slots as f32;}
        let source=Buffer::new(g,&table)?;let out=Buffer::new(g,&vec![0.;m])?;let mut samples=Vec::new();
        launch(g,f,source.address,out.address,m,slots,first)?;g.sync()?;
        for _ in 0..9 {let start=Instant::now();for _ in 0..16 {launch(g,f,source.address,out.address,m,slots,first)?;}g.sync()?;samples.push(start.elapsed().as_secs_f64()*1e6/16.);}
        println!("SELECTION CHECK TIME tokens={m} slots={slots} median_us={:.3} input_bytes={} status_bytes={}",median(samples),table.len()*4,m*4);g.clear()?;
    }Ok(())
}
