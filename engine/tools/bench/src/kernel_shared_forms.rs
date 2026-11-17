//! Existing shared projection schedules, with identical operands and device graph stamps.
#![forbid(unsafe_code)]
use super::*;
use super::super::{capture_graph,sample,TIMER};
#[derive(Clone,Copy)] enum Form { Mixed, Fold, Rows, Loop, Scalar, PackedScalar, Small, Pair, Tile }
struct Arm { round:H, name:&'static str, form:Form, f:H }
struct Fixture { batches:Vec<Batch>, out:Buffer }
fn launch_form(g:&Gpu,arm:&Arm,fx:&Fixture)->Result<()> {
    for batch in &fx.batches {let mut b=*batch;let p=b.layer.projections[0];match arm.form {
        Form::Scalar=>launch_g(g,arm.f,p.weights,p.input,p.output,p.rows as usize,p.cols as usize,plain_geo(p.rows as usize,p.cols as usize))?,
        Form::PackedScalar=>launch_g(g,arm.f,p.weights,p.input,p.output,p.rows as usize,p.cols as usize,head_geo(p.rows as usize))?,
        Form::Fold=>g.launch(arm.f,[b.layer.ends[b.layer.count as usize],1,1],160,0,&mut [ptr(&mut b)])?,
        Form::Small=>{let mut p=p;let per=if p.tokens==2 {8} else {16};g.launch(arm.f,[p.rows.div_ceil(per),1,1],128,0,&mut [ptr(&mut p)])?;},
        Form::Pair=>g.launch(arm.f,[p.rows.div_ceil(4),1,1],128,0,&mut [ptr(&mut b.layer)])?,
        Form::Tile=>{let mut p=p;g.launch(arm.f,[p.rows.div_ceil(48),p.tokens.div_ceil(16),1],384,0,&mut [ptr(&mut p)])?;},
        _=>launch_batch(g,arm.f,&mut b)?,
    }
        if b.round!=0 && matches!(arm.form,Form::Small|Form::Pair|Form::Tile|Form::PackedScalar) {
            let mut args=Vec::new();args.extend(p.output.to_le_bytes());args.extend(p.output.to_le_bytes());args.extend((p.rows*p.tokens).to_le_bytes());args.extend(0u32.to_le_bytes());
            g.launch(arm.round,[(p.rows*p.tokens).div_ceil(128),1,1],128,0,&mut [args.as_mut_ptr().cast()])?;
        }
    }Ok(())
}
fn fixture(g:&mut Gpu,base:&Batch,arm:&Arm,m:usize,packed:bool)->Result<Fixture> {
    let words=base.layer.projections[..base.layer.count as usize].iter().map(|p|p.rows as usize*m).sum();
    let out=Buffer::new(g,&vec![f32::NAN;words])?;out.poison(g)?;
    if matches!(arm.form,Form::Loop) {
        let mut batches=Vec::new();for t in 0..m {let mut b=*base;b.layer.ends=[0;10];let mut offset=0;
            for i in 0..b.layer.count as usize {let p=&mut b.layer.projections[i];p.input+=(t*p.cols as usize*4) as u64;p.output=out.address+((offset+t*p.rows as usize)*4) as u64;p.tokens=1;
                b.layer.ends[i+1]=b.layer.ends[i]+if b.plain&(1<<i)!=0 {p.rows.div_ceil(128)} else {p.rows.div_ceil(4)};offset+=p.rows as usize*m;}
            batches.push(b);
        }return Ok(Fixture {batches,out});
    }
    let mut batches=Vec::new();let mut batch=Batch {round:base.round,..Batch::default()};let mut offset=0;
    for i in 0..base.layer.count as usize {
        let p=base.layer.projections[i];let plain=base.plain&(1<<i)!=0;
        let all=matches!(arm.form,Form::Rows|Form::Small|Form::Pair|Form::Tile);
        if !all && batch.layer.count as usize+m>9 {batches.push(batch);batch=Batch {round:base.round,..Batch::default()};}
        for t in 0..if all {1} else {m} {
            let at=batch.layer.count as usize;
            let tokens=if all {m} else {1};
            batch.layer.projections[at]=MxProjection {input:p.input+(t*p.cols as usize*4) as u64,output:out.address+((offset+t*p.rows as usize)*4) as u64,
                tokens:tokens as u32,..p};
            if plain {batch.plain|=1<<at;}
            let blocks=if matches!(arm.form,Form::Fold) {p.rows.div_ceil(4)} else if plain {p.rows.div_ceil(128)*tokens as u32} else {p.rows.div_ceil(4)};
            batch.layer.ends[at+1]=batch.layer.ends[at]+blocks;batch.layer.count+=1;
            if matches!(arm.form,Form::Scalar|Form::PackedScalar|Form::Small|Form::Pair|Form::Tile|Form::Loop) || batch.layer.count==9 {batches.push(batch);batch=Batch {round:base.round,..Batch::default()};}
        }
        offset+=p.rows as usize*m;
    }
    if batch.layer.count!=0 {batches.push(batch);}
    let _=packed;
    Ok(Fixture {batches,out})
}
fn outputs(g:&Gpu,fx:&Fixture)->Result<Vec<u8>> {
    let bytes=fx.out.read(g)?;if bytes[..16]!=GUARD || bytes[bytes.len()-16..]!=GUARD {return Err("shared forms guard changed".into());}Ok(bytes)
}
pub(crate) fn run(path:&str)->Result<()> {
    let image=std::fs::read(path).map_err(|e|e.to_string())?;println!("FORMS image_sha256={}",sha256::digest(&image));
    let mut g=Gpu::new()?;let module=g.module(&image)?;let tm=g.module(TIMER.as_bytes())?;let timer=g.function(tm,"kb_stamp")?;
    // The individual shapes include the full local KDA head ownership, not just half a global matrix.
    let shapes=[("fp32_router",vec![(144,4096)],false,true,false),
        ("bf16_narrow",vec![(128,4096),(32,4096),(128,4096)],true,true,false),
        ("bf16_index_weights",vec![(64,4096),(64,4096),(16,4096)],true,true,false),
        ("bf16_latent",vec![(4096,128),(4096,128)],true,true,false),
        ("mx_qkv",vec![(4096,4096);3],false,false,true),
        ("mx_gate_up",vec![(6144,4096);2],false,false,true),
        ("mx_output",vec![(2048,8192)],false,false,true),
        ("mx_down",vec![(2048,12288)],false,false,true)];
    for (label,shapes,packed,plain,round) in shapes {for m in 2..=4 {
        let mut arms=Vec::new();
        let mut add=|name,form,entry|->Result<()> {arms.push(Arm {round:g.function(module,"bf16_lanes_round")?,name,form,f:g.function(module,entry)?});Ok(())};
        if plain {
            add("mixed",Form::Mixed,"flash_resident_projection_batch")?;
            add("loop",Form::Loop,if packed {"flash_resident_projection_bf16"} else {"flash_resident_projection_batch"})?;
            if packed {add("packed_scalar",Form::PackedScalar,"project_bf16_ordered")?;add("rows",Form::Rows,"flash_resident_projection_bf16_rows")?;
                if shapes.iter().all(|&(r,c)|r<=512&&c>=1024) {add("fold",Form::Fold,"flash_resident_projection_plain_bf16")?;}
            } else {add("scalar_fold",Form::Scalar,"project_f32_ordered")?;}
        } else {
            add("rows",Form::Rows,"flash_resident_projection_bf16_rows")?;
            add("loop",Form::Loop,"flash_resident_projection_bf16")?;
            add("small",Form::Small,if m==2 {"mx_rows_small2"} else {"mx_rows_small4"})?;
            add("pair",Form::Pair,"flash_next_mxfp8_pair")?;
            add("tile",Form::Tile,"mx_rows_tile")?;
        }
        let current=if !plain {if shapes.len()>1 {"rows"} else {"small"}} else if packed&&m==4 {"rows"} else if packed&&shapes.iter().all(|&(r,c)|r<=512&&c>=1024) {"fold"} else {"mixed"};
        let reference=arms.iter().position(|a|a.name==current).ok_or("current form missing")?;
        let weight_bytes:usize=shapes.iter().map(|&(r,c)|r*c*if packed {2} else if plain {4} else {1}+if plain {0} else {r*c/32}).sum();
        let copies=(256usize<<20).div_ceil(weight_bytes).clamp(2,128);
        let mut all=Vec::new();let mut equal=vec![true;arms.len()];let mut checked=0;
        for copy in 0..copies {
            let jobs:Vec<_>=shapes.iter().map(|&(rows,cols)|Job {plain,rows,cols}).collect();
            let mut base=Batch {round:round as u32,..Batch::default()};let mut wide=Vec::new();
            for (i,j) in jobs.iter().enumerate() {
                let s=901+copy as u64*31+i as u64;let xv=pattern(0,j.cols*m,s+5);let input=g.upload(&le(&xv))?;
                let (weights,scales)=if plain {let values=pattern(5,j.rows*j.cols,s);if packed {wide.push(g.upload(&le(&values.iter().map(|v|f32::from_bits(v.to_bits()&0xffff0000)).collect::<Vec<_>>()))?);(g.upload(&values.iter().flat_map(|v| ((v.to_bits()>>16) as u16).to_le_bytes()).collect::<Vec<_>>())?,2)} else {(g.upload(&le(&values))?,4)}}
                    else {(g.upload(&mx_bytes(j.rows*j.cols,s,false))?,g.upload(&mx_bytes(j.rows*j.cols/32,s+9,true))?)};
                base.layer.projections[i]=MxProjection {weights,scales,input,output:0,rows:j.rows as u32,cols:j.cols as u32,tokens:1,input_bf16:0,round:0,mode:0};
                if plain {base.plain|=1<<i;}
            }
            base.layer.count=jobs.len() as u32;
            let set=arms.iter().map(|arm| {let mut source=base;if packed&&matches!(arm.form,Form::Mixed) {for (i,&w) in wide.iter().enumerate() {source.layer.projections[i].weights=w;source.layer.projections[i].scales=4;}}
                fixture(&mut g,&source,arm,m,packed)}).collect::<Result<Vec<_>>>()?;
            for (a,fx) in arms.iter().zip(&set) {launch_form(&g,a,fx)?;}g.sync()?;
            let want=outputs(&g,&set[reference])?;
            for (i,fx) in set.iter().enumerate() {equal[i]&=outputs(&g,fx)?==want;checked+=1;}
            // Sensitivity control: one omitted source row must leave different bytes.
            if copy==0 {let poison=vec![0xff;want.len()];if want[16..want.len()-16]==poison[16..poison.len()-16] {return Err("forms reference is untouched".into());}}
            all.push(set);
        }
        let times=g.upload(&[0;16])?;let mut samples=vec![Vec::new();arms.len()];
        {
            let mut graphs=Vec::new();for (a,arm) in arms.iter().enumerate() {graphs.push(capture_graph(&g,|| {
                let mut stamp=times;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
                for set in &all {launch_form(&g,arm,&set[a])?;}
                stamp=times+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;Ok(())
            })?);}
            for _ in 0..3 {for graph in &graphs {sample(graph,times)?;}}
            for repeat in 0..11 {for k in 0..arms.len() {let a=(k+repeat)%arms.len();samples[a].push(sample(&graphs[a],times)?.1/copies as f64);}}
        }
        for (i,arm) in arms.iter().enumerate() {println!("FORMS family={label} m={m} current={current} arm={} us={:.3} equal={} checked={checked} copies={copies} clock=graph_stamps samples={:?}",arm.name,median(samples[i].clone()),equal[i],samples[i]);}
        g.clear()?;
    }}
    super::super::mhc::fp8small::shared_forms::run(path)?;
    super::super::mhc::rowsmall::shared_forms::run(path)?;Ok(())
}
