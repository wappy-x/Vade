//! Raw routing planes and native per-call timings. No model or transport load.
use std::{fs,path::Path,time::Instant};
use spark_gpu_probe::{causal_ops::{Op,Shape},driver::with_backend};
use spark_tp::Result;
fn raw(values:&[f32])->Vec<u8> {values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect()}
fn shape(rows:u32,groups:u32,selected_groups:u32,selected:u32,normalize:bool)->Shape {
    Shape {op:Op::Route,heads:groups,dim:selected_groups,taps:u32::from(normalize),rows,cols:selected,epsilon:0.0,bound:2.5}
}
fn edge_cases(full:usize)->Vec<(u32,u32,u32,u32,usize)> {
    let mut cases=vec![(2,1,1,1),(3,1,1,2),(8,2,1,3),(16,4,2,8),(33,3,2,8),
        (64,8,4,8),(128,1,1,8),(288,1,1,8),(288,8,2,8),(511,1,1,8),(512,1,1,8),(1024,8,4,8)]
        .into_iter().map(|(r,h,d,k)|(r,h,d,k,full)).collect::<Vec<_>>();
    // Valid shared routes have rows divisible by groups, at least two experts
    // per group, and rows+groups <= 512. Cover both selection-count endpoints,
    // adjacent counts, select-all, maximum groups, and exact shared capacity.
    // The original cases keep every special-value profile. Added boundaries
    // use zero ties, signed-zero ties, ordered finite values, saturation and
    // mixed special bits, including NaNs and infinities.
    for (r,h) in [(2u32,1u32),(3,1),(4,1),(4,2),(8,4),(16,8),(128,64),
        (256,128),(340,170),(384,128),(504,8),(510,2),(511,1)] {
        for d in [1,h.saturating_sub(1),h] {if d==0 {continue;}let available=d*(r/h);
            for k in [1,2,available-1,available] {
                if !cases.iter().any(|&(a,b,c,e,_)|(a,b,c,e)==(r,h,d,k)) {cases.push((r,h,d,k,5));}
            }
        }
    }
    cases
}
pub fn edges(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let started=Instant::now();let mut receipt=String::new();let mut cases=0usize;
    let patterns=[0u32,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,
        0x3f7fffff,0x3f800001,0x42ae0000,0xc2ae0000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
        0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fffffff];
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<29,|b| {
        for (rows,groups,kept,selected,profiles) in edge_cases(patterns.len()+5) {
            for normalize in [false,true] {let s=shape(rows,groups,kept,selected,normalize);
                for profile in 0..profiles {
                    let source=(0..rows as usize).map(|i|match profile {
                        0=>0.0,1=>-0.0,2=>(i as i32%31-15) as f32/8.0,
                        3=>if i%2==0 {87.0} else {-87.0},4=>f32::from_bits(patterns[i%patterns.len()]),
                        _=>f32::from_bits(patterns[profile-5]),
                    }).collect::<Vec<_>>();
                    let bias=(0..rows as usize).map(|i|if profile<2 {0.0} else if profile==4 {f32::from_bits(patterns[(i*7+3)%patterns.len()])} else {(i as i32%13-6) as f32/32.0}).collect::<Vec<_>>();
                    let input=b.upload_vector(&source)?;let weight=b.upload_vector(&bias)?;let mut previous=None;
                    for queued in [false,true] {
                        let queue=if queued {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
                        for repeat in 1..=3 {
                            let output=b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?;
                            let values=b.read_vector_values(&output)?;
                            if values.len()!=s.lengths().ok_or("route edge shape")?.3 {return Err("route edge extent differs".into());}
                            if values[2*selected as usize..].iter().any(|v|v.to_bits()!=0) {return Err("route scratch is not positive zero".into());}
                            let bytes=raw(&values);if previous.as_ref().is_some_and(|v|v!=&bytes) {return Err("route eager/recorded/repeat bytes differ".into());}previous=Some(bytes.clone());
                            super::write(&out.join(format!("r{rows}-g{groups}-d{kept}-k{selected}-n{normalize}-p{profile}-q{queued}-{repeat}.f32")),&bytes)?;
                        }
                        drop(queue);
                    }
                    cases+=1;
                }
                receipt.push_str(&format!("ROUTE EDGE PASS rows={rows} groups={groups} kept={kept} selected={selected} normalize={normalize} profiles={profiles} eager_and_recorded=true repeats=3 scratch_positive_zero=true\n"));
                super::write(&out.join(format!("partial-r{rows}-g{groups}-d{kept}-k{selected}-n{normalize}")),receipt.as_bytes())?;
            }
        }
        Ok(())
    })?;
    receipt.push_str(&format!("ROUTE EDGES COMPLETE cases={cases} raw_planes={} elapsed_ns={} baseline_comparison_required=true\n",cases*6,started.elapsed().as_nanos()));
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
/// Many seeded routes of one-group shapes (the shape a decode step uses), written as one raw file a shape. Values repeat
/// often (ties), cross zero with both signs, and include equal sums from different parts. The files of two builds must be
/// byte-equal: this is the gate for a change of the selection scan.
pub fn random(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();let mut cases=0usize;
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<29,|b| {
        for (rows,selected) in [(288u32,8u32),(288,1),(288,288),(128,8),(511,8),(16,8),(8,8),(9,1),(12,12),(2,1),(2,2),(3,2)] {
            for normalize in [false,true] {let s=shape(rows,1,1,selected,normalize);let mut all=Vec::new();let mut state=0x9e37_79b9_7f4a_7c15u64^(u64::from(rows)<<32|u64::from(selected));
                let mut next=||{state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(state>>33) as u32};
                for seed in 0..400usize {
                    let style=seed%8;let mut source=Vec::with_capacity(rows as usize);let mut bias=Vec::with_capacity(rows as usize);
                    for i in 0..rows as usize {
                        let r=next();let value=match style {
                            // 0: few distinct values (many ties); 1: copies of earlier entries; 2: tiny steps around zero, both signs;
                            // 3: wide range; 4: near the sigmoid's flat ends; 5: quarter steps; 6: all equal but one; 7: descending then ties.
                            0=>(r%5) as f32-2.0,1=>if i>0 && r%4==0 {source[(r as usize>>8)%i]} else {(r%2001) as f32/250.0-4.0},
                            2=>f32::from_bits((r%64)|if r&0x100!=0 {0x8000_0000} else {0}),3=>(r%160001) as f32/1000.0-80.0,
                            4=>if r&1==0 {30.0+(r%64) as f32} else {-30.0-(r%64) as f32},5=>((r%33) as f32-16.0)/4.0,
                            6=>if i==(seed*7)%rows as usize {1.0} else {0.25},_=>if i<rows as usize/2 {(rows as usize-i) as f32/16.0} else {0.5}};
                        source.push(value);let r=next();
                        bias.push(match style {0|6=>0.0,2=>if r&1==0 {0.0} else {-0.0},5=>((r%9) as f32-4.0)/4.0,_=>((r%257) as f32-128.0)/1024.0});
                    }
                    let input=b.upload_vector(&source)?;let weight=b.upload_vector(&bias)?;
                    let output=b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?;let values=b.read_vector_values(&output)?;
                    if values.len()!=s.lengths().ok_or("route random shape")?.3 {return Err("route random extent differs".into());}
                    // The selected IDs are distinct and inside the recipe (finite inputs only in this mode).
                    let ids=&values[..selected as usize];
                    if ids.iter().any(|id|*id<0.0 || id.fract()!=0.0 || *id>=rows as f32) || (1..ids.len()).any(|at|ids[..at].contains(&ids[at])) {return Err(format!("route random selected a wrong ID set rows={rows} selected={selected} seed={seed}"));}
                    all.extend_from_slice(&raw(&values));cases+=1;
                }
                super::write(&out.join(format!("random-r{rows}-k{selected}-n{normalize}.f32")),&all)?;
                receipt.push_str(&format!("ROUTE RANDOM rows={rows} selected={selected} normalize={normalize} seeds=400 bytes={}\n",all.len()));
            }
        }Ok(())
    })?;
    receipt.push_str(&format!("ROUTE RANDOM COMPLETE cases={cases} baseline_comparison_required=true\n"));
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
pub fn bench(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let cache=b.upload_vector(&vec![1.0;64<<20])?;b.set_profiling(true);
        for (rows,groups,kept,selected) in [(128,1,1,8),(288,1,1,8),(288,8,2,8),(511,1,1,8),(512,1,1,8),(1024,8,4,8)] {
            let s=shape(rows,groups,kept,selected,true);
            let input=b.upload_vector(&(0..rows).map(|i|(i as i32*13%61-30) as f32/16.0).collect::<Vec<_>>())?;
            let weight=b.upload_vector(&(0..rows).map(|i|(i as i32*17%23-11) as f32/32.0).collect::<Vec<_>>())?;
            let mut first=None;
            for cold in [false,true] {for sample in 1..=9 {
                if cold {let pressure=b.round_bf16_lanes(&cache)?;b.synchronize()?;drop(pressure);}
                b.synchronize()?;let before=b.stats();let started=Instant::now();
                let output=b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?;b.synchronize()?;
                let wall=started.elapsed().as_nanos();let gpu=b.stats().since(before).device_kernel_ns[9];
                let actual=raw(&b.read_vector_values(&output)?);if first.as_ref().is_some_and(|old|old!=&actual) {return Err("routing benchmark changed output bytes".into());}first=Some(actual.clone());
                super::write(&out.join(format!("r{rows}-g{groups}-cold{cold}-{sample}.f32")),&actual)?;
                receipt.push_str(&format!("ROUTE SAMPLE rows={rows} groups={groups} kept={kept} selected={selected} cold={cold} sample={sample} kernel_ns={gpu} public_call_wall_ns={wall}\n"));
            }}
        }Ok(())
    })?;super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    #[test]
    fn route_device_fixture_covers_selection_and_group_boundaries() {
        let cases=edge_cases(27);
        for (r,h,d,k,profiles) in cases.iter().copied() {
            assert!(profiles>=5);
            for normalize in [false,true] {assert!(shape(r,h,d,k,normalize).lengths().is_some());}
        }
        for (r,h,d,k) in [(2,1,1,2),(4,1,1,4),(4,2,1,2),(4,2,2,4),(340,170,1,2),
            (340,170,169,338),(340,170,170,340),(384,128,128,384),(504,8,8,504),(510,2,2,510),(511,1,1,511)] {
            assert!(cases.iter().any(|&(a,b,c,e,_)|(a,b,c,e)==(r,h,d,k)));
            assert!(spark_gpu_probe::latent_ops::route_shared_words(shape(r,h,d,k,true)).is_some());
        }
    }
    #[test]
    fn route_device_fixture_keeps_the_scalar_side_of_the_capacity_boundary() {
        let s=shape(512,1,1,8,true);assert!(s.lengths().is_some());
        assert!(spark_gpu_probe::latent_ops::route_shared_words(s).is_none());
        assert!(edge_cases(27).iter().any(|&(r,h,d,k,_)|(r,h,d,k)==(512,1,1,8)));
        assert!(spark_gpu_probe::latent_ops::route_shared_words(shape(511,1,1,8,true)).is_some());
    }
}
