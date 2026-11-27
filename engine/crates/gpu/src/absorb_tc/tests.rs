//! A host model of the kernel bodies (`kernels/src/latent_absorption_tile.rs` and the step entries of
//! `kernels/src/latent_absorption.rs`): the same blocks, warps, lanes, staged
//! chunks, range closes and register places, with the twin's step in place of the instruction. It proves that the
//! index rules give every output word the word function of `word`, for any shape; the device run proves the instruction.
#![allow(clippy::needless_range_loop,clippy::suspicious_assignment_formatting)] // index loops keep the kernel's order; dense style.
use super::*;
use crate::{tc_twin_bf16::step_any,prompt_ops::{Op,Shape},causal_ops::{Op as StepOp,Shape as StepShape}};
fn data(n:usize,seed:usize)->Vec<f32> {(0..n).map(|i|crate::causal_ops::round(((i*37+seed*11)%109) as f32/64.0-0.75)).collect()}
fn pack(wide:&[f32])->Vec<f32> {wide.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect()}
fn pattern(w:&[f32],i:usize)->u32 {let bits=w[i/2].to_bits();if i.is_multiple_of(2) {bits&0xffff} else {bits>>16}}
/// One launch on the host. `output` starts as NaN words, so a word no thread wrote is seen; `owners` counts writes.
fn model(f:Form,x:&[f32],latent:&[f32],selected:&[f32],w:&[f32],blocks:usize,words:usize)->Vec<f32> {
    let mut output=vec![f32::NAN;words];let mut owners=vec![0u8;words];
    for block in 0..blocks {
        let (outer,m0,n0)=f.block(block);if outer>=f.outer() {continue;}
        let (m_all,n_all)=(f.m(),f.n());let (scores,values)=(f.kind==Kind::Scores,f.kind==Kind::Values);
        let count=if f.attention() {selected[f.count_word(outer)] as usize} else {0};
        let id=|slot:usize|if f.table==0 {slot} else {selected[outer*f.table+slot] as usize};
        let terms=f.terms(count);let steps=steps(terms);
        let mut stage=vec![0u32;STAGE_BYTES/4];let mut ids=vec![NO_ROW;ID_BYTES/4];
        // acc[warp][tile][lane][register]
        let zero=[[[[0u32;4];32];TILES];WARPS];let (mut acc,mut p0,mut p2)=(zero,zero,zero);
        let (mut part,mut s,mut chunk)=(0usize,0usize,0usize);
        let add=|a:u32,b:u32|(f32::from_bits(a)+f32::from_bits(b)).to_bits();
        let close=|part:&mut usize,acc:&mut [[[[u32;4];32];TILES];WARPS],p0:&mut [[[[u32;4];32];TILES];WARPS],p2:&mut [[[[u32;4];32];TILES];WARPS]| {
            for warp in 0..WARPS {for tile in 0..TILES {for lane in 0..32 {for reg in 0..4 {
                let (a,h,k)=(&mut acc[warp][tile][lane][reg],&mut p0[warp][tile][lane][reg],&mut p2[warp][tile][lane][reg]);
                match *part {0=>{*h=*a;*a=0;},1=>{*h=add(*h,*a);*a=0;},2=>{*k=*a;*a=0;},_=>{*a=add(*h,add(*k,*a));}}
            }}}}
            *part+=1;
        };
        while chunk*CHUNK<steps {
            let first=chunk*CHUNK;
            for e in 0..CHUNK*TILES*32 {let (cs,tile,l)=(e/(TILES*32),e/32%TILES,e%32);
                for reg in 0..2 {let (kk,nn)=b_place(l,reg);let (k,n)=((first+cs)*16+kk,n0+tile*8+nn);let mut packed=0u32;
                    if n<n_all {
                        if k<terms {packed=x[f.b(outer,k,n)].to_bits()>>16;}
                        if k+1<terms {packed|=x[f.b(outer,k+1,n)].to_bits()&0xffff_0000;}
                    }
                    stage[(stage_at(cs,tile,l)+reg*4)/4]=packed;
                }
            }
            if values {for e in 0..CHUNK*STEP {let slot=first*STEP+e;ids[e]=if slot<count {(id(slot)*f.k) as u32} else {NO_ROW};}}
            let end=(first+CHUNK).min(steps);
            while s<end {
                while part<3 && s>=range_end(steps,part) {close(&mut part,&mut acc,&mut p0,&mut p2);}
                let cs=s-first;
                for warp in 0..WARPS {
                    let mw=m0+warp*16;if mw>=if scores {count} else {m_all} {continue;}
                    // The warp's A registers, then its tile as a matrix.
                    let mut a=[[0u16;16];16];
                    for lane in 0..32 {for reg in 0..4 {
                        let (mm,kk)=a_place(lane,reg);let (m,k)=(mw+mm,s*STEP+kk);
                        let half=|k:usize,kk:usize|->u16 {
                            if scores {if m<count && k<terms {(latent[id(m)*f.k+k].to_bits()>>16) as u16} else {0}}
                            else if values {let at=ids[cs*STEP+kk];if m<m_all && at!=NO_ROW {(latent[at as usize+m].to_bits()>>16) as u16} else {0}}
                            else if m<m_all && k<terms {pattern(w,f.weight(outer,m,k)) as u16} else {0}
                        };
                        a[mm][kk]=half(k,kk);a[mm][kk+1]=half(k+1,kk+1);
                    }}
                    for tile in 0..TILES {
                        if n0+tile*8>=n_all {continue;}
                        let mut b=[[0u16;8];16];
                        for lane in 0..32 {for reg in 0..2 {let (kk,nn)=b_place(lane,reg);let packed=stage[(stage_at(cs,tile,lane)+reg*4)/4];
                            b[kk][nn]=packed as u16;b[kk+1][nn]=(packed>>16) as u16;}}
                        for lane in 0..32 {for reg in 0..4 {let (mm,nn)=c_place(lane,reg);
                            let column:Vec<u16>=(0..16).map(|k|b[k][nn]).collect();
                            acc[warp][tile][lane][reg]=step_any(&a[mm],&column,acc[warp][tile][lane][reg]);
                        }}
                    }
                }
                s+=1;
            }
            chunk+=1;
        }
        while part<4 {close(&mut part,&mut acc,&mut p0,&mut p2);}
        for warp in 0..WARPS {for tile in 0..TILES {for lane in 0..32 {for reg in 0..4 {
            let (mm,nn)=c_place(lane,reg);let (m,n)=(m0+warp*16+mm,n0+tile*8+nn);
            if m<m_all && n<n_all {
                let sum=f32::from_bits(acc[warp][tile][lane][reg]);
                let value=if scores {if m<count {sum*(1.0/crate::causal_ops::root(f.d as f32))} else {0.0}}
                    else if f.kind==Kind::Project && f.table!=0 && selected[f.count_word(n)]==0.0 {0.0} else {crate::latent_absorb::round(sum)};
                output[f.out(outer,m,n)]=value;owners[f.out(outer,m,n)]+=1;
            }
        }}}}
    }
    assert!(owners.iter().all(|&n|n==1),"{f:?}: an output word has no owner or two");
    output
}
/// One launch of a step entry on the host (`kernels/src/latent_absorption.rs`): `dense_step` for query and project,
/// `attention_step` for scores and values. The same blocks, warps, lanes, staged places, buffers, range order and join.
fn model_step(f:Form,x:&[f32],latent:&[f32],selected:&[f32],w:&[f32],blocks:usize,words:usize)->Vec<f32> {
    let mut output=vec![f32::NAN;words];let mut owners=vec![0u8;words];
    let add=|a:u32,b:u32|(f32::from_bits(a)+f32::from_bits(b)).to_bits();
    let high=|v:f32|(v.to_bits()>>16) as u16;
    let column=|b:&[[u16;8];16],n:usize|->Vec<u16> {(0..16).map(|k|b[k][n]).collect()};
    for block in 0..blocks {
        let (outer,m0,n0)=f.step_block(block);if outer>=f.outer() {continue;}
        let (m_all,n_all)=(f.m(),f.n());
        if !f.attention() {
            // Query and project: 2 M tiles (32 rows) x 4 ranges in parallel across 8 warps, column 0 of B in lanes 0..4.
            assert_eq!((n_all,n0),(1,0));
            let terms=f.terms(0);let steps=steps(terms);assert!(steps*32<=STAGE_BYTES);
            let mut stage=vec![[0u32;2];steps*4];
            for (e,staged) in stage.iter_mut().enumerate() {
                let k=(e/4)*16+(e%4)*2;let mut raw=[0u32;4];
                for (i,word) in raw.iter_mut().enumerate() {let kk=k+(i/2)*8+i%2;if kk<terms {*word=x[f.b(outer,kk,0)].to_bits();}}
                *staged=[(raw[0]>>16)|(raw[1]&0xffff_0000),(raw[2]>>16)|(raw[3]&0xffff_0000)];
            }
            let mut acc=[[[0u32;4];32];WARPS];
            for warp in 0..WARPS {
                let (tile_m,part)=(warp/4,warp%4);let mw=m0+tile_m*16;if mw>=m_all {continue;}
                let (start,end)=(steps*part/4,steps*(part+1)/4);
                for s in start..end {
                    let mut a=[[0u16;16];16];
                    for lane in 0..32 {for reg in 0..4 {
                        let (mm,kk)=a_place(lane,reg);let (m,k)=(mw+mm,s*STEP+kk);
                        let half=|k:usize|->u16 {if m<m_all && k<terms {pattern(w,f.weight(outer,m,k)) as u16} else {0}};
                        a[mm][kk]=half(k);a[mm][kk+1]=half(k+1);
                    }}
                    let mut b=[[0u16;8];16];
                    for lane in 0..32 {let regs=if lane<4 {stage[s*4+lane]} else {[0,0]};
                        for (reg,packed) in regs.iter().enumerate() {let (kk,nn)=b_place(lane,reg);b[kk][nn]=*packed as u16;b[kk+1][nn]=(*packed>>16) as u16;}}
                    for lane in 0..32 {for reg in 0..4 {let (mm,nn)=c_place(lane,reg);acc[warp][lane][reg]=step_any(&a[mm],&column(&b,nn),acc[warp][lane][reg]);}}
                }
            }
            for tile_m in 0..STEP_M/16 {
                let mw=m0+tile_m*16;if mw>=m_all {continue;}
                for lane in (0..32).step_by(4) {for (reg,m) in [(0usize,mw+lane/4),(2,mw+lane/4+8)] {
                    assert_eq!(c_place(lane,reg),(m-mw,0));
                    let p=|part:usize|acc[tile_m*4+part][lane][reg];
                    let sum=add(add(p(0),p(1)),add(p(2),p(3)));
                    if m<m_all {let at=f.out(outer,m,0);output[at]=crate::latent_absorb::round(f32::from_bits(sum));owners[at]+=1;}
                }}
            }
            continue;
        }
        // Scores and values: warp `w` owns M tile `w / 4` and range `w % 4`; a chunk is 8 staged steps of each range.
        let scores=f.kind==Kind::Scores;
        let count=selected[0] as usize;assert!(count<=f.c);
        let terms=f.terms(count);let steps=steps(terms);let chunks=steps.div_ceil(4).div_ceil(STEP_CHUNK);
        let mut acc=[[[[0u32;4];32];TILES];WARPS];
        let mut buffers=vec![0u32;STAGE_BYTES/4];
        let stage_b=|c:usize,buffers:&mut Vec<u32>| {
            for thread in 0..THREADS as usize {
                let (warp,lane)=(thread/32,thread%32);let tile=warp%TILES;
                if n0+tile*8>=n_all {continue;}
                for j in 0..16 {
                    let place=thread/128+j*2;let (r,q)=(place/STEP_CHUNK,place%STEP_CHUNK);let s=steps*r/4+c*STEP_CHUNK+q;
                    let mut raw=[0u32;4];
                    if s<steps*(r+1)/4 {for reg in 0..2 {
                        let (kk,nn)=b_place(lane,reg);let (k,n)=(s*STEP+kk,n0+tile*8+nn);
                        if n<n_all {
                            if k<terms {raw[reg*2]=x[f.b(outer,k,n)].to_bits();}
                            if k+1<terms {raw[reg*2+1]=x[f.b(outer,k+1,n)].to_bits();}
                        }
                    }}
                    let at=stage_at(place,tile,lane)/4;
                    buffers[at]=(raw[0]>>16)|(raw[1]&0xffff_0000);buffers[at+1]=(raw[2]>>16)|(raw[3]&0xffff_0000);
                }
            }
        };
        for chunk in 0..chunks {
            // The chunk is staged, then every warp reads it, then the next chunk takes its place.
            stage_b(chunk,&mut buffers);
            for warp in 0..WARPS {
                let (tile_m,part)=(warp/4,warp%4);let mw=m0+tile_m*16;
                if mw>=if scores {count} else {m_all} {continue;}
                let (start,end)=(steps*part/4,steps*(part+1)/4);
                for i in 0..STEP_CHUNK {
                    let s=start+chunk*STEP_CHUNK+i;if s>=end {continue;}
                    let mut a=[[0u16;16];16];
                    for lane in 0..32 {for reg in 0..4 {
                        let (mm,kk)=a_place(lane,reg);let (m,k)=(mw+mm,s*STEP+kk);
                        let half=|k:usize|->u16 {
                            if scores {if m<count && k<terms {high(latent[m*f.k+k])} else {0}}
                            else if m<m_all && k<count {high(latent[k*f.k+m])} else {0}
                        };
                        a[mm][kk]=half(k);a[mm][kk+1]=half(k+1);
                    }}
                    for tile in 0..TILES {
                        if n0+tile*8>=n_all {continue;}
                        let mut b=[[0u16;8];16];
                        for lane in 0..32 {for reg in 0..2 {
                            let (kk,nn)=b_place(lane,reg);
                            let packed=buffers[stage_at(part*STEP_CHUNK+i,tile,lane)/4+reg];
                            b[kk][nn]=packed as u16;b[kk+1][nn]=(packed>>16) as u16;
                        }}
                        for lane in 0..32 {for reg in 0..4 {let (mm,nn)=c_place(lane,reg);
                            acc[warp][tile][lane][reg]=step_any(&a[mm],&column(&b,nn),acc[warp][tile][lane][reg]);}}
                    }
                }
            }
        }
        // The join: range 1 into range 0 and range 3 into range 2, then range 2 into range 0. Range 0 writes.
        for tile_m in 0..STEP_M/16 {for tile in 0..TILES {for lane in 0..32 {for reg in 0..4 {
            let p=|part:usize|acc[tile_m*4+part][tile][lane][reg];
            let sum=f32::from_bits(add(add(p(0),p(1)),add(p(2),p(3))));
            let (mm,nn)=c_place(lane,reg);let (m,n)=(m0+tile_m*16+mm,n0+tile*8+nn);
            if m<m_all && n<n_all {
                let value=if scores {if m<count {sum*(1.0/crate::causal_ops::root(f.d as f32))} else {0.0}} else {crate::latent_absorb::round(sum)};
                output[f.out(outer,m,n)]=value;owners[f.out(outer,m,n)]+=1;
            }
        }}}}
    }
    assert!(owners.iter().all(|&n|n==1),"{f:?}: an output word has no owner or two");
    output
}
fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
#[test]
fn register_places_cover_each_tile_once() {
    let (mut a,mut b,mut c)=([[0u8;16];16],[[0u8;8];16],[[0u8;8];16]);
    for lane in 0..32 {
        for reg in 0..4 {let (m,k)=a_place(lane,reg);a[m][k]+=1;a[m][k+1]+=1;let (m,n)=c_place(lane,reg);c[m][n]+=1;}
        for reg in 0..2 {let (k,n)=b_place(lane,reg);b[k][n]+=1;b[k+1][n]+=1;}
    }
    assert!(a.iter().flatten().chain(b.iter().flatten()).chain(c.iter().flatten()).all(|&n|n==1));
}
#[test]
fn the_kernel_model_gives_every_word_of_the_prompt_form_its_word_function() {
    // Shapes with short last steps, odd widths, more than one block on each axis, and more than one staged chunk.
    for (m,h,d,v,k,r,c,n) in [(1usize,1usize,17usize,9usize,2usize,31usize,13usize,19usize),(3,5,16,32,34,40,33,40),(5,3,48,20,64,700,600,640),(33,2,32,16,130,90,70,75),(2,33,16,16,18,50,17,30)] {
        let q=data(m*h*d,1);let latent=data(r*k,2);let weight=pack(&data(h*(d+v)*k,3));let mut selected=vec![-1.0f32;m*(n+1)];
        for token in 0..m {let count=if token==1 {0} else {(token*37+c)%(c+1)};
            for j in 0..count {selected[token*(n+1)+j]=((j*17+token*5)%r) as f32;}
            if count>1 {selected[token*(n+1)+1]=selected[token*(n+1)];}
            selected[token*(n+1)+n]=count as f32;}
        if m==1 {selected[n]=c as f32;for j in 0..c {selected[j]=((j*7)%r) as f32;}}
        let s=Shape {op:Op::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:r as u32,cols:k as u32,first:c as u32,slots:n as u32,..Shape::new(Op::AbsorbQuery,m).unwrap()};
        let run=|op:Op,x:&[f32]|->Vec<f32> {
            let s=Shape {op,..s};let f=Form::rows(s).unwrap();let (_,_,units,words)=s.lengths().unwrap();
            // The tile's model on every shape; a check pass is served by the rows form (latent_absorb_rows/rows_form_tests.rs).
            let (name,blocks,threads)=crate::latent_absorb_rows::launch(s).unwrap();assert_eq!(threads,THREADS);
            if name=="latent_absorption_tile" {assert_eq!(blocks as usize,f.blocks());} else {assert!(m<=SMALL_TOKENS && name.starts_with("latent_rows_"));}
            let table=if op==Op::AbsorbQuery {&[][..]} else {&selected[..]};
            let have=model(f,x,&latent,table,&weight,f.blocks(),units*words);
            let inputs:[&[f32];5]=match op {Op::AbsorbQuery=>[x,&[],&[],&[],&[]],Op::AbsorbProject=>[x,&selected,&[],&[],&[]],_=>[x,&latent,&selected,&[],&[]]};
            let want=crate::prompt_ops::reference(s,inputs,if matches!(op,Op::AbsorbQuery|Op::AbsorbProject) {&weight} else {&[]}).unwrap();
            assert_eq!(bits(&have),bits(&want),"{op:?} m={m} h={h} k={k}");have
        };
        let absorbed=run(Op::AbsorbQuery,&q);let scores=run(Op::AbsorbScores,&absorbed);
        let probability=crate::prompt_ops::reference(Shape {op:Op::AbsorbSoftmax,cols:c as u32,..s},[&scores,&selected,&[],&[],&[]],&[]).unwrap();
        let mixed=run(Op::AbsorbValues,&probability);run(Op::AbsorbProject,&mixed);
    }
}
#[test]
fn the_kernel_model_gives_every_word_of_the_step_form_its_word_function_on_the_bucket_grid() {
    // Short last steps, odd widths, empty and one-row selections, two N blocks (33 heads), one to five staged chunks.
    for (h,d,v,k,rows,count) in [(1usize,17usize,9usize,2usize,13usize,13usize),(5,16,32,34,300,290),(32,32,16,64,70,0),(33,16,48,130,515,515),(3,48,20,64,1100,1100),(2,32,16,130,40,17),(4,16,16,18,33,1),(2,16,16,66,64,64)] {
        let q=data(h*d,4);let latent=data(rows*k,5);let weight=pack(&data(h*(d+v)*k,6));let counted=[count as f32];
        let s=StepShape {op:StepOp::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:rows as u32,cols:k as u32,epsilon:0.0,bound:0.0};
        let run=|op:StepOp,x:&[f32]|->Vec<f32> {
            let s=StepShape {op,..s};let f=Form::step(s).unwrap();let (_,_,units,words)=s.lengths().unwrap();
            let blocks=crate::causal_ops::absorb_blocks(s).unwrap() as usize;assert!(blocks>=f.step_blocks());
            let dense=matches!(op,StepOp::AbsorbQuery|StepOp::AbsorbProject);
            let have=model_step(f,x,&latent,&counted,&weight,blocks,units*words);
            let inputs:[&[f32];4]=if dense {[x,&[],&[],&[]]} else {[x,&latent,&counted,&[]]};
            let want=crate::causal_ops::reference(s,inputs,if dense {&weight} else {&[]}).unwrap();
            assert_eq!(bits(&have),bits(&want),"{op:?} h={h} k={k} rows={rows}");have
        };
        let absorbed=run(StepOp::AbsorbQuery,&q);let scores=run(StepOp::AbsorbScores,&absorbed);
        let probability=crate::causal_ops::reference(StepShape {op:StepOp::Softmax,taps:0,cols:0,..s},[&scores,&counted,&[],&[]],&[]).unwrap();
        let mixed=run(StepOp::AbsorbValues,&probability);run(StepOp::AbsorbProject,&mixed);
    }
}
#[test]
fn a_selected_row_past_the_staged_offset_range_is_refused_at_launch() {
    let s=Shape {op:Op::AbsorbScores,heads:1,dim:16,taps:16,rows:1<<24,cols:512,first:8,slots:8,..Shape::new(Op::AbsorbScores,1).unwrap()};
    assert!(crate::latent_absorb_rows::launch(s).is_none());assert!(crate::latent_absorb_rows::launch(Shape {rows:1<<22,..s}).is_some());
    let step=StepShape {op:StepOp::AbsorbValues,heads:1,dim:16,taps:16,rows:1<<24,cols:512,epsilon:0.0,bound:0.0};
    assert!(crate::causal_ops::absorb_blocks(step).is_none());assert!(crate::causal_ops::absorb_blocks(StepShape {rows:1<<22,..step}).is_some());
}
