//! A host model of the ROWS FORM (`kernels/src/latent_rows.rs`): the same blocks, warps, lanes, staged places, staged
//! row offsets, range order and join, with the twin's step in place of the instruction. It proves that the index rules
//! give every output word of a check pass (1 to `SMALL_TOKENS` tokens) the word function of `absorb_tc::word` (the word
//! of `latent_absorption_tile`); the device run (`kernel-bench --latent-rows`) proves the instruction.
#![allow(clippy::needless_range_loop)] // index loops keep the kernel's order.
use super::*;
use crate::{absorb_tc::*,tc_twin_bf16::step_any};
fn data(n:usize,seed:usize)->Vec<f32> {(0..n).map(|i|crate::causal_ops::round(((i*37+seed*11)%109) as f32/64.0-0.75)).collect()}
fn pack(wide:&[f32])->Vec<f32> {wide.as_chunks::<2>().0.iter().map(|v|f32::from_bits((v[0].to_bits()>>16)|(v[1].to_bits()&0xffff0000))).collect()}
fn pattern(w:&[f32],i:usize)->u16 {let bits=w[i/2].to_bits();(if i.is_multiple_of(2) {bits&0xffff} else {bits>>16}) as u16}
fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
fn high(v:f32)->u16 {(v.to_bits()>>16) as u16}
fn column(b:&[[u16;8];16],n:usize)->Vec<u16> {(0..16).map(|k|b[k][n]).collect()}
fn add(a:u32,b:u32)->u32 {(f32::from_bits(a)+f32::from_bits(b)).to_bits()}
/// One launch of the rows form on the host, from the name and grid `launch` gave. `output` starts as NaN words, so a
/// word no thread wrote is seen; `owners` counts writes.
fn model_rows(name:&str,blocks:usize,f:Form,x:&[f32],latent:&[f32],selected:&[f32],w:&[f32],words:usize)->Vec<f32> {
    let mut output=vec![f32::NAN;words];let mut owners=vec![0u8;words];
    for block in 0..blocks {
        if name=="latent_rows_dense" {
            // The tile's small body: a block is one head and 128 rows of M; every step of B staged, lane l column l / 4.
            let (outer,m0,_)=f.block(block);if outer>=f.outer() {continue;}
            let (m_all,n_all)=(f.m(),f.n());let terms=f.terms(0);let steps=steps(terms);
            assert!(steps*SMALL_STEP_BYTES<=ROWS_DENSE_STAGE && n_all<=SMALL_TOKENS);
            let mut stage=vec![[0u32;2];steps*32];
            for (e,staged) in stage.iter_mut().enumerate() {
                let (s,l)=(e/32,e%32);let (n,k)=(l/4,s*STEP+(l%4)*2);let mut raw=[0u32;4];
                if n<n_all {for (i,word) in raw.iter_mut().enumerate() {let kk=k+(i/2)*8+i%2;if kk<terms {*word=x[f.b(outer,kk,n)].to_bits();}}}
                *staged=[(raw[0]>>16)|(raw[1]&0xffff_0000),(raw[2]>>16)|(raw[3]&0xffff_0000)];
            }
            for warp in 0..WARPS {
                let mw=m0+warp*16;if mw>=m_all {continue;}
                let (mut acc,mut p0,mut p2)=([[0u32;4];32],[[0u32;4];32],[[0u32;4];32]);let mut part=0usize;
                let close=|part:&mut usize,acc:&mut [[u32;4];32],p0:&mut [[u32;4];32],p2:&mut [[u32;4];32]| {
                    for lane in 0..32 {for reg in 0..4 {let (a,h,k)=(&mut acc[lane][reg],&mut p0[lane][reg],&mut p2[lane][reg]);
                        match *part {0=>{*h= *a;*a=0;},1=>{*h=add(*h,*a);*a=0;},2=>{*k= *a;*a=0;},_=>{*a=add(*h,add(*k,*a));}}}}
                    *part+=1;
                };
                for s in 0..steps {
                    while part<3 && s>=range_end(steps,part) {close(&mut part,&mut acc,&mut p0,&mut p2);}
                    let mut a=[[0u16;16];16];
                    for lane in 0..32 {for reg in 0..4 {
                        let (mm,kk)=a_place(lane,reg);let (m,k)=(mw+mm,s*STEP+kk);
                        let half=|k:usize|->u16 {if m<m_all && k<terms {pattern(w,f.weight(outer,m,k))} else {0}};
                        a[mm][kk]=half(k);a[mm][kk+1]=half(k+1);
                    }}
                    let mut b=[[0u16;8];16];
                    for lane in 0..32 {for (reg,packed) in stage[s*32+lane].iter().enumerate() {let (kk,nn)=b_place(lane,reg);b[kk][nn]=*packed as u16;b[kk+1][nn]=(*packed>>16) as u16;}}
                    for lane in 0..32 {for reg in 0..4 {let (mm,nn)=c_place(lane,reg);acc[lane][reg]=step_any(&a[mm],&column(&b,nn),acc[lane][reg]);}}
                }
                while part<4 {close(&mut part,&mut acc,&mut p0,&mut p2);}
                for lane in 0..32 {for reg in 0..4 {
                    let (mm,n)=c_place(lane,reg);let m=mw+mm;
                    if m<m_all && n<n_all {
                        let empty=f.kind==Kind::Project && f.table!=0 && selected[f.count_word(n)]==0.0;
                        let at=f.out(outer,m,n);output[at]=if empty {0.0} else {crate::latent_absorb::round(f32::from_bits(acc[lane][reg]))};owners[at]+=1;
                    }
                }}
            }
            continue;
        }
        assert_eq!(name,"latent_rows_attention");
        // The step form's attention body for one token: warp `w` owns M tile `w / 4` and range `w % 4`; a chunk is 8
        // staged steps of each range, with the cache row of each of their slots staged as a word offset.
        let (outer,m0,n0)=f.step_block(block);if outer>=f.outer() {continue;}
        let (m_all,n_all)=(f.m(),f.n());let scores=f.kind==Kind::Scores;
        let count=selected[f.count_word(outer)] as usize;assert!(count<=f.c);
        let id=|slot:usize|{let raw=selected[outer*f.table+slot];assert!(raw>=0.0 && (raw as usize)<f.rows);raw as usize};
        let terms=f.terms(count);let steps=steps(terms);let chunks=steps.div_ceil(4).div_ceil(STEP_CHUNK);
        let mut acc=[[[[0u32;4];32];TILES];WARPS];
        let mut buffers=vec![0u32;STAGE_BYTES/4];let mut ids=vec![NO_ROW;ID_BYTES/4];
        for chunk in 0..chunks {
            for thread in 0..THREADS as usize {
                let (warp,lane)=(thread/32,thread%32);let tile=warp%TILES;
                if !scores {for j in 0..2 {
                    let e=thread+j*THREADS as usize;let (r,q,i)=(e/128,e%128/16,e%16);
                    let s=steps*r/4+chunk*STEP_CHUNK+q;let slot=s*STEP+i;
                    ids[e]=if s<steps*(r+1)/4 && slot<count {(id(slot)*f.k) as u32} else {NO_ROW};
                }}
                if n0+tile*8>=n_all {continue;}
                for j in 0..16 {
                    let place=thread/128+j*2;let (r,q)=(place/STEP_CHUNK,place%STEP_CHUNK);let s=steps*r/4+chunk*STEP_CHUNK+q;
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
            for warp in 0..WARPS {
                let (tile_m,part)=(warp/4,warp%4);let mw=m0+tile_m*16;
                if mw>=if scores {count} else {m_all} {continue;}
                let (start,end)=(steps*part/4,steps*(part+1)/4);let base=start+chunk*STEP_CHUNK;
                for i in 0..STEP_CHUNK {
                    let s=base+i;if s>=end {continue;}
                    let mut a=[[0u16;16];16];
                    for lane in 0..32 {for reg in 0..4 {
                        let (mm,kk)=a_place(lane,reg);let (m,k)=(mw+mm,s*STEP+kk);
                        let half=|k:usize,kk:usize|->u16 {
                            if scores {if m<count && k<terms {high(latent[id(m)*f.k+k])} else {0}}
                            else {let at=ids[part*128+(s-base)*STEP+kk];if m<m_all && at!=NO_ROW {high(latent[at as usize+m])} else {0}}
                        };
                        a[mm][kk]=half(k,kk);a[mm][kk+1]=half(k+1,kk+1);
                    }}
                    for tile in 0..TILES {
                        if n0+tile*8>=n_all {continue;}
                        let mut b=[[0u16;8];16];
                        for lane in 0..32 {for reg in 0..2 {
                            let (kk,nn)=b_place(lane,reg);let packed=buffers[stage_at(part*STEP_CHUNK+i,tile,lane)/4+reg];
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
    assert!(owners.iter().all(|&n|n==1),"{name} {f:?}: an output word has no owner or two");
    output
}
/// Every word of a check pass of 1 to 8 tokens, through the rows form's index rules, is the word of the word function:
/// short last steps, odd widths, two N blocks (33 heads), empty, duplicate and reordered selections, values of one to
/// four staged chunks, and the model's own shape with served-like selections.
#[test]
fn the_rows_form_gives_every_word_of_a_check_pass_its_word_function() {
    for (m,h,d,v,k,r,c,n) in [(1usize,1usize,17usize,9usize,2usize,31usize,13usize,19usize),(3,5,16,32,34,40,33,40),(5,3,48,20,64,700,600,640),
        (8,2,32,16,130,90,70,75),(2,33,16,16,18,50,17,30),(4,2,16,16,66,1600,1550,1560),(7,4,16,48,512,300,300,2051),(2,4,256,256,512,260,260,2051)] {
        let q=data(m*h*d,1);let latent=data(r*k,2);let weight=pack(&data(h*(d+v)*k,3));let mut selected=vec![-1.0f32;m*(n+1)];
        for token in 0..m {let count=if token==1 {0} else if token==2 {c} else {(token*37+c)%(c+1)};
            for j in 0..count {selected[token*(n+1)+j]=((j*17+token*5)%r) as f32;}
            if count>1 {selected[token*(n+1)+1]=selected[token*(n+1)];}
            selected[token*(n+1)+n]=count as f32;}
        if m==1 {selected[n]=c as f32;for j in 0..c {selected[j]=((j*7)%r) as f32;}}
        let s=Shape {op:Op::AbsorbQuery,heads:h as u32,dim:d as u32,taps:v as u32,rows:r as u32,cols:k as u32,first:c as u32,slots:n as u32,..Shape::new(Op::AbsorbQuery,m).unwrap()};
        let run=|op:Op,x:&[f32]|->Vec<f32> {
            let s=Shape {op,..s};let f=Form::rows(s).unwrap();let (_,_,units,words)=s.lengths().unwrap();
            let (name,blocks,threads)=launch(s).unwrap();assert_eq!(threads,THREADS);
            assert_eq!((name,blocks as usize),if f.attention() {("latent_rows_attention",f.step_blocks())} else {("latent_rows_dense",f.blocks())},"{op:?} m={m}");
            let have=model_rows(name,blocks as usize,f,x,&latent,&selected,&weight,units*words);
            let inputs:[&[f32];5]=match op {Op::AbsorbQuery=>[x,&[],&[],&[],&[]],Op::AbsorbProject=>[x,&selected,&[],&[],&[]],_=>[x,&latent,&selected,&[],&[]]};
            let want=crate::prompt_ops::reference(s,inputs,if matches!(op,Op::AbsorbQuery|Op::AbsorbProject) {&weight} else {&[]}).unwrap();
            assert_eq!(bits(&have),bits(&want),"{op:?} m={m} h={h} k={k}");have
        };
        let absorbed=run(Op::AbsorbQuery,&q);let scores=run(Op::AbsorbScores,&absorbed);
        let probability=crate::prompt_ops::reference(Shape {op:Op::AbsorbSoftmax,cols:c as u32,..s},[&scores,&selected,&[],&[],&[]],&[]).unwrap();
        let mixed=run(Op::AbsorbValues,&probability);run(Op::AbsorbProject,&mixed);
    }
}
/// The rows form takes 1 to `SMALL_TOKENS` unpacked tokens, query and project only when every step of B fits its stage,
/// and never a launch the tile's 32-bit row offsets refuse.
#[test]
fn the_rows_form_takes_a_check_pass_and_nothing_else() {
    let model=|op,tokens:usize|Shape {op,heads:32,dim:256,taps:256,rows:8192,cols:512,first:2051,slots:2051,..Shape::new(op,tokens).unwrap()};
    for tokens in 1..=SMALL_TOKENS {
        for (op,name) in [(Op::AbsorbQuery,"latent_rows_dense"),(Op::AbsorbScores,"latent_rows_attention"),(Op::AbsorbValues,"latent_rows_attention"),(Op::AbsorbProject,"latent_rows_dense")] {
            let s=model(op,tokens);let f=Form::rows(s).unwrap();
            let blocks=if f.attention() {f.step_blocks()} else {f.blocks()};
            assert_eq!(launch(s),Some((name,blocks as u32,THREADS)),"{op:?} tokens={tokens}");
        }
    }
    // The model's shape: 32 heads x 4 M blocks for query, 32 x 2 for project, 65 M blocks of 32 a token for scores
    // (2,051 score words), 16 a token for values.
    assert_eq!(launch(model(Op::AbsorbQuery,4)).unwrap().1,128);assert_eq!(launch(model(Op::AbsorbProject,4)).unwrap().1,64);
    assert_eq!(launch(model(Op::AbsorbScores,4)).unwrap().1,4*65);assert_eq!(launch(model(Op::AbsorbValues,4)).unwrap().1,4*16);
    for op in [Op::AbsorbQuery,Op::AbsorbScores,Op::AbsorbValues,Op::AbsorbProject] {
        assert_eq!(launch(model(op,SMALL_TOKENS+1)).unwrap().0,"latent_absorption_tile","{op:?}");
        // Rows too many for a 32-bit word offset: no launch at all, as before.
        assert!(launch(Shape {rows:1<<24,..model(op,2)}).is_none(),"{op:?}");
    }
    // Query and project whose steps do not fit the stage keep the tile.
    assert_eq!(launch(Shape {dim:528,..model(Op::AbsorbQuery,2)}).unwrap().0,"latent_absorption_tile");
    assert_eq!(launch(Shape {dim:512,..model(Op::AbsorbQuery,2)}).unwrap().0,"latent_rows_dense");
    assert_eq!(launch(Shape {cols:528,..model(Op::AbsorbProject,2)}).unwrap().0,"latent_absorption_tile");
    // Packed caches never reach the tile or the rows form; the softmax keeps its own kernel.
    assert!(launch(model(Op::AbsorbScoresPacked,2)).is_none() && launch(model(Op::AbsorbValuesPacked,2)).is_none());
    assert_eq!(launch(model(Op::AbsorbSoftmax,2)).map(|l|l.0),Some("prompt_softmax_rows"));
}
/// A launch of the rows form counts under its own name and as the tile it replaces (the served BF16 gate reads the tile's
/// count for a verifying session), as a third-form launch does.
#[test]
fn a_rows_form_launch_counts_as_the_tile_it_replaces() {
    let count=|name:&str|crate::prompt_launches::line().split(' ').find_map(|f|f.strip_prefix(&format!("{name}="))?.parse::<u64>().ok()).unwrap();
    for name in ["latent_rows_dense","latent_rows_attention"] {
        let (tile,own)=(count("latent_absorption_tile"),count(name));crate::prompt_launches::record(name);
        assert!(count("latent_absorption_tile")>tile && count(name)>own,"{name}");
    }
}
