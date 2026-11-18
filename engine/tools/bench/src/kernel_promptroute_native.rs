//! Native descriptor, calibration, compact dispatch and complete output byte proof.
//! PAIRED PARTS: every case of two or more sets also builds the merged work list (one interval an expert over all its
//! sets' tokens) and runs the product with `merge` = sets; every output word must equal the rectangular launch's.
//! `PROMPT_NATIVE TIME` lines time the old and the paired launch at the model's expert shape (288 experts, two,
//! three and four sets of 1,024 rows, eight picks) in the order base, paired, paired, base, after checking that both
//! store the same words. Three sets are the 3,072-row prompt lease (THREE RUNS), four the 4,096-row one (FOUR RUNS).
#![forbid(unsafe_code)]
use super::{Gpu,H,Result,ptr,sha256,promptroute as meta};
use std::time::Instant;
use meta::{abi,words};
#[allow(dead_code)] #[path="../../../crates/gpu/src/tc_abi.rs"] mod tc;
fn part_bytes(p:tc::NvPart)->Vec<u8> {
    [p.weight,p.scales,p.output].into_iter().flat_map(u64::to_le_bytes)
        .chain([p.first,p.tokens,p.global.to_bits(),p.activation.to_bits()].into_iter().flat_map(u32::to_le_bytes)).collect()
}
fn headers(v:&[abi::NativeSet])->Vec<u8> {
    v.iter().flat_map(|s|[s.counts,s.starts,s.work,s.output,s.activations,s.order].into_iter().flat_map(u64::to_le_bytes)
        .chain([s.first,s.pairs,s.work_capacity,s.pad].into_iter().flat_map(u32::to_le_bytes))).collect()
}
fn random(n:usize,seed:&mut u64)->Vec<u8> {(0..n).map(|_|{*seed^=*seed<<13;*seed^=*seed>>7;*seed^=*seed<<17;*seed as u8}).collect()}
fn expected(templates:&[tc::NvPart],refs:&[abi::Reference],outputs:&[u64],starts:&[usize],rows:u32,bt:u32,capacity:usize)->Vec<Vec<u8>> {
    let mut parts=Vec::new();let mut work=Vec::new();let mut acts=refs.iter().map(|r|vec![0u8;r.order.len()*4]).collect::<Vec<_>>();
    for (expert,&template) in templates.iter().enumerate() {for (set,r) in refs.iter().enumerate() {
        let (first,count)=(r.starts[expert],r.counts[expert]);
        parts.extend(part_bytes(tc::NvPart {output:outputs[set]+16+4*u64::from(first)*u64::from(rows),first:starts[set] as u32+first,tokens:count,..template}));
        for at in first..first+count {acts[set][4*at as usize..4*at as usize+4].copy_from_slice(&template.activation.to_le_bytes());}
        for tile in 0..count.div_ceil(bt) {work.extend([expert as u32*refs.len() as u32+set as u32,tile]);}
    }}
    work.resize(2*capacity,u32::MAX);let mut out=vec![parts,words(work)];out.extend(acts);out
}
/// PAIRED PARTS: the merged work list, one interval an expert of ceil(sum of its set counts / bt) records naming part
/// e*sets, then empty records up to the capacity.
fn merged_work(refs:&[abi::Reference],experts:u32,sets:u32,bt:u32,capacity:usize)->Vec<u8> {
    let mut work=Vec::new();
    for e in 0..experts as usize {let total:u32=refs.iter().map(|r|r.counts[e]).sum();for tile in 0..total.div_ceil(bt) {work.extend([e as u32*sets,tile]);}}
    work.resize(2*capacity,u32::MAX);words(work)
}
fn launch_product(g:&Gpu,f:H,p:&mut tc::NvRows,most:u32,bt:u32)->Result<()> {
    let grid=if p.work==0 {[most.div_ceil(bt),p.rows/128,p.count]} else {[p.rows/128,p.work_count,1]};
    g.launch(f,grid,256,0,&mut [ptr(p)])
}
fn products(g:&Gpu,base:&[u64],candidate:&[u64],sizes:&[usize])->Result<()> {
    for ((a,x),&n) in base.iter().zip(candidate).zip(sizes) {
        let (a,x)=(g.read(*a,n+32)?,g.read(*x,n+32)?);
        if a[..16]!=[0xa5;16] || a[n+16..]!=[0xa5;16] || x[..16]!=[0xa5;16] || x[n+16..]!=[0xa5;16] {return Err("PROMPT_NATIVE output guard changed".into());}
        if a!=x {return Err("PROMPT_NATIVE mapped product differs from rectangular product".into());}
        if a[16..n+16].chunks_exact(4).any(|v|v==[0xa5;4]) {return Err("PROMPT_NATIVE product left an unwritten word".into());}
    }Ok(())
}
#[allow(clippy::too_many_arguments)]
fn case(g:&mut Gpu,functions:[H;4],m:u32,picks:u32,experts:u32,sets:u32,chains:u32,rows:u32,cols:u32,pattern:u32)->Result<usize> {
    let [count,tables,emit,product]=functions;let bt=if chains==1 {128} else {64};let mut seed=17+u64::from(m+experts+sets+chains+rows+cols+pattern);
    let mut templates=Vec::new();for e in 0..experts {
        let w=g.upload(&random((rows*cols/2) as usize,&mut seed))?;let s=g.upload(&vec![0x38+(e%3) as u8;(rows*cols/16) as usize])?;
        templates.push(tc::NvPart {weight:w,scales:s,output:0,first:0,tokens:0,global:0.0078125*(e+1) as f32,activation:0.125*(e%3+1) as f32});
    }
    let template=g.upload(&templates.iter().flat_map(|p|part_bytes(*p)).collect::<Vec<_>>())?;
    let (mut routes,mut refs,mut starts,mut sizes,mut baselines,mut candidates,mut acts)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());let mut total=0usize;let mut capacity=0usize;
    for set in 0..sets {
        let tokens=m.saturating_sub(set%3).max(1);let shape=abi::Shape {rows:tokens,picks,experts};let ids=(0..tokens).flat_map(|t|(0..picks).map(move |p|if pattern==0 {((t*13+p*3+set*5)%experts) as f32} else {p as f32})).collect::<Vec<_>>();
        let r=abi::reference(&ids,shape)?;let (mut p,owners,_)=meta::fixture(g,shape,&ids)?;meta::launch(g,count,tables,&mut p)?;g.sync()?;meta::compare(g,&owners,&meta::expected(&r))?;
        starts.push(total);total+=shape.pairs();capacity+=shape.work_capacity(bt);let n=shape.pairs()*rows as usize*4;sizes.push(n);
        baselines.push(g.upload(&vec![0xa5;n+32])?);candidates.push(g.upload(&vec![0xa5;n+32])?);acts.push(g.upload(&vec![0xa5;shape.pairs()*4+32])?);routes.push(p);refs.push(r);
    }
    let want=expected(&templates,&refs,&candidates,&starts,rows,bt,capacity);let parts=g.upload(&vec![0xa5;want[0].len()+32])?;let work=g.upload(&vec![0xa5;want[1].len()+32])?;
    let mut owners=vec![parts,work];owners.extend(acts.iter().copied());
    let h=routes.iter().zip(&starts).zip(&candidates).zip(&acts).map(|(((p,&first),&output),&activations)|abi::NativeSet {counts:p.counts,starts:p.starts,work:if bt==128 {p.work128} else {p.work64},output:output+16,activations:activations+16,order:p.pairs,
        first:first as u32,pairs:p.shape.pairs() as u32,work_capacity:p.shape.work_capacity(bt) as u32,pad:0}).collect::<Vec<_>>();
    let mut build=abi::NativeBuild {templates:template,sets:g.upload(&headers(&h))?,table:parts+16,work:work+16,rowmap:0,count:sets,experts,rows,tokens:bt,capacity:capacity as u32,merge:0};
    g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build)])?;g.sync()?;meta::compare(g,&owners,&want)?;
    let baseline=expected(&templates,&refs,&baselines,&starts,rows,bt,capacity);let base_table=g.upload(&baseline[0])?;
    let codes=g.upload(&random(total*cols as usize/2,&mut seed))?;let scales=g.upload(&vec![0x3a;total*cols as usize/16])?;
    let mut a=tc::NvRows {table:base_table,codes,scales,count:experts*sets,rows,cols,layout:1,skip:1,raw:0,chains,work:0,work_count:0,merge:0,rowmap:0};
    let mut x=tc::NvRows {table:parts+16,work:work+16,work_count:capacity as u32,..a};let most=refs.iter().flat_map(|r|r.counts.iter()).copied().max().unwrap();
    launch_product(g,product,&mut a,most,bt)?;launch_product(g,product,&mut x,most,bt)?;g.sync()?;products(g,&baselines,&candidates,&sizes)?;
    for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}g.upload_at(work+16,&[0xff;8])?;launch_product(g,product,&mut x,most,bt)?;g.sync()?;
    if products(g,&baselines,&candidates,&sizes).is_ok() {return Err("PROMPT_NATIVE wrong-work control was not detected".into());}
    let mut changed=templates.clone();changed[0].activation=2000.0;g.upload_at(template+36,&2000.0f32.to_le_bytes())?;
    g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build)])?;g.sync()?;let altered=expected(&changed,&refs,&candidates,&starts,rows,bt,capacity);meta::compare(g,&owners,&altered)?;
    if meta::compare(g,&owners,&want).is_ok() {return Err("PROMPT_NATIVE wrong-calibration control was not detected".into());}
    g.upload_at(template+36,&templates[0].activation.to_le_bytes())?;g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build)])?;
    for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}launch_product(g,product,&mut x,most,bt)?;g.sync()?;meta::compare(g,&owners,&want)?;products(g,&baselines,&candidates,&sizes)?;
    let mut bytes=want.iter().map(Vec::len).sum::<usize>()+sizes.iter().sum::<usize>();
    let mut paired="";let mut served=x;
    if sets>1 {
        // PAIRED PARTS: the same part records and calibration, the merged work list, and every product word of every set
        // equal to the rectangular launch's; a wrong work record must show; a rebuild and repeat must pass again.
        let cap2=abi::merged_work_capacity(experts,total,bt);if cap2>capacity {return Err("PROMPT_NATIVE paired capacity exceeds the sets' capacity".into());}
        let work_want=merged_work(&refs,experts,sets,bt,cap2);let work2=g.upload(&vec![0xa5;work_want.len()+32])?;
        let mut build2=abi::NativeBuild {work:work2+16,capacity:cap2 as u32,merge:1,..build};
        let mut owners2=vec![parts,work2];owners2.extend(acts.iter().copied());
        let mut want2=vec![want[0].clone(),work_want.clone()];want2.extend(want[2..].iter().cloned());
        g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build2)])?;g.sync()?;meta::compare(g,&owners2,&want2)?;
        let mut y=tc::NvRows {work:work2+16,work_count:cap2 as u32,merge:sets,..x};
        for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}
        launch_product(g,product,&mut y,most,bt)?;g.sync()?;products(g,&baselines,&candidates,&sizes)?;
        for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}g.upload_at(work2+16,&[0xff;8])?;launch_product(g,product,&mut y,most,bt)?;g.sync()?;
        if products(g,&baselines,&candidates,&sizes).is_ok() {return Err("PROMPT_NATIVE paired wrong-work control was not detected".into());}
        g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build2)])?;g.sync()?;meta::compare(g,&owners2,&want2)?;
        for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}launch_product(g,product,&mut y,most,bt)?;g.sync()?;products(g,&baselines,&candidates,&sizes)?;
        bytes+=work_want.len()+sizes.iter().sum::<usize>();paired=" PAIRED_PASS PAIRED_WRONG_WORK_PASS PAIRED_REPEAT_PASS";served=y;
    }
    // DIRECT PLACE: the row map build, then the product (paired when there are sets to pair) into fresh outputs in
    // original pair order; every word must equal ExpertPlaceTile's reference applied to the rectangular output.
    let rowmap=g.upload(&vec![0xa5;total*8+32])?;let mut build3=abi::NativeBuild {rowmap:rowmap+16,..build};
    g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build3)])?;g.sync()?;meta::compare(g,&owners,&want)?;
    let mut map=Vec::with_capacity(total*8);
    for (set,r) in refs.iter().enumerate() {for &pair in &r.pairs {map.extend((candidates[set]+16+4*(pair as u64)*u64::from(rows)).to_le_bytes());}}
    meta::compare(g,&[rowmap],&[map])?;
    let mut z=tc::NvRows {rowmap:rowmap+16,..served};
    for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}launch_product(g,product,&mut z,most,bt)?;g.sync()?;
    for (set,r) in refs.iter().enumerate() {
        let base=g.read(baselines[set]+16,sizes[set])?;let got=g.read(candidates[set],sizes[set]+32)?;
        if got[..16]!=[0xa5;16] || got[sizes[set]+16..]!=[0xa5;16] {return Err("PROMPT_NATIVE direct place guard changed".into());}
        let n=r.pairs.len();let d=rows as usize;let mut want_bytes=vec![0u8;sizes[set]];
        for pair in 0..n {let (earlier,tokens,at)=(r.placed[3*pair] as usize,r.placed[3*pair+1] as usize,r.placed[3*pair+2] as usize);
            for lane in 0..d {let from=4*(earlier*d+lane*tokens+at);want_bytes[4*(pair*d+lane)..4*(pair*d+lane)+4].copy_from_slice(&base[from..from+4]);}}
        if got[16..sizes[set]+16]!=want_bytes[..] {return Err(format!("PROMPT_NATIVE direct place differs from the placed rectangular output set={set}"));}
    }
    // Wrong map: swap the first two row addresses (still a permutation of the set's rows, so still one writer a word);
    // the stored words must then differ from the reference placement. Rebuild and repeat: equal again.
    let direct_words=|g:&Gpu|->Result<bool> {
        for (set,r) in refs.iter().enumerate() {
            let base=g.read(baselines[set]+16,sizes[set])?;let got=g.read(candidates[set]+16,sizes[set])?;let d=rows as usize;
            for pair in 0..r.pairs.len() {let (earlier,tokens,at)=(r.placed[3*pair] as usize,r.placed[3*pair+1] as usize,r.placed[3*pair+2] as usize);
                for lane in 0..d {let from=4*(earlier*d+lane*tokens+at);if got[4*(pair*d+lane)..4*(pair*d+lane)+4]!=base[from..from+4] {return Ok(false);}}}
        }Ok(true)
    };
    let mut wrong_map="";
    if total>1 {
        let first=g.read(rowmap+16,16)?;let mut swapped=first[8..16].to_vec();swapped.extend_from_slice(&first[..8]);g.upload_at(rowmap+16,&swapped)?;
        for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}launch_product(g,product,&mut z,most,bt)?;g.sync()?;
        if direct_words(g)? {return Err("PROMPT_NATIVE wrong-map control was not detected".into());}
        g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build3)])?;g.sync()?;
        for (&out,&n) in candidates.iter().zip(&sizes) {g.upload_at(out+16,&vec![0xa5;n])?;}launch_product(g,product,&mut z,most,bt)?;g.sync()?;
        if !direct_words(g)? {return Err("PROMPT_NATIVE direct place repeat after the wrong map differs".into());}
        wrong_map=" DIRECT_WRONG_MAP_PASS DIRECT_REPEAT_PASS";
    }
    bytes+=total*8+sizes.iter().sum::<usize>();let direct=" DIRECT_PLACE_PASS";
    println!("PROMPT_NATIVE BYTE_PASS m={m} picks={picks} experts={experts} sets={sets} chains={chains} rows={rows} cols={cols} pattern={pattern} bytes={bytes} WRONG_WORK_PASS WRONG_CALIBRATION_PASS REPEAT_PASS GUARDS_PASS{paired}{direct}{wrong_map}");
    g.clear()?;Ok(bytes)
}
fn median(mut v:Vec<f64>)->f64 {v.sort_by(f64::total_cmp);v[v.len()/2]}
/// PAIRED PARTS at the model's expert shape: 288 experts, `sets` sets of 1,024 rows, eight distinct picks a row (uniform,
/// or skewed toward low expert numbers). Both launches store into the same outputs; their words must be equal and
/// complete before any time is printed. Times are wall milliseconds of launch + drain, base/paired/paired/base x 9.
fn timing(g:&mut Gpu,functions:[H;4],chains:u32,rows:u32,cols:u32,skew:bool,sets:u32)->Result<()> {
    let [count,tables,emit,product]=functions;let (m,picks,experts)=(1024u32,8u32,288u32);let bt=if chains==1 {128} else {64};
    let mut seed=0x9e37_79b9_7f4a_7c15u64^u64::from(chains*7+u32::from(skew));
    let mut templates=Vec::new();for e in 0..experts {
        let w=g.upload(&random((rows*cols/2) as usize,&mut seed))?;let s=g.upload(&vec![0x38+(e%3) as u8;(rows*cols/16) as usize])?;
        templates.push(tc::NvPart {weight:w,scales:s,output:0,first:0,tokens:0,global:0.0078125*(e%7+1) as f32,activation:0.125*(e%3+1) as f32});
    }
    let template=g.upload(&templates.iter().flat_map(|p|part_bytes(*p)).collect::<Vec<_>>())?;
    let (mut h,mut outs,mut sizes,mut refs)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());let (mut total,mut capacity)=(0usize,0usize);
    for _ in 0..sets {
        let mut ids=Vec::with_capacity((m*picks) as usize);
        for _ in 0..m {let mut chosen:Vec<u32>=Vec::new();while chosen.len()<picks as usize {
            let r=u64::from_le_bytes(random(8,&mut seed).try_into().map_err(|_|"seed")?);
            let e=if skew {let u=(r%1_000_000) as f64/1e6;((u*u*f64::from(experts)) as u32).min(experts-1)} else {(r%u64::from(experts)) as u32};
            if !chosen.contains(&e) {chosen.push(e);}
        } ids.extend(chosen.iter().map(|&e|e as f32));}
        let shape=abi::Shape {rows:m,picks,experts};let r=abi::reference(&ids,shape)?;let (mut p,_,_)=meta::fixture(g,shape,&ids)?;meta::launch(g,count,tables,&mut p)?;g.sync()?;
        let n=shape.pairs()*rows as usize*4;let out=g.upload(&vec![0xa5;n+32])?;
        h.push(abi::NativeSet {counts:p.counts,starts:p.starts,work:if bt==128 {p.work128} else {p.work64},output:out+16,activations:0,order:p.pairs,first:total as u32,pairs:shape.pairs() as u32,work_capacity:shape.work_capacity(bt) as u32,pad:0});
        total+=shape.pairs();capacity+=shape.work_capacity(bt);outs.push(out);sizes.push(n);refs.push(r);
    }
    let cap2=abi::merged_work_capacity(experts,total,bt);
    let parts=g.upload(&vec![0xa5;(experts*sets) as usize*40+32])?;let work=g.upload(&vec![0xa5;capacity*8+32])?;let work2=g.upload(&vec![0xa5;cap2*8+32])?;
    let mut build=abi::NativeBuild {templates:template,sets:g.upload(&headers(&h))?,table:parts+16,work:work+16,rowmap:0,count:sets,experts,rows,tokens:bt,capacity:capacity as u32,merge:0};
    let mut build2=abi::NativeBuild {work:work2+16,capacity:cap2 as u32,merge:1,..build};
    g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build)])?;g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build2)])?;g.sync()?;
    let codes=g.upload(&random(total*cols as usize/2,&mut seed))?;let scales=g.upload(&vec![0x3a;total*cols as usize/16])?;
    let mut x=tc::NvRows {table:parts+16,codes,scales,count:experts*sets,rows,cols,layout:1,skip:1,raw:0,chains,work:work+16,work_count:capacity as u32,merge:0,rowmap:0};
    let mut y=tc::NvRows {work:work2+16,work_count:cap2 as u32,merge:sets,..x};
    let read=|g:&Gpu|outs.iter().zip(&sizes).map(|(&o,&n)|g.read(o,n+32)).collect::<Result<Vec<_>>>();
    launch_product(g,product,&mut x,0,bt)?;g.sync()?;let base_words=read(g)?;
    for (&o,&n) in outs.iter().zip(&sizes) {g.upload_at(o+16,&vec![0xa5;n])?;}
    launch_product(g,product,&mut y,0,bt)?;g.sync()?;let paired_words=read(g)?;
    if base_words!=paired_words {return Err(format!("PROMPT_NATIVE TIME paired words differ chains={chains} skew={skew}"));}
    for (v,&n) in base_words.iter().zip(&sizes) {if v[..16]!=[0xa5;16] || v[n+16..]!=[0xa5;16] || v[16..n+16].as_chunks::<4>().0.iter().any(|w|*w==[0xa5;4]) {return Err("PROMPT_NATIVE TIME output incomplete or guard changed".into());}}
    let (mut base,mut paired)=(Vec::new(),Vec::new());
    for _ in 0..9 {for candidate in [false,true,true,false] {
        let start=Instant::now();launch_product(g,product,if candidate {&mut y} else {&mut x},0,bt)?;g.sync()?;
        let ms=start.elapsed().as_secs_f64()*1e3;if candidate {paired.push(ms);} else {base.push(ms);}
    }}
    // DIRECT PLACE (second matrices, as served): the paired launch with the row map, into the same outputs in original
    // pair order; its words must equal the paired words placed by the reference rule.
    let mut direct_ms=0.0;
    if chains==4 {
        let rowmap=g.upload(&vec![0xa5;total*8+32])?;let mut build3=abi::NativeBuild {rowmap:rowmap+16,..build};
        g.launch(emit,[experts,1,1],256,0,&mut [ptr(&mut build3)])?;g.sync()?;
        let mut z=tc::NvRows {rowmap:rowmap+16,..y};
        for (&o,&n) in outs.iter().zip(&sizes) {g.upload_at(o+16,&vec![0xa5;n])?;}
        launch_product(g,product,&mut z,0,bt)?;g.sync()?;let direct_words=read(g)?;
        for (set,r) in refs.iter().enumerate() {
            let d=rows as usize;let mut want_bytes=vec![0u8;sizes[set]];
            for pair in 0..r.pairs.len() {let (earlier,tokens,at)=(r.placed[3*pair] as usize,r.placed[3*pair+1] as usize,r.placed[3*pair+2] as usize);
                for lane in 0..d {let from=16+4*(earlier*d+lane*tokens+at);want_bytes[4*(pair*d+lane)..4*(pair*d+lane)+4].copy_from_slice(&paired_words[set][from..from+4]);}}
            if direct_words[set][16..sizes[set]+16]!=want_bytes[..] {return Err(format!("PROMPT_NATIVE TIME direct place words differ set={set}"));}
        }
        let mut times=Vec::new();
        for _ in 0..9 {for direct in [false,true,true,false] {
            let start=Instant::now();launch_product(g,product,if direct {&mut z} else {&mut y},0,bt)?;g.sync()?;
            if direct {times.push(start.elapsed().as_secs_f64()*1e3);}
        }}
        direct_ms=median(times);
    }
    let (b,p)=(median(base),median(paired));let live=|tile:u32|(0..experts as usize).map(|e|refs.iter().map(|r:&abi::Reference|r.counts[e].div_ceil(tile)).sum::<u32>()).sum::<u32>();
    let merged=(0..experts as usize).map(|e|refs.iter().map(|r:&abi::Reference|r.counts[e]).sum::<u32>().div_ceil(bt)).sum::<u32>();
    let most=(0..experts as usize).map(|e|refs.iter().map(|r:&abi::Reference|r.counts[e]).sum::<u32>()).max().unwrap_or(0);
    println!("PROMPT_NATIVE TIME chains={chains} rows={rows} cols={cols} skew={skew} experts={experts} sets={sets} m={m} picks={picks} base_ms={b:.4} paired_ms={p:.4} ratio={:.4} base_tiles={} paired_tiles={merged} most_tokens={most} direct_ms={direct_ms:.4} WORDS_EQUAL order=AXXA x9",p/b,live(bt));
    g.clear()?;Ok(())
}
pub(crate) fn run(primary:&str,second:&str)->Result<()> {
    let image=std::fs::read(primary).map_err(|e|e.to_string())?;let tc_image=std::fs::read(second).map_err(|e|e.to_string())?;
    println!("PROMPT_NATIVE primary_sha256={} tc_sha256={}",sha256::digest(&image),sha256::digest(&tc_image));
    // The slot table also changes the one-part (S = 1) addressing of every product form, so each form is first held
    // to the independent CPU twin of the instruction (`--fp4-cohort ... gates`, every edge word, both word functions),
    // in its own process: a kernel trap there cannot poison this context.
    for mode in ["gates","gates:q"] {
        let output=std::process::Command::new(std::env::current_exe().map_err(|e|e.to_string())?).args(["--fp4-cohort",primary,second,mode]).output().map_err(|e|e.to_string())?;
        let stdout=String::from_utf8_lossy(&output.stdout);print!("{stdout}");
        if !output.status.success() || !stdout.contains("FP4COHORT GATES PASS") {return Err(format!("PROMPT_NATIVE twin gates failed mode={mode} status={} stderr={}",output.status,String::from_utf8_lossy(&output.stderr)));}
        println!("PROMPT_NATIVE TWIN_GATES_PASS mode={mode}");
    }
    let mut g=Gpu::new()?;let a=g.module(&image)?;let b=g.module(&tc_image)?;let count=g.function(a,"prompt_route_count")?;let tables=g.function(a,"prompt_route_tables")?;let emit=g.function(a,"prompt_route_native")?;
    let mut cases=0;let mut bytes=0;
    for m in [1u32,7,33,129] {for picks in [1u32,4] {for experts in [8u32,17] {for sets in [1u32,2,3,4,8] {for chains in [1u32,4] {for (rows,cols) in [(128u32,256u32),(256,512)] {for pattern in 0..2 {
        let product=g.function(b,if chains==1 {"fp4_cohort"} else {"fp4_cohort_t64"})?;bytes+=case(&mut g,[count,tables,emit,product],m,picks,experts,sets,chains,rows,cols,pattern)?;cases+=1;
    }}}}}}}
    for sets in [2u32,3,4] {for chains in [1u32,4] {let product=g.function(b,if chains==1 {"fp4_cohort"} else {"fp4_cohort_t64"})?;bytes+=case(&mut g,[count,tables,emit,product],1024,8,257,sets,chains,1024,4096,0)?;cases+=1;}}
    for (m,picks,experts,sets,rows,cols) in [(1024u32,8u32,8u32,8u32,128u32,256u32),(1,1,1024,8,128,256),(33,4,8,4,128,8192)] {
        for chains in [1u32,4] {let product=g.function(b,if chains==1 {"fp4_cohort"} else {"fp4_cohort_t64"})?;bytes+=case(&mut g,[count,tables,emit,product],m,picks,experts,sets,chains,rows,cols,0)?;cases+=1;}
    }
    for sets in [2u32,3,4] {for chains in [1u32,4] {for skew in [false,true] {
        let product=g.function(b,if chains==1 {"fp4_cohort"} else {"fp4_cohort_t64"})?;let (rows,cols)=if chains==1 {(1024,4096)} else {(2048,2048)};
        timing(&mut g,[count,tables,emit,product],chains,rows,cols,skew,sets)?;
    }}}
    if std::fs::read(primary).map_err(|e|e.to_string())?!=image || std::fs::read(second).map_err(|e|e.to_string())?!=tc_image {return Err("PROMPT_NATIVE image changed during proof".into());}
    println!("PROMPT_NATIVE PASS cases={cases} bytes={bytes} wrong_work={cases} wrong_calibration={cases} repeats={cases}");Ok(())
}
