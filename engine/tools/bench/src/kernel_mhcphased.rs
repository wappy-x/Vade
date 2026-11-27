//! Four compiled arms, dirty retained scratch, and fresh arguments through cached graphs.
#![forbid(unsafe_code)]
use super::*;
use super::super::super::{capture_graph,sample,Graph};
use crate::{byte_check::{self,Span},mhc_params::{self,FusedBoundary,FUSED_BLOCKS,FUSED_THREADS,FUSED_SCRATCH_WORDS,FUSED_ROWS_MAX,fused_row,fused_strip_index}};
use std::collections::BTreeMap;
const COMMAND_BYTES:usize=1040;
const KINDS:[u32;4]=[38,61,62,63];
const SEQUENCE:[u32;7]=[8,5,4,3,4,5,8];
struct Entries {fused:[H;2],phases:[H;3],recorded:[H;4]}
fn outputs(c:&Cell)->[&Buffer;4] {[&c.residual,&c.mixing,&c.coefficients,&c.other]}
fn inputs(c:&Cell)->[&Buffer;7] {[&c.input,&c.previous,&c.previous_coefficients,&c.weights,&c.base,&c.scale,&c.norm]}
fn span(p:&Buffer)->Span {Span {address:p.address,bytes:p.bytes}}
fn command(kind:u32,mut p:FusedBoundary)->Vec<u8> {
    p.reserved=KINDS.iter().position(|&k|k==kind).expect("boundary kind") as u32;
    let mut bytes=vec![0;COMMAND_BYTES];bytes[..4].copy_from_slice(&kind.to_ne_bytes());
    let payload=mhc_params::fused_payload(p);bytes[16..16+payload.len()].copy_from_slice(&payload);bytes
}
fn plan(p:FusedBoundary,phases:bool)->Vec<(usize,u32,FusedBoundary)> {
    let rows=p.boundary.shape.rows;
    if phases {(1..=3).map(|phase|(phase,if phase==3 {rows} else {rows*FUSED_BLOCKS as u32},p)).collect()}
    else {let mut one=p;one.boundary.shape.rows=1;(0..rows).map(|r|(0,FUSED_BLOCKS as u32,fused_row(one,r as usize))).collect()}
}
fn direct(g:&Gpu,e:&Entries,p:FusedBoundary,arm:usize,phases:bool,broken:bool)->Result<()> {
    for (kind,blocks,mut args) in plan(p,phases) {
        // Negative control: an actual missing finish launch, with the other phases still running.
        if broken && kind==3 {continue;}
        let entry=if kind==0 {e.fused[arm.min(1)]} else {e.phases[kind-1]};
        g.launch(entry,[blocks,1,1],FUSED_THREADS as u32,0,&mut [ptr(&mut args)])?;
    }Ok(())
}
fn refresh(g:&Gpu,c:&Cell,f:&Fixture,norm_bf16:bool)->Result<()> {
    for (p,values) in [(&c.input,&f.input),(&c.previous,&f.previous),(&c.previous_coefficients,&f.previous_coefficients),(&c.weights,&f.weights),(&c.base,&f.base),(&c.scale,&f.scale)] {g.upload_at(p.address,&le(values))?;}
    // Signed, varied weights, including values that do not round to themselves in BF16.
    let norm=(0..f.shape.hidden).map(|i| {let v=(i%29) as f32*0.071-0.83;if norm_bf16 {f32::from_bits((v.to_bits().wrapping_add(0x7fff+(v.to_bits()>>16&1)))&0xffff_0000)} else {v}}).collect::<Vec<_>>();
    g.upload_at(c.norm.address,&le(&norm))
}
fn strips(f:&Fixture,bf16:bool)->Vec<f32> {
    let (s,count)=(f.shape,f.shape.coefficients());let periods=s.wide().div_ceil(256);
    let mut words=vec![0.0;FUSED_BLOCKS*periods*count*8];
    for k in 0..FUSED_BLOCKS {for p in 0..periods {for r in 0..count {for e in 0..8 {
        let j=p*256+k*8+e;if j<s.wide() {words[fused_strip_index(periods,count,k,p,r,e)]=f.weights[r*s.wide()+j];}
    }}}}
    if bf16 {words.chunks_exact(2).map(|p|f32::from_bits((p[0].to_bits()>>16)|(p[1].to_bits()&0xffff_0000))).collect()} else {words}
}
struct Case<'a> {cells:&'a [Cell;4],scratch:&'a [Buffer;4],weights:&'a Buffer,table:&'a Buffer,f:&'a Fixture,strip_bf16:bool,norm_bf16:bool,dispatch:bool}
fn check<'g>(g:&'g Gpu,e:&Entries,c:Case<'_>,graphs:&mut BTreeMap<(u32,bool),Graph<'g>>,broken:bool)->Result<Vec<Vec<u8>>> {
    let rows=c.f.shape.rows;let phased=!c.dispatch || mhc_params::boundary_launches(c.f.shape,rows as usize)<rows as usize;
    let args=|arm:usize| {let mut boundary=c.cells[arm].args(c.f,c.cells[arm].weights.address);boundary.norm_bf16=u32::from(c.norm_bf16);
        FusedBoundary {boundary,strips:c.weights.address,scratch:c.scratch[arm].address,bf16:u32::from(c.strip_bf16),reserved:0}};
    let nodes=plan(args(3),phased);let table=nodes.iter().flat_map(|&(kind,_,p)|command(KINDS[kind],p)).collect::<Vec<_>>();
    g.upload_at(c.table.address,&table)?;
    if let std::collections::btree_map::Entry::Vacant(slot)=graphs.entry((rows,phased)) {
        slot.insert(capture_graph(g,|| {for (i,&(kind,blocks,_)) in nodes.iter().enumerate() {
            let mut address=c.table.address;let mut index=i as u64;
            g.launch(e.recorded[kind],[blocks,1,1],FUSED_THREADS as u32,0,&mut [ptr(&mut address),ptr(&mut index)])?;
        }Ok(())})?);
    }
    let readonly=c.cells.iter().flat_map(inputs).chain([c.weights,c.table]).collect::<Vec<_>>();
    let before=readonly.iter().map(|p|p.read(g)).collect::<Result<Vec<_>>>()?;
    let input_spans=readonly.iter().map(|p|span(p)).chain(c.scratch.iter().map(span)).collect::<Vec<_>>();
    let label=format!("BOUNDARY rows={rows} streams={} hidden={} after={} strips_bf16={} norm_bf16={} dispatch={}",c.f.shape.streams,c.f.shape.hidden,c.f.after,c.strip_bf16,c.norm_bf16,c.dispatch);
    let result=byte_check::arms::<4>(&label,std::array::from_fn(|i|outputs(&c.cells[i]).map(span).to_vec()),&input_spans,|arm,poison| {
        for p in outputs(&c.cells[arm]) {g.upload_at(p.address,&vec![poison;p.bytes])?;}
        // Never reset scratch here. Both poison rounds and every next input/cohort reuse its dirty words.
        let scratch_before=c.scratch[arm].read(g)?;
        if arm==3 {sample(&graphs[&(rows,phased)],0)?;} else {direct(g,e,args(arm),arm,arm==2 && phased,broken && arm==2)?;}
        g.sync()?;let scratch_after=c.scratch[arm].read(g)?;
        let inactive=16+rows as usize*FUSED_SCRATCH_WORDS*4;
        if scratch_after[..16]!=GUARD || scratch_after[inactive..]!=scratch_before[inactive..] {return Err(format!("{label}: scratch guard or inactive row changed arm={arm}"));}
        // Unused coefficient slots are canaries between each row's arithmetic scratch and stamps.
        let unused=mhc_params::FUSED_SUMS+FUSED_BLOCKS*(1+c.f.shape.coefficients());
        for row in 0..rows as usize {let at=16+4*(row*FUSED_SCRATCH_WORDS+unused);let end=16+4*(row*FUSED_SCRATCH_WORDS+mhc_params::FUSED_STAMPS_AT);
            if scratch_before[at..end]!=scratch_after[at..end] {return Err(format!("{label}: row scratch canary changed arm={arm} row={row}"));}}
        outputs(&c.cells[arm]).iter().map(|p|p.read(g)).collect()
    });
    if before!=readonly.iter().map(|p|p.read(g)).collect::<Result<Vec<_>>>()? {return Err(format!("{label}: input changed"));}
    result
}
pub(crate) fn run(base_path:&str,path:&str)->Result<()> {
    byte_check::self_test()?;
    let base=std::fs::read(base_path).map_err(|e|e.to_string())?;let image=std::fs::read(path).map_err(|e|e.to_string())?;
    println!("BOUNDARY IMAGES base_sha256={} candidate_sha256={}",sha256::digest(&base),sha256::digest(&image));
    let mut g=Gpu::new()?;let old=g.module(&base)?;let module=g.module(&image)?;
    let e=Entries {fused:[g.function(old,"mhc_boundary_fused")?,g.function(module,"mhc_boundary_fused")?],
        phases:[g.function(module,"mhc_boundary_phase_rms")?,g.function(module,"mhc_boundary_phase_dots")?,g.function(module,"mhc_boundary_phase_finish")?],
        recorded:[g.function(module,"resident_launch_mhc_boundary_fused")?,g.function(module,"resident_launch_mhc_phase_rms")?,g.function(module,"resident_launch_mhc_phase_dots")?,g.function(module,"resident_launch_mhc_phase_finish")?]};
    let (mut cases,mut sequences,mut rejected)=(0,0,false);
    for &(streams,hidden) in mhc_params::PHASE_SHAPES {for after in [false,true] {for strip_bf16 in [false,true] {for norm_bf16 in [false,true] {
        if strip_bf16 && !mhc_params::fused_stage_fits(shape(1,streams,hidden)) {continue;}
        let mut cells=Vec::new();
        for rows in 1..=8 {let f=Fixture::new(shape(rows,streams,hidden),after,0,0,171);cells.push([Cell::new(&mut g,&f)?,Cell::new(&mut g,&f)?,Cell::new(&mut g,&f)?,Cell::new(&mut g,&f)?]);}
        let weights=Buffer::new(&mut g,&strips(&Fixture::new(shape(1,streams,hidden),after,0,0,171),strip_bf16))?;
        let dirty=vec![f32::from_bits(0x85858585);FUSED_SCRATCH_WORDS*FUSED_ROWS_MAX];
        let scratch=[Buffer::new(&mut g,&dirty)?,Buffer::new(&mut g,&dirty)?,Buffer::new(&mut g,&dirty)?,Buffer::new(&mut g,&dirty)?];
        for p in &scratch {for row in 0..FUSED_ROWS_MAX {g.upload_at(p.address+(row*FUSED_SCRATCH_WORDS*4) as u64,&[0;8])?;}}
        let table=Buffer::new(&mut g,&vec![0.0;COMMAND_BYTES*8/4])?;let mut graphs=BTreeMap::new();let mut last:Option<Vec<u8>>=None;
        let sweep=(0..6).flat_map(|pattern|(1..=8).map(move |rows|(rows,pattern,false)));
        let sequence=SEQUENCE.into_iter().map(|rows|(rows,0,true));
        for (step,(rows,kind,dispatch)) in sweep.chain(sequence).enumerate() {
            let mut f=Fixture::new(shape(rows,streams,hidden),after,kind,0,171+step as u64*19);
            if strip_bf16 {for w in &mut f.weights {*w=f32::from_bits((w.to_bits().wrapping_add(0x7fff+(w.to_bits()>>16&1)))&0xffff_0000);}}
            g.upload_at(weights.address,&le(&strips(&f,strip_bf16)))?;
            let current=&cells[rows as usize-1];for cell in current {refresh(&g,cell,&f,norm_bf16)?;}
            let case=||Case {cells:current,scratch:&scratch,weights:&weights,table:&table,f:&f,strip_bf16,norm_bf16,dispatch};
            if !rejected {
                let error=check(&g,&e,case(),&mut graphs,true).err().ok_or("broken phase passed the byte check")?;
                if !error.contains("first_word=") {return Err(format!("broken phase failed outside byte verdict: {error}"));}
                println!("BOUNDARY BROKEN PHASE REJECTED missing=finish");rejected=true;
            }
            let data=check(&g,&e,case(),&mut graphs,false)?;
            if dispatch {let residual=&data[0][16..data[0].len()-16];if let Some(old)=&last {let n=old.len().min(residual.len());if old[..n]==residual[..n] {return Err("changed input did not change a residual word".into());}}
                last=Some(residual.to_vec());}
            cases+=1;
        }
        sequences+=1;drop(graphs);g.clear()?;
    }}}}
    if !rejected {return Err("missing broken phase control".into());}
    println!("BOUNDARY BYTES PASS cases={cases} arms=base-fused,candidate-fused,direct-phases,recorded-phases dirty-scratch=reused guards=pass");
    println!("BOUNDARY GRAPH SEQUENCE PASS sequences={sequences} rows=8,5,4,3,4,5,8 eager=recorded changed-words=pass");Ok(())
}
