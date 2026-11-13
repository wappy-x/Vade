#![forbid(unsafe_code)]
use super::{Gpu,H,Result,ptr,sha256,capture_graph,sample};
use crate::{mhc_params::{Shape,Boundary,Prepare},mhc_reference::{Fixture,bytes},mhc_math};


const GUARD:[u8;16]=[0x5a;16];
struct Buffer {base:u64,address:u64,len:usize}
impl Buffer {
    fn new(gpu:&mut Gpu,data:&[u8])->Result<Self> {
        let base=gpu.upload(&[GUARD.as_slice(),data,GUARD.as_slice()].concat())?;
        Ok(Self {base,address:base+16,len:data.len()})
    }
    fn floats(gpu:&mut Gpu,data:&[f32])->Result<Self> {Self::new(gpu,&bytes(data))}
    fn blank(gpu:&mut Gpu,words:usize)->Result<Self> {Self::new(gpu,&vec![0xff;words*4])}
    fn poison(&self,gpu:&Gpu)->Result<()> {gpu.upload_at(self.address,&vec![0xff;self.len])}
    fn read(&self,gpu:&Gpu)->Result<Vec<u8>> {
        let data=gpu.read(self.base,self.len+32)?;
        if data[..16]!=GUARD || data[data.len()-16..]!=GUARD {return Err("gate buffer guard changed".into());}
        Ok(data[16..data.len()-16].to_vec())
    }
    fn check(&self,gpu:&Gpu,expected:&[f32],label:&str)->Result<String> {
        let got=self.read(gpu)?;let want=bytes(expected);
        if got!=want {
            let at=got.chunks_exact(4).zip(want.chunks_exact(4)).position(|(got,want)|got!=want);
            let bits=at.map(|at|(u32::from_le_bytes(got[at*4..at*4+4].try_into().unwrap()),expected[at].to_bits()));
            return Err(format!("HOST_ORACLE FAIL {label} first_word={at:?} bits={bits:x?} got_sha256={} oracle_sha256={}",sha256::digest(&got),sha256::digest(&want)));
        }
        Ok(sha256::digest(&got))
    }
}
pub(super) fn rounded(value:f32)->f32 {let bits=value.to_bits();f32::from_bits(if bits&0x7fffffff>0x7f800000 {0x7fff0000} else {bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000})}
fn canonical_nan(value:f32)->f32 {if value.is_nan() {f32::from_bits(0x7fffffff)} else {value}}
pub(super) fn boundary_oracle(gpu:&Gpu,args:&Boundary,finished:bool)->Result<Vec<Vec<f32>>> {
    let read=|address,words|->Result<Vec<f32>> {Ok(gpu.read(address,words*4)?.chunks_exact(4).map(|word|f32::from_le_bytes(word.try_into().unwrap())).collect())};
    let shape=args.shape;let (batch,streams,hidden)=(shape.rows as usize,shape.streams as usize,shape.hidden as usize);
    let (wide,count,chunks)=(shape.wide(),shape.coefficients(),hidden.div_ceil(128));
    let after=args.previous_input!=0;let input=read(args.input,batch*if after {hidden} else {wide})?;
    let previous=if after {read(args.previous_input,batch*wide)?} else {vec![]};
    let previous_coeff=if after {read(args.previous_coefficients,batch*count)?} else {vec![]};
    let weights=read(args.weights,count*wide)?;let base=read(args.base,count)?;let scale=read(args.scale,3)?;
    let norm=if finished {read(args.norm_weight,hidden)?} else {vec![]};
    let mut residual=vec![0.0;batch*wide];let mut normalized=vec![0.0;batch*wide];let mut coefficients=vec![0.0;batch*count];
    let mut collapsed=vec![0.0;batch*hidden];let mut partials=vec![0.0;batch*chunks];let mut output=vec![0.0;batch*hidden];
    let fold=|mut lanes:[f32;32]| {for distance in [16,8,4,2,1] {let before=lanes;for lane in 0..32 {lanes[lane]+=before[lane^distance];}}lanes};
    for token in 0..batch {
        for feature in 0..wide {
            residual[token*wide+feature]=if !after {rounded(input[token*wide+feature])} else {
                let destination=feature/hidden;let feature=feature%hidden;let coeff=&previous_coeff[token*count..][..count];
                let mut sum=0.0;for source in 0..streams {sum+=rounded(coeff[2*streams+source*streams+destination])*previous[token*wide+source*hidden+feature];}
                rounded(rounded(sum)+rounded(rounded(coeff[streams+destination])*rounded(input[token*hidden+feature])))
            };
        }
        mhc_math::normalize(shape,&residual[token*wide..][..wide],&mut normalized[token*wide..][..wide]);
        for coefficient in 0..count {coefficients[token*count+coefficient]=mhc_math::project(shape,coefficient,&weights,&base,&scale,&normalized[token*wide..][..wide]);}
        if !finished {continue;}
        mhc_math::activate(shape,&mut coefficients[token*count..][..count]);
        for feature in 0..hidden {let mut sum=0.0;for source in 0..streams {sum+=coefficients[token*count+source]*residual[token*wide+source*hidden+feature];}collapsed[token*hidden+feature]=rounded(sum);}
        for chunk in 0..chunks {
            let mut lanes=[0.0;32];for lane in 0..32 {for feature in (chunk*128+lane..((chunk+1)*128).min(hidden)).step_by(32) {let value=collapsed[token*hidden+feature];lanes[lane]+=value*value;}}
            partials[token*chunks+chunk]=fold(lanes)[0];
        }
        let mut lanes=[0.0;32];for lane in 0..32 {for chunk in (lane..chunks).step_by(32) {lanes[lane]+=partials[token*chunks+chunk];}}
        let lanes=fold(lanes);
        for feature in 0..hidden {let inverse=1.0/crate::qsa_common::root(lanes[feature%32]/hidden as f32+args.norm_epsilon.abs());
            let value=rounded(collapsed[token*hidden+feature]*inverse*1.0)*norm[feature];output[token*hidden+feature]=if args.norm_bf16==0 {value} else {rounded(value)};}
    }
    for value in normalized.iter_mut().chain(coefficients.iter_mut()).chain(partials.iter_mut()) {*value=canonical_nan(*value);}
    if args.norm_bf16==0 {for value in &mut output {*value=canonical_nan(*value);}}
    let mut planes=vec![residual,normalized,coefficients];if finished {planes.extend([collapsed,partials,output]);}Ok(planes)
}
pub(super) fn guarded_oracle(values:&[f32],length:usize)->Vec<u8> {
    let mut out=vec![0xff;length];out[..16].copy_from_slice(&GUARD);out[length-16..].copy_from_slice(&GUARD);
    out[16..16+values.len()*4].copy_from_slice(&bytes(values));out
}
fn value(index:usize)->f32 {((index*37+index/7*11)%97) as f32/31.0-1.0}
fn gather(input:&[f32],outer:usize,axis:usize,parts:usize)->Vec<f32> {
    let mut full=vec![f32::NAN;input.len()];
    for rank in (0..parts).rev() {for row in 0..outer {
        let first=row*axis+axis*rank/parts;let end=row*axis+axis*(rank+1)/parts;
        full[first..end].copy_from_slice(&input[first..end]);
    }}
    assert_eq!(bytes(input),bytes(&full));full
}
fn timing(gpu:&Gpu,label:&str,launches:usize,mut main:impl FnMut()->Result<()>,mut tree:impl FnMut()->Result<()>)->Result<()> {
    let samples=match std::env::var("ORDER_GATE_SAMPLES") {Ok(value)=>value.parse::<usize>().map_err(|_|"ORDER_GATE_SAMPLES must be an even integer from 2 to 1000")?,Err(_)=>20};
    if !(2..=1000).contains(&samples) || samples%2!=0 {return Err("ORDER_GATE_SAMPLES must be an even integer from 2 to 1000".into());}
    let graphs=[capture_graph(gpu,&mut main)?,capture_graph(gpu,&mut tree)?];
    for _ in 0..10 {for graph in &graphs {sample(graph,0)?;}}
    let mut times=[Vec::new(),Vec::new()];
    for repeat in 0..samples {for arm in if repeat%2==0 {[0,1]} else {[1,0]} {times[arm].push(sample(&graphs[arm],0)?.0);}}
    println!("EVENT_RAW {label} warmup=10 iterations={samples} main_us={:?} tree_us={:?}",times[0],times[1]);
    for arm in &mut times {arm.sort_by(f64::total_cmp);}
    let medians=times.map(|arm|(arm[samples/2-1]+arm[samples/2])/2.0/launches as f64);
    println!("EVENT_MEDIAN {label} launches_per_sample={launches} main_us={:.6} tree_us={:.6} tree_over_main={:.6}",medians[0],medians[1],medians[1]/medians[0]);Ok(())
}
fn mhc(gpu:&mut Gpu,modules:[H;2],batch:usize,streams:usize,hidden:usize,parts:usize,mode:&str)->Result<()> {
    let time=mode.starts_with("time");let after=mode.ends_with("-after");let guard_block=u32::from(!time);
    let shape=Shape {rows:batch as u32,streams:streams as u32,hidden:hidden as u32,iterations:4,rms_eps:1e-6,eps:1e-6};
    let mut fixture=Fixture::new(shape);let wide=shape.wide();let count=shape.coefficients();
    fixture.input=(0..batch*wide).map(|index|rounded(value(index))).collect();
    fixture.weights=(0..count*wide).map(|index|value(index+19)/29.0).collect();
    let previous=fixture.input.clone();let previous_coeff=fixture.reference().coefficients;
    let current=(0..batch*hidden).map(|index|rounded(value(index+71))).collect::<Vec<_>>();
    if after {
        for token in 0..batch {for destination in 0..streams {for feature in 0..hidden {
            let coeff=&previous_coeff[token*count..][..count];let mut sum=0.0;
            for source in 0..streams {sum+=rounded(coeff[2*streams+source*streams+destination])*previous[token*wide+source*hidden+feature];}
            fixture.input[token*wide+destination*hidden+feature]=rounded(rounded(sum)+rounded(rounded(coeff[streams+destination])*rounded(current[token*hidden+feature])));
        }}}
    }
    let expected=fixture.reference();

    let raw=(0..batch*count).map(|index|mhc_math::project(shape,index%count,&fixture.weights,&fixture.base,&fixture.scale,&expected.normalized[index/count*wide..][..wide])).collect::<Vec<_>>();
    let full=gather(&fixture.input,batch*streams,hidden,parts);
    let input=Buffer::floats(gpu,if after {&current} else {&full})?;let previous_gpu=Buffer::floats(gpu,&previous)?;let previous_coeff_gpu=Buffer::floats(gpu,&previous_coeff)?;let weights=Buffer::floats(gpu,&fixture.weights)?;
    let base=Buffer::floats(gpu,&fixture.base)?;let scale=Buffer::floats(gpu,&fixture.scale)?;
    let residual=Buffer::blank(gpu,batch*wide)?;let normalized=Buffer::blank(gpu,batch*wide)?;
    let coefficients=Buffer::blank(gpu,batch*count)?;let collapsed=Buffer::blank(gpu,batch*hidden)?;
    let partials=Buffer::blank(gpu,batch*hidden.div_ceil(128))?;
    let make=||Boundary {input:input.address,previous_input:if after {previous_gpu.address} else {0},previous_coefficients:if after {previous_coeff_gpu.address} else {0},residual:residual.address,
        weights:weights.address,base:base.address,scale:scale.address,mixing_normalized:normalized.address,
        coefficients:coefficients.address,collapsed:collapsed.address,norm_weight:0,norm_partials:partials.address,output:0,
        shape,norm_epsilon:1e-6,norm_bf16:0,prefetch:0,prefetch_bytes:0,prefetch_policy:0,prefetch_reserved:0};
    let mut entries=vec!["mhc_prepare","resident_launch_prepare","mhc_boundary_project","mhc_boundary_rms","mhc_boundary_dots"];
    if wide%4==0 && count<=24 {entries.push("mhc_project_tile");}
    if after {entries.retain(|name|!matches!(*name,"mhc_prepare"|"resident_launch_prepare"));}
    if mode=="mhc-tile" {entries.retain(|name|*name=="mhc_project_tile");}
    if mode=="mhc-project" {entries.retain(|name|*name=="mhc_boundary_project" || *name=="mhc_boundary_dots");}
    let mut prepare_bytes=Vec::new();
    u64s(&mut prepare_bytes,&[input.address,weights.address,base.address,scale.address,normalized.address,coefficients.address,collapsed.address]);
    u32s(&mut prepare_bytes,&[shape.rows,shape.streams,shape.hidden,shape.iterations]);prepare_bytes.extend(bytes(&[shape.rms_eps,shape.eps]));
    let prepare_command=command(gpu,9,&prepare_bytes)?;
    for name in entries {

        if !enabled(name) {continue;}
        let functions=[gpu.function(modules[0],name)?,gpu.function(modules[1],name)?];
        let rms=[gpu.function(modules[0],"mhc_boundary_rms")?,gpu.function(modules[1],"mhc_boundary_rms")?];
        let residual_fn=gpu.function(modules[1],"mhc_residual_rows")?;
        let launch=|arm:usize|->Result<()> {
            let mut args=make();
            match name {
                "resident_launch_prepare"=>{let (mut address,mut index)=(prepare_command.address,0u64);gpu.launch(functions[arm],[batch as u32+guard_block,1,1],shape.prepare_threads(),0,&mut [ptr(&mut address),ptr(&mut index)])},
                "mhc_prepare"=>{let mut args=Prepare {input:input.address,weights:weights.address,base:base.address,scale:scale.address,
                    normalized:normalized.address,coefficients:coefficients.address,collapsed:collapsed.address,shape};
                    gpu.launch(functions[arm],[batch as u32+guard_block,1,1],shape.prepare_threads(),0,&mut [ptr(&mut args)])},
                "mhc_project_tile"=>{let (blocks,threads)=if arm==1 {(batch.div_ceil(8) as u32+guard_block,256)} else {(batch.div_ceil(8) as u32+guard_block,256)};gpu.launch(functions[arm],[blocks,1,1],threads,0,&mut [ptr(&mut args)])},
                "mhc_boundary_dots"=>gpu.launch(functions[arm],[(batch*count) as u32+guard_block,1,1],256,0,&mut [ptr(&mut args)]),
                "mhc_boundary_rms"=>gpu.launch(functions[arm],[batch as u32+guard_block,1,1],256,0,&mut [ptr(&mut args)]),
                _=>gpu.launch(functions[arm],[batch as u32+guard_block,1,1],shape.prepare_threads(),0,&mut [ptr(&mut args)]),
            }
        };
        let setup=|arm:usize|->Result<()> {let mut args=make();
            if name=="mhc_boundary_dots" {gpu.launch(rms[arm],[batch as u32,1,1],256,0,&mut [ptr(&mut args)])?;}
            if name=="mhc_project_tile" {gpu.launch(residual_fn,[(batch*wide).div_ceil(128) as u32,1,1],128,0,&mut [ptr(&mut args)])?;}
            Ok(())};
        let label=format!("{name} batch={batch} hidden={hidden} columns={wide} ranks={parts}{}",if after {" after=true"} else {""});
        if time {setup(1)?;gpu.sync()?;timing(gpu,&label,1,||launch(0),||launch(1))?;continue;}
        setup(1)?;
        for repeat in 1..=3 {
            if name=="mhc_boundary_rms" {
                partials.poison(gpu)?;residual.poison(gpu)?;launch(1)?;gpu.sync()?;
                let mut inverse=vec![f32::from_bits(0xffffffff);batch*hidden.div_ceil(128)];
                for token in 0..batch {inverse[token*hidden.div_ceil(128)]=mhc_math::inverse(shape,&fixture.input[token*wide..][..wide]);}
                let hash=partials.check(gpu,&inverse,&label)?;residual.check(gpu,&fixture.input,&label)?;
                println!("HOST_ORACLE PASS {label} repeat={repeat} words={} inverse_sha256={hash}",batch*wide+inverse.len());continue;
            }
            normalized.poison(gpu)?;coefficients.poison(gpu)?;launch(1)?;gpu.sync()?;
            let norm_hash=normalized.check(gpu,&expected.normalized,&label)?;
            let hash=coefficients.check(gpu,if matches!(name,"mhc_prepare"|"resident_launch_prepare") {&expected.coefficients} else {&raw},&label)?;
            if !matches!(name,"mhc_prepare"|"resident_launch_prepare") {residual.check(gpu,&fixture.input,&label)?;}
            println!("HOST_ORACLE PASS {label} repeat={repeat} words={} coeff_sha256={hash} norm_sha256={norm_hash}",batch*(wide+count));
        }
    }
    Ok(())
}
fn command(gpu:&mut Gpu,kind:u32,payload:&[u8])->Result<Buffer> {
    if payload.len()>1024 {return Err("gate payload exceeds command ABI".into());}
    let mut data=vec![0;1040];data[..4].copy_from_slice(&kind.to_le_bytes());data[16..16+payload.len()].copy_from_slice(payload);Buffer::new(gpu,&data)
}
fn enabled(label:&str)->bool {std::env::var("ORDER_GATE_ONLY").map_or(true,|filter|filter.split(',').any(|part|label.contains(part)))}
fn u32s(out:&mut Vec<u8>,values:&[u32]) {for value in values {out.extend(value.to_le_bytes());}}
fn u64s(out:&mut Vec<u8>,values:&[u64]) {for value in values {out.extend(value.to_le_bytes());}}
pub(crate) fn run(tree_path:&str,main_path:&str,mode:&str)->Result<()> {
    let tree=std::fs::read(tree_path).map_err(|error|error.to_string())?;let main=std::fs::read(main_path).map_err(|error|error.to_string())?;
    println!("ORDER_GATE mode={mode} oracle=host_strip_fold tree_sha256={} main_sha256={} ranks=host_modeled_not_physical",sha256::digest(&tree),sha256::digest(&main));
    let mut gpu=Gpu::new()?;let modules=[gpu.module(&main)?,gpu.module(&tree)?];
    // mHC-only branch: the projection, served and stationary gate modes stay on order-tree-fast.
    if !["mhc-project","mhc-sweep","mhc-tile"].contains(&mode) {return Err("invalid gate mode (mHC-only branch: mhc-project, mhc-sweep, mhc-tile)".into());}
    for batch in [127,128,129] {for parts in [1,2,4,8] {
        for streams in [1,4] {mhc(&mut gpu,modules,batch,streams,257,parts,mode)?;gpu.clear()?;}
    }}
    println!("ORDER_GATE PASS mode={mode} scope=targeted_boundary_matrix_not_legacy_full_sweep");Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_nan_preserves_all_non_nan_bits() {
        for bits in [0,0x80000000,1,0x807fffff,0x3f800000,0x7f7fffff,0x7f800000,0xff800000] {
            assert_eq!(canonical_nan(f32::from_bits(bits)).to_bits(),bits);
        }
        for bits in [0x7f800001,0x7fff0000,0x7fffffff,0xffc00001] {
            assert_eq!(canonical_nan(f32::from_bits(bits)).to_bits(),0x7fffffff);
        }
    }
    #[test]
    fn gathers_preserve_partial_rank_intervals_and_signed_zero() {
        for width in [1,16,257,272] {for parts in [1,2,4,8] {
            let input=(0..3*width).map(|index|if index%7==0 {-0.0} else {value(index)}).collect::<Vec<_>>();
            assert_eq!(bytes(&input),bytes(&gather(&input,3,width,parts)));
        }}
    }
    #[test]
    fn guarded_reference_preserves_unwritten_tail() {
        let got=guarded_oracle(&[1.0,-0.0],48);
        assert_eq!(&got[..16],&GUARD);assert_eq!(&got[16..24],&bytes(&[1.0,-0.0]));
        assert_eq!(&got[24..32],&[0xff;8]);assert_eq!(&got[32..],&GUARD);
    }
}
