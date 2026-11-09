//! Complete norm-fed projection rows with the dynamic MXFP8 activation recipe.
//! This probe stops before the first attention state producer.
use super::{checkpoint,child,equal,floats,prompt,read,sha256,topology,write,Prefix};
#[cfg(target_os="linux")]
use super::oracle;
use engine_model::plan::{admission::CheckpointAdmission,layer::{Kind,Topology},state::StatePlan,storage::WeightStorage};
use spark_tp::{Result,layer_driver::LayerBoundary,mxfp8_projection::Mxfp8Projection};
use spark_gpu_probe::dynamic_mxfp8::Activation;
use std::{fs,path::{Path,PathBuf}};
#[cfg(test)]
#[allow(dead_code)]
#[path="../../../tests/support/attention_checkpoint.rs"] mod checkpoint_fixture;
const META:[&str;7]=["COMMON","IDENTITY","SCOPE","INPUT_IDS","GEOMETRY","PLANE_WORDS","NEXT_CELL"];
const PREFIX:[&str;5]=super::PLANES;
const ACT:[&str;3]=["activation.f8","activation.scale","activation.f32"];
const COMPLETE:&[u8]=b"DYNAMIC MXFP8 PREFIX PROJECTIONS v1 PASS\n";
const SUITE:&[u8]=b"DYNAMIC MXFP8 PREFIX PROJECTIONS SUITE v1 PASS\n";
const SCOPE:&[u8]=b"activation=dynamic_mxfp8_block32_bf16_source\nscale=ceil_log2_amax_over_448_clamp_e8m0\nvalues=e4m3fn_rne\naccumulator=fp32_fixed_tree\nepilogue=rn_even_bf16\nactivation_staging=host\nfull_layer=false\ntransport=false\ngather_completed=false\noutput_token_ids=false\nruntime_load_policy=stored_identity_candidate\nbackend_parity=unproved\n";
struct Projections<'a> {prefix:Prefix<'a>,rows:Vec<Mxfp8Projection<'a>>,next:String}
impl<'a> Projections<'a> {
    fn new(c:&'a CheckpointAdmission,t:Topology)->Result<Self> {
        let prefix=Prefix::new(c,t)?;let state=StatePlan::new(c.description(),t,1,isize::MAX as usize).map_err(|e|e.to_string())?;
        let b=LayerBoundary::new(c.description(),c.headers(),&state,0,1,t,usize::MAX)?;
        let norm=&b.program().steps[2].outputs[0].name;let mut rows=Vec::new();
        // Stop at the first consumer after the independent norm-fed projections.
        // Later projections need real attention state and cannot use this prefix.
        let mut next=String::new();
        for (i,op) in b.program().steps.iter().enumerate().skip(3) {
            if op.kind!=Kind::Project || op.inputs!=[norm.clone()] || op.weights.len()!=1 || op.weights[0].storage!=WeightStorage::Mxfp8 {
                next=format!("{} {:?} {:?}\n",op.id,op.kind,op.inputs);break;
            }
            rows.push(b.prepare_mxfp8_projection_checkpoint(i,c)?);
        }
        if rows.is_empty() || next.is_empty() {return Err("dynamic prefix needs complete MXFP8 projections and a following boundary".into());}
        Ok(Self {prefix,rows,next})
    }
    fn setup(&self,root:&Path,out:&Path,ids:&[u32])->Result<()> {
        let t=self.rows[0].topology();
        // The common operand record uses the unsplit reference identity.
        // GEOMETRY records the actual topology for each output row interval.
        write(&out.join("COMMON"),&self.prefix.identity(root,ids,1)?)?;
        write(&out.join("SCOPE"),SCOPE)?;write(&out.join("NEXT_CELL"),self.next.as_bytes())?;
        write(&out.join("INPUT_IDS"),ids.iter().map(|id|format!("{id}\n")).collect::<String>().as_bytes())?;
        let mut identity=String::new();let mut geometry=format!("{} {}\n",t.rank,t.parts);
        for p in &self.rows {
            let (w,s)=p.read()?;
            identity.push_str(&format!("input={} source_dtype={:?} output={} components={:?} offsets={:?} weight={} scale={} gather={}\n",
                p.input_name(),p.shape().input(),p.output_name(),p.component_names(),p.component_file_offsets(),sha256::digest(&w),sha256::digest(&s),p.requires_gather()));
            geometry.push_str(&format!("{} {} {} {}\n",p.row_origin(),p.shape().rows(),p.shape().cols(),p.total_rows()));
        }
        write(&out.join("IDENTITY"),identity.as_bytes())?;write(&out.join("GEOMETRY"),geometry.as_bytes())?;
        let s=self.prefix.block.mhc().shape();write(&out.join("PLANE_WORDS"),format!("{} {} {} {} {}\n",s.wide(),s.wide(),s.coefficients(),self.prefix.width,self.prefix.width).as_bytes())
    }
}
fn bounded(input:&Path)->Result<Vec<u32>> {let ids=prompt(input)?;if ids.len()>64 {return Err("dynamic prefix permits at most 64 input IDs".into());}Ok(ids)}
fn integers(path:&Path)->Result<Vec<usize>> {
    let bytes=engine_model::safetensors::read_bounded(path,65536).map_err(|e|e.to_string())?;
    std::str::from_utf8(&bytes).map_err(|e|e.to_string())?.split_whitespace().map(|v|v.parse().map_err(|_|"invalid dynamic projection geometry".into())).collect()
}
fn names(ids:&[u32],count:usize)->Vec<String> {
    let mut names=META.map(str::to_owned).to_vec();
    for i in 0..ids.len() {
        names.extend(PREFIX.into_iter().chain(ACT).map(|n|format!("{i}-{n}")));
        for j in 0..count {for n in ["accumulator.f32","output.f32"] {names.push(format!("{i}-p{j}-{n}"));}}
    }names
}
fn manifest(out:&Path,ids:&[u32],count:usize)->Result<Vec<u8>> {
    let mut text=String::new();for n in names(ids,count) {text.push_str(&format!("{}  {n}\n",sha256::digest(&read(&out.join(&n))?)));}Ok(text.into_bytes())
}
fn seal(out:&Path,ids:&[u32],count:usize)->Result<()> {write(&out.join("OUTPUT.sha256"),&manifest(out,ids,count)?)?;write(&out.join("COMPLETE"),COMPLETE)}
fn values(out:&Path,name:&str,count:usize,bf16:bool)->Result<Vec<f32>> {
    let bytes=read(&out.join(name))?;
    if count.checked_mul(4)!=Some(bytes.len()) {return Err(format!("{name}: dynamic projection plane extent differs"));}
    let v=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    if v.iter().any(|&v|!v.is_finite() || bf16 && spark_gpu_probe::bf16::rounded(v).to_bits()!=v.to_bits()) {
        return Err(format!("{name}: invalid dynamic projection value"));
    }Ok(v)
}
fn complete(out:&Path)->Result<(Vec<u32>,Vec<usize>)> {
    equal("dynamic completion",&read(&out.join("COMPLETE"))?,COMPLETE)?;equal("dynamic scope",&read(&out.join("SCOPE"))?,SCOPE)?;
    let ids=bounded(&out.join("INPUT_IDS"))?;let g=integers(&out.join("GEOMETRY"))?;
    if g.len()<6 || (g.len()-2)%4!=0 {return Err("dynamic projection geometry is incomplete".into());}
    topology(&g[0].to_string(),&g[1].to_string())?;
    let words=integers(&out.join("PLANE_WORDS"))?;
    if words.len()!=5 || words.iter().any(|&n|n==0 || n>isize::MAX as usize/4) {return Err("invalid dynamic prefix extents".into());}
    for row in g[2..].chunks_exact(4) {
        if row[1]==0 || row[2]==0 || row[2]%32!=0 || row[2]!=words[4] || row[0].checked_add(row[1]).is_none_or(|n|n>row[3]) {
            return Err("dynamic projection row interval differs".into());
        }
    }
    for i in 0..ids.len() {
        let mut prefix=Vec::new();for (n,&count) in PREFIX.into_iter().zip(&words) {prefix.push(values(out,&format!("{i}-{n}"),count,false)?);}
        let activation=Activation::from_f32(&prefix[4])?;
        for (name,data) in [(ACT[0],activation.values().to_vec()),(ACT[1],activation.scales().to_vec()),(ACT[2],floats(activation.decoded())?)] {
            equal(name,&read(&out.join(format!("{i}-{name}")))?,&data)?;
        }
        for (j,row) in g[2..].chunks_exact(4).enumerate() {
            values(out,&format!("{i}-p{j}-accumulator.f32"),row[1],false)?;
            values(out,&format!("{i}-p{j}-output.f32"),row[1],true)?;
        }
    }
    equal("dynamic manifest",&read(&out.join("OUTPUT.sha256"))?,&manifest(out,&ids,(g.len()-2)/4)?)?;Ok((ids,g))
}
fn compare(a:&Path,b:&Path)->Result<()> {
    let (ids,g)=complete(a)?;let (other,h)=complete(b)?;if ids!=other || g!=h {return Err("dynamic input IDs or geometry differ".into());}
    for name in names(&ids,(g.len()-2)/4) {equal(&name,&read(&a.join(&name))?,&read(&b.join(&name))?)?;}
    println!("DYNAMIC MXFP8 COMPARE PASS input_tokens={} projections={} output_token_ids=false",ids.len(),(g.len()-2)/4);Ok(())
}
fn save_activation(out:&Path,i:usize,a:&Activation)->Result<()> {
    write(&out.join(format!("{i}-{}",ACT[0])),a.values())?;write(&out.join(format!("{i}-{}",ACT[1])),a.scales())?;
    write(&out.join(format!("{i}-{}",ACT[2])),&floats(a.decoded())?)
}
fn host(root:&Path,input:&Path,out:&Path,t:Topology)->Result<()> {
    let ids=bounded(input)?;let c=checkpoint(root)?;let p=Projections::new(&c,t)?;p.setup(root,out,&ids)?;
    for (i,&id) in ids.iter().enumerate() {
        let prefix=p.prefix.expected(id)?;
        for (name,v) in PREFIX.into_iter().zip(&prefix) {write(&out.join(format!("{i}-{name}")),&floats(v)?)?;}
        for (j,row) in p.rows.iter().enumerate() {
            let (a,acc,output)=row.dynamic_reference(&prefix[4])?;if j==0 {save_activation(out,i,&a)?;}
            for (name,v) in [("accumulator.f32",acc),("output.f32",output)] {write(&out.join(format!("{i}-p{j}-{name}")),&floats(&v)?)?;}
        }
    }
    seal(out,&ids,p.rows.len())?;println!("DYNAMIC MXFP8 HOST PASS projections={} next={}",p.rows.len(),p.next.trim());Ok(())
}
#[cfg(target_os="linux")]
fn device(root:&Path,input:&Path,expected:&Path,out:&Path,t:Topology)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx};
    let ids=bounded(input)?;if complete(expected)?.0!=ids {return Err("dynamic expected input IDs differ".into());}
    let c=checkpoint(root)?;let p=Projections::new(&c,t)?;p.setup(root,out,&ids)?;
    for name in META {equal(name,&read(&out.join(name))?,&read(&expected.join(name))?)?;}
    let budget=oracle::budget(&p.prefix.block)?;let mut extra=0usize;
    for row in &p.rows {
        let s=row.shape();extra=extra.checked_add(s.arena_bytes()?).and_then(|n|n.checked_add(s.rows().checked_mul(8)?))
            .and_then(|n|n.checked_add(s.cols().checked_mul(8)?)).ok_or("dynamic projection budget overflow")?;
    }
    let device=budget.0.checked_add(extra).ok_or("dynamic device budget overflow")?;
    let required=budget.1.checked_add(extra).ok_or("dynamic required budget overflow")?;
    with_backend(0,layer_ptx()?,device,|b| {
        let available=b.available_memory_bytes()?;if required>available {return Err(format!("dynamic prefix needs {required} bytes; available {available}"));}
        let block=p.prefix.block.upload(b)?;let rows=p.rows.iter().map(|r|r.load_dynamic(b)).collect::<Result<Vec<_>>>()?;
        for (i,&id) in ids.iter().enumerate() {
            let embedded=oracle::device_embedding(b,&p.prefix,id)?;let embedding=b.read_vector_values(&embedded)?;
            let pass=block.begin(b.round_bf16_lanes(&embedded)?)?;
            let prefix=[embedding,b.read_vector_values(pass.mhc().normalized())?,b.read_vector_values(pass.mhc().coefficients())?,
                b.read_vector_values(pass.mhc().collapsed().vector())?,b.read_vector_values(pass.normalized().vector())?];
            for (name,v) in PREFIX.into_iter().zip(prefix) {let name=format!("{i}-{name}");let bytes=floats(&v)?;
                write(&out.join(&name),&bytes)?;equal(&name,&bytes,&read(&expected.join(&name))?)?;}
            for (j,row) in rows.iter().enumerate() {
                let projected=row.apply(pass.normalized())?;if j==0 {save_activation(out,i,projected.activation())?;}
                for (name,v) in [("accumulator.f32",projected.projection().accumulator()),("output.f32",projected.projection().output().vector())] {
                    let name=format!("{i}-p{j}-{name}");let bytes=floats(&b.read_vector_values(v)?)?;write(&out.join(&name),&bytes)?;
                    equal(&name,&bytes,&read(&expected.join(&name))?)?;
                }
            }
        }
        if let Some(e)=b.fatal_error() {return Err(e);}Ok(())
    })?;
    seal(out,&ids,p.rows.len())?;compare(expected,out)?;println!("DYNAMIC MXFP8 DEVICE PASS rank={} full_layer=false transport=false",t.rank);Ok(())
}
#[cfg(not(target_os="linux"))]
fn device(_:&Path,_:&Path,_:&Path,_:&Path,_:Topology)->Result<()> {Err("dynamic device cell requires Linux and CUDA".into())}
fn joined(full:&Path,shards:&[PathBuf])->Result<()> {
    let (ids,f)=complete(full)?;let mut joined=vec![vec![[Vec::new(),Vec::new()];(f.len()-2)/4];ids.len()];let mut origins=vec![0;(f.len()-2)/4];
    for (rank,path) in shards.iter().enumerate() {
        let (other,g)=complete(path)?;if other!=ids || g.len()!=f.len() {return Err("dynamic shard set differs".into());}
        let mut expected=vec![rank,shards.len()];
        for (j,(r,all)) in g[2..].chunks_exact(4).zip(f[2..].chunks_exact(4)).enumerate() {
            expected.extend([origins[j],r[1],all[2],all[3]]);
        }
        if g!=expected {return Err("dynamic shards omit or overlap complete output rows".into());}
        for name in ["COMMON","SCOPE","INPUT_IDS","PLANE_WORDS","NEXT_CELL"] {equal(name,&read(&full.join(name))?,&read(&path.join(name))?)?;}
        for (j,r) in g[2..].chunks_exact(4).enumerate() {origins[j]+=r[1];}
        for (i,outputs) in joined.iter_mut().enumerate() {
            for name in PREFIX.into_iter().chain(ACT) {equal(name,&read(&full.join(format!("{i}-{name}")))?,&read(&path.join(format!("{i}-{name}")))?)?;}
            for (j,output) in outputs.iter_mut().enumerate() {for (k,name) in ["accumulator.f32","output.f32"].into_iter().enumerate() {
                output[k].extend(read(&path.join(format!("{i}-p{j}-{name}")))?);
            }}
        }
    }
    for (origin,all) in origins.iter().zip(f[2..].chunks_exact(4)) {if *origin!=all[3] {return Err("dynamic shards do not cover full rows".into());}}
    for (i,outputs) in joined.iter().enumerate() {for (j,output) in outputs.iter().enumerate() {
        for (k,name) in ["accumulator.f32","output.f32"].into_iter().enumerate() {equal(name,&output[k],&read(&full.join(format!("{i}-p{j}-{name}")))?)?;}
    }}
    println!("DYNAMIC MXFP8 JOIN PASS projections={} gather_completed=false",origins.len());Ok(())
}
fn suite(root:&Path,input:&Path,out:&Path,parts:u32)->Result<()> {
    bounded(input)?;let full=out.join("host-full");
    child(&[Path::new("attention-projections"),Path::new("host"),root,input,&full,Path::new("0"),Path::new("1")],&out.join("host-full.log"))?;
    write(&out.join("PARTS"),format!("{parts}\n").as_bytes())?;
    for rank in 0..parts {
        let rank_text=rank.to_string();let parts_text=parts.to_string();let host=out.join(format!("rank{rank}-host"));
        child(&[Path::new("attention-projections"),Path::new("host"),root,input,&host,Path::new(&rank_text),Path::new(&parts_text)],&out.join(format!("rank{rank}-host.log")))?;
        for run in 1..=3 {let result=out.join(format!("rank{rank}-run{run}"));
            child(&[Path::new("attention-projections"),Path::new("device"),root,input,&host,&result,Path::new(&rank_text),Path::new(&parts_text)],&out.join(format!("rank{rank}-run{run}.log")))?;
            compare(&host,&result)?;compare(&out.join(format!("rank{rank}-run1")),&result)?;
        }
    }
    for run in 1..=3 {joined(&full,&(0..parts).map(|r|out.join(format!("rank{r}-run{run}"))).collect::<Vec<_>>())?;}
    write(&out.join("COMPLETE"),SUITE)?;println!("DYNAMIC MXFP8 SUITE PASS parts={parts} repeats=3 output_token_ids=false");Ok(())
}
fn compare_suite(a:&Path,b:&Path)->Result<()> {
    for p in [a,b] {equal("dynamic suite completion",&read(&p.join("COMPLETE"))?,SUITE)?;}
    let parts=integers(&a.join("PARTS"))?;
    if parts.len()!=1 || parts!=integers(&b.join("PARTS"))? || parts[0]==0 {return Err("dynamic suite parts differ".into());}
    compare(&a.join("host-full"),&b.join("host-full"))?;
    for run in 1..=3 {
        for rank in 0..parts[0] {let name=format!("rank{rank}-run{run}");compare(&a.join(&name),&b.join(&name))?;
            for p in [a,b] {compare(&p.join(format!("rank{rank}-host")),&p.join(&name))?;compare(&p.join(format!("rank{rank}-run1")),&p.join(&name))?;}
        }
        for p in [a,b] {joined(&p.join("host-full"),&(0..parts[0]).map(|r|p.join(format!("rank{r}-run{run}"))).collect::<Vec<_>>())?;}
    }
    println!("DYNAMIC MXFP8 CROSS SPARK PASS parts={} repeats=3 output_token_ids=false",parts[0]);Ok(())
}
pub fn run(args:&[&str])->Result<()> {
    match args {
        ["host",root,input,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;host(Path::new(root),Path::new(input),Path::new(out),t)},
        ["device",root,input,expected,out,rank,parts]=>{let t=topology(rank,parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;device(Path::new(root),Path::new(input),Path::new(expected),Path::new(out),t)},
        ["suite",root,input,out,parts]=>{let t=topology("0",parts)?;fs::create_dir(out).map_err(|e|e.to_string())?;suite(Path::new(root),Path::new(input),Path::new(out),t.parts)},
        ["compare",a,b]=>compare(Path::new(a),Path::new(b)),["compare-suite",a,b]=>compare_suite(Path::new(a),Path::new(b)),
        _=>Err("usage: checkpoint-prefix-check attention-projections host MODEL INPUT_IDS NEW_OUT RANK PARTS | device MODEL INPUT_IDS EXPECTED NEW_OUT RANK PARTS | suite MODEL INPUT_IDS NEW_OUT PARTS | compare LEFT RIGHT | compare-suite LEFT RIGHT".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64,Ordering};
    struct Temp(PathBuf);
    impl Temp {fn new()->Self {
        static NEXT:AtomicU64=AtomicU64::new(0);
        let p=std::env::temp_dir().join(format!("vade-dynamic-receipt-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&p).unwrap();Self(p)
    }}
    impl Drop for Temp {fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();}}
    fn receipt(root:&Path,rank:usize,parts:usize,origin:usize,rows:usize)->PathBuf {
        let out=root.join(format!("p{parts}-r{rank}"));fs::create_dir(&out).unwrap();
        for (name,data) in [("COMMON",b"synthetic\n".as_slice()),("IDENTITY",b"synthetic rows\n"),("SCOPE",SCOPE),
            ("INPUT_IDS",b"1\n"),("NEXT_CELL",b"synthetic next cell\n"),("PLANE_WORDS",b"64 64 8 32 32\n")] {write(&out.join(name),data).unwrap();}
        write(&out.join("GEOMETRY"),format!("{rank} {parts}\n{origin} {rows} 32 4\n").as_bytes()).unwrap();
        for (n,count) in PREFIX.into_iter().zip([64,64,8,32,32]) {write(&out.join(format!("0-{n}")),&floats(&vec![1.0;count]).unwrap()).unwrap();}
        save_activation(&out,0,&Activation::from_f32(&[1.0;32]).unwrap()).unwrap();
        let output=(origin..origin+rows).map(|i|i as f32).collect::<Vec<_>>();
        for n in ["accumulator.f32","output.f32"] {write(&out.join(format!("0-p0-{n}")),&floats(&output).unwrap()).unwrap();}
        seal(&out,&[1],1).unwrap();out
    }
    #[test]
    fn dynamic_receipts_check_quantized_input_and_complete_joined_rows() {
        let temp=Temp::new();let full=receipt(&temp.0,0,1,0,4);let a=receipt(&temp.0,0,2,0,2);let b=receipt(&temp.0,1,2,2,2);
        compare(&a,&a).unwrap();joined(&full,&[a.clone(),b.clone()]).unwrap();
        assert!(joined(&full,&[b.clone(),a.clone()]).is_err());assert!(compare(&a,&b).is_err());
        fs::write(b.join("0-activation.f8"),[0;32]).unwrap();
        fs::write(b.join("OUTPUT.sha256"),manifest(&b,&[1],1).unwrap()).unwrap();
        assert!(complete(&b).is_err());
        fs::write(b.join("0-activation.f8"),Activation::from_f32(&[1.0;32]).unwrap().values()).unwrap();
        fs::write(b.join("GEOMETRY"),b"1 2\n1 2 32 4\n").unwrap();
        fs::write(b.join("OUTPUT.sha256"),manifest(&b,&[1],1).unwrap()).unwrap();
        assert!(joined(&full,&[a,b]).is_err());
    }
    #[test]
    fn incomplete_and_non_bf16_output_receipts_refuse() {
        let temp=Temp::new();let p=receipt(&temp.0,0,1,0,4);
        fs::write(p.join("0-p0-output.f32"),floats(&[1.0001;4]).unwrap()).unwrap();
        fs::write(p.join("OUTPUT.sha256"),manifest(&p,&[1],1).unwrap()).unwrap();assert!(complete(&p).is_err());
        fs::write(p.join("COMPLETE"),b"FAILED\n").unwrap();assert!(complete(&p).is_err());
    }
    #[test]
    fn real_operand_setup_keeps_common_identity_equal_across_row_splits() {
        let temp=Temp::new();let c=checkpoint_fixture::Checkpoint::new(engine_model::safetensors::Dtype::Bf16);let mut common=None;
        for parts in [1,2,4] {for rank in 0..parts {
            let p=Projections::new(&c.admission,Topology {rank,parts}).unwrap();assert_eq!(p.rows.len(),3);
            let out=temp.0.join(format!("setup-{parts}-{rank}"));fs::create_dir(&out).unwrap();
            p.setup(&c.root,&out,&[1,2]).unwrap();let record=read(&out.join("COMMON")).unwrap();
            if let Some(expected)=&common {assert_eq!(&record,expected);} else {common=Some(record);}
            assert!(p.next.contains("Conv"));
        }}
    }
}
