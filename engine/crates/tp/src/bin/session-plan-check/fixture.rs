//! Conditional physical storage fixture. It does not qualify cache producer types.
use engine_model::plan::{ConfigPlan, layer::{Topology,cast::Dtype}, state::{StatePlan,PlaneKind}};
use spark_tp::session_state::*;
use std::{collections::BTreeMap,fmt::Write};
#[allow(dead_code)]
#[path="../../../../../tools/export/src/sha256.rs"]
mod sha256;
pub type Result<T> = std::result::Result<T,String>;
pub fn need(ok: bool,why: &str) -> Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn kind(name: &str) -> Result<PlaneKind> {
    use PlaneKind::*;
    match name {"kda_state"=>Ok(KdaState),"kda_history"=>Ok(KdaHistory),"latent"=>Ok(Latent),
        "index_keys"=>Ok(IndexKeys),"index_gates"=>Ok(IndexGates),"pools"=>Ok(Pools),_=>Err("unknown cache kind".into())}
}
pub fn layout(config: &[u8],types: &str,capacity: u64,topology: Topology) -> Result<Layout> {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(config).map_err(|e|e.to_string())? else {return Err("state probe requires a deferred description".into());};
    let mut storage=BTreeMap::new(); let mut scratch=None;
    for line in types.lines().filter(|s| !s.trim().is_empty()) {
        let fields=line.split_whitespace().collect::<Vec<_>>();
        let [name,value]=fields.as_slice() else {return Err("cache type row needs a name and value".into());};
        if *name=="abi_scratch_bytes" {
            need(scratch.is_none(),"duplicate ABI scratch extent")?;
            scratch=Some(value.parse::<usize>().map_err(|_|"invalid ABI scratch extent")?); continue;
        }
        let dtype=match *value {"bf16"=>Dtype::Bf16,"f32"=>Dtype::F32,_=>return Err("cache dtype must be explicit bf16 or f32".into())};
        need(storage.insert(kind(name)?,dtype).is_none(),"duplicate cache kind")?;
    }
    let plan=StatePlan::new(&d,topology,capacity,isize::MAX as usize).map_err(|e|e.to_string())?;
    let types=plan.planes().iter().map(|p| Ok(CacheType {layer:p.layer,kind:p.kind,
        dtype:*storage.get(&p.kind).ok_or("cache kind has no explicit dtype")?})).collect::<Result<Vec<_>>>()?;
    Layout::physical_storage(&plan,&types,scratch.ok_or("ABI scratch extent is absent")?)
}
pub fn token(position: u64) -> u8 {(position as u8).wrapping_mul(29).wrapping_add(11)}
fn salt(p: &Plane) -> u8 {(p.layer as u8).wrapping_mul(13).wrapping_add((p.kind as u8).wrapping_mul(17))}
/// Independent closed-form reference. Fixed planes fold the causal byte
/// recurrence. Row planes use the token that completes each physical row.
pub fn expected(p: &Plane,context: u64) -> Result<Vec<u8>> {
    let mut sum=0u8; let mut bias=0u8;
    for position in 1..=context {sum=sum.wrapping_mul(3).wrapping_add(1);bias=bias.wrapping_mul(3).wrapping_add(token(position));}
    Ok((0..p.used_bytes(context)?).map(|j| match p.length {
        Length::Fixed(_)=>bias.wrapping_add(salt(p).wrapping_add(j as u8).wrapping_mul(sum)),
        Length::Rows {row_bytes,tokens_per_row}=>token((j/row_bytes+1) as u64*tokens_per_row).wrapping_add(salt(p)).wrapping_add(j as u8),
    }).collect())
}
fn row(out: &mut String,context: u64,i: usize,bytes: &[u8]) {
    writeln!(out,"context={context} plane={i} bytes={} sha256={}",bytes.len(),sha256::digest(bytes)).unwrap();
}
pub fn identity(config: &[u8],types: &str,l: &Layout,topology: Topology,boundary: u64) -> String {
    let mut out=format!("scope=conditional-physical-storage model_ids=false producer_types_qualified=false transport=false\nconfig_sha256={}\ntypes_sha256={}\nparts={} rank={} capacity={} boundary={} bank={} scratch={}\n",
        sha256::digest(config),sha256::digest(types.as_bytes()),topology.parts,topology.rank,l.capacity(),boundary,l.bank_bytes(),l.scratch_bytes());
    for (i,p) in l.planes().iter().enumerate() {writeln!(out,"plane={i} layer={} kind={:?} length={:?}",p.layer,p.kind,p.length).unwrap();} out
}
pub fn oracle(l: &Layout,boundary: u64) -> Result<String> {
    let mut out=String::new();
    for context in (0..=l.capacity()).chain([l.capacity(),boundary,l.capacity()]) {
        for (i,p) in l.planes().iter().enumerate() {row(&mut out,context,i,&expected(p,context)?);}
    } Ok(out)
}
fn check<S: Store>(s: &Session<S>,out: &mut String) -> Result<()> {
    for (i,p) in s.layout().planes().iter().enumerate() {
        let bytes=s.read(i)?; need(bytes==expected(p,s.context())?,"resident plane differs from independent causal reference")?;
        row(out,s.context(),i,&bytes);
    } Ok(())
}
fn advance<S: Store>(s: &mut Session<S>) -> Result<()> {
    let mut a=s.begin_append()?; let context=a.context(); let planes=a.layout().planes().to_vec();
    a.run(&mut ComputeOnly,|store,work| {
        need(store.read(work.scratch)?.iter().all(|&b|b==0),"token scratch was not cleared")?;
        store.write(work.scratch,&[0xa5])?; store.wait()?;
        for (i,p) in planes.iter().enumerate() {
            let old=store.read(&work.committed[i])?; let mut staged=store.read(&work.staged[i])?;
            need(old==staged,"append bank differs from committed bank")?;
            let used=p.used_bytes(context-1)?;
            need(old[..used]==expected(p,context-1)? && old[used..].iter().all(|&b|b==0),"committed bytes or capacity padding differ")?;
            let n=p.used_bytes(context)?;
            match p.length {
                Length::Fixed(_)=>for (j,b) in staged[..n].iter_mut().enumerate() {*b=b.wrapping_mul(3).wrapping_add(token(context)).wrapping_add(salt(p)).wrapping_add(j as u8);},
                Length::Rows {..}=>for (j,b) in staged[..n].iter_mut().enumerate().skip(used) {*b=token(context).wrapping_add(salt(p)).wrapping_add(j as u8);},
            }
            staged[n..].fill(0); store.write(&mut work.staged[i],&staged)?; store.wait()?;
            need(store.read(&work.staged[i])?==staged,"staged bytes or capacity padding differ")?;
        }
        Ok((0..planes.len()).collect())
    })?; a.commit()
}
fn rejected<S: Store>(s: &mut Session<S>) -> Result<()> {
    let context=s.context();
    // A complete staged bank followed by an invalid replacement must not commit.
    let mut a=s.begin_append()?;
    for (i,p) in a.layout().planes().to_vec().iter().enumerate() {a.write(i,&expected(p,context+1)?)?;}
    need(a.write(usize::MAX,&[]).is_err(),"invalid replacement was accepted")?;
    need(a.commit().is_err(),"invalid replacement published a staged bank")?;
    // An error after submission must drain. It must leave committed state intact.
    let mut a=s.begin_append()?;
    need(a.run(&mut ComputeOnly,|store,work| {store.write(&mut work.staged[0],&[0xff])?;Err("injected producer error".into())}).is_err(),"producer error was ignored")?;
    need(a.commit().is_err(),"failed producer committed")?;
    let mut a=s.begin_append()?; a.write(0,&expected(&a.layout().planes()[0],context+1)?)?; a.rollback();
    need(s.context()==context,"rejected append changed position")?;
    check(s,&mut String::new())
}
/// Allocate all lane and saved credits together. Compare every byte. Trace
/// hashes retain bounded receipts instead of complete checkpoint-size banks.
pub fn exercise<S: Store>(pool: &Pool<S>,boundary: u64) -> Result<String> {
    need(boundary>0 && boundary<pool.layout().capacity(),"saved boundary must be inside capacity")?;
    let mut out=String::new(); let mut alone=pool.session()?; let mut chunks=pool.session()?; let mut peer=pool.session()?;
    need(pool.session().is_err(),"lane credit limit was not enforced")?;
    check(&alone,&mut out)?;
    for _ in 0..pool.layout().capacity() {advance(&mut alone)?;check(&alone,&mut out)?;}
    need(alone.begin_append().is_err(),"token capacity was not enforced")?;
    // Different lane progress must not alter a producer's committed bank.
    for _ in 0..boundary {advance(&mut chunks)?;advance(&mut peer)?;}
    let saved=chunks.save()?;
    need(chunks.save().is_err(),"saved credit limit was not enforced")?;
    rejected(&mut chunks)?;
    for _ in boundary..pool.layout().capacity() {advance(&mut chunks)?;}
    check(&chunks,&mut out)?;
    check(&peer,&mut String::new())?;drop(chunks); drop(peer);
    let mut restored=pool.session()?;restored.restore(&saved)?;check(&restored,&mut out)?;
    // Exercise saved rollback across a physical pool completion boundary.
    advance(&mut restored)?; restored.truncate(&saved)?;
    for start in (boundary..pool.layout().capacity()).step_by(2) {
        for _ in start..(start+2).min(pool.layout().capacity()) {advance(&mut restored)?;}
    }
    check(&restored,&mut out)?;
    drop(restored);drop(saved);drop(alone);
    need(pool.used_slots()==(0,0),"owner credits leaked")?; Ok(out)
}
