//! Check conditional physical allocations and opaque continuation at plan dimensions.
#[cfg_attr(not(target_os="linux"),allow(dead_code))]
#[path="session-plan-check/fixture.rs"]
mod fixture;
use fixture::{Result,need};
use engine_model::plan::layer::Topology;
use spark_tp::session_state::Budget;
use std::{fs,path::{Path,PathBuf},process::Command};
fn main() {if let Err(e)=run() {eprintln!("session-plan-check: {e}");std::process::exit(1);}}
fn number<T: std::str::FromStr>(s: &str) -> Result<T> {s.parse().map_err(|_|"invalid state probe number".into())}
fn run() -> Result<()> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if let [command,left,right]=args.as_slice()&& command=="compare" {return compare(Path::new(left),Path::new(right));}
    if let [command,config,types,capacity,parts,boundary,destination]=args.as_slice()
        && command=="suite" {return suite([config,types,capacity,parts,boundary],Path::new(destination));}
    let [command,config,types,capacity,parts,rank,boundary,destination]=args.as_slice() else {
        return Err("usage: session-plan-check <expected|device> <config> <types> <capacity> <parts> <rank> <saved-boundary> <new-dir>".into());
    };
    need(command=="expected" || command=="device","unknown state plan command")?;
    let config=fs::read(config).map_err(|e|e.to_string())?;let types=fs::read_to_string(types).map_err(|e|e.to_string())?;
    let topology=Topology {parts:number(parts)?,rank:number(rank)?};let boundary=number(boundary)?;
    let l=fixture::layout(&config,&types,number(capacity)?,topology)?;
    need(boundary>0 && boundary<l.capacity(),"saved boundary must be inside capacity")?;
    let identity=fixture::identity(&config,&types,&l,topology,boundary);
    let budget=Budget {weights:0,runtime:0,reserve:0,lanes:3,saved:1,limit:isize::MAX as usize};
    let required=budget.required(&l)?;
    let expected=fixture::oracle(&l,boundary)?;
    let destination=Path::new(destination);fs::create_dir(destination).map_err(|e|e.to_string())?;
    fs::write(destination.join("IDENTITY"),identity).map_err(|e|e.to_string())?;
    fs::write(destination.join("SOURCE"),include_str!("../../../../REVISION")).map_err(|e|e.to_string())?;
    fs::write(destination.join("expected.txt"),&expected).map_err(|e|e.to_string())?;
    if command=="expected" {println!("SESSION_PLAN_HOST PASS required_device_bytes={required} producer_types_qualified=false model_ids=false");return Ok(());}
    #[cfg(target_os="linux")]
    {
        use spark_tp::session_state::device::DeviceStore;
        use spark_gpu_probe::driver::with_backend;
        need(!spark_gpu_probe::CUBIN_STAMP.is_empty(),"state proof requires an embedded prebuilt CUDA image")?;
        println!("SESSION_PLAN revision={} {}",include_str!("../../../../REVISION").trim(),spark_gpu_probe::CUBIN_STAMP);
        with_backend(0,spark_gpu_probe::layer_ptx()?,required,|backend| {
            let before=backend.accounted_bytes();
            let budget=Budget {runtime:before,limit:required.checked_add(before).ok_or("state device budget overflow")?,..budget};
            backend.set_allocation_budget(budget.limit)?;
            let mut short=budget;short.limit=budget.limit-1;
            need(DeviceStore::new(backend).admit(l.clone(),short).is_err(),"underfunded pool was admitted")?;
            need(backend.accounted_bytes()==before,"refused pool allocated device storage")?;
            let pool=DeviceStore::new(backend).admit(l,budget)?;
            let actual=fixture::exercise(&pool,boundary)?;
            need(actual==expected,"continuation trace differs from host reference")?;
            backend.synchronize()?;
            need(backend.accounted_bytes()==before,"session device storage leaked")?;
            need(backend.peak_accounted_bytes()==budget.limit,"allocated peak differs from admitted owner extents")?;
            fs::write(destination.join("actual.txt"),actual).map_err(|e|e.to_string())?;
            fs::write(destination.join("allocation.txt"),format!("required={} peak_device={} final_device={} peak_pinned={}\n",budget.limit,backend.peak_accounted_bytes(),backend.accounted_bytes(),backend.peak_pinned_staging_bytes())).map_err(|e|e.to_string())?;
            println!("SESSION_PLAN_DEVICE PASS required={} peak_device={} final_device={} peak_pinned={} producer_types_qualified=false model_ids=false",
                budget.limit,backend.peak_accounted_bytes(),backend.accounted_bytes(),backend.peak_pinned_staging_bytes());Ok(())
        })?;
        fs::write(destination.join("COMPLETE"),include_str!("../../../../REVISION")).map_err(|e|e.to_string())?;
        Ok(())
    }
    #[cfg(not(target_os="linux"))]
    {let _=expected;Err("device state gate requires a CUDA Linux host".into())}
}
fn cell(root: &Path,rank: u32,repeat: u32) -> PathBuf {root.join(format!("rank-{rank}-run-{repeat}"))}
fn read(root: &Path,name: &str) -> Result<Vec<u8>> {fs::read(root.join(name)).map_err(|e|e.to_string())}
fn same(left: &Path,right: &Path,name: &str) -> Result<()> {
    need(read(left,name)?==read(right,name)?,&format!("state comparison differs: {name}"))
}
fn ranks(root: &Path) -> Result<u32> {
    let count=number(&fs::read_to_string(root.join("PARTS")).map_err(|e|e.to_string())?)?;
    need(count>0,"state suite rank set is empty")?;Ok(count)
}
fn complete(root: &Path) -> Result<()> {
    let source=read(root,"SOURCE")?;
    need(!source.is_empty() && source==read(root,"COMPLETE")?,"state suite is incomplete")
}
fn compare(left: &Path,right: &Path) -> Result<()> {
    complete(left)?;complete(right)?;same(left,right,"SOURCE")?;
    same(left,right,"PARTS")?;let parts=ranks(left)?;
    for rank in 0..parts {for repeat in 0..3 {
        let a=cell(left,rank,repeat);let b=cell(right,rank,repeat);
        complete(&a)?;complete(&b)?;same(&a,left,"SOURCE")?;same(&b,right,"SOURCE")?;
        for name in ["SOURCE","IDENTITY","expected.txt","actual.txt"] {same(&a,&b,name)?;}
        need(read(&a,"expected.txt")?==read(&a,"actual.txt")?,"state suite differs from host reference")?;
    }}
    println!("SESSION_PLAN_COMPARE PASS parts={parts} repeats=3 model_ids=false");Ok(())
}
fn suite(args: [&str;5],destination: &Path) -> Result<()> {
    let [config,types,capacity,parts,boundary]=args;let parts: u32=number(parts)?;
    let bytes=fs::read(config).map_err(|e|e.to_string())?;let text=fs::read_to_string(types).map_err(|e|e.to_string())?;
    need(parts>0,"state suite rank set is empty")?;
    // Validate the complete rank set before starting a device process.
    let capacity: u64=number(capacity)?;let boundary_value: u64=number(boundary)?;
    need(boundary_value>0 && boundary_value<capacity,"saved boundary must be inside capacity")?;
    for rank in 0..parts {fixture::layout(&bytes,&text,capacity,Topology {parts,rank})?;}
    fs::create_dir(destination).map_err(|e|e.to_string())?;
    fs::write(destination.join("PARTS"),parts.to_string()).map_err(|e|e.to_string())?;
    let revision=include_str!("../../../../REVISION");
    fs::write(destination.join("SOURCE"),revision).map_err(|e|e.to_string())?;
    let executable=std::env::current_exe().map_err(|e|e.to_string())?;
    for rank in 0..parts {for repeat in 0..3 {
        let target=cell(destination,rank,repeat);
        let result=Command::new(&executable).args(["device",config,types,&capacity.to_string(),&parts.to_string(),&rank.to_string(),boundary])
            .arg(&target).output().map_err(|e|e.to_string())?;
        let mut log=result.stdout;log.extend(result.stderr);
        fs::write(destination.join(format!("rank-{rank}-run-{repeat}.log")),&log).map_err(|e|e.to_string())?;
        need(result.status.success(),"state device process failed; keep its log and stop the gate")?;
        need(String::from_utf8_lossy(&log).contains("gpu_backend module=prebuilt_cubin"),"state process did not load its prebuilt CUDA image")?;
        complete(&target)?;same(&target,destination,"SOURCE")?;
        need(read(&target,"expected.txt")?==read(&target,"actual.txt")?,"state bytes differ from host reference")?;
        let first=cell(destination,rank,0);
        for name in ["SOURCE","IDENTITY","expected.txt","actual.txt"] {same(&first,&target,name)?;}
        println!("SESSION_PLAN_CELL PASS rank={rank} repeat={repeat} fresh_process=true");
    }}
    fs::write(destination.join("COMPLETE"),revision).map_err(|e|e.to_string())?;
    println!("SESSION_PLAN_SUITE PASS parts={parts} fresh_processes={} model_ids=false",u64::from(parts)*3);Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let path=std::env::temp_dir().join(format!("session-plan-proof-{}-{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
        fs::create_dir(&path).unwrap();Self(path)
    }}
    impl Drop for Temp {fn drop(&mut self) {let _=fs::remove_dir_all(&self.0);}}
    fn receipt(root: &Path) {
        fs::write(root.join("PARTS"),"2").unwrap();
        for name in ["SOURCE","COMPLETE"] {fs::write(root.join(name),"source\n").unwrap();}
        for rank in 0..2 {for repeat in 0..3 {
            let target=cell(root,rank,repeat);fs::create_dir(&target).unwrap();
            for name in ["SOURCE","COMPLETE"] {fs::write(target.join(name),"source\n").unwrap();}
            for (name,bytes) in [("IDENTITY","geometry"),("expected.txt","bytes"),("actual.txt","bytes")] {fs::write(target.join(name),bytes).unwrap();}
        }}
    }
    #[test]
    fn compare_requires_complete_matching_rank_sets_and_host_bytes() {
        let left=Temp::new();let right=Temp::new();receipt(&left.0);receipt(&right.0);
        compare(&left.0,&right.0).unwrap();
        let last=cell(&right.0,1,2);
        for (name,bad) in [("actual.txt","different"),("expected.txt","different"),("IDENTITY","different"),("SOURCE","stale")] {
            let prior=read(&last,name).unwrap();fs::write(last.join(name),bad).unwrap();
            assert!(compare(&left.0,&right.0).is_err());fs::write(last.join(name),prior).unwrap();
        }
        fs::remove_file(last.join("COMPLETE")).unwrap();assert!(compare(&left.0,&right.0).is_err());
        fs::write(last.join("COMPLETE"),"source\n").unwrap();
        fs::write(right.0.join("PARTS"),"3").unwrap();assert!(compare(&left.0,&right.0).is_err());
        fs::write(right.0.join("PARTS"),"2").unwrap();
        fs::remove_file(right.0.join("COMPLETE")).unwrap();assert!(compare(&left.0,&right.0).is_err());
    }
}
