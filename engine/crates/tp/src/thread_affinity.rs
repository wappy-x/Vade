use crate::{Result,runtime,transport::Ring};
use std::{collections::{BTreeMap,BTreeSet},fs,marker::PhantomData,path::Path,rc::Rc};

const CPU_LIMIT:usize=1024;
const HOST_BYTES:usize=36;
const VERSION:u8=3;
const MASK_AT:usize=1+HOST_BYTES+8;
const FAR_AT:usize=MASK_AT+CPU_LIMIT/8;
const DOMAIN_AT:usize=FAR_AT+CPU_LIMIT*2;
const RECORD_BYTES:usize=(DOMAIN_AT+CPU_LIMIT/8).next_multiple_of(8);
const CPU_ROOT:&str="/sys/devices/system/cpu";
struct Snapshot {allowed:Vec<usize>,record:Record}
/// One rank's roles: near fast cores (mHC candidates), far fast cores (decode) and the far L3 domain.
#[derive(Clone,Debug)]
struct Record {host:Vec<u8>,near:Vec<usize>,far:Vec<usize>,domain:Vec<usize>,workers:usize}
#[derive(Clone,Debug,PartialEq,Eq)]
struct Assignment {host:Vec<usize>,decode:Vec<usize>,background:Vec<usize>}
#[derive(Clone,Debug)]
struct CudaThread {tid:usize,name:String}
struct Domain {members:Vec<usize>,fast:Vec<usize>,bytes:usize}
pub struct Placement {rank:usize,allowed:Vec<usize>,assignment:Option<Assignment>}
pub struct WorkerMask {previous:Option<Vec<usize>>,_thread:PhantomData<Rc<()>>}
/// The mask a thread sets for itself, and the initial mask it keeps if the kernel refuses that mask.
pub struct ThreadMask {rank:usize,wanted:Vec<usize>,fallback:Vec<usize>}

fn parse_allowed(value:&str)->Result<Vec<usize>> {
    let mut cpus=BTreeSet::new();
    for part in value.trim().split(',') {
        let (first,last)=match part.split_once('-') {Some((first,last))=>(first,last),None=>(part,part)};
        let first=first.parse::<usize>().map_err(|_|"thread affinity: invalid allowed CPU list")?;
        let last=last.parse::<usize>().map_err(|_|"thread affinity: invalid allowed CPU list")?;
        if first>last || last>=CPU_LIMIT {return Err("thread affinity: allowed CPU range exceeds its mask".into());}
        cpus.extend(first..=last);
    }
    if cpus.is_empty() {return Err("thread affinity: no allowed CPUs".into());}Ok(cpus.into_iter().collect())
}
fn status_allowed(path:&str)->Result<Vec<usize>> {
    let status=fs::read_to_string(path).map_err(|error|format!("thread affinity: cannot read CPU mask: {error}"))?;
    parse_allowed(status.lines().find_map(|line|line.strip_prefix("Cpus_allowed_list:")).ok_or("thread affinity: CPU mask is absent")?)
}
fn current_allowed()->Result<Vec<usize>> {status_allowed("/proc/thread-self/status")}
fn thread_allowed(tid:usize)->Result<Vec<usize>> {status_allowed(&format!("/proc/self/task/{tid}/status"))}
fn fast_allowed(allowed:&[usize],speeds:&[(usize,u64)])->Result<Vec<usize>> {
    let fastest=speeds.iter().map(|&(_,speed)|speed).max().ok_or("thread affinity: no CPU frequency is available")?;
    Ok(speeds.iter().filter(|&&(cpu,speed)|speed==fastest && allowed.binary_search(&cpu).is_ok()).map(|&(cpu,_)|cpu).collect())
}
fn cache_size(value:&str)->Option<usize> {
    let value=value.trim();let (number,scale)=if let Some(number)=value.strip_suffix('K') {(number,1024)}
        else if let Some(number)=value.strip_suffix('M') {(number,1024*1024)} else {(value,1)};
    number.parse::<usize>().ok()?.checked_mul(scale)
}
fn cache_domain(root:&Path,cpu:usize)->Result<(Vec<usize>,usize)> {
    let entries=fs::read_dir(root.join(format!("cpu{cpu}/cache"))).map_err(|error|error.to_string())?;
    for entry in entries.flatten() {
        let path=entry.path();
        if fs::read_to_string(path.join("level")).ok().is_none_or(|value|value.trim()!="3") {continue;}
        let members=parse_allowed(&fs::read_to_string(path.join("shared_cpu_list")).map_err(|error|error.to_string())?)?;
        if !members.contains(&cpu) {return Err("thread affinity: inconsistent L3 membership".into());}
        let size=fs::read_to_string(path.join("size")).ok().and_then(|value|cache_size(&value)).filter(|&bytes|bytes>0);
        return Ok((members,size.ok_or("thread affinity: L3 size is absent")?));
    }
    Err(format!("thread affinity: CPU {cpu} has no shared L3 metadata"))
}
fn cuda_threads()->Result<Vec<CudaThread>> {
    let mut threads=Vec::new();
    for entry in fs::read_dir("/proc/self/task").map_err(|error|error.to_string())?.flatten() {
        let Ok(name)=fs::read_to_string(entry.path().join("comm")) else {continue;};
        if !name.starts_with("cuda") {continue;}
        if let Ok(tid)=entry.file_name().to_string_lossy().parse::<usize>() {threads.push(CudaThread {tid,name:name.trim().to_owned()});}
    }
    threads.sort_by_key(|thread|thread.tid);Ok(threads)
}
/// The first fast L3 domain that holds the mHC workers and has a disjoint fast partner. The partner with the
/// larger L3 is the far domain: decode gets its fast cores, the wire and CUDA workers its whole allowed set.
fn choose_domains(domains:&[Domain],workers:usize)->Result<(Vec<usize>,Vec<usize>,Vec<usize>)> {
    for near in domains {
        if near.fast.len()<workers {continue;}
        let mut choices=domains.iter().filter(|far|far.members.iter().all(|cpu|!near.members.contains(cpu))).collect::<Vec<_>>();
        choices.sort_by_key(|far|std::cmp::Reverse(far.bytes));
        if let Some(far)=choices.first() {return Ok((near.fast.clone(),far.fast.clone(),far.members.clone()));}
    }
    Err("thread affinity: separate fast L3 domains cannot fit mHC and decode".into())
}
/// Role domains of one rank from a sysfs CPU tree. Offline CPUs leave every mask; a missing maximum frequency
/// or L3 record turns the whole map off.
fn plan(root:&Path,initial:&[usize],workers:usize)->Result<(Vec<usize>,Vec<usize>,Vec<usize>)> {
    let read=|name:String|fs::read_to_string(root.join(&name)).map_err(|error|format!("thread affinity: {name}: {error}"));
    let online=parse_allowed(&read("online".into())?)?;
    let allowed=initial.iter().copied().filter(|cpu|online.contains(cpu)).collect::<Vec<_>>();
    let speeds=online.iter().map(|&cpu| {
        let speed=read(format!("cpu{cpu}/cpufreq/cpuinfo_max_freq")).ok().and_then(|value|value.trim().parse::<u64>().ok()).filter(|&value|value>0);
        speed.map(|speed|(cpu,speed)).ok_or_else(||format!("thread affinity: CPU {cpu} maximum frequency is absent"))
    }).collect::<Result<Vec<_>>>()?;
    let mut groups=BTreeMap::<Vec<usize>,Domain>::new();
    for cpu in fast_allowed(&allowed,&speeds)? {
        let (members,bytes)=cache_domain(root,cpu)?;
        groups.entry(members.clone()).or_insert_with(||Domain {members,fast:Vec::new(),bytes}).fast.push(cpu);
    }
    let (near,far,members)=choose_domains(&groups.into_values().collect::<Vec<_>>(),workers)?;
    Ok((near,far,allowed.into_iter().filter(|cpu|members.contains(cpu)).collect()))
}
fn snapshot(workers:usize)->Result<Snapshot> {
    let allowed=current_allowed()?;
    if cuda_threads()?.is_empty() {return Err("thread affinity: CUDA worker threads are not visible".into());}
    let (near,far,domain)=plan(Path::new(CPU_ROOT),&allowed,workers)?;
    let host=fs::read_to_string("/proc/sys/kernel/random/boot_id").map_err(|error|error.to_string())?.trim().as_bytes().to_vec();
    if host.len()!=HOST_BYTES || !host.iter().all(|byte|byte.is_ascii_hexdigit() || *byte==b'-') {return Err("thread affinity: invalid kernel identity".into());}
    Ok(Snapshot {allowed,record:Record {host,near,far,domain,workers}})
}
fn encode(record:&Record)->Vec<u8> {
    let mut bytes=vec![0;RECORD_BYTES];bytes[0]=VERSION;bytes[1..1+HOST_BYTES].copy_from_slice(&record.host);
    bytes[1+HOST_BYTES..MASK_AT].copy_from_slice(&(record.workers as u64).to_le_bytes());
    for &cpu in &record.near {bytes[MASK_AT+cpu/8]|=1<<(cpu%8);}
    for &cpu in &record.domain {bytes[DOMAIN_AT+cpu/8]|=1<<(cpu%8);}
    bytes[FAR_AT..FAR_AT+CPU_LIMIT*2].fill(255);
    for (index,&cpu) in record.far.iter().enumerate() {bytes[FAR_AT+index*2..FAR_AT+index*2+2].copy_from_slice(&(cpu as u16).to_le_bytes());}bytes
}
fn mask_at(row:&[u8],at:usize)->Vec<usize> {(0..CPU_LIMIT).filter(|&cpu|row[at+cpu/8]>>(cpu%8)&1!=0).collect()}
fn decode(bytes:&[u8])->Result<Vec<Record>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(RECORD_BYTES) {return Err("thread affinity: invalid placement records".into());}
    bytes.chunks_exact(RECORD_BYTES).map(|row| {
        if row[0]!=VERSION {return Err("thread affinity: a rank lacks compatible CPU or cache metadata".into());}
        let workers=usize::try_from(u64::from_le_bytes(row[1+HOST_BYTES..MASK_AT].try_into().unwrap())).map_err(|_|"thread affinity: worker count overflows")?;
        if workers>CPU_LIMIT {return Err("thread affinity: invalid worker count".into());}
        let (near,domain)=(mask_at(row,MASK_AT),mask_at(row,DOMAIN_AT));let mut far=Vec::new();let mut unique=BTreeSet::new();
        for part in row[FAR_AT..FAR_AT+CPU_LIMIT*2].chunks_exact(2) {
            let cpu=u16::from_le_bytes(part.try_into().unwrap());if cpu==u16::MAX {break;}let cpu=cpu as usize;
            if cpu>=CPU_LIMIT || near.contains(&cpu) || !unique.insert(cpu) {return Err("thread affinity: overlapping or invalid L3 domains".into());}far.push(cpu);
        }
        if far.iter().any(|cpu|!domain.contains(cpu)) || domain.iter().any(|cpu|near.contains(cpu)) {
            return Err("thread affinity: decode cores lie outside the far domain, or the domains overlap".into());
        }
        Ok(Record {host:row[1..1+HOST_BYTES].to_vec(),near,far,domain,workers})
    }).collect()
}
fn allocate(records:&[Record])->Result<Vec<Assignment>> {
    let mut used=BTreeMap::<&[u8],BTreeSet<usize>>::new();let mut assignments=Vec::new();
    for record in records {
        let reserved=used.entry(&record.host).or_default();
        let host=record.near.iter().copied().filter(|cpu|!reserved.contains(cpu)).take(record.workers).collect::<Vec<_>>();
        let mut decode=record.far.iter().copied().filter(|cpu|!reserved.contains(cpu)).collect::<Vec<_>>();
        if host.len()!=record.workers || decode.is_empty() {return Err("thread affinity: rank-disjoint L3 reservations do not fit; no partial placement".into());}
        reserved.extend(&host);reserved.extend(&decode);decode.sort_unstable();assignments.push(Assignment {host,decode,background:Vec::new()});
    }
    // Spare CPUs of the far domain go once to the first co-located rank that can use them.
    for (record,assignment) in records.iter().zip(&mut assignments) {
        let reserved=used.entry(&record.host).or_default();
        let mut background=record.domain.iter().copied().filter(|cpu|!reserved.contains(cpu)).collect::<Vec<_>>();
        reserved.extend(&background);background.extend(&assignment.decode);background.sort_unstable();assignment.background=background;
    }Ok(assignments)
}
fn move_cuda_threads(rank:usize,mask:&[usize]) {
    let threads=match cuda_threads() {Ok(threads)=>threads,Err(error)=>{eprintln!("TP AFFINITY rank={rank} CUDA placement unavailable: {error}");return;}};
    for thread in threads {
        if thread_allowed(thread.tid).is_ok_and(|allowed|allowed==mask) {continue;}
        match runtime::pin_thread_cpus(thread.tid,mask) {
            Ok(())=>eprintln!("TP AFFINITY rank={rank} cuda={} tid={} mask={mask:?}",thread.name,thread.tid),
            Err(error)=>eprintln!("TP AFFINITY rank={rank} CUDA thread unpinned: {error}"),
        }
    }
}
impl Placement {
    pub fn new(ring:&mut Ring,workers:usize)->Result<Self> {
        let local=snapshot(workers);if let Err(error)=&local {eprintln!("TP AFFINITY rank={} unpinned: {error}",ring.rank());}
        let mut records=vec![0;ring.world()*RECORD_BYTES];
        if let Ok(local)=&local {records[ring.rank()*RECORD_BYTES..(ring.rank()+1)*RECORD_BYTES].copy_from_slice(&encode(&local.record));}
        ring.all_gather(&mut records,RECORD_BYTES)?;
        let assignment=match decode(&records).and_then(|records|allocate(&records)) {
            Ok(mut assignments)=>Some(assignments.swap_remove(ring.rank())),
            Err(error)=>{eprintln!("TP AFFINITY rank={} unpinned: {error}",ring.rank());None}
        };
        let allowed=local.map(|local|local.allowed).unwrap_or_default();
        if let Some(assignment)=&assignment {
            eprintln!("TP AFFINITY rank={} mhc={:?} decode_mask={:?} wire_cuda_mask={:?}",ring.rank(),assignment.host,assignment.decode,assignment.background);
        }
        Ok(Self {rank:ring.rank(),allowed,assignment})
    }
    pub fn workers(&mut self)->WorkerMask {
        let previous=match &self.assignment {
            Some(assignment) if !assignment.host.is_empty()=>match runtime::pin_cpus(&assignment.host) {
                Ok(())=>Some(self.allowed.clone()),
                Err(error)=>{eprintln!("TP AFFINITY rank={} unpinned: {error}",self.rank);self.assignment=None;None}
            },
            _=>None
        };WorkerMask {previous,_thread:PhantomData}
    }
    pub fn wire_mask(&self)->Option<ThreadMask> {
        self.assignment.as_ref().map(|assignment|ThreadMask {rank:self.rank,wanted:assignment.background.clone(),fallback:self.allowed.clone()})
    }
    /// Moves every CUDA worker whose mask differs from the far-domain mask, including workers started after decode.
    pub fn refresh_cuda(&self) {if let Some(assignment)=&self.assignment {move_cuda_threads(self.rank,&assignment.background);}}
    pub fn pin_decode(&self) {
        self.refresh_cuda();
        if let Some(assignment)=&self.assignment && let Err(error)=runtime::pin_cpus(&assignment.decode) {eprintln!("TP AFFINITY rank={} decode unpinned: {error}",self.rank);}
    }
}
impl ThreadMask {
    /// Places the calling thread and returns the mask it keeps. A thread started under the temporary host-worker
    /// mask must not stay on the mHC cores, so a refused role mask falls back to the rank's initial mask.
    pub fn apply(&self,role:&str)->Vec<usize> {
        let Err(error)=runtime::pin_cpus(&self.wanted) else {return self.wanted.clone();};
        match runtime::pin_cpus(&self.fallback) {
            Ok(())=>eprintln!("TP AFFINITY rank={} {role} kept the initial mask {:?}: {error}",self.rank,self.fallback),
            Err(second)=>eprintln!("TP AFFINITY rank={} {role} kept its inherited mask: {error}; {second}",self.rank),
        }
        current_allowed().unwrap_or_default()
    }
}
impl Drop for WorkerMask {
    fn drop(&mut self) {if let Some(previous)=&self.previous && let Err(error)=runtime::pin_cpus(previous) {eprintln!("TP AFFINITY: cannot restore main CPU mask: {error}");}}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(host:u8)->Record {Record {host:vec![host;HOST_BYTES],near:vec![5,6,7,8,9],far:vec![15,16,17,18,19],domain:(10..20).collect(),workers:3}}
    struct Tree(std::path::PathBuf);
    impl Drop for Tree {fn drop(&mut self) {let _=fs::remove_dir_all(&self.0);}}
    /// A GB10-shaped sysfs CPU tree: 0-4 and 10-14 slow, 5-9 and 15-19 fast; L3 0-9 (8 MiB) and 10-19 (16 MiB).
    fn gb10_tree(name:&str,online:&str,without_speed:Option<usize>)->Tree {
        let root=std::env::temp_dir().join(format!("vade-affinity-{name}-{}",std::process::id()));let _=fs::remove_dir_all(&root);
        for cpu in 0..20 {
            let dir=root.join(format!("cpu{cpu}"));fs::create_dir_all(dir.join("cache/index3")).unwrap();fs::create_dir_all(dir.join("cpufreq")).unwrap();
            let khz=if cpu%10>=5 {3900000} else {2808000};let (list,size)=if cpu<10 {("0-9","8192K")} else {("10-19","16384K")};
            if without_speed!=Some(cpu) {fs::write(dir.join("cpufreq/cpuinfo_max_freq"),format!("{khz}\n")).unwrap();}
            for (file,value) in [("level","3"),("shared_cpu_list",list),("size",size)] {fs::write(dir.join("cache/index3").join(file),format!("{value}\n")).unwrap();}
        }
        fs::write(root.join("online"),format!("{online}\n")).unwrap();Tree(root)
    }
    #[cfg(target_os="linux")]
    fn thread_id()->usize {fs::read_link("/proc/thread-self").unwrap().file_name().unwrap().to_string_lossy().parse().unwrap()}
    #[test] fn parses_allowed_cpu_ranges() {
        assert_eq!(parse_allowed("0-4,8,15-19").unwrap(),vec![0,1,2,3,4,8,15,16,17,18,19]);
        for invalid in ["","3-1","0-1024","-1","1,,2"] {assert!(parse_allowed(invalid).is_err());}
    }
    #[test] fn slower_only_masks_are_not_fast() {
        let speeds=[(0,2808000),(1,2808000),(5,3900000),(6,3900000)];
        assert!(fast_allowed(&[0,1],&speeds).unwrap().is_empty());assert_eq!(fast_allowed(&[0,6],&speeds).unwrap(),vec![6]);
    }
    #[test] fn cache_sizes_come_from_metadata() {assert_eq!(cache_size("8192K\n"),Some(8388608));assert_eq!(cache_size("16M"),Some(16777216));assert!(cache_size("unknown").is_none());}
    #[test] fn reference_tree_maps_mhc_decode_and_the_far_domain() {
        let tree=gb10_tree("reference","0-19",None);let (near,far,domain)=plan(&tree.0,&(0..20).collect::<Vec<_>>(),3).unwrap();
        assert_eq!(near,vec![5,6,7,8,9]);assert_eq!(far,vec![15,16,17,18,19]);assert_eq!(domain,(10..20).collect::<Vec<_>>());
    }
    #[test] fn missing_cpu_frequency_turns_the_map_off() {let tree=gb10_tree("no-speed","0-19",Some(12));assert!(plan(&tree.0,&(0..20).collect::<Vec<_>>(),3).is_err());}
    #[test] fn offline_cpus_stay_out_of_every_mask() {
        let tree=gb10_tree("offline","0-18",None);let (near,far,domain)=plan(&tree.0,&(0..20).collect::<Vec<_>>(),3).unwrap();
        assert_eq!(near,vec![5,6,7,8,9]);assert_eq!(far,vec![15,16,17,18]);assert_eq!(domain,(10..19).collect::<Vec<_>>());
    }
    #[test] fn container_cpuset_bounds_every_mask() {
        let tree=gb10_tree("cpuset","0-19",None);let (near,far,domain)=plan(&tree.0,&[0,1,5,6,7,12,15,16],3).unwrap();
        assert_eq!(near,vec![5,6,7]);assert_eq!(far,vec![15,16]);assert_eq!(domain,vec![12,15,16]);
        assert!(plan(&tree.0,&[5,6,7,8,9],3).is_err());assert!(plan(&tree.0,&[5,6,15,16],3).is_err());
    }
    #[test] fn physical_hosts_keep_wire_and_cuda_in_the_decode_cache_domain() {
        for assignment in allocate(&[record(b'a'),record(b'b')]).unwrap() {
            assert_eq!(assignment.host,vec![5,6,7]);assert_eq!(assignment.decode,vec![15,16,17,18,19]);
            assert_eq!(assignment.background,(10..20).collect::<Vec<_>>());assert!(assignment.background.iter().all(|cpu|*cpu>=10));
        }
    }
    #[test] fn colocated_overcommit_falls_back_instead_of_sharing_cores() {assert!(allocate(&[record(b'a'),record(b'a')]).is_err());}
    #[test] fn one_far_fast_core_is_enough_for_decode() {
        let record=Record {far:vec![15],domain:vec![10,11,15],..record(b'a')};let assignment=allocate(&[record]).unwrap().remove(0);
        assert_eq!(assignment.decode,vec![15]);assert_eq!(assignment.background,vec![10,11,15]);
    }
    #[test] fn numbering_does_not_define_the_domains() {
        let record=Record {host:vec![b'a';HOST_BYTES],near:vec![21,23,25],far:vec![2,4,6,8],domain:vec![1,2,3,4,6,8],workers:3};
        let assignment=allocate(&[record]).unwrap().remove(0);assert_eq!(assignment.host,vec![21,23,25]);
        assert_eq!(assignment.decode,vec![2,4,6,8]);assert_eq!(assignment.background,vec![1,2,3,4,6,8]);
    }
    #[test] fn missing_separate_domain_is_a_fallback() {
        let domains=[Domain {members:(0..10).collect(),fast:(5..10).collect(),bytes:8<<20}];assert!(choose_domains(&domains,3).is_err());
    }
    #[test] fn partial_shared_cache_overlap_is_not_isolation() {
        let domains=[Domain {members:(0..10).collect(),fast:(5..10).collect(),bytes:8<<20},Domain {members:(5..20).collect(),fast:(15..20).collect(),bytes:16<<20}];
        assert!(choose_domains(&domains,3).is_err());
    }
    #[test] fn record_round_trip() {
        let record=record(b'a');let decoded=decode(&encode(&record)).unwrap().remove(0);
        assert_eq!(decoded.host,record.host);assert_eq!(decoded.near,record.near);assert_eq!(decoded.far,record.far);
        assert_eq!(decoded.domain,record.domain);assert_eq!(decoded.workers,3);
        assert!(decode(&vec![0;RECORD_BYTES]).is_err());assert!(decode(&[0]).is_err());
    }
    #[test] fn invalid_domain_overlap_is_refused() {
        let mut decode_in_near=record(b'a');decode_in_near.far[0]=5;assert!(decode(&encode(&decode_in_near)).is_err());
        let mut domain_in_near=record(b'a');domain_in_near.domain.push(9);assert!(decode(&encode(&domain_in_near)).is_err());
    }
    #[test] fn masks_do_not_expand_a_restricted_initial_set() {
        let mut record=record(b'a');record.domain=vec![15,16,17,18,19];record.near=vec![5,6,7];
        let assignment=allocate(&decode(&encode(&record)).unwrap()).unwrap().remove(0);assert_eq!(assignment.background,assignment.decode);
        record.domain.retain(|&cpu|cpu!=19);assert!(decode(&encode(&record)).is_err());
    }
    #[test] fn colocated_background_masks_partition_spare_cpus() {
        let first=Record {host:vec![b'a';HOST_BYTES],near:vec![0,1,2],far:vec![4,5,6,7],domain:vec![3,4,5,6,7,11],workers:1};
        let second=Record {near:vec![8,9,10],far:vec![12,13,14,15],domain:vec![3,11,12,13,14,15],..first.clone()};
        let assignments=allocate(&[first,second]).unwrap();assert_eq!(assignments[0].background,vec![3,4,5,6,7,11]);
        for (index,assignment) in assignments.iter().enumerate() {
            let other=&assignments[1-index];assert!(assignment.decode.iter().all(|cpu|assignment.background.contains(cpu)));
            assert!(assignment.background.iter().all(|cpu|!assignment.host.contains(cpu) && !other.host.contains(cpu) && !other.background.contains(cpu)));
        }
    }
    #[test] fn deferred_pin_is_thread_local() {
        let guard=runtime::defer_cpu_pin();assert!(runtime::cpu_pin_deferred());
        std::thread::spawn(||assert!(!runtime::cpu_pin_deferred())).join().unwrap();drop(guard);assert!(!runtime::cpu_pin_deferred());
    }
    /// Runs two child copies of this test binary with TP_CPU set; setting it in this process would need unsafe.
    #[cfg(target_os="linux")]
    #[test] fn early_tp_cpu_does_not_narrow_managed_startup() {
        let cpu=current_allowed().unwrap()[0].to_string();let binary=std::env::current_exe().unwrap();
        for child in ["child_guarded_pin_keeps_the_mask","child_unguarded_pin_reads_tp_cpu"] {
            let name=format!("thread_affinity::tests::{child}");
            let output=std::process::Command::new(&binary).args([name.as_str(),"--exact","--ignored","--test-threads=1"]).env("TP_CPU",&cpu).output().unwrap();
            let text=String::from_utf8_lossy(&output.stdout);assert!(output.status.success() && text.contains("1 passed"),"{child}: {text}");
        }
    }
    #[cfg(target_os="linux")]
    #[test] #[ignore="child of early_tp_cpu_does_not_narrow_managed_startup, which sets TP_CPU"]
    fn child_guarded_pin_keeps_the_mask() {
        let before=current_allowed().unwrap();let _guard=runtime::defer_cpu_pin();runtime::pin().unwrap();assert_eq!(current_allowed().unwrap(),before);
    }
    #[cfg(target_os="linux")]
    #[test] #[ignore="child of early_tp_cpu_does_not_narrow_managed_startup, which sets TP_CPU"]
    fn child_unguarded_pin_reads_tp_cpu() {runtime::pin().unwrap();assert_eq!(current_allowed().unwrap().len(),1);}
    #[cfg(target_os="linux")]
    #[test] fn refused_affinity_reports_and_keeps_the_mask() {
        std::thread::spawn(|| {
            let before=current_allowed().unwrap();assert!(!before.contains(&(CPU_LIMIT-1)));
            assert!(runtime::pin_cpus(&[CPU_LIMIT-1]).is_err());assert_eq!(current_allowed().unwrap(),before);
        }).join().unwrap();
    }
    #[cfg(target_os="linux")]
    #[test] fn refused_wire_mask_falls_back_to_the_initial_mask() {
        std::thread::spawn(|| {
            let initial=current_allowed().unwrap();runtime::pin_cpus(&initial[..1]).unwrap();
            let mask=ThreadMask {rank:0,wanted:vec![CPU_LIMIT-1],fallback:initial.clone()};
            assert_eq!(mask.apply("wire"),initial);assert_eq!(current_allowed().unwrap(),initial);
        }).join().unwrap();
    }
    #[cfg(target_os="linux")]
    #[test] fn cuda_thread_started_after_decode_placement_is_moved() {
        std::thread::spawn(|| {
            let initial=current_allowed().unwrap();assert!(initial.len()>1,"needs two allowed CPUs");
            let assignment=Assignment {host:Vec::new(),decode:initial[..1].to_vec(),background:initial.clone()};
            let placement=Placement {rank:0,allowed:initial.clone(),assignment:Some(assignment)};
            placement.pin_decode();assert_eq!(current_allowed().unwrap(),initial[..1]);
            let (send,receive)=std::sync::mpsc::channel();let (done,wait)=std::sync::mpsc::channel::<()>();
            let late=std::thread::Builder::new().name("cuda-late-test".into()).spawn(move|| {send.send(thread_id()).unwrap();wait.recv().unwrap();}).unwrap();
            let tid=receive.recv().unwrap();assert_eq!(thread_allowed(tid).unwrap(),initial[..1]);
            placement.refresh_cuda();assert_eq!(thread_allowed(tid).unwrap(),initial);done.send(()).unwrap();late.join().unwrap();
        }).join().unwrap();
    }
    #[cfg(target_os="linux")]
    #[test] fn owned_thread_affinity_targets_only_that_thread() {
        use std::sync::{Arc,Barrier,mpsc};
        let initial=current_allowed().unwrap();let cpu=initial[0];let ready=Arc::new(Barrier::new(2));let worker_ready=ready.clone();let (send,receive)=mpsc::channel();
        let worker=std::thread::Builder::new().name("owned-pin-test".into()).spawn(move|| {send.send(thread_id()).unwrap();worker_ready.wait();current_allowed().unwrap()}).unwrap();
        let tid=receive.recv().unwrap();runtime::pin_thread_cpus(tid,&[cpu]).unwrap();assert_eq!(current_allowed().unwrap(),initial);
        ready.wait();assert_eq!(worker.join().unwrap(),vec![cpu]);assert!(runtime::pin_thread_cpus(usize::MAX,&[cpu]).is_err());
    }
    #[cfg(target_os="linux")]
    #[test]
    #[ignore="requires an authorized CUDA device"]
    fn cuda_context_workers_use_the_decode_cache_domain() {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),64<<20,|backend| {
            let _mailbox=backend.tp_mailbox(4096)?;let snapshot=snapshot(3)?;let near=snapshot.record.near.clone();
            let assignment=allocate(&[snapshot.record])?.remove(0);let initial=snapshot.allowed;
            let restore=WorkerMask {previous:Some(initial.clone()),_thread:PhantomData};
            let mut placement=Placement {rank:0,allowed:initial.clone(),assignment:Some(assignment.clone())};
            let workers=placement.workers();assert_eq!(current_allowed()?,assignment.host);
            let wire=placement.wire_mask().ok_or("no wire mask")?;let worker=std::thread::spawn(move||wire.apply("wire"));
            assert_eq!(worker.join().map_err(|_|"wire affinity test panicked")?,assignment.background);drop(workers);assert_eq!(current_allowed()?,initial);
            placement.pin_decode();assert_eq!(current_allowed()?,assignment.decode);
            let threads=cuda_threads()?;if threads.is_empty() {return Err("CUDA worker test found no threads".into());}
            for thread in threads {
                let allowed=thread_allowed(thread.tid)?;assert_eq!(allowed,assignment.background);
                assert!(allowed.iter().all(|cpu|!near.contains(cpu)));assert!(allowed.len()>1);
            }drop(restore);Ok(())
        }).unwrap();
    }
}
