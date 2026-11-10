use crate::{Result,runtime,transport::Ring};
use std::{cell::RefCell,collections::{BTreeMap,BTreeSet},fs,marker::PhantomData,rc::Rc};

const CPU_LIMIT:usize=1024;
const HOST_BYTES:usize=36;
const MASK_AT:usize=1+HOST_BYTES+16;
const FAR_AT:usize=MASK_AT+CPU_LIMIT/8;
const ALLOWED_AT:usize=FAR_AT+CPU_LIMIT*2;
const RECORD_BYTES:usize=(ALLOWED_AT+CPU_LIMIT/8).next_multiple_of(8);
struct Snapshot {allowed:Vec<usize>,record:Record,irq_known:bool}
#[derive(Clone,Debug)]
struct Record {host:Vec<u8>,near:Vec<usize>,far:Vec<usize>,allowed:Vec<usize>,workers:usize,cuda:usize}
#[derive(Clone,Debug,PartialEq,Eq)]
struct Assignment {host:Vec<usize>,decode:Vec<usize>,background:Vec<usize>}
#[derive(Clone,Debug)]
struct CudaThread {tid:usize,name:String}
struct Domain {members:Vec<usize>,fast:Vec<usize>,bytes:usize}
pub struct Placement {rank:usize,allowed:Vec<usize>,assignment:Option<Assignment>,pinned:RefCell<BTreeSet<usize>>}
pub struct WorkerMask {previous:Option<Vec<usize>>,_thread:PhantomData<Rc<()>>}

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
fn current_allowed()->Result<Vec<usize>> {
    let status=fs::read_to_string("/proc/thread-self/status").map_err(|error|format!("thread affinity: cannot read current CPU mask: {error}"))?;
    parse_allowed(status.lines().find_map(|line|line.strip_prefix("Cpus_allowed_list:")).ok_or("thread affinity: current CPU mask is absent")?)
}
fn fast_allowed(allowed:&[usize],speeds:&[(usize,u64)])->Result<Vec<usize>> {
    let fastest=speeds.iter().map(|&(_,speed)|speed).max().ok_or("thread affinity: no CPU frequency is available")?;
    Ok(speeds.iter().filter(|&&(cpu,speed)|speed==fastest && allowed.binary_search(&cpu).is_ok()).map(|&(cpu,_)|cpu).collect())
}
fn cache_size(value:&str)->Option<usize> {
    let value=value.trim();let (number,scale)=if let Some(number)=value.strip_suffix('K') {(number,1024)}
        else if let Some(number)=value.strip_suffix('M') {(number,1024*1024)} else {(value,1)};
    number.parse::<usize>().ok()?.checked_mul(scale)
}
fn cache_domain(cpu:usize)->Result<(Vec<usize>,usize)> {
    let entries=fs::read_dir(format!("/sys/devices/system/cpu/cpu{cpu}/cache")).map_err(|error|error.to_string())?;
    for entry in entries.flatten() {
        let path=entry.path();
        if fs::read_to_string(path.join("level")).ok().is_none_or(|value|value.trim()!="3") {continue;}
        let members=parse_allowed(&fs::read_to_string(path.join("shared_cpu_list")).map_err(|error|error.to_string())?)?;
        if !members.contains(&cpu) {return Err("thread affinity: inconsistent L3 membership".into());}
        let bytes=fs::read_to_string(path.join("size")).ok().and_then(|value|cache_size(&value)).filter(|&bytes|bytes>0).ok_or("thread affinity: L3 size is absent")?;
        return Ok((members,bytes));
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
    threads.sort_by_key(|thread|(thread.name.contains("Evt"),thread.tid));Ok(threads)
}
fn irq_costs()->(Vec<u64>,bool) {
    let mut costs=vec![0u64;CPU_LIMIT];let mut irqs=BTreeSet::new();
    for category in ["/sys/class/infiniband","/sys/class/net"] {if let Ok(devices)=fs::read_dir(category) {for device in devices.flatten() {
        if let Ok(entries)=fs::read_dir(device.path().join("device/msi_irqs")) {for entry in entries.flatten() {
            if let Ok(irq)=entry.file_name().to_string_lossy().parse::<usize>() {irqs.insert(irq);}
        }}
    }}}
    let Ok(text)=fs::read_to_string("/proc/interrupts") else {return (costs,false);};
    let mut lines=text.lines();let columns=lines.next().unwrap_or("").split_whitespace().filter_map(|name|name.strip_prefix("CPU").and_then(|value|value.parse::<usize>().ok())).collect::<Vec<_>>();
    let known=!irqs.is_empty() && !columns.is_empty();
    for line in lines {let Some((number,values))=line.split_once(':') else {continue;};
        if !number.trim().parse::<usize>().ok().is_some_and(|irq|irqs.contains(&irq)) {continue;}
        for (&cpu,value) in columns.iter().zip(values.split_whitespace()) {if cpu<CPU_LIMIT && let Ok(value)=value.parse::<u64>() {costs[cpu]=costs[cpu].saturating_add(value);}}
    }(costs,known)
}
fn choose_domains(domains:&[Domain],workers:usize,cuda:usize,costs:&[u64])->Result<(Vec<usize>,Vec<usize>)> {
    for near in domains {
        if near.fast.len()<workers {continue;}
        let mut choices=domains.iter().filter(|far|far.members.iter().all(|cpu|!near.members.contains(cpu)) && far.fast.len()>=cuda+2).collect::<Vec<_>>();
        choices.sort_by_key(|far|std::cmp::Reverse(far.bytes));
        if let Some(far)=choices.first() {
            let mut ordered=far.fast.clone();ordered.sort_by_key(|&cpu|(costs[cpu],cpu));return Ok((near.fast.clone(),ordered));
        }
    }
    Err("thread affinity: separate fast L3 domains cannot fit mHC, decode, wire and CUDA workers".into())
}
fn snapshot(workers:usize)->Result<Snapshot> {
    let allowed=current_allowed()?;let online=fs::read_to_string("/sys/devices/system/cpu/online").map_err(|error|error.to_string())?;
    let speeds=parse_allowed(&online)?.into_iter().map(|cpu| {
        let speed=fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"))
            .ok().and_then(|value|value.trim().parse::<u64>().ok()).filter(|&value|value>0).ok_or("thread affinity: CPU maximum frequency is absent")?;
        Ok((cpu,speed))
    }).collect::<Result<Vec<_>>>()?;
    let fast=fast_allowed(&allowed,&speeds)?;let mut groups=BTreeMap::<Vec<usize>,Domain>::new();
    for cpu in fast {let (members,bytes)=cache_domain(cpu)?;groups.entry(members.clone()).or_insert_with(||Domain {members,fast:Vec::new(),bytes}).fast.push(cpu);}
    let cuda=cuda_threads()?.len();if cuda==0 {return Err("thread affinity: CUDA worker threads are not visible".into());}
    let (costs,irq_known)=irq_costs();let domains=groups.into_values().collect::<Vec<_>>();let (near,far)=choose_domains(&domains,workers,cuda,&costs)?;
    let host=fs::read_to_string("/proc/sys/kernel/random/boot_id").map_err(|error|error.to_string())?.trim().as_bytes().to_vec();
    if host.len()!=HOST_BYTES || !host.iter().all(|byte|byte.is_ascii_hexdigit() || *byte==b'-') {return Err("thread affinity: invalid kernel identity".into());}
    Ok(Snapshot {record:Record {host,near,far,allowed:allowed.clone(),workers,cuda},allowed,irq_known})
}
fn encode(record:&Record)->Vec<u8> {
    let mut bytes=vec![0;RECORD_BYTES];bytes[0]=2;bytes[1..1+HOST_BYTES].copy_from_slice(&record.host);
    bytes[1+HOST_BYTES..1+HOST_BYTES+8].copy_from_slice(&(record.workers as u64).to_le_bytes());
    bytes[1+HOST_BYTES+8..MASK_AT].copy_from_slice(&(record.cuda as u64).to_le_bytes());
    for &cpu in &record.near {bytes[MASK_AT+cpu/8]|=1<<(cpu%8);}
    for &cpu in &record.allowed {bytes[ALLOWED_AT+cpu/8]|=1<<(cpu%8);}
    bytes[FAR_AT..FAR_AT+CPU_LIMIT*2].fill(255);
    for (index,&cpu) in record.far.iter().enumerate() {bytes[FAR_AT+index*2..FAR_AT+index*2+2].copy_from_slice(&(cpu as u16).to_le_bytes());}bytes
}
fn decode(bytes:&[u8])->Result<Vec<Record>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(RECORD_BYTES) {return Err("thread affinity: invalid placement records".into());}
    bytes.chunks_exact(RECORD_BYTES).map(|row| {
        if row[0]!=2 {return Err("thread affinity: a rank lacks compatible CPU, cache or CUDA thread metadata".into());}
        let workers=usize::try_from(u64::from_le_bytes(row[1+HOST_BYTES..1+HOST_BYTES+8].try_into().unwrap())).map_err(|_|"thread affinity: worker count overflows")?;
        let cuda=usize::try_from(u64::from_le_bytes(row[1+HOST_BYTES+8..MASK_AT].try_into().unwrap())).map_err(|_|"thread affinity: CUDA count overflows")?;
        if workers>CPU_LIMIT || cuda>CPU_LIMIT-2 {return Err("thread affinity: invalid worker counts".into());}
        let near=(0..CPU_LIMIT).filter(|&cpu|row[MASK_AT+cpu/8]>>(cpu%8)&1!=0).collect::<Vec<_>>();let mut far=Vec::new();let mut unique=BTreeSet::new();
        for part in row[FAR_AT..FAR_AT+CPU_LIMIT*2].chunks_exact(2) {
            let cpu=u16::from_le_bytes(part.try_into().unwrap());if cpu==u16::MAX {break;}let cpu=cpu as usize;
            if cpu>=CPU_LIMIT || near.contains(&cpu) || !unique.insert(cpu) {return Err("thread affinity: overlapping or invalid L3 domains".into());}far.push(cpu);
        }
        let allowed=(0..CPU_LIMIT).filter(|&cpu|row[ALLOWED_AT+cpu/8]>>(cpu%8)&1!=0).collect::<Vec<_>>();
        if allowed.is_empty() || near.iter().chain(&far).any(|cpu|!allowed.contains(cpu)) {return Err("thread affinity: role domain exceeds the initial CPU mask".into());}
        Ok(Record {host:row[1..1+HOST_BYTES].to_vec(),near,far,allowed,workers,cuda})
    }).collect()
}
fn allocate(records:&[Record])->Result<Vec<Assignment>> {
    let mut used=BTreeMap::<&[u8],BTreeSet<usize>>::new();let mut assignments=Vec::new();
    for record in records {
        let reserved=used.entry(&record.host).or_default();
        let host=record.near.iter().copied().filter(|cpu|!reserved.contains(cpu)).take(record.workers).collect::<Vec<_>>();
        let far=record.far.iter().copied().filter(|cpu|!reserved.contains(cpu)).collect::<Vec<_>>();
        if host.len()!=record.workers || far.len()<record.cuda+2 {return Err("thread affinity: rank-disjoint L3 reservations do not fit; no partial placement".into());}
        reserved.extend(&host);reserved.extend(&far);let mut decode=far;decode.sort_unstable();assignments.push(Assignment {host,decode,background:Vec::new()});
    }
    for (record,assignment) in records.iter().zip(&mut assignments) {
        let reserved=used.entry(&record.host).or_default();let mut background=record.allowed.iter().copied().filter(|cpu|!reserved.contains(cpu)).collect::<Vec<_>>();
        reserved.extend(&background);background.extend(&assignment.decode);background.sort_unstable();assignment.background=background;
    }Ok(assignments)
}
impl Placement {
    pub fn new(ring:&mut Ring,workers:usize)->Result<Self> {
        let local=snapshot(workers);if let Err(error)=&local {eprintln!("TP AFFINITY rank={} unpinned: {error}",ring.rank());}
        let mut records=vec![0;ring.world()*RECORD_BYTES];if let Ok(local)=&local {records[ring.rank()*RECORD_BYTES..(ring.rank()+1)*RECORD_BYTES].copy_from_slice(&encode(&local.record));}
        ring.all_gather(&mut records,RECORD_BYTES)?;
        let assignment=match decode(&records).and_then(|records|allocate(&records)) {Ok(mut assignments)=>Some(assignments.swap_remove(ring.rank())),Err(error)=>{eprintln!("TP AFFINITY rank={} unpinned: {error}",ring.rank());None}};
        if let Ok(local)=&local && !local.irq_known {eprintln!("TP AFFINITY rank={} NIC interrupt counts unavailable; far domain still uses its complete fast-core mask",ring.rank());}
        let allowed=local.map(|local|local.allowed).unwrap_or_default();
        if let Some(assignment)=&assignment {eprintln!("TP AFFINITY rank={} mhc={:?} decode_mask={:?} wire_cuda_mask={:?}",ring.rank(),assignment.host,assignment.decode,assignment.background);}
        Ok(Self {rank:ring.rank(),allowed,assignment,pinned:RefCell::new(BTreeSet::new())})
    }
    pub fn workers(&mut self)->WorkerMask {
        let previous=match &self.assignment {
            Some(assignment) if !assignment.host.is_empty()=>match runtime::pin_cpus(&assignment.host) {Ok(())=>Some(self.allowed.clone()),Err(error)=>{eprintln!("TP AFFINITY rank={} unpinned: {error}",self.rank);self.assignment=None;None}},
            _=>None
        };WorkerMask {previous,_thread:PhantomData}
    }
    pub fn wire_cpus(&self)->Option<Vec<usize>> {self.assignment.as_ref().map(|assignment|assignment.background.clone())}
    pub fn refresh_cuda(&self) {
        let Some(assignment)=&self.assignment else {return;};
        let threads=match cuda_threads() {Ok(threads)=>threads,Err(error)=>{eprintln!("TP AFFINITY rank={} CUDA placement unavailable: {error}",self.rank);return;}};
        let mut pinned=self.pinned.borrow_mut();pinned.retain(|tid|threads.iter().any(|thread|thread.tid==*tid));
        for thread in threads {
            if pinned.contains(&thread.tid) {continue;}
            match runtime::pin_thread_cpus(thread.tid,&assignment.background) {Ok(())=>{pinned.insert(thread.tid);eprintln!("TP AFFINITY rank={} cuda={} tid={} mask={:?}",self.rank,thread.name,thread.tid,assignment.background);},Err(error)=>eprintln!("TP AFFINITY rank={} CUDA thread unpinned: {error}",self.rank)}
        }
    }
    pub fn pin_decode(&self) {self.refresh_cuda();if let Some(assignment)=&self.assignment && let Err(error)=runtime::pin_cpus(&assignment.decode) {eprintln!("TP AFFINITY rank={} decode unpinned: {error}",self.rank);}}
}
impl Drop for WorkerMask {
    fn drop(&mut self) {if let Some(previous)=&self.previous && let Err(error)=runtime::pin_cpus(previous) {eprintln!("TP AFFINITY: cannot restore main CPU mask: {error}");}}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(host:u8)->Record {Record {host:vec![host;HOST_BYTES],near:vec![5,6,7,8,9],far:vec![15,16,17,18,19],allowed:(0..20).collect(),workers:3,cuda:2}}
    #[test] fn parses_allowed_cpu_ranges() {assert_eq!(parse_allowed("0-4,8,15-19").unwrap(),vec![0,1,2,3,4,8,15,16,17,18,19]);for invalid in ["","3-1","0-1024","-1","1,,2"] {assert!(parse_allowed(invalid).is_err());}}
    #[test] fn slower_only_masks_are_not_fast() {let speeds=[(0,2808000),(1,2808000),(5,3900000),(6,3900000)];assert!(fast_allowed(&[0,1],&speeds).unwrap().is_empty());assert_eq!(fast_allowed(&[0,6],&speeds).unwrap(),vec![6]);}
    #[test] fn cache_sizes_come_from_metadata() {assert_eq!(cache_size("8192K\n"),Some(8388608));assert_eq!(cache_size("16M"),Some(16777216));assert!(cache_size("unknown").is_none());}
    #[test] fn physical_hosts_isolate_all_roles_from_mhc() {
        for assignment in allocate(&[record(b'a'),record(b'b')]).unwrap() {
            assert_eq!(assignment.host,vec![5,6,7]);assert_eq!(assignment.decode,vec![15,16,17,18,19]);assert_eq!(assignment.background,vec![0,1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,19]);
            assert!(assignment.host.iter().all(|cpu|!assignment.decode.contains(cpu) && !assignment.background.contains(cpu)));
        }
    }
    #[test] fn colocated_overcommit_falls_back_instead_of_sharing_cores() {assert!(allocate(&[record(b'a'),record(b'a')]).is_err());}
    #[test] fn numbering_does_not_define_the_domains() {
        let record=Record {host:vec![b'a';HOST_BYTES],near:vec![21,23,25],far:vec![2,4,6,8],allowed:vec![2,4,6,8,21,23,25],workers:3,cuda:2};
        let assignment=allocate(&[record]).unwrap().remove(0);assert_eq!(assignment.host,vec![21,23,25]);assert_eq!(assignment.decode,vec![2,4,6,8]);assert_eq!(assignment.background,vec![2,4,6,8]);
    }
    #[test] fn busy_irq_cores_lose_priority_without_irq_changes() {
        let domains=[Domain {members:(0..10).collect(),fast:(5..10).collect(),bytes:8<<20},Domain {members:(10..20).collect(),fast:(15..20).collect(),bytes:16<<20}];
        let mut costs=vec![0;CPU_LIMIT];costs[15]=1000;let (_,far)=choose_domains(&domains,3,2,&costs).unwrap();assert_eq!(far,vec![16,17,18,19,15]);
    }
    #[test] fn missing_separate_domain_is_a_fallback() {let domains=[Domain {members:(0..10).collect(),fast:(5..10).collect(),bytes:8<<20}];assert!(choose_domains(&domains,3,2,&vec![0;CPU_LIMIT]).is_err());}
    #[test] fn partial_shared_cache_overlap_is_not_isolation() {let domains=[Domain {members:(0..10).collect(),fast:(5..10).collect(),bytes:8<<20},Domain {members:(5..20).collect(),fast:(15..20).collect(),bytes:16<<20}];assert!(choose_domains(&domains,3,2,&vec![0;CPU_LIMIT]).is_err());}
    #[test] fn record_round_trip_keeps_irq_priority() {
        let mut record=record(b'a');record.far=vec![17,18,19,15,16];let decoded=decode(&encode(&record)).unwrap().remove(0);
        assert_eq!(decoded.host,record.host);assert_eq!(decoded.near,record.near);assert_eq!(decoded.far,record.far);assert_eq!(decoded.allowed,record.allowed);assert_eq!(decoded.workers,3);assert_eq!(decoded.cuda,2);
        assert!(decode(&vec![0;RECORD_BYTES]).is_err());assert!(decode(&[0]).is_err());
    }
    #[test] fn invalid_domain_overlap_is_refused() {let mut record=record(b'a');record.far[0]=5;assert!(decode(&encode(&record)).is_err());}
    #[test] fn masks_do_not_expand_a_restricted_initial_set() {let mut record=record(b'a');record.allowed.retain(|cpu|[5,6,7,15,16,17,18,19].contains(cpu));record.near=vec![5,6,7];let assignment=allocate(&decode(&encode(&record)).unwrap()).unwrap().remove(0);assert_eq!(assignment.background,assignment.decode);record.allowed.retain(|&cpu|cpu!=19);assert!(decode(&encode(&record)).is_err());}
    #[test] fn colocated_background_masks_partition_spare_cpus() {
        let first=Record {host:vec![b'a';HOST_BYTES],near:vec![0,1,2],far:vec![4,5,6,7],allowed:(0..16).collect(),workers:1,cuda:1};
        let second=Record {host:first.host.clone(),near:vec![8,9,10],far:vec![12,13,14,15],allowed:first.allowed.clone(),workers:1,cuda:1};let assignments=allocate(&[first,second]).unwrap();
        for (index,assignment) in assignments.iter().enumerate() {let other=&assignments[1-index];assert!(assignment.background.iter().all(|cpu|!assignment.host.contains(cpu) && !other.host.contains(cpu) && !other.background.contains(cpu)));assert!(assignment.decode.iter().all(|cpu|assignment.background.contains(cpu)));}
    }
    #[test] fn deferred_pin_is_thread_local() {let guard=runtime::defer_cpu_pin();assert!(runtime::cpu_pin_deferred());std::thread::spawn(||assert!(!runtime::cpu_pin_deferred())).join().unwrap();drop(guard);assert!(!runtime::cpu_pin_deferred());}
    #[cfg(target_os="linux")]
    #[test] fn early_tp_cpu_does_not_narrow_managed_startup() {let before=current_allowed().unwrap();let _guard=runtime::defer_cpu_pin();runtime::pin().unwrap();assert_eq!(current_allowed().unwrap(),before);}
    #[cfg(target_os="linux")]
    #[test] fn owned_thread_affinity_targets_only_that_thread() {
        use std::sync::{Arc,Barrier,mpsc};
        let initial=current_allowed().unwrap();let cpu=initial[0];let ready=Arc::new(Barrier::new(2));let worker_ready=ready.clone();let (send,receive)=mpsc::channel();
        let worker=std::thread::Builder::new().name("cuda-pin-test".into()).spawn(move|| {let tid=fs::read_link("/proc/thread-self").unwrap().file_name().unwrap().to_string_lossy().parse::<usize>().unwrap();send.send(tid).unwrap();worker_ready.wait();current_allowed().unwrap()}).unwrap();
        let tid=receive.recv().unwrap();runtime::pin_thread_cpus(tid,&[cpu]).unwrap();assert_eq!(current_allowed().unwrap(),initial);ready.wait();assert_eq!(worker.join().unwrap(),vec![cpu]);
        assert!(runtime::pin_thread_cpus(usize::MAX,&[cpu]).is_err());
    }
    #[cfg(target_os="linux")]
    #[test]
    #[ignore="requires an authorized CUDA device"]
    fn cuda_context_workers_use_off_mhc_masks() {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),64<<20,|backend| {
            let _mailbox=backend.tp_mailbox(4096)?;let snapshot=snapshot(3)?;let assignment=allocate(&[snapshot.record])?.remove(0);
            let initial=snapshot.allowed;let restore=WorkerMask {previous:Some(initial.clone()),_thread:PhantomData};
            let mut placement=Placement {rank:0,allowed:initial.clone(),assignment:Some(assignment.clone()),pinned:RefCell::new(BTreeSet::new())};
            let workers=placement.workers();assert_eq!(current_allowed()?,assignment.host);
            let wire=assignment.background.clone();let worker=std::thread::spawn(move|| {runtime::pin_cpus(&wire)?;current_allowed()});
            assert_eq!(worker.join().map_err(|_|"wire affinity test panicked")??,assignment.background);drop(workers);assert_eq!(current_allowed()?,initial);
            placement.pin_decode();assert_eq!(current_allowed()?,assignment.decode);
            let threads=cuda_threads()?;if threads.is_empty() {return Err("CUDA worker test found no threads".into());}assert_eq!(placement.pinned.borrow().len(),threads.len());
            for thread in threads {
                let status=fs::read_to_string(format!("/proc/self/task/{}/status",thread.tid)).map_err(|error|error.to_string())?;
                let allowed=parse_allowed(status.lines().find_map(|line|line.strip_prefix("Cpus_allowed_list:")).ok_or("CUDA worker mask absent")?)?;
                assert_eq!(allowed,assignment.background);assert!(allowed.iter().all(|cpu|!assignment.host.contains(cpu)));assert!(allowed.len()>1);
            }drop(restore);Ok(())
        }).unwrap();
    }
}
