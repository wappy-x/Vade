//! Existing-trace exchange census and host/GPU timeline bounds. No device access.
//! Usage: xchg-timeline TRACE.sqlite RANK_LOG OUTPUT.tsv
use std::{env,fs,process::Command};
fn query(path:&str,sql:&str)->String {let out=Command::new("sqlite3").args(["-readonly","-separator","\t",path,sql]).output().expect("sqlite3");assert!(out.status.success(),"{}",String::from_utf8_lossy(&out.stderr));String::from_utf8(out.stdout).unwrap()}
fn number(s:&str,key:&str)->u64 {s.split_whitespace().find_map(|p|p.strip_prefix(key)).expect(key).parse().expect("integer")}
fn main() {
    let a=env::args().collect::<Vec<_>>();assert_eq!(a.len(),4,"TRACE LOG OUT");
    let log=fs::read_to_string(&a[2]).unwrap();let start=log.find("prompt_tokens=8192 queue_ms=").expect("read request");
    let mut rows=Vec::new();for l in log[start..].lines() {if l.starts_with("XTRACE rank=") {rows.push((number(l,"bytes="),number(l,"link_ns="),l.split_once("kind=").unwrap().1.to_owned()));}if rows.len()>0 && l.starts_with("SERVED STEP") && l.contains("phase=decode") {break;}}
    assert!(!rows.is_empty());let copies=query(&a[1],"SELECT m.start,m.end,m.bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId JOIN StringIds s ON s.id=r.nameId WHERE m.copyKind=2 AND s.value='cuMemcpyDtoDAsync_v2' ORDER BY m.start");
    let cp=copies.lines().map(|l|l.split('\t').map(|f|f.parse::<u64>().unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>();
    let sync=query(&a[1],"SELECT r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value='cuStreamSynchronize' ORDER BY r.end").lines().map(|l|l.parse::<u64>().unwrap()).collect::<Vec<_>>();
    let kernels=query(&a[1],"SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").lines().map(|l|l.parse::<u64>().unwrap()).collect::<Vec<_>>();
    let mut at=0;let mut receipt=String::from("index\tbytes_rank\tkind\tpack_start_ns\tpack_end_ns\thost_observes_ns\tlink_end_estimate_ns\tconsumer_start_ns\tpack_us\tobserve_delay_us\tlink_us\tconsumer_delay_us\n");
    let mut kinds=std::collections::BTreeMap::<String,(usize,u64,u64)>::new();
    for (i,(bytes,link,kind)) in rows.iter().enumerate() {
        let begin=at;let mut total=0;while total<*bytes {let c=cp.get(at).expect("pack missing");total+=c[2];at+=1;}assert_eq!(total,*bytes,"pack extent at exchange {i}");let(s,e)=(cp[begin][0],cp[at-1][1]);
        let observed=sync[sync.partition_point(|&end|end<e)];assert!(observed-e<2_000_000,"unmatched synchronization");
        let end=observed+link;let consumer=kernels[kernels.partition_point(|&start|start<end)];
        receipt.push_str(&format!("{i}\t{bytes}\t{kind}\t{s}\t{e}\t{observed}\t{end}\t{consumer}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\n",(e-s) as f64/1000.,(observed-e) as f64/1000.,*link as f64/1000.,(consumer-end) as f64/1000.));
        // Remove layer/expert indices only for the census; source names remain in the receipt.
        let k=kind.split('.').map(|p|if p.parse::<usize>().is_ok(){"L"}else{p}).collect::<Vec<_>>().join(".");let k=kinds.entry(k).or_default();k.0+=1;k.1+=bytes;k.2+=link;
    }
    fs::write(&a[3],receipt).unwrap();println!("MEASURED exchanges={} bytes_per_rank={} mapped_pack_copies={at}",rows.len(),rows.iter().map(|r|r.0).sum::<u64>());
    for (kind,(count,bytes,ns)) in kinds {println!("{count}\t{bytes}\t{:.6}\t{kind}",ns as f64/1e9);}
    println!("LIMIT: link_end_estimate = synchronize return + XTRACE link duration; timer starts after return and profiling hooks. It is not a common-clock post or NIC completion timestamp. Post and peer arrival lie inside that host link interval, with unmeasured hook overhead. No per-message NIC timestamp exists in this trace.");
}
