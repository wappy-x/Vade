use crate::Result;
use std::{borrow::Cow,collections::BTreeMap,sync::{Condvar,Mutex,PoisonError}};

/// Reader threads of a weight load: at most this many, and at most the CPUs.
#[cfg(target_os="linux")]
const READERS:usize=12;
/// Bytes read and not yet uploaded, at most; a larger matrix is read when nothing else is held.
#[cfg(target_os="linux")]
pub(super) const HELD_BYTES:u64=2<<30;
#[cfg(target_os="linux")]
pub(super) fn readers()->usize {std::thread::available_parallelism().map_or(1,usize::from).min(READERS)}

pub(super) fn padded_bytes(bytes:&[u8])->Cow<'_,[u8]> {
    if bytes.len().is_multiple_of(4) {return Cow::Borrowed(bytes);}
    let mut padded=bytes.to_vec();padded.resize(bytes.len().div_ceil(4)*4,0);Cow::Owned(padded)
}

/// What the readers share: the next index to read, the bytes read and not yet consumed, the finished reads.
struct Queue<T> {next:usize,held:u64,ready:BTreeMap<usize,Result<T>>,stop:bool}
/// Stops the readers when the consumer leaves, also by an error or a panic, so the scope can join them.
struct Stop<'q,T> {queue:&'q Mutex<Queue<T>>,changed:&'q Condvar}
impl<T> Drop for Stop<'_,T> {
    fn drop(&mut self) {self.queue.lock().unwrap_or_else(PoisonError::into_inner).stop=true;self.changed.notify_all();}
}

/// Reads run ahead of one consumer. Up to `workers` threads read items in index order, and the calling thread
/// consumes every item in index order, so the consumer's work (device uploads) overlaps the next reads.
/// A read starts only while the bytes read and not yet consumed stay within `budget`; with nothing held, one
/// read may exceed it. After a failed read or consume no new read starts, and the workers are joined before
/// the first error returns. A panicking read is an error.
pub(super) fn streamed<T:Send>(sizes:&[u64],workers:usize,budget:u64,read:impl Fn(usize)->Result<T>+Sync,mut consume:impl FnMut(usize,T)->Result<()>)->Result<()> {
    let queue=Mutex::new(Queue {next:0,held:0,ready:BTreeMap::new(),stop:false});let changed=Condvar::new();
    let lock=||queue.lock().unwrap_or_else(PoisonError::into_inner);
    std::thread::scope(|scope| {
        let stop=Stop {queue:&queue,changed:&changed};
        let worker=|| loop {
            let index={let mut q=lock();
                loop {
                    if q.stop || q.next>=sizes.len() {return;}
                    if q.held==0 || q.held.saturating_add(sizes[q.next])<=budget {break;}
                    q=changed.wait(q).unwrap_or_else(PoisonError::into_inner);
                }
                let index=q.next;q.next+=1;q.held=q.held.saturating_add(sizes[index]);index
            };
            let value=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||read(index))).unwrap_or_else(|_|Err("resident loader worker panicked".into()));
            lock().ready.insert(index,value);changed.notify_all();
        };
        let mut threads=Vec::new();
        for _ in 0..workers.clamp(1,sizes.len().max(1)) {
            match std::thread::Builder::new().name("weight-load".into()).spawn_scoped(scope,worker) {
                Ok(thread)=>threads.push(thread),
                Err(error)=>{drop(stop);for thread in threads {let _=thread.join();}return Err(format!("resident loader worker start failed: {error}"));},
            }
        }
        let mut result=Ok(());
        for (index,&bytes) in sizes.iter().enumerate() {
            let value={let mut q=lock();loop {if let Some(value)=q.ready.remove(&index) {break value;}q=changed.wait(q).unwrap_or_else(PoisonError::into_inner);}};
            let outcome=value.and_then(|value|consume(index,value));
            {let mut q=lock();q.held=q.held.saturating_sub(bytes);if outcome.is_err() {q.stop=true;}}changed.notify_all();
            if let Err(error)=outcome {result=Err(error);break;}
        }
        drop(stop);
        for thread in threads {if thread.join().is_err() && result.is_ok() {result=Err("resident loader worker panicked".into());}}
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize,Ordering};
    #[test]
    fn packed_upload_initializes_only_padding_without_changing_payload() {
        for count in 0..17 {
            let input=(0..count).map(|index|(index*23) as u8).collect::<Vec<_>>();let padded=padded_bytes(&input);
            assert_eq!(&padded[..count],input);assert!(padded[count..].iter().all(|&byte|byte==0));
            assert_eq!(padded.len(),count.div_ceil(4)*4);
            assert_eq!(matches!(padded,Cow::Borrowed(_)),count%4==0);
        }
    }
    #[test]
    fn streamed_reads_are_consumed_in_input_order() {
        for workers in [1,2,3,8,64] {for budget in [0,1,5,u64::MAX] {
            let sizes=(0..37).map(|index|(index%5) as u64).collect::<Vec<_>>();let mut seen=Vec::new();
            streamed(&sizes,workers,budget,|index|{std::thread::yield_now();Ok(index*7)},|index,value|{assert_eq!(value,index*7);seen.push(index);Ok(())}).unwrap();
            assert_eq!(seen,(0..37).collect::<Vec<_>>());
        }}
        streamed(&[],4,8,|_|->Result<()> {unreachable!()},|_,_|unreachable!()).unwrap();
    }
    #[test]
    fn streamed_reads_hold_at_most_the_budget_or_one_item() {
        let sizes=[3,3,3,9,1,1,1,1,3,3];let held=AtomicUsize::new(0);let peak=AtomicUsize::new(0);
        streamed(&sizes,8,6,|index| {
            let now=held.fetch_add(sizes[index] as usize,Ordering::SeqCst)+sizes[index] as usize;peak.fetch_max(now,Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(2));Ok(index)
        },|index,_|{held.fetch_sub(sizes[index] as usize,Ordering::SeqCst);Ok(())}).unwrap();
        // The 9-byte item exceeds the budget and is read alone; every other moment stays within 6 bytes.
        assert_eq!(peak.load(Ordering::SeqCst),9);
    }
    #[test]
    fn streamed_errors_stop_new_reads_and_join_every_worker() {
        let started=AtomicUsize::new(0);let sizes=vec![1u64;64];
        let result=streamed(&sizes,4,2,|index|{started.fetch_add(1,Ordering::SeqCst);if index==3 {Err("read failed".into())} else {Ok(index)}},|_,_|Ok(()));
        assert_eq!(result.unwrap_err(),"read failed");assert!(started.load(Ordering::SeqCst)<64);
        let result=streamed(&sizes,4,2,Ok,|index,_|if index==5 {Err("upload failed".into())} else {Ok(())});
        assert_eq!(result.unwrap_err(),"upload failed");
    }
    #[test]
    fn worker_panics_are_errors_after_all_workers_join() {
        let names=(0..8).collect::<Vec<usize>>();let finished=AtomicUsize::new(0);let sizes=vec![1u64;names.len()];
        let result=streamed(&sizes,8,u64::MAX,|index|{finished.fetch_add(1,Ordering::SeqCst);assert_ne!(index,0);Ok(index)},|_,_|Ok(()));
        assert_eq!(result.unwrap_err(),"resident loader worker panicked");
    }
}
