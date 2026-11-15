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

/// Copy threads of a weight load. They fill the page-locked slabs, so the upload thread copies no plane.
#[cfg(target_os="linux")]
pub(super) const STAGERS:usize=4;
/// Page-locked slabs of a weight load: one holds the two planes of one column-tile matrix from its read to its device copy.
#[cfg(target_os="linux")]
pub(super) const SLABS:usize=32;
/// Of the slabs, those left to the device: a slab is written again this many matrices after its device copy was queued.
#[cfg(target_os="linux")]
pub(super) const DEVICE_SLABS:usize=8;
/// A matrix with more plane bytes than this keeps the upload through the one shared staging buffer.
#[cfg(target_os="linux")]
pub(super) const SLAB_BYTES_MAX:usize=8<<20;

/// The two planes of a matrix in one slab: the codes, then the scales, each with the zero bytes that `padded_bytes`
/// adds. Returns the two padded byte counts. The slab bytes after them keep their old values and are never read.
pub(super) fn pack_planes(slab:&mut [u8],codes:&[u8],scales:&[u8])->Result<(usize,usize)> {
    let (code_bytes,scale_bytes)=(codes.len().div_ceil(4)*4,scales.len().div_ceil(4)*4);
    let end=code_bytes.checked_add(scale_bytes).filter(|&end|end<=slab.len()).ok_or("resident planes exceed their slab")?;
    slab[..codes.len()].copy_from_slice(codes);slab[codes.len()..code_bytes].fill(0);
    slab[code_bytes..code_bytes+scales.len()].copy_from_slice(scales);slab[code_bytes+scales.len()..end].fill(0);
    Ok((code_bytes,scale_bytes))
}

/// How the error of a delayed upload starts. A column-tile matrix that took a slab goes to the device later: while
/// a later matrix loads, or when the load ends. Its error names the matrix that failed.
const DELAYED:&str="delayed upload of ";
/// The error of matrix `name`, whose upload ran late.
pub(super) fn delayed(name:&str,error:&str)->String {format!("{DELAYED}{name}: {error}")}
/// The error met while matrix `name` loads. The matrix's own error gets its name. A delayed upload's error keeps the
/// name it has: the matrix that loads did not fail.
pub(super) fn at(name:&str,error:String)->String {if error.starts_with(DELAYED) {error} else {format!("{name}: {error}")}}

/// Which slab the next matrix takes, as numbers. `staged` matrices hold the newest slabs of the ring. A slab is
/// taken again only after its matrix was queued for the device and `device` more matrices were queued after it.
pub(super) struct SlabRing {slabs:usize,device:usize,next:usize,staged:usize}
impl SlabRing {
    pub(super) fn new(slabs:usize,device:usize)->Result<Self> {
        if device==0 || device>=slabs {return Err("resident slab ring needs a staged slab and a device slab".into());}
        Ok(Self {slabs,device,next:0,staged:0})
    }
    #[cfg(test)]
    pub(super) fn staged(&self)->usize {self.staged}
    /// The oldest staged matrix must be queued for the device before `take`.
    pub(super) fn full(&self)->bool {self.staged+self.device>=self.slabs}
    /// The slab of the next matrix; `full` was false.
    pub(super) fn take(&mut self)->Result<usize> {
        if self.full() {return Err("resident slab ring is full".into());}
        let slab=self.next;self.next=(self.next+1)%self.slabs;self.staged+=1;Ok(slab)
    }
    /// The oldest staged matrix was queued for the device.
    pub(super) fn queued(&mut self)->Result<()> {self.staged=self.staged.checked_sub(1).ok_or("resident slab ring holds no staged matrix")?;Ok(())}
    /// New slabs take the place of all: no matrix is staged.
    pub(super) fn restart(&mut self)->Result<()> {if self.staged!=0 {return Err("resident slab ring restarts with a staged matrix".into());}self.next=0;Ok(())}
}

/// Runs `run` with `workers` threads behind the calling thread's host preparation queue (`host_prep`): a task
/// that `run` submits there runs on one of them. Every submitted task has run when this returns, also after
/// an error in `run`.
pub(super) fn with_stagers<R>(workers:usize,run:impl FnOnce()->R)->Result<R> {
    use spark_gpu_probe::host_prep::{Task,with_dispatcher};
    let (send,tasks)=std::sync::mpsc::channel::<Task>();let tasks=Mutex::new(tasks);
    std::thread::scope(|scope| {
        // The sender leaves with this block, also on a failed start: the threads end when no task is left.
        let send=send;
        for _ in 0..workers.max(1) {
            std::thread::Builder::new().name("weight-stage".into()).spawn_scoped(scope,|| loop {
                // The lock covers the wait for one task, not its run.
                let task=tasks.lock().unwrap_or_else(PoisonError::into_inner).recv();
                match task {Ok(task)=>task(),Err(_)=>return}
            }).map_err(|error|format!("resident loader stager start failed: {error}"))?;
        }
        Ok(with_dispatcher(move |task|send.send(task).map_err(|error|error.0),run))
    })
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
    fn packed_planes_are_the_padded_planes_of_the_one_buffer_upload() {
        for codes in [0usize,1,3,4,5,64,127] {for scales in [0usize,1,2,4,9,16] {
            let code=(0..codes).map(|index|(index*29+1) as u8).collect::<Vec<_>>();let scale=(0..scales).map(|index|(index*31+7) as u8).collect::<Vec<_>>();
            let mut slab=vec![0xA5u8;codes.div_ceil(4)*4+scales.div_ceil(4)*4+8];
            let (code_bytes,scale_bytes)=pack_planes(&mut slab,&code,&scale).unwrap();
            // The device copies read exactly these two ranges: the bytes `padded_bytes` gives the one-buffer upload.
            assert_eq!(&slab[..code_bytes],&*padded_bytes(&code));assert_eq!(&slab[code_bytes..code_bytes+scale_bytes],&*padded_bytes(&scale));
            assert!(slab[code_bytes+scale_bytes..].iter().all(|&byte|byte==0xA5));
            if code_bytes+scale_bytes>0 {assert!(pack_planes(&mut slab[..code_bytes+scale_bytes-1],&code,&scale).is_err());}
        }}
    }
    #[test]
    fn a_delayed_upload_error_names_its_own_matrix_not_the_one_that_loads() {
        // "early" took a slab. Its device copy fails while "late" loads: the load's wrapper sees the error.
        let failed:Result<()>=Err(delayed("early","device copy failed"));
        let seen=failed.map_err(|error|at("late",error)).unwrap_err();
        assert_eq!(seen,"delayed upload of early: device copy failed");assert!(!seen.contains("late"));
        // The same error at the end of the load, where no matrix loads, already names its matrix.
        assert_eq!(delayed("early","repack failed"),"delayed upload of early: repack failed");
        // The error of the matrix that loads keeps the wrapper's name, also when it only mentions a delay.
        assert_eq!(at("late","read failed".into()),"late: read failed");
        assert_eq!(at("late","the delayed upload of nothing".into()),"late: the delayed upload of nothing");
    }
    #[test]
    fn a_slab_is_taken_again_only_after_its_matrix_was_queued_and_the_device_slabs_passed() {
        assert!(SlabRing::new(4,0).is_err());assert!(SlabRing::new(4,4).is_err());
        for (slabs,device) in [(2usize,1usize),(8,3),(32,8),(5,4)] {
            let mut ring=SlabRing::new(slabs,device).unwrap();
            // For each slab: the matrix that holds it (staged), or the count of matrices queued when its own was queued.
            let mut holder=vec![None::<usize>;slabs];let mut queued_at=vec![None::<usize>;slabs];let mut staged=std::collections::VecDeque::new();let mut queued=0usize;
            assert!(ring.queued().is_err());
            for matrix in 0..slabs*7+3 {
                while ring.full() {let (old,slab):(usize,usize)=staged.pop_front().unwrap();assert_eq!(holder[slab],Some(old));holder[slab]=None;queued+=1;queued_at[slab]=Some(queued);ring.queued().unwrap();}
                assert!(ring.staged()+device<slabs);
                let slab=ring.take().unwrap();
                assert_eq!(holder[slab],None,"a staged matrix still holds slab {slab}");
                // The device had at least `device` later matrices of time to read the slab; the slab's event covers the rest.
                // The first time around a slab has no earlier matrix.
                if let Some(at)=queued_at[slab] {assert!(queued+1-at>=device,"slab {slab} of matrix {matrix} was queued {} matrices ago",queued+1-at);}
                holder[slab]=Some(matrix);staged.push_back((matrix,slab));
            }
            assert!(ring.restart().is_err());
            while let Some((_,slab))=staged.pop_front() {holder[slab]=None;ring.queued().unwrap();}
            ring.restart().unwrap();assert_eq!(ring.take().unwrap(),0);
        }
    }
    #[test]
    fn stagers_run_every_submitted_task_before_the_return() {
        use std::sync::Arc;
        for workers in [0usize,1,4] {
            let ran=Arc::new(AtomicUsize::new(0));let elsewhere=Arc::new(AtomicUsize::new(0));let caller=std::thread::current().id();
            let value=with_stagers(workers,|| {
                for _ in 0..257 {let (ran,elsewhere)=(ran.clone(),elsewhere.clone());
                    spark_gpu_probe::host_prep::submit(Box::new(move || {
                        if std::thread::current().id()!=caller {elsewhere.fetch_add(1,Ordering::SeqCst);}
                        std::thread::yield_now();ran.fetch_add(1,Ordering::SeqCst);
                    }));
                }7
            }).unwrap();
            assert_eq!(value,7);assert_eq!(ran.load(Ordering::SeqCst),257);assert_eq!(elsewhere.load(Ordering::SeqCst),257);
        }
        // Outside the call a task runs where it is submitted, as before.
        let ran=Arc::new(AtomicUsize::new(0));let inside=ran.clone();
        spark_gpu_probe::host_prep::submit(Box::new(move || {inside.fetch_add(1,Ordering::SeqCst);}));assert_eq!(ran.load(Ordering::SeqCst),1);
    }
    #[test]
    fn stagers_drain_after_consumer_error_and_unwind() {
        use std::sync::Arc;
        for panic in [false,true] {
            let ran=Arc::new(AtomicUsize::new(0));let inside=ran.clone();
            let result=std::panic::catch_unwind(move ||with_stagers(2,||->Result<()> {
                for _ in 0..64 {let done=inside.clone();spark_gpu_probe::host_prep::submit(Box::new(move || {
                    std::thread::yield_now();done.fetch_add(1,Ordering::SeqCst);
                }));}
                assert!(!panic,"consumer stopped");Err("consumer error".into())
            }));
            if panic {assert!(result.is_err());} else {assert_eq!(result.unwrap().unwrap().unwrap_err(),"consumer error");}
            assert_eq!(ran.load(Ordering::SeqCst),64);
        }
    }
    #[test]
    fn column_tile_planes_need_no_arena_padding() {
        use spark_gpu_probe::expert_ops::{NV_TILE_ROWS,nv_tile_layout_admits};
        // The layout rule requires rows in whole tiles and columns in groups of 64.
        // Both planes must therefore be multiples of vector_arena's 64-byte alignment.
        for tiles in [1usize,2,3,19] {for cols in [64usize,128,256,1024,4096] {
            let rows=tiles*NV_TILE_ROWS as usize;assert!(nv_tile_layout_admits(rows,cols));
            for bytes in [rows*(cols/2),rows*(cols/16)] {assert_eq!(bytes.div_ceil(64)*64,bytes);}
        }}
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
