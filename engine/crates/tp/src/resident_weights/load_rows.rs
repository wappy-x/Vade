use crate::Result;
use std::borrow::Cow;

pub(super) fn batch_end(bytes:&[u64],start:usize,workers:usize,budget:u64)->usize {
    let mut end=start;let mut total=0u64;
    while end<bytes.len() && end-start<workers {
        let next=total.saturating_add(bytes[end]);
        if end>start && next>budget {break;}
        total=next;end+=1;
    }
    end
}

pub(super) fn padded_bytes(bytes:&[u8])->Cow<'_,[u8]> {
    if bytes.len().is_multiple_of(4) {return Cow::Borrowed(bytes);}
    let mut padded=bytes.to_vec();padded.resize(bytes.len().div_ceil(4)*4,0);Cow::Owned(padded)
}

pub(super) fn ordered<T:Send>(names:&[String],read:impl Fn(&str)->Result<T>+Sync)->Result<Vec<T>> {
    std::thread::scope(|scope| {
        let mut workers=Vec::new();let mut failure=None;
        for name in names {
            let read=&read;
            match std::thread::Builder::new().spawn_scoped(scope,move ||read(name)) {
                Ok(worker)=>workers.push(worker),Err(error)=>{failure=Some(format!("resident loader worker start failed: {error}"));break;},
            }
        }
        let mut output=Vec::new();
        for worker in workers {
            match worker.join() {
                Ok(Ok(value))=>output.push(value),
                Ok(Err(error))=>{if failure.is_none() {failure=Some(error);}},
                Err(_)=>{if failure.is_none() {failure=Some("resident loader worker panicked".into());}},
            }
        }
        if let Some(error)=failure {Err(error)} else {Ok(output)}
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize,Ordering};
    #[test]
    fn batches_bound_bytes_and_workers_but_allow_one_large_matrix() {
        assert_eq!(batch_end(&[3,3,3,3],0,8,8),2);
        assert_eq!(batch_end(&[3,3,3,3],2,8,8),4);
        assert_eq!(batch_end(&[3,3,3,3],0,1,8),1);
        assert_eq!(batch_end(&[100,3,3],0,8,8),1);
        assert_eq!(batch_end(&[u64::MAX,1],0,8,8),1);
        assert_eq!(batch_end(&[],0,8,8),0);
    }
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
    fn parallel_reads_return_in_input_order_and_join_on_failure() {
        let names=(0..8).map(|index|index.to_string()).collect::<Vec<_>>();let finished=AtomicUsize::new(0);
        assert_eq!(ordered(&names,|name|Ok(name.to_owned())).unwrap(),names);
        let result=ordered(&names,|name| {finished.fetch_add(1,Ordering::SeqCst);if name=="0" {Err("read failed".into())} else {Ok(name.to_owned())}});
        assert_eq!(result.unwrap_err(),"read failed");assert_eq!(finished.load(Ordering::SeqCst),8);
    }
    #[test]
    fn worker_panics_are_errors_after_all_workers_join() {
        let names=(0..8).map(|index|index.to_string()).collect::<Vec<_>>();let finished=AtomicUsize::new(0);
        let result=ordered(&names,|name| {finished.fetch_add(1,Ordering::SeqCst);assert_ne!(name,"0");Ok(name.to_owned())});
        assert_eq!(result.unwrap_err(),"resident loader worker panicked");assert_eq!(finished.load(Ordering::SeqCst),8);
    }
}
