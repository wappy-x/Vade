use super::*;
use crate::{mhc_params::Shape,norm_boundary::{NormShape,WeightType},tp_mailbox_host::ERROR};
use std::sync::atomic::AtomicU32;

fn boundary(streams:u32,hidden:u32)->Arc<HostBoundary> {
    let shape=Shape {rows:1,streams,hidden,iterations:3,rms_eps:1e-6,eps:1e-6};let count=shape.coefficients();
    let weights=(0..count*shape.wide()).map(|index|(index%23) as f32*0.001-0.01).collect::<Vec<_>>();
    Arc::new(HostBoundary::new(shape,NormShape::new(hidden as usize,1e-6,WeightType::Bf16).unwrap(),&weights,&vec![0.1;count],&[0.1,0.2,0.3],&vec![1.0;hidden as usize],3).unwrap())
}
fn mapping(slot_words:usize)->Vec<AtomicU32> {(0..HEADER+2*TP_MAILBOX_SLOTS*slot_words).map(|_|AtomicU32::new(0)).collect()}
fn input(words:usize,seed:usize)->Vec<f32> {(0..words).map(|index|if index%11==0 {-0.0} else {((index+seed)%29) as f32*0.125-1.5}).collect()}
fn publish_input(words:&[AtomicU32],slot_words:usize,sequence:u64,values:&[f32]) {
    let slot=sequence as usize%TP_MAILBOX_SLOTS;let base=HEADER+slot*slot_words;
    for (word,value) in words[base..].iter().zip(values) {word.store(value.to_bits(),Ordering::Relaxed);}
    words[SEND_LEN+slot].store((values.len()*4) as u32,Ordering::Relaxed);words[SEND_SEQ+slot].store(sequence as u32,Ordering::Release);
}
fn receive(words:&[AtomicU32],slot_words:usize,sequence:u64,expected:&[f32]) {
    let slot=sequence as usize%TP_MAILBOX_SLOTS;let deadline=Instant::now()+Duration::from_secs(5);
    while words[RECV_SEQ+slot].load(Ordering::Acquire)!=sequence as u32 {
        assert_eq!(words[ERROR].load(Ordering::Acquire),0,"service refused receive {sequence}");assert!(Instant::now()<deadline,"receive {sequence} timed out");std::hint::spin_loop();
    }
    let base=HEADER+(TP_MAILBOX_SLOTS+slot)*slot_words;
    for (index,(actual,expected)) in words[base..].iter().zip(expected).enumerate() {assert_eq!(actual.load(Ordering::Relaxed),expected.to_bits(),"receive {sequence}, word {index}");}
}
fn gone(queue:&Queue,take:u64) {
    let deadline=Instant::now()+Duration::from_secs(1);while queue.planes(take).is_some() {assert!(Instant::now()<deadline,"release {take} timed out");thread::yield_now();}
}
#[test]
fn tickets_are_monotonic_and_directions_are_independent() {
    let queue=Queue::new();let boundary=boundary(3,17);
    assert_eq!(queue.push(boundary.clone(),Kind::Run {previous:None}).unwrap(),Ticket {put:Some(1),take:1});
    assert_eq!(queue.push(boundary.clone(),Kind::Residual {of:1}).unwrap(),Ticket {put:None,take:2});
    assert_eq!(queue.push(boundary.clone(),Kind::Coefficients {of:1}).unwrap(),Ticket {put:None,take:3});
    assert_eq!(queue.push(boundary.clone(),Kind::Run {previous:Some(1)}).unwrap(),Ticket {put:Some(2),take:4});
    assert!(queue.push(boundary.clone(),Kind::Release {of:1}).is_err());queue.stop();assert!(queue.push(boundary,Kind::Run {previous:None}).is_err());
}
#[test]
fn exhausted_ticket_does_not_advance_either_direction() {
    let queue=Queue::new();let boundary=boundary(1,1);queue.state.lock().unwrap().puts=u32::MAX as u64;
    assert!(queue.push(boundary,Kind::Run {previous:None}).is_err());assert_eq!(queue.state.lock().unwrap().takes,0);
}
#[test]
fn lookahead_names_the_second_future_run_and_skips_deliveries() {
    let queue=Queue::new();let first=boundary(3,17);let second=boundary(3,17);let third=boundary(3,17);
    let active=queue.push(first.clone(),Kind::Run {previous:None}).unwrap();queue.push(first,Kind::Residual {of:active.take}).unwrap();
    queue.push(second,Kind::Run {previous:Some(active.take)}).unwrap();queue.push(third.clone(),Kind::Run {previous:Some(active.take)}).unwrap();
    queue.pop().unwrap().unwrap();assert!(Arc::ptr_eq(&queue.two_ahead().unwrap().unwrap(),&third));
}
#[test]
fn queued_chain_keeps_every_plane_equal_with_two_job_lookahead() {
    let boundaries=(0..4).map(|_|boundary(3,257)).collect::<Vec<_>>();let slot_words=boundaries[0].shape().wide();let words=mapping(slot_words);let queue=Queue::new();
    let mut tickets=Vec::new();let mut previous=None;
    for boundary in &boundaries {let ticket=queue.push(boundary.clone(),Kind::Run {previous}).unwrap();previous=Some(ticket.take);tickets.push(ticket);}
    thread::scope(|scope| {
        let worker=spawn(scope,TpMailboxHost::new(&words,slot_words),queue.clone(),2).unwrap();let mut previous:Option<Planes>=None;
        for (index,(boundary,ticket)) in boundaries.iter().zip(tickets).enumerate() {
            let source=input(if previous.is_some() {boundary.shape().hidden as usize} else {boundary.shape().wide()},index);let mut expected=boundary.planes();
            boundary.run(previous.as_ref().map(|planes|(&planes.residual[..],&planes.coefficients[..])),&source,&mut expected);
            publish_input(&words,slot_words,ticket.put.unwrap(),&source);receive(&words,slot_words,ticket.take,&expected.output);
            let actual=queue.planes(ticket.take).unwrap();
            for (actual,expected) in [&actual.residual,&actual.mixing,&actual.coefficients,&actual.collapsed,&actual.partials,&actual.output].into_iter().flatten()
                .zip([&expected.residual,&expected.mixing,&expected.coefficients,&expected.collapsed,&expected.partials,&expected.output].into_iter().flatten()) {assert_eq!(actual.to_bits(),expected.to_bits());}
            previous=Some(expected);
        }worker.finish().unwrap();
    });
}
#[test]
fn complete_chain_wraps_both_rings_and_delivers_raw_planes() {
    let first=boundary(3,17);let second=boundary(3,17);let slot_words=first.shape().wide().max(first.shape().coefficients());let words=mapping(slot_words);let queue=Queue::new();
    thread::scope(|scope| {
        let worker=spawn(scope,TpMailboxHost::new(&words,slot_words),queue.clone(),2).unwrap();let mut previous=None;
        for step in 0..18 {
            let boundary=if step%2==0 {first.clone()} else {second.clone()};let restart=step%6==0;
            let source=input(if restart {boundary.shape().wide()} else {boundary.shape().hidden as usize},step);let mut expected=boundary.planes();
            let prior=previous.as_ref().map(|(_,planes):&(u64,Planes)|(&planes.residual[..],&planes.coefficients[..]));boundary.run(if restart {None} else {prior},&source,&mut expected);
            let ticket=queue.push(boundary.clone(),Kind::Run {previous:if restart {None} else {previous.as_ref().map(|(take,_)|*take)}}).unwrap();
            let residual=queue.push(boundary.clone(),Kind::Residual {of:ticket.take}).unwrap();let coefficients=queue.push(boundary,Kind::Coefficients {of:ticket.take}).unwrap();
            if let Some((take,_))=&previous {queue.release(*take);}
            publish_input(&words,slot_words,ticket.put.unwrap(),&source);receive(&words,slot_words,ticket.take,&expected.output);
            receive(&words,slot_words,residual.take,&expected.residual);receive(&words,slot_words,coefficients.take,&expected.coefficients);
            if let Some((take,_))=&previous {gone(&queue,*take);}previous=Some((ticket.take,expected));
            if step==8 {thread::sleep(Duration::from_millis(5));}
        }
        queue.release(previous.unwrap().0);worker.finish().unwrap();assert_eq!(words[ERROR].load(Ordering::Acquire),0);
    });
}
#[test]
fn invalid_input_fails_the_mailbox_and_rejects_new_jobs() {
    let boundary=boundary(3,17);let slot_words=boundary.shape().wide();let words=mapping(slot_words);let queue=Queue::new();
    thread::scope(|scope| {
        let worker=scope.spawn(||serve(TpMailboxHost::new(&words,slot_words),&queue,1));let ticket=queue.push(boundary.clone(),Kind::Run {previous:None}).unwrap();
        publish_input(&words,slot_words,ticket.put.unwrap(),&[0.0]);assert!(worker.join().unwrap().unwrap_err().contains("plane length"));
    });assert_eq!(words[ERROR].load(Ordering::Acquire),1);assert!(queue.push(boundary,Kind::Run {previous:None}).is_err());assert!(queue.failed());
}
#[test]
fn stop_interrupts_a_missing_gpu_publication() {
    let boundary=boundary(3,17);let slot_words=boundary.shape().wide();let words=mapping(slot_words);let queue=Queue::new();queue.push(boundary,Kind::Run {previous:None}).unwrap();
    thread::scope(|scope| {
        let worker=scope.spawn(||serve(TpMailboxHost::new(&words,slot_words),&queue,1));let deadline=Instant::now()+Duration::from_secs(1);
        while queue.worker.get().is_none() {assert!(Instant::now()<deadline);thread::yield_now();}queue.stop();assert!(worker.join().unwrap().is_err());
    });assert_eq!(words[ERROR].load(Ordering::Acquire),1);
}
#[test]
fn a_release_queued_during_a_run_is_not_lost() {
    let boundary=boundary(3,17);let queue=Queue::new();let ticket=queue.push(boundary,Kind::Run {previous:None}).unwrap();let running=queue.pop().unwrap().unwrap();
    queue.release(ticket.take);let released=queue.pop().unwrap().unwrap();assert_eq!(running.ticket,ticket);assert_eq!(released.kind,Kind::Release {of:ticket.take});
}
#[test]
fn old_pass_delivery_keeps_its_own_generation() {
    let boundary=boundary(3,17);let slot_words=boundary.shape().wide();let words=mapping(slot_words);let queue=Queue::new();
    thread::scope(|scope| {
        let worker=spawn(scope,TpMailboxHost::new(&words,slot_words),queue.clone(),1).unwrap();let mut old=boundary.planes();let source=input(slot_words,0);boundary.run(None,&source,&mut old);
        let first=queue.push(boundary.clone(),Kind::Run {previous:None}).unwrap();publish_input(&words,slot_words,first.put.unwrap(),&source);receive(&words,slot_words,first.take,&old.output);
        let mut newer=boundary.planes();let source=input(slot_words,7);boundary.run(None,&source,&mut newer);let second=queue.push(boundary.clone(),Kind::Run {previous:None}).unwrap();
        publish_input(&words,slot_words,second.put.unwrap(),&source);receive(&words,slot_words,second.take,&newer.output);
        let delivery=queue.push(boundary,Kind::Residual {of:first.take}).unwrap();receive(&words,slot_words,delivery.take,&old.residual);queue.release(first.take);queue.release(second.take);worker.finish().unwrap();
    });
}
#[test]
fn delivery_after_release_fails_instead_of_returning_new_bytes() {
    let boundary=boundary(3,17);let slot_words=boundary.shape().wide();let words=mapping(slot_words);let queue=Queue::new();
    thread::scope(|scope| {
        let worker=scope.spawn(||serve(TpMailboxHost::new(&words,slot_words),&queue,1));let first=queue.push(boundary.clone(),Kind::Run {previous:None}).unwrap();
        queue.release(first.take);queue.push(boundary.clone(),Kind::Residual {of:first.take}).unwrap();publish_input(&words,slot_words,first.put.unwrap(),&input(slot_words,0));
        assert!(worker.join().unwrap().unwrap_err().contains("planes were released"));
    });assert_eq!(words[ERROR].load(Ordering::Acquire),1);
}
#[test]
fn too_many_deliveries_cannot_overwrite_a_receive_slot() {
    let boundary=boundary(3,17);let slot_words=boundary.shape().wide();let words=mapping(slot_words);let queue=Queue::new();
    thread::scope(|scope| {
        let worker=scope.spawn(||serve(TpMailboxHost::new(&words,slot_words),&queue,1));let first=queue.push(boundary.clone(),Kind::Run {previous:None}).unwrap();
        for _ in 0..TP_MAILBOX_SLOTS {queue.push(boundary.clone(),Kind::Residual {of:first.take}).unwrap();}publish_input(&words,slot_words,first.put.unwrap(),&input(slot_words,0));
        assert!(worker.join().unwrap().unwrap_err().contains("overwrite"));
    });assert_eq!(words[ERROR].load(Ordering::Acquire),1);assert_eq!(words[RECV_SEQ+1].load(Ordering::Acquire),1);
}
#[test]
fn startup_reservation_handles_heterogeneous_coefficient_counts() {
    let small=boundary(1,5);let large=boundary(3,5);let slot_words=large.shape().wide().max(large.shape().coefficients());let words=mapping(slot_words);let queue=Queue::new();queue.reserve_coefficients(large.shape().coefficients()).unwrap();
    thread::scope(|scope| {
        let worker=spawn(scope,TpMailboxHost::new(&words,slot_words),queue.clone(),2).unwrap();
        for boundary in [small,large] {let source=input(boundary.shape().wide(),0);let mut expected=boundary.planes();boundary.run(None,&source,&mut expected);
            let ticket=queue.push(boundary,Kind::Run {previous:None}).unwrap();publish_input(&words,slot_words,ticket.put.unwrap(),&source);receive(&words,slot_words,ticket.take,&expected.output);queue.release(ticket.take);}
        worker.finish().unwrap();
    });
}
#[test]
fn malformed_mapping_is_refused_before_polling() {
    let words=mapping(4);let queue=Queue::new();assert!(serve(TpMailboxHost::new(&words,usize::MAX),&queue,0).is_err());assert_eq!(words[ERROR].load(Ordering::Acquire),1);
}
#[test]
fn weight_touches_do_not_change_any_plane() {
    let boundary=boundary(3,257);let source=input(boundary.shape().wide(),0);let mut expected=boundary.planes();boundary.run(None,&source,&mut expected);
    for helpers in [0,1,2,4] {let pool=Pool::new(helpers,boundary.shape().coefficients());let mut actual=boundary.planes();pool.prepare(&boundary).unwrap();pool.run(&boundary,None,&source,&mut actual).unwrap();
        for (actual,expected) in [&actual.residual,&actual.mixing,&actual.coefficients,&actual.collapsed,&actual.partials,&actual.output].into_iter().flatten()
            .zip([&expected.residual,&expected.mixing,&expected.coefficients,&expected.collapsed,&expected.partials,&expected.output].into_iter().flatten()) {assert_eq!(actual.to_bits(),expected.to_bits());}
    }assert_eq!(boundary.host_bytes().unwrap(),HostBoundary::parameter_bytes(boundary.shape(),3).unwrap());
}
#[test]
fn parameter_accounting_includes_padded_rows_and_rejects_zero_cores() {
    for streams in [1,3,4,7,16] {for cores in [1,2,3,7] {
        let shape=Shape {rows:1,streams,hidden:17,iterations:3,rms_eps:1e-6,eps:1e-6};let count=shape.coefficients();
        let boundary=HostBoundary::new(shape,NormShape::new(17,1e-6,WeightType::F32).unwrap(),&vec![0.1;count*shape.wide()],&vec![0.2;count],&[0.1,0.2,0.3],&vec![1.0;17],cores).unwrap();
        assert_eq!(boundary.host_bytes().unwrap(),HostBoundary::parameter_bytes(shape,cores).unwrap());
    }}assert!(HostBoundary::parameter_bytes(boundary(3,17).shape(),0).is_err());
}
