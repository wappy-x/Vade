// Host arithmetic and rank protocol; device allocation and speed need the served gate.
#[test]
fn draft_admission_only_funds_eligible_lone_requests() {
    let c=fixture::Checkpoint::new_with_predictor(80);
    let make=||Split::new(Host,FaultWire {fault:std::rc::Rc::new(std::cell::Cell::new(false)),aborted:std::rc::Rc::new(std::cell::Cell::new(false))}).unwrap();
    let plain=Factory::new(&c.admission,make(),80,vec![0]).unwrap();
    let loaded=Factory::new(&c.admission,make(),80,vec![0]).unwrap().with_predictor(3).unwrap();
    let mut limits=spark_tp::row_split::limits(80,72).unwrap();limits.next_mtp_depth=Some(3);
    let mut request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":72,"temperature":0}"#,true,&limits).unwrap();
    request.mtp_depth=Some(3);
    let base=plain.request_reservation_bytes_for_prompt(3,75,true,None,false).unwrap();
    let price=|r:&spark_serve::protocol::Request,contended|loaded.request_reservation_bytes_for_request(3,75,contended,None,false,r).unwrap();
    assert!(price(&request,false)>base,"a lone trial must fund its predictor state");
    assert_eq!(price(&request,true),base,"contention must refund the whole optional charge");
    request.draft=Some(spark_serve::protocol::DraftMode::Strict);
    assert!(price(&request,false)>base);assert_eq!(price(&request,true),base);
    request.draft=Some(spark_serve::protocol::DraftMode::Plain);assert_eq!(price(&request,false),base);
    request.draft=None;request.max_tokens=64;assert_eq!(price(&request,false),base);
    request.max_tokens=72;request.mtp_depth=Some(0);assert_eq!(price(&request,false),base);
    request.mtp_depth=Some(3);request.sampling.temperature=1.0;assert_eq!(price(&request,false),base);
    plain.shutdown().unwrap();loaded.shutdown().unwrap();
}

#[test]
fn draft_admission_retires_rejected_rows_on_both_ranks_and_keeps_plain_bits() {
    let c=fixture::Checkpoint::new_with_predictor(80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {
            let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap().worker().unwrap();
        });
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap();
        let mut limits=spark_tp::row_split::limits(80,64).unwrap();limits.next_mtp_depth=Some(3);
        let mut request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"draft":"strict","mtp_depth":3}"#,true,&limits).unwrap();
        let mut draft=engine.session().unwrap();draft.configure_request(&request).unwrap();draft.prefill(&[3,5,1]).unwrap();
        Session::step(&mut draft,7).unwrap();Session::step(&mut draft,7).unwrap();
        let target=draft.draft_target().unwrap();target.limit_model_depth(3);
        assert!(!target.propose_model(&[3,5,1,7],2,3).unwrap().unwrap().tokens.is_empty());
        target.verify_batch(&[2,6,4,1]).unwrap();target.truncate(7).unwrap();target.truncate(6).unwrap();
        draft.disable_model_draft();assert!(draft.draft_over());assert!(!draft.draft_over());assert!(draft.draft_target().is_none());
        request.draft=Some(spark_serve::protocol::DraftMode::Plain);
        let mut plain=engine.session().unwrap();plain.configure_request(&request).unwrap();plain.prefill(&[3,5,1,7,7,2]).unwrap();
        for token in [6,4,1,5,3,2] {
            let want=Session::step(&mut plain,token).unwrap();let got=Session::step(&mut draft,token).unwrap();
            assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),want.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        drop(plain);drop(draft);
        // A later, newly funded lone chat can still start the predictor.
        request.draft=Some(spark_serve::protocol::DraftMode::Strict);
        let mut lone=engine.session().unwrap();lone.configure_request(&request).unwrap();lone.prefill(&[3,5,1]).unwrap();Session::step(&mut lone,7).unwrap();Session::step(&mut lone,7).unwrap();
        let target=lone.draft_target().unwrap();target.limit_model_depth(3);
        assert!(!target.propose_model(&[3,5,1,7],2,3).unwrap().unwrap().tokens.is_empty());
        drop(lone);engine.shutdown().unwrap();worker.join().unwrap();
    });
}
