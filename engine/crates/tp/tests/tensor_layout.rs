use engine_model::plan::tensor_split::Ownership;
use spark_tp::tensor_layout::GatherLayout;
use engine_fabric::exchange::Exchange;

fn layout(shape: &[u64], axis: usize, unit: u64, parts: u32, bytes: usize) -> GatherLayout {
    GatherLayout::new(Ownership::equal(shape,axis,unit,parts).unwrap(),bytes,usize::MAX).unwrap()
}
fn source(n: usize) -> Vec<u8> { (0..n).map(|i| (i*37+i/251) as u8).collect() }
fn local(p: &GatherLayout, rank: u32, global: &[u8]) -> Vec<u8> {
    let (outer,axis,inner) = p.ownership().geometry().unwrap();
    let r = &p.ownership().owner(rank).unwrap().interval;
    let stride = inner as usize*p.element_bytes();
    (0..outer as usize).flat_map(|row| global[row*axis as usize*stride+r.start as usize*stride..row*axis as usize*stride+r.end as usize*stride].iter().copied()).collect()
}
struct CopyWire { packets: Vec<Vec<u8>>, calls: usize, fail: bool }
impl Exchange for CopyWire {
    fn gather_bytes(&mut self, wire: &mut [u8], offsets: &[usize]) -> Result<(),String> {
        self.calls+=1;
        for (rank,packet) in self.packets.iter().enumerate().rev() { wire[offsets[rank]..offsets[rank+1]].copy_from_slice(packet); }
        if self.fail { Err("injected copy failure".into()) } else { Ok(()) }
    }
    fn gather_f32(&mut self, _: &mut [f32], _: &[usize]) -> Result<(),String> { panic!("byte gather must not cast") }
    fn reduce(&mut self, _: &mut [f32], _: &mut [f32]) -> Result<(),String> { panic!("tensor interval gather must not reduce") }
}

#[test]
fn global_order_matches_for_all_axes_counts_and_element_widths() {
    for parts in 1..=engine_model::shapes::MAX_SPARKS {
        for bytes in [1,2,4,8] {
            let n = u64::from(parts)*6;
            for (shape,axis,unit) in [(vec![2,n,3],1,2),(vec![n,3,2],0,3),(vec![2,3,n],2,1)] {
                let p = layout(&shape,axis,unit,parts,bytes); let global = source(p.bytes());
                let packets: Vec<_> = (0..parts).map(|r| local(&p,r,&global)).collect();
                let mut wire = vec![0;p.bytes()];
                for rank in (0..parts).rev() { p.pack(rank,&packets[rank as usize],&mut wire).unwrap(); }
                let mut got = vec![0;p.bytes()]; p.unpack(&wire,&mut got).unwrap(); assert_eq!(got,global);
                let mut exchange = CopyWire {packets,calls:0,fail:false};
                for rank in 0..parts { p.gather(&mut exchange,rank,&local(&p,rank,&global),&mut got).unwrap(); assert_eq!(got,global); }
                assert_eq!(exchange.calls,parts as usize);
            }
        }
    }
}

#[test]
fn wire_preserves_signed_zero_nan_and_quantized_bits() {
    let bits = [0x00000000u32,0x80000000,0x7fc12345,0xff812345,0x3f800001,0x00000001,0x7f800000,0xff800000];
    let global: Vec<_> = bits.into_iter().flat_map(u32::to_le_bytes).collect();
    let p = layout(&[2,4],1,1,2,4); let mut wire = vec![0;p.bytes()];
    for rank in 0..2 { p.pack(rank,&local(&p,rank,&global),&mut wire).unwrap(); }
    let mut output = vec![0;p.bytes()]; p.unpack(&wire,&mut output).unwrap(); assert_eq!(output,global);
    assert_ne!(wire,global,"rank-major wire is not global tensor order");
    println!("TENSOR_BYTES expected_words={bits:08x?} actual_words={:08x?}",output.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>());
}

#[test]
fn projection_input_has_full_k_before_any_dot_product() {
    use engine_model::plan::{tensor_split::ProjectionRows,storage::WeightStorage};
    for parts in 1..=engine_model::shapes::MAX_SPARKS {
        let k = u64::from(parts)*8;
        let p = layout(&[2,k/2],1,1,parts,4);
        let input: Vec<_> = (0..k as usize).map(|i| [1e20f32,-1e20,3.0,0.25][i%4]).collect();
        let bytes: Vec<_> = input.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect();
        let mut exchange = CopyWire {packets:(0..parts).map(|r|local(&p,r,&bytes)).collect(),calls:0,fail:false};
        let weights = Ownership::equal(&[u64::from(parts)*4,k],0,1,parts).unwrap();
        let reference = input.iter().fold(0f32,|sum,&x|sum+x);
        for rank in 0..parts {
            let projection = ProjectionRows::new(&weights,rank,WeightStorage::Plain).unwrap();
            let mut full = vec![0;p.bytes()];
            p.gather_projection_input(&projection,&mut exchange,rank,&local(&p,rank,&bytes),&mut full).unwrap();
            let dot = full.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).fold(0f32,|sum,x|sum+x);
            assert_eq!(dot.to_bits(),reference.to_bits());
        }
        let wrong = ProjectionRows::new(&Ownership::equal(&[4,k-1],0,1,1).unwrap(),0,WeightStorage::Plain).unwrap();
        let calls = exchange.calls;
        assert!(p.gather_projection_input(&wrong,&mut exchange,0,&local(&p,0,&bytes),&mut vec![0;p.bytes()]).is_err());
        assert_eq!(exchange.calls,calls);
    }
}

#[test]
fn bounds_overflow_and_exchange_failure_leave_output_unchanged() {
    let own = Ownership::equal(&[2,4,3],1,1,2).unwrap();
    assert!(GatherLayout::new(own.clone(),0,usize::MAX).is_err());
    assert!(GatherLayout::new(own.clone(),usize::MAX,usize::MAX).is_err());
    assert!(GatherLayout::new(own.clone(),4,95).is_err());
    let p = GatherLayout::new(own,4,96).unwrap();
    assert!(GatherLayout::new(Ownership::equal(&[u64::MAX],0,1,1).unwrap(),1,usize::MAX).is_err());
    let global = source(p.bytes()); let mut out = vec![0xa5;p.bytes()]; let before = out.clone();
    let mut e = CopyWire {packets:(0..2).map(|r|local(&p,r,&global)).collect(),calls:0,fail:true};
    assert!(p.gather(&mut e,0,&local(&p,0,&global),&mut out).is_err()); assert_eq!(out,before); assert_eq!(e.calls,1);
    assert!(p.gather(&mut e,2,&[],&mut out).is_err()); assert_eq!(e.calls,1);
    assert!(p.gather(&mut e,0,&[0],&mut out).is_err()); assert_eq!(e.calls,1);
    assert!(p.gather(&mut e,0,&local(&p,0,&global),&mut [0]).is_err()); assert_eq!(e.calls,1);
    assert!(p.unpack(&global[..95],&mut out).is_err()); assert_eq!(out,before);
    assert!(p.pack(0,&[1],&mut out).is_err()); assert_eq!(out,before);
}

#[test]
fn mailbox_word_positions_match_global_tensor_positions() {
    for parts in 1..=engine_model::shapes::MAX_SPARKS {
        let p = layout(&[3,u64::from(parts)*8,2],1,2,parts,2);
        let global = source(p.bytes()); let mut output = vec![0;p.bytes()];
        for rank in (0..parts).rev() {
            let bytes = local(&p,rank,&global); let m = p.mailbox(rank,bytes.len()).unwrap();
            assert_eq!(m.packet_bytes(),bytes.len()); assert_eq!(m.words()*4,p.bytes()); assert_eq!(m.ranks(),parts as usize);
            assert!(p.mailbox(rank,bytes.len()-1).is_err());
            for (word,b) in bytes.chunks_exact(4).enumerate() { let to=m.destination(word).unwrap()*4; output[to..to+4].copy_from_slice(b); }
            assert!(m.destination(bytes.len()/4).is_err());
        }
        assert_eq!(output,global);
    }
    assert!(layout(&[3,6],1,1,2,2).mailbox(0,usize::MAX).is_err());
    assert!(layout(&[3,6],1,1,2,1).mailbox(0,usize::MAX).is_err());
    assert!(layout(&[8],0,1,2,4).mailbox(2,usize::MAX).is_err());
    assert!(layout(&[u64::from(u32::MAX)+1],0,1,1,1).mailbox(0,usize::MAX).is_err());
}

#[test]
fn device_route_limits_are_separate_from_logical_topology() {
    for parts in [1,2,4,8] {
        let p = layout(&[2,8192],1,128,parts,2);
        for ring in [false,true] {
            let channels = p.mailbox_channels(ring);
            println!("TENSOR_MAILBOX parts={parts} ring={ring} binding={}",if channels.is_ok() {"LAYOUT"} else {"REFUSED"});
            assert_eq!(channels.is_ok(),parts != 8);
        }
    }
}

#[test]
fn checked_intervals_cross_the_existing_tcp_ring_and_path() {
    use std::net::TcpListener;
    use spark_tp::transport::Ring;
    for parts in 1..=engine_model::shapes::MAX_SPARKS {
        for path in [false,true] {
            let sockets: Vec<_> = (0..parts).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
            let addresses: Vec<_> = sockets.iter().map(|s| s.local_addr().unwrap()).collect();
            drop(sockets);
            let p = layout(&[3,u64::from(parts)*4,2],1,2,parts,2); let global = source(p.bytes());
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..parts as usize).map(|rank| {
                    let (p,global,addresses) = (&p,&global,&addresses);
                    s.spawn(move || {
                        let mut ring = Ring::connect_options(rank,parts as usize,addresses[rank],addresses[(rank+1)%parts as usize],path).unwrap();
                        for _ in 0..3 {
                            let mut got = vec![0xa5;p.bytes()];
                            p.gather_ring(&mut ring,&local(p,rank as u32,global),&mut got).unwrap();
                            assert_eq!(&got,global);
                        }
                    })
                }).collect();
                for h in handles { h.join().unwrap(); }
            });
            println!("TENSOR_TCP parts={parts} path={path} repeats=3 bytes={} identical=true",p.bytes());
        }
    }
    let address = "127.0.0.1:0".parse().unwrap();
    let mut ring = Ring::connect_options(0,1,address,address,false).unwrap();
    let p = layout(&[8],0,1,2,4); let mut got = vec![0xa5;p.bytes()]; let before = got.clone();
    assert!(p.gather_ring(&mut ring,&[0;16],&mut got).is_err()); assert_eq!(got,before);
}
