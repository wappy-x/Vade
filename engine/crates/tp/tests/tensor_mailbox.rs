//! A device copy test, not a model or inter-Spark transport test.
#[cfg(target_os="linux")]
#[test]
#[ignore = "requires CUDA; run the shipped test binary on a booked Spark"]
fn checked_tensor_mailbox_copies_preserve_bytes() -> Result<(),String> {
    use std::sync::atomic::Ordering;
    use engine_model::plan::tensor_split::Ownership;
    use spark_tp::tensor_layout::GatherLayout;
    use spark_gpu_probe::{driver::with_backend, layer_ptx, tp_mailbox_host::{SEND_SEQ,SEND_LEN}};
    with_backend(0,layer_ptx()?,usize::MAX,|b| {
        for parts in [2,4] {
            for element_bytes in [2,4] {
                let p = GatherLayout::new(Ownership::equal(&[3,32,2],1,2,parts).map_err(|e|e.to_string())?,element_bytes,usize::MAX)?;
                let channels = p.mailbox_channels(false)?;
                let mailbox = b.tp_mailbox_routed(channels)?;
                let host = mailbox.take_host().ok_or("test mailbox has no host endpoint")?;
                let words = host.words();
                let expected: Vec<_> = (0..p.bytes()/4).map(|i| [0u32,0x80000000,0x7fc12345,0xff812345,0x00000001][i%5] ^ ((i as u32)<<7)).collect();
                let expected_bytes: Vec<_> = expected.iter().flat_map(|v|v.to_le_bytes()).collect();
                let mut packets = Vec::new();
                for origin in 0..parts {
                    let m = p.mailbox(origin,mailbox.slot_bytes())?;
                    let mut own = vec![0xa5a5a5a5u32;expected.len()];
                    for word in 0..m.packet_bytes()/4 { let at=m.destination(word)?; own[at]=expected[at]; }
                    let mut v = b.vector(expected.len())?;
                    b.write_vector_bytes(&mut v,&own.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
                    let seq = u64::from(origin)+1;
                    m.put(b,&mailbox,&v,seq)?; b.synchronize()?;
                    let slot = seq as usize%channels.slots;
                    if words[SEND_SEQ+slot].load(Ordering::Acquire) != seq as u32 || words[SEND_LEN+slot].load(Ordering::Relaxed) as usize != m.packet_bytes() {
                        return Err("device tensor packet has the wrong sequence or extent".into());
                    }
                    let start = channels.send_slot(slot)+channels.frame;
                    let packet: Vec<_> = words[start..start+m.packet_bytes()/4].iter().map(|v|v.load(Ordering::Relaxed)).collect();
                    for (word,value) in packet.iter().enumerate() { if *value != expected[m.destination(word)?] { return Err("device put changed tensor bytes".into()); } }
                    packets.push(packet);
                }
                for receiver in 0..parts as usize {
                    let mut v = b.vector(expected.len())?;
                    let mut initial = vec![0xa5a5a5a5u32;expected.len()];
                    let own = p.mailbox(receiver as u32,mailbox.slot_bytes())?;
                    for word in 0..own.packet_bytes()/4 { let at=own.destination(word)?; initial[at]=expected[at]; }
                    b.write_vector_bytes(&mut v,&initial.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
                    for origin in 0..parts as usize {
                        if origin == receiver { continue; }
                        let seq = u64::from(parts)+1;
                        let slot = seq as usize%channels.slots;
                        let channel = channels.channel(origin,receiver);
                        let start = channels.recv_slot(channel,slot)+channels.frame;
                        for (target,value) in words[start..start+packets[origin].len()].iter().zip(&packets[origin]) { target.store(*value,Ordering::Relaxed); }
                        words[channels.recv_seq(channel,slot)].store(seq as u32,Ordering::Release);
                        p.mailbox(origin as u32,mailbox.slot_bytes())?.take(b,&mailbox,&mut v,receiver,seq,1_000_000_000)?;
                    }
                    b.synchronize()?;
                    if mailbox.failed() { return Err("device tensor mailbox failed".into()); }
                    let got: Vec<_> = b.read_vector_values(&v)?.into_iter().flat_map(|v|v.to_bits().to_le_bytes()).collect();
                    if got != expected_bytes { return Err("device take changed global tensor bytes".into()); }
                    println!("TENSOR_DEVICE parts={parts} receiver={receiver} element_bytes={element_bytes} bytes={} identical=true",p.bytes());
                }
            }
        }
        Ok(())
    })
}
