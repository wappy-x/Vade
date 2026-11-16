//! Host checks of the pending receive boundary. No RDMA pointer is dereferenced.
use super::*;
    #[test]
    fn pending_rank_receive_refuses_intervening_wire_operations() {
        type Operation=fn(&mut Ring)->Result<()>;
        let operations:[Operation;11]=[
            |r|r.command(&mut [0]),|r|r.command_payload(&mut [0]),|r|r.barrier(),|r|r.step_barrier(),
            |r|r.reduce_small(&mut [0.]),|r|r.all_gather(&mut [0;4],4),|r|r.gather_blocks(&mut [0.],1,1),
            |r|r.gather_batch(&[0.],1,1,1).map(|_|()),|r|r.gather_batch_f32(&[0.],1,1,1).map(|_|()),
            |r|r.transfer_from_last(0,&mut [0]),|r|r.transfer_from_last_batch(0,&mut [&mut [0]]),
        ];
        let address=SocketAddr::from(([127,0,0,1],0));
        for (case,operation) in operations.into_iter().enumerate() {
            let mut ring=Ring::connect_options(0,1,address,address,false).unwrap();
            // Simulate only the pending state. No pointer is dereferenced and no receive is posted on this host.
            ring.first_receive=Some((0,4,1));
            let error=operation(&mut ring).expect_err("an intervening wire operation must refuse before its no-op path");
            assert!(error.contains("requires its gather next"),"case={case}: {error}");
            assert!(ring.first_receive.is_none());
        }
    }
    #[test]
    fn pending_rank_receive_accepts_multiple_words_per_rank() {
        let address=SocketAddr::from(([127,0,0,1],0));
        let mut ring=Ring::connect_options(0,1,address,address,false).unwrap();
        let mut buffer=[0.0f32;16];
        // The prompt caller uses units=parts, width=1, tokens=words. The transport
        // normalizes this contiguous block to one token only in gather_rdma_ranges.
        // Simulate the pending descriptor; this host test posts no device receive.
        ring.first_receive=Some((buffer.as_ptr() as usize,std::mem::size_of_val(&buffer),1));
        ring.gather_rank_blocks(&mut buffer,1,1,16).unwrap();
        ring.abort();
    }
    #[test]
    fn pending_rank_receive_rejects_a_different_gather_buffer() {
        let address=SocketAddr::from(([127,0,0,1],0));
        let mut ring=Ring::connect_options(0,1,address,address,false).unwrap();
        ring.first_receive=Some((0,4,1));
        assert!(ring.gather_rank_blocks(&mut [0.],1,1,1).unwrap_err().contains("differs from its gather"));
        assert!(ring.first_receive.is_none());
    }
