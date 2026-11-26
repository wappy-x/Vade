//! Run the real open/close pair on CUDA and RDMA, then fail the work between.
use super::*;
use std::{io::{Read,Write},net::{SocketAddr,TcpListener,TcpStream},time::{Duration,Instant}};

#[test]
#[ignore="requires two CUDA/RDMA ranks; stdin gives rank, own socket, peer socket, fault role"]
fn lookahead_device_closes_after_a_failed_shared_step() {
    let mut line=String::new();std::io::stdin().read_line(&mut line).unwrap();
    let fields=line.split_whitespace().collect::<Vec<_>>();assert_eq!(fields.len(),4);
    let rank=fields[0].parse::<usize>().unwrap();let own=fields[1].parse::<SocketAddr>().unwrap();let peer=fields[2].parse::<SocketAddr>().unwrap();
    let inject=fields[3].parse::<bool>().unwrap();
    let sync=if inject {let mut socket=own;socket.set_port(socket.port()+1);Some(TcpListener::bind(socket).unwrap())} else {None};
    let ring=Ring::connect(rank,2,own,peer).unwrap();assert!(ring.uses_rdma(),"RDMA is required for the posted-receive proof");
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),512<<20,|b| {
        let mut weights=Weights::new(Some(ring),false);
        let (rows,cols,tokens)=(1024usize,2048usize,1024usize);
        let shape=Shape {op:Op::NvProject,rows:rows as u32,cols:cols as u32,block_rows:spark_gpu_probe::expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:0.01,activation:1.,bound:7.,role:0};
        for (name,byte) in [("gate",0x22u8),("up",0x33u8)] {
            let weight=raw(b,&vec![byte;rows*cols/2])?;let scales=raw(b,&vec![0x38;rows*cols/16])?;
            weights.matrices.insert(name.into(),Matrix {rows:Rows {start:rank*rows,count:rows,total:2*rows,split:true},cols,storage:Storage::Packed {shape,weight,scales},absorbed:None});
        }
        let input=b.upload_vector(&(0..tokens*cols).map(|i|((i/cols%7+1) as f32)/32.).collect::<Vec<_>>())?;
        let order=(0..tokens).rev().map(|row|row as f32).collect::<Vec<_>>();
        let places=(0..tokens).flat_map(|row|[0.,tokens as f32,row as f32]).collect::<Vec<_>>();
        let first=[("gate",tokens)];let second=[("up",tokens)];
        let queue=b.resident_queue()?;
        // The same gathered operands with the ordinary, closed exchange give
        // the reference. This is also a successful wire control before the fault.
        let expected=weights.apply_experts_gathered(b,&first,Some((&second,7.)),&input,&order,&places)?.ok_or("gathered fixture was refused")?;
        let expected=b.read_vector_values(&expected)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert!(weights.apply_experts_open(b,&first,(&second,7.),&input,Some(&order),&places)?);
        assert!(weights.ahead.as_ref().unwrap().gather.first,"the open must post a receive");
        let ahead=b.round_bf16_lanes(&input)?;
        let actual=weights.apply_experts_close(b,false)?;
        let actual=b.read_vector_values(&actual)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(actual,expected,"lookahead changed the gathered result");drop(ahead);
        println!("LOOKAHEAD DEVICE valid rank={rank} words={} exact=true",actual.len());
        assert!(weights.apply_experts_open(b,&first,(&second,7.),&input,Some(&order),&places)?);
        assert!(weights.ahead.as_ref().unwrap().gather.first,"fault must land after a real posted receive");
        // This separate fixture socket makes both opens complete before the
        // failure. No extra command is put on the production ring.
        let mut socket=if let Some(listener)=&sync {listener.accept().unwrap().0} else {let mut address=peer;address.set_port(address.port()+1);TcpStream::connect(address).unwrap()};
        socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();socket.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
        socket.write_all(&[1]).unwrap();socket.read_exact(&mut [0]).unwrap();
        let start=Instant::now();
        let failed=if inject {
            let mut out=b.vector(1)?;
            let error=b.copy_range(&input,&mut out,0,0,2).unwrap_err();
            assert!(error.contains("exceeds vector"));true
        } else {false};
        let error=weights.apply_experts_close(b,failed).err().expect("both ranks must fail the interrupted exchange");
        assert!(start.elapsed()<Duration::from_secs(5),"failure did not close the exchange promptly");
        if failed {assert_eq!(error,"work queued ahead of a prompt exchange failed");}
        assert!(weights.ahead.is_none(),"no open exchange can survive failure");
        assert!(weights.ring.as_mut().unwrap().barrier().is_err(),"a failed ring cannot accept another command");
        b.synchronize()?;drop(queue);
        println!("LOOKAHEAD DEVICE FAULT PASS rank={rank} posted=true open_cleared=true followup_refused=true error={error}");Ok(())
    }).unwrap();
}
