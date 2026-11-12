//! Host transport and packet placement. Device exactness still needs the pair gate.
use engine_model::plan::layer::Topology;
use spark_gpu_probe::prompt_ops::{self,Shape};
use spark_tp::{sparse_heads::{self,Layout},transport::Ring};
use std::net::TcpListener;

fn payload(rank:usize,tokens:usize,width:usize)->Vec<f32> {
    (0..tokens*width).map(|i|f32::from_bits([0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,0x3f800000+(rank*1009+i) as u32][i%7])).collect()
}
#[test]
fn token_major_packets_cross_two_ranks_in_both_directions() {
    let parts=2;
    for path in [false,true] {
        let listeners=(0..parts).map(|_|TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,0)).unwrap()).collect::<Vec<_>>();
        let addresses=listeners.iter().map(|l|l.local_addr().unwrap()).collect::<Vec<_>>();drop(listeners);
        std::thread::scope(|scope| {
            let handles=(0..parts).map(|rank| {let addresses=&addresses;scope.spawn(move || {
                let mut ring=Ring::connect_options(rank,parts,addresses[rank],addresses[(rank+1)%parts],path).unwrap();
                for tokens in [1,3,16,17,31,255,256,257,1024] {for width in [3,4,144] {for _ in 0..3 {
                    let packets=(0..parts).map(|r|payload(r,tokens,width)).collect::<Vec<_>>();let words=tokens*width;
                    let mut exchange=vec![f32::NAN;parts*words];exchange[rank*words..(rank+1)*words].copy_from_slice(&packets[rank]);
                    ring.gather_rank_blocks(&mut exchange,parts,1,words).unwrap();
                    let shape=Shape::rank_rows(tokens,width,parts,true).unwrap();
                    let output=prompt_ops::reference(shape,[&exchange,&[],&[],&[],&[]],&[]).unwrap();
                    let layouts=(0..parts).map(|r|Layout::new(parts as u64,1,width as u64,Topology {rank:r as u32,parts:parts as u32}).unwrap()).collect::<Vec<_>>();
                    let refs=layouts.iter().copied().zip(packets.iter().map(Vec::as_slice)).collect::<Vec<_>>();
                    let expected=sparse_heads::gather_attended(&refs,tokens).unwrap();
                    assert!(output.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()),"rank={rank} tokens={tokens} width={width}");
                }}}
            })}).collect::<Vec<_>>();
            for handle in handles {handle.join().unwrap();}
        });
    }
}
