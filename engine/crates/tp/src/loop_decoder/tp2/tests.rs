//! The cut of `tp2` on counts and words alone: where the two halves meet, and the order of the joined words.
use super::{BAND,joined,part};

#[test]
fn a_dense_cut_falls_on_a_band_and_covers_every_row_once() {
    for rows in [2*BAND,2*BAND+1,1024,1056,2304,6552,8600,200_064,200_058] {
        let (lower,upper)=(part(rows,true,0),part(rows,true,1));
        assert_eq!((lower.start,lower.end,upper.start,upper.end),(0,upper.start,lower.end,rows),"{rows} rows");
        assert!(lower.end.is_multiple_of(BAND) && !lower.is_empty() && !upper.is_empty(),"{rows} rows: {lower:?} {upper:?}");
        assert!(lower.len().abs_diff(upper.len())<=2*BAND,"{rows} rows: the halves differ by more than two bands");
    }
}
#[test]
fn a_pick_cut_gives_each_half_the_same_count() {
    assert_eq!((part(8,false,0),part(8,false,1)),(0..4,4..8));
    assert_eq!((part(7,false,0),part(7,false,1)),(0..3,3..7));
}
#[test]
fn joined_words_are_the_words_of_one_spark_on_either_half() {
    // Three input rows of a matrix of five rows cut 2 + 3: row r of the whole result is [r*10 .. r*10 + 5].
    let whole=(0..3).flat_map(|r|(0..5).map(move |c|(r*10+c) as f32)).collect::<Vec<_>>();
    let lower=(0..3).flat_map(|r|(0..2).map(move |c|(r*10+c) as f32)).collect::<Vec<_>>();
    let upper=(0..3).flat_map(|r|(2..5).map(move |c|(r*10+c) as f32)).collect::<Vec<_>>();
    assert_eq!(joined(&lower,&upper,3,0),whole);
    assert_eq!(joined(&upper,&lower,3,1),whole);
}
#[test]
fn the_joined_cut_of_a_visit_falls_inside_its_query_stack() {
    // The query stack (query, relative, sink, gate) of 24 heads of 128 and 16, then the key and the value of 8 heads.
    let (stack,key,value)=(24*128+24*16+24+24*128,8*128,8*128);
    let cut=part(stack+key+value,true,0).end;
    assert!(cut>0 && cut<stack && cut.is_multiple_of(BAND),"cut {cut} of {stack}");
    assert_eq!(part(stack+key+value,true,1),cut..stack+key+value);
}
#[test]
fn a_row_cut_gives_each_spark_whole_rows_once() {
    for rows in [2usize,3,8,16,31,32,93,4096] {
        let (lower,upper)=(part(rows,false,0),part(rows,false,1));
        assert_eq!((lower.start,lower.end,upper.end),(0,upper.start,rows));
        assert!(!lower.is_empty() && !upper.is_empty());
    }
}
#[test]
fn a_pair_start_takes_rank_0_or_1_of_world_2_and_no_other() {
    use super::super::service::start::settings;
    let start=|places:&str|settings(&format!("{places} 10.0.0.1:4000 10.0.0.2:4000 flash.ptx model 4096 127.0.0.1:8000 0 --diagnostic-unqualified-text --pair").split_whitespace().map(str::to_owned).collect::<Vec<_>>(),None);
    assert!(start("0 2").unwrap().options.pair && start("1 2").unwrap().options.pair);
    for bad in ["2 2","0 1","0 4"] {assert!(start(bad).is_err(),"{bad}");}
    assert!(super::serve::check(0,false).is_ok() && super::serve::check(7,false).is_ok() && super::serve::check(0,true).is_err());
}
#[test]
fn window_split_balances_causal_flops_and_skips_short_windows() {
    // Sliding window (view <= 1024): skipped so trade does not outweigh short compute.
    assert_eq!(super::window_split_mid(4096,1024,4096,64),None);
    // Global chunk 0 (0..4096): causal triangle balances near 4096 / sqrt(2) ≈ 2896, aligned to 64.
    let mid0=super::window_split_mid(4096,4096,0,64).unwrap();
    assert_eq!(mid0%64,0);
    assert!((2816..=2944).contains(&mid0),"mid0={mid0}");
    // Global chunk 4 (16384..20480): near 50/50 split around 2176.
    let mid4=super::window_split_mid(4096,20480,16384,64).unwrap();
    assert_eq!(mid4%64,0);
    assert!((2048..=2240).contains(&mid4),"mid4={mid4}");
}
#[test]
fn the_mirror_wire_gives_each_message_whole_and_in_order() {
    let listener=std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address=listener.local_addr().unwrap();
    let sender=std::thread::spawn(move ||{
        let mut wire=super::mirror::Wire::new(std::net::TcpStream::connect(address).unwrap()).unwrap();
        for n in [0usize,1,5,70_000] {wire.send_for_test(&(0..n as u64).collect::<Vec<_>>()).unwrap();}
    });
    let mut wire=super::mirror::Wire::new(listener.accept().unwrap().0).unwrap();
    for n in [0usize,1,5,70_000] {assert_eq!(wire.receive_for_test().unwrap(),Some((0..n as u64).collect::<Vec<_>>()));}
    sender.join().unwrap();
    assert_eq!(wire.receive_for_test().unwrap(),None);
}
#[test]
fn stack_split_picks_balanced_band_aligned_boundary() {
    // Separate K/V stack [3072, 384, 24, 3072]: k=2 gives 3456 (multiple of 16) vs 3096.
    assert_eq!(super::stack_split(&[3072,384,24,3072]),Some(2));
    // Shared QKV stack [3072, 384, 24, 3072, 1024, 1024]: k=2 gives 3456 (multiple of 16) vs 5144.
    assert_eq!(super::stack_split(&[3072,384,24,3072,1024,1024]),Some(2));
    // Single matrix or no band-aligned split: None.
    assert_eq!(super::stack_split(&[3072]),None);
    assert_eq!(super::stack_split(&[24,24]),None);
}
#[test]
fn odd_row_pick_shift_and_join_reconstructs_exact_pick_order() {
    let (rows,picks,id_stride,width)=(5usize,8usize,16usize,4usize);
    let half_picks=picks/2;
    let mut route=vec![0.0f32;rows*id_stride];
    for r in 0..rows {
        for p in 0..picks {
            route[r*id_stride+p]=(r*100+p) as f32;
            route[r*id_stride+picks+p]=0.125;
        }
    }
    // Half 1 shifts `route` left by `half_picks` (wrapping the first `half_picks` words to the end).
    let mut shifted=route[half_picks..].to_vec();
    shifted.extend_from_slice(&route[..half_picks]);
    let mut low=Vec::with_capacity(rows*half_picks*width);
    let mut high=Vec::with_capacity(rows*half_picks*width);
    for r in 0..rows {
        for p in 0..half_picks {
            let id0=route[r*id_stride+p];
            let id1=shifted[r*id_stride+p];
            assert_eq!(id0,(r*100+p) as f32);
            assert_eq!(id1,(r*100+half_picks+p) as f32);
            for w in 0..width {low.push(id0*10.0+w as f32);high.push(id1*10.0+w as f32);}
        }
    }
    let expected:Vec<f32>=(0..rows).flat_map(|r|(0..picks).flat_map(move |p|(0..width).map(move |w|((r*100+p) as f32)*10.0+w as f32))).collect();
    assert_eq!(joined(&low,&high,rows,0),expected);
    assert_eq!(joined(&high,&low,rows,1),expected);
}
#[test]
fn expert_shard_slots_partition_picks_symmetrically_in_order() {
    let (rows,picks,experts)=(32usize,8usize,128usize);
    let mid=experts/2;
    let route:Vec<usize>=(0..rows*picks).map(|k|(k*37+11)%experts).collect();
    let partition=|half:usize|->(Vec<usize>,Vec<usize>) {
        let (es,ec)=if half==0 {(0,mid)} else {(mid,experts-mid)};
        let (ps,pc)=if half==0 {(mid,experts-mid)} else {(0,mid)};
        let (mut mine,mut peer)=(vec![Vec::new();ec],vec![Vec::new();pc]);
        for (pick,&e) in route.iter().enumerate() {
            if e>=es && e<es+ec {mine[e-es].push(pick);} else {peer[e-ps].push(pick);}
        }
        (mine.into_iter().flatten().collect(),peer.into_iter().flatten().collect())
    };
    let (m0,p0)=partition(0);
    let (m1,p1)=partition(1);
    // Half 0's `mine` matches Half 1's `peer` identically (and vice versa), covering `0..rows*picks` once.
    assert_eq!(m0,p1);
    assert_eq!(m1,p0);
    let mut seen=vec![false;rows*picks];
    for &s in m0.iter().chain(&p0) {assert!(!seen[s]);seen[s]=true;}
    assert!(seen.into_iter().all(|b|b));
}
