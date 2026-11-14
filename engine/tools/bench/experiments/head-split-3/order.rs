use std::{env,fs};
struct Row {seq:usize,name:String,grid:String,ns:u64}
fn main() {
    let path=env::args().nth(1).expect("ORDER TSV path");
    let source=fs::read_to_string(&path).unwrap();
    let rows=source.lines().filter(|line|!line.starts_with('#')).skip(1).filter_map(|line| {
        let f=line.split('\t').collect::<Vec<_>>();if f.get(1)!=Some(&"kernel") {return None;}
        Some(Row {seq:f[0].parse().unwrap(),name:f[3].to_owned(),grid:f[4].to_owned(),ns:f[8].parse().unwrap()})
    }).collect::<Vec<_>>();
    let mut sums=[0u64;9];let mut layers=0;
    println!("SOURCE {path}");
    println!("layer\tconv_seq\tconv_ns\tprepare_ns\tstep_ns\tgate_ns\tquantize_ns\tput1_ns\ttake1_ns\tput2_ns\ttake2_ns");
    for (at,row) in rows.iter().enumerate().filter(|(_,r)|r.name=="resident_launch_causal_fast" && r.grid=="192x1x1") {
        let later=rows[at+1..].iter().filter(|r|r.name=="resident_launch_causal_fast").take(4).collect::<Vec<_>>();
        assert_eq!(later.iter().map(|r|r.grid.as_str()).collect::<Vec<_>>(),["16x1x1","64x1x1","16x1x1","64x1x1"]);
        let before_prepare=rows.iter().position(|r|r.seq==later[0].seq).unwrap();
        let take2=rows[..before_prepare].iter().rposition(|r|r.name=="resident_launch_take_scatter").unwrap();
        assert_eq!(rows[at-1].name,"resident_launch_take_scatter");assert_eq!(rows[at-2].name,"resident_launch_put_join");
        assert_eq!(rows[take2-1].name,"resident_launch_put_join");
        let values=[row.ns,later[0].ns,later[1].ns,later[2].ns,later[3].ns,rows[at-2].ns,rows[at-1].ns,rows[take2-1].ns,rows[take2].ns];
        println!("{layers}\t{}\t{}",row.seq,values.iter().map(u64::to_string).collect::<Vec<_>>().join("\t"));
        for (sum,value) in sums.iter_mut().zip(values) {*sum+=value;}layers+=1;
    }
    assert_eq!(layers,34);println!("SUM_NS {}",sums.iter().map(u64::to_string).collect::<Vec<_>>().join(" "));
    println!("SUM_MS {}",sums.iter().map(|v|format!("{:.6}",*v as f64/1e6)).collect::<Vec<_>>().join(" "));
    let q_b_bytes=11*32*256*4;
    let kda_first=34*((3*4096+32+128)-128)*4;
    let kda_second=34*(8192-4096)*4;
    println!("BYTE_MODEL latent_query_saved_per_rank={q_b_bytes} kda_first_saved_per_rank={kda_first} kda_second_saved_per_rank={kda_second} kda_total_saved_per_rank={} kda_messages_before=102 kda_messages_after=102",kda_first+kda_second);
}
