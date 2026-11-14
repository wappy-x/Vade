use std::{collections::BTreeMap,env,fs};
fn times(path:&str)->BTreeMap<String,[f64;2]> {
    fs::read_to_string(path).unwrap().lines().filter(|l|l.starts_with("PRICE ")).map(|l| {
        let f=l.split_whitespace().filter_map(|x|x.split_once('=')).collect::<BTreeMap<_,_>>();
        (f["op"].to_string(),[f["full_ns"].parse().unwrap(),f["local_ns"].parse().unwrap()])
    }).collect()
}
fn main() {
    let a=env::args().collect::<Vec<_>>();let h=times(&a[1]);let c=times(&a[2]);
    let order=fs::read_to_string(&a[4]).unwrap();
    let kernel_rows=order.lines().filter(|l|!l.starts_with('#')).skip(1).map(|l|l.split('\t').collect::<Vec<_>>()).filter(|f|f[1]=="kernel").collect::<Vec<_>>();
    let min_take=|grid:&str|kernel_rows.iter().filter(|f|f[3]=="resident_launch_take_scatter" && f[4]==grid).map(|f|f[8].parse::<u64>().unwrap()).min().unwrap() as f64/1e6;
    let latent_put=kernel_rows.iter().filter(|f|f[3]=="resident_launch_put_join" && f[4]=="6x1x1").map(|f|f[8].parse::<u64>().unwrap()).sum::<u64>() as f64/1e6;
    let detail=fs::read_to_string(&a[3]).unwrap();let mut cell=String::new();let mut rows=BTreeMap::new();
    for line in detail.lines() {
        let f=line.split_whitespace().collect::<Vec<_>>();
        if line.contains(" job ") {cell=f[2].replace(',',"");continue;}
        if f.first().is_some_and(|n|n.starts_with("resident_launch_")) {
            rows.insert((cell.clone(),f[0].to_string(),f[1].to_string()),[f[4].parse::<f64>().unwrap(),f[5].parse::<f64>().unwrap()]);
        }
    }
    let r=|cell:&str,name:&str,grid:&str|rows[&(cell.to_string(),format!("resident_launch_{name}"),format!("grid={grid}"))];
    let weight=[63.,63.,31.,47.];let cells=["15","127","1040","4158"];
    println!("METHOD model_only; rc8 census rows scaled by synthetic full/owned kernel ratios; no real-pair gain measured");
    println!("COMPUTE Step baseline 1.120128 ms from ORDER c1040, used at each cell because the shape is fixed; Prepare/GatedNorm split from ORDER 0.340064/0.176832 and scaled to each DETAIL grid16 total; Conv uses each DETAIL grid192 total; quantizer saving credited zero");
    println!("TRANSPORT sender: grid8 -> grid2 and grid5 -> grid3 using per-call DETAIL means; receiver byte proxy from ORDER minima: ({}-{}) ms / ((12448-1168)*4) bytes; saved KDA payload=2232576 bytes; fixed-message and prefetch saving=0; wire/peer timing is not separately observed",min_take("1x6x1"),min_take("1x5x1"));
    println!("LATENT only q_b remains: 360448 payload bytes; put6 reference {latent_put:.6} ms from one ORDER step, new put2 from each DETAIL row; no compute or message-count credit");
    println!("SENSITIVITY adds a larger 0.476680 ms byte budget instead of the census transport proxy: 34*((22.63-13.77)+(22.02-16.86)) us from fuse-exchange/form2 at fixed 8us proxy delay. This removes MORE payload than proposed and is a local proxy, not a fabric bound.");
    let receive_per_byte=(min_take("1x6x1")-min_take("1x5x1"))/((12448.-1168.)*4.);
    let mut sums=[0.;5];println!("cell\tlatent_ms\ttransport_ms\tkda_split_ms\tkda_register_ms\tkda_split_sensitivity_ms\tkda_register_sensitivity_ms");
    for (index,cell) in cells.iter().enumerate() {
        let put2=r(cell,"put_join","2x1");let put3=r(cell,"put_join","3x1");let put5=r(cell,"put_join","5x1");let put8=r(cell,"put_join","8x1");
        let transport=put8[1]-34.*put2[1]/put2[0]+34.*(put5[1]/put5[0]-put3[1]/put3[0])+2232576.*receive_per_byte;
        let latent=latent_put-11.*put2[1]/put2[0]+360448.*receive_per_byte;
        let prep_gate=r(cell,"causal_fast","16x1")[1];let conv=r(cell,"causal_fast","192x1")[1];
        let bases=[("ConvParts",conv),("Prepare",prep_gate*0.340064/0.516896),("StepPlanar",1.120128),("GatedNorm",prep_gate*0.176832/0.516896)];
        let split=bases.iter().map(|(op,ms)|ms*(1.-h[*op][1]/h[*op][0])).sum::<f64>();
        let register=bases.iter().map(|(op,ms)|ms*(1.-c[*op][1]/h[*op][0])).sum::<f64>();
        let values=[latent,split+transport,register+transport,split+0.476680,register+0.476680];
        println!("c{cell}\t{latent:.6}\t{transport:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",values[1],values[2],values[3],values[4]);
        for (sum,value) in sums.iter_mut().zip(values) {*sum+=value*weight[index]/204.;}
    }
    println!("WEIGHTED 63/63/31/47 {}",sums.iter().map(|x|format!("{x:.6}")).collect::<Vec<_>>().join(" "));
    for op in ["ConvParts","Prepare","StepPlanar","GatedNorm"] {println!("RAW {op} H64={} H32={} C64={} C32={} ns_per_34_calls",h[op][0],h[op][1],c[op][0],c[op][1]);}
    let split_ns=h.values().map(|v|v[0]-v[1]).sum::<f64>();
    let candidate_ns=h.iter().map(|(op,v)|v[0]-c[op][1]).sum::<f64>();
    println!("SYNTHETIC_EVENT_SPANS split_saved_ms={:.6} candidate_saved_ms={:.6}; includes recording/submission gaps, not a whole-model gain",split_ns/1e6,candidate_ns/1e6);
    for (arm,path) in [("H",&a[1]),("C",&a[2])] {
        let input=fs::read_to_string(path).unwrap();let mut groups:BTreeMap<(String,u32,u32),Vec<u64>>=BTreeMap::new();
        for line in input.lines().filter(|l|l.starts_with("SAMPLE ")) {
            let f=line.split_whitespace().filter_map(|x|x.split_once('=')).collect::<BTreeMap<_,_>>();
            let trial=f["trial"].parse::<u32>().unwrap();if trial<4 {continue;}
            groups.entry((f["op"].to_string(),f["heads"].parse().unwrap(),trial%2)).or_default().push(f["gpu_ns"].parse().unwrap());
        }
        for ((op,heads,parity),mut samples) in groups {samples.sort_unstable();println!("ORDER_SENSITIVITY arm={arm} op={op} heads={heads} trial_parity={parity} median_ns={} min_ns={} max_ns={}",samples[samples.len()/2],samples[0],samples[samples.len()-1]);}
    }
}
