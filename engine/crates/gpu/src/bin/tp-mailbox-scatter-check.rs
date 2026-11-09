//! One-device direct gather versus the original rank-major copy path. No RDMA or model.
//! Both arms run on one CUDA device, with the peer's wire words delivered by the host.
#![forbid(unsafe_code)]
#[cfg(target_os="linux")]
mod check {
    use spark_gpu_probe::{driver::{Backend,DeviceVector},tp_mailbox_host::{TpMailboxHost,SEND_SEQ,SEND_LEN},tp_route::{Channels,Links},resident_launch::CopyCounts};
    use std::{sync::atomic::Ordering,time::Instant};
    type Result<T>=std::result::Result<T,String>;
    /// Payload written per step by the eight traced gather families (brief, COPY-OWNERS).
    const JOIN_BYTES:u64=19_347_584;const LOCAL_BYTES:u64=9_673_792;const FAMILY_LAUNCHES:u64=219;
    fn upload<'a>(b:&Backend<'a>,bits:&[u32])->Result<DeviceVector<'a>> {b.upload_vector(&bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>())}
    fn read(b:&Backend<'_>,v:&DeviceVector<'_>)->Result<Vec<u32>> {Ok(b.read_vector_values(v)?.into_iter().map(f32::to_bits).collect())}
    fn deliver(host:&TpMailboxHost<'_>,layout:Channels,origin:usize,rank:usize,seq:u64,bits:&[u32]) {
        let channel=layout.channel(origin,rank);let slot=seq as usize%layout.slots;let words=host.words();let start=layout.recv_slot(channel,slot)+layout.frame;
        for (word,&value) in words[start..start+bits.len()].iter().zip(bits) {word.store(value,Ordering::Relaxed);}
        words[layout.recv_seq(channel,slot)].store(seq as u32,Ordering::Release);
    }
    fn sent(host:&TpMailboxHost<'_>,layout:Channels,seq:u64,expected:&[u32])->Result<()> {
        let words=host.words();let slot=seq as usize%layout.slots;
        if words[SEND_SEQ+slot].load(Ordering::Acquire)!=seq as u32 || words[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=expected.len()*4 {return Err("publication header differs".into());}
        let start=layout.send_slot(slot)+layout.frame;
        if words[start..start+expected.len()].iter().map(|w|w.load(Ordering::Relaxed)).ne(expected.iter().copied()) {return Err("wire bytes differ".into());}Ok(())
    }
    fn add(total:&mut CopyCounts,delta:CopyCounts) {
        for i in 0..3 {total.launches[i]+=delta.launches[i];total.bytes[i]+=delta.bytes[i];}
        total.merged_zero_launches+=delta.merged_zero_launches;total.merged_zero_bytes+=delta.merged_zero_bytes;
    }
    fn copies(c:&CopyCounts)->(u64,u64) {(c.launches[0]+c.launches[1]-c.merged_zero_launches,c.bytes[0]+c.bytes[1]-c.merged_zero_bytes)}
    #[derive(Default)]
    struct Result2 {head:CopyCounts,candidate:CopyCounts,compared:usize,head_ms:f64,candidate_ms:f64}
    struct Case<'s> {sizes:&'s [usize],replicated:&'s [bool],parts:usize,rank:usize,repeats:usize,seed:usize,record:bool,wide:bool}
    /// Iteration 0 is warm-up (it may capture graphs): it is checked, not timed or counted.
    fn case(b:&Backend<'_>,c:&Case<'_>)->Result<Result2> {
        let (parts,rank)=(c.parts,c.rank);
        let stride=c.sizes.iter().sum::<usize>();let layout=Channels::new(&Links::new(parts,false)?,stride)?;
        let old=b.tp_mailbox_routed(layout)?;let new=b.tp_mailbox_routed(layout)?;
        let mut oh=old.take_host().ok_or("old host")?;let mut nh=new.take_host().ok_or("new host")?;
        // The production mailbox settings (next_tp_proxy), or the zeroed defaults.
        if c.wide {for host in [&mut oh,&mut nh] {host.set_take_pause(128);host.set_wide(true);host.set_take_ctas(6);}}
        // Raw bits: signed zero, a subnormal and finite extremes among ordinary values.
        let specials=[0x80000000u32,0x00000001,0x80000001,0x007fffff,0x7f7fffff,0xff7fffff];
        let sources=(0..parts).map(|origin|(0..stride).map(|i|if i%7==0 {specials[(i/7+origin)%specials.len()]} else {(((origin*17+i+c.seed)%97) as f32/16.0).to_bits()}).collect::<Vec<_>>()).collect::<Vec<_>>();
        let local=upload(b,&sources[rank])?;let mut mappings=vec![Vec::new();parts];let mut expected=Vec::new();let mut jobs=Vec::new();let mut from=0;
        for (&n,&replicated) in c.sizes.iter().zip(c.replicated) {let at=expected.len();
            for origin in 0..if replicated {1} else {parts} {mappings[origin].push([from,expected.len(),n]);expected.extend_from_slice(&sources[origin][from..from+n]);}
            jobs.push(at..expected.len());from+=n;}
        let mut out=Result2::default();
        for iteration in 0..=c.repeats {
            let seq=iteration as u64*2+1;
            for origin in 0..parts {if origin!=rank {deliver(&oh,layout,origin,rank,seq,&sources[origin]);deliver(&nh,layout,origin,rank,seq,&sources[origin]);}}
            // The two arms take turns going first. Both run the production shape.
            for candidate in [iteration%2==1,iteration%2==0] {
                b.synchronize()?;let queue=if c.record {Some(b.resident_queue()?)} else {None};if c.record {b.enable_resident_launches()?;}
                let before=b.resident_copy_counts();let started=Instant::now();
                let result:Vec<u32>=if candidate {
                    let mut joined=b.vector(expected.len())?;b.zero_vector(&mut joined)?;
                    if !mappings[rank].is_empty() {b.resident_scatter_copy(&local,&mut joined,&mappings[rank])?;}
                    b.tp_mailbox_put_finite(&new,&local,1,stride,stride,0,seq)?;
                    for origin in 0..parts {if origin!=rank {b.tp_mailbox_take_scatter(&new,&mut joined,&mappings[origin],layout.channel(origin,rank),seq,1_000_000_000)?;}}
                    let joined=joined.into_resident_shared()?;
                    let views=jobs.iter().map(|job|joined.resident_range(job.clone())).collect::<Result<Vec<_>>>()?;
                    // Consumers read after the parent handle is gone.
                    drop(joined);b.synchronize()?;let elapsed=started.elapsed().as_secs_f64()*1000.0;
                    if iteration>0 {out.candidate_ms+=elapsed;add(&mut out.candidate,b.resident_copy_counts().since(before));}
                    let mut all=Vec::with_capacity(expected.len());for view in &views {all.extend(read(b,view)?);}all
                } else {
                    let mut packed=b.vector(stride*parts)?;b.zero_vector(&mut packed)?;b.copy_range(&local,&mut packed,0,rank*stride,stride)?;
                    b.tp_mailbox_put_finite(&old,&packed,1,stride*parts,stride,rank*stride,seq)?;
                    for origin in 0..parts {if origin!=rank {b.tp_mailbox_take_from(&old,&mut packed,1,stride*parts,stride,origin*stride,layout.channel(origin,rank),seq,1_000_000_000)?;}}
                    let packed=packed.into_resident_shared()?;let mut pieces=Vec::new();let mut offset=0;
                    for (&n,&replicated) in c.sizes.iter().zip(c.replicated) {for origin in 0..if replicated {1} else {parts} {pieces.push(packed.resident_range(origin*stride+offset..origin*stride+offset+n)?);}offset+=n;}
                    let joined=b.resident_join(&pieces.iter().map(|v|(v,v.len())).collect::<Vec<_>>(),&mut None,0)?;
                    b.synchronize()?;let elapsed=started.elapsed().as_secs_f64()*1000.0;
                    if iteration>0 {out.head_ms+=elapsed;add(&mut out.head,b.resident_copy_counts().since(before));}
                    read(b,&joined)?
                };
                if result!=expected {return Err(format!("{} values differ: parts={parts} rank={rank} seed={} seq={seq}",if candidate {"candidate"} else {"head"},c.seed));}
                out.compared+=result.len()*4;
                sent(if candidate {&nh} else {&oh},layout,seq,&sources[rank])?;drop(queue);
            }
            // Guarded pass: the target is a view inside poisoned storage, at a 16-byte and a 4-byte offset.
            let seq=seq+1;let guard=if iteration%2==0 {4} else {1};
            for origin in 0..parts {if origin!=rank {deliver(&nh,layout,origin,rank,seq,&sources[origin]);}}
            b.synchronize()?;let queue=if c.record {Some(b.resident_queue()?)} else {None};if c.record {b.enable_resident_launches()?;}
            let output=upload(b,&vec![0xdeadbeef;expected.len()+2*guard])?.into_resident_shared()?;
            let mut body=output.resident_range(guard..expected.len()+guard)?;
            if !mappings[rank].is_empty() {b.resident_scatter_copy(&local,&mut body,&mappings[rank])?;}
            b.tp_mailbox_put_finite(&new,&local,1,stride,stride,0,seq)?;
            for origin in 0..parts {if origin!=rank {b.tp_mailbox_take_scatter(&new,&mut body,&mappings[origin],layout.channel(origin,rank),seq,1_000_000_000)?;}}
            b.synchronize()?;let all=read(b,&output)?;
            if all[..guard].iter().chain(&all[all.len()-guard..]).any(|&word|word!=0xdeadbeef) {return Err("scatter guard changed".into());}
            if all[guard..all.len()-guard]!=expected[..] || read(b,&local)?!=sources[rank] {return Err(format!("guarded scatter values or source differ: parts={parts} rank={rank} seed={}",c.seed));}
            out.compared+=all.len()*4;sent(&nh,layout,seq,&sources[rank])?;drop(queue);
        }
        new.require_success()?;old.require_success()?;Ok(out)
    }
    /// Bad rows are refused, a timed-out take writes nothing, a nonfinite row is not published.
    fn failures(b:&Backend<'_>)->Result<()> {
        let mut v=upload(b,&[0xdeadbeef;16])?;let source=upload(b,&[0u32;16])?;
        {let m=b.tp_mailbox(4096)?;
            for rows in [vec![[0,15,2]],vec![[0,0,4],[0,3,4]],vec![[usize::MAX,0,1]],vec![[0,0,0]],vec![]] {
                if b.resident_scatter_copy(&source,&mut v,&rows).is_ok() || b.tp_mailbox_take_scatter(&m,&mut v,&rows,0,1,1000).is_ok() {return Err("invalid scatter accepted".into());}
            }
                        if b.tp_mailbox_take_scatter(&m,&mut v,&[[1021,0,4]],0,1,1000).is_ok() {return Err("scatter source outside the slot accepted".into());}
        }
        for record in [false,true] {
            let m=b.tp_mailbox(4096)?;let queue=if record {Some(b.resident_queue()?)} else {None};if record {b.enable_resident_launches()?;}
            for channel in [1,usize::MAX] {if b.tp_mailbox_take_scatter(&m,&mut v,&[[0,0,4]],channel,1,1000).is_ok() {return Err("scatter channel outside the layout accepted".into());}}
            b.tp_mailbox_take_scatter(&m,&mut v,&[[0,4,8],[8,0,2]],0,99,1000)?;b.synchronize()?;
            if !m.failed() || read(b,&v)?!=vec![0xdeadbeef;16] {return Err("failed scatter changed destination".into());}drop(queue);
            let m=b.tp_mailbox(64)?;let host=m.take_host().ok_or("nonfinite host")?;
            let bad=upload(b,&[1.0f32.to_bits(),0x7f800000,2.0f32.to_bits(),3.0f32.to_bits()])?;
            let queue=if record {Some(b.resident_queue()?)} else {None};if record {b.enable_resident_launches()?;}
            b.tp_mailbox_put_finite(&m,&bad,1,4,4,0,1)?;b.synchronize()?;
            if !m.failed() || host.words()[SEND_SEQ+1%m.layout().slots].load(Ordering::Acquire)==1 {return Err("nonfinite local row was published".into());}drop(queue);
        }Ok(())
    }
    pub fn run()->Result<()> {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            b.configure_vector_pool(64<<20);failures(b)?;let mut compared=0;
            // Launches per step and row words of the eight gather families of the traced head.
            let families=[(34,vec![4096,4096,4096,64,64,32]),(34,vec![4096,4096]),(3,vec![6144,6144]),(42,vec![144,1024,1024]),
                (42,[vec![1024;16],vec![2048]].concat()),(42,vec![2048;8]),(11,vec![768,256,64,64,16]),(11,vec![8192,2048])];
            let (mut head_total,mut candidate_total)=(0.0,0.0);let mut rounds=0u64;
            for seed in [15,127,1040,4158] {for rank in 0..2 {
                let mut head=CopyCounts::default();let mut candidate=CopyCounts::default();let (mut hm,mut cm)=(0.0,0.0);
                for (n,sizes) in &families {
                    let r=case(b,&Case {sizes,replicated:&vec![false;sizes.len()],parts:2,rank,repeats:*n,seed,record:true,wide:true})?;
                    println!("FAMILY seed={seed} rank={rank} exchanges={n} rows={} head_us_per_exchange={:.3} candidate_us_per_exchange={:.3}",sizes.len(),r.head_ms*1000.0/ *n as f64,r.candidate_ms*1000.0/ *n as f64);
                    compared+=r.compared;hm+=r.head_ms;cm+=r.candidate_ms;add(&mut head,r.head);add(&mut candidate,r.candidate);
                }
                let (head_copies,head_bytes)=copies(&head);let (candidate_copies,candidate_bytes)=copies(&candidate);
                if (head_copies,candidate_copies)!=(2*FAMILY_LAUNCHES,FAMILY_LAUNCHES) {return Err(format!("copy launch counter differs: {head_copies} {candidate_copies}"));}
                if (head_bytes,candidate_bytes)!=(JOIN_BYTES+LOCAL_BYTES,LOCAL_BYTES) {return Err(format!("copy byte counter differs: {head_bytes} {candidate_bytes}"));}
                println!("COUNTS seed={seed} rank={rank} arm=head {}",head.line());println!("COUNTS seed={seed} rank={rank} arm=candidate {}",candidate.line());
                println!("PLACEMENT seed={seed} rank={rank} head_copies={head_copies} candidate_copies={candidate_copies} head_copy_bytes={head_bytes} candidate_copy_bytes={candidate_bytes} head_ms={hm:.6} candidate_ms={cm:.6}");
                head_total+=hm;candidate_total+=cm;rounds+=1;
            }}
            println!("TIMING rounds={rounds} exchanges_per_round={FAMILY_LAUNCHES} head_ms_per_round={:.6} candidate_ms_per_round={:.6} candidate_minus_head_us_per_exchange={:.3} scope=one_device_host_inclusive_with_a_wait_after_each_exchange",
                head_total/rounds as f64,candidate_total/rounds as f64,(candidate_total-head_total)*1000.0/(rounds*FAMILY_LAUNCHES) as f64);
            // Replicated rows, unequal lengths, 2 and 4 ranks, eager and recorded, both copy paths.
            for sizes in [vec![1,3,17,127],vec![16,64,128,4]] {for record in [false,true] {for wide in [false,true] {for parts in [2,4] {for rank in 0..parts {
                let r=case(b,&Case {sizes:&sizes,replicated:&[false,true,false,false],parts,rank,repeats:3,seed:19,record,wide})?;compared+=r.compared;
            }}}}}
            println!("SCATTER SCREEN PASS compared_bytes={compared} differing_bytes=0 wire_equal=true guards=true source_unchanged=true replicated_rows=true eager_and_recorded=true invalid_bounds=true timeout_no_write=true nonfinite_refused=true scope=one_device_no_model");Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os="linux")] if let Err(error)=check::run() {eprintln!("SCATTER SCREEN FAIL {error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))] {eprintln!("scatter screen requires Linux CUDA");std::process::exit(1);}
}
