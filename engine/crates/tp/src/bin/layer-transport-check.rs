//! Synthetic distributed byte gate for the ordinary pair mailbox wire.
//! No checkpoint payload, decoder arithmetic, or token-ID result is produced.
#[cfg(target_os="linux")]
mod gate {
    use engine_model::plan::{ConfigPlan, layer::{LayerPlan,Topology}};
    use spark_tp::{layer_transport::{PairPlan,PairGroup,DevicePair,WireType,PAIR_RANKS},tensor_wire,transport::Ring};
    use spark_gpu_probe::driver::with_backend;
    use std::{fs,io::Write,net::SocketAddr,path::Path,sync::atomic::{AtomicBool,Ordering},time::{Duration,Instant}};
    type Result<T>=std::result::Result<T,String>;
    fn need(ok: bool, why: &str) -> Result<()> { if ok {Ok(())} else {Err(why.into())} }
    fn word(at: usize, salt: u64) -> u32 {
        let bits=[0u32,0x80000000,0x7fc12345,0xff812345,1,0x3f800001,0x7f800000,0xff800000];
        if at<bits.len() {return bits[at];}
        bits[at%bits.len()] ^ (at as u32).wrapping_mul(0x10001) ^ (salt as u32).wrapping_mul(0x1010101)
    }
    fn salt(call: u64, step: usize, output: usize) -> u64 {
        u64::from((call as u32).wrapping_mul(0x9e3779b9) ^ (step as u32).rotate_left(11) ^ (output as u32).rotate_left(22))
    }
    pub fn run(args: &[String]) -> Result<()> {
        need(args.len()==7,"usage: layer-transport-check RANK CONTROL_LISTEN CONTROL_NEXT WIRE_LISTEN WIRE_NEXT CONFIG NEW_OUT")?;
        let rank=args[0].parse::<usize>().map_err(|_|"invalid pair rank")?;
        need(rank<PAIR_RANKS,"rank is outside the pair")?;
        let address=|i:usize| args[i].parse::<SocketAddr>().map_err(|_|"invalid transport address".to_string());
        let (listen,next,wire_listen,wire_next)=(address(1)?,address(2)?,address(3)?,address(4)?);
        let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&fs::read(&args[5]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?
            else {return Err("byte gate needs a described decoder program".into());};
        let out=Path::new(&args[6]); fs::create_dir(out).map_err(|e|e.to_string())?;
        fs::copy(&args[5],out.join("config.json")).map_err(|e|e.to_string())?;
        let mut receipt=fs::File::create(out.join("BYTES.txt")).map_err(|e|e.to_string())?;
        let mut plans=Vec::new();
        for context in [1,7,33] { for layer in 0..description.layers.len() { for dtype in [WireType::Bf16,WireType::F32] {
            let p=LayerPlan::new(&description,layer,context,Topology {parts:PAIR_RANKS as u32,rank:rank as u32}).map_err(|e|e.to_string())?;
            let count=p.steps.iter().map(|s|s.gather.len()).sum();
            plans.push(PairPlan::new(p,&vec![dtype;count],64<<20)?);
        } } }
        need(!plans.is_empty() && plans.iter().any(|p| !p.gathers().is_empty()),"byte gate selected no gathers")?;
        let slot_bytes=plans.iter().map(PairPlan::slot_bytes).max().ok_or("no layer plans")?;
        let ring=Ring::connect_options(rank,PAIR_RANKS,listen,next,true)?;
        let group=PairGroup::new(ring)?;
        let (incoming,outgoing)=spark_seqpar::control::connect_peers((rank>0).then_some(wire_listen),
            (rank+1<PAIR_RANKS).then_some(wire_next),Instant::now()+Duration::from_secs(300))?;
        let link=incoming.or(outgoing).ok_or("pair wire has no peer")?;
        let wire=tensor_wire::Wire::Write(tensor_wire::WriteLink::new(&link)?);
        with_backend(0,spark_gpu_probe::layer_ptx()?,256<<20,|backend| {
            let mailbox=backend.tp_mailbox(slot_bytes)?;
            let host=mailbox.take_host().ok_or("mailbox host endpoint was already taken")?;
            let worker=wire.bind(host,rank==0,mailbox.layout())?;
            let (stop,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
            std::thread::scope(|scope| {
                let handle=tensor_wire::spawn(scope,worker,&stop,&attached,&failed)?;
                let result=(|| {
                    tensor_wire::wait_attached(&attached,&handle)?;
                    let mut pair=DevicePair::new(group,backend,&mailbox,10_000_000_000)?;
                    need(mailbox.claim_copy_group().is_err(),"a live mailbox accepted another group")?;
                    let (mut checked,mut byte_count)=(0usize,0u64);
                    // Repeats use the same live wire. A slot wraps many times.
                    // Each process launch is one fresh station trial.
                    for repeat in 0..3 { for (index,plan) in plans.iter().enumerate() {
                        let call=(repeat*plans.len()+index+1) as u64;
                        let mut layer=pair.begin(plan,repeat as u64,plan.program().context-1)?;
                        for step in 0..plan.program().steps.len() {
                            let mut vectors=Vec::new();
                            for g in plan.step_gathers(step) {
                                let salt=salt(call,step,g.output());
                                let words=g.layout().bytes()/4;
                                let mut bytes=vec![0xa5;words*4];
                                let own=g.rows(rank)?;
                                for i in 0..own.packet_bytes()/4 {
                                    let at=own.destination(i)?; bytes[at*4..at*4+4].copy_from_slice(&word(at,salt).to_le_bytes());
                                }
                                let mut vector=backend.vector(words)?; backend.write_vector_bytes(&mut vector,&bytes)?; vectors.push(vector);
                            }
                            layer.after_step(step,&mut vectors.iter_mut().collect::<Vec<_>>())?;
                            for (g,vector) in plan.step_gathers(step).zip(&vectors) {
                                let salt=salt(call,step,g.output());
                                let got=backend.read_vector_values(vector)?;
                                let expected=(0..got.len()).map(|at|word(at,salt)).collect::<Vec<_>>();
                                let actual=got.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
                                if actual!=expected {
                                    let stem=format!("FAIL-{repeat}-{index}-{step}-{}",g.output());
                                    fs::write(out.join(format!("{stem}-actual.bytes")),actual.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).map_err(|e|e.to_string())?;
                                    fs::write(out.join(format!("{stem}-expected.bytes")),expected.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).map_err(|e|e.to_string())?;
                                    return Err(format!("{stem}: distributed tensor bytes differ"));
                                }
                                checked+=1; byte_count+=g.layout().bytes() as u64;
                                writeln!(receipt,"GATHER repeat={repeat} plan={index} layer={} context={} step={step} output={} dtype={:?} bytes={} identical=true",
                                    plan.program().layer,plan.program().context,g.name(),g.dtype(),g.layout().bytes()).map_err(|e|e.to_string())?;
                            }
                        }
                        layer.finish()?;
                    } }
                    need(checked>0 && pair.sequence()==checked as u64,"byte gate selected no messages or has inconsistent counts")?;
                    writeln!(receipt,"PAIR_BYTES rank={rank} plans={} repeats=3 gathers={checked} sequence={} bytes={byte_count} identical=true",plans.len(),pair.sequence()).map_err(|e|e.to_string())?;
                    receipt.sync_all().map_err(|e|e.to_string())?;
                    println!("PAIR_BYTES rank={rank} gathers={checked} bytes={byte_count} identical=true");
                    Ok(())
                })();
                if result.is_err() {mailbox.fail();}
                stop.store(true,Ordering::Release); handle.thread().unpark();
                let joined=handle.join().map_err(|_|"pair wire panicked".to_string())?;
                let drained=backend.synchronize(); result.and(joined).and(drained)
            })
        })?;
        fs::write(out.join("PASS"),"Synthetic pair byte gate passed. No model IDs were checked.\n").map_err(|e|e.to_string())
    }
}

fn main() {
    #[cfg(target_os="linux")]
    let result=gate::run(&std::env::args().skip(1).collect::<Vec<_>>());
    #[cfg(not(target_os="linux"))]
    let result: Result<(),String>=Err("the pair device byte gate needs Linux and CUDA".into());
    if let Err(error)=result {eprintln!("{error}");std::process::exit(1);}
}
