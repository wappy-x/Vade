//! The start of the text service on the two Sparks of a cabled pair (`flash-serve 0|1 2 LISTEN NEXT ... --pair`).
//!
//! Both Sparks connect before the load: the ring of the pair on LISTEN and NEXT (`transport::Ring`), the wire link of
//! the primary mailbox on ports + 2, the control wire of the mirror (`super::mirror`) on ports + 4, and (when the
//! cable's second RDMA rail is discovered via `second_rail`) a second mailbox wire link on ports + 6. Both take the
//! smaller of their two memory budgets (`Link::agree`), so the ledger admits the same chats on both. When `--mtp N`
//! (`N > 0`) is enabled, only half 0 loads and executes `Drafter`; half 1 follows mirrored target verify/accept calls.
use super::RingPair;
use crate::Result;
use std::{net::{SocketAddr,TcpStream},time::{Duration,Instant}};

/// How long a Spark waits for the other to open a link: both start at the same moment, and a link opens before the
/// load of the weights.
const LINK_WAIT:Duration=Duration::from_secs(120);

/// The links of one Spark of the pair.
#[cfg_attr(not(target_os="linux"),allow(dead_code))] // Only the device start, which is Linux code, reads the wires.
pub struct Link {pub(super) ring:RingPair,mailbox:TcpStream,mailbox2:Option<TcpStream>,control:TcpStream}
fn shifted(address:SocketAddr,by:u16)->Result<SocketAddr> {
    Ok(SocketAddr::new(address.ip(),address.port().checked_add(by).ok_or("tp2 pair: a link port exceeds its range")?))
}
/// One link between the two Sparks: half 1 listens on `listen`, half 0 dials `next`.
fn peer(half:usize,listen:SocketAddr,next:SocketAddr)->Result<TcpStream> {
    let (out,inc)=spark_seqpar::control::connect_peers((half>0).then_some(listen),(half==0).then_some(next),Instant::now()+LINK_WAIT)?;
    inc.or(out).ok_or_else(||"tp2 pair: a link to the other Spark is missing".to_owned())
}
#[cfg(target_os="linux")]
fn discover_second_rail(half:usize,ring:&RingPair,listen:SocketAddr,next:SocketAddr)->Result<Option<TcpStream>> {
    use crate::second_rail::{OFFER_BYTES,Offer,candidates,decode,encode,pick};
    use std::net::{IpAddr,SocketAddr};
    let path=|interface:&str|std::fs::read_link(format!("/sys/class/net/{interface}/device")).ok()
        .and_then(|d|d.file_name().and_then(|n|n.to_string_lossy().split(':').next().map(str::to_owned))).unwrap_or_default();
    let links=crate::discovery::links().unwrap_or_default().into_iter()
        .map(|l|crate::second_rail::Link {address:l.address,prefix:l.prefix,path:path(&l.interface),mbps:l.mbps}).collect::<Vec<_>>();
    let own=match listen.ip() {
        IpAddr::V4(primary)=>Offer::Open(candidates(&links,primary)),
        _=>Offer::Closed,
    };
    let mut both=[0u8;2*OFFER_BYTES];
    both[half*OFFER_BYTES..(half+1)*OFFER_BYTES].copy_from_slice(&encode(&own));
    ring.ring.borrow_mut().all_gather(&mut both,OFFER_BYTES)?;
    let peer_offer=decode(&both[(1-half)*OFFER_BYTES..(2-half)*OFFER_BYTES]);
    let Some((own_ip,peer_ip))=pick(half,&own,&peer_offer) else {
        eprintln!("tp2_pair_rail2 rank={half} rails=1 reason={}",crate::second_rail::why_not(&own,&peer_offer));
        return Ok(None);
    };
    let (p_listen,p_next)=(listen.port().checked_add(6),next.port().checked_add(6));
    let (Some(p_listen),Some(p_next))=(p_listen,p_next) else {return Ok(None);};
    let (l2,n2)=(SocketAddr::new(IpAddr::V4(own_ip),p_listen),SocketAddr::new(IpAddr::V4(peer_ip),p_next));
    let connected=peer(half,l2,n2);
    let mut ok=[0u8;2];
    ok[half]=u8::from(connected.is_ok());
    ring.ring.borrow_mut().all_gather(&mut ok,1)?;
    if ok[0]==1 && ok[1]==1 {
        eprintln!("tp2_pair_rail2 rank={half} rails=2 listen={l2} next={n2}");
        return Ok(connected.ok());
    }
    if let Err(err)=connected {eprintln!("tp2_pair_rail2 rank={half} rails=1 fallback={err}");}
    Ok(None)
}
impl Link {
    /// Connect the Spark of `half` (0 or 1) to the other: `listen` is this Spark's address on the cable, `next` the
    /// other's.
    pub fn connect(half:usize,listen:SocketAddr,next:SocketAddr)->Result<Self> {
        let ring=RingPair::new(crate::transport::Ring::connect(half,2,listen,next)?)?;
        let mailbox=peer(half,shifted(listen,2)?,shifted(next,2)?)?;
        let control=peer(half,shifted(listen,4)?,shifted(next,4)?)?;
        #[cfg(target_os="linux")]
        let mailbox2=discover_second_rail(half,&ring,listen,next)?;
        #[cfg(not(target_os="linux"))]
        let mailbox2=None;
        Ok(Self {ring,mailbox,mailbox2,control})
    }
    pub fn half(&self)->usize {self.ring.half}
    /// True on the leader half (half 0), which owns HTTP serving and optional local-only drafting.
    pub fn is_leader(&self)->bool {self.ring.half==0}
    /// Exchange one 64-bit word with the other Spark over the TCP control wire (`control` has no 30 s RDMA ring
    /// timeout, so asymmetric startup work such as rank-0 draft folding under `compute-sanitizer` never times out).
    fn exchange_u64(&self,value:u64)->Result<u64> {
        use std::io::{Read,Write};
        let mut s=&self.control;
        let _=s.set_nodelay(true);
        let _=s.set_read_timeout(None);
        s.write_all(&value.to_le_bytes()).map_err(|e|format!("tp2 pair control write: {e}"))?;
        let mut peer=[0u8;8];
        s.read_exact(&mut peer).map_err(|e|format!("tp2 pair control read: {e}"))?;
        Ok(u64::from_le_bytes(peer))
    }
    /// The smaller of `bytes` on this Spark and on the other.
    pub fn agree(&self,bytes:u64)->Result<u64> {Ok(bytes.min(self.exchange_u64(bytes)?))}
    /// The larger of `value` on this Spark and on the other.
    pub fn agree_max(&self,value:u64)->Result<u64> {Ok(value.max(self.exchange_u64(value)?))}
    /// Wait for the other Spark.
    pub fn ready(&self)->Result<()> {self.exchange_u64(0)?;Ok(())}
}

/// The options of a pair start that the pair cannot take: the disk cache (each Spark would read its own disk, which
/// the mirror does not follow). Drafting (`--mtp N`) is supported via rank-0-only `Drafter` + mirrored target verify.
pub fn check(_depth:usize,disk:bool)->Result<()> {
    if disk {return Err("--pair does not take --disk-cache-dir yet: each Spark would read its own disk".into());}
    Ok(())
}

#[cfg(target_os="linux")]
pub use device::serve;
#[cfg(target_os="linux")]
mod device {
    use super::{super::{MailboxPair,Split,mirror,part},Link};
    use crate::{Result,bounded_decoder::{Compute,Device},loop_decoder::{draft::Drafter,operands::weights::LoadedWeights,service::{Admitted,Service,limits,mtp,start::{Settings,Text}}},tensor_wire};
    use engine_model::plan::description::LoopTextDescription;
    use spark_gpu_probe::driver::{Backend,DeviceVector};
    use std::{cell::Cell,net::TcpListener,path::Path,sync::atomic::{AtomicBool,Ordering}};

    /// The freed device storage that the service keeps for reuse, as the one-Spark start keeps it
    /// (`service::start::spares`).
    fn spares(step:usize,chat:usize,chats:usize,pool:usize)->usize {step.saturating_add(chat.saturating_mul(chats.min(32)).min(pool))}

    /// Serve on the Spark of `link`: half 0 serves HTTP on `listener` through the mirror (with optional rank-0-only
    /// `draft` model), half 1 follows mirrored target calls without binding an HTTP socket.
    #[allow(clippy::too_many_arguments)]
    pub fn serve<'b,'d>(link:Link,backend:&Backend<'d>,compute:&Device<'b,'d>,weights:&LoadedWeights<'_,DeviceVector<'d>>,d:&LoopTextDescription,text:Text,admitted:&Admitted,settings:&Settings,listener:Option<TcpListener>,root:&Path,stop:&AtomicBool,draft:Option<&Drafter<'_,DeviceVector<'d>,<Device<'b,'d> as Compute>::Value>>)->Result<()> {
        let half=link.ring.half;
        let seats=admitted.seats()?;
        // Size the mailbox for the longest single cut trade: batched verify head (`128` rows when drafting, covering
        // C8 d<=7, C16 d<=7, C32 d<=3 in one RDMA put; wider batches chunk cleanly by rows), admitted chats, and
        // prompt expert/attention halves. Agree the maximum across both ranks so mailbox layouts match byte for byte.
        let rows=Service::new(compute,weights,d,settings.context,text.eos.clone(),seats)?.chats().max(1);
        let head=d.head_rows as usize;
        let head_rows=if draft.is_some() || settings.options.depth>0 {rows.max(128)} else {rows};
        let slot=part(head,true,0).len().max(part(head,true,1).len()).checked_mul(head_rows).ok_or("tp2 pair: the mailbox exceeds this host")?;
        let prompt=usize::try_from(d.hidden).map_err(|_|"tp2 pair: the width exceeds this host")?.checked_mul(seats.limits.step_rows.div_ceil(2)).ok_or("tp2 pair: the mailbox exceeds this host")?;
        let slot=usize::try_from(link.agree_max(slot.max(prompt) as u64)?).map_err(|_|"tp2 pair: the mailbox exceeds this host")?;
        link.ready()?;
        let Link {ring,mailbox:mailbox_link,mailbox2:mailbox2_link,control}=link;
        ring.ring.borrow_mut().barrier()?;
        let mailbox=backend.tp_mailbox(slot*4)?;
        let wire=tensor_wire::Wire::Write(tensor_wire::WriteLink::new(&mailbox_link)?);
        let worker=wire.bind(mailbox.take_host().ok_or("tp2 pair: the mailbox host is taken")?,half==0,mailbox.layout())?;
        let mailbox2=match &mailbox2_link {Some(_)=>Some(backend.tp_mailbox(slot*4)?),None=>None};
        let worker2=match (&mailbox2,mailbox2_link) {
            (Some(m2),Some(link2))=>{
                let w2=tensor_wire::Wire::Write(tensor_wire::WriteLink::new(&link2)?);
                Some(w2.bind(m2.take_host().ok_or("tp2 pair: the second mailbox host is taken")?,half==0,m2.layout())?)
            },
            _=>None,
        };
        let (flag,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
        let (attached2,failed2)=(AtomicBool::new(false),AtomicBool::new(false));
        std::thread::scope(|scope|->Result<()> {
            let thread=tensor_wire::spawn(scope,worker,&flag,&attached,&failed)?;
            let thread2=worker2.map(|w2|tensor_wire::spawn(scope,w2,&flag,&attached2,&failed2)).transpose()?;
            let ready=tensor_wire::wait_attached(&attached,&thread).and_then(|()|match &thread2 {
                Some(t2)=>tensor_wire::wait_attached(&attached2,t2),
                None=>Ok(()),
            });
            let served=ready.and_then(|()|{
                let pair=MailboxPair::with_aux(&mailbox,mailbox2.as_ref(),ring);
                let split=Split::new(weights,&pair);
                let model:Option<&dyn mtp::Model<Device<'b,'d>,Split<'_,'_,DeviceVector<'d>,MailboxPair<'_,'d>,<Device<'b,'d> as Compute>::Value>>>=draft.map(|m|m as &_);
                run(half,compute,&split,split.bypass(),d,text,admitted,settings,listener,root,stop,control,model)
            });
            flag.store(true,Ordering::Relaxed);
            let ended=thread.join().map_err(|_|"tp2 pair: the mailbox wire thread stopped".to_owned()).and_then(|ended|ended);
            let ended2=match thread2 {
                Some(t2)=>t2.join().map_err(|_|"tp2 pair: the second mailbox wire thread stopped".to_owned()).and_then(|e|e),
                None=>Ok(()),
            };
            served?;ended?;ended2?;
            if mailbox.failed() || mailbox2.as_ref().is_some_and(|m|m.failed()) {return Err("tp2 pair: the mailbox failed".into());}
            Ok(())
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn run<'b,'d,W:crate::loop_decoder::mix::Weights<Device<'b,'d>>>(
        half:usize,compute:&Device<'b,'d>,weights:&W,bypass:&Cell<bool>,d:&LoopTextDescription,text:Text,
        admitted:&Admitted,settings:&Settings,listener:Option<TcpListener>,root:&Path,stop:&AtomicBool,
        control:std::net::TcpStream,model:Option<&dyn mtp::Model<Device<'b,'d>,W>>,
    )->Result<()> {
        let Text {mut tokenizer,identity,eos}=text;
        let mut service=Service::new(compute,weights,d,settings.context,eos,admitted.seats()?)?.with_prompt_keep(!settings.options.no_prompt_keep);
        if half==0 && let Some(model)=model && settings.options.depth>0 {
            let cfg=model.config();
            if settings.options.depth>cfg.gamma() || !d.declared_draft.as_ref().is_some_and(|dd|dd.taps.iter().copied().eq(cfg.target_layer_ids.iter().map(|&n|n as u64))) {
                return Err("draft depth or taps differ from the checkpoint".into());
            }
            service.set_draft(model,settings.options.depth);
        }
        compute.step_spares(spares(service.step_bytes(),service.fixed_chat_bytes(),service.chats(),service.pool()));
        compute.fund_kv_pages(service.pool());
        let mut limits=limits(settings.context,settings.output)?;
        limits.next_mtp_depth=Some(service.depth());
        let mut wire=mirror::Wire::new(control)?;
        crate::startup::done();
        if half==1 {
            drop(listener);
            let template=spark_serve::protocol::parse(br#"{"prompt":"x","max_tokens":1,"temperature":0}"#,false,&limits)?;
            eprintln!("text_service pair follower ready rank=1 world=2 path=loop_driver_pair chats={} context={}",service.chats(),settings.context);
            return mirror::follow_service(&service,&mut wire,&template);
        }
        let listener=listener.ok_or("tp2 pair: rank 0 needs an HTTP listener")?;
        let auth=spark_serve::auth::ApiKey::load(settings.options.key.as_deref(),false)?;
        limits.api_key=auth;limits.metadata=Some(spark_serve::metadata::Metadata::loop_text_pair(&identity,"device"));limits.identity=Some(identity);
        limits.validate()?;
        tokenizer.set_limits(limits.tokenizer_limits()).map_err(|e|e.to_string())?;
        let (cache,config)=service.schedule();
        let name=root.file_name().ok_or("checkpoint needs a directory name")?.to_string_lossy().to_ascii_lowercase();
        let link=mirror::Link::new(wire);
        let engine=mirror::Mirror::with_draft(&service,&link,bypass,service.verify_row_cap_value());
        eprintln!("text_service qualification=qualified ready bind={} rank=0 world=2 path=loop_driver_pair backend=device context={} output={} session_pool_bytes={} max_chats={} mtp_depth={} cache={}",
            listener.local_addr().map_err(|e|e.to_string())?,settings.context,settings.output,service.pool(),service.chats(),service.depth(),config.prefix_cache);
        let served=spark_serve::scheduler::serve_requests(listener,&engine,Some(&tokenizer),&limits,&cache,&config,stop,&name,settings.requests);
        link.stop();
        served
    }
}
