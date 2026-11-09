//! RC SEND/RECV for host K/V buffers, adapted from Vade scale transport.
//! Reusable host registrations share a protection domain across request queue pairs.
//! ABI declarations follow the installed rdma-core verbs.h on aarch64 Linux.
use super::Result;
use crate::discovery;
use crate::verbs::{Ah,Attr,Cap,Context,Gid,Init,Mr,Qp,Recv,Route,Send,Sge,Verbs,Wc};
use std::{ffi::{c_void,CStr},net::{Ipv6Addr,TcpStream},ops::Range,ptr,time::{Duration,Instant}};

fn code(value:i32,what:&str)->Result<()> {if value==0{Ok(())}else{Err(format!("{what}: {}",std::io::Error::from_raw_os_error(value)))}}
fn present<T>(p:*mut T,what:&str)->Result<*mut T>{if p.is_null(){Err(format!("{what}: {}",std::io::Error::last_os_error()))}else{Ok(p)}}

use std::sync::{Arc, Mutex, Weak};
struct Domain { name:String, context:*mut Context, pd:*mut c_void, verbs:&'static Verbs }
// rdma-core permits concurrent queue pairs and registration in one domain.
unsafe impl std::marker::Send for Domain {}
unsafe impl Sync for Domain {}
static DOMAINS:Mutex<Vec<Weak<Domain>>>=Mutex::new(Vec::new());
impl Domain {
    fn open(name:&str,verbs:&'static Verbs)->Result<Arc<Self>> {
        let mut domains=DOMAINS.lock().map_err(|_|"RDMA domain pool poisoned")?;
        domains.retain(|d|d.strong_count()!=0);
        if let Some(domain)=domains.iter().filter_map(Weak::upgrade).find(|d|d.name==name){return Ok(domain);}
        let mut domain=Self{name:name.into(),context:ptr::null_mut(),pd:ptr::null_mut(),verbs};
        unsafe {
            let mut count=0;let list=present((verbs.get_device_list)(&mut count),"verbs devices")?;
            for i in 0..count {let device=*list.add(i as usize);if CStr::from_ptr((verbs.get_device_name)(device)).to_bytes()==name.as_bytes(){domain.context=(verbs.open_device)(device);break;}}
            (verbs.free_device_list)(list);present(domain.context,"open neighbour HCA")?;
            domain.pd=present((verbs.alloc_pd)(domain.context),"allocate PD")?;
        }
        let domain=Arc::new(domain);domains.push(Arc::downgrade(&domain));Ok(domain)
    }
}
#[allow(clippy::possible_missing_else)] // Dense one-line guards.
impl Drop for Domain {fn drop(&mut self){unsafe{
    if !self.pd.is_null(){(self.verbs.dealloc_pd)(self.pd);}if !self.context.is_null(){(self.verbs.close_device)(self.context);}
}}}
pub(super) struct Region {mr:*mut Mr,domain:Arc<Domain>}
unsafe impl std::marker::Send for Region {}
unsafe impl Sync for Region {}
impl Drop for Region {fn drop(&mut self){unsafe{(self.domain.verbs.dereg_mr)(self.mr);}}}
pub struct Endpoint {context:*mut Context,pd:*mut c_void,cq:*mut c_void,qp:*mut Qp,gid:[u8;16],index:u8,mtu:i32,regions:Vec<Arc<Region>>,registrations_sealed:bool,pending:usize,domain:Arc<Domain>}
// A queue pair has one owner. Moving it to a transport worker is permitted;
// no verbs object or completion queue is accessed concurrently.
unsafe impl std::marker::Send for Endpoint {}
impl Endpoint {
    pub fn open(socket:&TcpStream)->Result<Self>{
        let std::net::IpAddr::V4(ip)=socket.local_addr().map_err(|e|e.to_string())?.ip() else{return Err("RDMA requires an IPv4 QSFP socket".into());};
        Self::open_address(ip)
    }
    pub fn open_address(ip:std::net::Ipv4Addr)->Result<Self>{
        let verbs=Verbs::load()?;
        let interface=discovery::links()?.into_iter().find(|l|l.address==ip).ok_or("RDMA socket has no QSFP interface")?.interface;
        let net_mtu=std::fs::read_to_string(format!("/sys/class/net/{interface}/mtu")).map_err(|e|e.to_string())?.trim().parse::<u32>().map_err(|e|e.to_string())?;
        let mtu=(1..=5).rev().find(|&m|(128u32<<m)+128<=net_mtu).ok_or("interface MTU too small for RoCE")?;
        let mut selected=None;
        for device in std::fs::read_dir("/sys/class/infiniband").map_err(|e|e.to_string())? {
            let device=device.map_err(|e|e.to_string())?;let port=device.path().join("ports/1");
            for entry in std::fs::read_dir(port.join("gids")).map_err(|e|e.to_string())? {
                let entry=entry.map_err(|e|e.to_string())?;
                let Ok(index)=entry.file_name().to_string_lossy().parse::<u8>() else{continue;};
                let read=|p:String|std::fs::read_to_string(port.join(p)).unwrap_or_default();
                if read(format!("gid_attrs/ndevs/{index}")).trim()==interface&&read(format!("gid_attrs/types/{index}")).trim()=="RoCE v2" {
                    let gid=read(format!("gids/{index}")).trim().parse::<Ipv6Addr>().map_err(|e|e.to_string())?.octets();
                    if gid[10..12]==[255,255]&&gid[12..]==ip.octets() {selected=Some((device.file_name().to_string_lossy().into_owned(),index,gid));break;}
                }
            }
            if selected.is_some(){break;}
        }
        let(name,index,gid)=selected.ok_or("no IPv4 RoCE-v2 GID on neighbour interface")?;
        let domain=Domain::open(&name,verbs)?;
        let mut result=Self{context:domain.context,pd:domain.pd,cq:ptr::null_mut(),qp:ptr::null_mut(),gid,index,mtu,regions:Vec::new(),registrations_sealed:false,pending:0,domain};
        unsafe {
            result.cq=present((verbs.create_cq)(result.context,512,ptr::null_mut(),ptr::null_mut(),0),"create CQ")?;
            let mut init=Init{user:ptr::null_mut(),send_cq:result.cq,recv_cq:result.cq,srq:ptr::null_mut(),cap:Cap{send:256,recv:256,send_sge:16,recv_sge:16,inline:0},kind:2,signal_all:0};
            result.qp=present((verbs.create_qp)(result.pd,&mut init),"create RC QP")?;
            let mut attr=Attr{state:1,port:1,..Default::default()};
            code((verbs.modify_qp)(result.qp,&mut attr,(1<<0)|(1<<3)|(1<<4)|(1<<5)),"QP INIT")?;
        }
        Ok(result)
    }
    pub fn hello(&self)->[u8;24]{let mut data=[0;24];data[..16].copy_from_slice(&self.gid);data[16..20].copy_from_slice(&unsafe{(*self.qp).number}.to_le_bytes());data[20..].copy_from_slice(&self.mtu.to_le_bytes());data}
    pub fn ready(&mut self,peer:&[u8;24])->Result<()> {
        let mtu=self.mtu.min(i32::from_le_bytes(peer[20..].try_into().unwrap()));if !(1..=5).contains(&mtu){return Err("invalid peer RoCE MTU".into());}
        let mut attr=Attr{state:2,mtu,destination:u32::from_le_bytes(peer[16..20].try_into().unwrap()),dest_atomic:1,rnr_timer:1,..Default::default()};
        attr.ah=Ah{route:Route{gid:Gid(peer[..16].try_into().unwrap()),index:self.index,hops:1,..Default::default()},global:1,port:1,..Default::default()};
        unsafe {code((self.domain.verbs.modify_qp)(self.qp,&mut attr,(1<<0)|(1<<7)|(1<<8)|(1<<12)|(1<<15)|(1<<17)|(1<<20)),"QP RTR")?;}
        let mut attr=Attr{state:3,timeout:14,retries:7,rnr_retries:7,atomic:1,..Default::default()};
        unsafe{code((self.domain.verbs.modify_qp)(self.qp,&mut attr,(1<<0)|(1<<9)|(1<<10)|(1<<11)|(1<<13)|(1<<16)),"QP RTS")}
    }
    pub fn clear_regions(&mut self){assert_eq!(self.pending,0);self.regions.clear();}
    pub fn seal_registrations(&mut self){self.registrations_sealed=true;}
    fn key(&mut self,base:*mut u8,bytes:usize)->Result<u32>{
        for region in &self.regions {
            let mr=region.mr;
            unsafe {let start=(*mr).address as usize;
                if base as usize>=start&&(base as usize+bytes)<=start+(*mr).length {return Ok((*mr).lkey);}
            }
        }
        if self.registrations_sealed {
            return Err(format!("FAIL RDMA registration attempted after sealing: bytes={bytes}; registrations sealed before weights"));
        }
        let mr=unsafe{present((self.domain.verbs.reg_mr)(self.pd,base.cast(),bytes,1),"register host K/V")?};
        self.regions.push(Arc::new(Region{mr,domain:self.domain.clone()}));Ok(unsafe{(*mr).lkey})
    }
    pub(super) unsafe fn register_cached(&mut self,base:*mut u8,bytes:usize,cache:&mut Vec<Arc<Region>>)->Result<()> {
        if bytes==0{return Ok(());}
        for region in cache.iter().filter(|r|Arc::ptr_eq(&r.domain,&self.domain)) {
            unsafe {let start=(*region.mr).address as usize;
                if base as usize>=start&&base as usize+bytes<=start+(*region.mr).length {
                    if !self.regions.iter().any(|r|Arc::ptr_eq(r,region)){self.regions.push(region.clone());}
                    return Ok(());
                }
            }
        }
        let key=self.key(base,bytes)?;
        cache.push(self.regions.iter().find(|r|unsafe{(*r.mr).lkey==key}).unwrap().clone());Ok(())
    }
    #[allow(clippy::missing_safety_doc)] // the safety contract is still to be written (RESULT-hygiene follow-up).
    pub unsafe fn register(&mut self,base:*mut u8,bytes:usize)->Result<()> {
        if bytes!=0 { self.key(base,bytes)?; }
        Ok(())
    }
    /// The caller retains the host allocation until completion and clears
    /// registrations before returning its borrow to the allocation owner.
    #[allow(clippy::missing_safety_doc)] // the safety contract is still to be written (RESULT-hygiene follow-up).
    pub unsafe fn post(&mut self,base:*mut u8,bytes:usize,send:Range<usize>,recv:Range<usize>)->Result<()> {
        if send.len()>u32::MAX as usize||recv.len()>u32::MAX as usize{return Err("RDMA range exceeds 4 GiB".into());}
        if send.is_empty()&&recv.is_empty(){return Ok(());}
        if send.end>bytes||recv.end>bytes{return Err("RDMA range exceeds staging allocation".into());}
        let region=if send.is_empty(){recv.clone()}else if recv.is_empty(){send.clone()}else{send.start.min(recv.start)..send.end.max(recv.end)};
        let key=self.key(unsafe{base.add(region.start)},region.len())?;
        unsafe {
            // Inline verbs.h wrappers index these three context operations.
            let post_recv:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.context).ops[26]);
            let post_send:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.context).ops[25]);
            // Keep both K/V streams in flight without another payload copy.
            // Two buffers use at most 128 entries per send/receive queue.
            const BATCH:usize=64;
            let mut sg=[Sge::default();BATCH];
            if !recv.is_empty(){
                let chunk=recv.len().div_ceil(BATCH).max(1024*1024);
                let count=recv.len().div_ceil(chunk);
                let mut wr:[Recv;BATCH]=std::array::from_fn(|_|Recv{id:0,next:ptr::null_mut(),sg:ptr::null_mut(),count:1});
                for i in 0..count {
                    let offset=recv.start+i*chunk;let length=(recv.end-offset).min(chunk);
                    sg[i]=Sge{address:base.add(offset) as u64,length:length as u32,key};
                    wr[i].id=(1<<63)|length as u64;wr[i].sg=&mut sg[i];
                    if i+1<count{wr[i].next=&mut wr[i+1];}
                }
                let mut bad=ptr::null_mut();code(post_recv(self.qp,wr.as_mut_ptr(),&mut bad),"post receive")?;self.pending+=count;
            }
            if !send.is_empty(){
                let chunk=send.len().div_ceil(BATCH).max(1024*1024);
                let count=send.len().div_ceil(chunk);
                let mut wr:[Send;BATCH]=std::array::from_fn(|_|Send{id:0,next:ptr::null_mut(),sg:ptr::null_mut(),count:1,opcode:2,flags:0,imm:0,wr:[0;4],qp_type:0,padding:0,tail:[0;6]});
                for i in 0..count {
                    let offset=send.start+i*chunk;let length=(send.end-offset).min(chunk);
                    sg[i]=Sge{address:base.add(offset) as u64,length:length as u32,key};wr[i].sg=&mut sg[i];
                    if i+1<count{wr[i].next=&mut wr[i+1];}
                }
                // RC completion of the last WR covers the preceding sends.
                wr[count-1].flags=2;
                let mut bad=ptr::null_mut();code(post_send(self.qp,wr.as_mut_ptr(),&mut bad),"post send")?;self.pending+=1;
            }
        }
        Ok(())
    }
    /// Mailbox receives stay posted between collectives and are not counted in
    /// `pending`; ids are slot+1. The caller registered `base` beforehand.
    #[allow(clippy::missing_safety_doc)] // the safety contract is still to be written (RESULT-hygiene follow-up).
    pub unsafe fn post_slot_receive(&mut self,base:*mut u8,bytes:usize,slot:usize)->Result<()> {
        let key=self.key(base,bytes)?;
        let mut sg=Sge{address:base as u64,length:bytes as u32,key};
        let mut wr=Recv{id:slot as u64+1,next:ptr::null_mut(),sg:&mut sg,count:1};
        unsafe{let post:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.context).ops[26]);
            let mut bad=ptr::null_mut();code(post(self.qp,&mut wr,&mut bad),"post mailbox receive")}
    }
    /// One signaled send from the mailbox send slot (completion id 0).
    #[allow(clippy::missing_safety_doc)] // the safety contract is still to be written (RESULT-hygiene follow-up).
    pub unsafe fn post_slot_send(&mut self,base:*mut u8,bytes:usize)->Result<()> {
        let key=self.key(base,bytes)?;
        let mut sg=Sge{address:base as u64,length:bytes as u32,key};
        let mut wr=Send{id:0,next:ptr::null_mut(),sg:&mut sg,count:1,opcode:2,flags:2,imm:0,wr:[0;4],qp_type:0,padding:0,tail:[0;6]};
        unsafe{let post:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.context).ops[25]);
            let mut bad=ptr::null_mut();code(post(self.qp,&mut wr,&mut bad),"post mailbox send")}
    }
    /// Alternate fixed stripes directly inside a contiguous mailbox slot.
    /// Both rails send a message, including an empty second stripe, so their
    /// two pre-posted receive slots always refer to the same logical message.
    /// `receive` posts capacity; a send supplies only the actual payload.
    #[allow(clippy::missing_safety_doc)] // the safety contract is still to be written (RESULT-hygiene follow-up).
    pub unsafe fn post_slot_stripes(&mut self, base:*mut u8, bytes:usize,
        stripe:usize, rail:usize, receive:Option<usize>) -> Result<()> {
        if stripe == 0 || rail > 1 || bytes.div_ceil(stripe) > 32 {
            return Err("invalid mailbox stripe shape".into());
        }
        let key = self.key(base, bytes)?;
        let mut sg = [Sge::default();16];
        let mut count = 0;
        for offset in (rail * stripe..bytes).step_by(2 * stripe) {
            sg[count] = Sge { address: unsafe { base.add(offset) } as u64,
                length: (bytes - offset).min(stripe) as u32, key };
            count += 1;
        }
        unsafe {
            if let Some(index) = receive {
                let mut wr = Recv { id:index as u64+1, next:ptr::null_mut(), sg:sg.as_mut_ptr(), count:count as i32 };
                let post:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.context).ops[26]);
                let mut bad=ptr::null_mut();code(post(self.qp,&mut wr,&mut bad),"post striped mailbox receive")
            } else {
                let mut wr=Send { id:0,next:ptr::null_mut(),sg:sg.as_mut_ptr(),count:count as i32,
                    opcode:2,flags:2,imm:0,wr:[0;4],qp_type:0,padding:0,tail:[0;6] };
                let post:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.context).ops[25]);
                let mut bad=ptr::null_mut();code(post(self.qp,&mut wr,&mut bad),"post striped mailbox send")
            }
        }
    }
    /// Poll once: count send completions and queue (slot, bytes) arrivals.
    /// Completions come from ibv_poll_cq, whose provider orders the payload reads.
    pub fn poll_slots(&mut self,sends:&mut usize,arrived:&mut std::collections::VecDeque<(usize,usize)>)->Result<()> {
        let mut wc=[Wc::default();16];
        let poll:unsafe extern "C" fn(*mut c_void,i32,*mut Wc)->i32=unsafe{std::mem::transmute((*self.context).ops[11])};
        let n=unsafe{poll(self.cq,16,wc.as_mut_ptr())};if n<0{return Err("poll mailbox CQ failed".into());}
        for w in &wc[..n as usize] {
            if w.status!=0{return Err(format!("RDMA mailbox completion status={} vendor={}",w.status,w.vendor));}
            if w.id==0 { *sends=sends.checked_sub(1).ok_or("unexpected mailbox send completion")?; }
            else { arrived.push_back((w.id as usize-1,w.bytes as usize)); }
        }
        Ok(())
    }
    pub fn finish(&mut self)->Result<()> {
        let start=Instant::now();let mut wc=[Wc::default();16];
        let poll:unsafe extern "C" fn(*mut c_void,i32,*mut Wc)->i32=unsafe{std::mem::transmute((*self.context).ops[11])};
        while self.pending>0 {
            let n=unsafe{poll(self.cq,16,wc.as_mut_ptr())};if n<0{return Err("poll CQ failed".into());}
            for w in &wc[..n as usize] {
                if w.status!=0{return Err(format!("RDMA completion status={} vendor={}",w.status,w.vendor));}
                if w.id>>63!=0&&w.bytes as u64!=w.id&!(1<<63){return Err("RDMA receive length mismatch".into());}
                self.pending-=1;
            }
            if start.elapsed()>Duration::from_secs(30){return Err("RDMA completion timed out".into());}std::hint::spin_loop();
        }
        Ok(())
    }
}
impl Drop for Endpoint {
    fn drop(&mut self){unsafe{
        if !self.qp.is_null(){(self.domain.verbs.destroy_qp)(self.qp);self.pending=0;}
        self.clear_regions();if !self.cq.is_null(){(self.domain.verbs.destroy_cq)(self.cq);}
    }}
}
