//! RC SEND/RECV and one-sided writes into CUDA's pinned exchange slots.
//! ABI declarations follow the installed rdma-core verbs.h on aarch64 Linux.
use crate::{Result,discovery};
use engine_fabric::verbs::{Ah,Attr,Cap,Context,Gid,Init,Mr,Qp,Recv,Route,Send,Sge,Verbs,Wc};
use std::{ffi::{c_void,CStr},net::{Ipv6Addr,TcpStream},ops::Range,ptr,time::{Duration,Instant}};

#[link(name="cuda")]
unsafe extern "C" {fn cuPointerGetAttribute(data:*mut c_void,attribute:i32,pointer:u64)->i32;}
fn code(value:i32,what:&str)->Result<()> {if value==0{Ok(())}else{Err(format!("{what}: {}",std::io::Error::from_raw_os_error(value)))}}
fn present<T>(p:*mut T,what:&str)->Result<*mut T>{if p.is_null(){Err(format!("{what}: {}",std::io::Error::last_os_error()))}else{Ok(p)}}

struct Region {mr:*mut Mr,id:u64}
struct PeerRegion {base:usize,bytes:usize,address:u64,key:u32,local_key:u32}
pub struct Endpoint {verbs:&'static Verbs,context:*mut Context,pd:*mut c_void,cq:*mut c_void,qp:*mut Qp,gid:[u8;16],index:u8,mtu:i32,regions:Vec<Region>,peers:Vec<PeerRegion>,pending:usize,retain:bool,control:TcpStream,receive_sge:Vec<Sge>,send_sge:Vec<Sge>,receives:Vec<Recv>,sends:Vec<Send>}
// A queue pair has one owner; no verbs object is accessed concurrently.
unsafe impl std::marker::Send for Endpoint {}
impl Endpoint {
    pub fn open(socket:&TcpStream)->Result<Self>{
        let verbs=Verbs::load()?;
        let ip=socket.local_addr().map_err(|e|e.to_string())?.ip();
        let interface=discovery::links()?.into_iter().find(|l|std::net::IpAddr::V4(l.address)==ip).ok_or("RDMA socket has no QSFP interface")?.interface;
        let net_mtu=std::fs::read_to_string(format!("/sys/class/net/{interface}/mtu")).map_err(|e|e.to_string())?.trim().parse::<u32>().map_err(|e|e.to_string())?;
        let mtu=(1..=5).rev().find(|&m|(128u32<<m)+128<=net_mtu).ok_or("interface MTU too small for RoCE")?;
        let mut selected=None;
        for device in std::fs::read_dir("/sys/class/infiniband").map_err(|e|e.to_string())? {
            let device=device.map_err(|e|e.to_string())?;let port=device.path().join("ports/1");
            for index in 0..16u8 {
                let read=|p:String|std::fs::read_to_string(port.join(p)).unwrap_or_default();
                if read(format!("gid_attrs/ndevs/{index}")).trim()==interface&&read(format!("gid_attrs/types/{index}")).trim()=="RoCE v2" {
                    let gid=read(format!("gids/{index}")).trim().parse::<Ipv6Addr>().map_err(|e|e.to_string())?.octets();
                    if gid[10..12]==[255,255] {selected=Some((device.file_name().to_string_lossy().into_owned(),index,gid));break;}
                }
            }
            if selected.is_some(){break;}
        }
        let(name,index,gid)=selected.ok_or("no IPv4 RoCE-v2 GID on neighbour interface")?;
        let mut result=Self{verbs,context:ptr::null_mut(),pd:ptr::null_mut(),cq:ptr::null_mut(),qp:ptr::null_mut(),gid,index,mtu,regions:Vec::new(),peers:Vec::new(),pending:0,retain:false,control:socket.try_clone().map_err(|e|e.to_string())?,receive_sge:Vec::new(),send_sge:Vec::new(),receives:Vec::new(),sends:Vec::new()};
        unsafe {
            let mut count=0;let list=present((verbs.get_device_list)(&mut count),"verbs devices")?;
            for i in 0..count {let device=*list.add(i as usize);if CStr::from_ptr((verbs.get_device_name)(device)).to_bytes()==name.as_bytes(){result.context=(verbs.open_device)(device);break;}}
            (verbs.free_device_list)(list);present(result.context,"open neighbour HCA")?;
            result.pd=present((verbs.alloc_pd)(result.context),"allocate PD")?;
            result.cq=present((verbs.create_cq)(result.context,512,ptr::null_mut(),ptr::null_mut(),0),"create CQ")?;
            let mut init=Init{user:ptr::null_mut(),send_cq:result.cq,recv_cq:result.cq,srq:ptr::null_mut(),cap:Cap{send:256,recv:256,send_sge:16,recv_sge:16,inline:0},kind:2,signal_all:0};
            result.qp=present((verbs.create_qp)(result.pd,&mut init),"create RC QP")?;
            let mut attr=Attr{state:1,port:1,access:2,..Default::default()};
            code((verbs.modify_qp)(result.qp,&mut attr,(1<<0)|(1<<3)|(1<<4)|(1<<5)),"QP INIT")?;
        }
        Ok(result)
    }
    pub fn hello(&self)->[u8;24]{let mut data=[0;24];data[..16].copy_from_slice(&self.gid);data[16..20].copy_from_slice(&unsafe{(*self.qp).number}.to_le_bytes());data[20..].copy_from_slice(&self.mtu.to_le_bytes());data}
    pub fn ready(&mut self,peer:&[u8;24])->Result<()> {
        let mtu=self.mtu.min(i32::from_le_bytes(peer[20..].try_into().unwrap()));if !(1..=5).contains(&mtu){return Err("invalid peer RoCE MTU".into());}
        let mut attr=Attr{state:2,mtu,destination:u32::from_le_bytes(peer[16..20].try_into().unwrap()),dest_atomic:1,rnr_timer:1,..Default::default()};
        attr.ah=Ah{route:Route{gid:Gid(peer[..16].try_into().unwrap()),index:self.index,hops:1,..Default::default()},global:1,port:1,..Default::default()};
        unsafe {code((self.verbs.modify_qp)(self.qp,&mut attr,(1<<0)|(1<<7)|(1<<8)|(1<<12)|(1<<15)|(1<<17)|(1<<20)),"QP RTR")?;}
        let mut attr=Attr{state:3,timeout:14,retries:7,rnr_retries:7,atomic:1,..Default::default()};
        unsafe{code((self.verbs.modify_qp)(self.qp,&mut attr,(1<<0)|(1<<9)|(1<<10)|(1<<11)|(1<<13)|(1<<16)),"QP RTS")}
    }
    pub fn clear_regions(&mut self){assert_eq!(self.pending,0);self.peers.clear();for region in self.regions.drain(..){unsafe{(self.verbs.dereg_mr)(region.mr);}}}
    /// The caller fixes all host allocations for this span, then clears their
    /// registrations before those allocations can be reused or freed.
    pub fn retain_regions(&mut self,retain:bool){self.retain=retain;if !retain{self.clear_regions();}}
    /// The complete pinned host extent remains fixed until retain_regions(false).
    pub unsafe fn reserve_region(&mut self,base:*mut u8,bytes:usize)->Result<()> {
        if !self.retain{return Err("transfer region reservation needs a retained scope".into());}
        if bytes!=0{self.key(base,bytes)?;}Ok(())
    }
    fn release_transient(&mut self){if !self.retain{self.regions.retain(|region|{if region.id==0&&!self.peers.iter().any(|peer|unsafe{peer.local_key==(*region.mr).lkey}){unsafe{(self.verbs.dereg_mr)(region.mr);}false}else{true}});}}
    fn check_control(&self)->Result<()> {
        match self.control.peek(&mut [0u8;1]) {
            Ok(0)=>Err("RDMA peer closed control connection".into()),
            Ok(_)=>Ok(()),
            Err(e) if matches!(e.kind(),std::io::ErrorKind::WouldBlock|std::io::ErrorKind::Interrupted)=>Ok(()),
            Err(e)=>Err(format!("RDMA control connection: {e}")),
        }
    }
    fn key(&mut self,base:*mut u8,bytes:usize)->Result<u32>{
        if self.retain {
            for region in &self.regions {unsafe{let mr=region.mr;let start=(*mr).address as usize;
                if region.id==0&&base as usize>=start&&base as usize+bytes<=start+(*mr).length{return Ok((*mr).lkey);}
            }}
        }
        let mut id=0u64;unsafe{cuPointerGetAttribute((&mut id as *mut u64).cast(),7,base as u64);}
        for region in &self.regions {let mr=region.mr;unsafe{let start=(*mr).address as usize;if (id!=0&&id==region.id||self.retain&&id==0&&region.id==0)&&base as usize>=start&&(base as usize+bytes)<=start+(*mr).length{return Ok((*mr).lkey);}}}
        // The backend rotates slots within one pinned allocation. Register its
        // driver-reported extent once, rather than each changing slot address.
        let(mut address,mut length)=(base as usize,bytes);let(mut start,mut size)=(0usize,0usize);
        unsafe {if cuPointerGetAttribute((&mut start as *mut usize).cast(),11,base as u64)==0&&cuPointerGetAttribute((&mut size as *mut usize).cast(),12,base as u64)==0&&start<=address&&address+bytes<=start+size{address=start;length=size;}}
        let mr=unsafe{present((self.verbs.reg_mr)(self.pd,address as *mut c_void,length,3),"register pinned activation")?};
        self.regions.push(Region{mr,id});Ok(unsafe{(*mr).lkey})
    }
    /// Exchange the fixed mapped allocation once, before projection work begins.
    /// No control-channel traffic is needed by subsequent writes.
    pub unsafe fn prepare_write(&mut self,base:*mut u8,bytes:usize)->Result<()> {
        if self.peers.iter().any(|p|p.base==base as usize&&p.bytes==bytes){return Ok(());}
        let local_key=self.key(base,bytes)?;
        let mr=self.regions.iter().find(|r|unsafe{(*r.mr).lkey==local_key}).ok_or("missing write registration")?.mr;
        let mut hello=[0u8;16];hello[..8].copy_from_slice(&(base as u64).to_le_bytes());
        hello[8..12].copy_from_slice(&unsafe{(*mr).rkey}.to_le_bytes());
        hello[12..].copy_from_slice(&u32::try_from(bytes).map_err(|_|"write allocation too large")?.to_le_bytes());
        let mut peer=[0u8;16];crate::transport::exchange(&self.control,&self.control,&hello,&mut peer)?;
        if u32::from_le_bytes(peer[12..].try_into().unwrap()) as usize!=bytes{return Err("peer write allocation mismatch".into());}
        self.peers.push(PeerRegion{base:base as usize,bytes,address:u64::from_le_bytes(peer[..8].try_into().unwrap()),key:u32::from_le_bytes(peer[8..12].try_into().unwrap()),local_key});
        Ok(())
    }
    pub fn receive_write(&mut self)->Result<()> {unsafe{
        let mut wr=Recv{id:0,next:ptr::null_mut(),sg:ptr::null_mut(),count:0};let mut bad=ptr::null_mut();
        let post:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.context).ops[26]);
        code(post(self.qp,&mut wr,&mut bad),"post write completion")?;self.pending+=1;Ok(())
    }}
    /// An unsignalled prefix may be posted while the GPU finishes the tail.
    /// The final write generates one CQ completion at each endpoint.
    pub unsafe fn write(&mut self,base:*mut u8,bytes:usize,range:Range<usize>,destination:usize,last:bool)->Result<()> {
        let peer=self.peers.iter().find(|p|p.base==base as usize&&p.bytes==bytes).ok_or("unprepared mapped write")?;
        if range.end>bytes||destination+range.len()>bytes||range.len()>u32::MAX as usize{return Err("mapped write range".into());}
        let mut sg=Sge{address:unsafe{base.add(range.start)} as u64,length:range.len() as u32,key:peer.local_key};
        let mut wr=Send{id:0,next:ptr::null_mut(),sg:&mut sg,count:1,opcode:if last{1}else{0},flags:if last{2}else{0},imm:0,wr:[peer.address+destination as u64,peer.key as u64,0,0],qp_type:0,padding:0,tail:[0;6]};
        unsafe{let mut bad=ptr::null_mut();let post:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.context).ops[25]);code(post(self.qp,&mut wr,&mut bad),"post mapped write")?;}
        if last{self.pending+=1;}Ok(())
    }
    /// The caller retains the pinned allocation through completion and clears
    /// registrations before its staging pool can grow or be destroyed.
    pub unsafe fn post(&mut self,base:*mut u8,bytes:usize,stride:usize,tokens:usize,send:Range<usize>,recv:Range<usize>)->Result<()> {
        if tokens.div_ceil(16)>256{return Err("RDMA batch exceeds queue capacity".into());}
        let key=self.key(base,bytes)?;
        // Reductions send one contiguous rank block. Keep their receive and
        // send posting separate without allocating scatter/gather work lists.
        if tokens==1 {
            if send.len()>u32::MAX as usize||recv.len()>u32::MAX as usize{return Err("RDMA block exceeds message size".into());}
            unsafe {
                if !recv.is_empty(){
                    let mut sg=Sge{address:base.add(recv.start) as u64,length:recv.len() as u32,key};
                    let mut wr=Recv{id:(1<<63)|recv.len() as u64,next:ptr::null_mut(),sg:&mut sg,count:1};let mut bad=ptr::null_mut();
                    let post:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.context).ops[26]);
                    code(post(self.qp,&mut wr,&mut bad),"post receive")?;self.pending+=1;
                }
                if !send.is_empty(){
                    let mut sg=Sge{address:base.add(send.start) as u64,length:send.len() as u32,key};
                    let mut wr=Send{id:0,next:ptr::null_mut(),sg:&mut sg,count:1,opcode:2,flags:2,imm:0,wr:[0;4],qp_type:0,padding:0,tail:[0;6]};let mut bad=ptr::null_mut();
                    let post:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.context).ops[25]);
                    code(post(self.qp,&mut wr,&mut bad),"post send")?;self.pending+=1;
                }
            }
            return Ok(());
        }
        let (rs,ss,receives,sends)=(&mut self.receive_sge,&mut self.send_sge,&mut self.receives,&mut self.sends);
        rs.clear();ss.clear();receives.clear();sends.clear();
        for (list,range) in [(&mut *rs,&recv),(&mut *ss,&send)] {
            if range.len()>u32::MAX as usize{return Err("RDMA block exceeds message size".into());}
            if !range.is_empty(){list.extend((0..tokens).map(|t|Sge{address:unsafe{base.add(t*stride+range.start)} as u64,length:range.len() as u32,key}));}
        }
        receives.extend(rs.chunks_mut(16).map(|s|Recv{id:(1<<63)|s.iter().map(|v|v.length as u64).sum::<u64>(),next:ptr::null_mut(),sg:s.as_mut_ptr(),count:s.len() as i32}));
        sends.extend(ss.chunks_mut(16).map(|s|Send{id:0,next:ptr::null_mut(),sg:s.as_mut_ptr(),count:s.len() as i32,opcode:2,flags:2,imm:0,wr:[0;4],qp_type:0,padding:0,tail:[0;6]}));
        for i in 1..receives.len(){receives[i-1].next=&mut receives[i];}for i in 1..sends.len(){sends[i-1].next=&mut sends[i];}
        unsafe {
            // Inline verbs.h wrappers index these three context operations.
            let post_recv:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.context).ops[26]);
            let post_send:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.context).ops[25]);
            if !receives.is_empty(){let mut bad=ptr::null_mut();code(post_recv(self.qp,receives.as_mut_ptr(),&mut bad),"post receive")?;self.pending+=receives.len();}
            if !sends.is_empty(){let mut bad=ptr::null_mut();code(post_send(self.qp,sends.as_mut_ptr(),&mut bad),"post send")?;self.pending+=sends.len();}
        }
        Ok(())
    }
    pub fn finish(&mut self)->Result<()> {
        let start=Instant::now();let mut control=start;let mut wc=[Wc::default();16];let mut polls=0u32;
        let poll:unsafe extern "C" fn(*mut c_void,i32,*mut Wc)->i32=unsafe{std::mem::transmute((*self.context).ops[11])};
        while self.pending>0 {
            let n=unsafe{poll(self.cq,16,wc.as_mut_ptr())};if n<0{return Err("poll CQ failed".into());}
            for w in &wc[..n as usize] {
                if w.status!=0{return Err(format!("RDMA completion status={} vendor={}",w.status,w.vendor));}
                if w.id>>63!=0&&w.bytes as u64!=w.id&!(1<<63){return Err("RDMA receive length mismatch".into());}
                self.pending-=1;
            }
            polls=polls.wrapping_add(1);
            if polls&4095==0 {
                if n==0&&control.elapsed()>Duration::from_millis(100){self.check_control()?;control=Instant::now();}
                if start.elapsed()>Duration::from_secs(30){return Err("RDMA completion timed out".into());}
            }
            std::hint::spin_loop();
        }
        self.release_transient();
        Ok(())
    }
}
impl Drop for Endpoint {
    fn drop(&mut self){unsafe{
        if !self.qp.is_null(){(self.verbs.destroy_qp)(self.qp);self.pending=0;}
        self.clear_regions();if !self.cq.is_null(){(self.verbs.destroy_cq)(self.cq);}if !self.pd.is_null(){(self.verbs.dealloc_pd)(self.pd);}if !self.context.is_null(){(self.verbs.close_device)(self.context);}
    }}
}

/// Receives stay posted while the GPU computes. This avoids the receiver-not-
/// ready retries seen when a short activation arrives before its peer's GPU sync.
pub struct Mailbox {endpoint:Endpoint,data:Vec<f32>,mr:*mut Mr,ready:std::collections::VecDeque<(usize,usize)>}
unsafe impl std::marker::Send for Mailbox {}
impl Mailbox {
    const SLOT:usize=16*4096;
    pub fn supports(&self,elements:usize)->bool{elements<=Self::SLOT}
    pub fn prepare(socket:&TcpStream)->Result<Self>{
        let endpoint=Endpoint::open(socket)?;
        let mut data=vec![0f32;3*Self::SLOT];
        let mr=unsafe{present((endpoint.verbs.reg_mr)(endpoint.pd,data.as_mut_ptr().cast(),data.len()*4,1),"register receive mailbox")?};
        let mut result=Self{endpoint,data,mr,ready:std::collections::VecDeque::new()};result.receive(0)?;result.receive(1)?;Ok(result)
    }
    pub fn hello(&self)->[u8;24]{self.endpoint.hello()}
    pub fn ready(&mut self,peer:&[u8;24])->Result<()>{self.endpoint.ready(peer)}
    pub fn open(socket:&TcpStream)->Result<Self>{
        let mut result=Self::prepare(socket)?;let mut peer=[0;24];
        crate::transport::exchange(socket,socket,&result.hello(),&mut peer)?;result.ready(&peer)?;Ok(result)
    }
    fn receive(&mut self,slot:usize)->Result<()> {unsafe{
        let mut sg=Sge{address:self.data.as_mut_ptr().add(slot*Self::SLOT) as u64,length:(Self::SLOT*4) as u32,key:(*self.mr).lkey};
        let mut wr=Recv{id:slot as u64+1,next:ptr::null_mut(),sg:&mut sg,count:1};let mut bad=ptr::null_mut();
        let post:unsafe extern "C" fn(*mut Qp,*mut Recv,*mut *mut Recv)->i32=std::mem::transmute((*self.endpoint.context).ops[26]);
        code(post(self.endpoint.qp,&mut wr,&mut bad),"post mailbox receive")
    }}
    pub fn clear_registered(&mut self){self.endpoint.clear_regions();}
    /// The send slot stays registered across commands and small collectives.
    pub unsafe fn post(&mut self,values:&[u8])->Result<()> {
        if values.len()>Self::SLOT*4{return Err("send exceeds mailbox".into());}
        if self.endpoint.pending!=0{return Err("mailbox send still pending".into());}
        if values.is_empty(){return Ok(());}
        let send=unsafe{self.data.as_mut_ptr().add(2*Self::SLOT).cast::<u8>()};
        unsafe{ptr::copy_nonoverlapping(values.as_ptr(),send,values.len());}
        let mut sg=Sge{address:send as u64,length:values.len() as u32,key:unsafe{(*self.mr).lkey}};
        let mut wr=Send{id:0,next:ptr::null_mut(),sg:&mut sg,count:1,opcode:2,flags:2,imm:0,wr:[0;4],qp_type:0,padding:0,tail:[0;6]};
        unsafe{let mut bad=ptr::null_mut();let post:unsafe extern "C" fn(*mut Qp,*mut Send,*mut *mut Send)->i32=std::mem::transmute((*self.endpoint.context).ops[25]);code(post(self.endpoint.qp,&mut wr,&mut bad),"send mailbox activation")?;}
        self.endpoint.pending=1;Ok(())
    }
    fn complete(&mut self,bytes:usize,idle:bool,consume:impl FnOnce(&[u8]))->Result<()> {
        let start=Instant::now();let mut control=start;let mut wc=[Wc::default();4];
        while self.endpoint.pending!=0||(bytes!=0&&self.ready.is_empty()){
            let poll:unsafe extern "C" fn(*mut c_void,i32,*mut Wc)->i32=unsafe{std::mem::transmute((*self.endpoint.context).ops[11])};
            let n=unsafe{poll(self.endpoint.cq,4,wc.as_mut_ptr())};if n<0{return Err("mailbox CQ poll failed".into());}
            for w in &wc[..n as usize]{if w.status!=0{return Err(format!("mailbox completion status={} vendor={}",w.status,w.vendor));}
                if w.id==0{self.endpoint.pending=0;}else{self.ready.push_back((w.id as usize-1,w.bytes as usize));}
            }
            let elapsed=start.elapsed();
            if n==0&&control.elapsed()>Duration::from_millis(100){self.endpoint.check_control()?;control=Instant::now();}
            if !idle && elapsed > crate::transport::COMMAND_TIMEOUT { return Err("command receive deadline expired".into()); }
            if idle&&n==0 {
                // Park between requests without putting TCP in token commands.
                if elapsed>Duration::from_millis(1){std::thread::sleep(Duration::from_micros(100));}
            }else if !idle&&elapsed>Duration::from_secs(30){return Err("mailbox timed out".into());}
            std::hint::spin_loop();
        }
        self.endpoint.release_transient();
        if bytes==0{return Ok(());}
        let(slot,length)=self.ready.pop_front().unwrap();
        if length!=bytes||slot>1{return Err("mailbox receive shape mismatch".into());}
        consume(unsafe{std::slice::from_raw_parts(self.data.as_ptr().add(slot*Self::SLOT).cast(),length)});
        self.receive(slot)
    }
    pub fn finish(&mut self,output:&mut [u8])->Result<()>{self.complete(output.len(),false,|bytes|output.copy_from_slice(bytes))}
    pub fn finish_command(&mut self,output:&mut [u8])->Result<()>{self.complete(output.len(),true,|bytes|output.copy_from_slice(bytes))}
    pub fn reduce(&mut self,values:&mut [f32])->Result<()> {
        let bytes=std::mem::size_of_val(values);
        unsafe{self.post(std::slice::from_raw_parts(values.as_ptr().cast(),bytes))?;}
        self.complete(bytes,false,|data|{
            let peer=unsafe{std::slice::from_raw_parts(data.as_ptr().cast::<f32>(),data.len()/4)};
            for(a,b)in values.iter_mut().zip(peer){*a+=*b;}
        })
    }
}
impl Drop for Mailbox {fn drop(&mut self){unsafe{
    if !self.endpoint.qp.is_null(){(self.endpoint.verbs.destroy_qp)(self.endpoint.qp);self.endpoint.qp=ptr::null_mut();self.endpoint.pending=0;}
    (self.endpoint.verbs.dereg_mr)(self.mr);
}}}
