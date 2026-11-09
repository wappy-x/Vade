//! Shared rdma-core ABI and lazy runtime loading for the Spark transports.
use crate::Result;
use std::{ffi::{c_char, c_void, CStr}, sync::OnceLock};

#[repr(C)] pub struct Context { pub device:*mut c_void, pub ops:[*const c_void;32] }
#[repr(C)] pub struct Mr { pub context:*mut Context,pub pd:*mut c_void,pub address:*mut c_void,pub length:usize,pub handle:u32,pub lkey:u32,pub rkey:u32 }
#[repr(C)] pub struct Qp { pub context:*mut Context,pub user:*mut c_void,pub pd:*mut c_void,pub send_cq:*mut c_void,pub recv_cq:*mut c_void,pub srq:*mut c_void,pub handle:u32,pub number:u32 }
#[repr(C)] #[derive(Default)] pub struct Cap { pub send:u32,pub recv:u32,pub send_sge:u32,pub recv_sge:u32,pub inline:u32 }
#[repr(C)] pub struct Init { pub user:*mut c_void,pub send_cq:*mut c_void,pub recv_cq:*mut c_void,pub srq:*mut c_void,pub cap:Cap,pub kind:i32,pub signal_all:i32 }
#[repr(C,align(8))] #[derive(Default)] pub struct Gid(pub [u8;16]);
#[repr(C)] #[derive(Default)] pub struct Route { pub gid:Gid,pub flow:u32,pub index:u8,pub hops:u8,pub traffic:u8 }
#[repr(C)] #[derive(Default)] pub struct Ah { pub route:Route,pub lid:u16,pub sl:u8,pub path:u8,pub rate:u8,pub global:u8,pub port:u8 }
#[repr(C)] #[derive(Default)] pub struct Attr { pub state:i32,pub current:i32,pub mtu:i32,pub migration:i32,pub qkey:u32,pub rq_psn:u32,pub sq_psn:u32,pub destination:u32,pub access:u32,pub cap:Cap,pub ah:Ah,pub alt_ah:Ah,pub pkey:u16,pub alt_pkey:u16,pub notify:u8,pub draining:u8,pub atomic:u8,pub dest_atomic:u8,pub rnr_timer:u8,pub port:u8,pub timeout:u8,pub retries:u8,pub rnr_retries:u8,pub alt_port:u8,pub alt_timeout:u8,pub rate:u32 }
#[repr(C)] #[derive(Clone,Copy,Default)] pub struct Sge { pub address:u64,pub length:u32,pub key:u32 }
#[repr(C)] pub struct Send { pub id:u64,pub next:*mut Send,pub sg:*mut Sge,pub count:i32,pub opcode:i32,pub flags:u32,pub imm:u32,pub wr:[u64;4],pub qp_type:u32,pub padding:u32,pub tail:[u64;6] }
#[repr(C)] pub struct Recv { pub id:u64,pub next:*mut Recv,pub sg:*mut Sge,pub count:i32 }
#[repr(C)] #[derive(Clone,Copy,Default)] pub struct Wc {pub id:u64,pub status:i32,pub opcode:i32,pub vendor:u32,pub bytes:u32,pub imm:u32,pub qp:u32,pub source:u32,pub flags:u32,pub pkey:u16,pub lid:u16,pub sl:u8,pub path:u8}
const _:()={assert!(size_of::<Attr>()==144);assert!(size_of::<Send>()==128);assert!(size_of::<Recv>()==32);assert!(size_of::<Wc>()==48);};

// Factory DGX OS glibc exports these; no libibverbs developer symlink is needed.
unsafe extern "C" {
    fn dlopen(name:*const c_char, flags:i32)->*mut c_void;
    fn dlsym(handle:*mut c_void, name:*const c_char)->*mut c_void;
    fn dlerror()->*const c_char;
    fn dlclose(handle:*mut c_void)->i32;
}

pub struct Verbs {
    pub get_device_list:unsafe extern "C" fn(*mut i32)->*mut *mut c_void,
    pub free_device_list:unsafe extern "C" fn(*mut *mut c_void),
    pub get_device_name:unsafe extern "C" fn(*mut c_void)->*const c_char,
    pub open_device:unsafe extern "C" fn(*mut c_void)->*mut Context,
    pub close_device:unsafe extern "C" fn(*mut Context)->i32,
    pub alloc_pd:unsafe extern "C" fn(*mut Context)->*mut c_void,
    pub dealloc_pd:unsafe extern "C" fn(*mut c_void)->i32,
    pub create_cq:unsafe extern "C" fn(*mut Context,i32,*mut c_void,*mut c_void,i32)->*mut c_void,
    pub destroy_cq:unsafe extern "C" fn(*mut c_void)->i32,
    pub create_qp:unsafe extern "C" fn(*mut c_void,*mut Init)->*mut Qp,
    pub modify_qp:unsafe extern "C" fn(*mut Qp,*mut Attr,i32)->i32,
    pub destroy_qp:unsafe extern "C" fn(*mut Qp)->i32,
    pub reg_mr:unsafe extern "C" fn(*mut c_void,*mut c_void,usize,i32)->*mut Mr,
    pub dereg_mr:unsafe extern "C" fn(*mut Mr)->i32,
}

impl Verbs {
    pub fn load()->Result<&'static Self> {
        static VERBS:OnceLock<Result<Verbs>>=OnceLock::new();
        VERBS.get_or_init(|| {
            // RTLD_NOW | RTLD_LOCAL. Keep a successful handle for process life:
            // endpoints also call provider operations stored in their contexts.
            let handle=unsafe{dlopen(c"libibverbs.so.1".as_ptr(),2)};
            if handle.is_null(){return Err(load_error("cannot load library"));}
            let result=(|| unsafe {Ok(Self {
                get_device_list:std::mem::transmute(symbol(handle,c"ibv_get_device_list")?),
                free_device_list:std::mem::transmute(symbol(handle,c"ibv_free_device_list")?),
                get_device_name:std::mem::transmute(symbol(handle,c"ibv_get_device_name")?),
                open_device:std::mem::transmute(symbol(handle,c"ibv_open_device")?),
                close_device:std::mem::transmute(symbol(handle,c"ibv_close_device")?),
                alloc_pd:std::mem::transmute(symbol(handle,c"ibv_alloc_pd")?),
                dealloc_pd:std::mem::transmute(symbol(handle,c"ibv_dealloc_pd")?),
                create_cq:std::mem::transmute(symbol(handle,c"ibv_create_cq")?),
                destroy_cq:std::mem::transmute(symbol(handle,c"ibv_destroy_cq")?),
                create_qp:std::mem::transmute(symbol(handle,c"ibv_create_qp")?),
                modify_qp:std::mem::transmute(symbol(handle,c"ibv_modify_qp")?),
                destroy_qp:std::mem::transmute(symbol(handle,c"ibv_destroy_qp")?),
                reg_mr:std::mem::transmute(symbol(handle,c"ibv_reg_mr")?),
                dereg_mr:std::mem::transmute(symbol(handle,c"ibv_dereg_mr")?),
            })})();
            if result.is_err(){unsafe{dlclose(handle);}}
            result
        }).as_ref().map_err(Clone::clone)
    }
}

fn symbol(handle:*mut c_void,name:&CStr)->Result<*mut c_void> {
    let address=unsafe{dlsym(handle,name.as_ptr())};
    if address.is_null(){Err(load_error(&format!("missing symbol {}",name.to_string_lossy())))}else{Ok(address)}
}

fn load_error(what:&str)->String {
    let error=unsafe{dlerror()};
    let detail=if error.is_null(){"dynamic loader failed".into()}else{unsafe{CStr::from_ptr(error)}.to_string_lossy()};
    format!("multi-Spark RDMA requires libibverbs.so.1 from the DGX OS libibverbs1 package ({what}): {detail}")
}
