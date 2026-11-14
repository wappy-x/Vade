#![forbid(unsafe_code)]
#[cfg(target_os="linux")]
#[path="conductor-net/device.rs"] mod device;
fn main() {
    #[cfg(target_os="linux")]
    if let Err(error)=device::run() {eprintln!("CONDUCTOR_NET FAIL {error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("conductor-net needs Linux and CUDA");std::process::exit(1);}
}
