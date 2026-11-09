//! Default-off real-checkpoint projection referee. No model or transport.
#![forbid(unsafe_code)]
#[cfg(target_os="linux")]
#[path="kernel-referee/device.rs"] mod device;
fn main() {
    #[cfg(target_os="linux")]
    if let Err(error)=device::run() {eprintln!("REFEREE FAIL {error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("REFEREE requires Linux and CUDA");std::process::exit(1);}
}
