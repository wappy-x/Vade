//! Real-weight, bench-only comparison of packed and absorbed attention.
#[cfg(target_os="linux")]
#[path="latent-absorb/device.rs"] mod device;
fn main() {
    #[cfg(target_os="linux")] if let Err(error)=device::run() {eprintln!("latent-absorb: {error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))] {eprintln!("latent-absorb requires Linux CUDA");std::process::exit(1);}
}
