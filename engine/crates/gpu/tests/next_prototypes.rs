//! Full-buffer GPU gates against the frozen first-party CUDA prototypes.
//! The oracle executable is built outside the product tree; see next-fast.md.
use std::{fs,process::Command};
fn oracle(kernel:&str) {
    let tool=std::env::var_os("VADE_FLASHNEXT_PROTO_CHECK").expect("set VADE_FLASHNEXT_PROTO_CHECK to the frozen prototype checker");
    let ptx=std::env::temp_dir().join(format!("vade-next-{}-{kernel}.ptx",std::process::id()));
    fs::write(&ptx,spark_gpu_probe::layer_ptx().expect("build cargo kernels").to_bytes()).unwrap();
    let result=Command::new(tool).arg(&ptx).arg(kernel).output();
    fs::remove_file(ptx).unwrap();
    let output=result.expect("run prototype checker");
    assert!(output.status.success(),"{}\n{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));
    let receipt=String::from_utf8(output.stdout).unwrap();
    assert!(receipt.contains(&format!("PASS kernel={kernel} cells=")),"missing nonempty oracle receipt: {receipt}");
    print!("{receipt}");
}
macro_rules! gate {($($name:ident),*)=>{$(#[test] #[ignore = "skipped: no GPU; requires GB10 and frozen prototype checker"] fn $name(){oracle(stringify!($name));})*};}
gate!(flash_next_dense,flash_next_route,flash_next_gather,flash_next_routed,
    flash_next_routed_activate,
    flash_next_hc_pre,flash_next_hc_up);
#[test]
fn abi_sizes() {
    use spark_gpu_probe::kernel_params::*;use std::mem::size_of;
    assert_eq!([size_of::<NextMatrix>(),size_of::<NextExpert>(),size_of::<NextDense>(),size_of::<NextMoe>(),size_of::<NextRouted>(),size_of::<NextActivation>(),size_of::<NextHc>()],[24,72,56,144,56,24,112]);
}
