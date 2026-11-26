//! Compose an optional TP entry point with the untouched shared PTX module.
fn main()->Result<(),Box<dyn std::error::Error>>{
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    if a.len()!=3{return Err("usage: tp-ptx SHARED_PTX TP_PTX OUTPUT".into());}
    // Rename only the entry, preserving the separate canonical strided kernel.
    let base=std::fs::read_to_string(&a[0])?.replace(".entry matvec_bf16_f32_packed8(",".entry tp_original_packed8(");
    let extra=std::fs::read_to_string(&a[1])?.lines().filter(|l|![".version",".target",".address_size"].iter().any(|s|l.trim_start().starts_with(s))).collect::<Vec<_>>().join("\n").replace("tp_matvec_packed_prefetch","matvec_bf16_f32_packed8");
    std::fs::write(&a[2],format!("{base}\n{extra}\n"))?;Ok(())
}
