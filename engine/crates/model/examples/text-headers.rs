//! Inspect deferred text headers. Do not read weight payloads or admit execution.
use engine_model::{Result,Error,plan::ConfigPlan,safetensors::read_bounded,shapes::{CONFIG_BYTES,HEADER_BYTES}};
use std::path::Path;
fn main()->Result<()> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=2 {return Err(Error("usage: text-headers CHECKPOINT_DIRECTORY".into()));}
    let root=Path::new(&args[1]);
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&read_bounded(&root.join("config.json"),CONFIG_BYTES)?)
        .map_err(|e|Error(e.to_string()))? else {return Err(Error("expected a deferred text description".into()));};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES)?;
    let a=d.open_headers(root,&index)?;
    println!("header-admission=pass text-weights={} stored-bytes={} execution=deferred",a.headers().weights().len(),a.headers().stored_bytes);
    Ok(())
}
