//! Print a deferred decoder program without opening a weight file.
use engine_model::{plan::{ConfigPlan,layer::{LayerPlan,Topology}},safetensors};
fn main() {
    if let Err(error)=run() { eprintln!("layer-plan: {error}"); std::process::exit(1); }
}
fn run() -> Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=5 { return Err("usage: layer-plan CONFIG LAYER CONTEXT PARTS RANK".into()); }
    let bytes=safetensors::read_bounded(std::path::Path::new(&args[0]),engine_model::shapes::CONFIG_BYTES)?;
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&bytes)? else { return Err("config does not contain deferred operation data".into()); };
    let plan=LayerPlan::new(&d,args[1].parse()?,args[2].parse()?,Topology {parts:args[3].parse()?,rank:args[4].parse()?})?;
    println!("validation=logical-plan-only layer={} context={} parts={} rank={} ops={} missing={}",plan.layer,plan.context,plan.topology.parts,plan.topology.rank,plan.steps.len(),plan.missing().len());
    println!("SETTINGS {:?}",plan.settings);
    for input in &plan.inputs { let split=plan.input_splits.get(&input.name).copied().unwrap_or(engine_model::plan::layer::Split::Replicated);
        println!("INPUT {} {:?} {:?} local={:?}",input.name,input.shape,input.element,plan.topology.interval(&input.shape,split)?); }
    for (index,step) in plan.steps.iter().enumerate() {
        println!("OP {index} {} {:?} inputs={:?} revision={} kernels={:?} gap={:?}",step.id,step.kind,step.inputs,step.target.revision,step.target.entries,step.target.gap);
        for w in &step.weights { println!("WEIGHT {} {:?} {:?} local={:?}",w.name,w.shape,w.storage,plan.topology.interval(&w.shape,w.split)?); }
        for (output,&split) in step.outputs.iter().zip(&step.output_splits) {
            println!("OUTPUT {} {:?} {:?} local={:?}",output.name,output.shape,output.element,plan.topology.interval(&output.shape,split)?);
        }
        for scratch in &step.scratch { println!("SCRATCH {} {:?}",scratch.name,scratch.shape); }
        for name in &step.gather { println!("GATHER {name} order=global-axis operation=copy"); }
    }
    println!("RETURNS {:?}",plan.returns);
    println!("LIFETIMES {:?}",plan.lifetimes()); Ok(())
}
