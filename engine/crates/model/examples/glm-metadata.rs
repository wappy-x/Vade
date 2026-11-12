//! Inspect saved text config and index data. Never open a weight file.
use engine_model::{Result, Error, plan::{ConfigPlan, description::TokenMixer, storage::StorageMap}, safetensors::read_bounded,
    shapes::{CONFIG_BYTES, HEADER_BYTES, Ffn, Place}};
use std::{collections::BTreeMap, path::Path};
fn invalid(message: impl Into<String>) -> Error { Error(message.into()) }
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) { return Err(invalid("usage: glm-metadata CONFIG INDEX [QUANT_CONFIG]")); }
    let ConfigPlan::Deferred(p) = ConfigPlan::from_config(&read_bounded(Path::new(&args[0]),CONFIG_BYTES)?)
        .map_err(|e| invalid(e.to_string()))? else { return Err(invalid("expected a deferred text description")); };
    let mapped = p.map_index(&read_bounded(Path::new(&args[1]),HEADER_BYTES)?)?;
    if let Some(path) = args.get(2) {
        let q = StorageMap::from_json(&read_bounded(Path::new(path),CONFIG_BYTES)?)?;
        if q != p.storage { return Err(invalid("text storage recipes differ between config and sidecar")); }
        println!("quant_sidecar=match");
    }
    let mut roles = BTreeMap::<String,usize>::new();
    for b in mapped.tensors.values() { *roles.entry(format!("{:?}",b.operation)).or_default() += 1; }
    println!("validation=index-names-only execution=deferred payload_files_opened=0");
    println!("hidden={} vocab={} decoder_layers={} declared_draft_layers={} eos={:?}",p.hidden,p.vocab,p.layers.len(),p.declared_draft_layers,p.eos_tokens);
    println!("kda_layers={} latent_layers={} dense_layers={} expert_layers={}",
        p.layers.iter().filter(|l| matches!(l.mixer,TokenMixer::Kda(_))).count(),
        p.layers.iter().filter(|l| matches!(l.mixer,TokenMixer::LatentAttention(_))).count(),
        p.layers.iter().filter(|l| matches!(l.ffn,Ffn::Dense { .. })).count(),
        p.layers.iter().filter(|l| matches!(l.ffn,Ffn::Experts(_))).count());
    println!("index_tensors={} decoder_tensors={} draft_tensors={} unused_vision_tensors={} declared_payload_bytes={}",mapped.tensors.len(),
        mapped.tensors.values().filter(|b| matches!(b.role.place,Place::Layer(_))).count(),
        mapped.tensors.values().filter(|b| matches!(b.role.place,Place::Draft { .. })).count(),
        mapped.tensors.values().filter(|b| b.role.place == Place::Unused).count(),mapped.index.total_size);
    for (role,count) in roles { println!("{role}={count}"); }
    Ok(())
}
