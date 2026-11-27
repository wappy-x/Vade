//! Checkpoint-backed logical FP32 decoder reference. This is not a GPU driver.
#[path="layer-reference/checkpoint.rs"] mod checkpoint;
#[path="layer-reference/receipt.rs"] mod receipt;
use engine_model::{Result,plan::{ConfigPlan,description::TokenMixer,layer::{LayerPlan,Topology,Element,reference::{self,Tensor,Data,Tensors}}},shapes::Ffn,safetensors};
use std::{fs,path::Path};
fn invalid(message: impl Into<String>) -> engine_model::Error { std::io::Error::new(std::io::ErrorKind::InvalidData,message.into()).into() }

fn main() { if let Err(e)=run() { eprintln!("layer-reference: {e}"); std::process::exit(1); } }
fn run() -> Result<()> {
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    match a.first().map(String::as_str) {
        Some("generate") if a.len()==4 => generate(Path::new(&a[1]),Path::new(&a[2]),&a[3]),
        Some("verify") if a.len()==2 => receipt::verify(Path::new(&a[1])),
        Some("compare") if a.len()==3 => receipt::compare(Path::new(&a[1]),Path::new(&a[2])),
        _ => Err(invalid("usage: layer-reference generate MODEL NEW_OUT IDS_CSV | verify DIR | compare EXPECTED ACTUAL")),
    }
}
fn generate(root: &Path,out: &Path,ids: &str) -> Result<()> {
    let ids=ids.split(',').map(|s| s.parse::<u32>().map_err(|_| invalid("invalid token ID"))).collect::<Result<Vec<_>>>()?;
    if ids.is_empty() || ids.len()>16 { return Err(invalid("reference needs 1 through 16 text IDs")); }
    let config=safetensors::read_bounded(&root.join("config.json"),engine_model::shapes::CONFIG_BYTES)?;
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(&config).map_err(|e| invalid(e.to_string()))? else { return Err(invalid("reference needs deferred text operation data")); };
    if !matches!(d.layers.first().map(|l| l.mixer),Some(TokenMixer::Kda(_))) { return Err(invalid("reference tool needs a KDA first decoder")); }
    if ids.iter().any(|&id| id as u64>=d.vocab) { return Err(invalid("token ID is outside configured vocabulary")); }
    let moe=d.layers.iter().enumerate().find(|(_,l)| matches!(l.mixer,TokenMixer::Kda(_)) && matches!(l.ffn,Ffn::Experts(_)))
        .map(|(j,_)| j).ok_or_else(|| invalid("no KDA expert layer"))?;
    let cp=checkpoint::Checkpoint::open(root,d.storage.clone())?;
    fs::create_dir(out)?;
    fs::write(out.join("config.json"),&config)?;
    let mut provenance=format!("profile=logical-f32-v1\nconfig_sha256={}\nindex_sha256={}\nids={ids:?}\nlayers=0,{moe}\n",
        engine_hash::digest(&config),cp.index_sha256);
    provenance.push_str("input=stored BF16 embedding row widened to FP32 and copied to each residual stream\nstate=zero initially; carry this layer state between text IDs\nlayer_isolation=true; expert layer input is the embedding, not earlier layer output\nprojection=decoded weights; FP32 multiply then ascending FP32 add; no activation quantization\ncasts=no checkpoint BF16 activation casts\nexecution=host-only; not checkpoint backend parity\n");
    fs::write(out.join("PROFILE.txt"),provenance)?;
    fs::write(out.join("POLICY.txt"),receipt::POLICY)?;
    let mut rows=String::from("file\tdtype\tshape\tbytes\tsha256\n");
    for layer in [0,moe].into_iter().collect::<std::collections::BTreeSet<_>>() {
        let mut carry=Tensors::new();
        for (token,&id) in ids.iter().enumerate() {
            let plan=LayerPlan::new(&d,layer,(token+1) as u64,Topology {parts:1,rank:0})?;
            let embedding=cp.embedding(id)?;
            if embedding.len()!=d.hidden as usize { return Err(invalid("embedding width differs from config")); }
            let mut inputs=Tensors::new();
            for b in &plan.inputs {
                let t=if b.name=="residual" { Tensor::f32(&b.shape,embedding.repeat(d.residual.streams as usize))? }
                    else if let Some(t)=carry.get(&b.name) { t.clone() }
                    else { match b.element {Element::F32=>Tensor::f32(&b.shape,vec![0.0;b.elements()?])?,
                        Element::U32=>Tensor {shape:b.shape.clone(),data:Data::U32((0..b.elements()? as u32).collect())}} };
                inputs.insert(b.name.clone(),t);
            }
            println!("REFERENCE_START layer={layer} token={token} id={id}");
            let f=reference::forward_from(&d,layer,(token+1) as u64,&cp,&inputs)?;
            f.check_plan(&plan)?;
            for (name,t) in &f.tensors {
                let (dtype,bytes)=match &t.data {Data::F32(v)=>("F32",v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
                    Data::U32(v)=>("U32",v.iter().flat_map(|v| v.to_le_bytes()).collect())};
                let file=format!("layer{layer}.token{token}.{name}.safetensors");
                let encoded=receipt::encode(dtype,&t.shape,&bytes);
                fs::write(out.join(&file),&encoded)?;
                rows.push_str(&format!("{file}\t{dtype}\t{:?}\t{}\t{}\n",t.shape,bytes.len(),engine_hash::digest(&encoded)));
            }
            for (name,next) in [("kda.history","next_history"),("kda.state","next_state")] {
                if let Some(t)=f.tensors.get(next) { carry.insert(name.into(),t.clone()); }
            }
            let trace=f.trace.iter().map(|e| format!("{}\t{:?}\t{:?}\t{:?}\n",e.id,e.kind,e.inputs,e.outputs)).collect::<String>();
            fs::write(out.join(format!("layer{layer}.token{token}.trace.tsv")),trace)?;
            let values=f.tensors["output"].floats()?;
            println!("REFERENCE_DONE layer={layer} token={token} elements={} max_abs={:.9e}",values.len(),values.iter().fold(0f32,|a,v| a.max(v.abs())));
        }
    }
    fs::write(out.join("TENSORS.tsv"),rows)?;
    fs::write(out.join("WEIGHTS.tsv"),format!("tensor\tdtype\tshape\tstart\tend\tshard\tsha256\n{}",cp.receipts.borrow().values().map(|s| s.clone()+"\n").collect::<String>()))?;
    fs::write(out.join("FP4-BANDS.tsv"),format!("tensor\telements\tmax_abs_dequant_round_error\tmax_relative_dequant_round_error\tmax_ulp_error\n{}",cp.bands.borrow().values().map(|s| s.clone()+"\n").collect::<String>()))?;
    receipt::seal(out)?; receipt::verify(out)?;
    println!("REFERENCE_READY profile=logical-f32-v1 device_pass=false out={}",out.display()); Ok(())
}
