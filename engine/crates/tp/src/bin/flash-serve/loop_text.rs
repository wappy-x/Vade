//! The start of the text service of the looped block family on one Spark, behind `--diagnostic-unqualified-text`.
//!
//! The library holds the service and its start for any backend (`spark_tp::loop_decoder::service::start`). This
//! file is the Linux half of a start: the memory rule, and for the device run the GPU context, the load of the
//! weights and the call. With `--diagnostic-host-arithmetic` it starts the host run of the library and opens no GPU.
//! The steps of a device start, in the words of the startup lines: `header_admission`, `tokenizer_identity`,
//! `memory`, `memory_admission`, `gpu_context_and_kernel_image`, `weights`, `host_weights`, then the ready line.
use engine_model::plan::{description::LoopTextDescription,loop_load::{self,ExpertCodes}};
use spark_gpu_probe::{packed_product,staged_product};
use spark_tp::{Result,bounded_decoder::Device,loop_decoder::{operands::{device::{DeviceStore,PanelPlan},load,weights::LoadedWeights},service::{STEP_LIMITS,admit,start}},startup};
use std::{net::TcpListener,path::Path,sync::atomic::AtomicBool,time::Instant};

/// The memory margin of the vector pool: the pool keeps no spare that would leave less than this free.
const VECTOR_POOL_MARGIN:usize=1<<30;
/// The rule of the grid of a launch whose extent follows the context: a straight plane holds whole buckets.
const BUCKET:u64=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;

/// Start the service when `root` holds a model of this family and the start gives the diagnostic option.
/// `None`: another model, or no diagnostic option; the caller goes on as it did before this file.
/// `output` and `memory` are the caller's settings of the output cap and of the memory ceiling, as text.
pub fn run(args:&[String],root:&Path,output:Option<String>,memory:Option<String>,started:Instant)->Option<Result<()>> {
    if !args.get(9..).unwrap_or_default().iter().any(|option|option==start::DIAGNOSTIC) {return None;}
    let config=engine_model::safetensors::read_bounded(&root.join("config.json"),engine_model::shapes::CONFIG_BYTES).ok()?;
    LoopTextDescription::from_config(&config).ok()?;
    Some(serve(args,root,output,memory,started))
}
/// The budget under the caller's memory ceiling, when it gives one.
fn ceiling(budget:usize,memory:Option<String>)->Result<usize> {
    Ok(match memory {Some(text)=>budget.min(text.parse::<usize>().ok().filter(|&bytes|bytes>0).ok_or("SEQPAR_MEMORY_BYTES must be a positive byte count")?),None=>budget})
}
fn serve(args:&[String],root:&Path,output:Option<String>,memory:Option<String>,started:Instant)->Result<()> {
    let settings=start::settings(args,output)?;
    startup::begin(settings.rank,started,"header_admission");
    if settings.options.host {
        // The host run opens no GPU. Its budget is the memory rule on the free host memory.
        startup::step("memory");
        let free=spark_gpu_probe::driver::unified_available_bytes(usize::MAX)?;
        let budget=ceiling(free.saturating_sub((free/10).max(2<<30)),memory)?;
        let listener=TcpListener::bind(settings.bind).map_err(|e|e.to_string())?;
        return start::serve_host(root,&settings,budget as u64,BUCKET,listener,&AtomicBool::new(false));
    }
    let mut checkpoint=loop_load::read_folder(root).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    // Keep the widened fallback whenever either packed projection cannot take the shape.
    let dropped=checkpoint.load.packed_experts_only(|packed| {
        let (Some(unit),Some([experts,..]))=(packed.unit(),packed.stack()) else {return false;};
        (1..=packed_product::PICKED_EXPERTS_MAX as u64).contains(&experts)
            && packed_product::shape_admitted(unit.rows,unit.columns,unit.codes==ExpertCodes::Mxfp8,unit.order.is_some())
            && staged_product::rows_shape_admitted(unit.rows,unit.columns,1)
    }).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    eprintln!("LOOP TEXT EXPERTS form=packed widened_stacks_not_loaded={dropped} packed_stacks={} packed_bytes={}",checkpoint.load.bytes.packed_stacks,checkpoint.load.bytes.packed);
    let d=&checkpoint.description;
    startup::step("tokenizer_identity");
    let text=start::text(root,d)?;
    // The kernel image of the block arms, which every step runs: a build without it stops here, before the GPU opens.
    startup::step("kernel_image");
    start::need_kernel_image(spark_gpu_probe::inkling_ptx().is_some())?;
    startup::step("memory");
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    // The memory rule of the text service: the free bytes less the larger of one tenth and 2 GiB.
    let budget=ceiling(free.saturating_sub((free/10).max(2<<30)),memory)?;
    startup::step("memory_admission");
    admit(d,&checkpoint.load.bytes,settings.context,budget as u64,BUCKET,STEP_LIMITS)?;
    let panels=PanelPlan::new(&checkpoint.load)?;
    let ptx=spark_gpu_probe::layer_ptx()?;
    let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
    startup::step("gpu_context_and_kernel_image");
    spark_gpu_probe::driver::with_backend(0,ptx,budget,|backend| {
        // The kernel image of the block arms, loaded before the weights, so the first chat does not pay for it.
        startup::step("kernel_image_load");
        backend.inkling_preload()?;
        let model=(settings.options.depth>0).then(||engine_model::mtp_draft::MtpModel::open_with_fold(root,|a,b,c,v,h,r|backend.fold_markov(a,b,c,v,h,r))).transpose().map_err(|e|e.to_string())?;
        let format=Some(engine_model::plan::loop_load::ExpertCodes::Nvfp4);
        let raw_draft=model.as_ref().map(|m|spark_tp::loop_decoder::draft::packed_weight_bytes(m,format,format)).transpose()?.unwrap_or(0);
        let main_budget=budget.saturating_sub(raw_draft) as u64;
        let baseline=admit(d,&checkpoint.load.bytes,settings.context,main_budget,BUCKET,STEP_LIMITS)?;
        let panel_load=panels.charge(checkpoint.load.bytes)?;
        let with_panels=admit(d,&panel_load,settings.context,main_budget,BUCKET,STEP_LIMITS).ok();
        let use_panels=with_panels.is_some();
        let mut main_load=if use_panels {panel_load} else {checkpoint.load.bytes};
        let admitted=with_panels.unwrap_or(baseline);
        eprintln!("LOOP TEXT ADMISSION context={} available_bytes={free} budget_bytes={budget} weights_bytes={} chat_bytes={} ledger_bytes={} session_pool_bytes={} largest_context={} step_rows={}",
            settings.context,admitted.ledger.weights,admitted.ledger.chat(),admitted.ledger.total,admitted.room()?,admitted.largest_context,admitted.limits.step_rows);
        let before=backend.allocated_bytes();
        startup::step("weights");
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,root,&mut store,workers)?;
        let files=load::Files::open(&checkpoint,root)?;
        if use_panels {
            let before_panels=backend.allocated_bytes();
            panels.upload(backend,&files,&loaded.operands)?;
            let panel_bytes=backend.allocated_bytes().saturating_sub(before_panels);
            let (tiles,outliers)=backend.bf16_panel_tiles();
            eprintln!("LOOP TEXT BF16 PANELS bytes={panel_bytes} budget={} tiles={tiles} outliers={outliers}",panels.bytes());
        }
        backend.synchronize()?;
        let resident=backend.allocated_bytes()-before;
        // The ledger admitted the context before the load. Variable-size lossless panels may hold less than budgeted.
        if resident as u64>admitted.ledger.weights {return Err(format!("the device holds {resident} bytes of weights; the ledger counted {}",admitted.ledger.weights));}
        if use_panels {
            main_load.device=resident as u64;
        }
        let admitted=if use_panels {admit(d,&main_load,settings.context,main_budget,BUCKET,STEP_LIMITS)?} else {admitted};
        startup::step("host_weights");
        let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
        eprintln!("LOOP TEXT LOAD tensors={} units={} fingerprint={} device_weights_bytes={resident} host_weight_bytes={} allocate_seconds={:.3} fill_seconds={:.3} seconds={:.3}",
            loaded.report.tensors,loaded.report.units,loaded.fingerprint(),weights.host_bytes(),loaded.report.allocate_seconds,loaded.report.fill_seconds,started.elapsed().as_secs_f64());
        let compute=Device::new(backend);
        let listener=TcpListener::bind(settings.bind).map_err(|e|e.to_string())?;
        let stop=AtomicBool::new(false);
        let before_draft=backend.allocated_bytes();
        let (main_tiles,main_outliers)=backend.bf16_panel_tiles();
        if use_panels {
            store.set_panel_room(admitted.room()?.saturating_sub(admitted.chat_bytes()?));
        }
        let draft=model.as_ref().map(|m|spark_tp::loop_decoder::draft::Drafter::with_packed(m,&mut store,&compute,format,format)).transpose()?;
        let admitted=if draft.is_some() {
            let available=spark_gpu_probe::driver::unified_available_bytes(total)?;
            let draft_bytes=backend.allocated_bytes().saturating_sub(before_draft);
            let (tiles,outliers)=backend.bf16_panel_tiles();
            eprintln!("LOOP TEXT DRAFT WEIGHTS bytes={draft_bytes} panel_bytes={} panel_tiles={} panel_outliers={}",
                draft_bytes.saturating_sub(raw_draft),tiles-main_tiles,outliers-main_outliers);
            let remaining=budget.saturating_sub(draft_bytes).min(resident+available.saturating_sub((free/10).max(2<<30)));
            admit(d,&main_load,settings.context,remaining as u64,BUCKET,STEP_LIMITS)?
        } else {
            admitted
        };
        // Retain freed working vectors and VMM KV pages only after measuring the loaded weight bytes.
        backend.configure_vector_pool(VECTOR_POOL_MARGIN);
        backend.configure_kv_reserve(VECTOR_POOL_MARGIN);
        let served=start::serve_with_draft(&compute,&weights,d,text,&admitted,&settings,listener,root,"device",&stop,draft.as_ref().map(|m|m as &_));
        let drained=backend.synchronize();
        served.and(drained)
    })
}
