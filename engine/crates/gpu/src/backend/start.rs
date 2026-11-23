//! Construct the backend on its owning context.
use super::*;
impl<'a> Backend<'a> {
    pub(super) fn load(context:&'a Context,ptx:&CStr,mode:MatvecMode,context_ms:f64)->Result<Self,String> {
        let start = Instant::now();
        let module = Module::load(context, ptx)?;
        let ptx_load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let matvec_function = module.function("matvec_bf16_f32_warp32")?;
        let requested_vec8 = mode == MatvecMode::Vec8;
        let vec8 = module.function("matvec_bf16_f32_vec8").ok();
        if requested_vec8 && vec8.is_none() { return Err("PTX has no vec8 kernel".into()); }
        let backend = Backend { context, module, mode, tc: tc_family::TcState::new(), inkling: inkling_family::InklingState::new(), vec8: Cell::new(vec8), ab: RefCell::new(None), arm_x: Cell::new(false), use_vec8: Cell::new(requested_vec8), batch: RefCell::new(None), multi: RefCell::new(None),
            moe: RefCell::new(None), fp8: RefCell::new(None), fp4: RefCell::new(None), decode_partials: RefCell::new(None),
            flash_input: RefCell::new(None), flash_mixed: RefCell::new(None), dense_offer: Cell::new(None),
            uploads: RefCell::new(UploadStaging { buffer: Pinned::new(context, crate::staging::UPLOAD_BYTES)?, capacity: crate::staging::UPLOAD_BYTES, used: 0 }),
            segment: Cell::new(None), capture_position: Cell::new(0), capture_base: Cell::new(0), prompt_order: Cell::new(false), programmatic: Cell::new(false),
            recorder: RefCell::new(resident_launch::Recorder::default()), side_launch: Cell::new(ptr::null_mut()), spin_waits: Cell::new(false), profiling: Cell::new(false), profile_kind: Cell::new(None), stats: RefCell::new(BackendStats::default()),
            stream: Stream::new(context)?, matvec_function: Cell::new(matvec_function), layers: LayerKernels::load(context), timings: RefCell::new(Vec::new()) };
        eprintln!("gpu_backend startup context_ms={context_ms:.3} ptx_load_ms={ptx_load_ms:.3} kernel_setup_ms={:.3} device_bytes={} pinned_staging_bytes={}",
            start.elapsed().as_secs_f64() * 1000.0, backend.allocated_bytes(), backend.pinned_staging_bytes());
        Ok(backend)
    }
}
