//! Module tables and request-boundary selection for test-only A/B serving.
use super::*;

// Only module ownership and resolved function tables differ between arms.
pub(super) struct AbKernels<'a> {
    #[cfg(feature="kernel-bench")]
    fixture_geometry:RefCell<Option<crate::fixture_image::Geometry>>,
    module: Module<'a>, layers: Cell<LayerFunctions>, matvec: Cell<Handle>, vec8: Cell<Option<Handle>>,
    batch: RefCell<Option<batch::BatchKernels>>, multi: RefCell<Option<multi::Kernels>>,
    moe: RefCell<Option<moe::Kernels>>, fp8: RefCell<Option<fp8::Kernels>>, fp4: RefCell<Option<fp4::Kernels>>,
}
impl<'a> Backend<'a> {
    #[cfg(feature="kernel-bench")]
    pub fn fixture_set_geometry(&self,geometry:Option<crate::fixture_image::Geometry>)->Result<(),String> {
        self.require_resident_stream()?;self.synchronize()?;
        if self.resident_queue_active() || self.context.graph_storage.get()!=0 {return Err("fixture geometry requires no resident queue or live graph".into());}
        let candidate=self.ab.borrow();let candidate=candidate.as_ref().ok_or("fixture geometry requires a loaded candidate image")?;
        *candidate.fixture_geometry.borrow_mut()=geometry;Ok(())
    }
    #[cfg(feature="kernel-bench")]
    pub(super) fn fixture_launch_dimensions(&self,name:&str,blocks:u32,threads:u32)->Result<(u32,u32),String> {
        if self.arm_x.get() {if let Some(candidate)=self.ab.borrow().as_ref() {if let Some(geometry)=&*candidate.fixture_geometry.borrow() {if geometry.kernel()==name {return geometry.dimensions(blocks,threads);}}}}
        Ok((blocks,threads))
    }
    #[cfg(feature="kernel-bench")]
    pub fn fixture_replace_baseline(&self,image:&[u8])->Result<(),String> {
        self.fixture_replace_image(image,true)?;self.select_arm(true)?;
        *self.ab.borrow_mut()=None;self.arm_x.set(false);Ok(())
    }
    #[cfg(feature="kernel-bench")]
    pub fn fixture_replace_candidate(&self,image:&[u8])->Result<(),String> {
        self.fixture_replace_image(image,false)
    }
    #[cfg(feature="kernel-bench")]
    pub fn fixture_replace_full_candidate(&self,image:&[u8])->Result<(),String> {
        self.fixture_replace_image(image,true)
    }
    #[cfg(feature="kernel-bench")]
    pub fn fixture_reset_resident(&self)->Result<(),String> {
        self.require_resident_stream()?;
        if self.resident_queue_active() {return Err("fixture reset requires the resident queue to be closed".into());}
        self.synchronize()?;
        if let Some(error)=self.fatal_error() {return Err(format!("fixture context is poisoned; reload required: {error}"));}
        *self.recorder.borrow_mut()=resident_launch::Recorder::default();
        self.synchronize()?;
        if self.context.graph_storage.get()!=0 {return Err("fixture reset found externally retained graphs; reload required".into());}
        self.dense_offer.set(None);self.flash_input.borrow_mut().take();self.flash_mixed.borrow_mut().take();
        self.clear_kv_reserve()?;self.synchronize()?;Ok(())
    }
    #[cfg(feature="kernel-bench")]
    fn fixture_replace_image(&self,image:&[u8],whole:bool)->Result<(),String> {
        use crate::module_loading::Driver;
        self.require_resident_stream()?;
        if self.resident_queue_active() || self.context.graph_storage.get()!=0 {return Err("fixture image replacement requires no resident queue or live graph".into());}
        self.synchronize()?;self.select_arm(false)?;
        let mut driver=ModuleDriver {context:self.context};
        let loaded=match crate::fixture_image::validate(image)? {
            crate::fixture_image::Image::Ptx(ptx)=>driver.ptx(&ptx,self.context.jit_receipts.is_some()).map_err(|failure| {
                if let Some(handle)=failure.handle {if let Err(cleanup)=driver.unload(handle) {return format!("{}; {cleanup}",failure.error);}}
                failure.error
            })?,
            crate::fixture_image::Image::Cubin=>{
                let (code,handle)=driver.cubin(image);
                if code!=0 {if let Some(handle)=handle {driver.unload(handle)?;}return Err(driver.error(code));}
                crate::module_loading::Loaded {handle:handle.ok_or("fixture image returned no module")?,receipt:None}
            },
        };
        let module=Module {raw:Cell::new(loaded.handle),context:self.context,jit_module:Cell::new(loaded.receipt),functions:Default::default()};
        let matvec=Cell::new(if whole {module.function("matvec_bf16_f32_warp32")?} else {self.matvec_function.get()});
        let vec8=Cell::new(if whole {module.function("matvec_bf16_f32_vec8").ok()} else {self.vec8.get()});
        let layers=Cell::new(if whole {LayerFunctions::load(&module)?} else {self.layer_kernels()?});
        *self.ab.borrow_mut()=Some(AbKernels {
            #[cfg(feature="kernel-bench")] fixture_geometry:RefCell::new(None),
            module,layers,matvec,vec8,batch:RefCell::new(None),multi:RefCell::new(None),
            moe:RefCell::new(None),fp8:RefCell::new(None),fp4:RefCell::new(None)});Ok(())
    }
    /// Test-only: a second module in this context; weights, stream and buffers stay shared.
    pub fn load_ab_ptx(&self, ptx: &CStr) -> Result<(), String> {
        if self.ab.borrow().is_some() { return Err("A/B module already loaded".into()); }
        let start = Instant::now();
        let module = Module::load(self.context, ptx)?;
        let matvec = Cell::new(module.function("matvec_bf16_f32_warp32")?);
        let vec8 = Cell::new(module.function("matvec_bf16_f32_vec8").ok());
        if self.mode == MatvecMode::Vec8 && vec8.get().is_none() { return Err("X PTX has no vec8 kernel".into()); }
        let layers = Cell::new(LayerFunctions::load(&module)?);
        *self.ab.borrow_mut() = Some(AbKernels { module, layers, matvec, vec8,
            #[cfg(feature="kernel-bench")] fixture_geometry:RefCell::new(None),
            batch: RefCell::new(None), multi: RefCell::new(None), moe: RefCell::new(None),
            fp8: RefCell::new(None), fp4: RefCell::new(None) });
        eprintln!("gpu_backend ab_ptx_load_ms={:.3}", start.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }
    pub fn ab_enabled(&self) -> bool { self.ab.borrow().is_some() }
    pub fn arm_index(&self) -> usize { usize::from(self.arm_x.get()) }
    /// Called only between serialized requests, before session creation. No launch branch.
    pub fn select_arm(&self, x: bool) -> Result<(), String> {
        if self.arm_x.get() == x { return Ok(()); }
        let mut ab = self.ab.borrow_mut();
        let other = ab.as_mut().ok_or("arm requires VADE_AB_PTX")?;
        let layers = self.layers.as_ref().map_err(Clone::clone)?;
        layers.functions.swap(&other.layers);
        self.module.swap(&other.module);
        self.matvec_function.swap(&other.matvec); self.vec8.swap(&other.vec8);
        self.batch.swap(&other.batch); self.multi.swap(&other.multi); self.moe.swap(&other.moe);
        self.fp8.swap(&other.fp8); self.fp4.swap(&other.fp4);
        self.arm_x.set(x);
        Ok(())
    }

    /// Arm declarations resolve only active functions; PTX and launch arithmetic are unchanged.
    pub fn require_arm_kernels(&self, names: &[&str]) -> Result<(), String> {
        for name in names { self.module.function(name)?; }
        Ok(())
    }
    pub fn require_layer_kernels(&self) -> Result<(), String> {
        self.layers.as_ref().map(|_| ()).map_err(Clone::clone)
    }
    #[inline]
    pub(super) fn layer_kernels(&self) -> Result<LayerFunctions, String> {
        self.layers.as_ref().map(|k| k.functions.get()).map_err(Clone::clone)
    }
    pub(super) fn layer_module(&self) -> Result<&Module<'a>, String> {
        // Used only when resolving a lazy function table, never per launch.
        if self.arm_x.get() { return Ok(&self.module); }
        self.layers.as_ref().map(|k| &k._module).map_err(Clone::clone)
    }
}
