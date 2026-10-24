//! Module tables and request-boundary selection for test-only A/B serving.
use super::*;

// Only module ownership and resolved function tables differ between arms.
pub(super) struct AbKernels<'a> {
    module: Module<'a>, layers: Cell<LayerFunctions>, matvec: Cell<Handle>, vec8: Cell<Option<Handle>>,
    batch: RefCell<Option<batch::BatchKernels>>, multi: RefCell<Option<multi::Kernels>>,
    moe: RefCell<Option<moe::Kernels>>, fp8: RefCell<Option<fp8::Kernels>>, fp4: RefCell<Option<fp4::Kernels>>,
}
impl<'a> Backend<'a> {
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
