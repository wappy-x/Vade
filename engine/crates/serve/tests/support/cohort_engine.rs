/// Host fake; excludes device execution and asynchronous session release.
struct Model {
    ab: Option<Arc<Mutex<Vec<bool>>>>,
    /// VADE_AB_COHORTS: arm of the live sessions and the most sessions seen live at once.
    cohort: Option<Arc<Mutex<(Option<bool>, usize)>>>,
    live: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    heads: Arc<AtomicUsize>,
    /// Session::settle_prefix and Session::fork, in the order they ran.
    order: Arc<Mutex<Vec<&'static str>>>,
    /// Engine::prefix_cache_opt_in (Flash-Next): only asking requests use the prefix cache.
    opt_in: bool,
    retention_extra: usize,
}
struct Kv {
    history: Vec<u32>,
    live: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    heads: Arc<AtomicUsize>,
    order: Arc<Mutex<Vec<&'static str>>>,
}
impl Drop for Kv {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}
impl Session for Kv {
    fn advance(&mut self, token: u32) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.history.push(token);
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.advance(token)?;
        self.heads.fetch_add(1, Ordering::Relaxed);
        Ok(self.logits())
    }

    fn truncate(&mut self, position: usize) -> Result<(), String> {
        self.history.truncate(position);
        Ok(())
    }
    fn settle_prefix(&mut self) -> Result<(), String> {
        self.order.lock().unwrap().push("settle");
        Ok(())
    }
    fn fork(&self) -> Result<Self, String> {
        self.live.fetch_add(1, Ordering::Relaxed);
        self.order.lock().unwrap().push("fork");
        Ok(Self {
            history: self.history.clone(),
            live: self.live.clone(),
            calls: self.calls.clone(),
            heads: self.heads.clone(),
            order: self.order.clone(),
        })
    }
}
impl Engine for Model {
    fn kv_branch_handles(&self) -> bool { true }
    fn prefix_cache_opt_in(&self) -> bool { self.opt_in }
    fn automatic_draft_allowed(&self, _: usize) -> bool { false }
    type Session<'a> = Kv;
    fn session_for_request(&self, request: &protocol::Request) -> Result<Kv, String> {
        if let Some(arms) = &self.ab {
            let live = self.live.load(Ordering::Relaxed);
            if let Some(cohort) = &self.cohort {
                let mut cohort = cohort.lock().unwrap();
                assert!(live == 0 || cohort.0 == Some(request.arm_x), "A/B cohort mixed arms");
                *cohort = (Some(request.arm_x), cohort.1.max(live + 1));
            } else {
                assert_eq!(live, 0, "A/B requests overlapped or reused state");
            }
            arms.lock().unwrap().push(request.arm_x);
        }
        self.session()
    }
    fn session(&self) -> Result<Kv, String> {
        self.live.fetch_add(1, Ordering::Relaxed);
        Ok(Kv {
            history: Vec::new(),
            live: self.live.clone(),
            calls: self.calls.clone(),
            heads: self.heads.clone(),
            order: self.order.clone(),
        })
    }
    fn vocab_size(&self) -> usize {
        8
    }
    fn context_limit(&self) -> usize {
        1024
    }
    fn eos_token_ids(&self) -> &[u32] {
        &[7]
    }
    fn session_reservation_bytes(&self) -> Option<usize> {
        Some(16384)
    }
    fn request_reservation_bytes_with_retention(&self, _: usize, _: bool, _: Option<usize>, retain: bool) -> Option<usize> {
        16384usize.checked_add(if retain { self.retention_extra } else { 0 })
    }
}
impl Kv {
    fn logits(&self) -> Vec<f32> {
        let hash = self
            .history
            .iter()
            .fold(0usize, |h, &t| (h * 17 + t as usize + 1) % 7);
        let mut logits = vec![0.; 8];
        logits[hash] = 1.;
        logits
    }
}
