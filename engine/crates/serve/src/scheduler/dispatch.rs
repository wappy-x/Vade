//! Cooperative admission, scheduling, batched model work and shutdown draining.
use super::{
    Active, AtomicBool, CacheLimits, CachedSession, Config, DraftBank, DraftPolicy,
    Duration, Engine, Event, Generation, Instant, Limits, ModelAdmin, Ordering,
    PrefillBudget, Prepared, Reading, Ready, RequestError, Retired, Session,
    SessionCache, SessionKey, Swap, TcpListener, Tokenizer, Wire, accept_connections,
    balance_cache, complete_batch, http, model_response, mpsc, protocol,
    reclaim_retired, reservations, shared_plain, shared_prefix, unavailable,
};

pub(super) fn serve_loop<'e, E: Engine, T: Tokenizer>(
    listener: &TcpListener,
    engine: &'e E,
    tokenizer: Option<&T>,
    limits: &Limits,
    cache_limits: &CacheLimits,
    config: &Config,
    shutdown: &AtomicBool,
    admin: Option<&ModelAdmin>,
    model: &str,
    requests: usize,
    prepare_tx: &mpsc::SyncSender<(http::HttpRequest, Wire)>,
    prepare_rx: &mpsc::Receiver<Result<Prepared, (Wire, String)>>,
) -> Result<Option<Swap>, String> {
    limits.validate()?;
    config.validate(cache_limits, engine.cache_room_bytes())?;
    // Collect only the initial burst before a cold short prompt starts prefill.
    // Cached suffixes, decoding and long prompts retain their existing paths.
    let min_prompt_age = crate::intake_grace::minimum_prompt_age(crate::intake_grace::load()?);
    let mut burst = crate::intake_burst::Burst::new(crate::intake_burst::load()?.unwrap_or(limits.intake_burst));
    // Retirement/admission gate passed; default on, explicit zero disables it.
    // Read once, before serving.
    let mut refresh_retired = super::retirement_refresh_hook(
        super::retirement_pressure_refresh(std::env::var("VADE_RETIRE_PRESSURE_REFRESH"))?,
        || engine.refresh_memory_pressure());
    let (shared_prompt_limit, shared_prompt_rows) = if engine.native_draft_policy() {
        (engine.shared_prefill_limit(), engine.shared_prefill_rows())
    } else { (engine.shared_prompt_limit(), 2048) };
    if engine.shared_batch() && shared_prompt_rows == 0 {
        return Err(format!("shared prompt threshold {shared_prompt_limit} cannot run with shared row budget {shared_prompt_rows}"));
    }
    // Admission keeps the engine's cache threshold; only scheduling must fit a turn.
    let shared_prefill_limit = shared_prompt_limit.min(shared_prompt_rows);
    let shared_prefill_max_tokens = engine.shared_prefill_max_tokens();
    let reservation = engine
        .session_reservation_bytes()
        .ok_or("scheduler requires bounded engine sessions")?;
    if reservation == 0 || config.active_bytes < reservation {
        return Err("active session bound is too small".into());
    }
    let reserve_owner = config.active >= 3 && config.sharing.owner_key.is_some();
    let public_slots = config.active - usize::from(reserve_owner);
    let output_window = config.background_rows.max(config.foreground_rows).saturating_add(1);
    // Reserve both a slot and short-request memory for the authenticated owner.
    let foreground_bytes = if reserve_owner { engine.request_reservation_bytes(64 + config.foreground_rows)
        .ok_or("missing foreground reservation")?.min(config.active_bytes - reservation) } else { 0 };
    let mut cache = SessionCache::new(cache_limits.clone())?.with_room(engine.cache_room_bytes());
    let domain = crate::contract::Domain::new()?;
    let handles = engine.kv_branch_handles() && cache_limits.checkpoint_bytes() != 0;
    let retain_prefix = |j: &Ready| {
        j.request.cache
            || config.prefix_cache
                && j.request.prefix_cache
                && (!engine.native_draft_policy() || j.prompt.len() > shared_prompt_limit)
            || handles && j.request.prefix_cache && j.request.kv_checkpoint
    };
    let reservation_for = |j: &Ready, tokens, plain, cached| {
        engine
            .request_reservation_bytes_with_retention(tokens, plain, cached, retain_prefix(j))?
            .checked_add(engine.request_option_bytes(&j.request))
    };
    // Unfunded retention leaves free what live requests may hold beyond their bound.
    let request_hold = |j: &Ready| engine.unfunded_request_bytes(retain_prefix(j));
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let access = http::AccessPolicy::new(listener.local_addr().map_err(|e| e.to_string())?, limits)?;
    let mut reading: Vec<Reading> = Vec::new();
    let mut queued: Vec<Ready> = Vec::new();
    let mut active: Vec<Active<'_, '_, E, T>> = Vec::new();
    let mut retired: Vec<Retired<E::Session<'e>>> = Vec::new();
    let mut next_id = 0u64;
    let admission_clock = Instant::now();
    let mut tick = 0u64;
    let mut prefill_budget = PrefillBudget::new();
    let mut swap: Option<Swap> = None;
    let mut accepted = 0usize;
    let mut preparing = 0usize;
    let mut cancellations = Vec::new();
    let mut draft_bank = DraftBank::default();
    let mut draft_policy = DraftPolicy::new();
    let mut cohort_cost = engine.cohort_cost_control().then(|| {
        if engine.cohort_guard_robust() { crate::cohort_cost::Controller::robust() }
        else { crate::cohort_cost::Controller::default() }
    });
    let mut intake_deferred = false;
    let mut fatal = None;
    let (mut target_passes, mut target_rows, mut target_total_ms, mut draft_total_ms) = (0usize, 0usize, 0.0, 0.0);
    while !shutdown.load(Ordering::Relaxed) {
        if let Some(error) = engine.fatal_error() {
            #[cfg(target_os = "linux")]
            spark_gpu_probe::teardown::arm();
            fatal = Some(error);
            break;
        }
        if requests != 0 && accepted == requests && reading.is_empty() && queued.is_empty() && active.is_empty() && preparing == 0 { break; }
        // Destruction is bounded to one quantum plus one indivisible backend
        // release. Service intake between quanta, but do not evict cache or
        // allocate more GPU state until completed ownership has been released.
        if let Some(refresh) = refresh_retired.as_mut() {
            super::reclaim_retired_refresh(&mut retired, Duration::from_millis(2), refresh);
        } else {
            reclaim_retired(&mut retired, Duration::from_millis(2));
        }
        if retired.is_empty() {
            balance_cache(&mut cache, config.session_budget_bytes,
                active.iter().map(|j| &j.generation.state),
                engine.cache_hold_bytes() + active.iter().map(|j| request_hold(&j.ready)).sum::<usize>(),
                |reserved| engine.cache_free_physical_bytes(reserved))?;
            engine.maintain_memory(active.is_empty() && queued.is_empty())?;
        }
        if swap.is_none() { if let Some(admin) = admin { admin.accept(&mut reading, limits); } }
        // Keep accepting control requests even if idle sockets fill intake.
        // Generation and prompt admission share the work cap below.
        if (requests == 0 || accepted < requests) && (swap.is_none() || !active.is_empty()) {
            let slots = (config.connections + 1).saturating_sub(queued.len() + active.len() + preparing);
            let attempts = if requests == 0 { 32 } else { (requests - accepted).min(32) };
            let count = accept_connections(listener, &mut reading, limits, slots, attempts);
            if requests != 0 { accepted += count; }
        }

        let mut i = 0;
        while i < reading.len() {
            if reading[i].responding {
                if reading[i].wire.flush().is_err() || reading[i].wire.drained() {
                    reading.swap_remove(i);
                } else {
                    i += 1;
                }
                continue;
            }
            match reading[i].read(limits, &access) {
                Ok(None) => i += 1,
                Ok(Some(http)) => {
                    let mut input = reading.swap_remove(i);
                    if http.path == "/v1/admin/swap" {
                        let started = Instant::now();
                        let result = if swap.is_some() { Err("model swap already in progress".into()) }
                            else { admin.ok_or_else(|| "model administration is disabled".to_owned())
                                .and_then(|a| a.path(&http.body, &input.wire)) };
                        match result {
                            Ok(path) => swap = Some(Swap { path, started, wire: input.wire }),
                            Err(error) => {
                                if input.wire.response(400, &protocol::error_json("invalid_request_error", &error), limits).is_ok() {
                                    input.responding = true; reading.push(input);
                                }
                            }
                        }
                        continue;
                    }
                    if http.path == "/v1/cancel" {
                        let (status, body) = match protocol::parse_cancel(&http.body) {
                            Ok(key) if !cancellations.contains(&key) && cancellations.len() >= config.connections =>
                                (503, protocol::error_json("overloaded", "pending cancellation capacity reached")),
                            Ok(key) => {
                                if !cancellations.contains(&key) {
                                    cancellations.push(key);
                                }
                                (200, "{\"cancelled\":true}".to_owned())
                            }
                            Err(error) => (400, protocol::error_json("invalid_request_error", &error)),
                        };
                        if input.wire.response(status, &body, limits).is_ok() {
                            input.responding = true;
                            reading.push(input);
                        }
                        continue;
                    }
                    if http.path == "/v1/kv/release" {
                        let (status, body) = match protocol::parse_checkpoint_release(&http.body) {
                            Ok(key) => {
                                if let Some(key) = key { cache.release_checkpoint(&key, http.authorization.as_deref()); }
                                (200, "{\"released\":true}".to_owned())
                            }
                            Err(error) => (400, protocol::error_json("invalid_request_error", &error)),
                        };
                        if input.wire.response(status, &body, limits).is_ok() {
                            input.responding = true; reading.push(input);
                        }
                        continue;
                    }
                    let service_body = if http.path == "/v1/capabilities" && swap.is_none() {
                        Some(domain.capabilities(model, engine.context_limit(), limits, http.authorization.as_deref(),
                            config.sharing.prefix_cache_per_api_key, config.prefix_cache && cache_limits.retained_bytes != 0,
                            handles, cache_limits))
                    } else { model_response(&http.path, if swap.is_some() { "draining" } else { "ready" },
                        model, active.len(), queued.len(), engine.memory_bytes().map(|m| m.0), limits, engine.context_limit(),
                        Some((config.draft, limits.next_mtp_depth.map_or(config.mtp_depth, |d| config.mtp_depth.min(d)))), config.prefix_cache.then(|| cache.status_json())) };
                    if let Some(body) = service_body {
                        if input.wire.response(if http.path == "/cors" { 204 } else if swap.is_some() { 503 } else { 200 }, &body, limits).is_ok() {
                            input.responding = true;
                            reading.push(input);
                        }
                    } else if swap.is_some() {
                        if input.wire.response(503, &protocol::error_json("model_swapping", "retry when ready"), limits).is_ok() {
                            input.responding = true; reading.push(input);
                        }
                    } else if queued.len() + active.len() + preparing >= config.connections {
                        engine_model::events::note("admission_refused", format_args!("reason=connections queued={} active={} preparing={preparing} connections={}",
                            queued.len(), active.len(), config.connections));
                        if input.wire.response(503, &protocol::error_json("overloaded", "connection capacity reached"), limits).is_ok() {
                            input.responding = true;
                            reading.push(input);
                        }
                    } else {
                        prepare_tx.send((http, input.wire)).map_err(|_| "request preparation worker stopped")?;
                        preparing += 1;
                    }
                }
                Err(error) => {
                    let mut input = reading.swap_remove(i);
                    let (status, body) = crate::auth::request_error(&error);
                    if input.wire.response(status, &body, limits).is_ok()
                    {
                        input.responding = true;
                        reading.push(input);
                    }
                }
            }
        }
        while let Ok(prepared) = prepare_rx.try_recv() {
            preparing -= 1;
            let prepared = prepared.and_then(|prepared| {
                let Prepared::Generation(mut job) = prepared else { return Ok(prepared); };
                // An opt-in prefix cache serves only requests that ask (cache:true or a
                // prompt_cache_key); every other request runs as cache:false, main's path.
                if engine.prefix_cache_opt_in() && !job.request.cache && job.request.prompt_cache_key.is_none() {
                    job.request.prefix_cache = false;
                }
                // Resolve the startup default before validation, funding and
                // Session::configure_request so the adapter and echo agree.
                if let Some(resident) = limits.next_mtp_depth {
                    let depth = *job
                        .request
                        .mtp_depth
                        .get_or_insert(config.mtp_depth.min(resident));
                    job.request.prompt_lookup &= depth > 0;
                }
                if queued.iter().any(|j| j.cancel == job.cancel)
                    || active.iter().any(|j| j.ready.cancel == job.cancel) {
                    Err((job.wire, "cancellation capability already in use".into()))
                } else if let Err(error) = engine.validate_request(&job.prompt, &job.request) {
                    Err((job.wire, error))
                } else if job.request.prompt_lookup
                    && reservation_for(&job, job.reservation_tokens(output_window), false, None)
                        .is_none_or(|bytes| bytes > config.active_bytes)
                {
                    Err((
                        job.wire,
                        "lookup snapshot reservation exceeds session memory pool".into(),
                    ))
                } else {
                    Ok(Prepared::Generation(job))
                }
            });
            match prepared {
                Ok(Prepared::Generation(mut job)) => {
                    job.request.priority = config.sharing.priority(job.authorization.as_deref(), job.request.priority, limits.api_key.is_some());
                    if !config.sharing.prefix_cache_per_api_key {
                        job.scope = crate::cache::namespace(None, job.request.cache_salt.as_deref());
                    }
                    queued.push(job);
                }
                Ok(Prepared::Control(mut wire, body)) => {
                    if wire.control_response(body, limits).is_ok() {
                        reading.push(Reading { wire, data: Vec::new(), head: None,
                            deadline: Instant::now(), responding: true });
                    }
                }
                Err((mut wire, error)) => {
                    if wire.response(400, &protocol::error_json("invalid_request_error", &error), limits).is_ok() {
                        reading.push(Reading { wire, data: Vec::new(), head: None,
                            deadline: Instant::now(), responding: true });
                    }
                }
            }
        }
        if let Some(pending) = &swap {
            for mut job in queued.drain(..) {
                if job.wire.response(503, &protocol::error_json("model_swapping", "request cancelled for model swap"), limits).is_ok() {
                    reading.push(Reading { wire: job.wire, data: Vec::new(), head: None,
                        deadline: Instant::now(), responding: true });
                }
            }
            if pending.started.elapsed() >= Duration::from_secs(1) {
                for mut job in active.drain(..) {
                    if job.error(RequestError::Swapping, "request cancelled for model swap", limits).is_ok() {
                        reading.push(Reading { wire: job.ready.wire, data: Vec::new(), head: None,
                            deadline: Instant::now(), responding: true });
                    }
                }
            }
            if active.is_empty() {
                for input in &mut reading {
                    if !input.responding {
                        if input.wire.response(503, &protocol::error_json("model_swapping", "retry when ready"), limits).is_err() {
                            continue;
                        }
                        input.responding = true;
                    }
                }
                if reading.is_empty() && preparing == 0 && retired.is_empty() { return Ok(swap); }
            }
        }
        // Apply after intake so a request admitted in the same turn cannot
        // overtake cancellation. Keep a terminal response for connected clients.
        let mut i = 0;
        while i < queued.len() {
            if queued[i].wire.connected(queued[i].request.stream).is_err() { queued.remove(i); continue; }
            if !cancellations.contains(&queued[i].cancel) && Instant::now() < queued[i].deadline {
                i += 1; continue;
            }
            let mut job = queued.remove(i);
            next_id += 1;
            let error = if cancellations.contains(&job.cancel) { RequestError::Cancelled } else { RequestError::Deadline };
            if job.cancel(next_id, error, limits, &domain, config.mtp_depth).is_ok() {
                reading.push(Reading {
                    wire: job.wire,
                    data: Vec::new(),
                    head: None,
                    deadline: Instant::now(),
                    responding: true,
                });
            }
        }
        let mut i = 0;
        while i < active.len() {
            if cancellations.contains(&active[i].ready.cancel) {
                let mut job = active.remove(i);
                if job.cancel(RequestError::Cancelled, limits).is_ok() {
                    reading.push(Reading { wire: job.ready.wire, data: Vec::new(), head: None,
                        deadline: Instant::now(), responding: true });
                }
                continue;
            }
            let job = &mut active[i];
            if job.ready.wire.connected(job.ready.request.stream).is_err()
                || job.flush(limits).is_err()
            {
                job.cancelled("connection_lost");
                active.remove(i);
                continue;
            }
            i += 1;
        }

        if preparing == 0 { cancellations.clear(); }
        if !retired.is_empty() { continue; }

        let mut i = 0;
        while i < active.len() {
            let job = &active[i];
            let tokens = job.ready.reservation_tokens(job.generation.usage.completion_tokens
                .saturating_add(output_window));
            if tokens <= job.reserved_tokens { i += 1; continue; }
            let mut bytes = reservation_for(&job.ready,tokens,job.reservation_plain,None).ok_or("missing request reservation")?;
            let mut delta = bytes.saturating_sub(job.generation.state.reserved_session_bytes);
            let (foreground_reserved, _) = reservations(active.iter().filter(|j| j.ready.request.priority == 255)
                .map(|j| &j.generation.state));
            let ceiling = if job.ready.request.priority == 255 { config.active_bytes }
                else { config.active_bytes - foreground_bytes.saturating_sub(foreground_reserved) };
            // Shared physical pages are funded once; live growth stays private.
            let mut reserved = reservations(active.iter().map(|j| &j.generation.state)).0;
            if !active[i].reservation_plain && reserved.checked_add(delta).is_none_or(|n| n > ceiling) {
                // Optional draft pages yield to canonical target growth. This
                // request remains plain even if its cohort subsequently drains.
                active[i].generation.prefer_plain_shared_prefix();
                active[i].generation.state.session.disable_model_draft();
                bytes = reservation_for(&active[i].ready,tokens,true,None).ok_or("missing plain reservation")?;
                active[i].generation.state.reserved_session_bytes = reservation_for(&active[i].ready,active[i].reserved_tokens,true,None).ok_or("missing plain reservation")?;
                active[i].reservation_plain = true;
                reserved = reservations(active.iter().map(|j| &j.generation.state)).0;
                delta = bytes.saturating_sub(active[i].generation.state.reserved_session_bytes);
                engine_model::events::note_line("draft_memory_pressure", &format!("draft_memory_pressure id={} action=plain reserved_bytes={bytes}", active[i].id));
            }
            if let Some(next_reserved) = reserved.checked_add(delta).filter(|&n| n <= ceiling) {
                // Reclaim unfunded cache pages before any backend allocation.
                // Growth preserves active_reserved + cache_unfunded <= pool.
                let (_, pages) = reservations(active.iter().map(|j| &j.generation.state));
                let hold = engine.cache_hold_bytes() + active.iter().map(|j| request_hold(&j.ready)).sum::<usize>();
                cache.rebalance_within(config.session_budget_bytes.checked_sub(next_reserved)
                    .ok_or_else(|| crate::budget::session_refusal(next_reserved as u128, config.session_budget_bytes, "live reservations; CONTEXT not supplied to cache balancing", "the scheduler session pool"))?
                    .saturating_sub(hold), engine.cache_free_physical_bytes(next_reserved)?, pages, Instant::now());
                active[i].generation.state.reserved_session_bytes = bytes;
                active[i].reserved_tokens = tokens;
                active[i].generation.state.session.reserve_model_draft(tokens);
                i += 1;
            } else {
                let mut job = active.remove(i);
                engine_model::events::note_line("request_capacity_exhausted", &format!("request capacity exhausted id={} completion_tokens={}", job.id, job.generation.usage.completion_tokens));
                if job.error(RequestError::Capacity, &crate::budget::refusal(reserved as u128 + delta as u128, ceiling, &format!("weights/rank already loaded; next reservation/session {bytes} bytes; growth {delta} bytes + live reservations {reserved} bytes; {} active sessions; CONTEXT={}; OUTPUT={}", active.len()+1, limits.context_tokens, job.ready.request.max_tokens), "the scheduler active pool, capped by owner-priority headroom; retained cache reclaimed first", crate::budget::SERVE), limits).is_ok() {
                    reading.push(Reading { wire: job.ready.wire, data: Vec::new(), head: None,
                        deadline: Instant::now(), responding: true });
                }
            }
        }
        while active.len() < config.active {
            let Some(oldest) = queued.first() else { break; };
            // VADE_AB_COHORTS: arms run in arrival order, one cohort at a time; the
            // queue head of another arm waits until every active session has ended,
            // so a module switch still never meets a live session of the other arm.
            if limits.ab_cohorts && active.iter().any(|a| a.ready.request.arm_x != oldest.request.arm_x) { break; }
            // Collect the CPU-prepared cohort before its first GPU pass.
            // Decode and control traffic keep running during this bounded wait.
            if preparing > 0 && oldest.enqueued.elapsed() < Duration::from_millis(100)
                && engine.shared_batch() {
                break;
            }
            // The idle maintenance above may predate this request; the cache balance below
            // can poll memory pressure and the forks can allocate.
            engine.admission_started();
            let (reserved, pages) = reservations(active.iter().map(|j| &j.generation.state));
            let hold = engine.cache_hold_bytes() + active.iter().map(|j| request_hold(&j.ready)).sum::<usize>();
            cache.rebalance_within(config.session_budget_bytes.checked_sub(reserved)
                .ok_or_else(|| crate::budget::session_refusal(reserved as u128, config.session_budget_bytes, "live reservations; CONTEXT not supplied to cache balancing", "the scheduler session pool"))?
                .saturating_sub(hold), engine.cache_free_physical_bytes(reserved)?, pages.clone(), Instant::now());
            let shared_charge = |allocations: Vec<(usize, usize)>| allocations.into_iter()
                .map(|(id, size)| if pages.contains(&id) { size } else { 0 }).sum::<usize>();
            let (foreground_reserved, _) = reservations(active.iter().filter(|j| j.ready.request.priority == 255)
                .map(|j| &j.generation.state));
            let background_limit = config.active_bytes - foreground_bytes.saturating_sub(foreground_reserved);
            let backgrounds = active
                .iter()
                .filter(|j| j.ready.request.priority != 255)
                .count();
            // Backfill may overlap older work, but must leave enough memory
            // and slots to coexist with the oldest waiter once that work ends.
            // Otherwise a stream of small requests can keep it queued forever.
            let checkpoint = |j: &Ready| {
                if !handles || !j.request.prefix_cache { return None; }
                let key = j.request.kv_handle.as_ref()?.as_ref().ok()?;
                cache.checkpoint_state(key, j.handle_scope.as_deref(), &j.prompt, Instant::now())
            };
            let cached_prefix = |j: &Ready| {
                if let Some((_, retain)) = checkpoint(j) { return retain; }
                if !config.prefix_cache || !j.request.prefix_cache { return 0; }
                let running = active.iter().filter(|a| a.automatic_cache && a.ready.scope == j.scope
                    && a.prefill_rows == 0 && a.long_prefill.is_none())
                    .map(|a| shared_prefix(&a.generation.state.tokens[..a.generation.state.tokens.len().min(a.ready.prompt.len())], &j.prompt))
                    .max().unwrap_or(0);
                running.max(cache.reusable_tokens(j.scope.as_deref(), &j.prompt))
            };
            let incremental = |j: &Ready| {
                let tokens = j.reservation_tokens(output_window);
                let full = reservation_for(j,tokens,shared_plain(j, config, active.iter().map(|a| &a.ready).chain(queued.iter())),Some(cached_prefix(j)))?;
                let shared = if let Some((state, retain)) = checkpoint(j) {
                    shared_charge(state.session.prefix_allocations(retain).unwrap_or_default())
                } else if config.prefix_cache && j.request.prefix_cache {
                    let running = active.iter().filter(|a| a.automatic_cache && a.ready.scope == j.scope
                        && a.prefill_rows == 0 && a.long_prefill.is_none())
                        .map(|a| shared_charge(a.generation.state.session.prefix_allocations(shared_prefix(&a.generation.state.tokens[..a.generation.state.tokens.len().min(a.ready.prompt.len())], &j.prompt))
                            .unwrap_or_default()))
                        .max().unwrap_or(0);
                    running.max(shared_charge(cache.reusable_allocations(j.scope.as_deref(), &j.prompt)))
                } else { 0 };
                Some(full.saturating_sub(shared))
            };
            // A retaining request that cannot fit alone must not block the queue.
            // Wait for active owners first: they can publish a cheaper cached prefix.
            // Test both the cached bound and the existing plain fallback without live
            // reservations, so temporary contention does not become a refusal.
            if active.is_empty() && retain_prefix(oldest)
                && incremental(oldest).is_none_or(|bytes| bytes > config.active_bytes)
                && reservation_for(oldest,oldest.reservation_tokens(output_window),true,None)
                    .is_none_or(|bytes| bytes > config.active_bytes) {
                let mut job = queued.remove(0);
                let error = "request and prefix snapshot reservation exceed the active session pool";
                if job.wire.response(503, &protocol::error_json("capacity_exhausted", error), limits).is_ok() {
                    reading.push(Reading { wire: job.wire, data: Vec::new(), head: None,
                        deadline: Instant::now(), responding: true });
                }
                continue;
            }
            let waiting_bytes = reservation_for(oldest,oldest.reservation_tokens(output_window),shared_plain(oldest, config, active.iter().map(|a| &a.ready).chain(queued.iter())),None)
                .or_else(|| retain_prefix(oldest).then_some(usize::MAX))
                .ok_or("missing request reservation")?;
            let (younger_count, younger_bytes, younger_backgrounds, younger_foreground) = active.iter()
                .filter(|j| j.ready.enqueued >= oldest.enqueued)
                .fold((0, 0, 0, 0), |(count, bytes, backgrounds, foreground), j|
                    (count + 1, bytes + j.generation.state.reserved_session_bytes,
                        backgrounds + usize::from(j.ready.request.priority != 255),
                        foreground + if j.ready.request.priority == 255 { j.generation.state.reserved_session_bytes } else { 0 }));
            let backfill_limit = config.active_bytes - foreground_bytes.saturating_sub(younger_foreground
                + if oldest.request.priority == 255 { waiting_bytes } else { 0 });
            let candidate = queued
                .iter()
                .enumerate()
                .filter(|(_, j)| {
                    // Short hybrid prompts share pooled projections. Long
                    // prompts instead wait for the owner's recurrent checkpoint.
                    if checkpoint(j).is_some() || !config.prefix_cache || !j.request.prefix_cache
                        || engine.native_draft_policy() && j.prompt.len() <= shared_prompt_limit { return true; }
                    let committed = active.iter().filter(|a| a.automatic_cache && a.ready.scope == j.scope
                        && a.prefill_rows == 0 && a.long_prefill.is_none())
                        .map(|a| shared_prefix(&a.generation.state.tokens[..a.generation.state.tokens.len().min(a.ready.prompt.len())], &j.prompt)).max().unwrap_or(0);
                    let available = committed.max(cache.reusable_tokens(j.scope.as_deref(), &j.prompt));
                    // A running owner computes the common prefix once. Waiters
                    // take its committed pages at the first divergence boundary.
                    !active.iter().any(|a| a.automatic_cache && a.ready.scope == j.scope
                        && shared_prefix(&a.ready.prompt, &j.prompt) > available
                            + shared_prompt_limit)
                })
                .filter(|(i, _)| !limits.ab_cohorts || *i == 0)
                .filter(|(_, j)| j.request.priority == 255 || backgrounds < public_slots)
                .filter(|(_, j)| incremental(j)
                    .and_then(|bytes| reserved.checked_add(bytes)).is_some_and(|bytes|
                        bytes <= if j.request.priority == 255 { config.active_bytes } else { background_limit }))
                .filter(|(i, j)| *i == 0 || (younger_count + 2 <= config.active
                    && younger_backgrounds + usize::from(j.request.priority != 255)
                        + usize::from(oldest.request.priority != 255) <= public_slots
                    && reservation_for(j,j.reservation_tokens(output_window),false,None)
                        .and_then(|bytes| younger_bytes.checked_add(bytes))
                        .and_then(|bytes| bytes.checked_add(waiting_bytes))
                        .is_some_and(|bytes| bytes <= if j.request.priority == 255 { config.active_bytes } else { backfill_limit })))
                // At equal priority the eligible queue head wins. Among other
                // backfill candidates affinity precedes age, within slot/byte bounds.
                .max_by_key(|(i, j)| (j.request.priority, *i == 0,
                    config.prefix_cache && j.request.prefix_cache && j.request.prompt_cache_key.as_ref().is_some_and(|key|
                        active.iter().any(|a| a.ready.affinity_scope == j.affinity_scope && a.ready.request.prefix_cache
                            && a.ready.request.prompt_cache_key.as_ref() == Some(key))), std::cmp::Reverse(*i)))
                .map(|(i, _)| i);
            let (index, pressure_plain) = if let Some(index) = candidate { (index, false) } else {
                // Prefix followers also wait here for their owner to prefill.
                // Only a failed memory bound permits dropping drafting.
                let ceiling = if oldest.request.priority == 255 { config.active_bytes } else { background_limit };
                if incremental(oldest).and_then(|bytes| reserved.checked_add(bytes))
                    .is_some_and(|n| n <= ceiling) { break; }
                let bytes = match reservation_for(oldest,oldest.reservation_tokens(output_window),true,None) {
                    Some(bytes) => bytes,
                    None if retain_prefix(oldest) => break,
                    None => return Err("missing plain reservation".into()),
                };
                if (oldest.request.priority == 255 || backgrounds < public_slots)
                    && reserved.checked_add(bytes).is_some_and(|n|
                        n <= if oldest.request.priority == 255 { config.active_bytes } else { background_limit }) {
                    (0, true)
                } else { break; }
            };
            // Reserve before forking: compact tails may allocate during
            // truncate, so LRU reclamation must precede any backend allocation.
            // An adapter may explicitly fund retained snapshots inside the
            // incoming full reservation. Preserve only a verified candidate's
            // identities, without deducting their bytes from that reservation.
            let next_reserved = (if pressure_plain {
                reservation_for(&queued[index],queued[index].reservation_tokens(output_window),true,None)
            } else { incremental(&queued[index]) })
                .and_then(|bytes| reserved.checked_add(bytes)).ok_or("request reservation overflow")?;
            let mut funded = pages.clone();
            let incoming = &queued[index];
            if config.prefix_cache && incoming.request.prefix_cache {
                funded.extend(cache.funded_candidate_allocations(incoming.scope.as_deref(),
                    &incoming.prompt, incoming.request.session.as_ref(),
                    !engine.native_draft_policy() || incoming.prompt.len() > shared_prompt_limit,
                    incoming.request.kv_handle.as_ref().filter(|_| handles).and_then(|key| key.as_ref().ok()),
                    incoming.handle_scope.as_deref(), Instant::now()).into_iter().map(|(id, _)| id));
            }
            let hold = hold + request_hold(&queued[index]);
            cache.rebalance_within(config.session_budget_bytes.checked_sub(next_reserved)
                .ok_or_else(|| crate::budget::session_refusal(next_reserved as u128, config.session_budget_bytes, "live reservations; CONTEXT not supplied to cache balancing", "the scheduler session pool"))?
                .saturating_sub(hold), engine.cache_free_physical_bytes(next_reserved)?, funded, Instant::now());
            let plain = pressure_plain
                || engine.discard_contended_draft() && active.iter().any(|job| !job.done)
                || shared_plain(&queued[index], config, active.iter().map(|a| &a.ready).chain(queued.iter()));
            let mut ready = queued.remove(index);
            let tokens = ready.reservation_tokens(output_window);
            let mut request_bytes = reservation_for(&ready,tokens,plain,None)
                .ok_or("missing request reservation")?;
            let cache_started = Instant::now();
            let profile = std::env::var("SPARK_PROFILE").as_deref() == Ok("1");
            if profile && config.prefix_cache && ready.request.prefix_cache {
                cache.profile_match(next_id + 1, ready.scope.as_deref(), &ready.prompt);
            }
            // Preserve pooled short-prompt prefill; long hybrid prompts can
            // reuse the committed recurrence and KV through the same cache.
            let automatic_cache = config.prefix_cache && ready.request.prefix_cache
                && (!engine.native_draft_policy() || ready.prompt.len() > shared_prompt_limit);
            let mut state = None;
            let mut kv = (ready.request.kv_checkpoint || ready.request.kv_handle.is_some()).then(|| crate::usage::Kv {
                cache_domain: domain.key(ready.authorization.as_deref()), reuse: "not_requested", reused_tokens: 0,
                retention: if !ready.request.kv_checkpoint { "not_requested" }
                    else if !ready.request.prefix_cache { "cache_disabled" }
                    else if !handles { "unsupported" } else { "pending" }, checkpoint: None,
            });
            if let Some(handle) = ready.request.kv_handle {
                let (checkpoint, reason) = if !ready.request.prefix_cache { (None, "cache_disabled") }
                    else if !handles { (None, "unsupported") }
                    else if let Ok(handle) = handle {
                        cache.fork_checkpoint(&handle, ready.handle_scope.as_deref(), &ready.prompt, Instant::now())
                    } else { (None, "invalid_handle") };
                kv.as_mut().unwrap().reuse = reason;
                if checkpoint.is_some() { state = checkpoint; }
            }
            if state.is_none() && ready.request.cache {
                if let Some(key) = ready.request.session.as_ref() {
                    let forked = cache.fork_scoped(key, ready.scope.as_deref(), &ready.prompt, Instant::now());
                    fork_failed("session", ready.prompt.len(), &forked.failed);
                    state = forked.state;
                }
            }
            if automatic_cache && kv.is_none_or(|kv| kv.reuse != "hit") {
                let forked = cache.fork_automatic(ready.scope.as_deref(), &ready.prompt, Instant::now());
                fork_failed("automatic", ready.prompt.len(), &forked.failed);
                let cached = forked.state;
                if cached.as_ref().is_some_and(|s| s.tokens.len() > state.as_ref().map_or(0, |s| s.tokens.len())) {
                    state = cached;
                }
                for source in active.iter().filter(|a| a.automatic_cache && a.ready.scope == ready.scope
                    && a.prefill_rows == 0 && a.long_prefill.is_none()) {
                    if shared_prefix(&source.generation.state.tokens, &ready.prompt).min(source.ready.prompt.len()) > state.as_ref().map_or(0, |s| s.tokens.len()) {
                        match source.generation.state.fork_prefix_limit(&ready.prompt, source.ready.prompt.len()) {
                            Ok(Some(fork)) => state = Some(fork),
                            Ok(None) => {}
                            Err(error) => fork_failed("active", ready.prompt.len(), &[error]),
                        }
                    }
                }
            }
            let reused = state.as_ref().map_or(0, |s| s.tokens.len()); if automatic_cache || ready.request.cache && ready.request.session.is_some() { cache.count_lookup(reused); }
            if reused == 0 && (automatic_cache || ready.request.cache && ready.request.session.is_some()) {
                let (reason, scoped) = cache.miss_reason(ready.scope.as_deref());
                engine_model::events::note("prefix_cache_miss", format_args!("prompt_tokens={} reason={reason} scoped_entries={scoped}", ready.prompt.len()));
            }
            // Recheck the actual fork: eviction or a different retained owner
            // can leave less cached work than the admission estimate.
            if plain {
                request_bytes = reservation_for(&ready,tokens,true,Some(reused))
                    .ok_or("missing cached request reservation")?;
            }
            if profile {
                eprintln!("request_cache_profile reused_tokens={reused} lookup_ms={:.3}", cache_started.elapsed().as_secs_f64()*1000.0);
            }
            let key = if ready.request.cache || automatic_cache {
                Some(if reused > 0 && !ready.request.fork && ready.request.session.is_some() {
                    ready.request.session.unwrap()
                } else {
                    SessionKey::generate()?
                })
            } else {
                None
            };
            engine.active_cohort(active.iter().filter(|j| !j.done).count() + 1);
            let state = match state {
                Some(mut state) => {
                    state.reserved_session_bytes = request_bytes;
                    Ok(state)
                }
                None => engine.session_for_admission(&ready.request, tokens).map(|session| CachedSession {
                    checkpoint_lease: None,
                    session,
                    tokens: Vec::new(),
                    next_logits: Vec::new(),
                    reserved_session_bytes: request_bytes,
                }),
            };
            let state = state.and_then(|mut state| {
                state.session.configure_request(&ready.request)?;
                // Include explicit checkpoints even without a conversation key.
                state.session.set_cache_retention(retain_prefix(&ready));
                state.session.reserve_model_draft(tokens);
                if plain || ready.request.draft.unwrap_or(config.draft) == protocol::DraftMode::Plain {
                    state.session.disable_model_draft();
                } else {
                    engine.restore_model_draft(&mut state.session, ready.prompt.len())?;
                }
                Ok(state)
            });
            let state = match state {
                Ok(state) => state,
                Err(error) => {
                    engine_model::events::note("session_unavailable", format_args!("prompt_tokens={} error={error}", ready.prompt.len()));
                    let _ = ready.wire.response(
                        503,
                        &protocol::error_json("engine_error", "session unavailable"),
                        limits,
                    );
                    reading.push(Reading {
                        wire: ready.wire,
                        data: Vec::new(),
                        head: None,
                        deadline: Instant::now(),
                        responding: true,
                    });
                    continue;
                }
            };
            // The selected cached fork can differ from the admission estimate.
            // Account its actual physical identities before accepting the stream.
            let shared = shared_charge(state.session.prefix_allocations(state.tokens.len()).unwrap_or_default());
            if reserved.checked_add(request_bytes.saturating_sub(shared)).is_none_or(|bytes|
                bytes > if ready.request.priority == 255 { config.active_bytes } else { background_limit }) {
                queued.insert(index, ready);
                break;
            }
            let active_reserved_bytes = balance_cache(&mut cache, config.session_budget_bytes,
                active.iter().map(|j| &j.generation.state)
                    .chain(std::iter::once(&state)),
                engine.cache_hold_bytes() + active.iter().map(|j| request_hold(&j.ready)).sum::<usize>()
                    + request_hold(&ready), |reserved| engine.cache_free_physical_bytes(reserved))?;
            let capability = key.filter(|_| ready.request.cache)
                .map(|k| {
                    format!(
                        "X-Session-Capability: {}\r\n",
                        k.expose()
                    )
                })
                .unwrap_or_default();
            let seed = ready.request.sampling.seed;
            let cancel = ready.cancel.expose();
            let reuse_header = if config.sharing.report_prefix_reuse {
                format!("X-Prefix-Reused-Tokens: {reused}\r\n")
            } else { String::new() };
            if ready.request.stream && ready.wire.push(&format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\nX-Accel-Buffering: no\r\nX-Sampling-Seed: {seed}\r\nX-Cancel-Capability: {cancel}\r\n{reuse_header}{capability}{}\r\n", ready.wire.cors), limits, false).is_err() {
                continue;
            }
            // A writable new foreground socket must not wait an extra model
            // step merely because its HTTP headers were admitted this round.
            if ready.wire.flush().is_err() {
                continue;
            }
            next_id += 1;
            eprintln!("request_admitted id={next_id} prompt_tokens={} queue_ms={:.3} reserved_bytes={request_bytes} prefix_reused_tokens={reused} active={} active_reserved_bytes={active_reserved_bytes} cache_retained_bytes={} cache_unfunded_bytes={} admitted_at_ms={:.3}",
                ready.prompt.len(), ready.enqueued.elapsed().as_secs_f64() * 1000.0,
                active.len() + 1, cache.retained_bytes(), cache.unfunded_bytes(), admission_clock.elapsed().as_secs_f64() * 1000.0);
            // Queue wait must not consume a thinking reply's generation budget.
            ready.deadline = Instant::now() + limits.generation_timeout;
            let mut generation = Generation::new(
                    state,
                    ready.prompt.len(),
                    spark_engine::sample::Sampler::new(ready.request.sampling)?,
                );
            generation.ignore_eos = ready.request.ignore_eos;
            generation.mtp_depth = engine
                .greedy_draft_only()
                .then_some(ready.request.mtp_depth.unwrap_or(config.mtp_depth));
            generation.verify_rows = ready
                .request
                .prompt_lookup
                .then_some(ready.request.prompt_lookup_max_draft + 1)
                // VADE_MTP_SAMPLED: sampled requests have no prompt lookup (greedy's 16-row budget);
                // their rows follow the resident depth so VADE_MTP_ADAPT can draft it.
                .or(limits.next_mtp_depth.filter(|_| engine.sampled_draft()
                    && !spark_engine::draft::plain_greedy(ready.request.sampling)).map(|d| d + 1));
            generation.usage.kv = kv;
            // The smaller reservation commits this request to plain even if its peers leave.
            if plain { generation.prefer_plain_shared_prefix(); }
            let mut mode = ready.request.draft.unwrap_or(config.draft);
            if engine.greedy_draft_only() && !engine.sampled_draft() && !spark_engine::draft::plain_greedy(ready.request.sampling) {mode=protocol::DraftMode::Plain;}
            if mode == protocol::DraftMode::Auto && !engine.automatic_draft_allowed(active.len() + 1) {
                mode = protocol::DraftMode::Plain;
            }
            if engine.native_draft_policy() {
                draft_policy.update(active.iter().filter(|j| !j.done).count()+1,Instant::now(),config.draft==protocol::DraftMode::Auto,engine.automatic_draft_limit());
                if mode == protocol::DraftMode::Auto { generation.state.session.automatic_draft(ready.prompt.len(),ready.request.max_tokens); }
                mode = if mode == protocol::DraftMode::Auto && ready.request.max_tokens < engine.automatic_draft_min_output() {
                    generation.state.session.disable_model_draft();
                    protocol::DraftMode::Plain
                } else { draft_policy.resolve(mode) };
            }
            let mode = generation.enable_draft(mode,ready.request.sampling);
            generation.seed_draft(draft_bank.propose(ready.handle_scope.as_deref(), ready.request.session, config.draft_global_bank, &ready.prompt));
            let decoder = tokenizer.map(|t| if ready.parser.is_some() { t.chat_decoder() } else { t.decoder() });
            let mut echo = protocol::SamplingEcho::new(&ready.request, mode, if mode == protocol::DraftMode::Plain && engine.greedy_draft_only() {0} else {generation.mtp_depth.unwrap_or(config.mtp_depth)}).with_metadata(limits);
            let mut prefill = crate::usage::Prefill::new(ready.prompt.len(), reused, cache_started);
            if reused > 0 { prefill.cache_restore = Some(cache_started.elapsed()); }
            if reused == ready.prompt.len() {
                prefill.commit(0, Instant::now(), true);
            }
            echo.prefill = Some(prefill);
            echo.log(next_id);
            let mut job = Active {
                generation,
                long_prefill: None,
                prefill_started: false,
                ready,
                key,
                automatic_cache,
                decoder,
                id: next_id,
                created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default().as_secs(),
                response: http::Completion::default(),
                content_bytes: 0,
                echo,
                stop: protocol::StopBuffer::default(),
                tool_index: 0,
                output_failed: false,
                done: false,
                terminal_logged: false,
                last_tick: tick,
                prefill_rows: 0,
                admitted: Instant::now(),
                usage_clock: None,
                memory_reported_tokens: 0,
                profile,
                first_token_sent: false,
                first_token_generated: false,
                reserved_tokens: tokens,
                reservation_plain: plain,
            };
            if job.ready.request.include_usage_timing {
                let mut clock = crate::usage::Clock::new(cache_started);
                if reused == job.ready.prompt.len() { clock.prefilled(); }
                job.usage_clock = Some(clock);
            }
            // An exact cached-logit hit produces no Prefill event.
            if reused == job.ready.prompt.len() { job.checkpoint(&mut cache); }
            if job.ready.request.stream {
                if job.ready.output.is_some() {
                    job.update_usage();
                    let events = job.ready.output.as_mut().unwrap().start(&job.ready.model, job.id, job.created, job.generation.usage);
                    if job.ready.wire.push(&events, limits, false).and_then(|()| job.ready.wire.flush()).is_err() {
                        job.cancelled("write_failed");
                        continue;
                    }
                }
            }
            active.push(job);
        }
        let concurrent_jobs = active.iter().filter(|j| !j.done).count();
        engine.active_cohort(concurrent_jobs);
        draft_policy.update(active.iter().filter(|j| !j.done).count(), Instant::now(),
            config.draft == protocol::DraftMode::Auto || active.iter().any(|j| j.ready.request.draft == Some(protocol::DraftMode::Auto)), engine.automatic_draft_limit());
        let mut jobs = Vec::new();
        for i in (0..active.len()).rev() {
            if !active[i].done && (active[i].ready.wire.bytes.is_empty()
                || Instant::now() >= active[i].ready.deadline) { jobs.push(active.remove(i)); }
        }
        if jobs.is_empty() { std::thread::park_timeout(Duration::from_millis(1)); continue; }
        tick += 1;
        engine.profile_tick_begin(tick);
        let foreground = jobs.iter().filter(|j| j.ready.request.priority > 0)
            .max_by_key(|j| (j.ready.request.priority, std::cmp::Reverse(j.last_tick))).map(|j| j.id);
        let mut budget = if foreground.is_some() { config.foreground_rows } else { config.background_rows };
        if !engine.shared_batch() { budget = 1; }
        if engine.native_draft_policy() {
            let decoders = jobs.iter().filter(|j| !j.done && j.ready.wire.bytes.is_empty() && j.generation.state.tokens.len() >= j.ready.prompt.len()).count();
            if decoders > 0 { budget = budget.min(engine.decode_row_limit(decoders)); }
        }
        if let Some(rows) = engine.tick_row_limit() { budget = budget.min(rows); }
        // Short cached suffixes share the same prompt batch as short cold
        // prompts. Preserve Session prefill while layer work is uncommitted.
        let chunk_rows = engine.shared_prefill_chunk_rows();
        // Count prompt-ready jobs first: a lone long prompt must not be
        // trapped in tile-sized cohort chunks (see prompt_uses_cohort).
        let prompt_ready_jobs = jobs.iter().filter(|j| !j.done && j.ready.wire.bytes.is_empty()
            && j.prefill_rows == 0 && j.long_prefill.is_none()
            && j.ready.prompt.len() <= shared_prefill_max_tokens
            && j.generation.state.tokens.len() < j.ready.prompt.len()
            && prompt_cohort_candidate(j.ready.prompt.len() - j.generation.state.tokens.len(),
                j.generation.state.tokens.len(), shared_prefill_limit, chunk_rows)).count();
        let shared_prompt = |j: &Active<'_, '_, E, T>| j.prefill_rows == 0 && j.long_prefill.is_none()
            && j.ready.prompt.len() <= shared_prefill_max_tokens
            && prompt_uses_cohort(j.ready.prompt.len().saturating_sub(j.generation.state.tokens.len()),
                j.generation.state.tokens.len(), shared_prefill_limit, chunk_rows, prompt_ready_jobs);
        let aged = jobs.iter().filter(|j| Some(j.id) != foreground
            && (shared_prompt(j) || j.generation.state.tokens.len() >= j.ready.prompt.len())
            && tick.saturating_sub(j.last_tick) >= 4)
            .min_by_key(|j| (j.last_tick, j.id)).map(|j| j.id);
        // A single-row engine must occasionally yield the foreground row;
        // shared batches reserve a row for the oldest waiting decode.
        let promoted = if budget == 1 { aged } else { None };
        jobs.sort_by_key(|j| (Some(j.id) != promoted, Some(j.id) != foreground, j.last_tick,
            j.generation.state.tokens.len() < j.ready.prompt.len(),
            std::cmp::Reverse(j.ready.request.priority), j.id));
        let mut plans: Vec<Option<crate::generation::Work>> = (0..jobs.len()).map(|_| None).collect();
        let mut errors: Vec<Option<String>> = (0..jobs.len()).map(|_| None).collect();
        // Bound long-context attention's share when shorter decoders are ready.
        // Equal-length cohorts run every tick; priority and four-tick aging win.
        let attention_base = jobs.iter().filter(|j| j.generation.state.tokens.len() >= j.ready.prompt.len())
            .map(|j| j.generation.state.tokens.len()).min().unwrap_or(0).max(8_192);
        let due: Vec<_> = jobs.iter().map(|j| Some(j.id) == foreground
            || j.generation.state.tokens.len() < j.ready.prompt.len()
            || tick.saturating_sub(j.last_tick) >= (j.generation.state.tokens.len() / attention_base).clamp(1, 4) as u64)
            .collect();
        let rows_after: Vec<_> = (0..jobs.len()).map(|i| jobs[i+1..].iter()
            .zip(&due[i+1..]).filter(|(j, due)| **due
                && (shared_prompt(j) || j.generation.state.tokens.len() >= j.ready.prompt.len()))
            .count()).collect();
        // Verification disables cascade attention and an empty draft must
        // replay the shared prefix. Choose plain before the owner starts decode,
        // and retain that choice as the cohort drains. Explicit modes still win.
        // A prefix owner must not run one token ahead while its admitted
        // followers finish short suffixes. Joining their first decode keeps
        // the expert routes at the same output step in the shared weight pass.
        let joining_plain: Vec<_> = jobs.iter().map(|source| source.generation.usage.completion_tokens == 1
            && shared_plain(&source.ready, config,
            jobs.iter().filter(|other| !other.done && shared_prompt(other)
                && other.generation.state.tokens.len() < other.ready.prompt.len()
                && other.ready.wire.bytes.is_empty()
                && other.ready.request.priority >= source.ready.request.priority)
                .map(|other| &other.ready))).collect();
        let shared_plain: Vec<_> = jobs.iter().map(|source| shared_plain(&source.ready, config,
            jobs.iter().map(|other| &other.ready).chain(queued.iter()))).collect();
        let draft_chats = jobs.iter().filter(|job|
            job.generation.state.tokens.len() >= job.ready.prompt.len()).count();
        // Even cohort rows: every row taker of this tick gets the same cap. The sequential
        // split below hands the remainder to the last chat (C5 at depth 3: 3,3,3,3,4), so one
        // chat's deepest guess cost the whole cohort another drafter step and verify row.
        let even_rows = engine.even_cohort_rows().then(|| {
            let takers = (0..jobs.len()).filter(|&i| due[i] && (shared_prompt(&jobs[i])
                || jobs[i].generation.state.tokens.len() >= jobs[i].ready.prompt.len())).count();
            budget / takers.max(1)
        }).filter(|&rows| rows > 1);
        // Observe only complete, ready, greedy automatic cohorts. An omitted
        // request mode uses FlashNext's Strict startup default but remains
        // eligible for this opt-in chooser; an explicit Strict stays fixed.
        // Explicit modes, prompt work and priorities keep the existing policy.
        // Short terminal tails finish plainly without starting another probe.
        let mut cohort_held = None;
        let mut cohort_trial = cohort_cost.as_mut().and_then(|control| {
            if !(6..=8).contains(&jobs.len()) || jobs.len() != concurrent_jobs
                || budget < jobs.len() * 2 || foreground.is_some()
                || jobs.iter().enumerate().any(|(i, job)| shared_plain[i]
                    || joining_plain[i] || job.done || !job.ready.wire.bytes.is_empty()
                    || job.generation.state.tokens.len() < job.ready.prompt.len()
                    || job.generation.usage.completion_tokens == 0
                    || job.ready.request.max_tokens.saturating_sub(job.generation.usage.completion_tokens) < 3
                    || job.ready.request.mtp_depth == Some(0)
                    || job.ready.request.draft.map_or(
                        !matches!(config.draft, protocol::DraftMode::Auto | protocol::DraftMode::Strict),
                        |mode| mode != protocol::DraftMode::Auto)
                    || !spark_engine::draft::plain_greedy(job.ready.request.sampling))
            { control.suspend(); return None; }
            let mut ids: Vec<_> = jobs.iter().map(|job| job.id).collect();
            ids.sort_unstable();
            if due.iter().any(|due| !due) {
                cohort_held = Some(control.fairness_skip(&ids));
                return None;
            }
            let head_miss = engine.cohort_draft_head_limit().is_some_and(|limit|
                jobs.iter().any(|job| job.generation.cohort_pending_token()
                    .is_some_and(|token| token as usize >= limit)));
            let draft = control.select(&ids, head_miss);
            let before = jobs.iter().map(|j| j.generation.cohort_cost_counters())
                .fold([0; 3], |a, b| std::array::from_fn(|i| a[i] + b[i]));
            Some((draft, Instant::now(), before))
        });
        // An automatic C6..8 cohort that cannot be measured must retain
        // main's plain policy, rather than falling through to raw Auto8 MTP.
        // In particular, mixed explicit depths (including zero), prompt work,
        // paused sockets and priorities do not form a valid cost sample.
        // Explicit request modes/depths are not cost-probed; the ordinary
        // C6..8 cohort limit applies while the chooser is ineligible.
        let cohort_fallback_plain = cohort_cost.is_some() && cohort_trial.is_none() && cohort_held.is_none()
            && (6..=8).contains(&concurrent_jobs);
        if cohort_cost.is_some() {
            engine.set_cohort_choice(cohort_trial.as_ref().map(|(draft, _, _)| *draft)
                .or(cohort_held).or(cohort_fallback_plain.then_some(false)));
        }
        let cohort_setup_error = cohort_trial.as_ref().and_then(|(draft, _, _)|
            engine.prepare_cohort_path(*draft).err().map(|error| error.message));
        // Robust mode scores the steady step, not the one-time pooled-decode release
        // that prepare_cohort_path runs on a plain-to-draft switch. The 2% path
        // (VADE_C8_GUARD_ROBUST=0) keeps the timer that started before that call.
        if cohort_setup_error.is_none() && cohort_cost.as_ref().is_some_and(|control| control.is_robust()) {
            if let Some((_, started, _)) = cohort_trial.as_mut() { *started = Instant::now(); }
        }
        for (i, job) in jobs.iter_mut().enumerate() {
            let result = (|| -> Result<(), String> {
                if let Some(error) = &cohort_setup_error { return Err(error.clone()); }
                job.ready.wire.connected(job.ready.request.stream).inspect_err(|_| job.cancelled("connection_lost"))?;
                if Instant::now() >= job.ready.deadline { return job.cancel(RequestError::Deadline, limits); }
                // Adaptive K compares measured cost and acceptance with one
                // canonical row; cohort size no longer disables profitable drafts.
                if shared_plain[i] { job.generation.prefer_plain_shared_prefix(); }
                let mut mode = job.ready.request.draft.unwrap_or(config.draft);
                if cohort_fallback_plain || cohort_held == Some(false) || cohort_trial.as_ref().is_some_and(|(draft, _, _)| !draft) {
                    mode = protocol::DraftMode::Plain;
                }
                // VADE_MTP_SAMPLED drafts a sampled chat only while it decodes alone: sampled
                // verify passes do not batch, so a cohort suspends them (one canonical row each).
                if engine.greedy_draft_only() && !(engine.sampled_draft() && (draft_chats <= 1 || engine.sampled_cohort_draft())) && !spark_engine::draft::plain_greedy(job.ready.request.sampling) {mode=protocol::DraftMode::Plain;}
                if mode == protocol::DraftMode::Auto && !engine.automatic_draft_allowed(draft_chats) {
                    mode = protocol::DraftMode::Plain;
                }
                if engine.native_draft_policy() { mode = if mode == protocol::DraftMode::Auto && job.ready.request.max_tokens < engine.automatic_draft_min_output() { protocol::DraftMode::Plain } else { draft_policy.resolve(mode) }; }
                if engine.discard_contended_draft() {
                    // Omitting draft-history kernels makes this request plain
                    // through completion: resuming would require target replay.
                    // Keep the speculator alive to consume its pending token.
                    if concurrent_jobs > 1 && !job.reservation_plain {
                        job.generation.state.session.disable_model_draft();
                        job.generation.state.reserved_session_bytes = reservation_for(&job.ready,job.reserved_tokens,true,None).ok_or("missing plain reservation")?;
                        job.reservation_plain = true;
                    }
                    if job.reservation_plain { mode = protocol::DraftMode::Plain; }
                }
                if concurrent_jobs > engine.draft_chat_limit() { mode = protocol::DraftMode::Plain; }
                job.echo.draft = job.generation.enable_draft(mode,job.ready.request.sampling);
                let remaining = job.ready.prompt.len().saturating_sub(job.generation.state.tokens.len());
                // Prompt work below follows the due decode's client flush.
                if remaining > 0 && (engine.shared_batch() || !shared_prompt(job)) { return Ok(()); }
                if remaining == 0 && joining_plain[i] { return Ok(()); }
                if budget == 0 || !due[i] { return Ok(()); }
                let cap = if Some(job.id) == foreground { budget.saturating_sub(usize::from(aged.is_some() && budget > 1)) }
                    else if engine.native_draft_policy() && !engine.shared_model_drafts() { budget.saturating_sub(rows_after[i]).max(1) }
                    else { (budget / (rows_after[i]+1)).max(1) };
                let cap = if remaining > 0 { cap.min(8) } else { cap };
                let cap = match even_rows { Some(even) if remaining == 0 && Some(job.id) != foreground => cap.min(even), _ => cap };
                let cap = if cohort_trial.is_some() || cohort_held.is_some() { cap.min(2) } else { cap };
                let (work, events) = job.generation.prepare(engine, &job.ready.prompt, job.ready.request.max_tokens,
                    cap, draft_chats, |id| tokenizer.is_none_or(|t| t.has_token(id)))?;
                if let Some(work) = work {
                    budget -= work.row_budget();
                    plans[i] = Some(work);
                } else { job.last_tick = tick; }
                job.events(engine, limits, &mut cache, events)
            })();
            errors[i] = result.err();
        }
        let depth = plans.iter().zip(&errors).filter_map(|(plan, error)|
            plan.as_ref().filter(|_| error.is_none()).map(|p| p.model_limit())).max().unwrap_or(0);
        if engine.shared_draft() && budget > 0 && depth > 0 {
            // Lagging decoders spend spare rows first, without adding a draft
            // weight pass or taking a row away from another ready request.
            let mut order: Vec<_> = (0..jobs.len()).filter(|&i| errors[i].is_none() && plans[i].is_some()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(jobs[i].ready.request.max_tokens
                .saturating_sub(jobs[i].generation.usage.completion_tokens)));
            for i in order {
                if budget == 0 { break; }
                let plan = plans[i].as_mut().unwrap();
                let cap = plan.row_budget()+budget;
                budget -= jobs[i].generation.share_model_depth(engine, plan, cap, depth);
            }
        }
        let draft_indices: Vec<_> = plans.iter().enumerate().filter_map(|(i, p)|
            p.as_ref().filter(|p| p.model_limit() > 0).map(|_| i)).collect();
        if !draft_indices.is_empty() {
            let mut rows: Vec<_> = jobs.iter_mut().zip(&plans).filter_map(|(job, plan)| {
                let p = plan.as_ref().filter(|p| p.model_limit() > 0)?;
                Some(crate::DraftRows { session: &mut job.generation.state.session,
                    history: &job.generation.state.tokens, next: p.tokens()[0], limit: p.model_limit() })
            }).collect();
            let started = Instant::now();
            let result = complete_batch(engine.propose_batch(&mut rows), rows.len());
            draft_total_ms += started.elapsed().as_secs_f64()*1000.0;
            drop(rows);
            match result {
                Ok(output) => for (i, proposal) in draft_indices.into_iter().zip(output) {
                    errors[i] = jobs[i].generation.complete_model(engine, plans[i].as_mut().unwrap(), proposal,
                        |id| tokenizer.is_none_or(|t| t.has_token(id))).err();
                },
                Err(error) => {
                    for i in draft_indices { errors[i] = Some(error.message.clone()); }
                }
            }
        }
        for (plan, error) in plans.iter_mut().zip(&errors) { if error.is_some() { *plan = None; } }
        let indices: Vec<_> = plans.iter().enumerate().filter_map(|(i, p)| p.as_ref().map(|_| i)).collect();
        if !indices.is_empty() && engine.shared_model_drafts() {
            let started = Instant::now();
            let result = {
                let mut rows: Vec<_> = jobs.iter_mut().zip(&plans).filter_map(|(job, plan)| plan.as_ref().map(|p|
                    crate::BatchRows { session: &mut job.generation.state.session, tokens: p.tokens(), output: p.output(),
                        greedy: spark_engine::draft::plain_greedy(job.ready.request.sampling), top_k: 0 })).collect();
                complete_batch(engine.prepare_drafts(&mut rows), rows.len())
            };
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            match result {
                Ok(drafts) => for (&i, draft) in indices.iter().zip(drafts) {
                    if let Some(tokens) = draft {
                        if let Err(error) = plans[i].as_mut().unwrap().replace_model_draft(tokens, ms,
                            |t| !jobs[i].ready.request.ignore_eos && engine.eos_token_ids().contains(&t), |id| tokenizer.is_none_or(|t| t.has_token(id))) {
                            errors[i] = Some(error); plans[i] = None;
                        } else {
                            jobs[i].generation.model_draft_used |= plans[i].as_ref().unwrap().tokens().len() > 1;
                        }
                    }
                },
                Err(error) => for &i in &indices { errors[i] = Some(error.message.clone()); plans[i] = None; },
            }
        }
        let indices: Vec<_> = plans.iter().enumerate().filter_map(|(i, p)| p.as_ref().map(|_| i)).collect();
        let cohort_ran = cohort_trial.is_some() && indices.len() == jobs.len()
            && jobs.iter().all(|job| job.ready.request.max_tokens
                .saturating_sub(job.generation.usage.completion_tokens) >= 3)
            && plans.iter().flatten().all(|p| !matches!(p, crate::generation::Work::Prefill { .. })
                && (1..=2).contains(&p.tokens().len()));
        if !indices.is_empty() {
            let mut rows: Vec<_> = jobs.iter_mut().zip(&plans).filter_map(|(job, plan)| plan.as_ref().map(|p|
                crate::BatchRows { session: &mut job.generation.state.session, tokens: p.tokens(), output: p.output(),
                    greedy: (!matches!(p, crate::generation::Work::Prefill { .. }) || !job.ready.request.kv_checkpoint)
                        && spark_engine::draft::plain_greedy(job.ready.request.sampling),
                    top_k: job.ready.request.sampling.compact_top_k() })).collect();
            let started = Instant::now();
            let result = complete_batch(engine.step_batch(&mut rows), rows.len());
            let finished = Instant::now();
            let elapsed = finished.duration_since(started);
            let target_ms = elapsed.as_secs_f64() * 1000.0;
            target_passes += 1;
            target_rows += rows.iter().map(|r| r.tokens.len()).sum::<usize>();
            target_total_ms += target_ms;
            drop(rows);
            // VADE_MTP_SAMPLED=2: the cohort's sampled drafting chats draw their verified rows at
            // once (one thread per chat); each chat's finish_exact then reads its own draws.
            if engine.sampled_cohort_draft() {
                if let Ok(output) = &result {
                    let at: Vec<_> = (0..jobs.len()).map(|j| indices.iter().position(|&i| i == j)).collect();
                    let tasks: Vec<_> = jobs.iter_mut().zip(&plans).zip(&at).filter_map(|((job, plan), k)| {
                        let (plan, k) = (plan.as_ref()?, (*k)?);
                        let crate::Predictions::Logits(rows) = &output[k] else { return None; };
                        (rows.len() > 1 && matches!(plan, crate::generation::Work::Draft(_))
                            && !spark_engine::draft::plain_greedy(job.ready.request.sampling)).then_some(())?;
                        Some((job.generation.speculator_mut()?, plan.tokens(), rows))
                    }).collect();
                    if tasks.len() > 1 {
                        let mask: Vec<bool> = (0..tasks[0].2[0].len() as u32).map(|id| tokenizer.is_none_or(|t| t.has_token(id))).collect();
                        std::thread::scope(|scope| for (spec, input, rows) in tasks {
                            let mask = &mask;
                            scope.spawn(move || spec.presample(input, rows, mask));
                        });
                    }
                }
            }
            match result {
                Ok(output) => for (i, output) in indices.into_iter().zip(output) {
                    let job = &mut jobs[i];
                    job.last_tick = tick;
                    let mut plan = plans[i].take().unwrap();
                    plan.record_target_ms(target_ms);
                    let prefill = match &plan {
                        crate::generation::Work::Prefill { tokens, last } => Some((tokens.len(), *last)),
                        _ => None,
                    };
                    errors[i] = job.generation.complete(engine, &job.ready.prompt, job.ready.request.max_tokens,
                        plan, output, |id| tokenizer.is_none_or(|t| t.has_token(id)))
                        .and_then(|events| {
                            if let Some((rows, last)) = prefill {
                                let prefill = job.echo.prefill.as_mut().unwrap();
                                prefill.begin(started);
                                prefill.commit_cohort(job.id, rows, 1, finished, last);
                            }
                            job.events(engine, limits, &mut cache, events)
                        }).err();
                },
                Err(error) => {
                    for i in indices { errors[i] = Some(error.message.clone()); }
                }
            }
        }
        if let (Some(control), Some((draft, started, before))) = (cohort_cost.as_mut(), cohort_trial) {
            if cohort_ran && errors.iter().all(Option::is_none) && jobs.iter().all(|j| !j.done) {
                let after = jobs.iter().map(|j| j.generation.cohort_cost_counters())
                    .fold([0; 3], |a, b| std::array::from_fn(|i| a[i] + b[i]));
                if let (Some(committed), Some(proposed), Some(accepted)) =
                    (after[0].checked_sub(before[0]), after[1].checked_sub(before[1]), after[2].checked_sub(before[2]))
                {
                    let ms = started.elapsed().as_secs_f64() * 1000.;
                    if control.observe(draft, ms, committed, proposed, accepted) {
                        let (samples, plain_rate, draft_rate) = control.rates();
                        eprintln!("cohort_cost chats={} samples={samples} draft={draft} cycle_ms={ms:.3} committed={committed} proposed={proposed} accepted={accepted} plain_tok_s={plain_rate:.3} draft_tok_s={draft_rate:.3} head_miss={}", jobs.len(), control.head_miss());
                    }
                }
            }
        }
        // Give control traffic received during decode a turn before launching
        // another chunk. Idle/incomplete sockets do not hold up prefill.
        if (requests == 0 || accepted < requests) && swap.is_none() {
            let slots = (config.connections + 1).saturating_sub(queued.len() + active.len() + preparing + jobs.len());
            let attempts = if requests == 0 { 32 } else { (requests - accepted).min(32) };
            let count = accept_connections(listener, &mut reading, limits, slots, attempts);
            if requests != 0 { accepted += count; }
        }
        // Intake may defer prompt work for one turn, even under continuous arrivals.
        // Prompts excluded from shared prefill cannot form a batch. Start their
        // serial work without a cohort-formation turn or socket readiness poll.
        let intake_pending = jobs.iter().any(|j| j.ready.prompt.len() <= shared_prefill_max_tokens)
            && !intake_deferred && reading.iter().any(|input| !input.responding && input.wire.socket.input_pending());
        intake_deferred = intake_pending;
        // Cold prompts briefly collect simultaneous arrivals. A committed
        // cached prefix already has an owner; its suffix need not wait again.
        // VADE_INTAKE_COLLECT: while requests are still in CPU preparation, cold
        // prompts admitted under 100 ms ago wait for them, as admission does.
        let collecting = preparing > 0 && engine.shared_batch() && intake_collect()
            && jobs.iter().filter(|j| shared_prompt(j) && j.generation.state.tokens.is_empty())
                .map(|j| j.admitted.elapsed()).max().is_some_and(|age| age < Duration::from_millis(100));
        let prompt_ready = !intake_pending && jobs.iter().any(|j| shared_prompt(j)
            && j.generation.state.tokens.len() < j.ready.prompt.len()
            && (j.generation.state.tokens.len() > 0
                || (!collecting && j.admitted.elapsed() >= min_prompt_age)));
        // Burst intake (intake_burst.rs): when every live job is a cold cohort candidate, the first pass (shared or serial) waits for late chats.
        let live: Vec<_> = jobs.iter().zip(&errors).filter(|(j, e)| e.is_none() && !j.done).map(|(j, _)| j).collect();
        let (idle, oldest) = (engine.shared_batch() && active.iter().all(|j| j.done || j.generation.state.tokens.is_empty()) && live.iter().all(|j| j.generation.state.tokens.is_empty() && j.prefill_rows == 0 && j.long_prefill.is_none() && j.ready.prompt.len() <= shared_prefill_max_tokens && prompt_cohort_candidate(j.ready.prompt.len(), 0, shared_prefill_limit, chunk_rows)), live.iter().map(|j| j.ready.enqueued.elapsed()).max().unwrap_or_default());
        let prompt_ready = burst.ready(Instant::now(), idle, prompt_ready || !intake_pending && live.iter().any(|j| !shared_prompt(j) && j.ready.wire.bytes.is_empty()), crate::intake_burst::View { oldest, cold: live.len(), room: active.len() + jobs.len() < config.active, arriving: preparing > 0 || !queued.is_empty() || reading.iter().any(|r| r.arriving(burst.window())) }) && prompt_ready;
        if engine.shared_batch() && prompt_ready {
            plans.iter_mut().for_each(|p| *p = None);
            let mut prompt_rows = if foreground.is_some() { shared_prefill_limit } else { shared_prompt_rows };
            for (i, job) in jobs.iter_mut().enumerate() {
                if errors[i].is_some() || job.done || !job.ready.wire.bytes.is_empty() || !shared_prompt(job) { continue; }
                let remaining = job.ready.prompt.len().saturating_sub(job.generation.state.tokens.len());
                let count = if chunk_rows == 0 { remaining } else { remaining.min(chunk_rows) };
                if count == 0 || count > prompt_rows { continue; }
                let start = job.generation.state.tokens.len();
                plans[i] = Some(crate::generation::Work::Prefill {
                    tokens: job.ready.prompt[start..start + count].to_vec(), last: count == remaining });
                prompt_rows -= count;
            }
            let indices: Vec<_> = plans.iter().enumerate().filter_map(|(i, p)| p.as_ref().map(|_| i)).collect();
            if !indices.is_empty() { if idle { engine_model::events::note_line("intake_burst", &burst.started(Instant::now(), indices.len(), oldest).1); }
                let mut rows: Vec<_> = jobs.iter_mut().zip(&plans).filter_map(|(job, plan)| plan.as_ref().map(|p|
                    crate::BatchRows { session: &mut job.generation.state.session, tokens: p.tokens(), output: p.output(),
                        greedy: job.key.is_none() && !job.ready.request.kv_checkpoint
                            && spark_engine::draft::plain_greedy(job.ready.request.sampling), top_k: 0 })).collect();
                let started = Instant::now();
                let result = complete_batch(engine.prefill_batch(&mut rows), rows.len());
                let finished = Instant::now();
                let cohort_width = rows.len();
                drop(rows);
                match result {
                    Ok(output) => for (i, output) in indices.into_iter().zip(output) {
                        let job = &mut jobs[i];
                        job.last_tick = tick;
                        let rows = plans[i].as_ref().unwrap().tokens().len();
                        let last = matches!(plans[i], Some(crate::generation::Work::Prefill { last: true, .. }));
                        errors[i] = job.generation.complete(engine, &job.ready.prompt, job.ready.request.max_tokens,
                            plans[i].take().unwrap(), output, |id| tokenizer.is_none_or(|t| t.has_token(id)))
                            .and_then(|events| {
                                let prefill = job.echo.prefill.as_mut().unwrap();
                                prefill.begin(started);
                                prefill.commit_cohort(job.id, rows, cohort_width, finished, last);
                                job.events(engine, limits, &mut cache, events)
                            }).err();
                    },
                    Err(error) => {
                        for i in indices { errors[i] = Some(error.message.clone()); }
                    }
                }
            }
        }
        // Foreground events above are flushed before any background GPU work.
        // Keep prompt chunk shapes fixed; bound layer work between decode steps
        // by measured decode time, including when a chunk is already in flight.
        // Give jobs holding peers a bounded turn before their exchanges time out.
        let aged_prefill = jobs.iter().enumerate().filter(|(i, j)| {
            let remaining = j.ready.prompt.len().saturating_sub(j.generation.state.tokens.len());
            remaining > 0 && (remaining <= 8 || (j.long_prefill.is_some()
                && errors[*i].is_none() && !j.done && j.ready.wire.bytes.is_empty()))
                && !shared_prompt(j) && tick.saturating_sub(j.last_tick) >= 4
        }).min_by_key(|(_, j)| (j.last_tick, j.id)).map(|(_, j)| j.id);
        let prefill = jobs.iter().enumerate().filter(|(i, j)| !intake_pending && !burst.holding() && errors[*i].is_none()
            && !j.done && j.ready.wire.bytes.is_empty()
            && j.generation.state.tokens.len() < j.ready.prompt.len() && !shared_prompt(j))
            .min_by_key(|(_, j)| (Some(j.id) != aged_prefill, Some(j.id) != foreground, j.last_tick, j.id))
            .map(|(i, _)| i);
        if let Some(i) = prefill {
            let source = &jobs[i];
            // Waiters for this owner's uncommitted prefix cannot run yet.
            // Keep the large prefill tile until an independent request is due.
            let queued_runnable = queued.iter().any(|waiting| !source.automatic_cache
                || !config.prefix_cache || !waiting.request.prefix_cache || source.ready.scope != waiting.scope
                || shared_prefix(&source.ready.prompt, &waiting.prompt)
                    <= source.generation.state.tokens.len() + shared_prompt_limit);
            let other_runnable = jobs.iter().enumerate().any(|(n, j)| n != i && !j.done
                && errors[n].is_none() && j.ready.wire.bytes.is_empty()) || queued_runnable;
            let decoding = jobs.iter().enumerate().any(|(n, j)| n != i && !j.done
                && errors[n].is_none() && j.ready.wire.bytes.is_empty()
                && j.generation.state.tokens.len() >= j.ready.prompt.len());
            let shared_waiters = !queued.is_empty() && !queued_runnable;
            let contended = Some(jobs[i].id) != foreground && other_runnable;
            let job = &mut jobs[i];
            let context = job.generation.state.tokens.len();
            // Keep in-flight shapes fixed. Engines with canonical tiles retain
            // their minimum row count under contention and yield between layers.
            let rows = if decoding || config.prefill_chunk < 512 { config.prefill_chunk }
                else { job.generation.state.session.prefill_chunk_size(job.ready.prompt.len()) };
            let rows = rows.max(job.generation.state.session.prefill_chunk_min());
            let chunk = if job.prefill_rows != 0 { job.prefill_rows }
                else { rows.min(job.ready.prompt.len() - context) };
            // Complete the owner's wide tile without layer waits while its
            // queued peers depend on this prefix; unrelated work still yields.
            let slice = (other_runnable || chunk > config.prefill_chunk && !shared_waiters || job.ready.request.priority == 0)
                .then(|| prefill_budget.slice(contended || !other_runnable));
            errors[i] = (|| -> Result<(), String> {
                job.ready.wire.connected(job.ready.request.stream).inspect_err(|_| job.cancelled("connection_lost"))?;
                job.flush(limits).inspect_err(|_| job.cancelled("write_failed"))?;
                if !job.ready.wire.bytes.is_empty() { return Ok(()); }
                if Instant::now() >= job.ready.deadline { return job.cancel(RequestError::Deadline, limits); }
                let started = Instant::now();
                job.echo.prefill.as_mut().unwrap().begin(started);
                if !job.prefill_started {
                    job.prefill_started = true; if idle { engine_model::events::note_line("intake_burst", &burst.started(started, 1, oldest).1); }
                    if context == 0 {
                        job.long_prefill = engine.prefill_long(&job.generation.state.session, &job.ready.prompt)?;
                    }
                }
                let event = if let Some(prefill) = &mut job.long_prefill {
                    // Coordinated work stays resumable even before another
                    // request arrives; local idle chunks keep main's fast path.
                    let logits = match prefill.advance(&mut job.generation.state.session, prefill_budget.slice(contended || !other_runnable)) {
                        Ok(Some(logits)) => logits,
                        result => {
                            if let Err(error) = result {
                                if !prefill.allows_fallback(&error) { return Err(error); }
                                engine_model::events::note_line("seqpar_fallback", &format!("seqpar_fallback tokens={} reason={error}", job.ready.prompt.len()));
                                job.long_prefill = None;
                                job.echo.prefill.as_mut().unwrap().abandon();
                            }
                            if contended { prefill_budget.max_slice = prefill_budget.max_slice.max(started.elapsed()); }
                            job.last_tick = tick;
                            return Ok(());
                        }
                    };
                    job.long_prefill = None;
                    job.generation.state.next_logits = logits;
                    job.generation.state.tokens.extend_from_slice(&job.ready.prompt);
                    Some(Event::Prefill)
                } else {
                    job.prefill_rows = chunk;
                    job.generation.prefill_slice(&job.ready.prompt, chunk, slice)?
                };
                if event.is_some() {
                    let end = job.generation.state.tokens.len();
                    job.echo.prefill.as_mut().unwrap().commit_cohort(job.id, end - context, 1, Instant::now(),
                        end == job.ready.prompt.len());
                }
                if job.profile {
                    eprintln!("request_prefill_profile id={} prefix_tokens={context} rows={chunk} wall_ms={:.3} committed={}",
                        job.id, started.elapsed().as_secs_f64()*1000.0, event.is_some());
                }
                if contended { prefill_budget.max_slice = prefill_budget.max_slice.max(started.elapsed()); }
                job.last_tick = tick;
                let Some(event) = event else { return Ok(()); };
                job.prefill_rows = 0;
                job.events(engine, limits, &mut cache, vec![event])?;
                if job.generation.state.tokens.len() == job.ready.prompt.len() {
                    let event = job.generation.next(engine, &job.ready.prompt, job.ready.request.max_tokens,
                        1, |id| tokenizer.is_none_or(|t| t.has_token(id)))?;
                    job.events(engine, limits, &mut cache, vec![event])?;
                }
                Ok(())
            })().err();
        }
        engine.profile_tick_end();
        if jobs.iter().any(|job| job.done) {
            // Release inherited snapshot funding before completed requests
            // publish their responses alongside the new prompt snapshots.
            balance_cache(&mut cache, config.session_budget_bytes,
                jobs.iter().map(|job| &job.generation.state).chain(retired.iter().map(|job| &job.state)),
                engine.cache_hold_bytes() + jobs.iter().map(|job| request_hold(&job.ready)).sum::<usize>(),
                |reserved| engine.cache_free_physical_bytes(reserved))?;
        }
        for (mut job, error) in jobs.into_iter().zip(errors) {
            if error.is_none() && job.done {
                if let Some(key) = job.key {
                    cache.retain_draft_prefix(key, job.ready.scope.as_deref(), &job.generation.state, Instant::now());
                }
                draft_bank.record(job.ready.handle_scope.as_deref(), job.key, config.draft_global_bank, job.ready.prompt.len(), &job.generation.state.tokens);
                // Retain prefilled snapshots only: decode-created KV is not
                // interchangeable with a cold prompt's BF16 prefill arithmetic.
                if let Some((current, peak, pinned)) = engine.memory_bytes() {
                    eprintln!("shared_decode passes={target_passes} rows={target_rows} target_ms={target_total_ms:.3} draft_ms={draft_total_ms:.3}");
                    eprintln!("engine_memory device_bytes={current} peak_device_bytes={peak} pinned_host_bytes={pinned} max_contended_prefill_slice_ms={:.3}",
                        prefill_budget.max_slice.as_secs_f64() * 1000.0);
                }
                if let Some((kv, peak)) = job.generation.state.session.memory_bytes() {
                    eprintln!("request_memory id={} prompt_tokens={} restored_tokens={} kv_bytes={kv} peak_device_bytes={peak}",
                        job.id, job.ready.prompt.len(), job.generation.usage.cached_tokens);
                }
                if let Some(stats) = job.generation.draft_stats() {
                    if let Some((kv, peak)) = job.generation.state.session.draft_model_memory_bytes() {
                        eprintln!("draft_model id={} kv_bytes={kv} kv_peak_bytes={peak}", job.id);
                    }
                    eprintln!("draft id={} cycles={} rows={} accepted={} proposed={} max_rows={} target_ms={:.3} draft_ms={:.3} guards={}/{} host_bytes={} bank_host_bytes={}",
                        job.id, stats.cycles, stats.rows, stats.accepted, stats.proposed, stats.max_rows,
                        stats.target_ms, stats.draft_ms, stats.guard_fires, stats.guarded,
                        job.generation.draft_memory_bytes(), draft_bank.memory_bytes());
                    if stats.model_proposals > 0 {
                        eprintln!("draft_cost id={} proposals={} proposal_ms={:.3} ms_per_proposal={:.3} catchup_ms={:.3} width_cycles={:?} target_costs_ms={:?} depth_trials={:?} depth_hits={:?} band_trials={:?} band_hits={:?} plan={:016x} draft_price={}",
                            job.id, stats.model_proposals, stats.model_proposal_ms,
                            stats.model_proposal_ms / stats.model_proposals as f64, stats.model_catchup_ms, stats.widths,
                            stats.costs_ms, stats.depth_trials[0], stats.depth_hits[0], stats.model_band_trials, stats.model_band_hits,
                            stats.plan_hash, stats.draft_price);
                    }
                }
            }
            if let Some(error) = error {
                engine_model::events::note_line("request_stopped", &format!("request stopped: {error}"));
                let (kind, message) = if job.output_failed { (RequestError::Output, error.as_str()) }
                    else if error.starts_with("Flash MoE routing rejected") || error.starts_with("MoE routing rejected") {
                        (RequestError::Generation, error.as_str())
                    }
                    else { (RequestError::Generation, "generation stopped before completion") };
                let _ = job.error(kind, message, limits);
                job.done = true;
            }
            if job.done {
                // Keep only bounded response bytes while a slow reader drains.
                reading.push(Reading {
                    wire: job.ready.wire,
                    data: Vec::new(),
                    head: None,
                    deadline: Instant::now(),
                    responding: true,
                });
                let mut state = job.generation.state;
                state.session.retire();
                if let Some(bytes) = state.session.retirement_bytes() {
                    state.reserved_session_bytes = bytes + state.tokens.capacity() * 4 + state.next_logits.capacity() * 4;
                }
                retired.push(Retired { state, finished: Instant::now(),
                    id: job.id });
            } else {
                active.push(job);
            }
        }
        if !retired.is_empty() {
            let active_charge = active.iter().map(|j| j.generation.state.reserved_session_bytes).sum::<usize>();
            let retired_charge = retired.iter().map(|r| r.state.reserved_session_bytes).sum::<usize>();
            let (reserved, pages) = reservations(active.iter().map(|j| &j.generation.state)
                .chain(retired.iter().map(|r| &r.state)));
            eprintln!("session_accounting active={} retired={} active_reserved_bytes={active_charge} retired_reserved_bytes={retired_charge} reserved_bytes={reserved} shared_credit_bytes={} shared_pages={} cache_retained_bytes={} cache_unfunded_bytes={}",
                active.len(), retired.len(), active_charge + retired_charge - reserved, pages.len(),
                cache.retained_bytes(), cache.unfunded_bytes());
        }
        if let Some(wait) = burst.park() { std::thread::park_timeout(wait); }
    }
    fatal = fatal.or_else(|| engine.fatal_error());
    if fatal.is_some() {
        #[cfg(target_os = "linux")]
        spark_gpu_probe::teardown::arm();
    }
    if let Some(refresh) = refresh_retired.as_mut() {
        for session in retired { session.release(); refresh(); }
    } else {
        for session in retired { session.release(); }
    }
    if shutdown.load(Ordering::Relaxed) || fatal.is_some() {
        // Fatal engine errors may set shutdown while their SSE error is still
        // buffered. Release sessions first, then drain terminal responses.
        let error = protocol::error_json("server_shutdown", "server is shutting down; request cancelled");
        for input in &mut reading {
            if !input.responding { let _ = input.wire.response(503, &error, limits); }
        }
        for mut job in queued {
            let _ = job.wire.response(503, &error, limits);
            reading.push(Reading { wire: job.wire, data: Vec::new(), head: None,
                deadline: Instant::now(), responding: true });
        }
        for mut job in active {
            let _ = job.error(RequestError::Shutdown, "server is shutting down; request cancelled", limits);
            reading.push(Reading { wire: job.ready.wire, data: Vec::new(), head: None,
                deadline: Instant::now(), responding: true });
        }
        drop(cache);
        let deadline = Instant::now() + limits.write_timeout.min(Duration::from_millis(500));
        while !reading.is_empty() && Instant::now() < deadline {
            reading.retain_mut(|input| input.wire.flush().is_ok() && !input.wire.drained());
            if !reading.is_empty() { std::thread::sleep(Duration::from_millis(1)); }
        }
    }
    if let Some(error) = fatal.or_else(|| engine.fatal_error()) {
        engine_model::events::note_line("gpu_context_poisoned", &format!("GPU context poisoned: {error}"));
        let stop = AtomicBool::new(false);
        // Briefly publish failed health while rejecting all new generation.
        let _ = unavailable(listener, limits, model, &stop, None, "poisoned",
            Some(Instant::now() + Duration::from_secs(1)));
        return Err(error);
    }
    Ok(None)
}

/// A prompt may share cohort chunks when it is short, or when it sits on a
/// canonical tile boundary and chunked sharing is available.
/// VADE_INTAKE_COLLECT (default on; 0 = off): a cold prompt's first prefill waits, at most
/// 100 ms after its admission, while other requests are still in CPU preparation, the same
/// bound admission uses for a CPU-prepared cohort. Fast admission (pooled session memory)
/// otherwise lets a burst's first requests start prefill before its last ones are admitted,
/// splitting one cohort into two. Batch membership never changes a request's bytes.
fn intake_collect() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_INTAKE_COLLECT").as_deref() != Ok("0"))
}
fn prompt_cohort_candidate(remaining: usize, position: usize, limit: usize, chunk_rows: usize) -> bool {
    remaining <= limit || chunk_rows != 0 && position % chunk_rows == 0
}

/// Long prompts join the cohort only when grouping is real: a lone long
/// prompt takes the long-prefill path with its wider grouped passes.
fn prompt_uses_cohort(remaining: usize, position: usize, limit: usize, chunk_rows: usize, ready: usize) -> bool {
    prompt_cohort_candidate(remaining, position, limit, chunk_rows)
        && (remaining <= limit || ready >= 2)
}
/// One event for each prefix fork that failed and fell through to the next candidate or a miss.
fn fork_failed(kind: &str, prompt_tokens: usize, errors: &[String]) {
    for error in errors {
        engine_model::events::note("prefix_cache_fork_failed", format_args!("kind={kind} prompt_tokens={prompt_tokens} error={error}"));
    }
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
