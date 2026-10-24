//! Fixed virtual KV addresses, append-only physical pages and shared prefixes.
use super::*;

#[repr(C)]
struct Properties { kind: c_int, handles: c_int, location: [c_int; 2], metadata: Handle, flags: [u8; 8] }
#[repr(C)]
struct Access { location: [c_int; 2], flags: c_int }
#[link(name = "cuda")]
extern "C" {
    fn cuMemGetAllocationGranularity(bytes: *mut usize, properties: *const Properties, option: c_uint) -> c_int;
    fn cuMemAddressReserve(address: *mut DevicePtr, bytes: usize, alignment: usize, requested: DevicePtr, flags: u64) -> c_int;
    fn cuMemAddressFree(address: DevicePtr, bytes: usize) -> c_int;
    fn cuMemCreate(handle: *mut u64, bytes: usize, properties: *const Properties, flags: u64) -> c_int;
    fn cuMemRelease(handle: u64) -> c_int;
    fn cuMemMap(address: DevicePtr, bytes: usize, offset: usize, handle: u64, flags: u64) -> c_int;
    fn cuMemUnmap(address: DevicePtr, bytes: usize) -> c_int;
    fn cuMemSetAccess(address: DevicePtr, bytes: usize, access: *const Access, count: usize) -> c_int;
}
struct Physical<'a> { handle: u64, bytes: usize, context: &'a Context }
struct Mapping<'a> { physical: Rc<Physical<'a>>, bytes: usize }
pub(super) struct Segments<'a> { reserved: usize, granularity: usize, maps: Vec<Mapping<'a>> }
struct Spare { handle: u64, bytes: usize, segment: bool }
struct MappedSpare { address: DevicePtr, reserved: usize, granularity: usize, maps: Vec<(u64, usize)> }
impl MappedSpare {
    fn bytes(&self) -> usize { self.maps.iter().map(|(_, bytes)| bytes).sum() }
}
/// Spare KV allocations belong to the context, never to a conversation. The vector pool
/// (VADE_VECTOR_POOL) is a second Reserve with the same limits for plain device vectors.
pub(super) struct Reserve {
    label: &'static str,
    pages: Vec<Spare>, bytes: usize, limit: usize, conversation: usize,
    mapped: Vec<MappedSpare>,
    margin: usize, pressure: bool, next_poll: Instant, idle_since: Option<Instant>,
    reused: usize, released: usize, reported: (usize, usize, usize),
    /// MemAvailable and the context's accounted device bytes at the last pressure poll.
    available: usize, accounted: usize,
}
impl Default for Reserve {
    fn default() -> Self {
        Self { label: "kv_reserve", pages: Vec::new(), mapped: Vec::new(), bytes: 0, limit: 0, conversation: 0, margin: 0,
            pressure: false, next_poll: Instant::now(), idle_since: None,
            reused: 0, released: 0, reported: (0, 0, 0), available: 0, accounted: 0 }
    }
}
impl Reserve {
    const MINIMUM: usize = 2_000_000_000;
    pub(super) fn vectors() -> Self { Self { label: "vector_pool", ..Self::default() } }
    pub(super) fn configured(&self) -> bool { self.limit != 0 }
    pub(super) fn configure(&mut self, margin: usize) {
        self.limit = Self::MINIMUM;
        self.margin = margin;
        self.next_poll = Instant::now();
    }
    pub(super) fn conversation_ended(&mut self, bytes: usize) {
        if self.limit == 0 { return; }
        self.conversation = self.conversation.max(bytes);
        self.limit = Self::MINIMUM.max(self.conversation.saturating_mul(4));
    }
    pub(super) fn take(&mut self, bytes: usize, segment: bool) -> Option<u64> {
        let index = self.pages.iter().position(|p| p.bytes == bytes && p.segment == segment)?;
        let page = self.pages.swap_remove(index);
        self.bytes -= bytes;
        self.reused = self.reused.saturating_add(bytes);
        Some(page.handle)
    }
    fn take_mapped(&mut self, reserved: usize, granularity: usize, initial: usize) -> Option<MappedSpare> {
        let index = self.mapped.iter().position(|m| m.reserved == reserved && m.granularity == granularity && m.bytes() <= initial.div_ceil(granularity) * granularity)?;
        let mapped = self.mapped.swap_remove(index);
        self.bytes -= mapped.bytes();
        self.reused = self.reused.saturating_add(mapped.bytes());
        Some(mapped)
    }
    pub(super) fn keep(&mut self, handle: u64, bytes: usize, segment: bool) -> bool {
        if self.pressure || self.bytes.saturating_add(bytes) > self.limit { return false; }
        self.pages.push(Spare { handle, bytes, segment });
        self.bytes += bytes;
        true
    }
    fn release(&mut self, context: &Context, page: Spare) {
        let _wait = context.progress.wait(WaitKind::Cleanup);
        let code = unsafe { if page.segment { cuMemRelease(page.handle) } else { cuMemFree_v2(page.handle) } };
        context.record_cleanup(code, "KV release reserve handle");
        if code == 0 {
            context.allocated.set(context.allocated.get() - page.bytes);
            self.released = self.released.saturating_add(page.bytes);
        }
    }
    fn release_mapped(&mut self, context: &Context, mapped: MappedSpare) {
        let _wait = context.progress.wait(WaitKind::Cleanup);
        unsafe {
            context.record_cleanup(cuMemUnmap(mapped.address, mapped.bytes()), "KV unmap reserve");
            context.record_cleanup(cuMemAddressFree(mapped.address, mapped.reserved), "KV free reserve address");
        }
        for (handle, bytes) in mapped.maps {
            self.release(context, Spare { handle, bytes, segment: true });
        }
    }
    pub(super) fn clear(&mut self, context: &Context) {
        while let Some(mapped) = self.mapped.pop() { self.bytes -= mapped.bytes(); self.release_mapped(context, mapped); }
        while let Some(page) = self.pages.pop() { self.bytes -= page.bytes; self.release(context, page); }
        self.conversation = 0;
        if self.limit != 0 { self.limit = Self::MINIMUM; }
    }
    fn pressure(&mut self, context: &Context, now: Instant) -> Result<(), String> {
        if self.limit == 0 || now < self.next_poll { return Ok(()); }
        self.next_poll = now + std::time::Duration::from_millis(100);
        let info = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
        let available = info.lines().find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1)).and_then(|n| n.parse::<usize>().ok())
            .and_then(|n| n.checked_mul(1024)).ok_or("missing MemAvailable")?;
        self.available = available;
        self.accounted = context.accounted_bytes();
        self.pressure = available < self.margin;
        if self.pressure { self.clear(context); }
        Ok(())
    }
    /// Host bytes left before the pressure rule releases every spare page (MemAvailable minus
    /// the margin at the last poll, less the device bytes accounted since it, so pages released
    /// or mapped between two polls count at once; negative under pressure) and the bytes this
    /// reserve may still keep under its cap. None while the reserve is off. An elastic consumer
    /// (the serving prefix cache) that leaves this room keeps admission on reused pages.
    pub(super) fn room(&mut self, context: &Context) -> Result<Option<(i128, usize)>, String> {
        if self.limit == 0 { return Ok(None); }
        self.pressure(context, Instant::now())?;
        let since = context.accounted_bytes() as i128 - self.accounted as i128;
        Ok(Some((self.available as i128 - self.margin as i128 - since, self.limit.saturating_sub(self.bytes))))
    }
    pub(super) fn maintain(&mut self, context: &Context, idle: bool) -> Result<(), String> {
        if self.limit == 0 { return Ok(()); }
        let now = Instant::now();
        if idle { self.idle_since.get_or_insert(now); } else { self.idle_since = None; }
        let poll = now >= self.next_poll;
        self.pressure(context, now)?;
        if self.idle_since.is_some_and(|start| now.duration_since(start).as_secs() >= 30) && self.bytes != 0 {
            self.clear(context);
        }
        let state = (self.bytes, self.reused, self.released);
        if poll && state != self.reported {
            eprintln!("{} reserve_bytes={} cap_bytes={} reused_bytes={} released_bytes={} device_bytes={} idle_ms={} pressure={}",
                self.label, self.bytes, self.limit, self.reused, self.released, context.allocated.get(),
                self.idle_since.map_or(0, |start| now.duration_since(start).as_millis()), self.pressure);
            self.reported = state;
        }
        Ok(())
    }
    pub(super) fn refresh_pressure(&mut self, context: &Context) -> Result<(), String> {
        self.next_poll = Instant::now();
        self.maintain(context, false)
    }
    pub(super) fn make_room(&mut self, context: &Context, bytes: usize) -> Result<(), String> {
        self.pressure(context, Instant::now())?;
        while context.accounted_bytes().saturating_add(bytes) > context.budget.get() {
            if let Some(page) = self.pages.pop() {
                self.bytes -= page.bytes;
                self.release(context, page);
            } else if let Some(mapped) = self.mapped.pop() {
                self.bytes -= mapped.bytes();
                self.release_mapped(context, mapped);
            } else { break; }
        }
        Ok(())
    }
}
fn properties(context: &Context) -> Properties {
    Properties { kind: 1, handles: 0, location: [1, context.device], metadata: ptr::null_mut(), flags: [0; 8] }
}
impl<'a> Buffer<'a> {
    /// Compact KV uses ordinary CUDA arenas until per-layer vectors reach a page.
    pub(super) fn kv_arena(context: &'a Context, bytes: usize) -> Result<Self, String> {
        context.activate()?;
        let address = {
            let mut reserve = context.kv_reserve.borrow_mut();
            reserve.pressure(context, Instant::now())?;
            reserve.take(bytes, false)
        };
        if let Some(address) = address {
            return Ok(Self { address, bytes, _context: context, owner: None, host: None, segments: None, reuse: Reuse::Kv });
        }
        let mut buffer = Self::new(context, bytes)?;
        buffer.reuse = Reuse::Kv;
        Ok(buffer)
    }
    pub(super) fn segmented(context: &'a Context, maximum: usize, initial: usize) -> Result<Self, String> {
        context.activate()?;
        let mut granularity = 0;
        unsafe { check(cuMemGetAllocationGranularity(&mut granularity, &properties(context), 0), "KV allocation granularity")?; }
        // Admission's page bound is valid only for divisors of the GB10 page.
        if granularity == 0 || (2 * 1024 * 1024) % granularity != 0 {
            return Err("unsupported KV allocation granularity".into());
        }
        let reserved = maximum.div_ceil(granularity).checked_mul(granularity).ok_or("KV reservation overflow")?;
        let spare = {
            let mut reserve = context.kv_reserve.borrow_mut();
            reserve.pressure(context, Instant::now())?;
            reserve.take_mapped(reserved, granularity, initial)
        };
        if let Some(spare) = spare {
            let bytes = spare.bytes();
            let maps = spare.maps.into_iter().map(|(handle, bytes)| Mapping {
                physical: Rc::new(Physical { handle, bytes, context }), bytes,
            }).collect();
            let mut buffer = Self { address: spare.address, bytes, _context: context, owner: None, host: None, reuse: Reuse::Kv,
                segments: Some(Segments { reserved, granularity, maps }) };
            buffer.grow_segments(initial)?;
            return Ok(buffer);
        }
        let mut buffer = Self::reserve_segments(context, reserved, granularity)?;
        buffer.grow_segments(initial)?;
        Ok(buffer)
    }
    fn reserve_segments(context: &'a Context, reserved: usize, granularity: usize) -> Result<Self, String> {
        context.activate()?;
        let mut address = 0;
        unsafe { check(cuMemAddressReserve(&mut address, reserved, granularity, 0, 0), "KV virtual reservation")?; }
        Ok(Self { address, bytes: 0, _context: context, owner: None, host: None, reuse: Reuse::Kv,
            segments: Some(Segments { reserved, granularity, maps: Vec::new() }) })
    }
    pub(super) fn segment_allocations(&self, out: &mut Vec<(usize, usize)>) {
        if let Some(pages) = &self.segments {
            out.extend(pages.maps.iter().map(|m| (Rc::as_ptr(&m.physical) as usize, m.physical.bytes)));
        }
    }
    pub(super) fn shared_prefix_bytes(&self, other: &Self) -> usize {
        // Compact prefixes may be shared arena views, without segmented maps.
        if std::ptr::eq(self._context, other._context) && self.address == other.address {
            return self.bytes.min(other.bytes);
        }
        let (Some(a), Some(b)) = (&self.segments, &other.segments) else { return 0; };
        a.maps.iter().zip(&b.maps).take_while(|(x, y)|
            x.bytes == y.bytes && Rc::ptr_eq(&x.physical, &y.physical))
            .map(|(x, _)| x.bytes).sum()
    }
    pub(super) fn prefix_allocations(&self, committed: usize, out: &mut Vec<(usize, usize)>) {
        if let Some(pages) = &self.segments {
            let mut end = 0;
            for m in &pages.maps {
                end += m.bytes;
                if end > committed { break; }
                out.push((Rc::as_ptr(&m.physical) as usize, m.bytes));
            }
        }
    }
    pub(super) fn writable_from(&self, start: usize) -> bool {
        let Some(pages) = &self.segments else { return true; };
        let mut end = 0;
        pages.maps.iter().all(|m| { end += m.bytes; end <= start || Rc::strong_count(&m.physical) == 1 })
    }
    /// GB10 permits whole-allocation aliases, but rejects partial aliases (801).
    pub(super) fn fork_segments(&self, committed: usize, needed: usize) -> Result<(Self, usize), String> {
        let pages = self.segments.as_ref().ok_or("vector is not segmented")?;
        if committed > self.bytes || committed > needed || needed > pages.reserved { return Err("KV fork range".into()); }
        let mut fork = Self::reserve_segments(self._context, pages.reserved, pages.granularity)?;
        for m in &pages.maps {
            if fork.bytes + m.bytes > committed { break; }
            fork.map_segment(m.physical.clone(), m.bytes, false)?;
        }
        let shared = fork.bytes;
        // Grant the complete alias once, including its new private tail. Each
        // access update walks the mapping and can stall pending GPU work.
        fork.grow_segments_access(needed, false)?;
        if fork.bytes != 0 {
            unsafe { check(cuMemSetAccess(fork.address, fork.bytes, &Access { location: [1, self._context.device], flags: 3 }, 1), "KV fork access")?; }
        }
        Ok((fork, shared))
    }
    pub(super) fn grow_segments(&mut self, needed: usize) -> Result<(), String> {
        self.grow_segments_access(needed, true)
    }
    /// Logical capacity follows the requested pages, even when a recycled
    /// mapping retains a larger physical reserve.
    pub(super) fn segment_capacity(&self, needed: usize) -> Result<usize, String> {
        let pages = self.segments.as_ref().ok_or("vector is not segmented")?;
        needed.div_ceil(pages.granularity).checked_mul(pages.granularity).ok_or_else(|| "KV growth overflow".into())
    }
    fn grow_segments_access(&mut self, needed: usize, access: bool) -> Result<(), String> {
        let pages = self.segments.as_ref().ok_or("vector is not segmented")?;
        let size = self.segment_capacity(needed)?;
        if size <= self.bytes { return Ok(()); }
        if size > pages.reserved { return Err("KV growth exceeds virtual reservation".into()); }
        let extra = size - self.bytes;
        let context = self._context;
        context.activate()?;
        let handle = {
            let mut reserve = context.kv_reserve.borrow_mut();
            reserve.pressure(context, Instant::now())?;
            reserve.take(extra, true)
        };
        if let Some(handle) = handle {
            return self.map_segment(Rc::new(Physical { handle, bytes: extra, context }), extra, access);
        }
        context.make_room(extra)?;
        let total = context.allocated.get().checked_add(extra).ok_or("allocation accounting overflow")?;
        if total.checked_add(context.graph_storage.get()).is_none_or(|n| n > context.budget.get()) { return Err(crate::admission::allocation(total as u128 + context.graph_storage.get() as u128, context.budget.get(), &format!("KV segment growth {extra} bytes; live {} bytes; graphs {} bytes; weights/rank, sessions, CONTEXT and OUTPUT not supplied to KV allocator", context.allocated.get(), context.graph_storage.get()))); }
        let mut handle = 0;
        let mut code = unsafe { cuMemCreate(&mut handle, extra, &properties(context), 0) };
        if code == 2 {
            context.clear_reserves();
            code = unsafe { cuMemCreate(&mut handle, extra, &properties(context), 0) };
        }
        check(code, "KV physical segment")?;
        let total = context.allocated.get() + extra;
        context.record_allocation(total);
        self.map_segment(Rc::new(Physical { handle, bytes: extra, context }), extra, access)
    }
    /// Release complete physical suffix blocks after the stream has finished
    /// using them. Keep the virtual reservation available for decode growth.
    pub(super) fn trim_segments(&mut self, needed: usize) -> Result<(), String> {
        let pages = self.segments.as_mut().ok_or("vector is not segmented")?;
        if needed == 0 || needed > self.bytes { return Err("KV trim range".into()); }
        self._context.activate()?;
        while let Some(last) = pages.maps.last() {
            let remaining = self.bytes - last.bytes;
            if remaining < needed { break; }
            unsafe { check(cuMemUnmap(self.address + remaining as u64, last.bytes), "KV trim segment")?; }
            self.bytes = remaining;
            pages.maps.pop();
        }
        Ok(())
    }
    fn map_segment(&mut self, physical: Rc<Physical<'a>>, bytes: usize, access: bool) -> Result<(), String> {
        let context = self._context;
        let at = self.address + self.bytes as u64;
        unsafe {
            check(cuMemMap(at, bytes, 0, physical.handle, 0), "KV map segment")?;
            if access {
                if let Err(error) = check(cuMemSetAccess(at, bytes, &Access { location: [1, context.device], flags: 3 }, 1), "KV segment access") {
                    context.record_cleanup(cuMemUnmap(at, bytes), "KV unmap failed segment");
                    return Err(error);
                }
            }
        }
        self.segments.as_mut().unwrap().maps.push(Mapping { physical, bytes });
        self.bytes += bytes;
        Ok(())
    }
}
impl Segments<'_> {
    pub(super) fn keep(&mut self, address: DevicePtr, context: &Context) -> bool {
        // Shared snapshots still own their mappings. Only an entirely private
        // range can move into the reserve; new sessions overwrite from length 0.
        if self.maps.is_empty() || self.maps.iter().any(|m| Rc::strong_count(&m.physical) != 1
            || m.bytes != m.physical.bytes) { return false; }
        let bytes: usize = self.maps.iter().map(|m| m.bytes).sum();
        let mut reserve = context.kv_reserve.borrow_mut();
        if reserve.pressure || reserve.bytes.saturating_add(bytes) > reserve.limit { return false; }
        let maps = self.maps.iter_mut().map(|m| {
            let physical = Rc::get_mut(&mut m.physical).unwrap();
            (std::mem::replace(&mut physical.handle, 0), physical.bytes)
        }).collect();
        self.maps.clear();
        reserve.mapped.push(MappedSpare { address, reserved: self.reserved, granularity: self.granularity, maps });
        reserve.bytes += bytes;
        true
    }
    pub(super) fn release(&self, address: DevicePtr, context: &Context) {
        // GB10 accepts complete adjacent mappings in one unmap range.
        let mapped: usize = self.maps.iter().map(|m| m.bytes).sum();
        if mapped != 0 {
            let code = unsafe { cuMemUnmap(address, mapped) };
            context.record_cleanup(code, "KV unmap conversation");
        }
        unsafe { context.record_cleanup(cuMemAddressFree(address, self.reserved), "KV free virtual reservation"); }
    }
}
impl Drop for Physical<'_> {
    fn drop(&mut self) {
        if self.handle == 0 { return; }
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        if self.context.cleanup_errors.borrow().is_empty()
            && self.context.kv_reserve.borrow_mut().keep(self.handle, self.bytes, true) { return; }
        let code = unsafe { cuMemRelease(self.handle) };
        self.context.record_cleanup(code, "KV release allocation handle");
        if code == 0 { self.context.allocated.set(self.context.allocated.get() - self.bytes); }
    }
}
