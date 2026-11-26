//! Paged CUDA VMM straight-KV plane operations for `Device`.
use super::*;

impl<'b, 'd> Device<'b, 'd> {
    pub(super) fn alloc_plane(&self, maximum_words: usize, initial_words: usize) -> Result<Value<'b, 'd>> {
        const VMM_PAGE_BYTES: usize = 2 * 1024 * 1024;
        let stamp = self.cost.start(self.backend);
        if maximum_words.saturating_mul(4) >= VMM_PAGE_BYTES {
            let init = initial_words.max(1).min(maximum_words);
            let mut vec = self.backend.segmented_vector(maximum_words, init)?;
            self.backend.skip_fill(&mut vec)?;
            let words = vec.len().min(maximum_words);
            let value = self.shared_value(Rc::new(vec), words)?;
            self.cost.record(|| "plane_alloc".into(), stamp, self.backend, 0);
            return Ok(value);
        }
        let mut vec = self.backend.vector(maximum_words.max(1))?;
        self.backend.skip_fill(&mut vec)?;
        let value = self.value(vec, maximum_words)?;
        self.cost.record(|| "plane_alloc".into(), stamp, self.backend, 0);
        Ok(value)
    }

    pub(super) fn grow_plane(&self, plane: &mut Value<'b, 'd>, needed_words: usize, maximum_words: usize) -> Result<()> {
        let target = needed_words.min(maximum_words);
        if plane.words >= target { return Ok(()); }
        let stamp = self.cost.start(self.backend);
        if plane.vector.is_segmented() {
            let old_words = plane.words;
            let old_len = plane.vector.len();
            let vec = Rc::get_mut(&mut plane.vector).ok_or("paged plane is shared during grow")?;
            if target > old_len { self.backend.grow_vector(vec, target)?; }
            let new_words = vec.len().min(maximum_words);
            if new_words > old_words { self.backend.skip_fill_range(vec, old_words..new_words)?; }
            plane.words = new_words;
            let source = self.source.get().checked_add(1).ok_or("immutable source identity exhausted")?;
            self.source.set(source);
            plane.lineage = Lineage::fresh(source, new_words);
            *plane.known.borrow_mut() = None;
            self.cost.record(|| "plane_grow".into(), stamp, self.backend, 0);
            return Ok(());
        }
        let mut out = self.backend.vector(target.max(1))?;
        self.backend.skip_fill(&mut out)?;
        if plane.words > 0 { self.backend.copy_range(&plane.vector, &mut out, 0, 0, plane.words)?; }
        *plane = self.value(out, target)?;
        self.cost.record(|| "plane_grow".into(), stamp, self.backend, 0);
        Ok(())
    }

    pub(super) fn fork_plane(&self, source: &Value<'b, 'd>, committed_words: usize, needed_words: usize, maximum_words: usize) -> Result<Value<'b, 'd>> {
        self.ready(source)?;
        need(committed_words <= source.words && committed_words <= maximum_words, "plane fork exceeds source")?;
        let target = needed_words.max(committed_words).min(maximum_words);
        let stamp = self.cost.start(self.backend);
        if source.vector.is_segmented() {
            if committed_words == 0 { return self.alloc_plane(maximum_words, target); }
            let mut vec = self.backend.fork_vector(&source.vector, committed_words, target.max(1))?;
            let words = vec.len().min(maximum_words);
            if committed_words < words { self.backend.skip_fill_range(&mut vec, committed_words..words)?; }
            let value = self.shared_value(Rc::new(vec), words)?;
            self.cost.record(|| "plane_fork".into(), stamp, self.backend, 0);
            return Ok(value);
        }
        let mut out = self.backend.vector(target.max(1))?;
        self.backend.skip_fill(&mut out)?;
        if committed_words > 0 { self.backend.copy_range(&source.vector, &mut out, 0, 0, committed_words)?; }
        let value = self.value(out, target)?;
        self.cost.record(|| "plane_fork".into(), stamp, self.backend, 0);
        Ok(value)
    }

    pub(super) fn trim_plane(&self, plane: &mut Value<'b, 'd>, keep_words: usize) -> Result<()> {
        if plane.vector.is_segmented() && keep_words > 0 && keep_words < plane.words
            && let Some(vec) = Rc::get_mut(&mut plane.vector)
        {
            self.backend.trim_vector(vec, keep_words)?;
            plane.words = vec.len().min(plane.words);
        }
        Ok(())
    }

    pub(super) fn range_segmented(&self, v: &Value<'b, 'd>, start: usize, len: usize) -> Result<Value<'b, 'd>> {
        let stamp = self.cost.start(self.backend);
        if start == 0 {
            let value = Value {
                vector: v.vector.clone(), words: len, lineage: v.lineage.range(0, len)?,
                known: RefCell::new(None), assumed: Cell::new(false), pending: None, held: None, prompt_middle: None,
            };
            if v.assumed.get() {
                value.assumed.set(true);
                *value.known.borrow_mut() = v.known.borrow().as_ref().map(|row| row[..len].to_vec());
            }
            self.cost.record(|| "range_view".into(), stamp, self.backend, 0);
            return Ok(value);
        }
        let mut out = self.backend.vector(len.max(1))?;
        if len > 0 { self.backend.copy_range(&v.vector, &mut out, start, 0, len)?; }
        let mut value = self.value(out, len)?;
        value.lineage = v.lineage.range(start, len)?;
        self.cost.record(|| "range_copy".into(), stamp, self.backend, 0);
        Ok(value)
    }

    pub(super) fn prefix_plane_allocations(&self, v: &Value<'b, 'd>, committed_words: usize, out: &mut Vec<(usize, usize)>) -> bool {
        if v.pending.as_ref().is_some_and(|p| p.borrow().post.is_some()) { return false; }
        if v.vector.is_segmented() {
            v.vector.prefix_allocations(committed_words, out);
            true
        } else {
            self.state_allocations(v, out)
        }
    }
}
