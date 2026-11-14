//! Session norm construction shared with the host ownership regression.
type Result<T> = std::result::Result<T, String>;
pub(crate) trait Storage {
    type Vector;
    fn vector(&self, len: usize) -> Result<Self::Vector>;
    fn vector_arena(&self, lengths: &[usize]) -> Result<Vec<Self::Vector>>;
    fn write_vector(&self, vector: &mut Self::Vector, values: &[f32]) -> Result<()>;
}
#[cfg(target_os = "linux")]
impl<'a> Storage for spark_gpu_probe::driver::Backend<'a> {
    type Vector = spark_gpu_probe::driver::DeviceVector<'a>;
    fn vector(&self, len: usize) -> Result<Self::Vector> { self.vector(len) }
    fn vector_arena(&self, lengths: &[usize]) -> Result<Vec<Self::Vector>> { self.vector_arena(lengths) }
    fn write_vector(&self, vector: &mut Self::Vector, values: &[f32]) -> Result<()> { self.write_vector(vector, values) }
}
/// Slot registration updates bounded views; the table must retain an arena owner.
pub(crate) fn state_slots<S: Storage>(b: &S) -> Result<S::Vector> {
    b.vector_arena(&[17*std::mem::size_of::<spark_gpu_probe::next_math::StateSlot>()/4])?
        .pop().ok_or_else(|| "Flash-Next state table allocation missing".into())
}
pub(crate) fn session_norms<S: Storage>(b: &S, mut values: impl FnMut(&str) -> Result<Vec<f32>>)
    -> Result<[Vec<Option<S::Vector>>; 2]> {
    let (mut gdn, mut qsa) = (Vec::new(), Vec::new());
    for li in 0..48 {
        let p = format!("model.language_model.layers.{li}");
        if li % 4 == 3 {
            let norms: Vec<f32> = ["q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"]
                .iter().map(|s| values(&format!("{p}.self_attn.{s}"))).collect::<Result<Vec<_>>>()?
                .into_iter().flatten().collect();
            if norms.len() != 768 { return Err("QSA needs all four gathered norm vectors".into()); }
            // QsaScratch::use_norm retains a zero-copy slice of this allocation.
            let mut v = b.vector_arena(&[norms.len()])?.pop().unwrap();
            b.write_vector(&mut v, &norms)?;
            gdn.push(None); qsa.push(Some(v));
        } else {
            let norms = values(&format!("{p}.linear_attn.norm.weight"))?;
            let mut v = b.vector(norms.len())?;
            b.write_vector(&mut v, &norms)?;
            gdn.push(Some(v)); qsa.push(None);
        }
    }
    Ok([gdn, qsa])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, ops::Range, rc::Rc};
    // Host storage preserves the device distinction between standalone vectors
    // and arena views; slices retain their backing allocation without copying.
    struct HostVector { data: Rc<RefCell<Vec<f32>>>, start: usize, len: usize, arena: bool }
    impl HostVector {
        fn slice(&self, range: Range<usize>) -> Result<Self> {
            if range.start >= range.end || range.end > self.len { return Err("invalid device vector slice".into()); }
            if !self.arena { return Err("slicing requires vector_arena storage".into()); }
            Ok(Self { data: self.data.clone(), start: self.start + range.start, len: range.len(), arena: true })
        }
        fn values(&self) -> Vec<f32> { self.data.borrow()[self.start..self.start + self.len].to_vec() }
    }
    struct Host;
    impl Storage for Host {
        type Vector = HostVector;
        fn vector(&self, len: usize) -> Result<HostVector> {
            Ok(HostVector { data: Rc::new(RefCell::new(vec![0.0; len])), start: 0, len, arena: false })
        }
        fn vector_arena(&self, lengths: &[usize]) -> Result<Vec<HostVector>> {
            let words = lengths.iter().map(|n| n.div_ceil(16) * 16).sum();
            let data = Rc::new(RefCell::new(vec![0.0; words]));
            let mut start = 0;
            Ok(lengths.iter().map(|&len| {
                let v = HostVector { data: data.clone(), start, len, arena: true };
                start += len.div_ceil(16) * 16; v
            }).collect())
        }
        fn write_vector(&self, vector: &mut HostVector, values: &[f32]) -> Result<()> {
            if vector.len != values.len() { return Err("host upload length mismatch".into()); }
            vector.data.borrow_mut()[vector.start..vector.start + vector.len].copy_from_slice(values); Ok(())
        }
    }
    fn norm_values(name: &str) -> Result<Vec<f32>> {
        let layer: usize = name.split('.').nth(3).unwrap().parse().unwrap();
        let (component, len) = if name.ends_with("indexer.q_layernorm.weight") { (2, 128) }
            else if name.ends_with("indexer.k_layernorm.weight") { (3, 128) }
            else if name.ends_with("q_norm.weight") { (0, 256) }
            else if name.ends_with("k_norm.weight") { (1, 256) } else { (4, 128) };
        Ok((0..len).map(|i| (layer * 2048 + component * 256 + i) as f32).collect())
    }
    #[test]
    fn nextmodel_session_slices_retain_norm_storage() -> Result<()> {
        let b = Host;
        let slots=state_slots(&b)?;
        for slot in 0..17 { let view=slots.slice(slot*292..(slot+1)*292)?;
            assert!(Rc::ptr_eq(&view.data,&slots.data)); b.write_vector(&mut slots.slice(slot*292..(slot+1)*292)?,&vec![slot as f32;292])?;
            assert_eq!(view.values(),vec![slot as f32;292]); }
        let [gdn, qsa] = session_norms(&b, norm_values)?;
        assert_eq!((gdn.len(), qsa.len()), (48, 48));
        assert_eq!(gdn.iter().flatten().count(), 36);
        assert_eq!(qsa.iter().flatten().count(), 12);
        assert_eq!(gdn.iter().chain(&qsa).flatten().map(|v| v.len * 4).sum::<usize>(), 55296);
        assert!(b.vector(768)?.slice(0..768).is_err()); // The original allocation fails the contract.
        let mut retained = Vec::new();
        // Prompt and cached decode both call use_norm on every full-attention layer.
        for _token in 0..3 { for li in 0..48 {
            assert_eq!(gdn[li].is_some(), li % 4 != 3);
            assert_eq!(qsa[li].is_some(), li % 4 == 3);
            if let Some(norm) = &qsa[li] {
                let view = norm.slice(0..768)?;
                assert!(Rc::ptr_eq(&view.data, &norm.data));
                for (suffix, range) in [("q_norm.weight", 0..256), ("k_norm.weight", 256..512),
                    ("indexer.q_layernorm.weight", 512..640), ("indexer.k_layernorm.weight", 640..768)] {
                    assert_eq!(view.slice(range)?.values(), norm_values(&format!("model.language_model.layers.{li}.self_attn.{suffix}"))?);
                }
                retained.push((view, norm.values()));
            }
        } }
        // QsaScratch's own arena also supplies the norm upload and cursor views.
        let scratch = b.vector_arena(&[770])?.pop().unwrap();
        b.write_vector(&mut scratch.slice(0..768)?, &[1.0; 768])?;
        b.write_vector(&mut scratch.slice(768..770)?, &[3.0, 1.0])?;
        assert_eq!(scratch.slice(768..770)?.values(), [3.0, 1.0]);
        drop((gdn, qsa));
        for (view, expected) in retained { assert_eq!(view.values(), expected); }
        Ok(())
    }
}
