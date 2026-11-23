//! The device as the owner of the weights: one device vector for each tensor of the load plan.
//! A vector holds the bytes of its tensor in whole 32-bit words. A unit of a load is written through a
//! view of its words, so no host copy of a whole tensor is kept and the pinned staging stays small.
use super::load::Store;
use crate::Result;
use spark_gpu_probe::driver::{Backend, DeviceVector};

pub struct DeviceStore<'b, 'd> { backend: &'b Backend<'d> }
impl<'b, 'd> DeviceStore<'b, 'd> {
    pub fn new(backend: &'b Backend<'d>) -> Self { Self { backend } }
}
impl<'d> Store for DeviceStore<'_, 'd> {
    type Value = DeviceVector<'d>;
    fn create(&mut self, words: usize) -> Result<Self::Value> { self.backend.vector(words)?.into_resident_shared() }
    fn write(&mut self, value: &mut Self::Value, word: usize, bytes: &[u8]) -> Result<()> {
        if !bytes.len().is_multiple_of(4) { return Err("device store: a write must hold whole words".into()); }
        let end = word.checked_add(bytes.len() / 4).ok_or("device store: write range overflow")?;
        let mut part = value.resident_range(word..end)?;
        self.backend.write_vector_bytes(&mut part, bytes)
    }
    fn views(&self) -> bool { true }
    fn view(&mut self, value: &Self::Value, words: std::ops::Range<usize>) -> Result<Self::Value> { value.resident_range(words) }
    fn read(&mut self, value: &Self::Value, word: usize, out: &mut [u8]) -> Result<()> {
        if !out.len().is_multiple_of(4) { return Err("device store: a read must hold whole words".into()); }
        let end = word.checked_add(out.len() / 4).ok_or("device store: read range overflow")?;
        let words = self.backend.read_vector_values(&value.resident_range(word..end)?)?;
        // A word keeps its bit pattern: the bytes of a weight are not numbers to this store.
        for (out, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) { *out = word.to_bits().to_le_bytes(); }
        Ok(())
    }
}
