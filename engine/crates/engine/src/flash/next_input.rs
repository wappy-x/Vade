//! The host-fed control prefix shares an allocation with device predictions.
type Result<T> = std::result::Result<T, String>;
pub(crate) const IDS_WORDS: usize = 4 + 16 * 2;
pub(crate) trait Storage {
    type Vector;
    fn vector(&self, len: usize) -> Result<Self::Vector>;
    fn write_vector_bytes(&self, vector: &mut Self::Vector, values: &[u8]) -> Result<()>;
}
#[cfg(target_os = "linux")]
impl<'a> Storage for spark_gpu_probe::driver::Backend<'a> {
    type Vector = spark_gpu_probe::driver::DeviceVector<'a>;
    fn vector(&self, len: usize) -> Result<Self::Vector> { self.vector(len) }
    fn write_vector_bytes(&self, vector: &mut Self::Vector, values: &[u8]) -> Result<()> { self.write_vector_bytes(vector, values) }
}
pub(crate) fn ids<S: Storage>(b: &S) -> Result<S::Vector> { b.vector(IDS_WORDS) }
pub(crate) fn begin<S: Storage>(b: &S, ids: &mut S::Vector, token: Option<u32>) -> Result<()> {
    if let Some(t) = token {
        let control = [0.0f32, t as f32, 1.0];
        // write_vector requires all 36 words; only these three are host input.
        let bytes = unsafe { std::slice::from_raw_parts(control.as_ptr().cast(), std::mem::size_of_val(&control)) };
        b.write_vector_bytes(ids, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_model::flash::{refcpu::{self, PleHash}, PLE_PART_ROWS, PLE_ROWS};
    use spark_gpu_probe::{check_vector_shape, check_vector_upload, frontend_math::rows};
    struct Vector { logical: usize, same_context: bool, bytes: Vec<u8> }
    struct Host;
    impl Storage for Host {
        type Vector = Vector;
        fn vector(&self, len: usize) -> Result<Vector> { Ok(Vector { logical: len, same_context: true, bytes: vec![0xa5; len * 4] }) }
        fn write_vector_bytes(&self, v: &mut Vector, values: &[u8]) -> Result<()> {
            check_vector_upload(v.logical, v.bytes.len(), v.same_context, values.len())?;
            v.bytes[..values.len()].copy_from_slice(values); Ok(())
        }
    }
    #[test]
    fn real_frontend_shape_prompt1024_decode128() -> Result<()> {
        let dims = refcpu::Dims::real();
        assert_eq!((dims.hidden, dims.hc, dims.vocab), (2560, 4, 248320));
        let (sizes, offsets, padded) = refcpu::ple_vocab(20_000_000, 0, 16, 128);
        assert_eq!((padded, PLE_ROWS / PLE_PART_ROWS, dims.hidden / sizes.len()), (PLE_ROWS, 128, 160));
        let mult = refcpu::ple_multipliers(dims.vocab, 3, 0, 1234);
        let p: [i64; 35] = [mult.clone(), sizes.clone(), offsets.clone()].concat().try_into().unwrap();
        let mut reference = PleHash::new(mult, sizes, offsets, dims.eos as i64);
        let mut ids = ids(&Host)?;
        assert_eq!((ids.logical, ids.bytes.len()), (36, 144));
        assert_eq!(check_vector_shape(ids.logical, ids.bytes.len(), true, 3).unwrap_err(),
            "device vector context or shape mismatch: requested=3 logical=36 mapped_bytes=144 same_context=true");
        assert!(check_vector_upload(36, 256, true, 145).is_err()); // mapped padding is not capacity
        ids.same_context = false; assert!(begin(&Host, &mut ids, Some(760)).is_err()); ids.same_context = true;
        for _ in 0..2 {
            reference.reset(); let mut history = [dims.eos as i64; 2]; let mut predicted = 0;
            for pos in 0..1024 + 128 {
                let token = if pos < 1024 { [760, 6511, 314, 9338, 369][pos % 5] } else { predicted };
                let before = ids.bytes.clone();
                begin(&Host, &mut ids, (pos < 1024).then_some(token))?;
                assert_eq!(&ids.bytes[12..], &before[12..]); // reserved word + 16 u64 predictions
                if pos >= 1024 { assert_eq!(ids.bytes, before); } // consume device argmax, no host upload
                let controls: Vec<f32> = ids.bytes[..12].chunks_exact(4).map(|v| f32::from_ne_bytes(v.try_into().unwrap())).collect();
                assert_eq!((controls[1] as u32, controls[2]), (token, 1.0));
                let actual = rows(controls[1] as u32, &mut history, &p);
                assert_eq!(actual.as_slice(), reference.advance(token));
                assert!(actual.iter().all(|&row| row / PLE_PART_ROWS < 128));
                predicted = if pos % 127 == 0 { dims.eos as u32 } else { (pos * 7919 % dims.vocab as usize) as u32 };
                // Model argmax writes the same three words, then publish adds the predicted rows.
                for (i, f) in [0.0f32, predicted as f32, 1.0].iter().enumerate() { ids.bytes[i*4..i*4+4].copy_from_slice(&f.to_ne_bytes()); }
                let next = rows(predicted, &mut history.clone(), &p);
                for (h, row) in next.iter().enumerate() { ids.bytes[16+h*8..24+h*8].copy_from_slice(&row.to_ne_bytes()); }
            }
        }
        Ok(())
    }
}
