//! The tensor split's mailbox wrappers refuse a mailbox or rows from another Backend (context).
#[cfg(target_os = "linux")]
#[test]
#[ignore = "skipped: no GPU; requires CUDA"]
fn mailbox_launches_refuse_another_backends_memory() {
    use spark_gpu_probe::{driver::with_backend, layer_ptx};
    with_backend(0, layer_ptx().unwrap(), usize::MAX, |a| {
        let own = a.tp_mailbox(64)?;
        let mut rows = a.vector(16)?;
        with_backend(0, layer_ptx().unwrap(), usize::MAX, |b| {
            let other = b.tp_mailbox(64)?;
            let mut foreign = b.vector(16)?;
            assert!(a.tp_mailbox_put(&other, &rows, 1, 16, 8, 0, 1).is_err(), "mailbox of another backend");
            assert!(a.tp_mailbox_take(&other, &mut rows, 1, 16, 8, 8, 1, 1_000).is_err(), "mailbox of another backend");
            assert!(a.tp_mailbox_put(&own, &foreign, 1, 16, 8, 0, 1).is_err(), "rows of another backend");
            assert!(a.tp_mailbox_take(&own, &mut foreign, 1, 16, 8, 8, 1, 1_000).is_err(), "rows of another backend");
            Ok(())
        })?;
        a.tp_mailbox_put(&own, &rows, 1, 16, 8, 0, 1)?;
        a.synchronize()
    }).unwrap();
}
