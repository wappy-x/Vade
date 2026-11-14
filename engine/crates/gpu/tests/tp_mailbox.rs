//! Mailbox wrappers refuse foreign storage and invalid extents without side effects.
#[cfg(target_os = "linux")]
#[path = "support/tp_mailbox_refusals.rs"]
mod refusals;

#[cfg(target_os = "linux")]
#[test]
#[ignore = "skipped: no GPU; requires CUDA"]
fn mailbox_launches_refuse_another_backends_memory() {
    refusals::run().unwrap();
}
