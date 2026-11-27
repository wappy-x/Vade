//! Generation time is independent of socket intake and output timeouts.
use spark_serve::Limits;
use std::time::Duration;

#[test]
fn long_generation_keeps_the_normal_default_and_socket_bounds() {
    let mut limits=Limits::default();
    assert_eq!(limits.generation_timeout,Duration::from_secs(3600));
    assert!(limits.validate().is_ok());
    limits.generation_timeout=Limits::MAX_GENERATION_TIMEOUT;
    assert!(limits.validate().is_ok());
    limits.generation_timeout+=Duration::from_nanos(1);
    assert!(limits.validate().is_err());
    limits.generation_timeout=Duration::ZERO;
    assert!(limits.validate().is_err());
    limits.generation_timeout=Limits::MAX_GENERATION_TIMEOUT;
    for timeout in [Duration::ZERO,Duration::from_secs(3601)] {
        limits.read_timeout=timeout;assert!(limits.validate().is_err());
        limits.read_timeout=Duration::from_secs(15);
        limits.write_timeout=timeout;assert!(limits.validate().is_err());
        limits.write_timeout=Duration::from_secs(15);
    }
    assert!(limits.validate().is_ok());
}
