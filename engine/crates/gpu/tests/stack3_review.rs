use spark_gpu_probe::kernel_params::{
    latency_mx4, latency_mx8_shared, latency_mx8_shared_rows, latency_split, LATENCY_RING,
};

#[test]
fn intra_rings_use_total_rows_and_never_compact_wide_storage() {
    for cols in [256, 512, 768, 1024, 2560, 6144, 10240] {
        for tokens in 1..=16 {
            for total in [64 + 64, 640, 641, 4096] {
                let bytes = latency_mx8_shared_rows(tokens, cols, 4, total, true);
                assert_eq!(bytes, 4 * LATENCY_RING);
                assert!(3 * LATENCY_RING + 5 * 1040 + cols / 16 <= bytes);
            }
        }
        assert!(latency_mx8_shared(1, cols, 4) >= 4 * LATENCY_RING);
        assert_eq!(latency_mx8_shared_rows(1, cols, 1, 640, false), LATENCY_RING);
    }
    assert!(latency_split(4096, true));
    assert!(!latency_split(4097, true));
    assert!(latency_split(640, false));
    assert!(!latency_split(641, false));
    for cols in [2560, 6144, 10240] {
        assert_eq!(latency_mx8_shared_rows(1, cols, 4, 4097, true), cols / 16 * 64 + 4 * 4864);
    }
}

#[test]
fn mx4_scale_burst_fits_or_keeps_mx2() {
    for cols in (256..=32768).step_by(256) {
        assert_eq!(latency_mx4(cols), cols <= 15104);
        if latency_mx4(cols) { assert!(5 * 1040 + cols / 16 <= LATENCY_RING); }
    }
    assert!(latency_mx4(15104));
    assert!(!latency_mx4(15360));
    assert!(!latency_mx4(640));
    for cols in [2560, 6144, 10240] { assert!(latency_mx4(cols)); }
}
