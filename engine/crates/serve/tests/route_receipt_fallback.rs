//! FOLD HONESTY: the served receipt must not say folded_every_down=true when a routed expert batch ran on the host
//! fallback (device_route counts()[2]), and must print its line when the device route is fully off. Its own test binary,
//! so the process counters start at zero and no other test moves them.
use spark_serve::route_receipt::{folded,line};
#[test]
fn host_fallback_downs_make_the_receipt_false_and_are_never_silent() {
    // Fully off: no fold, no device-route down, one routed batch on the host.
    spark_gpu_probe::device_route::fallback();
    let off=line(1).expect("a process whose routed downs all ran on the host must still print its receipt");
    assert!(off.contains("folded_every_down=false"),"{off}");
    // Partly on: every device-route down folded, but the host fallback ran too.
    spark_gpu_probe::device_route::record(true);folded();
    let partly=line(2).unwrap();
    assert!(partly.contains("routed_folds=1") && partly.contains("device_route_down=1"),"{partly}");
    assert!(partly.contains("folded_every_down=false"),"{partly}");
}
