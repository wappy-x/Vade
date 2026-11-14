//! A served proof that every routed down projection took the local fold.
//! The tensor-split decode folds the picked experts on the owner rank before the exchange (LOCAL FOLD). A step that
//! fell back to the two-step path would still give the same tokens, only slower, so a served receipt needs the two
//! counts: the folds, and the device-route down projections. They are process totals; equal totals at every finished
//! request mean no step fell back since the process started.
//! Coverage: a routed projection that took the host path (device_route::fallback, the third count) is not a fold and
//! not a device-route down, so the line names that count too and folded_every_down is true only while it is zero
//! (rc.12 review follow-up). A prompt group of more than one row never tries the fold; it moves none of the counts.
use std::sync::atomic::{AtomicU64, Ordering};

static FOLDS: AtomicU64 = AtomicU64::new(0);
/// One routed fold ran on this rank's rows before its exchange.
pub fn folded() { FOLDS.fetch_add(1, Ordering::Relaxed); }
/// The log line of one finished request, or None for a process that ran no device-route projection, no host-path
/// routed projection and no fold.
pub fn line(id: u64) -> Option<String> {
    let counts = spark_gpu_probe::device_route::counts();
    text(id, FOLDS.load(Ordering::Relaxed), counts[1], counts[2])
}
fn text(id: u64, folds: u64, down: u64, host: u64) -> Option<String> {
    if folds == 0 && down == 0 && host == 0 { return None; }
    Some(format!("request_route id={id} routed_folds={folds} device_route_down={down} route_host_fallback={host} folded_every_down={}",
        folds == down && host == 0))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_line_names_both_counts_and_stays_silent_without_a_device_route() {
        assert_eq!(text(7, 0, 0, 0), None);
        assert_eq!(text(7, 84, 84, 0).as_deref(),
            Some("request_route id=7 routed_folds=84 device_route_down=84 route_host_fallback=0 folded_every_down=true"));
        assert_eq!(text(8, 42, 84, 0).as_deref(),
            Some("request_route id=8 routed_folds=42 device_route_down=84 route_host_fallback=0 folded_every_down=false"));
        assert_eq!(text(9, 0, 3, 0).as_deref(),
            Some("request_route id=9 routed_folds=0 device_route_down=3 route_host_fallback=0 folded_every_down=false"));
    }
    #[test]
    fn a_host_path_projection_makes_the_line_false_even_when_the_device_counts_agree() {
        assert_eq!(text(10, 84, 84, 1).as_deref(),
            Some("request_route id=10 routed_folds=84 device_route_down=84 route_host_fallback=1 folded_every_down=false"));
        assert_eq!(text(11, 0, 0, 2).as_deref(),
            Some("request_route id=11 routed_folds=0 device_route_down=0 route_host_fallback=2 folded_every_down=false"));
    }
    #[test]
    fn a_fold_is_counted_once() {
        let before = FOLDS.load(Ordering::Relaxed); folded();
        assert!(FOLDS.load(Ordering::Relaxed) > before);
        assert!(line(1).is_some());
    }
}
