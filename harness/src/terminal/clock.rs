//! Display-only wall clock; journals and model messages keep their original bytes.
use std::time::{SystemTime, UNIX_EPOCH};
pub fn now() -> (String, String) {
    at(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64)
}
pub fn at(epoch: i64) -> (String, String) {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use std::ffi::{c_char, c_int, c_long};
        #[repr(C)]
        #[derive(Default)]
        struct Tm {
            sec: c_int,
            min: c_int,
            hour: c_int,
            day: c_int,
            month: c_int,
            year: c_int,
            weekday: c_int,
            yearday: c_int,
            dst: c_int,
            offset: c_long,
            zone: *const c_char,
        }
        unsafe extern "C" {
            fn localtime_r(time: *const c_long, out: *mut Tm) -> *mut Tm;
            fn gmtime_r(time: *const c_long, out: *mut Tm) -> *mut Tm;
        }
        let time = epoch as c_long;
        let mut tm = Tm::default();
        // Both libc calls write only the caller-owned tm; no shared tm pointer.
        let utc = unsafe { localtime_r(&time, &mut tm) }.is_null();
        if !utc || !unsafe { gmtime_r(&time, &mut tm) }.is_null() {
            let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
            let months = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ];
            return (
                format!(
                    "{:02}:{:02}:{:02}{}",
                    tm.hour,
                    tm.min,
                    tm.sec,
                    if utc { "Z" } else { "" }
                ),
                format!(
                    "{} {} {} {}",
                    days[tm.weekday as usize],
                    tm.day,
                    months[tm.month as usize],
                    tm.year + 1900
                ),
            );
        }
    }
    let seconds = epoch.rem_euclid(86400);
    (
        format!(
            "{:02}:{:02}:{:02}Z",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        ),
        "UTC date unavailable".into(),
    )
}

/// The one duration style for every footer and summary: `3.2s`, `19m44s`,
/// `20m`, `1h05m`. Values under a minute keep one decimal.
pub fn duration(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds < 60 {
        format!("{:.1}s", elapsed.as_secs_f64())
    } else if seconds >= 3600 {
        format!("{}h{:02}m", seconds / 3600, seconds / 60 % 60)
    } else if seconds.is_multiple_of(60) {
        format!("{}m", seconds / 60)
    } else {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    }
}

/// The one count style: `950`, `4.4k`, `131k`, `1.2M`.
pub fn count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6).replace(".0M", "M")
    } else if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0).replace(".0k", "k")
    } else {
        n.to_string()
    }
}

/// The one rate style: `23 tok/s`, `1.2k tok/s`.
pub fn rate(tokens: u64, seconds: f64) -> String {
    let rate = tokens as f64 / seconds;
    if rate >= 1000.0 {
        format!("{:.1}k tok/s", rate / 1000.0).replace(".0k", "k")
    } else {
        format!("{rate:.0} tok/s")
    }
}
