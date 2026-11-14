//! Burst intake: an idle batch collects a burst of chats before its first prompt pass.
//!
//! A client that sends several chats at once connects them within a few
//! milliseconds, but the request bytes of a chat can come after the serving
//! loop accepted its connection. Without a hold, the first prompt pass can
//! start with only the chats that the loop has read, and the late chats wait
//! for that whole pass. The hold applies only while no job decodes, every job
//! is a cold prompt that can join a shared prompt pass, and a session slot is
//! free. It holds the serial prompt path too: alone, such a prompt runs there,
//! and the first chat of a burst is often admitted alone. The hold needs
//! evidence of a burst: two or more cold prompts wait, or another request is
//! still arriving. A connection that has sent no request bytes counts as
//! arriving only for one window after its accept. The hold ends when the
//! connection of the oldest waiting chat is `window` old. After that, the
//! pass waits only for what it waited for before: the admission of the chats
//! that the loop has read. A turn that only the hold stops runs no prompt
//! work, so the loop parks after it. A lone request with nothing else
//! arriving never waits. Batch membership does not change token IDs.
use std::{sync::OnceLock, time::{Duration, Instant}};

/// Default window: covers the 7.75 ms accept spread of the late chats in the
/// two-Spark served cards of main 285fa8a.
pub const DEFAULT_WINDOW: Duration = Duration::from_millis(10);
/// Largest window: the bound of the existing preparation collect.
pub const MAX_WINDOW: Duration = Duration::from_millis(100);
const NAME: &str = "VADE_INTAKE_BURST_MS";
static WINDOW: OnceLock<Result<Option<Duration>, String>> = OnceLock::new();

fn parse(value: Result<String, std::env::VarError>) -> Result<Option<Duration>, String> {
    let error = || format!("{NAME} must be unset or whole milliseconds 0..={}", MAX_WINDOW.as_millis());
    match value {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(error()),
        Ok(text) => match text.parse::<u64>() {
            Ok(ms) if ms.to_string() == text && Duration::from_millis(ms) <= MAX_WINDOW => Ok(Some(Duration::from_millis(ms))),
            _ => Err(error()),
        },
    }
}
/// The startup override of `Limits::intake_burst`; None when unset. Invalid
/// values stay cached as startup errors, including when a service restarts.
pub fn load() -> Result<Option<Duration>, String> {
    WINDOW.get_or_init(|| parse(std::env::var(NAME))).clone()
}

/// What the serving loop sees when the first prompt pass of an idle batch could start.
#[derive(Clone, Copy, Debug)]
pub(crate) struct View {
    /// Time since the loop accepted the connection of the oldest cold prompt.
    pub oldest: Duration,
    /// Cold prompts that wait in the idle batch.
    pub cold: usize,
    /// A connection still sends its request (a silent one only for one
    /// window after its accept), or a request is in preparation or waits for
    /// admission.
    pub arriving: bool,
    /// One more chat can join: a session slot is free.
    pub room: bool,
}
/// True while the first prompt pass waits for the rest of a burst.
pub(crate) fn hold(window: Duration, view: View) -> bool {
    view.cold > 0 && view.room && view.oldest < window && (view.cold >= 2 || view.arriving)
}
/// Longest park after a held turn, as when the loop has no jobs: late request
/// bytes wait at most this long before the loop reads them. A finished
/// preparation unparks the loop at once.
const PARK: Duration = Duration::from_millis(1);

/// The hold of one serving loop: its window and the delay it adds to the
/// first prompt pass of the current idle batch.
pub(crate) struct Burst { window: Duration, blocked: Option<Instant>, park: Option<Duration> }
impl Burst {
    pub fn new(window: Duration) -> Self { Self { window, blocked: None, park: None } }
    /// A connection that has sent no request bytes counts as arriving only
    /// for this long after its accept.
    pub fn window(&self) -> Duration { self.window }
    /// Call once per loop turn. `idle`: no job decodes and every live job is
    /// a cold prompt that can join a shared prompt pass; alone, such a prompt
    /// can run on the serial prompt path. `ready`: a first pass, shared or
    /// serial, could start without the hold. Returns true to start it.
    pub fn ready(&mut self, now: Instant, idle: bool, ready: bool, view: View) -> bool {
        self.park = None;
        if !idle { self.blocked = None; return ready; }
        // Every chat of an earlier held batch was cancelled or disconnected:
        // the oldest chat now came after the hold began, so it inherits nothing.
        if self.blocked.is_some_and(|blocked| now.checked_sub(view.oldest).is_some_and(|first| blocked < first)) { self.blocked = None; }
        if !(ready && hold(self.window, view)) { return ready; }
        // Only the hold stops the pass: park, at most until the window ends.
        self.park = Some(self.window.saturating_sub(view.oldest).min(PARK));
        self.blocked.get_or_insert(now);
        false
    }
    /// How long the loop parks after this turn: only the hold stopped the
    /// pass, so the turn ran no prompt work. The park ends at the window.
    /// None when the hold did not stop the pass. Before the pass is ready,
    /// the loop turns as it did without the hold.
    pub fn park(&mut self) -> Option<Duration> { self.park.take() }
    /// True while this turn holds: the serial prompt path waits too.
    pub fn holding(&self) -> bool { self.park.is_some() }
    /// The first pass of an idle batch starts now with `chats` chats. Returns
    /// the delay the hold added and the log line: the chats, the time since
    /// the oldest accept and that delay.
    pub fn started(&mut self, now: Instant, chats: usize, oldest: Duration) -> (Duration, String) {
        let held = self.blocked.take().map_or(Duration::ZERO, |blocked| now.saturating_duration_since(blocked));
        (held, format!("intake_burst chats={chats} waited_ms={:.3} held_ms={:.3} window_ms={}",
            oldest.as_secs_f64() * 1000.0, held.as_secs_f64() * 1000.0, self.window.as_millis()))
    }
}

#[cfg(test)]
#[path = "intake_burst_tests.rs"]
mod tests;
