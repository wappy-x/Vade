//! Bounded UI lanes. Workers never wait for display consumption. Durable facts
//! and the terminal result have their own capacity; cancellation is independent.
use std::collections::VecDeque;
use std::io;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, Ordering},
    mpsc::{self, Receiver, TryRecvError},
};
use std::thread::{self, Thread};
use std::time::Instant;
use vade_harness::wake;

pub const FRAME_BYTES: usize = 16 * 1024;
const RENDER_BYTES: usize = 64 * 1024;
const FACTS: usize = 16;

#[derive(Default)]
pub struct WorkQueue {
    prompts: VecDeque<String>,
    bytes: usize,
}
impl WorkQueue {
    pub fn push(&mut self, text: String) -> Result<(), String> {
        if self.prompts.len() >= 8 || text.len() > (64 * 1024usize).saturating_sub(self.bytes) {
            return Err(text);
        }
        self.bytes += text.len();
        self.prompts.push_back(text);
        Ok(())
    }
    pub fn pop(&mut self) -> Option<String> {
        let text = self.prompts.pop_front()?;
        self.bytes -= text.len();
        Some(text)
    }
    pub fn len(&self) -> usize {
        self.prompts.len()
    }
    pub fn clear(&mut self) {
        self.prompts.clear();
        self.bytes = 0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    Preparing,
    Connecting,
    Generating,
    ToolRunning,
    WaitingForApproval,
    Compacting,
    Cancelling,
    Completed,
    Failed,
    OutcomeUnknown,
    Interrupted,
    NotStarted,
    LocalCancelled,
    Queued,
    Thinking,
    Counting,
}
impl State {
    pub fn label(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Connecting => "connecting",
            Self::Generating => "writing",
            Self::ToolRunning => "running tool",
            Self::WaitingForApproval => "waiting for approval",
            Self::Compacting => "compacting",
            Self::Cancelling => "canceling · awaiting acknowledgment",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::OutcomeUnknown => "outcome unknown",
            Self::Interrupted => "interruption acknowledged",
            Self::NotStarted => "not sent",
            Self::LocalCancelled => "local cleanup complete · remote stop unconfirmed",
            Self::Queued => "queued / awaiting upstream",
            Self::Thinking => "thinking",
            Self::Counting => "counting tokens",
        }
    }
    fn decode(value: u8) -> Self {
        match value {
            0 => Self::Preparing,
            1 => Self::Connecting,
            2 => Self::Generating,
            3 => Self::ToolRunning,
            4 => Self::WaitingForApproval,
            5 => Self::Compacting,
            6 => Self::Cancelling,
            7 => Self::Completed,
            8 => Self::Failed,
            9 => Self::OutcomeUnknown,
            10 => Self::Interrupted,
            11 => Self::NotStarted,
            13 => Self::Queued,
            14 => Self::Thinking,
            15 => Self::Counting,
            _ => Self::LocalCancelled,
        }
    }
}

#[derive(Default)]
struct Render {
    text: String,
    skipped: bool,
    first_text: Option<Instant>,
    characters: usize,
}
/// Thinking text never shares reply bytes or the durable fact queue. Completed
/// markers preserve tool ordering; completed model text repairs preview gaps.
pub struct ThinkingFrame {
    pub id: u64,
    pub text: String,
    pub tokens: Option<u64>,
    pub completed: bool,
    pub limited: bool,
}
#[derive(Default)]
struct ThinkingRender {
    text: String,
    skipped: bool,
    active: bool,
    completed: VecDeque<ThinkingFrame>,
    bytes: usize,
    next_id: u64,
}
struct Shared {
    receiver: Thread,
    render: Mutex<Render>,
    thinking: Mutex<ThinkingRender>,
    state: AtomicU8,
    notice: AtomicBool,
    closed: AtomicBool,
}
pub struct Sender<F, R> {
    shared: Arc<Shared>,
    facts: wake::Sender<F>,
    done: wake::Sender<R>,
}
pub struct Inbox<F, R> {
    shared: Arc<Shared>,
    facts: Receiver<F>,
    done: Receiver<R>,
}

pub fn lanes<F, R>() -> (Sender<F, R>, Inbox<F, R>) {
    let shared = Arc::new(Shared {
        receiver: thread::current(),
        render: Mutex::new(Render::default()),
        thinking: Mutex::new(ThinkingRender::default()),
        state: AtomicU8::new(State::Preparing as u8),
        notice: AtomicBool::new(false),
        closed: AtomicBool::new(false),
    });
    let (facts, facts_rx) = wake::channel(FACTS);
    let (done, done_rx) = wake::channel(1);
    (
        Sender {
            shared: Arc::clone(&shared),
            facts,
            done,
        },
        Inbox {
            shared,
            facts: facts_rx,
            done: done_rx,
        },
    )
}

impl<F, R> Sender<F, R> {
    pub fn state(&self, state: State) {
        if self.shared.state.swap(state as u8, Ordering::Relaxed) != state as u8 {
            self.shared.receiver.unpark();
        }
    }
    pub fn notice(&self) {
        self.shared.notice.store(true, Ordering::Release);
        self.shared.receiver.unpark();
    }
    pub fn fact(&self, fact: F) -> io::Result<()> {
        self.facts
            .try_send(fact)
            .map_err(|_| io::Error::other("control receipt queue unavailable"))
    }
    pub fn finish(&self, result: R) {
        let _ = self.done.try_send(result);
    }
    pub fn render(&self, text: &str) -> bool {
        if self.shared.closed.load(Ordering::Acquire) {
            return false;
        }
        let Ok(mut render) = self.shared.render.lock() else {
            return false;
        };
        if !text.is_empty() {
            render.first_text.get_or_insert_with(Instant::now);
            render.characters = render.characters.saturating_add(text.chars().count());
        }
        if render.skipped {
            return true;
        }
        if text.len() > RENDER_BYTES.saturating_sub(render.text.len()) {
            // Stop the provisional stream at its first gap. The authoritative
            // final result repairs display; facts/results never share this lane.
            render.skipped = true;
        } else {
            render.text.push_str(text);
        }
        drop(render);
        self.shared.receiver.unpark();
        true
    }
    pub fn thinking_delta(&self, text: &str) {
        if self.shared.closed.load(Ordering::Acquire) || text.is_empty() {
            return;
        }
        let Ok(mut render) = self.shared.thinking.lock() else {
            return;
        };
        render.active = true;
        if !render.skipped {
            if text.len() > RENDER_BYTES.saturating_sub(render.text.len()) {
                render.skipped = true;
            } else {
                render.text.push_str(text);
            }
        }
        drop(render);
        self.shared.receiver.unpark();
    }
    pub fn thinking_complete(&self, text: &str, tokens: Option<u64>, limited: bool) -> Option<u64> {
        if self.shared.closed.load(Ordering::Acquire) {
            return None;
        }
        let Ok(mut render) = self.shared.thinking.lock() else {
            return None;
        };
        let active = render.active;
        render.text.clear();
        render.skipped = false;
        render.active = false;
        let id = if active || !text.is_empty() {
            let cap = vade_harness::terminal::MAX_TRANSCRIPT_BYTES;
            let mut end = text.len().min(cap);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            while render.bytes.saturating_add(end) > cap {
                let Some(old) = render.completed.pop_front() else {
                    break;
                };
                render.bytes = render.bytes.saturating_sub(old.text.len());
            }
            render.bytes += end;
            render.next_id += 1;
            let id = render.next_id;
            render.completed.push_back(ThinkingFrame {
                id,
                text: text[..end].into(),
                tokens,
                completed: true,
                limited: limited || end < text.len(),
            });
            Some(id)
        } else {
            None
        };
        drop(render);
        self.shared.receiver.unpark();
        id
    }
}

impl<F, R> Inbox<F, R> {
    pub fn state(&self) -> State {
        State::decode(self.shared.state.load(Ordering::Relaxed))
    }
    /// Visible-text estimate only, never a claim about provider token counts.
    pub fn estimated_rate(&self) -> Option<f32> {
        let render = self.shared.render.lock().ok()?;
        let elapsed = render.first_text?.elapsed().as_secs_f32();
        (elapsed >= 0.1).then(|| render.characters as f32 / 4.0 / elapsed)
    }
    pub fn notice(&self) -> bool {
        self.shared.notice.swap(false, Ordering::AcqRel)
    }
    pub fn fact(&self) -> Result<F, TryRecvError> {
        self.facts.try_recv()
    }
    pub fn result(&self) -> Result<R, TryRecvError> {
        self.done.try_recv()
    }
    pub fn render(&self) -> (String, bool) {
        let Ok(mut render) = self.shared.render.lock() else {
            return (String::new(), true);
        };
        let mut end = FRAME_BYTES.min(render.text.len());
        while !render.text.is_char_boundary(end) {
            end -= 1;
        }
        let rest = render.text.split_off(end);
        let text = std::mem::replace(&mut render.text, rest);
        // A producer burst can coalesce to one wake token. Keep draining bounded
        // frames without parking for a timer when more display data is ready.
        if !render.text.is_empty() {
            self.shared.receiver.unpark();
        }
        (text, render.skipped)
    }
    pub fn thinking(&self) -> VecDeque<ThinkingFrame> {
        let Ok(mut render) = self.shared.thinking.lock() else {
            return VecDeque::new();
        };
        let mut frames = VecDeque::new();
        if !render.text.is_empty() || render.skipped {
            let mut end = FRAME_BYTES.min(render.text.len());
            while !render.text.is_char_boundary(end) {
                end -= 1;
            }
            let rest = render.text.split_off(end);
            let text = std::mem::replace(&mut render.text, rest);
            frames.push_back(ThinkingFrame {
                id: 0,
                text,
                tokens: None,
                completed: false,
                limited: render.skipped,
            });
            if !render.text.is_empty() {
                self.shared.receiver.unpark();
            }
        }
        frames
    }
    pub fn completed_thinking(&self, id: Option<u64>) -> VecDeque<ThinkingFrame> {
        let Ok(mut render) = self.shared.thinking.lock() else {
            return VecDeque::new();
        };
        let frames = if let Some(id) = id {
            render
                .completed
                .iter()
                .position(|frame| frame.id == id)
                .and_then(|at| render.completed.remove(at))
                .into_iter()
                .collect()
        } else {
            std::mem::take(&mut render.completed)
        };
        for frame in &frames {
            render.bytes = render.bytes.saturating_sub(frame.text.len());
        }
        frames
    }
    pub fn discard_render(&self) {
        if let Ok(mut render) = self.shared.render.lock() {
            render.text.clear();
            render.skipped = true;
        }
    }
    pub fn close(&mut self) {
        self.shared.closed.store(true, Ordering::Release);
        let (_, empty) = mpsc::channel();
        self.facts = empty;
        let (_, empty) = mpsc::channel();
        self.done = empty;
    }
}
impl<F, R> Drop for Inbox<F, R> {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    #[test]
    fn every_worker_lane_wakes_a_parked_receiver_without_timer_polling() {
        for lane in 0..5 {
            let (sender_tx, sender_rx) = mpsc::channel();
            let (ready_tx, ready_rx) = mpsc::channel();
            let worker = thread::spawn(move || {
                let (tx, rx) = lanes::<usize, usize>();
                sender_tx.send(tx).unwrap();
                // Unpark-before-park must also work: the producer may publish
                // just before this call. Five seconds is a failure fallback,
                // never the mechanism that makes the receipt available.
                thread::park_timeout(Duration::from_secs(5));
                let observed = match lane {
                    0 => rx.state() == State::Generating,
                    1 => rx.notice(),
                    2 => rx.fact() == Ok(17),
                    3 => rx.result() == Ok(23),
                    _ => rx.render().0 == "ready",
                };
                ready_tx.send(observed).unwrap();
            });
            let tx = sender_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            match lane {
                0 => tx.state(State::Generating),
                1 => tx.notice(),
                2 => tx.fact(17).unwrap(),
                3 => tx.finish(23),
                _ => assert!(tx.render("ready")),
            }
            let result = ready_rx.recv_timeout(Duration::from_secs(2));
            worker.thread().unpark();
            worker.join().unwrap();
            assert_eq!(result, Ok(true), "lane {lane} failed to wake its receiver");
        }
    }
    #[test]
    fn render_flood_cannot_delay_control_or_terminal_result() {
        let (tx, rx) = lanes::<usize, usize>();
        let started = Instant::now();
        for _ in 0..50_000 {
            assert!(tx.render("a burst of provisional output 🦀"));
        }
        tx.state(State::Cancelling);
        tx.fact(17).unwrap();
        tx.finish(23);
        assert_eq!(rx.state(), State::Cancelling);
        assert_eq!(rx.fact().unwrap(), 17);
        assert_eq!(rx.result().unwrap(), 23);
        let (text, skipped) = rx.render();
        assert!(skipped);
        assert!(text.len() <= FRAME_BYTES);
        vade_harness::diagnostics::stderr_line(format_args!(
            "render flood: 50000 pushes, receipt and result available in {} us (local in-memory fixture)",
            started.elapsed().as_micros()
        ));
    }
    #[test]
    fn receipts_are_bounded_explicit_and_do_not_consume_result_capacity() {
        let (tx, rx) = lanes::<usize, usize>();
        for i in 0..FACTS {
            tx.fact(i).unwrap();
        }
        assert!(tx.fact(999).is_err());
        tx.finish(42);
        assert_eq!(rx.result().unwrap(), 42);
        for i in 0..FACTS {
            assert_eq!(rx.fact().unwrap(), i);
        }
    }
    #[test]
    fn close_releases_returned_resources_and_stops_rendering() {
        let (tx, mut rx) = lanes::<(), Arc<()>>();
        let resource = Arc::new(());
        tx.finish(Arc::clone(&resource));
        assert_eq!(Arc::strong_count(&resource), 2);
        rx.close();
        assert_eq!(Arc::strong_count(&resource), 1);
        assert!(!tx.render("discarded"));
        assert!(tx.fact(()).is_err());
    }
    #[test]
    fn render_coalesces_preserves_unicode_and_bounds_each_frame() {
        let (tx, rx) = lanes::<(), ()>();
        let text = "🦀".repeat(6000);
        for chunk in text.as_bytes().chunks(400) {
            tx.render(std::str::from_utf8(chunk).unwrap());
        }
        let (first, skipped) = rx.render();
        let (second, _) = rx.render();
        assert!(!skipped);
        assert_eq!(first.len(), FRAME_BYTES);
        assert_eq!(format!("{first}{second}"), text);
    }

    #[test]
    fn followups_preserve_order_and_return_rejected_input_within_two_bounds() {
        let mut queue = WorkQueue::default();
        for i in 0..8 {
            queue.push(i.to_string()).unwrap();
        }
        assert_eq!(queue.push("retain me".into()), Err("retain me".into()));
        for i in 0..8 {
            assert_eq!(queue.pop(), Some(i.to_string()));
        }
        queue.push("x".repeat(64 * 1024)).unwrap();
        assert_eq!(queue.push("y".into()), Err("y".into()));
        queue.clear();
        queue.push("new prompt".into()).unwrap();
        assert_eq!(queue.pop().as_deref(), Some("new prompt"));
    }
}
