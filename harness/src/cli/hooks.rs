//! Chat-only hook wiring. Card and conformance callers never construct this.
use std::{io, sync::atomic::AtomicBool};
use vade_harness::{agent::{Conversation, Limits, RunObserver}, agent_wire::{Message, ToolCall, FinishReason}, context, hooks::Hooks, json::Value};
use super::events::{WorkSender, WorkerEvent};

pub(crate) fn notice(events: &WorkSender, text: String) {
    while events.fact(WorkerEvent::HookNotice(text.clone())).is_err() {
        if !events.render("") { return; }
        std::thread::park_timeout(std::time::Duration::from_millis(10));
    }
}

pub(crate) struct Observer<'a> {
    pub inner: &'a mut dyn RunObserver,
    pub hooks: &'a Hooks,
    pub events: &'a WorkSender,
}
impl RunObserver for Observer<'_> {
    fn before_tool(&mut self, call: &ToolCall, arguments: &Value, cancel: &AtomicBool) -> Option<String> {
        self.hooks.before(&call.name, arguments, cancel, &mut |text| notice(self.events, text))
    }
    fn after_tool(&mut self, call: &ToolCall, arguments: &Value, content: &mut String, cap: usize, cancel: &AtomicBool) {
        self.hooks.after(&call.name, arguments, content, cap, cancel, &mut |text| notice(self.events, text));
    }
    fn progress(&mut self, rounds: usize, limits: Limits, context: Option<(u64, u64)>, cancel: &AtomicBool) {
        self.inner.progress(rounds, limits, context, cancel);
    }
    fn auto_continued(&mut self, count: usize) { self.inner.auto_continued(count); }
    fn before_request(&mut self, c: &mut Conversation, limits: Limits, cancel: &AtomicBool) -> io::Result<bool> {
        self.inner.before_request(c, limits, cancel)
    }
    fn before_counted_request(&mut self, c: &mut Conversation, limits: Limits, cancel: &AtomicBool, usage: &context::Usage,
        count: &mut dyn FnMut(&Conversation) -> io::Result<context::Usage>) -> io::Result<bool> {
        self.inner.before_counted_request(c, limits, cancel, usage, count)
    }
    fn before_hosted_request(&mut self, c: &mut Conversation, limits: Limits, cancel: &AtomicBool, usage: &context::Usage,
        host: &mut dyn context::ContextHost) -> io::Result<bool> {
        self.inner.before_hosted_request(c, limits, cancel, usage, host)
    }
    fn checkpoint(&mut self, base: &Conversation, staged: &[Message]) -> io::Result<()> { self.inner.checkpoint(base, staged) }
    fn checkpoint_reply(&mut self, base: &Conversation, message: &Message, finish: &FinishReason) -> io::Result<()> {
        self.inner.checkpoint_reply(base, message, finish)
    }
}
