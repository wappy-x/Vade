//! Allocator-accounted native session memory; local scripted tools/model only.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use vade_harness::{
    agent::{
        self, Conversation, Limits, Model, ModelRequest, StopReason, ToolError, ToolExecutor,
        ToolInvocation,
    },
    agent_wire::{AssistantTurn, FinishReason},
    json::{self, Value},
    prompt::AppendOnlyRequest,
};

struct CountAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Forward the unchanged layout to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe {
            System.dealloc(pointer, layout);
        }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(pointer, layout, size) };
        if !next.is_null() {
            if size >= layout.size() {
                LIVE.fetch_add(size - layout.size(), Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed);
            }
        }
        next
    }
}
struct Mock {
    prefix: AppendOnlyRequest,
    calls: usize,
}
impl Model for Mock {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        _: &AtomicBool,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        self.prefix.prepare(
            request.serialized_options(),
            request.messages.iter().map(|m| m.serialized_json()),
        )?;
        let response = if request.messages.last().unwrap().role() == "user" {
            self.calls += 1;
            format!(
                r#"{{"choices":[{{"message":{{"role":"assistant","tool_calls":[{{"id":"call-{}","type":"function","function":{{"name":"lookup","arguments":"{{}}"}}}}]}},"finish_reason":"tool_calls"}}]}}"#,
                self.calls
            )
        } else {
            r#"{"choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}"#.into()
        };
        AssistantTurn::from_response(&json::parse(&response)?, request.wire)
    }
}
struct Tools;
impl ToolExecutor for Tools {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        Ok("x".repeat(1024))
    }
}
fn main() -> io::Result<()> {
    for _ in 0..3 {
        let before = LIVE.load(Ordering::Relaxed);
        let mut conversation = Conversation::from_request(
            json::parse(
                r#"{"model":"mock","stream":true,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":"fixed system"}]}"#,
            )?,
            Limits::default(),
        )?;
        let mut model = Mock {
            prefix: AppendOnlyRequest::default(),
            calls: 0,
        };
        let empty = LIVE.load(Ordering::Relaxed) - before;
        for _ in 0..100 {
            conversation.push_user(
                Value::Object(vec![
                    ("role".into(), Value::String("user".into())),
                    ("content".into(), Value::String("u".repeat(256))),
                ]),
                Limits::default(),
            )?;
            let report = agent::run(
                &mut conversation,
                &mut model,
                &mut Tools,
                Limits::default(),
                &AtomicBool::new(false),
                &mut |_| {},
            );
            assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
        }
        let request = conversation.request(Limits::default().wire);
        model.prefix.prepare(
            request.serialized_options(),
            request.messages.iter().map(|m| m.serialized_json()),
        )?;
        let live = LIVE.load(Ordering::Relaxed) - before;
        let serialized = conversation.serialized_bytes();
        let prefix_capacity = model.prefix.retained_bytes();
        let n = 1000;
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(model.prefix.prepare(
                request.serialized_options(),
                request.messages.iter().map(|m| m.serialized_json()),
            )?);
        }
        let prepare_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(request.to_json()?);
        }
        let serialize_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
        let start = Instant::now();
        for _ in 0..n {
            let mut bytes =
                request.serialized_options()[..request.serialized_options().len() - 1].to_owned();
            bytes.push_str(",\"messages\":[");
            for (i, message) in request.messages.iter().enumerate() {
                if i > 0 {
                    bytes.push(',');
                }
                bytes.push_str(&json::stringify(message.value())?);
            }
            bytes.push_str("]}");
            std::hint::black_box(bytes);
        }
        let reserialize_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
        // Stack data, allocator overhead/fragmentation and UI/socket/thread stacks
        // are excluded. perf_mock reports whole-process peak resident bytes.
        drop(model);
        drop(conversation);
        println!(
            "session turns=100 tool_results=100 tool_result_bytes_each=1024 user_bytes_each=256 empty_live_heap_bytes={empty} populated_live_heap_bytes={live} serialized_history_bytes={serialized} prefix_capacity_bytes={prefix_capacity} prepare_us={prepare_us:.3} cached_record_serialize_us={serialize_us:.3} reserialize_owned_us={reserialize_us:.3} timing_n={n}"
        );
    }
    Ok(())
}
