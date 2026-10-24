//! Reusable direct chat sessions and streaming completion callbacks.
use super::*;
use super::native::{ProgressObserver, ThinkingObserver, ThinkingProjection, generated_phase, notify_progress};

pub fn stream_chat(
    endpoint: &str,
    model: &str,
    messages: &[Message],
    on_delta: &mut dyn FnMut(&str),
) -> io::Result<Completion> {
    ChatClient::new(endpoint)?.stream(model, messages, on_delta, None)
}

/// Optional cancellation API; a pending socket read checks the flag every 250ms.
pub fn stream_chat_cancellable(
    endpoint: &str,
    model: &str,
    messages: &[Message],
    on_delta: &mut dyn FnMut(&str),
    cancel: &AtomicBool,
) -> io::Result<Completion> {
    ChatClient::new(endpoint)?.stream(model, messages, on_delta, Some(cancel))
}

/// A single session's reusable HTTP client. Construction performs no I/O.
pub struct ChatClient {
    pub(super) endpoint: Endpoint,
    pub(super) pool: ConnectionPool,
    contract: crate::contract::Contract,
    server_cancellation: bool,
    cancel_confirmed: Option<bool>,
    progress: Option<ProgressObserver>,
    thinking: Option<ThinkingObserver>,
    #[cfg(test)]
    pub(super) after_done_read_wait: Option<Box<dyn FnOnce() + Send>>,
}
impl ChatClient {
    /// Interactive Vade clients cancel by capability, including before token one.
    pub fn with_server_cancellation(mut self) -> Self {
        self.server_cancellation = true;
        self
    }
    pub fn server_cancel_confirmed(&self) -> Option<bool> { self.cancel_confirmed }

    pub fn set_progress_observer(&mut self, observer: Option<ProgressObserver>) {
        self.progress = observer;
    }
    pub fn set_thinking_observer(&mut self, observer: Option<ThinkingObserver>) {
        self.thinking = observer;
    }
    pub fn with_generation_options(mut self, options: crate::contract::GenerationOptions) -> Self {
        self.contract.options = options;
        self
    }
    pub fn new(endpoint: &str) -> io::Result<Self> {
        Ok(Self {
            endpoint: Endpoint::parse(endpoint)?,
            pool: ConnectionPool::default(),
            contract: crate::contract::Contract::default(),
            server_cancellation: false,
            cancel_confirmed: None,
            progress: None,
            thinking: None,
            #[cfg(test)]
            after_done_read_wait: None,
        })
    }
    pub fn context(&self) -> &crate::contract::Context {
        &self.contract.context
    }
    pub fn reset_context(&mut self) {
        self.contract = crate::contract::Contract::with_options(self.contract.options);
    }
    pub fn stream(
        &mut self,
        model: &str,
        messages: &[Message],
        on_delta: &mut dyn FnMut(&str),
        cancel: Option<&AtomicBool>,
    ) -> io::Result<Completion> {
        self.cancel_confirmed = None;
        notify_progress(&self.progress, NativePhase::Preparing);
        let result = self.stream_inner(model, messages, on_delta, cancel);
        notify_progress(&self.progress, NativePhase::Idle);
        result
    }
    fn stream_inner(
        &mut self,
        model: &str,
        messages: &[Message],
        on_delta: &mut dyn FnMut(&str),
        cancel: Option<&AtomicBool>,
    ) -> io::Result<Completion> {
        let started = Instant::now();
        let progress = self.progress.clone();
        let thinking_observer = self.thinking.clone();
        let mut thinking = ThinkingProjection::default();
        let mut phase = NativePhase::Queued;
        let mut prefix = crate::terminal::AnswerPrefix::default();
        let mut visible_text = |text: &str| {
            let text = prefix.visible(text);
            if !text.is_empty() {
                on_delta(&text);
            }
        };
        let check_cancel = || {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "request cancelled",
                ))
            } else {
                Ok(())
            }
        };
        check_cancel()?;
        self.contract.begin_run();
        self.contract
            .negotiate(&self.endpoint, model, &self.pool, cancel)?;
        let body = protocol::chat_request(model, messages)?;
        let automatic = self.contract.automatic_output(
            std::str::from_utf8(&body).map_err(|_| invalid("request encoding"))?,
        )?;
        let body = self
            .contract
            .decorate(std::str::from_utf8(&body).map_err(|_| invalid("request encoding"))?)?;
        let (body, control) = super::cancel::prepare(&self.endpoint, body.as_bytes(),
            self.server_cancellation && self.contract.context.capabilities.is_some())?;
        let body = self.contract.admit_adjusted(
            &self.endpoint,
            &body,
            &self.pool,
            cancel,
            automatic,
        )?;
        let (result, confirmed) = super::cancel::run(control, cancel, || {
        self.contract.context.requests.generations += 1;
        self.contract.context.last_http_stage = "generation";
        self.contract.context.last_http_status = None;
        let mut response = request_with_policy(
            "POST",
            &self.endpoint,
            &body,
            None,
            cancel,
            &self.pool,
            RequestPolicy {
                deadline: None,
                downstream: None,
                response_idle_timeout: IDLE_TIMEOUT,
                on_dispatched: Some(&|| notify_progress(&progress, NativePhase::Queued)),
                on_connect_failed: None,
            },
        )?;
        self.contract.context.last_http_status = Some(response.status);
        self.contract.context.requests.status(response.status);
        if !(200..300).contains(&response.status) {
            return Err(response.failure());
        }
        let content_type = response
            .head
            .get("content-type")
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let mut completion = Completion::default();
        let mut usage = crate::contract::UsageCollector::default();
        // A fully received answer stays complete even when steering lands in
        // the same instant; only an unfinished response can be cancelled,
        // and every path below this point has the whole response.
        if content_type.eq_ignore_ascii_case("text/event-stream") {
            self.contract.context.response_streamed = true;
            let mut decoder = SseDecoder::default();
            let done = std::cell::Cell::new(false);
            let mut event = |data: &str| -> io::Result<bool> {
                if done.get() {
                    return Err(invalid("chat data received after [DONE]"));
                }
                // Record [DONE] even if steering landed in the same
                // instant; EOF framing below still decides completion.
                if data.trim() == "[DONE]" {
                    done.set(true);
                    return Ok(false);
                }
                check_cancel()?;
                let event = protocol::apply_event_data(data, &mut completion, &mut visible_text)?;
                thinking.observe(event, &thinking_observer);
                if let Some(next) = generated_phase(data, phase)? {
                    let elapsed = started.elapsed();
                    self.contract
                        .context
                        .first_token_elapsed
                        .get_or_insert(elapsed);
                    self.contract.context.last_token_elapsed = Some(elapsed);
                    if next != phase {
                        phase = next;
                        notify_progress(&progress, phase);
                    }
                }
                usage.observe(event);
                check_cancel()?;
                Ok(false)
            };
            loop {
                // Framing is validated through EOF even after [DONE]: a
                // cancellation before EOF is an error, never a completion.
                #[cfg(test)]
                if done.get() {
                    response.body.reader.get_mut().on_read_wait = self.after_done_read_wait.take();
                }
                let bytes = response.body.fill_buf()?;
                if bytes.is_empty() {
                    break;
                }
                let n = bytes.len();
                decoder.feed(bytes, &mut event)?;
                response.body.consume(n);
            }
            if !done.get() || !decoder.line.is_empty() || decoder.has_data {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete chat SSE stream",
                ));
            }
        } else if content_type.eq_ignore_ascii_case("application/json") {
            let mut bytes = Vec::new();
            response
                .body
                .by_ref()
                .take((json::MAX_INPUT_BYTES + 1) as u64)
                .read_to_end(&mut bytes)?;
            notify_progress(&progress, NativePhase::Generating);
            protocol::apply_event(
                &json::parse_bytes(&bytes)?,
                &mut completion,
                &mut visible_text,
            )?;
            usage.observe(json::view(
                std::str::from_utf8(&bytes).map_err(|_| invalid("response encoding"))?,
            )?);
            thinking.observe(
                json::view(std::str::from_utf8(&bytes).map_err(|_| invalid("response encoding"))?)?,
                &thinking_observer,
            );
        } else {
            return Err(invalid(
                "upstream response must be JSON or text/event-stream",
            ));
        }
        response.recycle();
        self.contract.completed(usage);
        thinking.completed(&thinking_observer, None);
        Ok(completion)
        });
        self.cancel_confirmed = confirmed;
        result
    }
}
