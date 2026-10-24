//! Exact-admission HTTP child adapter. It uses the shared transport/contract,
//! typed response parser and existing native loop. No inference implementation.
use super::{ChildModel, ChildWorker, Factory, TokenMeter};
use crate::agent::ModelRequest;
use crate::agent_wire::AssistantTurn;
use crate::contract::{Context, Contract, InputCount, Usage, UsageCollector};
use crate::json;
use crate::prompt::AppendOnlyRequest;
use crate::transport::{self, ConnectionPool, Endpoint};
use std::io::{self, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Ready-to-bind native read/list factory. Workspace keeps its directory-handle
/// boundary, while each child owns its transport, contract and model history.
pub fn read_only_http_factory(
    endpoint: &str,
    workspace: Arc<crate::tools::Workspace>,
) -> io::Result<Arc<Factory>> {
    Endpoint::parse(endpoint)?;
    let endpoint = endpoint.to_owned();
    Ok(Arc::new(move |_, cancel| {
        check_cancel(cancel)?;
        let read = Arc::clone(&workspace);
        let list = Arc::clone(&workspace);
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(HttpChildModel::new(&endpoint)?),
            tools: Box::new(crate::agent::WorkspaceTools::new(
                move |path: &str, offset, limit| {
                    Ok(read
                        .read_range(path, offset, limit)
                        .unwrap_or_else(|e| format!("Read refused: {e}")))
                },
                move |path: &str| list.list(path),
            )),
        })
    }))
}

pub struct HttpChildModel {
    endpoint: Endpoint,
    pool: ConnectionPool,
    prompt: AppendOnlyRequest,
    contract: Contract,
}

impl HttpChildModel {
    /// No DNS or connection occurs until an authorized child actually runs.
    /// Every instance negotiates its own capability-gated cache scheduling key.
    pub fn new(endpoint: &str) -> io::Result<Self> {
        Ok(Self {
            endpoint: Endpoint::parse(endpoint)?,
            pool: ConnectionPool::default(),
            prompt: AppendOnlyRequest::default(),
            contract: Contract::default(),
        })
    }

    pub fn context(&self) -> &Context {
        &self.contract.context
    }

    pub fn with_generation_options(mut self, options: crate::contract::GenerationOptions) -> Self {
        self.prompt = AppendOnlyRequest::default();
        self.contract = Contract::with_options(options);
        self
    }
}

impl ChildModel for HttpChildModel {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        meter: &mut TokenMeter,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        check_cancel(cancel)?;
        if request.options.get("stream") != Some(&json::Value::Bool(false))
            || request.options.get("prompt_cache_key").is_some()
        {
            return Err(transport::invalid(
                "child request requires nonstreaming, independently negotiated affinity",
            ));
        }
        let model = request
            .options
            .get("model")
            .and_then(json::Value::as_str)
            .ok_or_else(|| transport::invalid("child model missing"))?;
        self.contract
            .negotiate(&self.endpoint, model, &self.pool, Some(cancel))?;
        if !self
            .contract
            .context
            .capabilities
            .as_ref()
            .is_some_and(|c| c.exact_admission)
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "child token quotas require exact admission; no generation sent",
            ));
        }
        let options = self.contract.decorate(request.serialized_options())?;
        let body = self.prompt.prepare(
            &options,
            request.messages.iter().map(|m| m.serialized_json()),
        )?;
        self.contract
            .admit(&self.endpoint, body, &self.pool, Some(cancel))?;
        let (prompt_tokens, output_tokens) = match self.contract.context.input {
            InputCount::Exact {
                prompt_tokens,
                output_tokens,
                admitted: true,
                ..
            } => (prompt_tokens, output_tokens),
            _ => return Err(transport::invalid("child exact admission unavailable")),
        };
        // The counted bytes are the exact bytes dispatched below. No second
        // decoration, negotiation, hidden prompt or cache-key substitution.
        meter.reserve(prompt_tokens, output_tokens)?;
        check_cancel(cancel)?;
        let mut response =
            transport::request_native_pooled(&self.endpoint, body, cancel, &self.pool, None)?;
        if !(200..300).contains(&response.status) {
            return Err(io::Error::other("child generation HTTP failure"));
        }
        if !response.head.get("content-type").is_some_and(|kind| {
            kind.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json")
        }) {
            return Err(transport::invalid(
                "child generation requires a JSON response",
            ));
        }
        let bound = request.wire.max_stream_bytes.min(json::MAX_INPUT_BYTES);
        let mut bytes = Vec::new();
        response
            .body
            .by_ref()
            .take(bound as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > bound {
            return Err(transport::invalid("child response exceeds byte quota"));
        }
        check_cancel(cancel)?;
        let value = json::parse_bytes(&bytes)?;
        let turn = AssistantTurn::from_response(&value, request.wire)?;
        let mut usage = UsageCollector::default();
        usage.observe(json::view(
            std::str::from_utf8(&bytes)
                .map_err(|_| transport::invalid("child response encoding"))?,
        )?);
        self.contract.completed(usage);
        if let Some(last) = self.contract.context.turns.back() {
            if last.invalid
                || last.usage.prompt_tokens.is_some_and(|n| n != prompt_tokens)
                || last
                    .usage
                    .completion_tokens
                    .is_some_and(|n| n > output_tokens)
            {
                // A provider violating its own admitted bound is not success;
                // do not execute the returned tools or refund its reservation.
                return Err(transport::invalid("child usage contradicts admission"));
            }
        }
        if let Some(text) = turn
            .message()
            .value()
            .get("content")
            .and_then(json::Value::as_str)
        {
            on_text(text)?;
        }
        check_cancel(cancel)?;
        response.recycle();
        Ok(turn)
    }

    fn usage(&self) -> Option<Usage> {
        self.contract.context.turns.back().map(|t| t.usage.clone())
    }
}

fn check_cancel(cancel: &AtomicBool) -> io::Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "child cancelled locally",
        ))
    } else {
        Ok(())
    }
}
