use std::io;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Receiver,
};
use vade_harness::{app_server, protocol::Completion};

use super::{
    RunState,
    events::{TurnResult, WorkSender, WorkerEvent, WorkerResult},
    worker::emit_delta,
};
use crate::delegation;

pub(crate) fn executable_path(path: &std::path::Path) -> io::Result<PathBuf> {
    // Preserve PATH lookup for a bare name. Resolve an explicit relative path
    // before the core changes the child's working directory to the workspace.
    if path.is_relative()
        && path
            .parent()
            .is_some_and(|parent| !parent.as_os_str().is_empty())
    {
        Ok(std::env::current_dir()?.join(path))
    } else {
        Ok(path.to_path_buf())
    }
}
pub(crate) fn delegated_turn(
    existing: Option<delegation::Backend>,
    options: delegation::Options,
    text: &str,
    cancel: &AtomicBool,
    controls: &Receiver<app_server::Control>,
    tx: &WorkSender,
) -> WorkerResult {
    let client = match existing {
        Some(client) if client.matches(&options) => Ok(client),
        Some(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "delegated backend does not match selected provider",
        )),
        None => delegation::Backend::connect(options, cancel),
    };
    let mut client = match client {
        Ok(client) => client,
        Err(error) => {
            return WorkerResult {
                http: None,
                result: Err(error),
                backend: None,
            };
        }
    };
    if client.session_id().is_empty()
        || client.session_id().len() > 512
        || client
            .session_id()
            .chars()
            .any(|c| c.is_control() || c.is_whitespace())
    {
        return WorkerResult {
            http: None,
            result: Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid thread identifier",
            )),
            backend: None,
        };
    }
    if tx
        .fact(WorkerEvent::ThreadReady(client.session_id().to_string()))
        .is_err()
        || cancel.load(Ordering::Relaxed)
    {
        return WorkerResult {
            http: None,
            result: Ok(TurnResult::NotStarted),
            backend: None,
        };
    }
    let mut emitted = 0usize;
    let mut delivery_failed = false;
    let result = client.turn(text, cancel, controls, &mut |event| match event {
        app_server::Event::AgentMessageDelta(delta) => {
            emit_delta(tx, cancel, &mut emitted, &delta);
        }
        app_server::Event::AgentMessageCompleted { .. } => (),
        app_server::Event::Phase(phase) => tx.state(match phase {
            app_server::Phase::Generating => RunState::Generating,
            app_server::Phase::ToolRunning => RunState::ToolRunning,
            app_server::Phase::WaitingForApproval => RunState::WaitingForApproval,
            app_server::Phase::Compacting => RunState::Compacting,
            app_server::Phase::Cancelling => RunState::Cancelling,
        }),
        app_server::Event::Steering { request_id, status } => {
            if tx
                .fact(WorkerEvent::Steering { request_id, status })
                .is_err()
            {
                delivery_failed = true;
                cancel.store(true, Ordering::Relaxed);
            }
        }
        app_server::Event::Notice(_) => {
            tx.notice();
        }
    });
    let result = if delivery_failed {
        Err(io::Error::other(
            "steering receipt delivery failed; outcome unconfirmed",
        ))
    } else {
        result
    };
    let result = result.and_then(|outcome| match outcome {
        app_server::TurnOutcome::Completed => client
            .completed_text()
            .map(|text| {
                TurnResult::Completed(Completion {
                    text: text.to_owned(),
                    finish_reason: None,
                })
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "completed delegated turn has no authoritative answer",
                )
            }),
        app_server::TurnOutcome::Interrupted => Ok(TurnResult::Interrupted),
        app_server::TurnOutcome::NotStarted => Ok(TurnResult::NotStarted),
        app_server::TurnOutcome::Failed(_) => Ok(TurnResult::Failed),
    });
    // Protocol/IO failure invalidates this connection. The known thread identity
    // stays in the UI so the next explicit prompt can reconnect with resume.
    let backend = if result.is_ok() { Some(client) } else { None };
    WorkerResult {
        result,
        backend,
        http: None,
    }
}
