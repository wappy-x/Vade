//! Start a bounded app-server connection and validate its initial handshake.
use super::{
    Client, Limits, Options, PermissionPolicy, check_cancel, child, contract, invalid, object, string,
    valid_id,
};
use crate::json::Value;
use std::{io, process::Command, sync::atomic::AtomicBool, time::Instant};

impl Client {
    #[cfg(unix)]
    pub(super) fn connect_unix(options: Options, cancel: &AtomicBool, limits: Limits) -> io::Result<Self> {
        check_cancel(cancel)?;
        let workspace = options.workspace.canonicalize()?;
        if !workspace.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Codex workspace must be a directory",
            ));
        }
        let workspace_text = workspace
            .to_str()
            .ok_or_else(|| invalid("Codex workspace must be UTF-8"))?
            .to_owned();
        if workspace_text.len() > 8192
            || options.thread_id.as_deref().is_some_and(|id| !valid_id(id))
            || options
                .model
                .as_deref()
                .is_some_and(|model| model.len() > 1024 || model.chars().any(char::is_control))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid or oversized Codex option",
            ));
        }
        let sandbox = match options.permission {
            PermissionPolicy::ReadOnly => "read-only",
            PermissionPolicy::WorkspaceWrite => "workspace-write",
        };
        let executable =
            if options.executable.is_relative() && options.executable.components().count() > 1 {
                options.executable.canonicalize()?
            } else {
                options.executable.clone()
            };
        let mut command = Command::new(executable);
        command
            .arg("app-server")
            .arg("-c")
            .arg("approval_policy=\"never\"")
            .arg("-c")
            .arg(format!("sandbox_mode=\"{sandbox}\""))
            .arg("-c")
            .arg("sandbox_workspace_write.network_access=false")
            .arg("-c")
            .arg("sandbox_workspace_write.exclude_tmpdir_env_var=true")
            .arg("-c")
            .arg("sandbox_workspace_write.exclude_slash_tmp=true")
            .arg("-c")
            .arg("mcp_servers={}")
            .current_dir(&workspace);
        let wire = child::Connection::spawn(command)?;
        let mut client = Self {
            wire,
            thread_id: String::new(),
            workspace: workspace_text,
            permission: options.permission,
            model: options.model,
            next_id: 1,
            limits,
            completed_text: None,
        };
        let deadline = Instant::now() + client.limits.rpc;
        let initialized = client.rpc(
            "initialize",
            object(vec![
                (
                    "clientInfo",
                    object(vec![
                        ("name", string("vade_harness")),
                        ("title", string("Vade Harness")),
                        ("version", string("0.1.0")),
                    ]),
                ),
                (
                    "capabilities",
                    object(vec![("experimentalApi", Value::Bool(false))]),
                ),
            ]),
            deadline,
            cancel,
        )?;
        contract::initialize(&initialized)?;
        client.send_cancellable(
            object(vec![
                ("method", string("initialized")),
                ("params", object(vec![])),
            ]),
            deadline,
            Some(cancel),
        )?;
        let mut params = vec![
            ("cwd", string(&client.workspace)),
            ("approvalPolicy", string("never")),
            ("sandbox", string(sandbox)),
        ];
        if let Some(model) = &client.model {
            params.push(("model", string(model)));
        }
        let method = if let Some(id) = &options.thread_id {
            params.push(("threadId", string(id)));
            "thread/resume"
        } else {
            "thread/start"
        };
        let result = client.rpc(method, object(params), deadline, cancel)?;
        client.validate_policy(&result)?;
        let thread_id = result
            .get("thread")
            .and_then(|thread| thread.get("id"))
            .and_then(Value::as_str)
            .filter(|id| valid_id(id))
            .ok_or_else(|| invalid("app-server thread response has no valid id"))?;
        if options
            .thread_id
            .as_deref()
            .is_some_and(|expected| expected != thread_id)
        {
            return Err(invalid("app-server resumed a different thread"));
        }
        client.thread_id = thread_id.to_owned();
        Ok(client)
    }

}
