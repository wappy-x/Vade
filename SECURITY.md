# Security policy

## Report a security problem

Use the repository's **Security** tab and choose **Report a vulnerability**.
Send the report privately. Do not put exploit details in a public issue, pull
request or discussion.

If private reporting is not available, use a private maintainer contact listed
by the repository. If none is listed, ask for a private reporting route without
posting the exploit.

Include the Vade version or source commit, the command and settings, and the
steps that show the problem. State what the attacker can access and where the
attacker runs. Include a small log sample if it helps. Remove keys, tokens,
private prompts, machine addresses and personal data before you send it.

## What is in scope

Report faults in code that Vade ships:

- Engine request parsing, streams, API key checks and cache access.
- Gateway access checks, request routing and data sent to an engine.
- CLI installation, remote commands, tunnels and files that Vade writes.
- Harness tools, file and command access, and credential handling.

Problems in the operating system, NVIDIA driver, CUDA, SSH or a model checkpoint
belong to those projects. Wrong model answers are normally bugs. An answer
that bypasses a Vade access check is a security report. A report must show what
access the fault adds; access the caller already has is not a new boundary.

## The gateway is local

The gateways bind only to loopback. The default address is `127.0.0.1`.
They also refuse connections whose peer address is not loopback. A program on
the same machine can reach the gateway. The gateway does not check which
local user owns that program.

The gateway checks the HTTP `Host` header and refuses browser `Origin` headers
unless you allow them. `--allow-host` and `--allow-origin` change these checks.
These checks do not authenticate a local program. A local program can send
its own headers and call the API routes that the gateway serves. Those routes
can use the configured engines, including engines reached through SSH tunnels.
See the [gateway routes and bind checks](harness/src/gateway.rs) and
[access checks](harness/src/gateway/access.rs).

A gateway you start yourself has no token by default. Use `--token` or
`VADE_GATEWAY_TOKEN` to require a bearer token. The flag takes precedence.
The gateway consumes that token and does not send it to the engine. An allowed
browser preflight is answered before the token check; it does not call the
engine.

The gateway started by `vade up --replicas` has no token. The CLI removes
`VADE_GATEWAY_TOKEN` from its environment and does not pass `--token`.
Setting that variable before `vade up` does not protect the replica gateway.
Local programs can use it without a credential. See the
[replica launcher](cli/src/app/replicas.rs).

Use the gateway only on a machine whose local programs and users you trust.
The gateway speaks plain HTTP. A token does not encrypt traffic. Keep tokens
out of source files, shell history and reports.

## Engine access

The installed CLI reaches the engine through SSH tunnels. Engine servers use
loopback by default. A direct remote bind needs both `--allow-remote` and
`--api-key-file`. Loopback serving does not require an API key by default.
See the [engine key checks](engine/crates/serve/src/auth.rs).

If your report needs a changed bind address, allowed host, allowed origin or
authentication setting, include that setting. This policy describes the
current access rules. It does not establish that a security review has passed.
