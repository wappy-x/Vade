# Security policy

This page tells you how to report a security problem in Vade privately. It
also says what is in scope and which access rules the gateways and the engine
apply today.

## Report a security problem

1. Open the repository's **Security** tab and choose **Report a
   vulnerability**.
2. Send the report privately. Do not put exploit details in a public issue,
   pull request or discussion.

If private reporting is not available, use a private maintainer contact that
the repository lists. If the repository lists none, ask for a private
reporting route. Do not post the exploit.

Put these items in the report:

- **Version.** The Vade version or the source commit.
- **Command.** The command and its settings.
- **Steps.** The steps that show the problem.
- **Attacker.** What the attacker can access and where the attacker runs.
- **Log.** A small log sample, if it helps.

Remove keys, tokens, private prompts, machine addresses and personal data
before you send the report.

## What is in scope

Report faults in code that Vade ships:

- Engine request parsing, streams, API key checks and cache access.
- Gateway access checks, request routing and data sent to an engine.
- CLI installation, remote commands, tunnels and files that Vade writes.
- Harness tools, file and command access, and credential handling.

Problems in the operating system, the NVIDIA driver, CUDA, SSH or a model
checkpoint belong to those projects. A wrong model answer is normally a bug.
An answer that bypasses a Vade access check is a security report. A report
must show what access the fault adds. Access that the caller already has is
not a new boundary.

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

`vade-gateway` accepts arbitrary HTTP paths for the engines you configure.
With replicas, it answers `GET /health` locally from engine probe results.
Other paths can reach a configured engine. It can use an engine key from
`--api-key-file` or `VADE_API_KEY_FILE`. A caller who
passes the gateway checks can use that key's engine access. The caller can
submit generation, cancellation and cache release requests. The engine still
checks those requests. With replicas, cancellation and cache release go to
every configured engine. A request cannot select a new upstream address.
See the [standalone gateway](harness/docs/invisible-gateway.md).

`vade-harness gateway` accepts its [listed API routes](harness/src/gateway.rs)
and a local cluster status route. It refuses other paths. Neither gateway runs
client tools, reads workspace files for a client or starts shell commands from
a request. A configured engine can expose more routes. The gateways do not
give each local user a separate set of engine permissions.

A gateway you start yourself has no token by default. Use `--token` or
`VADE_GATEWAY_TOKEN` to require a bearer token. The flag takes precedence.
The gateway consumes that token and does not send it to the engine. An allowed
browser preflight is answered before the token check. It does not call the
engine.

Without a gateway token, a caller's `Authorization` header can reach the
engine. A configured engine key replaces that header on upstream requests.
The gateway token and the engine key serve different access checks.

The gateway started by `vade up --replicas` has no token. The CLI removes
`VADE_GATEWAY_TOKEN` from its environment and does not pass `--token`.
Setting that variable before `vade up` does not protect the replica gateway.
Local programs can use it without a credential. See the
[replica launcher](cli/src/app/replicas.rs).

Use the gateway only on a machine whose local programs and users you trust.
The gateway speaks plain HTTP. A token does not encrypt traffic. Keep tokens
out of source files, shell history and reports.

The [access checks](harness/src/gateway/access.rs) apply before a request can
reach an engine:

<div align="center">

| Request or setting | Result |
| :---: | :---: |
| Bind name resolves to a non-loopback address | Startup fails |
| Peer address is not loopback | Connection is dropped |
| `Host` is missing or `Host` or `Origin` is duplicated | HTTP 400 |
| `Host` fails the loopback name or optional port check and is not allowed | HTTP 421 |
| `Origin` is not allowed | HTTP 403 |
| Configured gateway token is missing or wrong | HTTP 401 |
| Allowed browser preflight | HTTP 204 from the gateway; no engine request |
| Local request passes these checks | The gateway can send it to the configured engine |

</div>

`--allow-host` and `--allow-origin` do not permit a non-loopback bind or peer.
They do not check which local user sends a request.
A loopback `Host` may omit the port. An explicit port must match the listener
unless `--allow-host` admits the full header value.

### Check the gateway on a host

From the repository root, run these existing tests with the pinned Rust
toolchain. They use local socket fixtures and a mock CLI runner. They need no
GPU, model, SSH connection or running engine:

```sh
export RUSTFLAGS='-D warnings'
cargo +1.99.0 test --offline --locked --manifest-path harness/Cargo.toml --lib gateway:: -- --nocapture
cargo +1.99.0 test --offline --locked --manifest-path cli/Cargo.toml --bin vade replicas_start_world_one_servers_with_own_tunnels_behind_an_owned_gateway -- --nocapture
```

The gateway tests cover bind refusal, Host and Origin checks, tokens, local
preflight replies, token removal, custom paths and replica cancellation. The
CLI test checks that the replica launcher removes inherited authentication
settings. These host tests do not prove a live client's operation or GPU
exactness.

## Engine access

The installed CLI reaches the engine through SSH tunnels. Engine servers use
loopback by default. A direct remote bind needs both `--allow-remote` and
`--api-key-file`. Loopback serving does not require an API key by default.
See the [engine key checks](engine/crates/serve/src/auth.rs).

The engine also uses plain HTTP. An API key does not encrypt prompts or
credentials. Use a trusted SSH tunnel or TLS termination for remote access.

If your report needs a changed bind address, allowed host, allowed origin or
authentication setting, include that setting. This policy describes the
current access rules. It does not establish that a security review has passed.
