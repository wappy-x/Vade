# Security policy

## Supported versions

| Version | Security fixes |
| --- | --- |
| v0.1.x | Yes |
| Unreleased commits on `main` | No |

Fixes land on `main` and ship in the next v0.1.x release.

## Reporting a vulnerability

Please report vulnerabilities privately, through GitHub's private vulnerability
reporting:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability**.
3. Fill in the form. Only you and the maintainers can see it.

The form is also at
[github.com/wappy-x/vade/security/advisories/new](https://github.com/wappy-x/vade/security/advisories/new).
Please don't open a public issue, pull request or discussion about a
vulnerability before a fix is out.

If the **Security** tab has no **Report a vulnerability** button, or the form
above doesn't open for you, write to the maintainers at the contact listed in
the repository's **About** section.

A useful report includes:

- the Vade version (`vade --version`) or commit;
- the command, flags and configuration that reproduce the problem;
- what an attacker can do with it, and from where (the same machine, the local
  network, or the internet);
- the relevant lines from logs or receipts (the saved records of a run).

Remove credentials, API keys, private prompts and machine addresses before you
attach anything. We'll confirm that we received the report, work on a fix with
you in the private advisory, and agree on when to publish it.

## What is in scope

- **The engine's HTTP server** (`flash-serve` and `spark-serve`), including its
  API key check and how it handles requests and streams.
- **The gateways** (`vade-gateway`, which `vade up --replicas` starts, and
  `vade-harness gateway`), including the checks that refuse browser-driven
  requests.
- **The CLI's remote commands**, the ones that reach Sparks over SSH from a
  controller: `vade install`, `vade up`, `vade status`, `vade logs`,
  `vade verify` and `vade down`. That covers what they run on a Spark, the files
  they write there and the tunnels they open.

Problems in software Vade doesn't ship are out of scope: the NVIDIA driver and
CUDA, your SSH server or client, the operating system, or the model's
checkpoint. So is anything that needs an attacker who already controls your
account on the Spark or the controller. Wrong or unexpected model answers are
bugs rather than vulnerabilities; please report them through the normal issue
form, or, while Issues are off for this repository, through the maintainers'
contact listed in the repository's **About** section.

## How Vade is exposed by default

- The engine's HTTP server binds to loopback (`127.0.0.1`), so only programs on
  the same machine can reach it. With Vade's own config it listens on port
  18080. A controller reaches it through an SSH tunnel, not over the network.
- The server refuses to bind to any other address unless you pass
  `--allow-remote`, and `--allow-remote` refuses to start without
  `--api-key-file`. The server speaks plain HTTP, so put remote access behind a
  trusted tunnel or a TLS terminator.
- The gateways listen on loopback only and drop connections that come from
  anywhere else. A gateway you start yourself uses `127.0.0.1:8787` by
  default; the one `vade up --replicas` starts uses the controller's
  `local_port` (18080 by default). They also refuse requests with a
  non-loopback `Host` header or any `Origin` header, unless you name the host
  or browser origin you want to allow (`--allow-host`, `--allow-origin`).
- A gateway you start yourself (`vade-gateway` or `vade-harness gateway`) can
  also require a token, which is off by default. Pass `--token`, or set the
  `VADE_GATEWAY_TOKEN` environment variable; the flag wins if you use both.
  With a token set, every request must carry `Authorization: Bearer` followed
  by the token, or the gateway answers 401. The one exception is a browser's
  CORS preflight from an origin you allowed, which is answered before the
  check. The gateway does not pass the token on to the engine. The value must
  be 1 to 1024 visible ASCII characters. If you don't pass `--token` and
  `VADE_GATEWAY_TOKEN` is set to anything else, including an empty value, the
  gateway refuses to start.
- The gateway that `vade up --replicas` starts never uses a token. Vade
  doesn't pass it `--token` and removes `VADE_GATEWAY_TOKEN` from its
  environment, so exporting the variable before `vade up` does not protect it.
  It relies on its loopback and browser checks alone.

Treat the token as a password. Pick a long random value, keep it out of
committed files and shell history (for example, load it from a file only you
can read), and leave it out of issues and vulnerability reports. The token
does not encrypt anything: the gateways speak plain HTTP on loopback.

If your report depends on changing any of these defaults, tell us which ones
you changed.
