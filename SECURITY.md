# Security policy

## Supported versions

We ship security fixes only for the current public release, v0.1.0. Older
releases and development snapshots are not supported.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting for wappy-x/vade](https://github.com/wappy-x/vade/security/advisories/new).
Do not open a public issue with vulnerability details.

Include the affected Vade version or commit, the exact command and
configuration that reproduce the issue, its impact, and relevant excerpts from
logs or receipts (the saved records of a run). Remove credentials, private
prompts and other sensitive data from attachments. The maintainers will use
the private report to coordinate investigation, fixes and disclosure.

## Scope

This policy covers Vade's engine HTTP server (`flash-serve` / `next-http`), the
CLI, the harness tools and the gateway. The server normally listens on
`127.0.0.1:18080` and the gateway on `127.0.0.1:8787`. By default only programs
on the same machine can reach them (loopback access), and that is intended. If
you changed the bind address, authentication settings or tool permissions in a
way that affects your report, include those changes.
