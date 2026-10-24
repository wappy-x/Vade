# Security policy

## Supported versions

Security fixes target the current public release only. The first planned public
release is v0.1.0; it is unreleased until its tag is published. Older releases
and development snapshots are not supported release lines.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting for wappy-x/vade](https://github.com/wappy-x/vade/security/advisories/new).
Do not open a public issue with vulnerability details.

Include the affected Vade version or commit, the exact command and configuration
needed to reproduce the issue, its impact, and relevant receipt or log excerpts.
Remove credentials, private prompts and other sensitive data from attachments.
The maintainers will use the private report to coordinate investigation, fixes
and disclosure.

## Scope

This policy covers Vade's engine HTTP server (`flash-serve` / `next-http`),
CLI, harness tools and gateway. The server normally listens on
`127.0.0.1:18080`, and the gateway on `127.0.0.1:8787`; loopback access is the
intended default. Include any changed bind address, authentication settings and
tool permissions that affect a report.
