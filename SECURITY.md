# Security

Vade runs an inference engine on hardware you own. Please report anything that puts a running engine, its models, or the machines around it at risk.

## Report a problem

Use GitHub private vulnerability reporting on this repository: **Security** tab, then **Report a vulnerability**. Reports reach the maintainers only. Do not open a public issue for a security problem.

## What counts

- Anything in `vade`, `vade chat`, or `spark-serve` that lets one request read or change another request, files it should not reach, or the host.
- Remote code execution, path traversal, or unbounded resource use through the OpenAI-compatible API, the installer, or the model pull.
- A break in exactness: the same request with the same settings returning different token IDs across runs or Spark counts.
- Build or install steps that need more privilege than the README states.

## Scope

This repository, its installer scripts, and the releases it publishes. NVIDIA drivers, third-party model data, and services Vade does not ship are out of scope.

A report should name the commit it was tested on and the exact command that shows the problem. Maintainers answer within 7 days. Fixes land in a public release with credit to the reporter unless the reporter asks to stay unnamed.
