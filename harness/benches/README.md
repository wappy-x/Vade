# Gateway passthrough benchmark

Run from `harness/`:

```sh
RUSTFLAGS='-D warnings' cargo +1.99.0 bench --offline --locked -p vade-harness --bench gateway_passthrough
```

The benchmark uses the default `vade-gateway` relay from the library. It starts
the relay in a child process with an empty environment. It uses no user config
or credential files. It stops only its own child. It needs no engine or GPU.

The fixed request set is in `gateway_passthrough.rs`. It covers health, models,
chat JSON, chat SSE, Responses, a long chat input, and a tool request. The mock
engine uses `std::net::TcpListener`. It checks each request body byte for byte.
The client checks the status and each response body byte for byte.
It decodes the gateway's HTTP chunks after the timer stops.

The mock uses the server's connection framing. It closes after each request.
JSON replies have `Content-Length`. SSE replies end when the connection closes.
The mock sends the full SSE reply with no delay. It does no model work.

The command output names the warmup count, round count, sample count, and body
sizes. Each pair sends the same request directly to the mock and through the
gateway. The order reverses on each pair. Concurrency stays at one request.
The timer starts before TCP connect and ends at response EOF. The first byte
timer ends at the first HTTP byte. Request construction and response checks
stay outside the client timer. Mock checks run in both paths.

`added_p50_ns` is the nearest-rank median of signed paired differences. Negative
samples stay in the result. `difference_of_p50_ns` is the gateway median minus
the direct median. The output also gives added p95 and first byte added p50.
Each case has equal weight in the combined result. All units are nanoseconds.
The final rows contain all measured pairs. Output starts after measurement.

Save the command output with the source revision, Rust version, OS version,
and hardware class. Run on an idle host. These results measure local relay
cost. They do not measure model speed, GPU exactness, load, or remote network
cost. A normal test run prints the benchmark command and makes no measurement.
