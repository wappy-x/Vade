# Cache defaults and their evidence

The native harness already requests prefix caching when the engine advertises
cache support or a cache-affinity hint. The FP4-Local engine keeps a retained
prefix cache by default on one and two Sparks. These are separate controls:
sending `cache:true` cannot enable a server feature that was disabled at startup.

| Surface | Default | What the control means |
| --- | --- | --- |
| Direct and native agent harness | Request `cache:true` when supported; keep a stable session key when advertised | Permission for the engine to reuse a prefix, with exact admission seeing the same decorated request |
| `VADE_HARNESS_PREFIX_CACHE=0` | Unset | Request `cache:false`; keep the transcript and advertised affinity key unchanged |
| Gateway | Preserve caller cache fields | `--stable-prefix` explicitly enables supported request decoration; routing affinity alone is not a cache hit |
| FP4-Local engine `VADE_FLASHNEXT_PREFIX_CACHE` | Unset: retention on, on one Spark and on the two-Spark tensor split; off on the layer chain (three or more Sparks) | `0` turns retention off; `1` on the layer chain is refused |

The request rules are in [contract.rs](../src/contract.rs); the server defaults
and pipeline refusal are in [prefix_switch.rs](../../engine/crates/serve/src/prefix_switch.rs).
An explicit caller `cache` value is preserved. An unknown or absent engine
contract does not enable custom cache decoration.

## Engine retention default

The engine keeps prefixes by default on one Spark and on the tensor split. Each
engine start takes a memory room for the cache. On one Spark the room never
lowers the number of full-context chats; on the tensor split it holds one
full-context copy up to 4 GiB (all of a 32,768 context) and can lower that number by one (see [the prefix cache](../../INSTALL.md#the-prefix-cache)).
Requests without `cache:true` or a `prompt_cache_key` do not keep prefixes, so
their work does not change. The default-on gate measures repeated-prefix
savings, cold requests, concurrent admission, memory pressure and exact
restored token IDs in one session.

The 2026-10-22 engine qualification found a useful repeated-prefix benefit and a
separate capacity cost. These are historical engine measurements, not new
harness benchmark results:

| Measured configuration | Evidence and limit |
| --- | --- |
| One Spark, 32K context, retained-cache candidate on | 18/18 reuse checks; 96.8–99.8% reported reuse; first token 0.171–0.548 s, with an uncached reference of 6.122 s; compared outputs were exact |
| Earlier cache-on concurrency gate | Four-request throughput regressed 2.09%; a rerun regressed 1.54%. A full retained cache left no spare KV pages, adding about 72 ms per admission |
| Revised cache implementation, retention off | 928 recorded rows and 144 exactness-family checks had zero mismatches; nine timed cells tied and the tenth reported a win associated with a disclosed baseline outlier |
| 262K context, cache on | No prefix was retained; this is not evidence of long-context cache reuse |
| Layer pipeline | Cache on is refused because the back rank does not checkpoint and restore its own layers |

The revised implementation's **cache-on performance was not qualified** by the
default-off gate. Its default-off success does not erase the earlier capacity
regression or prove a current on/off speedup. A default-on decision needs a new
same-session engine gate covering repeated-prefix savings, cold requests,
concurrent admission, memory pressure and exact restored token IDs.

The 2026-10-23 one-Spark launcher smoke kept engine retention off. The actual
direct request and six engine/gateway replays produced the same 16 visible token
IDs; `cache:false/true/true` reported zero cached tokens. That verifies the
current request path and honest zero-hit reporting. It does not test a retained
cache or replace the engine's numerical gate.

## Host cost of requesting caching

A loopback fixture advertised `prefix_cache:false` and a supported affinity key,
matching the distinction above. For each of direct and native-agent mode, 31
runs per setting alternated unset, off and on: **186 actual CLI processes**.
Every generation matched its exact-admission request. After removing only the
cache boolean and random session key, all request fields were equal within each
mode. All replies reported zero cached tokens; the fixture cannot retain GPU KV.

| Mode | Unset / on request bytes | Off request bytes | Capability / admission calls per run |
| --- | ---: | ---: | ---: |
| Direct | 519 | 520 | 1 / 1 |
| Native agent | 15,281 | 15,282 | 1 / 1 |

The one-byte difference is `true` versus `false`. Enabling request permission
added no network round trip or transcript text. In the latest rerun (2026-10-23,
release build of the current harness source), paired on-minus-off total-process
medians were −0.005 ms for direct and −0.085 ms for native, with observed
ranges of −1.863 to +1.125 ms and −2.139 to +3.906 ms respectively; an earlier
same-day run measured +0.450 ms native. These noisy process timings do not
establish a cache-speed difference in either direction. Native completion
includes its normal journal and UI work; these are not engine TTFT measurements.
Fresh sessions without sampling flags carry no sampling field: neither mode
inserts a bookkeeping seed, so the upstream's default sampling policy applies.

The committed process test covers both modes, four combinations of retention
and affinity advertisements, and unset/off/on settings: 24 successful processes.
It asserts one generation, unchanged messages, exact admission, the expected
request flag/key and displayed zero cached tokens. It also asserts that no
sampling field is sent and that, within each combination, the three settings
produce identical requests apart from `cache` and the per-session
`prompt_cache_key`:

```sh
cd harness
cargo test --offline --test contract_cli both_cli_modes_negotiate_cache_permission
```

## Recommended defaults

Keep negotiated harness requests on and keep gateway cache decoration opt-in.
Keep the engine retention default: on for one Spark and the tensor split, off
on the layer chain. A request keeps a prefix only with `cache:true` or a
`prompt_cache_key`. This preserves the available repeated-prefix optimization
without claiming that a request flag changes server capacity or enables an
unqualified pipeline path.

Read cache savings from completed response usage, never from prompt similarity,
a stable key or a successful connection. Zero means a reported miss; missing
telemetry means unknown. Cached input still occupies context. See the
[engine contract](engine-contract.md) and [prefix contract](performance.md).
