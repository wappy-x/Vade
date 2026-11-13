# One-start kernel A/B measurements (test only)

Start the usual `flash-serve` command once with `VADE_AB_PTX="$X_PTX"`.
The positional PTX argument is arm A; the environment path is arm X. For
example, using the same values as your normal world-one FP4-Local launch:

```sh
VADE_AB_PTX="$X_PTX" "$FLASH_SERVE" "$RANK" "$WORLD" "$LISTEN" "$NEXT" \
  "$A_PTX" "$MODEL" "$CONTEXT" "$BIND" 0 --fp4-local --mtp 3
```

Keep your normal sampling, model, memory and kernel switches fixed. On a
multi-rank service set the flag on every rank; startup checks both PTX identities
through the existing artifact handshake. Missing or different X artifacts refuse
startup. Wait for `/v1/models` readiness before sending cells.

Send identical, sequential ABBA cells (A, X, X, A). Set `ENDPOINT` to the
usual HTTP origin and `MODEL_ID` to its advertised model ID:

```sh
cell=0
for arm in A X X A; do
  cell=$((cell + 1))
  curl --fail --silent --show-error "$ENDPOINT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -o "cell-$cell-$arm.json" \
    -w "cell=$cell arm=$arm first_byte_s=%{time_starttransfer} total_s=%{time_total}\n" \
    -d "{\"model\":\"$MODEL_ID\",\"arm\":\"$arm\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Rust function that sums a slice of integers.\"}],\"max_tokens\":256,\"temperature\":0,\"top_p\":1,\"top_k\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"cache\":false,\"draft\":\"plain\",\"enable_thinking\":false,\"include_usage_timing\":true}"
done
```

Read the usual `usage.vade.prefill_ms` and `usage.vade.decode_ms` fields
(`timing: admitted_wall`), usage token counts and effective sampling echo. Save
client timings alongside them. For MTP cells, repeat the same matrix with
`"draft":"strict"` (and a supported `mtp_depth` on FP4-Local); use identical
settings for all four cells. Warm each arm once outside timing to fund its
function tables and shared decode graphs. Record cold first requests separately.
Compare generated answer/token bytes, excluding request IDs and timing metadata.

The JSON field `"arm":"A" | "X"` applies to Completions, Chat Completions,
Responses and Messages. Omission selects A. Invalid values, including null,
return an error. With the environment flag absent, any `arm` field returns an
error explaining that test-only `VADE_AB_PTX` is required. Other servers do not
enable it by reading the environment.

The test service admits one request at a time. It selects the entire kernel table
before session allocation, including prefill, decode, helper and MTP functions;
workers receive the selection before opening a session. Both arms use one CUDA
context, one model/weight copy, the same tokenizer, ordered stream and shared
scratch/state storage. Function tables and shared CUDA graph handles are kept
per arm. Switching adds no branch to the kernel launch path. Startup prints
`gpu_backend ab_ptx_load_ms=...`; optional `VADE_JIT_RECEIPTS` records both
modules through the existing JIT receipt path.

Request prompt/prefix/KV retention is **disabled for both arms**, even with
`cache:true`; explicit session/fork/KV checkpoint fields are rejected. Draft
history is also never reused across requests. In-request prompt lookup starts
fresh and retains its normal policy. Immutable model file/page caches and weight
bytes are shared: ABBA measures kernels with common residency, not separate
checkpoint loads. CUDA graphs are arm-specific so an X replay cannot run A
handles. This is a single-chat measurement tool, not a concurrency benchmark.

Before trusting a candidate, the separate GPU proof must check the standard
bytegate streams against main in three cases: flag absent; flag present with
arm A; flag present with arm X where X is an exact copy of A's PTX. Include
plain/MTP and chunk boundaries in the usual gate, alternate arms repeatedly,
and measure request-selection overhead with identical PTX. Expected extra
per-launch overhead is zero; this CPU build does not establish GPU byte identity
or timing. Leave the flag unset for normal serving.

## Same-arm cohorts (`VADE_AB_COHORTS=1`)

With `VADE_AB_PTX` set, `VADE_AB_COHORTS=1` keeps the scheduler's normal pool
width instead of one request at a time, so C5 / C8 cells run on the one-start
A/B server. Admission takes only the queue head, and only while no active
session belongs to the other arm: a burst of arm-X requests runs as one cohort,
and the next arm waits until that cohort has ended. `select_arm` therefore still
runs only when no session of the other arm is live, and pooled decode graphs are
banked per arm. Prompt/prefix/KV retention stays disabled. Any value other than
`0` or `1`, or `1` without `VADE_AB_PTX`, refuses startup. Before timing a
candidate this way, its cells must match a fresh single-PTX start of the same
PTX byte for byte (and `VADE_LOAD_DIGEST=1` must match between the two loads).
