# Stored checkpoint prefix cell

`checkpoint-prefix-check` joins the stored embedding to attention mHC and input norm.
It uses the first layer's description and admitted checkpoint files.
It uses the existing safe device owners and kernels.
It preserves stored parameter precision.
It adds no execution kind or service Factory.

The probe stops before the first projection.
The separate [MXFP8 projection cell](MXFP8-PROJECTION.md) extends this prefix through selected output rows.
It does not execute a complete layer or publish causal state.
Each prompt position has an independent prefix before the causal mixer.
The prefix is replicated on each rank.
There is no rank transport in this cell.

The load policy is an explicit stored-identity candidate.
It does not qualify the checkpoint runtime's effective loads or quantized replacements.
The [runtime type map](../model/RUNTIME-TYPES.md) still applies.
The [cast contract](../model/CAST-CONTRACT.md) still refuses complete layer admission.
The [first-token driver](../../tools/bench/FIRST-TOKENS.md) still needs a complete decoder.

## Run the cell

Build on the Mac. Ship the prebuilt Linux AArch64 binary.
Do not run a compiler on a Spark.
Prepare `prompt.ids` with the first-token driver's shared tokenizer and template.
Each line must contain an input token ID.
The prompt limit comes from `prompt` in the probe source.

```sh
./checkpoint-prefix-check suite "$MODEL" prompt.ids NEW_RESULT "$RANK" "$PARTS"
./checkpoint-prefix-check compare RESULT_A/run1 RESULT_B/run1
```

The suite starts fresh device child processes.
The repeat count comes from `suite` in the probe source.
It compares each device output with the fixed primitive host oracle.
It also compares each device output with the first device output.
It prints input IDs and the SHA256 of the output manifest for each run.
These IDs are prompt IDs. They are not output token IDs.

The device copies each selected stored embedding row through the existing file uploader.
The embedding kernel repeats that single-row view across residual streams.
The probe then executes attention mHC, collapse, and input norm on the same backend.
It retains the residual and coefficient owners until the norm completes.
It saves each prefix plane before it checks that plane's exact bytes.
A failed check can stop the run before later planes are saved.
A failed result has no completion marker.
Keep failed bytes and logs. Do not increase a tolerance to accept them.

`IDENTITY` pins the source, config, index, prompt IDs, and consumed parameter values.
It pins each selected embedding row, not every unused checkpoint payload.
`OUTPUT.sha256` seals the identity, scope, input IDs, and output planes.
`RANK` labels station metadata separately from the replicated byte comparison.
The station must retain process and hardware logs for each rank.
The comparator proves only the saved prefix bytes.
It does not independently prove rank identity or fresh remote processes.

## Remaining cells

The first missing compute producer follows the prefix and appears in `IDENTITY` as `next`.
The checkpoint still needs complete projection bindings, mixer execution, and FFN execution.
It also needs real causal state, completed rank gathers, and final text integration.
Effective load policy and M2/M3 qualification remain open.
No prefix result proves output token IDs, a complete decoder, or speed.
