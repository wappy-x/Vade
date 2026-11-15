# Boundary phase device gate

This runbook is prepared. No device result is claimed. Use a booked pair after the base gate passes. Do not merge on host results alone.

READ: The base is `rows-launch` at `b3ceb8c0f3e571de927b50633608b955bfcb1812`. Use the frozen candidate revision from the pull request. Build each kernel image from its own source with the pinned toolchains. The commands in `AGENTS.md` use host `1.99.0` and kernel `nightly-2026-10-01`. Keep both image hashes, build revisions, checkpoint identity, and settings with the receipts.

READ: The new diagnostic flag is `--diagnostic-boundary-gate=DIR`. It requires `--diagnostic-unqualified-text --mtp 3`. It runs through the real served `Factory`, `Session`, and `Engine::step_batch` API. It does not open an HTTP listener. Worker ranks run the normal worker loop. The controller reads different prompts from `DIR/0.ids` through `DIR/7.ids`. Each file contains whitespace-separated token IDs from the prepared prompt kit. Output files must not exist.

READ: The gate saves complete scalar reference logit words and IDs, including the first prompt result. It compares each kept row of each changing cohort against that chat alone. It runs plain decode, strict drafting, and a separate plain pass with device token selection for timing. Drafting calls the real proposer. It replaces proposals on alternate ticks with the reference continuation or a wrong first proposal to force both full and partial acceptance. Those forced proposals are diagnostic cases. Only the separate plain pass is used for step timing. This covers the served engine API, not the HTTP scheduler's arrival policy.

READ: The diagnostic harness is absent from the base. Apply only its source overlay to a separate base gate kit. This adds the diagnostic entry and leaves the base's GPU code, boundary dispatch, and admission code intact. From the candidate checkout, prepare:

```sh
git diff b3ceb8c0f3e571de927b50633608b955bfcb1812 HEAD -- \
  engine/crates/tp/src/bin/flash-serve/text.rs \
  engine/crates/tp/src/bin/flash-serve/boundary_gate.rs \
  engine/crates/tp/src/flash/settings.rs > "$KIT_PREP/boundary-gate-only.patch"
```

Apply that patch only to the base kit. Record its hash as well as the base revision. Build `flash-serve` in both kits. Build `mhc-host-check` and the separate benchmark workspace's `kernel-bench` in the candidate kit. Build both production and TC images before the host binaries. Use the existing kit builder and pinned compilers, with `--locked -j 2`. The production image is embedded when the host binaries build; the PTX argument alone does not replace it. Set `BASE_PTX` and `CANDIDATE_PTX` to the corresponding production images. These steps must finish before the scarce device slot starts.

The chief supplies the existing private pair settings. In the commands below, `KIT` is the selected base or candidate kit; `RUN` is a fresh controller receipt directory; `CONTEXT`, `MODEL`, `LISTEN`, `NEXT`, and `BIND` come from the booked pair's existing setup. Do not put their private values in this report. Keep the same inputs and settings in every load. The following is the single ordered device command list.

1. Run all compiled byte arms on the booked device:

   ```sh
   "$CANDIDATE_KIT/kernel-bench" --mhc-phased "$BASE_PTX" "$CANDIDATE_PTX" > "$OUT/bytes.log"
   ```

   Required lines: `BOUNDARY BROKEN PHASE REJECTED missing=finish`, `BOUNDARY BYTES PASS`, and `BOUNDARY GRAPH SEQUENCE PASS`. The arms are base fused, candidate fused, direct phases, and recorded phases. Scratch stays dirty through changed inputs and shrinking cohorts. All arms use distinct poisons in both rounds. Every logical output word, input guard, output guard, scratch canary, inactive scratch row, and command-table guard must pass. Both norm output forms run with both strip forms and varied norm weights. A missing finish phase must fail the byte verdict.

2. Run the production recorder against eager dispatch:

   ```sh
   "$CANDIDATE_KIT/mhc-host-check" phased-rows > "$OUT/recorder.log"
   ```

   Required line: `BOUNDARY RECORDER SEQUENCE PASS`. This retains boundary owners through `8,5,4,3,4,5,8` rows. It checks both first and deferred boundaries, both weight forms, both norm forms, all admitted phase tuples, every exposed output plane, and the actual recorded boundary-node counts. The byte check in the first command also checks norm partial words.

3. Run the pair loads in this order: **M1 C1 M2 C2 M3 C3**. M is the base plus only the diagnostic harness overlay. C is the frozen candidate. For each load, make a fresh `RUN` on the controller and copy the same eight prepared ID files into it. In the two booked rank terminals, run the matching kit command below, with the rank's existing private arguments. Start the worker first, then the controller. Wait for both to exit successfully before the next load. No process-by-name termination is needed.

   ```sh
   "$KIT/flash-serve" "$RANK" 2 "$LISTEN" "$NEXT" "$KIT/flash.ptx" \
     "$MODEL" "$CONTEXT" "$BIND" 0 \
     --diagnostic-unqualified-text --mtp 3 \
     "--diagnostic-boundary-gate=$RUN" > "$RANK_LOG" 2>&1
   ```

   Every controller must print `BOUNDARY PAIR PASS changing=1 drafting=off,on`. Its `BOUNDARY COHORT PASS` lines must cover full-logit plain decode, full-logit strict drafting, and plain device token selection. The source sequence contains cohorts of `1,2,4,8,5,4,3,4,5` chats with row rotation and private progress. It repeats until every chat finishes. Returning chats keep their state. The two draft verdicts must both run.

   After collecting controller receipt directories under `OUT`, check raw reference bytes and IDs across all interleaved loads:

   ```sh
   "$CANDIDATE_KIT/kernel-bench" --boundary-receipts compare 90 \
     "$OUT/M1" "$OUT/C1" "$OUT/M2" "$OUT/C2" "$OUT/M3" "$OUT/C3"
   ```

   Required line: `BOUNDARY RECEIPTS PASS mode=compare`. This is a byte comparison of the full saved rows, not a text comparison or a logit tolerance. READ: `90` is the prepared boundary count for the product fixture: `45` layers in `crates/model/tests/fixtures/glm/config.json` and two boundary sites in `crates/model/src/plan/layer.rs`. If the checkpoint's prepared count differs, use that count and record its source.

4. Report startup and first-step bytes from those same loads:

   ```sh
   "$CANDIDATE_KIT/kernel-bench" --boundary-receipts memory 90 \
     "$OUT/M1" "$OUT/C1" "$OUT/M2" "$OUT/C2" "$OUT/M3" "$OUT/C3"
   ```

   Required line: `BOUNDARY RECEIPTS PASS mode=memory`. It requires a startup receipt and a first-step receipt for every chat. The gate checks current and peak accounted device bytes against the actual startup budget. Retain both ranks' `TEXT ADMISSION` lines too. READ: the candidate separately reserves `50,304` bytes for each prepared fused boundary owner, from `mhc_params::FUSED_SCRATCH_BYTES`. It does not rely on the shared margin to fund that scratch.

5. Report the raw plain step times and boundary launch counts:

   ```sh
   "$CANDIDATE_KIT/kernel-bench" --boundary-receipts times 90 \
     "$OUT/M1" "$OUT/C1" "$OUT/M2" "$OUT/C2" "$OUT/M3" "$OUT/C3"
   ```

   Required line: `BOUNDARY RECEIPTS PASS mode=times`. It prints every saved four-chat and eight-chat sample from the plain device-selection pass. The verdict also requires timed cohorts at one and two chats. READ expected boundary nodes per rank per plain step: base `90/180/360/720`, candidate `90/180/270/270`, at `1/2/4/8` chats. A missing width, a wrong count, a nonfinite time, unequal IDs or logits, or missing memory evidence fails the receipt check. These counts exclude other kernels and do not count host graph submissions.

Do not turn a host test or PTX instruction count into a measured speed claim. Keep the raw samples, warm-up policy, and per-load results. Device exactness, memory fit, and a visible time gain at the wider cohorts remain to be established by this run.
