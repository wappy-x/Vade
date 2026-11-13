# Historical SGLang corrections

The [earlier optimization research](earlier-optimization-research.md) preserves
loading, cached-chat, speculation and measurement lessons from the September
audits, with a [source-hash manifest](github-home-source-manifest.json).

These records describe September–October 1 comparison experiments, not the
current Vade baseline. The [final promotion decision](exactness-decision-2026-10-01.md)
was later corrected; raw artifacts remain private.

| Correction | Evidence and boundary |
| --- | --- |
| “Exact” had limited scope | Recorded A/A reproducibility passed. Decode versus prefill correctness had no independent reference, and lookup-on quality equivalence was not demonstrated. A nonsignificant difference does not prove equivalence. |
| Throughput protocols differ | Historical SGLang C8 rates divide all completion tokens by the longest individual request duration, in the follow-up measurements with at most 512 tokens. The Flash-Next benchmark rate divides by group wall time from first send to last DONE with 256 outputs and MTP off. Different models, rank counts and sampling further limit comparison. |
| Headline receipts existed outside exports | Qwen TP3 at 8,660 prompt tokens: 2.87 s TTFT, C8 178.2/169.5 tok/s thinking off/on. GLM TP4 baseline at 8,648: 4.00 s, 81.5/85.6; final at 8,650: 3.59 s, 92.8/98.9. Selected numeric fields were read directly from the raw JSON captures below and match the ledger; their hashes are in the privately kept import manifest. These measurements were not rerun. |
| Send/receive was later fixed | The corrected ring history records approximately 23.9 GB/s forward/reverse with zero wrong values. The earlier README failure is historical, not a present network conclusion. |
| Top-k provenance was corrected | A retained top-k patch (private, not imported) identifies a modified SGLang implementation. No implementation is imported. Transfer independent specifications and tests with attribution. |
| Incomplete hash comparisons can pass | The historical hash checker skipped missing ranks. A new gate must require the expected rank and target/draft coverage before equality counts. The old checker is not adopted here. |
| Fleet setup ages | October 1 cabling notes describe the historical deployment. Current role mapping and fleet addresses come only from ignored local configuration. |

| Retained raw JSON | Selected fields |
| --- | --- |
| Historical Qwen TP3 speed capture (private) | `prefill.8000`, `scale.nothink_8`, `scale.think_8` |
| Historical GLM baseline speed capture (private) | Same selected numeric fields |
| Historical candidate speed capture (private) | Same selected numeric fields |

The raw JSON stays private. The current Vade baseline uses a different recipe, detailed in the
[receipt catalog](../../../bench/receipts/README.md).

The import manifest, kept privately, records source ledger,
source-relative locators, dates and raw SHA256 hashes. Detailed BEFORE-VS-NOW
and experiment ledgers stay local; they contain many cross-load qualifications
and private logs. Upstream-derived patches, implementations, profiles, corpora and raw dumps
are excluded. Those originals remain untouched.
