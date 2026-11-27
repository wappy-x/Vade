| Part | What it proves | Run tag | Result |
|---|---|---|---|
| 1 | fresh-server repeat, rule 1 | a-gate-main-a1-1+a-gate-main-a2-1 / a-gate-main-b1-1 | NOT PROVEN |
| 2 | served groups, rules 2/5; available word sweeps | a-gate-main-a1-1+a-gate-main-a2-1+a-gate-main-a3-3 | PASS |
| 3 | continuation and cache/cold, rule 3 | a-gate-main-a1-1+a-gate-main-a2-1+a-gate-main-a3-3 | PASS |
| 4 | strict/plain depths 1/3/7, rule 4 | a-gate-main-a1-1+a-gate-main-a2-1+a-gate-main-a3-3 | PASS |
| tile smoke control | parts 1/3 detect planted magnitude fault | a-gate-control-tile-1 | PASS |
| small smoke control | broad fault must change tokens | - | NOT PROVEN |

Rule 6 and full rule 2 require format-specific device-to-twin word sweeps. Smoke controls do not prove sensitivity to a different sum order.

Quality: NOT RUN.

SERVED CHECKS AND SMOKE CONTROLS: INCOMPLETE OR FAILED
LANDING: NOT PROVEN by this table. Format word/twin, sum-order control, path and quality review remain required.

Partial fresh repeat: **PASS, 3/3 one-chat lists, 768 tokens**,
`a-gate-main-a1-1` / `a-gate-main-b1-1`. Distinct fresh server starts,
identical binary/kernel hashes, cache disabled. This does not complete the
four-load repeat. The small control and quality remain NOT RUN under the
lead's single-turn limit. See `tc-gate-repeat-subset.txt` and
`tc-gate-tile-control.txt` for exact diagnostics.
