# gatecheck

**Measures:** the merge-gate verdict for a ladder of candidate loads against a head load. It checks the receipt inventory, token and logit bytes, decode and prompt speed against fixed floors, step spikes, graph captures, counters, temperatures and reference age. A missing or bad input fails. It never passes.

**Run:** `VADE_KIT_HOSTS=<private hosts file> ruby tools/kit/gatecheck/gatecheck.rb SPEC`. The header comment of `gatecheck.rb` gives the SPEC line format. Keep `captures_rule.rb` in the same folder.

**Prints:** one `RUNG ... VERDICT PASS|FAIL|VOID|BLOCKED|PENDING` line per rung, one `HIGHEST PASSING RUNG` line, then `GATECHECK END rungs=<n> exit=<code>`. Exit 0: every rung passes. Exit 1: a rung fails. Exit 2: a host gate is pending. Exit 3: an input or internal error, with a `GATECHECK ERROR` line.

**Needs:** Ruby and the evidence folders from the device gate runs. No GPU.

**Change from the reviewed copy:** `GOLDEN`, `LONGREF` and `REPO` now come from the environment or from the [hosts file](../hosts.example). They no longer have private built-in paths. [SOURCES.txt](../SOURCES.txt) gives both hashes.

**Rule exception:** these two files are Ruby, not Rust. They stay as they are until someone ports them, so that the reviewed logic does not change.
