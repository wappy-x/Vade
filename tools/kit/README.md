# Measurement kit

These are the lab's measurement tools in one place. Each folder has a short README: what the tool measures, how to run it and what it prints.

| Folder | Tool | Kind |
| --- | --- | --- |
| [gatecheck](gatecheck/README.md) | Merge-gate verdict for a ladder of loads | Copy (Ruby) |
| [ptx-check](ptx-check/README.md) | PTX byte-copy and register check | Copy (bash and Ruby) |
| [kernel-bench](kernel-bench/README.md) | Single-kernel bench on one Spark | Points to the engine tree |
| [step-direct](step-direct/README.md) | Direct Step launch byte and time screen | Points to the engine tree |
| [flywheel](flywheel/README.md) | One-load screen of many settings | Copy (Rust) |

**Private settings:** box names, network prefixes and local paths are not in this tree. Copy [hosts.example](hosts.example) to a private place, fill it in and set `VADE_KIT_HOSTS` to that copy. The tools stop with a clear error when a value is missing.

**Sources:** [SOURCES.txt](SOURCES.txt) gives the SHA-256 of each file as copied and as committed here.

**Safety:** only `kernel-bench`, `step-direct` and `flywheel` use a GPU. Run them only on a Spark that you claimed and that has no compute process. With no arguments, each tool prints its usage and touches no device.
