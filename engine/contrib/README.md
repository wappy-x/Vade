# contrib

**Ready-made files for running the engine on a Spark.**

| File | What it does |
| --- | --- |
| [`vade-serve.service`](vade-serve.service) | A systemd user service that starts `spark-serve` with Qwen3-8B and restarts it if it stops. |

**Why:** so the server keeps running after you close the terminal, and starts again after a reboot.

**How it fits:** the steps that install it are in [Server and API, Keep it running](../docs/server.md#keep-it-running).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../docs/README.md)
