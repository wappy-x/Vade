# Quickstart: one Spark, from nothing to a first answer

You need a DGX Spark with a terminal, a free GPU and 112 GB of free disk (the model is 102.3 GB; Vade needs 9 GiB, about 9.7 GB, on the same disk). The installer checks the GPU, its driver and the free disk before it starts, and says what is missing. It needs no sudo.

**1. Install.** It prints `Install complete. Next steps:` near the end. Then do what its last line says: open a new terminal, or run the `export PATH=...` command it prints.

```sh
curl -fsSL https://vade.lol/install.sh | sh
```

**2. Get the model** (103 GB, one time). If the download stops, run the same line again. It goes on where it stopped.

```sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
```

**3. Start.** It ends with `Up ready` and the address of the engine. For the time the first start takes, see [first-run time](INSTALL.md#first-run-time-and-records).

```sh
vade up
```

**4. Check.** It prints one line for each check, then `PASS` or `FAIL`. A failed check says what to do next.

```sh
vade doctor
```

**5. Ask a question.** Type `/exit` to leave. The engine keeps running.

```sh
vade chat
```

- **Stop:** `vade down`
- **Update:** `vade down`, then the install line again, then `vade up`
- **Remove Vade and keep the model:** `curl -fsSL https://vade.lol/install.sh | sh -s -- --uninstall`

More: [two Sparks](INSTALL.md#two-and-more-sparks), [every setting](INSTALL.md), [the page for agents](agents/install.md).
