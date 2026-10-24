# Run, update, remove

**Keep Vade running in the background, update it to a new revision, or remove it, all as an ordinary user.**

For anyone who has finished [the canonical one-Spark guide](../../docs/quickstart-one-spark.md) and operates a Spark.

These service recipes use one-Spark `spark-serve`. For the current two-Spark
Flash server, use [INSTALL's shutdown](../../INSTALL.md#6-run-the-timing-benchmark)
and the [Flash connection guide](flash-server.md).

- Nothing here uses `sudo` or changes the OS, driver or network.
- Commands run from the monorepo’s `engine/` workspace.

## Keep it running

**Two ways, pick one: a systemd user service (Option A) or a detached background process (Option B).**

Stop the foreground server from [quickstart, step 4](../../docs/quickstart-one-spark.md#4-start-the-server) first.

- **Option A, a systemd user service:** best when you want automatic
  restarts on failure. Logs go to the user journal.
- **Option B, a background process:** best when you prefer not to use a
  systemd service. Logs go to a log file in a folder you choose.

A systemd user service is a background program that systemd runs under your
own account, not as the system administrator.

### Option A: a systemd user service

1. Create a local service file. The checkout path must have no spaces. The
   `runtime` folder is local configuration; keep it out of commits and exports.

   ```sh
   mkdir -p runtime
   cat > runtime/vade-serve.service <<EOF
   [Unit]
   Description=Vade inference server
   [Service]
   WorkingDirectory=$PWD
   ExecStart=$PWD/target/release/spark-serve $PWD/models/Qwen/Qwen3-8B
   Restart=on-failure
   [Install]
   WantedBy=default.target
   EOF
   ```

2. Link it, start it and read its log:

   ```sh
   systemctl --user link "$PWD/runtime/vade-serve.service"
   systemctl --user daemon-reload
   systemctl --user enable --now vade-serve.service
   journalctl --user -u vade-serve.service -n 30 --no-pager
   ```

   Wait for the same `ready` line, then repeat quickstart step 5's question.

   Whether the service keeps running after you log out, or starts at boot,
   depends on the Spark's existing user-service policy. This guide does not
   change it.

3. To stop it:

   ```sh
   systemctl --user stop vade-serve.service
   ```

### Option B: a background process, without systemd

1. Start it detached, with a local folder for its state. `--daemon` writes a
   log and a PID file (the process number) under `runtime/state/vade/`.

   ```sh
   XDG_STATE_HOME="$PWD/runtime/state" ./target/release/spark-serve \
     models/Qwen/Qwen3-8B --daemon
   ```

2. Read its log, then repeat quickstart step 5's question:

   ```sh
   tail -n 30 runtime/state/vade/spark-serve.log
   ```

3. To stop it:

   ```sh
   kill -TERM "$(cat runtime/state/vade/spark-serve.pid)"
   ```

### Ask your coding agent

```text
Keep Vade running on this DGX Spark using "Keep it running" in
docs/upkeep.md. Use Option A (systemd user service) unless I ask for Option B.
Stop the foreground server from quickstart step 4 first. Work from the existing
engine/ workspace with the same model folder. Run as the current user. Never use
sudo or change the OS, driver, network or user-service policy. Show the ready
line from the log and one answer. Do not commit, push or publish the repository.
```

## Update

**Stop, pull, rebuild, restart.** A successful build does not prove the answers still match.

### Do it yourself

1. Stop the service or background process with its stop command above.

2. From the engine/ workspace, fetch the new revision and look at what changes
   the build:

   ```sh
   git fetch
   git diff HEAD..origin/main -- ../rust-toolchain.toml crates/gpu/build-kernels
   git pull --ff-only
   ```

3. If the pinned toolchain versions changed, install the new ones first, as
   in [quickstart, step 1](../../docs/quickstart-one-spark.md#1-install-the-pinned-rust-tools).

4. Repeat [quickstart, step 2](../../docs/quickstart-one-spark.md#2-get-the-source-and-build-the-engine), restart the same way as
   before, and repeat quickstart step 5's question.

An update that changes the math needs the existing model checks.
[Validation](validation.md) records them and their measured limits.

### Ask your coding agent

```text
Update the existing Vade checkout using docs/upkeep.md's "Update" steps.
Stop its server, inspect the incoming revision, rebuild both kernels and
server, then restart with the same model and settings. Run as the current
user. Never use sudo or change the OS, driver or network. Stop on errors.
Show the ready line and one answer. Do not commit or push.
```

## Remove Vade

**Stop the server, disable the service, then remove the checkout and `runtime` folders.**

### Do it yourself

1. Stop the server.

2. If you used Option A, disable the service. This removes its user-service
   link from the configured service folder:

   ```sh
   systemctl --user disable vade-serve.service
   ```

3. Look inside the checkout and `runtime` folders, then remove only those.
   Keep the model files if another program uses them.

### Ask your coding agent

```text
Remove Vade from this DGX Spark using docs/upkeep.md's "Remove Vade" steps.
Stop its server and, if Option A was used, disable the user service. List
what the checkout and runtime folders contain, then ask me before deleting
anything. Keep the model files unless I say otherwise. Never use sudo.
```

**Where next:** [Server and API](server.md) · [Models](models.md) ·
[Documentation index](README.md)
