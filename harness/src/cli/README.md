# harness/src/cli

**The interactive terminal program: reading your input, running turns, showing results.**

| File | What it holds |
| --- | --- |
| [`mod.rs`](mod.rs) | `App`, the running program's state, and how it starts from a `Config`. |
| [`args.rs`](args.rs) | Command-line options: the mode (`demo`, `direct`, `agent`, `gateway`, delegated modes), the `Config` and permission levels. |
| [`ui.rs`](ui.rs) | The main input loop, status line and notices. |
| [`commands.rs`](commands.rs) | Slash commands typed in the interface, such as `/status`, and their shared metadata. |
| [`help.rs`](help.rs) | Help text, command lookup, and suggestions for mistyped commands. |
| [`worker.rs`](worker.rs) | Starting, steering and cancelling a turn, and running its tools. |
| [`events.rs`](events.rs) | Messages from the background worker back to the interface, and turn results. |
| [`approval.rs`](approval.rs) | Tool previews and approval prompts, shown in full before anything is approved. |
| [`native_setup.rs`](native_setup.rs) | Native agent conversations: building messages, compacting long sessions, finishing turns. |
| [`admission.rs`](admission.rs) | Exact admission and context-budget checks before a turn starts. |
| [`attachments.rs`](attachments.rs) | Explicit local-file attachments and completion. |
| [`steering.rs`](steering.rs) | Prompt steering and queued follow-up controls. |
| [`answers.rs`](answers.rs) | Tracking the latest answer: its sampling seed, retrying it, and cut-off answers in resumed sessions. |
| [`delegated.rs`](delegated.rs) | Handing a turn to an installed outside agent, and finding its program. |

Unit tests live beside their owning module (`args/tests.rs`, `commands/tests.rs`, `ui/tests.rs`, and `events/summary_tests.rs`), with small tests inline. [`test_support.rs`](test_support.rs) shares argument parsing for these fixtures. Startup-order tests remain in [`main/tests.rs`](../main/tests.rs).

**Why a folder:** the terminal program is large, so its parts are split by job.

**How it fits:** `main.rs` hands control to this module; the library code it drives lives in [`src/`](../README.md). Usage is in the [harness README](../../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
