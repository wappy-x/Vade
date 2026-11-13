# harness/tests/goldens

**Expected terminal screens.** A golden file is the known-good output a test compares against.

| File | What it is |
| --- | --- |
| `tui-80.txt`, `tui-160.txt` | The interface as plain text at 80 and 160 columns. |
| `tui-80.ansi`, `tui-160.ansi` | The same screens with terminal color codes. |
| `blocks-120.txt`, `blocks-60.txt` | The You, Thinking, Answer and Warning blocks plus the run footer, without color, at 120 and 60 columns. |
| `blocks-120.ansi`, `blocks-60.ansi` | The same blocks with color. |

**Why:** any unintended change to the layout shows up as a test failure.

**How it fits:** compared by the interface tests in [`tests/`](../README.md).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
