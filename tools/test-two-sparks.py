#!/usr/bin/env python3
"""examples/two-sparks.sh regressions: the real script, with vade and curl stubbed.

The checkout and temporary folders sit under paths with sed replacement syntax
(& | \\) and spaces. No Spark, SSH, network or Rust build is needed; these check
the script's config copy and exit statuses only, not a working installation.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
# Each stub appends its argv to CALLS; the call matching the STUB_FAIL pattern exits 42.
STUB = r"""#!/bin/sh
name=$(basename "$0")
printf '%s %s\n' "$name" "$*" >> "$CALLS"
case "$name $*" in
  "vade up --sparks 2 --dry-run") cp "$VADE_CONFIG" "$SEEN"; printf '%s\n' "$VADE_HOME" > "$SEEN.home"
    printf 'Dry run.\nLaunch plan: stub\n  Endpoint          http://localhost:18181/v1\nStart planned.\n' ;;
esac
case "$name $*" in $STUB_FAIL) exit 42 ;; esac
exit 0
"""


class TwoSparksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="two sparks ")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).resolve()
        self.checkout = base / "R&D|x\\y $HOME vade"
        (self.checkout / "examples").mkdir(parents=True)
        for name in ("two-sparks.sh", "two-sparks.conf"):
            shutil.copyfile(ROOT / "examples" / name, self.checkout / "examples" / name)
        self.tmp = base / "tmp &|\\"
        self.tmp.mkdir()
        tools = base / "bin"
        tools.mkdir()
        for name in ("vade", "curl"):
            (tools / name).write_text(STUB)
            (tools / name).chmod(0o755)
        # macOS mktemp ignores TMPDIR without a template, so the stub passes one.
        (tools / "mktemp").write_text('#!/bin/sh\nexec "%s" -d "$TMPDIR/tmp.XXXXXX"\n' % shutil.which("mktemp"))
        (tools / "mktemp").chmod(0o755)
        self.calls = base / "calls"
        self.seen = base / "seen.conf"
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("VADE_")}
        self.env.update(PATH=str(tools) + os.pathsep + os.environ["PATH"], TMPDIR=str(self.tmp),
                        CALLS=str(self.calls), SEEN=str(self.seen), STUB_FAIL="", VADE_API_URL="http://stale:1/v1")

    def run_script(self, *args, fail="", **env):
        result = subprocess.run(["sh", str(self.checkout / "examples/two-sparks.sh"), *args],
                                env=dict(self.env, STUB_FAIL=fail, **env), capture_output=True, text=True, timeout=30)
        calls = self.calls.read_text().splitlines() if self.calls.exists() else []
        return result, calls

    def test_dry_run_copies_paths_literally(self):
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["vade up --sparks 2 --dry-run"])
        self.assertIn("  Endpoint          http://localhost:18181/v1", result.stdout)
        config = self.seen.read_text().splitlines()
        for key, folder in (("engine_source", "engine"), ("harness_source", "harness"),
                            ("cli_source", "cli"), ("stage_dir", ".local/stage")):
            self.assertIn(key + " = " + str(self.checkout / folder), config)
        state = [line for line in config if line.startswith("state_dir = ") and line.endswith("/state")]
        self.assertEqual(len(state), 1, config)
        folder = Path(state[0][len("state_dir = "):]).parent
        self.assertEqual(folder.parent, self.tmp)
        self.assertEqual(self.seen.with_name("seen.conf.home").read_text(), str(folder / "home") + "\n")
        self.assertFalse(any("/path/to/vade" in line for line in config if not line.startswith("#")))

    def test_failed_preview_keeps_its_status(self):
        result, calls = self.run_script(fail="vade up --sparks 2 --dry-run")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertEqual(calls, ["vade up --sparks 2 --dry-run"])
        result, calls = self.run_script("--live", fail="vade up --sparks 2 --dry-run", VADE_CONFIG="/my/two.conf")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)

    def test_live_flow_checks_the_planned_endpoint_and_stops(self):
        result, calls = self.run_script("--live", VADE_CONFIG="/my/two.conf")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call.split(" --receipt ")[0] for call in calls],
                         ["vade up --sparks 2 --dry-run", "vade install", "vade up --sparks 2", "vade status",
                          "curl --fail-with-body -sS http://localhost:18181/v1/models", "vade verify", "vade down"])
        receipt = Path(calls[5].split(" --receipt ", 1)[1])
        self.assertEqual((receipt.name, receipt.parent.parent), ("vade-verify.json", self.tmp))

    def test_live_failures_stay_nonzero_and_stop_after_startup(self):
        for fail, stops in (("vade install", False), ("vade up --sparks 2", True), ("vade status", True),
                            ("curl *", True), ("vade verify *", True)):
            with self.subTest(fail=fail):
                self.calls.unlink(missing_ok=True)
                result, calls = self.run_script("--live", fail=fail, VADE_CONFIG="/my/two.conf")
                self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
                self.assertEqual(calls[-1] == "vade down", stops, calls)


if __name__ == "__main__":
    unittest.main()
