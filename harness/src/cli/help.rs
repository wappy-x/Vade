pub(crate) const HELP: &str = "Credentials: --api-key-file PATH (or VADE_API_KEY_FILE); sends Bearer to upstream.\nVade Harness — first-party Rust coding harness

Usage:
  vade-harness [demo] [OPTIONS]
  vade-harness demo-compaction
  vade-harness direct --endpoint URL --model NAME [OPTIONS]
  vade-harness agent --endpoint URL --model NAME --workspace PATH [OPTIONS]
  vade-harness gateway --upstream URL [--upstream URL ...] [--bind ADDRESS]
  vade-harness vade-gateway --upstream URL [--upstream URL ...] [OPTIONS]
  vade-harness codex --workspace PATH --permission POLICY [OPTIONS]
  vade-harness workload --endpoint URL --model NAME --json FILE [OPTIONS]
  vade-harness setup CLIENT --endpoint URL --model NAME [OPTIONS]
  vade-harness doctor URL [--model NAME] [--max-seconds N]
  vade-harness hermes --workspace PATH --permission deny-callbacks [OPTIONS]

Modes:
  demo                  Local scripted response; no network (default)
  demo-compaction       Offline frozen-prefix hash/byte audit and saved-prefill estimate
  direct                Text chat over HTTP; manual local tools, no native agent loop
  agent                 Native coding tools, approvals and private session journals
  gateway               Run the core HTTP forwarding gateway
  vade-gateway          Run the standalone vade-gateway relay; see vade-gateway --help
  codex                 Delegate exact prompts to optional installed Codex app-server
  workload              Concurrent native sessions; see workload --help
  setup                 Print offline client configuration; see setup --help
  doctor                Probe capabilities, admission and two-turn cache reuse
  hermes                Delegate exact prompts to optional Hermes ACP v1 agent

Options:
  --mode MODE           Select a mode; --demo is an alias for demo
  --endpoint URL        Direct/agent upstream, plain http:// only
  --model NAME          Required for direct/agent; optional Codex model override
  --prompt TEXT         One prompt and exit; interactive approvals are denied
                        Agent exits 3 after a normal run if its check failed or was not run
  --workspace PATH      Tool root / delegated cwd; required for agent/codex/hermes
  --session NAME        Agent: new private transcript journal; other modes: counter
  --vade-home DIR       Private state root (default VADE_HOME, else ~/.vade); journals
                        and input recall go to DIR/projects/NAME-HASH, outside the project
  --state-in-project    Keep journals and recall in WORKSPACE/.vade (git-ignored) instead
  --verbose             Show run diagnostics after each answer
  --resume NAME         Agent: resume a past journal without replaying tools
  --approval POLICY     Agent: ask (default), auto, never
                        Cancel may not stop programs that daemonize; they stay sandboxed
  --context-tokens N    Agent: override server context budget; default from capabilities
  --prewarm-compaction  Agent: opt-in one-token cache warm after compaction; reply discarded
  --reply-tokens N      Agent: estimated prose budget; 0 disables local budgeting
                        Default: measured decode × min(120s, remaining task time / 8)
                        Unknown speed: 2048 tokens / 120s; default ceiling 4096
                        Oversized unfinished tool proposals are discarded whole
  --max-output-tokens N Direct/agent: 1 through the advertised Engine output limit
  --no-auto-continue    Direct/agent: wait for /continue after an output cutoff
  --greedy              Temperature 0 on every turn; enables exact greedy drafting
  --temperature T       Explicit sampling temperature (0..2)
  --seed N              Pinned seed; preserved unchanged on retry and resume
  --thinking on|off     Direct/agent: override VADE_HARNESS_THINKING=1 (default off)
  --effort LEVEL        Direct/agent: low|medium|high|xhigh, turns thinking on (VADE_HARNESS_EFFORT)
  --check COMMAND       Agent: run /bin/sh -c COMMAND in the sandbox after edits (or .vade/check)
                        Requires macOS; noninteractive runs require --approval auto
  --no-check            Agent: ignore the workspace .vade/check file
  Environment          VADE_HARNESS_PREFIX_CACHE=0 disables negotiated shared-prefix caching
  --rules-scope PATH    Agent: include AGENTS.md ancestor chain (repeat ≤8); root always included
  --shell-exclude NAME  Agent: extra inaccessible unscanned directory name (repeat ≤8)
  --shell-toolchain DIR Agent: owner-selected Rust distribution; bin/lib read-only
  --plain               Line input; no raw terminal/color
  --bind ADDRESS        Gateway bind, default 127.0.0.1:8787
  --children            Agent only: enable bounded children (read-only by default)
  --child-edit NAME=PATH Host scope for exact file edits (repeat for each leased file)
  --child-shell NAME     Host scope for shell; exclusive workspace writes, approvals apply
  --upstream URL        HTTP upstream; repeat for cache-aware routing
  --permission POLICY   Codex: read-only/workspace-write; Hermes: deny-callbacks; required
  --hermes-bin PATH     Hermes executable (default hermes on PATH); fixed acp argument
  --external-session ID Resume/load a Hermes-owned session if negotiated
  --codex-bin PATH      Codex executable (default codex on PATH)
  --thread ID           Resume a Codex-owned thread
  -h, --help            Help; --version prints version

Agent writes private journals under VADE_HOME/projects/NAME-HASH/sessions by default
(~/.vade without VADE_HOME), never inside the workspace. --session names a
new journal; root AGENTS.md and explicit --rules-scope ancestor chains are frozen
with a small skills/catalog.json. Skill bodies load lazily; no scripts autorun.
--resume loads the frozen rules and the same model/tool definitions. /new starts
a new journal and preserves the old one. No automatic retry or tool replay.
Conversation commands:
  /help                 Common commands, grouped with one-line descriptions
  /help all             Every command available in this mode
  /help COMMAND         Usage and details for one command
  /copy [code|N]        Exact last answer, last fenced code block, or Nth answer back
                        64 KiB cap; pbcopy locally, OSC 52 over SSH (tmux passthrough)
  Type /                Common commands; keep typing to search all. Arrows select,
                        Enter runs, Tab completes, Esc closes

Agent tools: workspace_read/list, workspace_search, workspace_read_lines, workspace_edit_lines, apply_patch, workspace_shell (macOS only), tool_output_read,
skill_read, workspace_manifest, workspace_symbols.
Patch previews are shown before approval. ask requires y/n (or /approve, /deny) for
edits and commands outside its read-only allowlist. auto permits validated exact
patches and all commands inside the macOS OS sandbox without asking. It never
authorizes commands outside the sandbox; no unsandboxed fallback exists.
never denies edits and commands. Noninteractive runs deny approval requests.
Commands use explicit argv, cleared environment, macOS sandbox, no network,
120-second maximum timeout, capped output, and process-group cancellation.
On Linux, workspace_shell returns a clear unsupported-platform tool result;
workspace reads and patches remain available. No TLS or permission escalation.

/context shows per-message estimates plus the last request's contract admission
and reported usage. The runtime warns at 80% and stops above the local budget.
/compact KEEP_TURNS SUMMARY replaces older assistant prose, preserving all user
constraints, instructions, tool pairs and recent context. /compact-preview previews
it; /compaction inspects its source and KV impact. All later KV is invalidated.
Original journal records remain on disk; native history is never silently pruned.

Direct/native chat: input during a reply interrupts it, keeps the partial answer marked
[interrupted] in context, and sends the new message after cancellation. Vade Engine
uses /v1/cancel; foreign servers receive a socket close (stop is not guaranteed).
/queue on opts into queueing; /queue off restores steering; /queue TEXT queues once.
Raw terminal: Enter sends; reported Shift+Enter,
Option+Enter or Ctrl+J inserts a newline. Up/Down recall submitted messages;
Left/Right move the cursor. Home/End, Delete, Ctrl+U (clear), Ctrl+K (to line end),
Ctrl+W / Alt+D (delete word), Alt+B/F or Ctrl+Left/Right (word) edit input; Ctrl+L
redraws. Esc or Ctrl+C cancels response display and asks the transport/provider to
cancel; upstream compute may continue. Esc never discards a draft. Queued work waits
for success; failure/cancellation holds it for /next. Ctrl+C on an empty idle prompt exits.
Ctrl+D sends nonempty input or exits. Bracketed paste stays one draft until Enter.
Ctrl+T expands/collapses thinking; replies remain visible. PgUp/PgDn scroll output.
Ctrl+O opens conversation review: PgUp/PgDn or Space/b page; arrows/wheel scroll;
g/G first/last; / searches, n/N next/previous; { } jump prompts; Ctrl+T all thinking;
q or Esc returns to the draft. Review uses the alternate screen with native selection.
The terminal owns mouse selection and Cmd+C; mouse reporting stays off.
Runs lasting at least 30 seconds ring the bell and send an OSC 9 desktop notice
and OSC 2 title when they end or wait for approval (terminal support required).
Line mode uses /paste for multiline entry; Ctrl+C exits the process.


Direct/demo recall is in memory (32 turns/256 KiB); --session stores only a counter
for those modes. Codex owns its delegated policy and persistence. Inference and KV
state belong to the upstream.

/help in the conversation shows common commands; /help all lists everything.
";
pub(crate) const AGENT_HELP: &str =
    "Credentials: --api-key-file PATH (or VADE_API_KEY_FILE).\nVade native coding agent

Usage: vade-harness agent --endpoint URL --model NAME --workspace PATH [OPTIONS]

  --endpoint URL        Required HTTP upstream; HTTPS is unsupported
  --model NAME          Required literal upstream model ID
  --workspace PATH      Required root for bounded workspace tools
  --prompt TEXT         One prompt and exit; interactive approvals are denied
                        Exits 3 with check failing if --check / .vade/check still fails
  --session NAME        New private transcript journal
  --vade-home DIR       Journals/recall in DIR/projects/NAME-HASH (default VADE_HOME or ~/.vade)
  --state-in-project    Journals/recall in WORKSPACE/.vade, with a .gitignore
  --verbose             Show run diagnostics after each answer
  --resume NAME         Resume a journal without replaying tools
  --approval POLICY     ask (default), auto, never
                        Cancel may not stop programs that daemonize; they stay sandboxed
  --context-tokens N    Override server context budget (fallback 131072)
  --agent-time-limit-seconds N Active work per request (default 1200; 0 disables)
  --config PATH         Native JSON config; time limit and auto_continue settings
  --max-output-tokens N Output reservation; positive, within upstream limit
  --no-auto-continue    Wait for /continue after an output cutoff
  --greedy              Temperature 0 on every turn; enables exact greedy drafting
  --temperature T       Explicit sampling temperature (0..2)
  --seed N              Pinned seed; preserved unchanged on retry and resume
  --thinking on|off     Override VADE_HARNESS_THINKING=1 (default off)
  --effort LEVEL        low|medium|high|xhigh; turns thinking on (VADE_HARNESS_EFFORT).
                        Without it, thinking on uses the Engine default (xhigh for Qwen3.8)
  --check COMMAND       After any round that changed files, run /bin/sh -c COMMAND
                        in the same sandbox and send the bounded result to the model.
                        Default: first line of WORKSPACE/.vade/check, if present
                        Requires macOS; noninteractive runs require --approval auto
  --no-check            Ignore .vade/check
  --rules-scope PATH    Include AGENTS.md ancestor chain (repeat up to 8)
  --shell-exclude NAME  Block another directory name (repeat up to 8)
  --shell-toolchain DIR Owner-selected Rust distribution; bin/lib read-only
  --children            Enable bounded children; default scopes are read-only
  --child-edit NAME=PATH Grant an exact child file scope (repeatable)
  --child-shell NAME    Grant an exclusive child shell scope (repeatable)
  --plain               Line input without raw terminal or color
  -h, --help            Show this help
  --version             Print version

Without sampling flags, new sessions leave sampling unspecified (greedy on Vade Engine).
Environment: VADE_HARNESS_PREFIX_CACHE=0 disables negotiated shared-prefix caching.
Commands: /context, /status, /copy [code|N], /approve, /deny, /children, /resume, /compact, /quit.
Copy preserves exact text (64 KiB cap): pbcopy locally, OSC 52 over SSH and tmux.
Runs of at least 30 seconds ring and send a desktop notice/title on end or approval.
/help shows common commands; /help all lists everything.
Shell commands require the macOS sandbox; Linux supports workspace reads and patches.
Each request allows 60 tool rounds, 61 model turns and 240 proposed tool calls.
The default 20-minute work limit pauses while waiting for your approval.
Cut-off answers continue automatically, up to 32 times within those same limits.
Default output reservation uses at most 16,384 tokens, the advertised limit, or
remaining context minus 256 safety tokens, whichever is smaller. Truncated tool calls stop without running.
Native JSON config: \"auto_continue\": false also disables continuation.
A reply with no visible text and no tool call gets up to 2 reminders, then the
run stops with a notice. A final answer while the check fails gets 2 reminders.
";

pub(crate) const DIRECT_HELP: &str =
    "Credentials: --api-key-file PATH (or VADE_API_KEY_FILE).\nVade direct text chat

Usage: vade-harness direct --endpoint URL --model NAME [OPTIONS]

  --endpoint URL        Required HTTP upstream; HTTPS is unsupported
  --model NAME          Required literal upstream model ID
  --prompt TEXT         One prompt and exit
  --workspace PATH      Root for manual local read/list commands (default .)
  --session NAME        Optional completed-turn counter
  --vade-home DIR       Counter/recall in DIR/projects/NAME-HASH (default VADE_HOME or ~/.vade)
  --state-in-project    Counter/recall in WORKSPACE/.vade, with a .gitignore
  --max-output-tokens N Output reservation; positive, within upstream limit
  --no-auto-continue    Wait for /continue after an output cutoff
  --greedy              Temperature 0 on every turn; enables exact greedy drafting
  --temperature T       Explicit sampling temperature (0..2)
  --seed N              Pinned seed; preserved unchanged on retry and resume
  --thinking on|off     Override VADE_HARNESS_THINKING=1 (default off)
  --effort LEVEL        low|medium|high|xhigh; turns thinking on (VADE_HARNESS_EFFORT)
  --plain               Line input without raw terminal or color
  -h, --help            Show this help
  --version             Print version

Without sampling flags, new sessions leave sampling unspecified (greedy on Vade Engine).
Environment: VADE_HARNESS_PREFIX_CACHE=0 disables negotiated shared-prefix caching.
Commands: /context, /status, /read, /ls, /new, /undo, /quit; /help lists commands.
Direct chat keeps text history in memory and does not execute model tool calls.
Cut-off answers continue automatically, up to 32 times within 61 model turns and
20 minutes of active work per request. History limits can stop them sooner.
Default output reservation uses at most 16,384 tokens, the advertised limit, or
remaining context minus 256 safety tokens, whichever is smaller. An unfit continuation stops plainly.
";

pub(crate) const GATEWAY_HELP: &str =
    "Credentials: --api-key-file PATH (or VADE_API_KEY_FILE).\nVade HTTP gateway

Usage: vade-harness gateway --upstream URL [--upstream URL ...] [OPTIONS]

  --upstream URL        Required HTTP upstream; repeat for cluster routing
  --bind ADDRESS        Listener address (default 127.0.0.1:8787)
  --allow-origin ORIGIN Admit this exact browser Origin, e.g. https://app.example (repeatable)
  --allow-host HOST     Admit another Host name, with or without the port (repeatable)
  --token TOKEN         Require Authorization: Bearer TOKEN (or VADE_GATEWAY_TOKEN)
  -h, --help            Show this help
  --version             Print version

Routes: POST /v1/completions /completions /v1/chat/completions /chat/completions
  /v1/responses /responses /v1/messages /v1/messages/count_tokens /v1/admission /v1/cancel;
  GET /v1/capabilities. Other paths receive 404 before any upstream contact,
except that cluster mode (multiple upstreams) also exposes GET /_vade/gateway/stats.
One upstream uses the plain relay. Explicit caller fields retain their bytes.
Generation defaults: temperature=0 (greedy), seed=20260905, top_p=1.0, top_k=0,
repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, only when
none of temperature, top_p, top_k, min_p, typical_p, seed, frequency_penalty,
presence_penalty, repetition_penalty, logit_bias, n, best_of, sampling,
generation_config is present (including null). Any one suppresses all defaults;
the body is forwarded byte for byte and engine defaults apply.
Routing, affinity and input limits use original client bytes, before defaults.
GET /v1/capabilities advertises X-Vade-Sampling-Defaults, its condition in
X-Vade-Sampling-Defaults-When, and recognized fields in X-Vade-Sampling-Fields.
HTTPS is unsupported; the default listener is loopback. Ctrl+C stops the gateway.
Browser guard: Host must be 127.0.0.1, localhost or [::1] (with the listener port)
or an --allow-host value, else 421. Any request with an Origin header gets 403
unless --allow-origin names it; allowed origins get CORS headers. With --token,
a missing or wrong Authorization: Bearer gets 401 and the header stays local.
";

use super::{args::Mode, commands::COMMANDS};

pub(crate) struct Command {
    pub name: &'static str,
    pub args: &'static str,
    pub summary: &'static str,
    pub detail: &'static str,
    pub group: &'static str,
    pub mode: &'static str,
    pub common: bool,
}
impl Command {
    pub const fn new(
        name: &'static str,
        args: &'static str,
        summary: &'static str,
        detail: &'static str,
        group: &'static str,
        mode: &'static str,
        common: bool,
    ) -> Self {
        Self {
            name,
            args,
            summary,
            detail,
            group,
            mode,
            common,
        }
    }
    pub fn available(&self, mode: Mode) -> bool {
        match self.mode {
            "agent" => mode == Mode::Agent,
            "codex" => mode == Mode::Codex,
            "chat" => matches!(mode, Mode::Demo | Mode::Direct),
            "answers" => matches!(mode, Mode::Direct | Mode::Agent),
            _ => true,
        }
    }
    fn usage(&self) -> String {
        format!(
            "/{}{}{}",
            self.name,
            if self.args.is_empty() { "" } else { " " },
            self.args
        )
    }
}

pub(crate) fn lookup(name: &str) -> Option<&'static Command> {
    let name = name.trim_start_matches('/');
    COMMANDS.iter().find(|c| c.name == name)
}

pub(crate) fn command_help(query: &str, mode: Mode) -> String {
    if !query.is_empty() && query != "all" {
        return match lookup(query) {
            Some(c) => format!(
                "{}\n{}\n{}{}",
                c.usage(),
                c.summary,
                c.detail,
                if c.available(mode) {
                    ""
                } else {
                    "\nAvailable in a different mode."
                }
            ),
            None => unknown_command(query, mode),
        };
    }
    let mut text = String::new();
    let mut group = "";
    for c in COMMANDS.iter().filter(|c| {
        c.available(mode)
            && (query == "all"
                || (c.common && (mode != Mode::Agent || !matches!(c.name, "ls" | "read"))))
    }) {
        if group != c.group {
            group = c.group;
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(group);
            text.push('\n');
        }
        text.push_str(&format!("  /{:<12} {}\n", c.name, c.summary));
    }
    text.push_str("\n/help all · /help COMMAND for detail\n↑/↓ select · Enter run · Tab complete · Esc close · PgUp/PgDn scroll\nEnter send · Ctrl+J newline · Ctrl+C cancel · Ctrl+O review · //text sends a slash\n@path or drop a workspace text file · Tab completes @path; Tab again lists matches · 64 KiB/message");
    text
}

pub(crate) fn unknown_command(name: &str, mode: Mode) -> String {
    if let Some(c) = lookup(name) {
        return if c.available(mode) {
            format!("Usage: {}. /help {} for detail.", c.usage(), c.name)
        } else {
            format!(
                "/{} is available in {} mode. /help lists commands for this mode.",
                c.name, c.mode
            )
        };
    }
    let name = name.trim_start_matches('/');
    let closest = COMMANDS
        .iter()
        .filter(|c| c.available(mode))
        .map(|c| (distance(name, c.name), c))
        .min_by_key(|(score, _)| *score);
    match closest {
        Some((score, c)) if score <= 2 || (name.len() > 6 && score <= 3) => {
            format!("Unknown command /{name}. Did you mean /{}?", c.name)
        }
        _ => format!("Unknown command /{name}. /help lists commands; // sends a leading slash."),
    }
}

fn distance(a: &str, b: &str) -> usize {
    // Command names are short ASCII; cap untrusted input before allocating.
    if a.len() > 64 {
        return usize::MAX;
    }
    let b: Vec<_> = b.chars().collect();
    let mut row: Vec<_> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let old = row[j + 1];
            row[j + 1] = (row[j] + 1)
                .min(old + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = old;
        }
    }
    row[b.len()]
}

#[derive(Default)]
pub(crate) struct SlashMenu {
    prefix: String,
    dismissed: bool,
    selected: usize,
    navigated: bool,
}
impl SlashMenu {
    pub fn refresh(&mut self, input: &str, cursor: usize) {
        let prefix = if input.starts_with('/')
            && !input.starts_with("//")
            && !input.contains(char::is_whitespace)
            && cursor == input.len()
        {
            input
        } else {
            ""
        };
        if self.prefix != prefix {
            self.prefix = prefix.to_owned();
            self.dismissed = false;
            self.selected = 0;
            self.navigated = false;
        }
    }
    fn matches(&self, mode: Mode, approval: bool) -> Vec<&'static Command> {
        if self.prefix.is_empty() || self.dismissed {
            return Vec::new();
        }
        // A bare slash shows the short common list; typing searches everything.
        let bare = self.prefix == "/";
        COMMANDS
            .iter()
            .filter(|c| {
                c.available(mode)
                    && (!bare || c.common)
                    && c.name.starts_with(&self.prefix[1..])
                    && (!approval
                        || matches!(
                            c.name,
                            "approve" | "deny" | "cancel" | "status" | "quit" | "help"
                        ))
            })
            .collect()
    }
    pub fn open(&self, mode: Mode, approval: bool) -> bool {
        !self.matches(mode, approval).is_empty()
    }
    /// The typed text already names a command exactly (e.g. `/children`).
    pub fn exact(&self, mode: Mode, approval: bool) -> bool {
        self.matches(mode, approval)
            .iter()
            .any(|c| c.name == &self.prefix[1..])
    }
    pub fn close(&mut self) {
        self.dismissed = true;
    }
    pub fn step(&mut self, down: bool, mode: Mode, approval: bool) {
        let count = self.matches(mode, approval).len();
        if count > 0 {
            self.selected =
                (self.selected.min(count - 1) + if down { 1 } else { count - 1 }) % count;
            self.navigated = true;
        }
    }
    /// The user moved the selection with the arrow keys for this prefix.
    pub fn navigated(&self) -> bool {
        self.navigated
    }
    /// Exactly one command matches the typed prefix.
    pub fn unique(&self, mode: Mode, approval: bool) -> bool {
        self.matches(mode, approval).len() == 1
    }
    pub fn complete(&mut self, mode: Mode, approval: bool) -> Option<String> {
        let matches = self.matches(mode, approval);
        let c = matches.get(self.selected.min(matches.len().saturating_sub(1)))?;
        self.dismissed = true;
        let text = format!("/{}{}", c.name, if c.args.is_empty() { "" } else { " " });
        self.prefix = text.clone();
        Some(text)
    }
    pub fn panel(
        &self,
        mode: Mode,
        approval: bool,
        rows: usize,
        width: usize,
        color: bool,
    ) -> String {
        let matches = self.matches(mode, approval);
        if matches.is_empty() || rows < 2 {
            return String::new();
        }
        let entries: Vec<_> = matches.iter().map(|c| (c.name, c.summary)).collect();
        vade_harness::terminal::render::menu(
            &entries,
            self.selected,
            width.saturating_sub(1),
            rows,
            color,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gateway_help_states_the_complete_shared_sampling_policy() {
        use vade_harness::{gateway, json};
        let flat = GATEWAY_HELP.split_whitespace().collect::<Vec<_>>().join(" ");
        let json::Value::Object(defaults) = json::parse(gateway::SAMPLING_DEFAULTS_JSON).unwrap() else {
            panic!("shared sampling defaults must be one JSON object");
        };
        assert_eq!(defaults.len(), 7);
        let stated = flat.split("Generation defaults: ").nth(1).unwrap().split(", only when").next().unwrap();
        let stated: Vec<String> = stated.split(", ").map(|s| s.trim_end_matches(" (greedy)").to_owned()).collect();
        let expected: Vec<String> = defaults
            .iter()
            .map(|(name, value)| format!("{name}={}", json::stringify(value).unwrap()))
            .collect();
        assert_eq!(stated, expected, "help must state every shared default with its exact value");
        let fields = flat.split("only when none of ").nth(1).unwrap().split(" is present").next().unwrap();
        let fields: Vec<&str> = fields.split(", ").collect();
        assert_eq!(fields, gateway::SAMPLING_FIELDS.split(',').collect::<Vec<_>>());
        // The public guides restate the same policy; keep them complete too.
        for (name, doc) in [
            ("README.md", include_str!("../../README.md")),
            ("docs/invisible-gateway.md", include_str!("../../docs/invisible-gateway.md")),
        ] {
            let flat = doc.split_whitespace().collect::<Vec<_>>().join(" ");
            for (field, value) in &defaults {
                let stated = format!("`{field}: {}`", json::stringify(value).unwrap());
                assert!(flat.contains(&stated), "{name} omits default {stated}");
            }
            for field in gateway::SAMPLING_FIELDS.split(',') {
                assert!(flat.contains(&format!("`{field}`")), "{name} omits sampling field {field}");
            }
            assert!(flat.contains("including null"), "{name} omits the null-presence rule");
        }
    }
    #[test]
    fn gateway_help_lists_exactly_the_routes_the_core_relay_accepts() {
        let section: Vec<&str> = GATEWAY_HELP
            .split("Routes:")
            .nth(1)
            .unwrap()
            .split("Other paths")
            .next()
            .unwrap()
            .split(|c: char| c.is_whitespace() || c == ';' || c == '.')
            .filter(|token| token.starts_with('/'))
            .collect();
        for route in vade_harness::gateway::ROUTES {
            assert!(section.contains(route), "help omits accepted route {route}");
        }
        for token in &section {
            assert!(vade_harness::gateway::ROUTES.contains(token), "help lists unaccepted route {token}");
        }
        assert_eq!(section.len(), vade_harness::gateway::ROUTES.len());
    }
    #[test]
    fn registry_drives_help_matching_and_mode_scope() {
        for mode in [
            Mode::Demo,
            Mode::Direct,
            Mode::Agent,
            Mode::Codex,
            Mode::Hermes,
        ] {
            let help = command_help("", mode);
            let count = help.lines().filter(|l| l.starts_with("  /")).count();
            assert!((8..=10).contains(&count), "{mode:?}: {count}");
        }
        assert!(unknown_command("/statsu", Mode::Demo).contains("Did you mean /status?"));
        assert!(command_help("read", Mode::Demo).contains("/read PATH"));
        assert!(command_help("all", Mode::Agent).contains("/compact-preview"));
    }
    #[test]
    fn menu_selection_completion_and_dismissal_survive_redraw() {
        let mut menu = SlashMenu::default();
        menu.refresh("/", 1);
        assert!(menu.open(Mode::Demo, false));
        menu.step(true, Mode::Demo, false);
        assert_eq!(menu.complete(Mode::Demo, false).as_deref(), Some("/status"));
        menu.refresh("/status", 7);
        assert!(!menu.open(Mode::Demo, false));
        menu.refresh("/re", 3);
        assert_eq!(menu.complete(Mode::Demo, false).as_deref(), Some("/read "));
        menu.refresh("//", 2);
        assert!(!menu.open(Mode::Agent, false));
        menu.refresh("/", 1);
        for _ in 0..12 {
            menu.step(true, Mode::Agent, false);
        }
        assert_eq!(
            menu.panel(Mode::Agent, false, 10, 80, true).lines().count(),
            10
        );
        // A bare slash lists only common commands; a typed letter searches all.
        let common = COMMANDS
            .iter()
            .filter(|c| c.available(Mode::Agent) && c.common)
            .count();
        assert_eq!(menu.matches(Mode::Agent, false).len(), common);
        assert!(common < 12);
        menu.refresh("/j", 2);
        assert!(!menu.exact(Mode::Agent, false));
        assert_eq!(
            menu.complete(Mode::Agent, false).as_deref(),
            Some("/journal ")
        );
        menu.refresh("/children", 9);
        assert!(menu.exact(Mode::Agent, false));
        // "/c" is ambiguous: Enter completes rather than runs until the user
        // picks an entry with the arrows or the prefix becomes unique.
        menu.refresh("/c", 2);
        assert!(!menu.unique(Mode::Agent, false) && !menu.navigated());
        menu.step(true, Mode::Agent, false);
        assert!(menu.navigated());
        menu.refresh("/co", 3);
        assert!(!menu.navigated());
        menu.refresh("/que", 4);
        assert!(!menu.unique(Mode::Agent, false));
        menu.refresh("/quit", 5);
        assert!(menu.unique(Mode::Agent, false) && menu.exact(Mode::Agent, false));
        menu.refresh("/", 1);
        menu.close();
        assert!(menu.panel(Mode::Agent, false, 10, 80, true).is_empty());
    }
}
