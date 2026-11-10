//! Command classification for `workspace_shell`: safe, needs approval or
//! blocked. argv and `sh -c` scripts are parsed into simple commands (pipes,
//! lists, subshells, substitutions, redirections and environment prefixes);
//! nothing matches a pattern against the whole command string. Every class
//! still runs inside the OS sandbox; the class only decides who must agree.
use std::collections::HashSet;
use std::path::Path;

/// How much agreement a command needs. Order matters: the strictest wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// Read-only, build or test work: runs without a prompt.
    Safe,
    /// Needs a yes from a person, an allow-list rule or `--yes`.
    Ask,
    /// Never runs, also not with `--yes`.
    Blocked,
}
impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Ask => "needs approval",
            Self::Blocked => "blocked",
        }
    }
}

/// One reason a command is not plain safe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub class: Class,
    pub reason: String,
    /// The simple command as written (program, arguments, redirections), used
    /// to match allow-list rules. Empty for script-level findings.
    pub words: Vec<String>,
    /// False when no allow-list rule may approve it (root, credentials, a
    /// script piped into a shell, code built from text). `--yes` still can.
    pub listable: bool,
}

/// The classification of one complete command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub class: Class,
    pub findings: Vec<Finding>,
}
impl Verdict {
    /// Distinct reasons in first-seen order, joined for prompts and errors.
    pub fn reasons(&self) -> String {
        let mut seen: Vec<&str> = Vec::new();
        for finding in self.findings.iter().filter(|f| f.class == self.class) {
            if !seen.contains(&finding.reason.as_str()) {
                seen.push(&finding.reason);
            }
        }
        seen.join("; ")
    }
    /// Rules that would approve this command next time, or None when a
    /// finding cannot go on an allow list. Tools with subcommands get a
    /// prefix rule (`git push *`); other programs get their exact words.
    pub fn suggested_rules(&self) -> Option<Vec<Vec<String>>> {
        if self.class != Class::Ask {
            return None;
        }
        let mut rules: Vec<Vec<String>> = Vec::new();
        for finding in self.findings.iter().filter(|f| f.class == Class::Ask) {
            if !finding.listable || finding.words.is_empty() {
                return None;
            }
            let words = &finding.words;
            // A rule must load again and mean the same words: no empty word,
            // no substitution or variable text, no leading `*`.
            if words.len() > 128 || words[0] == "*" || words.iter().any(|w| w.is_empty() || w.len() > 4096 || w.contains(['\0', '$', '`'])) {
                return None;
            }
            let rule = if SUBCOMMAND_TOOLS.contains(&words[0].as_str())
                && words.len() >= 2
                // `cargo +1.99.0 fmt`: a toolchain is not a subcommand.
                && !words[1].starts_with(['-', '+'])
                && words[1] != "*"
                && words[1..].iter().all(|w| !w.contains(['>', '<']))
            {
                vec![words[0].clone(), words[1].clone(), "*".into()]
            } else if words.last().is_some_and(|w| w == "*") {
                // A literal `*` as the last word would read as "any words".
                return None;
            } else {
                words.clone()
            };
            if !rules.contains(&rule) {
                rules.push(rule);
            }
        }
        Some(rules)
    }
}

/// Programs whose first argument names an action. Their rules keep that
/// action and allow any arguments after it: only programs whose options
/// the checker reads from closed lists, so a command with any other option
/// gets no rule and no rule can approve it. Other programs (npm, docker,
/// make) get exact rules.
const SUBCOMMAND_TOOLS: &[&str] = &["git", "cargo", "go"];

/// Classify one argv as `workspace_shell` would run it. `root` is the
/// workspace root, used to tell writes inside it from writes outside it.
pub fn classify(argv: &[String], root: &Path) -> Verdict {
    glob::reset_work();
    let mut ctx = Ctx {
        root: root.to_string_lossy().trim_end_matches('/').to_owned(),
        findings: Vec::new(),
        depth: 0,
        work: 0,
        tainted: HashSet::new(),
        arith: Vec::new(),
    };
    let words: Vec<Word> = argv
        .iter()
        .map(|text| Word {
            text: text.clone(),
            ..Word::default()
        })
        .collect();
    let command = Simple {
        display: argv.to_vec(),
        words,
        ..Simple::default()
    };
    ctx.pipeline(&[command]);
    ctx.arith_findings();
    let class = ctx
        .findings
        .iter()
        .map(|f| f.class)
        .max()
        .unwrap_or(Class::Safe);
    Verdict {
        class,
        findings: ctx.findings,
    }
}

/// A word after quote removal. `dynamic` marks text the shell would still
/// expand (variables, substitutions, globs, `~`), so its final value is unknown.
#[derive(Clone, Debug, Default)]
struct Word {
    text: String,
    dynamic: bool,
    /// Variables that arithmetic in this word reads (see `lexer::arith_names`).
    arith: Vec<String>,
    /// Has an unquoted `*`, `?`, `[` or `{`, which can expand to other words.
    glob: bool,
}
#[derive(Clone, Debug, Default)]
struct Simple {
    words: Vec<Word>,
    /// Output redirection targets.
    writes: Vec<Word>,
    /// Input redirection sources.
    reads: Vec<Word>,
    /// Here-document bodies fed to standard input: (index, quoted delimiter).
    heredocs: Vec<(usize, bool)>,
    /// `<<<` here-strings fed to standard input. zsh feeds every input
    /// redirection, not only the last one.
    herestrings: Vec<Word>,
    display: Vec<String>,
}
impl Simple {
    /// A command with words or any redirection. A bare `<<EOF` or `<<< x`
    /// still runs the expansions in its text.
    fn has_parts(&self) -> bool {
        !(self.words.is_empty() && self.writes.is_empty() && self.reads.is_empty() && self.heredocs.is_empty() && self.herestrings.is_empty())
    }
}
enum Tok {
    Word(Word),
    Op(&'static str),
    Redirect(&'static str),
    Heredoc(usize, bool),
}

struct Ctx {
    root: String,
    findings: Vec<Finding>,
    depth: usize,
    /// Commands and scripts checked so far, against `MAX_WORK`.
    work: usize,
    /// Variables that any script in this command sets to text that is not a
    /// plain number. One set for the whole command, whatever the order or
    /// the branch, so a loop or a skipped `if` cannot hide a value.
    tainted: HashSet<String>,
    /// Variables read by arithmetic, with the command words; checked
    /// against `tainted` once the whole command is read.
    arith: Vec<(String, Vec<String>)>,
}
/// Nesting of scripts, substitutions and wrappers before the check refuses.
const MAX_DEPTH: usize = 24;
/// Commands and scripts per classification before the check refuses.
const MAX_WORK: usize = 4096;
/// Findings kept per classification; later ones count only if stricter.
const MAX_FINDINGS: usize = 256;

impl Ctx {
    fn add(&mut self, class: Class, reason: impl Into<String>, words: &[String], listable: bool) {
        let finding = Finding {
            class,
            reason: reason.into(),
            words: words.to_vec(),
            listable,
        };
        if self.findings.len() >= MAX_FINDINGS && self.findings.iter().any(|f| f.class >= finding.class) {
            // A dropped finding may be the part that no rule covers, so no
            // rule may approve the command.
            const DROPPED: &str = "has too many parts to approve with a rule";
            if !self.findings.iter().any(|f| f.reason == DROPPED) {
                self.findings.push(Finding { class: Class::Ask, reason: DROPPED.into(), words: Vec::new(), listable: false });
            }
            return;
        }
        if !self.findings.contains(&finding) {
            self.findings.push(finding);
        }
    }
    /// Count one step into a script or command. False, with a blocking
    /// finding, once the nesting or the total work is too large to check.
    fn enter(&mut self) -> bool {
        self.work += 1;
        if self.depth >= MAX_DEPTH || self.work > MAX_WORK {
            self.script_finding(Class::Blocked, "too deeply nested or too long to check; split it into simpler commands");
            return false;
        }
        self.depth += 1;
        true
    }
    fn script_finding(&mut self, class: Class, reason: &str) {
        self.add(class, reason, &[], false);
    }

    /// Classify shell source text, as given to `sh -c` or fed on stdin.
    fn script(&mut self, text: &str) {
        if !self.enter() {
            return;
        }
        if fork_bomb(text) {
            self.script_finding(Class::Blocked, "fork bomb (a function that starts copies of itself)");
        }
        // Safe needs plain forms only: whatever the shell could read in a way
        // the checker does not, needs approval (see plain.rs).
        if !plain::plain(text) {
            self.script_finding(Class::Ask, "uses shell forms beyond plain words, quotes and operators, which the checker does not read fully");
        }
        let mut lexer = Lexer::new(text);
        lexer.run();
        self.tainted.extend(std::mem::take(&mut lexer.assigned));
        for (class, problem) in std::mem::take(&mut lexer.problems) {
            self.script_finding(class, problem);
        }
        for nested in std::mem::take(&mut lexer.nested) {
            self.script(&nested);
        }
        let bodies = std::mem::take(&mut lexer.bodies);
        self.parse(lexer.tokens, &bodies);
        self.depth -= 1;
    }

    fn parse(&mut self, tokens: Vec<Tok>, bodies: &[Option<String>]) {
        let mut pipeline: Vec<Simple> = Vec::new();
        let mut current = Simple::default();
        let mut tokens = tokens.into_iter().peekable();
        // Words of a `for` header or `case` subject are data, not commands.
        let mut header = false;
        let mut test = false;
        // The words of a `[[ ... ]]` test, checked at its end.
        let mut tested: Vec<Word> = Vec::new();
        // After `time`: its own `-p` and `--` come before the command.
        let mut timed = false;
        // `coproc NAME { ...; }` names a group; `coproc cmd` runs cmd.
        let mut coproc: Option<Word> = None;
        // A `for` or `select` header: the variable, and its values so far.
        let mut looped: Option<(String, Vec<Word>)> = None;
        while let Some(token) = tokens.next() {
            if let Some(name) = coproc.take() {
                if !(matches!(&token, Tok::Word(w) if w.text == "{" && !w.dynamic) || matches!(token, Tok::Op("("))) {
                    current.display.push(name.text.clone());
                    current.words.push(name);
                }
            }
            match token {
                // A reserved word counts only first: after a redirection,
                // `time` is the program.
                Tok::Word(word) if current.words.is_empty() && !header && !test => {
                    let first = current.display.is_empty() && !word.dynamic;
                    match word.text.as_str() {
                        "{" | "}" | "!" | "if" | "then" | "else" | "elif" | "do" | "while"
                        | "until" | "fi" | "done" | "esac" if first => continue,
                        "-p" | "--" if timed && !word.dynamic => continue,
                        "time" if first => timed = true,
                        // zsh `repeat N command`: the count, then the command.
                        // zsh reads the count as arithmetic, so only a plain
                        // number is safe.
                        "repeat" if first && matches!(tokens.peek(), Some(Tok::Word(_))) => {
                            if let Some(Tok::Word(count)) = tokens.next() {
                                self.data_word(&count);
                                if count.dynamic || count.text.is_empty() || !count.text.bytes().all(|b| b.is_ascii_digit()) {
                                    self.script_finding(Class::Ask, "repeat reads a count that the checker cannot read as a plain number");
                                }
                            }
                        }
                        "coproc" if first => {
                            if matches!(tokens.peek(), Some(Tok::Word(w)) if w.text != "{") {
                                if let Some(Tok::Word(name)) = tokens.next() {
                                    coproc = Some(name);
                                }
                            }
                        }
                        "for" | "select" if first => {
                            header = true;
                            looped = Some((String::new(), Vec::new()));
                        }
                        "case" if first => {
                            self.script_finding(Class::Ask, "uses a case statement, which the checker does not read");
                            header = true;
                        }
                        "function" if first => {
                            self.script_finding(Class::Ask, "defines a shell function");
                            // The next word is the name; the body is checked as commands.
                            if matches!(tokens.peek(), Some(Tok::Word(_))) {
                                tokens.next();
                            }
                        }
                        "[[" if first => test = true,
                        _ => {
                            if matches!(tokens.peek(), Some(Tok::Op("("))) {
                                self.script_finding(Class::Ask, "defines a shell function");
                                tokens.next();
                                if matches!(tokens.peek(), Some(Tok::Op(")"))) {
                                    tokens.next();
                                }
                                continue;
                            }
                            timed = false;
                            current.display.push(word.text.clone());
                            current.words.push(word);
                        }
                    }
                }
                Tok::Word(word) => {
                    if test && word.text == "]]" {
                        test = false;
                        self.test_words(&std::mem::take(&mut tested));
                    } else if test {
                        tested.push(word);
                    } else if header && matches!(word.text.as_str(), "do" | "{") && !word.dynamic {
                        // `for name do ...` and zsh `for name in 1 { ...; }`:
                        // the body starts without a separator.
                        header = false;
                        self.loop_values(looped.take());
                    } else if header {
                        self.data_word(&word);
                        match &mut looped {
                            Some((name, _)) if name.is_empty() => *name = word.text,
                            Some((_, values)) => values.push(word),
                            None => {}
                        }
                    } else {
                        current.display.push(word.text.clone());
                        current.words.push(word);
                    }
                }
                Tok::Redirect(_) if test => {}
                Tok::Redirect(op) => {
                    let target = match tokens.peek() {
                        Some(Tok::Word(_)) => match tokens.next() {
                            Some(Tok::Word(word)) => word,
                            _ => unreachable!(),
                        },
                        _ => {
                            self.script_finding(Class::Ask, "has a redirection without a target");
                            continue;
                        }
                    };
                    current.display.push(op.into());
                    current.display.push(target.text.clone());
                    let duplicate = matches!(op, ">&" | "<&")
                        && (target.text == "-" || target.text.bytes().all(|b| b.is_ascii_digit()));
                    if duplicate {
                        continue;
                    }
                    if op == "<<<" {
                        current.herestrings.push(target);
                        continue;
                    }
                    if op == "<" {
                        current.reads.push(target);
                    } else {
                        current.writes.push(target);
                    }
                }
                Tok::Heredoc(index, quoted) => {
                    current.display.push("<<".into());
                    current.heredocs.push((index, quoted));
                }
                Tok::Op(op) => {
                    if current.has_parts() {
                        pipeline.push(std::mem::take(&mut current));
                    } else {
                        current = Simple::default();
                    }
                    // `[[ a && b ]]` keeps its operators; a separator ends the test.
                    if test && !matches!(op, "&&" | "||" | "(" | ")") {
                        test = false;
                        self.script_finding(Class::Ask, "has an unclosed [[ test");
                        self.test_words(&std::mem::take(&mut tested));
                    }
                    timed = false;
                    if op == "|" || op == "|&" {
                        continue;
                    }
                    header = false;
                    self.loop_values(looped.take());
                    let finished = std::mem::take(&mut pipeline);
                    self.pipeline_with(&finished, bodies);
                }
            }
        }
        if test {
            self.script_finding(Class::Ask, "has an unclosed [[ test");
            self.test_words(&tested);
        }
        self.loop_values(looped);
        if let Some(name) = coproc {
            current.display.push(name.text.clone());
            current.words.push(name);
        }
        if current.has_parts() {
            pipeline.push(current);
        }
        self.pipeline_with(&pipeline, bodies);
    }

    fn pipeline(&mut self, commands: &[Simple]) {
        self.pipeline_with(commands, &[]);
    }

    fn pipeline_with(&mut self, commands: &[Simple], bodies: &[Option<String>]) {
        let mut downloads = false;
        for (index, command) in commands.iter().enumerate() {
            let piped_in = index > 0;
            let resolved = self.simple(command, bodies, piped_in);
            if resolved.stdin_script && piped_in {
                let reason = if downloads {
                    "downloads a script and runs it"
                } else {
                    "pipes text into a shell or interpreter as code"
                };
                self.add(Class::Ask, reason, &command.display, false);
            }
            downloads |= resolved.downloads;
        }
    }

    /// Classify one simple command, including its redirections.
    fn simple(&mut self, command: &Simple, bodies: &[Option<String>], piped_in: bool) -> Resolved {
        let display = &command.display;
        for target in &command.writes {
            self.write_target(target, display, "redirects output to");
        }
        for word in command.words.iter().chain(&command.reads).chain(&command.writes).chain(&command.herestrings) {
            self.credentials(word, display);
            self.read_arith(&word.arith, display);
        }
        let mut resolved = Resolved::default();
        // Every here-document and here-string: bash and sh read the last one,
        // zsh reads them all in order.
        let mut stdin_bodies = Vec::new();
        for (index, quoted) in &command.heredocs {
            match bodies.get(*index).cloned().flatten() {
                Some(body) => {
                    if !quoted {
                        // Substitutions in an unquoted here-document run first.
                        let mut lexer = Lexer::expansions(&body, 0, Text::Plain);
                        self.tainted.extend(std::mem::take(&mut lexer.assigned));
                        for (class, problem) in std::mem::take(&mut lexer.problems) {
                            self.script_finding(class, problem);
                        }
                        for nested in std::mem::take(&mut lexer.nested) {
                            self.script(&nested);
                        }
                        self.read_arith(&lexer.arith, display);
                    }
                    stdin_bodies.push(body);
                }
                None => self.add(Class::Ask, "has an unfinished here-document", display, false),
            }
        }
        let mut stdin_dynamic = false;
        for word in &command.herestrings {
            // Text with an expansion asks, and is still checked as written.
            stdin_dynamic |= word.dynamic;
            stdin_bodies.push(format!("{}\n", word.text));
        }
        if command.words.is_empty() {
            return resolved;
        }
        self.command(&command.words, display, &mut resolved);
        if resolved.stdin_script {
            if stdin_dynamic {
                self.add(Class::Ask, "runs shell code built from a variable", display, false);
            }
            for body in &stdin_bodies {
                // A shell reading a here-document runs that text as code.
                if resolved.shell {
                    self.script(body);
                } else {
                    self.add(Class::Ask, "runs an interpreter on a here-document", display, true);
                }
            }
            if !command.reads.is_empty() {
                self.add(Class::Ask, "runs a script file through standard input", display, true);
            }
            let fed = stdin_dynamic || !stdin_bodies.is_empty() || !command.reads.is_empty();
            if !fed && !piped_in {
                self.add(Class::Ask, "starts an interactive shell or interpreter", display, true);
            }
            // A pipe still counts: zsh also feeds it with the redirections.
            resolved.stdin_script &= piped_in;
        }
        resolved
    }

    /// Classify a command's words: environment prefixes, wrappers, then the
    /// program itself.
    /// bash and zsh split a brace list into words (`/bin/{true,sh}` is two),
    /// sh does not: the words are checked both ways. A bracket (`/bin/[r]m`)
    /// is checked as each name it can match.
    fn command(&mut self, words: &[Word], display: &[String], resolved: &mut Resolved) {
        if self.enter() {
            self.command_words(words, display, resolved);
            let lists: Vec<Option<Vec<String>>> = words.iter().map(|w| if w.glob { braces(&w.text) } else { Some(vec![w.text.clone()]) }).collect();
            if lists.iter().any(Option::is_none) {
                self.add(Class::Ask, "has a brace expansion that the checker cannot list", display, false);
            } else if words.iter().zip(&lists).any(|(w, list)| list.as_deref() != Some(&[w.text.clone()])) {
                let split: Vec<Word> = words.iter().zip(lists).flat_map(|(w, list)| list.unwrap_or_default().into_iter().map(move |text| Word { text, ..w.clone() })).collect();
                self.command_words(&split, display, resolved);
            }
            self.depth -= 1;
        }
    }
    fn command_words(&mut self, words: &[Word], display: &[String], resolved: &mut Resolved) {
        let mut at = 0;
        while at < words.len() && assignment(&words[at].text) {
            self.assignment_word(&words[at], display);
            at += 1;
        }
        let Some(first) = words.get(at) else { return };
        let args = &words[at + 1..];
        if first.dynamic || options::expands(first) {
            self.add(Class::Ask, "takes its program name from a variable or pattern", display, true);
            // It can name a program that the checker blocks (`/bin/r? -rf /`):
            // the words get the checks of each such program it can name.
            for name in BLOCKING_PROGRAMS {
                if can_name(&first.text, name) {
                    self.program(name, args, display, resolved);
                }
            }
            // The variable may hold only options or a wrapper; check the rest as a command too.
            if !args.is_empty() {
                self.command(args, display, resolved);
            }
            return;
        }
        let Some(name) = program_name(&first.text) else {
            return self.add(Class::Ask, format!("runs a program by path: {}", first.text), display, true);
        };
        resolved.downloads |= matches!(name.as_str(), "curl" | "wget" | "fetch");
        self.program(&name, args, display, resolved);
    }

    /// A word outside any command (a loop or `case` header, a `repeat`
    /// count): its paths, arithmetic and stored code count as in a command.
    fn data_word(&mut self, word: &Word) {
        let display = [word.text.clone()];
        self.credentials(word, &display);
        self.read_arith(&word.arith, &display);
        self.stored_code(word, &display);
    }

    /// A variable value with literal `$(...)` or backquote text: arithmetic,
    /// `eval` or a later `sh -c "$x"` can run it, so check it as code.
    pub(super) fn stored_code(&mut self, word: &Word, display: &[String]) {
        let text = &word.text;
        let placeholders = text.matches("$(...)").count() + text.matches("$((...))").count();
        let code = text.matches("$(").count() > placeholders || text.matches('`').count() > 2 * text.matches("`...`").count();
        if !code {
            return;
        }
        self.add(Class::Ask, "stores shell code in a variable, which can run later", display, false);
        // The name's index (`a[$(cmd)]=1`) and the value both hold code.
        let (name, value) = text.split_once('=').unwrap_or((text, ""));
        for part in [name, value].into_iter().filter(|part| !part.is_empty()) {
            let lexer = Lexer::expansions(part, 0, Text::Plain);
            self.tainted.extend(lexer.assigned);
            for (class, problem) in lexer.problems {
                self.script_finding(class, problem);
            }
            for nested in lexer.nested {
                self.script(&nested);
            }
        }
    }

    /// `NAME=value` as a prefix, an `env` or `arch -e` argument or an
    /// `export`: a value that changes what runs asks, stored code is
    /// checked, and the value is noted for arithmetic.
    pub(super) fn assignment_word(&mut self, word: &Word, display: &[String]) {
        if risky_assignment(&word.text) {
            let name = variable_name(&word.text);
            self.add(Class::Ask, format!("sets {name}, which can change what runs"), display, true);
        }
        self.stored_code(word, display);
        self.assigns(word);
    }

    /// Record that `word` (`NAME=value`, or a bare name that a builtin
    /// sets) gives a variable a value. Anything but a plain number or
    /// `$((...))` marks the name, so arithmetic on it needs approval.
    pub(super) fn assigns(&mut self, word: &Word) {
        let name = variable_name(&word.text);
        let Some(value) = word.text.split_once('=').map(|(_, value)| value) else {
            self.tainted.insert(name.to_owned());
            return;
        };
        let plain = value.replace("$((...))", "");
        let computed = plain.len() != value.len();
        let digits = plain.trim_start_matches(['-', '+']);
        let number = computed == word.dynamic && (computed || !digits.is_empty()) && digits.bytes().all(|b| b.is_ascii_digit());
        if !number {
            self.tainted.insert(name.to_owned());
        }
    }

    /// The name that `test -v` or `[[ -v ]]` reads: the shell evaluates an
    /// array index in it, which can run code, and an expansion can hold one.
    pub(super) fn indexed_name(&mut self, word: &Word, display: &[String]) {
        if options::expands(word) || word.text.contains('[') {
            self.add(Class::Ask, "names a variable with an array index or from a value, which can run code", display, false);
            self.stored_code(word, display);
        }
    }

    /// Queue the variables that arithmetic reads, for `arith_findings`.
    fn read_arith(&mut self, names: &[String], display: &[String]) {
        for name in names {
            if self.arith.len() < MAX_WORK {
                self.arith.push((name.clone(), display.to_vec()));
            } else {
                self.script_finding(Class::Ask, "does too much arithmetic to check");
            }
        }
    }

    /// Ask for arithmetic on command output, on arguments, or on a variable
    /// that any part of the command sets to text: an array index in that
    /// text runs as a command (`a[$(...)]`).
    fn arith_findings(&mut self) {
        for (name, display) in std::mem::take(&mut self.arith) {
            let reason = if name == "$(" {
                "does arithmetic on command output, which can run code".to_owned()
            } else if !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
                "does arithmetic on a script argument, which can run code".to_owned()
            } else if self.tainted.contains(&name) {
                format!("does arithmetic on {name}, which is set to text that can run code")
            } else {
                continue;
            };
            self.add(Class::Ask, reason, &display, false);
        }
    }

    /// `[[ ... ]]`: the operands of `-eq`, `-lt` and the like are arithmetic,
    /// and `-v` takes a name that may have an array index.
    fn test_words(&mut self, words: &[Word]) {
        for (at, word) in words.iter().enumerate() {
            self.read_arith(&word.arith, &[]);
            if word.dynamic {
                continue;
            }
            if matches!(word.text.as_str(), "-eq" | "-ne" | "-lt" | "-le" | "-gt" | "-ge") {
                for operand in [at.checked_sub(1).and_then(|at| words.get(at)), words.get(at + 1)].into_iter().flatten() {
                    self.read_arith(&lexer::arith_names(&operand.text), &[]);
                    self.stored_code(operand, &[]);
                }
            } else if word.text == "-v" {
                if let Some(name) = words.get(at + 1) {
                    self.indexed_name(name, &[]);
                }
            }
        }
    }

    /// The end of a `for` or `select` header: the variable holds each value
    /// in turn (a `select` holds what the person typed).
    fn loop_values(&mut self, looped: Option<(String, Vec<Word>)>) {
        let Some((name, values)) = looped else { return };
        // zsh takes several names before `in` (`for y x in ...`).
        let split = values.iter().position(|w| w.text == "in" && !w.dynamic).unwrap_or(values.len());
        let names: Vec<String> = std::iter::once(name).chain(values[..split].iter().map(|w| w.text.clone())).collect();
        let values = values.get(split + 1..).unwrap_or_default();
        let numbers = !values.is_empty() && values.iter().all(|w| !w.dynamic && !w.text.is_empty() && w.text.bytes().all(|b| b.is_ascii_digit()));
        for name in names {
            if risky_variable(&name) {
                self.script_finding(Class::Ask, "sets a variable that can change what runs");
            }
            if !numbers {
                self.tainted.insert(name);
            }
        }
    }

    /// Ask for a change to files; name the place when it is outside the workspace.
    fn writes_files(&mut self, targets: &[&Word], display: &[String], reason: &str) {
        let mut outside = false;
        for target in targets {
            if raw_disk(&target.text) {
                self.add(Class::Blocked, format!("writes over a disk device: {}", target.text), display, false);
            }
            outside |= self.outside(&target.text);
        }
        let reason = if outside { format!("{reason} outside the workspace") } else { reason.into() };
        self.add(Class::Ask, reason, display, true);
    }

    fn write_target(&mut self, target: &Word, display: &[String], reason: &str) {
        let text = target.text.as_str();
        if matches!(text, "/dev/null" | "/dev/stdout" | "/dev/stderr" | "/dev/tty") || text.starts_with("/dev/fd/") {
            return;
        }
        if raw_disk(text) {
            return self.add(Class::Blocked, format!("writes over a disk device: {text}"), display, false);
        }
        let place = if self.outside(text) { "outside the workspace" } else { "a file" };
        self.add(Class::Ask, format!("{reason} {place}"), display, true);
    }

    fn credentials(&mut self, word: &Word, display: &[String]) {
        let text = word.text.rsplit('=').next().unwrap_or(&word.text);
        // Each word of a brace expansion (`~/.{ssh,x}/id_rsa`) counts, and
        // names match in any case: macOS file names ignore it.
        let Some(paths) = braces(text) else {
            return self.add(Class::Ask, format!("has a brace expansion that the checker cannot list: {}", word.text), display, false);
        };
        // An option word can end with an attached value (`-tfid_rsa`,
        // `-C.ssh`) on any program path: its first part counts when it ends
        // with a name.
        let option = word.text.starts_with('-');
        let parts = || paths.iter().flat_map(|path| path.split('/').enumerate()).map(|(at, part)| (option && at == 0, part.to_ascii_lowercase()));
        let named = |attached: bool, lower: &str, name: &str| if attached { lower.ends_with(name) } else { lower == name };
        let hit = parts().any(|(attached, lower)| {
            SECRET_DIRS.iter().chain(crate::tools::CREDENTIAL_NAMES).any(|name| named(attached, &lower, name))
                || (lower.len() > 1 && crate::tools::CREDENTIAL_SUFFIXES.iter().any(|s| lower.ends_with(s)))
        });
        if hit {
            self.add(Class::Ask, format!("touches credential files: {}", word.text), display, false);
        }
        if parts().any(|(attached, lower)| named(attached, &lower, super::permissions::FILE_NAME)) {
            self.add(Class::Ask, format!("touches an allow-list file: {}", word.text), display, false);
        }
    }

    /// True when a path is outside the workspace or cannot be known yet.
    fn outside(&self, path: &str) -> bool {
        // Each word of a brace expansion (`{~,x}`) counts, and one that
        // cannot be listed is outside.
        if braces(path).is_none_or(|paths| paths.iter().any(|p| p != path && self.outside(p))) {
            return true;
        }
        // `~`, `$HOME` and any other variable or substitution: not known yet.
        if path.starts_with(['~', '$', '`']) {
            return true;
        }
        if self.root.is_empty() && path.starts_with('/') {
            return false;
        }
        let relative = if let Some(rest) = path.strip_prefix('/') {
            match rest.strip_prefix(self.root.trim_start_matches('/')) {
                Some("") => return false,
                Some(tail) if tail.starts_with('/') && !self.root.is_empty() => tail.trim_start_matches('/'),
                _ => return true,
            }
        } else {
            path
        };
        let mut depth = 0usize;
        for part in relative.split('/') {
            match part {
                "" | "." => {}
                ".." => match depth.checked_sub(1) {
                    Some(d) => depth = d,
                    None => return true,
                },
                _ => depth += 1,
            }
        }
        false
    }
}

#[derive(Default)]
struct Resolved {
    /// The command reads a script from standard input (a shell or interpreter
    /// with no script argument).
    stdin_script: bool,
    shell: bool,
    downloads: bool,
}

/// Folder names that hold keys or tokens, in addition to the workspace tools'
/// credential names and suffixes.
pub(crate) const SECRET_DIRS: &[&str] = &[
    ".ssh", ".gnupg", ".aws", ".azure", ".netrc", ".npmrc", ".pypirc", ".git-credentials",
    ".docker", ".kube", "keychains", ".password-store", ".vault-token", ".gcloud",
];

/// Folders whose recursive removal or permission change is never wanted.
const SYSTEM_DIRS: &[&str] = &[
    "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64", "/opt", "/private", "/proc",
    "/root", "/sbin", "/sys", "/usr", "/var", "/System", "/Library", "/Applications", "/Users",
    "/Volumes", "/private/etc", "/private/var", "/usr/local", "/usr/bin", "/usr/lib",
];

/// True for `/`, the home folder and top-level system folders, in the usual
/// spellings (`/*`, `~/`, `$HOME`, `"${HOME}"/.`), for a pattern that can
/// match one of them (`/u*`, `/[!a]sr`), and for a brace word that cannot
/// be listed.
fn catastrophic(target: &str) -> bool {
    braces(target).is_none_or(|targets| targets.iter().any(|target| catastrophic_one(target)))
}
fn catastrophic_one(target: &str) -> bool {
    // A path under the home folder that climbs to it or above it (`~/..`).
    let home = ["~", "${HOME}", "$HOME"].iter().find_map(|prefix| target.strip_prefix(prefix));
    if let Some(rest) = home.filter(|rest| rest.is_empty() || rest.starts_with('/')) {
        let path = normalize(&format!("/h/home{rest}"));
        let path = path.trim_end_matches('/');
        if path == "/h/home" || !path.starts_with("/h/home/") {
            return true;
        }
    }
    let mut t = normalize(target);
    loop {
        if let Some(s) = t.strip_suffix("/*").or_else(|| t.strip_suffix("/.")) {
            t = s.to_owned();
        } else if t.len() > 1 && t.ends_with('/') {
            t.pop();
        } else {
            break;
        }
    }
    (t.is_empty() && target.starts_with('/'))
        || matches!(t.as_str(), "/" | "~" | "$HOME" | "${HOME}" | "/..")
        || SYSTEM_DIRS.iter().any(|dir| dir.eq_ignore_ascii_case(&t))
        || (glob::pattern(&t) && {
            let glob = glob::Glob::new(&t);
            SYSTEM_DIRS.iter().any(|dir| glob.matches(dir))
        })
}

/// At most this many words from one brace word, and this much text made on
/// the way, before the words count as not listed.
const BRACE_WORDS: usize = 1024;
const BRACE_BYTES: usize = 1 << 20;

/// The words a brace list, a `{a..c}` sequence or a `[abc]` bracket gives,
/// in order (`/{usr,etc}`, `{~,x}`, `/dev/sd[ab]`). A bracket stands for
/// each name it can match; `*`, `?` and a bracket that is not a short plain
/// list stay as written. None when the words cannot all be listed: more
/// than `BRACE_WORDS` words or `BRACE_BYTES` of text, or a `{...}` with
/// `..` that is not a plain sequence. Callers then ask, or count the word
/// as every path it could be.
fn braces(word: &str) -> Option<Vec<String>> {
    let mut done = Vec::new();
    let mut queue = vec![word.to_owned()];
    let mut bytes = 0;
    while let Some(text) = queue.pop() {
        bytes += text.len();
        // Each step scans the text once; the scans count toward the work
        // limit of the whole classification.
        if bytes > BRACE_BYTES || !glob::charge(text.len()) {
            return None;
        }
        let Some((open, close, parts)) = pattern(&text) else {
            done.push(text);
            continue;
        };
        // Last part first: the queue is a stack.
        for part in parts?.into_iter().rev() {
            if done.len() + queue.len() >= BRACE_WORDS {
                return None;
            }
            queue.push(format!("{}{part}{}", &text[..open], &text[close + 1..]));
        }
    }
    Some(done)
}

/// The first brace list, sequence or bracket in `text`: where it opens and
/// closes, and the texts it stands for (None: a form not read). One pass
/// pairs the braces, so the cost is linear in the text.
fn pattern(text: &str) -> Option<(usize, usize, Option<Vec<String>>)> {
    let bytes = text.as_bytes();
    // For each `{`: its `}`, and whether a `,` or `..` is at its top level.
    let mut close = vec![usize::MAX; bytes.len()];
    let mut comma = vec![false; bytes.len()];
    let mut dots = vec![false; bytes.len()];
    let mut open: Vec<usize> = Vec::new();
    for (at, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => open.push(at),
            b'}' => {
                if let Some(start) = open.pop() {
                    close[start] = at;
                }
            }
            b',' => open.last().into_iter().for_each(|&start| comma[start] = true),
            b'.' if bytes.get(at + 1) == Some(&b'.') => open.last().into_iter().for_each(|&start| dots[start] = true),
            _ => {}
        }
    }
    // The next `]`, found again only after the last one is passed.
    let mut bracket_close: Option<usize> = None;
    for (at, &b) in bytes.iter().enumerate() {
        match b {
            // `${...}` is a parameter, not a brace list.
            b'{' if close[at] != usize::MAX && (at == 0 || bytes[at - 1] != b'$') => {
                let inner = &text[at + 1..close[at]];
                if comma[at] {
                    return Some((at, close[at], list(inner)));
                }
                if dots[at] {
                    return Some((at, close[at], sequence(inner)));
                }
            }
            b'[' => {
                if bracket_close.is_none_or(|end| end != usize::MAX && end <= at) {
                    bracket_close = Some(text[at + 1..].find(']').map_or(usize::MAX, |len| at + 1 + len));
                }
                let end = bracket_close.unwrap_or(usize::MAX);
                if end != usize::MAX {
                    if let Some(parts) = bracket(&text[at + 1..end]) {
                        return Some((at, end, Some(parts)));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// `a,b{c,d}`: the parts at the top level, when there are two or more.
fn list(inner: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let (mut depth, mut from) = (0, 0);
    for (at, c) in inner.char_indices() {
        depth += i32::from(c == '{') - i32::from(c == '}');
        if c == ',' && depth == 0 {
            parts.push(inner[from..at].to_owned());
            from = at + 1;
        }
    }
    parts.push(inner[from..].to_owned());
    (parts.len() > 1).then_some(parts)
}

/// `1..3` or `a..c`, with fewer than `BRACE_WORDS` steps.
fn sequence(inner: &str) -> Option<Vec<String>> {
    if inner.len() > 48 {
        return None;
    }
    let (from, to) = inner.split_once("..")?;
    if let (Ok(from), Ok(to)) = (from.parse::<i64>(), to.parse::<i64>()) {
        let (low, high) = (from.min(to), from.max(to));
        return (from.abs_diff(to) < BRACE_WORDS as u64).then(|| (low..=high).map(|n| n.to_string()).collect());
    }
    let (Some(from), Some(to)) = (single(from), single(to)) else { return None };
    let (low, high) = (from.min(to), from.max(to));
    Some((low..=high).map(|c| (c as char).to_string()).collect())
}

fn single(text: &str) -> Option<u8> {
    let [byte] = text.as_bytes() else { return None };
    byte.is_ascii_alphabetic().then_some(*byte)
}

/// `abc` or `a-c` inside `[...]`, with at most 64 letters. A `!`, `^` or
/// `[:class:]` bracket is not a plain list; `glob` reads it.
fn bracket(inner: &str) -> Option<Vec<String>> {
    if inner.is_empty() || inner.len() > 192 || inner.starts_with(['!', '^']) || inner.contains(['[', ':', '\\']) || !inner.is_ascii() {
        return None;
    }
    let bytes = inner.as_bytes();
    let mut parts = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes.get(at + 1) == Some(&b'-') && at + 2 < bytes.len() {
            if bytes[at] > bytes[at + 2] {
                return None;
            }
            parts.extend((bytes[at]..=bytes[at + 2]).map(|b| (b as char).to_string()));
            at += 3;
        } else {
            parts.push((bytes[at] as char).to_string());
            at += 1;
        }
    }
    (parts.len() <= 64).then_some(parts)
}

/// An absolute path with `.`, `..` and repeated separators resolved as
/// text; other paths unchanged.
fn normalize(path: &str) -> String {
    if !path.starts_with('/') {
        return path.to_owned();
    }
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    let tail = if path.len() > 1 && path.ends_with('/') && !parts.is_empty() { "/" } else { "" };
    format!("/{}{tail}", parts.join("/"))
}

/// True for a disk device, a pattern that can match one (`/dev/sd?`), and
/// a brace word that cannot be listed.
fn raw_disk(path: &str) -> bool {
    braces(path).is_none_or(|paths| paths.iter().any(|path| raw_disk_one(path)))
}

const DISKS: &[&str] = &["disk", "rdisk", "sd", "hd", "nvme", "mmcblk", "vd", "xvd"];

fn raw_disk_one(path: &str) -> bool {
    let path = normalize(path);
    if glob::pattern(&path) {
        // Any `/dev/<disk><letter or digit>...` name the pattern can start.
        // The pattern is read once, and `/dev/` once for every disk name.
        let glob = glob::Glob::new(&path);
        let dev = glob.run(glob.start(), "/dev/");
        return glob::Glob::open(&dev)
            && DISKS.iter().any(|disk| {
                let named = glob.run(dev.clone(), disk);
                glob::Glob::open(&named) && (b'0'..=b'9').chain(b'a'..=b'z').any(|c| glob::Glob::open(&glob.run(named.clone(), &(c as char).to_string())))
            });
    }
    let Some(name) = path.strip_prefix("/dev/") else { return false };
    DISKS.iter().any(|prefix| name.strip_prefix(prefix).is_some_and(|rest| rest.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())))
}

/// The program's name, or None for a path that is not a fixed system folder.
/// A path is read with `.`, `..` and repeated `/` resolved (`/bin//rm`,
/// `/bin/./sh`), and a name in lower case: the macOS file system does not
/// tell case apart, so `RM` and `/BIN/RM` run `/bin/rm` (Argon on 729a44c
/// and beacb8e, item 1).
fn program_name(text: &str) -> Option<String> {
    let text = normalize(text).to_ascii_lowercase();
    if !text.contains('/') {
        return Some(text);
    }
    for dir in ["/bin/", "/usr/bin/", "/sbin/", "/usr/sbin/", "/usr/local/bin/", "/opt/homebrew/bin/"] {
        if let Some(name) = text.strip_prefix(dir) {
            if !name.is_empty() && !name.contains('/') {
                return Some(name.to_owned());
            }
        }
    }
    None
}

/// The variable that `NAME=value`, `NAME+=value` or `NAME[i]=value` sets.
fn variable_name(text: &str) -> &str {
    let name = text.split(['=', '[']).next().unwrap_or("");
    name.strip_suffix('+').unwrap_or(name)
}

fn assignment(text: &str) -> bool {
    text.split_once('=').is_some_and(|(name, _)| {
        let name = name.strip_suffix('+').unwrap_or(name);
        !name.is_empty()
            && !name.starts_with(|c: char| c.is_ascii_digit())
            && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    })
}

/// True for `NAME=value` that can change what runs. Rust flags count only
/// when they go beyond lints, `--cfg` and plain code generation settings
/// (a `-C linker=prog` runs a program).
fn risky_assignment(text: &str) -> bool {
    let (name, value) = text.split_once('=').unwrap_or((text, ""));
    let name = name.strip_suffix('+').unwrap_or(name);
    if !matches!(name, "RUSTFLAGS" | "RUSTDOCFLAGS") {
        return risky_variable(name);
    }
    const PLAIN: &[&str] = &["opt-level", "debuginfo", "target-cpu", "target-feature", "codegen-units", "overflow-checks", "debug-assertions", "panic", "lto", "strip", "force-frame-pointers", "incremental"];
    let mut words = value.split_ascii_whitespace();
    while let Some(word) = words.next() {
        let setting = match word {
            "-D" | "-W" | "-A" | "-F" | "--cap-lints" | "--cfg" => {
                words.next();
                continue;
            }
            "-g" | "-O" => continue,
            "-C" => words.next().unwrap_or(""),
            _ if ["-D", "-W", "-A", "-F", "--cfg="].iter().any(|p| word.starts_with(p)) => continue,
            _ => match word.strip_prefix("-C") {
                Some(setting) => setting,
                None => return true,
            },
        };
        if !PLAIN.contains(&setting.split('=').next().unwrap_or("")) {
            return true;
        }
    }
    false
}

/// True unless setting `name` is known to be harmless. A name in capitals
/// is how programs read the environment (`RIPGREP_CONFIG_PATH`, `HOME`,
/// `GIT_PAGER`), so only the closed list below is safe. Other names are
/// shell variables, except the ones that zsh or common programs read
/// (`path` is zsh's `PATH`, curl reads `http_proxy`).
fn risky_variable(name: &str) -> bool {
    const ENVIRONMENT: &[&str] = &[
        "RUST_BACKTRACE", "RUST_LOG", "RUST_TEST_THREADS", "RUST_MIN_STACK", "CARGO_TARGET_DIR", "CARGO_INCREMENTAL", "CARGO_BUILD_JOBS",
        "CARGO_TERM_COLOR", "CARGO_TERM_VERBOSE", "CARGO_TERM_QUIET", "CARGO_TERM_PROGRESS_WHEN", "NO_COLOR", "CLICOLOR", "CLICOLOR_FORCE",
        "FORCE_COLOR", "TERM", "COLUMNS", "LINES", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_MESSAGES", "LC_NUMERIC", "LC_TIME",
        "TZ", "CI", "GOOS", "GOARCH", "GOMAXPROCS", "NODE_ENV", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
    ];
    const READ_LOWER: &[&str] = &[
        "path", "cdpath", "fpath", "manpath", "mailpath", "module_path", "psvar", "watch", "histchars", "prompt", "argv", "status", "pipestatus",
        "signals", "options", "parameters", "commands", "functions", "dis_functions", "aliases", "galiases", "saliases", "builtins", "reswords",
        "modules", "dirstack", "history", "historywords", "jobdirs", "jobstates", "jobtexts", "nameddirs", "userdirs", "usergroups", "funcstack",
        "functrace", "funcfiletrace", "funcsourcetrace", "patchars", "mapfile", "keymaps", "widgets", "zsh_eval_context", "zsh_scheduled_events",
        "zle_highlight", "http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy",
    ];
    if name.bytes().any(|b| b.is_ascii_lowercase()) {
        return READ_LOWER.contains(&name);
    }
    !ENVIRONMENT.contains(&name)
}

/// True when a short option cluster such as `-rf` contains `flag`.
fn short_flag(arg: &str, flag: char) -> bool {
    arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 1 && arg[1..].chars().all(|c| c.is_ascii_alphanumeric()) && arg[1..].contains(flag)
}

/// Arguments that are not options (everything after `--` counts).
fn operands(args: &[Word]) -> Vec<&Word> {
    let mut result = Vec::new();
    let mut options = true;
    for arg in args {
        if options && arg.text == "--" {
            options = false;
        } else if !options || !arg.text.starts_with('-') || arg.text == "-" {
            result.push(arg);
        }
    }
    result
}


/// Find `NAME(){ ... NAME|NAME& ... }` in shell text, with or without spaces.
fn fork_bomb(text: &str) -> bool {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut from = 0;
    while let Some(at) = compact[from..].find("(){") {
        let at = from + at;
        let before = &compact[..at];
        let mut name = &before[before.rfind(|c: char| ";&|(){}".contains(c)).map_or(0, |p| p + 1)..];
        if let Some(rest) = name.strip_prefix("function").filter(|rest| !rest.is_empty()) {
            name = rest;
        }
        if !name.is_empty() && (compact.contains(&format!("{name}|{name}&")) || compact.contains(&format!("{name}&{name}&")) || compact.contains(&format!("{name}|{name};"))) {
            return true;
        }
        from = at + 3;
    }
    false
}

/// Programs with a check that blocks: `rm -rf /`, `chmod -R` and `mv` of a
/// system folder, `find -delete`, and writes over a disk device.
const BLOCKING_PROGRAMS: [&str; 16] =
    ["rm", "chmod", "chown", "chgrp", "chflags", "mv", "dd", "find", "cp", "install", "rsync", "tee", "truncate", "sort", "curl", "wget"];

/// True when a program word that expands can name `name`: it has a
/// variable, a substitution or a brace, or its last path part is a pattern
/// that matches `name`.
fn can_name(word: &str, name: &str) -> bool {
    let last = word.rsplit('/').next().unwrap_or(word);
    word.contains(['$', '`', '{']) || (glob::pattern(last) && glob::Glob::new(last).matches(name)) || last == name
}

mod lexer;
mod options;
mod glob;
mod plain;
mod programs;
use lexer::{Lexer, Text};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod hardening_tests;
#[cfg(test)]
mod corpus_tests;
