//! Shell text to tokens: words after quote removal, operators, redirections
//! and here-documents. Nested command text is queued for its own check.
//! Text this lexer cannot model becomes a problem with a class, so the
//! command needs approval or is blocked; it never reads as safe.
use super::{Class, Tok, Word};

/// How deep `${...}` and `$((...))` text may nest before the check stops.
const MAX_DEPTH: usize = 16;

/// Splits shell text into words and operators, removes quotes, and collects
/// nested command text (`$(...)`, backquotes, `<(...)`) for its own check.
pub(super) struct Lexer {
    chars: Vec<char>,
    at: usize,
    depth: usize,
    pub(super) tokens: Vec<Tok>,
    /// Here-documents waiting for their body: (body index, delimiter, strip tabs, quoted).
    pending: Vec<(usize, String, bool, bool)>,
    pub(super) bodies: Vec<Option<String>>,
    pub(super) nested: Vec<String>,
    pub(super) problems: Vec<(Class, &'static str)>,
    /// Variables that arithmetic reads by name since the last finished
    /// word (`$((n))`, `${a[n]}`); `"$("` stands for command output.
    pub(super) arith: Vec<String>,
    /// Variables that `${name:=value}` or `${name=value}` sets.
    pub(super) assigned: Vec<String>,
    /// For `expansions`: what the text is, which decides how quotes read.
    inside: Option<Text>,
}

/// The text that `expansions` reads.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Text {
    /// A here-document or a stored value: quotes are plain characters.
    Plain,
    /// The inside of an unquoted `${...}` or `$((...))`: quotes quote.
    Unquoted,
    /// The inside of a double-quoted (or here-document) `${...}` or `$((...))`.
    Double,
}

/// What `balanced` is reading inside of.
#[derive(Clone, Copy, PartialEq)]
enum Scan {
    /// `$(...)` or `<(...)`: shell code with comments, quotes and here-documents.
    Code,
    /// A `case` statement inside code: `)` ends patterns, not the code.
    Case,
    Arith,
    Brace,
    Double,
    Back,
}

impl Lexer {
    pub(super) fn new(text: &str) -> Self {
        Self::with_depth(text, 0)
    }
    fn with_depth(text: &str, depth: usize) -> Self {
        Self { chars: text.chars().collect(), at: 0, depth, tokens: Vec::new(), pending: Vec::new(), bodies: Vec::new(), nested: Vec::new(), problems: Vec::new(), arith: Vec::new(), assigned: Vec::new(), inside: None }
    }
    /// Collect the command text that expansions in `text` would run, as in
    /// an unquoted here-document, `${...}` or `$((...))`.
    pub(super) fn expansions(text: &str, depth: usize, inside: Text) -> Self {
        let mut lexer = Self::with_depth(text, depth);
        lexer.inside = Some(inside);
        let mut scratch = Word::default();
        // In an unquoted `${x:-'$(...)'}` the quotes quote, as in a word.
        let quotes = inside == Text::Unquoted;
        let mut double = false;
        while let Some(c) = lexer.peek(0) {
            match c {
                '\\' => lexer.skip(2),
                '"' if quotes => {
                    double = !double;
                    lexer.skip(1);
                }
                '\'' if quotes && !double => {
                    lexer.skip(1);
                    while lexer.peek(0).is_some_and(|c| c != '\'') {
                        lexer.skip(1);
                    }
                    lexer.skip(1);
                }
                // `${x:-<(cmd)}` runs cmd as process substitution.
                '<' | '>' if quotes && !double && lexer.peek(1) == Some('(') => {
                    lexer.skip(2);
                    let inner = lexer.balanced(Scan::Code, false);
                    lexer.nested.push(inner);
                }
                '$' | '`' => lexer.expansion(&mut scratch, !quotes || double),
                _ => lexer.skip(1),
            }
        }
        lexer
    }
    fn peek(&self, offset: usize) -> Option<char> {
        self.chars.get(self.at + offset).copied()
    }
    fn skip(&mut self, count: usize) {
        self.at = (self.at + count).min(self.chars.len());
    }
    pub(super) fn run(&mut self) {
        if split_operator(&self.chars) {
            self.problems.push((Class::Blocked, "has a line continuation inside an operator such as << or $(, which the checker cannot read reliably"));
        }
        let mut word = Word::default();
        let mut in_word = false;
        while let Some(c) = self.peek(0) {
            match c {
                ' ' | '\t' => {
                    self.finish(&mut word, &mut in_word);
                    self.skip(1);
                }
                '\n' => {
                    self.finish(&mut word, &mut in_word);
                    self.skip(1);
                    self.tokens.push(Tok::Op("\n"));
                    self.read_heredocs();
                }
                '#' if !in_word => {
                    while self.peek(0).is_some_and(|c| c != '\n') {
                        self.skip(1);
                    }
                }
                // A backslash-newline joins lines: it is not part of any word.
                '\\' if self.peek(1) == Some('\n') => self.skip(2),
                '\\' => {
                    in_word = true;
                    word.text.extend(self.peek(1));
                    self.skip(2);
                }
                '\'' => {
                    in_word = true;
                    self.skip(1);
                    while let Some(c) = self.peek(0) {
                        self.skip(1);
                        if c == '\'' {
                            break;
                        }
                        word.text.push(c);
                    }
                }
                '"' => {
                    in_word = true;
                    self.skip(1);
                    while let Some(c) = self.peek(0) {
                        match c {
                            '"' => {
                                self.skip(1);
                                break;
                            }
                            '\\' => {
                                if let Some(next) = self.peek(1) {
                                    if !matches!(next, '$' | '`' | '"' | '\\' | '\n') {
                                        word.text.push('\\');
                                    }
                                    if next != '\n' {
                                        word.text.push(next);
                                    }
                                }
                                self.skip(2);
                            }
                            '$' | '`' => self.expansion(&mut word, true),
                            _ => {
                                word.text.push(c);
                                self.skip(1);
                            }
                        }
                    }
                }
                '$' | '`' => {
                    in_word = true;
                    self.expansion(&mut word, false);
                }
                '*' | '?' => {
                    in_word = true;
                    word.dynamic = true;
                    word.glob = true;
                    word.text.push(c);
                    self.skip(1);
                }
                '~' if !in_word => {
                    in_word = true;
                    word.dynamic = true;
                    word.text.push(c);
                    self.skip(1);
                }
                '|' | '&' | ';' | '(' | ')' | '<' | '>' => {
                    // A number right before a redirection names a descriptor.
                    let fd = in_word && !word.dynamic && matches!(c, '<' | '>') && !word.text.is_empty() && word.text.bytes().all(|b| b.is_ascii_digit());
                    if fd {
                        word = Word::default();
                        in_word = false;
                    }
                    self.finish(&mut word, &mut in_word);
                    if c == '(' && self.peek(1) == Some('(') {
                        self.problems.push((Class::Ask, "runs an arithmetic command (( )), which can run code through a variable"));
                    }
                    if matches!(c, '<' | '>') && self.peek(1) == Some('(') {
                        self.skip(2);
                        let inner = self.balanced(Scan::Code, false);
                        self.nested.push(inner);
                        word.dynamic = true;
                        word.text.push_str("/dev/fd/63");
                        in_word = true;
                        continue;
                    }
                    self.operator(c);
                }
                _ => {
                    in_word = true;
                    // A bracket glob or a brace list can still expand.
                    word.glob |= matches!(c, '[' | '{');
                    word.text.push(c);
                    self.skip(1);
                }
            }
        }
        self.finish(&mut word, &mut in_word);
        self.read_heredocs();
    }

    fn finish(&mut self, word: &mut Word, in_word: &mut bool) {
        if *in_word {
            word.arith.append(&mut self.arith);
            self.tokens.push(Tok::Word(std::mem::take(word)));
            *in_word = false;
        }
    }

    fn operator(&mut self, c: char) {
        let two: String = [c, self.peek(1).unwrap_or(' ')].iter().collect();
        let three: String = [c, self.peek(1).unwrap_or(' '), self.peek(2).unwrap_or(' ')].iter().collect();
        let (token, len) = match (three.as_str(), two.as_str()) {
            ("<<<", _) => (Tok::Redirect("<<<"), 3),
            ("&>>", _) => (Tok::Redirect(">>"), 3),
            ("<<-", _) => return self.heredoc(3, true),
            (_, "<<") => return self.heredoc(2, false),
            (_, "&&") => (Tok::Op("&&"), 2),
            (_, "||") => (Tok::Op("||"), 2),
            (_, ";;") => (Tok::Op(";;"), 2),
            (_, "|&") => (Tok::Op("|&"), 2),
            (_, ">>") => (Tok::Redirect(">>"), 2),
            (_, ">|") => (Tok::Redirect(">"), 2),
            (_, "&>") => (Tok::Redirect(">"), 2),
            (_, ">&") => (Tok::Redirect(">&"), 2),
            (_, "<&") => (Tok::Redirect("<&"), 2),
            (_, "<>") => (Tok::Redirect("<>"), 2),
            _ => (
                match c {
                    '|' => Tok::Op("|"),
                    '&' => Tok::Op("&"),
                    ';' => Tok::Op(";"),
                    '(' => Tok::Op("("),
                    ')' => Tok::Op(")"),
                    '<' => Tok::Redirect("<"),
                    _ => Tok::Redirect(">"),
                },
                1,
            ),
        };
        self.skip(len);
        // `>&file` writes a file; `>&2` duplicates a descriptor. The parser
        // tells them apart from the target word.
        self.tokens.push(token);
    }

    fn heredoc(&mut self, len: usize, strip: bool) {
        self.skip(len);
        let (delimiter, quoted) = self.delimiter();
        if delimiter.is_empty() {
            self.problems.push((Class::Ask, "has a here-document without a delimiter"));
            return;
        }
        let index = self.bodies.len();
        self.bodies.push(None);
        self.pending.push((index, delimiter, strip, quoted));
        self.tokens.push(Tok::Heredoc(index, quoted));
    }

    /// A here-document delimiter after the shell's quote removal: quotes and
    /// backslashes go, and the text inside quotes stays literal.
    fn delimiter(&mut self) -> (String, bool) {
        // Blanks and backslash-newlines may come before the word.
        while matches!(self.peek(0), Some(' ' | '\t')) || (self.peek(0) == Some('\\') && self.peek(1) == Some('\n')) {
            self.skip(if self.peek(0) == Some('\\') { 2 } else { 1 });
        }
        let mut text = String::new();
        let mut quoted = false;
        while let Some(c) = self.peek(0) {
            if c.is_whitespace() || matches!(c, ';' | '|' | '&' | '<' | '>' | '(' | ')') {
                break;
            }
            self.skip(1);
            match c {
                '\'' => {
                    quoted = true;
                    while let Some(c) = self.peek(0) {
                        self.skip(1);
                        if c == '\'' {
                            break;
                        }
                        text.push(c);
                    }
                }
                '"' => {
                    quoted = true;
                    while let Some(c) = self.peek(0) {
                        self.skip(1);
                        match c {
                            '"' => break,
                            '\\' if self.peek(0) == Some('\n') => self.skip(1),
                            '\\' if matches!(self.peek(0), Some('$' | '`' | '"' | '\\')) => {
                                text.extend(self.peek(0));
                                self.skip(1);
                            }
                            _ => text.push(c),
                        }
                    }
                }
                // A backslash-newline joins the next line to the delimiter.
                '\\' if self.peek(0) == Some('\n') => self.skip(1),
                '\\' => {
                    quoted = true;
                    text.extend(self.peek(0));
                    self.skip(1);
                }
                '$' | '`' => {
                    self.problems.push((Class::Ask, "has a here-document delimiter with $ or a backquote"));
                    text.push(c);
                }
                _ => text.push(c),
            }
        }
        (text, quoted)
    }

    fn read_heredocs(&mut self) {
        for (index, delimiter, strip, quoted) in std::mem::take(&mut self.pending) {
            self.bodies[index] = self.body(&delimiter, strip, quoted);
        }
    }

    /// Read lines up to the delimiter line. Unquoted documents join a line
    /// that ends in an odd number of backslashes to the next, as shells do
    /// before they compare the delimiter.
    fn body(&mut self, delimiter: &str, strip: bool, quoted: bool) -> Option<String> {
        let mut body = String::new();
        let mut line = String::new();
        while self.at < self.chars.len() {
            let end = self.chars[self.at..].iter().position(|&c| c == '\n').map_or(self.chars.len(), |p| self.at + p);
            let part: String = self.chars[self.at..end].iter().collect();
            self.at = (end + 1).min(self.chars.len());
            line.push_str(if strip && line.is_empty() { part.trim_start_matches('\t') } else { &part });
            let slashes = line.chars().rev().take_while(|&c| c == '\\').count();
            if !quoted && slashes % 2 == 1 && end < self.chars.len() {
                line.pop();
                continue;
            }
            if line == delimiter {
                return Some(body);
            }
            body.push_str(&line);
            body.push('\n');
            line.clear();
        }
        None
    }

    /// Read a `$` or backquote expansion into `word` and queue any command
    /// text. `quoted` is true inside double quotes or a here-document.
    fn expansion(&mut self, word: &mut Word, quoted: bool) {
        let before = word.dynamic;
        word.dynamic = true;
        if self.peek(0) == Some('`') {
            self.skip(1);
            let inner = self.balanced(Scan::Back, quoted);
            word.text.push_str("`...`");
            // Only double quotes make `\"` a quote: a here-document and an
            // unquoted `${...}` keep the backslash, and in "${...}" bash keeps
            // it while zsh does not.
            if quoted && matches!(self.inside, Some(Text::Unquoted | Text::Double)) && inner.contains("\\\"") {
                self.problems.push((Class::Blocked, "has \\\" in backquotes inside ${...}, which shells read differently"));
            }
            self.nested.push(unescape_backquotes(&inner, quoted && self.inside.is_none()));
            return;
        }
        if self.peek(1) == Some('\\') && self.peek(2) == Some('\n') {
            // Shells join `$\<newline>(` into `$(`; the checker does not.
            self.problems.push((Class::Blocked, "has a line continuation right after $, which the checker cannot read reliably"));
        }
        match (self.peek(1), self.peek(2)) {
            (Some('\''), _) if !quoted => {
                self.skip(2);
                word.dynamic = before;
                self.ansi_c(word);
            }
            (Some('('), Some('(')) => {
                let open = self.at + 2;
                self.skip(3);
                let mark = self.problems.len();
                let inner = self.balanced(Scan::Arith, quoted);
                if self.peek(0) == Some(')') {
                    self.skip(1);
                    self.arith.extend(arith_names(&inner));
                    self.absorb(&inner, quoted);
                    word.text.push_str("$((...))");
                } else {
                    // `$( (list) ... )`: a command substitution that starts with a
                    // subshell. The arithmetic scan skips no comment or
                    // here-document, so read the text again from `(` as code.
                    self.problems.truncate(mark);
                    self.at = open;
                    let inner = self.balanced(Scan::Code, false);
                    self.nested.push(inner);
                    word.text.push_str("$(...)");
                }
            }
            (Some('('), _) => {
                self.skip(2);
                let inner = self.balanced(Scan::Code, false);
                word.text.push_str("$(...)");
                self.nested.push(inner);
            }
            (Some('{'), _) => {
                self.skip(2);
                let inner = self.balanced(Scan::Brace, quoted);
                match brace_form(&inner) {
                    Some(Some(name)) => self.assigned.push(name),
                    Some(None) => {}
                    None => self.problems.push((Class::Ask, "uses a ${...} form the checker does not read")),
                }
                self.arith.extend(brace_arith_names(&inner));
                self.absorb(&inner, quoted);
                word.text.push_str(&format!("${{{inner}}}"));
            }
            (Some(n), _) if n.is_ascii_alphanumeric() || n == '_' => {
                self.skip(1);
                word.text.push('$');
                while let Some(c) = self.peek(0).filter(|c| c.is_ascii_alphanumeric() || *c == '_') {
                    word.text.push(c);
                    self.skip(1);
                }
            }
            (Some(n), _) if "@*#?$!-".contains(n) => {
                word.text.push('$');
                word.text.push(n);
                self.skip(2);
            }
            _ => {
                // A literal dollar sign keeps what the word already was. Any
                // other `$` form (`$[...]` arithmetic, zsh `$~x` and `$=x`) is
                // not read, so it is never safe.
                if !self.peek(1).is_none_or(|n| n.is_whitespace() || "\"'/.,:;)]}|&<>%\\`".contains(n)) {
                    self.problems.push((Class::Ask, "uses a $ form the checker does not read"));
                }
                word.dynamic = before;
                word.text.push('$');
                self.skip(1);
            }
        }
    }

    /// The body of a `$'...'` string, after the opening quote.
    fn ansi_c(&mut self, word: &mut Word) {
        while let Some(c) = self.peek(0) {
            self.skip(1);
            match c {
                '\'' => return,
                '\\' => {
                    let Some(next) = self.peek(0) else { break };
                    self.skip(1);
                    match next {
                        '\'' | '"' | '\\' | '?' => word.text.push(next),
                        'n' => word.text.push('\n'),
                        't' => word.text.push('\t'),
                        // `\c` takes the next character, quotes included.
                        'c' => return self.problems.push((Class::Blocked, "uses a $'\\c' escape, which the checker cannot read reliably")),
                        _ => {
                            word.dynamic = true;
                            word.text.push('\\');
                            word.text.push(next);
                        }
                    }
                }
                _ => word.text.push(c),
            }
        }
        self.problems.push((Class::Ask, "has an unclosed $'...' string"));
    }

    /// Queue the command text inside `${...}` or `$((...))` text.
    fn absorb(&mut self, text: &str, quoted: bool) {
        if self.depth >= MAX_DEPTH {
            return self.problems.push((Class::Blocked, "nests expansions too deeply to check"));
        }
        let inside = if quoted { Text::Double } else { Text::Unquoted };
        let inner = Self::expansions(text, self.depth + 1, inside);
        self.nested.extend(inner.nested);
        self.problems.extend(inner.problems);
        self.arith.extend(inner.arith);
        self.assigned.extend(inner.assigned);
    }

    /// Raw text up to the end of the construct `first`, read with the
    /// shell's rules for quotes, comments, nested expansions, `case`
    /// patterns and here-documents. One loop with an explicit stack, so deep
    /// nesting cannot exhaust the thread stack.
    /// `quoted` is true when `first` starts inside double quotes.
    fn balanced(&mut self, first: Scan, quoted: bool) -> String {
        let start = self.at;
        let mut stack = vec![first];
        let mut heredocs: Vec<(String, bool, bool)> = Vec::new();
        let mut word_start = true;
        // True where a command starts, so `case` and `esac` are reserved words.
        let mut command = true;
        // Inside `$((...))`: the previous character, and whether a `<<` was seen.
        let mut previous = '(';
        let mut shifted = false;
        // Stack depths of `$(`, `$((`, `<(` and `>(`: these end inside a
        // word, so a `#` right after their `)` is text, not a comment.
        let mut in_word: Vec<usize> = Vec::new();
        while let Some(&scan) = stack.last() {
            let Some(c) = self.peek(0) else {
                self.problems.push((Class::Ask, "has an unclosed substitution or quote"));
                return self.chars[start..].iter().collect();
            };
            let end = self.at;
            self.skip(1);
            let code = matches!(scan, Scan::Code | Scan::Case);
            let starts = word_start;
            word_start = code && matches!(c, ' ' | '\t' | '\n' | ';' | '&' | '|' | '(' | ')');
            // In arithmetic, `#` is valid only in a base (`16#ff`) or in `$#`.
            let based = previous.is_ascii_digit() || previous == '$';
            let bounded = !(previous.is_ascii_alphanumeric() || previous == '_');
            let redirect = matches!(previous, '<' | '>');
            previous = c;
            // `case` patterns end in `)`, and shells disagree on where the
            // statement starts inside `$(...)` (zsh, `[[ ... && case`), so the
            // end of the substitution cannot be found reliably.
            if matches!(scan, Scan::Code | Scan::Case | Scan::Arith) && c == 'c' && bounded && self.keyword(end, "case") {
                self.problems.push((Class::Blocked, "has the word case inside $(...), which the checker cannot read reliably"));
            }
            let operator = matches!(c, '\n' | ';' | '&' | '|' | '(' | ')');
            let separator = matches!(c, ' ' | '\t') || (c == '\\' && self.peek(0) == Some('\n'));
            let at_command = code && starts && command && !separator && !operator;
            if code && operator {
                command = true;
            } else if code && starts && !separator {
                // A word in argument position, such as `echo case`, is not a keyword.
                command = at_command && ["then", "do", "else", "elif", "if", "while", "until", "!", "{", "time"].iter().any(|w| self.keyword(end, w));
            }
            if scan == Scan::Arith && matches!(c, '\'' | '"' | ';') {
                // Not arithmetic: zsh then runs the text as commands.
                self.problems.push((Class::Blocked, "has quotes or ; inside $((...)), which shells read differently"));
            }
            match (scan, c) {
                (Scan::Back, '`') | (Scan::Double, '"') | (Scan::Brace, '}') | (Scan::Code | Scan::Arith, ')') => {
                    if in_word.last() == Some(&stack.len()) {
                        in_word.pop();
                        word_start = false;
                    }
                    stack.pop();
                    if stack.is_empty() {
                        return self.chars[start..end].iter().collect();
                    }
                }
                (_, '\\') => {
                    // A backslash-newline joins lines and keeps the word start.
                    if self.peek(0) == Some('\n') {
                        word_start = starts;
                        previous = ' ';
                    }
                    self.skip(1);
                }
                (Scan::Back, _) => {}
                // Valid arithmetic has no comment and no here-document; the
                // text is then a `$( (...) ...)` form the checker cannot split.
                (Scan::Arith, '#') if !based => self.problems.push((Class::Blocked, "has a comment inside $((...)), which the checker cannot read reliably")),
                (Scan::Arith, '<') if self.peek(0) == Some('<') => shifted = true,
                (Scan::Arith, '\n') if shifted => self.problems.push((Class::Blocked, "has a here-document inside $((...)), which the checker cannot read reliably")),
                (Scan::Case, '(' | ')') => {}
                (Scan::Code, '(') | (Scan::Arith, '(') => {
                    stack.push(scan);
                    if scan == Scan::Code && redirect {
                        in_word.push(stack.len());
                    }
                }
                (_, '`') => stack.push(Scan::Back),
                (_, '$') => match self.peek(0) {
                    Some('(') if self.peek(1) == Some('(') => {
                        self.skip(2);
                        in_word.push(stack.len() + 1);
                        stack.extend([Scan::Arith, Scan::Arith]);
                    }
                    Some('(') => {
                        self.skip(1);
                        stack.push(Scan::Code);
                        in_word.push(stack.len());
                    }
                    Some('{') => {
                        self.skip(1);
                        stack.push(Scan::Brace);
                    }
                    Some('\\') if self.peek(1) == Some('\n') => {
                        self.problems.push((Class::Blocked, "has a line continuation right after $, which the checker cannot read reliably"));
                    }
                    Some('\'') if scan != Scan::Double => {
                        // In "${x:-$'...'}" zsh reads the quote as text and
                        // bash as a quote: the end cannot be known.
                        let below = if stack.len() >= 2 { Some(stack[stack.len() - 2]) } else { quoted.then_some(Scan::Double) };
                        if scan == Scan::Brace && below == Some(Scan::Double) {
                            self.problems.push((Class::Blocked, "has $' inside \"${...}\", which shells read differently"));
                            continue;
                        }
                        self.skip(1);
                        let mut scratch = Word::default();
                        self.ansi_c(&mut scratch);
                    }
                    _ => {}
                },
                (Scan::Double, _) => {}
                (_, '\'') => {
                    // bash and sh read a quote in a double-quoted `${...}`; zsh does not.
                    let below = if stack.len() >= 2 { Some(stack[stack.len() - 2]) } else { quoted.then_some(Scan::Double) };
                    if scan == Scan::Brace && below == Some(Scan::Double) {
                        self.problems.push((Class::Blocked, "has a single quote inside \"${...}\", which shells read differently"));
                    }
                    while self.peek(0).is_some_and(|c| c != '\'') {
                        self.skip(1);
                    }
                    self.skip(1);
                }
                (_, '"') => stack.push(Scan::Double),
                (Scan::Code | Scan::Case, '#') if starts => {
                    while self.peek(0).is_some_and(|c| c != '\n') {
                        self.skip(1);
                    }
                }
                (Scan::Code | Scan::Case, '<') if self.peek(0) == Some('<') && self.peek(1) != Some('<') => {
                    self.skip(1);
                    let strip = self.peek(0) == Some('-');
                    if strip {
                        self.skip(1);
                    }
                    let (delimiter, quoted) = self.delimiter();
                    heredocs.push((delimiter, strip, quoted));
                }
                (Scan::Code | Scan::Case, '\n') => {
                    for (delimiter, strip, quoted) in std::mem::take(&mut heredocs) {
                        self.body(&delimiter, strip, quoted);
                    }
                    word_start = true;
                }
                (Scan::Code | Scan::Case, 'c') if at_command && self.keyword(end, "case") => {
                    self.skip(3);
                    stack.push(Scan::Case);
                }
                (Scan::Case, 'e') if at_command && self.keyword(end, "esac") => {
                    self.skip(3);
                    stack.pop();
                }
                _ => {}
            }
        }
        String::new()
    }

    /// True when the text at `from` is the reserved word `word`.
    fn keyword(&self, from: usize, word: &str) -> bool {
        let mut chars = self.chars[from..].iter();
        word.chars().all(|w| chars.next() == Some(&w))
            && chars.next().is_none_or(|c| c.is_whitespace() || matches!(c, ';' | '&' | '|' | '(' | ')' | '<' | '>'))
    }
}

/// Variables that arithmetic text reads, by name or as `$name`, in order.
/// `"$("` stands for command output, `"1"` or `"@"` for arguments. Numbers
/// (`16#ff`, `0x1f`), `$#`, `$?`, `$$`, `$!` and `${#name}` are not variables.
pub(super) fn arith_names(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut names = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            while chars.get(i).is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '#' | '@' | '_')) {
                i += 1;
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let from = i;
            while chars.get(i).is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_') {
                i += 1;
            }
            names.push(chars[from..i].iter().collect());
        } else if c == '`' {
            names.push("$(".into());
            i += 1;
        } else if c == '$' {
            match (chars.get(i + 1), chars.get(i + 2)) {
                // Nested arithmetic: its names follow.
                (Some('('), Some('(')) => i += 3,
                (Some('('), _) => {
                    names.push("$(".into());
                    i += 2;
                }
                // `${1}` and `${@}` are arguments; `${!x}` reads a name
                // from a value, so its value is not known.
                (Some('{'), Some(&d)) if d.is_ascii_digit() || matches!(d, '@' | '*' | '!') => {
                    names.push(if d == '!' { "$(".into() } else { d.to_string() });
                    i += 3;
                }
                (Some('{'), Some('#')) => {
                    while chars.get(i).is_some_and(|c| *c != '}') {
                        i += 1;
                    }
                }
                (Some(&d), _) if d.is_ascii_digit() || matches!(d, '@' | '*') => {
                    names.push(d.to_string());
                    i += 2;
                }
                (Some('#' | '?' | '$' | '!' | '-'), _) => i += 2,
                _ => i += 1,
            }
        } else {
            i += 1;
        }
    }
    names
}

/// The forms of `${...}` the checker reads: an optional `#` or `!`, a name,
/// a number or a special parameter, an optional index, then nothing or a
/// known operator. None for anything else (zsh `${(e)x}`, `${~x}`, bash
/// `${x@P}`), which can expand or run the value. `Some(Some(name))` when
/// the form sets the variable (`${x:=value}`, `${x=value}`, zsh `${x::=value}`).
fn brace_form(inner: &str) -> Option<Option<String>> {
    if inner.len() == 1 && "#!?$@*-".contains(inner) {
        return Some(None);
    }
    let body = inner.strip_prefix(['#', '!']).filter(|body| !body.is_empty()).unwrap_or(inner);
    let special = body.starts_with(|c: char| "@*#?$!-".contains(c));
    let length = if special { 1 } else { body.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(body.len()) };
    if length == 0 {
        return None;
    }
    let (name, mut rest) = body.split_at(length);
    if let Some(index) = rest.strip_prefix('[') {
        let mut depth = 1;
        let close = index.char_indices().find(|&(_, c)| {
            depth += i32::from(c == '[') - i32::from(c == ']');
            depth == 0
        })?;
        rest = &index[close.0 + 1..];
    }
    if ["=", ":=", "::="].iter().any(|op| rest.starts_with(op)) {
        return Some(Some(name.to_owned()));
    }
    let known = rest.is_empty()
        || rest.starts_with(['-', '?', '+', '#', '%', '/', '^', ',', ':'])
        || matches!(rest, "@" | "*")
        || (rest.len() == 2 && rest.starts_with('@') && "QEAKaUuLk".contains(&rest[1..]));
    known.then_some(None)
}

/// Variables that arithmetic reads in the text of a `${...}`: an array
/// index (`${a[n]}`) and the offset and length of `${x:n:m}`.
pub(super) fn brace_arith_names(inner: &str) -> Vec<String> {
    let body = inner.trim_start_matches(['#', '!']);
    let end = body.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(body.len());
    let mut rest = &body[end..];
    let mut names = Vec::new();
    // `${!x}` reads the variable that the value of x names, index and all.
    if inner.starts_with('!') && end > 0 {
        names.push(body[..end].to_owned());
    }
    if let Some(index) = rest.strip_prefix('[') {
        let mut depth = 1;
        let close = index.char_indices().find(|&(_, c)| {
            depth += i32::from(c == '[') - i32::from(c == ']');
            depth == 0
        });
        let close = close.map_or(index.len(), |(at, _)| at);
        if !matches!(&index[..close], "@" | "*") {
            names.extend(arith_names(&index[..close]));
        }
        rest = index.get(close + 1..).unwrap_or("");
    }
    if let Some(tail) = rest.strip_prefix(':').filter(|tail| !tail.starts_with(['-', '=', '?', '+'])) {
        names.extend(arith_names(tail));
    }
    names
}

/// Backquoted text as the shell reads it: a backslash only quotes `$`,
/// a backquote, another backslash, a newline, and `"` inside double quotes.
fn unescape_backquotes(raw: &str, quoted: bool) -> String {
    let mut text = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek().copied() {
                Some(next @ ('$' | '`' | '\\')) => {
                    chars.next();
                    text.push(next);
                    continue;
                }
                Some('"') if quoted => {
                    chars.next();
                    text.push('"');
                    continue;
                }
                Some('\n') => {
                    chars.next();
                    continue;
                }
                _ => {}
            }
        }
        text.push(c);
    }
    text
}

/// True when a line continuation (`\` and a newline, which the shell
/// removes first) joins two characters into an operator: `<\`, newline,
/// `<EOF` is `<<EOF`, and `$(\`, newline, `(` is `$((`. A `$` right before
/// one is blocked too (`$\`, newline, `HOME`).
fn split_operator(chars: &[char]) -> bool {
    (1..chars.len()).any(|at| {
        if chars[at] != '\n' {
            return false;
        }
        let run = chars[..at].iter().rev().take_while(|&&c| c == '\\').count();
        let before = at.checked_sub(run + 1).map(|b| chars[b]);
        let after = chars.get(at + 1).copied();
        run % 2 == 1
            && (before == Some('$') || (before.is_some_and(|c| "<>&|;(".contains(c)) && after.is_some_and(|c| "(<>&|;{'\"`".contains(c))))
    })
}
