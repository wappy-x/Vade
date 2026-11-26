//! Positive option reading. An option counts as harmless only when it is on
//! the program's closed list. Any other option, and any word that the shell
//! can still turn into an option, is not read, so the command is not safe.
//!
//! A list is one string of options separated by spaces: `-x` takes no
//! value, `-x=` takes one (attached, `--name=value` or the next word), `-x?`
//! takes only an attached value (`-U3`, `--stat=50`), and `-#` is a number
//! option such as `-5`.
use super::Word;

/// How a program reads its options.
#[derive(Clone, Copy)]
pub(super) struct Mode {
    /// The first operand ends the options (getopt as POSIX has it, and a
    /// wrapper, whose command starts there). Otherwise options can follow
    /// operands, as GNU tools and git read them.
    pub stop: bool,
    /// Short options combine in one word (`-rf`, `-n5`). Otherwise each
    /// option is a whole word and `--name` is the same as `-name` (Go flags).
    pub clusters: bool,
}
pub(super) const GNU: Mode = Mode { stop: false, clusters: true };
pub(super) const POSIX: Mode = Mode { stop: true, clusters: true };
pub(super) const WHOLE: Mode = Mode { stop: true, clusters: false };
pub(super) const WHOLE_ANY: Mode = Mode { stop: false, clusters: false };

/// What `read` found.
#[derive(Default)]
pub(super) struct Read {
    /// Each option read, as listed (without `=` or `?`), with its value.
    pub options: Vec<(&'static str, Option<Word>)>,
    /// Indexes of the operands, in order.
    pub operands: Vec<usize>,
    /// Index of the `--` that ended the options.
    pub dashdash: Option<usize>,
    /// Index of the first word that is not read.
    pub unread: Option<usize>,
}
impl Read {
    pub(super) fn has(&self, names: &[&str]) -> bool {
        self.options.iter().any(|(option, _)| names.contains(option))
    }
    pub(super) fn values<'a>(&'a self, names: &'a [&str]) -> impl Iterator<Item = &'a Word> {
        self.options.iter().filter(move |(option, _)| names.contains(option)).filter_map(|(_, value)| value.as_ref())
    }
}

/// True when the shell can still turn `word` into other words: an
/// expansion, a glob or a brace list, at the start or inside the word
/// (`/dev/{null,sda}` is two words). One of them may start with `-`, and
/// a split moves each word after it to another place.
pub(super) fn expands(word: &Word) -> bool {
    let text = word.text.as_str();
    // A `[` with no `]` after it is a plain letter (the `[` command).
    let bracket = text.find('[').is_some_and(|at| text[at + 1..].contains(']'));
    (word.dynamic && text.contains(['$', '`'])) || (word.glob && (text.contains(['*', '?']) || bracket || (text.contains('{') && (text.contains(',') || text.contains("..")))))
}

/// Read `args` against the closed option list `spec`. An operand that can
/// split is not read, also after `--` or after the options stop: a split
/// moves each later operand, and a program may write to one by its place.
pub(super) fn read(args: &[Word], spec: &'static str, mode: Mode) -> Read {
    scan(args, spec, mode, true)
}

/// `read` for a wrapper: the words after its options are the command it
/// runs, which is checked as a command of its own.
pub(super) fn read_head(args: &[Word], spec: &'static str, mode: Mode) -> Read {
    scan(args, spec, mode, false)
}

fn scan(args: &[Word], spec: &'static str, mode: Mode, operands: bool) -> Read {
    let mut read = Read::default();
    let mut options = true;
    let mut at = 0;
    while let Some(word) = args.get(at) {
        let text = word.text.as_str();
        if !options && operands && expands(word) {
            read.unread = Some(at);
            return read;
        } else if !options {
            read.operands.push(at);
        } else if text == "--" && !word.dynamic {
            read.dashdash = Some(at);
            options = false;
        } else if expands(word) {
            read.unread = Some(at);
            return read;
        } else if !text.starts_with('-') || text == "-" {
            read.operands.push(at);
            options = !mode.stop;
        } else {
            match option(args, at, spec, mode) {
                Some((taken, found)) => {
                    read.options.extend(found);
                    at += taken;
                }
                None => {
                    read.unread = Some(at);
                    return read;
                }
            }
        }
        at += 1;
    }
    read
}

type Found = Vec<(&'static str, Option<Word>)>;

/// Read the option word at `at`: how many words after it are its value,
/// and the options it holds. None when any part is not on the list.
fn option(args: &[Word], at: usize, spec: &'static str, mode: Mode) -> Option<(usize, Found)> {
    let word = &args[at];
    let text = match word.text.strip_prefix('-') {
        Some(long) if !mode.clusters && long.starts_with('-') => long,
        _ => word.text.as_str(),
    };
    let next = || args.get(at + 1).filter(|value| value_word(value)).cloned();
    let attached = |value: &str| Some(Word { text: value.to_owned(), ..word.clone() });
    if let Some(token) = find(spec, text) {
        return match token.as_bytes().last() {
            Some(b'=') => Some((1, vec![(name(token), Some(next()?))])),
            _ => Some((0, vec![(name(token), None)])),
        };
    }
    if text.starts_with("--") || !mode.clusters {
        let (key, value) = text.split_once('=')?;
        let token = find(spec, key).filter(|token| token.ends_with(['=', '?']))?;
        return Some((0, vec![(name(token), attached(value))]));
    }
    let letters = &text[1..];
    if letters.bytes().all(|b| b.is_ascii_digit()) && find(spec, "-#").is_some() {
        return Some((0, vec![("-#", None)]));
    }
    let mut found = Vec::new();
    for (index, letter) in letters.char_indices() {
        let token = find(spec, &format!("-{letter}"))?;
        let rest = &letters[index + letter.len_utf8()..];
        match token.as_bytes().last() {
            Some(b'=') if rest.is_empty() => {
                found.push((name(token), Some(next()?)));
                return Some((1, found));
            }
            Some(b'=' | b'?') => {
                found.push((name(token), attached(rest)));
                return Some((0, found));
            }
            _ => found.push((name(token), None)),
        }
    }
    Some((0, found))
}

/// A word that can be an option's value: it cannot expand, and it does not
/// look like an option itself (a negative number such as `-1` can).
pub(super) fn value_word(word: &Word) -> bool {
    let text = word.text.as_str();
    !expands(word) && (!text.starts_with('-') || text == "-" || text[1..].starts_with(|c: char| c.is_ascii_digit()))
}

fn find(spec: &'static str, option: &str) -> Option<&'static str> {
    spec.split_ascii_whitespace().find(|token| name(token) == option)
}

fn name(token: &'static str) -> &'static str {
    token.trim_end_matches(['=', '?'])
}

/// True for a variable name with nothing the shell would read further.
pub(super) fn identifier(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') && text.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

// The closed lists of the programs with many options. Each holds only
// options that print, select or format; none writes a file, runs a
// program or sets a variable.

pub(super) const RG: &str = "-i --ignore-case -s --case-sensitive -S --smart-case -w --word-regexp -x --line-regexp -F --fixed-strings -v --invert-match \
    -n --line-number -N --no-line-number -l --files-with-matches --files-without-match -c --count --count-matches -o --only-matching -e= --regexp= -f= --file= \
    -g= --glob= --iglob= --glob-case-insensitive -t= --type= -T= --type-not= --type-list -A= --after-context= -B= --before-context= -C= --context= -m= --max-count= \
    -d= --max-depth= -u --unrestricted -. --hidden --no-ignore --no-ignore-vcs --no-ignore-parent --no-ignore-dot --no-ignore-global --no-ignore-files --ignore-file= \
    --files -U --multiline --multiline-dotall -P --pcre2 --engine= -a --text -L --follow -0 --null -H --with-filename -I --no-filename --heading --no-heading \
    --json --vimgrep --column --no-column --trim --stats -q --quiet -r= --replace= --passthru --no-messages -M= --max-columns= --max-columns-preview \
    --max-filesize= --sort= --sortr= --color= --colors= -p --pretty -b --byte-offset --one-file-system --null-data --crlf --line-buffered --block-buffered \
    -j= --threads= --path-separator= --context-separator= --no-context-separator --field-match-separator= --field-context-separator= -E= --encoding= \
    --binary --include-zero --unicode --no-unicode -h --help -V --version";

pub(super) const FD: &str = "-H --hidden -I --no-ignore -u --unrestricted --no-ignore-vcs --no-ignore-parent --no-require-git -s --case-sensitive -i --ignore-case \
    -g --glob --regex -F --fixed-strings -a --absolute-path -l --list-details -L --follow -p --full-path -0 --print0 -d= --max-depth= --min-depth= --exact-depth= \
    -t= --type= -e= --extension= -E= --exclude= -S= --size= --changed-within= --changed-before= --newer= --older= -c= --color= -j= --threads= --max-results= -1 \
    -q --quiet --show-errors --prune --one-file-system --mount --xdev --strip-cwd-prefix? --base-directory= --search-path= --path-separator= --ignore-file= \
    --and= --format= -o= --owner= -h --help -V --version";

pub(super) const SORT: &str = "-b --ignore-leading-blanks -d --dictionary-order -f --ignore-case -g --general-numeric-sort -h --human-numeric-sort \
    -i --ignore-nonprinting -M --month-sort -n --numeric-sort -R --random-sort -r --reverse -V --version-sort -c --check? -C -k= --key= -m --merge \
    -o= --output= -s --stable -S= --buffer-size= -t= --field-separator= -u --unique -z --zero-terminated --parallel= --debug --sort= --help --version";

/// `find` tests and actions that print or select, with no value and with one.
pub(super) const FIND_FLAGS: &str = "( ) ! , -not -and -a -or -o -print -print0 -ls -prune -depth -d -empty -true -false -follow -xdev -mount -x \
    -nouser -nogroup -readable -writable -executable -quit -daystart -nowarn -warn -noleaf -ignore_readdir_race -noignore_readdir_race";
pub(super) const FIND_VALUES: &str = "-name -iname -path -ipath -wholename -iwholename -regex -iregex -type -xtype -maxdepth -mindepth -mtime -mmin \
    -atime -amin -ctime -cmin -Btime -Bmin -Bnewer -anewer -cnewer -mnewer -newer -size -perm -user -group -uid -gid -links -inum -samefile -lname -ilname \
    -regextype -fstype -used -printf -flags";

/// The options of each git command that only shows history, files or
/// settings. A list holds only the command's own options, so a letter
/// reads as git reads it there (`-n` takes no value in `git grep`, and a
/// cluster such as `-nOsh` then still has to match). `-x?` marks a value
/// that git takes attached; a value that git takes as the next word too
/// is also marked `?`, so that word must be on the list on its own.
macro_rules! diff_options {
    () => {
        "-p -u -s -z -R -a -b -w -W -D -U? -M? -C? -B? -l? -S? -G? -I? -c -m -t -r --cc --patch --no-patch --raw --patch-with-raw \
        --patch-with-stat --stat? --shortstat --numstat --dirstat? --compact-summary --summary --name-only --name-status --submodule? --color? --no-color \
        --color-moved? --no-color-moved --color-moved-ws? --word-diff? --word-diff-regex? --color-words? --no-renames --renames --rename-empty \
        --no-rename-empty --check --ws-error-highlight? --full-index --binary --abbrev? --no-abbrev --break-rewrites? --find-renames? --find-copies? \
        --find-copies-harder --irreversible-delete --diff-filter? --text --ignore-cr-at-eol --ignore-space-at-eol --ignore-space-change --ignore-all-space \
        --ignore-blank-lines --ignore-matching-lines? --inter-hunk-context? --function-context --exit-code --quiet --no-ext-diff --no-textconv \
        --ignore-submodules? --src-prefix? --dst-prefix? --no-prefix --default-prefix --line-prefix? --relative? --no-relative --minimal --histogram \
        --patience --diff-algorithm? --anchored? --indent-heuristic --no-indent-heuristic --unified? --stat-width? --stat-name-width? --stat-graph-width? \
        --stat-count? --pickaxe-all --pickaxe-regex --combined-all-paths --remerge-diff --diff-merges? --no-diff-merges --dd"
    };
}
/// `git diff` and the diff plumbing commands.
pub(super) const GIT_DIFF: &str = concat!(diff_options!(), " -0 -1 -2 -3 --base --ours --theirs --cached --staged --merge-base --no-index --root \
    --creation-factor? --no-dual-color --left-only --right-only");
/// `git log`, `show`, `whatchanged` and `rev-list`.
pub(super) const GIT_LOG: &str = concat!(diff_options!(), " -# -n? -i -E -F -P -g -L? --oneline --graph --all --decorate? --no-decorate --format? \
    --pretty? --abbrev-commit --no-abbrev-commit --date? --relative-date --since? --until? --after? --before? --author? --committer? --grep? --all-match \
    --invert-grep --regexp-ignore-case --extended-regexp --basic-regexp --fixed-strings --perl-regexp --merges --no-merges --first-parent --reverse \
    --follow --max-count? --skip? --min-parents? --max-parents? --no-min-parents --no-max-parents --branches? --tags? --remotes? --glob? --exclude? \
    --full-history --dense --sparse --simplify-by-decoration --ancestry-path? --left-right --left-only --right-only --cherry-pick --cherry-mark --cherry \
    --boundary --topo-order --date-order --author-date-order --no-walk? --do-walk --no-notes --notes? --show-notes? --encoding? --expand-tabs? \
    --no-expand-tabs --log-size --source --use-mailmap --mailmap --no-use-mailmap --walk-reflogs --merge --parents --children --show-pulls --count \
    --objects --header --timestamp --decorate-refs? --decorate-refs-exclude? --clear-decorations --in-commit-order --reflog --single-worktree");
/// `git grep`. It has no `-O` (`--open-files-in-pager` runs a program).
pub(super) const GIT_GREP: &str = "-# -a --text -i --ignore-case -I -w --word-regexp -v --invert-match -h -H --full-name -E --extended-regexp \
    -G --basic-regexp -F --fixed-strings -P --perl-regexp -n --line-number --column -l --files-with-matches --name-only -L --files-without-match \
    -c --count -o --only-matching -q --quiet -z --null -W --function-context -p --show-function --break --heading --color? --no-color -r --recursive \
    --no-recursive -A= --after-context= -B= --before-context= -C= --context= -m= --max-count= --max-depth= --threads= -e= -f= --and --or --not \
    --all-match --cached --untracked --no-index --exclude-standard --no-exclude-standard --recurse-submodules --no-textconv";

/// The closed option list of a git command that only reads, by name.
pub(super) fn git_read(command: &str) -> Option<&'static str> {
    Some(match command {
        "diff" | "diff-tree" | "diff-files" | "diff-index" | "range-diff" => GIT_DIFF,
        "log" | "show" | "whatchanged" | "rev-list" => GIT_LOG,
        "grep" => GIT_GREP,
        "status" => "-s --short -b --branch --show-stash --porcelain? --long -v --verbose -u? --untracked-files? --ignore-submodules? --ignored? -z --null \
            --column? --no-column --ahead-behind --no-ahead-behind --renames --no-renames --find-renames?",
        "blame" => "-b --root --show-stats -f --show-name -n --show-number -p --porcelain --line-porcelain -c -t -l -s -e --show-email -w --incremental \
            -L? -S? -M? -C? --reverse? --first-parent --encoding? --contents? --date? --ignore-rev? --ignore-revs-file? --abbrev? --color-lines --color-by-age \
            --progress --no-progress --minimal --score-debug",
        "describe" => "--all --tags --contains --abbrev? --candidates? --exact-match --debug --long --match? --exclude? --always --first-parent --dirty? --broken?",
        "rev-parse" => "-q --quiet --short? --verify --abbrev-ref? --symbolic --symbolic-full-name --show-toplevel --show-prefix --show-cdup --git-dir \
            --git-common-dir --absolute-git-dir --is-inside-work-tree --is-inside-git-dir --is-bare-repository --is-shallow-repository --default? \
            --show-object-format? --show-ref-format --local-env-vars --git-path? --path-format? --show-superproject-working-tree --all --branches? --tags? \
            --remotes? --glob? --exclude? --since? --until? --after? --before? --not --revs-only --no-revs --flags --no-flags --sq --shared-index-path",
        "ls-files" => "-z -t -v -f -c --cached -d --deleted -m --modified -o --others -i --ignored -s --stage -k --killed -u --unmerged --resolve-undo \
            --directory --no-empty-directory --eol --deduplicate --full-name --recurse-submodules --error-unmatch --with-tree? --abbrev? --format? \
            --exclude-standard --sparse -x? --exclude? -X? --exclude-from? --exclude-per-directory? --debug",
        "ls-tree" => "-d -r -t -l --long -z --name-only --name-status --object-only --full-name --full-tree --abbrev? --format?",
        "cat-file" => "-t -s -e -p --batch? --batch-check? --batch-all-objects --unordered --buffer --follow-symlinks --allow-unknown-type -Z -z --mailmap \
            --no-mailmap --use-mailmap",
        "merge-base" => "-a --all --octopus --independent --is-ancestor --fork-point",
        "for-each-ref" => "--count? --sort? --format? --color? -s --shell -p --perl --python --tcl --points-at? --merged? --no-merged? --contains? \
            --no-contains? --ignore-case --omit-empty --exclude?",
        "show-ref" => "--head --heads --branches --tags -d --dereference -s --hash? --verify --abbrev? -q --quiet --exists",
        "count-objects" => "-v --verbose -H --human-readable",
        "check-ignore" => "-q --quiet -v --verbose -n --non-matching --no-index -z",
        "version" => "--build-options",
        _ => return None,
    })
}
/// `git branch` and `git tag` options that only list.
pub(super) const GIT_LIST: &str = "-a --all -r --remotes -l --list -v -vv --verbose -q --quiet --contains? --no-contains? --merged? --no-merged? --points-at? \
    --sort? --format? --color? --no-color --column? --no-column --show-current --abbrev? --no-abbrev -i --ignore-case --omit-empty -n?";
/// `git config` options that only read.
pub(super) const GIT_CONFIG: &str = "--get --get-all --get-regexp --get-urlmatch --list -l --show-origin --show-scope --global --local --system --worktree \
    -f= --file= --blob= --name-only -z --null --type= --bool --int --bool-or-int --path --expiry-date --default= --includes --no-includes --all --regexp --value= --url=";

/// cargo options that pick packages, targets, features, profiles or output
/// format. `--open` (doc), `--fix` (clippy) and `--check` (fmt) are read so
/// that each command can decide on them.
pub(super) const CARGO: &str = "-p= --package= --workspace --all --exclude= --lib --bins --bin= --examples --example= --tests --test= --benches --bench= \
    --all-targets -F= --features= --all-features --no-default-features -r --release --profile= -j= --jobs= --target= --target-dir= --manifest-path= \
    --message-format= -v -vv --verbose -q --quiet --color= --frozen --locked --offline --keep-going --timings? --no-run --no-fail-fast --doc \
    --ignore-rust-version --future-incompat-report --document-private-items --no-deps --format-version= --filter-platform= -e= --edges= -i= --invert= \
    -d --duplicates --depth= --prefix= --charset= --format= -f= --no-dedupe -E= --filterset= --test-threads= -P= --cargo-profile= --retries= --nocapture \
    --no-capture --run-ignored= --partition= --status-level= --final-status-level= --failure-output= --success-output= --hide-progress-bar --no-tests= \
    --open --fix --allow-dirty --allow-staged --allow-no-vcs --check -h --help -V --version --list";

/// Go build and test flags that do not run another program. Flags that
/// write a file (`-o`, profiles) are read so that the command can ask.
pub(super) const GO_BUILD: &str = "-v -x -n -a -race -msan -asan -cover -covermode= -coverpkg= -count= -run= -skip= -bench= -benchmem -benchtime= \
    -timeout= -short -failfast -p= -parallel= -cpu= -tags= -mod= -trimpath -json -buildvcs? -gcflags= -work -shuffle= -list= -fullpath -C= -f= -m -deps \
    -e -find -test -compiled -export -o= -coverprofile= -cpuprofile= -memprofile= -blockprofile= -mutexprofile= -trace= -outputdir=";
/// The Go flags of `GO_BUILD` that write a file.
pub(super) const GO_WRITES: &[&str] = &["-o", "-coverprofile", "-cpuprofile", "-memprofile", "-blockprofile", "-mutexprofile", "-trace", "-outputdir"];
