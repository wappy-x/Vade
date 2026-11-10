//! The program table: which programs and arguments are read-only or build
//! work, which need approval and which are blocked. A program rule calls a
//! command safe only by positive match: each option is on the program's
//! closed list (see options.rs) and each other word is a plain operand.
use super::options::{self, Mode, Read, GNU, POSIX, WHOLE, WHOLE_ANY};
use super::*;

impl Ctx {
    pub(super) fn program(&mut self, name: &str, args: &[Word], display: &[String], resolved: &mut Resolved) {
        let text: Vec<&str> = args.iter().map(|w| w.text.as_str()).collect();
        let positional: Vec<&Word> = operands(args);
        let ask = |ctx: &mut Self, reason: &str| ctx.add(Class::Ask, reason, display, true);
        match name {
            // Wrappers: classify the command they run.
            "env" => self.env(args, display, resolved),
            "arch" | "nice" | "nohup" | "time" | "builtin" | "exec" | "stdbuf" | "caffeinate" | "timeout" | "gtimeout" | "xargs" | "command" | "sudo" | "doas" | "su" | "runas" | "pkexec" => {
                self.wrapper(name, args, display, resolved)
            }
            // Shells: read the script they would run.
            "sh" | "bash" | "zsh" | "dash" | "ksh" | "fish" | "csh" | "tcsh" => self.shell(name, args, display, resolved),
            "python" | "python2" | "python3" | "perl" | "ruby" | "node" | "nodejs" | "deno"
            | "php" | "lua" | "osascript" | "Rscript" | "tclsh" | "irb" => {
                // Only flags that print and exit; `python3 -v` and `ruby -v`
                // turn on verbose output and still run a script.
                let exits: &[&str] = match name {
                    "python" | "python2" | "python3" => &["--version", "-V", "--help", "-h"],
                    "perl" | "node" | "nodejs" | "php" => &["--version", "-v", "--help", "-h"],
                    _ => &["--version", "--help"],
                };
                if !text.is_empty() && text.iter().all(|a| exits.contains(a)) {
                    return;
                }
                if positional.is_empty() && !text.iter().any(|a| a.starts_with("-c") || a.starts_with("-e") || *a == "-m") {
                    resolved.stdin_script = true;
                    return;
                }
                ask(self, "runs a script interpreter");
            }
            "eval" | "source" | "." | "trap" | "alias" => {
                self.add(Class::Ask, "runs text as shell code", display, false);
                // Check literal code too, so a blocked command stays blocked.
                // Code with an expansion is still checked as written.
                if name == "eval" && !args.is_empty() {
                    let text: Vec<&str> = args.iter().map(|w| w.text.as_str()).collect();
                    self.script(&text.join(" "));
                }
                // Any `trap` word can be the action (`trap -- 'code' EXIT`).
                if name == "trap" {
                    for word in args {
                        self.script(&word.text);
                    }
                }
                if name == "alias" {
                    for value in args.iter().filter_map(|w| w.text.split_once('=').map(|(_, v)| v)) {
                        self.script(value);
                    }
                }
            }
            "export" | "declare" | "typeset" | "local" | "readonly" => {
                let Some(read) = self.options(args, "-x -r -p -g -l -u", GNU, display) else {
                    // `-i` reads values as arithmetic: code in them stays blocked.
                    return args.iter().for_each(|word| self.stored_code(word, display));
                };
                for &at in &read.operands {
                    // The name before `=` must be plain; the value may expand.
                    let word = &args[at];
                    let name = word.text.split('=').next().unwrap_or("");
                    if !options::identifier(name) {
                        self.add(Class::Ask, format!("names a variable that the checker cannot read: {}", word.text), display, false);
                    }
                    self.assignment_word(word, display);
                }
            }
            "let" => {
                self.add(Class::Ask, "runs arithmetic, which can run code through a variable", display, false);
                args.iter().for_each(|word| self.stored_code(word, display));
            }
            "test" | "[" => {
                // `-v` takes a name, and an index in it runs as code. A word
                // that expands can itself be `-v`.
                for (at, word) in args.iter().enumerate() {
                    if (word.text == "-v" && !word.dynamic) || options::expands(word) {
                        if let Some(name) = args.get(at + 1).filter(|w| w.text != "]") {
                            self.indexed_name(name, display);
                        }
                    }
                }
            }
            // Read-only tools: no option of these writes a file, runs a
            // program or sets a variable, so any argument is safe.
            "ls" | "cat" | "head" | "tail" | "wc" | "grep" | "egrep" | "fgrep" | "pwd" | "echo"
            | "true" | "false" | ":" | "which" | "whereis" | "type"
            | "stat" | "du" | "df" | "basename" | "dirname" | "realpath" | "readlink"
            | "cut" | "tr" | "nl" | "diff" | "cmp" | "comm" | "column" | "fold" | "expand"
            | "unexpand" | "rev" | "tac" | "od" | "hexdump" | "strings" | "md5" | "md5sum"
            | "shasum" | "sha1sum" | "sha256sum" | "sha512sum" | "cksum" | "jq" | "whoami"
            | "id" | "groups" | "uname" | "printenv" | "sleep" | "seq" | "yes"
            | "ps" | "uptime" | "sw_vers" | "nproc" | "getconf" | "locale" | "tty"
            | "lsof" | "pgrep" | "cd" | "pushd" | "popd" | "paste" | "join" | "look" | "cal"
            | "vm_stat" | "free" | "lscpu" | "nm" | "otool" | "objdump" | "size"
            | "tokei" | "zcat" | "gzcat" | "bzcat" | "xzcat" => {}
            // zsh reads a count or a status as arithmetic, which can run code.
            "shift" | "exit" | "return" | "wait" => {
                let plain = |w: &Word| (!w.dynamic && !w.text.is_empty() && w.text.trim_start_matches('%').bytes().all(|b| b.is_ascii_digit())) || matches!(w.text.as_str(), "$?" | "$!");
                if (name != "wait" && args.len() > 1) || !args.iter().all(plain) {
                    self.add(Class::Ask, format!("{name} reads a value that the checker cannot read as a plain number"), display, false);
                    args.iter().for_each(|word| self.stored_code(word, display));
                }
            }
            "set" => self.set(args, display),
            "printf" => self.printf(args, display),
            "read" | "unset" | "mapfile" | "readarray" | "getopts" => self.names(name, args, display),
            "less" | "more" | "bat" | "man" => {
                let spec = match name {
                    "bat" => "-n --number -p --plain -A --show-all -l= --language= -H= --highlight-line= -r= --line-range= --style= --color= --paging= --theme= --tabs= --wrap= -S --chop-long-lines --decorations= --list-languages -L --list-themes -d --diff --file-name= -u --unbuffered --terminal-width= -h --help -V --version",
                    "man" => "-k -f -w -a -W -M= -S= -s=",
                    _ => "-N -S -R -r -X -F -E -e -i -I -n -M -m -Q -q -s -w -J -g -G -c -a -f -B -L -x= -z= -j= -y= -h= -b= --chop-long-lines --RAW-CONTROL-CHARS --no-init --quit-if-one-screen --quit-at-eof --ignore-case --LINE-NUMBERS --line-numbers --squeeze-blank-lines --tabs= --version --help",
                };
                // `+cmd` runs a pager command first.
                if text.iter().any(|a| a.starts_with('+')) {
                    return self.add(Class::Ask, "runs a pager command", display, false);
                }
                self.options(args, spec, GNU, display);
            }
            "file" => {
                self.options(args, "-b --brief -i --mime --mime-type --mime-encoding -L --dereference -h --no-dereference -z --uncompress -Z -k --keep-going -N --no-pad -0 --print0 -s --special-files -E -I -r --raw -l --list -f= --files-from= -F= --separator= -m= --magic-file= -e= --exclude= --extension --apple -n --no-buffer --help -v --version", GNU, display);
            }
            "fd" | "fdfind" => self.fd(args, display, resolved),
            "rg" => {
                self.options(args, options::RG, GNU, display);
            }
            "sort" => {
                self.options(args, options::SORT, GNU, display);
                // Also after an option that is not read (`sort --mmap -o /dev/sda`).
                let outputs = output_values(args, &['o'], &["output"]);
                if !outputs.is_empty() {
                    self.writes_files(&outputs.iter().collect::<Vec<_>>(), display, "writes sorted output to a file");
                }
            }
            "uniq" => {
                let Some(read) = self.options(args, "-c --count -d --repeated -D -u --unique -i --ignore-case -f= --skip-fields= -s= --skip-chars= -w= --check-chars= -z --zero-terminated", GNU, display) else { return };
                let files: Vec<&Word> = read.operands.iter().skip(1).map(|&at| &args[at]).collect();
                if !files.is_empty() {
                    self.writes_files(&files, display, "writes a file");
                }
            }
            "tree" => {
                self.options(args, "-a -d -l -f -x -L= -P= -I= --gitignore --ignore-case --matchdirs --noreport -i -q -N -Q -p -u -g -s -h --si --du -D -F --inodes -v -t -c -U -r --dirsfirst --filesfirst --sort= -n -C -J --charset= --filelimit= --prune", GNU, display);
            }
            "xxd" | "base64" => {
                let spec = match name {
                    "xxd" => "-a -b -c= -C -e -E -g= -h -i -l= -n= -o= -p -ps -plain -R= -s= -u -d -cols= -len= -seek= -groupsize=",
                    _ => "-d -D --decode -i= --input= -b= --break= -w= --wrap= --ignore-garbage -o= --output=",
                };
                let Some(read) = self.options(args, spec, GNU, display) else { return };
                let mut files: Vec<&Word> = read.operands.iter().skip(1).map(|&at| &args[at]).collect();
                if name == "base64" {
                    files.extend(read.values(&["-o", "--output"]));
                }
                if !files.is_empty() {
                    self.writes_files(&files, display, "writes a file");
                }
            }
            "date" => {
                let Some(read) = self.options(args, "-u --utc --universal -R --rfc-email -j -n -I? --iso-8601? --rfc-3339= -d= --date= -r= --reference= -f= --file= -v= --debug", GNU, display) else { return };
                // Without `-j`, an operand that is not `+format` sets the clock.
                if !read.has(&["-j"]) && read.operands.iter().any(|&at| !args[at].text.starts_with('+')) {
                    ask(self, "can set the system clock");
                }
            }
            "hostname" => {
                let Some(read) = self.options(args, "-s --short -f --fqdn --long -d --domain -i --ip-address -I --all-ip-addresses -a --alias -A --all-fqdns", GNU, display) else { return };
                if !read.operands.is_empty() {
                    ask(self, "changes system settings");
                }
            }
            "scutil" => {
                if !(matches!(text.as_slice(), ["--dns" | "--proxy" | "--nwi"]) || (text.len() == 2 && text[0] == "--get" && !options::expands(&args[1]))) {
                    ask(self, "changes system settings");
                }
            }
            "sysctl" => {
                let Some(read) = self.options(args, "-a -A -b -d -e -h -i -N -n -o -q -x -X --all --binary --values --names --pattern=", GNU, display) else { return };
                if read.operands.iter().any(|&at| args[at].text.contains('=')) {
                    ask(self, "changes system settings");
                }
            }
            "find" => self.find(args, display, resolved),
            "sed" | "gsed" => self.sed(args, display),
            "awk" | "gawk" | "nawk" | "mawk" => {
                let Some(read) = self.options(args, "-F= -v= --field-separator= --assign= --posix --traditional -S --sandbox -b --characters-as-bytes", POSIX, display) else { return };
                let Some(&program) = read.operands.first() else { return };
                // awk's only ways to write files or run commands: `system`,
                // `close`, `fflush`, `getline`, `|`, `>` and gawk's `@` forms.
                let program = &args[program];
                if program.dynamic || ["system", "getline", "close", "fflush", "|", ">", "@"].iter().any(|t| program.text.contains(t)) {
                    ask(self, "awk program can write files or run commands");
                }
            }
            "tee" => {
                let Some(read) = self.options(args, "-a --append -i --ignore-interrupts -p", GNU, display) else { return };
                let files: Vec<&Word> = read.operands.iter().map(|&at| &args[at]).filter(|w| w.text != "/dev/null").collect();
                if !files.is_empty() {
                    self.writes_files(&files, display, "writes a file");
                }
            }
            "git" => self.git(args, display),
            "cargo" => self.cargo(args, display),
            "rustfmt" => {
                let Some(read) = self.options(args, "--check --edition= -v --verbose -q --quiet --color= -l --files-with-diff --version -V --help -h", GNU, display) else { return };
                let prints = read.operands.is_empty() && read.has(&["--version", "-V", "--help", "-h"]);
                if !(read.has(&["--check"]) || prints) {
                    ask(self, "rewrites source files");
                }
            }
            "rustc" | "rustdoc" | "clang" | "gcc" | "cc" | "c++" | "g++" | "swiftc" => {
                // Only words that print and exit; `--print NAME=PATH` writes a file.
                let print = |a: &str| !a.is_empty() && a.bytes().all(|b| b.is_ascii_lowercase() || b == b'-');
                let mut words = args.iter().map(|w| (w.text.as_str(), w.dynamic));
                let mut prints = !args.is_empty();
                while let Some((word, dynamic)) = words.next() {
                    prints &= !dynamic
                        && (matches!(word, "--version" | "-V" | "-vV" | "--help" | "-h" | "-dumpversion" | "-dumpmachine" | "-print-search-dirs")
                            || word.strip_prefix("--print=").is_some_and(print)
                            || (word == "--print" && words.next().is_some_and(|(name, dynamic)| !dynamic && print(name))));
                }
                if !prints {
                    ask(self, "compiles and writes output files");
                }
            }
            "rustup" => {
                let listing = match text.as_slice() {
                    [] | ["--version" | "-V" | "--help" | "help"] => true,
                    ["show", rest @ ..] => rest.iter().all(|a| matches!(*a, "active-toolchain" | "home" | "profile" | "-v" | "--verbose")),
                    ["toolchain" | "target" | "component", "list", rest @ ..] => rest.iter().all(|a| matches!(*a, "--installed" | "-v" | "--verbose")),
                    ["which", program] => options::identifier(&program.replace('-', "_")),
                    _ => false,
                };
                if !listing || args.iter().any(|w| w.dynamic) {
                    ask(self, "installs or changes Rust toolchains");
                }
            }
            "go" => self.go(args, display),
            "npm" | "pnpm" | "yarn" | "bun" => {
                let Some(read) = self.options(args, "--json --depth= --all -a --long -l --parseable -p --global -g --prod --production --dev --omit= --include= -w= --workspace= --workspaces --link --unicode --recursive -r --audit-level= --package-lock-only --version -v --help -h --color= --no-color --silent -s", GNU, display) else { return };
                let words: Vec<&str> = read.operands.iter().map(|&at| text[at]).collect();
                match words.first().copied() {
                    None if !read.options.is_empty() && read.has(&["--version", "-v", "--help", "-h"]) => {}
                    Some("ls" | "list" | "outdated" | "view" | "info" | "why" | "help" | "explain" | "audit") if !words.contains(&"fix") => {}
                    Some("publish" | "unpublish" | "deprecate" | "dist-tag" | "owner" | "login" | "logout" | "adduser") => ask(self, "publishes packages or changes a registry account"),
                    Some("install" | "i" | "add" | "ci" | "update" | "upgrade" | "up" | "remove" | "rm" | "uninstall" | "un" | "link" | "unlink" | "dedupe" | "prune" | "rebuild") | None => ask(self, "installs or removes packages"),
                    Some(_) => ask(self, "runs package scripts"),
                }
            }
            "npx" | "bunx" | "pnpx" | "uvx" | "pipx" => ask(self, "downloads and runs a package"),
            "pip" | "pip3" | "uv" | "poetry" | "conda" | "gem" | "bundle" | "composer" => {
                let read = options::read(args, "--outdated -o --uptodate -u --format= --not-required --local -l --user --path= --exclude-editable --include-editable -e --editable --all -f --files -v --verbose -q --quiet --json --depth= --invert --tree --version -V --help -h --no-color --pre --exclude=", GNU);
                let first = read.operands.first().map(|&at| text[at]);
                let listing = match first {
                    Some(sub) => matches!(sub, "list" | "show" | "freeze" | "check" | "help" | "search" | "info" | "tree"),
                    None => read.has(&["--version", "-V", "--help", "-h"]),
                };
                if read.unread.is_some() || !listing {
                    self.add(Class::Ask, "installs or removes packages", display, read.unread.is_none());
                }
            }
            "brew" | "port" => {
                let read = options::read(args, "--cask --casks --formula --formulae --versions -1 -l --json? --installed --quiet -q --verbose -v --full-name --tree --direct --pinned --desc --installed-on-request --installed-as-dependency --include-build --include-test --recursive --topological --prefix --repository --cellar --version --help -h", GNU);
                let first = read.operands.first().map(|&at| text[at]);
                let listing = match first {
                    Some(sub) => matches!(sub, "list" | "ls" | "info" | "search" | "config" | "doctor" | "deps" | "uses" | "leaves" | "outdated" | "help") || (text.first() == Some(&"--prefix") && read.operands.len() == 1),
                    None => !read.options.is_empty(),
                };
                if read.unread.is_some() || !listing {
                    self.add(Class::Ask, "installs, removes or updates system packages", display, read.unread.is_none());
                }
            }
            "apt" | "apt-get" | "apt-cache" | "dpkg" | "yum" | "dnf" | "pacman" | "zypper" | "apk" | "snap" | "flatpak" | "softwareupdate" | "installer" | "mas" => {
                let read = options::read(args, "--installed --upgradable --upgradeable -a --all-versions --names-only --full -l -L --list --version -q", GNU);
                let listing = match read.operands.first().map(|&at| text[at]) {
                    Some("list" | "search" | "show" | "policy" | "info") => true,
                    _ => read.has(&["-l", "-L", "--list", "--version"]),
                };
                if read.unread.is_some() || !listing {
                    self.add(Class::Ask, "installs, removes or updates system packages", display, read.unread.is_none());
                }
            }
            "rm" | "rmdir" | "unlink" | "shred" | "srm" | "trash" => {
                let recursive = name == "rm" && text.iter().any(|a| *a == "--recursive" || short_flag(a, 'r') || short_flag(a, 'R'));
                if text.contains(&"--no-preserve-root") {
                    self.add(Class::Blocked, "deletes the whole file system (--no-preserve-root)", display, false);
                }
                for target in &positional {
                    if recursive && catastrophic(&target.text) {
                        self.add(Class::Blocked, format!("deletes a system or home folder: rm -r {}", target.text), display, false);
                    }
                }
                self.writes_files(&positional, display, "deletes files");
            }
            "xattr" => {
                self.options(args, "-l -p -x -r -s -v", GNU, display);
            }
            "chmod" | "chown" | "chgrp" | "chflags" | "setfacl" | "chattr" => {
                let recursive = text.iter().any(|a| *a == "--recursive" || short_flag(a, 'R'));
                let targets = if matches!(name, "chmod" | "chown" | "chgrp" | "chflags") && !positional.is_empty() { &positional[1..] } else { &positional[..] };
                // Every word, so a mode such as `-x` cannot hide the target.
                for target in args {
                    if recursive && catastrophic(&target.text) {
                        self.add(Class::Blocked, format!("changes permissions of a system or home folder: {name} -R {}", target.text), display, false);
                    }
                }
                self.writes_files(targets, display, "changes file permissions or owner");
            }
            "mv" => {
                for target in &positional {
                    if catastrophic(&target.text) {
                        self.add(Class::Blocked, format!("moves a system or home folder: {}", target.text), display, false);
                    }
                }
                self.writes_files(&positional, display, "moves files");
            }
            "cp" | "ln" | "install" | "rsync" | "ditto" | "scp" => {
                if name == "scp" || (name == "rsync" && text.iter().any(|a| a.contains(':'))) {
                    ask(self, "copies files to or from another machine");
                }
                self.writes_files(&positional[positional.len().saturating_sub(1)..], display, "copies or links files");
            }
            "mkdir" | "touch" | "truncate" | "patch" | "split" | "csplit" | "mkfifo" | "mktemp" => {
                self.writes_files(&positional, display, "creates or changes files");
            }
            "tar" | "unzip" | "zip" | "gzip" | "gunzip" | "bzip2" | "bunzip2" | "xz" | "unxz" | "zstd" | "7z" | "ar" => {
                // Listing only: `tar -tf` (also `tar tzf`) and `unzip -l`.
                let listed = match name {
                    "tar" => {
                        let mut words = args.to_vec();
                        if let Some(first) = words.first_mut().filter(|w| !w.text.starts_with('-') && !w.dynamic) {
                            first.text.insert(0, '-');
                        }
                        let read = options::read(&words, "-t --list -v --verbose -z --gzip -j --bzip2 -J --xz -f= --file= -O --to-stdout", GNU);
                        read.unread.is_none() && read.has(&["-t", "--list"])
                    }
                    "unzip" => {
                        let read = options::read(args, "-l -v -t -Z -z -p -c -q -qq -1 -2 -s -m -h -T", GNU);
                        read.unread.is_none() && read.has(&["-l", "-v", "-t", "-Z", "-z", "-p", "-c"])
                    }
                    _ => false,
                };
                if !listed {
                    self.writes_files(&positional, display, "creates or unpacks archives");
                }
            }
            "dd" => {
                for arg in &text {
                    if let Some(target) = arg.strip_prefix("of=") {
                        if raw_disk(target) {
                            self.add(Class::Blocked, format!("writes over a disk device: {target}"), display, false);
                        } else if self.outside(target) {
                            ask(self, &format!("writes outside the workspace: {target}"));
                        }
                    }
                }
                ask(self, "copies raw data with dd");
            }
            _ if name.starts_with("mkfs") || name.starts_with("newfs") || matches!(name, "fdisk" | "sfdisk" | "parted" | "gpt" | "wipefs" | "mkswap" | "fsck" | "format") => {
                ask(self, "formats or partitions disks");
            }
            "diskutil" | "hdiutil" | "mount" | "umount" => {
                let read = options::read(args, "-plist -all", GNU);
                let listing = match name {
                    "mount" => args.is_empty(),
                    "umount" => false,
                    _ => read.unread.is_none() && read.operands.first().is_some_and(|&at| matches!(text[at], "list" | "info" | "activity")),
                };
                if !listing {
                    ask(self, "changes disks or mounted volumes");
                }
            }
            "kill" | "killall" | "pkill" | "launchctl" | "systemctl" | "service" | "shutdown" | "reboot" | "halt" | "poweroff" => {
                if name == "kill" && text.iter().all(|a| *a == "-l" || *a == "-L") && !text.is_empty() {
                    return;
                }
                ask(self, if matches!(name, "kill" | "killall" | "pkill") { "stops processes" } else { "changes system services or power state" });
            }
            "crontab" | "defaults" | "pmset" | "nvram" | "csrutil" | "spctl" | "networksetup" | "ifconfig" | "route" | "dscl" | "systemsetup" | "tmutil" | "sysadminctl" => {
                let plain = |w: &Word| !w.dynamic && !w.glob && !w.text.is_empty() && w.text.bytes().all(|b| b.is_ascii_alphanumeric());
                let read = match name {
                    "crontab" => text == ["-l"],
                    "defaults" => {
                        let read = options::read(args, "-currentHost -g -app=", GNU);
                        read.unread.is_none() && read.operands.first().is_some_and(|&at| matches!(text[at], "read" | "domains" | "read-type" | "find"))
                    }
                    "ifconfig" => {
                        let read = options::read(args, "-a -l -v -L -m", GNU);
                        read.unread.is_none() && read.operands.len() <= 1 && read.operands.iter().all(|&at| plain(&args[at]))
                    }
                    "pmset" => text.first() == Some(&"-g") && args.len() <= 2 && args[1..].iter().all(plain),
                    _ => false,
                };
                if !read {
                    ask(self, "changes system settings");
                }
            }
            "curl" | "wget" | "fetch" | "http" | "aria2c" => {
                ask(self, "uses the network");
                let short: &[char] = if name == "wget" { &['o', 'O'] } else { &['o'] };
                let outputs = output_values(args, short, &["output", "output-document"]);
                let outputs: Vec<&Word> = outputs.iter().collect();
                let remote_name = name == "curl" && text.iter().any(|a| matches!(*a, "-O" | "--remote-name" | "--remote-name-all") || short_flag(a, 'O'));
                if !outputs.is_empty() || remote_name || (name == "wget" && !text.iter().any(|a| *a == "-O-" || *a == "-qO-")) {
                    self.writes_files(&outputs, display, "downloads files");
                }
            }
            "ssh" | "sftp" | "telnet" | "nc" | "ncat" | "netcat" | "ftp" | "socat" | "mosh" => {
                ask(self, "connects to another machine");
            }
            // Digests and certificate details, without output files or keys.
            "openssl" if {
                let spec = match text.first().copied() {
                    Some("version") => Some("-a -v -b -o -f -p -d -e -m -r -c"),
                    Some("dgst" | "sha1" | "sha256" | "sha512" | "md5") => Some("-sha1 -sha224 -sha256 -sha384 -sha512 -md5 -sha3-256 -sha3-512 -r -hex -c"),
                    Some("x509") => Some("-in= -inform= -noout -text -subject -issuer -dates -startdate -enddate -fingerprint -serial -hash -subject_hash -issuer_hash -purpose -pubkey -modulus -email -ocsp_uri -ext= -nameopt= -certopt= -sha1 -sha256 -md5 -alias -checkend="),
                    _ => None,
                };
                spec.is_some_and(|spec| options::read(&args[1..], spec, WHOLE_ANY).unread.is_none())
            } => {}
            "security" | "ssh-add" | "ssh-keygen" | "ssh-agent" | "ssh-copy-id" | "gpg" | "gpg2" | "pass" | "op" | "keychain" | "certutil" | "openssl" => {
                self.add(Class::Ask, "uses keys, passwords or other credentials", display, false);
            }
            "open" | "xdg-open" => ask(self, "opens a file or app outside the terminal"),
            _ => ask(self, &format!("{name} is not on the list of read-only and build commands")),
        }
    }

    /// Read `args` against `spec`. A word that is not read asks, with no
    /// rule, and gives None.
    fn options(&mut self, args: &[Word], spec: &'static str, mode: Mode, display: &[String]) -> Option<Read> {
        let read = options::read(args, spec, mode);
        // A value attached to a short option (`-fid_rsa`) is part of a word
        // that the command's own check does not split.
        for value in read.options.iter().filter_map(|(_, value)| value.as_ref()) {
            self.credentials(value, display);
        }
        match read.unread {
            Some(at) => {
                self.unread(&args[at], display);
                None
            }
            None => Some(read),
        }
    }
    fn unread(&mut self, word: &Word, display: &[String]) {
        self.add(Class::Ask, format!("has an option or a word that the checker does not read: {}", word.text), display, false);
    }

    fn wrapper(&mut self, name: &str, args: &[Word], display: &[String], resolved: &mut Resolved) {
        let spec = match name {
            "nice" => "-n= --adjustment= -#",
            // macOS `arch -arm64 prog`: runs prog; no arguments prints the architecture.
            "arch" => "-arch= -d= -e= -c -h -arm64 -arm64e -x86_64 -x86_64h -i386",
            "time" => "-a -h -l -p -q -v -o= -f= --output= --format= --append --verbose --quiet --portability",
            "exec" => "-a= -c -l",
            "stdbuf" => "-i= -o= -e= --input= --output= --error=",
            "caffeinate" => "-d -i -m -s -u -t= -w=",
            "timeout" | "gtimeout" => "-s= -k= -v -f -p --signal= --kill-after= --verbose --foreground --preserve-status",
            // GNU `--replace` and `--eof` take a value only as `--replace=R`.
            "xargs" => "-0 -o -p -r -t -x -E= -I= -J= -L= -n= -P= -R= -S= -s= -a= -d= --null --interactive --verbose --exit --no-run-if-empty --open-tty --show-limits --arg-file= --delimiter= --max-args= --max-procs= --max-lines= --max-chars= --process-slot-var= --replace? --eof?",
            "command" => "-p -v -V",
            "sudo" | "doas" | "runas" | "pkexec" => "-A -b -E -H -i -K -k -l -n -P -S -s -V -v -u= -g= -C= -p= -h= -U= -r= -t= -D= -R= -T= --user= --group= --prompt= --chdir= --host= --close-from= --role= --type= --chroot= --command-timeout= --other-user= --preserve-env? --login --non-interactive --stdin --shell --background --set-home --preserve-groups --askpass",
            _ => "",
        };
        if matches!(name, "sudo" | "doas" | "su" | "runas" | "pkexec") {
            self.add(Class::Ask, "runs as another user (root)", display, false);
            if name == "su" {
                return;
            }
        }
        let read = options::read_head(args, spec, if name == "arch" { WHOLE } else { POSIX });
        if let Some(at) = read.unread {
            return self.unread_wrapper(args, at, display, resolved);
        }
        for value in read.values(&["-e"]).filter(|_| name == "arch") {
            // `arch -e NAME=value` sets a variable for the program.
            self.assignment_word(value, display);
        }
        if name == "command" && read.has(&["-v", "-V"]) {
            // `command -v name` only looks the name up.
            return;
        }
        let outputs: Vec<&Word> = read.values(&["-o", "--output"]).filter(|_| name == "time").collect();
        if !outputs.is_empty() {
            self.writes_files(&outputs, display, "writes a file");
        }
        // The duration comes before the command.
        let rest = read.operands.get(usize::from(matches!(name, "timeout" | "gtimeout"))).copied().unwrap_or(args.len());
        if rest >= args.len() {
            if name == "exec" && !args.is_empty() {
                self.add(Class::Ask, "changes the shell's open files", display, true);
            }
            return;
        }
        if name != "xargs" {
            return self.command(&args[rest..], display, resolved);
        }
        // xargs adds words from its input, or puts them where the replace
        // string is: those words can be anything, options too.
        let mut words = args[rest..].to_vec();
        let replace = read.options.iter().rev().find(|(option, _)| matches!(*option, "-I" | "-J" | "--replace")).map(|(_, value)| value.as_ref().map_or("{}".to_owned(), |v| v.text.clone()));
        match replace.filter(|r| !r.is_empty()) {
            Some(replace) => {
                for word in words.iter_mut().filter(|w| w.text.contains(&replace)) {
                    word.text = word.text.replace(&replace, "$XARGS");
                    word.dynamic = true;
                }
            }
            None => words.push(Word { text: "$XARGS".into(), dynamic: true, ..Word::default() }),
        }
        self.command(&words, display, resolved);
    }

    /// A wrapper option at `at` is not read, so the command can start at
    /// any later word: more options, values and a `timeout` duration may
    /// come first. Check the command from each word that is not an
    /// option, so a blocked command stays blocked.
    fn unread_wrapper(&mut self, args: &[Word], at: usize, display: &[String], resolved: &mut Resolved) {
        self.unread(&args[at], display);
        for start in at + 1..args.len() {
            if !args[start].text.starts_with('-') {
                self.command(&args[start..], display, resolved);
            }
        }
    }

    fn env(&mut self, args: &[Word], display: &[String], resolved: &mut Resolved) {
        let read = options::read_head(args, "-i -0 -v --ignore-environment --null --debug -u= --unset= -C= --chdir= -S= --split-string= -P=", POSIX);
        // `-S` splits one string into a whole command line, also in a
        // cluster (`-iS line`) and when the options are not read.
        for (at, word) in args.iter().enumerate() {
            let text = word.text.as_str();
            let rest = text.strip_prefix("--split-string").filter(|r| r.is_empty() || r.starts_with('=')).map(|r| r.trim_start_matches('='))
                .or_else(|| text.strip_prefix('-').filter(|l| !l.starts_with('-')).and_then(|l| l.find('S').map(|s| &l[s + 1..])));
            let Some(rest) = rest else { continue };
            self.add(Class::Ask, "runs a command line split by env -S", display, false);
            let line = if rest.is_empty() { args.get(at + 1).cloned() } else { Some(Word { text: rest.to_owned(), ..word.clone() }) };
            self.code(line.as_ref(), display);
        }
        if let Some(at) = read.unread {
            return self.unread_wrapper(args, at, display, resolved);
        }
        if read.has(&["-P"]) {
            self.add(Class::Ask, "looks for the program in another folder (env -P)", display, true);
        }
        let mut rest = read.operands.first().copied().unwrap_or(args.len());
        // `env -` is `env -i`.
        rest += usize::from(args.get(rest).is_some_and(|w| w.text == "-"));
        while let Some(word) = args.get(rest).filter(|w| assignment(&w.text)) {
            self.assignment_word(word, display);
            rest += 1;
        }
        if rest < args.len() {
            self.command(&args[rest..], display, resolved);
        }
    }

    /// `set`: shell options from a closed list; the words after them are
    /// the arguments. zsh `set -A name` and bash `set -k` are not read.
    fn set(&mut self, args: &[Word], display: &[String]) {
        const NAMES: &[&str] = &["errexit", "nounset", "pipefail", "xtrace", "verbose", "noglob", "noclobber", "noexec"];
        let mut at = 0;
        while let Some(word) = args.get(at) {
            at += 1;
            if word.text == "--" && !word.dynamic {
                return;
            }
            if options::expands(word) {
                return self.unread(word, display);
            }
            let Some(letters) = word.text.strip_prefix(['-', '+']).filter(|l| !l.is_empty()) else { return };
            for (index, letter) in letters.char_indices() {
                match letter {
                    'e' | 'u' | 'x' | 'v' | 'f' | 'C' | 'n' => {}
                    'o' if index + 1 == letters.len() => {
                        if args.get(at).is_some_and(|name| name.dynamic || !NAMES.contains(&name.text.as_str())) {
                            return self.unread(&args[at], display);
                        }
                        at += 1;
                    }
                    _ => return self.unread(word, display),
                }
            }
        }
    }

    /// `printf`: a format with only plain directives. `-v` sets a variable
    /// that a word names, `%n` stores a count in one, and zsh reads the
    /// value of a number directive as arithmetic.
    fn printf(&mut self, args: &[Word], display: &[String]) {
        let at = usize::from(args.first().is_some_and(|w| w.text == "--" && !w.dynamic));
        let Some(format) = args.get(at) else { return };
        let numbers = match printf_format(&format.text) {
            Some(numbers) if !format.dynamic && !(at == 0 && format.text.starts_with('-')) => numbers,
            plain => {
                match plain {
                    Some(_) => self.unread(format, display),
                    None => self.add(Class::Ask, format!("uses a printf format that the checker does not read: {}", format.text), display, false),
                }
                // A name for `-v` or `%n` with code in it stays blocked.
                for word in args {
                    self.stored_code(word, display);
                }
                return;
            }
        };
        let number = |w: &Word| {
            let digits = w.text.strip_prefix(['-', '+']).unwrap_or(&w.text);
            !w.dynamic && !digits.is_empty() && digits.matches('.').count() <= 1 && digits.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        };
        if numbers && !args[at + 1..].iter().all(number) {
            self.add(Class::Ask, "printf reads a value as a number, which zsh reads as arithmetic", display, false);
            args.iter().for_each(|word| self.stored_code(word, display));
        }
    }

    /// `read`, `mapfile`, `getopts` and `unset` set or clear the variables
    /// that their words name. Only plain names are read; each one set now
    /// holds text, so arithmetic on it needs approval.
    fn names(&mut self, name: &str, args: &[Word], display: &[String]) {
        let (spec, mode) = match name {
            "read" => ("-r", GNU),
            "mapfile" | "readarray" => ("-t", GNU),
            "unset" => ("-v -f", GNU),
            _ => ("", POSIX),
        };
        let Some(read) = self.options(args, spec, mode, display) else {
            // A name with code in it stays blocked.
            return args.iter().for_each(|word| self.stored_code(word, display));
        };
        let mut names: Vec<&Word> = read.operands.iter().map(|&at| &args[at]).collect();
        if name == "getopts" {
            names = names.into_iter().skip(1).take(1).collect();
        }
        for word in &names {
            if !options::identifier(&word.text) {
                self.add(Class::Ask, format!("names a variable that the checker cannot read: {}", word.text), display, false);
                self.stored_code(word, display);
            } else if risky_variable(&word.text) {
                self.add(Class::Ask, "sets a variable that can change what runs", display, false);
            }
        }
        let mut set: Vec<&str> = names.iter().map(|w| w.text.as_str()).collect();
        match name {
            "read" if set.is_empty() => set.push("REPLY"),
            "mapfile" | "readarray" if set.is_empty() => set.push("MAPFILE"),
            "getopts" => set.push("OPTARG"),
            "unset" => set.clear(),
            _ => {}
        }
        for name in set {
            self.assigns(&Word { text: name.to_owned(), ..Word::default() });
        }
    }

    fn fd(&mut self, args: &[Word], display: &[String], resolved: &mut Resolved) {
        // `-x cmd ;` and `-X cmd ;` run a command on each result, also in a
        // cluster or with the first word attached (`-HIx cmd ;`, `-xcmd ;`,
        // `--exec=cmd ;`).
        let mut own = Vec::new();
        let mut index = 0;
        while let Some(word) = args.get(index) {
            let text = word.text.as_str();
            let long = ["--exec", "--exec-batch"].iter().find_map(|name| text.strip_prefix(name).filter(|r| r.is_empty() || r.starts_with('=')));
            let short = text.strip_prefix('-').filter(|l| !l.starts_with('-')).and_then(|l| l.find(['x', 'X']).map(|at| (&l[..at], &l[at + 1..])));
            let (flags, attached) = match (long, short) {
                (Some(rest), _) => ("", rest.strip_prefix('=').unwrap_or(rest)),
                (None, Some(split)) => split,
                (None, None) => {
                    own.push(word.clone());
                    index += 1;
                    continue;
                }
            };
            if !flags.is_empty() {
                own.push(Word { text: format!("-{flags}"), ..word.clone() });
            }
            let end = args[index + 1..].iter().position(|w| w.text == ";").map_or(args.len(), |end| index + 1 + end);
            self.add(Class::Ask, "fd runs a command on each result", display, true);
            let mut command: Vec<Word> = Vec::new();
            if !attached.is_empty() {
                command.push(Word { text: attached.to_owned(), ..word.clone() });
            }
            command.extend_from_slice(&args[index + 1..end]);
            if !command.is_empty() {
                // Without a `{}` slot, fd puts the path at the end.
                const SLOTS: &[&str] = &["{//}", "{/.}", "{/}", "{.}", "{}"];
                if !command.iter().any(|w| SLOTS.iter().any(|slot| w.text.contains(slot))) {
                    command.push(Word { text: "{}".into(), ..Word::default() });
                }
                let command: Vec<Word> = command.iter().map(|w| file_slots(w, SLOTS)).collect();
                self.command(&command, display, resolved);
            }
            index = end + 1;
        }
        self.options(&own, options::FD, GNU, display);
    }

    fn shell(&mut self, name: &str, args: &[Word], display: &[String], resolved: &mut Resolved) {
        resolved.shell = true;
        // The checker reads sh, bash, dash and zsh. The code of other shells
        // asks with no rule, and is still checked for blocked commands.
        if !matches!(name, "sh" | "bash" | "dash" | "zsh") {
            self.add(Class::Ask, format!("runs {name} code, which the checker does not read"), display, false);
        }
        let mut index = 0;
        let mut wants_script = false;
        // The shell runs the value of fish `-c` or `--command` itself.
        let mut fish_command = false;
        // Shell options from a closed list; fish `-o` and `-p` write files.
        let letters_read = if name == "fish" { "cCdfilnNPvh" } else { "cefuxvnlsiCEhPBoO" };
        const LONG: &[&str] = &["--login", "--norc", "--noprofile", "--posix", "--noediting", "--restricted", "--verbose", "--version", "--help", "--no-rcs", "--no-globalrcs", "--private", "--no-config", "--interactive", "--no-execute"];
        const NAMES: &[&str] = &["errexit", "nounset", "pipefail", "xtrace", "verbose", "noglob", "noclobber", "noexec", "posix", "errtrace", "extglob", "globstar", "nullglob", "dotglob", "failglob", "nocaseglob", "lastpipe", "inherit_errexit"];
        while let Some(arg) = args.get(index) {
            let flag = arg.text.as_str();
            if flag == "--" || flag == "-" {
                index += 1;
                break;
            }
            if options::expands(arg) {
                self.unread(arg, display);
                break;
            }
            if matches!(flag, "--rcfile" | "--init-file" | "--init-command") || (name == "fish" && flag.starts_with("--init-command")) {
                self.add(Class::Ask, "runs a shell start-up file or command", display, false);
                let value = match flag.split_once('=') {
                    Some((_, value)) => Some(Word { text: value.to_owned(), ..arg.clone() }),
                    None => args.get(index + 1).cloned(),
                };
                // fish runs the `--init-command` code first, as with `-C`.
                if flag.starts_with("--init-command") {
                    self.code(value.as_ref(), display);
                }
                index += if flag.contains('=') { 1 } else { 2 };
                continue;
            }
            // `--command=code` and `--command code`: fish runs every one, with
            // more options after it, so the scan goes on.
            if flag == "--command" || flag.starts_with("--command=") {
                let code = match flag.strip_prefix("--command=") {
                    Some(code) => Some(Word { text: code.to_owned(), ..arg.clone() }),
                    None => {
                        index += 1;
                        args.get(index).cloned()
                    }
                };
                self.code(code.as_ref(), display);
                fish_command = true;
                index += 1;
                continue;
            }
            if flag.starts_with("--") {
                if !LONG.contains(&flag) {
                    self.unread(arg, display);
                }
                index += 1;
                continue;
            }
            if (flag.starts_with('-') || flag.starts_with('+')) && flag.len() > 1 {
                // `-o name` and `-O name` take a value, also inside a cluster
                // (`-eo pipefail`); in fish so do `-c`, `-C` (elsewhere
                // noclobber) and the debug and feature options. zsh and fish
                // read a value right after the letter (`-oerrexit`); other
                // shells take the next word.
                let letters: Vec<char> = flag[1..].chars().collect();
                let attach = matches!(name, "zsh" | "fish");
                let mut taken = 0;
                for (at, &letter) in letters.iter().enumerate() {
                    if !letters_read.contains(letter) {
                        self.unread(arg, display);
                    }
                    let fish_value = name == "fish" && matches!(letter, 'c' | 'C' | 'd' | 'f');
                    if letter == 'c' && !fish_value && flag.starts_with('-') {
                        wants_script = true;
                    }
                    // zsh `-O` is an option of its own (correct_all), not `-O name`.
                    let named = name != "fish" && (letter == 'o' || (letter == 'O' && name != "zsh"));
                    if !fish_value && !named {
                        continue;
                    }
                    let attached: String = letters[at + 1..].iter().collect();
                    let value = if attach && !attached.is_empty() {
                        Some(Word { text: attached, ..arg.clone() })
                    } else {
                        taken += 1;
                        args.get(index + taken).cloned()
                    };
                    if named && value.as_ref().is_some_and(|v| v.dynamic || !NAMES.contains(&v.text.as_str())) {
                        self.unread(value.as_ref().unwrap_or(arg), display);
                    }
                    if fish_value && matches!(letter, 'c' | 'C') {
                        if letter == 'C' {
                            self.add(Class::Ask, "runs a shell start-up file or command", display, false);
                        }
                        fish_command |= letter == 'c';
                        self.code(value.as_ref(), display);
                    }
                    if attach && at + 1 < letters.len() {
                        break;
                    }
                }
                index += 1 + taken;
                continue;
            }
            break;
        }
        if fish_command {
            return;
        }
        let ask = |ctx: &mut Self, reason: &str| ctx.add(Class::Ask, reason, display, true);
        match (wants_script, args.get(index)) {
            (true, Some(script)) => self.code(Some(script), display),
            (true, None) => ask(self, "runs a shell with no script"),
            (false, Some(file)) => ask(self, &format!("runs a shell script file: {}", file.text)),
            (false, None) => resolved.stdin_script = true,
        }
    }

    /// Shell code to run. Code with an expansion asks, and its text is
    /// still checked as written, so a blocked command in it stays blocked.
    fn code(&mut self, code: Option<&Word>, display: &[String]) {
        if code.is_none_or(|code| code.dynamic) {
            self.add(Class::Ask, "runs shell code built from a variable", display, false);
        }
        if let Some(code) = code {
            self.script(&code.text);
        }
    }

    fn find(&mut self, args: &[Word], display: &[String], resolved: &mut Resolved) {
        // Leading options (`-H`, `-L`, `-P`, `-E`, `-X`, `-d`, `-s`, `-x`,
        // `-O3`, `-D list`, `-f root`) come before the start folders. BSD find
        // also takes `-froot` and `-Lf root`.
        // The first word that is not read. The scan goes on past it, so a
        // later `-delete` or `-exec` still counts.
        let mut unread: Option<Word> = None;
        let mut index = 0;
        let mut roots: Vec<Word> = Vec::new();
        while let Some(arg) = args.get(index) {
            let flag = arg.text.as_str();
            let cluster = flag.strip_prefix('-').filter(|f| !f.starts_with('-')).map(|f| f.trim_start_matches(|c| "HLPEXdsx".contains(c)));
            match flag {
                _ if cluster.is_some_and(|rest| rest.starts_with('f')) => {
                    let attached = &cluster.unwrap_or_default()[1..];
                    if attached.is_empty() {
                        roots.extend(args.get(index + 1).cloned());
                        index += 2;
                    } else {
                        roots.push(Word { text: attached.to_owned(), ..arg.clone() });
                        index += 1;
                    }
                }
                // Read below as a start folder or a test.
                _ if options::expands(arg) => break,
                "-D" => {
                    if !args.get(index + 1).is_some_and(options::value_word) {
                        unread = unread.or_else(|| args.get(index + 1).cloned());
                    }
                    index += 2;
                }
                "--" => {
                    index += 1;
                    break;
                }
                _ if flag.len() > 1 && flag.starts_with('-') && (flag[1..].chars().all(|c| "HLPEXdsx".contains(c)) || (flag.starts_with("-O") && flag[2..].bytes().all(|b| b.is_ascii_digit()))) => index += 1,
                _ => break,
            }
        }
        let index = index.min(args.len());
        let named = args[index..].iter().take_while(|w| !w.text.starts_with('-') && w.text != "(" && w.text != "!").count();
        roots.extend(args[index..index + named].iter().cloned());
        for root in &roots {
            // `-froot` is one word, so the command's own check missed it.
            self.credentials(root, display);
        }
        if let Some(root) = roots.iter().find(|root| options::expands(root)) {
            unread = unread.or_else(|| Some(root.clone()));
        }
        // The expression: each test and action must be on the closed lists.
        let mut index = index + named;
        while let Some(word) = args.get(index) {
            let arg = word.text.as_str();
            match arg {
                _ if options::expands(word) => unread = unread.or_else(|| Some(word.clone())),
                "-delete" => {
                    for root in &roots {
                        if catastrophic(&root.text) {
                            self.add(Class::Blocked, format!("deletes a system or home folder: find {} -delete", root.text), display, false);
                        }
                    }
                    self.writes_files(&roots.iter().collect::<Vec<_>>(), display, "deletes files");
                }
                "-exec" | "-execdir" | "-ok" | "-okdir" => {
                    // `+` ends the command only right after `{}`; any other
                    // `+` is an argument.
                    let end = (index + 1..args.len()).find(|&at| args[at].text == ";" || (args[at].text == "+" && at > index + 1 && args[at - 1].text == "{}")).unwrap_or(args.len());
                    if index + 1 < end {
                        let command: Vec<Word> = args[index + 1..end].iter().map(|w| file_slots(w, &["{}"])).collect();
                        self.command(&command, display, resolved);
                    }
                    index = end;
                }
                "-fprint" | "-fprint0" | "-fls" | "-fprintf" => {
                    let file: Vec<&Word> = args.get(index + 1).into_iter().collect();
                    self.writes_files(&file, display, "writes a file");
                    index += 1 + usize::from(arg == "-fprintf");
                }
                _ if options::FIND_FLAGS.split_ascii_whitespace().any(|f| f == arg) => {}
                _ if options::FIND_VALUES.split_ascii_whitespace().any(|f| f == arg) || arg.strip_prefix("-newer").is_some_and(|xy| xy.len() == 2) => {
                    if !args.get(index + 1).is_some_and(|value| !options::expands(value)) {
                        unread = unread.or_else(|| Some(word.clone()));
                    }
                    index += 1;
                }
                _ => unread = unread.or_else(|| Some(word.clone())),
            }
            index += 1;
        }
        if let Some(word) = unread {
            self.unread(&word, display);
        }
    }

    fn sed(&mut self, args: &[Word], display: &[String]) {
        let ask = |ctx: &mut Self, reason: &str| ctx.add(Class::Ask, reason, display, true);
        // A short option cluster with `letter`, before any `-e` value in it.
        let short = |w: &Word, letters: &str| w.text.starts_with('-') && !w.text.starts_with("--") && w.text[1..].split('e').next().unwrap_or("").contains(|c| letters.contains(c));
        if args.iter().any(|w| w.text.starts_with("--in-place") || short(w, "iI")) {
            return ask(self, "edits files in place");
        }
        if args.iter().any(|w| w.text.starts_with("--file") || short(w, "f")) {
            return ask(self, "runs a sed script file");
        }
        let Some(read) = self.options(args, "-n --quiet --silent -E -r --regexp-extended -e= --expression= -s --separate -u --unbuffered -z --null-data --zero-terminated --posix --debug --sandbox --help --version", GNU, display) else { return };
        let mut scripts: Vec<&Word> = read.values(&["-e", "--expression"]).collect();
        if scripts.is_empty() {
            scripts.extend(read.operands.first().map(|&at| &args[at]));
        }
        if scripts.iter().any(|script| script.dynamic || !sed_read_only(&script.text)) {
            ask(self, "sed program can write files or run commands");
        }
    }

    fn git(&mut self, args: &[Word], display: &[String]) {
        let mut index = 0;
        while index < args.len() {
            let arg = args[index].text.as_str();
            match arg {
                _ if options::expands(&args[index]) => return self.unread(&args[index], display),
                // A value that can split into words (`-C $x`) moves the command.
                "-C" | "--git-dir" | "--work-tree" | "--namespace" | "-c" if !args.get(index + 1).is_some_and(options::value_word) => {
                    return self.unread(args.get(index + 1).unwrap_or(&args[index]), display);
                }
                "-C" | "--git-dir" | "--work-tree" | "--namespace" => index += 2,
                "-c" => {
                    let key = args.get(index + 1).map_or("", |w| w.text.split('=').next().unwrap_or(""));
                    if !(key.starts_with("color.") || matches!(key, "core.quotepath" | "log.showsignature" | "advice.detachedHead")) {
                        return self.add(Class::Ask, "sets git configuration that can run programs", display, false);
                    }
                    index += 2;
                }
                // `--config-env` sets configuration; `--exec-path=dir` picks the programs git runs.
                _ if arg.starts_with("--config-env") || arg.starts_with("--exec-path=") => {
                    return self.add(Class::Ask, "sets git configuration that can run programs", display, false);
                }
                _ if matches!(arg, "--no-pager" | "-P" | "-p" | "--paginate" | "--bare" | "--no-replace-objects" | "--literal-pathspecs" | "--glob-pathspecs" | "--noglob-pathspecs" | "--icase-pathspecs" | "--no-optional-locks" | "--version" | "--help" | "-h" | "--exec-path" | "--html-path" | "--man-path" | "--info-path")
                    || ["--git-dir=", "--work-tree=", "--namespace="].iter().any(|p| arg.starts_with(p)) =>
                {
                    index += 1
                }
                // Any other option (`-cname=value` too) is not read, so it asks.
                _ if arg.starts_with('-') => return self.add(Class::Ask, format!("uses a git option the checker does not read: {arg}"), display, false),
                _ => break,
            }
        }
        let Some(sub) = args.get(index) else { return };
        let rest = &args[index + 1..];
        let words: Vec<&str> = rest.iter().map(|w| w.text.as_str()).collect();
        let has = |flags: &[&str]| words.iter().any(|a| flags.contains(a));
        let short = |letters: &str| words.iter().any(|a| a.starts_with('-') && !a.starts_with("--") && a[1..].chars().any(|c| letters.contains(c)));
        let listing = |spec: &'static str| options::read(rest, spec, GNU);
        if let Some(spec) = options::git_read(&sub.text) {
            self.options(rest, spec, GNU, display);
            return;
        }
        match sub.text.as_str() {
            "help" => {
                self.options(rest, "-a --all -g --guides -c --config -m --man --no-external-commands --no-aliases --verbose", GNU, display);
            }
            "branch" | "tag" => {
                let read = listing(options::GIT_LIST);
                let list = read.operands.is_empty() || read.has(&["-l", "--list", "--contains", "--no-contains", "--merged", "--no-merged", "--points-at"]);
                if read.unread.is_some() || !list {
                    let change = match sub.text.as_str() {
                        "branch" => "-d -D --delete -m -M --move -c -C --copy -f --force -u= --set-upstream-to= --unset-upstream -t --track? --no-track -r --remotes -q --quiet",
                        _ => "-a --annotate -m= --message= -F= --file= -d --delete -f --force",
                    };
                    let reason = if sub.text == "branch" && (short("D") || ((short("d") || has(&["--delete"])) && (short("f") || has(&["--force"])))) {
                        "deletes a branch, even if not merged"
                    } else if sub.text == "branch" {
                        "creates, renames or deletes branches"
                    } else {
                        "creates or deletes tags"
                    };
                    self.git_change(rest, Some(change), reason, display);
                }
            }
            "remote" => {
                let read = matches!(words.as_slice(), [] | ["-v" | "--verbose"]) || (words.first() == Some(&"get-url") && rest.iter().all(|w| !options::expands(w)) && words[1..].iter().all(|a| !a.starts_with('-') || matches!(*a, "--push" | "--all")));
                if !read {
                    self.git_change(rest, Some("-v --verbose -f --tags --no-tags -t= -m= --push --add --delete --dry-run -n --all --auto -a -d"), "changes git remotes or contacts one", display);
                }
            }
            "stash" if matches!(words.first(), Some(&"list" | &"show")) => {
                self.options(&rest[1..], if words[0] == "list" { options::GIT_LOG } else { options::GIT_DIFF }, GNU, display);
            }
            "stash" => self.git_change(rest, Some("-u --include-untracked -a --all -k --keep-index --no-keep-index -m= --message= -q --quiet --index -S --staged"), "changes the stash and the working tree", display),
            "config" => {
                let read = listing(options::GIT_CONFIG);
                let get = read.has(&["--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l"]) || read.operands.first().is_some_and(|&at| matches!(words[at], "get" | "list"));
                if read.unread.is_some() || !get {
                    // A setting can make git run a program later.
                    self.add(Class::Ask, "changes git configuration, which can make git run programs", display, false);
                }
            }
            "worktree" if words.first() == Some(&"list") => {
                self.options(&rest[1..], "--porcelain -v --verbose -z --expire=", GNU, display);
            }
            "worktree" => self.git_change(rest, Some("-b= -B= -f --force --detach --checkout --no-checkout --lock --reason= -q --quiet --track --no-track --guess-remote --orphan -n --dry-run -v --verbose --expire="), "adds or removes worktrees", display),
            "reflog" if matches!(words.first(), None | Some(&"show")) || words.first().is_some_and(|a| a.starts_with('-')) => {
                self.options(&rest[usize::from(words.first() == Some(&"show"))..], options::GIT_LOG, GNU, display);
            }
            "reflog" => self.git_change(rest, Some("--expire= --expire-unreachable= --all --dry-run -n --rewrite --updateref --stale-fix -v --verbose"), "deletes reflog entries", display),
            "push" => {
                let reason = if has(&["-f", "--force", "--force-with-lease", "--delete", "-d", "--mirror"]) || words.iter().any(|a| a.starts_with("--force-with-lease=")) { "force-pushes or deletes remote history" } else { "publishes commits to a remote" };
                self.git_change(rest, Some("-u --set-upstream -f --force --force-with-lease? --force-if-includes --tags --follow-tags --all --branches --delete -d --dry-run -n -v --verbose -q --quiet --atomic --porcelain --no-verify --prune --mirror -o= --push-option="), reason, display);
            }
            "reset" => self.git_change(rest, Some("--hard --soft --mixed --merge --keep -q --quiet -N --no-refresh"), if has(&["--hard", "--merge", "--keep"]) { "discards uncommitted changes (reset --hard)" } else { "moves the branch or index" }, display),
            "clean" => self.git_change(rest, Some("-f --force -d -x -X -n --dry-run -q --quiet -e= --exclude="), "deletes untracked files", display),
            "checkout" | "restore" | "switch" => {
                let discards = has(&["--", ".", "-f", "--force", "--theirs", "--ours", "-p", "--patch"]) || sub.text == "restore";
                let spec = "-b= -B= -c= -C= --detach -t --track? --no-track -f --force -q --quiet -m --merge --orphan= --ours --theirs --guess --no-guess --discard-changes --create= --force-create= -s= --source= -S --staged -W --worktree --ignore-unmerged --overlay --no-overlay";
                self.git_change(rest, Some(spec), if discards { "discards uncommitted changes" } else { "changes the repository" }, display);
            }
            "filter-branch" | "filter-repo" | "gc" | "prune" | "update-ref" | "replace" | "fsck" => self.git_change(rest, None, "rewrites or prunes repository history", display),
            "fetch" => self.git_change(rest, Some("--all -p --prune -P --prune-tags -t --tags --no-tags -q --quiet -v --verbose --depth= --deepen= --shallow-since= --unshallow --dry-run -f --force --multiple -j= --jobs= --no-recurse-submodules --atomic --append -a"), "uses the network", display),
            "pull" => self.git_change(rest, Some("--rebase -r --no-rebase --ff --no-ff --ff-only --squash --autostash --no-autostash -q --quiet -v --verbose --tags -p --prune --depth= --all --no-edit --stat -n --no-stat --allow-unrelated-histories"), "uses the network", display),
            "clone" => self.git_change(rest, Some("--depth= -b= --branch= --single-branch --no-single-branch --recursive --recurse-submodules --shallow-submodules -q --quiet -v --verbose --bare --mirror --no-checkout -n --filter= --origin= -o= --no-tags --sparse"), "uses the network", display),
            "ls-remote" | "submodule" | "send-email" | "request-pull" => self.git_change(rest, None, "uses the network", display),
            "credential" | "credential-cache" | "credential-store" => self.add(Class::Ask, "reads or stores git credentials", display, false),
            "add" => self.git_change(rest, Some("-A --all -u --update -N --intent-to-add -f --force -v --verbose -n --dry-run --ignore-errors --renormalize"), "changes the repository", display),
            "commit" => self.git_change(rest, Some("-m= --message= -a --all --amend --no-edit -q --quiet -v --verbose --allow-empty --allow-empty-message -s --signoff --no-verify -n --author= --date= --fixup= --squash= -F= --file= --only -o --include -i --reset-author --no-gpg-sign --cleanup="), "changes the repository", display),
            "merge" => self.git_change(rest, Some("--no-ff --ff --ff-only --squash --no-squash --abort --continue --quit -m= --no-edit -q --quiet -v --stat -n --no-stat --no-verify --allow-unrelated-histories --autostash -X= --strategy-option="), "changes the repository", display),
            "rebase" => self.git_change(rest, Some("--onto= --continue --abort --skip --quit --autosquash --no-autosquash --autostash --no-autostash -q --quiet -v --root --update-refs --keep-base --fork-point --no-fork-point --empty= --committer-date-is-author-date --reset-author-date --ignore-date --no-verify"), "changes the repository", display),
            "cherry-pick" | "revert" => self.git_change(rest, Some("-n --no-commit -x -m= --mainline= --continue --abort --skip --quit --allow-empty --ff -s --signoff"), "changes the repository", display),
            "rm" => self.git_change(rest, Some("-r -f --force --cached -q --quiet -n --dry-run --ignore-unmatch --sparse"), "changes the repository", display),
            "mv" => self.git_change(rest, Some("-f --force -k -n --dry-run -v --verbose"), "changes the repository", display),
            "init" => self.git_change(rest, Some("-q --quiet --bare -b= --initial-branch="), "changes the repository", display),
            _ => self.git_change(rest, None, "changes the repository", display),
        }
    }

    /// A git command that changes something. A rule `git <command> *` would
    /// match any options, so only a command whose options are all on the
    /// command's closed list may get one.
    fn git_change(&mut self, rest: &[Word], spec: Option<&'static str>, reason: &str, display: &[String]) {
        let listable = options::read(rest, spec.unwrap_or(""), GNU).unread.is_none();
        self.add(Class::Ask, reason, display, listable);
    }

    fn cargo(&mut self, args: &[Word], display: &[String]) {
        // `--config` sets a runner, linker or wrapper, before or after the subcommand.
        if args.iter().any(|a| a.text == "--config" || a.text.starts_with("--config=")) {
            return self.add(Class::Ask, "sets cargo configuration that can run programs", display, false);
        }
        let start = usize::from(args.first().is_some_and(|w| w.text.starts_with('+') && !w.dynamic));
        let Some(global) = self.options(&args[start..], options::CARGO, POSIX, display) else { return };
        let Some(&sub) = global.operands.first() else { return };
        let sub = start + sub;
        let rest = &args[sub + 1..];
        let ask = |ctx: &mut Self, reason: &str| ctx.add(Class::Ask, reason, display, true);
        // A command whose options are not read gets a rule `cargo clean *`
        // only with plain operands, so such a rule never covers an option.
        let change = |ctx: &mut Self, reason: &str| ctx.add(Class::Ask, reason, display, plain_operands(rest));
        match args[sub].text.as_str() {
            "check" | "build" | "b" | "c" | "test" | "t" | "bench" | "run" | "r" | "doc" | "d"
            | "tree" | "metadata" | "version" | "help" | "locate-project" | "verify-project"
            | "pkgid" | "read-manifest" | "rustc" | "rustdoc" | "kernels" | "expand" | "nextest" | "clippy" | "fmt" => {
                let Some(read) = self.options(rest, options::CARGO, GNU, display) else { return };
                // After `--`: arguments for the test or program, which run
                // anyway, or flags for rustc, clippy or rustfmt.
                let tail = read.dashdash.map_or(&rest[..0], |at| &rest[at + 1..]);
                let sub = args[sub].text.as_str();
                match sub {
                    "test" | "t" | "bench" | "run" | "r" => {}
                    "nextest" if read.operands.first().is_some_and(|&at| matches!(rest[at].text.as_str(), "run" | "r" | "list")) => {}
                    "nextest" => return ask(self, "runs a cargo command that is not on the safe list"),
                    "clippy" => {
                        self.options(tail, "-D= -W= -A= -F= --deny= --warn= --allow= --forbid= --cap-lints= --no-deps", GNU, display);
                    }
                    "fmt" => {
                        let Some(flags) = self.options(tail, "--check --edition= -v --verbose -q --quiet --color= -l --files-with-diff", GNU, display) else { return };
                        if !(read.has(&["--check"]) || flags.has(&["--check"])) {
                            ask(self, "rewrites source files");
                        }
                    }
                    _ if !tail.is_empty() => self.add(Class::Ask, "passes flags to the compiler, which can run programs such as a linker", display, false),
                    _ => {}
                }
                if read.has(&["--open"]) {
                    ask(self, "opens a browser");
                }
                if read.has(&["--fix"]) {
                    ask(self, "rewrites source files");
                }
            }
            "install" | "uninstall" | "binstall" => change(self, "installs or removes programs"),
            "publish" | "login" | "logout" | "owner" | "yank" => change(self, "publishes crates or changes a registry account"),
            "clean" => change(self, "deletes build output"),
            "add" | "remove" | "rm" | "update" | "upgrade" | "fix" | "new" | "init" | "vendor" | "generate-lockfile" | "fetch" => change(self, "changes the project or its dependencies"),
            _ => change(self, "runs a cargo command that is not on the safe list"),
        }
    }

    fn go(&mut self, args: &[Word], display: &[String]) {
        let Some(sub) = args.first().map(|w| w.text.as_str()) else { return };
        let mut rest = &args[1..];
        let (spec, mode) = match sub {
            "build" | "run" | "list" => (options::GO_BUILD, WHOLE),
            // go test and go vet take flags after the packages too; the words
            // after `-args` go to the test program.
            "test" | "vet" => {
                rest = &rest[..rest.iter().position(|w| matches!(w.text.as_str(), "-args" | "--args")).unwrap_or(rest.len())];
                (options::GO_BUILD, WHOLE_ANY)
            }
            "version" => ("-m -v -json", WHOLE),
            "env" => ("-json -changed", WHOLE),
            "doc" => ("-all -c -cmd -short -src -u", WHOLE),
            "help" => ("", WHOLE),
            "mod" if matches!(rest.first().map(|w| w.text.as_str()), Some("graph" | "why" | "verify")) => {
                rest = &rest[1..];
                ("-m -vendor -go= -x", WHOLE)
            }
            "install" | "get" => return self.add(Class::Ask, "installs packages", display, plain_operands(rest)),
            "fmt" | "fix" | "generate" | "mod" | "work" | "clean" => return self.add(Class::Ask, "changes files", display, plain_operands(rest)),
            _ => return self.add(Class::Ask, "runs a go command that is not on the safe list", display, plain_operands(rest)),
        };
        let Some(read) = self.options(rest, spec, mode, display) else { return };
        let outputs: Vec<&Word> = read.values(options::GO_WRITES).collect();
        if !outputs.is_empty() {
            let place = if outputs.iter().any(|w| self.outside(&w.text)) { " outside the workspace" } else { "" };
            self.add(Class::Ask, format!("writes a file{place}"), display, false);
        }
        // `go run pkg@version` downloads a program.
        if sub == "run" && read.operands.iter().any(|&at| rest[at].text.contains('@')) {
            self.add(Class::Ask, "go option runs or downloads another program", display, false);
        }
    }
}

/// The directives of a printf format, when each one is plain: Some(true)
/// when one reads a number (`%d`, a `*` width), None for anything else
/// (`%n`, `%1$s`, `%(...)T`).
fn printf_format(format: &str) -> Option<bool> {
    let mut chars = format.chars().peekable();
    let mut numbers = false;
    while let Some(c) = chars.next() {
        if c != '%' {
            continue;
        }
        if chars.next_if_eq(&'%').is_some() {
            continue;
        }
        while chars.next_if(|c| "-+ #0'".contains(*c)).is_some() {}
        for part in 0..2 {
            if part == 1 && chars.next_if_eq(&'.').is_none() {
                break;
            }
            if chars.next_if_eq(&'*').is_some() {
                numbers = true;
            }
            while chars.next_if(|c| c.is_ascii_digit()).is_some() {}
        }
        match chars.next()? {
            's' | 'b' | 'c' | 'q' => {}
            'd' | 'i' | 'o' | 'u' | 'x' | 'X' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'a' | 'A' => numbers = true,
            _ => return None,
        }
    }
    Some(numbers)
}

/// True for a sed program made only of `p`, `d` and `s///` commands (with
/// the flags `g`, `p`, `i`, `I` or a number), each with optional line, `$`
/// or `/regex/` addresses. Anything else, such as `w`, `e`, `r`, blocks,
/// branches and labels, is not read.
pub(super) fn sed_read_only(script: &str) -> bool {
    let chars: Vec<char> = script.chars().collect();
    let mut i = 0;
    let blank = |i: &mut usize| {
        while matches!(chars.get(*i), Some(' ' | '\t')) {
            *i += 1;
        }
    };
    loop {
        while matches!(chars.get(i), Some(' ' | '\t' | '\n' | ';')) {
            i += 1;
        }
        if i >= chars.len() {
            return true;
        }
        for second in [false, true] {
            match chars.get(i) {
                Some('$') => i += 1,
                Some(c) if c.is_ascii_digit() => {
                    while chars.get(i).is_some_and(|c| c.is_ascii_digit()) {
                        i += 1;
                    }
                }
                Some('/') => match sed_regex(&chars, i + 1, '/') {
                    Some(end) => i = end,
                    None => return false,
                },
                _ if !second => break,
                _ => return false,
            }
            if second || chars.get(i) != Some(&',') {
                break;
            }
            i += 1;
        }
        blank(&mut i);
        match chars.get(i) {
            Some('p' | 'd') => i += 1,
            Some('s') => {
                let Some(&delim) = chars.get(i + 1).filter(|c| matches!(c, '/' | '|' | '#' | ':' | ',' | '@')) else { return false };
                let Some(end) = sed_regex(&chars, i + 2, delim) else { return false };
                // The replacement: escapes, but no bracket sets and no newline.
                i = end;
                loop {
                    match chars.get(i) {
                        Some(&c) if c == delim => break,
                        Some('\\') if chars.get(i + 1).is_some_and(|&n| n != '\n') => i += 2,
                        None | Some('\\' | '\n') => return false,
                        _ => i += 1,
                    }
                }
                i += 1;
                while chars.get(i).is_some_and(|c| matches!(c, 'g' | 'p' | 'i' | 'I') || c.is_ascii_digit()) {
                    i += 1;
                }
            }
            _ => return false,
        }
        blank(&mut i);
        match chars.get(i) {
            None => return true,
            Some(';' | '\n') => i += 1,
            _ => return false,
        }
    }
}

/// The index after the `delim` that ends a sed regex starting at `at`. A
/// bracket set may hold only plain characters (no `delim`, `[`, `\` or
/// newline), so every sed finds the end at the same place.
fn sed_regex(chars: &[char], mut at: usize, delim: char) -> Option<usize> {
    while let Some(&c) = chars.get(at) {
        match c {
            _ if c == delim => return Some(at + 1),
            '\n' => return None,
            '\\' => match chars.get(at + 1) {
                None | Some('\n' | 'c') => return None,
                Some(_) => at += 2,
            },
            '[' => {
                at += 1 + usize::from(chars.get(at + 1) == Some(&'^'));
                let close = at + chars[at..].iter().position(|&c| c == ']')?;
                if close == at || chars[at..close].iter().any(|&c| c == delim || matches!(c, '[' | '\\' | '\n')) {
                    return None;
                }
                at = close + 1;
            }
            _ => at += 1,
        }
    }
    None
}

/// True when no word is an option or can still expand into one.
/// The words that can name an output file: the word after `-o` or
/// `--output`, the value of `--output=x`, and the rest of a short cluster
/// after its `o` (`-so/dev/sda`). Other options are not read, so this can
/// find more than the program writes, but not less.
fn output_values(args: &[Word], short: &[char], long: &[&str]) -> Vec<Word> {
    let mut found = Vec::new();
    for (index, word) in args.iter().enumerate() {
        let next = || args.get(index + 1).cloned();
        let value = match word.text.strip_prefix("--") {
            Some(option) => match option.split_once('=') {
                Some((key, value)) if long.contains(&key) => Some(Word { text: value.to_owned(), ..word.clone() }),
                None if long.contains(&option) => next(),
                _ => None,
            },
            None => match word.text.strip_prefix('-').and_then(|letters| letters.find(short).map(|at| &letters[at + 1..])) {
                Some("") => next(),
                Some(rest) => Some(Word { text: rest.to_owned(), ..word.clone() }),
                None => None,
            },
        };
        found.extend(value);
    }
    found
}

fn plain_operands(words: &[Word]) -> bool {
    words.iter().all(|w| !w.text.starts_with('-') && !options::expands(w))
}

/// `word` with each file name slot (`{}`, also inside a word such as
/// `sh -c 'echo {}'`) as an unknown value: find and fd put a file name
/// there, and a name can be any text, an option or shell code too.
fn file_slots(word: &Word, slots: &[&str]) -> Word {
    let mut word = word.clone();
    if slots.iter().any(|slot| word.text.contains(slot)) {
        for slot in slots {
            word.text = word.text.replace(slot, "$FILE");
        }
        word.dynamic = true;
    }
    word
}
