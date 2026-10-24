    use super::*;
    #[cfg(target_os = "macos")]
    use crate::test_workspace::Fixture;
    fn spec(args: &[&str], timeout_ms: u64) -> CommandSpec {
        CommandSpec {
            argv: args.iter().map(|s| (*s).into()).collect(),
            timeout_ms,
        }
    }
    #[cfg(target_os = "macos")]
    fn fixture_git(f: &Fixture, args: &[&str]) -> String {
        // Only temporary local repositories are used. No inherited Git
        // configuration, identity, credentials, hooks or remote is needed.
        let output = Command::new("/usr/bin/git")
            .args([
                "-c",
                "user.name=Sandbox Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(&f.0)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", &f.0)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(output.status.success(), "local git fixture setup failed");
        String::from_utf8(output.stdout).unwrap()
    }
    #[cfg(target_os = "macos")]
    fn git_fixture() -> Fixture {
        let f = Fixture::new();
        fixture_git(&f, &["init", "--quiet", "--initial-branch=main"]);
        f.write("source.rs", "fn baseline() {}\n");
        fixture_git(&f, &["add", "--", "source.rs"]);
        fixture_git(&f, &["commit", "--quiet", "-m", "fixture baseline"]);
        f
    }
    #[cfg(target_os = "macos")]
    fn sandbox_git(f: &Fixture, args: &[&str]) -> CommandOutput {
        let command = spec(args, 5000);
        assert!(command.allowlisted());
        run(&f.workspace(), &command, 16 * 1024, &AtomicBool::new(false)).unwrap()
    }
    #[cfg(target_os = "macos")]
    fn git_snapshot(
        root: &std::path::Path,
    ) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
        let mut contents = std::collections::BTreeMap::new();
        let mut dirs = vec![root.to_owned()];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                } else {
                    contents.insert(
                        path.strip_prefix(root).unwrap().to_owned(),
                        std::fs::read(path).unwrap(),
                    );
                }
            }
        }
        contents
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn read_only_git_status_diff_log_run_without_changing_git_metadata() {
        let f = git_fixture();
        f.write("source.rs", "fn changed() {}\n");
        let before = git_snapshot(&f.0.join(".git"));
        for args in [
            vec!["git", "status"],
            vec!["git", "status", "--short"],
            vec!["git", "diff"],
            vec!["git", "diff", "--stat"],
            vec!["git", "log"],
            vec!["git", "log", "--oneline"],
            vec!["git", "log", "-n", "1"],
        ] {
            let output = sandbox_git(&f, &args);
            assert_eq!(
                output.outcome,
                Outcome::Exited(Some(0)),
                "{}",
                output.render()
            );
            let expected = if args[1] == "log" {
                "fixture baseline"
            } else {
                "source.rs"
            };
            assert!(output.stdout.contains(expected), "{}", output.render());
            assert_eq!(git_snapshot(&f.0.join(".git")), before, "{args:?}");
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn read_only_git_excludes_tracked_and_untracked_hidden_and_credential_paths() {
        let f = git_fixture();
        let mut protected: Vec<String> = crate::tools::CREDENTIAL_NAMES
            .iter()
            .map(|name| (*name).to_owned())
            .chain(
                crate::tools::CREDENTIAL_SUFFIXES
                    .iter()
                    .map(|suffix| format!("fixture{suffix}")),
            )
            .collect();
        protected.extend([
            ".private".into(),
            ".private-dir/payload".into(),
            "nested/.private".into(),
            "nested/.private-dir/payload".into(),
        ]);
        let original = protected.clone();
        for path in original {
            if !path.starts_with('.') && !path.contains("/.") {
                protected.push(format!("nested/{}", path.to_ascii_uppercase()));
                protected.push(format!("{path}/payload"));
            }
        }
        // Directory names and files are separate fixtures so credentials can
        // be covered as path components without conflicting with a file.
        for path in &mut protected {
            if path.ends_with("/payload") && !path.starts_with('.') && !path.contains("/.") {
                *path = format!("directories/{path}");
            }
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(f.0.join(parent)).unwrap();
            }
            f.write(path, "synthetic-protected-baseline\n");
        }
        fixture_git(&f, &["add", "--all"]);
        fixture_git(&f, &["commit", "--quiet", "-m", "protected fixture paths"]);
        for path in &protected {
            f.write(path, "synthetic-protected-changed\n");
        }
        std::fs::create_dir_all(f.0.join("untracked/.private-dir")).unwrap();
        f.write(
            "untracked/.private-dir/payload",
            "synthetic-protected-untracked\n",
        );
        f.write("untracked/SECRETS.json", "synthetic-protected-untracked\n");
        f.write("untracked/client.PEM", "synthetic-protected-untracked\n");
        f.write("source.rs", "fn visible_change() {}\n");
        f.write("visible-new.rs", "fn visible_new() {}\n");
        for args in [
            vec!["git", "status", "--short"],
            vec!["git", "diff"],
            vec!["git", "diff", "--stat"],
        ] {
            let output = sandbox_git(&f, &args);
            assert_eq!(
                output.outcome,
                Outcome::Exited(Some(0)),
                "{}",
                output.render()
            );
            assert!(output.stdout.contains("source.rs"), "{}", output.render());
            if args[1] == "status" {
                assert!(
                    output.stdout.contains("visible-new.rs"),
                    "{}",
                    output.render()
                );
            } else if args.len() == 2 {
                assert!(
                    output.stdout.contains("visible_change"),
                    "{}",
                    output.render()
                );
            }
            for path in &protected {
                assert!(
                    !output.stdout.contains(path),
                    "protected path {path}: {}",
                    output.render()
                );
            }
            assert!(
                !output.stdout.contains("synthetic-protected"),
                "{}",
                output.render()
            );
            assert!(
                !output.stdout.contains("SECRETS.json"),
                "{}",
                output.render()
            );
            assert!(!output.stdout.contains("client.PEM"), "{}", output.render());
            assert!(!output.stdout.contains(".private"), "{}", output.render());
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn read_only_git_reports_ignore_file_denial_without_exposing_ignored_contents() {
        let f = git_fixture();
        f.write(".gitignore", "target/\nnode_modules/\ngenerated/\n");
        fixture_git(&f, &["add", "--", ".gitignore"]);
        fixture_git(
            &f,
            &["commit", "--quiet", "-m", "ignore fixture directories"],
        );
        for name in ["target", "node_modules", "generated"] {
            std::fs::create_dir(f.0.join(name)).unwrap();
            f.write(&format!("{name}/artifact"), "synthetic-ignored-contents\n");
        }
        assert!(fixture_git(&f, &["status", "--short"]).is_empty());
        let output = sandbox_git(&f, &["git", "status", "--short"]);
        assert_eq!(
            output.outcome,
            Outcome::Exited(Some(0)),
            "{}",
            output.render()
        );
        // Hidden .gitignore stays opaque: read-only git can report ignored
        // visible directories as untracked. Excluded build trees stay unreadable.
        assert!(output.stderr.contains(".gitignore"), "{}", output.render());
        assert!(output.stdout.contains("generated/"), "{}", output.render());
        assert!(!output.stdout.contains(".gitignore"), "{}", output.render());
        assert!(!output.stdout.contains("synthetic-ignored-contents"));
        assert_eq!(f.read(".gitignore"), "target/\nnode_modules/\ngenerated/\n");
        for name in ["target", "node_modules"] {
            assert!(
                output.stderr.contains(&format!("'{name}/'")),
                "{}",
                output.render()
            );
            assert!(
                !output.stdout.contains(&format!("{name}/")),
                "{}",
                output.render()
            );
            let denied = run(
                &f.workspace(),
                &spec(&["cat", &format!("{name}/artifact")], 2000),
                1024,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_ne!(denied.outcome, Outcome::Exited(Some(0)));
            assert!(denied.stdout.is_empty());
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn read_only_git_profile_denies_git_index_and_config_writes() {
        let f = git_fixture();
        let workspace = f.workspace();
        let session = BuildSession::default();
        let build = session.prepare(workspace.root()).unwrap();
        let sandbox = profile(
            workspace.root(),
            &Exclusions::default(),
            &build.as_ref().unwrap().path,
            None,
            true,
            true,
        )
        .unwrap();
        let before = git_snapshot(&f.0.join(".git"));
        assert!(
            !sandbox.contains("(subpath \"/usr\")")
                && !sandbox.contains("(subpath \"/usr/local\")")
        );
        // Force actual mutations: --refresh can be a successful no-op when
        // the index already matches file metadata. The .git read exception
        // must still deny writes under the allowlist's OS policy.
        for args in [
            vec!["update-index", "--force-remove", "--", "source.rs"],
            vec!["config", "fixture.probe", "changed"],
        ] {
            let mut command = Command::new("/usr/bin/sandbox-exec");
            command
                .args(["-p", &sandbox, "--", "/usr/bin/git"])
                .args(&args)
                .current_dir(workspace.root())
                .env_clear()
                .stdin(Stdio::null());
            build.as_ref().unwrap().environment(&mut command, None);
            let output = command.output().unwrap();
            assert!(
                !output.status.success(),
                "{args:?}: write unexpectedly allowed"
            );
            assert_eq!(git_snapshot(&f.0.join(".git")), before, "{args:?}");
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn read_only_git_worktree_file_cannot_expand_the_readable_gitdir() {
        let repository = git_fixture();
        let f = Fixture::new();
        fixture_git(
            &repository,
            &[
                "worktree",
                "add",
                "--quiet",
                "--detach",
                f.0.to_str().unwrap(),
            ],
        );
        assert!(f.0.join(".git").is_file());
        assert!(fixture_git(&f, &["status", "--short"]).is_empty());
        let before = git_snapshot(&repository.0.join(".git"));
        for args in [
            vec!["git", "status", "--short"],
            vec!["git", "diff"],
            vec!["git", "log", "--oneline"],
        ] {
            let output = sandbox_git(&f, &args);
            assert!(
                matches!(output.outcome, Outcome::Exited(Some(code)) if code != 0),
                "{}",
                output.render()
            );
            assert!(output.stdout.is_empty(), "{}", output.render());
            assert!(
                !output.stderr.contains("fixture baseline"),
                "{}",
                output.render()
            );
            assert_eq!(git_snapshot(&repository.0.join(".git")), before);
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn read_only_git_cannot_execute_required_clean_filters() {
        use std::os::unix::fs::PermissionsExt;
        let f = git_fixture();
        f.write(".git/info/attributes", "source.rs filter=fixture\n");
        f.write(
            "filter-probe",
            "#!/bin/sh\nprintf synthetic-filter-executed >&2\n/bin/cat\n",
        );
        std::fs::set_permissions(
            f.0.join("filter-probe"),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        fixture_git(&f, &["config", "filter.fixture.clean", "./filter-probe"]);
        fixture_git(&f, &["config", "filter.fixture.required", "true"]);
        // Equal lengths require status to hash the file instead of declaring
        // it modified from its size alone.
        f.write("source.rs", "fn modified() {}\n");
        // The ordinary local Git control proves that these attributes trigger
        // the filter. The sandbox must deny its shell before it can run.
        for args in [
            vec!["status", "--short"],
            vec!["diff", "--no-ext-diff", "--no-textconv"],
        ] {
            let control = Command::new("/usr/bin/git")
                .args(&args)
                .current_dir(&f.0)
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .output()
                .unwrap();
            assert!(control.status.success());
            assert!(
                String::from_utf8_lossy(&control.stderr).contains("synthetic-filter-executed"),
                "{args:?}: control did not execute the filter"
            );
        }
        let before = git_snapshot(&f.0.join(".git"));
        for args in [vec!["git", "status", "--short"], vec!["git", "diff"]] {
            let output = sandbox_git(&f, &args);
            assert_ne!(
                output.outcome,
                Outcome::Exited(Some(0)),
                "{}",
                output.render()
            );
            assert!(!output.stdout.contains("synthetic-filter-executed"));
            assert!(
                !output.stderr.contains("synthetic-filter-executed"),
                "{}",
                output.render()
            );
            assert_eq!(git_snapshot(&f.0.join(".git")), before);
        }
        // A config-selected attributes file is disabled entirely, rather
        // than allowing it to trigger another attempted helper execution.
        std::fs::remove_file(f.0.join(".git/info/attributes")).unwrap();
        f.write("attributes", "source.rs filter=fixture\n");
        fixture_git(&f, &["config", "core.attributesFile", "attributes"]);
        let output = sandbox_git(&f, &["git", "diff"]);
        assert_eq!(
            output.outcome,
            Outcome::Exited(Some(0)),
            "{}",
            output.render()
        );
        assert!(output.stdout.contains("modified"), "{}", output.render());
        assert!(!output.stderr.contains("filter"), "{}", output.render());
    }
    #[test]
    fn command_preview_is_lossless_under_terminal_sanitization() {
        let command = spec(
            &[
                "/bin/sh",
                "-c",
                "printf 'x\u{0085}\u{202e}\u{200b}\u{fe0f}\u{e0100}\u{2800}\u{00a0}y'\n# literal \\u202e \\u200b \\udb40\\udd00",
            ],
            1000,
        );
        command.validate().unwrap();
        let preview = command.preview();
        assert_eq!(crate::terminal::sanitize(&preview), preview);
        assert!(
            !preview
                .chars()
                .any(crate::terminal::preview_char_needs_escape)
        );
        assert!(preview.contains("\\u200b"));
        assert!(preview.contains("\\ufe0f"));
        assert!(preview.contains("\\udb40\\udd00"));
        assert!(preview.contains("\\\\u200b"));
        assert!(preview.contains("\\\\udb40\\\\udd00"));
        let argv = preview
            .strip_prefix("command argv=")
            .unwrap()
            .split_once(" · timeout=")
            .unwrap()
            .0;
        let parsed = crate::json::parse(argv).unwrap();
        let roundtrip: Vec<_> = parsed
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(roundtrip, command.argv);
    }
    #[test]
    fn allowlist_is_exact_and_arguments_are_bounded() {
        assert!(spec(&["/bin/pwd"], 1000).allowlisted());
        for args in [
            vec!["pwd", ";"],
            vec!["cat", "../outside"],
            vec!["git", "diff", "--output=changed"],
            vec!["git", "-c", "core.pager=unsafe", "log"],
            vec!["/bin/pwd", ";"],
            vec!["/bin/sh", "-c", "pwd"],
            vec!["/bin/ls", "-la"],
            vec!["./true"],
        ] {
            assert!(!spec(&args, 1000).allowlisted());
        }
        assert!(spec(&[], 1000).validate().is_err());
        assert!(spec(&["/bin/pwd"], 120001).validate().is_err());
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn executes_in_workspace_with_cleared_environment_and_exit_code() {
        let f = Fixture::new();
        let w = f.workspace();
        let out = run(
            &w,
            &spec(
                &[
                    "/bin/sh",
                    "-c",
                    "printf ok; printf error >&2; test -z \"$VADE_TEST_SECRET\"; exit 7",
                ],
                2000,
            ),
            1024,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(out.outcome, Outcome::Exited(Some(7)), "{}", out.render());
        assert_eq!(out.stdout, "ok");
        assert_eq!(out.stderr, "error");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn sandbox_denies_outside_writes_hidden_reads_and_network() {
        let f = Fixture::new();
        let outside = Fixture::new();
        let w = f.workspace();
        f.write(".env", "fake-private-marker");
        outside.write("outside", "outside-private-marker");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let outside_path = outside.0.canonicalize().unwrap();
        let script = format!(
            "printf no > '{0}/escape'; cat .env '{0}/outside' '/System/Volumes/Data{0}/outside'; ln -s '{0}' link; cat link/outside; printf no > link/escaped; /usr/bin/curl --max-time 1 http://{1}; printf yes > allowed",
            outside_path.display(),
            listener.local_addr().unwrap()
        );
        let out = run(
            &w,
            &spec(&["/bin/sh", "-c", &script], 3000),
            4096,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!outside.0.join("escape").exists(), "{}", out.render());
        assert!(!out.stdout.contains("fake-private-marker"));
        assert!(
            !out.stdout.contains("outside-private-marker"),
            "{}",
            out.render()
        );
        assert!(!outside.0.join("escaped").exists());
        assert_eq!(f.read("allowed"), "yes");
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn sandbox_credential_rules_cover_names_extensions_and_case_aliases() {
        let f = Fixture::new();
        let names = [
            "credentials.json",
            "secrets.yaml",
            "id_rsa",
            "authorized_keys",
            "config.key",
            "cert.p12",
            "client.keystore",
            "token.p8",
            "store.jks",
            "terraform.tfstate",
            "service-account.json",
        ];
        let mut script = String::new();
        for name in names {
            f.write(name, "synthetic-private-file");
            for alias in [name.to_owned(), name.to_ascii_uppercase()] {
                script.push_str(&format!("cat '{alias}'; printf changed > '{alias}'; "));
            }
        }
        script.push_str("printf okay > allowed");
        let out = run(
            &f.workspace(),
            &spec(&["/bin/sh", "-c", &script], 3000),
            8192,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            !out.stdout.contains("synthetic-private-file"),
            "{}",
            out.render()
        );
        for name in names {
            assert_eq!(f.read(name), "synthetic-private-file", "{name}");
        }
        assert_eq!(f.read("allowed"), "okay");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn sandbox_cannot_read_other_process_environments_or_hostname() {
        const MARKER: &str = "VADE_SYNTHETIC_SANDBOX_ENV_MARKER";
        // A subprocess receives the marker at exec time, without unsafe global
        // environment mutation in this multithreaded Rust test process.
        if std::env::var_os(MARKER).is_none() {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "shell::tests::sandbox_cannot_read_other_process_environments_or_hostname",
                ])
                .env(MARKER, "synthetic-never-export-this-marker")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .unwrap();
            assert!(
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
                "isolated sandbox privacy regression failed"
            );
            return;
        }
        let f = Fixture::new();
        let w = f.workspace();
        // Positive control: prove the platform probe can see this synthetic
        // marker outside the sandbox; never print inherited environment data.
        let control = Command::new("/bin/ps")
            .args(["-E", "-p", &std::process::id().to_string(), "-o", "command"])
            .output()
            .unwrap();
        assert!(control.status.success());
        assert!(
            String::from_utf8_lossy(&control.stdout).contains("synthetic-never-export-this-marker")
        );
        let output = run(
            &w,
            &spec(&["/bin/ps", "-E", "-x", "-o", "command"], 2000),
            64 * 1024,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!output.stdout.contains("synthetic-never-export-this-marker"));
        assert!(!output.stderr.contains("synthetic-never-export-this-marker"));
        // Also exercise KERN_PROCARGS2 for this exact same-UID parent process.
        let script = format!("/bin/ps -E -p {} -o command", std::process::id());
        let output = run(
            &w,
            &spec(&["/bin/sh", "-c", &script], 2000),
            64 * 1024,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!output.stdout.contains("synthetic-never-export-this-marker"));
        let output = run(
            &w,
            &spec(&["/usr/sbin/sysctl", "-n", "kern.hostname"], 2000),
            4096,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(output.outcome != Outcome::Exited(Some(0)));
        assert!(output.stdout.trim().is_empty());
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn timeout_truncated_display_and_cancellation_are_bounded() {
        let f = Fixture::new();
        let w = f.workspace();
        let start = Instant::now();
        let out = run(
            &w,
            &spec(&["/bin/sh", "-c", "sleep 10 & wait"], 100),
            1024,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(out.outcome, Outcome::Timeout, "{}", out.render());
        assert!(start.elapsed() < Duration::from_secs(2));
        let out = run(
            &w,
            &spec(&["/bin/sh", "-c", "printf 'BEGIN\\n'; /usr/bin/yes | /usr/bin/head -c 8192; printf 'END\\n'; printf 'error: repair this\\n' >&2; exit 7"], 2000),
            128,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(out.outcome, Outcome::Exited(Some(7)), "{}", out.render());
        assert!(out.stdout.starts_with("BEGIN\n") && out.stdout.ends_with("END\n"));
        assert_eq!(out.stderr, "error: repair this\n");
        assert_eq!(out.stdout_bytes - out.stdout_omitted, 64);
        assert!(out.stdout_omitted > 8000 && out.stderr_omitted == 0);
        let cancel = AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(100));
                cancel.store(true, Ordering::Relaxed);
            });
            let out = run(&w, &spec(&["/bin/sleep", "10"], 2000), 1024, &cancel).unwrap();
            assert_eq!(out.outcome, Outcome::Cancelled);
        });
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn refuses_linked_workspaces() {
        let f = Fixture::new();
        let other = Fixture::new();
        other.write("a", "outside");
        std::fs::hard_link(other.0.join("a"), f.0.join("hard")).unwrap();
        let error = run(
            &f.workspace(),
            &spec(&["/bin/pwd"], 1000),
            1024,
            &AtomicBool::new(false),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("\"hard\""), "{error}");
        assert!(error.contains("hard-linked"), "{error}");
        std::fs::create_dir(f.0.join("keys.pem")).unwrap();
        std::fs::rename(f.0.join("hard"), f.0.join("keys.pem/hard")).unwrap();
        let spec = spec(&["/bin/pwd"], 1000);
        let error = run(&f.workspace(), &spec, 1024, &AtomicBool::new(false)).unwrap_err();
        assert!(error.to_string().contains("keys.pem/hard"));
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn build_and_dependency_hardlinks_cannot_be_read_mutated_or_moved() {
        let f = Fixture::new();
        let outside = Fixture::new();
        outside.write("artifact", "original-artifact");
        let mut exclusions = Exclusions::default();
        exclusions.add("build.cache").unwrap();
        for dir in [
            "target/debug/deps",
            "packages/ui/node_modules/pkg",
            "build.cache",
        ] {
            std::fs::create_dir_all(f.0.join(dir)).unwrap();
            std::fs::hard_link(outside.0.join("artifact"), f.0.join(dir).join("artifact")).unwrap();
        }
        let data_alias = format!(
            "/System/Volumes/Data{}/target/debug/deps/artifact",
            f.0.canonicalize().unwrap().display()
        );
        let script = format!(
            r#"
cat target/debug/deps/artifact
cat packages/ui/node_modules/pkg/artifact build.cache/artifact '{data_alias}'
cat TARGET/debug/deps/artifact packages/ui/NODE_MODULES/pkg/artifact
printf changed > target/debug/deps/artifact
printf changed > packages/ui/node_modules/pkg/artifact
printf changed > build.cache/artifact
printf changed > '{data_alias}'
printf changed > TARGET/debug/deps/artifact
printf changed > packages/ui/NODE_MODULES/pkg/artifact
rm target/debug/deps/artifact
mv target renamed
mv TARGET renamed-uppercase
mv packages/ui/node_modules packages/ui/renamed
mv build.cache renamed-cache
mkdir target/new
printf allowed > source.txt
"#
        );
        let out = run_with_exclusions(
            &f.workspace(),
            &spec(&["/bin/sh", "-c", &script], 3000),
            8192,
            &AtomicBool::new(false),
            &exclusions,
        )
        .unwrap();
        assert_eq!(out.outcome, Outcome::Exited(Some(0)), "{}", out.render());
        assert!(out.stdout.is_empty(), "{}", out.render());
        assert_eq!(outside.read("artifact"), "original-artifact");
        for dir in [
            "target/debug/deps",
            "packages/ui/node_modules/pkg",
            "build.cache",
        ] {
            assert_eq!(f.read(&format!("{dir}/artifact")), "original-artifact");
        }
        for denied in [
            "renamed",
            "packages/ui/renamed",
            "renamed-cache",
            "renamed-uppercase",
            "target/new",
        ] {
            assert!(!f.0.join(denied).exists(), "{denied}: {}", out.render());
        }
        assert_eq!(f.read("source.txt"), "allowed");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn unicode_case_aliases_do_not_expose_excluded_or_credential_files() {
        let f = Fixture::new();
        let outside = Fixture::new();
        outside.write("private", "synthetic-private-unicode-marker");
        let mut exclusions = Exclusions::default();
        exclusions.add("pkg").unwrap();
        for name in ["node_modules", "pkg"] {
            std::fs::create_dir(f.0.join(name)).unwrap();
            std::fs::hard_link(outside.0.join("private"), f.0.join(name).join("alias")).unwrap();
        }
        f.write("secrets", "synthetic-private-unicode-marker");
        f.write("known_hosts", "synthetic-private-unicode-marker");
        let out = run_with_exclusions(&f.workspace(), &spec(&["/bin/sh", "-c", "cat node_moduleſ/alias pKg/alias ſecrets Known_hosts; printf changed > node_moduleſ/alias; printf changed > pKg/alias; true"], 2000), 4096, &AtomicBool::new(false), &exclusions).unwrap();
        assert!(out.stdout.is_empty(), "{}", out.render());
        assert_eq!(outside.read("private"), "synthetic-private-unicode-marker");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn build_environment_cannot_expand_roots_or_expose_other_sessions_and_toolchain_secrets() {
        let f = Fixture::new();
        let outside = Fixture::new();
        outside.write("private", "synthetic-outside-marker");
        std::fs::create_dir_all(outside.0.join(".rustup/toolchains/fixture/bin")).unwrap();
        std::fs::create_dir(outside.0.join(".rustup/toolchains/fixture/lib")).unwrap();
        let toolchain = outside.0.join(".rustup/toolchains/fixture");
        for name in ["cargo", "rustc", "rustdoc"] {
            std::fs::write(toolchain.join("bin").join(name), "fixture").unwrap();
        }
        std::fs::write(toolchain.join("lib/library"), "public-library").unwrap();
        std::fs::write(toolchain.join("private"), "synthetic-outside-marker").unwrap();
        std::fs::write(toolchain.join("lib/secret.key"), "synthetic-outside-marker").unwrap();
        let other = BuildSession::default();
        let other_path = other.prepare(&f.0).unwrap().as_ref().unwrap().path.clone();
        std::fs::write(other_path.join("private"), "synthetic-other-session-marker").unwrap();
        let session = BuildSession::new(Some(&toolchain)).unwrap();
        let root = session
            .prepare(&f.0)
            .unwrap()
            .as_ref()
            .unwrap()
            .path
            .clone();
        let script = r#"
set -eu
printf cache > "$CARGO_TARGET_DIR/cache"
printf hidden > "$TMPDIR/.generated"
printf npm > "$npm_config_cache/data"
test "$npm_config_userconfig" != "$npm_config_globalconfig"
cat "$3/lib/library"
if cat "$1/private" "$2/private" "$3/private" "$3/lib/secret.key"; then exit 1; fi
if printf changed > "$1/private"; then exit 2; fi
if printf changed > "$3/lib/library"; then exit 3; fi
if ln "$1/private" "$CARGO_TARGET_DIR/link"; then exit 4; fi
ln -s "$1" "$TMPDIR/outside-link"
if cat "$TMPDIR/outside-link/private"; then exit 5; fi
if /usr/bin/env CARGO_TARGET_DIR="$1" /bin/sh -c 'printf changed > "$CARGO_TARGET_DIR/private"'; then exit 6; fi
if mv "$4" stolen-build; then exit 7; fi
if chmod 755 "$4"; then exit 8; fi
printf allowed > source.txt
"#;
        let out = run_in_session(
            &f.workspace(),
            &spec(
                &[
                    "/bin/sh",
                    "-c",
                    script,
                    "fixture",
                    outside.0.to_str().unwrap(),
                    other_path.to_str().unwrap(),
                    toolchain.to_str().unwrap(),
                    root.to_str().unwrap(),
                ],
                3000,
            ),
            8192,
            &AtomicBool::new(false),
            &Exclusions::default(),
            &session,
        )
        .unwrap();
        assert_eq!(out.outcome, Outcome::Exited(Some(0)), "{}", out.render());
        assert_eq!(out.stdout, "public-library", "{}", out.render());
        assert_eq!(outside.read("private"), "synthetic-outside-marker");
        assert_eq!(
            std::fs::read_to_string(toolchain.join("lib/library")).unwrap(),
            "public-library"
        );
        assert_eq!(f.read("source.txt"), "allowed");
        assert!(!f.0.join("stolen-build").exists());
        drop(session);
        assert!(!root.exists());
        assert_eq!(outside.read("private"), "synthetic-outside-marker");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn excluded_entries_do_not_consume_scan_budget_and_limit_names_the_path() {
        let f = Fixture::new();
        std::fs::create_dir(f.0.join("target")).unwrap();
        for i in 0..20_001 {
            f.write(&format!("target/{i}"), "");
        }
        let cancel = AtomicBool::new(false);
        let out = run(
            &f.workspace(),
            &spec(&["/usr/bin/true"], 2000),
            1024,
            &cancel,
        )
        .unwrap();
        assert_eq!(out.outcome, Outcome::Exited(Some(0)), "{}", out.render());
        std::fs::rename(f.0.join("target"), f.0.join("unexcluded")).unwrap();
        let error = verify_tree(&f.0, &cancel, &Exclusions::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("unexcluded/") && error.contains("20,000"),
            "{error}"
        );
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn exclusion_roots_cannot_be_symlinks_or_hide_a_writable_alias() {
        let f = Fixture::new();
        std::fs::create_dir(f.0.join("target")).unwrap();
        f.write("target/a", "original");
        std::fs::hard_link(f.0.join("target/a"), f.0.join("source-alias")).unwrap();
        let cancel = AtomicBool::new(false);
        let error = verify_tree(&f.0, &cancel, &Exclusions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("source-alias"), "{error}");
        std::fs::remove_file(f.0.join("source-alias")).unwrap();
        std::os::unix::fs::symlink("target", f.0.join("node_modules")).unwrap();
        let error = verify_tree(&f.0, &cancel, &Exclusions::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("node_modules") && error.contains("symbolic"),
            "{error}"
        );
    }
    #[test]
    fn exclusions_are_bounded_visible_names_not_paths_or_regexes() {
        let mut exclusions = Exclusions::default();
        for name in [
            "", ".", "..", ".git", "foo/bar", "foo\\bar", "x|y", "*", "a b", "a\n",
        ] {
            assert!(exclusions.add(name).is_err(), "{name}");
        }
        for i in 0..8 {
            exclusions.add(&format!("build-{i}")).unwrap();
        }
        assert!(exclusions.add("extra").is_err());
        assert!(exclusions.add("target").is_ok());
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn refuses_a_replaced_workspace_root() {
        use std::os::unix::fs::symlink;
        let f = Fixture::new();
        let other = Fixture::new();
        let w = f.workspace();
        let moved = f.0.with_extension("moved");
        std::fs::rename(&f.0, &moved).unwrap();
        symlink(&other.0, &f.0).unwrap();
        assert!(
            run(
                &w,
                &spec(&["/bin/pwd"], 1000),
                1024,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        std::fs::remove_file(&f.0).unwrap();
        std::fs::rename(&moved, &f.0).unwrap();
    }
