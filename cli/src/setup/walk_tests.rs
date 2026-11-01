use super::*;
use std::collections::VecDeque;

fn facts(spark: bool) -> Facts {
    Facts {
        spark,
        explicit: false,
        path: PathBuf::from("/home/you/.config/vade/config"),
        checkout: Some(PathBuf::from("/home/you/vade")),
        home: PathBuf::from("/home/you/.vade"),
        models: None,
    }
}

fn pinned(_: &Path, fp8: bool) -> Result<Profile, String> {
    Ok(if fp8 {
        Profile { fp8, model: "qwen3.8-flash-next-fp8".into(), repo: "Qwen/Qwen3.8-Flash-Next-FP8".into(), revision: "b".repeat(40) }
    } else {
        Profile { fp8, model: "qwen3.8-flash-next-fp4-local".into(), repo: "nvidia/Qwen3.8-Flash-Next-FP4-Local".into(), revision: "a".repeat(40) }
    })
}

fn flags(yes: bool, hosts: &[&str], links: &[(&str, &str, &str)]) -> SetupOptions {
    SetupOptions {
        yes,
        hosts: hosts.iter().map(|host| (*host).into()).collect(),
        links: links.iter().map(|(from, to, address)| ((*from).into(), (*to).into(), (*address).into())).collect(),
        ..SetupOptions::default()
    }
}

/// Answer each question from `replies` in order, and record the questions.
fn scripted(options: &SetupOptions, facts: &Facts, replies: &[&str]) -> (Result<Target, String>, Vec<String>) {
    let mut replies: VecDeque<String> = replies.iter().map(|reply| (*reply).into()).collect();
    let mut asked = Vec::new();
    let result = answer(options, facts, &mut |question| {
        asked.push(question.to_owned());
        replies.pop_front().ok_or_else(|| format!("unexpected question: {question}"))
    }, pinned);
    (result, asked)
}

fn controller(target: Result<Target, String>) -> (String, Vec<String>) {
    match target {
        Ok(Target::Controller { text: Some(text), up, .. }) => (text, up),
        other => panic!("not a new controller config: {other:?}"),
    }
}

#[test]
fn yes_on_a_spark_installs_this_spark_with_the_found_model() {
    let (target, asked) = scripted(&flags(true, &[], &[]), &facts(true), &[]);
    assert_eq!(target, Ok(Target::Local { model: None, installed: false }));
    assert!(asked.is_empty());
    let mut options = flags(true, &[], &[]);
    options.model = Some("/data/models/fp4".into());
    assert_eq!(scripted(&options, &facts(true), &[]).0, Ok(Target::Local { model: Some("/data/models/fp4".into()), installed: false }));
    // install expands ~/ in VADE_MODELS, as the default folder shows it.
    options.model = Some("~/models/fp4".into());
    assert_eq!(scripted(&options, &facts(true), &[]).0, Ok(Target::Local { model: Some("~/models/fp4".into()), installed: false }));
    for model in ["models/fp4", "~user/models", "/models/fp4\nx"] {
        options.model = Some(model.into());
        assert!(scripted(&options, &facts(true), &[]).0.unwrap_err().contains("absolute path or a path under ~/"), "{model}");
    }
}

#[test]
fn yes_off_a_spark_needs_the_sparks_and_every_cable_end() {
    let error = scripted(&flags(true, &[], &[]), &facts(false), &[]).0.unwrap_err();
    assert!(error.contains("Add --hosts"), "{error}");
    let error = scripted(&flags(true, &["spark-a", "spark-b"], &[("spark-a", "spark-b", "192.0.2.1")]), &facts(false), &[]).0.unwrap_err();
    assert!(error.contains("Add --link spark-b:spark-a=ADDRESS"), "{error}");
    let error = scripted(&flags(true, &["spark-a"], &[("spark-a", "spark-b", "192.0.2.1")]), &facts(false), &[]).0.unwrap_err();
    assert!(error.contains("does not name two neighbors"), "{error}");
}

#[test]
fn one_spark_from_a_controller_gets_a_config_the_parser_accepts() {
    let (target, asked) = scripted(&flags(true, &["you@192.0.2.10"], &[]), &facts(false), &[]);
    assert!(asked.is_empty());
    let (text, up) = controller(target);
    assert!(up.is_empty(), "{up:?}");
    assert!(text.starts_with("# Written by vade setup. Start: vade up\n"), "{text}");
    for line in ["engine_source = /home/you/vade/engine", "stage_dir = /home/you/.vade/controller/stage", "state_dir = /home/you/.vade/controller/state",
        "model_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "[machine 192-0-2-10]", "ssh = you@192.0.2.10",
        "model_path = ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"] {
        assert!(text.lines().any(|row| row == line), "{line}:\n{text}");
    }
    let config = crate::config::Config::parse(&text, Path::new("/home/you/.config/vade/config")).unwrap();
    assert_eq!(config.machines.len(), 1);
    assert_eq!(start_options(&text, &config), Vec::<String>::new());
}

#[test]
fn questions_show_defaults_and_build_an_fp8_pair() {
    let (target, asked) = scripted(&SetupOptions::default(), &facts(false),
        &["spark-a, spark-b", "FP8", "", "192.0.2.1", "192.0.2.2"]);
    let (text, up) = controller(target);
    assert_eq!(asked, [
        "Which Sparks? Type what you type after ssh, in cable order, separated by commas:",
        "Which checkpoint? FP4-Local or FP8 [FP4-Local]:",
        "Where is the model on each Spark? [~/models/Qwen/Qwen3.8-Flash-Next-FP8]:",
        "Address of spark-a on its cable to spark-b:",
        "Address of spark-b on its cable to spark-a:",
    ]);
    assert_eq!(up, ["--fp8", "--hosts", "spark-a,spark-b"]);
    assert!(text.contains("[machine spark-a]\nssh = spark-a\nmodel_path = ~/models/Qwen/Qwen3.8-Flash-Next-FP8\nfabric.spark-b = 192.0.2.1\n"), "{text}");
    assert!(text.contains("[machine spark-b]\nssh = spark-b\nmodel_path = ~/models/Qwen/Qwen3.8-Flash-Next-FP8\nfabric.spark-a = 192.0.2.2\n"), "{text}");
    let config = crate::config::Config::parse(&text, Path::new("/home/you/.config/vade/config")).unwrap();
    assert_eq!(config.model, "qwen3.8-flash-next-fp8");
    assert_eq!(start_options(&text, &config), up);
}

#[test]
fn on_a_spark_enter_takes_this_spark_and_the_found_model() {
    let (target, asked) = scripted(&SetupOptions::default(), &facts(true), &["", ""]);
    assert_eq!(target, Ok(Target::Local { model: None, installed: false }));
    assert_eq!(asked, [
        "Which Sparks? Type what you type after ssh, in cable order, separated by commas [this Spark]:",
        "Where is the FP4-Local model on this Spark? [found under ~/models]:",
    ]);
}

#[test]
fn wrong_answers_write_nothing_and_name_the_fix() {
    for (options, spark, replies, expected) in [
        (SetupOptions::default(), false, vec![""], "Give one or more distinct SSH destinations"),
        (SetupOptions::default(), false, vec!["a,a"], "Give one or more distinct SSH destinations"),
        (SetupOptions::default(), false, vec!["a,b", "fp16"], "Answer FP4-Local or FP8"),
        // A bad model folder stops before the cable questions.
        (SetupOptions::default(), false, vec!["a,b", "", "models/fp4"], "Give the model folder on each Spark"),
        (SetupOptions { model: Some("~/m\nverify_url = http://192.0.2.9:1".into()), ..flags(true, &["a"], &[]) }, false, vec![], "Give the model folder on each Spark"),
        (SetupOptions::default(), false, vec!["a,b", "", "", "192.0.2.1:80"], "without a port"),
        // vade up's own cable check runs before anything is written.
        (SetupOptions::default(), false, vec!["a,b", "", "", "192.0.2.1", "192.0.2.1"], "Nothing was written: vade up would refuse this setup. Machines a and b need different addresses"),
        (SetupOptions { fp8: Some(true), ..flags(true, &["a", "b", "c"], &[]) }, false, vec![], "This setup has 3. Use 2 or 4 Sparks in --hosts"),
        (SetupOptions { fp8: Some(true), ..flags(true, &[], &[]) }, true, vec![], "This Spark alone cannot serve FP8"),
    ] {
        let error = scripted(&options, &facts(spark), &replies).0.unwrap_err();
        assert!(error.contains(expected), "{expected}: {error}");
    }
    let mut explicit = facts(true);
    explicit.explicit = true;
    assert!(scripted(&flags(true, &[], &[]), &explicit, &[]).0.unwrap_err().contains("Add --hosts"));
}

#[test]
fn three_fp4_sparks_link_each_neighbor_pair_in_cable_order() {
    let links = [("a", "b", "192.0.2.1"), ("b", "a", "192.0.2.2"), ("b", "c", "192.0.2.3"), ("c", "b", "192.0.2.4")];
    let (text, up) = controller(scripted(&flags(true, &["a", "b", "c"], &links), &facts(false), &[]).0);
    assert_eq!(up, ["--hosts", "a,b,c"]);
    assert!(text.contains("[machine b]\nssh = b\nmodel_path = ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local\nfabric.a = 192.0.2.2\nfabric.c = 192.0.2.3\n"), "{text}");
    assert!(crate::config::Config::parse(&text, Path::new("/home/you/.config/vade/config")).is_ok());
}

#[test]
fn machine_names_come_from_the_host_and_stay_unique() {
    let hosts = ["you@192.0.2.10", "box-a.example", "tester@box-a.example", "_x"].map(String::from);
    assert_eq!(machine_names(&hosts), ["192-0-2-10", "box-a-example", "box-a-example-2", "spark_x"]);
    assert!(machine_names(&hosts).iter().all(|name| crate::cli::identifier(name)));
}

#[test]
fn a_config_already_there_starts_with_its_own_checkpoint() {
    let (text, _) = controller(scripted(&SetupOptions::default(), &facts(false),
        &["spark-a, spark-b", "FP8", "", "192.0.2.1", "192.0.2.2"]).0);
    let config = crate::config::Config::parse(&text, Path::new("/home/you/.config/vade/config")).unwrap();
    let body = text.split_once('\n').unwrap().1;
    assert_eq!(start_options(&format!("# Written by vade setup. Start: vade up --hosts spark-a,spark-b\n{body}"), &config),
        ["--fp8", "--hosts", "spark-a,spark-b"]);
    // A setup line it cannot use leaves the Sparks to vade up's own default.
    for first in ["# Written by vade setup. Start: vade up --fp8 --hosts spark-a,spark-c", "# Written by vade setup. Start: vade up --context 9 --hosts spark-a,spark-b",
        "# Written by vade setup. Start: vade up --hosts spark-a;spark-b"] {
        assert_eq!(start_options(&format!("{first}\n{body}"), &config), ["--fp8"], "{first}");
    }
    // Without setup's line, as in a config written by hand: the FP8 model gives --fp8, and
    // every machine it lists starts, in its order, except a vade verify engine.
    let by_hand = format!("# Start: vade up\n{body}\n[machine spark-v]\nssh = spark-v\nverify_url = http://192.0.2.9:18080\n");
    let config = crate::config::Config::parse(&by_hand, Path::new("/home/you/.config/vade/config")).unwrap();
    assert_eq!(start_options(&by_hand, &config), ["--fp8", "--hosts", "spark-a,spark-b"]);
    // One serving machine after a verify engine is still named: a plain vade up
    // would take the first machine.
    let verify_first = "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n[machine check]\nssh = check\nverify_url = http://192.0.2.9:18080\n\n[machine serve]\nssh = serve\n";
    let config = crate::config::Config::parse(verify_first, Path::new("/home/you/.config/vade/config")).unwrap();
    assert_eq!(start_options(verify_first, &config), ["--hosts", "serve"]);
    assert_eq!(check_start(&config, &start_options(verify_first, &config)), Ok(()));
}

#[test]
fn a_running_engine_gets_a_bare_up_and_a_stopped_one_is_checked_first() {
    let (text, _) = controller(scripted(&SetupOptions::default(), &facts(false),
        &["spark-a, spark-b", "FP8", "", "192.0.2.1", "192.0.2.2"]).0);
    let config = crate::config::Config::parse(&text, Path::new("/home/you/.config/vade/config")).unwrap();
    // Bare: up reuses the running engine's saved limits, such as --context 8192.
    assert_eq!(reuse_start(true, &text, &config), Ok(Vec::new()));
    assert_eq!(reuse_start(false, &text, &config), Ok(vec!["--fp8".to_owned(), "--hosts".to_owned(), "spark-a,spark-b".to_owned()]));
    // A stopped config that up would refuse stops setup before install.
    let same = text.replace("fabric.spark-a = 192.0.2.2", "fabric.spark-a = 192.0.2.1");
    let config = crate::config::Config::parse(&same, Path::new("/home/you/.config/vade/config")).unwrap();
    assert!(reuse_start(false, &same, &config).unwrap_err().contains("need different addresses"));
    assert_eq!(reuse_start(true, &same, &config), Ok(Vec::new()));
}

#[test]
fn steps_run_install_then_up_then_doctor() {
    let local = steps(&Target::Local { model: Some("/m".into()), installed: false });
    assert_eq!(local.iter().map(|(args, _)| args.join(" ")).collect::<Vec<_>>(), ["install", "up", "doctor"]);
    assert!(local.iter().all(|(_, env)| env == &[("VADE_MODELS".to_owned(), "/m".to_owned())]));
    let pair = steps(&Target::Controller { path: PathBuf::new(), text: None, up: vec!["--hosts".into(), "a,b".into()], installed: false });
    assert_eq!(pair[1].0, ["up", "--hosts", "a,b"]);
    assert!(pair.iter().all(|(_, env)| env.is_empty()));
    // A second install would make a new generation, and a running engine would not attach.
    let installed = steps(&Target::Controller { path: PathBuf::new(), text: None, up: vec!["--fp8".into()], installed: true });
    assert_eq!(installed.iter().map(|(args, _)| args.join(" ")).collect::<Vec<_>>(), ["up --fp8", "doctor"]);
    // After scripts/install.sh, a local setup with current files does not install again.
    let local = steps(&Target::Local { model: None, installed: true });
    assert_eq!(local.iter().map(|(args, _)| args.join(" ")).collect::<Vec<_>>(), ["up", "doctor"]);
}

fn shell(script: &str) -> Command {
    let mut command = Command::new("/bin/sh");
    command.args(["-c", script]).stdin(Stdio::null());
    command
}

#[test]
fn a_stop_reaches_the_running_step() {
    let marker = std::env::temp_dir().join(format!("vade-setup-step-{}", std::process::id()));
    let ready = marker.with_extension("ready");
    let _ = std::fs::remove_file(&marker);
    let _ = std::fs::remove_file(&ready);
    let started = std::time::Instant::now();
    // The stop comes only after the step has set its trap and said so.
    let script = format!("trap 'echo term > \"{}\"; kill $sleeper; exit 7' TERM; sleep 30 & sleeper=$!; : > \"{}\"; wait", marker.display(), ready.display());
    let (passed, stopped) = run_step(shell(&script), &|| ready.exists()).unwrap();
    assert!(!passed && stopped);
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "the step ran on after the stop");
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "term\n", "the step did not get SIGTERM");
    let _ = std::fs::remove_file(&marker);
    let _ = std::fs::remove_file(&ready);
    assert_eq!(run_step(shell("exit 0"), &|| false).unwrap(), (true, false));
    assert_eq!(run_step(shell("exit 3"), &|| false).unwrap(), (false, false));
    // Under --yes a step that reads gets end of input, whatever setup's own input is.
    let reads = step_command(Path::new("/bin/sh"), &["-c".into(), "read answer || exit 0; exit 4".into()], &[], true);
    assert_eq!(run_step(reads, &|| false).unwrap(), (true, false));
}

#[test]
fn a_closed_input_is_an_error_not_a_default() {
    assert!(read_answer(&mut &b""[..]).unwrap_err().contains("end of input"));
    assert_eq!(read_answer(&mut &b"  spark-a \nnext\n"[..]).unwrap(), "spark-a");
    // A line with no end is cut at the limit, not read into memory whole.
    let mut endless = std::io::Cursor::new(vec![b'a'; 1 << 20]);
    assert!(read_answer(&mut endless).unwrap_err().contains("too long"));
    assert_eq!(endless.position(), 4097);
}

#[test]
fn the_local_model_shows_where_install_will_look() {
    assert_eq!(local_model(Some("/m"), true, Some("/e")), "/m (as VADE_MODELS)");
    assert_eq!(local_model(None, true, Some("/e")), "the model saved in the local setup");
    assert_eq!(local_model(None, false, Some("/e")), "/e (VADE_MODELS)");
    assert_eq!(local_model(None, false, None), "found under ~/models by vade install");
    let facts = Facts { models: Some("/data/models".into()), ..facts(true) };
    let (_, asked) = scripted(&SetupOptions::default(), &facts, &["", ""]);
    assert_eq!(asked[1], "Where is the FP4-Local model on this Spark? [/data/models from VADE_MODELS]:");
}

#[test]
fn a_local_setup_moves_only_with_the_flag_or_a_yes_at_the_terminal() {
    let setup = LocalSetup { file: "/home/you/.vade/config".into(), aside: "/home/you/.vade/config.local".into() };
    let no_question = |_: &str| -> Result<String, String> { panic!("setup must not ask") };
    let replace = SetupOptions { replace_local: true, ..flags(true, &["a"], &[]) };
    setup.confirm(&replace, false, &mut no_question.clone()).unwrap();
    let refused = "Run vade setup again with --replace-local to move \"/home/you/.vade/config\" aside and write a controller config";
    // --yes asks nothing, and no terminal cannot ask: both refuse and name the file.
    for (options, terminal) in [(flags(true, &["a"], &[]), true), (flags(false, &["a"], &[]), false)] {
        assert!(setup.confirm(&options, terminal, &mut no_question.clone()).unwrap_err().contains(refused));
    }
    let asked = std::cell::RefCell::new(Vec::new());
    for (reply, yes) in [("y", true), ("YES", true), ("", false), ("n", false), ("later", false)] {
        let result = setup.confirm(&flags(false, &["a"], &[]), true, &mut |question| { asked.borrow_mut().push(question.to_owned()); Ok(reply.into()) });
        assert_eq!(result.is_ok(), yes, "{reply}");
    }
    assert_eq!(asked.borrow()[0], "This Spark has local setup in \"/home/you/.vade/config\". Move it to \"/home/you/.vade/config.local\" and write a controller config? [y/N]:");
}

#[test]
fn moving_a_local_setup_keeps_every_file_and_can_go_back() {
    let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-local-setup-{}", text::random_hex(8).unwrap()));
    text::private_dir(&root).unwrap();
    std::fs::write(root.join("config"), "saved\n").unwrap();
    std::fs::write(root.join("config.local"), "older\n").unwrap();
    let setup = LocalSetup::find(&root).unwrap();
    assert_eq!(setup.aside, root.join("config.local-2"), "an older file keeps its name");
    setup.move_aside(&root).unwrap();
    assert_eq!(std::fs::read_to_string(&setup.aside).unwrap(), "saved\n");
    assert_eq!(std::fs::read_to_string(root.join("config.local")).unwrap(), "older\n");
    assert!(!root.join("config").exists());
    setup.restore();
    assert_eq!(std::fs::read_to_string(root.join("config")).unwrap(), "saved\n");
    // A running engine still needs the file.
    text::private_dir(&root.join("state")).unwrap();
    std::fs::write(root.join("state/run.conf"), "version=1\n").unwrap();
    let error = LocalSetup::find(&root).unwrap_err();
    assert!(error.contains("The local engine is running") && error.contains("Run vade down"), "{error}");
    assert!(setup.move_aside(&root).is_err() && root.join("config").exists());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn more_sparks_with_no_checkout_name_the_clone_and_write_nothing() {
    let facts = Facts { checkout: None, ..facts(true) };
    let error = scripted(&flags(true, &["a"], &[]), &facts, &[]).0.unwrap_err();
    assert!(error.starts_with("More Sparks need a Vade checkout on this computer") && error.contains("Run git clone "), "{error}");
}
