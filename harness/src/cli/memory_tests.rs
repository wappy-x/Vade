use super::*;
use vade_harness::{json, memory, tools::StateLocation};

#[test]
fn chat_memory_startup_switches_new_and_resume() {
    const CHILD: &str = "VADE_MEMORY_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Child processes set the environment without unsafe global mutation.
        for value in ["0", "1"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "cli::memory_tests::chat_memory_startup_switches_new_and_resume", "--nocapture"])
                .env(CHILD, "1").env("VADE_MEMORY", value).output().unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
        }
        return;
    }
    let root = std::env::temp_dir().join(format!("vade-memory-chat-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
    let _cleanup = Cleanup(root.clone());
    let location = StateLocation::home(root.join("state")).unwrap();
    let workspace = Workspace::open(&root, location.clone()).unwrap();
    memory::change(&workspace, "response-style", Some("Use short answers.")).unwrap();
    let enabled = std::env::var("VADE_MEMORY").unwrap() != "0";
    for (mode, children) in [("direct", false), ("agent", false), ("agent", true)] {
        for flag in [false, true] {
            let mut args = vec![mode, "--endpoint", "http://localhost:1", "--model", "mock",
                "--workspace", root.to_str().unwrap(), "--no-instructions", "--plain"];
            if mode == "agent" { args.push("--no-check"); }
            if children { args.push("--children"); }
            if flag { args.push("--no-memory"); }
            let args::Parsed::Run(mut config) = args::Config::parse(args.into_iter().map(String::from)).unwrap() else { panic!() };
            assert_eq!(config.memory, enabled && !flag);
            config.state = Some(location.clone());
            let app = App::new(config).unwrap();
            assert_eq!(app.memory.contains("Use short answers."), enabled && !flag);
            if let Some(conversation) = &app.native {
                let request = conversation.request(Default::default()).to_json().unwrap();
                assert_eq!(request.contains(memory::LABEL), enabled && !flag);
                assert_eq!(request.contains("memory_save"), enabled && !flag);
                native_setup::validate_resumed(conversation, "mock", children, &Default::default(), Default::default()).unwrap();
                assert_eq!(native_setup::check_memory_resume(conversation, false).is_err(), enabled && !flag);
            }
        }
    }
    let limits = Default::default();
    let grants = Default::default();
    let old = native_setup::chat_conversation("mock", &workspace, &[], "", Some(&memory::Memory::load(&workspace).unwrap().content()), false, &grants, limits).unwrap();
    memory::change(&workspace, "response-style", Some("Explain tradeoffs.")).unwrap();
    let new = native_setup::chat_conversation("mock", &workspace, &[], "", Some(&memory::Memory::load(&workspace).unwrap().content()), false, &grants, limits).unwrap();
    assert!(old.request(Default::default()).to_json().unwrap().contains("Use short answers."));
    assert!(new.request(Default::default()).to_json().unwrap().contains("Explain tradeoffs."));
    let journal = Journal::create(&workspace, "memory-round-trip", &new).unwrap();
    let name = journal.name().to_owned();
    drop(journal);
    let (_, saved, _) = Journal::resume(&workspace, &name, limits).unwrap();
    assert_eq!(new.request(Default::default()).to_json().unwrap(), saved.request(Default::default()).to_json().unwrap());
    assert!(json::stringify(saved.options()).unwrap().contains("memory_forget"));
}
