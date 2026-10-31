#![cfg(unix)]
mod common;
#[path = "common/api_server.rs"] mod server;
use vade_tools::json;
use std::process::Command;
#[test] fn shell_client_streams_and_sends_tool_and_cache_turns() {
    let t = common::Temp::new("shell API"); let s = server::Server::start(server::replies());
    let out = common::run(Command::new("sh").arg(vade_tools::root().join("examples/openai_client.sh"))
        .env("VADE_API_URL",&s.url).env_remove("VADE_API_KEY_FILE").env("VADE_TOOLS",env!("CARGO_BIN_EXE_vade-tools")),&t.0);
    assert!(out.status.success(),"{}",common::diagnostic(&out));
    let stdout = String::from_utf8(out.stdout).unwrap(); assert!(stdout.contains("hello é")); assert!(stdout.contains("[usage] prompt 3, completion 2"));
    assert!(stdout.contains("[answer] Paris is clear.")); assert!(stdout.contains("[turn 2] prompt 905 tokens, cached 899: 19"));
    server::check(&s.finish());
}
#[test] fn shell_stream_rejects_engine_errors_and_missing_completion() {
    for (body,expected) in [
        ("data: {\"error\":{\"message\":\"generation canceled\"}}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":0}}\n\ndata: [DONE]\n\n","stream error"),
        ("","Incomplete event stream"),
        ("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n","Incomplete event stream"),
        ("data: [DONE]\n\n","Incomplete event stream"),
        ("data: {\"choices\":[]}\n\ndata: [DONE]","Incomplete event stream"),
    ] {
        let t = common::Temp::new("shell SSE error"); let s = server::Server::start(vec![
            (r#"{"data":[{"id":"test-model"}]}"#.into(),false),(body.into(),false)]);
        let out = common::run(Command::new("sh").arg(vade_tools::root().join("examples/openai_client.sh")).arg("stream")
            .env("VADE_API_URL",&s.url).env_remove("VADE_API_KEY_FILE").env("VADE_TOOLS",env!("CARGO_BIN_EXE_vade-tools")),&t.0);
        s.finish(); assert!(!out.status.success(),"{}",common::diagnostic(&out));
        assert!(String::from_utf8_lossy(&out.stderr).contains(expected),"{}",common::diagnostic(&out));
    }
}
#[test] fn shell_stream_keeps_a_curl_failure_status() {
    let t = common::Temp::new("shell API failure");
    common::executable(&t.0.join("bin/curl"),"#!/bin/sh\ncase $* in */v1/models*) printf '%s' '{\"data\":[{\"id\":\"test\"}]}' ;; *) exit 42 ;; esac\n");
    let out = common::run(Command::new("sh").arg(vade_tools::root().join("examples/openai_client.sh")).arg("stream")
        .env("VADE_API_URL","http://localhost:1").env_remove("VADE_API_KEY_FILE").env("VADE_TOOLS",env!("CARGO_BIN_EXE_vade-tools"))
        .env("PATH",format!("{}:{}",t.0.join("bin").display(),std::env::var("PATH").unwrap())),&t.0);
    assert_eq!(out.status.code(),Some(42),"{}",common::diagnostic(&out));
}
// A mktemp stub on PATH puts the client's private folder where the test can see it.
fn private_temp(root: &std::path::Path) -> std::path::PathBuf {
    let tmp = root.join("tmp"); std::fs::create_dir(&tmp).unwrap();
    common::executable(&root.join("bin/mktemp"),&format!("#!/bin/sh\n: > {}\nd={}/run\nmkdir -m 700 \"$d\" && printf '%s\\n' \"$d\"\n",common::quote(root.join("mktemp-ran").to_str().unwrap()),common::quote(tmp.to_str().unwrap())));
    tmp
}
#[test] fn shell_client_keeps_the_bearer_key_out_of_curl_arguments() {
    let t = common::Temp::new("shell API key"); let s = server::Server::start(server::replies()); let key = "test-key-7f3a9c";
    common::write(&t.0.join("key"),format!("{key}\n")); let log = t.0.join("curl-args"); let tmp = private_temp(&t.0);
    common::executable(&t.0.join("bin/curl"),&format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> {}\nexec {} \"$@\"\n",common::quote(log.to_str().unwrap()),common::quote(&common::which("curl"))));
    let out = common::run(Command::new("sh").arg(vade_tools::root().join("examples/openai_client.sh"))
        .env("VADE_API_URL",&s.url).env("VADE_API_KEY_FILE",t.0.join("key")).env("VADE_TOOLS",env!("CARGO_BIN_EXE_vade-tools"))
        .env("PATH",format!("{}:{}",t.0.join("bin").display(),std::env::var("PATH").unwrap())),&t.0);
    assert!(out.status.success(),"{}",common::diagnostic(&out));
    let requests = s.finish(); server::check(&requests);
    for request in &requests { assert!(request.contains(&format!("Authorization: Bearer {key}\r\n")),"{request}"); }
    let args = std::fs::read_to_string(&log).unwrap(); assert_eq!(args.lines().count(),6,"{args}"); assert!(!args.contains(key),"{args}");
    assert!(args.contains(&format!("@{}/run/authorization",tmp.display())),"{args}"); assert_eq!(std::fs::read_dir(&tmp).unwrap().count(),0,"the header file outlived the run");
}
#[test] fn shell_client_refuses_a_bearer_key_with_a_line_break() {
    let t = common::Temp::new("shell API key break"); common::write(&t.0.join("key"),"one\nX-Injected: two\n");
    let log = t.0.join("curl-args"); let tmp = private_temp(&t.0);
    common::executable(&t.0.join("bin/curl"),&format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> {}\n",common::quote(log.to_str().unwrap())));
    let out = common::run(Command::new("sh").arg(vade_tools::root().join("examples/openai_client.sh"))
        .env("VADE_API_URL","http://localhost:1").env("VADE_API_KEY_FILE",t.0.join("key")).env("VADE_TOOLS",env!("CARGO_BIN_EXE_vade-tools"))
        .env("PATH",format!("{}:{}",t.0.join("bin").display(),std::env::var("PATH").unwrap())),&t.0);
    assert_eq!(out.status.code(),Some(1),"{}",common::diagnostic(&out)); assert!(String::from_utf8_lossy(&out.stderr).contains("line break"),"{}",common::diagnostic(&out));
    assert!(!log.exists(),"curl ran before the refusal: {}",std::fs::read_to_string(&log).unwrap_or_default());
    assert!(t.0.join("mktemp-ran").exists(),"the client did not use the mktemp stub"); assert_eq!(std::fs::read_dir(&tmp).unwrap().count(),0,"the refusal left a temporary folder");
}
