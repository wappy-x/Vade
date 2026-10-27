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
