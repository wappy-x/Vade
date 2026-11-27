#![cfg(unix)]
mod common;
use common::*;
use std::{fs,collections::BTreeMap,path::PathBuf,process::{Command,Output}};
const STUB: &str = r#"#!/bin/sh
name=$(basename "$0")
printf '%s %s\n' "$name" "$*" >> "$CALLS"
case "$name $*" in
  "vade up --sparks 2 --dry-run") cp "$VADE_CONFIG" "$SEEN"; printf '%s\n' "$VADE_HOME" > "$SEEN.home"
    printf 'Dry run.\nLaunch plan: stub\n  Endpoint          http://localhost:18181/v1\nStart planned.\n' ;;
esac
case "$name $*" in $STUB_FAIL) exit 42 ;; esac
exit 0
"#;
struct Example { temp: Temp, checkout: PathBuf, tmp: PathBuf, env: BTreeMap<String,String> }
impl Example {
    fn new() -> Self {
        let temp = Temp::new("two sparks"); let checkout = temp.0.join("R&D|x\\y $HOME vade"); let tmp = temp.0.join("tmp &|\\"); fs::create_dir(&tmp).unwrap();
        for name in ["two-sparks.sh","two-sparks.conf"] { write(&checkout.join("examples").join(name),fs::read(vade_tools::root().join("examples").join(name)).unwrap()); }
        for name in ["vade","curl"] { executable(&temp.0.join("bin").join(name),STUB); }
        executable(&temp.0.join("bin/mktemp"),&format!("#!/bin/sh\nexec {} -d \"$TMPDIR/tmp.XXXXXX\"\n",quote(&which("mktemp"))));
        let env = [("PATH",format!("{}:{}",temp.0.join("bin").display(),std::env::var("PATH").unwrap())),("TMPDIR",tmp.display().to_string()),
            ("CALLS",temp.0.join("calls").display().to_string()),("SEEN",temp.0.join("seen.conf").display().to_string()),
            ("VADE_API_URL","http://stale:1/v1".into())].into_iter().map(|(k,v)|(k.into(),v)).collect();
        Self {temp,checkout,tmp,env}
    }
    fn run(&self, live: bool, fail: &str) -> (Output,Vec<String>) {
        let mut c = Command::new("sh"); c.arg(self.checkout.join("examples/two-sparks.sh"));
        for (key,_) in std::env::vars_os() { if key.to_string_lossy().starts_with("VADE_") { c.env_remove(key); } }
        c.envs(&self.env).env("STUB_FAIL",fail);
        if live { c.arg("--live").env("VADE_CONFIG","/my/two.conf"); }
        let _ = fs::remove_file(self.temp.0.join("calls")); let out = run(&mut c,&self.temp.0);
        let calls = fs::read_to_string(self.temp.0.join("calls")).unwrap_or_default().lines().map(str::to_owned).collect(); (out,calls)
    }
}
#[test] fn dry_run_copies_paths_literally() {
    let e = Example::new(); let (out,calls) = e.run(false,""); assert!(out.status.success(),"{}",diagnostic(&out)); assert_eq!(calls,["vade up --sparks 2 --dry-run"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("  Endpoint          http://localhost:18181/v1"));
    let config = fs::read_to_string(e.temp.0.join("seen.conf")).unwrap(); let lines: Vec<_> = config.lines().collect();
    for (key,folder) in [("engine_source","engine"),("harness_source","harness"),("cli_source","cli"),("stage_dir",".local/stage")] {
        assert!(lines.contains(&format!("{key} = {}",e.checkout.join(folder).display()).as_str()));
    }
    let states: Vec<_> = lines.iter().filter_map(|s|s.strip_prefix("state_dir = ")).filter(|s|s.ends_with("/state")).collect(); assert_eq!(states.len(),1);
    let state = PathBuf::from(states[0]); let folder = state.parent().unwrap(); assert_eq!(folder.parent().unwrap(),e.tmp);
    assert_eq!(fs::read_to_string(e.temp.0.join("seen.conf.home")).unwrap(),format!("{}\n",folder.join("home").display()));
    assert!(!lines.iter().any(|s|!s.starts_with('#') && s.contains("/path/to/vade")));
}
#[test] fn failed_preview_keeps_its_status() {
    let e = Example::new(); for live in [false,true] { let (out,calls) = e.run(live,"vade up --sparks 2 --dry-run");
        assert_eq!(out.status.code(),Some(42),"{}",diagnostic(&out)); assert_eq!(calls,["vade up --sparks 2 --dry-run"]); }
}
#[test] fn live_flow_checks_the_planned_endpoint_and_stops() {
    let e = Example::new(); let (out,calls) = e.run(true,""); assert!(out.status.success(),"{}",diagnostic(&out));
    let clean: Vec<_> = calls.iter().map(|s|s.split(" --receipt ").next().unwrap()).collect();
    assert_eq!(clean,["vade up --sparks 2 --dry-run","vade install","vade up --sparks 2","vade status",
        "curl --fail-with-body -sS http://localhost:18181/v1/models","vade verify","vade down"]);
    let receipt = PathBuf::from(calls[5].split_once(" --receipt ").unwrap().1); assert_eq!(receipt.file_name().unwrap(),"vade-verify.json"); assert_eq!(receipt.parent().unwrap().parent().unwrap(),e.tmp);
}
#[test] fn live_failures_stay_nonzero_and_stop_after_startup() {
    let e = Example::new(); for (fail,stops) in [("vade install",false),("vade up --sparks 2",true),("vade status",true),("curl *",true),("vade verify *",true)] {
        let (out,calls) = e.run(true,fail); assert_eq!(out.status.code(),Some(42),"{}",diagnostic(&out)); assert_eq!(calls.last().unwrap() == "vade down",stops,"{calls:?}");
    }
}
