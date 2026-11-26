//! The startup steps of a rank service. Print only: one `STARTUP` line a step, with the
//! seconds the step took and the seconds since the process started. A failed start prints
//! one more line with its rank, the step it stopped in and the cause; the `FAIL` line that
//! follows it stays as it was. No step changes what the service computes.
use std::{sync::Mutex,time::Instant};

/// `label`: the rank number as text. The lines only print it.
struct Marks {label:String,first:Instant,last:Instant,name:&'static str}
static MARKS:Mutex<Option<Marks>>=Mutex::new(None);

fn seconds(from:Instant,to:Instant)->f64 {to.saturating_duration_since(from).as_secs_f64()}
fn line(marks:&Marks,now:Instant)->String {format!("STARTUP rank={} step={} seconds={:.3} at={:.3}",marks.label,marks.name,seconds(marks.last,now),seconds(marks.first,now))}

/// Start the steps of this process. `first` is when the process started; `name` is the step
/// that ran from then to this call.
pub fn begin(rank:usize,first:Instant,name:&'static str) {
    let mut marks=MARKS.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
    *marks=Some(Marks {label:rank.to_string(),first,last:first,name});
}
/// The step in work ends here and `name` starts. A process that never called `begin` prints nothing.
pub fn step(name:&'static str) {
    let mut marks=MARKS.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
    if let Some(marks)=marks.as_mut() {let now=Instant::now();eprintln!("{}",line(marks,now));marks.last=now;marks.name=name;}
}
/// Seconds the step in work has run. None after `done` or without `begin`.
pub fn waited()->Option<f64> {
    MARKS.lock().unwrap_or_else(|poisoned|poisoned.into_inner()).as_ref().map(|marks|seconds(marks.last,Instant::now()))
}
/// Startup is over: prints the last step and the whole time. A later failure is not a startup failure.
pub fn done() {
    let mut marks=MARKS.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
    if let Some(marks)=marks.take() {let now=Instant::now();eprintln!("{}",line(&marks,now));eprintln!("{}",line(&Marks {name:"ready",last:now,..marks},now));}
}
/// The line of a failed start: its rank, the step it stopped in, how long that step ran and the cause.
/// None after `done` or without `begin`: a failure of a served request is not a failed start.
fn failure(error:&str)->Option<String> {
    let marks=MARKS.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
    marks.as_ref().map(|marks|failure_line(marks,Instant::now(),error))
}
fn failure_line(marks:&Marks,now:Instant,error:&str)->String {format!("STARTUP rank={} failed in step {} after {:.1} s ({:.1} s since the process started): {error}",marks.label,marks.name,seconds(marks.last,now),seconds(marks.first,now))}
/// Print the line of a failed start. The caller prints its own `FAIL` line after it, unchanged.
pub fn failed(error:&str) {
    if let Some(line)=failure(error) {eprintln!("{line}");}
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The steps share one process-wide record, so one test walks the whole life of it. Another test of this
    /// binary may call `step` at any moment (the decoder names its `weights` step), so the step name is
    /// asserted on a record of this test's own, never on the shared one.
    #[test] fn a_failed_start_names_its_rank_and_step_and_a_served_failure_does_not() {
        assert!(failure("plain").is_none());assert!(waited().is_none());step("ignored");
        let first=Instant::now();begin(1,first,"header_admission");
        step("rank_connect");
        let message=failure("rank connection timed out").unwrap();
        assert!(message.starts_with("STARTUP rank=1 failed in step "),"{message}");
        assert!(message.ends_with(": rank connection timed out"),"{message}");
        assert!(waited().is_some());
        let marks=Marks {label:"0".into(),first,last:first,name:"tokenizer"};
        assert_eq!(line(&marks,first+std::time::Duration::from_millis(1500)),"STARTUP rank=0 step=tokenizer seconds=1.500 at=1.500");
        let stopped=Marks {label:"1".into(),first,last:first+std::time::Duration::from_millis(2000),name:"rank_connect"};
        assert_eq!(failure_line(&stopped,first+std::time::Duration::from_millis(32000),"rank connection timed out"),"STARTUP rank=1 failed in step rank_connect after 30.0 s (32.0 s since the process started): rank connection timed out");
        done();
        assert!(failure("later").is_none());failed("later");
    }
}
