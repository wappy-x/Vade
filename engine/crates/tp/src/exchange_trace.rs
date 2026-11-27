//! EXCHANGE TRACE (measurement only): host timers for the exchanges of a prompt group between the ranks.
//!
//! Off unless the process starts with `VADE_EXCHANGE_TRACE=1`. Off, every hook is one load of a flag that was read
//! once: nothing is timed, stored or printed, no stream is drained and no value changes. On, each host-staged exchange
//! (`Weights::gather_mapped_parts`) keeps five host times and one line is printed for it when its prompt group ends:
//!
//! - `grow_ns`: a larger mapped exchange buffer was made (the stream drained first);
//! - `pack_ns`: this rank's block was queued for its copy into the mapped buffer;
//! - `drain_ns`: the host waited for the stream (the math before the exchange, and that copy);
//! - `link_ns`: the link step: this rank's block out, the other rank's block in. It also holds the time the other rank
//!   still needed before it sent. Both ranks print the same exchanges in the same order, so the smaller `link_ns` of
//!   the two is the wire time and the difference is the wait for the slower rank;
//! - `after_ns`: from the end of the link step until the kernel that reads the exchange buffer was queued.
//!
//! The trace adds no synchronization: the timers stand around waits the path already makes.
use std::{cell::RefCell,collections::BTreeMap,sync::OnceLock,time::Instant};
/// True when this process traces its exchanges. Read once.
pub fn on()->bool {static ON:OnceLock<bool>=OnceLock::new();*ON.get_or_init(||std::env::var("VADE_EXCHANGE_TRACE").as_deref()==Ok("1"))}
/// A host time, only when tracing.
pub fn now()->Option<Instant> {on().then(Instant::now)}
struct Row {kind:String,bytes:usize,legs:[u128;5]}
#[derive(Default)]
struct Trace {kind:String,rows:Vec<Row>,host:BTreeMap<String,[u128;2]>,device:BTreeMap<String,[u64;2]>,link_end:Option<Instant>}
thread_local! {static TRACE:RefCell<Trace>=RefCell::new(Trace::default());}
/// Names the next exchange of this thread.
pub fn kind(name:impl FnOnce()->String) {if on() {TRACE.with(|t|t.borrow_mut().kind=name());}}
/// One host-staged exchange of `bytes` bytes a rank: the times at its start, after a buffer growth, after the copy was
/// queued, after the stream drained, and (now) after the link step.
pub fn exchange(bytes:usize,times:[Option<Instant>;4]) {
    let [Some(start),Some(grown),Some(packed),Some(drained)]=times else {return;};let end=Instant::now();
    TRACE.with(|t| {let mut t=t.borrow_mut();let kind=std::mem::take(&mut t.kind);
        t.rows.push(Row {kind,bytes,legs:[(grown-start).as_nanos(),(packed-grown).as_nanos(),(drained-packed).as_nanos(),(end-drained).as_nanos(),0]});t.link_end=Some(end);});
}
/// The kernel that reads the last exchange was queued.
pub fn reader() {if on() {TRACE.with(|t| {let mut t=t.borrow_mut();if let Some(end)=t.link_end.take()&& let Some(row)=t.rows.last_mut() {row.legs[4]=end.elapsed().as_nanos();}});}}
/// One exchange on the device mailbox (no host wait): counted by kind.
pub fn device(bytes:usize) {if on() {TRACE.with(|t| {let mut t=t.borrow_mut();let kind=std::mem::take(&mut t.kind);let row=t.device.entry(kind).or_default();row[0]+=1;row[1]+=bytes as u64;});}}
/// Host work of a prompt group that the stream does not see (`Compute::plan_cost`).
pub fn host(name:&str,ns:u128) {if on() {TRACE.with(|t| {let mut t=t.borrow_mut();let row=t.host.entry(name.into()).or_default();row[0]+=1;row[1]+=ns;});}}
/// Prints and clears this thread's trace: one line an exchange, in order. Called when a prompt group ends.
pub fn report(rank:u32) {
    if !on() {return;}
    let trace=TRACE.with(|t|std::mem::take(&mut *t.borrow_mut()));
    for (index,row) in trace.rows.iter().enumerate() {
        eprintln!("XTRACE rank={rank} index={index} bytes={} grow_ns={} pack_ns={} drain_ns={} link_ns={} after_ns={} kind={}",row.bytes,row.legs[0],row.legs[1],row.legs[2],row.legs[3],row.legs[4],row.kind);
    }
    for (kind,[calls,bytes]) in &trace.device {eprintln!("XDEVICE rank={rank} calls={calls} bytes={bytes} kind={kind}");}
    for (name,[calls,ns]) in &trace.host {eprintln!("XHOST rank={rank} calls={calls} ns={ns} name={name}");}
    eprintln!("XTRACE END rank={rank} exchanges={}",trace.rows.len());
}
#[cfg(test)]
mod tests {
    /// The switch is off in a test process (the variable is not set), so every hook returns at its flag.
    #[test]
    fn hooks_do_nothing_when_the_switch_is_off() {
        assert!(!super::on());assert!(super::now().is_none());
        super::kind(||unreachable!("the name of an exchange is not built when the trace is off"));
        super::exchange(4,[None;4]);super::reader();super::device(4);super::host("x",1);super::report(0);
        super::TRACE.with(|t| {let t=t.borrow();assert!(t.rows.is_empty() && t.host.is_empty() && t.device.is_empty() && t.kind.is_empty() && t.link_end.is_none());});
    }
}
