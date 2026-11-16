//! RECEIVE FIRST: when a host-staged exchange of a prompt group may post its receive before the wait for this rank's GPU.
//!
//! What it is for. A rank used to post its receive only at the link step, after its own GPU had drained. The rank that was
//! first then sent to a receiver that was not ready: the link refused the message and the sender sent it again. Measured
//! on a pair (run `read3-c1-xcount-1301-2`): about one refusal an exchange, and about 0.3 ms on most exchanges of every
//! size (a 0.5 MiB message: 46 µs when it goes well, 366 µs most times).
//!
//! The rule. The wire writes the other rank's block of the mapped exchange buffer as soon as a receive is posted. So the
//! receive may be posted early only when no kernel that reads that block is queued or running:
//! - the buffer was made in this call, after a drain: no reader exists;
//! - or the reader of the last exchange marked the stream after its last read, and the stream has passed that mark.
//!
//! A reader that sets no mark keeps the old order for the exchange after it: nothing is posted before the drain.

/// What a host-staged exchange does with its receive before it waits for this rank's GPU.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Rule {
    /// Post now: the buffer is new and nothing reads it.
    Post,
    /// Wait until the stream has passed the last reader's mark, then post. Later stream work keeps running.
    AfterMark,
    /// Post nothing before the drain: a reader of the last exchange may still be queued. The gather posts the receive.
    AtGather,
}
/// `fresh`: the exchange buffer was allocated in this call, after a drain. `marked`: the reader of the last exchange
/// marked the stream after its last read.
pub fn rule(fresh:bool,marked:bool)->Rule {if fresh {Rule::Post} else if marked {Rule::AfterMark} else {Rule::AtGather}}

#[cfg(test)]
mod tests {
    use super::{Rule,rule};
    #[test]
    fn a_receive_is_posted_early_only_when_no_reader_of_the_buffer_can_be_left() {
        // A reader that set no mark may still be queued: the wire must not write its block before the drain.
        assert_eq!(rule(false,false),Rule::AtGather);
        // A marked reader: the host waits for the mark first.
        assert_eq!(rule(false,true),Rule::AfterMark);
        // A new buffer has no reader, marked or not (an old mark belongs to the buffer that was freed).
        assert_eq!(rule(true,false),Rule::Post);
        assert_eq!(rule(true,true),Rule::Post);
    }
}
