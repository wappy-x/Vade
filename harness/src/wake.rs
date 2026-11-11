//! Bounded channels that wake their receiving UI thread without a polling delay.
use std::sync::mpsc::{self, Receiver, SendError, SyncSender, TrySendError};
use std::thread::{self, Thread};

/// The sending half of a [`channel`]; each send also wakes the receiving thread.
pub struct Sender<T> {
    tx: SyncSender<T>,
    receiver: Thread,
}

/// A bounded channel whose receiver is the calling thread, which each send
/// unparks.
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = mpsc::sync_channel(capacity);
    (
        Sender {
            tx,
            receiver: thread::current(),
        },
        rx,
    )
}
impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        let result = self.tx.send(value);
        self.receiver.unpark();
        result
    }
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        let result = self.tx.try_send(value);
        self.receiver.unpark();
        result
    }
}
